#!/usr/bin/env python3
"""
Milestone 2 Environment Smoke Rollout Tool

Run deterministic smoke rollouts to validate the SelectiveMaintenanceEnv.

Usage:
    python scripts/run_m2_environment_smoke.py [--split SPLIT] [--seeds N]

Examples:
    # Main config: N=5, K=2, horizon=100, predictor_train, 5 seeds
    python scripts/run_m2_environment_smoke.py --split predictor_train --seeds 5

    # Validation config
    python scripts/run_m2_environment_smoke.py --split rl_validation --seeds 5

    # K=1 sensitivity
    python scripts/run_m2_environment_smoke.py --split predictor_train --seeds 3 --k-capacity 1

    # All cost regimes
    python scripts/run_m2_environment_smoke.py --all-regimes --split predictor_train

    # Full validation matrix
    python scripts/run_m2_environment_smoke.py --validation-matrix --seeds 5

    # Policy modes
    python scripts/run_m2_environment_smoke.py --split predictor_train --policy corrective-only
    python scripts/run_m2_environment_smoke.py --split predictor_train --policy boundary
    python scripts/run_m2_environment_smoke.py --split predictor_train --policy simultaneous-failure
    python scripts/run_m2_environment_smoke.py --split predictor_train --policy mixed-event

This tool:
- Uses SelectiveMaintenanceEnv with production PredictionStore
- Selects random feasible actions from a seeded RNG
- Prints concise deterministic summary
- Writes no model, checkpoint, replay buffer, or result artifact
- Tracks actual NaN/Inf occurrences in observations
- Rejects rl_test split with clear error
"""

import argparse
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs import (
    SelectiveMaintenanceEnv,
    EnvironmentConfig,
    get_default_config,
    load_scenario_bank,
)
from src.envs.scenario_bank import Scenario, ScenarioBank
from src.envs.costs import list_cost_regimes, get_cost_regime
from src.predictors.prediction_store import load_default_prediction_store


# Allowed splits for development smoke (rl_test is NOT allowed)
ALLOWED_SMOKE_SPLITS = frozenset({"predictor_train", "rl_validation"})


@dataclass
class SmokeResult:
    """Result from a single smoke rollout with complete metadata."""
    split: str
    maintenance_capacity: int
    cost_regime_id: str
    seed: int
    policy: str
    episodes: int = 0
    steps: int = 0
    preventive_replacements: int = 0
    failures: int = 0
    preventive_cost: float = 0.0
    failure_cost: float = 0.0
    wasted_life_cost: float = 0.0
    total_cost: float = 0.0
    nan_observation_count: int = 0
    inf_observation_count: int = 0
    missing_lookup_count: int = 0
    split_violation_count: int = 0
    failure_boundary_cycles: List[int] = field(default_factory=list)
    offsets_tested: List[int] = field(default_factory=list)
    boundary_cases: List[Dict[str, Any]] = field(default_factory=list)
    simultaneous_cases: List[Dict[str, Any]] = field(default_factory=list)
    mixed_cases: List[Dict[str, Any]] = field(default_factory=list)
    completed: bool = False
    errors: List[str] = field(default_factory=list)


def _step_and_track(
    env: SelectiveMaintenanceEnv,
    action: int,
    stats: SmokeResult,
    verbose: bool,
    step_index: int,
) -> tuple:
    """
    Execute one environment step and track statistics.

    Returns:
        (obs, reward, terminated, truncated, step_info) - caller MUST assign truncated
    """
    try:
        obs, reward, terminated, truncated, step_info = env.step(action)
    except Exception as e:
        stats.errors.append(f"Step {step_index} failed: {e}")
        return None, None, True, True, {}

    # Track statistics (exactly once per step)
    stats.steps += 1
    stats.preventive_replacements += step_info.get("num_preventive", 0)
    stats.failures += step_info.get("num_failures", 0)
    stats.preventive_cost += step_info.get("preventive_cost", 0.0)
    stats.failure_cost += step_info.get("failure_cost", 0.0)
    stats.wasted_life_cost += step_info.get("wasted_life_cost", 0.0)
    stats.total_cost += step_info.get("total_cost", 0.0)

    # Track NaN/Inf in observations
    if np.any(np.isnan(obs)):
        stats.nan_observation_count += 1
    if np.any(np.isinf(obs)):
        stats.inf_observation_count += 1

    if verbose and step_index < 5:
        print(f"  Step {step_index}: action={action}, reward={reward:.4f}, "
              f"pm={step_info.get('num_preventive', 0)}, fail={step_info.get('num_failures', 0)}")

    return obs, reward, terminated, truncated, step_info


def _capture_slot_snapshot(env: Any, slot_idx: int, step_info: dict) -> Optional[Dict[str, Any]]:
    """
    Capture actual post-step SlotState for a single slot from diagnostic info.

    Reads the actual state from env._fleet_state.slots[slot_idx] and returns:
    - slot_index
    - unit_id
    - cycle
    - age_since_replacement_cycles

    Does not return expected or inferred values - reads actual attributes.

    Args:
        env: SelectiveMaintenanceEnv instance
        slot_idx: Index of the slot to snapshot
        step_info: Step info dict from env.step()

    Returns:
        Dict with slot state or None if slot not found
    """
    diag_key = f"slot_{slot_idx}_diagnostic"
    if diag_key not in step_info:
        return None

    diag = step_info[diag_key]
    return {
        "slot_index": slot_idx,
        "unit_id": diag.get("unit_id"),
        "cycle": diag.get("cycle"),
        "age_since_replacement_cycles": diag.get("age_since_replacement_cycles"),
        "true_rul": diag.get("true_rul"),
        "trajectory_length": diag.get("trajectory_length"),
    }


def _load_prediction_store() -> Any:
    """Load the default V2 prediction store."""
    cache_dir = Path("data/processed/fd001/v2/06_PREDICTIONS/")
    return load_default_prediction_store(cache_dir)


def _load_scenario_bank(split: str, k_capacity: int) -> ScenarioBank:
    """Load scenario bank for split and K, deriving K=1 from K=2 if needed."""
    if k_capacity == 1:
        scenario_bank_path = Path(f"data/scenario_banks/{split}_k1_smoke.json")
    else:
        scenario_bank_path = Path(f"data/scenario_banks/{split}_smoke.json")

    if scenario_bank_path.exists():
        return load_scenario_bank(scenario_bank_path)

    # K=1 JSON missing: derive from K=2 by modifying maintenance_capacity in-memory
    if k_capacity == 1:
        k2_path = Path(f"data/scenario_banks/{split}_smoke.json")
        if k2_path.exists():
            from dataclasses import replace
            k2_bank = load_scenario_bank(k2_path)
            k1_scenarios = tuple(
                replace(s, maintenance_capacity=1) for s in k2_bank.scenarios
            )
            return ScenarioBank(
                bank_id=f"{k2_bank.bank_id}_k1",
                split=split,
                scenarios=k1_scenarios,
            )

    raise FileNotFoundError(f"Scenario bank not found: {scenario_bank_path}")


def _create_boundary_scenario(
    split: str,
    k_capacity: int,
    prediction_store: Any,
    failure_offset: int,
) -> Optional[Scenario]:
    """
    Create a deterministic boundary-failure scenario.

    Finds a real unit with trajectory_length = failure_cycle and sets
    start_cycle = trajectory_length - offset to trigger failure at step 1.
    """
    units = prediction_store.get_units(split)

    # Find a unit where we can trigger failure at the specified offset
    for unit_id in units:
        pred_cycle1 = prediction_store.get(split, unit_id, 1)
        traj_len = pred_cycle1.trajectory_length

        # We need a unit where start_cycle + offset <= trajectory_length
        # and start_cycle >= 1
        start_cycle = traj_len - failure_offset
        if start_cycle >= 1:
            # Verify the failure cycle exists
            failure_cycle = start_cycle + failure_offset
            if failure_cycle <= traj_len:
                pred_failure = prediction_store.get(split, unit_id, failure_cycle)
                if pred_failure.true_rul <= 0:
                    # Found a valid unit for this offset
                    # Fill other 4 slots with healthy units at cycle 1
                    other_units = [u for u in units if u != unit_id][:4]
                    if len(other_units) < 4:
                        continue

                    initial_unit_ids = [unit_id] + other_units
                    initial_cycles = [start_cycle] + [1] * 4

                    return Scenario(
                        scenario_id=f"boundary_offset_{failure_offset}",
                        split=split,
                        initial_unit_ids=tuple(initial_unit_ids),
                        initial_cycles=tuple(initial_cycles),
                        replacement_seed=6521,
                        environment_seed=6521,
                        episode_horizon=100,
                        maintenance_capacity=k_capacity,
                        cost_regime_id="failure-light-no-waste",
                    )

    return None


def _create_simultaneous_failure_scenario(
    split: str,
    k_capacity: int,
    prediction_store: Any,
) -> Optional[Scenario]:
    """
    Create a deterministic simultaneous-failure scenario.

    Finds two distinct units and places them at cycles where both will
    fail when advanced by delta_cycles=5 in the same step.
    """
    units = prediction_store.get_units(split)

    # Find two units that can fail at the same step when advanced
    # Strategy: find units where cycle + 1 has true_rul <= 0 (immediate failure on advance)
    candidates = []
    for unit_id in units:
        pred_cycle1 = prediction_store.get(split, unit_id, 1)
        traj_len = pred_cycle1.trajectory_length
        # Check if this unit fails at cycle 2 (start at 1, advance by 1)
        # Or we can set start_cycle = traj_len - 1, so advancing by 1 hits failure
        # Actually, let's find units where trajectory_length - 1 has true_rul > 0
        # and trajectory_length has true_rul <= 0
        if traj_len >= 2:
            pred_at_end = prediction_store.get(split, unit_id, traj_len)
            if pred_at_end.true_rul <= 0:
                # This unit fails at trajectory_length
                candidates.append((unit_id, traj_len - 1))  # Start one cycle before failure

    if len(candidates) >= 2:
        # Pick two units and set their start cycles so they both fail on step 1
        unit_a, start_a = candidates[0]
        unit_b, start_b = candidates[1]
        other_units = [u for u in units if u not in [unit_a, unit_b]][:3]

        if len(other_units) >= 3:
            return Scenario(
                scenario_id="simultaneous_failure",
                split=split,
                initial_unit_ids=(unit_a, unit_b) + tuple(other_units),
                initial_cycles=(start_a, start_b, 1, 1, 1),
                replacement_seed=6521,
                environment_seed=6521,
                episode_horizon=100,
                maintenance_capacity=k_capacity,
                cost_regime_id="failure-light-no-waste",
            )

    return None


def _create_mixed_event_scenario(
    split: str,
    k_capacity: int,
    prediction_store: Any,
) -> Optional[Scenario]:
    """
    Create a deterministic mixed-event scenario.

    One PM slot (unit at cycle 1, will be preventively maintained)
    One failure slot (unit placed at trajectory_length - 1, will fail on advance)
    Three healthy filler slots at cycle 1.
    """
    units = prediction_store.get_units(split)

    # Find a unit to place at its failure boundary
    failure_unit = None
    failure_start_cycle = None
    for unit_id in units:
        pred_cycle1 = prediction_store.get(split, unit_id, 1)
        traj_len = pred_cycle1.trajectory_length
        if traj_len >= 2:
            pred_at_end = prediction_store.get(split, unit_id, traj_len)
            if pred_at_end.true_rul <= 0:
                failure_unit = unit_id
                failure_start_cycle = traj_len - 1
                break

    if failure_unit is None:
        return None

    # Find 4 other healthy units (not the failure unit)
    other_units = [u for u in units if u != failure_unit][:4]
    if len(other_units) < 4:
        return None

    # failure_unit goes in slot 1 (will fail on step 1 when advanced)
    # other_units[0] goes in slot 0 (will be PM target)
    pm_unit = other_units[0]
    filler_units = other_units[1:4]

    return Scenario(
        scenario_id="mixed_event",
        split=split,
        initial_unit_ids=(pm_unit, failure_unit) + tuple(filler_units),
        initial_cycles=(1, failure_start_cycle, 1, 1, 1),
        replacement_seed=6521,
        environment_seed=6521,
        episode_horizon=100,
        maintenance_capacity=k_capacity,
        cost_regime_id="failure-light-no-waste",
    )


def run_smoke_rollout(
    split: str,
    k_capacity: int,
    cost_regime_id: str,
    seed: int,
    policy: str = "random",
    verbose: bool = False,
    prediction_store: Any = None,
    scenario_bank: ScenarioBank = None,
    max_steps: int = 150,  # Safety bound
) -> SmokeResult:
    """
    Run a single smoke rollout with specified policy.

    Args:
        split: Environment split (must be predictor_train or rl_validation)
        k_capacity: Maintenance capacity K
        cost_regime_id: Cost regime
        seed: Random seed
        policy: Action policy mode
        verbose: Print step-by-step output
        prediction_store: Optional pre-loaded prediction store
        scenario_bank: Optional pre-loaded scenario bank
        max_steps: Safety bound to prevent infinite loops

    Returns:
        SmokeResult with actual tracked values and complete metadata
    """
    result = SmokeResult(
        split=split,
        maintenance_capacity=k_capacity,
        cost_regime_id=cost_regime_id,
        seed=seed,
        policy=policy,
    )

    # Validate split - reject rl_test before any loading
    if split not in ALLOWED_SMOKE_SPLITS:
        result.errors.append(
            f"Split '{split}' is not allowed for smoke testing. "
            f"Allowed splits: {sorted(ALLOWED_SMOKE_SPLITS)}. "
            f"rl_test is reserved and must not be used for development smoke."
        )
        return result

    # Load prediction store if not provided
    if prediction_store is None:
        prediction_store = _load_prediction_store()

    # Load or create scenario bank
    use_scenario_bank = scenario_bank
    if use_scenario_bank is None:
        try:
            use_scenario_bank = _load_scenario_bank(split, k_capacity)
        except FileNotFoundError as e:
            result.errors.append(str(e))
            return result

    # Validate scenarios belong to split
    valid_units = set(prediction_store.get_units(split))
    for scenario in use_scenario_bank.scenarios:
        if scenario.split != split:
            result.split_violation_count += 1
        for unit_id in scenario.initial_unit_ids:
            if unit_id not in valid_units:
                result.split_violation_count += 1

    # Create environment config
    config = get_default_config(
        split=split,
        maintenance_capacity=k_capacity,
        cost_regime_id=cost_regime_id,
        scenario_bank_path=str(use_scenario_bank.scenarios[0].scenario_id if use_scenario_bank.scenarios else "unknown"),
    )

    env = SelectiveMaintenanceEnv(
        config=config,
        prediction_store=prediction_store,
        scenario_bank=use_scenario_bank,
        info_mode="diagnostic",
    )

    # Create seeded RNG for action selection
    action_rng = np.random.default_rng(seed)

    # Reset
    try:
        obs, info = env.reset(seed=seed)
    except Exception as e:
        result.errors.append(f"Reset failed: {e}")
        env.close()
        return result

    if verbose:
        print(f"Seed {seed}, Split {split}, K={k_capacity}, Regime={cost_regime_id}, Policy={policy}")
        print(f"Initial obs shape: {obs.shape}")

    # Check initial observation
    if np.any(np.isnan(obs)):
        result.nan_observation_count += 1
    if np.any(np.isinf(obs)):
        result.inf_observation_count += 1

    result.episodes = 1
    truncated = False
    step_count = 0

    try:
        if policy == "random":
            # Seeded random feasible actions with safety bound
            while not truncated and step_count < max_steps:
                action_mask = env.get_action_mask()
                valid_actions = np.where(action_mask)[0]
                action = int(action_rng.choice(valid_actions))
                _, _, _, truncated, _ = _step_and_track(env, action, result, verbose, step_count)
                step_count += 1

            if step_count >= max_steps and not truncated:
                result.errors.append(f"Random policy did not terminate within {max_steps} steps")

        elif policy == "corrective-only":
            # Always action 0 (no PM, only corrective replacements on failure)
            while not truncated and step_count < max_steps:
                _, _, _, truncated, _ = _step_and_track(env, 0, result, verbose, step_count)
                step_count += 1

            if step_count >= max_steps and not truncated:
                result.errors.append(f"Corrective-only policy did not terminate within {max_steps} steps")

        elif policy == "boundary":
            # Controlled in-memory failure scenarios at offsets 1, 2, 3, 4, 5
            # Each case is a separate episode: reset -> step(action=0) -> close
            # The initial env.reset() at line 412 doesn't count for boundary policy
            result.episodes = 0  # Reset - will count each controlled case separately
            result.offsets_tested = []
            result.failure_boundary_cycles = []
            result.boundary_cases = []

            for offset in [1, 2, 3, 4, 5]:
                # Create boundary scenario for this offset
                boundary_scenario = _create_boundary_scenario(split, k_capacity, prediction_store, offset)
                if boundary_scenario is None:
                    result.errors.append(f"Could not create boundary scenario for offset {offset}")
                    continue

                # Create single-scenario bank
                single_bank = ScenarioBank(
                    bank_id=f"boundary_offset_{offset}",
                    split=split,
                    scenarios=(boundary_scenario,),
                )

                # Reset a fresh environment with this scenario
                env_boundary = SelectiveMaintenanceEnv(
                    config=config,
                    prediction_store=prediction_store,
                    scenario_bank=single_bank,
                    info_mode="diagnostic",
                )

                obs_b, info_b = env_boundary.reset(seed=seed)
                if np.any(np.isnan(obs_b)):
                    result.nan_observation_count += 1
                if np.any(np.isinf(obs_b)):
                    result.inf_observation_count += 1

                # Execute one step (action 0 to let failure occur)
                _, _, _, truncated, step_info = _step_and_track(env_boundary, 0, result, verbose, step_count)
                step_count += 1
                result.episodes += 1  # Each controlled case is one episode
                result.offsets_tested.append(offset)

                # Compute the actual failure cycle from scenario data
                # failure_cycle = start_cycle + offset = trajectory_length (for V2 cache)
                scenario = boundary_scenario
                failure_unit_id = scenario.initial_unit_ids[0]
                start_cycle = scenario.initial_cycles[0]
                failure_cycle = start_cycle + offset

                # Find the slot that failed (slot 0 contains the failure unit)
                pred_failure = prediction_store.get(split, failure_unit_id, failure_cycle)
                if pred_failure.found and pred_failure.true_rul <= 0:
                    result.failure_boundary_cycles.append(failure_cycle)

                # Capture actual post-step SlotState for the failed slot
                # to record replacement_cycle and replacement_age from state.
                failure_snapshot = _capture_slot_snapshot(env_boundary, 0, step_info)
                replacement_cycle = failure_snapshot["cycle"] if failure_snapshot else None
                replacement_age = failure_snapshot["age_since_replacement_cycles"] if failure_snapshot else None

                # Build boundary case metadata
                result.boundary_cases.append({
                    "offset": offset,
                    "unit_id": failure_unit_id,
                    "start_cycle": start_cycle,
                    "failure_cycle": failure_cycle,
                    "trajectory_length": pred_failure.trajectory_length,
                    "observed_failures": step_info.get("num_failures", 0),
                    "replacement_cycle": replacement_cycle,
                    "replacement_age": replacement_age,
                })

                env_boundary.close()

            # STOP after 5 controlled cases - do not continue random episode
            # Validate expected results
            if result.episodes != 5:
                result.errors.append(f"Boundary policy: expected 5 episodes, got {result.episodes}")
            if result.steps != 5:
                result.errors.append(f"Boundary policy: expected 5 steps, got {result.steps}")
            if result.failures != 5:
                result.errors.append(f"Boundary policy: expected 5 failures, got {result.failures}")
            if result.preventive_replacements != 0:
                result.errors.append(f"Boundary policy: expected 0 PM, got {result.preventive_replacements}")

            # Validate actual replacement cycle and age from state for all boundary cases
            for case in result.boundary_cases:
                if case.get("replacement_cycle") != 1:
                    result.errors.append(
                        f"Boundary offset {case['offset']}: replacement_cycle must be 1, "
                        f"got {case['replacement_cycle']}"
                    )
                if case.get("replacement_age") != 0:
                    result.errors.append(
                        f"Boundary offset {case['offset']}: replacement_age must be 0, "
                        f"got {case['replacement_age']}"
                    )

            result.completed = len(result.errors) == 0

        elif policy == "simultaneous-failure":
            # Create deterministic in-memory scenario with two failures
            result.simultaneous_cases = []
            sim_scenario = _create_simultaneous_failure_scenario(split, k_capacity, prediction_store)
            if sim_scenario is None:
                result.errors.append(
                    "Could not create simultaneous-failure scenario - "
                    "no two units fail at same cycle in this split"
                )
            else:
                single_bank = ScenarioBank(
                    bank_id="simultaneous_failure",
                    split=split,
                    scenarios=(sim_scenario,),
                )

                env_sim = SelectiveMaintenanceEnv(
                    config=config,
                    prediction_store=prediction_store,
                    scenario_bank=single_bank,
                    info_mode="diagnostic",
                )

                obs_s, _ = env_sim.reset(seed=seed)
                if np.any(np.isnan(obs_s)):
                    result.nan_observation_count += 1
                if np.any(np.isinf(obs_s)):
                    result.inf_observation_count += 1

                # Execute one step (action 0)
                _, _, _, truncated, step_info = _step_and_track(env_sim, 0, result, verbose, step_count)
                step_count += 1

                # Strengthened checks for simultaneous-failure
                if result.steps != 1:
                    result.errors.append(f"Simultaneous-failure: expected 1 step, got {result.steps}")
                if result.preventive_replacements != 0:
                    result.errors.append(f"Simultaneous-failure: expected 0 PM, got {result.preventive_replacements}")
                if result.failures != 2:
                    result.errors.append(f"Simultaneous-failure: expected 2 failures, got {result.failures}")
                expected_failure_cost = 2 * config.get_cost_regime().c_f
                if abs(result.failure_cost - expected_failure_cost) > 1e-6:
                    result.errors.append(f"Simultaneous-failure: expected failure_cost={expected_failure_cost}, got {result.failure_cost}")

                # Verify both failed slots end at cycle 1 and age 0 by reading
                # actual post-step SlotState from diagnostic info.
                replacement_cycles = []
                replacement_ages = []
                failure_slot_indices = []
                replacement_unit_ids = []

                for slot_idx in range(5):
                    snapshot = _capture_slot_snapshot(env_sim, slot_idx, step_info)
                    if snapshot is not None and snapshot["cycle"] == 1:
                        failure_slot_indices.append(slot_idx)
                        replacement_unit_ids.append(snapshot["unit_id"])
                        replacement_cycles.append(snapshot["cycle"])
                        replacement_ages.append(snapshot["age_since_replacement_cycles"])

                if len(replacement_cycles) != 2:
                    result.errors.append(f"Simultaneous-failure: expected 2 replacements at cycle 1, found {len(replacement_cycles)}")
                if replacement_cycles != [1, 1]:
                    result.errors.append(f"Simultaneous-failure: both replacements must end at cycle 1, got {replacement_cycles}")
                if replacement_ages != [0, 0]:
                    result.errors.append(f"Simultaneous-failure: both replacements must have age 0, got {replacement_ages}")

                result.simultaneous_cases.append({
                    "failure_slot_indices": failure_slot_indices,
                    "replacement_unit_ids": replacement_unit_ids,
                    "replacement_cycles": replacement_cycles,
                    "replacement_ages": replacement_ages,
                })

                env_sim.close()

            result.completed = len(result.errors) == 0

        elif policy == "mixed-event":
            # Create deterministic in-memory scenario with PM + failure
            result.mixed_cases = []
            mixed_scenario = _create_mixed_event_scenario(split, k_capacity, prediction_store)
            if mixed_scenario is None:
                result.errors.append(
                    "Could not create mixed-event scenario - "
                    "no unit fails at cycle 2 in this split"
                )
            else:
                single_bank = ScenarioBank(
                    bank_id="mixed_event",
                    split=split,
                    scenarios=(mixed_scenario,),
                )

                env_mix = SelectiveMaintenanceEnv(
                    config=config,
                    prediction_store=prediction_store,
                    scenario_bank=single_bank,
                    info_mode="diagnostic",
                )

                obs_m, _ = env_mix.reset(seed=seed)
                if np.any(np.isnan(obs_m)):
                    result.nan_observation_count += 1
                if np.any(np.isinf(obs_m)):
                    result.inf_observation_count += 1

                # Execute PM on slot 0 (action 1 = slot 0 only)
                pm_action = 1  # Action 1 selects slot 0
                _, _, _, truncated, step_info = _step_and_track(env_mix, pm_action, result, verbose, step_count)
                step_count += 1

                # Strengthened checks for mixed-event
                if result.steps != 1:
                    result.errors.append(f"Mixed-event: expected 1 step, got {result.steps}")
                if result.preventive_replacements != 1:
                    result.errors.append(f"Mixed-event: expected 1 PM, got {result.preventive_replacements}")
                if result.failures != 1:
                    result.errors.append(f"Mixed-event: expected 1 failure, got {result.failures}")
                expected_pm_cost = config.get_cost_regime().c_pm
                expected_failure_cost = config.get_cost_regime().c_f
                if abs(result.preventive_cost - expected_pm_cost) > 1e-6:
                    result.errors.append(f"Mixed-event: expected preventive_cost={expected_pm_cost}, got {result.preventive_cost}")
                if abs(result.failure_cost - expected_failure_cost) > 1e-6:
                    result.errors.append(f"Mixed-event: expected failure_cost={expected_failure_cost}, got {result.failure_cost}")

                # Verify PM and failure occur on different slots
                pm_slots = []
                failure_slots = []
                for slot_idx in range(5):
                    diag_key = f"slot_{slot_idx}_diagnostic"
                    if diag_key in step_info:
                        diag = step_info[diag_key]
                        if diag.get("cycle") == 1:
                            # This slot was replaced - determine if PM or failure
                            # PM slot: was in selected_slots (action target)
                            # Failure slot: was NOT in selected_slots
                            selected = step_info.get("selected_slots", [])
                            if slot_idx in selected:
                                pm_slots.append(slot_idx)
                            else:
                                failure_slots.append(slot_idx)

                if len(pm_slots) != 1:
                    result.errors.append(f"Mixed-event: expected 1 PM slot, found {len(pm_slots)}")
                if len(failure_slots) != 1:
                    result.errors.append(f"Mixed-event: expected 1 failure slot, found {len(failure_slots)}")
                if pm_slots and failure_slots and pm_slots[0] == failure_slots[0]:
                    result.errors.append(f"Mixed-event: PM and failure must be on different slots")

                # Verify both replacement cycles are 1 and ages are 0 by reading
                # actual post-step SlotState from diagnostic info.
                replacement_cycles = []
                replacement_ages = []
                replacement_unit_ids = []

                for slot_idx in range(5):
                    snapshot = _capture_slot_snapshot(env_mix, slot_idx, step_info)
                    if snapshot is not None and snapshot["cycle"] == 1:
                        replacement_unit_ids.append(snapshot["unit_id"])
                        replacement_cycles.append(snapshot["cycle"])
                        replacement_ages.append(snapshot["age_since_replacement_cycles"])

                if len(replacement_cycles) != 2:
                    result.errors.append(f"Mixed-event: expected 2 total replacements, found {len(replacement_cycles)}")
                if replacement_cycles != [1, 1]:
                    result.errors.append(f"Mixed-event: both replacements must end at cycle 1, got {replacement_cycles}")
                if replacement_ages != [0, 0]:
                    result.errors.append(f"Mixed-event: both replacements must have age 0, got {replacement_ages}")

                result.mixed_cases.append({
                    "pm_slot": pm_slots[0] if pm_slots else None,
                    "failure_slot": failure_slots[0] if failure_slots else None,
                    "replacement_unit_ids": replacement_unit_ids,
                    "replacement_cycles": replacement_cycles,
                    "replacement_ages": replacement_ages,
                })

                env_mix.close()
                result.offsets_tested = [1]  # One controlled step

            result.completed = len(result.errors) == 0

        else:
            result.errors.append(f"Unknown policy: {policy}")

    finally:
        env.close()

    # Validate step count matches expected horizon
    if policy not in ["boundary", "simultaneous-failure", "mixed-event"]:
        if result.steps != 100:
            result.errors.append(
                f"Expected 100 steps for {policy} policy, got {result.steps}"
            )
        else:
            result.completed = True
    else:
        # Controlled policies - just mark completed if no errors
        result.completed = len(result.errors) == 0

    return result


def run_validation_matrix(
    splits: Optional[List[str]] = None,
    k_values: Optional[List[int]] = None,
    seeds: Optional[List[int]] = None,
    cost_regimes: Optional[List[str]] = None,
    policy: str = "random",
    verbose: bool = False,
) -> List[SmokeResult]:
    """
    Run validation matrix across configurations.

    Covers:
    - predictor_train K=1
    - predictor_train K=2
    - rl_validation K=1
    - rl_validation K=2
    """
    if splits is None:
        splits = ["predictor_train", "rl_validation"]
    if k_values is None:
        k_values = [1, 2]  # K=1 first, then K=2
    if seeds is None:
        seeds = [6521, 6522, 6523, 6524, 6525]
    if cost_regimes is None:
        cost_regimes = ["failure-light-no-waste"]

    results = []

    print(f"Running validation matrix:")
    print(f"  Splits: {splits}")
    print(f"  K values: {k_values}")
    print(f"  Seeds: {seeds}")
    print(f"  Cost regimes: {cost_regimes}")
    print(f"  Policy: {policy}")
    print()

    # Load prediction store once
    prediction_store = _load_prediction_store()

    for split in splits:
        if split not in ALLOWED_SMOKE_SPLITS:
            print(f"ERROR: Split '{split}' is not allowed. Skipping.")
            continue

        for k in k_values:
            for regime in cost_regimes:
                # Load scenario bank for this split/K (K=1 derived internally if needed)
                scenario_bank = _load_scenario_bank(split, k)

                for seed in seeds:
                    result = run_smoke_rollout(
                        split=split,
                        k_capacity=k,
                        cost_regime_id=regime,
                        seed=seed,
                        policy=policy,
                        verbose=verbose,
                        prediction_store=prediction_store,
                        scenario_bank=scenario_bank,
                    )
                    results.append(result)

                    if verbose:
                        print(f"  => Steps={result.steps}, "
                              f"PM={result.preventive_replacements}, "
                              f"Fail={result.failures}")

    return results


def run_all_regimes(
    split: str,
    k_capacity: int,
    seeds: List[int],
    policy: str = "random",
    verbose: bool = False,
) -> List[SmokeResult]:
    """
    Run all four frozen cost regimes.

    Fails with clear usage error if split is not explicitly provided.
    """
    if split not in ALLOWED_SMOKE_SPLITS:
        raise ValueError(
            f"--all-regimes requires an explicit split from {sorted(ALLOWED_SMOKE_SPLITS)}. "
            f"Got split='{split}'. rl_test is not allowed for development smoke."
        )

    results = []
    regime_ids = list_cost_regimes()

    print(f"Running all {len(regime_ids)} cost regimes on split={split}, K={k_capacity}, policy={policy}")

    # Load prediction store once
    prediction_store = _load_prediction_store()

    for regime_id in regime_ids:
        # Load scenario bank and modify cost regime in-memory
        scenario_bank = _load_scenario_bank(split, k_capacity)

        # Create modified scenarios with the new regime
        from dataclasses import replace
        modified_scenarios = tuple(
            replace(s, cost_regime_id=regime_id) for s in scenario_bank.scenarios
        )
        modified_bank = ScenarioBank(
            bank_id=f"{scenario_bank.bank_id}_{regime_id}",
            split=split,
            scenarios=modified_scenarios,
        )

        for seed in seeds:
            result = run_smoke_rollout(
                split=split,
                k_capacity=k_capacity,
                cost_regime_id=regime_id,
                seed=seed,
                policy=policy,
                verbose=verbose,
                prediction_store=prediction_store,
                scenario_bank=modified_bank,
            )
            results.append(result)

    return results


def print_summary(results: List[SmokeResult]) -> None:
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("SMOKE ROLLOUT SUMMARY")
    print("=" * 70)

    total_rollouts = len(results)
    successful = sum(1 for r in results if len(r.errors) == 0)
    failed = total_rollouts - successful

    print(f"\nTotal rollouts: {total_rollouts}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if successful > 0:
        successful_results = [r for r in results if len(r.errors) == 0]

        total_steps = sum(r.steps for r in successful_results)
        total_episodes = sum(r.episodes for r in successful_results)
        total_pm = sum(r.preventive_replacements for r in successful_results)
        total_failures = sum(r.failures for r in successful_results)
        total_preventive_cost = sum(r.preventive_cost for r in successful_results)
        total_failure_cost = sum(r.failure_cost for r in successful_results)
        total_wasted_cost = sum(r.wasted_life_cost for r in successful_results)
        total_cost = sum(r.total_cost for r in successful_results)

        print(f"\nSuccessful rollout statistics:")
        print(f"  Total episodes: {total_episodes}")
        print(f"  Total environment steps: {total_steps}")
        print(f"  Total preventive replacements: {total_pm}")
        print(f"  Total failures: {total_failures}")
        print(f"  Total preventive cost: {total_preventive_cost:.2f}")
        print(f"  Total failure cost: {total_failure_cost:.2f}")
        print(f"  Total wasted-life cost: {total_wasted_cost:.2f}")
        print(f"  Total cost: {total_cost:.2f}")

        # Check for NaN/Inf
        total_nan = sum(r.nan_observation_count for r in successful_results)
        total_inf = sum(r.inf_observation_count for r in successful_results)
        print(f"  NaN observations: {total_nan}")
        print(f"  Inf observations: {total_inf}")

        # Split violations
        total_split_violations = sum(r.split_violation_count for r in successful_results)
        print(f"  Split violations: {total_split_violations}")

        # Completion rate
        completed = sum(1 for r in successful_results if r.completed)
        print(f"  Completed episodes: {completed}/{len(successful_results)}")

        # Breakdown by configuration
        print(f"\nBreakdown by configuration:")

        by_config = {}
        for r in successful_results:
            key = (r.split, r.maintenance_capacity, r.cost_regime_id, r.policy)
            if key not in by_config:
                by_config[key] = []
            by_config[key].append(r)

        for (split, k, regime, policy), results_list in sorted(by_config.items()):
            avg_steps = np.mean([r.steps for r in results_list])
            print(f"  split={split}, K={k}, regime={regime}, policy={policy}: "
                  f"{len(results_list)} rollouts, avg {avg_steps:.1f} steps")

    if failed > 0:
        print(f"\nFailed rollouts:")
        for r in results:
            if r.errors:
                config = f"split={r.split}, K={r.maintenance_capacity}, seed={r.seed}, policy={r.policy}"
                errors = "; ".join(r.errors[:3])
                print(f"  {config}: {errors}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Milestone 2 Environment Smoke Rollout")
    parser.add_argument("--split", type=str, default=None,
                        help="Environment split (predictor_train, rl_validation). rl_test is NOT allowed.")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of seeds to run")
    parser.add_argument("--k-capacity", type=int, default=None,
                        help="Maintenance capacity K (1 or 2)")
    parser.add_argument("--cost-regime", type=str, default=None,
                        help="Cost regime ID")
    parser.add_argument("--all-regimes", action="store_true",
                        help="Run all four cost regimes")
    parser.add_argument("--validation-matrix", action="store_true",
                        help="Run full validation matrix (predictor_train/rl_validation x K=1/K=2)")
    parser.add_argument("--policy", type=str, default="random",
                        choices=["random", "corrective-only", "boundary", "simultaneous-failure", "mixed-event"],
                        help="Action policy mode")
    parser.add_argument("--verbose", action="store_true",
                        help="Print step-by-step output")

    args = parser.parse_args()

    # Validate split for modes that require it
    if args.validation_matrix:
        # Validation matrix mode
        splits = ["predictor_train", "rl_validation"]
        k_values = [1, 2] if args.k_capacity is None else [args.k_capacity]
        seeds = list(range(6521, 6521 + args.seeds))
        cost_regimes = list_cost_regimes() if args.all_regimes else (
            [args.cost_regime] if args.cost_regime else ["failure-light-no-waste"]
        )

        results = run_validation_matrix(
            splits=splits,
            k_values=k_values,
            seeds=seeds,
            cost_regimes=cost_regimes,
            policy=args.policy,
            verbose=args.verbose,
        )
    elif args.all_regimes:
        # All regimes mode - requires explicit split
        if not args.split:
            print("ERROR: --all-regimes requires an explicit --split (predictor_train or rl_validation)")
            print("Usage: python run_m2_environment_smoke.py --all-regimes --split predictor_train")
            sys.exit(1)

        k_value = 2 if args.k_capacity is None else args.k_capacity
        seeds = list(range(6521, 6521 + args.seeds))

        results = run_all_regimes(
            split=args.split,
            k_capacity=k_value,
            seeds=seeds,
            policy=args.policy,
            verbose=args.verbose,
        )
    elif args.split:
        # Single split mode
        if args.split not in ALLOWED_SMOKE_SPLITS:
            print(f"ERROR: Split '{args.split}' is not allowed for smoke testing.")
            print(f"Allowed splits: {sorted(ALLOWED_SMOKE_SPLITS)}")
            print("rl_test is reserved and must not be used for development smoke.")
            sys.exit(1)

        k_values = [2] if args.k_capacity is None else [args.k_capacity]
        seeds = list(range(6521, 6521 + args.seeds))
        cost_regimes = list_cost_regimes() if args.all_regimes else (
            [args.cost_regime] if args.cost_regime else ["failure-light-no-waste"]
        )

        results = []
        # Load prediction store once
        prediction_store = _load_prediction_store()

        for split in [args.split]:
            for k in k_values:
                for regime in cost_regimes:
                    for seed in seeds:
                        result = run_smoke_rollout(
                            split=split,
                            k_capacity=k,
                            cost_regime_id=regime,
                            seed=seed,
                            policy=args.policy,
                            verbose=args.verbose,
                            prediction_store=prediction_store,
                        )
                        results.append(result)
    else:
        # Default: quick smoke test
        print("Running default smoke test (single rollout)...")
        k_value = 2 if args.k_capacity is None else args.k_capacity
        result = run_smoke_rollout(
            split="rl_validation",
            k_capacity=k_value,
            cost_regime_id="failure-light-no-waste",
            seed=6521,
            policy=args.policy,
            verbose=args.verbose,
        )
        results = [result]

    print_summary(results)

    # Exit with error if any failed
    failed = sum(1 for r in results if r.errors)
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()