"""
Selective Maintenance Environment for Milestone 2.

Implements a Gymnasium-compatible continuing-fleet environment where:
- Agent observes N=5 engines with predicted RUL and age
- Agent selects up to K engines for preventive maintenance
- Non-maintained engines advance by delta_cycles=5
- Failures are detected during advancement and correctively replaced
- Reward is negative total cost (PM + failure + wasted life)

This is a synthetic benchmark using NASA C-MAPSS FD001 trajectories
as a degradation library. It is not a real aviation digital twin.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from .action_table import (
    ACTION_TABLE_N5_K1,
    ACTION_TABLE_N5_K2,
    ActionSubset,
    action_id_to_slots,
    build_action_table,
)
from .config import EnvironmentConfig, get_default_config
from .costs import CostRegime, calculate_total_cost
from .errors import (
    ContractViolationError,
    InformationLeakageError,
    InvalidActionError,
    MissingPredictionError,
    ScenarioValidationError,
    SplitViolationError,
)
from .scenario_bank import Scenario, ScenarioBank, load_scenario_bank
from .state import SlotState

# Try to import PredictionStore from predictors module
try:
    from ..predictors.prediction_store import PredictionResult, load_default_prediction_store

    HAS_PREDICTORS = True
except ImportError:
    HAS_PREDICTORS = False
    PredictionResult = None  # type: ignore


@dataclass
class FleetState:
    """Internal fleet state container."""

    slots: List[SlotState]
    step_index: int
    episode_return: float
    episode_completed: bool
    replacement_rng: np.random.Generator


class SelectiveMaintenanceEnv(gym.Env[np.ndarray, int]):
    """
    Gymnasium-compatible Selective Maintenance Environment.

    Observation space (per slot):
        [normalized_age_since_replacement, normalized_predicted_rul]

    For N=5, flattened to shape (10,):
        [slot_0_age, slot_0_pred_rul, slot_1_age, slot_1_pred_rul, ...]

    Action space:
        Discrete(number_of_actions) where:
        - N=5, K=2: 16 actions
        - N=5, K=1: 6 actions

    Episode terminates via truncation at horizon, not via failure.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Optional[EnvironmentConfig] = None,
        prediction_store: Optional[Any] = None,
        scenario_bank: Optional[ScenarioBank] = None,
        scenario_selection: Optional[List[str]] = None,
        info_mode: str = "normal",
    ):
        """
        Initialize the environment.

        Args:
            config: Environment configuration. Uses defaults if None.
            prediction_store: PredictionStore instance for prediction lookup.
                If None, loads default V2 store.
            scenario_bank: ScenarioBank for episode initialization.
                If None, loads from config.scenario_bank_path.
            scenario_selection: Optional list of scenario IDs to use.
                If None, uses all scenarios in the bank.
            info_mode: "normal" (training-safe) or "diagnostic" (eval-only).
        """
        super().__init__()

        # Load or validate configuration
        self.config = config if config is not None else get_default_config()

        # Validate info_mode
        if info_mode not in {"normal", "diagnostic"}:
            raise ValueError(f"info_mode must be 'normal' or 'diagnostic', got '{info_mode}'")
        self.info_mode = info_mode
        if self.info_mode != self.config.info_mode:
            # Override config info_mode for this instance
            pass

        # Set N and K from config
        self.N = self.config.fleet_size  # Should be 5
        self.K = self.config.maintenance_capacity  # 1 or 2

        # Build action table for this K
        if self.K == 1:
            self.action_table = ACTION_TABLE_N5_K1
        elif self.K == 2:
            self.action_table = ACTION_TABLE_N5_K2
        else:
            self.action_table = build_action_table(self.N, self.K)

        self.num_actions = len(self.action_table)

        # Define action space
        self.action_space = gym.spaces.Discrete(self.num_actions)

        # Define observation space: (N * 2,) with values in [0, 1]
        obs_shape = (self.N * 2,)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=obs_shape,
            dtype=np.float32,
        )

        # Load prediction store
        if prediction_store is not None:
            self.prediction_store = prediction_store
        else:
            if not HAS_PREDICTORS:
                raise ImportError(
                    "Predictors module not available. "
                    "Cannot load default PredictionStore."
                )
            from pathlib import Path

            cache_dir = Path(self.config.prediction_cache_path)
            self.prediction_store = load_default_prediction_store(cache_dir)

        # Load scenario bank
        if scenario_bank is not None:
            self.scenario_bank = scenario_bank
        else:
            from pathlib import Path

            scenario_path = Path(self.config.scenario_bank_path)
            if scenario_path.exists():
                self.scenario_bank = load_scenario_bank(scenario_path)
            else:
                # Create empty scenario bank - scenarios must be provided
                self.scenario_bank = ScenarioBank(
                    bank_id="empty",
                    split=self.config.split,
                    scenarios=(),
                )

        # Filter scenarios if selection provided
        if scenario_selection is not None:
            selected = tuple(
                s for s in self.scenario_bank.scenarios
                if s.scenario_id in scenario_selection
            )
            self.scenario_bank = ScenarioBank(
                bank_id=self.scenario_bank.bank_id + "_filtered",
                split=self.scenario_bank.split,
                scenarios=selected,
            )

        # Validate scenario bank split matches config
        if self.scenario_bank.split != self.config.split:
            raise SplitViolationError(
                expected_split=self.config.split,
                actual_split=self.scenario_bank.split,
            )

        # Internal state (set by reset)
        self._fleet_state: Optional[FleetState] = None
        self._current_scenario: Optional[Scenario] = None
        self._reset_called = False
        self._effective_reset_seed: Optional[int] = None

        # Cost regime
        self.cost_regime = self.config.get_cost_regime()

        # Constants from config
        self.delta_cycles = self.config.delta_cycles
        self.horizon = self.config.episode_horizon
        self.rul_scale = self.config.rul_scale  # 125.0
        self.age_scale = self.config.age_scale_cycles  # 341

    # ------------------------------------------------------------------
    # Centralized required-prediction access (contract section 13)
    # ------------------------------------------------------------------
    def _require_prediction(
        self,
        split: str,
        unit_id: int,
        cycle: int,
        slot_index: int,
        env_step: int,
    ) -> PredictionResult:
        """
        Look up a required prediction. Every required PredictionStore lookup
        in this environment MUST go through this helper.

        Raises MissingPredictionError with all five pieces of context
        (split, unit_id, cycle, slot_index, env_step) if the record is absent.
        Never silently substitutes, never uses a neighboring cycle, never
        falls back to true RUL, and never converts a missing prediction into
        normal episode behavior.
        """
        pred_result = self.prediction_store.get(split, unit_id, cycle)
        if not pred_result.found:
            raise MissingPredictionError(
                split=split,
                unit_id=unit_id,
                cycle=cycle,
                slot_index=slot_index,
                env_step=env_step,
            )
        return pred_result

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility. If None, uses config.seed.
            options: Optional dict with:
                - "scenario_id": Specific scenario to use

        Returns:
            observation: Initial observation ndarray (shape (10,), dtype np.float32)
            info: Initial info dict (training-safe or diagnostic)

        Raises:
            ScenarioValidationError: If scenario validation fails
            MissingPredictionError: If initial predictions are missing
        """
        super().reset(seed=seed)

        # Effective reset seed
        env_seed = seed if seed is not None else self.config.seed
        self._effective_reset_seed = int(env_seed)
        self.np_random = np.random.default_rng(env_seed)

        # Select scenario
        if options is not None and "scenario_id" in options:
            scenario_id = options["scenario_id"]
            scenario = None
            for s in self.scenario_bank.scenarios:
                if s.scenario_id == scenario_id:
                    scenario = s
                    break
            if scenario is None:
                raise ScenarioValidationError(
                    scenario_id=scenario_id,
                    reason=f"Scenario not found in bank",
                )
        else:
            # Deterministic selection from scenario bank under effective seed
            if len(self.scenario_bank.scenarios) == 0:
                raise ScenarioValidationError(
                    scenario_id="<none>",
                    reason="Scenario bank is empty",
                )
            # Use numpy RNG for deterministic selection
            idx = self.np_random.integers(0, len(self.scenario_bank.scenarios))
            scenario = self.scenario_bank.scenarios[idx]

        self._current_scenario = scenario

        # Validate scenario against config completely
        self._validate_scenario(scenario)

        # Initialize slot states from scenario via the centralized helper
        slots: List[SlotState] = []

        # Replacement RNG derived deterministically from both the effective
        # reset seed and the scenario's replacement_seed (contract section 14).
        seed_sequence = np.random.SeedSequence(
            [int(env_seed), int(scenario.replacement_seed)]
        )
        replacement_rng = np.random.default_rng(seed_sequence)

        for slot_idx in range(self.N):
            unit_id = scenario.initial_unit_ids[slot_idx]
            initial_cycle = scenario.initial_cycles[slot_idx]

            pred_result = self._require_prediction(
                scenario.split, unit_id, initial_cycle, slot_idx, 0
            )

            # Validate initial true_rul > 0 (also enforced by _validate_scenario)
            if pred_result.true_rul <= 0:
                raise ScenarioValidationError(
                    scenario_id=scenario.scenario_id,
                    reason=f"Initial slot {slot_idx} has true_rul <= 0",
                    details={
                        "unit_id": unit_id,
                        "cycle": initial_cycle,
                        "true_rul": pred_result.true_rul,
                    },
                )

            # Compute initial age: age = cycle - 1
            initial_age = initial_cycle - 1

            slot_state = SlotState(
                slot_index=slot_idx,
                split=scenario.split,
                unit_id=unit_id,
                cycle=initial_cycle,
                trajectory_length=pred_result.trajectory_length,
                age_since_replacement_cycles=initial_age,
                trajectory_id=f"{scenario.split}_{unit_id}",
            )
            slots.append(slot_state)

        # Initialize fleet state
        self._fleet_state = FleetState(
            slots=slots,
            step_index=0,
            episode_return=0.0,
            episode_completed=False,
            replacement_rng=replacement_rng,
        )

        self._reset_called = True

        # Build initial observation
        observation = self._build_observation()

        # Build info
        info = self._build_reset_info()

        return observation, info

    def step(
        self,
        action_id: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one decision window.

        Transition order (12 steps):
        1. Read current active fleet state (already done - observation built)
        2. Decode action ID to selected slots
        3. Mark selected slots for preventive replacement
        4. Compute preventive-maintenance cost
        5. Compute normalized wasted-life cost for selected slots
        6. Replace selected slots immediately
        7. Advance every non-selected slot by up to delta_cycles=5
        8. Detect any failure crossed during that interval
        9. Compute failure cost
        10. Correctively replace every failed slot
        11. Update age and internal bookkeeping
        12. Build next observation, reward, flags, and info

        Args:
            action_id: Action ID from action table

        Returns:
            observation: Next observation
            reward: Step reward (negative cost)
            terminated: Always False (failures don't terminate)
            truncated: True when step >= horizon
            info: Step info dict

        Raises:
            InvalidActionError: If action invalid or step at wrong time
        """
        if not self._reset_called:
            raise InvalidActionError(
                "step() called before reset()",
                action_id=action_id,
            )

        if self._fleet_state is None:
            raise InvalidActionError(
                "step() called with no fleet state",
                action_id=action_id,
            )

        if self._fleet_state.episode_completed:
            raise InvalidActionError(
                "step() called after episode truncation",
                action_id=action_id,
            )

        # Validate action_id
        self._validate_action(action_id)

        # Get current state
        fleet = self._fleet_state
        slots = list(fleet.slots)  # Mutable copy
        step_index = fleet.step_index

        # Decision boundary contract: every slot that is about to be
        # processed must have a current cached record with true_rul > 0.
        # A slot sitting at trajectory_length without a recorded failure is
        # a contract violation, not a free replacement.
        for slot_idx in range(self.N):
            slot = slots[slot_idx]
            pred = self._require_prediction(
                slot.split, slot.unit_id, slot.cycle, slot_idx, step_index
            )
            if pred.true_rul <= 0:
                raise ContractViolationError(
                    f"Decision-boundary slot {slot_idx} already at true_rul<=0 "
                    f"before advancement (unit {slot.unit_id}, cycle {slot.cycle})",
                    context={
                        "slot_index": slot_idx,
                        "unit_id": slot.unit_id,
                        "cycle": slot.cycle,
                        "true_rul": pred.true_rul,
                        "step_index": step_index,
                    },
                )

        # Step 2: Decode action ID to selected slots
        selected_slots: ActionSubset = action_id_to_slots(action_id, self.action_table)
        selected_set = set(selected_slots)

        # Track which slots are preventively maintained
        pm_slots: List[int] = []
        # Track which slots fail during advancement
        failed_slots: List[int] = []

        # Step 4, 5, 6: Process preventive maintenance
        preventive_cost = 0.0
        wasted_rul_sum = 0.0

        for slot_idx in selected_slots:
            slot = slots[slot_idx]

            # Get true RUL for wasted life calculation via the centralized
            # helper. (It was already looked up at the decision boundary
            # above, but use the helper again for a single, explicit path.)
            pred_result = self._require_prediction(
                slot.split, slot.unit_id, slot.cycle, slot_idx, step_index
            )

            # Compute wasted life: clip(true_rul, 0, 125) / 125 (per-engine
            # contribution is in [0, 1]).
            true_rul_capped = min(max(pred_result.true_rul, 0.0), self.rul_scale)
            wasted_life = true_rul_capped / self.rul_scale
            wasted_rul_sum += wasted_life

            # Step 6: Replace slot with new trajectory
            new_slot = self._sample_replacement(
                slot_idx=slot_idx,
                retired_unit_id=slot.unit_id,
                current_split=slot.split,
            )
            slots[slot_idx] = new_slot
            pm_slots.append(slot_idx)

        preventive_cost = self.cost_regime.c_pm * len(pm_slots)

        # Step 7, 8, 9, 10: Advance non-PM slots and detect failures
        failure_cost = 0.0

        for slot_idx in range(self.N):
            if slot_idx in selected_set:
                # Already preventively maintained - does not advance
                continue

            slot = slots[slot_idx]

            # Inspect cycles c+1, c+2, ..., c+5 in order. At the first cached
            # true_rul <= 0: record exactly one failure, charge exactly one
            # failure cost, stop all further lookup for that slot, correctively
            # replace at cycle 1, reset age to 0. Corrective replacement does
            # NOT consume PM capacity and does NOT advance through residual
            # cycles. If trajectory_length is reached or exceeded without a
            # cached true_rul <= 0, raise ContractViolationError. Never perform
            # a free replacement.
            failure_detected = False
            failure_cycle: Optional[int] = None

            for delta in range(1, self.delta_cycles + 1):
                target_cycle = slot.cycle + delta

                # Reaching or exceeding trajectory_length without a cached
                # true_rul <= 0 is a contract violation, NOT a silent replace.
                if target_cycle > slot.trajectory_length:
                    raise ContractViolationError(
                        f"Slot {slot_idx} (unit {slot.unit_id}) reached trajectory "
                        f"length {slot.trajectory_length} without a cached true_rul<=0 "
                        f"record while advancing from cycle {slot.cycle}",
                        context={
                            "slot_index": slot_idx,
                            "unit_id": slot.unit_id,
                            "start_cycle": slot.cycle,
                            "target_cycle": target_cycle,
                            "trajectory_length": slot.trajectory_length,
                        },
                    )

                # Required-prediction lookup via the centralized helper.
                pred_result = self._require_prediction(
                    slot.split, slot.unit_id, target_cycle, slot_idx, step_index
                )

                # Check for contract violation: reached trajectory_length endpoint
                # without a cached failure (true_rul > 0 at the endpoint).
                if target_cycle >= slot.trajectory_length and pred_result.true_rul > 0:
                    raise ContractViolationError(
                        f"Slot {slot_idx} (unit {slot.unit_id}) reached trajectory "
                        f"length {slot.trajectory_length} without a cached true_rul<=0 "
                        f"record at the endpoint (true_rul={pred_result.true_rul})",
                        context={
                            "slot_index": slot_idx,
                            "unit_id": slot.unit_id,
                            "start_cycle": slot.cycle,
                            "target_cycle": target_cycle,
                            "trajectory_length": slot.trajectory_length,
                            "true_rul": pred_result.true_rul,
                        },
                    )

                if pred_result.true_rul <= 0:
                    # Failure detected at this cycle.
                    failure_detected = True
                    failure_cycle = target_cycle
                    break

            if failure_detected:
                # Record exactly one failure and charge exactly one failure cost.
                failed_slots.append(slot_idx)
                failure_cost += self.cost_regime.c_f

                # Step 10: Corrective replacement at cycle 1, age 0. Does not
                # consume K capacity; no residual-cycle advancement.
                new_slot = self._sample_replacement(
                    slot_idx=slot_idx,
                    retired_unit_id=slot.unit_id,
                    current_split=slot.split,
                )
                slots[slot_idx] = new_slot
            else:
                # No failure within c+1..c+5: advance the slot by exactly
                # delta_cycles. All of those cycles were just verified to have
                # true_rul > 0, so the cached record at c+delta_cycles exists.
                new_cycle = slot.cycle + self.delta_cycles
                pred_result = self._require_prediction(
                    slot.split, slot.unit_id, new_cycle, slot_idx, step_index
                )
                new_age = slot.age_since_replacement_cycles + self.delta_cycles
                slots[slot_idx] = replace(
                    slot,
                    cycle=new_cycle,
                    age_since_replacement_cycles=new_age,
                )

        # Step 12: Compute total cost and reward
        wasted_life_cost = self.cost_regime.c_u * wasted_rul_sum
        total_cost = preventive_cost + failure_cost + wasted_life_cost
        reward = -total_cost

        # Update fleet state
        fleet.slots = slots
        fleet.step_index = step_index + 1
        fleet.episode_return += reward

        # Check truncation
        truncated = fleet.step_index >= self.horizon
        terminated = False  # Failures don't terminate in Milestone 2

        if truncated:
            fleet.episode_completed = True

        # Build next observation
        observation = self._build_observation()

        # Build info
        info = self._build_step_info(
            step_index=fleet.step_index,
            action_id=action_id,
            selected_slots=list(selected_slots),
            num_preventive=len(pm_slots),
            num_failures=len(failed_slots),
            preventive_cost=preventive_cost,
            failure_cost=failure_cost,
            wasted_life_cost=wasted_life_cost,
            total_cost=total_cost,
            reward=reward,
            truncated=truncated,
        )

        return observation, reward, terminated, truncated, info

    def get_action_mask(self) -> np.ndarray:
        """
        Return boolean array of valid actions.

        For direct-subset action table, all actions are valid by construction.

        Returns:
            Boolean array of shape (num_actions,) with all True values.
        """
        return np.ones(self.num_actions, dtype=bool)

    def close(self) -> None:
        """Cleanup resources (no-op for this environment)."""
        self._fleet_state = None
        self._reset_called = False

    def _validate_action(self, action_id: Any) -> None:
        """Validate action ID."""
        # Check integral
        if not isinstance(action_id, (int, np.integer)):
            raise InvalidActionError(
                f"action_id must be integral, got {type(action_id).__name__}",
                action_id=action_id,
            )

        # Check boolean
        if isinstance(action_id, bool):
            raise InvalidActionError(
                "action_id cannot be boolean",
                action_id=action_id,
            )

        # Check range
        if action_id < 0 or action_id >= self.num_actions:
            raise InvalidActionError(
                f"action_id {action_id} out of range [0, {self.num_actions - 1}]",
                action_id=action_id,
            )

    def _validate_scenario(self, scenario: Scenario) -> None:
        """
        Validate the selected scenario against the config completely.

        Enforces:
        - scenario.split == config.split
        - scenario.maintenance_capacity == config.maintenance_capacity
        - scenario.episode_horizon == config.episode_horizon
        - scenario.cost_regime_id == config.cost_regime_id
        - len(initial_unit_ids) == config.fleet_size
        - len(initial_cycles) == config.fleet_size
        - every initial unit belongs to the configured split
        - every initial record exists in the PredictionStore
        - every initial true_rul > 0

        A cost-regime mismatch raises ScenarioValidationError so a scenario
        is never silently run under another regime's cost coefficients.
        """
        # Check split
        if scenario.split != self.config.split:
            raise ScenarioValidationError(
                scenario_id=scenario.scenario_id,
                reason=f"Scenario split '{scenario.split}' does not match "
                f"config split '{self.config.split}'",
            )

        # Check K matches
        if scenario.maintenance_capacity != self.K:
            raise ScenarioValidationError(
                scenario_id=scenario.scenario_id,
                reason=f"Scenario K={scenario.maintenance_capacity} does not match "
                f"config K={self.K}",
            )

        # Check horizon matches
        if scenario.episode_horizon != self.horizon:
            raise ScenarioValidationError(
                scenario_id=scenario.scenario_id,
                reason=f"Scenario horizon={scenario.episode_horizon} does not match "
                f"config horizon={self.horizon}",
            )

        # Check cost regime matches
        if scenario.cost_regime_id != self.config.cost_regime_id:
            raise ScenarioValidationError(
                scenario_id=scenario.scenario_id,
                reason=f"Scenario cost_regime_id='{scenario.cost_regime_id}' "
                f"does not match config cost_regime_id="
                f"'{self.config.cost_regime_id}'",
            )

        # Check fleet size for units
        if len(scenario.initial_unit_ids) != self.N:
            raise ScenarioValidationError(
                scenario_id=scenario.scenario_id,
                reason=f"Scenario has {len(scenario.initial_unit_ids)} units, "
                f"expected {self.N}",
            )

        # Check fleet size for cycles
        if len(scenario.initial_cycles) != self.N:
            raise ScenarioValidationError(
                scenario_id=scenario.scenario_id,
                reason=f"Scenario has {len(scenario.initial_cycles)} cycles, "
                f"expected {self.N}",
            )

        # Every initial unit belongs to the configured split, every initial
        # record exists, and every initial true_rul > 0.
        valid_units = set(self.prediction_store.get_units(scenario.split))
        for slot_idx, (unit_id, cycle) in enumerate(
            zip(scenario.initial_unit_ids, scenario.initial_cycles)
        ):
            if unit_id not in valid_units:
                raise ScenarioValidationError(
                    scenario_id=scenario.scenario_id,
                    reason=f"Initial unit {unit_id} for slot {slot_idx} is not in "
                    f"split '{scenario.split}'",
                )
            # Use _require_prediction to allow MissingPredictionError to propagate
            # with full context (split, unit_id, cycle, slot_index, env_step=0)
            # instead of translating to ScenarioValidationError.
            pred_result = self._require_prediction(
                scenario.split, unit_id, cycle, slot_idx, 0
            )
            if pred_result.true_rul is None or pred_result.true_rul <= 0:
                raise ScenarioValidationError(
                    scenario_id=scenario.scenario_id,
                    reason=f"Initial slot {slot_idx} has true_rul<=0 "
                    f"(unit {unit_id}, cycle {cycle})",
                    details={"true_rul": pred_result.true_rul},
                )

    def _sample_replacement(
        self,
        slot_idx: int,
        retired_unit_id: int,
        current_split: str,
    ) -> SlotState:
        """
        Sample a replacement trajectory for a slot.

        - Samples from current split's trajectory library (no cross-split)
        - Starts at cycle 1
        - Resets age to 0
        - Preserves slot index
        - Anti-repeat: does not immediately assign retired unit back to same
          slot when another trajectory is available
        - Single-unit fallback: if the split has only one unit, the retired
          unit is unavoidably reselected

        The replacement RNG is derived deterministically from both the
        effective reset seed and the scenario replacement_seed at reset(),
        so identical (scenario, reset seed, action sequence) reproduces
        identical replacement unit identities and transitions.

        Args:
            slot_idx: Fleet slot index
            retired_unit_id: Unit ID being replaced
            current_split: Current environment split

        Returns:
            New SlotState for the replacement

        Raises:
            MissingPredictionError: if the cycle-1 prediction for the
                sampled replacement unit is absent.
        """
        if self._fleet_state is None:
            raise RuntimeError("Cannot sample replacement before reset")

        # Get available units for this split (split-specific sampling)
        available_units = self.prediction_store.get_units(current_split)

        # Apply anti-repeat rule
        if len(available_units) > 1:
            # Remove retired unit from candidates
            candidates = [u for u in available_units if u != retired_unit_id]
        else:
            # Single unit pool - unavoidable fallback
            candidates = available_units

        # Sample replacement unit deterministically from the replacement RNG.
        idx = self._fleet_state.replacement_rng.integers(0, len(candidates))
        new_unit_id = candidates[idx]

        # Required cycle-1 prediction via the centralized helper.
        env_step = self._fleet_state.step_index if self._fleet_state else 0
        pred_result = self._require_prediction(
            current_split, new_unit_id, 1, slot_idx, env_step
        )

        # Create fresh slot state (cycle 1, age 0)
        return SlotState.create_fresh(
            slot_index=slot_idx,
            split=current_split,
            unit_id=new_unit_id,
            trajectory_length=pred_result.trajectory_length,
            trajectory_id=f"{current_split}_{new_unit_id}",
        )

    def _build_observation(self) -> np.ndarray:
        """
        Build observation from current fleet state.

        Observation features per slot:
        - normalized_age_since_replacement: clip(age / 341, 0, 1)
        - normalized_predicted_rul: clip(predicted_rul / 125, 0, 1)

        Returns:
            Flat np.float32 array of shape (10,)
        """
        if self._fleet_state is None:
            raise RuntimeError("Cannot build observation before reset")

        features: List[float] = []

        for slot in self._fleet_state.slots:
            pred_result = self._require_prediction(
                slot.split,
                slot.unit_id,
                slot.cycle,
                slot.slot_index,
                self._fleet_state.step_index,
            )

            # Normalize age: clip(age / 341, 0, 1)
            age_normalized = slot.age_since_replacement_cycles / self.age_scale
            age_normalized = min(max(age_normalized, 0.0), 1.0)

            # Normalize predicted RUL: clip(predicted_rul / 125, 0, 1)
            pred_rul_normalized = pred_result.predicted_rul / self.rul_scale
            pred_rul_normalized = min(max(pred_rul_normalized, 0.0), 1.0)

            # Validate finite
            if not np.isfinite(age_normalized) or not np.isfinite(pred_rul_normalized):
                raise InformationLeakageError(
                    f"Non-finite observation feature",
                    field="age_normalized or pred_rul_normalized",
                )

            features.extend([age_normalized, pred_rul_normalized])

        obs_array = np.array(features, dtype=np.float32)

        # Ensure shape is (10,)
        assert obs_array.shape == (self.N * 2,), \
            f"Observation shape mismatch: expected {(self.N * 2,)}, got {obs_array.shape}"

        return obs_array

    def _build_reset_info(self) -> Dict[str, Any]:
        """Build info dict for reset."""
        if self._fleet_state is None:
            raise RuntimeError("Cannot build reset info before reset")

        info: Dict[str, Any] = {}

        if self.info_mode == "diagnostic":
            info["scenario_id"] = self._current_scenario.scenario_id if self._current_scenario else None
            for slot in self._fleet_state.slots:
                # Get true RUL from prediction store for diagnostic mode
                pred_result = self.prediction_store.get(
                    slot.split, slot.unit_id, slot.cycle
                )
                info[f"slot_{slot.slot_index}_diagnostic"] = {
                    "unit_id": slot.unit_id,
                    "cycle": slot.cycle,
                    "age_since_replacement_cycles": slot.age_since_replacement_cycles,
                    "true_rul": pred_result.true_rul if pred_result.found else None,
                    "trajectory_length": slot.trajectory_length,
                }

        return info

    def _build_step_info(
        self,
        step_index: int,
        action_id: int,
        selected_slots: List[int],
        num_preventive: int,
        num_failures: int,
        preventive_cost: float,
        failure_cost: float,
        wasted_life_cost: float,
        total_cost: float,
        reward: float,
        truncated: bool,
    ) -> Dict[str, Any]:
        """Build info dict for step."""
        info: Dict[str, Any] = {
            "step_index": step_index,
            "action_id": action_id,
            "selected_slots": selected_slots,
            "num_preventive": num_preventive,
            "num_failures": num_failures,
            "preventive_cost": preventive_cost,
            "failure_cost": failure_cost,
            "wasted_life_cost": wasted_life_cost,
            "total_cost": total_cost,
            "reward": reward,
            "truncated": truncated,
        }

        if self.info_mode == "diagnostic":
            if self._fleet_state is not None:
                for slot in self._fleet_state.slots:
                    pred_result = self.prediction_store.get(
                        slot.split, slot.unit_id, slot.cycle
                    )
                    info[f"slot_{slot.slot_index}_diagnostic"] = {
                        "unit_id": slot.unit_id,
                        "cycle": slot.cycle,
                        "age_since_replacement_cycles": slot.age_since_replacement_cycles,
                        "true_rul": pred_result.true_rul if pred_result.found else None,
                        "trajectory_length": slot.trajectory_length,
                    }

        return info
