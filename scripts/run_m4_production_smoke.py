#!/usr/bin/env python3
"""
Milestone 4 Production Smoke Matrix Runner.

TWO LANE STRUCTURE (Step 2 Fix):

LANE 1 - PRIMARY CONTRACT POLICY (default):
  - risk_model_id = "hard_window_v1" (frozen)
  - matrix_role = "primary_contract_policy"
  - Output: results/milestone4/m4_repair_primary_hard/
  - 16 configs, 80 episodes, 8000 steps
  - This is the official M4 scientific evaluation, even if action 0 is always selected

LANE 2 - ENGINEERING BEHAVIOR COVERAGE (separate):
  - risk_model_id = "logistic_window_v1" (fixed temperature)
  - matrix_role = "engineering_behavior_coverage"
  - Output: results/milestone4/m4_behavior_coverage/
  - NOT a scientific result, NOT a policy comparison
  -Purpose: exercise preventive-maintenance accounting path

The two lanes must have separate:
- resolved configs, config hashes, manifests, metrics, reports
- output directories
- artifact files

Do not merge the two policies into one scientific aggregate.

Runs real production environment rollouts for all 16 configurations:
- 2 splits (predictor_train, rl_validation)
- 2 K values (1, 2)
- 4 cost regimes

Each configuration runs actual episodes with:
- Production PredictionStore
- Real scenario bank
- SelectiveMaintenanceEnv
- ExactMyopic optimizer called on each observation
- Full episode through horizon truncation
- Actual environment reward and cost components recorded
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from optimizers import (
    MyopicContext,
    ExactMyopicOptimizer,
    get_git_commit,
    write_atomic_json,
    compute_file_hash,
    compute_data_hash,
    convert_for_json,
    build_complete_scientific_config,
    compute_complete_config_hash,
)
from optimizers.myopic_provenance import compute_action_table_content_hash
from envs.action_table import ACTION_TABLE_N5_K1, ACTION_TABLE_N5_K2
from envs.costs import COST_REGIMES, get_cost_regime
from envs.config import get_default_config, EnvironmentConfig
from envs.scenario_bank import load_scenario_bank, ScenarioBank
from envs.selective_maintenance_env import SelectiveMaintenanceEnv

# Import prediction store
from predictors.prediction_store import PredictionStore


# Primary policy configuration (frozen contract policy)
# STEP 2 FIX: The primary production matrix MUST use the frozen hard_window_v1
# policy, even if it selects action 0 for every observed state.
# This is an honest evaluation of the contract policy.
PRIMARY_RISK_MODEL_ID = "hard_window_v1"
PRIMARY_RISK_TEMPERATURE = None  # Not used for hard_window

# Engineering behavior-coverage configuration (separate lane)
# The preventive-maintenance accounting path needs real production coverage.
# This separate lane uses logistic_window_v1 ONLY for branch coverage,
# NOT as a replacement for the primary policy.
ENGINEERING_COVERAGE_RISK_MODEL_ID = "logistic_window_v1"
ENGINEERING_COVERAGE_RISK_TEMPERATURE = 10.0  # Fixed documented temperature

# Import engineering coverage threshold from single authoritative source
from optimizers.m4_constants import ENGINEERING_COVERAGE_THRESHOLD_CYCLES


@dataclass
class ProductionRunConfig:
    """Configuration for a single production run."""
    split: str
    k_capacity: int
    cost_regime_id: str
    risk_model_id: str
    scenario_bank_path: Path
    prediction_cache_path: Path
    episode_horizon: int = 100
    seeds: Tuple[int, ...] = (6521, 6522, 6523, 6524, 6525)


@dataclass
class EpisodeResult:
    """Results from a single episode."""
    scenario_id: str
    episode_return: float
    total_steps: int
    terminated: bool
    truncated: bool
    preventive_replacements: int
    failures: int
    preventive_cost: float
    failure_cost: float
    unused_life_cost: float
    total_cost: float
    nan_inf_count: int
    missing_prediction_count: int
    split_violation_count: int
    action_ids: List[int] = field(default_factory=list)
    estimated_myopic_costs: List[float] = field(default_factory=list)
    failure_reasons: List[str] = field(default_factory=list)  # BLOCKER 4: Explicit failure reasons


@dataclass
class ConfigResult:
    """Results for a single configuration."""
    split: str
    k_capacity: int
    cost_regime_id: str
    risk_model_id: str
    episode_count: int
    total_steps: int
    action_ids: List[int]
    preventive_replacements: int
    failures: int
    preventive_cost: float
    failure_cost: float
    unused_life_cost: float
    total_environment_cost: float
    estimated_myopic_cost: float
    terminated_count: int
    truncated_count: int
    nan_inf_count: int
    missing_prediction_count: int
    split_violation_count: int
    episode_results: List[EpisodeResult] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None
    failure_reasons: List[str] = field(default_factory=list)  # BLOCKER 4: Explicit failure reasons


def load_prediction_store_for_split(split: str, cache_dir: Path) -> PredictionStore:
    """Load prediction store for a specific split."""
    cache_path = cache_dir / "fd001_prediction_cache_v2.parquet"
    manifest_path = cache_dir / "milestone_1_artifact_manifest_v2.json"

    return PredictionStore(
        cache_path=cache_path,
        manifest_path=manifest_path,
        allow_invalidated=False,
    )


def create_optimizer(
    k_capacity: int,
    cost_regime_id: str,
    risk_model_id: str = None,
    risk_temperature: float = None,
    matrix_role: str = "primary_contract_policy",
) -> ExactMyopicOptimizer:
    """
    Create optimizer with given parameters.

    Args:
        matrix_role: "primary_contract_policy" uses hard_window_v1 (frozen).
                     "engineering_behavior_coverage" uses logistic_window_v1.
    """
    # STEP 2 FIX: Primary policy uses hard_window_v1 (frozen contract policy)
    # Engineering coverage lane uses logistic_window_v1 (for branch coverage only)
    if risk_model_id is None:
        if matrix_role == "engineering_behavior_coverage":
            risk_model_id = ENGINEERING_COVERAGE_RISK_MODEL_ID
            risk_temperature = ENGINEERING_COVERAGE_RISK_TEMPERATURE
        else:
            # Default: primary contract policy
            risk_model_id = PRIMARY_RISK_MODEL_ID
            risk_temperature = PRIMARY_RISK_TEMPERATURE
    if risk_temperature is None:
        if risk_model_id == "logistic_window_v1":
            risk_temperature = ENGINEERING_COVERAGE_RISK_TEMPERATURE
        else:
            risk_temperature = None  # hard_window doesn't use temperature

    cost_regime = get_cost_regime(cost_regime_id)
    action_table = ACTION_TABLE_N5_K1 if k_capacity == 1 else ACTION_TABLE_N5_K2

    context = MyopicContext(
        maintenance_capacity=k_capacity,
        delta_cycles=5,
        rul_scale=125.0,
        age_scale_cycles=341,
        action_table=action_table,
        c_pm=cost_regime.c_pm,
        c_f=cost_regime.c_f,
        c_u=cost_regime.c_u,
        risk_model_id=risk_model_id,
    )

    return ExactMyopicOptimizer(
        context=context,
        risk_temperature=risk_temperature,
    )


EPISODE_HORIZON = 100  # Expected episode horizon


def validate_episode_result(
    result: EpisodeResult,
    expected_horizon: int = EPISODE_HORIZON,
) -> List[str]:
    """
    BLOCKER 4: Strict pass criteria validation for a single episode.

    Returns list of failure reasons (empty if all pass).
    """
    failure_reasons = []

    # Check terminated is False
    if result.terminated:
        failure_reasons.append("terminated=True (expected False)")

    # Check truncated is True
    if not result.truncated:
        failure_reasons.append("truncated=False (expected True)")

    # Check exactly episode_horizon steps
    if result.total_steps != expected_horizon:
        failure_reasons.append(
            f"total_steps={result.total_steps} (expected {expected_horizon})"
        )

    # Check all observations, rewards, costs are finite
    if result.nan_inf_count > 0:
        failure_reasons.append(f"nan_inf_count={result.nan_inf_count} (expected 0)")

    # Check missing_prediction_count == 0
    if result.missing_prediction_count > 0:
        failure_reasons.append(
            f"missing_prediction_count={result.missing_prediction_count} (expected 0)"
        )

    # Check split_violation_count == 0
    if result.split_violation_count > 0:
        failure_reasons.append(
            f"split_violation_count={result.split_violation_count} (expected 0)"
        )

    # Check cumulative reward equals negative cumulative cost
    if not np.isclose(result.episode_return, -result.total_cost, rtol=1e-9):
        failure_reasons.append(
            f"episode_return ({result.episode_return}) != -total_cost ({result.total_cost})"
        )

    # Check total_cost equals sum of components
    computed_total = result.preventive_cost + result.failure_cost + result.unused_life_cost
    if not np.isclose(result.total_cost, computed_total, rtol=1e-9):
        failure_reasons.append(
            f"total_cost ({result.total_cost}) != sum of components ({computed_total})"
        )

    return failure_reasons


def validate_config_result(result: ConfigResult, expected_scenarios: int) -> List[str]:
    """
    BLOCKER 4: Strict pass criteria validation for a configuration.

    Returns list of failure reasons (empty if all pass).
    """
    failure_reasons = []

    # Check expected episode count was executed
    if result.episode_count != expected_scenarios:
        failure_reasons.append(
            f"episode_count={result.episode_count} (expected {expected_scenarios})"
        )

    # Check every requested scenario executed exactly once
    scenario_ids = [ep.scenario_id for ep in result.episode_results]
    unique_scenarios = set(scenario_ids)
    if len(unique_scenarios) != len(scenario_ids):
        duplicates = [
            sid for sid in scenario_ids
            if scenario_ids.count(sid) > 1
        ]
        failure_reasons.append(f"Duplicate scenarios: {set(duplicates)}")

    # Aggregate failure reasons from all episodes
    for i, ep in enumerate(result.episode_results):
        ep_failures = validate_episode_result(ep)
        for reason in ep_failures:
            failure_reasons.append(f"Episode {i} ({ep.scenario_id}): {reason}")

    return failure_reasons


def run_episode(
    env: SelectiveMaintenanceEnv,
    optimizer: ExactMyopicOptimizer,
    scenario_id: str,
) -> EpisodeResult:
    """
    Run a single episode with the optimizer in the loop.

    Records actual environment rewards and cost components.
    """
    # BLOCKER 1 FIX: Explicitly pass scenario_id through options
    obs, info = env.reset(options={"scenario_id": scenario_id})

    # Verify reset info identifies the requested scenario
    if "scenario_id" in info:
        assert info["scenario_id"] == scenario_id, (
            f"Scenario mismatch: requested {scenario_id}, "
            f"got {info['scenario_id']}"
        )

    total_steps = 0
    terminated = False
    truncated = False

    action_ids: List[int] = []
    estimated_costs: List[float] = []

    # BLOCKER 3 FIX: Explicitly accumulate episode_return
    episode_return = 0.0
    episode_total_cost = 0.0

    preventive_replacements = 0
    failures = 0
    preventive_cost = 0.0
    failure_cost = 0.0
    wasted_life_cost = 0.0

    nan_inf_count = 0
    missing_prediction_count = 0
    split_violation_count = 0

    while not (terminated or truncated):
        # Get action from optimizer using only public observation
        action_id, selected_slots, estimated_cost = optimizer.select_action(obs)

        # Validate optimizer output
        if not np.isfinite(estimated_cost):
            nan_inf_count += 1

        action_ids.append(action_id)
        estimated_costs.append(float(estimated_cost))

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action_id)

        # BLOCKER 3 FIX: Explicitly accumulate episode_return
        episode_return += reward

        # Check for NaN/Inf in observation
        if not np.all(np.isfinite(obs)):
            nan_inf_count += 1

        # BLOCKER 2 FIX: Use actual top-level M2 info schema fields
        # Fields: num_preventive, num_failures, preventive_cost,
        #         failure_cost, wasted_life_cost, total_cost, reward, truncated
        num_preventive = info.get("num_preventive", 0)
        num_failures = info.get("num_failures", 0)
        step_preventive_cost = info.get("preventive_cost", 0.0)
        step_failure_cost = info.get("failure_cost", 0.0)
        step_wasted_life_cost = info.get("wasted_life_cost", 0.0)
        step_total_cost = info.get("total_cost", 0.0)

        # Accumulate cost components
        preventive_replacements += num_preventive
        failures += num_failures
        preventive_cost += step_preventive_cost
        failure_cost += step_failure_cost
        wasted_life_cost += step_wasted_life_cost

        # Verify reward == -total_cost at each step
        step_reward = info.get("reward", reward)
        assert np.isclose(-step_reward, step_total_cost, rtol=1e-9), (
            f"Reward/cost mismatch: reward={step_reward}, "
            f"total_cost={step_total_cost}"
        )

        # Verify total_cost == sum of components
        computed_total = step_preventive_cost + step_failure_cost + step_wasted_life_cost
        assert np.isclose(step_total_cost, computed_total, rtol=1e-9), (
            f"Cost component mismatch: total={step_total_cost}, "
            f"sum={computed_total}"
        )

        # Accumulate total cost
        episode_total_cost += step_total_cost

        # Track missing predictions
        if "missing_predictions" in info:
            missing_prediction_count += info.get("missing_predictions", 0)

        # Track split violations
        if "split_violations" in info:
            split_violation_count += info.get("split_violations", 0)

        total_steps += 1

    # Final cost computation from accumulated components
    total_cost = preventive_cost + failure_cost + wasted_life_cost

    # BLOCKER 3 FIX: Verify cumulative reward equals negative cumulative cost
    assert np.isclose(episode_return, -episode_total_cost, rtol=1e-9), (
        f"Episode return/cost mismatch: return={episode_return}, "
        f"total_cost={episode_total_cost}"
    )

    # Also verify against accumulated component sum
    assert np.isclose(episode_return, -total_cost, rtol=1e-9), (
        f"Episode return/total_cost mismatch: return={episode_return}, "
        f"total_cost={total_cost}"
    )

    return EpisodeResult(
        scenario_id=scenario_id,
        episode_return=episode_return,
        total_steps=total_steps,
        terminated=terminated,
        truncated=truncated,
        preventive_replacements=preventive_replacements,
        failures=failures,
        preventive_cost=preventive_cost,
        failure_cost=failure_cost,
        unused_life_cost=wasted_life_cost,  # Keep field name for backward compatibility
        total_cost=total_cost,
        nan_inf_count=nan_inf_count,
        missing_prediction_count=missing_prediction_count,
        split_violation_count=split_violation_count,
        action_ids=action_ids,
        estimated_myopic_costs=estimated_costs,
    )


def run_production_config(
    run_config: ProductionRunConfig,
    prediction_store: PredictionStore,
) -> ConfigResult:
    """Run all episodes for a single production configuration."""
    # Load scenario bank
    scenario_bank = load_scenario_bank(run_config.scenario_bank_path)

    # Create optimizer
    optimizer = create_optimizer(
        k_capacity=run_config.k_capacity,
        cost_regime_id=run_config.cost_regime_id,
        risk_model_id=run_config.risk_model_id,
    )

    episode_results: List[EpisodeResult] = []
    all_action_ids: List[int] = []
    total_estimated_cost = 0.0

    total_preventive_replacements = 0
    total_failures = 0
    total_preventive_cost = 0.0
    total_failure_cost = 0.0
    total_unused_life_cost = 0.0
    total_environment_cost = 0.0

    total_nan_inf = 0
    total_missing = 0
    total_split_violations = 0
    terminated_count = 0
    truncated_count = 0
    total_steps = 0

    success = True
    error = None

    env = None
    try:
        # Create config with correct split
        from envs.config import EnvironmentConfig

        config = EnvironmentConfig(
            environment_version="m2_v1",
            split=run_config.split,
            fleet_size=5,
            maintenance_capacity=run_config.k_capacity,
            delta_cycles=5,
            episode_horizon=run_config.episode_horizon,
            age_scale_cycles=341,
            rul_scale=125.0,
            cost_regime_id=run_config.cost_regime_id,
            scenario_bank_path=str(run_config.scenario_bank_path),
            prediction_cache_path=str(run_config.prediction_cache_path),
            info_mode="normal",
            seed=run_config.seeds[0] if run_config.seeds else 6521,
        )

        # Create environment
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=scenario_bank,
            info_mode="normal",
        )

        # BLOCKER 1 FIX: Track executed scenarios to ensure each is executed exactly once
        executed_scenarios = set()
        expected_scenario_ids = {s.scenario_id for s in scenario_bank.scenarios}

        # Run each scenario as an episode
        for scenario in scenario_bank.scenarios:
            result = run_episode(env, optimizer, scenario.scenario_id)
            episode_results.append(result)

            # BLOCKER 1: Verify scenario was executed and recorded correctly
            assert result.scenario_id == scenario.scenario_id, (
                f"Scenario ID mismatch: expected {scenario.scenario_id}, "
                f"got {result.scenario_id}"
            )
            executed_scenarios.add(result.scenario_id)

            # Aggregate
            all_action_ids.extend(result.action_ids)
            total_estimated_cost += sum(result.estimated_myopic_costs)

            total_preventive_replacements += result.preventive_replacements
            total_failures += result.failures
            total_preventive_cost += result.preventive_cost
            total_failure_cost += result.failure_cost
            total_unused_life_cost += result.unused_life_cost
            total_environment_cost += result.total_cost

            total_nan_inf += result.nan_inf_count
            total_missing += result.missing_prediction_count
            total_split_violations += result.split_violation_count

            if result.terminated:
                terminated_count += 1
            if result.truncated:
                truncated_count += 1

            total_steps += result.total_steps

            # BLOCKER 4: Validate each episode result with strict pass criteria
            ep_failure_reasons = validate_episode_result(result)
            if ep_failure_reasons:
                result.failure_reasons = ep_failure_reasons

        # BLOCKER 1: Verify every scenario executed exactly once
        assert executed_scenarios == expected_scenario_ids, (
            f"Scenario execution mismatch: "
            f"expected {expected_scenario_ids}, got {executed_scenarios}"
        )

    except Exception as e:
        success = False
        error = str(e)
        import traceback
        traceback.print_exc()

    finally:
        # BLOCKER 7: Close environment in finally block
        if env is not None:
            env.close()

    # BLOCKER 4: Validate config result with strict pass criteria
    expected_scenarios = len(scenario_bank.scenarios) if scenario_bank else 0
    config_failure_reasons = validate_config_result(
        ConfigResult(
            split=run_config.split,
            k_capacity=run_config.k_capacity,
            cost_regime_id=run_config.cost_regime_id,
            risk_model_id=run_config.risk_model_id,
            episode_count=len(episode_results),
            total_steps=total_steps,
            action_ids=sorted(set(all_action_ids)),
            preventive_replacements=total_preventive_replacements,
            failures=total_failures,
            preventive_cost=total_preventive_cost,
            failure_cost=total_failure_cost,
            unused_life_cost=total_unused_life_cost,
            total_environment_cost=total_environment_cost,
            estimated_myopic_cost=total_estimated_cost,
            terminated_count=terminated_count,
            truncated_count=truncated_count,
            nan_inf_count=total_nan_inf,
            missing_prediction_count=total_missing,
            split_violation_count=total_split_violations,
            episode_results=episode_results,
            success=success,
            error=error,
        ),
        expected_scenarios=expected_scenarios,
    )

    # Determine final success based on validation
    if config_failure_reasons:
        success = False
        if error:
            error = f"{error}; Validation failures: {config_failure_reasons}"
        else:
            error = f"Validation failures: {config_failure_reasons}"

    # BLOCKER 7: Use deterministic action ID serialization (sorted set)
    return ConfigResult(
        split=run_config.split,
        k_capacity=run_config.k_capacity,
        cost_regime_id=run_config.cost_regime_id,
        risk_model_id=run_config.risk_model_id,
        episode_count=len(episode_results),
        total_steps=total_steps,
        action_ids=sorted(set(all_action_ids)),  # Already sorted, but explicit
        preventive_replacements=total_preventive_replacements,
        failures=total_failures,
        preventive_cost=total_preventive_cost,
        failure_cost=total_failure_cost,
        unused_life_cost=total_unused_life_cost,
        total_environment_cost=total_environment_cost,
        estimated_myopic_cost=total_estimated_cost,
        terminated_count=terminated_count,
        truncated_count=truncated_count,
        nan_inf_count=total_nan_inf,
        missing_prediction_count=total_missing,
        split_violation_count=total_split_violations,
        episode_results=episode_results,
        success=success,
        error=error,
        failure_reasons=config_failure_reasons,  # BLOCKER 4: Explicit failure reasons
    )


def config_result_to_dict(result: ConfigResult) -> Dict[str, Any]:
    """Convert ConfigResult to JSON-serializable dict."""
    return {
        "split": result.split,
        "k_capacity": result.k_capacity,
        "cost_regime_id": result.cost_regime_id,
        "risk_model_id": result.risk_model_id,
        "episode_count": result.episode_count,
        "total_steps": result.total_steps,
        "action_ids": result.action_ids,
        "preventive_replacements": result.preventive_replacements,
        "failures": result.failures,
        "preventive_cost": result.preventive_cost,
        "failure_cost": result.failure_cost,
        "unused_life_cost": result.unused_life_cost,
        "total_environment_cost": result.total_environment_cost,
        "estimated_myopic_cost": result.estimated_myopic_cost,
        "terminated_count": result.terminated_count,
        "truncated_count": result.truncated_count,
        "nan_inf_count": result.nan_inf_count,
        "missing_prediction_count": result.missing_prediction_count,
        "split_violation_count": result.split_violation_count,
        "success": result.success,
        "error": result.error,
        "failure_reasons": result.failure_reasons,  # BLOCKER 4: Explicit failure reasons
    }


def episode_result_to_dict(ep: EpisodeResult) -> Dict[str, Any]:
    """Convert EpisodeResult to JSON-serializable dict."""
    return {
        "scenario_id": ep.scenario_id,
        "episode_return": ep.episode_return,
        "total_steps": ep.total_steps,
        "terminated": ep.terminated,
        "truncated": ep.truncated,
        "preventive_replacements": ep.preventive_replacements,
        "failures": ep.failures,
        "preventive_cost": ep.preventive_cost,
        "failure_cost": ep.failure_cost,
        "unused_life_cost": ep.unused_life_cost,
        "total_cost": ep.total_cost,
        "nan_inf_count": ep.nan_inf_count,
        "missing_prediction_count": ep.missing_prediction_count,
        "split_violation_count": ep.split_violation_count,
        "action_ids": ep.action_ids,
        "estimated_myopic_costs": ep.estimated_myopic_costs,
        "failure_reasons": ep.failure_reasons,  # BLOCKER 4: Explicit failure reasons
    }


def compute_scenario_bank_hash(scenario_bank_path: Path) -> str:
    """Compute SHA256 hash of a scenario bank file."""
    return compute_file_hash(scenario_bank_path)


def extract_scenario_seeds_from_bank(scenario_bank_path: Path) -> list[int]:
    """
    Extract scenario seeds from an actual scenario bank file.

    Args:
        scenario_bank_path: Path to scenario bank JSON file.

    Returns:
        List of scenario seeds (replacement_seed values) from the bank,
        sorted in scenario_id order for determinism.

    Raises:
        FileNotFoundError: If bank file does not exist.
        ValueError: If bank is malformed.
    """
    from envs.scenario_bank import load_scenario_bank

    if not scenario_bank_path.exists():
        raise FileNotFoundError(f"Scenario bank not found: {scenario_bank_path}")

    bank = load_scenario_bank(scenario_bank_path)
    # Extract seeds in scenario order (scenarios are already ordered in the bank)
    return [s.replacement_seed for s in bank.scenarios]


def write_production_artifacts(
    output_dir: Path,
    all_results: List[ConfigResult],
    config: Dict[str, Any],
    git_commit: str,
    repo_root: Path,
    prediction_cache_path: Path,
    scenario_bank_paths: Dict[str, Path],
    overwrite: bool = False,
    matrix_role: str = "primary_contract_policy",
) -> Dict[str, Path]:
    """
    Write all required production artifacts.

    Required artifacts:
    - resolved_config.json
    - run_manifest.json
    - action_cost_summary.json
    - episode_metrics.json
    - aggregate_metrics.json
    - smoke_report.json

    STEP 2 FIX: Two separate lanes - primary contract policy and engineering coverage.
    Each lane has separate artifacts, config hashes, and output directories.
    Do not mix the two policies into one scientific aggregate.

    Args:
        output_dir: Output directory for artifacts (must be inside repo_root)
        all_results: Results from all configurations
        config: Base configuration
        git_commit: Git commit hash
        repo_root: Repository root path
        prediction_cache_path: Prediction cache path
        scenario_bank_paths: Map of scenario bank paths
        overwrite: If False, raise FileExistsError if artifacts exist
        matrix_role: "primary_contract_policy" or "engineering_behavior_coverage"

    Returns:
        Dict mapping artifact name to written path.

    Raises:
        ValueError: If output_dir is outside repository
        FileExistsError: If artifacts exist and overwrite=False
    """
    output_dir = Path(output_dir).resolve()
    repo_root = Path(repo_root).resolve()

    # STEP 4 FIX: Reinstate strict repository-local outputs.
    # Production output directories MUST be inside the repository.
    # Reject external output paths BEFORE any artifact writing.
    try:
        output_dir.relative_to(repo_root)
        output_inside_repo = True
    except ValueError:
        # Output directory is outside repository - reject before any writing
        raise ValueError(
            f"Output directory is outside the repository.\n"
            f"  Output dir: {output_dir}\n"
            f"  Repo root: {repo_root}\n"
            f"Production artifacts must be written to repository-relative paths only.\n"
            f"Use an output directory inside: {repo_root}/results/milestone4/"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths = {}

    # BLOCKER 5: Check for existing artifacts BEFORE writing
    required_artifacts = [
        "resolved_config.json",
        "run_manifest.json",
        "action_cost_summary.json",
        "episode_metrics.json",
        "aggregate_metrics.json",
        "smoke_report.json",
    ]
    existing_artifacts = [
        name for name in required_artifacts
        if (output_dir / name).exists()
    ]

    if existing_artifacts and not overwrite:
        raise FileExistsError(
            f"Existing artifacts found and overwrite=False: {existing_artifacts}. "
            f"Use --overwrite flag to replace, or remove output directory."
        )

    # Compute scenario bank hashes and extract seeds from actual banks
    scenario_bank_hashes = {}
    scenario_bank_seeds = {}  # bank_id -> list of seeds
    for key, path in scenario_bank_paths.items():
        if path.exists():
            scenario_bank_hashes[key] = compute_scenario_bank_hash(path)
            scenario_bank_seeds[key] = extract_scenario_seeds_from_bank(path)

    # PROVENANCE FIX: Verify all banks have consistent seeds
    all_seeds = set()
    for seeds in scenario_bank_seeds.values():
        all_seeds.update(seeds)
    actual_scenario_seeds = sorted(all_seeds)

    # Use repository-relative paths for artifacts
    try:
        pred_cache_rel = prediction_cache_path.relative_to(repo_root)
    except ValueError:
        pred_cache_rel = str(prediction_cache_path)

    # Compute coverage metrics for provenance (BLOCKER 6)
    splits_covered = set(r.split for r in all_results if r.success)
    k_values_covered = set(r.k_capacity for r in all_results if r.success)
    cost_regimes_covered = set(r.cost_regime_id for r in all_results if r.success)
    total_episodes = sum(r.episode_count for r in all_results if r.success)

    # STEP 6 FIX: Use centralized complete config builder
    # This ensures consistency and includes all behavior-affecting fields.

    # STEP 2 FIX: Determine risk model from actual results (matrix_role)
    # Primary lane uses hard_window_v1, engineering lane uses logistic_window_v1
    # Get the risk model from the first result (all results should have same role)
    actual_risk_model_id = all_results[0].risk_model_id if all_results else PRIMARY_RISK_MODEL_ID
    actual_risk_temperature = ENGINEERING_COVERAGE_RISK_TEMPERATURE if actual_risk_model_id == "logistic_window_v1" else None

    # D2 FIX: Compute prediction cache hash from the actual parquet file, not directory
    pred_cache_file = prediction_cache_path / "fd001_prediction_cache_v2.parquet"
    pred_cache_hash = compute_file_hash(pred_cache_file) if pred_cache_file.exists() else None

    # Build action table identity and compute content hashes
    action_table_N5_K1 = ACTION_TABLE_N5_K1
    action_table_N5_K2 = ACTION_TABLE_N5_K2
    action_table_K1_content_hash = compute_action_table_content_hash(action_table_N5_K1)
    action_table_K2_content_hash = compute_action_table_content_hash(action_table_N5_K2)

    # STEP 6 FIX: Build the COMPLETE resolved scientific configuration using
    # the centralized function
    complete_scientific_config = build_complete_scientific_config(
        schema_version="m4_v1",
        policy_id="exact_myopic_v1",
        matrix_role=matrix_role,
        risk_model_id=actual_risk_model_id,
        risk_temperature=actual_risk_temperature,
        tie_tolerance=1e-9,
        environment_version="m2_v1",
        delta_cycles=5,
        rul_scale=125.0,
        age_scale_cycles=341,
        fleet_size=5,
        episode_horizon=100,
        active_k_values=list(k_values_covered),
        active_cost_regimes=list(cost_regimes_covered),
        active_splits=list(splits_covered),
        action_table_K1_identity="ACTION_TABLE_N5_K1_M2_V1",
        action_table_K1_num_actions=len(action_table_N5_K1),
        action_table_K2_identity="ACTION_TABLE_N5_K2_M2_V1",
        action_table_K2_num_actions=len(action_table_N5_K2),
        action_table_K1_content_hash=action_table_K1_content_hash,
        action_table_K2_content_hash=action_table_K2_content_hash,
        prediction_cache_path=str(pred_cache_rel),
        prediction_cache_sha256=pred_cache_hash,
        scenario_bank_ids=list(scenario_bank_hashes.keys()),
        scenario_bank_sha256_values=scenario_bank_hashes,
        scenario_generation_version="m4_production_v1",
        scenario_seeds=actual_scenario_seeds,  # Extracted from actual banks
        scenario_selection_basis="predicted_rul_and_cache_row_continuity",
        episode_count_per_config=len(actual_scenario_seeds),
        information_mode="normal",
        engineering_coverage_threshold_cycles=ENGINEERING_COVERAGE_THRESHOLD_CYCLES,
    )

    # STEP 6 FIX: Compute config_hash using the centralized function
    config_hash = compute_complete_config_hash(complete_scientific_config)

    # Build resolved_config.json with the complete config plus metadata
    resolved_config = {
        **complete_scientific_config,
        "git_commit": git_commit,
        "config_hash": config_hash,
        "repository_root": str(repo_root.relative_to(repo_root.parent) if repo_root.parent else str(repo_root)),
    }

    # 1. resolved_config.json
    config_data = convert_for_json(resolved_config)
    config_path = output_dir / "resolved_config.json"
    write_atomic_json(config_data, config_path)
    written_paths["resolved_config"] = config_path

    # 2. run_manifest.json
    passed_configs = sum(1 for r in all_results if r.success)
    total_steps = sum(r.total_steps for r in all_results if r.success)

    manifest = {
        "schema_version": "m4_v1",
        "mode": "production_smoke_matrix",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "config_hash": config_data["config_hash"],
        "environment_version": "m2_v1",
        "policy_id": "exact_myopic_v1",
        # BLOCKER 6: Full provenance
        "scenario_bank_hashes": convert_for_json(scenario_bank_hashes),
        "split_coverage": sorted(splits_covered),
        "k_coverage": sorted(k_values_covered),
        "cost_regime_coverage": sorted(cost_regimes_covered),
        "total_configs": len(all_results),
        "passed_configs": passed_configs,
        "total_episodes": total_episodes,
        "total_steps": total_steps,
        "artifacts_written": [],  # Will be updated below
    }

    # 3. action_cost_summary.json
    all_action_costs: Dict[int, Dict[str, float]] = {}
    for result in all_results:
        if not result.success:
            continue
        for ep in result.episode_results:
            for i, (action_id, est_cost) in enumerate(zip(ep.action_ids, ep.estimated_myopic_costs)):
                if action_id not in all_action_costs:
                    all_action_costs[action_id] = {
                        "count": 0,
                        "total_estimated_cost": 0.0,
                        "min_estimated_cost": float("inf"),
                        "max_estimated_cost": float("-inf"),
                    }
                all_action_costs[action_id]["count"] += 1
                all_action_costs[action_id]["total_estimated_cost"] += est_cost
                all_action_costs[action_id]["min_estimated_cost"] = min(
                    all_action_costs[action_id]["min_estimated_cost"], est_cost
                )
                all_action_costs[action_id]["max_estimated_cost"] = max(
                    all_action_costs[action_id]["max_estimated_cost"], est_cost
                )

    # Compute averages
    for action_id in all_action_costs:
        count = all_action_costs[action_id]["count"]
        all_action_costs[action_id]["avg_estimated_cost"] = (
            all_action_costs[action_id]["total_estimated_cost"] / count if count > 0 else 0.0
        )
        if all_action_costs[action_id]["min_estimated_cost"] == float("inf"):
            all_action_costs[action_id]["min_estimated_cost"] = 0.0
        if all_action_costs[action_id]["max_estimated_cost"] == float("-inf"):
            all_action_costs[action_id]["max_estimated_cost"] = 0.0

    # D2 FIX: Define path first, then use it in the artifact
    action_cost_path = output_dir / "action_cost_summary.json"

    # D2 FIX: Compute repository-relative path if inside repo, else absolute path
    if output_inside_repo:
        action_cost_rel = action_cost_path.relative_to(repo_root).as_posix()
    else:
        action_cost_rel = str(action_cost_path)

    action_cost_summary = {
        "schema_version": "m4_v1",
        "git_commit": git_commit,
        "config_hash": config_data["config_hash"],
        "environment_version": "m2_v1",
        "policy_id": "exact_myopic_v1",
        "total_unique_actions": len(all_action_costs),
        "action_costs": convert_for_json(all_action_costs),
        # BLOCKER 6: Provenance
        "split_coverage": sorted(splits_covered),
        "k_coverage": sorted(k_values_covered),
        "cost_regime_coverage": sorted(cost_regimes_covered),
        "episode_count": total_episodes,
        # D2 FIX: Use actual path (repository-relative if inside repo, else absolute)
        "repository_relative_path": action_cost_rel,
    }
    write_atomic_json(convert_for_json(action_cost_summary), action_cost_path)
    written_paths["action_cost_summary"] = action_cost_path

    # 4. episode_metrics.json
    episode_metrics_list = []
    for result in all_results:
        if not result.success:
            continue
        for ep in result.episode_results:
            episode_metrics_list.append({
                "config_key": f"{result.split}_K{result.k_capacity}_{result.cost_regime_id}",
                **episode_result_to_dict(ep),
            })

    # D2 FIX: Define path first, then compute relative path
    episode_metrics_path = output_dir / "episode_metrics.json"
    episode_metrics_rel = episode_metrics_path.relative_to(repo_root).as_posix() if output_inside_repo else str(episode_metrics_path)

    episode_metrics = {
        "schema_version": "m4_v1",
        "git_commit": git_commit,
        "config_hash": config_data["config_hash"],
        "environment_version": "m2_v1",
        "policy_id": "exact_myopic_v1",
        "total_episodes": len(episode_metrics_list),
        "episodes": convert_for_json(episode_metrics_list),
        # BLOCKER 6: Provenance
        "split_coverage": sorted(splits_covered),
        "k_coverage": sorted(k_values_covered),
        "cost_regime_coverage": sorted(cost_regimes_covered),
        # D2 FIX: Use actual path (repository-relative if inside repo, else absolute)
        "repository_relative_path": episode_metrics_rel,
    }
    write_atomic_json(convert_for_json(episode_metrics), episode_metrics_path)
    written_paths["episode_metrics"] = episode_metrics_path

    # 5. aggregate_metrics.json
    # D2 FIX: Define path first, then compute relative path
    aggregate_path = output_dir / "aggregate_metrics.json"
    aggregate_rel = aggregate_path.relative_to(repo_root).as_posix() if output_inside_repo else str(aggregate_path)

    aggregate = {
        "schema_version": "m4_v1",
        "git_commit": git_commit,
        "config_hash": config_data["config_hash"],
        "environment_version": "m2_v1",
        "policy_id": "exact_myopic_v1",
        "total_configs": len(all_results),
        "passed_configs": passed_configs,
        "total_episodes": total_episodes,
        "total_steps": total_steps,
        "total_preventive_replacements": sum(r.preventive_replacements for r in all_results if r.success),
        "total_failures": sum(r.failures for r in all_results if r.success),
        "total_preventive_cost": sum(r.preventive_cost for r in all_results if r.success),
        "total_failure_cost": sum(r.failure_cost for r in all_results if r.success),
        "total_unused_life_cost": sum(r.unused_life_cost for r in all_results if r.success),
        "total_environment_cost": sum(r.total_environment_cost for r in all_results if r.success),
        "total_estimated_myopic_cost": sum(r.estimated_myopic_cost for r in all_results if r.success),
        "total_nan_inf_count": sum(r.nan_inf_count for r in all_results if r.success),
        "total_missing_prediction_count": sum(r.missing_prediction_count for r in all_results if r.success),
        "total_split_violation_count": sum(r.split_violation_count for r in all_results if r.success),
        "avg_steps_per_episode": total_steps / total_episodes if total_episodes > 0 else 0.0,
        "avg_cost_per_step": (
            sum(r.total_environment_cost for r in all_results if r.success) / total_steps
            if total_steps > 0 else 0.0
        ),
        # BLOCKER 6: Provenance
        "split_coverage": sorted(splits_covered),
        "k_coverage": sorted(k_values_covered),
        "cost_regime_coverage": sorted(cost_regimes_covered),
        # D2 FIX: Use actual repository-relative path
        "repository_relative_path": aggregate_rel,
    }
    write_atomic_json(convert_for_json(aggregate), aggregate_path)
    written_paths["aggregate_metrics"] = aggregate_path

    # 6. smoke_report.json (the main report)
    # D2 FIX: Define path first, then compute relative path
    smoke_report_path = output_dir / "smoke_report.json"
    smoke_report_rel = smoke_report_path.relative_to(repo_root).as_posix() if output_inside_repo else str(smoke_report_path)

    smoke_report = {
        "schema_version": "m4_v1",
        "mode": "production_smoke_matrix",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "config_hash": config_data["config_hash"],
        "environment_version": "m2_v1",
        "policy_id": "exact_myopic_v1",
        "total_configs": len(all_results),
        "passed_configs": passed_configs,
        "failed_configs": len(all_results) - passed_configs,
        "all_passed": passed_configs == len(all_results),
        "matrix_dimensions": {
            "splits": ["predictor_train", "rl_validation"],
            "k_values": [1, 2],
            "cost_regimes": list(COST_REGIMES.keys()),
            "expected_total": 16,
        },
        # BLOCKER 6: Actual coverage
        "split_coverage": sorted(splits_covered),
        "k_coverage": sorted(k_values_covered),
        "cost_regime_coverage": sorted(cost_regimes_covered),
        "prediction_cache_path": str(pred_cache_rel),
        "scenario_bank_dir": "data/scenario_banks/m4_production",
        "scenario_bank_hashes": convert_for_json(scenario_bank_hashes),
        "results": [config_result_to_dict(r) for r in all_results],
        # D2 FIX: Use actual repository-relative path
        "repository_relative_path": smoke_report_rel,
    }
    write_atomic_json(convert_for_json(smoke_report), smoke_report_path)
    written_paths["smoke_report"] = smoke_report_path

    # Update manifest with artifact list
    # D2 FIX: Define path first, then compute relative path
    manifest_path = output_dir / "run_manifest.json"
    manifest_rel = manifest_path.relative_to(repo_root).as_posix() if output_inside_repo else str(manifest_path)

    # Add repository_relative_path to manifest
    manifest["artifacts_written"] = list(written_paths.keys())
    manifest["repository_relative_path"] = manifest_rel
    write_atomic_json(convert_for_json(manifest), manifest_path)
    written_paths["run_manifest"] = manifest_path

    return written_paths


def run_production_smoke_matrix(
    output_dir: Path,
    config_path: Optional[Path] = None,
    overwrite: bool = False,
    matrix_role: str = "primary_contract_policy",
) -> Dict[str, Any]:
    """
    Run full production smoke matrix.

    STEP 2 FIX: Two separate lanes - primary contract policy and engineering coverage.
    Each lane has separate artifacts, config hashes, and output directories.
    Do not mix the two policies into one scientific aggregate.

    16 configurations:
    - 2 splits × 2 K values × 4 cost regimes

    Args:
        output_dir: Output directory for artifacts
        config_path: Optional config file path
        overwrite: If False and artifacts exist, raise FileExistsError before running
        matrix_role: "primary_contract_policy" (hard_window_v1) or
                     "engineering_behavior_coverage" (logistic_window_v1)

    Returns:
        Report dictionary with results

    Raises:
        FileExistsError: If artifacts exist and overwrite=False
    """
    print("\n=== Running Production Smoke Matrix ===")

    # STEP 2 FIX: Determine risk model based on matrix_role
    if matrix_role == "engineering_behavior_coverage":
        risk_model_id = ENGINEERING_COVERAGE_RISK_MODEL_ID
        risk_temperature = ENGINEERING_COVERAGE_RISK_TEMPERATURE
        print(f"  Lane: ENGINEERING BEHAVIOR COVERAGE (logistic_window_v1)")
    else:
        risk_model_id = PRIMARY_RISK_MODEL_ID
        risk_temperature = PRIMARY_RISK_TEMPERATURE
        print(f"  Lane: PRIMARY CONTRACT POLICY (hard_window_v1)")

    # STEP 4 FIX: Reinstate strict repository-local outputs.
    # Production output directories MUST be inside the repository.
    # Reject external output paths BEFORE any environment execution.
    if output_dir:
        output_dir = Path(output_dir).resolve()
        repo_root = Path(__file__).parent.parent.resolve()
        try:
            output_dir.relative_to(repo_root)
            print(f"  Output directory verified inside repository: {output_dir.relative_to(repo_root)}")
        except ValueError:
            # Output directory is outside repository - reject before execution
            raise ValueError(
                f"Output directory is outside the repository.\n"
                f"  Output dir: {output_dir}\n"
                f"  Repo root: {repo_root}\n"
                f"Production artifacts must be written to repository-relative paths only.\n"
                f"Use an output directory inside: {repo_root}/results/milestone4/"
            )

    # D5 FIX: Document that the centralized production artifact writer supersedes
    # MyopicArtifactWriter for multi-configuration production runs.
    # MyopicArtifactWriter is available for single-configuration runs but is not
    # used by the production smoke matrix runner.

    # Define all 16 configurations
    splits = ["predictor_train", "rl_validation"]
    k_values = [1, 2]
    regimes = list(COST_REGIMES.keys())

    # Base paths
    repo_root = Path(__file__).parent.parent
    prediction_cache_dir = repo_root / "data" / "processed" / "fd001" / "v2" / "06_PREDICTIONS"
    scenario_bank_dir = repo_root / "data" / "scenario_banks" / "m4_production"

    # BLOCKER 5: Check for existing artifacts BEFORE running (not just before writing)
    required_artifacts = [
        "resolved_config.json",
        "run_manifest.json",
        "action_cost_summary.json",
        "episode_metrics.json",
        "aggregate_metrics.json",
        "smoke_report.json",
    ]
    if output_dir:
        existing_artifacts = [
            name for name in required_artifacts
            if (output_dir / name).exists()
        ]
        if existing_artifacts and not overwrite:
            raise FileExistsError(
                f"Existing artifacts found and overwrite=False: {existing_artifacts}. "
                f"Use --overwrite flag to replace, or remove output directory."
            )

    # Load prediction store once
    print("Loading production PredictionStore...")
    prediction_store = load_prediction_store_for_split("predictor_train", prediction_cache_dir)
    print(f"  Loaded {len(prediction_store)} predictions")

    all_results: List[ConfigResult] = []

    for split in splits:
        for k in k_values:
            for regime_id in regimes:
                config_key = f"{split}_K{k}_{regime_id}"
                print(f"\nRunning: {config_key}")

                # Select scenario bank for this configuration
                scenario_bank_path = scenario_bank_dir / f"{split}_K{k}_{regime_id}.json"

                if not scenario_bank_path.exists():
                    print(f"  SKIP: Scenario bank not found: {scenario_bank_path}")
                    all_results.append(ConfigResult(
                        split=split,
                        k_capacity=k,
                        cost_regime_id=regime_id,
                        risk_model_id=risk_model_id,
                        episode_count=0,
                        total_steps=0,
                        action_ids=[],
                        preventive_replacements=0,
                        failures=0,
                        preventive_cost=0.0,
                        failure_cost=0.0,
                        unused_life_cost=0.0,
                        total_environment_cost=0.0,
                        estimated_myopic_cost=0.0,
                        terminated_count=0,
                        truncated_count=0,
                        nan_inf_count=0,
                        missing_prediction_count=0,
                        split_violation_count=0,
                        episode_results=[],
                        success=False,
                        error=f"Scenario bank not found: {scenario_bank_path}",
                    ))
                    continue

                run_config = ProductionRunConfig(
                    split=split,
                    k_capacity=k,
                    cost_regime_id=regime_id,
                    risk_model_id=risk_model_id,
                    scenario_bank_path=scenario_bank_path,
                    prediction_cache_path=prediction_cache_dir,
                )

                result = run_production_config(run_config, prediction_store)
                all_results.append(result)

                status = "PASS" if result.success else f"FAIL: {result.error}"
                print(f"  {status} ({result.episode_count} episodes, {result.total_steps} steps)")

    # Build report
    passed_configs = sum(1 for r in all_results if r.success)
    total_configs = len(all_results)

    # Load config for artifact generation
    base_config = {}
    if config_path and config_path.exists():
        with open(config_path, "r") as f:
            base_config = json.load(f)

    git_commit = get_git_commit()

    # Build scenario bank paths map
    scenario_bank_paths = {}
    for split in splits:
        for k in k_values:
            for regime_id in regimes:
                key = f"{split}_K{k}_{regime_id}"
                scenario_bank_paths[key] = scenario_bank_dir / f"{split}_K{k}_{regime_id}.json"

    # Write all required artifacts
    if output_dir:
        output_dir = Path(output_dir).resolve()  # Resolve to match written_paths
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nWriting production artifacts to: {output_dir}")

        written_paths = write_production_artifacts(
            output_dir=output_dir,
            all_results=all_results,
            config=base_config,
            git_commit=git_commit,
            repo_root=repo_root,
            prediction_cache_path=prediction_cache_dir,
            scenario_bank_paths=scenario_bank_paths,
            overwrite=overwrite,  # BLOCKER 5: Thread overwrite parameter
            matrix_role=matrix_role,  # STEP 2 FIX: Pass matrix role
        )

        print(f"  Written artifacts:")
        for name, path in written_paths.items():
            print(f"    - {name}: {path.relative_to(output_dir)}")

    report = {
        "schema_version": "m4_v1",
        "mode": "production_smoke_matrix",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "config_path": str(config_path) if config_path else None,
        "overwrite": overwrite,
        "matrix_role": matrix_role,  # STEP 2 FIX: Record which lane was run
        "total_configs": total_configs,
        "passed_configs": passed_configs,
        "failed_configs": total_configs - passed_configs,
        "all_passed": passed_configs == total_configs,
        "matrix_dimensions": {
            "splits": splits,
            "k_values": k_values,
            "cost_regimes": regimes,
            "expected_total": len(splits) * len(k_values) * len(regimes),
        },
        "prediction_cache_path": str(prediction_cache_dir.relative_to(repo_root)),
        "scenario_bank_dir": str(scenario_bank_dir.relative_to(repo_root)),
        "results": [config_result_to_dict(r) for r in all_results],
        "artifacts_written": list(written_paths.keys()) if output_dir else [],
    }

    # Write report
    if output_dir:
        report_path = output_dir / "smoke_matrix_report.json"
        write_atomic_json(convert_for_json(report), report_path)
        print(f"\nReport written to: {report_path}")

    return report


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="run_m4_production_smoke",
        description="Milestone 4 Production Smoke Matrix Runner (two-lane structure)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config JSON file",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing artifacts. If False and artifacts exist, exit with nonzero code.",
    )

    parser.add_argument(
        "--matrix-role",
        type=str,
        default="primary_contract_policy",
        choices=["primary_contract_policy", "engineering_behavior_coverage"],
        help="Which lane to run: primary contract policy (hard_window_v1) or engineering behavior coverage (logistic_window_v1)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    try:
        report = run_production_smoke_matrix(
            output_dir=output_dir,
            config_path=args.config,
            overwrite=args.overwrite,
            matrix_role=args.matrix_role,
        )

        if report["all_passed"]:
            print(f"\n✓ Production smoke matrix PASSED ({report['passed_configs']}/{report['total_configs']} configs)")
            return 0
        else:
            print(f"\n✗ Production smoke matrix FAILED ({report['failed_configs']}/{report['total_configs']} configs failed)")
            return 1

    except FileExistsError as e:
        # BLOCKER 5: Overwrite protection triggered - nonzero exit
        print(f"\n✗ Overwrite protection: {e}")
        return 1

    except Exception as e:
        print(f"\n✗ Production smoke matrix FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())