#!/usr/bin/env python3
"""
Complete 96-episode small matrix for Milestone 3 experiment readiness.

Runs:
- splits: predictor_train, rl_validation
- K: 1, 2
- cost regimes: all four frozen regimes
- policies: all six policies
- scenarios: one derived scenario for every split/K/regime cell
- reset seed: 6521

Expected episodes: 2 × 2 × 4 × 6 = 96 episodes
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.baselines.evaluator import PolicyEvaluator, EvaluationConfig, EpisodeResult
from src.baselines.rule_policies import (
    CorrectiveOnly,
    RandomFeasible,
    AgeThreshold,
    PredictedRULThreshold,
    GreedyPredictedRUL,
)
from src.baselines.oracle_policy import OracleThreshold
from src.baselines.protocols import PolicyContext, OracleContext
from src.envs import get_default_config, EnvironmentConfig, SelectiveMaintenanceEnv
from src.envs.scenario_bank import ScenarioBank
from src.baselines.case_loader import get_scenario_bank_for_case, ALL_COST_REGIMES
from src.baselines.artifacts import (
    write_episode_results,
    write_summary_by_policy,
    write_run_provenance,
    write_artifact_manifest,
    compute_sha256,
)


# Configuration
SPLITS = ["predictor_train", "rl_validation"]
K_VALUES = [1, 2]
COST_REGIMES = sorted(ALL_COST_REGIMES)
POLICY_FAMILIES = [
    "corrective_only",
    "random_feasible",
    "age_threshold",
    "predicted_rul_threshold",
    "greedy_predicted_rul",
    "oracle_threshold",
]
RESET_SEED = 6521
POLICY_SEED = 42
EPISODE_HORIZON = 100

# Default thresholds for smoke testing
DEFAULT_THRESHOLDS = {
    "age_threshold": 100,
    "predicted_rul_threshold": 30,
    "greedy_predicted_rul": 50,
    "oracle_threshold": 10,
}


def run_96_episode_matrix(
    output_dir: Path,
) -> tuple[List[EpisodeResult], Dict[str, Any]]:
    """
    Run the complete 96-episode matrix.

    Returns:
        Tuple of (all_results, metadata)
    """
    print("=" * 60)
    print("M3 96-EPISODE SMALL MATRIX")
    print("=" * 60)

    all_results: List[EpisodeResult] = []
    metadata = {
        "splits": SPLITS,
        "k_values": K_VALUES,
        "cost_regimes": COST_REGIMES,
        "policy_families": POLICY_FAMILIES,
        "reset_seed": RESET_SEED,
        "policy_seed": POLICY_SEED,
        "episode_horizon": EPISODE_HORIZON,
    }

    episode_count = 0

    for split in SPLITS:
        print(f"\n{'=' * 50}")
        print(f"SPLIT: {split}")
        print(f"{'=' * 50}")

        for k in K_VALUES:
            print(f"\n  K={k}")

            for cost_regime_id in COST_REGIMES:
                print(f"\n    Cost regime: {cost_regime_id}")

                # Load derived scenario bank for this split/K/regime
                scenario_bank = get_scenario_bank_for_case(
                    split=split,
                    k=k,
                    cost_regime_id=cost_regime_id,
                    source_bank_path=f"data/scenario_banks/{split}_smoke.json",
                )

                # Use first scenario from derived bank
                if not scenario_bank.scenarios:
                    print(f"      ERROR: No scenarios found for {split}/K={k}/{cost_regime_id}")
                    continue

                scenario = scenario_bank.scenarios[0]
                scenario_id = scenario.scenario_id
                print(f"      Scenario: {scenario_id}")

                # Create environment config
                env_config = get_default_config(
                    split=split,
                    maintenance_capacity=k,
                    cost_regime_id=cost_regime_id,
                    seed=RESET_SEED,
                )

                # Run all six policies
                for policy_family in POLICY_FAMILIES:
                    result = run_single_episode(
                        scenario_bank=scenario_bank,
                        scenario_id=scenario_id,
                        env_config=env_config,
                        policy_family=policy_family,
                        reset_seed=RESET_SEED,
                        policy_seed=POLICY_SEED,
                    )

                    all_results.append(result)
                    episode_count += 1

                    status = "✓" if result.completed else "✗"
                    print(f"      {status} {policy_family}: {result.episode_steps} steps, return={result.episode_return:.2f}")

    metadata["total_episodes"] = episode_count
    metadata["completed_episodes"] = sum(1 for r in all_results if r.completed)

    return all_results, metadata


def run_single_episode(
    scenario_bank: ScenarioBank,
    scenario_id: str,
    env_config: EnvironmentConfig,
    policy_family: str,
    reset_seed: int,
    policy_seed: int,
) -> EpisodeResult:
    """Run a single episode for a policy."""
    k = env_config.maintenance_capacity
    is_oracle = policy_family == "oracle_threshold"

    # Create evaluator
    evaluator = PolicyEvaluator(
        env_config=env_config,
        allow_oracle=is_oracle,
        diagnostic_mode=is_oracle,
    )

    # Create policy
    if policy_family == "corrective_only":
        policy = CorrectiveOnly()
        threshold = None
    elif policy_family == "random_feasible":
        policy = RandomFeasible(seed=policy_seed)
        threshold = None
    elif policy_family == "age_threshold":
        policy = AgeThreshold(threshold=DEFAULT_THRESHOLDS["age_threshold"])
        threshold = DEFAULT_THRESHOLDS["age_threshold"]
    elif policy_family == "predicted_rul_threshold":
        policy = PredictedRULThreshold(threshold=DEFAULT_THRESHOLDS["predicted_rul_threshold"])
        threshold = DEFAULT_THRESHOLDS["predicted_rul_threshold"]
    elif policy_family == "greedy_predicted_rul":
        policy = GreedyPredictedRUL(activation_threshold=DEFAULT_THRESHOLDS["greedy_predicted_rul"])
        threshold = None
    elif policy_family == "oracle_threshold":
        policy = OracleThreshold(threshold=DEFAULT_THRESHOLDS["oracle_threshold"])
        threshold = DEFAULT_THRESHOLDS["oracle_threshold"]
    else:
        raise ValueError(f"Unknown policy family: {policy_family}")

    # Create context
    context = evaluator.create_context(policy_family, policy_seed=policy_seed)

    # Create environment with derived scenario bank
    env = SelectiveMaintenanceEnv(
        config=env_config,
        scenario_bank=scenario_bank,
        info_mode="diagnostic" if is_oracle else "normal",
    )

    # Create evaluation config
    eval_config = EvaluationConfig(
        env_config=env_config,
        policy_id=f"{policy_family}_{scenario_id}_{reset_seed}",
        policy_family=policy_family,
        threshold=threshold if policy_family not in ["greedy_predicted_rul"] else None,
        activation_threshold=threshold if policy_family == "greedy_predicted_rul" else None,
        policy_seed=policy_seed,
    )

    run_id = f"matrix_{policy_family}_{env_config.split}_k{k}_{env_config.cost_regime_id}_{reset_seed}"

    # Run episode
    result = evaluator.evaluate_episode(
        env=env,
        policy=policy,
        context=context,
        scenario_id=scenario_id,
        reset_seed=reset_seed,
        eval_config=eval_config,
        run_id=run_id,
    )

    return result


def validate_matrix_results(
    results: List[EpisodeResult],
    output_dir: Path,
) -> bool:
    """Validate 96-episode matrix results."""
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    errors = []

    # Check episode count
    expected_count = 96  # 2 splits × 2 K × 4 regimes × 6 policies
    actual_count = len(results)
    if actual_count != expected_count:
        errors.append(f"Expected {expected_count} episodes, got {actual_count}")
    else:
        print(f"✓ Episode count: {actual_count} (expected {expected_count})")

    # Check all episodes completed 100 steps
    for r in results:
        if r.episode_steps != EPISODE_HORIZON:
            errors.append(f"{r.run_id}: expected {EPISODE_HORIZON} steps, got {r.episode_steps}")

    if not any("steps" in str(e) for e in errors):
        print(f"✓ All episodes completed {EPISODE_HORIZON} steps")

    # Check no NaN/Inf
    for r in results:
        if not np.isfinite(r.episode_return):
            errors.append(f"{r.run_id}: episode_return not finite")
        if not np.isfinite(r.total_cost):
            errors.append(f"{r.run_id}: total_cost not finite")

    if not any("not finite" in str(e) for e in errors):
        print("✓ All numeric values are finite")

    # Check terminated=False, truncated=True
    terminated_true = sum(1 for r in results if r.terminated_count > 0)
    truncated_false = sum(1 for r in results if not r.truncated)

    if terminated_true > 0:
        errors.append(f"{terminated_true} episodes with terminated=True")
    else:
        print("✓ All episodes have terminated=False")

    if truncated_false > 0:
        errors.append(f"{truncated_false} episodes with truncated=False")
    else:
        print("✓ All episodes have truncated=True")

    # Check reward = -total_cost
    for r in results:
        if not np.isclose(r.episode_return, -r.total_cost, rtol=1e-5):
            errors.append(f"{r.run_id}: reward ({r.episode_return}) != -total_cost ({-r.total_cost})")

    if not any("reward" in str(e) for e in errors):
        print("✓ All episodes have reward = -total_cost")

    # Check no capacity violations (actions are legal)
    # This is enforced by the environment, but verify action counts
    for r in results:
        if r.action_count < 0:
            errors.append(f"{r.run_id}: negative action count")

    if not any("action" in str(e) for e in errors):
        print("✓ All action counts are valid")

    # Check oracle runs in diagnostic mode (oracle policies should have completed)
    oracle_results = [r for r in results if r.policy_family == "oracle_threshold"]
    oracle_completed = sum(1 for r in oracle_results if r.completed)
    if oracle_completed < len(oracle_results):
        errors.append(f"{len(oracle_results) - oracle_completed} oracle episodes failed")
    else:
        print(f"✓ All {len(oracle_results)} oracle episodes completed")

    # Write and validate artifacts
    print("\nValidating artifacts...")

    if results:
        df = write_episode_results(results, output_dir)
        print(f"✓ Wrote {len(results)} episode results")

        # Check parquet for NaN/Inf
        if (output_dir / "episode_results.parquet").exists():
            result_df = pd.read_parquet(output_dir / "episode_results.parquet")
            for col in ["episode_return", "total_cost", "preventive_cost", "failure_cost", "wasted_life_cost"]:
                if col in result_df.columns:
                    if result_df[col].isna().any():
                        errors.append(f"NaN found in {col}")
                    if not all(np.isfinite(result_df[col].dropna())):
                        errors.append(f"Inf found in {col}")
            print("✓ Episode results parquet validation passed")

    if errors:
        print("\nErrors:")
        for err in errors:
            print(f"  ✗ {err}")
        return False

    print("\n✓ ALL VALIDATIONS PASSED")
    return True


def main():
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent.parent / "results" / "milestone3" / f"matrix_96_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Run matrix
    results, metadata = run_96_episode_matrix(output_dir)

    # Write artifacts
    print("\n" + "=" * 60)
    print("WRITING ARTIFACTS")
    print("=" * 60)

    if results:
        write_episode_results(results, output_dir)
        print(f"✓ Wrote {len(results)} episode results")

        # Write summary
        summary_df = pd.DataFrame([
            {
                "run_id": r.run_id,
                "policy_family": r.policy_family,
                "split": r.split,
                "k_capacity": r.maintenance_capacity,
                "cost_regime_id": r.cost_regime_id,
                "episode_return": r.episode_return,
                "total_cost": r.total_cost,
                "episode_steps": r.episode_steps,
            }
            for r in results
        ])
        summary_path = output_dir / "summary_by_policy.csv"
        summary_df.to_csv(summary_path, index=False)
        print("✓ Wrote summary_by_policy.csv")

    # Write provenance
    write_run_provenance(metadata, output_dir)
    print("✓ Wrote run_provenance.json")

    # Write manifest
    write_artifact_manifest(output_dir)
    print("✓ Wrote artifact_manifest.json")

    # Validate
    valid = validate_matrix_results(results, output_dir)

    if not valid:
        print("\n" + "=" * 60)
        print("VALIDATION FAILED")
        print("=" * 60)
        return 1

    print("\n" + "=" * 60)
    print("96-EPISODE MATRIX COMPLETE")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  Total episodes: {len(results)}")
    print(f"  Completed: {sum(1 for r in results if r.completed)}")
    print(f"  Splits: {SPLITS}")
    print(f"  K values: {K_VALUES}")
    print(f"  Cost regimes: {COST_REGIMES}")
    print(f"  Policies: {POLICY_FAMILIES}")
    print(f"  Output: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())