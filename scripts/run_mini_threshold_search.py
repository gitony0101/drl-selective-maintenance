#!/usr/bin/env python3
"""
Mini threshold search for Milestone 3 experiment readiness.

Runs a genuine mini search with exactly:
- Age: [50, 150]
- Predicted RUL: [20, 60]
- Greedy activation: [20, 60]
- Oracle: [10, 30]

Uses:
- rl_validation only
- K=1 and K=2
- failure-light-no-waste cost regime
- one scenario per K
- reset seed 6521

Expected candidates: 4 families × 2 thresholds × 2 K = 16
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.baselines.tuning import (
    tune_threshold,
    select_best_threshold,
    ThresholdCandidate,
    candidates_to_dataframe,
)
from src.baselines.evaluator import EpisodeResult
from src.envs import get_default_config, EnvironmentConfig
from src.envs.scenario_bank import load_scenario_bank
from src.baselines.case_loader import load_cases, get_scenario_bank_for_case
from src.baselines.artifacts import (
    write_threshold_search_results,
    write_threshold_search_summary,
    write_selected_thresholds,
    write_resolved_config,
    write_artifact_manifest,
    validate_artifacts,
    compute_sha256,
)


# Mini-grid from task specification
MINI_GRIDS = {
    "age_threshold": [50, 150],
    "predicted_rul_threshold": [20, 60],
    "greedy_predicted_rul": [20, 60],
    "oracle_threshold": [10, 30],
}

THRESHOLD_FAMILIES = [
    "age_threshold",
    "predicted_rul_threshold",
    "greedy_predicted_rul",
    "oracle_threshold",
]

K_VALUES = [1, 2]
COST_REGIME = "failure-light-no-waste"
SPLIT = "rl_validation"
RESET_SEEDS = [6521]  # Single seed for mini search
POLICY_SEED = 42


def run_mini_search(
    output_dir: Path,
) -> tuple[
    List[ThresholdCandidate],
    Dict[str, Any],
    Dict[str, Any],
    List[Dict[str, Any]],
]:
    """
    Run mini threshold search.

    Returns:
        Tuple of (all_candidates, metadata)
    """
    print("=" * 60)
    print("M3 MINI THRESHOLD SEARCH")
    print("=" * 60)

    all_candidates: List[ThresholdCandidate] = []
    selected_thresholds: Dict[str, Any] = {}
    all_episode_rows: List[Dict[str, Any]] = []

    # Run tuning for each K and policy family
    for k in K_VALUES:
        print(f"\n{'=' * 40}")
        print(f"K={k}")
        print(f"{'=' * 40}")

        # Get the derived scenario bank for this K
        scenario_bank = get_scenario_bank_for_case(
            split=SPLIT,
            k=k,
            cost_regime_id=COST_REGIME,
            source_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )
        scenario_ids = [s.scenario_id for s in scenario_bank.scenarios]
        print(f"K={k} scenario bank: {len(scenario_ids)} scenarios")
        print(f"  Scenario IDs: {scenario_ids}")

        for policy_family in THRESHOLD_FAMILIES:
            print(f"\nTuning {policy_family}...")

            # Get custom mini-grid for this policy
            threshold_grid = MINI_GRIDS[policy_family]

            # Create env config with the scenario bank path set to None
            # and pass the scenario_bank directly
            env_config = get_default_config(
                split=SPLIT,
                maintenance_capacity=k,
                cost_regime_id=COST_REGIME,
                seed=RESET_SEEDS[0],
            )

            # Tune with custom grid - need to pass scenario_bank
            # The tune_threshold function needs to accept scenario_bank
            from src.baselines.evaluator import PolicyEvaluator, EvaluationConfig
            from src.envs import SelectiveMaintenanceEnv

            evaluator = PolicyEvaluator(
                env_config=env_config,
                allow_oracle=policy_family == "oracle_threshold",
                diagnostic_mode=policy_family == "oracle_threshold",
            )

            candidates_for_policy: List[ThresholdCandidate] = []
            episode_results_all = []

            for threshold in threshold_grid:
                episode_results: List[EpisodeResult] = []

                for scenario_id in scenario_ids:
                    for reset_seed in RESET_SEEDS:
                        # Create policy
                        if policy_family == "greedy_predicted_rul":
                            policy = evaluator.create_policy(
                                policy_family,
                                activation_threshold=threshold,
                                policy_seed=POLICY_SEED,
                            )
                        else:
                            policy = evaluator.create_policy(
                                policy_family,
                                threshold=threshold,
                                policy_seed=POLICY_SEED,
                            )

                        context = evaluator.create_context(policy_family, policy_seed=POLICY_SEED)

                        # Create environment WITH the derived scenario bank
                        is_oracle = policy_family == "oracle_threshold"
                        env = SelectiveMaintenanceEnv(
                            config=env_config,
                            scenario_bank=scenario_bank,  # Pass derived bank directly
                            info_mode="diagnostic" if is_oracle else "normal",
                        )

                        eval_config = EvaluationConfig(
                            env_config=env_config,
                            policy_id=f"{policy_family}_{threshold}",
                            policy_family=policy_family,
                            threshold=threshold if policy_family != "greedy_predicted_rul" else None,
                            activation_threshold=threshold if policy_family == "greedy_predicted_rul" else None,
                            policy_seed=POLICY_SEED,
                        )

                        run_id = f"tune_{policy_family}_{threshold}_k{k}_{COST_REGIME}_{scenario_id}_{reset_seed}"

                        result = evaluator.evaluate_episode(
                            env=env,
                            policy=policy,
                            context=context,
                            scenario_id=scenario_id,
                            reset_seed=reset_seed,
                            eval_config=eval_config,
                            run_id=run_id,
                        )

                        episode_results.append(result)
                        all_episode_rows.append({
                            "policy_family": policy_family,
                            "threshold": float(threshold),
                            "k_capacity": int(k),
                            "cost_regime_id": COST_REGIME,
                            "scenario_id": scenario_id,
                            "reset_seed": int(reset_seed),
                            "total_cost": float(result.total_cost),
                            "preventive_cost": float(result.preventive_cost),
                            "failure_cost": float(result.failure_cost),
                            "wasted_life_cost": float(result.wasted_life_cost),
                            "failure_count": int(result.failure_count),
                            "episode_steps": int(result.episode_steps),
                            "completed": bool(result.completed),
                        })

                # Check for failed episodes
                failed_episodes = [r for r in episode_results if not r.completed]
                if failed_episodes:
                    error_details = []
                    for r in failed_episodes[:3]:
                        error_details.append(
                            f"scenario={r.scenario_id}, seed={r.reset_seed}: {r.error}"
                        )
                    raise RuntimeError(
                        f"Threshold tuning failed for {policy_family} at threshold={threshold}, "
                        f"K={k}, regime={COST_REGIME}: "
                        f"{len(failed_episodes)}/{len(episode_results)} episodes failed. "
                        f"Errors: {'; '.join(error_details)}"
                    )

                # Aggregate results
                total_costs = [r.total_cost for r in episode_results if r.completed]
                failures = [r.failure_count for r in episode_results if r.completed]
                wasted_life_costs = [r.wasted_life_cost for r in episode_results if r.completed]

                mean_total_cost = float(np.mean(total_costs))
                total_failures = sum(failures)
                mean_wasted_life_cost = float(np.mean(wasted_life_costs))

                candidates_for_policy.append(ThresholdCandidate(
                    policy_family=policy_family,
                    threshold=threshold,
                    k_capacity=k,
                    cost_regime_id=COST_REGIME,
                    mean_total_cost=mean_total_cost,
                    total_failures=total_failures,
                    mean_wasted_life_cost=mean_wasted_life_cost,
                    episode_count=len(total_costs),
                ))

                episode_results_all.extend(episode_results)

            # Select best threshold for this policy/K
            selected = select_best_threshold(candidates_for_policy)
            key = f"{policy_family}_k{k}_{COST_REGIME}"
            selected_thresholds[key] = {
                "threshold": selected.threshold,
                "policy_family": selected.policy_family,
                "k_capacity": selected.k_capacity,
                "cost_regime_id": selected.cost_regime_id,
                "mean_total_cost": selected.mean_total_cost,
                "total_failures": selected.total_failures,
                "mean_wasted_life_cost": selected.mean_wasted_life_cost,
                "episode_count": selected.episode_count,
                "tie_break_reason": selected.tie_break_reason,
            }

            all_candidates.extend(candidates_for_policy)

            print(f"  Best threshold: {selected.threshold}")
            print(f"  Mean cost: {selected.mean_total_cost:.2f}")
            print(f"  Candidates: {len(candidates_for_policy)}")

    return all_candidates, selected_thresholds, {
        "split": SPLIT,
        "k_values": K_VALUES,
        "cost_regime": COST_REGIME,
        "mini_grids": MINI_GRIDS,
        "reset_seeds": RESET_SEEDS,
        "policy_seed": POLICY_SEED,
        "total_candidates": len(all_candidates),
    }, all_episode_rows


def validate_mini_search_results(
    candidates: List[ThresholdCandidate],
    output_dir: Path,
) -> bool:
    """Validate mini search results."""
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    errors = []

    # Check candidate count
    expected_count = 16  # 4 families × 2 thresholds × 2 K
    actual_count = len(candidates)
    if actual_count != expected_count:
        errors.append(f"Expected {expected_count} candidates, got {actual_count}")
    else:
        print(f"✓ Candidate count: {actual_count} (expected {expected_count})")

    # Check all episodes complete
    for c in candidates:
        if c.episode_count == 0:
            errors.append(f"{c.policy_family}_k{c.k_capacity}: episode_count=0")
        if not np.isfinite(c.mean_total_cost):
            errors.append(f"{c.policy_family}_k{c.k_capacity}: mean_total_cost not finite")

    if not errors:
        print("✓ All episodes completed (episode_count > 0)")
        print("✓ All mean_total_cost values are finite")

    # Check oracle candidates are included
    oracle_candidates = [c for c in candidates if c.policy_family == "oracle_threshold"]
    if len(oracle_candidates) == 0:
        errors.append("No oracle candidates found")
    else:
        print(f"✓ Oracle candidates included: {len(oracle_candidates)}")

    # Check for duplicates
    candidate_keys = [(c.policy_family, c.threshold, c.k_capacity, c.cost_regime_id) for c in candidates]
    if len(candidate_keys) != len(set(candidate_keys)):
        errors.append("Duplicate candidates found")
    else:
        print("✓ No duplicate candidates")

    # Validate artifacts
    print("\nValidating artifacts...")

    # Check mini-search specific artifacts
    required_files = [
        "threshold_search_results.parquet",
        "threshold_search_summary.csv",
        "selected_thresholds.json",
        "resolved_config.json",
        "artifact_manifest.json",
    ]

    missing = []
    for f in required_files:
        if not (output_dir / f).exists():
            missing.append(f)

    if missing:
        errors.append(f"Missing required files: {missing}")
    else:
        print("✓ All required artifact files present")

    # Validate parquet has no NaN/Inf
    df = pd.read_parquet(output_dir / "threshold_search_results.parquet")
    for col in ["total_cost", "threshold"]:
        if df[col].isna().any():
            errors.append(f"NaN found in {col}")
        if not all(np.isfinite(df[col].dropna())):
            errors.append(f"Inf found in {col}")

    if not errors:
        print("✓ Parquet numeric validation passed")

    # Validate JSON has no NaN/Inf
    with open(output_dir / "selected_thresholds.json") as f:
        json_data = json.load(f)
    json_str = json.dumps(json_data)
    if "NaN" in json_str or "Infinity" in json_str or "nan" in json_str.lower():
        errors.append("NaN or Inf found in JSON")
    else:
        print("✓ JSON numeric validation passed")

    # Validate manifest SHA256 values match
    with open(output_dir / "artifact_manifest.json") as f:
        manifest = json.load(f)

    manifest_valid = True
    for entry in manifest.get("files", []):
        file_path = output_dir / entry["path"]
        if file_path.exists():
            actual_hash = compute_sha256(file_path)
            if actual_hash != entry["sha256"]:
                errors.append(f"SHA256 mismatch for {entry['path']}")
                manifest_valid = False

    if manifest_valid:
        print("✓ Manifest SHA256 values match")

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
    output_dir = Path(__file__).parent.parent / "results" / "milestone3" / f"mini_search_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Run mini search
    (
        candidates,
        selected_thresholds,
        metadata,
        episode_rows,
    ) = run_mini_search(output_dir)

    # Write results
    print("\n" + "=" * 60)
    print("WRITING ARTIFACTS")
    print("=" * 60)

    # Persist the genuine episode outputs produced above. The formal validator
    # will still reject this mini artifact because its exact identity universe
    # is intentionally much smaller than formal closeout.
    write_threshold_search_results(episode_rows, output_dir)
    print(
        f"✓ Wrote {len(episode_rows)} genuine tuning episodes to "
        "threshold_search_results.parquet"
    )

    # Write summary
    write_threshold_search_summary(candidates, output_dir)
    print("✓ Wrote threshold_search_summary.csv")

    # Write selected thresholds
    # Convert selected_thresholds dict values to SelectedThreshold objects
    from src.baselines.tuning import SelectedThreshold
    selected_objs: Dict[str, SelectedThreshold] = {}
    for key, val in selected_thresholds.items():
        selected_objs[key] = SelectedThreshold(
            policy_family=val["policy_family"],
            threshold=val["threshold"],
            k_capacity=val["k_capacity"],
            cost_regime_id=val["cost_regime_id"],
            mean_total_cost=val["mean_total_cost"],
            total_failures=val["total_failures"],
            mean_wasted_life_cost=val["mean_wasted_life_cost"],
            episode_count=val["episode_count"],
            tie_break_reason=val.get("tie_break_reason", "best"),
        )
    write_selected_thresholds(selected=selected_objs, output_dir=output_dir)
    print("✓ Wrote selected_thresholds.json")

    # Write resolved config
    write_resolved_config(metadata, output_dir)
    print("✓ Wrote resolved_config.json")

    # Write artifact manifest
    write_artifact_manifest(output_dir)
    print("✓ Wrote artifact_manifest.json")

    # Validate
    valid = validate_mini_search_results(candidates, output_dir)

    if not valid:
        print("\n" + "=" * 60)
        print("VALIDATION FAILED")
        print("=" * 60)
        return 1

    print("\n" + "=" * 60)
    print("MINI SEARCH COMPLETE")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  Total candidates: {len(candidates)}")
    print(f"  K values: {K_VALUES}")
    print(f"  Policy families: {THRESHOLD_FAMILIES}")
    print(f"  Cost regime: {COST_REGIME}")
    print(f"  Output: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
