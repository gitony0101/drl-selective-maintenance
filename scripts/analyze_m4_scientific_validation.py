#!/usr/bin/env python3
"""
Analyze M4 Scientific Validation results.

Implements the exact frozen selection rule from the protocol:
- Loads all completed candidate runs
- Requires identical paired scenario identity
- Rejects mixed HEADs, mixed bank hashes, missing episodes
- Computes paired differences
- Runs stratified paired bootstrap (10,000 resamples, seed 652104)
- Evaluates eligibility
- Applies frozen tie-breaking
- Produces deterministic decision
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Frozen protocol constants
PROTOCOL_VERSION = "m4_scientific_validation_v1"
CANDIDATES = [
    "hard_window_v1",
    "logistic_T1",
    "logistic_T2",
    "logistic_T5",
    "logistic_T10",
    "logistic_T20",
]
K_VALUES = [1, 2]
COST_REGIMES = [
    "failure-heavy-no-waste",
    "failure-heavy-waste-aware",
    "failure-light-no-waste",
    "failure-light-waste-aware",
]
SPLITS = ["predictor_train", "rl_validation"]
SELECTION_SPLIT = "rl_validation"
BOOTSTRAP_SEED = 652104
BOOTSTRAP_RESAMPLES = 10000
ELIGIBILITY_WORSEN_THRESHOLD = 0.10  # 10%
ELIGIBILITY_MAX_WORSEN_CONFIGS = 2

# 8 configurations for selection (2 K × 4 regimes on rl_validation)
SELECTION_CONFIGS = [
    (K, regime) for K in K_VALUES for regime in COST_REGIMES
]


class AnalysisError(Exception):
    """Raised when analysis fails."""
    pass


def load_candidate_results(candidate_dir: Path) -> Dict[str, Any]:
    """Load all artifacts for a candidate."""
    artifacts = {}

    # Load episode metrics
    episode_metrics_path = candidate_dir / "episode_metrics.json"
    if episode_metrics_path.exists():
        with open(episode_metrics_path) as f:
            artifacts["episode_metrics"] = json.load(f)
    else:
        raise AnalysisError(f"Missing episode_metrics.json in {candidate_dir}")

    # Load aggregate metrics
    aggregate_path = candidate_dir / "aggregate_metrics.json"
    if aggregate_path.exists():
        with open(aggregate_path) as f:
            artifacts["aggregate_metrics"] = json.load(f)
    else:
        raise AnalysisError(f"Missing aggregate_metrics.json in {candidate_dir}")

    # Load smoke report
    smoke_path = candidate_dir / "smoke_report.json"
    if smoke_path.exists():
        with open(smoke_path) as f:
            artifacts["smoke_report"] = json.load(f)
    else:
        raise AnalysisError(f"Missing smoke_report.json in {candidate_dir}")

    # Load config
    config_path = candidate_dir / "resolved_config.json"
    if config_path.exists():
        with open(config_path) as f:
            artifacts["config"] = json.load(f)
    else:
        raise AnalysisError(f"Missing resolved_config.json in {candidate_dir}")

    # Load candidate status
    status_path = candidate_dir / "candidate_status.json"
    if status_path.exists():
        with open(status_path) as f:
            artifacts["status"] = json.load(f)
    else:
        raise AnalysisError(f"Missing candidate_status.json in {candidate_dir}")

    return artifacts


def extract_episode_costs(
    artifacts: Dict[str, Any],
    split: str,
    K: int,
    regime: str,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract per-episode total environment costs for a specific configuration.

    Returns (costs_array, scenario_ids) where costs_array is ordered by scenario_id.
    """
    config_key = f"{split}_K{K}_{regime}"
    episodes = artifacts["episode_metrics"].get("episodes", [])

    config_episodes = [
        ep for ep in episodes if ep.get("config_key") == config_key
    ]

    if not config_episodes:
        raise AnalysisError(f"No episodes found for {config_key}")

    # Sort by scenario_id for deterministic pairing
    config_episodes.sort(key=lambda x: x["scenario_id"])

    costs = np.array([ep["total_cost"] for ep in config_episodes], dtype=np.float64)
    scenario_ids = [ep["scenario_id"] for ep in config_episodes]

    return costs, scenario_ids


def validate_paired_identity(
    hard_artifacts: Dict[str, Any],
    candidate_artifacts: Dict[str, Any],
    split: str,
    K: int,
    regime: str,
) -> None:
    """Validate that two candidates have identical paired scenario IDs for a config."""
    config_key = f"{split}_K{K}_{regime}"

    hard_eps = hard_artifacts["episode_metrics"].get("episodes", [])
    cand_eps = candidate_artifacts["episode_metrics"].get("episodes", [])

    hard_scenarios = sorted([ep["scenario_id"] for ep in hard_eps if ep.get("config_key") == config_key])
    cand_scenarios = sorted([ep["scenario_id"] for ep in cand_eps if ep.get("config_key") == config_key])

    if hard_scenarios != cand_scenarios:
        raise AnalysisError(
            f"Paired scenario mismatch for {config_key}: "
            f"hard={hard_scenarios}, candidate={cand_scenarios}"
        )


def validate_bank_hashes(
    hard_artifacts: Dict[str, Any],
    candidate_artifacts: Dict[str, Any],
) -> None:
    """Validate that both candidates used the same bank hashes."""
    hard_hashes = hard_artifacts["config"].get("scenario_bank_sha256_values", {})
    cand_hashes = candidate_artifacts["config"].get("scenario_bank_sha256_values", {})

    if hard_hashes != cand_hashes:
        raise AnalysisError("Bank hashes differ between candidates")


def validate_prediction_cache_hash(
    hard_artifacts: Dict[str, Any],
    candidate_artifacts: Dict[str, Any],
) -> None:
    """Validate same prediction cache."""
    hard_cache = hard_artifacts["config"].get("prediction_cache_sha256")
    cand_cache = candidate_artifacts["config"].get("prediction_cache_sha256")

    if hard_cache != cand_cache:
        raise AnalysisError("Prediction cache hash differs between candidates")


def validate_git_head(
    hard_artifacts: Dict[str, Any],
    candidate_artifacts: Dict[str, Any],
) -> None:
    """Validate same git HEAD."""
    hard_head = hard_artifacts["config"].get("git_commit")
    cand_head = candidate_artifacts["config"].get("git_commit")

    if hard_head != cand_head:
        raise AnalysisError(f"Git HEAD mismatch: hard={hard_head}, candidate={cand_head}")


def validate_protocol_hash(
    hard_artifacts: Dict[str, Any],
    candidate_artifacts: Dict[str, Any],
) -> None:
    """Validate same protocol hash."""
    hard_protocol = hard_artifacts["config"].get("protocol_file_sha256")
    cand_protocol = candidate_artifacts["config"].get("protocol_file_sha256")

    if hard_protocol != cand_protocol:
        raise AnalysisError("Protocol hash differs between candidates")


def compute_paired_differences(
    hard_costs: np.ndarray,
    candidate_costs: np.ndarray,
) -> np.ndarray:
    """Compute paired differences (candidate - hard)."""
    if len(hard_costs) != len(candidate_costs):
        raise AnalysisError(f"Episode count mismatch: hard={len(hard_costs)}, candidate={len(candidate_costs)}")
    return candidate_costs - hard_costs


def normalize_differences(
    differences: np.ndarray,
    hard_mean_cost: float,
) -> np.ndarray:
    """Normalize differences by max(|hard_mean|, 1e-9)."""
    divisor = max(abs(hard_mean_cost), 1e-9)
    return differences / divisor


def stratified_paired_bootstrap(
    config_differences: Dict[str, np.ndarray],
    config_hard_means: Dict[str, float],
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """
    Run stratified paired bootstrap.

    Stratification: within each of the 8 configurations.
    """
    np.random.seed(seed)

    config_names = list(config_differences.keys())
    n_configs = len(config_names)

    # For each config, we have paired differences
    config_data = {}
    for name in config_names:
        diffs = config_differences[name]
        hard_mean = config_hard_means[name]
        config_data[name] = {
            "differences": diffs,
            "hard_mean": hard_mean,
            "n_episodes": len(diffs),
        }

    # Bootstrap resamples
    bootstrap_macro_estimates = []
    bootstrap_per_config = {name: [] for name in config_names}

    for _ in range(n_resamples):
        config_estimates = []

        for name in config_names:
            data = config_data[name]
            diffs = data["differences"]
            n = data["n_episodes"]

            # Resample indices with replacement
            indices = np.random.choice(n, size=n, replace=True)
            resampled_diffs = diffs[indices]

            # Normalize by original hard_mean (fixed)
            hard_mean = data["hard_mean"]
            normalized = resampled_diffs / max(abs(hard_mean), 1e-9)

            # Mean for this config
            config_mean = np.mean(normalized)
            config_estimates.append(config_mean)
            bootstrap_per_config[name].append(config_mean)

        # Macro-average across 8 configs
        macro_estimate = np.mean(config_estimates)
        bootstrap_macro_estimates.append(macro_estimate)

    # Compute percentiles
    macro_ci_lower = np.percentile(bootstrap_macro_estimates, 2.5)
    macro_ci_upper = np.percentile(bootstrap_macro_estimates, 97.5)

    # Per-config CIs
    per_config_cis = {}
    for name in config_names:
        samples = bootstrap_per_config[name]
        per_config_cis[name] = {
            "mean": np.mean(samples),
            "ci_lower": np.percentile(samples, 2.5),
            "ci_upper": np.percentile(samples, 97.5),
            "std": np.std(samples),
        }

    return {
        "macro_estimate": np.mean(bootstrap_macro_estimates),
        "macro_ci_lower": macro_ci_lower,
        "macro_ci_upper": macro_ci_upper,
        "macro_std": np.std(bootstrap_macro_estimates),
        "per_config": per_config_cis,
        "n_resamples": n_resamples,
        "seed": seed,
    }


def evaluate_eligibility(
    candidate_id: str,
    bootstrap_results: Dict[str, Any],
    config_differences: Dict[str, np.ndarray],
    config_hard_means: Dict[str, float],
) -> Dict[str, Any]:
    """Evaluate if a logistic candidate is eligible per frozen rule."""

    # Rule 1: Upper bound of 95% CI for macro-average < 0
    macro_ci_upper = bootstrap_results["macro_ci_upper"]
    rule1_pass = macro_ci_upper < 0

    # Rule 2: Does not worsen mean total cost by >10% in more than 2 of 8 configs
    worsen_count = 0
    config_worsen = {}
    for name in config_differences:
        hard_mean = config_hard_means[name]
        if hard_mean == 0:
            # If hard mean is 0, any positive difference is infinite worsening
            # But we'll handle: if candidate is also 0, no worsening; else worsening
            candidate_mean = np.mean(config_differences[name]) + hard_mean
            if candidate_mean > 0:
                worsen_count += 1
                config_worsen[name] = float('inf')
            else:
                config_worsen[name] = 0.0
        else:
            candidate_mean = np.mean(config_differences[name]) + hard_mean
            relative_change = (candidate_mean - hard_mean) / abs(hard_mean)
            if relative_change > ELIGIBILITY_WORSEN_THRESHOLD:
                worsen_count += 1
            config_worsen[name] = relative_change

    rule2_pass = worsen_count <= ELIGIBILITY_MAX_WORSEN_CONFIGS

    # Rule 3: All production and provenance checks pass (already validated above)
    rule3_pass = True

    eligible = rule1_pass and rule2_pass and rule3_pass

    return {
        "candidate_id": candidate_id,
        "eligible": eligible,
        "rule1_ci_upper_below_zero": rule1_pass,
        "rule1_macro_ci_upper": macro_ci_upper,
        "rule2_worsen_count": worsen_count,
        "rule2_max_worsen_configs": ELIGIBILITY_MAX_WORSEN_CONFIGS,
        "rule2_threshold": ELIGIBILITY_WORSEN_THRESHOLD,
        "rule2_config_worsen": config_worsen,
        "rule2_pass": rule2_pass,
        "rule3_pass": rule3_pass,
    }


def apply_tie_breaking(eligible_candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Apply frozen tie-breaking to select among eligible candidates."""

    if not eligible_candidates:
        return None

    if len(eligible_candidates) == 1:
        return eligible_candidates[0]

    # Sort by tie-breaking criteria:
    # 1. Lowest primary macro-average normalized paired cost difference
    # 2. Lowest worst-configuration normalized difference
    # 3. Lowest mean preventive-maintenance cost
    # 4. Lower temperature

    def get_tie_breaker(c):
        macro = c.get("macro_estimate", float('inf'))
        # Worst config = max per-config mean
        per_config = c.get("bootstrap", {}).get("per_config", {})
        worst_config = max(pc["mean"] for pc in per_config.values()) if per_config else float('inf')
        # Mean preventive cost - need to extract from artifacts
        mean_preventive = c.get("mean_preventive_cost", float('inf'))
        # Temperature from candidate_id
        if c["candidate_id"] == "hard_window_v1":
            temp = float('inf')
        else:
            temp_str = c["candidate_id"].replace("logistic_T", "")
            temp = float(temp_str)

        return (macro, worst_config, mean_preventive, temp)

    eligible_candidates.sort(key=get_tie_breaker)
    return eligible_candidates[0]


def analyze_all_candidates(results_root: Path) -> Dict[str, Any]:
    """Run complete analysis on all candidates."""

    # Load all candidate artifacts
    candidate_artifacts = {}
    for candidate_id in CANDIDATES:
        candidate_dir = results_root / candidate_id
        if not candidate_dir.exists():
            raise AnalysisError(f"Candidate directory not found: {candidate_dir}")

        try:
            artifacts = load_candidate_results(candidate_dir)
            if artifacts["status"].get("status") != "completed":
                raise AnalysisError(f"Candidate {candidate_id} not completed: {artifacts['status'].get('status')}")
            candidate_artifacts[candidate_id] = artifacts
            print(f"  Loaded {candidate_id}")
        except Exception as e:
            raise AnalysisError(f"Failed to load {candidate_id}: {e}")

    # Verify all candidates have same protocol, banks, cache, HEAD
    hard_artifacts = candidate_artifacts["hard_window_v1"]
    for candidate_id in CANDIDATES[1:]:
        cand_artifacts = candidate_artifacts[candidate_id]
        validate_bank_hashes(hard_artifacts, cand_artifacts)
        validate_prediction_cache_hash(hard_artifacts, cand_artifacts)
        validate_git_head(hard_artifacts, cand_artifacts)
        validate_protocol_hash(hard_artifacts, cand_artifacts)

    print("  ✓ All provenance checks passed")

    # Extract per-config costs for each candidate
    candidate_costs = {}
    config_scenario_ids = None

    for candidate_id in CANDIDATES:
        artifacts = candidate_artifacts[candidate_id]
        costs_by_config = {}
        scenario_ids_by_config = {}

        for K in K_VALUES:
            for regime in COST_REGIMES:
                config_key = f"{SELECTION_SPLIT}_K{K}_{regime}"
                costs, scenario_ids = extract_episode_costs(
                    artifacts, SELECTION_SPLIT, K, regime
                )
                costs_by_config[config_key] = costs
                scenario_ids_by_config[config_key] = scenario_ids

                # Validate paired identity against hard comparator
                if candidate_id != "hard_window_v1":
                    validate_paired_identity(hard_artifacts, artifacts, SELECTION_SPLIT, K, regime)

        candidate_costs[candidate_id] = costs_by_config

        # Store scenario IDs from first candidate (should be same for all)
        if config_scenario_ids is None:
            config_scenario_ids = scenario_ids_by_config

    print("  ✓ Paired identity validated across all candidates")

    # Compute paired differences for each logistic candidate
    hard_costs = candidate_costs["hard_window_v1"]

    results = {
        "protocol_version": PROTOCOL_VERSION,
        "selection_split": SELECTION_SPLIT,
        "selection_configs": SELECTION_CONFIGS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "candidates": {},
    }

    for candidate_id in CANDIDATES[1:]:  # Skip hard_window_v1 (comparator)
        costs = candidate_costs[candidate_id]

        config_differences = {}
        config_hard_means = {}
        config_candidate_means = {}
        config_normalized_diffs = {}

        for (K, regime) in SELECTION_CONFIGS:
            config_name = f"rl_validation_K{K}_{regime}"
            hard_c = hard_costs[config_name]
            cand_c = costs[config_name]

            diffs = compute_paired_differences(hard_c, cand_c)
            hard_mean = float(np.mean(hard_c))
            cand_mean = float(np.mean(cand_c))
            normalized = normalize_differences(diffs, hard_mean)

            config_differences[config_name] = diffs
            config_hard_means[config_name] = hard_mean
            config_candidate_means[config_name] = cand_mean
            config_normalized_diffs[config_name] = normalized

        # Macro-average normalized paired difference
        macro_avg = float(np.mean([np.mean(config_normalized_diffs[k]) for k in config_normalized_diffs]))

        # Worst configuration
        worst_config = max(config_normalized_diffs.keys(),
                          key=lambda k: np.mean(config_normalized_diffs[k]))
        worst_value = float(np.mean(config_normalized_diffs[worst_config]))

        # Bootstrap
        bootstrap_results = stratified_paired_bootstrap(
            config_differences, config_hard_means
        )

        # Extract mean preventive cost (secondary metric for tie-breaking)
        artifacts = candidate_artifacts[candidate_id]
        mean_preventive = float(
            np.mean([ep["preventive_cost"] for ep in artifacts["episode_metrics"]["episodes"]
                     if ep.get("config_key", "").startswith(SELECTION_SPLIT)])
        )

        candidate_results = {
            "candidate_id": candidate_id,
            "macro_avg_normalized_paired_diff": macro_avg,
            "worst_config": worst_config,
            "worst_config_value": worst_value,
            "per_config_normalized_diff": {k: float(np.mean(v)) for k, v in config_normalized_diffs.items()},
            "per_config_hard_mean": config_hard_means,
            "per_config_candidate_mean": config_candidate_means,
            "per_config_paired_diff_mean": {k: float(np.mean(v)) for k, v in config_differences.items()},
            "mean_preventive_cost": mean_preventive,
            "bootstrap": bootstrap_results,
        }

        results["candidates"][candidate_id] = candidate_results

        print(f"\n  {candidate_id}:")
        print(f"    Macro avg normalized diff: {macro_avg:.6f}")
        print(f"    95% CI: [{bootstrap_results['macro_ci_lower']:.6f}, {bootstrap_results['macro_ci_upper']:.6f}]")
        print(f"    Worst config: {worst_config} ({worst_value:.6f})")
        print(f"    Mean preventive cost: {mean_preventive:.2f}")

    # Evaluate eligibility
    eligible = []
    for candidate_id, cand_results in results["candidates"].items():
        eligibility = evaluate_eligibility(
            candidate_id,
            cand_results["bootstrap"],
            {k: cand_results["per_config_paired_diff_mean"][k] for k in cand_results["per_config_paired_diff_mean"]},
            cand_results["per_config_hard_mean"],
        )
        cand_results["eligibility"] = eligibility

        if eligibility["eligible"]:
            eligible.append(cand_results)
            print(f"\n  ✓ {candidate_id} ELIGIBLE")
        else:
            print(f"\n  ✗ {candidate_id} NOT ELIGIBLE")
            print(f"    Rule 1 (CI upper < 0): {eligibility['rule1_ci_upper_below_zero']} (CI upper: {eligibility['rule1_macro_ci_upper']:.6f})")
            print(f"    Rule 2 (worsen ≤ 2): {eligibility['rule2_pass']} (worsen count: {eligibility['rule2_worsen_count']})")

    # Apply tie-breaking
    selected = apply_tie_breaking(eligible)

    if selected:
        decision = {
            "decision": "select_logistic",
            "selected_candidate": selected["candidate_id"],
            "eligible_candidates": [c["candidate_id"] for c in eligible],
            "tie_breaking_applied": len(eligible) > 1,
        }
        print(f"\n✓ SELECTED: {selected['candidate_id']}")
    else:
        decision = {
            "decision": "retain_hard_window_v1",
            "selected_candidate": "hard_window_v1",
            "eligible_candidates": [],
            "reason": "No logistic candidate satisfied eligibility criteria",
        }
        print(f"\n✓ RETAINED: hard_window_v1 (no eligible logistic candidate)")

    results["decision"] = decision

    return results


def write_analysis_outputs(
    results: Dict[str, Any],
    output_root: Path,
) -> Dict[str, Path]:
    """Write all analysis outputs."""

    written = {}

    # 1. selection_decision.json
    decision_path = output_root / "selection_decision.json"
    with open(decision_path, 'w') as f:
        json.dump(results["decision"], f, indent=2)
    written["selection_decision"] = decision_path

    # 2. paired_episode_metrics.csv
    # We need to create this from the raw episode data
    # For now, write a placeholder
    paired_csv_path = output_root / "paired_episode_metrics.csv"
    # TODO: Implement full CSV export
    with open(paired_csv_path, 'w') as f:
        f.write("# Paired episode metrics - placeholder\n")
    written["paired_episode_metrics"] = paired_csv_path

    # 3. candidate_summary.csv
    summary_rows = []
    for candidate_id, cand_results in results["candidates"].items():
        row = {
            "candidate_id": candidate_id,
            "macro_avg_normalized_paired_diff": cand_results["macro_avg_normalized_paired_diff"],
            "macro_ci_lower": cand_results["bootstrap"]["macro_ci_lower"],
            "macro_ci_upper": cand_results["bootstrap"]["macro_ci_upper"],
            "worst_config": cand_results["worst_config"],
            "worst_config_value": cand_results["worst_config_value"],
            "mean_preventive_cost": cand_results["mean_preventive_cost"],
            "eligible": cand_results["eligibility"]["eligible"],
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_root / "candidate_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    written["candidate_summary"] = summary_path

    # 4. per_configuration_summary.csv
    config_rows = []
    for candidate_id, cand_results in results["candidates"].items():
        for config_name in cand_results["per_config_normalized_diff"]:
            row = {
                "candidate_id": candidate_id,
                "config": config_name,
                "normalized_paired_diff_mean": cand_results["per_config_normalized_diff"][config_name],
                "hard_mean": cand_results["per_config_hard_mean"][config_name],
                "candidate_mean": cand_results["per_config_candidate_mean"][config_name],
                "paired_diff_mean": cand_results["per_config_paired_diff_mean"][config_name],
                "bootstrap_ci_lower": cand_results["bootstrap"]["per_config"][config_name]["ci_lower"],
                "bootstrap_ci_upper": cand_results["bootstrap"]["per_config"][config_name]["ci_upper"],
            }
            config_rows.append(row)

    config_df = pd.DataFrame(config_rows)
    config_path = output_root / "per_configuration_summary.csv"
    config_df.to_csv(config_path, index=False)
    written["per_configuration_summary"] = config_path

    # 5. bootstrap_summary.json
    bootstrap_summary = {
        "seed": BOOTSTRAP_SEED,
        "resamples": BOOTSTRAP_RESAMPLES,
        "candidates": {},
    }
    for candidate_id, cand_results in results["candidates"].items():
        bootstrap_summary["candidates"][candidate_id] = cand_results["bootstrap"]
    bootstrap_path = output_root / "bootstrap_summary.json"
    with open(bootstrap_path, 'w') as f:
        json.dump(bootstrap_summary, f, indent=2)
    written["bootstrap_summary"] = bootstrap_path

    # 6. validation_analysis_manifest.json
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "analysis_timestamp": pd.Timestamp.now().isoformat(),
        "git_head": results.get("git_head", "unknown"),
        "selection_split": SELECTION_SPLIT,
        "decision": results["decision"],
        "artifacts": {k: str(v) for k, v in written.items()},
    }
    manifest_path = output_root / "validation_analysis_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    written["validation_analysis_manifest"] = manifest_path

    return written


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="analyze_m4_scientific_validation",
        description="Analyze M4 scientific validation results",
    )

    parser.add_argument(
        "--results-root",
        type=str,
        default=None,
        help="Results root directory (default: results/milestone4/scientific_validation_v1/)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for analysis (default: same as results-root)",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    if args.results_root:
        results_root = Path(args.results_root)
    else:
        results_root = repo_root / "results" / "milestone4" / "scientific_validation_v1"

    if args.output_dir:
        output_root = Path(args.output_dir)
    else:
        output_root = results_root

    if not results_root.exists():
        print(f"ERROR: Results root not found: {results_root}", file=sys.stderr)
        return 1

    print(f"Analyzing results from: {results_root}")
    print(f"Writing analysis to: {output_root}")

    try:
        results = analyze_all_candidates(results_root)
        results["git_head"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=repo_root
        ).stdout.strip()

        written = write_analysis_outputs(results, output_root)

        print(f"\nAnalysis complete. Outputs:")
        for name, path in written.items():
            print(f"  {name}: {path}")

        return 0

    except AnalysisError as e:
        print(f"\n✗ Analysis failed: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import subprocess
    sys.exit(main())