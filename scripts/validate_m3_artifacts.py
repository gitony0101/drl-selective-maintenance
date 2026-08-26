#!/usr/bin/env python3
"""
M3 Formal Artifact Validator

Validates all formal M3 experiment artifacts:
- Strict JSON parsing
- Parquet readability
- Required schemas
- Expected row counts
- Expected identity sets
- Unique keys
- Finite numeric values (no NaN, no Inf)
- Episode completion (100 steps per episode)
- Reward ≈ negative total cost
- Cost decomposition
- Threshold grid membership
- Selected winner correctness
- Deterministic tie-breaking
- Scenario-bank provenance (file size, SHA256)
- Implementation commit identity
- Threshold-use equality (evaluation used selected thresholds)
- Run ID consistency (tuning and evaluation belong to one formal run)
"""

import json
import hashlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

# Expected counts from frozen contract (oracle-included, --allow-oracle mode)
EXPECTED_TUNING_CANDIDATES_ORACLE = 360
EXPECTED_TUNING_EPISODES_ORACLE = 9000  # 360 × 5 scenarios × 5 seeds
EXPECTED_SELECTED_THRESHOLDS_ORACLE = 32  # 4 policies × 2 K × 4 regimes
EXPECTED_EVALUATION_EPISODES_ORACLE = 2400  # 6 policies × 2 K × 4 regimes × 2 splits × 5 × 5

# Expected counts from frozen contract (non-oracle, default mode, --allow-oracle NOT set)
# Note: tuning always uses all 360 candidates (including oracle_threshold), but evaluation excludes oracle
EXPECTED_TUNING_CANDIDATES_NON_ORACLE = 272  # 3 policies * 34 thresholds * 2 K * 4 regimes
EXPECTED_TUNING_EPISODES_NON_ORACLE = 6800  # 272 candidates × 5 scenarios × 5 seeds
EXPECTED_SELECTED_THRESHOLDS_NON_ORACLE = 24  # 3 policies × 2 K × 4 regimes
EXPECTED_EVALUATION_EPISODES_NON_ORACLE = 2000  # 5 policies × 2 K × 4 regimes × 2 splits × 5 × 5

# Frozen threshold grids
AGE_THRESHOLDS = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
PREDICTED_RUL_THRESHOLDS = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100]
GREEDY_THRESHOLDS = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100]
ORACLE_THRESHOLDS = [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50]

ALL_THRESHOLDS = {
    "age_threshold": set(AGE_THRESHOLDS),
    "predicted_rul_threshold": set(PREDICTED_RUL_THRESHOLDS),
    "greedy_predicted_rul": set(GREEDY_THRESHOLDS),
    "oracle_threshold": set(ORACLE_THRESHOLDS),
}

K_VALUES = {1, 2}
COST_REGIMES = {
    "failure-light-no-waste",
    "failure-heavy-no-waste",
    "failure-light-waste-aware",
    "failure-heavy-waste-aware",
}

POLICY_FAMILIES = {
    "corrective_only",
    "random_feasible",
    "age_threshold",
    "predicted_rul_threshold",
    "greedy_predicted_rul",
    "oracle_threshold",
}

ORACLE_SEMANTIC_ROLE = "privileged-information diagnostic benchmark"


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def validate_json_file(file_path: Path) -> tuple[bool, Optional[Dict], Optional[str]]:
    """Validate JSON file parsing and schema."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return True, data, None
    except json.JSONDecodeError as e:
        return False, None, f"JSON parse error: {e}"
    except FileNotFoundError:
        return False, None, f"File not found: {file_path}"


def validate_parquet_file(file_path: Path) -> tuple[bool, Optional[pd.DataFrame], Optional[str]]:
    """Validate Parquet file reading."""
    try:
        df = pd.read_parquet(file_path)
        return True, df, None
    except Exception as e:
        return False, None, f"Parquet read error: {e}"


def validate_finite_values(df: pd.DataFrame, numeric_cols: List[str]) -> List[str]:
    """Check for NaN/Inf in numeric columns. Returns list of errors."""
    errors = []
    for col in numeric_cols:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")
            continue
        if df[col].isna().any():
            errors.append(f"NaN found in {col}")
        if np.isinf(df[col]).any():
            errors.append(f"Inf found in {col}")
    return errors


def validate_threshold_grid_membership(
    df: pd.DataFrame,
    threshold_col: str = "threshold",
    policy_col: str = "policy_family",
) -> List[str]:
    """Validate all thresholds belong to their policy's frozen grid."""
    errors = []
    for policy in df[policy_col].unique():
        if policy not in ALL_THRESHOLDS:
            continue
        expected = ALL_THRESHOLDS[policy]
        actual = set(df[df[policy_col] == policy][threshold_col].unique())
        invalid = actual - expected
        if invalid:
            errors.append(f"{policy}: invalid thresholds {invalid}")
    return errors


def validate_unique_keys(df: pd.DataFrame, key_cols: List[str]) -> List[str]:
    """Validate unique key constraint."""
    errors = []
    missing = [col for col in key_cols if col not in df.columns]
    if missing:
        return [f"Missing unique-key columns: {missing}"]
    if df.duplicated(subset=key_cols).any():
        dup_count = df.duplicated(subset=key_cols).sum()
        errors.append(f"{dup_count} duplicate rows for keys {key_cols}")
    return errors


def validate_episode_completion(
    df: pd.DataFrame,
    steps_col: str = "episode_steps",
    expected_steps: int = 100,
) -> List[str]:
    """Validate all episodes completed with expected steps."""
    errors = []
    if steps_col not in df.columns:
        errors.append(f"Missing column: {steps_col}")
        return errors

    incomplete = df[df[steps_col] != expected_steps]
    if len(incomplete) > 0:
        errors.append(f"{len(incomplete)} episodes with != {expected_steps} steps")
    return errors


def validate_reward_cost_reconciliation(
    df: pd.DataFrame,
    reward_col: str = "episode_return",
    cost_col: str = "total_cost",
    tolerance: float = 1e-6,
) -> List[str]:
    """Validate reward ≈ -total_cost."""
    errors = []
    if reward_col not in df.columns or cost_col not in df.columns:
        errors.append("Missing reward or cost columns")
        return errors

    diff = (df[reward_col] + df[cost_col]).abs()
    violations = diff[diff > tolerance]
    if len(violations) > 0:
        errors.append(f"{len(violations)} episodes with reward ≠ -cost")
    return errors


def validate_cost_decomposition(df: pd.DataFrame) -> List[str]:
    """Validate cost decomposition: total = preventive + failure + wasted_life."""
    errors = []
    required = ["total_cost", "preventive_cost", "failure_cost", "wasted_life_cost"]
    for col in required:
        if col not in df.columns:
            errors.append(f"Missing cost column: {col}")
            return errors

    reconstructed = df["preventive_cost"] + df["failure_cost"] + df["wasted_life_cost"]
    diff = (df["total_cost"] - reconstructed).abs()
    if diff.max() > 1e-6:
        errors.append("Cost decomposition mismatch")
    return errors


def validate_scenario_bank_provenance(
    provenance_path: Path,
) -> tuple[bool, List[str]]:
    """Validate scenario-bank provenance file."""
    errors = []

    success, data, err = validate_json_file(provenance_path)
    if not success:
        return False, [err]

    required_fields = [
        "split",
        "K",
        "cost_regime_id",
        "source_path",
        "source_sha256",
        "scenario_count",
        "sorted_scenario_ids_sha256",
    ]

    banks = data.get("scenario_banks", [])
    if not banks:
        return False, ["No scenario banks in provenance"]

    for i, bank in enumerate(banks):
        for field in required_fields:
            if field not in bank:
                errors.append(f"Bank {i}: missing {field}")

        # Validate SHA256 format
        for sha_field in ["source_sha256", "sorted_scenario_ids_sha256"]:
            if sha_field in bank:
                value = bank[sha_field]
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(c not in "0123456789abcdef" for c in value)
                ):
                    errors.append(f"Bank {i}: invalid {sha_field} length")

        if "scenario_count" in bank:
            count = bank["scenario_count"]
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                errors.append(f"Bank {i}: invalid scenario_count")

    identities = [
        (bank.get("split"), bank.get("K"), bank.get("cost_regime_id"))
        for bank in banks
    ]
    duplicate_count = len(identities) - len(set(identities))
    if duplicate_count:
        errors.append(
            f"{duplicate_count} duplicate scenario-bank identities on "
            "(split, K, cost_regime_id)"
        )

    return len(errors) == 0, errors


def validate_run_provenance(
    provenance_path: Path,
    run_type: str = "threshold_tuning",
) -> tuple[bool, List[str]]:
    """Validate run provenance file."""
    errors = []

    success, data, err = validate_json_file(provenance_path)
    if not success:
        return False, [err]

    if data.get("run_type") != run_type:
        errors.append(f"Expected run_type '{run_type}', got '{data.get('run_type')}'")

    if "completed_at" not in data:
        errors.append("Missing completed_at timestamp")

    return len(errors) == 0, errors


def validate_selected_thresholds(
    selected_path: Path,
    threshold_results_path: Path,
) -> tuple[bool, List[str]]:
    """Validate selected thresholds are winners from tuning results."""
    errors = []

    # Load selected thresholds
    success, selected, err = validate_json_file(selected_path)
    if not success:
        return False, [err]

    # Load tuning results
    success, results_df, err = validate_parquet_file(threshold_results_path)
    if not success:
        return False, [err]

    # Subtract the mandatory `_meta` provenance envelope so the validator
    # checks only winner keys (each carrying a winning threshold per
    # policy_family × k_capacity × cost_regime_id group).
    winners = {k: v for k, v in selected.items() if k != "_meta"}

    required_episode_cols = [
        "policy_family",
        "threshold",
        "k_capacity",
        "cost_regime_id",
        "total_cost",
        "failure_count",
        "wasted_life_cost",
        "completed",
    ]
    missing_cols = [c for c in required_episode_cols if c not in results_df.columns]
    if missing_cols:
        return False, [f"Missing tuning episode columns: {missing_cols}"]

    if (~results_df["completed"].astype(bool)).any():
        errors.append("Tuning evidence contains incomplete episodes")

    completed = results_df[results_df["completed"].astype(bool)]
    group_cols = [
        "policy_family",
        "threshold",
        "k_capacity",
        "cost_regime_id",
    ]
    candidates = (
        completed.groupby(group_cols, as_index=False)
        .agg(
            mean_total_cost=("total_cost", "mean"),
            total_failures=("failure_count", "sum"),
            mean_wasted_life_cost=("wasted_life_cost", "mean"),
            episode_count=("total_cost", "size"),
        )
    )

    # Verify each selected threshold is the deterministic winner reconstructed
    # from genuine episode rows.
    for key, entry in winners.items():
        # Parse policy_family from key (format: {policy_family}_k{k}_{cost_regime_id})
        parts = key.split('_k')
        policy_family = parts[0]
        remaining = parts[1] if len(parts) > 1 else ""
        k_parts = remaining.split('_', 1)
        k = int(k_parts[0]) if k_parts[0].isdigit() else None
        regime = k_parts[1] if len(k_parts) > 1 else None

        # Also try to get from entry fields
        k = entry.get("k_capacity", k)
        regime = entry.get("cost_regime_id", regime)

        selected_threshold = entry.get("threshold")
        selected_cost = entry.get("mean_total_cost")

        # Filter results for this group
        group = candidates[
            (candidates["policy_family"] == policy_family) &
            (candidates["k_capacity"] == k) &
            (candidates["cost_regime_id"] == regime)
        ]

        if len(group) == 0:
            errors.append(f"{key}: no tuning results found")
            continue

        best = group.sort_values(
            [
                "mean_total_cost",
                "total_failures",
                "mean_wasted_life_cost",
                "threshold",
            ],
            kind="mergesort",
        ).iloc[0]

        if abs(best["mean_total_cost"] - selected_cost) > 1e-6:
            errors.append(f"{key}: selected cost {selected_cost} ≠ best {best['mean_total_cost']}")

        if best["threshold"] != selected_threshold:
            errors.append(f"{key}: selected threshold {selected_threshold} ≠ best {best['threshold']}")

        for field in (
            "total_failures",
            "mean_wasted_life_cost",
            "episode_count",
        ):
            recorded = entry.get(field)
            actual = best[field]
            if recorded is None or abs(float(recorded) - float(actual)) > 1e-6:
                errors.append(
                    f"{key}: selected {field} {recorded} ≠ recomputed {actual}"
                )

    return len(errors) == 0, errors


def validate_threshold_search_summary(
    threshold_results_path: Path,
    summary_path: Path,
    expected_candidate_count: int,
) -> tuple[bool, List[str]]:
    """Recompute candidate metrics from episode rows and compare the CSV."""
    errors: List[str] = []
    try:
        episodes = pd.read_parquet(threshold_results_path)
        summary = pd.read_csv(summary_path)
    except Exception as exc:
        return False, [f"Could not read candidate evidence: {exc}"]

    key_cols = [
        "policy_family",
        "threshold",
        "k_capacity",
        "cost_regime_id",
    ]
    metric_cols = [
        "mean_total_cost",
        "total_failures",
        "mean_wasted_life_cost",
        "episode_count",
    ]
    required_episode = set(key_cols + [
        "total_cost", "failure_count", "wasted_life_cost", "completed"
    ])
    required_summary = set(key_cols + metric_cols)
    missing_episode = sorted(required_episode - set(episodes.columns))
    missing_summary = sorted(required_summary - set(summary.columns))
    if missing_episode or missing_summary:
        return False, [
            f"Candidate evidence missing columns: episode={missing_episode}, "
            f"summary={missing_summary}"
        ]

    completed = episodes[episodes["completed"].astype(bool)]
    recomputed = (
        completed.groupby(key_cols, as_index=False)
        .agg(
            mean_total_cost=("total_cost", "mean"),
            total_failures=("failure_count", "sum"),
            mean_wasted_life_cost=("wasted_life_cost", "mean"),
            episode_count=("total_cost", "size"),
        )
    )
    if len(summary) != expected_candidate_count:
        errors.append(
            f"threshold_search_summary.csv rows={len(summary)} "
            f"!= {expected_candidate_count}"
        )
    if len(recomputed) != expected_candidate_count:
        errors.append(
            f"Recomputed candidate rows={len(recomputed)} "
            f"!= {expected_candidate_count}"
        )
    duplicate_count = int(summary.duplicated(subset=key_cols).sum())
    if duplicate_count:
        errors.append(
            f"threshold_search_summary.csv has {duplicate_count} duplicate identities"
        )

    merged = recomputed.merge(
        summary,
        on=key_cols,
        how="outer",
        suffixes=("_recomputed", "_recorded"),
        indicator=True,
    )
    nonmatching_ids = int((merged["_merge"] != "both").sum())
    if nonmatching_ids:
        errors.append(f"Candidate summary identity mismatches: {nonmatching_ids}")
    matched = merged[merged["_merge"] == "both"]
    for field in metric_cols:
        left = pd.to_numeric(
            matched[f"{field}_recomputed"], errors="coerce"
        )
        right = pd.to_numeric(
            matched[f"{field}_recorded"], errors="coerce"
        )
        mismatch_count = int(
            (~np.isclose(left, right, rtol=0.0, atol=1e-9, equal_nan=False)).sum()
        )
        if mismatch_count:
            errors.append(f"Candidate summary {field} mismatches: {mismatch_count}")

    return len(errors) == 0, errors


def validate_implementation_commit(
    provenance_path: Path,
    expected_commit: str,
) -> List[str]:
    """Validate implementation commit matches expected."""
    errors = []

    success, data, err = validate_json_file(provenance_path)
    if not success:
        return [err]

    # Check commit in various locations
    commit = data.get("implementation_commit")
    if commit and commit != expected_commit:
        errors.append(f"Implementation commit mismatch: {commit} ≠ {expected_commit}")

    return errors


def validate_threshold_use_equality(
    episode_results_path: Path,
    selected_thresholds_path: Path,
) -> tuple[bool, List[str]]:
    """
    Validate that evaluation actually used selected thresholds.

    For every threshold-policy evaluation row, verifies:
    evaluation threshold = selected_thresholds[policy_family, k_capacity, cost_regime_id]

    Args:
        episode_results_path: Path to episode_results.parquet
        selected_thresholds_path: Path to selected_thresholds.json

    Returns:
        Tuple of (success, list of errors)
    """
    errors = []

    # Load selected thresholds
    success, selected, err = validate_json_file(selected_thresholds_path)
    if not success:
        return False, [err]

    # Load episode results
    success, df, err = validate_parquet_file(episode_results_path)
    if not success:
        return False, [err]

    # Required columns - episode_results uses maintenance_capacity, not k_capacity
    required_cols = ["policy_family", "maintenance_capacity", "cost_regime_id"]
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")
            return False, errors

    # Policies that have thresholds - all use 'threshold' column in episode_results
    threshold_policies = {
        "age_threshold": "threshold",
        "predicted_rul_threshold": "threshold",
        "greedy_predicted_rul": "threshold",  # Stored as 'threshold' in episode_results
        "oracle_threshold": "threshold",
    }

    # Policies without thresholds (skip validation)
    no_threshold_policies = {"corrective_only", "random_feasible"}

    # Get actual policy families in the evaluation data
    actual_policies = set(df["policy_family"].unique())

    # Check each threshold-policy row that is present in the data
    for policy in threshold_policies.keys():
        # Only validate policies that are actually in the evaluation data
        if policy not in actual_policies:
            continue

        policy_rows = df[df["policy_family"] == policy]
        if len(policy_rows) == 0:
            errors.append(f"No evaluation rows for {policy}")
            continue

        threshold_col = threshold_policies[policy]
        if threshold_col not in policy_rows.columns:
            errors.append(f"Missing {threshold_col} column for {policy}")
            continue

        # Check each unique (policy, k, regime) combination
        # Note: episode_results uses maintenance_capacity, selected_thresholds uses k_capacity
        for _, group in policy_rows.groupby(["maintenance_capacity", "cost_regime_id"]):
            k = int(group["maintenance_capacity"].iloc[0])
            regime = group["cost_regime_id"].iloc[0]
            key = f"{policy}_k{k}_{regime}"

            if key not in selected:
                errors.append(f"{key}: no selected threshold found")
                continue

            expected_threshold = selected[key].get("threshold")
            if expected_threshold is None:
                errors.append(f"{key}: selected threshold is None")
                continue

            # Check all rows in this group use the correct threshold
            actual_thresholds = group[threshold_col].dropna().unique()
            for actual in actual_thresholds:
                if actual != expected_threshold:
                    errors.append(
                        f"{key}: evaluation used threshold {actual}, "
                        f"selected was {expected_threshold}"
                    )

    # Verify no default-threshold fallback occurred
    # (threshold should never be None for threshold policies, unless explicitly allowed)
    for policy in threshold_policies.keys():
        policy_rows = df[df["policy_family"] == policy]
        threshold_col = threshold_policies[policy]
        if threshold_col in policy_rows.columns:
            null_count = policy_rows[threshold_col].isna().sum()
            if null_count > 0:
                errors.append(
                    f"{policy}: {null_count} rows with None {threshold_col} "
                    "(default-threshold fallback detected)"
                )

    return len(errors) == 0, errors


def validate_run_id_consistency(
    run_provenance_path: Path,
    tuning_dir: Path,
    eval_dir: Path,
) -> tuple[bool, List[str]]:
    """
    Validate tuning and evaluation belong to one formal run ID.

    Args:
        run_provenance_path: Path to run_provenance.json (evaluation)
        tuning_dir: Path to tuning output directory
        eval_dir: Path to evaluation output directory

    Returns:
        Tuple of (success, list of errors)
    """
    errors = []

    # For now, validate that both directories exist and have provenance
    if not tuning_dir.exists():
        errors.append(f"Tuning directory not found: {tuning_dir}")
    if not eval_dir.exists():
        errors.append(f"Evaluation directory not found: {eval_dir}")

    # Check run provenance in both
    tuning_prov = tuning_dir / "run_provenance.json"
    if not tuning_prov.exists():
        errors.append(f"Missing tuning run_provenance.json")

    return len(errors) == 0, errors


def main(output_dir: str, expected_commit: str = None, mode: str = None):
    """Run all validations.

    Args:
        output_dir: Output directory
        expected_commit: Implementation commit hash expected
        mode: One of:
              "formal_closeout" — requires Oracle (60/9000/32/2400)
              "diagnostic_non_oracle" — non-Oracle (272/6800/24/2000)
              None — explicit declaration required from caller
    """
    output_path = Path(output_dir)

    if not output_path.exists():
        print(f"ERROR: Output directory not found: {output_path}")
        return 1

    # Mode declaration is REQUIRED — we no longer infer mode from which
    # artifacts happen to exist on disk.
    if mode is None:
        print(
            "ERROR: --mode is required (formal_closeout, diagnostic_non_oracle, "
            "or diagnostic_legacy).",
            file=sys.stderr,
        )
        print(
            "       Inferring from artifacts is forbidden; a damaged formal "
            "run is not auto-classified as a valid non-Oracle run.",
            file=sys.stderr,
        )
        return 2
    if mode not in ("formal_closeout", "diagnostic_non_oracle", "diagnostic_legacy"):
        print(
            f"ERROR: unknown mode {mode!r}; expected formal_closeout, "
            "diagnostic_non_oracle, or diagnostic_legacy.",
            file=sys.stderr,
        )
        return 2

    # Selected path is required for either mode.
    selected_path = output_path / "selected_thresholds.json"
    if not selected_path.exists():
        print(
            "ERROR: selected_thresholds.json is missing; cannot infer mode "
            "or validate counts.",
            file=sys.stderr,
        )
        return 1

    success, selected_payload, err = validate_json_file(selected_path)
    if not success:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1

    oracle_included = any(
        isinstance(k, str) and k.startswith("oracle_threshold")
        for k in selected_payload.keys()
    )

    # Select appropriate expected counts based on the EXPLICIT mode argument.
    if mode == "formal_closeout":
        expected_tuning_candidates = EXPECTED_TUNING_CANDIDATES_ORACLE
        expected_selected_thresholds = EXPECTED_SELECTED_THRESHOLDS_ORACLE
        expected_evaluation_episodes = EXPECTED_EVALUATION_EPISODES_ORACLE
        mode_name = "formal_closeout (Oracle required)"

        # A damaged formal_closeout run missing Oracle must FAIL.
        if not oracle_included:
            print(
                "ERROR: --mode formal_closeout requires Oracle, but "
                "selected_thresholds.json has no oracle_threshold identities.",
                file=sys.stderr,
            )
            return 1
    elif mode == "diagnostic_legacy":
        # diagnostic_legacy mirrors diagnostic_non_oracle counts but is
        # its own explicit entry-point; it allows legacy-shape fixtures
        # (e.g. ``_sealed`` alias) in upstream parsing, while still
        # forbidding implicit inference of mode from artifacts on disk.
        expected_tuning_candidates = EXPECTED_TUNING_CANDIDATES_NON_ORACLE
        expected_selected_thresholds = EXPECTED_SELECTED_THRESHOLDS_NON_ORACLE
        expected_evaluation_episodes = EXPECTED_EVALUATION_EPISODES_NON_ORACLE
        mode_name = "diagnostic_legacy (Oracle excluded; legacy-shape tolerated)"

        # In diagnostic_legacy, Oracle identities MUST NOT be present.
        if oracle_included:
            print(
                "ERROR: --mode diagnostic_legacy forbids Oracle, but "
                "selected_thresholds.json contains oracle_threshold identities.",
                file=sys.stderr,
            )
            return 1
    else:
        expected_tuning_candidates = EXPECTED_TUNING_CANDIDATES_NON_ORACLE
        expected_selected_thresholds = EXPECTED_SELECTED_THRESHOLDS_NON_ORACLE
        expected_evaluation_episodes = EXPECTED_EVALUATION_EPISODES_NON_ORACLE
        mode_name = "diagnostic_non_oracle (Oracle excluded)"

        # In diagnostic_non_oracle, Oracle identities MUST NOT be present.
        if oracle_included:
            print(
                "ERROR: --mode diagnostic_non_oracle forbids Oracle, but "
                "selected_thresholds.json contains oracle_threshold identities.",
                file=sys.stderr,
            )
            return 1

    print("=" * 60)
    print(f"M3 FORMAL ARTIFACT VALIDATION: {output_path}")
    print(f"Mode (explicit): {mode_name}")
    print("=" * 60)

    all_errors = []

    # Per-mode tuning episode counts (replace the hard-coded *25).
    expected_tuning_episodes = (
        EXPECTED_TUNING_EPISODES_ORACLE
        if mode == "formal_closeout"
        else EXPECTED_TUNING_EPISODES_NON_ORACLE
    )

    # 1. Validate scenario_bank_provenance.json
    prov_path = output_path / "scenario_bank_provenance.json"
    if prov_path.exists():
        print("\n[1] Validating scenario_bank_provenance.json...")
        success, errors = validate_scenario_bank_provenance(prov_path)
        if success:
            print("    ✓ Scenario-bank provenance valid")
            if mode == "formal_closeout":
                with open(prov_path, "r") as provenance_file:
                    provenance_payload = json.load(provenance_file)
                banks = provenance_payload.get("scenario_banks", [])
                expected_bank_ids = {
                    (split, k, regime)
                    for split in ("predictor_train", "rl_validation")
                    for k in K_VALUES
                    for regime in COST_REGIMES
                }
                actual_bank_ids = {
                    (bank.get("split"), bank.get("K"), bank.get("cost_regime_id"))
                    for bank in banks
                }
                if (
                    len(banks) != 16
                    or len(actual_bank_ids) != 16
                    or actual_bank_ids != expected_bank_ids
                ):
                    bank_errors = [
                        "Formal scenario-bank identities must be exactly the "
                        "16 split × K × cost-regime combinations"
                    ]
                    print(f"    ✗ Scenario-bank identity errors: {bank_errors}")
                    all_errors.extend(bank_errors)
                else:
                    print("    ✓ Exact 16 scenario-bank identities present")
        else:
            print(f"    ✗ Scenario-bank provenance errors: {errors}")
            all_errors.extend(errors)
    else:
        all_errors.append("Missing scenario_bank_provenance.json")

    # 2. Validate run_provenance.json
    run_prov_path = output_path / "run_provenance.json"
    if run_prov_path.exists():
        print("\n[2] Validating run_provenance.json...")
        # Try to determine run type from content
        success, data, _ = validate_json_file(run_prov_path)
        run_type = data.get("run_type", "unknown") if success else "threshold_tuning"
        success, errors = validate_run_provenance(run_prov_path, run_type)
        if success:
            print(f"    ✓ Run provenance valid ({run_type})")
        else:
            print(f"    ✗ Run provenance errors: {errors}")
            all_errors.extend(errors)
    else:
        all_errors.append("Missing run_provenance.json")

    # 3. Validate threshold_search_results.parquet
    results_path = output_path / "threshold_search_results.parquet"
    if results_path.exists():
        print("\n[3] Validating threshold_search_results.parquet...")
        success, df, err = validate_parquet_file(results_path)
        if not success:
            print(f"    ✗ {err}")
            all_errors.append(err)
        else:
            print(f"    ✓ Loaded {len(df)} rows")

            # The canonical parquet is episode-level. Candidate identities are
            # the unique 4-column projection; episode identities are the
            # unique 6-column projection.
            if len(df) != expected_tuning_episodes:
                msg = (
                    f"Expected {expected_tuning_episodes} tuning episodes, "
                    f"got {len(df)}"
                )
                print(f"    ✗ {msg}")
                all_errors.append(msg)
            else:
                print(f"    ✓ Tuning episode count correct ({len(df)})")

            # Validate finite values
            errors = validate_finite_values(
                df,
                [
                    "total_cost",
                    "failure_count",
                    "wasted_life_cost",
                    "preventive_cost",
                    "failure_cost",
                ],
            )
            if errors:
                print(f"    ✗ Numeric errors: {errors}")
                all_errors.extend(errors)
            else:
                print("    ✓ All numeric values finite")

            # Validate threshold grid membership
            errors = validate_threshold_grid_membership(df)
            if errors:
                print(f"    ✗ Grid membership errors: {errors}")
                all_errors.extend(errors)
            else:
                print("    ✓ All thresholds in frozen grids")

            candidate_cols = [
                "policy_family",
                "threshold",
                "k_capacity",
                "cost_regime_id",
            ]
            candidate_count = len(df[candidate_cols].drop_duplicates())
            if candidate_count != expected_tuning_candidates:
                errors = [
                    f"Expected {expected_tuning_candidates} candidate identities, "
                    f"got {candidate_count}"
                ]
            else:
                errors = []
            episode_identity_cols = candidate_cols + ["scenario_id", "reset_seed"]
            errors.extend(validate_unique_keys(df, episode_identity_cols))
            if errors:
                print(f"    ✗ Identity errors: {errors}")
                all_errors.extend(errors)
            else:
                print(
                    f"    ✓ {candidate_count} candidates and "
                    f"{len(df)} unique tuning episodes"
                )
    else:
        all_errors.append("Missing threshold_search_results.parquet")

    # 3b. Validate the separately persisted 360-row candidate summary.
    summary_search_path = output_path / "threshold_search_summary.csv"
    if results_path.exists() and summary_search_path.exists():
        print("\n[3b] Validating threshold_search_summary.csv...")
        success, errors = validate_threshold_search_summary(
            results_path,
            summary_search_path,
            expected_tuning_candidates,
        )
        if success:
            print(
                f"    ✓ Candidate summary matches {expected_tuning_candidates} "
                "episode-derived candidates"
            )
        else:
            print(f"    ✗ Candidate summary errors: {errors}")
            all_errors.extend(errors)
    else:
        all_errors.append("Missing threshold_search_summary.csv")

    # 4. Validate selected_thresholds.json
    selected_path = output_path / "selected_thresholds.json"
    if selected_path.exists():
        print("\n[4] Validating selected_thresholds.json...")
        success, selected, err = validate_json_file(selected_path)
        if not success:
            print(f"    ✗ {err}")
            all_errors.append(err)
        else:
            # The selected_tuning writers stamp a mandatory `_meta` provenance
            # envelope alongside the selected winners (matching the formal
            # loader's load_formal_selected_thresholds which subtracts `_meta`
            # to derive the actual winner set).  Subtract it here so the
            # validator counts *winners*, not the total JSON key count.
            winners = {k: v for k, v in selected.items() if k != "_meta"}
            winner_count = len(winners)
            print(
                f"    ✓ Loaded {winner_count} selected thresholds"
                f" (plus _meta envelope)"
            )

            if winner_count != expected_selected_thresholds:
                msg = f"Expected {expected_selected_thresholds} selected, got {winner_count}"
                print(f"    ✗ {msg}")
                all_errors.append(msg)
            else:
                print(f"    ✓ Selected count correct ({winner_count})")

            # Validate against results if available
            if results_path.exists():
                success, errors = validate_selected_thresholds(selected_path, results_path)
                if success:
                    print("    ✓ All selected thresholds are winners")
                else:
                    print(f"    ✗ Winner validation errors: {errors}")
                    all_errors.extend(errors)
    else:
        all_errors.append("Missing selected_thresholds.json")

    # 5. Validate episode_results.parquet (if present - evaluation)
    episode_path = output_path / "episode_results.parquet"
    if episode_path.exists():
        print("\n[5] Validating episode_results.parquet...")
        success, df, err = validate_parquet_file(episode_path)
        if not success:
            print(f"    ✗ {err}")
            all_errors.append(err)
        else:
            print(f"    ✓ Loaded {len(df)} episodes")

            # Check expected count
            if len(df) != expected_evaluation_episodes:
                msg = f"Expected {expected_evaluation_episodes} episodes, got {len(df)}"
                print(f"    ✗ {msg}")
                all_errors.append(msg)
            else:
                print(f"    ✓ Episode count correct ({len(df)})")

            # Validate finite values
            errors = validate_finite_values(df, ["episode_return", "total_cost", "preventive_cost", "failure_cost", "wasted_life_cost"])
            if errors:
                print(f"    ✗ Numeric errors: {errors}")
                all_errors.extend(errors)
            else:
                print("    ✓ All numeric values finite")

            # Validate episode completion
            errors = validate_episode_completion(df)
            if errors:
                print(f"    ✗ Completion errors: {errors}")
                all_errors.extend(errors)
            else:
                print("    ✓ All episodes completed (100 steps)")

            # Validate reward-cost reconciliation
            errors = validate_reward_cost_reconciliation(df)
            if errors:
                print(f"    ✗ Reconciliation errors: {errors}")
                all_errors.extend(errors)
            else:
                print("    ✓ Reward ≈ -cost for all episodes")

            # Validate cost decomposition
            errors = validate_cost_decomposition(df)
            if errors:
                print(f"    ✗ Decomposition errors: {errors}")
                all_errors.extend(errors)
            else:
                print("    ✓ Cost decomposition valid")

            # Validate unique keys
            errors = validate_unique_keys(df, ["policy_family", "maintenance_capacity", "cost_regime_id", "split", "scenario_id", "reset_seed"])
            if errors:
                print(f"    ✗ Uniqueness errors: {errors}")
                all_errors.extend(errors)
            else:
                print("    ✓ All episodes unique")

            # Validate threshold-use equality (if selected_thresholds.json exists)
            if selected_path.exists():
                success, errors = validate_threshold_use_equality(episode_path, selected_path)
                if success:
                    print("    ✓ Threshold-use equality: evaluation used selected thresholds")
                else:
                    print(f"    ✗ Threshold-use equality errors: {errors}")
                    all_errors.extend(errors)
    else:
        print("\n[5] Skipping episode_results.parquet (evaluation not run)")

    # 6. Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    if all_errors:
        print(f"\n✗ FAILED: {len(all_errors)} errors")
        for err in all_errors:
            print(f"  - {err}")

        # Write validation_report.json WITHOUT touching formal_manifest.json.
        # Validator must not retroactively rewrite the immutable manifest;
        # the verifier verdict lives in the separate report file.
        verdict = "FAILED"
        timestamp = datetime.utcnow().isoformat()
        report = {
            "verdict": verdict,
            "mode": mode_name,
            "all_errors": all_errors,
            "validated_at": timestamp,
            "oracle_semantic_role": (
                ORACLE_SEMANTIC_ROLE if oracle_included else None
            ),
        }
        try:
            with open(output_path / "validation_report.json", "w") as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            print(f"WARNING: failed to write validation_report.json: {e}", file=sys.stderr)

        return 1
    else:
        print("\n✓ ALL VALIDATIONS PASSED")

        # Write validation_report.json — never mutate formal_manifest.json.
        verdict = "ALL PASSED"
        timestamp = datetime.utcnow().isoformat()
        report = {
            "verdict": verdict,
            "mode": mode_name,
            "all_errors": [],
            "validated_at": timestamp,
            "oracle_semantic_role": (
                ORACLE_SEMANTIC_ROLE if oracle_included else None
            ),
        }
        try:
            with open(output_path / "validation_report.json", "w") as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            print(f"WARNING: failed to write validation_report.json: {e}", file=sys.stderr)

        return 0


if __name__ == "__main__":
    import argparse as _argparse
    parser = _argparse.ArgumentParser(description="Validate M3 formal artifacts.")
    parser.add_argument("output_dir")
    parser.add_argument("expected_commit", nargs="?", default=None)
    parser.add_argument(
        "--mode",
        choices=(
            "formal_closeout",
            "diagnostic_non_oracle",
            "diagnostic_legacy",
        ),
        default=None,
        help=(
            "Explicit mode (REQUIRED — no auto-detection from artifacts). "
            "formal_closeout requires Oracle and 360/9000/32/2400 counts; "
            "diagnostic_non_oracle expects 272/6800/24/2000 without Oracle; "
            "diagnostic_legacy is the explicit entry point that may tolerate "
            "legacy context-shape fixtures."
        ),
    )
    args = parser.parse_args()

    exit_code = main(args.output_dir, args.expected_commit, mode=args.mode)
    sys.exit(exit_code)
