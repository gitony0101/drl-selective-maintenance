"""Predictor Collapse Detector — V2 Formal Report Schema

Detects degenerate predictor behavior and produces a machine-readable
V2 collapse report with full provenance identity.

The report is used as a hard gate in the Milestone 1 validator:
cache generation must not proceed if any required split is collapsed.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.predictors.io_utils import atomic_write_json

# ---------------------------------------------------------------------------
# V2 Collapse Report Schema
# ---------------------------------------------------------------------------

_COLLAPSE_REPORT_SCHEMA_VERSION = "fd001_collapse_report_v2"

_REQUIRED_SPLITS = ["predictor_train", "rl_validation", "rl_test"]

_DEFAULT_STD_RATIO_THRESHOLD = 0.1
_DEFAULT_UNIQUE_RATIO_THRESHOLD = 0.01
_DEFAULT_MIN_CORRELATION = 0.1
_DEFAULT_MIN_PREDICTION_RANGE = 1.0


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_full_git_commit() -> str:
    """Get current full git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _split_result(
    predictions: np.ndarray,
    true_rul: np.ndarray,
    std_ratio_threshold: float,
    unique_ratio_threshold: float,
    min_correlation: float,
    min_prediction_range: float,
) -> Dict[str, Any]:
    """Run collapse detection for one split and return a structured result dict.

    Returns:
        Dict with keys: passed, is_collapsed, failure_reasons, prediction_count,
        prediction_mean, prediction_std, prediction_min, prediction_max,
        prediction_range, unique_count, unique_ratio, std_ratio, pearson_correlation.
    """
    assert len(predictions) == len(true_rul), "predictions and true_rul must have same length"
    n = len(predictions)
    failure_reasons: List[str] = []

    # Non-finite check
    nan_count = int(np.isnan(predictions).sum())
    inf_count = int(np.isinf(predictions).sum())
    if nan_count > 0:
        failure_reasons.append(f"Found {nan_count} NaN predictions")
    if inf_count > 0:
        failure_reasons.append(f"Found {inf_count} Inf predictions")

    # Filter to finite for statistics
    finite_mask = np.isfinite(predictions)
    pred_finite = predictions[finite_mask]
    true_finite = true_rul[finite_mask]
    n_finite = int(finite_mask.sum())

    if n_finite == 0:
        return {
            "passed": False,
            "is_collapsed": True,
            "failure_reasons": failure_reasons + ["No finite predictions"],
            "prediction_count": n,
            "prediction_mean": None,
            "prediction_std": None,
            "prediction_min": None,
            "prediction_max": None,
            "prediction_range": None,
            "unique_count": 0,
            "unique_ratio": 0.0,
            "std_ratio": None,
            "pearson_correlation": None,
        }

    pred_mean = float(np.mean(pred_finite))
    pred_std = float(np.std(pred_finite))
    pred_min = float(np.min(pred_finite))
    pred_max = float(np.max(pred_finite))
    pred_range = pred_max - pred_min
    true_std = float(np.std(true_finite))
    std_ratio = pred_std / true_std if true_std > 0 else 0.0
    unique_count = int(len(np.unique(pred_finite)))
    unique_ratio = unique_count / n_finite if n_finite > 0 else 0.0

    # Pearson correlation
    try:
        corr, _ = stats.pearsonr(pred_finite, true_finite)
        corr = float(corr)
    except Exception:
        corr = 0.0

    # Threshold checks
    if pred_std == 0 or unique_count == 1:
        failure_reasons.append("Exactly constant predictions (std=0)")
    if std_ratio < std_ratio_threshold:
        failure_reasons.append(
            f"Near-constant predictions: std_ratio={std_ratio:.4f} < {std_ratio_threshold}"
        )
    if pred_range < min_prediction_range:
        failure_reasons.append(
            f"Prediction range too small: {pred_range:.4f} < {min_prediction_range}"
        )
    if unique_ratio < unique_ratio_threshold:
        failure_reasons.append(
            f"Too few unique values: {unique_count}/{n_finite} ({unique_ratio:.2%})"
        )
    if abs(corr) < min_correlation:
        failure_reasons.append(
            f"No correlation with true RUL: r={corr:.4f} < {min_correlation}"
        )

    is_collapsed = len(failure_reasons) > 0

    return {
        "passed": not is_collapsed,
        "is_collapsed": is_collapsed,
        "failure_reasons": failure_reasons,
        "prediction_count": n,
        "prediction_mean": pred_mean,
        "prediction_std": pred_std,
        "prediction_min": pred_min,
        "prediction_max": pred_max,
        "prediction_range": pred_range,
        "unique_count": unique_count,
        "unique_ratio": round(unique_ratio, 6),
        "std_ratio": round(std_ratio, 6),
        "pearson_correlation": round(corr, 6) if corr is not None else None,
    }


def build_collapse_report(
    cache_path: Path,
    predictor_id: str,
    checkpoint_id: str,
    cache_sha256: str,
    training_git_commit: str,
    std_ratio_threshold: float = _DEFAULT_STD_RATIO_THRESHOLD,
    unique_ratio_threshold: float = _DEFAULT_UNIQUE_RATIO_THRESHOLD,
    min_correlation: float = _DEFAULT_MIN_CORRELATION,
    min_prediction_range: float = _DEFAULT_MIN_PREDICTION_RANGE,
) -> Dict[str, Any]:
    """Build a formal V2 collapse report from a prediction cache parquet.

    Args:
        cache_path: Path to V2 prediction cache parquet.
        predictor_id: Predictor ID from canonical metadata.
        checkpoint_id: Full 64-char SHA256 of best_checkpoint.pt.
        cache_sha256: Full SHA256 of the cache parquet file.
        training_git_commit: Git commit hash from training checkpoint.
        std_ratio_threshold: Threshold for std_ratio below which to flag.
        unique_ratio_threshold: Minimum fraction of unique values.
        min_correlation: Minimum acceptable correlation with true RUL.
        min_prediction_range: Minimum acceptable prediction range.

    Returns:
        Complete V2 collapse report dict.
    """
    df = pd.read_parquet(cache_path)
    predictions = df["predicted_rul"].values.astype(np.float64)
    true_rul = df["true_rul_capped"].values.astype(np.float64)

    # Overall result across all data
    overall = _split_result(
        predictions, true_rul,
        std_ratio_threshold, unique_ratio_threshold,
        min_correlation, min_prediction_range,
    )

    # Per-split results
    per_split: Dict[str, Dict[str, Any]] = {}
    for split in _REQUIRED_SPLITS:
        split_df = df[df["split"] == split]
        if len(split_df) > 0:
            split_pred = split_df["predicted_rul"].values.astype(np.float64)
            split_true = split_df["true_rul_capped"].values.astype(np.float64)
            per_split[split] = _split_result(
                split_pred, split_true,
                std_ratio_threshold, unique_ratio_threshold,
                min_correlation, min_prediction_range,
            )
        else:
            per_split[split] = {
                "passed": False,
                "is_collapsed": True,
                "failure_reasons": [f"Split '{split}' has no data"],
                "prediction_count": 0,
                "prediction_mean": None,
                "prediction_std": None,
                "prediction_min": None,
                "prediction_max": None,
                "prediction_range": None,
                "unique_count": 0,
                "unique_ratio": 0.0,
                "std_ratio": None,
                "pearson_correlation": None,
            }

    # Overall top-level passed requires all splits pass + no NaN/Inf
    nan_count = int(np.isnan(predictions).sum())
    inf_count = int(np.isinf(predictions).sum())
    has_non_finite = nan_count > 0 or inf_count > 0
    all_splits_exist = all(s in per_split for s in _REQUIRED_SPLITS)
    all_splits_pass = all(per_split[s]["passed"] for s in _REQUIRED_SPLITS)
    overall_passed = (
        overall["passed"]
        and all_splits_pass
        and all_splits_exist
        and not has_non_finite
    )
    overall["passed"] = overall_passed

    tooling_git_commit = _get_full_git_commit()

    report: Dict[str, Any] = {
        "schema_version": _COLLAPSE_REPORT_SCHEMA_VERSION,
        "passed": overall_passed,
        "overall": overall,
        "per_split": per_split,
        "thresholds": {
            "std_ratio_threshold": std_ratio_threshold,
            "unique_ratio_threshold": unique_ratio_threshold,
            "min_correlation": min_correlation,
            "min_prediction_range": min_prediction_range,
        },
        "predictor_id": predictor_id,
        "checkpoint_id": checkpoint_id,
        "cache_sha256": cache_sha256,
        "training_git_commit": training_git_commit,
        "tooling_git_commit": tooling_git_commit,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return report


def main() -> int:
    """Run collapse detection and write V2 report atomically."""
    parser = argparse.ArgumentParser(
        description="Generate V2 collapse report for prediction cache"
    )
    parser.add_argument(
        "--cache-path", type=Path, required=True,
        help="Path to prediction cache parquet",
    )
    parser.add_argument(
        "--output-path", type=Path, required=True,
        help="Path to write the V2 collapse report JSON",
    )
    parser.add_argument(
        "--predictor-id", type=str, required=True,
        help="Predictor ID from canonical metadata",
    )
    parser.add_argument(
        "--checkpoint-id", type=str, required=True,
        help="Full 64-char SHA256 of best_checkpoint.pt",
    )
    parser.add_argument(
        "--training-git-commit", type=str, required=True,
        help="Git commit hash from training checkpoint",
    )
    parser.add_argument(
        "--std-ratio-threshold", type=float,
        default=_DEFAULT_STD_RATIO_THRESHOLD,
    )
    parser.add_argument(
        "--unique-ratio-threshold", type=float,
        default=_DEFAULT_UNIQUE_RATIO_THRESHOLD,
    )
    parser.add_argument(
        "--min-correlation", type=float,
        default=_DEFAULT_MIN_CORRELATION,
    )
    parser.add_argument(
        "--min-prediction-range", type=float,
        default=_DEFAULT_MIN_PREDICTION_RANGE,
    )
    args = parser.parse_args()

    if not args.cache_path.exists():
        print(f"Error: cache not found: {args.cache_path}", file=sys.stderr)
        return 1

    cache_sha256 = compute_file_hash(args.cache_path)

    report = build_collapse_report(
        cache_path=args.cache_path,
        predictor_id=args.predictor_id,
        checkpoint_id=args.checkpoint_id,
        cache_sha256=cache_sha256,
        training_git_commit=args.training_git_commit,
        std_ratio_threshold=args.std_ratio_threshold,
        unique_ratio_threshold=args.unique_ratio_threshold,
        min_correlation=args.min_correlation,
        min_prediction_range=args.min_prediction_range,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_path, report)

    if report["passed"]:
        print("PREDICTOR NOT COLLAPSED — All splits pass")
        return 0
    else:
        print("PREDICTOR COLLAPSED — Cache generation blocked", file=sys.stderr)
        for split_name, split_res in report["per_split"].items():
            if split_res["failure_reasons"]:
                print(
                    f"  {split_name}: {'; '.join(split_res['failure_reasons'])}",
                    file=sys.stderr,
                )
        return 1


# ---------------------------------------------------------------------------
# Backward-compatible API (preserved for tests)
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Optional as OptionalType


@dataclass
class CollapseResult:
    """Result of collapse detection (backward-compatible API)."""
    is_collapsed: bool
    failure_reason: OptionalType[str] = None
    prediction_mean: float = 0.0
    prediction_std: float = 0.0
    prediction_range: float = 0.0
    true_rul_std: float = 0.0
    std_ratio: float = 0.0
    correlation: float = 0.0
    unique_count: int = 0
    constant_threshold: float = 0.01


def detect_collapse(
    predictions: np.ndarray,
    true_rul: np.ndarray,
    std_ratio_threshold: float = 0.1,
    unique_ratio_threshold: float = 0.01,
    min_correlation: float = 0.1,
) -> CollapseResult:
    """Detect if predictor has collapsed (backward-compatible API)."""
    res = _split_result(
        predictions, true_rul,
        std_ratio_threshold, unique_ratio_threshold,
        min_correlation, _DEFAULT_MIN_PREDICTION_RANGE,
    )
    return CollapseResult(
        is_collapsed=res["is_collapsed"],
        failure_reason="; ".join(res["failure_reasons"]) if res["failure_reasons"] else None,
        prediction_mean=res["prediction_mean"] or 0.0,
        prediction_std=res["prediction_std"] or 0.0,
        prediction_range=res["prediction_range"] or 0.0,
        true_rul_std=float(np.std(true_rul[np.isfinite(true_rul)])),
        std_ratio=res["std_ratio"] or 0.0,
        correlation=res["pearson_correlation"] or 0.0,
        unique_count=res["unique_count"],
        constant_threshold=std_ratio_threshold,
    )


def validate_cache_for_collapse(
    cache_path: Path,
    std_ratio_threshold: float = 0.1,
) -> Tuple[bool, Dict[str, Any]]:
    """Validate a prediction cache for collapse (backward-compatible API).

    Args:
        cache_path: Path to prediction cache parquet
        std_ratio_threshold: Threshold for std_ratio

    Returns:
        (passed, results_dict)
    """
    df = pd.read_parquet(cache_path)

    overall = detect_collapse(
        df["predicted_rul"].values,
        df["true_rul_capped"].values,
        std_ratio_threshold=std_ratio_threshold,
    )

    per_split = {}
    for split in df["split"].unique():
        split_df = df[df["split"] == split]
        per_split[split] = detect_collapse(
            split_df["predicted_rul"].values,
            split_df["true_rul_capped"].values,
            std_ratio_threshold=std_ratio_threshold,
        )

    report = {
        "overall": {
            "passed": not overall.is_collapsed,
            "failure_reason": overall.failure_reason,
            "prediction_mean": overall.prediction_mean,
            "prediction_std": overall.prediction_std,
            "prediction_range": overall.prediction_range,
            "true_rul_std": overall.true_rul_std,
            "std_ratio": overall.std_ratio,
            "correlation": overall.correlation,
            "unique_count": overall.unique_count,
        },
        "per_split": {
            split: {
                "passed": not result.is_collapsed,
                "failure_reason": result.failure_reason,
                "prediction_mean": result.prediction_mean,
                "prediction_std": result.prediction_std,
                "std_ratio": result.std_ratio,
                "correlation": result.correlation,
            }
            for split, result in per_split.items()
        },
    }

    return not overall.is_collapsed, report


if __name__ == "__main__":
    sys.exit(main())