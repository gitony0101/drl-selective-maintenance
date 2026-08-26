"""M8 Formal Metrics Module.

Computes all 17 formal validation metrics for M8 protocol predictor evaluation.
No external dependencies beyond numpy and pandas (validation/recomputation only).
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# Frozen constants per Protocol v1.3
_NASA_A1 = 13.0  # Denominator for underestimation (error < 0)
_NASA_A2 = 10.0  # Denominator for overestimation (error >= 0)
_VALID_STATUSES = frozenset({"COMPLETED", "EARLY_STOPPED", "FAILED"})
_REQUIRED_COLUMNS = (
    "split",
    "unit_id",
    "cycle",
    "true_rul_capped",
    "predicted_rul",
    "error",
)
_VALIDATION_SPLIT_VALUE = "predictor_validation"
_EXPECTED_ROW_COUNT = 3146
_EXPECTED_DTYPES = {
    "unit_id": np.int32,
    "cycle": np.int32,
    "true_rul_capped": np.float32,
    "predicted_rul": np.float32,
    "error": np.float32,
}


def compute_formal_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    best_epoch: int,
    final_epoch: int,
    training_status: str,
) -> Dict[str, Any]:
    """Compute all 17 formal validation metrics for a single formal run.

    Args:
        y_true: Ground truth RUL (capped at 125), 1D array of length N
        y_pred: Predicted RUL, 1D array of length N
        best_epoch: Integer epoch index of best checkpoint (1-indexed per protocol)
        final_epoch: Integer last completed epoch (1-indexed per protocol)
        training_status: Terminal status enum ("COMPLETED" | "EARLY_STOPPED" | "FAILED")

    Returns:
        Dictionary with exactly 17 keys in canonical order:
        1. row_count (int, exact)
        2. mae (float, RUL cycles, 4dp)
        3. rmse (float, RUL cycles, 4dp)
        4. nasa_score (float, dimensionless SUM, 4dp)
        5. mean_prediction_error (float, RUL cycles, 4dp)
        6. bias (float, RUL cycles, 4dp, equals mean_prediction_error)
        7. overestimation_count (int, exact)
        8. overestimation_rate (float, fraction [0,1], 4dp)
        9. mean_positive_error (float, RUL cycles, 4dp)
        10. positive_error_p90 (float, RUL cycles, 4dp)
        11. positive_error_p95 (float, RUL cycles, 4dp)
        12. underestimation_count (int, exact)
        13. underestimation_rate (float, fraction [0,1], 4dp)
        14. non_finite_count (int, exact)
        15. best_epoch (int, exact)
        16. final_epoch (int, exact)
        17. training_status (str, enum)

    Raises:
        ValueError: If y_true/y_pred shapes mismatch, or training_status not in enum
        TypeError: If inputs not array-like or not numeric
    """
    # Convert to numpy arrays
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Input validation
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1D arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    if not np.issubdtype(y_true.dtype, np.number) or not np.issubdtype(y_pred.dtype, np.number):
        raise TypeError("Inputs must be numeric")

    if training_status not in _VALID_STATUSES:
        raise ValueError(
            f"training_status must be one of {sorted(_VALID_STATUSES)}, got {training_status!r}"
        )

    row_count = int(y_true.shape[0])

    # Empty input edge case
    if row_count == 0:
        return {
            "row_count": 0,
            "mae": 0.0,
            "rmse": 0.0,
            "nasa_score": 0.0,
            "mean_prediction_error": 0.0,
            "bias": 0.0,
            "overestimation_count": 0,
            "overestimation_rate": 0.0,
            "mean_positive_error": 0.0,
            "positive_error_p90": 0.0,
            "positive_error_p95": 0.0,
            "underestimation_count": 0,
            "underestimation_rate": 0.0,
            "non_finite_count": 0,
            "best_epoch": int(best_epoch),
            "final_epoch": int(final_epoch),
            "training_status": "FAILED",  # Overrides input status per protocol
        }

    # Compute error = predicted_rul - true_rul_capped
    error = y_pred - y_true

    # Non-finite detection
    non_finite_mask = ~np.isfinite(error)
    non_finite_count = int(np.sum(non_finite_mask))

    # If any non-finite values, force status to FAILED
    if non_finite_count > 0:
        training_status = "FAILED"

    # Filter to finite values for metric computation
    finite_mask = np.isfinite(error)
    if np.any(finite_mask):
        error_finite = error[finite_mask]
    else:
        # All non-finite
        error_finite = np.array([], dtype=np.float64)

    # Basic statistics
    mae = float(np.mean(np.abs(error_finite))) if error_finite.size > 0 else 0.0
    rmse = float(np.sqrt(np.mean(error_finite**2))) if error_finite.size > 0 else 0.0
    mean_prediction_error = float(np.mean(error_finite)) if error_finite.size > 0 else 0.0

    # NASA Score (SUM, not mean)
    nasa_score = _compute_nasa_score_sum(error_finite)

    # Overestimation (error > 0) - dangerous
    overestimation_mask = error_finite > 0
    overestimation_count = int(np.sum(overestimation_mask))
    overestimation_rate = overestimation_count / row_count if row_count > 0 else 0.0

    pos_errors = error_finite[overestimation_mask]
    if pos_errors.size > 0:
        mean_positive_error = float(np.mean(pos_errors))
        positive_error_p90 = float(np.percentile(pos_errors, 90))
        positive_error_p95 = float(np.percentile(pos_errors, 95))
    else:
        mean_positive_error = 0.0
        positive_error_p90 = 0.0
        positive_error_p95 = 0.0

    # Underestimation (error < 0) - conservative
    underestimation_mask = error_finite < 0
    underestimation_count = int(np.sum(underestimation_mask))
    underestimation_rate = underestimation_count / row_count if row_count > 0 else 0.0

    # Round to 4 decimal places (Python 3 round-half-even)
    def _r4(x: float) -> float:
        return round(x, 4)

    return {
        "row_count": row_count,
        "mae": _r4(mae),
        "rmse": _r4(rmse),
        "nasa_score": _r4(nasa_score),
        "mean_prediction_error": _r4(mean_prediction_error),
        "bias": _r4(mean_prediction_error),  # Always equals mean_prediction_error
        "overestimation_count": overestimation_count,
        "overestimation_rate": _r4(overestimation_rate),
        "mean_positive_error": _r4(mean_positive_error),
        "positive_error_p90": _r4(positive_error_p90),
        "positive_error_p95": _r4(positive_error_p95),
        "underestimation_count": underestimation_count,
        "underestimation_rate": _r4(underestimation_rate),
        "non_finite_count": non_finite_count,
        "best_epoch": int(best_epoch),
        "final_epoch": int(final_epoch),
        "training_status": training_status,
    }


def _compute_nasa_score_sum(error: np.ndarray) -> float:
    """Compute NASA C-MAPSS score (SUM of per-row scores).

    Formula (frozen per Protocol v1.3):
        if error < 0:  exp(-error / 13.0) - 1.0
        else:          exp(error / 10.0) - 1.0

    Returns SUM, not mean.
    """
    if error.size == 0:
        return 0.0

    # Vectorized computation
    neg_mask = error < 0
    pos_mask = ~neg_mask

    score = 0.0
    if np.any(neg_mask):
        score += np.sum(np.exp(-error[neg_mask] / _NASA_A1) - 1.0)
    if np.any(pos_mask):
        score += np.sum(np.exp(error[pos_mask] / _NASA_A2) - 1.0)

    return float(score)


def compute_nasa_score(error: np.ndarray) -> float:
    """Standalone NASA C-MAPSS scoring function.

    Args:
        error: 1D array of (predicted_rul - true_rul_capped) values

    Returns:
        SUM of per-row scores (NOT mean)

    Formula (frozen):
        if error < 0:  exp(-error / 13.0) - 1.0
        else:          exp(error / 10.0) - 1.0
    """
    error = np.asarray(error, dtype=np.float64)
    if error.ndim != 1:
        raise ValueError("error must be 1D array")
    return _compute_nasa_score_sum(error)


def validate_prediction_frame(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate a predictor_validation_predictions.parquet DataFrame has required schema.

    Args:
        df: DataFrame loaded from parquet

    Returns:
        (is_valid, error_messages)

    Required columns (exact names, types):
        - split: string, all values == "predictor_validation"
        - unit_id: int32
        - cycle: int32
        - true_rul_capped: float32
        - predicted_rul: float32
        - error: float32 (must equal predicted_rul - true_rul_capped)
    """
    errors = []

    # 1. Required columns present
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")

    # Early return if missing columns to avoid downstream errors
    if errors:
        return False, errors

    # 2. Row count
    if len(df) != _EXPECTED_ROW_COUNT:
        errors.append(f"Row count {len(df)} != {_EXPECTED_ROW_COUNT}")

    # 3. Split column all "predictor_validation"
    if not (df["split"] == _VALIDATION_SPLIT_VALUE).all():
        errors.append(f"split column contains non-{_VALIDATION_SPLIT_VALUE} values")

    # 4. Dtypes
    for col, expected_dtype in _EXPECTED_DTYPES.items():
        if col in df.columns:
            actual_dtype = df[col].dtype
            if actual_dtype != expected_dtype:
                errors.append(f"{col} dtype {actual_dtype} != {expected_dtype}")

    # 5. True RUL capped range [0, 125]
    if "true_rul_capped" in df.columns:
        if (df["true_rul_capped"] < 0).any() or (df["true_rul_capped"] > 125).any():
            errors.append("true_rul_capped out of range [0, 125]")

    # 6. Error consistency: error == predicted_rul - true_rul_capped
    if all(c in df.columns for c in ("predicted_rul", "true_rul_capped", "error")):
        recomputed_error = df["predicted_rul"] - df["true_rul_capped"]
        if (recomputed_error - df["error"]).abs().max() > 1e-6:
            errors.append("error column != predicted_rul - true_rul_capped")

    # 7. No duplicate (unit_id, cycle) pairs
    if "unit_id" in df.columns and "cycle" in df.columns:
        if df.duplicated(subset=["unit_id", "cycle"]).any():
            errors.append("Duplicate (unit_id, cycle) pairs found")

    # 8. Row ordering: sorted by (unit_id, cycle) ascending
    if "unit_id" in df.columns and "cycle" in df.columns:
        if not df["unit_id"].is_monotonic_increasing:
            errors.append("Rows not sorted by unit_id")
        else:
            for uid, group in df.groupby("unit_id"):
                if not group["cycle"].is_monotonic_increasing:
                    errors.append(f"Rows for unit {uid} not sorted by cycle")
                    break

    return len(errors) == 0, errors


def attach_training_metadata(
    metrics: Dict[str, Any],
    training_history: List[Dict[str, Any]],
    checkpoint_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Attach epoch metadata to metrics dict.

    Args:
        metrics: Output from compute_formal_metrics (without best_epoch/final_epoch)
        training_history: List of per-epoch dicts from training_history.json
        checkpoint_info: Dict with 'best_epoch', 'final_epoch', 'best_val_rmse'

    Returns:
        metrics with best_epoch, final_epoch populated
    """
    result = metrics.copy()
    result["best_epoch"] = int(checkpoint_info.get("best_epoch", metrics.get("best_epoch", 0)))
    result["final_epoch"] = int(checkpoint_info.get("final_epoch", metrics.get("final_epoch", 0)))
    return result


def write_metrics_json(metrics: Dict[str, Any], path: Path) -> None:
    """Atomically write metrics dict to JSON with fsync + os.replace.

    Args:
        metrics: 17-field dict from compute_formal_metrics
        path: Target path (e.g., predictor_validation_metrics.json)

    Requirements:
        - Sort keys for deterministic output
        - 4 decimal places for floats (round-half-even)
        - Integers as JSON integers (no trailing .0)
        - Atomic write: tmp.<pid> -> fsync -> os.replace
    """
    path = Path(path)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.{os.urandom(4).hex()}")

    # Ensure deterministic JSON: sort keys, 4dp for floats
    def _json_serializer(obj: Any) -> Any:
        if isinstance(obj, float):
            return round(obj, 4)
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True, default=_json_serializer)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, path)


def independently_recompute_metrics_from_parquet(
    parquet_path: Path,
    best_epoch: int,
    final_epoch: int,
    training_status: str,
) -> Dict[str, Any]:
    """Independent recomputation for audit verification.

    Loads parquet, extracts y_true (re)computes error = predicted_rul - true_rul_capped,
    calls compute_formal_metrics, returns full 17-field dict.

    Used by: independent audit, CI verification, manual validation.
    """
    parquet_path = Path(parquet_path)
    df = pd.read_parquet(parquet_path)

    # Validate frame first
    is_valid, errors = validate_prediction_frame(df)
    if not is_valid:
        raise ValueError(f"Invalid prediction frame: {'; '.join(errors)}")

    # Extract arrays (recompute error, don't trust stored error column)
    y_true = df["true_rul_capped"].to_numpy(dtype=np.float32)
    y_pred = df["predicted_rul"].to_numpy(dtype=np.float32)

    # Call main computation function
    return compute_formal_metrics(
        y_true=y_true,
        y_pred=y_pred,
        best_epoch=best_epoch,
        final_epoch=final_epoch,
        training_status=training_status,
    )