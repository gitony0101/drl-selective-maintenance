"""M8 Formal Metrics Tests.

Exact test cases per M8_FORMAL_METRICS_TEST_PLAN.md.
All tests must pass for Gate B2 verification.
"""

import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.predictors.formal_metrics import (
    attach_training_metadata,
    compute_formal_metrics,
    compute_nasa_score,
    independently_recompute_metrics_from_parquet,
    validate_prediction_frame,
    write_metrics_json,
)


class TestComputeFormalMetrics:
    """Tests for compute_formal_metrics function."""

    def test_against_reference_implementation(self):
        """Verify all 17 metrics against hand-computed reference values.

        Reference: error = [5, 5, -5, 2, -2]
        y_true = [100, 50, 25, 10, 5]
        y_pred = [105, 55, 20, 12, 3]
        """
        y_true = np.array([100.0, 50.0, 25.0, 10.0, 5.0], dtype=np.float32)
        y_pred = np.array([105.0, 55.0, 20.0, 12.0, 3.0], dtype=np.float32)

        metrics = compute_formal_metrics(
            y_true=y_true,
            y_pred=y_pred,
            best_epoch=10,
            final_epoch=15,
            training_status="COMPLETED",
        )

        # Exact integer checks
        assert metrics["row_count"] == 5
        assert metrics["overestimation_count"] == 3
        assert metrics["underestimation_count"] == 2
        assert metrics["non_finite_count"] == 0
        assert metrics["best_epoch"] == 10
        assert metrics["final_epoch"] == 15

        # Float checks (4 decimal places, round-half-even)
        assert round(metrics["mae"], 4) == 3.8000
        assert round(metrics["rmse"], 4) == 4.0743
        assert round(metrics["nasa_score"], 4) == 2.1542
        assert round(metrics["mean_prediction_error"], 4) == 1.0000
        assert round(metrics["bias"], 4) == 1.0000
        assert round(metrics["overestimation_rate"], 4) == 0.6000
        assert round(metrics["mean_positive_error"], 4) == 4.0000
        assert round(metrics["positive_error_p90"], 4) == 5.0000
        assert round(metrics["positive_error_p95"], 4) == 5.0000
        assert round(metrics["underestimation_rate"], 4) == 0.4000

        assert metrics["training_status"] == "COMPLETED"

        # Critical: bias == mean_prediction_error always
        assert metrics["bias"] == metrics["mean_prediction_error"]

    def test_zero_error_vector(self):
        """All predictions perfect: error = [0, 0, 0, 0, 0]"""
        y_true = np.array([100.0, 50.0, 25.0, 10.0, 5.0], dtype=np.float32)
        y_pred = np.array([100.0, 50.0, 25.0, 10.0, 5.0], dtype=np.float32)

        metrics = compute_formal_metrics(y_true, y_pred, 1, 1, "COMPLETED")

        assert metrics["row_count"] == 5
        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["nasa_score"] == 0.0  # exp(0)-1 = 0 per row, sum = 0
        assert metrics["mean_prediction_error"] == 0.0
        assert metrics["bias"] == 0.0
        assert metrics["overestimation_count"] == 0
        assert metrics["overestimation_rate"] == 0.0
        assert metrics["mean_positive_error"] == 0.0
        assert metrics["positive_error_p90"] == 0.0
        assert metrics["positive_error_p95"] == 0.0
        assert metrics["underestimation_count"] == 0
        assert metrics["underestimation_rate"] == 0.0
        assert metrics["non_finite_count"] == 0

    def test_positive_only_errors(self):
        """All errors > 0: error = [10, 20, 5, 15, 8]"""
        y_true = np.array([100.0, 50.0, 25.0, 10.0, 5.0], dtype=np.float32)
        y_pred = np.array([110.0, 70.0, 30.0, 25.0, 13.0], dtype=np.float32)

        metrics = compute_formal_metrics(y_true, y_pred, 1, 1, "COMPLETED")

        assert metrics["overestimation_count"] == 5
        assert metrics["overestimation_rate"] == 1.0
        assert metrics["underestimation_count"] == 0
        assert metrics["underestimation_rate"] == 0.0
        assert metrics["mean_positive_error"] == 11.6  # (10+20+5+15+8)/5
        # np.percentile([5,8,10,15,20], 90, interpolation='linear') = 18.0
        assert round(metrics["positive_error_p90"], 4) == 18.0
        # np.percentile([5,8,10,15,20], 95, interpolation='linear') = 19.0
        assert round(metrics["positive_error_p95"], 4) == 19.0
        assert metrics["non_finite_count"] == 0

    def test_negative_only_errors(self):
        """All errors < 0: error = [-10, -20, -5, -15, -8]"""
        y_true = np.array([100.0, 50.0, 25.0, 10.0, 5.0], dtype=np.float32)
        y_pred = np.array([90.0, 30.0, 20.0, -5.0, -3.0], dtype=np.float32)

        metrics = compute_formal_metrics(y_true, y_pred, 1, 1, "COMPLETED")

        assert metrics["overestimation_count"] == 0
        assert metrics["overestimation_rate"] == 0.0
        assert metrics["mean_positive_error"] == 0.0
        assert metrics["positive_error_p90"] == 0.0
        assert metrics["positive_error_p95"] == 0.0
        assert metrics["underestimation_count"] == 5
        assert metrics["underestimation_rate"] == 1.0
        assert metrics["non_finite_count"] == 0

    def test_no_positive_error_percentiles_zero(self):
        """When no positive errors, p90/p95/mean_positive_error must be 0.0"""
        y_true = np.array([100.0, 50.0], dtype=np.float32)
        y_pred = np.array([90.0, 40.0], dtype=np.float32)  # both underestimation

        metrics = compute_formal_metrics(y_true, y_pred, 1, 1, "COMPLETED")

        assert metrics["overestimation_count"] == 0
        assert metrics["mean_positive_error"] == 0.0
        assert metrics["positive_error_p90"] == 0.0
        assert metrics["positive_error_p95"] == 0.0

    def test_no_negative_error_underestimation_zero(self):
        """When no negative errors, underestimation_count/rate must be 0"""
        y_true = np.array([100.0, 50.0], dtype=np.float32)
        y_pred = np.array([110.0, 60.0], dtype=np.float32)  # both overestimation

        metrics = compute_formal_metrics(y_true, y_pred, 1, 1, "COMPLETED")

        assert metrics["underestimation_count"] == 0
        assert metrics["underestimation_rate"] == 0.0

    def test_nan_in_predictions(self):
        """NaN in predictions -> non_finite_count > 0, status forced to FAILED"""
        y_true = np.array([100.0, 50.0, 25.0], dtype=np.float32)
        y_pred = np.array([105.0, np.nan, 20.0], dtype=np.float32)

        metrics = compute_formal_metrics(y_true, y_pred, 10, 15, "COMPLETED")

        assert metrics["non_finite_count"] == 1
        assert metrics["training_status"] == "FAILED"  # Overrides input status

    def test_positive_infinity_in_predictions(self):
        """+Inf in predictions -> non_finite_count > 0, status FAILED"""
        y_true = np.array([100.0, 50.0], dtype=np.float32)
        y_pred = np.array([np.inf, 20.0], dtype=np.float32)

        metrics = compute_formal_metrics(y_true, y_pred, 10, 15, "COMPLETED")

        assert metrics["non_finite_count"] == 1
        assert metrics["training_status"] == "FAILED"

    def test_negative_infinity_in_predictions(self):
        """-Inf in predictions -> non_finite_count > 0, status FAILED"""
        y_true = np.array([100.0, 50.0], dtype=np.float32)
        y_pred = np.array([-np.inf, 20.0], dtype=np.float32)

        metrics = compute_formal_metrics(y_true, y_pred, 10, 15, "COMPLETED")

        assert metrics["non_finite_count"] == 1
        assert metrics["training_status"] == "FAILED"

    def test_empty_input(self):
        """Empty arrays -> row_count=0, all metrics 0.0, status FAILED"""
        y_true = np.array([], dtype=np.float32)
        y_pred = np.array([], dtype=np.float32)

        metrics = compute_formal_metrics(y_true, y_pred, 0, 0, "COMPLETED")

        assert metrics["row_count"] == 0
        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["nasa_score"] == 0.0
        assert metrics["training_status"] == "FAILED"

    def test_row_count_mismatch_raises(self):
        """Different length arrays -> ValueError"""
        y_true = np.array([100.0, 50.0], dtype=np.float32)
        y_pred = np.array([105.0], dtype=np.float32)

        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_formal_metrics(y_true, y_pred, 1, 1, "COMPLETED")


class TestValidatePredictionFrame:
    """Tests for validate_prediction_frame function."""

    def test_duplicate_prediction_rows_detected(self):
        """validate_prediction_frame detects duplicate (unit_id, cycle) pairs"""
        df = pd.DataFrame({
            "split": ["predictor_validation"] * 3,
            "unit_id": [1, 1, 2],
            "cycle": [50, 50, 50],  # duplicate for unit 1
            "true_rul_capped": [25.0, 25.0, 30.0],
            "predicted_rul": [20.0, 22.0, 28.0],
            "error": [-5.0, -3.0, -2.0],
        })

        is_valid, errors = validate_prediction_frame(df)
        assert not is_valid
        assert any("duplicate" in e.lower() for e in errors)

    def test_wrong_split_value_detected(self):
        """validate_prediction_frame rejects non-predictor_validation split"""
        df = pd.DataFrame({
            "split": ["train"] * 2,
            "unit_id": [1, 2],
            "cycle": [50, 50],
            "true_rul_capped": [25.0, 30.0],
            "predicted_rul": [20.0, 28.0],
            "error": [-5.0, -2.0],
        })

        is_valid, errors = validate_prediction_frame(df)
        assert not is_valid
        assert any("split" in e.lower() for e in errors)

    def test_wrong_target_column_detected(self):
        """validate_prediction_frame detects missing required columns"""
        df = pd.DataFrame({
            "split": ["predictor_validation"] * 2,
            "unit_id": [1, 2],
            "cycle": [50, 50],
            "true_rul": [25.0, 30.0],  # wrong column name
            "predicted_rul": [20.0, 28.0],
            "error": [-5.0, -2.0],
        })

        is_valid, errors = validate_prediction_frame(df)
        assert not is_valid
        assert any("true_rul_capped" in e for e in errors)


class TestPercentileMethod:
    """Tests for percentile method matching numpy linear interpolation."""

    def test_percentile_method_matches_numpy_linear(self):
        """Verify percentile uses np.percentile with linear interpolation (default)"""
        # For 4 elements [1,2,3,4], p90 with linear = 3.7, p95 = 3.85
        y_true = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32)
        y_pred = np.array([11.0, 12.0, 13.0, 14.0], dtype=np.float32)  # errors = [1,2,3,4]

        metrics = compute_formal_metrics(y_true, y_pred, 1, 1, "COMPLETED")

        # np.percentile([1,2,3,4], 90, interpolation='linear') = 3.7
        assert round(metrics["positive_error_p90"], 4) == 3.7
        # np.percentile([1,2,3,4], 95, interpolation='linear') = 3.85
        assert round(metrics["positive_error_p95"], 4) == 3.85


class TestWriteMetricsJSON:
    """Tests for write_metrics_json function."""

    def test_json_round_trip_preserves_precision(self):
        """Write metrics to JSON, read back, verify 4dp preserved"""
        y_true = np.array([100.0, 50.0, 25.0], dtype=np.float32)
        y_pred = np.array([105.0, 55.0, 22.0], dtype=np.float32)

        metrics = compute_formal_metrics(y_true, y_pred, 10, 15, "COMPLETED")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            write_metrics_json(metrics, path)

            with open(path) as f:
                loaded = json.load(f)

            # All float values should match to 4 decimal places
            for key in [
                "mae", "rmse", "nasa_score", "mean_prediction_error", "bias",
                "overestimation_rate", "mean_positive_error", "positive_error_p90",
                "positive_error_p95", "underestimation_rate",
            ]:
                assert round(loaded[key], 4) == round(metrics[key], 4), f"Mismatch in {key}"

            # Integer fields exact
            for key in [
                "row_count", "overestimation_count", "underestimation_count",
                "non_finite_count", "best_epoch", "final_epoch",
            ]:
                assert loaded[key] == metrics[key], f"Mismatch in {key}"

        finally:
            os.unlink(path)

    def test_write_metrics_json_atomic(self):
        """write_metrics_json uses tmp + fsync + os.replace"""
        y_true = np.array([100.0, 50.0], dtype=np.float32)
        y_pred = np.array([105.0, 55.0], dtype=np.float32)
        metrics = compute_formal_metrics(y_true, y_pred, 1, 1, "COMPLETED")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.json"

            write_metrics_json(metrics, path)

            assert path.exists()

            # No temp files left behind
            temp_files = list(Path(tmpdir).glob("*.tmp.*"))
            assert len(temp_files) == 0

            # Content valid
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["row_count"] == 2


def _make_synthetic_3146_row_frame() -> pd.DataFrame:
    """Create a deterministic synthetic 3,146-row prediction frame.

    Returns DataFrame with exact formal schema:
    - 6 columns: split, unit_id, cycle, true_rul_capped, predicted_rul, error
    - split = "predictor_validation" for all rows
    - unit_id: int32, cycle: int32
    - true_rul_capped, predicted_rul, error: float32
    - error = predicted_rul - true_rul_capped
    - Unique (unit_id, cycle) keys
    - Sorted by (unit_id, cycle) ascending
    - Exactly 3,146 rows
    """
    # Use 15 validation units (matching protocol)
    # Distribute 3146 rows across 15 units deterministically
    np.random.seed(6521)  # Deterministic seed

    n_units = 15
    total_rows = 3146
    rows_per_unit = total_rows // n_units
    remainder = total_rows % n_units

    rows = []
    row_idx = 0
    for unit_id in range(1, n_units + 1):
        n_rows = rows_per_unit + (1 if unit_id <= remainder else 0)
        for i in range(n_rows):
            cycle = 50 + i  # Cycle starts at sequence length
            # Deterministic true_rul_capped in [0, 125]
            true_rul = float(125 - (row_idx % 126))
            true_rul_capped = min(true_rul, 125.0)
            # Deterministic error in [-50, 50]
            error_val = float((row_idx % 101) - 50)
            predicted_rul = true_rul_capped + error_val
            # Ensure error consistency
            assert abs(predicted_rul - true_rul_capped - error_val) < 1e-6

            rows.append({
                "split": "predictor_validation",
                "unit_id": unit_id,
                "cycle": cycle,
                "true_rul_capped": true_rul_capped,
                "predicted_rul": predicted_rul,
                "error": error_val,
            })
            row_idx += 1

    df = pd.DataFrame(rows)

    # Enforce exact dtypes
    df = df.astype({
        "split": "object",  # string
        "unit_id": np.int32,
        "cycle": np.int32,
        "true_rul_capped": np.float32,
        "predicted_rul": np.float32,
        "error": np.float32,
    })

    # Verify invariants
    assert len(df) == 3146
    assert (df["split"] == "predictor_validation").all()
    assert df[["unit_id", "cycle"]].duplicated().sum() == 0
    assert df["unit_id"].is_monotonic_increasing
    for uid, group in df.groupby("unit_id"):
        assert group["cycle"].is_monotonic_increasing
    # Error consistency
    recomputed_error = df["predicted_rul"] - df["true_rul_capped"]
    assert (recomputed_error - df["error"]).abs().max() < 1e-6

    return df


def _compute_expected_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                              best_epoch: int, final_epoch: int,
                              training_status: str) -> dict:
    """Independently compute expected metrics from arrays (not via production code).

    Matches production exactly: converts to float64 for intermediate computation,
    uses round-half-even to 4dp.
    """
    # Production converts to float64 for intermediate computation
    y_true_f64 = y_true.astype(np.float64)
    y_pred_f64 = y_pred.astype(np.float64)
    error = y_pred_f64 - y_true_f64
    row_count = len(error)

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
            "best_epoch": best_epoch,
            "final_epoch": final_epoch,
            "training_status": "FAILED",
        }

    # Non-finite detection
    finite_mask = np.isfinite(error)
    non_finite_count = int(np.sum(~finite_mask))
    error_finite = error[finite_mask]

    # Basic stats
    mae = float(np.mean(np.abs(error_finite))) if error_finite.size > 0 else 0.0
    rmse = float(np.sqrt(np.mean(error_finite**2))) if error_finite.size > 0 else 0.0
    mean_error = float(np.mean(error_finite)) if error_finite.size > 0 else 0.0

    # NASA score (SUM) - matches production formula exactly
    nasa_score = 0.0
    if error_finite.size > 0:
        neg_mask = error_finite < 0
        pos_mask = ~neg_mask
        if np.any(neg_mask):
            nasa_score += np.sum(np.exp(-error_finite[neg_mask] / 13.0) - 1.0)
        if np.any(pos_mask):
            nasa_score += np.sum(np.exp(error_finite[pos_mask] / 10.0) - 1.0)
    nasa_score = float(nasa_score)

    # Over/under estimation
    pos_mask = error_finite > 0
    neg_mask = error_finite < 0
    over_count = int(np.sum(pos_mask))
    under_count = int(np.sum(neg_mask))
    over_rate = over_count / row_count
    under_rate = under_count / row_count

    pos_errors = error_finite[pos_mask]
    if pos_errors.size > 0:
        mean_pos = float(np.mean(pos_errors))
        p90 = float(np.percentile(pos_errors, 90))
        p95 = float(np.percentile(pos_errors, 95))
    else:
        mean_pos = 0.0
        p90 = 0.0
        p95 = 0.0

    # Round to 4dp (round-half-even)
    def r4(x):
        return round(x, 4)

    return {
        "row_count": row_count,
        "mae": r4(mae),
        "rmse": r4(rmse),
        "nasa_score": r4(nasa_score),
        "mean_prediction_error": r4(mean_error),
        "bias": r4(mean_error),
        "overestimation_count": over_count,
        "overestimation_rate": r4(over_rate),
        "mean_positive_error": r4(mean_pos),
        "positive_error_p90": r4(p90),
        "positive_error_p95": r4(p95),
        "underestimation_count": under_count,
        "underestimation_rate": r4(under_rate),
        "non_finite_count": non_finite_count,
        "best_epoch": best_epoch,
        "final_epoch": final_epoch,
        "training_status": "FAILED" if non_finite_count > 0 else training_status,
    }


class TestIndependentRecomputation:
    """Tests for independent recomputation from parquet via public function."""

    def test_independent_recomputation_from_parquet_3146_rows(self):
        """End-to-end: create 3146-row parquet -> call public function -> match independent expected values."""
        # Create synthetic 3146-row frame
        df = _make_synthetic_3146_row_frame()

        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "predictions.parquet"
            df.to_parquet(parquet_path, index=False)

            # Call the ACTUAL public function under test
            recomputed = independently_recompute_metrics_from_parquet(
                parquet_path=parquet_path,
                best_epoch=42,
                final_epoch=50,
                training_status="COMPLETED",
            )

        # Independently compute expected values from the synthetic arrays
        y_true = df["true_rul_capped"].to_numpy(dtype=np.float32)
        y_pred = df["predicted_rul"].to_numpy(dtype=np.float32)
        expected = _compute_expected_metrics(y_true, y_pred, 42, 50, "COMPLETED")

        # Assert exact 17 keys
        assert set(recomputed.keys()) == set(expected.keys())
        assert len(recomputed) == 17

        # Integer fields exact
        for key in ["row_count", "overestimation_count", "underestimation_count",
                    "non_finite_count", "best_epoch", "final_epoch"]:
            assert recomputed[key] == expected[key], f"Integer field {key} mismatch: {recomputed[key]} != {expected[key]}"

        # Float fields match to 4dp
        for key in ["mae", "rmse", "nasa_score", "mean_prediction_error", "bias",
                    "overestimation_rate", "mean_positive_error", "positive_error_p90",
                    "positive_error_p95", "underestimation_rate"]:
            assert round(recomputed[key], 4) == round(expected[key], 4), \
                f"Float field {key} mismatch: {recomputed[key]} != {expected[key]}"

        # Status
        assert recomputed["training_status"] == expected["training_status"]

    def test_independent_recomputation_wrong_row_count_3145(self):
        """Public function fails closed for 3145 rows."""
        df = _make_synthetic_3146_row_frame().iloc[:-1]  # 3145 rows
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "predictions.parquet"
            df.to_parquet(parquet_path, index=False)

            with pytest.raises(ValueError, match="Row count 3145 != 3146"):
                independently_recompute_metrics_from_parquet(
                    parquet_path=parquet_path,
                    best_epoch=1, final_epoch=1, training_status="COMPLETED",
                )

    def test_independent_recomputation_wrong_row_count_3147(self):
        """Public function fails closed for 3147 rows."""
        df = _make_synthetic_3146_row_frame()
        extra_row = df.iloc[:1].copy()
        df = pd.concat([df, extra_row], ignore_index=True)  # 3147 rows
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "predictions.parquet"
            df.to_parquet(parquet_path, index=False)

            with pytest.raises(ValueError, match="Row count 3147 != 3146"):
                independently_recompute_metrics_from_parquet(
                    parquet_path=parquet_path,
                    best_epoch=1, final_epoch=1, training_status="COMPLETED",
                )

    def test_independent_recomputation_missing_column(self):
        """Public function fails closed for missing required column."""
        df = _make_synthetic_3146_row_frame().drop(columns=["true_rul_capped"])
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "predictions.parquet"
            df.to_parquet(parquet_path, index=False)

            with pytest.raises(ValueError, match="Missing columns"):
                independently_recompute_metrics_from_parquet(
                    parquet_path=parquet_path,
                    best_epoch=1, final_epoch=1, training_status="COMPLETED",
                )

    def test_independent_recomputation_duplicate_keys(self):
        """Public function fails closed for duplicate (unit_id, cycle)."""
        df = _make_synthetic_3146_row_frame()
        # Duplicate first row
        df = pd.concat([df, df.iloc[:1]], ignore_index=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "predictions.parquet"
            df.to_parquet(parquet_path, index=False)

            with pytest.raises(ValueError, match="Duplicate"):
                independently_recompute_metrics_from_parquet(
                    parquet_path=parquet_path,
                    best_epoch=1, final_epoch=1, training_status="COMPLETED",
                )

    def test_independent_recomputation_wrong_split(self):
        """Public function fails closed for wrong split value."""
        df = _make_synthetic_3146_row_frame()
        df.loc[0, "split"] = "train"
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "predictions.parquet"
            df.to_parquet(parquet_path, index=False)

            with pytest.raises(ValueError, match="split"):
                independently_recompute_metrics_from_parquet(
                    parquet_path=parquet_path,
                    best_epoch=1, final_epoch=1, training_status="COMPLETED",
                )

    def test_independent_recomputation_error_inconsistent(self):
        """Public function fails closed when error != predicted - true."""
        df = _make_synthetic_3146_row_frame()
        df.loc[0, "error"] = 999.0  # Inconsistent
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "predictions.parquet"
            df.to_parquet(parquet_path, index=False)

            with pytest.raises(ValueError, match="error column"):
                independently_recompute_metrics_from_parquet(
                    parquet_path=parquet_path,
                    best_epoch=1, final_epoch=1, training_status="COMPLETED",
                )

    def test_independent_recomputation_nan(self):
        """Public function handles NaN correctly via validation (fails validation first)."""
        df = _make_synthetic_3146_row_frame()
        df.loc[0, "predicted_rul"] = np.nan
        # Also update error to be consistent
        df.loc[0, "error"] = df.loc[0, "predicted_rul"] - df.loc[0, "true_rul_capped"]
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "predictions.parquet"
            df.to_parquet(parquet_path, index=False)

            # NaN causes non_finite_count > 0 -> status forced to FAILED
            result = independently_recompute_metrics_from_parquet(
                parquet_path=parquet_path,
                best_epoch=1, final_epoch=1, training_status="COMPLETED",
            )
            assert result["non_finite_count"] >= 1
            assert result["training_status"] == "FAILED"

    def test_independent_recomputation_pos_inf(self):
        """Public function handles +Inf correctly."""
        df = _make_synthetic_3146_row_frame()
        df.loc[0, "predicted_rul"] = np.inf
        # Also update error to be consistent
        df.loc[0, "error"] = df.loc[0, "predicted_rul"] - df.loc[0, "true_rul_capped"]
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "predictions.parquet"
            df.to_parquet(parquet_path, index=False)

            result = independently_recompute_metrics_from_parquet(
                parquet_path=parquet_path,
                best_epoch=1, final_epoch=1, training_status="COMPLETED",
            )
            assert result["non_finite_count"] >= 1
            assert result["training_status"] == "FAILED"

    def test_independent_recomputation_neg_inf(self):
        """Public function handles -Inf correctly."""
        df = _make_synthetic_3146_row_frame()
        df.loc[0, "predicted_rul"] = -np.inf
        # Also update error to be consistent
        df.loc[0, "error"] = df.loc[0, "predicted_rul"] - df.loc[0, "true_rul_capped"]
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "predictions.parquet"
            df.to_parquet(parquet_path, index=False)

            result = independently_recompute_metrics_from_parquet(
                parquet_path=parquet_path,
                best_epoch=1, final_epoch=1, training_status="COMPLETED",
            )
            assert result["non_finite_count"] >= 1
            assert result["training_status"] == "FAILED"

    def test_independent_recomputation_wrong_int_dtype(self):
        """Public function fails closed for wrong integer dtype."""
        df = _make_synthetic_3146_row_frame()
        df["unit_id"] = df["unit_id"].astype(np.int64)  # Wrong dtype
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "predictions.parquet"
            df.to_parquet(parquet_path, index=False)

            with pytest.raises(ValueError, match="dtype"):
                independently_recompute_metrics_from_parquet(
                    parquet_path=parquet_path,
                    best_epoch=1, final_epoch=1, training_status="COMPLETED",
                )

    def test_independent_recomputation_wrong_float_dtype(self):
        """Public function fails closed for wrong float dtype."""
        df = _make_synthetic_3146_row_frame()
        df["true_rul_capped"] = df["true_rul_capped"].astype(np.float64)  # Wrong dtype
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "predictions.parquet"
            df.to_parquet(parquet_path, index=False)

            with pytest.raises(ValueError, match="dtype"):
                independently_recompute_metrics_from_parquet(
                    parquet_path=parquet_path,
                    best_epoch=1, final_epoch=1, training_status="COMPLETED",
                )

    def test_independent_recomputation_malformed_parquet(self):
        """Public function fails closed for unreadable/malformed Parquet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "predictions.parquet"
            # Write garbage
            parquet_path.write_bytes(b"not a parquet file")

            with pytest.raises(Exception):
                independently_recompute_metrics_from_parquet(
                    parquet_path=parquet_path,
                    best_epoch=1, final_epoch=1, training_status="COMPLETED",
                )


class TestNASAOverestimationAsymmetry:
    """Tests verifying NASA penalizes overestimation more than underestimation."""

    def test_nasa_overestimation_asymmetry(self):
        """Verify NASA penalizes overestimation more than underestimation for same magnitude"""
        pos_error = np.array([10.0], dtype=np.float32)
        neg_error = np.array([-10.0], dtype=np.float32)

        pos_score = compute_nasa_score(pos_error)
        neg_score = compute_nasa_score(neg_error)

        # exp(10/10) - 1 = e - 1 ≈ 1.7183
        # exp(10/13) - 1 = exp(0.76923...) - 1 ≈ 1.1581
        assert pos_score > neg_score
        assert round(pos_score, 4) == 1.7183
        assert round(neg_score, 4) == 1.1581


class TestBiasEqualsMeanPredictionError:
    """Test that bias field always equals mean_prediction_error field."""

    def test_bias_equals_mean_prediction_error_always(self):
        """bias must always equal mean_prediction_error"""
        for _ in range(100):
            n = np.random.randint(1, 100)
            y_true = np.random.uniform(0, 125, n).astype(np.float32)
            y_pred = np.random.uniform(0, 125, n).astype(np.float32)

            metrics = compute_formal_metrics(y_true, y_pred, 1, 1, "COMPLETED")

            assert metrics["bias"] == metrics["mean_prediction_error"]


class TestDeterminism:
    """Test deterministic output."""

    def test_deterministic_output(self):
        """Same inputs must produce bitwise-identical outputs"""
        y_true = np.array([100.0, 50.0, 25.0, 10.0, 5.0], dtype=np.float32)
        y_pred = np.array([105.0, 55.0, 20.0, 12.0, 3.0], dtype=np.float32)

        m1 = compute_formal_metrics(y_true, y_pred, 10, 15, "COMPLETED")
        m2 = compute_formal_metrics(y_true, y_pred, 10, 15, "COMPLETED")

        # Exact equality including float bit patterns
        for key in m1:
            assert m1[key] == m2[key], f"Non-deterministic: {key}"


class TestInvalidTrainingStatus:
    """Test invalid training status handling."""

    def test_invalid_training_status_raises(self):
        """Invalid status string -> ValueError"""
        y_true = np.array([100.0], dtype=np.float32)
        y_pred = np.array([105.0], dtype=np.float32)

        with pytest.raises(ValueError, match="training_status"):
            compute_formal_metrics(y_true, y_pred, 1, 1, "UNKNOWN_STATUS")


class TestComputeNASAScore:
    """Standalone tests for compute_nasa_score function."""

    def test_compute_nasa_score_standalone(self):
        """Test compute_nasa_score function independently"""
        error = np.array([5.0, 5.0, -5.0, 2.0, -2.0], dtype=np.float32)
        score = compute_nasa_score(error)

        # Reference: 2.1542059338553727
        assert round(score, 4) == 2.1542


class TestAttachTrainingMetadata:
    """Tests for attach_training_metadata function."""

    def test_attach_training_metadata(self):
        """Attach epoch metadata to metrics dict"""
        metrics = {"mae": 1.0, "rmse": 2.0}
        training_history = [{"epoch": 1, "val_rmse": 0.5}]
        checkpoint_info = {"best_epoch": 5, "final_epoch": 10, "best_val_rmse": 0.3}

        result = attach_training_metadata(metrics, training_history, checkpoint_info)

        assert result["best_epoch"] == 5
        assert result["final_epoch"] == 10
        assert result["mae"] == 1.0  # Original metrics preserved


# Additional edge case tests per plan

class TestEdgeCases:
    """Additional edge case tests."""

    def test_single_row_input(self):
        """Single row input should work correctly"""
        y_true = np.array([50.0], dtype=np.float32)
        y_pred = np.array([55.0], dtype=np.float32)

        metrics = compute_formal_metrics(y_true, y_pred, 1, 1, "COMPLETED")

        assert metrics["row_count"] == 1
        assert metrics["mae"] == 5.0
        assert metrics["rmse"] == 5.0
        assert metrics["overestimation_count"] == 1
        assert metrics["underestimation_count"] == 0

    def test_all_zero_errors(self):
        """All zero errors (already tested in zero_error_vector but with different values)"""
        y_true = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        y_pred = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        metrics = compute_formal_metrics(y_true, y_pred, 1, 1, "COMPLETED")

        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["nasa_score"] == 0.0
        assert metrics["overestimation_count"] == 0
        assert metrics["underestimation_count"] == 0

    def test_large_errors(self):
        """Large but finite errors"""
        y_true = np.array([100.0, 50.0], dtype=np.float32)
        y_pred = np.array([200.0, 0.0], dtype=np.float32)  # errors = [100, -50]

        metrics = compute_formal_metrics(y_true, y_pred, 1, 1, "COMPLETED")

        assert metrics["overestimation_count"] == 1
        assert metrics["underestimation_count"] == 1
        assert metrics["mae"] == 75.0
        assert metrics["nasa_score"] > 0  # Should compute without overflow

    def test_float32_and_float64_inputs(self):
        """Both float32 and float64 inputs should work"""
        y_true_32 = np.array([100.0, 50.0], dtype=np.float32)
        y_pred_32 = np.array([105.0, 55.0], dtype=np.float32)
        y_true_64 = np.array([100.0, 50.0], dtype=np.float64)
        y_pred_64 = np.array([105.0, 55.0], dtype=np.float64)

        m32 = compute_formal_metrics(y_true_32, y_pred_32, 1, 1, "COMPLETED")
        m64 = compute_formal_metrics(y_true_64, y_pred_64, 1, 1, "COMPLETED")

        for key in m32:
            if isinstance(m32[key], float):
                assert round(m32[key], 4) == round(m64[key], 4), f"Mismatch in {key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])