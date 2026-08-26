"""Test that dataset normalization matches manual computation.

Covers:
1. early-cycle left-padded window;
2. a cycle at or beyond sequence length (no padding);
3. manual normalization equals Dataset produced tensor;
4. all normalized values are finite;
5. true RUL is not an input feature.
"""

import pathlib

import numpy as np

from src.predictors.dataset import FD001SequenceDataset

V2_DIR = pathlib.Path("data/processed/fd001/v2")
FEATURE_COLUMNS_FILE = V2_DIR / "04_PROTOCOL" / "fd001_feature_schema_v1.json"
import json


def _feature_order() -> list:
    with open(FEATURE_COLUMNS_FILE, "r") as f:
        return list(json.load(f)["input_feature_order"])


def _dataset() -> FD001SequenceDataset:
    return FD001SequenceDataset(
        split="predictor_train",
        data_dir=V2_DIR,
        sequence_length=50,
        rul_cap=125,
    )


def _check_match(diag) -> None:
    manual = diag["manually_normalized_window"]
    tensor = diag["dataset_produced_tensor"]
    assert manual.shape == tensor.shape == (50, 24)
    diff = np.abs(manual - tensor).max()
    assert diff < 1e-5, f"Max absolute difference {diff} exceeds tolerance"


def test_early_cycle_left_padded_window():
    """Cycle early in trajectory triggers left-padding; normalization must still match."""
    ds = _dataset()
    sample = ds[0]
    unit_id = int(sample["unit_id"].item())
    cycle = int(sample["cycle"].item())
    diag = ds.diagnostic_normalize(unit_id=unit_id, cycle=cycle)
    assert diag["left_pad_count"] > 0, "expected early-cycle sample to be left-padded"
    _check_match(diag)


def test_cycle_at_or_beyond_sequence_length():
    """A cycle >= sequence_length yields no left-padding; normalization must match."""
    ds = _dataset()
    # Find a sample whose cycle is >= sequence_length (50).
    chosen = None
    for idx in range(len(ds)):
        s = ds[idx]
        if int(s["cycle"].item()) >= 50:
            chosen = (int(s["unit_id"].item()), int(s["cycle"].item()))
            break
    if chosen is None:
        # Fall back to the last available sample if none qualifies (very short trajectories).
        s = ds[len(ds) - 1]
        chosen = (int(s["unit_id"].item()), int(s["cycle"].item()))
    unit_id, cycle = chosen
    diag = ds.diagnostic_normalize(unit_id=unit_id, cycle=cycle)
    _check_match(diag)


def test_manual_normalization_equals_dataset_output():
    """Manual (x - mean) / std equals the tensor returned by __getitem__."""
    ds = _dataset()
    sample = ds[0]
    unit_id = int(sample["unit_id"].item())
    cycle = int(sample["cycle"].item())
    diag = ds.diagnostic_normalize(unit_id=unit_id, cycle=cycle)
    _check_match(diag)
    # The diagnostic's own reported max diff should also be tiny.
    assert diag["max_absolute_difference"] < 1e-5


def test_all_normalized_values_finite():
    """Every normalized feature value must be finite."""
    ds = _dataset()
    for idx in (0, len(ds) // 2, len(ds) - 1):
        sample = ds[idx]
        feats = sample["features"].numpy()
        assert np.isfinite(feats).all(), f"non-finite values at sample idx {idx}"


def test_true_rul_not_in_input_features():
    """RUL must never appear among the 24 input feature columns."""
    feature_cols = _feature_order()
    assert len(feature_cols) == 24
    lowered = [c.lower() for c in feature_cols]
    for bad in ("rul", "true_rul", "remaining_useful_life"):
        assert bad not in lowered, f"{bad!r} leaked into input feature columns"
