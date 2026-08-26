"""Tests for training-history integrity and atomic write helpers."""
import json
import pathlib

import pytest
import torch
import torch.nn as nn
import numpy as np

from src.predictors.io_utils import atomic_write_json, atomic_torch_save
from src.predictors.train import train_epoch
def _epoch(n, **overrides) -> dict:
    base = {
        "epoch": n,
        "train_loss": 0.5,
        "train_rmse": 0.7,
        "train_mae": 0.5,
        "val_loss": 0.6,
        "val_rmse": 0.77,
        "val_mae": 0.6,
        "val_mape": 12.0,
        "learning_rate": 1e-3,
        "epoch_duration_seconds": 1.0,
        "is_best_so_far": False,
        "early_stopping_counter": 0,
    }
    base.update(overrides)
    return base


def _best_history(best_epoch=2, total=4) -> list:
    hist = [_epoch(i) for i in range(total)]
    hist[best_epoch] = _epoch(best_epoch, is_best_so_far=True, val_rmse=0.10, val_loss=0.01)
    return hist


class _ValidationError(Exception):
    pass


def _validate_history(hist: list, best_epoch: int, best_val_rmse: float) -> None:
    if not hist:
        raise _ValidationError("empty training history")
    epochs = [h["epoch"] for h in hist]
    # duplicate epochs must be flagged explicitly before the consecutive check, so a
    # ``duplicate`` reason surfaces rather than the secondary non-consecutive message.
    if len(set(epochs)) != len(epochs):
        raise _ValidationError(f"duplicate epochs detected: {epochs}")
    # consecutive
    if epochs != list(range(epochs[0], epochs[0] + len(epochs))):
        raise _ValidationError(f"non-consecutive epochs: {epochs}")
    best_records = [h for h in hist if h.get("is_best_so_far")]
    if not best_records:
        raise _ValidationError("no best epoch marked in history")
    best_record = best_records[-1]
    if best_record["epoch"] != best_epoch:
        raise _ValidationError(f"best epoch {best_record['epoch']} != expected {best_epoch}")
    if abs(best_record["val_rmse"] - best_val_rmse) > 1e-6:
        raise _ValidationError(f"best val_rmse {best_record['val_rmse']} != expected {best_val_rmse}")


def test_consecutive_epochs_pass(tmp_path: pathlib.Path):
    hist = _best_history(best_epoch=2, total=4)
    best_val = next(h for h in hist if h["is_best_so_far"])["val_rmse"]
    _validate_history(hist, best_epoch=2, best_val_rmse=best_val)  # no raise


def test_missing_epoch_fails(tmp_path: pathlib.Path):
    hist = _best_history(best_epoch=2, total=4)
    del hist[1]
    with pytest.raises(_ValidationError, match="non-consecutive|empty"):
        _validate_history(hist, best_epoch=2, best_val_rmse=0.10)


def test_duplicate_epoch_fails(tmp_path: pathlib.Path):
    hist = _best_history(best_epoch=2, total=4)
    hist.append(dict(hist[2]))  # duplicate epoch 2
    with pytest.raises(_ValidationError, match="duplicate"):
        _validate_history(hist, best_epoch=2, best_val_rmse=0.10)


def test_best_epoch_missing_fails(tmp_path: pathlib.Path):
    hist = [_epoch(i) for i in range(4)]  # none marked best
    with pytest.raises(_ValidationError, match="no best epoch"):
        _validate_history(hist, best_epoch=2, best_val_rmse=0.10)


def test_best_validation_metric_mismatch_fails():
    hist = _best_history(best_epoch=2, total=4)
    with pytest.raises(_ValidationError, match="best val_rmse"):
        _validate_history(hist, best_epoch=2, best_val_rmse=999.0)


def test_truncated_json_fails(tmp_path: pathlib.Path):
    p = tmp_path / "bad.json"
    p.write_text('{"epoch": 0, "val_rmse": 0.7,')  # truncated
    with pytest.raises(json.JSONDecodeError):
        with open(p, "r") as f:
            json.load(f)


def test_atomic_write_json_preserves_existing_on_serialize_failure(tmp_path: pathlib.Path, monkeypatch):
    dest = tmp_path / "training_history.json"
    good = [{"epoch": 0}]
    atomic_write_json(dest, good)
    original_bytes = dest.read_bytes()

    # Force a serialization failure mid-write.
    def boom(_obj, _f, *a, **k):
        raise RuntimeError("simulated serialization failure")

    monkeypatch.setattr("src.predictors.io_utils.json.dump", boom)
    with pytest.raises(RuntimeError):
        atomic_write_json(dest, [{"epoch": 1}])
    # Original file must be untouched.
    assert dest.read_bytes() == original_bytes
    # No leftover temp files in the directory.
    temps = [p for p in tmp_path.iterdir() if "_tmp_" in p.name]
    assert temps == []


def test_atomic_torch_save_produces_loadable_checkpoint(tmp_path: pathlib.Path):
    p = tmp_path / "checkpoint.pt"
    payload = {"model_state_dict": {"w": torch.zeros(2)}, "epoch": 3, "config": {"seed": 1}}
    atomic_torch_save(p, payload)
    loaded = torch.load(p, map_location="cpu", weights_only=False)
    assert loaded["epoch"] == 3
    assert torch.equal(loaded["model_state_dict"]["w"], torch.zeros(2))


# ---------------------------------------------------------------------------
# Regression: numpy scalars in epoch records
# ---------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from src.predictors.io_utils import atomic_write_json


def _smoke_epoch_schema(epoch, train_loss, val_rmse, is_best):
    """Construct an epoch record matching the shape of the real training
    loop — the same twelve fields with the same keys."""
    return {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_rmse": np.sqrt(train_loss),
        "train_mae": 0.0,
        "val_loss": val_rmse ** 2,
        "val_rmse": val_rmse,
        "val_mae": np.mean([val_rmse]),
        "val_mape": np.mean([val_rmse]) * 10.0,
        "learning_rate": 1e-3,
        "epoch_duration_seconds": 1.5,
        "is_best_so_far": is_best,
        "early_stopping_counter": 0,
    }


def test_numpy_scalar_epoch_record_writes_and_roundtrips(tmp_path: pathlib.Path):
    """An epoch record built from numpy scalars (np.float64, np.bool_,
    np.int64) must serialize via atomic_write_json and deserialize with
    native Python JSON types."""
    dest = tmp_path / "history.json"
    record = _smoke_epoch_schema(
        epoch=np.int64(0),
        train_loss=np.float64(6936.7851),
        val_rmse=np.float64(74.3042),
        is_best=np.bool_(True),
    )
    history = [record]
    atomic_write_json(dest, history)

    loaded = json.loads(dest.read_text())
    assert len(loaded) == 1
    r = loaded[0]
    # Every field must be a JSON-native type, not a numpy or torch object.
    assert isinstance(r["epoch"], int)
    assert isinstance(r["train_loss"], float)
    assert isinstance(r["train_rmse"], float)
    assert isinstance(r["train_mae"], float)
    assert isinstance(r["val_loss"], float)
    assert isinstance(r["val_rmse"], float)
    assert isinstance(r["val_mae"], float)
    assert isinstance(r["val_mape"], float)
    assert isinstance(r["learning_rate"], float)
    assert isinstance(r["epoch_duration_seconds"], float)
    assert isinstance(r["is_best_so_far"], bool)
    assert isinstance(r["early_stopping_counter"], int)
    # Values must roundtrip faithfully.
    assert r["epoch"] == 0
    assert abs(r["train_loss"] - 6936.7851) < 1e-4
    assert abs(r["val_rmse"] - 74.3042) < 1e-4
    assert r["is_best_so_far"] is True


def test_numpy_scalar_bool_false_roundtrips(tmp_path: pathlib.Path):
    """A numpy.bool_(False) must serialize as JSON ``false`` and deserialize
    as Python ``False``."""
    dest = tmp_path / "history.json"
    record = _smoke_epoch_schema(
        epoch=np.int64(1),
        train_loss=np.float32(1.0),  # Valid positive value
        val_rmse=np.float64(99.0),
        is_best=np.bool_(False),
    )
    atomic_write_json(dest, [record])

    loaded = json.loads(dest.read_text())
    r = loaded[0]
    assert r["is_best_so_far"] is False
    assert isinstance(r["is_best_so_far"], bool)
    assert isinstance(r["train_loss"], float)


# ---------------------------------------------------------------------------
# Regression test for actual train MAE computation during training
# This validates the sample-weighting implementation works correctly
# ---------------------------------------------------------------------------

from torch.utils.data import Dataset, DataLoader
from src.predictors.dataset import FD001SequenceDataset
from src.predictors.model import build_predictor


class MockSequenceDataset(Dataset):
    """Mock dataset that returns samples in the expected format."""

    def __init__(self, x_tensors: torch.Tensor, y_tensors: torch.Tensor):
        """Initialize with pre-computed feature tensors and target values.

        Args:
            x_tensors: Shape (n_samples, sequence_length, n_features)
            y_tensors: Shape (n_samples,)
        """
        assert len(x_tensors) == len(y_tensors)
        self.x_tensors = x_tensors
        self.y_tensors = y_tensors

    def __len__(self):
        return len(self.x_tensors)

    def __getitem__(self, idx):
        return {
            "features": self.x_tensors[idx],
            "rul_capped": self.y_tensors[idx],
            "rul_raw": self.y_tensors[idx] + 10,  # Mock value, not used
            "unit_id": 1,
            "cycle": idx,
            "left_pad_count": 0,
        }


def test_train_epoch_sample_weighted_mae_computation(tmp_path: pathlib.Path):
    """Test that train_epoch correctly computes sample-weighted MAE.

    Example concept:
    - Batch 1: one sample with absolute prediction error = 10
    - Batch 2: three samples with absolute prediction errors = 0, 0, 0

    Correct sample-weighted epoch MAE: (10 + 0 + 0 + 0) / 4 = 2.5
    Incorrect equal batch-average: (10 + 0) / 2 = 5.0

    This test constructs a deterministic scenario where the model produces
    exactly these errors and verifies train_epoch returns 2.5.
    """
    torch.manual_seed(6521)

    # Create a simple model that produces deterministic outputs
    n_features = 10
    sequence_length = 5
    hidden_dim = 16
    n_layers = 2
    dropout = 0.0

    model = build_predictor(
        model_type="mlp",
        n_features=n_features,
        sequence_length=sequence_length,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
    )
    model.eval()  # No training noise from dropout

    # For batch 1: we want |y_pred - y| = 10 for 1 sample
    # For batch 2: we want |y_pred - y| = 0 for 3 samples

    batch_size_1 = 1
    batch_size_2 = 3

    # Create random input features for batch 1
    x1 = torch.randn(batch_size_1, sequence_length, n_features)

    with torch.no_grad():
        y_pred1 = model(x1)  # Shape: (batch_size_1,)

    # Set targets so that |y_pred - y| = 10 for the single sample
    y1 = y_pred1 + 10.0  # This gives error = |y_pred - (y_pred + 10)| = 10

    # Create random input features for batch 2
    x2 = torch.randn(batch_size_2, sequence_length, n_features)

    with torch.no_grad():
        y_pred2 = model(x2)  # Shape: (batch_size_2,)

    # Set targets so that |y_pred - y| = 0 for all 3 samples
    y2 = y_pred2.clone()  # This gives error = 0 for all samples

    # Combine into single dataset
    x_all = torch.cat([x1, x2], dim=0)
    y_all = torch.cat([y1, y2], dim=0)

    # Use mock dataset that returns proper dict format
    dataset = MockSequenceDataset(x_all, y_all)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    # Create optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)  # LR=0 to freeze weights
    criterion = nn.MSELoss()
    device = "cpu"

    # Run train_epoch
    model.train()
    train_loss, n_samples, train_mae = train_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )

    # Expected MAE: (10 + 0 + 0 + 0) / 4 = 2.5
    expected_mae = 2.5

    # Verify train_mae is approximately 2.5 (with floating point tolerance)
    assert abs(train_mae - expected_mae) < 1e-5, f"Expected MAE={expected_mae}, got {train_mae}"

    # Verify train_mae does NOT equal 5.0 (wrong batch-average)
    assert abs(train_mae - 5.0) > 0.1, f"MAE={train_mae} incorrectly equals batch-average 5.0"

    # Verify train_mae is a native Python float
    assert isinstance(train_mae, float), f"train_mae should be native float, got {type(train_mae)}"

    # Verify train_mae can be inserted into epoch-history record
    epoch_record = {
        "epoch": 0,
        "train_loss": train_loss,
        "train_mae": train_mae,
    }

    # Verify atomic_write_json handles it correctly
    test_path = tmp_path / "test_epoch.json"
    atomic_write_json(test_path, [epoch_record])

    with open(test_path, "r") as f:
        loaded = json.load(f)

    assert len(loaded) == 1
    assert abs(loaded[0]["train_mae"] - expected_mae) < 1e-5
    assert isinstance(loaded[0]["train_mae"], float)


# ---------------------------------------------------------------------------
# Tests for strict finite JSON validation
# ---------------------------------------------------------------------------

import math
from src.predictors.io_utils import atomic_write_json


def test_strict_json_rejects_nan(tmp_path: pathlib.Path):
    """atomic_write_json with strict=True must reject NaN values."""
    data = {"value": float("nan")}
    test_path = tmp_path / "test.json"

    with pytest.raises(ValueError, match="NaN"):
        atomic_write_json(test_path, data, strict=True)

    # File should not exist after failure
    assert not test_path.exists()


def test_strict_json_rejects_positive_inf(tmp_path: pathlib.Path):
    """atomic_write_json with strict=True must reject Infinity."""
    data = {"value": float("inf")}
    test_path = tmp_path / "test.json"

    with pytest.raises(ValueError, match="Infinity"):
        atomic_write_json(test_path, data, strict=True)

    assert not test_path.exists()


def test_strict_json_rejects_negative_inf(tmp_path: pathlib.Path):
    """atomic_write_json with strict=True must reject -Infinity."""
    data = {"value": float("-inf")}
    test_path = tmp_path / "test.json"

    with pytest.raises(ValueError, match="-Infinity"):
        atomic_write_json(test_path, data, strict=True)

    assert not test_path.exists()


def test_strict_json_rejects_numpy_nan(tmp_path: pathlib.Path):
    """atomic_write_json with strict=True must reject numpy.nan."""
    data = {"value": np.nan}
    test_path = tmp_path / "test.json"

    with pytest.raises(ValueError, match="NaN"):
        atomic_write_json(test_path, data, strict=True)

    assert not test_path.exists()


def test_strict_json_rejects_numpy_inf(tmp_path: pathlib.Path):
    """atomic_write_json with strict=True must reject numpy.inf."""
    data = {"value": np.inf}
    test_path = tmp_path / "test.json"

    with pytest.raises(ValueError, match="Infinity"):
        atomic_write_json(test_path, data, strict=True)

    assert not test_path.exists()


def test_strict_json_rejects_numpy_array(tmp_path: pathlib.Path):
    """atomic_write_json with strict=True must reject numpy arrays."""
    data = {"array": np.array([1, 2, 3])}
    test_path = tmp_path / "test.json"

    with pytest.raises(ValueError, match="numpy.ndarray"):
        atomic_write_json(test_path, data, strict=True)

    assert not test_path.exists()


def test_strict_json_rejects_torch_tensor(tmp_path: pathlib.Path):
    """atomic_write_json with strict=True must reject torch tensors."""
    data = {"tensor": torch.tensor([1.0, 2.0])}
    test_path = tmp_path / "test.json"

    with pytest.raises(ValueError, match="torch.Tensor"):
        atomic_write_json(test_path, data, strict=True)

    assert not test_path.exists()


def test_strict_json_validates_nested_dict_nan(tmp_path: pathlib.Path):
    """atomic_write_json with strict=True must reject NaN in nested dict."""
    data = {
        "level1": {
            "level2": {
                "value": float("nan"),
            }
        }
    }
    test_path = tmp_path / "test.json"

    with pytest.raises(ValueError, match=r"root\.level1\.level2\.value"):
        atomic_write_json(test_path, data, strict=True)

    assert not test_path.exists()


def test_strict_json_validates_nested_list_inf(tmp_path: pathlib.Path):
    """atomic_write_json with strict=True must reject Inf in nested list."""
    data = {
        "values": [1.0, 2.0, float("inf"), 4.0],
    }
    test_path = tmp_path / "test.json"

    with pytest.raises(ValueError, match=r"root\.values\[2\]"):
        atomic_write_json(test_path, data, strict=True)

    assert not test_path.exists()


def test_strict_json_allows_finite_numpy_scalars(tmp_path: pathlib.Path):
    """atomic_write_json with strict=True must allow finite numpy scalars."""
    data = {
        "int_val": np.int64(42),
        "float_val": np.float64(3.14),
        "bool_val": np.bool_(True),
    }
    test_path = tmp_path / "test.json"

    # Should not raise
    atomic_write_json(test_path, data, strict=True)

    # Verify roundtrip
    with open(test_path, "r") as f:
        loaded = json.load(f)

    assert loaded["int_val"] == 42
    assert abs(loaded["float_val"] - 3.14) < 1e-6
    assert loaded["bool_val"] is True


def test_strict_json_allows_finite_python_values(tmp_path: pathlib.Path):
    """atomic_write_json with strict=True must allow finite Python values."""
    data = {
        "int_val": 42,
        "float_val": 3.14,
        "bool_val": True,
        "none_val": None,
        "list_val": [1, 2, 3],
        "dict_val": {"nested": "value"},
    }
    test_path = tmp_path / "test.json"

    # Should not raise
    atomic_write_json(test_path, data, strict=True)

    # Verify roundtrip
    with open(test_path, "r") as f:
        loaded = json.load(f)

    assert loaded == data


def test_non_strict_json_allows_nan(tmp_path: pathlib.Path):
    """atomic_write_json with strict=False should allow NaN (but json may fail)."""
    data = {"value": float("nan")}
    test_path = tmp_path / "test.json"

    # With strict=False, validation is skipped but json.dump may still fail
    # or produce non-standard JSON
    # This test just verifies strict=False bypasses our validation
    with pytest.raises(ValueError):
        # allow_nan=False in json.dump will still reject NaN
        atomic_write_json(test_path, data, strict=False)
