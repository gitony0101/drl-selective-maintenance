"""Utility helpers for atomic file writes.

The repository writes JSON configuration, training history, model checkpoints,
and parquet prediction caches. Partial writes (e.g., a process crash while
writing a JSON file) can leave an invalid artifact that downstream code
relies on. All write operations are therefore performed atomically via a
temporary sibling file which is flushed, synced, closed, and finally moved
into place with ``os.replace``.

The temporary file name must be unique to avoid race conditions when the same
function is invoked concurrently. We use ``tempfile.NamedTemporaryFile`` with
``delete=False`` and place the file in the same directory as the final
destination.

Functions provided:
    atomic_write_json(path, data, strict=True) -> None
    atomic_torch_save(path, obj)   -> None
    atomic_parquet_write(df, path) -> None

All helpers raise ``OSError`` on failure and ensure that a partially written
temporary file is removed, leaving any existing valid destination untouched.

With strict=True (default), atomic_write_json rejects NaN, Infinity, -Infinity,
numpy arrays, and torch tensors with ValueError. This ensures formal artifacts
contain only standard finite JSON values.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import torch

PathLike = pathlib.Path | str


def _json_default_for_scalars(obj: Any) -> Any:
    """Convert known numeric/boolean scalars to native Python types for JSON.

    Only handles ``numpy.integer``, ``numpy.floating``, and ``numpy.bool_``.
    Arrays, tensors, and every other object are intentionally not handled so
    the error surface remains tight and discoverable.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__!r} is not JSON serializable")


def _is_finite_value(obj: Any) -> bool:
    """Recursively check if a value is finite (no NaN, Inf, -Inf).

    Handles:
    - Python int/float/bool: floats must be finite
    - NumPy integer/floating/bool scalars: floats must be finite
    - Lists and dicts: recursively check all elements/values
    - Tensors and ndarrays: not allowed (must be converted first)

    Returns True if the value is finite and JSON-serializable.
    Returns False if any nested value is NaN, Inf, -Inf, or unsupported.
    """
    if obj is None:
        return True

    # Python bool (must check before int since bool is subclass of int)
    if isinstance(obj, bool):
        return True

    # Python int
    if isinstance(obj, int):
        return True

    # Python float (must be finite)
    if isinstance(obj, float):
        return math.isfinite(obj)

    # NumPy scalars
    if isinstance(obj, np.integer):
        return True

    if isinstance(obj, np.floating):
        return math.isfinite(float(obj))

    if isinstance(obj, np.bool_):
        return True

    # NumPy arrays and torch tensors are not allowed
    if isinstance(obj, np.ndarray):
        return False

    if isinstance(obj, torch.Tensor):
        return False

    # Lists: recursively check all elements
    if isinstance(obj, list):
        return all(_is_finite_value(item) for item in obj)

    # Dicts: recursively check all values
    if isinstance(obj, dict):
        return all(_is_finite_value(v) for v in obj.values())

    # Tuples: recursively check all elements
    if isinstance(obj, tuple):
        return all(_is_finite_value(item) for item in obj)

    # Other types: not supported
    return False


def _validate_finite_json(data: Any, path: str = "root") -> None:
    """Validate that data contains only finite JSON-serializable values.

    Raises ValueError with detailed path information on first non-finite value.
    """
    if data is None:
        return

    # Python bool (must check before other types)
    if isinstance(data, bool):
        return

    # Python str (always allowed)
    if isinstance(data, str):
        return

    # Python int
    if isinstance(data, int):
        return

    # Python float (must be finite)
    if isinstance(data, float):
        if not math.isfinite(data):
            if math.isnan(data):
                raise ValueError(f"Non-finite value at {path}: NaN is not JSON-serializable")
            elif math.isinf(data):
                if data > 0:
                    raise ValueError(f"Non-finite value at {path}: Infinity is not JSON-serializable")
                else:
                    raise ValueError(f"Non-finite value at {path}: -Infinity is not JSON-serializable")
        return

    # NumPy scalars
    if isinstance(data, np.integer):
        return

    if isinstance(data, np.floating):
        val = float(data)
        if not math.isfinite(val):
            if math.isnan(val):
                raise ValueError(f"Non-finite numpy scalar at {path}: numpy.nan is not JSON-serializable")
            elif val > 0:
                raise ValueError(f"Non-finite numpy scalar at {path}: numpy.inf is not JSON-serializable")
            else:
                raise ValueError(f"Non-finite numpy scalar at {path}: -numpy.inf is not JSON-serializable")
        return

    if isinstance(data, np.bool_):
        return

    if isinstance(data, np.ndarray):
        raise ValueError(f"Unsupported type at {path}: numpy.ndarray must be converted to list/scalar before JSON serialization")

    if isinstance(data, torch.Tensor):
        raise ValueError(f"Unsupported type at {path}: torch.Tensor must be converted to numpy/python scalar before JSON serialization")

    if isinstance(data, list):
        for i, item in enumerate(data):
            _validate_finite_json(item, f"{path}[{i}]")
        return

    if isinstance(data, dict):
        for key, value in data.items():
            _validate_finite_json(value, f"{path}.{key}")
        return

    if isinstance(data, tuple):
        for i, item in enumerate(data):
            _validate_finite_json(item, f"{path}[{i}]")
        return

    # Other types that json.dump can handle (e.g., str subclasses)
    if isinstance(data, (str,)):
        return

    raise ValueError(f"Unsupported type at {path}: {type(data).__name__} is not JSON-serializable")


def _temp_path(destination: pathlib.Path) -> pathlib.Path:
    """Create a unique temporary file path in the same directory as *destination*.

    The temporary file is created with ``tempfile.NamedTemporaryFile`` using a
    random name (based on a UUID) so concurrent calls do not clash. The file is
    closed immediately; the caller will open it for writing.
    """
    directory = destination.parent
    tmp = tempfile.NamedTemporaryFile(
        prefix=destination.stem + "_tmp_",
        suffix=destination.suffix,
        dir=directory,
        delete=False,
    )
    tmp_path = pathlib.Path(tmp.name)
    tmp.close()
    return tmp_path


def atomic_write_json(path: PathLike, data: Any, strict: bool = True) -> None:
    """Write *data* as JSON to *path* atomically.

    The JSON is first written to a temporary sibling file, flushed, ``os.fsync``
    called for durability, then ``os.replace`` moves the temp file into place.
    If any step fails, the temporary file is removed and the original file is
    left untouched.

    NumPy scalars (``np.float64``, ``np.bool_``, etc.) are converted to their
    native Python equivalents during serialization so the writer never fails
    with ``TypeError`` on otherwise-valid numeric/metric data. NumPy arrays,
    PyTorch tensors, and complex objects remain rejected — this is a safety
    net for scalars only.

    With strict=True (default), the function rejects NaN, Infinity, -Infinity,
    numpy arrays, and torch tensors with ValueError before attempting to write.
    This ensures formal artifacts contain only standard finite JSON values.
    Set strict=False to allow NaN (though json.dump may still fail).

    Args:
        path: Destination file path
        data: Data to serialize as JSON
        strict: If True (default), reject NaN, Inf, -Inf, numpy arrays, and tensors
                with ValueError. If False, skip finite-value validation.
    """
    dest = pathlib.Path(path)

    # Strict finite-value validation
    if strict:
        _validate_finite_json(data)

    tmp_path = _temp_path(dest)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True,
                      default=_json_default_for_scalars,
                      allow_nan=False)  # Reject NaN/Inf at json.dump level too
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, dest)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        finally:
            raise


def atomic_torch_save(path: PathLike, obj: Any) -> None:
    """Save a PyTorch object (e.g., checkpoint) atomically.

    ``torch.save`` writes to the temporary file, then we ``fsync`` and replace.
    """
    dest = pathlib.Path(path)
    tmp_path = _temp_path(dest)
    try:
        torch.save(obj, tmp_path)
        fd = os.open(tmp_path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp_path, dest)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        finally:
            raise


def atomic_parquet_write(df: pd.DataFrame, path: PathLike) -> None:
    """Write a pandas DataFrame to Parquet atomically.

    The DataFrame is first written to a temporary file, read back to verify
    integrity, then the temporary file is moved into place.
    """
    dest = pathlib.Path(path)
    tmp_path = _temp_path(dest)
    try:
        df.to_parquet(tmp_path, index=False)
        # Verify readability
        _ = pd.read_parquet(tmp_path)
        fd = os.open(tmp_path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp_path, dest)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        finally:
            raise