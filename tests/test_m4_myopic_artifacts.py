"""
Test Milestone 4 Exact Myopic artifact utilities.

Tests:
- JSON validation (NaN, Inf, numpy arrays rejected)
- Atomic JSON writing (temp file, rename)
- File hash computation
- Artifact writer with provenance tracking
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from optimizers import (
    validate_json_serializable,
    convert_for_json,
    write_atomic_json,
    compute_file_hash,
    compute_data_hash,
    MyopicArtifactWriter,
    get_git_commit,
)


class TestJSONValidation:
    """Test JSON serialization validation."""

    def test_valid_primitives(self):
        """Valid primitives pass validation."""
        validate_json_serializable(None)
        validate_json_serializable(True)
        validate_json_serializable(False)
        validate_json_serializable(42)
        validate_json_serializable(3.14)
        validate_json_serializable("hello")

    def test_nan_rejected(self):
        """NaN values are rejected."""
        with pytest.raises(ValueError, match="NaN"):
            validate_json_serializable(float('nan'))

        with pytest.raises(ValueError, match="NaN"):
            validate_json_serializable({"value": float('nan')})

    def test_inf_rejected(self):
        """Infinity values are rejected."""
        with pytest.raises(ValueError, match="Infinity"):
            validate_json_serializable(float('inf'))

        with pytest.raises(ValueError, match="Infinity"):
            validate_json_serializable({"value": float('-inf')})

    def test_numpy_array_rejected(self):
        """NumPy arrays are rejected."""
        with pytest.raises(ValueError, match="NumPy array"):
            validate_json_serializable(np.array([1, 2, 3]))

        with pytest.raises(ValueError, match="NumPy array"):
            validate_json_serializable(np.array([[1, 2], [3, 4]]))

    def test_numpy_scalar_rejected(self):
        """NumPy integer scalars are rejected (np.float64 passes since it's a float subclass)."""
        # np.int64 is NOT an int subclass, so it's rejected
        with pytest.raises(ValueError, match="NumPy integer"):
            validate_json_serializable(np.int64(42))

        # np.float64 IS a float subclass, so it passes validation
        # (this is acceptable since float subclasses are JSON-serializable)
        validate_json_serializable(np.float64(3.14))  # Should not raise

        # np.int32 also rejected
        with pytest.raises(ValueError, match="NumPy integer"):
            validate_json_serializable(np.int32(42))

    def test_nested_structure_valid(self):
        """Nested valid structures pass."""
        data = {
            "list": [1, 2, 3],
            "nested": {
                "a": True,
                "b": None,
                "c": "text",
            },
        }
        validate_json_serializable(data)  # Should not raise

    def test_nested_structure_invalid(self):
        """Nested invalid structures are caught."""
        data = {
            "valid": [1, 2, 3],
            "invalid": {"nested_nan": float('nan')},
        }
        with pytest.raises(ValueError, match="NaN"):
            validate_json_serializable(data)


class TestConvertForJSON:
    """Test JSON conversion utilities."""

    def test_primitives_unchanged(self):
        """Primitives are returned unchanged."""
        assert convert_for_json(None) is None
        assert convert_for_json(True) is True
        assert convert_for_json(42) == 42
        assert convert_for_json(3.14) == 3.14
        assert convert_for_json("hello") == "hello"

    def test_numpy_scalars_converted(self):
        """NumPy scalars are converted to Python natives."""
        assert convert_for_json(np.int64(42)) == 42
        assert isinstance(convert_for_json(np.int64(42)), int)

        assert abs(convert_for_json(np.float64(3.14)) - 3.14) < 1e-9
        assert isinstance(convert_for_json(np.float64(3.14)), float)

    def test_numpy_array_1d_converted(self):
        """1D NumPy arrays are converted to lists."""
        arr = np.array([1, 2, 3])
        result = convert_for_json(arr)
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_numpy_array_2d_converted(self):
        """2D NumPy arrays are converted to nested lists."""
        arr = np.array([[1, 2], [3, 4]])
        result = convert_for_json(arr)
        assert result == [[1, 2], [3, 4]]

    def test_numpy_array_3d_rejected(self):
        """3D+ NumPy arrays are rejected."""
        arr = np.zeros((2, 3, 4))
        with pytest.raises(ValueError, match="ndim=3"):
            convert_for_json(arr)

    def test_tuple_converted_to_list(self):
        """Tuples are converted to lists."""
        result = convert_for_json((1, 2, 3))
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_dict_keys_stringified(self):
        """Dictionary keys are stringified."""
        result = convert_for_json({1: "a", 2: "b"})
        assert result == {"1": "a", "2": "b"}

    def test_datetime_converted(self):
        """Datetime objects are converted to ISO format."""
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = convert_for_json(dt)
        assert result == "2024-01-15T10:30:00"

    def test_nan_rejected_in_conversion(self):
        """NaN cannot be converted."""
        with pytest.raises(ValueError, match="NaN"):
            convert_for_json(float('nan'))

    def test_inf_rejected_in_conversion(self):
        """Infinity cannot be converted."""
        with pytest.raises(ValueError, match="Infinity"):
            convert_for_json(float('inf'))


class TestAtomicJSONWriting:
    """Test atomic JSON writing."""

    def test_write_valid_json(self, tmp_path):
        """Valid JSON is written correctly."""
        data = {"key": "value", "number": 42}
        path = tmp_path / "test.json"

        write_atomic_json(data, path)

        assert path.exists()
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_atomic_on_error(self, tmp_path):
        """Partial writes don't leave corrupt files."""
        path = tmp_path / "test.json"

        # Try to write invalid data
        with pytest.raises(ValueError):
            write_atomic_json({"bad": float('nan')}, path)

        # File should not exist after error
        assert not path.exists()

    def test_creates_parent_directory(self, tmp_path):
        """Parent directories are created if needed."""
        path = tmp_path / "subdir" / "nested" / "test.json"

        write_atomic_json({"key": "value"}, path)

        assert path.exists()

    def test_trailing_newline(self, tmp_path):
        """Written JSON has trailing newline."""
        path = tmp_path / "test.json"

        write_atomic_json({"key": "value"}, path)

        with open(path, 'rb') as f:
            content = f.read()
        assert content.endswith(b'\n')


class TestFileHash:
    """Test file hash computation."""

    def test_compute_sha256(self, tmp_path):
        """SHA256 hash is computed correctly."""
        path = tmp_path / "test.txt"
        path.write_text("hello world")

        hash_val = compute_file_hash(path)

        assert len(hash_val) == 64  # SHA256 hex length
        assert hash_val.islower()
        assert all(c in '0123456789abcdef' for c in hash_val)

    def test_hash_deterministic(self, tmp_path):
        """Same file produces same hash."""
        path = tmp_path / "test.txt"
        path.write_text("hello world")

        hash1 = compute_file_hash(path)
        hash2 = compute_file_hash(path)

        assert hash1 == hash2

    def test_different_content_different_hash(self, tmp_path):
        """Different content produces different hash."""
        path1 = tmp_path / "test1.txt"
        path1.write_text("hello world")

        path2 = tmp_path / "test2.txt"
        path2.write_text("hello universe")

        assert compute_file_hash(path1) != compute_file_hash(path2)

    def test_missing_file_raises_error(self, tmp_path):
        """Missing file raises FileNotFoundError."""
        path = tmp_path / "missing.txt"

        with pytest.raises(FileNotFoundError):
            compute_file_hash(path)


class TestDataHash:
    """Test data hash computation."""

    def test_hash_deterministic(self):
        """Same data produces same hash."""
        data = {"key": "value", "number": 42}

        hash1 = compute_data_hash(data)
        hash2 = compute_data_hash(data)

        assert hash1 == hash2

    def test_sorted_keys_deterministic(self):
        """Dictionary key order doesn't affect hash."""
        data1 = {"a": 1, "b": 2}
        data2 = {"b": 2, "a": 1}

        # Should produce same hash due to sorted_keys
        assert compute_data_hash(data1) == compute_data_hash(data2)

    def test_different_data_different_hash(self):
        """Different data produces different hash."""
        data1 = {"key": "value1"}
        data2 = {"key": "value2"}

        assert compute_data_hash(data1) != compute_data_hash(data2)


class TestMyopicArtifactWriter:
    """Test artifact writer."""

    def test_write_artifact(self, tmp_path):
        """Artifact is written with provenance."""
        run_dir = tmp_path / "run"
        config = {"test": "config"}

        writer = MyopicArtifactWriter(
            run_dir=run_dir,
            config=config,
            git_commit="abc123",
            scenario_bank_id="test_bank",
            environment_version="v1",
        )

        path = writer.write(
            "test_artifact.json",
            {"data": "value"},
        )

        assert path.exists()
        assert "test_artifact.json" in writer.written_artifacts

        # Verify provenance added
        with open(path) as f:
            data = json.load(f)

        assert data["git_commit"] == "abc123"
        assert data["scenario_bank_id"] == "test_bank"
        assert data["environment_version"] == "v1"
        assert "written_at" in data
        assert "schema_version" in data

    def test_compute_manifest(self, tmp_path):
        """Manifest lists all artifacts with hashes."""
        run_dir = tmp_path / "run"
        config = {"test": "config"}

        writer = MyopicArtifactWriter(
            run_dir=run_dir,
            config=config,
            git_commit="abc123",
            scenario_bank_id="test_bank",
            environment_version="v1",
        )

        writer.write("artifact1.json", {"data": 1})
        writer.write("artifact2.json", {"data": 2})

        manifest = writer.compute_manifest()

        assert "artifact1.json" in manifest
        assert "artifact2.json" in manifest

        for name in ["artifact1.json", "artifact2.json"]:
            assert "relative_path" in manifest[name]
            assert "byte_size" in manifest[name]
            assert "sha256" in manifest[name]

    def test_get_run_metadata(self, tmp_path):
        """Run metadata includes provenance."""
        run_dir = tmp_path / "run"
        config = {"test": "config"}

        writer = MyopicArtifactWriter(
            run_dir=run_dir,
            config=config,
            git_commit="abc123",
            scenario_bank_id="test_bank",
            environment_version="v1",
        )

        metadata = writer.get_run_metadata()

        assert metadata["git_commit"] == "abc123"
        assert metadata["scenario_bank_id"] == "test_bank"
        assert metadata["environment_version"] == "v1"
        assert "run_dir" in metadata
        assert "written_artifacts" in metadata


class TestGetGitCommit:
    """Test git commit retrieval."""

    def test_get_commit(self):
        """Git commit is retrieved successfully."""
        commit = get_git_commit()

        assert isinstance(commit, str)
        assert len(commit) >= 7  # Short hash minimum