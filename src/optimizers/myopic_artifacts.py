"""
Artifact utilities for Milestone 4 Exact Myopic Optimizer.

Implements:
- Atomic JSON writing (write temp, rename)
- JSON validation (reject NaN, Inf, tensors)
- File hash computation (SHA256)
- Artifact writer with provenance tracking
"""

from __future__ import annotations

import json
import hashlib
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union

import numpy as np


def validate_json_serializable(obj: Any, path: str = "root") -> None:
    """
    Recursively validate that an object is JSON-serializable with strict rules.

    Rejects:
    - NaN, Infinity, negative Infinity
    - NumPy arrays and tensors
    - NumPy scalar types (convert to Python native)

    Args:
        obj: Object to validate.
        path: Current path in object structure (for error messages).

    Raises:
        ValueError: If object contains invalid values.
    """
    if obj is None:
        return

    if isinstance(obj, bool):
        return

    if isinstance(obj, (int, float, str)):
        # Check for float-specific issues
        if isinstance(obj, float):
            if np.isnan(obj):
                raise ValueError(f"{path}: NaN is not JSON-serializable")
            if np.isinf(obj):
                raise ValueError(f"{path}: Infinity is not JSON-serializable")
        return

    # Convert numpy scalars to Python native
    if isinstance(obj, (np.integer, np.floating)):
        # This is for detection - we want to reject these and require conversion
        if isinstance(obj, np.integer):
            raise ValueError(
                f"{path}: NumPy integer {obj} should be converted to Python int"
            )
        if isinstance(obj, np.floating):
            raise ValueError(
                f"{path}: NumPy float {obj} should be converted to Python float"
            )

    if isinstance(obj, np.ndarray):
        raise ValueError(
            f"{path}: NumPy array of shape {obj.shape} is not directly JSON-serializable"
        )

    if isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            validate_json_serializable(item, f"{path}[{i}]")
        return

    if isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"{path}: Dictionary key {key!r} is not a string"
                )
            validate_json_serializable(value, f"{path}.{key}")
        return

    raise ValueError(
        f"{path}: Type {type(obj).__name__} is not JSON-serializable"
    )


def convert_for_json(obj: Any) -> Any:
    """
    Convert an object to JSON-serializable form.

    Handles:
    - NumPy scalars -> Python native
    - NumPy arrays -> list (if 1D or 2D)
    - Tuples -> list
    - datetime -> ISO format string

    Args:
        obj: Object to convert.

    Returns:
        JSON-serializable object.

    Raises:
        ValueError: If object cannot be converted.
    """
    if obj is None:
        return None

    if isinstance(obj, bool):
        return obj

    if isinstance(obj, (int, str)):
        return obj

    if isinstance(obj, float):
        if np.isnan(obj):
            raise ValueError("NaN cannot be converted to JSON")
        if np.isinf(obj):
            raise ValueError("Infinity cannot be converted to JSON")
        return obj

    # NumPy scalars
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        if np.isnan(f) or np.isinf(f):
            raise ValueError(f"NumPy float {f} is not JSON-serializable")
        return f
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return convert_for_json(obj.item())
        if obj.ndim == 1:
            return [convert_for_json(x) for x in obj]
        if obj.ndim == 2:
            return [[convert_for_json(x) for x in row] for row in obj]
        raise ValueError(f"NumPy array with ndim={obj.ndim} cannot be converted")

    if isinstance(obj, (list, tuple)):
        return [convert_for_json(item) for item in obj]

    if isinstance(obj, dict):
        return {str(k): convert_for_json(v) for k, v in obj.items()}

    if isinstance(obj, datetime):
        return obj.isoformat()

    # For other types, try string conversion
    return str(obj)


def write_atomic_json(
    data: Dict[str, Any],
    path: Path,
    indent: int = 2,
) -> None:
    """
    Write JSON atomically (write to temp, rename).

    Args:
        data: Data to write.
        path: Output path.
        indent: JSON indentation level.

    Raises:
        ValueError: If data contains invalid values.
        IOError: If write fails.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Validate data before writing
    validate_json_serializable(data)

    # Convert to JSON-serializable form
    converted_data = convert_for_json(data)

    # Write to temp file first
    fd, temp_path = tempfile.mkstemp(
        suffix=".json.tmp",
        dir=path.parent,
    )
    try:
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(converted_data, f, indent=indent, ensure_ascii=False)
            f.write("\n")  # Trailing newline
        # Atomic rename
        Path(temp_path).rename(path)
    except Exception:
        # Clean up temp file on error
        try:
            Path(temp_path).unlink()
        except OSError:
            pass
        raise


def compute_file_hash(path: Path, algorithm: str = "sha256") -> str:
    """
    Compute file hash.

    Args:
        path: File path.
        algorithm: Hash algorithm (default sha256).

    Returns:
        Hex-encoded hash string.

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    hasher = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_data_hash(data: Dict[str, Any], algorithm: str = "sha256") -> str:
    """
    Compute hash of JSON-serializable data.

    Args:
        data: Data to hash (will be serialized with sorted keys).
        algorithm: Hash algorithm (default sha256).

    Returns:
        Hex-encoded hash string.
    """
    # Convert to JSON with sorted keys for determinism
    json_str = json.dumps(
        convert_for_json(data),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    hasher = hashlib.new(algorithm)
    hasher.update(json_str.encode("utf-8"))
    return hasher.hexdigest()


class MyopicArtifactWriter:
    """
    Artifact writer for Milestone 4 runs.

    Tracks:
    - Run directory
    - Written artifacts
    - Git provenance
    - Config hash
    """

    def __init__(
        self,
        run_dir: Path,
        config: Dict[str, Any],
        git_commit: str,
        scenario_bank_id: str,
        environment_version: str,
    ) -> None:
        """
        Initialize artifact writer.

        Args:
            run_dir: Output directory for artifacts.
            config: Run configuration.
            git_commit: Git commit hash.
            scenario_bank_id: Scenario bank identifier.
            environment_version: Environment version string.
        """
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.config = config
        self.git_commit = git_commit
        self.scenario_bank_id = scenario_bank_id
        self.environment_version = environment_version
        self.written_artifacts: list[str] = []

        # Compute config hash
        self.config_hash = compute_data_hash(config)

    def write(
        self,
        name: str,
        data: Dict[str, Any],
        add_schema: bool = True,
    ) -> Path:
        """
        Write an artifact.

        Args:
            name: Artifact name (e.g., "resolved_config.json").
            data: Data to write.
            add_schema: Whether to add schema_version field.

        Returns:
            Path to written file.
        """
        # Add provenance fields
        if add_schema and "schema_version" not in data:
            data["schema_version"] = "m4_v1"

        if "git_commit" not in data:
            data["git_commit"] = self.git_commit

        if "config_hash" not in data:
            data["config_hash"] = self.config_hash

        if "scenario_bank_id" not in data:
            data["scenario_bank_id"] = self.scenario_bank_id

        if "environment_version" not in data:
            data["environment_version"] = self.environment_version

        if "written_at" not in data:
            data["written_at"] = datetime.now(timezone.utc).isoformat()

        # Write file
        path = self.run_dir / name
        write_atomic_json(data, path)
        self.written_artifacts.append(name)
        return path

    def compute_manifest(self) -> Dict[str, Any]:
        """
        Compute artifact manifest.

        Returns:
            Manifest with paths, sizes, and hashes.
        """
        manifest = {}
        for name in sorted(self.written_artifacts):
            path = self.run_dir / name
            manifest[name] = {
                "relative_path": name,
                "byte_size": path.stat().st_size,
                "sha256": compute_file_hash(path),
            }
        return manifest

    def get_run_metadata(self) -> Dict[str, Any]:
        """
        Get run metadata.

        Returns:
            Metadata dict with provenance info.
        """
        return {
            "git_commit": self.git_commit,
            "config_hash": self.config_hash,
            "scenario_bank_id": self.scenario_bank_id,
            "environment_version": self.environment_version,
            "run_dir": str(self.run_dir),
            "written_artifacts": sorted(self.written_artifacts),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }


def get_git_commit() -> str:
    """
    Get current Git commit hash.

    Returns:
        Commit hash string.

    Raises:
        RuntimeError: If git is not available or not in a repo.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to get git commit: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("git not found in PATH")


# =============================================================================
# STEP 6 FIX: Centralized complete scientific config and hash functions
# =============================================================================

def build_complete_scientific_config(
    schema_version: str,
    policy_id: str,
    matrix_role: str,
    risk_model_id: str,
    risk_temperature: Optional[float],
    tie_tolerance: float,
    environment_version: str,
    delta_cycles: int,
    rul_scale: float,
    age_scale_cycles: int,
    fleet_size: int,
    episode_horizon: int,
    active_k_values: list,
    active_cost_regimes: list,
    active_splits: list,
    action_table_K1_identity: str,
    action_table_K1_num_actions: int,
    action_table_K2_identity: str,
    action_table_K2_num_actions: int,
    action_table_K1_content_hash: Optional[str],
    action_table_K2_content_hash: Optional[str],
    prediction_cache_path: str,
    prediction_cache_sha256: Optional[str],
    scenario_bank_ids: list,
    scenario_bank_sha256_values: dict,
    scenario_generation_version: str,
    scenario_seeds: list,
    scenario_selection_basis: str,
    episode_count_per_config: int,
    information_mode: str,
    validation_split: str = "rl_validation",
    forbidden_splits: Optional[list] = None,
    supported_capacities: Optional[list] = None,
    supported_cost_regimes: Optional[list] = None,
    supported_splits: Optional[list] = None,
    engineering_coverage_threshold_cycles: Optional[float] = None,
) -> dict:
    """
    STEP 6 FIX: Build the complete resolved scientific configuration.

    This function centralizes config building to ensure consistency across
    all production artifacts. The config includes all behavior-affecting fields
    and excludes nondeterministic fields (timestamp, output_dir, git_commit,
    config_hash itself).

    Args:
        schema_version: Schema version string (e.g., "m4_v1")
        policy_id: Policy identifier (e.g., "exact_myopic_v1")
        matrix_role: "primary_contract_policy" or "engineering_behavior_coverage"
        risk_model_id: Risk model identifier
        risk_temperature: Temperature for logistic model (None for hard_window)
        tie_tolerance: Tie-breaking tolerance
        environment_version: Environment version (e.g., "m2_v1")
        delta_cycles: Decision window in cycles
        rul_scale: RUL normalization scale
        age_scale_cycles: Age normalization scale
        fleet_size: Number of slots/units
        episode_horizon: Steps per episode
        active_k_values: K capacities covered in this run
        active_cost_regimes: Cost regimes covered in this run
        active_splits: Data splits covered in this run
        action_table_K1_identity: K=1 action table identity string
        action_table_K1_num_actions: Number of K=1 actions
        action_table_K2_identity: K=2 action table identity string
        action_table_K2_num_actions: Number of K=2 actions
        action_table_K1_content_hash: SHA256 of K=1 action table content
        action_table_K2_content_hash: SHA256 of K=2 action table content
        prediction_cache_path: Path to prediction cache (repo-relative)
        prediction_cache_sha256: SHA256 of prediction cache parquet file
        scenario_bank_ids: List of scenario bank IDs used
        scenario_bank_sha256_values: Dict mapping scenario bank ID to SHA256
        scenario_generation_version: Scenario generator version
        scenario_seeds: List of scenario seeds used in generation
        scenario_selection_basis: Selection basis string (e.g., "predicted_rul_and_cache_row_continuity")
        episode_count_per_config: Episodes per configuration
        information_mode: Info mode (e.g., "normal")
        validation_split: Split reserved for validation
        forbidden_splits: Splits explicitly forbidden (e.g., ["rl_test"])
        supported_capacities: All supported K capacities (default: [1, 2])
        supported_cost_regimes: All supported cost regimes
        supported_splits: All supported splits

    Returns:
        Complete scientific configuration dict (excludes metadata fields)
    """
    if supported_capacities is None:
        supported_capacities = [1, 2]
    if supported_cost_regimes is None:
        from envs.costs import COST_REGIMES
        supported_cost_regimes = sorted(list(COST_REGIMES.keys()))
    if supported_splits is None:
        supported_splits = ["predictor_train", "rl_validation"]
    if forbidden_splits is None:
        forbidden_splits = ["rl_test", "predictor_validation"]

    return {
        # Core identity
        "schema_version": schema_version,
        "policy_id": policy_id,
        "matrix_role": matrix_role,
        "environment_version": environment_version,

        # Risk model parameters
        "risk_model_id": risk_model_id,
        "risk_temperature": risk_temperature,
        "tie_tolerance": tie_tolerance,

        # Environment parameters
        "delta_cycles": delta_cycles,
        "rul_scale": rul_scale,
        "age_scale_cycles": age_scale_cycles,
        "fleet_size": fleet_size,
        "episode_horizon": episode_horizon,

        # Supported/active coverage
        "supported_capacities": supported_capacities,
        "active_k_values": sorted(active_k_values),
        "supported_cost_regimes": sorted(supported_cost_regimes),
        "active_cost_regimes": sorted(active_cost_regimes),
        "supported_splits": sorted(supported_splits),
        "active_splits": sorted(active_splits),
        "validation_split": validation_split,
        "forbidden_splits": sorted(forbidden_splits),

        # Action table identity (frozen M2 tables)
        "action_table_K1_identity": action_table_K1_identity,
        "action_table_K1_num_actions": action_table_K1_num_actions,
        "action_table_K2_identity": action_table_K2_identity,
        "action_table_K2_num_actions": action_table_K2_num_actions,
        "action_table_K1_content_hash": action_table_K1_content_hash,
        "action_table_K2_content_hash": action_table_K2_content_hash,

        # Prediction cache identity
        "prediction_cache_path": prediction_cache_path,
        "prediction_cache_sha256": prediction_cache_sha256,

        # Scenario bank identities
        "scenario_bank_ids": sorted(scenario_bank_ids),
        "scenario_bank_sha256_values": scenario_bank_sha256_values,

        # Scenario generation
        "scenario_generation_version": scenario_generation_version,
        "scenario_seeds": sorted(scenario_seeds),
        "scenario_selection_basis": scenario_selection_basis,

        # Episode configuration
        "episode_count_per_config": episode_count_per_config,
        "information_mode": information_mode,
        "engineering_coverage_threshold_cycles": engineering_coverage_threshold_cycles if engineering_coverage_threshold_cycles is not None else 6.0,
    }


def build_runtime_metadata(
    output_dir: str,
    overwrite: bool,
    timestamp: str,
    git_commit: str,
    command_line: str,
    log_path: str,
    temporary_path: str,
) -> dict:
    """
    Build runtime metadata that does NOT affect config_hash.

    Runtime metadata includes:
    - output directory
    - overwrite flag
    - timestamp
    - Git commit
    - command line
    - log path
    - temporary path

    These fields are for provenance but do not affect the scientific configuration.

    Args:
        output_dir: Output directory path
        overwrite: Overwrite flag
        timestamp: ISO timestamp
        git_commit: Git commit hash
        command_line: Full command line string
        log_path: Log file path
        temporary_path: Temporary directory path

    Returns:
        Runtime metadata dict (excluded from config_hash)
    """
    return {
        "output_dir": output_dir,
        "overwrite": overwrite,
        "timestamp": timestamp,
        "git_commit": git_commit,
        "command_line": command_line,
        "log_path": log_path,
        "temporary_path": temporary_path,
    }


def compute_complete_config_hash(
    complete_scientific_config: dict,
    algorithm: str = "sha256",
) -> str:
    """
    STEP 6 FIX: Compute hash of the complete scientific configuration.

    Uses strict deterministic JSON serialization:
    - sorted keys
    - compact separators (",", ":")
    - UTF-8 encoding

    Args:
        complete_scientific_config: Config dict from build_complete_scientific_config()
        algorithm: Hash algorithm (default sha256)

    Returns:
        Hex-encoded hash string.

    Raises:
        ValueError: If config contains non-JSON-serializable values.
    """
    return compute_data_hash(complete_scientific_config, algorithm=algorithm)


__all__ = [
    "write_atomic_json",
    "validate_json_serializable",
    "convert_for_json",
    "compute_file_hash",
    "compute_data_hash",
    "MyopicArtifactWriter",
    "get_git_commit",
    # STEP 6 FIX: Centralized config functions
    "build_complete_scientific_config",
    "compute_complete_config_hash",
    # Runtime metadata (not hashed)
    "build_runtime_metadata",
]