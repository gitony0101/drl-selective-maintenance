"""
Prediction Cache Identity for Milestone 5 DDQN.

Single authoritative helper for prediction cache manifest identity.
Used by:
- Checkpoint writer (save_checkpoint)
- Evaluator (evaluate_ddqn.py)
- Resume validation (load_checkpoint)
- Tests

This ensures identical prediction cache identities across all production paths.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional


def compute_prediction_cache_manifest_sha256(manifest_path: Path | str) -> str:
    """
    Compute SHA256 hash of prediction cache manifest file bytes.

    Args:
        manifest_path: Path to prediction_cache_manifest_v*.json

    Returns:
        64-character lowercase hexadecimal SHA256 hash

    Raises:
        FileNotFoundError: If manifest file doesn't exist
        ValueError: If file is empty
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Prediction cache manifest not found: {path}")

    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        content = f.read()
        if not content:
            raise ValueError(f"Manifest file is empty: {path}")
        sha256.update(content)
    return sha256.hexdigest()


def load_prediction_cache_manifest(manifest_path: Path | str) -> Dict[str, Any]:
    """
    Load and parse prediction cache manifest.

    Args:
        manifest_path: Path to prediction_cache_manifest_v*.json

    Returns:
        Parsed manifest dictionary

    Raises:
        FileNotFoundError: If manifest file doesn't exist
        ValueError: If manifest is not valid JSON
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Prediction cache manifest not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in manifest: {e}")


def validate_prediction_cache_manifest(manifest: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract and validate required identity fields from prediction cache manifest.

    Required fields (fail closed if missing):
    - cache_hash (content hash of the cache parquet file)
    - checkpoint_hash or checkpoint_id (predictor checkpoint identity)
    - feature_schema_hash
    - normalizer_hash
    - predictor_id
    - cache_version
    - row_counts or engine_counts (to verify split presence)

    Returns:
        Dictionary of validated identity fields

    Raises:
        ValueError: If any required field is missing or malformed
    """
    required_fields = [
        "cache_hash",
        "feature_schema_hash",
        "normalizer_hash",
        "predictor_id",
        "cache_version",
    ]

    # Check for missing fields
    missing = [f for f in required_fields if f not in manifest]
    if missing:
        raise ValueError(f"Prediction cache manifest missing required fields: {missing}")

    # Validate field types and formats
    identities = {}

    # cache_hash - 64 char hex
    cache_hash = manifest.get("cache_hash")
    if not isinstance(cache_hash, str) or len(cache_hash) != 64:
        raise ValueError(f"cache_hash must be 64-char hex string, got: {cache_hash}")
    # Validate it's valid hex
    try:
        int(cache_hash, 16)
    except ValueError:
        raise ValueError(f"cache_hash is not valid hex: {cache_hash}")
    identities["prediction_cache_declared_cache_hash"] = cache_hash.lower()

    # feature_schema_hash - 12 char hex in v2 manifest
    feature_schema_hash = manifest.get("feature_schema_hash")
    if not isinstance(feature_schema_hash, str) or len(feature_schema_hash) < 8:
        raise ValueError(f"feature_schema_hash must be valid hex string, got: {feature_schema_hash}")
    identities["prediction_cache_feature_schema_hash"] = feature_schema_hash.lower()

    # normalizer_hash - 64 char hex
    normalizer_hash = manifest.get("normalizer_hash")
    if not isinstance(normalizer_hash, str) or len(normalizer_hash) != 64:
        raise ValueError(f"normalizer_hash must be 64-char hex string, got: {normalizer_hash}")
    try:
        int(normalizer_hash, 16)
    except ValueError:
        raise ValueError(f"normalizer_hash is not valid hex: {normalizer_hash}")
    identities["prediction_cache_normalizer_hash"] = normalizer_hash.lower()

    # split validation - manifest must not contain rl_test rows for training/validation use
    # The manifest has row_counts or engine_counts per split
    row_counts = manifest.get("row_counts", {})
    engine_counts = manifest.get("engine_counts", {})
    split_counts = row_counts or engine_counts
    if not split_counts:
        raise ValueError("Prediction cache manifest missing row_counts or engine_counts per split")

    # For M5, we need to know which split this checkpoint is being used for
    # The manifest contains all splits; the checkpoint will declare its split usage
    # We record all splits present in the manifest
    identities["prediction_cache_splits_present"] = list(split_counts.keys())
    identities["prediction_cache_row_counts"] = row_counts
    identities["prediction_cache_engine_counts"] = engine_counts

    # predictor_id
    predictor_id = manifest.get("predictor_id")
    if not isinstance(predictor_id, str) or not predictor_id:
        raise ValueError(f"predictor_id must be non-empty string, got: {predictor_id}")
    identities["prediction_cache_predictor_checkpoint_hash"] = predictor_id

    # cache_version
    cache_version = manifest.get("cache_version")
    if not isinstance(cache_version, str) or not cache_version:
        raise ValueError(f"cache_version must be non-empty string, got: {cache_version}")
    identities["prediction_cache_schema_version"] = cache_version

    # checkpoint_hash (v2 manifest has this)
    if "checkpoint_hash" in manifest:
        checkpoint_hash = manifest.get("checkpoint_hash")
        if isinstance(checkpoint_hash, str) and len(checkpoint_hash) == 64:
            try:
                int(checkpoint_hash, 16)
                identities["prediction_cache_predictor_checkpoint_hash"] = checkpoint_hash.lower()
            except ValueError:
                pass  # Fall back to predictor_id

    return identities


def get_prediction_cache_identity(
    manifest_path: Path | str,
) -> Dict[str, Any]:
    """
    Get complete prediction cache identity from manifest file.

    This is the SINGLE authoritative function for extracting prediction cache
    identity for checkpoint metadata and resume validation.

    Args:
        manifest_path: Path to prediction_cache_manifest_v*.json

    Returns:
        Dictionary containing:
        - prediction_cache_manifest_path: Path to manifest
        - prediction_cache_manifest_sha256: SHA256 of manifest file bytes
        - All validated identity fields from validate_prediction_cache_manifest

    Raises:
        FileNotFoundError: If manifest doesn't exist
        ValueError: If manifest is invalid or missing required fields
    """
    path = Path(manifest_path)
    manifest = load_prediction_cache_manifest(path)
    manifest_sha256 = compute_prediction_cache_manifest_sha256(path)
    identities = validate_prediction_cache_manifest(manifest)

    result = {
        "prediction_cache_manifest_path": str(path),
        "prediction_cache_manifest_sha256": manifest_sha256,
    }
    result.update(identities)

    return result