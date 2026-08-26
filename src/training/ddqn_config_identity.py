"""Authoritative resolved-config identity helper (M5 M5 training path)."""
from __future__ import annotations

import json
import hashlib
from typing import Dict, Any


# Fields that are run-specific and should NOT be part of the semantic config identity.
# These fields can vary between training runs without changing the semantic configuration.
_RUN_SPECIFIC_FIELDS = frozenset({
    "output_dir",
    "run_id",
    "device",
    "prediction_cache_path",  # Base path, not the manifest; manifest_path is the identity
})


def _strip_run_specific_fields(config: Dict[str, Any]) -> Dict[str, Any]:
    """Remove run-specific fields from config to produce semantic config for identity."""
    return {k: v for k, v in config.items() if k not in _RUN_SPECIFIC_FIELDS}


def compute_resolved_config_identity(config: Dict[str, Any]) -> str:
    """Canonical deterministic SHA256 identity for a resolved config object.

    Contract (authoritative for ALL production artifacts):
    - UTF-8 encoding.
    - sort_keys=True.
    - separators=(",", ":").
    - ensure_ascii=False.
    - Deterministic supported value types (str, int, float, bool, None,
      list, dict; no dates, no sets, no tuples, no bytes, no torch tensors).
    - No timestamps included in serialized object.
    - No post-run metrics included in serialized object.
    - No dynamically discovered current Git HEAD.
    - No mutable runtime-only exclusions.

    Treatment of run_id / output_dir:
    - These are stripped before hashing as they are run-specific.
    - The identity is computed over the SEMANTIC configuration only.

    Returns lowercase 64-character hexadecimal SHA256 string.
    Raises ValueError if canonical serialization is non-deterministic.
    """
    # Strip run-specific fields to compute semantic identity
    semantic_config = _strip_run_specific_fields(config)
    try:
        canonical_bytes = json.dumps(
            semantic_config,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    except TypeError as exc:
        raise ValueError(
            f"Resolved config contains non-serializable types: {exc}"
        ) from exc
    return hashlib.sha256(canonical_bytes).hexdigest()


def validate_resolved_config_identity(value: str) -> None:
    """Validate that a string is a valid 64-char lowercase hex SHA256 identity."""
    if not isinstance(value, str):
        raise ValueError(f"Identity is not a string: {type(value)}")
    if len(value) != 64:
        raise ValueError(f"Identity length {len(value)} != 64: got '{value}'")
    try:
        int(value, 16)
    except ValueError:
        raise ValueError(f"Identity is not lowercase hex: '{value}'")
    # Enforce lowercase explicitly
    if value != value.lower():
        raise ValueError(f"Identity contains uppercase characters: '{value}'")
    # Reject all-zeros placeholder
    if value == "0" * 64:
        raise ValueError("Placeholder all-zero identity is not permitted.")