"""
Provenance utilities for Milestone 4 Exact Myopic Optimizer.

Implements action-table content hashing in the M4-owned namespace.
The hashing function accepts ACTION_TABLE_N5_K1 or ACTION_TABLE_N5_K2 as input,
but lives in the M4 namespace to maintain milestone isolation.
"""

from __future__ import annotations

import json
from typing import Tuple

# Type alias for action subset compatibility
ActionSubset = Tuple[int, ...]


def compute_action_table_content_hash(
    action_table: Tuple[ActionSubset, ...],
    algorithm: str = "sha256",
) -> str:
    """
    Compute SHA256 hash of action table content.

    The hash is computed from the ordered action table contents,
    not just the identity string or action count. Changing any
    action subset while preserving the action count will change
    this hash.

    Args:
        action_table: Action table tuple from build_action_table().
        algorithm: Hash algorithm (default sha256).

    Returns:
        Hex-encoded hash string.
    """
    import hashlib

    # Serialize action table to deterministic JSON string
    # Each action is a tuple of slot indices, sorted within each tuple
    # The outer list is in action-ID order (0, 1, 2, ...)
    normalized = [list(action) for action in action_table]
    json_str = json.dumps(normalized, sort_keys=True, separators=(",", ":"))

    hasher = hashlib.new(algorithm)
    hasher.update(json_str.encode("utf-8"))
    return hasher.hexdigest()


__all__ = [
    "compute_action_table_content_hash",
]