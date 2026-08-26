"""
Action table generation for Milestone 2 Selective Maintenance Environment.

Implements deterministic direct-subset action enumeration for N=5 fleet slots
with maintenance capacity K=2 (main) or K=1 (sensitivity analysis).

Action mapping for N=5, K=2 (16 actions):
    0: ()          8: (0, 3)
    1: (0,)        9: (0, 4)
    2: (1,)       10: (1, 2)
    3: (2,)       11: (1, 3)
    4: (3,)       12: (1, 4)
    5: (4,)       13: (2, 3)
    6: (0, 1)     14: (2, 4)
    7: (0, 2)     15: (3, 4)

Ordering: empty set, then singletons in ascending order, then pairs in lexicographic order.
"""

from __future__ import annotations

from itertools import combinations
from typing import FrozenSet, Tuple


# Type alias for a frozen action subset
ActionSubset = Tuple[int, ...]


def build_action_table(n_fleet: int, k_capacity: int) -> Tuple[ActionSubset, ...]:
    """
    Build a deterministic action table for direct-subset maintenance actions.

    Args:
        n_fleet: Number of fleet slots (N). Must be positive.
        k_capacity: Maximum maintenance capacity per step (K). Must satisfy 0 <= K <= N.

    Returns:
        A tuple of action subsets, where each subset is a sorted tuple of slot indices.
        Action ID i maps to action_table[i].

    Ordering:
        1. Empty subset (ID 0)
        2. All one-slot subsets in ascending slot order
        3. All two-slot subsets in lexicographic order (if K >= 2)
        ... up to K slots

    Raises:
        ValueError: If n_fleet <= 0, k_capacity < 0, or k_capacity > n_fleet.
    """
    if n_fleet <= 0:
        raise ValueError(f"n_fleet must be positive, got {n_fleet}")
    if k_capacity < 0:
        raise ValueError(f"k_capacity must be non-negative, got {k_capacity}")
    if k_capacity > n_fleet:
        raise ValueError(f"k_capacity ({k_capacity}) cannot exceed n_fleet ({n_fleet})")

    subsets: list[ActionSubset] = []

    # Empty subset always first
    subsets.append(())

    # Add subsets of size 1 to K
    for size in range(1, k_capacity + 1):
        for combo in combinations(range(n_fleet), size):
            subsets.append(combo)

    return tuple(subsets)


def action_id_to_slots(action_id: int, action_table: Tuple[ActionSubset, ...]) -> ActionSubset:
    """
    Convert an action ID to its corresponding slot subset.

    Args:
        action_id: The action index (0 to len(action_table) - 1).
        action_table: The action table from build_action_table().

    Returns:
        A sorted tuple of slot indices for this action.

    Raises:
        ValueError: If action_id is out of range.
    """
    if action_id < 0 or action_id >= len(action_table):
        raise ValueError(
            f"action_id {action_id} out of range [0, {len(action_table) - 1}]"
        )
    return action_table[action_id]


def slots_to_action_id(
    slots: ActionSubset, action_table: Tuple[ActionSubset, ...]
) -> int:
    """
    Convert a slot subset to its corresponding action ID.

    Args:
        slots: A sorted tuple of slot indices (may be empty).
        action_table: The action table from build_action_table().

    Returns:
        The action ID for this subset.

    Raises:
        ValueError: If the slot subset is not found in the action table.
    """
    # Normalize: ensure slots are sorted
    normalized = tuple(sorted(slots))

    for action_id, table_slots in enumerate(action_table):
        if table_slots == normalized:
            return action_id

    raise ValueError(f"Slot subset {slots} not found in action table")


def validate_action_table(
    action_table: Tuple[ActionSubset, ...],
    expected_n: int,
    expected_k: int,
) -> bool:
    """
    Validate an action table against expected N and K parameters.

    Args:
        action_table: The action table to validate.
        expected_n: Expected fleet size N.
        expected_k: Expected maintenance capacity K.

    Returns:
        True if the action table is valid.

    Raises:
        ValueError: If validation fails with a descriptive message.
    """
    # Check expected count
    expected_count = 1 + sum(
        len(list(combinations(range(expected_n), k))) for k in range(1, expected_k + 1)
    )
    if len(action_table) != expected_count:
        raise ValueError(
            f"Action table has {len(action_table)} entries, "
            f"expected {expected_count} for N={expected_n}, K={expected_k}"
        )

    # Check action 0 is empty
    if action_table[0] != ():
        raise ValueError(f"Action 0 must be empty subset, got {action_table[0]}")

    # Check all subsets are valid
    seen: set[ActionSubset] = set()
    for action_id, subset in enumerate(action_table):
        # Check no duplicates
        if subset in seen:
            raise ValueError(f"Duplicate subset {subset} at action {action_id}")
        seen.add(subset)

        # Check subset size <= K
        if len(subset) > expected_k:
            raise ValueError(
                f"Action {action_id} has {len(subset)} slots, exceeds K={expected_k}"
            )

        # Check all slots in valid range
        for slot in subset:
            if not (0 <= slot < expected_n):
                raise ValueError(
                    f"Action {action_id} contains invalid slot {slot} "
                    f"(must be in [0, {expected_n - 1}])"
                )

        # Check subset is sorted
        if tuple(sorted(subset)) != subset:
            raise ValueError(f"Action {action_id} subset {subset} is not sorted")

    return True


def get_action_table_config(n_fleet: int = 5, k_capacity: int = 2) -> dict:
    """
    Get action table configuration and metadata.

    Args:
        n_fleet: Fleet size N (default 5).
        k_capacity: Maintenance capacity K (default 2).

    Returns:
        Dictionary with action table metadata and the table itself.
    """
    action_table = build_action_table(n_fleet, k_capacity)

    return {
        "n_fleet": n_fleet,
        "k_capacity": k_capacity,
        "action_count": len(action_table),
        "action_table": action_table,
    }


# Pre-built action tables for common configurations
ACTION_TABLE_N5_K2: Tuple[ActionSubset, ...] = build_action_table(5, 2)
ACTION_TABLE_N5_K1: Tuple[ActionSubset, ...] = build_action_table(5, 1)