"""
Test Milestone 4 Exact Myopic action enumeration.

Tests:
- K=1 has exactly 6 actions (empty + 5 single slots)
- K=2 has exactly 16 actions (empty + 5 singles + 10 pairs)
- Action 0 is empty for both capacities
- All actions respect capacity constraint
- All slot indices are in valid range [0, 4]
- Actions are sorted lexicographically
"""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from envs.action_table import ACTION_TABLE_N5_K1, ACTION_TABLE_N5_K2


class TestActionCounts:
    """Test action table sizes."""

    def test_k1_action_count(self):
        """K=1 has exactly 6 actions."""
        assert len(ACTION_TABLE_N5_K1) == 6

    def test_k2_action_count(self):
        """K=2 has exactly 16 actions."""
        assert len(ACTION_TABLE_N5_K2) == 16


class TestActionStructure:
    """Test action table structure."""

    def test_action_0_empty_k1(self):
        """Action 0 is empty for K=1."""
        assert ACTION_TABLE_N5_K1[0] == ()

    def test_action_0_empty_k2(self):
        """Action 0 is empty for K=2."""
        assert ACTION_TABLE_N5_K2[0] == ()

    def test_all_actions_within_capacity_k1(self):
        """All K=1 actions have <= 1 slot."""
        for action in ACTION_TABLE_N5_K1:
            assert len(action) <= 1

    def test_all_actions_within_capacity_k2(self):
        """All K=2 actions have <= 2 slots."""
        for action in ACTION_TABLE_N5_K2:
            assert len(action) <= 2


class TestSlotIndices:
    """Test slot index validity."""

    def test_k1_slot_indices_in_bounds(self):
        """All K=1 slot indices are in [0, 4]."""
        for action in ACTION_TABLE_N5_K1:
            for slot in action:
                assert 0 <= slot < 5, f"Invalid slot {slot} in action {action}"

    def test_k2_slot_indices_in_bounds(self):
        """All K=2 slot indices are in [0, 4]."""
        for action in ACTION_TABLE_N5_K2:
            for slot in action:
                assert 0 <= slot < 5, f"Invalid slot {slot} in action {action}"


class TestActionCoverage:
    """Test action space coverage."""

    def test_k1_all_singles_present(self):
        """K=1 has all 5 single-slot actions."""
        singles = {a[0] for a in ACTION_TABLE_N5_K1 if len(a) == 1}
        assert singles == {0, 1, 2, 3, 4}

    def test_k2_all_singles_present(self):
        """K=2 has all 5 single-slot actions."""
        singles = {a[0] for a in ACTION_TABLE_N5_K2 if len(a) == 1}
        assert singles == {0, 1, 2, 3, 4}

    def test_k2_all_pairs_present(self):
        """K=2 has all 10 pair actions."""
        pairs = {tuple(sorted(a)) for a in ACTION_TABLE_N5_K2 if len(a) == 2}
        expected_pairs = {
            (0, 1), (0, 2), (0, 3), (0, 4),
            (1, 2), (1, 3), (1, 4),
            (2, 3), (2, 4),
            (3, 4),
        }
        assert pairs == expected_pairs


class TestActionOrdering:
    """Test action table ordering."""

    def test_k1_lexicographic_order(self):
        """K=1 actions are in lexicographic order."""
        actions = list(ACTION_TABLE_N5_K1)
        assert actions == sorted(actions)

    def test_k2_actions_grouped_by_size(self):
        """K=2 actions are grouped: empty, then singles, then pairs."""
        actions = list(ACTION_TABLE_N5_K2)

        # Action 0: empty
        assert actions[0] == ()

        # Actions 1-5: singles (0,), (1,), ..., (4,)
        for i in range(1, 6):
            assert len(actions[i]) == 1
            assert actions[i] == (i - 1,)

        # Actions 6-15: pairs (sorted within group)
        pairs = actions[6:]
        expected_pairs = [
            (0, 1), (0, 2), (0, 3), (0, 4),
            (1, 2), (1, 3), (1, 4),
            (2, 3), (2, 4),
            (3, 4),
        ]
        assert pairs == expected_pairs


class TestNoDuplicates:
    """Test action table has no duplicates."""

    def test_k1_no_duplicates(self):
        """K=1 has no duplicate actions."""
        actions = list(ACTION_TABLE_N5_K1)
        assert len(actions) == len(set(actions))

    def test_k2_no_duplicates(self):
        """K=2 has no duplicate actions."""
        actions = list(ACTION_TABLE_N5_K2)
        assert len(actions) == len(set(actions))


class TestCombinatorialCorrectness:
    """Test combinatorial correctness."""

    def test_k1_combinatorial_formula(self):
        """K=1 matches C(5,0) + C(5,1) = 1 + 5 = 6."""
        from math import comb
        expected = comb(5, 0) + comb(5, 1)
        assert len(ACTION_TABLE_N5_K1) == expected

    def test_k2_combinatorial_formula(self):
        """K=2 matches C(5,0) + C(5,1) + C(5,2) = 1 + 5 + 10 = 16."""
        from math import comb
        expected = comb(5, 0) + comb(5, 1) + comb(5, 2)
        assert len(ACTION_TABLE_N5_K2) == expected