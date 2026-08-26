"""
Unit tests for Milestone 2 action table generation.

Tests cover:
- Action counts for N=5, K=2 and K=1
- Action 0 is empty subset
- Exact frozen mapping
- Round-trip conversion (action_id -> slots -> action_id)
- No duplicates
- No action exceeds K
- Invalid action IDs fail clearly
- Invalid slot subsets fail clearly
"""

import pytest

from src.envs.action_table import (
    ACTION_TABLE_N5_K1,
    ACTION_TABLE_N5_K2,
    action_id_to_slots,
    build_action_table,
    slots_to_action_id,
    validate_action_table,
)


class TestActionTableCounts:
    """Test action table sizes."""

    def test_n5_k2_gives_16_actions(self) -> None:
        """N=5, K=2 should produce exactly 16 actions."""
        table = build_action_table(5, 2)
        assert len(table) == 16

    def test_n5_k1_gives_6_actions(self) -> None:
        """N=5, K=1 should produce exactly 6 actions."""
        table = build_action_table(5, 1)
        assert len(table) == 6

    def test_prebuilt_n5_k2_has_16_actions(self) -> None:
        """Pre-built ACTION_TABLE_N5_K2 should have 16 actions."""
        assert len(ACTION_TABLE_N5_K2) == 16

    def test_prebuilt_n5_k1_has_6_actions(self) -> None:
        """Pre-built ACTION_TABLE_N5_K1 should have 6 actions."""
        assert len(ACTION_TABLE_N5_K1) == 6


class TestActionZeroIsEmpty:
    """Test that action 0 is the empty subset."""

    def test_n5_k2_action_0_is_empty(self) -> None:
        """Action 0 for N=5, K=2 should be empty tuple."""
        assert ACTION_TABLE_N5_K2[0] == ()

    def test_n5_k1_action_0_is_empty(self) -> None:
        """Action 0 for N=5, K=1 should be empty tuple."""
        assert ACTION_TABLE_N5_K1[0] == ()

    def test_generic_action_0_is_empty(self) -> None:
        """Action 0 should be empty for any valid N, K."""
        for n in [3, 5, 10]:
            for k in [1, 2, min(n, 3)]:
                table = build_action_table(n, k)
                assert table[0] == ()


class TestFrozenMapping:
    """Test exact frozen mapping for N=5, K=2."""

    def test_full_mapping_n5_k2(self) -> None:
        """Test complete action ID to slots mapping for N=5, K=2."""
        expected = {
            0: (),
            1: (0,),
            2: (1,),
            3: (2,),
            4: (3,),
            5: (4,),
            6: (0, 1),
            7: (0, 2),
            8: (0, 3),
            9: (0, 4),
            10: (1, 2),
            11: (1, 3),
            12: (1, 4),
            13: (2, 3),
            14: (2, 4),
            15: (3, 4),
        }
        for action_id, expected_slots in expected.items():
            actual_slots = ACTION_TABLE_N5_K2[action_id]
            assert actual_slots == expected_slots, f"Mismatch at action {action_id}"

    def test_full_mapping_n5_k1(self) -> None:
        """Test complete action ID to slots mapping for N=5, K=1."""
        expected = {
            0: (),
            1: (0,),
            2: (1,),
            3: (2,),
            4: (3,),
            5: (4,),
        }
        for action_id, expected_slots in expected.items():
            actual_slots = ACTION_TABLE_N5_K1[action_id]
            assert actual_slots == expected_slots, f"Mismatch at action {action_id}"


class TestRoundTrip:
    """Test round-trip conversion between action_id and slots."""

    def test_action_id_to_slots_to_action_id_n5_k2(self) -> None:
        """Round-trip should be identity for N=5, K=2."""
        for action_id in range(len(ACTION_TABLE_N5_K2)):
            slots = action_id_to_slots(action_id, ACTION_TABLE_N5_K2)
            recovered_id = slots_to_action_id(slots, ACTION_TABLE_N5_K2)
            assert recovered_id == action_id

    def test_slots_to_action_id_to_slots_n5_k2(self) -> None:
        """Round-trip should be identity for N=5, K=2 (slots first)."""
        for slots in ACTION_TABLE_N5_K2:
            action_id = slots_to_action_id(slots, ACTION_TABLE_N5_K2)
            recovered_slots = action_id_to_slots(action_id, ACTION_TABLE_N5_K2)
            assert recovered_slots == slots

    def test_action_id_to_slots_to_action_id_n5_k1(self) -> None:
        """Round-trip should be identity for N=5, K=1."""
        for action_id in range(len(ACTION_TABLE_N5_K1)):
            slots = action_id_to_slots(action_id, ACTION_TABLE_N5_K1)
            recovered_id = slots_to_action_id(slots, ACTION_TABLE_N5_K1)
            assert recovered_id == action_id


class TestNoDuplicates:
    """Test that action tables have no duplicate subsets."""

    def test_no_duplicates_n5_k2(self) -> None:
        """N=5, K=2 action table should have no duplicates."""
        seen = set()
        for subset in ACTION_TABLE_N5_K2:
            assert subset not in seen, f"Duplicate subset: {subset}"
            seen.add(subset)

    def test_no_duplicates_n5_k1(self) -> None:
        """N=5, K=1 action table should have no duplicates."""
        seen = set()
        for subset in ACTION_TABLE_N5_K1:
            assert subset not in seen, f"Duplicate subset: {subset}"
            seen.add(subset)


class TestNoActionExceedsK:
    """Test that no action subset exceeds K slots."""

    def test_all_actions_have_at_most_k_slots_n5_k2(self) -> None:
        """All actions for N=5, K=2 should have at most 2 slots."""
        for subset in ACTION_TABLE_N5_K2:
            assert len(subset) <= 2, f"Action {subset} exceeds K=2"

    def test_all_actions_have_at_most_k_slots_n5_k1(self) -> None:
        """All actions for N=5, K=1 should have at most 1 slot."""
        for subset in ACTION_TABLE_N5_K1:
            assert len(subset) <= 1, f"Action {subset} exceeds K=1"


class TestInvalidActionIds:
    """Test that invalid action IDs raise clear errors."""

    def test_negative_action_id_raises(self) -> None:
        """Negative action ID should raise ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            action_id_to_slots(-1, ACTION_TABLE_N5_K2)

    def test_action_id_too_large_raises(self) -> None:
        """Action ID >= len(table) should raise ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            action_id_to_slots(16, ACTION_TABLE_N5_K2)  # 16 is out of range [0, 15]

    def test_large_negative_action_id_raises(self) -> None:
        """Large negative action ID should raise ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            action_id_to_slots(-100, ACTION_TABLE_N5_K2)


class TestInvalidSlotSubsets:
    """Test that invalid slot subsets fail clearly."""

    def test_slot_out_of_range_raises(self) -> None:
        """Slot >= N should raise ValueError."""
        with pytest.raises(ValueError, match="not found in action table"):
            slots_to_action_id((5,), ACTION_TABLE_N5_K2)  # Slot 5 is invalid for N=5

    def test_negative_slot_raises(self) -> None:
        """Negative slot should raise ValueError."""
        with pytest.raises(ValueError, match="not found in action table"):
            slots_to_action_id((-1,), ACTION_TABLE_N5_K2)

    def test_subset_exceeds_k_raises(self) -> None:
        """Subset with more than K slots should raise ValueError."""
        with pytest.raises(ValueError, match="not found in action table"):
            slots_to_action_id((0, 1, 2), ACTION_TABLE_N5_K1)  # 3 slots for K=1

    def test_unsorted_slots_still_works(self) -> None:
        """Unsorted slots should be normalized and still work."""
        # slots_to_action_id normalizes by sorting
        action_id = slots_to_action_id((1, 0), ACTION_TABLE_N5_K2)
        assert action_id == slots_to_action_id((0, 1), ACTION_TABLE_N5_K2)


class TestValidateActionTable:
    """Test action table validation function."""

    def test_valid_table_returns_true(self) -> None:
        """Valid action table should return True."""
        assert validate_action_table(ACTION_TABLE_N5_K2, 5, 2) is True

    def test_wrong_n_raises(self) -> None:
        """Validating with wrong N should raise."""
        with pytest.raises(ValueError, match="expected"):
            validate_action_table(ACTION_TABLE_N5_K2, 4, 2)  # Table has slots up to 4

    def test_wrong_k_raises(self) -> None:
        """Validating with wrong K should raise."""
        with pytest.raises(ValueError, match="expected"):
            validate_action_table(ACTION_TABLE_N5_K2, 5, 1)  # Table has pairs

    def test_empty_table_raises(self) -> None:
        """Empty action table should raise."""
        with pytest.raises(ValueError, match="expected"):
            validate_action_table((), 5, 2)


class TestBuildActionTableEdgeCases:
    """Test build_action_table with edge cases."""

    def test_k_equals_n(self) -> None:
        """K=N should produce all possible subsets."""
        table = build_action_table(3, 3)
        # Should have: 1 empty + 3 singletons + 3 pairs + 1 triple = 8
        assert len(table) == 8

    def test_k_zero_gives_only_empty(self) -> None:
        """K=0 should produce only the empty set."""
        table = build_action_table(5, 0)
        assert len(table) == 1
        assert table[0] == ()

    def test_n_one_k_one(self) -> None:
        """N=1, K=1 should produce 2 actions."""
        table = build_action_table(1, 1)
        assert len(table) == 2
        assert table[0] == ()
        assert table[1] == (0,)

    def test_invalid_n_zero_raises(self) -> None:
        """N=0 should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            build_action_table(0, 0)

    def test_invalid_negative_k_raises(self) -> None:
        """Negative K should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            build_action_table(5, -1)

    def test_invalid_k_greater_than_n_raises(self) -> None:
        """K > N should raise ValueError."""
        with pytest.raises(ValueError, match="cannot exceed"):
            build_action_table(3, 5)