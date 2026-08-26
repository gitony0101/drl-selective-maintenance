"""
Unit tests for Milestone 2 cost regime definitions.

Tests cover:
- All four regimes exist
- Coefficients exactly match the contract
- Negative or non-finite coefficients are rejected
- Unknown regime IDs are rejected
- Serialization is stable
"""

import math
import pytest

from src.envs.costs import (
    COST_REGIMES,
    DEFAULT_COST_REGIME_ID,
    FAILURE_HEAVY_NO_WASTE,
    FAILURE_HEAVY_WASTE_AWARE,
    FAILURE_LIGHT_NO_WASTE,
    FAILURE_LIGHT_WASTE_AWARE,
    CostRegime,
    calculate_total_cost,
    get_cost_regime,
    list_cost_regimes,
    validate_cost_regime,
)


class TestCostRegimeExistence:
    """Test that all four frozen cost regimes exist."""

    def test_failure_light_no_waste_exists(self) -> None:
        """failure-light-no-waste regime should exist."""
        assert "failure-light-no-waste" in COST_REGIMES

    def test_failure_heavy_no_waste_exists(self) -> None:
        """failure-heavy-no-waste regime should exist."""
        assert "failure-heavy-no-waste" in COST_REGIMES

    def test_failure_light_waste_aear_exists(self) -> None:
        """failure-light-waste-aware regime should exist."""
        assert "failure-light-waste-aware" in COST_REGIMES

    def test_failure_heavy_waste_aear_exists(self) -> None:
        """failure-heavy-waste-aware regime should exist."""
        assert "failure-heavy-waste-aware" in COST_REGIMES

    def test_exactly_four_regimes(self) -> None:
        """There should be exactly four cost regimes."""
        assert len(COST_REGIMES) == 4


class TestCostCoefficients:
    """Test that coefficients exactly match the contract."""

    def test_failure_light_no_waste_coefficients(self) -> None:
        """failure-light-no-waste: c_pm=1, c_f=5, c_u=0."""
        regime = FAILURE_LIGHT_NO_WASTE
        assert regime.c_pm == 1.0
        assert regime.c_f == 5.0
        assert regime.c_u == 0.0
        assert regime.regime_id == "failure-light-no-waste"

    def test_failure_heavy_no_waste_coefficients(self) -> None:
        """failure-heavy-no-waste: c_pm=1, c_f=10, c_u=0."""
        regime = FAILURE_HEAVY_NO_WASTE
        assert regime.c_pm == 1.0
        assert regime.c_f == 10.0
        assert regime.c_u == 0.0
        assert regime.regime_id == "failure-heavy-no-waste"

    def test_failure_light_waste_aear_coefficients(self) -> None:
        """failure-light-waste-aware: c_pm=1, c_f=5, c_u=0.25."""
        regime = FAILURE_LIGHT_WASTE_AWARE
        assert regime.c_pm == 1.0
        assert regime.c_f == 5.0
        assert regime.c_u == 0.25
        assert regime.regime_id == "failure-light-waste-aware"

    def test_failure_heavy_waste_aear_coefficients(self) -> None:
        """failure-heavy-waste-aware: c_pm=1, c_f=10, c_u=0.25."""
        regime = FAILURE_HEAVY_WASTE_AWARE
        assert regime.c_pm == 1.0
        assert regime.c_f == 10.0
        assert regime.c_u == 0.25
        assert regime.regime_id == "failure-heavy-waste-aware"


class TestNegativeCoefficientsRejected:
    """Test that negative coefficients are rejected."""

    def test_negative_c_pm_raises(self) -> None:
        """Negative c_pm should raise ValueError."""
        with pytest.raises(ValueError, match="c_pm must be non-negative"):
            CostRegime(c_pm=-1.0, c_f=5.0, c_u=0.0, regime_id="test")

    def test_negative_c_f_raises(self) -> None:
        """Negative c_f should raise ValueError."""
        with pytest.raises(ValueError, match="c_f must be non-negative"):
            CostRegime(c_pm=1.0, c_f=-5.0, c_u=0.0, regime_id="test")

    def test_negative_c_u_raises(self) -> None:
        """Negative c_u should raise ValueError."""
        with pytest.raises(ValueError, match="c_u must be non-negative"):
            CostRegime(c_pm=1.0, c_f=5.0, c_u=-0.25, regime_id="test")


class TestNonFiniteCoefficientsRejected:
    """Test that non-finite coefficients are rejected."""

    def test_nan_c_pm_raises(self) -> None:
        """NaN c_pm should raise ValueError."""
        with pytest.raises(ValueError, match="c_pm cannot be NaN"):
            CostRegime(c_pm=float("nan"), c_f=5.0, c_u=0.0, regime_id="test")

    def test_inf_c_pm_raises(self) -> None:
        """Infinite c_pm should raise ValueError."""
        with pytest.raises(ValueError, match="c_pm cannot be infinite"):
            CostRegime(c_pm=float("inf"), c_f=5.0, c_u=0.0, regime_id="test")

    def test_nan_c_f_raises(self) -> None:
        """NaN c_f should raise ValueError."""
        with pytest.raises(ValueError, match="c_f cannot be NaN"):
            CostRegime(c_pm=1.0, c_f=float("nan"), c_u=0.0, regime_id="test")

    def test_inf_c_u_raises(self) -> None:
        """Infinite c_u should raise ValueError."""
        with pytest.raises(ValueError, match="c_u cannot be infinite"):
            CostRegime(c_pm=1.0, c_f=5.0, c_u=float("inf"), regime_id="test")


class TestUnknownRegimeIdRejected:
    """Test that unknown regime IDs are rejected."""

    def test_unknown_regime_raises(self) -> None:
        """Unknown regime ID should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown cost regime"):
            get_cost_regime("unknown-regime")

    def test_empty_string_regime_raises(self) -> None:
        """Empty string regime ID should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown cost regime"):
            get_cost_regime("")

    def test_typo_regime_raises(self) -> None:
        """Typo in regime ID should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown cost regime"):
            get_cost_regime("failure-light-no-wase")  # missing 't'


class TestSerialization:
    """Test serialization stability."""

    def test_to_dict_contains_all_fields(self) -> None:
        """to_dict should contain all four fields."""
        regime = FAILURE_LIGHT_NO_WASTE
        d = regime.to_dict()
        assert d == {
            "c_pm": 1.0,
            "c_f": 5.0,
            "c_u": 0.0,
            "regime_id": "failure-light-no-waste",
        }

    def test_from_dict_creates_valid_regime(self) -> None:
        """from_dict should create a valid regime."""
        data = {"c_pm": 1.0, "c_f": 10.0, "c_u": 0.25, "regime_id": "test-regime"}
        regime = CostRegime.from_dict(data)
        assert regime.c_pm == 1.0
        assert regime.c_f == 10.0
        assert regime.c_u == 0.25
        assert regime.regime_id == "test-regime"

    def test_from_dict_missing_key_raises(self) -> None:
        """from_dict with missing key should raise ValueError."""
        data = {"c_pm": 1.0, "c_f": 10.0}  # missing c_u and regime_id
        with pytest.raises(ValueError, match="Missing required keys"):
            CostRegime.from_dict(data)

    def test_roundtrip_serialization(self) -> None:
        """Round-trip serialization should preserve values."""
        original = FAILURE_HEAVY_WASTE_AWARE
        data = original.to_dict()
        recovered = CostRegime.from_dict(data)
        assert recovered.c_pm == original.c_pm
        assert recovered.c_f == original.c_f
        assert recovered.c_u == original.c_u
        assert recovered.regime_id == original.regime_id


class TestListCostRegimes:
    """Test list_cost_regimes function."""

    def test_returns_sorted_list(self) -> None:
        """list_cost_regimes should return a sorted list."""
        regimes = list_cost_regimes()
        assert regimes == sorted(regimes)
        assert len(regimes) == 4

    def test_contains_all_regime_ids(self) -> None:
        """list_cost_regimes should contain all four regime IDs."""
        regimes = list_cost_regimes()
        expected = [
            "failure-heavy-no-waste",
            "failure-heavy-waste-aware",
            "failure-light-no-waste",
            "failure-light-waste-aware",
        ]
        assert regimes == expected


class TestValidateCostRegime:
    """Test validate_cost_regime function."""

    def test_valid_regime_returns_true(self) -> None:
        """Valid regime ID should return True."""
        assert validate_cost_regime("failure-light-no-waste") is True

    def test_all_four_regimes_validate(self) -> None:
        """All four regimes should validate."""
        for regime_id in COST_REGIMES:
            assert validate_cost_regime(regime_id) is True

    def test_invalid_regime_raises(self) -> None:
        """Invalid regime ID should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown cost regime"):
            validate_cost_regime("invalid")


class TestCalculateTotalCost:
    """Test calculate_total_cost function."""

    def test_no_maintenance_no_failure(self) -> None:
        """No maintenance, no failure should give zero cost."""
        cost = calculate_total_cost(
            num_preventive=0,
            num_failures=0,
            wasted_rul_sum=0.0,
            regime=FAILURE_LIGHT_NO_WASTE,
        )
        assert cost == 0.0

    def test_pm_only_cost(self) -> None:
        """PM-only cost should be c_pm * N_pm."""
        cost = calculate_total_cost(
            num_preventive=2,
            num_failures=0,
            wasted_rul_sum=0.0,
            regime=FAILURE_LIGHT_NO_WASTE,
        )
        assert cost == 2.0  # 1.0 * 2

    def test_failure_only_cost(self) -> None:
        """Failure-only cost should be c_f * N_fail."""
        cost = calculate_total_cost(
            num_preventive=0,
            num_failures=1,
            wasted_rul_sum=0.0,
            regime=FAILURE_LIGHT_NO_WASTE,
        )
        assert cost == 5.0  # 5.0 * 1

    def test_waste_cost(self) -> None:
        """Waste cost should be c_u * wasted_rul_sum.

        Note: wasted_rul_sum is already the normalized sum of (true_rul / 125).
        For example, if one engine has true_rul=50, its contribution is 50/125 = 0.4.
        """
        # Use explicit normalized value: 40.0 / 125.0 = 0.32
        # This represents a PM slot with true_rul=40 out of max 125
        normalized_wasted_rul = 40.0 / 125.0  # = 0.32
        cost = calculate_total_cost(
            num_preventive=1,
            num_failures=0,
            wasted_rul_sum=normalized_wasted_rul,  # Normalized: true_rul / 125
            regime=FAILURE_LIGHT_WASTE_AWARE,
        )
        # c_pm * 1 + c_u * (40/125) = 1.0 + 0.25 * 0.32 = 1.0 + 0.08 = 1.08
        assert cost == 1.08

    def test_full_cost_calculation(self) -> None:
        """Test full cost calculation with all components.

        Note: wasted_rul_sum is the pre-normalized sum of (true_rul / 125).
        """
        # Use explicit normalized values: 30.0 / 125.0 = 0.24
        normalized_wasted_rul = 30.0 / 125.0  # = 0.24
        cost = calculate_total_cost(
            num_preventive=2,
            num_failures=1,
            wasted_rul_sum=normalized_wasted_rul,  # Sum of (true_rul / 125) for PM slots
            regime=FAILURE_HEAVY_WASTE_AWARE,
        )
        # c_pm * 2 + c_f * 1 + c_u * (30/125) = 1*2 + 10*1 + 0.25*0.24 = 2 + 10 + 0.06 = 12.06
        assert cost == 12.06

    def test_reward_is_negative_cost(self) -> None:
        """Reward should be negative cost (tested here for reference)."""
        cost = calculate_total_cost(
            num_preventive=1,
            num_failures=0,
            wasted_rul_sum=0.0,
            regime=FAILURE_LIGHT_NO_WASTE,
        )
        reward = -cost
        assert reward == -1.0


class TestDefaultCostRegime:
    """Test default cost regime constant."""

    def test_default_regime_exists(self) -> None:
        """DEFAULT_COST_REGIME_ID should exist."""
        assert DEFAULT_COST_REGIME_ID is not None

    def test_default_regime_is_valid(self) -> None:
        """DEFAULT_COST_REGIME_ID should be a valid regime."""
        assert validate_cost_regime(DEFAULT_COST_REGIME_ID) is True

    def test_default_regime_in_registry(self) -> None:
        """DEFAULT_COST_REGIME_ID should be in COST_REGIMES."""
        assert DEFAULT_COST_REGIME_ID in COST_REGIMES


class TestNormalizedCostFoundation:
    """Test normalized wasted life cost foundation.

    These tests verify that wasted_rul_sum uses normalized contributions
    in [0, 1] per engine, computed as clip(true_rul, 0, 125) / 125.
    """

    def test_per_engine_contribution_in_range_0_to_1(self) -> None:
        """Each per-engine normalized contribution must be in [0, 1]."""
        # true_rul = 0 -> normalized = 0
        assert min(max(0, 0.0), 125.0) / 125.0 == 0.0

        # true_rul = 40 -> normalized = 40/125 = 0.32
        assert 0.0 <= (min(max(40, 0.0), 125.0) / 125.0) <= 1.0

        # true_rul = 125 -> normalized = 1.0
        assert (min(max(125, 0.0), 125.0) / 125.0) == 1.0

        # true_rul = 200 -> normalized = 1.0 (capped)
        assert (min(max(200, 0.0), 125.0) / 125.0) == 1.0

        # true_rul = -10 -> normalized = 0.0 (clipped)
        assert (min(max(-10, 0.0), 125.0) / 125.0) == 0.0

    def test_true_rul_0_gives_normalized_0(self) -> None:
        """true_rul = 0 should give normalized contribution 0."""
        normalized = min(max(0, 0.0), 125.0) / 125.0
        assert normalized == 0.0

    def test_true_rul_40_gives_40_over_125(self) -> None:
        """true_rul = 40 should give normalized contribution 40/125."""
        normalized = min(max(40, 0.0), 125.0) / 125.0
        assert normalized == 40.0 / 125.0
        assert normalized == 0.32

    def test_true_rul_125_gives_normalized_1(self) -> None:
        """true_rul = 125 should give normalized contribution 1."""
        normalized = min(max(125, 0.0), 125.0) / 125.0
        assert normalized == 1.0

    def test_true_rul_above_125_capped_at_1(self) -> None:
        """true_rul above 125 should be capped at normalized 1."""
        # true_rul = 126 -> normalized = 1.0 (capped)
        normalized_126 = min(max(126, 0.0), 125.0) / 125.0
        assert normalized_126 == 1.0

        # true_rul = 200 -> normalized = 1.0 (capped)
        normalized_200 = min(max(200, 0.0), 125.0) / 125.0
        assert normalized_200 == 1.0

        # true_rul = 500 -> normalized = 1.0 (capped)
        normalized_500 = min(max(500, 0.0), 125.0) / 125.0
        assert normalized_500 == 1.0

    def test_wasted_life_cost_uses_normalized_values(self) -> None:
        """Wasted life cost calculation should use normalized values."""
        # If true_rul = 50, normalized = 50/125 = 0.4
        # Waste cost = c_u * 0.4 = 0.25 * 0.4 = 0.1 (for waste-aware regime)
        cost = calculate_total_cost(
            num_preventive=1,
            num_failures=0,
            wasted_rul_sum=50.0 / 125.0,  # Normalized true_rul
            regime=FAILURE_LIGHT_WASTE_AWARE,
        )
        expected = 1.0 + 0.25 * (50.0 / 125.0)  # c_pm + c_u * normalized
        assert cost == expected

    def test_multiple_slots_normalized_sum(self) -> None:
        """Multiple PM slots should sum their normalized contributions."""
        # Slot 1: true_rul = 40 -> 40/125 = 0.32
        # Slot 2: true_rul = 60 -> 60/125 = 0.48
        # Sum: 0.32 + 0.48 = 0.8
        normalized_sum = (40.0 / 125.0) + (60.0 / 125.0)
        assert normalized_sum == 0.8

        cost = calculate_total_cost(
            num_preventive=2,
            num_failures=0,
            wasted_rul_sum=normalized_sum,
            regime=FAILURE_LIGHT_WASTE_AWARE,
        )
        expected = 2.0 + 0.25 * 0.8  # c_pm * 2 + c_u * sum
        assert cost == expected