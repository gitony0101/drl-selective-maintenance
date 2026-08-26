"""
Test Milestone 4 Exact Myopic cost decomposition.

Tests:
- Preventive cost equals c_pm × selected count
- Unused-life cost uses predicted RUL normalized by rul_scale
- Selected slots have no estimated failure charge
- Non-selected slots receive failure risk
- Component sum equals total
- All four cost regimes produce expected components
- Invalid and non-finite inputs fail clearly
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from optimizers import MyopicContext, ExactMyopicOptimizer
from envs.action_table import ACTION_TABLE_N5_K1, ACTION_TABLE_N5_K2
from envs.costs import COST_REGIMES


def make_optimizer(
    k_capacity: int = 2,
    cost_regime_id: str = "failure-light-no-waste",
    risk_model_id: str = "hard_window_v1",
) -> ExactMyopicOptimizer:
    """Create optimizer with given parameters."""
    cost_regime = COST_REGIMES[cost_regime_id]
    action_table = ACTION_TABLE_N5_K1 if k_capacity == 1 else ACTION_TABLE_N5_K2

    context = MyopicContext(
        maintenance_capacity=k_capacity,
        delta_cycles=5,
        rul_scale=125.0,
        age_scale_cycles=341,
        action_table=action_table,
        c_pm=cost_regime.c_pm,
        c_f=cost_regime.c_f,
        c_u=cost_regime.c_u,
        risk_model_id=risk_model_id,
    )

    return ExactMyopicOptimizer(context=context)


def make_observation(
    pred_rul_cycles: list[float],
    age_cycles: list[float],
) -> np.ndarray:
    """
    Create observation from denormalized values.

    Args:
        pred_rul_cycles: Predicted RUL in cycles for each slot (5 values)
        age_cycles: Age in cycles for each slot (5 values)

    Returns:
        Observation ndarray, shape (10,), dtype float32
    """
    assert len(pred_rul_cycles) == 5, "Need 5 RUL values"
    assert len(age_cycles) == 5, "Need 5 age values"

    # Normalize
    pred_rul_norm = np.clip(np.array(pred_rul_cycles) / 125.0, 0, 1)
    age_norm = np.clip(np.array(age_cycles) / 341, 0, 1)

    # Interleave: [slot_0_age, slot_0_rul, slot_1_age, slot_1_rul, ...]
    features = []
    for i in range(5):
        features.append(float(age_norm[i]))
        features.append(float(pred_rul_norm[i]))

    return np.array(features, dtype=np.float32)


class TestPreventiveCost:
    """Test preventive cost calculation."""

    def test_empty_action_zero_cost(self):
        """Action 0 (empty) has zero preventive cost."""
        optimizer = make_optimizer(k_capacity=2, cost_regime_id="failure-light-no-waste")
        observation = make_observation(
            pred_rul_cycles=[50] * 5,
            age_cycles=[100] * 5,
        )

        evaluations = optimizer.evaluate_all_actions(observation)
        empty_action = evaluations[0]

        assert empty_action.action_id == 0
        assert empty_action.selected_slots == ()
        assert empty_action.preventive_cost == 0.0

    def test_single_slot_pm_cost(self):
        """Preventive cost equals c_pm for one slot."""
        optimizer = make_optimizer(k_capacity=2, cost_regime_id="failure-light-no-waste")
        observation = make_observation(
            pred_rul_cycles=[50] * 5,
            age_cycles=[100] * 5,
        )

        evaluations = optimizer.evaluate_all_actions(observation)
        # Action 1 selects slot 0
        single_action = evaluations[1]

        assert single_action.selected_slots == (0,)
        assert single_action.preventive_cost == 1.0  # c_pm = 1.0

    def test_two_slot_pm_cost(self):
        """Preventive cost equals 2 × c_pm for two slots."""
        optimizer = make_optimizer(k_capacity=2, cost_regime_id="failure-light-no-waste")
        observation = make_observation(
            pred_rul_cycles=[50] * 5,
            age_cycles=[100] * 5,
        )

        evaluations = optimizer.evaluate_all_actions(observation)
        # Action 6 selects slots (0, 1)
        two_action = evaluations[6]

        assert two_action.selected_slots == (0, 1)
        assert two_action.preventive_cost == 2.0  # 2 × c_pm

    def test_k1_single_slot(self):
        """K=1 preventive cost for single slot."""
        optimizer = make_optimizer(k_capacity=1, cost_regime_id="failure-heavy-no-waste")
        observation = make_observation(
            pred_rul_cycles=[50] * 5,
            age_cycles=[100] * 5,
        )

        evaluations = optimizer.evaluate_all_actions(observation)
        # Action 1 selects slot 0
        single_action = evaluations[1]

        assert single_action.preventive_cost == 1.0  # c_pm = 1.0


class TestUnusedLifeCost:
    """Test unused life cost calculation."""

    def test_empty_action_zero_unused_life(self):
        """Empty action has zero unused life cost."""
        optimizer = make_optimizer(k_capacity=2)
        observation = make_observation(
            pred_rul_cycles=[50] * 5,
            age_cycles=[100] * 5,
        )

        evaluations = optimizer.evaluate_all_actions(observation)
        empty_action = evaluations[0]

        assert empty_action.unused_life_cost == 0.0

    def test_single_slot_unused_life(self):
        """Unused life cost for one slot with RUL=50."""
        # c_u = 0 for no-waste regimes, use waste-aware
        optimizer = make_optimizer(k_capacity=2, cost_regime_id="failure-light-waste-aware")
        observation = make_observation(
            pred_rul_cycles=[50] * 5,  # RUL = 50 cycles
            age_cycles=[100] * 5,
        )

        evaluations = optimizer.evaluate_all_actions(observation)
        # Action 1 selects slot 0
        single_action = evaluations[1]

        # Expected: c_u × (RUL / rul_scale) = 0.25 × (50 / 125) = 0.25 × 0.4 = 0.1
        expected = 0.25 * (50.0 / 125.0)
        assert abs(single_action.unused_life_cost - expected) < 1e-6

    def test_two_slot_unused_life(self):
        """Unused life cost for two slots."""
        optimizer = make_optimizer(k_capacity=2, cost_regime_id="failure-light-waste-aware")
        observation = make_observation(
            pred_rul_cycles=[25, 75] + [50] * 3,  # Slots 0,1 have RUL 25, 75
            age_cycles=[100] * 5,
        )

        evaluations = optimizer.evaluate_all_actions(observation)
        # Action 6 selects slots (0, 1)
        two_action = evaluations[6]

        # Expected: c_u × (RUL0/125 + RUL1/125) = 0.25 × (25/125 + 75/125) = 0.25 × 0.8 = 0.2
        expected = 0.25 * (25.0 / 125.0 + 75.0 / 125.0)
        assert abs(two_action.unused_life_cost - expected) < 1e-6

    def test_rul_clipping(self):
        """Unused life cost clips RUL > 125 to 1."""
        optimizer = make_optimizer(k_capacity=2, cost_regime_id="failure-light-waste-aware")
        observation = make_observation(
            pred_rul_cycles=[150] * 5,  # RUL > 125, should clip to 1.0
            age_cycles=[100] * 5,
        )

        evaluations = optimizer.evaluate_all_actions(observation)
        single_action = evaluations[1]

        # Expected: c_u × clip(150/125, 0, 1) = 0.25 × 1.0 = 0.25
        expected = 0.25 * 1.0
        assert abs(single_action.unused_life_cost - expected) < 1e-9


class TestFailureCost:
    """Test failure cost calculation."""

    def test_empty_action_all_slots_at_risk(self):
        """Empty action: all slots receive failure risk."""
        optimizer = make_optimizer(
            k_capacity=2,
            cost_regime_id="failure-light-no-waste",
            risk_model_id="hard_window_v1",
        )
        # All slots with RUL <= 5 will have risk = 1
        observation = make_observation(
            pred_rul_cycles=[3, 3, 3, 3, 3],  # All <= delta_cycles
            age_cycles=[100] * 5,
        )

        evaluations = optimizer.evaluate_all_actions(observation)
        empty_action = evaluations[0]

        # All 5 slots at risk, each with risk=1, c_f=5
        expected = 5.0 * 5.0  # c_f × 5 slots
        assert abs(empty_action.failure_cost - expected) < 1e-9

    def test_selected_slots_no_failure_risk(self):
        """Selected slots do NOT receive failure cost."""
        optimizer = make_optimizer(
            k_capacity=2,
            cost_regime_id="failure-light-no-waste",
            risk_model_id="hard_window_v1",
        )
        # Slot 0 has low RUL (would fail), slot 1-4 safe
        observation = make_observation(
            pred_rul_cycles=[3, 50, 50, 50, 50],
            age_cycles=[100] * 5,
        )

        evaluations = optimizer.evaluate_all_actions(observation)
        # Action 1 selects slot 0 (the one at risk)
        select_action = evaluations[1]

        # Slot 0 is selected, so no failure risk for it
        # Slots 1-4 are safe (RUL=50 > 5), so risk=0
        assert select_action.failure_cost == 0.0

    def test_non_selected_slots_at_risk(self):
        """Non-selected slots receive failure risk."""
        optimizer = make_optimizer(
            k_capacity=2,
            cost_regime_id="failure-light-no-waste",
            risk_model_id="hard_window_v1",
        )
        # Slots 2,3 have low RUL (at risk)
        observation = make_observation(
            pred_rul_cycles=[50, 50, 3, 3, 50],
            age_cycles=[100] * 5,
        )

        evaluations = optimizer.evaluate_all_actions(observation)
        # Action 1 selects only slot 0 (not slots 2,3 at risk)
        action = evaluations[1]

        # Slots 2, 3 are non-selected and at risk (RUL=3 <= 5)
        # Each has risk=1, c_f=5
        expected = 5.0 * 2.0  # c_f × 2 at-risk slots
        assert abs(action.failure_cost - expected) < 1e-9


class TestTotalCost:
    """Test total cost calculation."""

    def test_component_sum(self):
        """Total cost equals sum of components."""
        optimizer = make_optimizer(
            k_capacity=2,
            cost_regime_id="failure-light-waste-aware",
            risk_model_id="hard_window_v1",
        )
        observation = make_observation(
            pred_rul_cycles=[3, 50, 50, 50, 50],  # Slot 0 at risk
            age_cycles=[100] * 5,
        )

        evaluations = optimizer.evaluate_all_actions(observation)
        action = evaluations[1]  # Select slot 0

        # Components
        pm = action.preventive_cost
        waste = action.unused_life_cost
        failure = action.failure_cost
        total = action.total_cost

        # Total must equal sum
        assert abs(total - (pm + waste + failure)) < 1e-9

    def test_all_four_regimes(self):
        """All four cost regimes produce valid components."""
        regimes = list(COST_REGIMES.keys())
        observation = make_observation(
            pred_rul_cycles=[10, 50, 50, 50, 50],
            age_cycles=[100] * 5,
        )

        for regime_id in regimes:
            optimizer = make_optimizer(
                k_capacity=2,
                cost_regime_id=regime_id,
                risk_model_id="hard_window_v1",
            )
            evaluations = optimizer.evaluate_all_actions(observation)

            for action in evaluations:
                # All components must be finite and non-negative
                assert action.preventive_cost >= 0, f"{regime_id}: preventive_cost < 0"
                assert action.unused_life_cost >= 0, f"{regime_id}: unused_life_cost < 0"
                assert action.failure_cost >= 0, f"{regime_id}: failure_cost < 0"
                assert np.isfinite(action.total_cost), f"{regime_id}: total_cost not finite"


class TestInvalidInputs:
    """Test invalid input rejection."""

    def test_invalid_observation_shape(self):
        """Reject observation with wrong shape."""
        optimizer = make_optimizer()

        # Wrong shape
        bad_obs = np.zeros((5,), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            optimizer.select_action(bad_obs)

        # Too long
        long_obs = np.zeros((12,), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            optimizer.select_action(long_obs)

    def test_non_finite_values(self):
        """Reject observations with NaN or Inf."""
        optimizer = make_optimizer()

        # NaN
        nan_obs = np.array([0.5, 0.5, float('nan')] + [0.5] * 7, dtype=np.float32)
        with pytest.raises(ValueError, match="non-finite"):
            optimizer.select_action(nan_obs)

        # Inf
        inf_obs = np.array([0.5, float('inf')] + [0.5] * 8, dtype=np.float32)
        with pytest.raises(ValueError, match="non-finite"):
            optimizer.select_action(inf_obs)

    def test_out_of_range_values(self):
        """Reject observations outside [0, 1]."""
        optimizer = make_optimizer()

        # Negative
        neg_obs = np.array([-0.1] + [0.5] * 9, dtype=np.float32)
        with pytest.raises(ValueError, match="[0, 1]"):
            optimizer.select_action(neg_obs)

        # Greater than 1
        high_obs = np.array([1.5] + [0.5] * 9, dtype=np.float32)
        with pytest.raises(ValueError, match="[0, 1]"):
            optimizer.select_action(high_obs)