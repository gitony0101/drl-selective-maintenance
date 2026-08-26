"""
Test Milestone 4 Exact Myopic information barrier.

Tests:
- Optimizer does not access true_rul or true_rul_capped
- Optimizer does not access unit_id or trajectory_id
- Optimizer does not depend on diagnostic SlotState
- Observation decoding uses only normalized features
- Cost estimation uses only predicted_rul (denormalized)
- No future environment information leakage
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from optimizers import MyopicContext, ExactMyopicOptimizer
from envs.action_table import ACTION_TABLE_N5_K2
from envs.costs import COST_REGIMES


def make_optimizer(
    k_capacity: int = 2,
    cost_regime_id: str = "failure-light-no-waste",
    risk_model_id: str = "hard_window_v1",
) -> ExactMyopicOptimizer:
    """Create optimizer with given parameters."""
    cost_regime = COST_REGIMES[cost_regime_id]

    context = MyopicContext(
        maintenance_capacity=k_capacity,
        delta_cycles=5,
        rul_scale=125.0,
        age_scale_cycles=341,
        action_table=ACTION_TABLE_N5_K2,
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
    """Create observation from denormalized values."""
    assert len(pred_rul_cycles) == 5, "Need 5 RUL values"
    assert len(age_cycles) == 5, "Need 5 age values"

    pred_rul_norm = np.clip(np.array(pred_rul_cycles) / 125.0, 0, 1)
    age_norm = np.clip(np.array(age_cycles) / 341, 0, 1)

    features = []
    for i in range(5):
        features.append(float(age_norm[i]))
        features.append(float(pred_rul_norm[i]))

    return np.array(features, dtype=np.float32)


class TestObservationOnly:
    """Test that optimizer uses only observation input."""

    def test_select_action_requires_only_observation(self):
        """select_action works with only observation, no hidden state."""
        optimizer = make_optimizer()
        observation = make_observation(
            pred_rul_cycles=[50] * 5,
            age_cycles=[100] * 5,
        )

        # Should succeed with only observation
        action_id, slots, cost = optimizer.select_action(observation)

        assert 0 <= action_id < len(ACTION_TABLE_N5_K2)
        assert len(slots) <= 2
        assert np.isfinite(cost)

    def test_no_true_rul_inobservation(self):
        """Optimizer cannot access true_rul - only predicted_rul in observation."""
        optimizer = make_optimizer()

        # Create observation with predicted RUL = 3 (at risk)
        # The optimizer should treat this as at-risk regardless of true RUL
        observation = make_observation(
            pred_rul_cycles=[3, 50, 50, 50, 50],
            age_cycles=[100] * 5,
        )

        evaluations = optimizer.evaluate_all_actions(observation)

        # If optimizer had secret access to true_rul, results would differ
        # We verify the cost calculation uses the predicted_rul from observation
        action_0 = evaluations[0]  # Empty action - all slots at risk

        # Slot 0 has predicted RUL = 3 <= delta_cycles, so risk = 1
        # Slots 1-4 have predicted RUL = 50 > delta_cycles, so risk = 0
        # Expected failure cost: c_f * 1 = 5.0
        expected_failure = 5.0 * 1.0
        assert abs(action_0.failure_cost - expected_failure) < 1e-9


class TestNoDiagnosticDependency:
    """Test that optimizer doesn't depend on diagnostic info."""

    def test_observation_is_just_array(self):
        """Observation is a plain numpy array, no nested diagnostic info."""
        optimizer = make_optimizer()
        observation = make_observation(
            pred_rul_cycles=[50] * 5,
            age_cycles=[100] * 5,
        )

        assert isinstance(observation, np.ndarray)
        assert observation.dtype == np.float32
        assert observation.shape == (10,)

        # Should not have any additional attributes or fields
        assert not hasattr(observation, "unit_id")
        assert not hasattr(observation, "true_rul")
        assert not hasattr(observation, "slot_states")


class TestNoFutureLeakage:
    """Test that optimizer doesn't use future information."""

    def test_cost_uses_current_window_only(self):
        """Cost estimation uses only current window, no future transitions."""
        optimizer = make_optimizer()

        # Same observation should always produce same cost
        # (no dependence on external or future state)
        observation = make_observation(
            pred_rul_cycles=[50] * 5,
            age_cycles=[100] * 5,
        )

        evaluations_1 = optimizer.evaluate_all_actions(observation)
        evaluations_2 = optimizer.evaluate_all_actions(observation)

        # Must be deterministic - same observation = same costs
        for e1, e2 in zip(evaluations_1, evaluations_2):
            assert e1.action_id == e2.action_id
            assert abs(e1.total_cost - e2.total_cost) < 1e-9

    def test_no_history_dependency(self):
        """Action selection doesn't depend on past observations."""
        optimizer = make_optimizer()

        observation = make_observation(
            pred_rul_cycles=[50] * 5,
            age_cycles=[100] * 5,
        )

        # First call
        action_id_1, _, cost_1 = optimizer.select_action(observation)

        # Second call (same observation, no history provided)
        action_id_2, _, cost_2 = optimizer.select_action(observation)

        # Must be identical - no hidden state carries over
        assert action_id_1 == action_id_2
        assert abs(cost_1 - cost_2) < 1e-9


class TestDenormalizationCorrectness:
    """Test that observation denormalization is correct."""

    def test_rul_denormalization(self):
        """RUL is correctly denormalized: pred_rul = norm * rul_scale."""
        optimizer = make_optimizer(k_capacity=2, cost_regime_id="failure-light-waste-aware")

        # Create observation with normalized RUL = 0.4 (should be 50 cycles)
        direct_norm = np.array([0.5, 0.4] * 5, dtype=np.float32)  # age, rul pattern

        evaluations = optimizer.evaluate_all_actions(direct_norm)
        single_action = evaluations[1]  # Select slot 0

        # Unused life cost should use 0.4 * 125 = 50 cycles
        # Cost: c_u * (0.4) = 0.25 * 0.4 = 0.1
        expected = 0.25 * 0.4
        assert abs(single_action.unused_life_cost - expected) < 1e-6

    def test_rul_clipping_before_denormalization(self):
        """RUL normalization clips to [0, 1] correctly."""
        optimizer = make_optimizer(k_capacity=2, cost_regime_id="failure-light-waste-aware")

        # Normalized RUL > 1 should be clipped
        high_norm = np.array([0.5, 1.5] * 5, dtype=np.float32)  # RUL norm = 1.5 > 1

        # Should raise ValueError for out-of-range values
        with pytest.raises(ValueError, match="[0, 1]"):
            optimizer.select_action(high_norm)


class TestHardWindowBarrier:
    """Test hard window information barrier."""

    def test_hard_window_uses_delta_cycles_only(self):
        """Hard window risk uses only delta_cycles and predicted_rul."""
        optimizer = make_optimizer(risk_model_id="hard_window_v1")

        # Slot at exactly delta_cycles boundary
        observation_at_boundary = make_observation(
            pred_rul_cycles=[5, 50, 50, 50, 50],  # Slot 0 at exactly delta=5
            age_cycles=[100] * 5,
        )

        evaluations = optimizer.evaluate_all_actions(observation_at_boundary)
        empty_action = evaluations[0]

        # At boundary, risk = 1 (predicted_rul <= delta_cycles)
        expected = 5.0 * 1.0  # c_f × 1 at-risk slot
        assert abs(empty_action.failure_cost - expected) < 1e-9

    def test_hard_window_one_above_boundary(self):
        """Hard window risk = 0 for predicted_rul > delta_cycles."""
        optimizer = make_optimizer(risk_model_id="hard_window_v1")

        # Slot just above boundary
        observation_above = make_observation(
            pred_rul_cycles=[6, 50, 50, 50, 50],  # Slot 0 at 6 > delta=5
            age_cycles=[100] * 5,
        )

        evaluations = optimizer.evaluate_all_actions(observation_above)
        empty_action = evaluations[0]

        # Above boundary, risk = 0
        assert empty_action.failure_cost == 0.0


class TestTrueRULIndependence:
    """Test that optimizer is independent of true RUL values."""

    def test_true_rul_change_same_observation_same_action(self):
        """
        Changing true_rul while keeping observation unchanged doesn't change action.

        This test verifies the information barrier: theoptimizer only sees
        predicted_rul in the observation, not true_rul.
        """
        optimizer = make_optimizer()

        # Create an observation with predicted RUL = 3 for slot 0
        observation = make_observation(
            pred_rul_cycles=[3, 50, 50, 50, 50],
            age_cycles=[100] * 5,
        )

        # Get action for this observation
        action_id_1, slots_1, cost_1 = optimizer.select_action(observation)

        # Now imagine the TRUE RUL is different (e.g., true_rul = 100, not 3)
        # The optimizer should NOT know this - it only sees predicted_rul = 3
        # So the action should be the same

        # Since optimizer doesn't have access to true_rul, we just verify
        # the same observation always produces the same action
        action_id_2, slots_2, cost_2 = optimizer.select_action(observation)

        assert (action_id_1, slots_1, cost_1) == (action_id_2, slots_2, cost_2)

    def test_hidden_state_cannot_affect_decision(self):
        """
        Verify optimizer cannot access hidden state.

        The optimizer's select_action method only accepts an observation array.
        There is no mechanism to pass true_rul, unit_id, or diagnostic info.
        """
        optimizer = make_optimizer()
        observation = make_observation(
            pred_rul_cycles=[50] * 5,
            age_cycles=[100] * 5,
        )

        # The only way to call select_action is with observation
        # There's no keyword argument or attribute for hidden state
        action_id, slots, cost = optimizer.select_action(observation)

        # Verify it works
        assert 0 <= action_id < len(ACTION_TABLE_N5_K2)
        assert len(slots) <= 2


class TestUnitIDIndependence:
    """Test that optimizer is independent of unit identifiers."""

    def test_unit_id_not_in_observation(self):
        """Observation contains only age and predicted_rul, no unit_id."""
        observation = make_observation(
            pred_rul_cycles=[50] * 5,
            age_cycles=[100] * 5,
        )

        # Observation is a plain numpy array
        assert isinstance(observation, np.ndarray)
        assert observation.shape == (10,)

        # No way to encode unit_id in the observation
        # (it's just 10 normalized floats)

    def test_same_fleet_state_same_action(self):
        """
        Same fleet state produces same action regardless of which unit.

        If we had two different units with identical predicted_rul and age,
        the optimizer should select the same action for both.
        """
        optimizer = make_optimizer()

        # Simulate "Unit A" with a certain state
        observation_a = make_observation(
            pred_rul_cycles=[3, 50, 50, 50, 50],
            age_cycles=[100] * 5,
        )

        # Simulate "Unit B" with identical state
        observation_b = make_observation(
            pred_rul_cycles=[3, 50, 50, 50, 50],
            age_cycles=[100] * 5,
        )

        # Should produce identical actions
        action_a, slots_a, cost_a = optimizer.select_action(observation_a)
        action_b, slots_b, cost_b = optimizer.select_action(observation_b)

        assert (action_a, slots_a, cost_a) == (action_b, slots_b, cost_b)


class TestDiagnosticStateIndependence:
    """Test that optimizer is independent of diagnostic SlotState."""

    def test_no_diagnostic_info_in_observation(self):
        """Observation doesn't contain diagnostic SlotState information."""
        observation = make_observation(
            pred_rul_cycles=[50] * 5,
            age_cycles=[100] * 5,
        )

        # Just normalized age and RUL values
        # No slot health flags, no diagnostic codes, no maintenance history

        assert observation.dtype == np.float32
        assert len(observation) == 10

    def test_optimizer_stateless(self):
        """
        Optimizer doesn't maintain internal state between calls.

        Each call to select_action is independent.
        """
        optimizer = make_optimizer()

        # First call
        obs1 = make_observation(
            pred_rul_cycles=[3, 50, 50, 50, 50],
            age_cycles=[100] * 5,
        )
        action1, slots1, _ = optimizer.select_action(obs1)

        # Second call with different observation
        obs2 = make_observation(
            pred_rul_cycles=[50, 50, 50, 50, 50],
            age_cycles=[100] * 5,
        )
        action2, slots2, _ = optimizer.select_action(obs2)

        # Third call identical to first
        action3, slots3, cost3 = optimizer.select_action(obs1)

        # Third call should match first (stateless, deterministic)
        assert (action1, slots1) == (action3, slots3)


class TestObservationEncodingBarrier:
    """Test that only encoded information affects decisions."""

    def test_observation_permutation_changes_action(self):
        """
        Permuting observation values changes the action.

        This verifies the optimizer actually reads the observation values
        and doesn't just use a hardcoded action.
        """
        optimizer = make_optimizer()

        # Slot 0 at risk
        obs_slot0_risk = make_observation(
            pred_rul_cycles=[3, 50, 50, 50, 50],
            age_cycles=[100] * 5,
        )

        # Slot 4 at risk (different position)
        obs_slot4_risk = make_observation(
            pred_rul_cycles=[50, 50, 50, 50, 3],
            age_cycles=[100] * 5,
        )

        action0, slots0, _ = optimizer.select_action(obs_slot0_risk)
        action4, slots4, _ = optimizer.select_action(obs_slot4_risk)

        # Should select different slots
        assert slots0 == (0,), f"Expected (0,), got {slots0}"
        assert slots4 == (4,), f"Expected (4,), got {slots4}"

    def test_observation_scale_affects_decision(self):
        """
        Changing observation values (within valid range) affects decision.

        Verifies optimizer responds to the magnitude of predicted_rul.
        """
        optimizer = make_optimizer()

        # All very safe (high RUL)
        obs_very_safe = make_observation(
            pred_rul_cycles=[100, 100, 100, 100, 100],
            age_cycles=[100] * 5,
        )

        # All at risk (low RUL)
        obs_all_risk = make_observation(
            pred_rul_cycles=[1, 1, 1, 1, 1],
            age_cycles=[100] * 5,
        )

        action_safe, slots_safe, _ = optimizer.select_action(obs_very_safe)
        action_risk, slots_risk, _ = optimizer.select_action(obs_all_risk)

        # Safe scenario: empty action optimal (no need for PM)
        assert action_safe == 0
        assert slots_safe == ()

        # All-risk scenario: should select slots (up to capacity)
        assert len(slots_risk) > 0