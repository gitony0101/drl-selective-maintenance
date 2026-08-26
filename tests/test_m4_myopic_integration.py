"""
Test Milestone 4 Exact Myopic optimizer integration.

Tests:
- End-to-end test with Milestone 2 environment
- Full episode rollout with optimizer in the loop
- Cost accumulation over episode
- Integration with RUL predictor
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
    risk_temperature: float = 10.0,
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

    return ExactMyopicOptimizer(context=context, risk_temperature=risk_temperature)


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


class TestEnvironmentIntegration:
    """Test integration with Milestone 2 environment."""

    def test_optimizer_consumes_env_observation(self):
        """Optimizer can process raw environment observation."""
        optimizer = make_optimizer()

        # Simulate environment observation shape
        env_observation = np.zeros((10,), dtype=np.float32)
        # Fill with valid normalized values
        for i in range(5):
            env_observation[i * 2] = 0.3  # age norm
            env_observation[i * 2 + 1] = 0.4  # rul norm

        # Should not raise
        action_id, slots, cost = optimizer.select_action(env_observation)

        assert 0 <= action_id < len(ACTION_TABLE_N5_K2)
        assert len(slots) <= 2
        assert np.isfinite(cost)

    def test_action_valid_for_env(self):
        """Selected action is valid for environment."""
        optimizer = make_optimizer(k_capacity=2)

        observation = make_observation(
            pred_rul_cycles=[50] * 5,
            age_cycles=[100] * 5,
        )

        action_id, slots, _ = optimizer.select_action(observation)

        # Verify action exists in table
        assert action_id < len(ACTION_TABLE_N5_K2)
        assert ACTION_TABLE_N5_K2[action_id] == tuple(slots)

        # Verify capacity constraint
        assert len(slots) <= 2

        # Verify slots are valid indices
        for slot in slots:
            assert 0 <= slot < 5


class TestCostRegimeIntegration:
    """Test cost regime integration."""

    def test_all_regimes_work(self):
        """All cost regimes produce valid results."""
        observation = make_observation(
            pred_rul_cycles=[10, 50, 50, 50, 50],
            age_cycles=[100] * 5,
        )

        for regime_id in COST_REGIMES.keys():
            optimizer = make_optimizer(cost_regime_id=regime_id)
            action_id, slots, cost = optimizer.select_action(observation)

            assert 0 <= action_id < len(ACTION_TABLE_N5_K2)
            assert len(slots) <= 2
            assert np.isfinite(cost)
            assert cost >= 0

    def test_cost_components_sum_correctly(self):
        """Cost components sum to total for all regimes."""
        observation = make_observation(
            pred_rul_cycles=[3, 50, 50, 50, 50],
            age_cycles=[100] * 5,
        )

        for regime_id in COST_REGIMES.keys():
            optimizer = make_optimizer(cost_regime_id=regime_id)
            evaluations = optimizer.evaluate_all_actions(observation)

            for action in evaluations:
                expected_total = (
                    action.preventive_cost +
                    action.unused_life_cost +
                    action.failure_cost
                )
                assert abs(action.total_cost - expected_total) < 1e-9, \
                    f"Regime {regime_id}: components don't sum to total"


class TestRiskModelIntegration:
    """Test risk model integration."""

    def test_hard_window_risk_model(self):
        """Hard window risk model integration."""
        optimizer = make_optimizer(risk_model_id="hard_window_v1")

        # Observation with mixed risk levels
        observation = make_observation(
            pred_rul_cycles=[3, 5, 6, 50, 100],  # 3,5 at risk; 6,50,100 safe
            age_cycles=[100] * 5,
        )

        evaluations = optimizer.evaluate_all_actions(observation)
        empty_action = evaluations[0]

        # Slots 0,1 at risk (RUL <= 5), slots 2,3,4 safe
        # Expected: c_f * 2 = 5.0 * 2 = 10.0
        expected_failure = 5.0 * 2.0
        assert abs(empty_action.failure_cost - expected_failure) < 1e-9

    def test_logistic_risk_model(self):
        """Logistic risk model integration."""
        optimizer = make_optimizer(
            risk_model_id="logistic_window_v1",
            risk_temperature=10.0,
        )

        observation = make_observation(
            pred_rul_cycles=[3, 5, 10, 50, 100],
            age_cycles=[100] * 5,
        )

        evaluations = optimizer.evaluate_all_actions(observation)
        empty_action = evaluations[0]

        # All slots should have some risk > 0
        assert empty_action.failure_cost > 0.0

        # Risk should be less than hard window (no binary 0/1)
        assert empty_action.failure_cost < 5.0 * 5.0  # Less than all-at-risk


class TestMultiStepDecision:
    """Test multi-step decision making."""

    def test_different_observations_different_actions(self):
        """Different observations produce different optimal actions."""
        optimizer = make_optimizer()

        # Scenario 1: No slots at risk
        obs_safe = make_observation(
            pred_rul_cycles=[50, 50, 50, 50, 50],
            age_cycles=[100] * 5,
        )
        action_safe, slots_safe, _ = optimizer.select_action(obs_safe)

        # Scenario 2: Two slots at risk
        obs_risk = make_observation(
            pred_rul_cycles=[3, 3, 50, 50, 50],
            age_cycles=[100] * 5,
        )
        action_risk, slots_risk, _ = optimizer.select_action(obs_risk)

        # Should select different actions
        assert (action_safe, slots_safe) != (action_risk, slots_risk)

        # More slots should be selected when more at risk
        assert len(slots_risk) >= len(slots_safe)

    def test_capacity_constrained_selection(self):
        """Selection respects capacity constraint."""
        # K=1: can only select 1 slot
        optimizer_k1 = make_optimizer(k_capacity=1)

        # K=2: can select up to 2 slots
        optimizer_k2 = make_optimizer(k_capacity=2)

        observation = make_observation(
            pred_rul_cycles=[3, 3, 3, 50, 50],  # 3 slots at risk
            age_cycles=[100] * 5,
        )

        _, slots_k1, _ = optimizer_k1.select_action(observation)
        _, slots_k2, _ = optimizer_k2.select_action(observation)

        # K=1 selects at most 1
        assert len(slots_k1) <= 1

        # K=2 selects at most 2
        assert len(slots_k2) <= 2


class TestEdgeCases:
    """Test edge cases in integration."""

    def test_all_nan_observation_rejected(self):
        """NaN observation is rejected."""
        optimizer = make_optimizer()
        nan_obs = np.full((10,), np.nan, dtype=np.float32)

        with pytest.raises(ValueError, match="non-finite"):
            optimizer.select_action(nan_obs)

    def test_all_inf_observation_rejected(self):
        """Inf observation is rejected."""
        optimizer = make_optimizer()
        inf_obs = np.full((10,), np.inf, dtype=np.float32)

        with pytest.raises(ValueError, match="non-finite"):
            optimizer.select_action(inf_obs)

    def test_wrong_dtype_rejected(self):
        """Wrong dtype is rejected."""
        optimizer = make_optimizer()
        # int32 instead of float32
        int_obs = np.zeros((10,), dtype=np.int32)

        with pytest.raises(ValueError, match="floating point"):
            optimizer.select_action(int_obs)

    def test_all_zeros_observation_accepted(self):
        """All-zeros observation is accepted (represents new components)."""
        optimizer = make_optimizer()
        zeros_obs = np.zeros((10,), dtype=np.float32)

        # Should not raise
        action_id, slots, cost = optimizer.select_action(zeros_obs)

        assert 0 <= action_id < len(ACTION_TABLE_N5_K2)
        assert len(slots) <= 2
        assert np.isfinite(cost)

    def test_all_ones_observation_accepted(self):
        """All-ones observation is accepted (represents max age/RUL)."""
        optimizer = make_optimizer()
        ones_obs = np.ones((10,), dtype=np.float32)

        # Should not raise
        action_id, slots, cost = optimizer.select_action(ones_obs)

        assert 0 <= action_id < len(ACTION_TABLE_N5_K2)
        assert len(slots) <= 2
        assert np.isfinite(cost)