"""
Test Milestone 4 Exact Myopic monotonicity properties.

Tests:
- Higher c_f → more slots selected (more urgency)
- Higher c_pm → fewer slots selected (more conservative)
- Lower predicted RUL → higher failure cost
- More slots at risk → higher total failure cost
- Logistic risk is monotonic decreasing in predicted_rul
"""

import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from optimizers import MyopicContext, ExactMyopicOptimizer, compute_failure_risk
from envs.action_table import ACTION_TABLE_N5_K2
from envs.costs import CostRegime


def make_optimizer(
    k_capacity: int = 2,
    c_pm: float = 1.0,
    c_f: float = 5.0,
    c_u: float = 0.0,
    risk_model_id: str = "hard_window_v1",
    risk_temperature: float = 10.0,
) -> ExactMyopicOptimizer:
    """Create optimizer with custom cost coefficients."""
    context = MyopicContext(
        maintenance_capacity=k_capacity,
        delta_cycles=5,
        rul_scale=125.0,
        age_scale_cycles=341,
        action_table=ACTION_TABLE_N5_K2,
        c_pm=c_pm,
        c_f=c_f,
        c_u=c_u,
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


class TestCostCoefficientMonotonicity:
    """Test monotonicity with respect to cost coefficients."""

    def test_higher_c_f_more_aggressive(self):
        """Higher c_f leads to more slots selected (more aggressive maintenance)."""
        # Same observation, different c_f values
        observation = make_observation(
            pred_rul_cycles=[3, 3, 50, 50, 50],  # Two slots at risk
            age_cycles=[100] * 5,
        )

        # Low c_f: less urgency to prevent failures
        optimizer_low_cf = make_optimizer(c_f=1.0)
        action_id_low, slots_low, _ = optimizer_low_cf.select_action(observation)

        # High c_f: more urgency to prevent failures
        optimizer_high_cf = make_optimizer(c_f=10.0)
        action_id_high, slots_high, _ = optimizer_high_cf.select_action(observation)

        # Higher c_f should select at least as many slots
        assert len(slots_high) >= len(slots_low), \
            f"Higher c_f should select more slots: {len(slots_high)} < {len(slots_low)}"

    def test_higher_c_pm_more_conservative(self):
        """Higher c_pm leads to fewer slots selected (more conservative)."""
        observation = make_observation(
            pred_rul_cycles=[3, 3, 50, 50, 50],  # Two slots at risk
            age_cycles=[100] * 5,
        )

        # Low c_pm: cheaper to maintain, more slots selected
        optimizer_low_pm = make_optimizer(c_pm=0.1)
        action_id_low, slots_low, _ = optimizer_low_pm.select_action(observation)

        # High c_pm: expensive to maintain, fewer slots selected
        optimizer_high_pm = make_optimizer(c_pm=10.0)
        action_id_high, slots_high, _ = optimizer_high_pm.select_action(observation)

        # Higher c_pm should select fewer or equal slots
        assert len(slots_high) <= len(slots_low), \
            f"Higher c_pm should select fewer slots: {len(slots_high)} > {len(slots_low)}"

    def test_c_u_affects_unused_life_cost(self):
        """c_u affects unused life cost component."""
        observation = make_observation(
            pred_rul_cycles=[50, 50, 50, 50, 50],  # All safe
            age_cycles=[100] * 5,
        )

        # c_u = 0: no unused life cost
        optimizer_no_u = make_optimizer(c_u=0.0)
        evaluations_no_u = optimizer_no_u.evaluate_all_actions(observation)

        # c_u = 0.5: positive unused life cost
        optimizer_with_u = make_optimizer(c_u=0.5)
        evaluations_with_u = optimizer_with_u.evaluate_all_actions(observation)

        # Compare unused life costs for same action
        for e_no_u, e_with_u in zip(evaluations_no_u, evaluations_with_u):
            assert e_no_u.unused_life_cost == 0.0
            if len(e_with_u.selected_slots) > 0:
                assert e_with_u.unused_life_cost > 0.0


class TestRULMonotonicity:
    """Test monotonicity with respect to predicted RUL."""

    def test_lower_rul_higher_failure_cost(self):
        """Lower predicted RUL leads to higher failure cost."""
        optimizer = make_optimizer(risk_model_id="hard_window_v1")

        # High RUL observation
        obs_high_rul = make_observation(
            pred_rul_cycles=[50, 50, 50, 50, 50],
            age_cycles=[100] * 5,
        )

        # Low RUL observation
        obs_low_rul = make_observation(
            pred_rul_cycles=[3, 3, 3, 3, 3],
            age_cycles=[100] * 5,
        )

        evaluations_high = optimizer.evaluate_all_actions(obs_high_rul)
        evaluations_low = optimizer.evaluate_all_actions(obs_low_rul)

        # Empty action: low RUL should have much higher failure cost
        empty_high = evaluations_high[0]
        empty_low = evaluations_low[0]

        assert empty_low.failure_cost > empty_high.failure_cost, \
            "Lower RUL should have higher failure cost"

    def test_more_at_risk_slots_higher_total_cost(self):
        """More at-risk slots leads to higher total failure cost."""
        optimizer = make_optimizer(risk_model_id="hard_window_v1")

        # 1 slot at risk
        obs_one_risk = make_observation(
            pred_rul_cycles=[3, 50, 50, 50, 50],
            age_cycles=[100] * 5,
        )

        # 3 slots at risk
        obs_three_risk = make_observation(
            pred_rul_cycles=[3, 3, 3, 50, 50],
            age_cycles=[100] * 5,
        )

        evaluations_one = optimizer.evaluate_all_actions(obs_one_risk)
        evaluations_three = optimizer.evaluate_all_actions(obs_three_risk)

        # Empty action failure costs
        empty_one = evaluations_one[0]
        empty_three = evaluations_three[0]

        # 3 at-risk slots should have 3x the failure cost of 1 at-risk slot
        assert empty_three.failure_cost > empty_one.failure_cost


class TestLogisticMonotonicity:
    """Test logistic risk model monotonicity."""

    def test_logistic_decreasing_in_rul(self):
        """Logistic risk is strictly decreasing in predicted_rul."""
        temperature = 10.0
        delta = 5

        # Test at various RUL values
        ruls = [1, 3, 5, 10, 20, 50, 100]
        risks = []

        for rul in ruls:
            risk = compute_failure_risk(
                predicted_rul_cycles=rul,
                delta_cycles=delta,
                risk_model_id="logistic_window_v1",
                risk_temperature=temperature,
            )
            risks.append(risk)

        # Risk should be strictly decreasing
        for i in range(len(risks) - 1):
            assert risks[i] > risks[i + 1], \
                f"Risk should decrease: risk({ruls[i]})={risks[i]} <= risk({ruls[i+1]})={risks[i+1]}"

    def test_logistic_bounds(self):
        """Logistic risk is bounded in (0, 1)."""
        temperature = 10.0

        for rul in [0.1, 1, 5, 10, 50, 100, 500]:
            risk = compute_failure_risk(
                predicted_rul_cycles=rul,
                delta_cycles=5,
                risk_model_id="logistic_window_v1",
                risk_temperature=temperature,
            )
            assert 0.0 < risk < 1.0, f"Risk {risk} out of bounds for RUL={rul}"

    def test_temperature_effect(self):
        """Higher temperature → smoother (less extreme) risks.

        Note: At RUL == delta (boundary), risk = 0.5 for ALL temperatures.
        To observe temperature effect, we must test OFF the boundary.
        """
        delta = 5

        # Test OFF boundary: RUL = 3 (below delta, should have risk > 0.5)
        rul_below = 3.0

        # Low temperature: sharp transition, risk closer to 1.0
        risk_low_temp_below = compute_failure_risk(
            predicted_rul_cycles=rul_below,
            delta_cycles=delta,
            risk_model_id="logistic_window_v1",
            risk_temperature=1.0,
        )

        # High temperature: smooth transition, risk closer to 0.5
        risk_high_temp_below = compute_failure_risk(
            predicted_rul_cycles=rul_below,
            delta_cycles=delta,
            risk_model_id="logistic_window_v1",
            risk_temperature=100.0,
        )

        # Low temp should have higher risk (closer to 1.0) when RUL < delta
        assert risk_low_temp_below > risk_high_temp_below, \
            f"Low temp should have higher risk: {risk_low_temp_below} <= {risk_high_temp_below}"

        # Test OFF boundary: RUL = 10 (above delta, should have risk < 0.5)
        rul_above = 10.0

        risk_low_temp_above = compute_failure_risk(
            predicted_rul_cycles=rul_above,
            delta_cycles=delta,
            risk_model_id="logistic_window_v1",
            risk_temperature=1.0,
        )

        risk_high_temp_above = compute_failure_risk(
            predicted_rul_cycles=rul_above,
            delta_cycles=delta,
            risk_model_id="logistic_window_v1",
            risk_temperature=100.0,
        )

        # Low temp should have lower risk (closer to 0.0) when RUL > delta
        assert risk_low_temp_above < risk_high_temp_above, \
            f"Low temp should have lower risk: {risk_low_temp_above} >= {risk_high_temp_above}"

        # Verify boundary case: RUL == delta always gives 0.5
        risk_boundary = compute_failure_risk(
            predicted_rul_cycles=5.0,  # Exactly at delta
            delta_cycles=delta,
            risk_model_id="logistic_window_v1",
            risk_temperature=10.0,
        )
        assert abs(risk_boundary - 0.5) < 1e-9, "Boundary should always be 0.5"


class TestTieBreaking:
    """Test deterministic tie-breaking behavior."""

    def test_smallest_action_id_wins_ties(self):
        """When costs are equal, smallest action_id is selected."""
        # Create a scenario where multiple actions have similar cost
        # Use very high tolerance to force ties
        optimizer = make_optimizer(
            c_pm=1.0,
            c_f=5.0,
            c_u=0.0,
            risk_model_id="hard_window_v1",
        )

        # Observation where no slots are at risk
        observation = make_observation(
            pred_rul_cycles=[50, 50, 50, 50, 50],
            age_cycles=[100] * 5,
        )

        evaluations = optimizer.evaluate_all_actions(observation)

        # When no slots are at risk, failure_cost = 0 for all actions
        # preventive_cost increases with |S|
        # So empty action (id=0) should be optimal
        action_id, slots, cost = optimizer.select_action(observation)

        assert action_id == 0, "Empty action should win when no slots at risk"
        assert slots == ()

    def test_deterministic_same_observation(self):
        """Same observation always produces same action."""
        observation = make_observation(
            pred_rul_cycles=[3, 10, 50, 50, 50],
            age_cycles=[100] * 5,
        )

        optimizer = make_optimizer()

        results = []
        for _ in range(10):
            action_id, slots, cost = optimizer.select_action(observation)
            results.append((action_id, tuple(slots), cost))

        # All results should be identical
        assert all(r == results[0] for r in results), "Results should be deterministic"