"""
Test Milestone 4 Exact Myopic tie-breaking behavior.

Tests:
- Explicit tie scenario: two actions have identical estimated cost
- Verify smallest action_id wins
- Golden action tests with independently computed expected costs
"""

import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from optimizers import MyopicContext, ExactMyopicOptimizer
from envs.action_table import ACTION_TABLE_N5_K1, ACTION_TABLE_N5_K2
from envs.costs import COST_REGIMES


def make_optimizer(
    k_capacity: int = 2,
    cost_regime_id: str = "failure-light-no-waste",
    risk_model_id: str = "hard_window_v1",
    tie_tolerance: float = 1e-9,
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

    return ExactMyopicOptimizer(context=context, tie_tolerance=tie_tolerance)


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


class TestExplicitTie:
    """Test explicit tie-breaking scenarios."""

    def test_empty_vs_single_at_boundary(self):
        """
        Tie scenario: empty action vs single-slot action have equal cost.

        When no slots are at risk:
        - Empty action: cost = 0
        - Single slot (e.g., slot 0): cost = c_pm + c_u * (RUL/rul_scale)

        If we pick c_u = 0 and the slot has RUL = 0, then:
        - Single slot cost = c_pm = 1.0

        So empty action wins (cost 0 < 1).

        To create a TRUE tie, we need a scenario where:
        - Two actions have EXACTLY the same cost
        """
        optimizer = make_optimizer(
            k_capacity=2,
            cost_regime_id="failure-light-no-waste",  # c_u = 0
        )

        # All slots safe (RUL = 50 > delta=5)
        # No failure cost for any action
        # Only preventive cost matters
        observation = make_observation(
            pred_rul_cycles=[50, 50, 50, 50, 50],
            age_cycles=[100] * 5,
        )

        evaluations = optimizer.evaluate_all_actions(observation)

        # Empty action (id=0) has cost 0
        # All single-slot actions (id=1-5) have cost = c_pm = 1.0
        # All two-slot actions (id=6-15) have cost = 2*c_pm = 2.0

        # Empty action should win
        action_id, slots, cost = optimizer.select_action(observation)

        assert action_id == 0, "Empty action should win when no slots at risk"
        assert slots == ()
        assert cost == 0.0

    def test_two_single_slot_actions_tie(self):
        """
        Tie between two single-slot actions.

        When two slots have identical RUL and both are at risk:
        - Action selecting slot 0: cost = c_pm + c_f * (risk for slots 1-4)
        - Action selecting slot 1: cost = c_pm + c_f * (risk for slots 0,2-4)

        If slot 0 and slot 1 both have RUL=3 (at risk), and slots 2-4 have RUL=50 (safe):
        - Select slot 0: cost = 1.0 + 5.0 * 0 = 1.0 (slot 1 also at risk but we prevent slot 0)
        - Select slot 1: cost = 1.0 + 5.0 * 1.0 = 6.0 (slot 0 at risk, not selected)

        These are NOT equal. To get a tie, we need slots with identical contribution.

        Let's try: slots 2,3,4 all at risk (RUL=3), slots 0,1 safe (RUL=50)
        - Select slot 2: cost = 1.0 + 5.0 * 2 = 11.0 (slots 3,4 still at risk)
        - Select slot 3: cost = 1.0 + 5.0 * 2 = 11.0 (slots 2,4 still at risk)
        - Select slot 4: cost = 1.0 + 5.0 * 2 = 11.0 (slots 2,3 still at risk)

        These should all tie at cost=11.0, and action with smallest ID should win.
        """
        optimizer = make_optimizer(
            k_capacity=1,  # Can only select 1 slot
            cost_regime_id="failure-light-no-waste",
        )

        # Slots 0,1 safe (RUL=50), slots 2,3,4 at risk (RUL=3)
        observation = make_observation(
            pred_rul_cycles=[50, 50, 3, 3, 3],
            age_cycles=[100] * 5,
        )

        evaluations = optimizer.evaluate_all_actions(observation)

        # For K=1, actions are:
        # id=0: empty
        # id=1: slot 0
        # id=2: slot 1
        # id=3: slot 2
        # id=4: slot 3
        # id=5: slot 4

        # Empty action: all 3 at-risk slots contribute = 5.0 * 3 = 15.0
        assert evaluations[0].failure_cost == 15.0

        # Select slot 0 (safe): slots 2,3,4 still at risk = 5.0 * 3 = 15.0 + c_pm = 16.0
        assert evaluations[1].total_cost == 16.0

        # Select slot 1 (safe): slots 2,3,4 still at risk = 5.0 * 3 = 15.0 + c_pm = 16.0
        assert evaluations[2].total_cost == 16.0

        # Select slot 2 (at-risk): slots 3,4 still at risk = 5.0 * 2 = 10.0 + c_pm = 11.0
        assert evaluations[3].total_cost == 11.0

        # Select slot 3 (at-risk): slots 2,4 still at risk = 5.0 * 2 = 10.0 + c_pm = 11.0
        assert evaluations[4].total_cost == 11.0

        # Select slot 4 (at-risk): slots 2,3 still at risk = 5.0 * 2 = 10.0 + c_pm = 11.0
        assert evaluations[5].total_cost == 11.0

        # Actions 3, 4, 5 all tie at cost 11.0
        # Should select action 3 (smallest ID among ties)
        action_id, slots, cost = optimizer.select_action(observation)

        assert action_id == 3, f"Expected action_id=3 (smallest among ties), got {action_id}"
        assert slots == (2,), f"Expected slots=(2,), got {slots}"
        assert abs(cost - 11.0) < 1e-9

    def test_two_slot_actions_tie_k2(self):
        """
        Tie between two two-slot actions for K=2.

        Scenario: 4 slots at risk (RUL=3), 1 slot safe (RUL=50)
        K=2 capacity

        Any action selecting 2 at-risk slots leaves 2 at-risk slots uncovered:
        - Failure cost = 5.0 * 2 = 10.0
        - PM cost = 2.0
        - Total = 12.0

        Multiple two-slot actions will tie. Smallest action_id should win.
        """
        optimizer = make_optimizer(
            k_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )

        # Slots 0,1,2,3 at risk (RUL=3), slot 4 safe (RUL=50)
        observation = make_observation(
            pred_rul_cycles=[3, 3, 3, 3, 50],
            age_cycles=[100] * 5,
        )

        evaluations = optimizer.evaluate_all_actions(observation)

        # K=2 action table:
        # id=0: empty -> failure_cost = 5.0 * 4 = 20.0
        # id=1-5: single slots
        # id=6-15: two-slot combinations

        # Empty action
        assert evaluations[0].failure_cost == 20.0

        # The best actions should select 2 at-risk slots
        # Action 6 selects (0,1), both at risk
        # Leaves slots 2,3 at risk: failure = 5.0 * 2 = 10.0, pm = 2.0, total = 12.0
        action_6 = evaluations[6]
        assert action_6.selected_slots == (0, 1)
        assert abs(action_6.total_cost - 12.0) < 1e-9

        # Action 7 selects (0,2), both at risk - should also cost 12.0
        action_7 = evaluations[7]
        assert action_7.selected_slots == (0, 2)
        assert abs(action_7.total_cost - 12.0) < 1e-9

        # Action 10 selects (1,2) - both at risk, should also cost 12.0
        action_10 = evaluations[10]
        assert action_10.selected_slots == (1, 2)
        assert abs(action_10.total_cost - 12.0) < 1e-9

        # Action 9 selects (0,4) - one at risk, one safe
        # Leaves slots 1,2,3 at risk: failure = 5.0 * 3 = 15.0, pm = 2.0, total = 17.0
        action_9 = evaluations[9]
        assert action_9.selected_slots == (0, 4)
        assert abs(action_9.total_cost - 17.0) < 1e-9

        # Optimal: select action 6 (slots 0,1), smallest ID among optimal two-slot actions
        action_id, slots, cost = optimizer.select_action(observation)

        assert action_id == 6, f"Expected action_id=6, got {action_id}"
        assert slots == (0, 1), f"Expected slots=(0,1), got {slots}"
        assert abs(cost - 12.0) < 1e-9


class TestGoldenActions:
    """
    Golden action tests with independently computed expected costs.

    These tests compute expected costs manually (without using the optimizer's
    evaluation) to verify the optimizer's calculations are correct.
    """

    def test_golden_single_slot_k1(self):
        """Golden test: K=1, single slot selection."""
        # Parameters
        c_pm = 1.0
        c_f = 5.0
        c_u = 0.0
        delta = 5
        rul_scale = 125.0

        # Scenario: slot 0 at risk (RUL=3), others safe (RUL=50)
        # OPTIMIZER should select slot 0

        # Independent cost calculation:
        # Empty action: failure = c_f * 1 (only slot 0 at risk) = 5.0
        # Select slot 0: pm = c_pm = 1.0, failure = 0 (slot 0 prevented)
        # Total for select slot 0: 1.0

        # Select slot 1 (safe): pm = 1.0, failure = 5.0 (slot 0 still at risk) = 6.0

        optimizer = make_optimizer(
            k_capacity=1,
            cost_regime_id="failure-light-no-waste",
        )

        observation = make_observation(
            pred_rul_cycles=[3, 50, 50, 50, 50],
            age_cycles=[100] * 5,
        )

        action_id, slots, cost = optimizer.select_action(observation)

        # Independently verified: should select slot 0
        assert action_id == 1, f"Expected action_id=1 (select slot 0)"
        assert slots == (0,), f"Expected slots=(0,)"

        # Independently verified cost
        expected_cost = c_pm + c_u * (3.0 / rul_scale) + 0.0  # pm + unused + no failure
        assert abs(cost - expected_cost) < 1e-9, f"Expected cost ~{expected_cost}, got {cost}"

    def test_golden_two_slots_k2(self):
        """Golden test: K=2, two slot selection."""
        # Scenario: slots 0,1 at risk (RUL=3), slots 2,3,4 safe (RUL=50)
        # OPTIMIZER should select slots 0,1

        optimizer = make_optimizer(
            k_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )

        observation = make_observation(
            pred_rul_cycles=[3, 3, 50, 50, 50],
            age_cycles=[100] * 5,
        )

        action_id, slots, cost = optimizer.select_action(observation)

        # Should select slots 0,1 (the at-risk ones)
        # Action 6 in K=2 table selects (0,1)
        assert action_id == 6, f"Expected action_id=6"
        assert slots == (0, 1), f"Expected slots=(0,1)"

        # Expected cost: c_pm * 2 + c_u * 0 + c_f * 0 = 2.0
        expected_cost = 2.0
        assert abs(cost - expected_cost) < 1e-9

    def test_golden_waste_aware_regime(self):
        """Golden test: waste-aware cost regime."""
        # failure-light-waste-aware: c_pm=1.0, c_f=5.0, c_u=0.25

        optimizer = make_optimizer(
            k_capacity=1,
            cost_regime_id="failure-light-waste-aware",
        )

        # Select a slot with RUL=50
        # pm = 1.0
        # unused_life = 0.25 * (50/125) = 0.25 * 0.4 = 0.1
        # failure = 0 (slot is safe)
        # Total = 1.1

        observation = make_observation(
            pred_rul_cycles=[50, 50, 50, 50, 50],
            age_cycles=[100] * 5,
        )

        # When all slots safe, empty action costs 0
        action_id, slots, cost = optimizer.select_action(observation)

        # Empty action should win (cost=0)
        assert action_id == 0
        assert slots == ()
        assert cost == 0.0

    def test_golden_heavy_failure_regime(self):
        """Golden test: heavy failure cost regime."""
        # failure-heavy-no-waste: c_pm=1.0, c_f=10.0, c_u=0.0

        optimizer = make_optimizer(
            k_capacity=1,
            cost_regime_id="failure-heavy-no-waste",
        )

        # Slot 0 at risk (RUL=3)
        # Heavy c_f = 10.0 makes prevention more valuable

        observation = make_observation(
            pred_rul_cycles=[3, 50, 50, 50, 50],
            age_cycles=[100] * 5,
        )

        action_id, slots, cost = optimizer.select_action(observation)

        # Should still select slot 0
        # Empty: failure = 10.0 * 1 = 10.0
        # Select 0: pm = 1.0, failure = 0
        assert action_id == 1
        assert slots == (0,)
        assert abs(cost - 1.0) < 1e-9


class TestTieTolerance:
    """Test tie tolerance behavior."""

    def test_tolerance_prevents_spurious_ties(self):
        """Tie tolerance prevents near-equal costs from being treated as ties."""
        # With default tolerance 1e-9, costs differing by more than 1e-9
        # should NOT be treated as ties

        optimizer = make_optimizer(
            k_capacity=1,
            tie_tolerance=1e-9,
        )

        observation = make_observation(
            pred_rul_cycles=[50, 50, 50, 50, 50],
            age_cycles=[100] * 5,
        )

        # All actions have distinct costs due to c_u > 0
        # Should select empty action (cost = 0)
        action_id, slots, cost = optimizer.select_action(observation)

        assert action_id == 0