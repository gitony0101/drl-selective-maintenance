"""
Test Milestone 4 Exact Myopic reproducibility.

Tests:
- Same observation produces same action selection
- Same observation produces same cost evaluations
- Results are deterministic across multiple runs
- Random seed affects only synthetic test data, not optimizer logic
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


class TestDeterministicSelection:
    """Test deterministic action selection."""

    def test_same_observation_same_action(self):
        """Same observation always produces same action."""
        optimizer = make_optimizer()
        observation = make_observation(
            pred_rul_cycles=[3, 10, 50, 50, 50],
            age_cycles=[100] * 5,
        )

        # Run multiple times
        results = []
        for _ in range(100):
            action_id, slots, cost = optimizer.select_action(observation)
            results.append((action_id, tuple(slots)))

        # All results should be identical
        unique_results = set(results)
        assert len(unique_results) == 1, f"Expected 1 unique result, got {len(unique_results)}"

    def test_same_observation_same_cost(self):
        """Same observation always produces same cost."""
        optimizer = make_optimizer()
        observation = make_observation(
            pred_rul_cycles=[3, 10, 50, 50, 50],
            age_cycles=[100] * 5,
        )

        # Run multiple times
        costs = []
        for _ in range(100):
            _, _, cost = optimizer.select_action(observation)
            costs.append(cost)

        # All costs should be identical
        assert len(set(costs)) == 1, f"Expected 1 unique cost, got {len(set(costs))}"


class TestDeterministicEvaluation:
    """Test deterministic cost evaluation."""

    def test_same_observation_same_evaluations(self):
        """Same observation produces same cost evaluations."""
        optimizer = make_optimizer()
        observation = make_observation(
            pred_rul_cycles=[3, 10, 50, 50, 50],
            age_cycles=[100] * 5,
        )

        # Get evaluations multiple times
        all_evaluations = []
        for _ in range(10):
            evaluations = optimizer.evaluate_all_actions(observation)
            all_evaluations.append(evaluations)

        # All evaluations should be identical
        for evaluations in all_evaluations[1:]:
            for i, (e0, ei) in enumerate(zip(all_evaluations[0], evaluations)):
                assert e0.action_id == ei.action_id
                assert e0.selected_slots == ei.selected_slots
                assert abs(e0.total_cost - ei.total_cost) < 1e-9
                assert abs(e0.preventive_cost - ei.preventive_cost) < 1e-9
                assert abs(e0.unused_life_cost - ei.unused_life_cost) < 1e-9
                assert abs(e0.failure_cost - ei.failure_cost) < 1e-9


class TestReproducibilityAcrossSeeds:
    """Test that optimizer itself is not affected by random seeds."""

    def test_optimizer_independent_of_numpy_seed(self):
        """Optimizer results don't depend on numpy random seed."""
        observation = make_observation(
            pred_rul_cycles=[3, 10, 50, 50, 50],
            age_cycles=[100] * 5,
        )

        results = []
        for seed in [42, 123, 456, 789, 6521]:
            np.random.seed(seed)
            optimizer = make_optimizer()
            action_id, slots, cost = optimizer.select_action(observation)
            results.append((action_id, tuple(slots), cost))

        # All results should be identical regardless of seed
        assert all(r == results[0] for r in results), \
            "Optimizer should be independent of random seed"


class TestK1Reproducibility:
    """Test reproducibility for K=1 capacity."""

    def test_k1_deterministic(self):
        """K=1 optimizer is deterministic."""
        optimizer = make_optimizer(k_capacity=1)
        observation = make_observation(
            pred_rul_cycles=[3, 10, 50, 50, 50],
            age_cycles=[100] * 5,
        )

        results = []
        for _ in range(50):
            action_id, slots, cost = optimizer.select_action(observation)
            results.append((action_id, tuple(slots), cost))

        assert all(r == results[0] for r in results)


class TestAllCostRegimesReproducible:
    """Test reproducibility across all cost regimes."""

    def test_all_regimes_deterministic(self):
        """All cost regimes produce deterministic results."""
        observation = make_observation(
            pred_rul_cycles=[3, 10, 50, 50, 50],
            age_cycles=[100] * 5,
        )

        for regime_id in COST_REGIMES.keys():
            optimizer = make_optimizer(cost_regime_id=regime_id)

            results = []
            for _ in range(20):
                action_id, slots, cost = optimizer.select_action(observation)
                results.append((action_id, tuple(slots), cost))

            assert all(r == results[0] for r in results), \
                f"Regime {regime_id} should be deterministic"


class TestEdgeCaseReproducibility:
    """Test reproducibility for edge cases."""

    def test_all_slots_safe(self):
        """Reproducible when all slots are safe."""
        optimizer = make_optimizer()
        observation = make_observation(
            pred_rul_cycles=[100, 100, 100, 100, 100],  # All safe
            age_cycles=[100] * 5,
        )

        results = []
        for _ in range(50):
            action_id, slots, cost = optimizer.select_action(observation)
            results.append((action_id, tuple(slots), cost))

        assert all(r == results[0] for r in results)

    def test_all_slots_at_risk(self):
        """Reproducible when all slots are at risk."""
        optimizer = make_optimizer()
        observation = make_observation(
            pred_rul_cycles=[1, 1, 1, 1, 1],  # All at risk
            age_cycles=[100] * 5,
        )

        results = []
        for _ in range(50):
            action_id, slots, cost = optimizer.select_action(observation)
            results.append((action_id, tuple(slots), cost))

        assert all(r == results[0] for r in results)

    def test_empty_action_optimal(self):
        """Reproducible when empty action is optimal."""
        optimizer = make_optimizer()
        observation = make_observation(
            pred_rul_cycles=[50, 50, 50, 50, 50],  # All safe, no benefit from PM
            age_cycles=[100] * 5,
        )

        results = []
        for _ in range(50):
            action_id, slots, cost = optimizer.select_action(observation)
            results.append((action_id, tuple(slots), cost))

        assert all(r == results[0] for r in results)
        # Empty action should be optimal when no slots at risk
        assert results[0][0] == 0
        assert results[0][1] == ()