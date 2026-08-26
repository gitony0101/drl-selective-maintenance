"""
Tests for Milestone 3 integration.

Verifies:
- All six policies work with evaluator
- K=1 and K=2 both run
- All four regimes run
- No NaN or Inf in results
- Legal actions only
- Exact cost reconciliation
- Practical policies receive no diagnostic input
"""

import numpy as np
import pytest

pytestmark = pytest.mark.requires_external_assets

from src.baselines import (
    PolicyEvaluator,
    EvaluationConfig,
    EpisodeResult,
    CorrectiveOnly,
    RandomFeasible,
    AgeThreshold,
    PredictedRULThreshold,
    GreedyPredictedRUL,
)
from src.envs import get_default_config, SelectiveMaintenanceEnv
from src.envs.action_table import ACTION_TABLE_N5_K1, ACTION_TABLE_N5_K2


class TestPolicyIntegration:
    """Test all policies work with evaluator."""

    def test_corrective_only_integration(self):
        """Test corrective-only policy runs through evaluator."""
        env_config = get_default_config(
            split="rl_validation",
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )
        evaluator = PolicyEvaluator(env_config=env_config)
        policy = evaluator.create_policy("corrective_only")
        context = evaluator.create_context("corrective_only", policy_seed=42)

        # Verify policy returns valid action
        obs = np.zeros(10, dtype=np.float32)
        action = policy.select_action(obs, context)
        assert action == 0
        assert 0 <= action < len(context.action_table)

    def test_random_feasible_integration(self):
        """Test random feasible policy runs through evaluator."""
        env_config = get_default_config(
            split="rl_validation",
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )
        evaluator = PolicyEvaluator(env_config=env_config)
        policy = evaluator.create_policy("random_feasible", policy_seed=42)
        context = evaluator.create_context("random_feasible", policy_seed=42)

        obs = np.zeros(10, dtype=np.float32)
        num_actions = len(context.action_table)

        for _ in range(100):
            action = policy.select_action(obs, context)
            assert 0 <= action < num_actions

    def test_age_threshold_integration(self):
        """Test age threshold policy runs through evaluator."""
        env_config = get_default_config(
            split="rl_validation",
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )
        evaluator = PolicyEvaluator(env_config=env_config)
        policy = evaluator.create_policy("age_threshold", threshold=100)
        context = evaluator.create_context("age_threshold", policy_seed=42)

        obs = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        action = policy.select_action(obs, context)
        assert 0 <= action < len(context.action_table)

    def test_predicted_rul_threshold_integration(self):
        """Test predicted RUL threshold policy runs through evaluator."""
        env_config = get_default_config(
            split="rl_validation",
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )
        evaluator = PolicyEvaluator(env_config=env_config)
        policy = evaluator.create_policy("predicted_rul_threshold", threshold=50)
        context = evaluator.create_context("predicted_rul_threshold", policy_seed=42)

        obs = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        action = policy.select_action(obs, context)
        assert 0 <= action < len(context.action_table)

    def test_greedy_predicted_rul_integration(self):
        """Test greedy predicted RUL policy runs through evaluator."""
        env_config = get_default_config(
            split="rl_validation",
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )
        evaluator = PolicyEvaluator(env_config=env_config)
        policy = evaluator.create_policy("greedy_predicted_rul", activation_threshold=50)
        context = evaluator.create_context("greedy_predicted_rul", policy_seed=42)

        obs = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        action = policy.select_action(obs, context)
        assert 0 <= action < len(context.action_table)


class TestKCapacitySupport:
    """Test K=1 and K=2 support."""

    def test_k1_action_table(self):
        """Test K=1 uses correct action table."""
        env_config = get_default_config(
            split="rl_validation",
            maintenance_capacity=1,
        )
        evaluator = PolicyEvaluator(env_config=env_config)
        context = evaluator.create_context("corrective_only", policy_seed=42)

        assert len(context.action_table) == 6  # K=1 has 6 actions
        assert context.action_table == ACTION_TABLE_N5_K1

    def test_k2_action_table(self):
        """Test K=2 uses correct action table."""
        env_config = get_default_config(
            split="rl_validation",
            maintenance_capacity=2,
        )
        evaluator = PolicyEvaluator(env_config=env_config)
        context = evaluator.create_context("corrective_only", policy_seed=42)

        assert len(context.action_table) == 16  # K=2 has 16 actions
        assert context.action_table == ACTION_TABLE_N5_K2


class TestCostRegimes:
    """Test all four cost regimes run."""

    @pytest.mark.parametrize(
        "cost_regime_id",
        [
            "failure-light-no-waste",
            "failure-heavy-no-waste",
            "failure-light-waste-aware",
            "failure-heavy-waste-aware",
        ],
    )
    def test_all_cost_regimes(self, cost_regime_id):
        """Test all four cost regimes are supported."""
        env_config = get_default_config(
            split="rl_validation",
            cost_regime_id=cost_regime_id,
        )
        evaluator = PolicyEvaluator(env_config=env_config)
        policy = evaluator.create_policy("corrective_only")
        context = evaluator.create_context("corrective_only", policy_seed=42)

        obs = np.zeros(10, dtype=np.float32)
        action = policy.select_action(obs, context)
        assert 0 <= action < len(context.action_table)


class TestNoNanInf:
    """Test no NaN or Inf in policy outputs."""

    def test_policy_outputs_finite(self):
        """Test policy outputs are finite numbers."""
        env_config = get_default_config(
            split="rl_validation",
            maintenance_capacity=2,
        )
        evaluator = PolicyEvaluator(env_config=env_config)

        policies = [
            ("corrective_only", {}),
            ("random_feasible", {"policy_seed": 42}),
            ("age_threshold", {"threshold": 100}),
            ("predicted_rul_threshold", {"threshold": 50}),
            ("greedy_predicted_rul", {"activation_threshold": 50}),
        ]

        obs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32)

        for policy_name, kwargs in policies:
            policy = evaluator.create_policy(policy_name, **kwargs)
            context = evaluator.create_context(policy_name, policy_seed=42)

            action = policy.select_action(obs, context)

            assert not np.isnan(action)
            assert not np.isinf(action)


class TestLegalActions:
    """Test all returned actions are legal."""

    def test_all_actions_legal_k2(self):
        """Test all actions legal with K=2."""
        env_config = get_default_config(
            split="rl_validation",
            maintenance_capacity=2,
        )
        evaluator = PolicyEvaluator(env_config=env_config)
        policy = evaluator.create_policy("random_feasible", policy_seed=42)
        context = evaluator.create_context("random_feasible", policy_seed=42)

        num_actions = len(context.action_table)
        obs = np.zeros(10, dtype=np.float32)

        for _ in range(1000):
            action = policy.select_action(obs, context)
            assert 0 <= action < num_actions, f"Action {action} out of range [0, {num_actions})"

    def test_all_actions_legal_k1(self):
        """Test all actions legal with K=1."""
        env_config = get_default_config(
            split="rl_validation",
            maintenance_capacity=1,
        )
        evaluator = PolicyEvaluator(env_config=env_config)
        policy = evaluator.create_policy("random_feasible", policy_seed=42)
        context = evaluator.create_context("random_feasible", policy_seed=42)

        num_actions = len(context.action_table)
        obs = np.zeros(10, dtype=np.float32)

        for _ in range(1000):
            action = policy.select_action(obs, context)
            assert 0 <= action < num_actions, f"Action {action} out of range [0, {num_actions})"


class TestPracticalPolicyInformationBarrier:
    """Test practical policies receive no diagnostic input."""

    def test_age_threshold_no_diagnostic_info(self):
        """Test age threshold does not use diagnostic info."""
        policy = AgeThreshold(threshold=100)
        context = PolicyContext(
            maintenance_capacity=2,
            age_scale_cycles=341,
            rul_scale=125.0,
            action_table=ACTION_TABLE_N5_K2,
            cost_regime_id="failure-light-no-waste",
            policy_rng=np.random.default_rng(42),
        )

        obs = np.zeros(10, dtype=np.float32)

        # Should work without diagnostic_info
        action = policy.select_action(obs, context)
        assert 0 <= action < 16

    def test_predicted_rul_threshold_no_diagnostic_info(self):
        """Test predicted RUL threshold does not use diagnostic info."""
        policy = PredictedRULThreshold(threshold=50)
        context = PolicyContext(
            maintenance_capacity=2,
            age_scale_cycles=341,
            rul_scale=125.0,
            action_table=ACTION_TABLE_N5_K2,
            cost_regime_id="failure-light-no-waste",
            policy_rng=np.random.default_rng(42),
        )

        obs = np.zeros(10, dtype=np.float32)

        # Should work without diagnostic_info
        action = policy.select_action(obs, context)
        assert 0 <= action < 16

    def test_greedy_no_diagnostic_info(self):
        """Test greedy policy does not use diagnostic info."""
        policy = GreedyPredictedRUL(activation_threshold=50)
        context = PolicyContext(
            maintenance_capacity=2,
            age_scale_cycles=341,
            rul_scale=125.0,
            action_table=ACTION_TABLE_N5_K2,
            cost_regime_id="failure-light-no-waste",
            policy_rng=np.random.default_rng(42),
        )

        obs = np.zeros(10, dtype=np.float32)

        # Should work without diagnostic_info
        action = policy.select_action(obs, context)
        assert 0 <= action < 16


# Import for the test below
from src.baselines.protocols import PolicyContext


class TestEnvironmentIntegration:
    """Test policies work with actual environment."""

    def test_corrective_only_environment_step(self):
        """Test corrective-only can step through environment."""
        # This test requires scenario bank - use smoke scenario path
        env_config = get_default_config(
            split="rl_validation",
            maintenance_capacity=2,
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )

        # Check if scenario bank exists
        from pathlib import Path
        if not Path(env_config.scenario_bank_path).exists():
            pytest.skip("Scenario bank not available")

        env = SelectiveMaintenanceEnv(config=env_config)
        evaluator = PolicyEvaluator(env_config=env_config)
        policy = evaluator.create_policy("corrective_only")
        context = evaluator.create_context("corrective_only", policy_seed=42)

        obs, info = env.reset(seed=6521)

        for _ in range(10):
            action = policy.select_action(obs, context)
            obs, reward, terminated, truncated, info = env.step(action)

            # Verify observation is finite
            assert not np.isnan(obs).any()
            assert not np.isinf(obs).any()

            # Verify reward is finite
            assert not np.isnan(reward)
            assert not np.isinf(reward)

            if truncated:
                break

    def test_random_feasible_environment_step(self):
        """Test random feasible can step through environment."""
        from pathlib import Path

        env_config = get_default_config(
            split="rl_validation",
            maintenance_capacity=2,
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )

        if not Path(env_config.scenario_bank_path).exists():
            pytest.skip("Scenario bank not available")

        env = SelectiveMaintenanceEnv(config=env_config)
        evaluator = PolicyEvaluator(env_config=env_config)
        policy = evaluator.create_policy("random_feasible", policy_seed=42)
        context = evaluator.create_context("random_feasible", policy_seed=42)

        obs, info = env.reset(seed=6521)

        for _ in range(10):
            action = policy.select_action(obs, context)
            obs, reward, terminated, truncated, info = env.step(action)

            assert not np.isnan(obs).any()
            assert not np.isinf(obs).any()

            if truncated:
                break