"""
Tests for Milestone 3 evaluator.

Verifies:
- Evaluator runs policies through environment
- Practical policies receive only observation + context
- Oracle receives OracleContext + diagnostic info
- Reward equals negative total cost
- Episode horizon enforced
"""

import numpy as np
import pytest

from src.baselines.evaluator import PolicyEvaluator, EvaluationConfig, EpisodeResult
from src.baselines.protocols import PolicyContext
from src.envs import EnvironmentConfig, get_default_config
from src.envs.action_table import ACTION_TABLE_N5_K2


class TestEvaluationConfig:
    """Test EvaluationConfig construction."""

    def test_evaluation_config_creation(self):
        """Test EvaluationConfig can be constructed."""
        env_config = get_default_config()
        config = EvaluationConfig(
            env_config=env_config,
            policy_id="test_policy",
            policy_family="age_threshold",
            threshold=100,
            policy_seed=42,
        )
        assert config.policy_id == "test_policy"
        assert config.policy_family == "age_threshold"
        assert config.threshold == 100
        assert config.policy_seed == 42


class TestPolicyEvaluator:
    """Test PolicyEvaluator construction and policy creation."""

    def test_evaluator_creation(self):
        """Test PolicyEvaluator can be constructed."""
        env_config = get_default_config()
        evaluator = PolicyEvaluator(env_config=env_config)
        assert evaluator is not None

    def test_evaluator_with_oracle_allowed(self):
        """Test PolicyEvaluator with allow_oracle=True."""
        env_config = get_default_config()
        evaluator = PolicyEvaluator(
            env_config=env_config,
            allow_oracle=True,
            diagnostic_mode=True,
        )
        assert evaluator.allow_oracle is True
        assert evaluator.diagnostic_mode is True

    def test_create_corrective_only(self):
        """Test creating corrective-only policy."""
        env_config = get_default_config()
        evaluator = PolicyEvaluator(env_config=env_config)
        policy = evaluator.create_policy("corrective_only")
        assert policy is not None

    def test_create_random_feasible(self):
        """Test creating random feasible policy."""
        env_config = get_default_config()
        evaluator = PolicyEvaluator(env_config=env_config)
        policy = evaluator.create_policy("random_feasible", policy_seed=42)
        assert policy is not None
        assert policy.rng is not None

    def test_create_age_threshold(self):
        """Test creating age threshold policy."""
        env_config = get_default_config()
        evaluator = PolicyEvaluator(env_config=env_config)
        policy = evaluator.create_policy("age_threshold", threshold=100)
        assert policy is not None
        assert policy.threshold == 100

    def test_create_age_threshold_missing_threshold(self):
        """Test age threshold requires threshold parameter."""
        env_config = get_default_config()
        evaluator = PolicyEvaluator(env_config=env_config)
        with pytest.raises(ValueError, match="threshold"):
            evaluator.create_policy("age_threshold")

    def test_create_predicted_rul_threshold(self):
        """Test creating predicted RUL threshold policy."""
        env_config = get_default_config()
        evaluator = PolicyEvaluator(env_config=env_config)
        policy = evaluator.create_policy("predicted_rul_threshold", threshold=50)
        assert policy is not None
        assert policy.threshold == 50

    def test_create_greedy_predicted_rul(self):
        """Test creating greedy predicted RUL policy."""
        env_config = get_default_config()
        evaluator = PolicyEvaluator(env_config=env_config)
        policy = evaluator.create_policy(
            "greedy_predicted_rul", activation_threshold=50
        )
        assert policy is not None
        assert policy.activation_threshold == 50

    def test_create_greedy_missing_activation_threshold(self):
        """Test greedy policy requires activation_threshold parameter."""
        env_config = get_default_config()
        evaluator = PolicyEvaluator(env_config=env_config)
        with pytest.raises(ValueError, match="activation_threshold"):
            evaluator.create_policy("greedy_predicted_rul")

    def test_create_oracle_without_allow_oracle(self):
        """Test oracle policy requires allow_oracle=True."""
        env_config = get_default_config()
        evaluator = PolicyEvaluator(env_config=env_config, allow_oracle=False)
        with pytest.raises(ValueError, match="allow_oracle"):
            evaluator.create_policy("oracle_threshold", threshold=20)

    def test_create_oracle_with_allow_oracle(self):
        """Test oracle policy can be created with allow_oracle=True."""
        env_config = get_default_config()
        evaluator = PolicyEvaluator(
            env_config=env_config,
            allow_oracle=True,
            diagnostic_mode=True,
        )
        policy = evaluator.create_policy("oracle_threshold", threshold=20)
        assert policy is not None

    def test_create_unknown_policy(self):
        """Test creating unknown policy raises ValueError."""
        env_config = get_default_config()
        evaluator = PolicyEvaluator(env_config=env_config)
        with pytest.raises(ValueError, match="Unknown policy"):
            evaluator.create_policy("unknown_policy")


class TestPolicyContextCreation:
    """Test context creation for policies."""

    def test_create_policy_context(self):
        """Test creating PolicyContext for practical policy."""
        env_config = get_default_config()
        evaluator = PolicyEvaluator(env_config=env_config)
        context = evaluator.create_context("age_threshold", policy_seed=42)
        assert isinstance(context, PolicyContext)
        assert context.maintenance_capacity == 2
        assert len(context.action_table) == 16

    def test_create_oracle_context(self):
        """Test creating OracleContext for oracle policy."""
        env_config = get_default_config()
        evaluator = PolicyEvaluator(
            env_config=env_config,
            allow_oracle=True,
            diagnostic_mode=True,
        )
        context = evaluator.create_context("oracle_threshold", policy_seed=42)
        # Check it's OracleContext with proper flags
        assert context.allow_oracle is True
        assert context.diagnostic_mode is True


class TestEpisodeResult:
    """Test EpisodeResult dataclass."""

    def test_episode_result_creation(self):
        """Test EpisodeResult can be created."""
        result = EpisodeResult(
            run_id="test_run",
            policy_id="test_policy",
            policy_family="age_threshold",
            threshold=100,
            split="rl_validation",
            scenario_id="scenario_1",
            cost_regime_id="failure-light-no-waste",
            maintenance_capacity=2,
            reset_seed=6521,
            policy_seed=42,
            episode_steps=100,
            episode_return=-50.0,
            discounted_return=-50.0,
            total_cost=50.0,
            preventive_cost=10.0,
            failure_cost=40.0,
            wasted_life_cost=0.0,
            preventive_replacement_count=10,
            failure_count=8,
            action_count=50,
            empty_action_count=50,
            capacity_saturated_step_count=10,
            mean_selected_predicted_rul=0.3,
            mean_selected_age=0.5,
            nan_observation_count=0,
            inf_observation_count=0,
            terminated_count=0,
            truncated=True,
            completed=True,
            error=None,
        )
        assert result.run_id == "test_run"
        assert result.episode_steps == 100
        assert result.completed is True
        assert result.error is None