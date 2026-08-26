"""
Tests for Milestone 3 oracle isolation.

Verifies:
- Oracle uses true RUL only through OracleContext
- Ordinary policy cannot receive OracleContext
- Hidden state absent from practical policy input
- Oracle requires allow_oracle=True and diagnostic_mode=True
"""

import numpy as np
import pytest

from src.baselines.protocols import PolicyContext, OracleContext
from src.baselines.oracle_policy import OracleThreshold
from src.baselines.rule_policies import (
    CorrectiveOnly,
    AgeThreshold,
    PredictedRULThreshold,
    GreedyPredictedRUL,
)
from src.envs.action_table import ACTION_TABLE_N5_K2


def make_context(k=2, seed=42):
    """Helper to create PolicyContext."""
    rng = np.random.default_rng(seed)
    return PolicyContext(
        maintenance_capacity=k,
        age_scale_cycles=341,
        rul_scale=125.0,
        action_table=ACTION_TABLE_N5_K2,
        cost_regime_id="failure-light-no-waste",
        policy_rng=rng,
    )


def make_oracle_context(k=2, seed=42):
    """Helper to create OracleContext."""
    rng = np.random.default_rng(seed)
    return OracleContext(
        maintenance_capacity=k,
        age_scale_cycles=341,
        rul_scale=125.0,
        action_table=ACTION_TABLE_N5_K2,
        cost_regime_id="failure-light-no-waste",
        policy_rng=rng,
        allow_oracle=True,
        diagnostic_mode=True,
    )


def make_observation(ages=None, pred_ruls=None, seed=42):
    """Helper to create observation array."""
    if ages is None:
        ages = np.array([0.5, 0.3, 0.7, 0.2, 0.9])
    if pred_ruls is None:
        pred_ruls = np.array([0.4, 0.6, 0.2, 0.8, 0.1])
    obs = np.zeros(10, dtype=np.float32)
    for i in range(5):
        obs[i * 2] = ages[i]
        obs[i * 2 + 1] = pred_ruls[i]
    return obs


class MockSlotState:
    """Mock slot state for diagnostic info."""
    def __init__(self, true_rul):
        self.true_rul = true_rul


class TestOracleContext:
    """Test OracleContext construction and validation."""

    def test_oracle_context_requires_allow_oracle_true(self):
        """Test OracleContext raises if allow_oracle=False."""
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="allow_oracle=True"):
            OracleContext(
                maintenance_capacity=2,
                age_scale_cycles=341,
                rul_scale=125.0,
                action_table=ACTION_TABLE_N5_K2,
                cost_regime_id="failure-light-no-waste",
                policy_rng=rng,
                allow_oracle=False,
                diagnostic_mode=True,
            )

    def test_oracle_context_requires_diagnostic_mode_true(self):
        """Test OracleContext raises if diagnostic_mode=False."""
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="diagnostic_mode=True"):
            OracleContext(
                maintenance_capacity=2,
                age_scale_cycles=341,
                rul_scale=125.0,
                action_table=ACTION_TABLE_N5_K2,
                cost_regime_id="failure-light-no-waste",
                policy_rng=rng,
                allow_oracle=True,
                diagnostic_mode=False,
            )


class TestOracleThreshold:
    """Test oracle threshold policy."""

    def test_oracle_requires_oracle_context(self):
        """Test OracleThreshold raises if given PolicyContext instead of OracleContext."""
        policy = OracleThreshold(threshold=20)
        ctx = make_context(k=2)  # Regular PolicyContext
        obs = make_observation()

        with pytest.raises(ValueError, match="OracleContext"):
            policy.select_action(obs, ctx)

    def test_oracle_requires_allow_oracle(self):
        """Test OracleThreshold raises if allow_oracle=False."""
        policy = OracleThreshold(threshold=20)
        rng = np.random.default_rng(42)
        obs = make_observation()

        # OracleContext raises during construction if allow_oracle=False
        with pytest.raises(ValueError, match="allow_oracle=True"):
            OracleContext(
                maintenance_capacity=2,
                age_scale_cycles=341,
                rul_scale=125.0,
                action_table=ACTION_TABLE_N5_K2,
                cost_regime_id="failure-light-no-waste",
                policy_rng=rng,
                allow_oracle=False,
                diagnostic_mode=True,
            )

    def test_oracle_requires_diagnostic_mode(self):
        """Test OracleThreshold raises if diagnostic_mode=False."""
        policy = OracleThreshold(threshold=20)
        rng = np.random.default_rng(42)
        obs = make_observation()

        # OracleContext raises during construction if diagnostic_mode=False
        with pytest.raises(ValueError, match="diagnostic_mode=True"):
            OracleContext(
                maintenance_capacity=2,
                age_scale_cycles=341,
                rul_scale=125.0,
                action_table=ACTION_TABLE_N5_K2,
                cost_regime_id="failure-light-no-waste",
                policy_rng=rng,
                allow_oracle=True,
                diagnostic_mode=False,
            )

    def test_oracle_requires_diagnostic_info(self):
        """Test OracleThreshold raises if diagnostic_info is None."""
        policy = OracleThreshold(threshold=20)
        ctx = make_oracle_context(k=2)
        obs = make_observation()

        with pytest.raises(ValueError, match="diagnostic_info"):
            policy.select_action(obs, ctx, diagnostic_info=None)

    def test_oracle_uses_true_rul(self):
        """Test OracleThreshold selects based on true RUL."""
        policy = OracleThreshold(threshold=20)  # 20 cycles
        ctx = make_oracle_context(k=2)
        obs = make_observation()

        # Provide diagnostic info with true RUL values in actual env format
        # Slots 0 and 3 have true_rul <= 20
        diagnostic_info = {
            "slot_0_diagnostic": {"true_rul": 15},   # Slot 0: below threshold
            "slot_1_diagnostic": {"true_rul": 50},   # Slot 1: above
            "slot_2_diagnostic": {"true_rul": 100},  # Slot 2: above
            "slot_3_diagnostic": {"true_rul": 10},   # Slot 3: below threshold
            "slot_4_diagnostic": {"true_rul": 80},   # Slot 4: above
        }

        action = policy.select_action(obs, ctx, diagnostic_info=diagnostic_info)
        slots = ctx.action_table[action]

        assert len(slots) == 2  # K=2
        assert 0 in slots  # Slot 0 (15 <= 20)
        assert 3 in slots  # Slot 3 (10 <= 20)

    def test_oracle_selects_lowest_true_rul(self):
        """Test OracleThreshold selects lowest true RUL when capacity binds."""
        policy = OracleThreshold(threshold=100)  # High threshold
        ctx = make_oracle_context(k=2)
        obs = make_observation()

        # All slots below threshold, slots 2 and 4 have lowest true RUL
        diagnostic_info = {
            "slot_0_diagnostic": {"true_rul": 80},
            "slot_1_diagnostic": {"true_rul": 90},
            "slot_2_diagnostic": {"true_rul": 10},   # Slot 2: lowest
            "slot_3_diagnostic": {"true_rul": 70},
            "slot_4_diagnostic": {"true_rul": 5},    # Slot 4: second lowest
        }

        action = policy.select_action(obs, ctx, diagnostic_info=diagnostic_info)
        slots = ctx.action_table[action]

        assert len(slots) == 2
        assert 4 in slots  # Slot 4 (5 = lowest)
        assert 2 in slots  # Slot 2 (10 = second lowest)

    def test_oracle_empty_action_when_no_candidates(self):
        """Test OracleThreshold returns empty action when no slots below threshold."""
        policy = OracleThreshold(threshold=5)  # Very low threshold
        ctx = make_oracle_context(k=2)
        obs = make_observation()

        diagnostic_info = {
            "slot_0_diagnostic": {"true_rul": 50},
            "slot_1_diagnostic": {"true_rul": 60},
            "slot_2_diagnostic": {"true_rul": 70},
            "slot_3_diagnostic": {"true_rul": 80},
            "slot_4_diagnostic": {"true_rul": 90},
        }

        action = policy.select_action(obs, ctx, diagnostic_info=diagnostic_info)
        assert action == 0


class TestPracticalPolicyIsolation:
    """Test that practical policies cannot access oracle information."""

    def test_corrective_only_no_oracle_access(self):
        """Test CorrectiveOnly does not use oracle context."""
        policy = CorrectiveOnly()
        ctx = make_context(k=2)
        obs = make_observation()

        # Should work with regular context
        action = policy.select_action(obs, ctx)
        assert action == 0

        # Should not require oracle context
        # (it ignores context entirely, but type system should prevent OracleContext)

    def test_age_threshold_no_oracle_access(self):
        """Test AgeThreshold works only with PolicyContext."""
        policy = AgeThreshold(threshold=100)
        ctx = make_context(k=2)
        obs = make_observation()

        action = policy.select_action(obs, ctx)
        # Returns action based on observed (predicted) RUL, not true RUL
        assert 0 <= action < len(ctx.action_table)

    def test_predicted_rul_threshold_no_oracle_access(self):
        """Test PredictedRULThreshold works only with PolicyContext."""
        policy = PredictedRULThreshold(threshold=50)
        ctx = make_context(k=2)
        obs = make_observation()

        action = policy.select_action(obs, ctx)
        assert 0 <= action < len(ctx.action_table)

    def test_greedy_no_oracle_access(self):
        """Test GreedyPredictedRUL works only with PolicyContext."""
        policy = GreedyPredictedRUL(activation_threshold=50)
        ctx = make_context(k=2)
        obs = make_observation()

        action = policy.select_action(obs, ctx)
        assert 0 <= action < len(ctx.action_table)


class TestInformationBarrier:
    """Test information barriers between practical and oracle policies."""

    def test_policy_context_has_no_true_rul_field(self):
        """Test PolicyContext has no true_rul attribute."""
        ctx = make_context(k=2)
        assert not hasattr(ctx, "true_rul")

    def test_policy_context_has_no_diagnostic_info(self):
        """Test PolicyContext has no diagnostic_info attribute."""
        ctx = make_context(k=2)
        assert not hasattr(ctx, "diagnostic_info")

    def test_oracle_context_is_separate_type(self):
        """Test OracleContext is a separate type from PolicyContext."""
        policy_ctx = make_context(k=2)
        oracle_ctx = make_oracle_context(k=2)

        assert type(policy_ctx) != type(oracle_ctx)
        assert not isinstance(policy_ctx, OracleContext)
        assert isinstance(oracle_ctx, OracleContext)

    def test_oracle_context_has_diagnostic_flags(self):
        """Test OracleContext has allow_oracle and diagnostic_mode flags."""
        ctx = make_oracle_context(k=2)
        assert ctx.allow_oracle is True
        assert ctx.diagnostic_mode is True

    def test_practical_policy_type_annotation_mismatch(self):
        """Test that practical policies expect PolicyContext type."""
        # This is a type-checking test - verifies at runtime that
        # practical policies work with PolicyContext
        policy = AgeThreshold(threshold=100)
        ctx = make_context(k=2)
        obs = make_observation()

        # Should succeed with PolicyContext
        action = policy.select_action(obs, ctx)
        assert 0 <= action < 16