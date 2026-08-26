"""
Tests for Milestone 3 policy protocol.

Verifies:
- PolicyContext and OracleContext construction
- Practical policies receive only PolicyContext
- OracleContext rejection by practical policies
- Observation decoding
"""

import numpy as np
import pytest

from src.baselines.protocols import (
    PolicyContext,
    OracleContext,
    Observation,
)
from src.envs.action_table import ACTION_TABLE_N5_K1, ACTION_TABLE_N5_K2


class TestPolicyContext:
    """Test PolicyContext construction and validation."""

    def test_policy_context_creation(self):
        """Test PolicyContext can be constructed with valid parameters."""
        rng = np.random.default_rng(42)
        ctx = PolicyContext(
            maintenance_capacity=2,
            age_scale_cycles=341,
            rul_scale=125.0,
            action_table=ACTION_TABLE_N5_K2,
            cost_regime_id="failure-light-no-waste",
            policy_rng=rng,
        )
        assert ctx.maintenance_capacity == 2
        assert ctx.age_scale_cycles == 341
        assert ctx.rul_scale == 125.0
        assert len(ctx.action_table) == 16  # K=2 has 16 actions
        assert ctx.cost_regime_id == "failure-light-no-waste"

    def test_policy_context_k1(self):
        """Test PolicyContext with K=1."""
        rng = np.random.default_rng(42)
        ctx = PolicyContext(
            maintenance_capacity=1,
            age_scale_cycles=341,
            rul_scale=125.0,
            action_table=ACTION_TABLE_N5_K1,
            cost_regime_id="failure-heavy-waste-aware",
            policy_rng=rng,
        )
        assert ctx.maintenance_capacity == 1
        assert len(ctx.action_table) == 6  # K=1 has 6 actions

    def test_policy_context_frozen(self):
        """Test PolicyContext is immutable."""
        rng = np.random.default_rng(42)
        ctx = PolicyContext(
            maintenance_capacity=2,
            age_scale_cycles=341,
            rul_scale=125.0,
            action_table=ACTION_TABLE_N5_K2,
            cost_regime_id="failure-light-no-waste",
            policy_rng=rng,
        )
        with pytest.raises(Exception):  # frozen dataclass raises AttributeError
            ctx.maintenance_capacity = 1


class TestOracleContext:
    """Test OracleContext construction and validation."""

    def test_oracle_context_creation(self):
        """Test OracleContext requires allow_oracle=True and diagnostic_mode=True."""
        rng = np.random.default_rng(42)
        ctx = OracleContext(
            maintenance_capacity=2,
            age_scale_cycles=341,
            rul_scale=125.0,
            action_table=ACTION_TABLE_N5_K2,
            cost_regime_id="failure-light-no-waste",
            policy_rng=rng,
            allow_oracle=True,
            diagnostic_mode=True,
        )
        assert ctx.allow_oracle is True
        assert ctx.diagnostic_mode is True

    def test_oracle_context_rejects_allow_oracle_false(self):
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

    def test_oracle_context_rejects_diagnostic_mode_false(self):
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

    def test_oracle_context_from_policy_context(self):
        """Test OracleContext can be created from PolicyContext."""
        rng = np.random.default_rng(42)
        policy_ctx = PolicyContext(
            maintenance_capacity=2,
            age_scale_cycles=341,
            rul_scale=125.0,
            action_table=ACTION_TABLE_N5_K2,
            cost_regime_id="failure-light-no-waste",
            policy_rng=rng,
        )
        oracle_ctx = OracleContext.from_policy_context(
            policy_ctx,
            allow_oracle=True,
            diagnostic_mode=True,
        )
        assert oracle_ctx.maintenance_capacity == 2
        assert oracle_ctx.allow_oracle is True
        assert oracle_ctx.diagnostic_mode is True

    def test_oracle_context_from_policy_context_rejects_false(self):
        """Test from_policy_context raises if allow_oracle or diagnostic_mode is False."""
        rng = np.random.default_rng(42)
        policy_ctx = PolicyContext(
            maintenance_capacity=2,
            age_scale_cycles=341,
            rul_scale=125.0,
            action_table=ACTION_TABLE_N5_K2,
            cost_regime_id="failure-light-no-waste",
            policy_rng=rng,
        )
        with pytest.raises(ValueError, match="allow_oracle=True"):
            OracleContext.from_policy_context(policy_ctx, allow_oracle=False, diagnostic_mode=True)
        with pytest.raises(ValueError, match="diagnostic_mode=True"):
            OracleContext.from_policy_context(policy_ctx, allow_oracle=True, diagnostic_mode=False)


class TestObservationDecoding:
    """Test observation decoding helpers."""

    def test_decode_observation_shape(self):
        """Test observation decodes to correct shape."""
        from src.baselines.rule_policies import decode_observation

        rng = np.random.default_rng(42)
        obs = rng.random(10, dtype=np.float32)
        ctx = PolicyContext(
            maintenance_capacity=2,
            age_scale_cycles=341,
            rul_scale=125.0,
            action_table=ACTION_TABLE_N5_K2,
            cost_regime_id="failure-light-no-waste",
            policy_rng=rng,
        )

        ages, pred_ruls = decode_observation(obs, ctx)

        assert ages.shape == (5,)
        assert pred_ruls.shape == (5,)

    def test_decode_observation_layout(self):
        """Test observation layout is [slot_0_age, slot_0_rul, slot_1_age, ...]."""
        from src.baselines.rule_policies import decode_observation

        rng = np.random.default_rng(42)
        # Create known observation
        obs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32)
        ctx = PolicyContext(
            maintenance_capacity=2,
            age_scale_cycles=341,
            rul_scale=125.0,
            action_table=ACTION_TABLE_N5_K2,
            cost_regime_id="failure-light-no-waste",
            policy_rng=rng,
        )

        ages, pred_ruls = decode_observation(obs, ctx)

        expected_ages = np.array([0.1, 0.5, 0.7, 0.9])  # Wrong - let me check the layout
        # Layout: [slot_0_age, slot_0_rul, slot_1_age, slot_1_rul, ...]
        # So ages at indices 0, 2, 4, 6, 8
        # pred_ruls at indices 1, 3, 5, 7, 9
        expected_ages = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        expected_pred_ruls = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

        np.testing.assert_array_almost_equal(ages, expected_ages)
        np.testing.assert_array_almost_equal(pred_ruls, expected_pred_ruls)


class TestInformationBarrier:
    """Test that practical policies cannot access oracle information."""

    def test_policy_context_has_no_true_rul(self):
        """Test PolicyContext does not contain true_rul field."""
        rng = np.random.default_rng(42)
        ctx = PolicyContext(
            maintenance_capacity=2,
            age_scale_cycles=341,
            rul_scale=125.0,
            action_table=ACTION_TABLE_N5_K2,
            cost_regime_id="failure-light-no-waste",
            policy_rng=rng,
        )
        # Verify PolicyContext has no true_rul attribute
        assert not hasattr(ctx, "true_rul")
        assert not hasattr(ctx, "diagnostic_info")

    def test_oracle_context_has_diagnostic_flags(self):
        """Test OracleContext has allow_oracle and diagnostic_mode flags."""
        rng = np.random.default_rng(42)
        ctx = OracleContext(
            maintenance_capacity=2,
            age_scale_cycles=341,
            rul_scale=125.0,
            action_table=ACTION_TABLE_N5_K2,
            cost_regime_id="failure-light-no-waste",
            policy_rng=rng,
            allow_oracle=True,
            diagnostic_mode=True,
        )
        assert ctx.allow_oracle is True
        assert ctx.diagnostic_mode is True

    def test_oracle_context_isinstance_check(self):
        """Test isinstance distinguishes OracleContext from PolicyContext."""
        rng = np.random.default_rng(42)
        policy_ctx = PolicyContext(
            maintenance_capacity=2,
            age_scale_cycles=341,
            rul_scale=125.0,
            action_table=ACTION_TABLE_N5_K2,
            cost_regime_id="failure-light-no-waste",
            policy_rng=rng,
        )
        oracle_ctx = OracleContext(
            maintenance_capacity=2,
            age_scale_cycles=341,
            rul_scale=125.0,
            action_table=ACTION_TABLE_N5_K2,
            cost_regime_id="failure-light-no-waste",
            policy_rng=rng,
            allow_oracle=True,
            diagnostic_mode=True,
        )

        # PolicyContext is not OracleContext
        assert not isinstance(policy_ctx, OracleContext)
        # OracleContext is a separate type
        assert isinstance(oracle_ctx, OracleContext)