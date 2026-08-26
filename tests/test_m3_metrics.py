"""
Tests for Milestone 3 metrics.

Verifies:
- Summary statistics computation
- No NaN or Inf in results
- JSON serializable validation
- Result aggregation by policy
"""

import numpy as np
import pytest
import pandas as pd

from src.baselines.metrics import (
    compute_summary_statistics,
    summarize_results,
    results_to_parquet,
    validate_json_serializable,
    validate_artifact_values,
    PolicySummary,
)
from src.baselines.evaluator import EpisodeResult


class TestComputeSummaryStatistics:
    """Test summary statistics computation."""

    def test_summary_with_values(self):
        """Test compute_summary_statistics with normal values."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = compute_summary_statistics(values)

        assert stats["mean"] == 3.0
        assert stats["sample_std"] > 0
        assert stats["episode_count"] == 5
        assert stats["ci_95_lower"] < stats["mean"]
        assert stats["ci_95_upper"] > stats["mean"]

    def test_summary_with_single_value(self):
        """Test compute_summary_statistics with single value."""
        values = np.array([5.0])
        stats = compute_summary_statistics(values)

        assert stats["mean"] == 5.0
        assert stats["sample_std"] == 0.0  # Cannot compute sample std with n=1
        assert stats["episode_count"] == 1

    def test_summary_with_empty_values(self):
        """Test compute_summary_statistics with empty array."""
        values = np.array([])
        stats = compute_summary_statistics(values)

        assert np.isnan(stats["mean"])
        assert np.isnan(stats["sample_std"])
        assert stats["episode_count"] == 0

    def test_summary_standard_error(self):
        """Test standard error computation."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = compute_summary_statistics(values)

        expected_se = stats["sample_std"] / np.sqrt(5)
        assert abs(stats["standard_error"] - expected_se) < 1e-10

    def test_summary_ci_95(self):
        """Test 95% confidence interval computation."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = compute_summary_statistics(values)

        expected_lower = stats["mean"] - 1.96 * stats["standard_error"]
        expected_upper = stats["mean"] + 1.96 * stats["standard_error"]

        assert abs(stats["ci_95_lower"] - expected_lower) < 1e-10
        assert abs(stats["ci_95_upper"] - expected_upper) < 1e-10


class TestValidateJsonSerializable:
    """Test JSON serializability validation."""

    def test_valid_dict(self):
        """Test validate_json_serializable with valid dict."""
        data = {"a": 1, "b": 2.0, "c": "string", "d": True, "e": None}
        # Should not raise
        validate_json_serializable(data)

    def test_valid_list(self):
        """Test validate_json_serializable with valid list."""
        data = [1, 2.0, "string", True, None]
        # Should not raise
        validate_json_serializable(data)

    def test_nan_raises(self):
        """Test validate_json_serializable raises for NaN."""
        data = {"value": float("nan")}
        with pytest.raises(ValueError, match="NaN"):
            validate_json_serializable(data)

    def test_inf_raises(self):
        """Test validate_json_serializable raises for Inf."""
        data = {"value": float("inf")}
        with pytest.raises(ValueError, match="Inf"):
            validate_json_serializable(data)

    def test_negative_inf_raises(self):
        """Test validate_json_serializable raises for -Inf."""
        data = {"value": float("-inf")}
        with pytest.raises(ValueError, match="Inf"):
            validate_json_serializable(data)

    def test_numpy_scalar_valid(self):
        """Test numpy scalars are valid."""
        data = {"value": np.float64(5.0)}
        # Should not raise
        validate_json_serializable(data)

    def test_nested_dict(self):
        """Test validate_json_serializable with nested dict."""
        data = {"outer": {"inner": 1.0}}
        # Should not raise
        validate_json_serializable(data)

    def test_nested_list(self):
        """Test validate_json_serializable with nested list."""
        data = {"list": [1, 2, [3, 4]]}
        # Should not raise
        validate_json_serializable(data)

    def test_non_serializable_type_raises(self):
        """Test validate_json_serializable raises for non-serializable type."""
        data = {"value": complex(1, 2)}
        with pytest.raises(ValueError, match="not JSON-serializable"):
            validate_json_serializable(data)


class TestResultsToParquet:
    """Test episode results conversion to parquet."""

    def test_results_to_parquet(self):
        """Test results_to_parquet creates correct DataFrame."""
        results = [
            EpisodeResult(
                run_id="run_1",
                policy_id="policy_a",
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
            ),
        ]
        df = results_to_parquet(results)
        assert len(df) == 1
        assert df.iloc[0]["run_id"] == "run_1"
        assert df.iloc[0]["total_cost"] == 50.0


class TestSummarizeResults:
    """Test result summarization."""

    def test_summarize_by_policy(self):
        """Test summarize_results groups by policy."""
        results = [
            EpisodeResult(
                run_id=f"run_{i}",
                policy_id="policy_a",
                policy_family="age_threshold",
                threshold=100,
                split="rl_validation",
                scenario_id=f"scenario_{i}",
                cost_regime_id="failure-light-no-waste",
                maintenance_capacity=2,
                reset_seed=6521,
                policy_seed=42,
                episode_steps=100,
                episode_return=-50.0 - i,
                discounted_return=-50.0 - i,
                total_cost=50.0 + i,
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
            for i in range(5)
        ]
        summary = summarize_results(results)
        assert len(summary) == 1  # One group (policy_a)
        assert summary.iloc[0]["episode_count"] == 5


class TestValidateArtifactValues:
    """Test artifact value validation."""

    def test_valid_dataframe(self):
        """Test validate_artifact_values with valid DataFrame."""
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        # Should not raise
        validate_artifact_values(df, "test")

    def test_inf_in_dataframe_raises(self):
        """Test validate_artifact_values raises for Inf in DataFrame."""
        df = pd.DataFrame({"a": [1.0, float("inf")]})
        with pytest.raises(ValueError, match="Inf"):
            validate_artifact_values(df, "test")

    def test_dataframe_with_nan_allowed(self):
        """Test validate_artifact_values allows NaN in DataFrame."""
        df = pd.DataFrame({"a": [1.0, np.nan]})
        # Should not raise (NaN is allowed)
        validate_artifact_values(df, "test")


class TestPolicySummary:
    """Test PolicySummary dataclass."""

    def test_policy_summary_creation(self):
        """Test PolicySummary can be created."""
        summary = PolicySummary(
            policy_id="policy_a",
            metric="total_cost",
            mean=50.0,
            sample_std=5.0,
            standard_error=2.0,
            ci_95_lower=46.08,
            ci_95_upper=53.92,
            episode_count=10,
            split="rl_validation",
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )
        assert summary.policy_id == "policy_a"
        assert summary.metric == "total_cost"
        assert summary.mean == 50.0
        assert summary.episode_count == 10