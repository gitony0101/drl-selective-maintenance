"""
Tests for Milestone 3 threshold tuning.

Verifies:
- Threshold grids match frozen contract
- Tuning uses only rl_validation
- rl_test rejection before load
- Selection uses frozen objective and deterministic tie-break
- Validation-only threshold selection
"""

import numpy as np
import pytest

from src.baselines.tuning import (
    AGE_THRESHOLDS,
    PREDICTED_RUL_THRESHOLDS,
    GREEDY_ACTIVATION_THRESHOLDS,
    ORACLE_THRESHOLDS,
    THRESHOLD_POLICIES,
    NON_TUNED_POLICIES,
    get_threshold_grid,
    ThresholdCandidate,
    SelectedThreshold,
    select_best_threshold,
    tune_threshold,
    candidates_to_dataframe,
    selected_thresholds_to_dict,
)
from src.envs import get_default_config


class TestThresholdGrids:
    """Test frozen threshold grids."""

    def test_age_thresholds_match_contract(self):
        """Test age thresholds match frozen contract."""
        expected = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
        assert AGE_THRESHOLDS == expected

    def test_predicted_rul_thresholds_match_contract(self):
        """Test predicted RUL thresholds match frozen contract."""
        expected = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100]
        assert PREDICTED_RUL_THRESHOLDS == expected

    def test_greedy_activation_thresholds_match_contract(self):
        """Test greedy activation thresholds match frozen contract."""
        expected = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100]
        assert GREEDY_ACTIVATION_THRESHOLDS == expected

    def test_oracle_thresholds_match_contract(self):
        """Test oracle thresholds match frozen contract."""
        expected = [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50]
        assert ORACLE_THRESHOLDS == expected

    def test_get_threshold_grid_age(self):
        """Test get_threshold_grid for age_threshold."""
        grid = get_threshold_grid("age_threshold")
        assert grid == AGE_THRESHOLDS

    def test_get_threshold_grid_predicted_rul(self):
        """Test get_threshold_grid for predicted_rul_threshold."""
        grid = get_threshold_grid("predicted_rul_threshold")
        assert grid == PREDICTED_RUL_THRESHOLDS

    def test_get_threshold_grid_greedy(self):
        """Test get_threshold_grid for greedy_predicted_rul."""
        grid = get_threshold_grid("greedy_predicted_rul")
        assert grid == GREEDY_ACTIVATION_THRESHOLDS

    def test_get_threshold_grid_oracle(self):
        """Test get_threshold_grid for oracle_threshold."""
        grid = get_threshold_grid("oracle_threshold")
        assert grid == ORACLE_THRESHOLDS

    def test_get_threshold_grid_invalid_policy(self):
        """Test get_threshold_grid raises for non-threshold policy."""
        with pytest.raises(ValueError, match="not in threshold policies"):
            get_threshold_grid("corrective_only")

    def test_threshold_policies_contains_all_tunable(self):
        """Test THRESHOLD_POLICIES contains all tunable policies."""
        expected_keys = {
            "age_threshold",
            "predicted_rul_threshold",
            "greedy_predicted_rul",
            "oracle_threshold",
        }
        assert set(THRESHOLD_POLICIES.keys()) == expected_keys

    def test_non_tuned_policies(self):
        """Test NON_TUNED_POLICIES contains correct policies."""
        expected = {"corrective_only", "random_feasible"}
        assert NON_TUNED_POLICIES == expected


class TestThresholdCandidate:
    """Test ThresholdCandidate dataclass."""

    def test_candidate_creation(self):
        """Test ThresholdCandidate can be created."""
        candidate = ThresholdCandidate(
            policy_family="age_threshold",
            threshold=100,
            k_capacity=2,
            cost_regime_id="failure-light-no-waste",
            mean_total_cost=50.0,
            total_failures=5,
            mean_wasted_life_cost=2.5,
            episode_count=10,
        )
        assert candidate.policy_family == "age_threshold"
        assert candidate.threshold == 100
        assert candidate.mean_total_cost == 50.0


class TestSelectBestThreshold:
    """Test threshold selection with frozen objective and tie-break."""

    def test_select_lowest_cost(self):
        """Test selects threshold with lowest mean total cost."""
        candidates = [
            ThresholdCandidate(
                policy_family="age_threshold",
                threshold=100,
                k_capacity=2,
                cost_regime_id="failure-light-no-waste",
                mean_total_cost=50.0,
                total_failures=5,
                mean_wasted_life_cost=2.0,
                episode_count=10,
            ),
            ThresholdCandidate(
                policy_family="age_threshold",
                threshold=150,
                k_capacity=2,
                cost_regime_id="failure-light-no-waste",
                mean_total_cost=45.0,  # Lower cost
                total_failures=6,
                mean_wasted_life_cost=3.0,
                episode_count=10,
            ),
        ]
        selected = select_best_threshold(candidates)
        assert selected.threshold == 150
        assert selected.tie_break_reason == "lowest mean total cost"

    def test_select_fewest_failures_tiebreak(self):
        """Test selects fewer failures when costs tie."""
        candidates = [
            ThresholdCandidate(
                policy_family="age_threshold",
                threshold=100,
                k_capacity=2,
                cost_regime_id="failure-light-no-waste",
                mean_total_cost=50.0,
                total_failures=5,
                mean_wasted_life_cost=2.0,
                episode_count=10,
            ),
            ThresholdCandidate(
                policy_family="age_threshold",
                threshold=150,
                k_capacity=2,
                cost_regime_id="failure-light-no-waste",
                mean_total_cost=50.0,  # Same cost
                total_failures=3,  # Fewer failures
                mean_wasted_life_cost=2.0,
                episode_count=10,
            ),
        ]
        selected = select_best_threshold(candidates)
        assert selected.threshold == 150
        assert "fewest failures" in selected.tie_break_reason

    def test_select_lowest_wasted_life_tiebreak(self):
        """Test selects lowest wasted-life when cost and failures tie."""
        candidates = [
            ThresholdCandidate(
                policy_family="age_threshold",
                threshold=100,
                k_capacity=2,
                cost_regime_id="failure-light-no-waste",
                mean_total_cost=50.0,
                total_failures=5,
                mean_wasted_life_cost=3.0,
                episode_count=10,
            ),
            ThresholdCandidate(
                policy_family="age_threshold",
                threshold=150,
                k_capacity=2,
                cost_regime_id="failure-light-no-waste",
                mean_total_cost=50.0,
                total_failures=5,
                mean_wasted_life_cost=2.0,  # Lower wasted life
                episode_count=10,
            ),
        ]
        selected = select_best_threshold(candidates)
        assert selected.threshold == 150
        assert "lowest wasted-life" in selected.tie_break_reason

    def test_select_lowest_threshold_tiebreak(self):
        """Test selects lower threshold when all metrics tie."""
        candidates = [
            ThresholdCandidate(
                policy_family="age_threshold",
                threshold=150,
                k_capacity=2,
                cost_regime_id="failure-light-no-waste",
                mean_total_cost=50.0,
                total_failures=5,
                mean_wasted_life_cost=2.0,
                episode_count=10,
            ),
            ThresholdCandidate(
                policy_family="age_threshold",
                threshold=100,
                k_capacity=2,
                cost_regime_id="failure-light-no-waste",
                mean_total_cost=50.0,
                total_failures=5,
                mean_wasted_life_cost=2.0,
                episode_count=10,
            ),
        ]
        selected = select_best_threshold(candidates)
        assert selected.threshold == 100
        assert "lowest threshold" in selected.tie_break_reason

    def test_select_raises_with_empty_candidates(self):
        """Test select_best_threshold raises with empty candidates."""
        with pytest.raises(ValueError, match="No threshold candidates"):
            select_best_threshold([])


class TestRlTestBarrier:
    """Test rl_test rejection barrier."""

    def test_tune_threshold_rejects_rl_test(self):
        """Test tune_threshold rejects rl_test scenarios."""
        # This test verifies the barrier without actual tuning
        # by checking that rl_test in scenario_ids raises ValueError
        env_config = get_default_config(split="rl_validation")

        with pytest.raises(ValueError, match="rl_test"):
            tune_threshold(
                policy_family="age_threshold",
                k_capacity=2,
                cost_regime_id="failure-light-no-waste",
                env_config=env_config,
                scenario_ids=["rl_test_scenario_1"],
                reset_seeds=[6521],
            )


class TestCandidatesToDataFrame:
    """Test threshold candidates conversion."""

    def test_candidates_to_dataframe(self):
        """Test candidates_to_dataframe creates correct DataFrame."""
        candidates = [
            ThresholdCandidate(
                policy_family="age_threshold",
                threshold=100,
                k_capacity=2,
                cost_regime_id="failure-light-no-waste",
                mean_total_cost=50.0,
                total_failures=5,
                mean_wasted_life_cost=2.0,
                episode_count=10,
            ),
        ]
        df = candidates_to_dataframe(candidates)
        assert len(df) == 1
        assert df.iloc[0]["policy_family"] == "age_threshold"
        assert df.iloc[0]["threshold"] == 100


class TestSelectedThresholdsToDict:
    """Test selected thresholds conversion."""

    def test_selected_thresholds_to_dict(self):
        """Test selected_thresholds_to_dict creates correct dict."""
        selected = {
            "age_threshold": SelectedThreshold(
                policy_family="age_threshold",
                threshold=100,
                k_capacity=2,
                cost_regime_id="failure-light-no-waste",
                mean_total_cost=50.0,
                total_failures=5,
                mean_wasted_life_cost=2.0,
                episode_count=10,
                tie_break_reason="lowest mean total cost",
            )
        }
        result = selected_thresholds_to_dict(selected)
        assert "age_threshold" in result
        assert result["age_threshold"]["threshold"] == 100
        assert result["age_threshold"]["tie_break_reason"] == "lowest mean total cost"