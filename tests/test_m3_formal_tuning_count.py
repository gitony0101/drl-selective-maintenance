"""
Formal tuning count check for Milestone 3.

Statically computes and tests the expected formal candidate count.

Frozen grid sizes:
- age threshold: 12
- predicted-RUL threshold: 11
- greedy activation: 11
- oracle threshold: 11

For:
- 2 K values
- 4 cost regimes

Expected:
(12 + 11 + 11 + 11) × 2 × 4 = 360 candidates

Oracle candidates:
11 oracle thresholds × 2 K × 4 regimes = 88
"""

import pytest

from src.baselines.tuning import (
    THRESHOLD_POLICIES,
    AGE_THRESHOLDS,
    PREDICTED_RUL_THRESHOLDS,
    GREEDY_ACTIVATION_THRESHOLDS,
    ORACLE_THRESHOLDS,
    get_threshold_grid,
)


class TestFormalTuningCount:
    """Test expected formal tuning candidate count."""

    def test_frozen_grid_sizes(self) -> None:
        """Verify frozen grid sizes match contract."""
        # Age threshold: 12 values
        assert len(AGE_THRESHOLDS) == 12, \
            f"Expected 12 age thresholds, got {len(AGE_THRESHOLDS)}"

        # Predicted RUL threshold: 11 values
        assert len(PREDICTED_RUL_THRESHOLDS) == 11, \
            f"Expected 11 predicted RUL thresholds, got {len(PREDICTED_RUL_THRESHOLDS)}"

        # Greedy activation: 11 values
        assert len(GREEDY_ACTIVATION_THRESHOLDS) == 11, \
            f"Expected 11 greedy activation thresholds, got {len(GREEDY_ACTIVATION_THRESHOLDS)}"

        # Oracle threshold: 11 values
        assert len(ORACLE_THRESHOLDS) == 11, \
            f"Expected 11 oracle thresholds, got {len(ORACLE_THRESHOLDS)}"

    def test_get_threshold_grid_returns_correct_sizes(self) -> None:
        """get_threshold_grid should return correct grid sizes."""
        assert len(get_threshold_grid("age_threshold")) == 12
        assert len(get_threshold_grid("predicted_rul_threshold")) == 11
        assert len(get_threshold_grid("greedy_predicted_rul")) == 11
        assert len(get_threshold_grid("oracle_threshold")) == 11

    def test_total_candidate_count(self) -> None:
        """Total candidate count should be 360.

        Calculation:
        (12 age + 11 pred_rul + 11 greedy + 11 oracle) × 2 K × 4 regimes = 360
        """
        # Grid sizes
        age_count = len(AGE_THRESHOLDS)  # 12
        pred_rul_count = len(PREDICTED_RUL_THRESHOLDS)  # 11
        greedy_count = len(GREEDY_ACTIVATION_THRESHOLDS)  # 11
        oracle_count = len(ORACLE_THRESHOLDS)  # 11

        # K values and cost regimes
        k_count = 2
        regime_count = 4

        # Total candidates per K per regime
        total_per_k_regime = age_count + pred_rul_count + greedy_count + oracle_count

        # Total candidates
        total_candidates = total_per_k_regime * k_count * regime_count

        assert total_candidates == 360, \
            f"Expected 360 candidates, calculated {total_candidates}"

    def test_oracle_candidate_count(self) -> None:
        """Oracle candidate count should be 88.

        Calculation:
        11 oracle thresholds × 2 K × 4 regimes = 88
        """
        oracle_count = len(ORACLE_THRESHOLDS)  # 11
        k_count = 2
        regime_count = 4

        oracle_candidates = oracle_count * k_count * regime_count

        assert oracle_candidates == 88, \
            f"Expected 88 oracle candidates, calculated {oracle_candidates}"

    def test_non_oracle_candidate_count(self) -> None:
        """Non-oracle candidate count should be 272.

        Calculation:
        (12 age + 11 pred_rul + 11 greedy) × 2 K × 4 regimes = 272
        """
        age_count = len(AGE_THRESHOLDS)  # 12
        pred_rul_count = len(PREDICTED_RUL_THRESHOLDS)  # 11
        greedy_count = len(GREEDY_ACTIVATION_THRESHOLDS)  # 11
        k_count = 2
        regime_count = 4

        non_oracle_per_k_regime = age_count + pred_rul_count + greedy_count
        non_oracle_candidates = non_oracle_per_k_regime * k_count * regime_count

        assert non_oracle_candidates == 272, \
            f"Expected 272 non-oracle candidates, calculated {non_oracle_candidates}"

    def test_oracle_plus_non_oracle_equals_total(self) -> None:
        """Oracle + non-oracle should equal total."""
        oracle_count = len(ORACLE_THRESHOLDS)  # 11
        age_count = len(AGE_THRESHOLDS)  # 12
        pred_rul_count = len(PREDICTED_RUL_THRESHOLDS)  # 11
        greedy_count = len(GREEDY_ACTIVATION_THRESHOLDS)  # 11
        k_count = 2
        regime_count = 4

        oracle_candidates = oracle_count * k_count * regime_count
        non_oracle_candidates = (age_count + pred_rul_count + greedy_count) * k_count * regime_count
        total = (age_count + pred_rul_count + greedy_count + oracle_count) * k_count * regime_count

        assert oracle_candidates + non_oracle_candidates == total
        assert oracle_candidates == 88
        assert non_oracle_candidates == 272
        assert total == 360

    def test_policy_families_in_threshold_policies(self) -> None:
        """All four threshold policy families should be in THRESHOLD_POLICIES."""
        expected_families = {
            "age_threshold",
            "predicted_rul_threshold",
            "greedy_predicted_rul",
            "oracle_threshold",
        }

        actual_families = set(THRESHOLD_POLICIES.keys())

        assert actual_families == expected_families, \
            f"Expected {expected_families}, got {actual_families}"

    def test_tuning_plan_not_272_only(self) -> None:
        """Formal tuning plan must include oracle (not 272 only).

        This test ensures the formal plan includes all 360 candidates,
        not just the 272 non-oracle candidates.
        """
        total_per_k_regime = sum(
            len(THRESHOLD_POLICIES[family])
            for family in THRESHOLD_POLICIES
        )

        # 12 + 11 + 11 + 11 = 45
        assert total_per_k_regime == 45, \
            f"Expected 45 thresholds per K/regime, got {total_per_k_regime}"

        # Total should be 45 × 2 × 4 = 360
        total = total_per_k_regime * 2 * 4
        assert total == 360, \
            f"Expected 360 total candidates, got {total}"

        # Fail if only 272 (non-oracle only)
        assert total != 272, \
            "Tuning plan has only 272 candidates (missing oracle)"