"""
Tests for the centralized case loader.

Tests verify:
- predictor_train K=1 derivation from K=2 source
- predictor_train K=2 direct load
- rl_validation K=1 derivation from K=2 source
- rl_validation K=2 direct load
- Four cost regimes derivation
- rl_test rejection barrier
- Scenario ID stability and uniqueness
- Split/K/cost_regime matching
"""

import pytest
from pathlib import Path

from src.baselines.case_loader import (
    load_cases,
    load_all_regimes,
    get_scenario_bank_for_case,
    get_k1_from_k2_source,
    verify_predictor_train_k1_derivation,
    verify_rl_validation_k1_derivation,
    CaseLoadError,
    RlTestBarrierError,
    ALL_COST_REGIMES,
)
from src.envs.scenario_bank import load_scenario_bank


class TestRlTestBarrier:
    """Test rl_test rejection barrier."""

    def test_rl_test_split_rejected(self) -> None:
        """rl_test split must be rejected before any file loading."""
        with pytest.raises(RlTestBarrierError, match="rl_test split is forbidden"):
            load_cases(
                split="rl_test",
                k=1,
                cost_regime_id="failure-light-no-waste",
            )

    def test_rl_test_k2_rejected(self) -> None:
        """rl_test must be rejected for K=2 as well."""
        with pytest.raises(RlTestBarrierError):
            load_cases(
                split="rl_test",
                k=2,
                cost_regime_id="failure-heavy-no-waste",
            )


class TestPredictorTrainK1:
    """Test predictor_train K=1 derivation."""

    def test_k1_derivation_succeeds(self) -> None:
        """predictor_train K=1 should derive from K=2 source."""
        result = load_cases(
            split="predictor_train",
            k=1,
            cost_regime_id="failure-light-no-waste",
            source_bank_path="data/scenario_banks/predictor_train_smoke.json",
        )

        assert result.split == "predictor_train"
        assert result.k == 1
        assert result.cost_regime_id == "failure-light-no-waste"
        assert len(result.scenario_ids) > 0
        assert result.derived_from_k == 2

    def test_k1_scenario_ids_have_k1_suffix(self) -> None:
        """K=1 derived scenarios should have _k1_ suffix in IDs."""
        result = load_cases(
            split="predictor_train",
            k=1,
            cost_regime_id="failure-light-no-waste",
            source_bank_path="data/scenario_banks/predictor_train_smoke.json",
        )

        for scenario_id in result.scenario_ids:
            assert "_k1_" in scenario_id, f"K=1 scenario ID should contain '_k1_': {scenario_id}"

    def test_k1_scenarios_have_correct_k(self) -> None:
        """K=1 derived scenarios should have maintenance_capacity=1."""
        bank = get_scenario_bank_for_case(
            split="predictor_train",
            k=1,
            cost_regime_id="failure-light-no-waste",
            source_bank_path="data/scenario_banks/predictor_train_smoke.json",
        )

        for scenario in bank.scenarios:
            assert scenario.maintenance_capacity == 1

    def test_verify_helper_succeeds(self) -> None:
        """verify_predictor_train_k1_derivation should succeed."""
        provenance = verify_predictor_train_k1_derivation()
        assert "predictor_train" in provenance
        assert "K=1" in provenance or "derived" in provenance


class TestPredictorTrainK2:
    """Test predictor_train K=2 direct load."""

    def test_k2_load_succeeds(self) -> None:
        """predictor_train K=2 should load directly."""
        result = load_cases(
            split="predictor_train",
            k=2,
            cost_regime_id="failure-light-no-waste",
            source_bank_path="data/scenario_banks/predictor_train_smoke.json",
        )

        assert result.split == "predictor_train"
        assert result.k == 2
        assert result.cost_regime_id == "failure-light-no-waste"
        assert len(result.scenario_ids) > 0
        # K=2 is direct load, not derived from K=2
        assert result.derived_from_k is None

    def test_k2_scenario_ids_have_regime_suffix(self) -> None:
        """K=2 derived scenarios should have cost regime suffix."""
        result = load_cases(
            split="predictor_train",
            k=2,
            cost_regime_id="failure-light-no-waste",
            source_bank_path="data/scenario_banks/predictor_train_smoke.json",
        )

        for scenario_id in result.scenario_ids:
            assert "_failure-light-no-waste" in scenario_id


class TestRlValidationK1:
    """Test rl_validation K=1 derivation."""

    def test_k1_derivation_succeeds(self) -> None:
        """rl_validation K=1 should derive from K=2 source."""
        result = load_cases(
            split="rl_validation",
            k=1,
            cost_regime_id="failure-light-no-waste",
            source_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )

        assert result.split == "rl_validation"
        assert result.k == 1
        assert result.cost_regime_id == "failure-light-no-waste"
        assert len(result.scenario_ids) > 0

    def test_verify_helper_succeeds(self) -> None:
        """verify_rl_validation_k1_derivation should succeed."""
        provenance = verify_rl_validation_k1_derivation()
        assert "rl_validation" in provenance


class TestRlValidationK2:
    """Test rl_validation K=2 direct load."""

    def test_k2_load_succeeds(self) -> None:
        """rl_validation K=2 should load directly."""
        result = load_cases(
            split="rl_validation",
            k=2,
            cost_regime_id="failure-light-no-waste",
            source_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )

        assert result.split == "rl_validation"
        assert result.k == 2
        assert len(result.scenario_ids) > 0


class TestFourCostRegimes:
    """Test all four cost regimes."""

    def test_all_regimes_loadable(self) -> None:
        """All four cost regimes should be loadable."""
        for regime_id in ALL_COST_REGIMES:
            result = load_cases(
                split="predictor_train",
                k=2,
                cost_regime_id=regime_id,
                source_bank_path="data/scenario_banks/predictor_train_smoke.json",
            )
            assert result.cost_regime_id == regime_id
            assert len(result.scenario_ids) > 0

    def test_load_all_regimes_returns_four(self) -> None:
        """load_all_regimes should return all four regimes."""
        results = load_all_regimes(
            split="predictor_train",
            k=2,
            source_bank_path="data/scenario_banks/predictor_train_smoke.json",
        )

        assert len(results) == 4
        for regime_id in ALL_COST_REGIMES:
            assert regime_id in results
            assert len(results[regime_id].scenario_ids) > 0

    def test_regime_scenario_ids_have_regime_suffix(self) -> None:
        """Each regime's scenarios should have the regime suffix."""
        results = load_all_regimes(
            split="predictor_train",
            k=2,
            source_bank_path="data/scenario_banks/predictor_train_smoke.json",
        )

        for regime_id, result in results.items():
            for scenario_id in result.scenario_ids:
                assert f"_{regime_id}" in scenario_id, \
                    f"Scenario ID should contain regime suffix: {scenario_id}"


class TestScenarioValidation:
    """Test scenario validation."""

    def test_split_matches_request(self) -> None:
        """Derived scenarios should have matching split."""
        result = load_cases(
            split="predictor_train",
            k=1,
            cost_regime_id="failure-light-no-waste",
            source_bank_path="data/scenario_banks/predictor_train_smoke.json",
        )

        bank = get_scenario_bank_for_case(
            split="predictor_train",
            k=1,
            cost_regime_id="failure-light-no-waste",
            source_bank_path="data/scenario_banks/predictor_train_smoke.json",
        )

        for scenario in bank.scenarios:
            assert scenario.split == "predictor_train"

    def test_k_matches_request(self) -> None:
        """Derived scenarios should have matching K."""
        bank_k1 = get_scenario_bank_for_case(
            split="predictor_train",
            k=1,
            cost_regime_id="failure-light-no-waste",
            source_bank_path="data/scenario_banks/predictor_train_smoke.json",
        )
        bank_k2 = get_scenario_bank_for_case(
            split="predictor_train",
            k=2,
            cost_regime_id="failure-light-no-waste",
            source_bank_path="data/scenario_banks/predictor_train_smoke.json",
        )

        for scenario in bank_k1.scenarios:
            assert scenario.maintenance_capacity == 1

        for scenario in bank_k2.scenarios:
            assert scenario.maintenance_capacity == 2

    def test_cost_regime_matches_request(self) -> None:
        """Derived scenarios should have matching cost regime."""
        for regime_id in ALL_COST_REGIMES:
            bank = get_scenario_bank_for_case(
                split="predictor_train",
                k=2,
                cost_regime_id=regime_id,
                source_bank_path="data/scenario_banks/predictor_train_smoke.json",
            )

            for scenario in bank.scenarios:
                assert scenario.cost_regime_id == regime_id


class TestScenarioIdUniqueness:
    """Test scenario ID uniqueness and stability."""

    def test_no_duplicate_ids_within_case(self) -> None:
        """No duplicate scenario IDs within a single case."""
        result = load_cases(
            split="predictor_train",
            k=1,
            cost_regime_id="failure-light-no-waste",
            source_bank_path="data/scenario_banks/predictor_train_smoke.json",
        )

        assert len(result.scenario_ids) == len(set(result.scenario_ids)), \
            "Duplicate scenario IDs detected"

    def test_k1_and_k2_ids_dont_collide(self) -> None:
        """K=1 and K=2 scenario IDs should not collide."""
        result_k1 = load_cases(
            split="predictor_train",
            k=1,
            cost_regime_id="failure-light-no-waste",
            source_bank_path="data/scenario_banks/predictor_train_smoke.json",
        )
        result_k2 = load_cases(
            split="predictor_train",
            k=2,
            cost_regime_id="failure-light-no-waste",
            source_bank_path="data/scenario_banks/predictor_train_smoke.json",
        )

        k1_ids = set(result_k1.scenario_ids)
        k2_ids = set(result_k2.scenario_ids)

        overlap = k1_ids & k2_ids
        assert len(overlap) == 0, f"K=1 and K=2 IDs should not overlap: {overlap}"

    def test_different_regimes_dont_collide(self) -> None:
        """Different cost regimes should have distinct scenario IDs."""
        results = load_all_regimes(
            split="predictor_train",
            k=2,
            source_bank_path="data/scenario_banks/predictor_train_smoke.json",
        )

        all_ids = []
        for result in results.values():
            all_ids.extend(result.scenario_ids)

        assert len(all_ids) == len(set(all_ids)), \
            "Duplicate scenario IDs across cost regimes"


class TestInvalidInputs:
    """Test invalid input handling."""

    def test_invalid_k_rejected(self) -> None:
        """Invalid K values should be rejected."""
        with pytest.raises(CaseLoadError, match="Invalid K"):
            load_cases(
                split="predictor_train",
                k=0,
                cost_regime_id="failure-light-no-waste",
            )

        with pytest.raises(CaseLoadError):
            load_cases(
                split="predictor_train",
                k=3,
                cost_regime_id="failure-light-no-waste",
            )

    def test_invalid_cost_regime_rejected(self) -> None:
        """Invalid cost regime should be rejected."""
        with pytest.raises(CaseLoadError, match="Invalid cost regime"):
            load_cases(
                split="predictor_train",
                k=1,
                cost_regime_id="nonexistent_regime",
            )

    def test_invalid_split_rejected(self) -> None:
        """Invalid split should be rejected."""
        with pytest.raises(CaseLoadError, match="Invalid split"):
            load_cases(
                split="invalid_split",
                k=1,
                cost_regime_id="failure-light-no-waste",
            )

    def test_missing_source_bank_fails(self) -> None:
        """Missing source bank should fail with clear error."""
        with pytest.raises(CaseLoadError, match="Source bank not found"):
            load_cases(
                split="predictor_train",
                k=1,
                cost_regime_id="failure-light-no-waste",
                source_bank_path="nonexistent/path.json",
            )


class TestK1DerivationHelper:
    """Test the K=1 derivation helper."""

    def test_get_k1_from_k2_source(self) -> None:
        """get_k1_from_k2_source should derive K=1 scenarios."""
        source_path = Path("data/scenario_banks/predictor_train_smoke.json")
        source_bank = load_scenario_bank(source_path)

        # Source should be K=2
        assert source_bank.scenarios[0].maintenance_capacity == 2

        k1_bank = get_k1_from_k2_source(source_bank)

        # Derived should be K=1
        for scenario in k1_bank.scenarios:
            assert scenario.maintenance_capacity == 1

        # Should have same number of scenarios
        assert len(k1_bank.scenarios) == len(source_bank.scenarios)