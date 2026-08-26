"""
Unit tests for Milestone 2 scenario bank data structures.

Tests cover:
- Correct scenario validates
- Wrong number of units fails
- Wrong number of cycles fails
- cycle 0 fails
- cross-split unit fails
- predictor_validation unit fails
- duplicate scenario ID fails
- mismatched K fails
- mismatched horizon fails
- initial age equals cycle - 1
- deterministic serialization
- identical seed and scenario input reproduce identical loaded data
"""

import pytest

pytestmark = pytest.mark.requires_external_assets
from pathlib import Path
from typing import Any

from src.envs.scenario_bank import (
    Scenario,
    ScenarioBank,
    load_scenario_bank,
    save_scenario_bank,
    validate_full_scenario_bank,
    validate_scenario_against_config,
    validate_scenario_cycles_exist,
    validate_scenario_units_against_split,
)
from src.envs.config import EnvironmentConfig, get_default_config


class MockPredictionStore:
    """Mock prediction store for testing.

    Matches the actual PredictionStore interface with get() and get_units() methods.
    """

    def __init__(self, unit_splits: dict[int, str]) -> None:
        self._unit_splits = unit_splits
        # Build reverse index: split -> list of unit_ids
        self._split_units: dict[str, list[int]] = {}
        for unit_id, split in unit_splits.items():
            if split not in self._split_units:
                self._split_units[split] = []
            self._split_units[split].append(unit_id)

    def get_units(self, split: str) -> list[int]:
        """Get all unit IDs for a given split."""
        return self._split_units.get(split, [])

    def get(
        self,
        split: str,
        unit_id: int,
        cycle: int,
    ) -> "MockPredictionResult":
        """Get prediction for a specific (split, unit_id, cycle)."""
        # Check if unit exists and belongs to the split
        if unit_id not in self._unit_splits:
            return MockPredictionResult(found=False)
        if self._unit_splits[unit_id] != split:
            return MockPredictionResult(found=False)
        if cycle <= 0:
            return MockPredictionResult(found=False)
        return MockPredictionResult(found=True)


class MockPredictionResult:
    """Mock prediction result for testing."""

    def __init__(self, found: bool) -> None:
        self.found = found


class TestValidScenario:
    """Test that valid scenarios pass validation."""

    def test_valid_rl_validation_scenario(self) -> None:
        """Valid rl_validation scenario should validate."""
        scenario = Scenario(
            scenario_id="val_001",
            split="rl_validation",
            initial_unit_ids=(1, 2, 3, 4, 5),
            initial_cycles=(1, 1, 1, 1, 1),
            replacement_seed=123,
            environment_seed=456,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )
        assert scenario.scenario_id == "val_001"
        assert len(scenario.initial_unit_ids) == 5

    def test_valid_rl_test_scenario(self) -> None:
        """Valid rl_test scenario should validate."""
        scenario = Scenario(
            scenario_id="test_001",
            split="rl_test",
            initial_unit_ids=(10, 20, 30, 40, 50),
            initial_cycles=(1, 20, 40, 60, 80),
            replacement_seed=789,
            environment_seed=101112,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-heavy-waste-aware",
        )
        assert scenario.scenario_id == "test_001"


class TestWrongNumberOfUnits:
    """Test that wrong number of units fails."""

    def test_four_units_raises(self) -> None:
        """4 units instead of 5 should raise ValueError."""
        with pytest.raises(ValueError, match="initial_unit_ids must have exactly 5"):
            Scenario(
                scenario_id="test",
                split="rl_validation",
                initial_unit_ids=(1, 2, 3, 4),  # Only 4
                initial_cycles=(1, 1, 1, 1),
                replacement_seed=123,
                environment_seed=456,
                episode_horizon=100,
                maintenance_capacity=2,
                cost_regime_id="failure-light-no-waste",
            )

    def test_six_units_raises(self) -> None:
        """6 units instead of 5 should raise ValueError."""
        with pytest.raises(ValueError, match="initial_unit_ids must have exactly 5"):
            Scenario(
                scenario_id="test",
                split="rl_validation",
                initial_unit_ids=(1, 2, 3, 4, 5, 6),  # 6 units
                initial_cycles=(1, 1, 1, 1, 1, 1),
                replacement_seed=123,
                environment_seed=456,
                episode_horizon=100,
                maintenance_capacity=2,
                cost_regime_id="failure-light-no-waste",
            )


class TestWrongNumberOfCycles:
    """Test that wrong number of cycles fails."""

    def test_four_cycles_raises(self) -> None:
        """4 cycles instead of 5 should raise ValueError."""
        with pytest.raises(ValueError, match="initial_cycles must have exactly 5"):
            Scenario(
                scenario_id="test",
                split="rl_validation",
                initial_unit_ids=(1, 2, 3, 4, 5),
                initial_cycles=(1, 1, 1, 1),  # Only 4
                replacement_seed=123,
                environment_seed=456,
                episode_horizon=100,
                maintenance_capacity=2,
                cost_regime_id="failure-light-no-waste",
            )

    def test_six_cycles_raises(self) -> None:
        """6 cycles instead of 5 should raise ValueError."""
        with pytest.raises(ValueError, match="initial_cycles must have exactly 5"):
            Scenario(
                scenario_id="test",
                split="rl_validation",
                initial_unit_ids=(1, 2, 3, 4, 5),
                initial_cycles=(1, 1, 1, 1, 1, 1),  # 6 cycles
                replacement_seed=123,
                environment_seed=456,
                episode_horizon=100,
                maintenance_capacity=2,
                cost_regime_id="failure-light-no-waste",
            )


class TestCycleZeroFails:
    """Test that cycle 0 or negative cycles fail."""

    def test_cycle_zero_raises(self) -> None:
        """cycle 0 should raise ValueError."""
        with pytest.raises(ValueError, match="initial_cycles.*must be positive"):
            Scenario(
                scenario_id="test",
                split="rl_validation",
                initial_unit_ids=(1, 2, 3, 4, 5),
                initial_cycles=(0, 1, 1, 1, 1),  # cycle 0
                replacement_seed=123,
                environment_seed=456,
                episode_horizon=100,
                maintenance_capacity=2,
                cost_regime_id="failure-light-no-waste",
            )

    def test_negative_cycle_raises(self) -> None:
        """Negative cycle should raise ValueError."""
        with pytest.raises(ValueError, match="initial_cycles.*must be positive"):
            Scenario(
                scenario_id="test",
                split="rl_validation",
                initial_unit_ids=(1, 2, 3, 4, 5),
                initial_cycles=(1, -1, 1, 1, 1),  # negative cycle
                replacement_seed=123,
                environment_seed=456,
                episode_horizon=100,
                maintenance_capacity=2,
                cost_regime_id="failure-light-no-waste",
            )


class TestCrossSplitUnitFails:
    """Test that cross-split units are detected."""

    def test_unit_from_wrong_split_raises(self) -> None:
        """Unit from different split should raise ValueError."""
        # Create mock store where unit 99 belongs to rl_test, not rl_validation
        store = MockPredictionStore({
            1: "rl_validation",
            2: "rl_validation",
            3: "rl_validation",
            4: "rl_validation",
            99: "rl_test",  # Wrong split!
        })

        scenario = Scenario(
            scenario_id="test",
            split="rl_validation",
            initial_unit_ids=(1, 2, 3, 4, 99),
            initial_cycles=(1, 1, 1, 1, 1),
            replacement_seed=123,
            environment_seed=456,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )

        with pytest.raises(ValueError, match="belongs to split"):
            validate_scenario_units_against_split(scenario, store)

    def test_unit_not_in_store_raises(self) -> None:
        """Unit not in store should raise ValueError."""
        store = MockPredictionStore({
            1: "rl_validation",
            2: "rl_validation",
            3: "rl_validation",
            4: "rl_validation",
            5: "rl_validation",
        })

        scenario = Scenario(
            scenario_id="test",
            split="rl_validation",
            initial_unit_ids=(1, 2, 3, 4, 999),  # 999 not in store
            initial_cycles=(1, 1, 1, 1, 1),
            replacement_seed=123,
            environment_seed=456,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )

        with pytest.raises(ValueError, match="not found"):
            validate_scenario_units_against_split(scenario, store)


class TestPredictorValidationUnitFails:
    """Test that predictor_validation units are rejected."""

    def test_predictor_validation_unit_raises(self) -> None:
        """Unit from predictor_validation split should raise ValueError."""
        # Create mock store where unit 50 is from predictor_validation
        store = MockPredictionStore({
            1: "rl_validation",
            2: "rl_validation",
            3: "rl_validation",
            4: "rl_validation",
            50: "predictor_validation",  # Not allowed!
        })

        scenario = Scenario(
            scenario_id="test",
            split="rl_validation",
            initial_unit_ids=(1, 2, 3, 4, 50),
            initial_cycles=(1, 1, 1, 1, 1),
            replacement_seed=123,
            environment_seed=456,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )

        with pytest.raises(ValueError, match="belongs to split"):
            validate_scenario_units_against_split(scenario, store)


class TestDuplicateScenarioIdFails:
    """Test that duplicate scenario IDs are detected."""

    def test_duplicate_id_in_bank_raises(self) -> None:
        """Duplicate scenario ID in bank should raise ValueError."""
        s1 = Scenario(
            scenario_id="dup_001",
            split="rl_validation",
            initial_unit_ids=(1, 2, 3, 4, 5),
            initial_cycles=(1, 1, 1, 1, 1),
            replacement_seed=123,
            environment_seed=456,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )
        s2 = Scenario(
            scenario_id="dup_001",  # Same ID!
            split="rl_validation",
            initial_unit_ids=(10, 20, 30, 40, 50),
            initial_cycles=(1, 1, 1, 1, 1),
            replacement_seed=789,
            environment_seed=101112,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )

        with pytest.raises(ValueError, match="Duplicate scenario ID"):
            ScenarioBank(bank_id="test_bank", split="rl_validation", scenarios=(s1, s2))


class TestMismatchedK:
    """Test that mismatched K is detected."""

    def test_scenario_k_differs_from_config_raises(self) -> None:
        """Scenario K different from config should raise ValueError."""
        scenario = Scenario(
            scenario_id="test",
            split="rl_validation",
            initial_unit_ids=(1, 2, 3, 4, 5),
            initial_cycles=(1, 1, 1, 1, 1),
            replacement_seed=123,
            environment_seed=456,
            episode_horizon=100,
            maintenance_capacity=1,  # K=1
            cost_regime_id="failure-light-no-waste",
        )
        config = get_default_config(maintenance_capacity=2)  # Config K=2

        with pytest.raises(ValueError, match="maintenance_capacity.*does not match"):
            validate_scenario_against_config(scenario, config)


class TestMismatchedHorizon:
    """Test that mismatched horizon is detected."""

    def test_scenario_horizon_differs_from_config_raises(self) -> None:
        """Scenario horizon different from config should raise ValueError."""
        scenario = Scenario(
            scenario_id="test",
            split="rl_validation",
            initial_unit_ids=(1, 2, 3, 4, 5),
            initial_cycles=(1, 1, 1, 1, 1),
            replacement_seed=123,
            environment_seed=456,
            episode_horizon=50,  # Different horizon
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )
        config = get_default_config()  # Default horizon=100

        with pytest.raises(ValueError, match="episode_horizon.*does not match"):
            validate_scenario_against_config(scenario, config)


class TestInitialAgeRule:
    """Test that initial age equals cycle - 1."""

    def test_cycle_1_gives_age_0(self) -> None:
        """Cycle 1 should give age 0."""
        scenario = Scenario(
            scenario_id="test",
            split="rl_validation",
            initial_unit_ids=(1, 2, 3, 4, 5),
            initial_cycles=(1, 1, 1, 1, 1),
            replacement_seed=123,
            environment_seed=456,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )
        for i in range(5):
            assert scenario.get_initial_age(i) == 0

    def test_cycle_81_gives_age_80(self) -> None:
        """Cycle 81 should give age 80."""
        scenario = Scenario(
            scenario_id="test",
            split="rl_validation",
            initial_unit_ids=(1, 2, 3, 4, 5),
            initial_cycles=(81, 81, 81, 81, 81),
            replacement_seed=123,
            environment_seed=456,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )
        for i in range(5):
            assert scenario.get_initial_age(i) == 80

    def test_mixed_cycles_give_mixed_ages(self) -> None:
        """Mixed cycles should give corresponding ages."""
        scenario = Scenario(
            scenario_id="test",
            split="rl_validation",
            initial_unit_ids=(1, 2, 3, 4, 5),
            initial_cycles=(1, 20, 40, 60, 80),
            replacement_seed=123,
            environment_seed=456,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )
        expected_ages = [0, 19, 39, 59, 79]
        for i, expected in enumerate(expected_ages):
            assert scenario.get_initial_age(i) == expected


class TestDeterministicSerialization:
    """Test deterministic serialization."""

    def test_serialize_deterministic_returns_json(self) -> None:
        """serialize_deterministic should return valid JSON."""
        scenario = Scenario(
            scenario_id="test",
            split="rl_validation",
            initial_unit_ids=(1, 2, 3, 4, 5),
            initial_cycles=(1, 1, 1, 1, 1),
            replacement_seed=123,
            environment_seed=456,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )
        json_str = scenario.serialize_deterministic()
        import json
        parsed = json.loads(json_str)
        assert parsed["scenario_id"] == "test"

    def test_identical_scenarios_produce_same_hash(self) -> None:
        """Identical scenarios should produce the same hash."""
        s1 = Scenario(
            scenario_id="test",
            split="rl_validation",
            initial_unit_ids=(1, 2, 3, 4, 5),
            initial_cycles=(1, 1, 1, 1, 1),
            replacement_seed=123,
            environment_seed=456,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )
        s2 = Scenario(
            scenario_id="test",
            split="rl_validation",
            initial_unit_ids=(1, 2, 3, 4, 5),
            initial_cycles=(1, 1, 1, 1, 1),
            replacement_seed=123,
            environment_seed=456,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )
        assert s1.compute_hash() == s2.compute_hash()

    def test_different_scenarios_produce_different_hashes(self) -> None:
        """Different scenarios should produce different hashes."""
        s1 = Scenario(
            scenario_id="test1",
            split="rl_validation",
            initial_unit_ids=(1, 2, 3, 4, 5),
            initial_cycles=(1, 1, 1, 1, 1),
            replacement_seed=123,
            environment_seed=456,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )
        s2 = Scenario(
            scenario_id="test2",  # Different ID
            split="rl_validation",
            initial_unit_ids=(1, 2, 3, 4, 5),
            initial_cycles=(1, 1, 1, 1, 1),
            replacement_seed=123,
            environment_seed=456,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )
        assert s1.compute_hash() != s2.compute_hash()


class TestScenarioBankValidation:
    """Test ScenarioBank validation."""

    def test_bank_with_mixed_splits_raises(self) -> None:
        """Bank with scenarios from different splits should raise."""
        s1 = Scenario(
            scenario_id="s1",
            split="rl_validation",
            initial_unit_ids=(1, 2, 3, 4, 5),
            initial_cycles=(1, 1, 1, 1, 1),
            replacement_seed=123,
            environment_seed=456,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )
        s2 = Scenario(
            scenario_id="s2",
            split="rl_test",  # Different split!
            initial_unit_ids=(10, 20, 30, 40, 50),
            initial_cycles=(1, 1, 1, 1, 1),
            replacement_seed=789,
            environment_seed=101112,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )

        with pytest.raises(ValueError, match="has split.*expected"):
            ScenarioBank(bank_id="mixed_bank", split="rl_validation", scenarios=(s1, s2))


class TestSaveAndLoadScenarioBank:
    """Test saving and loading scenario banks."""

    def test_save_and_load_roundtrip(self, tmp_path: Any) -> None:
        """Round-trip save/load should preserve data."""
        s1 = Scenario(
            scenario_id="test_001",
            split="rl_validation",
            initial_unit_ids=(1, 2, 3, 4, 5),
            initial_cycles=(1, 20, 40, 60, 80),
            replacement_seed=123,
            environment_seed=456,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )
        bank = ScenarioBank(bank_id="test_bank", split="rl_validation", scenarios=(s1,))

        path = tmp_path / "test_bank.json"
        save_scenario_bank(bank, path)

        loaded_bank = load_scenario_bank(path)
        assert loaded_bank.bank_id == bank.bank_id
        assert loaded_bank.split == bank.split
        assert len(loaded_bank.scenarios) == len(bank.scenarios)
        assert loaded_bank.scenarios[0].scenario_id == s1.scenario_id


class TestValidateCyclesExist:
    """Test cycle existence validation."""

    def test_missing_cycle_raises(self) -> None:
        """Missing cycle should raise ValueError."""
        # Mock store that returns found=False for cycle 999
        class BadStore(MockPredictionStore):
            def get(
                self,
                split: str,
                unit_id: int,
                cycle: int,
            ) -> "MockPredictionResult":
                if cycle == 999:
                    return MockPredictionResult(found=False)
                return super().get(split, unit_id, cycle)

        store = BadStore({1: "rl_validation", 2: "rl_validation", 3: "rl_validation", 4: "rl_validation", 5: "rl_validation"})

        scenario = Scenario(
            scenario_id="test",
            split="rl_validation",
            initial_unit_ids=(1, 2, 3, 4, 5),
            initial_cycles=(1, 1, 999, 1, 1),  # cycle 999 doesn't exist
            replacement_seed=123,
            environment_seed=456,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )

        with pytest.raises(ValueError, match="Missing prediction"):
            validate_scenario_cycles_exist(scenario, store)


class TestProductionPredictionStoreIntegration:
    """Integration tests with the actual PredictionStore."""

    def test_prediction_store_has_required_methods(self) -> None:
        """Production PredictionStore should have get() and get_units() methods."""
        from src.predictors.prediction_store import PredictionStore

        # Check that PredictionStore has the required methods
        assert hasattr(PredictionStore, "get")
        assert hasattr(PredictionStore, "get_units")

    def test_prediction_store_get_units_returns_list(self, tmp_path: Any) -> None:
        """Production PredictionStore.get_units() should return a list."""
        from src.predictors.prediction_store import PredictionStore

        cache_path = Path("data/processed/fd001/v2/06_PREDICTIONS/fd001_prediction_cache_v2.parquet")

        if cache_path.exists():
            store = PredictionStore(cache_path)
            units = store.get_units("rl_validation")
            assert isinstance(units, list)
            assert len(units) > 0
            # All units should be integers
            assert all(isinstance(u, int) for u in units)

    def test_prediction_store_get_returns_result_with_found(self, tmp_path: Any) -> None:
        """Production PredictionStore.get() should return object with 'found' attribute."""
        from src.predictors.prediction_store import PredictionStore

        cache_path = Path("data/processed/fd001/v2/06_PREDICTIONS/fd001_prediction_cache_v2.parquet")

        if cache_path.exists():
            store = PredictionStore(cache_path)
            # Unit 12 is in rl_validation split
            # Get a valid prediction
            result = store.get("rl_validation", 12, 1)
            assert hasattr(result, "found")
            assert result.found is True
            assert result.split == "rl_validation"
            assert result.unit_id == 12
            assert result.cycle == 1

            # Get an invalid prediction
            result = store.get("rl_validation", 99999, 1)  # Non-existent unit
            assert result.found is False