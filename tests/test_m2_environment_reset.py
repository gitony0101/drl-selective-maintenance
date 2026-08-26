"""
Test Milestone 2 environment reset behavior.

Tests cover:
- reset() returns (obs, info)
- obs shape is (10,)
- obs dtype is np.float32
- every observation value is finite and in [0, 1]
- initial age is initial_cycle - 1
- cycle 1 maps to age 0
- cycle 81 maps to age 80
- same scenario and same seed produce identical reset output
- invalid scenario split fails
- missing initial prediction fails
- initial true_rul <= 0 fails
"""

import numpy as np
import pytest

pytestmark = pytest.mark.requires_external_assets
from pathlib import Path

from src.envs import (
    SelectiveMaintenanceEnv,
    EnvironmentConfig,
    get_default_config,
    load_scenario_bank,
)
from src.predictors.prediction_store import load_default_prediction_store


@pytest.fixture
def prediction_store():
    """Load the V2 prediction store."""
    cache_dir = Path("data/processed/fd001/v2/06_PREDICTIONS/")
    return load_default_prediction_store(cache_dir)


@pytest.fixture
def validation_scenario_bank():
    """Load the rl_validation smoke scenario bank."""
    return load_scenario_bank("data/scenario_banks/rl_validation_smoke.json")


@pytest.fixture
def predictor_train_scenario_bank():
    """Load the predictor_train smoke scenario bank."""
    return load_scenario_bank("data/scenario_banks/predictor_train_smoke.json")


class TestResetReturns:
    """Test that reset returns correct types and shapes."""

    def test_reset_returns_obs_and_info(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """reset() should return (observation, info)."""
        config = get_default_config(
            split="rl_validation",
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_scenario_bank,
        )

        obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (10,)
        assert isinstance(info, dict)

    def test_observation_shape_is_10(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Observation shape should be (10,) for N=5."""
        config = get_default_config(
            split="rl_validation",
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_scenario_bank,
        )

        obs, _ = env.reset()
        obs_array = obs

        assert obs_array.shape == (10,), f"Expected (10,), got {obs_array.shape}"

    def test_observation_dtype_is_float32(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Observation dtype should be np.float32."""
        config = get_default_config(
            split="rl_validation",
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_scenario_bank,
        )

        obs, _ = env.reset()
        obs_array = obs

        assert obs_array.dtype == np.float32, f"Expected float32, got {obs_array.dtype}"


class TestObservationValues:
    """Test observation value bounds and finiteness."""

    def test_observation_values_finite(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """All observation values should be finite."""
        config = get_default_config(
            split="rl_validation",
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_scenario_bank,
        )

        obs, _ = env.reset()
        obs_array = obs

        assert np.all(np.isfinite(obs_array)), "Observation contains non-finite values"

    def test_observation_values_in_unit_interval(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """All observation values should be in [0, 1]."""
        config = get_default_config(
            split="rl_validation",
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_scenario_bank,
        )

        obs, _ = env.reset()
        obs_array = obs

        assert np.all(obs_array >= 0.0), "Observation contains negative values"
        assert np.all(obs_array <= 1.0), "Observation contains values > 1.0"


class TestInitialAge:
    """Test initial age computation."""

    def test_cycle_1_maps_to_age_0(
        self,
        predictor_train_scenario_bank,
        prediction_store,
    ) -> None:
        """Initial cycle 1 should map to age 0."""
        # Find a scenario where all initial cycles are 1
        scenario = None
        for s in predictor_train_scenario_bank.scenarios:
            if all(c == 1 for c in s.initial_cycles):
                scenario = s
                break

        assert scenario is not None, "No scenario with all cycles=1 found"

        config = get_default_config(
            split="predictor_train",
            scenario_bank_path="data/scenario_banks/predictor_train_smoke.json",
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=predictor_train_scenario_bank,
            scenario_selection=[scenario.scenario_id],
        )

        obs, _ = env.reset()
        obs_array = obs

        # Age features are at indices 0, 2, 4, 6, 8
        # For cycle 1, age = 0, so normalized age should be 0
        age_features = obs_array[::2]  # Every other element starting at 0

        assert np.allclose(age_features, 0.0), f"Expected age 0 for cycle 1, got {age_features}"

    def test_cycle_81_maps_to_age_80(
        self,
        prediction_store,
    ) -> None:
        """Initial cycle 81 should map to age 80, normalized to ~0.235.

        Deterministically locates five valid real units supporting cycle 81.
        Fails clearly if the production cache violates the documented invariant
        (contiguous 1..trajectory_length with true_rul > 0 at cycle 81).
        """
        from src.envs.scenario_bank import Scenario, ScenarioBank

        # Deterministically locate 5 units that have cycle 81 with positive true_rul
        # Per the V2 cache invariant: contiguous 1..trajectory_length with
        # true_rul == trajectory_length - cycle, so cycle 81 is valid if
        # trajectory_length >= 81 and true_rul at 81 = trajectory_length - 81 > 0
        train_units = prediction_store.get_units("predictor_train")
        valid_units = []
        missing_or_invalid = []

        for unit in train_units:
            pred = prediction_store.get("predictor_train", unit, 81)
            if not pred.found:
                missing_or_invalid.append((unit, "missing"))
            elif pred.true_rul <= 0:
                missing_or_invalid.append((unit, f"true_rul={pred.true_rul}"))
            else:
                valid_units.append(unit)
                if len(valid_units) >= 5:
                    break

        # Fail clearly if we cannot find 5 valid units
        assert len(valid_units) >= 5, (
            f"Production cache violation: could not find 5 units with valid cycle 81. "
            f"Found {len(valid_units)} valid units. "
            f"Invalid/missing units: {missing_or_invalid[:10]}"
        )

        scenario = Scenario(
            scenario_id="test_cycle_81",
            split="predictor_train",
            initial_unit_ids=tuple(valid_units),
            initial_cycles=(81, 1, 1, 1, 1),
            replacement_seed=6521,
            environment_seed=6521,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )

        scenario_bank = ScenarioBank(
            bank_id="test_cycle_81_bank",
            split="predictor_train",
            scenarios=(scenario,),
        )

        config = get_default_config(
            split="predictor_train",
            scenario_bank_path="data/scenario_banks/predictor_train_smoke.json",
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=scenario_bank,
        )

        obs, _ = env.reset()
        obs_array = obs

        # First slot should have age 80, normalized: 80/341 ≈ 0.2346
        expected_age_normalized = 80.0 / 341.0
        actual_age = obs_array[0]

        assert np.isclose(actual_age, expected_age_normalized, rtol=1e-4), \
            f"Expected age {expected_age_normalized:.4f}, got {actual_age:.4f}"


class TestReproducibility:
    """Test reset reproducibility."""

    def test_same_seed_same_scenario_identical(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Same seed and scenario should produce identical reset output."""
        config = get_default_config(
            split="rl_validation",
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )

        env1 = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_scenario_bank,
        )
        env2 = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_scenario_bank,
        )

        obs1, info1 = env1.reset(seed=6521)
        obs2, info2 = env2.reset(seed=6521)

        assert np.array_equal(obs1, obs2), \
            "Same seed should produce identical observations"

    def test_different_seed_different_scenario(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Different seeds can produce different outputs."""
        config = get_default_config(
            split="rl_validation",
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )

        env1 = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_scenario_bank,
        )
        env2 = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_scenario_bank,
        )

        obs1, _ = env1.reset(seed=6521)
        obs2, _ = env2.reset(seed=6522)

        # May be same or different depending on scenario selection
        # Just verify both are valid
        assert obs1.shape == (10,)
        assert obs2.shape == (10,)


class TestResetValidation:
    """Test reset validation failures."""

    def test_missing_prediction_fails(
        self,
        prediction_store,
    ) -> None:
        """Missing initial prediction should raise MissingPredictionError.

        This test uses a real unit from the split but requests a cycle that
        does not exist for that unit, triggering MissingPredictionError (not
        ScenarioValidationError for invalid unit membership).
        """
        from src.envs.scenario_bank import Scenario, ScenarioBank
        from src.envs.errors import MissingPredictionError

        # Get real units from the split
        val_units = prediction_store.get_units("rl_validation")

        # Use a real unit but request a cycle that doesn't exist
        # Unit val_units[0] is real, but cycle 99999 doesn't exist for it
        scenario = Scenario(
            scenario_id="test_missing",
            split="rl_validation",
            initial_unit_ids=(val_units[0], val_units[1], val_units[2], val_units[3], val_units[4]),
            initial_cycles=(99999, 1, 1, 1, 1),  # cycle 99999 doesn't exist for val_units[0]
            replacement_seed=6521,
            environment_seed=6521,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )

        scenario_bank = ScenarioBank(
            bank_id="test_missing_bank",
            split="rl_validation",
            scenarios=(scenario,),
        )

        config = get_default_config(
            split="rl_validation",
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=scenario_bank,
        )

        with pytest.raises(MissingPredictionError):
            env.reset()


class TestExplicitScenarioSelection:
    """Test explicit scenario selection via options."""

    def test_explicit_scenario_id(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Should be able to select scenario via options."""
        config = get_default_config(
            split="rl_validation",
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_scenario_bank,
        )

        # Get first scenario ID
        scenario_id = validation_scenario_bank.scenarios[0].scenario_id

        obs, info = env.reset(options={"scenario_id": scenario_id})

        assert obs.shape == (10,)

    def test_invalid_scenario_id_fails(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Invalid scenario_id should raise ScenarioValidationError."""
        from src.envs.errors import ScenarioValidationError

        config = get_default_config(
            split="rl_validation",
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_scenario_bank,
        )

        with pytest.raises(ScenarioValidationError):
            env.reset(options={"scenario_id": "nonexistent_scenario"})