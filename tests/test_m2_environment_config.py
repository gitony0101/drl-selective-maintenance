"""
Unit tests for Milestone 2 environment configuration.

Tests cover:
- Frozen defaults load
- Invalid N, K, horizon, cycle interval and split fail
- predictor_validation fails as environment split
- V1 cache path fails
- Unknown cost regime fails
- Invalid info mode fails
- K=1 and K=2 both validate
"""

import pytest

pytestmark = pytest.mark.requires_external_assets
from pathlib import Path

from src.envs.config import (
    ALLOWED_SPLITS,
    DEFAULT_AGE_SCALE_CYCLES,
    DEFAULT_DELTA_CYCLES,
    DEFAULT_ENVIRONMENT_VERSION,
    DEFAULT_EPISODE_HORIZON,
    DEFAULT_FLEET_SIZE,
    DEFAULT_INFO_MODE,
    DEFAULT_MAINTENANCE_CAPACITY,
    DEFAULT_RUL_SCALE,
    DEFAULT_SEED,
    INFO_MODES,
    EnvironmentConfig,
    get_default_config,
)


class TestFrozenDefaults:
    """Test that frozen defaults are correctly defined."""

    def test_default_fleet_size_is_5(self) -> None:
        """DEFAULT_FLEET_SIZE should be 5."""
        assert DEFAULT_FLEET_SIZE == 5

    def test_default_maintenance_capacity_is_2(self) -> None:
        """DEFAULT_MAINTENANCE_CAPACITY should be 2."""
        assert DEFAULT_MAINTENANCE_CAPACITY == 2

    def test_default_delta_cycles_is_5(self) -> None:
        """DEFAULT_DELTA_CYCLES should be 5."""
        assert DEFAULT_DELTA_CYCLES == 5

    def test_default_episode_horizon_is_100(self) -> None:
        """DEFAULT_EPISODE_HORIZON should be 100."""
        assert DEFAULT_EPISODE_HORIZON == 100

    def test_default_age_scale_cycles_is_341(self) -> None:
        """DEFAULT_AGE_SCALE_CYCLES should be 341."""
        assert DEFAULT_AGE_SCALE_CYCLES == 341

    def test_default_rul_scale_is_125(self) -> None:
        """DEFAULT_RUL_SCALE should be 125.0."""
        assert DEFAULT_RUL_SCALE == 125.0

    def test_default_info_mode_is_normal(self) -> None:
        """DEFAULT_INFO_MODE should be 'normal'."""
        assert DEFAULT_INFO_MODE == "normal"

    def test_default_seed_is_6521(self) -> None:
        """DEFAULT_SEED should be 6521."""
        assert DEFAULT_SEED == 6521

    def test_default_environment_version_is_m2_v1(self) -> None:
        """DEFAULT_ENVIRONMENT_VERSION should be 'm2_v1'."""
        assert DEFAULT_ENVIRONMENT_VERSION == "m2_v1"


class TestAllowedSplits:
    """Test allowed splits configuration."""

    def test_predictor_train_allowed(self) -> None:
        """predictor_train should be an allowed split."""
        assert "predictor_train" in ALLOWED_SPLITS

    def test_rl_validation_allowed(self) -> None:
        """rl_validation should be an allowed split."""
        assert "rl_validation" in ALLOWED_SPLITS

    def test_rl_test_allowed(self) -> None:
        """rl_test should be an allowed split."""
        assert "rl_test" in ALLOWED_SPLITS

    def test_predictor_validation_not_allowed(self) -> None:
        """predictor_validation should NOT be an allowed split."""
        assert "predictor_validation" not in ALLOWED_SPLITS

    def test_exactly_three_allowed_splits(self) -> None:
        """There should be exactly three allowed splits."""
        assert len(ALLOWED_SPLITS) == 3


class TestValidEnvironmentConfig:
    """Test valid environment configuration."""

    def test_get_default_config_returns_valid_config(self) -> None:
        """get_default_config should return a valid config."""
        config = get_default_config()
        assert config.fleet_size == DEFAULT_FLEET_SIZE
        assert config.maintenance_capacity == DEFAULT_MAINTENANCE_CAPACITY
        assert config.delta_cycles == DEFAULT_DELTA_CYCLES
        assert config.episode_horizon == DEFAULT_EPISODE_HORIZON
        assert config.age_scale_cycles == DEFAULT_AGE_SCALE_CYCLES
        assert config.rul_scale == DEFAULT_RUL_SCALE
        assert config.info_mode == DEFAULT_INFO_MODE
        assert config.seed == DEFAULT_SEED

    def test_default_config_has_rl_validation_split(self) -> None:
        """Default config should use rl_validation split."""
        config = get_default_config()
        assert config.split == "rl_validation"

    def test_k2_config_validates(self) -> None:
        """K=2 configuration should validate."""
        config = get_default_config(maintenance_capacity=2)
        assert config.maintenance_capacity == 2

    def test_k1_config_validates(self) -> None:
        """K=1 configuration should validate."""
        config = get_default_config(maintenance_capacity=1)
        assert config.maintenance_capacity == 1

    def test_predictor_train_split_validates(self) -> None:
        """predictor_train split should validate."""
        config = get_default_config(split="predictor_train")
        assert config.split == "predictor_train"

    def test_rl_test_split_validates(self) -> None:
        """rl_test split should validate."""
        config = get_default_config(split="rl_test")
        assert config.split == "rl_test"


class TestInvalidN:
    """Test that invalid N values are rejected."""

    def test_n_zero_raises(self) -> None:
        """N=0 should raise ValueError."""
        with pytest.raises(ValueError, match="fleet_size must be positive"):
            EnvironmentConfig(
                environment_version="m2_v1",
                split="rl_validation",
                fleet_size=0,
                maintenance_capacity=2,
                delta_cycles=5,
                episode_horizon=100,
                age_scale_cycles=341,
                rul_scale=125.0,
                cost_regime_id="failure-light-no-waste",
                scenario_bank_path="test.json",
                prediction_cache_path="data/processed/fd001/v2/06_PREDICTIONS/",
                info_mode="normal",
                seed=6521,
            )

    def test_n_negative_raises(self) -> None:
        """Negative N should raise ValueError."""
        with pytest.raises(ValueError, match="fleet_size must be positive"):
            EnvironmentConfig(
                environment_version="m2_v1",
                split="rl_validation",
                fleet_size=-1,
                maintenance_capacity=2,
                delta_cycles=5,
                episode_horizon=100,
                age_scale_cycles=341,
                rul_scale=125.0,
                cost_regime_id="failure-light-no-waste",
                scenario_bank_path="test.json",
                prediction_cache_path="data/processed/fd001/v2/06_PREDICTIONS/",
                info_mode="normal",
                seed=6521,
            )


class TestInvalidK:
    """Test that invalid K values are rejected."""

    def test_k_negative_raises(self) -> None:
        """Negative K should raise ValueError."""
        with pytest.raises(ValueError, match="maintenance_capacity must be non-negative"):
            get_default_config(maintenance_capacity=-1)

    def test_k_greater_than_n_raises(self) -> None:
        """K > N should raise ValueError."""
        with pytest.raises(ValueError, match="cannot exceed fleet_size"):
            get_default_config(maintenance_capacity=6)  # N=5


class TestInvalidDeltaCycles:
    """Test that invalid delta_cycles values are rejected."""

    def test_delta_cycles_zero_raises(self) -> None:
        """delta_cycles=0 should raise ValueError."""
        config_dict = get_default_config().to_dict()
        config_dict["delta_cycles"] = 0
        with pytest.raises(ValueError, match="delta_cycles must be positive"):
            EnvironmentConfig.from_dict(config_dict)

    def test_delta_cycles_negative_raises(self) -> None:
        """Negative delta_cycles should raise ValueError."""
        config_dict = get_default_config().to_dict()
        config_dict["delta_cycles"] = -5
        with pytest.raises(ValueError, match="delta_cycles must be positive"):
            EnvironmentConfig.from_dict(config_dict)


class TestInvalidEpisodeHorizon:
    """Test that invalid episode_horizon values are rejected."""

    def test_episode_horizon_zero_raises(self) -> None:
        """episode_horizon=0 should raise ValueError."""
        config_dict = get_default_config().to_dict()
        config_dict["episode_horizon"] = 0
        with pytest.raises(ValueError, match="episode_horizon must be positive"):
            EnvironmentConfig.from_dict(config_dict)

    def test_episode_horizon_negative_raises(self) -> None:
        """Negative episode_horizon should raise ValueError."""
        config_dict = get_default_config().to_dict()
        config_dict["episode_horizon"] = -100
        with pytest.raises(ValueError, match="episode_horizon must be positive"):
            EnvironmentConfig.from_dict(config_dict)


class TestInvalidSplit:
    """Test that invalid splits are rejected."""

    def test_predictor_validation_raises(self) -> None:
        """predictor_validation should be rejected as environment split."""
        with pytest.raises(ValueError, match="split must be one of"):
            get_default_config(split="predictor_validation")

    def test_unknown_split_raises(self) -> None:
        """Unknown split should raise ValueError."""
        with pytest.raises(ValueError, match="split must be one of"):
            get_default_config(split="unknown_split")

    def test_empty_string_split_raises(self) -> None:
        """Empty string split should raise ValueError."""
        with pytest.raises(ValueError, match="split must be one of"):
            get_default_config(split="")


class TestV1CachePathRejected:
    """Test that V1 cache paths are rejected."""

    def test_v1_path_raises(self) -> None:
        """V1 cache path should raise ValueError."""
        with pytest.raises(ValueError, match="V1 cache path is not accepted"):
            get_default_config(
                prediction_cache_path="data/processed/fd001/v1/06_PREDICTIONS/"
            )

    def test_path_without_v2_raises(self) -> None:
        """Path without v2 identifier should raise ValueError."""
        with pytest.raises(ValueError, match="must contain.*v2/06_PREDICTIONS"):
            get_default_config(
                prediction_cache_path="data/processed/fd001/old_cache/"
            )


class TestUnknownCostRegime:
    """Test that unknown cost regimes are rejected."""

    def test_unknown_regime_raises(self) -> None:
        """Unknown cost regime should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown cost regime"):
            get_default_config(cost_regime_id="unknown-regime")


class TestInvalidInfoMode:
    """Test that invalid info modes are rejected."""

    def test_unknown_info_mode_raises(self) -> None:
        """Unknown info mode should raise ValueError."""
        with pytest.raises(ValueError, match="info_mode must be one of"):
            get_default_config(info_mode="unknown")

    def test_empty_info_mode_raises(self) -> None:
        """Empty info mode should raise ValueError."""
        with pytest.raises(ValueError, match="info_mode must be one of"):
            get_default_config(info_mode="")

    def test_normal_mode_validates(self) -> None:
        """'normal' info mode should validate."""
        config = get_default_config(info_mode="normal")
        assert config.info_mode == "normal"

    def test_diagnostic_mode_validates(self) -> None:
        """'diagnostic' info mode should validate."""
        config = get_default_config(info_mode="diagnostic")
        assert config.info_mode == "diagnostic"


class TestAgeScaleCyclesValidation:
    """Test age_scale_cycles validation."""

    def test_wrong_age_scale_cycles_raises(self) -> None:
        """age_scale_cycles != 341 for m2_v1 should raise ValueError."""
        config_dict = get_default_config().to_dict()
        config_dict["age_scale_cycles"] = 340
        with pytest.raises(ValueError, match="age_scale_cycles must be 341"):
            EnvironmentConfig.from_dict(config_dict)

    def test_correct_age_scale_cycles_validates(self) -> None:
        """age_scale_cycles == 341 should validate."""
        config = get_default_config()
        assert config.age_scale_cycles == 341


class TestRulScaleValidation:
    """Test rul_scale validation."""

    def test_wrong_rul_scale_raises(self) -> None:
        """rul_scale != 125.0 for m2_v1 should raise ValueError."""
        config_dict = get_default_config().to_dict()
        config_dict["rul_scale"] = 100.0
        with pytest.raises(ValueError, match="rul_scale must be 125.0"):
            EnvironmentConfig.from_dict(config_dict)

    def test_correct_rul_scale_validates(self) -> None:
        """rul_scale == 125.0 should validate."""
        config = get_default_config()
        assert config.rul_scale == 125.0


class TestConfigSerialization:
    """Test configuration serialization."""

    def test_to_dict_contains_all_fields(self) -> None:
        """to_dict should contain all required fields."""
        config = get_default_config()
        d = config.to_dict()
        expected_keys = {
            "environment_version",
            "split",
            "fleet_size",
            "maintenance_capacity",
            "delta_cycles",
            "episode_horizon",
            "age_scale_cycles",
            "rul_scale",
            "cost_regime_id",
            "scenario_bank_path",
            "prediction_cache_path",
            "info_mode",
            "seed",
        }
        assert set(d.keys()) == expected_keys

    def test_from_dict_creates_valid_config(self) -> None:
        """from_dict should create a valid config."""
        config = get_default_config()
        d = config.to_dict()
        recovered = EnvironmentConfig.from_dict(d)
        assert recovered.fleet_size == config.fleet_size
        assert recovered.split == config.split

    def test_roundtrip_serialization(self) -> None:
        """Round-trip serialization should preserve values."""
        original = get_default_config()
        d = original.to_dict()
        recovered = EnvironmentConfig.from_dict(d)
        assert recovered.to_dict() == original.to_dict()


class TestInfoModes:
    """Test INFO_MODES configuration."""

    def test_normal_in_info_modes(self) -> None:
        """'normal' should be in INFO_MODES."""
        assert "normal" in INFO_MODES

    def test_diagnostic_in_info_modes(self) -> None:
        """'diagnostic' should be in INFO_MODES."""

        assert "diagnostic" in INFO_MODES

    def test_exactly_two_info_modes(self) -> None:
        """There should be exactly two info modes."""
        assert len(INFO_MODES) == 2


class TestActionSpaceSize:
    """Test get_action_space_size method."""

    def test_k2_action_space_size_is_16(self) -> None:
        """K=2 should give action space size 16."""
        config = get_default_config(maintenance_capacity=2)
        assert config.get_action_space_size() == 16

    def test_k1_action_space_size_is_6(self) -> None:
        """K=1 should give action space size 6."""
        config = get_default_config(maintenance_capacity=1)
        assert config.get_action_space_size() == 6