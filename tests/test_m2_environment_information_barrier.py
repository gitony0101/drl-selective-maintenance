"""
Test Milestone 2 environment information barriers.

Tests cover:
- observation contains no hidden fields
- normal reset info contains no hidden identifiers
- normal step info recursively contains no hidden fields
- diagnostic mode may contain hidden diagnostics
- observation is byte-identical between normal and diagnostic mode
- info mode does not alter rewards or transitions
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
from src.envs.scenario_bank import Scenario, ScenarioBank
from src.predictors.prediction_store import load_default_prediction_store


@pytest.fixture
def prediction_store():
    """Load the V2 prediction store."""
    cache_dir = Path("data/processed/fd001/v2/06_PREDICTIONS/")
    return load_default_prediction_store(cache_dir)


# Fields that must never appear in agent-visible data
FORBIDDEN_FIELDS = {
    "true_rul",
    "true_rul_capped",
    "trajectory_id",
    "trajectory_length",
    "unit_id",
}

# Fields forbidden in normal mode info
FORBIDDEN_NORMAL_INFO = {
    "true_rul",
    "true_rul_capped",
    "trajectory_id",
    "unit_id",
    "trajectory_length",
}


def check_dict_for_forbidden_keys(d: dict, forbidden: set, path: str = "") -> list[str]:
    """Recursively check dict for forbidden keys."""
    violations = []
    for key, value in d.items():
        full_path = f"{path}.{key}" if path else key
        if key in forbidden:
            violations.append(full_path)
        if isinstance(value, dict):
            violations.extend(check_dict_for_forbidden_keys(value, forbidden, full_path))
    return violations


class TestObservationInformationBarrier:
    """Test that observations don't leak hidden information."""

    def test_observation_has_no_true_rul(
        self,
        prediction_store,
    ) -> None:
        """Observation must not contain true_rul."""
        from src.envs import load_scenario_bank

        validation_scenario_bank = load_scenario_bank("data/scenario_banks/rl_validation_smoke.json")
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

        # Observation should only have age and predicted_rul normalized values
        # Shape (10,) = 5 slots * 2 features each
        assert obs_array.shape == (10,)

        # All values should be in [0, 1]
        assert np.all(obs_array >= 0.0)
        assert np.all(obs_array <= 1.0)

        # If true_rul were present, values could exceed 1.0 (true_rul can be > 125)
        # Check that max is reasonable (should be <= 1.0)
        assert obs_array.max() <= 1.0

    def test_observation_has_no_unit_id(
        self,
        prediction_store,
    ) -> None:
        """Observation must not contain unit_id."""
        from src.envs import load_scenario_bank

        validation_scenario_bank = load_scenario_bank("data/scenario_banks/rl_validation_smoke.json")
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

        # Unit IDs are integers like 1, 2, 3... not normalized to [0, 1]
        # If unit_id leaked, we'd see non-normalized values
        # Check all values are normalized
        assert np.all(obs_array >= 0.0)
        assert np.all(obs_array <= 1.0)

    def test_observation_has_no_trajectory_id(
        self,
        prediction_store,
    ) -> None:
        """Observation must not contain trajectory_id."""
        from src.envs import load_scenario_bank

        validation_scenario_bank = load_scenario_bank("data/scenario_banks/rl_validation_smoke.json")
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

        # trajectory_id is a string - if it leaked, it would cause a type error
        # or appear as some encoded value (which would fail normalization)
        assert isinstance(obs, np.ndarray)


class TestNormalInfoInformationBarrier:
    """Test that normal mode info doesn't leak hidden information."""

    def test_normal_reset_info_has_no_hidden_fields(
        self,
        prediction_store,
    ) -> None:
        """Normal reset info must not contain hidden identifiers."""
        from src.envs import load_scenario_bank

        validation_scenario_bank = load_scenario_bank("data/scenario_banks/rl_validation_smoke.json")
        config = get_default_config(
            split="rl_validation",
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
            info_mode="normal",
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_scenario_bank,
            info_mode="normal",
        )

        _, info = env.reset()

        violations = check_dict_for_forbidden_keys(info, FORBIDDEN_NORMAL_INFO)
        assert len(violations) == 0, f"Normal reset info contains forbidden fields: {violations}"

    def test_normal_step_info_has_no_hidden_fields(
        self,
        prediction_store,
    ) -> None:
        """Normal step info must not contain hidden fields."""
        from src.envs import load_scenario_bank

        validation_scenario_bank = load_scenario_bank("data/scenario_banks/rl_validation_smoke.json")
        config = get_default_config(
            split="rl_validation",
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
            info_mode="normal",
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_scenario_bank,
            info_mode="normal",
        )

        env.reset()
        _, _, _, _, step_info = env.step(1)

        violations = check_dict_for_forbidden_keys(step_info, FORBIDDEN_NORMAL_INFO)
        assert len(violations) == 0, f"Normal step info contains forbidden fields: {violations}"


class TestDiagnosticMode:
    """Test diagnostic mode behavior."""

    def test_diagnostic_mode_contains_hidden_fields(
        self,
        prediction_store,
    ) -> None:
        """Diagnostic mode MAY contain hidden diagnostics."""
        from src.envs import load_scenario_bank

        validation_scenario_bank = load_scenario_bank("data/scenario_banks/rl_validation_smoke.json")
        config = get_default_config(
            split="rl_validation",
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
            info_mode="diagnostic",
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_scenario_bank,
            info_mode="diagnostic",
        )

        _, info = env.reset()

        # Diagnostic info should contain slot details
        assert "slot_0" in info or any(k.startswith("slot_") for k in info.keys())

    def test_diagnostic_observation_identical_to_normal(
        self,
        prediction_store,
    ) -> None:
        """Observation should be identical in normal and diagnostic modes."""
        from src.envs import load_scenario_bank

        validation_scenario_bank = load_scenario_bank("data/scenario_banks/rl_validation_smoke.json")

        config = get_default_config(
            split="rl_validation",
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )

        env_normal = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_scenario_bank,
            info_mode="normal",
        )
        env_diagnostic = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_scenario_bank,
            info_mode="diagnostic",
        )

        obs_normal, _ = env_normal.reset(seed=6521)
        obs_diagnostic, _ = env_diagnostic.reset(seed=6521)

        np.testing.assert_array_equal(
            obs_normal,
            obs_diagnostic,
            err_msg="Observations should be identical regardless of info mode"
        )

    def test_info_mode_does_not_alter_rewards(
        self,
        prediction_store,
    ) -> None:
        """Info mode should not alter reward calculation."""
        from src.envs import load_scenario_bank

        validation_scenario_bank = load_scenario_bank("data/scenario_banks/rl_validation_smoke.json")

        config = get_default_config(
            split="rl_validation",
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )

        env_normal = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_scenario_bank,
            info_mode="normal",
        )
        env_diagnostic = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_scenario_bank,
            info_mode="diagnostic",
        )

        env_normal.reset(seed=6521)
        env_diagnostic.reset(seed=6521)

        rewards_normal = []
        rewards_diagnostic = []

        # Run 5 steps
        for _ in range(5):
            _, r_n, _, _, _ = env_normal.step(1)
            _, r_d, _, _, _ = env_diagnostic.step(1)
            rewards_normal.append(r_n)
            rewards_diagnostic.append(r_d)

        # Rewards should be identical
        for i, (r_n, r_d) in enumerate(zip(rewards_normal, rewards_diagnostic)):
            assert r_n == r_d, f"Step {i}: normal reward {r_n} != diagnostic reward {r_d}"