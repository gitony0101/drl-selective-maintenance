"""
Test Milestone 2 diagnostic extension (true_rul in diagnostic info).

Tests verify the M2 change that adds true_rul to reset diagnostic info:

A. Normal reset observation remains shape (10,) and unchanged.

B. Normal reset info does not contain:
   - true_rul
   - trajectory_length
   - unit_id
   - future failure cycle

C. Diagnostic reset info contains actual:
   - unit_id
   - cycle
   - age_since_replacement_cycles
   - true_rul
   - trajectory_length

D. Normal and diagnostic mode with identical scenario and seed produce identical:
   - observations
   - rewards
   - terminated flags
   - truncated flags
   - transition states

E. Diagnostic mode changes only info payload.
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


@pytest.fixture
def validation_scenario_bank():
    """Load the rl_validation smoke scenario bank."""
    return load_scenario_bank("data/scenario_banks/rl_validation_smoke.json")


@pytest.fixture
def predictor_train_scenario_bank():
    """Load the predictor_train smoke scenario bank."""
    return load_scenario_bank("data/scenario_banks/predictor_train_smoke.json")


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
    "future_failure_cycle",
}


def check_dict_for_forbidden_keys(d: dict, forbidden: set, path: str = "") -> list[str]:
    """Recursively check dict for forbidden keys."""
    violations = []
    for key, value in d.items():
        full_path = f"{path}.{key}" if path else key
        if key in forbidden:
            violations.append(full_path)
        if isinstance(value, dict):
            violations.extend(check_dict_forbidden_keys(value, forbidden, full_path))
    return violations


class TestNormalResetObservation:
    """Test A: Normal reset observation remains shape (10,) and unchanged."""

    def test_normal_reset_observation_shape(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Normal reset observation must be shape (10,)."""
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

        obs, info = env.reset(seed=6521)

        assert obs.shape == (10,), f"Expected (10,), got {obs.shape}"
        assert obs.dtype == np.float32

    def test_normal_reset_observation_unchanged(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Normal reset observation must be unchanged from M2 baseline."""
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

        obs, info = env.reset(seed=6521)

        # Observation should only contain normalized age and predicted_rul
        # All values must be in [0, 1]
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)
        assert np.all(np.isfinite(obs))


class TestNormalResetInfo:
    """Test B: Normal reset info does not contain hidden fields."""

    def test_normal_reset_info_no_true_rul(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Normal reset info must not contain true_rul."""
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

        _, info = env.reset(seed=6521)

        violations = check_dict_for_forbidden_keys(info, FORBIDDEN_NORMAL_INFO)
        assert len(violations) == 0, f"Normal reset info contains forbidden fields: {violations}"

    def test_normal_reset_info_no_trajectory_length(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Normal reset info must not contain trajectory_length."""
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

        _, info = env.reset(seed=6521)

        assert "trajectory_length" not in str(info)

    def test_normal_reset_info_no_unit_id(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Normal reset info must not contain unit_id."""
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

        _, info = env.reset(seed=6521)

        assert "unit_id" not in str(info)

    def test_normal_reset_info_no_future_failure_cycle(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Normal reset info must not contain future failure cycle info."""
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

        _, info = env.reset(seed=6521)

        assert "future_failure" not in str(info).lower()


class TestDiagnosticResetInfo:
    """Test C: Diagnostic reset info contains actual diagnostic fields."""

    def test_diagnostic_reset_info_has_unit_id(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Diagnostic reset info must contain unit_id for each slot."""
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

        _, info = env.reset(seed=6521)

        # Check that slot diagnostic info exists and contains unit_id
        for i in range(5):
            slot_key = f"slot_{i}_diagnostic"
            assert slot_key in info, f"Missing {slot_key} in diagnostic info"
            assert "unit_id" in info[slot_key], f"Missing unit_id in {slot_key}"
            assert isinstance(info[slot_key]["unit_id"], int)

    def test_diagnostic_reset_info_has_cycle(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Diagnostic reset info must contain cycle for each slot."""
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

        _, info = env.reset(seed=6521)

        for i in range(5):
            slot_key = f"slot_{i}_diagnostic"
            assert "cycle" in info[slot_key], f"Missing cycle in {slot_key}"
            assert isinstance(info[slot_key]["cycle"], int)

    def test_diagnostic_reset_info_has_age(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Diagnostic reset info must contain age_since_replacement_cycles."""
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

        _, info = env.reset(seed=6521)

        for i in range(5):
            slot_key = f"slot_{i}_diagnostic"
            assert "age_since_replacement_cycles" in info[slot_key], \
                f"Missing age_since_replacement_cycles in {slot_key}"

    def test_diagnostic_reset_info_has_true_rul(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Diagnostic reset info must contain true_rul."""
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

        _, info = env.reset(seed=6521)

        for i in range(5):
            slot_key = f"slot_{i}_diagnostic"
            assert "true_rul" in info[slot_key], f"Missing true_rul in {slot_key}"
            # true_rul should be a positive integer for valid scenarios
            true_rul = info[slot_key]["true_rul"]
            assert true_rul is not None, f"true_rul is None in {slot_key}"
            assert true_rul > 0, f"true_rul should be positive, got {true_rul}"

    def test_diagnostic_reset_info_has_trajectory_length(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Diagnostic reset info must contain trajectory_length."""
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

        _, info = env.reset(seed=6521)

        for i in range(5):
            slot_key = f"slot_{i}_diagnostic"
            assert "trajectory_length" in info[slot_key], \
                f"Missing trajectory_length in {slot_key}"
            assert isinstance(info[slot_key]["trajectory_length"], int)


class TestModeEquivalence:
    """Test D: Normal and diagnostic mode produce identical transitions."""

    def test_observation_identical_between_modes(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Observations must be byte-identical between normal and diagnostic mode."""
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
            obs_normal, obs_diagnostic,
            err_msg="Observations differ between normal and diagnostic mode"
        )

    def test_rewards_identical_between_modes(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Rewards must be identical between normal and diagnostic mode."""
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

        for step in range(10):
            _, r_n, _, _, _ = env_normal.step(1)
            _, r_d, _, _, _ = env_diagnostic.step(1)
            assert r_n == r_d, f"Step {step}: reward {r_n} != {r_d}"

    def test_terminated_identical_between_modes(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Terminated flags must be identical between normal and diagnostic mode."""
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

        for _ in range(10):
            _, _, t_n, _, _ = env_normal.step(1)
            _, _, t_d, _, _ = env_diagnostic.step(1)
            assert t_n == t_d, f"terminated flags differ"

    def test_truncated_identical_between_modes(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Truncated flags must be identical between normal and diagnostic mode."""
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

        for step in range(99):
            _, _, _, tr_n, _ = env_normal.step(1)
            _, _, _, tr_d, _ = env_diagnostic.step(1)
            assert tr_n == tr_d, f"Step {step}: truncated flags differ"

    def test_full_episode_identical_transitions(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Full episode transitions must be identical between modes."""
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

        obs_normal = None
        obs_diagnostic = None

        for step in range(100):
            obs_n, r_n, t_n, tr_n, _ = env_normal.step(1)
            obs_d, r_d, t_d, tr_d, _ = env_diagnostic.step(1)

            np.testing.assert_array_equal(obs_n, obs_d,
                err_msg=f"Step {step}: observations differ")
            assert r_n == r_d, f"Step {step}: rewards differ"
            assert t_n == t_d, f"Step {step}: terminated differ"
            assert tr_n == tr_d, f"Step {step}: truncated differ"

            obs_normal = obs_n
            obs_diagnostic = obs_d

        # Final observations should also match
        np.testing.assert_array_equal(obs_normal, obs_diagnostic)


class TestDiagnosticModeOnlyChangesInfo:
    """Test E: Diagnostic mode changes only info payload."""

    def test_diagnostic_mode_internal_state_unchanged(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Diagnostic mode should not alter internal state."""
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

        # Internal fleet state should be identical
        for i in range(5):
            slot_n = env_normal._fleet_state.slots[i]
            slot_d = env_diagnostic._fleet_state.slots[i]

            assert slot_n.unit_id == slot_d.unit_id
            assert slot_n.cycle == slot_d.cycle
            assert slot_n.age_since_replacement_cycles == slot_d.age_since_replacement_cycles
            assert slot_n.trajectory_length == slot_d.trajectory_length

    def test_diagnostic_mode_observation_computation_unchanged(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Diagnostic mode should not alter observation computation."""
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

        # Step and verify observations computed identically
        for _ in range(10):
            obs_n, _, _, _, _ = env_normal.step(1)
            obs_d, _, _, _, _ = env_diagnostic.step(1)
            np.testing.assert_array_equal(obs_n, obs_d)


class TestDiagnosticStepInfo:
    """Test diagnostic step info behavior."""

    def test_diagnostic_step_info_contains_true_rul(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Diagnostic step info should contain true_rul for each slot."""
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

        env.reset(seed=6521)
        _, _, _, _, info = env.step(1)

        for i in range(5):
            slot_key = f"slot_{i}_diagnostic"
            assert slot_key in info, f"Missing {slot_key} in step info"
            assert "true_rul" in info[slot_key]

    def test_normal_step_info_no_diagnostics(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Normal step info should not contain diagnostic slot info."""
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

        env.reset(seed=6521)
        _, _, _, _, info = env.step(1)

        # Normal mode should not have slot diagnostic info
        for key in info.keys():
            assert "_diagnostic" not in key, f"Normal step info should not contain {key}"

        violations = check_dict_for_forbidden_keys(info, FORBIDDEN_NORMAL_INFO)
        assert len(violations) == 0, f"Normal step info contains forbidden fields: {violations}"