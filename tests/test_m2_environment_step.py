"""
Test Milestone 2 environment step behavior.

Tests cover:
- action 0 performs no preventive replacement
- selected PM slots are replaced at cycle 1
- PM slot age resets to 0
- PM slot does not advance during the same step
- non-selected healthy slots advance exactly 5 cycles
- non-selected healthy-slot age increases exactly 5
- K=1 accepts 0 or 1 selected slot
- K=2 accepts 0, 1, or 2 selected slots
- invalid action ID fails
- bool action fails
- step before reset fails
- step after truncation fails
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
    ACTION_TABLE_N5_K1,
    ACTION_TABLE_N5_K2,
)
from src.envs.errors import InvalidActionError
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
def validation_k1_scenario_bank():
    """Load the rl_validation K=1 smoke scenario bank."""
    return load_scenario_bank("data/scenario_banks/rl_validation_k1_smoke.json")


@pytest.fixture
def predictor_train_scenario_bank():
    """Load the predictor_train smoke scenario bank."""
    return load_scenario_bank("data/scenario_banks/predictor_train_smoke.json")


class TestStepBeforeReset:
    """Test that step before reset fails."""

    def test_step_before_reset_fails(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """step() called before reset() should raise InvalidActionError."""
        config = get_default_config(
            split="rl_validation",
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_scenario_bank,
        )

        with pytest.raises(InvalidActionError, match="before reset"):
            env.step(0)


class TestActionZero:
    """Test action 0 (no preventive maintenance)."""

    def test_action_0_no_pm(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Action 0 should perform no preventive maintenance."""
        config = get_default_config(
            split="rl_validation",
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_scenario_bank,
            info_mode="diagnostic",
        )

        obs, info = env.reset()

        # Take action 0 (no PM)
        next_obs, reward, terminated, truncated, step_info = env.step(0)

        # No preventive maintenance
        assert step_info["num_preventive"] == 0
        assert step_info["preventive_cost"] == 0.0
        assert step_info["selected_slots"] == []

        # All slots should have advanced (unless they failed)
        assert step_info["step_index"] == 1


class TestSelectedSlotsReplaced:
    """Test that selected slots are replaced."""

    def test_pm_slot_replaced_at_cycle_1(
        self,
        predictor_train_scenario_bank,
        prediction_store,
    ) -> None:
        """Preventively maintained slots should be replaced at cycle 1."""
        from src.envs.scenario_bank import Scenario, ScenarioBank

        train_units = prediction_store.get_units("predictor_train")
        scenario = Scenario(
            scenario_id="test_pm_replace",
            split="predictor_train",
            initial_unit_ids=(train_units[0], train_units[1], train_units[2], train_units[3], train_units[4]),
            initial_cycles=(50, 50, 50, 50, 50),
            replacement_seed=6521,
            environment_seed=6521,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )

        scenario_bank = ScenarioBank(
            bank_id="test_pm_replace_bank",
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
            info_mode="diagnostic",
        )

        env.reset()

        # Take action to maintain slot 0 (action 1 = {0})
        next_obs, reward, terminated, truncated, step_info = env.step(1)

        # Slot 0 should be preventively maintained
        assert 0 in step_info["selected_slots"]
        assert step_info["num_preventive"] == 1
        # Exact assertion: selected slot cycle == 1
        assert "slot_0_diagnostic" in step_info
        assert step_info["slot_0_diagnostic"]["cycle"] == 1, \
            f"PM slot 0 should be at cycle 1, got {step_info['slot_0_diagnostic']['cycle']}"

    def test_pm_slot_age_resets_to_0(
        self,
        predictor_train_scenario_bank,
        prediction_store,
    ) -> None:
        """Preventively maintained slot age should reset to 0."""
        from src.envs.scenario_bank import Scenario, ScenarioBank

        train_units = prediction_store.get_units("predictor_train")
        scenario = Scenario(
            scenario_id="test_pm_age",
            split="predictor_train",
            initial_unit_ids=(train_units[0], train_units[1], train_units[2], train_units[3], train_units[4]),
            initial_cycles=(50, 50, 50, 50, 50),
            replacement_seed=6521,
            environment_seed=6521,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )

        scenario_bank = ScenarioBank(
            bank_id="test_pm_age_bank",
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
            info_mode="diagnostic",
        )

        env.reset()

        # Maintain slot 0
        next_obs, reward, terminated, truncated, step_info = env.step(1)

        # Exact assertion: selected slot age_since_replacement_cycles == 0
        assert "slot_0_diagnostic" in step_info
        slot_0_info = step_info["slot_0_diagnostic"]
        # After replacement, cycle should be 1, age should be 0
        assert slot_0_info["cycle"] == 1, \
            f"PM slot 0 cycle should be 1, got {slot_0_info['cycle']}"
        # Verify age is 0 via the fleet state
        assert env._fleet_state.slots[0].age_since_replacement_cycles == 0, \
            f"PM slot 0 age should be 0, got {env._fleet_state.slots[0].age_since_replacement_cycles}"

    def test_pm_slot_unit_in_configured_split(
        self,
        predictor_train_scenario_bank,
        prediction_store,
    ) -> None:
        """Preventively maintained slot replacement unit must belong to configured split."""
        from src.envs.scenario_bank import Scenario, ScenarioBank

        train_units = prediction_store.get_units("predictor_train")
        scenario = Scenario(
            scenario_id="test_pm_split",
            split="predictor_train",
            initial_unit_ids=(train_units[0], train_units[1], train_units[2], train_units[3], train_units[4]),
            initial_cycles=(50, 50, 50, 50, 50),
            replacement_seed=6521,
            environment_seed=6521,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )

        scenario_bank = ScenarioBank(
            bank_id="test_pm_split_bank",
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
            info_mode="diagnostic",
        )

        env.reset()
        original_unit_0 = env._fleet_state.slots[0].unit_id

        # Maintain slot 0
        next_obs, reward, terminated, truncated, step_info = env.step(1)

        # Exact assertion: replacement unit belongs to configured split
        new_unit_0 = env._fleet_state.slots[0].unit_id
        all_train_units = set(prediction_store.get_units("predictor_train"))
        assert new_unit_0 in all_train_units, \
            f"Replacement unit {new_unit_0} not in predictor_train split"

    def test_pm_slot_not_advanced_same_step(
        self,
        predictor_train_scenario_bank,
        prediction_store,
    ) -> None:
        """Selected slot must not be advanced again during the same step."""
        from src.envs.scenario_bank import Scenario, ScenarioBank

        train_units = prediction_store.get_units("predictor_train")
        scenario = Scenario(
            scenario_id="test_pm_no_advance",
            split="predictor_train",
            initial_unit_ids=(train_units[0], train_units[1], train_units[2], train_units[3], train_units[4]),
            initial_cycles=(50, 50, 50, 50, 50),
            replacement_seed=6521,
            environment_seed=6521,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )

        scenario_bank = ScenarioBank(
            bank_id="test_pm_no_advance_bank",
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
            info_mode="diagnostic",
        )

        env.reset()

        # Maintain slot 0
        _, _, _, _, step_info = env.step(1)

        # Exact assertion: selected slot is at cycle 1 (not advanced)
        # If it were advanced, it would be at cycle 6
        assert step_info["slot_0_diagnostic"]["cycle"] == 1, \
            f"PM slot 0 should be at cycle 1 (not advanced), got {step_info['slot_0_diagnostic']['cycle']}"


class TestHealthySlotAdvancement:
    """Test that healthy non-PM slots advance correctly."""

    def test_healthy_slot_identity_unchanged(
        self,
        predictor_train_scenario_bank,
        prediction_store,
    ) -> None:
        """Non-selected slot unit identity must remain unchanged."""
        from src.envs.scenario_bank import Scenario, ScenarioBank

        train_units = prediction_store.get_units("predictor_train")
        scenario = Scenario(
            scenario_id="test_healthy_identity",
            split="predictor_train",
            initial_unit_ids=(train_units[0], train_units[1], train_units[2], train_units[3], train_units[4]),
            initial_cycles=(10, 10, 10, 10, 10),
            replacement_seed=6521,
            environment_seed=6521,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )

        scenario_bank = ScenarioBank(
            bank_id="test_healthy_identity_bank",
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
            info_mode="diagnostic",
        )

        env.reset()
        original_unit_1 = env._fleet_state.slots[1].unit_id

        # Maintain only slot 0, so slot 1 should advance but keep identity
        _, _, _, _, step_info = env.step(1)

        # Exact assertion: non-selected slot unit identity unchanged
        assert env._fleet_state.slots[1].unit_id == original_unit_1, \
            f"Non-selected slot 1 unit changed from {original_unit_1} to {env._fleet_state.slots[1].unit_id}"

    def test_healthy_slot_cycle_increases_by_5(
        self,
        predictor_train_scenario_bank,
        prediction_store,
    ) -> None:
        """Non-selected slot cycle must increase exactly by 5."""
        from src.envs.scenario_bank import Scenario, ScenarioBank

        train_units = prediction_store.get_units("predictor_train")
        scenario = Scenario(
            scenario_id="test_healthy_cycle",
            split="predictor_train",
            initial_unit_ids=(train_units[0], train_units[1], train_units[2], train_units[3], train_units[4]),
            initial_cycles=(10, 10, 10, 10, 10),
            replacement_seed=6521,
            environment_seed=6521,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )

        scenario_bank = ScenarioBank(
            bank_id="test_healthy_cycle_bank",
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
            info_mode="diagnostic",
        )

        env.reset()
        original_cycle_1 = env._fleet_state.slots[1].cycle

        # Maintain only slot 0, so slot 1 should advance
        _, _, _, _, step_info = env.step(1)

        # Exact assertion: non-selected slot cycle increases by exactly 5
        new_cycle_1 = env._fleet_state.slots[1].cycle
        assert new_cycle_1 == original_cycle_1 + 5, \
            f"Non-selected slot 1 cycle should increase by 5 from {original_cycle_1} to {original_cycle_1 + 5}, got {new_cycle_1}"

    def test_healthy_slot_age_increases_by_5(
        self,
        predictor_train_scenario_bank,
        prediction_store,
    ) -> None:
        """Non-selected slot age_since_replacement_cycles must increase exactly by 5."""
        from src.envs.scenario_bank import Scenario, ScenarioBank

        train_units = prediction_store.get_units("predictor_train")
        scenario = Scenario(
            scenario_id="test_healthy_age",
            split="predictor_train",
            initial_unit_ids=(train_units[0], train_units[1], train_units[2], train_units[3], train_units[4]),
            initial_cycles=(10, 10, 10, 10, 10),
            replacement_seed=6521,
            environment_seed=6521,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )

        scenario_bank = ScenarioBank(
            bank_id="test_healthy_age_bank",
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
            info_mode="diagnostic",
        )

        env.reset()
        original_age_1 = env._fleet_state.slots[1].age_since_replacement_cycles

        # Maintain only slot 0, so slot 1 should advance
        _, _, _, _, step_info = env.step(1)

        # Exact assertion: non-selected slot age increases by exactly 5
        new_age_1 = env._fleet_state.slots[1].age_since_replacement_cycles
        assert new_age_1 == original_age_1 + 5, \
            f"Non-selected slot 1 age should increase by 5 from {original_age_1} to {original_age_1 + 5}, got {new_age_1}"


class TestInvalidActions:
    """Test invalid action handling."""

    def test_invalid_action_id_fails(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Action ID >= num_actions should raise InvalidActionError."""
        config = get_default_config(
            split="rl_validation",
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_scenario_bank,
        )

        env.reset()

        # K=2 has 16 actions (0-15), so 16 should fail
        with pytest.raises(InvalidActionError):
            env.step(16)

    def test_negative_action_id_fails(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Negative action ID should raise InvalidActionError."""
        config = get_default_config(
            split="rl_validation",
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_scenario_bank,
        )

        env.reset()

        with pytest.raises(InvalidActionError):
            env.step(-1)

    def test_bool_action_fails(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Boolean action should raise InvalidActionError."""
        config = get_default_config(
            split="rl_validation",
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_scenario_bank,
        )

        env.reset()

        with pytest.raises(InvalidActionError, match="boolean"):
            env.step(True)  # type: ignore


class TestKCapacity:
    """Test K=1 and K=2 capacity."""

    def test_k1_action_table_has_6_actions(
        self,
        validation_k1_scenario_bank,
        prediction_store,
    ) -> None:
        """K=1 should have 6 actions."""
        config = get_default_config(
            split="rl_validation",
            maintenance_capacity=1,
            scenario_bank_path="data/scenario_banks/rl_validation_k1_smoke.json",
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_k1_scenario_bank,
        )

        assert env.action_space.n == 6, f"Expected 6 actions for K=1, got {env.action_space.n}"

    def test_k2_action_table_has_16_actions(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """K=2 should have 16 actions."""
        config = get_default_config(
            split="rl_validation",
            maintenance_capacity=2,
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_scenario_bank,
        )

        assert env.action_space.n == 16, f"Expected 16 actions for K=2, got {env.action_space.n}"

    def test_k1_accepts_singleton_action(
        self,
        validation_k1_scenario_bank,
        prediction_store,
    ) -> None:
        """K=1 should accept actions with 1 slot."""
        config = get_default_config(
            split="rl_validation",
            maintenance_capacity=1,
            scenario_bank_path="data/scenario_banks/rl_validation_k1_smoke.json",
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_k1_scenario_bank,
        )

        env.reset()

        # Action 1 in K=1 is {0}
        next_obs, reward, terminated, truncated, info = env.step(1)

        assert info["num_preventive"] == 1

    def test_k1_full_episode(
        self,
        validation_k1_scenario_bank,
        prediction_store,
    ) -> None:
        """K=1 should support full 100-step episodes."""
        config = get_default_config(
            split="rl_validation",
            maintenance_capacity=1,
            scenario_bank_path="data/scenario_banks/rl_validation_k1_smoke.json",
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_k1_scenario_bank,
        )

        assert env.action_space.n == 6, f"Expected 6 actions for K=1, got {env.action_space.n}"

        obs, _ = env.reset()
        assert obs.shape == (10,)

        # Run 100 steps - should truncate at step 100
        for i in range(99):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == (10,), f"Step {i+1}: observation shape error"
            assert np.all(np.isfinite(obs)), f"Step {i+1}: non-finite observation"
            assert terminated is False, f"Step {i+1}: terminated should be False"
            assert truncated is False, f"Step {i+1}: should not truncate yet"

        # Step 100 should truncate
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert truncated is True
        assert terminated is False

    def test_k2_full_episode(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """K=2 should support full 100-step episodes."""
        config = get_default_config(
            split="rl_validation",
            maintenance_capacity=2,
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_scenario_bank,
        )

        assert env.action_space.n == 16, f"Expected 16 actions for K=2, got {env.action_space.n}"

        obs, _ = env.reset()
        assert obs.shape == (10,)

        # Run 100 steps - should truncate at step 100
        for i in range(99):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == (10,), f"Step {i+1}: observation shape error"
            assert np.all(np.isfinite(obs)), f"Step {i+1}: non-finite observation"
            assert terminated is False, f"Step {i+1}: terminated should be False"
            assert truncated is False, f"Step {i+1}: should not truncate yet"

        # Step 100 should truncate
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert truncated is True
        assert terminated is False


class TestTruncation:
    """Test episode truncation."""

    def test_horizon_1_truncates_after_one_step(
        self,
        prediction_store,
    ) -> None:
        """Horizon=1 should truncate after exactly one step."""
        from src.envs.scenario_bank import Scenario, ScenarioBank

        val_units = prediction_store.get_units("rl_validation")[:5]
        scenario = Scenario(
            scenario_id="test_horizon_1",
            split="rl_validation",
            initial_unit_ids=tuple(val_units),
            initial_cycles=(1, 1, 1, 1, 1),
            replacement_seed=6521,
            environment_seed=6521,
            episode_horizon=1,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )

        scenario_bank = ScenarioBank(
            bank_id="test_horizon_1_bank",
            split="rl_validation",
            scenarios=(scenario,),
        )

        config = EnvironmentConfig(
            environment_version="m2_v1",
            split="rl_validation",
            fleet_size=5,
            maintenance_capacity=2,
            delta_cycles=5,
            episode_horizon=1,
            age_scale_cycles=341,
            rul_scale=125.0,
            cost_regime_id="failure-light-no-waste",
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
            prediction_cache_path="data/processed/fd001/v2/06_PREDICTIONS/",
            info_mode="normal",
            seed=6521,
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=scenario_bank,
        )

        env.reset()

        # First step should truncate
        next_obs, reward, terminated, truncated, info = env.step(0)
        assert truncated is True
        assert terminated is False
        assert info["step_index"] == 1

    def test_horizon_100_truncates_after_100_steps(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Horizon=100 should truncate after exactly 100 steps."""
        config = get_default_config(
            split="rl_validation",
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_scenario_bank,
        )

        env.reset()

        # Steps 1-99 should not truncate
        for i in range(99):
            _, _, terminated, truncated, _ = env.step(0)
            assert truncated is False, f"Should not truncate at step {i+1}"
            assert terminated is False, f"terminated should be False at step {i+1}"

        # Step 100 should truncate
        _, _, terminated, truncated, _ = env.step(0)
        assert truncated is True
        assert terminated is False

    def test_terminated_always_false_on_normal_step(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """terminated should be False on every normal step."""
        config = get_default_config(
            split="rl_validation",
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_scenario_bank,
        )

        env.reset()

        for i in range(50):
            _, _, terminated, _, _ = env.step(0)
            assert terminated is False, f"terminated should be False at step {i+1}"

    def test_step_after_truncation_fails(
        self,
        prediction_store,
    ) -> None:
        """step() after truncation should raise InvalidActionError."""
        from src.envs.scenario_bank import Scenario, ScenarioBank

        val_units = prediction_store.get_units("rl_validation")[:5]
        scenario = Scenario(
            scenario_id="test_horizon_1",
            split="rl_validation",
            initial_unit_ids=tuple(val_units),
            initial_cycles=(1, 1, 1, 1, 1),
            replacement_seed=6521,
            environment_seed=6521,
            episode_horizon=1,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )

        scenario_bank = ScenarioBank(
            bank_id="test_horizon_1_bank",
            split="rl_validation",
            scenarios=(scenario,),
        )

        config = EnvironmentConfig(
            environment_version="m2_v1",
            split="rl_validation",
            fleet_size=5,
            maintenance_capacity=2,
            delta_cycles=5,
            episode_horizon=1,
            age_scale_cycles=341,
            rul_scale=125.0,
            cost_regime_id="failure-light-no-waste",
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
            prediction_cache_path="data/processed/fd001/v2/06_PREDICTIONS/",
            info_mode="normal",
            seed=6521,
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=scenario_bank,
        )

        env.reset()

        # First step should truncate
        next_obs, reward, terminated, truncated, info = env.step(0)
        assert truncated is True

        # Second step should fail
        with pytest.raises(InvalidActionError, match="truncation"):
            env.step(0)