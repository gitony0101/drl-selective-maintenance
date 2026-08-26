"""
Test Milestone 2 environment integration.

Comprehensive integration tests covering:
- Gymnasium interface compliance
- Full episode rollout
- All components working together
- Edge cases and boundary conditions
"""

import numpy as np
import pytest

pytestmark = pytest.mark.requires_external_assets
from dataclasses import replace
from pathlib import Path

from src.envs import (
    SelectiveMaintenanceEnv,
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


class TestGymnasiumInterface:
    """Test Gymnasium interface compliance."""

    def test_env_inherits_gymnasium_env(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Environment should inherit from gymnasium.Env."""
        import gymnasium as gym

        config = get_default_config(
            split="rl_validation",
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_scenario_bank,
        )

        assert isinstance(env, gym.Env)

    def test_gymnasium_env_checker(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Gymnasium env_checker.check_env should pass."""
        from gymnasium.utils.env_checker import check_env

        config = get_default_config(
            split="rl_validation",
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_scenario_bank,
        )

        # check_env will raise if the environment doesn't comply
        check_env(env)

    def test_action_space_is_discrete(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Action space should be Discrete."""
        import gymnasium as gym

        config = get_default_config(
            split="rl_validation",
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_scenario_bank,
        )

        assert isinstance(env.action_space, gym.spaces.Discrete)

    def test_observation_space_is_box(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Observation space should be Box."""
        import gymnasium as gym

        config = get_default_config(
            split="rl_validation",
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_scenario_bank,
        )

        assert isinstance(env.observation_space, gym.spaces.Box)

    def test_observation_space_shape_is_10(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Observation space shape should be (10,)."""
        config = get_default_config(
            split="rl_validation",
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_scenario_bank,
        )

        assert env.observation_space.shape == (10,)

    def test_observation_space_bounds(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Observation space should be bounded [0, 1]."""
        config = get_default_config(
            split="rl_validation",
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_scenario_bank,
        )

        assert np.all(env.observation_space.low == 0.0)
        assert np.all(env.observation_space.high == 1.0)

    def test_get_action_mask(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """get_action_mask() should return all-True boolean array."""
        config = get_default_config(
            split="rl_validation",
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=validation_scenario_bank,
        )

        mask = env.get_action_mask()

        assert mask.shape == (env.action_space.n,)
        assert mask.dtype == bool
        assert np.all(mask == True)


class TestFullEpisode:
    """Test full episode rollout."""

    def test_full_100_step_episode(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Should be able to run 100 steps."""
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
        assert obs.shape == (10,)

        total_reward = 0.0
        num_steps = 0

        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            num_steps += 1

            # Validate observation
            assert obs.shape == (10,)
            assert np.all(np.isfinite(obs))
            assert np.all(obs >= 0.0)
            assert np.all(obs <= 1.0)

            # Validate info
            assert "step_index" in info
            assert "action_id" in info
            assert "total_cost" in info
            assert "reward" in info

        # After 100 steps, should be truncated
        assert truncated is True
        assert terminated is False
        assert num_steps == 100

    def test_episode_return_tracked(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Episode return should be tracked."""
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

        total_reward = 0.0
        for _ in range(10):
            _, reward, _, _, _ = env.step(0)
            total_reward += reward

        # Episode return should match sum of rewards
        assert env._fleet_state.episode_return == total_reward


class TestK1K2Support:
    """Test K=1 and K=2 support."""

    def test_k1_full_episode(
        self,
        validation_k1_scenario_bank,
        prediction_store,
    ) -> None:
        """K=1 should support full episodes."""
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

        assert env.action_space.n == 6  # K=1 has 6 actions

        obs, _ = env.reset()
        assert obs.shape == (10,)

        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == (10,)

    def test_k2_full_episode(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """K=2 should support full episodes."""
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

        assert env.action_space.n == 16  # K=2 has 16 actions

        obs, _ = env.reset()
        assert obs.shape == (10,)

        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == (10,)


class TestAllCostRegimes:
    """Test all cost regimes work in integration."""

    def test_all_regimes_run_full_episodes(
        self,
        prediction_store,
    ) -> None:
        """All four cost regimes should support full episodes."""
        from src.envs.scenario_bank import Scenario, ScenarioBank

        regime_ids = [
            "failure-light-no-waste",
            "failure-heavy-no-waste",
            "failure-light-waste-aware",
            "failure-heavy-waste-aware",
        ]

        train_units = prediction_store.get_units("predictor_train")

        for regime_id in regime_ids:
            # Create an in-memory scenario for this regime
            scenario = Scenario(
                scenario_id=f"test_regime_{regime_id}",
                split="predictor_train",
                initial_unit_ids=(train_units[0], train_units[1], train_units[2], train_units[3], train_units[4]),
                initial_cycles=(1, 1, 1, 1, 1),
                replacement_seed=6521,
                environment_seed=6521,
                episode_horizon=100,
                maintenance_capacity=2,
                cost_regime_id=regime_id,
            )

            scenario_bank = ScenarioBank(
                bank_id=f"test_regime_{regime_id}_bank",
                split="predictor_train",
                scenarios=(scenario,),
            )

            # Config must match scenario's cost_regime_id
            config = get_default_config(
                split="predictor_train",
                cost_regime_id=regime_id,
                scenario_bank_path="data/scenario_banks/predictor_train_smoke.json",
            )
            env = SelectiveMaintenanceEnv(
                config=config,
                prediction_store=prediction_store,
                scenario_bank=scenario_bank,
            )

            obs, _ = env.reset()
            for _ in range(10):
                obs, reward, terminated, truncated, info = env.step(1)
                assert np.isfinite(reward)
                assert info["total_cost"] >= 0


class TestHorizonTruncation:
    """Test horizon truncation."""

    def test_horizon_1_truncates_after_one_step(
        self,
        prediction_store,
    ) -> None:
        """Horizon=1 should truncate after 1 step."""
        from src.envs.scenario_bank import Scenario, ScenarioBank

        # Create scenario with horizon=1
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

        # Use dataclasses.replace to override episode_horizon
        base_config = get_default_config(
            split="rl_validation",
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )
        config = replace(base_config, episode_horizon=1)

        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=scenario_bank,
        )

        env.reset()
        _, _, terminated, truncated, _ = env.step(0)

        assert truncated is True
        assert terminated is False

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

        for i in range(99):
            _, _, terminated, truncated, _ = env.step(0)
            assert truncated is False, f"Should not truncate at step {i+1}"

        # 100th step should truncate
        _, _, terminated, truncated, _ = env.step(0)
        assert truncated is True
        assert terminated is False


class TestTerminatedAlwaysFalse:
    """Test that terminated is always False."""

    def test_terminated_always_false(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """terminated should be False for all normal steps."""
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

        for _ in range(50):
            _, _, terminated, _, _ = env.step(0)
            assert terminated is False, "terminated should always be False"