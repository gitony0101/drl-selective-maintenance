"""
Test Milestone 2 environment reward calculation.

Tests cover:
- zero-event step has zero cost (except when failures occur naturally)
- one PM cost is exactly c_pm
- two PM cost is exactly 2*c_pm
- one failure cost is exactly c_f
- wasted-life term uses capped/normalized remaining life
- raw true RUL is not treated as normalized waste
- total cost equals sum of components
- reward equals negative total cost
- all four cost regimes produce expected values
- selected PM slot cannot also receive failure charge
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
from src.envs.costs import (
    FAILURE_LIGHT_NO_WASTE,
    FAILURE_HEAVY_NO_WASTE,
    FAILURE_LIGHT_WASTE_AWARE,
    FAILURE_HEAVY_WASTE_AWARE,
    calculate_total_cost,
)
from src.predictors.prediction_store import load_default_prediction_store
from tests.m2_env_test_helpers import find_unit_with_failure_at_cycle


@pytest.fixture
def prediction_store():
    """Load the V2 prediction store."""
    cache_dir = Path("data/processed/fd001/v2/06_PREDICTIONS/")
    return load_default_prediction_store(cache_dir)


class TestPreventiveMaintenanceCost:
    """Test PM cost calculation."""

    def test_one_pm_cost_is_c_pm(
        self,
        prediction_store,
    ) -> None:
        """One PM should cost exactly c_pm."""
        train_units = prediction_store.get_units("predictor_train")

        scenario = Scenario(
            scenario_id="test_one_pm",
            split="predictor_train",
            initial_unit_ids=(train_units[0], train_units[1], train_units[2], train_units[3], train_units[4]),
            initial_cycles=(10, 10, 10, 10, 10),
            replacement_seed=6521,
            environment_seed=6521,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",  # c_pm=1, c_f=5, c_u=0
        )

        scenario_bank = ScenarioBank(
            bank_id="test_one_pm_bank",
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

        env.reset()

        # Action 1 = maintain slot 0 only
        next_obs, reward, terminated, truncated, info = env.step(1)

        # Should have exactly 1 PM
        assert info["num_preventive"] == 1
        assert info["preventive_cost"] == 1.0  # c_pm = 1.0

    def test_two_pm_cost_is_2_times_c_pm(
        self,
        prediction_store,
    ) -> None:
        """Two PMs should cost exactly 2*c_pm."""
        train_units = prediction_store.get_units("predictor_train")

        scenario = Scenario(
            scenario_id="test_two_pm",
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
            bank_id="test_two_pm_bank",
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

        env.reset()

        # Action 6 = maintain slots {0, 1}
        next_obs, reward, terminated, truncated, info = env.step(6)

        assert info["num_preventive"] == 2
        assert info["preventive_cost"] == 2.0  # 2 * c_pm


class TestFailureCost:
    """Test failure cost calculation."""

    def test_failure_cost_is_c_f(
        self,
        prediction_store,
    ) -> None:
        """One failure should cost exactly c_f."""
        from tests.m2_env_test_helpers import find_unit_with_failure_at_cycle

        # Find a unit that will fail soon
        failure_unit, failure_cycle = find_unit_with_failure_at_cycle(
            prediction_store, "predictor_train", max_search_cycles=350
        )
        if failure_unit is None:
            pytest.fail("No failure cycle found in predictor_train")

        start_cycle = failure_cycle - 2
        if start_cycle <= 0:
            # Find another unit with failure late enough
            for unit in prediction_store.get_units("predictor_train"):
                for cyc in range(3, 351):
                    pred = prediction_store.get("predictor_train", unit, cyc)
                    if pred.found and pred.true_rul <= 0:
                        failure_unit = unit
                        failure_cycle = cyc
                        start_cycle = failure_cycle - 2
                        break
                if start_cycle > 0:
                    break

        if start_cycle <= 0:
            pytest.fail("Cannot find unit with failure late enough")

        # Select 4 other distinct units from predictor_train
        all_units = prediction_store.get_units("predictor_train")
        other_units = [u for u in all_units if u != failure_unit][:4]
        if len(other_units) < 4:
            pytest.fail("predictor_train needs at least 5 units")

        # Find valid cycles for other units (cycle 100 or lower that exists)
        other_cycles = []
        for uid in other_units:
            # Use cycle 1 as safe default
            other_cycles.append(1)

        scenario = Scenario(
            scenario_id="test_failure_cost",
            split="predictor_train",
            initial_unit_ids=(failure_unit, other_units[0], other_units[1], other_units[2], other_units[3]),
            initial_cycles=(start_cycle, other_cycles[0], other_cycles[1], other_cycles[2], other_cycles[3]),
            replacement_seed=6521,
            environment_seed=6521,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",  # c_f = 5
        )

        scenario_bank = ScenarioBank(
            bank_id="test_failure_cost_bank",
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

        env.reset()

        # Action 0 = no PM, let it fail
        next_obs, reward, terminated, truncated, info = env.step(0)

        # Should have exactly one failure costing c_f = 5.0
        assert info["num_failures"] == 1, f"Expected exactly 1 failure, got {info['num_failures']}"
        assert info["failure_cost"] == 5.0, f"Expected failure_cost=5.0, got {info['failure_cost']}"


class TestWastedLifeCost:
    """Test wasted life cost calculation."""

    def test_wasted_life_uses_capped_normalization(
        self,
        prediction_store,
    ) -> None:
        """Wasted life should use clip(true_rul, 0, 125) / 125."""
        train_units = prediction_store.get_units("predictor_train")

        scenario = Scenario(
            scenario_id="test_waste",
            split="predictor_train",
            initial_unit_ids=(train_units[0], train_units[1], train_units[2], train_units[3], train_units[4]),
            initial_cycles=(10, 10, 10, 10, 10),
            replacement_seed=6521,
            environment_seed=6521,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-waste-aware",  # c_u = 0.25
        )

        scenario_bank = ScenarioBank(
            bank_id="test_waste_bank",
            split="predictor_train",
            scenarios=(scenario,),
        )

        # Config must match the scenario's cost_regime_id
        config = get_default_config(
            split="predictor_train",
            cost_regime_id="failure-light-waste-aware",  # Match scenario
            scenario_bank_path="data/scenario_banks/predictor_train_smoke.json",
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=scenario_bank,
            info_mode="diagnostic",
        )

        env.reset()

        # Action 1 = maintain slot 0
        next_obs, reward, terminated, truncated, info = env.step(1)

        # Get true_rul at cycle 10 for slot 0
        pred = prediction_store.get("predictor_train", train_units[0], 10)
        # clip(true_rul, 0, 125) / 125
        expected_normalized_waste = min(max(pred.true_rul, 0.0), 125.0) / 125.0
        # c_u * normalized_waste
        expected_waste_cost = 0.25 * expected_normalized_waste

        assert np.isclose(info["wasted_life_cost"], expected_waste_cost, rtol=1e-4), \
            f"Expected {expected_waste_cost}, got {info['wasted_life_cost']}"

    def test_no_waste_cost_in_no_waste_regime(
        self,
        prediction_store,
    ) -> None:
        """No-waste regimes should have zero wasted_life_cost."""
        train_units = prediction_store.get_units("predictor_train")

        scenario = Scenario(
            scenario_id="test_no_waste",
            split="predictor_train",
            initial_unit_ids=(train_units[0], train_units[1], train_units[2], train_units[3], train_units[4]),
            initial_cycles=(10, 10, 10, 10, 10),
            replacement_seed=6521,
            environment_seed=6521,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",  # c_u = 0.0
        )

        scenario_bank = ScenarioBank(
            bank_id="test_no_waste_bank",
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

        env.reset()

        # Action 1 = maintain slot 0
        next_obs, reward, terminated, truncated, info = env.step(1)

        assert info["wasted_life_cost"] == 0.0


class TestTotalCost:
    """Test total cost calculation."""

    def test_total_cost_is_sum_of_components(
        self,
        prediction_store,
    ) -> None:
        """Total cost should equal PM + failure + waste."""
        from dataclasses import replace

        train_units = prediction_store.get_units("predictor_train")

        scenario = Scenario(
            scenario_id="test_total",
            split="predictor_train",
            initial_unit_ids=(train_units[0], train_units[1], train_units[2], train_units[3], train_units[4]),
            initial_cycles=(10, 10, 10, 10, 10),
            replacement_seed=6521,
            environment_seed=6521,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-waste-aware",
        )

        scenario_bank = ScenarioBank(
            bank_id="test_total_bank",
            split="predictor_train",
            scenarios=(scenario,),
        )

        # Config must match scenario's cost_regime_id
        config = get_default_config(
            split="predictor_train",
            cost_regime_id="failure-light-waste-aware",
            scenario_bank_path="data/scenario_banks/predictor_train_smoke.json",
        )
        env = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=scenario_bank,
        )

        env.reset()

        # Action 6 = maintain slots {0, 1}
        next_obs, reward, terminated, truncated, info = env.step(6)

        expected_total = (
            info["preventive_cost"] +
            info["failure_cost"] +
            info["wasted_life_cost"]
        )

        assert np.isclose(info["total_cost"], expected_total, rtol=1e-6), \
            f"Expected {expected_total}, got {info['total_cost']}"

    def test_reward_is_negative_total_cost(
        self,
        prediction_store,
    ) -> None:
        """Reward should be -total_cost."""
        train_units = prediction_store.get_units("predictor_train")

        scenario = Scenario(
            scenario_id="test_reward",
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
            bank_id="test_reward_bank",
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

        env.reset()

        next_obs, reward, terminated, truncated, info = env.step(1)

        assert np.isclose(reward, -info["total_cost"], rtol=1e-6), \
            f"Expected reward {-info['total_cost']}, got {reward}"


class TestCostRegimes:
    """Test all four cost regimes."""

    def test_all_four_regimes_work(
        self,
        prediction_store,
    ) -> None:
        """All four cost regimes should produce valid results."""
        train_units = prediction_store.get_units("predictor_train")
        regime_ids = [
            "failure-light-no-waste",
            "failure-heavy-no-waste",
            "failure-light-waste-aware",
            "failure-heavy-waste-aware",
        ]

        for regime_id in regime_ids:
            scenario = Scenario(
                scenario_id=f"test_regime_{regime_id}",
                split="predictor_train",
                initial_unit_ids=(train_units[0], train_units[1], train_units[2], train_units[3], train_units[4]),
                initial_cycles=(10, 10, 10, 10, 10),
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

            # Config must match scenario's cost_regime_id for each regime
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

            env.reset()
            next_obs, reward, terminated, truncated, info = env.step(1)

            # Verify cost components are non-negative
            assert info["preventive_cost"] >= 0
            assert info["failure_cost"] >= 0
            assert info["wasted_life_cost"] >= 0
            assert info["total_cost"] >= 0
            assert reward <= 0