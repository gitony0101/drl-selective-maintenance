"""
Test Milestone 2 environment reproducibility.

Tests cover:
- same scenario + same seed produces identical reset output
- same seed reproduces replacement choices
- different replacement seeds can produce different replacement sequences
- deterministic replay: same scenario, seed, action sequence = identical transitions
"""

import numpy as np
import pytest

pytestmark = pytest.mark.requires_external_assets
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


class TestResetReproducibility:
    """Test reset reproducibility."""

    def test_same_seed_same_scenario_identical_reset(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Same seed and scenario should produce identical reset."""
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

        np.testing.assert_array_equal(
            obs1,
            obs2,
            err_msg="Same seed should produce identical observations"
        )


class TestStepReproducibility:
    """Test step reproducibility."""

    def test_same_seed_same_action_sequence_identical(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Same seed and action sequence should produce identical transitions."""
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
        obs2, _ = env2.reset(seed=6521)

        # Run same action sequence
        action_sequence = [1, 6, 0, 3, 1]
        results1 = []
        results2 = []

        for action in action_sequence:
            obs1_new, reward1, term1, trunc1, info1 = env1.step(action)
            obs2_new, reward2, term2, trunc2, info2 = env2.step(action)
            results1.append((obs1_new.copy(), reward1, term1, trunc1))
            results2.append((obs2_new.copy(), reward2, term2, trunc2))
            obs1 = obs1_new
            obs2 = obs2_new

        # Compare all results
        for i, (res1, res2) in enumerate(zip(results1, results2)):
            obs_arr1, reward1, term1, trunc1 = res1
            obs_arr2, reward2, term2, trunc2 = res2

            np.testing.assert_array_equal(
                obs_arr1, obs_arr2,
                err_msg=f"Step {i}: observations differ"
            )
            assert reward1 == reward2, f"Step {i}: rewards differ"
            assert term1 == term2, f"Step {i}: terminated differs"
            assert trunc1 == trunc2, f"Step {i}: truncated differs"


class TestReplacementReproducibility:
    """Test replacement sampling reproducibility."""

    def test_same_replacement_seed_reproduces_choices(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Same replacement seed should reproduce replacement choices."""
        # This test verifies that the replacement RNG is deterministic
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

        env1.reset(seed=6521)
        env2.reset(seed=6521)

        # Run identical action sequences that trigger replacements
        for _ in range(10):
            _, _, _, _, info1 = env1.step(1)  # Maintain slot 0
            _, _, _, _, info2 = env2.step(1)

            # Info should be identical
            assert info1["num_preventive"] == info2["num_preventive"]
            assert info1["total_cost"] == info2["total_cost"]

    def test_same_seed_same_scenario_identical_replacement_units(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Same explicit scenario + same reset seed + same action sequence
        should produce identical replacement unit identities, observations,
        rewards, failure counts, and truncation flags."""
        from src.envs.scenario_bank import Scenario, ScenarioBank

        # Use a specific scenario for reproducibility
        val_units = prediction_store.get_units("rl_validation")[:5]
        scenario = Scenario(
            scenario_id="test_repro_same_seed",
            split="rl_validation",
            initial_unit_ids=tuple(val_units),
            initial_cycles=(10, 10, 10, 10, 10),
            replacement_seed=6521,
            environment_seed=6521,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )

        scenario_bank = ScenarioBank(
            bank_id="test_repro_same_seed_bank",
            split="rl_validation",
            scenarios=(scenario,),
        )

        config = get_default_config(
            split="rl_validation",
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )

        env1 = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=scenario_bank,
            info_mode="diagnostic",
        )
        env2 = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=scenario_bank,
            info_mode="diagnostic",
        )

        obs1, _ = env1.reset(seed=6521)
        obs2, _ = env2.reset(seed=6521)

        # Record replacement unit sequences
        replacement_units_1 = []
        replacement_units_2 = []
        observations_1 = [obs1.copy()]
        observations_2 = [obs2.copy()]
        rewards_1 = []
        rewards_2 = []
        failure_counts_1 = []
        failure_counts_2 = []
        truncation_flags_1 = []
        truncation_flags_2 = []

        # Run 20 steps with deterministic actions
        action_sequence = [1, 6, 0, 3, 1, 15, 0, 2, 4, 1, 0, 6, 3, 1, 0, 7, 2, 5, 1, 0]
        for i, action in enumerate(action_sequence):
            obs1_new, reward1, term1, trunc1, info1 = env1.step(action)
            obs2_new, reward2, term2, trunc2, info2 = env2.step(action)

            # Record replacement units (slot 0 and slot 1)
            replacement_units_1.append((env1._fleet_state.slots[0].unit_id, env1._fleet_state.slots[1].unit_id))
            replacement_units_2.append((env2._fleet_state.slots[0].unit_id, env2._fleet_state.slots[1].unit_id))

            observations_1.append(obs1_new.copy())
            observations_2.append(obs2_new.copy())
            rewards_1.append(reward1)
            rewards_2.append(reward2)
            failure_counts_1.append(info1["num_failures"])
            failure_counts_2.append(info2["num_failures"])
            truncation_flags_1.append(trunc1)
            truncation_flags_2.append(trunc2)

            obs1 = obs1_new
            obs2 = obs2_new

        # Verify identical replacement unit sequences
        assert replacement_units_1 == replacement_units_2, \
            "Replacement unit sequences differ for same seed/scenario/actions"

        # Verify identical observations
        for i, (o1, o2) in enumerate(zip(observations_1, observations_2)):
            np.testing.assert_array_equal(o1, o2, err_msg=f"Observation {i} differs")

        # Verify identical rewards
        assert rewards_1 == rewards_2, "Rewards differ"

        # Verify identical failure counts
        assert failure_counts_1 == failure_counts_2, "Failure counts differ"

        # Verify identical truncation flags
        assert truncation_flags_1 == truncation_flags_2, "Truncation flags differ"

    def test_different_reset_seeds_produce_different_replacement_units(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Different reset seeds + repeated replacement-producing actions
        should produce at least one different replacement unit identity
        for a verified seed pair."""
        from src.envs.scenario_bank import Scenario, ScenarioBank

        # Use a specific scenario for controlled comparison
        val_units = prediction_store.get_units("rl_validation")[:5]
        scenario = Scenario(
            scenario_id="test_repro_diff_seed",
            split="rl_validation",
            initial_unit_ids=tuple(val_units),
            initial_cycles=(10, 10, 10, 10, 10),
            replacement_seed=6521,
            environment_seed=6521,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )

        scenario_bank = ScenarioBank(
            bank_id="test_repro_diff_seed_bank",
            split="rl_validation",
            scenarios=(scenario,),
        )

        config = get_default_config(
            split="rl_validation",
            scenario_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )

        # Seed pair that produces different replacement sequences
        seed_1 = 6521
        seed_2 = 9999

        env1 = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=scenario_bank,
        )
        env2 = SelectiveMaintenanceEnv(
            config=config,
            prediction_store=prediction_store,
            scenario_bank=scenario_bank,
        )

        env1.reset(seed=seed_1)
        env2.reset(seed=seed_2)

        # Record replacement unit identities over 20 PM actions
        replacement_units_1 = set()
        replacement_units_2 = set()

        for _ in range(20):
            # Always maintain slot 0 to trigger replacement
            _, _, _, _, info1 = env1.step(1)
            _, _, _, _, info2 = env2.step(1)

            # Record the current unit in slot 0 (after replacement)
            replacement_units_1.add(env1._fleet_state.slots[0].unit_id)
            replacement_units_2.add(env2._fleet_state.slots[0].unit_id)

        # At least one different replacement unit identity must occur
        # for this verified seed pair
        # Note: They may share some units, but the sequences should differ
        # We verify by checking that the full sequence differs, not just the set
        # Re-run with sequence tracking
        env1.reset(seed=seed_1)
        env2.reset(seed=seed_2)

        sequence_1 = []
        sequence_2 = []
        for _ in range(20):
            env1.step(1)
            env2.step(1)
            sequence_1.append(env1._fleet_state.slots[0].unit_id)
            sequence_2.append(env2._fleet_state.slots[0].unit_id)

        # Verify sequences differ (at least one position has different unit)
        assert sequence_1 != sequence_2, \
            f"Replacement sequences should differ for seeds {seed_1} vs {seed_2}"

    def test_replacement_unit_belongs_to_configured_split(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Every replacement unit must belong to the configured split."""
        from src.envs.scenario_bank import Scenario, ScenarioBank

        val_units = prediction_store.get_units("rl_validation")[:5]
        scenario = Scenario(
            scenario_id="test_split_isolation",
            split="rl_validation",
            initial_unit_ids=tuple(val_units),
            initial_cycles=(10, 10, 10, 10, 10),
            replacement_seed=6521,
            environment_seed=6521,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )

        scenario_bank = ScenarioBank(
            bank_id="test_split_isolation_bank",
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
            info_mode="diagnostic",
        )

        env.reset(seed=6521)

        # Get all valid rl_validation units
        valid_units = set(prediction_store.get_units("rl_validation"))

        # Run 50 steps with various actions, checking split isolation
        for action in [1, 6, 0, 15, 3, 7, 0, 2, 4, 1]:
            _, _, _, _, info = env.step(action)

            # Check all slots have units from the configured split
            for slot_idx in range(5):
                slot_unit = env._fleet_state.slots[slot_idx].unit_id
                assert slot_unit in valid_units, \
                    f"Slot {slot_idx} has unit {slot_unit} not in rl_validation split"

    def test_anti_repeat_replacement(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """When more than one unit exists, a slot must not immediately
        receive its just-retired unit."""
        from src.envs.scenario_bank import Scenario, ScenarioBank

        val_units = prediction_store.get_units("rl_validation")[:5]
        scenario = Scenario(
            scenario_id="test_anti_repeat",
            split="rl_validation",
            initial_unit_ids=tuple(val_units),
            initial_cycles=(10, 10, 10, 10, 10),
            replacement_seed=6521,
            environment_seed=6521,
            episode_horizon=100,
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
        )

        scenario_bank = ScenarioBank(
            bank_id="test_anti_repeat_bank",
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
            info_mode="diagnostic",
        )

        env.reset(seed=6521)

        # Perform repeated PM on slot 0 and verify no immediate re-use
        retired_units = []
        new_units = []

        for _ in range(20):
            retired_before = env._fleet_state.slots[0].unit_id
            _, _, _, _, info = env.step(1)  # Maintain slot 0
            new_unit = env._fleet_state.slots[0].unit_id

            retired_units.append(retired_before)
            new_units.append(new_unit)

            # Anti-repeat: new unit must not be the same as retired unit
            # (when pool has > 1 unit, which rl_validation does)
            assert new_unit != retired_before, \
                f"Slot 0 immediately received its just-retired unit {retired_before}"


class TestDeterministicReplay:
    """Test full deterministic replay."""

    def test_full_episode_deterministic(
        self,
        validation_scenario_bank,
        prediction_store,
    ) -> None:
        """Full episode replay should be deterministic."""
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
        obs2, _ = env2.reset(seed=6521)

        np.testing.assert_array_equal(obs1, obs2)

        # Run 20 steps with deterministic actions
        for step in range(20):
            action = step % 7  # Cycle through actions 0-6
            obs1, r1, t1, tr1, i1 = env1.step(action)
            obs2, r2, t2, tr2, i2 = env2.step(action)

            np.testing.assert_array_equal(
                obs1, obs2,
                err_msg=f"Step {step}: observations differ"
            )
            assert r1 == r2, f"Step {step}: rewards differ"
            assert t1 == t2, f"Step {step}: terminated differs"
            assert tr1 == tr2, f"Step {step}: truncated differs"
            assert i1["total_cost"] == i2["total_cost"], f"Step {step}: costs differ"