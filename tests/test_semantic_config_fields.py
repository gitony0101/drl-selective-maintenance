"""
Test for semantic config field set (Task 6).
Verifies exact treatment of each field in resolved_config_identity."""
import sys
import tempfile
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import pytest
import json
import hashlib

pytestmark = pytest.mark.requires_external_assets

from src.training.ddqn_trainer import DDQNTrainer, TrainerConfig
from src.training.ddqn_config_identity import compute_resolved_config_identity, _strip_run_specific_fields


class TestSemanticConfigFieldSet:
    """Test exact treatment of each field in semantic config identity."""

    def test_max_steps_is_in_semantic_identity(self):
        """
        max_steps MUST be in semantic identity.
        A 6000-step pilot and 100000-step formal run must have DIFFERENT identities.
        """
        config_6k = TrainerConfig(
            split="predictor_train",
            validation_split="rl_validation",
            maintenance_capacity=1,
            cost_regime_id="failure-light-no-waste",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            max_steps=6_000,  # Pilot budget
            training_seed=6521,
            output_dir="/tmp/out1",
        )

        config_100k = TrainerConfig(
            split="predictor_train",
            validation_split="rl_validation",
            maintenance_capacity=1,
            cost_regime_id="failure-light-no-waste",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            max_steps=100_000,  # Formal budget
            training_seed=6521,
            output_dir="/tmp/out2",
        )

        dict_6k = {**config_6k.to_dict(), "num_actions": config_6k.num_actions}
        dict_100k = {**config_100k.to_dict(), "num_actions": config_100k.num_actions}

        semantic_6k = _strip_run_specific_fields(dict_6k)
        semantic_100k = _strip_run_specific_fields(dict_100k)

        # max_steps MUST be in semantic config
        assert "max_steps" in semantic_6k
        assert "max_steps" in semantic_100k
        assert semantic_6k["max_steps"] == 6_000
        assert semantic_100k["max_steps"] == 100_000

        # Identities MUST differ
        id_6k = compute_resolved_config_identity(dict_6k)
        id_100k = compute_resolved_config_identity(dict_100k)

        assert id_6k != id_100k, "6000-step and 100000-step runs must have different identities"
        assert len(id_6k) == 64
        assert len(id_100k) == 64

    def test_num_actions_is_in_semantic_identity(self):
        """num_actions MUST be in semantic identity (derives from maintenance_capacity)."""
        config = TrainerConfig(
            split="predictor_train",
            validation_split="rl_validation",
            maintenance_capacity=1,
            cost_regime_id="failure-light-no-waste",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            max_steps=100_000,
            training_seed=6521,
            output_dir="/tmp/out",
        )

        config_dict = {**config.to_dict(), "num_actions": config.num_actions}
        semantic = _strip_run_specific_fields(config_dict)

        assert "num_actions" in semantic
        assert semantic["num_actions"] == 6  # K=1 -> 6 actions

    def test_run_id_is_excluded_from_semantic_identity(self):
        """run_id is run-specific, must NOT affect identity."""
        config1 = TrainerConfig(
            split="predictor_train",
            validation_split="rl_validation",
            maintenance_capacity=1,
            cost_regime_id="failure-light-no-waste",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            max_steps=100_000,
            training_seed=6521,
            output_dir="/tmp/out",
            run_id="run_001",
        )

        config2 = TrainerConfig(
            split="predictor_train",
            validation_split="rl_validation",
            maintenance_capacity=1,
            cost_regime_id="failure-light-no-waste",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            max_steps=100_000,
            training_seed=6521,
            output_dir="/tmp/out",
            run_id="run_002",  # Different run_id
        )

        dict1 = {**config1.to_dict(), "num_actions": config1.num_actions}
        dict2 = {**config2.to_dict(), "num_actions": config2.num_actions}

        id1 = compute_resolved_config_identity(dict1)
        id2 = compute_resolved_config_identity(dict2)

        assert id1 == id2, "run_id must not affect semantic identity"

    def test_output_dir_is_excluded_from_semantic_identity(self):
        """output_dir is run-specific, must NOT affect identity."""
        config1 = TrainerConfig(
            split="predictor_train",
            validation_split="rl_validation",
            maintenance_capacity=1,
            cost_regime_id="failure-light-no-waste",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            max_steps=100_000,
            training_seed=6521,
            output_dir="/tmp/out_a",
        )

        config2 = TrainerConfig(
            split="predictor_train",
            validation_split="rl_validation",
            maintenance_capacity=1,
            cost_regime_id="failure-light-no-waste",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            max_steps=100_000,
            training_seed=6521,
            output_dir="/tmp/out_b",  # Different output_dir
        )

        dict1 = {**config1.to_dict(), "num_actions": config1.num_actions}
        dict2 = {**config2.to_dict(), "num_actions": config2.num_actions}

        id1 = compute_resolved_config_identity(dict1)
        id2 = compute_resolved_config_identity(dict2)

        assert id1 == id2, "output_dir must not affect semantic identity"

    def test_device_is_excluded_from_semantic_identity(self):
        """device is runtime choice, must NOT affect identity."""
        config1 = TrainerConfig(
            split="predictor_train",
            validation_split="rl_validation",
            maintenance_capacity=1,
            cost_regime_id="failure-light-no-waste",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            max_steps=100_000,
            training_seed=6521,
            output_dir="/tmp/out",
            device="cpu",
        )

        config2 = TrainerConfig(
            split="predictor_train",
            validation_split="rl_validation",
            maintenance_capacity=1,
            cost_regime_id="failure-light-no-waste",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            max_steps=100_000,
            training_seed=6521,
            output_dir="/tmp/out",
            device="cuda",  # Different device
        )

        dict1 = {**config1.to_dict(), "num_actions": config1.num_actions}
        dict2 = {**config2.to_dict(), "num_actions": config2.num_actions}

        id1 = compute_resolved_config_identity(dict1)
        id2 = compute_resolved_config_identity(dict2)

        assert id1 == id2, "device must not affect semantic identity"

    def test_scenario_bank_paths_are_in_semantic_identity(self):
        """Scenario bank paths MUST be in semantic identity (identify WHICH data)."""
        config = TrainerConfig(
            split="predictor_train",
            validation_split="rl_validation",
            maintenance_capacity=1,
            cost_regime_id="failure-light-no-waste",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            max_steps=100_000,
            training_seed=6521,
            output_dir="/tmp/out",
        )

        config_dict = {**config.to_dict(), "num_actions": config.num_actions}
        semantic = _strip_run_specific_fields(config_dict)

        assert "training_scenario_bank_path" in semantic
        assert "validation_scenario_bank_path" in semantic
        assert semantic["training_scenario_bank_path"] == "configs/scenarios/m5_pilot_k1.json"
        assert semantic["validation_scenario_bank_path"] == "configs/scenarios/m5_validation_k1.json"

    def test_prediction_cache_manifest_path_is_in_semantic_identity(self):
        """Prediction cache manifest path MUST be in semantic identity (identifies predictor)."""
        config = TrainerConfig(
            split="predictor_train",
            validation_split="rl_validation",
            maintenance_capacity=1,
            cost_regime_id="failure-light-no-waste",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            max_steps=100_000,
            training_seed=6521,
            output_dir="/tmp/out",
        )

        config_dict = {**config.to_dict(), "num_actions": config.num_actions}
        semantic = _strip_run_specific_fields(config_dict)

        # prediction_cache_manifest_path MUST be in semantic config
        assert "prediction_cache_manifest_path" in semantic
        # prediction_cache_path (base path) is excluded
        assert "prediction_cache_path" not in semantic

    def test_all_scientific_fields_included(self):
        """Verify all scientific fields are in semantic config."""
        config = TrainerConfig(
            split="predictor_train",
            validation_split="rl_validation",
            maintenance_capacity=1,
            cost_regime_id="failure-light-no-waste",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            prediction_cache_path="data/processed/fd001/v2/06_PREDICTIONS/",
            max_steps=100_000,
            batch_size=128,
            warmup_transitions=5_000,
            update_frequency=1,
            validation_interval=5_000,
            checkpoint_interval=5_000,
            replay_capacity=100_000,
            training_seed=6521,
            validation_seed=6521,
            hidden_dim=128,
            num_hidden_layers=2,
            learning_rate=1e-4,
            gamma=0.95,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_steps=50_000,
            gradient_clip=10.0,
            target_update_interval=1_000,
            output_dir="/tmp/out",
        )

        config_dict = {**config.to_dict(), "num_actions": config.num_actions}
        semantic = _strip_run_specific_fields(config_dict)

        # All these MUST be in semantic config
        required_fields = [
            "split", "validation_split", "maintenance_capacity", "cost_regime_id",
            "episode_horizon", "training_scenario_bank_path", "validation_scenario_bank_path",
            "prediction_cache_manifest_path", "max_steps", "batch_size", "warmup_transitions",
            "update_frequency", "validation_interval", "checkpoint_interval",
            "replay_capacity", "training_seed", "validation_seed", "hidden_dim",
            "num_hidden_layers", "learning_rate", "gamma", "epsilon_start",
            "epsilon_end", "epsilon_decay_steps", "gradient_clip",
            "target_update_interval", "num_actions",
        ]

        for field in required_fields:
            assert field in semantic, f"Field '{field}' missing from semantic config"

    def test_only_run_specific_fields_excluded(self):
        """Only run-specific fields should be excluded."""
        config = TrainerConfig(
            split="predictor_train",
            validation_split="rl_validation",
            maintenance_capacity=1,
            cost_regime_id="failure-light-no-waste",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            max_steps=100_000,
            training_seed=6521,
            output_dir="/tmp/out",
            run_id="my_run",
            device="cuda",
        )

        config_dict = {**config.to_dict(), "num_actions": config.num_actions}
        semantic = _strip_run_specific_fields(config_dict)

        # These MUST NOT be in semantic
        excluded = {"output_dir", "run_id", "device", "prediction_cache_path"}
        for field in excluded:
            assert field not in semantic, f"Field '{field}' should be excluded but is in semantic config"

        # Verify all other fields present
        all_fields = set(config_dict.keys())
        expected_semantic = all_fields - excluded
        assert set(semantic.keys()) == expected_semantic

    def test_identity_recomputable_from_resolved_config_json(self):
        """Identity must be independently recomputable from resolved_config.json artifact."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            config = TrainerConfig(
                split="predictor_train",
                validation_split="rl_validation",
                maintenance_capacity=1,
                cost_regime_id="failure-light-no-waste",
                training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
                validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
                max_steps=100_000,
                training_seed=6521,
                output_dir=str(tmp_path),
            )

            trainer = DDQNTrainer(config=config)
            trainer._current_obs, _ = trainer.train_env.reset()
            for _ in range(10):
                trainer.train_step()
            trainer.save_checkpoint("checkpoint_latest.pt")
            trainer._write_artifacts()

            # Read resolved_config.json
            resolved_config_path = trainer.run_dir / "resolved_config.json"
            with open(resolved_config_path) as f:
                resolved_config = json.load(f)

            # Recompute identity from artifact
            recomputed_identity = compute_resolved_config_identity(resolved_config)

            # Read manifest identity
            manifest_path = trainer.run_dir / "run_manifest.json"
            with open(manifest_path) as f:
                manifest = json.load(f)
            manifest_identity = manifest["resolved_config_identity"]

            # Read checkpoint identity
            import torch
            ckpt_path = trainer.run_dir / "checkpoint_latest.pt"
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            checkpoint_identity = ckpt["metadata"]["resolved_config_identity"]

            # All three MUST match
            assert recomputed_identity == manifest_identity == checkpoint_identity
            assert len(recomputed_identity) == 64
            assert recomputed_identity == recomputed_identity.lower()
            assert recomputed_identity != "0" * 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])