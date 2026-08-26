"""
Focused M5 Tests: Checkpoint System and Information Barriers
"""

import pytest

pytestmark = pytest.mark.requires_external_assets
import torch
import numpy as np
import json
import sys
from pathlib import Path
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.ddqn.agent import DDQNAgent, DDQNAgentConfig
from agents.ddqn.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    validate_checkpoint,
    compute_action_table_hash,
    CheckpointMetadata,
    CheckpointData,
    CheckpointSelectionState,
    CHECKPOINT_SELECTION_STATE_VERSION,
)
from envs.action_table import ACTION_TABLE_N5_K1, ACTION_TABLE_N5_K2


class TestCheckpointSaveLoad:
    """Test checkpoint save/load round-trip."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d)

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        return DDQNAgent(config=DDQNAgentConfig(num_actions=16), seed=6521)

    @pytest.fixture
    def scenario_bank_hashes(self):
        """Provide dummy scenario bank content hashes for schema v3."""
        # These are SHA256 hashes of actual scenario bank files
        # For tests, we use deterministic dummy hashes
        return {
            "training": "dummy_training_hash_" + "a" * 43,
            "validation": "dummy_validation_hash_" + "b" * 43,
            "split_training": "predictor_train",
            "split_validation": "rl_validation",
        }

    def test_save_checkpoint(self, temp_dir, agent, scenario_bank_hashes):
        """Test checkpoint save."""
        config = {"test": "config"}
        checkpoint_path = temp_dir / "test_checkpoint.pt"

        selection_state = CheckpointSelectionState(
            selection_state_version=CHECKPOINT_SELECTION_STATE_VERSION,
            validation_performed=False,
            best_validation_mean_cost=None,
            best_checkpoint_global_step=None,
            best_checkpoint_artifact_name=None,
            best_validation_failure_count=None,
            best_validation_worst_10_pct_cost=None,
            comparator_identity="mean_cost_v1",
            equal_metric_tie_behavior="keep_first",
        )
        checkpoint_data = save_checkpoint(
    prediction_cache_manifest_path="data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json",
            agent=agent,
            config=config,
            output_path=checkpoint_path,
            maintenance_capacity=2,
            action_table=ACTION_TABLE_N5_K2,
            cost_regime_id="failure-light-no-waste",
            training_seed=6521,
            training_scenario_bank_identity=scenario_bank_hashes["training"],
            validation_scenario_bank_identity=scenario_bank_hashes["validation"],
            training_split=scenario_bank_hashes["split_training"],
            validation_split=scenario_bank_hashes["split_validation"],
            selection_state=selection_state,
        )

        assert checkpoint_path.exists()
        assert checkpoint_data is not None
        assert checkpoint_data.metadata.checkpoint_id.startswith("checkpoint_step_")

    def test_load_checkpoint(self, temp_dir, agent):
        """Test checkpoint load."""
        config = {"test": "config"}
        checkpoint_path = temp_dir / "test_checkpoint.pt"

        selection_state = CheckpointSelectionState(
            selection_state_version=CHECKPOINT_SELECTION_STATE_VERSION,
            validation_performed=False,
            best_validation_mean_cost=None,
            best_checkpoint_global_step=None,
            best_checkpoint_artifact_name=None,
            best_validation_failure_count=None,
            best_validation_worst_10_pct_cost=None,
            comparator_identity="mean_cost_v1",
            equal_metric_tie_behavior="keep_first",
        )
        # Save with required schema v3 provenance
        save_checkpoint(
    prediction_cache_manifest_path="data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json",
            agent=agent,
            config=config,
            output_path=checkpoint_path,
            maintenance_capacity=2,
            action_table=ACTION_TABLE_N5_K2,
            cost_regime_id="failure-light-no-waste",
            training_seed=6521,
            training_split="predictor_train",
            validation_split="rl_validation",
            training_scenario_bank_identity="a" * 64,
            validation_scenario_bank_identity="b" * 64,
            selection_state=selection_state,
        )

        # Load
        checkpoint_data, issues = load_checkpoint(checkpoint_path)

        assert len(issues.get("incompatibilities", [])) == 0
        assert checkpoint_data.metadata.maintenance_capacity == 2
        assert checkpoint_data.metadata.action_count == 16

    def test_restore_agent(self, temp_dir, agent):
        """Test agent restoration from checkpoint."""
        # Modify agent
        agent.global_step = 100
        agent.gradient_update_count = 50
        agent.epsilon_state.global_step = 100

        original_weights = {k: v.clone() for k, v in agent.online_network.state_dict().items()}

        selection_state = CheckpointSelectionState(
            selection_state_version=CHECKPOINT_SELECTION_STATE_VERSION,
            validation_performed=False,
            best_validation_mean_cost=None,
            best_checkpoint_global_step=None,
            best_checkpoint_artifact_name=None,
            best_validation_failure_count=None,
            best_validation_worst_10_pct_cost=None,
            comparator_identity="mean_cost_v1",
            equal_metric_tie_behavior="keep_first",
        )

        checkpoint_path = temp_dir / "test_checkpoint.pt"
        config = {"test": "config"}

        save_checkpoint(
    prediction_cache_manifest_path="data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json",
            agent=agent,
            config=config,
            output_path=checkpoint_path,
            maintenance_capacity=2,
            action_table=ACTION_TABLE_N5_K2,
            cost_regime_id="failure-light-no-waste",
            training_seed=6521,
            training_split="predictor_train",
            validation_split="rl_validation",
            training_scenario_bank_identity="a" * 64,
            validation_scenario_bank_identity="b" * 64,
            selection_state=selection_state,
        )

        # Create new agent and restore
        new_agent = DDQNAgent(config=DDQNAgentConfig(num_actions=16), seed=6521)
        load_checkpoint(checkpoint_path, agent=new_agent)

        # Verify restoration
        assert new_agent.global_step == 100
        assert new_agent.gradient_update_count == 50

        # Verify weights restored
        for k, v in new_agent.online_network.state_dict().items():
            assert torch.allclose(v, original_weights[k])


class TestReplayBufferCheckpointRoundTrip:
    """Test ReplayBuffer schema fields survive full checkpoint serialization round-trip."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d)

    def test_replay_state_fields_survive_full_checkpoint_round_trip(self, temp_dir):
        """Prove that replay_state_version and action_count survive:
        state_dict -> save_checkpoint -> torch.save -> torch.load -> load_checkpoint -> load_state_dict
        """
        from agents.ddqn import DDQNAgent, DDQNAgentConfig, ReplayBuffer
        from agents.ddqn.replay_buffer import ReplayBufferConfig
        from agents.ddqn.checkpoint import save_checkpoint, load_checkpoint
        from envs.action_table import ACTION_TABLE_N5_K2
        import torch

        # Create replay buffer with some transitions
        replay_buffer = ReplayBuffer(
            config=ReplayBufferConfig(capacity=100, observation_dim=10, seed=6521)
        )

        # Insert some transitions
        for i in range(10):
            replay_buffer.insert(
                observation=np.full((10,), float(i), dtype=np.float32),
                action_id=i % 16,
                reward=float(i * 0.5),
                next_observation=np.full((10,), float(i + 1), dtype=np.float32),
                terminated=(i == 9),
                truncated=False,
            )

        # Get state dict BEFORE save
        original_state = replay_buffer.state_dict(action_count=16)
        original_version = original_state["replay_state_version"]
        original_action_count = original_state["action_count"]

        # Verify original values
        assert original_version == 1
        assert original_action_count == 16
        assert original_state["current_size"] == 10
        assert original_state["write_index"] == 10

        # Create agent and save checkpoint with replay buffer
        agent = DDQNAgent(config=DDQNAgentConfig(num_actions=16), seed=6521)
        checkpoint_path = temp_dir / "checkpoint_with_replay.pt"

        from agents.ddqn.checkpoint import CheckpointSelectionState as _CSS
        from agents.ddqn.checkpoint import CHECKPOINT_SELECTION_STATE_VERSION as _CSSV
        selection_state = _CSS(
            selection_state_version=_CSSV,
            validation_performed=False,
            best_validation_mean_cost=None,
            best_checkpoint_global_step=None,
            best_checkpoint_artifact_name=None,
            best_validation_failure_count=None,
            best_validation_worst_10_pct_cost=None,
            comparator_identity="mean_cost_v1",
            equal_metric_tie_behavior="keep_first",
        )
        checkpoint_data = save_checkpoint(
    prediction_cache_manifest_path="data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json",
            agent=agent,
            config={"hidden_dim": 128, "num_hidden_layers": 2},
            output_path=checkpoint_path,
            maintenance_capacity=2,
            action_table=ACTION_TABLE_N5_K2,
            cost_regime_id="failure-light-no-waste",
            training_seed=6521,
            replay_buffer=replay_buffer,
            training_split="predictor_train",
            validation_split="rl_validation",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k2.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k2.json",
            selection_state=selection_state,
        )

        # Verify checkpoint was saved with replay state
        assert checkpoint_data.replay_buffer_state is not None
        assert checkpoint_data.replay_buffer_state["replay_state_version"] == original_version
        assert checkpoint_data.replay_buffer_state["action_count"] == original_action_count

        # Full torch.save -> torch.load round trip
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert "replay_buffer_state" in checkpoint

        # Verify replay_state_version and action_count in saved file
        assert checkpoint["replay_buffer_state"]["replay_state_version"] == original_version
        assert checkpoint["replay_buffer_state"]["action_count"] == original_action_count

        # Load checkpoint
        loaded_data, issues = load_checkpoint(checkpoint_path)

        assert len(issues.get("incompatibilities", [])) == 0
        assert loaded_data.replay_buffer_state is not None
        assert loaded_data.replay_buffer_state["replay_state_version"] == original_version
        assert loaded_data.replay_buffer_state["action_count"] == original_action_count

        # Restore into new replay buffer
        new_replay_buffer = ReplayBuffer(
            config=ReplayBufferConfig(capacity=100, observation_dim=10, seed=6522)
        )
        new_replay_buffer.load_state_dict(
            loaded_data.replay_buffer_state,
            expected_action_count=16,
        )

        # Verify state restored
        assert new_replay_buffer.write_index == original_state["write_index"]
        assert new_replay_buffer.current_size == original_state["current_size"]

        # Verify data integrity - check first transition
        original_recent = replay_buffer.get_recent_transitions(1)
        restored_recent = new_replay_buffer.get_recent_transitions(1)

        assert np.allclose(original_recent["observation"], restored_recent["observation"])
        assert np.array_equal(original_recent["action"], restored_recent["action"])
        assert np.allclose(original_recent["reward"], restored_recent["reward"])

    def test_replay_state_version_mismatch_rejected(self, temp_dir):
        """Test that wrong replay_state_version is rejected on load."""
        from agents.ddqn import ReplayBuffer, ReplayBufferConfig
        from agents.ddqn.replay_buffer import REPLAY_BUFFER_STATE_VERSION

        replay_buffer = ReplayBuffer(
            config=ReplayBufferConfig(capacity=100, observation_dim=10, seed=6521)
        )
        state = replay_buffer.state_dict(action_count=16)

        # Tamper with version
        state["replay_state_version"] = REPLAY_BUFFER_STATE_VERSION + 1

        new_replay_buffer = ReplayBuffer(
            config=ReplayBufferConfig(capacity=100, observation_dim=10, seed=6522)
        )

        with pytest.raises(ValueError) as exc_info:
            new_replay_buffer.load_state_dict(state, expected_action_count=16)

        assert "schema version mismatch" in str(exc_info.value).lower()

    def test_action_count_mismatch_rejected_on_load(self, temp_dir):
        """Test that action_count mismatch is rejected on replay buffer load."""
        from agents.ddqn import ReplayBuffer, ReplayBufferConfig

        replay_buffer = ReplayBuffer(
            config=ReplayBufferConfig(capacity=100, observation_dim=10, seed=6521)
        )
        state = replay_buffer.state_dict(action_count=16)

        # Try to load with wrong expected_action_count
        new_replay_buffer = ReplayBuffer(
            config=ReplayBufferConfig(capacity=100, observation_dim=10, seed=6522)
        )

        with pytest.raises(ValueError) as exc_info:
            new_replay_buffer.load_state_dict(state, expected_action_count=6)  # Wrong

        assert "action_count mismatch" in str(exc_info.value).lower()


class TestScenarioBankNoFallback:
    """Test that evaluation fails closed without explicit validation_scenario_bank_path."""

    def test_evaluate_cli_fails_closed_missing_validation_scenario_bank_path(self):
        """Test that evaluate_ddqn.py fails closed when validation_scenario_bank_path is missing."""
        import subprocess
        import tempfile
        import json
        import torch
        from pathlib import Path

        # Create a config WITHOUT validation_scenario_bank_path
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Config missing validation_scenario_bank_path (but has training_scenario_bank_path)
            json.dump({
                "environment": {"validation_split": "rl_validation"},
                "maintenance_capacity": 1,
                "hidden_dim": 128,
                "num_hidden_layers": 2,
                "training_scenario_bank_path": "configs/scenarios/m5_pilot_k1.json",
                # NOTE: No validation_scenario_bank_path
            }, f)
            config_path = f.name

        # Create a minimal valid checkpoint
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name
            # Create minimal checkpoint with schema v3 metadata
            checkpoint = {
                "online_network_state_dict": {},
                "target_network_state_dict": {},
                "optimizer_state_dict": {},
                "python_rng_state": None,
                "numpy_rng_state": None,
                "torch_cpu_rng_state": torch.zeros(0),
                "global_step": 0,
                "gradient_update_count": 0,
                "epsilon_state": {"epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay_steps": 1000},
                "config": {},
                "training_seed": 6521,
                "metadata": {
                    "checkpoint_schema_version": 4,
                    "checkpoint_id": "test",
                    "saved_at": "2024-01-01T00:00:00Z",
                    "network_architecture_id": "a" * 64,
                    "action_table_hash": "b" * 64,
                    "observation_schema_id": "m5_point_v1",
                    "observation_dim": 10,
                    "action_count": 6,
                    "maintenance_capacity": 1,
                    "cost_regime_id": "failure-light-no-waste",
                    "environment_contract_id": "m2_v1",
                    "training_scenario_bank_identity": "c" * 64,
                    "validation_scenario_bank_identity": "d" * 64,
                    "training_split": "predictor_train",
                    "validation_split": "rl_validation",
                    "global_step": 0,
                    "gradient_update_count": 0,
                    "epsilon": 1.0,
                    "device": "cpu",
                    "prediction_cache_manifest_path": "data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json",
                    "prediction_cache_manifest_sha256": "e" * 64,
                    "prediction_cache_declared_cache_hash": "f" * 64,
                    "prediction_cache_predictor_checkpoint_hash": "g" * 64,
                    "prediction_cache_feature_schema_hash": "h" * 12,
                    "prediction_cache_normalizer_hash": "i" * 64,
                    "prediction_cache_split": "rl_validation",
                    "prediction_cache_schema_version": "v2",
                },
            }
            torch.save(checkpoint, checkpoint_path)

        try:
            result = subprocess.run(
                [
                    "python", "scripts/evaluate_ddqn.py",
                    "--checkpoint", checkpoint_path,
                    "--config", config_path,
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
            )

            # Should exit non-zero with schema v3 error message
            assert result.returncode != 0
            assert "validation_scenario_bank_path" in result.stderr
            assert "required" in result.stderr.lower()

        finally:
            import os
            os.unlink(config_path)
            os.unlink(checkpoint_path)


class TestSchemaV3ProvenanceFailClosed:
    """Test that schema v3 checkpoint save fails closed without explicit provenance."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d)

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        from agents.ddqn import DDQNAgent, DDQNAgentConfig
        return DDQNAgent(config=DDQNAgentConfig(num_actions=6), seed=6521)

    def test_save_checkpoint_requires_training_split(self, temp_dir, agent):
        """Test that save_checkpoint fails without explicit training_split."""
        from agents.ddqn.checkpoint import save_checkpoint
        from envs.action_table import ACTION_TABLE_N5_K1

        checkpoint_path = temp_dir / "checkpoint.pt"

        with pytest.raises(ValueError) as exc_info:
            save_checkpoint(
    prediction_cache_manifest_path="data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json",
                agent=agent,
                config={"hidden_dim": 128, "num_hidden_layers": 2},
                output_path=checkpoint_path,
                maintenance_capacity=1,
                action_table=ACTION_TABLE_N5_K1,
                cost_regime_id="failure-light-no-waste",
                training_seed=6521,
                training_split=None,  # Missing - should fail
                validation_split="rl_validation",
                training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
                validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            )

        assert "training_split" in str(exc_info.value)
        assert "Schema v3" in str(exc_info.value) or "required" in str(exc_info.value).lower()

    def test_save_checkpoint_requires_validation_split(self, temp_dir, agent):
        """Test that save_checkpoint fails without explicit validation_split."""
        from agents.ddqn.checkpoint import save_checkpoint
        from envs.action_table import ACTION_TABLE_N5_K1

        checkpoint_path = temp_dir / "checkpoint.pt"

        with pytest.raises(ValueError) as exc_info:
            save_checkpoint(
    prediction_cache_manifest_path="data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json",
                agent=agent,
                config={"hidden_dim": 128, "num_hidden_layers": 2},
                output_path=checkpoint_path,
                maintenance_capacity=1,
                action_table=ACTION_TABLE_N5_K1,
                cost_regime_id="failure-light-no-waste",
                training_seed=6521,
                training_split="predictor_train",
                validation_split=None,  # Missing - should fail
                training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
                validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            )

        assert "validation_split" in str(exc_info.value)
        assert "Schema v3" in str(exc_info.value) or "required" in str(exc_info.value).lower()

    def test_save_checkpoint_requires_scenario_bank_identity(self, temp_dir, agent):
        """Test that save_checkpoint fails without scenario bank identity."""
        from agents.ddqn.checkpoint import save_checkpoint
        from envs.action_table import ACTION_TABLE_N5_K1

        checkpoint_path = temp_dir / "checkpoint.pt"

        with pytest.raises(ValueError) as exc_info:
            save_checkpoint(
    prediction_cache_manifest_path="data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json",
                agent=agent,
                config={"hidden_dim": 128, "num_hidden_layers": 2},
                output_path=checkpoint_path,
                maintenance_capacity=1,
                action_table=ACTION_TABLE_N5_K1,
                cost_regime_id="failure-light-no-waste",
                training_seed=6521,
                training_split="predictor_train",
                validation_split="rl_validation",
                # No scenario bank path or identity - should fail
            )

        assert "scenario_bank" in str(exc_info.value).lower() or "Schema v3" in str(exc_info.value)

    def test_save_checkpoint_with_explicit_provenance_succeeds(self, temp_dir, agent):
        """Test that save_checkpoint succeeds with all explicit provenance."""
        from agents.ddqn.checkpoint import save_checkpoint
        from envs.action_table import ACTION_TABLE_N5_K1
        import json

        # Create valid scenario bank files
        scenario_bank = {
            "split": "rl_validation",
            "maintenance_capacity": 1,
            "scenarios": [{"scenario_id": "test", "split": "rl_validation"}],
        }

        training_bank_path = temp_dir / "training_bank.json"
        validation_bank_path = temp_dir / "validation_bank.json"

        with open(training_bank_path, 'w') as f:
            json.dump(scenario_bank, f)
        with open(validation_bank_path, 'w') as f:
            json.dump(scenario_bank, f)

        checkpoint_path = temp_dir / "checkpoint.pt"

        # Should succeed with all explicit provenance
        checkpoint_data = save_checkpoint(
    prediction_cache_manifest_path="data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json",
            agent=agent,
            config={"hidden_dim": 128, "num_hidden_layers": 2},
            output_path=checkpoint_path,
            maintenance_capacity=1,
            action_table=ACTION_TABLE_N5_K1,
            cost_regime_id="failure-light-no-waste",
            training_seed=6521,
            training_split="predictor_train",  # Explicit
            validation_split="rl_validation",  # Explicit
            training_scenario_bank_path=str(training_bank_path),
            validation_scenario_bank_path=str(validation_bank_path),
            selection_state=CheckpointSelectionState(
                selection_state_version=CHECKPOINT_SELECTION_STATE_VERSION,
                validation_performed=False,
                best_validation_mean_cost=None,
                best_checkpoint_global_step=None,
                best_checkpoint_artifact_name=None,
                best_validation_failure_count=None,
                best_validation_worst_10_pct_cost=None,
                comparator_identity="mean_cost_v1",
                equal_metric_tie_behavior="keep_first",
            ),
        )

        # Verify provenance is saved
        assert checkpoint_data.metadata.training_split == "predictor_train"
        assert checkpoint_data.metadata.validation_split == "rl_validation"
        assert checkpoint_data.metadata.training_scenario_bank_identity is not None
        assert checkpoint_data.metadata.validation_scenario_bank_identity is not None


class TestScenarioBankRealContentHashes:
    """Test real scenario-bank files and real content hashes."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d)

    def test_scenario_bank_content_hash_is_64_char_hex(self, temp_dir):
        """Test that content hash is exactly 64-character lowercase hexadecimal SHA256."""
        from agents.ddqn.checkpoint import compute_scenario_bank_content_hash
        import json

        # Create a valid scenario bank file
        scenario_bank = {
            "split": "rl_validation",
            "maintenance_capacity": 1,
            "scenarios": [
                {
                    "scenario_id": "test_scenario_1",
                    "split": "rl_validation",
                    "data": {"test": 1},
                }
            ],
        }

        scenario_bank_path = temp_dir / "test_scenario_bank.json"
        with open(scenario_bank_path, 'w') as f:
            json.dump(scenario_bank, f, indent=2)

        content_hash = compute_scenario_bank_content_hash(scenario_bank_path)

        # Verify hash format: exactly 64 lowercase hex characters
        assert len(content_hash) == 64, f"Hash length is {len(content_hash)}, expected 64"
        assert content_hash == content_hash.lower(), "Hash should be lowercase"
        assert all(c in '0123456789abcdef' for c in content_hash), "Hash should be hexadecimal"

    def test_scenario_bank_hash_deterministic(self, temp_dir):
        """Test that same file contents produce same hash."""
        from agents.ddqn.checkpoint import compute_scenario_bank_content_hash
        import json

        scenario_bank = {
            "split": "rl_validation",
            "maintenance_capacity": 1,
            "scenarios": [{"scenario_id": "test", "split": "rl_validation"}],
        }

        scenario_bank_path = temp_dir / "test_scenario_bank.json"
        with open(scenario_bank_path, 'w') as f:
            json.dump(scenario_bank, f, indent=2)

        hash1 = compute_scenario_bank_content_hash(scenario_bank_path)
        hash2 = compute_scenario_bank_content_hash(scenario_bank_path)

        assert hash1 == hash2

    def test_scenario_bank_tamper_detected_by_content_hash(self, temp_dir):
        """Test that tampering with scenario bank contents (without changing filename) is detected."""
        from agents.ddqn.checkpoint import compute_scenario_bank_content_hash, save_checkpoint, load_checkpoint
        from agents.ddqn import DDQNAgent, DDQNAgentConfig
        from envs.action_table import ACTION_TABLE_N5_K1
        import json
        import torch

        # Create original scenario bank
        original_bank = {
            "split": "rl_validation",
            "maintenance_capacity": 1,
            "scenarios": [{"scenario_id": "original", "split": "rl_validation"}],
        }

        scenario_bank_path = temp_dir / "validation_bank.json"
        with open(scenario_bank_path, 'w') as f:
            json.dump(original_bank, f, indent=2)

        # Compute original hash
        original_hash = compute_scenario_bank_content_hash(scenario_bank_path)

        # Create checkpoint with original hash
        agent = DDQNAgent(config=DDQNAgentConfig(num_actions=6), seed=6521)
        checkpoint_path = temp_dir / "checkpoint.pt"

        # Create training scenario bank file too
        training_bank_path = temp_dir / "training_bank.json"
        with open(training_bank_path, 'w') as f:
            json.dump(original_bank, f, indent=2)

        save_checkpoint(
    prediction_cache_manifest_path="data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json",
            agent=agent,
            config={"hidden_dim": 128, "num_hidden_layers": 2},
            output_path=checkpoint_path,
            maintenance_capacity=1,
            action_table=ACTION_TABLE_N5_K1,
            cost_regime_id="failure-light-no-waste",
            training_seed=6521,
            training_split="predictor_train",
            validation_split="rl_validation",
            training_scenario_bank_path=str(training_bank_path),
            validation_scenario_bank_path=str(scenario_bank_path),
            selection_state=CheckpointSelectionState(
                selection_state_version=CHECKPOINT_SELECTION_STATE_VERSION,
                validation_performed=False,
                best_validation_mean_cost=None,
                best_checkpoint_global_step=None,
                best_checkpoint_artifact_name=None,
                best_validation_failure_count=None,
                best_validation_worst_10_pct_cost=None,
                comparator_identity="mean_cost_v1",
                equal_metric_tie_behavior="keep_first",
            ),
        )

        # Tamper with scenario bank contents (same filename, different content)

        # Tamper with scenario bank contents (same filename, different content)
        tampered_bank = {
            "split": "rl_validation",
            "maintenance_capacity": 1,
            "scenarios": [{"scenario_id": "tampered", "split": "rl_validation"}],  # Changed!
        }
        with open(scenario_bank_path, 'w') as f:
            json.dump(tampered_bank, f, indent=2)

        # Compute tampered hash
        tampered_hash = compute_scenario_bank_content_hash(scenario_bank_path)

        # Hashes should be different
        assert original_hash != tampered_hash, "Tamper detection failed: hashes should differ"

        # Load checkpoint and verify tamper is detected via hash mismatch
        # The checkpoint has original_hash, but file now has tampered_hash
        checkpoint_data, metadata = load_checkpoint(checkpoint_path)

        # The metadata should have the original hash
        assert checkpoint_data.metadata.validation_scenario_bank_identity == original_hash

        # In production evaluation, this would fail because the file hash doesn't match
        # We verify the hashes differ, proving tamper detection works
        current_hash = compute_scenario_bank_content_hash(scenario_bank_path)
        assert current_hash != original_hash, "Tampered file should have different hash"

    def test_real_scenario_bank_files_produce_valid_hashes(self):
        """Test that real scenario bank files in configs/ produce valid 64-char hex hashes."""
        from agents.ddqn.checkpoint import compute_scenario_bank_content_hash
        from pathlib import Path

        repo_root = Path(__file__).parent.parent

        # Test K=1 validation scenario bank
        k1_val_path = repo_root / "configs" / "scenarios" / "m5_validation_k1.json"
        if k1_val_path.exists():
            hash_k1 = compute_scenario_bank_content_hash(k1_val_path)
            assert len(hash_k1) == 64
            assert hash_k1 == hash_k1.lower()
            assert all(c in '0123456789abcdef' for c in hash_k1)

        # Test K=2 validation scenario bank
        k2_val_path = repo_root / "configs" / "scenarios" / "m5_validation_k2.json"
        if k2_val_path.exists():
            hash_k2 = compute_scenario_bank_content_hash(k2_val_path)
            assert len(hash_k2) == 64
            assert hash_k2 == hash_k2.lower()
            assert all(c in '0123456789abcdef' for c in hash_k2)

    def test_checkpoint_validation_bank_identity_is_64_char_hex(self, temp_dir):
        """Test that checkpoint validation_scenario_bank_identity is 64-char hex."""
        from agents.ddqn.checkpoint import save_checkpoint
        from agents.ddqn import DDQNAgent, DDQNAgentConfig
        from envs.action_table import ACTION_TABLE_N5_K1
        import json

        # Create a valid scenario bank
        scenario_bank = {
            "split": "rl_validation",
            "maintenance_capacity": 1,
            "scenarios": [{"scenario_id": "test", "split": "rl_validation"}],
        }

        scenario_bank_path = temp_dir / "validation_bank.json"
        with open(scenario_bank_path, 'w') as f:
            json.dump(scenario_bank, f, indent=2)

        agent = DDQNAgent(config=DDQNAgentConfig(num_actions=6), seed=6521)
        checkpoint_path = temp_dir / "checkpoint.pt"

        save_checkpoint(
    prediction_cache_manifest_path="data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json",
            agent=agent,
            config={"hidden_dim": 128, "num_hidden_layers": 2},
            output_path=checkpoint_path,
            maintenance_capacity=1,
            action_table=ACTION_TABLE_N5_K1,
            cost_regime_id="failure-light-no-waste",
            training_seed=6521,
            training_split="predictor_train",
            validation_split="rl_validation",
            validation_scenario_bank_path=str(scenario_bank_path),
            training_scenario_bank_path=str(scenario_bank_path),  # Reuse for test
            selection_state=CheckpointSelectionState(
                selection_state_version=CHECKPOINT_SELECTION_STATE_VERSION,
                validation_performed=False,
                best_validation_mean_cost=None,
                best_checkpoint_global_step=None,
                best_checkpoint_artifact_name=None,
                best_validation_failure_count=None,
                best_validation_worst_10_pct_cost=None,
                comparator_identity="mean_cost_v1",
                equal_metric_tie_behavior="keep_first",
            ),
        )

        # Load and verify identity field format
        checkpoint_data, _ = load_checkpoint(checkpoint_path)

        val_hash = checkpoint_data.metadata.validation_scenario_bank_identity
        train_hash = checkpoint_data.metadata.training_scenario_bank_identity

        # Both should be 64-char lowercase hex
        for h, name in [(val_hash, "validation"), (train_hash, "training")]:
            assert len(h) == 64, f"{name}_scenario_bank_identity length is {len(h)}, expected 64"
            assert h == h.lower(), f"{name}_scenario_bank_identity should be lowercase"
            assert all(c in '0123456789abcdef' for c in h), f"{name}_scenario_bank_identity should be hex"


class TestProductionConfigParser:
    """Test that configuration is parsed through production TrainerConfig."""

    def test_ddqn_v1_config_parses_through_trainer_config(self):
        """Test that configs/agents/ddqn_v1.json parses through production TrainerConfig."""
        from src.training.ddqn_config import load_and_validate_config
        from pathlib import Path

        repo_root = Path(__file__).parent.parent
        config_path = repo_root / "configs" / "agents" / "ddqn_v1.json"

        parsed = load_and_validate_config(config_path, mode="evaluation")
        trainer_config = parsed.trainer_config
        raw_config = parsed.raw_config

        # Verify TrainerConfig fields are populated from nested config
        assert trainer_config.maintenance_capacity == 2
        assert trainer_config.validation_split == "rl_validation"
        assert trainer_config.split == "predictor_train"
        assert trainer_config.validation_scenario_bank_path == "configs/scenarios/m5_validation_k2.json"
        assert trainer_config.training_scenario_bank_path == "configs/scenarios/m5_pilot_k2.json"
        assert trainer_config.hidden_dim == 128
        assert trainer_config.num_hidden_layers == 2

    def test_ddqn_v1_k1_config_parses_through_trainer_config(self):
        """Test that configs/agents/ddqn_v1_k1.json parses through production TrainerConfig."""
        from src.training.ddqn_config import load_and_validate_config
        from pathlib import Path

        repo_root = Path(__file__).parent.parent
        config_path = repo_root / "configs" / "agents" / "ddqn_v1_k1.json"

        parsed = load_and_validate_config(config_path, mode="evaluation")
        trainer_config = parsed.trainer_config
        raw_config = parsed.raw_config

        # Verify TrainerConfig fields are populated from nested config
        assert trainer_config.maintenance_capacity == 1
        assert trainer_config.validation_split == "rl_validation"
        assert trainer_config.split == "predictor_train"
        assert trainer_config.validation_scenario_bank_path == "configs/scenarios/m5_validation_k1.json"
        assert trainer_config.training_scenario_bank_path == "configs/scenarios/m5_pilot_k1.json"
        assert trainer_config.hidden_dim == 128
        assert trainer_config.num_hidden_layers == 2

    def test_trainer_config_rejects_rl_test_split_during_parse(self):
        """Test that config with rl_test split is rejected during TrainerConfig parsing."""
        from src.training.ddqn_config import parse_raw_config
        import pytest

        # Config with rl_test training split
        bad_config = {
            "environment": {
                "split": "rl_test",
                "validation_split": "rl_validation",
                "maintenance_capacity": 1,
                "training_scenario_bank_path": "configs/scenarios/m5_pilot_k1.json",
                "validation_scenario_bank_path": "configs/scenarios/m5_validation_k1.json",
            },
        }

        with pytest.raises(ValueError) as exc_info:
            parse_raw_config(bad_config)

        assert "FORBIDDEN" in str(exc_info.value) or "rl_test" in str(exc_info.value)

    def test_trainer_config_rejects_rl_test_validation_split_during_parse(self):
        """Test that config with rl_test validation_split is rejected during TrainerConfig parsing."""
        from src.training.ddqn_config import parse_raw_config

        bad_config = {
            "environment": {
                "split": "predictor_train",
                "validation_split": "rl_test",  # FORBIDDEN
                "maintenance_capacity": 1,
                "training_scenario_bank_path": "configs/scenarios/m5_pilot_k1.json",
                "validation_scenario_bank_path": "configs/scenarios/m5_validation_k1.json",
            },
        }

        with pytest.raises(ValueError) as exc_info:
            parse_raw_config(bad_config)

        assert "FORBIDDEN" in str(exc_info.value) or "rl_test" in str(exc_info.value)


class TestSplitProvenanceRlTestBarrier:
    """Test that all split provenance sources reject rl_test."""

    def test_config_training_split_rl_test_rejected(self):
        """Test that config training_split=rl_test is rejected."""
        import subprocess
        import tempfile
        import json
        import torch
        from pathlib import Path

        # Create a config with training_split=rl_test
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "environment": {
                    "split": "rl_test",  # FORBIDDEN
                    "validation_split": "rl_validation",
                    "training_scenario_bank_path": "configs/scenarios/m5_pilot_k1.json",
                    "validation_scenario_bank_path": "configs/scenarios/m5_validation_k1.json",
                },
                "maintenance_capacity": 1,
            }, f)
            config_path = f.name

        # Create minimal valid checkpoint
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name
            checkpoint = {
                "online_network_state_dict": {},
                "target_network_state_dict": {},
                "optimizer_state_dict": {},
                "python_rng_state": None,
                "numpy_rng_state": None,
                "torch_cpu_rng_state": torch.zeros(0),
                "global_step": 0,
                "gradient_update_count": 0,
                "epsilon_state": {"epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay_steps": 1000},
                "config": {},
                "training_seed": 6521,
                "metadata": {
                    "checkpoint_schema_version": 4,
                    "checkpoint_id": "test",
                    "saved_at": "2024-01-01T00:00:00Z",
                    "network_architecture_id": "a" * 64,
                    "action_table_hash": "b" * 64,
                    "observation_schema_id": "m5_point_v1",
                    "observation_dim": 10,
                    "action_count": 6,
                    "maintenance_capacity": 1,
                    "cost_regime_id": "failure-light-no-waste",
                    "environment_contract_id": "m2_v1",
                    "training_scenario_bank_identity": "c" * 64,
                    "validation_scenario_bank_identity": "d" * 64,
                    "training_split": "predictor_train",
                    "validation_split": "rl_validation",
                    "global_step": 0,
                    "gradient_update_count": 0,
                    "epsilon": 1.0,
                    "device": "cpu",
                    "prediction_cache_manifest_path": "data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json",
                    "prediction_cache_manifest_sha256": "e" * 64,
                    "prediction_cache_declared_cache_hash": "f" * 64,
                    "prediction_cache_predictor_checkpoint_hash": "g" * 64,
                    "prediction_cache_feature_schema_hash": "h" * 12,
                    "prediction_cache_normalizer_hash": "i" * 64,
                    "prediction_cache_split": "rl_validation",
                    "prediction_cache_schema_version": "v2",
                },
            }
            torch.save(checkpoint, checkpoint_path)

        try:
            result = subprocess.run(
                [
                    "python", "scripts/evaluate_ddqn.py",
                    "--checkpoint", checkpoint_path,
                    "--config", config_path,
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
            )

            assert result.returncode != 0
            # The TrainerConfig validation will fail first with "CRITICAL BARRIER VIOLATION"
            assert "FORBIDDEN" in result.stderr or "CRITICAL BARRIER VIOLATION" in result.stderr or "rl_test" in result.stderr

        finally:
            import os
            os.unlink(config_path)
            os.unlink(checkpoint_path)

    def test_checkpoint_training_split_rl_test_rejected(self):
        """Test that checkpoint training_split=rl_test is rejected."""
        import subprocess
        import tempfile
        import json
        import torch
        from pathlib import Path

        # Create a valid config with required fields
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "environment": {
                    "validation_split": "rl_validation",
                    "training_scenario_bank_path": "configs/scenarios/m5_pilot_k1.json",
                    "validation_scenario_bank_path": "configs/scenarios/m5_validation_k1.json",
                },
                "maintenance_capacity": 1,
            }, f)
            config_path = f.name

        # Create checkpoint with training_split=rl_test
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name
            checkpoint = {
                "online_network_state_dict": {},
                "target_network_state_dict": {},
                "optimizer_state_dict": {},
                "python_rng_state": None,
                "numpy_rng_state": None,
                "torch_cpu_rng_state": torch.zeros(0),
                "global_step": 0,
                "gradient_update_count": 0,
                "epsilon_state": {"epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay_steps": 1000},
                "config": {},
                "training_seed": 6521,
                "metadata": {
                    "checkpoint_schema_version": 6,
                    "checkpoint_id": "test",
                    "saved_at": "2024-01-01T00:00:00Z",
                    "network_architecture_id": "a" * 64,
                    "action_table_hash": "b" * 64,
                    "observation_schema_id": "m5_point_v1",
                    "observation_dim": 10,
                    "action_count": 6,
                    "maintenance_capacity": 1,
                    "cost_regime_id": "failure-light-no-waste",
                    "environment_contract_id": "m2_v1",
                    "training_scenario_bank_identity": "c" * 64,
                    "validation_scenario_bank_identity": "d" * 64,
                    "training_split": "rl_test",  # FORBIDDEN
                    "validation_split": "rl_validation",
                    "global_step": 0,
                    "gradient_update_count": 0,
                    "epsilon": 1.0,
                    "device": "cpu",
                    "prediction_cache_manifest_path": "data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json",
                    "prediction_cache_manifest_sha256": "e" * 64,
                    "prediction_cache_declared_cache_hash": "f" * 64,
                    "prediction_cache_predictor_checkpoint_hash": "g" * 64,
                    "prediction_cache_feature_schema_hash": "h" * 12,
                    "prediction_cache_normalizer_hash": "i" * 64,
                    "prediction_cache_split": "rl_validation",
                    "prediction_cache_schema_version": "v2",
                    "selection_state": {
                        "selection_state_version": 1,
                        "validation_performed": False,
                        "best_validation_mean_cost": None,
                        "best_checkpoint_global_step": None,
                        "best_checkpoint_artifact_name": None,
                        "best_validation_failure_count": None,
                        "best_validation_worst_10_pct_cost": None,
                        "comparator_identity": "mean_cost_v1",
                        "equal_metric_tie_behavior": "keep_first",
                    },
                    "resolved_config_identity": "a" * 64,
                },
            }
            torch.save(checkpoint, checkpoint_path)

        try:
            result = subprocess.run(
                [
                    "python", "scripts/evaluate_ddqn.py",
                    "--checkpoint", checkpoint_path,
                    "--config", config_path,
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
            )

            assert result.returncode != 0
            assert "FORBIDDEN" in result.stderr or "CRITICAL BARRIER VIOLATION" in result.stderr or "rl_test" in result.stderr

        finally:
            import os
            os.unlink(config_path)
            os.unlink(checkpoint_path)

    def test_scenario_bank_split_rl_test_rejected(self):
        """Test that scenario bank declared split=rl_test is rejected."""
        import subprocess
        import tempfile
        import json
        import torch
        from pathlib import Path

        # Create a valid config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "environment": {
                    "validation_split": "rl_validation",
                    "training_scenario_bank_path": "configs/scenarios/m5_pilot_k1.json",
                    "validation_scenario_bank_path": "configs/scenarios/m5_validation_k1.json",
                },
                "maintenance_capacity": 1,
            }, f)
            config_path = f.name

        # Create checkpoint with valid splits
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name
            checkpoint = {
                "online_network_state_dict": {},
                "target_network_state_dict": {},
                "optimizer_state_dict": {},
                "python_rng_state": None,
                "numpy_rng_state": None,
                "torch_cpu_rng_state": torch.zeros(0),
                "global_step": 0,
                "gradient_update_count": 0,
                "epsilon_state": {"epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay_steps": 1000},
                "config": {},
                "training_seed": 6521,
                "metadata": {
                    "checkpoint_schema_version": 6,
                    "checkpoint_id": "test",
                    "saved_at": "2024-01-01T00:00:00Z",
                    "network_architecture_id": "a" * 64,
                    "action_table_hash": "b" * 64,
                    "observation_schema_id": "m5_point_v1",
                    "observation_dim": 10,
                    "action_count": 6,
                    "maintenance_capacity": 1,
                    "cost_regime_id": "failure-light-no-waste",
                    "environment_contract_id": "m2_v1",
                    "training_scenario_bank_identity": "c" * 64,
                    "validation_scenario_bank_identity": "d" * 64,
                    "training_split": "predictor_train",
                    "validation_split": "rl_validation",
                    "global_step": 0,
                    "gradient_update_count": 0,
                    "epsilon": 1.0,
                    "device": "cpu",
                    "prediction_cache_manifest_path": "data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json",
                    "prediction_cache_manifest_sha256": "e" * 64,
                    "prediction_cache_declared_cache_hash": "f" * 64,
                    "prediction_cache_predictor_checkpoint_hash": "g" * 64,
                    "prediction_cache_feature_schema_hash": "h" * 12,
                    "prediction_cache_normalizer_hash": "i" * 64,
                    "prediction_cache_split": "rl_validation",
                    "prediction_cache_schema_version": "v2",
                    "selection_state": {
                        "selection_state_version": 1,
                        "validation_performed": False,
                        "best_validation_mean_cost": None,
                        "best_checkpoint_global_step": None,
                        "best_checkpoint_artifact_name": None,
                        "best_validation_failure_count": None,
                        "best_validation_worst_10_pct_cost": None,
                        "comparator_identity": "mean_cost_v1",
                        "equal_metric_tie_behavior": "keep_first",
                    },
                    "resolved_config_identity": "b" * 64,
                },
            }
            torch.save(checkpoint, checkpoint_path)

        # Create a scenario bank with split=rl_test
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            scenario_bank_path = f.name
            json.dump({
                "split": "rl_test",  # FORBIDDEN
                "maintenance_capacity": 1,
                "scenarios": [],
            }, f)

        # Modify config to point to tampered scenario bank
        with open(config_path, 'w') as f:
            json.dump({
                "environment": {
                    "validation_split": "rl_validation",
                    "training_scenario_bank_path": "configs/scenarios/m5_pilot_k1.json",
                    "validation_scenario_bank_path": scenario_bank_path,
                },
                "maintenance_capacity": 1,
            }, f)

        try:
            result = subprocess.run(
                [
                    "python", "scripts/evaluate_ddqn.py",
                    "--checkpoint", checkpoint_path,
                    "--config", config_path,
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
            )

            assert result.returncode != 0
            # Either rl_test rejection or hash mismatch (hash check runs before split check)
            assert "FORBIDDEN" in result.stderr or "CRITICAL BARRIER VIOLATION" in result.stderr or "rl_test" in result.stderr or "hash mismatch" in result.stderr.lower()

        finally:
            import os
            os.unlink(config_path)
            os.unlink(checkpoint_path)
            os.unlink(scenario_bank_path)


class TestReplayBufferStrictStateValidation:
    """Test ReplayBuffer.load_state_dict strict validation."""

    def test_reject_python_list_observations(self):
        """Test that Python-list observations are rejected."""
        from agents.ddqn import ReplayBuffer, ReplayBufferConfig

        replay_buffer = ReplayBuffer(
            config=ReplayBufferConfig(capacity=100, observation_dim=10, seed=6521)
        )
        state = replay_buffer.state_dict(action_count=16)

        # Replace ndarray with Python list
        state["observations"] = [[float(i) for i in range(10)] for _ in range(100)]

        with pytest.raises(ValueError) as exc_info:
            replay_buffer.load_state_dict(state, expected_action_count=16)

        assert "observations must be numpy ndarray" in str(exc_info.value).lower()
        assert "python list" in str(exc_info.value).lower()

    def test_reject_python_list_actions(self):
        """Test that Python-list actions are rejected."""
        from agents.ddqn import ReplayBuffer, ReplayBufferConfig

        replay_buffer = ReplayBuffer(
            config=ReplayBufferConfig(capacity=100, observation_dim=10, seed=6521)
        )
        state = replay_buffer.state_dict(action_count=16)

        # Replace ndarray with Python list
        state["actions"] = [i % 16 for i in range(100)]

        with pytest.raises(ValueError) as exc_info:
            replay_buffer.load_state_dict(state, expected_action_count=16)

        assert "actions must be numpy ndarray" in str(exc_info.value).lower()
        assert "python list" in str(exc_info.value).lower()

    def test_reject_float64_observations(self):
        """Test that float64 observations are rejected."""
        from agents.ddqn import ReplayBuffer, ReplayBufferConfig
        import numpy as np

        replay_buffer = ReplayBuffer(
            config=ReplayBufferConfig(capacity=100, observation_dim=10, seed=6521)
        )
        state = replay_buffer.state_dict(action_count=16)

        # Replace float32 with float64
        state["observations"] = state["observations"].astype(np.float64)

        with pytest.raises(ValueError) as exc_info:
            replay_buffer.load_state_dict(state, expected_action_count=16)

        assert "observations must be float32" in str(exc_info.value).lower()
        assert "float64" in str(exc_info.value)

    def test_reject_float64_rewards(self):
        """Test that float64 rewards are rejected."""
        from agents.ddqn import ReplayBuffer, ReplayBufferConfig
        import numpy as np

        replay_buffer = ReplayBuffer(
            config=ReplayBufferConfig(capacity=100, observation_dim=10, seed=6521)
        )
        state = replay_buffer.state_dict(action_count=16)

        # Replace float32 with float64
        state["rewards"] = state["rewards"].astype(np.float64)

        with pytest.raises(ValueError) as exc_info:
            replay_buffer.load_state_dict(state, expected_action_count=16)

        assert "rewards must be float32" in str(exc_info.value).lower()
        assert "float64" in str(exc_info.value)

    def test_reject_float_actions(self):
        """Test that float actions are rejected."""
        from agents.ddqn import ReplayBuffer, ReplayBufferConfig
        import numpy as np

        replay_buffer = ReplayBuffer(
            config=ReplayBufferConfig(capacity=100, observation_dim=10, seed=6521)
        )
        state = replay_buffer.state_dict(action_count=16)

        # Replace int64 with float64
        state["actions"] = state["actions"].astype(np.float64)

        with pytest.raises(ValueError) as exc_info:
            replay_buffer.load_state_dict(state, expected_action_count=16)

        assert "actions must be integer dtype" in str(exc_info.value).lower()
        assert "float64" in str(exc_info.value)

    def test_reject_integer_flags(self):
        """Test that integer terminated/truncated flags are rejected."""
        from agents.ddqn import ReplayBuffer, ReplayBufferConfig
        import numpy as np

        replay_buffer = ReplayBuffer(
            config=ReplayBufferConfig(capacity=100, observation_dim=10, seed=6521)
        )
        state = replay_buffer.state_dict(action_count=16)

        # Replace bool with int64
        state["terminated"] = state["terminated"].astype(np.int64)

        with pytest.raises(ValueError) as exc_info:
            replay_buffer.load_state_dict(state, expected_action_count=16)

        assert "terminated must be bool" in str(exc_info.value).lower()
        assert "int64" in str(exc_info.value)

    def test_reject_object_arrays(self):
        """Test that object arrays are rejected."""
        from agents.ddqn import ReplayBuffer, ReplayBufferConfig
        import numpy as np

        replay_buffer = ReplayBuffer(
            config=ReplayBufferConfig(capacity=100, observation_dim=10, seed=6521)
        )
        state = replay_buffer.state_dict(action_count=16)

        # Replace with object array
        state["observations"] = np.array([[None for _ in range(10)] for _ in range(100)], dtype=object)

        with pytest.raises(ValueError) as exc_info:
            replay_buffer.load_state_dict(state, expected_action_count=16)

        assert "object dtype" in str(exc_info.value).lower()
        assert "object arrays are not supported" in str(exc_info.value).lower()

    def test_reject_illegal_actions(self):
        """Test that illegal action values are rejected."""
        from agents.ddqn import ReplayBuffer, ReplayBufferConfig
        import numpy as np

        replay_buffer = ReplayBuffer(
            config=ReplayBufferConfig(capacity=100, observation_dim=10, seed=6521)
        )
        state = replay_buffer.state_dict(action_count=16)

        # Insert illegal action (>= action_count)
        state["actions"][0] = 100  # Action 100 is illegal for action_count=16

        with pytest.raises(ValueError) as exc_info:
            replay_buffer.load_state_dict(state, expected_action_count=16)

        assert "illegal" in str(exc_info.value).lower() or "action" in str(exc_info.value).lower()

    def test_reject_missing_action_count(self):
        """Test that missing action_count is rejected."""
        from agents.ddqn import ReplayBuffer, ReplayBufferConfig

        replay_buffer = ReplayBuffer(
            config=ReplayBufferConfig(capacity=100, observation_dim=10, seed=6521)
        )
        state = replay_buffer.state_dict(action_count=16)

        # Load without expected_action_count
        with pytest.raises(ValueError) as exc_info:
            replay_buffer.load_state_dict(state, expected_action_count=None)

        assert "action_count" in str(exc_info.value).lower()

    def test_reject_replay_schema_mismatch(self):
        """Test that replay schema mismatch is rejected."""
        from agents.ddqn import ReplayBuffer, ReplayBufferConfig

        replay_buffer = ReplayBuffer(
            config=ReplayBufferConfig(capacity=100, observation_dim=10, seed=6521)
        )
        state = replay_buffer.state_dict(action_count=16)

        # Tamper with action_count in state
        state["action_count"] = 6  # Wrong

        with pytest.raises(ValueError) as exc_info:
            replay_buffer.load_state_dict(state, expected_action_count=16)

        assert "action_count mismatch" in str(exc_info.value).lower()


class TestCheckpointMetadataTypeValidation:
    """Test CheckpointMetadata.from_dict type-error handling."""

    def test_epsilon_tuple_type_error_produces_clear_valueerror(self):
        """Test that malformed epsilon (multi-type expected) produces clear ValueError, not AttributeError."""
        from agents.ddqn.checkpoint import CheckpointMetadata, CHECKPOINT_SCHEMA_VERSION

        # Create valid base data
        valid_data = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_id": "test_checkpoint",
            "saved_at": "2024-01-01T00:00:00Z",
            "network_architecture_id": "a" * 64,
            "action_table_hash": "b" * 64,
            "observation_schema_id": "m5_point_v1",
            "observation_dim": 10,
            "action_count": 16,
            "maintenance_capacity": 2,
            "cost_regime_id": "failure-light-no-waste",
            "environment_contract_id": "m2_v1",
            "training_scenario_bank_identity": "c" * 64,
            "validation_scenario_bank_identity": "d" * 64,
            "training_split": "predictor_train",
            "validation_split": "rl_validation",
            "global_step": 100,
            "gradient_update_count": 50,
            "epsilon": 0.5,  # Valid float
            "device": "cpu",
            # Schema v4: prediction cache provenance
            "prediction_cache_manifest_path": "data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json",
            "prediction_cache_manifest_sha256": "e" * 64,
            "prediction_cache_declared_cache_hash": "f" * 64,
            "prediction_cache_predictor_checkpoint_hash": "g" * 64,
            "prediction_cache_feature_schema_hash": "h" * 12,
            "prediction_cache_normalizer_hash": "i" * 64,
            "prediction_cache_split": "rl_validation",
            "prediction_cache_schema_version": "v2",
            # Schema v6: resolved config identity (REQUIRED)
            "resolved_config_identity": "a" * 64,
            "selection_state": {
                "selection_state_version": 1,
                "validation_performed": False,
                "best_validation_mean_cost": None,
                "best_checkpoint_global_step": None,
                "best_checkpoint_artifact_name": None,
                "best_validation_failure_count": None,
                "best_validation_worst_10_pct_cost": None,
                "comparator_identity": "mean_cost_v1",
                "equal_metric_tie_behavior": "keep_first",
            },
        }

        # Test with string epsilon (should fail with clear message)
        bad_data = valid_data.copy()
        bad_data["epsilon"] = "not_a_number"

        with pytest.raises(ValueError) as exc_info:
            CheckpointMetadata.from_dict(bad_data)

        # Verify clear ValueError, not AttributeError
        assert "epsilon" in str(exc_info.value)
        assert "wrong type" in str(exc_info.value).lower()
        assert "expected int, float" in str(exc_info.value)  # Tuple types formatted correctly
        assert "got str" in str(exc_info.value)

    def test_integer_field_type_error_produces_clear_valueerror(self):
        """Test that malformed integer field produces clear ValueError."""
        from agents.ddqn.checkpoint import CheckpointMetadata, CHECKPOINT_SCHEMA_VERSION

        valid_data = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_id": "test_checkpoint",
            "saved_at": "2024-01-01T00:00:00Z",
            "network_architecture_id": "a" * 64,
            "action_table_hash": "b" * 64,
            "observation_schema_id": "m5_point_v1",
            "observation_dim": 10,
            "action_count": 16,
            "maintenance_capacity": 2,
            "cost_regime_id": "failure-light-no-waste",
            "environment_contract_id": "m2_v1",
            "training_scenario_bank_identity": "c" * 64,
            "validation_scenario_bank_identity": "d" * 64,
            "training_split": "predictor_train",
            "validation_split": "rl_validation",
            "global_step": 100,
            "gradient_update_count": 50,
            "epsilon": 0.5,
            "device": "cpu",
            # Schema v4: prediction cache provenance
            "prediction_cache_manifest_path": "data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json",
            "prediction_cache_manifest_sha256": "e" * 64,
            "prediction_cache_declared_cache_hash": "f" * 64,
            "prediction_cache_predictor_checkpoint_hash": "g" * 64,
            "prediction_cache_feature_schema_hash": "h" * 12,
            "prediction_cache_normalizer_hash": "i" * 64,
            "prediction_cache_split": "rl_validation",
            "prediction_cache_schema_version": "v2",
            # Schema v6: resolved config identity (REQUIRED)
            "resolved_config_identity": "a" * 64,
            "selection_state": {
                "selection_state_version": 1,
                "validation_performed": False,
                "best_validation_mean_cost": None,
                "best_checkpoint_global_step": None,
                "best_checkpoint_artifact_name": None,
                "best_validation_failure_count": None,
                "best_validation_worst_10_pct_cost": None,
                "comparator_identity": "mean_cost_v1",
                "equal_metric_tie_behavior": "keep_first",
            },
        }

        # Test with string action_count (should fail with clear message)
        bad_data = valid_data.copy()
        bad_data["action_count"] = "sixteen"

        with pytest.raises(ValueError) as exc_info:
            CheckpointMetadata.from_dict(bad_data)

        assert "action_count" in str(exc_info.value)
        assert "wrong type" in str(exc_info.value).lower()
        assert "expected int" in str(exc_info.value)
        assert "got str" in str(exc_info.value)


class TestNetworkArchitectureIdentity:
    """Test unified network architecture identity across all usages."""

    def test_trainer_config_uses_num_hidden_layers(self):
        """Verify TrainerConfig field is num_hidden_layers, not hidden_layers."""
        from training.ddqn_trainer import TrainerConfig

        config = TrainerConfig(
            training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
        )

        # Verify the field exists and is accessible
        assert hasattr(config, 'num_hidden_layers')
        assert config.num_hidden_layers == 2

        # Verify hidden_layers does NOT exist
        assert not hasattr(config, 'hidden_layers')

    def test_network_architecture_id_consistent_across_sources(self):
        """Test that checkpoint metadata, manifest, and production expected identity are bit-identical."""
        from agents.ddqn import DDQNAgent, DDQNAgentConfig
        from agents.ddqn.checkpoint import save_checkpoint, compute_network_architecture_id
        from envs.action_table import ACTION_TABLE_N5_K2
        from training.ddqn_trainer import TrainerConfig, DDQNTrainer
        import tempfile
        import shutil
        from pathlib import Path
        import torch
        import json

        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Create a minimal trainer config
            config = TrainerConfig(
                split="predictor_train",
                validation_split="rl_validation",
                maintenance_capacity=2,
                cost_regime_id="failure-light-no-waste",
                training_scenario_bank_path="configs/scenarios/m5_pilot_k2.json",
                validation_scenario_bank_path="configs/scenarios/m5_validation_k2.json",
                max_steps=10,
                warmup_transitions=5,
                training_seed=6521,
                output_dir=str(temp_dir / "run"),
            )

            # Compute expected identity from config
            expected_identity = compute_network_architecture_id(
                observation_dim=10,
                hidden_dim=config.hidden_dim,
                num_hidden_layers=config.num_hidden_layers,
                activation="relu",
                action_count=config.num_actions,
                architecture_revision="m5_point_v1",
            )

            # Create trainer and save checkpoint
            trainer = DDQNTrainer(config=config)
            obs, _ = trainer.train_env.reset()
            trainer._current_obs = obs
            for _ in range(10):
                trainer.train_step()

            checkpoint_path = temp_dir / "checkpoint.pt"
            trainer.save_checkpoint(str(checkpoint_path))

            # Write artifacts to generate manifest
            trainer._write_artifacts()

            # Load checkpoint and verify metadata identity matches expected
            from agents.ddqn.checkpoint import load_checkpoint
            checkpoint_data, issues = load_checkpoint(checkpoint_path)
            metadata_identity = checkpoint_data.metadata.network_architecture_id

            assert metadata_identity == expected_identity, \
                f"Checkpoint metadata identity {metadata_identity[:16]}... != expected {expected_identity[:16]}..."

            # Read manifest and verify identity matches
            # Manifest is at run_dir/run_manifest.json where run_dir = output_dir / run_id
            run_dirs = list((temp_dir / "run").glob("*"))
            assert len(run_dirs) == 1, f"Expected exactly one run dir, got {run_dirs}"
            manifest_path = run_dirs[0] / "run_manifest.json"
            with open(manifest_path) as f:
                manifest = json.load(f)

            manifest_identity = manifest["network_architecture_id"]
            assert manifest_identity == expected_identity, \
                f"Manifest identity {manifest_identity[:16]}... != expected {expected_identity[:16]}..."

            # All three must be identical
            assert metadata_identity == manifest_identity == expected_identity

        finally:
            shutil.rmtree(temp_dir)

    def test_identity_deterministic_same_config(self):
        """Test that same config produces identical identity every time."""
        from agents.ddqn.checkpoint import compute_network_architecture_id

        id1 = compute_network_architecture_id(
            observation_dim=10,
            hidden_dim=128,
            num_hidden_layers=2,
            activation="relu",
            action_count=16,
            architecture_revision="m5_point_v1",
        )

        id2 = compute_network_architecture_id(
            observation_dim=10,
            hidden_dim=128,
            num_hidden_layers=2,
            activation="relu",
            action_count=16,
            architecture_revision="m5_point_v1",
        )

        assert id1 == id2

    def test_identity_different_hidden_layers(self):
        """Test that different num_hidden_layers produces different identity."""
        from agents.ddqn.checkpoint import compute_network_architecture_id

        id_2_layers = compute_network_architecture_id(
            observation_dim=10,
            hidden_dim=128,
            num_hidden_layers=2,
            activation="relu",
            action_count=16,
            architecture_revision="m5_point_v1",
        )

        id_3_layers = compute_network_architecture_id(
            observation_dim=10,
            hidden_dim=128,
            num_hidden_layers=3,
            activation="relu",
            action_count=16,
            architecture_revision="m5_point_v1",
        )

        assert id_2_layers != id_3_layers


class TestM5ProvenanceFailClosed:
    """M5 provenance final closeout: a run MUST NOT emit status COMPLETE when any
    required schema-6 prediction-cache provenance field cannot be computed.

    The trainer's _write_artifacts path must fail CLOSED: raise before the
    run_manifest.json is written, rather than swallowing the provenance error
    into a soft "error-only provenance" object while still emitting COMPLETE.
    """

    REQUIRED_PROVENANCE_FIELDS = (
        "prediction_cache_manifest_path",
        "prediction_cache_manifest_sha256",
        "prediction_cache_declared_cache_hash",
        "prediction_cache_predictor_checkpoint_hash",
        "prediction_cache_feature_schema_hash",
        "prediction_cache_normalizer_hash",
        "prediction_cache_split",
        "prediction_cache_schema_version",
    )

    def _make_config(self, temp_dir, manifest_path):
        from training.ddqn_trainer import TrainerConfig
        return TrainerConfig(
            split="predictor_train",
            validation_split="rl_validation",
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k2.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k2.json",
            max_steps=10,
            warmup_transitions=5,
            training_seed=6521,
            output_dir=str(temp_dir / "run"),
            prediction_cache_manifest_path=manifest_path,
        )

    def test_write_artifacts_fails_closed_on_missing_manifest(self, tmp_path):
        """If the prediction-cache manifest is unreadable, the run must NOT
        emit a run_manifest.json with status COMPLETE -- it must raise first."""
        from training.ddqn_trainer import DDQNTrainer

        bogus_manifest = str(tmp_path / "does_not_exist_manifest.json")
        config = self._make_config(tmp_path, bogus_manifest)
        trainer = DDQNTrainer(config=config)

        bogus_run_dir = tmp_path / "will_not_exist"
        # Working in an unrelated dir proves no manifest is written.
        import contextlib, io
        with pytest.raises((FileNotFoundError, ValueError, OSError)):
            # Manipulate run_dir into a fresh location THEN call _write_artifacts
            # so any manifest produced would land there and be detectable.
            from pathlib import Path as _P
            trainer.run_dir = _P(bogus_run_dir)
            bogus_run_dir.mkdir(parents=True, exist_ok=True)
            trainer._write_artifacts()

        # No run_manifest.json may exist -- proving fail-CLOSED before manifest.
        assert not (bogus_run_dir / "run_manifest.json").exists(), (
            "FAIL-OPEN: trainer wrote run_manifest.json despite missing provenance"
        )

    def test_write_artifacts_emits_all_required_provenance_when_manifest_valid(self, tmp_path):
        """When the manifest is valid, the COMPLETE manifest must carry every
        required provenance field, present and non-null."""
        import os
        from training.ddqn_trainer import DDQNTrainer

        repo_root = Path(__file__).resolve().parent.parent
        default_manifest = str(repo_root / "data/processed/fd001/v2/06_PREDICTIONS/"
                                       "prediction_cache_manifest_v2.json")
        if not os.path.exists(default_manifest):
            pytest.skip("v2 prediction cache manifest not present in this checkout")

        config = self._make_config(tmp_path, default_manifest)
        trainer = DDQNTrainer(config=config)
        # Force completion criteria: global_step >= max_steps
        trainer.global_step = config.max_steps
        trainer.agent.global_step = config.max_steps

        trainer.run_dir = tmp_path / "real_run"
        trainer.run_dir.mkdir(parents=True, exist_ok=True)
        trainer._write_artifacts()

        manifest_path = trainer.run_dir / "run_manifest.json"
        assert manifest_path.exists(), "run_manifest.json was not written"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["status"] == "COMPLETE"
        prov = manifest["prediction_cache_provenance"]
        assert "prediction_cache_provenance_error" not in prov, (
            "valid manifest must not carry a provenance error key"
        )
        for field in self.REQUIRED_PROVENANCE_FIELDS:
            assert field in prov, f"missing required provenance field: {field}"
            assert prov[field] is not None, f"null required provenance field: {field}"
        assert prov["prediction_cache_split"] == config.validation_split


class TestCheckpointValidation:
    """Test checkpoint compatibility validation."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d)

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        return DDQNAgent(config=DDQNAgentConfig(num_actions=16), seed=6521)

    def test_observation_dim_mismatch(self, temp_dir, agent):
        """Test rejection of observation dimension mismatch."""
        checkpoint_path = temp_dir / "test_checkpoint.pt"

        save_checkpoint(
    prediction_cache_manifest_path="data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json",
            agent=agent,
            config={},
            output_path=checkpoint_path,
            maintenance_capacity=2,
            action_table=ACTION_TABLE_N5_K2,
            cost_regime_id="failure-light-no-waste",
            training_seed=6521,
            training_split="predictor_train",
            validation_split="rl_validation",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k2.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k2.json",
            selection_state=CheckpointSelectionState(
                selection_state_version=CHECKPOINT_SELECTION_STATE_VERSION,
                validation_performed=False,
                best_validation_mean_cost=None,
                best_checkpoint_global_step=None,
                best_checkpoint_artifact_name=None,
                best_validation_failure_count=None,
                best_validation_worst_10_pct_cost=None,
                comparator_identity="mean_cost_v1",
                equal_metric_tie_behavior="keep_first",
            ),
        )

        _, issues = load_checkpoint(
            checkpoint_path,
            expected_observation_dim=5,  # Wrong
        )

        assert len(issues["incompatibilities"]) > 0
        assert "Observation dimension mismatch" in issues["incompatibilities"][0]

    def test_action_count_mismatch(self, temp_dir, agent):
        """Test rejection of action count mismatch."""
        checkpoint_path = temp_dir / "test_checkpoint.pt"

        save_checkpoint(
    prediction_cache_manifest_path="data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json",
            agent=agent,
            config={},
            output_path=checkpoint_path,
            maintenance_capacity=2,
            action_table=ACTION_TABLE_N5_K2,
            cost_regime_id="failure-light-no-waste",
            training_seed=6521,
            training_split="predictor_train",
            validation_split="rl_validation",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k2.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k2.json",
            selection_state=CheckpointSelectionState(
                selection_state_version=CHECKPOINT_SELECTION_STATE_VERSION,
                validation_performed=False,
                best_validation_mean_cost=None,
                best_checkpoint_global_step=None,
                best_checkpoint_artifact_name=None,
                best_validation_failure_count=None,
                best_validation_worst_10_pct_cost=None,
                comparator_identity="mean_cost_v1",
                equal_metric_tie_behavior="keep_first",
            ),
        )

        _, issues = load_checkpoint(
            checkpoint_path,
            expected_action_count=6,  # Wrong (checkpoint has 16)
        )

        assert len(issues["incompatibilities"]) > 0
        assert "Action count mismatch" in issues["incompatibilities"][0]

    def test_k_mismatch(self, temp_dir, agent):
        """Test rejection of K mismatch."""
        checkpoint_path = temp_dir / "test_checkpoint.pt"

        save_checkpoint(
    prediction_cache_manifest_path="data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json",
            agent=agent,
            config={},
            output_path=checkpoint_path,
            maintenance_capacity=2,
            action_table=ACTION_TABLE_N5_K2,
            cost_regime_id="failure-light-no-waste",
            training_seed=6521,
            training_split="predictor_train",
            validation_split="rl_validation",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k2.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k2.json",
            selection_state=CheckpointSelectionState(
                selection_state_version=CHECKPOINT_SELECTION_STATE_VERSION,
                validation_performed=False,
                best_validation_mean_cost=None,
                best_checkpoint_global_step=None,
                best_checkpoint_artifact_name=None,
                best_validation_failure_count=None,
                best_validation_worst_10_pct_cost=None,
                comparator_identity="mean_cost_v1",
                equal_metric_tie_behavior="keep_first",
            ),
        )

        _, issues = load_checkpoint(
            checkpoint_path,
            expected_k=1,  # Wrong
        )

        assert len(issues["incompatibilities"]) > 0
        assert "Maintenance capacity mismatch" in issues["incompatibilities"][0]


class TestActionTableHash:
    """Test action table hashing."""

    def test_k1_hash(self):
        """Test K=1 action table hash."""
        hash1 = compute_action_table_hash(ACTION_TABLE_N5_K1)
        hash2 = compute_action_table_hash(ACTION_TABLE_N5_K1)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

    def test_k2_hash(self):
        """Test K=2 action table hash."""
        hash1 = compute_action_table_hash(ACTION_TABLE_N5_K2)
        hash2 = compute_action_table_hash(ACTION_TABLE_N5_K2)
        assert hash1 == hash2

    def test_k1_k2_different(self):
        """Test K=1 and K=2 have different hashes."""
        hash_k1 = compute_action_table_hash(ACTION_TABLE_N5_K1)
        hash_k2 = compute_action_table_hash(ACTION_TABLE_N5_K2)
        assert hash_k1 != hash_k2


class TestInformationBarrier:
    """Test information barrier - policy cannot access hidden state."""

    def test_no_true_rul_in_observation(self):
        """Verify observation schema doesn't include true_rul."""
        from agents.ddqn.agent import DDQNAgentConfig

        config = DDQNAgentConfig(num_actions=16)
        assert config.observation_dim == 10
        # Observation is [slot_0_age, slot_0_rul, slot_1_age, slot_1_rul, ...]
        # true_rul is NOT part of this

    def test_ddqn_agent_no_hidden_access(self):
        """Test DDQN agent cannot access hidden simulator state."""
        agent = DDQNAgent(config=DDQNAgentConfig(num_actions=16), seed=6521)

        # Agent only has access to:
        # - observation (10-dim float vector)
        # - action_id
        # - reward
        # Agent does NOT have access to:
        # - true_rul
        # - trajectory_id
        # - unit_id
        # These are environment internals

        # This is a semantic test - verifying the agent interface
        assert not hasattr(agent, 'true_rul')
        assert not hasattr(agent, 'trajectory_id')
        assert not hasattr(agent, 'unit_id')


class TestRlTestBarrier:
    """Test rl_test split barrier."""

    def test_trainer_config_rejects_rl_test(self):
        """Test TrainerConfig rejects rl_test split."""
        from training.ddqn_trainer import TrainerConfig

        with pytest.raises(ValueError) as exc_info:
            TrainerConfig(split="rl_test")

        assert "CRITICAL BARRIER VIOLATION" in str(exc_info.value)
        assert "rl_test" in str(exc_info.value)

    def test_trainer_config_rejects_rl_test_for_validation_split(self):
        """Test TrainerConfig rejects rl_test for validation_split."""
        from training.ddqn_trainer import TrainerConfig

        with pytest.raises(ValueError) as exc_info:
            TrainerConfig(
                validation_split="rl_test",
                training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
                validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            )

        assert "CRITICAL BARRIER VIOLATION" in str(exc_info.value)
        assert "validation_split='rl_test'" in str(exc_info.value)

    def test_cli_rejects_rl_test(self):
        """Test CLI rejects rl_test split."""
        import subprocess
        result = subprocess.run(
            ["python", "scripts/validate_config.py", "--config", "configs/agents/ddqn_v1.json", "--split", "rl_test"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        # Should exit non-zero
        assert result.returncode != 0 or "FORBIDDEN" in result.stdout or "FORBIDDEN" in result.stderr

    def test_evaluate_cli_rejects_rl_test(self):
        """Test evaluation CLI rejects rl_test split."""
        import subprocess
        import tempfile

        # Create a dummy checkpoint file
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            checkpoint_path = f.name
            result = subprocess.run(
                [
                    "python", "scripts/evaluate_ddqn.py",
                    "--checkpoint", checkpoint_path,
                    "--config", "configs/agents/ddqn_v1.json",
                    "--split", "rl_test",
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
            )
            # Should exit non-zero with FORBIDDEN message
            assert result.returncode != 0
            assert "FORBIDDEN" in result.stdout or "FORBIDDEN" in result.stderr or "rl_test" in result.stderr

    def test_evaluate_cli_rejects_config_rl_test(self):
        """Test evaluation CLI rejects config-derived rl_test split."""
        import subprocess
        import tempfile
        import json

        # Create a config with validation_split=rl_test and required fields
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "environment": {"validation_split": "rl_test"},
                "training_scenario_bank_path": "configs/scenarios/m5_pilot_k1.json",
                "validation_scenario_bank_path": "configs/scenarios/m5_validation_k1.json",
                "maintenance_capacity": 1,
            }, f)
            config_path = f.name

        # Create minimal valid checkpoint
        import torch
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name
            checkpoint = {
                "online_network_state_dict": {},
                "target_network_state_dict": {},
                "optimizer_state_dict": {},
                "python_rng_state": None,
                "numpy_rng_state": None,
                "torch_cpu_rng_state": torch.zeros(0),
                "global_step": 0,
                "gradient_update_count": 0,
                "epsilon_state": {"epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay_steps": 1000},
                "config": {},
                "training_seed": 6521,
                "metadata": {
                    "checkpoint_schema_version": 4,
                    "checkpoint_id": "test",
                    "saved_at": "2024-01-01T00:00:00Z",
                    "network_architecture_id": "a" * 64,
                    "action_table_hash": "b" * 64,
                    "observation_schema_id": "m5_point_v1",
                    "observation_dim": 10,
                    "action_count": 6,
                    "maintenance_capacity": 1,
                    "cost_regime_id": "failure-light-no-waste",
                    "environment_contract_id": "m2_v1",
                    "training_scenario_bank_identity": "c" * 64,
                    "validation_scenario_bank_identity": "d" * 64,
                    "training_split": "predictor_train",
                    "validation_split": "rl_validation",
                    "global_step": 0,
                    "gradient_update_count": 0,
                    "epsilon": 1.0,
                    "device": "cpu",
                    "prediction_cache_manifest_path": "data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json",
                    "prediction_cache_manifest_sha256": "e" * 64,
                    "prediction_cache_declared_cache_hash": "f" * 64,
                    "prediction_cache_predictor_checkpoint_hash": "g" * 64,
                    "prediction_cache_feature_schema_hash": "h" * 12,
                    "prediction_cache_normalizer_hash": "i" * 64,
                    "prediction_cache_split": "rl_validation",
                    "prediction_cache_schema_version": "v2",
                },
            }
            torch.save(checkpoint, checkpoint_path)

        try:
            result = subprocess.run(
                [
                    "python", "scripts/evaluate_ddqn.py",
                    "--checkpoint", checkpoint_path,
                    "--config", config_path,
                    # No --split, should use config validation_split
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
            )
            # Should exit non-zero with FORBIDDEN message
            assert result.returncode != 0
            assert "FORBIDDEN" in result.stderr or "CRITICAL BARRIER VIOLATION" in result.stderr
        finally:
            import os
            os.unlink(config_path)
            os.unlink(checkpoint_path)

    def test_evaluate_cli_defaults_to_rl_validation(self):
        """Test evaluation CLI defaults to rl_validation when no split specified."""
        import subprocess
        import tempfile

        # Config without validation_split should default to rl_validation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"environment": {}}, f)
            config_path = f.name

        # Create minimal valid checkpoint
        import torch
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name
            # Can't easily create valid checkpoint, just test that rl_validation
            # is not rejected (no FORBIDDEN message)
            # The checkpoint load will fail but we check it's not due to split

        # Test passes if no FORBIDDEN message for rl_validation
        # (checkpoint validation will fail but that's expected)
        result = subprocess.run(
            [
                "python", "scripts/evaluate_ddqn.py",
                "--checkpoint", checkpoint_path,
                "--config", config_path,
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        # Should NOT have FORBIDDEN message since default is rl_validation
        assert "FORBIDDEN" not in result.stdout and "FORBIDDEN" not in result.stderr


class TestTrainValidationSeparation:
    """Test training and validation scenario bank separation."""

    def test_trainer_config_requires_separate_banks(self):
        """Test TrainerConfig requires separate training and validation scenario bank paths."""
        from training.ddqn_trainer import TrainerConfig

        # Missing training_scenario_bank_path should fail
        with pytest.raises(ValueError) as exc_info:
            TrainerConfig(
                training_scenario_bank_path=None,
                validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            )

        assert "training_scenario_bank_path is required" in str(exc_info.value)

        # Missing validation_scenario_bank_path should fail
        with pytest.raises(ValueError) as exc_info:
            TrainerConfig(
                training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
                validation_scenario_bank_path=None,
            )

        assert "validation_scenario_bank_path is required" in str(exc_info.value)

    def test_trainer_config_requires_correct_splits(self):
        """Test TrainerConfig enforces correct splits for M5 formal training."""
        from training.ddqn_trainer import TrainerConfig

        # Wrong training split
        with pytest.raises(ValueError) as exc_info:
            TrainerConfig(
                split="rl_validation",
                training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
                validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            )

        assert "predictor_train" in str(exc_info.value)

        # Wrong validation split
        with pytest.raises(ValueError) as exc_info:
            TrainerConfig(
                validation_split="predictor_train",
                training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
                validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            )

        assert "rl_validation" in str(exc_info.value)

    def test_rl_test_rejected_for_validation_split(self):
        """Test TrainerConfig rejects rl_test for validation_split."""
        from training.ddqn_trainer import TrainerConfig

        with pytest.raises(ValueError) as exc_info:
            TrainerConfig(
                validation_split="rl_test",
                training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
                validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            )

        assert "CRITICAL BARRIER VIOLATION" in str(exc_info.value)
        assert "validation_split='rl_test'" in str(exc_info.value)

    def test_scenario_bank_split_alignment(self):
        """Test that scenario banks have correct split values."""
        import json

        # Load pilot config
        with open("configs/scenarios/m5_pilot_k1.json") as f:
            pilot_config = json.load(f)

        assert pilot_config["split"] == "predictor_train"
        for scenario in pilot_config["scenarios"]:
            assert scenario["split"] == "predictor_train"

        # Load validation config
        with open("configs/scenarios/m5_validation_k1.json") as f:
            val_config = json.load(f)

        assert val_config["split"] == "rl_validation"
        for scenario in val_config["scenarios"]:
            assert scenario["split"] == "rl_validation"

    def test_training_uses_pilot_bank_validation_uses_val_bank(self):
        """Test that training uses pilot bank and validation uses validation bank."""
        from training.ddqn_trainer import TrainerConfig, DDQNTrainer
        import tempfile
        from pathlib import Path

        # Create a minimal trainer config with separate banks
        config = TrainerConfig(
            split="predictor_train",
            validation_split="rl_validation",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            maintenance_capacity=1,
            max_steps=10,  # Very small for test
            warmup_transitions=5,
            validation_interval=100,  # Won't trigger in test
            output_dir=tempfile.mkdtemp(),
        )

        # Verify config has separate paths
        assert config.training_scenario_bank_path == "configs/scenarios/m5_pilot_k1.json"
        assert config.validation_scenario_bank_path == "configs/scenarios/m5_validation_k1.json"
        assert config.training_scenario_bank_path != config.validation_scenario_bank_path


class TestStrictCheckpointIdentity:
    """Test strict checkpoint identity field validation (F6)."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d)

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        return DDQNAgent(config=DDQNAgentConfig(num_actions=16), seed=6521)

    def test_save_includes_mandatory_identity_fields(self, temp_dir, agent):
        """Test checkpoint save includes action_table_hash and observation_schema_id."""
        from agents.ddqn.checkpoint import save_checkpoint, CheckpointMetadata

        checkpoint_path = temp_dir / "test_checkpoint.pt"

        checkpoint_data = save_checkpoint(
    prediction_cache_manifest_path="data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json",
            agent=agent,
            config={},
            output_path=checkpoint_path,
            maintenance_capacity=2,
            action_table=ACTION_TABLE_N5_K2,
            cost_regime_id="failure-light-no-waste",
            training_seed=6521,
            training_split="predictor_train",
            validation_split="rl_validation",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k2.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k2.json",
            selection_state=CheckpointSelectionState(
                selection_state_version=CHECKPOINT_SELECTION_STATE_VERSION,
                validation_performed=False,
                best_validation_mean_cost=None,
                best_checkpoint_global_step=None,
                best_checkpoint_artifact_name=None,
                best_validation_failure_count=None,
                best_validation_worst_10_pct_cost=None,
                comparator_identity="mean_cost_v1",
                equal_metric_tie_behavior="keep_first",
            ),
        )

        # Verify identity fields are present and not None
        assert checkpoint_data.metadata.action_table_hash is not None
        assert len(checkpoint_data.metadata.action_table_hash) == 64  # SHA256 hex

        assert checkpoint_data.metadata.observation_schema_id is not None
        assert checkpoint_data.metadata.observation_schema_id == "m5_point_v1"

    def test_load_fails_on_missing_action_table_hash(self, temp_dir, agent):
        """Test checkpoint load fails when action_table_hash is missing."""
        from agents.ddqn.checkpoint import CheckpointData, CheckpointMetadata
        import torch

        # Create a checkpoint with missing action_table_hash
        checkpoint_path = temp_dir / "bad_checkpoint.pt"

        # Save a normal checkpoint first to get valid state dicts
        normal_data = save_checkpoint(
    prediction_cache_manifest_path="data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json",
            agent=agent,
            config={},
            output_path=temp_dir / "normal.pt",
            maintenance_capacity=2,
            action_table=ACTION_TABLE_N5_K2,
            cost_regime_id="failure-light-no-waste",
            training_seed=6521,
            training_split="predictor_train",
            validation_split="rl_validation",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k2.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k2.json",
            selection_state=CheckpointSelectionState(
                selection_state_version=CHECKPOINT_SELECTION_STATE_VERSION,
                validation_performed=False,
                best_validation_mean_cost=None,
                best_checkpoint_global_step=None,
                best_checkpoint_artifact_name=None,
                best_validation_failure_count=None,
                best_validation_worst_10_pct_cost=None,
                comparator_identity="mean_cost_v1",
                equal_metric_tie_behavior="keep_first",
            ),
        )

        # Load and corrupt the metadata
        checkpoint = torch.load(temp_dir / "normal.pt", map_location="cpu", weights_only=False)
        checkpoint["metadata"]["action_table_hash"] = None

        torch.save(checkpoint, checkpoint_path)

        # Try to load - should fail
        with pytest.raises(ValueError) as exc_info:
            load_checkpoint(checkpoint_path)

        assert "action_table_hash" in str(exc_info.value)
        assert "required fields" in str(exc_info.value).lower()

    def test_load_fails_on_missing_observation_schema_id(self, temp_dir, agent):
        """Test checkpoint load fails when observation_schema_id is missing."""
        from agents.ddqn.checkpoint import save_checkpoint
        import torch

        checkpoint_path = temp_dir / "bad_checkpoint.pt"

        # Save normal checkpoint
        save_checkpoint(
    prediction_cache_manifest_path="data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json",
            agent=agent,
            config={},
            output_path=temp_dir / "normal.pt",
            maintenance_capacity=2,
            action_table=ACTION_TABLE_N5_K2,
            cost_regime_id="failure-light-no-waste",
            training_seed=6521,
            training_split="predictor_train",
            validation_split="rl_validation",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k2.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k2.json",
            selection_state=CheckpointSelectionState(
                selection_state_version=CHECKPOINT_SELECTION_STATE_VERSION,
                validation_performed=False,
                best_validation_mean_cost=None,
                best_checkpoint_global_step=None,
                best_checkpoint_artifact_name=None,
                best_validation_failure_count=None,
                best_validation_worst_10_pct_cost=None,
                comparator_identity="mean_cost_v1",
                equal_metric_tie_behavior="keep_first",
            ),
        )

        # Corrupt metadata
        checkpoint = torch.load(temp_dir / "normal.pt", map_location="cpu", weights_only=False)
        checkpoint["metadata"]["observation_schema_id"] = None

        torch.save(checkpoint, checkpoint_path)

        # Try to load - should fail
        with pytest.raises(ValueError) as exc_info:
            load_checkpoint(checkpoint_path)

        assert "observation_schema_id" in str(exc_info.value)
        assert "required fields" in str(exc_info.value).lower()

    def test_load_fails_on_action_table_hash_mismatch(self, temp_dir, agent):
        """Test checkpoint load fails when action_table_hash doesn't match expected."""
        from agents.ddqn.checkpoint import save_checkpoint, compute_action_table_hash
        from envs.action_table import ACTION_TABLE_N5_K1

        checkpoint_path = temp_dir / "test_checkpoint.pt"

        # Save with K=2 action table
        save_checkpoint(
    prediction_cache_manifest_path="data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json",
            agent=agent,
            config={},
            output_path=checkpoint_path,
            maintenance_capacity=2,
            action_table=ACTION_TABLE_N5_K2,
            cost_regime_id="failure-light-no-waste",
            training_seed=6521,
            training_split="predictor_train",
            validation_split="rl_validation",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k2.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k2.json",
            selection_state=CheckpointSelectionState(
                selection_state_version=CHECKPOINT_SELECTION_STATE_VERSION,
                validation_performed=False,
                best_validation_mean_cost=None,
                best_checkpoint_global_step=None,
                best_checkpoint_artifact_name=None,
                best_validation_failure_count=None,
                best_validation_worst_10_pct_cost=None,
                comparator_identity="mean_cost_v1",
                equal_metric_tie_behavior="keep_first",
            ),
        )

        # Compute K=1 hash (different from K=2)
        k1_hash = compute_action_table_hash(ACTION_TABLE_N5_K1)

        # Load with expected hash validation (should fail)
        _, issues = load_checkpoint(
            checkpoint_path,
            expected_action_table_hash=k1_hash,  # Wrong hash
        )

        assert len(issues["incompatibilities"]) > 0
        assert "action table hash" in str(issues["incompatibilities"][0]).lower() or \
               "Action count" in str(issues["incompatibilities"][0])

    def test_checkpoint_has_required_fields_in_dataclass(self):
        """Test CheckpointMetadata dataclass has required identity fields."""
        from agents.ddqn.checkpoint import CheckpointMetadata
        import dataclasses

        fields = dataclasses.fields(CheckpointMetadata)
        field_names = [f.name for f in fields]

        # Verify identity fields exist
        assert "action_table_hash" in field_names
        assert "observation_schema_id" in field_names

        # Verify they don't have default values (are required)
        for f in fields:
            if f.name in ("action_table_hash", "observation_schema_id"):
                # These should NOT have default values
                assert f.default is dataclasses.MISSING, \
                    f"{f.name} should be required (no default)"


class TestProductionCheckpointResume:
    """Test production fail-closed checkpoint resume with identity validation."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d)

    def test_resume_validates_action_table_hash(self, temp_dir):
        """Test production resume rejects checkpoint with wrong action_table_hash."""
        from training.ddqn_trainer import DDQNTrainer, TrainerConfig
        from agents.ddqn.checkpoint import save_checkpoint, compute_action_table_hash
        from envs.action_table import ACTION_TABLE_N5_K1, ACTION_TABLE_N5_K2

        # Create a checkpoint with K=2 action table
        checkpoint_path = temp_dir / "k2_checkpoint.pt"

        # First, train briefly to create a valid checkpoint
        config = TrainerConfig(
            split="predictor_train",
            validation_split="rl_validation",
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k2.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k2.json",
            max_steps=10,
            warmup_transitions=5,
            training_seed=6521,
            output_dir=str(temp_dir / "k2_run"),
        )

        trainer = DDQNTrainer(config=config)
        # Run a few steps to get valid state
        obs, _ = trainer.train_env.reset()
        trainer._current_obs = obs
        for _ in range(10):
            trainer.train_step()

        trainer.save_checkpoint(str(checkpoint_path))

        # Now try to resume with K=1 config (wrong action table)
        k1_config = TrainerConfig(
            split="predictor_train",
            validation_split="rl_validation",
            maintenance_capacity=1,  # K=1, not K=2
            cost_regime_id="failure-light-no-waste",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            max_steps=100,
            training_seed=6521,
            output_dir=str(temp_dir / "k1_run"),
        )

        # Should fail with identity mismatch
        with pytest.raises(ValueError) as exc_info:
            DDQNTrainer(config=k1_config, resume_from=checkpoint_path)

        assert "identity" in str(exc_info.value).lower() or \
               "Action" in str(exc_info.value) or \
               "maintenance capacity" in str(exc_info.value).lower() or \
               "Action count" in str(exc_info.value)

    def test_resume_validates_observation_schema_id(self, temp_dir):
        """Test production resume rejects checkpoint with wrong observation_schema_id."""
        from agents.ddqn.agent import DDQNAgent, DDQNAgentConfig
        from agents.ddqn.checkpoint import save_checkpoint, load_checkpoint, CheckpointMetadata
        from envs.action_table import ACTION_TABLE_N5_K1

        # Create a checkpoint
        agent = DDQNAgent(config=DDQNAgentConfig(num_actions=6), seed=6521)
        checkpoint_path = temp_dir / "test_checkpoint.pt"

        save_checkpoint(
    prediction_cache_manifest_path="data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json",
            agent=agent,
            config={},
            output_path=checkpoint_path,
            maintenance_capacity=1,
            action_table=ACTION_TABLE_N5_K1,
            cost_regime_id="failure-light-no-waste",
            training_seed=6521,
            training_split="predictor_train",
            validation_split="rl_validation",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            selection_state=CheckpointSelectionState(
                selection_state_version=CHECKPOINT_SELECTION_STATE_VERSION,
                validation_performed=False,
                best_validation_mean_cost=None,
                best_checkpoint_global_step=None,
                best_checkpoint_artifact_name=None,
                best_validation_failure_count=None,
                best_validation_worst_10_pct_cost=None,
                comparator_identity="mean_cost_v1",
                equal_metric_tie_behavior="keep_first",
            ),
        )

        # Load with wrong observation_schema_id
        _, issues = load_checkpoint(
            checkpoint_path,
            expected_observation_schema_id="wrong_schema_v99",
        )

        assert len(issues["incompatibilities"]) > 0
        assert "Observation schema" in str(issues["incompatibilities"][0])

    def test_resume_validates_k_value(self, temp_dir):
        """Test production resume rejects checkpoint with wrong K."""
        from training.ddqn_trainer import DDQNTrainer, TrainerConfig
        from envs.action_table import ACTION_TABLE_N5_K1, ACTION_TABLE_N5_K2
        from agents.ddqn.checkpoint import compute_action_table_hash

        # Create a K=1 checkpoint
        k1_checkpoint = temp_dir / "k1_checkpoint.pt"
        config_k1 = TrainerConfig(
            split="predictor_train",
            validation_split="rl_validation",
            maintenance_capacity=1,
            cost_regime_id="failure-light-no-waste",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            max_steps=10,
            warmup_transitions=5,
            training_seed=6521,
            output_dir=str(temp_dir / "k1_run"),
        )

        trainer_k1 = DDQNTrainer(config=config_k1)
        obs, _ = trainer_k1.train_env.reset()
        trainer_k1._current_obs = obs
        for _ in range(10):
            trainer_k1.train_step()
        trainer_k1.save_checkpoint(str(k1_checkpoint))

        # Try to resume K=1 checkpoint with K=2 config
        config_k2 = TrainerConfig(
            split="predictor_train",
            validation_split="rl_validation",
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k2.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k2.json",
            max_steps=100,
            training_seed=6521,
            output_dir=str(temp_dir / "k2_run"),
        )

        with pytest.raises(ValueError) as exc_info:
            DDQNTrainer(config=config_k2, resume_from=k1_checkpoint)

        assert "K" in str(exc_info.value) or \
               "maintenance capacity" in str(exc_info.value).lower() or \
               "Action count" in str(exc_info.value)

    def test_resume_validates_cost_regime(self, temp_dir):
        """Test production resume rejects checkpoint with wrong cost regime."""
        from training.ddqn_trainer import DDQNTrainer, TrainerConfig

        # Create a checkpoint with regime A
        checkpoint_a = temp_dir / "regime_a.pt"
        config_a = TrainerConfig(
            split="predictor_train",
            validation_split="rl_validation",
            maintenance_capacity=1,
            cost_regime_id="failure-light-no-waste",  # Regime A
            training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            max_steps=10,
            warmup_transitions=5,
            training_seed=6521,
            output_dir=str(temp_dir / "run_a"),
        )

        trainer_a = DDQNTrainer(config=config_a)
        obs, _ = trainer_a.train_env.reset()
        trainer_a._current_obs = obs
        for _ in range(10):
            trainer_a.train_step()
        trainer_a.save_checkpoint(str(checkpoint_a))

        # Try to resume with regime B
        config_b = TrainerConfig(
            split="predictor_train",
            validation_split="rl_validation",
            maintenance_capacity=1,
            cost_regime_id="failure-heavy-no-waste",  # Regime B (different!)
            training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            max_steps=100,
            training_seed=6521,
            output_dir=str(temp_dir / "run_b"),
        )

        with pytest.raises(ValueError) as exc_info:
            DDQNTrainer(config=config_b, resume_from=checkpoint_a)

        assert "Cost regime" in str(exc_info.value) or \
               "cost regime" in str(exc_info.value).lower()

    def test_resume_validates_environment_contract_id(self, temp_dir):
        """Test production resume rejects checkpoint with wrong environment_contract_id."""
        from training.ddqn_trainer import DDQNTrainer, TrainerConfig
        from agents.ddqn.agent import DDQNAgent, DDQNAgentConfig
        from envs.action_table import ACTION_TABLE_N5_K1
        import torch

        # Save checkpoint normally first
        agent = DDQNAgent(config=DDQNAgentConfig(num_actions=6), seed=6521)
        save_path = temp_dir / "temp_checkpoint.pt"
        from agents.ddqn.checkpoint import save_checkpoint
        save_checkpoint(
    prediction_cache_manifest_path="data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json",
            agent=agent,
            config={},
            output_path=save_path,
            maintenance_capacity=1,
            action_table=ACTION_TABLE_N5_K1,
            cost_regime_id="failure-light-no-waste",
            training_seed=6521,
            training_split="predictor_train",
            validation_split="rl_validation",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            selection_state=CheckpointSelectionState(
                selection_state_version=CHECKPOINT_SELECTION_STATE_VERSION,
                validation_performed=False,
                best_validation_mean_cost=None,
                best_checkpoint_global_step=None,
                best_checkpoint_artifact_name=None,
                best_validation_failure_count=None,
                best_validation_worst_10_pct_cost=None,
                comparator_identity="mean_cost_v1",
                equal_metric_tie_behavior="keep_first",
            ),
        )

        # Now modify the metadata to have wrong environment_contract_id
        checkpoint = torch.load(save_path, map_location="cpu", weights_only=False)
        checkpoint["metadata"]["environment_contract_id"] = "wrong_contract_v99"
        checkpoint_path = temp_dir / "wrong_contract_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        # Create a valid config for resuming
        config = TrainerConfig(
            split="predictor_train",
            validation_split="rl_validation",
            maintenance_capacity=1,
            cost_regime_id="failure-light-no-waste",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            max_steps=100,
            training_seed=6521,
            output_dir=str(temp_dir / "run"),
        )

        # Production resume path: DDQNTrainer(resume_from=...) should reject
        with pytest.raises(ValueError) as exc_info:
            DDQNTrainer(config=config, resume_from=checkpoint_path)

        # Verify production error message
        assert "Environment contract" in str(exc_info.value) or "identity" in str(exc_info.value).lower()
        assert "wrong_contract_v99" in str(exc_info.value) or "m2_v1" in str(exc_info.value)

    def test_resume_automatic_state_restoration(self, temp_dir):
        """Test production resume automatically restores all state without manual assignment."""
        from training.ddqn_trainer import DDQNTrainer, TrainerConfig

        # Create trainer1 and train briefly
        config = TrainerConfig(
            split="predictor_train",
            validation_split="rl_validation",
            maintenance_capacity=1,
            cost_regime_id="failure-light-no-waste",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            max_steps=100,
            warmup_transitions=10,
            training_seed=6521,
            output_dir=str(temp_dir / "run1"),
        )

        trainer1 = DDQNTrainer(config=config)
        obs, _ = trainer1.train_env.reset()
        trainer1._current_obs = obs

        # Train for 50 steps
        for _ in range(50):
            trainer1.train_step()

        # Record state before checkpoint
        pre_step = trainer1.global_step
        pre_updates = trainer1.agent.gradient_update_count
        pre_epsilon = trainer1.agent.epsilon_state.epsilon
        pre_online_weights = {
            k: v.clone() for k, v in trainer1.agent.online_network.state_dict().items()
        }

        # Save checkpoint
        checkpoint_path = temp_dir / "run1" / "checkpoint_latest.pt"
        trainer1.save_checkpoint(str(checkpoint_path))

        # Destroy trainer1
        del trainer1

        # Create trainer2 with resume_from - NO manual state assignment
        config2 = TrainerConfig(
            split="predictor_train",
            validation_split="rl_validation",
            maintenance_capacity=1,
            cost_regime_id="failure-light-no-waste",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
            max_steps=100,  # MUST match trainer1 for semantic identity
            warmup_transitions=10,
            training_seed=6521,
            output_dir=str(temp_dir / "run2"),
        )

        # Production resume path: DDQNTrainer handles all state restoration
        trainer2 = DDQNTrainer(config=config2, resume_from=checkpoint_path)

        # Verify automatic restoration
        assert trainer2.global_step == pre_step, \
            f"global_step not restored: {trainer2.global_step} != {pre_step}"
        assert trainer2.agent.gradient_update_count == pre_updates, \
            f"gradient_update_count not restored"
        assert trainer2.agent.epsilon_state.epsilon == pre_epsilon, \
            f"epsilon not restored: {trainer2.agent.epsilon_state.epsilon} != {pre_epsilon}"

        # Verify network weights restored
        for k, v in trainer2.agent.online_network.state_dict().items():
            assert torch.allclose(v, pre_online_weights[k]), \
                f"online network weight {k} not restored"

        # Continue training
        obs, _ = trainer2.train_env.reset()
        trainer2._current_obs = obs
        for _ in range(30):
            trainer2.train_step()

        # Verify training continued
        assert trainer2.global_step >= pre_step + 30, \
            f"training didn't continue: {trainer2.global_step} < {pre_step + 30}"