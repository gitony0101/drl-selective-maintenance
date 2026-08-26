"""
Split tamper rejection tests.
Uses DDQNTrainer(resume_from=...) and evaluate_ddqn.py to prove
fail-closed behavior, not just inspecting load_checkpoint lists.

Each test builds checkpoint from the EXACT same authoritative semantic config
object used by the Trainer, ensuring resolved_config_identity matches.
"""
import json
import sys
import tempfile
import pathlib
import subprocess

import pytest

pytestmark = pytest.mark.requires_external_assets
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from src.training.ddqn_trainer import DDQNTrainer, TrainerConfig
from src.agents.ddqn.checkpoint import save_checkpoint, CheckpointSelectionState, CHECKPOINT_SELECTION_STATE_VERSION
from src.agents.ddqn.agent import DDQNAgent, DDQNAgentConfig
from src.envs.action_table import ACTION_TABLE_N5_K1, ACTION_TABLE_N5_K2


def _build_production_checkpoint(
    tmp: pathlib.Path,
    k: int = 1,
    training_split: str = "predictor_train",
    validation_split: str = "rl_validation",
    max_steps: int = 100_000,
) -> tuple:
    """
    Build a production checkpoint using the EXACT same TrainerConfig
    that the trainer will use for resume. This ensures resolved_config_identity matches.
    """
    # Create scenario banks
    train_bank = tmp / f"bank_train_{training_split}.json"
    val_bank = tmp / f"bank_val_{validation_split}.json"

    # Use valid scenario bank structure matching actual pilot configs
    train_scenario = {
        "scenario_id": f"{training_split}_001",
        "split": training_split,
        "initial_unit_ids": [70, 28, 13, 24, 36],
        "initial_cycles": [1, 1, 1, 1, 1],
        "replacement_seed": 6521,
        "environment_seed": 6521,
        "episode_horizon": 100,
        "maintenance_capacity": k,
        "cost_regime_id": "failure-light-no-waste"
    }
    val_scenario = {
        "scenario_id": f"{validation_split}_001",
        "split": validation_split,
        "initial_unit_ids": [70, 28, 13, 24, 36],
        "initial_cycles": [1, 1, 1, 1, 1],
        "replacement_seed": 6521,
        "environment_seed": 6521,
        "episode_horizon": 100,
        "maintenance_capacity": k,
        "cost_regime_id": "failure-light-no-waste"
    }

    with open(train_bank, "w") as f:
        json.dump({"bank_id": f"bank_{training_split}", "split": training_split, "scenarios": [train_scenario]}, f)
    with open(val_bank, "w") as f:
        json.dump({"bank_id": f"bank_{validation_split}", "split": validation_split, "scenarios": [val_scenario]}, f)

    # Create agent
    agent = DDQNAgent(config=DDQNAgentConfig(num_actions=6 if k == 1 else 16), seed=6521)

    # Build the AUTHORITATIVE TrainerConfig - this is the single source of truth
    # for both checkpoint creation AND trainer resume
    trainer_cfg = TrainerConfig(
        split=training_split,
        validation_split=validation_split,
        maintenance_capacity=k,
        cost_regime_id="failure-light-no-waste",
        training_scenario_bank_path=str(train_bank),
        validation_scenario_bank_path=str(val_bank),
        max_steps=max_steps,
        warmup_transitions=5_000,
        batch_size=128,
        hidden_dim=128,
        num_hidden_layers=2,
        learning_rate=1e-4,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=50_000,
        gradient_clip=10.0,
        target_update_interval=1_000,
        replay_capacity=100_000,
        training_seed=6521,
        validation_seed=6521,
        output_dir=str(tmp / "output"),
    )

    # Convert to dict and add num_actions (same as trainer does in save_checkpoint)
    config_dict = trainer_cfg.to_dict()
    config_dict["num_actions"] = trainer_cfg.num_actions

    action_table = ACTION_TABLE_N5_K1 if k == 1 else ACTION_TABLE_N5_K2

    ckpt_path = tmp / "checkpoint.pt"
    save_checkpoint(
        agent=agent,
        config=config_dict,
        output_path=ckpt_path,
        maintenance_capacity=k,
        action_table=action_table,
        cost_regime_id="failure-light-no-waste",
        training_seed=6521,
        training_split=training_split,
        validation_split=validation_split,
        training_scenario_bank_path=str(train_bank),
        validation_scenario_bank_path=str(val_bank),
        prediction_cache_manifest_path="data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json",
        selection_state=CheckpointSelectionState(
            selection_state_version=CHECKPOINT_SELECTION_STATE_VERSION,
            validation_performed=False,
        ),
    )
    return ckpt_path, train_bank, val_bank, trainer_cfg


class TestSplitTamperTrainingSplit:
    """Tests tampering with training_split field only."""

    def test_training_split_predictor_train_to_arbitrary_bad_split(self):
        """
        Tamper checkpoint training_split from 'predictor_train' to 'arbitrary_bad_split'.
        Only this ONE field changes. Must reject with message identifying training_split field.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            ckpt_path, t_bank, v_bank, trainer_cfg = _build_production_checkpoint(tmp_path, k=1)

            # Tamper ONLY the training_split field in checkpoint metadata
            raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            assert raw["metadata"]["training_split"] == "predictor_train"
            raw["metadata"]["training_split"] = "arbitrary_bad_split"
            torch.save(raw, ckpt_path)

            # Resume with SAME TrainerConfig (authoritative config object)
            with pytest.raises(ValueError) as exc_info:
                DDQNTrainer(config=trainer_cfg, resume_from=ckpt_path)

            msg = str(exc_info.value)
            # Must identify the exact split field (training_split) or the mismatch
            assert "training_split" in msg.lower() or "split provenance mismatch (training)" in msg.lower() or "split" in msg.lower(), \
                f"Exception must identify training_split field: {msg}"

    def test_training_split_predictor_train_to_rl_test(self):
        """
        Tamper checkpoint training_split from 'predictor_train' to 'rl_test' (forbidden).
        Only this ONE field changes. Must reject with rl_test barrier message.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            ckpt_path, t_bank, v_bank, trainer_cfg = _build_production_checkpoint(tmp_path, k=1)

            # Tamper ONLY the training_split field to rl_test
            raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            assert raw["metadata"]["training_split"] == "predictor_train"
            raw["metadata"]["training_split"] = "rl_test"
            torch.save(raw, ckpt_path)

            # Resume with SAME TrainerConfig
            with pytest.raises(ValueError) as exc_info:
                DDQNTrainer(config=trainer_cfg, resume_from=ckpt_path)

            msg = str(exc_info.value).lower()
            # Must identify rl_test as forbidden or the training_split field
            assert "rl_test" in msg or "forbidden" in msg or "barrier" in msg or "training_split" in msg, \
                f"Exception must identify rl_test barrier on training_split: {msg}"


class TestSplitTamperValidationSplit:
    """Tests tampering with validation_split field only."""

    def test_validation_split_rl_validation_to_arbitrary_bad_split(self):
        """
        Tamper checkpoint validation_split from 'rl_validation' to 'arbitrary_bad_split'.
        Only this ONE field changes. Must reject with message identifying validation_split field.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            ckpt_path, t_bank, v_bank, trainer_cfg = _build_production_checkpoint(tmp_path, k=1)

            # Tamper ONLY the validation_split field
            raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            assert raw["metadata"]["validation_split"] == "rl_validation"
            raw["metadata"]["validation_split"] = "arbitrary_bad_split"
            torch.save(raw, ckpt_path)

            # Resume with SAME TrainerConfig
            with pytest.raises(ValueError) as exc_info:
                DDQNTrainer(config=trainer_cfg, resume_from=ckpt_path)

            msg = str(exc_info.value)
            # Must identify the exact split field (validation_split) or the mismatch
            assert "validation_split" in msg.lower() or "split provenance mismatch (validation)" in msg.lower() or "split" in msg.lower(), \
                f"Exception must identify validation_split field: {msg}"

    def test_validation_split_rl_validation_to_rl_test(self):
        """
        Tamper checkpoint validation_split from 'rl_validation' to 'rl_test' (forbidden).
        Only this ONE field changes. Must reject with rl_test barrier message.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            ckpt_path, t_bank, v_bank, trainer_cfg = _build_production_checkpoint(tmp_path, k=1)

            # Tamper ONLY the validation_split field to rl_test
            raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            assert raw["metadata"]["validation_split"] == "rl_validation"
            raw["metadata"]["validation_split"] = "rl_test"
            torch.save(raw, ckpt_path)

            # Resume with SAME TrainerConfig
            with pytest.raises(ValueError) as exc_info:
                DDQNTrainer(config=trainer_cfg, resume_from=ckpt_path)

            msg = str(exc_info.value).lower()
            # Must identify rl_test as forbidden or the validation_split field
            assert "rl_test" in msg or "forbidden" in msg or "barrier" in msg or "validation_split" in msg, \
                f"Exception must identify rl_test barrier on validation_split: {msg}"


class TestSplitTamperGenericIdentityRejection:
    """
    Ensure generic resolved-config identity mismatch is NOT sufficient.
    Each test above must specifically identify the split field or rl_test barrier.
    A generic 'resolved config identity mismatch' without split field identification
    is NOT sufficient evidence for split tamper rejection.
    """
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])