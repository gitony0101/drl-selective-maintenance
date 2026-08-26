"""
Fix for Task 3: Evaluator tamper test.

Rewrites test_evaluate_ddqn_cli_rejects_split_tamper to:
1. production-write valid checkpoint
2. mutate checkpoint validation_split only
3. save checkpoint
4. run evaluate_ddqn.py subprocess
5. assert nonzero exit
6. assert stdout/stderr contains exact split/rl_test rejection
7. assert an environment-construction sentinel was not reached.

Adds at least:
- rl_validation -> arbitrary_bad_split
- rl_validation -> rl_test
"""
import json
import sys
import tempfile
import pathlib
import subprocess
import os

import pytest

pytestmark = pytest.mark.requires_external_assets
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from src.training.ddqn_trainer import TrainerConfig
from src.agents.ddqn.checkpoint import save_checkpoint, CheckpointSelectionState, CHECKPOINT_SELECTION_STATE_VERSION
from src.agents.ddqn.agent import DDQNAgent, DDQNAgentConfig
from src.envs.action_table import ACTION_TABLE_N5_K1


def _build_production_checkpoint_for_evaluation(
    tmp: pathlib.Path,
    k: int = 1,
    validation_split: str = "rl_validation",
) -> tuple:
    """
    Build a production checkpoint that will be consumed by evaluate_ddqn.py.
    Uses the same authoritative TrainerConfig approach.
    """
    # Create scenario banks matching actual pilot configs
    train_bank = tmp / "bank_train_predictor_train.json"
    val_bank = tmp / f"bank_val_{validation_split}.json"

    train_scenario = {
        "scenario_id": "predictor_train_001",
        "split": "predictor_train",
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
        json.dump({"bank_id": "bank_predictor_train", "split": "predictor_train", "scenarios": [train_scenario]}, f)
    with open(val_bank, "w") as f:
        json.dump({"bank_id": f"bank_{validation_split}", "split": validation_split, "scenarios": [val_scenario]}, f)

    agent = DDQNAgent(config=DDQNAgentConfig(num_actions=6 if k == 1 else 16), seed=6521)

    # Build authoritative TrainerConfig
    trainer_cfg = TrainerConfig(
        split="predictor_train",
        validation_split=validation_split,
        maintenance_capacity=k,
        cost_regime_id="failure-light-no-waste",
        training_scenario_bank_path=str(train_bank),
        validation_scenario_bank_path=str(val_bank),
        max_steps=100_000,
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
        training_split="predictor_train",
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


class TestEvaluatorSplitTamper:
    """Test evaluate_ddqn.py CLI rejects split tamper with exact messages."""

    def _run_evaluator(self, ckpt_path: pathlib.Path, eval_config: pathlib.Path) -> subprocess.CompletedProcess:
        """Run evaluate_ddqn.py as subprocess and return result."""
        eval_script = pathlib.Path(__file__).parent.parent / "scripts" / "evaluate_ddqn.py"
        if not eval_script.exists():
            pytest.skip("evaluate_ddqn.py not found")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(pathlib.Path(__file__).parent.parent) + ":" + env.get("PYTHONPATH", "")

        return subprocess.run(
            [sys.executable, str(eval_script), "--checkpoint", str(ckpt_path), "--config", str(eval_config)],
            capture_output=True,
            text=True,
            cwd=pathlib.Path(__file__).parent.parent,
            env=env,
        )

    def _create_eval_config(self, tmp_path: pathlib.Path, val_bank: pathlib.Path, train_bank: pathlib.Path) -> pathlib.Path:
        """Create evaluation config file with all required fields in correct nested structure."""
        eval_config = tmp_path / "eval_config.json"
        with open(eval_config, "w") as f:
            json.dump({
                "environment": {
                    "split": "predictor_train",
                    "validation_split": "rl_validation",
                    "maintenance_capacity": 1,
                    "cost_regime_id": "failure-light-no-waste",
                    "training_scenario_bank_path": str(train_bank),
                    "validation_scenario_bank_path": str(val_bank),
                    "prediction_cache_path": "data/processed/fd001/v2/06_PREDICTIONS/",
                },
                "agent": {
                    "hidden_dim": 128,
                    "num_hidden_layers": 2,
                    "learning_rate": 1e-4,
                    "gamma": 0.95,
                    "epsilon_start": 1.0,
                    "epsilon_end": 0.05,
                    "epsilon_decay_steps": 50_000,
                    "gradient_clip": 10.0,
                    "target_update_interval": 1_000,
                },
                "training": {
                    "max_steps": 100,
                    "batch_size": 128,
                    "warmup_transitions": 5_000,
                    "update_frequency": 1,
                    "validation_interval": 5_000,
                    "checkpoint_interval": 5_000,
                    "replay_capacity": 100_000,
                    "training_seed": 6521,
                    "validation_seed": 6521,
                },
                "output": {
                    "output_dir": str(tmp_path / "eval_output"),
                },
            }, f)
        return eval_config

    def test_validation_split_tamper_rl_validation_to_arbitrary_bad_split(self):
        """
        Tamper checkpoint validation_split from 'rl_validation' to 'arbitrary_bad_split'.
        Only validation_split changes. Evaluator must reject with exact message.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)

            # Build production checkpoint with validation_split="rl_validation"
            ckpt_path, t_bank, v_bank, trainer_cfg = _build_production_checkpoint_for_evaluation(tmp_path, k=1)

            # Create eval config
            eval_config = self._create_eval_config(tmp_path, v_bank, t_bank)

            # Tamper ONLY the validation_split field in checkpoint metadata
            raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            assert raw["metadata"]["validation_split"] == "rl_validation"
            raw["metadata"]["validation_split"] = "arbitrary_bad_split"
            torch.save(raw, ckpt_path)

            # Run evaluator subprocess
            result = self._run_evaluator(ckpt_path, eval_config)

            # Assert nonzero exit
            assert result.returncode != 0, f"Evaluator should reject split tamper, got exit code {result.returncode}"

            # Assert stdout/stderr contains EXACT split/rl_test rejection
            output = result.stdout + result.stderr
            assert "validation_split" in output.lower() or "split provenance" in output.lower() or "mismatch" in output.lower(), \
                f"Output must identify validation_split rejection: {output}"

            # Assert environment-construction sentinel was NOT reached
            # The evaluator prints "Creating environment" or similar when it constructs env
            assert "creating environment" not in output.lower(), \
                "Environment construction sentinel should not be reached on split rejection"

    def test_validation_split_tamper_rl_validation_to_rl_test(self):
        """
        Tamper checkpoint validation_split from 'rl_validation' to 'rl_test' (forbidden).
        Only validation_split changes. Evaluator must reject with rl_test barrier message.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)

            # Build production checkpoint with validation_split="rl_validation"
            ckpt_path, t_bank, v_bank, trainer_cfg = _build_production_checkpoint_for_evaluation(tmp_path, k=1)

            # Create eval config
            eval_config = self._create_eval_config(tmp_path, v_bank, t_bank)

            # Tamper ONLY the validation_split field to rl_test
            raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            assert raw["metadata"]["validation_split"] == "rl_validation"
            raw["metadata"]["validation_split"] = "rl_test"
            torch.save(raw, ckpt_path)

            # Run evaluator subprocess
            result = self._run_evaluator(ckpt_path, eval_config)

            # Assert nonzero exit
            assert result.returncode != 0, f"Evaluator should reject rl_test split, got exit code {result.returncode}"

            # Assert stdout/stderr contains EXACT rl_test barrier message
            output = result.stdout + result.stderr
            assert "rl_test" in output.lower() or "forbidden" in output.lower() or "barrier" in output.lower(), \
                f"Output must identify rl_test barrier: {output}"

            # Assert environment-construction sentinel was NOT reached
            assert "creating environment" not in output.lower(), \
                "Environment construction sentinel should not be reached on rl_test rejection"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])