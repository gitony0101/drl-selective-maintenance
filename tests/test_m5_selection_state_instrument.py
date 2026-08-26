"""
Instrumentation test: verify CheckpointSelectionState validates before mutable
state restoration, and that trainer.agent state remains unchanged when resume
is rejected.

This test validates the atomic-resume contract:
1. Construct trainer
2. Snapshot trainer.agent and trainer mutable state
3. Tamper checkpoint
4. Call trainer._resume_from_checkpoint(...)
5. Assert failure
6. Snapshot trainer.agent and trainer state again
7. Compare EVERY component
"""
import sys
import tempfile
import pathlib
import copy
import json

import torch
import numpy as np
import pytest

pytestmark = pytest.mark.requires_external_assets

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from src.agents.ddqn.agent import DDQNAgent, DDQNAgentConfig
from src.agents.ddqn.checkpoint import (
    save_checkpoint, load_checkpoint, CheckpointSelectionState,
    CHECKPOINT_SELECTION_STATE_VERSION, CHECKPOINT_SCHEMA_VERSION,
    compute_action_table_hash, compute_scenario_bank_content_hash
)
from src.training.ddqn_trainer import DDQNTrainer, TrainerConfig
from src.envs.action_table import ACTION_TABLE_N5_K1


def snapshot_trainer_state(trainer: DDQNTrainer) -> dict:
    """Snapshot ALL mutable state of trainer and its agent."""
    agent = trainer.agent
    replay = trainer.replay_buffer

    return {
        # Agent state
        "agent": {
            "online_state": {k: v.clone().cpu().numpy() for k, v in agent.online_network.state_dict().items()},
            "target_state": {k: v.clone().cpu().numpy() for k, v in agent.target_network.state_dict().items()},
            "optimizer_state": copy.deepcopy(agent.optimizer.state_dict()),
            "global_step": agent.global_step,
            "gradient_update_count": agent.gradient_update_count,
            "epsilon_state": copy.deepcopy(agent.epsilon_state.to_dict()),
        },
        # Trainer state
        "trainer": {
            "global_step": trainer.global_step,
            "episode_count": trainer.episode_count,
            "current_episode_return": trainer.current_episode_return,
            "current_episode_cost": copy.deepcopy(trainer.current_episode_cost),
        },
        # Replay buffer state
        "replay": {
            "current_size": replay.current_size,
            "write_index": replay.write_index,
            "observations": replay.observations.copy() if hasattr(replay, 'observations') and replay.observations is not None else None,
            "actions": replay.actions.copy() if hasattr(replay, 'actions') and replay.actions is not None else None,
            "rewards": replay.rewards.copy() if hasattr(replay, 'rewards') and replay.rewards is not None else None,
            "next_observations": replay.next_observations.copy() if hasattr(replay, 'next_observations') and replay.next_observations is not None else None,
            "terminated": replay.terminated.copy() if hasattr(replay, 'terminated') and replay.terminated is not None else None,
            "truncated": replay.truncated.copy() if hasattr(replay, 'truncated') and replay.truncated is not None else None,
            "rng_state": copy.deepcopy(replay.rng.bit_generator.state) if hasattr(replay, 'rng') and replay.rng is not None else None,
        },
        # RNG states
        "rng": {
            "python": copy.deepcopy(random.getstate()),
            "numpy": copy.deepcopy(np.random.get_state()),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [t.clone() for t in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
        },
        # Checkpoint selection state
        "selection_state": copy.deepcopy(trainer._selection_state.to_dict()) if hasattr(trainer, '_selection_state') else None,
    }


import random


def compare_snapshots(before: dict, after: dict) -> list:
    """Compare two state snapshots, return list of mismatches."""
    mismatches = []

    # Compare agent online network
    for k in before["agent"]["online_state"]:
        if not np.array_equal(before["agent"]["online_state"][k], after["agent"]["online_state"][k]):
            mismatches.append(f"agent.online_network.{k}")

    # Compare agent target network
    for k in before["agent"]["target_state"]:
        if not np.array_equal(before["agent"]["target_state"][k], after["agent"]["target_state"][k]):
            mismatches.append(f"agent.target_network.{k}")

    # Compare optimizer state (check keys and tensor values)
    opt_before = before["agent"]["optimizer_state"]
    opt_after = after["agent"]["optimizer_state"]
    if set(opt_before.keys()) != set(opt_after.keys()):
        mismatches.append("agent.optimizer_state.keys")
    else:
        for k in opt_before:
            if isinstance(opt_before[k], dict) and "state" in opt_before[k]:
                # Compare optimizer param state
                for pk, pv in opt_before[k].get("state", {}).items():
                    if pk in opt_after[k].get("state", {}):
                        for tensor_k, tensor_v in pv.items():
                            if isinstance(tensor_v, torch.Tensor):
                                if not torch.allclose(tensor_v, opt_after[k]["state"][pk][tensor_k]):
                                    mismatches.append(f"agent.optimizer_state[{k}].state[{pk}].{tensor_k}")
                            else:
                                if tensor_v != opt_after[k]["state"][pk][tensor_k]:
                                    mismatches.append(f"agent.optimizer_state[{k}].state[{pk}].{tensor_k}")

    # Compare agent counters
    for field in ["global_step", "gradient_update_count"]:
        if before["agent"][field] != after["agent"][field]:
            mismatches.append(f"agent.{field}")

    # Compare epsilon state
    eps_before = before["agent"]["epsilon_state"]
    eps_after = after["agent"]["epsilon_state"]
    for field in ["epsilon", "epsilon_start", "epsilon_end", "epsilon_decay_steps", "global_step"]:
        if eps_before.get(field) != eps_after.get(field):
            mismatches.append(f"agent.epsilon_state.{field}")

    # Compare trainer state
    for field in ["global_step", "episode_count", "current_episode_return"]:
        if before["trainer"][field] != after["trainer"][field]:
            mismatches.append(f"trainer.{field}")

    # Compare replay buffer
    for field in ["current_size", "write_index"]:
        if before["replay"][field] != after["replay"][field]:
            mismatches.append(f"replay.{field}")

    # Compare replay arrays if present
    for field in ["observations", "actions", "rewards", "next_observations", "terminated", "truncated"]:
        if before["replay"][field] is not None and after["replay"][field] is not None:
            if not np.array_equal(before["replay"][field], after["replay"][field]):
                mismatches.append(f"replay.{field}")
        elif before["replay"][field] != after["replay"][field]:
            mismatches.append(f"replay.{field} (None mismatch)")

    # Compare replay RNG state
    if before["replay"]["rng_state"] is not None and after["replay"]["rng_state"] is not None:
        if before["replay"]["rng_state"] != after["replay"]["rng_state"]:
            mismatches.append("replay.rng_state")
    elif before["replay"]["rng_state"] != after["replay"]["rng_state"]:
        mismatches.append("replay.rng_state (None mismatch)")

    # Compare RNG states
    py_before = before["rng"]["python"]
    py_after = after["rng"]["python"]
    if py_before != py_after:
        mismatches.append("rng.python")

    np_before = before["rng"]["numpy"]
    np_after = after["rng"]["numpy"]
    if np_before[0] != np_after[0] or not np.array_equal(np_before[1], np_after[1]):
        mismatches.append("rng.numpy")

    if not torch.equal(before["rng"]["torch_cpu"], after["rng"]["torch_cpu"]):
        mismatches.append("rng.torch_cpu")

    if before["rng"]["torch_cuda"] is not None and after["rng"]["torch_cuda"] is not None:
        for i, (b, a) in enumerate(zip(before["rng"]["torch_cuda"], after["rng"]["torch_cuda"])):
            if not torch.equal(b, a):
                mismatches.append(f"rng.torch_cuda[{i}]")
    elif before["rng"]["torch_cuda"] != after["rng"]["torch_cuda"]:
        mismatches.append("rng.torch_cuda (None mismatch)")

    # Compare selection state
    if before["selection_state"] != after["selection_state"]:
        mismatches.append("selection_state")

    return mismatches


def test_selection_state_validates_before_mutation():
    """Test that _resume_from_checkpoint validates selection_state BEFORE mutating any state."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)

        # Create a minimal valid checkpoint
        agent = DDQNAgent(config=DDQNAgentConfig(num_actions=6), seed=6521)
        train_bank = tmp_path / "bank_train.json"
        val_bank = tmp_path / "bank_val.json"
        for p, s in [(train_bank, "predictor_train"), (val_bank, "rl_validation")]:
            with open(p, "w") as f:
                json.dump({
                    "bank_id": f"bank_{s}", "split": s,
                    "scenarios": [{
                        "scenario_id": f"{s}_001", "split": s,
                        "initial_unit_ids": [1]*5, "initial_cycles": [1]*5,
                        "replacement_seed": 6500, "environment_seed": 6500,
                        "episode_horizon": 100, "maintenance_capacity": 1,
                        "cost_regime_id": "failure-light-no-waste"
                    }]
                }, f)

        ckpt_path = tmp_path / "checkpoint.pt"
        save_checkpoint(
            agent=agent, config={"hidden_dim": 128, "num_hidden_layers": 2},
            output_path=ckpt_path, maintenance_capacity=1, action_table=ACTION_TABLE_N5_K1,
            cost_regime_id="failure-light-no-waste", training_seed=6521,
            training_split="predictor_train", validation_split="rl_validation",
            training_scenario_bank_path=str(train_bank), validation_scenario_bank_path=str(val_bank),
            prediction_cache_manifest_path="data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json",
            selection_state=CheckpointSelectionState(
                selection_state_version=CHECKPOINT_SELECTION_STATE_VERSION,
                validation_performed=False,
            ),
        )

        # Create trainer (which creates its own agent internally)
        cfg = TrainerConfig(
            split="predictor_train", validation_split="rl_validation",
            maintenance_capacity=1, cost_regime_id="failure-light-no-waste",
            training_scenario_bank_path=str(train_bank), validation_scenario_bank_path=str(val_bank),
            max_steps=100, warmup_transitions=5, training_seed=6521,
            output_dir=str(tmp_path / "resume"), run_id="resume_test",
        )
        trainer = DDQNTrainer(config=cfg)

        # STEP 2: Snapshot trainer.agent and trainer mutable state BEFORE tampering
        snapshot_before = snapshot_trainer_state(trainer)

        # STEP 3: Tamper checkpoint - corrupt selection_state (wrong version)
        raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        raw["metadata"]["selection_state"] = {
            "selection_state_version": 99,  # wrong version - should fail validation
            "validation_performed": False,
            "best_validation_mean_cost": None,
            "best_checkpoint_global_step": None,
            "best_checkpoint_artifact_name": None,
            "best_validation_failure_count": None,
            "best_validation_worst_10_pct_cost": None,
            "comparator_identity": "mean_cost_v1",
            "equal_metric_tie_behavior": "keep_first",
        }
        torch.save(raw, str(ckpt_path))

        # STEP 4: Call trainer._resume_from_checkpoint(...) - must fail
        with pytest.raises(ValueError) as exc_info:
            trainer._resume_from_checkpoint(pathlib.Path(ckpt_path))

        # STEP 5: Assert failure with correct error message
        msg = str(exc_info.value).lower()
        assert "selection_state_version" in msg or "expected" in msg, f"Wrong error message: {msg}"

        # STEP 6: Snapshot trainer.agent and trainer state AGAIN
        snapshot_after = snapshot_trainer_state(trainer)

        # STEP 7: Compare EVERY component - must be identical (no mutation)
        mismatches = compare_snapshots(snapshot_before, snapshot_after)

        assert not mismatches, (
            f"State was mutated before validation failure! Mismatches:\n" + "\n".join(mismatches)
        )
