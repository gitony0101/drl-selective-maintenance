"""E3 (M10) n-step DDQN formal trainer with controlled warmup (A/B/C/D).

Runs one cell x seed training to the frozen 100000-raw-transition budget,
reusing the frozen ``DDQNAgent`` (architecture, optimizer, Huber loss, grad
clip, target sync, epsilon schedule) and ``checkpoint.save_checkpoint`` while
replacing the replay path with the generic episode-aware n-step pipeline and
supporting two warmup modes (Section 15/16/17/18):

  Cell A: n=1, standard warmup
  Cell B: n=3, standard warmup
  Cell C: n=1, seeded warmup  (consumes the frozen seeded-warmup manifest)
  Cell D: n=3, seeded warmup  (consumes the SAME manifest as C for that seed)

Accounting (Section 17): all four cells consume exactly ``max_steps`` raw
transitions. For seeded cells the 5000 manifest transitions count as the first
5000 raw transitions; the first gradient update sees
global_step == epsilon_step == 5000, then 95000 online transitions follow.
Standard cells advance epsilon/global_step identically during their 5000-step
warmup. Validation (rl_validation, epsilon=0) and checkpoint selection run on
the frozen scheduling (every ``validation_interval`` steps, lowest mean cost,
``keep_first`` tie behavior).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.agents.ddqn import DDQNAgent, DDQNAgentConfig
from src.agents.ddqn.checkpoint import (
    CheckpointSelectionState,
    CHECKPOINT_SELECTION_STATE_VERSION,
    compute_action_table_hash,
    compute_network_architecture_id,
    compute_scenario_bank_content_hash,
    get_git_commit,
    save_checkpoint,
)
from src.envs.action_table import ACTION_TABLE_N5_K2
from src.envs.config import get_default_config, ALLOWED_SPLITS
from src.envs.scenario_bank import load_scenario_bank
from src.envs.selective_maintenance_env import SelectiveMaintenanceEnv
from src.training.ddqn_config_identity import compute_resolved_config_identity
from src.agents.ddqn.q_network import resolve_device


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FORMAL_SEEDS = (6521, 6522, 6523, 6524, 6525)


@dataclass(frozen=True)
class E3CellConfig:
    """Frozen E3 cell hyperparameters (Section 4). All fields mirror the frozen
    ``configs/agents/ddqn_v1.json`` exactly for the evaluated regime."""

    observation_dim: int = 10
    num_actions: int = 16          # K=2
    hidden_dim: int = 128
    num_hidden_layers: int = 2
    learning_rate: float = 1e-4
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    gradient_clip: float = 10.0
    target_update_interval: int = 1_000
    use_huber_loss: bool = True
    huber_delta: float = 1.0
    batch_size: int = 128
    replay_capacity: int = 100_000
    warmup_transitions: int = 5_000
    update_frequency: int = 1
    validation_interval: int = 5_000
    checkpoint_interval: int = 5_000
    max_steps: int = 100_000

    # E3 treatment
    n: int = 1                      # effective n-step horizon (1 or 3)
    cell: str = "A"                 # A/B/C/D
    warmup_mode: str = "standard"   # "standard" | "seeded"


def cell_label_to(n: int, mode: str) -> str:
    if n == 1 and mode == "standard":
        return "A"
    if n == 3 and mode == "standard":
        return "B"
    if n == 1 and mode == "seeded":
        return "C"
    if n == 3 and mode == "seeded":
        return "D"
    raise ValueError(f"unknown (n={n}, mode={mode})")


# ---------------------------------------------------------------------------
# Env + warmup helpers
# ---------------------------------------------------------------------------

def build_train_env(env_seed: int, cache_path: str, split: str, bank: str):
    cfg = get_default_config(
        split=split, cost_regime_id="failure-light-no-waste",
        maintenance_capacity=2, scenario_bank_path=bank,
        prediction_cache_path=cache_path, seed=env_seed, info_mode="normal",
    )
    bank_obj = load_scenario_bank(bank)
    return cfg, SelectiveMaintenanceEnv(config=cfg, scenario_bank=bank_obj, info_mode="normal")


# ---------------------------------------------------------------------------
# E3 Trainer
# ---------------------------------------------------------------------------

class E3Trainer:
    """Frozen-hyperparameter n-step DDQN trainer for one cell x seed."""

    def __init__(
        self,
        seed: int,
        cell_cfg: E3CellConfig,
        cache_path: str,
        output_dir: str,
        run_id: str,
        training_bank: str = "configs/scenarios/m5_pilot_k2.json",
        validation_bank: str = "configs/scenarios/m5_validation_k2.json",
        device: Optional[str] = None,
        seeded_manifest_path: Optional[Path] = None,
    ) -> None:
        self.seed = seed
        self.cfg = cell_cfg
        self.cell = cell_cfg.cell
        self.n = cell_cfg.n
        self.mode = cell_cfg.warmup_mode
        self.cache_path = cache_path
        self.run_id = run_id

        # Reproducible seeding (mirror set_all_seeds in ddqn_trainer)
        import random, torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = resolve_device(device)

        # Training env seed = training seed (frozen protocol); validation env
        # seed = frozen validation_seed (6521), NOT the training seed (the M9
        # wrapper only overrides --training-seed; validation_seed stays 6521).
        self.frozen_validation_seed = 6521
        self.train_cfg, self.train_env = build_train_env(
            seed, cache_path, "predictor_train", training_bank)
        self.val_cfg, self.val_env = build_train_env(
            self.frozen_validation_seed, cache_path, "rl_validation", validation_bank)

        agent_cfg = DDQNAgentConfig(
            observation_dim=self.cfg.observation_dim,
            num_actions=self.cfg.num_actions,
            hidden_dim=self.cfg.hidden_dim,
            num_hidden_layers=self.cfg.num_hidden_layers,
            learning_rate=self.cfg.learning_rate,
            gamma=self.cfg.gamma,
            epsilon_start=self.cfg.epsilon_start,
            epsilon_end=self.cfg.epsilon_end,
            epsilon_decay_steps=self.cfg.epsilon_decay_steps,
            gradient_clip=self.cfg.gradient_clip,
            target_update_interval=self.cfg.target_update_interval,
            use_huber_loss=self.cfg.use_huber_loss,
            huber_delta=self.cfg.huber_delta,
            explicit_device=device,
        )
        self.agent = DDQNAgent(config=agent_cfg, seed=seed)

        from src.milestone10.e3.nstep import NStepReplayBuffer
        self.replay = NStepReplayBuffer(
            capacity=self.cfg.replay_capacity, observation_dim=self.cfg.observation_dim,
            seed=seed)

        self.global_step = 0
        self.seeded_manifest_path = seeded_manifest_path

        self.run_dir = Path(output_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.metrics: Dict[str, Any] = {"step_metrics": [], "validation": [],
                                        "episode_returns": [], "episode_lengths": []}
        self._selection = CheckpointSelectionState(
            selection_state_version=CHECKPOINT_SELECTION_STATE_VERSION,
            equal_metric_tie_behavior="keep_first")
        self.best_validation_mean_cost: Optional[float] = None
        self.best_checkpoint_path: Optional[str] = None

    # -- n-step pipeline ----------------------------------------------------

    def _insert_raw_frame(self, obs, action_id, reward, next_obs, terminated, truncated):
        from src.milestone10.e3.nstep import EpisodeNStepBuffer, RawStep, NStepTransition
        if not hasattr(self, "_episode_buf"):
            self._episode_buf = EpisodeNStepBuffer(self.n, self.cfg.gamma)
        raw = RawStep(
            observation=np.asarray(obs, dtype=np.float32), action_id=int(action_id),
            reward=float(reward),
            next_observation=np.asarray(next_obs, dtype=np.float32),
            terminated=bool(terminated), truncated=bool(truncated))
        emitted = self._episode_buf.push(raw)
        flushed = False
        if truncated or terminated:
            emitted += self._episode_buf.flush_episode()
            self._episode_buf = EpisodeNStepBuffer(self.n, self.cfg.gamma)
            flushed = True
        for tr in emitted:
            self.replay.insert(tr)
        return flushed

    def _insert_seeded_frames(self, manifest_path: Path) -> None:
        from src.milestone10.e3.nstep import EpisodeNStepBuffer, RawStep
        frames = []
        with open(manifest_path) as f:
            for line in f:
                frames.append(json.loads(line))
        buf = EpisodeNStepBuffer(self.n, self.cfg.gamma)
        for d in frames:
            raw = RawStep(
                observation=np.asarray(d["observation_t"], dtype=np.float32),
                action_id=int(d["action_id_t"]), reward=float(d["reward_t"]),
                next_observation=np.asarray(d["next_observation_t"], dtype=np.float32),
                terminated=bool(d["terminated_t"]), truncated=bool(d["truncated_t"]))
            emitted = buf.push(raw)
            if d["truncated_t"] or d["terminated_t"]:
                emitted += buf.flush_episode()
                buf = EpisodeNStepBuffer(self.n, self.cfg.gamma)
            for tr in emitted:
                self.replay.insert(tr)
        if buf.pending:
            for tr in buf.flush_episode():
                self.replay.insert(tr)

    # -- env stepping -------------------------------------------------------

    def _warmup(self) -> None:
        """Standard warmup: step the env with epsilon-greedy actions filling the
        n-step buffer for ``warmup_transitions`` raw transitions, advancing
        global_step and epsilon exactly as canonical (Section 18)."""
        from src.milestone10.e3.nstep import NStepTransition, RawStep
        self._current_obs, _ = self.train_env.reset()
        for _ in range(self.cfg.warmup_transitions):
            action = self.agent.select_action(self._current_obs, training=True)
            next_obs, reward, terminated, truncated, info = self.train_env.step(action)
            flushed = self._insert_raw_frame(self._current_obs, action, reward, next_obs,
                                             terminated, truncated)
            self._record_episode(reward, info, truncated)
            if truncated:
                self._current_obs, _ = self.train_env.reset()
            else:
                self._current_obs = next_obs
            self.global_step += 1
            self.agent.global_step = self.global_step
            self.agent.epsilon_state.step()

    def _record_episode(self, reward, info, truncated):
        if truncated:
            self.metrics["episode_lengths"].append(info.get("step_index"))
        self.metrics["episode_returns"].append(float(reward))

    # -- core training loop -------------------------------------------------

    def train(self) -> Dict[str, Any]:
        n = self.n
        batch_size = self.cfg.batch_size
        max_steps = self.cfg.max_steps

        if self.mode == "seeded":
            # Load frozen seeded warmup -> convert n-step -> buffer. The 5000
            # manifest transitions are the first 5000 raw transitions.
            self._insert_seeded_frames(self.seeded_manifest_path)
            self.global_step = self.cfg.warmup_transitions
            self.agent.global_step = self.global_step
            self.agent.epsilon_state.global_step = self.cfg.warmup_transitions
            self._current_obs, _ = self.train_env.reset()
        else:
            self._warmup()  # global_step == warmup_transitions now

        from src.milestone10.e3.agent_update import update_nstep
        from src.milestone10.e3.nstep import NStepTransition  # noqa

        # Online learning phase: finish the raw transition budget.
        while self.global_step < max_steps:
            action = self.agent.select_action(self._current_obs, training=True)
            next_obs, reward, terminated, truncated, info = self.train_env.step(action)
            self._insert_raw_frame(self._current_obs, action, reward, next_obs,
                                   terminated, truncated)
            if truncated:
                self.metrics["episode_returns"].append(float(reward))
                self.metrics["episode_lengths"].append(info.get("step_index"))
                self._current_obs, _ = self.train_env.reset()
            else:
                self._current_obs = next_obs

            self.global_step += 1
            self.agent.global_step = self.global_step
            self.agent.epsilon_state.step()

            # n-step gradient update every step after warmup buffer is filled.
            if len(self.replay) >= self.cfg.warmup_transitions and len(self.replay) >= batch_size:
                # (update_frequency == 1 in frozen config)
                batch = self.replay.sample_batch(batch_size)
                metrics = update_nstep(self.agent, batch)
                self.metrics["step_metrics"].append({"global_step": self.global_step, **metrics})
                target_synced = self.agent.maybe_sync_target()

                if self.global_step % self.cfg.validation_interval == 0:
                    self._validate_and_select()

        # Final validation + checkpoint at completion boundary.
        self._validate_and_select(force=True)
        self._save_latest()
        self._write_manifest()
        return self.metrics

    def _validate_and_select(self, force: bool = False) -> None:
        from src.training.ddqn_trainer import compute_validation_metrics
        val_metrics = compute_validation_metrics(
            self.val_env, self.agent,
            num_scenarios=min(10, len(self.val_env.scenario_bank.scenarios)),
            max_steps_per_scenario=100,
        )
        val_metrics["global_step"] = self.global_step
        self.metrics["validation"].append(val_metrics)

        current_mean = val_metrics["mean_total_cost"]
        should_update_best = False
        if not self._selection.validation_performed:
            should_update_best = True
        elif current_mean < self._selection.best_validation_mean_cost:
            should_update_best = True
        elif current_mean == self._selection.best_validation_mean_cost:
            should_update_best = self._selection.equal_metric_tie_behavior != "keep_first"

        if should_update_best:
            self._selection = CheckpointSelectionState(
                selection_state_version=CHECKPOINT_SELECTION_STATE_VERSION,
                validation_performed=True,
                best_validation_mean_cost=current_mean,
                best_checkpoint_global_step=self.global_step,
                best_checkpoint_artifact_name="checkpoint_best.pt",
                best_validation_failure_count=val_metrics.get("total_failures"),
                best_validation_worst_10_pct_cost=val_metrics.get("worst_10_pct_cost"),
                comparator_identity="mean_cost_v1",
                equal_metric_tie_behavior="keep_first")
            path = self._save_checkpoint("checkpoint_best.pt", val_metrics)
            self.best_checkpoint_path = path
            self.best_validation_mean_cost = current_mean

    def _save_checkpoint(self, name: str, validation_metrics: Optional[Dict[str, Any]] = None) -> str:
        checkpoint_path = self.run_dir / name
        config_dict = self._resolved_config_dict()
        save_checkpoint(
            agent=self.agent,
            config=config_dict,
            output_path=checkpoint_path,
            maintenance_capacity=2,
            action_table=ACTION_TABLE_N5_K2,
            cost_regime_id="failure-light-no-waste",
            training_seed=self.seed,
            replay_buffer=None,
            training_split="predictor_train",
            validation_split="rl_validation",
            training_scenario_bank_path="configs/scenarios/m5_pilot_k2.json",
            validation_scenario_bank_path="configs/scenarios/m5_validation_k2.json",
            prediction_cache_manifest_path=f"{self.cache_path.rstrip('/')}/prediction_cache_manifest_v2.json",
            selection_state=self._selection,
            validation_metrics=validation_metrics,
        )
        return str(checkpoint_path)

    def _save_latest(self) -> None:
        self._save_checkpoint("checkpoint_latest.pt")

    def _resolved_config_dict(self) -> Dict[str, Any]:
        return {
            "observation_dim": self.cfg.observation_dim,
            "num_actions": self.cfg.num_actions,
            "hidden_dim": self.cfg.hidden_dim,
            "num_hidden_layers": self.cfg.num_hidden_layers,
            "learning_rate": self.cfg.learning_rate,
            "gamma": self.cfg.gamma,
            "epsilon_start": self.cfg.epsilon_start,
            "epsilon_end": self.cfg.epsilon_end,
            "epsilon_decay_steps": self.cfg.epsilon_decay_steps,
            "gradient_clip": self.cfg.gradient_clip,
            "target_update_interval": self.cfg.target_update_interval,
            "batch_size": self.cfg.batch_size,
            "replay_capacity": self.cfg.replay_capacity,
            "warmup_transitions": self.cfg.warmup_transitions,
            "max_steps": self.cfg.max_steps,
            "split": "predictor_train",
            "validation_split": "rl_validation",
            "maintenance_capacity": 2,
            "cost_regime_id": "failure-light-no-waste",
            "use_huber_loss": True,
            "huber_delta": 1.0,
        }

    def _write_manifest(self) -> None:
        manifest = {
            "run_id": self.run_id,
            "cell": self.cell,
            "n": self.n,
            "warmup_mode": self.mode,
            "training_seed": self.seed,
            "final_global_step": self.global_step,
            "max_steps": self.cfg.max_steps,
            "status": "COMPLETE" if self.global_step >= self.cfg.max_steps else "INCOMPLETE",
            "training_split": "predictor_train",
            "validation_split": "rl_validation",
            "training_scenario_bank": "configs/scenarios/m5_pilot_k2.json",
            "validation_scenario_bank": "configs/scenarios/m5_validation_k2.json",
            "prediction_cache_path": self.cache_path,
            "seeded_manifest_path": str(self.seeded_manifest_path) if self.seeded_manifest_path else None,
            "best_checkpoint_path": self.best_checkpoint_path,
            "best_validation_mean_cost": self.best_validation_mean_cost,
            "num_validation_episodes": len(self.metrics["validation"]),
            "num_gradient_updates": len(self.metrics["step_metrics"]),
            "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": get_git_commit(),
        }
        (self.run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
        (self.run_dir / "step_metrics.jsonl").write_text(
            "\n".join(json.dumps(m) for m in self.metrics["step_metrics"]))
        (self.run_dir / "validation_metrics.json").write_text(
            json.dumps(self.metrics["validation"], indent=2))