"""
DDQN Trainer for Milestone 5 Point-Estimate Double DQN.

Implements:
- Training loop with environment interaction
- Periodic validation (epsilon=0, no updates)
- Checkpoint selection (lowest validation cost)
- Artifact emission (metrics, manifests)
- rl_test rejection barrier
"""

from __future__ import annotations

import json
import csv
import random
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import hashlib

import numpy as np
import torch

from src.envs.selective_maintenance_env import SelectiveMaintenanceEnv
from src.envs.config import EnvironmentConfig, get_default_config, ALLOWED_SPLITS
from src.envs.action_table import ACTION_TABLE_N5_K1, ACTION_TABLE_N5_K2, ActionSubset
from src.envs.scenario_bank import ScenarioBank

from src.agents.ddqn import DDQNAgent, ReplayBuffer, DDQNAgentConfig
from src.agents.ddqn.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    compute_file_hash,
    get_git_commit,
    compute_action_table_hash,
    compute_network_architecture_id,
    compute_scenario_bank_content_hash,
    CHECKPOINT_SCHEMA_VERSION,
    CHECKPOINT_SELECTION_STATE_VERSION,
    CheckpointSelectionState,
)


from src.training.ddqn_config_identity import (
    compute_resolved_config_identity,
    validate_resolved_config_identity,
)

# Remove duplicate contract; the single authoritative helper is in ddqn_config_identity.


@dataclass(frozen=True)
class TrainerConfig:
    """Immutable DDQN trainer configuration."""

    # Environment
    split: str = "predictor_train"
    validation_split: str = "rl_validation"
    maintenance_capacity: int = 2  # K=1 or K=2
    cost_regime_id: str = "failure-light-no-waste"
    episode_horizon: int = 100
    training_scenario_bank_path: Optional[str] = None
    validation_scenario_bank_path: Optional[str] = None
    prediction_cache_path: str = "data/processed/fd001/v2/06_PREDICTIONS/"
    prediction_cache_manifest_path: str = "data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json"

    # Training
    max_steps: int = 100_000
    batch_size: int = 128
    warmup_transitions: int = 5_000
    update_frequency: int = 1  # Update every N steps
    validation_interval: int = 5_000  # Validate every N steps
    checkpoint_interval: int = 5_000  # Save checkpoint every N steps

    # Agent
    hidden_dim: int = 128
    num_hidden_layers: int = 2
    learning_rate: float = 1e-4
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    gradient_clip: float = 10.0
    target_update_interval: int = 1_000

    # Replay buffer
    replay_capacity: int = 100_000

    # Checkpointing
    output_dir: str = "results/milestone5"
    run_id: Optional[str] = None

    # Seeds
    training_seed: int = 6521
    validation_seed: int = 6521

    # Device
    device: Optional[str] = None  # Auto-resolve

    def __post_init__(self) -> None:
        errors = []

        # Validate split
        if self.split not in ALLOWED_SPLITS:
            errors.append(f"Invalid split '{self.split}', must be one of {ALLOWED_SPLITS}")

        # Validate validation_split
        if self.validation_split not in ALLOWED_SPLITS:
            errors.append(
                f"Invalid validation_split '{self.validation_split}', "
                f"must be one of {ALLOWED_SPLITS}"
            )

        # rl_test barrier - training path
        if self.split == "rl_test":
            errors.append(
                "CRITICAL BARRIER VIOLATION: split='rl_test' is forbidden for training. "
                "Training must use 'predictor_train'."
            )

        # rl_test barrier - validation path
        if self.validation_split == "rl_test":
            errors.append(
                "CRITICAL BARRIER VIOLATION: validation_split='rl_test' is forbidden. "
                "Validation must use 'rl_validation'."
            )

        # M5 formal trainer: enforce split = predictor_train
        if self.split != "predictor_train":
            errors.append(
                f"M5 formal trainer requires split='predictor_train', got '{self.split}'"
            )

        # M5 formal trainer: enforce validation_split = rl_validation
        if self.validation_split != "rl_validation":
            errors.append(
                f"M5 formal trainer requires validation_split='rl_validation', got '{self.validation_split}'"
            )

        # Validate scenario bank paths are provided
        if self.training_scenario_bank_path is None:
            errors.append("training_scenario_bank_path is required")
        if self.validation_scenario_bank_path is None:
            errors.append("validation_scenario_bank_path is required")

        # Validate K
        if self.maintenance_capacity not in (1, 2):
            errors.append(f"maintenance_capacity must be 1 or 2, got {self.maintenance_capacity}")

        # Validate numeric params
        if self.max_steps <= 0:
            errors.append(f"max_steps must be positive, got {self.max_steps}")
        if self.batch_size <= 0:
            errors.append(f"batch_size must be positive, got {self.batch_size}")
        if self.warmup_transitions < 0:
            errors.append(f"warmup_transitions must be non-negative, got {self.warmup_transitions}")
        if self.learning_rate <= 0:
            errors.append(f"learning_rate must be positive, got {self.learning_rate}")
        if not (0.0 < self.gamma < 1.0):
            errors.append(f"gamma must be in (0, 1), got {self.gamma}")

        if errors:
            raise ValueError("TrainerConfig validation failed:\n  - " + "\n  - ".join(errors))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @property
    def num_actions(self) -> int:
        """Get action count for this K."""
        if self.maintenance_capacity == 1:
            return len(ACTION_TABLE_N5_K1)  # 6
        else:
            return len(ACTION_TABLE_N5_K2)  # 16

    @property
    def action_table(self) -> Tuple[ActionSubset, ...]:
        """Get action table for this K."""
        if self.maintenance_capacity == 1:
            return ACTION_TABLE_N5_K1
        else:
            return ACTION_TABLE_N5_K2


@dataclass
class TrainingMetrics:
    """Accumulated training metrics."""

    # Per-step metrics (accumulated)
    step_metrics: List[Dict[str, Any]] = field(default_factory=list)

    # Episode metrics
    episode_returns: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    episode_costs: List[Dict[str, float]] = field(default_factory=list)

    # Validation metrics
    validation_results: List[Dict[str, Any]] = field(default_factory=list)

    # Checkpoint tracking
    checkpoints_saved: List[Dict[str, Any]] = field(default_factory=list)
    best_checkpoint_path: Optional[str] = None
    best_validation_mean_cost: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)


def set_all_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_validation_metrics(
    env: SelectiveMaintenanceEnv,
    agent: DDQNAgent,
    num_scenarios: int = 10,
    max_steps_per_scenario: int = 100,
) -> Dict[str, Any]:
    """
    Compute validation metrics with epsilon=0 (deterministic greedy).

    Args:
        env: Validation environment
        agent: DDQN agent
        num_scenarios: Number of scenarios to evaluate
        max_steps_per_scenario: Maximum steps per episode

    Returns:
        Dict with validation metrics
    """
    episode_returns = []
    total_costs = []
    failure_counts = []
    pm_counts = []

    for _ in range(num_scenarios):
        obs, info = env.reset()
        episode_return = 0.0
        total_cost = 0.0
        failure_count = 0
        pm_count = 0

        for step in range(max_steps_per_scenario):
            action = agent.evaluate_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_return += reward
            total_cost += info["total_cost"]
            failure_count += info["num_failures"]
            pm_count += info["num_preventive"]

            if truncated:
                break

        episode_returns.append(episode_return)
        total_costs.append(total_cost)
        failure_counts.append(failure_count)
        pm_counts.append(pm_count)

    # Compute statistics
    mean_cost = np.mean(total_costs)
    std_cost = np.std(total_costs)
    worst_10_pct = np.percentile(total_costs, 90)

    return {
        "mean_total_cost": float(mean_cost),
        "std_total_cost": float(std_cost),
        "worst_10_pct_cost": float(worst_10_pct),
        "mean_episode_return": float(np.mean(episode_returns)),
        "total_failures": int(sum(failure_counts)),
        "total_pm_actions": int(sum(pm_counts)),
        "num_episodes": num_scenarios,
    }


class DDQNTrainer:
    """
    DDQN Training orchestrator.

    Responsibilities:
    - Environment construction and management
    - Agent construction and training loop
    - Periodic validation
    - Checkpoint saving and selection
    - Artifact emission
    """

    def __init__(
        self,
        config: Optional[TrainerConfig] = None,
        resume_from: Optional[Path | str] = None,
    ):
        """
        Initialize DDQN trainer.

        Args:
            config: Trainer configuration
            resume_from: Optional checkpoint path for resume
        """
        if config is None:
            config = TrainerConfig()

        self.config = config

        # Set seeds
        set_all_seeds(config.training_seed)

        # Resolve device
        from src.agents.ddqn.q_network import resolve_device
        self.device = resolve_device(config.device)

        # Capture the run start timestamp at trainer construction.
        self.started_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        # Build run directory
        run_id = config.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(config.output_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metrics
        self.metrics = TrainingMetrics()

        # Create training environment
        train_config = get_default_config(
            split=config.split,
            cost_regime_id=config.cost_regime_id,
            maintenance_capacity=config.maintenance_capacity,
            scenario_bank_path=config.training_scenario_bank_path,
            prediction_cache_path=config.prediction_cache_path,
            seed=config.training_seed,
        )
        self.train_env = SelectiveMaintenanceEnv(config=train_config)

        # Create validation environment
        val_config = get_default_config(
            split=config.validation_split,
            cost_regime_id=config.cost_regime_id,
            maintenance_capacity=config.maintenance_capacity,
            scenario_bank_path=config.validation_scenario_bank_path,
            prediction_cache_path=config.prediction_cache_path,
            seed=config.validation_seed,
        )
        self.val_env = SelectiveMaintenanceEnv(config=val_config)

        # Create replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=config.replay_capacity,
            observation_dim=10,
            seed=config.training_seed,
        )

        # Create agent
        agent_config = DDQNAgentConfig(
            observation_dim=10,
            num_actions=config.num_actions,
            hidden_dim=config.hidden_dim,
            num_hidden_layers=config.num_hidden_layers,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            epsilon_start=config.epsilon_start,
            epsilon_end=config.epsilon_end,
            epsilon_decay_steps=config.epsilon_decay_steps,
            gradient_clip=config.gradient_clip,
            target_update_interval=config.target_update_interval,
            explicit_device=config.device,
        )

        self.agent = DDQNAgent(config=agent_config, seed=config.training_seed)

        # Training state (will be overwritten by resume if applicable)
        self.global_step = self.agent.global_step
        self.episode_count = 0
        self.current_episode_return = 0.0
        self.current_episode_cost: Dict[str, float] = {}

        # Checkpoint selection state (historical best across all validations)
        self._selection_state = CheckpointSelectionState()

        # Resume handling - overwrites training state if checkpoint loaded
        self.resumed_from: Optional[str] = None
        if resume_from is not None:
            self._resume_from_checkpoint(Path(resume_from))

    def _resume_from_checkpoint(self, checkpoint_path: Path) -> None:
        """Resume from checkpoint with strict production identity validation.

        Validation sequence (fail-closed, reject before any state restoration):
        1. Parse metadata (load_checkpoint with agent=None - PHASE A only)
        2. Validate all mandatory identities (schema v6, splits, scenario banks, architecture)
        3. Validate split provenance agreement with TrainerConfig
        4. Validate resolved_config_identity agreement
        5. Validate selection-state version
        6. Reject incompatibility before ANY state restoration
        7. Only then restore networks, optimizer, replay, epsilon, counters, and RNG
        """
        # Compute expected identity fields from active production configuration
        action_table = self.config.action_table
        expected_action_table_hash = compute_action_table_hash(action_table)
        expected_observation_schema_id = "m5_point_v1"  # Frozen observation schema
        expected_environment_contract_id = "m2_v1"  # Frozen environment contract

        # Compute expected network architecture ID from configuration.
        expected_network_architecture_id = compute_network_architecture_id(
            observation_dim=10,
            hidden_dim=self.config.hidden_dim,
            num_hidden_layers=self.config.num_hidden_layers,
            activation="relu",
            action_count=self.config.num_actions,
            architecture_revision="m5_point_v1",
        )

        # PHASE A: Load checkpoint WITHOUT agent to validate all identities first
        checkpoint_data, issues = load_checkpoint(
            checkpoint_path,
            agent=None,  # CRITICAL: Do NOT restore agent yet - validate first!
            expected_observation_dim=10,
            expected_action_count=self.config.num_actions,
            expected_k=self.config.maintenance_capacity,
            expected_cost_regime=self.config.cost_regime_id,
            expected_action_table_hash=expected_action_table_hash,
            expected_observation_schema_id=expected_observation_schema_id,
            expected_environment_contract_id=expected_environment_contract_id,
            expected_network_architecture_id=expected_network_architecture_id,
            # Same-filename tamper detection.
            expected_training_scenario_bank_path=str(self.config.training_scenario_bank_path) if self.config.training_scenario_bank_path else None,
            expected_validation_scenario_bank_path=str(self.config.validation_scenario_bank_path) if self.config.validation_scenario_bank_path else None,
            expected_prediction_cache_manifest_path=str(self.config.prediction_cache_manifest_path) if self.config.prediction_cache_manifest_path else None,
        )

        # If load_checkpoint returned incompatibilities from its validation stage, fail immediately
        if issues.get("incompatibilities"):
            raise ValueError(
                "CRITICAL: Checkpoint identity validation failed (production fail-closed):\n  - "
                + "\n  - ".join(issues["incompatibilities"])
            )

        # PHASE A continued: Strict split provenance agreement
        checkpoint_train_split = checkpoint_data.metadata.training_split
        checkpoint_val_split = checkpoint_data.metadata.validation_split
        expected_train_split = self.config.split
        expected_val_split = self.config.validation_split

        split_issues = []
        if checkpoint_train_split != expected_train_split:
            split_issues.append(
                f"Split provenance mismatch (training): checkpoint has '{checkpoint_train_split}', "
                f"expected '{expected_train_split}' (from active TrainerConfig)"
            )
        if checkpoint_val_split != expected_val_split:
            split_issues.append(
                f"Split provenance mismatch (validation): checkpoint has '{checkpoint_val_split}', "
                f"expected '{expected_val_split}' (from active TrainerConfig)"
            )
        # Forbidden split barrier at resume time
        if checkpoint_train_split == "rl_test" or checkpoint_val_split == "rl_test":
            split_issues.append(
                "FORBIDDEN: checkpoint contains 'rl_test' split provenance. "
                "rl_test is sealed and forbidden for training and evaluation."
            )
        # Schema v6: resolved_config_identity agreement — fail before mutation
        checkpoint_identity = checkpoint_data.metadata.resolved_config_identity
        # Use the same config dict format that was saved (includes num_actions)
        expected_config_dict = self.config.to_dict()
        expected_config_dict["num_actions"] = self.config.num_actions
        expected_identity = compute_resolved_config_identity(expected_config_dict)
        from src.training.ddqn_config_identity import validate_resolved_config_identity
        validate_resolved_config_identity(checkpoint_identity)
        validate_resolved_config_identity(expected_identity)
        if checkpoint_identity != expected_identity:
            split_issues.append(
                f"Resolved config identity mismatch: checkpoint has '{checkpoint_identity}', "
                f"expected '{expected_identity}' (computed from active TrainerConfig)."
            )

        # Schema v6: selection-state version must be exactly 1
        if checkpoint_data.metadata.selection_state is not None:
            sel_state_version = checkpoint_data.metadata.selection_state.get("selection_state_version")
            if sel_state_version != CHECKPOINT_SELECTION_STATE_VERSION:
                split_issues.append(
                    f"Selection state version mismatch: checkpoint has {sel_state_version}, "
                    f"expected {CHECKPOINT_SELECTION_STATE_VERSION}."
                )
        else:
            # Selection state is REQUIRED for schema v6
            split_issues.append(
                f"Checkpoint missing required selection_state (schema v6 requires it)."
            )

        if split_issues:
            raise ValueError(
                "CRITICAL: Checkpoint identity validation failed (production fail-closed):\n  - "
                + "\n  - ".join(split_issues)
            )

        # PHASE B: All validation passed - NOW restore mutable state
        # Restore replay buffer state if present (schema v1 requires action_count)
        if checkpoint_data.replay_buffer_state is not None:
            self.replay_buffer.load_state_dict(
                checkpoint_data.replay_buffer_state,
                expected_action_count=self.config.num_actions,
            )

        # Restore checkpoint selection state (historical best validation)
        if checkpoint_data.metadata.selection_state is not None:
            self._selection_state = CheckpointSelectionState.from_dict(
                checkpoint_data.metadata.selection_state
            )

        # NOW restore agent state (networks, optimizer, RNG, counters)
        # Restore network weights
        self.agent.online_network.load_state_dict(checkpoint_data.online_network_state_dict)
        self.agent.target_network.load_state_dict(checkpoint_data.target_network_state_dict)
        self.agent.optimizer.load_state_dict(checkpoint_data.optimizer_state_dict)

        # Restore agent state
        self.agent.global_step = checkpoint_data.global_step
        self.agent.gradient_update_count = checkpoint_data.gradient_update_count
        self.agent.epsilon_state = type(self.agent.epsilon_state).from_dict(checkpoint_data.epsilon_state)

        # Restore RNG states
        import random
        random.setstate(checkpoint_data.python_rng_state)
        np.random.set_state(checkpoint_data.numpy_rng_state)
        torch.set_rng_state(checkpoint_data.torch_cpu_rng_state)
        if checkpoint_data.torch_cuda_rng_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint_data.torch_cuda_rng_state)

        self.resumed_from = str(checkpoint_path)
        self.global_step = checkpoint_data.global_step

    def train_step(self) -> Optional[Dict[str, float]]:
        """
        Execute one training step.

        Returns:
            Training metrics dict or None (during warmup or no update)
        """
        # Current observation from last step (stored by train())
        if not hasattr(self, "_current_obs"):
            return None

        obs = self._current_obs

        # Select epsilon-greedy action
        action = self.agent.select_action(obs, training=True)

        # Execute action
        next_obs, reward, terminated, truncated, info = self.train_env.step(action)

        # Store transition
        self.replay_buffer.insert(
            observation=obs,
            action_id=action,
            reward=reward,
            next_observation=next_obs,
            terminated=terminated,
            truncated=truncated,
        )

        # Update episode tracking
        self.current_episode_return += reward
        self.current_episode_cost = {
            "total_cost": info["total_cost"],
            "preventive_cost": info["preventive_cost"],
            "failure_cost": info["failure_cost"],
            "wasted_life_cost": info["wasted_life_cost"],
        }

        # Handle episode end
        if truncated:
            self.episode_count += 1
            self.metrics.episode_returns.append(self.current_episode_return)
            self.metrics.episode_lengths.append(info["step_index"])
            self.metrics.episode_costs.append(self.current_episode_cost)
            self.current_episode_return = 0.0
            self.current_episode_cost = {}

            # Reset environment
            self._current_obs, _ = self.train_env.reset()
        else:
            self._current_obs = next_obs

        # Update global step
        self.global_step += 1
        self.agent.global_step = self.global_step

        # Advance epsilon on every environment step (independent of warmup/update_frequency)
        self.agent.epsilon_state.step()

        # Check if we should skip update (warmup or frequency)
        post_warmup = len(self.replay_buffer) >= self.config.warmup_transitions
        should_update = post_warmup and (
            self.global_step % self.config.update_frequency == 0
            and len(self.replay_buffer) >= self.config.batch_size
        )

        if not should_update:
            return None

        # Sample batch
        batch = self.replay_buffer.sample_batch(self.config.batch_size)

        # Update agent (gradient update only - epsilon already stepped above)
        update_metrics = self.agent.update(batch)

        # Target network sync
        self.agent.maybe_sync_target()

        return update_metrics

    def validate(self) -> Dict[str, Any]:
        """
        Perform validation with epsilon=0.

        Returns:
            Validation metrics dict
        """
        return compute_validation_metrics(
            self.val_env,
            self.agent,
            num_scenarios=min(10, len(self.val_env.scenario_bank.scenarios)),
            max_steps_per_scenario=self.config.episode_horizon,
        )

    def save_checkpoint(
        self,
        name: str = "checkpoint_latest.pt",
        validation_metrics: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save checkpoint.

        Args:
            name: Checkpoint filename
            validation_metrics: Optional validation metrics for selection

        Returns:
            Absolute path to saved checkpoint
        """
        checkpoint_path = self.run_dir / name

        config_dict = self.config.to_dict()
        config_dict["num_actions"] = self.config.num_actions

        # Do not mutate caller's validation_metrics dict (keep separate)
        # Schema v5: pass selection_state explicitly to save_checkpoint
        # Do not embed it inside validation_metrics
        checkpoint_data = save_checkpoint(
            agent=self.agent,
            config=config_dict,
            output_path=checkpoint_path,
            maintenance_capacity=self.config.maintenance_capacity,
            action_table=self.config.action_table,
            cost_regime_id=self.config.cost_regime_id,
            training_seed=self.config.training_seed,
            replay_buffer=self.replay_buffer,
            training_split=self.config.split,
            validation_split=self.config.validation_split,
            training_scenario_bank_path=self.config.training_scenario_bank_path,
            validation_scenario_bank_path=self.config.validation_scenario_bank_path,
            prediction_cache_manifest_path=self.config.prediction_cache_manifest_path,
            selection_state=self._selection_state,
            validation_metrics=validation_metrics,
        )

        self.metrics.checkpoints_saved.append({
            "path": str(checkpoint_path),
            "checkpoint_id": checkpoint_data.metadata.checkpoint_id,
            "global_step": checkpoint_data.metadata.global_step,
            "validation_mean_cost": checkpoint_data.metadata.validation_mean_cost,
        })

        return str(checkpoint_path)

    def train(self) -> TrainingMetrics:
        """
        Run complete training loop.

        Returns:
            TrainingMetrics with all accumulated metrics
        """
        # Initialize first observation
        self._current_obs, _ = self.train_env.reset()

        while self.global_step < self.config.max_steps:
            # Training step
            update_metrics = self.train_step()

            # Record step metrics
            if update_metrics is not None:
                self.metrics.step_metrics.append({
                    "global_step": self.global_step,
                    **update_metrics,
                })

            # Validation
            if (
                self.global_step > 0
                and self.global_step % self.config.validation_interval == 0
            ):
                val_metrics = self.validate()
                val_metrics["global_step"] = self.global_step
                self.metrics.validation_results.append(val_metrics)

                # Checkpoint selection using persistent selection state
                current_mean_cost = val_metrics["mean_total_cost"]
                should_update_best = False

                if not self._selection_state.validation_performed:
                    # First validation ever - always update
                    should_update_best = True
                elif current_mean_cost < self._selection_state.best_validation_mean_cost:
                    # Better cost - update best
                    should_update_best = True
                elif current_mean_cost == self._selection_state.best_validation_mean_cost:
                    # Equal cost - tie behavior (keep_first means don't update)
                    if self._selection_state.equal_metric_tie_behavior == "keep_first":
                        should_update_best = False
                    else:
                        should_update_best = True

                if should_update_best:
                    # Update selection state
                    self._selection_state = CheckpointSelectionState(
                        selection_state_version=CHECKPOINT_SELECTION_STATE_VERSION,
                        validation_performed=True,
                        best_validation_mean_cost=current_mean_cost,
                        best_checkpoint_global_step=self.global_step,
                        best_checkpoint_artifact_name="checkpoint_best.pt",
                        best_validation_failure_count=val_metrics.get("total_failures"),
                        best_validation_worst_10_pct_cost=val_metrics.get("worst_10_pct_cost"),
                        comparator_identity=self._selection_state.comparator_identity,
                        equal_metric_tie_behavior=self._selection_state.equal_metric_tie_behavior,
                    )

                    # Save best checkpoint with updated selection state
                    best_checkpoint_path = self.save_checkpoint(
                        "checkpoint_best.pt",
                        validation_metrics=val_metrics,
                    )
                    self.metrics.best_checkpoint_path = best_checkpoint_path
                    self.metrics.best_validation_mean_cost = current_mean_cost

            # Regular checkpoint (latest)
            if (
                self.global_step > 0
                and self.global_step % self.config.checkpoint_interval == 0
            ):
                self.save_checkpoint("checkpoint_latest.pt")

        # Final checkpoint
        self.save_checkpoint("checkpoint_latest.pt")

        # Write metrics
        self._write_artifacts()

        return self.metrics

    def _write_artifacts(self) -> None:
        """Write all training artifacts."""
        # Capture completed_at timestamp
        completed_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        # Training metrics JSONL
        metrics_path = self.run_dir / "training_metrics.jsonl"
        with metrics_path.open("w", encoding="utf-8") as f:
            for step_metric in self.metrics.step_metrics:
                f.write(json.dumps(step_metric, sort_keys=True) + "\n")

        # Episode metrics CSV
        if self.metrics.episode_costs:
            episode_path = self.run_dir / "episode_metrics.csv"
            with episode_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "episode", "return", "length", "total_cost",
                    "preventive_cost", "failure_cost", "wasted_life_cost",
                ])
                for i, costs in enumerate(self.metrics.episode_costs):
                    writer.writerow([
                        i + 1,
                        self.metrics.episode_returns[i],
                        self.metrics.episode_lengths[i],
                        costs["total_cost"],
                        costs["preventive_cost"],
                        costs["failure_cost"],
                        costs["wasted_life_cost"],
                    ])

        # Validation metrics JSON
        val_path = self.run_dir / "validation_metrics.json"
        with val_path.open("w", encoding="utf-8") as f:
            json.dump(self.metrics.validation_results, f, indent=2, sort_keys=True)

        # Resolved config artifact.
        # Write the exact same semantic config used for checkpoint/manifest identity
        resolved_config_dict = {**self.config.to_dict(), "num_actions": self.config.num_actions}
        resolved_config_identity = compute_resolved_config_identity(resolved_config_dict)
        # Strip run-specific fields for the artifact (same as identity computation)
        from src.training.ddqn_config_identity import _strip_run_specific_fields
        semantic_config = _strip_run_specific_fields(resolved_config_dict)
        resolved_config_path = self.run_dir / "resolved_config.json"
        with resolved_config_path.open("w", encoding="utf-8") as f:
            json.dump(semantic_config, f, indent=2, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

        # Run manifest with explicit timestamps and validation status.
        # Compute scenario bank identities using content-based hash
        training_bank_identity = None
        validation_bank_identity = None
        if self.config.training_scenario_bank_path:
            training_bank_identity = compute_scenario_bank_content_hash(Path(self.config.training_scenario_bank_path))
        if self.config.validation_scenario_bank_path:
            validation_bank_identity = compute_scenario_bank_content_hash(Path(self.config.validation_scenario_bank_path))

        # M5 provenance: surface the full schema-6 prediction-cache provenance at
        # the top level of the run manifest.  The checkpoint metadata already
        # records these (see src/agents/ddqn/checkpoint.py).  Surfacing them
        # here makes the manifest auditable WITHOUT requiring a torch.load
        # of the checkpoint, closing provenance at both layers.
        #
        # FAIL-CLOSED CONTRACT: the manifest MUST NOT be emitted (and status
        # MUST NOT be COMPLETE) if any of the eight required provenance fields
        # cannot be computed or is null.  No path-only identity, no
        # conditional acceptance, no soft "error-only provenance" object;
        # we raise before the manifest is written.
        from src.training.prediction_cache_identity import get_prediction_cache_identity
        pc_ident = get_prediction_cache_identity(self.config.prediction_cache_manifest_path)
        prediction_cache_provenance: Dict[str, Any] = {
            "prediction_cache_manifest_path": pc_ident["prediction_cache_manifest_path"],
            "prediction_cache_manifest_sha256": pc_ident["prediction_cache_manifest_sha256"],
            "prediction_cache_declared_cache_hash": pc_ident["prediction_cache_declared_cache_hash"],
            "prediction_cache_predictor_checkpoint_hash": pc_ident["prediction_cache_predictor_checkpoint_hash"],
            "prediction_cache_feature_schema_hash": pc_ident["prediction_cache_feature_schema_hash"],
            "prediction_cache_normalizer_hash": pc_ident["prediction_cache_normalizer_hash"],
            "prediction_cache_split": self.config.validation_split,
            "prediction_cache_schema_version": pc_ident["prediction_cache_schema_version"],
        }

        # Defense-in-depth: every required provenance field must be present and
        # non-null before a COMPLETE manifest is permitted.  This runs in
        # addition to the fail-closed construction above so that a future
        # regression in the identity helper cannot silently emit a null field.
        _required_pc_fields = (
            "prediction_cache_manifest_path",
            "prediction_cache_manifest_sha256",
            "prediction_cache_declared_cache_hash",
            "prediction_cache_predictor_checkpoint_hash",
            "prediction_cache_feature_schema_hash",
            "prediction_cache_normalizer_hash",
            "prediction_cache_split",
            "prediction_cache_schema_version",
        )
        _missing = [f for f in _required_pc_fields if f not in prediction_cache_provenance
                    or prediction_cache_provenance[f] is None]
        if _missing:
            raise RuntimeError(
                "CRITICAL: production provenance not fail-closed -- required "
                "prediction-cache provenance field(s) absent/null: "
                + ", ".join(_missing)
                + ". Refusing to emit run manifest."
            )

        # Determine status - COMPLETE if we reached max_steps, otherwise INCOMPLETE/RUNNING
        status = "COMPLETE" if self.global_step >= self.config.max_steps else "INCOMPLETE"

        # validation_performed is True if we have any validation results
        # Use explicit boolean, not truthiness of best_validation_mean_cost
        validation_performed = len(self.metrics.validation_results) > 0

        # Replay buffer state version.
        replay_state_version = 1

        manifest = {
            "status": status,
            "run_id": self.run_dir.name,
            "started_at": self.started_at,
            "completed_at": completed_at if status == "COMPLETE" else None,
            "maintenance_capacity": self.config.maintenance_capacity,
            "cost_regime_id": self.config.cost_regime_id,
            "training_seed": self.config.training_seed,
            "final_global_step": self.global_step,
            "max_steps": self.config.max_steps,
            "training_split": self.config.split,
            "validation_split": self.config.validation_split,
            "training_scenario_bank_identity": training_bank_identity,
            "validation_scenario_bank_identity": validation_bank_identity,
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "replay_state_version": replay_state_version,
            "network_architecture_id": compute_network_architecture_id(
                observation_dim=10,
                hidden_dim=self.config.hidden_dim,
                num_hidden_layers=self.config.num_hidden_layers,
                activation="relu",
                action_count=self.config.num_actions,
                architecture_revision="m5_point_v1",
            ),
            "checkpoint_latest": str(self.run_dir / "checkpoint_latest.pt"),
            "checkpoint_best": self.metrics.best_checkpoint_path,
            "validation_performed": validation_performed,
            "metric_artifacts": {
                "training_metrics": str(metrics_path),
                "episode_metrics": str(episode_path) if self.metrics.episode_costs else None,
                "validation_metrics": str(val_path),
            },
            "resolved_config_identity": compute_resolved_config_identity({**self.config.to_dict(), "num_actions": self.config.num_actions}),
            "prediction_cache_provenance": prediction_cache_provenance,
            "git_commit": get_git_commit(),
        }
        # Schema v6: enforce identity is a valid 64-char lowercase hex SHA and not placeholder
        manifest_id = manifest["resolved_config_identity"]
        validate_resolved_config_identity(manifest_id)
        manifest_path = self.run_dir / "run_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)


def create_trainer(
    config: Optional[TrainerConfig] = None,
    resume_from: Optional[Path | str] = None,
) -> DDQNTrainer:
    """
    Factory function to create DDQN trainer.

    Args:
        config: Optional configuration
        resume_from: Optional checkpoint path for resume

    Returns:
        Initialized DDQNTrainer
    """
    return DDQNTrainer(config=config, resume_from=resume_from)