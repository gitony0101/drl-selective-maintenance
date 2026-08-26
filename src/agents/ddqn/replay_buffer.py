"""
Replay Buffer for Milestone 5 Point-Estimate Double DQN.

Implements fixed-capacity ring buffer with:
- Transition storage: (obs, action_id, reward, next_obs, terminated, truncated)
- Deterministic seeded sampling
- CPU-backed storage
- Device transfer after sampling
- Wraparound semantics
- Finite-value validation
- Legal-action validation
- Strict state schema v1 validation
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, NamedTuple, Dict, Any
from dataclasses import dataclass

# Replay buffer state schema version
REPLAY_BUFFER_STATE_VERSION = 1


class Transition(NamedTuple):
    """
    Single transition tuple.

    Fields:
        observation: Current observation (10,) float32
        action_id: Action taken (int64)
        reward: Reward received (float32)
        next_observation: Next observation (10,) float32
        terminated: Episode terminated flag (bool)
        truncated: Episode truncated flag (bool)
    """

    observation: np.ndarray
    action_id: int
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class ReplayBufferConfig:
    """Immutable replay buffer configuration."""

    capacity: int = 100_000
    observation_dim: int = 10
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        errors = []
        if self.capacity <= 0:
            errors.append(f"capacity must be positive, got {self.capacity}")
        if self.observation_dim <= 0:
            errors.append(f"observation_dim must be positive, got {self.observation_dim}")
        if errors:
            raise ValueError("ReplayBufferConfig validation failed:\n  - " + "\n  - ".join(errors))


class ReplayBuffer:
    """
    Fixed-capacity ring replay buffer.

    Storage layout (CPU-backed numpy arrays):
        - observations: (capacity, observation_dim) float32
        - actions: (capacity,) int64
        - rewards: (capacity,) float32
        - next_observations: (capacity, observation_dim) float32
        - terminated: (capacity,) bool
        - truncated: (capacity,) bool

    Ring buffer semantics:
        - write_index advances with each insertion
        - When write_index >= capacity, wrap around (oldest overwritten)
        - current_size = min(write_index, capacity)

    Sampling:
        - Deterministic with seeded numpy RNG
        - Uniform without replacement
        - Returns batch of transitions
        - Device transfer after sampling (to torch)
    """

    def __init__(
        self,
        config: Optional[ReplayBufferConfig] = None,
        capacity: int = 100_000,
        observation_dim: int = 10,
        seed: Optional[int] = None,
    ):
        """
        Initialize replay buffer.

        Args:
            config: Optional configuration (takes precedence)
            capacity: Maximum number of transitions
            observation_dim: Observation dimension (default 10)
            seed: Optional seed for deterministic sampling
        """
        if config is not None:
            self.config = config
        else:
            self.config = ReplayBufferConfig(
                capacity=capacity,
                observation_dim=observation_dim,
                seed=seed,
            )

        self.capacity = self.config.capacity
        self.observation_dim = self.config.observation_dim

        # Allocate storage (CPU-backed)
        self.observations = np.zeros(
            (self.capacity, self.observation_dim), dtype=np.float32
        )
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_observations = np.zeros(
            (self.capacity, self.observation_dim), dtype=np.float32
        )
        self.terminated = np.zeros(self.capacity, dtype=bool)
        self.truncated = np.zeros(self.capacity, dtype=bool)

        # Ring buffer state
        self.write_index = 0
        self.current_size = 0

        # RNG for sampling
        if self.config.seed is not None:
            self.rng = np.random.default_rng(self.config.seed)
        else:
            self.rng = np.random.default_rng()

    def __len__(self) -> int:
        """Return current number of stored transitions."""
        return self.current_size

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(capacity={self.capacity}, "
            f"size={self.current_size}, "
            f"write_index={self.write_index})"
        )

    def insert(
        self,
        observation: np.ndarray,
        action_id: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
        Insert a single transition.

        Args:
            observation: Current observation (observation_dim,) float32
            action_id: Action taken (legal action ID)
            reward: Reward received
            next_observation: Next observation (observation_dim,) float32
            terminated: Episode terminated flag
            truncated: Episode truncated flag

        Raises:
            ValueError: If transition contains invalid values
        """
        # Validate observation
        observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        if observation.shape[0] != self.observation_dim:
            raise ValueError(
                f"observation shape mismatch: expected ({self.observation_dim},), "
                f"got {observation.shape}"
            )
        if not np.isfinite(observation).all():
            raise ValueError(
                f"observation contains non-finite values: "
                f"nan={np.isnan(observation).sum()}, inf={np.isinf(observation).sum()}"
            )

        # Validate next_observation
        next_observation = np.asarray(next_observation, dtype=np.float32).reshape(-1)
        if next_observation.shape[0] != self.observation_dim:
            raise ValueError(
                f"next_observation shape mismatch: expected ({self.observation_dim},), "
                f"got {next_observation.shape}"
            )
        if not np.isfinite(next_observation).all():
            raise ValueError(
                f"next_observation contains non-finite values: "
                f"nan={np.isnan(next_observation).sum()}, inf={np.isinf(next_observation).sum()}"
            )

        # Validate action_id
        if not isinstance(action_id, (int, np.integer)):
            raise ValueError(f"action_id must be integral, got {type(action_id).__name__}")
        if isinstance(action_id, bool):
            raise ValueError("action_id cannot be boolean")
        if action_id < 0:
            raise ValueError(f"action_id must be non-negative, got {action_id}")

        # Validate reward
        if not np.isfinite(reward):
            raise ValueError(f"reward must be finite, got {reward}")

        # Write to ring buffer
        write_idx = self.write_index

        self.observations[write_idx] = observation
        self.actions[write_idx] = action_id
        self.rewards[write_idx] = float(reward)
        self.next_observations[write_idx] = next_observation
        self.terminated[write_idx] = terminated
        self.truncated[write_idx] = truncated

        # Advance ring buffer
        self.write_index = (self.write_index + 1) % self.capacity
        if self.current_size < self.capacity:
            self.current_size += 1

    def sample(
        self,
        batch_size: int,
        device: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample
            device: Optional device for tensor conversion ("cpu", "mps", "cuda")

        Returns:
            Tuple of numpy arrays:
                - observations: (batch_size, observation_dim) float32
                - actions: (batch_size,) int64
                - rewards: (batch_size,) float32
                - next_observations: (batch_size, observation_dim) float32
                - terminated: (batch_size,) bool
                - truncated: (batch_size,) bool

        Raises:
            ValueError: If buffer has insufficient transitions
        """
        if self.current_size < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer with "
                f"current_size={self.current_size}"
            )

        # Sample indices without replacement
        indices = self.rng.choice(self.current_size, size=batch_size, replace=False)

        # Gather batch
        obs_batch = self.observations[indices]
        actions_batch = self.actions[indices]
        rewards_batch = self.rewards[indices]
        next_obs_batch = self.next_observations[indices]
        terminated_batch = self.terminated[indices]
        truncated_batch = self.truncated[indices]

        return (
            obs_batch,
            actions_batch,
            rewards_batch,
            next_obs_batch,
            terminated_batch,
            truncated_batch,
        )

    def sample_batch(
        self,
        batch_size: int,
        device: Optional[str] = None,
    ) -> dict:
        """
        Sample a batch of transitions and return as dict.

        Args:
            batch_size: Number of transitions to sample
            device: Optional device for tensor conversion

        Returns:
            Dict with keys:
                - observation: (batch_size, observation_dim) float32
                - action: (batch_size,) int64
                - reward: (batch_size,) float32
                - next_observation: (batch_size, observation_dim) float32
                - terminated: (batch_size,) bool
                - truncated: (batch_size,) bool
        """
        (obs, actions, rewards, next_obs, terminated, truncated) = self.sample(
            batch_size, device
        )
        return {
            "observation": obs,
            "action": actions,
            "reward": rewards,
            "next_observation": next_obs,
            "terminated": terminated,
            "truncated": truncated,
        }

    def get_recent_transitions(self, n: int) -> dict:
        """
        Get the n most recent transitions (for testing/inspection).

        Args:
            n: Number of transitions to retrieve

        Returns:
            Dict with same structure as sample_batch()
        """
        if self.current_size == 0:
            raise ValueError("Buffer is empty")

        n = min(n, self.current_size)

        if self.write_index >= n:
            # Recent transitions are contiguous before write_index
            start_idx = self.write_index - n
            indices = np.arange(start_idx, self.write_index)
        else:
            # Wrap around case
            indices = np.concatenate([
                np.arange(self.capacity - (n - self.write_index), self.capacity),
                np.arange(0, self.write_index),
            ])

        return {
            "observation": self.observations[indices],
            "action": self.actions[indices],
            "reward": self.rewards[indices],
            "next_observation": self.next_observations[indices],
            "terminated": self.terminated[indices],
            "truncated": self.truncated[indices],
        }

    def clear(self) -> None:
        """Clear all stored transitions."""
        self.write_index = 0
        self.current_size = 0

    def state_dict(self, action_count: Optional[int] = None) -> Dict[str, Any]:
        """
        Get replay buffer state for checkpointing.

        Schema v1 includes:
        - replay_state_version: Schema version (1)
        - observation_dim: Observation dimension
        - action_count: Number of legal actions (REQUIRED for validation)
        - capacity: Buffer capacity
        - current_size: Current fill level
        - write_index: Next write position
        - observations: (capacity, observation_dim) float32
        - actions: (capacity,) int64
        - rewards: (capacity,) float32
        - next_observations: (capacity, observation_dim) float32
        - terminated: (capacity,) bool
        - truncated: (capacity,) bool
        - rng_state: Numpy RNG state dict

        Args:
            action_count: Number of legal actions (REQUIRED for schema v1)

        Returns:
            Dict containing replay buffer state

        Raises:
            ValueError: If action_count is not provided
        """
        if action_count is None:
            raise ValueError(
                "Schema v1 requires action_count to be provided. "
                "This validates that stored actions are legal for the current action space."
            )

        return {
            "replay_state_version": REPLAY_BUFFER_STATE_VERSION,
            "observation_dim": self.observation_dim,
            "action_count": action_count,
            "observations": self.observations.copy(),
            "actions": self.actions.copy(),
            "rewards": self.rewards.copy(),
            "next_observations": self.next_observations.copy(),
            "terminated": self.terminated.copy(),
            "truncated": self.truncated.copy(),
            "write_index": self.write_index,
            "current_size": self.current_size,
            "capacity": self.capacity,
            "observation_dim": self.observation_dim,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: Dict[str, Any], expected_action_count: Optional[int] = None) -> None:
        """
        Restore replay buffer from checkpoint state with strict schema v1 validation.

        Validation sequence (fail-closed, reject before any data copy):
        1. Check required fields
        2. Validate replay_state_version
        3. Validate capacity and observation_dim
        4. Validate dtypes BEFORE conversion (reject float64, float actions, etc.)
        5. Validate shapes
        6. Validate finite values
        7. Validate action bounds (0 <= action < action_count)
        8. Validate current_size and write_index
        9. Validate RNG state
        10. Only then copy data into internal arrays

        Args:
            state: State dict from state_dict()
            expected_action_count: Expected number of legal actions (REQUIRED for schema v1)

        Raises:
            ValueError: If state is invalid or incompatible
        """
        # Schema v1 requires action_count validation
        if expected_action_count is None:
            raise ValueError(
                "Schema v1 requires expected_action_count. "
                "This validates that stored actions are legal for the current action space."
            )

        # Validate required fields
        required_fields = [
            "replay_state_version",
            "observation_dim",
            "action_count",
            "observations", "actions", "rewards",
            "next_observations", "terminated", "truncated",
            "write_index", "current_size", "capacity",
            "rng_state",
        ]
        missing = [f for f in required_fields if f not in state]
        if missing:
            raise ValueError(
                f"Replay buffer state missing required fields: {missing}. "
                f"Schema v1 requires all fields including replay_state_version and action_count."
            )

        # Validate schema version
        if state["replay_state_version"] != REPLAY_BUFFER_STATE_VERSION:
            raise ValueError(
                f"Replay buffer schema version mismatch: state has version {state['replay_state_version']}, "
                f"current version is {REPLAY_BUFFER_STATE_VERSION}. "
                f"Please regenerate the replay buffer state."
            )

        # Validate action_count matches expected
        if state["action_count"] != expected_action_count:
            raise ValueError(
                f"Replay buffer action_count mismatch: state has {state['action_count']}, "
                f"expected {expected_action_count}. "
                f"Actions stored with a different action space are incompatible."
            )

        # Validate capacity
        if state["capacity"] != self.capacity:
            raise ValueError(
                f"Replay buffer capacity mismatch: state has {state['capacity']}, "
                f"current buffer has {self.capacity}"
            )

        # Validate observation dimension
        if state["observation_dim"] != self.observation_dim:
            raise ValueError(
                f"Replay buffer observation_dim mismatch: state has {state['observation_dim']}, "
                f"current buffer has {self.observation_dim}"
            )

        # STRICT DTYPE VALIDATION BEFORE CONVERSION
        # Check source dtypes BEFORE np.asarray conversion

        # Observations: must be float32 numpy ndarray (reject object arrays, Python lists)
        obs_raw = state["observations"]
        if not isinstance(obs_raw, np.ndarray):
            raise ValueError(
                f"Observations must be numpy ndarray, got {type(obs_raw).__name__}. "
                f"Python lists are not supported in schema v1."
            )
        if obs_raw.dtype == np.object_:
            raise ValueError(
                f"Observations must be float32, got object dtype. "
                f"Object arrays are not supported in schema v1."
            )
        if obs_raw.dtype != np.float32:
            raise ValueError(
                f"Observations must be float32, got {obs_raw.dtype}. "
                f"Float64 observations are not supported in schema v1."
            )
        obs = obs_raw.copy()

        # Next observations: must be float32 numpy ndarray
        next_obs_raw = state["next_observations"]
        if not isinstance(next_obs_raw, np.ndarray):
            raise ValueError(
                f"Next observations must be numpy ndarray, got {type(next_obs_raw).__name__}. "
                f"Python lists are not supported in schema v1."
            )
        if next_obs_raw.dtype == np.object_:
            raise ValueError(
                f"Next observations must be float32, got object dtype. "
                f"Object arrays are not supported in schema v1."
            )
        if next_obs_raw.dtype != np.float32:
            raise ValueError(
                f"Next observations must be float32, got {next_obs_raw.dtype}. "
                f"Float64 next_observations are not supported in schema v1."
            )
        next_obs = next_obs_raw.copy()

        # Rewards: must be float32 numpy ndarray
        rewards_raw = state["rewards"]
        if not isinstance(rewards_raw, np.ndarray):
            raise ValueError(
                f"Rewards must be numpy ndarray, got {type(rewards_raw).__name__}. "
                f"Python lists are not supported in schema v1."
            )
        if rewards_raw.dtype == np.object_:
            raise ValueError(
                f"Rewards must be float32, got object dtype. "
                f"Object arrays are not supported in schema v1."
            )
        if rewards_raw.dtype != np.float32:
            raise ValueError(
                f"Rewards must be float32, got {rewards_raw.dtype}. "
                f"Float64 rewards are not supported in schema v1."
            )
        rewards = rewards_raw.copy()

        # Actions: must be integer dtype numpy ndarray (int64)
        actions_raw = state["actions"]
        if not isinstance(actions_raw, np.ndarray):
            raise ValueError(
                f"Actions must be numpy ndarray, got {type(actions_raw).__name__}. "
                f"Python lists are not supported in schema v1."
            )
        if actions_raw.dtype == np.object_:
            raise ValueError(
                f"Actions must be integer dtype, got object dtype. "
                f"Object arrays are not supported in schema v1."
            )
        if not np.issubdtype(actions_raw.dtype, np.integer):
            raise ValueError(
                f"Actions must be integer dtype, got {actions_raw.dtype}. "
                f"Float actions are not supported in schema v1."
            )
        actions = actions_raw.copy()

        # Terminated: must be bool numpy ndarray
        terminated_raw = state["terminated"]
        if not isinstance(terminated_raw, np.ndarray):
            raise ValueError(
                f"terminated must be numpy ndarray, got {type(terminated_raw).__name__}. "
                f"Python lists are not supported in schema v1."
            )
        if terminated_raw.dtype == np.object_:
            raise ValueError(
                f"terminated must be bool, got object dtype. "
                f"Object arrays are not supported in schema v1."
            )
        if terminated_raw.dtype != bool:
            raise ValueError(
                f"terminated must be bool, got {terminated_raw.dtype}. "
                f"Integer terminated flags are not supported in schema v1."
            )
        terminated = terminated_raw.copy()

        # Truncated: must be bool numpy ndarray
        truncated_raw = state["truncated"]
        if not isinstance(truncated_raw, np.ndarray):
            raise ValueError(
                f"truncated must be numpy ndarray, got {type(truncated_raw).__name__}. "
                f"Python lists are not supported in schema v1."
            )
        if truncated_raw.dtype == np.object_:
            raise ValueError(
                f"truncated must be bool, got object dtype. "
                f"Object arrays are not supported in schema v1."
            )
        if truncated_raw.dtype != bool:
            raise ValueError(
                f"truncated must be bool, got {truncated_raw.dtype}. "
                f"Integer truncated flags are not supported in schema v1."
            )
        truncated = truncated_raw.copy()

        # Validate shapes
        if obs.shape != (self.capacity, self.observation_dim):
            raise ValueError(
                f"Observations shape mismatch: expected ({self.capacity}, {self.observation_dim}), "
                f"got {obs.shape}"
            )
        if next_obs.shape != (self.capacity, self.observation_dim):
            raise ValueError(
                f"Next observations shape mismatch: expected ({self.capacity}, {self.observation_dim}), "
                f"got {next_obs.shape}"
            )
        if rewards.shape != (self.capacity,):
            raise ValueError(
                f"Rewards shape mismatch: expected ({self.capacity},), got {rewards.shape}"
            )
        if actions.shape != (self.capacity,):
            raise ValueError(
                f"Actions shape mismatch: expected ({self.capacity},), got {actions.shape}"
            )
        if terminated.shape != (self.capacity,):
            raise ValueError(
                f"terminated shape mismatch: expected ({self.capacity},), got {terminated.shape}"
            )
        if truncated.shape != (self.capacity,):
            raise ValueError(
                f"truncated shape mismatch: expected ({self.capacity},), got {truncated.shape}"
            )

        # Validate finite values
        if not np.isfinite(obs).all():
            raise ValueError("Observations contain non-finite values")
        if not np.isfinite(next_obs).all():
            raise ValueError("Next observations contain non-finite values")
        if not np.isfinite(rewards).all():
            raise ValueError("Rewards contain non-finite values")

        # Validate action bounds (0 <= action < action_count)
        if actions.min() < 0:
            raise ValueError(
                f"Actions contain negative values: min={actions.min()}. "
                f"Action IDs must be non-negative."
            )
        if actions.max() >= expected_action_count:
            raise ValueError(
                f"Actions contain illegal values: max={actions.max()}, "
                f"but action_count={expected_action_count}. "
                f"Action IDs must be in [0, {expected_action_count - 1}]."
            )

        # Validate size and write position
        if not (0 <= state["current_size"] <= self.capacity):
            raise ValueError(
                f"Invalid current_size: {state['current_size']}. "
                f"Must be in [0, {self.capacity}]."
            )
        if not (0 <= state["write_index"] < self.capacity):
            raise ValueError(
                f"Invalid write_index: {state['write_index']}. "
                f"Must be in [0, {self.capacity})."
            )

        # Validate RNG state format
        rng_state = state["rng_state"]
        if not isinstance(rng_state, dict):
            raise ValueError(
                f"RNG state must be a dict, got {type(rng_state).__name__}"
            )
        if "bit_generator" not in rng_state:
            raise ValueError(
                "RNG state missing 'bit_generator' field. "
                "Malformed or missing RNG state."
            )

        # ALL VALIDATIONS PASSED - now restore state
        self.observations[:] = obs
        self.actions[:] = actions
        self.rewards[:] = rewards
        self.next_observations[:] = next_obs
        self.terminated[:] = terminated
        self.truncated[:] = truncated
        self.write_index = state["write_index"]
        self.current_size = state["current_size"]

        # Restore RNG state
        self.rng.bit_generator.state = state["rng_state"]