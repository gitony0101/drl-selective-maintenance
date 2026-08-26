"""
DDQN Agent for Milestone 5 Point-Estimate Double DQN.

Implements:
- Epsilon-greedy action selection with linear decay
- Double DQN target computation
- Gradient updates with Huber loss
- Target network synchronization
- Checkpoint serialization
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field

from .q_network import QNetwork, resolve_device
from .replay_buffer import ReplayBuffer


@dataclass(frozen=True)
class DDQNAgentConfig:
    """Immutable DDQN agent configuration."""

    # Observation/action
    observation_dim: int = 10
    num_actions: int = 16  # 6 for K=1, 16 for K=2

    # Epsilon-greedy
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000

    # Training
    gamma: float = 0.95
    learning_rate: float = 1e-4
    batch_size: int = 128
    gradient_clip: float = 10.0
    target_update_interval: int = 1_000
    use_huber_loss: bool = True
    huber_delta: float = 1.0

    # Device
    explicit_device: Optional[str] = None

    # Architecture
    hidden_dim: int = 128
    num_hidden_layers: int = 2

    def __post_init__(self) -> None:
        errors = []
        if self.observation_dim <= 0:
            errors.append(f"observation_dim must be positive, got {self.observation_dim}")
        if self.num_actions <= 0:
            errors.append(f"num_actions must be positive, got {self.num_actions}")
        if not (0.0 <= self.epsilon_start <= 1.0):
            errors.append(f"epsilon_start must be in [0, 1], got {self.epsilon_start}")
        if not (0.0 <= self.epsilon_end <= 1.0):
            errors.append(f"epsilon_end must be in [0, 1], got {self.epsilon_end}")
        if self.epsilon_end > self.epsilon_start:
            errors.append(f"epsilon_end must be <= epsilon_start")
        if self.epsilon_decay_steps <= 0:
            errors.append(f"epsilon_decay_steps must be positive, got {self.epsilon_decay_steps}")
        if not (0.0 < self.gamma < 1.0):
            errors.append(f"gamma must be in (0, 1), got {self.gamma}")
        if self.learning_rate <= 0:
            errors.append(f"learning_rate must be positive, got {self.learning_rate}")
        if self.batch_size <= 0:
            errors.append(f"batch_size must be positive, got {self.batch_size}")
        if self.gradient_clip <= 0:
            errors.append(f"gradient_clip must be positive, got {self.gradient_clip}")
        if self.target_update_interval <= 0:
            errors.append(f"target_update_interval must be positive, got {self.target_update_interval}")
        if errors:
            raise ValueError("DDQNAgentConfig validation failed:\n  - " + "\n  - ".join(errors))


@dataclass
class EpsilonState:
    """Mutable epsilon-greedy state."""

    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    global_step: int = 0

    @property
    def epsilon(self) -> float:
        """
        Compute current epsilon with linear decay.

        Returns:
            Current epsilon value in [epsilon_end, epsilon_start]
        """
        if self.global_step >= self.epsilon_decay_steps:
            return self.epsilon_end

        progress = self.global_step / self.epsilon_decay_steps
        return self.epsilon_start - progress * (self.epsilon_start - self.epsilon_end)

    def step(self) -> None:
        """Advance global step by 1."""
        self.global_step += 1

    def reset(self, global_step: int = 0) -> None:
        """Reset epsilon state."""
        self.global_step = global_step

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "global_step": self.global_step,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpsilonState":
        """Deserialize from dictionary."""
        return cls(
            epsilon_start=data["epsilon_start"],
            epsilon_end=data["epsilon_end"],
            epsilon_decay_steps=data["epsilon_decay_steps"],
            global_step=data["global_step"],
        )


class DDQNAgent:
    """
    Double DQN Agent.

    Components:
    - Online network: Used for action selection and gradient updates
    - Target network: Used for TD target computation (updated periodically)
    - Replay buffer: Stores transitions for off-policy learning

    Double DQN Target:
    1. Select action: a* = argmax_a Q_online(next_obs, a)
    2. Evaluate with target: Q_target(next_obs, a*)
    3. TD target: y = r + gamma * (1 - terminated) * Q_target(next_obs, a*)

    Key properties:
    - Target network excluded from optimizer
    - Target tensor detached from computation graph
    - Huber loss for robustness
    - Gradient clipping for stability
    """

    def __init__(
        self,
        config: Optional[DDQNAgentConfig] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize DDQN agent.

        Args:
            config: Agent configuration
            seed: Optional seed for action selection RNG
        """
        if config is None:
            config = DDQNAgentConfig()

        self.config = config
        self.device = resolve_device(config.explicit_device)

        # Create networks
        self.online_network = QNetwork(
            input_dim=config.observation_dim,
            hidden_dim=config.hidden_dim,
            num_hidden_layers=config.num_hidden_layers,
            output_dim=config.num_actions,
            explicit_device=config.explicit_device,
        )

        self.target_network = QNetwork(
            input_dim=config.observation_dim,
            hidden_dim=config.hidden_dim,
            num_hidden_layers=config.num_hidden_layers,
            output_dim=config.num_actions,
            explicit_device=config.explicit_device,
        )

        # Initialize target network with online network weights
        self.target_network.load_state_dict(self.online_network.state_dict())

        # Target network requires no gradients
        for param in self.target_network.parameters():
            param.requires_grad = False

        # Optimizer (online network only)
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=config.learning_rate,
        )

        # Epsilon state
        self.epsilon_state = EpsilonState(
            epsilon_start=config.epsilon_start,
            epsilon_end=config.epsilon_end,
            epsilon_decay_steps=config.epsilon_decay_steps,
        )

        # Action selection RNG (separate from environment)
        if seed is not None:
            self.action_rng = np.random.default_rng(seed)
        else:
            self.action_rng = np.random.default_rng()

        # Training counters
        self.global_step = 0
        self.gradient_update_count = 0

    def select_action(
        self,
        observation: np.ndarray,
        training: bool = True,
        legal_actions_mask: Optional[np.ndarray] = None,
    ) -> int:
        """
        Select action using epsilon-greedy (training) or greedy (evaluation).

        Args:
            observation: Observation array (observation_dim,) float32
            training: If True, use epsilon-greedy; if False, use greedy
            legal_actions_mask: Optional boolean mask of legal actions (num_actions,)

        Returns:
            Selected action ID

        Raises:
            ValueError: If observation shape invalid or no legal actions
        """
        obs_tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).reshape(-1)

        if obs_tensor.shape[0] != self.config.observation_dim:
            raise ValueError(
                f"Observation shape mismatch: expected ({self.config.observation_dim},), "
                f"got {obs_tensor.shape}"
            )

        if training and self.epsilon_state.epsilon > 0:
            # Epsilon-greedy exploration
            if self.action_rng.random() < self.epsilon_state.epsilon:
                # Random action
                if legal_actions_mask is not None:
                    legal_indices = np.where(legal_actions_mask)[0]
                    if len(legal_indices) == 0:
                        raise ValueError("No legal actions available")
                    return int(self.action_rng.choice(legal_indices))
                else:
                    return int(self.action_rng.integers(0, self.config.num_actions))

        # Greedy action selection
        if legal_actions_mask is not None:
            mask_tensor = torch.as_tensor(
                legal_actions_mask, dtype=torch.bool, device=self.device
            )
        else:
            mask_tensor = None

        return self.online_network.select_action(obs_tensor, mask_tensor)

    def compute_td_target(
        self,
        batch: Dict[str, np.ndarray],
    ) -> torch.Tensor:
        """
        Compute Double DQN TD target.

        Double DQN target:
        1. a* = argmax_a Q_online(next_obs, a)
        2. target_value = Q_target(next_obs, a*)
        3. y = reward + gamma * (1 - terminated) * target_value

        Args:
            batch: Batch dict from replay buffer with keys:
                - next_observation: (batch_size, observation_dim)
                - reward: (batch_size,)
                - terminated: (batch_size,)

        Returns:
            TD target tensor (batch_size,), detached
        """
        next_obs = torch.as_tensor(
            batch["next_observation"], dtype=torch.float32, device=self.device
        )
        rewards = torch.as_tensor(
            batch["reward"], dtype=torch.float32, device=self.device
        )
        terminated = torch.as_tensor(
            batch["terminated"], dtype=torch.float32, device=self.device
        )

        # Step 1: Select action with online network
        with torch.no_grad():
            next_q_values = self.online_network(next_obs)
            next_actions = next_q_values.argmax(dim=1)

            # Step 2: Evaluate with target network
            target_q_values_all = self.target_network(next_obs)
            target_q_selected = target_q_values_all.gather(
                dim=1, index=next_actions.unsqueeze(1)
            ).squeeze(1)

            # Step 3: Compute TD target
            bootstrap_mask = 1.0 - terminated
            td_target = rewards + self.config.gamma * bootstrap_mask * target_q_selected

        # Detach target (no gradient through target network)
        return td_target.detach()

    def update(
        self,
        batch: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """
        Perform one gradient update.

        Args:
            batch: Batch dict from replay buffer

        Returns:
            Dict with training metrics:
                - td_loss: Huber loss value
                - td_error_mean: Mean absolute TD error
                - q_values_mean: Mean Q-value
                - grad_norm: Gradient norm after clipping
        """
        # Compute TD target
        td_target = self.compute_td_target(batch)

        # Compute current Q-values
        obs = torch.as_tensor(
            batch["observation"], dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            batch["action"], dtype=torch.int64, device=self.device
        )

        q_values_all = self.online_network(obs)
        q_selected = q_values_all.gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)

        # Compute Huber loss
        if self.config.use_huber_loss:
            td_error = q_selected - td_target
            td_loss = F.huber_loss(q_selected, td_target, delta=self.config.huber_delta)
        else:
            td_error = q_selected - td_target
            td_loss = F.mse_loss(q_selected, td_target)
            td_error = td_error.detach()

        # Optimize
        self.optimizer.zero_grad()
        td_loss.backward()

        # Gradient clipping
        grad_norm = nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            max_norm=self.config.gradient_clip,
        )

        self.optimizer.step()

        # Update counters
        self.gradient_update_count += 1
        # Note: epsilon_state.step() is now called by the trainer on every environment step,
        # not here in update(). This ensures epsilon advances independently of warmup
        # and update_frequency. The trainer calls epsilon_state.step() before checking
        # should_update, so epsilon advances on every env step regardless of gradient update.

        # Return metrics
        return {
            "td_loss": float(td_loss.item()),
            "td_error_mean": float(td_error.abs().mean().item()),
            "q_values_mean": float(q_selected.mean().item()),
            "grad_norm": float(grad_norm.item()),
        }

    def sync_target_network(self) -> None:
        """
        Synchronize target network with online network.

        Hard copy: target_network.state_dict = online_network.state_dict
        """
        self.target_network.load_state_dict(self.online_network.state_dict())

    def maybe_sync_target(self) -> bool:
        """
        Synchronize target network if global_step is multiple of interval.

        Returns:
            True if sync occurred, False otherwise
        """
        if self.global_step > 0 and self.global_step % self.config.target_update_interval == 0:
            self.sync_target_network()
            return True
        return False

    @torch.no_grad()
    def evaluate_action(
        self,
        observation: np.ndarray,
    ) -> int:
        """
        Evaluate-mode action selection (greedy, epsilon=0).

        Args:
            observation: Observation array

        Returns:
            Greedy action ID
        """
        return self.select_action(observation, training=False)

    def get_checkpoint_data(self) -> Dict[str, Any]:
        """
        Get all data needed for checkpoint.

        Returns:
            Dict with:
                - online_network_state_dict
                - target_network_state_dict
                - optimizer_state_dict
                - global_step
                - gradient_update_count
                - epsilon_state (dict)
        """
        return {
            "online_network_state_dict": self.online_network.get_state_dict(),
            "target_network_state_dict": self.target_network.get_state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "gradient_update_count": self.gradient_update_count,
            "epsilon_state": self.epsilon_state.to_dict(),
        }

    def load_checkpoint_data(self, data: Dict[str, Any]) -> None:
        """
        Load checkpoint data.

        Args:
            data: Checkpoint dict
        """
        self.online_network.load_state_dict(data["online_network_state_dict"])
        self.target_network.load_state_dict(data["target_network_state_dict"])
        self.optimizer.load_state_dict(data["optimizer_state_dict"])
        self.global_step = data["global_step"]
        self.gradient_update_count = data["gradient_update_count"]
        self.epsilon_state = EpsilonState.from_dict(data["epsilon_state"])