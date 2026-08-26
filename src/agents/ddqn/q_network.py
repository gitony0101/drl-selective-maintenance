"""
Q-Network for Milestone 5 Point-Estimate Double DQN.

Implements MLP architecture:
  Input: (batch, 10) - flattened observation (5 slots x 2 features)
  Hidden 1: Linear(10, 128) + ReLU
  Hidden 2: Linear(128, 128) + ReLU
  Output: Linear(128, num_actions) - Q-values per action

Supports:
- CPU, Apple MPS, CUDA device resolution
- Batched and single observations
- float32 throughout
- Finite output validation
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple, Optional, Literal, Union
from dataclasses import dataclass

DeviceLiteral = Literal["cpu", "mps", "cuda"]


@dataclass(frozen=True)
class QNetworkConfig:
    """Immutable Q-network configuration."""

    input_dim: int = 10
    hidden_dim: int = 128
    num_hidden_layers: int = 2
    output_dim: Optional[int] = None  # Derived from action table
    activation: str = "relu"
    device: Optional[DeviceLiteral] = None  # Auto-resolve if None

    def __post_init__(self) -> None:
        errors = []
        if self.input_dim <= 0:
            errors.append(f"input_dim must be positive, got {self.input_dim}")
        if self.hidden_dim <= 0:
            errors.append(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.num_hidden_layers <= 0:
            errors.append(f"num_hidden_layers must be positive, got {self.num_hidden_layers}")
        if self.output_dim is not None and self.output_dim <= 0:
            errors.append(f"output_dim must be positive, got {self.output_dim}")
        if errors:
            raise ValueError("QNetworkConfig validation failed:\n  - " + "\n  - ".join(errors))


def resolve_device(explicit_device: Optional[str] = None) -> torch.device:
    """
    Resolve torch device with explicit override support.

    Priority (if explicit_device is None):
    1. CUDA if available
    2. MPS if available
    3. CPU

    Args:
        explicit_device: Optional explicit device ("cpu", "mps", "cuda")

    Returns:
        torch.device instance
    """
    if explicit_device is not None:
        if explicit_device == "cpu":
            return torch.device("cpu")
        elif explicit_device == "mps":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            # MPS requested but not available - fall back to CPU
            return torch.device("cpu")
        elif explicit_device == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            # CUDA requested but not available - fall back to CPU
            return torch.device("cpu")
        else:
            return torch.device(explicit_device)

    # Auto-resolution
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class QNetwork(nn.Module):
    """
    MLP Q-network for Double DQN.

    Architecture:
        Input (batch, input_dim)
          -> Linear(input_dim, hidden_dim)
          -> ReLU
          -> Linear(hidden_dim, hidden_dim) [repeated num_hidden_layers times]
          -> ReLU
          -> Linear(hidden_dim, output_dim)
          -> Q-values (batch, output_dim)

    Attributes:
        config: Network configuration
        device: Device network resides on
        input_dim: Input dimension (10 for M5)
        output_dim: Output dimension (6 for K=1, 16 for K=2)
    """

    config: QNetworkConfig
    device: torch.device

    def __init__(
        self,
        config: Optional[QNetworkConfig] = None,
        input_dim: int = 10,
        hidden_dim: int = 128,
        num_hidden_layers: int = 2,
        output_dim: Optional[int] = None,
        explicit_device: Optional[str] = None,
    ):
        """
        Initialize Q-network.

        Args:
            config: Optional configuration (takes precedence over individual args)
            input_dim: Input dimension (default 10 for M5)
            hidden_dim: Hidden layer dimension (default 128)
            num_hidden_layers: Number of hidden layers (default 2)
            output_dim: Output dimension (derived from action table)
            explicit_device: Explicit device override ("cpu", "mps", "cuda")

        Raises:
            ValueError: If configuration is invalid
        """
        super().__init__()

        # Build config from args or use provided config
        if config is not None:
            self.config = config
        else:
            self.config = QNetworkConfig(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_hidden_layers=num_hidden_layers,
                output_dim=output_dim,
            )

        self.device = resolve_device(explicit_device)
        self.input_dim = self.config.input_dim
        self.output_dim = self.config.output_dim

        # Validate output_dim is set
        if self.output_dim is None:
            raise ValueError(
                "output_dim must be provided either via config or explicit argument. "
                "For K=1 use output_dim=6, for K=2 use output_dim=16."
            )

        # Build layers
        layers = []

        # Input layer
        layers.append(nn.Linear(self.input_dim, self.config.hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers (num_hidden_layers - 1 additional layers since we already added one)
        for _ in range(self.config.num_hidden_layers - 1):
            layers.append(nn.Linear(self.config.hidden_dim, self.config.hidden_dim))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(self.config.hidden_dim, self.output_dim))

        self.network = nn.Sequential(*layers)

        # Move to device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Q-network.

        Args:
            x: Input tensor of shape (batch_size, input_dim) or (input_dim,)

        Returns:
            Q-values tensor of shape (batch_size, output_dim) or (output_dim,)

        Raises:
            ValueError: If input shape is invalid
        """
        # Handle single observation (input_dim,) -> (1, input_dim)
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single = True
        else:
            single = False

        # Validate input shape
        if x.dim() != 2:
            raise ValueError(f"Input must be (batch, input_dim) or (input_dim,), got shape {x.shape}")
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Input dimension mismatch: expected {self.input_dim}, got {x.shape[1]}"
            )

        # Forward pass
        q_values = self.network(x)

        # Validate finite outputs
        if not torch.isfinite(q_values).all():
            raise ValueError(
                f"Q-network produced non-finite outputs: "
                f"nan={torch.isnan(q_values).sum().item()}, "
                f"inf={torch.isinf(q_values).sum().item()}"
            )

        # Squeeze for single observation
        if single:
            q_values = q_values.squeeze(0)

        return q_values

    @torch.no_grad()
    def select_action(self, x: torch.Tensor, legal_actions_mask: Optional[torch.Tensor] = None) -> int:
        """
        Select greedy action (argmax Q).

        Args:
            x: Input observation (input_dim,) or (batch, input_dim)
            legal_actions_mask: Optional boolean mask of legal actions (output_dim,)

        Returns:
            Action ID (integer in [0, output_dim))

        Raises:
            ValueError: If no legal actions available
        """
        q_values = self.forward(x)

        # Handle batched input - take first
        if q_values.dim() == 2:
            q_values = q_values[0]

        if legal_actions_mask is not None:
            # Mask illegal actions with -inf
            masked_q = q_values.clone()
            masked_q[~legal_actions_mask] = float('-inf')
            if masked_q.max() == float('-inf'):
                raise ValueError("No legal actions available")
            return int(masked_q.argmax().item())
        else:
            return int(q_values.argmax().item())

    def get_state_dict(self) -> dict:
        """Get network state dict."""
        return self.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """Load network state dict."""
        super().load_state_dict(state_dict)


def create_q_network_for_action_table(
    action_table: Tuple[Tuple[int, ...], ...],
    hidden_dim: int = 128,
    num_hidden_layers: int = 2,
    explicit_device: Optional[str] = None,
) -> QNetwork:
    """
    Create Q-network configured for a specific action table.

    Args:
        action_table: Action table from src.envs.action_table
        hidden_dim: Hidden layer dimension
        num_hidden_layers: Number of hidden layers
        explicit_device: Explicit device override

    Returns:
        Configured QNetwork instance
    """
    output_dim = len(action_table)
    return QNetwork(
        input_dim=10,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        output_dim=output_dim,
        explicit_device=explicit_device,
    )