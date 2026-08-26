"""Baseline MSE RUL Predictor Models

Simple, well-regularized models for RUL prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class RULPredictorMSE(nn.Module):
    """Baseline MSE RUL Predictor.

    Architecture: MLP with layer normalization and dropout.
    Input: (batch, sequence_length, n_features) - flattened to (batch, sequence_length * n_features)
    Output: (batch,) - single RUL prediction per sample

    Args:
        n_features: Number of input features (default: 24 for 3 op_settings + 21 sensors)
        sequence_length: Sequence length (default: 50)
        hidden_dim: Hidden layer dimension (default: 128)
        n_layers: Number of hidden layers (default: 3)
        dropout: Dropout rate (default: 0.2)
    """

    def __init__(
        self,
        n_features: int = 24,
        sequence_length: int = 50,
        hidden_dim: int = 128,
        n_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.n_features = n_features
        self.sequence_length = sequence_length
        self.input_dim = sequence_length * n_features
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout_rate = dropout

        # Input projection
        self.input_proj = nn.Linear(self.input_dim, hidden_dim)

        # Hidden layers
        hidden_layers = []
        for _ in range(n_layers - 1):
            hidden_layers.extend([
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            ])

        self.hidden = nn.Sequential(*hidden_layers) if hidden_layers else nn.Identity()

        # Output head
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, sequence_length, n_features)

        Returns:
            RUL prediction of shape (batch,)
        """
        # Flatten sequence and features
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)  # (batch, sequence_length * n_features)

        # Input projection
        x = self.input_proj(x)

        # Hidden layers
        x = self.hidden(x)

        # Output
        x = self.output(x).squeeze(-1)  # (batch,)

        return x


class RULPredictorCNN(nn.Module):
    """CNN-based RUL Predictor (for comparison with existing rul_predictor.py).

    Architecture: 1D CNNs over sequence + global pooling + FC head.
    Input: (batch, sequence_length, n_features)
    Output: (batch,)

    Args:
        n_features: Number of input features (default: 24)
        sequence_length: Sequence length (default: 50)
        hidden_dim: Hidden layer dimension after CNN (default: 128)
        dropout: Dropout rate (default: 0.2)
    """

    def __init__(
        self,
        n_features: int = 24,
        sequence_length: int = 50,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.n_features = n_features
        self.sequence_length = sequence_length

        # CNN layers (operate over sequence dimension)
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)

        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)  # (batch, 128, 1)

        # FC head
        self.fc = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, sequence_length, n_features)

        Returns:
            RUL prediction of shape (batch,)
        """
        # Rearrange for Conv1d: (batch, n_features, sequence_length)
        x = x.transpose(1, 2)

        # CNN layers
        x = F.relu(self.bn1(self.conv1(x)))  # (batch, 64, seq_len)
        x = F.relu(self.bn2(self.conv2(x)))  # (batch, 128, seq_len)

        # Global pooling
        x = self.pool(x).squeeze(-1)  # (batch, 128)

        # FC head
        x = self.fc(x).squeeze(-1)  # (batch,)

        return x


def build_predictor(
    model_type: str = "mlp",
    n_features: int = 24,
    sequence_length: int = 50,
    hidden_dim: int = 128,
    n_layers: int = 3,
    dropout: float = 0.2,
    **kwargs
) -> nn.Module:
    """Factory function to build RUL predictor.

    Args:
        model_type: "mlp" or "cnn"
        n_features: Number of input features
        sequence_length: Sequence length
        hidden_dim: Hidden dimension
        n_layers: Number of layers (MLP only)
        dropout: Dropout rate
        **kwargs: Additional arguments

    Returns:
        RUL predictor model
    """
    if model_type == "mlp":
        return RULPredictorMSE(
            n_features=n_features,
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
        )
    elif model_type == "cnn":
        return RULPredictorCNN(
            n_features=n_features,
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'mlp' or 'cnn'.")