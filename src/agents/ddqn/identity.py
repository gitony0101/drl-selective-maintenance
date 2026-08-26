"""
Authoritative Network Architecture Identity for Milestone 5 DDQN.

Single source of truth for network architecture identity computation.
Used by:
- Checkpoint writer (save_checkpoint)
- Manifest writer (run_manifest.json)
- Resume validation (load_checkpoint)
- Evaluator (evaluate_ddqn.py)
- Tests

This ensures identical architecture IDs across all production paths.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Any


# Frozen architecture revision for M5 point-estimate DDQN
ARCHITECTURE_REVISION = "m5_point_v1"


def compute_network_architecture_id(
    observation_dim: int,
    hidden_dim: int,
    num_hidden_layers: int,
    activation: str,
    action_count: int,
    architecture_revision: str = ARCHITECTURE_REVISION,
) -> str:
    """
    Compute a unique identifier for network architecture.

    Encodes:
    - observation dimension
    - hidden-layer widths
    - number of hidden layers
    - activation family
    - action count (output dimension)
    - architecture revision tag

    This is the SINGLE authoritative implementation used by all production code paths.

    Args:
        observation_dim: Input observation dimension (10 for M5)
        hidden_dim: Hidden layer width
        num_hidden_layers: Number of hidden layers
        activation: Activation function name (e.g., "relu")
        action_count: Number of output actions (6 for K=1, 16 for K=2)
        architecture_revision: Architecture revision tag

    Returns:
        SHA256 hash of architecture specification (64-char lowercase hex)
    """
    spec = {
        "observation_dim": observation_dim,
        "hidden_dim": hidden_dim,
        "num_hidden_layers": num_hidden_layers,
        "activation": activation,
        "action_count": action_count,
        "architecture_revision": architecture_revision,
    }
    serialized = json.dumps(spec, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def compute_expected_network_architecture_id(
    config: Dict[str, Any],
    action_count: int,
    architecture_revision: str = ARCHITECTURE_REVISION,
) -> str:
    """
    Compute expected network architecture ID from trainer configuration.

    Args:
        config: Trainer configuration dict with hidden_dim, num_hidden_layers
        action_count: Number of output actions
        architecture_revision: Architecture revision tag

    Returns:
        SHA256 hash of expected architecture specification
    """
    return compute_network_architecture_id(
        observation_dim=config.get("observation_dim", 10),
        hidden_dim=config.get("hidden_dim", 128),
        num_hidden_layers=config.get("num_hidden_layers", 2),
        activation="relu",  # Frozen activation for M5
        action_count=action_count,
        architecture_revision=architecture_revision,
    )


def get_architecture_revision() -> str:
    """Get the frozen architecture revision for M5."""
    return ARCHITECTURE_REVISION


def get_observation_schema_id() -> str:
    """Get the frozen observation schema ID for M5."""
    return "m5_point_v1"


def get_environment_contract_id() -> str:
    """Get the frozen environment contract ID for M5."""
    return "m2_v1"