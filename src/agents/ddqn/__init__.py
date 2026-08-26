"""
DDQN Agent Package for Milestone 5 Point-Estimate Double DQN.

Exports:
- QNetwork: MLP Q-network (10 -> 128 -> 128 -> K actions)
- ReplayBuffer: Fixed-capacity ring replay buffer
- DDQNAgent: Double DQN agent with epsilon-greedy action selection
- save_checkpoint, load_checkpoint: Checkpoint I/O
- compute_network_architecture_id: Authoritative architecture identity
- compute_scenario_bank_content_hash: Scenario bank content-based hashing
"""

from .q_network import QNetwork
from .replay_buffer import ReplayBuffer, ReplayBufferConfig, REPLAY_BUFFER_STATE_VERSION
from .agent import DDQNAgent, DDQNAgentConfig
from .checkpoint import save_checkpoint, load_checkpoint, CheckpointData, compute_scenario_bank_content_hash
from .identity import compute_network_architecture_id, compute_expected_network_architecture_id, get_architecture_revision, get_observation_schema_id, get_environment_contract_id

__all__ = [
    "QNetwork",
    "ReplayBuffer",
    "ReplayBufferConfig",
    "REPLAY_BUFFER_STATE_VERSION",
    "DDQNAgent",
    "DDQNAgentConfig",
    "save_checkpoint",
    "load_checkpoint",
    "CheckpointData",
    "compute_scenario_bank_content_hash",
    "compute_network_architecture_id",
    "compute_expected_network_architecture_id",
    "get_architecture_revision",
    "get_observation_schema_id",
    "get_environment_contract_id",
]