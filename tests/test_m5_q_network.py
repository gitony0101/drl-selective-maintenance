"""
Focused M5 Tests: Q-Network
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.ddqn.q_network import QNetwork, QNetworkConfig, resolve_device, create_q_network_for_action_table
from envs.action_table import ACTION_TABLE_N5_K1, ACTION_TABLE_N5_K2


class TestQNetworkConfig:
    """Test QNetworkConfig validation."""

    def test_default_config(self):
        """Test default configuration."""
        config = QNetworkConfig(input_dim=10, output_dim=16)
        assert config.input_dim == 10
        assert config.hidden_dim == 128
        assert config.num_hidden_layers == 2
        assert config.output_dim == 16

    def test_invalid_input_dim(self):
        """Test rejection of invalid input_dim."""
        with pytest.raises(ValueError, match="input_dim must be positive"):
            QNetworkConfig(input_dim=0, output_dim=16)

    def test_invalid_output_dim(self):
        """Test rejection of invalid output_dim."""
        with pytest.raises(ValueError, match="output_dim must be positive"):
            QNetworkConfig(input_dim=10, output_dim=-1)

    def test_invalid_hidden_dim(self):
        """Test rejection of invalid hidden_dim."""
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            QNetworkConfig(input_dim=10, hidden_dim=0, output_dim=16)


class TestQNetworkForward:
    """Test QNetwork forward pass."""

    @pytest.fixture
    def network_k2(self):
        """Create QNetwork for K=2."""
        return QNetwork(input_dim=10, hidden_dim=128, num_hidden_layers=2, output_dim=16, explicit_device="cpu")

    @pytest.fixture
    def network_k1(self):
        """Create QNetwork for K=1."""
        return QNetwork(input_dim=10, hidden_dim=128, num_hidden_layers=2, output_dim=6, explicit_device="cpu")

    def test_single_observation_k2(self, network_k2):
        """Test single observation forward (K=2)."""
        obs = torch.randn(10)
        q_values = network_k2(obs)
        assert q_values.shape == (16,)
        assert torch.isfinite(q_values).all()

    def test_batched_observation_k2(self, network_k2):
        """Test batched observation forward (K=2)."""
        obs = torch.randn(4, 10)
        q_values = network_k2(obs)
        assert q_values.shape == (4, 16)
        assert torch.isfinite(q_values).all()

    def test_single_observation_k1(self, network_k1):
        """Test single observation forward (K=1)."""
        obs = torch.randn(10)
        q_values = network_k1(obs)
        assert q_values.shape == (6,)
        assert torch.isfinite(q_values).all()

    def test_batched_observation_k1(self, network_k1):
        """Test batched observation forward (K=1)."""
        obs = torch.randn(8, 10)
        q_values = network_k1(obs)
        assert q_values.shape == (8, 6)
        assert torch.isfinite(q_values).all()

    def test_invalid_input_shape(self, network_k2):
        """Test rejection of invalid input shape."""
        obs = torch.randn(5)  # Wrong dimension
        with pytest.raises(ValueError, match="Input dimension mismatch"):
            network_k2(obs)

    def test_2d_invalid_input_shape(self, network_k2):
        """Test rejection of 2D invalid input shape."""
        obs = torch.randn(4, 5)  # Wrong dimension
        with pytest.raises(ValueError, match="Input dimension mismatch"):
            network_k2(obs)

    def test_finite_outputs(self, network_k2):
        """Test that outputs are always finite."""
        for _ in range(10):
            obs = torch.randn(4, 10)
            q_values = network_k2(obs)
            assert torch.isfinite(q_values).all()


class TestQNetworkSelectAction:
    """Test QNetwork action selection."""

    @pytest.fixture
    def network(self):
        """Create QNetwork for testing."""
        return QNetwork(input_dim=10, hidden_dim=128, num_hidden_layers=2, output_dim=16, explicit_device="cpu")

    def test_greedy_action(self, network):
        """Test greedy action selection."""
        obs = torch.randn(10)
        action = network.select_action(obs)
        assert isinstance(action, int)
        assert 0 <= action < 16

    def test_greedy_action_batched(self, network):
        """Test greedy action selection with batched input."""
        obs = torch.randn(4, 10)
        action = network.select_action(obs)
        assert isinstance(action, int)
        assert 0 <= action < 16

    def test_legal_actions_mask(self, network):
        """Test action selection with legal actions mask."""
        obs = torch.randn(10)
        mask = torch.tensor([True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False])
        action = network.select_action(obs, legal_actions_mask=mask)
        assert 0 <= action < 6

    def test_no_legal_actions(self, network):
        """Test rejection when no legal actions."""
        obs = torch.randn(10)
        mask = torch.zeros(16, dtype=torch.bool)
        with pytest.raises(ValueError, match="No legal actions"):
            network.select_action(obs, legal_actions_mask=mask)


class TestCreateQNetworkForActionTable:
    """Test create_q_network_for_action_table factory."""

    def test_k1_action_table(self):
        """Test network creation for K=1 action table."""
        network = create_q_network_for_action_table(ACTION_TABLE_N5_K1, explicit_device="cpu")
        assert network.output_dim == 6

    def test_k2_action_table(self):
        """Test network creation for K=2 action table."""
        network = create_q_network_for_action_table(ACTION_TABLE_N5_K2, explicit_device="cpu")
        assert network.output_dim == 16


class TestResolveDevice:
    """Test device resolution."""

    def test_cpu_explicit(self):
        """Test explicit CPU selection."""
        device = resolve_device("cpu")
        assert device.type == "cpu"

    def test_auto_resolve(self):
        """Test auto device resolution."""
        device = resolve_device()
        assert device.type in ("cpu", "cuda", "mps")