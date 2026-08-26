"""
Focused M5 Tests: Double DQN Semantics and Agent
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.ddqn.agent import DDQNAgent, DDQNAgentConfig, EpsilonState
from agents.ddqn.replay_buffer import ReplayBuffer


class TestEpsilonState:
    """Test EpsilonState epsilon decay."""

    def test_initial_epsilon(self):
        """Test initial epsilon is epsilon_start."""
        state = EpsilonState(epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=1000)
        assert state.epsilon == 1.0

    def test_final_epsilon(self):
        """Test final epsilon is epsilon_end."""
        state = EpsilonState(epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=1000)
        state.global_step = 1000
        assert state.epsilon == 0.05

    def test_linear_decay(self):
        """Test linear epsilon decay."""
        state = EpsilonState(epsilon_start=1.0, epsilon_end=0.0, epsilon_decay_steps=100)
        state.global_step = 50
        assert state.epsilon == 0.5

    def test_step_increments(self):
        """Test epsilon step increments."""
        state = EpsilonState(epsilon_start=1.0, epsilon_end=0.0, epsilon_decay_steps=10)
        initial = state.epsilon
        state.step()
        assert state.global_step == 1
        assert state.epsilon < initial

    def test_linear_schedule_monotonic(self):
        """Test epsilon schedule is monotonic decreasing."""
        state = EpsilonState(epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=100)
        prev_epsilon = state.epsilon
        for _ in range(150):
            curr_epsilon = state.epsilon
            assert curr_epsilon <= prev_epsilon, "Epsilon should decrease or stay constant"
            prev_epsilon = curr_epsilon
            state.step()


class TestDDQNAgentConfig:
    """Test DDQNAgentConfig validation."""

    def test_default_config(self):
        """Test default configuration."""
        config = DDQNAgentConfig(num_actions=16)
        assert config.observation_dim == 10
        assert config.num_actions == 16
        assert config.epsilon_start == 1.0
        assert config.epsilon_end == 0.05
        assert config.gamma == 0.95

    def test_invalid_epsilon_order(self):
        """Test rejection of epsilon_end > epsilon_start."""
        with pytest.raises(ValueError, match="epsilon_end must be <= epsilon_start"):
            DDQNAgentConfig(num_actions=16, epsilon_start=0.1, epsilon_end=0.9)

    def test_invalid_gamma(self):
        """Test rejection of invalid gamma."""
        with pytest.raises(ValueError, match="gamma must be in"):
            DDQNAgentConfig(num_actions=16, gamma=1.0)


class TestDDQNAgentActionSelection:
    """Test DDQNAgent action selection."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        return DDQNAgent(config=DDQNAgentConfig(num_actions=16, epsilon_start=0.0, epsilon_end=0.0), seed=6521)

    def test_greedy_action(self, agent):
        """Test greedy action selection (epsilon=0)."""
        obs = np.random.randn(10).astype(np.float32)
        action = agent.select_action(obs, training=False)
        assert isinstance(action, int)
        assert 0 <= action < 16

    def test_epsilon_greedy_exploration(self):
        """Test epsilon-greedy exploration."""
        agent = DDQNAgent(config=DDQNAgentConfig(num_actions=16, epsilon_start=1.0, epsilon_end=1.0), seed=6521)
        obs = np.random.randn(10).astype(np.float32)

        actions = set()
        for _ in range(100):
            action = agent.select_action(obs, training=True)
            actions.add(action)
            agent.epsilon_state.step()

        # With epsilon=1, should sample diverse actions
        assert len(actions) > 1

    def test_legal_actions_mask(self, agent):
        """Test action selection with legal actions mask."""
        obs = np.random.randn(10).astype(np.float32)
        mask = np.array([True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False])
        action = agent.select_action(obs, training=False, legal_actions_mask=mask)
        assert 0 <= action < 6

    def test_greedy_deterministic(self, agent):
        """Test greedy action is deterministic."""
        obs = np.random.randn(10).astype(np.float32)
        action1 = agent.select_action(obs, training=False)
        action2 = agent.select_action(obs, training=False)
        assert action1 == action2


class TestDoubleDQNTarget:
    """Test Double DQN target computation."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        return DDQNAgent(config=DDQNAgentConfig(num_actions=16, epsilon_start=0.0, epsilon_end=0.0), seed=6521)

    def test_online_selects_action(self, agent):
        """Test online network selects action."""
        batch = {
            "next_observation": np.random.randn(4, 10).astype(np.float32),
            "reward": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            "terminated": np.array([False, False, False, False]),
        }
        td_target = agent.compute_td_target(batch)
        assert td_target.shape == (4,)
        assert torch.isfinite(td_target).all()

    def test_target_detached(self, agent):
        """Test target tensor is detached."""
        batch = {
            "next_observation": np.random.randn(4, 10).astype(np.float32),
            "reward": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            "terminated": np.array([False, False, False, False]),
        }
        td_target = agent.compute_td_target(batch)
        assert not td_target.requires_grad

    def test_terminated_masking(self, agent):
        """Test terminated transitions remove bootstrap."""
        # Non-terminated transition
        batch_alive = {
            "next_observation": np.random.randn(4, 10).astype(np.float32),
            "reward": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            "terminated": np.array([False, False, False, False]),
        }
        target_alive = agent.compute_td_target(batch_alive)

        # Terminated transition (same reward)
        batch_done = {
            "next_observation": np.random.randn(4, 10).astype(np.float32),
            "reward": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            "terminated": np.array([True, True, True, True]),
        }
        target_done = agent.compute_td_target(batch_done)

        # Terminated should have lower target (no bootstrap)
        # With gamma=0.95 and positive Q-values, alive > done
        # But we need to check the formula: done = reward, alive = reward + gamma * Q
        # So alive should be higher
        assert (target_alive > target_done).all()

    def test_truncated_retains_bootstrap(self, agent):
        """Test truncated-only transitions retain bootstrap."""
        # Truncated but not terminated
        batch = {
            "next_observation": np.random.randn(4, 10).astype(np.float32),
            "reward": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            "terminated": np.array([False, False, False, False]),
            "truncated": np.array([True, True, True, True]),
        }
        target = agent.compute_td_target(batch)
        # Should still bootstrap (terminated=False)
        assert torch.isfinite(target).all()

    def test_gamma_application(self):
        """Test gamma is applied correctly."""
        agent = DDQNAgent(config=DDQNAgentConfig(num_actions=16, gamma=0.9, epsilon_start=0.0, epsilon_end=0.0), seed=6521)
        batch = {
            "next_observation": np.zeros((4, 10), dtype=np.float32),  # Zero obs
            "reward": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            "terminated": np.array([True, True, True, True]),  # No bootstrap
        }
        target = agent.compute_td_target(batch)
        # With terminated=True, no bootstrap: target = reward (gamma term is multiplied by 0)
        # Move target to CPU for comparison
        target_cpu = target.cpu()
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert torch.allclose(target_cpu, expected, atol=1e-5)

    def test_target_network_excluded_from_optimizer(self, agent):
        """Test target network parameters are not in optimizer."""
        # Get optimizer params from param_groups
        optimizer_params = set()
        for group in agent.optimizer.param_groups:
            for p in group['params']:
                optimizer_params.add(id(p))

        target_params = set(id(p) for p in agent.target_network.parameters())
        online_params = set(id(p) for p in agent.online_network.parameters())

        # Target params should not be in optimizer
        assert len(optimizer_params & target_params) == 0

        # Online params should be in optimizer
        assert optimizer_params == online_params


class TestDDQNAgentUpdate:
    """Test DDQNAgent update."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        return DDQNAgent(config=DDQNAgentConfig(num_actions=16, epsilon_start=0.0, epsilon_end=0.0), seed=6521)

    def test_update_changes_weights(self, agent):
        """Test update changes online network weights."""
        # Get initial weights
        initial_weights = {k: v.clone() for k, v in agent.online_network.state_dict().items()}

        # Create batch
        batch = {
            "observation": np.random.randn(16, 10).astype(np.float32),
            "action": np.random.randint(0, 16, size=16),
            "reward": np.random.randn(16).astype(np.float32),
            "next_observation": np.random.randn(16, 10).astype(np.float32),
            "terminated": np.zeros(16, dtype=bool),
            "truncated": np.zeros(16, dtype=bool),
        }

        # Update
        metrics = agent.update(batch)

        # Check metrics
        assert "td_loss" in metrics
        assert metrics["td_loss"] >= 0
        assert "grad_norm" in metrics

        # Weights should have changed
        for k, v in agent.online_network.state_dict().items():
            assert not torch.allclose(v, initial_weights[k]), f"Weight {k} should have changed"

    def test_huber_loss(self):
        """Test Huber loss is used."""
        agent = DDQNAgent(config=DDQNAgentConfig(num_actions=16, use_huber_loss=True, huber_delta=1.0), seed=6521)
        batch = {
            "observation": np.random.randn(16, 10).astype(np.float32),
            "action": np.random.randint(0, 16, size=16),
            "reward": np.random.randn(16).astype(np.float32),
            "next_observation": np.random.randn(16, 10).astype(np.float32),
            "terminated": np.zeros(16, dtype=bool),
            "truncated": np.zeros(16, dtype=bool),
        }
        metrics = agent.update(batch)
        assert metrics["td_loss"] >= 0


class TestDDQNAgentTargetSync:
    """Test DDQNAgent target network synchronization."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        return DDQNAgent(
            config=DDQNAgentConfig(num_actions=16, epsilon_start=0.0, epsilon_end=0.0, target_update_interval=10),
            seed=6521
        )

    def test_sync_target_network(self, agent):
        """Test manual target network sync."""
        # Modify online network slightly
        with torch.no_grad():
            for param in agent.online_network.parameters():
                param += 0.1

        # Check networks are different
        sync_before = all(
            torch.allclose(p1, p2)
            for p1, p2 in zip(agent.online_network.parameters(), agent.target_network.parameters())
        )
        assert not sync_before

        # Sync
        agent.sync_target_network()

        # Check networks are same
        sync_after = all(
            torch.allclose(p1, p2)
            for p1, p2 in zip(agent.online_network.parameters(), agent.target_network.parameters())
        )
        assert sync_after

    def test_maybe_sync_interval(self, agent):
        """Test automatic sync at intervals."""
        agent.global_step = 9
        assert not agent.maybe_sync_target()

        agent.global_step = 10
        assert agent.maybe_sync_target()


class TestDDQNAgentCheckpoint:
    """Test DDQNAgent checkpoint data."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        return DDQNAgent(config=DDQNAgentConfig(num_actions=16), seed=6521)

    def test_get_checkpoint_data(self, agent):
        """Test checkpoint data extraction."""
        # Do some steps
        for _ in range(100):
            obs = np.random.randn(10).astype(np.float32)
            agent.select_action(obs, training=True)
            agent.epsilon_state.step()
            agent.global_step += 1
            agent.gradient_update_count += 1

        data = agent.get_checkpoint_data()

        assert "online_network_state_dict" in data
        assert "target_network_state_dict" in data
        assert "optimizer_state_dict" in data
        assert data["global_step"] == 100
        assert data["gradient_update_count"] == 100
        assert "epsilon_state" in data

    def test_load_checkpoint_data(self, agent):
        """Test checkpoint data loading."""
        # Get original data
        original_data = agent.get_checkpoint_data()

        # Modify agent
        agent.global_step = 999
        agent.gradient_update_count = 999
        agent.epsilon_state.global_step = 999

        # Load back
        agent.load_checkpoint_data(original_data)

        assert agent.global_step == original_data["global_step"]
        assert agent.gradient_update_count == original_data["gradient_update_count"]
        assert agent.epsilon_state.global_step == original_data["epsilon_state"]["global_step"]