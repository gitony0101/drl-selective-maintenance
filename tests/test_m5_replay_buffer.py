"""
Focused M5 Tests: Replay Buffer
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.ddqn.replay_buffer import ReplayBuffer, ReplayBufferConfig, Transition


class TestReplayBufferConfig:
    """Test ReplayBufferConfig validation."""

    def test_default_config(self):
        """Test default configuration."""
        config = ReplayBufferConfig(capacity=1000, observation_dim=10)
        assert config.capacity == 1000
        assert config.observation_dim == 10

    def test_invalid_capacity(self):
        """Test rejection of invalid capacity."""
        with pytest.raises(ValueError, match="capacity must be positive"):
            ReplayBufferConfig(capacity=0, observation_dim=10)

    def test_invalid_observation_dim(self):
        """Test rejection of invalid observation_dim."""
        with pytest.raises(ValueError, match="observation_dim must be positive"):
            ReplayBufferConfig(capacity=1000, observation_dim=0)


class TestReplayBufferInsert:
    """Test ReplayBuffer insertion."""

    @pytest.fixture
    def buffer(self):
        """Create replay buffer."""
        return ReplayBuffer(capacity=100, observation_dim=10, seed=6521)

    def test_single_insert(self, buffer):
        """Test single transition insertion."""
        obs = np.random.randn(10).astype(np.float32)
        next_obs = np.random.randn(10).astype(np.float32)
        buffer.insert(obs, action_id=5, reward=1.0, next_observation=next_obs, terminated=False, truncated=False)
        assert len(buffer) == 1

    def test_multiple_inserts(self, buffer):
        """Test multiple transitions insertion."""
        for i in range(10):
            obs = np.random.randn(10).astype(np.float32)
            next_obs = np.random.randn(10).astype(np.float32)
            buffer.insert(obs, action_id=i, reward=float(i), next_observation=next_obs, terminated=False, truncated=False)
        assert len(buffer) == 10

    def test_wraparound(self, buffer):
        """Test ring buffer wraparound."""
        for i in range(150):  # More than capacity
            obs = np.random.randn(10).astype(np.float32)
            next_obs = np.random.randn(10).astype(np.float32)
            buffer.insert(obs, action_id=i % 16, reward=float(i), next_observation=next_obs, terminated=False, truncated=False)
        assert len(buffer) == 100  # Capacity

    def test_invalid_observation_shape(self, buffer):
        """Test rejection of invalid observation shape."""
        obs = np.random.randn(5)  # Wrong dimension
        next_obs = np.random.randn(10).astype(np.float32)
        with pytest.raises(ValueError, match="observation shape mismatch"):
            buffer.insert(obs, 5, 1.0, next_obs, False, False)

    def test_nan_observation(self, buffer):
        """Test rejection of NaN observation."""
        obs = np.full(10, np.nan, dtype=np.float32)
        next_obs = np.random.randn(10).astype(np.float32)
        with pytest.raises(ValueError, match="non-finite"):
            buffer.insert(obs, 5, 1.0, next_obs, False, False)

    def test_inf_observation(self, buffer):
        """Test rejection of Inf observation."""
        obs = np.full(10, np.inf, dtype=np.float32)
        next_obs = np.random.randn(10).astype(np.float32)
        with pytest.raises(ValueError, match="non-finite"):
            buffer.insert(obs, 5, 1.0, next_obs, False, False)

    def test_nan_next_observation(self, buffer):
        """Test rejection of NaN next_observation."""
        obs = np.random.randn(10).astype(np.float32)
        next_obs = np.full(10, np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match="non-finite"):
            buffer.insert(obs, 5, 1.0, next_obs, False, False)

    def test_nan_reward(self, buffer):
        """Test rejection of NaN reward."""
        obs = np.random.randn(10).astype(np.float32)
        next_obs = np.random.randn(10).astype(np.float32)
        with pytest.raises(ValueError, match="finite"):
            buffer.insert(obs, 5, np.nan, next_obs, False, False)

    def test_invalid_action_type(self, buffer):
        """Test rejection of invalid action type."""
        obs = np.random.randn(10).astype(np.float32)
        next_obs = np.random.randn(10).astype(np.float32)
        with pytest.raises(ValueError, match="integral"):
            buffer.insert(obs, "invalid", 1.0, next_obs, False, False)

    def test_negative_action(self, buffer):
        """Test rejection of negative action."""
        obs = np.random.randn(10).astype(np.float32)
        next_obs = np.random.randn(10).astype(np.float32)
        with pytest.raises(ValueError, match="non-negative"):
            buffer.insert(obs, -1, 1.0, next_obs, False, False)


class TestReplayBufferSample:
    """Test ReplayBuffer sampling."""

    @pytest.fixture
    def filled_buffer(self):
        """Create filled replay buffer."""
        buffer = ReplayBuffer(capacity=100, observation_dim=10, seed=6521)
        for i in range(50):
            obs = np.random.randn(10).astype(np.float32)
            next_obs = np.random.randn(10).astype(np.float32)
            buffer.insert(obs, action_id=i % 16, reward=float(i), next_observation=next_obs, terminated=i % 10 == 0, truncated=False)
        return buffer

    def test_sample_shape(self, filled_buffer):
        """Test sample returns correct shapes."""
        batch = filled_buffer.sample_batch(16)
        assert batch["observation"].shape == (16, 10)
        assert batch["action"].shape == (16,)
        assert batch["reward"].shape == (16,)
        assert batch["next_observation"].shape == (16, 10)
        assert batch["terminated"].shape == (16,)
        assert batch["truncated"].shape == (16,)

    def test_sample_dtypes(self, filled_buffer):
        """Test sample returns correct dtypes."""
        batch = filled_buffer.sample_batch(16)
        assert batch["observation"].dtype == np.float32
        assert batch["action"].dtype == np.int64
        assert batch["reward"].dtype == np.float32
        assert batch["next_observation"].dtype == np.float32
        assert batch["terminated"].dtype == bool
        assert batch["truncated"].dtype == bool

    def test_deterministic_sampling(self):
        """Test seeded sampling is deterministic."""
        buffer1 = ReplayBuffer(capacity=100, observation_dim=10, seed=6521)
        buffer2 = ReplayBuffer(capacity=100, observation_dim=10, seed=6521)

        for i in range(50):
            obs = np.random.randn(10).astype(np.float32)
            next_obs = np.random.randn(10).astype(np.float32)
            buffer1.insert(obs, i % 16, float(i), next_obs, False, False)
            buffer2.insert(obs, i % 16, float(i), next_obs, False, False)

        batch1 = buffer1.sample_batch(16)
        batch2 = buffer2.sample_batch(16)

        assert np.array_equal(batch1["observation"], batch2["observation"])
        assert np.array_equal(batch1["action"], batch2["action"])

    def test_insufficient_transitions(self, filled_buffer):
        """Test rejection when sampling more than available."""
        with pytest.raises(ValueError, match="Cannot sample"):
            filled_buffer.sample(100)  # Buffer has 50

    def test_terminated_truncated_separation(self, filled_buffer):
        """Test terminated and truncated are returned separately."""
        batch = filled_buffer.sample_batch(16)
        assert "terminated" in batch
        assert "truncated" in batch
        assert isinstance(batch["terminated"], np.ndarray)
        assert isinstance(batch["truncated"], np.ndarray)


class TestReplayBufferClear:
    """Test ReplayBuffer clear."""

    def test_clear(self):
        """Test buffer clear."""
        buffer = ReplayBuffer(capacity=100, observation_dim=10)
        for i in range(50):
            obs = np.random.randn(10).astype(np.float32)
            next_obs = np.random.randn(10).astype(np.float32)
            buffer.insert(obs, i, 1.0, next_obs, False, False)
        assert len(buffer) == 50
        buffer.clear()
        assert len(buffer) == 0


class TestReplayBufferRecentTransitions:
    """Test ReplayBuffer get_recent_transitions."""

    def test_get_recent(self):
        """Test getting recent transitions."""
        buffer = ReplayBuffer(capacity=100, observation_dim=10, seed=6521)
        for i in range(50):
            obs = np.full(10, i, dtype=np.float32)
            next_obs = np.random.randn(10).astype(np.float32)
            buffer.insert(obs, i, float(i), next_obs, False, False)

        recent = buffer.get_recent_transitions(10)
        assert recent["observation"].shape == (10, 10)
        # Last 10 inserted have values 40-49
        for j in range(10):
            assert recent["observation"][j, 0] == 40 + j


class TestReplayBufferPersistence:
    """Test ReplayBuffer state_dict/load_state_dict persistence."""

    def test_state_dict_contains_required_fields(self):
        """Test state_dict contains all required fields."""
        buffer = ReplayBuffer(capacity=100, observation_dim=10, seed=6521)
        for i in range(50):
            obs = np.random.randn(10).astype(np.float32)
            next_obs = np.random.randn(10).astype(np.float32)
            buffer.insert(obs, i % 16, float(i), next_obs, False, False)

        state = buffer.state_dict(action_count=16)

        required_fields = [
            "observations", "actions", "rewards",
            "next_observations", "terminated", "truncated",
            "write_index", "current_size", "capacity",
            "observation_dim", "rng_state",
        ]
        for field in required_fields:
            assert field in state, f"Missing required field: {field}"

    def test_load_state_dict_restores_state(self):
        """Test load_state_dict restores full buffer state."""
        buffer1 = ReplayBuffer(capacity=100, observation_dim=10, seed=6521)
        for i in range(50):
            obs = np.random.randn(10).astype(np.float32)
            next_obs = np.random.randn(10).astype(np.float32)
            buffer1.insert(obs, i % 16, float(i), next_obs, i % 10 == 0, False)

        # Save state
        state = buffer1.state_dict(action_count=16)

        # Create fresh buffer and restore
        buffer2 = ReplayBuffer(capacity=100, observation_dim=10, seed=9999)  # Different seed
        buffer2.load_state_dict(state, expected_action_count=16)

        # Verify restored
        assert buffer2.current_size == buffer1.current_size
        assert buffer2.write_index == buffer1.write_index
        assert np.array_equal(buffer2.observations[:50], buffer1.observations[:50])
        assert np.array_equal(buffer2.actions[:50], buffer1.actions[:50])
        assert np.array_equal(buffer2.rewards[:50], buffer1.rewards[:50])

    def test_load_state_dict_restores_rng_state(self):
        """Test load_state_dict restores RNG state for identical sampling."""
        buffer1 = ReplayBuffer(capacity=100, observation_dim=10, seed=6521)
        buffer2 = ReplayBuffer(capacity=100, observation_dim=10, seed=9999)

        # Fill both buffers identically
        for i in range(50):
            obs = np.full(10, i, dtype=np.float32)
            next_obs = np.full(10, i + 100, dtype=np.float32)
            buffer1.insert(obs, i % 16, float(i), next_obs, False, False)
            buffer2.insert(obs, i % 16, float(i), next_obs, False, False)

        # Sample from buffer1 to advance its RNG
        _ = buffer1.sample_batch(16)

        # Save buffer1 state (includes RNG after sample)
        state = buffer1.state_dict(action_count=16)

        # Restore buffer2 to buffer1's state (including RNG)
        buffer2.load_state_dict(state, expected_action_count=16)

        # Now sample from both - should be identical since RNG state is restored
        batch1_after = buffer1.sample_batch(16)
        batch2_after = buffer2.sample_batch(16)

        # Verify identical sampling due to RNG restore
        assert np.array_equal(batch1_after["observation"], batch2_after["observation"])
        assert np.array_equal(batch1_after["action"], batch2_after["action"])
        assert np.array_equal(batch1_after["reward"], batch2_after["reward"])

    def test_load_state_dict_validates_capacity(self):
        """Test load_state_dict rejects wrong capacity."""
        buffer = ReplayBuffer(capacity=100, observation_dim=10, seed=6521)
        state = buffer.state_dict(action_count=16)

        wrong_buffer = ReplayBuffer(capacity=200, observation_dim=10, seed=6521)
        with pytest.raises(ValueError, match="capacity mismatch"):
            wrong_buffer.load_state_dict(state, expected_action_count=16)

    def test_load_state_dict_validates_observation_dim(self):
        """Test load_state_dict rejects wrong observation dimension."""
        buffer = ReplayBuffer(capacity=100, observation_dim=10, seed=6521)
        state = buffer.state_dict(action_count=16)

        wrong_buffer = ReplayBuffer(capacity=100, observation_dim=5, seed=6521)
        with pytest.raises(ValueError, match="observation_dim mismatch"):
            wrong_buffer.load_state_dict(state, expected_action_count=16)

    def test_load_state_dict_validates_required_fields(self):
        """Test load_state_dict rejects incomplete state."""
        buffer = ReplayBuffer(capacity=100, observation_dim=10, seed=6521)
        state = buffer.state_dict(action_count=16)
        del state["rng_state"]  # Remove required field

        with pytest.raises(ValueError, match="missing required fields"):
            buffer.load_state_dict(state, expected_action_count=16)

    def test_load_state_dict_validates_shapes(self):
        """Test load_state_dict rejects wrong array shapes."""
        buffer = ReplayBuffer(capacity=100, observation_dim=10, seed=6521)
        state = buffer.state_dict(action_count=16)
        state["observations"] = np.zeros((50, 10), dtype=np.float32)  # Wrong capacity

        with pytest.raises(ValueError, match="shape mismatch"):
            buffer.load_state_dict(state, expected_action_count=16)

    def test_load_state_dict_validates_finite_values(self):
        """Test load_state_dict rejects non-finite values."""
        buffer = ReplayBuffer(capacity=100, observation_dim=10, seed=6521)
        state = buffer.state_dict(action_count=16)
        state["observations"][0, 0] = np.nan

        with pytest.raises(ValueError, match="non-finite"):
            buffer.load_state_dict(state, expected_action_count=16)