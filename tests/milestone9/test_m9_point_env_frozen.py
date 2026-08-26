"""M9 Point-Estimate — frozen environment invariants (Step 5).

Verifies invariants 2, 3, 8, 9 against the REAL frozen env fed by the
session-scoped real seed-6521 cache fixture (generated via the frozen CLI,
TEMP dir — never the formal cache root, never 06_PREDICTIONS).

  2. observation is exactly (10,) float32, normalized [age/341, pred_rul/125]
  3. action space is exactly Discrete(16) (K=2)
  8. reward = -C_t <= 0 (cost semantics unchanged)
  9. terminated == False always; truncated == (step >= horizon)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.m9_slow

REPO_ROOT = Path(__file__).resolve().parents[2]
VAL_BANK = str(
    REPO_ROOT / "data" / "scenario_banks" / "m4_production"
    / "rl_validation_K2_failure-light-no-waste.json"
)


def _make_env(cache_dir: Path, split: str, seed: int = 6521):
    import sys
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from src.envs import SelectiveMaintenanceEnv, get_default_config
    cfg = get_default_config(
        split=split,
        cost_regime_id="failure-light-no-waste",
        maintenance_capacity=2,
        scenario_bank_path=VAL_BANK,
        prediction_cache_path=str(cache_dir),
        seed=seed,
    )
    return SelectiveMaintenanceEnv(config=cfg)


def test_observation_shape_dtype_normalization(real_seed6521_cache_dir):
    """Invariant 2: obs shape (10,) float32; per engine [age/341, pred_rul/125]
    each clipped to [0,1]."""
    env = _make_env(real_seed6521_cache_dir, "rl_validation")
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (10,)
    assert obs.dtype == np.float32
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)


def test_action_space_is_16(real_seed6521_cache_dir):
    """Invariant 3: action space is exactly Discrete(16) (K=2)."""
    env = _make_env(real_seed6521_cache_dir, "rl_validation")
    assert env.action_space.n == 16


def test_reward_nonpositive_terminated_false(real_seed6521_cache_dir):
    """Invariants 8 & 9a: reward = -C_t <= 0; terminated always False."""
    env = _make_env(real_seed6521_cache_dir, "rl_validation")
    obs, _ = env.reset()
    terminated = False
    total_reward = 0.0
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        assert reward <= 0.0, f"reward {reward} > 0 violates reward = -C_t"
        assert terminated is False, "terminated must always be False"
        total_reward += reward
    assert total_reward <= 0.0


def test_truncated_at_horizon(real_seed6521_cache_dir):
    """Invariant 9b: truncated == (step_index >= horizon). Step until truncation
    and confirm it eventually fires (and never coincides with termination)."""
    env = _make_env(real_seed6521_cache_dir, "rl_validation")
    horizon = env.config.episode_horizon
    env.reset()
    truncated = False
    terminated = False
    steps = 0
    guard = horizon + 5
    while not truncated and steps < guard:
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        steps += 1
    assert truncated is True, "episode must truncate after >= horizon steps"
    assert terminated is False
