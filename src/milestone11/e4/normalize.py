"""E4 (M11) train-only normalization statistics (Section 11).

Computes mean/std of the delta-observation targets (10-dim) and the reward
targets (1-dim) using dynamics_train only. Never uses validation or holdout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .dataset import E4RawTransition
from .paths import OBSERVATION_DIM


def build_inputs(transitions: List[E4RawTransition],
                 device=None) -> Dict[str, torch.Tensor]:
    """Build model inputs/targets from a list of transitions.

    Returns dict with:
      x        [B,26]  = concat(obs_t[10], one_hot(action)[16])
      delta    [B,10]  = next_obs - obs
      reward   [B]     = reward_t
    """
    obs = torch.tensor(
        np.array([t.observation_t for t in transitions], dtype=np.float32)
    )
    nxt = torch.tensor(
        np.array([t.next_observation_t for t in transitions], dtype=np.float32)
    )
    actions = torch.tensor([t.action_id_t for t in transitions], dtype=torch.long)
    onehot = torch.nn.functional.one_hot(actions, num_classes=16).float()
    delta = nxt - obs
    reward = torch.tensor([t.reward_t for t in transitions], dtype=torch.float32)
    if device is None:
        return {
            "x": torch.cat([obs, onehot], dim=1),
            "delta": delta,
            "reward": reward,
            "obs": obs,
            "onehot": onehot,
        }
    return {k: v.to(device) for k, v in {
        "x": torch.cat([obs, onehot], dim=1),
        "delta": delta, "reward": reward, "obs": obs, "onehot": onehot,
    }.items()}


def compute_normalization(transitions: List[E4RawTransition]) -> Dict[str, torch.Tensor]:
    """Mean/std of delta-observation and reward targets from THIS transition list."""
    d = build_inputs(transitions)
    obs_std = d["delta"].std(dim=0, unbiased=False).clamp_min(1e-8)
    reward_std = d["reward"].std(dim=0, unbiased=False).clamp_min(1e-8) \
        if d["reward"].numel() > 1 else torch.ones(1)
    return {
        "obs_mean": d["delta"].mean(dim=0),
        "obs_std": obs_std,
        "reward_mean": d["reward"].mean(),
        "reward_std": reward_std,
    }


def to_json(norm: Dict[str, torch.Tensor]) -> Dict[str, List[float]]:
    return {
        "obs_mean": norm["obs_mean"].tolist(),
        "obs_std": norm["obs_std"].tolist(),
        "reward_mean": float(norm["reward_mean"]),
        "reward_std": float(norm["reward_std"]),
    }


def from_json(d: Dict) -> Dict[str, torch.Tensor]:
    return {
        "obs_mean": torch.tensor(d["obs_mean"], dtype=torch.float32),
        "obs_std": torch.tensor(d["obs_std"], dtype=torch.float32),
        "reward_mean": torch.tensor(float(d["reward_mean"]), dtype=torch.float32),
        "reward_std": torch.tensor(float(d["reward_std"]), dtype=torch.float32),
    }