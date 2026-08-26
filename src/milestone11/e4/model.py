"""E4 (M11) learned dynamics ensemble (Task 5).

A fixed 3-member MLP ensemble predicting (delta_observation[10], reward[1])
from concatenated (observation[10], one_hot(action)[16]) input x_t[26]:

    x_t[26] -> Linear(128) -> ReLU -> Linear(128) -> ReLU
    -> Linear(11): [10 delta-observation, 1 reward]

No recurrent memory, no hidden state, no trajectory/cycle pointer. Each member
owns its own deterministic initialization seed. Forward pass is gradient-free
for planning (never backpropagated through in MPC).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

from .paths import (
    HIDDEN_DIMS,
    INPUT_DIM,
    OUTPUT_DIM,
    ENSEMBLE_SIZE,
)


class DynamicsNetwork(nn.Module):
    """One dynamics model f_phi_m: x_t -> (delta_o_{t+1}, r_t)."""

    def __init__(self, input_dim: int = INPUT_DIM,
                 hidden_dims: Tuple[int, int] = HIDDEN_DIMS,
                 output_dim: int = OUTPUT_DIM,
                 init_seed: int = 0) -> None:
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        # Deterministic init (independent of global RNG state).
        torch.manual_seed(init_seed)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward: [B,26] -> (pred_delta_obs[B,10], pred_reward[B])."""
        if x.ndim != 2 or x.shape[1] != INPUT_DIM:
            raise ValueError(f"expected [B,{INPUT_DIM}], got {tuple(x.shape)}")
        out = self.net(x)
        delta = out[:, :10]
        reward = out[:, 10]
        return delta, reward


@dataclass
class DynamicsEnsemble:
    """The fixed M-member ensemble plus its normalization statistics."""

    members: nn.ModuleList
    normalization: Dict[str, torch.Tensor]  # obs_mean/obs_std/reward_mean/reward_std

    def __init__(self, members: nn.ModuleList, normalization: Dict[str, torch.Tensor]):
        self.members = members
        self.normalization = normalization

    @property
    def M(self) -> int:
        return len(self.members)

    def predict(self, x: torch.Tensor, denorm_delta: bool = True,
                denorm_reward: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ensemble-mean prediction across members.

        x: [B,26]. Returns (delta_obs[B,10], reward[B]) in raw (unnormalized)
        units after applying the member-normalization de-standardization.
        """
        normal = self._norm_on(x.device)
        deltas = []
        rewards = []
        for m in self.members:
            d, r = m(x)
            deltas.append(d)
            rewards.append(r)
        mean_d = torch.stack(deltas, dim=0).mean(0)   # [B,10] normalized delta
        mean_r = torch.stack(rewards, dim=0).mean(0)  # [B]   normalized reward

        delta_raw = mean_d * normal["obs_std"]
        reward_raw = (mean_r * normal["reward_std"] + normal["reward_mean"])
        if not denorm_delta:
            delta_raw = mean_d
        if not denorm_reward:
            reward_raw = mean_r
        return delta_raw, reward_raw

    def predict_member(self, m: int, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-member prediction (unnormalized), for disagreement diag."""
        normal = self._norm_on(x.device)
        d, r = self.members[m](x)
        delta_raw = d * normal["obs_std"]
        reward_raw = (r * normal["reward_std"] + normal["reward_mean"])
        return delta_raw, reward_raw

    def _norm_on(self, device) -> Dict[str, torch.Tensor]:
        """Normalization statistics moved to ``device`` (cached-free, cheap)."""
        return {k: v.to(device) if torch.is_tensor(v) else v
                for k, v in self.normalization.items()}

    def eval(self) -> "DynamicsEnsemble":
        for m in self.members:
            m.eval()
        return self


def build_ensemble(member_init_seeds: Tuple[int, ...],
                   normalization: Dict[str, torch.Tensor],
                   input_dim: int = INPUT_DIM) -> DynamicsEnsemble:
    members = nn.ModuleList(
        [DynamicsNetwork(input_dim=input_dim, init_seed=s) for s in member_init_seeds]
    )
    return DynamicsEnsemble(members, normalization)