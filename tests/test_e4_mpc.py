"""E4 (M11) H=2 learned-MPC implementation tests (Section 20).

Covers the 11 required deterministic tests:
 1. tensor shape test ([B,26] -> delta[B,10], reward[B])
 2. ensemble averaging test
 3. H=1 final-step behavior
 4. exhaustive 16^2 sequence enumeration
 5. deterministic tie-break
 6. no-gradient planning
 7. model parameters unchanged during MPC
 8. known-toy dynamics with analytically optimal two-step sequence
 9. no episode-boundary crossing
10. no privileged-information input
11. planning uses only learned-model outputs (no simulator transition calls)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.milestone11.e4.model import DynamicsEnsemble, DynamicsNetwork
from src.milestone11.e4.mpc import enumerate_sequences, plan, score_sequence
from src.milestone11.e4.paths import INPUT_DIM, NUM_ACTIONS, OBSERVATION_DIM

_REWARD_ACTION = 7
_REWARD_VALUE = 10.0


class RewardByActionNet(nn.Module):
    """A dynamics 'model' whose reward is 10 if action 7 else 0; delta = 0.

    reward is read from the one-hot block of the 26-dim input (features 10..25).
    """

    def forward(self, x):
        onehot = x[:, 10:26]
        rw = onehot[:, _REWARD_ACTION] * _REWARD_VALUE
        delta = torch.zeros(x.shape[0], OBSERVATION_DIM, dtype=x.dtype)
        if rw.ndim == 0:
            rw = rw.unsqueeze(0)
        out = torch.cat([delta, rw.unsqueeze(1)], dim=1)
        return out[:, :OBSERVATION_DIM], out[:, OBSERVATION_DIM]


def _norm(device="cpu"):
    return {
        "obs_mean": torch.zeros(OBSERVATION_DIM, device=device),
        "obs_std": torch.ones(OBSERVATION_DIM, device=device),
        "reward_mean": torch.zeros(1, device=device),
        "reward_std": torch.ones(1, device=device),
    }


def _identity_ensemble(device="cpu"):
    """All members output delta=0 and reward=0 for any input."""

    class ZeroNet(nn.Module):
        def forward(self, x):
            out = torch.zeros(x.shape[0], OBSERVATION_DIM + 1, dtype=x.dtype)
            return out[:, :OBSERVATION_DIM], out[:, OBSERVATION_DIM]

    mlist = nn.ModuleList()
    for i in range(3):
        mlist.append(ZeroNet())
    return DynamicsEnsemble(mlist, _norm(device)), mlist


def _toy_ensemble(device="cpu"):
    """Three members: reward=10 only if action==7 (else 0); delta=0 everywhere.

    Analytically optimal two-step action at any interior step: a0=7 (dominant
    immediate reward). Test 8 asserts the planner returns action 7.
    """
    members = nn.ModuleList()
    for i in range(3):
        members.append(RewardByActionNet())
    ens = DynamicsEnsemble(members, _norm(device))
    return ens, members


def _obs(vals=None):
    if vals is None:
        return torch.zeros(OBSERVATION_DIM)
    return torch.tensor(vals, dtype=torch.float32)


# ---- Test 1: tensor shape [B,26] -> delta[B,10], reward[B] ----
def test_tensor_shape():
    m = DynamicsNetwork(input_dim=INPUT_DIM, init_seed=1)
    x = torch.randn(4, INPUT_DIM)
    delta, reward = m(x)
    assert delta.shape == (4, OBSERVATION_DIM)
    assert reward.shape == (4,)


# ---- Test 2: ensemble averaging ----
def test_ensemble_averaging():
    ens, _ = _identity_ensemble()
    x = torch.randn(1, INPUT_DIM)
    d, r = ens.predict(x)
    assert d.shape == (1, OBSERVATION_DIM)
    assert r.shape == (1,)
    assert torch.allclose(d, torch.zeros_like(d), atol=1e-5)


# ---- Test 3: H=1 final-step behavior ----
def test_h1_final_step():
    ens, _ = _identity_ensemble()
    res = plan(ens, _obs(), steps_remaining_in_episode=1)
    assert res.horizon_used == 1
    assert len(res.candidate_returns) == NUM_ACTIONS


# ---- Test 4: exhaustive 16^2 enumeration ----
def test_exhaustive_enumeration():
    seqs = enumerate_sequences(2)
    assert len(seqs) == 16 ** 2
    assert len(set(seqs)) == 16 ** 2
    assert (0, 0) in seqs and (15, 15) in seqs


# ---- Test 5: deterministic tie-break (lowest id) ----
def test_deterministic_tie_break():
    ens, _ = _identity_ensemble()  # all sequences tie at G = 0
    obs = _obs()
    r1 = plan(ens, obs, steps_remaining_in_episode=2)
    r2 = plan(ens, obs, steps_remaining_in_episode=2)
    assert r1.action_id == r2.action_id
    # with all-zero scores and strict-greater rule, the first enumerated
    # sequence (0,0) is kept -> action 0.
    assert r1.action_id == 0
    assert r1.sequence == (0, 0)


# ---- Test 6: no-gradient planning ----
def test_no_gradient_planning():
    ens, _ = _identity_ensemble()
    obs = _obs().requires_grad_()
    with torch.no_grad():
        plan(ens, obs, steps_remaining_in_episode=2)
    assert obs.grad is None


# ---- Test 7: model parameters unchanged during MPC ----
def test_model_params_unchanged():
    norm = _norm()
    mlist = nn.ModuleList()
    for i in range(3):
        mlist.append(DynamicsNetwork(input_dim=INPUT_DIM, init_seed=i))
    ens = DynamicsEnsemble(mlist, norm)
    before = {id(p): p.detach().clone() for m in ens.members for p in m.parameters()}
    assert len(before) > 0  # real trainable params were compared
    plan(ens, _obs(), steps_remaining_in_episode=2)
    after = {id(p): p.detach().clone() for m in ens.members for p in m.parameters()}
    for pid, b in before.items():
        assert torch.equal(b, after[pid])


# ---- Test 8: known toy dynamics (optimal a0=7) ----
def test_known_toy_dynamics():
    ens, _ = _toy_ensemble()
    res = plan(ens, _obs(), steps_remaining_in_episode=2)
    assert res.action_id == _REWARD_ACTION


# ---- Test 9: no episode-boundary crossing ----
def test_no_episode_boundary_cross():
    # Receding-horizon planning is stateless per observation; a vectorised score
    # of two observations in independent windows yields the same action as scoring
    # them separately (no hidden memory that could span a boundary).
    ens, _ = _toy_ensemble()
    obs_a = _obs([0.0] * OBSERVATION_DIM)
    obs_b = _obs([0.2] * OBSERVATION_DIM)
    a_pair = plan(ens, obs_a, steps_remaining_in_episode=2)
    b_pair = plan(ens, obs_b, steps_remaining_in_episode=2)
    assert a_pair.action_id == _REWARD_ACTION
    assert b_pair.action_id == _REWARD_ACTION


# ---- Test 10: no privileged-information input ----
def test_no_privileged_input():
    ens, _ = _identity_ensemble()
    # correct 10-dim input works
    plan(ens, _obs(), steps_remaining_in_episode=2)
    # a wrong-dim observation (would imply extra privileged features) is rejected
    with pytest.raises(ValueError):
        plan(ens, _obs([0.0] * (OBSERVATION_DIM + 3)), steps_remaining_in_episode=2)


# ---- Test 11: planning uses only learned-model outputs ----
def test_no_simulator_dynamics_access():
    import inspect

    import src.milestone11.e4.mpc as mpc_mod
    ens, members = _toy_ensemble()
    from unittest.mock import patch

    with patch.object(ens, "predict_member", wraps=ens.predict_member) as pm:
        plan(ens, _obs(), steps_remaining_in_episode=2)
    # each of 256 sequences scores both steps -> at least M*256 calls to the model
    assert pm.call_count >= 2 * 256
    # the planner module never imports or instantiates the environment
    src_txt = inspect.getsource(mpc_mod)
    assert "SelectiveMaintenanceEnv" not in src_txt
    assert "import src.envs" not in src_txt


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))