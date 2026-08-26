"""E3 (M10) deterministic n-step pipeline unit tests (Section 14).

Includes the HARD GATE: n=1 exact regression equivalence to the canonical
one-step DDQN, and the frozen arithmetic check R0 = 1 + 0.9*2 + 0.9^2*3 = 5.23.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.milestone10.e3.nstep import (
    NStepReplayBuffer,
    NStepTransition,
    RawStep,
    build_nstep_transitions,
)
from src.milestone10.e3.agent_update import (
    compute_nstep_td_target,
    update_nstep,
)
from src.agents.ddqn.agent import DDQNAgent, DDQNAgentConfig


def _obs(v: float) -> np.ndarray:
    return np.full(10, v, dtype=np.float32)


def _step(reward, terminated=False, truncated=False, obs_v=None, next_v=None) -> RawStep:
    return RawStep(
        observation=_obs(obs_v if obs_v is not None else 0.0),
        action_id=0,
        reward=reward,
        next_observation=_obs(next_v if next_v is not None else 0.0),
        terminated=terminated,
        truncated=truncated,
    )


# ---------------------------------------------------------------------------
# Test 1 (HARD GATE): n=1 exact regression equivalence
# ---------------------------------------------------------------------------

def test_n1_single_transition_equivalence():
    st = _step(reward=2.5, terminated=False, obs_v=1.0, next_v=2.0)
    out = build_nstep_transitions([st], n=1, gamma=0.95)
    assert len(out) == 1
    e = out[0]
    assert e.bootstrap_steps == 1
    assert e.discounted_reward_sum == pytest.approx(2.5)
    assert np.allclose(e.observation, np.full(10, 1.0))
    assert np.allclose(e.bootstrap_observation, np.full(10, 2.0))
    assert e.terminated_for_bootstrap is False
    assert e.truncated_boundary is False
    assert np.allclose(e.observation, st.observation)


def test_n1_preserves_every_raw_transition():
    steps = [_step(reward=float(i), terminated=False) for i in range(7)]
    out = build_nstep_transitions(steps, n=1, gamma=0.95)
    assert len(out) == len(steps)
    for e, st in zip(out, steps):
        assert e.bootstrap_steps == 1
        assert e.discounted_reward_sum == pytest.approx(st.reward)
        assert e.terminated_for_bootstrap == st.terminated
        assert e.truncated_boundary == st.truncated
        assert np.allclose(e.bootstrap_observation, st.next_observation)


def test_n1_preserves_terminated_flags():
    steps = [_step(reward=1.0, terminated=(i == 3)) for i in range(5)]
    out = build_nstep_transitions(steps, n=1, gamma=0.95)
    for i, e in enumerate(out):
        assert e.terminated_for_bootstrap == (i == 3)


def _random_batch(n: int) -> dict:
    return {
        "observation": np.random.RandomState(0).randn(n, 10).astype(np.float32),
        "action": np.random.RandomState(1).randint(0, 16, size=n).astype(np.int64),
        "discounted_reward_sum": np.asarray(
            [float(j) for j in range(n)], dtype=np.float32),
        "bootstrap_observation": np.random.RandomState(2).randn(n, 10).astype(np.float32),
        "bootstrap_steps": np.ones(n, dtype=np.int64),
        "terminated_for_bootstrap": np.zeros(n, dtype=bool),
        "truncated_boundary": np.zeros(n, dtype=bool),
    }


def test_n1_target_equals_frozen_onestep_target():
    """HARD GATE: with k=1, reward_sum=r, bootstrap_obs=next_obs, the n-step
    target must numerically equal the canonical one-step Double DQN target."""
    agent = DDQNAgent(config=DDQNAgentConfig(), seed=123)
    n = 8
    batch = _random_batch(n)
    batch["next_observation"] = batch["bootstrap_observation"].copy()
    batch["reward"] = batch["discounted_reward_sum"].copy()
    batch["terminated"] = batch["terminated_for_bootstrap"].copy()

    nstep_target = compute_nstep_td_target(agent, batch)
    canonical_target = agent.compute_td_target(batch)

    assert nstep_target.shape == canonical_target.shape
    assert nstep_target.dtype == canonical_target.dtype
    assert torch.allclose(nstep_target, canonical_target, atol=1e-5), (
        "\nmax abs diff = "
        f"{float((nstep_target - canonical_target).abs().max())}"
    )
    # stop gradient: both are detached
    assert nstep_target.requires_grad is False


def test_n1_update_produces_metrics_and_updates_online_only():
    agent = DDQNAgent(config=DDQNAgentConfig(), seed=0)
    online_before = {k: v.clone() for k, v in agent.online_network.state_dict().items()}
    target_before = {k: v.clone() for k, v in agent.target_network.state_dict().items()}
    n = 32
    batch = _random_batch(n)
    upd = update_nstep(agent, batch)
    for key in ("td_loss", "td_error_mean", "q_values_mean", "grad_norm"):
        assert key in upd
        assert np.isfinite(upd[key])
    # target network must be unchanged by update
    for k, v in agent.target_network.state_dict().items():
        assert torch.equal(v, target_before[k]), f"target param {k} changed"
    # online network must have changed (nonzero grad -> at least one param moved)
    moved = any(
        not torch.equal(agent.online_network.state_dict()[k], online_before[k])
        for k in online_before
    )
    assert moved


# ---------------------------------------------------------------------------
# Test 2: frozen arithmetic check
# ---------------------------------------------------------------------------

def test_n3_discounted_return_arithmetic():
    """gamma=0.9, r0=1, r1=2, r2=3 => R0 = 1 + 0.9*2 + 0.9^2*3 = 5.23."""
    steps = [_step(reward=r) for r in (1.0, 2.0, 3.0)]
    out = build_nstep_transitions(steps, n=3, gamma=0.9)
    assert out[0].bootstrap_steps == 3
    assert out[0].discounted_reward_sum == pytest.approx(5.23, abs=1e-9)
    # sub-windows
    assert out[1].discounted_reward_sum == pytest.approx(2.0 + 0.9 * 3.0, abs=1e-9)
    assert out[1].bootstrap_steps == 2
    assert out[2].discounted_reward_sum == pytest.approx(3.0, abs=1e-9)
    assert out[2].bootstrap_steps == 1


# ---------------------------------------------------------------------------
# termination / truncation / episode boundary behavior
# ---------------------------------------------------------------------------

def test_normal_threestep_sequence():
    steps = [_step(reward=float(i), terminated=False) for i in range(6)]
    out = build_nstep_transitions(steps, n=3, gamma=0.95)
    for i in range(4):
        assert out[i].bootstrap_steps == 3
        expected = sum((0.95 ** k) * float(i + k) for k in range(3))
        assert out[i].discounted_reward_sum == pytest.approx(expected, abs=1e-9)


def test_termination_after_one_step_masks_bootstrap():
    steps = [_step(reward=1.0), _step(reward=10.0, terminated=True)]
    out = build_nstep_transitions(steps, n=3, gamma=0.95)
    # base step 0 is not terminal, but step 1 (t+1) is -> window stops at 2 steps,
    # terminated_for_bootstrap=True (bootstrap masked)
    assert out[0].bootstrap_steps == 2
    assert out[0].terminated_for_bootstrap is True
    assert out[0].discounted_reward_sum == pytest.approx(1.0 + 0.95 * 10.0, abs=1e-9)
    assert out[0].truncated_boundary is False
    # terminal step its own entry
    assert out[1].bootstrap_steps == 1
    assert out[1].terminated_for_bootstrap is True


def test_termination_after_two_steps():
    steps = [_step(reward=1.0), _step(reward=2.0), _step(reward=50.0, terminated=True)]
    out = build_nstep_transitions(steps, n=3, gamma=0.9)
    # base step 0 accumulates exactly 3 steps? No: step 2 is terminal -> stop there.
    assert out[0].bootstrap_steps == 3
    assert out[0].terminated_for_bootstrap is True
    assert out[0].discounted_reward_sum == pytest.approx(
        1.0 + 0.9 * 2.0 + 0.9 ** 2 * 50.0, abs=1e-9)


def test_truncation_before_three_steps_records_boundary_no_mask():
    steps = [_step(reward=1.0), _step(reward=2.0, truncated=True)]
    out = build_nstep_transitions(steps, n=3, gamma=0.9)
    # truncated ends sequence => window stops at 2 steps, bootstrap NOT masked
    assert out[0].bootstrap_steps == 2
    assert out[0].truncated_boundary is True
    assert out[0].terminated_for_bootstrap is False
    assert out[0].discounted_reward_sum == pytest.approx(1.0 + 0.9 * 2.0, abs=1e-9)
    # trailing truncated step: its own entry, still not masked
    assert out[1].bootstrap_steps == 1
    assert out[1].truncated_boundary is True
    assert out[1].terminated_for_bootstrap is False


def test_exact_episode_end_no_cross_episode():
    steps = [_step(reward=float(i), terminated=False) for i in range(3)]
    out = build_nstep_transitions(steps, n=5, gamma=0.95)
    assert len(out) == 3
    assert out[0].bootstrap_steps == 3  # episode end caps window at remaining steps
    assert np.allclose(out[0].bootstrap_observation, steps[2].next_observation)
    assert out[1].bootstrap_steps == 2
    assert out[2].bootstrap_steps == 1


def test_two_consecutive_episodes_no_contamination():
    ep1 = [_step(reward=1.0), _step(reward=2.0, truncated=True)]
    ep2 = [_step(reward=100.0), _step(reward=200.0, truncated=True)]
    o1 = build_nstep_transitions(ep1, n=3, gamma=0.95)
    o2 = build_nstep_transitions(ep2, n=3, gamma=0.95)
    # ep1 entry 0 must not reach into ep2's 100.0/200.0
    assert o1[0].bootstrap_steps == 2
    assert o1[0].discounted_reward_sum == pytest.approx(1.0 + 0.95 * 2.0, abs=1e-9)
    assert o1[0].truncated_boundary is True
    # ep2 fresh
    assert o2[0].bootstrap_steps == 2
    assert o2[0].discounted_reward_sum == pytest.approx(100.0 + 0.95 * 200.0, abs=1e-9)


# ---------------------------------------------------------------------------
# buffer / batch / tensor-shape / stop-gradient
# ---------------------------------------------------------------------------

def test_mixed_k_in_one_minibatch():
    agent = DDQNAgent(config=DDQNAgentConfig(), seed=0)
    n = 8
    batch = _random_batch(n)
    # mixed effective horizons 1..4
    ks = np.array([1, 2, 3, 4, 1, 1, 3, 2], dtype=np.int64)
    batch["bootstrap_steps"] = ks
    batch["terminated_for_bootstrap"] = np.zeros(n, dtype=bool)
    target = compute_nstep_td_target(agent, batch)
    assert target.shape == (n,)
    # manual check for a couple rows using greedy action on bootstrap obs
    dev = agent.device
    with torch.no_grad():
        q_online = agent.online_network(torch.as_tensor(batch["bootstrap_observation"], device=dev))
        acts = q_online.argmax(dim=1)
        q_target = agent.target_network(torch.as_tensor(batch["bootstrap_observation"], device=dev))
        q_sel = q_target.gather(1, acts.unsqueeze(1)).squeeze(1)
        expected = []
        for row in range(n):
            rt = batch["discounted_reward_sum"][row]
            gamma = agent.config.gamma
            expected.append(
                rt + (gamma ** ks[row]) * q_sel[row].item()
            )
        expected = np.asarray(expected, dtype=np.float32)
    assert np.allclose(target.cpu().numpy(), expected, atol=1e-4)


def test_n1_batch_target_equivalence():
    agent = DDQNAgent(config=DDQNAgentConfig(), seed=1)
    n = 16
    batch = _random_batch(n)
    batch["next_observation"] = batch["bootstrap_observation"].copy()
    batch["reward"] = batch["discounted_reward_sum"].copy()
    batch["terminated"] = batch["terminated_for_bootstrap"].copy()
    nt = compute_nstep_td_target(agent, batch)
    ct = agent.compute_td_target(batch)
    assert torch.allclose(nt, ct, atol=1e-5)


def test_tensor_shapes():
    agent = DDQNAgent(config=DDQNAgentConfig(), seed=2)
    n = 64
    batch = _random_batch(n)
    target = compute_nstep_td_target(agent, batch)
    assert target.shape == (n,)
    assert target.dtype == torch.float32
    # batch paths propagate the correct shapes
    buf = NStepReplayBuffer(capacity=100_000, seed=7)
    for s in range(6):
        buf.insert(
            NStepTransition(
                observation=np.full(10, float(s), dtype=np.float32),
                action_id=s % 3, discounted_reward_sum=float(s),
                bootstrap_observation=np.full(10, float(s + 1), dtype=np.float32),
                bootstrap_steps=1, terminated_for_bootstrap=False,
                truncated_boundary=False,
            ))
    sample = buf.sample_batch(4)
    assert sample["observation"].shape == (4, 10)
    assert sample["bootstrap_observation"].shape == (4, 10)
    for k in ("action", "discounted_reward_sum", "bootstrap_steps",
              "terminated_for_bootstrap", "truncated_boundary"):
        assert sample[k].shape == (4,)


def test_stop_gradient_no_grad_through_target():
    agent = DDQNAgent(config=DDQNAgentConfig(), seed=3)
    batch = _random_batch(8)
    target = compute_nstep_td_target(agent, batch)
    assert target.requires_grad is False
    # computing target must not accumulate into target network grads
    for p in agent.target_network.parameters():
        assert p.grad is None
        assert p.requires_grad is False


def test_buffer_roundtrip_state_dict():
    buf = NStepReplayBuffer(capacity=200, seed=5)
    for s in range(50):
        buf.insert(NStepTransition(
            observation=np.zeros(10, dtype=np.float32) + float(s),
            action_id=int(s % 16), discounted_reward_sum=float(s),
            bootstrap_observation=np.zeros(10, dtype=np.float32) + float(s + 1),
            bootstrap_steps=(s % 3) + 1, terminated_for_bootstrap=False,
            truncated_boundary=False))
    st = buf.state_dict()
    buf2 = NStepReplayBuffer(capacity=200, seed=5)
    buf2.load_state_dict(st)
    assert len(buf2) == 50
    a = buf.sample_batch(8)
    b = buf2.sample_batch(8)  # same RNG state restored
    assert np.allclose(a["observation"], b["observation"])
    assert np.allclose(a["discounted_reward_sum"], b["discounted_reward_sum"])


def test_capacity_ring_wrap():
    buf = NStepReplayBuffer(capacity=100, seed=9)
    for i in range(150):
        buf.insert(NStepTransition(
            observation=np.full(10, float(i), dtype=np.float32),
            action_id=i % 16, discounted_reward_sum=float(i),
            bootstrap_observation=np.full(10, float(i + 1), dtype=np.float32),
            bootstrap_steps=1, terminated_for_bootstrap=False,
            truncated_boundary=False))
    assert len(buf) == 100  # wrapped: oldest 50 overwritten
    assert buf.current_size == 100


# ---------------------------------------------------------------------------
# streaming EpisodeNStepBuffer equivalence
# ---------------------------------------------------------------------------

def test_streaming_matches_offline_episode_frames():
    from src.milestone10.e3.nstep import EpisodeNStepBuffer
    # one long episode, 47 steps, several terminated/truncated embedded
    steps = []
    for i in range(12):
        steps.append(_step(reward=float(i)))
    steps.append(_step(reward=100.0, truncated=True))   # episode 1 ends
    ep2 = []
    for i in range(20):
        ep2.append(_step(reward=float(1000 + i)))
    ep2.append(_step(reward=500.0, terminated=False, truncated=True))  # episode 2 ends
    all_episodes = [steps, ep2]

    for n in (1, 3):
        offline = []
        for ep in all_episodes:
            offline.extend(build_nstep_transitions(ep, n, gamma=0.95))

        streamed = []
        buf = EpisodeNStepBuffer(n=n, gamma=0.95)
        for ep in all_episodes:
            for s in ep:
                streamed.extend(buf.push(s))
            streamed.extend(buf.flush_episode())
        assert len(streamed) == len(offline), (n, len(streamed), len(offline))
        for a, b in zip(streamed, offline):
            assert a.bootstrap_steps == b.bootstrap_steps
            assert a.discounted_reward_sum == pytest.approx(b.discounted_reward_sum, abs=1e-9)
            assert a.terminated_for_bootstrap == b.terminated_for_bootstrap
            assert a.truncated_boundary == b.truncated_boundary
            assert np.allclose(a.observation, b.observation)
            assert np.allclose(a.bootstrap_observation, b.bootstrap_observation)


def test_stream_n1_emits_immediately():
    from src.milestone10.e3.nstep import EpisodeNStepBuffer
    buf = EpisodeNStepBuffer(n=1, gamma=0.95)
    out0 = buf.push(_step(reward=1.0, truncated=False))
    assert len(out0) == 1  # n=1 transition emitted on the same push
    assert out0[0].bootstrap_steps == 1
    assert out0[0].discounted_reward_sum == pytest.approx(1.0)