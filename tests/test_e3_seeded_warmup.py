"""E3 (M10) seeded-warmup manifest tests (Section 16/17 + streaming buffer)."""

from __future__ import annotations

import numpy as np
import pytest

from src.milestone10.e3.nstep import EpisodeNStepBuffer, RawStep
from src.milestone10.e3.seeded_warmup import (
    SEEDED_EXPL_COUNT,
    SEEDED_H2_COUNT,
    SEEDED_M4_COUNT,
    SEEDED_WARMUP_TOTAL,
    WarmupRawTransition,
    build_seeded_warmup_frames,
)

pytestmark = pytest.mark.requires_external_assets


def _fr(source_policy="exact_myopic", i=0):
    return WarmupRawTransition(
        source_policy=source_policy, scenario_id=f"sc{i % 5 + 1}",
        reset_seed=6521 + i, episode_id=f"ep{i}", step_index=i % 100,
        training_seed=6521, observation_t=list(np.zeros(10, dtype=float)),
        action_id_t=0, reward_t=0.0, next_observation_t=list(np.zeros(10, dtype=float)),
        terminated_t=False, truncated_t=False, cost_preventive=0.0,
        cost_failure=0.0, cost_wasted_life=0.0, cost_total=0.0)


def test_frozen_mixture_constants():
    assert SEEDED_M4_COUNT + SEEDED_H2_COUNT + SEEDED_EXPL_COUNT == SEEDED_WARMUP_TOTAL


def test_build_seeded_warmup_frames_counts_and_integrity():
    frames = build_seeded_warmup_frames(6521)
    assert len(frames) == SEEDED_WARMUP_TOTAL
    counts = {}
    for f in frames:
        counts[f.source_policy] = counts.get(f.source_policy, 0) + 1
    assert counts["exact_myopic"] == SEEDED_M4_COUNT
    assert counts["h2"] == SEEDED_H2_COUNT
    assert counts["exploration"] == SEEDED_EXPL_COUNT
    # every frame: obs/next_obs dim 10, finite reward, action in K=2 range
    for f in frames:
        assert len(f.observation_t) == 10 and len(f.next_observation_t) == 10
        assert 0 <= f.action_id_t < 16
        assert np.isfinite(f.reward_t)
    # successor integrity within each episode
    eps = {}
    for f in frames:
        eps.setdefault(f.episode_id, []).append(f)
    for ep, trs in eps.items():
        ordered = sorted(trs, key=lambda t: t.step_index)
        for i in range(len(ordered) - 1):
            assert np.allclose(ordered[i].next_observation_t,
                               ordered[i + 1].observation_t, atol=1e-5)


def test_streaming_buffer_offline_equivalence_narrow():
    # terminal/truncated embedded across an episode
    raw = [
        RawStep(np.zeros(10, dtype=np.float32), 0, 1.0, np.ones(10, dtype=np.float32), False, False),
        RawStep(np.ones(10, dtype=np.float32), 0, 2.0, np.full(10, 2.0, dtype=np.float32), False, False),
        RawStep(np.full(10, 2.0, dtype=np.float32), 0, 3.0, np.full(10, 9.0, dtype=np.float32), True, False),
    ]
    from src.milestone10.e3.nstep import build_nstep_transitions
    offline = build_nstep_transitions(raw, 3, 0.9)
    sb = EpisodeNStepBuffer(3, 0.9)
    streamed = []
    for s in raw:
        streamed.extend(sb.push(s))
    streamed.extend(sb.flush_episode())
    assert len(streamed) == len(offline)
    for a, b in zip(streamed, offline):
        assert a.bootstrap_steps == b.bootstrap_steps
        assert a.discounted_reward_sum == pytest.approx(b.discounted_reward_sum, abs=1e-9)
        assert a.terminated_for_bootstrap == b.terminated_for_bootstrap