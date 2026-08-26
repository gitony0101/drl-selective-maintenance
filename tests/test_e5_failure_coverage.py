"""E5 (M12) failure-coverage ablation unit tests.

Focused on the E5-specific logic that differs from E4: the 4-source
failure-enriched dataset contract, the episode-level 80/10/10 split (unequal
per-source episode counts), and the failure-coverage manipulation check.

The MPC planner and its unit tests are reused VERBATIM from E4
(tests/test_e4_mpc.py); E5 does not duplicate the planner.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.milestone11.e5.paths import (
    EPISODES_PER_SOURCE,
    SPLIT_EPISODES_PER_SOURCE,
    SPLIT_EPISODE_TARGETS,
    TOTAL_EPISODES_PER_SEED,
    TOTAL_TRANSITIONS_PER_SEED,
)
from src.milestone11.e5.split import (
    SOURCES,
    SPLITS,
    partition_episodes,
)


def _mk_trans(seed, source, ep_no, nsteps=100):
    """Build a synthetic episode of E4RawTransition-like objects."""
    from src.milestone11.e5.dataset import gen_deterministic_reset_seed

    scenario = f"m5_pilot_k2_s{(ep_no % 5) + 1:03d}"
    eid = f"{source}_seed{seed}_ep{ep_no:03d}_{scenario}"
    rseed = gen_deterministic_reset_seed(source, seed, ep_no, ep_no % 5)
    rows = []
    for st in range(nsteps):
        from src.milestone11.e4.dataset import E4RawTransition

        # Failures appear only in no_maintenance episodes, deterministically.
        fail = 1.0 if (source == "no_maintenance" and st % 10 == 9) else 0.0
        # random_feasible spans all 16 actions (coverage source); others use a
        # small deterministic action subset that nevertheless couples with obs.
        if source == "random_feasible":
            act = st % 16
        else:
            act = 0
        rows.append(E4RawTransition(
            dataset_episode_id=eid, scenario_id=scenario, behavior_policy=source,
            environment_reset_seed=rseed, step_index=st,
            observation_t=[0.5] * 10, action_id_t=act,
            reward_t=float(-fail * 5.0 - (st % 2)),
            next_observation_t=[0.5] * 10, terminated_t=False, truncated_t=False,
            cost_preventive=float(1.0 if (st % 2 == 0 and source != "no_maintenance") else 0.0),
            cost_failure=fail * 5.0, cost_wasted_life=0.0, cost_total=0.0,
        ))
    return rows


def _source_episodes(seed, source, n_ep):
    ep = {}
    for i in range(n_ep):
        for t in _mk_trans(seed, source, i):
            ep.setdefault(t.dataset_episode_id, []).append(t)
    return [e for g in ep.values() for e in g]


def test_dataset_contract_totals():
    assert sum(EPISODES_PER_SOURCE.values()) == TOTAL_EPISODES_PER_SEED == 150
    assert sum(EPISODES_PER_SOURCE.values()) * 100 == TOTAL_TRANSITIONS_PER_SEED == 15000
    assert set(SOURCES) == {"random_feasible", "exact_myopic", "h2", "no_maintenance"}
    assert EPISODES_PER_SOURCE["no_maintenance"] == 30


def test_split_targets_sum():
    assert sum(SPLIT_EPISODE_TARGETS.values()) == TOTAL_EPISODES_PER_SEED
    for src, (t, v, h) in SPLIT_EPISODES_PER_SOURCE.items():
        assert t + v + h == EPISODES_PER_SOURCE[src], src


def test_partition_no_cross_episode_and_totals():
    seed = 7
    all_ok = {}
    for src in SOURCES:
        epilist = _source_episodes(seed, src, EPISODES_PER_SOURCE[src])
        parts = partition_episodes(epilist, src)
        n_tr = len(parts["dynamics_train"])
        n_va = len(parts["dynamics_validate"])
        n_hd = len(parts["dynamics_holdout"])
        exp_t, exp_v, exp_h = SPLIT_EPISODES_PER_SOURCE[src]
        assert n_tr == exp_t * 100
        assert n_va == exp_v * 100
        assert n_hd == exp_h * 100
        # every episode is entirely in exactly one split, with all 100 steps
        for split in SPLITS:
            from collections import Counter

            per_ep = Counter(t.dataset_episode_id for t in parts[split])
            assert all(c == 100 for c in per_ep.values()), split
            eps_in_split = set(per_ep)
            for other in SPLITS:
                if other != split:
                    other_eps = {t.dataset_episode_id for t in parts[other]}
                    assert not (eps_in_split & other_eps), (split, other)
        all_ok[src] = True
    assert all(all_ok.values())


def test_partition_all_sources_present_in_train():
    from src.milestone11.e5.split import partition_all_sources

    seed = 11
    per_src = {}
    for src in SOURCES:
        per_src[src] = _source_episodes(seed, src, EPISODES_PER_SOURCE[src])
    parts = partition_all_sources(per_src)
    assert len(parts["dynamics_train"]) == 12000
    assert len(parts["dynamics_validate"]) == 1500
    assert len(parts["dynamics_holdout"]) == 1500
    # all four behaviors in train
    train_behav = {t.behavior_policy for t in parts["dynamics_train"]}
    assert train_behav == set(SOURCES)


def test_no_maintenance_represented_in_all_splits():
    from src.milestone11.e5.split import partition_all_sources

    seed = 13
    per_src = {}
    for src in SOURCES:
        per_src[src] = _source_episodes(seed, src, EPISODES_PER_SOURCE[src])
    parts = partition_all_sources(per_src)
    for split in SPLITS:
        nm = {t.dataset_episode_id for t in parts[split]
              if t.behavior_policy == "no_maintenance"}
        assert len(nm) > 0, split


def test_failure_coverage_manipulation_gate():
    """no-maintenance adds genuine failure transitions; original sources have 0."""
    from src.milestone11.e5.coverage import audit_seed

    seed = 17
    per_src = {}
    for src in SOURCES:
        per_src[src] = _source_episodes(seed, src, EPISODES_PER_SOURCE[src])
    aud = audit_seed(per_src)
    gate = aud["manipulation_gate"]
    assert gate["dynamics_train_has_gt0_failures"] is True
    assert gate["all_16_actions_in_train"] is True
    # In the synthetic fixture only no_maintenance produces failures, but the
    # train split must see >0 genuine failure events.
    assert aud["train_failure_events"] > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))