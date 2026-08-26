"""E4 (M11) data coverage + integrity audit (Task 4).

Reports per-source transition/episode/scenario counts, the 16-action histogram
(overall and per source), observation min/max/mean/std/quantiles, reward
distribution, failure/maintenance event counts, and held-out coverage of the
training feature ranges. Enforces the requirement that all 16 actions appear in
dynamics_train. Diagnostic only — never used to change the dataset except to
flag a simple generation bug.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .dataset import E4RawTransition, load_dataset
from .paths import (
    E4_OUTPUT_ROOT,
    FORMAL_SEEDS,
    NUM_ACTIONS,
    OBSERVATION_DIM,
)
from .split import SOURCES, SPLITS, partition_episodes

EVENT_COST_FIELDS = {
    "preventive": "cost_preventive",
    "failure": "cost_failure",
    "wasted_life": "cost_wasted_life",
}


def _obs_matrix(transitions: List[E4RawTransition]) -> np.ndarray:
    return np.array([t.observation_t for t in transitions], dtype=np.float32)


def audit_source(source_transitions: List[E4RawTransition],
                 source_label: str) -> Dict[str, Any]:
    n_tr = len(source_transitions)
    episodes = Counter(t.dataset_episode_id for t in source_transitions)
    scenarios = Counter(t.scenario_id for t in source_transitions)
    actions = Counter(t.action_id_t for t in source_transitions)
    obs = _obs_matrix(source_transitions)
    rewards = np.array([t.reward_t for t in source_transitions], dtype=np.float32)

    event_counts = {"preventive": 0, "failure": 0, "wasted_life": 0}
    for t in source_transitions:
        if t.cost_preventive != 0:
            event_counts["preventive"] += 1
        if t.cost_failure != 0:
            event_counts["failure"] += 1
        if t.cost_wasted_life != 0:
            event_counts["wasted_life"] += 1

    return {
        "source": source_label,
        "transitions": n_tr,
        "episodes": len(episodes),
        "scenarios": dict(scenarios),
        "action_histogram": {str(a): actions[a] for a in range(NUM_ACTIONS)},
        "actions_finite_all": all(a in actions for a in range(NUM_ACTIONS)),
        "obs": {
            "min": float(obs.min()), "max": float(obs.max()),
            "mean": float(obs.mean()), "std": float(obs.std()),
            "q01": float(np.quantile(obs, 0.01)), "q99": float(np.quantile(obs, 0.99)),
            "per_dim_min": [float(v) for v in obs.min(0)],
            "per_dim_max": [float(v) for v in obs.max(0)],
        },
        "reward": {
            "min": float(rewards.min()), "max": float(rewards.max()),
            "mean": float(rewards.mean()), "std": float(rewards.std()),
            "n_zero": int(np.count_nonzero(rewards == 0)),
        },
        "event_counts": event_counts,
    }


def audit_seed(per_source: Dict[str, List[E4RawTransition]]) -> Dict[str, Any]:
    report: Dict[str, Any] = {"per_source": {}, "overall": {}}
    combined: List[E4RawTransition] = []
    for src in SOURCES:
        report["per_source"][src] = audit_source(per_source[src], src)
        combined.extend(per_source[src])

    parts = {s: [] for s in SPLITS}
    for src in SOURCES:
        for split, trans in partition_episodes(per_source[src]).items():
            parts[split].extend(trans)

    train_actions = Counter(t.action_id_t for t in parts["dynamics_train"])
    report["overall"] = {
        "transitions": len(combined),
        "episodes": len({t.dataset_episode_id for t in combined}),
        "spans_16_actions": all(a in train_actions for a in range(NUM_ACTIONS)),
        "train_actions": {str(a): train_actions[a] for a in range(NUM_ACTIONS)},
        "split_transition_counts": {s: len(v) for s, v in parts.items()},
    }
    # Held-out OOR (Section 9): held-out observations outside training ranges.
    train_obs = _obs_matrix(parts["dynamics_train"])
    hold_obs = _obs_matrix(parts["dynamics_holdout"])
    lo, hi = train_obs.min(0), train_obs.max(0)
    oob = np.any((hold_obs < lo) | (hold_obs > hi), axis=1)
    report["heldout_oor"] = {
        "n_total": int(len(hold_obs)),
        "n_out_of_train_range": int(oob.sum()),
        "frac_out_of_train_range": float(oob.mean()),
        "per_dim_train_min": [float(v) for v in lo],
        "per_dim_train_max": [float(v) for v in hi],
    }
    return report


def audit_all_seeds(out_root: Path = E4_OUTPUT_ROOT) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for seed in FORMAL_SEEDS:
        per_source = load_dataset(seed, out_root)
        result[str(seed)] = audit_seed(per_source)
    return result


def write_audit(out_root: Path = E4_OUTPUT_ROOT) -> Path:
    report = audit_all_seeds(out_root)
    p = out_root / "coverage_audit.json"
    p.write_text(json.dumps(report, indent=2))
    return p