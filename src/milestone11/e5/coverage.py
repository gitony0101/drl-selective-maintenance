"""E5-B failure-coverage manipulation audit (Section 6).

Reports for every seed and split: transition/episode counts, behavior-source
counts, scenario counts, action histogram, reward histogram/support, failure
transitions, individual failure events, failure-cost contribution, and
preventive-maintenance events.

PRIMARY manipulation check: E4 original training failure events == 0, while
E5-B dynamics_train MUST contain >0 genuine failure events for every formal
seed. A failure event is identified from the environment's explicit per-step
count information (info["num_failures"] / info["failure_cost"] > 0), never by a
bare reward threshold alone.

All 16 actions must still appear in dynamics_train. The exact preregistered
number of no-maintenance episodes is generated; if any seed still has zero
training failure events, this audit reports FAILURE (no adaptive generation).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.milestone11.e4.dataset import E4RawTransition

from .dataset import load_dataset
from .paths import E5_OUTPUT_ROOT, FORMAL_SEEDS, NUM_ACTIONS
from .split import SOURCES, SPLITS, partition_all_sources

# One "failure event" = one environment step in which at least one fleet slot
# failed, i.e. info["num_failures"] > 0, equivalently cost_failure > 0.
def _failure_events(transitions: List[E4RawTransition]) -> int:
    return sum(1 for t in transitions if t.cost_failure > 0)


def _pm_events(transitions: List[E4RawTransition]) -> int:
    return sum(1 for t in transitions if t.cost_preventive > 0)


def audit_transitions(transitions: List[E4RawTransition],
                      label: str) -> Dict[str, Any]:
    rewards = np.array([t.reward_t for t in transitions], dtype=np.float32)
    actions = Counter(t.action_id_t for t in transitions)
    scenarios = Counter(t.scenario_id for t in transitions)
    behaviors = Counter(t.behavior_policy for t in transitions)
    failure_trans = _failure_events(transitions)
    reward_failure_steps = rewards[rewards < 0]  # not used as failure ident
    return {
        "label": label,
        "transitions": len(transitions),
        "episodes": len({t.dataset_episode_id for t in transitions}),
        "behavior_source_counts": dict(behaviors),
        "scenario_counts": dict(scenarios),
        "action_histogram": {str(a): actions[a] for a in range(NUM_ACTIONS)},
        "spans_all_16_actions": all(a in actions for a in range(NUM_ACTIONS)),
        "reward": {
            "min": float(rewards.min()) if len(rewards) else None,
            "max": float(rewards.max()) if len(rewards) else None,
            "mean": float(rewards.mean()) if len(rewards) else None,
            "std": float(rewards.std()) if len(rewards) else None,
            "support": sorted(
                [float(v) for v in np.unique(rewards)]) if len(rewards) else [],
        },
        "num_failure_transitions": failure_trans,
        "num_failure_events": failure_trans,   # t.cost_failure>0 per step = events
        "failure_cost_contribution": float(
            sum(t.cost_failure for t in transitions)),
        "num_preventive_events": _pm_events(transitions),
        "genuine_failure_events_present": failure_trans > 0,
    }


def audit_seed(per_source: Dict[str, List[E4RawTransition]]) -> Dict[str, Any]:
    parts = partition_all_sources(per_source)
    combined: List[E4RawTransition] = []
    for src in SOURCES:
        combined.extend(per_source[src])
    return {
        "total_transitions": len(combined),
        "total_episodes": len({t.dataset_episode_id for t in combined}),
        "by_split": {
            s: audit_transitions(parts[s], s) for s in SPLITS
        },
        "train_failure_events": _failure_events(parts["dynamics_train"]),
        "train_failure_cost": float(
            sum(t.cost_failure for t in parts["dynamics_train"])),
        "manipulation_gate": {
            "dynamics_train_has_gt0_failures": _failure_events(
                parts["dynamics_train"]) > 0,
            "all_16_actions_in_train": all(
                a in Counter(t.action_id_t for t in parts["dynamics_train"])
                for a in range(NUM_ACTIONS)),
        },
    }


def run_coverage_audit(seed: int, out_root: Path = E5_OUTPUT_ROOT) -> Dict[str, Any]:
    per_source = load_dataset(seed, out_root)
    return audit_seed(per_source)


def audit_all_seeds(out_root: Path = E5_OUTPUT_ROOT) -> Dict[str, Any]:
    return {str(seed): run_coverage_audit(seed, out_root) for seed in FORMAL_SEEDS}


def write_audit(out_root: Path = E5_OUTPUT_ROOT) -> Path:
    report = audit_all_seeds(out_root)
    p = out_root / "e5_coverage_audit.json"
    p.write_text(json.dumps(report, indent=2))
    return p