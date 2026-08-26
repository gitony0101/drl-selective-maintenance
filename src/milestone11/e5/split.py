"""E5-B episode-level train/validate/holdout split (Section 5).

Splits at the EPISODE level only — never at the transition level. For each
formal seed and behavior source, the collected (complete 100-step) episodes are
partitioned into:

    dynamics_train    : 120 episodes  (12000 transitions) total per seed
    dynamics_validate :  15 episodes  ( 1500 transitions)
    dynamics_holdout  :  15 episodes  ( 1500 transitions)

Per-source allocations (80/10/10, deterministic):

    random_feasible 50 -> 40/5/5   (per-scenario 8/1/1, E4-identical)
    exact_myopic    50 -> 40/5/5   (per-scenario 8/1/1, E4-identical)
    h2              20 -> 16/2/2
    no_maintenance  30 -> 24/3/3

Stratification is by behavior source AND scenario, as evenly as integer
constraints permit. All four behavior sources appear in dynamics_train, and
failure-inducing no-maintenance episodes appear in dynamics_train,
dynamics_validate, and dynamics_holdout. Reset seeds are disjoint across splits.
A frozen split manifest is written before any training consumes the data.
"""

from __future__ import annotations

import json
import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.milestone11.e4.dataset import E4RawTransition

from .dataset import load_dataset
from .paths import (
    E5_OUTPUT_ROOT,
    EPISODE_HORIZON,
    FORMAL_SEEDS,
    SPLIT_EPISODES_PER_SOURCE,
)

SOURCES = ("random_feasible", "exact_myopic", "h2", "no_maintenance")
SPLITS = ("dynamics_train", "dynamics_validate", "dynamics_holdout")

# Per-scenario per-source train/val/hold assignment templates. For each source
# and scenario, ``(n_train, n_val, n_hold)`` gives the split of that scenario's
# episodes. The templates were chosen so the per-source allocations sum to the
# fixed 80/10/10 targets and every scenario is represented in every split as
# evenly as integer constraints permit.
_PER_SCENARIO = {
    # 10 episodes/scenario -> 8 train, 1 val, 1 hold (E4-identical scheme).
    "random_feasible": {f"m5_pilot_k2_s{s:03d}": (8, 1, 1) for s in range(1, 6)},
    "exact_myopic": {f"m5_pilot_k2_s{s:03d}": (8, 1, 1) for s in range(1, 6)},
    # 4 episodes/scenario -> 3 train + (1 val | 1 hold alternating), s005 all train.
    "h2": {
        "m5_pilot_k2_s001": (3, 1, 0),
        "m5_pilot_k2_s002": (3, 0, 1),
        "m5_pilot_k2_s003": (3, 1, 0),
        "m5_pilot_k2_s004": (3, 0, 1),
        "m5_pilot_k2_s005": (4, 0, 0),
    },
    # 6 episodes/scenario -> first scenario 4/1/1, rest 5/(1|0)/(0|1) alternating.
    "no_maintenance": {
        "m5_pilot_k2_s001": (4, 1, 1),
        "m5_pilot_k2_s002": (5, 0, 1),
        "m5_pilot_k2_s003": (5, 1, 0),
        "m5_pilot_k2_s004": (5, 0, 1),
        "m5_pilot_k2_s005": (5, 1, 0),
    },
}


def _scenario_index(episode_id: str) -> str:
    # dataset_episode_id = "<source>_seed<seed>_ep<nnn>_<scenario_id>"
    return episode_id.rsplit("_", 1)[-1]


def _ep_number(eid: str) -> int:
    """Numeric episode number embedded in a dataset_episode_id."""
    pos = eid.rfind("_ep")
    return int(eid[pos + 3 : pos + 6])


def partition_episodes(
    transitions: List[E4RawTransition],
    source: str,
) -> Dict[str, List[E4RawTransition]]:
    """Episodic 80/10/10 split for one source, stratified by scenario."""
    per_scenario: Dict[str, Dict[str, List[E4RawTransition]]] = defaultdict(dict)
    for t in transitions:
        # Use the authoritative scenario_id field (full id, e.g.
        # "m5_pilot_k2_s001") rather than parsing the episode-id suffix.
        per_scenario[t.scenario_id].setdefault(
            t.dataset_episode_id, []
        ).append(t)

    per_scenario_template = _PER_SCENARIO[source]
    out: Dict[str, List[E4RawTransition]] = {s: [] for s in SPLITS}
    for scenario_id, template in per_scenario_template.items():
        episodes = per_scenario.get(scenario_id)
        if episodes is None:
            raise ValueError(f"source {source}: no episodes for scenario {scenario_id}")
        ordered = sorted(episodes.items(), key=lambda kv: _ep_number(kv[0]))
        n_tr, n_val, n_hold = template
        if len(ordered) != (n_tr + n_val + n_hold):
            raise ValueError(
                f"source {source} scenario {scenario_id}: {len(ordered)} episodes "
                f"but template requires {n_tr + n_val + n_hold}"
            )
        for idx, (_eid, ep_trans) in enumerate(ordered):
            if idx < n_tr:
                split = "dynamics_train"
            elif idx < n_tr + n_val:
                split = "dynamics_validate"
            else:
                split = "dynamics_holdout"
            out[split].extend(ep_trans)
    return out


def partition_all_sources(
    per_source: Dict[str, List[E4RawTransition]]
) -> Dict[str, List[E4RawTransition]]:
    """Combine all four sources into the three splits (as transition lists)."""
    out: Dict[str, List[E4RawTransition]] = {s: [] for s in SPLITS}
    for source in SOURCES:
        parts = partition_episodes(per_source[source], source)
        for split in SPLITS:
            out[split].extend(parts[split])
    return out


def build_split_manifest(seed: int, out_root: Path = E5_OUTPUT_ROOT) -> Dict[str, Any]:
    """Build and freeze the split manifest for ``seed``, writing it to disk."""
    per_source = load_dataset(seed, out_root)
    manifest: Dict[str, Any] = {
        "formal_seed": seed,
        "split_scheme": "episode-level, 80/10/10, deterministic per-scenario "
                        "stratification by source+scenario, disjoint reset seeds",
        "per_source": {},
        "totals": {"dynamics_train": 0, "dynamics_validate": 0, "dynamics_holdout": 0},
        "frozen_timestamp_utc": None,
        "manifest_sha256": None,
    }
    split_seed_sets: Dict[str, set] = {s: set() for s in SPLITS}

    for source in SOURCES:
        parts = partition_episodes(per_source[source], source)
        per_src = {"totals": {s: len(v) for s, v in parts.items()}, "episodes": {}}
        ep_counts: Dict[str, Dict[str, int]] = {}
        for split in SPLITS:
            d: Dict[str, int] = {}
            for t in parts[split]:
                d[t.dataset_episode_id] = d.get(t.dataset_episode_id, 0) + 1
                per_src["episodes"].setdefault(split, [])
                split_seed_sets[split].add(int(t.environment_reset_seed))
            ep_counts[split] = d
        per_src["episode_transition_counts"] = ep_counts
        per_src["scenario_breakdown"] = {}
        for split in SPLITS:
            scen = defaultdict(int)
            for t in parts[split]:
                scen[t.scenario_id] += 1
            per_src["scenario_breakdown"][split] = dict(scen)
        manifest["per_source"][source] = per_src
        for split in SPLITS:
            manifest["totals"][split] += per_src["totals"][split]

    # Disjoint reset-seed check across the three splits.
    for i in range(len(SPLITS)):
        for j in range(i + 1, len(SPLITS)):
            overlap = split_seed_sets[SPLITS[i]] & split_seed_sets[SPLITS[j]]
            assert not overlap, (
                f"reset seeds overlap across {SPLITS[i]}/{SPLITS[j]}: {overlap}")
    manifest["reset_seeds_disjoint"] = True

    expected = {
        "dynamics_train": 120 * EPISODE_HORIZON,
        "dynamics_validate": 15 * EPISODE_HORIZON,
        "dynamics_holdout": 15 * EPISODE_HORIZON,
    }
    assert manifest["totals"] == expected, (
        f"split totals {manifest['totals']} != expected {expected}")

    manifest["frozen_timestamp_utc"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    ser = json.dumps(manifest, indent=2, sort_keys=True)
    manifest["manifest_sha256"] = hashlib.sha256(ser.encode()).hexdigest()

    out_dir = out_root / "splits" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_split_data(seed: int, out_root: Path = E5_OUTPUT_ROOT,
                    split: str = "dynamics_train",
                    source=None) -> Dict[str, List[E4RawTransition]]:
    """Load the transitions of one split for ``seed`` grouped by episode."""
    per_source = load_dataset(seed, out_root)
    sources = (source,) if source else SOURCES
    collected: Dict[str, List[E4RawTransition]] = {}
    for src in sources:
        parts = partition_episodes(per_source[src], src)
        for s, trans in parts.items():
            if s == split:
                for t in trans:
                    collected.setdefault(t.dataset_episode_id, []).append(t)
    return collected