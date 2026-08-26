"""E4 (M11) episode-level train / model-val / model-holdout split (Task 3).

Splits at the EPISODE level, never at the transition level. For each behavior
source and formal seed, the 50 collected episodes are partitioned into:

    dynamics_train   : 40 episodes  (8 per scenario)
    dynamics_validate:  5 episodes  (1 per scenario)
    dynamics_holdout :  5 episodes  (1 per scenario)

Stratified across source and scenario; reset seeds are disjoint across the three
splits (each episode already carries a distinct deterministic reset seed). A
frozen split manifest is written before any model training consumes the data.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .dataset import E4RawTransition, load_dataset, write_dataset
from .paths import (
    E4_OUTPUT_ROOT,
    FORMAL_SEEDS,
    N_TRAIN_EPISODES,
    N_VAL_EPISODES,
    N_HOLDOUT_EPISODES,
)

SOURCES = ("random_feasible", "exact_myopic", "h2")
SPLITS = ("dynamics_train", "dynamics_validate", "dynamics_holdout")


def _scenario_index(episode_id: str) -> str:
    # dataset_episode_id = "<source>_seed<seed>_ep<nnn>_<scenario_id>"
    return episode_id.rsplit("_", 1)[-1]


def partition_episodes(
    transitions: List[E4RawTransition],
) -> Dict[str, List[E4RawTransition]]:
    """Episodic 40/5/5 split stratified by scenario, deterministic.

    Within each scenario the episodes are sorted by episode number and assigned
    8 -> train, 1 -> val, 1 -> holdout (per scenario), preserving all five
    scenarios in every split.
    """
    # Group episodes per scenario
    from collections import defaultdict

    per_scenario: Dict[str, Dict[str, List[E4RawTransition]]] = defaultdict(dict)
    for t in transitions:
        per_scenario[_scenario_index(t.dataset_episode_id)].setdefault(
            t.dataset_episode_id, []
        ).append(t)

    out: Dict[str, List[E4RawTransition]] = {s: [] for s in SPLITS}
    scenario_counts: Dict[str, Dict[str, int]] = {}
    for scenario_id, episodes in sorted(per_scenario.items()):
        # Sort episodes by the numeric episode number embedded in the id.
        def _ep_no(eid: str) -> int:
            # "..._epDDD_<scen>" boundary: index of the "_ep" token before final.
            pos = eid.rfind("_ep")
            return int(eid[pos + 3 : pos + 6])

        ordered = sorted(episodes.items(), key=lambda kv: _ep_no(kv[0]))
        if len(ordered) != 10:
            raise ValueError(
                f"expected 10 episodes for scenario {scenario_id}, got {len(ordered)}"
            )
        scenario_counts[scenario_id] = {s: 0 for s in SPLITS}
        for idx, (_eid, ep_trans) in enumerate(ordered):
            if idx < 8:
                split = "dynamics_train"
            elif idx == 8:
                split = "dynamics_validate"
            else:
                split = "dynamics_holdout"
            out[split].extend(ep_trans)
            scenario_counts[scenario_id][split] += 1

    # Stratification invariant: every split sees every scenario exactly once.
    for split in SPLITS:
        for scen, counts in scenario_counts.items():
            expected = {"dynamics_train": 8, "dynamics_validate": 1,
                        "dynamics_holdout": 1}[split]
            if counts[split] != expected:
                raise ValueError(
                    f"stratification broken for {scen} {split}: "
                    f"{counts[split]} != {expected}"
                )
    return out


def build_split_manifest(seed: int, out_root: Path = E4_OUTPUT_ROOT) -> Dict[str, Any]:
    """Build and freeze the split manifest for ``seed``, writing it to disk."""
    per_source = load_dataset(seed, out_root)
    manifest: Dict[str, Any] = {
        "formal_seed": seed,
        "split_scheme": "episode-level, 40/5/5, stratified by source+scenario, "
                        "disjoint reset seeds",
        "per_source": {},
        "totals": {"dynamics_train": 0, "dynamics_validate": 0, "dynamics_holdout": 0},
        "frozen_timestamp_utc": None,
        "manifest_sha256": None,
    }
    for source in SOURCES:
        parts = partition_episodes(per_source[source])
        per_src = {"totals": {s: len(v) for s, v in parts.items()},
                   "episodes": {}}
        # Record episode ids + reset seeds per split.
        for split in SPLITS:
            for t in parts[split]:
                per_src["episodes"].setdefault(split, []).append({
                    "dataset_episode_id": t.dataset_episode_id,
                    "scenario_id": t.scenario_id,
                    "environment_reset_seed": t.environment_reset_seed,
                    "behavior_policy": t.behavior_policy,
                    "n_transitions": 1,  # counted below
                })
        # collapse per-episode counts
        ep_counts: Dict[str, Dict[str, int]] = {}
        for split, trans in parts.items():
            d: Dict[str, int] = {}
            for t in trans:
                d[t.dataset_episode_id] = d.get(t.dataset_episode_id, 0) + 1
            ep_counts[split] = d
        per_src["episode_transition_counts"] = ep_counts
        per_src["totals"] = {s: len(v) for s, v in parts.items()}
        manifest["per_source"][source] = per_src
        for split in SPLITS:
            manifest["totals"][split] += per_src["totals"][split]

    # Disjoint reset-seed check across the three splits: an episode's reset seed
    # used in one split must never equal an episode's reset seed in another
    # split. Compare the SET of unique episode reset seeds per split, not raw
    # per-transition lists (each episode shares one reset seed across its steps).
    split_seed_sets: Dict[str, set] = {s: set() for s in SPLITS}
    for source in SOURCES:
        parts = partition_episodes(per_source[source])
        for split in SPLITS:
            for t in parts[split]:
                split_seed_sets[split].add(int(t.environment_reset_seed))
    for i in range(len(SPLITS)):
        for j in range(i + 1, len(SPLITS)):
            overlap = split_seed_sets[SPLITS[i]] & split_seed_sets[SPLITS[j]]
            assert not overlap, (
                f"reset seeds overlap across {SPLITS[i]}/{SPLITS[j]}: {overlap}")
    manifest["reset_seeds_disjoint"] = True

    num_vals = sum(manifest["totals"].values())
    assert num_vals == len(per_source["random_feasible"]) \
        + len(per_source["exact_myopic"]) + len(per_source["h2"]), \
        "split counts != total transitions"

    manifest["frozen_timestamp_utc"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    ser = json.dumps(manifest, indent=2, sort_keys=True)
    manifest["manifest_sha256"] = hashlib.sha256(ser.encode()).hexdigest()

    out_dir = out_root / "splits" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_split_data(seed: int, out_root: Path = E4_OUTPUT_ROOT,
                    split: str = "dynamics_train",
                    source=None) -> Dict[str, List[E4RawTransition]]:
    """Load transitions belonging to one split for ``seed``.

    Returns a dict of {episode_id: [transitions]} for the requested split and
    (optionally) single source across all sources.
    """
    manifest = json.loads(
        (out_root / "splits" / f"seed_{seed}" / "split_manifest.json").read_text()
    )
    per_source = load_dataset(seed, out_root)
    sources = (source,) if source else SOURCES
    collected: Dict[str, List[E4RawTransition]] = {}
    for src in sources:
        parts = partition_episodes(per_source[src])
        for s, trans in parts.items():
            if s == split:
                collected.update(group_by_episode(trans, src))
    return collected


def group_by_episode(transitions: List[E4RawTransition], source: str):
    out: Dict[str, List[E4RawTransition]] = {}
    for t in transitions:
        out.setdefault(t.dataset_episode_id, []).append(t)
    return out