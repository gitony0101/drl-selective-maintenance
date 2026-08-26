"""E4 (M11) training-only dynamics dataset generation.

For each formal seed, collect 15,000 raw transitions (150 complete 100-step
episodes) on split=predictor_train, training bank configs/scenarios/m5_pilot_k2.json,
the seed's frozen M9 prediction cache, from three behavior sources:

  - random_feasible : uniform over the 16 feasible actions (data-coverage only)
  - exact_myopic    : M4 exact_myopic logistic_T5
  - h2              : canonical M6 H2Planner m6_h2_v1

Contract (Sections 6-7): cycle through all five training scenarios, deterministic
distinct reset seeds, 50 complete 100-step episodes per source. Never accesses
rl_validation or rl_test. The learned model receives only
{(obs[10], onehot(action)[16]) -> (delta_obs[10], reward)}. Every transition
records the E4 provenance fields required by Section 7.
"""

from __future__ import annotations

import json
import hashlib
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.envs.action_table import ACTION_TABLE_N5_K2
from src.envs.config import get_default_config
from src.envs.scenario_bank import load_scenario_bank
from src.envs.selective_maintenance_env import SelectiveMaintenanceEnv

from .paths import (
    COST_REGIME_ID,
    EPISODE_HORIZON,
    EVALUATION_SPLIT,
    MAINTENANCE_CAPACITY,
    N_EPISODES_PER_SOURCE,
    TRAINING_BANK,
    TRAINING_SPLIT,
    seed_cache_dir,
)


class RandomFeasiblePolicy:
    """Uniform over the canonical feasible action table, own seeded RNG."""

    def __init__(self, seed: int) -> None:
        if not isinstance(seed, int):
            raise TypeError(f"seed must be int, got {type(seed)}")
        self.rng = np.random.default_rng(seed)

    def select_action_id(self, observation: np.ndarray) -> int:
        return int(self.rng.integers(0, len(ACTION_TABLE_N5_K2)))


class M4ExactMyopicPolicy:
    """M4 exact_myopic logistic_T5 (no privileged info)."""

    def __init__(self, cost_regime_id: str = COST_REGIME_ID) -> None:
        from src.milestone10.e3.trajectories import M4ExactMyopicSelector

        self.select_action_id = M4ExactMyopicSelector(cost_regime_id).select_action_id


class H2Policy:
    """Canonical M6 H2Planner m6_h2_v1 under the per-seed context."""

    def __init__(self, seed: int, cost_regime_id: str = COST_REGIME_ID) -> None:
        from src.milestone10.e3.trajectories import H2PlannerSelector

        self.select_action_id = H2PlannerSelector(seed, cost_regime_id).select_action_id


SOURCE_POLICY_IDS = {
    "random_feasible": "random_feasible (data coverage)",
    "exact_myopic": "M4 exact_myopic logistic_T5",
    "h2": "H2 m6_h2_v1",
}


def _build_policy(source: str, seed: int):
    """Return a callable producing a policy for a given episode-reset seed."""
    if source == "random_feasible":
        # Fresh per-episode RNG so the 50 coverage episodes differ deterministically.
        return lambda eps: RandomFeasiblePolicy(eps)
    if source == "exact_myopic":
        m = M4ExactMyopicPolicy()
        return lambda _eps: m
    if source == "h2":
        h = H2Policy(seed)
        return lambda _eps: h
    raise ValueError(f"unknown source {source}")


@dataclass(frozen=True)
class E4RawTransition:
    dataset_episode_id: str
    scenario_id: str
    behavior_policy: str
    environment_reset_seed: int
    step_index: int
    observation_t: List[float]
    action_id_t: int
    reward_t: float
    next_observation_t: List[float]
    terminated_t: bool
    truncated_t: bool
    cost_preventive: float
    cost_failure: float
    cost_wasted_life: float
    cost_total: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _sha256(path_s: str) -> str:
    return hashlib.sha256(Path(path_s).read_bytes()).hexdigest()


def _git_identities(src_root: Path) -> Dict[str, Optional[str]]:
    out = {}
    for key, args in [
        ("e4_git_commit", ["git", "rev-parse", "HEAD"]),
        ("e4_git_branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        ("e4_git_tree", ["git", "rev-parse", "HEAD^{tree}"]),
    ]:
        try:
            out[key] = subprocess.check_output(args, cwd=str(src_root), text=True).strip()
        except Exception:
            out[key] = None
    return out


def _env_config_identity(env_config) -> Dict[str, Any]:
    return {
        "split": getattr(env_config, "split", None),
        "cost_regime_id": getattr(env_config, "cost_regime_id", None),
        "maintenance_capacity": getattr(env_config, "maintenance_capacity", None),
        "episode_horizon": getattr(env_config, "episode_horizon", None),
        "fleet_size": getattr(env_config, "fleet_size", None),
        "delta_cycles": getattr(env_config, "delta_cycles", None),
        "age_scale_cycles": getattr(env_config, "age_scale_cycles", None),
        "rul_scale": getattr(env_config, "rul_scale", None),
    }


_SOURCE_OFFSET = {  # keeps reset seeds disjoint across behavior sources
    "random_feasible": 0,
    "exact_myopic": 3_000_000,
    "h2": 6_000_000,
}


def _gen_deterministic_reset_seed(source: str, seed: int, episode_number: int,
                                  scenario_index: int) -> int:
    """Deterministic distinct reset seed per (source, seed, episode, scenario).

    Cyclic over the five training scenarios; a fixed derivation function so the
    same (source, seed, episode) always maps to the same reset seed. A per-source
    offset keeps seeds disjoint even across different behavior sources.
    """
    h = hashlib.sha256(f"{source}|{seed}|{episode_number}".encode()).digest()
    v = int.from_bytes(h[:4], "big")
    return 1_000_000 + (v % 100_000) + scenario_index + _SOURCE_OFFSET[source]


def collect_dataset(
    seed: int,
    src_root: Path,
    num_episodes_per_source: int = N_EPISODES_PER_SOURCE,
) -> Dict[str, Dict[str, Any]]:
    """Collect the full E4 dynamics dataset for ``seed``.

    Returns {"random_feasible"/"exact_myopic"/"h2": {"transitions": [...],
    "provenance": {...}, "integrity": {...}}}.
    """
    env_config = get_default_config(
        split=TRAINING_SPLIT,
        cost_regime_id=COST_REGIME_ID,
        maintenance_capacity=MAINTENANCE_CAPACITY,
        scenario_bank_path=TRAINING_BANK,
        prediction_cache_path=str(seed_cache_dir(seed)),
        seed=seed,
        info_mode="normal",
    )
    bank = load_scenario_bank(TRAINING_BANK)
    if bank.split != TRAINING_SPLIT:
        raise ValueError(f"bank split {bank.split} != {TRAINING_SPLIT}")
    scenario_ids = [s.scenario_id for s in bank.scenarios]

    bank_path = Path(TRAINING_BANK)
    bank_sha = _sha256(str(bank_path))
    cache_dir = seed_cache_dir(seed)
    cache_manifest = cache_dir / "prediction_cache_manifest_v2.json"
    cache_sha = _sha256(str(cache_manifest)) if cache_manifest.exists() else None

    git = _git_identities(src_root)

    per_source: Dict[str, Dict[str, Any]] = {}
    for source in ("random_feasible", "exact_myopic", "h2"):
        make_policy = _build_policy(source, seed)
        trans: List[E4RawTransition] = []
        for episode_number in range(num_episodes_per_source):
            scenario_index = episode_number % len(scenario_ids)
            scenario_id = scenario_ids[scenario_index]
            reset_seed = _gen_deterministic_reset_seed(source, seed, episode_number,
                                                        scenario_index)
            # random_feasible gets a distinct seeded RNG per episode (its own
            # policy RNG, using the episode's already-distinct environment reset
            # seed so the action stream differs across episodes); M4/H2 reuse a
            # single deterministic selector instance across episodes.
            policy = make_policy(reset_seed)

            env = SelectiveMaintenanceEnv(
                config=env_config, scenario_bank=bank, info_mode="normal"
            )
            obs, _info = env.reset(seed=reset_seed, options={"scenario_id": scenario_id})

            dataset_episode_id = (
                f"{source}_seed{seed}_ep{episode_number:03d}_{scenario_id}"
            )
            for step_index in range(EPISODE_HORIZON):
                action_id = policy.select_action_id(obs)
                next_obs, reward, terminated, truncated, info = env.step(action_id)
                trans.append(
                    E4RawTransition(
                        dataset_episode_id=dataset_episode_id,
                        scenario_id=scenario_id,
                        behavior_policy=source,
                        environment_reset_seed=reset_seed,
                        step_index=step_index,
                        observation_t=[float(x) for x in obs],
                        action_id_t=int(action_id),
                        reward_t=float(reward),
                        next_observation_t=[float(x) for x in next_obs],
                        terminated_t=bool(terminated),
                        truncated_t=bool(truncated),
                        cost_preventive=float(info.get("preventive_cost", 0.0)),
                        cost_failure=float(info.get("failure_cost", 0.0)),
                        cost_wasted_life=float(info.get("wasted_life_cost", 0.0)),
                        cost_total=float(info.get("total_cost", 0.0)),
                    )
                )
                obs = next_obs
                if truncated or terminated:
                    break  # horizon 100 episodes normally run full length

        integrity = _integrity_report(trans)
        provenance = {
            "formal_seed": seed,
            "behavior_policy": source,
            "source_policy_id": SOURCE_POLICY_IDS[source],
            "scenario_bank_path": TRAINING_BANK,
            "scenario_bank_sha256": bank_sha,
            "prediction_cache_path": str(cache_dir),
            "prediction_cache_manifest_sha256": cache_sha,
            "training_split": TRAINING_SPLIT,
            "evaluation_split_never_touched": EVALUATION_SPLIT,
            "environment_config": _env_config_identity(env_config),
            "episodes_requested": num_episodes_per_source,
            "episodes_collected": len(
                {t.dataset_episode_id for t in trans}
            ),
            "transitions_collected": len(trans),
            "generation_timestamp_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "git": git,
        }
        per_source[source] = {
            "transitions": trans,
            "provenance": provenance,
            "integrity": integrity,
        }
    return per_source


def _integrity_report(transitions: List[E4RawTransition]) -> Dict[str, bool]:
    # group by episode
    per_ep: Dict[str, List[E4RawTransition]] = {}
    for t in transitions:
        per_ep.setdefault(t.dataset_episode_id, []).append(t)
    steps_contig = True
    succ = True
    for ep, trs in per_ep.items():
        idxs = sorted(t.step_index for t in trs)
        if idxs != list(range(len(trs))):
            steps_contig = False
        ordered = sorted(trs, key=lambda t: t.step_index)
        for i in range(len(ordered) - 1):
            a = np.array(ordered[i].next_observation_t, dtype=np.float32)
            b = np.array(ordered[i + 1].observation_t, dtype=np.float32)
            if not np.allclose(a, b, atol=1e-6):
                succ = False
    return {
        "obs_shape_ok": all(len(t.observation_t) == 10 and len(t.next_observation_t) == 10
                            for t in transitions),
        "action_valid": all(0 <= t.action_id_t < len(ACTION_TABLE_N5_K2)
                            for t in transitions),
        "reward_finite": all(np.isfinite(t.reward_t) and np.isfinite(t.cost_total)
                             for t in transitions),
        "obs_finite": all(np.all(np.isfinite(t.observation_t)) and
                          np.all(np.isfinite(t.next_observation_t)) for t in transitions),
        "step_indices_contiguous": steps_contig,
        "next_obs_successor": succ,
        "no_cross_episode_boundary": all(
            int(t.step_index) < EPISODE_HORIZON for t in transitions),
        "no_duplicate_transition_ids": _no_dupes(transitions),
    }


def _no_dupes(transitions: List[E4RawTransition]) -> bool:
    seen = set()
    for t in transitions:
        key = (t.dataset_episode_id, t.step_index)
        if key in seen:
            return False
        seen.add(key)
    return True


def write_dataset(per_source: Dict[str, Dict[str, Any]], seed: int,
                  out_root: Path) -> Dict[str, Path]:
    """Write the dataset for ``seed`` as jsonl + provenance + integrity."""
    import pandas as pd

    paths: Dict[str, Path] = {}
    for source, blob in per_source.items():
        subdir = out_root / "dataset" / f"seed_{seed}" / source
        subdir.mkdir(parents=True, exist_ok=True)
        raw_path = subdir / "raw_transitions.jsonl"
        with open(raw_path, "w") as f:
            for t in blob["transitions"]:
                f.write(json.dumps(t.to_dict()) + "\n")
        # Parquet copy for fast model training.
        df = pd.DataFrame([t.to_dict() for t in blob["transitions"]])
        parquet_path = subdir / "transitions.parquet"
        df.to_parquet(parquet_path, index=False)
        prov_path = subdir / "provenance.json"
        prov_path.write_text(json.dumps(blob["provenance"], indent=2))
        integ_path = subdir / "integrity.json"
        integ_path.write_text(json.dumps(blob["integrity"], indent=2))
        paths[source] = raw_path
        paths[f"{source}_provenance"] = prov_path
        paths[f"{source}_integrity"] = integ_path
    return paths


def load_dataset(seed: int, out_root: Path,
                 sources=("random_feasible", "exact_myopic", "h2")) -> Dict[str, List[E4RawTransition]]:
    """Load the frozen E4 dataset for ``seed`` from disk."""
    loaded: Dict[str, List[E4RawTransition]] = {}
    for source in sources:
        p = out_root / "dataset" / f"seed_{seed}" / source / "raw_transitions.jsonl"
        rows = []
        with open(p) as f:
            for line in f:
                rows.append(E4RawTransition(**json.loads(line)))
        loaded[source] = rows
    return loaded