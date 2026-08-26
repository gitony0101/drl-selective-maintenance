"""E5-B training-only dynamics dataset generation (Section 4).

For each formal seed, collect 15,000 raw transitions (150 complete 100-step
episodes) on split=predictor_train, training bank configs/scenarios/m5_pilot_k2.json,
the seed's frozen M9 prediction cache, from FOUR behavior sources:

  - random_feasible (50 ep): uniform over the 16 feasible actions (data coverage)
  - exact_myopic    (50 ep): M4 exact_myopic logistic_T5
  - h2              (20 ep): canonical M6 H2Planner m6_h2_v1
  - no_maintenance  (30 ep): a_t = 0 for every step (coverage EXCLUSIVELY to
                             induce genuine failure trajectories and failure-cost
                             events in the training data)

The no-maintenance source is NOT a scientific comparator, a new exploration
algorithm, or a new RL method. Its entire purpose is to add real failure-event
coverage to the model-training dataset (Section 4).

The learned model receives only {(obs[10], onehot(action)[16]) -> (delta_obs[10], reward)}.
Never accesses rl_validation or rl_test. Deterministic distinct reset seeds are
recorded in the manifest. Every transition carries the E4 provenance fields.
"""

from __future__ import annotations

import json
import hashlib
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.envs.config import get_default_config
from src.envs.scenario_bank import load_scenario_bank
from src.envs.selective_maintenance_env import SelectiveMaintenanceEnv

from src.milestone11.e4.dataset import (
    E4RawTransition,
    H2Policy,
    M4ExactMyopicPolicy,
    RandomFeasiblePolicy,
    _integrity_report as e4_integrity_report,
)
from .paths import (
    COST_REGIME_ID,
    EPISODES_PER_SOURCE,
    EPISODE_HORIZON,
    EVALUATION_SPLIT,
    MAINTENANCE_CAPACITY,
    NO_MAINTENANCE_ACTION_ID,
    TRAINING_BANK,
    TRAINING_SPLIT,
    seed_cache_dir,
)


class NoMaintenancePolicy:
    """Coverage policy: a_t = 0 for every step (never maintain).

    Purpose is ONLY to drive fleet slots to genuine failure so the training
    dataset contains decision-critical failure-cost events.
    """

    def select_action_id(self, observation: np.ndarray) -> int:
        return NO_MAINTENANCE_ACTION_ID


SOURCE_POLICY_IDS = {
    "random_feasible": "random_feasible (data coverage)",
    "exact_myopic": "M4 exact_myopic logistic_T5",
    "h2": "H2 m6_h2_v1",
    "no_maintenance": "a_t = 0 (failure-coverage; NOT a comparator)",
}


def _build_policy(source: str, seed: int) -> Callable[[int], Any]:
    """Return a factory producing a policy for a given episode-reset seed."""
    if source == "random_feasible":
        # Fresh per-episode RNG so the 50 coverage episodes differ deterministically.
        return lambda eps: RandomFeasiblePolicy(eps)
    if source == "exact_myopic":
        m = M4ExactMyopicPolicy()
        return lambda _eps: m
    if source == "h2":
        h = H2Policy(seed)
        return lambda _eps: h
    if source == "no_maintenance":
        n = NoMaintenancePolicy()
        return lambda _eps: n
    raise ValueError(f"unknown source {source}")


def _sha256(path_s: str) -> str:
    return hashlib.sha256(Path(path_s).read_bytes()).hexdigest()


def _git_identities(src_root: Path) -> Dict[str, Optional[str]]:
    out = {}
    for key, args in [
        ("e5_git_commit", ["git", "rev-parse", "HEAD"]),
        ("e5_git_branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        ("e5_git_tree", ["git", "rev-parse", "HEAD^{tree}"]),
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


# Keeps reset seeds disjoint across behavior sources (E4 used 0/3M/6M offsets;
# E5 preserves those and adds a distinct offset for no_maintenance).
_SOURCE_OFFSET = {
    "random_feasible": 0,
    "exact_myopic": 3_000_000,
    "h2": 6_000_000,
    "no_maintenance": 9_000_000,
}


def gen_deterministic_reset_seed(source: str, seed: int, episode_number: int,
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
) -> Dict[str, Dict[str, Any]]:
    """Collect the full E5-B dynamics dataset for ``seed``.

    Returns {"random_feasible"/"exact_myopic"/"h2"/"no_maintenance":
    {"transitions": [...], "provenance": {...}, "integrity": {...}}}.
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
    for source in EPISODES_PER_SOURCE:
        num_episodes = EPISODES_PER_SOURCE[source]
        make_policy = _build_policy(source, seed)
        trans: List[E4RawTransition] = []
        for episode_number in range(num_episodes):
            scenario_index = episode_number % len(scenario_ids)
            scenario_id = scenario_ids[scenario_index]
            reset_seed = gen_deterministic_reset_seed(source, seed, episode_number,
                                                      scenario_index)
            # random_feasible gets a distinct seeded RNG per episode; M4/H2/no-main
            # reuse a single deterministic selector instance across episodes.
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
                    break  # horizon-100 episodes normally run full length

        integrity = e4_integrity_report(trans)
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
            "episodes_requested": num_episodes,
            "episodes_collected": len({t.dataset_episode_id for t in trans}),
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


def write_dataset(per_source: Dict[str, Dict[str, Any]], seed: int,
                  out_root: Path) -> Dict[str, Path]:
    """Write the E5-B dataset for ``seed`` as jsonl + provenance + integrity."""
    import pandas as pd

    paths: Dict[str, Path] = {}
    for source, blob in per_source.items():
        subdir = out_root / "dataset" / f"seed_{seed}" / source
        subdir.mkdir(parents=True, exist_ok=True)
        raw_path = subdir / "raw_transitions.jsonl"
        with open(raw_path, "w") as f:
            for t in blob["transitions"]:
                f.write(json.dumps(asdict(t)) + "\n")
        df = pd.DataFrame([asdict(t) for t in blob["transitions"]])
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
                 sources=("random_feasible", "exact_myopic", "h2", "no_maintenance")
                 ) -> Dict[str, List[E4RawTransition]]:
    """Load the E5-B dataset for ``seed`` from disk."""
    loaded: Dict[str, List[E4RawTransition]] = {}
    for source in sources:
        p = out_root / "dataset" / f"seed_{seed}" / source / "raw_transitions.jsonl"
        rows = []
        with open(p) as f:
            for line in f:
                rows.append(E4RawTransition(**json.loads(line)))
        loaded[source] = rows
    return loaded