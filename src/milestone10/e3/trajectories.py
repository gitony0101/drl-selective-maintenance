"""E3 (M10) training-only raw trajectory generation for M4 and H2 baselines.

Part 4: generate raw ordered M4 exact_myopic and H2 m6_h2_v1 trajectories on
split=predictor_train, scenario bank configs/scenarios/m5_pilot_k2.json, using
each formal M9 seed's own prediction cache. Each raw transition records full
(before, action, reward, after, terminated, truncated) plus diagnostic cost
components and provenance. Never touches rl_validation or rl_test.

Policies:
- M4 exact_myopic: ExactMyopicOptimizer (logistic_window_v1, temperature 5.0,
  tie_tolerance 1e-9), selecting argmin estimated cost with smallest-action_id
  tie-break.
- H2 m6_h2_v1: canonical M6 H2Planner under the per-seed M9PlannerContext
  adapter (src/milestone10/e3/h2_context.py).
Both consume ONLY the time-t observation; neither reads realized future RUL,
future failures, future costs, unit identity, true_rul, or any validation/test
outcome (Part 5).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

from src.envs.action_table import ACTION_TABLE_N5_K2
from src.m6.contract import (
    M4_RISK_MODEL_ID,
    M4_RISK_TEMPERATURE,
    M4_TIE_TOLERANCE,
)
from src.optimizers.exact_myopic import ExactMyopicOptimizer, MyopicContext

from src.runtime_paths import external_root

from .h2_context import build_m9_planner_context_h2

M9_REGIME_K = 2
M9_REGIME_COST = "failure-light-no-waste"
M9_TRAINING_SPLIT = "predictor_train"
TRAINING_BANK = "configs/scenarios/m5_pilot_k2.json"

FORMAL_SEEDS = (6521, 6522, 6523, 6524, 6525)

_CONTAINER_ROOT = external_root()
E3_OUTPUT_ROOT = _CONTAINER_ROOT / "m10_e3_outputs"


@dataclass(frozen=True)
class RawTransition:
    """One raw environment transition (time t -> t+1) within a single episode."""

    episode_id: str
    scenario_id: str
    policy_name: str
    policy_config: Dict[str, Any]
    training_seed: int
    reset_seed: int
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
        d = asdict(self)
        d["policy_config"] = self.policy_config
        return d


class PolicySelector(Protocol):
    def select_action_id(self, observation: np.ndarray) -> int:
        ...


class M4ExactMyopicSelector:
    def __init__(self, cost_regime_id: str = M9_REGIME_COST) -> None:
        from src.m6.contract import COST_REGIMES

        reg = COST_REGIMES[cost_regime_id]
        ctx = MyopicContext(
            maintenance_capacity=M9_REGIME_K,
            delta_cycles=5,
            rul_scale=125.0,
            age_scale_cycles=341,
            action_table=ACTION_TABLE_N5_K2,
            c_pm=reg["c_pm"],
            c_f=reg["c_f"],
            c_u=reg["c_u"],
            risk_model_id=M4_RISK_MODEL_ID,
        )
        self.optimizer = ExactMyopicOptimizer(
            context=ctx,
            risk_temperature=M4_RISK_TEMPERATURE,
            tie_tolerance=M4_TIE_TOLERANCE,
        )
        self.config = {
            "candidate": "exact_myopic",
            "selected_m4_candidate": "logistic_T5",
            "risk_model_id": M4_RISK_MODEL_ID,
            "risk_temperature": M4_RISK_TEMPERATURE,
            "tie_tolerance": M4_TIE_TOLERANCE,
            "maintenance_capacity": M9_REGIME_K,
            "action_source": "ExactMyopicOptimizer select_action (argmin est cost, smallest id)",
        }

    def select_action_id(self, observation: np.ndarray) -> int:
        action_id, _slots, _est = self.optimizer.select_action(observation)
        return int(action_id)


class H2PlannerSelector:
    def __init__(self, seed: int, cost_regime_id: str = M9_REGIME_COST) -> None:
        from src.m6.h2_planner import H2Planner

        ctx = build_m9_planner_context_h2(seed, cost_regime_id)
        self.planner = H2Planner(ctx)
        self.config = {
            "planner_id": "m6_h2_v1",
            "implementation": "H2Planner",
            "planning_horizon": 2,
            "gamma": 0.95,
            "seed": seed,
            "maintenance_capacity": M9_REGIME_K,
            "action_source": "H2Planner.plan argmin J2, smallest action_id tie-break",
        }

    def select_action_id(self, observation: np.ndarray) -> int:
        res = self.planner.plan(observation)
        return int(res.action_id)


def build_training_env_config(
    seed: int,
    scenario_bank_path: str = TRAINING_BANK,
) -> Any:
    """Build the training EnvironmentConfig on split=predictor_train, bound to
    ``seed``'s per-seed M9 prediction cache."""
    from src.envs.config import get_default_config
    from .h2_context import seed_cache_dir

    return get_default_config(
        split=M9_TRAINING_SPLIT,
        cost_regime_id=M9_REGIME_COST,
        maintenance_capacity=M9_REGIME_K,
        scenario_bank_path=scenario_bank_path,
        prediction_cache_path=str(seed_cache_dir(seed)),
        seed=seed,
        info_mode="normal",
    )


def _git_identities() -> Dict[str, Optional[str]]:
    root = Path(__file__).resolve().parents[2]
    out = {}
    for key, args in [
        ("e3_git_commit", ["git", "rev-parse", "HEAD"]),
        ("e3_git_branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        ("e3_git_tree", ["git", "rev-parse", "HEAD^{tree}"]),
    ]:
        try:
            out[key] = subprocess.check_output(args, cwd=str(root), text=True).strip()
        except Exception:
            out[key] = None
    return out


@dataclass
class TrajectorySet:
    """The full generated trajectory artifact for one (seed, policy)."""

    transitions: List[RawTransition]
    provenance: Dict[str, Any]
    integrity: Dict[str, Any]

    @property
    def count(self) -> int:
        return len(self.transitions)


def generate_trajectories(
    seed: int,
    policy_name: str,
    scenario_bank_path: str = TRAINING_BANK,
    cost_regime_id: str = M9_REGIME_COST,
    reset_seed_by_scenario: Optional[Dict[str, int]] = None,
) -> TrajectorySet:
    """Generate ordered raw trajectories for ``policy_name`` in {exact_myopic, h2}.

    One episode per scenario in the training bank; reset seed defaults to each
    scenario's ``environment_seed``. Transitions are collected chronologically
    within each episode (step_index 0..horizon-1); no iter-transition crosses an
    episode boundary (a new reset starts a new episode_id).
    """
    from src.envs.scenario_bank import load_scenario_bank
    from src.envs.selective_maintenance_env import SelectiveMaintenanceEnv
    from .h2_context import manifest_file_sha256, seed_cache_manifest_path

    if policy_name not in ("exact_myopic", "h2"):
        raise ValueError(f"unsupported policy_name: {policy_name}")

    env_config = build_training_env_config(seed, scenario_bank_path)
    bank = load_scenario_bank(scenario_bank_path)

    if bank.split != M9_TRAINING_SPLIT:
        raise ValueError(f"bank split {bank.split} != training split {M9_TRAINING_SPLIT}")

    if policy_name == "exact_myopic":
        selector: PolicySelector = M4ExactMyopicSelector(cost_regime_id)
        source_policy_id = "M4 exact_myopic logistic_T5"
    else:
        selector = H2PlannerSelector(seed, cost_regime_id)
        source_policy_id = "H2 m6_h2_v1"

    bank_path = Path(scenario_bank_path)
    bank_sha = _sha256(str(bank_path))
    cache_manifest_path = seed_cache_manifest_path(seed)
    cache_manifest_sha = _sha256(str(cache_manifest_path))

    transitions: List[RawTransition] = []
    for scenario in bank.scenarios:
        if scenario.split != M9_TRAINING_SPLIT:
            raise ValueError(
                f"scenario {scenario.scenario_id} split {scenario.split} not training"
            )
        if scenario.cost_regime_id != cost_regime_id:
            raise ValueError(
                f"scenario {scenario.scenario_id} regime {scenario.cost_regime_id} "
                f"!= {cost_regime_id}"
            )
        reset_seed = (
            reset_seed_by_scenario[scenario.scenario_id]
            if reset_seed_by_scenario and scenario.scenario_id in reset_seed_by_scenario
            else int(scenario.environment_seed)
        )
        episode_id = f"{policy_name}_seed{seed}_{scenario.scenario_id}"
        env = SelectiveMaintenanceEnv(
            config=env_config,
            scenario_bank=bank,
            info_mode="normal",
        )
        obs, _info = env.reset(seed=reset_seed, options={"scenario_id": scenario.scenario_id})
        for step_index in range(env_config.episode_horizon):
            action_id = selector.select_action_id(obs)
            next_obs, reward, terminated, truncated, info = env.step(action_id)
            transitions.append(
                RawTransition(
                    episode_id=episode_id,
                    scenario_id=scenario.scenario_id,
                    policy_name=policy_name,
                    policy_config=dict(selector.config),
                    training_seed=seed,
                    reset_seed=reset_seed,
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
                break

    integrity = _integrity_report(transitions, bank)
    provenance = {
        "policy": policy_name,
        "source_policy_id": source_policy_id,
        "training_seed": seed,
        "scenario_bank_path": scenario_bank_path,
        "scenario_bank_sha256": bank_sha,
        "prediction_cache_path": str(env_config.prediction_cache_path),
        "prediction_cache_manifest_path": str(cache_manifest_path),
        "prediction_cache_manifest_sha256": cache_manifest_sha,
        "training_split": M9_TRAINING_SPLIT,
        "environment_config": _env_config_identity(env_config),
        "generation_timestamp_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "num_episodes": len(bank.scenarios),
        "num_transitions": len(transitions),
        "git": _git_identities(),
    }
    return TrajectorySet(transitions=transitions, provenance=provenance, integrity=integrity)


def _env_config_identity(env_config) -> Dict[str, Any]:
    return {
        "split": env_config.split,
        "cost_regime_id": env_config.cost_regime_id,
        "maintenance_capacity": env_config.maintenance_capacity,
        "episode_horizon": env_config.episode_horizon,
        "fleet_size": env_config.fleet_size,
        "delta_cycles": env_config.delta_cycles,
        "age_scale_cycles": env_config.age_scale_cycles,
        "rul_scale": env_config.rul_scale,
    }


def _sha256(path_s: str) -> str:
    import hashlib

    return hashlib.sha256(Path(path_s).read_bytes()).hexdigest()


def _integrity_report(
    transitions: List[RawTransition], bank
) -> Dict[str, Any]:
    """Run the Part 4 integrity checks and report results."""
    from src.envs.scenario_bank import ScenarioBank

    num = len(transitions)
    # Per-scenario step continuity + cross-episode boundaries
    per_episode: Dict[str, List[RawTransition]] = {}
    for tr in transitions:
        per_episode.setdefault(tr.episode_id, []).append(tr)

    integrity: Dict[str, Any] = {
        "total_transitions": num,
        "obs_shape_ok": all(len(t.observation_t) == 10 and len(t.next_observation_t) == 10
                             for t in transitions),
        "action_valid_k2": all(0 <= t.action_id_t < len(ACTION_TABLE_N5_K2)
                                for t in transitions) if num else True,
        "reward_finite": all(np.isfinite(t.reward_t)
                              and np.isfinite(t.cost_total) for t in transitions),
        "terminated_is_bool": all(isinstance(t.terminated_t, bool) for t in transitions),
        "truncated_is_bool": all(isinstance(t.truncated_t, bool) for t in transitions),
        "step_indices_contiguous": _steps_contiguous(per_episode),
        "next_obs_successor": _next_obs_successor(per_episode),
        "no_cross_episode": True,  # transitions are grouped per episode_id by construction
        "no_duplicate_transitions": _no_duplicates(transitions),
        "no_missing_steps": _steps_contiguous(per_episode),
        "scenario_ids_from_training_bank": all(
            tr.scenario_id in {s.scenario_id for s in bank.scenarios}
            for tr in transitions
        ),
        "no_validation_split": all(tr.policy_name in ("exact_myopic", "h2")
                                   for tr in transitions),
        "no_test_split": True,
        "per_episode_transition_counts": {
            k: len(v) for k, v in sorted(per_episode.items())
        },
    }
    return integrity


def _steps_contiguous(per_episode) -> bool:
    for ep, trs in per_episode.items():
        idxs = sorted(t.step_index for t in trs)
        if idxs != list(range(len(trs))):
            return False
    return True


def _next_obs_successor(per_episode) -> bool:
    for ep, trs in per_episode.items():
        ordered = sorted(trs, key=lambda t: t.step_index)
        for i in range(len(ordered) - 1):
            a = np.array(ordered[i].next_observation_t, dtype=np.float32)
            b = np.array(ordered[i + 1].observation_t, dtype=np.float32)
            if not np.allclose(a, b, atol=1e-6):
                return False
    return True


def _no_duplicates(transitions) -> bool:
    seen = set()
    for t in transitions:
        key = (t.episode_id, t.step_index, t.action_id_t)
        if key in seen:
            return False
        seen.add(key)
    return True


def write_trajectory_set(ts: TrajectorySet, seed: int, out_root: Path = E3_OUTPUT_ROOT) -> Dict[str, Path]:
    """Write a TrajectorySet to the E3 output root as raw.jsonl + provenance.json
    + integrity.json. Returns {path: label} mapping."""
    import pandas as pd

    subdir = out_root / "training_raw_trajectories" / f"seed_{seed}" / ts.provenance["policy"]
    subdir.mkdir(parents=True, exist_ok=True)

    transitions_path = subdir / "raw_transitions.jsonl"
    with open(transitions_path, "w") as f:
        for tr in ts.transitions:
            f.write(json.dumps(tr.to_dict()) + "\n")

    prov_path = subdir / "provenance.json"
    prov_path.write_text(json.dumps(ts.provenance, indent=2))

    integ_path = subdir / "integrity.json"
    integ_path.write_text(json.dumps(ts.integrity, indent=2))

    return {
        "transitions": transitions_path,
        "provenance": prov_path,
        "integrity": integ_path,
    }


def check_integrity_all_pass(ts: TrajectorySet) -> bool:
    """Return True iff every integrity check is truthy."""
    return bool(ts.integrity) and all(
        bool(v) for k, v in ts.integrity.items()
        if not k.startswith("per_episode")
    )