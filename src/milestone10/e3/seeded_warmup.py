"""E3 (M10) frozen seeded-warmup manifest builder (Section 16/17/8).

For every formal training seed (6521..6525) creates a FROZEN seeded warmup
manifest of exactly 5000 raw training-only transitions:

    1667 M4 (exact_myopic) transitions
    1667 H2 (m6_h2_v1) transitions
    1666 canonical exploration warmup transitions
    --------------------------------------------
    5000 total

All transitions are generated on split=predictor_train with scenario bank
``configs/scenarios/m5_pilot_k2.json`` and the given seed's OWN per-seed M9
prediction cache (never rl_validation / rl_test). Each transition records the
raw (obs, action, reward, next_obs, terminated, truncated) fields plus episode/
scenario/reset-seed/step provenance and diagnostic cost components.

Cell C (n=1) and Cell D (n=3) for a seed consume the SAME manifest; the only
difference is the n-step conversion of the SAME raw experience. The 5000 seeded
transitions count as the FIRST 5000 raw transitions of the total training
budget, so the first gradient update sees global_step == epsilon_step == 5000.

The exploration-warmup portion replicates the canonical DDQN warmup
action-selection mechanism (epsilon=1.0, uniform over the K=2 action space) so
the seeded and standard warmup cells differ ONLY in replay-recall provenance,
not in the kind of early experience.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.envs.action_table import ACTION_TABLE_N5_K2
from src.m6.contract import M4_RISK_MODEL_ID, M4_RISK_TEMPERATURE, M4_TIE_TOLERANCE
from src.optimizers.exact_myopic import ExactMyopicOptimizer, MyopicContext
from src.envs.config import get_default_config

from .trajectories import (
    FORMAL_SEEDS,
    E3_OUTPUT_ROOT,
    M9_REGIME_K,
    M9_REGIME_COST,
    M9_TRAINING_SPLIT,
    TRAINING_BANK,
    M4ExactMyopicSelector,
    H2PlannerSelector,
)
from .h2_context import build_m9_planner_context_h2, seed_cache_dir

# Frozen mixture (Section 16). Do not tune.
SEEDED_M4_COUNT = 1667
SEEDED_H2_COUNT = 1667
SEEDED_EXPL_COUNT = 1666
SEEDED_WARMUP_TOTAL = 5000

_RESET_STRIDE = 31


@dataclass(frozen=True)
class WarmupRawTransition:
    """One raw warmup transition with frozen manifest provenance."""

    source_policy: str            # "exact_myopic" | "h2" | "exploration"
    scenario_id: str
    reset_seed: int
    episode_id: str
    step_index: int
    training_seed: int
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


def _run_episode_frames(
    seed: int,
    policy: str,
    scenario_id: str,
    reset_seed: int,
    env_config,
    bank,
    selectors: Dict[str, object],
    explored_agent=None,
    explored_rng_key: Optional[int] = None,
) -> List[WarmupRawTransition]:
    """Run one episode for ``policy`` and return its raw frames (no budget cut)."""
    from src.envs.selective_maintenance_env import SelectiveMaintenanceEnv

    env = SelectiveMaintenanceEnv(config=env_config, scenario_bank=bank, info_mode="normal")
    obs, _info = env.reset(seed=reset_seed, options={"scenario_id": scenario_id})
    episode_id = f"warmup_{policy}_seed{seed}_{scenario_id}_r{reset_seed}"
    frames: List[WarmupRawTransition] = []

    for step_index in range(env_config.episode_horizon):
        if policy == "exploration":
            legal = env.get_action_mask()
            rng = np.random.default_rng(explored_rng_key + step_index) if explored_rng_key is not None else None
            legal_indices = np.flatnonzero(legal)
            if rng is not None:
                action_id = int(rng.choice(legal_indices))
            else:
                action_id = int(np.random.choice(legal_indices))
        elif policy == "exact_myopic":
            action_id = selectors["exact_myopic"].select_action_id(obs)
        elif policy == "h2":
            action_id = selectors["h2"].select_action_id(obs)
        else:
            raise ValueError(f"unknown warmup policy: {policy}")

        next_obs, reward, terminated, truncated, info = env.step(action_id)
        frames.append(
            WarmupRawTransition(
                source_policy=policy, scenario_id=scenario_id, reset_seed=reset_seed,
                episode_id=episode_id, step_index=step_index, training_seed=seed,
                observation_t=[float(x) for x in obs], action_id_t=int(action_id),
                reward_t=float(reward), next_observation_t=[float(x) for x in next_obs],
                terminated_t=bool(terminated), truncated_t=bool(truncated),
                cost_preventive=float(info.get("preventive_cost", 0.0)),
                cost_failure=float(info.get("failure_cost", 0.0)),
                cost_wasted_life=float(info.get("wasted_life_cost", 0.0)),
                cost_total=float(info.get("total_cost", 0.0)),
            )
        )
        obs = next_obs
        if terminated or truncated:
            break
    return frames


def build_seeded_warmup_frames(
    seed: int,
    scenario_bank_path: str = TRAINING_BANK,
    cost_regime_id: str = M9_REGIME_COST,
) -> List[WarmupRawTransition]:
    """Generate the frozen 5000-transition seeded warmup for ``seed``."""
    from src.envs.scenario_bank import load_scenario_bank

    if seed not in FORMAL_SEEDS:
        raise ValueError(f"seed {seed} not in {list(FORMAL_SEEDS)}")

    env_config = get_default_config(
        split=M9_TRAINING_SPLIT, cost_regime_id=cost_regime_id,
        maintenance_capacity=M9_REGIME_K, scenario_bank_path=scenario_bank_path,
        prediction_cache_path=str(seed_cache_dir(seed)), seed=seed, info_mode="normal",
    )
    bank = load_scenario_bank(scenario_bank_path)
    if bank.split != M9_TRAINING_SPLIT:
        raise ValueError("bank split != predictor_train")
    scenarios = bank.scenarios

    selectors = {
        "exact_myopic": M4ExactMyopicSelector(cost_regime_id),
        "h2": H2PlannerSelector(seed, cost_regime_id),
    }

    frames: List[WarmupRawTransition] = []
    for policy, target in (("exact_myopic", SEEDED_M4_COUNT),
                           ("h2", SEEDED_H2_COUNT)):
        collected: List[WarmupRawTransition] = []
        ep_idx = 0
        # Cycle scenarios with deterministic distinct reset seeds.
        while len(collected) < target:
            sc = scenarios[ep_idx % len(scenarios)]
            reset_seed = int(sc.environment_seed) + (ep_idx // len(scenarios)) * _RESET_STRIDE
            episode_frames = _run_episode_frames(
                seed, policy, sc.scenario_id, reset_seed, env_config, bank, selectors,
            )
            need = target - len(collected)
            if len(episode_frames) > need:
                episode_frames = episode_frames[:need]
            collected.extend(episode_frames)
            ep_idx += 1
            # safety valve (should never trigger with 5 x 100-step scenarios)
            if ep_idx > 10_000:
                raise RuntimeError(f"could not reach target {target} for {policy}")
        frames.extend(collected)

    # Exploration warmup: 1666 random-legal transitions (canonical epsilon=1.0 warmup).
    expl: List[WarmupRawTransition] = []
    ep_idx = 0
    rng_key_base = seed * 7919
    while len(expl) < SEEDED_EXPL_COUNT:
        sc = scenarios[ep_idx % len(scenarios)]
        reset_seed = int(sc.environment_seed) + (ep_idx // len(scenarios)) * _RESET_STRIDE
        e_frames = _run_episode_frames(
            seed, "exploration", sc.scenario_id, reset_seed, env_config, bank,
            None, explored_rng_key=rng_key_base + ep_idx,
        )
        need = SEEDED_EXPL_COUNT - len(expl)
        if len(e_frames) > need:
            e_frames = e_frames[:need]
        expl.extend(e_frames)
        ep_idx += 1
        if ep_idx > 10_000:
            raise RuntimeError("exploration warmup did not converge")
    frames.extend(expl)

    if len(frames) != SEEDED_WARMUP_TOTAL:
        raise RuntimeError(f"expected {SEEDED_WARMUP_TOTAL} frames, got {len(frames)}")
    return frames


def write_seeded_warmup_manifest(
    seed: int,
    frames: List[WarmupRawTransition],
    out_root: Path = E3_OUTPUT_ROOT,
) -> Dict[str, Path]:
    """Write the frozen seeded-warmup manifest (raw.jsonl + provenance.json) for a seed."""
    import hashlib

    subdir = out_root / "seeded_warmup_manifests" / f"seed_{seed}"
    subdir.mkdir(parents=True, exist_ok=True)

    raw_path = subdir / "seeded_warmup_raw.jsonl"
    with open(raw_path, "w") as f:
        for fr in frames:
            f.write(json.dumps(asdict(fr)) + "\n")

    counts: Dict[str, int] = {}
    for fr in frames:
        counts[fr.source_policy] = counts.get(fr.source_policy, 0) + 1

    digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    prov = {
        "training_seed": seed,
        "training_split": M9_TRAINING_SPLIT,
        "scenario_bank_path": TRAINING_BANK,
        "total_transitions": len(frames),
        "counts": counts,
        "frozen_mixture": {
            "exact_myopic": SEEDED_M4_COUNT,
            "h2": SEEDED_H2_COUNT,
            "exploration": SEEDED_EXPL_COUNT,
        },
        "prediction_cache_path": str(seed_cache_dir(seed)),
        "manifest_raw_sha256": digest,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    prov_path = subdir / "seeded_warmup_provenance.json"
    prov_path.write_text(json.dumps(prov, indent=2))

    return {"raw": raw_path, "provenance": prov_path}


def load_seeded_warmup_frames(seed: int, out_root: Path = E3_OUTPUT_ROOT) -> List[WarmupRawTransition]:
    """Load a seed's frozen seeded-warmup raw frames back into dataclasses."""
    raw_path = out_root / "seeded_warmup_manifests" / f"seed_{seed}" / "seeded_warmup_raw.jsonl"
    frames = []
    with open(raw_path) as f:
        for line in f:
            frames.append(WarmupRawTransition(**json.loads(line)))
    return frames