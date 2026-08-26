"""E4 (M11) Task 15-16: formal Learned H=2 MPC evaluation on the frozen protocol.

Evaluates the learned-dynamics H=2 MPC policy on the SAME frozen paired
rl_validation protocol used by E3: one episode per scenario of
m5_validation_k2.json, reset seeds [6521..6525], per-seed M9 prediction cache,
cost regime failure-light-no-waste.

The learned model is COMPLETELY FROZEN before evaluation (no updates, no
retraining). Only rl_validation is touched; rl_test is never accessed.

Learned MPC output -> <m11_e4_outputs>/formal/eval_LMPC.json

Frozen comparators (C-DDQN, M4, H2) are taken verbatim from the E3 frozen
paired-eval artifact (m10_e3_outputs/eval/eval_results.json) — they are NOT
re-evaluated here. This keeps E4 on the verified E3 numbers and avoids
introducing any new comparator variance.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import torch

from src.envs.config import get_default_config
from src.envs.scenario_bank import load_scenario_bank
from src.envs.selective_maintenance_env import SelectiveMaintenanceEnv
from src.milestone11.e4.mpc import LearnedMPCPolicy
from src.milestone11.e4.paths import (
    COST_REGIME_ID,
    E4_OUTPUT_ROOT,
    EPISODE_HORIZON,
    E3_OUTPUT_ROOT,
    EVALUATION_BANK,
    EVALUATION_SPLIT,
    MAINTENANCE_CAPACITY,
    seed_cache_dir,
)
from src.milestone11.e4.train import load_ensemble

FORMAL_SEEDS = (6521, 6522, 6523, 6524, 6525)
FIXED_RESET_SEEDS = [6521, 6522, 6523, 6524, 6525]


def build_eval_env(seed: int):
    cache_path = str(seed_cache_dir(seed))
    cfg = get_default_config(
        split=EVALUATION_SPLIT, cost_regime_id=COST_REGIME_ID,
        maintenance_capacity=MAINTENANCE_CAPACITY, scenario_bank_path=EVALUATION_BANK,
        prediction_cache_path=cache_path, seed=seed, info_mode="normal",
    )
    bank = load_scenario_bank(EVALUATION_BANK)
    if bank.split != EVALUATION_SPLIT:
        raise ValueError(f"bank split {bank.split} != {EVALUATION_SPLIT}")
    env = SelectiveMaintenanceEnv(config=cfg, scenario_bank=bank, info_mode="normal")
    return env, [s.scenario_id for s in bank.scenarios]


def evaluate_lmpc(policy: LearnedMPCPolicy, seed: int,
                  ) -> List[Dict[str, Any]]:
    """Run the frozen paired 5-episode rl_validation protocol for Learned MPC."""
    env, scenario_ids = build_eval_env(seed)
    episodes = []
    for i, scenario_id in enumerate(scenario_ids):
        reset_seed = FIXED_RESET_SEEDS[i]
        obs, _info = env.reset(seed=reset_seed, options={"scenario_id": scenario_id})
        total_cost = 0.0
        p_cost = f_cost = w_cost = 0.0
        n_fail = n_pm = 0
        action_counts: Dict[int, int] = {}
        n_plans = 0
        plan_std_sum = 0.0
        sum_oor1 = 0.0
        steps = 0
        while True:
            obs_a = np.asarray(obs, dtype=np.float32)
            if steps >= EPISODE_HORIZON - 1:
                action = policy.select_action_id_last_step(obs_a)
            else:
                action = policy.select_action_id(obs_a)
            lp = policy.last_plan
            action_counts[action] = action_counts.get(action, 0) + 1
            n_plans += 1
            plan_std_sum += lp.selected_sequence_std
            sum_oor1 += lp.raw_out_of_range_fraction_step1
            obs, _r, terminated, truncated, info = env.step(action)
            total_cost += info["total_cost"]
            p_cost += info["preventive_cost"]
            f_cost += info["failure_cost"]
            w_cost += info["wasted_life_cost"]
            n_fail += info["num_failures"]
            n_pm += info["num_preventive"]
            steps += 1
            if truncated or terminated:
                break
        episodes.append({
            "scenario_id": scenario_id, "reset_seed": reset_seed,
            "total_cost": float(total_cost), "preventive_cost": float(p_cost),
            "failure_cost": float(f_cost), "wasted_life_cost": float(w_cost),
            "num_failures": int(n_fail), "num_preventive": int(n_pm),
            "steps": steps, "action_counts": action_counts,
            "n_plans": n_plans,
            "mean_selected_sequence_std": float(plan_std_sum / n_plans) if n_plans else 0.0,
            "mean_raw_oor_step1": float(sum_oor1 / n_plans) if n_plans else 0.0,
        })
    return episodes


def load_frozen_e3_comparators() -> Dict[str, Dict[str, Any]]:
    """Per-policy per-seed mean total cost from the frozen E3 paired eval."""
    path = E3_OUTPUT_ROOT / "eval" / "eval_results.json"
    d = json.loads(path.read_text())
    out: Dict[str, Dict[str, Any]] = {}
    for p in ("A", "C", "M4", "H2"):
        per_seed: Dict[str, float] = {}
        for s in FORMAL_SEEDS:
            row = d["seeds"][str(s)].get(p)
            if row and "mean_total_cost" in row:
                per_seed[str(s)] = float(row["mean_total_cost"])
        out[p] = {"mean_total_cost": float(np.mean(list(per_seed.values()))),
                  "per_seed": per_seed}
    return out


def main() -> None:
    device = sys.argv[1] if len(sys.argv) > 1 else "mps"
    results: Dict[str, Any] = {
        "policy": "learned_h2_mpc",
        "split": EVALUATION_SPLIT,
        "scenario_bank": EVALUATION_BANK,
        "reset_seeds": list(FIXED_RESET_SEEDS),
        "model_frozen": True,
        "model_updated_during_eval": False,
        "rl_test_accessed": False,
        "per_seed": {},
    }
    for seed in FORMAL_SEEDS:
        ensemble = load_ensemble(seed, E4_OUTPUT_ROOT, device)
        policy = LearnedMPCPolicy(ensemble, device=device)
        episodes = evaluate_lmpc(policy, seed)
        costs = [e["total_cost"] for e in episodes]
        results["per_seed"][str(seed)] = {
            "episodes": episodes,
            "mean_total_cost": float(np.mean(costs)),
            "total_failures": int(sum(e["num_failures"] for e in episodes)),
            "total_pm": int(sum(e["num_preventive"] for e in episodes)),
            "n_episodes": len(costs),
        }
    # Cross-seed aggregate
    vals = [results["per_seed"][str(s)]["mean_total_cost"] for s in FORMAL_SEEDS]
    results["overall_mean_total_cost"] = float(np.mean(vals))
    results["frozen_e3_comparators"] = load_frozen_e3_comparators()

    out_dir = E4_OUTPUT_ROOT / "formal"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lmpc_eval_results.json").write_text(
        json.dumps(results, indent=2))
    print(f"wrote {out_dir / 'lmpc_eval_results.json'}")
    print(f"Learned MPC overall mean total cost: {results['overall_mean_total_cost']:.3f}")
    for s in FORMAL_SEEDS:
        print(f"  seed {s}: mean={results['per_seed'][str(s)]['mean_total_cost']:.3f}")


if __name__ == "__main__":
    main()