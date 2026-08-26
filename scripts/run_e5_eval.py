"""E5 Task 14: formal E5-B evaluation on the frozen E4 rl_validation protocol.

Evaluates the E5-B failure-enriched learned-dynamics H=2 MPC on the SAME frozen
paired rl_validation protocol used by E4: one episode per scenario of
m5_validation_k2.json, reset seeds [6521..6525], per-seed M9 prediction cache,
cost regime failure-light-no-waste. The MPC planner (src.milestone11.e4.mpc) and
the evaluation semantics are reused verbatim from E4.

Only rl_validation is touched; rl_test is never accessed. The learned model is
completely frozen during evaluation. No online retraining.

Outputs: <m12_e5_outputs>/formal/eval_LMPC.json (E5-B contingency) and a copy of
the frozen E4-A control evaluation for paired statistics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np

from src.milestone11.e4.mpc import LearnedMPCPolicy
from src.milestone11.e4.paths import (
    COST_REGIME_ID,
    E4_OUTPUT_ROOT,
    EPISODE_HORIZON,
    EVALUATION_BANK,
    EVALUATION_SPLIT,
    MAINTENANCE_CAPACITY,
    seed_cache_dir as e4_seed_cache_dir,
)
from src.milestone11.e4.train import load_ensemble as load_ensemble_e4

from src.milestone11.e5.paths import (
    E5_OUTPUT_ROOT,
    FORMAL_SEEDS,
    seed_cache_dir,
)
from src.envs.config import get_default_config
from src.envs.scenario_bank import load_scenario_bank
from src.envs.selective_maintenance_env import SelectiveMaintenanceEnv

FIXED_RESET_SEEDS = [6521, 6522, 6523, 6524, 6525]


def evaluate_lmpc(policy: LearnedMPCPolicy, seed: int) -> list[dict]:
    """Run the frozen paired 5-episode rl_validation protocol for Learned MPC."""
    cache_path = str(seed_cache_dir(seed))
    cfg = get_default_config(
        split=EVALUATION_SPLIT, cost_regime_id=COST_REGIME_ID,
        maintenance_capacity=MAINTENANCE_CAPACITY, scenario_bank_path=EVALUATION_BANK,
        prediction_cache_path=cache_path, seed=seed, info_mode="normal",
    )
    bank = load_scenario_bank(EVALUATION_BANK)
    assert bank.split == EVALUATION_SPLIT
    env = SelectiveMaintenanceEnv(config=cfg, scenario_bank=bank, info_mode="normal")
    scenario_ids = [s.scenario_id for s in bank.scenarios]
    episodes = []
    for i, scenario_id in enumerate(scenario_ids):
        reset_seed = FIXED_RESET_SEEDS[i]
        obs, _info = env.reset(seed=reset_seed, options={"scenario_id": scenario_id})
        total_cost = p_cost = f_cost = w_cost = 0.0
        n_fail = n_pm = 0
        action_counts = {}
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
            "mean_selected_sequence_std":
                float(plan_std_sum / max(1, steps)),
            "mean_raw_oor_step1": float(sum_oor1 / max(1, steps)),
        })
    return episodes


def load_frozen_e4_control() -> dict:
    """Frozen E4-A original Learned-MPC result from m11_e4_outputs.

    E5-A control is EXACTLY the frozen E4 result; it is never retrained.
    """
    d = json.loads((E4_OUTPUT_ROOT / "formal" / "lmpc_eval_results.json").read_text())
    out = {"policy": "learned_h2_mpc_e4_control", "per_seed": {}, "overall": {},
           "frozen_e3_comparators": d.get("frozen_e3_comparators", {})}
    for s in FORMAL_SEEDS:
        v = d["per_seed"][str(s)]
        out["per_seed"][str(s)] = {
            "mean_total_cost": float(v["mean_total_cost"]),
            "total_failures": int(v["total_failures"]),
            "total_pm": int(v["total_pm"]),
            "episodes": v["episodes"],
        }
    out["overall_mean_total_cost"] = float(d["overall_mean_total_cost"])
    return out


def main() -> None:
    device = sys.argv[1] if len(sys.argv) > 1 else "mps"
    results = {
        "policy": "learned_h2_mpc_e5b_failure_enriched",
        "split": EVALUATION_SPLIT,
        "scenario_bank": EVALUATION_BANK,
        "reset_seeds": list(FIXED_RESET_SEEDS),
        "model_frozen": True,
        "model_updated_during_eval": False,
        "rl_test_accessed": False,
        "protective_intervention": "dataset_failure_coverage",
        "controller_is_frozen_e4_mpc": True,
        "per_seed": {},
    }
    for seed in FORMAL_SEEDS:
        ensemble = load_ensemble_e4(seed, E5_OUTPUT_ROOT, device)
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
    vals = [results["per_seed"][str(s)]["mean_total_cost"] for s in FORMAL_SEEDS]
    results["overall_mean_total_cost"] = float(np.mean(vals))
    results["e4_control"] = load_frozen_e4_control()

    out_dir = E5_OUTPUT_ROOT / "formal"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_LMPC.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out_dir / 'eval_LMPC.json'}")
    print(f"E5-B Learned MPC overall mean total cost: {results['overall_mean_total_cost']:.3f}")
    for s in FORMAL_SEEDS:
        print(f"  seed {s}: mean={results['per_seed'][str(s)]['mean_total_cost']:.3f}")


if __name__ == "__main__":
    main()