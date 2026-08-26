"""E4 (M11) Task 19 (SECONDARY DIAGNOSTIC, not primary E4): Learned-MPC H=1.

Uses the SAME frozen learned ensembles (no retraining, no new data, no tuning).
At each step the H=1 planner selects

    a* = argmax_a (1/M) sum_m r_hat_m(o, a)

i.e. the action maximizing the ensemble-mean IMMEDIATE predicted reward. This
isolates the reward-side of the learned model and reveals whether H=2 lookahead
adds decision value over one-step learned-model decision making.

Evaluated under the SAME frozen paired rl_validation protocol as the primary E4
(5 episodes/seed, reset seeds 6521..6525, m5_validation_k2.json, per-seed cache).
Label clearly as SECONDARY DIAGNOSTIC. rl_test is never accessed.
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
from src.milestone11.e4.paths import (
    COST_REGIME_ID,
    E4_OUTPUT_ROOT,
    EPISODE_HORIZON,
    EVALUATION_BANK,
    EVALUATION_SPLIT,
    MAINTENANCE_CAPACITY,
    NUM_ACTIONS,
    OBSERVATION_DIM,
    seed_cache_dir,
)
from src.milestone11.e4.train import load_ensemble

FORMAL_SEEDS = (6521, 6522, 6523, 6524, 6525)
FIXED_RESET_SEEDS = [6521, 6522, 6523, 6524, 6525]


class LearnedH1Policy:
    """One-step learned reward argmax (ensemble-mean immediate reward)."""

    def __init__(self, ensemble, device="cpu") -> None:
        self.ensemble = ensemble.eval()
        self.device = device

    def select_action_id(self, observation) -> int:
        o = torch.tensor([float(x) for x in observation], dtype=torch.float32)
        if self.device == "mps":
            o = o.to("mps")
        onehot = torch.nn.functional.one_hot(
            torch.arange(NUM_ACTIONS, device=o.device),
            num_classes=NUM_ACTIONS).float()          # [16,16]
        x = torch.cat([o.repeat(NUM_ACTIONS, 1), onehot], dim=1)  # [16,26]
        r_mean = torch.zeros(NUM_ACTIONS, device=o.device)
        for m in range(self.ensemble.M):
            d, r = self.ensemble.predict_member(m, x)
            r_mean = r_mean + r
        r_mean = r_mean / self.ensemble.M
        a = int(torch.argmax(r_mean).item())          # lowest-id tie-break
        return a


def build_eval_env(seed: int):
    cfg = get_default_config(
        split=EVALUATION_SPLIT, cost_regime_id=COST_REGIME_ID,
        maintenance_capacity=MAINTENANCE_CAPACITY, scenario_bank_path=EVALUATION_BANK,
        prediction_cache_path=str(seed_cache_dir(seed)), seed=seed, info_mode="normal",
    )
    bank = load_scenario_bank(EVALUATION_BANK)
    env = SelectiveMaintenanceEnv(config=cfg, scenario_bank=bank, info_mode="normal")
    return env, [s.scenario_id for s in bank.scenarios]


def main() -> None:
    device = sys.argv[1] if len(sys.argv) > 1 else "mps"
    out: Dict[str, Any] = {"policy": "learned_h1_mpc_secondary", "per_seed": {}}
    for seed in FORMAL_SEEDS:
        ens = load_ensemble(seed, E4_OUTPUT_ROOT, device)
        pol = LearnedH1Policy(ens, device)
        env, scenario_ids = build_eval_env(seed)
        episodes = []
        for i, scen_id in enumerate(scenario_ids):
            obs, _ = env.reset(seed=FIXED_RESET_SEEDS[i],
                               options={"scenario_id": scen_id})
            total_cost = 0.0
            p_cost = f_cost = w_cost = 0.0
            n_fail = n_pm = 0
            action_counts: Dict[int, int] = {}
            steps = 0
            while True:
                a = pol.select_action_id(np.asarray(obs, dtype=np.float32))
                action_counts[a] = action_counts.get(a, 0) + 1
                obs, _r, term, trunc, info = env.step(a)
                total_cost += info["total_cost"]
                p_cost += info["preventive_cost"]
                f_cost += info["failure_cost"]
                w_cost += info["wasted_life_cost"]
                n_fail += info["num_failures"]
                n_pm += info["num_preventive"]
                steps += 1
                if trunc or term:
                    break
            episodes.append({
                "scenario_id": scen_id, "total_cost": float(total_cost),
                "num_failures": int(n_fail), "num_preventive": int(n_pm),
                "steps": steps, "action_counts": action_counts,
            })
        costs = [e["total_cost"] for e in episodes]
        out["per_seed"][str(seed)] = {
            "episodes": episodes,
            "mean_total_cost": float(np.mean(costs)),
            "total_failures": int(sum(e["num_failures"] for e in episodes)),
            "total_pm": int(sum(e["num_preventive"] for e in episodes)),
        }
    vals = [out["per_seed"][str(s)]["mean_total_cost"] for s in FORMAL_SEEDS]
    out["overall_mean_total_cost"] = float(np.mean(vals))
    p = E4_OUTPUT_ROOT / "diag_h1" / "h1_secondary_results.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p}")
    print(f"Learned H=1 (secondary diagnostic) overall mean total cost: "
          f"{out['overall_mean_total_cost']:.3f}")
    for s in FORMAL_SEEDS:
        print(f"  seed {s}: mean={out['per_seed'][str(s)]['mean_total_cost']:.3f}")


if __name__ == "__main__":
    main()