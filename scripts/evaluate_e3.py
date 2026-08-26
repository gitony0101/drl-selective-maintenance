"""E3 (M10) frozen rl_validation evaluation driver (Step 10).

Evaluates greedy policies on the frozen paired 5-episode rl_validation protocol
(one episode per scenario_id x FIXED_RESET_SEEDS=[6521..6525], matching the M9
corrected paired report). Evaluates:

  - each E3 cell x seed best checkpoint (online network, greedy, eps=0)
  - M4 exact_myopic
  - H2 m6_h2_v1
  - (optionally the historical M9 best checkpoints for the A reference)

Writes per-episode + per-seed aggregates to ``<m10_e3_outputs>/eval/``.
NEVER accesses rl_test.
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

from src.agents.ddqn.agent import DDQNAgent, DDQNAgentConfig
from src.envs.config import get_default_config
from src.envs.scenario_bank import load_scenario_bank
from src.envs.selective_maintenance_env import SelectiveMaintenanceEnv
from src.milestone10.e3.h2_context import seed_cache_dir
from src.milestone10.e3.trajectories import M4ExactMyopicSelector, H2PlannerSelector

FORMAL_SEEDS = (6521, 6522, 6523, 6524, 6525)
FIXED_RESET_SEEDS = [6521, 6522, 6523, 6524, 6525]

from src.runtime_paths import external_root as _external_root

_CONTAINER = _external_root()
E3_OUTPUT = _CONTAINER / "m10_e3_outputs"
FORMAL_ROOT = E3_OUTPUT / "formal"
EVAL_ROOT = E3_OUTPUT / "eval"

BR = "rl_validation"
VAL_BANK = "configs/scenarios/m5_validation_k2.json"


def build_eval_env(seed: int):
    """Build an rl_validation env bound to ``seed``'s per-seed cache."""
    cache_path = str(seed_cache_dir(seed))
    cfg = get_default_config(
        split=BR, cost_regime_id="failure-light-no-waste",
        maintenance_capacity=2, scenario_bank_path=VAL_BANK,
        prediction_cache_path=cache_path, seed=seed, info_mode="normal",
    )
    bank = load_scenario_bank(VAL_BANK)
    env = SelectiveMaintenanceEnv(config=cfg, scenario_bank=bank, info_mode="normal")
    return env, [s.scenario_id for s in bank.scenarios]


def evaluate_greedy_agent(
    agent: DDQNAgent, seed: int, cache_path: str,
) -> List[Dict[str, Any]]:
    env, scenario_ids = build_eval_env(seed)
    episodes = []
    for i, scenario_id in enumerate(scenario_ids):
        reset_seed = FIXED_RESET_SEEDS[i]
        obs, _info = env.reset(seed=reset_seed, options={"scenario_id": scenario_id})
        total_cost = 0.0
        p_cost = f_cost = w_cost = 0.0
        n_fail = n_pm = 0
        action_counts: Dict[int, int] = {}
        steps = 0
        while True:
            action = agent.evaluate_action(obs)
            action_counts[action] = action_counts.get(action, 0) + 1
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
        })
    return episodes


def load_best_agent(cell: str, seed: int) -> DDQNAgent:
    """Load a cell x seed best checkpoint's online network into a fresh agent."""
    ckpt_path = FORMAL_ROOT / cell / f"seed_{seed}" / "checkpoint_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"no best checkpoint: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = DDQNAgentConfig(num_actions=16, observation_dim=10)
    agent = DDQNAgent(config=cfg, seed=seed)
    agent.online_network.load_state_dict(sd["online_network_state_dict"])
    return agent


def build_m4_agent():
    m = M4ExactMyopicSelector("failure-light-no-waste")

    class _Sel:
        def evaluate_action(self, obs):
            return m.select_action_id(np.asarray(obs, dtype=np.float32))
    return _Sel()


def build_h2_agent(seed: int):
    h = H2PlannerSelector(seed, "failure-light-no-waste")

    class _Sel:
        def evaluate_action(self, obs):
            return h.select_action_id(np.asarray(obs, dtype=np.float32))
    return _Sel()


def main() -> None:
    import datetime
    results: Dict[str, Any] = {"generated_at_utc": None, "seeds": {},
                               "per_seed": {}}
    cells = ["A", "B", "C", "D"]
    agent_builders = {
        "M4": lambda s: build_m4_agent(),
        "H2": lambda s: build_h2_agent(s),
    }
    for cell in cells:
        agent_builders[cell] = lambda s, c=cell: load_best_agent(c, s)

    for seed in FORMAL_SEEDS:
        results["seeds"][str(seed)] = {}
        for policy in cells + ["M4", "H2"]:
            # M4/H2 do not carry a per-seed cache pairing in the same way; use the seed's cache.
            cache_path = str(seed_cache_dir(seed))
            try:
                ag = agent_builders[policy](seed)
                eps = evaluate_greedy_agent(ag, seed, cache_path)
            except FileNotFoundError as e:
                results["seeds"][str(seed)][policy] = {"error": str(e)}
                continue
            costs = [e["total_cost"] for e in eps]
            results["seeds"][str(seed)][policy] = {
                "episodes": eps,
                "mean_total_cost": float(np.mean(costs)),
                "total_failures": int(sum(e["num_failures"] for e in eps)),
                "total_pm": int(sum(e["num_preventive"] for e in eps)),
                "n_episodes": len(costs),
            }

    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_ROOT / "eval_results.json"
    results["generated_at_utc"] = datetime.datetime.utcnow().isoformat() + "Z"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}")

    # Per-policy cross-seed means.
    cross = {}
    for policy in cells + ["M4", "H2"]:
        vals = []
        for seed in FORMAL_SEEDS:
            row = results["seeds"][str(seed)].get(policy)
            if row and "mean_total_cost" in row:
                vals.append(row["mean_total_cost"])
        cross[policy] = {
            "mean_total_cost": float(np.mean(vals)) if vals else None,
            "per_seed": {str(s): results["seeds"][str(s)][policy]["mean_total_cost"]
                         for s in FORMAL_SEEDS
                         if results["seeds"][str(s)].get(policy, {}).get("mean_total_cost") is not None},
        }
    print("=== cross-seed mean per-episode total cost ===")
    for p, v in cross.items():
        print(f"  {p}: mean={v['mean_total_cost']} per_seed={v['per_seed']}")


if __name__ == "__main__":
    main()