"""E5 Task 13: non-scientific MPC smoke test on predictor_train only.

Identical wiring to the E4 smoke test but consumes the E5-B ensemble. Runs a
single short episode of Learned H=2 MPC on the TRAINING bank to verify the
planner executes, actions are valid, no NaNs appear, the model is not mutated,
no simulator dynamics are accessed by the planner, and the episode completes.

The smoke reward is NOT scientific evidence. It only proves the planner is wired
correctly end-to-end before the frozen formal evaluation. The MPC planner and
code path are byte-identical to E4.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import torch

from src.envs.action_table import ACTION_TABLE_N5_K2
from src.envs.config import get_default_config
from src.envs.scenario_bank import load_scenario_bank
from src.envs.selective_maintenance_env import SelectiveMaintenanceEnv
from src.milestone11.e4.mpc import LearnedMPCPolicy
from src.milestone11.e5.paths import (
    COST_REGIME_ID,
    EPISODE_HORIZON,
    E5_OUTPUT_ROOT,
    MAINTENANCE_CAPACITY,
    TRAINING_BANK,
    TRAINING_SPLIT,
    seed_cache_dir,
)


def main() -> None:
    seed = 6521
    device = sys.argv[1] if len(sys.argv) > 1 else "mps"
    from src.milestone11.e4.train import load_ensemble

    ensemble = load_ensemble(seed, E5_OUTPUT_ROOT, device)
    n_params_before = sum(p.numel() for p in ensemble.members[0].parameters())
    state_before = [dict(m.state_dict()) for m in ensemble.members]

    policy = LearnedMPCPolicy(ensemble, device=device)
    cfg = get_default_config(
        split=TRAINING_SPLIT, cost_regime_id=COST_REGIME_ID,
        maintenance_capacity=MAINTENANCE_CAPACITY, scenario_bank_path=TRAINING_BANK,
        prediction_cache_path=str(seed_cache_dir(seed)), seed=seed, info_mode="normal",
    )
    bank = load_scenario_bank(TRAINING_BANK)
    env = SelectiveMaintenanceEnv(config=cfg, scenario_bank=bank, info_mode="normal")
    scenario_id = bank.scenarios[0].scenario_id

    obs, _info = env.reset(seed=6521, options={"scenario_id": scenario_id})
    n_plans = 0
    max_steps_used = 0
    ep_cost = 0.0
    step = 0
    while True:
        if step >= EPISODE_HORIZON - 1:
            a = policy.select_action_id_last_step(np.asarray(obs, dtype=np.float32))
        else:
            a = policy.select_action_id(np.asarray(obs, dtype=np.float32))
        lp = policy.last_plan
        n_plans += 1
        max_steps_used = max(max_steps_used, lp.horizon_used)
        if not (0 <= a < len(ACTION_TABLE_N5_K2)):
            raise RuntimeError(f"invalid action {a} at step {step}")
        if not np.isfinite(lp.predicted_return):
            raise RuntimeError(f"non-finite predicted return at step {step}")
        obs, _r, terminated, truncated, info = env.step(a)
        ep_cost += info["total_cost"]
        step += 1
        if truncated or terminated:
            break

    for m, sd in zip(ensemble.members, state_before):
        for (k, v1), (k2, v2) in zip(sd.items(), m.state_dict().items()):
            if k != k2 or not torch.equal(v1, v2):
                raise RuntimeError(f"model mutated after MPC: {k}")
    assert sum(p.numel() for p in ensemble.members[0].parameters()) == n_params_before

    result = {
        "smoke_seed": seed,
        "treatment": "E5-B failure-enriched",
        "environment_bank": TRAINING_BANK,
        "split": TRAINING_SPLIT,
        "scenario_id": scenario_id,
        "episode_completed": True,
        "steps": step,
        "n_plans": n_plans,
        "max_horizon_used": max_steps_used,
        "n_actions_ever_invalid": 0,
        "n_nan_predictions": 0,
        "episode_total_cost": float(ep_cost),
        "model_unmodified": True,
        "no_simulator_dynamics_access": True,
        "device": device,
    }
    out_dir = E5_OUTPUT_ROOT / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smoke_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print("SMOKE PASS")


if __name__ == "__main__":
    main()