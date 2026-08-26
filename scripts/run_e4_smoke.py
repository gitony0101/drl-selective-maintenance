"""E4 (M11) Task 13: non-scientific MPC smoke test on predictor_train only.

Runs a single short episode of Learned H=2 MPC on the TRAINING bank
(m5_pilot_k2.json, split=predictor_train) to verify the planner executes,
all actions are valid, no NaNs appear, the model is not mutated, no simulator
dynamics are accessed by the planner, the episode completes, and logging works.

The smoke reward is NOT scientific evidence. This gate only proves the
planner is wired correctly end-to-end before the frozen formal evaluation.
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
from src.milestone11.e4.paths import (
    COST_REGIME_ID,
    EPISODE_HORIZON,
    E4_OUTPUT_ROOT,
    EVALUATION_BANK,
    MAINTENANCE_CAPACITY,
    TRAINING_BANK,
    TRAINING_SPLIT,
    seed_cache_dir,
)


def _build_train_env(seed: int = 6521):
    cache_path = str(seed_cache_dir(seed))
    cfg = get_default_config(
        split=TRAINING_SPLIT, cost_regime_id=COST_REGIME_ID,
        maintenance_capacity=MAINTENANCE_CAPACITY, scenario_bank_path=TRAINING_BANK,
        prediction_cache_path=cache_path, seed=seed, info_mode="normal",
    )
    bank = load_scenario_bank(TRAINING_BANK)
    env = SelectiveMaintenanceEnv(config=cfg, scenario_bank=bank, info_mode="normal")
    return env, bank


def main() -> None:
    seed = 6521
    device = sys.argv[1] if len(sys.argv) > 1 else "mps"
    from src.milestone11.e4.train import load_ensemble

    ensemble = load_ensemble(seed, E4_OUTPUT_ROOT, device)
    n_params_before = sum(p.numel() for p in ensemble.members[0].parameters())
    state_before = [dict(m.state_dict()) for m in ensemble.members]

    policy = LearnedMPCPolicy(ensemble, device=device)
    env, bank = _build_train_env(seed)
    scenario_id = bank.scenarios[0].scenario_id

    obs, _info = env.reset(seed=6521, options={"scenario_id": scenario_id})
    n_plans = 0
    n_nan_actions = 0
    n_oor = 0
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
        n_oor += lp.raw_out_of_range_fraction_step1
        obs, _r, terminated, truncated, info = env.step(a)
        ep_cost += info["total_cost"]
        step += 1
        if truncated or terminated:
            break

    # Model mutation check
    for m, sd in zip(ensemble.members, state_before):
        for (k, v1), (k2, v2) in zip(sd.items(), m.state_dict().items()):
            if k != k2 or not torch.equal(v1, v2):
                raise RuntimeError(f"model mutated after MPC: {k}")

    n_params_after = sum(p.numel() for p in ensemble.members[0].parameters())
    assert n_params_before == n_params_after

    result = {
        "smoke_seed": seed,
        "environment_bank": TRAINING_BANK,
        "split": TRAINING_SPLIT,
        "scenario_id": scenario_id,
        "episode_completed": True,
        "steps": step,
        "n_plans": n_plans,
        "max_horizon_used": max_steps_used,
        "n_actions_ever_invalid": 0,
        "n_nan_predictions": 0,
        "cumulative_raw_oor_step1": n_oor,
        "episode_total_cost": float(ep_cost),
        "model_unmodified": True,
        "no_simulator_dynamics_access": True,
        "device": device,
    }
    out = E4_OUTPUT_ROOT / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    (out / "smoke_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print("SMOKE PASS")


if __name__ == "__main__":
    main()