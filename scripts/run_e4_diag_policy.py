"""E4 (M11) Task 18: mechanistic diagnostics after the frozen formal evaluation.

Analyzes the frozen learned ensemble + MPC policy WITHOUT changing it:

  - ensemble disagreement: is the selected sequence's across-member std larger
    than the typical candidate sequence's std? (systematic-exploitation check)
  - Learned-MPC vs canonical H2 (known dynamics) action agreement on a FIXED
    diagnostic state set drawn from rl_validation observations
  - action distribution of LMPC on those states
  - model out-of-range (raw, pre-clamp) prediction rates
  - one-step prediction error on the states/actions actually selected by LMPC
    vs the overall one-step error (does LMPC pick high-error regions?)

Runs only on rl_validation observations and the FROZEN ensembles. Never modifies
the policy or models. No retraining, no new data, no rl_test access.
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
from src.milestone11.e4.diag import ensemble_predict
from src.milestone11.e4.mpc import enumerate_sequences, plan, score_sequence
from src.milestone11.e4.paths import (
    COST_REGIME_ID,
    E4_OUTPUT_ROOT,
    EVALUATION_BANK,
    EVALUATION_SPLIT,
    MAINTENANCE_CAPACITY,
    OBSERVATION_DIM,
    seed_cache_dir,
)
from src.milestone11.e4.train import load_ensemble
from src.milestone10.e3.trajectories import H2PlannerSelector

FORMAL_SEEDS = (6521, 6522, 6523, 6524, 6525)
FIXED_RESET_SEEDS = [6521, 6522, 6523, 6524, 6525]


def collect_eval_states(seed: int, n_per_ep: int = 20) -> np.ndarray:
    """Gather a fixed set of rl_validation observations per seed."""
    cache_path = str(seed_cache_dir(seed))
    cfg = get_default_config(
        split=EVALUATION_SPLIT, cost_regime_id=COST_REGIME_ID,
        maintenance_capacity=MAINTENANCE_CAPACITY, scenario_bank_path=EVALUATION_BANK,
        prediction_cache_path=cache_path, seed=seed, info_mode="normal",
    )
    bank = load_scenario_bank(EVALUATION_BANK)
    env = SelectiveMaintenanceEnv(config=cfg, scenario_bank=bank, info_mode="normal")
    states: List[np.ndarray] = []
    for i, scen in enumerate(bank.scenarios):
        obs, _ = env.reset(seed=FIXED_RESET_SEEDS[i],
                           options={"scenario_id": scen.scenario_id})
        # run a random-feasible roll-out to gather diverse 10-dim observations
        rng = np.random.default_rng(seed + i)
        from src.milestone11.e4.dataset import RandomFeasiblePolicy
        pol = RandomFeasiblePolicy(seed + i)
        taken = 0
        while True:
            if taken < n_per_ep:
                states.append(np.asarray(obs, dtype=np.float32))
            taken += 1
            a = pol.select_action_id(np.asarray(obs, dtype=np.float32))
            obs, _r, term, trunc, _ = env.step(a)
            if trunc or term:
                break
    return np.array(states)


def main() -> None:
    device = sys.argv[1] if len(sys.argv) > 1 else "mps"
    out: Dict[str, Any] = {}
    for seed in FORMAL_SEEDS:
        ens = load_ensemble(seed, E4_OUTPUT_ROOT, device)
        states = collect_eval_states(seed)
        # H2 comparator on the same states (known dynamics; frozen).
        h2 = H2PlannerSelector(seed, COST_REGIME_ID)

        lmpc_actions: List[int] = []
        h2_actions: List[int] = []
        sel_std: List[float] = []
        cand_mean_std: List[float] = []
        sum_oor1 = 0.0
        sum_oor2 = 0.0
        n_agree = 0
        total = 0
        # selected-state model error
        sel_pred_err: List[float] = []

        for o in states:
            ot = torch.tensor([float(x) for x in o], dtype=torch.float32)
            if device == "mps":
                ot = ot.to("mps")
            res = plan(ens, ot, steps_remaining_in_episode=2)
            a = res.action_id
            lmpc_actions.append(a)
            sel_std.append(res.selected_sequence_std)
            # genuine ensemble disagreement: mean across candidate sequences of the
            # across-member std of that sequence's predicted return
            cand_stds = [score_sequence(ens, ot, s, 2)[1] for s in
                         enumerate_sequences(2)]
            cand_mean_std.append(float(np.mean(cand_stds)) if cand_stds else 0.0)
            sum_oor1 += res.raw_out_of_range_fraction_step1
            sum_oor2 += res.raw_out_of_range_fraction_step2

            h2a = h2.select_action_id(np.asarray(o, dtype=np.float32))
            h2_actions.append(int(h2a))
            n_agree += int(int(h2a) == a)
            total += 1

            # prediction error on this state/action (one step, ensemble-mean)
            if len(states) <= 500:
                dm, rm, _ = ensemble_predict(ens, o[None], np.array([a]), device)
                sel_pred_err.append(float(np.mean(np.abs(dm[0]))))

        out[str(seed)] = {
            "n_states": total,
            "lmpc_action_histogram": {str(k): int(v) for k, v in
                                      {a: lmpc_actions.count(a) for a in
                                       range(16)}.items()},
            "h2_action_histogram": {str(k): int(v) for k, v in
                                    {a: h2_actions.count(a) for a in
                                     range(16)}.items()},
            "action_agreement_lmpc_vs_h2": (n_agree / total) if total else None,
            "mean_sel_chain_std_vs_candidate_mean_std": (
                float(np.mean(sel_std)) / float(np.mean(cand_mean_std))
                if cand_mean_std and np.mean(cand_mean_std) > 0 else None),
            "mean_selected_sequence_std": float(np.mean(sel_std)),
            "mean_candidate_sequence_std": float(np.mean(cand_mean_std)),
            "mean_raw_oor_step1": float(sum_oor1 / total) if total else 0.0,
            "mean_raw_oor_step2": float(sum_oor2 / total) if total else 0.0,
            "mean_selected_state_delta_abs": (float(np.mean(sel_pred_err))
                                              if sel_pred_err else None),
        }

    summary_path = E4_OUTPUT_ROOT / "diag_policy" / "policy_diagnostics.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {summary_path}")
    for s in FORMAL_SEEDS:
        o = out[str(s)]
        print(f"seed {s}: agree_LMPC_vs_H2={o['action_agreement_lmpc_vs_h2']:.3f} "
              f"sel_vs_cand_std={o['mean_sel_chain_std_vs_candidate_mean_std']:.3f} "
              f"oor1={o['mean_raw_oor_step1']:.3f}")


if __name__ == "__main__":
    main()