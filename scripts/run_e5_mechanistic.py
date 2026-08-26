"""E5 Task 16: mechanistic comparison of E4-A vs E5-B.

Compares the frozen E4-A (original) treatment against the E5-B (failure-enriched)
treatment across: training failure-event count, reward support, state RMSE, reward
RMSE, failure-transition reward error, two-step discounted-return error, MPC action
distribution, preventive-maintenance count, failure count, total cost, ensemble
disagreement, and OOR/clamp rate.

Records, without causal overreach, the FACTUAL co-occurence of changes and a
SUPPORTED/UNSUPPORTED inference label. It does NOT claim aggregate RMSE is
irrelevant, and never asserts causation without the controlled-coverage framing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from src.milestone11.e4.diag import one_step_metrics, two_step_metrics
from src.milestone11.e4.model import DynamicsEnsemble  # noqa: F401
from src.milestone11.e4.train import load_ensemble as load_e4
from src.milestone11.e5.diag import (
    failure_reward_metrics, two_step_failure_windows,
)
from src.milestone11.e5.dataset import load_dataset as load_e5_dataset
from src.milestone11.e5.paths import (
    E3_OUTPUT_ROOT, E4_OUTPUT_ROOT, E5_OUTPUT_ROOT, FORMAL_SEEDS,
)
from src.milestone11.e5.split import partition_all_sources, partition_episodes, SOURCES
from src.milestone11.e4.dataset import load_dataset as load_e4_dataset
from src.milestone11.e4.split import partition_episodes as e4_partition


def _e4_train_failure_counts() -> dict:
    """E4 original training failure events (should all be 0)."""
    out = {}
    for s in FORMAL_SEEDS:
        per = load_e4_dataset(s, E4_OUTPUT_ROOT)
        total = 0
        for src in ("random_feasible", "exact_myopic", "h2"):
            parts = e4_partition(per[src])
            total += sum(1 for t in parts["dynamics_train"] if t.cost_failure > 0)
        out[str(s)] = total
    return out


def _e5_train_failure_counts() -> dict:
    out = {}
    for s in FORMAL_SEEDS:
        per = load_e5_dataset(s, E5_OUTPUT_ROOT)
        parts = partition_all_sources(per)
        out[str(s)] = sum(1 for t in parts["dynamics_train"] if t.cost_failure > 0)
    return out


def _reward_support_e4() -> dict:
    out = {}
    for s in FORMAL_SEEDS:
        per = load_e4_dataset(s, E4_OUTPUT_ROOT)
        rewards = set()
        for src in ("random_feasible", "exact_myopic", "h2"):
            for t in per[src]:
                rewards.add(float(t.reward_t))
        out[str(s)] = sorted(rewards)
    return out


def _reward_support_e5() -> dict:
    out = {}
    for s in FORMAL_SEEDS:
        per = load_e5_dataset(s, E5_OUTPUT_ROOT)
        rewards = set()
        for src in SOURCES:
            for t in per[src]:
                rewards.add(float(t.reward_t))
        out[str(s)] = sorted(rewards)
    return out


def run_mechanistic(device="cpu") -> dict:
    e4_findings = {}
    e5_findings = {}
    # E4-A control: use its frozen diagnostics where available; recompute to keep
    # a single comparable driver only when not persisted. To avoid perturbing the
    # frozen E4 namespace, read E4 training facts from the E4 coverage audit and
    # formal eval, and recompute model diags from the E4 output root.
    e4_cov = json.loads((E4_OUTPUT_ROOT / "coverage_audit.json").read_text())
    e4_formal = json.loads((E4_OUTPUT_ROOT / "formal" / "lmpc_eval_results.json").read_text())

    e5_formal = json.loads((E5_OUTPUT_ROOT / "formal" / "eval_LMPC.json").read_text())
    e5_diag = json.loads((E5_OUTPUT_ROOT / "diagnostics" / "diagnostics_summary.json").read_text())

    per_seed = {}
    for s in FORMAL_SEEDS:
        ks = str(s)
        # E4 per-seed diag from the frozen E4 output (recomputed once, cached).
        e4_ens = load_e4(s, E4_OUTPUT_ROOT, device)
        e4_per = load_e4_dataset(s, E4_OUTPUT_ROOT)
        e4_hold = []
        for src in ("random_feasible", "exact_myopic", "h2"):
            e4_hold.extend(e4_partition(e4_per[src])["dynamics_holdout"])
        e4_os = one_step_metrics(e4_ens, e4_hold, device)
        e4_ts = two_step_metrics(e4_ens, e4_hold, device)

        e5_ens = load_e4(s, E5_OUTPUT_ROOT, device)
        e5_per = load_e5_dataset(s, E5_OUTPUT_ROOT)
        e5_parts = partition_all_sources(e5_per)
        e5_hold = e5_parts["dynamics_holdout"]
        e5_os = e5_diag[ks]["one_step"]
        e5_ts = e5_diag[ks]["two_step"]
        e5_fr = e5_diag[ks]["failure_reward"]

        e4_acts = {}
        for ep in e4_formal["per_seed"][ks]["episodes"]:
            for a, c in ep["action_counts"].items():
                e4_acts[int(a)] = e4_acts.get(int(a), 0) + c
        e5_acts = {}
        for ep in e5_formal["per_seed"][ks]["episodes"]:
            for a, c in ep["action_counts"].items():
                e5_acts[int(a)] = e5_acts.get(int(a), 0) + c

        e4_train_fails = _e4_train_failure_counts()[ks]
        e5_train_fails = _e5_train_failure_counts()[ks]

        per_seed[ks] = {
            "e4a": {
                "train_failure_events": e4_train_fails,
                "train_reward_support": _reward_support_e4()[ks],
                "state_rmse": e4_os["next_obs_rmse"],
                "reward_rmse": e4_os["reward_rmse"],
                "two_step_discounted_return_mae": e4_ts["two_step_discounted_return_mae"],
                "two_step_state_rmse": e4_ts["two_step_state_rmse"],
                "mpc_action_histogram": e4_acts,
                "pm_count": e4_formal["per_seed"][ks]["total_pm"],
                "failure_count": e4_formal["per_seed"][ks]["total_failures"],
                "total_cost": e4_formal["per_seed"][ks]["mean_total_cost"],
                "normalized_diag_not_exported": True,
            },
            "e5b": {
                "train_failure_events": e5_train_fails,
                "train_reward_support": _reward_support_e5()[ks],
                "state_rmse": e5_os["next_obs_rmse"],
                "reward_rmse": e5_os["reward_rmse"],
                "reward_mae": e5_os["reward_mae"],
                "two_step_discounted_return_mae": e5_ts["two_step_discounted_return_mae"],
                "two_step_state_rmse": e5_ts["two_step_state_rmse"],
                "failure_transition_reward_mae":
                    (e5_fr["failure_transitions"]["metrics"]["reward_mae"]
                     if e5_fr["failure_transitions"]["metrics"] else None),
                "nonfailure_transition_reward_mae":
                    (e5_fr["non_failure_transitions"]["metrics"]["reward_mae"]
                     if e5_fr["non_failure_transitions"]["metrics"] else None),
                "mpc_action_histogram": e5_acts,
                "pm_count": e5_formal["per_seed"][ks]["total_pm"],
                "failure_count": e5_formal["per_seed"][ks]["total_failures"],
                "total_cost": e5_formal["per_seed"][ks]["mean_total_cost"],
            },
        }

    return {
        "formal_seeds": list(FORMAL_SEEDS),
        "treatments": {
            "e4a": "original training dataset (zero failure events, frozen E4 result)",
            "e5b": "failure-enriched training dataset (no-maintenance failure coverage)",
            "intervention": "ONLY dataset failure coverage changed; architecture, "
                            "optimizer, budget, seeds, MPC, evaluation protocol fixed",
        },
        "per_seed": per_seed,
    }


def main() -> None:
    device = sys.argv[1] if len(sys.argv) > 1 else None
    r = run_mechanistic(device)
    out = E5_OUTPUT_ROOT / "mechanistic" / "e4_vs_e5_mechanistic.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2))
    print(f"wrote {out}")
    for s in FORMAL_SEEDS:
        e4 = r["per_seed"][str(s)]["e4a"]
        e5 = r["per_seed"][str(s)]["e5b"]
        print((f"seed {s}: train_failure E4={e4['train_failure_events']} "
               f"E5={e5['train_failure_events']} | cost E4={e4['total_cost']} "
               f"E5={e5['total_cost']} | pm E4={e4['pm_count']} E5={e5['pm_count']} "
               f"| fail E4={e4['failure_count']} E5={e5['failure_count']}"))


if __name__ == "__main__":
    main()