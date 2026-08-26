"""Aggregate the five M9 point-estimate formal result manifests into the paired
report the M9 contract (§line 133) requires:

  - per-seed: DDQN episode cost mean/median + std; failures; catastrophic-
    episode rate; wasted life; action distribution; Q-value diagnostics
    (q_values_mean trajectory); gradient norms; comparison with M4 exact-myopic
    + the 5 heuristic baselines (apples-to-apples mean_total_cost).
  - paired five-seed statistics: mean/median/std across seeds; CIs where
    supported; DDQN-vs-each-baseline paired deltas.

Reads each seed's ``m9_point_result_manifest.json`` (written by
``scripts/run_m9_point_estimate.py``) + the trainer-written
``run_manifest.json`` / ``validation_metrics.json`` /
``training_metrics.jsonl`` / ``episode_metrics.csv`` siblings.

USAGE:
  python scripts/m9_point_aggregate.py --phase formal
  python scripts/m9_point_aggregate.py --phase pilot   # sanity on the 2 pilot seeds

External git-ignored inputs:
  <container>/m9_point_runs/<phase>/<run_id>/  — per-seed run dir.
External git-ignored output:
  <container>/m9_point_runs/<phase>/aggregate_report.json  — the paired report.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.milestone9.point import pairing

_CONTAINER_ROOT = pairing._CONTAINER_ROOT
_RUNS_ROOT = _CONTAINER_ROOT / "m9_point_runs"

_SEEDS = [6521, 6522, 6523, 6524, 6525]
_CATASTROPHIC_COST_THRESHOLD = 50.0  # episodes with cost >= this are catastrophic


def _load_per_seed(phase: str, seed: int) -> Dict[str, Any]:
    """Load one seed's run artifacts into a structured record."""
    run_id = pairing.run_id_for_seed(seed)
    run_dir = _RUNS_ROOT / phase / run_id
    result_path = run_dir / "m9_point_result_manifest.json"
    if not result_path.exists():
        raise FileNotFoundError(
            f"seed {seed} result manifest missing: {result_path} -- run "
            f"scripts/run_m9_point_estimate.py --seed {seed} --phase {phase} first"
        )
    rec: Dict[str, Any] = {"seed": seed, "run_dir": str(run_dir)}
    result = json.loads(result_path.read_text())

    # DDQN validation metrics (mean_total_cost over 5 episodes).
    vm_path = run_dir / "validation_metrics.json"
    if vm_path.exists():
        vm = json.loads(vm_path.read_text())
        if vm:
            last = vm[-1]
            rec["ddqn_validation"] = {
                "global_step": last.get("global_step"),
                "mean_total_cost": last.get("mean_total_cost"),
                "std_total_cost": last.get("std_total_cost"),
                "num_episodes": last.get("num_episodes"),
                "total_failures": last.get("total_failures"),
                "total_pm_actions": last.get("total_pm_actions"),
                "worst_10_pct_cost": last.get("worst_10_pct_cost"),
                "mean_episode_return": last.get("mean_episode_return"),
            }

    # Episode metrics (per-episode cost breakdown from the trainer).
    ep_csv = run_dir / "episode_metrics.csv"
    if ep_csv.exists():
        episodes: List[Dict[str, float]] = []
        with ep_csv.open() as f:
            for row in csv.DictReader(f):
                episodes.append({
                    "episode": int(row["episode"]),
                    "return_": float(row["return"]),
                    "length": int(row["length"]),
                    "total_cost": float(row["total_cost"]),
                    "preventive_cost": float(row["preventive_cost"]),
                    "failure_cost": float(row["failure_cost"]),
                    "wasted_life_cost": float(row["wasted_life_cost"]),
                })
        if episodes:
            costs = [e["total_cost"] for e in episodes]
            wasted = [e["wasted_life_cost"] for e in episodes]
            cat = sum(1 for c in costs if c >= _CATASTROPHIC_COST_THRESHOLD)
            rec["ddqn_training_episodes"] = {
                "n_episodes": len(episodes),
                "mean_total_cost": statistics.mean(costs),
                "median_total_cost": statistics.median(costs),
                "std_total_cost": statistics.pstdev(costs) if len(costs) > 1 else 0.0,
                "catastrophic_episode_rate": cat / len(costs),
                "mean_wasted_life_cost": statistics.mean(wasted),
            }

    # Training metrics (Q-values, grad norms, td_loss over time).
    tm_path = run_dir / "training_metrics.jsonl"
    if tm_path.exists():
        rows: List[Dict[str, float]] = []
        with tm_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if rows:
            grad_norms = [r.get("grad_norm", 0.0) for r in rows if r.get("grad_norm") is not None]
            q_means = [r.get("q_values_mean", 0.0) for r in rows if r.get("q_values_mean") is not None]
            td_losses = [r.get("td_loss", 0.0) for r in rows if r.get("td_loss") is not None]
            rec["ddqn_training_diagnostics"] = {
                "n_logged_steps": len(rows),
                "final_grad_norm": grad_norms[-1] if grad_norms else None,
                "max_grad_norm": max(grad_norms) if grad_norms else None,
                "final_q_values_mean": q_means[-1] if q_means else None,
                "final_td_loss": td_losses[-1] if td_losses else None,
                "all_finite": all(
                    abs(v) < float("inf") for v in grad_norms + q_means + td_losses
                ),
            }

    # Baselines (apples-to-apples, same cache/bank/horizon/reset_seeds).
    rec["baselines"] = result.get("baselines", {})

    # Run manifest provenance (frozen checkpoint identity, cache pairing).
    rm_path = run_dir / "run_manifest.json"
    if rm_path.exists():
        rm = json.loads(rm_path.read_text())
        ckpt_best_field = rm.get("checkpoint_best")
        # checkpoint_best is a path string (run_manifest.json schema); the
        # SHA256 is in the result manifest's run_artifacts block instead.
        ckpt_best_path = ckpt_best_field if isinstance(ckpt_best_field, str) else None
        rec["training_provenance"] = {
            "status": rm.get("status"),
            "final_global_step": rm.get("final_global_step"),
            "max_steps": rm.get("max_steps"),
            "validation_performed": rm.get("validation_performed"),
            "training_seed": rm.get("training_seed"),
            "validation_split": rm.get("validation_split"),
            "checkpoint_best_path": ckpt_best_path,
            "checkpoint_best_sha256": (
                result.get("run_artifacts", {})
                .get("checkpoint_best.pt", {})
                .get("sha256")
            ),
        }
    return rec


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def aggregate(phase: str, seeds: List[int]) -> Dict[str, Any]:
    """Build the paired five-seed report."""
    per_seed = {s: _load_per_seed(phase, s) for s in seeds}

    # DDQN validation across seeds.
    ddqn_costs = [per_seed[s]["ddqn_validation"]["mean_total_cost"]
                  for s in seeds if "ddqn_validation" in per_seed[s]]
    ddqn_failures = [per_seed[s]["ddqn_validation"]["total_failures"]
                     for s in seeds if "ddqn_validation" in per_seed[s]]
    ddqn_pm = [per_seed[s]["ddqn_validation"]["total_pm_actions"]
               for s in seeds if "ddqn_validation" in per_seed[s]]

    # Baseline costs across seeds (per family).
    families = set()
    for s in seeds:
        families.update(per_seed[s].get("baselines", {}).keys())
    baseline_costs: Dict[str, List[float]] = {f: [] for f in families}
    for s in seeds:
        b = per_seed[s].get("baselines", {})
        for fam in families:
            if fam in b:
                baseline_costs[fam].append(b[fam]["mean_total_cost"])

    # Paired deltas: DDQN - baseline per seed (positive = DDQN worse).
    paired_deltas: Dict[str, Dict[str, float]] = {}
    for fam in sorted(families):
        deltas = []
        for s in seeds:
            if ("ddqn_validation" in per_seed[s] and fam in per_seed[s].get("baselines", {})):
                deltas.append(
                    per_seed[s]["ddqn_validation"]["mean_total_cost"]
                    - per_seed[s]["baselines"][fam]["mean_total_cost"]
                )
        paired_deltas[fam] = _stats(deltas) if deltas else {}

    return {
        "phase": phase,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "per_seed": per_seed,
        "ddqn_validation_stats": {
            "mean_total_cost": _stats(ddqn_costs),
            "total_failures": _stats([float(v) for v in ddqn_failures]),
            "total_pm_actions": _stats([float(v) for v in ddqn_pm]),
        },
        "baseline_stats": {fam: _stats(costs) for fam, costs in baseline_costs.items()},
        "paired_deltas_ddqn_minus_baseline": paired_deltas,
        "catastrophic_threshold_cost": _CATASTROPHIC_COST_THRESHOLD,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate M9 point-estimate formal/pilot result manifests")
    ap.add_argument("--phase", choices=["pilot", "formal"], required=True)
    ap.add_argument("--seeds", type=str, default=None,
                    help="comma-separated seeds (default: 6521-6525 for formal, 6521,6522 for pilot)")
    args = ap.parse_args()

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = [6521, 6522] if args.phase == "pilot" else _SEEDS

    report = aggregate(args.phase, seeds)
    out_dir = _RUNS_ROOT / args.phase
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "aggregate_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"=== M9 point-estimate {args.phase} aggregate report ===")
    print(f"seeds: {seeds}")
    print(f"report: {out_path}")
    print()
    print("DDQN validation mean_total_cost across seeds:")
    for s in seeds:
        v = report["per_seed"][s].get("ddqn_validation", {})
        print(f"  seed {s}: mean={v.get('mean_total_cost')} "
              f"pm={v.get('total_pm_actions')} failures={v.get('total_failures')}")
    print(f"  stats: {report['ddqn_validation_stats']['mean_total_cost']}")
    print()
    print("Baseline mean_total_cost (cross-seed mean):")
    for fam, st in report["baseline_stats"].items():
        print(f"  {fam:28s} {st.get('mean'):.2f} +/- {st.get('std'):.2f}")
    print()
    print("Paired delta (DDQN - baseline); positive = DDQN worse:")
    for fam, st in report["paired_deltas_ddqn_minus_baseline"].items():
        if st:
            print(f"  {fam:28s} mean delta={st.get('mean'):.2f} +/- {st.get('std'):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
