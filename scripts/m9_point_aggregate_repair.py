"""Aggregate the M9 point-estimate baseline + DDQN-replay evidence into the
corrected paired scientific report.

The prior aggregation script (``scripts/m9_point_aggregate.py``) had:
  - statistics.pstdev (population std) -- MUST be sample std ddof=1 for
    n=5 inferential statistics;
  - no 95% CI; no paired-t CI;
  - unlabeled "X +/- Y" (ambiguous mean+/-SD vs CI);
  - NO action distribution (16-action counts/freqs);
  - no M3/M4/cache/bank SHA provenance;
  - catastrophic threshold only for training episodes.

This corrected aggregator reads:
  - corrected baseline results from ``<container>/m9_point_runs/baseline_repair/
    seed_/baseline_repair_results.json`` (the corrected run_baselines output,
    written by ``scripts/run_m9_baseline_repair.py`` -- NOT the frozen DDQN
    training-driver manifests, which carried the superseded baseline configuration);
  - DDQN best-checkpoint action-distribution + reproduced validation cost from
    ``<container>/m9_point_runs/ddqn_eval_replay/seed_/ddqn_eval_replay.json``
    (authenticated deterministic evaluation replay of the frozen best
    checkpoint; reproduction check vs the checkpoint metadata
    validation_mean_cost);
  - the frozen ``validation_metrics.json`` + ``training_metrics.jsonl`` +
    ``episode_metrics.csv`` + ``run_manifest.json`` (read-only) for training
    diagnostics.

Primary paired comparison: 5 episodes per policy per seed, 1:1 over
(scenario_id, reset_seed), the same five rl_validation episodes -- NOT the
M3 25-episode formal-closeout design.

USAGE:
  python scripts/m9_point_aggregate.py --phase formal
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.milestone9.point import pairing

_CONTAINER_ROOT = pairing._CONTAINER_ROOT
_RUNS_ROOT = _CONTAINER_ROOT / "m9_point_runs"
_FORMAL = _RUNS_ROOT / "formal"
_BRR = _RUNS_ROOT / "baseline_repair"
_REPLAY = _RUNS_ROOT / "ddqn_eval_replay"

_SEEDS = [6521, 6522, 6523, 6524, 6525]
_FAMILIES = [
    "corrective_only", "random_feasible", "age_threshold",
    "predicted_rul_threshold", "greedy_predicted_rul", "exact_myopic",
]
_FAMILY_LABELS = {
    "corrective_only": "CorrectiveOnly",
    "random_feasible": "RandomFeasible",
    "age_threshold": "AgeThreshold(125)",
    "predicted_rul_threshold": "PredictedRULThreshold(10)",
    "greedy_predicted_rul": "GreedyPredictedRUL(10)",
    "exact_myopic": "M4 exact_myopic (logistic_T5)",
}
_CATASTROPHIC_THRESHOLD = 50.0
# paired-t 95% CI for n=5 (df=4): t_(0.975, df=4) = 2.7764451051977985
_T_CRIT_5 = 2.7764451051977985


def _stats(values: List[float], ddof: int = 1) -> Dict[str, float]:
    """Descriptive statistics with SAMPLE std (ddof=1) by default for
    inferential n=5 use; population std (ddof=0) only when explicitly
    requested for descriptive-only (e.g. within-seed 5-episode dispersion
    under std-total-cost). Labeled explicitly in output."""
    if not values:
        return {"n": 0}
    sd = statistics.stdev(values) if len(values) > 1 else 0.0  # ddof=1
    if ddof == 0:
        sd = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std_sample_ddof1": statistics.stdev(values) if len(values) > 1 else 0.0,
        "std_population_ddof0": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "std": sd,  # = std_sample_ddof1 for default ddof=1
    }


def _paired_t_ci(deltas: List[float]) -> Dict[str, float]:
    """Paired t-distribution 95% CI for the mean delta (n=5, df=4):
      mean_delta +/- t_(0.975, df=4) * sample_std_delta / sqrt(n)
    sample_std uses ddof=1. Returns mean, sd, ci_lower, ci_upper, n, t_crit."""
    n = len(deltas)
    if n == 0:
        return {"n": 0}
    m = statistics.mean(deltas)
    sd = statistics.stdev(deltas) if n > 1 else 0.0  # ddof=1
    if n == 5:
        t = _T_CRIT_5
    else:
        # generic fallback (would need scipy for arbitrary df; use normal approx)
        t = 1.96
    half = t * sd / math.sqrt(n)
    return {
        "n": n,
        "mean_delta": m,
        "sample_std_ddof1": sd,
        "t_crit": t,
        "ci95_lower": m - half,
        "ci95_upper": m + half,
        "ci95_includes_zero": (m - half <= 0.0 <= m + half),
    }


def _load_baseline_repair(seed: int) -> Dict[str, Any]:
    p = _BRR / f"seed_{seed}" / "baseline_repair_results.json"
    if not p.exists():
        raise FileNotFoundError(
            f"repaired baseline results missing for seed {seed}: {p} -- run "
            f"scripts/run_m9_baseline_repair.py first"
        )
    return json.loads(p.read_text())


def _load_replay(seed: int) -> Dict[str, Any]:
    p = _REPLAY / f"seed_{seed}" / "ddqn_eval_replay.json"
    if not p.exists():
        raise FileNotFoundError(
            f"DDQN eval replay missing for seed {seed}: {p} -- run "
            f"scripts/run_m9_ddqn_eval_replay.py first"
        )
    return json.loads(p.read_text())


def _load_training_diagnostics(seed: int) -> Dict[str, Any]:
    run_id = pairing.run_id_for_seed(seed)
    rd = _FORMAL / run_id
    diag: Dict[str, Any] = {}
    tm = rd / "training_metrics.jsonl"
    if tm.exists():
        rows = []
        with tm.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if rows:
            grad = [r.get("grad_norm") for r in rows if r.get("grad_norm") is not None]
            qmeans = [r.get("q_values_mean") for r in rows if r.get("q_values_mean") is not None]
            td = [r.get("td_loss") for r in rows if r.get("td_loss") is not None]
            all_vals = grad + qmeans + td
            diag = {
                "n_logged_steps": len(rows),
                "final_grad_norm": grad[-1] if grad else None,
                "max_grad_norm": max(grad) if grad else None,
                "final_q_values_mean": qmeans[-1] if qmeans else None,
                "min_q_values_mean": min(qmeans) if qmeans else None,
                "max_q_values_mean": max(qmeans) if qmeans else None,
                "final_td_loss": td[-1] if td else None,
                "max_td_loss": max(td) if td else None,
                "all_finite": all(
                    (v is not None and not (math.isnan(v) or math.isinf(v)))
                    for v in all_vals
                ),
            }
    # episode_metrics.csv for catastrophic/wasted under TRAINING
    ep = rd / "episode_metrics.csv"
    if ep.exists():
        episodes = []
        with ep.open() as f:
            for row in csv.DictReader(f):
                episodes.append({
                    "total_cost": float(row["total_cost"]),
                    "wasted_life_cost": float(row.get("wasted_life_cost", 0.0) or 0.0),
                })
        if episodes:
            tc = [e["total_cost"] for e in episodes]
            wl = [e["wasted_life_cost"] for e in episodes]
            diag["training_episodes"] = {
                "n_episodes": len(episodes),
                "mean_total_cost": statistics.mean(tc),
                "catastrophic_episodes": sum(1 for c in tc if c >= _CATASTROPHIC_THRESHOLD),
                "catastrophic_rate": sum(1 for c in tc if c >= _CATASTROPHIC_THRESHOLD) / len(tc),
                "mean_wasted_life_cost": statistics.mean(wl),
            }
    return diag


def aggregate(phase: str, seeds: List[int]) -> Dict[str, Any]:
    """Build the corrected paired five-seed report from the corrected baseline
    results + the authenticated DDQN eval replay."""
    per_seed: Dict[int, Dict[str, Any]] = {}
    for s in seeds:
        brr = _load_baseline_repair(s)
        rep = _load_replay(s)
        prov = brr["results"].get("__provenance__", {})
        rep_r = rep["replayed"]
        # DDQN best-checkpoint validation cost (authenticated replay == ckpt metadata).
        ddqn_cost = rep_r["mean_total_cost"]
        ddqn_act = rep_r["action_distribution"]

        seed_rec: Dict[str, Any] = {
            "seed": s,
            "ddqn": {
                "best_checkpoint_validation_mean_cost": ddqn_cost,
                "replay_reproduced": rep_r["reproduced"],
                "num_episodes": rep_r["num_episodes"],
                "per_episode_costs": rep_r["total_costs_per_episode"],
                "total_failures": rep_r["total_failures"],
                "total_pm_actions": rep_r["total_pm_actions"],
                "action_distribution": ddqn_act,
                "best_checkpoint": rep.get("best_checkpoint"),
                "metric_selection_disclosure": rep.get("metric_selection_disclosure"),
            },
            "baselines": {},
            "provenance": prov,
        }
        for fam in _FAMILIES:
            rec = brr["results"].get(fam, {})
            seed_rec["baselines"][fam] = {
                "label": _FAMILY_LABELS[fam],
                "mean_total_cost": rec.get("mean_total_cost"),
                "median_total_cost": rec.get("median_total_cost"),
                "per_episode_costs": rec.get("per_episode_costs"),
                "total_failures": rec.get("total_failures"),
                "total_pm_actions": rec.get("total_pm_actions"),
                "total_corrective_actions": rec.get("total_corrective_actions"),
                "mean_wasted_life": rec.get("mean_wasted_life"),
                "catastrophic_episodes": rec.get("catastrophic_episodes"),
                "catastrophic_rate": rec.get("catastrophic_rate"),
                "num_episodes": rec.get("num_episodes"),
                "per_episode_failures": rec.get("per_episode_failures"),
                "per_episode_pm": rec.get("per_episode_pm"),
                "threshold": rec.get("threshold"),
                "activation_threshold": rec.get("activation_threshold"),
            }
        seed_rec["training_diagnostics"] = _load_training_diagnostics(s)
        per_seed[s] = seed_rec

    # ---- Cross-seed DDQN ----
    ddqn_costs = [per_seed[s]["ddqn"]["best_checkpoint_validation_mean_cost"] for s in seeds]
    ddqn_failures = [float(per_seed[s]["ddqn"]["total_failures"]) for s in seeds]
    ddqn_pm = [float(per_seed[s]["ddqn"]["total_pm_actions"]) for s in seeds]

    # Aggregate DDQN action distribution (concatenate per-seed action counts).
    agg_act_counts = [0] * 16
    for s in seeds:
        cs = per_seed[s]["ddqn"]["action_distribution"]["action_counts"]
        for i, c in enumerate(cs):
            agg_act_counts[i] += c
    agg_total = sum(agg_act_counts)
    agg_freq = [c / agg_total if agg_total else 0.0 for c in agg_act_counts]
    dom = sorted(range(16), key=lambda a: agg_act_counts[a], reverse=True)[:5]

    # ---- Baseline cross-seed + paired deltas ----
    baseline_stats: Dict[str, Any] = {}
    paired: Dict[str, Any] = {}
    per_seed_table: List[Dict[str, Any]] = []
    for fam in _FAMILIES:
        bc = [per_seed[s]["baselines"][fam]["mean_total_cost"] for s in seeds]
        baseline_stats[fam] = _stats(bc)  # ddof=1 sample std
        deltas = [per_seed[s]["ddqn"]["best_checkpoint_validation_mean_cost"]
                  - per_seed[s]["baselines"][fam]["mean_total_cost"] for s in seeds]
        ci = _paired_t_ci(deltas)
        paired[fam] = {
            "label": _FAMILY_LABELS[fam],
            "deltas_per_seed": {str(s): d for s, d in zip(seeds, deltas)},
            "deltas": deltas,
            **ci,
            "ddqn_lower_count": sum(1 for d in deltas if d < 0),
            "ddqn_lower_5_of_5": all(d < 0 for d in deltas),
        }

    for s in seeds:
        row = {
            "seed": s,
            "ddqn_best_ckpt_cost": per_seed[s]["ddqn"]["best_checkpoint_validation_mean_cost"],
            "ddqn_final_step_cost": per_seed[s]["ddqn"]["best_checkpoint"].get("final_step_mean_cost"),
            "ddqn_reproduced": per_seed[s]["ddqn"]["replay_reproduced"],
        }
        for fam in _FAMILIES:
            row[fam] = per_seed[s]["baselines"][fam]["mean_total_cost"]
        per_seed_table.append(row)

    # ---- Aggregate action distribution + maintenance summary ----
    maintenance: Dict[str, Any] = {}
    for fam in _FAMILIES:
        pm = sum(per_seed[s]["baselines"][fam]["total_pm_actions"] for s in seeds)
        co = sum(per_seed[s]["baselines"][fam]["total_corrective_actions"] for s in seeds)
        wl = statistics.mean([per_seed[s]["baselines"][fam]["mean_wasted_life"] for s in seeds])
        maintenance[fam] = {"total_pm_actions_5seeds": pm, "total_corrective_5seeds": co,
                           "mean_wasted_life": wl}
    ddqn_pm_total = sum(per_seed[s]["ddqn"]["total_pm_actions"] for s in seeds)
    ddqn_co_total = sum(per_seed[s]["ddqn"]["total_failures"] for s in seeds)
    maintenance["ddqn"] = {"total_pm_actions_5seeds": ddqn_pm_total,
                           "total_corrective_5seeds": ddqn_co_total,
                           "total_failures_5seeds": int(sum(per_seed[s]["ddqn"]["total_failures"] for s in seeds))}

    # ---- Failures + catastrophic ----
    failures: Dict[str, Any] = {}
    for fam in _FAMILIES:
        fr = [per_seed[s]["baselines"][fam]["total_failures"] for s in seeds]
        cat = sum(per_seed[s]["baselines"][fam]["catastrophic_episodes"] for s in seeds)
        failures[fam] = {"per_seed": fr, "total_5seeds": sum(fr), "rate_per_25ep": sum(fr)/25.0,
                         "catastrophic_episodes_5seeds": cat, "catastrophic_rate_25ep": cat/25.0}
    failures["ddqn"] = {
        "per_seed": [per_seed[s]["ddqn"]["total_failures"] for s in seeds],
        "total_5seeds": int(sum(per_seed[s]["ddqn"]["total_failures"] for s in seeds)),
        "rate_per_25ep": sum(per_seed[s]["ddqn"]["total_failures"] for s in seeds)/25.0,
        "catastrophic_episodes_5seeds": sum(
            1 for s in seeds for c in per_seed[s]["ddqn"]["per_episode_costs"]
            if c >= _CATASTROPHIC_THRESHOLD),
    }

    # ---- Provenance (M3/M4/cache/bank SHAs) from seed 6521's record ----
    sample_prov = per_seed[seeds[0]]["provenance"]
    provenance = {
        "m3_selected_thresholds": sample_prov.get("m3_selected_thresholds"),
        "m4_scientific_selection": sample_prov.get("m4_scientific_selection"),
        "m9_regime": sample_prov.get("m9_regime"),
        "eval_context_seed6521": sample_prov.get("eval_context"),
        "source_git": sample_prov.get("source_git"),
        "per_seed_cache_manifest_sha256": {
            str(s): per_seed[s]["provenance"]["eval_context"]["cache_manifest_sha256"]
            for s in seeds
        },
    }

    report = {
        "report_type": "M9 point-estimate CORRECTED paired baseline report (the corrected report)",
        "phase": phase,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "primary_evaluation_design": (
            "5 paired episodes per policy per seed (1:1 over scenario_id x "
            "FIXED_RESET_SEEDS=[6521..6525]); the SAME five rl_validation "
            "episodes used by the DDQN. NOT the M3 25-episode formal-closeout."
        ),
        "ddqn_metric_used": "best checkpoint validation_mean_cost (NOT final-step validation)",
        "ddqn_validation_stats": {
            "best_ckpt_cost": _stats(ddqn_costs),
            "total_failures": _stats(ddqn_failures),
            "total_pm_actions": _stats(ddqn_pm),
            "per_seed_cost": {str(s): c for s, c in zip(seeds, ddqn_costs)},
        },
        "ddqn_action_distribution_aggregate": {
            "action_counts": agg_act_counts,
            "action_freq": agg_freq,
            "n_actions": 16,
            "total_actions": agg_total,
            "dominant_action_ids": dom,
            "source": "authenticated evaluation replay (epsilon=0 greedy) on the 5 rl_validation episodes",
            "per_seed": {str(s): per_seed[s]["ddqn"]["action_distribution"] for s in seeds},
        },
        "baseline_stats": baseline_stats,
        "paired_deltas_ddqn_minus_baseline": paired,
        "failures": failures,
        "maintenance": maintenance,
        "catastrophic_threshold_cost": _CATASTROPHIC_THRESHOLD,
        "per_seed_table": per_seed_table,
        "training_diagnostics_per_seed": {str(s): per_seed[s]["training_diagnostics"] for s in seeds},
        "provenance": provenance,
        "supersedes_prior_report": (
            "This corrected report supersedes the prior aggregate_report.json, "
            "which used population std (ddof=0), no 95% CI, unlabeled +/-, no "
            "action distribution, no M3/M4/cache/bank SHA provenance, AND used "
            "the superseded baseline configuration (M3 smoke 100/50 thresholds + "
            "M4 hard_window_v1/10.0). See the archive note for the prior SHA."
        ),
    }
    return report


def _print_summary(report: Dict[str, Any]) -> None:
    seeds = report["seeds"]
    print("=== M9 point-estimate CORRECTED paired report ===")
    print(f"seeds: {seeds} | DDQN metric: {report['ddqn_metric_used']}")
    print()
    print("DDQN best-ckpt validation cost per seed:")
    for s in seeds:
        print(f"  seed {s}: {report['ddqn_validation_stats']['per_seed_cost'][str(s)]}")
    print(f"  cross-seed: mean={report['ddqn_validation_stats']['best_ckpt_cost']['mean']:.3f} "
          f"(sample SD ddof=1 = {report['ddqn_validation_stats']['best_ckpt_cost']['std_sample_ddof1']:.3f})")
    print()
    print("DDQN action distribution (aggregate over 5 seeds):")
    ad = report["ddqn_action_distribution_aggregate"]
    print(f"  dominant actions (id:count): "
          f"{[(a, ad['action_counts'][a]) for a in ad['dominant_action_ids']]}")
    print(f"  total_actions={ad['total_actions']} no-op(a=0) freq={ad['action_freq'][0]:.3f}")
    print()
    print("Baseline cross-seed mean_total_cost (sample SD ddof=1):")
    for fam in _FAMILIES:
        st = report["baseline_stats"][fam]
        print(f"  {report['paired_deltas_ddqn_minus_baseline'][fam]['label']:28s} "
              f"mean={st['mean']:.3f} (sample SD = {st['std_sample_ddof1']:.3f})")
    print()
    print("Paired delta (DDQN_best - baseline); negative = DDQN lower cost:")
    for fam in _FAMILIES:
        p = report["paired_deltas_ddqn_minus_baseline"][fam]
        ci_str = f"(95% CI: [{p['ci95_lower']:.3f}, {p['ci95_upper']:.3f}])"
        flag = "DDQN<BL 5/5" if p["ddqn_lower_5_of_5"] else f"DDQN<BL {p['ddqn_lower_count']}/5"
        includes_zero = "CI includes 0" if p["ci95_includes_zero"] else "CI excludes 0"
        print(f"  {p['label']:28s} mean delta={p['mean_delta']:>7.3f} "
              f"(sample SD = {p['sample_std_ddof1']:.3f}) {ci_str} [{flag}, {includes_zero}]")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Aggregate the CORRECTED M9 point-estimate paired report")
    ap.add_argument("--phase", choices=["formal", "pilot"], default="formal")
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--out", type=str, default=None,
                    help="output path (default: <_RUNS_ROOT>/formal/corrected_aggregate_report.json)")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else _SEEDS

    report = aggregate(args.phase, seeds)

    # Archive the prior (invalid) report if still on disk: record its SHA
    # before overwriting, so the supersession is evidence-backed.
    prior_path = _RUNS_ROOT / args.phase / "aggregate_report.json"
    archive_note = {}
    if prior_path.exists():
        import hashlib
        prior_sha = hashlib.sha256(prior_path.read_bytes()).hexdigest()
        archive_note = {
            "prior_report_sha256": prior_sha,
            "prior_report_path": str(prior_path),
            "prior_report_status": (
                "Superseded: used population std (ddof=0), no 95% CI, unlabeled +/-, "
                "no action distribution, no SHA provenance, AND the superseded "
                "baseline configuration (M3 smoke 100/50 + M4 hard_window_v1/10.0). "
                "Preserved in place (NOT deleted) for supersession evidence."
            ),
        }
    report["prior_report_archive"] = archive_note

    out_path = Path(args.out) if args.out else _RUNS_ROOT / args.phase / "corrected_aggregate_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        import hashlib
        report["_overwrite_note"] = {
            "prev_corrected_report_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        }
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, default=str))
    tmp.replace(out_path)

    # New report SHA.
    import hashlib
    new_sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"corrected report: {out_path}")
    print(f"corrected report SHA256: {new_sha}")
    if archive_note:
        print(f"prior (invalid) report SHA256: {archive_note['prior_report_sha256']}")
    print()
    _print_summary(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())