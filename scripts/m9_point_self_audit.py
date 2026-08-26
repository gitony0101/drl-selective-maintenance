"""M9 point-estimate baseline-corrected scientific self-audit (scientific self-audit).

Read-only validation of the corrected result set: does NOT modify any artifact.
Validates the consistency and statistical invariants of the corrected
result set + corrected baseline driver + DDQN eval replay. Exits 0 only if
EVERY guarded requirement holds.

Run:
  python3 scripts/m9_point_self_audit.py
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.milestone9.point import pairing

_CONTAINER_ROOT = pairing._CONTAINER_ROOT
_RUNS_ROOT = _CONTAINER_ROOT / "m9_point_runs"
_FORMAL = _RUNS_ROOT / "formal"
_BRR = _RUNS_ROOT / "baseline_repair"
_REPLAY = _RUNS_ROOT / "ddqn_eval_replay"

_SEEDS = [6521, 6522, 6523, 6524, 6525]
_FAMILIES = ["corrective_only", "random_feasible", "age_threshold",
             "predicted_rul_threshold", "greedy_predicted_rul", "exact_myopic"]
_T_CRIT_5 = 2.7764451051977985
_CATASTROPHIC = 50.0

checks = []

def chk(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, ok, detail))

def report() -> int:
    failed = [(n, d) for (n, ok, d) in checks if not ok]
    for n, ok, d in checks:
        flag = "PASS" if ok else "FAIL"
        print(f"[{flag}] {n}" + (f" -- {d}" if d else ""))
    print()
    if failed:
        print(f"SELF-AUDIT FAILED: {len(failed)} requirement(s) violated.")
        for n, d in failed:
            print(f"  - {n}: {d}")
        return 1
    print(f"SELF-AUDIT PASS: {len(checks)} requirements verified.")
    return 0

corr = json.loads((_FORMAL / "corrected_aggregate_report.json").read_text())
old = json.loads((_FORMAL / "aggregate_report.json").read_text())

# ====== A. Regime / design (fail-closed) ======
chk("regime K=2", corr["provenance"]["m9_regime"]["k"] == 2,
    f"k={corr['provenance']['m9_regime']['k']}")
chk("regime split=rl_validation", corr["provenance"]["m9_regime"]["split"] == "rl_validation",
    f"split={corr['provenance']['m9_regime']['split']}")
chk("regime cost=failure-light-no-waste",
    corr["provenance"]["m9_regime"]["cost_regime_id"] == "failure-light-no-waste",
    f"cost={corr['provenance']['m9_regime']['cost_regime_id']}")
chk("5 seeds", corr["seeds"] == _SEEDS, f"seeds={corr['seeds']}")
chk("primary design is 5 paired episodes (NOT 25-episode M3)",
    "5 paired" in corr["primary_evaluation_design"] and "25-episode" in corr["primary_evaluation_design"],
    corr["primary_evaluation_design"][:80])
chk("ddqn metric = best checkpoint (NOT final step)",
    corr["ddqn_metric_used"].startswith("best checkpoint"),
    corr["ddqn_metric_used"])

# ====== B. M3 thresholds (frozen SHA + 125/10/10) ======
m3 = corr["provenance"]["m3_selected_thresholds"]
chk("M3 selected_thresholds SHA 47762b3f",
    m3["sha256"] == "47762b3fd8fbba62a0c159f71a9d3778b6d99d5332e955dc19cf242751a395d6",
    m3["sha256"])
chk("M3 age=125",
    m3["age_threshold_k2_failure-light-no-waste"] == 125,
    f"{m3['age_threshold_k2_failure-light-no-waste']}")
chk("M3 predicted_rul=10",
    m3["predicted_rul_threshold_k2_failure-light-no-waste"] == 10,
    f"{m3['predicted_rul_threshold_k2_failure-light-no-waste']}")
chk("M3 greedy activation=10",
    m3["greedy_predicted_rul_k2_failure-light-no-waste"] == 10,
    f"{m3['greedy_predicted_rul_k2_failure-light-no-waste']}")

# ====== C. M4 candidate (frozen SHA + logistic_window_v1 / 5.0) ======
m4 = corr["provenance"]["m4_scientific_selection"]
chk("M4 selection SHA 7684b1ff",
    m4["decision_sha256"] == "7684b1ff5f605429c8639ed89160e74dcc2a45cec5f6e533e465ac703ea2c6c8",
    m4["decision_sha256"])
chk("M4 candidate=logistic_T5", m4["selected_candidate"] == "logistic_T5",
    m4["selected_candidate"])
chk("M4 risk_model=logistic_window_v1", m4["risk_model_id"] == "logistic_window_v1",
    m4["risk_model_id"])
chk("M4 risk_temperature=5.0", m4["risk_temperature"] == 5.0,
    f"{m4['risk_temperature']}")
chk("OLD report used hard_window_v1 (must NOT)",
    "logistic" not in json.dumps(old) and "hard_window" in json.dumps(old),
    "old used hard_window_v1; new uses logistic_window_v1 -- supersession confirmed")
chk("NEW report uses logistic as the LIVE M4 candidate",
    corr["provenance"]["m4_scientific_selection"]["risk_model_id"] == "logistic_window_v1",
    "the provenance M4 risk_model must be logistic_window_v1, not hard_window_v1")
# Confirm `hard_window` only appears in the supersession/archive note (describing
# what the OLD report did wrong), NOT in any LIVE field of the corrected report.
corr_blob_less_archive = json.dumps({k: v for k, v in corr.items()
                                     if k not in ("supersedes_prior_report",
                                                  "prior_report_archive")})
chk("NEW report has hard_window ONLY in archive note",
    "hard_window" not in corr_blob_less_archive,
    "hard_window must not appear outside the archive/supersession note")

# ====== D. Per-seed baseline corrected (5 seeds x 6 families, finite) ======
brr_idx = json.loads((_BRR / "baseline_repair_index.json").read_text())
for s in _SEEDS:
    rec_path = _BRR / f"seed_{s}" / "baseline_repair_results.json"
    chk(f"baseline_repair seed{s} exists", rec_path.exists(), str(rec_path))
    if not rec_path.exists():
        continue
    r = json.loads(rec_path.read_text())
    res = r.get("results", {})
    fams = {k for k in res if k != "__provenance__"}
    chk(f"seed{s} 6 families", fams == set(_FAMILIES), f"{sorted(fams)}")
    chk(f"seed{s} __provenance__ present", "__provenance__" in res)
    # Threshold assertions on the per-seed baseline record
    chk(f"seed{s} age threshold=125", res["age_threshold"]["threshold"] == 125,
        f"{res['age_threshold']['threshold']}")
    chk(f"seed{s} pred_rul threshold=10",
        res["predicted_rul_threshold"]["threshold"] == 10,
        f"{res['predicted_rul_threshold']['threshold']}")
    chk(f"seed{s} greedy activation=10",
        res["greedy_predicted_rul"]["activation_threshold"] == 10,
        f"{res['greedy_predicted_rul']['activation_threshold']}")
    # exact_myopic provenance on per-seed record
    ep = res["exact_myopic"]["provenance"]
    chk(f"seed{s} m4 risk_model logistic",
        ep["risk_model_id"] == "logistic_window_v1", ep["risk_model_id"])
    chk(f"seed{s} m4 temp 5.0", ep["risk_temperature"] == 5.0,
        f"{ep['risk_temperature']}")
    # Finite, num_episodes=5
    for fam in _FAMILIES:
        mtc = res[fam]["mean_total_cost"]
        chk(f"seed{s} {fam} finite",
            isinstance(mtc, (int, float)) and not (math.isnan(mtc) or math.isinf(mtc)),
            f"{mtc}")
        chk(f"seed{s} {fam} num_episodes=5",
            res[fam]["num_episodes"] == 5, f"{res[fam]['num_episodes']}")

# ====== E. DDQN eval replay (reproduction PASS, action distribution) ======
rep_idx = json.loads((_REPLAY / "ddqn_eval_replay_index.json").read_text())
per_seed_reproduced = []
for s in _SEEDS:
    rec = json.loads((_REPLAY / f"seed_{s}" / "ddqn_eval_replay.json").read_text())
    r = rec["replayed"]
    per_seed_reproduced.append(r["reproduced"])
    chk(f"seed{s} replay reproduced", r["reproduced"],
        f"replayed={r['mean_total_cost']} frozen={r['frozen_mean_total_cost']}")
    chk(f"seed{s} replay num_episodes=5", r["num_episodes"] == 5,
        f"{r['num_episodes']}")
    ad = r["action_distribution"]
    chk(f"seed{s} 16 actions", ad["n_actions"] == 16 and len(ad["action_counts"]) == 16,
        f"{ad['n_actions']}/{len(ad['action_counts'])}")
    chk(f"seed{s} action_distribution source=eval replay",
        "deterministic evaluation replay" in ad["source"], ad["source"][:60])
    # metric_selection_disclosure present
    chk(f"seed{s} metric_selection_disclosure present",
        "metric_selection_disclosure" in rec, "missing disclosure block")
chk("all 5 seeds replay reproduced", all(per_seed_reproduced), str(per_seed_reproduced))

# ====== F. DDQN best-ckpt costs are 13/13/13/14/13 (NOT final-step) ======
dv = corr["ddqn_validation_stats"]["per_seed_cost"]
chk("ddqn best-ckpt costs 13/13/13/14/13",
    [dv[str(s)] for s in _SEEDS] == [13.0, 13.0, 13.0, 14.0, 13.0],
    str({s: dv[str(s)] for s in _SEEDS}))

# ====== G. Aggregate statistics are SAMPLE std (ddof=1) + paired-t CI ======
# Check baseline_stats and paired_deltas use sample std
for fam in _FAMILIES:
    bs = corr["baseline_stats"][fam]
    # Recompute sample std (ddof=1) from per-seed means and verify match
    means = [corr["per_seed_table"][i][fam] for i in range(5)]
    expected_sd = statistics.stdev(means)  # ddof=1
    chk(f"{fam} baseline std_sample_ddof1",
        abs(bs["std_sample_ddof1"] - expected_sd) < 1e-9,
        f"{bs['std_sample_ddof1']} vs recomputed {expected_sd}")
    # paired delta CI recomputation
    deltas = [corr["per_seed_table"][i]["ddqn_best_ckpt_cost"]
              - corr["per_seed_table"][i][fam] for i in range(5)]
    p = corr["paired_deltas_ddqn_minus_baseline"][fam]
    chk(f"{fam} paired delta recalc mean",
        abs(p["mean_delta"] - statistics.mean(deltas)) < 1e-9,
        f"{p['mean_delta']} vs {statistics.mean(deltas)}")
    chk(f"{fam} paired-t 95% CI present",
        "ci95_lower" in p and "ci95_upper" in p, str(list(p.keys())))
    # verify the CI formula with t_crit=2.7764
    m = statistics.mean(deltas)
    sd = statistics.stdev(deltas)
    half = _T_CRIT_5 * sd / math.sqrt(5)
    chk(f"{fam} paired-t CI formula",
        abs(p["ci95_lower"] - (m - half)) < 1e-6 and abs(p["ci95_upper"] - (m + half)) < 1e-6,
        f"recomputed [{m-half:.4f}, {m+half:.4f}] vs reported [{p['ci95_lower']:.4f}, {p['ci95_upper']:.4f}]")
    chk(f"{fam} t_crit=2.7764451051977985",
        abs(p["t_crit"] - _T_CRIT_5) < 1e-9, f"t_crit={p['t_crit']}")

# ====== H. Aggregate action distribution + maintenance/catastrophic ======
ad = corr["ddqn_action_distribution_aggregate"]
chk("aggregate 16 actions", ad["n_actions"] == 16 and len(ad["action_counts"]) == 16)
chk("aggregate action_counts sum=total_actions", sum(ad["action_counts"]) == ad["total_actions"],
    f"sum={sum(ad['action_counts'])} total={ad['total_actions']}")
chk("aggregate total_actions=2500", ad["total_actions"] == 2500, str(ad["total_actions"]))
chk("aggregate per_seed has 5", len(ad["per_seed"]) == 5, str(list(ad["per_seed"].keys())))
chk("DDQN total_failures=0", corr["failures"]["ddqn"]["total_5seeds"] == 0,
    str(corr["failures"]["ddqn"]["total_5seeds"]))
chk("DDQN catastrophic=0", corr["failures"]["ddqn"]["catastrophic_episodes_5seeds"] == 0,
    str(corr["failures"]["ddqn"]["catastrophic_episodes_5seeds"]))
chk("maintenance ddqn total_pm=330", corr["maintenance"]["ddqn"]["total_pm_actions_5seeds"] == 330,
    str(corr["maintenance"]["ddqn"]["total_pm_actions_5seeds"]))
# baseline maintenance present for all 6 families
chk("maintenance all 6 families present",
    all(f in corr["maintenance"] for f in _FAMILIES),
    str(list(corr["maintenance"].keys())))

# ====== I. Training diagnostics (read-only, all_finite) ======
td = corr["training_diagnostics_per_seed"]
for s in _SEEDS:
    sd = td[str(s)]
    chk(f"seed{s} training all_finite", sd.get("all_finite") is True, str(sd.get("all_finite")))
    chk(f"seed{s} has final_grad_norm", "final_grad_norm" in sd, str(sorted(sd.keys())))
    chk(f"seed{s} has final_q_values_mean", "final_q_values_mean" in sd)
    chk(f"seed{s} has training_episodes", "training_episodes" in sd,
        str(sorted(sd.keys())))

# ====== J. rl_test NOT accessed (fail-closed guard exists in source) ======
bl_src = (Path(__file__).resolve().parent.parent / "src" / "milestone9" / "point"
          / "baselines.py").read_text()
chk("baselines.py has rl_test barrier",
    "rl_test" in bl_src and "_rl_test_barrier" in bl_src, "must reject rl_test split")
chk("baselines.py has fail-closed regime guard",
    "_fail_closed_regime" in bl_src, "must assert K/regime/split")
chk("baselines.py allow_oracle controls oracle (NOT evaluated)",
    "allow_oracle=False" in bl_src, "PolicyEvaluator must be allow_oracle=False")

# ====== K. Forbidden claims NOT in the corrected report ======
corr_blob = json.dumps(corr)
for forbidden in ["universally dominates", "generally superior",
                  "proves the best policy", "test-set superiority",
                  "production superiority", "generalization across",
                  "rl_test superiority"]:
    chk(f"forbidden phrase absent: {forbidden!r}",
        forbidden.lower() not in corr_blob.lower(),
        "forbidden claim phrase detected in corrected report")

# ====== L. Supersession evidence (OLD archived SHA recorded) ======
chk("prior_report_archive present with SHA",
    corr["prior_report_archive"].get("prior_report_sha256") ==
    "15b84d56a4104987dcfc97edbf27a18973b66ef72074efa6e4314ab8e488d5cf",
    corr["prior_report_archive"].get("prior_report_sha256"))
chk("OLD report has NO ci95 (must be missing)",
    "ci95" not in json.dumps(old), "old must lack 95% CI")
chk("OLD report has NO action_distribution (must be missing)",
    "action_distribution" not in json.dumps(old) and "action_counts" not in json.dumps(old),
    "old must lack action distribution")

# ====== M. DDQN dirs UNTOUCHED — files predate corrected outputs ======
# The protocol forbids retraining/overwriting/resuming/moving/renaming/
# regenerating the DDQN runs. The operative invariant is therefore NOT "older
# than N days" (the DDQN runs were legitimately produced ~12h before the
# corrected) but "the corrected outputs were produced after the frozen DDQN
# runs": the DDQN run files must PREDATE the corrected outputs
# (baseline_corrected / ddqn_eval_replay / corrected report). If any DDQN file
# is NEWER than the earliest corrected output, that would indicate the frozen
# run was overwritten (FAIL).
import time as _time
_repair_mtime_baseline = (_BRR / "seed_6521" / "baseline_repair_results.json").stat().st_mtime
_sess_min = _repair_mtime_baseline
for other in [(_REPLAY / "seed_6521" / "ddqn_eval_replay.json"),
              (_FORMAL / "corrected_aggregate_report.json")]:
    if other.exists():
        _sess_min = min(_sess_min, other.stat().st_mtime)
ddqn_max_overall = 0.0
for s in _SEEDS:
    rid = pairing.run_id_for_seed(s)
    ddqn_dir = _FORMAL / rid
    seed_max = 0.0
    for fn in ["checkpoint_best.pt", "training_metrics.jsonl", "episode_metrics.csv",
               "validation_metrics.json", "runtime_config.json"]:
        fp = ddqn_dir / fn
        if fp.exists():
            mt = fp.stat().st_mtime
            seed_max = max(seed_max, mt)
            ddqn_max_overall = max(ddqn_max_overall, mt)
    chk(f"seed{s} DDQN files predate repair outputs",
        seed_max < _sess_min,
        f"seed max mtime={_time.ctime(seed_max)} repair-min={_time.ctime(_sess_min)}")
chk("ALL 5 seeds' DDQN files predate repair outputs",
    ddqn_max_overall < _sess_min,
    f"ddqn max={_time.ctime(ddqn_max_overall)} repair-min={_time.ctime(_sess_min)}")

sys.exit(report())