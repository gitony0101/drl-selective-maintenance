"""E5 Task 15: paired E5-B vs E4-A statistics.

Primary comparison:

    E5-B failure-enriched Learned MPC  vs  frozen E4-A original Learned MPC

For each formal seed:

    Delta_seed = Cost_E5B(seed) - Cost_E4A(seed)

Delta < 0 means failure enrichment is numerically better. Lower total cost is
better. Reports the paired mean Delta, a 95% paired confidence interval (paired
t, n=5, df=4), per-seed differences, and contextual comparators.

Uses the EXACT paired-t method verified for E3/E4 (t_(0.975,4) = 2.776445).
E4-A control is the FROZEN E4 result; it is never retrained.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from src.milestone11.e5.paths import E3_OUTPUT_ROOT, E5_OUTPUT_ROOT, E4_OUTPUT_ROOT, FORMAL_SEEDS

T_975_DF4 = 2.7764451051977985


def _paired_stats(deltas):
    n = len(deltas)
    mean = sum(deltas) / n
    var = (sum((d - mean) ** 2 for d in deltas) / (n - 1)) if n > 1 else 0.0
    se = math.sqrt(var / n)
    half = T_975_DF4 * se
    return {"n": n, "mean": mean, "se": se,
            "var": var, "ci_lo": mean - half, "ci_hi": mean + half}


def _classify(ci_lo, ci_hi):
    if ci_hi < 0:
        return "CI_below_zero_statistically_supported_improvement"
    if ci_lo < 0 < ci_hi:
        return "mean_negative_numerical_improvement_only"
    if ci_lo == 0 or ci_hi == 0:
        return "CI_touches_zero_no_direction"
    return "CI_at_or_above_zero_no_support"


def main() -> None:
    e5 = json.loads((E5_OUTPUT_ROOT / "formal" / "eval_LMPC.json").read_text())
    e5_per_seed = {int(s): e5["per_seed"][s]["mean_total_cost"] for s in
                   map(str, FORMAL_SEEDS)}
    # E4-A control (frozen E4 result).
    e4 = json.loads((E4_OUTPUT_ROOT / "formal" / "lmpc_eval_results.json").read_text())
    e4_per_seed = {int(s): e4["per_seed"][s]["mean_total_cost"] for s in
                   map(str, FORMAL_SEEDS)}
    e3 = json.loads((E3_OUTPUT_ROOT / "eval" / "eval_results.json").read_text())

    # Per-scenario detail.
    e5_scen = {int(s): {e["scenario_id"]: e["total_cost"]
                        for e in e5["per_seed"][s]["episodes"]}
               for s in map(str, FORMAL_SEEDS)}
    e4_scen = {int(s): {e["scenario_id"]: e["total_cost"]
                        for e in e4["per_seed"][s]["episodes"]}
               for s in map(str, FORMAL_SEEDS)}
    scenarios = sorted(e5_scen[6521].keys())

    deltas = [e5_per_seed[s] - e4_per_seed[s] for s in FORMAL_SEEDS]
    primary = _paired_stats(deltas)
    primary["label"] = "Cost_E5B - Cost_E4A (failure-enriched vs original)"
    primary["e5b_per_seed"] = {str(s): e5_per_seed[s] for s in FORMAL_SEEDS}
    primary["e4a_per_seed"] = {str(s): e4_per_seed[s] for s in FORMAL_SEEDS}
    primary["per_seed_delta"] = {str(s): e5_per_seed[s] - e4_per_seed[s]
                                 for s in FORMAL_SEEDS}
    primary["interpretation"] = _classify(primary["ci_lo"], primary["ci_hi"])

    # Contextual comparators (frozen, NOT the primary contrast).
    cmp_per_seed = {}
    for p in ("C", "M4", "H2", "A"):
        d = {}
        for s in map(str, FORMAL_SEEDS):
            row = e3["seeds"][s].get(p)
            if row and "mean_total_cost" in row:
                d[int(s)] = float(row["mean_total_cost"])
        cmp_per_seed[p] = d

    contextual = {}
    for p in ("C", "M4", "H2", "A"):
        if p not in cmp_per_seed:
            continue
        contextual[f"E5B_minus_{p}"] = dict(
            _paired_stats([e5_per_seed[s] - cmp_per_seed[p][s] for s in FORMAL_SEEDS]),
            label=f"Cost_E5B - Cost_{p}", e5b_per_seed={str(s): e5_per_seed[s]
                                                        for s in FORMAL_SEEDS},
            cmp_per_seed={str(s): cmp_per_seed[p][s] for s in FORMAL_SEEDS},
        )

    report = {
        "method": "paired t, n=5, df=4, t_(0.975,4)=2.776445; "
                  "Delta = Cost_E5B - Cost_E4A; lower cost better; Delta<0 => "
                  "failure-enriched training data numerically better",
        "primary_contrast": primary,
        "per_scenario_E5B": {str(s): e5_scen[s] for s in FORMAL_SEEDS},
        "per_scenario_E4A": {str(s): e4_scen[s] for s in FORMAL_SEEDS},
        "per_scenario_mean_delta": {
            scen: float(sum(e5_scen[s][scen] - e4_scen[s][scen]
                           for s in FORMAL_SEEDS) / len(FORMAL_SEEDS))
            for scen in scenarios},
        "contextual_comparators": contextual,
        "frozen_e3_comparator_cross_seed_mean": {
            p: float(sum(v.values()) / len(v)) if v else None
            for p, v in cmp_per_seed.items()},
    }
    out = E5_OUTPUT_ROOT / "stats" / "e5_paired_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")
    print("=== PRIMARY: Delta = Cost_E5B - Cost_E4A (mean, [95% CI]) ===")
    print(f"  mean={primary['mean']:.3f} [{primary['ci_lo']:.3f}, "
          f"{primary['ci_hi']:.3f}] -> {primary['interpretation']}")
    print("  per-seed deltas:", primary["per_seed_delta"])
    print("=== Contextual: E5-B vs frozen comparators ===")
    for k, v in contextual.items():
        print(f"  {k}: mean={v['mean']:.3f} [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]")


if __name__ == "__main__":
    main()