"""E4 (M11) Task 17: paired statistics for Learned MPC vs frozen comparators.

Computes the paired per-seed mean differences and 95% confidence intervals
(paired t, n=5, df=4, t_(0.975,4)=2.776445...) for:

    Delta_LMPC_C  = Cost_LMPC - Cost_C   (C  = 1-step DDQN + seeded replay)
    Delta_LMPC_M4 = Cost_LMPC - Cost_M4  (M4 = exact_myopic logistic_T5)
    Delta_LMPC_H2 = Cost_LMPC - Cost_H2  (H2 = canonical m6_h2_v1)

Lower total cost is better, so Delta < 0 means Learned MPC is numerically better.
Consumes the frozen E3 per-policy per-seed means (from the E3 paired eval) and the
LMPC per-seed means (from the E4 formal eval), all under the SAME 5-episode-per-seed
paired rl_validation protocol.

Statistics are identical to the verified E3 analysis method.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from src.milestone11.e4.paths import E4_OUTPUT_ROOT, E3_OUTPUT_ROOT

FORMAL_SEEDS = (6521, 6522, 6523, 6524, 6525)
T_975_DF4 = 2.7764451051977985  # t_(0.975, df=4)


def _paired_stats(deltas: List[float]) -> Dict[str, float]:
    n = len(deltas)
    mean = sum(deltas) / n
    var = (sum((d - mean) ** 2 for d in deltas) / (n - 1)) if n > 1 else 0.0
    se = math.sqrt(var / n)
    half = T_975_DF4 * se
    return {"n": n, "mean": mean, "se": se,
            "ci_lo": mean - half, "ci_hi": mean + half, "var": var}


def _classify(ci_lo: float, ci_hi: float) -> str:
    if ci_hi < 0:
        return "Ci_below_zero_statistical_improvement"
    if ci_lo < 0 < ci_hi:
        return "mean_negative_but_CI_crosses_zero_numerical_only"
    return "ci_at_or_above_zero_no_support"


def main() -> None:
    # LMPC per-seed mean total cost from the E4 formal eval.
    lmpc_path = E4_OUTPUT_ROOT / "formal" / "lmpc_eval_results.json"
    lmpc = json.loads(lmpc_path.read_text())
    lmpc_per_seed = {int(s): lmpc["per_seed"][s]["mean_total_cost"] for s in
                     map(str, FORMAL_SEEDS)}
    # Frozen comparators per-seed means from the E3 paired eval.
    e3 = json.loads((E3_OUTPUT_ROOT / "eval" / "eval_results.json").read_text())
    cmp_per_seed: Dict[str, Dict[int, float]] = {}
    for p in ("C", "M4", "H2", "A"):
        d = {}
        for s in map(str, FORMAL_SEEDS):
            row = e3["seeds"][s].get(p)
            if row and "mean_total_cost" in row:
                d[int(s)] = float(row["mean_total_cost"])
        cmp_per_seed[p] = d

    report: Dict[str, object] = {
        "method": "paired t, n=5, df=4, t_(0.975,4)=2.776445; "
                  "Delta = Cost_LMPC - Cost_comparator; lower cost better; "
                  "Delta<0 => LMPC numerically better",
        "lmpc_per_seed": lmpc_per_seed,
        "comparator_per_seed": {p: {str(s): v for s, v in d.items()}
                                for p, d in cmp_per_seed.items()},
        "contrasts": {},
        "per_policy_cross_seed_mean": {},
    }

    for p, label in (("C", "Cost_LMPC - Cost_C (C-DDQN)"),
                     ("M4", "Cost_LMPC - Cost_M4 (exact_myopic)"),
                     ("H2", "Cost_LMPC - Cost_H2 (m6_h2_v1)"),
                     ("A", "Cost_LMPC - Cost_A (context Original DDQN)")):
        if p not in cmp_per_seed:
            continue
        deltas = [lmpc_per_seed[s] - cmp_per_seed[p][s] for s in FORMAL_SEEDS]
        st = _paired_stats(deltas)
        st["label"] = label
        st["lmpc_per_seed"] = {str(s): lmpc_per_seed[s] for s in FORMAL_SEEDS}
        st["cmp_per_seed"] = {str(s): cmp_per_seed[p][s] for s in FORMAL_SEEDS}
        st["interpretation"] = _classify(st["ci_lo"], st["ci_hi"])
        report["contrasts"][f"LMPC_minus_{p}"] = st

    # Cross-seed means.
    for pol, ps in {"LMPC": lmpc_per_seed, **{f"CMP_{p}": d for p, d in
                                              cmp_per_seed.items()}}.items():
        report["per_policy_cross_seed_mean"][pol] = {
            s: ps.get(int(s)) for s in FORMAL_SEEDS}

    out = E4_OUTPUT_ROOT / "stats" / "lmpc_paired_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")
    print("=== paired contrasts: Delta = LMPC - comparator (mean, [95% CI]) ===")
    for k, v in report["contrasts"].items():
        print(f"  {k} ({v['label']}): mean={v['mean']:.3f} "
              f"[{v['ci_lo']:.3f}, {v['ci_hi']:.3f}] -> {v['interpretation']}")


if __name__ == "__main__":
    main()