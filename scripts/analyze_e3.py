"""E3 (M10) paired statistical analysis (Step 11).

Consumes ``m10_e3_outputs/eval/eval_results.json`` (per-seed per-policy 5-episode
mean costs) and reports the Section 23 factorial contrasts + M4/H2 comparisons
with paired mean differences and 95% confidence intervals (paired t, n=5, df=4,
t_(0.975,4) = 2.776445...). Lower total cost is better: a negative Delta(X minus Y)
means policy X is cheaper than Y.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List

from src.runtime_paths import external_root as _external_root

_CONTAINER = _external_root()
EVAL = _CONTAINER / "m10_e3_outputs" / "eval" / "eval_results.json"

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


def _load() -> Dict:
    return json.loads(EVAL.read_text())


def _per_seed_cost(res: Dict, policy: str) -> Dict[int, float]:
    out = {}
    for s in FORMAL_SEEDS:
        row = res["seeds"][str(s)].get(policy) or {}
        if "mean_total_cost" in row:
            out[s] = row["mean_total_cost"]
    return out


def _contrast(res: Dict, label: str, minus_policy: str, base_policy: str) -> Dict:
    """Report paired deltas = Cost_(minus_policy) - Cost_(base_policy)."""
    cM = _per_seed_cost(res, minus_policy)
    cB = _per_seed_cost(res, base_policy)
    seeds = sorted(set(cM) & set(cB))
    if not seeds:
        return {"label": label, "error": "no paired seeds"}
    deltas = [cM[s] - cB[s] for s in seeds]
    st = _paired_stats(deltas)
    st["label"] = label
    st[f"{minus_policy}_per_seed"] = {str(s): cM[s] for s in seeds}
    st[f"{base_policy}_per_seed"] = {str(s): cB[s] for s in seeds}
    return st


def main() -> None:
    res = _load()
    report: Dict[str, object] = {}

    # Primary factorial contrasts (lower cost better; Delta < 0 => improvement).
    report["n_step_under_standard"] = _contrast(res, "Cost_B - Cost_A", "B", "A")
    report["seeded_under_1step"] = _contrast(res, "Cost_C - Cost_A", "C", "A")
    report["n_step_under_seeded"] = _contrast(res, "Cost_D - Cost_C", "D", "C")
    report["seeded_under_3step"] = _contrast(res, "Cost_D - Cost_B", "D", "B")

    # Interaction: (Cost_D - Cost_C) - (Cost_B - Cost_A).
    ns_seeded = report["n_step_under_seeded"]["mean"]
    ns_std = report["n_step_under_standard"]["mean"]
    report["interaction"] = {
        "value": ns_seeded - ns_std,
        "definition": "(D-C) - (B-A)",
        "n_step_effect_seeded": ns_seeded,
        "n_step_effect_standard": ns_std,
    }

    # Enhanced cells vs M4 and H2.
    for pol in ("A", "B", "C", "D"):
        report[f"{pol}_minus_M4"] = _contrast(res, f"Cost_{pol} - M4", pol, "M4")
        report[f"{pol}_minus_H2"] = _contrast(res, f"Cost_{pol} - H2", pol, "H2")

    # Cross-seed means per policy.
    report["per_policy_mean"] = {}
    for pol in ("A", "B", "C", "D", "M4", "H2"):
        c = _per_seed_cost(res, pol)
        report["per_policy_mean"][pol] = {
            "mean": (sum(c.values()) / len(c)) if c else None,
            "per_seed": {str(s): v for s, v in c.items()},
        }
    report["historical_M9_A_reference_per_seed"] = {
        "6521": 13.0, "6522": 13.0, "6523": 13.0, "6524": 14.0, "6525": 13.0,
    }

    out_path = _CONTAINER / "m10_e3_outputs" / "eval" / "analysis_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path}")

    print("\n=== paired contrasts (mean, [95% CI]) ===")
    for k, v in report.items():
        if isinstance(v, dict) and "mean" in v and "ci_lo" in v:
            print(f"  {k} ({v['label']}): mean={v['mean']:.3f} "
                  f"[{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]")
    print(f"\n  interaction (D-C)-(B-A) = {report['interaction']['value']:.3f}")


if __name__ == "__main__":
    main()