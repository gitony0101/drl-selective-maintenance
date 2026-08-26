"""E3 Part 4 driver: generate + write all M4/H2 training-only raw trajectories.

For each formal seed (6521..6525) and each policy (exact_myopic, h2), generate
raw ordered transitions on split=predictor_train, bank m5_pilot_k2.json, using
that seed's per-seed M9 prediction cache. Writes raw_transitions.jsonl,
provenance.json, integrity.json into:
    m10_e3_outputs/training_raw_trajectories/seed_<s>/<policy>/
plus a top-level summary manifest.

Prints the actual per-seed/per-policy transition counts and integrity verdicts.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.milestone10.e3.trajectories import (
    FORMAL_SEEDS,
    E3_OUTPUT_ROOT,
    TrajectorySet,
    check_integrity_all_pass,
    generate_trajectories,
    write_trajectory_set,
)


def main() -> None:
    summary = {"generated_at_utc": None, "seeds": {}, "all_integrity_pass": True}
    import datetime as _dt

    for seed in FORMAL_SEEDS:
        summary["seeds"][str(seed)] = {}
        for policy in ("exact_myopic", "h2"):
            ts: TrajectorySet = generate_trajectories(seed, policy)
            paths = write_trajectory_set(ts, seed)
            ok = check_integrity_all_pass(ts)
            summary["seeds"][str(seed)][policy] = {
                "count": ts.count,
                "per_episode": ts.integrity["per_episode_transition_counts"],
                "integrity_pass": ok,
                "integrity": ts.integrity,
                "provenance": ts.provenance,
            }
            if not ok:
                summary["all_integrity_pass"] = False
            print(
                f"seed {seed} {policy}: {ts.count} transitions, "
                f"integrity={'PASS' if ok else 'FAIL'}"
            )
            for label, p in paths.items():
                print(f"    {label}: {p}")
    if summary["all_integrity_pass"]:
        print("\nALL INTEGRITY CHECKS PASS for all seeds and policies.")
    else:
        print("\nFAILURE: at least one integrity check did not pass.")

    summary_path = E3_OUTPUT_ROOT / "training_raw_trajectories" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary: {summary_path}")
    return summary


if __name__ == "__main__":
    main()