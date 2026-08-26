"""E4 (M11) Task 4: run the data coverage + integrity audit for all seeds."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from src.milestone11.e4.audit import write_audit
from src.milestone11.e4.paths import E4_OUTPUT_ROOT


def main() -> None:
    p = write_audit(E4_OUTPUT_ROOT)
    report = json.loads(p.read_text())
    for seed, r in report.items():
        o = r["overall"]
        print(f"seed {seed}: {o['transitions']} trans {o['episodes']} eps "
              f"span16={o['spans_16_actions']} splits={o['split_transition_counts']} "
              f"holdout_OOR={r['heldout_oor']['frac_out_of_train_range']:.4f}")
        for src in ("random_feasible", "exact_myopic", "h2"):
            a = r["per_source"][src]
            print(f"   {src}: actions16={a['actions_finite_all']} "
                  f"evt PM={a['event_counts']['preventive']} "
                  f"fail={a['event_counts']['failure']} "
                  f"waste={a['event_counts']['wasted_life']}")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()