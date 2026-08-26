"""E5 Task: failure-coverage manipulation audit (Section 6).

Writes e5_coverage_audit.json and prints the per-seed manipulation gate. Fails
(exit 1) if any formal seed's dynamics_train does not contain >0 genuine
failure events, or if all 16 actions are not covered in dynamics_train.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from src.milestone11.e5.coverage import audit_all_seeds, write_audit
from src.milestone11.e5.paths import E5_OUTPUT_ROOT, FORMAL_SEEDS


def main() -> None:
    report = audit_all_seeds(E5_OUTPUT_ROOT)
    p = write_audit(E5_OUTPUT_ROOT)
    all_pass = True
    for seed in FORMAL_SEEDS:
        g = report[str(seed)]["manipulation_gate"]
        ok = g["dynamics_train_has_gt0_failures"] and g["all_16_actions_in_train"]
        train_fails = report[str(seed)]["train_failure_events"]
        print(f"seed {seed}: train_failure_events={train_fails} "
              f"train_gate_ok={ok} {g}")
        if not ok:
            all_pass = False
    print(f"wrote {p}")
    if not all_pass:
        print("FAILURE-COVERAGE MANIPULATION GATE FAILED")
        sys.exit(1)
    print("FAILURE-COVERAGE MANIPULATION GATE PASSED for all seeds")


if __name__ == "__main__":
    main()