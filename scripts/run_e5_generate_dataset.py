"""E5 (M12) Task: generate the E5-B failure-enriched training dataset per seed.

For each of the 5 formal seeds, collect the preregistered episodes per behavior
source on predictor_train + m5_pilot_k2.json + the seed's frozen M9 cache:

    random_feasible  50 episodes (5000 transitions)
    exact_myopic (M4)    50 episodes (5000 transitions)
    h2 (canonical H2)    20 episodes (2000 transitions)
    no_maintenance      30 episodes (3000 transitions)
                          ----------
    total              150 episodes (15000 transitions/seed)

The no-maintenance source uses a_t = 0 and exists ONLY to add genuine
failure-event coverage to the training data. Never touches rl_validation or
rl_test.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from src.milestone11.e5.dataset import collect_dataset, write_dataset
from src.milestone11.e5.paths import E5_OUTPUT_ROOT, FORMAL_SEEDS


def main() -> None:
    t0 = time.time()
    for seed in FORMAL_SEEDS:
        st = time.time()
        per_source = collect_dataset(seed, _SRC)
        write_dataset(per_source, seed, E5_OUTPUT_ROOT)
        for source, blob in per_source.items():
            ok = all(bool(v) for k, v in blob["integrity"].items())
            n_fail = sum(1 for t in blob["transitions"] if t.cost_failure > 0)
            print(f"seed {seed} [{source}] {len(blob['transitions'])} transitions "
                  f"integrity_ok={ok} failure_events={n_fail} ({time.time()-st:.1f}s)")
    summary = E5_OUTPUT_ROOT / "dataset" / "generation_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(__import__("json").dumps(
        {"seeds_generated": list(FORMAL_SEEDS),
         "episodes_per_source": {
             "random_feasible": 50, "exact_myopic": 50,
             "h2": 20, "no_maintenance": 30},
         "transitions_per_seed_target": 15000,
         "wall_clock_seconds": round(time.time() - t0, 1)},
        indent=2))
    print(f"wrote E5-B dataset for seeds {FORMAL_SEEDS} under {E5_OUTPUT_ROOT}")
    print(f"wall clock {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()