"""E4 (M11) Task 2: generate the training-only 15k dynamics dataset per formal seed.

For each of the 5 formal seeds, collect 50 episodes (5000 transitions) per
behavior source (random_feasible / exact_myopic / h2) on predictor_train +
m5_pilot_k2.json + the seed's frozen M9 cache, and write the E4 dataset +
provenance + integrity. Never touches rl_validation or rl_test.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from src.milestone11.e4.dataset import collect_dataset, write_dataset
from src.milestone11.e4.paths import E4_OUTPUT_ROOT, FORMAL_SEEDS


def main() -> None:
    t0 = time.time()
    written = {}
    for seed in FORMAL_SEEDS:
        st = time.time()
        per_source = collect_dataset(seed, _SRC)
        paths = write_dataset(per_source, seed, E4_OUTPUT_ROOT)
        # Validate integrity for this seed.
        for source, blob in per_source.items():
            ok = all(bool(v) for k, v in blob["integrity"].items())
            print(f"seed {seed} [{source}] {len(blob['transitions'])} transitions "
                  f"integrity_ok={ok} ({time.time()-st:.1f}s)")
        written[int(seed)] = paths
    summary = E4_OUTPUT_ROOT / "dataset" / "generation_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(
        {"seeds_generated": list(FORMAL_SEEDS),
         "episodes_per_source": 50,
         "transitions_per_source_target": 5000,
         "wall_clock_seconds": round(time.time() - t0, 1)},
        indent=2))
    print(f"wrote dataset for seeds {FORMAL_SEEDS} under {E4_OUTPUT_ROOT}")
    print(f"wall clock {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()