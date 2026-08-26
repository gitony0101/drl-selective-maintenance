"""E5 Task: build and freeze E5-B episode-level split manifests per seed.

12000 train / 1500 dynamics_validate / 1500 dynamics_holdout transitions.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from src.milestone11.e5.paths import E5_OUTPUT_ROOT, FORMAL_SEEDS
from src.milestone11.e5.split import build_split_manifest


def main() -> None:
    for seed in FORMAL_SEEDS:
        m = build_split_manifest(seed, E5_OUTPUT_ROOT)
        print(f"seed {seed}: train={m['totals']['dynamics_train']} "
              f"val={m['totals']['dynamics_validate']} "
              f"holdout={m['totals']['dynamics_holdout']} "
              f"sha={m['manifest_sha256'][:12]}")


if __name__ == "__main__":
    main()