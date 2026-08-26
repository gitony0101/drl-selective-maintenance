"""E4 (M11) Task 3: build and freeze episode-level split manifests per seed."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from src.milestone11.e4.paths import E4_OUTPUT_ROOT, FORMAL_SEEDS
from src.milestone11.e4.split import build_split_manifest


def main() -> None:
    for seed in FORMAL_SEEDS:
        m = build_split_manifest(seed, E4_OUTPUT_ROOT)
        print(f"seed {seed}: train={m['totals']['dynamics_train']} "
              f"val={m['totals']['dynamics_validate']} "
              f"holdout={m['totals']['dynamics_holdout']} "
              f"sha={m['manifest_sha256'][:12]}")


if __name__ == "__main__":
    main()