"""E4 (M11) Task 7: train the 3-member dynamics ensemble for all formal seeds.

Trains each ensemble member from dynamics_train with validation early-stopping
and freezes model + normalization + per-member results under
<out_root>/models/seed_<s>/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from src.milestone11.e4.paths import E4_OUTPUT_ROOT, FORMAL_SEEDS
from src.milestone11.e4.split import load_split_data
from src.milestone11.e4.train import train_ensemble


def main() -> None:
    device = sys.argv[1] if len(sys.argv) > 1 else None
    all_res = {}
    for seed in FORMAL_SEEDS:
        train_trans = []
        for src in ("random_feasible", "exact_myopic", "h2"):
            for trans in load_split_data(seed, E4_OUTPUT_ROOT, "dynamics_train", src).values():
                train_trans.extend(trans)
        val_trans = []
        for src in ("random_feasible", "exact_myopic", "h2"):
            for trans in load_split_data(seed, E4_OUTPUT_ROOT, "dynamics_validate", src).values():
                val_trans.extend(trans)
        res = train_ensemble(seed, train_trans, val_trans, E4_OUTPUT_ROOT, device=device)
        all_res[str(seed)] = {
            "formal_seed": seed,
            "member_results": res["member_results"],
            "member_init_seeds": res["member_init_seeds"],
            "device": res["device"],
        }
        for mi, mr in enumerate(res["member_results"]):
            print(f"seed {seed} member {mi}: best_epoch={mr['best_epoch']} "
                  f"best_val={mr['best_validation_loss']:.4f} "
                  f"val_state={mr['val_state_loss']:.4f} val_reward={mr['val_reward_loss']:.4f}")
    (E4_OUTPUT_ROOT / "training_summary.json").write_text(
        json.dumps(all_res, indent=2))
    print("training complete")


if __name__ == "__main__":
    main()