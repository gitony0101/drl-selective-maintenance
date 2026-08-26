"""E5 Task: train the 3-member dynamics ensemble for all formal seeds on E5-B.

Reuses the EXACT E4 dynamics-ensemble architecture and hyperparameters (Adam
1e-3, batch 256, 200 max epochs, patience 20, episode-level bootstrap, same
initialization-seed derivation and normalization protocol). The ONLY
experimental intervention is dataset failure coverage: it trains on E5-B
dynamics_train (failure-enriched) instead of the E4 original dataset.

The learned-dynamics architecture, optimizer, and bootstrap are byte-identical
to E4 (src.milestone11.e4.train.train_ensemble is reused verbatim). E4 and E5
differences are therefore as paired as practical by construction.

Frozen E4 model-training dataset per seed is NOT retrained (E5-A control =
frozen E4 result). This driver trains only the E5-B failure-enriched treatment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from src.milestone11.e4.train import train_ensemble
from src.milestone11.e5.paths import E5_OUTPUT_ROOT, FORMAL_SEEDS
from src.milestone11.e5.split import SOURCES, load_split_data


def main() -> None:
    device = sys.argv[1] if len(sys.argv) > 1 else None
    all_res = {}
    for seed in FORMAL_SEEDS:
        train_trans = []
        val_trans = []
        for src in SOURCES:
            for trans in load_split_data(seed, E5_OUTPUT_ROOT, "dynamics_train", src).values():
                train_trans.extend(trans)
        for src in SOURCES:
            for trans in load_split_data(seed, E5_OUTPUT_ROOT, "dynamics_validate", src).values():
                val_trans.extend(trans)
        res = train_ensemble(seed, train_trans, val_trans, E5_OUTPUT_ROOT, device=device)
        all_res[str(seed)] = {
            "formal_seed": seed,
            "member_results": res["member_results"],
            "member_init_seeds": res["member_init_seeds"],
            "device": res["device"],
            "control_e4_never_retrained": True,
        }
        for mi, mr in enumerate(res["member_results"]):
            print(f"seed {seed} member {mi}: best_epoch={mr['best_epoch']} "
                  f"best_val={mr['best_validation_loss']:.4f} "
                  f"val_state={mr['val_state_loss']:.4f} val_reward={mr['val_reward_loss']:.4f}")
    (E5_OUTPUT_ROOT / "training_summary.json").write_text(
        json.dumps(all_res, indent=2))
    print("E5-B training complete")


if __name__ == "__main__":
    main()