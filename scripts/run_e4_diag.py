"""E4 (M11) Tasks 8-10: one-step, two-step, and partial-observability diagnostics.

Runs held-out model diagnostics for the frozen ensemble of every formal seed and
writes <out_root>/diagnostics/diag_<seed>.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from src.milestone11.e4.diag import run_all_diags
from src.milestone11.e4.paths import E4_OUTPUT_ROOT, FORMAL_SEEDS
from src.milestone11.e4.train import load_ensemble


def main() -> None:
    device = sys.argv[1] if len(sys.argv) > 1 else None
    diags = {}
    for seed in FORMAL_SEEDS:
        ensemble = load_ensemble(seed, E4_OUTPUT_ROOT, device)
        model_device = next(ensemble.members[0].parameters()).device.type
        d = run_all_diags(ensemble, seed, E4_OUTPUT_ROOT, model_device)
        diags[str(seed)] = d
        o = d["one_step"]
        t = d["two_step"]
        gate = (o["hard_gate"]["model_beats_persistence"]
                and o["hard_gate"]["model_beats_per_action_reward"])
        print(f"seed {seed}: obsRMSE={o['next_obs_rmse']:.4f} (persist={o['baselines']['persistence_next_obs_rmse']:.4f}) "
              f"rewRMSE={o['reward_rmse']:.4f} 2stepRMSE={t['two_step_state_rmse'] if t['two_step_state_rmse'] else 0:.4f} "
              f"hard_gate={gate}")
    out = E4_OUTPUT_ROOT / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    (out / "diagnostics_summary.json").write_text(json.dumps(diags, indent=2))
    print(f"wrote {out / 'diagnostics_summary.json'}")


if __name__ == "__main__":
    main()