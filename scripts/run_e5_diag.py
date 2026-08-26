"""E5 Task 10-11: one-step, two-step, and failure-specific diagnostics.

Runs held-out model diagnostics (including the new failure-specific reward
metrics) for the E5-B frozen ensemble of every formal seed on dynamics_holdout
and writes <m12_e5_outputs>/diagnostics/diag_<seed>.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from src.milestone11.e4.train import load_ensemble
from src.milestone11.e5.diag import run_diagnostics
from src.milestone11.e5.paths import E5_OUTPUT_ROOT, FORMAL_SEEDS


def main() -> None:
    device = sys.argv[1] if len(sys.argv) > 1 else None
    diags = {}
    for seed in FORMAL_SEEDS:
        ensemble = load_ensemble(seed, E5_OUTPUT_ROOT, device)
        model_device = next(ensemble.members[0].parameters()).device.type
        d = run_diagnostics(ensemble, seed, E5_OUTPUT_ROOT, model_device)
        diags[str(seed)] = d
        o = d["one_step"]
        fr = d["failure_reward"]["failure_transitions"]["metrics"]
        print(f"seed {seed}: obsRMSE={o['next_obs_rmse']:.4f} "
              f"(persist={o['baselines']['persistence_next_obs_rmse']:.4f}) "
              f"rewRMSE={o['reward_rmse']:.4f} "
              f"failure_rewMAE={fr['reward_mae'] if fr else 'NA'}")
    out = E5_OUTPUT_ROOT / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    (out / "diagnostics_summary.json").write_text(json.dumps(diags, indent=2))
    print(f"wrote {out / 'diagnostics_summary.json'}")


if __name__ == "__main__":
    main()