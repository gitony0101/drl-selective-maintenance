"""E3 formal training driver: all four cells A/B/C/D across seeds 6521-6525.

Usage:
    python scripts/run_e3_formal.py --smoke            # tiny runs for execution correctness
    python scripts/run_e3_formal.py --cell A           # one cell, all 5 seeds
    python scripts/run_e3_formal.py --cell A --seed 6521
    python scripts/run_e3_formal.py --all              # all cells x all seeds

Outputs (per cell x seed) go to <m10_e3_outputs>/formal/<cell>/seed_<s>/
Each run writes run_manifest.json, step_metrics.jsonl, validation_metrics.json,
checkpoint_best.pt, checkpoint_latest.pt via the E3 trainer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


from src.milestone10.e3.trainer import E3CellConfig, E3Trainer, FORMAL_SEEDS  # noqa
from src.milestone10.e3.h2_context import seed_cache_dir  # noqa


from src.runtime_paths import external_root as _external_root

_CONTAINER = _external_root()
E3_OUTPUT = _CONTAINER / "m10_e3_outputs"
FORMAL_ROOT = E3_OUTPUT / "formal"

MANIFEST_ROOT = E3_OUTPUT / "seeded_warmup_manifests"

CELL_DEFS = {
    "A": {"n": 1, "mode": "standard"},
    "B": {"n": 3, "mode": "standard"},
    "C": {"n": 1, "mode": "seeded"},
    "D": {"n": 3, "mode": "seeded"},
}


def _seeded_manifest(seed: int) -> Path:
    return MANIFEST_ROOT / f"seed_{seed}" / "seeded_warmup_raw.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=sorted(CELL_DEFS), default=None)
    ap.add_argument("--seed", type=int, choices=list(FORMAL_SEEDS), default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny max_steps run for execution-correctness only")
    ap.add_argument("--max-steps", type=int, default=100_000,
                    help="override max_steps (smoke uses this)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cells = sorted(CELL_DEFS) if args.all or args.smoke else [args.cell]
    if args.cell is None and not args.all and not args.smoke:
        raise SystemExit("must pass --cell, --all, or --smoke")

    # NOTE: warmup is always the frozen 5000 transitions (Section 17/18); the
    # smoke budget must exceed 5000 to exercise the online n-step update loop.
    max_steps = 6_000 if args.smoke else args.max_steps

    records = []
    for cell in cells:
        d = CELL_DEFS[cell]
        n, mode = d["n"], d["mode"]
        seeds = list(FORMAL_SEEDS) if args.seed is None else [args.seed]
        for seed in seeds:
            run_id = f"e3_{cell}_seed{seed}"
            run_dir = FORMAL_ROOT / cell / f"seed_{seed}"
            if (run_dir / "run_manifest.json").exists():
                print(f"SKIP existing: {cell} seed {seed}")
                records.append({"cell": cell, "seed": seed, "status": "SKIP"})
                continue
            cell_cfg = E3CellConfig(n=n, cell=cell, warmup_mode=mode, max_steps=max_steps)
            trainer = E3Trainer(
                seed=seed, cell_cfg=cell_cfg,
                cache_path=str(seed_cache_dir(seed)),
                output_dir=str(FORMAL_ROOT / cell),
                run_id=f"seed_{seed}",
                device=args.device,
                seeded_manifest_path=_seeded_manifest(seed) if mode == "seeded" else None,
            )
            print(f"RUN {cell} seed {seed}: n={n} mode={mode} max_steps={max_steps}", flush=True)
            metrics = trainer.train()
            status = "COMPLETE" if trainer.global_step >= max_steps else "INCOMPLETE"
            print(f"  DONE {cell} seed {seed}: steps={trainer.global_step} "
                  f"best_val={trainer.best_validation_mean_cost} status={status}", flush=True)
            records.append({"cell": cell, "seed": seed, "status": status,
                            "best_validation_mean_cost": trainer.best_validation_mean_cost,
                            "checkpoint": trainer.best_checkpoint_path})

    # Write a run summary.
    summary_path = FORMAL_ROOT / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({
        "smoke": args.smoke, "max_steps": max_steps, "records": records,
        "generated_at_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    }, indent=2))
    print(f"\nsummary: {summary_path}")


if __name__ == "__main__":
    main()