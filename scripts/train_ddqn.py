#!/usr/bin/env python3
"""
DDQN Training CLI for Milestone 5.

Usage:
    python scripts/train_ddqn.py --config configs/agents/ddqn_v1.json
    python scripts/train_ddqn.py --config configs/agents/ddqn_v1.json --resume checkpoints/latest.pt
    python scripts/train_ddqn.py --help

Barrier: Rejects rl_test split before loading any data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add repository root to path for src. imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.ddqn_trainer import DDQNTrainer
from src.training.ddqn_config import load_and_validate_config, apply_cli_overrides
from src.training.resolver import resolve_argparse_namespace, ExplicitBankError
from src.training.preflight import validate_row_asset_contract


FORBIDDEN_SPLITS = frozenset({"rl_test"})

# M5 provenance binding: the mandatory explicit-bank gate lives in the SHARED
# authoritative resolver (src/training.resolver), not here.  Every formal
# launch -- dry-run, validate-only, smoke, normal training, and resume -- MUST
# pass BOTH --training-scenario-bank <path> and --validation-scenario-bank
# <path> so the effective config matches the matrix row exactly.  The base
# agent configs only point at the baseline light-no-waste banks, so omitting
# these flags would silently bind to the wrong regime.  There is NO bypass:
# the previous opt-out flag is removed entirely.


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DDQN agent for selective maintenance (Milestone 5)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configuration JSON file",
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint for resume",
    )

    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Override training split (must not be rl_test)",
    )

    parser.add_argument(
        "--validation-split",
        type=str,
        default=None,
        help="Override validation split (must not be rl_test)",
    )

    parser.add_argument(
        "--cost-regime",
        type=str,
        default=None,
        help="Override cost regime",
    )

    parser.add_argument(
        "--k-capacity",
        type=int,
        default=None,
        help="Override maintenance capacity K (1 or 2)",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override maximum training steps",
    )

    parser.add_argument(
        "--training-seed",
        type=int,
        default=None,
        help="Override training seed",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )

    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Explicit run identifier for deterministic output path",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "mps", "cuda"],
        help="Override device",
    )

    parser.add_argument(
        "--training-scenario-bank",
        type=str,
        default=None,
        help=("M5 provenance binding: path to the regime-specific training scenario "
              "bank, e.g. configs/scenarios/m5_pilot_k1__heavy.json.  REQUIRED "
              "for formal launch so the effective config matches the matrix "
              "row exactly; overrides the base config's training_scenario_bank_path."),
    )

    parser.add_argument(
        "--validation-scenario-bank",
        type=str,
        default=None,
        help=("M5 provenance binding: path to the regime-specific validation "
              "scenario bank, e.g. configs/scenarios/m5_validation_k1__heavy.json. "
              "REQUIRED for formal launch so the effective config matches the "
              "matrix row exactly; overrides the base config's "
              "validation_scenario_bank_path."),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and exit without training",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate configuration, splits, and assets without training (implies --dry-run)",
    )

    return parser.parse_args()


def validate_split(split: str, split_name: str) -> None:
    """Validate that split is not forbidden."""
    if split in FORBIDDEN_SPLITS:
        print(f"ERROR: {split_name}='rl_test' is FORBIDDEN.", file=sys.stderr)
        print(
            "Milestone 5 training must use split='predictor_train'.",
            file=sys.stderr,
        )
        print(
            "Milestone 5 validation must use split='rl_validation'.",
            file=sys.stderr,
        )
        print(
            "rl_test is sealed and must not be accessed for training or tuning.",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Load and validate configuration using the SHARED production resolver,
    # which applies CLI overrides through the SAME override-dict semantics
    # that ``src.training.resolver.resolve_command_to_effective`` uses for
    # matrix / preflight / smoke paths AND enforces the mandatory explicit-
    # bank gate (the frozen "always require explicit banks" decision).  This
    # is the M5 provenance binding: one authoritative resolver + one gate, no
    # synthetic config drift and no bypass.
    try:
        trainer_config = resolve_argparse_namespace(args, mode="training")
    except ExplicitBankError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except SystemExit as e:
        sys.exit(e.code)
    except ValueError as e:
        print(f"ERROR: Config validation failed: {e}", file=sys.stderr)
        sys.exit(1)

    # BARRIER: Validate splits BEFORE loading any data
    validate_split(trainer_config.split, "split")
    validate_split(trainer_config.validation_split, "validation_split")

    # Validate K
    if trainer_config.maintenance_capacity not in (1, 2):
        print(
            f"ERROR: maintenance_capacity must be 1 or 2, "
            f"got {trainer_config.maintenance_capacity}",
            file=sys.stderr,
        )
        sys.exit(1)

    # If --validate-only, run the strengthened asset-contract preflight that
    # verifies BOTH scenario banks load, every scenario's cost_regime_id /
    # maintenance_capacity / split matches the effective row, the
    # prediction-cache compatibility check holds, and that no rl_test access
    # is permitted.  This must run BEFORE any Trainer construction (which
    # would create an output directory / checkpoint immediately) so that the
    # exact blocker that escaped the prior preflight is now caught.
    if args.validate_only:
        report = validate_row_asset_contract(
            training_scenario_bank_path=trainer_config.training_scenario_bank_path,
            validation_scenario_bank_path=trainer_config.validation_scenario_bank_path,
            cost_regime_id=trainer_config.cost_regime_id,
            maintenance_capacity=trainer_config.maintenance_capacity,
            prediction_cache_path=trainer_config.prediction_cache_path,
            training_split=trainer_config.split,
            validation_split=trainer_config.validation_split,
        )
        if not report.ok:
            print(
                "ERROR: Asset-contract preflight FAILED:",
                file=sys.stderr,
            )
            for err in report.errors:
                print(f"  - {err}", file=sys.stderr)
            for warn in report.warnings:
                print(f"  WARN: {warn}", file=sys.stderr)
            sys.exit(1)

    # Dry run mode: validate configuration and assets, then exit
    if args.dry_run or args.validate_only:
        print("Configuration validated successfully (dry-run mode).")
        print(f"  Split: {trainer_config.split}")
        print(f"  Validation split: {trainer_config.validation_split}")
        print(f"  K: {trainer_config.maintenance_capacity}")
        print(f"  Actions: {trainer_config.num_actions}")
        print(f"  Cost regime: {trainer_config.cost_regime_id}")
        print(f"  Max steps: {trainer_config.max_steps}")
        print(f"  Output dir: {trainer_config.output_dir}")
        if trainer_config.run_id:
            print(f"  Run ID: {trainer_config.run_id}")
        sys.exit(0)

    # Validate checkpoint exists if resuming
    if args.resume is not None:
        if not args.resume.exists():
            print(f"ERROR: Checkpoint not found: {args.resume}", file=sys.stderr)
            sys.exit(1)

    # Create trainer and train
    print(f"Starting DDQN training...")
    print(f"  Config: {args.config}")
    print(f"  Split: {trainer_config.split}")
    print(f"  Validation split: {trainer_config.validation_split}")
    print(f"  K: {trainer_config.maintenance_capacity}")
    print(f"  Cost regime: {trainer_config.cost_regime_id}")
    print(f"  Max steps: {trainer_config.max_steps}")
    print(f"  Output dir: {trainer_config.output_dir}")

    if args.resume:
        print(f"  Resuming from: {args.resume}")

    try:
        trainer = DDQNTrainer(config=trainer_config, resume_from=args.resume)
        metrics = trainer.train()

        print(f"\nTraining complete!")
        print(f"  Episodes: {trainer.episode_count}")
        print(f"  Global steps: {trainer.global_step}")
        print(f"  Best validation mean cost: {metrics.best_validation_mean_cost}")
        print(f"  Best checkpoint: {metrics.best_checkpoint_path}")
        print(f"  Run directory: {trainer.run_dir}")

    except Exception as e:
        print(f"\nERROR: Training failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()