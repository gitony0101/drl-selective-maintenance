#!/usr/bin/env python3
"""
Configuration Validation CLI for Milestone 5.

Uses the shared production config parser for consistency.

Usage:
    python scripts/validate_config.py --config configs/agents/ddqn_v1.json
    python scripts/validate_config.py --config configs/agents/ddqn_v1.json --check-assets
    python scripts/validate_config.py --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add repository root to path for src. imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.ddqn_config import (
    load_and_validate_config,
)


FORBIDDEN_SPLITS = frozenset({"rl_test"})


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate DDQN configuration (Milestone 5)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configuration JSON file",
    )

    parser.add_argument(
        "--check-assets",
        action="store_true",
        help="Check that referenced asset files exist",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="training",
        choices=["training", "evaluation", "matrix"],
        help="Validation mode (determines which validation rules apply)",
    )

    return parser.parse_args()


def check_asset_exists(path: str | None, asset_name: str) -> bool:
    """Check if an asset file exists."""
    if path is None:
        print(f"WARNING: {asset_name} path is None", file=sys.stderr)
        return False

    asset_path = Path(path)
    if not asset_path.exists():
        print(f"ERROR: {asset_name} not found: {asset_path}", file=sys.stderr)
        return False

    return True


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Load and validate using SHARED PRODUCTION PARSER
    try:
        parsed = load_and_validate_config(args.config, mode=args.mode)
    except SystemExit as e:
        sys.exit(e.code)
    except ValueError as e:
        print(f"ERROR: Config validation failed: {e}", file=sys.stderr)
        sys.exit(1)

    trainer_config = parsed.trainer_config

    # Print results
    print(f"Configuration validation: {args.config}")
    print(f"Mode: {args.mode}")
    print(f"{'='*50}")

    print(f"\n✓ Configuration valid")
    print(f"  Split: {trainer_config.split}")
    print(f"  Validation split: {trainer_config.validation_split}")
    print(f"  K: {trainer_config.maintenance_capacity}")
    print(f"  Cost regime: {trainer_config.cost_regime_id}")
    print(f"  Observation dim: 10")
    print(f"  Action count: {trainer_config.num_actions}")
    print(f"  Hidden dim: {trainer_config.hidden_dim}")
    print(f"  Num hidden layers: {trainer_config.num_hidden_layers}")
    print(f"  Learning rate: {trainer_config.learning_rate}")
    print(f"  Max steps: {trainer_config.max_steps}")
    print(f"  Training scenario bank: {trainer_config.training_scenario_bank_path}")
    print(f"  Validation scenario bank: {trainer_config.validation_scenario_bank_path}")

    if args.check_assets:
        print("\nAsset checks:")
        all_ok = True

        ok = check_asset_exists(
            trainer_config.training_scenario_bank_path, "training scenario bank"
        )
        all_ok = all_ok and ok

        ok = check_asset_exists(
            trainer_config.validation_scenario_bank_path, "validation scenario bank"
        )
        all_ok = all_ok and ok

        ok = check_asset_exists(
            trainer_config.prediction_cache_path, "prediction cache"
        )
        all_ok = all_ok and ok

        if all_ok:
            print("  ✓ All assets found")
        else:
            print("  ✗ Some assets missing")
            sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()