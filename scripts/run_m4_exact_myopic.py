#!/usr/bin/env python3
"""
Milestone 4 Exact Myopic Optimizer - Main CLI.

Usage:
    python scripts/run_m4_exact_myopic.py --help
    python scripts/run_m4_exact_myopic.py --smoke
    python scripts/run_m4_exact_myopic.py --evaluate --split rl_validation --k-capacity 2
    python scripts/run_m4_exact_myopic.py --tune --split rl_validation

Rejects rl_test split before loading any data.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from optimizers import (
    MyopicContext,
    ExactMyopicOptimizer,
    MyopicArtifactWriter,
    get_git_commit,
    write_atomic_json,
)
from envs.action_table import ACTION_TABLE_N5_K1, ACTION_TABLE_N5_K2
from envs.costs import COST_REGIMES, get_cost_regime
from envs.config import get_default_config


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="run_m4_exact_myopic",
        description="Milestone 4 Exact Myopic Current-Window Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Smoke test (quick validation with synthetic observations)
    python run_m4_exact_myopic.py --smoke

    # Production evaluation (16-config smoke matrix with real environment)
    python run_m4_exact_myopic.py --evaluate --overwrite

    # Threshold tuning (rl_validation only, rl_test rejected)
    # NOTE: --tune is rejected with nonzero exit. Tuning deferred to M3.
    python run_m4_exact_myopic.py --tune --split rl_validation  # Returns exit code 1
        """,
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke test (quick validation with minimal episodes)",
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run full evaluation",
    )

    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning (rl_validation only)",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="rl_validation",
        choices=["predictor_train", "rl_validation", "rl_test"],
        help="Data split to use (rl_test is rejected)",
    )

    parser.add_argument(
        "--k-capacity",
        type=int,
        default=2,
        choices=[1, 2],
        help="Maintenance capacity K (1 or 2)",
    )

    parser.add_argument(
        "--cost-regime",
        type=str,
        default="failure-light-no-waste",
        choices=list(COST_REGIMES.keys()),
        help="Cost regime to use",
    )

    parser.add_argument(
        "--risk-model",
        type=str,
        default="hard_window_v1",
        choices=["hard_window_v1", "logistic_window_v1"],
        help="Failure risk model",
    )

    parser.add_argument(
        "--risk-temperature",
        type=float,
        default=10.0,
        help="Temperature for logistic risk model",
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default="6521,6522,6523,6524,6525",
        help="Comma-separated random seeds",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config JSON file (default: configs/myopic/m4_exact_myopic_v1.json)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/milestone4/<timestamp>)",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and exit without running",
    )

    return parser.parse_args()


def validate_split_barrier(split: str) -> None:
    """
    Validate split is allowed.

    Rejects rl_test before any data loading.

    Args:
        split: Split name to validate.

    Raises:
        SystemExit: If split is rl_test.
    """
    if split == "rl_test":
        print(
            "ERROR: rl_test split is forbidden during development.\n"
            "The rl_test split is reserved for final evaluation after Milestone 5.\n"
            "Use predictor_train or rl_validation instead.",
            file=sys.stderr,
        )
        sys.exit(1)

    if split == "predictor_validation":
        print(
            "ERROR: predictor_validation is not a valid environment split.\n"
            "Use predictor_train, rl_validation, or rl_test (rl_test forbidden in dev).",
            file=sys.stderr,
        )
        sys.exit(1)


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Load configuration from file or defaults."""
    if args.config is not None:
        with open(args.config, "r") as f:
            return json.load(f)

    # Default config path
    default_config = Path(__file__).parent.parent / "configs" / "myopic" / "m4_exact_myopic_v1.json"
    if default_config.exists():
        with open(default_config, "r") as f:
            return json.load(f)

    # Return minimal defaults
    return {
        "schema_version": "m4_v1",
        "policy_id": "exact_myopic_v1",
        "risk_model_id": "hard_window_v1",
        "risk_temperature": 10.0,
        "tie_tolerance": 1e-9,
    }


def create_optimizer(
    k_capacity: int,
    cost_regime_id: str,
    risk_model_id: str,
    risk_temperature: float,
    tie_tolerance: float = 1e-9,
) -> ExactMyopicOptimizer:
    """Create optimizer with given parameters."""
    # Get cost regime
    cost_regime = get_cost_regime(cost_regime_id)

    # Get action table
    if k_capacity == 1:
        action_table = ACTION_TABLE_N5_K1
    elif k_capacity == 2:
        action_table = ACTION_TABLE_N5_K2
    else:
        raise ValueError(f"Invalid K capacity: {k_capacity}")

    # Create context
    context = MyopicContext(
        maintenance_capacity=k_capacity,
        delta_cycles=5,
        rul_scale=125.0,
        age_scale_cycles=341,
        action_table=action_table,
        c_pm=cost_regime.c_pm,
        c_f=cost_regime.c_f,
        c_u=cost_regime.c_u,
        risk_model_id=risk_model_id,
    )

    return ExactMyopicOptimizer(
        context=context,
        risk_temperature=risk_temperature,
        tie_tolerance=tie_tolerance,
    )


def run_smoke_test(
    optimizer: ExactMyopicOptimizer,
    split: str,
    k_capacity: int,
    cost_regime_id: str,
    seeds: list[int],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Run smoke test.

    Creates synthetic test observations and verifies:
    - Action selection succeeds
    - Costs are finite
    - Action IDs are valid
    """
    num_actions = len(optimizer.context.action_table)
    results: List[Dict[str, Any]] = []

    for seed in seeds:
        rng = np.random.default_rng(seed)

        # Generate synthetic observation (normalized, in [0, 1])
        observation = rng.uniform(0.1, 0.9, size=(10,)).astype(np.float32)

        # Select action
        action_id, selected_slots, estimated_cost = optimizer.select_action(observation)

        # Validate
        assert 0 <= action_id < num_actions, f"Invalid action_id: {action_id}"
        assert len(selected_slots) <= k_capacity, f"Selected {len(selected_slots)} > K={k_capacity}"
        assert np.isfinite(estimated_cost), f"Cost not finite: {estimated_cost}"

        # Evaluate all actions for smoke report
        evaluations = optimizer.evaluate_all_actions(observation)
        all_costs = [e.total_cost for e in evaluations]

        results.append({
            "seed": seed,
            "action_id": action_id,
            "selected_slots": list(selected_slots),
            "estimated_cost": float(estimated_cost),
            "min_action_cost": float(min(all_costs)),
            "max_action_cost": float(max(all_costs)),
            "all_costs_finite": all(np.isfinite(c) for c in all_costs),
        })

    # Write smoke report
    report = {
        "schema_version": "m4_v1",
        "mode": "smoke",
        "split": split,
        "k_capacity": k_capacity,
        "cost_regime_id": cost_regime_id,
        "risk_model_id": optimizer.context.risk_model_id,
        "seeds": seeds,
        "num_episodes": len(results),
        "all_passed": all(r["all_costs_finite"] for r in results),
        "episode_results": results,
    }

    write_atomic_json(report, output_dir / "smoke_report.json")

    return report


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Validate split barrier BEFORE any data loading
    validate_split_barrier(args.split)

    # Load configuration
    config = load_config(args)

    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = Path(config.get("output_base_path", "results/milestone4"))
        output_dir = output_base / f"m4_run_{timestamp}"

    # Check if output dir exists
    if output_dir.exists() and not args.overwrite and not args.smoke:
        print(f"ERROR: Output directory exists: {output_dir}", file=sys.stderr)
        print("Use --overwrite to overwrite or --smoke for smoke test", file=sys.stderr)
        return 1

    # Dry run - just validate and exit
    if args.dry_run:
        print("Configuration validated successfully.")
        print(f"  Split: {args.split}")
        print(f"  K capacity: {args.k_capacity}")
        print(f"  Cost regime: {args.cost_regime}")
        print(f"  Risk model: {args.risk_model}")
        print(f"  Seeds: {seeds}")
        return 0

    # Reject unimplemented modes early
    if args.tune:
        print("\nERROR: --tune mode is not implemented.", file=sys.stderr)
        print("  Threshold tuning is deferred to M3 integration.", file=sys.stderr)
        print("  This M4 implementation covers optimizer and production evaluation,", file=sys.stderr)
        print("  not scientific tuning closeout.", file=sys.stderr)
        return 1

    # Create optimizer
    try:
        optimizer = create_optimizer(
            k_capacity=args.k_capacity,
            cost_regime_id=args.cost_regime,
            risk_model_id=args.risk_model,
            risk_temperature=args.risk_temperature,
            tie_tolerance=config.get("tie_tolerance", 1e-9),
        )
    except ValueError as e:
        print(f"ERROR: Failed to create optimizer: {e}", file=sys.stderr)
        return 1

    print(f"Milestone 4 Exact Myopic Optimizer")
    print(f"  Split: {args.split}")
    print(f"  K capacity: {args.k_capacity}")
    print(f"  Cost regime: {args.cost_regime}")
    print(f"  Risk model: {args.risk_model}")
    print(f"  Output: {output_dir}")

    # Run production evaluation
    if args.evaluate:
        print("\nRunning production evaluation smoke matrix...")
        # Import and run production smoke matrix
        from run_m4_production_smoke import run_production_smoke_matrix

        report = run_production_smoke_matrix(output_dir, args.config)

        if report["all_passed"]:
            print(f"\n  Production evaluation PASSED ({report['passed_configs']}/{report['total_configs']} configs)")
            print(f"  Artifacts written to: {output_dir}")
            return 0
        else:
            print(f"\n  Production evaluation FAILED ({report['failed_configs']}/{report['total_configs']} configs failed)", file=sys.stderr)
            return 1

    # Run smoke test
    if args.smoke:
        print("\nRunning smoke test...")
        report = run_smoke_test(
            optimizer=optimizer,
            split=args.split,
            k_capacity=args.k_capacity,
            cost_regime_id=args.cost_regime,
            seeds=seeds[:1] if len(seeds) > 1 else seeds,  # Just 1 seed for smoke
            output_dir=output_dir,
        )

        if report["all_passed"]:
            print(f"  Smoke test PASSED ({len(report['episode_results'])} episodes)")
            return 0
        else:
            print("  Smoke test FAILED", file=sys.stderr)
            return 1

    # Default: smoke test (if no mode specified)
    print("\nNo mode specified, running smoke test...")
    report = run_smoke_test(
        optimizer=optimizer,
        split=args.split,
        k_capacity=args.k_capacity,
        cost_regime_id=args.cost_regime,
        seeds=seeds[:1],
        output_dir=output_dir,
    )

    if report["all_passed"]:
        print(f"  Smoke test PASSED")
        return 0
    else:
        print("  Smoke test FAILED", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())