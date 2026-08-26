#!/usr/bin/env python3
"""
Milestone 4 Exact Myopic Validator.

Runs focused validation tests and smoke matrix.

Usage:
    python scripts/validate_m4_exact_myopic.py --help
    python scripts/validate_m4_exact_myopic.py --smoke-matrix
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from optimizers import (
    MyopicContext,
    ExactMyopicOptimizer,
    get_git_commit,
    write_atomic_json,
    compute_failure_risk,
)
from envs.action_table import ACTION_TABLE_N5_K1, ACTION_TABLE_N5_K2
from envs.costs import COST_REGIMES, get_cost_regime


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="validate_m4_exact_myopic",
        description="Milestone 4 Validator",
    )

    parser.add_argument(
        "--smoke-matrix",
        action="store_true",
        help="Run full smoke matrix (K=1,2 × all regimes × both splits)",
    )

    parser.add_argument(
        "--test-risk-models",
        action="store_true",
        help="Test failure risk models",
    )

    parser.add_argument(
        "--test-enumeration",
        action="store_true",
        help="Test action enumeration",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for validation report",
    )

    return parser.parse_args()


def test_risk_models() -> bool:
    """Test failure risk models."""
    print("\n=== Testing Failure Risk Models ===")
    passed = 0
    failed = 0

    # Test hard window risk
    print("\n1. Hard window risk...")
    test_ruls = [1, 3, 5, 6, 10, 50, 100]
    for rul in test_ruls:
        risk = compute_failure_risk(
            predicted_rul_cycles=rul,
            delta_cycles=5,
            risk_model_id="hard_window_v1",
        )
        expected = 1.0 if rul <= 5 else 0.0
        if abs(risk - expected) < 1e-9:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: RUL={rul}, risk={risk}, expected={expected}")

    # Test logistic risk bounds
    print("\n2. Logistic risk bounds...")
    for rul in [1, 5, 10, 50, 100]:
        risk = compute_failure_risk(
            predicted_rul_cycles=rul,
            delta_cycles=5,
            risk_model_id="logistic_window_v1",
            risk_temperature=10.0,
        )
        if 0.0 < risk < 1.0:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: RUL={rul}, risk={risk} not in (0, 1)")

    # Test logistic monotonicity (lower RUL → higher risk)
    print("\n3. Logistic monotonicity...")
    risk_low = compute_failure_risk(
        predicted_rul_cycles=10,
        delta_cycles=5,
        risk_model_id="logistic_window_v1",
        risk_temperature=10.0,
    )
    risk_high = compute_failure_risk(
        predicted_rul_cycles=50,
        delta_cycles=5,
        risk_model_id="logistic_window_v1",
        risk_temperature=10.0,
    )
    if risk_low > risk_high:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: risk(10)={risk_low} <= risk(50)={risk_high}")

    # Test invalid temperature
    print("\n4. Invalid temperature rejection...")
    try:
        compute_failure_risk(
            predicted_rul_cycles=10,
            delta_cycles=5,
            risk_model_id="logistic_window_v1",
            risk_temperature=-1.0,
        )
        failed += 1
        print("  FAIL: Should have raised ValueError for negative temperature")
    except ValueError:
        passed += 1

    print(f"\nRisk model tests: {passed} passed, {failed} failed")
    return failed == 0


def test_enumeration() -> bool:
    """Test action enumeration."""
    print("\n=== Testing Action Enumeration ===")
    passed = 0
    failed = 0

    # K=1: should have 6 actions
    print("\n1. K=1 action count...")
    if len(ACTION_TABLE_N5_K1) == 6:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: K=1 has {len(ACTION_TABLE_N5_K1)} actions, expected 6")

    # K=2: should have 16 actions
    print("\n2. K=2 action count...")
    if len(ACTION_TABLE_N5_K2) == 16:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: K=2 has {len(ACTION_TABLE_N5_K2)} actions, expected 16")

    # Action 0 should be empty for both
    print("\n3. Action 0 is empty...")
    if ACTION_TABLE_N5_K1[0] == () and ACTION_TABLE_N5_K2[0] == ():
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: Action 0 not empty: K1={ACTION_TABLE_N5_K1[0]}, K2={ACTION_TABLE_N5_K2[0]}")

    # All actions should have <= K slots
    print("\n4. Capacity constraint...")
    k1_ok = all(len(a) <= 1 for a in ACTION_TABLE_N5_K1)
    k2_ok = all(len(a) <= 2 for a in ACTION_TABLE_N5_K2)
    if k1_ok and k2_ok:
        passed += 1
    else:
        failed += 1
        print("  FAIL: Some actions exceed capacity")

    # All slot indices should be in [0, 4]
    print("\n5. Slot index bounds...")
    k1_valid = all(0 <= s < 5 for a in ACTION_TABLE_N5_K1 for s in a)
    k2_valid = all(0 <= s < 5 for a in ACTION_TABLE_N5_K2 for s in a)
    if k1_valid and k2_valid:
        passed += 1
    else:
        failed += 1
        print("  FAIL: Some slot indices out of bounds")

    print(f"\nEnumeration tests: {passed} passed, {failed} failed")
    return failed == 0


def run_smoke_matrix(output_dir: Path | None = None) -> bool:
    """Run full smoke matrix."""
    print("\n=== Running Smoke Matrix ===")

    splits = ["predictor_train", "rl_validation"]
    k_values = [1, 2]
    regimes = list(COST_REGIMES.keys())

    results = []
    all_passed = True

    for split in splits:
        for k in k_values:
            for regime_id in regimes:
                config_key = f"{split}_K{k}_{regime_id}"

                # Create optimizer
                cost_regime = get_cost_regime(regime_id)
                action_table = ACTION_TABLE_N5_K1 if k == 1 else ACTION_TABLE_N5_K2

                context = MyopicContext(
                    maintenance_capacity=k,
                    delta_cycles=5,
                    rul_scale=125.0,
                    age_scale_cycles=341,
                    action_table=action_table,
                    c_pm=cost_regime.c_pm,
                    c_f=cost_regime.c_f,
                    c_u=cost_regime.c_u,
                    risk_model_id="hard_window_v1",
                )

                optimizer = ExactMyopicOptimizer(context=context)

                # Test with synthetic observation
                rng = np.random.default_rng(6521)
                observation = rng.uniform(0.1, 0.9, size=(10,)).astype(np.float32)

                try:
                    action_id, slots, cost = optimizer.select_action(observation)
                    evaluations = optimizer.evaluate_all_actions(observation)

                    # Validate
                    valid = (
                        0 <= action_id < len(action_table) and
                        len(slots) <= k and
                        np.isfinite(cost) and
                        all(np.isfinite(e.total_cost) for e in evaluations)
                    )

                    results.append({
                        "config": config_key,
                        "split": split,
                        "k_capacity": k,
                        "cost_regime_id": regime_id,
                        "action_id": action_id,
                        "selected_slots": list(slots),
                        "estimated_cost": float(cost),
                        "num_actions_evaluated": len(evaluations),
                        "passed": valid,
                    })

                    if not valid:
                        all_passed = False
                        print(f"  FAIL: {config_key}")

                except Exception as e:
                    all_passed = False
                    results.append({
                        "config": config_key,
                        "error": str(e),
                        "passed": False,
                    })
                    print(f"  ERROR: {config_key}: {e}")

    # Write report
    report = {
        "schema_version": "m4_v1",
        "mode": "smoke_matrix",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "total_configs": len(results),
        "passed_configs": sum(1 for r in results if r.get("passed", False)),
        "failed_configs": sum(1 for r in results if not r.get("passed", False)),
        "all_passed": all_passed,
        "results": results,
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_atomic_json(report, output_dir / "smoke_matrix_report.json")

    print(f"\nSmoke matrix: {report['passed_configs']}/{report['total_configs']} passed")
    return all_passed


def main() -> int:
    """Main entry point."""
    args = parse_args()

    all_passed = True

    if args.test_risk_models:
        if not test_risk_models():
            all_passed = False

    if args.test_enumeration:
        if not test_enumeration():
            all_passed = False

    if args.smoke_matrix:
        output_dir = Path(args.output_dir) if args.output_dir else None
        if not run_smoke_matrix(output_dir):
            all_passed = False

    # Default: run all tests
    if not (args.test_risk_models or args.test_enumeration or args.smoke_matrix):
        print("No tests specified, running all...")
        if not test_risk_models():
            all_passed = False
        if not test_enumeration():
            all_passed = False
        run_smoke_matrix()

    if all_passed:
        print("\n✓ All validation tests passed")
        return 0
    else:
        print("\n✗ Some validation tests failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    # Need numpy for smoke matrix
    import numpy as np
    sys.exit(main())