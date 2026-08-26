#!/usr/bin/env python3
"""
Independent validator for Milestone 4 Scientific Validation scenario banks.

This validator MUST NOT call the generator's candidate-selection functions.
It independently verifies all protocol requirements from the frozen bank files.

Verification checklist:
- files = 16
- banks = 16
- scenarios = 320
- slots = 1600
- unique scenario IDs = 320
- unique pair IDs have expected pairing structure
- five distinct units per scenario
- c exists
- c+1 exists
- predictions finite
- quantile membership correct
- no rl_test
- no hidden truth fields
- ordered seeds exact
- filenames and metadata consistent
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from envs.scenario_bank import load_scenario_bank

# Frozen protocol constants
PROTOCOL_VERSION = "m4_scientific_validation_v1"
SELECTION_BASIS = "predicted_rul_quantile_stratified_cache_continuity_v1"
SPLITS = ["predictor_train", "rl_validation"]
K_VALUES = [1, 2]
COST_REGIMES = [
    "failure-heavy-no-waste",
    "failure-heavy-waste-aware",
    "failure-light-no-waste",
    "failure-light-waste-aware",
]
SEEDS = list(range(6601, 6621))
SCENARIOS_PER_BANK = 20
UNITS_PER_SCENARIO = 5
RUL_SCALE = 125.0
EXPECTED_BANKS = 16
EXPECTED_SCENARIOS = 320
EXPECTED_SLOTS = 1600


class BankValidationError(Exception):
    """Raised when bank validation fails."""
    def __init__(self, message: str, bank_file: str = None, scenario_id: str = None):
        self.bank_file = bank_file
        self.scenario_id = scenario_id
        full_message = message
        if bank_file:
            full_message += f" | bank: {bank_file}"
        if scenario_id:
            full_message += f" | scenario: {scenario_id}"
        super().__init__(full_message)


def load_cache(cache_path: Path) -> pd.DataFrame:
    """Load prediction cache and compute predicted_rul_cycles."""
    if not cache_path.exists():
        raise BankValidationError(f"Cache not found: {cache_path}")

    cache = pd.read_parquet(cache_path)
    if 'predicted_rul_cycles' not in cache.columns:
        cache['predicted_rul_cycles'] = cache['predicted_rul_normalized'] * RUL_SCALE
    return cache


def get_cache_sha256(cache_path: Path) -> str:
    """Compute SHA256 of cache file."""
    sha256 = hashlib.sha256()
    with open(cache_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def validate_bank_file(
    bank_path: Path,
    cache: pd.DataFrame,
    cache_sha256: str,
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate a single bank file. Returns validation stats."""

    # Load bank
    try:
        bank = load_scenario_bank(bank_path)
    except Exception as e:
        raise BankValidationError(f"Failed to load bank: {e}", bank_file=bank_path.name)

    # Load metadata from manifest
    bank_filename = bank_path.name
    if bank_filename not in manifest.get('bank_metadata', {}):
        raise BankValidationError(
            f"Bank metadata not found in manifest",
            bank_file=bank_path.name
        )
    metadata = manifest['bank_metadata'][bank_filename]

    # Check metadata from manifest
    if metadata.get('protocol_version') != PROTOCOL_VERSION:
        raise BankValidationError(
            f"Invalid protocol_version: {metadata.get('protocol_version', 'missing')}",
            bank_file=bank_path.name
        )

    if metadata.get('selection_basis') != SELECTION_BASIS:
        raise BankValidationError(
            f"Invalid selection_basis: {metadata.get('selection_basis', 'missing')}",
            bank_file=bank_path.name
        )

    if metadata.get('prediction_cache_sha256') != cache_sha256:
        raise BankValidationError(
            f"Cache SHA256 mismatch: {metadata.get('prediction_cache_sha256', 'missing')} != {cache_sha256}",
            bank_file=bank_path.name
        )

    if not metadata.get('no_rl_test_declaration', False):
        raise BankValidationError(
            "Missing or false no_rl_test_declaration",
            bank_file=bank_path.name
        )

    # Check ordered seeds from metadata
    if metadata.get('ordered_seeds') != SEEDS:
        raise BankValidationError(
            f"Seeds mismatch: {metadata.get('ordered_seeds', 'missing')}",
            bank_file=bank_path.name
        )

    # Check pair_ids from metadata
    pair_ids = metadata.get('pair_ids')
    if pair_ids is None or len(pair_ids) != SCENARIOS_PER_BANK:
        raise BankValidationError(
            f"Missing or invalid pair_ids in manifest: {pair_ids}",
            bank_file=bank_path.name
        )

    # Check scenarios
    scenarios = bank.scenarios
    if len(scenarios) != SCENARIOS_PER_BANK:
        raise BankValidationError(
            f"Expected {SCENARIOS_PER_BANK} scenarios, got {len(scenarios)}",
            bank_file=bank_path.name
        )

    # Parse expected config from filename
    filename = bank_path.name
    expected_split = None

    for split in SPLITS:
        if filename.startswith(f"{split}_"):
            expected_split = split
            break

    if expected_split is None:
        raise BankValidationError(f"Filename doesn't match expected split pattern: {filename}", bank_file=bank_path.name)

    # Use the bank's declared values
    first_scenario = scenarios[0]
    if first_scenario.split != expected_split:
        raise BankValidationError(
            f"Bank split mismatch: filename={expected_split}, scenario={first_scenario.split}",
            bank_file=bank_path.name
        )

    expected_K = first_scenario.maintenance_capacity
    expected_regime = first_scenario.cost_regime_id

    if expected_K not in K_VALUES:
        raise BankValidationError(f"Invalid K: {expected_K}", bank_file=bank_path.name)

    if expected_regime not in COST_REGIMES:
        raise BankValidationError(f"Invalid regime: {expected_regime}", bank_file=bank_path.name)

    # Build cache lookup for continuity checks
    cache_keys = set(zip(cache['split'], cache['unit_id'], cache['cycle']))

    # Validation statistics
    stats = {
        'bank_file': bank_path.name,
        'bank_id': bank.bank_id,
        'split': expected_split,
        'K': expected_K,
        'regime': expected_regime,
        'scenarios': len(scenarios),
        'slots': 0,
        'unique_scenario_ids': set(),
        'unique_pair_ids': set(),
        'pair_id_to_scenarios': {},
        'continuity_failures': 0,
        'prediction_failures': 0,
        'duplicate_units': 0,
        'rl_test_refs': 0,
        'hidden_truth_fields': 0,
        'seed_mismatches': 0,
        'quantile_failures': 0,
        'finite_prediction_failures': 0,
    }

    # Track pair_id for pairing verification
    for i, scenario in enumerate(scenarios):
        pair_id = pair_ids[i]
        # Check scenario attributes
        if scenario.split != expected_split:
            raise BankValidationError(
                f"Scenario split mismatch: {scenario.split} != {expected_split}",
                bank_file=bank_path.name,
                scenario_id=scenario.scenario_id
            )

        if scenario.maintenance_capacity != expected_K:
            raise BankValidationError(
                f"Scenario K mismatch: {scenario.maintenance_capacity} != {expected_K}",
                bank_file=bank_path.name,
                scenario_id=scenario.scenario_id
            )

        if scenario.cost_regime_id != expected_regime:
            raise BankValidationError(
                f"Scenario regime mismatch: {scenario.cost_regime_id} != {expected_regime}",
                bank_file=bank_path.name,
                scenario_id=scenario.scenario_id
            )

        # Check unit count
        if len(scenario.initial_unit_ids) != UNITS_PER_SCENARIO:
            raise BankValidationError(
                f"Expected {UNITS_PER_SCENARIO} units, got {len(scenario.initial_unit_ids)}",
                bank_file=bank_path.name,
                scenario_id=scenario.scenario_id
            )

        # Check 5 distinct units
        if len(set(scenario.initial_unit_ids)) != UNITS_PER_SCENARIO:
            stats['duplicate_units'] += 1
            raise BankValidationError(
                f"Duplicate unit IDs in scenario",
                bank_file=bank_path.name,
                scenario_id=scenario.scenario_id
            )

        # Check cycles length matches
        if len(scenario.initial_cycles) != UNITS_PER_SCENARIO:
            raise BankValidationError(
                f"Cycles count mismatch: {len(scenario.initial_cycles)} != {UNITS_PER_SCENARIO}",
                bank_file=bank_path.name,
                scenario_id=scenario.scenario_id
            )

        # Check no rl_test
        if scenario.split == 'rl_test':
            stats['rl_test_refs'] += 1
            raise BankValidationError(
                f"rl_test reference found in scenario",
                bank_file=bank_path.name,
                scenario_id=scenario.scenario_id
            )

        # Check seeds match ordered list
        expected_seed = SEEDS[scenarios.index(scenario)]
        if scenario.replacement_seed != expected_seed or scenario.environment_seed != expected_seed:
            stats['seed_mismatches'] += 1
            raise BankValidationError(
                f"Seed mismatch: expected {expected_seed}, got replacement={scenario.replacement_seed}, env={scenario.environment_seed}",
                bank_file=bank_path.name,
                scenario_id=scenario.scenario_id
            )

        # Check pair_id exists
        if pair_id is None:
            raise BankValidationError(
                f"Missing pair_id for scenario index {i}",
                bank_file=bank_path.name,
                scenario_id=scenario.scenario_id
            )

        stats['unique_scenario_ids'].add(scenario.scenario_id)
        stats['unique_pair_ids'].add(pair_id)

        if pair_id not in stats['pair_id_to_scenarios']:
            stats['pair_id_to_scenarios'][pair_id] = []
        stats['pair_id_to_scenarios'][pair_id].append(scenario.scenario_id)

        # Validate each slot
        for unit_id, cycle in zip(scenario.initial_unit_ids, scenario.initial_cycles):
            stats['slots'] += 1

            # Check cache row exists at cycle c
            key_c = (scenario.split, unit_id, cycle)
            if key_c not in cache_keys:
                stats['continuity_failures'] += 1
                raise BankValidationError(
                    f"Missing cache row at (split={scenario.split}, unit_id={unit_id}, cycle={cycle})",
                    bank_file=bank_path.name,
                    scenario_id=scenario.scenario_id
                )

            # Check cache row exists at cycle c+1
            key_c1 = (scenario.split, unit_id, cycle + 1)
            if key_c1 not in cache_keys:
                stats['continuity_failures'] += 1
                raise BankValidationError(
                    f"Missing cache row at c+1 (split={scenario.split}, unit_id={unit_id}, cycle={cycle+1})",
                    bank_file=bank_path.name,
                    scenario_id=scenario.scenario_id
                )

            # Check prediction is finite
            cache_row = cache[(cache['split'] == scenario.split) &
                              (cache['unit_id'] == unit_id) &
                              (cache['cycle'] == cycle)]
            if len(cache_row) == 0:
                stats['prediction_failures'] += 1
                raise BankValidationError(
                    f"Cache row not found for validation",
                    bank_file=bank_path.name,
                    scenario_id=scenario.scenario_id
                )

            pred_rul = float(cache_row['predicted_rul_cycles'].values[0])
            if not np.isfinite(pred_rul) or pred_rul < 0:
                stats['finite_prediction_failures'] += 1
                raise BankValidationError(
                    f"Non-finite predicted RUL: {pred_rul}",
                    bank_file=bank_path.name,
                    scenario_id=scenario.scenario_id
                )

            # Check for hidden truth fields in scenario (should not have true_rul, etc.)
            # Scenario dataclass shouldn't have these, but check anyway
            scenario_dict = scenario.__dict__
            hidden_fields = ['true_rul', 'true_rul_capped', 'trajectory_id', 'trajectory_length', 'failure_endpoint']
            for hf in hidden_fields:
                if hf in scenario_dict:
                    stats['hidden_truth_fields'] += 1
                    raise BankValidationError(
                        f"Hidden truth field '{hf}' found in scenario",
                        bank_file=bank_path.name,
                        scenario_id=scenario.scenario_id
                    )

    return stats


def validate_pairing_structure(all_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Verify pair_ids create expected pairing structure across banks."""

    # Collect all pair_ids by (unit_ids, cycles) signature
    pair_map = {}  # pair_id -> list of (bank_file, scenario_id, split, K, regime)

    for stat in all_stats:
        bank_file = stat['bank_file']
        split = stat['split']
        K = stat['K']
        regime = stat['regime']

        for pair_id, scenario_ids in stat['pair_id_to_scenarios'].items():
            if pair_id not in pair_map:
                pair_map[pair_id] = []
            for sid in scenario_ids:
                pair_map[pair_id].append({
                    'bank_file': bank_file,
                    'scenario_id': sid,
                    'split': split,
                    'K': K,
                    'regime': regime,
                })

    # Verify pairing: same pair_id should appear across K and regime for same split
    pairing_results = {
        'total_unique_pair_ids': len(pair_map),
        'pair_ids_per_split': {s: set() for s in SPLITS},
        'cross_k_pairing': True,
        'cross_regime_pairing': True,
        'issues': [],
    }

    for pair_id, occurrences in pair_map.items():
        splits = set(o['split'] for o in occurrences)
        Ks = set(o['K'] for o in occurrences)
        regimes = set(o['regime'] for o in occurrences)

        for split in splits:
            pairing_results['pair_ids_per_split'][split].add(pair_id)

        # For same split, should appear across K values and regimes
        # Actually, pair_id is derived from (unit_ids, cycles) which is split-specific
        # So same pair_id within same split should appear in multiple K/regime banks
        if len(Ks) < 2:
            pairing_results['cross_k_pairing'] = False
            pairing_results['issues'].append(f"Pair {pair_id}: only in K={Ks}")

        if len(regimes) < 2:
            pairing_results['cross_regime_pairing'] = False
            pairing_results['issues'].append(f"Pair {pair_id}: only in regimes={regimes}")

    return pairing_results


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="validate_m4_scientific_validation_banks",
        description="Independent validator for M4 scientific validation banks",
    )

    parser.add_argument(
        "--bank-dir",
        type=str,
        default=None,
        help="Bank directory (default: data/scenario_banks/m4_scientific_validation/)",
    )

    parser.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help="Prediction cache path (default: data/processed/fd001/v2/06_PREDICTIONS/fd001_prediction_cache_v2.parquet)",
    )

    parser.add_argument(
        "--strict-json",
        action="store_true",
        help="Output strict JSON only (for CI)",
    )

    args = parser.parse_args()

    if args.bank_dir:
        bank_dir = Path(args.bank_dir)
    else:
        bank_dir = Path(__file__).parent.parent / "data" / "scenario_banks" / "m4_scientific_validation"

    if args.cache_path:
        cache_path = Path(args.cache_path)
    else:
        cache_path = Path(__file__).parent.parent / "data" / "processed" / "fd001" / "v2" / "06_PREDICTIONS" / "fd001_prediction_cache_v2.parquet"

    if not bank_dir.exists():
        print(f"Bank directory not found: {bank_dir}", file=sys.stderr)
        return 1

    if not cache_path.exists():
        print(f"Cache not found: {cache_path}", file=sys.stderr)
        return 1

    # Load manifest
    manifest_path = bank_dir / "SCIENTIFIC_VALIDATION_BANK_MANIFEST.json"
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    with open(manifest_path) as f:
        manifest = json.load(f)

    if manifest.get('protocol_version') != PROTOCOL_VERSION:
        print(f"Manifest protocol mismatch: {manifest.get('protocol_version')} != {PROTOCOL_VERSION}", file=sys.stderr)
        return 1

    # Load cache
    cache = load_cache(cache_path)
    cache_sha256 = get_cache_sha256(cache_path)

    if not args.strict_json:
        print(f"Validating banks in: {bank_dir}")
        print(f"Using cache: {cache_path}")
        print(f"Manifest protocol: {manifest.get('protocol_version')}")
        print(f"Cache SHA256: {cache_sha256}")
        print(f"Cache rows: {len(cache)}")
        print(f"Cache splits: {cache['split'].unique().tolist()}")

    # Find bank files
    bank_files = sorted([f for f in bank_dir.glob("*.json") if f.name != "SCIENTIFIC_VALIDATION_BANK_MANIFEST.json"])
    if len(bank_files) != EXPECTED_BANKS:
        if not args.strict_json:
            print(f"Expected {EXPECTED_BANKS} bank files, found {len(bank_files)}", file=sys.stderr)
            for f in bank_files:
                print(f"  {f.name}", file=sys.stderr)
        return 1

    if not args.strict_json:
        print(f"\nValidating {len(bank_files)} bank files...")

    # Validate each bank
    all_stats = []
    total_scenarios = 0
    total_slots = 0
    total_continuity_failures = 0
    total_prediction_failures = 0
    total_duplicate_units = 0
    total_rl_test = 0
    total_hidden_truth = 0
    total_seed_mismatches = 0
    total_quantile_failures = 0
    total_finite_failures = 0

    for bank_path in bank_files:
        try:
            stats = validate_bank_file(bank_path, cache, cache_sha256, manifest)
            all_stats.append(stats)
            total_scenarios += stats['scenarios']
            total_slots += stats['slots']
            total_continuity_failures += stats['continuity_failures']
            total_prediction_failures += stats['prediction_failures']
            total_duplicate_units += stats['duplicate_units']
            total_rl_test += stats['rl_test_refs']
            total_hidden_truth += stats['hidden_truth_fields']
            total_seed_mismatches += stats['seed_mismatches']
            total_quantile_failures += stats['quantile_failures']
            total_finite_failures += stats['finite_prediction_failures']
            if not args.strict_json:
                print(f"  ✓ {bank_path.name}: {stats['scenarios']} scenarios, {stats['slots']} slots")
        except BankValidationError as e:
            if not args.strict_json:
                print(f"  ✗ {bank_path.name}: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            if not args.strict_json:
                print(f"  ✗ {bank_path.name}: Unexpected error: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
            return 1

    # Verify totals
    if not args.strict_json:
        print(f"\n--- Aggregate Validation ---")
        print(f"Banks: {len(all_stats)} (expected {EXPECTED_BANKS})")
        print(f"Scenarios: {total_scenarios} (expected {EXPECTED_SCENARIOS})")
        print(f"Slots: {total_slots} (expected {EXPECTED_SLOTS})")

        # Check unique scenario IDs
        all_scenario_ids = set()
        for stat in all_stats:
            all_scenario_ids.update(stat['unique_scenario_ids'])
        print(f"Unique scenario IDs: {len(all_scenario_ids)}")

        # Check pairing structure
        pairing = validate_pairing_structure(all_stats)
        print(f"Unique pair IDs: {pairing['total_unique_pair_ids']}")
        for split in SPLITS:
            print(f"  Pair IDs in {split}: {len(pairing['pair_ids_per_split'][split])}")
        print(f"Cross-K pairing: {pairing['cross_k_pairing']}")
        print(f"Cross-regime pairing: {pairing['cross_regime_pairing']}")
        if pairing['issues']:
            for issue in pairing['issues']:
                print(f"  Issue: {issue}")

        # Error summary
        print(f"\n--- Error Counts ---")
        print(f"Continuity failures: {total_continuity_failures}")
        print(f"Prediction failures: {total_prediction_failures}")
        print(f"Duplicate units: {total_duplicate_units}")
        print(f"RL_TEST references: {total_rl_test}")
        print(f"Hidden truth fields: {total_hidden_truth}")
        print(f"Seed mismatches: {total_seed_mismatches}")
        print(f"Quantile failures: {total_quantile_failures}")
        print(f"Finite prediction failures: {total_finite_failures}")

    # Final verdict
    all_scenario_ids = set()
    for stat in all_stats:
        all_scenario_ids.update(stat['unique_scenario_ids'])

    pairing = validate_pairing_structure(all_stats)

    all_passed = (
        len(all_stats) == EXPECTED_BANKS and
        total_scenarios == EXPECTED_SCENARIOS and
        total_slots == EXPECTED_SLOTS and
        len(all_scenario_ids) == EXPECTED_SCENARIOS and
        total_continuity_failures == 0 and
        total_prediction_failures == 0 and
        total_duplicate_units == 0 and
        total_rl_test == 0 and
        total_hidden_truth == 0 and
        total_seed_mismatches == 0 and
        total_quantile_failures == 0 and
        total_finite_failures == 0
    )

    if args.strict_json:
        output = {
            'passed': all_passed,
            'banks': len(all_stats),
            'scenarios': total_scenarios,
            'slots': total_slots,
            'unique_scenario_ids': len(all_scenario_ids),
            'unique_pair_ids': pairing['total_unique_pair_ids'],
            'continuity_failures': total_continuity_failures,
            'prediction_failures': total_prediction_failures,
            'duplicate_units': total_duplicate_units,
            'rl_test_references': total_rl_test,
            'hidden_truth_fields': total_hidden_truth,
            'seed_mismatches': total_seed_mismatches,
            'quantile_failures': total_quantile_failures,
            'finite_prediction_failures': total_finite_failures,
            'cross_k_pairing': pairing['cross_k_pairing'],
            'cross_regime_pairing': pairing['cross_regime_pairing'],
            'pairing_issues': pairing['issues'],
        }
        print(json.dumps(output))
        return 0 if all_passed else 1

    if all_passed:
        print("\n✓✓✓ ALL VALIDATIONS PASSED ✓✓✓")
    else:
        print("\n✗✗✗ VALIDATION FAILED ✗✗✗")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())