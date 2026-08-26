#!/usr/bin/env python3
"""
Generate Milestone 4 Scientific Validation scenario banks.

Implements the frozen quantile-stratified protocol:
- predicted_rul_quantile_stratified_cache_continuity_v1
- 2 splits × 2 K values × 4 cost regimes = 16 banks
- 20 scenarios per bank (seeds 6601-6620)
- 5 units per scenario, one from each predicted-RUL quantile stratum
- c/c+1 cache-row continuity required
- Finite predicted RUL required
- 5 distinct unit IDs per scenario
- No rl_test, no hidden truth, no policy-dependent construction
- Deterministic selection from frozen seeds
- Stable pair_id for paired analysis across candidates, K, regimes

FAIL-CLOSED: Raises on any data insufficiency.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from envs.scenario_bank import Scenario, ScenarioBank, save_scenario_bank
from envs.costs import COST_REGIMES

# Frozen protocol constants
PROTOCOL_VERSION = "m4_scientific_validation_v1"
SELECTION_BASIS = "predicted_rul_quantile_stratified_cache_continuity_v1"
SPLITS = ["predictor_train", "rl_validation"]
K_VALUES = [1, 2]
COST_REGIMES = list(COST_REGIMES.keys())
SEEDS = list(range(6601, 6621))  # 6601 through 6620 inclusive
SCENARIOS_PER_BANK = 20
UNITS_PER_SCENARIO = 5
RUL_SCALE = 125.0
QUANTILE_BOUNDARIES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


class ScientificValidationBankError(Exception):
    """Raised when scientific validation bank generation fails."""
    def __init__(
        self,
        message: str,
        split: str = None,
        K: int = None,
        regime: str = None,
        missing_condition: str = None,
    ):
        self.split = split
        self.K = K
        self.regime = regime
        self.missing_condition = missing_condition
        full_message = message
        if split:
            full_message += f" | split: {split}"
        if K:
            full_message += f" | K: {K}"
        if regime:
            full_message += f" | regime: {regime}"
        if missing_condition:
            full_message += f" | missing: {missing_condition}"
        super().__init__(full_message)


def load_prediction_cache(cache_path: Path) -> pd.DataFrame:
    """Load and validate prediction cache."""
    if not cache_path.exists():
        raise ScientificValidationBankError(
            f"Prediction cache not found: {cache_path}",
            missing_condition=f"cache file: {cache_path}"
        )

    try:
        cache = pd.read_parquet(cache_path)
    except Exception as e:
        raise ScientificValidationBankError(
            f"Could not read prediction cache: {e}",
            missing_condition="cache read failure"
        )

    required_columns = ['split', 'unit_id', 'cycle', 'predicted_rul_normalized']
    missing = [c for c in required_columns if c not in cache.columns]
    if missing:
        raise ScientificValidationBankError(
            f"Cache missing required columns: {missing}",
            missing_condition=f"missing columns: {missing}"
        )

    # Compute predicted_rul_cycles if not present
    if 'predicted_rul_cycles' not in cache.columns:
        cache['predicted_rul_cycles'] = cache['predicted_rul_normalized'] * RUL_SCALE

    return cache


def get_cache_sha256(cache_path: Path) -> str:
    """Compute SHA256 of prediction cache file."""
    sha256 = hashlib.sha256()
    with open(cache_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def build_continuity_map(cache: pd.DataFrame, split: str) -> Dict[int, List[Tuple[int, float]]]:
    """
    Build map of unit_id -> list of (cycle, pred_rul_cycles) with c/c+1 continuity.

    Only includes rows where:
    - Row exists at cycle c for split/unit_id
    - Row exists at cycle c+1 for same split/unit_id
    - predicted_rul_cycles is finite
    - split is predictor_train or rl_validation (not rl_test)
    """
    if split == 'rl_test':
        raise ScientificValidationBankError(
            "rl_test split is forbidden",
            split=split,
            missing_condition="rl_test is forbidden"
        )

    split_cache = cache[cache['split'] == split].copy()

    if len(split_cache) == 0:
        raise ScientificValidationBankError(
            f"No cache rows for split '{split}'",
            split=split,
            missing_condition=f"split '{split}' has no data"
        )

    # Build set of existing (split, unit_id, cycle) keys
    existing_keys = set(zip(
        split_cache['split'],
        split_cache['unit_id'],
        split_cache['cycle']
    ))

    # Group by unit_id and find cycles with continuity
    result = {}
    for unit_id in split_cache['unit_id'].unique():
        unit_data = split_cache[split_cache['unit_id'] == unit_id]
        cycles_data = []
        for _, row in unit_data.iterrows():
            c = int(row['cycle'])
            # Check c+1 continuity
            if (split, unit_id, c + 1) not in existing_keys:
                continue

            pred_rul = float(row['predicted_rul_cycles'])
            if not np.isfinite(pred_rul) or pred_rul < 0:
                continue

            cycles_data.append((c, pred_rul))

        if cycles_data:
            # Sort by predicted RUL ascending
            cycles_data.sort(key=lambda x: x[1])
            result[int(unit_id)] = cycles_data

    return result


def compute_quantile_strata(
    continuity_map: Dict[int, List[Tuple[int, float]]]
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Stratify units into 5 quantile strata based on their minimum predicted RUL.

    For each unit, use the minimum predicted RUL across all its valid cycles
    to assign it to a stratum. Then within each stratum, keep all valid cycles.

    Returns: stratum_idx -> list of (unit_id, cycle, pred_rul) tuples
    """
    # First, find each unit's minimum predicted RUL
    unit_min_rul = {}
    for unit_id, cycles in continuity_map.items():
        min_rul = min(c[1] for c in cycles)
        unit_min_rul[unit_id] = min_rul

    if len(unit_min_rul) < UNITS_PER_SCENARIO:
        raise ScientificValidationBankError(
            f"Insufficient units with continuity: {len(unit_min_rul)} < {UNITS_PER_SCENARIO}",
            missing_condition=f"need {UNITS_PER_SCENARIO} units, have {len(unit_min_rul)}"
        )

    # Compute quantile boundaries from unit minimum RULs
    min_rul_values = np.array(list(unit_min_rul.values()))
    boundaries = np.quantile(min_rul_values, QUANTILE_BOUNDARIES)

    # Ensure boundaries are strictly increasing (handle ties)
    for i in range(1, len(boundaries)):
        if boundaries[i] <= boundaries[i-1]:
            boundaries[i] = boundaries[i-1] + 1e-9

    # Assign each unit to a stratum
    stratum_units = {i: [] for i in range(5)}
    for unit_id, min_rul in unit_min_rul.items():
        # Find stratum: rightmost boundary <= min_rul
        stratum = np.searchsorted(boundaries, min_rul, side='right') - 1
        stratum = max(0, min(4, stratum))  # Clamp to [0, 4]
        # Add all valid cycles for this unit to its stratum
        for cycle, pred_rul in continuity_map[unit_id]:
            stratum_units[stratum].append((unit_id, cycle, pred_rul))

    # Verify each stratum has at least one unit
    for i in range(5):
        if len(stratum_units[i]) == 0:
            raise ScientificValidationBankError(
                f"Stratum {i} has zero units - quantile stratification failed",
                missing_condition=f"stratum {i} empty"
            )

    return stratum_units, boundaries.tolist()


def select_scenario_from_strata(
    stratum_units: Dict[int, List[Tuple[int, float]]],
    seed: int,
    scenario_idx: int,
) -> List[Tuple[int, int, float]]:
    """
    Deterministically select one unit from each stratum to form a 5-unit scenario.

    Uses the seed to create a deterministic but varied selection.
    Returns list of (unit_id, cycle, pred_rul) tuples.
    """
    np.random.seed(seed)

    selected = []
    used_unit_ids = set()

    for stratum in range(5):
        candidates = [(uid, c, rul) for (uid, c, rul) in stratum_units[stratum]
                      if uid not in used_unit_ids]

        if not candidates:
            raise ScientificValidationBankError(
                f"Stratum {stratum} exhausted - cannot form 5 distinct units",
                missing_condition=f"stratum {stratum} no available units"
            )

        # Deterministic selection: use seed + scenario_idx + stratum
        sel_idx = (seed + scenario_idx * 5 + stratum) % len(candidates)
        selected.append(candidates[sel_idx])
        used_unit_ids.add(candidates[sel_idx][0])

    # Verify 5 distinct units
    assert len(used_unit_ids) == 5
    assert len(selected) == 5

    # Sort by unit_id for deterministic scenario identity
    selected.sort(key=lambda x: x[0])
    return selected


def build_bank_for_config(
    cache: pd.DataFrame,
    cache_sha256: str,
    split: str,
    K: int,
    regime: str,
    quantile_boundaries: List[float],
) -> Tuple[ScenarioBank, Dict[str, Any]]:
    """Build a single scenario bank for a (split, K, regime) configuration.

    Returns (bank, metadata) where metadata contains protocol-required fields.
    """

    # Build continuity map for this split
    continuity_map = build_continuity_map(cache, split)

    # Compute quantile strata
    stratum_units, boundaries = compute_quantile_strata(continuity_map)

    bank_id = f"{split}_K{K}_{regime}_bank"
    scenarios = []
    pair_ids = []

    for i, seed in enumerate(SEEDS):
        # Select one unit from each stratum
        selected = select_scenario_from_strata(stratum_units, seed, i)

        unit_ids = tuple(int(s[0]) for s in selected)
        cycles = tuple(int(s[1]) for s in selected)

        # Create stable pair_id based on unit_ids and cycles (not seed)
        # This allows pairing across seeds, K, regimes
        pair_key = tuple(sorted(zip(unit_ids, cycles)))
        pair_id = hashlib.sha256(
            json.dumps(pair_key, sort_keys=True).encode()
        ).hexdigest()[:16]
        pair_ids.append(pair_id)

        scenario = Scenario(
            scenario_id=f"{split}_K{K}_{regime}_{i:03d}",
            split=split,
            initial_unit_ids=unit_ids,
            initial_cycles=cycles,
            replacement_seed=seed,
            environment_seed=seed,
            episode_horizon=100,
            maintenance_capacity=K,
            cost_regime_id=regime,
        )
        scenarios.append(scenario)

    # Create bank
    bank = ScenarioBank(
        bank_id=bank_id,
        split=split,
        scenarios=tuple(scenarios),
    )

    # Build metadata (stored in manifest, not in bank object)
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "selection_basis": SELECTION_BASIS,
        "quantile_boundaries": quantile_boundaries,
        "prediction_cache_sha256": cache_sha256,
        "generation_version": PROTOCOL_VERSION,
        "no_rl_test_declaration": True,
        "ordered_seeds": SEEDS,
        "pair_ids": pair_ids,
    }

    return bank, metadata


def generate_scientific_validation_banks(output_dir: Path) -> Dict[str, Any]:
    """Generate all 16 scientific validation banks."""

    # Find prediction cache
    repo_root = Path(__file__).parent.parent
    cache_path = repo_root / "data" / "processed" / "fd001" / "v2" / "06_PREDICTIONS" / "fd001_prediction_cache_v2.parquet"

    print(f"Loading prediction cache from: {cache_path}")
    cache = load_prediction_cache(cache_path)
    cache_sha256 = get_cache_sha256(cache_path)
    print(f"Prediction cache SHA256: {cache_sha256}")

    # Build continuity maps and quantile strata for each split
    # We compute quantile boundaries from the full data to ensure consistency
    all_stratum_units = {}
    all_boundaries = {}

    for split in SPLITS:
        print(f"\nProcessing split: {split}")
        continuity_map = build_continuity_map(cache, split)
        print(f"  Units with c/c+1 continuity: {len(continuity_map)}")

        stratum_units, boundaries = compute_quantile_strata(continuity_map)
        print(f"  Quantile boundaries: {[f'{b:.2f}' for b in boundaries]}")
        for s in range(5):
            print(f"  Stratum {s}: {len(stratum_units[s])} candidate slots")

        all_stratum_units[split] = stratum_units
        all_boundaries[split] = boundaries

    # Use boundaries from rl_validation for consistency (or average)
    # Protocol says: "Compute five deterministic predicted-RUL quantile strata"
    # We'll use boundaries from the combined data for consistency
    # For strict determinism, we should compute boundaries once from all valid units
    # Let's compute from combined continuity maps
    combined_units = {}
    for split in SPLITS:
        for uid, cycles in build_continuity_map(cache, split).items():
            if uid not in combined_units:
                combined_units[uid] = []
            combined_units[uid].extend(cycles)

    combined_stratum, global_boundaries = compute_quantile_strata(combined_units)
    print(f"\nGlobal quantile boundaries: {[f'{b:.2f}' for b in global_boundaries]}")

    # Now rebuild strata for each split using global boundaries
    # Actually, the protocol says compute strata from the data for each split
    # Let's use split-specific strata as computed above

    output_dir.mkdir(parents=True, exist_ok=True)

    banks_created = []
    bank_metadata = {}

    for split in SPLITS:
        for K in K_VALUES:
            for regime in COST_REGIMES:
                print(f"\nGenerating bank: {split}_K{K}_{regime}")

                # Use split-specific strata
                continuity_map = build_continuity_map(cache, split)
                stratum_units, boundaries = compute_quantile_strata(continuity_map)

                bank, metadata = build_bank_for_config(
                    cache=cache,
                    cache_sha256=cache_sha256,
                    split=split,
                    K=K,
                    regime=regime,
                    quantile_boundaries=boundaries,
                )

                filename = f"{split}_K{K}_{regime}.json"
                output_path = output_dir / filename
                save_scenario_bank(bank, output_path)
                print(f"  Created: {output_path} ({len(bank.scenarios)} scenarios)")

                # Store metadata in manifest
                bank_metadata[filename] = metadata

                banks_created.append({
                    'file': filename,
                    'bank_id': bank.bank_id,
                    'split': split,
                    'K': K,
                    'regime': regime,
                    'scenarios': len(bank.scenarios),
                    'quantile_boundaries': boundaries,
                })

    # Generate manifest
    manifest = {
        'protocol_version': PROTOCOL_VERSION,
        'selection_basis': SELECTION_BASIS,
        'prediction_cache_sha256': cache_sha256,
        'generation_version': PROTOCOL_VERSION,
        'no_rl_test_declaration': True,
        'splits': SPLITS,
        'k_values': K_VALUES,
        'cost_regimes': COST_REGIMES,
        'seeds': SEEDS,
        'scenarios_per_bank': SCENARIOS_PER_BANK,
        'units_per_scenario': UNITS_PER_SCENARIO,
        'rul_scale': RUL_SCALE,
        'quantile_boundaries_global': global_boundaries,
        'banks': banks_created,
        'bank_metadata': bank_metadata,
    }

    manifest_path = output_dir / "SCIENTIFIC_VALIDATION_BANK_MANIFEST.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved: {manifest_path}")

    return manifest


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="generate_m4_scientific_validation_banks",
        description="Generate M4 scientific validation scenario banks (frozen protocol)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/scenario_banks/m4_scientific_validation/)",
    )

    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent.parent / "data" / "scenario_banks" / "m4_scientific_validation"

    print(f"Generating M4 scientific validation banks in: {output_dir}")
    print(f"Protocol: {PROTOCOL_VERSION}")
    print(f"Selection basis: {SELECTION_BASIS}")
    print(f"Seeds: {SEEDS}")
    print(f"Scenarios per bank: {SCENARIOS_PER_BANK}")
    print(f"Total banks: {len(SPLITS) * len(K_VALUES) * len(COST_REGIMES)}")

    try:
        manifest = generate_scientific_validation_banks(output_dir)
        print(f"\n✓ Successfully generated {len(manifest['banks'])} banks")
        print(f"  Total scenarios: {sum(b['scenarios'] for b in manifest['banks'])}")
        return 0
    except ScientificValidationBankError as e:
        print(f"\n✗ Bank generation failed: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())