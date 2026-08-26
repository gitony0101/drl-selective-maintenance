#!/usr/bin/env python3
"""Build Validation Cycle Table for M8 Formal Runner.

Deterministically constructs fd001_val_cycle_table_v1.parquet from five frozen inputs.
This script MUST NOT be executed during the M8-RUNNER-1 implementation phase.
It is created here for audit verification purposes only.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Establish PROJECT_ROOT before any local imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


# Frozen source input paths and hashes (from M8_FORMAL_INPUT_HASHES.json v1.3)
SOURCE_INPUTS = {
    "unit_split": {
        "path": "data/processed/fd001/v2/01_SPLIT/fd001_unit_split_v1.csv",
        "hash": "a86fe8cb1e01d4c7b47fd76d9bcc23351e64b4641386838bee6475bd2863dc9a",
    },
    "train_cycle_table": {
        "path": "data/processed/fd001/v2/02_CYCLE_TABLE/fd001_train_cycle_table_v1.parquet",
        "hash": "d51cddbdd5c4851cf679a0a468674717fc91aba1d0457cdf83b423c2d37e7264",
    },
    "window_index": {
        "path": "data/processed/fd001/v2/05_WINDOW_INDEX/fd001_window_index_v1.parquet",
        "hash": "f2a3a671b944d7b99dba8ea49baa6b21fd63252f97670bab90befa9a02b0f86f",
    },
    "feature_schema": {
        "path": "data/processed/fd001/v2/04_PROTOCOL/fd001_feature_schema_v1.json",
        "hash": "43772bbcaab99e79264fac54780025a54de6e29c75fdccab6dd4ef4d2cbe21da",
    },
    "normalizer": {
        "path": "data/processed/fd001/v2/04_PROTOCOL/fd001_normalizer_v2.json",
        "hash": "08477180719d004dc8f962762735b6344f8198a7719c1c510f9ad7ee15784fde",
    },
}

# Output specification
OUTPUT_PATH = "data/processed/fd001/v2/02_CYCLE_TABLE/fd001_val_cycle_table_v1.parquet"
EXPECTED_ROW_COUNT = 3146
EXPECTED_VALIDATION_UNITS = 15


def compute_sha256(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_source_hashes(data_root: Path) -> Dict[str, str]:
    """Verify all five frozen source inputs match expected hashes.

    Returns:
        Dict mapping input name to computed hash

    Raises:
        RuntimeError: If any hash mismatches or file missing
    """
    computed = {}
    for name, spec in SOURCE_INPUTS.items():
        full_path = data_root / spec["path"]
        if not full_path.exists():
            raise RuntimeError(f"Missing source input: {full_path}")
        actual_hash = compute_sha256(full_path)
        expected_hash = spec["hash"]
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Hash mismatch for {name}: expected {expected_hash}, got {actual_hash}"
            )
        computed[name] = actual_hash
    return computed


def load_validation_unit_ids(data_root: Path) -> List[int]:
    """Load validation unit IDs from unit split manifest.

    Returns:
        Sorted list of 15 validation unit IDs
    """
    split_path = data_root / SOURCE_INPUTS["unit_split"]["path"]
    df = pd.read_csv(split_path)

    # Validate structure
    required_cols = {"unit_id", "split"}
    if not required_cols.issubset(df.columns):
        raise RuntimeError(f"Unit split missing required columns: {required_cols - set(df.columns)}")

    # Filter validation split
    val_units = df[df["split"] == "predictor_validation"]["unit_id"].unique()
    val_units = sorted(val_units.tolist())

    if len(val_units) != EXPECTED_VALIDATION_UNITS:
        raise RuntimeError(
            f"Expected {EXPECTED_VALIDATION_UNITS} validation units, got {len(val_units)}"
        )

    return val_units


def load_window_index(data_root: Path, val_unit_ids: List[int]) -> pd.DataFrame:
    """Load and filter window index to predictor_validation split for validation units.

    Returns:
        DataFrame with columns: unit_id, cycle (using target_cycle from window index)
    """
    window_path = data_root / SOURCE_INPUTS["window_index"]["path"]
    df = pd.read_parquet(window_path)

    # Validate required columns - window index uses target_cycle as cycle
    required = {"unit_id", "target_cycle", "split"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"Window index missing required columns: {required - set(df.columns)}")

    # Filter to predictor_validation split and validation units
    df = df[(df["split"] == "predictor_validation") & (df["unit_id"].isin(val_unit_ids))].copy()

    # Rename target_cycle to cycle for join with train_cycle_table
    df = df.rename(columns={"target_cycle": "cycle"})

    # Sort deterministically
    df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)

    return df


def load_train_cycle_table(data_root: Path) -> pd.DataFrame:
    """Load train cycle table containing true RUL values.

    Returns:
        DataFrame with columns: unit_id, cycle, true_rul (using true_rul_raw from source)
    """
    cycle_path = data_root / SOURCE_INPUTS["train_cycle_table"]["path"]
    df = pd.read_parquet(cycle_path)

    # Validate required columns - source uses true_rul_raw
    required = {"unit_id", "cycle", "true_rul_raw"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"Train cycle table missing required columns: {required - set(df.columns)}")

    # Rename true_rul_raw to true_rul for internal use
    df = df.rename(columns={"true_rul_raw": "true_rul"})

    return df


def build_val_cycle_table(data_root: Path) -> pd.DataFrame:
    """Build validation cycle table from frozen inputs.

    Steps:
    1. Load validation unit IDs from split manifest
    2. Load window index, filter to predictor_validation + validation units
    3. Load train cycle table for true RUL values
    4. Join on (unit_id, cycle) to get true_rul for validation windows
    4. Compute true_rul_capped = min(true_rul, 125)
    5. Select and order columns: unit_id, cycle, true_rul, true_rul_capped
    6. Sort by (unit_id, cycle) ascending
    7. Verify row count = 3146

    Returns:
        Validated DataFrame with exact schema
    """
    # Step 1: Get validation units
    val_unit_ids = load_validation_unit_ids(data_root)
    val_unit_set = set(val_unit_ids)

    # Step 2: Get validation windows from window index
    window_df = load_window_index(data_root, val_unit_ids)

    # Step 3: Get true RUL from train cycle table
    cycle_df = load_train_cycle_table(data_root)

    # Step 4: Join on (unit_id, cycle)
    # The window index gives us the validation windows (unit_id, cycle pairs)
    # The train cycle table has true_rul for all (unit_id, cycle) combinations
    merged = window_df.merge(
        cycle_df[["unit_id", "cycle", "true_rul"]],
        on=["unit_id", "cycle"],
        how="left",
        validate="one_to_one",
    )

    # Verify no missing true_rul
    if merged["true_rul"].isna().any():
        missing = merged[merged["true_rul"].isna()][["unit_id", "cycle"]]
        raise RuntimeError(f"Missing true_rul for validation windows:\n{missing}")

    # Step 5: Compute capped RUL
    merged["true_rul_capped"] = np.minimum(merged["true_rul"], 125.0)

    # Step 6: Select and order columns
    result = merged[["unit_id", "cycle", "true_rul", "true_rul_capped"]].copy()

    # Step 7: Sort deterministically
    result = result.sort_values(["unit_id", "cycle"]).reset_index(drop=True)

    # Step 8: Enforce dtypes
    result = result.astype({
        "unit_id": np.int32,
        "cycle": np.int32,
        "true_rul": np.float32,
        "true_rul_capped": np.float32,
    })

    # Step 9: Verify invariants
    assert len(result) == EXPECTED_ROW_COUNT, f"Row count {len(result)} != {EXPECTED_ROW_COUNT}"
    assert result[["unit_id", "cycle"]].duplicated().sum() == 0, "Duplicate keys found"
    assert set(result["unit_id"].unique()) == val_unit_set, "Unit ID mismatch"
    assert result["unit_id"].is_monotonic_increasing, "Not sorted by unit_id"
    for uid, group in result.groupby("unit_id"):
        assert group["cycle"].is_monotonic_increasing, f"Unit {uid} not sorted by cycle"
    assert (result["true_rul_capped"] <= 125).all(), "Capped RUL exceeds 125"
    assert (result["true_rul_capped"] >= 0).all(), "Capped RUL below 0"
    assert (result["true_rul_capped"] == np.minimum(result["true_rul"], 125)).all(), "Capping incorrect"

    return result


def write_output_atomic(df: pd.DataFrame, output_path: Path) -> str:
    """Write parquet atomically and return SHA256 hash.

    Args:
        df: DataFrame to write
        output_path: Target path

    Returns:
        SHA256 hash of written file
    """
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write pattern
    import os
    tmp_path = output_path.with_name(f"{output_path.name}.tmp.{os.getpid()}.{os.urandom(4).hex()}")

    df.to_parquet(tmp_path, index=False)

    # fsync the temp file
    fd = os.open(tmp_path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

    # Atomic replace
    os.replace(tmp_path, output_path)

    # fsync directory
    dir_fd = os.open(output_path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)

    # Compute and return hash
    return compute_sha256(output_path)


def main() -> int:
    """Main entry point.

    Note: This script should only be executed in a separately authorized step,
    not during the M8-RUNNER-1 implementation phase.
    """
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Build Validation Cycle Table for M8 Formal Runner")
        print("Usage: python scripts/build_val_cycle_table.py [data_root]")
        print("  data_root: Root of processed data (default: current directory)")
        print()
        print("This script reads 5 frozen inputs and produces:")
        print(f"  {OUTPUT_PATH}")
        print("with exactly 3,146 rows, 4 columns, deterministic sort order.")
        return 0

    data_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

    print("=" * 60)
    print("M8 Validation Cycle Table Builder")
    print("=" * 60)
    print(f"Data root: {data_root.absolute()}")
    print(f"Output: {OUTPUT_PATH}")
    print()

    try:
        # Verify source hashes
        print("Verifying source input hashes...")
        computed_hashes = verify_source_hashes(data_root)
        for name, h in computed_hashes.items():
            print(f"  {name}: {h[:16]}... (verified)")

        # Build table
        print("\nBuilding validation cycle table...")
        val_table = build_val_cycle_table(data_root)
        print(f"  Rows: {len(val_table)}")
        print(f"  Units: {val_table['unit_id'].nunique()}")
        print(f"  Columns: {list(val_table.columns)}")
        print(f"  Dtypes: {val_table.dtypes.to_dict()}")

        # Write output
        print("\nWriting output...")
        output_path = data_root / OUTPUT_PATH
        output_hash = write_output_atomic(val_table, output_path)
        print(f"  Written: {output_path}")
        print(f"  SHA256: {output_hash}")

        # Final verification
        print("\nFinal verification...")
        verify_df = pd.read_parquet(output_path)
        assert len(verify_df) == EXPECTED_ROW_COUNT
        assert list(verify_df.columns) == ["unit_id", "cycle", "true_rul", "true_rul_capped"]
        verify_hash = compute_sha256(output_path)
        assert verify_hash == output_hash

        print(f"  Verification PASSED")
        print(f"  Output SHA256: {output_hash}")

        # Output JSON for provenance recording
        provenance = {
            "output_path": str(OUTPUT_PATH),
            "output_sha256": output_hash,
            "row_count": EXPECTED_ROW_COUNT,
            "schema": {
                "unit_id": "int32",
                "cycle": "int32",
                "true_rul": "float32",
                "true_rul_capped": "float32",
            },
            "unique_keys": True,
            "sort_order": ["unit_id", "cycle"],
            "source_hashes": computed_hashes,
            "validation_units": sorted(val_table["unit_id"].unique().tolist()),
        }
        print("\nProvenance record (for M8_FORMAL_RELEASE_MANIFEST.json):")
        print(json.dumps(provenance, indent=2))

        return 0

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())