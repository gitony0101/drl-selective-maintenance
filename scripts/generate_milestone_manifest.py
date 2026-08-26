#!/usr/bin/env python3
"""
FD001 Milestone 1 Artifact Manifest Generator

Generates a machine-readable manifest for all Milestone 1 artifacts.
"""

import json
import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "fd001" / "v2"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "fd001" / "v2" / "06_PREDICTIONS"
RESULTS_DIR = PROJECT_ROOT / "results" / "predictor" / "mse_baseline"


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def generate_manifest() -> Dict[str, Any]:
    """Generate complete artifact manifest."""

    # Load prediction cache
    cache_path = OUTPUT_DIR / "fd001_prediction_cache_v1.parquet"
    cache_df = pd.read_parquet(cache_path)

    # Load prediction manifest
    pred_manifest_path = OUTPUT_DIR / "prediction_cache_manifest_v1.json"
    with open(pred_manifest_path, "r") as f:
        pred_manifest = json.load(f)

    # Load checkpoint metadata
    checkpoint_path = RESULTS_DIR / "checkpoints" / "best_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Load training history
    history_path = RESULTS_DIR / "training_history.json"
    with open(history_path, "r") as f:
        history = json.load(f)

    # Compute statistics
    row_counts = {}
    engine_counts = {}
    cycle_ranges = {}

    for split in cache_df["split"].unique():
        split_df = cache_df[cache_df["split"] == split]
        row_counts[split] = len(split_df)
        engine_counts[split] = split_df["unit_id"].nunique()
        cycle_ranges[split] = {
            "min": int(split_df["cycle"].min()),
            "max": int(split_df["cycle"].max()),
        }

    # Build manifest
    manifest = {
        "milestone": "Milestone 1 - Baseline MSE RUL Predictor and Prediction Cache",
        "cache_version": "v1",
        "predictor_id": pred_manifest.get("predictor_id", "unknown"),
        "checkpoint_path": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "checkpoint_hash": compute_file_hash(checkpoint_path),
        "cache_path": str(cache_path.relative_to(PROJECT_ROOT)),
        "cache_hash": compute_file_hash(cache_path),
        "split_manifest_path": "data/processed/fd001/v2/01_SPLIT/fd001_unit_split_v1.csv",
        "split_manifest_hash": compute_file_hash(
            DATA_DIR / "01_SPLIT" / "fd001_unit_split_v1.csv"
        ),
        "feature_schema": pred_manifest.get("feature_schema", []),
        "feature_schema_hash": pred_manifest.get("feature_schema_hash", "unknown"),
        "normalizer_id": "fd001_normalizer_v2",
        "normalizer_hash": compute_file_hash(
            DATA_DIR / "04_PROTOCOL" / "fd001_normalizer_v2.json"
        ),
        "sequence_length": pred_manifest.get("sequence_length", 50),
        "rul_cap": pred_manifest.get("rul_cap", 125),
        "row_counts": row_counts,
        "engine_counts": engine_counts,
        "cycle_ranges": cycle_ranges,
        "total_rows": int(len(cache_df)),
        "training_summary": {
            "best_epoch": int(checkpoint.get("epoch", 0)),
            "best_val_rmse": float(checkpoint.get("val_rmse", 0.0)),
            "final_train_loss": float(history[-1]["train_loss"]) if history else 0.0,
            "epochs_trained": len(history),
            "config": {k: (int(v) if isinstance(v, (int, np.integer)) else float(v) if isinstance(v, (float, np.floating)) else v) for k, v in checkpoint.get("config", {}).items()},
        },
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "software_versions": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "python": platform.python_version(),
        },
        "device_used": pred_manifest.get("device", "unknown"),
        "random_seed": pred_manifest.get("random_seed", 6521),
        "git_commit_hash": get_git_commit(),
    }

    return manifest


def main():
    """Generate and save artifact manifest."""
    print("Generating Milestone 1 Artifact Manifest...")

    manifest = generate_manifest()

    # Save manifest
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_DIR / "milestone_1_artifact_manifest_v1.json"

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest saved: {manifest_path}")
    print()
    print("=" * 60)
    print("MILESTONE 1 ARTIFACT MANIFEST")
    print("=" * 60)
    print(f"Predictor ID: {manifest['predictor_id']}")
    print(f"Total cache rows: {manifest['total_rows']}")
    print(f"Row counts: {manifest['row_counts']}")
    print(f"Engine counts: {manifest['engine_counts']}")
    print(f"Best validation RMSE: {manifest['training_summary']['best_val_rmse']:.4f}")
    print(f"Git commit: {manifest['git_commit_hash'][:12]}...")
    print()
    print("Manifest generation complete.")


if __name__ == "__main__":
    main()