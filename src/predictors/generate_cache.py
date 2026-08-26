"""Generate point prediction cache for RL environment.

Generates predictions for all valid (split, unit_id, cycle) combinations
in RL training, rl_validation, and rl_test splits. This is the V2 version –
outputs are always written to the V2 filenames and never touch V1 artifacts.

The generator requires explicit read-only inputs and verifies checkpoint
identity against predictor metadata before generating any cache.
"""

import argparse
import json
import hashlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predictors.dataset import FD001SequenceDataset
from src.predictors.model import build_predictor
# Atomic write utilities
from src.predictors.io_utils import atomic_write_json, atomic_parquet_write


def get_device() -> str:
    """Get available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_checkpoint(checkpoint_path: Path, device: str = "cpu") -> tuple[nn.Module, Dict[str, Any]]:
    """Load a model checkpoint.

    Returns:
        (model, checkpoint_dict)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_predictor(
        model_type=config["model_type"],
        n_features=config["n_features"],
        sequence_length=config["sequence_length"],
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_predictor_identity(
    checkpoint_path: Path,
    predictor_metadata_path: Path,
    training_summary_path: Path,
    resolved_config_path: Path,
) -> Dict[str, Any]:
    """Validate predictor identity across all canonical artifacts.

    Args:
        checkpoint_path: Path to best_checkpoint.pt
        predictor_metadata_path: Path to predictor_metadata.json
        training_summary_path: Path to training_summary.json
        resolved_config_path: Path to resolved_config.json

    Returns:
        Dict with validated predictor_id, checkpoint_id, and identity fields

    Raises:
        ValueError: If any identity check fails
    """
    # Refuse last_checkpoint.pt - only best_checkpoint.pt is authoritative
    if checkpoint_path.name == "last_checkpoint.pt":
        raise ValueError(
            "Refusing to use last_checkpoint.pt - it is non-authoritative and non-resumable. "
            "Use best_checkpoint.pt only for cache generation."
        )

    # Load predictor metadata (source of truth for predictor_id)
    if not predictor_metadata_path.exists():
        raise ValueError(f"Predictor metadata not found: {predictor_metadata_path}")
    with open(predictor_metadata_path, "r") as f:
        metadata = json.load(f)

    predictor_id = metadata.get("predictor_id")
    if not predictor_id:
        raise ValueError(f"predictor_id not found in {predictor_metadata_path}")

    # Load training summary
    if not training_summary_path.exists():
        raise ValueError(f"Training summary not found: {training_summary_path}")
    with open(training_summary_path, "r") as f:
        summary = json.load(f)

    # Load resolved config
    if not resolved_config_path.exists():
        raise ValueError(f"Resolved config not found: {resolved_config_path}")
    with open(resolved_config_path, "r") as f:
        resolved_config = json.load(f)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Verify checkpoint Git commit matches metadata Git commit
    ckpt_git = checkpoint.get("git_commit_hash", "unknown")
    meta_git = metadata.get("git_commit_hash", "unknown")
    if ckpt_git != meta_git:
        raise ValueError(
            f"Git commit mismatch: checkpoint={ckpt_git}, metadata={meta_git}"
        )

    # Verify summary best_epoch equals checkpoint epoch
    ckpt_epoch = checkpoint.get("epoch")
    summary_best_epoch = summary.get("best_epoch")
    if ckpt_epoch != summary_best_epoch:
        raise ValueError(
            f"Best epoch mismatch: checkpoint epoch={ckpt_epoch}, summary best_epoch={summary_best_epoch}"
        )

    # Verify config identity across checkpoint, metadata, and resolved config
    ckpt_config = checkpoint.get("config", {})
    meta_config_identity = {
        "seed": metadata.get("seed"),
        "sequence_length": metadata.get("sequence_length"),
        "rul_cap": metadata.get("rul_cap"),
        "model_type": metadata.get("model_type"),
        "hidden_dim": metadata.get("hidden_dim"),
        "n_layers": metadata.get("n_layers"),
        "dropout": metadata.get("dropout"),
        "batch_size": metadata.get("batch_size"),
        "learning_rate": metadata.get("learning_rate"),
        "weight_decay": metadata.get("weight_decay"),
        "feature_schema_id": metadata.get("feature_schema_id"),
        "normalizer_id": metadata.get("normalizer_id"),
    }
    resolved_config_identity = {
        "seed": resolved_config.get("seed"),
        "sequence_length": resolved_config.get("sequence_length"),
        "rul_cap": resolved_config.get("rul_cap"),
        "model_type": resolved_config.get("model", {}).get("type"),
        "hidden_dim": resolved_config.get("model", {}).get("hidden_dim"),
        "n_layers": resolved_config.get("model", {}).get("n_layers"),
        "dropout": resolved_config.get("model", {}).get("dropout"),
        "batch_size": resolved_config.get("training", {}).get("batch_size"),
        "learning_rate": resolved_config.get("training", {}).get("learning_rate"),
        "weight_decay": resolved_config.get("training", {}).get("weight_decay"),
        "feature_schema_id": ckpt_config.get("feature_schema_id"),
        "normalizer_id": ckpt_config.get("normalizer_id"),
    }

    # Compare key identity fields
    for key in ["seed", "sequence_length", "rul_cap"]:
        if ckpt_config.get(key) != meta_config_identity.get(key):
            raise ValueError(f"Config mismatch for {key}: checkpoint={ckpt_config.get(key)}, metadata={meta_config_identity.get(key)}")
        if ckpt_config.get(key) != resolved_config_identity.get(key):
            raise ValueError(f"Config mismatch for {key}: checkpoint={ckpt_config.get(key)}, resolved_config={resolved_config_identity.get(key)}")

    # Compute full checkpoint SHA256 for checkpoint_id
    checkpoint_sha256 = compute_file_hash(checkpoint_path)

    return {
        "predictor_id": predictor_id,
        "checkpoint_id": checkpoint_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "best_epoch": ckpt_epoch,
        "git_commit_hash": ckpt_git,
        "predictor_metadata_path": str(predictor_metadata_path),
        "training_summary_path": str(training_summary_path),
        "resolved_config_path": str(resolved_config_path),
        "checkpoint_path": str(checkpoint_path),
    }


# ---------------------------------------------------------------------------
# V2 cache safety guards
# ---------------------------------------------------------------------------

# Canonical V2 invariants. The cache generator refuses checkpoints or paths
# that violate any of these so a stale V1 artifact can never be silently
# regenerated as a V2 cache. These are safety/utility checks only; they do
# not change predictor semantics, the cost model, or observation features.
_V2_SEQUENCE_LENGTH = 50
_V2_RUL_CAP = 125
_V2_N_FEATURES = 24
_V2_NORMALIZER_ID = "fd001_normalizer_v2"
_V2_FEATURE_SCHEMA_ID = "fd001_feature_schema_v1"

_V2_CACHE_NAME = "fd001_prediction_cache_v2.parquet"
_V2_MANIFEST_NAME = "prediction_cache_manifest_v2.json"
_V1_CACHE_NAME = "fd001_prediction_cache_v1.parquet"
_V1_MANIFEST_NAME = "prediction_cache_manifest_v1.json"


def _assert_checkpoint_safe(checkpoint_path: Path) -> None:
    """Refuse checkpoints located under a results/invalidated/ directory.

    V1 artifacts invalidated after the normalization fix live under
    ``results/invalidated/``. Regenerating a V2 cache from one of them would
    silently resurrect the defective model, so the cache generator must refuse.
    """
    try:
        resolved = checkpoint_path.resolve()
    except Exception:
        resolved = checkpoint_path
    parts = resolved.parts
    if "results" in parts and "invalidated" in parts:
        idx = parts.index("results")
        if idx + 1 < len(parts) and parts[idx + 1] == "invalidated":
            raise ValueError(
                f"Refusing to use checkpoint under invalidated directory: {checkpoint_path}. "
                "Invalidated V1 checkpoints must not be used to generate V2 caches."
            )
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")


def _validate_checkpoint_config(checkpoint_path: Path) -> None:
    """Validate the checkpoint's embedded config carries V2 invariants.

    A V1 checkpoint lacks the V2 normalizer/schema identifiers and must be
    rejected so the cache produced from it is genuinely V2.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    required = {
        "sequence_length": _V2_SEQUENCE_LENGTH,
        "rul_cap": _V2_RUL_CAP,
        "n_features": _V2_N_FEATURES,
        "normalizer_id": _V2_NORMALIZER_ID,
        "feature_schema_id": _V2_FEATURE_SCHEMA_ID,
    }
    for key, expected in required.items():
        actual = cfg.get(key)
        if actual is None:
            raise ValueError(
                f"V1 or otherwise V2-incompatible checkpoint: missing '{key}' in "
                f"checkpoint config at {checkpoint_path}."
            )
        if actual != expected:
            raise ValueError(
                f"Checkpoint config mismatch for '{key}': expected {expected!r}, "
                f"got {actual!r} (checkpoint: {checkpoint_path})."
            )


def _assert_writable_output(output_path: Path, overwrite_v2: bool = False) -> None:
    """Guard the V2 output path.

    - The output file name must be the V2 cache name; a V1 path is never
      writable here, regardless of ``overwrite_v2``.
    - If a V2 file already exists at the destination, it may only be replaced
      when ``overwrite_v2`` is explicitly True.
    """
    name = output_path.name
    if name == _V1_CACHE_NAME or name == _V1_MANIFEST_NAME:
        raise ValueError(
            f"Refusing to write a V1 path ({output_path}); the cache generator "
            "only emits V2 artifacts."
        )
    if name == _V2_CACHE_NAME and output_path.exists() and not overwrite_v2:
        raise ValueError(
            f"Existing V2 cache at {output_path}; pass --overwrite-v2 to replace it."
        )


def generate_predictions_for_split(
    split: str,
    data_dir: Path,
    model: nn.Module,
    device: str,
    sequence_length: int,
    rul_cap: int,
) -> pd.DataFrame:
    """Generate predictions for a single split and return a DataFrame."""
    print(f"  Generating predictions for {split}...")
    dataset = FD001SequenceDataset(
        split=split,
        data_dir=data_dir,
        sequence_length=sequence_length,
        rul_cap=rul_cap,
    )
    if len(dataset) == 0:
        print(f"    No data for {split}, skipping")
        return pd.DataFrame()

    records = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            x = sample["features"].unsqueeze(0).to(device)
            y_pred = model(x).squeeze(0).cpu().item()
            true_rul_capped = sample["rul_capped"].item()
            true_rul_raw = sample["rul_raw"].item()
            predicted_rul_normalized = y_pred / rul_cap
            # Resolve trajectory length from the split's cycle table
            traj_len = dataset.split_cycle_df[
                dataset.split_cycle_df["unit_id"] == sample["unit_id"].item()
            ]["max_cycle"].values[0]
            records.append({
                "split": split,
                "unit_id": sample["unit_id"].item(),
                "cycle": sample["cycle"].item(),
                "trajectory_length": int(traj_len),
                "true_rul": true_rul_raw,
                "true_rul_capped": true_rul_capped,
                "predicted_rul": y_pred,
                "predicted_rul_normalized": predicted_rul_normalized,
                "valid_window": 1,
                "left_pad_count": sample["left_pad_count"].item(),
            })
    return pd.DataFrame(records)


def generate_prediction_cache(
    data_dir: Path,
    output_dir: Path,
    checkpoint_path: Path,
    predictor_metadata_path: Path,
    training_summary_path: Path,
    resolved_config_path: Path,
    seed: int = 6521,
    device: Optional[str] = None,
    overwrite_v2: bool = False,
) -> Dict[str, Any]:
    """Generate the full V2 prediction cache and its manifest.

    Args:
        data_dir: Path to FD001 V2 processed directory
        output_dir: Output directory for V2 prediction cache
        checkpoint_path: Path to best_checkpoint.pt (not last_checkpoint.pt)
        predictor_metadata_path: Path to predictor_metadata.json
        training_summary_path: Path to training_summary.json
        resolved_config_path: Path to resolved_config.json
        seed: Random seed for reproducibility
        device: Device to use (default: auto-detect)
        overwrite_v2: Allow replacing existing V2 cache

    Returns:
        Dict with cache metadata and identity fields

    Raises:
        ValueError: If identity validation fails
    """
    if device is None:
        device = get_device()

    print(f"Using device: {device}")

    # Validate predictor identity before generating any cache
    print("Validating predictor identity...")
    identity = validate_predictor_identity(
        checkpoint_path=checkpoint_path,
        predictor_metadata_path=predictor_metadata_path,
        training_summary_path=training_summary_path,
        resolved_config_path=resolved_config_path,
    )
    print(f"  Predictor ID: {identity['predictor_id']}")
    print(f"  Checkpoint ID (SHA256): {identity['checkpoint_sha256'][:16]}...")
    print(f"  Best epoch: {identity['best_epoch']}")
    print(f"  Git commit: {identity['git_commit_hash'][:12]}...")

    # Load checkpoint after validation
    print(f"Loading checkpoint: {checkpoint_path}")
    _assert_checkpoint_safe(checkpoint_path)
    _validate_checkpoint_config(checkpoint_path)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint["config"]
    sequence_length = config["sequence_length"]
    rul_cap = config["rul_cap"]

    # Use validated predictor_id from metadata (not synthesized from checkpoint hash)
    predictor_id = identity["predictor_id"]
    checkpoint_sha256 = identity["checkpoint_sha256"]

    # Load ancillary artifacts for manifest
    normalizer_path = data_dir / "04_PROTOCOL" / "fd001_normalizer_v2.json"
    with open(normalizer_path, "r") as f:
        normalizer = json.load(f)
    split_path = data_dir / "01_SPLIT" / "fd001_unit_split_v1.csv"
    split_df = pd.read_csv(split_path)
    schema_path = data_dir / "04_PROTOCOL" / "fd001_feature_schema_v1.json"
    with open(schema_path, "r") as f:
        feature_schema = json.load(f)

    # Reproducibility seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Splits we need to generate (RL training uses predictor_train as its trajectory source)
    rl_splits = ["predictor_train", "rl_validation", "rl_test"]

    all_dfs = []
    row_counts: Dict[str, int] = {}
    engine_counts: Dict[str, int] = {}
    cycle_ranges: Dict[str, Dict[str, int]] = {}

    for split in rl_splits:
        df = generate_predictions_for_split(
            split=split,
            data_dir=data_dir,
            model=model,
            device=device,
            sequence_length=sequence_length,
            rul_cap=rul_cap,
        )
        if not df.empty:
            all_dfs.append(df)
            row_counts[split] = len(df)
            engine_counts[split] = df["unit_id"].nunique()
            cycle_ranges[split] = {"min": int(df["cycle"].min()), "max": int(df["cycle"].max())}
            print(f"    {split}: {len(df)} rows, {engine_counts[split]} engines")

    if not all_dfs:
        raise ValueError("No predictions generated for any V2 split")

    predictions_df = pd.concat(all_dfs, ignore_index=True)

    # Add per-row metadata columns required by PredictionStore's V2 schema
    # so the cache is directly loadable without downstream patching.
    # Use validated predictor_id and full checkpoint SHA256 from identity
    n_rows = len(predictions_df)
    predictions_df["predictor_id"] = [predictor_id] * n_rows
    predictions_df["checkpoint_id"] = [checkpoint_sha256] * n_rows  # Full 64-char SHA256
    predictions_df["normalizer_id"] = [config["normalizer_id"]] * n_rows
    predictions_df["feature_schema_id"] = [config["feature_schema_id"]] * n_rows
    predictions_df["split_manifest_id"] = ["fd001_unit_split_v1"] * n_rows
    predictions_df["sequence_length"] = [config["sequence_length"]] * n_rows
    predictions_df["rul_cap"] = [config["rul_cap"]] * n_rows
    predictions_df["cache_version"] = ["v2"] * n_rows

    # Validation checks
    print("\nValidating predictions...")
    numeric_cols = ["true_rul", "true_rul_capped", "predicted_rul", "predicted_rul_normalized"]
    for col in numeric_cols:
        if predictions_df[col].isna().any():
            raise ValueError(f"NaN found in {col}")
        if np.isinf(predictions_df[col]).any():
            raise ValueError(f"Inf found in {col}")
    if predictions_df.duplicated(subset=["split", "unit_id", "cycle"]).any():
        raise ValueError("Duplicate (split, unit_id, cycle) keys found")
    if not np.isfinite(predictions_df["predicted_rul"]).all():
        raise ValueError("Non-finite predicted_rul values found")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "fd001_prediction_cache_v2.parquet"
    manifest_path = output_dir / "prediction_cache_manifest_v2.json"
    # Guard: refuse V1 paths, require permission to overwrite an existing V2 cache.
    _assert_writable_output(cache_path, overwrite_v2=overwrite_v2)
    _assert_writable_output(manifest_path, overwrite_v2=overwrite_v2)
    atomic_parquet_write(predictions_df, cache_path)
    print(f"\nSaved prediction cache: {cache_path}")
    cache_hash = compute_file_hash(cache_path)

    # Build manifest with all required fields (no placeholders)
    manifest = {
        "cache_version": "v2",
        "predictor_id": predictor_id,
        "checkpoint_id": checkpoint_sha256,  # Full 64-char SHA256
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_hash": checkpoint_sha256,
        "cache_path": str(cache_path),
        "cache_hash": cache_hash,
        "split_manifest_path": str(split_path),
        "split_manifest_hash": compute_file_hash(split_path),
        "feature_schema": feature_schema["input_feature_order"],
        "feature_schema_hash": feature_schema.get("schema_hash", "unknown"),
        "normalizer_id": "fd001_normalizer_v2",
        "normalizer_hash": compute_file_hash(normalizer_path),
        "sequence_length": sequence_length,
        "rul_cap": rul_cap,
        "row_counts": row_counts,
        "engine_counts": engine_counts,
        "cycle_ranges": cycle_ranges,
        "total_rows": int(predictions_df.shape[0]),
        "creation_timestamp": datetime.utcnow().isoformat(),
        "random_seed": seed,
        "device": device,
        "software_versions": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        # Identity validation fields
        "best_epoch": identity["best_epoch"],
        "git_commit_hash": identity["git_commit_hash"],
        "predictor_metadata_path": identity["predictor_metadata_path"],
        "training_summary_path": identity["training_summary_path"],
        "resolved_config_path": identity["resolved_config_path"],
    }

    atomic_write_json(manifest_path, manifest)
    print(f"Saved manifest: {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FD001 Prediction Cache (V2)")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "fd001" / "v2",
        help="Path to FD001 V2 processed directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "fd001" / "v2" / "06_PREDICTIONS",
        help="Output directory for V2 prediction cache",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        required=True,
        help="Path to best_checkpoint.pt (required; last_checkpoint.pt is rejected)",
    )
    parser.add_argument(
        "--predictor-metadata-path",
        type=Path,
        required=True,
        help="Path to predictor_metadata.json (source of truth for predictor_id)",
    )
    parser.add_argument(
        "--training-summary-path",
        type=Path,
        required=True,
        help="Path to training_summary.json",
    )
    parser.add_argument(
        "--resolved-config-path",
        type=Path,
        required=True,
        help="Path to resolved_config.json",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=6521,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--overwrite-v2",
        action="store_true",
        help="Allow replacing an existing V2 prediction cache at the output path. "
             "This flag can never target V1 paths; a V1 path is always refused.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("FD001 Point Prediction Cache Generation (V2)")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Predictor metadata: {args.predictor_metadata_path}")
    print(f"Training summary: {args.training_summary_path}")
    print(f"Resolved config: {args.resolved_config_path}")
    print(f"Seed: {args.seed}")
    print(f"Overwrite V2: {args.overwrite_v2}\n")

    # Validate required files exist before proceeding
    for path_arg, path_val in [
        ("--checkpoint-path", args.checkpoint_path),
        ("--predictor-metadata-path", args.predictor_metadata_path),
        ("--training-summary-path", args.training_summary_path),
        ("--resolved-config-path", args.resolved_config_path),
    ]:
        if not path_val.exists():
            print(f"Error: {path_arg} not found: {path_val}")
            sys.exit(1)

    manifest = generate_prediction_cache(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        predictor_metadata_path=args.predictor_metadata_path,
        training_summary_path=args.training_summary_path,
        resolved_config_path=args.resolved_config_path,
        seed=args.seed,
        overwrite_v2=args.overwrite_v2,
    )

    print("\n" + "=" * 60)
    print("Cache Generation Complete!")
    print("=" * 60)
    print(f"Total rows: {manifest['total_rows']}")
    print(f"Row counts: {manifest['row_counts']}")
    print(f"Engine counts: {manifest['engine_counts']}")
    print(f"Cache hash: {manifest['cache_hash'][:16]}...")


if __name__ == "__main__":
    main()
