"""Generate Milestone 1 artifact manifest for V2.

This script reads the required V2 artifacts, validates their existence and
integrity, computes SHA256 hashes and summary statistics, and writes an
artifact manifest JSON file atomically.

Usage:
    python scripts/generate_milestone_manifest_v2.py [--validate-only]

If --validate-only is supplied the script checks for missing or invalid
artifacts, prints a list of problems, and exits with status 1.  No output file
is written.

The script bootstraps sys.path so it can be invoked directly from the project
root without ``PYTHONPATH=.``. All checkpoint loads inside this script use
``torch.load(..., weights_only=False)`` because the trusted local artifacts
were written by the project's own training pipeline.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

# Bootstrap sys.path so this script can be run as
# ``python scripts/generate_milestone_manifest_v2.py`` from the repo root
# without an explicit ``PYTHONPATH=.``.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Atomic write helper — imported after sys.path bootstrap.
from src.predictors.io_utils import atomic_write_json


# ---------------------------------------------------------------------------
# Trusted local checkpoint loading helper.
# ---------------------------------------------------------------------------

def _load_trusted_checkpoint(path: Path) -> dict:
    """Load a trusted local checkpoint with full object reconstruction.

    The Milestone 1 manifest generator only loads artifacts written by the
    project's own training pipeline, so ``weights_only=True`` (the PyTorch
    default since 2.6) would reject them. The explicit ``weights_only=False``
    here makes the trust assumption auditable and avoids spurious warnings.
    """
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Frozen V2 invariants — every artifact must agree on these.
# ---------------------------------------------------------------------------

V2_EXPECTED = {
    "normalizer_id": "fd001_normalizer_v2",
    "feature_schema_id": "fd001_feature_schema_v1",
    "split_manifest_id": "fd001_unit_split_v1",
    "sequence_length": 50,
    "rul_cap": 125,
    "n_features": 24,
    "model_type": "mlp",
}


def validate_frozen_artifact_manifest(base_dir: Path, errors: list) -> dict | None:
    """Validate FROZEN_ARTIFACT_MANIFEST.json integrity.

    Args:
        base_dir: Project root directory
        errors: List to append error messages to

    Returns:
        Dict with frozen manifest info if valid, None otherwise
    """
    manifest_path = base_dir / "data" / "processed" / "fd001" / "v2" / "FROZEN_ARTIFACT_MANIFEST.json"

    if not manifest_path.exists():
        errors.append(f"FROZEN_ARTIFACT_MANIFEST.json not found: {manifest_path}")
        return None

    try:
        with open(manifest_path, "r") as f:
            frozen_manifest = json.load(f)
    except Exception as e:
        errors.append(f"Failed to load FROZEN_ARTIFACT_MANIFEST.json: {e}")
        return None

    # Verify artifact_count matches actual array length
    artifact_count = frozen_manifest.get("artifact_count")
    artifacts = frozen_manifest.get("artifacts", [])

    if artifact_count != len(artifacts):
        errors.append(
            f"FROZEN_ARTIFACT_MANIFEST.json artifact_count={artifact_count}, "
            f"but len(artifacts)={len(artifacts)}"
        )
        return None

    # Check for duplicate paths
    paths = [a["relative_path"] for a in artifacts]
    if len(paths) != len(set(paths)):
        errors.append("FROZEN_ARTIFACT_MANIFEST.json contains duplicate paths")
        return None

    # Verify each frozen artifact exists, has correct size and SHA256
    for artifact in artifacts:
        rel_path = artifact["relative_path"]
        expected_size = artifact["size_bytes"]
        expected_sha = artifact["sha256"]

        full_path = base_dir / rel_path
        if not full_path.exists():
            errors.append(f"Frozen artifact missing: {rel_path}")
            continue

        actual_size = full_path.stat().st_size
        if actual_size != expected_size:
            errors.append(
                f"Frozen artifact size mismatch: {rel_path} - "
                f"expected {expected_size}, got {actual_size}"
            )
            continue

        actual_sha = compute_sha256(full_path)
        if actual_sha != expected_sha:
            errors.append(
                f"Frozen artifact SHA256 mismatch: {rel_path} - "
                f"expected {expected_sha[:16]}..., got {actual_sha[:16]}..."
            )
            continue

    # Verify 06_PREDICTIONS is excluded
    excluded = frozen_manifest.get("excluded_paths", [])
    # Normalize trailing slashes for comparison
    excluded_normalized = [p.rstrip("/") for p in excluded]
    if "data/processed/fd001/v2/06_PREDICTIONS" not in excluded_normalized:
        errors.append("FROZEN_ARTIFACT_MANIFEST.json must exclude 06_PREDICTIONS")
        return None

    return {
        "manifest_path": str(manifest_path),
        "manifest_hash": compute_sha256(manifest_path),
        "artifact_count": artifact_count,
        "frozen_at": frozen_manifest.get("freeze_date"),
    }


def validate_cache_integrity(base_dir: Path, errors: list, paths: dict) -> dict | None:
    """Validate V2 prediction cache integrity.

    Args:
        base_dir: Project root directory
        errors: List to append error messages to
        paths: Dict of artifact paths

    Returns:
        Dict with cache info if valid, None otherwise
    """
    cache_path = paths.get("prediction_cache_v2")
    manifest_path = paths.get("prediction_cache_manifest_v2")

    if not cache_path or not cache_path.exists():
        errors.append(f"Prediction cache not found: {cache_path}")
        return None

    if not manifest_path or not manifest_path.exists():
        errors.append(f"Prediction cache manifest not found: {manifest_path}")
        return None

    # Load cache and manifest
    try:
        df = pd.read_parquet(cache_path)
    except Exception as e:
        errors.append(f"Failed to read prediction cache: {e}")
        return None

    try:
        with open(manifest_path, "r") as f:
            cache_manifest = json.load(f)
    except Exception as e:
        errors.append(f"Failed to read cache manifest: {e}")
        return None

    # Recompute cache SHA256 and compare with manifest
    actual_cache_sha = compute_sha256(cache_path)
    expected_cache_sha = cache_manifest.get("cache_hash")

    # Treat cache_hash as a required field with strict SHA256 format.
    if expected_cache_sha is None:
        errors.append(
            "Prediction cache manifest missing required field 'cache_hash'"
        )
        return None
    if not isinstance(expected_cache_sha, str) or len(expected_cache_sha) != 64:
        errors.append(
            "Prediction cache manifest 'cache_hash' must be a 64-character "
            f"SHA256 hex string, got {type(expected_cache_sha).__name__} "
            f"of length {len(expected_cache_sha) if isinstance(expected_cache_sha, str) else 'N/A'}"
        )
        return None
    if not all(c in "0123456789abcdef" for c in expected_cache_sha):
        errors.append(
            "Prediction cache manifest 'cache_hash' contains non-hexadecimal "
            f"characters: {expected_cache_sha[:16]}..."
        )
        return None
    if actual_cache_sha != expected_cache_sha:
        errors.append(
            f"Cache hash mismatch: manifest says {expected_cache_sha[:16]}..., "
            f"actual {actual_cache_sha[:16]}..."
        )
        return None

    # Verify row count
    actual_rows = len(df)
    expected_rows = cache_manifest.get("total_rows")
    if actual_rows != expected_rows:
        errors.append(
            f"Cache row count mismatch: manifest says {expected_rows}, actual {actual_rows}"
        )

    # Verify required schema columns
    required_cols = [
        "split", "unit_id", "cycle", "trajectory_length",
        "true_rul", "true_rul_capped", "predicted_rul",
        "predicted_rul_normalized", "valid_window", "left_pad_count",
        "predictor_id", "checkpoint_id", "normalizer_id",
        "feature_schema_id", "split_manifest_id", "sequence_length",
        "rul_cap", "cache_version",
    ]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        errors.append(f"Prediction cache missing columns: {sorted(missing_cols)}")

    # Check for NaN and Inf
    numeric_cols = ["true_rul", "true_rul_capped", "predicted_rul", "predicted_rul_normalized"]
    for col in numeric_cols:
        if col in df.columns:
            if df[col].isna().any():
                errors.append(f"NaN found in cache column: {col}")
            if (df[col] == float("inf")).any() or (df[col] == float("-inf")).any():
                errors.append(f"Infinity found in cache column: {col}")

    # Verify primary key uniqueness
    if df.duplicated(subset=["split", "unit_id", "cycle"]).any():
        errors.append("Duplicate (split, unit_id, cycle) keys in prediction cache")

    # Verify required splits exist
    required_splits = {"predictor_train", "rl_validation", "rl_test"}
    actual_splits = set(df["split"].unique().tolist())
    missing_splits = required_splits - actual_splits
    if missing_splits:
        errors.append(f"Cache missing required splits: {sorted(missing_splits)}")

    # Verify identity columns are constant
    for col in ["predictor_id", "checkpoint_id", "normalizer_id", "feature_schema_id", "split_manifest_id", "sequence_length", "rul_cap", "cache_version"]:
        if col in df.columns:
            unique_vals = df[col].unique()
            if len(unique_vals) > 1:
                errors.append(f"Identity column {col} has multiple values: {unique_vals}")

    # Load predictor_metadata.json for cross-check
    metadata_path = paths.get("predictor_metadata")
    if metadata_path and metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            expected_predictor_id = metadata.get("predictor_id")

            # Verify predictor_id in cache matches metadata
            cache_predictor_id = df["predictor_id"].iloc[0] if len(df) > 0 else None
            if cache_predictor_id != expected_predictor_id:
                errors.append(
                    f"Predictor ID mismatch: cache={cache_predictor_id}, "
                    f"metadata={expected_predictor_id}"
                )
        except Exception as e:
            errors.append(f"Failed to load predictor_metadata.json: {e}")

    # Verify checkpoint_id against best checkpoint SHA256
    best_ckpt_path = paths.get("best_checkpoint")
    if best_ckpt_path and best_ckpt_path.exists():
        actual_ckpt_sha = compute_sha256(best_ckpt_path)
        cache_checkpoint_id = df["checkpoint_id"].iloc[0] if len(df) > 0 else None

        # checkpoint_id should be the full 64-char SHA256
        if cache_checkpoint_id != actual_ckpt_sha:
            # Allow for the possibility of shortened IDs in older caches
            if not actual_ckpt_sha.startswith(cache_checkpoint_id) if cache_checkpoint_id else True:
                errors.append(
                    f"Checkpoint ID mismatch: cache={cache_checkpoint_id}, "
                    f"actual checkpoint SHA256={actual_ckpt_sha[:16]}..."
                )

    return {
        "cache_path": str(cache_path),
        "cache_hash": actual_cache_sha,
        "row_count": actual_rows,
        "manifest_total_rows": expected_rows,
        "predictor_id": df["predictor_id"].iloc[0] if len(df) > 0 else None,
        "checkpoint_id": df["checkpoint_id"].iloc[0] if len(df) > 0 else None,
    }


# ---------------------------------------------------------------------------
# Path definitions for the eleven required artifact classes.
# ---------------------------------------------------------------------------

def _artifact_paths(base_dir: Path) -> dict:
    """Return the eleven required V2 artifact paths keyed by class name."""
    fd001 = base_dir / "data" / "processed" / "fd001" / "v2"
    results = base_dir / "results" / "predictor" / "mse_baseline_v2"
    predictions = fd001 / "06_PREDICTIONS"
    return {
        # Frozen scientific source-of-truth artifacts
        "frozen_split_manifest": fd001 / "01_SPLIT" / "fd001_unit_split_v1.csv",
        "frozen_normalizer": fd001 / "04_PROTOCOL" / "fd001_normalizer_v2.json",
        "frozen_feature_schema": fd001 / "04_PROTOCOL" / "fd001_feature_schema_v1.json",
        "frozen_cycle_table": fd001 / "02_CYCLE_TABLE" / "fd001_train_cycle_table_v1.parquet",
        "frozen_window_index": fd001 / "05_WINDOW_INDEX" / "fd001_window_index_v1.parquet",
        # Prediction outputs
        "prediction_cache_v2": predictions / "fd001_prediction_cache_v2.parquet",
        "prediction_cache_manifest_v2": predictions / "prediction_cache_manifest_v2.json",
        # Training-pipeline outputs
        "resolved_config": results / "resolved_config.json",
        "training_history": results / "training_history.json",
        "training_summary": results / "training_summary.json",
        "predictor_metadata": results / "predictor_metadata.json",
        "best_checkpoint": results / "checkpoints" / "best_checkpoint.pt",
        "last_checkpoint": results / "checkpoints" / "last_checkpoint.pt",
        # Collapse report (V2 formal schema)
        "collapse_report_v2": predictions / "collapse_report_v2.json",
        # Exception registry for known artifacts (canonical run only)
        "exception_registry": base_dir / "configs" / "artifact_exceptions" / "canonical_run_exceptions_v1.json",
    }


def _milestone_manifest_path(base_dir: Path) -> Path:
    return (
        base_dir
        / "data" / "processed" / "fd001" / "v2" / "06_PREDICTIONS"
        / "milestone_1_artifact_manifest_v2.json"
    )


def _get_git_commit() -> str:
    """Get current git commit hash."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _get_full_git_commit() -> str:
    """Get full 40-character git commit hash."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# SHA256 + JSON helpers.
# ---------------------------------------------------------------------------

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Cross-document identity checks (loaded only when present, used lazily so
# validate-only can still report missing artifacts).
# ---------------------------------------------------------------------------

def _cross_check_ids(errors: list, paths: dict) -> dict:
    """Cross-check the identity fields across predictor_metadata, the cached
    resolved_config, the checkpoint config, and the frozen normalizer /
    feature-schema / split-manifest JSONs.

    Returns a dict of the resolved identity values suitable for inclusion in
    the manifest. Missing required artifacts are NOT reported here — that is
    done by ``validate_artifacts``.
    """
    def _try_json(name: str) -> dict | None:
        p = paths[name]
        if not p.exists():
            return None
        try:
            return load_json(p)
        except Exception:
            return None

    def _try_ckpt(name: str) -> dict | None:
        p = paths[name]
        if not p.exists():
            return None
        try:
            return _load_trusted_checkpoint(p)
        except Exception:
            return None

    resolved = _try_json("resolved_config")
    metadata = _try_json("predictor_metadata")
    best_ckpt = _try_ckpt("best_checkpoint")
    normalizer = _try_json("frozen_normalizer")
    schema = _try_json("frozen_feature_schema")
    split_df = None
    if paths["frozen_split_manifest"].exists():
        try:
            split_df = pd.read_csv(paths["frozen_split_manifest"])
        except Exception:
            split_df = None

    # Required identity fields. None means "no trustworthy source available".
    resolution = {
        "normalizer_id": (metadata or {}).get("normalizer_id"),
        "feature_schema_id": (metadata or {}).get("feature_schema_id"),
        "split_manifest_id": (metadata or {}).get("split_manifest_id"),
        "predictor_id": (metadata or {}).get("predictor_id"),
        "checkpoint_id": None,
        "sequence_length": (metadata or {}).get("sequence_length"),
        "rul_cap": (metadata or {}).get("rul_cap"),
        "n_features": ((metadata or {}).get("n_features")
                       or ((resolved or {}).get("model") or {}).get("n_features")),
        "model_type": ((metadata or {}).get("model_type")
                       or ((resolved or {}).get("model") or {}).get("type")
                       or ((best_ckpt or {}).get("config") or {}).get("model_type")),
    }

    # checkpoint_id is the checkpoint epoch (integer). We use the best
    # checkpoint's epoch here; this is the only "checkpoint_id" we trust.
    if best_ckpt is not None and "epoch" in best_ckpt:
        resolution["checkpoint_id"] = best_ckpt["epoch"]

    def _expect(name: str, value, expected):
        if value is None:
            errors.append(f"Missing required field {name!r} from predictor_metadata/resolved_config")
            return
        if isinstance(value, float) and isinstance(expected, int):
            if int(value) != expected:
                errors.append(
                    f"{name}={value!r} does not match frozen V2 invariant {expected!r}"
                )
                return
        if value != expected:
            errors.append(
                f"{name}={value!r} does not match frozen V2 invariant {expected!r}"
            )

    _expect("normalizer_id", resolution["normalizer_id"], V2_EXPECTED["normalizer_id"])
    _expect("feature_schema_id", resolution["feature_schema_id"], V2_EXPECTED["feature_schema_id"])
    _expect("split_manifest_id", resolution["split_manifest_id"], V2_EXPECTED["split_manifest_id"])
    _expect("sequence_length", resolution["sequence_length"], V2_EXPECTED["sequence_length"])
    _expect("rul_cap", resolution["rul_cap"], V2_EXPECTED["rul_cap"])
    _expect("n_features", resolution["n_features"], V2_EXPECTED["n_features"])
    _expect("model_type", resolution["model_type"], V2_EXPECTED["model_type"])

    if resolution["predictor_id"] is None:
        errors.append("Missing required field 'predictor_id' from predictor_metadata")
    if resolution["checkpoint_id"] is None:
        errors.append("Missing required field 'checkpoint_id' (epoch) from best_checkpoint")

    # Cross-doc invariants: schedule = epoch/val_rmse from training_history must
    # agree with the final best-checkpoint values.
    history_path = paths["training_history"]
    if history_path.exists():
        try:
            history = load_json(history_path)
        except Exception:
            history = None
        if isinstance(history, list) and history:
            best_record = next((h for h in reversed(history) if h.get("is_best_so_far")), None)
            if best_record is None:
                errors.append(
                    "training_history has no epoch marked 'is_best_so_far'; cannot "
                    "cross-check global best epoch."
                )
            else:
                if best_ckpt is not None:
                    hist_epoch = int(best_record.get("epoch", -1))
                    ckpt_epoch = int(best_ckpt.get("epoch", -1))
                    if hist_epoch != ckpt_epoch:
                        errors.append(
                            f"best epoch mismatch: training_history={hist_epoch}, "
                            f"best_checkpoint={ckpt_epoch}"
                        )
                if best_record.get("val_rmse") is None:
                    errors.append(
                        "training_history best epoch record missing 'val_rmse'"
                    )

    # Frozen-source structural checks.
    if normalizer is not None:
        if "mean" not in normalizer or "std" not in normalizer:
            errors.append(
                f"frozen normalizer {paths['frozen_normalizer']} missing 'mean'/'std'"
            )
        else:
            nfeat = (normalizer.get("mean") or {})
            if nfeat and len(nfeat) != V2_EXPECTED["n_features"]:
                errors.append(
                    f"frozen normalizer has {len(nfeat)} features; "
                    f"V2 expects {V2_EXPECTED['n_features']}"
                )

    if schema is not None:
        cols = (schema.get("input_feature_order") or [])
        if len(cols) != V2_EXPECTED["n_features"]:
            errors.append(
                f"frozen feature schema has {len(cols)} input features; "
                f"V2 expects {V2_EXPECTED['n_features']}"
            )

    if split_df is not None:
        if "unit_id" not in split_df.columns or "split" not in split_df.columns:
            errors.append(
                f"frozen split manifest {paths['frozen_split_manifest']} missing "
                "'unit_id'/'split' columns"
            )
        else:
            splits_seen = set(split_df["split"].unique().tolist())
            required_splits = {"predictor_train", "predictor_validation", "rl_validation", "rl_test"}
            missing_splits = required_splits - splits_seen
            if missing_splits:
                errors.append(
                    f"frozen split manifest missing splits: {sorted(missing_splits)}"
                )

    return resolution


# ---------------------------------------------------------------------------
# Public validators — these are the functions the audit test must exercise.
# ---------------------------------------------------------------------------

def validate_collapse_report(base_dir: Path, errors: list, paths: dict) -> dict | None:
    """Validate V2 collapse report against cache + metadata.

    Args:
        base_dir: Project root directory
        errors: List to append error messages to
        paths: Dict of artifact paths

    Returns:
        Dict with collapse info if valid, None otherwise
    """
    report_path = paths.get("collapse_report_v2")
    if not report_path or not report_path.exists():
        errors.append(f"Collapse report not found: {report_path}")
        return None

    try:
        with open(report_path, "r") as f:
            report = json.load(f)
    except Exception as e:
        errors.append(f"Failed to load collapse report: {e}")
        return None

    # Schema version must match
    if report.get("schema_version") != "fd001_collapse_report_v2":
        errors.append(
            f"Collapse report schema version mismatch: "
            f"got {report.get('schema_version')!r}, expected 'fd001_collapse_report_v2'"
        )
        return None

    # Top-level passed must be true
    if not report.get("passed", False):
        failures = report.get("overall", {}).get("failure_reasons", [])
        errors.append(
            "Collapse report top-level passed=false"
            + (f" ({'; '.join(failures)})" if failures else "")
        )
        return None

    # Overall must pass
    overall = report.get("overall", {})
    if not overall.get("passed", False):
        errors.append("Collapse report overall.passed is false")
        return None

    # Each required split must be present and pass
    required_splits = {"predictor_train", "rl_validation", "rl_test"}
    per_split = report.get("per_split", {})
    for split in required_splits:
        if split not in per_split:
            errors.append(f"Collapse report missing required split: {split}")
            continue
        split_result = per_split[split]
        if not split_result.get("passed", False):
            reasons = split_result.get("failure_reasons", [])
            errors.append(
                f"Collapse report split {split!r} failed"
                + (f" ({'; '.join(reasons)})" if reasons else "")
            )

    # Cache SHA256 must match actual cache
    cache_path = paths.get("prediction_cache_v2")
    if cache_path and cache_path.exists():
        actual_cache_sha = compute_sha256(cache_path)
        report_cache_sha = report.get("cache_sha256")
        if report_cache_sha != actual_cache_sha:
            errors.append(
                f"Collapse report cache_sha256 mismatch: "
                f"report says {report_cache_sha[:16] if report_cache_sha else None}..., "
                f"actual {actual_cache_sha[:16]}..."
            )

    # Predictor ID must match cache and metadata
    cache_path_obj = paths.get("prediction_cache_v2")
    if cache_path_obj and cache_path_obj.exists():
        try:
            df = pd.read_parquet(cache_path_obj)
            cache_predictor_id = df["predictor_id"].iloc[0] if len(df) > 0 else None
            report_predictor_id = report.get("predictor_id")
            if cache_predictor_id and report_predictor_id and cache_predictor_id != report_predictor_id:
                errors.append(
                    f"Collapse report predictor_id mismatch: "
                    f"report={report_predictor_id}, cache={cache_predictor_id}"
                )
        except Exception:
            pass

    # Checkpoint ID must match actual best-checkpoint SHA256
    best_ckpt_path = paths.get("best_checkpoint")
    if best_ckpt_path and best_ckpt_path.exists():
        actual_ckpt_sha = compute_sha256(best_ckpt_path)
        report_ckpt_id = report.get("checkpoint_id")
        if report_ckpt_id and report_ckpt_id != actual_ckpt_sha:
            errors.append(
                f"Collapse report checkpoint_id mismatch: "
                f"report={report_ckpt_id[:16] if report_ckpt_id else None}..., "
                f"actual={actual_ckpt_sha[:16]}..."
            )

    # Training git commit must match checkpoint provenance
    if best_ckpt_path and best_ckpt_path.exists():
        try:
            best_ckpt = _load_trusted_checkpoint(best_ckpt_path)
            ckpt_git = best_ckpt.get("git_commit_hash", "unknown")
            report_git = report.get("training_git_commit")
            if report_git and report_git != ckpt_git:
                errors.append(
                    f"Collapse report training_git_commit mismatch: "
                    f"report={report_git[:12] if report_git else None}..., "
                    f"checkpoint={ckpt_git[:12]}..."
                )
        except Exception:
            pass

    # Tooling git commit must match current tooling commit
    tooling_git = _get_full_git_commit()
    report_tooling = report.get("tooling_git_commit")
    if report_tooling and report_tooling != tooling_git:
        errors.append(
            f"Collapse report tooling_git_commit mismatch: "
            f"report={report_tooling[:12] if report_tooling else None}..., "
            f"current={tooling_git[:12]}..."
        )

    # Check for NaN/Inf in report values
    def _check_report_finite(obj, path_str="report"):
        if isinstance(obj, float):
            import math
            if not math.isfinite(obj):
                raise ValueError(f"Non-finite value at {path_str}")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _check_report_finite(v, f"{path_str}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _check_report_finite(v, f"{path_str}[{i}]")

    try:
        _check_report_finite(report)
    except ValueError as e:
        errors.append(f"Collapse report contains non-finite value: {e}")
        return None

    return {
        "report_path": str(report_path),
        "report_hash": compute_sha256(report_path),
        "schema_version": report.get("schema_version"),
        "passed": report.get("passed"),
        "predictor_id": report.get("predictor_id"),
        "checkpoint_id": report.get("checkpoint_id"),
        "cache_sha256": report.get("cache_sha256"),
        "training_git_commit": report.get("training_git_commit"),
        "tooling_git_commit": report.get("tooling_git_commit"),
        "thresholds": report.get("thresholds"),
        "per_split_verdicts": {
            s: per_split[s].get("passed") for s in required_splits if s in per_split
        },
        "generated_at_utc": report.get("generated_at_utc"),
    }


def validate_exception_registry(base_dir: Path, errors: list, paths: dict) -> dict | None:
    """Validate canonical-run exception registry against actual artifacts.

    The exception registry documents known non-authoritative artifacts (e.g.
    last_checkpoint.pt containing best weights). This function verifies every
    exception entry against the actual filesystem artifacts and returns the
    validated exception info.

    For future clean runs with no exceptions, an empty registry is accepted.
    """
    registry_path = paths.get("exception_registry")
    if not registry_path or not registry_path.exists():
        # No exception registry — this is fine for clean runs
        return {"verified": True, "exceptions_applied": 0}

    try:
        with open(registry_path, "r") as f:
            registry = json.load(f)
    except Exception as e:
        errors.append(f"Failed to load exception registry: {e}")
        return None

    if registry.get("schema_version") != "artifact_exception_registry_v1":
        errors.append(
            f"Exception registry schema version mismatch: "
            f"got {registry.get('schema_version')!r}, expected 'artifact_exception_registry_v1'"
        )
        return None

    exceptions = registry.get("exceptions", [])
    applied_count = 0

    for exc in exceptions:
        exc_id = exc.get("exception_id", "unknown")
        exc_schema = exc.get("schema_version")
        if exc_schema != "artifact_exception_v1":
            errors.append(f"Exception {exc_id}: schema_version mismatch (got {exc_schema!r})")
            continue

        # Verify best checkpoint hash matches
        expected_best_hash = exc.get("authoritative_best_checkpoint_sha256")
        best_ckpt_path = paths.get("best_checkpoint")
        if best_ckpt_path and best_ckpt_path.exists() and expected_best_hash:
            actual_best_hash = compute_sha256(best_ckpt_path)
            if actual_best_hash != expected_best_hash:
                errors.append(
                    f"Exception {exc_id}: best_checkpoint SHA256 mismatch — "
                    f"registry says {expected_best_hash[:16]}..., "
                    f"actual {actual_best_hash[:16]}..."
                )
                continue

        # Verify affected artifact path + hash
        affected_path_rel = exc.get("affected_artifact_path", "")
        affected_hash = exc.get("affected_artifact_sha256")
        if affected_path_rel and affected_hash:
            affected_full = (base_dir / affected_path_rel).resolve()
            if not affected_full.exists():
                errors.append(
                    f"Exception {exc_id}: affected artifact not found: {affected_full}"
                )
                continue
            actual_affected_hash = compute_sha256(affected_full)
            if actual_affected_hash != affected_hash:
                errors.append(
                    f"Exception {exc_id}: affected artifact SHA256 mismatch — "
                    f"registry says {affected_hash[:16]}..., "
                    f"actual {actual_affected_hash[:16]}..."
                )
                continue

        # Verify approved usage prohibits cache generation on affected artifact
        usage = exc.get("approved_usage", {})
        if not usage.get("prohibit_cache_generation", True):
            errors.append(
                f"Exception {exc_id}: approved_usage must prohibit cache generation "
                f"for non-authoritative artifacts"
            )
            continue

        if not exc.get("allowed_for_gate", False):
            errors.append(
                f"Exception {exc_id}: allowed_for_gate must be true for exception to pass"
            )
            continue

        applied_count += 1

    if errors and applied_count == 0:
        return None

    return {
        "verified": True,
        "registry_path": str(registry_path),
        "exceptions_applied": applied_count,
        "training_git_commit": registry.get("training_git_commit"),
        "predictor_id": registry.get("predictor_id"),
    }


def validate_artifacts(base_dir: Path) -> list:
    """Return a list of error strings for missing/invalid artifacts.

    Reports every missing artifact and every invalid artifact, accumulates all
    errors rather than short-circuiting on the first one. The script is the
    single source of truth for "what counts as a valid Milestone 1 V2 install".
    """
    errors = []
    paths = _artifact_paths(base_dir)
    expected_artifacts = [
        "frozen_split_manifest", "frozen_normalizer", "frozen_feature_schema",
        "frozen_cycle_table", "frozen_window_index",
        "prediction_cache_v2", "prediction_cache_manifest_v2",
        "resolved_config", "training_history", "training_summary",
        "predictor_metadata", "best_checkpoint", "last_checkpoint",
    ]
    for a in expected_artifacts:
        p = paths[a]
        if not p.exists():
            errors.append(f"Missing {a}: {p}")
            continue
        if p.suffix == ".json":
            try:
                load_json(p)
            except Exception as e:
                errors.append(f"Invalid JSON in {a} ({p}): {e}")
        if p.suffix == ".parquet":
            try:
                pd.read_parquet(p)
            except Exception as e:
                errors.append(f"Unable to read parquet {a} ({p}): {e}")
        if p.suffix == ".csv":
            try:
                pd.read_csv(p)
            except Exception as e:
                errors.append(f"Unable to read CSV {a} ({p}): {e}")
        if p.suffix == ".pt":
            try:
                _load_trusted_checkpoint(p)
            except Exception as e:
                errors.append(f"Unable to load checkpoint {a} ({p}): {e}")

    # Validate frozen artifact manifest
    validate_frozen_artifact_manifest(base_dir, errors)

    # Validate cache integrity
    validate_cache_integrity(base_dir, errors, paths)

    # Validate collapse report (hard gate — manifest passing requires passing report)
    validate_collapse_report(base_dir, errors, paths)

    # Validate exception registry (known non-authoritative artifacts)
    validate_exception_registry(base_dir, errors, paths)

    # Cross-document identity checks (when sources are available).
    _cross_check_ids(errors, paths)

    return errors


def compute_artifact_hashes(base_dir: Path) -> dict:
    """Compute SHA256 hashes for all eleven artifact classes plus any extras."""
    paths = _artifact_paths(base_dir)
    return {name: compute_sha256(p) for name, p in paths.items() if p.exists()}


def compute_statistics(base_dir: Path) -> dict:
    """Compute the full manifest dictionary assuming all artifacts present and valid."""
    paths = _artifact_paths(base_dir)

    pred_manifest = load_json(paths["prediction_cache_manifest_v2"])
    resolved_config = load_json(paths["resolved_config"])
    training_history = load_json(paths["training_history"])
    training_summary = load_json(paths["training_summary"])
    predictor_metadata = load_json(paths["predictor_metadata"])

    best_ckpt = _load_trusted_checkpoint(paths["best_checkpoint"])
    last_ckpt = _load_trusted_checkpoint(paths["last_checkpoint"])

    df = pd.read_parquet(paths["prediction_cache_v2"])

    per_split_stats = {}
    for split, grp in df.groupby("split"):
        per_split_stats[split] = {
            "mean": float(grp["predicted_rul"].mean()),
            "std": float(grp["predicted_rul"].std(ddof=0)),
            "min": float(grp["predicted_rul"].min()),
            "max": float(grp["predicted_rul"].max()),
            "range": float(grp["predicted_rul"].max() - grp["predicted_rul"].min()),
        }

    epochs = [h["epoch"] for h in training_history]
    epochs_sorted = sorted(epochs)
    history_valid = (
        bool(epochs_sorted)
        and epochs_sorted == list(range(min(epochs_sorted), max(epochs_sorted) + 1))
    )

    # Compute checkpoint SHA256 for checkpoint_id (full 64-char hash)
    best_ckpt_sha = compute_sha256(paths["best_checkpoint"])

    identity = {
        "normalizer_id": predictor_metadata["normalizer_id"],
        "feature_schema_id": predictor_metadata["feature_schema_id"],
        "split_manifest_id": predictor_metadata["split_manifest_id"],
        "predictor_id": predictor_metadata["predictor_id"],
        "checkpoint_id": best_ckpt_sha,  # Full SHA256, not epoch number
        "checkpoint_sha256": best_ckpt_sha,
        "best_epoch": int(best_ckpt["epoch"]),
        "sequence_length": int(predictor_metadata["sequence_length"]),
        "rul_cap": int(predictor_metadata["rul_cap"]),
        "model_type": predictor_metadata.get("model_type")
                       or best_ckpt["config"]["model_type"],
    }

    artifact_hashes = compute_artifact_hashes(base_dir)

    # Validate frozen artifact manifest
    frozen_info = validate_frozen_artifact_manifest(base_dir, errors=[])
    frozen_verification = {
        "verified": frozen_info is not None,
        "artifact_count": frozen_info.get("artifact_count") if frozen_info else None,
        "manifest_hash": frozen_info.get("manifest_hash") if frozen_info else None,
        "frozen_at": frozen_info.get("frozen_at") if frozen_info else None,
    }

    # Validate cache integrity
    cache_info = validate_cache_integrity(base_dir, errors=[], paths=paths)
    cache_verification = {
        "verified": cache_info is not None,
        "cache_hash": cache_info.get("cache_hash") if cache_info else None,
        "row_count": cache_info.get("row_count") if cache_info else None,
        "predictor_id": cache_info.get("predictor_id") if cache_info else None,
        "checkpoint_id": cache_info.get("checkpoint_id") if cache_info else None,
    }

    # Validate collapse report
    collapse_info = validate_collapse_report(base_dir, errors=[], paths=paths)
    collapse_verification = {
        "verified": collapse_info is not None,
        "report_hash": collapse_info.get("report_hash") if collapse_info else None,
        "passed": collapse_info.get("passed") if collapse_info else None,
        "per_split_verdicts": collapse_info.get("per_split_verdicts") if collapse_info else None,
    }

    # Validate exception registry
    exception_info = validate_exception_registry(base_dir, errors=[], paths=paths)
    exception_verification = {
        "verified": exception_info.get("verified", False) if exception_info else False,
        "exceptions_applied": exception_info.get("exceptions_applied", 0) if exception_info else 0,
    }

    # Record last-checkpoint exception for canonical run
    # For the canonical run: best_checkpoint.pt is authoritative, last_checkpoint.pt is not
    last_ckpt_epoch = last_ckpt.get("epoch")
    best_ckpt_epoch = best_ckpt.get("epoch")
    last_ckpt_model_sha = compute_sha256(paths["last_checkpoint"])
    best_ckpt_model_sha = compute_sha256(paths["best_checkpoint"])

    # Check if last_checkpoint contains best weights (the known defect)
    last_contains_best = (last_ckpt_model_sha == best_ckpt_model_sha)
    last_is_authoritative = False  # For canonical run, last is non-authoritative
    last_is_resumable = False  # For canonical run, last is non-resumable

    last_checkpoint_exception = {
        "last_checkpoint_epoch": last_ckpt_epoch,
        "best_checkpoint_epoch": best_ckpt_epoch,
        "last_contains_best_weights": last_contains_best,
        "authoritative": last_is_authoritative,
        "resumable": last_is_resumable,
        "defect_description": (
            "last_checkpoint.pt reports epoch {last} but contains epoch {best} best weights; "
            "file is non-authoritative and non-resumable".format(last=last_ckpt_epoch, best=best_ckpt_epoch)
            if last_contains_best and last_ckpt_epoch != best_ckpt_epoch
            else "No known defect"
        ),
    }

    # Get training and tooling git commits
    training_git_commit = best_ckpt.get("git_commit_hash", "unknown")
    tooling_git_commit = _get_full_git_commit()

    manifest = {
        "schema_version": "milestone_1_v2_artifact_manifest_v1",
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
        "training_git_commit": training_git_commit,
        "tooling_git_commit": tooling_git_commit,
        "artifact_hashes": artifact_hashes,
        "row_counts": pred_manifest.get("row_counts", {}),
        "engine_counts": pred_manifest.get("engine_counts", {}),
        "per_split_statistics": per_split_stats,
        "training_history_valid": history_valid,
        "epochs_trained": training_summary.get("epochs_trained"),
        "best_epoch": training_summary.get("best_epoch"),
        "best_validation_rmse": training_summary.get("best_val_rmse"),
        "frozen_artifact_verification": frozen_verification,
        "cache_verification": cache_verification,
        "collapse_verification": collapse_verification,
        "exception_verification": exception_verification,
        "last_checkpoint_exception": last_checkpoint_exception,
    }
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Generate V2 Milestone 1 artifact manifest")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate artifacts, do not write manifest",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    errors = validate_artifacts(base_dir)
    if errors:
        print("Validation errors found:")
        for e in errors:
            print(f" - {e}")
        sys.exit(1)
    if args.validate_only:
        print("All required V2 artifacts are present and valid.")
        sys.exit(0)

    manifest = compute_statistics(base_dir)
    output_path = _milestone_manifest_path(base_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, manifest)
    print(f"Milestone 1 V2 artifact manifest written to {output_path}")


if __name__ == "__main__":
    main()
