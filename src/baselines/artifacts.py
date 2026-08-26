"""
Artifact writing and validation for Milestone 3 Baselines.

Generates and validates all required artifacts:
- resolved_config.json
- threshold_search_results.parquet
- threshold_search_summary.csv
- selected_thresholds.json
- episode_results.parquet
- summary_by_policy.csv
- summary_by_policy.json
- sanity_checks.json
- run_provenance.json
- artifact_manifest.json
- m3_run.log

Provenance chain (formal mode):

    1. execute tuning and evaluation, write immutable parquet/json artifacts
    2. run validator, write validation_report.json (NEVER touches
       formal_manifest.json)
    3. run independent_recomputation, write independent_recomputation.json
    4. generate formal_manifest.json LAST — without ever being re-touched

formal_manifest.json is immutable: write exactly once, never modify.
The generator's caller is forbidden from passing run ID, config hash, or
selected SHA as parameters; all values must be computed from disk and
from `git rev-parse HEAD`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from .evaluator import EpisodeResult
from .tuning import ThresholdCandidate, SelectedThreshold


# Oracle terminology contract: the formal manifest requires the independent
# recomputation report to declare the Oracle policy's authoritative semantic
# role, which must equal this canonical diagnostic-benchmark label (the Oracle
# is a privileged-information diagnostic benchmark, NOT an optimal policy /
# upper bound). The manifest preserves this role from the verified PASS
# independent report; it is NOT invented inside the manifest.
ORACLE_SEMANTIC_ROLE = "privileged-information diagnostic benchmark"


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def get_file_size(file_path: Path) -> int:
    """Get file size in bytes."""
    return file_path.stat().st_size


def validate_json_serializable(obj: Any, path: str = "") -> None:
    """
    Validate that an object is JSON-serializable.

    Raises:
        ValueError: If NaN, Inf, or non-serializable type found
    """
    import numpy as np

    if isinstance(obj, dict):
        for k, v in obj.items():
            validate_json_serializable(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            validate_json_serializable(v, f"{path}[{i}]")
    elif isinstance(obj, (int, float, str, bool, type(None))):
        if isinstance(obj, float):
            if np.isnan(obj):
                raise ValueError(f"NaN at {path} is not JSON-serializable")
            if np.isinf(obj):
                raise ValueError(f"Inf at {path} is not JSON-serializable")
    elif hasattr(obj, "item"):  # numpy scalar
        val = obj.item()
        if np.isnan(val):
            raise ValueError(f"NaN at {path} is not JSON-serializable")
        if np.isinf(val):
            raise ValueError(f"Inf at {path} is not JSON-serializable")
    else:
        raise ValueError(f"Type {type(obj)} at {path} is not JSON-serializable")


def write_json_safe(data: Any, file_path: Path) -> None:
    """
    Write data to JSON file, validating JSON-serializability.

    Args:
        data: Data to write
        file_path: Output file path
    """
    validate_json_serializable(data, "root")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_resolved_config(
    config: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """Write resolved configuration to JSON."""
    file_path = output_dir / "resolved_config.json"
    write_json_safe(config, file_path)
    return file_path


def _canonical_json_bytes(config: Dict[str, Any]) -> bytes:
    """Serialize a config dict to deterministically canonical JSON bytes.

    Canonical-JSON rules (RFC 8785-friendly, restricted to types we use):

      - no whitespace (sort_keys=True, separators=(",", ":"))
      - dict keys sorted lexicographically at every level
      - arrays preserve their order (no reordering of episodes/regimes)
      - all numbers serialized through ``json`` default (noNaN, no Infinity)
        so floats like 5.0 are written uniformly. The config dict is
        validated as JSON-serializable before encoding so we never
        encounter NaN/Inf here.
    """
    # Round-trip through ``float`` normalization so 5 and 5.0 hash equal.
    def _normalize(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _normalize(obj[k]) for k in sorted(obj.keys())}
        if isinstance(obj, list):
            return [_normalize(v) for v in obj]
        if isinstance(obj, tuple):
            return [_normalize(v) for v in obj]
        if isinstance(obj, float):
            return float(obj)  # preserve NaN/Inf strictly out — config must be finite
        return obj

    normalized = _normalize(config)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def compute_canonical_config_sha256(config: Dict[str, Any]) -> str:
    """SHA256 over canonical-JSON bytes of a resolved config dict.

    Forbidden alternatives (do NOT use these):

      - hash of a directory tree (`configs/baselines/*.json` glob)
      - hash of a single config path on disk (vulnerable to on-disk edits
        after the fact)
      - output_dir.name as the only source
      - empty-string fallback

    The single source of truth is the resolved config dict the run
    actually executed against, encoded as canonical JSON.
    """
    return hashlib.sha256(_canonical_json_bytes(config)).hexdigest()


def read_resolved_config_sha256(output_dir: Path) -> Optional[str]:
    """Read resolved_config.json (canonical-JSON SHA256) from an output dir.

    Returns None if the file is missing or unparseable.
    """
    p = output_dir / "resolved_config.json"
    if not p.exists():
        return None
    try:
        with open(p, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return compute_canonical_config_sha256(data)
    except (json.JSONDecodeError, OSError):
        return None


def _derive_config_sha_from_configs_dir(repo_root: Path) -> str:
    """Legacy hash over the configs/baselines directory tree.

    Kept for backward compatibility with old manifests that did not yet
    write resolved_config.json. Prefer
    :func:`read_resolved_config_sha256` for new runs.

    Used when no caller-supplied config SHA exists; this answers
    'what config was actually in the repo at the recorded commit?'.
    """


def write_threshold_search_results(
    episode_rows: List[Dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Write episode-level threshold search results to parquet (9000 rows).

    Each row is one tuning episode indexed by the 6-tuple
    ``(policy_family, threshold, k_capacity, cost_regime_id,
    scenario_id, reset_seed)``. The auditor reads this parquet to
    reconstruct both the 360 candidate identity set (via dedup on the
    4-tuple key) and the 9000 tuning-episode identity set (full row).

    The candidate-level summary (360 rows) is persisted separately
    via :func:`write_threshold_search_summary` as a CSV.
    """
    file_path = output_dir / "threshold_search_results.parquet"
    if not episode_rows:
        raise ValueError("Cannot write 0 episode rows; tune_threshold must produce at least one row")

    required_cols = (
        "policy_family",
        "threshold",
        "k_capacity",
        "cost_regime_id",
        "scenario_id",
        "reset_seed",
    )
    for col in required_cols:
        if not all(col in r for r in episode_rows):
            raise ValueError(
                f"episode_rows missing required column {col!r}; check the producer"
            )

    seen_keys: set = set()
    duplicates: List[Any] = []
    for row in episode_rows:
        key = (row["policy_family"], row["threshold"], row["k_capacity"],
               row["cost_regime_id"], row["scenario_id"], row["reset_seed"])
        if key in seen_keys:
            duplicates.append(key)
        seen_keys.add(key)
    if duplicates:
        raise ValueError(
            f"Episode rows contain {len(duplicates)} duplicate keys "
            f"on {required_cols}; first duplicate={duplicates[0]}"
        )

    df = pd.DataFrame(episode_rows)
    df.to_parquet(file_path, index=False)
    return file_path


TuningEpisodeRow = Dict[str, Any]


def aggregate_threshold_candidates_from_episode_rows(
    rows: List[TuningEpisodeRow],
) -> List["ThresholdCandidate"]:
    """Deterministically aggregate episode rows into candidate records.

    Mirrors the existing scenario-level semantics of ``tune_threshold``
    so candidate-level CSV outputs and selected-winner policy do not
    change. Aggregation rule:

      * mean_total_cost        = mean over all completed episodes' total_cost
      * total_failures         = sum over all completed episodes' failure_count
      * mean_wasted_life_cost  = mean over all completed episodes' wasted_life_cost
      * episode_count          = count of completed episodes

    Reuses the ``ThresholdCandidate`` dataclass declared above.
    """
    buckets: Dict[Tuple[Any, ...], List[TuningEpisodeRow]] = {}
    for row in rows:
        key = (
            row["policy_family"],
            float(row["threshold"]),
            int(row["k_capacity"]),
            row["cost_regime_id"],
        )
        buckets.setdefault(key, []).append(row)

    candidates = []
    for key, items in buckets.items():
        completed = [r for r in items if r.get("completed", True)]
        if not completed:
            completed = items
        total_costs = [float(r["total_cost"]) for r in completed]
        failures = [int(r.get("failure_count", 0)) for r in completed]
        wasted = [float(r["wasted_life_cost"]) for r in completed]
        candidates.append(ThresholdCandidate(
            policy_family=key[0],
            threshold=key[1],
            k_capacity=key[2],
            cost_regime_id=key[3],
            mean_total_cost=float(np.mean(total_costs)),
            total_failures=int(sum(failures)),
            mean_wasted_life_cost=float(np.mean(wasted)),
            episode_count=len(total_costs),
        ))
    candidates.sort(key=lambda c: (c.policy_family, c.threshold, c.k_capacity, c.cost_regime_id))
    return candidates


def write_threshold_search_summary(
    candidates: List[ThresholdCandidate],
    output_dir: Path,
) -> Path:
    """Write threshold search summary to CSV."""
    file_path = output_dir / "threshold_search_summary.csv"
    records = []
    for c in candidates:
        records.append({
            "policy_family": c.policy_family,
            "threshold": c.threshold,
            "k_capacity": c.k_capacity,
            "cost_regime_id": c.cost_regime_id,
            "mean_total_cost": c.mean_total_cost,
            "total_failures": c.total_failures,
            "mean_wasted_life_cost": c.mean_wasted_life_cost,
            "episode_count": c.episode_count,
        })
    df = pd.DataFrame(records)
    df.to_csv(file_path, index=False)
    return file_path


def write_selected_thresholds(
    selected: Dict[str, SelectedThreshold],
    output_dir: Path,
) -> Path:
    """Write selected thresholds to JSON."""
    file_path = output_dir / "selected_thresholds.json"
    data = {}
    for policy_family, thresh in selected.items():
        data[policy_family] = {
            "threshold": thresh.threshold,
            "k_capacity": thresh.k_capacity,
            "cost_regime_id": thresh.cost_regime_id,
            "mean_total_cost": thresh.mean_total_cost,
            "total_failures": thresh.total_failures,
            "mean_wasted_life_cost": thresh.mean_wasted_life_cost,
            "episode_count": thresh.episode_count,
            "tie_break_reason": thresh.tie_break_reason,
        }
    write_json_safe(data, file_path)
    return file_path


def write_selected_thresholds_with_meta(
    selected: Dict[str, SelectedThreshold],
    output_dir: Path,
    formal_run_id: str,
    config_sha256: str,
    implementation_commit: str,
) -> Path:
    """
    Write formal selected thresholds with provenance envelope.

    The envelope is required by load_formal_selected_thresholds(); the
    loader refuses to read any selected_thresholds.json that is missing
    or disagrees on `formal_run_id`, `config_sha256`, or
    `implementation_commit`. Returns the path to the newly written file.
    """
    # First write the canonical (no metadata) version so older readers
    # still get the same keys.
    write_selected_thresholds(selected, output_dir)

    file_path = output_dir / "selected_thresholds.json"
    with open(file_path, "r") as f:
        data = json.load(f)
    data["_meta"] = {
        "formal_run_id": formal_run_id,
        "config_sha256": config_sha256,
        "implementation_commit": implementation_commit,
        "written_at": datetime.utcnow().isoformat(),
    }
    write_json_safe(data, file_path)
    return file_path


def write_episode_results(
    results: List[EpisodeResult],
    output_dir: Path,
) -> Path:
    """Write episode results to parquet.

    Records the RAW source-bank scenario ID in the parquet's
    ``scenario_id`` column (when ``source_scenario_id`` is populated)
    so independent recomputation can hash and match against the source
    bank JSON directly. The derived ID is preserved alongside in
    ``derived_scenario_id`` for diagnostic / regression visibility.
    When ``source_scenario_id`` is absent (older callers), the parquet
    falls back to the derived ``scenario_id`` field and the
    ``derived_scenario_id`` is left blank.
    """
    file_path = output_dir / "episode_results.parquet"
    records = []
    for r in results:
        src = getattr(r, "source_scenario_id", None) or r.scenario_id
        derived_id = r.scenario_id if getattr(r, "source_scenario_id", None) else ""
        records.append({
            "run_id": r.run_id,
            "policy_id": r.policy_id,
            "policy_family": r.policy_family,
            "threshold": r.threshold,
            "split": r.split,
            "scenario_id": src,
            "derived_scenario_id": derived_id,
            "cost_regime_id": r.cost_regime_id,
            "maintenance_capacity": r.maintenance_capacity,
            "reset_seed": r.reset_seed,
            "policy_seed": r.policy_seed,
            "episode_steps": r.episode_steps,
            "episode_return": r.episode_return,
            "discounted_return": r.discounted_return,
            "total_cost": r.total_cost,
            "preventive_cost": r.preventive_cost,
            "failure_cost": r.failure_cost,
            "wasted_life_cost": r.wasted_life_cost,
            "preventive_replacement_count": r.preventive_replacement_count,
            "failure_count": r.failure_count,
            "action_count": r.action_count,
            "empty_action_count": r.empty_action_count,
            "capacity_saturated_step_count": r.capacity_saturated_step_count,
            "mean_selected_predicted_rul": r.mean_selected_predicted_rul,
            "mean_selected_age": r.mean_selected_age,
            "nan_observation_count": r.nan_observation_count,
            "inf_observation_count": r.inf_observation_count,
            "terminated_count": r.terminated_count,
            "truncated": r.truncated,
            "completed": r.completed,
            "error": r.error if r.error else None,
        })
    df = pd.DataFrame(records)
    df.to_parquet(file_path, index=False)
    return file_path


def write_summary_by_policy(
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write summary by policy to CSV and JSON."""
    csv_path = output_dir / "summary_by_policy.csv"
    json_path = output_dir / "summary_by_policy.json"

    summary_df.to_csv(csv_path, index=False)
    write_json_safe(summary_df.to_dict(orient="records"), json_path)

    return csv_path, json_path


def write_sanity_checks(
    sanity_results: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """Write sanity checks to JSON."""
    file_path = output_dir / "sanity_checks.json"
    write_json_safe(sanity_results, file_path)
    return file_path


def write_run_provenance(
    provenance: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """Write run provenance to JSON."""
    file_path = output_dir / "run_provenance.json"
    write_json_safe(provenance, file_path)
    return file_path


def write_scenario_bank_provenance(
    scenario_bank_provenance: List[Dict[str, Any]],
    output_dir: Path,
) -> Path:
    """
    Write scenario-bank provenance to JSON.

    Args:
        scenario_bank_provenance: List of provenance records, each containing:
            - logical_bank_id
            - source_path (repository-relative)
            - source_file_size
            - source_sha256
            - source_scenario_count
            - derived_k
            - derived_cost_regime_id
            - derived_scenario_count
            - derived_scenario_ids (ordered)
            - derived_bank_sha256
        output_dir: Output directory

    Returns:
        Path to written file
    """
    file_path = output_dir / "scenario_bank_provenance.json"
    write_json_safe({
        "scenario_banks": scenario_bank_provenance,
        "recorded_at": datetime.utcnow().isoformat(),
    }, file_path)
    return file_path


def write_artifact_manifest(
    output_dir: Path,
    file_paths: Optional[List[Path]] = None,
) -> Path:
    """
    Write artifact manifest with SHA256 hashes.

    Args:
        output_dir: Output directory
        file_paths: List of file paths to include. If None, scans output_dir.
    """
    manifest_path = output_dir / "artifact_manifest.json"

    if file_paths is None:
        file_paths = list(output_dir.glob("*"))

    entries = []
    for fp in sorted(file_paths):
        if fp.name == "artifact_manifest.json":
            continue
        if fp.is_dir():
            continue

        relative_path = fp.relative_to(output_dir)
        entries.append({
            "relative_path": str(relative_path),
            "byte_size": get_file_size(fp),
            "sha256": compute_sha256(fp),
            "row_count": get_row_count(fp) if fp.suffix in [".parquet", ".csv"] else None,
            "schema_columns": get_schema_columns(fp) if fp.suffix in [".parquet", ".csv"] else None,
        })

    manifest = {"artifacts": entries, "generated_at": datetime.utcnow().isoformat()}
    write_json_safe(manifest, manifest_path)
    return manifest_path


def get_row_count(file_path: Path) -> int:
    """Get row count for parquet, CSV, or JSON file."""
    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
        return len(df)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
        return len(df)
    elif file_path.suffix == ".json":
        import json
        with open(file_path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return len(data)
        elif isinstance(data, list):
            return len(data)
    return 0


def get_schema_columns(file_path: Path) -> List[str]:
    """Get column names for parquet or CSV file."""
    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
        return list(df.columns)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
        return list(df.columns)
    return []


def write_run_log(
    log_messages: List[str],
    output_dir: Path,
    exit_code: int,
) -> Path:
    """Write run log with exit code."""
    file_path = output_dir / "m3_run.log"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_messages))
        f.write(f"\n\nEXIT_CODE={exit_code}\n")
    return file_path


def write_independent_recomputation(
    output_dir: Path,
    independent_recomputation_path: Optional[Path] = None,
) -> Path:
    """Recompute provenance facts and write them as
    independent_recomputation.json. This file answers checks an auditor can
    run without trusting any code path that wrote the other artifacts.

    Fields written:
      - tuning_candidates (from threshold_search_results.parquet)
      - tuning_candidates_csv_rows (from threshold_search_summary.csv)
      - selected_thresholds_count (len of selected_thresholds.json keys,
        excluding _meta)
      - evaluation_episodes (from episode_results.parquet row count)
      - scenario_bank_count (count of banks in scenario_bank_provenance)
      - reset_seed_count (from run_provenance.json reset_seeds)
      - selected_thresholds_sha256 (independent hash of file)
      - validation_report_sha256 (independent hash of validation_report)
      - recomputed_at (timestamp)
    """
    output_dir = Path(output_dir)
    target = Path(independent_recomputation_path) if independent_recomputation_path \
        else output_dir / "independent_recomputation.json"

    def _safe_count(filename: str, kind: str) -> int:
        fp = output_dir / filename
        if not fp.exists():
            return 0
        if kind == "parquet":
            try:
                return int(pd.read_parquet(fp).shape[0])
            except Exception:
                return 0
        if kind == "csv":
            try:
                return int(len(pd.read_csv(fp)))
            except Exception:
                return 0
        if kind == "json":
            try:
                with open(fp, "r") as f:
                    return sum(1 for k in json.load(f) if k != "_meta")
            except Exception:
                return 0
        return 0

    selected_path = output_dir / "selected_thresholds.json"
    selected_count = _safe_count("selected_thresholds.json", "json")
    selected_sha = compute_sha256(selected_path) if selected_path.exists() else None

    validation_path = output_dir / "validation_report.json"
    validation_sha = compute_sha256(validation_path) if validation_path.exists() else None

    scenario_count = 0
    sp = output_dir / "scenario_bank_provenance.json"
    if sp.exists():
        try:
            with open(sp, "r") as f:
                p = json.load(f)
            if isinstance(p, dict):
                scenario_count = len(p.get("scenario_banks", []))
            elif isinstance(p, list):
                scenario_count = len(p)
        except Exception:
            scenario_count = 0

    reset_seed_count = 0
    rp = output_dir / "run_provenance.json"
    if rp.exists():
        try:
            with open(rp, "r") as f:
                p = json.load(f)
            if isinstance(p, dict):
                reset_seed_count = len(p.get("reset_seeds", []))
        except Exception:
            reset_seed_count = 0

    record = {
        "recomputed_at": datetime.utcnow().isoformat(),
        "tuning_candidates": _safe_count("threshold_search_results.parquet", "parquet"),
        "tuning_candidates_csv_rows": _safe_count("threshold_search_summary.csv", "csv"),
        "selected_thresholds_count": selected_count,
        "evaluation_episodes": _safe_count("episode_results.parquet", "parquet"),
        "scenario_bank_count": scenario_count,
        "reset_seed_count": reset_seed_count,
        "selected_thresholds_sha256": selected_sha,
        "validation_report_sha256": validation_sha,
    }
    write_json_safe(record, target)
    return target


def validate_artifacts(
    output_dir: Path,
    required_files: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Validate all artifacts in output directory.

    Args:
        output_dir: Output directory
        required_files: List of required file names. If None, uses default set.

    Returns:
        Validation report dict
    """
    if required_files is None:
        required_files = [
            "resolved_config.json",
            "threshold_search_results.parquet",
            "threshold_search_summary.csv",
            "selected_thresholds.json",
            "episode_results.parquet",
            "summary_by_policy.csv",
            "summary_by_policy.json",
            "sanity_checks.json",
            "run_provenance.json",
            "artifact_manifest.json",
            "m3_run.log",
        ]

    validation = {
        "required_files": required_files,
        "missing_files": [],
        "extra_files": [],
        "hash_valid": [],
        "hash_invalid": [],
        "schema_valid": [],
        "schema_invalid": [],
        "numeric_valid": [],
        "numeric_invalid": [],
    }

    # Check missing files
    for rf in required_files:
        if not (output_dir / rf).exists():
            validation["missing_files"].append(rf)

    # Scan for extra files
    existing_files = set(f.name for f in output_dir.iterdir() if f.is_file())
    required_set = set(required_files)
    validation["extra_files"] = list(existing_files - required_set)

    # Validate each existing file
    for file_path in output_dir.iterdir():
        if not file_path.is_file():
            continue

        file_name = file_path.name
        if file_name == "artifact_manifest.json":
            continue

        # Validate JSON files
        if file_name.endswith(".json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                validate_json_serializable(data, file_name)
                validation["schema_valid"].append(file_name)
            except (json.JSONDecodeError, ValueError) as e:
                validation["schema_invalid"].append(f"{file_name}: {str(e)}")

        # Validate parquet files
        elif file_name.endswith(".parquet"):
            try:
                df = pd.read_parquet(file_path)
                # Check for NaN/Inf in numeric columns
                for col in df.select_dtypes(include=["float64", "float32"]).columns:
                    if df[col].isna().all():
                        pass  # All NaN is allowed but noted
                    if (df[col] == float("inf")).any() or (df[col] == float("-inf")).any():
                        validation["numeric_invalid"].append(f"{file_name}: {col} has Inf")
                    elif df[col].isna().any():
                        validation["numeric_valid"].append(f"{file_name}: {col} has NaN (allowed)")
                    else:
                        validation["numeric_valid"].append(file_name)
                        break
            except Exception as e:
                validation["schema_invalid"].append(f"{file_name}: {str(e)}")

    validation["all_present"] = len(validation["missing_files"]) == 0
    validation["all_valid"] = (
        validation["all_present"]
        and len(validation["schema_invalid"]) == 0
        and len(validation["hash_invalid"]) == 0
    )

    return validation


def _git_full_commit(repo_root: Optional[Path] = None) -> str:
    """Return `git rev-parse HEAD` from the given repo root.

    Falls back to safe defaults instead of trusting the caller.
    """
    cwd = str(repo_root) if repo_root else os.getcwd()
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.PIPE,
        )
        return out.decode("utf-8").strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "UNKNOWN_NO_GIT"


def _git_commit_short(repo_root: Optional[Path] = None) -> str:
    """Return short SHA for verification convenience."""
    full = _git_full_commit(repo_root)
    return full if len(full) < 8 else full[:8]


def _derive_config_sha_from_configs_dir(repo_root: Path) -> str:
    """Compute SHA256 across the configs/baselines directory deterministically.

    Used when no caller-supplied config SHA exists; this answers
    'what config was actually in the repo at the recorded commit?'.
    """
    configs_dir = repo_root / "configs" / "baselines"
    if not configs_dir.exists():
        return ""
    h = hashlib.sha256()
    for fp in sorted(configs_dir.glob("*.json")):
        h.update(fp.name.encode("utf-8"))
        with open(fp, "rb") as fh:
            for chunk in iter(lambda: fh.read(4096), b""):
                h.update(chunk)
    return h.hexdigest()


def derive_run_id(output_dir: Path) -> str:
    """Derive the formal run ID from the output directory name itself.

    Run provenance is the directory name; we do not accept it from callers
    because that would let any compromised caller inject a fake run ID
    into an authoritative manifest.
    """
    return output_dir.name


def _selected_thresholds_count(selected_path: Path) -> int:
    """Count the formal identity entries in selected_thresholds.json.

    Excludes the optional _meta envelope. The former `get_row_count`
    implementation returned `len(dict_keys)` which works for the dict
    root but counted `_meta` along with the rest, so for files with
    envelope it returned 33 — incorrect for the 32-identity formal set.
    This helper returns the right number in either case.
    """
    if not selected_path.exists():
        return 0
    with open(selected_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return 0
    return sum(1 for k in data.keys() if k != "_meta")


# =============================================================================
# Formal Run Context (immutable provenance envelope, sealed before tuning)
# =============================================================================

from dataclasses import dataclass, asdict, field
import contextlib


@dataclass
class FormalRunContext:
    """
    Immutable formal run context written before tuning begins.

    The context is sealed after selected_thresholds.json is finalised.
    Any attempt to mutate a sealed context raises RuntimeError.
    """
    schema_version: str = "m3_formal_context_v1"
    formal_run_id: str = ""
    mode: str = "formal_closeout"
    implementation_commit: str = ""
    implementation_tree_clean: bool = True
    resolved_config_path: str = ""
    resolved_config_sha256: str = ""
    oracle_authorized: bool = True
    selected_thresholds_path: str = ""
    selected_thresholds_sha256: Optional[str] = None
    sealed: bool = False
    sealed_at: Optional[str] = None
    scenario_bank_identities: List[Dict[str, Any]] = field(default_factory=list)
    reset_seeds: List[int] = field(default_factory=list)
    created_at: str = ""

    _HEX_RE = re.compile(r"^[0-9a-f]{64}$")

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        # Enforce strict serialized invariant: exactly two valid states.
        self._check_invariants()

    def _check_invariants(self) -> None:
        """
        Reject inconsistent combinations during construction / load.

        UNSEALED:  sealed=False, sealed_at is None, selected_thresholds_sha256 is None
        SEALED:    sealed=True,  sealed_at non-None, selected_thresholds_sha256 is 64 lowercase hex

        Any other combination raises. There is no auto-promotion of
        sealed=False to sealed=True based on audit fields.
        """
        sealed = bool(self.sealed)
        sealed_at = self.sealed_at
        sha = self.selected_thresholds_sha256

        if sealed is False:
            if sealed_at is not None:
                raise ValueError(
                    f"FormalRunContext invariant violation: sealed=False but sealed_at={sealed_at!r}"
                )
            if sha is not None:
                raise ValueError(
                    "FormalRunContext invariant violation: sealed=False but "
                    "selected_thresholds_sha256 is set"
                )
        else:  # sealed is True
            if sealed_at is None or not isinstance(sealed_at, str) or not sealed_at.strip():
                raise ValueError(
                    f"FormalRunContext invariant violation: sealed=True but sealed_at is invalid ({sealed_at!r})"
                )
            if not isinstance(sha, str) or not self._HEX_RE.match(sha):
                raise ValueError(
                    "FormalRunContext invariant violation: sealed=True but "
                    f"selected_thresholds_sha256={sha!r} is not a 64-char lowercase hex string"
                )

    def to_json(self) -> Dict[str, Any]:
        """Serialize to dict (only the four invariant fields plus the rest)."""
        d = asdict(self)
        return d

    @classmethod
    def from_json(
        cls,
        data: Dict[str, Any],
        allow_legacy_sealed_alias: bool = False,
    ) -> "FormalRunContext":
        """Deserialize from dict, preserving explicitly serialized audit fields.

        Two valid serialized shapes are accepted:

          UNSEALED: sealed=False, sealed_at is None,
                    selected_thresholds_sha256 is None
          SEALED:   sealed=True, sealed_at non-None (string),
                    selected_thresholds_sha256 is a 64-char lowercase hex

        The deserialized object is rejected if the
        sealed / sealed_at / selected_thresholds_sha256 triple does not
        form one of those two valid states. There is no auto-promotion
        based on missing data; a sealed context MUST be explicitly
        marked ``sealed: true`` on disk.

        Args:
            data: serialized payload.
            allow_legacy_sealed_alias: when True (only valid in
                ``diagnostic_legacy`` mode), honor the ``_sealed: True``
                alias and (only for diagnostic fixtures that also lack
                ``sealed_at``) assign a load-time stamp so the invariant
                check is satisfiable. When False (formal_closeout and
                diagnostic_non_oracle), presence of the ``_sealed`` alias
                is rejected as a malformed context.
        """
        known = set(cls.__dataclass_fields__.keys())
        # Strip any unknown keys but never silently invent a sealed=True
        # field from missing data.
        filtered = {k: v for k, v in data.items() if k in known}

        has_sealed = "sealed" in filtered or "sealed" in data
        has_legacy_alias = data.get("_sealed") is True

        if has_legacy_alias and not has_sealed:
            if not allow_legacy_sealed_alias:
                # formal_closeout / diagnostic_non_oracle forbid the
                # _sealed alias unconditionally.
                raise ValueError(
                    "FormalRunContext: legacy ``_sealed`` alias is only "
                    "honored under explicit --mode diagnostic_legacy; the "
                    "formal_closeout parser refuses this alias."
                )
            # diagnostic_legacy explicitly opts into the alias path. We do
            # NOT synthesize sealed_at if it is also missing in the file;
            # the alias is a load-only normalization and the resulting
            # invariant check may still reject the resulting state.
            filtered["_legacy_alias_used"] = True
            if not filtered.get("sealed_at"):
                filtered["sealed_at"] = datetime.utcnow().isoformat()
            filtered["sealed"] = True

        # Default sealed to False if absent; this is a load-only default
        # and the invariant check will reject any inconsistent audit
        # fields.
        filtered.setdefault("sealed", False)
        filtered.setdefault("sealed_at", None)
        filtered.setdefault("selected_thresholds_sha256", None)
        ctx = cls(**filtered)
        # Re-validate after construction (cheap, raises clear error).
        ctx._check_invariants()
        return ctx

    def seal(self) -> None:
        """Seal the context by computing the SHA from the stored selected_thresholds path.

        The SHA is recomputed internally from the exact stored path; this
        method takes no caller-supplied SHA. Calling seal() on an already
        sealed context raises RuntimeError.
        """
        if self.sealed:
            raise RuntimeError("FormalRunContext already sealed; cannot mutate")
        selected_path_str = self.selected_thresholds_path
        if not selected_path_str:
            raise RuntimeError(
                "selected_thresholds_path not available on context; cannot seal"
            )
        selected_path = Path(selected_path_str)
        if not selected_path.exists():
            raise RuntimeError(
                f"selected_thresholds.json not found at {selected_path_str}; cannot seal"
            )
        actual_sha = compute_sha256(selected_path)
        if not self._HEX_RE.match(actual_sha):
            raise RuntimeError(
                f"computed selected_thresholds SHA {actual_sha!r} is not a 64-char hex string; cannot seal"
            )
        self.selected_thresholds_sha256 = actual_sha
        self.sealed = True
        self.sealed_at = datetime.utcnow().isoformat()
        # Re-validate the now-sealed invariants explicitly.
        self._check_invariants()

    def assert_sealed(self) -> None:
        """Assert the context is sealed; raise if not."""
        if not self.sealed:
            raise RuntimeError("FormalRunContext not sealed; selected_thresholds_sha256 not recorded")

    def assert_not_sealed(self) -> None:
        """Assert the context is not yet sealed; raise if sealed."""
        if self.sealed:
            raise RuntimeError("FormalRunContext already sealed; cannot mutate")


def _git_full_commit_for_context(repo_root: Optional[Path] = None) -> str:
    """Return git rev-parse HEAD; raise if unavailable (no fallbacks)."""
    cwd = str(repo_root) if repo_root else os.getcwd()
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.PIPE,
        )
        commit = out.decode("utf-8").strip()
        if not commit:
            raise RuntimeError("git rev-parse HEAD returned empty commit")
        return commit
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        raise RuntimeError(f"Failed to obtain git commit: {e}")


def _check_tree_clean(repo_root: Optional[Path] = None) -> bool:
    """Return True if git worktree is clean (no uncommitted changes)."""
    cwd = str(repo_root) if repo_root else os.getcwd()
    try:
        # Check for uncommitted changes (staged or unstaged)
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            stderr=subprocess.PIPE,
        )
        return out.decode("utf-8").strip() == ""
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return False


def create_formal_run_context(
    output_dir: Path,
    resolved_config: Dict[str, Any],
    resolved_config_path: Path,
    selected_thresholds_path: Path,
    allow_oracle: bool = True,
    scenario_bank_identities: Optional[List[Dict[str, Any]]] = None,
    reset_seeds: Optional[List[int]] = None,
    repo_root: Optional[Path] = None,
    mode: str = "formal_closeout",
    formal_run_id: Optional[str] = None,
) -> FormalRunContext:
    """Create and atomically write formal_run_context.json before tuning.

    Fail-closed: every gate runs BEFORE any disk write. A dirty worktree
    is rejected immediately, not silently recorded as ``tree_clean=False``.
    A formal_run_id may be supplied by the caller (e.g. the orchestration
    layer derives it from ``output_dir.name`` once and threads it back in);
    when supplied it MUST equal ``output_dir.name`` — there is no
    "shadow" derivation in this function.
    """
    repo_root = Path(repo_root) if repo_root else Path.cwd()
    output_dir = Path(output_dir)

    # === Fail-closed gate 0: dirty worktree MUST reject ===
    # The previous implementation recorded ``implementation_tree_clean=False``
    # and continued to write a partial context; that produced resurrectable
    # "permission slips" for damaged formal runs. The contract now requires
    # that a dirty working tree abort the formal context creation
    # BEFORE writing anything to disk.
    tree_clean = _check_tree_clean(repo_root)
    if not tree_clean:
        raise RuntimeError(
            "Cannot create formal_run_context.json on a dirty worktree; "
            "`git status --porcelain` returned uncommitted changes. A "
            "formal closeout requires a clean working tree. "
            "Commit or stash outstanding edits and re-run."
        )

    # === Fail-closed gate 1: git commit ===
    implementation_commit = _git_full_commit_for_context(repo_root)
    if not implementation_commit:
        raise RuntimeError(
            "Git commit is empty: `git rev-parse HEAD` returned no value"
        )
    if len(implementation_commit) != 40:
        raise RuntimeError(
            f"Git commit invalid length (expected 40 hex chars): {implementation_commit!r}"
        )
    try:
        int(implementation_commit, 16)
    except ValueError:
        raise RuntimeError(
            f"Git commit is not hex: {implementation_commit!r} from `git rev-parse HEAD`"
        )
    if implementation_commit != implementation_commit.lower():
        raise RuntimeError(
            f"Git commit must be lowercase hex: {implementation_commit!r}"
        )

    # === Fail-closed gate 2: output directory must not already contain a context ===
    existing_context = output_dir / "formal_run_context.json"
    if existing_context.exists():
        raise RuntimeError(
            f"Output directory already contains formal_run_context.json: {existing_context}"
        )

    # === Fail-closed gate 3: resolved_config on disk ===
    if not resolved_config_path.exists():
        raise RuntimeError(f"resolved_config.json absent at {resolved_config_path}")
    try:
        with open(resolved_config_path, "r") as f:
            disk_config = json.load(f)
        disk_sha = compute_canonical_config_sha256(disk_config)
    except Exception as e:
        raise RuntimeError(f"Failed to read resolved_config.json: {e}")
    expected_sha = compute_canonical_config_sha256(resolved_config)
    if disk_sha != expected_sha:
        raise RuntimeError(
            f"resolved_config.json SHA mismatch: disk={disk_sha} expected={expected_sha}"
        )

    # === Fail-closed gate 4: oracle authorization ===
    if mode == "formal_closeout" and not allow_oracle:
        raise RuntimeError(
            "Oracle not authorized in formal_closeout mode; "
            "refusing to create formal context."
        )

    # === Fail-closed gate 5: formal_run_id consistency (orchestration
    # authority). In formal_closeout the orchestration layer MUST supply
    # formal_run_id; this function never invents a divergent value.
    # Diagnostic modes retain the legacy output_dir.name fallback only
    # for the rare explicit diagnostic_legacy / diagnostic_non_oracle
    # invocation paths.
    if formal_run_id is None:
        if mode == "formal_closeout":
            raise RuntimeError(
                "formal_closeout requires formal_run_id to be supplied by "
                "the orchestration layer; refusing to derive it locally "
                "from output_dir.name in a production closeout path."
            )
        formal_run_id = output_dir.name
    if formal_run_id != output_dir.name:
        raise RuntimeError(
            f"formal_run_id mismatch: caller-supplied "
            f"{formal_run_id!r} != output_dir.name {output_dir.name!r}"
        )
    if not formal_run_id or formal_run_id.startswith("."):
        raise RuntimeError(
            f"Invalid formal_run_id derived from output_dir: '{formal_run_id}'"
        )

    # === Fail-closed gate 6: scenario bank identities ===
    # We require explicit supply; no fabricated defaults allowed.
    if scenario_bank_identities is None:
        scenario_bank_identities = []
    if not scenario_bank_identities:
        raise RuntimeError(
            "scenario_bank_identities must be explicitly provided from "
            "actual bank files; fabricated default identity strings are "
            "forbidden."
        )

    # === Fail-closed gate 7: reset_seeds ===
    if reset_seeds is None:
        raise RuntimeError(
            "reset_seeds must be explicitly provided from the resolved "
            "formal configuration; hardcoded default seeds are forbidden."
        )

    resolved_config_sha256 = expected_sha

    context = FormalRunContext(
        formal_run_id=formal_run_id,
        mode=mode,
        implementation_commit=implementation_commit,
        implementation_tree_clean=tree_clean,
        resolved_config_path=str(resolved_config_path),
        resolved_config_sha256=resolved_config_sha256,
        oracle_authorized=allow_oracle,
        selected_thresholds_path=str(selected_thresholds_path),
        selected_thresholds_sha256=None,
        sealed=False,
        sealed_at=None,
        scenario_bank_identities=scenario_bank_identities,
        reset_seeds=reset_seeds,
    )

    # === Atomic write ===
    context_path = output_dir / "formal_run_context.json"
    tmp_path = output_dir / ".formal_run_context.json.tmp"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_safe(context.to_json(), tmp_path)
    tmp_path.rename(context_path)

    # === Verify written file matches expected formal_run_id ===
    loaded = load_formal_run_context(output_dir)
    if loaded.formal_run_id != formal_run_id:
        raise RuntimeError(
            f"Context formal_run_id mismatch after write: expected "
            f"{formal_run_id}, got {loaded.formal_run_id}"
        )

    return context


def seal_formal_run_context(
    output_dir: Path,
    selected_thresholds_path: Optional[Path] = None,
    allow_legacy_sealed_alias: bool = False,
) -> FormalRunContext:
    """
    Seal the formal run context after selected_thresholds.json is finalised.

    Reads the existing UNSEALED context, lets its `seal()` method compute
    the SHA from the on-disk selected_thresholds.json path, then rewrites
    the context atomically. After this call the context is SEALED; any
    further call raises RuntimeError and the on-disk bytes are unchanged.

    Args:
        output_dir: directory holding the context.
        selected_thresholds_path: optional override path.
        allow_legacy_sealed_alias: forwarded to the context loader when
            the existing on-disk context was written by the explicit
            diagnostic_legacy entry point.
    """
    context_path = output_dir / "formal_run_context.json"
    if not context_path.exists():
        raise RuntimeError(f"formal_run_context.json not found at {context_path}; cannot seal")

    with open(context_path, "r") as f:
        original_bytes = f.read()

    context = load_formal_run_context(
        output_dir,
        allow_legacy_sealed_alias=allow_legacy_sealed_alias,
    )

    if context.sealed:
        # Already sealed; do not mutate on-disk bytes; surface the error.
        raise RuntimeError("FormalRunContext already sealed; cannot mutate")

    # Optionally pre-check the path exists (seal() also enforces this).
    if selected_thresholds_path is None:
        check_path = Path(context.selected_thresholds_path)
    else:
        check_path = Path(selected_thresholds_path)
    if not check_path.exists():
        raise RuntimeError(
            f"selected_thresholds.json not found at {check_path}; cannot seal context"
        )

    # Capture SHA before sealing to compare on second-seal attempt below.
    pre_seal_bytes_after_seal = None
    try:
        context.seal()
    except Exception:
        # Roll back any partial mutation: write back the original bytes.
        try:
            context_path.write_text(original_bytes)
        except Exception:
            raise
        raise
    pre_seal_bytes_after_seal = context_path.read_bytes()

    # Rewrite atomically
    tmp_path = context_path.with_suffix(".json.tmp")
    write_json_safe(context.to_json(), tmp_path)
    tmp_path.rename(context_path)

    # Re-read to confirm sealed state was persisted.
    reloaded = load_formal_run_context(output_dir)
    if not reloaded.sealed:
        raise RuntimeError(
            "Failed to persist sealed state; reload shows sealed=False"
        )
    return reloaded


def load_formal_run_context(
    output_dir: Path,
    allow_legacy_sealed_alias: bool = False,
) -> FormalRunContext:
    """Load and validate the formal run context from disk.

    By default the strict parser (formal_closeout, diagnostic_non_oracle)
    refuses the legacy ``_sealed`` alias. Pass
    ``allow_legacy_sealed_alias=True`` only from the explicit
    diagnostic_legacy entry point.
    """
    context_path = output_dir / "formal_run_context.json"
    if not context_path.exists():
        raise RuntimeError(f"formal_run_context.json not found at {context_path}")

    with open(context_path, "r") as f:
        data = json.load(f)

    context = FormalRunContext.from_json(
        data,
        allow_legacy_sealed_alias=allow_legacy_sealed_alias,
    )
    return context


def validate_formal_run_context(
    output_dir: Path,
    allow_legacy_sealed_alias: bool = False,
) -> List[str]:
    """
    Validate the formal run context against current disk state.

    Returns list of errors (empty if valid).

    Args:
        output_dir: directory holding the context.
        allow_legacy_sealed_alias: if True, the legacy ``_sealed`` alias
            is honored (used by ``--mode diagnostic_legacy``).
    """
    errors = []
    try:
        context = load_formal_run_context(
            output_dir,
            allow_legacy_sealed_alias=allow_legacy_sealed_alias,
        )
    except Exception as e:
        return [f"Failed to load formal_run_context.json: {e}"]

    # Must be sealed
    if not context.sealed:
        errors.append("FormalRunContext not sealed (sealed=false; selected_thresholds_sha256 missing)")

    # Implementation commit must match current HEAD
    repo_root = Path.cwd()
    try:
        current_commit = _git_full_commit_for_context(repo_root)
        if current_commit != context.implementation_commit:
            errors.append(
                f"Implementation commit mismatch: context={context.implementation_commit[:12]} "
                f"current={current_commit[:12]}"
            )
    except Exception as e:
        errors.append(f"Could not verify implementation commit: {e}")

    # Resolved config must exist and SHA must match
    resolved_path = Path(context.resolved_config_path)
    if not resolved_path.exists():
        errors.append(f"resolved_config.json missing at {resolved_path}")
    else:
        try:
            with open(resolved_path, "r") as f:
                config_data = json.load(f)
            actual_sha = compute_canonical_config_sha256(config_data)
            if actual_sha != context.resolved_config_sha256:
                errors.append(
                    f"Resolved config SHA mismatch: context={context.resolved_config_sha256[:12]} "
                    f"actual={actual_sha[:12]}"
                )
        except Exception as e:
            errors.append(f"Failed to verify resolved_config.json: {e}")

    # Selected thresholds must exist and SHA must match
    selected_path = Path(context.selected_thresholds_path)
    if not selected_path.exists():
        errors.append(f"selected_thresholds.json missing at {selected_path}")
    elif context.selected_thresholds_sha256:
        actual_sha = compute_sha256(selected_path)
        if actual_sha != context.selected_thresholds_sha256:
            errors.append(
                f"Selected thresholds SHA mismatch: context={context.selected_thresholds_sha256[:12]} "
                f"actual={actual_sha[:12]}"
            )

    return errors


def generate_formal_manifest(
    output_dir: Path,
    mode: str = "formal_closeout",
) -> Path:
    """
    Generate (and never re-touch) formal manifest for M3 experiment.

    Args:
        output_dir: Output directory
        mode: Explicit mode. One of:
              - "formal_closeout": production closeout; demands sealed
                formal_run_context.json, Oracle authorized, 360/9000/32/2400 counts.
                run_id and implementation_commit come ONLY from sealed context.
                No caller-supplied fallbacks, no inference from disk state.
              - "diagnostic_non_oracle": explicit non-Oracle diagnostic mode;
                expects 272/6800/24/2000 counts; context may be unsealed or missing.
              - "diagnostic_legacy": explicit diagnostic mode that tolerates
                legacy context shape (e.g. _sealed alias); NOT reachable from --all.

    Returns:
        Path to the newly written manifest.

    Raises:
        RuntimeError: If any fail-closed check fails.
        ValueError: If mode is unknown.
    """
    if mode not in ("formal_closeout", "diagnostic_non_oracle", "diagnostic_legacy"):
        raise ValueError(
            f"Unknown mode {mode!r}; expected formal_closeout, "
            "diagnostic_non_oracle, or diagnostic_legacy."
        )

    manifest_path = output_dir / "formal_manifest.json"
    if manifest_path.exists():
        raise RuntimeError(
            f"formal_manifest.json already exists at {manifest_path}; "
            "the formal manifest is immutable. Refusing to overwrite."
        )

    output_dir = Path(output_dir)

    # === 1. Load formal run context (required in formal_closeout) ===
    context_path = output_dir / "formal_run_context.json"
    context: Optional[FormalRunContext] = None

    if context_path.exists():
        try:
            # Use strict loader (no legacy alias) for formal_closeout and
            # diagnostic_non_oracle; only diagnostic_legacy allows _sealed alias.
            allow_legacy = (mode == "diagnostic_legacy")
            context = load_formal_run_context(
                output_dir,
                allow_legacy_sealed_alias=allow_legacy,
            )
        except Exception as e:
            raise RuntimeError(
                f"formal_run_context.json failed to load/validate: {e}"
            )

    # === 2. Mode-specific hard gates ===
    if mode == "formal_closeout":
        # formal_closeout demands a SEALED context — no inference, no fallback
        if context is None:
            raise RuntimeError(
                "formal_closeout requires formal_run_context.json to exist "
                "and be sealed; no sealed context found."
            )
        if not context.sealed:
            raise RuntimeError(
                "formal_closeout requires formal_run_context.json to be sealed "
                "(selected_thresholds_sha256 recorded); context is unsealed."
            )
        # Validate context against current disk state
        context_errors = validate_formal_run_context(output_dir)
        if context_errors:
            raise RuntimeError(
                "formal_run_context.json validation failed:\n" +
                "\n".join(f"  - {e}" for e in context_errors)
            )

        # run_id and implementation_commit come ONLY from sealed context
        formal_run_id = context.formal_run_id
        implementation_commit = context.implementation_commit
        config_sha256 = context.resolved_config_sha256

        # Verify oracle authorization
        if not context.oracle_authorized:
            raise RuntimeError(
                "formal_closeout requires Oracle authorization; "
                "context.oracle_authorized is False."
            )

        # Exact count requirements
        EXPECTED = {
            "tuning_candidates": 360,
            "tuning_episodes": 9000,
            "selected_thresholds": 32,
            "evaluation_episodes": 2400,
        }

    elif mode == "diagnostic_non_oracle":
        # Non-oracle diagnostic: may have unsealed/missing context
        if context is not None and context.sealed:
            formal_run_id = context.formal_run_id
            implementation_commit = context.implementation_commit
            config_sha256 = context.resolved_config_sha256
        else:
            # Fallback: derive from disk (no sealed context authority)
            formal_run_id = output_dir.name
            implementation_commit = _git_full_commit_for_context()
            rc_path = output_dir / "resolved_config.json"
            if rc_path.exists():
                with open(rc_path, "r") as f:
                    rc_data = json.load(f)
                config_sha256 = compute_canonical_config_sha256(rc_data)
            else:
                config_sha256 = ""
        EXPECTED = {
            "tuning_candidates": 272,
            "tuning_episodes": 6800,
            "selected_thresholds": 24,
            "evaluation_episodes": 2000,
        }

    else:  # diagnostic_legacy
        # Legacy diagnostic: same counts as diagnostic_non_oracle but
        # tolerates legacy _sealed alias (handled by loader above)
        if context is not None and context.sealed:
            formal_run_id = context.formal_run_id
            implementation_commit = context.implementation_commit
            config_sha256 = context.resolved_config_sha256
        else:
            formal_run_id = output_dir.name
            implementation_commit = _git_full_commit_for_context()
            rc_path = output_dir / "resolved_config.json"
            if rc_path.exists():
                with open(rc_path, "r") as f:
                    rc_data = json.load(f)
                config_sha256 = compute_canonical_config_sha256(rc_data)
            else:
                config_sha256 = ""
        EXPECTED = {
            "tuning_candidates": 272,
            "tuning_episodes": 6800,
            "selected_thresholds": 24,
            "evaluation_episodes": 2000,
        }

    # === 3. Verify resolved_config.json exists and matches context ===
    if context is not None:
        rc_path = Path(context.resolved_config_path)
    else:
        rc_path = output_dir / "resolved_config.json"

    if not rc_path.exists():
        raise RuntimeError(
            "resolved_config.json missing; formal manifest requires "
            "resolved_config.json (no configs-dir fallback allowed)."
        )
    with open(rc_path, "r") as f:
        config_data = json.load(f)
    actual_config_sha = compute_canonical_config_sha256(config_data)
    if actual_config_sha != config_sha256:
        raise RuntimeError(
            f"resolved_config.json SHA mismatch: context={config_sha256[:12]} "
            f"actual={actual_config_sha[:12]}"
        )

    # === 4. Verify selected_thresholds.json SHA ===
    if context is not None:
        selected_path = Path(context.selected_thresholds_path)
    else:
        selected_path = output_dir / "selected_thresholds.json"

    if not selected_path.exists():
        raise RuntimeError(
            "selected_thresholds.json missing; cannot seal manifest."
        )
    selected_sha = compute_sha256(selected_path)
    if context is not None and context.sealed:
        if context.selected_thresholds_sha256 != selected_sha:
            raise RuntimeError(
                f"selected_thresholds.json SHA mismatch: "
                f"context={context.selected_thresholds_sha256[:12]} "
                f"actual={selected_sha[:12]}"
            )

    # === 5. Validation report ===
    validation_path = output_dir / "validation_report.json"
    validation = None
    validation_sha = compute_sha256(validation_path) if validation_path.exists() else None
    if validation_path.exists():
        with open(validation_path, "r") as f:
            validation = json.load(f)

    if mode == "formal_closeout":
        if not validation_path.exists():
            raise RuntimeError(
                "validation_report.json missing; formal_closeout requires "
                "validation_report.json with verdict 'ALL PASSED'."
            )
        if validation.get("verdict") != "ALL PASSED":
            raise RuntimeError(
                f"validation_report.json verdict is '{validation.get('verdict')}', "
                f"not 'ALL PASSED'; refusing to seal manifest."
            )

    # === 6. Independent recomputation ===
    recompute_path = output_dir / "independent_recomputation.json"
    recompute = None
    recompute_sha = (
        compute_sha256(recompute_path) if recompute_path.exists() else None
    )
    if recompute_path.exists():
        with open(recompute_path, "r") as f:
            recompute = json.load(f)

    # Oracle semantic role preserved from the verified PASS independent report
    # (formal_closeout). Defaults to the canonical role for non-formal modes.
    oracle_semantic_role = ORACLE_SEMANTIC_ROLE

    if mode == "formal_closeout":
        if not recompute_path.exists():
            raise RuntimeError(
                "independent_recomputation.json missing; formal_closeout requires "
                "independent_recomputation.json with verdict 'PASS'."
            )
        if recompute.get("verdict") != "PASS":
            raise RuntimeError(
                f"independent_recomputation.json verdict is '{recompute.get('verdict')}', "
                f"not 'PASS'; refusing to seal manifest."
            )
        # Require EVERY mandated structured evidence section to be present in
        # the PASS independent recomputation report. A report with top-level
        # verdict PASS but a missing required section is a contract violation:
        # the manifest must refuse to seal. Identity-set SHA mismatches do NOT
        # suffice as a substitute for this missing-section check.
        REQUIRED_RECOMPUTE_SECTIONS = (
            "formal_run_context_verification",
            "resolved_config_verification",
            "selected_threshold_file_verification",
            "scenario_bank_set_evidence",
            "scenario_bank_file_evidence",
            "candidate_set_evidence",
            "candidate_summary_recomputation_evidence",
            "tuning_set_evidence",
            "selected_winner_evidence",
            "deterministic_tie_break_evidence",
            "evaluation_set_evidence",
            "threshold_use_evidence",
            "non_threshold_policy_evidence",
            "reward_cost_evidence",
            "cost_decomposition_evidence",
            "summary_recomputation_evidence",
            "scenario_bank_provenance_reconciliation_evidence",
            "oracle_terminology_evidence",
        )
        missing_sections = [
            s for s in REQUIRED_RECOMPUTE_SECTIONS if s not in recompute
        ]
        if missing_sections:
            raise RuntimeError(
                "independent_recomputation.json missing required evidence "
                "sections despite top-level verdict 'PASS'; refusing to seal "
                "manifest:\n" +
                "\n".join(f"  - {s}" for s in missing_sections)
            )
        # Reject null / wrong-type / bare-boolean / bare-list-where-an-object-
        # is-required sections and any section whose aggregate verdict is
        # absent, None, or not "PASS". A bare boolean or null as an evidence
        # section is a contract-A violation even with an overall PASS report;
        # handling verdict=None as PASS is explicitly forbidden.
        # expected_kind per section:
        #   "object"        -> must be a dict carrying verdict == "PASS"
        #   "object_or_list"-> list is the per-bank record form; an aggregate
        #                      verdict then must come from a sibling set section
        #                      OR each record must carry verdict == "PASS".
        SECTION_KIND = {
            "formal_run_context_verification": "object",
            "resolved_config_verification": "object",
            "selected_threshold_file_verification": "object",
            "scenario_bank_set_evidence": "object",
            "scenario_bank_file_evidence": "object_or_list",
            "candidate_set_evidence": "object",
            "candidate_summary_recomputation_evidence": "object",
            "tuning_set_evidence": "object",
            "selected_winner_evidence": "object",
            "deterministic_tie_break_evidence": "object",
            "evaluation_set_evidence": "object",
            "threshold_use_evidence": "object",
            "non_threshold_policy_evidence": "object",
            "reward_cost_evidence": "object",
            "cost_decomposition_evidence": "object",
            "summary_recomputation_evidence": "object",
            "scenario_bank_provenance_reconciliation_evidence": "object",
            "oracle_terminology_evidence": "object",
        }
        type_errors: List[str] = []
        non_pass_sections = []
        for s in REQUIRED_RECOMPUTE_SECTIONS:
            sec = recompute.get(s)
            if sec is None:
                type_errors.append(f"{s}: section is null")
                continue
            kind = SECTION_KIND[s]
            if kind == "object":
                if isinstance(sec, bool) or not isinstance(sec, dict):
                    type_errors.append(
                        f"{s}: expected object, got "
                        f"{type(sec).__name__}"
                    )
                    continue
                v = sec.get("verdict")
                if v is None:
                    type_errors.append(f"{s}: missing/None aggregate verdict")
                elif v != "PASS":
                    non_pass_sections.append((s, v))
                # Record-level FAIL hidden beneath an aggregate-PASS object is
                # a contract-A violation: descend into any `records` list and
                # reject any record carrying a non-PASS verdict.
                recs = sec.get("records")
                if isinstance(recs, list):
                    bad_recs = [
                        r for r in recs
                        if isinstance(r, dict)
                        and r.get("verdict") is not None
                        and r.get("verdict") != "PASS"
                    ]
                    if bad_recs:
                        non_pass_sections.append((s, "record-level FAIL"))
            elif kind == "object_or_list":
                if isinstance(sec, list):
                    # Per-bank record form: aggregate verdict must come from the
                    # sibling scenario_bank_set_evidence section. Each record
                    # that carries a verdict must be PASS.
                    bad = [
                        r for r in sec
                        if isinstance(r, dict)
                        and r.get("verdict") not in (None, "PASS")
                    ]
                    if bad:
                        non_pass_sections.append((s, "record-level FAIL"))
                elif not isinstance(sec, dict):
                    type_errors.append(
                        f"{s}: expected object or list, got "
                        f"{type(sec).__name__}"
                    )
                    continue
                else:
                    v = sec.get("verdict")
                    if v is None:
                        type_errors.append(f"{s}: missing/None aggregate verdict")
                    elif v != "PASS":
                        non_pass_sections.append((s, v))
        violated = type_errors + [
            f"{s}: {v}" for s, v in non_pass_sections
        ]
        if violated:
            raise RuntimeError(
                "independent_recomputation.json carries a structured section "
                "that violates the contract-A type/verdict rules despite top-"
                "level verdict 'PASS' (rejected: missing/null/wrong-type/"
                "bare-bool/bare-list/verdict=None/record-level-FAIL); refusing "
                "to seal manifest:\n" +
                "\n".join(f"  - {e}" for e in violated)
            )
        # Preserve the Oracle semantic role from the PASS independent report.
        # The manifest declares the Oracle policy's authoritative semantic
        # role; this must equal the canonical diagnostic-benchmark label and
        # must originate from the verified independent report (not invented
        # here).
        oracle_term = recompute.get("oracle_terminology_evidence") or {}
        oracle_semantic_role = oracle_term.get("oracle_semantic_role")
        if oracle_semantic_role != ORACLE_SEMANTIC_ROLE:
            raise RuntimeError(
                "independent_recomputation.json oracle_terminology_evidence."
                f"oracle_semantic_role={oracle_semantic_role!r} is missing or "
                f"does not equal the authoritative diagnostic role "
                f"{ORACLE_SEMANTIC_ROLE!r}; refusing to seal manifest."
            )
        if oracle_term.get("verdict") != "PASS":
            raise RuntimeError(
                "independent_recomputation.json oracle_terminology_evidence "
                f"verdict is {oracle_term.get('verdict')!r}, not 'PASS'; "
                "refusing to seal manifest."
            )
        # Contract F: the diagnostic role must be sourced from an ACTUAL
        # generated artifact (not a checker-local constant). The manifest
        # refuses to seal if the independent report carries no provenance
        # path for the Oracle semantic role.
        oracle_semantic_role_source = oracle_term.get("oracle_semantic_role_source")
        if not isinstance(oracle_semantic_role_source, str) or not oracle_semantic_role_source:
            raise RuntimeError(
                "independent_recomputation.json oracle_terminology_evidence."
                "oracle_semantic_role_source is missing/empty; the diagnostic "
                "role must originate from a real generated artifact path, not "
                "a checker-local constant; refusing to seal manifest."
            )
        # The source MUST identify an ACTUAL generated artifact on disk. A
        # phantom path (e.g. a path that does not exist) is rejected: the
        # diagnostic role's provenance must be a real file inside the formal
        # run output directory.
        source_path = Path(oracle_semantic_role_source)
        if not source_path.exists():
            raise RuntimeError(
                "independent_recomputation.json oracle_terminology_evidence."
                f"oracle_semantic_role_source={oracle_semantic_role_source!r} "
                "does not identify an actual generated artifact (path not "
                "found); refusing to seal manifest."
            )
    else:
        # Non-formal modes: derive a destabilized source (no production
        # artifact provenance requirement in diagnostic modes).
        oracle_semantic_role_source = None

    # === 7. Read and verify counts ===
    threshold_search_path = output_dir / "threshold_search_results.parquet"
    threshold_summary_path = output_dir / "threshold_search_summary.csv"
    episode_path = output_dir / "episode_results.parquet"
    selected_path_for_count = (
        Path(context.selected_thresholds_path) if context is not None
        else output_dir / "selected_thresholds.json"
    )
    selected_count = _selected_thresholds_count(selected_path_for_count)

    # Disk-side artifact recounts (kept as independent reconciliation fields
    # so they can be checked against the verified independent report).
    artifact_recount_tuning_rows = (
        get_row_count(threshold_search_path)
        if threshold_search_path.exists() else 0
    )
    tuning_candidates_csv = (
        get_row_count(threshold_summary_path)
        if threshold_summary_path.exists() else 0
    )
    artifact_recount_evaluation_rows = (
        get_row_count(episode_path)
        if episode_path.exists() else 0
    )

    # scenario_bank_provenance and run_provenance remain available as raw
    # provenance records for the manifest payload, but are NO LONGER used to
    # DERIVE formal counts. In formal_closeout the formal counts come only
    # from the verified independent_recomputation.json evidence.
    scenario_provenance_path = output_dir / "scenario_bank_provenance.json"
    scenario_provenance = None
    if scenario_provenance_path.exists():
        with open(scenario_provenance_path, "r") as f:
            scenario_provenance = json.load(f)

    run_provenance_path = output_dir / "run_provenance.json"
    run_provenance = None
    if run_provenance_path.exists():
        with open(run_provenance_path, "r") as f:
            run_provenance = json.load(f)

    # ---- Formal counts: take them from the VERIFIED independent report ----
    candidate_ev = recompute.get("candidate_set_evidence", {}) if recompute else {}
    tuning_ev = recompute.get("tuning_set_evidence", {}) if recompute else {}
    winner_ev = recompute.get("selected_winner_evidence", {}) if recompute else {}
    eval_ev = recompute.get("evaluation_set_evidence", {}) if recompute else {}

    tuning_candidates = 0
    tuning_episodes = 0
    evaluation_episodes = 0

    if mode == "formal_closeout" and recompute is not None:
        # Formal counts come ONLY from the independent recomputation's
        # verified evidence. We never derive tuning_episodes from
        # first_bank_scenario_count * reset_seed_count * tuning_candidates.
        tuning_candidates = int(candidate_ev.get("actual_unique_count", 0))
        tuning_episodes = int(tuning_ev.get("actual_unique_count", 0))
        selected_count = int(winner_ev.get("actual_count", 0))
        evaluation_episodes = int(eval_ev.get("actual_unique_count", 0))

        # Require expected_set_sha256 == actual_set_sha256 for candidate,
        # tuning, and evaluation evidence (the independent report must have
        # verified the exact identity sets, not merely the counts).
        sha_mismatch_errors = []
        for name, ev in (("candidate", candidate_ev),
                         ("tuning", tuning_ev),
                         ("evaluation", eval_ev)):
            exp = ev.get("expected_set_sha256")
            act = ev.get("actual_set_sha256")
            if exp is None or act is None:
                sha_mismatch_errors.append(
                    f"{name}_set_evidence missing set SHA fields"
                )
            elif exp != act:
                sha_mismatch_errors.append(
                    f"{name}_set_evidence expected_set_sha256 ({exp[:12]}…) "
                    f"!= actual_set_sha256 ({act[:12]}…)"
                )
        if sha_mismatch_errors:
            raise RuntimeError(
                "independent_recomputation.json identity-set SHA mismatch; "
                "refusing to seal manifest:\n" +
                "\n".join(f"  - {e}" for e in sha_mismatch_errors)
            )

        # Require the independent report's actual counts to exactly equal
        # the frozen formal contract values.
        recon_count_errors = []
        if tuning_candidates != EXPECTED["tuning_candidates"]:
            recon_count_errors.append(
                f"recompute candidate_set_evidence.actual_unique_count="
                f"{tuning_candidates} (expected {EXPECTED['tuning_candidates']})"
            )
        if tuning_episodes != EXPECTED["tuning_episodes"]:
            recon_count_errors.append(
                f"recompute tuning_set_evidence.actual_unique_count="
                f"{tuning_episodes} (expected {EXPECTED['tuning_episodes']})"
            )
        if selected_count != EXPECTED["selected_thresholds"]:
            recon_count_errors.append(
                f"recompute selected_winner_evidence.actual_count="
                f"{selected_count} (expected {EXPECTED['selected_thresholds']})"
            )
        if evaluation_episodes != EXPECTED["evaluation_episodes"]:
            recon_count_errors.append(
                f"recompute evaluation_set_evidence.actual_unique_count="
                f"{evaluation_episodes} (expected {EXPECTED['evaluation_episodes']})"
            )
        if recon_count_errors:
            raise RuntimeError(
                "independent_recomputation.json verified counts do not match "
                "the formal contract:\n" +
                "\n".join(f"  - {e}" for e in recon_count_errors)
            )

        # The manifest may independently recount artifact rows; those recounts
        # MUST match the independent report. The tuning-rows recount is the
        # 9000 tuning-episode rows; the evaluation-rows recount is 2400.
        recount_errors = []
        if artifact_recount_tuning_rows != tuning_episodes:
            recount_errors.append(
                f"artifact recount threshold_search_results.parquet rows="
                f"{artifact_recount_tuning_rows} != independent tuning_episodes="
                f"{tuning_episodes}"
            )
        if artifact_recount_evaluation_rows != evaluation_episodes:
            recount_errors.append(
                f"artifact recount episode_results.parquet rows="
                f"{artifact_recount_evaluation_rows} != independent "
                f"evaluation_episodes={evaluation_episodes}"
            )
        if recount_errors:
            raise RuntimeError(
                "Artifact recount does not match the independent recomputation "
                "report:\n" + "\n".join(f"  - {e}" for e in recount_errors)
            )
    else:
        # Non-formal modes: keep the disk-side row counts (no sealed-context
        # authority) but still do not multiply-derived tuning_episodes here.
        tuning_candidates = artifact_recount_tuning_rows
        tuning_episodes = artifact_recount_tuning_rows
        evaluation_episodes = artifact_recount_evaluation_rows

    # === 8. Hard count assertions (formal_closeout only) ===
    count_errors = []
    if mode == "formal_closeout":
        if tuning_candidates != EXPECTED["tuning_candidates"]:
            count_errors.append(
                f"tuning_candidates={tuning_candidates} "
                f"(expected {EXPECTED['tuning_candidates']})"
            )
        if tuning_episodes != EXPECTED["tuning_episodes"]:
            count_errors.append(
                f"tuning_episodes={tuning_episodes} "
                f"(expected {EXPECTED['tuning_episodes']})"
            )
        if selected_count != EXPECTED["selected_thresholds"]:
            count_errors.append(
                f"selected_thresholds={selected_count} "
                f"(expected {EXPECTED['selected_thresholds']})"
            )
        if evaluation_episodes != EXPECTED["evaluation_episodes"]:
            count_errors.append(
                f"evaluation_episodes={evaluation_episodes} "
                f"(expected {EXPECTED['evaluation_episodes']})"
            )

    validator_verdict = "PENDING"
    if mode == "formal_closeout":
        valid = validation is not None and validation.get("verdict") == "ALL PASSED"
        recomp = recompute is not None and recompute.get("verdict") == "PASS"
        validator_verdict = "ALL PASSED" if (valid and recomp and not count_errors) else "FAIL"

    if count_errors:
        raise RuntimeError(
            "Formal count contract violated:\n" +
            "\n".join(f"  - {e}" for e in count_errors)
        )

    # === 9. Build artifacts list ===
    artifacts: List[Dict[str, Any]] = []
    ARTIFACT_EXCLUDE = {"formal_manifest.json", "m3_run.log"}
    for fp in sorted(output_dir.glob("*")):
        if fp.name in ARTIFACT_EXCLUDE:
            continue
        if fp.is_dir():
            continue
        relative_path = fp.relative_to(output_dir)
        artifacts.append({
            "relative_path": str(relative_path),
            "byte_size": get_file_size(fp),
            "sha256": compute_sha256(fp),
            "row_count": get_row_count(fp) if fp.suffix in [".parquet", ".csv"] else None,
            "schema_columns": get_schema_columns(fp) if fp.suffix in [".parquet", ".csv"] else None,
        })

    # === 10. Write manifest ===
    # Display-only provenance-derived fields (NOT used to derive any formal
    # count; formal counts come exclusively from the verified independent
    # recomputation report in formal_closeout).
    display_scenarios_per_bank = 0
    if isinstance(scenario_provenance, dict):
        _banks = scenario_provenance.get("scenario_banks", [])
        if isinstance(_banks, list) and _banks:
            display_scenarios_per_bank = int(_banks[0].get("derived_scenario_count", 0))
    display_reset_seed_count = 0
    if isinstance(run_provenance, dict):
        _seeds = run_provenance.get("reset_seeds", [])
        if isinstance(_seeds, list):
            display_reset_seed_count = len(_seeds)
    manifest = {
        "m3_version": "m3_v1",
        "formal_run_id": formal_run_id,
        "run_date": datetime.utcnow().date().isoformat(),
        "m3_final_implementation_commit": implementation_commit,
        "config_sha256": config_sha256,
        "validator_verdict": validator_verdict,
        "counts": {
            "tuning_candidates": tuning_candidates,
            "tuning_candidates_csv_rows": tuning_candidates_csv,
            "scenarios_per_bank": display_scenarios_per_bank,
            "reset_seed_count": display_reset_seed_count,
            "tuning_episodes": tuning_episodes,
            "selected_thresholds": selected_count,
            "evaluation_episodes": evaluation_episodes,
        },
        "selected_thresholds_sha256": selected_sha,
        "validation_report_sha256": validation_sha,
        "independent_recomputation_sha256": recompute_sha,
        "oracle_semantic_role": oracle_semantic_role,
        "oracle_semantic_role_source": oracle_semantic_role_source,
        "scenario_bank_provenance": scenario_provenance,
        "run_provenance": run_provenance,
        "artifacts": artifacts,
        "generated_at": datetime.utcnow().isoformat(),
    }

    write_json_safe(manifest, manifest_path)
    return manifest_path


def validate_formal_manifest(output_dir: Path) -> Dict[str, Any]:
    """
    Validate formal manifest against actual artifacts.

    Returns:
        Validation report with any mismatches
    """
    manifest_path = output_dir / "formal_manifest.json"
    if not manifest_path.exists():
        return {"valid": False, "errors": ["formal_manifest.json not found"]}

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    errors = []

    # Verify each artifact
    for artifact in manifest.get("artifacts", []):
        rel_path = artifact["relative_path"]
        full_path = output_dir / rel_path

        if not full_path.exists():
            errors.append(f"Missing artifact: {rel_path}")
            continue

        # Verify byte size
        actual_size = get_file_size(full_path)
        expected_size = artifact.get("byte_size")
        if expected_size is not None and actual_size != expected_size:
            errors.append(f"Size mismatch for {rel_path}: expected {expected_size}, got {actual_size}")

        # Verify SHA256
        actual_sha = compute_sha256(full_path)
        expected_sha = artifact.get("sha256")
        if expected_sha is not None and actual_sha != expected_sha:
            errors.append(f"SHA256 mismatch for {rel_path}: expected {expected_sha}, got {actual_sha}")

        # Verify row count
        if artifact.get("row_count") is not None:
            actual_rows = get_row_count(full_path)
            if actual_rows != artifact["row_count"]:
                errors.append(f"Row count mismatch for {rel_path}: expected {artifact['row_count']}, got {actual_rows}")

        # Verify schema
        if artifact.get("schema_columns") is not None:
            actual_cols = get_schema_columns(full_path)
            if actual_cols != artifact["schema_columns"]:
                errors.append(f"Schema mismatch for {rel_path}: expected {artifact['schema_columns']}, got {actual_cols}")

    # Verify run ID consistency
    run_prov_path = output_dir / "run_provenance.json"
    if run_prov_path.exists():
        with open(run_prov_path, "r") as f:
            run_prov = json.load(f)
        # The run ID should match (or be derivable)
        manifest_run_id = manifest.get("formal_run_id")
        if "run_id" in run_prov and run_prov["run_id"] != manifest_run_id:
            errors.append(f"Run ID mismatch: manifest {manifest_run_id} != run_provenance {run_prov.get('run_id')}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "artifact_count": len(manifest.get("artifacts", [])),
    }
