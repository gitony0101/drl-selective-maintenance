#!/usr/bin/env python3
"""
Run M4 Scientific Validation candidate evaluations.

This script evaluates all 6 candidates (1 hard_window_v1 comparator + 5 logistic temperatures)
on the frozen scientific validation banks.

Each candidate runs in its own directory with complete artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Candidate configurations - FROZEN
CANDIDATES = [
    {
        "candidate_id": "hard_window_v1",
        "risk_model_id": "hard_window_v1",
        "risk_temperature": None,
        "matrix_role": "primary_contract_policy",
    },
    {
        "candidate_id": "logistic_T1",
        "risk_model_id": "logistic_window_v1",
        "risk_temperature": 1.0,
        "matrix_role": "scientific_validation_candidate",
    },
    {
        "candidate_id": "logistic_T2",
        "risk_model_id": "logistic_window_v1",
        "risk_temperature": 2.0,
        "matrix_role": "scientific_validation_candidate",
    },
    {
        "candidate_id": "logistic_T5",
        "risk_model_id": "logistic_window_v1",
        "risk_temperature": 5.0,
        "matrix_role": "scientific_validation_candidate",
    },
    {
        "candidate_id": "logistic_T10",
        "risk_model_id": "logistic_window_v1",
        "risk_temperature": 10.0,
        "matrix_role": "scientific_validation_candidate",
    },
    {
        "candidate_id": "logistic_T20",
        "risk_model_id": "logistic_window_v1",
        "risk_temperature": 20.0,
        "matrix_role": "scientific_validation_candidate",
    },
]

# Frozen parameters
DELTA_CYCLES = 5
TIE_TOLERANCE = 1e-9
FLEET_SIZE = 5
EPISODE_HORIZON = 100
RUL_SCALE = 125.0
AGE_SCALE_CYCLES = 341

# Cost regimes and K values
COST_REGIMES = [
    "failure-heavy-no-waste",
    "failure-heavy-waste-aware",
    "failure-light-no-waste",
    "failure-light-waste-aware",
]
K_VALUES = [1, 2]
SPLITS = ["predictor_train", "rl_validation"]

# Protocol references
PROTOCOL_VERSION = "m4_scientific_validation_v1"
PROTOCOL_FILE = "docs/MILESTONE_4_SCIENTIFIC_VALIDATION_PROTOCOL.md"


def get_git_head() -> str:
    """Get current Git HEAD."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    return result.stdout.strip()


def get_git_status() -> str:
    """Get git status --short."""
    result = subprocess.run(
        ["git", "status", "--short"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    return result.stdout.strip()


def compute_protocol_hash() -> str:
    """Compute SHA256 of protocol file."""
    protocol_path = Path(__file__).parent.parent / PROTOCOL_FILE
    if not protocol_path.exists():
        return "PROTOCOL_NOT_FOUND"
    sha256 = hashlib.sha256()
    with open(protocol_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_bank_manifest(bank_dir: Path) -> Dict[str, Any]:
    """Load bank manifest."""
    manifest_path = bank_dir / "SCIENTIFIC_VALIDATION_BANK_MANIFEST.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return {}


def get_bank_hashes(bank_dir: Path) -> Dict[str, str]:
    """Compute SHA256 of all bank files."""
    hashes = {}
    for bank_file in sorted(bank_dir.glob("*.json")):
        sha256 = hashlib.sha256()
        with open(bank_file, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        hashes[bank_file.name] = sha256.hexdigest()
    return hashes


def get_prediction_cache_hash(cache_path: Path) -> str:
    """Get prediction cache SHA256."""
    sha256 = hashlib.sha256()
    with open(cache_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_action_table_hashes() -> Dict[str, str]:
    """Get action table content hashes."""
    from envs.action_table import ACTION_TABLE_N5_K1, ACTION_TABLE_N5_K2
    import json

    def hash_table(table):
        return hashlib.sha256(
            json.dumps([list(a) for a in table], sort_keys=True, separators=(',', ':')).encode()
        ).hexdigest()

    return {
        "action_table_K1_content_hash": hash_table(ACTION_TABLE_N5_K1),
        "action_table_K2_content_hash": hash_table(ACTION_TABLE_N5_K2),
    }


def build_candidate_config(candidate: Dict[str, Any], bank_dir: Path, cache_path: Path) -> Dict[str, Any]:
    """Build complete scientific config for a candidate."""
    bank_manifest = get_bank_manifest(bank_dir)
    bank_hashes = get_bank_hashes(bank_dir)
    cache_hash = get_prediction_cache_hash(cache_path)
    action_hashes = get_action_table_hashes()
    protocol_hash = compute_protocol_hash()

    config = {
        "schema_version": "m4_scientific_validation_v1",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_file_sha256": protocol_hash,
        "candidate_identity": candidate["candidate_id"],
        "risk_model": candidate["risk_model_id"],
        "risk_temperature": candidate["risk_temperature"],
        "tie_tolerance": TIE_TOLERANCE,
        "delta_cycles": DELTA_CYCLES,
        "environment_version": "m2_v1",
        "horizon": EPISODE_HORIZON,
        "fleet_size": FLEET_SIZE,
        "rul_scale": RUL_SCALE,
        "age_scale_cycles": AGE_SCALE_CYCLES,
        "k_values": K_VALUES,
        "cost_regimes": COST_REGIMES,
        "splits": SPLITS,
        "ordered_seeds": list(range(6601, 6621)),
        "scenario_bank_manifest": bank_manifest,
        "scenario_bank_sha256_values": bank_hashes,
        "prediction_cache_sha256": cache_hash,
        "action_table_K1_identity": "ACTION_TABLE_N5_K1_M2_V1",
        "action_table_K1_num_actions": 6,
        "action_table_K2_identity": "ACTION_TABLE_N5_K2_M2_V1",
        "action_table_K2_num_actions": 16,
        **action_hashes,
        "pairing_basis": "stable_pair_id_from_unit_cycles",
        "selection_metric_version": "macro_avg_normalized_paired_cost_diff_v1",
        "bootstrap_seed": 652104,
        "bootstrap_resamples": 10000,
    }
    return config


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute SHA256 of scientific config (excluding runtime metadata)."""
    # Create a copy without runtime fields
    scientific_fields = {k: v for k, v in config.items()
                         if k not in ["output_dir", "overwrite", "timestamp", "git_commit",
                                     "command_line", "log_path", "temporary_path", "matrix_role"]}
    canonical = json.dumps(scientific_fields, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode()).hexdigest()


def run_candidate(candidate: Dict[str, Any], output_root: Path, bank_dir: Path, cache_path: Path,
                  resume: bool = False) -> Dict[str, Any]:
    """Run a single candidate evaluation."""

    candidate_id = candidate["candidate_id"]
    candidate_dir = output_root / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)

    # Build config
    config = build_candidate_config(candidate, bank_dir, cache_path)
    config["matrix_role"] = candidate["matrix_role"]
    config["output_dir"] = str(candidate_dir)
    config["overwrite"] = not resume
    config["timestamp"] = datetime.now().isoformat()
    config["git_commit"] = get_git_head()
    config["command_line"] = " ".join(sys.argv)
    config["log_path"] = str(candidate_dir / f"{candidate_id}.log")

    config_hash = compute_config_hash(config)
    config["config_hash"] = config_hash

    # Check if already completed with matching config
    status_file = candidate_dir / "candidate_status.json"
    if resume and status_file.exists():
        with open(status_file) as f:
            existing_status = json.load(f)
        if existing_status.get("config_hash") == config_hash and existing_status.get("status") == "completed":
            print(f"  {candidate_id}: Already completed with matching config (resuming)")
            return {"candidate_id": candidate_id, "status": "completed", "config_hash": config_hash,
                    "candidate_dir": str(candidate_dir)}
        elif existing_status.get("config_hash") != config_hash:
            raise RuntimeError(f"Config hash mismatch for {candidate_id}: "
                             f"existing={existing_status.get('config_hash')}, new={config_hash}")

    # Write config
    config_path = candidate_dir / "resolved_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Update status to running
    status = {
        "candidate_id": candidate_id,
        "status": "running",
        "config_hash": config_hash,
        "git_head": get_git_head(),
        "protocol_hash": compute_protocol_hash(),
        "started_at": datetime.now().isoformat(),
    }
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)

    print(f"  {candidate_id}: Running evaluation...")

    # Run the production smoke script for this candidate
    # We need to call run_m4_production_smoke.py with appropriate arguments
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "run_m4_production_smoke.py"),
        "--output-dir", str(candidate_dir),
        "--matrix-role", candidate["matrix_role"],
        "--risk-model", candidate["risk_model_id"],
        "--bank-dir", str(bank_dir),
        "--cache-path", str(cache_path),
    ]
    if candidate["risk_temperature"] is not None:
        cmd.extend(["--risk-temperature", str(candidate["risk_temperature"])])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            raise RuntimeError(f"Exit code {result.returncode}: {result.stderr}")

        # Update status to completed
        status["status"] = "completed"
        status["completed_at"] = datetime.now().isoformat()
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)

        print(f"  {candidate_id}: ✓ Completed")
        return {"candidate_id": candidate_id, "status": "completed", "config_hash": config_hash,
                "candidate_dir": str(candidate_dir)}

    except subprocess.TimeoutExpired:
        status["status"] = "failed"
        status["error"] = "Timeout after 1 hour"
        status["failed_at"] = datetime.now().isoformat()
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
        print(f"  {candidate_id}: ✗ Timeout")
        return {"candidate_id": candidate_id, "status": "failed", "config_hash": config_hash,
                "error": "Timeout"}

    except Exception as e:
        status["status"] = "failed"
        status["error"] = str(e)
        status["failed_at"] = datetime.now().isoformat()
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
        print(f"  {candidate_id}: ✗ Failed: {e}")
        return {"candidate_id": candidate_id, "status": "failed", "config_hash": config_hash,
                "error": str(e)}


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="run_m4_scientific_validation",
        description="Run M4 scientific validation candidate evaluations",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Output root directory (default: results/milestone4/scientific_validation_v1/)",
    )

    parser.add_argument(
        "--bank-dir",
        type=str,
        default=None,
        help="Scientific validation bank directory (default: data/scenario_banks/m4_scientific_validation/)",
    )

    parser.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help="Prediction cache path (default: data/processed/fd001/v2/06_PREDICTIONS/fd001_prediction_cache_v2.parquet)",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing candidate runs (requires matching config hash)",
    )

    parser.add_argument(
        "--candidates",
        type=str,
        nargs="+",
        default=None,
        help="Specific candidates to run (default: all 6)",
    )

    args = parser.parse_args()

    # Verify clean git state
    git_status = get_git_status()
    if git_status:
        print(f"ERROR: Working tree not clean:\n{git_status}", file=sys.stderr)
        return 1

    git_head = get_git_head()
    print(f"Git HEAD: {git_head}")
    print(f"Protocol: {PROTOCOL_VERSION}")

    # Resolve paths
    repo_root = Path(__file__).parent.parent
    if args.output_root:
        output_root = Path(args.output_root)
    else:
        output_root = repo_root / "results" / "milestone4" / "scientific_validation_v1"

    if args.bank_dir:
        bank_dir = Path(args.bank_dir)
    else:
        bank_dir = repo_root / "data" / "scenario_banks" / "m4_scientific_validation"

    if args.cache_path:
        cache_path = Path(args.cache_path)
    else:
        cache_path = repo_root / "data" / "processed" / "fd001" / "v2" / "06_PREDICTIONS" / "fd001_prediction_cache_v2.parquet"

    # Verify inputs exist
    if not bank_dir.exists():
        print(f"ERROR: Bank directory not found: {bank_dir}", file=sys.stderr)
        return 1

    if not cache_path.exists():
        print(f"ERROR: Prediction cache not found: {cache_path}", file=sys.stderr)
        return 1

    manifest_path = bank_dir / "SCIENTIFIC_VALIDATION_BANK_MANIFEST.json"
    if not manifest_path.exists():
        print(f"ERROR: Bank manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    with open(manifest_path) as f:
        manifest = json.load(f)

    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        print(f"ERROR: Bank manifest protocol mismatch: {manifest.get('protocol_version')} != {PROTOCOL_VERSION}", file=sys.stderr)
        return 1

    print(f"Bank dir: {bank_dir}")
    print(f"Cache: {cache_path}")
    print(f"Output root: {output_root}")

    # Filter candidates if specified
    candidates_to_run = CANDIDATES
    if args.candidates:
        candidates_to_run = [c for c in CANDIDATES if c["candidate_id"] in args.candidates]
        if len(candidates_to_run) != len(args.candidates):
            found = {c["candidate_id"] for c in candidates_to_run}
            missing = set(args.candidates) - found
            print(f"ERROR: Unknown candidates: {missing}", file=sys.stderr)
            return 1

    print(f"\nRunning {len(candidates_to_run)} candidates:")
    for c in candidates_to_run:
        print(f"  - {c['candidate_id']} ({c['risk_model_id']}, T={c['risk_temperature']})")

    # Run each candidate
    results = []
    for candidate in candidates_to_run:
        result = run_candidate(candidate, output_root, bank_dir, cache_path, resume=args.resume)
        results.append(result)

    # Create parent manifest
    parent_manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_file_sha256": compute_protocol_hash(),
        "protocol_commit": "TO_BE_FILLED",  # Will be filled after protocol commit
        "git_head": git_head,
        "scientific_validation_head": git_head,
        "bank_dir": str(bank_dir),
        "bank_manifest": manifest,
        "prediction_cache_sha256": get_prediction_cache_hash(cache_path),
        "action_table_hashes": get_action_table_hashes(),
        "candidates": results,
        "candidate_index": {r["candidate_id"]: r for r in results},
        "started_at": datetime.now().isoformat(),
    }

    parent_manifest_path = output_root / "parent_run_manifest.json"
    with open(parent_manifest_path, 'w') as f:
        json.dump(parent_manifest, f, indent=2)

    # Also save candidate_index.json
    candidate_index_path = output_root / "candidate_index.json"
    with open(candidate_index_path, 'w') as f:
        json.dump({r["candidate_id"]: r for r in results}, f, indent=2)

    print(f"\nParent manifest: {parent_manifest_path}")
    print(f"Candidate index: {candidate_index_path}")

    # Check all completed
    failed = [r for r in results if r["status"] != "completed"]
    if failed:
        print(f"\n✗ {len(failed)} candidate(s) failed:")
        for r in failed:
            print(f"  - {r['candidate_id']}: {r.get('error', 'Unknown error')}")
        return 1

    print(f"\n✓ All {len(results)} candidates completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())