#!/usr/bin/env python3
"""
M5 eight-cell cross-regime smoke driver.

Runs ONE bounded 6000-step training per (K, regime) cell:

    2 K * 4 cost regimes = 8 cells.

Per-cell configuration:
  - seed 6521
  - max_steps=6000
  - predictor_train (training) / rl_validation (validation)
  - schema 6 (default of current branch)
  - unique output root:  results/m5_smoke_v1/<run_id>/
  - prediction cache (V2 default)
  - no rl_test, no reuse of old pilot/formal directories

This produces compatibility/provenance smoke evidence.  It is NOT a formal
performance comparison.  No superiority claim is made.

Per cell the driver records: the exact invocation command, the owning PID,
the process exit code, the expected and observed SHA256 values, and a
manifest/latest/best identity-chain audit.

Modes:
  default            : run all 8 cells and audit.
  --audit-only       : audit existing run directories without retraining.
  --output-json PATH : write the smoke inventory JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# Add repository root to path for src. imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.preflight import validate_row_asset_contract
from src.training.resolver import (
    resolve_command_to_effective,
    derive_prediction_cache_provenance,
    ExplicitBankError,
)
from src.training.ddqn_config_identity import compute_resolved_config_identity


PYTHON = sys.executable
REPO = Path(__file__).resolve().parent.parent


# Frozen 4 regimes + suffix mapping (same constants as the formal generator).
REGIMES = [
    "failure-light-no-waste",
    "failure-heavy-no-waste",
    "failure-light-waste-aware",
    "failure-heavy-waste-aware",
]
REGIME_SUFFIX = {
    "failure-light-no-waste": "light",
    "failure-heavy-no-waste": "heavy",
    "failure-light-waste-aware": "light_waste",
    "failure-heavy-waste-aware": "heavy_waste",
}

# Template configs (K=1 and K=2).  These point at the BASELINE
# (light-no-waste) banks; the per-cell command overrides the bank via the
# explicit --training-scenario-bank / --validation-scenario-bank flags so the
# effective config matches the smoke row exactly, the same path the
# formal matrix uses.  NO temporary per-cell configs are materialised.
CONFIG_TEMPLATE_K1 = "configs/agents/ddqn_v1_k1.json"
CONFIG_TEMPLATE_K2 = "configs/agents/ddqn_v1.json"

# Unique, non-colliding smoke output root.  Overridable for isolation/tests.
DEFAULT_OUTPUT_ROOT = "results/m5_smoke_v1"
OUTPUT_ROOT_ENV = os.environ.get("M5_SMOKE_OUTPUT_ROOT")
SEED = 6521
MAX_STEPS = 6000

# Prediction-cache manifest path.
PC_MANIFEST_PATH = "data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json"
PC_CACHE_DIR = "data/processed/fd001/v2/06_PREDICTIONS/"


# ---------------------------------------------------------------------------
# Cell helpers.
# ---------------------------------------------------------------------------

def cell_run_id(k: int, regime: str) -> str:
    suf = REGIME_SUFFIX[regime]
    return f"m5_smoke_k{k}__{suf}_seed{SEED}"


def cell_bank_paths(k: int, regime: str) -> tuple[str, str]:
    suf = REGIME_SUFFIX[regime]
    return (
        f"configs/scenarios/m5_pilot_k{k}__{suf}.json",
        f"configs/scenarios/m5_validation_k{k}__{suf}.json",
    )


def cell_config_template(k: int) -> str:
    return CONFIG_TEMPLATE_K1 if k == 1 else CONFIG_TEMPLATE_K2


def build_smoke_command(k: int, regime: str, output_root: Path,
                        run_id: str | None = None) -> list[str]:
    """Build the per-cell smoke training command using the SAME production
    CLI binding as a formal row, differing ONLY in: max_steps=6000, the
    smoke output root, the smoke run_id, and seed fixed to 6521.

    No temporary per-cell configs.  The regime-specific banks are bound via
    explicit --training-scenario-bank / --validation-scenario-bank flags and
    the splits via --split / --validation-split, exactly like a formal row.
    """
    train_bank, val_bank = cell_bank_paths(k, regime)
    rid = run_id if run_id is not None else cell_run_id(k, regime)
    return [
        PYTHON, "scripts/train_ddqn.py",
        "--config", cell_config_template(k),
        "--k-capacity", str(k),
        "--cost-regime", regime,
        "--training-seed", str(SEED),
        "--max-steps", str(MAX_STEPS),
        "--output-dir", str(output_root),
        "--run-id", rid,
        "--training-scenario-bank", train_bank,
        "--validation-scenario-bank", val_bank,
        "--split", "predictor_train",
        "--validation-split", "rl_validation",
    ]


def expected_git_head() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_repo(path: str | Path) -> str:
    """SHA256 of a repo-relative path, resolved against REPO root.

    Used so the driver computes bank / pc-manifest hashes correctly regardless
    of the caller's current working directory.
    """
    return sha256_of(REPO / str(path))


# ---------------------------------------------------------------------------
# Audit.
# ---------------------------------------------------------------------------

def _read_checkpoint_identity(ckpt_path: Path) -> str | None:
    """Recompute the resolved_config_identity from a checkpoint's stored config.

    The schema-6 checkpoint stores the full effective config dict under the
    ``config`` key (including split, validation_split, maintenance_capacity,
    cost_regime_id, all agent/training hyperparameters, and num_actions).
    ``resolved_config_identity`` is the canonical hash of that dict.  We
    recompute it here from the checkpoint's stored config so the smoke
    audit verifies the manifest/latest/best identity chain WITHOUT trusting
    any stored hash field.  Returns the 64-char hex identity, or None if the
    checkpoint is unreadable or its config is incomplete.
    """
    try:
        import torch  # local import; only needed when auditing real runs
    except Exception:
        return None
    if not ckpt_path.exists():
        return None
    try:
        data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    cfg = data.get("config")
    if not isinstance(cfg, dict):
        return None
    try:
        from src.training.ddqn_config_identity import compute_resolved_config_identity
    except Exception:
        return None
    ident_dict = dict(cfg)
    if "num_actions" not in ident_dict:
        k = ident_dict.get("maintenance_capacity")
        if k == 1:
            ident_dict["num_actions"] = 6
        elif k == 2:
            ident_dict["num_actions"] = 16
        else:
            return None
    try:
        return compute_resolved_config_identity(ident_dict)
    except Exception:
        return None


def _read_checkpoint_metadata(ckpt_path: Path) -> dict | None:
    """Read the raw ``metadata`` dict from a schema-6 checkpoint file.

    Returns None if the checkpoint is unreadable or has no metadata.  The
    schema-6 metadata stores the full prediction-cache provenance fields
    (prediction_cache_manifest_sha256, prediction_cache_declared_cache_hash,
    prediction_cache_predictor_checkpoint_hash,
    prediction_cache_feature_schema_hash, prediction_cache_normalizer_hash,
    prediction_cache_split, prediction_cache_schema_version) which the smoke
    audit must verify against recomputed disk values — NOT via a
    path-only resolved_config_identity or a conditional acceptance pattern.
    """
    try:
        import torch
    except Exception:
        return None
    if not ckpt_path.exists():
        return None
    try:
        data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    md = data.get("metadata")
    if not isinstance(md, dict):
        return None
    return md


# Schema-6 prediction-cache provenance fields that the checkpoint metadata
# MUST store (and that the smoke audit recomputes from disk and compares).
PC_PROVENANCE_FIELDS = (
    "prediction_cache_manifest_path",
    "prediction_cache_manifest_sha256",
    "prediction_cache_declared_cache_hash",
    "prediction_cache_predictor_checkpoint_hash",
    "prediction_cache_feature_schema_hash",
    "prediction_cache_normalizer_hash",
    "prediction_cache_split",
    "prediction_cache_schema_version",
)


def audit_prediction_cache_provenance(ckpt_path: Path,
                                     expected_validation_split: str) -> dict:
    """Schema-6 prediction-cache provenance audit (no softening).

    Recomputes the manifest bytes hash + the canonical cached-identity dict
    from the on-disk prediction-cache manifest, then compares them against
    the checkpoint metadata's stored provenance fields.  Every required field
    must be present in the checkpoint metadata AND equal to its recomputed
    disk-side value; any missing or mismatching field fails the audit —
    there is no conditional acceptance.  A path-only ``resolved_config_identity``
    does not pin file contents; only the manifest-sha plus the cached
    identity dict do, and those are recomputed here from the on-disk
    manifest.

    ``prediction_cache_split`` is set by the schema-6 trainer to the resolved
    ``validation_split`` at save time (see src/agents/ddqn/checkpoint.py).  The
    disk-side comparison value for that scalar is therefore the resolved
    command's validation split (no manifest can recover a single split).

    Returns a dict with keys: ok, errors, notes, provenance.
    """
    errors: list[str] = []
    notes: list[str] = []
    md = _read_checkpoint_metadata(ckpt_path)
    if md is None:
        return {"ok": False,
                "errors": [f"checkpoint metadata unreadable: {ckpt_path}"],
                "notes": notes, "provenance": {}}

    # Recompute the disk-side provenance from the manifest file the
    # checkpoint claims to have used.
    ckpt_manifest_path = md.get("prediction_cache_manifest_path")
    if not isinstance(ckpt_manifest_path, str) or not ckpt_manifest_path:
        return {"ok": False,
                "errors": ["checkpoint missing prediction_cache_manifest_path"],
                "notes": notes, "provenance": {}}

    manifest_disk_path = (REPO / ckpt_manifest_path
                         if not Path(ckpt_manifest_path).is_absolute()
                         else Path(ckpt_manifest_path))
    if not manifest_disk_path.exists():
        return {"ok": False,
                "errors": [f"prediction-cache manifest not on disk: {manifest_disk_path}"],
                "notes": notes, "provenance": {}}

    from src.training.prediction_cache_identity import get_prediction_cache_identity
    disk_ident = get_prediction_cache_identity(manifest_disk_path)
    # Disk-side scalar split is the resolved command's validation split.
    disk_ident["prediction_cache_split"] = expected_validation_split
    # Disk-side manifest path is the canonical repo-relative string.
    disk_ident["prediction_cache_manifest_path"] = ckpt_manifest_path

    provenance: dict[str, Any] = {}
    for f in PC_PROVENANCE_FIELDS:
        ckpt_val = md.get(f)
        disk_val = disk_ident.get(f)
        provenance[f] = {"checkpoint": ckpt_val, "disk": disk_val}
        notes.append(f"{f}: ckpt={ckpt_val!r} disk={disk_val!r}")
        # Fail closed: each schema-6 provenance field must be present and equal.
        if ckpt_val is None:
            errors.append(f"checkpoint metadata {f} is null (schema-6 requires it)")
        elif disk_val is None:
            errors.append(f"disk-side {f} is null (schema-6 requires it)")
        elif str(ckpt_val) != str(disk_val):
            errors.append(
                f"prediction-cache provenance mismatch {f}: "
                f"checkpoint={ckpt_val!r} disk={disk_val!r}"
            )
    return {"ok": len(errors) == 0, "errors": errors,
            "notes": notes, "provenance": provenance}


def audit_run(run_dir: Path, expected: dict) -> dict:
    """Audit a completed smoke run against the expected field set.

    Returns a dict with keys: ok, errors, notes, manifest, identity_chain.
    """
    errors: list[str] = []
    notes: list[str] = []

    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return {"ok": False, "errors": ["run_manifest.json missing"],
                "notes": notes, "identity_chain": {}}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("status") != "COMPLETE":
        errors.append(f"manifest status != COMPLETE: {manifest.get('status')!r}")
    if int(manifest.get("final_global_step", -1)) != MAX_STEPS:
        errors.append(
            f"final_global_step {manifest.get('final_global_step')} != {MAX_STEPS}"
        )
    if manifest.get("maintenance_capacity") != expected["k"]:
        errors.append(
            f"maintenance_capacity {manifest.get('maintenance_capacity')} != {expected['k']}"
        )
    if manifest.get("cost_regime_id") != expected["regime"]:
        errors.append(
            f"cost_regime_id {manifest.get('cost_regime_id')} != {expected['regime']!r}"
        )
    if manifest.get("checkpoint_schema_version") != expected["schema"]:
        errors.append(
            f"checkpoint_schema_version {manifest.get('checkpoint_schema_version')} "
            f"!= {expected['schema']}"
        )
    if manifest.get("training_split") != "predictor_train":
        errors.append(f"training_split {manifest.get('training_split')!r} != 'predictor_train'")
    if manifest.get("validation_split") != "rl_validation":
        errors.append(f"validation_split {manifest.get('validation_split')!r} != 'rl_validation'")
    if manifest.get("git_commit") != expected["git_head"]:
        errors.append(
            f"git_commit {manifest.get('git_commit')!r} != {expected['git_head']!r}"
        )
    if manifest.get("training_split") == "rl_test" or manifest.get("validation_split") == "rl_test":
        errors.append("manifest splits include rl_test (FORBIDDEN)")

    # Bank hashes - the manifest identity must equal the on-disk bank SHA256
    # (canonical banks: content hash == raw file sha).
    train_sha = sha256_repo(expected["train_bank"])
    val_sha = sha256_repo(expected["val_bank"])
    if manifest.get("training_scenario_bank_identity") != train_sha:
        errors.append(
            f"training_scenario_bank_identity {manifest.get('training_scenario_bank_identity')}"
            f" != sha({expected['train_bank']})={train_sha}"
        )
    if manifest.get("validation_scenario_bank_identity") != val_sha:
        errors.append(
            f"validation_scenario_bank_identity {manifest.get('validation_scenario_bank_identity')}"
            f" != sha({expected['val_bank']})={val_sha}"
        )

    # Prediction-cache provenance (M5 provenance).  The schema-6 *checkpoint*
    # metadata stores the full prediction-cache provenance fields, fully
    # populated by ``save_checkpoint`` via ``get_prediction_cache_identity``.
    # We audit them directly and fail-closed: every required field must be
    # present in the checkpoint metadata AND equal to recomputed disk values,
    # with NO conditional acceptance.  A path-only resolved_config_identity
    # does NOT pin file contents; only the manifest-sha + cached identity dict
    # do, and those are recomputed here from the on-disk prediction-cache
    # manifest file.
    pc_sha = sha256_repo(PC_MANIFEST_PATH)
    notes.append(f"on-disk prediction_cache_manifest_sha256={pc_sha}")
    cp_latest = run_dir / "checkpoint_latest.pt"
    cp_best = run_dir / "checkpoint_best.pt"
    # Derive the prediction_cache_split scalar the schema-6 trainer wrote at
    # save time (== resolved validation_split) from the resolved effective
    # config, NOT a hard-coded literal.  The audit compares this against the
    # checkpoint's stored prediction_cache_split and fails closed on mismatch.
    expected_pc_split = expected.get("validation_split", "rl_validation")
    pc_audit = audit_prediction_cache_provenance(cp_latest,
                                                 expected_validation_split=expected_pc_split)
    notes.extend(pc_audit["notes"])
    if not pc_audit["ok"]:
        errors.extend(pc_audit["errors"])
        # If the provenance is not closed this is a hard failure per the
        # M5 provenance contract: FAIL - PREDICTION-CACHE PROVENANCE NOT CLOSED.

    # resolved_config_identity: present, 64-char hex, and agree across
    # manifest, checkpoint_latest, checkpoint_best, and resolved_config.json.
    rci_manifest = manifest.get("resolved_config_identity")
    if not isinstance(rci_manifest, str) or len(rci_manifest) != 64:
        errors.append(f"resolved_config_identity not a 64-char hex: {rci_manifest!r}")
        try:
            if isinstance(rci_manifest, str):
                int(rci_manifest, 16)
        except ValueError:
            errors.append("resolved_config_identity is not valid hex")
        rci_manifest = None
    else:
        notes.append(f"resolved_config_identity={rci_manifest}")

    # Checkpoints present.
    if not cp_latest.exists():
        errors.append("checkpoint_latest.pt missing")
    if manifest.get("validation_performed") is True and not cp_best.exists():
        errors.append("checkpoint_best.pt missing (validation_performed=True)")

    # Identity-chain audit across manifest / resolved_config.json / checkpoints.
    identity_chain: dict[str, Any] = {"manifest": rci_manifest}
    rc_path = run_dir / "resolved_config.json"
    if rc_path.exists():
        try:
            rc_data = json.loads(rc_path.read_text(encoding="utf-8"))
        except Exception:
            rc_data = None
        if isinstance(rc_data, dict):
            # resolved_config.json stores the effective config dict (NOT a
            # resolved_config_identity field).  Recompute the identity from
            # it and compare to the manifest identity for a real cross-check.
            try:
                from src.training.ddqn_config_identity import compute_resolved_config_identity
                rc_ident_dict = dict(rc_data)
                if "num_actions" not in rc_ident_dict:
                    k = rc_ident_dict.get("maintenance_capacity")
                    rc_ident_dict["num_actions"] = 6 if k == 1 else (16 if k == 2 else None)
                rc_id = compute_resolved_config_identity(rc_ident_dict) if rc_ident_dict.get("num_actions") else None
            except Exception:
                rc_id = None
            identity_chain["resolved_config_json"] = rc_id
            if rc_id is not None and rci_manifest is not None and rc_id != rci_manifest:
                errors.append(
                    f"resolved_config.json identity {rc_id} != manifest {rci_manifest}"
                )
        else:
            identity_chain["resolved_config_json"] = None
    rci_latest = _read_checkpoint_identity(cp_latest)
    identity_chain["checkpoint_latest"] = rci_latest
    if rci_latest is not None and rci_manifest is not None and rci_latest != rci_manifest:
        errors.append(
            f"checkpoint_latest identity {rci_latest} != manifest {rci_manifest}"
        )
    rci_best = _read_checkpoint_identity(cp_best)
    identity_chain["checkpoint_best"] = rci_best
    if rci_best is not None and rci_manifest is not None and rci_best != rci_manifest:
        errors.append(
            f"checkpoint_best identity {rci_best} != manifest {rci_manifest}"
        )
    notes.append(f"identity_chain={identity_chain}")

    return {"ok": len(errors) == 0, "errors": errors, "notes": notes,
            "manifest": manifest, "identity_chain": identity_chain}


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def run_one(k: int, regime: str, output_root: Path,
            expected_head: str) -> dict:
    """Run one cell; return a structured record.

    The cell's training command uses the SAME production CLI binding as a
    formal row (explicit --training-scenario-bank / --validation-scenario-bank
    / --split / --validation-split flags), differing ONLY in max_steps=6000,
    the smoke output root, the smoke run_id, and seed fixed to 6521.  NO
    temporary per-cell config files are materialised.
    """
    train_bank, val_bank = cell_bank_paths(k, regime)
    run_id = cell_run_id(k, regime)
    run_dir = output_root / run_id
    expected = {
        "k": k,
        "regime": regime,
        "schema": 6,
        "train_bank": train_bank,
        "val_bank": val_bank,
        "git_head": expected_head,
        "pc_manifest_sha256": sha256_repo(PC_MANIFEST_PATH),
        # validation_split is populated below from the resolved effective
        # config so the prediction_cache_split provenance scalar is derived
        # from the command, not hard-coded.
        "validation_split": "rl_validation",
    }

    record: dict[str, Any] = {
        "run_id": run_id,
        "k": k,
        "regime": regime,
        "train_bank": train_bank,
        "val_bank": val_bank,
        "expected": expected,
        "train_bank_sha256": sha256_repo(train_bank),
        "val_bank_sha256": sha256_repo(val_bank),
        "pc_manifest_sha256": expected["pc_manifest_sha256"],
        "output_dir": str(run_dir),
    }

    # Run preflight before training to confirm fail-closed behavior on the
    # regime-specific banks.  Resolve bank paths against REPO so the preflight
    # succeeds regardless of the caller's CWD.
    pre = validate_row_asset_contract(
        training_scenario_bank_path=str((REPO / train_bank).resolve()),
        validation_scenario_bank_path=str((REPO / val_bank).resolve()),
        cost_regime_id=regime,
        maintenance_capacity=k,
        prediction_cache_path=str((REPO / PC_CACHE_DIR).resolve()),
    )
    record["preflight_ok"] = pre.ok
    if not pre.ok:
        record["preflight_errors"] = pre.errors
        return record

    # Build the command via the shared smoke builder (explicit bank/split
    # flags) and audit its effective config through the SAME production
    # resolver used by the formal matrix.  No temp configs.
    cmd = build_smoke_command(k, regime, output_root, run_id=run_id)
    record["exact_command"] = " ".join(cmd)
    record["cwd"] = str(REPO)
    record["owner_pid"] = os.getpid()

    # Resolve the exact command through the shared resolver and record the
    # effective config it determines, so smoke and formal launch provenance
    # are demonstrably identical at the resolved-config level.  The resolver
    # enforces the mandatory explicit-bank gate (no bypass) and surfaces the
    # derived prediction_cache_split scalar from the resolved effective
    # config's validation_split.
    try:
        eff = resolve_command_to_effective(cmd[1:], cwd=REPO)
        # Derive the prediction-cache provenance dict (manifest-sha + cached
        # identity + prediction_cache_split scalar) from the resolved
        # effective config, fail-closed if any required provenance field
        # cannot be computed.
        try:
            pc_provenance = derive_prediction_cache_provenance(
                eff, manifest_path=str(REPO / PC_MANIFEST_PATH))
            pc_provenance["prediction_cache_split"] = eff.validation_split
        except Exception as exc:
            record["provenance_error"] = str(exc)
            # Fail closed: required prediction-cache provenance cannot be
            # computed -- do not proceed to training.
            record["ok"] = False
            record["errors"] = [f"prediction-cache provenance not closed: {exc}"]
            return record
        record["effective_config"] = {
            "training_scenario_bank_path": eff.training_scenario_bank_path,
            "validation_scenario_bank_path": eff.validation_scenario_bank_path,
            "cost_regime_id": eff.cost_regime_id,
            "maintenance_capacity": eff.maintenance_capacity,
            "split": eff.split,
            "validation_split": eff.validation_split,
            "max_steps": eff.max_steps,
            "num_actions": eff.num_actions,
            "resolved_config_identity": compute_resolved_config_identity(eff.effective_identity_dict),
            "prediction_cache_split": eff.validation_split,
            "prediction_cache_manifest_sha256": pc_provenance.get(
                "prediction_cache_manifest_sha256"),
        }
        # Thread the derived scalar split into expected so the post-run
        # audit compares the checkpoint's stored prediction_cache_split
        # against the resolved command's validation_split, not a literal.
        expected["validation_split"] = eff.validation_split
    except ExplicitBankError as exc:
        record["effective_config_error"] = str(exc)
        record["ok"] = False
        record["errors"] = [str(exc)]
        return record
    except Exception as exc:  # pragma: no cover - defensive
        record["effective_config_error"] = str(exc)
        record["ok"] = False
        record["errors"] = [str(exc)]
        return record

    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    record["returncode"] = proc.returncode
    if proc.returncode != 0:
        record["stderr_tail"] = proc.stderr[-1500:]
        record["stdout_tail"] = proc.stdout[-1500:]
        return record

    audit = audit_run(run_dir, expected)
    record.update(audit)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="M5 eight-cell cross-regime smoke driver")
    parser.add_argument("--audit-only", action="store_true",
                        help="Audit existing run directories without retraining.")
    parser.add_argument("--output-json", type=str,
                        default="results/m5_smoke_v1/smoke_inventory.json",
                        help="Path to write the smoke inventory JSON.")
    args = parser.parse_args()

    output_root = Path(OUTPUT_ROOT_ENV or DEFAULT_OUTPUT_ROOT)
    if not args.audit_only:
        output_root.mkdir(parents=True, exist_ok=True)

    head = expected_git_head()
    print(f"FROZEN HEAD: {head}")
    print(f"OUTPUT_ROOT: {output_root}")
    print(f"PYTHON:      {PYTHON}")
    print("=" * 72)

    all_records: list[dict] = []
    all_ok = True

    for k in (1, 2):
        for regime in REGIMES:
            run_id = cell_run_id(k, regime)
            print(f"\n--- CELL k={k} regime={regime} run_id={run_id}")
            if args.audit_only:
                run_dir = output_root / run_id
                train_bank, val_bank = cell_bank_paths(k, regime)
                # The smoke command fixes --validation-split rl_validation, so
                # the derived prediction_cache_split scalar for audit is
                # rl_validation.  Stated explicitly (not via a hard-coded
                # literal elsewhere) to keep the audit self-describing.
                audit = audit_run(run_dir, {
                    "k": k, "regime": regime, "schema": 6,
                    "train_bank": train_bank, "val_bank": val_bank,
                    "git_head": head,
                    "pc_manifest_sha256": sha256_repo(PC_MANIFEST_PATH),
                    "validation_split": "rl_validation",
                })
                rec = {
                    "run_id": run_id, "k": k, "regime": regime,
                    "train_bank": train_bank, "val_bank": val_bank,
                    "train_bank_sha256": sha256_repo(train_bank),
                    "val_bank_sha256": sha256_repo(val_bank),
                    "pc_manifest_sha256": sha256_repo(PC_MANIFEST_PATH),
                    "output_dir": str(run_dir),
                    "audit": audit,
                }
            else:
                rec = run_one(k, regime, output_root, head)
            all_records.append(rec)
            ok = rec.get("ok", rec.get("audit", {}).get("ok", False))
            all_ok = all_ok and ok
            if not ok:
                print(f"  FAIL; errors={rec.get('errors') or rec.get('audit',{}).get('errors')}")
            else:
                print(f"  OK")
    print("\n" + "=" * 72)

    # Build a SHA256SUMS for the per-cell artifacts inside the smoke root.
    shaums_lines: list[str] = []
    for rec in all_records:
        run_dir = Path(rec["output_dir"])
        for art in ("run_manifest.json", "checkpoint_latest.pt", "checkpoint_best.pt"):
            ap = run_dir / art
            if ap.exists():
                shaums_lines.append(f"{sha256_of(ap)}  {ap.relative_to(output_root).as_posix()}")
    shaums_path = output_root / "SHA256SUMS.txt"
    shaums_path.write_text("\n".join(shaums_lines) + ("\n" if shaums_lines else ""),
                           encoding="utf-8")
    shaums_sha = sha256_of(shaums_path) if shaums_path.exists() else ""

    inventory = {
        "frozen_git_head": head,
        "output_root": str(output_root),
        "max_steps_per_cell": MAX_STEPS,
        "seed_per_cell": SEED,
        "training_split": "predictor_train",
        "validation_split": "rl_validation",
        "schema_version": 6,
        "no_rl_test": True,
        "cells": all_records,
        "sha256sums_path": str(shaums_path) if shaums_path.exists() else None,
        "sha256sums_sha256": shaums_sha or None,
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(inventory, indent=2, sort_keys=True, default=str) + "\n",
                        encoding="utf-8")
    print(f"Inventory written: {out_json}")
    print(f"SHA256SUMS:        {shaums_path}  sha={shaums_sha}")
    print(f"OK cells: {sum(1 for r in all_records if r.get('ok', r.get('audit',{}).get('ok', False)))}/8")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
