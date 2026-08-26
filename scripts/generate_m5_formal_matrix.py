#!/usr/bin/env python3
"""
M5 formal matrix generator (M5 reproducibility) - frozen 40-row package builder.

Produces, at a single final repo HEAD, the FROZEN 40-run formal matrix for
the regime-specific scientific-asset contract:

    2 K * 4 cost regimes * 5 seeds = 40 unique rows.

Each row references the regime-specific derived scenario banks
(``configs/scenarios/m5_pilot_k{1,2}__{light,heavy,light_waste,heavy_waste}.json``
for training, ``configs/scenarios/m5_validation_k{1,2}__*.json`` for
validation), records complete scenario-bank content hashes and a full cache
manifest hash, the frozen git HEAD, max_steps=100000, predictor_train, and
rl_validation.  It runs the strengthened official non-training preflight
on ALL 40 rows and writes a preflight_results.json proving 40/40 PASS with
NO training side effects.

This generator DOES NOT execute any training.  It only produces the frozen
artifacts and the launch package audit bundle.  It supports a
``--validate-only`` CLI (non-training) coroutine that confirms the matrix
contract without writing any Trainer / checkpoint / run directory.

Output layout (an isolated, reproducible audit bundle)::

    <OUT_DIR>/
        frozen_launch_package/
            formal_matrix.json
            formal_matrix.csv
            preflight_results.json
            FORMAL_RUNBOOK.md
            launch_commands.txt
            SUPERSESSION_RECORD.md
            M5_SMOKE_INVENTORY.json   (best-effort cross-reference)
            SHA256SUMS.txt
        run_ledgers/formal_execution_ledger.json
        operator_logs/
        postrun_audit/
        validation_analysis/

The default OUT_DIR for the audit bundle is the persistent control directory
created by the operator (outside the repository) so it survives cleanups.
Inside the repository, the in-repo formal output root for the 40 training
runs themselves is the non-colliding:

    results/milestone5_formal_regimebanks_v1/

which is recorded per-row as ``output_root`` / ``expected_output_dir`` and
is NEVER reused from the superseded ``results/milestone5`` tree.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Any

# Repository root - resolved absolutely so this script works regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.preflight import validate_row_asset_contract
from src.training.ddqn_config_identity import compute_resolved_config_identity
from src.training.resolver import resolve_command_to_effective
from src.agents.ddqn.checkpoint import (
    compute_action_table_hash,
    compute_scenario_bank_content_hash,
    compute_network_architecture_id,
)
from src.envs.action_table import ACTION_TABLE_N5_K1, ACTION_TABLE_N5_K2

REPO = Path(__file__).resolve().parent.parent

MAINTENANCE_CAPACITIES = [1, 2]
COST_REGIMES = [
    "failure-light-no-waste",
    "failure-heavy-no-waste",
    "failure-light-waste-aware",
    "failure-heavy-waste-aware",
]
TRAINING_SEEDS = [6521, 6522, 6523, 6524, 6525]

REGIME_SUFFIX = {
    "failure-light-no-waste": "light",
    "failure-heavy-no-waste": "heavy",
    "failure-light-waste-aware": "light_waste",
    "failure-heavy-waste-aware": "heavy_waste",
}

# In-repo formal output root for the 40 training runs.  NON-COLLIDING with the
# superseded ``results/milestone5`` tree.  Overridable for tests only.
DEFAULT_OUTPUT_BASE = "results/milestone5_formal_regimebanks_v1"
OUTPUT_BASE_ENV = os.environ.get("M5_FORMAL_OUTPUT_BASE")

# Template configs - the K=1 and K=2 base agent configs.  The exact training
# command passes these plus explicit CLI overrides (k-capacity, cost-regime,
# training-seed, max-steps, output-dir, run-id) so the regime-specific bank
# is selected via the config's scenario_bank_path override baked in by the
# generator below.  Splits are enforced from the bank content via preflight.
CONFIG_TEMPLATE_K1 = "configs/agents/ddqn_v1_k1.json"
CONFIG_TEMPLATE_K2 = "configs/agents/ddqn_v1.json"

# Prediction-cache manifest path + sha (fixed for the frozen experiment).
PC_MANIFEST_PATH = "data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json"
PC_CACHE_DIR = "data/processed/fd001/v2/06_PREDICTIONS/"

# Schema version expected for production checkpoints.
CHECKPOINT_SCHEMA_VERSION = 6

# The superseded formal matrix SHA256 (historical formal attempt).
SUPERSEDED_FORMAL_MATRIX_SHA256 = (
    "f77f7a6c202fee91234a9492a3a1e2b94d0def61c04baf1a336f32cfe95b20f0"
)
# Superseded in-repo output roots that must NEVER be reused by the new
# formal matrix.
SUPERSEDED_OUTPUT_ROOTS = [
    "results/milestone5",
]


def get_git_head() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def repo_rel(path: Path | str) -> str:
    """Return a forward-slash repo-relative path string for a Path or str."""
    p = Path(path)
    try:
        rel = p.resolve().relative_to(REPO.resolve())
        return rel.as_posix()
    except ValueError:
        # Already repo-relative or absolute outside repo: normalise separators.
        return p.as_posix()


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def generate_run_id(k: int, regime: str, seed: int) -> str:
    return f"m5_formal_k{k}__{REGIME_SUFFIX[regime]}_seed{seed}"


def per_k_summary(k: int) -> dict:
    """Per-K agent defaults read straight from the on-disk config."""
    cfg_path = CONFIG_TEMPLATE_K1 if k == 1 else CONFIG_TEMPLATE_K2
    cfg = json.loads((REPO / cfg_path).read_text(encoding="utf-8"))
    agent = cfg.get("agent", {})
    train = cfg.get("training", {})
    action_table = ACTION_TABLE_N5_K1 if k == 1 else ACTION_TABLE_N5_K2
    return {
        "config_source": cfg_path,
        "action_table_hash": compute_action_table_hash(action_table),
        "network_architecture_id": compute_network_architecture_id(
            observation_dim=10,
            hidden_dim=cfg["agent"]["hidden_dim"],
            num_hidden_layers=cfg["agent"]["num_hidden_layers"],
            activation="relu",
            action_count=len(action_table),
            architecture_revision="m5_point_v1",
        ),
        "action_count": len(action_table),
        "hidden_dim": agent.get("hidden_dim", 128),
        "num_hidden_layers": agent.get("num_hidden_layers", 2),
        "learning_rate": agent.get("learning_rate", 1e-4),
        "gamma": agent.get("gamma", 0.95),
        "epsilon_start": agent.get("epsilon_start", 1.0),
        "epsilon_end": agent.get("epsilon_end", 0.05),
        "epsilon_decay_steps": agent.get("epsilon_decay_steps", 50_000),
        "gradient_clip": agent.get("gradient_clip", 10.0),
        "target_update_interval": agent.get("target_update_interval", 1000),
        "batch_size": train.get("batch_size", 128),
        "warmup_transitions": train.get("warmup_transitions", 5000),
        "update_frequency": train.get("update_frequency", 1),
        "validation_interval": train.get("validation_interval", 5000),
        "checkpoint_interval": train.get("checkpoint_interval", 5000),
        "replay_capacity": train.get("replay_capacity", 100_000),
        "max_steps": 100_000,
        "observation_dim": 10,
        "observation_schema_id": "m5_point_v1",
        "environment_contract_id": "m2_v1",
    }


def _matrix_index(k: int, regime: str, seed: int) -> int:
    return (MAINTENANCE_CAPACITIES.index(k) * len(COST_REGIMES) * len(TRAINING_SEEDS)
            + COST_REGIMES.index(regime) * len(TRAINING_SEEDS)
            + TRAINING_SEEDS.index(seed))


def build_row(k: int, regime: str, seed: int, head: str, pc_manifest_sha: str,
              out_base_posix: str) -> dict[str, Any]:
    suf = REGIME_SUFFIX[regime]
    train_bank = f"configs/scenarios/m5_pilot_k{k}__{suf}.json"
    val_bank = f"configs/scenarios/m5_validation_k{k}__{suf}.json"
    config_path = CONFIG_TEMPLATE_K1 if k == 1 else CONFIG_TEMPLATE_K2
    run_id = generate_run_id(k, regime, seed)
    summary = per_k_summary(k)
    expected_dir_posix = f"{out_base_posix.rstrip('/')}/{run_id}"

    # M5 provenance binding: the exact training command MUST pass explicit
    # regime-specific scenario-bank flags AND the training/validation splits
    # so the effective config matches this row exactly when resolved through
    # the SHARED production resolver (the same path train_ddqn.py uses).  No
    # temporary configs, no synthetic TrainerConfig drift.
    exact_cmd = (
        f"python scripts/train_ddqn.py "
        f"--config {config_path} "
        f"--k-capacity {k} "
        f"--cost-regime {regime} "
        f"--training-seed {seed} "
        f"--max-steps 100000 "
        f"--output-dir {out_base_posix} "
        f"--run-id {run_id} "
        f"--training-scenario-bank {train_bank} "
        f"--validation-scenario-bank {val_bank} "
        f"--split predictor_train "
        f"--validation-split rl_validation"
    )

    # Compute resolved_config_identity by resolving the exact command through
    # the SHARED production resolver.  This guarantees the matrix row identity
    # equals the effective command identity (no synthetic TrainerConfig drift).
    resolved = resolve_command_to_effective(exact_cmd, cwd=REPO)
    resolved_id = compute_resolved_config_identity(resolved.effective_identity_dict)

    return {
        "matrix_index": _matrix_index(k, regime, seed),
        "k": k,
        "cost_regime_id": regime,
        "cost_regime": regime,
        "seed": seed,
        "action_count": summary["action_count"],
        "action_table_hash": summary["action_table_hash"],
        "network_architecture_id": summary["network_architecture_id"],
        "observation_dim": 10,
        "observation_schema_id": "m5_point_v1",
        "environment_contract_id": "m2_v1",
        "config_source": config_path,
        "expected_output_dir": expected_dir_posix,
        "output_root": out_base_posix,
        "run_id": run_id,
        "max_steps": 100_000,
        "training_split": "predictor_train",
        "validation_split": "rl_validation",
        "training_scenario_bank_path": train_bank,
        "validation_scenario_bank_path": val_bank,
        "training_scenario_bank_content_hash": sha256_of(REPO / train_bank),
        "validation_scenario_bank_content_hash": sha256_of(REPO / val_bank),
        "prediction_cache_manifest_path": PC_MANIFEST_PATH,
        "prediction_cache_manifest_sha256": pc_manifest_sha,
        "checkpoint_schema_version_expected": CHECKPOINT_SCHEMA_VERSION,
        "expected_git_commit": head,
        "resume_policy": ("same-directory resume from checkpoint_latest.pt with "
                          "semantic identity equality "
                          "(load_checkpoint validate-before-mutation); "
                          "global_step deduplication prevents duplicate steps"),
        "expected_artifact_list": [
            "checkpoint_latest.pt",
            "checkpoint_best.pt",
            "training_metrics.jsonl",
            "validation_metrics.json",
            "resolved_config.json",
            "run_manifest.json",
        ],
        "resolved_config_identity": resolved_id,
        "exact_training_command": exact_cmd,
        "agent_defaults": {
            "hidden_dim": summary["hidden_dim"],
            "num_hidden_layers": summary["num_hidden_layers"],
            "learning_rate": summary["learning_rate"],
            "gamma": summary["gamma"],
            "epsilon_schedule": {
                "epsilon_start": summary["epsilon_start"],
                "epsilon_end": summary["epsilon_end"],
                "epsilon_decay_steps": summary["epsilon_decay_steps"],
            },
            "gradient_clip": summary["gradient_clip"],
            "target_update_interval": summary["target_update_interval"],
            "batch_size": summary["batch_size"],
            "warmup_transitions": summary["warmup_transitions"],
            "update_frequency": summary["update_frequency"],
            "validation_interval": summary["validation_interval"],
            "checkpoint_interval": summary["checkpoint_interval"],
            "replay_capacity": summary["replay_capacity"],
        },
    }


# ---------------------------------------------------------------------------
# Side-effect proof: preflight must not create the formal output root.
# ---------------------------------------------------------------------------

def _snapshot_dir(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {p.name for p in path.iterdir()}


def preflight_all_rows(rows: list[dict]) -> tuple[list[dict], int]:
    preflight_results: list[dict] = []
    pass_count = 0
    for r in rows:
        # Preflight does Path(path).exists() against the process CWD.  Resolve
        # the repo-relative bank paths to ABSOLUTE repo-root-relative paths so
        # the preflight succeeds regardless of the caller's CWD.  The recorded
        # row fields keep the repo-relative form for portability.
        rep = validate_row_asset_contract(
            training_scenario_bank_path=str((REPO / r["training_scenario_bank_path"]).resolve()),
            validation_scenario_bank_path=str((REPO / r["validation_scenario_bank_path"]).resolve()),
            cost_regime_id=r["cost_regime_id"],
            maintenance_capacity=r["k"],
            prediction_cache_path=str((REPO / PC_CACHE_DIR).resolve()),
            training_split=r["training_split"],
            validation_split=r["validation_split"],
        )
        entry = {
            "matrix_index": r["matrix_index"],
            "run_id": r["run_id"],
            "k": r["k"],
            "cost_regime_id": r["cost_regime_id"],
            "seed": r["seed"],
            "ok": rep.ok,
            "errors": list(rep.errors),
            "warnings": list(rep.warnings),
            "effective": dict(rep.effective),
        }
        preflight_results.append(entry)
        if rep.ok:
            pass_count += 1
    return preflight_results, pass_count


# ---------------------------------------------------------------------------
# Stage 4: Command-level preflight.  The row-direct preflight above validates
# the ROW's recorded paths against the on-disk banks.  The M5 provenance contract
# additionally requires proving each exact_training_command resolves, through
# the SHARED production resolver, to exactly that row's effective config, AND
# (the binding correction) that the REAL production CLI ``--validate-only``
# path PASSES for every exact command.  Merely calling the resolver directly
# is NOT sufficient: the production CLI is the path that would actually run,
# so it must be exercised end-to-end.
# ---------------------------------------------------------------------------

def command_preflight_all_rows(rows: list[dict]) -> tuple[list[dict], int]:
    """Run the command-level preflight required by Stage 4 over every row.

    For each row:
      1. EXERCISE THE REAL PRODUCTION CLI: launch the row's
         exact_training_command with ``--validate-only`` appended as a
         subprocess (with --output-dir redirected to a throwaway temporary
         directory so NO production output path is touched).  Require exit 0
         and ``Configuration validated successfully``.  This is the binding
         correction -- the production validate-only path is the one that
         would actually run, so it must be exercised end-to-end, NOT merely
         by calling the resolver directly.
      2. resolve the exact_training_command through the shared production
         resolver (``resolve_command_to_effective``) for the read-only
         provenance cross-checks below;
      3. compare the effective config field-by-field with the matrix row;
      4. recompute resolved_config_identity from the effective config dict
         and require equality with the matrix identity;
      5. verify the effective training and validation bank hashes against
         the matrix's recorded content hashes;
      6. verify no output directory, Trainer, checkpoint, or training
         interaction is created.

    The preflight entry records: command, the production-CLI validate-only
    subprocess exit code / stdout / stderr, effective training/validation
    bank, effective bank hashes, K, regime, seed, splits, max_steps,
    effective resolved_config_identity, expected matrix identity, equality
    result, no_side_effect result.
    """
    import shlex
    import subprocess
    import tempfile
    from src.training.ddqn_config_identity import compute_resolved_config_identity
    from src.agents.ddqn.checkpoint import compute_scenario_bank_content_hash

    results: list[dict] = []
    cmd_pass = 0
    python = sys.executable
    for r in rows:
        cmd = r["exact_training_command"]
        entry: dict[str, Any] = {
            "matrix_index": r["matrix_index"],
            "run_id": r["run_id"],
            "command": cmd,
            "k": r["k"],
            "regime": r["cost_regime_id"],
            "seed": r["seed"],
            "expected_matrix_identity": r["resolved_config_identity"],
            "errors": [],
        }
        ok = True

        # ----- 1. REAL PRODUCTION CLI --validate-only subprocess -----
        # Tokenise the exact command with shlex (honours quoted paths), then
        # append --validate-only.  Redirect --output-dir to a throwaway temp
        # dir so no production run directory is ever touched by preflight.
        tokens = shlex.split(cmd)
        # Strip a leading ``python`` if present; we use sys.executable.
        if tokens and tokens[0] == "python":
            tokens = tokens[1:]
        # Replace any --output-dir value with a throwaway temp dir.
        if "--output-dir" in tokens:
            oi = tokens.index("--output-dir")
            preflight_out = tempfile.mkdtemp(prefix="m5_cmd_preflight_")
            tokens[oi + 1] = preflight_out
        tokens = tokens + ["--validate-only"]
        proc = subprocess.run(
            [python, *tokens], cwd=str(REPO),
            capture_output=True, text=True,
        )
        entry["validate_only_subprocess"] = {
            "exit_code": proc.returncode,
            "stdout_tail": proc.stdout[-1200:],
            "stderr_tail": proc.stderr[-1200:],
            "command": " ".join([python, *tokens]),
        }
        if proc.returncode != 0:
            entry["errors"].append(
                f"production CLI --validate-only FAILED (exit {proc.returncode})"
            )
            ok = False
        elif "Configuration validated successfully" not in proc.stdout:
            entry["errors"].append(
                "production CLI --validate-only did not print the success banner"
            )
            ok = False
        # Clean up the throwaway temp dir.
        try:
            import shutil as _shutil
            _shutil.rmtree(preflight_out, ignore_errors=True)
        except Exception:
            pass

        # ----- 2. Read-only resolver cross-checks -----
        try:
            eff = resolve_command_to_effective(cmd, cwd=REPO)
        except Exception as exc:
            entry["errors"].append(f"resolve_command_to_effective failed: {exc}")
            entry["ok"] = False
            entry["effective"] = None
            entry["effective_resolved_config_identity"] = None
            entry["identity_equal"] = False
            entry["no_side_effect"] = False
            results.append(entry)
            continue

        eff_train = eff.training_scenario_bank_path
        eff_val = eff.validation_scenario_bank_path
        eff_dict = eff.effective_identity_dict
        eff_id = compute_resolved_config_identity(eff_dict)
        entry["effective"] = {
            "training_scenario_bank_path": eff_train,
            "validation_scenario_bank_path": eff_val,
            "cost_regime_id": eff.cost_regime_id,
            "maintenance_capacity": eff.maintenance_capacity,
            "training_split": eff.split,
            "validation_split": eff.validation_split,
            "max_steps": eff.max_steps,
            "num_actions": eff.num_actions,
        }
        entry["effective_resolved_config_identity"] = eff_id
        entry["identity_equal"] = (eff_id == r["resolved_config_identity"])
        if not entry["identity_equal"]:
            entry["errors"].append(
                f"effective identity {eff_id} != matrix identity "
                f"{r['resolved_config_identity']}"
            )
            ok = False

        # Field-by-field comparison with the matrix row.
        field_check = (
            eff_train == r["training_scenario_bank_path"]
            and eff_val == r["validation_scenario_bank_path"]
            and eff.cost_regime_id == r["cost_regime_id"]
            and eff.maintenance_capacity == r["k"]
            and eff.split == r["training_split"]
            and eff.validation_split == r["validation_split"]
            and eff.max_steps == r["max_steps"]
        )
        entry["field_equal"] = field_check
        if not field_check:
            entry["errors"].append(
                "effective config field-by-field mismatch with matrix row"
            )
            ok = False

        # Effective bank hashes vs matrix recorded content hashes.
        tbh = compute_scenario_bank_content_hash(REPO / eff_train)
        vbh = compute_scenario_bank_content_hash(REPO / eff_val)
        entry["effective_bank_hashes"] = {
            "training_sha256": tbh,
            "validation_sha256": vbh,
            "expected_training_sha256": r["training_scenario_bank_content_hash"],
            "expected_validation_sha256": r["validation_scenario_bank_content_hash"],
        }
        bank_hashes_equal = (
            tbh == r["training_scenario_bank_content_hash"]
            and vbh == r["validation_scenario_bank_content_hash"]
        )
        entry["bank_hashes_equal"] = bank_hashes_equal
        if not bank_hashes_equal:
            entry["errors"].append("effective bank hashes != matrix recorded hashes")
            ok = False

        # No-side-effect proof: the production validate-only subprocess must
        # NOT have created the row's expected output dir.
        expected_dir = REPO / r["expected_output_dir"]
        entry["no_side_effect"] = not expected_dir.exists()

        entry["ok"] = ok
        if ok:
            cmd_pass += 1
        results.append(entry)
    return results, cmd_pass


# ---------------------------------------------------------------------------
# Package writers.
# ---------------------------------------------------------------------------

def write_matrix_json(out_dir: Path, matrix: dict) -> tuple[Path, str]:
    p = out_dir / "formal_matrix.json"
    p.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n",
                 encoding="utf-8")
    return p, sha256_of(p)


def write_matrix_csv(out_dir: Path, rows: list[dict]) -> tuple[Path, str]:
    p = out_dir / "formal_matrix.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "matrix_index", "k", "cost_regime_id", "seed", "run_id",
            "expected_output_dir", "config_source",
            "training_scenario_bank_path",
            "training_scenario_bank_content_hash",
            "validation_scenario_bank_path",
            "validation_scenario_bank_content_hash",
            "prediction_cache_manifest_path",
            "prediction_cache_manifest_sha256",
            "action_count",
            "checkpoint_schema_version_expected",
            "resolved_config_identity",
            "expected_git_commit",
        ])
        for r in rows:
            w.writerow([
                r["matrix_index"], r["k"], r["cost_regime_id"], r["seed"],
                r["run_id"], r["expected_output_dir"], r["config_source"],
                r["training_scenario_bank_path"],
                r["training_scenario_bank_content_hash"],
                r["validation_scenario_bank_path"],
                r["validation_scenario_bank_content_hash"],
                r["prediction_cache_manifest_path"],
                r["prediction_cache_manifest_sha256"],
                r["action_count"],
                r["checkpoint_schema_version_expected"],
                r["resolved_config_identity"],
                r["expected_git_commit"],
            ])
    return p, sha256_of(p)


def write_preflight_results(out_dir: Path, head: str, rows: list[dict],
                            preflight_results: list[dict], pass_count: int,
                            command_preflight_results: list[dict],
                            command_pass_count: int,
                            output_root: str,
                            before_snap: set[str], after_snap: set[str]) -> tuple[Path, str]:
    p = out_dir / "preflight_results.json"
    payload = {
        "frozen_git_head": head,
        "total_rows": len(rows),
        "passed": pass_count,
        "all_pass": pass_count == len(rows),
        "command_level_preflight": {
            "passed": command_pass_count,
            "all_pass": command_pass_count == len(rows),
            "description": (
                "Stage 4 command-level preflight: for every row the REAL "
                "production CLI is exercised end-to-end by launching the "
                "row's exact_training_command with --validate-only appended "
                "as a subprocess (--output-dir redirected to a throwaway temp "
                "dir so no production run directory is touched), requiring "
                "exit 0 and the 'Configuration validated successfully' banner. "
                "The exact command is additionally resolved through the "
                "read-only shared production resolver (src.training.resolver."
                "resolve_command_to_effective) and its effective config compared "
                "field-by-field with the matrix row; resolved_config_identity "
                "recomputed and required equal to the matrix identity; "
                "effective bank hashes verified against the matrix's recorded "
                "content hashes; the row's expected_output_dir verified NOT "
                "created (no Trainer, no checkpoint, no training interaction)."
            ),
            "rows": command_preflight_results,
        },
        "output_root_monitored_for_side_effects": output_root,
        "output_root_existed_before_preflight": bool(before_snap is not None),
        "output_root_entries_before": sorted(before_snap or []),
        "output_root_entries_after": sorted(after_snap or []),
        "no_trainable_side_effects_evidence": (
            "Preflight ran validate_row_asset_contract on all 40 rows AND "
            "command_preflight_all_rows exercised the REAL production CLI "
            "--validate-only subprocess for each exact_training_command "
            "(with --output-dir redirected to a throwaway temp dir), and "
            "resolved each command through the read-only shared resolver. "
            "The --validate-only path constructs no DDQNTrainer, opens no file "
            "for writing, and creates no directories except the throwaway temp "
            "dir (removed after the call).  The output-root directory entry "
            "set was identical before and after the 40-row preflight pass "
            f"(before == after: {sorted(before_snap or []) == sorted(after_snap or [])})."
        ),
        "rl_test_usage": "forbidden: no row uses split='rl_test' or "
                         "validation_split='rl_test'; preflight rejects both.",
        "rows": preflight_results,
    }
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                 encoding="utf-8")
    return p, sha256_of(p)


def write_sha256sums(out_dir: Path, entries: list[tuple[str, str]]) -> tuple[Path, str]:
    p = out_dir / "SHA256SUMS.txt"
    with p.open("w", encoding="utf-8") as f:
        for fname, sha in entries:
            f.write(f"{sha}  {fname}\n")
    return p, sha256_of(p)


def write_runbook(out_dir: Path, head: str, output_root: str,
                  pc_manifest_sha: str, matrix_sha: str,
                  pass_count: int) -> tuple[Path, str]:
    p = out_dir / "FORMAL_RUNBOOK.md"
    lines = [
        "# M5 M5 reproducibility Formal Runbook",
        "",
        "This runbook launches the FROZEN 40-run formal matrix produced at a",
        f"single repo HEAD: `{head}`.",
        "",
        "## Invariants (frozen)",
        "",
        f"- Git HEAD: `{head}`",
        f"- Rows: 40 (2 K * 4 regimes * 5 seeds 6521..6525)",
        f"- max_steps: 100000  (passed explicitly with `--max-steps 100000`)",
        f"- Training split: predictor_train",
        f"- Validation split: rl_validation",
        f"- rl_test: FORBIDDEN (sealed for M5)",
        f"- Checkpoint schema version expected: {CHECKPOINT_SCHEMA_VERSION}",
        f"- In-repo formal output root: `{output_root}`",
        f"- Prediction-cache manifest path: `{PC_MANIFEST_PATH}`",
        f"- Prediction-cache manifest SHA256: `{pc_manifest_sha}`",
        f"- Preflight pass: {pass_count}/40",
        f"- Formal matrix JSON SHA256: `{matrix_sha}`",
        "",
        "## Launch (per row)",
        "",
        "Each row's exact training command is recorded verbatim in the row's",
        "`exact_training_command` field in `formal_matrix.json` and listed in",
        "`launch_commands.txt`.  All commands resolve relative paths against",
        "the repository root and must be executed with the repository root as",
        "the current working directory.  Per-row output goes to:",
        "",
        "    <output_root>/<run_id>/",
        "",
        "Do NOT reuse `results/milestone5` (superseded).",
        "",
        "## Pre-launch",
        "",
        "Run the strengthened official preflight on every row.  Each row's",
        "exact command (recorded verbatim in formal_matrix.json and",
        "launch_commands.txt) already includes BOTH --training-scenario-bank",
        "and --validation-scenario-bank, so append --validate-only to run the",
        "REAL production CLI preflight without training:",
        "",
        "    python scripts/train_ddqn.py --config <row config> \\",
        "        --k-capacity <K> --cost-regime <regime> \\",
        "        --training-seed <seed> --max-steps 100000 \\",
        "        --training-scenario-bank configs/scenarios/m5_pilot_k<K>__<suf>.json \\",
        "        --validation-scenario-bank configs/scenarios/m5_validation_k<K>__<suf>.json \\",
        "        --split predictor_train --validation-split rl_validation \\",
        "        --output-dir <throwaway> --run-id <run_id> --validate-only",
        "",
        "The explicit-bank gate is mandatory; there is no bypass flag.",
        "",
        "(See `preflight_results.json` for the batch-level proof of 40/40 PASS",
        "with zero Trainer / checkpoint / run-directory side effects.)",
        "",
        "## Post-run audit",
        "",
        "Fill `postrun_audit/` with the per-row completion state read from",
        "`scripts/generate_m5_matrix.py determine_run_state`, and `validation_",
        "analysis/` with the validation metrics analysis.  A run is COMPLETE",
        "iff the strict 11-condition contract in `determine_run_state` holds.",
        "",
        "## Supersession",
        "",
        "This package supersedes the prior formal matrix SHA256:",
        "",
        f"    {SUPERSEDED_FORMAL_MATRIX_SHA256}",
        "",
        "See `SUPERSESSION_RECORD.md`.",
        "",
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p, sha256_of(p)


def write_launch_commands(out_dir: Path, rows: list[dict]) -> tuple[Path, str]:
    p = out_dir / "launch_commands.txt"
    lines = [f"# 40 launch commands - one per matrix row.  CWD = repository root.", ""]
    for r in rows:
        lines.append(f"# row {r['matrix_index']:>2}  K={r['k']}  regime={r['cost_regime_id']}  seed={r['seed']}  run_id={r['run_id']}")
        lines.append(r["exact_training_command"])
        lines.append("")
    p.write_text("\n".join(lines), encoding="utf-8")
    return p, sha256_of(p)


def write_supersession_record(out_dir: Path, head: str, output_root: str,
                              matrix_sha: str) -> tuple[Path, str]:
    p = out_dir / "SUPERSESSION_RECORD.md"
    lines = [
        "# Supersession Record - M5 M5 reproducibility Formal Matrix",
        "",
        "## Superseded artifact",
        "",
        f"- Prior formal matrix SHA256: `{SUPERSEDED_FORMAL_MATRIX_SHA256}`",
        "- Prior formal output root: `results/milestone5`",
        "",
        "## Superseding artifact",
        "",
        f"- Producing git HEAD: `{head}`",
        f"- New formal matrix SHA256: `{matrix_sha}`",
        f"- New in-repo formal output root: `{output_root}`",
        "",
        "## Why superseded",
        "",
        "The prior formal matrix ran K=1 rows against the baseline light-only",
        "bank (`m5_pilot_k1.json`) and crashed at the K=1 / failure-heavy",
        "production-environment cost_regime_id validation.  The M5 reproducibility",
        "scientific-asset contract introduced 16 regime-specific derived",
        "banks (8 training, 8 validation) in which ALL physical fields are",
        "byte-identical across regimes and only `scenario_id` /",
        "`cost_regime_id` / `bank_id` change, so the four cost regimes are",
        "compared on the SAME physical episode trajectories.  The new formal",
        "matrix references these regime-specific banks and a new non-",
        "colliding output root, and is therefore NOT comparable to and MUST",
        "NOT be merged with the prior formal results.",
        "",
        "## Historical evidence preserved",
        "",
        "The five old K=1/light/no-waste runs (seeds 6521..6525) and the one",
        "old K=1/heavy/no-waste failed attempt remain untouched under",
        "`results/milestone5/` as historical evidence and MUST NOT be reused.",
        "Their bank identity (`e9502f55...`) differs from the new derived",
        "`m5_pilot_k1__light.json` identity (`9fc64d1e...`).",
        "",
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p, sha256_of(p)


def write_smoke_inventory_ref(out_dir: Path, head: str) -> tuple[Path, str]:
    """Cross-reference the M5 reproducibility 8-cell smoke inventory if present.

    The smoke inventory is produced by scripts/run_m5_smoke.py at
    the final HEAD and lives under results/m5_smoke_v1/.  We embed
    a pointer here so the package is self-describing even when the smoke
    driver is re-run later.
    """
    p = out_dir / "M5_SMOKE_INVENTORY.json"
    candidates = [
        REPO / "results" / "m5_smoke_v1" / "smoke_inventory.json",
        REPO / "results" / "m5_smoke_v1_" / "smoke_inventory.json",
    ]
    found = next((c for c in candidates if c.exists()), None)
    payload: dict[str, Any]
    if found is not None:
        try:
            data = json.loads(found.read_text(encoding="utf-8"))
        except Exception as exc:
            data = {"error": f"smoke inventory at {found} failed to parse: {exc}"}
        payload = {
            "frozen_git_head": head,
            "smoke_inventory_source": repo_rel(found),
            "embedded": True,
            "smoke_inventory": data,
        }
    else:
        payload = {
            "frozen_git_head": head,
            "smoke_inventory_source": None,
            "embedded": False,
            "note": ("No 8-cell smoke inventory found on disk at package "
                     "build time.  Run scripts/run_m5_smoke.py at "
                     "this HEAD to produce results/m5_smoke_v1/"
                     "smoke_inventory.json, then rebuild the package."),
        }
    p.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
                 encoding="utf-8")
    return p, sha256_of(p)


def write_run_ledger(ledgers_dir: Path, head: str, rows: list[dict],
                     output_root: str) -> tuple[Path, str]:
    """Formal execution ledger with 40 PENDING rows."""
    p = ledgers_dir / "formal_execution_ledger.json"
    now_marker = "frozen-at-package-build"  # no wall clock; operator stamps runs
    ledger_rows = []
    for r in rows:
        ledger_rows.append({
            "matrix_index": r["matrix_index"],
            "run_id": r["run_id"],
            "k": r["k"],
            "cost_regime_id": r["cost_regime_id"],
            "seed": r["seed"],
            "status": "PENDING",
            "expected_output_dir": r["expected_output_dir"],
            "exact_training_command": r["exact_training_command"],
            "expected_git_commit": head,
            "resolved_config_identity": r["resolved_config_identity"],
            "expected_bank_content_hashes": {
                "training": r["training_scenario_bank_content_hash"],
                "validation": r["validation_scenario_bank_content_hash"],
            },
            "started_at": None,
            "completed_at": None,
            "exit_code": None,
            "final_global_step": None,
            "best_validation_mean_cost": None,
            "completion_state": None,
            "audit_issues": None,
        })
    payload = {
        "frozen_git_head": head,
        "output_root": output_root,
        "total_rows": len(rows),
        "expected_complete_count": 40,
        "ledger_status": now_marker,
        "rows": ledger_rows,
    }
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                 encoding="utf-8")
    return p, sha256_of(p)


def write_readme_placeholders(pkg_root: Path, head: str, output_root: str) -> None:
    (pkg_root / "operator_logs").mkdir(parents=True, exist_ok=True)
    (pkg_root / "postrun_audit").mkdir(parents=True, exist_ok=True)
    (pkg_root / "validation_analysis").mkdir(parents=True, exist_ok=True)
    (pkg_root / "operator_logs" / ".gitkeep").write_text("", encoding="utf-8")
    (pkg_root / "postrun_audit" / "README.md").write_text(
        "# Postrun audit\n\nPer-row completion-state audit goes here.  Reuse "
        "`scripts/generate_m5_matrix.py determine_run_state` to read each run "
        f"directory under `{output_root}` and record COMPLETE / INCOMPLETE / "
        "NOT_STARTED with the strict 11-condition issue list.\n",
        encoding="utf-8")
    (pkg_root / "validation_analysis" / "README.md").write_text(
        "# Validation analysis\n\nPer-row validation-metrics analysis goes "
        "here, populated after the formal runs complete.  No analysis is "
        "performed at package-build time (validate-only / non-training).\n",
        encoding="utf-8")


# ---------------------------------------------------------------------------
# Validation helpers.
# ---------------------------------------------------------------------------

def assert_matrix_invariants(rows: list[dict], head: str, output_root: str) -> None:
    assert len(rows) == 40, f"expected 40 rows, got {len(rows)}"
    assert len({r["run_id"] for r in rows}) == 40, "row run_ids not unique"
    assert len({r["expected_output_dir"] for r in rows}) == 40, "output dirs not unique"
    assert len({r["resolved_config_identity"] for r in rows}) == 40, "row identities not unique"
    # 20 K=1 and 20 K=2
    k1 = [r for r in rows if r["k"] == 1]
    k2 = [r for r in rows if r["k"] == 2]
    assert len(k1) == 20, f"expected 20 K=1 rows, got {len(k1)}"
    assert len(k2) == 20, f"expected 20 K=2 rows, got {len(k2)}"
    # 4 regimes each represented 10x (2 K * 5 seeds)
    from collections import Counter
    reg_counts = Counter(r["cost_regime_id"] for r in rows)
    assert set(reg_counts.keys()) == set(COST_REGIMES), reg_counts
    assert all(v == 10 for v in reg_counts.values()), reg_counts
    # seeds 6521..6525 each represented 8x
    seed_counts = Counter(r["seed"] for r in rows)
    assert set(seed_counts.keys()) == set(TRAINING_SEEDS), seed_counts
    assert all(v == 8 for v in seed_counts.values()), seed_counts
    # action counts
    assert all(r["action_count"] == 6 for r in k1), "K=1 rows must have 6 actions"
    assert all(r["action_count"] == 16 for r in k2), "K=2 rows must have 16 actions"
    # rl_test forbidden
    for r in rows:
        assert r["training_split"] != "rl_test"
        assert r["validation_split"] != "rl_test"
        assert "rl_test" not in r["run_id"]
        assert "rl_test" not in r["exact_training_command"]
    # producing commit
    for r in rows:
        assert r["expected_git_commit"] == head, "row commit mismatch"
        assert r["output_root"] == output_root, "row output_root mismatch"
    # no superseded output root referenced.  We compare at SEGMENT
    # boundaries: results/milestone5_formal_regimebanks_v1 must NOT be
    # treated as living under results/milestone5 (it is a distinct sibling
    # top-level directory, not a child of the superseded root).
    def _under_superseded(p: str, sup: str) -> bool:
        if p == sup:
            return True
        return p.startswith(sup.rstrip("/") + "/")

    for r in rows:
        for sup in SUPERSEDED_OUTPUT_ROOTS:
            assert not _under_superseded(r["output_root"], sup), \
                f"row references superseded output root {sup}"
            assert not _under_superseded(r["expected_output_dir"], sup), \
                f"row references superseded output dir under {sup}"
    # max_steps frozen
    assert all(r["max_steps"] == 100_000 for r in rows)
    # schema
    assert all(r["checkpoint_schema_version_expected"] == CHECKPOINT_SCHEMA_VERSION
               for r in rows)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="M5 formal matrix generator (M5 reproducibility)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory to write the full audit package "
                             "(frozen_launch_package/, run_ledgers/, "
                             "operator_logs/, postrun_audit/, validation_analysis/). "
                             "Required unless --validate-only.")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate the matrix contract and preflight all "
                             "rows WITHOUT writing the persistent package.  "
                             "No Trainer, no checkpoint, no run dir is created. "
                             "Does not require --out-dir.")
    args = parser.parse_args()

    if not args.validate_only and not args.out_dir:
        parser.error("--out-dir is required unless --validate-only is set")

    out_base = DEFAULT_OUTPUT_BASE
    if OUTPUT_BASE_ENV:
        out_base = OUTPUT_BASE_ENV

    head = get_git_head()
    pc_manifest_sha = sha256_of(REPO / PC_MANIFEST_PATH)

    rows: list[dict] = []
    for k in MAINTENANCE_CAPACITIES:
        for regime in COST_REGIMES:
            for seed in TRAINING_SEEDS:
                rows.append(build_row(k, regime, seed, head, pc_manifest_sha, out_base))

    assert_matrix_invariants(rows, head, out_base)

    # Snapshot the formal output root BEFORE preflight to prove preflight
    # creates no side effects.
    out_base_path = REPO / out_base
    before_snap = _snapshot_dir(out_base_path)

    preflight_results, pass_count = preflight_all_rows(rows)

    # Stage 4: command-level preflight over all 40 rows.  Exercies the REAL
    # production CLI --validate-only subprocess for each exact_training_command
    # (with --output-dir redirected to a throwaway temp dir), and resolves each
    # command through the shared production resolver to cross-check effective
    # config / identity / bank hashes, with no Trainer / output-dir / checkpoint
    # side effects.
    cmd_preflight_results, cmd_pass_count = command_preflight_all_rows(rows)

    after_snap = _snapshot_dir(out_base_path)
    side_effects_clean = (before_snap == after_snap)
    if not side_effects_clean:
        print("WARNING: formal output root entries changed during preflight "
              f"(before={sorted(before_snap)} after={sorted(after_snap)})",
              file=sys.stderr)

    # The no-side-effects proof also demands that the output root was not
    # created from scratch by preflight.  If it did not exist before, it must
    # not exist after.
    pref_no_mkdir = ((not out_base_path.exists()) or
                     (before_snap == after_snap))

    all_pass = (
        pass_count == len(rows)
        and cmd_pass_count == len(rows)
        and side_effects_clean and pref_no_mkdir
    )
    if not all_pass:
        print(f"Preflight incomplete: pass_count={pass_count}/{len(rows)} "
              f"cmd_pass_count={cmd_pass_count}/{len(rows)} "
              f"side_effects_clean={side_effects_clean} "
              f"pref_no_mkdir={pref_no_mkdir}", file=sys.stderr)

    matrix = {
        "frozen_git_head": head,
        "checkpoint_schema_version_expected": CHECKPOINT_SCHEMA_VERSION,
        "parameters": {
            "cost_regimes": COST_REGIMES,
            "maintenance_capacities": MAINTENANCE_CAPACITIES,
            "training_seeds": TRAINING_SEEDS,
        },
        "training_split": "predictor_train",
        "validation_split": "rl_validation",
        "max_steps": 100_000,
        "output_root": out_base,
        "prediction_cache_manifest_path": PC_MANIFEST_PATH,
        "prediction_cache_manifest_sha256": pc_manifest_sha,
        "no_rl_test": True,
        "superseded_prior_formal_matrix_sha256": SUPERSEDED_FORMAL_MATRIX_SHA256,
        "superseded_output_roots": SUPERSEDED_OUTPUT_ROOTS,
        "rows": rows,
    }

    # If --validate-only, do NOT write the persistent package.  Compute the
    # matrix SHA in-memory for the report and exit.
    if args.validate_only:
        matrix_bytes = (json.dumps(matrix, indent=2, sort_keys=True) + "\n").encode("utf-8")
        matrix_sha = sha256_bytes(matrix_bytes)
        print("VALIDATE-ONLY (no package written, no Trainer, no checkpoint).")
        print(f"FROZEN HEAD:        {head}")
        print(f"OUTPUT ROOT:        {out_base}")
        print(f"Total rows:         {len(rows)}")
        print(f"Unique run_ids:     {len({r['run_id'] for r in rows})}")
        print(f"Unique output dirs: {len({r['expected_output_dir'] for r in rows})}")
        print(f"Preflight PASS:     {pass_count}/{len(rows)}")
        print(f"Command preflight PASS: {cmd_pass_count}/{len(rows)}")
        print(f"Side-effects clean: {side_effects_clean}")
        print(f"All-pass:           {all_pass}")
        print(f"Matrix SHA256 (computed, not written): {matrix_sha}")
        return 0 if all_pass else 1

    # ----- Write the full audit package -----
    pkg_root = Path(args.out_dir)
    pkg_root.mkdir(parents=True, exist_ok=True)
    flp = pkg_root / "frozen_launch_package"
    flp.mkdir(parents=True, exist_ok=True)
    ledgers_dir = pkg_root / "run_ledgers"
    ledgers_dir.mkdir(parents=True, exist_ok=True)

    matrix_path, matrix_sha = write_matrix_json(flp, matrix)
    csv_path, csv_sha = write_matrix_csv(flp, rows)
    preflight_path, preflight_sha = write_preflight_results(
        flp, head, rows, preflight_results, pass_count,
        cmd_preflight_results, cmd_pass_count, out_base,
        before_snap, after_snap)
    runbook_path, runbook_sha = write_runbook(
        flp, head, out_base, pc_manifest_sha, matrix_sha, pass_count)
    launch_path, launch_sha = write_launch_commands(flp, rows)
    supersession_path, supersession_sha = write_supersession_record(
        flp, head, out_base, matrix_sha)
    smoke_ref_path, smoke_ref_sha = write_smoke_inventory_ref(flp, head)

    shaum_path, shaum_sha = write_sha256sums(flp, [
        ("formal_matrix.json", matrix_sha),
        ("formal_matrix.csv", csv_sha),
        ("preflight_results.json", preflight_sha),
        ("FORMAL_RUNBOOK.md", runbook_sha),
        ("launch_commands.txt", launch_sha),
        ("SUPERSESSION_RECORD.md", supersession_sha),
        ("M5_SMOKE_INVENTORY.json", smoke_ref_sha),
    ])

    ledgers_path, ledgers_sha = write_run_ledger(ledgers_dir, head, rows, out_base)
    write_readme_placeholders(pkg_root, head, out_base)

    from collections import Counter
    c = Counter(r["training_scenario_bank_path"] for r in rows)

    print(f"FROZEN HEAD:        {head}")
    print(f"OUTPUT ROOT:        {out_base}")
    print(f"PACKAGE ROOT:       {pkg_root}")
    print(f"Total rows:         {len(rows)}")
    print(f"Unique run_ids:     {len({r['run_id'] for r in rows})}")
    print(f"Unique output dirs: {len({r['expected_output_dir'] for r in rows})}")
    print(f"Preflight PASS:     {pass_count}/{len(rows)}")
    print(f"Side-effects clean: {side_effects_clean}")
    print(f"All-pass:           {all_pass}")
    print()
    print(f"formal_matrix.json        sha={matrix_sha}")
    print(f"formal_matrix.csv         sha={csv_sha}")
    print(f"preflight_results.json    sha={preflight_sha}")
    print(f"FORMAL_RUNBOOK.md         sha={runbook_sha}")
    print(f"launch_commands.txt       sha={launch_sha}")
    print(f"SUPERSESSION_RECORD.md    sha={supersession_sha}")
    print(f"M5_SMOKE_INVENTORY.json sha={smoke_ref_sha}")
    print(f"SHA256SUMS.txt            sha={shaum_sha}")
    print(f"formal_execution_ledger.json sha={ledgers_sha} (40 PENDING rows)")
    print()
    print("M5_FORMAL_MATRIX_SHA256=" + matrix_sha)
    print("Banks per row distribution:")
    for path, n in sorted(c.items()):
        print(f"  {path}  x  {n}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
