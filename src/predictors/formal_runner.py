#!/usr/bin/env python3
"""M8 Formal Predictor Training Runner.

Executes formal predictor training runs per M8_FORMAL_COMMAND_MATRIX.json.
Produces complete 14-artifact evidence package per M8_FORMAL_ARTIFACT_LIFECYCLE_PLAN.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Establish PROJECT_ROOT before any local imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Verify PROJECT_ROOT is correct
if PROJECT_ROOT != Path.cwd():
    raise RuntimeError(
        f"PROJECT_ROOT {PROJECT_ROOT} != cwd {Path.cwd()}. "
        f"File: formal_runner.py, Field: PROJECT_ROOT, Stage: initialization"
    )

import numpy as np
import pandas as pd

# Import formal metrics module
from src.predictors.formal_metrics import (
    compute_formal_metrics,
    validate_prediction_frame,
    write_metrics_json,
)

# Import MPS lock utility
from scripts.drl_heavy_mps_lock import acquire_mps_heavy_lock, MPSHeavyLock

# Import training components (read-only usage)
from src.predictors.dataset import build_dataloaders
from src.predictors.model import build_predictor
from src.predictors.losses import build_loss_fn
from src.predictors.io_utils import atomic_write_json, atomic_torch_save

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


# ============================================================================
# Constants and Configuration
# ============================================================================

# Frozen constants from protocol
VALID_LOSS_TYPES = {"mse", "linex"}
VALID_SEEDS = {6521, 6522, 6523, 6524, 6525}
EXPECTED_ROW_COUNT = 3146
RUL_CAP = 125
VALIDATION_SPLIT = "predictor_validation"
TRAIN_SPLIT = "predictor_train"

# ============================================================================
# Canonical Condition Identity Helper
# ============================================================================

# Exact formal LinEx coefficient mappings - ONLY source of truth
_FORMAL_LINEX_IDS = {
    Decimal("0.05"): "linex_a05",
    Decimal("0.10"): "linex_a10",
}

# Derived valid coefficients set (for validation) - derived from _FORMAL_LINEX_IDS
VALID_LINEEX_A_DECIMAL = set(_FORMAL_LINEX_IDS.keys())


def _to_formal_decimal(value: Union[float, str]) -> Decimal:
    """Convert a float or string to Decimal for exact formal comparison.

    Uses Decimal(str(value)) to preserve exact decimal representation.
    Does NOT quantize or round unsupported inputs.
    """
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as e:
        raise ValueError(f"Invalid linex_a value: {value}") from e


def get_formal_condition_id(loss_type: str, linex_a: Optional[float]) -> str:
    """Canonical formal condition identifier.

    Required mapping:
    - loss_type=mse -> mse_control
    - loss_type=linex, linex_a=0.05 -> linex_a05
    - loss_type=linex, linex_a=0.10 -> linex_a10

    Unsupported formal LinEx coefficients raise ValueError.
    No float-string slicing, no rounding, no quantization.
    Uses Decimal(str(value)) for exact comparison.
    """
    if loss_type == "mse":
        if linex_a is not None:
            raise ValueError("linex_a must be None for MSE loss")
        return "mse_control"
    elif loss_type == "linex":
        if linex_a is None:
            raise ValueError("linex_a required for LinEx loss")
        # Exact Decimal-based comparison - no rounding, no slicing
        a_decimal = _to_formal_decimal(linex_a)
        if a_decimal in _FORMAL_LINEX_IDS:
            return _FORMAL_LINEX_IDS[a_decimal]
        else:
            raise ValueError(
                f"Unsupported formal LinEx coefficient: {linex_a}. "
                f"Valid coefficients: 0.05, 0.10"
            )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


# Expected input file hashes (from M8_FORMAL_INPUT_HASHES.json)
# Keys are data-root-relative paths per frozen protocol
EXPECTED_INPUT_HASHES = {
    "04_PROTOCOL/fd001_feature_schema_v1.json": "43772bbcaab99e79264fac54780025a54de6e29c75fdccab6dd4ef4d2cbe21da",
    "04_PROTOCOL/fd001_normalizer_v2.json": "08477180719d004dc8f962762735b6344f8198a7719c1c510f9ad7ee15784fde",
    "01_SPLIT/fd001_unit_split_v1.csv": "a86fe8cb1e01d4c7b47fd76d9bcc23351e64b4641386838bee6475bd2863dc9a",
    "02_CYCLE_TABLE/fd001_train_cycle_table_v1.parquet": "d51cddbdd5c4851cf679a0a468674717fc91aba1d0457cdf83b423c2d37e7264",
    "05_WINDOW_INDEX/fd001_window_index_v1.parquet": "f2a3a671b944d7b99dba8ea49baa6b21fd63252f97670bab90befa9a02b0f86f",
}

# Val cycle table path (frozen dependency)
VAL_CYCLE_TABLE_REL = "02_CYCLE_TABLE/fd001_val_cycle_table_v1.parquet"

# Valid training statuses
VALID_STATUSES = {"COMPLETED", "EARLY_STOPPED", "FAILED"}


# ============================================================================
# Exception Classes
# ============================================================================

class PreflightError(Exception):
    """Preflight check failure - no leaf directory created."""
    pass


class ValidationError(Exception):
    """Input validation failure."""
    pass


class IncompleteRunError(Exception):
    """Run could not be finalized cleanly (writer barrier failure)."""
    pass


def classify_temp_file(name: str) -> str:
    """Classify a temporary file name.

    Returns:
        ACTIVE_TEMP: correctly formatted temp + live PID
        STALE_TEMP: correctly formatted temp + dead PID
        MALFORMED_TEMP: incorrectly formatted temp name
    """
    # Expected pattern: stdout_stderr.log.tmp.{pid}.{random_hex}
    prefix = "stdout_stderr.log.tmp."
    if not name.startswith(prefix):
        return "MALFORMED_TEMP"

    suffix = name[len(prefix):]
    parts = suffix.split(".")
    if len(parts) != 2:
        return "MALFORMED_TEMP"

    pid_str, rand_hex = parts

    # Check PID is valid integer
    try:
        pid = int(pid_str)
    except ValueError:
        return "MALFORMED_TEMP"

    # Check random hex is exactly 8 chars (4 bytes = 8 hex chars)
    if len(rand_hex) != 8:
        return "MALFORMED_TEMP"
    try:
        int(rand_hex, 16)
    except ValueError:
        return "MALFORMED_TEMP"

    # Check if PID is alive
    try:
        os.kill(pid, 0)  # Signal 0 just checks process existence
        return "ACTIVE_TEMP"
    except ProcessLookupError:
        # Process doesn't exist
        return "STALE_TEMP"
    except PermissionError:
        # Process exists but we can't signal it (different user)
        return "ACTIVE_TEMP"
    except OSError:
        return "ACTIVE_TEMP"


# ============================================================================
# Git and Environment Identity
# ============================================================================

def get_git_identity() -> Dict[str, Any]:
    """Capture git state at run start."""
    def run_git(args: List[str]) -> str:
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout.strip()
        except Exception:
            return "unknown"

    dirty_output = run_git(["status", "--porcelain=v1", "-uall"])
    return {
        "git_commit": run_git(["rev-parse", "HEAD"]),
        "git_tree": run_git(["write-tree"]),
        "git_branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(dirty_output),
    }


def get_environment_identity() -> Dict[str, Any]:
    """Capture runtime environment."""
    return {
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "device": "mps",
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False,
        "mps_built": torch.backends.mps.is_built() if hasattr(torch.backends, "mps") else False,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "hostname": platform.node(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat() + "Z",
    }


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_input_hashes(data_dir: Path) -> Dict[str, str]:
    """Verify all 5 training input files match protocol hashes."""
    computed = {}
    for filename, expected_hash in EXPECTED_INPUT_HASHES.items():
        full_path = data_dir / filename
        if not full_path.exists():
            raise PreflightError(f"Required input file missing: {full_path}")
        actual_hash = compute_file_hash(full_path)
        if actual_hash != expected_hash:
            raise PreflightError(
                f"Input hash mismatch for {filename}: expected {expected_hash}, got {actual_hash}"
            )
        computed[filename] = actual_hash
    return computed


def verify_val_cycle_table(data_dir: Path) -> str:
    """Verify validation cycle table exists and return its hash."""
    val_table_path = data_dir / VAL_CYCLE_TABLE_REL
    if not val_table_path.exists():
        raise PreflightError(
            f"Validation cycle table missing: {val_table_path}. "
            f"Run scripts/build_val_cycle_table.py first."
        )
    return compute_file_hash(val_table_path)


# ============================================================================
# CLI Argument Parsing
# ============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse and validate CLI arguments."""
    parser = argparse.ArgumentParser(
        description="M8 Formal Predictor Training Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments (all 15 command cells)
    parser.add_argument("--config", type=Path, required=True, help="Path to base config JSON")
    parser.add_argument("--data-dir", type=Path, required=True, help="Processed data directory (fd001/v2)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Unique output directory for this run")
    parser.add_argument("--seed", type=int, required=True, help="Random seed (6521-6525)")
    parser.add_argument("--sequence-length", type=int, required=True, help="Sequence length (50)")
    parser.add_argument("--rul-cap", type=int, required=True, help="RUL cap (125)")
    parser.add_argument("--model-type", type=str, required=True, choices=["mlp"], help="Model architecture (mlp)")
    parser.add_argument("--hidden-dim", type=int, required=True, help="Hidden dimension (128)")
    parser.add_argument("--n-layers", type=int, required=True, help="Number of layers (3)")
    parser.add_argument("--dropout", type=float, required=True, help="Dropout rate (0.2)")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size (64)")
    parser.add_argument("--learning-rate", type=float, required=True, help="Learning rate (1e-3)")
    parser.add_argument("--weight-decay", type=float, required=True, help="Weight decay (1e-4)")
    parser.add_argument("--max-epochs", type=int, required=True, help="Max epochs (200)")
    parser.add_argument("--patience", type=int, required=True, help="Early stopping patience (20)")
    parser.add_argument("--device", type=str, required=True, choices=["mps"], help="Device (mps)")
    parser.add_argument("--loss-type", type=str, required=True, choices=["mse", "linex"], help="Loss function type")
    parser.add_argument("--linex-a", type=float, required=False, help="LinEx parameter (0.05 or 0.10; OMIT for MSE)")
    parser.add_argument("--linex-overflow-threshold", type=float, required=True, help="LinEx overflow threshold (20.0)")

    args = parser.parse_args(argv)

    # Validate seed
    if args.seed not in VALID_SEEDS:
        parser.error(f"Invalid seed {args.seed}. Must be one of {sorted(VALID_SEEDS)}")

    # Validate loss-type / linex-a combination
    if args.loss_type == "mse":
        if args.linex_a is not None:
            parser.error("--linex-a must be omitted for MSE loss")
    elif args.loss_type == "linex":
        if args.linex_a is None:
            parser.error("--linex-a is required for LinEx loss")
        # Validate using Decimal-based comparison
        try:
            a_decimal = _to_formal_decimal(args.linex_a)
            if a_decimal not in VALID_LINEEX_A_DECIMAL:
                parser.error(f"--linex-a must be one of 0.05, 0.10 for LinEx loss")
        except ValueError as e:
            parser.error(str(e))
    else:
        parser.error(f"Invalid loss-type: {args.loss_type}")

    # Validate sequence length, rul_cap, model type, device
    if args.sequence_length != 50:
        parser.error("--sequence-length must be 50")
    if args.rul_cap != 125:
        parser.error("--rul-cap must be 125")
    if args.model_type != "mlp":
        parser.error("--model-type must be mlp")
    if args.device != "mps":
        parser.error("--device must be mps")

    return args


def capture_command_line(args: argparse.Namespace) -> str:
    """Reconstruct exact command line for command.txt artifact."""
    # Build argument list in deterministic order
    arg_order = [
        "config", "data_dir", "output_dir", "seed", "sequence_length", "rul_cap",
        "model_type", "hidden_dim", "n_layers", "dropout", "batch_size",
        "learning_rate", "weight_decay", "max_epochs", "patience", "device",
        "loss_type", "linex_a", "linex_overflow_threshold",
    ]

    parts = [sys.executable, str(Path(__file__).resolve())]
    for arg_name in arg_order:
        value = getattr(args, arg_name)
        if value is not None:
            parts.append(f"--{arg_name.replace('_', '-')}")
            if isinstance(value, Path):
                parts.append(str(value))
            else:
                parts.append(str(value))
    return " ".join(parts)


def resolve_configuration(args: argparse.Namespace) -> Dict[str, Any]:
    """Merge config file + CLI overrides into effective configuration."""
    with open(args.config, "r") as f:
        config = json.load(f)

    # Extract defaults from nested config structure
    default_data_dir = Path(config.get("data", {}).get("data_dir", "data/processed/fd001/v2"))
    default_output_dir = Path(config.get("_output_dir", "results/predictor/mse_baseline_v2"))
    default_seed = config.get("seed", 6521)
    default_seq_len = config.get("sequence_length", 50)
    default_rul_cap = config.get("rul_cap", 125)
    default_model = config.get("model", {})
    default_model_type = default_model.get("type", "mlp")
    default_hidden_dim = default_model.get("hidden_dim", 128)
    default_n_layers = default_model.get("n_layers", 3)
    default_dropout = default_model.get("dropout", 0.2)
    default_training = config.get("training", {})
    default_batch_size = default_training.get("batch_size", 64)
    default_lr = default_training.get("learning_rate", 1e-3)
    default_weight_decay = default_training.get("weight_decay", 1e-4)
    default_max_epochs = default_training.get("max_epochs", 200)
    default_patience = default_training.get("patience", 20)
    default_device = config.get("device", "auto")
    default_loss = config.get("loss", {})
    default_loss_type = default_loss.get("type", "mse")
    default_linex_a = default_loss.get("linex_a", 0.1)
    default_linex_overflow = default_loss.get("linex_overflow_threshold", 20.0)

    cli_overrides = {}
    effective = config.copy()

    # Apply CLI overrides (same logic as train.py)
    overrides_map = {
        "data_dir": ("data", "data_dir", args.data_dir, default_data_dir),
        "output_dir": ("_output_dir", None, args.output_dir, default_output_dir),
        "seed": ("seed", None, args.seed, default_seed),
        "sequence_length": ("sequence_length", None, args.sequence_length, default_seq_len),
        "rul_cap": ("rul_cap", None, args.rul_cap, default_rul_cap),
        "model_type": ("model", "type", args.model_type, default_model_type),
        "hidden_dim": ("model", "hidden_dim", args.hidden_dim, default_hidden_dim),
        "n_layers": ("model", "n_layers", args.n_layers, default_n_layers),
        "dropout": ("model", "dropout", args.dropout, default_dropout),
        "batch_size": ("training", "batch_size", args.batch_size, default_batch_size),
        "learning_rate": ("training", "learning_rate", args.learning_rate, default_lr),
        "weight_decay": ("training", "weight_decay", args.weight_decay, default_weight_decay),
        "max_epochs": ("training", "max_epochs", args.max_epochs, default_max_epochs),
        "patience": ("training", "patience", args.patience, default_patience),
        "device": ("device", None, args.device, default_device),
        "loss_type": ("loss", "type", args.loss_type, default_loss_type),
        "linex_a": ("loss", "linex_a", args.linex_a, default_linex_a),
        "linex_overflow_threshold": ("loss", "linex_overflow_threshold", args.linex_overflow_threshold, default_linex_overflow),
    }

    for key, (section, subkey, cli_val, default_val) in overrides_map.items():
        if cli_val != default_val:
            if section in effective:
                if subkey:
                    orig = effective[section].get(subkey, default_val)
                else:
                    orig = effective.get(section, default_val)
            else:
                orig = default_val
            cli_overrides[key] = {"original": orig, "override": cli_val}

    # Build resolved config for artifact
    resolved_config = {
        "config_path": str(args.config),
        "config_hash": compute_file_hash(args.config) if args.config.exists() else "unknown",
        "cli_overrides": cli_overrides,
        "effective_config": {
            "seed": args.seed,
            "sequence_length": args.sequence_length,
            "rul_cap": args.rul_cap,
            "model": {
                "type": args.model_type,
                "hidden_dim": args.hidden_dim,
                "n_layers": args.n_layers,
                "dropout": args.dropout,
            },
            "training": {
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "max_epochs": args.max_epochs,
                "patience": args.patience,
            },
            "data": {
                "data_dir": str(args.data_dir),
                "train_split": TRAIN_SPLIT,
                "validation_split": VALIDATION_SPLIT,
            },
            "device": args.device,
            "loss": {
                "type": args.loss_type,
                "linex_a": args.linex_a,
                "linex_overflow_threshold": args.linex_overflow_threshold,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "git_commit": get_git_identity()["git_commit"],
    }

    return resolved_config


# ============================================================================
# Runner-Owned Logging (FD-Level Capture)
# ============================================================================

class RunnerLogging:
    """Manages runner-owned stdout/stderr capture via file descriptor redirection."""

    def __init__(self, output_dir: Path, event_recorder: Optional[List[str]] = None):
        self.output_dir = output_dir
        self.log_tmp: Optional[Path] = None
        self.log_fd: Optional[int] = None
        self.saved_stdout_fd: Optional[int] = None
        self.saved_stderr_fd: Optional[int] = None
        self._writer_registry: List[Any] = []
        self._events: List[str] = event_recorder if event_recorder is not None else []

    def record_event(self, event: str) -> None:
        """Record an event for ordering verification (test-only)."""
        self._events.append(event)

    def setup(self) -> None:
        """Set up fd-level capture for stdout (fd 1) and stderr (fd 2)."""
        # Create temp log in same directory (same filesystem)
        self._log_tmp = self.output_dir / f"stdout_stderr.log.tmp.{os.getpid()}.{os.urandom(4).hex()}"

        # Open temp log
        self._log_fd = os.open(self._log_tmp, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)

        # Save original stdout/stderr
        self._saved_stdout_fd = os.dup(1)
        self._saved_stderr_fd = os.dup(2)

        # Redirect fd 1 and fd 2 to temp log
        os.dup2(self._log_fd, 1)
        os.dup2(self._log_fd, 2)

        # Rebind sys.stdout/stderr to new fds
        sys.stdout = os.fdopen(1, "w", buffering=1, encoding="utf-8")
        sys.stderr = os.fdopen(2, "w", buffering=1, encoding="utf-8")

    def register_writer(self, writer: Any) -> None:
        """Register a writer (process, thread, QueueListener, etc.) for barrier."""
        self._writer_registry.append(writer)

    def finalize(self) -> None:
        """Finalize logging: flush, fsync, atomic install, restore fds.

        Required order:
        1. stop accepting new writer work;
        2. stop and join every registered writer;
        3. verify no registered writer remains alive;
        4. record writer_barrier_complete;
        5. sys.stdout.flush();
        6. sys.stderr.flush();
        7. logging.shutdown();
        8. os.fsync(log_fd);
        9. restore fd 1 and fd 2;
        10. close saved descriptors and runner-owned log fd;
        11. os.replace(temp_log, stdout_stderr.log);
        12. fsync the leaf directory;
        13. write COMPLETED or FAILED last;
        14. fsync the leaf directory;
        15. perform no further artifact or log writes.

        INVARIANT: If writer_barrier() returns False or any writer stop/join raises,
        we MUST:
        - stop the normal finalization sequence immediately;
        - restore fd 1 and fd 2 safely;
        - close owned descriptors safely;
        - NOT os.replace the temporary log into stdout_stderr.log;
        - NOT claim the temporary log as authoritative;
        - NOT write COMPLETED;
        - NOT write FAILED;
        - propagate IncompleteRunError;
        - return a non-zero process status;
        - leave the run in INCOMPLETE state.
        """
        # Initialize all cleanup-state variables BEFORE the try block
        log_fd = self._log_fd
        log_tmp = self._log_tmp
        saved_stdout_fd = self._saved_stdout_fd
        saved_stderr_fd = self._saved_stderr_fd

        # Clear instance references to make cleanup idempotent
        self._log_fd = None
        self._log_tmp = None
        self._saved_stdout_fd = None
        self._saved_stderr_fd = None

        # If logging was never set up, nothing to do
        if log_fd is None:
            return

        writer_barrier_error: Optional[BaseException] = None

        try:
            # 1. Stop accepting new writer work (no new registrations after this point)
            # 2. Stop and join all registered writers FIRST (writer barrier)
            writer_barrier_success = self.writer_barrier()

            # 3. Verify no registered writer remains alive
            for writer in self._writer_registry:
                if hasattr(writer, "is_alive") and writer.is_alive():
                    writer_barrier_error = IncompleteRunError("Writer still alive after barrier")
                    raise writer_barrier_error

            if not writer_barrier_success:
                writer_barrier_error = IncompleteRunError("Writer barrier failed - some writers did not join")
                raise writer_barrier_error

            # Writer barrier succeeded - proceed with log finalization
            self.record_event("writer_barrier_complete")

            # 5. sys.stdout.flush()
            sys.stdout.flush()
            self.record_event("stdout_flush")

            # 6. sys.stderr.flush()
            sys.stderr.flush()
            self.record_event("stderr_flush")

            # 7. logging.shutdown() - flushes and closes ALL handlers on ALL loggers
            self.record_event("logging_shutdown")
            import logging
            logging.shutdown()

            # 8. os.fsync(log_fd)
            os.fsync(log_fd)
            self.record_event("log_fsync")

            # 9. Restore fd 1 and fd 2
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            self.record_event("fd_restore")

            # 10. Close saved descriptors and runner-owned log fd
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
            os.close(log_fd)
            self.record_event("fd_close")

            # 11. os.replace(temp_log, stdout_stderr.log)
            log_final = self.output_dir / "stdout_stderr.log"
            os.replace(log_tmp, log_final)
            self.record_event("log_atomic_replace")

            # 12. fsync the leaf directory
            dir_fd = os.open(self.output_dir, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
            self.record_event("directory_fsync")

            # NOTE: Steps 13 and 14 (write COMPLETED/FAILED and fsync leaf directory)
            # are performed by write_all_artifacts / write_failed_artifacts AFTER
            # logging.finalize() returns, per the required ordering.

        except Exception:
            # On ANY error (including writer_barrier_error), clean up but don't mask the original error
            self._cleanup_descriptors_saved(log_fd, saved_stdout_fd, saved_stderr_fd)
            raise

    def _cleanup_descriptors_saved(self, log_fd, saved_stdout_fd, saved_stderr_fd) -> None:
        """Clean up file descriptors using saved references."""
        if saved_stdout_fd is not None:
            try:
                os.dup2(saved_stdout_fd, 1)
            except OSError:
                pass
            try:
                os.close(saved_stdout_fd)
            except OSError:
                pass
        if saved_stderr_fd is not None:
            try:
                os.dup2(saved_stderr_fd, 2)
            except OSError:
                pass
            try:
                os.close(saved_stderr_fd)
            except OSError:
                pass
        if log_fd is not None:
            try:
                os.close(log_fd)
            except OSError:
                pass

    def writer_barrier(self) -> bool:
        """Stop and join all registered writers.

        Returns:
            True if all writers joined successfully, False otherwise
        """
        all_joined = True

        for writer in self._writer_registry:
            try:
                if hasattr(writer, "stop"):
                    writer.stop()
                    self.record_event("queue_listener_stop")
                if hasattr(writer, "join"):
                    writer.join(timeout=30)
                    self.record_event("queue_listener_join")
                    if writer.is_alive() if hasattr(writer, "is_alive") else False:
                        all_joined = False
                elif hasattr(writer, "wait"):
                    writer.wait(timeout=30)
                elif hasattr(writer, "terminate"):
                    writer.terminate()
                    writer.wait(timeout=10)
            except Exception:
                all_joined = False

        return all_joined


# ============================================================================
# Atomic Write Utilities
# ============================================================================

def atomic_write_target(target: Path) -> Path:
    """Generate unique temp path for atomic write."""
    pid = os.getpid()
    rand = os.urandom(4).hex()
    return target.with_name(f"{target.name}.tmp.{pid}.{rand}")


def atomic_write_json(target: Path, payload: Dict[str, Any]) -> None:
    """Atomically write JSON with fsync + os.replace."""
    tmp = atomic_write_target(target)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, target)
    # Fsync directory
    dir_fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def atomic_write_parquet(target: Path, df: pd.DataFrame) -> None:
    """Atomically write Parquet with fsync + os.replace."""
    tmp = atomic_write_target(target)
    df.to_parquet(tmp, index=False)
    fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, target)
    # Fsync directory
    dir_fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def atomic_write_sha256(target: Path, hash_hex: str, filename: str) -> None:
    """Write SHA256 sidecar atomically."""
    tmp = atomic_write_target(target)
    with open(tmp, "w") as f:
        f.write(f"{hash_hex}  {filename}\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, target)
    # Fsync directory
    dir_fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def atomic_copy_checkpoint(src: Path, dst: Path) -> None:
    """Atomically copy checkpoint from checkpoints/ to output root."""
    tmp = atomic_write_target(dst)
    shutil.copy2(src, tmp)
    fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, dst)
    # Fsync directory
    dir_fd = os.open(dst.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def write_terminal_marker(output_dir: Path, status: str) -> None:
    """Write COMPLETED or FAILED marker atomically.

    Per M8_FORMAL_ARTIFACT_LIFECYCLE_PLAN.md, terminal markers are only
    COMPLETED (for COMPLETED or EARLY_STOPPED runs) or FAILED (for FAILED runs).
    EARLY_STOPPED is a valid training outcome that still produces a COMPLETED marker.
    """
    if status not in ("COMPLETED", "EARLY_STOPPED", "FAILED"):
        raise ValueError(
            f"write_terminal_marker: invalid status '{status}'. "
            f"Expected one of ('COMPLETED', 'EARLY_STOPPED', 'FAILED'). "
            f"File: formal_runner.py, Field: status, Stage: terminal_marker"
        )

    # EARLY_STOPPED maps to COMPLETED marker per protocol
    marker_name = "COMPLETED" if status in ("COMPLETED", "EARLY_STOPPED") else "FAILED"
    marker = output_dir / marker_name
    tmp = atomic_write_target(marker)
    tmp.touch()
    fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, marker)
    # Fsync directory
    dir_fd = os.open(output_dir, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


# ============================================================================
# Training Execution
# ============================================================================

def run_training(
    args: argparse.Namespace,
    output_dir: Path,
    identity: Dict[str, Any],
    command_line: str,
    logging: RunnerLogging,
) -> Tuple[str, Path, Path, List[Dict]]:
    """Execute predictor training.

    Returns:
        (training_status, best_checkpoint_path, last_checkpoint_path, training_history)
    """
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device

    # Build dataloaders
    dataloaders = build_dataloaders(
        data_dir=args.data_dir,
        sequence_length=args.sequence_length,
        rul_cap=args.rul_cap,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    train_loader = dataloaders[TRAIN_SPLIT]
    val_loader = dataloaders[VALIDATION_SPLIT]

    # Build model
    n_features = train_loader.dataset.n_features
    model = build_predictor(
        model_type=args.model_type,
        n_features=n_features,
        sequence_length=args.sequence_length,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    model = model.to(device)

    # Loss and optimizer
    if args.loss_type == "mse":
        criterion = build_loss_fn(loss_type="mse")
    else:
        criterion = build_loss_fn(
            loss_type="linex",
            a=args.linex_a,
            overflow_threshold=args.linex_overflow_threshold,
        )
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.patience // 2,
        min_lr=1e-6,
    )

    # Checkpoint config
    checkpoint_config = {
        "seed": args.seed,
        "sequence_length": args.sequence_length,
        "rul_cap": args.rul_cap,
        "model_type": args.model_type,
        "n_features": n_features,
        "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "loss_type": args.loss_type,
        "linex_a": args.linex_a,
        "linex_overflow_threshold": args.linex_overflow_threshold,
        "normalizer_id": "fd001_normalizer_v2",
        "feature_schema_id": "fd001_feature_schema_v1",
    }

    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    best_val_rmse = float("inf")
    best_epoch = 0
    early_stop_counter = 0
    train_history = []

    training_status = "COMPLETED"

    try:
        for epoch in range(args.max_epochs):
            epoch_start = time.time()

            # Train
            model.train()
            total_loss = 0.0
            n_samples = 0
            abs_error_sum = 0.0

            for batch in train_loader:
                x = batch["features"].to(device)
                y = batch["rul_capped"].to(device)

                optimizer.zero_grad()
                y_pred = model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                n_batch = int(y.numel())
                total_loss += loss.item() * n_batch
                n_samples += n_batch
                abs_error_sum += float(torch.sum(torch.abs(y_pred.detach() - y.detach())).to("cpu"))

            train_loss = total_loss / n_samples if n_samples > 0 else 0.0
            train_rmse = np.sqrt(train_loss)
            train_mae = abs_error_sum / n_samples if n_samples > 0 else 0.0

            # Validate
            model.eval()
            val_loss_total = 0.0
            y_true_list = []
            y_pred_list = []

            with torch.no_grad():
                for batch in val_loader:
                    x = batch["features"].to(device)
                    y = batch["rul_capped"].to(device)
                    y_pred = model(x)
                    loss = criterion(y_pred, y)
                    val_loss_total += loss.item() * len(y)
                    y_true_list.append(y.cpu().numpy())
                    y_pred_list.append(y_pred.cpu().numpy())

            y_true = np.concatenate(y_true_list)
            y_pred = np.concatenate(y_pred_list)
            val_loss = val_loss_total / len(y_true)
            val_rmse = np.sqrt(val_loss)
            val_mae = np.mean(np.abs(y_pred - y_true))

            # MAPE
            mask = y_true > 0
            val_mape = float(np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100) if mask.sum() > 0 else float("inf")

            epoch_duration = time.time() - epoch_start
            scheduler.step(val_rmse)

            # Check if best
            is_best = val_rmse < best_val_rmse
            if is_best:
                best_val_rmse = val_rmse
                best_epoch = epoch
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            # Log epoch
            train_history.append({
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train_rmse": float(train_rmse),
                "train_mae": float(train_mae),
                "val_loss": float(val_loss),
                "val_rmse": float(val_rmse),
                "val_mae": float(val_mae),
                "val_mape": float(val_mape) if val_mape != float("inf") else None,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "epoch_duration_seconds": float(epoch_duration),
                "is_best_so_far": bool(is_best),
                "early_stopping_counter": int(early_stop_counter),
            })

            # Save best checkpoint atomically
            if is_best:
                checkpoint = {
                    "schema_version": "fd001_checkpoint_v2",
                    "checkpoint_type": "best",
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_mae": train_mae,
                    "val_rmse": val_rmse,
                    "val_mae": val_mae,
                    "config": checkpoint_config,
                    "git_commit_hash": identity["git_commit"],
                    "timestamp": datetime.utcnow().isoformat(),
                }
                best_path = checkpoints_dir / "best_checkpoint.pt"
                atomic_torch_save(best_path, checkpoint)

            # Save training history atomically every epoch
            atomic_write_json(output_dir / "training_history.json", train_history)

            # Save last checkpoint every epoch
            checkpoint = {
                "schema_version": "fd001_checkpoint_v2",
                "checkpoint_type": "last",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mae": train_mae,
                "val_rmse": val_rmse,
                "val_mae": val_mae,
                "config": checkpoint_config,
                "git_commit_hash": identity["git_commit"],
                "timestamp": datetime.utcnow().isoformat(),
            }
            last_path = checkpoints_dir / "last_checkpoint.pt"
            atomic_torch_save(last_path, checkpoint)

            # Early stopping
            if early_stop_counter >= args.patience:
                training_status = "EARLY_STOPPED"
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

            if epoch % 10 == 0 or is_best:
                print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_rmse={val_rmse:.4f}, val_mae={val_mae:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}, best={is_best}")

    except Exception as e:
        training_status = "FAILED"
        raise

    # Locate checkpoints
    best_checkpoint_path = checkpoints_dir / "best_checkpoint.pt"
    last_checkpoint_path = checkpoints_dir / "last_checkpoint.pt"

    if not best_checkpoint_path.exists():
        raise RuntimeError("Best checkpoint not found")

    return training_status, best_checkpoint_path, last_checkpoint_path, train_history


def generate_predictions(
    best_checkpoint: Path,
    data_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    """Load best checkpoint, run inference on predictor_validation, write parquet."""
    device = args.device

    # Build validation dataloader
    dataloaders = build_dataloaders(
        data_dir=data_dir,
        sequence_length=args.sequence_length,
        rul_cap=args.rul_cap,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    val_loader = dataloaders[VALIDATION_SPLIT]

    # Load model
    n_features = val_loader.dataset.n_features
    model = build_predictor(
        model_type=args.model_type,
        n_features=n_features,
        sequence_length=args.sequence_length,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    model = model.to(device)

    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Run inference
    y_true_list = []
    y_pred_list = []
    unit_id_list = []
    cycle_list = []

    with torch.no_grad():
        for batch in val_loader:
            x = batch["features"].to(device)
            y = batch["rul_capped"].to(device)
            unit_ids = batch["unit_id"].numpy()
            cycles = batch["cycle"].numpy()

            y_pred = model(x)

            y_true_list.append(y.cpu().numpy())
            y_pred_list.append(y_pred.cpu().numpy())
            unit_id_list.append(unit_ids)
            cycle_list.append(cycles)

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    unit_ids = np.concatenate(unit_id_list)
    cycles = np.concatenate(cycle_list)

    # Build prediction DataFrame
    error = y_pred - y_true
    pred_df = pd.DataFrame({
        "split": [VALIDATION_SPLIT] * len(y_true),
        "unit_id": unit_ids.astype(np.int32),
        "cycle": cycles.astype(np.int32),
        "true_rul_capped": y_true.astype(np.float32),
        "predicted_rul": y_pred.astype(np.float32),
        "error": error.astype(np.float32),
    })

    # Validate
    is_valid, errors = validate_prediction_frame(pred_df)
    if not is_valid:
        raise ValidationError(f"Generated predictions invalid: {'; '.join(errors)}")

    # Write atomically
    predictions_path = output_dir / "predictor_validation_predictions.parquet"
    atomic_write_parquet(predictions_path, pred_df)

    return predictions_path


def compute_and_write_metrics(
    predictions_path: Path,
    val_cycle_table_path: Path,
    best_epoch: int,
    final_epoch: int,
    training_status: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """Load predictions + true RUL from validation cycle table, compute metrics, write JSON."""
    # The predictions already have true_rul_capped, so we can directly use them
    pred_df = pd.read_parquet(predictions_path)

    y_true = pred_df["true_rul_capped"].to_numpy(dtype=np.float32)
    y_pred = pred_df["predicted_rul"].to_numpy(dtype=np.float32)

    metrics = compute_formal_metrics(
        y_true=y_true,
        y_pred=y_pred,
        best_epoch=best_epoch,
        final_epoch=final_epoch,
        training_status=training_status,
    )

    # Write metrics atomically
    metrics_path = output_dir / "predictor_validation_metrics.json"
    write_metrics_json(metrics, metrics_path)

    return metrics


def write_all_artifacts(
    output_dir: Path,
    identity: Dict[str, Any],
    command_line: str,
    training_history: List[Dict],
    best_checkpoint_src: Path,
    last_checkpoint_src: Path,
    metrics: Dict[str, Any],
    predictions_path: Path,
    training_status: str,
    logging: RunnerLogging,
    args: argparse.Namespace,
) -> None:
    """Write all 14 authoritative artifacts atomically in correct order."""

    # 1. resolved_config.json
    resolved_config = {
        "config_path": str(args.config),
        "config_hash": compute_file_hash(args.config) if args.config.exists() else "unknown",
        "cli_overrides": {},
        "effective_config": {
            "seed": args.seed,
            "sequence_length": args.sequence_length,
            "rul_cap": args.rul_cap,
            "model": {
                "type": args.model_type,
                "hidden_dim": args.hidden_dim,
                "n_layers": args.n_layers,
                "dropout": args.dropout,
            },
            "training": {
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "max_epochs": args.max_epochs,
                "patience": args.patience,
            },
            "data": {
                "data_dir": str(args.data_dir),
                "train_split": TRAIN_SPLIT,
                "validation_split": VALIDATION_SPLIT,
            },
            "device": args.device,
            "loss": {
                "type": args.loss_type,
                "linex_a": args.linex_a,
                "linex_overflow_threshold": args.linex_overflow_threshold,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "git_commit": identity["git_commit"],
    }
    atomic_write_json(output_dir / "resolved_config.json", resolved_config)

    # 2. command.txt
    (output_dir / "command.txt").write_text(command_line + "\n")

    # 3. environment_identity.json
    env_identity = get_environment_identity()
    env_identity.update({
        "git_commit": identity["git_commit"],
        "git_branch": identity["git_branch"],
        "git_dirty": identity["git_dirty"],
    })
    atomic_write_json(output_dir / "environment_identity.json", env_identity)

    # 4. training_history.json (already written during training, ensure final)
    atomic_write_json(output_dir / "training_history.json", training_history)

    # 5. Copy best checkpoint to root
    atomic_copy_checkpoint(best_checkpoint_src, output_dir / "best_checkpoint.pt")

    # 6. Copy last checkpoint to root
    atomic_copy_checkpoint(last_checkpoint_src, output_dir / "last_checkpoint.pt")

    # 7. predictor_validation_predictions.parquet (already written)

    # 8. predictor_validation_metrics.json (already written)

    # 9. best_checkpoint.sha256
    best_hash = compute_file_hash(output_dir / "best_checkpoint.pt")
    atomic_write_sha256(output_dir / "best_checkpoint.sha256", best_hash, "best_checkpoint.pt")

    # 10. last_checkpoint.sha256
    last_hash = compute_file_hash(output_dir / "last_checkpoint.pt")
    atomic_write_sha256(output_dir / "last_checkpoint.sha256", last_hash, "last_checkpoint.pt")

    # 11. predictions.sha256
    pred_hash = compute_file_hash(predictions_path)
    atomic_write_sha256(output_dir / "predictions.sha256", pred_hash, predictions_path.name)

    # 12. predictor_metadata.json
    # Condition must match M8_FORMAL_COMMAND_MATRIX.json identity
    condition = get_formal_condition_id(args.loss_type, args.linex_a)

    metadata = {
        "predictor_id": f"m8_formal_{condition}_seed_{args.seed}",
        "condition": condition,
        "seed": args.seed,
        "best_val_rmse": min((h["val_rmse"] for h in training_history if h.get("is_best_so_far")), default=0.0),
        "best_epoch": metrics["best_epoch"],
        "final_epoch": metrics["final_epoch"],
        "training_status": training_status,
        "git_commit": identity["git_commit"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat() + "Z",
        "formal_metrics": metrics,
    }
    atomic_write_json(output_dir / "predictor_metadata.json", metadata)

    # 13. stdout_stderr.log (finalized by logging.finalize())
    logging.finalize()

    # 14. Terminal marker (COMPLETED or FAILED)
    write_terminal_marker(output_dir, training_status)


def write_failed_artifacts(
    output_dir: Path,
    identity: Dict[str, Any],
    command_line: str,
    training_history: List[Dict],
    failure_stage: str,
    error: Exception,
    completed_artifacts: List[str],
    logging: RunnerLogging,
    args: argparse.Namespace,
) -> None:
    """Write minimal FAILED artifacts for caught exceptions."""

    # 1. resolved_config.json
    resolved_config = {
        "config_path": str(args.config),
        "config_hash": compute_file_hash(args.config) if args.config.exists() else "unknown",
        "cli_overrides": {},
        "effective_config": {
            "seed": args.seed,
            "sequence_length": args.sequence_length,
            "rul_cap": args.rul_cap,
            "model": {
                "type": args.model_type,
                "hidden_dim": args.hidden_dim,
                "n_layers": args.n_layers,
                "dropout": args.dropout,
            },
            "training": {
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "max_epochs": args.max_epochs,
                "patience": args.patience,
            },
            "data": {
                "data_dir": str(args.data_dir),
                "train_split": TRAIN_SPLIT,
                "validation_split": VALIDATION_SPLIT,
            },
            "device": args.device,
            "loss": {
                "type": args.loss_type,
                "linex_a": args.linex_a,
                "linex_overflow_threshold": args.linex_overflow_threshold,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "git_commit": identity["git_commit"],
    }
    atomic_write_json(output_dir / "resolved_config.json", resolved_config)

    # 2. command.txt
    (output_dir / "command.txt").write_text(command_line + "\n")

    # 3. environment_identity.json
    env_identity = get_environment_identity()
    env_identity.update({
        "git_commit": identity["git_commit"],
        "git_branch": identity["git_branch"],
        "git_dirty": identity["git_dirty"],
    })
    atomic_write_json(output_dir / "environment_identity.json", env_identity)

    # 4. training_history.json
    atomic_write_json(output_dir / "training_history.json", training_history)

    # 5. predictor_metadata.json (failure schema)
    # Condition must match M8_FORMAL_COMMAND_MATRIX.json identity
    condition = get_formal_condition_id(args.loss_type, args.linex_a)

    metadata = {
        "status": "FAILED",
        "failure_stage": failure_stage,
        "condition": condition,
        "seed": args.seed,
        "loss_type": args.loss_type,
        "linex_a": args.linex_a,
        "git_identity": identity,
        "completed_artifacts": completed_artifacts,
        "missing_artifacts": [
            "best_checkpoint.pt", "last_checkpoint.pt",
            "predictor_validation_predictions.parquet",
            "predictor_validation_metrics.json",
            "best_checkpoint.sha256", "last_checkpoint.sha256", "predictions.sha256",
        ],
        "partial_artifacts": [],
    }
    atomic_write_json(output_dir / "predictor_metadata.json", metadata)

    # 6. stdout_stderr.log
    logging.finalize()

    # 7. FAILED marker
    write_terminal_marker(output_dir, "FAILED")

    # 8. failure_reason.txt (non-authoritative diagnostic)
    failure_reason = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "stage": failure_stage,
        "timestamp_utc": datetime.now(timezone.utc).isoformat() + "Z",
        "partial_artifacts": completed_artifacts,  # Only truly completed artifacts, not missing/partial
    }
    atomic_write_json(output_dir / "failure_reason.txt", failure_reason)


# ============================================================================
# Preflight Checks
# ============================================================================

def run_preflight(args: argparse.Namespace) -> Dict[str, Any]:
    """Run all preflight checks before creating leaf directory."""
    # 1. Validate git identity
    git_identity = get_git_identity()
    if git_identity["git_dirty"]:
        raise PreflightError(f"Git worktree is dirty: commit {git_identity['git_commit'][:12]}")

    # 2. Validate input hashes
    verify_input_hashes(args.data_dir)

    # 3. Validate val cycle table exists
    val_table_hash = verify_val_cycle_table(args.data_dir)

    # 4. Check output directory doesn't exist
    if args.output_dir.exists():
        raise PreflightError(f"Output directory already exists: {args.output_dir}")

    # 5. Check for stale/malformed temp files in output parent
    parent = args.output_dir.parent
    if parent.exists():
        for item in parent.iterdir():
            if item.name.endswith(".tmp.") or item.name.startswith("stdout_stderr.log.tmp."):
                # Classify the temp file
                classification = classify_temp_file(item.name)
                if classification == "MALFORMED_TEMP":
                    raise PreflightError(f"MALFORMED_TEMP: {item.name}")
                elif classification == "STALE_TEMP":
                    raise PreflightError(f"STALE_TEMP: {item.name}")
                elif classification == "ACTIVE_TEMP":
                    raise PreflightError(f"ACTIVE_TEMP: {item.name}")

    # 6. MPS lock acquisition (deferred until after leaf creation to match lifecycle)
    # Lock will be acquired in main() after output dir creation

    return {
        "git_identity": git_identity,
        "val_cycle_table_hash": val_table_hash,
    }


# ============================================================================
# Main Entry Point
# ============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for formal runner."""
    args = parse_args(argv)
    command_line = capture_command_line(args)

    # Ensure output dir is absolute
    output_dir = args.output_dir.resolve()
    args.output_dir = output_dir
    args.data_dir = args.data_dir.resolve()
    args.config = args.config.resolve()

    # Preflight (before leaf directory creation)
    try:
        preflight = run_preflight(args)
    except PreflightError as e:
        print(f"PRESTART_REJECTED: {e}", file=sys.stderr)
        return 2  # Preflight rejection exit code

    git_identity = preflight["git_identity"]

    # Acquire MPS heavy lock BEFORE MPS initialization
    from scripts.drl_heavy_mps_lock import acquire_mps_heavy_lock
    worktree = str(PROJECT_ROOT)
    mps_lock = acquire_mps_heavy_lock(
        milestone="M8",
        worktree=worktree,
        command=command_line,
    )

    # Create leaf output directory (after preflight, after lock)
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        mps_lock.release()
        raise PreflightError(f"Output directory created during preflight: {output_dir}")

    # Initialize runner-owned logging
    logging = RunnerLogging(output_dir)
    logging.setup()

    completed_artifacts = []
    training_history = []
    training_status = "FAILED"
    failure_stage = "initialization"
    error_to_report = None

    try:
        # Write initial artifacts that exist before training
        # resolved_config.json
        resolved_config = {
            "config_path": str(args.config),
            "config_hash": compute_file_hash(args.config) if args.config.exists() else "unknown",
            "cli_overrides": {},
            "effective_config": {
                "seed": args.seed,
                "sequence_length": args.sequence_length,
                "rul_cap": args.rul_cap,
                "model": {
                    "type": args.model_type,
                    "hidden_dim": args.hidden_dim,
                    "n_layers": args.n_layers,
                    "dropout": args.dropout,
                },
                "training": {
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "max_epochs": args.max_epochs,
                    "patience": args.patience,
                },
                "data": {
                    "data_dir": str(args.data_dir),
                    "train_split": TRAIN_SPLIT,
                    "validation_split": VALIDATION_SPLIT,
                },
                "device": args.device,
                "loss": {
                    "type": args.loss_type,
                    "linex_a": args.linex_a,
                    "linex_overflow_threshold": args.linex_overflow_threshold,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "git_commit": git_identity["git_commit"],
        }
        atomic_write_json(output_dir / "resolved_config.json", resolved_config)
        completed_artifacts.append("resolved_config.json")

        # command.txt
        (output_dir / "command.txt").write_text(command_line + "\n")
        completed_artifacts.append("command.txt")

        # environment_identity.json
        env_identity = get_environment_identity()
        env_identity.update({
            "git_commit": git_identity["git_commit"],
            "git_branch": git_identity["git_branch"],
            "git_dirty": git_identity["git_dirty"],
        })
        atomic_write_json(output_dir / "environment_identity.json", env_identity)
        completed_artifacts.append("environment_identity.json")

        # Run training
        failure_stage = "training"
        training_status, best_checkpoint_src, last_checkpoint_src, training_history = run_training(
            args=args,
            output_dir=output_dir,
            identity=git_identity,
            command_line=command_line,
            logging=logging,
        )
        completed_artifacts.append("training_history.json")

        # Copy checkpoints to root
        atomic_copy_checkpoint(best_checkpoint_src, output_dir / "best_checkpoint.pt")
        completed_artifacts.append("best_checkpoint.pt")

        atomic_copy_checkpoint(last_checkpoint_src, output_dir / "last_checkpoint.pt")
        completed_artifacts.append("last_checkpoint.pt")

        # Generate predictions
        failure_stage = "prediction"
        predictions_path = generate_predictions(
            best_checkpoint=best_checkpoint_src,
            data_dir=args.data_dir,
            output_dir=output_dir,
            args=args,
        )
        completed_artifacts.append("predictor_validation_predictions.parquet")

        # Compute and write metrics
        failure_stage = "metrics"
        val_cycle_table_path = args.data_dir / VAL_CYCLE_TABLE_REL

        # Derive best_epoch from the loaded best checkpoint (protocol uses 1-indexed epochs)
        best_checkpoint = torch.load(best_checkpoint_src, map_location="cpu", weights_only=False)
        best_epoch_0based = best_checkpoint.get("epoch", 0)
        best_epoch_1based = int(best_epoch_0based) + 1

        # Verify against training history
        history_best_epochs = [h["epoch"] for h in training_history if h.get("is_best_so_far")]
        if history_best_epochs:
            history_best = history_best_epochs[-1]
            if history_best != best_epoch_0based:
                raise RuntimeError(
                    f"Best epoch mismatch: checkpoint epoch={best_epoch_0based}, "
                    f"training history last is_best_so_far epoch={history_best}"
                )

        # Protocol uses 1-indexed epochs
        final_epoch_1based = len(training_history)  # len is already 1-based for count

        metrics = compute_and_write_metrics(
            predictions_path=predictions_path,
            val_cycle_table_path=val_cycle_table_path,
            best_epoch=best_epoch_1based,
            final_epoch=final_epoch_1based,
            training_status=training_status,
            output_dir=output_dir,
        )
        completed_artifacts.append("predictor_validation_metrics.json")

        # Write sidecars
        best_hash = compute_file_hash(output_dir / "best_checkpoint.pt")
        atomic_write_sha256(output_dir / "best_checkpoint.sha256", best_hash, "best_checkpoint.pt")
        completed_artifacts.append("best_checkpoint.sha256")

        last_hash = compute_file_hash(output_dir / "last_checkpoint.pt")
        atomic_write_sha256(output_dir / "last_checkpoint.sha256", last_hash, "last_checkpoint.pt")
        completed_artifacts.append("last_checkpoint.sha256")

        pred_hash = compute_file_hash(predictions_path)
        atomic_write_sha256(output_dir / "predictions.sha256", pred_hash, predictions_path.name)
        completed_artifacts.append("predictions.sha256")

        # predictor_metadata.json
        # Condition must match M8_FORMAL_COMMAND_MATRIX.json identity
        condition = get_formal_condition_id(args.loss_type, args.linex_a)

        metadata = {
            "predictor_id": f"m8_formal_{condition}_seed_{args.seed}",
            "condition": condition,
            "seed": args.seed,
            "best_val_rmse": min((h["val_rmse"] for h in training_history if h.get("is_best_so_far")), default=0.0),
            "best_epoch": metrics["best_epoch"],
            "final_epoch": metrics["final_epoch"],
            "training_status": training_status,
            "git_commit": git_identity["git_commit"],
            "timestamp_utc": datetime.now(timezone.utc).isoformat() + "Z",
            "formal_metrics": metrics,
        }
        atomic_write_json(output_dir / "predictor_metadata.json", metadata)
        completed_artifacts.append("predictor_metadata.json")

        # Finalize logging and write terminal marker
        write_all_artifacts(
            output_dir=output_dir,
            identity=git_identity,
            command_line=command_line,
            training_history=training_history,
            best_checkpoint_src=best_checkpoint_src,
            last_checkpoint_src=last_checkpoint_src,
            metrics=metrics,
            predictions_path=predictions_path,
            training_status=training_status,
            logging=logging,
            args=args,
        )

        print(f"\nFormal run {training_status}: {output_dir}")
        return 0

    except Exception as e:
        error_to_report = e
        training_status = "FAILED"
        print(f"\nRun FAILED at {failure_stage}: {e}", file=sys.stderr)

        try:
            write_failed_artifacts(
                output_dir=output_dir,
                identity=git_identity,
                command_line=command_line,
                training_history=training_history,
                failure_stage=failure_stage,
                error=e,
                completed_artifacts=completed_artifacts,
                logging=logging,
                args=args,
            )
        except Exception as cleanup_error:
            # If we can't even write FAILED artifacts, it's INCOMPLETE
            print(f"CRITICAL: Failed to write FAILED artifacts: {cleanup_error}", file=sys.stderr)
            return 3  # INCOMPLETE

        return 1  # FAILED

    finally:
        # Always release MPS lock
        mps_lock.release()


if __name__ == "__main__":
    sys.exit(main())