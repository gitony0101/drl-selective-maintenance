'''Training script for baseline MSE RUL predictor.

Trains a predictor using predictor_train split, validates on predictor_validation,
and saves the best checkpoint based on validation RMSE.

The JSON config (configs/predictor/mse_baseline.json) is the authoritative source.
CLI overrides are allowed but recorded in resolved_config.json.
'''

import argparse
import json
import hashlib
import os
import platform
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predictors.dataset import FD001SequenceDataset, build_dataloaders
from src.predictors.model import build_predictor
from src.predictors.losses import build_loss_fn
# Atomic write utilities
from src.predictors.io_utils import atomic_write_json, atomic_torch_save

# Default config path
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "predictor" / "mse_baseline.json"


def get_device() -> str:
    """Get available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from JSON file.

    Args:
        config_path: Path to config JSON file

    Returns:
        Configuration dictionary
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        return json.load(f)


def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Merge config with CLI args, recording overrides.

    CLI args override config values. Returns merged dict with _overrides recorded.
    """
    merged = {}
    overrides = {}

    # Map CLI arg names to config keys
    arg_to_config = {
        "seed": "seed",
        "sequence_length": "sequence_length",
        "rul_cap": "rul_cap",
        "model_type": "model.type",
        "hidden_dim": "model.hidden_dim",
        "n_layers": "model.n_layers",
        "dropout": "model.dropout",
        "batch_size": "training.batch_size",
        "learning_rate": "training.learning_rate",
        "weight_decay": "training.weight_decay",
        "max_epochs": "training.max_epochs",
        "patience": "training.patience",
        "data_dir": "data.data_dir",
        "output_dir": "_output_dir",  # Special case
        "device": "device",
    }

    for arg_name, config_key in arg_to_config.items():
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            # Record override
            if config_key.startswith("model."):
                key = config_key
                original = config.get("model", {}).get(key.split(".")[1], None)
            elif config_key.startswith("training."):
                key = config_key
                original = config.get("training", {}).get(key.split(".")[1], None)
            elif config_key.startswith("data."):
                key = config_key
                original = config.get("data", {}).get(key.split(".")[1], None)
            else:
                key = config_key
                original = config.get(key, None)

            if arg_value != original:
                overrides[key] = {"original": original, "override": arg_value}

    merged["_overrides"] = overrides
    return merged


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute evaluation metrics.

    Args:
        y_true: True RUL values
        y_pred: Predicted RUL values

    Returns:
        Dict with rmse, mae, mape (if applicable)
    """
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    mae = np.mean(np.abs(y_pred - y_true))

    # MAPE (avoid division by zero)
    mask = y_true > 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100
    else:
        mape = float("inf")

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, int, float]:
    """Train for one epoch.

    Train MAE is accumulated from the same per-batch predictions and targets
    already produced for the optimization step — no additional full pass over
    the training set is performed. The accumulation uses detached tensors so
    no computation graph is retained and gradients/optimizer behavior are
    unaffected.

    Returns:
        (total_loss, n_samples, train_mae) where ``train_mae`` is the
        sample-weighted mean absolute error over the whole epoch (a native
        Python float, never a batch-count average).
    """
    model.train()
    total_loss = 0.0
    n_samples = 0
    abs_error_sum = 0.0  # sample-weighted sum of |y_pred - y|

    for batch in loader:
        x = batch["features"].to(device)  # (batch, seq_len, n_features)
        y = batch["rul_capped"].to(device)  # (batch,)

        optimizer.zero_grad()
        y_pred = model(x)

        loss = criterion(y_pred, y)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Metric accumulation uses detached tensors only — must not touch the
        # autograd graph that backward()/optimizer.step() just consumed. `y`
        # is a 1-D vector per the loader contract (batch,). Count samples by
        # the number of target elements, not by the number of batches.
        n_batch = int(y.numel())
        total_loss += loss.item() * n_batch
        n_samples += n_batch
        abs_error_sum += float(
            torch.sum(torch.abs(y_pred.detach() - y.detach()))
            .to("cpu")
        )

    train_mae = abs_error_sum / n_samples if n_samples > 0 else 0.0
    return total_loss / n_samples, n_samples, float(train_mae)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate model.

    Returns:
        (avg_loss, y_true, y_pred)
    """
    model.eval()
    total_loss = 0.0
    y_true_list = []
    y_pred_list = []

    for batch in loader:
        x = batch["features"].to(device)
        y = batch["rul_capped"].to(device)

        y_pred = model(x)
        loss = criterion(y_pred, y)

        total_loss += loss.item() * len(y)
        y_true_list.append(y.cpu().numpy())
        y_pred_list.append(y_pred.cpu().numpy())

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)

    return total_loss / len(y_true), y_true, y_pred


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler.ReduceLROnPlateau],
    epoch: int,
    train_loss: float,
    train_mae: float,
    val_rmse: float,
    val_mae: float,
    config: Dict[str, Any],
    checkpoint_dir: Path,
    filename: str = "checkpoint.pt",
    checkpoint_type: str = "best",
) -> Path:
    """Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler (ReduceLROnPlateau)
        epoch: Current epoch
        train_loss: Training loss
        train_mae: Training MAE
        val_rmse: Validation RMSE
        val_mae: Validation MAE
        config: Training configuration
        checkpoint_dir: Directory to save checkpoint
        filename: Checkpoint filename
        checkpoint_type: "best" for best validation RMSE, "last" for final epoch

    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Get git commit hash if available
    try:
        import subprocess
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        commit_hash = "unknown"

    checkpoint = {
        "schema_version": "fd001_checkpoint_v2",
        "checkpoint_type": checkpoint_type,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "train_loss": train_loss,
        "train_mae": train_mae,
        "val_rmse": val_rmse,
        "val_mae": val_mae,
        "config": config,
        "git_commit_hash": commit_hash,
        "timestamp": datetime.utcnow().isoformat(),
    }

    checkpoint_path = checkpoint_dir / filename
    # Atomic save
    atomic_torch_save(checkpoint_path, checkpoint)

    return checkpoint_path


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def train_predictor(
    data_dir: Path,
    output_dir: Path,
    seed: int = 6521,
    sequence_length: int = 50,
    rul_cap: int = 125,
    model_type: str = "mlp",
    hidden_dim: int = 128,
    n_layers: int = 3,
    dropout: float = 0.2,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 200,
    patience: int = 20,
    device: Optional[str] = None,
    loss_type: str = "mse",
    linex_a: float = 0.1,
    linex_overflow_threshold: float = 20.0,
) -> Dict[str, Any]:
    """Train RUL predictor.

    Args:
        data_dir: Path to FD001 V2 processed directory
        output_dir: Path to output directory for checkpoints and logs
        seed: Random seed
        sequence_length: Sequence window length
        rul_cap: RUL cap
        model_type: "mlp" or "cnn"
        hidden_dim: Hidden dimension
        n_layers: Number of layers
        dropout: Dropout rate
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay
        max_epochs: Maximum epochs
        patience: Early stopping patience
        device: Device to use (default: auto-detect)
        loss_type: "mse" or "linex"
        linex_a: LinEx asymmetry parameter (a > 0)
        linex_overflow_threshold: LinEx overflow threshold

    Returns:
        Training results dict
    """
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = get_device()

    print(f"Using device: {device}")

    # Build dataloaders
    print("Building dataloaders...")
    dataloaders = build_dataloaders(
        data_dir=data_dir,
        sequence_length=sequence_length,
        rul_cap=rul_cap,
        batch_size=batch_size,
        seed=seed,
    )

    if "predictor_train" not in dataloaders:
        raise ValueError("No predictor_train data found")
    if "predictor_validation" not in dataloaders:
        raise ValueError("No predictor_validation data found")

    train_loader = dataloaders["predictor_train"]
    val_loader = dataloaders["predictor_validation"]

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Build model
    n_features = train_loader.dataset.n_features
    print(f"Building {model_type} model with {n_features} features...")

    model = build_predictor(
        model_type=model_type,
        n_features=n_features,
        sequence_length=sequence_length,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
    )

    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Loss and optimizer
    if loss_type == "mse":
        criterion = build_loss_fn(loss_type="mse")
    else:
        criterion = build_loss_fn(
            loss_type="linex",
            a=linex_a,
            overflow_threshold=linex_overflow_threshold,
        )
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=patience // 2,
        min_lr=1e-6,
    )

    # Training loop
    print(f"\nTraining for {max_epochs} epochs (patience={patience})...")

    best_val_rmse = float("inf")
    best_epoch = 0
    early_stop_counter = 0
    train_history = []

    # Canonical checkpoint config, shared by best/last/per-epoch checkpoints so
    # the V2 cache generator's safety guards accept the trained artifact.
    checkpoint_config = {
        "seed": seed,
        "sequence_length": sequence_length,
        "rul_cap": rul_cap,
        "model_type": model_type,
        "n_features": n_features,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "dropout": dropout,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "loss_type": loss_type,
        "linex_a": linex_a,
        "linex_overflow_threshold": linex_overflow_threshold,
        "normalizer_id": "fd001_normalizer_v2",
        "feature_schema_id": "fd001_feature_schema_v1",
    }

    import time

    for epoch in range(max_epochs):
        epoch_start = time.time()

        # Train
        train_loss, n_train, train_mae = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        train_rmse = np.sqrt(train_loss)
        # `train_mae` is the real sample-weighted training-mode MAE accumulated
        # during the optimization pass above — no 0.0 placeholder.

        # Validate
        val_loss, y_true, y_pred = evaluate(
            model, val_loader, criterion, device
        )
        val_metrics = compute_metrics(y_true, y_pred)
        val_rmse = val_metrics["rmse"]
        val_mae = val_metrics["mae"]
        val_mape = val_metrics["mape"]

        epoch_duration = time.time() - epoch_start

        # Update scheduler
        scheduler.step(val_rmse)

        # Check if best
        is_best = val_rmse < best_val_rmse
        if is_best:
            best_val_rmse = val_rmse
            best_epoch = epoch
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Log with all required metrics — every value is cast to a builtin
        # Python scalar so ``json.dump`` never encounters a numpy scalar
        # (``numpy.float64``, ``numpy.bool_``, etc.) that is not natively
        # JSON-serializable.
        train_history.append({
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "train_rmse": float(train_rmse),
            "train_mae": float(train_mae),
            "val_loss": float(val_loss),
            "val_rmse": float(val_rmse),
            "val_mae": float(val_mae),
            "val_mape": float(val_mape),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_duration_seconds": float(epoch_duration),
            "is_best_so_far": bool(is_best),
            "early_stopping_counter": int(early_stop_counter),
        })

        if epoch % 10 == 0 or is_best:
            print(
                f"Epoch {epoch:3d}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"val_rmse={val_rmse:.4f}, "
                f"val_mae={val_mae:.4f}, "
                f"lr={optimizer.param_groups[0]['lr']:.6f}, "
                f"best={is_best}"
            )

        # Save best checkpoint atomically (only when validation RMSE improves)
        if is_best:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_loss=train_loss,
                train_mae=train_mae,
                val_rmse=val_rmse,
                val_mae=val_mae,
                config=checkpoint_config,
                checkpoint_dir=output_dir / "checkpoints",
                filename="best_checkpoint.pt",
                checkpoint_type="best",
            )

        # Per-epoch atomic persistence: the complete training history and the
        # last checkpoint are written after EVERY completed epoch so a crash or
        # interrupt never leaves an inconsistent or stale training_history.json
        # or last_checkpoint.pt. These must live inside the epoch loop, not
        # after it. Public training_history.json also keeps the canonical best
        # markers current so downstream manifest validation never sees a stale
        # record mid-training.
        atomic_write_json(output_dir / "training_history.json", train_history)

        # Save last checkpoint at every epoch - this is the authoritative
        # resumable checkpoint with final epoch weights, optimizer and scheduler state
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            train_loss=train_loss,
            train_mae=train_mae,
            val_rmse=val_rmse,
            val_mae=val_mae,
            config=checkpoint_config,
            checkpoint_dir=output_dir / "checkpoints",
            filename="last_checkpoint.pt",
            checkpoint_type="last",
        )

        # Early stopping
        if early_stop_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    print(f"\nTraining complete!")
    print(f"Best validation RMSE: {best_val_rmse:.4f} at epoch {best_epoch}")

    # Compute final train metrics at best checkpoint
    print("\nReloading best checkpoint to compute final metrics...")
    best_checkpoint = torch.load(
        output_dir / "checkpoints" / "best_checkpoint.pt",
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()

    # Compute metrics on full train and validation sets
    _, train_y_true, train_y_pred = evaluate(model, train_loader, criterion, device)
    _, val_y_true, val_y_pred = evaluate(model, val_loader, criterion, device)

    train_metrics = compute_metrics(train_y_true, train_y_pred)
    val_metrics = compute_metrics(val_y_true, val_y_pred)

    # Note: per-epoch train_mae already holds the real sample-weighted
    # training-mode MAE accumulated inside train_epoch, so there is no longer
    # any need to overwrite the best epoch's train_mae here. The evaluation-mode
    # MAE computed from the reloaded best checkpoint above (train_metrics["mae"])
    # is reported separately in the final summary/metadata — it is a
    # deterministic eval-pass metric and differs numerically from the
    # training-mode MAE because train_epoch accumulates under model.train()
    # using the predictions seen during optimization, while this is model.eval()
    # on the saved best weights. The two are both legitimate but semantically
    # distinct; the history keeps the training-mode value for every epoch.

    # Validate history integrity
    epochs_trained = len(train_history)
    assert train_history[best_epoch]["epoch"] == best_epoch, "Best epoch not in history"
    assert abs(train_history[best_epoch]["val_rmse"] - best_val_rmse) < 1e-4, \
        f"Stored best RMSE {best_val_rmse} != history[{best_epoch}] RMSE {train_history[best_epoch]['val_rmse']}"

    print(f"Epochs trained: {epochs_trained}")
    print(f"Best epoch: {best_epoch}")
    print(f"Final train RMSE: {train_metrics['rmse']:.4f}")
    print(f"Final train MAE: {train_metrics['mae']:.4f}")
    print(f"Final val RMSE: {val_metrics['rmse']:.4f}")
    print(f"Final val MAE: {val_metrics['mae']:.4f}")

    # Save training history atomically
    history_path = output_dir / "training_history.json"
    atomic_write_json(history_path, train_history)
    print(f"Saved training history: {history_path}")

    # Save training summary atomically
    summary = {
        "epochs_trained": epochs_trained,
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "best_val_mae": val_metrics["mae"],
        "final_train_rmse": train_metrics["rmse"],
        "final_train_mae": train_metrics["mae"],
        "final_val_rmse": val_metrics["rmse"],
        "final_val_mae": val_metrics["mae"],
        "early_stopping_patience": patience,
        "stopped_early": early_stop_counter >= patience,
    }
    summary_path = output_dir / "training_summary.json"
    atomic_write_json(summary_path, summary)
    print(f"Saved training summary: {summary_path}")

    # Save final model metadata atomically
    metadata = {
        "predictor_id": f"fd001_{loss_type}_baseline_v2_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "version": "v2",
        "description": "Retrained baseline after normalization fix",
        "seed": seed,
        "sequence_length": sequence_length,
        "rul_cap": rul_cap,
        "model_type": model_type,
        "n_features": n_features,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "dropout": dropout,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_epochs": max_epochs,
        "patience": patience,
        "loss_type": loss_type,
        "linex_a": linex_a if loss_type == "linex" else None,
        "linex_overflow_threshold": linex_overflow_threshold if loss_type == "linex" else None,
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "best_val_mae": val_metrics["mae"],
        "final_train_rmse": train_metrics["rmse"],
        "device": device,
        "n_parameters": n_params,
        "git_commit_hash": get_git_commit(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "feature_schema_id": "fd001_feature_schema_v1",
        "normalizer_id": "fd001_normalizer_v2",
        "split_manifest_id": "fd001_unit_split_v1",
    }
    metadata_path = output_dir / "predictor_metadata.json"
    atomic_write_json(metadata_path, metadata)
    print(f"Saved predictor metadata: {metadata_path}")

    # NOTE: last_checkpoint.pt is already saved at every epoch inside the
    # training loop above. Do NOT save it again here - the model was reloaded
    # from best_checkpoint.pt for final evaluation, so calling save_checkpoint
    # now would overwrite last_checkpoint.pt with the best weights instead of
    # the final epoch weights. The last checkpoint saved in the loop is the
    # authoritative resumable checkpoint.

    return {
        "metadata": metadata,
        "train_history": train_history,
        "best_val_rmse": best_val_rmse,
        "best_epoch": best_epoch,
        "checkpoint_path": output_dir / "checkpoints" / "best_checkpoint.pt",
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }


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


def main():
    """Main entry point.

    Loads configuration from configs/predictor/mse_baseline.json as authoritative.
    CLI arguments override config values and are recorded in resolved_config.json.
    """
    parser = argparse.ArgumentParser(description="Train FD001 RUL Predictor")

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to config JSON file (default: configs/predictor/mse_baseline.json)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to FD001 V2 processed directory (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for checkpoints and logs (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="Sequence window length (overrides config)",
    )
    parser.add_argument(
        "--rul-cap",
        type=int,
        default=None,
        help="RUL cap value (overrides config)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["mlp", "cnn"],
        help="Model architecture (overrides config)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Hidden dimension (overrides config)",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=None,
        help="Number of hidden layers (overrides config)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout rate (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay (overrides config)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Maximum epochs (overrides config)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience (overrides config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: cpu, cuda, mps, or auto (overrides config)",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default=None,
        choices=["mse", "linex"],
        help="Loss function type (overrides config)",
    )
    parser.add_argument(
        "--linex-a",
        type=float,
        default=None,
        help="LinEx asymmetry parameter a > 0 (overrides config)",
    )
    parser.add_argument(
        "--linex-overflow-threshold",
        type=float,
        default=None,
        help="LinEx overflow threshold (overrides config)",
    )

    args = parser.parse_args()

    # Load authoritative config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Extract config values (with defaults from nested structure)
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

    # Loss configuration
    default_loss = config.get("loss", {})
    default_loss_type = default_loss.get("type", "mse")
    default_linex_a = default_loss.get("linex_a", 0.1)
    default_linex_overflow = default_loss.get("linex_overflow_threshold", 20.0)

    # Apply CLI overrides
    data_dir = args.data_dir if args.data_dir else default_data_dir
    output_dir = args.output_dir if args.output_dir else default_output_dir
    seed = args.seed if args.seed is not None else default_seed
    sequence_length = args.sequence_length if args.sequence_length is not None else default_seq_len
    rul_cap = args.rul_cap if args.rul_cap is not None else default_rul_cap
    model_type = args.model_type if args.model_type else default_model_type
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else default_hidden_dim
    n_layers = args.n_layers if args.n_layers is not None else default_n_layers
    dropout = args.dropout if args.dropout is not None else default_dropout
    batch_size = args.batch_size if args.batch_size is not None else default_batch_size
    learning_rate = args.learning_rate if args.learning_rate is not None else default_lr
    weight_decay = args.weight_decay if args.weight_decay is not None else default_weight_decay
    max_epochs = args.max_epochs if args.max_epochs is not None else default_max_epochs
    patience = args.patience if args.patience is not None else default_patience
    device = args.device if args.device else default_device
    loss_type = args.loss_type if args.loss_type is not None else default_loss_type
    linex_a = args.linex_a if args.linex_a is not None else default_linex_a
    linex_overflow_threshold = args.linex_overflow_threshold if args.linex_overflow_threshold is not None else default_linex_overflow

    print("=" * 60)
    print("FD001 Baseline MSE RUL Predictor Training (V2 - Normalization Fixed)")
    print("=" * 60)
    print(f"Config file: {args.config}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Seed: {seed}")
    print(f"Sequence length: {sequence_length}")
    print(f"RUL cap: {rul_cap}")
    print(f"Model type: {model_type}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Num layers: {n_layers}")
    print(f"Dropout: {dropout}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Max epochs: {max_epochs}")
    print(f"Patience: {patience}")
    print(f"Device: {device}")
    print()

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config with overrides recorded (atomic)
    resolved_config = {
        "source_config_path": str(args.config),
        "source_config_hash": compute_file_hash(args.config) if args.config.exists() else "unknown",
        "seed": seed,
        "sequence_length": sequence_length,
        "rul_cap": rul_cap,
        "model": {
            "type": model_type,
            "n_features": 24,  # Will be confirmed by dataset
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "dropout": dropout,
        },
        "training": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "max_epochs": max_epochs,
            "patience": patience,
            "gradient_clipping": config.get("training", {}).get("gradient_clipping", 1.0),
            "early_stopping": config.get("training", {}).get("early_stopping", True),
        },
        "data": {
            "data_dir": str(data_dir),
            "train_split": "predictor_train",
            "validation_split": "predictor_validation",
        },
        "device": device,
        "loss": {
            "type": loss_type,
            "linex_a": linex_a,
            "linex_overflow_threshold": linex_overflow_threshold,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    resolved_config_path = output_dir / "resolved_config.json"
    atomic_write_json(resolved_config_path, resolved_config)
    print(f"Saved resolved config: {resolved_config_path}")

    # Train
    results = train_predictor(
        data_dir=Path(data_dir),
        output_dir=Path(output_dir),
        seed=seed,
        sequence_length=sequence_length,
        rul_cap=rul_cap,
        model_type=model_type,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        patience=patience,
        device=None if device == "auto" else device,
        loss_type=loss_type,
        linex_a=linex_a,
        linex_overflow_threshold=linex_overflow_threshold,
    )

    print()
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best validation RMSE: {results['best_val_rmse']:.4f}")
    print(f"Best epoch: {results['best_epoch']}")
    print(f"Checkpoint path: {results['checkpoint_path']}")
    print(f"Predictor ID: {results['metadata']['predictor_id']}")
    print(f"Git commit: {results['metadata']['git_commit_hash'][:12]}...")
    print()
    print("Milestone 1 V2 training complete. Ready for cache regeneration.")


if __name__ == "__main__":
    main()
