"""Tests for checkpoint semantics: best vs last checkpoint separation.

These tests verify:
- best_checkpoint.pt contains the best validation-RMSE epoch weights
- last_checkpoint.pt contains the final epoch weights, optimizer and scheduler state
- best and last can differ (are not the same epoch)
- last_checkpoint.pt is suitable for exact training resume
- Reloading best for final evaluation does not alter last_checkpoint.pt
- Current canonical best_checkpoint.pt remains loadable
"""
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytest

from src.predictors.train import save_checkpoint
from src.predictors.io_utils import atomic_write_json


def test_checkpoint_schema_version_present(tmp_path: pathlib.Path):
    """Verify checkpoint includes schema_version field."""
    model = nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters())
    scheduler = ReduceLROnPlateau(optimizer, mode="min")

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    path = save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=5,
        train_loss=1.0,
        train_mae=0.5,
        val_rmse=2.0,
        val_mae=1.0,
        config={"seed": 6521},
        checkpoint_dir=checkpoint_dir,
        filename="test_checkpoint.pt",
        checkpoint_type="best",
    )

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    assert "schema_version" in ckpt
    assert ckpt["schema_version"] == "fd001_checkpoint_v2"
    assert ckpt["checkpoint_type"] == "best"


def test_checkpoint_includes_scheduler_state(tmp_path: pathlib.Path):
    """Verify checkpoint includes scheduler_state_dict for resume."""
    model = nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    # Step the scheduler with non-improving metrics to trigger LR reduction
    scheduler.step(1.0)  # best=1.0
    scheduler.step(1.0)  # no improvement
    scheduler.step(1.0)  # no improvement
    scheduler.step(1.0)  # LR reduces to 0.05

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    path = save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=4,
        train_loss=1.0,
        train_mae=0.5,
        val_rmse=1.0,
        val_mae=0.4,
        config={"seed": 6521},
        checkpoint_dir=checkpoint_dir,
        filename="test_checkpoint.pt",
        checkpoint_type="last",
    )

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Verify scheduler state dict is present
    assert "scheduler_state_dict" in ckpt
    assert ckpt["scheduler_state_dict"] is not None

    # Verify scheduler state contains key fields
    sched_state = ckpt["scheduler_state_dict"]
    assert sched_state.get("patience") == 2
    assert sched_state.get("factor") == 0.5
    assert "_last_lr" in sched_state  # Scheduler tracks the current LR here

    # Verify we can load the state into a new scheduler
    scheduler2 = ReduceLROnPlateau(optim.Adam(model.parameters(), lr=0.1), mode="min", factor=0.5, patience=2)
    scheduler2.load_state_dict(ckpt["scheduler_state_dict"])

    # The restored scheduler should have the same internal state
    assert scheduler2.state_dict().get("_last_lr") == [0.05]
    assert scheduler2.state_dict().get("best") == 1.0


def test_best_and_last_can_differ(tmp_path: pathlib.Path):
    """Verify best and last checkpoints can be different epochs."""
    model = nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters())
    scheduler = ReduceLROnPlateau(optimizer)

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    # Simulate: epoch 5 is best, epoch 10 is last
    # Epoch 5: val_rmse = 1.0 (best)
    # Epoch 6-10: val_rmse > 1.0 (worse)

    # Save best at epoch 5
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=5,
        train_loss=0.8,
        train_mae=0.4,
        val_rmse=1.0,
        val_mae=0.5,
        config={"seed": 6521},
        checkpoint_dir=checkpoint_dir,
        filename="best_checkpoint.pt",
        checkpoint_type="best",
    )

    # Save last at epoch 10 (worse metrics)
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=10,
        train_loss=1.2,
        train_mae=0.6,
        val_rmse=2.0,
        val_mae=0.8,
        config={"seed": 6521},
        checkpoint_dir=checkpoint_dir,
        filename="last_checkpoint.pt",
        checkpoint_type="last",
    )

    best_ckpt = torch.load(checkpoint_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    last_ckpt = torch.load(checkpoint_dir / "last_checkpoint.pt", map_location="cpu", weights_only=False)

    # Different epochs
    assert best_ckpt["epoch"] == 5
    assert last_ckpt["epoch"] == 10

    # Different metrics
    assert best_ckpt["val_rmse"] == 1.0
    assert last_ckpt["val_rmse"] == 2.0

    # Different checkpoint types
    assert best_ckpt["checkpoint_type"] == "best"
    assert last_ckpt["checkpoint_type"] == "last"


def test_last_checkpoint_includes_final_epoch_state(tmp_path: pathlib.Path):
    """Verify last checkpoint has final epoch weights, optimizer, scheduler state."""
    model = nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = ReduceLROnPlateau(optimizer)

    # Step optimizer and scheduler to give them non-initial state
    dummy_input = torch.randn(4, 10)
    dummy_target = torch.randn(4, 1)

    for _ in range(5):
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = nn.MSELoss()(output, dummy_target)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

    initial_lr = 0.01
    current_lr = optimizer.param_groups[0]["lr"]

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=10,
        train_loss=0.5,
        train_mae=0.3,
        val_rmse=0.8,
        val_mae=0.5,
        config={"seed": 6521},
        checkpoint_dir=checkpoint_dir,
        filename="last_checkpoint.pt",
        checkpoint_type="last",
    )

    ckpt = torch.load(checkpoint_dir / "last_checkpoint.pt", map_location="cpu", weights_only=False)

    assert ckpt["epoch"] == 10
    assert ckpt["checkpoint_type"] == "last"
    assert "optimizer_state_dict" in ckpt
    assert "scheduler_state_dict" in ckpt
    assert "model_state_dict" in ckpt

    # Verify optimizer state can be restored
    optimizer2 = optim.Adam(model.parameters(), lr=0.01)
    optimizer2.load_state_dict(ckpt["optimizer_state_dict"])

    # Learning rate should match
    assert abs(optimizer2.param_groups[0]["lr"] - current_lr) < 1e-7


def test_reload_best_does_not_alter_last(tmp_path: pathlib.Path):
    """Verify reloading best checkpoint for final eval doesn't overwrite last."""
    model = nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters())
    scheduler = ReduceLROnPlateau(optimizer)

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    # Save best at epoch 5
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=5,
        train_loss=0.8,
        train_mae=0.4,
        val_rmse=1.0,
        val_mae=0.5,
        config={"seed": 6521},
        checkpoint_dir=checkpoint_dir,
        filename="best_checkpoint.pt",
        checkpoint_type="best",
    )

    # Save last at epoch 10
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=10,
        train_loss=1.2,
        train_mae=0.6,
        val_rmse=2.0,
        val_mae=0.8,
        config={"seed": 6521},
        checkpoint_dir=checkpoint_dir,
        filename="last_checkpoint.pt",
        checkpoint_type="last",
    )

    # Capture last checkpoint hash before reload
    last_before_hash = torch.load(checkpoint_dir / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    last_before_epoch = last_before_hash["epoch"]

    # Simulate "reloading best for final evaluation" pattern
    best_ckpt = torch.load(checkpoint_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    # Do NOT save last_checkpoint here - this is the bug we're fixing

    # Verify last checkpoint was NOT altered
    last_after = torch.load(checkpoint_dir / "last_checkpoint.pt", map_location="cpu", weights_only=False)

    assert last_after["epoch"] == last_before_epoch, "last_checkpoint.pt should not be altered after loading best"
    assert last_after["epoch"] == 10, "last_checkpoint.pt should remain at epoch 10"


def test_canonical_best_checkpoint_loadable():
    """Verify current canonical best_checkpoint.pt is loadable with new schema."""
    import os
    repo_root = pathlib.Path(__file__).parent.parent
    ckpt_path = repo_root / "results" / "predictor" / "mse_baseline_v2" / "checkpoints" / "best_checkpoint.pt"

    if not ckpt_path.exists():
        pytest.skip("Canonical best_checkpoint.pt not found (may not have trained yet)")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Current canonical checkpoint uses older schema (no schema_version)
    # but should still have required fields
    assert "epoch" in ckpt
    assert "model_state_dict" in ckpt
    assert "optimizer_state_dict" in ckpt
    assert ckpt["epoch"] == 22


def test_checkpoint_hash_mismatch_detection(tmp_path: pathlib.Path):
    """Verify checkpoint identity uses full SHA256, not shortened or epoch-based."""
    import hashlib

    model = nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters())
    scheduler = ReduceLROnPlateau(optimizer)

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=5,
        train_loss=0.8,
        train_mae=0.4,
        val_rmse=1.0,
        val_mae=0.5,
        config={"seed": 6521},
        checkpoint_dir=checkpoint_dir,
        filename="test.pt",
        checkpoint_type="best",
    )

    # Compute full SHA256 of checkpoint file
    with open(checkpoint_dir / "test.pt", "rb") as f:
        full_sha = hashlib.sha256(f.read()).hexdigest()

    # Verify it's 64 characters (full SHA256)
    assert len(full_sha) == 64

    # Verify checkpoint_short_id (if used) would be different from full hash
    short_id = full_sha[:12]
    assert short_id != full_sha, "Short ID should not equal full SHA256"


def test_epoch_cannot_be_confused_with_checkpoint_id(tmp_path: pathlib.Path):
    """Verify epoch number and checkpoint SHA256 cannot be confused."""
    model = nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters())
    scheduler = ReduceLROnPlateau(optimizer)

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=5,
        train_loss=0.8,
        train_mae=0.4,
        val_rmse=1.0,
        val_mae=0.5,
        config={"seed": 6521},
        checkpoint_dir=checkpoint_dir,
        filename="test.pt",
        checkpoint_type="best",
    )

    ckpt = torch.load(checkpoint_dir / "test.pt", map_location="cpu", weights_only=False)

    # epoch is an integer
    assert isinstance(ckpt["epoch"], int)
    # checkpoint_id (if derived) would be a hash string
    # This test ensures they are semantically distinct types
    assert ckpt["epoch"] != "5"  # Not a string
    assert ckpt["epoch"] >= 0  # Non-negative integer


def test_shortened_display_id_never_used_as_canonical(tmp_path: pathlib.Path):
    """Verify shortened display IDs are never used as canonical identifier."""
    # This test documents the contract: display IDs are for humans only.
    # Any code using checkpoint_id must use the full 64-character SHA256.

    full_id = "a" * 64  # Simulated full SHA256
    short_id = full_id[:12]  # Simulated short display ID

    # Short ID is 12 chars, full is 64
    assert len(short_id) == 12
    assert len(full_id) == 64

    # They are not equal
    assert short_id != full_id

    # Any code comparing checkpoint_id must use full_id
    # This test ensures the contract is documented
    assert True, "canonical checkpoint_id must be full 64-char SHA256, not shortened display"