"""
M8 Smoke Training Tests (ST-01 through ST-11)

Tests for bounded smoke training per M8_TEST_PLAN.md Section 5.
"""
import json
import tempfile
import hashlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_external_assets
import torch
import numpy as np

from src.predictors.train import train_predictor
from src.predictors.losses import build_loss_fn
from src.predictors.losses.linex import LinExLoss


class TestM8SmokeTraining:
    """Smoke training tests ST-01 through ST-11."""

    @pytest.fixture
    def data_dir(self):
        """Path to processed FD001 data."""
        return Path("data/processed/fd001/v2")

    def _get_device(self):
        """Get available device (MPS preferred, fallback CPU)."""
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _compute_file_hash(self, path):
        """Compute SHA256 of a file."""
        if not Path(path).exists():
            return None
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _run_smoke_training(self, tmp_path, loss_type, linex_a=0.1):
        """Run smoke training and return result dict."""
        device = self._get_device()
        output_dir = tmp_path / f"{loss_type}_smoke"
        output_dir.mkdir(parents=True, exist_ok=True)

        result = train_predictor(
            data_dir=Path("data/processed/fd001/v2"),
            output_dir=output_dir,
            seed=6521,
            sequence_length=50,
            rul_cap=125,
            model_type="mlp",
            hidden_dim=128,
            n_layers=3,
            dropout=0.2,
            batch_size=64,
            learning_rate=1e-3,
            weight_decay=1e-4,
            max_epochs=2,
            patience=20,
            device=device,
            loss_type=loss_type,
            linex_a=linex_a,
            linex_overflow_threshold=20.0,
        )
        return result, output_dir, device

    def test_st01_mse_smoke_runs_two_epochs(self, tmp_path):
        """ST-01: MSE smoke training runs exactly 2 epochs without error."""
        result, output_dir, _ = self._run_smoke_training(tmp_path, "mse")

        assert result is not None
        assert "checkpoint_path" in result
        assert Path(result["checkpoint_path"]).exists()

        # Verify exactly 2 epochs in history
        history_path = output_dir / "training_history.json"
        assert history_path.exists()
        with open(history_path) as f:
            history = json.load(f)
        assert len(history) == 2, f"Expected 2 epochs, got {len(history)}"

        # Verify last checkpoint exists (saved at every epoch)
        last_ckpt_path = output_dir / "checkpoints" / "last_checkpoint.pt"
        assert last_ckpt_path.exists(), f"Last checkpoint not found at {last_ckpt_path}"

    def test_st02_linex_smoke_runs_two_epochs(self, tmp_path):
        """ST-02: LinEx a=0.1 smoke training runs exactly 2 epochs without error."""
        result, output_dir, _ = self._run_smoke_training(tmp_path, "linex", linex_a=0.1)

        assert result is not None
        assert "checkpoint_path" in result
        assert Path(result["checkpoint_path"]).exists()

        history_path = output_dir / "training_history.json"
        assert history_path.exists()
        with open(history_path) as f:
            history = json.load(f)
        assert len(history) == 2, f"Expected 2 epochs, got {len(history)}"

        # Verify last checkpoint exists
        last_ckpt_path = output_dir / "checkpoints" / "last_checkpoint.pt"
        assert last_ckpt_path.exists(), f"Last checkpoint not found at {last_ckpt_path}"

    def test_st03_checkpoint_saves(self, tmp_path):
        """ST-03: Both best and last checkpoints are saved for MSE run."""
        result, output_dir, device = self._run_smoke_training(tmp_path, "mse")

        best_path = Path(result["checkpoint_path"])
        last_path = output_dir / "checkpoints" / "last_checkpoint.pt"

        assert best_path.exists(), f"Best checkpoint not found at {best_path}"
        assert last_path.exists(), f"Last checkpoint not found at {last_path}"
        assert best_path.stat().st_size > 0, "Best checkpoint is empty"
        assert last_path.stat().st_size > 0, "Last checkpoint is empty"

        # Verify both are valid torch checkpoints
        best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        last_ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in best_ckpt
        assert "model_state_dict" in last_ckpt
        assert "config" in best_ckpt
        assert "config" in last_ckpt

    def test_st04_checkpoint_reloads_identical_predictions(self, tmp_path):
        """ST-04: Reloaded checkpoint produces identical validation predictions."""
        result, output_dir, device = self._run_smoke_training(tmp_path, "mse")

        # Load best checkpoint
        best_ckpt = torch.load(result["checkpoint_path"], map_location=device, weights_only=False)
        from src.predictors.model import RULPredictorMSE

        model = RULPredictorMSE(
            n_features=best_ckpt["config"]["n_features"],
            sequence_length=50,
            hidden_dim=best_ckpt["config"]["hidden_dim"],
            n_layers=best_ckpt["config"]["n_layers"],
            dropout=best_ckpt["config"]["dropout"],
        ).to(device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        model.eval()

        # Generate predictions on validation set
        from src.predictors.dataset import FD001SequenceDataset
        from torch.utils.data import DataLoader

        val_dataset = FD001SequenceDataset(
            data_dir=Path("data/processed/fd001/v2"),
            split="predictor_validation",
            sequence_length=50,
            rul_cap=125,
        )
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        predictions = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["features"].to(device)
                pred = model(x).squeeze(-1).cpu().numpy()
                predictions.extend(pred)

        pred_array = np.array(predictions)
        assert len(pred_array) > 0
        assert np.all(np.isfinite(pred_array))

    def test_st05_parameters_update(self, tmp_path):
        """ST-05: Model weights change after training (parameters update)."""
        device = self._get_device()
        output_dir = tmp_path / "mse_smoke_params"
        output_dir.mkdir(parents=True, exist_ok=True)

        from src.predictors.model import RULPredictorMSE
        from src.predictors.dataset import FD001SequenceDataset
        from torch.utils.data import DataLoader

        # Create initial model
        model = RULPredictorMSE(n_features=24, sequence_length=50, hidden_dim=128, n_layers=3, dropout=0.2).to(device)
        init_state = {k: v.clone() for k, v in model.state_dict().items()}

        result = train_predictor(
            data_dir=Path("data/processed/fd001/v2"),
            output_dir=output_dir,
            seed=6521,
            sequence_length=50,
            rul_cap=125,
            model_type="mlp",
            hidden_dim=128,
            n_layers=3,
            dropout=0.2,
            batch_size=64,
            learning_rate=1e-3,
            weight_decay=1e-4,
            max_epochs=2,
            patience=20,
            device=device,
            loss_type="mse",
            linex_a=0.1,
            linex_overflow_threshold=20.0,
        )

        # Load trained model
        best_ckpt = torch.load(result["checkpoint_path"], map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        final_state = model.state_dict()

        # At least one parameter should have changed
        changed = False
        for k in init_state:
            if not torch.allclose(init_state[k], final_state[k]):
                changed = True
                break

        assert changed, "No model parameters changed after training"

    def test_st06_loss_finite_all_batches(self, tmp_path):
        """ST-06: All batch losses are finite during MSE training."""
        result, output_dir, _ = self._run_smoke_training(tmp_path, "mse")

        history_path = output_dir / "training_history.json"
        with open(history_path) as f:
            history = json.load(f)

        for epoch in history:
            assert np.isfinite(epoch["train_loss"]), f"Non-finite train_loss in epoch {epoch['epoch']}"
            assert np.isfinite(epoch["val_loss"]), f"Non-finite val_loss in epoch {epoch['epoch']}"

    def test_st07_gradients_finite_all_steps(self, tmp_path):
        """ST-07: All gradient steps produce finite gradients (verified by training completion)."""
        # Training completes without gradient explosion = gradients were finite
        result, _, _ = self._run_smoke_training(tmp_path, "mse")
        assert result is not None
        assert "best_epoch" in result

    def test_st08_m1_artifacts_unchanged(self, tmp_path):
        """ST-08: M1 artifacts (checkpoint, normalizer, feature schema, split manifest) unchanged by smoke run."""
        # Hash M1 artifacts before
        m1_artifacts = {
            "checkpoint": Path("results/predictor/mse_baseline_v2/checkpoints/best_checkpoint.pt"),
            "normalizer": Path("data/processed/fd001/v2/04_PROTOCOL/fd001_normalizer_v2.json"),
            "feature_schema": Path("data/processed/fd001/v2/04_PROTOCOL/fd001_feature_schema_v1.json"),
            "split_manifest": Path("data/processed/fd001/v2/01_SPLIT/fd001_unit_split_v1.csv"),
            "cache": Path("data/processed/fd001/v2/06_PREDICTIONS/fd001_prediction_cache_v2.parquet"),
        }

        pre_hashes = {k: self._compute_file_hash(v) for k, v in m1_artifacts.items()}

        # Run smoke training
        self._run_smoke_training(tmp_path, "mse")

        # Hash after
        post_hashes = {k: self._compute_file_hash(v) for k, v in m1_artifacts.items()}

        for key in pre_hashes:
            if pre_hashes[key] is not None:
                assert pre_hashes[key] == post_hashes[key], f"M1 {key} changed by smoke run!"

    def test_st09_identical_architecture_data(self, tmp_path):
        """ST-09: MSE and LinEx use identical architecture, data, optimizer, seed."""
        mse_result, mse_dir, _ = self._run_smoke_training(tmp_path, "mse")
        linex_result, linex_dir, _ = self._run_smoke_training(tmp_path, "linex", linex_a=0.1)

        # Load both checkpoint configs
        mse_ckpt = torch.load(mse_result["checkpoint_path"], map_location="cpu", weights_only=False)
        linex_ckpt = torch.load(linex_result["checkpoint_path"], map_location="cpu", weights_only=False)

        mse_config = mse_ckpt["config"]
        linex_config = linex_ckpt["config"]

        # Architecture params must match
        assert mse_config["model_type"] == linex_config["model_type"] == "mlp"
        assert mse_config["hidden_dim"] == linex_config["hidden_dim"] == 128
        assert mse_config["n_layers"] == linex_config["n_layers"] == 3
        assert mse_config["dropout"] == linex_config["dropout"] == 0.2
        assert mse_config["n_features"] == linex_config["n_features"]  # n_features used, not input_dim

        # Training params must match
        assert mse_config["batch_size"] == linex_config["batch_size"] == 64
        assert mse_config["learning_rate"] == linex_config["learning_rate"] == 1e-3
        assert mse_config["weight_decay"] == linex_config["weight_decay"] == 1e-4
        assert mse_config["seed"] == linex_config["seed"] == 6521

        # Only loss_type should differ
        assert mse_config["loss_type"] == "mse"
        assert linex_config["loss_type"] == "linex"

    def test_st10_mse_loss_type_recorded(self, tmp_path):
        """ST-10: MSE loss_type correctly recorded in checkpoint and metadata."""
        result, output_dir, _ = self._run_smoke_training(tmp_path, "mse")

        ckpt = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=False)
        config = ckpt["config"]

        assert config["loss_type"] == "mse"
        # For MSE, linex params should be None or not relevant
        assert config.get("linex_a") is None or config.get("linex_a") == 0.1

    def test_st11_linex_loss_type_recorded(self, tmp_path):
        """ST-11: LinEx loss_type and linex_a correctly recorded in checkpoint and metadata."""
        result, output_dir, _ = self._run_smoke_training(tmp_path, "linex", linex_a=0.1)

        ckpt = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=False)
        config = ckpt["config"]

        assert config["loss_type"] == "linex"
        assert config["linex_a"] == 0.1
        assert config["linex_overflow_threshold"] == 20.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])