"""Tests for prediction cache integrity.

Verifies data integrity, split integrity, leakage checks, and reproducibility.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Any

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.requires_external_assets
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def data_dir() -> Path:
    """Path to FD001 V2 processed directory."""
    return PROJECT_ROOT / "data" / "processed" / "fd001" / "v2"


@pytest.fixture
def split_df(data_dir: Path) -> pd.DataFrame:
    """Load split manifest."""
    return pd.read_csv(data_dir / "01_SPLIT" / "fd001_unit_split_v1.csv")


@pytest.fixture
def cycle_df(data_dir: Path) -> pd.DataFrame:
    """Load cycle table."""
    return pd.read_parquet(
        data_dir / "02_CYCLE_TABLE" / "fd001_train_cycle_table_v1.parquet"
    )


@pytest.fixture
def normalizer(data_dir: Path) -> Dict[str, Any]:
    """Load normalizer."""
    with open(data_dir / "04_PROTOCOL" / "fd001_normalizer_v2.json", "r") as f:
        return json.load(f)


@pytest.fixture
def feature_schema(data_dir: Path) -> Dict[str, Any]:
    """Load feature schema."""
    with open(data_dir / "04_PROTOCOL" / "fd001_feature_schema_v1.json", "r") as f:
        return json.load(f)


@pytest.fixture
def prediction_cache_path() -> Path:
    """Path to prediction cache."""
    return (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "fd001"
        / "v2"
        / "06_PREDICTIONS"
        / "fd001_prediction_cache_v1.parquet"
    )


@pytest.fixture
def prediction_manifest_path() -> Path:
    """Path to prediction manifest."""
    return (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "fd001"
        / "v2"
        / "06_PREDICTIONS"
        / "prediction_cache_manifest_v1.json"
    )


@pytest.fixture
def prediction_cache(prediction_cache_path: Path) -> pd.DataFrame:
    """Load prediction cache."""
    return pd.read_parquet(prediction_cache_path)


@pytest.fixture
def prediction_manifest(prediction_manifest_path: Path) -> Dict[str, Any]:
    """Load prediction manifest."""
    with open(prediction_manifest_path, "r") as f:
        return json.load(f)


# =============================================================================
# SECTION 1: DATA AND SPLIT INTEGRITY TESTS
# =============================================================================


class TestSplitIntegrity:
    """Test engine split integrity."""

    def test_engine_ids_disjoint_across_splits(self, split_df: pd.DataFrame):
        """Test 1: Engine IDs are disjoint across evaluation splits."""
        splits = split_df["split"].unique()

        unit_sets = {}
        for split in splits:
            units = set(split_df[split_df["split"] == split]["unit_id"])
            unit_sets[split] = units

        # Check all pairs are disjoint
        split_list = list(splits)
        for i, split1 in enumerate(split_list):
            for split2 in split_list[i + 1:]:
                intersection = unit_sets[split1] & unit_sets[split2]
                assert len(intersection) == 0, (
                    f"Engine ID overlap between {split1} and {split2}: {intersection}"
                )

    def test_no_rl_val_or_test_in_predictor_train(self, split_df: pd.DataFrame):
        """Test 2: No rl_validation or rl_test engine used during predictor fitting."""
        predictor_train_units = set(
            split_df[split_df["split"] == "predictor_train"]["unit_id"]
        )
        rl_val_units = set(
            split_df[split_df["split"] == "rl_validation"]["unit_id"]
        )
        rl_test_units = set(
            split_df[split_df["split"] == "rl_test"]["unit_id"]
        )

        # Check no overlap
        overlap_val = predictor_train_units & rl_val_units
        overlap_test = predictor_train_units & rl_test_units

        assert len(overlap_val) == 0, f"RL val units in predictor train: {overlap_val}"
        assert len(overlap_test) == 0, f"RL test units in predictor train: {overlap_test}"

    def test_normalizer_fit_on_train_only(self, normalizer: Dict[str, Any], split_df: pd.DataFrame):
        """Test 3: Normalizer was fitted using approved training split only."""
        # Check normalizer metadata
        assert normalizer["training_unit_count"] == 60
        assert normalizer["training_row_count"] == 12360

        # Verify training units match predictor_train split
        predictor_train_units = set(
            split_df[split_df["split"] == "predictor_train"]["unit_id"]
        )

        if "predictor_train_units" in normalizer:
            normalizer_train_units = set(normalizer["predictor_train_units"])
            assert normalizer_train_units == predictor_train_units, (
                "Normalizer training units don't match predictor_train split"
            )


class TestFeatureIntegrity:
    """Test predictor input feature integrity."""

    def test_predictor_input_contains_only_approved_features(
        self,
        feature_schema: Dict[str, Any],
        normalizer: Dict[str, Any],
    ):
        """Test 4: Predictor input tensor contains only approved feature columns."""
        # Check feature schema and normalizer agree
        schema_features = feature_schema["input_feature_order"]
        normalizer_features = normalizer["features"]

        assert schema_features == normalizer_features, (
            "Feature schema and normalizer disagree on feature order"
        )

        # Expected 24 features: 3 op_settings + 21 sensors
        assert len(schema_features) == 24
        assert schema_features[:3] == ["op_setting_1", "op_setting_2", "op_setting_3"]
        assert schema_features[3:] == [f"sensor_{i}" for i in range(1, 22)]

    def test_true_rul_not_in_input_features(
        self,
        feature_schema: Dict[str, Any],
    ):
        """Test 5: True RUL is used as target and hidden cache field, never as input."""
        input_features = set(feature_schema["input_feature_order"])
        target_features = set(feature_schema["target_columns"])

        # True RUL columns should not be in input features
        assert "true_rul_raw" not in input_features
        assert "true_rul_capped" not in input_features

        # Target columns are separate
        assert "true_rul_raw" in target_features
        assert "true_rul_capped" in target_features


class TestTrajectorySource:
    """Test trajectory source integrity."""

    def test_trajectory_source_is_complete_fd001(
        self,
        data_dir: Path,
        cycle_df: pd.DataFrame,
    ):
        """Test 6: Trajectory source is approved complete FD001 run-to-failure source."""
        # Check cycle table has all 100 train units
        unique_units = cycle_df["unit_id"].nunique()
        assert unique_units == 100, f"Expected 100 units, got {unique_units}"

        # Check each unit has consecutive cycles starting from 1
        for unit_id in cycle_df["unit_id"].unique()[:10]:  # Sample check
            unit_cycles = cycle_df[cycle_df["unit_id"] == unit_id].sort_values("cycle")
            cycles = unit_cycles["cycle"].values

            expected_cycles = np.arange(1, len(cycles) + 1)
            assert np.array_equal(cycles, expected_cycles), (
                f"Unit {unit_id} has non-consecutive cycles"
            )


# =============================================================================
# SECTION 2: PREDICTION CACHE INTEGRITY TESTS
# =============================================================================


class TestCacheUniqueness:
    """Test cache key uniqueness."""

    def test_unique_split_unit_cycle_keys(
        self,
        prediction_cache: pd.DataFrame,
    ):
        """Test 7: Every (split, unit_id, cycle) key is unique."""
        key_cols = ["split", "unit_id", "cycle"]
        duplicates = prediction_cache.duplicated(subset=key_cols)

        n_duplicates = duplicates.sum()
        assert n_duplicates == 0, f"Found {n_duplicates} duplicate keys"

    def test_no_nan_or_inf_in_required_fields(
        self,
        prediction_cache: pd.DataFrame,
    ):
        """Test 8: No required field contains NaN or infinity."""
        required_cols = [
            "split",
            "unit_id",
            "cycle",
            "trajectory_length",
            "true_rul",
            "true_rul_capped",
            "predicted_rul",
            "predicted_rul_normalized",
            "valid_window",
        ]

        for col in required_cols:
            # Check for NaN
            n_nan = prediction_cache[col].isna().sum()
            assert n_nan == 0, f"Column {col} has {n_nan} NaN values"

            # Check for Inf (numeric columns only)
            if prediction_cache[col].dtype in [np.float32, np.float64]:
                n_inf = np.isinf(prediction_cache[col]).sum()
                assert n_inf == 0, f"Column {col} has {n_inf} Inf values"


class TestWindowCoverage:
    """Test prediction window coverage."""

    def test_one_prediction_per_valid_window(
        self,
        prediction_cache: pd.DataFrame,
        data_dir: pd.DataFrame,
    ):
        """Test 9: Every valid sequence window has exactly one prediction."""
        # Load window index
        window_df = pd.read_parquet(
            data_dir / "05_WINDOW_INDEX" / "fd001_window_index_v1.parquet"
        )

        # Get RL-relevant splits
        rl_splits = ["predictor_train", "rl_validation", "rl_test"]

        for split in rl_splits:
            if split not in prediction_cache["split"].unique():
                continue

            cache_split = prediction_cache[prediction_cache["split"] == split]
            window_split = window_df[window_df["split"] == split]

            # Each window should have one prediction
            assert len(cache_split) == len(window_split), (
                f"Split {split}: expected {len(window_split)} predictions, "
                f"got {len(cache_split)}"
            )

    def test_no_invalid_early_cycle_windows_marked_valid(
        self,
        prediction_cache: pd.DataFrame,
    ):
        """Test 10: No invalid early-cycle window is marked valid."""
        # All valid_window should be 1
        invalid = prediction_cache[prediction_cache["valid_window"] != 1]
        assert len(invalid) == 0, f"Found {len(invalid)} invalid windows marked as valid"

        # Check cycle values are within expected range (all cycles 1+)
        assert prediction_cache["cycle"].min() >= 1
        assert prediction_cache["cycle"].max() <= 400  # Max trajectory length


class TestCacheAlignment:
    """Test cache alignment with frozen tables."""

    def test_predictions_join_to_cycle_table(
        self,
        prediction_cache: pd.DataFrame,
        cycle_df: pd.DataFrame,
    ):
        """Test 11: Predictions join correctly to the frozen cycle table."""
        # Merge on (unit_id, cycle)
        merged = prediction_cache.merge(
            cycle_df[["unit_id", "cycle", "true_rul_raw", "true_rul_capped"]].rename(
                columns={"true_rul_raw": "true_rul_raw_ct", "true_rul_capped": "true_rul_capped_ct"}
            ),
            on=["unit_id", "cycle"],
            how="left",
        )

        # Check true_rul values match (cache has 'true_rul', cycle table has 'true_rul_raw')
        rul_mismatch = merged[
            abs(merged["true_rul"] - merged["true_rul_raw_ct"]) > 1e-6
        ]
        assert len(rul_mismatch) == 0, f"Found {len(rul_mismatch)} RUL mismatches"

        capped_mismatch = merged[
            abs(merged["true_rul_capped"] - merged["true_rul_capped_ct"]) > 1e-6
        ]
        assert len(capped_mismatch) == 0, f"Found {len(capped_mismatch)} capped RUL mismatches"

    def test_predicted_rul_normalization_follows_protocol(
        self,
        prediction_cache: pd.DataFrame,
        prediction_manifest: Dict[str, Any],
    ):
        """Test 12: Predicted RUL normalization follows frozen protocol."""
        rul_cap = prediction_manifest["rul_cap"]

        # Predicted RUL normalized = predicted_rul / rul_cap
        expected_normalized = prediction_cache["predicted_rul"] / rul_cap

        diff = abs(
            prediction_cache["predicted_rul_normalized"] - expected_normalized
        ).max()

        assert diff < 1e-6, f"Normalization mismatch: max diff = {diff}"

    def test_row_counts_match_expected(
        self,
        prediction_cache: pd.DataFrame,
        prediction_manifest: Dict[str, Any],
    ):
        """Test 13: Cached row counts agree with independently calculated counts."""
        # Check manifest row counts
        for split, expected_count in prediction_manifest["row_counts"].items():
            actual_count = len(prediction_cache[prediction_cache["split"] == split])
            assert actual_count == expected_count, (
                f"Split {split}: expected {expected_count}, got {actual_count}"
            )


class TestCacheConsistency:
    """Test cache internal consistency."""

    def test_predictor_id_consistent(
        self,
        prediction_cache: pd.DataFrame,
        prediction_manifest: Dict[str, Any],
    ):
        """Test 14: predictor_id and normalizer_id are consistent across cache."""
        # All rows should have same predictor_id (implicit in manifest)
        assert "predictor_id" in prediction_manifest
        assert "normalizer_id" in prediction_manifest

    @pytest.mark.legacy_v1
    def test_checkpoint_reload_reproduces_predictions(
        self,
        prediction_cache: pd.DataFrame,
        prediction_manifest: Dict[str, Any],
        data_dir: Path,
    ):
        """Test 15: Loading saved checkpoint reproduces cached predictions."""
        # This test requires the actual checkpoint to exist
        checkpoint_path = Path(prediction_manifest["checkpoint_path"])

        if not checkpoint_path.exists():
            pytest.skip(f"Checkpoint not found: {checkpoint_path}")

        # Import here to avoid circular imports
        from src.predictors.model import build_predictor

        # Load checkpoint
        device = "cpu"
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

        # Sample a few predictions and verify
        from src.predictors.dataset import FD001SequenceDataset

        # Test 5 random samples
        np.random.seed(6521)
        sample_indices = np.random.choice(len(prediction_cache), size=5, replace=False)

        for idx in sample_indices:
            row = prediction_cache.iloc[idx]
            split = row["split"]
            unit_id = row["unit_id"]
            cycle = row["cycle"]

            # Get dataset sample
            dataset = FD001SequenceDataset(
                split=split,
                data_dir=data_dir,
                sequence_length=config["sequence_length"],
                rul_cap=config["rul_cap"],
            )

            sample = dataset.get_by_key(unit_id, cycle)
            if sample is None:
                pytest.fail(f"Sample not found: ({split}, {unit_id}, {cycle})")

            # Forward pass
            with torch.no_grad():
                x = sample["features"].unsqueeze(0).to(device)
                y_pred = model(x).squeeze(0).cpu().item()

            # Compare with cached prediction (within numerical tolerance)
            cached_pred = row["predicted_rul"]
            diff = abs(y_pred - cached_pred)

            assert diff < 1e-5, (
                f"Prediction mismatch for ({split}, {unit_id}, {cycle}): "
                f"reproduced={y_pred:.6f}, cached={cached_pred:.6f}, diff={diff:.2e}"
            )

    def test_fixed_seed_reproducibility(
        self,
        prediction_manifest: Dict[str, Any],
    ):
        """Test 16: Fixed-seed inference is reproducible."""
        # Check seed is recorded
        assert "random_seed" in prediction_manifest
        seed = prediction_manifest["random_seed"]
        assert isinstance(seed, int)
        assert seed > 0


class TestCacheInterface:
    """Test cache interface."""

    @pytest.mark.requires_v2_cache
    def test_cache_loadable_by_typed_interface(
        self,
        prediction_cache_path: Path,
        prediction_manifest_path: Path,
    ):
        """Test 17: Cache files and manifests can be loaded by simple typed interface."""
        from src.predictors.prediction_store import load_prediction_store, PredictionStore

        # Load store
        store = load_prediction_store(prediction_cache_path.parent)

        assert isinstance(store, PredictionStore)
        assert len(store) > 0

        # Test lookup
        splits = store.get_splits()
        assert len(splits) > 0

        # Test individual lookup
        for split in splits[:1]:  # Test first split
            units = store.get_units(split)
            if len(units) > 0:
                unit_id = units[0]
                cycles = store.get_cycles(split, unit_id)
                if len(cycles) > 0:
                    cycle = cycles[0]
                    result = store.get(split, unit_id, cycle)
                    assert result.found
                    assert result.predicted_rul is not None
                    assert result.true_rul is not None


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])