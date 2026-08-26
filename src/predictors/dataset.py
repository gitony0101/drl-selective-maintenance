"""FD001 Sequence Dataset for RUL Prediction

Loads pre-computed feature windows from the frozen V2 data package.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class FD001SequenceDataset(Dataset):
    """PyTorch Dataset for FD001 RUL prediction.

    Reads normalized feature windows from the frozen V2 data package.
    Each sample is a (sequence_length, n_features) window ending at target_cycle.

    Args:
        split: One of 'predictor_train', 'predictor_validation', 'rl_validation', 'rl_test'
        data_dir: Path to processed FD001 V2 directory
        sequence_length: Sequence window length (default: 50)
        rul_cap: RUL cap value (default: 125)
        use_capped_rul: If True, use capped RUL as target; else use raw RUL
    """

    def __init__(
        self,
        split: str,
        data_dir: Union[str, Path],
        sequence_length: int = 50,
        rul_cap: int = 125,
        use_capped_rul: bool = True,
    ):
        super().__init__()

        self.split = split
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.rul_cap = rul_cap
        self.use_capped_rul = use_capped_rul

        # Validate split
        valid_splits = ['predictor_train', 'predictor_validation', 'rl_validation', 'rl_test']
        if split not in valid_splits:
            raise ValueError(f"Invalid split: {split}. Must be one of {valid_splits}")

        # Load frozen artifacts
        self._load_artifacts()

        # Build index for this split
        self._build_index()

    def _load_artifacts(self):
        """Load frozen V2 artifacts: cycle table, split manifest, normalizer, feature schema."""

        # Split manifest
        split_path = self.data_dir / "01_SPLIT" / "fd001_unit_split_v1.csv"
        self.split_df = pd.read_csv(split_path)

        # Get units for this split
        self.split_units = set(
            self.split_df[self.split_df["split"] == self.split]["unit_id"]
        )

        # Cycle table (contains true RUL labels)
        cycle_table_path = self.data_dir / "02_CYCLE_TABLE" / "fd001_train_cycle_table_v1.parquet"
        self.cycle_df = pd.read_parquet(cycle_table_path)

        # Filter to current split units only
        self.split_cycle_df = self.cycle_df[self.cycle_df["unit_id"].isin(self.split_units)].copy()

        # Window index
        window_index_path = self.data_dir / "05_WINDOW_INDEX" / "fd001_window_index_v1.parquet"
        self.window_df = pd.read_parquet(window_index_path)

        # Filter to current split
        self.split_window_df = self.window_df[self.window_df["unit_id"].isin(self.split_units)].copy()

        # Normalizer
        normalizer_path = self.data_dir / "04_PROTOCOL" / "fd001_normalizer_v2.json"
        with open(normalizer_path, "r") as f:
            self.normalizer = json.load(f)

        # Feature schema
        schema_path = self.data_dir / "04_PROTOCOL" / "fd001_feature_schema_v1.json"
        with open(schema_path, "r") as f:
            self.feature_schema = json.load(f)

        # Extract feature order and normalizer arrays
        self.feature_cols = self.feature_schema["input_feature_order"]  # 24 features
        self.n_features = len(self.feature_cols)

        # Validate normalizer metadata matches feature schema
        mean_dict = self.normalizer["mean"]
        std_dict = self.normalizer["std"]

        # Check that normalizer has all required features
        normalizer_features = set(mean_dict.keys())
        schema_features = set(self.feature_cols)

        if normalizer_features != schema_features:
            missing = schema_features - normalizer_features
            extra = normalizer_features - schema_features
            raise ValueError(
                f"Normalizer features ({len(normalizer_features)}) do not match "
                f"feature schema features ({len(schema_features)}). "
                f"Missing: {missing}, Extra: {extra}"
            )

        self.mean = np.array([mean_dict[f] for f in self.feature_cols], dtype=np.float32)
        self.std = np.array([std_dict[f] for f in self.feature_cols], dtype=np.float32)

        # Validate shapes
        assert self.mean.shape == (self.n_features,), \
            f"Expected mean shape ({self.n_features},), got {self.mean.shape}"
        assert self.std.shape == (self.n_features,), \
            f"Expected std shape ({self.n_features},), got {self.std.shape}"

        # Log zero-std replacement policy (for audit trail)
        self._zero_std_replacement_count = np.sum(np.isclose(self.std, 1.0, atol=1e-10) &
                                                   np.array([std_dict[f] == 1.0 for f in self.feature_cols]))

    def _build_index(self):
        """Build (unit_id, cycle) index for this split."""

        # Create index with all necessary fields
        self.index = self.split_window_df[
            ["unit_id", "target_cycle", "true_rul_raw", "true_rul_capped", "left_pad_count"]
        ].reset_index(drop=True)

        # Map (unit_id, cycle) to row index for fast lookup
        self._lookup = {}
        for idx in range(len(self.index)):
            row = self.index.iloc[idx]
            key = (row["unit_id"], row["target_cycle"])
            self._lookup[key] = idx

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Returns:
            Dict with:
                - "features": (sequence_length, n_features) float32 tensor
                - "rul_raw": float32 scalar (uncapped RUL)
                - "rul_capped": float32 scalar (capped RUL)
                - "unit_id": int
                - "cycle": int
                - "left_pad_count": int
        """
        row = self.index.iloc[idx]
        unit_id = int(row["unit_id"])
        cycle = int(row["target_cycle"])
        left_pad_count = int(row["left_pad_count"])

        # Get feature window from cycle table
        # For left padding, we use the first observed row repeated
        window_start = max(1, cycle - self.sequence_length + 1)

        # Get all cycles for this unit in order
        unit_cycles = self.cycle_df[self.cycle_df["unit_id"] == unit_id].sort_values("cycle")

        # Build feature window
        features = []
        for c in range(window_start, cycle + 1):
            cycle_row = unit_cycles[unit_cycles["cycle"] == c]
            if len(cycle_row) == 1:
                feature_row = cycle_row[self.feature_cols].values[0]
            else:
                # Should not happen in valid data
                raise ValueError(f"Missing or duplicate cycle {c} for unit {unit_id}")
            features.append(feature_row)

        features = np.stack(features, axis=0).astype(np.float32)  # (window_len, n_features)

        # Apply left padding if needed
        if left_pad_count > 0:
            # Repeat first row for padding
            first_row = features[0:1].repeat(left_pad_count, axis=0)
            features = np.concatenate([first_row, features], axis=0)

        # Ensure exact sequence length
        assert features.shape[0] == self.sequence_length, \
            f"Expected {self.sequence_length} rows, got {features.shape[0]}"

        # CRITICAL FIX: Apply frozen normalizer before converting to tensor
        # Formula: x_normalized = (x_raw - mean) / std_safe
        # This was the root cause of Milestone 1 failure - normalizer was loaded
        # but never applied, causing predictor collapse to ~9.0993 constant predictions
        features = (features - self.mean) / self.std  # Broadcast: (50, 24) - (24,) -> (50, 24)

        # Validate output tensor shape
        assert features.shape == (self.sequence_length, self.n_features), \
            f"Expected shape ({self.sequence_length}, {self.n_features}), got {features.shape}"

        # Validate that every output value is finite
        if not np.all(np.isfinite(features)):
            raise ValueError(
                f"Non-finite values in normalized features for unit {unit_id}, cycle {cycle}. "
                f"nan_count={np.isnan(features).sum()}, inf_count={np.isinf(features).sum()}"
            )

        # Convert to tensor with explicit floating-point dtype
        features_tensor = torch.from_numpy(features).to(torch.float32)  # (sequence_length, n_features)

        return {
            "features": features_tensor,
            "rul_raw": torch.tensor(float(row["true_rul_raw"]), dtype=torch.float32),
            "rul_capped": torch.tensor(float(row["true_rul_capped"]), dtype=torch.float32),
            "unit_id": torch.tensor(unit_id, dtype=torch.int32),
            "cycle": torch.tensor(cycle, dtype=torch.int32),
            "left_pad_count": torch.tensor(left_pad_count, dtype=torch.int32),
        }

    def iterrows(self):
        """Iterate over index rows."""
        for i in range(len(self)):
            yield self.index.iloc[i]

    def get_by_key(self, unit_id: int, cycle: int) -> Optional[Dict[str, torch.Tensor]]:
        """Get sample by (unit_id, cycle) key."""
        key = (unit_id, cycle)
        if key not in self._lookup:
            return None
        idx = self._lookup[key]
        return self[idx]

    def get_feature_window(
        self,
        unit_id: int,
        cycle: int
    ) -> Optional[np.ndarray]:
        """Get normalized feature window as numpy array.

        Returns None if (unit_id, cycle) not in this split.
        """
        sample = self.get_by_key(unit_id, cycle)
        if sample is None:
            return None
        return sample["features"].numpy()

    def diagnostic_normalize(
        self,
        unit_id: int,
        cycle: int,
    ) -> Dict[str, Any]:
        """Diagnostic function for auditor verification of normalization.

        Accepts split (via dataset constructor), unit_id, and cycle.
        Returns comprehensive normalization diagnostics.

        Args:
            unit_id: Unit/engine ID
            cycle: Target cycle number

        Returns:
            Dict with:
                - raw_feature_window: (window_len, 24) before padding/normalization
                - padded_raw_window: (50, 24) after left padding, before normalization
                - manually_normalized_window: (50, 24) after manual (x-mean)/std
                - dataset_produced_tensor: (50, 24) tensor from __getitem__
                - max_absolute_difference: max|manual - dataset|
                - mean_absolute_difference: mean|manual - dataset|
                - feature_order: list of 24 feature names
                - normalizer_id: normalizer identifier
        """
        # Get the row
        key = (unit_id, cycle)
        if key not in self._lookup:
            raise KeyError(f"(unit_id={unit_id}, cycle={cycle}) not found in {self.split}")

        idx = self._lookup[key]
        row = self.index.iloc[idx]
        left_pad_count = int(row["left_pad_count"])
        target_cycle = int(row["target_cycle"])  # target_cycle is the column name in index

        # Build raw window (same as __getitem__)
        window_start = max(1, cycle - self.sequence_length + 1)
        unit_cycles = self.cycle_df[self.cycle_df["unit_id"] == unit_id].sort_values("cycle")

        raw_features = []
        for c in range(window_start, cycle + 1):
            cycle_row = unit_cycles[unit_cycles["cycle"] == c]
            if len(cycle_row) == 1:
                feature_row = cycle_row[self.feature_cols].values[0]
            else:
                raise ValueError(f"Missing or duplicate cycle {c} for unit {unit_id}")
            raw_features.append(feature_row)

        raw_features = np.stack(raw_features, axis=0).astype(np.float32)
        raw_feature_window = raw_features.copy()  # Before padding

        # Apply left padding
        if left_pad_count > 0:
            first_row = raw_features[0:1].repeat(left_pad_count, axis=0)
            raw_features = np.concatenate([first_row, raw_features], axis=0)

        padded_raw_window = raw_features.copy()  # Before normalization

        # Manual normalization
        manually_normalized = (padded_raw_window - self.mean) / self.std

        # Get dataset-produced tensor
        sample = self[idx]
        dataset_tensor = sample["features"].numpy()

        # Compute differences
        max_abs_diff = np.max(np.abs(manually_normalized - dataset_tensor))
        mean_abs_diff = np.mean(np.abs(manually_normalized - dataset_tensor))

        return {
            "raw_feature_window": raw_feature_window,
            "padded_raw_window": padded_raw_window,
            "manually_normalized_window": manually_normalized,
            "dataset_produced_tensor": dataset_tensor,
            "max_absolute_difference": float(max_abs_diff),
            "mean_absolute_difference": float(mean_abs_diff),
            "feature_order": self.feature_cols,
            "normalizer_id": "fd001_normalizer_v2",
            "unit_id": unit_id,
            "cycle": cycle,
            "left_pad_count": left_pad_count,
        }


def build_dataloaders(
    data_dir: Union[str, Path],
    sequence_length: int = 50,
    rul_cap: int = 125,
    batch_size: int = 64,
    train_batch_size: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 6521,
) -> Dict[str, DataLoader]:
    """Build DataLoaders for all splits.

    Args:
        data_dir: Path to FD001 V2 processed directory
        sequence_length: Sequence window length
        rul_cap: RUL cap value
        batch_size: Batch size for val/test loaders
        train_batch_size: Batch size for train loader (default: same as batch_size)
        num_workers: Number of DataLoader workers
        pin_memory: Pin memory for faster GPU transfer
        seed: Random seed for train loader shuffling

    Returns:
        Dict with 'train', 'validation', 'rl_validation', 'rl_test' DataLoaders
        (only splits with data are included)
    """
    data_dir = Path(data_dir)
    if train_batch_size is None:
        train_batch_size = batch_size

    # Load split manifest to know which splits exist
    split_path = data_dir / "01_SPLIT" / "fd001_unit_split_v1.csv"
    split_df = pd.read_csv(split_path)

    dataloaders = {}

    # Build dataset and loader for each split
    splits = ['predictor_train', 'predictor_validation', 'rl_validation', 'rl_test']

    for split in splits:
        units = split_df[split_df["split"] == split]["unit_id"]
        if len(units) == 0:
            continue

        dataset = FD001SequenceDataset(
            split=split,
            data_dir=data_dir,
            sequence_length=sequence_length,
            rul_cap=rul_cap,
        )

        if len(dataset) == 0:
            continue

        shuffle = (split == 'predictor_train')
        bs = train_batch_size if split == 'predictor_train' else batch_size

        loader = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=torch.Generator().manual_seed(seed) if shuffle else None,
        )

        dataloaders[split] = loader

    return dataloaders


class FD001BatchCollator:
    """Collator for batching FD001 samples.

    Pads sequences if needed (though all sequences should have same length).
    """

    def __init__(self, sequence_length: int = 50):
        self.sequence_length = sequence_length

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples.

        Returns:
            Dict with:
                - "features": (batch, sequence_length, n_features)
                - "rul_raw": (batch,)
                - "rul_capped": (batch,)
                - "unit_id": (batch,)
                - "cycle": (batch,)
                - "left_pad_count": (batch,)
        """
        features = torch.stack([item["features"] for item in batch])

        return {
            "features": features,
            "rul_raw": torch.stack([item["rul_raw"] for item in batch]),
            "rul_capped": torch.stack([item["rul_capped"] for item in batch]),
            "unit_id": torch.stack([item["unit_id"] for item in batch]),
            "cycle": torch.stack([item["cycle"] for item in batch]),
            "left_pad_count": torch.stack([item["left_pad_count"] for item in batch]),
        }