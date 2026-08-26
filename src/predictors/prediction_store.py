"""Prediction Store Interface

Read-only interface for retrieving cached RUL predictions by (split, unit_id, cycle).
Supports v2 cache as default, with unsafe/audit-only option for v1.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Literal

import numpy as np
import pandas as pd


@dataclass
class PredictionResult:
    """Result of a prediction lookup.

    Attributes:
        found: Whether the prediction was found
        split: Split name (if found)
        unit_id: Unit/engine ID (if found)
        cycle: Cycle (if found)
        predicted_rul: Predicted RUL (agent-visible)
        predicted_rul_normalized: Normalized predicted RUL
        true_rul: True RUL (hidden, for simulator accounting only)
        true_rul_capped: True RUL capped (hidden)
        trajectory_length: Full trajectory length
        valid_window: Whether this is a valid prediction window
        metadata: Additional metadata
        cache_version: Version of cache (v1 or v2)
    """

    found: bool
    split: Optional[str] = None
    unit_id: Optional[int] = None
    cycle: Optional[int] = None
    predicted_rul: Optional[float] = None
    predicted_rul_normalized: Optional[float] = None
    true_rul: Optional[float] = None
    true_rul_capped: Optional[float] = None
    trajectory_length: Optional[int] = None
    valid_window: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    cache_version: Literal["v1", "v2"] = "v2"


class PredictionStore:
    """Read-only prediction store.

    Loads cached predictions and provides lookup by (split, unit_id, cycle).

    V2 cache is the default. V1 cache is rejected by default unless
    allow_invalidated=True is explicitly set (audit-only mode).

    Args:
        cache_path: Path to prediction cache parquet file
        manifest_path: Path to manifest JSON file
        allow_invalidated: If True, allow loading v1 (invalidated) cache
    """

    def __init__(
        self,
        cache_path: Path,
        manifest_path: Optional[Path] = None,
        allow_invalidated: bool = False,
    ):
        self.cache_path = Path(cache_path)
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.allow_invalidated = allow_invalidated

        # Step 1: file exists (clear project-level error before any other work).
        if not self.cache_path.exists():
            raise FileNotFoundError(
                f"V2 prediction cache file not found: {self.cache_path}"
            )

        # Step 2: cache version accepted (refuse V1 by default).
        cache_version = self._detect_cache_version()
        if cache_version != "v2" and not allow_invalidated:
            raise ValueError(
                f"Attempted to load non-v2 cache: {cache_path}. "
                "Only V2 caches are accepted by PredictionStore. "
                "V1 was invalidated after the normalization-fix incident. "
                "Set allow_invalidated=True only for audit reads."
            )

        # Load cache rows once. Subsequent validation steps reuse this frame.
        self.cache_df = pd.read_parquet(self.cache_path)

        # Step 3: required columns present (must raise a clear project error,
        # not a pandas KeyError from later lookups).
        required_columns = [
            "split", "unit_id", "cycle", "trajectory_length",
            "true_rul", "predicted_rul", "predicted_rul_normalized",
            "valid_window", "predictor_id", "checkpoint_id",
            "normalizer_id", "feature_schema_id", "split_manifest_id",
            "sequence_length", "rul_cap", "cache_version",
        ]
        missing = set(required_columns) - set(self.cache_df.columns)
        if missing:
            raise ValueError(
                f"V2 prediction cache {self.cache_path} is missing required "
                f"columns: {sorted(missing)}"
            )

        # Step 4: numeric agent-visible and hidden fields are finite.
        for col in (
            "true_rul", "true_rul_capped", "predicted_rul",
            "predicted_rul_normalized",
        ):
            if col not in self.cache_df.columns:
                continue
            series = self.cache_df[col]
            if series.isna().any():
                raise ValueError(
                    f"V2 prediction cache column {col!r} contains NaN values "
                    f"in {self.cache_path}."
                )
            if not pd.api.types.is_numeric_dtype(series):
                raise ValueError(
                    f"V2 prediction cache column {col!r} must be numeric in "
                    f"{self.cache_path}."
                )
            if (series == float("inf")).any() or (series == float("-inf")).any():
                raise ValueError(
                    f"V2 prediction cache column {col!r} contains Inf values "
                    f"in {self.cache_path}."
                )

        # Step 5: duplicate (split, unit_id, cycle) keys rejected.
        if self.cache_df.duplicated(subset=["split", "unit_id", "cycle"]).any():
            raise ValueError(
                f"V2 prediction cache contains duplicate (split, unit_id, cycle) "
                f"keys in {self.cache_path}; uniqueness is required."
            )

        # Step 6: index constructed (last so we never build a non-unique index).
        self.cache_df = self.cache_df.set_index(["split", "unit_id", "cycle"])

        # Optional manifest loading — never blocks index construction.
        self.manifest = None
        if self.manifest_path and self.manifest_path.exists():
            with open(self.manifest_path, "r") as f:
                self.manifest = json.load(f)

        self._cache_version = cache_version

    def _detect_cache_version(self) -> Literal["v1", "v2"]:
        """Detect cache version from file name or manifest."""
        cache_name = self.cache_path.name
        if "v2" in cache_name:
            return "v2"
        elif "v1" in cache_name:
            return "v1"
        else:
            # Try to infer from manifest
            if self.manifest_path and self.manifest_path.exists():
                with open(self.manifest_path, "r") as f:
                    manifest = json.load(f)
                return manifest.get("cache_version", "v1")
            return "v1"  # Default to v1 for backward compatibility

    def _validate_v2_schema(self):
        """Validate v2 cache schema has required columns."""
        required_columns = [
            "split", "unit_id", "cycle", "trajectory_length",
            "true_rul", "predicted_rul", "predicted_rul_normalized",
            "valid_window", "predictor_id", "checkpoint_id",
            "normalizer_id", "feature_schema_id", "split_manifest_id",
            "sequence_length", "rul_cap", "cache_version"
        ]
        missing = set(required_columns) - set(self.cache_df.columns)
        if missing:
            raise ValueError(f"V2 cache missing required columns: {missing}")

    def get(
        self,
        split: str,
        unit_id: int,
        cycle: int,
    ) -> PredictionResult:
        """Get prediction by (split, unit_id, cycle).

        Args:
            split: Split name
            unit_id: Unit/engine ID
            cycle: Cycle number

        Returns:
            PredictionResult with found=True if found, found=False otherwise
        """
        key = (split, unit_id, cycle)

        if key not in self.cache_df.index:
            return PredictionResult(found=False, cache_version=self._cache_version)

        row = self.cache_df.loc[key]

        return PredictionResult(
            found=True,
            split=split,
            unit_id=int(unit_id),
            cycle=int(cycle),
            predicted_rul=float(row["predicted_rul"]),
            predicted_rul_normalized=float(row["predicted_rul_normalized"]),
            true_rul=float(row["true_rul"]),
            true_rul_capped=float(row.get("true_rul_capped", row["true_rul"])),
            trajectory_length=int(row["trajectory_length"]),
            valid_window=int(row.get("valid_window", 1)),
            metadata={
                "predictor_id": row.get("predictor_id", "unknown"),
                "checkpoint_id": row.get("checkpoint_id", "unknown"),
                "normalizer_id": row.get("normalizer_id", "unknown"),
                "feature_schema_id": row.get("feature_schema_id", "unknown"),
                "left_pad_count": int(row["left_pad_count"]) if "left_pad_count" in row else None,
            },
            cache_version=self._cache_version,
        )

    def get_agent_visible(
        self,
        split: str,
        unit_id: int,
        cycle: int,
    ) -> Optional[Dict[str, float]]:
        """Get only agent-visible fields.

        This method returns only the fields that should be visible
        to the RL agent (no true RUL).

        Args:
            split: Split name
            unit_id: Unit/engine ID
            cycle: Cycle number

        Returns:
            Dict with agent-visible fields, or None if not found
        """
        result = self.get(split, unit_id, cycle)

        if not result.found:
            return None

        return {
            "predicted_rul": result.predicted_rul,
            "predicted_rul_normalized": result.predicted_rul_normalized,
            "valid_window": result.valid_window,
        }

    def get_hidden_state(
        self,
        split: str,
        unit_id: int,
        cycle: int,
    ) -> Optional[Dict[str, float]]:
        """Get hidden simulator state (true RUL).

        This method returns only the hidden fields that should NOT
        be visible to the RL agent.

        Args:
            split: Split name
            unit_id: Unit/engine ID
            cycle: Cycle number

        Returns:
            Dict with hidden state fields, or None if not found
        """
        result = self.get(split, unit_id, cycle)

        if not result.found:
            return None

        return {
            "true_rul": result.true_rul,
            "true_rul_capped": result.true_rul_capped,
            "trajectory_length": result.trajectory_length,
        }

    def get_all_for_unit(
        self,
        split: str,
        unit_id: int,
    ) -> pd.DataFrame:
        """Get all predictions for a unit in a split.

        Args:
            split: Split name
            unit_id: Unit/engine ID

        Returns:
            DataFrame with all predictions for the unit
        """
        if (split, unit_id) not in zip(self.cache_df.index.get_level_values(0), self.cache_df.index.get_level_values(1)):
            return pd.DataFrame()

        return self.cache_df.xs((split, unit_id), level=["split", "unit_id"]).reset_index()

    def get_splits(self) -> List[str]:
        """Get list of splits in the cache."""
        return self.cache_df.index.get_level_values("split").unique().tolist()

    def get_units(self, split: str) -> List[int]:
        """Get list of unit IDs in a split."""
        return self.cache_df.xs(split, level="split").index.get_level_values(
            "unit_id"
        ).unique().tolist()

    def get_cycles(self, split: str, unit_id: int) -> List[int]:
        """Get list of cycles for a unit in a split."""
        return self.cache_df.xs(
            (split, unit_id), level=["split", "unit_id"]
        ).index.get_level_values("cycle").tolist()

    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """Get cache metadata from manifest."""
        return self.manifest

    def get_cache_version(self) -> Literal["v1", "v2"]:
        """Get cache version."""
        return self._cache_version

    def __len__(self) -> int:
        """Get total number of cached predictions."""
        return len(self.cache_df)

    def __repr__(self) -> str:
        return (
            f"PredictionStore({len(self)} predictions, "
            f"version={self._cache_version}, "
            f"splits={self.get_splits()})"
        )


def load_prediction_store(
    cache_dir: Path,
    version: Literal["v1", "v2"] = "v2",
    allow_invalidated: bool = False,
) -> PredictionStore:
    """Load prediction store from directory.

    Args:
        cache_dir: Directory containing prediction cache and manifest
        version: Cache version to load ("v1" or "v2")
        allow_invalidated: If True, allow loading v1 (invalidated) cache

    Returns:
        PredictionStore instance
    """
    cache_dir = Path(cache_dir)

    if version == "v2":
        cache_path = cache_dir / "fd001_prediction_cache_v2.parquet"
        manifest_path = cache_dir / "prediction_cache_manifest_v2.json"
    else:
        cache_path = cache_dir / "fd001_prediction_cache_v1.parquet"
        manifest_path = cache_dir / "prediction_cache_manifest_v1.json"

    # Fall back to versioned manifest if primary not found
    if not manifest_path.exists():
        manifest_path = cache_dir / f"prediction_cache_manifest_{version}.json"

    return PredictionStore(
        cache_path=cache_path,
        manifest_path=manifest_path,
        allow_invalidated=allow_invalidated,
    )


def load_default_prediction_store(
    cache_dir: Path,
) -> PredictionStore:
    """Load default prediction store (v2).

    Args:
        cache_dir: Directory containing prediction cache and manifest

    Returns:
        PredictionStore instance with v2 cache
    """
    return load_prediction_store(cache_dir, version="v2", allow_invalidated=False)


# Convenience function for quick lookup
def lookup_prediction(
    split: str,
    unit_id: int,
    cycle: int,
    cache_dir: Path,
    version: Literal["v1", "v2"] = "v2",
) -> Optional[Dict[str, float]]:
    """Quick lookup of agent-visible prediction.

    Args:
        split: Split name
        unit_id: Unit/engine ID
        cycle: Cycle number
        cache_dir: Directory containing prediction cache
        version: Cache version to use

    Returns:
        Dict with agent-visible fields, or None if not found
    """
    store = load_prediction_store(cache_dir, version=version)
    return store.get_agent_visible(split, unit_id, cycle)