"""
Environment configuration for Milestone 2 Selective Maintenance Environment.

Implements a typed, immutable configuration contract for the M2 environment.
All configuration values are loaded from JSON and validated against the contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .costs import CostRegime, get_cost_regime, validate_cost_regime


# Allowed splits for environment instances
ALLOWED_SPLITS = frozenset({"predictor_train", "rl_validation", "rl_test"})

# Valid info modes
INFO_MODES = frozenset({"normal", "diagnostic"})

# Frozen defaults for M2 v1
DEFAULT_ENVIRONMENT_VERSION = "m2_v1"
DEFAULT_FLEET_SIZE = 5
DEFAULT_MAINTENANCE_CAPACITY = 2
DEFAULT_DELTA_CYCLES = 5
DEFAULT_EPISODE_HORIZON = 100
DEFAULT_AGE_SCALE_CYCLES = 341
DEFAULT_RUL_SCALE = 125.0
DEFAULT_INFO_MODE = "normal"
DEFAULT_SEED = 6521

# Required V2 cache path identifier
V2_CACHE_PATH_IDENTIFIER = "v2/06_PREDICTIONS"
V1_CACHE_PATH_IDENTIFIER = "v1"


@dataclass(frozen=True)
class EnvironmentConfig:
    """
    Immutable environment configuration for Milestone 2.

    All fields are validated on construction and must satisfy the contract.
    """

    environment_version: str
    split: str
    fleet_size: int
    maintenance_capacity: int
    delta_cycles: int
    episode_horizon: int
    age_scale_cycles: int
    rul_scale: float
    cost_regime_id: str
    scenario_bank_path: str
    prediction_cache_path: str
    info_mode: str
    seed: int

    def __post_init__(self) -> None:
        """Validate configuration against the M2 contract."""
        errors: list[str] = []

        # Validate N > 0
        if self.fleet_size <= 0:
            errors.append(f"fleet_size must be positive, got {self.fleet_size}")

        # Validate 0 <= K <= N
        if self.maintenance_capacity < 0:
            errors.append(
                f"maintenance_capacity must be non-negative, got {self.maintenance_capacity}"
            )
        if self.maintenance_capacity > self.fleet_size:
            errors.append(
                f"maintenance_capacity ({self.maintenance_capacity}) cannot exceed "
                f"fleet_size ({self.fleet_size})"
            )

        # Validate delta_cycles > 0
        if self.delta_cycles <= 0:
            errors.append(f"delta_cycles must be positive, got {self.delta_cycles}")

        # Validate episode_horizon > 0
        if self.episode_horizon <= 0:
            errors.append(f"episode_horizon must be positive, got {self.episode_horizon}")

        # Validate age_scale_cycles == 341 for m2_v1
        if self.environment_version == "m2_v1" and self.age_scale_cycles != 341:
            errors.append(
                f"age_scale_cycles must be 341 for m2_v1, got {self.age_scale_cycles}"
            )

        # Validate rul_scale == 125 for m2_v1
        if self.environment_version == "m2_v1" and self.rul_scale != 125.0:
            errors.append(f"rul_scale must be 125.0 for m2_v1, got {self.rul_scale}")

        # Validate split
        if self.split not in ALLOWED_SPLITS:
            errors.append(
                f"split must be one of {ALLOWED_SPLITS}, got '{self.split}'"
            )

        # Validate info_mode
        if self.info_mode not in INFO_MODES:
            errors.append(
                f"info_mode must be one of {INFO_MODES}, got '{self.info_mode}'"
            )

        # Validate cost regime
        try:
            validate_cost_regime(self.cost_regime_id)
        except ValueError as e:
            errors.append(str(e))

        # Validate prediction cache path (must be V2, not V1)
        if V1_CACHE_PATH_IDENTIFIER in self.prediction_cache_path:
            errors.append(
                "V1 cache path is not accepted. Must use V2 cache: "
                f"{self.prediction_cache_path}"
            )
        if V2_CACHE_PATH_IDENTIFIER not in self.prediction_cache_path:
            errors.append(
                f"prediction_cache_path must contain '{V2_CACHE_PATH_IDENTIFIER}', "
                f"got: {self.prediction_cache_path}"
            )

        if errors:
            raise ValueError("Configuration validation failed:\n  - " + "\n  - ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "environment_version": self.environment_version,
            "split": self.split,
            "fleet_size": self.fleet_size,
            "maintenance_capacity": self.maintenance_capacity,
            "delta_cycles": self.delta_cycles,
            "episode_horizon": self.episode_horizon,
            "age_scale_cycles": self.age_scale_cycles,
            "rul_scale": self.rul_scale,
            "cost_regime_id": self.cost_regime_id,
            "scenario_bank_path": self.scenario_bank_path,
            "prediction_cache_path": self.prediction_cache_path,
            "info_mode": self.info_mode,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnvironmentConfig":
        """
        Create from dictionary.

        Raises:
            ValueError: If required keys are missing or validation fails.
        """
        required_keys = {
            "environment_version",
            "split",
            "fleet_size",
            "maintenance_capacity",
            "delta_cycles",
            "episode_horizon",
            "age_scale_cycles",
            "rul_scale",
            "cost_regime_id",
            "scenario_bank_path",
            "prediction_cache_path",
            "info_mode",
            "seed",
        }
        missing = required_keys - set(data.keys())
        if missing:
            raise ValueError(f"Missing required keys: {missing}")

        return cls(
            environment_version=str(data["environment_version"]),
            split=str(data["split"]),
            fleet_size=int(data["fleet_size"]),
            maintenance_capacity=int(data["maintenance_capacity"]),
            delta_cycles=int(data["delta_cycles"]),
            episode_horizon=int(data["episode_horizon"]),
            age_scale_cycles=int(data["age_scale_cycles"]),
            rul_scale=float(data["rul_scale"]),
            cost_regime_id=str(data["cost_regime_id"]),
            scenario_bank_path=str(data["scenario_bank_path"]),
            prediction_cache_path=str(data["prediction_cache_path"]),
            info_mode=str(data["info_mode"]),
            seed=int(data["seed"]),
        )

    @classmethod
    def from_json_file(cls, path: Path | str) -> "EnvironmentConfig":
        """
        Load configuration from a JSON file.

        Args:
            path: Path to the JSON configuration file.

        Returns:
            Validated EnvironmentConfig instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
            ValueError: If validation fails.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_cost_regime(self) -> CostRegime:
        """Get the cost regime instance for this configuration."""
        return get_cost_regime(self.cost_regime_id)

    def get_action_space_size(self) -> int:
        """
        Calculate the action space size for this configuration.

        Returns the number of feasible subsets: sum of C(N, k) for k in [0, K].
        """
        from math import comb

        return sum(comb(self.fleet_size, k) for k in range(self.maintenance_capacity + 1))


def get_default_config(
    split: str = "rl_validation",
    cost_regime_id: str = "failure-light-no-waste",
    maintenance_capacity: int = 2,
    scenario_bank_path: str = "configs/scenarios/m2_validation.json",
    prediction_cache_path: str = "data/processed/fd001/v2/06_PREDICTIONS/",
    seed: int = 6521,
    info_mode: str = "normal",
) -> EnvironmentConfig:
    """
    Get a default environment configuration with customizable overrides.

    Args:
        split: Environment split (default: rl_validation).
        cost_regime_id: Cost regime (default: failure-light-no-waste).
        maintenance_capacity: K value (default: 2 for main experiment).
        scenario_bank_path: Path to scenario bank JSON.
        prediction_cache_path: Path to prediction cache.
        seed: Environment seed.
        info_mode: Info mode (default: normal).

    Returns:
        Validated EnvironmentConfig with default values.
    """
    return EnvironmentConfig(
        environment_version=DEFAULT_ENVIRONMENT_VERSION,
        split=split,
        fleet_size=DEFAULT_FLEET_SIZE,
        maintenance_capacity=maintenance_capacity,
        delta_cycles=DEFAULT_DELTA_CYCLES,
        episode_horizon=DEFAULT_EPISODE_HORIZON,
        age_scale_cycles=DEFAULT_AGE_SCALE_CYCLES,
        rul_scale=DEFAULT_RUL_SCALE,
        cost_regime_id=cost_regime_id,
        scenario_bank_path=scenario_bank_path,
        prediction_cache_path=prediction_cache_path,
        info_mode=info_mode,
        seed=seed,
    )