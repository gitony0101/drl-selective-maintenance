"""
Shared Configuration Parser for Milestone 5 DDQN.

This module provides the SINGLE authoritative configuration parser used by:
- train_ddqn.py (training CLI)
- evaluate_ddqn.py (evaluation CLI)
- validate_config.py (validation CLI)
- generate_m5_matrix.py (matrix generator)
- tests (for config validation)

All config parsing logic is centralized here to ensure consistency across
all entry points.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from src.training.ddqn_trainer import TrainerConfig


@dataclass(frozen=True)
class ParsedConfig:
    """Container for parsed configuration."""
    trainer_config: TrainerConfig
    raw_config: Dict[str, Any]


def parse_raw_config(raw_config: Dict[str, Any]) -> TrainerConfig:
    """
    Parse raw config JSON dict into production TrainerConfig.

    This is the SINGLE authoritative config parser used by ALL entry points.

    Handles nested config structure:
    - environment.* -> TrainerConfig fields
    - training.* -> TrainerConfig fields
    - agent.* -> TrainerConfig fields
    - output.* -> TrainerConfig fields
    - device.* -> TrainerConfig fields

    Args:
        raw_config: Raw configuration dictionary from JSON file

    Returns:
        Validated TrainerConfig instance

    Raises:
        ValueError: If configuration validation fails
    """
    env_cfg = raw_config.get("environment", {})
    agent_cfg = raw_config.get("agent", {})
    train_cfg = raw_config.get("training", {})
    output_cfg = raw_config.get("output", {})
    device_cfg = raw_config.get("device", {})

    # Support both old scenario_bank_path and new separate paths
    # For M5 formal training, both training and validation paths should be specified
    training_scenario_bank_path = env_cfg.get("training_scenario_bank_path")
    validation_scenario_bank_path = env_cfg.get("validation_scenario_bank_path")

    # Backward compatibility: if old field is present but new ones aren't, use old field
    if training_scenario_bank_path is None and validation_scenario_bank_path is None:
        old_path = env_cfg.get("scenario_bank_path")
        if old_path is not None:
            # Default both to the old path for backward compatibility
            # But this will fail M5 formal validation which requires distinct paths
            training_scenario_bank_path = old_path
            validation_scenario_bank_path = old_path

    merged = {
        "split": env_cfg.get("split", "predictor_train"),
        "validation_split": env_cfg.get("validation_split", "rl_validation"),
        "maintenance_capacity": env_cfg.get("maintenance_capacity", 2),
        "cost_regime_id": env_cfg.get("cost_regime_id", "failure-light-no-waste"),
        "episode_horizon": env_cfg.get("episode_horizon", 100),
        "training_scenario_bank_path": training_scenario_bank_path,
        "validation_scenario_bank_path": validation_scenario_bank_path,
        "prediction_cache_path": env_cfg.get(
            "prediction_cache_path", "data/processed/fd001/v2/06_PREDICTIONS/"
        ),
        "prediction_cache_manifest_path": env_cfg.get(
            "prediction_cache_manifest_path",
            "data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json",
        ),
        "max_steps": train_cfg.get("max_steps", 100_000),
        "batch_size": train_cfg.get("batch_size", 128),
        "warmup_transitions": train_cfg.get("warmup_transitions", 5_000),
        "update_frequency": train_cfg.get("update_frequency", 1),
        "validation_interval": train_cfg.get("validation_interval", 5_000),
        "checkpoint_interval": train_cfg.get("checkpoint_interval", 5_000),
        "replay_capacity": train_cfg.get("replay_capacity", 100_000),
        "training_seed": train_cfg.get("training_seed", 6521),
        "validation_seed": train_cfg.get("validation_seed", 6521),
        "hidden_dim": agent_cfg.get("hidden_dim", 128),
        "num_hidden_layers": agent_cfg.get("num_hidden_layers", 2),
        "learning_rate": agent_cfg.get("learning_rate", 1e-4),
        "gamma": agent_cfg.get("gamma", 0.95),
        "epsilon_start": agent_cfg.get("epsilon_start", 1.0),
        "epsilon_end": agent_cfg.get("epsilon_end", 0.05),
        "epsilon_decay_steps": agent_cfg.get("epsilon_decay_steps", 50_000),
        "gradient_clip": agent_cfg.get("gradient_clip", 10.0),
        "target_update_interval": agent_cfg.get("target_update_interval", 1_000),
        "output_dir": output_cfg.get("output_dir", "results/milestone5"),
        "run_id": output_cfg.get("run_id"),
        "device": device_cfg.get("explicit_device"),
    }

    return TrainerConfig(**merged)


def validate_config_for_training(raw_config: Dict[str, Any]) -> ParsedConfig:
    """
    Validate configuration for training entry point.

    This is the SINGLE validation entry point for training.

    Args:
        raw_config: Raw configuration dictionary from JSON file

    Returns:
        ParsedConfig containing validated TrainerConfig and raw config

    Raises:
        ValueError: If configuration validation fails
    """
    trainer_config = parse_raw_config(raw_config)

    # Additional training-specific validations
    # TrainerConfig.__post_init__ already validates:
    # - split and validation_split are in ALLOWED_SPLITS
    # - rl_test barrier for both splits
    # - M5 formal trainer requires split='predictor_train' and validation_split='rl_validation'
    # - maintenance_capacity in (1, 2)
    # - scenario bank paths provided
    # - numeric params positive

    # Training-specific: ensure distinct training/validation scenario banks
    if trainer_config.training_scenario_bank_path == trainer_config.validation_scenario_bank_path:
        raise ValueError(
            "M5 formal training requires distinct training_scenario_bank_path and "
            "validation_scenario_bank_path. They cannot be the same file."
        )

    return ParsedConfig(trainer_config=trainer_config, raw_config=raw_config)


def validate_config_for_evaluation(raw_config: Dict[str, Any]) -> ParsedConfig:
    """
    Validate configuration for evaluation entry point.

    This is the SINGLE validation entry point for evaluation.

    Args:
        raw_config: Raw configuration dictionary from JSON file

    Returns:
        ParsedConfig containing validated TrainerConfig and raw config

    Raises:
        ValueError: If configuration validation fails
    """
    trainer_config = parse_raw_config(raw_config)

    # Evaluation-specific validations
    # TrainerConfig.__post_init__ already validates:
    # - split and validation_split in ALLOWED_SPLITS
    # - rl_test barrier for both splits
    # - maintenance_capacity in (1, 2)
    # - numeric params positive

    # Evaluation-specific: validation_scenario_bank_path MUST be specified (schema v3 fail-closed)
    if trainer_config.validation_scenario_bank_path is None:
        raise ValueError(
            "Evaluation config missing validation_scenario_bank_path (schema v3 required). "
            "Evaluation config must explicitly specify validation_scenario_bank_path. "
            "Fallback to K-based default paths is not allowed for schema v3."
        )

    return ParsedConfig(trainer_config=trainer_config, raw_config=raw_config)


def validate_config_for_matrix(raw_config: Dict[str, Any]) -> ParsedConfig:
    """
    Validate configuration for matrix generation.

    Matrix generation uses dry-run mode and needs config validation.

    Args:
        raw_config: Raw configuration dictionary from JSON file

    Returns:
        ParsedConfig containing validated TrainerConfig and raw config

    Raises:
        ValueError: If configuration validation fails
    """
    trainer_config = parse_raw_config(raw_config)

    # Matrix generation uses dry-run; validation is same as training but
    # we don't require distinct scenario banks for dry-run validation
    # (the actual runs will use distinct banks from matrix spec)

    return ParsedConfig(trainer_config=trainer_config, raw_config=raw_config)


def load_and_validate_config(config_path: Path | str, mode: str = "training") -> ParsedConfig:
    """
    Load config file and validate with appropriate validator.

    Args:
        config_path: Path to configuration JSON file
        mode: One of "training", "evaluation", "matrix"

    Returns:
        ParsedConfig with validated TrainerConfig

    Raises:
        SystemExit: On file not found or parse error
        ValueError: On validation failure
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise SystemExit(f"ERROR: Configuration file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as f:
            raw_config = json.load(f)
    except json.JSONDecodeError as e:
        raise SystemExit(f"ERROR: Config JSON parse error: {e}")

    if mode == "training":
        return validate_config_for_training(raw_config)
    elif mode == "evaluation":
        return validate_config_for_evaluation(raw_config)
    elif mode == "matrix":
        return validate_config_for_matrix(raw_config)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def apply_cli_overrides(
    trainer_config: TrainerConfig,
    overrides: Dict[str, Any],
) -> TrainerConfig:
    """
    Apply CLI overrides to TrainerConfig.

    Since TrainerConfig is frozen, creates a new instance with merged values.

    Args:
        trainer_config: Base TrainerConfig
        overrides: Dictionary of override values (None values are ignored)

    Returns:
        New TrainerConfig with overrides applied
    """
    # Build merged dict from existing config + overrides
    merged = trainer_config.to_dict()

    for key, value in overrides.items():
        if value is not None:
            merged[key] = value

    return TrainerConfig(**merged)