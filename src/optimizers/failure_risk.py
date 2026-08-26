"""
Failure risk models for Milestone 4 Exact Myopic Optimizer.

Implements deterministic failure-risk estimation for the current decision window.

Two risk models are provided:
1. hard_window_v1: Binary risk (0 or 1) based on 5-cycle window
2. logistic_window_v1: Smooth risk in (0, 1) with temperature parameter

All risk estimates use predicted RUL (cycles) from the observation,
never true RUL from hidden simulator state.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

import numpy as np


class RiskModelId(str, Enum):
    """Identifiers for failure risk models."""

    HARD_WINDOW_V1 = "hard_window_v1"
    LOGISTIC_WINDOW_V1 = "logistic_window_v1"


def compute_hard_window_risk(
    predicted_rul_cycles: np.ndarray | float,
    delta_cycles: int = 5,
) -> np.ndarray | float:
    """
    Compute binary failure risk based on 5-cycle decision window.

    Risk model: hard_window_v1

    ```
    p_fail = 1  if predicted_rul_cycles <= delta_cycles
           = 0  otherwise
    ```

    Args:
        predicted_rul_cycles: Predicted RUL in cycles (denormalized).
                             Can be scalar or ndarray of any shape.
        delta_cycles: Decision window (default 5).

    Returns:
        Failure probability (0 or 1), same shape as input.

    Raises:
        ValueError: If predicted_rul_cycles contains non-finite values.
    """
    predicted_rul_cycles = np.asarray(predicted_rul_cycles, dtype=np.float64)

    # Validate finiteness
    if not np.all(np.isfinite(predicted_rul_cycles)):
        raise ValueError(
            f"predicted_rul_cycles must be finite, "
            f"got non-finite values: {predicted_rul_cycles[~np.isfinite(predicted_rul_cycles)]}"
        )

    # Validate non-negative
    if np.any(predicted_rul_cycles < 0):
        raise ValueError(
            f"predicted_rul_cycles must be non-negative, "
            f"got min={predicted_rul_cycles.min():.2f}"
        )

    # Binary risk: 1 if RUL <= delta_cycles, else 0
    risk = (predicted_rul_cycles <= delta_cycles).astype(np.float64)

    return risk


def compute_logistic_window_risk(
    predicted_rul_cycles: np.ndarray | float,
    delta_cycles: int = 5,
    temperature: float = 10.0,
) -> np.ndarray | float:
    """
    Compute smooth logistic failure risk.

    Risk model: logistic_window_v1

    ```
    p_fail = sigmoid((delta_cycles - predicted_rul_cycles) / temperature)
           = 1 / (1 + exp((predicted_rul_cycles - delta_cycles) / temperature))
    ```

    Properties:
    - Risk in (0, 1) for finite inputs
    - Risk = 0.5 when predicted_rul_cycles == delta_cycles
    - Risk decreases as predicted_rul_cycles increases
    - Temperature controls steepness (higher = smoother)

    Args:
        predicted_rul_cycles: Predicted RUL in cycles (denormalized).
                             Can be scalar or ndarray of any shape.
        delta_cycles: Decision window (default 5).
        temperature: Logistic steepness parameter. Must be finite and > 0.

    Returns:
        Failure probability in (0, 1), same shape as input.

    Raises:
        ValueError: If temperature <= 0, not finite, or predicted_rul_cycles not finite.
    """
    # Validate temperature
    if not isinstance(temperature, (int, float)):
        raise ValueError(f"temperature must be numeric, got {type(temperature).__name__}")
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if not np.isfinite(temperature):
        raise ValueError(f"temperature must be finite, got {temperature}")

    predicted_rul_cycles = np.asarray(predicted_rul_cycles, dtype=np.float64)

    # Validate finiteness
    if not np.all(np.isfinite(predicted_rul_cycles)):
        raise ValueError(
            f"predicted_rul_cycles must be finite, "
            f"got non-finite values"
        )

    # Validate non-negative
    if np.any(predicted_rul_cycles < 0):
        raise ValueError(
            f"predicted_rul_cycles must be non-negative, "
            f"got min={predicted_rul_cycles.min():.2f}"
        )

    # Compute logistic risk
    # sigmoid((delta - rul) / temp) = 1 / (1 + exp((rul - delta) / temp))
    arg = (predicted_rul_cycles - delta_cycles) / temperature
    risk = 1.0 / (1.0 + np.exp(arg))

    return risk


def compute_failure_risk(
    predicted_rul_cycles: np.ndarray | float,
    delta_cycles: int = 5,
    risk_model_id: Literal["hard_window_v1", "logistic_window_v1"] = "hard_window_v1",
    risk_temperature: float = 10.0,
) -> np.ndarray | float:
    """
    Compute failure risk using specified model.

    Args:
        predicted_rul_cycles: Predicted RUL in cycles (denormalized).
        delta_cycles: Decision window (default 5).
        risk_model_id: Risk model identifier.
        risk_temperature: Temperature for logistic model (ignored for hard).

    Returns:
        Failure probability, same shape as input.

    Raises:
        ValueError: If risk_model_id is unknown or parameters invalid.
    """
    if risk_model_id == "hard_window_v1":
        return compute_hard_window_risk(
            predicted_rul_cycles=predicted_rul_cycles,
            delta_cycles=delta_cycles,
        )
    elif risk_model_id == "logistic_window_v1":
        return compute_logistic_window_risk(
            predicted_rul_cycles=predicted_rul_cycles,
            delta_cycles=delta_cycles,
            temperature=risk_temperature,
        )
    else:
        raise ValueError(
            f"Unknown risk_model_id: {risk_model_id}. "
            f"Valid options: 'hard_window_v1', 'logistic_window_v1'"
        )


def validate_risk_model_parameters(
    risk_model_id: str,
    risk_temperature: float = 10.0,
) -> None:
    """
    Validate risk model parameters.

    Args:
        risk_model_id: Risk model identifier.
        risk_temperature: Temperature for logistic model.

    Raises:
        ValueError: If parameters are invalid.
    """
    if risk_model_id not in {"hard_window_v1", "logistic_window_v1"}:
        raise ValueError(
            f"Unknown risk_model_id: {risk_model_id}. "
            f"Valid options: 'hard_window_v1', 'logistic_window_v1'"
        )

    if risk_model_id == "logistic_window_v1":
        if risk_temperature <= 0:
            raise ValueError(
                f"logistic_window_v1 requires temperature > 0, got {risk_temperature}"
            )
        if not np.isfinite(risk_temperature):
            raise ValueError(
                f"logistic_window_v1 requires finite temperature, got {risk_temperature}"
            )


def get_risk_model_description(risk_model_id: str) -> str:
    """
    Get human-readable description of a risk model.

    Args:
        risk_model_id: Risk model identifier.

    Returns:
        Description string.
    """
    descriptions = {
        "hard_window_v1": (
            "Binary failure risk: p=1 if predicted_rul <= delta_cycles, else 0. "
            "Direct encoding of the 5-cycle decision window."
        ),
        "logistic_window_v1": (
            "Smooth logistic failure risk: p=sigmoid((delta-rul)/temperature). "
            "Risk in (0,1), steeper with lower temperature."
        ),
    }
    return descriptions.get(risk_model_id, f"Unknown risk model: {risk_model_id}")