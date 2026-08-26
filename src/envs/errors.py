"""
Custom exceptions for Milestone 2 Selective Maintenance Environment.

Defines environment-specific exception classes for clear error signaling:
- MissingPredictionError: Required prediction not found in PredictionStore
- ContractViolationError: Fundamental contract violation (e.g., no failure endpoint)
- InvalidActionError: Action ID invalid or step called at wrong time
- InformationLeakageError: Hidden field accidentally exposed
- SplitViolationError: Cross-split access attempt
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class Milestone2EnvironmentError(Exception):
    """Base exception for Milestone 2 environment errors."""

    pass


class MissingPredictionError(Milestone2EnvironmentError):
    """
    Raised when a required prediction is not found in PredictionStore.

    Contains full context for debugging:
    - split
    - unit_id
    - requested cycle
    - slot index
    - environment step
    """

    def __init__(
        self,
        split: str,
        unit_id: int,
        cycle: int,
        slot_index: int,
        env_step: int,
        message: Optional[str] = None,
    ):
        self.split = split
        self.unit_id = unit_id
        self.cycle = cycle
        self.slot_index = slot_index
        self.env_step = env_step

        if message is None:
            message = (
                f"Missing prediction in PredictionStore: "
                f"split='{split}', unit_id={unit_id}, cycle={cycle}, "
                f"slot_index={slot_index}, env_step={env_step}"
            )

        super().__init__(message)


class ContractViolationError(Milestone2EnvironmentError):
    """
    Raised for fundamental contract violations.

    Examples:
    - Trajectory ends without true_rul <= 0 record
    - Failure endpoint cannot be determined
    - Age or RUL normalization produces non-finite values
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        self.context = context or {}
        super().__init__(message)


class InvalidActionError(Milestone2EnvironmentError):
    """
    Raised for invalid action IDs or step timing violations.

    Examples:
    - Step called before reset
    - Step called after truncation
    - Non-integral action ID
    - Boolean action ID
    - Action ID outside action table
    """

    def __init__(self, message: str, action_id: Optional[Any] = None):
        self.action_id = action_id
        super().__init__(message)


class InformationLeakageError(Milestone2EnvironmentError):
    """
    Raised when hidden information is accidentally exposed.

    Examples:
    - True RUL in observation
    - Trajectory ID in normal info
    - Unit ID in agent-visible data
    """

    def __init__(self, message: str, field: Optional[str] = None):
        self.field = field
        super().__init__(message)


class SplitViolationError(Milestone2EnvironmentError):
    """
    Raised for cross-split access attempts.

    Examples:
    - Sampling replacement from wrong split
    - Scenario unit not belonging to declared split
    """

    def __init__(
        self,
        expected_split: str,
        actual_split: str,
        unit_id: Optional[int] = None,
    ):
        self.expected_split = expected_split
        self.actual_split = actual_split
        self.unit_id = unit_id

        message = (
            f"Split violation: expected '{expected_split}', "
            f"got '{actual_split}'"
        )
        if unit_id is not None:
            message += f" for unit {unit_id}"

        super().__init__(message)


class ScenarioValidationError(Milestone2EnvironmentError):
    """
    Raised when a scenario fails validation at reset time.

    Examples:
    - Scenario split does not match config split
    - Scenario K does not match config K
    - Initial cycle has true_rul <= 0
    - Required prediction missing for initial state
    """

    def __init__(
        self,
        scenario_id: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.scenario_id = scenario_id
        self.reason = reason
        self.details = details or {}

        message = f"Scenario '{scenario_id}' validation failed: {reason}"
        super().__init__(message)