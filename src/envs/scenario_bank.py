"""
Scenario bank data structures for Milestone 2 Selective Maintenance Environment.

Implements scenario definitions and validators for serialized fixed scenario banks.
Each scenario defines initial fleet state, replacement RNG seed, and episode parameters.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Set

from .config import EnvironmentConfig, ALLOWED_SPLITS


class PredictionStoreLike(Protocol):
    """Protocol for prediction store lookup.

    Matches the actual PredictionStore interface from src/predictors/prediction_store.py.
    """

    def get(
        self,
        split: str,
        unit_id: int,
        cycle: int,
    ) -> Any:
        """Get prediction for a specific (split, unit_id, cycle).

        Returns a PredictionResult-like object with a 'found' attribute.
        """
        ...

    def get_units(self, split: str) -> list[int]:
        """Get all unit IDs for a given split."""
        ...


@dataclass(frozen=True)
class Scenario:
    """
    A single scenario definition for episode initialization.

    Attributes:
        scenario_id: Unique identifier for this scenario.
        split: Environment split (predictor_train, rl_validation, rl_test).
        initial_unit_ids: List of N unit IDs for initial fleet state.
        initial_cycles: List of N initial cycle indices (one per unit).
        replacement_seed: RNG seed for replacement sampling in this scenario.
        environment_seed: Master environment seed for this scenario.
        episode_horizon: Number of decision windows in this episode.
        maintenance_capacity: K value for this scenario.
        cost_regime_id: Cost regime identifier.
    """

    scenario_id: str
    split: str
    initial_unit_ids: tuple[int, ...]
    initial_cycles: tuple[int, ...]
    replacement_seed: int
    environment_seed: int
    episode_horizon: int
    maintenance_capacity: int
    cost_regime_id: str

    def __post_init__(self) -> None:
        """Basic structural validation (does not check against PredictionStore)."""
        errors: list[str] = []

        # Check N = 5
        if len(self.initial_unit_ids) != 5:
            errors.append(
                f"initial_unit_ids must have exactly 5 units, got {len(self.initial_unit_ids)}"
            )

        # Check N cycles
        if len(self.initial_cycles) != 5:
            errors.append(
                f"initial_cycles must have exactly 5 values, got {len(self.initial_cycles)}"
            )

        # Check all cycles are positive
        for i, cycle in enumerate(self.initial_cycles):
            if cycle <= 0:
                errors.append(f"initial_cycles[{i}] must be positive, got {cycle}")

        # Check episode_horizon > 0
        if self.episode_horizon <= 0:
            errors.append(
                f"episode_horizon must be positive, got {self.episode_horizon}"
            )

        # Check maintenance_capacity in valid range
        if self.maintenance_capacity < 0 or self.maintenance_capacity > 5:
            errors.append(
                f"maintenance_capacity must be in [0, 5], got {self.maintenance_capacity}"
            )

        if errors:
            raise ValueError("Scenario validation failed:\n  - " + "\n  - ".join(errors))

    def get_initial_age(self, unit_index: int) -> int:
        """
        Compute initial age for a slot based on initial cycle.

        age_since_replacement_cycles = initial_cycle - 1

        Examples:
            cycle 1 -> age 0
            cycle 81 -> age 80
        """
        return self.initial_cycles[unit_index] - 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary with deterministic ordering."""
        return {
            "scenario_id": self.scenario_id,
            "split": self.split,
            "initial_unit_ids": list(self.initial_unit_ids),
            "initial_cycles": list(self.initial_cycles),
            "replacement_seed": self.replacement_seed,
            "environment_seed": self.environment_seed,
            "episode_horizon": self.episode_horizon,
            "maintenance_capacity": self.maintenance_capacity,
            "cost_regime_id": self.cost_regime_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scenario":
        """Create from dictionary."""
        # Convert lists to tuples for immutability
        return cls(
            scenario_id=str(data["scenario_id"]),
            split=str(data["split"]),
            initial_unit_ids=tuple(data["initial_unit_ids"]),
            initial_cycles=tuple(data["initial_cycles"]),
            replacement_seed=int(data["replacement_seed"]),
            environment_seed=int(data["environment_seed"]),
            episode_horizon=int(data["episode_horizon"]),
            maintenance_capacity=int(data["maintenance_capacity"]),
            cost_regime_id=str(data["cost_regime_id"]),
        )

    def serialize_deterministic(self) -> str:
        """Return a deterministic JSON string representation."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def compute_hash(self) -> str:
        """Compute a SHA256 hash of this scenario for deduplication."""
        return hashlib.sha256(self.serialize_deterministic().encode()).hexdigest()


@dataclass(frozen=True)
class ScenarioBank:
    """
    A collection of scenarios for a specific split.

    Attributes:
        bank_id: Unique identifier for this scenario bank.
        split: The split these scenarios belong to.
        scenarios: List of Scenario objects.
    """

    bank_id: str
    split: str
    scenarios: tuple[Scenario, ...]

    def __post_init__(self) -> None:
        """Validate scenario bank structure."""
        errors: list[str] = []

        # Check all scenarios have the correct split
        for scenario in self.scenarios:
            if scenario.split != self.split:
                errors.append(
                    f"Scenario {scenario.scenario_id} has split '{scenario.split}', "
                    f"expected '{self.split}'"
                )

        # Check unique scenario IDs
        seen_ids: set[str] = set()
        for scenario in self.scenarios:
            if scenario.scenario_id in seen_ids:
                errors.append(f"Duplicate scenario ID: {scenario.scenario_id}")
            seen_ids.add(scenario.scenario_id)

        if errors:
            raise ValueError("ScenarioBank validation failed:\n  - " + "\n  - ".join(errors))

    def __len__(self) -> int:
        """Return number of scenarios in this bank."""
        return len(self.scenarios)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "bank_id": self.bank_id,
            "split": self.split,
            "scenarios": [s.to_dict() for s in self.scenarios],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScenarioBank":
        """Create from dictionary."""
        return cls(
            bank_id=str(data["bank_id"]),
            split=str(data["split"]),
            scenarios=tuple(Scenario.from_dict(s) for s in data["scenarios"]),
        )

    def serialize_deterministic(self) -> str:
        """Return a deterministic JSON string representation."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def load_scenario_bank(path: Path | str) -> ScenarioBank:
    """
    Load a scenario bank from a JSON file.

    Args:
        path: Path to the scenario bank JSON file.

    Returns:
        Validated ScenarioBank instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If validation fails.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return ScenarioBank.from_dict(data)


def save_scenario_bank(bank: ScenarioBank, path: Path | str) -> None:
    """
    Save a scenario bank to a JSON file.

    Args:
        bank: The scenario bank to save.
        path: Path to the output JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(bank.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")


def validate_scenario_against_config(
    scenario: Scenario, config: EnvironmentConfig
) -> bool:
    """
    Validate a scenario against an environment configuration.

    Args:
        scenario: The scenario to validate.
        config: The environment configuration.

    Returns:
        True if valid.

    Raises:
        ValueError: If validation fails.
    """
    errors: list[str] = []

    # Check episode_horizon matches
    if scenario.episode_horizon != config.episode_horizon:
        errors.append(
            f"Scenario episode_horizon ({scenario.episode_horizon}) does not match "
            f"config ({config.episode_horizon})"
        )

    # Check maintenance_capacity matches
    if scenario.maintenance_capacity != config.maintenance_capacity:
        errors.append(
            f"Scenario maintenance_capacity ({scenario.maintenance_capacity}) does not match "
            f"config ({config.maintenance_capacity})"
        )

    if errors:
        raise ValueError("Scenario/config mismatch:\n  - " + "\n  - ".join(errors))

    return True


def validate_scenario_units_against_split(
    scenario: Scenario,
    prediction_store: PredictionStoreLike,
) -> bool:
    """
    Validate that all unit IDs in a scenario belong to the declared split.

    Uses the PredictionStore.get_units(split) method to get valid units for
    the scenario's declared split, then checks that all scenario units are
    in that set.

    Args:
        scenario: The scenario to validate.
        prediction_store: PredictionStore for unit split lookup.

    Returns:
        True if valid.

    Raises:
        ValueError: If any unit does not belong to the scenario's split.
    """
    errors: list[str] = []

    # Get the set of valid units for the scenario's declared split
    try:
        valid_units: Set[int] = set(prediction_store.get_units(scenario.split))
    except (KeyError, AttributeError) as e:
        raise ValueError(
            f"Cannot get units for split '{scenario.split}': {e}"
        ) from e

    for unit_id in scenario.initial_unit_ids:
        if unit_id not in valid_units:
            # Check if unit exists in any split (including predictor_validation)
            found_in_split = None
            all_splits = ALLOWED_SPLITS | {"predictor_validation"}
            for split in all_splits:
                try:
                    if unit_id in set(prediction_store.get_units(split)):
                        found_in_split = split
                        break
                except (KeyError, AttributeError):
                    continue

            if found_in_split:
                errors.append(
                    f"Unit {unit_id} belongs to split '{found_in_split}', "
                    f"but scenario declares split '{scenario.split}'"
                )
            else:
                errors.append(f"Unit {unit_id} not found in any split")

    if errors:
        raise ValueError("Unit/split validation failed:\n  - " + "\n  - ".join(errors))

    return True


def validate_scenario_cycles_exist(
    scenario: Scenario,
    prediction_store: PredictionStoreLike,
) -> bool:
    """
    Validate that all initial cycles have valid predictions in the store.

    Uses the PredictionStore.get(split, unit_id, cycle) method which returns
    a PredictionResult-like object with a 'found' attribute.

    Args:
        scenario: The scenario to validate.
        prediction_store: PredictionStore for cycle lookup.

    Returns:
        True if all cycles exist.

    Raises:
        ValueError: If any cycle is missing.
    """
    errors: list[str] = []

    for unit_id, cycle in zip(scenario.initial_unit_ids, scenario.initial_cycles):
        result = prediction_store.get(scenario.split, unit_id, cycle)
        if not result.found:
            errors.append(
                f"Missing prediction for unit {unit_id} at cycle {cycle} "
                f"(split: {scenario.split})"
            )

    if errors:
        raise ValueError("Cycle validation failed:\n  - " + "\n  - ".join(errors))

    return True

    if errors:
        raise ValueError("Cycle validation failed:\n  - " + "\n  - ".join(errors))

    return True


def validate_full_scenario_bank(
    bank: ScenarioBank,
    prediction_store: PredictionStoreLike,
) -> bool:
    """
    Perform full validation of a scenario bank.

    Validates:
    - Structural integrity (from ScenarioBank.__post_init__)
    - All units belong to the declared split
    - All initial cycles exist in the prediction store

    Args:
        bank: The scenario bank to validate.
        prediction_store: PredictionStore for lookups.

    Returns:
        True if fully valid.

    Raises:
        ValueError: If any validation step fails.
    """
    # Structural validation already done in __post_init__

    # Validate each scenario
    for scenario in bank.scenarios:
        validate_scenario_units_against_split(scenario, prediction_store)
        validate_scenario_cycles_exist(scenario, prediction_store)

    return True