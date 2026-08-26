"""
Cost regime definitions for Milestone 2 Selective Maintenance Environment.

Implements immutable cost regime configurations for the shared cost equation:

    C_t = c_pm * N_pm + c_f * N_fail + c_u * sum(true_rul / 125 for maintained slots)
    reward_t = -C_t

Four frozen cost regimes are supported:
    1. failure-light, no-waste:     c_pm=1.0, c_f=5.0,  c_u=0.0
    2. failure-heavy, no-waste:     c_pm=1.0, c_f=10.0, c_u=0.0
    3. failure-light, waste-aware:  c_pm=1.0, c_f=5.0,  c_u=0.25
    4. failure-heavy, waste-aware:  c_pm=1.0, c_f=10.0, c_u=0.25
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, FrozenSet


@dataclass(frozen=True)
class CostRegime:
    """
    Immutable cost regime configuration.

    Attributes:
        c_pm: Preventive maintenance cost per engine.
        c_f: Failure cost per engine.
        c_u: Unused remaining life cost coefficient (per unit of wasted RUL).
        regime_id: Unique identifier for this regime.
    """

    c_pm: float
    c_f: float
    c_u: float
    regime_id: str

    def __post_init__(self) -> None:
        """Validate cost regime coefficients."""
        # Check all coefficients are finite and non-negative
        for name, value in [("c_pm", self.c_pm), ("c_f", self.c_f), ("c_u", self.c_u)]:
            if not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric, got {type(value).__name__}")
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
            if value != value:  # NaN check
                raise ValueError(f"{name} cannot be NaN")
            if value == float("inf"):
                raise ValueError(f"{name} cannot be infinite")

    def to_dict(self) -> dict[str, float | str]:
        """Serialize to dictionary."""
        return {
            "c_pm": self.c_pm,
            "c_f": self.c_f,
            "c_u": self.c_u,
            "regime_id": self.regime_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CostRegime":
        """
        Create from dictionary.

        Raises:
            ValueError: If required keys are missing or values are invalid.
        """
        required_keys = {"c_pm", "c_f", "c_u", "regime_id"}
        missing = required_keys - set(data.keys())
        if missing:
            raise ValueError(f"Missing required keys: {missing}")

        return cls(
            c_pm=float(data["c_pm"]),
            c_f=float(data["c_f"]),
            c_u=float(data["c_u"]),
            regime_id=str(data["regime_id"]),
        )

    def with_updates(self, **kwargs: Any) -> "CostRegime":
        """Create a modified copy (for testing)."""
        return replace(self, **kwargs)


# Frozen cost regime instances
FAILURE_LIGHT_NO_WASTE = CostRegime(
    c_pm=1.0, c_f=5.0, c_u=0.0, regime_id="failure-light-no-waste"
)

FAILURE_HEAVY_NO_WASTE = CostRegime(
    c_pm=1.0, c_f=10.0, c_u=0.0, regime_id="failure-heavy-no-waste"
)

FAILURE_LIGHT_WASTE_AWARE = CostRegime(
    c_pm=1.0, c_f=5.0, c_u=0.25, regime_id="failure-light-waste-aware"
)

FAILURE_HEAVY_WASTE_AWARE = CostRegime(
    c_pm=1.0, c_f=10.0, c_u=0.25, regime_id="failure-heavy-waste-aware"
)

# Registry of all valid cost regimes
COST_REGIMES: dict[str, CostRegime] = {
    regime.regime_id: regime
    for regime in [
        FAILURE_LIGHT_NO_WASTE,
        FAILURE_HEAVY_NO_WASTE,
        FAILURE_LIGHT_WASTE_AWARE,
        FAILURE_HEAVY_WASTE_AWARE,
    ]
}

# Default cost regime for initial experiments
DEFAULT_COST_REGIME_ID = "failure-light-no-waste"


def get_cost_regime(regime_id: str) -> CostRegime:
    """
    Get a cost regime by ID.

    Args:
        regime_id: The regime identifier.

    Returns:
        The matching CostRegime instance.

    Raises:
        ValueError: If regime_id is not found.
    """
    if regime_id not in COST_REGIMES:
        available = ", ".join(sorted(COST_REGIMES.keys()))
        raise ValueError(
            f"Unknown cost regime '{regime_id}'. Available: {available}"
        )
    return COST_REGIMES[regime_id]


def list_cost_regimes() -> list[str]:
    """Return list of available cost regime IDs."""
    return sorted(COST_REGIMES.keys())


def validate_cost_regime(regime_id: str) -> bool:
    """
    Validate that a regime ID exists and has valid coefficients.

    Args:
        regime_id: The regime identifier to validate.

    Returns:
        True if valid.

    Raises:
        ValueError: If regime is unknown or invalid.
    """
    regime = get_cost_regime(regime_id)
    # Access properties to trigger __post_init__ validation
    _ = regime.c_pm, regime.c_f, regime.c_u
    return True


def calculate_total_cost(
    num_preventive: int,
    num_failures: int,
    wasted_rul_sum: float,
    regime: CostRegime,
) -> float:
    """
    Calculate total cost for one decision window.

    Args:
        num_preventive: Number of engines preventively maintained.
        num_failures: Number of engines that failed.
        wasted_rul_sum: Sum of (true_rul / RUL_MAX) for all preventively maintained slots.
        regime: The cost regime to use.

    Returns:
        Total cost C_t for this step.
    """
    pm_cost = regime.c_pm * num_preventive
    failure_cost = regime.c_f * num_failures
    waste_cost = regime.c_u * wasted_rul_sum

    return pm_cost + failure_cost + waste_cost