"""
M6 Receding-Horizon Planning Contract Definitions.

This module defines the frozen dataclasses and types for M6 planning contracts.
All identities are frozen per the M6 V2 contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


# Frozen cost regimes (from MILESTONE_6_RECEDING_HORIZON_CONTRACT.md)
COST_REGIMES: Dict[str, Dict[str, float]] = {
    "failure-light-no-waste": {"c_pm": 1.0, "c_f": 5.0, "c_u": 0.0},
    "failure-heavy-no-waste": {"c_pm": 1.0, "c_f": 10.0, "c_u": 0.0},
    "failure-light-waste-aware": {"c_pm": 1.0, "c_f": 5.0, "c_u": 0.25},
    "failure-heavy-waste-aware": {"c_pm": 1.0, "c_f": 10.0, "c_u": 0.25},
}

# Frozen M4/M5 identities
M4_SELECTED_CANDIDATE = "logistic_T5"
M4_RISK_MODEL_ID = "logistic_window_v1"
M4_RISK_TEMPERATURE = 5.0
M4_DELTA_CYCLES = 5
M4_TIE_TOLERANCE = 1e-9
M4_RUL_SCALE = 125.0
M4_AGE_SCALE_CYCLES = 341
M4_FLEET_SIZE = 5

M5_GAMMA = 0.95
M5_OBSERVATION_SCHEMA_ID = "m5_point_v1"
M5_OBSERVATION_DIM = 10
M5_PREDICTION_CACHE_MANIFEST_SHA256 = (
    "007c36af6cc0f5e1ffd2ce7b254a5177bd76c6d6bb487a2dcd05a53c32e44fd0"
)

# Action table hashes (frozen)
ACTION_TABLE_K1_SHA256 = (
    "1e6d24ad856e122c7440b75173f2fbdfae4e9af2a30ff8de79200d1afa875ccf"
)
ACTION_TABLE_K2_SHA256 = (
    "20212080a85d1d82a6aa3031042ea04bd0a7c2a0e60ffb314ed73f3f17dbc2b4"
)

# Environment contract
ENVIRONMENT_CONTRACT_ID = "m2_v1"

# H1 planner identity
H1_PLANNER_ID = "m6_h1_v1"

# H2 planner identity
H2_PLANNER_ID = "m6_h2_v1"


class ContractViolationError(ValueError):
    """Raised when a frozen contract identity is violated."""
    pass


class IdentityMismatchError(ContractViolationError):
    """Raised when a frozen identity check fails."""
    pass


class SplitViolationError(ContractViolationError):
    """Raised when rl_test split is accessed."""
    pass


@dataclass(frozen=True)
class PlannerContext:
    """
    Immutable planner context carrying all frozen configuration.

    This is the single source of truth for M6 planning parameters.
    All fields are validated on construction.
    """

    maintenance_capacity: int              # 1 or 2 (K)
    delta_cycles: int                      # 5 (frozen)
    rul_scale: float                       # 125.0 (frozen)
    age_scale_cycles: int                  # 341 (frozen)
    action_table: Tuple[Tuple[int, ...], ...]
    action_table_sha256: str               # frozen per K
    cost_regime_id: str                    # one of 4 frozen regimes
    c_pm: float
    c_f: float
    c_u: float
    risk_model_id: str                     # "logistic_window_v1"
    risk_temperature: float                # 5.0 (NOT 10.0)
    gamma: float                           # 0.95 (M5 frozen)
    observation_schema_id: str             # "m5_point_v1"
    environment_contract_id: str           # "m2_v1"
    prediction_cache_manifest_sha256: str  # frozen manifest hash
    horizon: int                           # 1 for H1, 2 for H2, 0 for DDQN
    forbid_rl_test: bool = True            # always True

    # H2-specific (null for H1/DDQN/M3/M4)
    R1_hat_cycles: float | None = None
    R1_hat_provenance: Dict[str, str] | None = None

    def __post_init__(self) -> None:
        """Validate all frozen identities."""
        # Maintenance capacity
        if self.maintenance_capacity not in (1, 2):
            raise IdentityMismatchError(
                f"maintenance_capacity must be 1 or 2, got {self.maintenance_capacity}"
            )

        # Frozen constants
        if self.delta_cycles != 5:
            raise IdentityMismatchError(
                f"delta_cycles must be 5, got {self.delta_cycles}"
            )
        if self.rul_scale != 125.0:
            raise IdentityMismatchError(
                f"rul_scale must be 125.0, got {self.rul_scale}"
            )
        if self.age_scale_cycles != 341:
            raise IdentityMismatchError(
                f"age_scale_cycles must be 341, got {self.age_scale_cycles}"
            )
        if self.gamma != 0.95:
            raise IdentityMismatchError(
                f"gamma must be 0.95, got {self.gamma}"
            )
        if self.risk_model_id != M4_RISK_MODEL_ID:
            raise IdentityMismatchError(
                f"risk_model_id must be '{M4_RISK_MODEL_ID}', got '{self.risk_model_id}'"
            )
        if self.risk_temperature != M4_RISK_TEMPERATURE:
            raise IdentityMismatchError(
                f"risk_temperature must be {M4_RISK_TEMPERATURE}, got {self.risk_temperature}"
            )
        if self.observation_schema_id != M5_OBSERVATION_SCHEMA_ID:
            raise IdentityMismatchError(
                f"observation_schema_id must be '{M5_OBSERVATION_SCHEMA_ID}', "
                f"got '{self.observation_schema_id}'"
            )
        if self.environment_contract_id != ENVIRONMENT_CONTRACT_ID:
            raise IdentityMismatchError(
                f"environment_contract_id must be '{ENVIRONMENT_CONTRACT_ID}', "
                f"got '{self.environment_contract_id}'"
            )
        if self.prediction_cache_manifest_sha256 != M5_PREDICTION_CACHE_MANIFEST_SHA256:
            raise IdentityMismatchError(
                f"prediction_cache_manifest_sha256 mismatch: "
                f"expected {M5_PREDICTION_CACHE_MANIFEST_SHA256}, "
                f"got {self.prediction_cache_manifest_sha256}"
            )

        # Cost regime validation
        if self.cost_regime_id not in COST_REGIMES:
            raise IdentityMismatchError(
                f"cost_regime_id must be one of {list(COST_REGIMES.keys())}, "
                f"got '{self.cost_regime_id}'"
            )
        regime = COST_REGIMES[self.cost_regime_id]
        if self.c_pm != regime["c_pm"]:
            raise IdentityMismatchError(
                f"c_pm mismatch for regime {self.cost_regime_id}: "
                f"expected {regime['c_pm']}, got {self.c_pm}"
            )
        if self.c_f != regime["c_f"]:
            raise IdentityMismatchError(
                f"c_f mismatch for regime {self.cost_regime_id}: "
                f"expected {regime['c_f']}, got {self.c_f}"
            )
        if self.c_u != regime["c_u"]:
            raise IdentityMismatchError(
                f"c_u mismatch for regime {self.cost_regime_id}: "
                f"expected {regime['c_u']}, got {self.c_u}"
            )

        # Action table hash validation
        expected_hash = (
            ACTION_TABLE_K1_SHA256 if self.maintenance_capacity == 1
            else ACTION_TABLE_K2_SHA256
        )
        if self.action_table_sha256 != expected_hash:
            raise IdentityMismatchError(
                f"action_table_sha256 mismatch for K={self.maintenance_capacity}: "
                f"expected {expected_hash}, got {self.action_table_sha256}"
            )

        # Horizon validation
        if self.horizon not in (0, 1, 2):
            raise IdentityMismatchError(
                f"horizon must be 0, 1, or 2, got {self.horizon}"
            )

        # H2-specific validation
        if self.horizon == 2:
            if self.R1_hat_cycles is None:
                raise IdentityMismatchError(
                    "R1_hat_cycles required for H2 (horizon=2)"
                )
            if self.R1_hat_provenance is None:
                raise IdentityMismatchError(
                    "R1_hat_provenance required for H2 (horizon=2)"
                )
            required_provenance_keys = {
                "predictor_train_manifest_sha256",
                "computed_at_utc",
                "n_cycle1_records",
            }
            if not required_provenance_keys.issubset(self.R1_hat_provenance.keys()):
                raise IdentityMismatchError(
                    f"R1_hat_provenance missing required keys: "
                    f"{required_provenance_keys - self.R1_hat_provenance.keys()}"
                )
        else:
            # H1/DDQN/M3/M4 must have null H2 fields
            if self.R1_hat_cycles is not None:
                raise IdentityMismatchError(
                    f"R1_hat_cycles must be null for horizon={self.horizon}, "
                    f"got {self.R1_hat_cycles}"
                )
            if self.R1_hat_provenance is not None:
                raise IdentityMismatchError(
                    f"R1_hat_provenance must be null for horizon={self.horizon}, "
                    f"got {self.R1_hat_provenance}"
                )

        # rl_test barrier
        if not self.forbid_rl_test:
            raise IdentityMismatchError("forbid_rl_test must be True")


def validate_observation(observation: np.ndarray) -> None:
    """
    Validate observation against M5 point schema contract.

    Args:
        observation: np.ndarray of shape (10,), dtype floating, values in [0, 1].

    Raises:
        ContractViolationError: If observation does not conform.
    """
    if not isinstance(observation, np.ndarray):
        raise ContractViolationError(
            f"Observation must be np.ndarray, got {type(observation).__name__}"
        )
    if observation.shape != (10,):
        raise ContractViolationError(
            f"Observation shape must be (10,), got {observation.shape}"
        )
    if not np.issubdtype(observation.dtype, np.floating):
        raise ContractViolationError(
            f"Observation dtype must be floating, got {observation.dtype}"
        )
    if not np.all(np.isfinite(observation)):
        raise ContractViolationError(
            "Observation contains non-finite values"
        )
    if np.any(observation < 0) or np.any(observation > 1):
        raise ContractViolationError(
            f"Observation values must be in [0, 1], "
            f"got range [{observation.min():.4f}, {observation.max():.4f}]"
        )


def validate_planner_context(ctx: PlannerContext) -> None:
    """
    Validate a PlannerContext by triggering its __post_init__.

    Args:
        ctx: PlannerContext to validate.

    Raises:
        IdentityMismatchError: If any frozen identity check fails.
    """
    # Trigger validation via dataclass __post_init__
    object.__setattr__(ctx, "_validation_triggered", True)
    # The __post_init__ already ran on construction; this is a no-op
    # but signals explicit validation intent.
    # (Frozen dataclass prevents mutation; validation is in __post_init__)


def get_cost_regime(regime_id: str) -> Dict[str, float]:
    """Get frozen cost regime by ID."""
    if regime_id not in COST_REGIMES:
        raise ContractViolationError(
            f"Unknown cost regime '{regime_id}'. Available: {list(COST_REGIMES.keys())}"
        )
    return COST_REGIMES[regime_id].copy()