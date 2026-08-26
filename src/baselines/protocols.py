"""
Policy protocols for Milestone 3 Rule Baselines.

Defines the immutable context types that policies may receive:
- PolicyContext: Public information available to all practical policies
- OracleContext: Extended diagnostic information for true-RUL oracle only

Practical policies must NEVER receive OracleContext.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..envs.action_table import ActionSubset


@dataclass(frozen=True)
class PolicyContext:
    """
    Immutable context for practical policies.

    Contains only public information that practical policies may access:
    - maintenance_capacity (K)
    - age_scale_cycles (341)
    - rul_scale (125.0)
    - action_table (tuple of action subsets)
    - cost_regime_id (for reporting only)
    - policy_rng (policy-owned RNG, separate from environment)

    Does NOT contain:
    - true_rul, true_rul_capped
    - trajectory_id, unit_id, trajectory_length
    - split identity, scenario_id
    - diagnostic info
    """

    maintenance_capacity: int
    age_scale_cycles: int
    rul_scale: float
    action_table: Tuple[Tuple[int, ...], ...]
    cost_regime_id: str
    policy_rng: np.random.Generator

    def __post_init__(self) -> None:
        """Validate that policy context contains only allowed information."""
        # Immutable by design - no validation needed beyond type checking
        pass


@dataclass(frozen=True)
class OracleContext:
    """
    Immutable context for diagnostic oracle policy.

    Extends PolicyContext with true-RUL access through a separate diagnostic
    interface. Requires explicit allow_oracle=True and diagnostic_mode=True
    to construct.

    Contains all PolicyContext fields plus:
    - allow_oracle: Must be True
    - diagnostic_mode: Must be True

    Practical policies must reject OracleContext.
    """

    maintenance_capacity: int
    age_scale_cycles: int
    rul_scale: float
    action_table: Tuple[Tuple[int, ...], ...]
    cost_regime_id: str
    policy_rng: np.random.Generator
    allow_oracle: bool
    diagnostic_mode: bool

    def __post_init__(self) -> None:
        """Validate oracle context requirements."""
        if not self.allow_oracle:
            raise ValueError("OracleContext requires allow_oracle=True")
        if not self.diagnostic_mode:
            raise ValueError("OracleContext requires diagnostic_mode=True")

    @classmethod
    def from_policy_context(
        cls,
        policy_ctx: PolicyContext,
        allow_oracle: bool = True,
        diagnostic_mode: bool = True,
    ) -> "OracleContext":
        """
        Construct OracleContext from PolicyContext.

        Args:
            policy_ctx: Base policy context
            allow_oracle: Must be True to enable oracle access
            diagnostic_mode: Must be True to enable diagnostic info

        Returns:
            OracleContext instance

        Raises:
            ValueError: If allow_oracle or diagnostic_mode is False
        """
        if not allow_oracle:
            raise ValueError("OracleContext requires allow_oracle=True")
        if not diagnostic_mode:
            raise ValueError("OracleContext requires diagnostic_mode=True")

        return cls(
            maintenance_capacity=policy_ctx.maintenance_capacity,
            age_scale_cycles=policy_ctx.age_scale_cycles,
            rul_scale=policy_ctx.rul_scale,
            action_table=policy_ctx.action_table,
            cost_regime_id=policy_ctx.cost_regime_id,
            policy_rng=policy_ctx.policy_rng,
            allow_oracle=True,
            diagnostic_mode=True,
        )


# Type aliases for policy functions
Observation = np.ndarray  # Shape (10,), dtype float32
ActionId = int  # Integer in [0, num_actions)


def validate_practical_policy_context(context: PolicyContext | OracleContext) -> None:
    """
    Validate that a practical policy is not receiving OracleContext.

    Practical policies must NEVER receive OracleContext, which contains
    allow_oracle and diagnostic_mode flags that indicate true-RUL access.

    Args:
        context: Context passed to policy

    Raises:
        ValueError: If context is OracleContext instead of PolicyContext
    """
    if isinstance(context, OracleContext):
        raise ValueError(
            "Practical policies must not receive OracleContext. "
            "OracleContext contains allow_oracle=True and diagnostic_mode=True, "
            "which indicates access to true RUL through diagnostic info. "
            "Practical policies may only use PolicyContext with public information."
        )