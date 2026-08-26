'''
M6 Receding-Horizon Planning Package (H=2 comparator).

Provides the H2Planner, a two-step receding-horizon planner that evaluates
the expected cost of each feasible action using a forward model of the
fleet maintenance environment.

The planner takes as input a PlannerContext containing the current
observation, the action table, cost regimes, and a prediction cache. It
enumerates possible failure branches probabilistically and selects the
action that minimizes the expected cost over a two-step horizon.

This package includes:
- contract.py: PlannerContext definition and validation
- context.py: Context construction utilities
- h2_forward_model.py: State transition and failure model
- h2_planner.py: Core planning logic and result types

The H2 planner is a structurally privileged comparator: it encodes the
analytic transition and failure structure of the environment, unlike the
DDQN agent which learns from experience.
'''

from .contract import (
    PlannerContext,
    ContractViolationError,
    IdentityMismatchError,
    SplitViolationError,
    COST_REGIMES,
    H1_PLANNER_ID,
    H2_PLANNER_ID,
    M4_RISK_MODEL_ID,
    M4_RISK_TEMPERATURE,
    M4_DELTA_CYCLES,
    M4_TIE_TOLERANCE,
    M5_GAMMA,
    M5_OBSERVATION_SCHEMA_ID,
    M5_PREDICTION_CACHE_MANIFEST_SHA256,
    ACTION_TABLE_K1_SHA256,
    ACTION_TABLE_K2_SHA256,
    ENVIRONMENT_CONTRACT_ID,
    validate_observation,
    validate_planner_context,
    get_cost_regime,
)

from .context import (
    build_planner_context_h1,
    build_planner_context_h2,
    build_planner_context_ddqn,
    load_R1_hat,
    serialize_planner_context,
    deserialize_planner_context,
)

from .h2_planner import (
    H2Planner,
    H2PlanResult,
    H2PerActionDiagnostics,
    ForwardModel,
    PublicNextState,
    Branch,
    build_h2_planner,
    h2_result_to_decision_trace,
    H2_PLANNER_ID,
    H2_GAMMA,
)

__all__ = [
    # Contract
    "PlannerContext",
    "ContractViolationError",
    "IdentityMismatchError",
    "SplitViolationError",
    "COST_REGIMES",
    "H1_PLANNER_ID",
    "H2_PLANNER_ID",
    "M4_RISK_MODEL_ID",
    "M4_RISK_TEMPERATURE",
    "M4_DELTA_CYCLES",
    "M4_TIE_TOLERANCE",
    "M5_GAMMA",
    "M5_OBSERVATION_SCHEMA_ID",
    "M5_PREDICTION_CACHE_MANIFEST_SHA256",
    "ACTION_TABLE_K1_SHA256",
    "ACTION_TABLE_K2_SHA256",
    "ENVIRONMENT_CONTRACT_ID",
    "validate_observation",
    "validate_planner_context",
    "get_cost_regime",
    # Context
    "build_planner_context_h1",
    "build_planner_context_h2",
    "build_planner_context_ddqn",
    "load_R1_hat",
    "serialize_planner_context",
    "deserialize_planner_context",
    # H2 Planner
    "H2Planner",
    "H2PlanResult",
    "H2PerActionDiagnostics",
    "ForwardModel",
    "PublicNextState",
    "Branch",
    "build_h2_planner",
    "h2_result_to_decision_trace",
    "H2_PLANNER_ID",
    "H2_GAMMA",
]