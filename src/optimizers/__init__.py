"""
Milestone 4 Exact Myopic Optimizers.

Implements deterministic current-window optimization for selective maintenance:
- Exact enumeration of all feasible actions
- Cost estimation using only implementable information
- Deterministic tie-breaking (smallest action_id)
- Information barrier (no true RUL access)
"""

from .failure_risk import (
    RiskModelId,
    compute_failure_risk,
    compute_hard_window_risk,
    compute_logistic_window_risk,
    get_risk_model_description,
    validate_risk_model_parameters,
)

from .exact_myopic import (
    MyopicContext,
    ExactMyopicOptimizer,
    ActionCostBreakdown,
)

from .myopic_artifacts import (
    write_atomic_json,
    validate_json_serializable,
    convert_for_json,
    compute_file_hash,
    compute_data_hash,
    MyopicArtifactWriter,
    get_git_commit,
    build_complete_scientific_config,
    compute_complete_config_hash,
    build_runtime_metadata,
)


__all__ = [
    # Risk models
    "RiskModelId",
    "compute_failure_risk",
    "compute_hard_window_risk",
    "compute_logistic_window_risk",
    "get_risk_model_description",
    "validate_risk_model_parameters",
    # Optimizer
    "MyopicContext",
    "ExactMyopicOptimizer",
    "ActionCostBreakdown",
    # Artifacts
    "write_atomic_json",
    "validate_json_serializable",
    "convert_for_json",
    "compute_file_hash",
    "compute_data_hash",
    "MyopicArtifactWriter",
    "get_git_commit",
    # Config functions
    "build_complete_scientific_config",
    "compute_complete_config_hash",
    "build_runtime_metadata",
]