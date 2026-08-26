"""
Milestone 3 Rule Baselines Package.

Implements six policy families for selective maintenance:
1. CorrectiveOnly - Never perform preventive maintenance
2. RandomFeasible - Uniformly sample legal actions
3. AgeThreshold - Maintain slots where age >= threshold
4. PredictedRULThreshold - Maintain slots where predicted RUL <= threshold
5. GreedyPredictedRUL - Maintain K lowest-RUL slots when activated
6. OracleThreshold - True-RUL oracle (diagnostic only)

All practical policies receive only:
- observation: np.ndarray, shape (10,), dtype float32
- context: PolicyContext with public information only

Oracle policy requires:
- context: OracleContext with allow_oracle=True, diagnostic_mode=True
- diagnostic_info: Dict with true_rul values

Usage:
    from src.baselines import (
        PolicyContext,
        CorrectiveOnly,
        AgeThreshold,
        PolicyEvaluator,
    )

    context = PolicyContext(...)
    policy = AgeThreshold(threshold=100)
    action = policy.select_action(observation, context)
"""

from .protocols import PolicyContext, OracleContext, Observation, ActionId, validate_practical_policy_context
from .rule_policies import (
    CorrectiveOnly,
    RandomFeasible,
    AgeThreshold,
    PredictedRULThreshold,
    GreedyPredictedRUL,
    decode_observation,
    denormalize_age,
    denormalize_rul,
)
from .oracle_policy import OracleThreshold
from .evaluator import PolicyEvaluator, EvaluationConfig, EpisodeResult
from .metrics import (
    compute_summary_statistics,
    summarize_results,
    results_to_parquet,
    validate_json_serializable,
    PolicySummary,
)
from .tuning import (
    tune_threshold,
    tune_all_thresholds,
    select_best_threshold,
    get_threshold_grid,
    THRESHOLD_POLICIES,
    NON_TUNED_POLICIES,
    AGE_THRESHOLDS,
    PREDICTED_RUL_THRESHOLDS,
    GREEDY_ACTIVATION_THRESHOLDS,
    ORACLE_THRESHOLDS,
    ThresholdCandidate,
    SelectedThreshold,
)
from .artifacts import (
    write_resolved_config,
    write_threshold_search_results,
    write_threshold_search_summary,
    write_selected_thresholds,
    write_selected_thresholds_with_meta,
    write_episode_results,
    write_summary_by_policy,
    write_sanity_checks,
    write_run_provenance,
    write_scenario_bank_provenance,
    write_artifact_manifest,
    write_run_log,
    write_independent_recomputation,
    generate_formal_manifest,
    validate_artifacts,
    compute_sha256,
    compute_canonical_config_sha256,
    read_resolved_config_sha256,
    # Formal run context (M3 Step 7)
    FormalRunContext,
    create_formal_run_context,
    seal_formal_run_context,
    load_formal_run_context,
    validate_formal_run_context,
)

__all__ = [
    # Protocols
    "PolicyContext",
    "OracleContext",
    "Observation",
    "ActionId",
    # Rule policies
    "CorrectiveOnly",
    "RandomFeasible",
    "AgeThreshold",
    "PredictedRULThreshold",
    "GreedyPredictedRUL",
    "decode_observation",
    "denormalize_age",
    "denormalize_rul",
    # Oracle policy
    "OracleThreshold",
    # Evaluator
    "PolicyEvaluator",
    "EvaluationConfig",
    "EpisodeResult",
    # Metrics
    "compute_summary_statistics",
    "summarize_results",
    "results_to_parquet",
    "validate_json_serializable",
    "PolicySummary",
    # Tuning
    "tune_threshold",
    "tune_all_thresholds",
    "select_best_threshold",
    "get_threshold_grid",
    "THRESHOLD_POLICIES",
    "NON_TUNED_POLICIES",
    "AGE_THRESHOLDS",
    "PREDICTED_RUL_THRESHOLDS",
    "GREEDY_ACTIVATION_THRESHOLDS",
    "ORACLE_THRESHOLDS",
    "ThresholdCandidate",
    "SelectedThreshold",
    # Artifacts
    "write_resolved_config",
    "write_threshold_search_results",
    "write_threshold_search_summary",
    "write_selected_thresholds",
    "write_selected_thresholds_with_meta",
    "write_episode_results",
    "write_summary_by_policy",
    "write_sanity_checks",
    "write_run_provenance",
    "write_scenario_bank_provenance",
    "write_artifact_manifest",
    "write_run_log",
    "write_independent_recomputation",
    "generate_formal_manifest",
    "validate_artifacts",
    "compute_sha256",
    "compute_canonical_config_sha256",
    "read_resolved_config_sha256",
    # Formal run context (M3 Step 7)
    "FormalRunContext",
    "create_formal_run_context",
    "seal_formal_run_context",
    "load_formal_run_context",
    "validate_formal_run_context",
]