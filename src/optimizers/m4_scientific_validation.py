"""
M4 Scientific Validation Optimizer Wrapper.

This module provides a clean interface for running scientific validation
candidates without modifying the Exact Myopic core.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .exact_myopic import MyopicContext, ExactMyopicOptimizer
from .myopic_artifacts import (
    build_complete_scientific_config,
    compute_complete_config_hash,
    get_git_commit,
    write_atomic_json,
    convert_for_json,
)
from .myopic_provenance import compute_action_table_content_hash
from envs.action_table import ACTION_TABLE_N5_K1, ACTION_TABLE_N5_K2
from envs.costs import get_cost_regime


@dataclass(frozen=True)
class ScientificValidationConfig:
    """Immutable configuration for scientific validation candidate."""
    candidate_id: str
    risk_model_id: str
    risk_temperature: Optional[float]
    matrix_role: str
    delta_cycles: int = 5
    tie_tolerance: float = 1e-9
    fleet_size: int = 5
    episode_horizon: int = 100
    rul_scale: float = 125.0
    age_scale_cycles: int = 341


def create_scientific_optimizer(
    config: ScientificValidationConfig,
    k_capacity: int,
    cost_regime_id: str,
) -> ExactMyopicOptimizer:
    """
    Create an Exact Myopic optimizer for scientific validation.

    This does not modify the core optimizer - it's just a factory.
    """
    cost_regime = get_cost_regime(cost_regime_id)
    action_table = ACTION_TABLE_N5_K1 if k_capacity == 1 else ACTION_TABLE_N5_K2

    context = MyopicContext(
        maintenance_capacity=k_capacity,
        delta_cycles=config.delta_cycles,
        rul_scale=config.rul_scale,
        age_scale_cycles=config.age_scale_cycles,
        action_table=action_table,
        c_pm=cost_regime.c_pm,
        c_f=cost_regime.c_f,
        c_u=cost_regime.c_u,
        risk_model_id=config.risk_model_id,
    )

    return ExactMyopicOptimizer(
        context=context,
        risk_temperature=config.risk_temperature,
        tie_tolerance=config.tie_tolerance,
    )


def build_scientific_config(
    candidate: ScientificValidationConfig,
    bank_manifest: Dict[str, Any],
    bank_hashes: Dict[str, str],
    prediction_cache_sha256: str,
    action_table_hashes: Dict[str, str],
    protocol_hash: str,
) -> Dict[str, Any]:
    """Build complete scientific config for a candidate."""

    return {
        "schema_version": "m4_scientific_validation_v1",
        "protocol_version": "m4_scientific_validation_v1",
        "protocol_file_sha256": protocol_hash,
        "candidate_identity": candidate.candidate_id,
        "risk_model": candidate.risk_model_id,
        "risk_temperature": candidate.risk_temperature,
        "tie_tolerance": candidate.tie_tolerance,
        "delta_cycles": candidate.delta_cycles,
        "environment_version": "m2_v1",
        "horizon": candidate.episode_horizon,
        "fleet_size": candidate.fleet_size,
        "rul_scale": candidate.rul_scale,
        "age_scale_cycles": candidate.age_scale_cycles,
        "k_values": [1, 2],
        "cost_regimes": [
            "failure-heavy-no-waste",
            "failure-heavy-waste-aware",
            "failure-light-no-waste",
            "failure-light-waste-aware",
        ],
        "splits": ["predictor_train", "rl_validation"],
        "ordered_seeds": list(range(6601, 6621)),
        "scenario_bank_manifest": bank_manifest,
        "scenario_bank_sha256_values": bank_hashes,
        "prediction_cache_sha256": prediction_cache_sha256,
        "action_table_K1_identity": "ACTION_TABLE_N5_K1_M2_V1",
        "action_table_K1_num_actions": 6,
        "action_table_K2_identity": "ACTION_TABLE_N5_K2_M2_V1",
        "action_table_K2_num_actions": 16,
        **action_table_hashes,
        "pairing_basis": "stable_pair_id_from_unit_cycles",
        "selection_metric_version": "macro_avg_normalized_paired_cost_diff_v1",
        "bootstrap_seed": 652104,
        "bootstrap_resamples": 10000,
        "matrix_role": candidate.matrix_role,
    }


# Frozen candidate configurations for scientific validation
SCIENTIFIC_VALIDATION_CANDIDATES = [
    ScientificValidationConfig(
        candidate_id="hard_window_v1",
        risk_model_id="hard_window_v1",
        risk_temperature=None,
        matrix_role="primary_contract_policy",
    ),
    ScientificValidationConfig(
        candidate_id="logistic_T1",
        risk_model_id="logistic_window_v1",
        risk_temperature=1.0,
        matrix_role="scientific_validation_candidate",
    ),
    ScientificValidationConfig(
        candidate_id="logistic_T2",
        risk_model_id="logistic_window_v1",
        risk_temperature=2.0,
        matrix_role="scientific_validation_candidate",
    ),
    ScientificValidationConfig(
        candidate_id="logistic_T5",
        risk_model_id="logistic_window_v1",
        risk_temperature=5.0,
        matrix_role="scientific_validation_candidate",
    ),
    ScientificValidationConfig(
        candidate_id="logistic_T10",
        risk_model_id="logistic_window_v1",
        risk_temperature=10.0,
        matrix_role="scientific_validation_candidate",
    ),
    ScientificValidationConfig(
        candidate_id="logistic_T20",
        risk_model_id="logistic_window_v1",
        risk_temperature=20.0,
        matrix_role="scientific_validation_candidate",
    ),
]


def get_action_table_hashes() -> Dict[str, str]:
    """Get action table content hashes."""
    return {
        "action_table_K1_content_hash": compute_action_table_content_hash(ACTION_TABLE_N5_K1),
        "action_table_K2_content_hash": compute_action_table_content_hash(ACTION_TABLE_N5_K2),
    }


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute SHA256 of scientific config (excluding runtime metadata)."""
    import hashlib
    import json

    scientific_fields = {k: v for k, v in config.items()
                         if k not in ["output_dir", "overwrite", "timestamp", "git_commit",
                                     "command_line", "log_path", "temporary_path", "matrix_role"]}
    canonical = json.dumps(scientific_fields, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode()).hexdigest()