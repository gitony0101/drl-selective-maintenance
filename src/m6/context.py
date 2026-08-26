"""
M6 Planner Context and Helper Functions.

Provides the frozen PlannerContext dataclass and factory functions for building
validated planner contexts for H1, H2, and DDQN-greedy methods.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

import numpy as np

from .contract import (
    PlannerContext,
    COST_REGIMES,
    ACTION_TABLE_K1_SHA256,
    ACTION_TABLE_K2_SHA256,
    M5_PREDICTION_CACHE_MANIFEST_SHA256,
    ENVIRONMENT_CONTRACT_ID,
    M5_OBSERVATION_SCHEMA_ID,
    M4_RISK_MODEL_ID,
    M4_RISK_TEMPERATURE,
    M4_DELTA_CYCLES,
    M4_TIE_TOLERANCE,
    M5_GAMMA,
    IdentityMismatchError,
    ContractViolationError,
)

# Import action tables from frozen M4/M5 source
from src.envs.action_table import ACTION_TABLE_N5_K1, ACTION_TABLE_N5_K2


def compute_action_table_hash(action_table: Tuple[Tuple[int, ...], ...]) -> str:
    """
    Compute SHA256 hash of action table for identity verification.

    Uses standard JSON serialization (json.dumps default) to match frozen hashes.

    Args:
        action_table: Tuple of action subsets.

    Returns:
        SHA256 hex digest.
    """
    import hashlib
    import json

    # Serialize as JSON array of arrays (standard json.dumps, not compact)
    data = json.dumps([list(x) for x in action_table]).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


# Verify frozen hashes at module load
_K1_HASH = compute_action_table_hash(ACTION_TABLE_N5_K1)
_K2_HASH = compute_action_table_hash(ACTION_TABLE_N5_K2)

if _K1_HASH != ACTION_TABLE_K1_SHA256:
    raise RuntimeError(
        f"K=1 action table hash mismatch: computed {_K1_HASH}, "
        f"expected {ACTION_TABLE_K1_SHA256}"
    )

if _K2_HASH != ACTION_TABLE_K2_SHA256:
    raise RuntimeError(
        f"K=2 action table hash mismatch: computed {_K2_HASH}, "
        f"expected {ACTION_TABLE_K2_SHA256}"
    )


def build_planner_context_h1(
    maintenance_capacity: int,
    cost_regime_id: str,
) -> PlannerContext:
    """
    Build PlannerContext for H=1 planning.

    Args:
        maintenance_capacity: K (1 or 2)
        cost_regime_id: One of 4 frozen cost regimes

    Returns:
        Validated PlannerContext with horizon=1, R1_hat fields null.

    Raises:
        IdentityMismatchError: If any frozen identity is violated.
    """
    if maintenance_capacity not in (1, 2):
        raise IdentityMismatchError(
            f"maintenance_capacity must be 1 or 2, got {maintenance_capacity}"
        )

    if cost_regime_id not in COST_REGIMES:
        raise IdentityMismatchError(
            f"cost_regime_id must be one of {list(COST_REGIMES.keys())}, "
            f"got '{cost_regime_id}'"
        )

    action_table = ACTION_TABLE_N5_K1 if maintenance_capacity == 1 else ACTION_TABLE_N5_K2
    action_table_hash = ACTION_TABLE_K1_SHA256 if maintenance_capacity == 1 else ACTION_TABLE_K2_SHA256
    regime = COST_REGIMES[cost_regime_id]

    return PlannerContext(
        maintenance_capacity=maintenance_capacity,
        delta_cycles=M4_DELTA_CYCLES,
        rul_scale=125.0,
        age_scale_cycles=341,
        action_table=action_table,
        action_table_sha256=action_table_hash,
        cost_regime_id=cost_regime_id,
        c_pm=regime["c_pm"],
        c_f=regime["c_f"],
        c_u=regime["c_u"],
        risk_model_id=M4_RISK_MODEL_ID,
        risk_temperature=M4_RISK_TEMPERATURE,
        gamma=M5_GAMMA,
        observation_schema_id=M5_OBSERVATION_SCHEMA_ID,
        environment_contract_id=ENVIRONMENT_CONTRACT_ID,
        prediction_cache_manifest_sha256=M5_PREDICTION_CACHE_MANIFEST_SHA256,
        horizon=1,
        forbid_rl_test=True,
        R1_hat_cycles=None,
        R1_hat_provenance=None,
    )


def build_planner_context_h2(
    maintenance_capacity: int,
    cost_regime_id: str,
    R1_hat_cycles: float,
    R1_hat_provenance: Dict[str, str],
) -> PlannerContext:
    """
    Build PlannerContext for H=2 planning.

    Args:
        maintenance_capacity: K (1 or 2)
        cost_regime_id: One of 4 frozen cost regimes
        R1_hat_cycles: Precomputed mean predicted RUL for cycle==1 in predictor_train
        R1_hat_provenance: Dict with keys predictor_train_manifest_sha256, computed_at_utc, n_cycle1_records

    Returns:
        Validated PlannerContext with horizon=2, R1_hat fields populated.

    Raises:
        IdentityMismatchError: If any frozen identity is violated.
    """
    if maintenance_capacity not in (1, 2):
        raise IdentityMismatchError(
            f"maintenance_capacity must be 1 or 2, got {maintenance_capacity}"
        )

    if cost_regime_id not in COST_REGIMES:
        raise IdentityMismatchError(
            f"cost_regime_id must be one of {list(COST_REGIMES.keys())}, "
            f"got '{cost_regime_id}'"
        )

    required_provenance_keys = {
        "predictor_train_manifest_sha256",
        "computed_at_utc",
        "n_cycle1_records",
    }
    if not required_provenance_keys.issubset(R1_hat_provenance.keys()):
        raise IdentityMismatchError(
            f"R1_hat_provenance missing required keys: "
            f"{required_provenance_keys - set(R1_hat_provenance.keys())}"
        )

    action_table = ACTION_TABLE_N5_K1 if maintenance_capacity == 1 else ACTION_TABLE_N5_K2
    action_table_hash = ACTION_TABLE_K1_SHA256 if maintenance_capacity == 1 else ACTION_TABLE_K2_SHA256
    regime = COST_REGIMES[cost_regime_id]

    return PlannerContext(
        maintenance_capacity=maintenance_capacity,
        delta_cycles=M4_DELTA_CYCLES,
        rul_scale=125.0,
        age_scale_cycles=341,
        action_table=action_table,
        action_table_sha256=action_table_hash,
        cost_regime_id=cost_regime_id,
        c_pm=regime["c_pm"],
        c_f=regime["c_f"],
        c_u=regime["c_u"],
        risk_model_id=M4_RISK_MODEL_ID,
        risk_temperature=M4_RISK_TEMPERATURE,
        gamma=M5_GAMMA,
        observation_schema_id=M5_OBSERVATION_SCHEMA_ID,
        environment_contract_id=ENVIRONMENT_CONTRACT_ID,
        prediction_cache_manifest_sha256=M5_PREDICTION_CACHE_MANIFEST_SHA256,
        horizon=2,
        forbid_rl_test=True,
        R1_hat_cycles=R1_hat_cycles,
        R1_hat_provenance=R1_hat_provenance,
    )


def build_planner_context_ddqn(
    maintenance_capacity: int,
    cost_regime_id: str,
) -> PlannerContext:
    """
    Build PlannerContext for DDQN-greedy inference.

    Args:
        maintenance_capacity: K (1 or 2)
        cost_regime_id: One of 4 frozen cost regimes

    Returns:
        Validated PlannerContext with horizon=0, R1_hat fields null.
    """
    if maintenance_capacity not in (1, 2):
        raise IdentityMismatchError(
            f"maintenance_capacity must be 1 or 2, got {maintenance_capacity}"
        )

    if cost_regime_id not in COST_REGIMES:
        raise IdentityMismatchError(
            f"cost_regime_id must be one of {list(COST_REGIMES.keys())}, "
            f"got '{cost_regime_id}'"
        )

    action_table = ACTION_TABLE_N5_K1 if maintenance_capacity == 1 else ACTION_TABLE_N5_K2
    action_table_hash = ACTION_TABLE_K1_SHA256 if maintenance_capacity == 1 else ACTION_TABLE_K2_SHA256
    regime = COST_REGIMES[cost_regime_id]

    return PlannerContext(
        maintenance_capacity=maintenance_capacity,
        delta_cycles=M4_DELTA_CYCLES,
        rul_scale=125.0,
        age_scale_cycles=341,
        action_table=action_table,
        action_table_sha256=action_table_hash,
        cost_regime_id=cost_regime_id,
        c_pm=regime["c_pm"],
        c_f=regime["c_f"],
        c_u=regime["c_u"],
        risk_model_id=M4_RISK_MODEL_ID,
        risk_temperature=M4_RISK_TEMPERATURE,
        gamma=M5_GAMMA,
        observation_schema_id=M5_OBSERVATION_SCHEMA_ID,
        environment_contract_id=ENVIRONMENT_CONTRACT_ID,
        prediction_cache_manifest_sha256=M5_PREDICTION_CACHE_MANIFEST_SHA256,
        horizon=0,
        forbid_rl_test=True,
        R1_hat_cycles=None,
        R1_hat_provenance=None,
    )


def load_R1_hat(manifest_sha256: str) -> Dict[str, Any]:
    """
    Load R1_hat statistic from predictor_train cache.

    Args:
        manifest_sha256: SHA256 of prediction_cache_manifest_v2.json (must match frozen value).

    Returns:
        Dict with keys: R1_hat_cycles, predictor_train_manifest_sha256, computed_at_utc, n_cycle1_records.

    Raises:
        ContractViolationError: If manifest SHA mismatches or zero cycle==1 records.
    """
    if manifest_sha256 != M5_PREDICTION_CACHE_MANIFEST_SHA256:
        raise ContractViolationError(
            f"Prediction cache manifest SHA256 mismatch: "
            f"expected {M5_PREDICTION_CACHE_MANIFEST_SHA256}, got {manifest_sha256}"
        )

    from pathlib import Path
    from src.predictors.prediction_store import load_default_prediction_store

    cache_dir = Path("data/processed/fd001/v2/06_PREDICTIONS")
    if not (cache_dir / "fd001_prediction_cache_v2.parquet").exists():
        raise ContractViolationError(
            f"Prediction cache not found at {cache_dir}. "
            "Run M6 setup step to provision cache."
        )

    store = load_default_prediction_store(cache_dir)

    # Get all predictor_train records using get_all_for_unit for each unit
    # First get all unit IDs in predictor_train split
    units = store.get_units("predictor_train")
    if not units:
        raise ContractViolationError("No predictor_train records found in cache")

    cycle1_records = []
    for unit_id in units:
        unit_df = store.get_all_for_unit("predictor_train", unit_id)
        if not unit_df.empty:
            cycle1_unit = unit_df[unit_df["cycle"] == 1]
            for _, row in cycle1_unit.iterrows():
                cycle1_records.append(row)

    n_cycle1 = len(cycle1_records)

    if n_cycle1 == 0:
        raise ContractViolationError(
            "Zero cycle==1 records in predictor_train; M6 setup fails closed"
        )

    # Compute mean predicted RUL for cycle 1
    R1_hat_cycles = float(np.mean([r["predicted_rul"] for r in cycle1_records]))

    return {
        "R1_hat_cycles": R1_hat_cycles,
        "predictor_train_manifest_sha256": manifest_sha256,
        "computed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_cycle1_records": n_cycle1,
    }


def validate_observation(observation: np.ndarray) -> None:
    """
    Validate observation against M5 point observation schema.

    Args:
        observation: Observation array.

    Raises:
        ContractViolationError: If observation is invalid.
    """
    if observation.shape != (10,):
        raise ContractViolationError(
            f"Observation shape must be (10,), got {observation.shape}"
        )

    if not np.issubdtype(observation.dtype, np.floating):
        raise ContractViolationError(
            f"Observation dtype must be floating point, got {observation.dtype}"
        )

    if not np.all(np.isfinite(observation)):
        raise ContractViolationError("Observation contains non-finite values")

    if np.any(observation < 0) or np.any(observation > 1):
        raise ContractViolationError(
            f"Observation values must be in [0, 1], "
            f"got range [{observation.min():.4f}, {observation.max():.4f}]"
        )


def serialize_planner_context(ctx: PlannerContext) -> Dict[str, Any]:
    """
    Serialize PlannerContext to JSON-serializable dict for artifact output.

    Args:
        ctx: PlannerContext instance.

    Returns:
        Dictionary matching m6_planner_context_v2 schema.
    """
    # Convert action_table to list of lists for JSON
    action_table_serializable = [list(slots) for slots in ctx.action_table]

    data = {
        "schema_version": "m6_planner_context_v2",
        "method": {
            1: "h1",
            2: "h2",
            0: "ddqn_greedy",
        }[ctx.horizon],
        "maintenance_capacity": ctx.maintenance_capacity,
        "delta_cycles": ctx.delta_cycles,
        "rul_scale": ctx.rul_scale,
        "age_scale_cycles": ctx.age_scale_cycles,
        "action_table_sha256": ctx.action_table_sha256,
        "cost_regime_id": ctx.cost_regime_id,
        "c_pm": ctx.c_pm,
        "c_f": ctx.c_f,
        "c_u": ctx.c_u,
        "risk_model_id": ctx.risk_model_id,
        "risk_temperature": ctx.risk_temperature,
        "gamma": ctx.gamma,
        "observation_schema_id": ctx.observation_schema_id,
        "environment_contract_id": ctx.environment_contract_id,
        "prediction_cache_manifest_sha256": ctx.prediction_cache_manifest_sha256,
        "horizon": ctx.horizon,
        "forbid_rl_test": ctx.forbid_rl_test,
    }

    if ctx.horizon == 2:
        data["R1_hat_cycles"] = ctx.R1_hat_cycles
        data["R1_hat_provenance"] = ctx.R1_hat_provenance
    else:
        data["R1_hat_cycles"] = None
        data["R1_hat_provenance"] = None

    return data


def deserialize_planner_context(data: Dict[str, Any]) -> PlannerContext:
    """
    Deserialize PlannerContext from JSON dict.

    Args:
        data: Dictionary with PlannerContext fields.

    Returns:
        PlannerContext instance.
    """
    # Rebuild action_table from maintenance_capacity
    action_table = (
        ACTION_TABLE_N5_K1 if data["maintenance_capacity"] == 1
        else ACTION_TABLE_N5_K2
    )

    return PlannerContext(
        maintenance_capacity=data["maintenance_capacity"],
        delta_cycles=data["delta_cycles"],
        rul_scale=data["rul_scale"],
        age_scale_cycles=data["age_scale_cycles"],
        action_table=action_table,
        action_table_sha256=data["action_table_sha256"],
        cost_regime_id=data["cost_regime_id"],
        c_pm=data["c_pm"],
        c_f=data["c_f"],
        c_u=data["c_u"],
        risk_model_id=data["risk_model_id"],
        risk_temperature=data["risk_temperature"],
        gamma=data["gamma"],
        observation_schema_id=data["observation_schema_id"],
        environment_contract_id=data["environment_contract_id"],
        prediction_cache_manifest_sha256=data["prediction_cache_manifest_sha256"],
        horizon=data["horizon"],
        forbid_rl_test=data["forbid_rl_test"],
        R1_hat_cycles=data.get("R1_hat_cycles"),
        R1_hat_provenance=data.get("R1_hat_provenance"),
    )