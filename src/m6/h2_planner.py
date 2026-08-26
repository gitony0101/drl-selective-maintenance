"""
M6 H=2 Planner — Probability-weighted two-step receding horizon planner.

Implements J2(o, a) = immediate_estimated_cost(o, a) + γ · Σ_F P(F|o,a) · min_{a'} future_estimated_cost(o'(o,a,F), a')

Planner identity: m6_h2_v1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.m6.contract import (
    PlannerContext,
    validate_observation,
    M4_RISK_MODEL_ID,
    M4_RISK_TEMPERATURE,
    M4_DELTA_CYCLES,
    M4_TIE_TOLERANCE,
    M5_GAMMA,
)
from src.m6.h2_forward_model import ForwardModel, Branch, PublicNextState
from src.envs.action_table import ACTION_TABLE_N5_K1, ACTION_TABLE_N5_K2


# H2 Planner identity
H2_PLANNER_ID = "m6_h2_v1"
H2_GAMMA = 0.95


@dataclass(frozen=True)
class H2PerActionDiagnostics:
    """Per-action diagnostics for H2 decision trace."""
    action_id: int
    selected_slots: Tuple[int, ...]
    immediate_cost: float
    immediate_preventive: float
    immediate_unused_life: float
    immediate_failure: float
    J2: float
    branches: List[Branch]


@dataclass(frozen=True)
class H2PlanResult:
    """
    Result of H2 planning with full provenance.

    Includes chosen action, all per-action J2 values, and branch diagnostics.
    """
    planner_id: str
    action_id: int
    selected_slots: Tuple[int, ...]
    immediate_cost: float
    J2: float
    gamma: float
    K: int
    action_table_hash: str
    selected_candidate: str
    risk_model_id: str
    risk_temperature: float
    delta_cycles: int
    tie_tolerance: float
    observation_schema_id: str
    R1_hat_cycles: float
    per_action: List[H2PerActionDiagnostics]
    provenance: Dict[str, Any]


class H2Planner:
    """
    H=2 Receding-Horizon Planner.

    Computes J2(o, a) for each legal action and selects argmin with M4 tie-breaking.
    """

    def __init__(self, ctx: PlannerContext) -> None:
        """
        Initialize H2 planner with validated context.

        Args:
            ctx: PlannerContext with horizon=2, R1_hat_cycles populated.
        """
        if ctx.horizon != 2:
            raise ValueError(f"H2Planner requires horizon=2, got {ctx.horizon}")
        if ctx.R1_hat_cycles is None:
            raise ValueError("H2Planner requires R1_hat_cycles")
        if ctx.R1_hat_provenance is None:
            raise ValueError("H2Planner requires R1_hat_provenance")

        self.ctx = ctx
        self.K = ctx.maintenance_capacity
        self.gamma = ctx.gamma  # 0.95 frozen

        # Verify frozen identities
        if ctx.risk_model_id != M4_RISK_MODEL_ID:
            raise ValueError(f"H2 requires risk_model_id={M4_RISK_MODEL_ID}")
        if ctx.risk_temperature != M4_RISK_TEMPERATURE:
            raise ValueError(f"H2 requires risk_temperature={M4_RISK_TEMPERATURE}")
        if ctx.delta_cycles != M4_DELTA_CYCLES:
            raise ValueError(f"H2 requires delta_cycles={M4_DELTA_CYCLES}")

        # Build forward model
        self.forward_model = ForwardModel(ctx)

        # Action table info
        self.action_table = ctx.action_table
        self.action_table_hash = ctx.action_table_sha256
        self.num_actions = len(ctx.action_table)

    def plan(self, observation: np.ndarray) -> H2PlanResult:
        """
        Execute H=2 planning for a single observation.

        Args:
            observation: Public observation array, shape (10,), dtype float32, range [0,1].

        Returns:
            H2PlanResult with chosen action and full diagnostics.
        """
        validate_observation(observation)

        # Compute J2 for each action
        per_action = []
        min_J2 = float("inf")
        best_action_id = -1
        best_selected_slots = ()
        best_immediate_cost = 0.0

        for action_id in range(self.num_actions):
            J2, immediate_cost, branches = self.forward_model.compute_J2(observation, action_id)

            # Get selected slots and immediate cost breakdown
            selected_slots = self.action_table[action_id]
            age_cycles, pred_rul_cycles = self.forward_model._decode_observation(observation)
            prev_cost, unused_cost, fail_cost, _ = self.forward_model._compute_immediate_cost(
                action_id, pred_rul_cycles
            )

            diagnostics = H2PerActionDiagnostics(
                action_id=action_id,
                selected_slots=selected_slots,
                immediate_cost=immediate_cost,
                immediate_preventive=prev_cost,
                immediate_unused_life=unused_cost,
                immediate_failure=fail_cost,
                J2=J2,
                branches=branches,
            )
            per_action.append(diagnostics)

            # Tie-breaking: smallest action_id wins (same as M4)
            if J2 < min_J2 - M4_TIE_TOLERANCE:
                min_J2 = J2
                best_action_id = action_id
                best_selected_slots = selected_slots
                best_immediate_cost = immediate_cost
            # If within tie_tolerance, keep current best (smaller action_id wins)

        # Verify branch probability sum for chosen action
        chosen_branches = per_action[best_action_id].branches
        branch_prob_sum = sum(b.probability for b in chosen_branches)

        # Build provenance
        provenance = {
            "planner_class": "H2Planner",
            "forward_model_class": "ForwardModel",
            "gamma": self.gamma,
            "branch_prob_sum": branch_prob_sum,
            "branch_count": len(chosen_branches),
            "max_branch_count_K1": 32,
            "max_branch_count_K2": 32,
            "tie_tolerance": M4_TIE_TOLERANCE,
            "tie_rule": "smallest action_id wins",
        }

        return H2PlanResult(
            planner_id=H2_PLANNER_ID,
            action_id=best_action_id,
            selected_slots=best_selected_slots,
            immediate_cost=best_immediate_cost,
            J2=min_J2,
            gamma=self.gamma,
            K=self.K,
            action_table_hash=self.action_table_hash,
            selected_candidate=M4_RISK_MODEL_ID,  # "logistic_window_v1" is the M4 candidate
            risk_model_id=M4_RISK_MODEL_ID,
            risk_temperature=M4_RISK_TEMPERATURE,
            delta_cycles=M4_DELTA_CYCLES,
            tie_tolerance=M4_TIE_TOLERANCE,
            observation_schema_id=self.ctx.observation_schema_id,
            R1_hat_cycles=self.ctx.R1_hat_cycles,
            per_action=per_action,
            provenance=provenance,
        )


def build_h2_planner(
    maintenance_capacity: int,
    cost_regime_id: str,
    R1_hat_cycles: float,
    R1_hat_provenance: Dict[str, str],
) -> H2Planner:
    """
    Factory function to build H2Planner from basic parameters.

    Args:
        maintenance_capacity: K (1 or 2)
        cost_regime_id: One of 4 frozen cost regimes
        R1_hat_cycles: Precomputed mean predicted RUL for cycle==1 in predictor_train
        R1_hat_provenance: Dict with predictor_train_manifest_sha256, computed_at_utc, n_cycle1_records

    Returns:
        Configured H2Planner instance.
    """
    from src.m6.context import build_planner_context_h2

    ctx = build_planner_context_h2(
        maintenance_capacity=maintenance_capacity,
        cost_regime_id=cost_regime_id,
        R1_hat_cycles=R1_hat_cycles,
        R1_hat_provenance=R1_hat_provenance,
    )

    return H2Planner(ctx)


def h2_result_to_decision_trace(
    result: H2PlanResult,
    run_id: str,
    step_index: int,
    scenario_id: str,
    reset_seed: int,
    observation: np.ndarray,
    branch_prob_sum: float,
    min_a_prime: int,
    next_state: PublicNextState,
) -> Dict[str, Any]:
    """
    Convert H2PlanResult to M6 decision_trace schema record (method="h2").

    Args:
        result: H2PlanResult from plan()
        run_id: Run identifier
        step_index: Episode step index
        scenario_id: Scenario identifier
        reset_seed: Environment reset seed
        observation: Observation used (will be serialized as list)
        branch_prob_sum: Sum of branch probabilities for chosen action (should be ~1.0)
        min_a_prime: Best future action for chosen action (branch-aggregated)
        next_state: Next public state for chosen action's best branch

    Returns:
        Dictionary matching m6_decision_trace_v2 schema for method="h2".
    """
    from datetime import datetime, timezone

    # Get chosen action's immediate partition
    chosen_diag = result.per_action[result.action_id]
    immediate_partition = chosen_diag.immediate_cost

    return {
        "schema_version": "m6_decision_trace_v2",
        "run_id": run_id,
        "method": "h2",
        "step_index": step_index,
        "scenario_id": scenario_id,
        "reset_seed": reset_seed,
        "observation": observation.tolist(),
        "chosen_action_id": result.action_id,
        "chosen_selected_slots": list(result.selected_slots),
        "checkpoint_run_id": None,
        "checkpoint_sha256": None,
        "planner_context_identity": f"{H2_PLANNER_ID}:{result.action_table_hash}",
        "estimated_preventive_cost": chosen_diag.immediate_preventive,
        "estimated_unused_life_cost": chosen_diag.immediate_unused_life,
        "estimated_failure_cost": chosen_diag.immediate_failure,
        "estimated_total_cost": chosen_diag.immediate_cost,
        "posthoc_m4_estimated_preventive_cost": None,
        "posthoc_m4_estimated_unused_life_cost": None,
        "posthoc_m4_estimated_failure_cost": None,
        "posthoc_m4_estimated_total_cost": None,
        "h2_chosen_J2": result.J2,
        "h2_min_a_prime": min_a_prime,
        "h2_immediate_partition": immediate_partition,
        "h2_gamma": result.gamma,
        "h2_branch_probs_sum": branch_prob_sum,
        "h2_next_public_state_age_norm": next_state.age_norm.tolist(),
        "h2_next_public_state_pred_rul_norm": next_state.pred_rul_norm.tolist(),
        "r1_hat_cycles": result.R1_hat_cycles,
        "predictor_train_manifest_sha256": result.ctx.R1_hat_provenance["predictor_train_manifest_sha256"],
    }