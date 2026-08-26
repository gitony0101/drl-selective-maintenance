"""
M6 H=2 Forward Model — Public probabilistic state transition.

Implements the probability-weighted expected-cost forward model for H=2 planning.
Enumerates all failure masks F ⊆ U(a) and computes P(F|o,a) and next public state.

Never queries future-cycle cache, never reads true_rul, never accesses rl_test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Set

import numpy as np

from src.m6.contract import (
    PlannerContext,
    M4_RISK_MODEL_ID,
    M4_RISK_TEMPERATURE,
    M4_DELTA_CYCLES,
    M4_RUL_SCALE,
    M4_AGE_SCALE_CYCLES,
)
from src.optimizers.failure_risk import compute_failure_risk
from src.optimizers.exact_myopic import ExactMyopicOptimizer, MyopicContext
from src.envs.action_table import ACTION_TABLE_N5_K1, ACTION_TABLE_N5_K2


@dataclass(frozen=True)
class PublicNextState:
    """
    Next public state after one transition step.

    Only contains agent-visible information:
    - age_norm: normalized age since replacement for each slot [0, 1]
    - pred_rul_norm: normalized predicted RUL for each slot [0, 1]
    """
    age_norm: np.ndarray          # shape (5,), float64, [0, 1]
    pred_rul_norm: np.ndarray     # shape (5,), float64, [0, 1]

    def to_observation(self) -> np.ndarray:
        """Convert to flat observation array (10,) for planner consumption."""
        obs = np.zeros(10, dtype=np.float32)
        obs[0::2] = self.age_norm.astype(np.float32)
        obs[1::2] = self.pred_rul_norm.astype(np.float32)
        return obs


@dataclass(frozen=True)
class Branch:
    """
    Single failure-mask branch in the H=2 forward model.

    Represents one subset F of unmaintained slots U(a) that fail.
    """
    failure_mask: Tuple[int, ...]     # slots that fail (subset of U_a)
    probability: float                # P(F | o, a)
    next_state: PublicNextState       # public state after this branch
    best_future_action_id: int        # argmin_{a'} future_estimated_cost
    best_future_cost: float           # min_{a'} future_estimated_cost


class ForwardModel:
    """
    H=2 Forward Model — Probability-weighted expected future cost.

    For action a, defines U(a) = unmaintained slots.
    Enumerates all failure masks F ⊆ U(a).
    For each F: computes P(F|o,a), constructs next public state o'(o,a,F),
    computes min_{a'} estimated_cost(o', a').

    Never queries:
    - future-cycle prediction cache
    - true_rul
    - unit_id, trajectory_id, trajectory_length
    - actual replacement identity
    - rl_test split
    """

    def __init__(self, ctx: PlannerContext) -> None:
        """
        Initialize forward model with frozen planner context.

        Args:
            ctx: Validated PlannerContext with horizon=2, R1_hat_cycles populated.

        Raises:
            ValueError: If ctx is not valid for H2.
        """
        if ctx.horizon != 2:
            raise ValueError(f"ForwardModel requires horizon=2, got {ctx.horizon}")
        if ctx.R1_hat_cycles is None:
            raise ValueError("ForwardModel requires R1_hat_cycles for H2")
        if ctx.R1_hat_provenance is None:
            raise ValueError("ForwardModel requires R1_hat_provenance for H2")

        self.ctx = ctx
        self.K = ctx.maintenance_capacity
        self.R1_hat_cycles = ctx.R1_hat_cycles
        self.delta_cycles = ctx.delta_cycles
        self.rul_scale = ctx.rul_scale
        self.age_scale_cycles = ctx.age_scale_cycles
        self.action_table = ctx.action_table
        self.num_actions = len(ctx.action_table)

        # Cost regime parameters
        self.c_pm = ctx.c_pm
        self.c_f = ctx.c_f
        self.c_u = ctx.c_u

        # Risk model (frozen)
        self.risk_model_id = ctx.risk_model_id
        self.risk_temperature = ctx.risk_temperature

        # Build M4 optimizer for future cost computation
        self._m4_context = MyopicContext(
            maintenance_capacity=ctx.maintenance_capacity,
            delta_cycles=ctx.delta_cycles,
            rul_scale=ctx.rul_scale,
            age_scale_cycles=ctx.age_scale_cycles,
            action_table=ctx.action_table,
            c_pm=ctx.c_pm,
            c_f=ctx.c_f,
            c_u=ctx.c_u,
            risk_model_id=ctx.risk_model_id,
        )
        self._m4_optimizer = ExactMyopicOptimizer(
            context=self._m4_context,
            risk_temperature=M4_RISK_TEMPERATURE,
            tie_tolerance=1e-9,
        )

        # Precompute R1_hat normalized (for reset slots)
        self._R1_hat_norm = np.clip(self.R1_hat_cycles / self.rul_scale, 0.0, 1.0)

    def _decode_observation(self, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode observation into age_cycles and pred_rul_cycles.

        Args:
            observation: shape (10,), float32, range [0,1]

        Returns:
            Tuple of (age_cycles, pred_rul_cycles) each shape (5,), float64
        """
        # Validate
        if observation.shape != (10,):
            raise ValueError(f"Observation shape must be (10,), got {observation.shape}")
        if not np.issubdtype(observation.dtype, np.floating):
            raise ValueError(f"Observation dtype must be floating, got {observation.dtype}")
        if not np.all(np.isfinite(observation)):
            raise ValueError("Observation contains non-finite values")
        if np.any(observation < 0) or np.any(observation > 1):
            raise ValueError(f"Observation values must be in [0,1], got range [{observation.min():.4f}, {observation.max():.4f}]")

        obs = observation.reshape(5, 2)
        ages_norm = obs[:, 0]
        pred_ruls_norm = obs[:, 1]

        pred_rul_cycles = pred_ruls_norm * self.rul_scale
        age_cycles = ages_norm * self.age_scale_cycles

        return age_cycles, pred_rul_cycles

    def _compute_immediate_cost(
        self,
        action_id: int,
        pred_rul_cycles: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """
        Compute immediate cost breakdown for an action.

        Returns: (preventive_cost, unused_life_cost, failure_cost, total_cost)
        """
        selected_slots = self.action_table[action_id]
        selected_set = set(selected_slots)
        N = 5

        # Preventive cost
        preventive_cost = self.c_pm * len(selected_slots)

        # Unused life cost
        if len(selected_slots) == 0:
            unused_life_cost = 0.0
        else:
            selected_ruls = pred_rul_cycles[list(selected_slots)]
            normalized_ruls = np.clip(selected_ruls / self.rul_scale, 0.0, 1.0)
            unused_life_cost = self.c_u * float(np.sum(normalized_ruls))

        # Failure cost
        non_selected = [i for i in range(N) if i not in selected_set]
        if len(non_selected) == 0:
            failure_cost = 0.0
        else:
            non_selected_ruls = pred_rul_cycles[non_selected]
            failure_risks = compute_failure_risk(
                predicted_rul_cycles=non_selected_ruls,
                delta_cycles=self.delta_cycles,
                risk_model_id=self.risk_model_id,
                risk_temperature=self.risk_temperature,
            )
            failure_cost = self.c_f * float(np.sum(failure_risks))

        total_cost = preventive_cost + unused_life_cost + failure_cost
        return preventive_cost, unused_life_cost, failure_cost, total_cost

    def _get_unmaintained_slots(self, action_id: int) -> List[int]:
        """Get unmaintained slot indices U(a) for action a."""
        selected = set(self.action_table[action_id])
        return [i for i in range(5) if i not in selected]

    def _compute_branch_probability(
        self,
        failure_mask: Tuple[int, ...],
        unmaintained: List[int],
        pred_rul_cycles: np.ndarray,
    ) -> float:
        """
        Compute P(F | o, a) under conditional independence.

        P(F | o, a) = prod_{i in F} p_fail_i * prod_{i in U\F} (1 - p_fail_i)
        """
        unmaintained_set = set(unmaintained)
        failure_set = set(failure_mask)

        if not unmaintained:
            return 1.0

        # Get failure probabilities for unmaintained slots
        unmaintained_ruls = pred_rul_cycles[unmaintained]
        p_fail = compute_failure_risk(
            predicted_rul_cycles=unmaintained_ruls,
            delta_cycles=self.delta_cycles,
            risk_model_id=self.risk_model_id,
            risk_temperature=self.risk_temperature,
        )

        prob = 1.0
        for idx, slot in enumerate(unmaintained):
            p = p_fail[idx]
            if slot in failure_set:
                prob *= p
            else:
                prob *= (1.0 - p)

        return prob

    def _construct_next_public_state(
        self,
        observation: np.ndarray,
        action_id: int,
        failure_mask: Tuple[int, ...],
    ) -> PublicNextState:
        """
        Construct next public state o'(o, a, F).

        - Maintained slots S(a) AND failed slots F: reset to age=0, pred_rul_norm = R1_hat_norm (canonical)
        - Surviving unmaintained slots U(a)\F: age += delta, pred_rul = clip(pred_rul - delta, 0, 125), renormalized
        """
        age_cycles, pred_rul_cycles = self._decode_observation(observation)

        selected_slots = set(self.action_table[action_id])
        unmaintained = [i for i in range(5) if i not in selected_slots]
        failure_set = set(failure_mask)

        new_age_cycles = age_cycles.copy()
        new_pred_rul_cycles = pred_rul_cycles.copy()

        # Reset maintained slots - use canonical normalized R1_hat directly
        for slot in selected_slots:
            new_age_cycles[slot] = 0
            new_pred_rul_cycles[slot] = self.R1_hat_cycles  # will be normalized below

        # Reset failed slots - use canonical normalized R1_hat directly
        for slot in failure_mask:
            new_age_cycles[slot] = 0
            new_pred_rul_cycles[slot] = self.R1_hat_cycles  # will be normalized below

        # Age and degrade surviving unmaintained slots
        for slot in unmaintained:
            if slot not in failure_set:
                new_age_cycles[slot] = age_cycles[slot] + self.delta_cycles
                new_pred_rul_cycles[slot] = np.clip(
                    pred_rul_cycles[slot] - self.delta_cycles, 0.0, self.rul_scale
                )

        # Normalize
        new_age_norm = np.clip(new_age_cycles / self.age_scale_cycles, 0.0, 1.0)
        new_pred_rul_norm = np.clip(new_pred_rul_cycles / self.rul_scale, 0.0, 1.0).astype(np.float64)

        # FIX: Override reset slots with canonical precomputed _R1_hat_norm
        # This ensures reset pred_rul_norm equals the single authoritative computation
        reset_slots = selected_slots | failure_set
        for slot in reset_slots:
            new_pred_rul_norm[slot] = self._R1_hat_norm

        return PublicNextState(
            age_norm=new_age_norm.astype(np.float64),
            pred_rul_norm=new_pred_rul_norm.astype(np.float64),
        )

    def _compute_future_cost(self, next_state: PublicNextState) -> Tuple[int, float]:
        """
        Compute min_{a'} estimated_cost(o', a') using M4 optimizer.

        Returns: (best_action_id, min_cost)
        """
        next_obs = next_state.to_observation()
        result = self._m4_optimizer.select_action_with_details(next_obs)
        action_id = result["selected_action_id"]
        cost = result["estimated_cost"]
        return action_id, cost

    def compute_J2(self, observation: np.ndarray, action_id: int) -> Tuple[float, float, List[Branch]]:
        """
        Compute J2(o, a) for a specific action.

        J2(o, a) = immediate_estimated_cost(o, a) + γ * Σ_F P(F|o,a) * min_{a'} future_estimated_cost(o'(o,a,F), a')

        Args:
            observation: Current public observation (10,)
            action_id: Current action to evaluate

        Returns:
            Tuple of (J2, immediate_cost, branches_list)
        """
        # Decode observation
        age_cycles, pred_rul_cycles = self._decode_observation(observation)

        # Immediate cost breakdown
        prev_cost, unused_cost, fail_cost, immediate_cost = self._compute_immediate_cost(
            action_id, pred_rul_cycles
        )

        # Get unmaintained slots
        unmaintained = self._get_unmaintained_slots(action_id)

        # Enumerate all failure masks F ⊆ U(a)
        branches = []
        expected_future_cost = 0.0

        num_unmaintained = len(unmaintained)
        for mask_int in range(1 << num_unmaintained):
            # Build failure mask
            failure_mask = tuple(
                unmaintained[i] for i in range(num_unmaintained) if (mask_int >> i) & 1
            )

            # Branch probability
            prob = self._compute_branch_probability(
                failure_mask, unmaintained, pred_rul_cycles
            )

            # Next public state
            next_state = self._construct_next_public_state(
                observation, action_id, failure_mask
            )

            # Future cost: min over a' of estimated_cost(o', a')
            best_future_a, best_future_cost = self._compute_future_cost(next_state)

            branch = Branch(
                failure_mask=failure_mask,
                probability=prob,
                next_state=next_state,
                best_future_action_id=best_future_a,
                best_future_cost=best_future_cost,
            )
            branches.append(branch)

            expected_future_cost += prob * best_future_cost

        # J2 = immediate + gamma * expected_future
        J2 = immediate_cost + self.ctx.gamma * expected_future_cost

        return J2, immediate_cost, branches

    def get_branches_for_action(self, observation: np.ndarray, action_id: int) -> List[Branch]:
        """Get all branches for an action (for diagnostics)."""
        _, _, branches = self.compute_J2(observation, action_id)
        return branches