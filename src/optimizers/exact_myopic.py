"""
Exact Myopic Current-Window Optimizer for Milestone 4.

Implements deterministic optimization by:
1. Enumerating all feasible maintenance actions
2. Estimating cost for each action using implementable information
3. Selecting action with minimum estimated cost
4. Tie-breaking by smallest action_id

Information Barrier:
- Uses only normalized predicted RUL and age from observation
- Never uses true_rul, true_rul_capped, unit_id, trajectory_id
- Never uses diagnostic info or future transitions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from .failure_risk import compute_failure_risk, validate_risk_model_parameters


@dataclass(frozen=True)
class MyopicContext:
    """
    Immutable context for Exact Myopic optimizer.

    Contains only public information that implementable policies may access:
    - maintenance_capacity (K)
    - delta_cycles (5)
    - rul_scale (125.0)
    - age_scale_cycles (341)
    - action_table (tuple of action subsets)
    - cost regime coefficients (c_pm, c_f, c_u)
    - risk_model_id

    Does NOT contain:
    - true_rul, true_rul_capped
    - trajectory_id, unit_id, trajectory_length
    - split identity, scenario_id
    - diagnostic info
    """

    maintenance_capacity: int
    delta_cycles: int
    rul_scale: float
    age_scale_cycles: int
    action_table: Tuple[Tuple[int, ...], ...]
    c_pm: float
    c_f: float
    c_u: float
    risk_model_id: str

    def __post_init__(self) -> None:
        """Validate context parameters."""
        if self.maintenance_capacity < 0:
            raise ValueError(
                f"maintenance_capacity must be non-negative, got {self.maintenance_capacity}"
            )
        if self.delta_cycles <= 0:
            raise ValueError(
                f"delta_cycles must be positive, got {self.delta_cycles}"
            )
        if self.rul_scale <= 0:
            raise ValueError(
                f"rul_scale must be positive, got {self.rul_scale}"
            )
        if self.age_scale_cycles <= 0:
            raise ValueError(
                f"age_scale_cycles must be positive, got {self.age_scale_cycles}"
            )
        if self.c_pm < 0:
            raise ValueError(f"c_pm must be non-negative, got {self.c_pm}")
        if self.c_f < 0:
            raise ValueError(f"c_f must be non-negative, got {self.c_f}")
        if self.c_u < 0:
            raise ValueError(f"c_u must be non-negative, got {self.c_u}")

        # Validate all action subsets are within fleet size
        fleet_size = 5  # Fixed N=5
        for action_id, subset in enumerate(self.action_table):
            for slot in subset:
                if not (0 <= slot < fleet_size):
                    raise ValueError(
                        f"Action {action_id} contains invalid slot {slot}"
                    )
            if len(subset) > self.maintenance_capacity:
                raise ValueError(
                    f"Action {action_id} has {len(subset)} slots, "
                    f"exceeds K={self.maintenance_capacity}"
                )

        # Validate risk model
        validate_risk_model_parameters(self.risk_model_id)


@dataclass
class ActionCostBreakdown:
    """Cost breakdown for a single action."""

    action_id: int
    selected_slots: Tuple[int, ...]
    preventive_cost: float
    unused_life_cost: float
    failure_cost: float
    total_cost: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "action_id": self.action_id,
            "selected_slots": list(self.selected_slots),
            "preventive_cost": self.preventive_cost,
            "unused_life_cost": self.unused_life_cost,
            "failure_cost": self.failure_cost,
            "total_cost": self.total_cost,
        }


class ExactMyopicOptimizer:
    """
    Exact Myopic Current-Window Optimizer.

    At every state, enumerates every feasible maintenance action and chooses
    the action with minimum estimated cost for the next maintenance window.

    Estimated cost formula:
        total_cost(S) = c_pm * |S|
                      + c_u * sum(clip(predicted_rul_i / rul_scale, 0, 1) for i in S)
                      + c_f * sum(p_fail_i for i not in S)

    Tie-breaking: Among equal-cost actions, choose smallest action_id.
    """

    def __init__(
        self,
        context: MyopicContext,
        risk_temperature: float = 10.0,
        tie_tolerance: float = 1e-9,
    ) -> None:
        """
        Initialize optimizer.

        Args:
            context: Immutable optimizer context.
            risk_temperature: Temperature for logistic risk model.
            tie_tolerance: Numerical tolerance for cost comparison.

        Raises:
            ValueError: If context or parameters invalid.
        """
        self.context = context
        self.risk_temperature = risk_temperature
        self.tie_tolerance = tie_tolerance

        # Validate risk temperature for logistic model
        if context.risk_model_id == "logistic_window_v1":
            if risk_temperature <= 0:
                raise ValueError(
                    f"logistic_window_v1 requires temperature > 0, got {risk_temperature}"
                )
            if not np.isfinite(risk_temperature):
                raise ValueError(
                    f"logistic_window_v1 requires finite temperature, got {risk_temperature}"
                )

        # Fleet size is fixed at N=5
        self.N = 5

    def _decode_observation(
        self,
        observation: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode observation into per-slot features.

        Args:
            observation: Environment observation, shape (10,), dtype float32

        Returns:
            Tuple of (predicted_rul_cycles, age_cycles), each shape (5,)

        Raises:
            ValueError: If observation shape, dtype, or values invalid.
        """
        # Validate shape
        if observation.shape != (10,):
            raise ValueError(
                f"Observation shape must be (10,), got {observation.shape}"
            )

        # Validate dtype
        if not np.issubdtype(observation.dtype, np.floating):
            raise ValueError(
                f"Observation dtype must be floating point, got {observation.dtype}"
            )

        # Validate finite
        if not np.all(np.isfinite(observation)):
            raise ValueError(
                f"Observation contains non-finite values"
            )

        # Validate range [0, 1]
        if np.any(observation < 0) or np.any(observation > 1):
            raise ValueError(
                f"Observation values must be in [0, 1], "
                f"got range [{observation.min():.4f}, {observation.max():.4f}]"
            )

        # Reshape to (N, 2)
        obs = observation.reshape(self.N, 2)

        # Extract features
        ages_norm = obs[:, 0]  # normalized_age_since_replacement
        pred_ruls_norm = obs[:, 1]  # normalized_predicted_rul

        # Denormalize
        pred_rul_cycles = pred_ruls_norm * self.context.rul_scale
        age_cycles = ages_norm * self.context.age_scale_cycles

        return pred_rul_cycles, age_cycles

    def _compute_preventive_cost(
        self,
        selected_slots: Tuple[int, ...],
    ) -> float:
        """
        Compute preventive maintenance cost estimate.

        Formula: c_pm * |S|

        Args:
            selected_slots: Slots selected for maintenance.

        Returns:
            Preventive cost estimate.
        """
        return self.context.c_pm * len(selected_slots)

    def _compute_unused_life_cost(
        self,
        selected_slots: Tuple[int, ...],
        pred_rul_cycles: np.ndarray,
    ) -> float:
        """
        Compute unused life cost estimate.

        Formula: c_u * sum(clip(predicted_rul_i / rul_scale, 0, 1) for i in S)

        Selected slots receive unused life cost because they are replaced
        while still having remaining predicted life.

        Args:
            selected_slots: Slots selected for maintenance.
            pred_rul_cycles: Predicted RUL in cycles for all slots.

        Returns:
            Unused life cost estimate.
        """
        if len(selected_slots) == 0:
            return 0.0

        # Get predicted RUL for selected slots
        selected_ruls = pred_rul_cycles[list(selected_slots)]

        # Normalize and clip: clip(predicted_rul / rul_scale, 0, 1)
        normalized_ruls = np.clip(
            selected_ruls / self.context.rul_scale,
            0.0,
            1.0,
        )

        # Sum and scale by c_u
        return self.context.c_u * float(np.sum(normalized_ruls))

    def _compute_failure_cost(
        self,
        selected_slots: Tuple[int, ...],
        pred_rul_cycles: np.ndarray,
    ) -> float:
        """
        Compute estimated failure cost.

        Formula: c_f * sum(p_fail_i for i not in S)

        Non-selected slots receive failure risk because they continue operating
        without preventive maintenance during this window.

        Args:
            selected_slots: Slots selected for maintenance.
            pred_rul_cycles: Predicted RUL in cycles for all slots.

        Returns:
            Estimated failure cost.
        """
        # Identify non-selected slots
        selected_set = set(selected_slots)
        non_selected_indices = [i for i in range(self.N) if i not in selected_set]

        if len(non_selected_indices) == 0:
            return 0.0

        # Get predicted RUL for non-selected slots
        non_selected_ruls = pred_rul_cycles[non_selected_indices]

        # Compute failure risk for non-selected slots
        failure_risks = compute_failure_risk(
            predicted_rul_cycles=non_selected_ruls,
            delta_cycles=self.context.delta_cycles,
            risk_model_id=self.context.risk_model_id,
            risk_temperature=self.risk_temperature,
        )

        # Sum and scale by c_f
        return self.context.c_f * float(np.sum(failure_risks))

    def _evaluate_action(
        self,
        action_id: int,
        pred_rul_cycles: np.ndarray,
    ) -> ActionCostBreakdown:
        """
        Evaluate a single action.

        Args:
            action_id: Action ID from action table.
            pred_rul_cycles: Predicted RUL in cycles for all slots.

        Returns:
            Cost breakdown for the action.
        """
        # Get selected slots
        selected_slots = self.context.action_table[action_id]

        # Compute cost components
        preventive_cost = self._compute_preventive_cost(selected_slots)
        unused_life_cost = self._compute_unused_life_cost(selected_slots, pred_rul_cycles)
        failure_cost = self._compute_failure_cost(selected_slots, pred_rul_cycles)
        total_cost = preventive_cost + unused_life_cost + failure_cost

        return ActionCostBreakdown(
            action_id=action_id,
            selected_slots=selected_slots,
            preventive_cost=preventive_cost,
            unused_life_cost=unused_life_cost,
            failure_cost=failure_cost,
            total_cost=total_cost,
        )

    def evaluate_all_actions(
        self,
        observation: np.ndarray,
    ) -> List[ActionCostBreakdown]:
        """
        Evaluate all feasible actions for the current observation.

        Args:
            observation: Environment observation, shape (10,), dtype float32

        Returns:
            List of cost breakdowns for all actions, ordered by action_id.

        Raises:
            ValueError: If observation invalid.
        """
        # Decode observation
        pred_rul_cycles, _ = self._decode_observation(observation)

        # Evaluate all actions
        num_actions = len(self.context.action_table)
        results = []
        for action_id in range(num_actions):
            breakdown = self._evaluate_action(action_id, pred_rul_cycles)
            results.append(breakdown)

        return results

    def select_action(
        self,
        observation: np.ndarray,
    ) -> Tuple[int, Tuple[int, ...], float]:
        """
        Select action with minimum estimated cost.

        Uses deterministic tie-breaking: smallest action_id wins.

        Args:
            observation: Environment observation, shape (10,), dtype float32

        Returns:
            Tuple of:
            - action_id: Selected action ID
            - selected_slots: Tuple of selected slot indices
            - estimated_cost: Total estimated cost for selected action

        Raises:
            ValueError: If observation invalid.
        """
        # Evaluate all actions
        evaluations = self.evaluate_all_actions(observation)

        # Find minimum cost with deterministic tie-breaking
        min_cost = float("inf")
        best_action_id = -1
        best_slots: Tuple[int, ...] = ()

        for eval_result in evaluations:
            # Use tolerance-aware comparison for "strictly better"
            if eval_result.total_cost < min_cost - self.tie_tolerance:
                min_cost = eval_result.total_cost
                best_action_id = eval_result.action_id
                best_slots = eval_result.selected_slots
            # Tie: keep current best (smallest action_id wins)

        return best_action_id, best_slots, min_cost

    def select_action_with_details(
        self,
        observation: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Select action and return complete diagnostic information.

        Args:
            observation: Environment observation, shape (10,), dtype float32

        Returns:
            Dictionary with:
            - selected_action_id
            - selected_slots
            - estimated_cost
            - all_action_costs: List of all action cost breakdowns
            - risk_model_id
            - maintenance_capacity
        """
        evaluations = self.evaluate_all_actions(observation)
        action_id, slots, cost = self.select_action(observation)

        return {
            "selected_action_id": action_id,
            "selected_slots": list(slots),
            "estimated_cost": cost,
            "all_action_costs": [e.to_dict() for e in evaluations],
            "risk_model_id": self.context.risk_model_id,
            "maintenance_capacity": self.context.maintenance_capacity,
        }