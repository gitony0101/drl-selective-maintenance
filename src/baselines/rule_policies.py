"""
Rule-based policies for Milestone 3 Baselines.

Implements five practical policy families that receive only
observation + PolicyContext (no true RUL, no diagnostic info):

1. CorrectiveOnly: Always return action 0 (empty subset)
2. RandomFeasible: Uniformly sample from legal actions
3. AgeThreshold: Select slots where age >= T_age
4. PredictedRULThreshold: Select slots where predicted_rul <= T_rul
5. GreedyPredictedRUL: Select K lowest-RUL slots when activated

All policies return native Python action IDs.
All actions come from the existing M2 ActionTable.
No policy may exceed K.
No random tie-breaking — deterministic by lower slot index.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .protocols import PolicyContext, Observation, ActionId, validate_practical_policy_context


def decode_observation(
    observation: Observation,
    context: PolicyContext,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode observation into per-slot features.

    Args:
        observation: Environment observation ndarray, shape (10,), dtype float32
        context: Policy context (used to get N=5)

    Returns:
        Tuple of (ages, predicted_ruls), each shape (5,)
        - ages: normalized age_since_replacement for each slot
        - predicted_ruls: normalized predicted RUL for each slot

    Observation layout (N=5, 2 features per slot):
        [slot_0_age, slot_0_pred_rul, slot_1_age, slot_1_pred_rul, ...]
    """
    n_slots = 5  # Fixed fleet size
    obs = observation.reshape(n_slots, 2)
    ages = obs[:, 0]  # normalized_age_since_replacement
    pred_ruls = obs[:, 1]  # normalized_predicted_rul
    return ages, pred_ruls


def denormalize_age(
    normalized_age: np.ndarray,
    age_scale_cycles: int = 341,
) -> np.ndarray:
    """
    Convert normalized age to cycles.

    Args:
        normalized_age: Age values in [0, 1], clip(age / 341, 0, 1)
        age_scale_cycles: Divisor used for normalization (341)

    Returns:
        Age in cycles (approximate, reversing the clip)
    """
    return normalized_age * age_scale_cycles


def denormalize_rul(
    normalized_rul: np.ndarray,
    rul_scale: float = 125.0,
) -> np.ndarray:
    """
    Convert normalized RUL to cycles.

    Args:
        normalized_rul: RUL values in [0, 1], clip(rul / 125, 0, 1)
        rul_scale: Divisor used for normalization (125.0)

    Returns:
        RUL in cycles (approximate, reversing the clip)
    """
    return normalized_rul * rul_scale


class CorrectiveOnly:
    """
    Corrective-only policy.

    Always returns action ID 0 (empty subset).
    Never performs preventive maintenance.
    Failures are handled by the environment's corrective replacement.

    Tests:
    - Empty action for K=1 and K=2
    - Zero PM count in full episode
    - Failures may occur
    """

    def __init__(self) -> None:
        pass

    def select_action(
        self,
        observation: Observation,
        context: PolicyContext,
    ) -> ActionId:
        """
        Select action ID 0 (empty subset).

        Args:
            observation: Environment observation (ignored)
            context: Policy context (ignored)

        Returns:
            Action ID 0

        Raises:
            ValueError: If context is OracleContext (practical policies cannot use true RUL)
        """
        # Validate that practical policy is not receiving OracleContext
        validate_practical_policy_context(context)
        return 0


class RandomFeasible:
    """
    Random feasible policy.

    Uniformly samples from all legal action IDs using policy-owned RNG.

    Requirements:
    - Use policy-owned numpy.random.Generator
    - Fixed seed reproduces identical action sequences
    - Never exceed maintenance capacity
    - Do not reinitialize RNG every step

    Tests:
    - Same seed gives same sequence
    - Different seed can differ
    - Every action legal
    - K never exceeded
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Initialize random policy with its own RNG.

        Args:
            seed: Fixed seed for reproducibility. If None, uses random seed.
        """
        self.rng = np.random.default_rng(seed)

    def select_action(
        self,
        observation: Observation,
        context: PolicyContext,
    ) -> ActionId:
        """
        Uniformly sample from all legal action IDs.

        Args:
            observation: Environment observation (ignored)
            context: Policy context (used to get num_actions)

        Returns:
            Random action ID in [0, num_actions)

        Raises:
            ValueError: If context is OracleContext (practical policies cannot use true RUL)
        """
        # Validate that practical policy is not receiving OracleContext
        validate_practical_policy_context(context)

        num_actions = len(context.action_table)
        return int(self.rng.integers(0, num_actions))


class AgeThreshold:
    """
    Age threshold policy.

    Selects slots where age_since_replacement >= T_age (in cycles).

    Candidate condition:
        age_i >= T_age

    Tie-break (when candidates > K):
        1. Highest age first
        2. Lower slot index

    If no candidate exists: Return empty action (ID 0).

    Tests:
    - Below threshold excluded
    - Equal threshold included
    - Highest age selected when capacity binds
    - Deterministic slot tie-break
    """

    def __init__(self, threshold: float) -> None:
        """
        Initialize age threshold policy.

        Args:
            threshold: Age threshold in cycles. Slots with age >= threshold are candidates.
        """
        self.threshold = threshold

    def select_action(
        self,
        observation: Observation,
        context: PolicyContext,
    ) -> ActionId:
        """
        Select slots where age >= threshold.

        Args:
            observation: Environment observation, shape (10,), dtype float32
            context: Policy context with action_table and maintenance_capacity

        Returns:
            Action ID for selected subset

        Raises:
            ValueError: If context is OracleContext (practical policies cannot use true RUL)
        """
        # Validate that practical policy is not receiving OracleContext
        validate_practical_policy_context(context)

        ages_norm, _ = decode_observation(observation, context)

        # Convert threshold to normalized age
        threshold_norm = self.threshold / context.age_scale_cycles

        # Find candidate slots: age >= threshold
        candidates = np.where(ages_norm >= threshold_norm)[0]

        if len(candidates) == 0:
            return 0  # Empty action

        if len(candidates) <= context.maintenance_capacity:
            # All candidates fit — sort by slot index for determinism
            selected = tuple(sorted(candidates))
        else:
            # Need to select top-K by age, tie-break by slot index
            # Sort by (-age, slot_index) — highest age first, then lowest index
            candidate_ages = ages_norm[candidates]
            sorted_indices = np.lexsort((candidates, -candidate_ages))
            top_k = sorted_indices[: context.maintenance_capacity]
            selected = tuple(sorted(candidates[top_k]))

        # Convert slot subset to action ID
        return self._slots_to_action_id(selected, context.action_table)

    def _slots_to_action_id(
        self,
        slots: Tuple[int, ...],
        action_table: Tuple[Tuple[int, ...], ...],
    ) -> ActionId:
        """Convert slot subset to action ID."""
        normalized = tuple(sorted(slots))
        for action_id, table_slots in enumerate(action_table):
            if table_slots == normalized:
                return action_id
        raise ValueError(f"Slot subset {slots} not found in action table")


class PredictedRULThreshold:
    """
    Predicted RUL threshold policy.

    Selects slots where predicted_rul <= T_rul (in cycles).

    Candidate condition:
        predicted_rul_i <= T_rul

    Tie-break (when candidates > K):
        1. Lowest predicted RUL first
        2. Lower slot index

    If no candidate exists: Return empty action (ID 0).

    Tests:
    - Above threshold excluded
    - Equal threshold included
    - Lowest predicted RUL selected
    - Deterministic slot tie-break
    """

    def __init__(self, threshold: float) -> None:
        """
        Initialize predicted RUL threshold policy.

        Args:
            threshold: RUL threshold in cycles. Slots with RUL <= threshold are candidates.
        """
        self.threshold = threshold

    def select_action(
        self,
        observation: Observation,
        context: PolicyContext,
    ) -> ActionId:
        """
        Select slots where predicted_rul <= threshold.

        Args:
            observation: Environment observation, shape (10,), dtype float32
            context: Policy context with action_table and maintenance_capacity

        Returns:
            Action ID for selected subset

        Raises:
            ValueError: If context is OracleContext (practical policies cannot use true RUL)
        """
        # Validate that practical policy is not receiving OracleContext
        validate_practical_policy_context(context)
        _, pred_ruls_norm = decode_observation(observation, context)

        # Convert threshold to normalized RUL
        threshold_norm = self.threshold / context.rul_scale

        # Find candidate slots: predicted_rul <= threshold
        candidates = np.where(pred_ruls_norm <= threshold_norm)[0]

        if len(candidates) == 0:
            return 0  # Empty action

        if len(candidates) <= context.maintenance_capacity:
            # All candidates fit — sort by slot index for determinism
            selected = tuple(sorted(candidates))
        else:
            # Need to select top-K by lowest RUL, tie-break by slot index
            # Sort by (rul, slot_index) — lowest RUL first, then lowest index
            candidate_ruls = pred_ruls_norm[candidates]
            sorted_indices = np.lexsort((candidates, candidate_ruls))
            top_k = sorted_indices[: context.maintenance_capacity]
            selected = tuple(sorted(candidates[top_k]))

        # Convert slot subset to action ID
        return self._slots_to_action_id(selected, context.action_table)

    def _slots_to_action_id(
        self,
        slots: Tuple[int, ...],
        action_table: Tuple[Tuple[int, ...], ...],
    ) -> ActionId:
        """Convert slot subset to action ID."""
        normalized = tuple(sorted(slots))
        for action_id, table_slots in enumerate(action_table):
            if table_slots == normalized:
                return action_id
        raise ValueError(f"Slot subset {slots} not found in action table")


class GreedyPredictedRUL:
    """
    Greedy predicted RUL policy with activation threshold.

    Uses activation threshold T_activate:
    - If min(predicted_rul) > T_activate: return empty action
    - Otherwise: select K slots with lowest predicted RUL

    Activation rule:
        if min_i(predicted_rul_i) > T_activate:
            return empty_action

    Once activated:
        1. Rank every slot by predicted RUL ascending
        2. Select the lowest-RUL K slots
        3. Tie-break by lower slot index

    Distinct from PredictedRULThreshold:
    - Greedy always selects up to K once activated
    - Threshold selects only those below threshold

    Tests:
    - No activation returns empty action
    - Activation selects up to K lowest-RUL slots
    - Behavior differs from threshold policy in a crafted state
    """

    def __init__(self, activation_threshold: float) -> None:
        """
        Initialize greedy policy with activation threshold.

        Args:
            activation_threshold: RUL threshold in cycles. If min RUL > threshold, do nothing.
        """
        self.activation_threshold = activation_threshold

    def select_action(
        self,
        observation: Observation,
        context: PolicyContext,
    ) -> ActionId:
        """
        Select K lowest-RUL slots if activated.

        Args:
            observation: Environment observation, shape (10,), dtype float32
            context: Policy context with action_table and maintenance_capacity

        Returns:
            Action ID for selected subset

        Raises:
            ValueError: If context is OracleContext (practical policies cannot use true RUL)
        """
        # Validate that practical policy is not receiving OracleContext
        validate_practical_policy_context(context)
        _, pred_ruls_norm = decode_observation(observation, context)

        # Convert threshold to normalized RUL
        threshold_norm = self.activation_threshold / context.rul_scale

        # Check activation: min(predicted_rul) <= threshold
        min_rul = pred_ruls_norm.min()
        if min_rul > threshold_norm:
            return 0  # Not activated — empty action

        # Activated: select K slots with lowest RUL
        # Sort by (rul, slot_index) — lowest RUL first, then lowest index
        all_slots = np.arange(5)
        sorted_indices = np.lexsort((all_slots, pred_ruls_norm))
        top_k = sorted_indices[: context.maintenance_capacity]
        selected = tuple(sorted(top_k))

        # Convert slot subset to action ID
        return self._slots_to_action_id(selected, context.action_table)

    def _slots_to_action_id(
        self,
        slots: Tuple[int, ...],
        action_table: Tuple[Tuple[int, ...], ...],
    ) -> ActionId:
        """Convert slot subset to action ID."""
        normalized = tuple(sorted(slots))
        for action_id, table_slots in enumerate(action_table):
            if table_slots == normalized:
                return action_id
        raise ValueError(f"Slot subset {slots} not found in action table")