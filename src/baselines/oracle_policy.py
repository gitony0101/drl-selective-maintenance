"""
True-RUL Oracle Policy for Milestone 3 Baselines.

Implements a diagnostic-only oracle policy that uses true RUL through
OracleContext. This is a privileged-information diagnostic benchmark.

Requires:
- allow_oracle=True
- diagnostic_mode=True

The oracle selects slots where true_rul <= T_oracle.

Candidate condition:
    true_rul_i <= T_oracle

Tie-break (when candidates > K):
    1. Lowest true RUL first
    2. Lower slot index

If no candidate exists: Return empty action (ID 0).

Tests:
- Uses true RUL only through OracleContext
- Ordinary policy cannot receive OracleContext
- Hidden state absent from practical policy input
- Lowest true RUL selected
- Diagnostic label preserved
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .protocols import OracleContext, Observation, ActionId


# Type for diagnostic info containing true RUL
DiagnosticInfo = dict  # Contains slot_states with true_rul


class OracleThreshold:
    """
    True-RUL oracle threshold policy.

    Uses true RUL through OracleContext (diagnostic-only interface).
    This is a privileged-information diagnostic benchmark.

    Label: "privileged-information diagnostic benchmark"

    Requires explicit allow_oracle=True and diagnostic_mode=True.
    """

    def __init__(self, threshold: float) -> None:
        """
        Initialize oracle threshold policy.

        Args:
            threshold: True RUL threshold in cycles. Slots with true_rul <= threshold are candidates.
        """
        self.threshold = threshold

    def select_action(
        self,
        observation: Observation,
        context: OracleContext,
        diagnostic_info: Optional[DiagnosticInfo] = None,
    ) -> ActionId:
        """
        Select slots where true_rul <= threshold.

        Args:
            observation: Environment observation (not used for selection)
            context: OracleContext with allow_oracle=True and diagnostic_mode=True
            diagnostic_info: Dict containing slot_states with true_rul values

        Returns:
            Action ID for selected subset

        Raises:
            ValueError: If context is not OracleContext or missing diagnostic_info
        """
        # Validate oracle context
        if not isinstance(context, OracleContext):
            raise ValueError(
                "OracleThreshold requires OracleContext. "
                "Practical policies must not use true RUL."
            )
        if not context.allow_oracle:
            raise ValueError("OracleThreshold requires allow_oracle=True")
        if not context.diagnostic_mode:
            raise ValueError("OracleThreshold requires diagnostic_mode=True")
        if diagnostic_info is None:
            raise ValueError("OracleThreshold requires diagnostic_info with true_rul values")

        # Extract true RUL from diagnostic info
        # Environment provides slot_{N}_diagnostic keys with true_rul inside each
        n_slots = 5  # Fixed fleet size
        true_ruls = np.zeros(n_slots)

        for slot_idx in range(n_slots):
            key = f"slot_{slot_idx}_diagnostic"
            if key not in diagnostic_info:
                raise ValueError(
                    f"diagnostic_info must contain '{key}' but only has keys: "
                    f"{list(diagnostic_info.keys())}"
                )
            slot_diag = diagnostic_info[key]
            if "true_rul" not in slot_diag:
                raise ValueError(f"'{key}' must contain 'true_rul'")
            true_ruls[slot_idx] = slot_diag["true_rul"]

        # Find candidate slots: true_rul <= threshold
        candidates = np.where(true_ruls <= self.threshold)[0]

        if len(candidates) == 0:
            return 0  # Empty action

        if len(candidates) <= context.maintenance_capacity:
            # All candidates fit — sort by slot index for determinism
            selected = tuple(sorted(candidates))
        else:
            # Need to select top-K by lowest true RUL, tie-break by slot index
            candidate_ruls = true_ruls[candidates]
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
