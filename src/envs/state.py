"""
Slot state representation for Milestone 2 Selective Maintenance Environment.

Implements an immutable slot-state dataclass that tracks:
- slot_index (persistent fleet slot index 0..N-1)
- split (environment split)
- unit_id (current engine/unit ID)
- cycle (current cycle index in trajectory)
- trajectory_length (full length of current trajectory)
- age_since_replacement_cycles (cycles since last replacement)
- trajectory_id (identifier for current trajectory)

The slot state is carefully controlled to prevent information leakage
to the agent while maintaining correct simulator accounting.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class SlotState:
    """
    Immutable state representation for a single fleet slot.

    Attributes:
        slot_index: Persistent fleet slot index (0 to N-1). Never changes.
        split: Environment split (predictor_train, rl_validation, rl_test).
        unit_id: Current engine/unit ID assigned to this slot.
        cycle: Current cycle index within the trajectory.
        trajectory_length: Full length of the current trajectory.
        age_since_replacement_cycles: Cycles since last replacement.
        trajectory_id: Identifier for the current trajectory.

    Invariants:
        - slot_index is fixed for the lifetime of the slot
        - age_since_replacement_cycles = cycle - 1 for fresh replacements
        - Active slots must have true_rul > 0 at decision boundaries
        - Replacements reset cycle to 1 and age to 0
    """

    slot_index: int
    split: str
    unit_id: int
    cycle: int
    trajectory_length: int
    age_since_replacement_cycles: int
    trajectory_id: str

    def __post_init__(self) -> None:
        """Validate slot state invariants."""
        errors: list[str] = []

        # Validate slot index in valid range
        if not (0 <= self.slot_index < 5):
            errors.append(
                f"slot_index must be in [0, 4], got {self.slot_index}"
            )

        # Validate split is non-empty
        if not self.split:
            errors.append("split cannot be empty")

        # Validate unit_id is positive
        if self.unit_id <= 0:
            errors.append(f"unit_id must be positive, got {self.unit_id}")

        # Validate cycle is positive
        if self.cycle <= 0:
            errors.append(f"cycle must be positive, got {self.cycle}")

        # Validate cycle does not exceed trajectory length
        if self.cycle > self.trajectory_length:
            errors.append(
                f"cycle ({self.cycle}) exceeds trajectory_length "
                f"({self.trajectory_length})"
            )

        # Validate trajectory_length is reasonable
        if self.trajectory_length <= 0:
            errors.append(f"trajectory_length must be positive, got {self.trajectory_length}")

        # Validate age is non-negative
        if self.age_since_replacement_cycles < 0:
            errors.append(
                f"age_since_replacement_cycles must be non-negative, "
                f"got {self.age_since_replacement_cycles}"
            )

        # Validate age is consistent with cycle
        # age = cycle - 1 for fresh trajectories; may differ after advancement
        if self.age_since_replacement_cycles > self.cycle:
            errors.append(
                f"age_since_replacement_cycles ({self.age_since_replacement_cycles}) "
                f"exceeds cycle ({self.cycle})"
            )

        # Validate trajectory_id is non-empty
        if not self.trajectory_id:
            errors.append("trajectory_id cannot be empty")

        if errors:
            raise ValueError("SlotState validation failed:\n  - " + "\n  - ".join(errors))

    def is_active(self) -> bool:
        """
        Check if this slot is active (not failed, not under maintenance).

        Returns:
            True if the slot is ready for decision-making.
        """
        # In the minimal environment, all slots are active unless failed
        # Failure is detected during advancement, not stored here
        return True

    def is_fresh(self) -> bool:
        """
        Check if this slot is freshly replaced (age = 0, cycle = 1).

        Returns:
            True if the slot was just replaced.
        """
        return self.cycle == 1 and self.age_since_replacement_cycles == 0

    def advance_cycles(self, delta: int) -> "SlotState":
        """
        Create a new state with advanced cycle and age.

        This is a pure function that returns a new SlotState.

        Args:
            delta: Number of cycles to advance (typically 5).

        Returns:
            New SlotState with updated cycle and age.

        Raises:
            ValueError: If advancing would exceed trajectory length.
        """
        new_cycle = self.cycle + delta
        if new_cycle > self.trajectory_length:
            raise ValueError(
                f"Cannot advance slot {self.slot_index} by {delta} cycles: "
                f"would exceed trajectory_length {self.trajectory_length} "
                f"(current cycle {self.cycle})"
            )

        return replace(
            self,
            cycle=new_cycle,
            age_since_replacement_cycles=self.age_since_replacement_cycles + delta,
        )

    @classmethod
    def create_fresh(
        cls,
        slot_index: int,
        split: str,
        unit_id: int,
        trajectory_length: int,
        trajectory_id: str,
    ) -> "SlotState":
        """
        Create a fresh slot state (just replaced).

        Sets cycle = 1 and age = 0.

        Args:
            slot_index: Fleet slot index.
            split: Environment split.
            unit_id: Unit/engine ID.
            trajectory_length: Full trajectory length.
            trajectory_id: Trajectory identifier.

        Returns:
            New SlotState with fresh settings.
        """
        return cls(
            slot_index=slot_index,
            split=split,
            unit_id=unit_id,
            cycle=1,
            trajectory_length=trajectory_length,
            age_since_replacement_cycles=0,
            trajectory_id=trajectory_id,
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary with deterministic ordering."""
        return {
            "slot_index": self.slot_index,
            "split": self.split,
            "unit_id": self.unit_id,
            "cycle": self.cycle,
            "trajectory_length": self.trajectory_length,
            "age_since_replacement_cycles": self.age_since_replacement_cycles,
            "trajectory_id": self.trajectory_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SlotState":
        """Create from dictionary."""
        return cls(
            slot_index=int(data["slot_index"]),
            split=str(data["split"]),
            unit_id=int(data["unit_id"]),
            cycle=int(data["cycle"]),
            trajectory_length=int(data["trajectory_length"]),
            age_since_replacement_cycles=int(data["age_since_replacement_cycles"]),
            trajectory_id=str(data["trajectory_id"]),
        )