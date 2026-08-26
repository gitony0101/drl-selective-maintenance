"""
Reusable helper functions for Milestone 2 environment tests.

This module contains non-fixture helper functions that were previously
in conftest.py. Keep conftest.py limited to pytest fixtures only.

These helpers support:
- Finding real units in a split from PredictionStore
- Locating the verified first failure cycle for a unit
- Building exact failure-offset scenarios with real units
- Building simultaneous-failure scenarios
- Building mixed-event scenarios (PM + a different-slot failure)
- Recording PredictionStore wrapper for tracking lookups
"""

from typing import List, Tuple, Optional, Iterable
from pathlib import Path

from src.predictors.prediction_store import PredictionStore, PredictionResult
from src.envs.scenario_bank import Scenario, ScenarioBank


def find_unit_with_failure_at_cycle(
    prediction_store: PredictionStore,
    split: str,
    max_search_cycles: int = 350,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Find a unit that has a verified failure (true_rul <= 0) within max_search_cycles.

    The V2 cache is contiguous 1..trajectory_length with true_rul == trajectory_length -
    cycle, so the first failure cycle for a unit is always its trajectory_length.
    Returns (unit_id, trajectory_length) (i.e. the failure cycle), or (None, None).

    This function reads from the actual PredictionStore rather than inferring
    unit IDs or failure cycles.
    """
    units = prediction_store.get_units(split)
    for unit_id in units:
        for cycle in range(1, min(max_search_cycles + 1, 351)):
            pred: PredictionResult = prediction_store.get(split, unit_id, cycle)
            if pred.found and pred.true_rul <= 0:
                return (unit_id, cycle)
    return (None, None)


def find_unit_supporting_failure_offset(
    prediction_store: PredictionStore,
    split: str,
    failure_offset: int,
    exclude_units: Optional[Iterable[int]] = None,
) -> Optional[Tuple[int, int, int]]:
    """
    Find a real (unit, start_cycle, trajectory_length) such that a non-PM slot
    started at start_cycle will fail exactly at start_cycle + failure_offset.

    Given the cache invariant (contiguous 1..trajectory_length, true_rul decreases
    by 1 per cycle, first true_rul<=0 at trajectory_length), the failing slot must
    start at start_cycle = trajectory_length - failure_offset. For that to be a
    valid start state, start_cycle >= 1 (so trajectory_length > failure_offset) and
    the true_rul at start_cycle must be > 0 (it will be failure_offset > 0).

    Returns:
        (unit_id, start_cycle, trajectory_length) or None
    """
    excluded = set(exclude_units or ())
    units = prediction_store.get_units(split)
    for unit_id in units:
        if unit_id in excluded:
            continue
        # trajectory_length is the failure cycle for this unit.
        any_pred = prediction_store.get(split, unit_id, 1)
        if not any_pred.found:
            continue
        trajectory_length = any_pred.trajectory_length
        start_cycle = trajectory_length - failure_offset
        if start_cycle < 1:
            continue
        # Verify the cache rows exist and true_rul at start_cycle is > 0.
        start_pred = prediction_store.get(split, unit_id, start_cycle)
        if not start_pred.found or start_pred.true_rul <= 0:
            continue
        # Verify the failure-cycle row exists with true_rul <= 0.
        fail_pred = prediction_store.get(split, unit_id, trajectory_length)
        if not fail_pred.found or fail_pred.true_rul > 0:
            continue
        return (unit_id, start_cycle, trajectory_length)
    return None


def find_unit_with_no_failure_in_offset(
    prediction_store: PredictionStore,
    split: str,
    no_failure_window: int,
    exclude_units: Optional[Iterable[int]] = None,
) -> Optional[Tuple[int, int, int]]:
    """
    Find a real (unit, start_cycle, trajectory_length) such that for a non-PM slot
    started at start_cycle, NO failure occurs within c+1..c+no_failure_window.

    The first failure for the unit is at trajectory_length. For no failure in
    c+1..c+no_failure_window, we need trajectory_length > start + no_failure_window
    (i.e., the first failure is at c+(no_failure_window+1) or later). We also need
    the delta_cycles=5 advance to stay within the cache: start + 5 <= trajectory_length.

    For the brief's "no failure when first failure is at c+6" case, pass
    no_failure_window=5: the first failure is at c+6 (start + 6 == trajectory_length),
    and c+1..c+5 all have true_rul > 0. With start = trajectory_length - 6 and
    delta_cycles=5, the slot advances to start+5 = trajectory_length-1 (true_rul=1>0).
    """
    excluded = set(exclude_units or ())
    units = prediction_store.get_units(split)
    for unit_id in units:
        if unit_id in excluded:
            continue
        any_pred = prediction_store.get(split, unit_id, 1)
        if not any_pred.found:
            continue
        trajectory_length = any_pred.trajectory_length
        # start such that first failure is at c + (no_failure_window + 1):
        # trajectory_length = start + no_failure_window + 1
        target_start = trajectory_length - (no_failure_window + 1)
        if target_start < 1:
            continue
        # 5-cycle advance must stay within the cache (no overflow).
        if target_start + 5 > trajectory_length:
            continue
        start_pred = prediction_store.get(split, unit_id, target_start)
        if not start_pred.found or start_pred.true_rul <= 0:
            continue
        return (unit_id, target_start, trajectory_length)
    return None


def fill_other_units(
    prediction_store: PredictionStore,
    split: str,
    exclude_unit: int,
    n_other: int = 4,
) -> List[int]:
    """Return n_other distinct real units from the split, excluding exclude_unit."""
    units = prediction_store.get_units(split)
    return [u for u in units if u != exclude_unit][:n_other]


def build_failure_fixture_scenario(
    prediction_store: PredictionStore,
    split: str,
    failure_unit_id: int,
    failure_cycle: int,
    failure_offset: int,  # fail at current_cycle + failure_offset
    scenario_id: str,
    k_capacity: int = 2,
    horizon: int = 100,
    cost_regime_id: str = "failure-light-no-waste",
    replacement_seed: int = 6521,
    environment_seed: int = 6521,
) -> Scenario:
    """
    Build a scenario where a specific unit fails at a specific offset.

    The scenario starts the failure unit at cycle = failure_cycle - failure_offset,
    so that failure occurs exactly failure_offset cycles into advancement.

    Other slots are filled with distinct units from the same split that have
    valid predictions at cycle 1.

    Raises:
        ValueError: If required units or cycles are not found.
    """
    units = prediction_store.get_units(split)
    if len(units) < 5:
        raise ValueError(f"Split {split} has only {len(units)} units; need at least 5")

    # Compute start cycle for failure unit
    start_cycle = failure_cycle - failure_offset
    if start_cycle <= 0:
        raise ValueError(
            f"failure_cycle={failure_cycle} - offset={failure_offset} = {start_cycle} <= 0"
        )

    # Verify the failure unit has valid predictions at start_cycle
    pred = prediction_store.get(split, failure_unit_id, start_cycle)
    if not pred.found:
        raise ValueError(
            f"Missing prediction for {split} unit {failure_unit_id} at cycle {start_cycle}"
        )
    if pred.true_rul <= 0:
        raise ValueError(
            f"Unit {failure_unit_id} already has true_rul <= 0 at cycle {start_cycle}; "
            f"cannot test failure advancement"
        )

    # Select 4 other distinct units from the same split
    other_units = [u for u in units if u != failure_unit_id][:4]
    if len(other_units) < 4:
        raise ValueError(f"Need 4 other units in {split}, found {len(other_units)}")

    # Verify other units have valid predictions at cycle 1
    initial_cycles = [start_cycle] + [1, 1, 1, 1]
    initial_unit_ids = [failure_unit_id] + other_units

    for uid, cyc in zip(initial_unit_ids, initial_cycles):
        p = prediction_store.get(split, uid, cyc)
        if not p.found:
            raise ValueError(
                f"Missing prediction for {split} unit {uid} at cycle {cyc}"
            )
        if p.true_rul <= 0:
            raise ValueError(
                f"Unit {uid} has true_rul <= 0 at cycle {cyc}; cannot use as initial state"
            )

    return Scenario(
        scenario_id=scenario_id,
        split=split,
        initial_unit_ids=tuple(initial_unit_ids),
        initial_cycles=tuple(initial_cycles),
        replacement_seed=replacement_seed,
        environment_seed=environment_seed,
        episode_horizon=horizon,
        maintenance_capacity=k_capacity,
        cost_regime_id=cost_regime_id,
    )


def build_simultaneous_failure_scenario(
    prediction_store: PredictionStore,
    split: str,
    scenario_id: str,
    failure_offset: int = 2,
    k_capacity: int = 2,
    horizon: int = 100,
    cost_regime_id: str = "failure-light-no-waste",
    replacement_seed: int = 6521,
    environment_seed: int = 6521,
) -> Scenario:
    """
    Build a scenario where exactly two non-PM slots fail in the same decision step.

    Slot 0 and slot 1 are each placed at start_cycle = trajectory_length - failure_offset
    using two real units with sufficiently long trajectories. Slots 2-4 use real
    units started at cycle 1 that will NOT fail at c+1..c+failure_offset.

    Both non-PM failing slots fail at c+failure_offset in the SAME step.

    Raises:
        ValueError: if two suitable failing units cannot be found.
    """
    units = prediction_store.get_units(split)
    failing_units: List[Tuple[int, int, int]] = []
    used: set = set()
    for _ in range(2):
        result = find_unit_supporting_failure_offset(
            prediction_store, split, failure_offset, exclude_units=used
        )
        if result is None:
            break
        used.add(result[0])
        failing_units.append(result)

    if len(failing_units) < 2:
        raise ValueError(
            f"Could not find two real units in {split} supporting failure offset "
            f"{failure_offset} (found {len(failing_units)})"
        )

    # Fill slots 2 and 3 with healthy units; slot 4 as well.
    other_units = [u for u in units if u not in {failing_units[0][0], failing_units[1][0]}]
    healthy = []
    for u in other_units:
        if len(healthy) >= 3:
            break
        p = prediction_store.get(split, u, 1)
        if p.found and p.true_rul > 0:
            healthy.append(u)

    if len(healthy) < 3:
        raise ValueError(f"Need 3 healthy units in {split}, found {len(healthy)}")

    initial_unit_ids = [failing_units[0][0], failing_units[1][0]] + healthy[:3]
    initial_cycles = [failing_units[0][1], failing_units[1][1]] + [1, 1, 1]

    return Scenario(
        scenario_id=scenario_id,
        split=split,
        initial_unit_ids=tuple(initial_unit_ids),
        initial_cycles=tuple(initial_cycles),
        replacement_seed=replacement_seed,
        environment_seed=environment_seed,
        episode_horizon=horizon,
        maintenance_capacity=k_capacity,
        cost_regime_id=cost_regime_id,
    )


def build_mixed_event_scenario(
    prediction_store: PredictionStore,
    split: str,
    scenario_id: str,
    pm_slot: int = 0,
    failure_slot: int = 2,
    failure_offset: int = 2,
    k_capacity: int = 2,
    horizon: int = 100,
    cost_regime_id: str = "failure-light-no-waste",
    replacement_seed: int = 6521,
    environment_seed: int = 6521,
) -> Scenario:
    """
    Build a scenario where one non-PM slot is preventively maintained (agent action)
    while a DIFFERENT non-PM slot fails in the same decision step.

    pm_slot is prevented (agent-selected). failure_slot is placed at
    start_cycle = trajectory_length - failure_offset so it fails exactly at
    c+failure_offset. The two slots use distinct real units and are at different
    slot indices.

    Raises:
        ValueError if pm_slot == failure_slot or a suitable failing unit cannot
        be found.
    """
    if pm_slot == failure_slot:
        raise ValueError("pm_slot and failure_slot must differ")

    units = prediction_store.get_units(split)
    fail = find_unit_supporting_failure_offset(prediction_store, split, failure_offset)
    if fail is None:
        raise ValueError(
            f"No real unit in {split} supports failure offset {failure_offset}"
        )

    used = {fail[0]}
    # Pick a distinct unit for the PM slot started at a healthy cycle (cycle 1).
    pm_unit = None
    for u in units:
        if u in used:
            continue
        p = prediction_store.get(split, u, 1)
        if p.found and p.true_rul > 0:
            pm_unit = u
            break
    if pm_unit is None:
        raise ValueError(f"No healthy pm unit found in {split}")
    used.add(pm_unit)

    healthy: List[int] = []
    for u in units:
        if u in used:
            continue
        if len(healthy) >= 3:
            break
        p = prediction_store.get(split, u, 1)
        if p.found and p.true_rul > 0:
            healthy.append(u)
    if len(healthy) < 3:
        raise ValueError(f"Need 3 healthy filler units in {split}, found {len(healthy)}")

    # Assemble per-slot unit ids and initial cycles.
    slots_order: List[Tuple[int, int, int]] = []  # (slot_index, unit_id, cycle)
    filler_iter = iter(healthy)
    fail_placed = False
    pm_placed = False
    for slot_idx in range(5):
        if slot_idx == pm_slot:
            slots_order.append((slot_idx, pm_unit, 1))
            pm_placed = True
        elif slot_idx == failure_slot:
            slots_order.append((slot_idx, fail[0], fail[1]))
            fail_placed = True
        else:
            u = next(filler_iter)
            slots_order.append((slot_idx, u, 1))

    assert pm_placed and fail_placed

    initial_unit_ids = [t[1] for t in slots_order]
    initial_cycles = [t[2] for t in slots_order]

    return Scenario(
        scenario_id=scenario_id,
        split=split,
        initial_unit_ids=tuple(initial_unit_ids),
        initial_cycles=tuple(initial_cycles),
        replacement_seed=replacement_seed,
        environment_seed=environment_seed,
        episode_horizon=horizon,
        maintenance_capacity=k_capacity,
        cost_regime_id=cost_regime_id,
    )


def build_scenario_bank_for_split(
    split: str,
    prediction_store: PredictionStore,
    k_capacity: int = 2,
    num_scenarios: int = 5,
    bank_id_suffix: str = "",
) -> ScenarioBank:
    """
    Build a scenario bank for a given split using real units from PredictionStore.

    Uses actual units from the split and verifies all initial cycles exist.
    Does not use unit_id + k inference.
    """
    units = prediction_store.get_units(split)
    if len(units) < 5:
        raise ValueError(f"Split {split} has only {len(units)} units; need at least 5")

    scenarios = []
    for i in range(num_scenarios):
        # Select 5 distinct units deterministically
        start_idx = (i * 5) % len(units)
        selected_units = []
        for j in range(5):
            idx = (start_idx + j) % len(units)
            selected_units.append(units[idx])

        # Vary initial cycles slightly for diversity
        initial_cycles = tuple(1 + (i * 10) % 50 for _ in range(5))

        scenario = Scenario(
            scenario_id=f"{split}_smoke_{i:03d}{bank_id_suffix}",
            split=split,
            initial_unit_ids=tuple(selected_units),
            initial_cycles=initial_cycles,
            replacement_seed=6521 + i,
            environment_seed=6521 + i,
            episode_horizon=100,
            maintenance_capacity=k_capacity,
            cost_regime_id="failure-light-no-waste",
        )
        scenarios.append(scenario)

    return ScenarioBank(
        bank_id=f"{split}_smoke_bank{bank_id_suffix}",
        split=split,
        scenarios=tuple(scenarios),
    )


class RecordingPredictionStore:
    """
    Wrapper around PredictionStore that tracks all lookup requests.

    Use this in tests to verify that the environment requests the correct
    cycles and doesn't make spurious lookups beyond the first failure.
    """

    def __init__(self, wrapped: PredictionStore):
        self._wrapped = wrapped
        self._requests: List[Tuple[str, int, int]] = []  # (split, unit_id, cycle)

    def get(self, split: str, unit_id: int, cycle: int) -> PredictionResult:
        """Record and forward a prediction lookup."""
        self._requests.append((split, unit_id, cycle))
        return self._wrapped.get(split, unit_id, cycle)

    def get_units(self, split: str) -> List[int]:
        """Forward get_units call (not tracked)."""
        return self._wrapped.get_units(split)

    @property
    def requests(self) -> List[Tuple[str, int, int]]:
        """Return all recorded lookup requests."""
        return self._requests.copy()

    def clear(self) -> None:
        """Clear recorded requests."""
        self._requests.clear()

    def find_first_failure_request(self, unit_id: int) -> Optional[int]:
        """
        Find the first cycle requested for a unit that resulted in a failure.

        Returns the cycle at which true_rul <= 0 was first requested, or None
        if no failure was encountered.
        """
        for split, uid, cycle in self._requests:
            if uid == unit_id:
                pred = self._wrapped.get(split, uid, cycle)
                if pred.found and pred.true_rul <= 0:
                    return cycle
        return None

    def get_requested_cycles_for_unit(self, unit_id: int) -> List[int]:
        """Get all cycles requested for a specific unit."""
        return [cyc for split, uid, cyc in self._requests if uid == unit_id]

    def get_max_requested_cycle_for_unit(self, unit_id: int) -> Optional[int]:
        """Get the maximum cycle requested for a specific unit."""
        cycles = self.get_requested_cycles_for_unit(unit_id)
        return max(cycles) if cycles else None
