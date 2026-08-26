"""
Test Milestone 2 environment failure boundary detection.

Tests cover (exact assertions, no >= 1 shortcuts):
- failure at c+1, c+2, c+3, c+4, c+5 (each exactly one failure, exactly one failure cost)
- no failure when the first failure is at c+6 (window 1..5 has true_rul > 0)
- no PredictionStore request beyond the first failure cycle for that slot
- failure charged exactly once; no double charging
- exactly two simultaneous failures in one decision step (both counted, both replaced)
- one PM slot plus one DIFFERENT failing slot (PM and failure independent)
- corrective replacement cycle exactly 1
- corrective replacement age exactly 0
- corrective replacement does not consume K
- no residual-cycle advancement after corrective replacement
- current decision-boundary true_rul <= 0 raises ContractViolationError
- endpoint without a true_rul <= 0 record raises ContractViolationError
- missing intermediate prediction raises MissingPredictionError
- no silent replacement at any boundary

All fixtures use real units and cycles read from PredictionStore; start cycles
are derived from the unit's own trajectory_length (see m2_env_test_helpers).
"""

import numpy as np
import pytest

pytestmark = pytest.mark.requires_external_assets
from dataclasses import replace
from pathlib import Path

from src.envs import (
    SelectiveMaintenanceEnv,
    EnvironmentConfig,
    get_default_config,
    load_scenario_bank,
)
from src.envs.scenario_bank import Scenario, ScenarioBank
from src.envs.errors import (
    ContractViolationError,
    MissingPredictionError,
)
from src.predictors.prediction_store import load_default_prediction_store
from tests.m2_env_test_helpers import (
    RecordingPredictionStore,
    build_failure_fixture_scenario,
    build_mixed_event_scenario,
    build_simultaneous_failure_scenario,
    find_unit_supporting_failure_offset,
    find_unit_with_failure_at_cycle,
    find_unit_with_no_failure_in_offset,
)


PREDICTOR_TRAIN_SPLIT = "predictor_train"
K2 = 2
SMOKE_BANK = "data/scenario_banks/predictor_train_smoke.json"
DEFAULT_REGIME = "failure-light-no-waste"


@pytest.fixture
def prediction_store():
    """Load the V2 prediction store."""
    cache_dir = Path("data/processed/fd001/v2/06_PREDICTIONS/")
    return load_default_prediction_store(cache_dir)


def _make_env(
    prediction_store,
    scenario,
    k_capacity=K2,
    cost_regime_id=DEFAULT_REGIME,
    wrap_recording=False,
):
    """Construct an env over a single in-memory scenario, matching config to scenario."""
    bank = ScenarioBank(
        bank_id=f"{scenario.scenario_id}_bank",
        split=scenario.split,
        scenarios=(scenario,),
    )
    config = EnvironmentConfig(
        environment_version="m2_v1",
        split=scenario.split,
        fleet_size=5,
        maintenance_capacity=k_capacity,
        delta_cycles=5,
        episode_horizon=scenario.episode_horizon,
        age_scale_cycles=341,
        rul_scale=125.0,
        cost_regime_id=cost_regime_id,
        scenario_bank_path=SMOKE_BANK,
        prediction_cache_path="data/processed/fd001/v2/06_PREDICTIONS/",
        info_mode="normal",
        seed=6521,
    )
    store = prediction_store
    if wrap_recording:
        store = RecordingPredictionStore(prediction_store)
    env = SelectiveMaintenanceEnv(
        config=config,
        prediction_store=store,
        scenario_bank=bank,
        info_mode="diagnostic",
    )
    return env, store


def _build_failure_offset_scenario(prediction_store, offset, scenario_id, split=PREDICTOR_TRAIN_SPLIT):
    """Build a real-unit scenario whose slot 0 fails exactly at c+offset."""
    fail = find_unit_supporting_failure_offset(prediction_store, split, offset)
    assert fail is not None, f"No real unit supports failure offset {offset} in {split}"
    unit_id, start_cycle, trajectory_length = fail
    return build_failure_fixture_scenario(
        prediction_store,
        split=split,
        failure_unit_id=unit_id,
        failure_cycle=trajectory_length,
        failure_offset=offset,
        scenario_id=scenario_id,
        k_capacity=K2,
        horizon=100,
        cost_regime_id=DEFAULT_REGIME,
    )


class TestFailureOffsetExact:
    """Exact single-failure detection at c+1 through c+5, plus c+6 no-failure."""

    @pytest.mark.parametrize("offset", [1, 2, 3, 4, 5])
    def test_failure_at_offset(self, prediction_store, offset) -> None:
        """Failing slot started at trajectory_length - offset fails exactly at c+offset."""
        scenario = _build_failure_offset_scenario(
            prediction_store, offset, f"test_fail_offset_{offset}"
        )
        env, _ = _make_env(prediction_store, scenario, wrap_recording=True)
        recording = env.prediction_store
        env.reset()

        _, _, _, _, step_info = env.step(0)  # action 0 = no PM

        # Exactly one failure, charged exactly once.
        assert step_info["num_failures"] == 1, (
            f"offset {offset}: expected exactly 1 failure, got {step_info['num_failures']}"
        )
        assert step_info["failure_cost"] == 5.0, (
            f"offset {offset}: expected 5.0 (c_f=5), got {step_info['failure_cost']}"
        )
        # No PM was performed.
        assert step_info["num_preventive"] == 0
        assert step_info["preventive_cost"] == 0.0

        # Corrective replacement cycle must be exactly 1, age exactly 0.
        # Read both from actual post-step SlotState via diagnostic info.
        slot_diag = step_info["slot_0_diagnostic"]
        assert slot_diag["cycle"] == 1, f"expected cycle 1, got {slot_diag['cycle']}"
        assert slot_diag["age_since_replacement_cycles"] == 0, (
            f"corrective replacement age must be exactly 0, got "
            f"{slot_diag['age_since_replacement_cycles']}"
        )

    def test_no_failure_when_first_failure_at_c6(self, prediction_store) -> None:
        """When the first failure is at c+6, no failure occurs within c+1..c+5."""
        nf = find_unit_with_no_failure_in_offset(
            prediction_store, PREDICTOR_TRAIN_SPLIT, no_failure_window=5
        )
        assert nf is not None, "No real unit supports the c+6 no-failure window"
        unit_id, start_cycle, trajectory_length = nf

        scenario = build_failure_fixture_scenario(
            prediction_store,
            split=PREDICTOR_TRAIN_SPLIT,
            failure_unit_id=unit_id,
            failure_cycle=trajectory_length,
            failure_offset=6,  # not used by env, just for naming of the scenario
            scenario_id="test_no_failure_c6",
        )
        # Override the initial cycle to the verified no-failure start cycle.
        scenario = replace(
            scenario,
            initial_cycles=(start_cycle, 1, 1, 1, 1),
        )
        env, _ = _make_env(prediction_store, scenario, wrap_recording=True)
        recording = env.prediction_store
        env.reset()

        _, _, _, _, step_info = env.step(0)

        assert step_info["num_failures"] == 0, (
            f"expected 0 failures (first failure at c+6), got {step_info['num_failures']}"
        )
        assert step_info["failure_cost"] == 0.0


class TestNoLookupAfterFailure:
    """No PredictionStore request beyond the first failure cycle for that slot."""

    def test_no_request_beyond_failure_cycle(self, prediction_store) -> None:
        """Once true_rul<=0 is requested at the failure cycle, no further cycle is
        requested for that slot's retired unit in the same step."""
        offset = 3
        scenario = _build_failure_offset_scenario(
            prediction_store, offset, "test_no_lookup_after_fail"
        )
        env, _ = _make_env(prediction_store, scenario, wrap_recording=True)
        recording = env.prediction_store
        env.reset()
        recording.clear()

        env.step(0)

        # The failing unit's first requested failure cycle is its trajectory_length.
        fail = find_unit_supporting_failure_offset(
            prediction_store, PREDICTOR_TRAIN_SPLIT, offset
        )
        unit_id, _, trajectory_length = fail
        requested = recording.get_requested_cycles_for_unit(unit_id)
        assert trajectory_length in requested, (
            f"failure cycle {trajectory_length} for unit {unit_id} was never requested"
        )
        # No cycle strictly greater than trajectory_length should be requested.
        beyond = [c for c in requested if c > trajectory_length]
        assert beyond == [], (
            f"requested cycles beyond the failure endpoint ({trajectory_length}): {beyond}"
        )


class TestFailureChargedOnce:
    """Failure is charged exactly once; no double charging."""

    def test_failure_charged_exactly_once(self, prediction_store) -> None:
        scenario = _build_failure_offset_scenario(
            prediction_store, 2, "test_failure_charged_once"
        )
        env, _ = _make_env(prediction_store, scenario)
        env.reset()
        _, _, _, _, step_info = env.step(0)
        assert step_info["num_failures"] == 1
        # Failure cost is exactly c_f (5.0) — not 2*c_f, not some residual.
        assert step_info["failure_cost"] == 5.0
        # Only one corrective replacement happened (slot 0). Other slots did not fail.
        assert step_info["num_preventive"] == 0


class TestCorrectiveReplacementContract:
    """Corrective replacement enters at cycle 1, age 0, does not consume K,
    and produces no residual-cycle advancement."""

    def test_corrective_replacement_cycle_1_age_0(self, prediction_store) -> None:
        scenario = _build_failure_offset_scenario(
            prediction_store, 2, "test_corrective_contract"
        )
        env, _ = _make_env(prediction_store, scenario)
        env.reset()
        _, _, _, _, step_info = env.step(0)

        # Corrective replacement column for slot 0.
        assert "slot_0_diagnostic" in step_info
        slot_info = step_info["slot_0_diagnostic"]
        assert slot_info["cycle"] == 1, (
            f"corrective replacement cycle must be exactly 1, got {slot_info['cycle']}"
        )

        # Confirm via the post-step observation that slot 0's normalized age is 0
        # (cycle 1 -> age 0). We rebuild the obs and inspect by reconstructing
        # through the env's own state.
        slot_state = env._fleet_state.slots[0]
        assert slot_state.cycle == 1
        assert slot_state.age_since_replacement_cycles == 0, (
            f"corrective replacement age must be exactly 0, got "
            f"{slot_state.age_since_replacement_cycles}"
        )

    def test_corrective_replacement_does_not_consume_K(self, prediction_store) -> None:
        """A failing slot replaced correctively must not count as PM and must not
        consume K capacity. If we additionally PM two other slots (action 15 =
        {3,4}), the PM count is 2 (not 3), proving the failure did not consume K."""
        scenario = _build_failure_offset_scenario(
            prediction_store, 2, "test_corrective_no_k"
        )
        env, _ = _make_env(prediction_store, scenario)
        env.reset()
        # action 15 = {3, 4} (two PM slots distinct from slot 0)
        _, _, _, _, step_info = env.step(15)
        # The failure happened on slot 0 (not in {3,4}); K=2 was fully used by PM.
        assert step_info["num_preventive"] == 2, (
            f"expected 2 PM (action {15} -> {{3,4}}), got {step_info['num_preventive']}"
        )
        # Slot 0 failed and was correctively replaced; failure count is 1.
        assert step_info["num_failures"] == 1

    def test_no_residual_cycle_advancement_after_corrective_replacement(
        self, prediction_store
    ) -> None:
        """After corrective replacement the new slot is at cycle 1, not advanced
        through residual cycles of the failed trajectory."""
        scenario = _build_failure_offset_scenario(
            prediction_store, 3, "test_corrective_no_residual"
        )
        env, _ = _make_env(prediction_store, scenario)
        env.reset()
        env.step(0)
        slot_state = env._fleet_state.slots[0]
        # No residual advancement: replaced unit sits exactly at cycle 1, age 0.
        assert slot_state.cycle == 1
        assert slot_state.age_since_replacement_cycles == 0


class TestSimultaneousFailure:
    """Exactly two simultaneous failures in one decision step."""

    def test_two_simultaneous_failures(self, prediction_store) -> None:
        scenario = build_simultaneous_failure_scenario(
            prediction_store,
            split=PREDICTOR_TRAIN_SPLIT,
            scenario_id="test_two_simultaneous",
            failure_offset=3,
        )
        env, _ = _make_env(prediction_store, scenario)
        env.reset()
        _, _, _, _, step_info = env.step(0)
        assert step_info["num_failures"] == 2, (
            f"expected exactly 2 simultaneous failures, got {step_info['num_failures']}"
        )
        assert step_info["failure_cost"] == 10.0, (
            f"expected failure_cost=10.0 (2 * c_f=5), got {step_info['failure_cost']}"
        )
        # No PM in this step.
        assert step_info["num_preventive"] == 0


class TestPMPlusDifferentFailingSlot:
    """One PM slot plus one DIFFERENT non-PM failing slot, in the same step."""

    def test_pm_plus_different_slot_failure(self, prediction_store) -> None:
        scenario = build_mixed_event_scenario(
            prediction_store,
            split=PREDICTOR_TRAIN_SPLIT,
            scenario_id="test_mixed_pm_failure",
            pm_slot=0,
            failure_slot=3,
            failure_offset=3,
        )
        env, _ = _make_env(prediction_store, scenario)
        env.reset()
        # PM the configured pm slot (slot 0) via action 1 = {0}.
        _, _, _, _, step_info = env.step(1)

        # Exactly one PM (slot 0).
        assert step_info["num_preventive"] == 1
        assert step_info["preventive_cost"] == 1.0
        # Exactly one failure on the different slot.
        assert step_info["num_failures"] == 1, (
            f"expected exactly 1 failure on the different slot, got {step_info['num_failures']}"
        )
        assert step_info["failure_cost"] == 5.0
        # Preventive and failure costs are separate components (not summed into one).
        assert step_info["preventive_cost"] > 0.0
        assert step_info["failure_cost"] > 0.0
        assert step_info["preventive_cost"] != step_info["failure_cost"] or (
            step_info["preventive_cost"] == 1.0 and step_info["failure_cost"] == 5.0
        )


class TestDecisionBoundaryViolation:
    """Current decision-boundary true_rul <= 0 raises ContractViolationError."""

    def test_decision_boundary_true_rul_le_0_raises(self, prediction_store) -> None:
        """A slot placed at trajectory_length (true_rul == 0) cannot be stepped:
        the decision-boundary check raises ContractViolationError, not a free replacement.

        Per the contract: do not initialize a scenario directly at true_rul <= 0.
        Instead: (1) construct a valid scenario with slot 0 at an active cycle;
        (2) call env.reset(); (3) after reset, replace only slot 0's internal test
        state so that cycle == trajectory_length and true_rul <= 0; (4) call step(0)
        and assert ContractViolationError.
        """
        # Find a real unit that supports a failure at offset 1
        fail = find_unit_supporting_failure_offset(
            prediction_store, PREDICTOR_TRAIN_SPLIT, 1
        )
        unit_id, start_cycle, trajectory_length = fail

        # Build a VALID scenario where slot 0 starts at start_cycle (true_rul > 0)
        # NOT at trajectory_length (which would be true_rul == 0 and fail reset validation)
        scenario = build_failure_fixture_scenario(
            prediction_store,
            split=PREDICTOR_TRAIN_SPLIT,
            failure_unit_id=unit_id,
            failure_cycle=trajectory_length,
            failure_offset=1,
            scenario_id="test_decision_boundary_violation",
        )
        # Use valid initial_cycles: start_cycle puts unit at true_rul=1 (valid),
        # NOT trajectory_length which would be true_rul=0 (invalid at reset)
        scenario = replace(scenario, initial_cycles=(start_cycle, 1, 1, 1, 1))
        env, _ = _make_env(prediction_store, scenario)

        # Reset with valid state (all slots have true_rul > 0)
        env.reset()

        # After reset, manually force slot 0 to its trajectory_length (true_rul == 0)
        # This simulates reaching the decision boundary without a cached failure
        slot = env._fleet_state.slots[0]
        env._fleet_state.slots[0] = replace(slot, cycle=slot.trajectory_length)

        # step() should detect true_rul <= 0 at decision boundary and raise
        with pytest.raises(ContractViolationError):
            env.step(0)


class TestEndpointWithoutFailureRaises:
    """Endpoint without a true_rul <= 0 record raises ContractViolationError
    (no silent replacement)."""

    def test_advancing_past_endpoint_without_failure_raises(
        self, prediction_store
    ) -> None:
        """If a slot's non-PM advancement would exceed trajectory_length without
        crossing a cached true_rul <= 0, step() raises ContractViolationError
        rather than silently replacing the unit."""
        # Build a scenario where slot 0 is at trajectory_length - 4 with a
        # trajectory such that no true_rul<=0 lies within c+1..c+5. With the cache
        # invariant this cannot occur for real curves (failure always at
        # trajectory_length), so we force a synthetic scenario instead:
        scenario = _build_failure_offset_scenario(
            prediction_store, 5, "test_endpoint_without_failure_raises"
        )
        env, _ = _make_env(prediction_store, scenario)
        env.reset()
        # Force slot 0 to a cycle near the end such that advancing by delta=5
        # would exceed trajectory_length without any cached true_rul<=0 in between.
        # Concretely, replace slot 0's current unit with one whose trajectory is
        # long, then move slot 0 to a cycle where c+5 > trajectory_length and the
        # rows c+1..trajectory_length all have true_rul > 0. That requires a unit
        # whose trajectory_length IS its failure cycle (always), which means c+1
        # ..trajectory_length always has true_rul decreasing to 0 at the end. To
        # construct the exceptional case (no failure before exceeding), we set the
        # slot to c = trajectory_length - 1, then tamper with the cached rows by
        # forcing true_rul positive beyond the endpoint via a recording store that
        # rewrites. That is heavy; instead, exercise the simplest reproducible
        # trigger: set slot 0 cycle to trajectory_length+? is illegal per SlotState.
        #
        # The concrete contract path: place slot 0 at c s.t. c+1..c+5 scan hits
        # trajectory_length (true_rul==0) before exceeding -> a normal failure.
        # To force the ContractViolation branch we need c+5 > trajectory_length AND
        # no true_rul<=0 in c+1..c+4. That is c == trajectory_length - 4: then
        # c+1..c+4 cover trajectory_length-3..trajectory_length and the failure is
        # at trajectory_length (within range), so no violation. Hence the
        # ContractViolation branch is unreachable for real C-MAPSS trajectories
        # without tampering. We test the branch via a recording store that hides
        # the failure-cycle row.
        env.reset()
        slot = env._fleet_state.slots[0]
        original = env.prediction_store
        wrapped = recording_hide_failure_cycle(original, slot.unit_id, slot.trajectory_length)
        env.prediction_store = wrapped

        # Slot 0 at a cycle such that c+5 reaches exactly trajectory_length but the
        # failure-cycle row is now hidden (true_rul never <= 0 in c+1..c+5).
        start = slot.trajectory_length - 5
        env._fleet_state.slots[0] = replace(slot, cycle=start)

        with pytest.raises(ContractViolationError):
            env.step(0)


class TestMissingIntermediatePredictionRaises:
    """A missing intermediate prediction raises MissingPredictionError
    (no silent substitution)."""

    def test_missing_intermediate_prediction_raises(self, prediction_store) -> None:
        scenario = _build_failure_offset_scenario(
            prediction_store, 4, "test_missing_intermediate_raises"
        )
        env, _ = _make_env(prediction_store, scenario)
        env.reset()
        slot = env._fleet_state.slots[0]
        original = env.prediction_store
        # Hide an intermediate cycle c+1 <= hidden <= c+5 so the env cannot look up
        # the required prediction. This must raise MissingPredictionError.
        hidden_cycle = slot.cycle + 2
        wrapped = recording_hide_cycle(original, slot.unit_id, hidden_cycle)
        env.prediction_store = wrapped
        with pytest.raises(MissingPredictionError):
            env.step(0)


class TestNoSilentReplacement:
    """No boundary results in a free (silent) replacement."""

    def test_no_silent_replacement_at_endpoint(self, prediction_store) -> None:
        """When slot 0 is placed such that c+5 lands exactly on trajectory_length,
        the env detects the true_rul<=0 failure (c_f charged) and performs a
        corrective replacement — NOT a silent free replacement with no cost."""
        scenario = _build_failure_offset_scenario(
            prediction_store, 5, "test_no_silent_replacement"
        )
        env, _ = _make_env(prediction_store, scenario)
        env.reset()
        _, _, _, _, step_info = env.step(0)
        # A failure WAS charged at the endpoint (not silently replaced for free).
        assert step_info["num_failures"] == 1
        assert step_info["failure_cost"] == 5.0
        # Replacement happened (corrective) but counted as failure, not PM.
        assert step_info["num_preventive"] == 0


# ----------------------------------------------------------------------
# Test helper wrappers used only by failure-boundary contract tests
# ----------------------------------------------------------------------

class _HideCycleStore:
    """PredictionStore wrapper that forces found=False for a single (unit, cycle)."""

    def __init__(self, wrapped, hide_unit, hide_cycle):
        self._wrapped = wrapped
        self._hide_unit = hide_unit
        self._hide_cycle = hide_cycle

    def get(self, split, unit_id, cycle):
        if unit_id == self._hide_unit and cycle == self._hide_cycle:
            from src.predictors.prediction_store import PredictionResult
            return PredictionResult(found=False)
        return self._wrapped.get(split, unit_id, cycle)

    def get_units(self, split):
        return self._wrapped.get_units(split)


def recording_hide_cycle(wrapped, unit_id, cycle):
    return _HideCycleStore(wrapped, unit_id, cycle)


class _HideFailureCycleStore:
    """PredictionStore wrapper that rewrites the unit's failure-cycle row to remove
    the true_rul<=0 marker, so no cached failure is encountered within c+1..c+offset.

    This makes the smaller-than-trajectory-length rows reveal true_rul<=0 unchanged
    but rewrites the single trajectory_length row to a positive value so the env's
    scan through c+1..c+5 (when c+5 == trajectory_length) finds no failure and then
    must raise ContractViolationError at the exceeding-TL branch.
    """

    def __init__(self, wrapped, hide_unit, hide_cycle, fake_true_rul=1.0):
        self._wrapped = wrapped
        self._hide_unit = hide_unit
        self._hide_cycle = hide_cycle
        self._fake_true_rul = fake_true_rul

    def get(self, split, unit_id, cycle):
        pred = self._wrapped.get(split, unit_id, cycle)
        if not pred.found:
            return pred
        if unit_id == self._hide_unit and cycle == self._hide_cycle:
            # Rewrite true_rul to a positive value so the env's true_rul<=0 check
            # does not fire at the endpoint.
            from src.predictors.prediction_store import PredictionResult
            return PredictionResult(
                found=True,
                split=pred.split,
                unit_id=int(pred.unit_id),
                cycle=int(pred.cycle),
                predicted_rul=pred.predicted_rul,
                predicted_rul_normalized=pred.predicted_rul_normalized,
                true_rul=self._fake_true_rul,
                true_rul_capped=self._fake_true_rul,
                trajectory_length=pred.trajectory_length,
                valid_window=pred.valid_window,
                metadata=pred.metadata,
                cache_version=pred.cache_version,
            )
        return pred

    def get_units(self, split):
        return self._wrapped.get_units(split)


def recording_hide_failure_cycle(wrapped, unit_id, cycle):
    return _HideFailureCycleStore(wrapped, unit_id, cycle)
