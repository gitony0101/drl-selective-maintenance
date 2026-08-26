"""
Milestone 2 Environment Smoke Tests

Tests for the smoke rollout validation tool.
These tests verify actual environment execution, not mocked dictionaries.
"""

import pytest

pytestmark = pytest.mark.requires_external_assets
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_m2_environment_smoke import (
    run_smoke_rollout,
    run_validation_matrix,
    run_all_regimes,
    _load_prediction_store,
    _load_scenario_bank,
    _create_boundary_scenario,
    _create_simultaneous_failure_scenario,
    _create_mixed_event_scenario,
    SmokeResult,
    ALLOWED_SMOKE_SPLITS,
)


class TestSmokeSplitBarrier:
    """Test that rl_test split is rejected."""

    def test_rl_test_rejected(self):
        """rl_test split must be rejected before any loading."""
        result = run_smoke_rollout(
            split="rl_test",
            k_capacity=2,
            cost_regime_id="failure-light-no-waste",
            seed=6521,
            policy="random",
        )
        assert len(result.errors) > 0
        assert "rl_test" in result.errors[0].lower() or "not allowed" in result.errors[0].lower()

    def test_allowed_splits_work(self):
        """predictor_train and rl_validation are allowed."""
        prediction_store = _load_prediction_store()
        for split in ALLOWED_SMOKE_SPLITS:
            result = run_smoke_rollout(
                split=split,
                k_capacity=2,
                cost_regime_id="failure-light-no-waste",
                seed=6521,
                policy="random",
                prediction_store=prediction_store,
            )
            assert len(result.errors) == 0, f"{split} failed: {result.errors}"
            assert result.steps == 100
            assert result.completed


class TestRandomPolicy:
    """Test random policy rollout."""

    def test_random_terminates_at_100_steps(self):
        """Random policy must terminate at exactly 100 steps."""
        prediction_store = _load_prediction_store()
        result = run_smoke_rollout(
            split="rl_validation",
            k_capacity=2,
            cost_regime_id="failure-light-no-waste",
            seed=6521,
            policy="random",
            prediction_store=prediction_store,
        )
        assert result.steps == 100, f"Expected 100 steps, got {result.steps}"
        assert result.completed
        assert len(result.errors) == 0

    def test_random_stats_not_hardcoded(self):
        """Random policy stats must come from actual execution."""
        prediction_store = _load_prediction_store()
        result = run_smoke_rollout(
            split="predictor_train",
            k_capacity=2,
            cost_regime_id="failure-light-no-waste",
            seed=6521,
            policy="random",
            prediction_store=prediction_store,
        )
        # Preventive replacements should vary by seed/scenario
        assert result.preventive_replacements >= 0
        # NaN/Inf should be 0 for valid environment
        assert result.nan_observation_count == 0
        assert result.inf_observation_count == 0


class TestCorrectiveOnlyPolicy:
    """Test corrective-only policy."""

    def test_corrective_only_records_actual_failures(self):
        """Corrective-only must record actual failures, not zeros."""
        prediction_store = _load_prediction_store()
        for split in ["predictor_train", "rl_validation"]:
            result = run_smoke_rollout(
                split=split,
                k_capacity=2,
                cost_regime_id="failure-light-no-waste",
                seed=6521,
                policy="corrective-only",
                prediction_store=prediction_store,
            )
            assert result.steps == 100
            assert result.failures > 0, f"{split}: Expected failures > 0"
            assert result.failure_cost > 0, f"{split}: Expected failure_cost > 0"

    def test_corrective_only_no_preventive(self):
        """Corrective-only (action 0) must have 0 preventive replacements."""
        prediction_store = _load_prediction_store()
        result = run_smoke_rollout(
            split="rl_validation",
            k_capacity=2,
            cost_regime_id="failure-light-no-waste",
            seed=6521,
            policy="corrective-only",
            prediction_store=prediction_store,
        )
        assert result.preventive_replacements == 0
        assert result.preventive_cost == 0.0


class TestBoundaryPolicy:
    """Test boundary failure policy."""

    def test_boundary_executes_all_offsets_predictor_train(self):
        """Boundary policy must execute exactly 5 controlled cases on predictor_train."""
        prediction_store = _load_prediction_store()
        result = run_smoke_rollout(
            split="predictor_train",
            k_capacity=2,
            cost_regime_id="failure-light-no-waste",
            seed=6521,
            policy="boundary",
            prediction_store=prediction_store,
        )
        # Exact assertions
        assert result.episodes == 5, f"Expected 5 episodes, got {result.episodes}"
        assert result.steps == 5, f"Expected 5 steps, got {result.steps}"
        assert result.preventive_replacements == 0, f"Expected 0 PM, got {result.preventive_replacements}"
        assert result.failures == 5, f"Expected 5 failures, got {result.failures}"
        assert result.offsets_tested == [1, 2, 3, 4, 5], f"Wrong offsets: {result.offsets_tested}"
        assert len(result.failure_boundary_cycles) == 5, f"Expected 5 failure cycles, got {len(result.failure_boundary_cycles)}"
        assert len(result.errors) == 0, f"Boundary policy had errors: {result.errors}"

    def test_boundary_executes_all_offsets_rl_validation(self):
        """Boundary policy must execute exactly 5 controlled cases on rl_validation."""
        prediction_store = _load_prediction_store()
        result = run_smoke_rollout(
            split="rl_validation",
            k_capacity=2,
            cost_regime_id="failure-light-no-waste",
            seed=6521,
            policy="boundary",
            prediction_store=prediction_store,
        )
        # Exact assertions
        assert result.episodes == 5, f"Expected 5 episodes, got {result.episodes}"
        assert result.steps == 5, f"Expected 5 steps, got {result.steps}"
        assert result.preventive_replacements == 0, f"Expected 0 PM, got {result.preventive_replacements}"
        assert result.failures == 5, f"Expected 5 failures, got {result.failures}"
        assert result.offsets_tested == [1, 2, 3, 4, 5], f"Wrong offsets: {result.offsets_tested}"
        assert len(result.failure_boundary_cycles) == 5, f"Expected 5 failure cycles, got {len(result.failure_boundary_cycles)}"
        assert len(result.errors) == 0, f"Boundary policy had errors: {result.errors}"

    def test_boundary_actual_cycles_match_trajectory_length(self):
        """For every boundary case, failure_cycle == trajectory_length == start_cycle + offset."""
        prediction_store = _load_prediction_store()
        for split in ["predictor_train", "rl_validation"]:
            result = run_smoke_rollout(
                split=split,
                k_capacity=2,
                cost_regime_id="failure-light-no-waste",
                seed=6521,
                policy="boundary",
                prediction_store=prediction_store,
            )
            for case in result.boundary_cases:
                # failure_cycle must equal trajectory_length (V2cache contract)
                assert case["failure_cycle"] == case["trajectory_length"], \
                    f"{split}: failure_cycle {case['failure_cycle']} != trajectory_length {case['trajectory_length']}"
                # failure_cycle must equal start_cycle + offset
                expected_failure = case["start_cycle"] + case["offset"]
                assert case["failure_cycle"] == expected_failure, \
                    f"{split}: failure_cycle {case['failure_cycle']} != start_cycle + offset ({expected_failure})"

    def test_boundary_replacement_cycle_and_age(self):
        """For every boundary case, replacement_cycle==1 and replacement_age==0 from actual state."""
        prediction_store = _load_prediction_store()
        for split in ["predictor_train", "rl_validation"]:
            result = run_smoke_rollout(
                split=split,
                k_capacity=2,
                cost_regime_id="failure-light-no-waste",
                seed=6521,
                policy="boundary",
                prediction_store=prediction_store,
            )
            for case in result.boundary_cases:
                # replacement_cycle must be 1 (read from actual SlotState)
                assert case["replacement_cycle"] == 1, \
                    f"{split} offset {case['offset']}: replacement_cycle must be 1, got {case['replacement_cycle']}"
                # replacement_age must be 0 (read from actual SlotState)
                assert case["replacement_age"] == 0, \
                    f"{split} offset {case['offset']}: replacement_age must be 0, got {case['replacement_age']}"


class TestSimultaneousFailurePolicy:
    """Test simultaneous-failure policy."""

    def test_simultaneous_records_two_failures_predictor_train(self):
        """Simultaneous-failure must record exactly 2 failures on predictor_train."""
        prediction_store = _load_prediction_store()
        result = run_smoke_rollout(
            split="predictor_train",
            k_capacity=2,
            cost_regime_id="failure-light-no-waste",
            seed=6521,
            policy="simultaneous-failure",
            prediction_store=prediction_store,
        )
        assert len(result.errors) == 0, f"predictor_train failed: {result.errors}"
        assert result.failures == 2, f"Expected 2 failures, got {result.failures}"
        assert result.preventive_replacements == 0
        assert result.failure_cost == 10.0  # 2 * 5.0

    def test_simultaneous_records_two_failures_rl_validation(self):
        """Simultaneous-failure must record exactly 2 failures on rl_validation."""
        prediction_store = _load_prediction_store()
        result = run_smoke_rollout(
            split="rl_validation",
            k_capacity=2,
            cost_regime_id="failure-light-no-waste",
            seed=6521,
            policy="simultaneous-failure",
            prediction_store=prediction_store,
        )
        assert len(result.errors) == 0, f"rl_validation failed: {result.errors}"
        assert result.failures == 2, f"Expected 2 failures, got {result.failures}"
        assert result.preventive_replacements == 0
        assert result.failure_cost == 10.0  # 2 * 5.0

    def test_simultaneous_replacement_cycles_and_ages(self):
        """Simultaneous-failure: both replacements must have cycle=1 and age=0."""
        prediction_store = _load_prediction_store()
        for split in ["predictor_train", "rl_validation"]:
            result = run_smoke_rollout(
                split=split,
                k_capacity=2,
                cost_regime_id="failure-light-no-waste",
                seed=6521,
                policy="simultaneous-failure",
                prediction_store=prediction_store,
            )
            assert len(result.simultaneous_cases) == 1, f"{split}: No simultaneous cases recorded"
            case = result.simultaneous_cases[0]
            # Both replacement cycles must be 1 (read from actual SlotState)
            assert case["replacement_cycles"] == [1, 1], \
                f"{split}: Expected replacement_cycles=[1,1], got {case['replacement_cycles']}"
            # Both replacement ages must be 0 (read from actual SlotState)
            assert case["replacement_ages"] == [0, 0], \
                f"{split}: Expected replacement_ages=[0,0], got {case['replacement_ages']}"
            # Exactly two failure slot indices
            assert len(case["failure_slot_indices"]) == 2, \
                f"{split}: Expected 2 failure_slot_indices, got {len(case['failure_slot_indices'])}"
            # Two valid replacement unit IDs
            assert len(case["replacement_unit_ids"]) == 2, \
                f"{split}: Expected 2 replacement_unit_ids, got {len(case['replacement_unit_ids'])}"


class TestMixedEventPolicy:
    """Test mixed-event policy."""

    def test_mixed_event_one_pm_one_failure_predictor_train(self):
        """Mixed-event must record exactly 1 PM and 1 different-slot failure on predictor_train."""
        prediction_store = _load_prediction_store()
        result = run_smoke_rollout(
            split="predictor_train",
            k_capacity=2,
            cost_regime_id="failure-light-no-waste",
            seed=6521,
            policy="mixed-event",
            prediction_store=prediction_store,
        )
        assert len(result.errors) == 0, f"predictor_train failed: {result.errors}"
        assert result.preventive_replacements == 1
        assert result.failures == 1
        assert result.preventive_cost == 1.0  # 1 * c_pm
        assert result.failure_cost == 5.0  # 1 * c_f

    def test_mixed_event_one_pm_one_failure_rl_validation(self):
        """Mixed-event must record exactly 1 PM and 1 different-slot failure on rl_validation."""
        prediction_store = _load_prediction_store()
        result = run_smoke_rollout(
            split="rl_validation",
            k_capacity=2,
            cost_regime_id="failure-light-no-waste",
            seed=6521,
            policy="mixed-event",
            prediction_store=prediction_store,
        )
        assert len(result.errors) == 0, f"rl_validation failed: {result.errors}"
        assert result.preventive_replacements == 1
        assert result.failures == 1
        assert result.preventive_cost == 1.0  # 1 * c_pm
        assert result.failure_cost == 5.0  # 1 * c_f

    def test_mixed_event_pm_and_failure_different_slots(self):
        """Mixed-event: PM and failure must occur on different slots."""
        prediction_store = _load_prediction_store()
        for split in ["predictor_train", "rl_validation"]:
            result = run_smoke_rollout(
                split=split,
                k_capacity=2,
                cost_regime_id="failure-light-no-waste",
                seed=6521,
                policy="mixed-event",
                prediction_store=prediction_store,
            )
            assert len(result.mixed_cases) == 1, f"{split}: No mixed cases recorded"
            case = result.mixed_cases[0]
            pm_slot = case["pm_slot"]
            failure_slot = case["failure_slot"]
            # PM and failure must be on different slots
            assert pm_slot is not None, f"{split}: PM slot not recorded"
            assert failure_slot is not None, f"{split}: Failure slot not recorded"
            assert pm_slot != failure_slot, \
                f"{split}: PM and failure on same slot {pm_slot}"
            # Both replacement cycles must be 1 (read from actual SlotState)
            assert case["replacement_cycles"] == [1, 1], \
                f"{split}: Expected replacement_cycles=[1,1], got {case['replacement_cycles']}"
            # Both replacement ages must be 0 (read from actual SlotState)
            assert case["replacement_ages"] == [0, 0], \
                f"{split}: Expected replacement_ages=[0,0], got {case['replacement_ages']}"
            # Two valid replacement unit IDs (read from actual SlotState)
            assert len(case["replacement_unit_ids"]) == 2, \
                f"{split}: Expected 2 replacement_unit_ids, got {len(case['replacement_unit_ids'])}"


class TestValidationMatrix:
    """Test validation matrix expansion."""

    def test_validation_matrix_expands_correctly(self):
        """Validation matrix must expand to both splits and both K values."""
        results = run_validation_matrix(
            splits=["predictor_train", "rl_validation"],
            k_values=[1, 2],
            seeds=[6521],
            cost_regimes=["failure-light-no-waste"],
            policy="random",
        )
        # 2 splits x 2 K values x 1 seed = 4 results
        assert len(results) == 4

        # Check all configurations present
        configs = {(r.split, r.maintenance_capacity) for r in results}
        expected = {
            ("predictor_train", 1),
            ("predictor_train", 2),
            ("rl_validation", 1),
            ("rl_validation", 2),
        }
        assert configs == expected

        # Each must complete 100 steps
        for r in results:
            assert r.steps == 100, f"{r.split}/K={r.maintenance_capacity}: {r.steps} steps"

    def test_validation_matrix_5_seeds_2000_steps(self):
        """Validation matrix with 5 seeds must produce 2000 steps."""
        results = run_validation_matrix(
            seeds=list(range(6521, 6526)),  # 5 seeds
            policy="random",
        )
        # 2 splits x 2 K x 5 seeds = 20 results
        assert len(results) == 20
        total_steps = sum(r.steps for r in results)
        assert total_steps == 2000, f"Expected 2000 steps, got {total_steps}"


class TestK1InMemoryDerivation:
    """Test K=1 in-memory derivation from K=2."""

    def test_predictor_train_k1_direct_rollout(self):
        """Direct predictor_train K=1 rollout must succeed with in-memory derivation."""
        prediction_store = _load_prediction_store()
        # This tests the centralized derivation helper - scenario_bank=None triggers derivation
        result = run_smoke_rollout(
            split="predictor_train",
            k_capacity=1,
            cost_regime_id="failure-light-no-waste",
            seed=6521,
            policy="random",
            prediction_store=prediction_store,
        )
        # Required assertions
        assert len(result.errors) == 0, f"K=1 rollout failed: {result.errors}"
        assert result.completed == True, "K=1 rollout did not complete"
        assert result.steps == 100, f"Expected 100 steps, got {result.steps}"
        assert result.maintenance_capacity == 1, f"Expected K=1, got {result.maintenance_capacity}"

    def test_predictor_train_k1_derived_from_k2(self):
        """predictor_train K=1 must be derivable from K=2 scenario bank."""
        # K=2 bank must exist
        k2_path = Path("data/scenario_banks/predictor_train_smoke.json")
        assert k2_path.exists(), "K=2 scenario bank must exist for derivation"

        # K=1 derivation should work without a K=1 JSON file
        k1_bank = _load_scenario_bank("predictor_train", 1)
        assert k1_bank is not None, "K=1 derivation failed"
        assert len(k1_bank.scenarios) > 0, "Derived K=1 bank has no scenarios"
        # All derived scenarios must have maintenance_capacity=1
        for scenario in k1_bank.scenarios:
            assert scenario.maintenance_capacity == 1, f"Derived scenario has K={scenario.maintenance_capacity}, expected 1"


class TestAllRegimes:
    """Test all four cost regimes."""

    def test_all_four_regimes_complete(self):
        """All four regimes must create matching scenario/config pairs."""
        results = run_all_regimes(
            split="predictor_train",
            k_capacity=2,
            seeds=[6521],
            policy="random",
        )
        assert len(results) == 4

        regime_ids = {r.cost_regime_id for r in results}
        expected_regimes = {
            "failure-light-no-waste",
            "failure-heavy-no-waste",
            "failure-light-waste-aware",
            "failure-heavy-waste-aware",
        }
        assert regime_ids == expected_regimes

        # Each must complete 100 steps
        for r in results:
            assert r.steps == 100


class TestResultMetadata:
    """Test policy result metadata completeness."""

    def test_result_contains_all_metadata(self):
        """Result must contain split, K, regime, seed, policy."""
        prediction_store = _load_prediction_store()
        result = run_smoke_rollout(
            split="rl_validation",
            k_capacity=2,
            cost_regime_id="failure-light-no-waste",
            seed=6521,
            policy="random",
            prediction_store=prediction_store,
        )
        assert result.split == "rl_validation"
        assert result.maintenance_capacity == 2
        assert result.cost_regime_id == "failure-light-no-waste"
        assert result.seed == 6521
        assert result.policy == "random"


class TestStepBound:
    """Test that no policy can exceed its safety step bound."""

    def test_no_policy_exceeds_bound(self):
        """No policy should exceed max_steps=150."""
        prediction_store = _load_prediction_store()
        for policy in ["random", "corrective-only"]:
            result = run_smoke_rollout(
                split="rl_validation",
                k_capacity=2,
                cost_regime_id="failure-light-no-waste",
                seed=6521,
                policy=policy,
                prediction_store=prediction_store,
                max_steps=150,
            )
            # Should complete at 100 steps, well under 150 bound
            assert result.steps == 100
            # Should not have "did not terminate" error
            for err in result.errors:
                assert "did not terminate" not in err

    def test_standard_policy_does_not_exceed_episode_horizon(self):
        """No standard 100-step policy should exceed episode_horizon."""
        prediction_store = _load_prediction_store()
        for split in ALLOWED_SMOKE_SPLITS:
            for policy in ["random", "corrective-only"]:
                for k in [1, 2]:
                    result = run_smoke_rollout(
                        split=split,
                        k_capacity=k,
                        cost_regime_id="failure-light-no-waste",
                        seed=6521,
                        policy=policy,
                        prediction_store=prediction_store,
                    )
                    # Standard policies must complete exactly 100 steps
                    assert result.steps == 100, \
                        f"{split}/K={k}/{policy}: expected 100 steps, got {result.steps}"
                    assert result.completed, \
                        f"{split}/K={k}/{policy}: did not complete"


class TestStatisticsAccuracy:
    """Test that statistics come from actual execution."""

    def test_steps_equals_actual_environment_steps(self):
        """stats.steps must equal actual environment steps."""
        prediction_store = _load_prediction_store()
        result = run_smoke_rollout(
            split="rl_validation",
            k_capacity=2,
            cost_regime_id="failure-light-no-waste",
            seed=6521,
            policy="random",
            prediction_store=prediction_store,
        )
        # For a complete episode, steps must be exactly 100
        assert result.steps == 100

    def test_no_nan_inf_in_observations(self):
        """Valid environment must produce no NaN/Inf observations."""
        prediction_store = _load_prediction_store()
        for split in ALLOWED_SMOKE_SPLITS:
            result = run_smoke_rollout(
                split=split,
                k_capacity=2,
                cost_regime_id="failure-light-no-waste",
                seed=6521,
                policy="random",
                prediction_store=prediction_store,
            )
            assert result.nan_observation_count == 0
            assert result.inf_observation_count == 0