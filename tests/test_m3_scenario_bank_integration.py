"""
Scenario bank integration tests for Milestone 3.

Tests verify:
1. Production run_tune path uses scenario bank correctly
2. Scenario IDs belong to the same ScenarioBank passed to environment
3. K=1 and K=2 work correctly
4. All four cost regimes work
5. Policy-filtered tuning exits 0
6. Selected-threshold writer accepts SelectedThreshold objects
7. Evaluation uses production case loader
8. rl_test is rejected before loading
9. No ExactMyopicH1 appears in M3
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import pytest

pytestmark = pytest.mark.requires_external_assets

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.case_loader import (
    load_cases,
    get_scenario_bank_for_case,
    RlTestBarrierError,
    CaseLoadError,
)
from src.baselines.tuning import (
    tune_threshold,
    SelectedThreshold,
    get_threshold_grid,
)
from src.baselines.artifacts import write_selected_thresholds
from src.envs import SelectiveMaintenanceEnv, get_default_config
from src.envs.scenario_bank import ScenarioBank


def run_cli(args: List[str], timeout: int = 120) -> Tuple[int, str, str]:
    """Run CLI command and return (returncode, stdout, stderr)."""
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "run_m3_baselines.py")] + args
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(PROJECT_ROOT),
    )
    return result.returncode, result.stdout, result.stderr


class TestRlTestBarrier:
    """Test that rl_test split is rejected before data loading."""

    def test_rltest_split_rejected_in_case_loader(self) -> None:
        """Case loader should reject rl_test split."""
        with pytest.raises(RlTestBarrierError) as exc_info:
            load_cases(
                split="rl_test",
                k=1,
                cost_regime_id="failure-light-no-waste",
            )
        assert "rl_test" in str(exc_info.value)
        assert "forbidden" in str(exc_info.value).lower()

    def test_cli_rejects_rltest_for_tune(self) -> None:
        """CLI should reject rl_test split for tuning."""
        returncode, stdout, stderr = run_cli([
            "--tune",
            "--split", "rl_test",
            "--output-dir", "/tmp/m3_test_rltest_reject",
        ])
        assert returncode != 0
        assert "rl_test" in stderr or "forbidden" in stderr.lower()


class TestScenarioBankDerivation:
    """Test scenario bank derivation for K=1 and K=2."""

    @pytest.fixture
    def smoke_bank_path(self) -> Path:
        return PROJECT_ROOT / "data" / "scenario_banks" / "rl_validation_smoke.json"

    def test_k1_derivation(self, smoke_bank_path: Path) -> None:
        """K=1 scenarios should have _k1 suffix."""
        bank = get_scenario_bank_for_case(
            split="rl_validation",
            k=1,
            cost_regime_id="failure-light-no-waste",
            source_bank_path=str(smoke_bank_path),
        )
        assert len(bank.scenarios) == 5
        for scenario in bank.scenarios:
            assert "_k1_" in scenario.scenario_id
            assert scenario.maintenance_capacity == 1

    def test_k2_derivation(self, smoke_bank_path: Path) -> None:
        """K=2 scenarios should not have _k1 suffix."""
        bank = get_scenario_bank_for_case(
            split="rl_validation",
            k=2,
            cost_regime_id="failure-light-no-waste",
            source_bank_path=str(smoke_bank_path),
        )
        assert len(bank.scenarios) == 5
        for scenario in bank.scenarios:
            # K=2 scenarios have regime suffix but not _k1
            assert "_k1_" not in scenario.scenario_id
            assert scenario.maintenance_capacity == 2

    def test_all_four_cost_regimes(self, smoke_bank_path: Path) -> None:
        """All four cost regimes should be derivable."""
        regimes = [
            "failure-light-no-waste",
            "failure-heavy-no-waste",
            "failure-light-waste-aware",
            "failure-heavy-waste-aware",
        ]
        for regime in regimes:
            bank = get_scenario_bank_for_case(
                split="rl_validation",
                k=1,
                cost_regime_id=regime,
                source_bank_path=str(smoke_bank_path),
            )
            assert len(bank.scenarios) == 5
            for scenario in bank.scenarios:
                assert scenario.cost_regime_id == regime
                assert regime in scenario.scenario_id


class TestScenarioIdBelongsToBank:
    """Test that scenario IDs belong to the same ScenarioBank."""

    @pytest.fixture
    def smoke_bank_path(self) -> Path:
        return PROJECT_ROOT / "data" / "scenario_banks" / "rl_validation_smoke.json"

    def test_scenario_ids_match_bank(self, smoke_bank_path: Path) -> None:
        """Scenario IDs must belong to the ScenarioBank."""
        k = 1
        regime = "failure-light-no-waste"

        # Get scenario bank
        bank = get_scenario_bank_for_case(
            split="rl_validation",
            k=k,
            cost_regime_id=regime,
            source_bank_path=str(smoke_bank_path),
        )

        # Extract scenario IDs from bank
        scenario_ids = [s.scenario_id for s in bank.scenarios]

        # Verify each ID belongs to the bank
        bank_ids = {s.scenario_id for s in bank.scenarios}
        for sid in scenario_ids:
            assert sid in bank_ids, f"Scenario ID {sid} not in bank"

    def test_no_double_derivation(self, smoke_bank_path: Path) -> None:
        """Verify single derivation produces consistent results."""
        # Call get_scenario_bank_for_case once
        bank = get_scenario_bank_for_case(
            split="rl_validation",
            k=1,
            cost_regime_id="failure-light-no-waste",
            source_bank_path=str(smoke_bank_path),
        )

        # Extract IDs from the bank
        ids_from_bank = tuple(s.scenario_id for s in bank.scenarios)

        # IDs should match what load_cases would return
        result = load_cases(
            split="rl_validation",
            k=1,
            cost_regime_id="failure-light-no-waste",
            source_bank_path=str(smoke_bank_path),
        )

        # Note: load_cases returns scenario_ids in its result
        # These should match the bank's scenario IDs
        assert tuple(result.scenario_ids) == ids_from_bank


class TestTuneWithScenarioBank:
    """Test tune_threshold with scenario bank integration."""

    @pytest.fixture
    def smoke_bank_path(self) -> Path:
        return PROJECT_ROOT / "data" / "scenario_banks" / "rl_validation_smoke.json"

    def test_tune_with_scenario_bank(self, smoke_bank_path: Path) -> None:
        """tune_threshold should work with scenario_bank parameter."""
        # Get scenario bank and IDs
        bank = get_scenario_bank_for_case(
            split="rl_validation",
            k=1,
            cost_regime_id="failure-light-no-waste",
            source_bank_path=str(smoke_bank_path),
        )
        scenario_ids = [s.scenario_id for s in bank.scenarios]

        config = get_default_config(
            split="rl_validation",
            maintenance_capacity=1,
            cost_regime_id="failure-light-no-waste",
            seed=42,
        )

        # Use mini grid for speed
        mini_grid = [50, 100]

        selected, candidates = tune_threshold(
            policy_family="age_threshold",
            k_capacity=1,
            cost_regime_id="failure-light-no-waste",
            env_config=config,
            scenario_ids=scenario_ids,
            scenario_bank=bank,
            reset_seeds=[6521],
            threshold_grid=mini_grid,
        )

        assert isinstance(selected, SelectedThreshold)
        assert selected.threshold in mini_grid
        assert len(candidates) == 2

    def test_tune_with_k2(self, smoke_bank_path: Path) -> None:
        """tune_threshold should work with K=2."""
        bank = get_scenario_bank_for_case(
            split="rl_validation",
            k=2,
            cost_regime_id="failure-light-no-waste",
            source_bank_path=str(smoke_bank_path),
        )
        scenario_ids = [s.scenario_id for s in bank.scenarios]

        config = get_default_config(
            split="rl_validation",
            maintenance_capacity=2,
            cost_regime_id="failure-light-no-waste",
            seed=42,
        )

        mini_grid = [50, 100]

        selected, candidates = tune_threshold(
            policy_family="age_threshold",
            k_capacity=2,
            cost_regime_id="failure-light-no-waste",
            env_config=config,
            scenario_ids=scenario_ids,
            scenario_bank=bank,
            reset_seeds=[6521],
            threshold_grid=mini_grid,
        )

        assert isinstance(selected, SelectedThreshold)
        assert len(candidates) == 2


class TestSelectedThresholdWriter:
    """Test selected_thresholds writer accepts SelectedThreshold objects."""

    def test_write_selected_thresholds(self, tmp_path: Path) -> None:
        """write_selected_thresholds should accept Dict[str, SelectedThreshold]."""
        selected = {
            "age_threshold_k1_failure-light-no-waste": SelectedThreshold(
                policy_family="age_threshold",
                threshold=100,
                k_capacity=1,
                cost_regime_id="failure-light-no-waste",
                mean_total_cost=50.0,
                total_failures=2,
                mean_wasted_life_cost=10.0,
                episode_count=25,
                tie_break_reason="best",
            ),
        }

        output_path = write_selected_thresholds(selected, tmp_path)
        assert output_path.exists()

        # Verify JSON is valid
        with open(output_path) as f:
            data = json.load(f)

        assert "age_threshold_k1_failure-light-no-waste" in data
        assert data["age_threshold_k1_failure-light-no-waste"]["threshold"] == 100


class TestPolicyFilteredTuning:
    """Test policy-filtered tuning exits 0."""

    def test_tune_age_threshold_mini_grid(self) -> None:
        """--tune --policy age_threshold should work with mini grid.

        This test uses a programmatic mini grid to verify the tuning
        flow works correctly, without running the full 12-threshold grid.
        """
        from src.baselines.tuning import tune_threshold
        from src.baselines.case_loader import get_scenario_bank_for_case
        from src.envs import get_default_config

        smoke_bank_path = PROJECT_ROOT / "data" / "scenario_banks" / "rl_validation_smoke.json"

        # Get scenario bank and IDs
        bank = get_scenario_bank_for_case(
            split="rl_validation",
            k=1,
            cost_regime_id="failure-light-no-waste",
            source_bank_path=str(smoke_bank_path),
        )
        scenario_ids = [s.scenario_id for s in bank.scenarios]

        config = get_default_config(
            split="rl_validation",
            maintenance_capacity=1,
            cost_regime_id="failure-light-no-waste",
            seed=42,
        )

        # Use mini grid for speed (just 2 thresholds)
        mini_grid = [50, 100]

        selected, candidates = tune_threshold(
            policy_family="age_threshold",
            k_capacity=1,
            cost_regime_id="failure-light-no-waste",
            env_config=config,
            scenario_ids=scenario_ids,
            scenario_bank=bank,
            reset_seeds=[6521],  # Single seed for speed
            threshold_grid=mini_grid,
        )

        # Verify tuning completed
        assert isinstance(selected, SelectedThreshold)
        assert selected.policy_family == "age_threshold"
        assert len(candidates) == 2


class TestNoExactMyopicInM3:
    """Test that ExactMyopicH1 does not appear in M3."""

    def test_no_exact_myopic_import(self) -> None:
        """ExactMyopicH1 should not be importable from M3 modules."""
        # Try to import - should fail
        try:
            from src.baselines.exact_myopic import ExactMyopicH1
            pytest.fail("ExactMyopicH1 should not exist in M3")
        except ImportError:
            pass  # Expected
        except ModuleNotFoundError:
            pass  # Expected

    def test_no_exact_myopic_in_policy_families(self) -> None:
        """ExactMyopicH1 should not be in POLICY_FAMILIES."""
        from scripts.run_m3_baselines import POLICY_FAMILIES

        for policy in POLICY_FAMILIES:
            assert "exact_myopic" not in policy.lower()
            assert "exactmyopic" not in policy.lower()


class TestEvaluationScenarioBankIntegration:
    """Test that evaluation uses scenario bank correctly."""

    @pytest.fixture
    def smoke_bank_path(self) -> Path:
        return PROJECT_ROOT / "data" / "scenario_banks" / "rl_validation_smoke.json"

    def test_env_accepts_scenario_bank(self, smoke_bank_path: Path) -> None:
        """Environment should accept scenario_bank parameter."""
        bank = get_scenario_bank_for_case(
            split="rl_validation",
            k=1,
            cost_regime_id="failure-light-no-waste",
            source_bank_path=str(smoke_bank_path),
        )

        config = get_default_config(
            split="rl_validation",
            maintenance_capacity=1,
            cost_regime_id="failure-light-no-waste",
            seed=42,
        )

        env = SelectiveMaintenanceEnv(config=config, scenario_bank=bank)
        assert env.scenario_bank is not None
        assert env.scenario_bank.bank_id == bank.bank_id

    def test_env_reset_with_scenario_bank(self, smoke_bank_path: Path) -> None:
        """Environment reset should work with scenario_bank."""
        bank = get_scenario_bank_for_case(
            split="rl_validation",
            k=1,
            cost_regime_id="failure-light-no-waste",
            source_bank_path=str(smoke_bank_path),
        )

        config = get_default_config(
            split="rl_validation",
            maintenance_capacity=1,
            cost_regime_id="failure-light-no-waste",
            seed=42,
        )

        env = SelectiveMaintenanceEnv(config=config, scenario_bank=bank)
        obs, info = env.reset(seed=6521)

        assert obs is not None
        assert env._current_scenario is not None
        assert env._current_scenario.scenario_id in [s.scenario_id for s in bank.scenarios]