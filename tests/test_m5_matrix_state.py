"""
Focused M5 Tests: Experiment Matrix State Machine
"""

import pytest

pytestmark = pytest.mark.requires_external_assets
import json
import os
import sys
import subprocess
import tempfile
import shutil
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.agents.ddqn.agent import DDQNAgent, DDQNAgentConfig
from src.agents.ddqn.checkpoint import (
    save_checkpoint, CheckpointSelectionState, CHECKPOINT_SELECTION_STATE_VERSION,
    CHECKPOINT_SCHEMA_VERSION, compute_action_table_hash
)
from src.envs.action_table import ACTION_TABLE_N5_K1


def _create_valid_checkpoint(ckpt_path: Path):
    """Create a valid schema-v6 checkpoint for testing."""
    agent = DDQNAgent(config=DDQNAgentConfig(num_actions=6), seed=6521)
    save_checkpoint(
        agent=agent,
        config={"hidden_dim": 128, "num_hidden_layers": 2, "num_actions": 6},
        output_path=ckpt_path,
        maintenance_capacity=1,
        action_table=ACTION_TABLE_N5_K1,
        cost_regime_id="failure-light-no-waste",
        training_seed=6521,
        training_split="predictor_train",
        validation_split="rl_validation",
        training_scenario_bank_path="configs/scenarios/m5_pilot_k1.json",
        validation_scenario_bank_path="configs/scenarios/m5_validation_k1.json",
        prediction_cache_manifest_path="data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json",
        selection_state=CheckpointSelectionState(
            selection_state_version=CHECKPOINT_SELECTION_STATE_VERSION,
            validation_performed=True,
            best_validation_mean_cost=10.0,
            best_checkpoint_global_step=100000,
            best_checkpoint_artifact_name="checkpoint_best.pt",
            best_validation_failure_count=0,
            best_validation_worst_10_pct_cost=20.0,
            comparator_identity="mean_cost_v1",
            equal_metric_tie_behavior="keep_first",
        ),
    )


class TestMatrixGeneratorDryRun:
    """Test matrix generator dry-run mode."""

    def test_dry_run_produces_40_runs(self):
        """Test dry-run produces exactly 40 unique combinations."""
        result = subprocess.run(
            ["python", "scripts/generate_m5_matrix.py", "--dry-run", "--validate-configs"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0
        assert "Total runs: 40" in result.stdout
        assert "Expected total: 40" in result.stdout

    def test_dry_run_no_checkpoints_created(self):
        """Test dry-run creates no checkpoint files."""
        result = subprocess.run(
            ["python", "scripts/generate_m5_matrix.py", "--dry-run"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0
        assert "Dry-run complete. No training executed." in result.stdout


class TestMatrixGeneratorManifest:
    """Test matrix generator manifest output."""

    @pytest.fixture
    def temp_manifest(self):
        """Create temporary manifest path."""
        d = tempfile.mkdtemp()
        manifest_path = Path(d) / "test_manifest.json"
        yield manifest_path
        shutil.rmtree(d)

    def test_manifest_written(self, temp_manifest):
        """Test manifest is written to specified path.

        Note: this regression test exercises the EMPTY state-machine branch
        (all 40 rows NOT_STARTED).  Because the shared ``results/milestone5``
        tree on disk carries historical runs that the new regime-bank matrix
        state machine correctly identifies as COMPLETE, the empty-branch
        assertion is only valid against a temporary, isolated output root.
        We achieve isolation by exporting M5_MATRIX_OUTPUT_BASE to a new
        tempdir for just this subprocess.
        """
        d = tempfile.mkdtemp()
        isolated_output_root = Path(d) / "empty_runs"
        isolated_output_root.mkdir(parents=True)
        env = dict(os.environ)
        env["M5_MATRIX_OUTPUT_BASE"] = str(isolated_output_root)
        try:
            result = subprocess.run(
                [
                    "python", "scripts/generate_m5_matrix.py",
                    "--dry-run",
                    "--output-manifest", str(temp_manifest),
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
                env=env,
            )
            assert result.returncode == 0
            assert temp_manifest.exists()

            with open(temp_manifest) as f:
                manifest = json.load(f)

            assert manifest["total_runs"] == 40
            assert "runs" in manifest
            assert "state_counts" in manifest
            assert manifest["state_counts"]["NOT_STARTED"] == 40
        finally:
            shutil.rmtree(d)

    def test_manifest_run_spec_structure(self, temp_manifest):
        """Test manifest run specifications have required fields."""
        result = subprocess.run(
            [
                "python", "scripts/generate_m5_matrix.py",
                "--dry-run",
                "--output-manifest", str(temp_manifest),
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0

        with open(temp_manifest) as f:
            manifest = json.load(f)

        required_fields = [
            "run_id", "k", "cost_regime", "seed", "state",
            "output_dir", "training_scenario_bank_path",
            "validation_scenario_bank_path", "split", "validation_split", "command"
        ]

        for run in manifest["runs"]:
            for field in required_fields:
                assert field in run, f"Missing field: {field}"


class TestMatrixGeneratorValidation:
    """Test matrix generator validation."""

    def test_validate_configs_passes(self):
        """Test --validate-configs passes for valid configs."""
        result = subprocess.run(
            ["python", "scripts/generate_m5_matrix.py", "--dry-run", "--validate-configs"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0
        assert "Validation PASSED" in result.stdout

    def test_validates_training_bank_split(self):
        """Test validation verifies training bank split is predictor_train."""
        # The pilot configs should have predictor_train split
        with open("configs/scenarios/m5_pilot_k1.json") as f:
            pilot = json.load(f)
        assert pilot["split"] == "predictor_train"

        with open("configs/scenarios/m5_pilot_k2.json") as f:
            pilot = json.load(f)
        assert pilot["split"] == "predictor_train"

    def test_validates_validation_bank_split(self):
        """Test validation verifies validation bank split is rl_validation."""
        with open("configs/scenarios/m5_validation_k1.json") as f:
            val = json.load(f)
        assert val["split"] == "rl_validation"

        with open("configs/scenarios/m5_validation_k2.json") as f:
            val = json.load(f)
        assert val["split"] == "rl_validation"


class TestMatrixGeneratorRlTestBarrier:
    """Test matrix generator rl_test barrier."""

    def test_never_generates_rl_test(self):
        """Test matrix generator never generates rl_test runs."""
        result = subprocess.run(
            ["python", "scripts/generate_m5_matrix.py", "--dry-run"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0

        # Parse manifest
        manifest_path = Path("results/milestone5/experiment_matrix.json")
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)

            for run in manifest["runs"]:
                assert run["split"] != "rl_test"
                assert run["validation_split"] != "rl_test"
                assert "rl_test" not in run["run_id"]


class TestStrictRunCompletionState:
    """Test strict 11-condition run completion validation."""

    @pytest.fixture
    def temp_run_dir(self):
        """Create temporary run directory."""
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d)

    def _determine_state_strict(self, run_dir: Path, expected_run_id: str = "test_run",
                                 expected_k: int = 1, expected_regime: str = "failure-light-no-waste",
                                 expected_seed: int = 6521) -> tuple:
        """Helper using strict determine_run_state from script."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from generate_m5_matrix import determine_run_state

        state, checkpoint, issues = determine_run_state(
            run_dir.name,  # Use just the run_id part
            expected_k=expected_k,
            expected_cost_regime=expected_regime,
            expected_seed=expected_seed,
            expected_max_steps=100000,
        )
        return state.value, checkpoint, issues

    def test_not_started_no_artifacts(self, temp_run_dir):
        """Test NOT_STARTED when no artifacts exist."""
        # Create the run directory under OUTPUT_BASE
        run_dir = temp_run_dir / "test_run"
        run_dir.mkdir(parents=True, exist_ok=True)

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from generate_m5_matrix import determine_run_state, OUTPUT_BASE

        # Temporarily change OUTPUT_BASE
        original_base = OUTPUT_BASE
        import generate_m5_matrix
        generate_m5_matrix.OUTPUT_BASE = str(temp_run_dir)

        state, checkpoint, issues = determine_run_state(
            "test_run",
            expected_k=1,
            expected_cost_regime="failure-light-no-waste",
            expected_seed=6521,
        )

        generate_m5_matrix.OUTPUT_BASE = original_base

        assert state.value == "NOT_STARTED"
        assert checkpoint is None
        assert len(issues) > 0  # Should report why

    def test_incomplete_checkpoint_no_manifest(self, temp_run_dir):
        """Test INCOMPLETE when checkpoint exists but no manifest."""
        run_dir = temp_run_dir / "test_run"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create checkpoint_latest.pt only
        checkpoint = run_dir / "checkpoint_latest.pt"
        checkpoint.touch()

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        import generate_m5_matrix
        original_base = generate_m5_matrix.OUTPUT_BASE
        generate_m5_matrix.OUTPUT_BASE = str(temp_run_dir)

        from generate_m5_matrix import determine_run_state

        state, checkpoint_path, issues = determine_run_state(
            "test_run",
            expected_k=1,
            expected_cost_regime="failure-light-no-waste",
            expected_seed=6521,
        )

        generate_m5_matrix.OUTPUT_BASE = original_base

        assert state.value == "INCOMPLETE"
        assert checkpoint_path is not None
        assert "no valid run_manifest.json" in issues

    def test_complete_requires_all_conditions(self, temp_run_dir):
        """Test COMPLETE requires all 11 conditions satisfied."""
        run_dir = temp_run_dir / "test_run"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create required artifacts with valid checkpoint
        _create_valid_checkpoint(run_dir / "checkpoint_latest.pt")
        (run_dir / "checkpoint_best.pt").touch()
        (run_dir / "training_metrics.jsonl").touch()
        (run_dir / "validation_metrics.json").touch()

        # Create valid run_manifest.json with all required fields
        manifest = {
            "run_id": "test_run",
            "status": "COMPLETE",
            "maintenance_capacity": 1,
            "cost_regime_id": "failure-light-no-waste",
            "training_seed": 6521,
            "final_global_step": 100000,
            "max_steps": 100000,
            "training_split": "predictor_train",
            "validation_split": "rl_validation",
            "checkpoint_schema_version": 6,
            "git_commit": "a" * 40,
            "best_validation_mean_cost": 10.0,
            "validation_results": [{"mean_total_cost": 10.0}],
            "validation_performed": True,
            "resolved_config_identity": "a" * 64,
        }

        with open(run_dir / "run_manifest.json", "w") as f:
            json.dump(manifest, f)

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        import generate_m5_matrix
        original_base = generate_m5_matrix.OUTPUT_BASE
        generate_m5_matrix.OUTPUT_BASE = str(temp_run_dir)

        from generate_m5_matrix import determine_run_state

        state, checkpoint_path, issues = determine_run_state(
            "test_run",
            expected_k=1,
            expected_cost_regime="failure-light-no-waste",
            expected_seed=6521,
        )

        generate_m5_matrix.OUTPUT_BASE = original_base

        assert state.value == "COMPLETE"
        assert len(issues) == 0

    def test_incomplete_wrong_run_id(self, temp_run_dir):
        """Test INCOMPLETE when manifest run_id doesn't match expected."""
        run_dir = temp_run_dir / "test_run"
        run_dir.mkdir(parents=True, exist_ok=True)

        (run_dir / "checkpoint_latest.pt").touch()

        manifest = {
            "run_id": "wrong_run_id",  # Mismatch
            "status": "COMPLETE",
            "config": {
                "maintenance_capacity": 1,
                "cost_regime_id": "failure-light-no-waste",
                "training_seed": 6521,
            },
            "final_metrics": {"final_global_step": 100000},
        }

        with open(run_dir / "run_manifest.json", "w") as f:
            json.dump(manifest, f)

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        import generate_m5_matrix
        original_base = generate_m5_matrix.OUTPUT_BASE
        generate_m5_matrix.OUTPUT_BASE = str(temp_run_dir)

        from generate_m5_matrix import determine_run_state

        state, checkpoint_path, issues = determine_run_state(
            "test_run",
            expected_k=1,
            expected_cost_regime="failure-light-no-waste",
            expected_seed=6521,
        )

        generate_m5_matrix.OUTPUT_BASE = original_base

        assert state.value == "INCOMPLETE"
        assert any("run_id mismatch" in issue for issue in issues)

    def test_incomplete_wrong_k(self, temp_run_dir):
        """Test INCOMPLETE when manifest K doesn't match expected."""
        run_dir = temp_run_dir / "test_run"
        run_dir.mkdir(parents=True, exist_ok=True)

        (run_dir / "checkpoint_latest.pt").touch()

        manifest = {
            "run_id": "test_run",
            "status": "COMPLETE",
            "config": {
                "maintenance_capacity": 2,  # Wrong K
                "cost_regime_id": "failure-light-no-waste",
                "training_seed": 6521,
            },
            "final_metrics": {"final_global_step": 100000},
        }

        with open(run_dir / "run_manifest.json", "w") as f:
            json.dump(manifest, f)

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        import generate_m5_matrix
        original_base = generate_m5_matrix.OUTPUT_BASE
        generate_m5_matrix.OUTPUT_BASE = str(temp_run_dir)

        from generate_m5_matrix import determine_run_state

        state, checkpoint_path, issues = determine_run_state(
            "test_run",
            expected_k=1,
            expected_cost_regime="failure-light-no-waste",
            expected_seed=6521,
        )

        generate_m5_matrix.OUTPUT_BASE = original_base

        assert state.value == "INCOMPLETE"
        assert any("K mismatch" in issue for issue in issues)

    def test_incomplete_insufficient_global_step(self, temp_run_dir):
        """Test INCOMPLETE when global_step < max_steps."""
        run_dir = temp_run_dir / "test_run"
        run_dir.mkdir(parents=True, exist_ok=True)

        (run_dir / "checkpoint_latest.pt").touch()
        (run_dir / "checkpoint_best.pt").touch()
        (run_dir / "training_metrics.jsonl").touch()
        (run_dir / "validation_metrics.json").touch()

        manifest = {
            "run_id": "test_run",
            "status": "COMPLETE",
            "config": {
                "maintenance_capacity": 1,
                "cost_regime_id": "failure-light-no-waste",
                "training_seed": 6521,
            },
            "final_metrics": {
                "final_global_step": 50000,  # Only half the required steps
                "best_validation_mean_cost": 10.0,
            },
            "validation_results": [{"mean_total_cost": 10.0}],
        }

        with open(run_dir / "run_manifest.json", "w") as f:
            json.dump(manifest, f)

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        import generate_m5_matrix
        original_base = generate_m5_matrix.OUTPUT_BASE
        generate_m5_matrix.OUTPUT_BASE = str(temp_run_dir)

        from generate_m5_matrix import determine_run_state

        state, checkpoint_path, issues = determine_run_state(
            "test_run",
            expected_k=1,
            expected_cost_regime="failure-light-no-waste",
            expected_seed=6521,
            expected_max_steps=100000,
        )

        generate_m5_matrix.OUTPUT_BASE = original_base

        assert state.value == "INCOMPLETE"
        assert any("global_step" in issue for issue in issues)

    def test_incomplete_final_metrics_without_status(self, temp_run_dir):
        """Test that final_metrics alone cannot mark a run as COMPLETE.

        This proves the strict completion-state reader requires an explicit
        'status: COMPLETE' or 'status: SUCCESS' field - final_metrics alone
        is NOT sufficient.
        """
        run_dir = temp_run_dir / "test_run"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create all artifacts including checkpoint
        _create_valid_checkpoint(run_dir / "checkpoint_latest.pt")
        (run_dir / "checkpoint_best.pt").touch()
        (run_dir / "training_metrics.jsonl").touch()
        (run_dir / "validation_metrics.json").touch()

        # Create manifest with final_metrics but NO explicit status
        manifest = {
            "run_id": "test_run",
            # NOTE: No "status" field - this should NOT be COMPLETE
            "config": {
                "maintenance_capacity": 1,
                "cost_regime_id": "failure-light-no-waste",
                "training_seed": 6521,
            },
            "final_metrics": {
                "final_global_step": 100000,
                "best_validation_mean_cost": 10.0,
            },
            "validation_results": [{"mean_total_cost": 10.0}],
            "checkpoint_schema_version": 6,
            "git_commit": "a" * 40,
            "resolved_config_identity": "a" * 64,
        }

        with open(run_dir / "run_manifest.json", "w") as f:
            json.dump(manifest, f)

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        import generate_m5_matrix
        original_base = generate_m5_matrix.OUTPUT_BASE
        generate_m5_matrix.OUTPUT_BASE = str(temp_run_dir)

        from generate_m5_matrix import determine_run_state

        state, checkpoint_path, issues = determine_run_state(
            "test_run",
            expected_k=1,
            expected_cost_regime="failure-light-no-waste",
            expected_seed=6521,
        )

        generate_m5_matrix.OUTPUT_BASE = original_base

        # PROOF: final_metrics alone cannot make a run COMPLETE
        assert state.value == "INCOMPLETE", \
            "Run with final_metrics but no explicit status should be INCOMPLETE"
        assert checkpoint_path is not None
        assert any("status not COMPLETE/SUCCESS" in issue for issue in issues)

    def test_complete_explicit_status_required(self, temp_run_dir):
        """Test that an explicit COMPLETE status makes a run COMPLETE.

        This proves the writer-to-reader contract: an explicit status field
        is required for completion.
        """
        run_dir = temp_run_dir / "test_run"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create all required artifacts with valid checkpoint
        _create_valid_checkpoint(run_dir / "checkpoint_latest.pt")
        (run_dir / "checkpoint_best.pt").touch()
        (run_dir / "training_metrics.jsonl").touch()
        (run_dir / "validation_metrics.json").touch()

        # Create manifest with EXPLICIT status: COMPLETE
        manifest = {
            "run_id": "test_run",
            "status": "COMPLETE",  # Explicit status required
            "maintenance_capacity": 1,
            "cost_regime_id": "failure-light-no-waste",
            "training_seed": 6521,
            "final_global_step": 100000,
            "max_steps": 100000,
            "training_split": "predictor_train",
            "validation_split": "rl_validation",
            "checkpoint_schema_version": 6,
            "git_commit": "a" * 40,  # Valid 40-char hex
            "best_validation_mean_cost": 10.0,
            "validation_results": [{"mean_total_cost": 10.0}],
            "validation_performed": True,
            "resolved_config_identity": "a" * 64,  # Valid 64-char hex
        }

        with open(run_dir / "run_manifest.json", "w") as f:
            json.dump(manifest, f)

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        import generate_m5_matrix
        original_base = generate_m5_matrix.OUTPUT_BASE
        generate_m5_matrix.OUTPUT_BASE = str(temp_run_dir)

        from generate_m5_matrix import determine_run_state

        state, checkpoint_path, issues = determine_run_state(
            "test_run",
            expected_k=1,
            expected_cost_regime="failure-light-no-waste",
            expected_seed=6521,
        )

        generate_m5_matrix.OUTPUT_BASE = original_base

        # PROOF: explicit COMPLETE status works
        assert state.value == "COMPLETE"
        assert len(issues) == 0
        assert checkpoint_path is not None

        generate_m5_matrix.OUTPUT_BASE = original_base

        # PROOF: explicit COMPLETE status works
        assert state.value == "COMPLETE"
        assert len(issues) == 0
        assert checkpoint_path is not None


class TestResumeIncomplete:
    """Test --resume-incomplete flag behavior."""

    def test_resume_incomplete_shows_only_incomplete(self):
        """Test --resume-incomplete only shows INCOMPLETE runs."""
        # Since no runs exist yet, should show 0 runs
        result = subprocess.run(
            ["python", "scripts/generate_m5_matrix.py", "--dry-run", "--resume-incomplete"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0
        # Should show 0 total runs since none are INCOMPLETE
        assert "Total runs: 0" in result.stdout


class TestSkipCompleted:
    """Test --skip-completed flag behavior."""

    def test_skip_completed_shows_all_when_none_complete(self):
        """Test --skip-completed shows all runs when none are complete.

        This regression test exercises the EMPTY state-machine branch under
        --skip-completed.  We isolate the output root via M5_MATRIX_OUTPUT_BASE
        so the shared in-repo results tree (which carries historical COMPLETE
        runs) cannot inadvertently shorten the skip list.
        """
        d = tempfile.mkdtemp()
        isolated_output_root = Path(d) / "empty_runs"
        isolated_output_root.mkdir(parents=True)
        env = dict(os.environ)
        env["M5_MATRIX_OUTPUT_BASE"] = str(isolated_output_root)
        try:
            result = subprocess.run(
                ["python", "scripts/generate_m5_matrix.py", "--dry-run", "--skip-completed"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
                env=env,
            )
            assert result.returncode == 0
            assert "Total runs: 40" in result.stdout
        finally:
            shutil.rmtree(d)


class TestM5ProductionPathStrictCompletionContract:
    """M5 training path strict COMPLETE contract: every condition is required."""

    @pytest.fixture
    def temp_run_dir(self):
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d)

    def _setup(self, temp_run_dir, manifest_overrides=None):
        # Required artifacts present
        run_dir = temp_run_dir / "test_run"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create valid checkpoint
        _create_valid_checkpoint(run_dir / "checkpoint_latest.pt")
        (run_dir / "checkpoint_best.pt").touch()
        (run_dir / "training_metrics.jsonl").touch()
        (run_dir / "validation_metrics.json").touch()

        manifest = {
            "run_id": "test_run",
            "status": "COMPLETE",
            "maintenance_capacity": 1,
            "cost_regime_id": "failure-light-no-waste",
            "training_seed": 6521,
            "final_global_step": 100000,
            "max_steps": 100000,
            "training_split": "predictor_train",
            "validation_split": "rl_validation",
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "git_commit": "a" * 40,  # Valid 40-char hex
            "best_validation_mean_cost": 10.0,
            "validation_results": [{"mean_total_cost": 10.0}],
            "validation_performed": True,
            "resolved_config_identity": "a" * 64,  # Valid 64-char hex
        }
        if manifest_overrides:
            manifest.update(manifest_overrides)
        with open(run_dir / "run_manifest.json", "w") as f:
            json.dump(manifest, f)
        return run_dir

    def _call(self, temp_run_dir):
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        import generate_m5_matrix
        original_base = generate_m5_matrix.OUTPUT_BASE
        generate_m5_matrix.OUTPUT_BASE = str(temp_run_dir)
        from generate_m5_matrix import determine_run_state
        state, ckpt, issues = determine_run_state(
            "test_run",
            expected_k=1,
            expected_cost_regime="failure-light-no-waste",
            expected_seed=6521,
            expected_max_steps=100000,
        )
        generate_m5_matrix.OUTPUT_BASE = original_base
        return state.value, ckpt, issues

    def test_complete_when_strict_conditions_satisfied(self, temp_run_dir):
        self._setup(temp_run_dir)
        state, _, issues = self._call(temp_run_dir)
        assert state == "COMPLETE", f"expected COMPLETE got {state} issues={issues}"

    def test_incomplete_when_status_not_complete_or_success(self, temp_run_dir):
        self._setup(temp_run_dir, {"status": "RUNNING"})
        state, _, issues = self._call(temp_run_dir)
        assert state == "INCOMPLETE"

    def test_incomplete_when_run_id_mismatch(self, temp_run_dir):
        self._setup(temp_run_dir, {"run_id": "different"})
        state, _, issues = self._call(temp_run_dir)
        assert state == "INCOMPLETE"
        assert any("run_id mismatch" in i for i in issues)

    def test_incomplete_when_k_mismatch(self, temp_run_dir):
        self._setup(temp_run_dir, {"maintenance_capacity": 2})
        state, _, issues = self._call(temp_run_dir)
        assert state == "INCOMPLETE"
        assert any("K mismatch" in i for i in issues)

    def test_incomplete_when_cost_regime_mismatch(self, temp_run_dir):
        self._setup(temp_run_dir, {"cost_regime_id": "failure-heavy-no-waste"})
        state, _, issues = self._call(temp_run_dir)
        assert state == "INCOMPLETE"

    def test_incomplete_when_seed_mismatch(self, temp_run_dir):
        self._setup(temp_run_dir, {"training_seed": 6522})
        state, _, issues = self._call(temp_run_dir)
        assert state == "INCOMPLETE"

    def test_incomplete_when_global_step_less_than_max(self, temp_run_dir):
        self._setup(temp_run_dir, {"final_global_step": 50000})
        state, _, issues = self._call(temp_run_dir)
        assert state == "INCOMPLETE"

    def test_incomplete_when_checkpoint_latest_missing(self, temp_run_dir):
        run_dir = self._setup(temp_run_dir)
        (run_dir / "checkpoint_latest.pt").unlink()
        state, _, issues = self._call(temp_run_dir)
        assert state == "INCOMPLETE"
        assert any("checkpoint_latest" in i.lower() for i in issues)

    def test_incomplete_when_validation_occurred_but_no_best(self, temp_run_dir):
        run_dir = self._setup(temp_run_dir)
        (run_dir / "checkpoint_best.pt").unlink()
        state, _, issues = self._call(temp_run_dir)
        assert state == "INCOMPLETE"
        assert any("checkpoint_best" in i.lower() for i in issues)

    def test_incomplete_when_training_metric_missing(self, temp_run_dir):
        run_dir = self._setup(temp_run_dir)
        (run_dir / "training_metrics.jsonl").unlink()
        state, _, issues = self._call(temp_run_dir)
        assert state == "INCOMPLETE"

    def test_incomplete_when_validation_metric_missing(self, temp_run_dir):
        run_dir = self._setup(temp_run_dir)
        (run_dir / "validation_metrics.json").unlink()
        state, _, issues = self._call(temp_run_dir)
        assert state == "INCOMPLETE"

    def test_incomplete_when_training_split_is_rl_test(self, temp_run_dir):
        self._setup(temp_run_dir, {"training_split": "rl_test"})
        state, _, issues = self._call(temp_run_dir)
        assert state == "INCOMPLETE"
        assert any("rl_test" in i.lower() or "FORBIDDEN" in i for i in issues)

    def test_incomplete_when_validation_split_is_rl_test(self, temp_run_dir):
        self._setup(temp_run_dir, {"validation_split": "rl_test"})
        state, _, issues = self._call(temp_run_dir)
        assert state == "INCOMPLETE"
        assert any("rl_test" in i.lower() or "FORBIDDEN" in i for i in issues)

    def test_incomplete_when_git_commit_wrong_length(self, temp_run_dir):
        # 12-character commit short (numeric-truth guard) — must NOT be accepted
        self._setup(temp_run_dir, {"git_commit": "123456"})
        state, _, issues = self._call(temp_run_dir)
        assert state == "INCOMPLETE"
        assert any("git_commit" in i.lower() and "40" in i for i in issues)

    def test_incomplete_when_git_commit_missing(self, temp_run_dir):
        self._setup(temp_run_dir)
        # Remove git_commit entirely
        run_dir = temp_run_dir / "test_run"
        with open(run_dir / "run_manifest.json") as f:
            manifest = json.load(f)
        del manifest["git_commit"]
        with open(run_dir / "run_manifest.json", "w") as f:
            json.dump(manifest, f)
        state, _, issues = self._call(temp_run_dir)
        assert state == "INCOMPLETE"

    def test_incomplete_when_checkpoint_schema_version_wrong(self, temp_run_dir):
        self._setup(temp_run_dir, {"checkpoint_schema_version": 4})
        state, _, issues = self._call(temp_run_dir)
        assert state == "INCOMPLETE"
        assert any("schema" in i.lower() for i in issues)