"""
Focused M5 Tests: CLI and Integration
"""

import pytest

pytestmark = pytest.mark.requires_external_assets
import subprocess
import sys
from pathlib import Path


@pytest.mark.slow
class TestCLI:
    """Test CLI scripts."""

    @pytest.fixture
    def project_root(self):
        """Get project root."""
        return Path(__file__).parent.parent

    def test_train_help(self, project_root):
        """Test training CLI help."""
        result = subprocess.run(
            ["python", "scripts/train_ddqn.py", "--help"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        assert result.returncode == 0
        assert "Train DDQN" in result.stdout or "train" in result.stdout.lower()

    def test_train_help_has_no_allow_baseline_banks(self, project_root):
        """The --allow-baseline-banks bypass must be removed entirely."""
        result = subprocess.run(
            ["python", "scripts/train_ddqn.py", "--help"],
            capture_output=True, text=True, cwd=project_root,
        )
        assert result.returncode == 0
        assert "allow-baseline-banks" not in result.stdout, \
            "--allow-baseline-banks must not appear in --help (removed)"

    def test_train_missing_banks_fails_closed(self, project_root):
        """A command that omits BOTH explicit bank flags must fail closed.

        There is no --allow-baseline-banks bypass (frozen decision: always
        require explicit banks).  The gate lives in the shared resolver."""
        result = subprocess.run(
            ["python", "scripts/train_ddqn.py", "--config", "configs/agents/ddqn_v1_k1.json",
             "--k-capacity", "1", "--cost-regime", "failure-light-no-waste",
             "--training-seed", "6521", "--run-id", "gate_k1", "--dry-run"],
            capture_output=True, text=True, cwd=project_root,
        )
        assert result.returncode != 0, "missing explicit banks must fail closed"
        combined = result.stdout + result.stderr
        assert "explicit-bank gate FAILED" in combined, combined

    def test_train_run_id_arg(self, project_root):
        """Test training CLI accepts --run-id argument with explicit banks."""
        result = subprocess.run(
            ["python", "scripts/train_ddqn.py", "--config", "configs/agents/ddqn_v1_k1.json",
             "--k-capacity", "1", "--cost-regime", "failure-light-no-waste",
             "--training-scenario-bank", "configs/scenarios/m5_pilot_k1__light.json",
             "--validation-scenario-bank", "configs/scenarios/m5_validation_k1__light.json",
             "--run-id", "test_run", "--dry-run"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
        assert "Run ID: test_run" in result.stdout

    def test_validate_config_help(self, project_root):
        """Test validation CLI help."""
        result = subprocess.run(
            ["python", "scripts/validate_config.py", "--help"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        assert result.returncode == 0
        assert "Validate" in result.stdout or "config" in result.stdout.lower()

    def test_generate_matrix_help(self, project_root):
        """Test matrix generator CLI help."""
        result = subprocess.run(
            ["python", "scripts/generate_m5_matrix.py", "--help"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        assert result.returncode == 0
        assert "matrix" in result.stdout.lower() or "Experiment" in result.stdout

    def test_validate_config_valid(self, project_root):
        """Test validation with valid config."""
        result = subprocess.run(
            ["python", "scripts/validate_config.py", "--config", "configs/agents/ddqn_v1.json"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        # Should pass validation
        assert result.returncode == 0 or "valid" in result.stdout.lower()

    def test_generate_matrix_dry_run(self, project_root):
        """Test matrix generator dry-run."""
        result = subprocess.run(
            ["python", "scripts/generate_m5_matrix.py", "--dry-run", "--validate-configs"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        assert result.returncode == 0
        assert "40" in result.stdout  # Should mention 40 runs
        assert "DRY-RUN" in result.stdout

    def test_generate_matrix_k1_config(self, project_root):
        """Test matrix generator uses K=1 config for K=1 runs."""
        result = subprocess.run(
            ["python", "scripts/generate_m5_matrix.py", "--dry-run", "--validate-configs"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        assert result.returncode == 0
        assert "ddqn_v1_k1.json" in result.stdout

    def test_generate_matrix_k2_config(self, project_root):
        """Test matrix generator uses K=2 config for K=2 runs."""
        result = subprocess.run(
            ["python", "scripts/generate_m5_matrix.py", "--dry-run", "--validate-configs"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        assert result.returncode == 0
        assert "ddqn_v1.json" in result.stdout
        # Verify ddqn_v1.json is used for K=2 (not ddqn_v1_k1.json)
        assert "--config configs/agents/ddqn_v1.json --k-capacity 2" in result.stdout

    def test_train_dry_run_k1(self, project_root):
        """Test training dry-run with K=1 config and explicit matching banks."""
        result = subprocess.run(
            ["python", "scripts/train_ddqn.py", "--config", "configs/agents/ddqn_v1_k1.json",
             "--k-capacity", "1", "--cost-regime", "failure-light-no-waste",
             "--training-seed", "6521", "--run-id", "test_k1", "--dry-run",
             "--training-scenario-bank", "configs/scenarios/m5_pilot_k1__light.json",
             "--validation-scenario-bank", "configs/scenarios/m5_validation_k1__light.json"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
        assert "K: 1" in result.stdout
        assert "Actions: 6" in result.stdout

    def test_train_dry_run_k2(self, project_root):
        """Test training dry-run with K=2 config and explicit matching banks."""
        result = subprocess.run(
            ["python", "scripts/train_ddqn.py", "--config", "configs/agents/ddqn_v1.json",
             "--k-capacity", "2", "--cost-regime", "failure-heavy-no-waste",
             "--training-seed", "6522", "--run-id", "test_k2", "--dry-run",
             "--training-scenario-bank", "configs/scenarios/m5_pilot_k2__heavy.json",
             "--validation-scenario-bank", "configs/scenarios/m5_validation_k2__heavy.json"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
        assert "K: 2" in result.stdout
        assert "Actions: 16" in result.stdout


@pytest.mark.slow
class TestTrainingSmoke:
    """Smoke test for training."""

    @pytest.fixture
    def project_root(self):
        """Get project root."""
        return Path(__file__).parent.parent

    def test_training_dry_run(self, project_root):
        """Test training dry-run mode with explicit matching banks."""
        result = subprocess.run(
            ["python", "scripts/train_ddqn.py", "--config", "configs/agents/ddqn_v1.json",
             "--k-capacity", "2", "--cost-regime", "failure-light-no-waste", "--dry-run",
             "--training-scenario-bank", "configs/scenarios/m5_pilot_k2__light.json",
             "--validation-scenario-bank", "configs/scenarios/m5_validation_k2__light.json"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        # Should validate and exit cleanly
        assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
        assert "Configuration validated" in result.stdout or "Dry-run" in result.stdout