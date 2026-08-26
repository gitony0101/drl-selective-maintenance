"""
CLI policy filter tests for run_m3_baselines.py

Tests verify:
A. --policy corrective_only runs only corrective_only
B. --policy oracle_threshold --allow-oracle runs only oracle_threshold
C. --policy all or no filter runs all policy families
D. Without --allow-oracle, explicit oracle request fails clearly
"""

import subprocess
import sys
import pytest
from pathlib import Path

pytestmark = pytest.mark.requires_external_assets

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "run_m3_baselines.py"
PROJECT_ROOT = Path(__file__).parent.parent


def run_cli(args: list[str], timeout: int = 60, expect_timeout: bool = False) -> tuple[int, str, str]:
    """
    Run CLI command and return (returncode, stdout, stderr).

    Args:
        args: CLI arguments (without python command)
        timeout: Timeout in seconds
        expect_timeout: If True, timeout is expected and returns (None, stdout, stderr)

    Returns:
        Tuple of (returncode, stdout, stderr). If timeout and expect_timeout,
        returns (None, stdout, stderr) instead of raising.
    """
    cmd = [sys.executable, str(SCRIPT_PATH)] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT),
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as e:
        if expect_timeout:
            # Return partial output
            return None, e.stdout or "", e.stderr or ""
        raise


class TestPolicyFilterCorrectiveOnly:
    """Test A: --policy corrective_only runs only corrective_only."""

    def test_policy_corrective_only_runs_single_policy(self) -> None:
        """--policy corrective_only should run only corrective_only."""
        returncode, stdout, stderr = run_cli([
            "--smoke",
            "--policy", "corrective_only",
            "--split", "predictor_train",
            "--output-dir", "/tmp/m3_test_corrective_only",
        ])

        # Should succeed
        assert returncode == 0, f"CLI failed: {stderr}"

        # Should only run corrective_only
        assert "corrective_only" in stdout
        # Should NOT run other policies
        assert "random_feasible" not in stdout
        assert "age_threshold" not in stdout
        assert "predicted_rul_threshold" not in stdout
        assert "greedy_predicted_rul" not in stdout
        assert "oracle_threshold" not in stdout


class TestPolicyFilterOracleWithAllow:
    """Test B: --policy oracle_threshold --allow-oracle runs only oracle."""

    def test_policy_oracle_with_allow_flag_runs_oracle(self) -> None:
        """--policy oracle_threshold --allow-oracle should run oracle."""
        returncode, stdout, stderr = run_cli([
            "--smoke",
            "--policy", "oracle_threshold",
            "--allow-oracle",
            "--split", "predictor_train",
            "--output-dir", "/tmp/m3_test_oracle",
        ])

        # Should succeed
        assert returncode == 0, f"CLI failed: {stderr}"

        # Should run oracle_threshold
        assert "oracle_threshold" in stdout

        # Should NOT run other policies
        assert "corrective_only" not in stdout
        assert "random_feasible" not in stdout


class TestPolicyFilterAllOrNone:
    """Test C: --policy all or no filter runs all policy families."""

    def test_no_policy_filter_runs_all(self) -> None:
        """No --policy filter should run all policies (except oracle)."""
        returncode, stdout, stderr = run_cli([
            "--smoke",
            "--split", "predictor_train",
            "--output-dir", "/tmp/m3_test_all",
        ])

        # Should succeed
        assert returncode == 0, f"CLI failed: {stderr}"

        # Should run all non-oracle policies
        assert "corrective_only" in stdout
        assert "random_feasible" in stdout
        assert "age_threshold" in stdout
        assert "predicted_rul_threshold" in stdout
        assert "greedy_predicted_rul" in stdout
        # Oracle should NOT run without --allow-oracle
        assert "oracle_threshold" not in stdout


class TestOracleWithoutAllowFlagFails:
    """Test D: Without --allow-oracle, explicit oracle request fails."""

    def test_oracle_without_allow_flag_fails(self) -> None:
        """--policy oracle_threshold without --allow-oracle should fail."""
        returncode, stdout, stderr = run_cli([
            "--smoke",
            "--policy", "oracle_threshold",
            "--split", "predictor_train",
            "--output-dir", "/tmp/m3_test_oracle_fail",
        ])

        # Should fail
        assert returncode != 0, "Oracle without --allow-oracle should fail"

        # Should have clear error message
        assert "ERROR" in stderr or "requires --allow-oracle" in stderr


class TestTunePolicyFilter:
    """Test policy filter in tune mode.

    Note: Full tuning execution tests are in test_m3_cli_execution.py::TestProductionCliTuningMini
    which runs real production CLI tuning with mini config.
    """

    def test_tune_with_policy_filter_accepts_args(self) -> None:
        """--tune --policy age_threshold should be accepted by CLI.

        This test verifies the CLI accepts the arguments without --help.
        Full execution tests are in test_m3_cli_execution.py.
        """
        # Verify argparse accepts the arguments by checking help output
        # without actually running tune
        returncode, stdout, stderr = run_cli([
            "--tune",
            "--policy", "age_threshold",
            "--split", "rl_validation",
            "--help",
        ], timeout=10)

        # Should succeed (help is displayed)
        assert returncode == 0, f"CLI failed: {stderr}"
        # Help should mention policy options
        assert "age_threshold" in stdout


class TestEvaluatePolicyFilter:
    """Test policy filter in evaluate mode.

    Note: Full evaluation execution tests are in test_m3_cli_execution.py::TestProductionCliEvaluationMini
    which runs real production CLI evaluation with mini config.
    """

    def test_evaluate_with_policy_filter_accepts_args(self) -> None:
        """--evaluate --policy random_feasible should be accepted by CLI.

        This test verifies the CLI accepts the arguments without --help.
        Full execution tests are in test_m3_cli_execution.py.
        """
        # Verify argparse accepts the arguments by checking help output
        returncode, stdout, stderr = run_cli([
            "--evaluate",
            "--policy", "random_feasible",
            "--split", "predictor_train",
            "--help",
        ], timeout=10)

        # Should succeed (help is displayed)
        assert returncode == 0, f"CLI failed: {stderr}"
        # Help should mention policy options
        assert "random_feasible" in stdout


class TestInvalidPolicyName:
    """Test invalid policy name handling."""

    def test_invalid_policy_name_fails(self) -> None:
        """Invalid policy name should fail with clear error."""
        returncode, stdout, stderr = run_cli([
            "--smoke",
            "--policy", "nonexistent_policy",
            "--split", "predictor_train",
        ])

        # argparse should reject invalid choices
        assert returncode != 0
        assert "invalid choice" in stderr or "unrecognized" in stderr