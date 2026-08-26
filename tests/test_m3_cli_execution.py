"""
Real production CLI execution tests for run_m3_baselines.py

These tests invoke the actual CLI script as a subprocess and verify:
- Return code 0
- Expected policy executed
- Real scenario-bank loading
- Real environment rollout
- Real threshold result
- Real selected-threshold artifact
- Finite measured statistics
- Artifact row counts
- No rl_test

DO NOT use --help as execution evidence.
DO NOT accept timeout as success.
DO NOT accept nonzero return code.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import pytest

pytestmark = pytest.mark.requires_external_assets

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "run_m3_baselines.py"
PROJECT_ROOT = Path(__file__).parent.parent
MINI_CONFIG_PATH = Path(__file__).parent / "m3_mini_fixtures" / "mini_config.json"


def run_cli(
    args: list[str],
    timeout: int = 300,
    expect_timeout: bool = False,
) -> Tuple[int, str, str]:
    """
    Run CLI command and return (returncode, stdout, stderr).

    Args:
        args: CLI arguments (without python command)
        timeout: Timeout in seconds (default 120 for mini tests)
        expect_timeout: If True, timeout is expected

    Returns:
        Tuple of (returncode, stdout, stderr)

    Raises:
        subprocess.TimeoutExpired: If timeout and not expect_timeout
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
            return None, e.stdout or "", e.stderr or ""
        raise


class TestProductionCliTuningMini:
    """
    Test real production CLI tuning execution.

    Runs actual tune command with mini config:
    - One policy (age_threshold)
    - One split (rl_validation)
    - One K (1)
    - One cost regime (failure-light-no-waste)
    - One scenario bank (rl_validation_smoke)
    - Two reset seeds
    - Two-value threshold grid
    """

    @pytest.fixture(autouse=True)
    def setup_teardown(self, tmp_path):
        """Create temporary output directory."""
        self.output_dir = tmp_path / "m3_tune_mini"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        yield
        # Cleanup is automatic for tmp_path

    def test_real_tuning_execution_age_threshold(self) -> None:
        """
        Real production CLI tuning must execute successfully.

        Verifies:
        - Return code 0
        - age_threshold policy executed
        - Real scenario bank loaded
        - Real threshold search completed
        - Selected threshold artifact written
        - Finite mean cost
        """
        returncode, stdout, stderr = run_cli([
            "--tune",
            "--policy", "age_threshold",
            "--split", "rl_validation",
            "--config", str(MINI_CONFIG_PATH),
            "--output-dir", str(self.output_dir),
        ], timeout=300)

        # Must succeed
        assert returncode == 0, f"CLI tuning failed with code {returncode}: {stderr}"

        # Must run age_threshold
        assert "age_threshold" in stdout, "age_threshold not mentioned in output"
        assert "Tuning age_threshold" in stdout or "age_threshold..." in stdout

        # Must not run rl_test
        assert "rl_test" not in stdout, "rl_test should not be mentioned"
        assert "rl_test" not in stderr, "rl_test should not be mentioned"

        # Must have threshold search output
        assert "Best threshold:" in stdout or "threshold=" in stdout

        # Must have finite cost
        assert "Mean cost:" in stdout or "cost=" in stdout

        # Verify artifacts written
        selected_thresholds_path = self.output_dir / "selected_thresholds.json"
        assert selected_thresholds_path.exists(), "selected_thresholds.json not written"

        with open(selected_thresholds_path, "r") as f:
            thresholds = json.load(f)

        # Must have age_threshold entry
        age_key = "age_threshold_k1_failure-light-no-waste"
        assert age_key in thresholds, f"Key {age_key} not in thresholds: {list(thresholds.keys())}"

        # Must have valid threshold value
        threshold_entry = thresholds[age_key]
        assert "threshold" in threshold_entry
        threshold_value = threshold_entry["threshold"]
        assert isinstance(threshold_value, (int, float))
        assert 0 < threshold_value < 300, f"Threshold {threshold_value} out of expected range"

        # Must have finite mean cost
        assert "mean_total_cost" in threshold_entry
        mean_cost = threshold_entry["mean_total_cost"]
        assert isinstance(mean_cost, (int, float))
        assert mean_cost > 0, f"Mean cost {mean_cost} should be positive"
        assert mean_cost < 1e6, f"Mean cost {mean_cost} unreasonably high"

        # Verify threshold search results written
        search_results_path = self.output_dir / "threshold_search_results.parquet"
        assert search_results_path.exists(), "threshold_search_results.parquet not written"

        # Verify summary written
        summary_path = self.output_dir / "threshold_search_summary.csv"
        assert summary_path.exists(), "threshold_search_summary.csv not written"

    def test_real_tuning_execution_predicted_rul(self) -> None:
        """
        Real production CLI tuning for predicted_rul_threshold.

        Verifies same guarantees as age_threshold test.
        """
        returncode, stdout, stderr = run_cli([
            "--tune",
            "--policy", "predicted_rul_threshold",
            "--split", "rl_validation",
            "--config", str(MINI_CONFIG_PATH),
            "--output-dir", str(self.output_dir),
        ], timeout=300)

        assert returncode == 0, f"CLI tuning failed: {stderr}"
        assert "predicted_rul_threshold" in stdout

        # Verify artifact
        selected_thresholds_path = self.output_dir / "selected_thresholds.json"
        assert selected_thresholds_path.exists()

        with open(selected_thresholds_path, "r") as f:
            thresholds = json.load(f)

        rul_key = "predicted_rul_threshold_k1_failure-light-no-waste"
        assert rul_key in thresholds

        threshold_entry = thresholds[rul_key]
        assert "threshold" in threshold_entry
        assert "mean_total_cost" in threshold_entry


class TestProductionCliEvaluationMini:
    """
    Test real production CLI evaluation execution.

    Runs actual evaluate command with mini config:
    - One policy (corrective_only)
    - One split (predictor_train)
    - One K (1)
    - One cost regime (failure-light-no-waste)
    - One scenario bank (predictor_train_smoke)
    - Two reset seeds
    """

    @pytest.fixture(autouse=True)
    def setup_teardown(self, tmp_path):
        """Create temporary output directory."""
        self.output_dir = tmp_path / "m3_eval_mini"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        yield

    def test_real_evaluation_execution_corrective_only(self) -> None:
        """Smoke coverage for corrective_only family.

        The --evaluate coverage for corrective_only
        lives in TestProductionCliFormalEvaluationNonThreshold; this
        test exercises --smoke to keep a smoke-only signal. formal
        --evaluate is fail-closed: it refuses to construct any
        environment without a fully-formed selected_thresholds.json.
        """
        returncode, stdout, stderr = run_cli([
            "--smoke",
            "--policy", "corrective_only",
            "--split", "predictor_train",
            "--k-capacity", "1",
            "--cost-regime", "failure-light-no-waste",
            "--config", str(MINI_CONFIG_PATH),
            "--output-dir", str(self.output_dir),
        ], timeout=120)

        # Must succeed
        assert returncode == 0, f"CLI smoke failed with code {returncode}: {stderr}"
        assert "corrective_only" in stdout
        assert "rl_test" not in stdout
        assert "rl_test" not in stderr

    def test_real_evaluation_execution_random_feasible(self) -> None:
        """
        random_feasible is also --smoke only (no threshold policy).
        """
        returncode, stdout, stderr = run_cli([
            "--smoke",
            "--policy", "random_feasible",
            "--split", "predictor_train",
            "--k-capacity", "1",
            "--cost-regime", "failure-light-no-waste",
            "--config", str(MINI_CONFIG_PATH),
            "--output-dir", str(self.output_dir),
        ], timeout=120)

        assert returncode == 0, f"CLI smoke failed: {stderr}"
        assert "random_feasible" in stdout

        # Verify artifacts
        episode_path = self.output_dir / "episode_results.parquet"
        assert episode_path.exists()


class TestCliRlTestBarrier:
    """
    Test rl_test barrier - must reject before any data loading.
    """

    def test_rl_test_tuning_rejected(self) -> None:
        """rl_test tuning must be rejected with nonzero exit code."""
        returncode, stdout, stderr = run_cli([
            "--tune",
            "--split", "rl_test",
            "--config", str(MINI_CONFIG_PATH),
        ], timeout=30)

        # Must fail
        assert returncode != 0, "rl_test tuning should fail"

        # Must have clear error
        assert "rl_test" in stderr or "ERROR" in stderr or "forbidden" in stderr

    def test_rl_test_evaluation_rejected(self) -> None:
        """rl_test evaluation must be rejected with nonzero exit code."""
        returncode, stdout, stderr = run_cli([
            "--evaluate",
            "--split", "rl_test",
            "--config", str(MINI_CONFIG_PATH),
        ], timeout=30)

        # Must fail
        assert returncode != 0, "rl_test evaluation should fail"

        # Must have clear error
        assert "rl_test" in stderr or "ERROR" in stderr or "forbidden" in stderr


class TestCliSmokeExecution:
    """
    Test real production CLI smoke execution.
    """

    @pytest.fixture(autouse=True)
    def setup_teardown(self, tmp_path):
        """Create temporary output directory."""
        self.output_dir = tmp_path / "m3_smoke_mini"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        yield

    def test_real_smoke_execution(self) -> None:
        """
        Real production CLI smoke test must execute successfully.

        Verifies:
        - Return code 0
        - Multiple policies executed
        - Real scenario bank loaded
        - Real environment rollout
        - Episode results artifact written
        """
        returncode, stdout, stderr = run_cli([
            "--smoke",
            "--split", "predictor_train",
            "--k-capacity", "1",
            "--cost-regime", "failure-light-no-waste",
            "--config", str(MINI_CONFIG_PATH),
            "--output-dir", str(self.output_dir),
        ], timeout=180)

        # Must succeed
        assert returncode == 0, f"CLI smoke failed with code {returncode}: {stderr}"

        # Must run multiple policies
        assert "corrective_only" in stdout
        assert "random_feasible" in stdout

        # Must have episode results
        episode_path = self.output_dir / "episode_results.parquet"
        assert episode_path.exists()

        # Verify log
        log_path = self.output_dir / "m3_run.log"
        assert log_path.exists()

        with open(log_path, "r") as f:
            log_content = f.read()

        assert "EXIT_CODE=0" in log_content
        assert "M3 Smoke Test" in log_content


class TestProductionCliFormalEvaluationNonThreshold:
    """
    --evaluate coverage for the two non-threshold policy
    families (corrective_only and random_feasible).

    These tests deliberately do NOT use --smoke. formal --evaluate must
    fail closed when selected_thresholds.json is missing, malformed,
    short, or has the wrong _meta envelope; the tests build a full
    32-identity envelope aligned with the formal_closeout contract so
    the strict loader accepts it, then invoke --evaluate with the same
    non-threshold policy and assert the run produced real
    episode_results.parquet rows tagged with that policy_family.

    Without this coverage the only signal we had was --smoke, which is
    insufficient: smoke substitutions are forbidden in formal mode.
    """

    @pytest.fixture(autouse=True)
    def setup_teardown(self, tmp_path):
        self.output_dir = tmp_path / "m3_formal_eval_nonthreshold"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resolved_config_path = self.output_dir / "resolved_config.json"
        self.selected_thresholds_path = self.output_dir / "selected_thresholds.json"
        yield

    def _write_resolved_config(self) -> str:
        """Write resolved_config.json and return its canonical-JSON SHA256.

        Mirrors what run_m3_baselines.py writes in run_tune: an actual
        snapshot of the config dict,not a directory hash.
        """
        with open(MINI_CONFIG_PATH, "r") as f:
            config = json.load(f)
        with open(self.resolved_config_path, "w") as f:
            # Canonical-JSON sort_keys=True to be deterministic.
            json.dump(config, f, sort_keys=True, separators=(",", ":"))
        import hashlib
        with open(self.resolved_config_path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        return sha

    def _write_full_32_identity_envelope(
        self,
        config_sha256: str,
        formal_run_id: str,
        implementation_commit: str,
    ) -> None:
        """Write a full 32-identity selected_thresholds.json with _meta.

        All 32 identities (4 threshold policies × 2 K × 4 regimes) are
        stamped; threshold values are taken from the frozen grids used by
        load_formal_selected_thresholds so the loader will accept them.
        corrective_only and random_feasible receive no tuple here;
        evaluation runs those via the same six-policy loop without
        requiring a threshold identity.
        """
        import hashlib
        import tempfile
        from src.baselines.tuning import (
            AGE_THRESHOLDS,
            PREDICTED_RUL_THRESHOLDS,
            GREEDY_ACTIVATION_THRESHOLDS,
            ORACLE_THRESHOLDS,
        )

        threshold_policies = {
            "age_threshold": AGE_THRESHOLDS,
            "predicted_rul_threshold": PREDICTED_RUL_THRESHOLDS,
            "greedy_predicted_rul": GREEDY_ACTIVATION_THRESHOLDS,
            "oracle_threshold": ORACLE_THRESHOLDS,
        }
        cost_regimes = (
            "failure-light-no-waste",
            "failure-heavy-no-waste",
            "failure-light-waste-aware",
            "failure-heavy-waste-aware",
        )

        data: dict = {}
        for policy, grid in threshold_policies.items():
            # Use a deterministic first-grid element as the (stub) winner
            # value; this is only ever used by threshold-based policies
            # and is not what we are validating here.
            winner = next(iter(grid))
            for k in (1, 2):
                for regime in cost_regimes:
                    key = f"{policy}_k{k}_{regime}"
                    data[key] = {
                        "threshold": int(winner),
                        "k_capacity": k,
                        "cost_regime_id": regime,
                        "mean_total_cost": 10.0,
                    }

        data["_meta"] = {
            "formal_run_id": formal_run_id,
            "config_sha256": config_sha256,
            "implementation_commit": implementation_commit,
            "written_at": "2026-07-24T00:00:00",
        }

        with open(self.selected_thresholds_path, "w") as f:
            json.dump(data, f, sort_keys=True, separators=(",", ":"))

    def test_real_evaluation_corrective_only_through_formal_six_policy(self):
        """corrective_only executes through the real formal --evaluate.

        The full 32-identity envelope is staged BEFORE evaluation; the
        test asserts that downstream policy rows for corrective_only
        appear in episode_results.parquet.
        """
        config_sha = self._write_resolved_config()
        formal_run_id = self.output_dir.name
        impl_commit = os.environ.get("M3_FINAL_IMPLEMENTATION_COMMIT", "")
        self._write_full_32_identity_envelope(
            config_sha256=config_sha,
            formal_run_id=formal_run_id,
            implementation_commit=impl_commit,
        )

        env = {**os.environ, "M3_FINAL_IMPLEMENTATION_COMMIT": impl_commit} if impl_commit else None
        cmd = [
            sys.executable, str(SCRIPT_PATH),
            "--evaluate",
            "--split", "predictor_train",
            "--k-capacity", "1",
            "--cost-regime", "failure-light-no-waste",
            "--config", str(MINI_CONFIG_PATH),
            "--output-dir", str(self.output_dir),
            "--policy", "corrective_only",
            "--allow-oracle",
            "--mode", "diagnostic_legacy",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PROJECT_ROOT),
            env=env if env else None,
        )
        assert result.returncode == 0, (
            f"--evaluate corrective_only failed: stdout={result.stdout[:400]}, "
            f"stderr={result.stderr[:400]}"
        )

        episode_path = self.output_dir / "episode_results.parquet"
        assert episode_path.exists(), "episode_results.parquet not written"

        import pandas as pd
        df = pd.read_parquet(episode_path)
        # In --policy corrective_only mode, only that family emits rows.
        assert "corrective_only" in set(df["policy_family"].unique())
        # Threshold must be null/None (corrective_only never uses a threshold).
        if "threshold" in df.columns:
            assert df["threshold"].isna().all() or (df["threshold"].fillna(-1) < 0).all()

    def test_real_evaluation_random_feasible_through_formal_six_policy(self):
        """random_feasible executes through the real formal --evaluate."""
        config_sha = self._write_resolved_config()
        formal_run_id = self.output_dir.name
        impl_commit = os.environ.get("M3_FINAL_IMPLEMENTATION_COMMIT", "")
        self._write_full_32_identity_envelope(
            config_sha256=config_sha,
            formal_run_id=formal_run_id,
            implementation_commit=impl_commit,
        )

        env = {**os.environ, "M3_FINAL_IMPLEMENTATION_COMMIT": impl_commit} if impl_commit else None
        cmd = [
            sys.executable, str(SCRIPT_PATH),
            "--evaluate",
            "--split", "predictor_train",
            "--k-capacity", "1",
            "--cost-regime", "failure-light-no-waste",
            "--config", str(MINI_CONFIG_PATH),
            "--output-dir", str(self.output_dir),
            "--policy", "random_feasible",
            "--allow-oracle",
            "--mode", "diagnostic_legacy",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PROJECT_ROOT),
            env=env if env else None,
        )
        assert result.returncode == 0, (
            f"--evaluate random_feasible failed: stdout={result.stdout[:400]}, "
            f"stderr={result.stderr[:400]}"
        )

        episode_path = self.output_dir / "episode_results.parquet"
        assert episode_path.exists(), "episode_results.parquet not written"

        import pandas as pd
        df = pd.read_parquet(episode_path)
        assert "random_feasible" in set(df["policy_family"].unique())
