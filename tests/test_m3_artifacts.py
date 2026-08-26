"""
Tests for Milestone 3 artifacts.

Verifies:
- Artifact writing (JSON, parquet, CSV)
- SHA256 hash computation
- File size computation
- Schema validation
- Numeric value validation (no NaN/Inf)
- Artifact manifest generation
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import pandas as pd

from src.baselines.artifacts import (
    compute_sha256,
    get_file_size,
    validate_json_serializable,
    write_json_safe,
    write_resolved_config,
    write_threshold_search_results,
    write_threshold_search_summary,
    write_selected_thresholds,
    write_episode_results,
    write_summary_by_policy,
    write_sanity_checks,
    write_run_provenance,
    write_artifact_manifest,
    write_run_log,
    validate_artifacts,
    get_row_count,
    get_schema_columns,
)
from src.baselines.evaluator import EpisodeResult
from src.baselines.tuning import ThresholdCandidate


class TestComputeSha256:
    """Test SHA256 hash computation."""

    def test_sha256_computation(self):
        """Test compute_sha256 returns valid hash."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("test content")
            f.flush()
            hash1 = compute_sha256(Path(f.name))

        # Same content should produce same hash
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("test content")
            f.flush()
            hash2 = compute_sha256(Path(f.name))

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 is 64 hex characters

    def test_sha256_different_content(self):
        """Test different content produces different hash."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("content 1")
            f.flush()
            hash1 = compute_sha256(Path(f.name))

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("content 2")
            f.flush()
            hash2 = compute_sha256(Path(f.name))

        assert hash1 != hash2


class TestGetFileSize:
    """Test file size computation."""

    def test_file_size(self):
        """Test get_file_size returns correct size."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("12345")
            f.flush()
            size = get_file_size(Path(f.name))

        assert size == 5


class TestValidateJsonSerializable:
    """Test JSON validation."""

    def test_valid_json(self):
        """Test valid JSON passes validation."""
        data = {"a": 1, "b": 2.0, "c": "string", "d": True, "e": None}
        # Should not raise
        validate_json_serializable(data)

    def test_nan_rejected(self):
        """Test NaN rejected from JSON."""
        data = {"value": float("nan")}
        with pytest.raises(ValueError, match="NaN"):
            validate_json_serializable(data)

    def test_inf_rejected(self):
        """Test Inf rejected from JSON."""
        data = {"value": float("inf")}
        with pytest.raises(ValueError, match="Inf"):
            validate_json_serializable(data)


class TestWriteJsonSafe:
    """Test safe JSON writing."""

    def test_write_valid_json(self):
        """Test write_json_safe writes valid JSON."""
        data = {"a": 1, "b": 2.0}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            write_json_safe(data, path)

            with open(path, "r") as f:
                loaded = json.load(f)

            assert loaded == data

    def test_write_rejects_nan(self):
        """Test write_json_safe rejects NaN."""
        data = {"value": float("nan")}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            with pytest.raises(ValueError, match="NaN"):
                write_json_safe(data, path)


class TestWriteResolvedConfig:
    """Test resolved config writing."""

    def test_write_resolved_config(self):
        """Test write_resolved_config writes valid JSON."""
        config = {
            "m3_version": "m3_v1",
            "split": "rl_validation",
            "k_capacity": 2,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            path = write_resolved_config(config, output_dir)

            assert path.exists()
            with open(path, "r") as f:
                loaded = json.load(f)

            assert loaded["m3_version"] == "m3_v1"


class TestWriteThresholdSearchResults:
    """Test threshold search results writing."""

    def test_write_threshold_search_results(self):
        """Test write_threshold_search_results writes parquet."""
        candidates = [
            ThresholdCandidate(
                policy_family="age_threshold",
                threshold=100,
                k_capacity=2,
                cost_regime_id="failure-light-no-waste",
                mean_total_cost=50.0,
                total_failures=5,
                mean_wasted_life_cost=2.0,
                episode_count=10,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            # The canonical parquet is now episode-level (9000 rows for
            # the formal oracle run). The 360-row candidate summary
            # lives in ``threshold_search_summary.csv``. Wrap one
            # candidate into a single episode row for this minimal
            # unit test.
            ep_rows = [{
                "policy_family": "age_threshold",
                "threshold": 100,
                "k_capacity": 2,
                "cost_regime_id": "failure-light-no-waste",
                "scenario_id": "synth",
                "reset_seed": 6521,
                "total_cost": 50.0,
                "preventive_cost": 0.0,
                "failure_cost": 0.0,
                "wasted_life_cost": 2.0,
                "failure_count": 5,
                "episode_steps": 100,
                "completed": True,
            }]
            path = write_threshold_search_results(ep_rows, output_dir)

            assert path.exists()
            assert path.suffix == ".parquet"

            df = pd.read_parquet(path)
            assert len(df) == 1
            assert df.iloc[0]["threshold"] == 100


class TestWriteEpisodeResults:
    """Test episode results writing."""

    def test_write_episode_results(self):
        """Test write_episode_results writes parquet."""
        results = [
            EpisodeResult(
                run_id="run_1",
                policy_id="policy_a",
                policy_family="age_threshold",
                threshold=100,
                split="rl_validation",
                scenario_id="scenario_1",
                cost_regime_id="failure-light-no-waste",
                maintenance_capacity=2,
                reset_seed=6521,
                policy_seed=42,
                episode_steps=100,
                episode_return=-50.0,
                discounted_return=-50.0,
                total_cost=50.0,
                preventive_cost=10.0,
                failure_cost=40.0,
                wasted_life_cost=0.0,
                preventive_replacement_count=10,
                failure_count=8,
                action_count=50,
                empty_action_count=50,
                capacity_saturated_step_count=10,
                mean_selected_predicted_rul=0.3,
                mean_selected_age=0.5,
                nan_observation_count=0,
                inf_observation_count=0,
                terminated_count=0,
                truncated=True,
                completed=True,
                error=None,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            path = write_episode_results(results, output_dir)

            assert path.exists()
            df = pd.read_parquet(path)
            assert len(df) == 1
            assert df.iloc[0]["run_id"] == "run_1"


class TestValidateArtifacts:
    """Test artifact validation."""

    def test_validate_artifacts_all_present(self):
        """Test validate_artifacts when all files present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create all required files per validate_artifacts default list
            (output_dir / "resolved_config.json").write_text("{}")
            (output_dir / "selected_thresholds.json").write_text("{}")
            (output_dir / "sanity_checks.json").write_text("{}")
            (output_dir / "run_provenance.json").write_text("{}")
            (output_dir / "artifact_manifest.json").write_text("{}")
            (output_dir / "m3_run.log").write_text("log content")

            # Create parquet files
            df = pd.DataFrame({"a": [1, 2]})
            df.to_parquet(output_dir / "threshold_search_results.parquet")
            df.to_parquet(output_dir / "episode_results.parquet")

            # Create CSV files
            df.to_csv(output_dir / "threshold_search_summary.csv", index=False)
            df.to_csv(output_dir / "summary_by_policy.csv", index=False)

            # Create summary JSON
            (output_dir / "summary_by_policy.json").write_text("[]")

            validation = validate_artifacts(output_dir)

            assert validation["all_present"] is True

    def test_validate_artifacts_missing_files(self):
        """Test validate_artifacts detects missing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create no files
            validation = validate_artifacts(output_dir)

            assert validation["all_present"] is False
            assert len(validation["missing_files"]) > 0


class TestGetRowCount:
    """Test row count computation."""

    def test_get_row_count_parquet(self):
        """Test get_row_count for parquet file."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.parquet"
            df.to_parquet(path)
            count = get_row_count(path)
            assert count == 3

    def test_get_row_count_csv(self):
        """Test get_row_count for CSV file."""
        df = pd.DataFrame({"a": [1, 2, 3, 4]})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            df.to_csv(path, index=False)
            count = get_row_count(path)
            assert count == 4


class TestGetSchemaColumns:
    """Test schema columns computation."""

    def test_get_schema_columns_parquet(self):
        """Test get_schema_columns for parquet file."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.parquet"
            df.to_parquet(path)
            cols = get_schema_columns(path)
            assert cols == ["a", "b", "c"]

    def test_get_schema_columns_csv(self):
        """Test get_schema_columns for CSV file."""
        df = pd.DataFrame({"x": [1], "y": [2]})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            df.to_csv(path, index=False)
            cols = get_schema_columns(path)
            assert cols == ["x", "y"]


class TestWriteArtifactManifest:
    """Test artifact manifest writing."""

    def test_write_artifact_manifest(self):
        """Test write_artifact_manifest writes valid manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create a file
            (output_dir / "test.txt").write_text("content")

            manifest_path = write_artifact_manifest(output_dir)

            assert manifest_path.exists()
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            assert "artifacts" in manifest
            assert len(manifest["artifacts"]) == 1
            assert manifest["artifacts"][0]["relative_path"] == "test.txt"


class TestWriteRunLog:
    """Test run log writing."""

    def test_write_run_log(self):
        """Test write_run_log writes log with exit code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            log_messages = ["Step 1", "Step 2"]

            path = write_run_log(log_messages, output_dir, exit_code=0)

            assert path.exists()
            content = path.read_text()

            assert "Step 1" in content
            assert "Step 2" in content
            assert "EXIT_CODE=0" in content


class TestThresholdUseEquality:
    """Test threshold-use equality validation."""

    def test_threshold_use_equality_correct(self):
        """Test validation passes when evaluation uses selected thresholds."""
        import pandas as pd
        from scripts.validate_m3_artifacts import validate_threshold_use_equality
        from pathlib import Path
        import tempfile
        import json

        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path)

            # Create selected thresholds
            selected = {
                "age_threshold_k1_failure-light-no-waste": {
                    "threshold": 125,
                    "k_capacity": 1,
                    "cost_regime_id": "failure-light-no-waste",
                    "mean_total_cost": 16.8,
                    "total_failures": 0,
                    "mean_wasted_life_cost": 0.0,
                    "episode_count": 25,
                },
            }
            selected_path = tmp_path / "selected_thresholds.json"
            with open(selected_path, "w") as f:
                json.dump(selected, f)

            # Create episode results with correct thresholds
            df = pd.DataFrame({
                "policy_family": ["age_threshold"] * 100,
                "maintenance_capacity": [1] * 100,
                "cost_regime_id": ["failure-light-no-waste"] * 100,
                "threshold": [125.0] * 100,  # Matches selected
                "split": ["rl_validation"] * 100,
                "scenario_id": [f"scenario_{i%5}" for i in range(100)],
                "reset_seed": [6521 + (i%5) for i in range(100)],
                "episode_return": [-50.0] * 100,
                "total_cost": [50.0] * 100,
                "preventive_cost": [10.0] * 100,
                "failure_cost": [40.0] * 100,
                "wasted_life_cost": [0.0] * 100,
                "episode_steps": [100] * 100,
            })
            episode_path = tmp_path / "episode_results.parquet"
            df.to_parquet(episode_path)

            # Run validation
            success, errors = validate_threshold_use_equality(episode_path, selected_path)

            # Should pass
            assert success is True, f"Validation failed: {errors}"

    def test_threshold_use_equality_wrong_threshold(self):
        """Test validation fails when evaluation uses wrong threshold."""
        import pandas as pd
        from scripts.validate_m3_artifacts import validate_threshold_use_equality
        from pathlib import Path
        import tempfile
        import json

        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path)

            # Create selected thresholds
            selected = {
                "age_threshold_k1_failure-light-no-waste": {
                    "threshold": 125,  # Selected threshold is 125
                    "k_capacity": 1,
                    "cost_regime_id": "failure-light-no-waste",
                    "mean_total_cost": 16.8,
                    "total_failures": 0,
                    "mean_wasted_life_cost": 0.0,
                    "episode_count": 25,
                },
            }
            selected_path = tmp_path / "selected_thresholds.json"
            with open(selected_path, "w") as f:
                json.dump(selected, f)

            # Create episode results with WRONG threshold (100 instead of 125)
            df = pd.DataFrame({
                "policy_family": ["age_threshold"] * 100,
                "maintenance_capacity": [1] * 100,
                "cost_regime_id": ["failure-light-no-waste"] * 100,
                "threshold": [100.0] * 100,  # WRONG: should be 125
                "split": ["rl_validation"] * 100,
                "scenario_id": [f"scenario_{i%5}" for i in range(100)],
                "reset_seed": [6521 + (i%5) for i in range(100)],
                "episode_return": [-50.0] * 100,
                "total_cost": [50.0] * 100,
                "preventive_cost": [10.0] * 100,
                "failure_cost": [40.0] * 100,
                "wasted_life_cost": [0.0] * 100,
                "episode_steps": [100] * 100,
            })
            episode_path = tmp_path / "episode_results.parquet"
            df.to_parquet(episode_path)

            # Run validation
            success, errors = validate_threshold_use_equality(episode_path, selected_path)

            # Should fail
            assert success is False, "Validation should have failed"
            assert len(errors) > 0
            assert "100.0" in errors[0] or "125" in errors[0]

    def test_threshold_use_equality_none_threshold(self):
        """Test validation fails when evaluation has None threshold (default fallback)."""
        import pandas as pd
        from scripts.validate_m3_artifacts import validate_threshold_use_equality
        from pathlib import Path
        import tempfile
        import json

        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path)

            # Create selected thresholds
            selected = {
                "age_threshold_k1_failure-light-no-waste": {
                    "threshold": 125,
                    "k_capacity": 1,
                    "cost_regime_id": "failure-light-no-waste",
                    "mean_total_cost": 16.8,
                    "total_failures": 0,
                    "mean_wasted_life_cost": 0.0,
                    "episode_count": 25,
                },
            }
            selected_path = tmp_path / "selected_thresholds.json"
            with open(selected_path, "w") as f:
                json.dump(selected, f)

            # Create episode results with None threshold (default fallback)
            df = pd.DataFrame({
                "policy_family": ["age_threshold"] * 100,
                "maintenance_capacity": [1] * 100,
                "cost_regime_id": ["failure-light-no-waste"] * 100,
                "threshold": [None] * 100,  # None = default fallback
                "split": ["rl_validation"] * 100,
                "scenario_id": [f"scenario_{i%5}" for i in range(100)],
                "reset_seed": [6521 + (i%5) for i in range(100)],
                "episode_return": [-50.0] * 100,
                "total_cost": [50.0] * 100,
                "preventive_cost": [10.0] * 100,
                "failure_cost": [40.0] * 100,
                "wasted_life_cost": [0.0] * 100,
                "episode_steps": [100] * 100,
            })
            episode_path = tmp_path / "episode_results.parquet"
            df.to_parquet(episode_path)

            # Run validation
            success, errors = validate_threshold_use_equality(episode_path, selected_path)

            # Should fail
            assert success is False, "Validation should have failed"
            assert len(errors) > 0
            assert "None" in errors[0] or "default" in errors[0].lower()


class TestValidatorOracleModeDetection:
    """Test validator correctly detects Oracle-included vs non-Oracle mode."""

    def test_oracle_included_mode_detection(self):
        """Test validator detects Oracle-included mode from selected_thresholds.json."""
        from scripts.validate_m3_artifacts import main
        from pathlib import Path
        import tempfile
        import json
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create selected_thresholds.json WITH oracle_threshold (32 entries = 4 policies x 2 K x 4 regimes)
            selected = {}
            policies = ["age_threshold", "predicted_rul_threshold", "greedy_predicted_rul", "oracle_threshold"]
            k_values = [1, 2]
            regimes = ["failure-light-no-waste", "failure-heavy-no-waste", "failure-light-waste-aware", "failure-heavy-waste-aware"]

            for policy in policies:
                for k in k_values:
                    for regime in regimes:
                        key = f"{policy}_k{k}_{regime}"
                        selected[key] = {
                            "threshold": 1 if policy == "oracle_threshold" else (5 if policy in ["predicted_rul_threshold", "greedy_predicted_rul"] else 25),
                            "k_capacity": k,
                            "cost_regime_id": regime,
                            "mean_total_cost": 10.0,
                            "total_failures": 0,
                            "mean_wasted_life_cost": 0.0,
                            "episode_count": 25,
                        }
            (output_dir / "selected_thresholds.json").write_text(json.dumps(selected))

            # Create the exact 16 canonical scenario-bank identities.
            banks = []
            for split in ["predictor_train", "rl_validation"]:
                for k in k_values:
                    for regime in regimes:
                        banks.append({
                            "split": split,
                            "K": k,
                            "cost_regime_id": regime,
                            "source_path": "test.json",
                            "source_sha256": "a" * 64,
                            "scenario_count": 5,
                            "sorted_scenario_ids_sha256": "b" * 64,
                        })
            (output_dir / "scenario_bank_provenance.json").write_text(
                json.dumps({"scenario_banks": banks})
            )
            (output_dir / "run_provenance.json").write_text('{"run_type": "threshold_tuning", "completed_at": "2026-07-22"}')
            (output_dir / "artifact_manifest.json").write_text("{}")
            (output_dir / "m3_run.log").write_text("log")

            # Create parquet files with 360 tuning candidates where selected ARE winners (lowest cost)
            rows = []
            thresholds_map = {
                "age_threshold": [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300],
                "predicted_rul_threshold": [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100],
                "greedy_predicted_rul": [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100],
                "oracle_threshold": [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50],
            }
            for policy in policies:
                for k in k_values:
                    for regime in regimes:
                        for thresh in thresholds_map[policy]:
                            # Selected threshold has cost 10.0 (best), others higher
                            if (policy == "age_threshold" and thresh == 25) or \
                               (policy in ["predicted_rul_threshold", "greedy_predicted_rul"] and thresh == 5) or \
                               (policy == "oracle_threshold" and thresh == 1):
                                cost = 10.0
                            else:
                                cost = 20.0 + thresh * 0.1
                            rows.append({
                                "policy_family": policy,
                                "threshold": thresh,
                                "k_capacity": k,
                                "cost_regime_id": regime,
                                "mean_total_cost": cost,
                                "total_failures": 0,
                                "mean_wasted_life_cost": 0.0,
                                "episode_count": 25,
                            })
            candidate_summary = pd.DataFrame(rows)
            tuning_episodes = []
            for row in rows:
                for scenario_idx in range(5):
                    for seed_idx in range(5):
                        tuning_episodes.append({
                            "policy_family": row["policy_family"],
                            "threshold": row["threshold"],
                            "k_capacity": row["k_capacity"],
                            "cost_regime_id": row["cost_regime_id"],
                            "scenario_id": f"scenario_{scenario_idx}",
                            "reset_seed": 6521 + seed_idx,
                            "total_cost": row["mean_total_cost"],
                            "preventive_cost": 0.0,
                            "failure_cost": row["mean_total_cost"],
                            "wasted_life_cost": 0.0,
                            "failure_count": 0,
                            "episode_steps": 100,
                            "completed": True,
                        })
            pd.DataFrame(tuning_episodes).to_parquet(
                output_dir / "threshold_search_results.parquet"
            )
            candidate_summary.to_csv(
                output_dir / "threshold_search_summary.csv", index=False
            )

            # Create episode results (2400 episodes for Oracle mode: 6 policies x 2 K x 4 regimes x 2 splits x 5 scenarios x 5 seeds)
            episodes = []
            eval_policies = ["corrective_only", "random_feasible", "age_threshold", "predicted_rul_threshold", "greedy_predicted_rul", "oracle_threshold"]
            for policy in eval_policies:
                for k in k_values:
                    for regime in regimes:
                        for split in ["predictor_train", "rl_validation"]:
                            for scenario_idx in range(5):
                                for seed_idx in range(5):
                                    if policy == "oracle_threshold":
                                        threshold = 1.0
                                    elif policy in ["predicted_rul_threshold", "greedy_predicted_rul"]:
                                        threshold = 5.0
                                    elif policy == "age_threshold":
                                        threshold = 25.0
                                    else:
                                        threshold = None  # corrective_only, random_feasible have no threshold
                                    episodes.append({
                                        "policy_family": policy,
                                        "maintenance_capacity": k,
                                        "cost_regime_id": regime,
                                        "threshold": threshold,
                                        "split": split,
                                        "scenario_id": f"scenario_{scenario_idx}",
                                        "reset_seed": 6521 + seed_idx,
                                        "episode_return": -50.0,
                                        "total_cost": 50.0,
                                        "preventive_cost": 10.0,
                                        "failure_cost": 40.0,
                                        "wasted_life_cost": 0.0,
                                        "episode_steps": 100,
                                    })
            df_episodes = pd.DataFrame(episodes)
            df_episodes.to_parquet(output_dir / "episode_results.parquet")

            (output_dir / "summary_by_policy.json").write_text("[]")
            (output_dir / "summary_by_policy.csv").write_text("")

            # Run validator - should detect Oracle mode and expect 2400 episodes
            exit_code = main(str(output_dir), mode="formal_closeout")
            # With Oracle mode, 2400 episodes expected - should pass
            assert exit_code == 0, "Oracle-included mode validation should pass with 2400 episodes"

    def test_non_oracle_mode_detection(self):
        """Test validator detects non-Oracle mode from selected_thresholds.json."""
        from scripts.validate_m3_artifacts import main
        from pathlib import Path
        import tempfile
        import json
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create selected_thresholds.json WITHOUT oracle_threshold (24 entries = 3 policies x 2 K x 4 regimes)
            selected = {}
            policies = ["age_threshold", "predicted_rul_threshold", "greedy_predicted_rul"]
            k_values = [1, 2]
            regimes = ["failure-light-no-waste", "failure-heavy-no-waste", "failure-light-waste-aware", "failure-heavy-waste-aware"]

            for policy in policies:
                for k in k_values:
                    for regime in regimes:
                        key = f"{policy}_k{k}_{regime}"
                        selected[key] = {
                            "threshold": 5 if policy in ["predicted_rul_threshold", "greedy_predicted_rul"] else 25,
                            "k_capacity": k,
                            "cost_regime_id": regime,
                            "mean_total_cost": 10.0,
                            "total_failures": 0,
                            "mean_wasted_life_cost": 0.0,
                            "episode_count": 25,
                        }
            (output_dir / "selected_thresholds.json").write_text(json.dumps(selected))

            # Create canonical scenario-bank provenance (diagnostic mode does
            # not require the full 16-set, but every record uses canonical
            # fields).
            (output_dir / "scenario_bank_provenance.json").write_text(json.dumps({
                "scenario_banks": [{
                    "split": "rl_validation",
                    "K": 1,
                    "cost_regime_id": "failure-light-no-waste",
                    "source_path": "test.json",
                    "source_sha256": "a" * 64,
                    "scenario_count": 5,
                    "sorted_scenario_ids_sha256": "b" * 64,
                }]
            }))
            (output_dir / "run_provenance.json").write_text('{"run_type": "threshold_tuning", "completed_at": "2026-07-22"}')
            (output_dir / "artifact_manifest.json").write_text("{}")
            (output_dir / "m3_run.log").write_text("log")

            # Create tuning results where selected ARE winners
            rows = []
            thresholds_map = {
                "age_threshold": [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300],
                "predicted_rul_threshold": [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100],
                "greedy_predicted_rul": [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100],
            }
            for policy in policies:
                for k in k_values:
                    for regime in regimes:
                        for thresh in thresholds_map[policy]:
                            if (policy == "age_threshold" and thresh == 25) or \
                               (policy in ["predicted_rul_threshold", "greedy_predicted_rul"] and thresh == 5):
                                cost = 10.0
                            else:
                                cost = 20.0 + thresh * 0.1
                            rows.append({
                                "policy_family": policy,
                                "threshold": thresh,
                                "k_capacity": k,
                                "cost_regime_id": regime,
                                "mean_total_cost": cost,
                                "total_failures": 0,
                                "mean_wasted_life_cost": 0.0,
                                "episode_count": 25,
                            })
            candidate_summary = pd.DataFrame(rows)
            tuning_episodes = []
            for row in rows:
                for scenario_idx in range(5):
                    for seed_idx in range(5):
                        tuning_episodes.append({
                            "policy_family": row["policy_family"],
                            "threshold": row["threshold"],
                            "k_capacity": row["k_capacity"],
                            "cost_regime_id": row["cost_regime_id"],
                            "scenario_id": f"scenario_{scenario_idx}",
                            "reset_seed": 6521 + seed_idx,
                            "total_cost": row["mean_total_cost"],
                            "preventive_cost": 0.0,
                            "failure_cost": row["mean_total_cost"],
                            "wasted_life_cost": 0.0,
                            "failure_count": 0,
                            "episode_steps": 100,
                            "completed": True,
                        })
            pd.DataFrame(tuning_episodes).to_parquet(
                output_dir / "threshold_search_results.parquet"
            )
            candidate_summary.to_csv(
                output_dir / "threshold_search_summary.csv", index=False
            )

            # Create episode results with 2000 episodes (5 policies x 2 K x 4 regimes x 2 splits x 5 scenarios x 5 seeds)
            episodes = []
            eval_policies = ["corrective_only", "random_feasible", "age_threshold", "predicted_rul_threshold", "greedy_predicted_rul"]
            for policy in eval_policies:
                for k in k_values:
                    for regime in regimes:
                        for split in ["predictor_train", "rl_validation"]:
                            for scenario_idx in range(5):
                                for seed_idx in range(5):
                                    if policy in ["predicted_rul_threshold", "greedy_predicted_rul"]:
                                        threshold = 5.0
                                    elif policy == "age_threshold":
                                        threshold = 25.0
                                    else:
                                        threshold = None
                                    episodes.append({
                                        "policy_family": policy,
                                        "maintenance_capacity": k,
                                        "cost_regime_id": regime,
                                        "threshold": threshold,
                                        "split": split,
                                        "scenario_id": f"scenario_{scenario_idx}",
                                        "reset_seed": 6521 + seed_idx,
                                        "episode_return": -50.0,
                                        "total_cost": 50.0,
                                        "preventive_cost": 10.0,
                                        "failure_cost": 40.0,
                                        "wasted_life_cost": 0.0,
                                        "episode_steps": 100,
                                    })
            df_episodes = pd.DataFrame(episodes)
            df_episodes.to_parquet(output_dir / "episode_results.parquet")

            (output_dir / "summary_by_policy.json").write_text("[]")
            (output_dir / "summary_by_policy.csv").write_text("")

            # Run validator - should detect non-Oracle mode and expect 2000 episodes
            exit_code = main(str(output_dir), mode="diagnostic_non_oracle")
            # With non-Oracle mode, 2000 episodes expected - should pass
            assert exit_code == 0, "Non-Oracle mode validation should pass with 2000 episodes"


class TestFormalManifest:
    """Test formal manifest generation and validation."""

    def test_generate_formal_manifest(self):
        """Test generate_formal_manifest creates valid manifest."""
        from src.baselines.artifacts import generate_formal_manifest
        from pathlib import Path
        import tempfile
        import json
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create minimal valid artifacts
            # selected_thresholds.json
            selected = {
                "age_threshold_k1_failure-light-no-waste": {"threshold": 25, "k_capacity": 1, "cost_regime_id": "failure-light-no-waste", "mean_total_cost": 10.0, "total_failures": 0, "mean_wasted_life_cost": 0.0, "episode_count": 25},
                "predicted_rul_threshold_k1_failure-light-no-waste": {"threshold": 5, "k_capacity": 1, "cost_regime_id": "failure-light-no-waste", "mean_total_cost": 10.0, "total_failures": 0, "mean_wasted_life_cost": 0.0, "episode_count": 25},
            }
            (output_dir / "selected_thresholds.json").write_text(json.dumps(selected))

            # threshold_search_results.parquet
            df_tuning = pd.DataFrame({
                "policy_family": ["age_threshold", "predicted_rul_threshold"],
                "threshold": [25, 5],
                "k_capacity": [1, 1],
                "cost_regime_id": ["failure-light-no-waste", "failure-light-no-waste"],
                "mean_total_cost": [10.0, 10.0],
                "total_failures": [0, 0],
                "mean_wasted_life_cost": [0.0, 0.0],
                "episode_count": [25, 25],
            })
            df_tuning.to_parquet(output_dir / "threshold_search_results.parquet")

            # episode_results.parquet
            df_episodes = pd.DataFrame({
                "policy_family": ["age_threshold", "predicted_rul_threshold"],
                "maintenance_capacity": [1, 1],
                "cost_regime_id": ["failure-light-no-waste", "failure-light-no-waste"],
                "threshold": [25.0, 5.0],
                "split": ["rl_validation", "rl_validation"],
                "scenario_id": ["s1", "s1"],
                "reset_seed": [6521, 6521],
                "episode_return": [-50.0, -50.0],
                "total_cost": [50.0, 50.0],
                "preventive_cost": [10.0, 10.0],
                "failure_cost": [40.0, 40.0],
                "wasted_life_cost": [0.0, 0.0],
                "episode_steps": [100, 100],
            })
            df_episodes.to_parquet(output_dir / "episode_results.parquet")

            # scenario_bank_provenance.json (now expects list-shaped or
            # wrapped containing derived_scenario_count)
            (output_dir / "scenario_bank_provenance.json").write_text(json.dumps({
                "scenario_banks": [{
                    "logical_bank_id": "test",
                    "source_path": "test.json",
                    "source_file_size": 100,
                    "source_sha256": "a" * 64,
                    "source_scenario_count": 5,
                    "derived_k": 1,
                    "derived_cost_regime_id": "failure-light-no-waste",
                    "derived_scenario_count": 5,
                    "derived_scenario_ids": ["s1", "s2", "s3", "s4", "s5"],
                    "derived_bank_sha256": "b" * 64,
                }]
            }))

            # run_provenance.json with reset_seeds list
            (output_dir / "run_provenance.json").write_text(json.dumps({
                "run_type": "baseline_evaluation",
                "completed_at": "2026-07-22",
                "reset_seeds": [6521, 6522, 6523, 6524, 6525],
            }))

            # validation_report.json + independent_recomputation.json
            # (now required to exist before manifest generation).
            (output_dir / "validation_report.json").write_text(json.dumps({
                "verdict": "ALL PASSED",
                "mode": "diagnostic_non_oracle",
                "all_errors": [],
                "validated_at": "2026-07-22T12:00:00",
            }))
            (output_dir / "independent_recomputation.json").write_text(json.dumps({
                "recomputed_at": "2026-07-22T12:00:01",
                "tuning_candidates": 2,
                "selected_thresholds_count": 2,
                "evaluation_episodes": 2,
                "scenario_bank_count": 1,
                "reset_seed_count": 5,
                "selected_thresholds_sha256": "x" * 64,
                "validation_report_sha256": "y" * 64,
            }))

            # resolved_config.json (required in all modes now)
            (output_dir / "resolved_config.json").write_text(json.dumps({
                "placeholder": "test config"
            }))

            # Compute SHA256 of selected_thresholds.json
            import hashlib
            selected_path = output_dir / "selected_thresholds.json"
            with open(selected_path, "rb") as f:
                selected_sha256 = hashlib.sha256(f.read()).hexdigest()

            # Generate manifest (new signature: requires explicit mode)
            manifest_path = generate_formal_manifest(
                output_dir,
                mode="diagnostic_non_oracle",
            )

            assert manifest_path.exists()
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            assert manifest["formal_run_id"] == output_dir.name
            # In diagnostic_non_oracle mode without sealed context, commit comes from git
            assert manifest["m3_final_implementation_commit"] != ""
            # Manifest generator now computes SHA256 itself
            assert manifest["selected_thresholds_sha256"] == selected_sha256
            assert manifest["validator_verdict"] == "PENDING"
            assert len(manifest["artifacts"]) > 0

            # Verify all artifacts have required fields
            for artifact in manifest["artifacts"]:
                assert "relative_path" in artifact
                assert "byte_size" in artifact
                assert "sha256" in artifact

    def test_validate_formal_manifest_passes(self):
        """Test validate_formal_manifest passes for valid manifest."""
        from src.baselines.artifacts import generate_formal_manifest, validate_formal_manifest
        from pathlib import Path
        import tempfile
        import json
        import pandas as pd
        import hashlib

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create artifacts
            selected = {"age_threshold_k1_failure-light-no-waste": {"threshold": 25, "k_capacity": 1, "cost_regime_id": "failure-light-no-waste", "mean_total_cost": 10.0}}
            (output_dir / "selected_thresholds.json").write_text(json.dumps(selected))

            df_tuning = pd.DataFrame({"policy_family": ["age_threshold"], "threshold": [25], "k_capacity": [1], "cost_regime_id": ["failure-light-no-waste"], "mean_total_cost": [10.0], "total_failures": [0], "mean_wasted_life_cost": [0.0], "episode_count": [25]})
            df_tuning.to_parquet(output_dir / "threshold_search_results.parquet")

            df_episodes = pd.DataFrame({"policy_family": ["age_threshold"], "maintenance_capacity": [1], "cost_regime_id": ["failure-light-no-waste"], "threshold": [25.0], "split": ["rl_validation"], "scenario_id": ["s1"], "reset_seed": [6521], "episode_return": [-50.0], "total_cost": [50.0], "preventive_cost": [10.0], "failure_cost": [40.0], "wasted_life_cost": [0.0], "episode_steps": [100]})
            df_episodes.to_parquet(output_dir / "episode_results.parquet")

            (output_dir / "resolved_config.json").write_text(json.dumps({"test": "config"}))
            (output_dir / "scenario_bank_provenance.json").write_text(json.dumps({"scenario_banks": []}))
            (output_dir / "run_provenance.json").write_text(json.dumps({"run_type": "baseline_evaluation", "reset_seeds": [6521, 6522, 6523, 6524, 6525]}))
            (output_dir / "validation_report.json").write_text(json.dumps({"verdict": "ALL PASSED", "mode": "diagnostic_non_oracle"}))
            (output_dir / "independent_recomputation.json").write_text(json.dumps({"tuning_candidates": 1, "selected_thresholds_count": 1}))

            generate_formal_manifest(output_dir, mode="diagnostic_non_oracle")

            # Validate
            result = validate_formal_manifest(output_dir)
            assert result["valid"] is True
            assert len(result["errors"]) == 0

    def test_validate_formal_manifest_detects_modified_artifact(self):
        """Test validate_formal_manifest detects when artifact is modified."""
        from src.baselines.artifacts import generate_formal_manifest, validate_formal_manifest
        from pathlib import Path
        import tempfile
        import json
        import pandas as pd
        import hashlib

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create artifacts
            selected = {"age_threshold_k1_failure-light-no-waste": {"threshold": 25, "k_capacity": 1, "cost_regime_id": "failure-light-no-waste", "mean_total_cost": 10.0}}
            (output_dir / "selected_thresholds.json").write_text(json.dumps(selected))

            df_tuning = pd.DataFrame({"policy_family": ["age_threshold"], "threshold": [25], "k_capacity": [1], "cost_regime_id": ["failure-light-no-waste"], "mean_total_cost": [10.0], "total_failures": [0], "mean_wasted_life_cost": [0.0], "episode_count": [25]})
            df_tuning.to_parquet(output_dir / "threshold_search_results.parquet")

            df_episodes = pd.DataFrame({"policy_family": ["age_threshold"], "maintenance_capacity": [1], "cost_regime_id": ["failure-light-no-waste"], "threshold": [25.0], "split": ["rl_validation"], "scenario_id": ["s1"], "reset_seed": [6521], "episode_return": [-50.0], "total_cost": [50.0], "preventive_cost": [10.0], "failure_cost": [40.0], "wasted_life_cost": [0.0], "episode_steps": [100]})
            df_episodes.to_parquet(output_dir / "episode_results.parquet")

            (output_dir / "resolved_config.json").write_text(json.dumps({"test": "config"}))
            (output_dir / "scenario_bank_provenance.json").write_text(json.dumps({"scenario_banks": []}))
            (output_dir / "run_provenance.json").write_text(json.dumps({"run_type": "baseline_evaluation", "reset_seeds": [6521, 6522, 6523, 6524, 6525]}))
            (output_dir / "validation_report.json").write_text(json.dumps({"verdict": "ALL PASSED", "mode": "diagnostic_non_oracle"}))
            (output_dir / "independent_recomputation.json").write_text(json.dumps({"tuning_candidates": 1, "selected_thresholds_count": 1}))

            generate_formal_manifest(output_dir, mode="diagnostic_non_oracle")

            # Modify an artifact after manifest generation
            modified_selected = {"age_threshold_k1_failure-light-no-waste": {"threshold": 100, "k_capacity": 1, "cost_regime_id": "failure-light-no-waste", "mean_total_cost": 10.0}}
            (output_dir / "selected_thresholds.json").write_text(json.dumps(modified_selected))

            # Validate should fail
            result = validate_formal_manifest(output_dir)
            assert result["valid"] is False
            assert len(result["errors"]) > 0
            assert any("SHA256 mismatch" in e or "selected_thresholds" in e for e in result["errors"])
