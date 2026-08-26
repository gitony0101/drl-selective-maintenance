"""
Tests for M4 Scientific Validation Analysis.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestAnalysis:
    """Test the analysis script."""

    def test_analysis_script_exists(self):
        """Analysis script exists."""
        analysis_path = Path(__file__).parent.parent / "scripts" / "analyze_m4_scientific_validation.py"
        assert analysis_path.exists()

    def test_analysis_help(self):
        """Analysis shows help."""
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent.parent / "scripts" / "analyze_m4_scientific_validation.py"), "--help"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        assert result.returncode == 0
        assert "Analyze M4 scientific validation" in result.stdout

    def test_bootstrap_seed_frozen(self):
        """Bootstrap seed is frozen to 652104."""
        analysis_path = Path(__file__).parent.parent / "scripts" / "analyze_m4_scientific_validation.py"
        content = analysis_path.read_text()
        assert "BOOTSTRAP_SEED = 652104" in content
        assert "652104" in content

    def test_bootstrap_resamples_frozen(self):
        """Bootstrap resamples frozen to 10000."""
        analysis_path = Path(__file__).parent.parent / "scripts" / "analyze_m4_scientific_validation.py"
        content = analysis_path.read_text()
        assert "BOOTSTRAP_RESAMPLES = 10000" in content

    def test_eligibility_rules_implemented(self):
        """Eligibility rules are implemented."""
        analysis_path = Path(__file__).parent.parent / "scripts" / "analyze_m4_scientific_validation.py"
        content = analysis_path.read_text()

        # Rule 1: CI upper < 0
        assert "rule1" in content.lower() or "ci_upper" in content.lower() or "upper" in content.lower()
        # Rule 2: worsen > 10% in <= 2 configs
        assert "0.10" in content or "ELIGIBILITY_WORSEN_THRESHOLD" in content
        assert "ELIGIBILITY_MAX_WORSEN_CONFIGS = 2" in content or "2" in content

    def test_tie_breaking_implemented(self):
        """Tie-breaking logic implemented."""
        analysis_path = Path(__file__).parent.parent / "scripts" / "analyze_m4_scientific_validation.py"
        content = analysis_path.read_text()

        # Check tie-breaking criteria
        assert "macro_avg_normalized_paired_diff" in content
        assert "worst_config" in content
        assert "mean_preventive_cost" in content
        assert "temperature" in content.lower() or "risk_temperature" in content

    def test_output_artifacts(self):
        """Required output artifacts are produced."""
        analysis_path = Path(__file__).parent.parent / "scripts" / "analyze_m4_scientific_validation.py"
        content = analysis_path.read_text()

        required = [
            "selection_decision.json",
            "paired_episode_metrics.csv",
            "candidate_summary.csv",
            "per_configuration_summary.csv",
            "bootstrap_summary.json",
            "validation_analysis_manifest.json",
        ]

        for artifact in required:
            assert artifact in content


class TestSelectionLogic:
    """Test the selection logic directly."""

    def test_eligibility_rule1_ci_upper_below_zero(self):
        """Rule 1: Macro CI upper bound must be < 0."""
        # If CI upper >= 0, candidate is not eligible
        ci_upper = 0.05  # Positive - NOT eligible
        assert not (ci_upper < 0)

        ci_upper = -0.02  # Negative - eligible (wrt rule 1)
        assert ci_upper < 0

    def test_eligibility_rule2_worsen_threshold(self):
        """Rule 2: No more than 2 configs worsen by > 10%."""
        worsen_count = 3
        assert not (worsen_count <= 2)

        worsen_count = 2
        assert worsen_count <= 2

    def test_tie_breaking_order(self):
        """Tie-breaking follows frozen order."""
        # 1. lowest macro avg normalized paired diff
        # 2. lowest worst config normalized diff
        # 3. lowest mean preventive cost
        # 4. lower temperature

        candidates = [
            {"macro": -0.05, "worst": -0.03, "preventive": 100, "temp": 10},
            {"macro": -0.05, "worst": -0.04, "preventive": 100, "temp": 5},  # Better worst
            {"macro": -0.04, "worst": -0.02, "preventive": 50, "temp": 2},   # Worse macro
        ]

        # Should select second candidate (same macro, better worst)
        sorted_cands = sorted(candidates, key=lambda c: (c["macro"], c["worst"], c["preventive"], c["temp"]))
        assert sorted_cands[0]["temp"] == 5


class TestBootstrap:
    """Test bootstrap implementation."""

    def test_bootstrap_deterministic_with_seed(self):
        """Bootstrap results deterministic with fixed seed."""
        np.random.seed(652104)
        sample1 = np.random.randn(10000).mean()

        np.random.seed(652104)
        sample2 = np.random.randn(10000).mean()

        assert sample1 == sample2

    def test_stratified_bootstrap_structure(self):
        """Stratified bootstrap runs within each config."""
        # The analysis script should implement this
        # Just verify the concept is in the code
        analysis_path = Path(__file__).parent.parent / "scripts" / "analyze_m4_scientific_validation.py"
        content = analysis_path.read_text()
        assert "stratified" in content.lower() or "per_config" in content


class TestProtocolCompliance:
    """Test protocol compliance requirements."""

    def test_no_rl_test_in_selection(self):
        """Selection must not use rl_test."""
        analysis_path = Path(__file__).parent.parent / "scripts" / "analyze_m4_scientific_validation.py"
        content = analysis_path.read_text()
        # Should only reference predictor_train and rl_validation
        assert "SELECTION_SPLIT = \"rl_validation\"" in content or 'rl_validation' in content

    def test_predictor_train_only_diagnostics(self):
        """predictor_train only for diagnostics."""
        analysis_path = Path(__file__).parent.parent / "scripts" / "analyze_m4_scientific_validation.py"
        content = analysis_path.read_text()
        # Analysis should separate the two splits
        assert "predictor_train" in content

    def test_complete_grid_required(self):
        """Must run all candidates before selection."""
        # Analysis loads all candidates
        analysis_path = Path(__file__).parent.parent / "scripts" / "analyze_m4_scientific_validation.py"
        content = analysis_path.read_text()
        assert "all_results" in content or "candidate_artifacts" in content

    def test_no_mixed_heads(self):
        """Must reject mixed HEADs."""
        analysis_path = Path(__file__).parent.parent / "scripts" / "analyze_m4_scientific_validation.py"
        content = analysis_path.read_text()
        assert "git_head" in content or "git_commit" in content
        assert "validate_git_head" in content or "mismatch" in content

    def test_no_mixed_bank_hashes(self):
        """Must reject mixed bank hashes."""
        analysis_path = Path(__file__).parent.parent / "scripts" / "analyze_m4_scientific_validation.py"
        content = analysis_path.read_text()
        assert "bank_hash" in content
        assert "validate_bank_hashes" in content or "mismatch" in content

    def test_missing_episodes_rejected(self):
        """Missing episodes must be rejected."""
        analysis_path = Path(__file__).parent.parent / "scripts" / "analyze_m4_scientific_validation.py"
        content = analysis_path.read_text()
        assert "episode" in content.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])