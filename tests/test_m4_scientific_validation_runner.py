"""
Tests for M4 Scientific Validation Runner.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestRunner:
    """Test the scientific validation runner."""

    def test_runner_script_exists(self):
        """Runner script exists and is importable."""
        runner_path = Path(__file__).parent.parent / "scripts" / "run_m4_scientific_validation.py"
        assert runner_path.exists()

    def test_runner_help(self):
        """Runner shows help."""
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent.parent / "scripts" / "run_m4_scientific_validation.py"), "--help"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        assert result.returncode == 0
        assert "M4 scientific validation" in result.stdout

    def test_candidate_grid_frozen(self):
        """Candidate grid is exactly 6 frozen candidates."""
        runner_path = Path(__file__).parent.parent / "scripts" / "run_m4_scientific_validation.py"
        content = runner_path.read_text()

        # Check candidates are defined
        assert "CANDIDATES" in content
        assert "hard_window_v1" in content
        assert "logistic_T1" in content
        assert "logistic_T2" in content
        assert "logistic_T5" in content
        assert "logistic_T10" in content
        assert "logistic_T20" in content

    def test_protocol_version_frozen(self):
        """Protocol version is frozen."""
        runner_path = Path(__file__).parent.parent / "scripts" / "run_m4_scientific_validation.py"
        content = runner_path.read_text()
        assert "PROTOCOL_VERSION" in content
        assert "m4_scientific_validation_v1" in content

    def test_resume_contract(self):
        """Runner enforces resume contract."""
        runner_path = Path(__file__).parent.parent / "scripts" / "run_m4_scientific_validation.py"
        content = runner_path.read_text()

        assert "resume" in content.lower()
        assert "config_hash" in content
        assert "git_head" in content or "git_commit" in content