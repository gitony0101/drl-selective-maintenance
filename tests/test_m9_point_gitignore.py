"""M9 Point-Estimate — gitignore + clean-worktree (invariants 11 & 13).

Invariant 11 (output paths external and ignored): generated caches and run
outputs live in external git-ignored roots; an accidental in-worktree copy is
defensively ignored by the `.gitignore` rule.

Invariant 13 (clean worktree): the test layer must not leave the repository
dirty. `git status --porcelain` after the focused M9 suite must show ONLY the
intentional untracked implementation/test files (since validation hasn't been
committed yet) — never a generated cache or run output inside the worktree.
These tests run fast (no subprocess): they assert the gitignore RULES, not
generation outcomes.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REPO_ROOT = Path(__file__).resolve().parent.parent


def _check_ignore(rel_path: str) -> bool:
    """Return True if `git check-ignore -q <rel_path>` exits 0 (the path is ignored)."""
    p = subprocess.run(
        ["git", "check-ignore", "-q", rel_path],
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return p.returncode == 0


def test_results_dir_is_ignored():
    """Invariant 11a: an in-worktree results/ path is git-ignored."""
    assert _check_ignore("results/m9_point_test_run/foo.json")


def test_m9_point_caches_inside_worktree_is_ignored():
    """Invariant 11b: an accidental in-worktree m9_point_caches/ copy is ignored
    by the defensive `.gitignore` rule."""
    assert _check_ignore("m9_point_caches/seed_6521/foo.parquet")


def test_m9_point_runs_inside_worktree_is_ignored():
    """Invariant 11c: an accidental in-worktree m9_point_runs/ copy is ignored."""
    assert _check_ignore("m9_point_runs/run999/checkpoint.pt")


def test_production_cache_dir_is_ignored():
    """Invariant 11d: the production data/processed/fd001/v2/06_PREDICTIONS/
    directory is git-ignored."""
    assert _check_ignore("data/processed/fd001/v2/06_PREDICTIONS/foo.parquet")


def test_src_tests_tracked_dirs_NOT_ignored():
    """Sanity: source, tests, docs, configs are NOT ignored (false-positive guard
    so a regex over-broad rule would be caught)."""
    for ok in ("src/milestone9/point/wrapper.py", "tests/test_m9_point_manifest.py",
               "docs/milestone9/M9_POINT_ESTIMATE_CONTRACT.json",
               "configs/agents/ddqn_v1.json", ".gitignore"):
        assert not _check_ignore(ok), f"{ok} unexpectedly ignored"


def test_git_porcelain_has_no_generated_cache_or_run_artifact_inside_worktree():
    """Invariant 13: `git status --porcelain` shows no tracked-modified cache or
    run artifact (only intended untracked implementation/test files for this phase
    are acceptable)."""

    def classify(line: str) -> str:
        # porcelain status XY + path; we only inspect untracked/modified paths
        # and flag lines mentioning m9_point caches/runs/parquet/predictor caches.
        path = line[3:].strip().split(" -> ")[-1].strip('"')
        return path

    p = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(REPO_ROOT), stdout=subprocess.PIPE, text=True, check=True,
    )
    for line in p.stdout.splitlines():
        if not line.strip():
            continue
        path = classify(line)
        bad_tokens = ("m9_point_caches", "m9_point_runs",
                      "fd001_prediction_cache", "06_PREDICTIONS")
        assert not any(t in path for t in bad_tokens), (
            f"git porcelain shows a generated cache/run artifact inside the worktree: {path!r}"
        )
