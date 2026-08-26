"""M9 point-estimate driver -- staging runtime-config writer regression test.

formal runs failed at the staging-config write with FileNotFoundError
because the driver wrote ``<runs_root>/<phase>/<run_id>_runtime_config.json``
without ensuring ``<runs_root>/<phase>/`` exists. The pilot only worked because
an earlier attempt had pre-created ``<runs_root>/pilot/``.

This test pins the fix: ``_write_staging_runtime_config`` ensures the parent
dir exists before writing the sibling staging config.
"""

from __future__ import annotations

import json
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytestmark = pytest.mark.requires_external_assets

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_write_staging_runtime_config_creates_parent_dir(tmp_path):
    """The staging-config writer creates the phase runs dir (run_dir.parent)
    when it does not yet exist, so the first formal/pilot run for a phase does
    not FileNotFoundError on the staging write."""
    from scripts.run_m9_point_estimate import _write_staging_runtime_config

    # A fresh phase root that does NOT exist yet -- mirrors the formal-phase
    # first-run scenario (m9_point_runs/formal/ absent before seed 6521).
    runs_root = tmp_path / "m9_point_runs"
    phase = "formal"
    run_id = "m9_point_mse_control_seed6521"
    run_dir = runs_root / phase / run_id
    assert not run_dir.parent.exists(), "test setup: phase dir must not exist yet"

    staging = _write_staging_runtime_config(run_dir, run_id, {"seed": 6521})

    assert staging == run_dir.parent / f"{run_id}_runtime_config.json"
    assert staging.exists()
    assert json.loads(staging.read_text()) == {"seed": 6521}
    # run_dir itself must NOT be created (the trainer creates it; the wrapper
    # asserts run_dir is absent before training).
    assert not run_dir.exists(), (
        "staging writer must not create run_dir -- the trainer's mkdir must be "
        "the first to create it, and run_training asserts run_dir is absent"
    )
