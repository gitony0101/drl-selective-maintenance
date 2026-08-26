"""M9 Point-Estimate — REAL end-to-end CLI cache integration (Step 13).

The mandated gate BEFORE the pilot: invoke the REAL frozen
``src/predictors/generate_cache.py`` CLI for seed 6521 into an isolated
pytest-session temp dir (NEVER the formal ``m9_point_caches`` root, NEVER
``data/.../06_PREDICTIONS``), assert exit 0, and assert the produced cache
manifest's ``checkpoint_sha256`` equals the frozen seed-6521 SHA256.

This proves ``cache_prep.generate_for_seed`` actually binds a cache to the
frozen mse_control checkpoint through the frozen generator's own identity checks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from src.runtime_paths import external_root as _EXTERNAL

pytestmark = pytest.mark.m9_slow


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTAINER_ROOT = _EXTERNAL()


def _frozen_sha(seed: int) -> str:
    contract = json.loads(
        (REPO_ROOT / "docs" / "milestone9" / "M9_POINT_ESTIMATE_CONTRACT.json").read_text()
    )
    return contract["m8_frozen_predictor_checkpoints_mse_control"]["seeds_and_sha256"][
        str(seed)
    ]


def test_real_cli_cache_binds_to_frozen_seed6521(real_seed6521_cache_dir):
    """The real-tagged session fixture already generated + verified the cache.
    Here we additionally assert the manifest is on disk in the temp dir and
    that it was NOT written anywhere forbidden."""
    manifest = real_seed6521_cache_dir / "prediction_cache_manifest_v2.json"
    assert manifest.exists(), f"cache manifest missing: {manifest}"
    m = json.loads(manifest.read_text())
    # The frozen generator writes ``checkpoint_id`` (the SHA256), NOT
    # ``checkpoint_sha256``. Assert the identity the generator actually emits.
    assert m["checkpoint_id"] == _frozen_sha(6521)
    # If the manifest also carries checkpoint_sha256, it must agree; if absent,
    # checkpoint_id is the binding identity.
    if "checkpoint_sha256" in m:
        assert m["checkpoint_sha256"] == _frozen_sha(6521)

    resolved = str((real_seed6521_cache_dir).resolve())
    assert "/m9_point_caches/" not in resolved, (
        f"integration cache must NOT be under the formal cache root: {resolved}"
    )
    # The integration cache must be in pytest temp storage (NOT under the M9
    # point worktree, NOT under the production cache dir). A pytest temp
    # sub-layout MAY legitimately contain 'v2/06_PREDICTIONS' as a path
    # component (the env config validator requires that substring), so we
    # anchor the check to the worktree root, not to the bare '06_PREDICTIONS' token.
    m9_worktree = str(REPO_ROOT)
    assert not resolved.startswith(m9_worktree), (
        f"integration cache must NOT be inside the repository: {resolved}"
    )

    # The parquet cache file itself must be present.
    parquet = real_seed6521_cache_dir / "fd001_prediction_cache_v2.parquet"
    assert parquet.exists(), f"cache parquet missing: {parquet}"
