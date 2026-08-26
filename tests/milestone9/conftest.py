"""Pytest configuration for the M9 point-estimate test layer.

Provides the session-scoped REAL per-seed cache integration fixture, gated
behind ``--runslow`` (or the ``M9_HEAVY`` env var). It regenerates ONE frozen
seed-6521 V2 cache via the frozen ``src/predictors/generate_cache.py`` CLI
(subprocess, per the cache-generation directive) into a pytest-session
temporary directory that is NEVER the formal ``m9_point_caches`` root and NEVER
under ``data/processed/fd001/v2/06_PREDICTIONS``. The cache is generated ONCE
per session and reused by every env-frozen test.

The validation stage does NOT generate all five formal caches; this fixture proves the
cache-binding contract end-to-end for seed 6521 only, and feeds the frozen-env
invariants (2, 3, 8, 9).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run the slow M9 real-CLI cache integration tests (~30s subprocess).",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "m9_slow: real CLI cache generation, ~30s")


def _frozen_sha(seed: int) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    contract = json.loads(
        (repo_root / "docs" / "milestone9" / "M9_POINT_ESTIMATE_CONTRACT.json").read_text()
    )
    return contract["m8_frozen_predictor_checkpoints_mse_control"]["seeds_and_sha256"][
        str(seed)
    ]


@pytest.fixture(scope="session")
def real_seed6521_cache_dir(request, tmp_path_factory):
    """Session-scoped: regenerate the frozen seed-6521 cache via the real
    frozen CLI into a session temp dir; verify the manifest binds the frozen
    SHA256. Skipped unless ``--runslow`` (or ``M9_HEAVY`` set). Returns the
    directory containing ``prediction_cache_manifest_v2.json`` and the parquet.
    """
    if not (request.config.getoption("--runslow") or os.environ.get("M9_HEAVY") == "1"):
        pytest.skip("needs --runslow (or M9_HEAVY=1); real ~30s cache regeneration")

    from src.milestone9.point import cache_prep

    # The env config validator (src/envs/config.py:122-126) requires the string
    # 'v2/06_PREDICTIONS' to be present in prediction_cache_path. To satisfy it
    # from a pytest temp dir, lay the cache under a sub-path containing that
    # identifier; this is NOT the production 06_PREDICTIONS dir (cache_prep's
    # rejector only forbids the anchored production path + formal cache root).
    out_dir = tmp_path_factory.mktemp("m9_real_seed6521_cache")
    out_dir = out_dir / "data" / "processed" / "fd001" / "v2" / "06_PREDICTIONS" / "seed_6521"
    rec = cache_prep.generate_for_seed(6521, out_dir)
    assert rec.returncode == 0
    assert rec.cache_manifest_sha256 == _frozen_sha(6521), (
        f"real cache manifest SHA {rec.cache_manifest_sha256} != "
        f"frozen seed-6521 SHA {_frozen_sha(6521)}"
    )
    assert rec.checkpoint_sha256 == _frozen_sha(6521)
    return Path(str(rec.cache_manifest_path)).parent
