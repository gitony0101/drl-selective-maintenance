"""
M9 Point-Estimate — exact predictor-seed <-> DDQN-seed pairing (Step 3).

Invariant 7: for each seed s, the per-seed V2 prediction cache's manifest
``checkpoint_sha256`` MUST equal the frozen mse_control ``best_checkpoint.pt``
SHA256 for s. Pairing is 1:1 (Interpretation B): seed 6521's cache binds to
seed 6521's frozen checkpoint, and ONLY to it. Any cross-seed mismatch is a
hard block (fail-closed).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_external_assets
from src.runtime_paths import external_root as _EXTERNAL

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACT_JSON = REPO_ROOT / "docs" / "milestone9" / "M9_POINT_ESTIMATE_CONTRACT.json"
CONTAINER_ROOT = _EXTERNAL()
CACHE_ROOT = CONTAINER_ROOT / "m9_point_caches"

EXPECTED_RUN_ID_PATTERN = "m9_point_mse_control_seed{seed}"


def _frozen_sha_for_seed(seed: int) -> str:
    contract = json.loads(CONTRACT_JSON.read_text())
    return contract["m8_frozen_predictor_checkpoints_mse_control"]["seeds_and_sha256"][str(seed)]


def _stub_manifest(tmp_path: Path, seed: int, checkpoint_sha256: str) -> Path:
    """Write a valid-looking cache manifest carrying the given checkpoint_sha256,
    to exercise pairing logic without regenerating a real cache."""
    cache_dir = tmp_path / f"seed_{seed}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "prediction_cache_manifest_v2.json"
    manifest = {
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_id": checkpoint_sha256,
        "checkpoint_hash": checkpoint_sha256,
        "predictor_id": f"m8_formal_mse_control_seed_{seed}",
        "seed": seed,
        "cache_hash": "stub",
    }
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


def test_pairing_run_id_unique_per_seed():
    """Each seed gets a unique --run-id (no run_dir overwrite)."""
    from src.milestone9.point.pairing import run_id_for_seed

    ids = {run_id_for_seed(s) for s in (6521, 6522, 6523, 6524, 6525)}
    assert ids == {
        "m9_point_mse_control_seed6521",
        "m9_point_mse_control_seed6522",
        "m9_point_mse_control_seed6523",
        "m9_point_mse_control_seed6524",
        "m9_point_mse_control_seed6525",
    }
    assert len(ids) == 5


def test_pairing_cache_dir_per_seed_external():
    """Each per-seed cache dir lives under the external git-ignored cache root
    and is uniquely named seed_<s>."""
    from src.milestone9.point.pairing import cache_dir_for_seed

    for seed in (6521, 6522, 6523, 6524, 6525):
        d = cache_dir_for_seed(seed)
        assert d == CACHE_ROOT / f"seed_{seed}"
    dirs = {cache_dir_for_seed(s) for s in (6521, 6522, 6523, 6524, 6525)}
    assert len(dirs) == 5


def test_pairing_cache_env_path_for_seed_is_nested_with_v2_token():
    """The env config's ``prediction_cache_path`` for a seed is the NESTED path
    the frozen V2 generator writes the cache into:
    ``m9_point_caches/seed_<s>/data/processed/fd001/v2/06_PREDICTIONS/seed_<s>``.
    The env validator (src/envs/config.py:V2_CACHE_PATH_IDENTIFIER) requires
    the substr ``v2/06_PREDICTIONS`` in ``prediction_cache_path``; this nested
    path satisfies it while leaving the production repo ``06_PREDICTIONS`` dir
    untouched. It ALSO matches the ``--output-dir`` cache_prep generates into,
    so the manifest the trainer reads at save_checkpoint time is the one the
    generator wrote."""
    from src.milestone9.point.pairing import cache_env_path_for_seed, cache_dir_for_seed

    for seed in (6521, 6522, 6523, 6524, 6525):
        env_path = cache_env_path_for_seed(seed)
        # The env path is nested INSIDE the per-seed cache dir.
        assert env_path == (
            cache_dir_for_seed(seed)
            / "data" / "processed" / "fd001" / "v2" / "06_PREDICTIONS" / f"seed_{seed}"
        )
        # Required by the env validator.
        assert "v2/06_PREDICTIONS" in str(env_path)
        # NOT the production repo V2 dir.
        prod = REPO_ROOT / "data" / "processed" / "fd001" / "v2" / "06_PREDICTIONS"
        assert env_path != prod


def test_pairing_validate_cache_manifest_matches_frozen(tmp_path):
    """Invariant 7: a cache manifest carrying seed 6521's frozen SHA256 validates
    for seed 6521."""
    from src.milestone9.point.pairing import validate_cache_pairing

    seed = 6521
    mp = _stub_manifest(tmp_path, seed, _frozen_sha_for_seed(seed))
    pairing = validate_cache_pairing(seed, mp)
    assert pairing.seed == seed
    assert pairing.frozen_checkpoint_sha256 == _frozen_sha_for_seed(seed)
    assert pairing.cache_manifest_sha256 == _frozen_sha_for_seed(seed)
    assert pairing.match is True


def test_pairing_rejects_cross_seed_mismatch(tmp_path):
    """Invariant 7 (fail-closed): a cache manifest carrying seed 6522's SHA256
    presented for seed 6521 MUST raise (cross-seed mismatch)."""
    from src.milestone9.point.pairing import validate_cache_pairing

    seed = 6521
    wrong_sha = _frozen_sha_for_seed(6522)
    mp = _stub_manifest(tmp_path, seed, wrong_sha)
    with pytest.raises(ValueError, match="cross-seed mismatch|checkpoint identity"):
        validate_cache_pairing(seed, mp)


def test_pairing_rejects_non_frozen_sha256(tmp_path):
    """Invariant 7 (fail-closed): a cache manifest carrying a SHA256 that is not
    any of the 5 frozen values MUST raise."""
    from src.milestone9.point.pairing import validate_cache_pairing

    seed = 6521
    mp = _stub_manifest(
        tmp_path, seed, "ade3688496de7672367fcb58bbcba384f6835f81fe2c89d7ce9f88eeebe5b2b7"
    )
    with pytest.raises(ValueError, match="frozen"):
        validate_cache_pairing(seed, mp)


def test_pairing_reads_checkpoint_id_when_checkpoint_sha256_absent(tmp_path):
    """REGRESSION: the frozen V2 generator writes the manifest with
    ``checkpoint_id`` (and ``checkpoint_hash``) as the checkpoint-identity field;
    it does NOT write ``checkpoint_sha256``. The earlier
    ``validate_cache_pairing`` read ``checkpoint_sha256`` and got None on a REAL
    manifest (the FROZEN manifest at m9_point_caches/seed_6521/.../manifest has
    KEYS checkpoint_id + checkpoint_hash but NO checkpoint_sha256). The pairing
    validator MUST read ``checkpoint_id`` (the authoritative field the generator
    writes), falling back to ``checkpoint_sha256`` only if present. A manifest
    carrying ONLY ``checkpoint_id`` (mirroring the real generator output) for
    seed 6521's frozen SHA must validate pair==True."""
    from src.milestone9.point.pairing import validate_cache_pairing

    seed = 6521
    frozen_sha = _frozen_sha_for_seed(seed)
    # A manifest mirroring the REAL generator output: checkpoint_id present,
    # checkpoint_sha256 ABSENT.
    cache_dir = tmp_path / f"seed_{seed}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    mp = cache_dir / "prediction_cache_manifest_v2.json"
    real_like = {
        "checkpoint_id": frozen_sha,
        "checkpoint_hash": frozen_sha,
        "predictor_id": f"m8_formal_mse_control_seed_{seed}",
        "seed": seed,
        "cache_hash": "real",
        "best_epoch": 99,
    }
    mp.write_text(json.dumps(real_like))
    pairing = validate_cache_pairing(seed, mp)
    assert pairing.match is True
    assert pairing.cache_manifest_sha256 == frozen_sha

