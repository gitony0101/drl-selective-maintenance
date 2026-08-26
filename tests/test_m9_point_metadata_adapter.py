"""M9 Point-Estimate — metadata compatibility adapter (Step 1.5).

The frozen V2 cache generator ``src/predictors/generate_cache.py:129`` reads
``metadata['git_commit_hash']`` and compares it to ``checkpoint['git_commit_hash']``.
The original M8 ``predictor_metadata.json`` carries the git commit under the key
``git_commit`` (no ``_hash``). This module authorizes ONE temporary normalized
COPY — adding ``git_commit_hash = metadata['git_commit']`` — to pass to the
generator. The ORIGINAL M8 metadata file MUST remain byte-identical.

Directive invariants verified here:
  - missing metadata['git_commit'] fails closed
  - missing checkpoint['git_commit_hash'] fails closed
  - metadata['git_commit'] != checkpoint['git_commit_hash'] fails closed
  - existing metadata['git_commit_hash'] conflict fails closed
  - unauthorized commit fails closed
  - normalized copy preserves every original field + adds exactly git_commit_hash
  - original metadata file remains byte-identical (SHA256 unchanged before/after)
  - adapter provenance (original SHA256, normalized SHA256, checkpoint SHA256,
    identities, argv, exe, cwd, cache-manifest SHA256) recorded

These are FAST unit tests: they read the real frozen M8 metadata + checkpoint
(no subprocess), and write normalized copies to ``tmp_path`` (never the original
location, never the formal cache root).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_external_assets

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REPO_ROOT = Path(__file__).resolve().parent.parent
from src.milestone9.point import manifest
M8_ROOT = manifest._m8_worktree()


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _orig_metadata_path(seed: int) -> Path:
    return M8_ROOT / "results" / "milestone8_formal" / "mse_control" / f"seed_{seed}" / "predictor_metadata.json"


def _orig_checkpoint_path(seed: int) -> Path:
    return M8_ROOT / "results" / "milestone8_formal" / "mse_control" / f"seed_{seed}" / "best_checkpoint.pt"


def _orig_resolved_config_path(seed: int) -> Path:
    return M8_ROOT / "results" / "milestone8_formal" / "mse_control" / f"seed_{seed}" / "resolved_config.json"


def _authorized_commit() -> str:
    """Recovered from the frozen artifacts themselves (not assumed from text):
    read the seed-6521 checkpoint's git_commit_hash and the original metadata's
    git_commit and require them equal."""
    import torch
    ck = torch.load(_orig_checkpoint_path(6521), map_location="cpu", weights_only=False)
    ck_git = ck.get("git_commit_hash")
    md = json.loads(_orig_metadata_path(6521).read_text())
    md_git = md.get("git_commit")
    assert ck_git == md_git, "authorized-commit recovery discrepancy"
    return ck_git


# ---------------------------------------------------------------------------
# Adapter produces a correct normalized copy and never mutates the original.
# ---------------------------------------------------------------------------


def test_adapter_creates_normalized_copy_with_git_commit_hash(seed_fixture_6521: tuple):
    """The normalized copy carries every original field PLUS git_commit_hash,
    equal to the original git_commit."""
    from src.milestone9.point import metadata_adapter, manifest

    seed, orig_sha_before = seed_fixture_6521
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        rec = metadata_adapter.normalize_for_seed(seed, Path(td))
        norm = json.loads(Path(rec.normalized_metadata_path).read_text())
        orig = json.loads(_orig_metadata_path(seed).read_text())
        # Original fields all present and unchanged.
        for k, v in orig.items():
            assert k in norm and norm[k] == v, f"original field {k!r} not preserved"
        # Exactly one new key added.
        assert "git_commit_hash" in norm
        assert norm["git_commit_hash"] == orig["git_commit"]
        # The original metadata file MUST be byte-identical.
        assert _sha256_file(_orig_metadata_path(seed)) == orig_sha_before


def test_adapter_normalized_copy_is_NOT_under_m8_worktree(seed_fixture_6521: tuple):
    """The normalized copy must never be written inside the upstream checkpoint store."""
    from src.milestone9.point import metadata_adapter

    seed, _ = seed_fixture_6521
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        rec = metadata_adapter.normalize_for_seed(seed, Path(td))
        assert str(rec.normalized_metadata_path).startswith(str(Path(td)))
        assert "upstream-checkpoint-worktree" not in str(rec.normalized_metadata_path)


def test_adapter_provenance_records_original_and_normalized_sha256(seed_fixture_6521: tuple):
    """The provenance record carries original-metadata SHA256, normalized SHA256,
    checkpoint SHA256, the git identities, the authorized commit, and the
    transformation version."""
    from src.milestone9.point import metadata_adapter, manifest

    seed, orig_sha_before = seed_fixture_6521
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        rec = metadata_adapter.normalize_for_seed(seed, Path(td))
        orig = _orig_metadata_path(seed)
        assert rec.seed == seed
        assert rec.original_metadata_sha256 == orig_sha_before
        assert rec.original_metadata_path == str(orig)
        assert rec.normalized_metadata_sha256 == _sha256_file(rec.normalized_metadata_path)
        assert rec.checkpoint_sha256 == manifest.load_frozen_checkpoints()[seed].sha256
        assert rec.original_git_commit_value == "fe52e71dd65a08f1dd29fa4795cfabeefac60864"
        assert rec.alias_git_commit_hash_value == rec.original_git_commit_value
        assert rec.authorized_git_identity == rec.original_git_commit_value
        assert rec.transformation_version  # non-empty string


# ---------------------------------------------------------------------------
# Fail-closed cases.
# ---------------------------------------------------------------------------


def test_adapter_fails_when_metadata_git_commit_missing(tmp_path, monkeypatch):
    """Invariant 18: missing metadata['git_commit'] fails closed."""
    from src.milestone9.point import metadata_adapter

    # Make a fake M8 seed dir with a metadata file lacking git_commit.
    fake_m8 = tmp_path / "fake_m8" / "results" / "milestone8_formal" / "mse_control" / "seed_6521"
    fake_m8.mkdir(parents=True)
    (fake_m8 / "predictor_metadata.json").write_text(json.dumps({"seed": 6521}))
    # Don't need a real checkpoint for this path because the adapter reads the
    # original metadata FIRST and must fail before touching the checkpoint.
    monkeypatch.setattr(
        metadata_adapter.manifest, "load_frozen_checkpoints",
        lambda: {6521: type("C", (), {
            "seed": 6521, "sha256": "x" * 64,
            "checkpoint_path": fake_m8 / "best_checkpoint.pt",
            "resolved_config_path": fake_m8 / "resolved_config.json",
            "predictor_metadata_path": fake_m8 / "predictor_metadata.json",
            "training_history_path": fake_m8 / "training_history.json",
        })()},
    )
    with pytest.raises(ValueError, match="git_commit"):
        metadata_adapter.normalize_for_seed(6521, tmp_path / "norm")


def test_adapter_fails_when_checkpoint_git_commit_hash_missing(tmp_path, monkeypatch):
    """Invariant 19: missing checkpoint['git_commit_hash'] fails closed."""
    import torch
    from src.milestone9.point import metadata_adapter

    fake_m8 = tmp_path / "fake_m8" / "results" / "milestone8_formal" / "mse_control" / "seed_6521"
    fake_m8.mkdir(parents=True)
    (fake_m8 / "predictor_metadata.json").write_text(
        json.dumps({"seed": 6521, "git_commit": "fe52e71dd65a08f1dd29fa4795cfabeefac60864"})
    )
    # Fake checkpoint without git_commit_hash.
    ck = fake_m8 / "best_checkpoint.pt"
    torch.save({"epoch": 22, "seed": 6521}, ck)
    monkeypatch.setattr(
        metadata_adapter.manifest, "load_frozen_checkpoints",
        lambda: {6521: type("C", (), {
            "seed": 6521, "sha256": "x" * 64,
            "checkpoint_path": ck,
            "resolved_config_path": fake_m8 / "resolved_config.json",
            "predictor_metadata_path": fake_m8 / "predictor_metadata.json",
            "training_history_path": fake_m8 / "training_history.json",
        })()},
    )
    with pytest.raises(ValueError, match="git_commit_hash"):
        metadata_adapter.normalize_for_seed(6521, tmp_path / "norm")


def test_adapter_fails_on_identity_mismatch(tmp_path, monkeypatch):
    """Invariant 20: metadata['git_commit'] != checkpoint['git_commit_hash'] fails closed."""
    import torch
    from src.milestone9.point import metadata_adapter

    fake_m8 = tmp_path / "fake_m8" / "results" / "milestone8_formal" / "mse_control" / "seed_6521"
    fake_m8.mkdir(parents=True)
    (fake_m8 / "predictor_metadata.json").write_text(
        json.dumps({"seed": 6521, "git_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"})
    )
    ck = fake_m8 / "best_checkpoint.pt"
    torch.save({"epoch": 22, "git_commit_hash": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}, ck)
    monkeypatch.setattr(
        metadata_adapter.manifest, "load_frozen_checkpoints",
        lambda: {6521: type("C", (), {
            "seed": 6521, "sha256": "x" * 64,
            "checkpoint_path": ck,
            "resolved_config_path": fake_m8 / "resolved_config.json",
            "predictor_metadata_path": fake_m8 / "predictor_metadata.json",
            "training_history_path": fake_m8 / "training_history.json",
        })()},
    )
    with pytest.raises(ValueError, match="mismatch"):
        metadata_adapter.normalize_for_seed(6521, tmp_path / "norm")


def test_adapter_fails_on_existing_conflicting_git_commit_hash(tmp_path, monkeypatch):
    """Invariant 21: original metadata already carries a CONTRADICTORY git_commit_hash -> fail closed."""
    import torch
    from src.milestone9.point import metadata_adapter

    fake_m8 = tmp_path / "fake_m8" / "results" / "milestone8_formal" / "mse_control" / "seed_6521"
    fake_m8.mkdir(parents=True)
    conflicting = "9999999999999999999999999999999999999999"
    (fake_m8 / "predictor_metadata.json").write_text(json.dumps({
        "seed": 6521,
        "git_commit": "fe52e71dd65a08f1dd29fa4795cfabeefac60864",
        "git_commit_hash": conflicting,
    }))
    ck = fake_m8 / "best_checkpoint.pt"
    torch.save({"epoch": 22, "git_commit_hash": "fe52e71dd65a08f1dd29fa4795cfabeefac60864"}, ck)
    monkeypatch.setattr(
        metadata_adapter.manifest, "load_frozen_checkpoints",
        lambda: {6521: type("C", (), {
            "seed": 6521, "sha256": "x" * 64,
            "checkpoint_path": ck,
            "resolved_config_path": fake_m8 / "resolved_config.json",
            "predictor_metadata_path": fake_m8 / "predictor_metadata.json",
            "training_history_path": fake_m8 / "training_history.json",
        })()},
    )
    with pytest.raises(ValueError, match="conflict|contradict"):
        metadata_adapter.normalize_for_seed(6521, tmp_path / "norm")


def test_adapter_fails_on_unauthorized_commit(tmp_path, monkeypatch):
    """Invariant 22: even if metadata==checkpoint, a non-authorized commit fails closed.
    The authorized commit is fe52e71… (recovered from the frozen artifacts)."""
    import torch
    from src.milestone9.point import metadata_adapter

    fake_m8 = tmp_path / "fake_m8" / "results" / "milestone8_formal" / "mse_control" / "seed_6521"
    fake_m8.mkdir(parents=True)
    bad = "cccccccccccccccccccccccccccccccccccccccc"
    (fake_m8 / "predictor_metadata.json").write_text(json.dumps({"seed": 6521, "git_commit": bad}))
    ck = fake_m8 / "best_checkpoint.pt"
    torch.save({"epoch": 22, "git_commit_hash": bad}, ck)
    monkeypatch.setattr(
        metadata_adapter.manifest, "load_frozen_checkpoints",
        lambda: {6521: type("C", (), {
            "seed": 6521, "sha256": "x" * 64,
            "checkpoint_path": ck,
            "resolved_config_path": fake_m8 / "resolved_config.json",
            "predictor_metadata_path": fake_m8 / "predictor_metadata.json",
            "training_history_path": fake_m8 / "training_history.json",
        })()},
    )
    with pytest.raises(ValueError, match="unauthorized|not the authorized"):
        metadata_adapter.normalize_for_seed(6521, tmp_path / "norm")


def test_adapter_preserves_byte_identity_of_original_after_normalize(seed_fixture_6521: tuple):
    """Invariant 17: the ORIGINAL M8 metadata file is byte-identical before vs after."""
    from src.milestone9.point import metadata_adapter

    seed, orig_sha_before = seed_fixture_6521
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        metadata_adapter.normalize_for_seed(seed, Path(td))
        orig_sha_after = _sha256_file(_orig_metadata_path(seed))
        assert orig_sha_before == orig_sha_after


def test_adapter_provenance_hashes_reproduce(seed_fixture_6521: tuple):
    """Invariant 23: regenerating the normalized copy reproduces the same
    normalized-metadata SHA256 (deterministic transformation)."""
    from src.milestone9.point import metadata_adapter

    seed, _ = seed_fixture_6521
    import tempfile
    with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
        r1 = metadata_adapter.normalize_for_seed(seed, Path(td1))
        r2 = metadata_adapter.normalize_for_seed(seed, Path(td2))
        assert r1.normalized_metadata_sha256 == r2.normalized_metadata_sha256
        assert r1.original_metadata_sha256 == r2.original_metadata_sha256


# ---------------------------------------------------------------------------
# And: the documented authorized commit equals the recovered-from-artifacts one,
# so the adapter does NOT silently trust a hardcoded literal.
# ---------------------------------------------------------------------------


def test_authorized_commit_matches_artifacts():
    """The adapter's authorized commit must be the one recovered from the frozen
    artifacts (checkpoint.git_commit_hash == metadata.git_commit), not assumed."""
    from src.milestone9.point import metadata_adapter

    assert metadata_adapter.AUTHORIZED_GIT_COMMIT == _authorized_commit()


# ---------------------------------------------------------------------------
# resolved_config normalization (the second sidecar the generator reads raw).
# ---------------------------------------------------------------------------


def test_adapter_creates_normalized_resolved_config_with_root_identity(seed_fixture_all: tuple):
    """The normalized resolved_config copy carries root-level
    seed/sequence_length/rul_cap sourced from the frozen checkpoint's own
    config, and the original effective_config nesting is preserved."""
    from src.milestone9.point import metadata_adapter

    seed, _ = seed_fixture_all
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        rec = metadata_adapter.normalize_for_seed(seed, Path(td))
        norm_rc = json.loads(Path(rec.normalized_resolved_config_path).read_text())
        orig_rc = json.loads(_orig_resolved_config_path(seed).read_text())
        # Every original field preserved (including effective_config nesting).
        for k, v in orig_rc.items():
            assert k in norm_rc and norm_rc[k] == v, f"original rc field {k!r} not preserved"
        # Root identity fields present and equal to checkpoint.config values.
        for f in ["seed", "sequence_length", "rul_cap"]:
            assert f in norm_rc
            assert norm_rc[f] == orig_rc["effective_config"][f]


def test_adapter_root_identity_values_match_checkpoint_config(seed_fixture_all: tuple):
    """The flat root identity values in BOTH normalized copies equal the frozen
    checkpoint's embedded config (the SHA256-bound authoritative source)."""
    import torch
    from src.milestone9.point import metadata_adapter

    seed, _ = seed_fixture_all
    ck = torch.load(_orig_checkpoint_path(seed), map_location="cpu", weights_only=False)
    ck_cfg = ck["config"]
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        rec = metadata_adapter.normalize_for_seed(seed, Path(td))
        norm_md = json.loads(Path(rec.normalized_metadata_path).read_text())
        norm_rc = json.loads(Path(rec.normalized_resolved_config_path).read_text())
        for f in ["sequence_length", "rul_cap"]:
            assert norm_md[f] == ck_cfg[f]
        for f in ["seed", "sequence_length", "rul_cap"]:
            assert norm_rc[f] == ck_cfg[f]
        # And the provenance field_sources record both checkpoint & effective_config.
        for f in ["seed", "sequence_length", "rul_cap"]:
            fs = rec.root_identity_field_sources[f]
            assert fs["checkpoint_config_value"] == ck_cfg[f]
            assert fs["effective_config_value"] == ck_cfg[f]


def test_adapter_fails_when_checkpoint_config_disagrees_with_effective_config(tmp_path, monkeypatch):
    """Invariant: checkpoint.config identity != effective_config nesting fails closed."""
    import torch
    from src.milestone9.point import metadata_adapter

    fake_m8 = tmp_path / "fake_m8" / "results" / "milestone8_formal" / "mse_control" / "seed_6521"
    fake_m8.mkdir(parents=True)
    auth = "fe52e71dd65a08f1dd29fa4795cfabeefac60864"
    (fake_m8 / "predictor_metadata.json").write_text(json.dumps({"seed": 6521, "git_commit": auth}))
    (fake_m8 / "resolved_config.json").write_text(json.dumps({
        "effective_config": {"seed": 6521, "sequence_length": 50, "rul_cap": 125},
    }))
    # Checkpoint config with a CONTRADICTORY rul_cap (130 != 125).
    ck = fake_m8 / "best_checkpoint.pt"
    torch.save({
        "epoch": 22, "git_commit_hash": auth,
        "config": {"seed": 6521, "sequence_length": 50, "rul_cap": 130, "n_features": 24,
                   "normalizer_id": "fd001_normalizer_v2", "feature_schema_id": "fd001_feature_schema_v1"},
    }, ck)
    monkeypatch.setattr(
        metadata_adapter.manifest, "load_frozen_checkpoints",
        lambda: {6521: type("C", (), {
            "seed": 6521, "sha256": "x" * 64,
            "checkpoint_path": ck,
            "resolved_config_path": fake_m8 / "resolved_config.json",
            "predictor_metadata_path": fake_m8 / "predictor_metadata.json",
            "training_history_path": fake_m8 / "training_history.json",
        })()},
    )
    with pytest.raises(ValueError, match="root identity|disagreement|effective_config"):
        metadata_adapter.normalize_for_seed(6521, tmp_path / "norm")


def test_adapter_normalized_resolved_config_is_NOT_under_m8_worktree(seed_fixture_all: tuple):
    """Invariant: the normalized resolved_config copy must never live in the upstream checkpoint store."""
    from src.milestone9.point import metadata_adapter

    seed, _ = seed_fixture_all
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        rec = metadata_adapter.normalize_for_seed(seed, Path(td))
        assert str(rec.normalized_resolved_config_path).startswith(str(Path(td)))
        assert "upstream-checkpoint-worktree" not in str(rec.normalized_resolved_config_path)


def test_adapter_preserves_byte_identity_of_original_resolved_config(seed_fixture_all: tuple):
    """Invariant 17 extended: the ORIGINAL M8 resolved_config.json is byte-identical
    before vs after normalization."""
    from src.milestone9.point import metadata_adapter

    seed, _ = seed_fixture_all
    orig_sha_before = _sha256_file(_orig_resolved_config_path(seed))
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        metadata_adapter.normalize_for_seed(seed, Path(td))
        orig_sha_after = _sha256_file(_orig_resolved_config_path(seed))
        assert orig_sha_before == orig_sha_after


def test_adapter_provenance_records_resolved_config_sha256s(seed_fixture_all: tuple):
    """The provenance record carries original + normalized resolved_config SHA256s."""
    from src.milestone9.point import metadata_adapter

    seed, _ = seed_fixture_all
    orig_rc_sha = _sha256_file(_orig_resolved_config_path(seed))
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        rec = metadata_adapter.normalize_for_seed(seed, Path(td))
        assert rec.original_resolved_config_sha256 == orig_rc_sha
        assert rec.normalized_resolved_config_sha256 == _sha256_file(Path(rec.normalized_resolved_config_path))


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(params=[6521])
def seed_fixture_6521(request):
    seed = request.param
    orig_sha = _sha256_file(_orig_metadata_path(seed))
    return seed, orig_sha


@pytest.fixture(params=[6521, 6522, 6523, 6524, 6525])
def seed_fixture_all(request):
    seed = request.param
    return seed, None
