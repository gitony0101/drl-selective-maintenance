"""
M9 Point-Estimate — per-seed cache preparation via the frozen CLI (Step 8).

cache_prep.generate_for_seed(seed, output_dir) synthesizes the 1-field
training_summary.json from the checkpoint's own ``epoch`` field and invokes the
frozen ``src/predictors/generate_cache.py`` as a SUBPROCESS (per the
provenance-capture directive): sys.executable + explicit argv, shell=False,
fixed worktree cwd, captured stdout/stderr, check=True. It gates the call on
the per-seed resolved_config loss.type == "mse" (the frozen generator does NOT
enforce non-LinEx identity — it only checks seed/sequence_length/rul_cap), and
records the exact command/exe/cwd/exit/logs + checkpoint SHA256 + cache manifest
SHA256 into a provenance record.

This test suite MOCKS the subprocess boundary (fast unit tests). The
real end-to-end CLI invocation is exercised by the session-scoped integration
fixture in tests/test_m9_point_real_cli_cache_integration.py (REQUIRED green
before the pilot; writes only to a temp git-ignored dir, never to the formal
m9_point_caches/).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_external_assets
from src.runtime_paths import external_root as _EXTERNAL

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REPO_ROOT = Path(__file__).resolve().parent.parent
M8_ROOT = _EXTERNAL() / "drl-selective-maintenance-m8"
WORKTREE_ROOT = REPO_ROOT
CONTRACT_JSON = REPO_ROOT / "docs" / "milestone9" / "M9_POINT_ESTIMATE_CONTRACT.json"


def _frozen_sha(seed: int) -> str:
    return json.loads(CONTRACT_JSON.read_text())[
        "m8_frozen_predictor_checkpoints_mse_control"
    ]["seeds_and_sha256"][str(seed)]


class _FakeCompleted:
    def __init__(self, stdout: str = "ok", stderr: str = ""):
        self.returncode = 0
        self.stdout = stdout
        self.stderr = stderr


def test_cache_prep_synthesizes_training_summary_from_checkpoint_epoch(tmp_path, monkeypatch):
    """The synthesized training_summary.json's ``best_epoch`` equals the
    checkpoint's OWN embedded ``epoch`` (NOT the metadata's best_epoch, which
    is off-by-one across all 5 seeds)."""
    import torch
    from src.milestone9.point import cache_prep

    ck = torch.load(
        M8_ROOT / "results" / "milestone8_formal" / "mse_control" / "seed_6521"
        / "best_checkpoint.pt",
        map_location="cpu", weights_only=False,
    )
    expected_epoch = ck["epoch"]

    out_dir = tmp_path / "seed_6521"
    # Stop before the real subprocess: just synthesize the file and assert it.
    summary_path = cache_prep.write_synthesized_training_summary(6521, out_dir)
    data = json.loads(summary_path.read_text())
    assert data == {"best_epoch": expected_epoch}
    assert summary_path == out_dir / "training_summary.json"


def test_cache_prep_gates_on_loss_type_mse_before_subprocess(tmp_path, monkeypatch):
    """The wrapper-level pre-check rejects a resolved_config with loss.type != 'mse'
    BEFORE invoking the generator subprocess (invariant 6)."""
    from src.milestone9.point import cache_prep

    # Point the loader at a synthetic resolved_config carrying linex.
    fake_rc_dir = tmp_path / "fake_seed"
    fake_rc_dir.mkdir()
    (fake_rc_dir / "resolved_config.json").write_text(json.dumps({
        "effective_config": {"loss": {"type": "linex"}, "model": {"type": "mlp"}}
    }))

    monkeypatch.setattr(
        cache_prep.manifest, "load_frozen_checkpoints",
        lambda: {6521: type("C", (), {
            "seed": 6521, "sha256": _frozen_sha(6521),
            "checkpoint_path": M8_ROOT / "x.pt",
            "resolved_config_path": fake_rc_dir / "resolved_config.json",
            "predictor_metadata_path": fake_rc_dir / "pm.json",
            "training_history_path": fake_rc_dir / "th.json",
        })()},
    )
    with pytest.raises(ValueError, match="loss.*type.*mse|linex"):
        cache_prep.generate_for_seed(6521, tmp_path / "out")


def test_cache_prep_builds_exact_subprocess_argv(tmp_path, monkeypatch):
    """The subprocess argv is sys.executable + the script path + the documented
    flags, with shell=False, cwd==worktree root, check=True, captured output."""
    from src.milestone9.point import cache_prep

    captured = {}

    def fake_run(argv, **kw):
        captured["argv"] = argv
        captured["kw"] = kw
        # Mimic the generator writing a manifest so the post-run SHA read works.
        out_flag_idx = argv.index("--output-dir")
        od = Path(argv[out_flag_idx + 1])
        od.mkdir(parents=True, exist_ok=True)
        (od / "prediction_cache_manifest_v2.json").write_text(json.dumps({
            "checkpoint_sha256": _frozen_sha(6521),
            "checkpoint_id": _frozen_sha(6521),
        }))
        return _FakeCompleted()

    monkeypatch.setattr(subprocess, "run", fake_run)
    out_dir = tmp_path / "seed_6521_out"
    rec = cache_prep.generate_for_seed(6521, out_dir)

    argv = captured["argv"]
    assert argv[0] == sys.executable
    # The frozen generator script path, relative to the worktree root.
    assert argv[1].endswith("src/predictors/generate_cache.py")
    assert "--checkpoint-path" in argv
    assert "--predictor-metadata-path" in argv
    assert "--training-summary-path" in argv
    assert "--resolved-config-path" in argv
    assert "--output-dir" in argv
    assert "--seed" in argv and "6521" in argv
    assert "--overwrite-v2" in argv  # store_true flag present as a bare flag

    kw = captured["kw"]
    assert kw["shell"] is False
    assert kw["cwd"] == str(WORKTREE_ROOT)
    assert kw["check"] is True
    assert kw["capture_output"] is True

    # argv must NOT pass any override of --data-dir into the formal cache paths
    # and must NOT target the PRODUCTION data/processed/.../06_PREDICTIONS dir.
    out_flag_idx = argv.index("--output-dir")
    out_val = argv[out_flag_idx + 1]
    assert "data/processed/fd001/v2/06_PREDICTIONS" not in out_val
    assert out_val.startswith(str(out_dir))


def test_cache_prep_records_provenance_after_generation(tmp_path, monkeypatch):
    """The returned record carries the exact command, exe, cwd, exit code,
    logs, checkpoint SHA256, and (later) the generated cache manifest SHA256."""
    from src.milestone9.point import cache_prep

    def fake_run(argv, **kw):
        # Mimic the generator also writing a manifest, so the SHA can be read.
        out_flag = argv.index("--output-dir")
        out_dir = Path(argv[out_flag + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "prediction_cache_manifest_v2.json").write_text(json.dumps({
            "checkpoint_sha256": _frozen_sha(6521),
            "checkpoint_id": _frozen_sha(6521),
        }))
        return _FakeCompleted(stdout="generated", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    rec = cache_prep.generate_for_seed(6521, tmp_path / "out")
    assert rec.returncode == 0
    assert rec.executable == sys.executable
    assert rec.cwd == str(WORKTREE_ROOT)
    assert rec.shell is False
    assert isinstance(rec.command, list) and rec.command[0] == sys.executable
    assert rec.checkpoint_sha256 == _frozen_sha(6521)
    assert rec.cache_manifest_sha256 == _frozen_sha(6521)
    assert rec.stdout == "generated"
    assert rec.stderr == ""
    # The synthesized training_summary is recorded alongside the command.
    assert "best_epoch" in rec.training_summary


def test_cache_prep_rejects_production_v2_dir_and_descendants(tmp_path):
    """The rejector ALWAYS forbids the production V2 cache dir and its
    descendants (the canonical repo cache dir — never overwritten), even
    after the a documented relaxation that lets production write into the formal
    cache root."""
    from src.milestone9.point import cache_prep
    import pytest
    prod = cache_prep.PRODUCTION_V2_CACHE_DIR
    with pytest.raises(ValueError, match="production V2"):
        cache_prep._reject_forbidden_output_dirs(prod)
    with pytest.raises(ValueError, match="production V2"):
        cache_prep._reject_forbidden_output_dirs(prod / "seed_6521")
    # A pytest tmp dir laid under v2/06_PREDICTIONS/... containing the bare
    # token is NOT the production dir (anchored reject only) — permitted.
    notprod = tmp_path / "data" / "processed" / "fd001" / "v2" / "06_PREDICTIONS" / "seed_6521"
    notprod.mkdir(parents=True)
    cache_prep._reject_forbidden_output_dirs(notprod)  # must not raise


def test_cache_prep_rejects_writing_directly_into_cache_root(tmp_path):
    """Writing files directly into the cache root m9_point_caches/ (no seed
    subdir) is rejected — caches must live under a per-seed subdir."""
    from src.milestone9.point import cache_prep
    import pytest
    with pytest.raises(ValueError, match="cache root"):
        cache_prep._reject_forbidden_output_dirs(cache_prep.CACHE_ROOT)


def test_cache_prep_silent_overwrite_rejected_when_manifest_present(tmp_path):
    """No silent overwrite: if a target cache subdir already contains the
    cache manifest, generation is refused (the evidence would be clobbered
    by --overwrite-v2). The caller must remove the existing dir explicitly
    first — matching the assert_run_dir_absent discipline for training."""
    from src.milestone9.point import cache_prep
    import pytest
    target = tmp_path / "seed_6521_formal"
    target.mkdir()
    (target / "prediction_cache_manifest_v2.json").write_text('{"checkpoint_id":"x"}')
    with pytest.raises(ValueError, match="already exists|overwrite|manifest"):
        cache_prep._reject_forbidden_output_dirs(target)


def test_cache_prep_allows_empty_seed_subdir_under_cache_root(tmp_path):
    """Production writes into m9_point_caches/seed_<s>/ are PERMITTED (pilot
    relaxation): the protected anchors are only the production V2 dir and the
    cache root itself (no dumping files into the root). An empty per-seed
    subdir, or a subdir without a manifest, is allowed for production.

    NOTE: this test MUST NOT point at the real cache root ``m9_point_caches`` —
    the real seed-6521/6522 caches already exist there from the prior pilot
    cache generation, so the no-silent-overwrite guard would correctly fire.
    Use a tmp_path mirror of the per-seed structure instead."""
    from src.milestone9.point import cache_prep
    fake_seed = tmp_path / "fake_cache_root" / "seed_6521" / "data" / "processed" \
        / "fd001" / "v2" / "06_PREDICTIONS" / "seed_6521"
    fake_seed.mkdir(parents=True)
    # Must not raise (empty target mirroring the real structure, NOT the real
    # production V2 dir and NOT the real cache root).
    cache_prep._reject_forbidden_output_dirs(fake_seed)
