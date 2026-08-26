"""
M9 Point-Estimate — manifest invariants (Step 1).

Covers contract invariants:
  1. all five frozen mse_control checkpoint SHA256s match the contract table
  4. predictor params frozen (mse_control family; no LinEx)
  6. no LinEx checkpoint loaded (each per-seed resolved_config carries loss.type=="mse")

The manifest loader is the single source of truth used by the rest of the M9
point-estimate stack; it MUST read the frozen checkpoint SHA256s from the
committed contract JSON sidecar (NOT hardcode them) and verify every checkpoint
exists on disk with the expected byte count.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_external_assets

# Make src.* importable as in the project's other tests.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACT_JSON = REPO_ROOT / "docs" / "milestone9" / "M9_POINT_ESTIMATE_CONTRACT.json"

EXPECTED_SEEDS = (6521, 6522, 6523, 6524, 6525)
EXPECTED_BYTES = 2_271_313


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_contract_json_sidecar_exists_and_loads():
    """The manifest source-of-truth sidecar must exist and parse."""
    assert CONTRACT_JSON.exists(), f"contract JSON missing: {CONTRACT_JSON}"
    data = json.loads(CONTRACT_JSON.read_text())
    block = data["m8_frozen_predictor_checkpoints_mse_control"]
    assert block["selected_family"] == "mse_control"
    assert block["per_seed_size_bytes"] == EXPECTED_BYTES
    assert set(block["seeds_and_sha256"].keys()) == {str(s) for s in EXPECTED_SEEDS}


def test_manifest_loads_frozen_checkpoint_sha256s_from_contract():
    """Invariant 1: the manifest exposes exactly the 5 frozen SHA256s from the
    contract sidecar (loaded, not hardcoded)."""
    from src.milestone9.point.manifest import load_frozen_checkpoints

    ckpts = load_frozen_checkpoints()
    assert set(ckpts.keys()) == set(EXPECTED_SEEDS)
    contract = json.loads(CONTRACT_JSON.read_text())
    table = contract["m8_frozen_predictor_checkpoints_mse_control"]["seeds_and_sha256"]
    for seed in EXPECTED_SEEDS:
        assert ckpts[seed].sha256 == table[str(seed)], f"seed {seed} sha256 mismatch"
        assert ckpts[seed].sha256 == table[str(seed)]
        assert len(ckpts[seed].sha256) == 64


def test_manifest_checkpoint_paths_resolve_to_m8_worktree():
    """Each frozen checkpoint path points at the upstream checkpoint store's mse_control dir."""
    from src.milestone9.point.manifest import load_frozen_checkpoints, _m8_worktree

    ckpts = load_frozen_checkpoints()
    m8_root = _m8_worktree()
    for seed in EXPECTED_SEEDS:
        p = ckpts[seed].checkpoint_path
        assert p.exists(), f"seed {seed}: checkpoint missing on disk: {p}"
        assert p.parent == (
            m8_root
            / "results"
            / "milestone8_formal"
            / "mse_control"
            / f"seed_{seed}"
        ), f"seed {seed}: unexpected parent dir: {p.parent}"
        assert p.name == "best_checkpoint.pt"


def test_manifest_5_checkpoints_on_disk_match_frozen_sha256():
    """Invariant 1 (on-disk): each .pt file's recomputed SHA256 == frozen value."""
    from src.milestone9.point.manifest import load_frozen_checkpoints

    ckpts = load_frozen_checkpoints()
    for seed in EXPECTED_SEEDS:
        ck = ckpts[seed]
        assert ck.checkpoint_path.stat().st_size == EXPECTED_BYTES, (
            f"seed {seed}: size {ck.checkpoint_path.stat().st_size} != {EXPECTED_BYTES}"
        )
        assert _sha256(ck.checkpoint_path) == ck.sha256, (
            f"seed {seed}: on-disk SHA256 != frozen SHA256"
        )


def test_manifest_per_seed_resolved_config_loss_type_is_mse():
    """Invariant 6 (no LinEx): each per-seed resolved_config.json carries
    effective_config.loss.type == "mse" — the actual LinEx-rejection surface.
    (model.type is always "mlp"; the mse_control family is encoded by loss.type.)"""
    from src.milestone9.point.manifest import load_frozen_checkpoints

    ckpts = load_frozen_checkpoints()
    for seed in EXPECTED_SEEDS:
        rc = ckpts[seed].resolved_config_path
        assert rc.exists(), f"seed {seed}: resolved_config.json missing: {rc}"
        cfg = json.loads(rc.read_text())
        eff = cfg["effective_config"]
        assert eff["loss"]["type"] == "mse", (
            f"seed {seed}: loss.type={eff['loss']['type']!r} expected 'mse'"
        )
        assert eff["loss"]["type"] != "linex"
        assert eff["model"]["type"] == "mlp"


def test_manifest_per_seed_predictor_id_carries_mse_control():
    """Invariant 4 (frozen mse_control family): each predictor_metadata.json
    carries predictor_id of the form m8_formal_mse_control_seed_<s> and
    condition=='mse_control'."""
    from src.milestone9.point.manifest import load_frozen_checkpoints

    ckpts = load_frozen_checkpoints()
    for seed in EXPECTED_SEEDS:
        md_path = ckpts[seed].predictor_metadata_path
        assert md_path.exists(), f"seed {seed}: predictor_metadata.json missing"
        md = json.loads(md_path.read_text())
        assert md["predictor_id"] == f"m8_formal_mse_control_seed_{seed}"
        assert md["condition"] == "mse_control"