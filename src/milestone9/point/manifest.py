"""M9 point-estimate frozen-predictor manifest.

Single source of truth for the 5 frozen mse_control checkpoints, loaded from
the committed contract JSON sidecar
(``docs/milestone9/M9_POINT_ESTIMATE_CONTRACT.json``) — NOT hardcoded — and
verified against on-disk files at load time.

This is the only permitted predictor set for the point-estimate M9 fallback.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_JSON = REPO_ROOT / "docs" / "milestone9" / "M9_POINT_ESTIMATE_CONTRACT.json"

def _m8_worktree() -> Path:
    """Resolve M8 worktree path via DRL_EXTERNAL_ROOT."""
    from src.runtime_paths import external_root as _EXTERNAL
    return _EXTERNAL() / "drl-selective-maintenance-m8"

_FROZEN_SEEDS = (6521, 6522, 6523, 6524, 6525)


@dataclass(frozen=True)
class FrozenCheckpoint:
    """One frozen mse_control best_checkpoint.pt and its sidecars."""

    seed: int
    sha256: str
    checkpoint_path: Path
    resolved_config_path: Path
    predictor_metadata_path: Path
    training_history_path: Path


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _contract_block() -> Dict:
    if not CONTRACT_JSON.exists():
        raise FileNotFoundError(f"M9 point-estimate contract JSON missing: {CONTRACT_JSON}")
    data = json.loads(CONTRACT_JSON.read_text())
    block = data["m8_frozen_predictor_checkpoints_mse_control"]
    if block["selected_family"] != "mse_control":
        raise ValueError(
            f"contract selected_family != mse_control: {block['selected_family']!r}"
        )
    return block


def load_frozen_checkpoints() -> Dict[int, FrozenCheckpoint]:
    """Load and on-disk-verify the 5 frozen mse_control checkpoints.

    Raises FileNotFoundError if any checkpoint or required sidecar is missing on
    disk, and ValueError if the on-disk SHA256 does not match the frozen value.
    The SHA256s are read from the contract JSON sidecar (never hardcoded).
    """
    block = _contract_block()
    m8_worktree = _m8_worktree()
    size_bytes = int(block["per_seed_size_bytes"])
    table = block["seeds_and_sha256"]

    checkpoints: Dict[int, FrozenCheckpoint] = {}
    for seed in _FROZEN_SEEDS:
        key = str(seed)
        if key not in table:
            raise ValueError(f"contract table missing seed {seed}")
        frozen_sha = table[key]
        seed_dir = (
            m8_worktree
            / "results"
            / "milestone8_formal"
            / "mse_control"
            / f"seed_{seed}"
        )
        ckpt_path = seed_dir / "best_checkpoint.pt"
        rc_path = seed_dir / "resolved_config.json"
        md_path = seed_dir / "predictor_metadata.json"
        th_path = seed_dir / "training_history.json"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"seed {seed}: checkpoint missing on disk: {ckpt_path}")
        actual = ckpt_path.stat().st_size
        if actual != size_bytes:
            raise ValueError(
                f"seed {seed}: checkpoint size {actual} != frozen {size_bytes}"
            )
        on_disk_sha = _compute_sha256(ckpt_path)
        if on_disk_sha != frozen_sha:
            raise ValueError(
                f"seed {seed}: on-disk SHA256 {on_disk_sha} != frozen {frozen_sha}"
            )
        checkpoints[seed] = FrozenCheckpoint(
            seed=seed,
            sha256=frozen_sha,
            checkpoint_path=ckpt_path,
            resolved_config_path=rc_path,
            predictor_metadata_path=md_path,
            training_history_path=th_path,
        )
    return checkpoints