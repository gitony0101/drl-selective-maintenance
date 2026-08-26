"""M9 point-estimate per-seed cache preparation via the frozen CLI.

This module is the ONLY sanctioned path to regenerate a per-seed V2 prediction
cache for the point-estimate M9. It invokes the frozen
``src/predictors/generate_cache.py`` as a SUBPROCESS (per the provenance-capture
directive): sys.executable + explicit argv, shell=False, fixed worktree cwd,
captured stdout/stderr, check=True.

Pre-flight gate: each per-seed ``resolved_config.json`` must carry
``effective_config.loss.type == "mse"``. The frozen generator does NOT enforce
non-LinEx identity (it checks only seed/sequence_length/rul_cap at
``generate_cache.py:175-179``); the wrapper adds this explicit check BEFORE
invoking the generator, so a ``linex``-typed checkpoint can never produce an M9
cache.

The synthesized ``training_summary.json`` carries the SINGLE field
``{"best_epoch": <checkpoint's own embedded epoch>}`` — the generator compares
this to ``checkpoint.get("epoch")`` (generate_cache.py:136-140). Note the
metadata's ``best_epoch`` is one-indexed and differs from the checkpoint's
0-indexed ``epoch`` across all 5 seeds; the generator compares summary↔checkpoint
(not summary↔metadata), so the summary MUST carry the checkpoint's own epoch.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from src.runtime_paths import external_root

from . import manifest
from . import metadata_adapter


_REPO_ROOT = manifest.REPO_ROOT
_GENERATOR_REL = "src/predictors/generate_cache.py"
_GENERATOR = _REPO_ROOT / _GENERATOR_REL

# The formal cache root (production, git-ignored). Tests MUST NOT write here.
CACHE_ROOT = external_root() / "m9_point_caches"
# The production V2 prediction-cache directory inside the repo. Never overwritten
# by an M9 cache generation (cludes writing to it or any subdir of it).
PRODUCTION_V2_CACHE_DIR = _REPO_ROOT / "data" / "processed" / "fd001" / "v2" / "06_PREDICTIONS"
_CACHE_MANIFEST_NAME = "prediction_cache_manifest_v2.json"


@dataclass
class CacheGenRecord:
    """Provenance record for one per-seed cache generation subprocess."""

    seed: int
    command: List[str]
    executable: str
    cwd: str
    shell: bool
    check: bool
    returncode: int
    stdout: str
    stderr: str
    checkpoint_sha256: str
    cache_manifest_sha256: str
    cache_manifest_path: str
    training_summary: Dict[str, Any] = field(default_factory=dict)
    adapter_provenance: Dict[str, Any] = field(default_factory=dict)


def _read_checkpoint_epoch(checkpoint_path: Path) -> int:
    import torch
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    epoch = ck.get("epoch")
    if epoch is None:
        raise ValueError(f"checkpoint has no 'epoch' field: {checkpoint_path}")
    return int(epoch)


def write_synthesized_training_summary(seed: int, output_dir: Path) -> Path:
    """Write the 1-field training_summary.json under ``output_dir`` and return
    its path. ``best_epoch`` is the checkpoint's OWN embedded epoch (the value
    the frozen generator compares against). NOT synthetic data."""
    ckpts = manifest.load_frozen_checkpoints()
    if seed not in ckpts:
        raise ValueError(f"seed {seed} is not in the frozen checkpoint table")
    ck = ckpts[seed]
    epoch = _read_checkpoint_epoch(ck.checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps({"best_epoch": epoch}))
    return summary_path


def _assert_loss_type_mse(seed: int) -> None:
    """Gate: the per-seed resolved_config.json MUST carry loss.type == 'mse'.
    Fail-closed before invoking the generator (invariant 6)."""
    ckpts = manifest.load_frozen_checkpoints()
    if seed not in ckpts:
        raise ValueError(f"seed {seed} is not in the frozen checkpoint table")
    rc = ckpts[seed].resolved_config_path
    if not rc.exists():
        raise FileNotFoundError(f"seed {seed}: resolved_config.json missing: {rc}")
    cfg = json.loads(rc.read_text())
    eff = cfg.get("effective_config", {})
    loss_type = eff.get("loss", {}).get("type")
    if loss_type != "mse":
        raise ValueError(
            f"seed {seed}: per-seed resolved_config loss.type={loss_type!r} — "
            f"only 'mse' is permitted for the point-estimate M9 (no LinEx)"
        )


def _read_manifest_checkpoint_sha256(manifest_path: Path) -> str:
    """Read the frozen generator's written checkpoint identity from the cache
    manifest. The generator emits ``checkpoint_id`` (and ``checkpoint_hash``)
    — NOT ``checkpoint_sha256`` — so read ``checkpoint_id`` and require a valid
    64-char SHA256 there."""
    data = json.loads(manifest_path.read_text())
    sha = data.get("checkpoint_id")
    # Some manifests may also carry checkpoint_sha256; prefer checkpoint_id as
    # the authoritative field the generator actually writes.
    if not isinstance(sha, str) or len(sha) != 64:
        raise ValueError(
            f"cache manifest {manifest_path} has no valid checkpoint_id "
            f"(frozen generator writes checkpoint_id, not checkpoint_sha256): {sha!r}"
        )
    return sha


def generate_for_seed(seed: int, output_dir: Path) -> CacheGenRecord:
    """Generate the per-seed V2 cache into ``output_dir`` via the frozen CLI.

    ``output_dir`` MUST NOT be the formal cache root or data/.../06_PREDICTIONS:
    this entry point rejects both so a test can never trample a formal cache.

    Returns a CacheGenRecord with the full command/exe/cwd/exit/logs/SHA provenance.
    """
    _assert_loss_type_mse(seed)
    output_dir = Path(output_dir)
    _reject_forbidden_output_dirs(output_dir)

    ckpts = manifest.load_frozen_checkpoints()
    if seed not in ckpts:
        raise ValueError(f"seed {seed} is not in the frozen checkpoint table")
    ck = ckpts[seed]
    summary_path = write_synthesized_training_summary(seed, output_dir)
    # Produce the temporary normalized metadata COPY (adapter adds
    # git_commit_hash = metadata['git_commit']; originals stay byte-identical).
    # The copy lives under output_dir (outside the M8 worktree, outside the
    # formal cache root / 06_PREDICTIONS).
    adapter_rec = metadata_adapter.normalize_for_seed(seed, output_dir)
    normalized_metadata_path = Path(adapter_rec.normalized_metadata_path)

    argv = [
        sys.executable,
        str(_GENERATOR),
        "--checkpoint-path", str(ck.checkpoint_path),
        "--predictor-metadata-path", str(normalized_metadata_path),
        "--training-summary-path", str(summary_path),
        "--resolved-config-path", str(adapter_rec.normalized_resolved_config_path_as_path),
        "--output-dir", str(output_dir),
        "--seed", str(seed),
        "--overwrite-v2",
    ]
    completed = subprocess.run(
        argv,
        shell=False,
        cwd=str(_REPO_ROOT),
        capture_output=True,
        check=True,
        text=True,
    )
    manifest_path = output_dir / _CACHE_MANIFEST_NAME
    cache_sha = _read_manifest_checkpoint_sha256(manifest_path)
    if cache_sha != ck.sha256:
        raise ValueError(
            f"seed {seed}: generated cache manifest checkpoint_sha256 {cache_sha} "
            f"!= frozen checkpoint SHA256 {ck.sha256}"
        )
    return CacheGenRecord(
        seed=seed,
        command=argv,
        executable=sys.executable,
        cwd=str(_REPO_ROOT),
        shell=False,
        check=True,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        checkpoint_sha256=ck.sha256,
        cache_manifest_sha256=cache_sha,
        cache_manifest_path=str(manifest_path),
        training_summary=json.loads(summary_path.read_text()),
        adapter_provenance=adapter_rec.to_manifest_dict(),
    )


def _reject_forbidden_output_dirs(output_dir: Path) -> None:
    """Guard a cache generation target. Refuses:
      1. The production V2 prediction-cache directory
         ``data/processed/fd001/v2/06_PREDICTIONS`` or any descendant (the
         canonical repo cache dir — never overwritten, even by production).
      2. The cache root ``m9_point_caches`` itself (no dumping files directly
         into the root — every cache MUST live under a per-seed subdir).
      3. A per-seed target subdir that already contains the cache manifest
         (NO SILENT OVERWRITE: the caller passes --overwrite-v2, so an existing
         manifest means the prior cache would be clobbered; the caller must
         remove the existing dir explicitly first — matching the
         assert_run_dir_absent discipline for training).

    Production per-seed caches are PERMITTED: they live at
    ``m9_point_caches/seed_<s>/...v2/06_PREDICTIONS/seed_<s>/`` (the env config
    validator requires the string ``v2/06_PREDICTIONS`` in
    ``prediction_cache_path``; the generation ``--output-dir`` IS that path). A
    pytest tmp dir laid under ``.../v2/06_PREDICTIONS/...`` (NOT the production
    dir) is permitted; only the anchored production dir and its descendants
    are rejected.
    """
    resolved = output_dir.resolve()

    # (1) Production V2 dir and descendants — never, even for production.
    prod = PRODUCTION_V2_CACHE_DIR.resolve()
    if resolved == prod or resolved.is_relative_to(prod):
        raise ValueError(
            f"refusing to generate cache into the production V2 cache "
            f"directory (or a subdir of it): {output_dir}"
        )

    # (2) The cache root itself — caches must live in a per-seed subdir.
    root = CACHE_ROOT.resolve()
    if resolved == root:
        raise ValueError(
            f"refusing to generate cache directly into the cache root "
            f"(caches must live under a per-seed subdir): {output_dir}"
        )

    # (3) No silent overwrite: a target that already contains the cache
    # manifest is refused (the caller must remove it explicitly first).
    if resolved.exists() and (resolved / _CACHE_MANIFEST_NAME).exists():
        raise ValueError(
            f"refusing to silently overwrite an existing cache manifest "
            f"(remove the existing cache dir explicitly first): {output_dir}"
        )

