"""M9 point-estimate exact predictor-seed <-> DDQN-seed pairing.

Paired design: 5 per-seed V2 caches, each
generated from seed_<s>'s frozen mse_control checkpoint, paired 1:1 with the
DDQN run for seed s. This module maps a seed to its cache directory + unique
run_id and validates that a per-seed cache manifest's ``checkpoint_sha256``
equals the frozen SHA256 for that seed (fail-closed on cross-seed or
non-frozen SHA256).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from src.runtime_paths import external_root

from . import manifest


_CONTAINER_ROOT = external_root()
CACHE_ROOT = _CONTAINER_ROOT / "m9_point_caches"

_RUN_ID_PATTERN = "m9_point_mse_control_seed{seed}"


@dataclass(frozen=True)
class CachePairing:
    seed: int
    frozen_checkpoint_sha256: str
    cache_manifest_sha256: str
    match: bool


def run_id_for_seed(seed: int) -> str:
    """Unique --run-id per seed (no run_dir overwrite)."""
    return _RUN_ID_PATTERN.format(seed=seed)


def cache_dir_for_seed(seed: int) -> Path:
    """External git-ignored per-seed cache directory ``m9_point_caches/seed_<s>``."""
    return CACHE_ROOT / f"seed_{seed}"


# The V2 cache is generated into a NESTED directory inside cache_dir_for_seed
# that mirrors the canonical repo layout (data/processed/fd001/v2/06_PREDICTIONS
# /seed_<s>) so the env validator (src/envs/config.py:V2_CACHE_PATH_IDENTIFIER
# == "v2/06_PREDICTIONS") accepts prediction_cache_path. This nested path is
# ALSO the --output-dir cache_prep.generate_for_seed writes the manifest into,
# so the trainer's save_checkpoint reads the same manifest the generator wrote.
_V2_CACHE_SUBPATH = Path("data") / "processed" / "fd001" / "v2" / "06_PREDICTIONS"


def cache_env_path_for_seed(seed: int) -> Path:
    """The env config's ``prediction_cache_path`` for a seed: the NESTED path
    the frozen V2 generator writes the cache into
    (``m9_point_caches/seed_<s>/data/processed/fd001/v2/06_PREDICTIONS/seed_<s>``),
    satisfying the env validator's required ``v2/06_PREDICTIONS`` substr while
    leaving the production repo ``06_PREDICTIONS`` dir untouched. Matches the
    ``--output-dir`` cache_prep generates into."""
    return cache_dir_for_seed(seed) / _V2_CACHE_SUBPATH / f"seed_{seed}"


def _all_frozen_sha256s() -> Dict[int, str]:
    ckpts = manifest.load_frozen_checkpoints()
    return {s: c.sha256 for s, c in ckpts.items()}


def validate_cache_pairing(seed: int, manifest_path: Path) -> CachePairing:
    """Assert the per-seed cache manifest's checkpoint identity == frozen SHA256
    for ``seed``. Fail-closed on cross-seed or non-frozen SHA256.

    The frozen V2 generator writes the manifest with ``checkpoint_id`` (and a
    duplicate ``checkpoint_hash``) as the checkpoint-identity field; it does NOT
    write ``checkpoint_sha256`` (the documented manifest-key discrepancy — see
    ``cache_prep._read_manifest_checkpoint_sha256`` which reads the same
    ``checkpoint_id`` field). Read ``checkpoint_id`` (authoritative), falling
    back to ``checkpoint_sha256`` only if a manifest happens to carry it.

    Raises:
        FileNotFoundError: if the manifest path does not exist.
        ValueError: if the manifest's checkpoint identity is not the frozen
            value for ``seed``.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"cache manifest missing: {manifest_path}")
    data = json.loads(manifest_path.read_text())
    cache_sha = data.get("checkpoint_id") or data.get("checkpoint_sha256")
    if not isinstance(cache_sha, str) or len(cache_sha) != 64:
        raise ValueError(
            f"cache manifest at {manifest_path} has no valid checkpoint_id "
            f"(the frozen generator writes checkpoint_id, not checkpoint_sha256): "
            f"{data.get('checkpoint_id')!r}"
        )
    frozen = _all_frozen_sha256s()
    if seed not in frozen:
        raise ValueError(f"seed {seed} is not in the frozen checkpoint table")
    expected = frozen[seed]
    if cache_sha != expected:
        if cache_sha in frozen.values():
            raise ValueError(
                f"cross-seed mismatch: cache manifest checkpoint identity "
                f"{cache_sha} is a frozen SHA256 but not for seed {seed} "
                f"(expected {expected})"
            )
        raise ValueError(
            f"cache manifest checkpoint identity {cache_sha} is not any of the "
            f"5 frozen mse_control SHA256s (expected {expected} for seed {seed})"
        )
    return CachePairing(
        seed=seed,
        frozen_checkpoint_sha256=expected,
        cache_manifest_sha256=cache_sha,
        match=True,
    )
