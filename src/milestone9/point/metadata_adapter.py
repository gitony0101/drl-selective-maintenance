"""M9 Point-Estimate — temporary sidecar compatibility adapter.

The frozen V2 cache generator ``src/predictors/generate_cache.py`` validates
predictor identity by comparing three sources (checkpoint, metadata,
resolved_config) on a handful of flat root-level fields. The frozen M8 sidecars
were serialized in the project's standard ``effective_config`` nesting and so
omit some of those flat root-level fields:

  1. ``git_commit_hash`` — the generator (line 129) reads
     ``metadata.get('git_commit_hash', 'unknown')`` and compares to the
     checkpoint's ``git_commit_hash``. The M8 metadata carries the SAME commit
     under the key ``git_commit`` (no ``_hash``), so the generator saw
     ``unknown`` and refused to bind the cache for all 5 seeds.

  2. ``seed`` / ``sequence_length`` / ``rul_cap`` — the generator (lines
     157-160) reads ``resolved_config.get('seed' | 'sequence_length' |
     'rul_cap')`` at the ROOT and compares to the checkpoint's embedded
     ``config``. The M8 resolved_config nests those under ``effective_config``
     (root-level values are ``None``), so the comparison failed for all 5
     seeds. The M8 metadata also carries only ``seed`` at root (missing
     ``sequence_length`` / ``rul_cap``).

This adapter authorizes ONE temporary normalized COPY of EACH sidecar:

  - metadata copy: original fields + ``git_commit_hash = metadata['git_commit']``
    and root-level ``sequence_length`` / ``rul_cap`` sourced from the frozen
    checkpoint's own ``config`` (the SHA256-bound authoritative values).
  - resolved_config copy: original fields + root-level ``seed`` /
    ``sequence_length`` / ``rul_cap`` sourced from the frozen checkpoint's own
    ``config``.

The flat values are taken from the checkpoint's embedded ``config`` (which is
inside the SHA256-checkpointed file and therefore frozen), and are first
verified to AGREE with the ``effective_config`` nesting in the resolved_config
and the existing root ``seed`` in the metadata — so the adapter never
introduces a value that the frozen artifacts do not already entail; it only
re-serializes them at the root the generator reads.

The ORIGINAL M8 sidecar files are NEVER mutated — byte-identical before and
after. Every identity + SHA256 is recorded into the adapter provenance
manifest so the normalization is auditable, non-replay, and non-relaxing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import manifest


_TRANSFORMATION_VERSION = "m9-sidecar-adapter-v1"
# Authorized predictor commit: recovered once from the frozen artifacts
# (checkpoint.git_commit_hash == metadata.git_commit) — NOT a hardcoded guess.
# Tests assert this equals the artifacts.
_AUTHORIZED_GIT_COMMIT_RESOLVED: Optional[str] = None

# Root-level identity fields the generator reads from both metadata and
# resolved_config (generate_cache.py:144-179). These are the ONLY compared
# fields; the others are read into the identity dicts but never compared.
_ROOT_IDENTITY_FIELDS = ("seed", "sequence_length", "rul_cap")


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _resolve_authorized_commit() -> str:
    """Recover the single authorized predictor commit from the frozen artifacts:
    the seed-6521 checkpoint's ``git_commit_hash`` must equal the seed-6521
    original metadata's ``git_commit``. This value is then the authorized
    identity every per-seed metadata+checkpoint pair must agree on.
    """
    global _AUTHORIZED_GIT_COMMIT_RESOLVED
    if _AUTHORIZED_GIT_COMMIT_RESOLVED is not None:
        return _AUTHORIZED_GIT_COMMIT_RESOLVED
    import torch
    ckpts = manifest.load_frozen_checkpoints()
    ck6521 = ckpts[6521]
    ck = torch.load(ck6521.checkpoint_path, map_location="cpu", weights_only=False)
    ck_git = ck.get("git_commit_hash")
    md = json.loads(ck6521.predictor_metadata_path.read_text())
    md_git = md.get("git_commit")
    if not isinstance(ck_git, str) or not isinstance(md_git, str) or ck_git != md_git:
        raise ValueError(
            "authorized-commit recovery discrepancy: checkpoint.git_commit_hash="
            f"{ck_git!r}, metadata.git_commit={md_git!r}"
        )
    if not (len(ck_git) == 40 and all(c in "0123456789abcdef" for c in ck_git)):
        raise ValueError(f"recovered authorized commit is not a 40-char hex SHA1: {ck_git!r}")
    _AUTHORIZED_GIT_COMMIT_RESOLVED = ck_git
    return _AUTHORIZED_GIT_COMMIT_RESOLVED


AUTHORIZED_GIT_COMMIT: str = _resolve_authorized_commit()


def _load_checkpoint_config(checkpoint_path: Path) -> Dict[str, Any]:
    import torch
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    if not isinstance(cfg, dict):
        raise ValueError(f"checkpoint 'config' is not a dict at {checkpoint_path}")
    return cfg


@dataclass
class AdapterProvenance:
    """Provenance record for one per-seed sidecar normalization."""

    seed: int
    transformation_version: str
    original_metadata_path: str
    original_metadata_sha256: str
    normalized_metadata_path: str
    normalized_metadata_sha256: str
    original_resolved_config_path: str
    original_resolved_config_sha256: str
    normalized_resolved_config_path: str
    normalized_resolved_config_sha256: str
    checkpoint_path: str
    checkpoint_sha256: str
    original_git_commit_value: str
    alias_git_commit_hash_value: str
    authorized_git_identity: str
    checkpoint_git_commit_hash_value: str
    root_identity_field_sources: Dict[str, Any] = field(default_factory=dict)
    # Where each added root-level field got its value (checkpoint.config.<field>),
    # and the verification records (effective_config.<field>, metadata.<field>).

    @property
    def normalized_metadata_path_as_path(self) -> Path:
        return Path(self.normalized_metadata_path)

    @property
    def normalized_resolved_config_path_as_path(self) -> Path:
        return Path(self.normalized_resolved_config_path)

    def to_manifest_dict(self) -> Dict:
        return {
            "seed": self.seed,
            "transformation_version": self.transformation_version,
            "original_metadata_path": self.original_metadata_path,
            "original_metadata_sha256": self.original_metadata_sha256,
            "normalized_metadata_path": self.normalized_metadata_path,
            "normalized_metadata_sha256": self.normalized_metadata_sha256,
            "original_resolved_config_path": self.original_resolved_config_path,
            "original_resolved_config_sha256": self.original_resolved_config_sha256,
            "normalized_resolved_config_path": self.normalized_resolved_config_path,
            "normalized_resolved_config_sha256": self.normalized_resolved_config_sha256,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "original_git_commit_value": self.original_git_commit_value,
            "alias_git_commit_hash_value": self.alias_git_commit_hash_value,
            "authorized_git_identity": self.authorized_git_identity,
            "checkpoint_git_commit_hash_value": self.checkpoint_git_commit_hash_value,
            "root_identity_field_sources": self.root_identity_field_sources,
            "fields_added_to_metadata": ["git_commit_hash", "sequence_length", "rul_cap"],
            "fields_added_to_resolved_config": ["seed", "sequence_length", "rul_cap"],
            "fields_kept_untouched": ["git_commit", "effective_config"],
        }


def normalize_for_seed(seed: int, output_dir: Path) -> AdapterProvenance:
    """Produce temporary normalized sidecar COPIES for ``seed``.

    Writes two files into ``output_dir`` (OUTSIDE the upstream checkpoint store):
      - ``predictor_metadata_normalized_seed_<s>.json``
      - ``resolved_config_normalized_seed_<s>.json``

    The metadata copy adds ``git_commit_hash = metadata['git_commit']`` and
    root-level ``sequence_length`` / ``rul_cap`` (sourced from the frozen
    checkpoint's ``config``). The resolved_config copy adds root-level
    ``seed`` / ``sequence_length`` / ``rul_cap`` (sourced from the frozen
    checkpoint's ``config``).

    The ORIGINAL M8 sidecar files are never mutated.

    Fail-closed on any identity discrepancy:
      - missing original ``metadata['git_commit']``                     -> ValueError
      - missing ``checkpoint['git_commit_hash']``                       -> ValueError
      - ``metadata['git_commit'] != checkpoint['git_commit_hash']``     -> ValueError
      - existing contradictory ``metadata['git_commit_hash']``          -> ValueError
      - identity != authorized commit                                   -> ValueError
      - checkpoint ``config`` identity != effective_config nesting      -> ValueError
      - checkpoint ``config`` identity != existing metadata root seed  -> ValueError (if present)
      - ``output_dir`` inside the M8 worktree                            -> ValueError
    """
    ckpts = manifest.load_frozen_checkpoints()
    if seed not in ckpts:
        raise ValueError(f"seed {seed} is not in the frozen checkpoint table")
    ck = ckpts[seed]
    original_md_path = ck.predictor_metadata_path
    original_rc_path = ck.resolved_config_path

    output_dir = Path(output_dir)
    _reject_output_dir_inside_m8(output_dir, original_md_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(original_md_path.read_text())
    original_md_sha = _sha256_file(original_md_path)

    # ---- git identity three-way agreement (invariants 18-22) — run BEFORE
    # touching resolved_config, so the documented fail-closed conditions fire
    # independently of the resolved_config sidecar. ----
    md_git = metadata.get("git_commit")
    if not isinstance(md_git, str) or not md_git:
        raise ValueError(
            f"seed {seed}: original predictor_metadata.json has no string 'git_commit'"
        )
    if "git_commit_hash" in metadata:
        existing = metadata["git_commit_hash"]
        if existing != md_git:
            raise ValueError(
                f"seed {seed}: original metadata already carries a contradictory "
                f"git_commit_hash={existing!r} (conflicts with git_commit={md_git!r})"
            )
    import torch
    ckpt = torch.load(ck.checkpoint_path, map_location="cpu", weights_only=False)
    ck_git = ckpt.get("git_commit_hash")
    if not isinstance(ck_git, str) or not ck_git:
        raise ValueError(
            f"seed {seed}: checkpoint has no string 'git_commit_hash'"
        )
    if md_git != ck_git:
        raise ValueError(
            f"seed {seed}: git-identity mismatch — metadata.git_commit={md_git!r} "
            f"!= checkpoint.git_commit_hash={ck_git!r}"
        )
    auth = AUTHORIZED_GIT_COMMIT
    if md_git != auth:
        raise ValueError(
            f"seed {seed}: git identity {md_git!r} is not the authorized predictor "
            f"commit {auth!r}"
        )

    # ---- now read resolved_config + checkpoint.config for root identity ----
    resolved_config = json.loads(original_rc_path.read_text())
    original_rc_sha = _sha256_file(original_rc_path)
    checkpoint_config = ckpt.get("config", {})
    if not isinstance(checkpoint_config, dict):
        raise ValueError(f"seed {seed}: checkpoint 'config' is not a dict")

    # ---- root identity field sourcing + verification ----
    # Values come from the checkpoint's OWN config (frozen, SHA256-bound) and
    # must agree with the effective_config nesting (resolved_config) and the
    # existing metadata root seed (if present).
    effective_config = resolved_config.get("effective_config", {})
    if not isinstance(effective_config, dict):
        raise ValueError(
            f"seed {seed}: resolved_config.effective_config is not a dict"
        )
    field_sources: Dict[str, Any] = {}
    for f in _ROOT_IDENTITY_FIELDS:
        ck_val = checkpoint_config.get(f)
        ec_val = effective_config.get(f)
        if ck_val is None:
            raise ValueError(
                f"seed {seed}: checkpoint.config has no '{f}' (cannot source root identity)"
            )
        if ec_val is None:
            raise ValueError(
                f"seed {seed}: resolved_config.effective_config has no '{f}' "
                "(cannot verify root identity against nesting)"
            )
        if ck_val != ec_val:
            raise ValueError(
                f"seed {seed}: root identity '{f}' disagreement — "
                f"checkpoint.config={ck_val!r} vs effective_config={ec_val!r}"
            )
        if f == "seed":
            md_root_seed = metadata.get("seed")
            if md_root_seed is not None and md_root_seed != ck_val:
                raise ValueError(
                    f"seed {seed}: metadata.root seed={md_root_seed!r} != "
                    f"checkpoint.config.seed={ck_val!r}"
                )
        field_sources[f] = {
            "checkpoint_config_value": ck_val,
            "effective_config_value": ec_val,
            "metadata_root_value": metadata.get(f),
        }

    # ---- build normalized metadata copy ----
    new_meta: Dict[str, Any] = {}
    for k, v in metadata.items():
        if k == "git_commit_hash":
            continue  # re-emit below, right after git_commit
        new_meta[k] = v
        if k == "git_commit":
            new_meta["git_commit_hash"] = md_git
    # Add root identity fields that the metadata lacks (sequence_length, rul_cap).
    for f in _ROOT_IDENTITY_FIELDS:
        if metadata.get(f) is None:
            new_meta[f] = checkpoint_config[f]
        elif metadata[f] != checkpoint_config[f]:
            raise ValueError(
                f"seed {seed}: metadata already carries root '{f}'={metadata[f]!r} "
                f"that disagrees with checkpoint.config.{f}={checkpoint_config[f]!r}"
            )

    # ---- build normalized resolved_config copy ----
    new_rc: Dict[str, Any] = {}
    for k, v in resolved_config.items():
        new_rc[k] = v
    for f in _ROOT_IDENTITY_FIELDS:
        if resolved_config.get(f) is None:
            new_rc[f] = checkpoint_config[f]
        elif resolved_config[f] != checkpoint_config[f]:
            raise ValueError(
                f"seed {seed}: resolved_config already carries root '{f}'="
                f"{resolved_config[f]!r} that disagrees with checkpoint.config.{f}="
                f"{checkpoint_config[f]!r}"
            )

    # ---- write copies (atomic) ----
    norm_md_path = output_dir / f"predictor_metadata_normalized_seed_{seed}.json"
    norm_rc_path = output_dir / f"resolved_config_normalized_seed_{seed}.json"
    norm_md_path.write_bytes((json.dumps(new_meta, indent=2) + "\n").encode("utf-8"))
    norm_rc_path.write_bytes((json.dumps(new_rc, indent=2) + "\n").encode("utf-8"))

    return AdapterProvenance(
        seed=seed,
        transformation_version=_TRANSFORMATION_VERSION,
        original_metadata_path=str(original_md_path),
        original_metadata_sha256=original_md_sha,
        normalized_metadata_path=str(norm_md_path),
        normalized_metadata_sha256=_sha256_file(norm_md_path),
        original_resolved_config_path=str(original_rc_path),
        original_resolved_config_sha256=original_rc_sha,
        normalized_resolved_config_path=str(norm_rc_path),
        normalized_resolved_config_sha256=_sha256_file(norm_rc_path),
        checkpoint_path=str(ck.checkpoint_path),
        checkpoint_sha256=ck.sha256,
        original_git_commit_value=md_git,
        alias_git_commit_hash_value=md_git,
        authorized_git_identity=auth,
        checkpoint_git_commit_hash_value=ck_git,
        root_identity_field_sources=field_sources,
    )


def _reject_output_dir_inside_m8(output_dir: Path, original_md_path: Path) -> None:
    """The normalized copies must never be written into the M8 worktree. The M8
    worktree root is the ``upstream-checkpoint-worktree`` ancestor of the
    original metadata path."""
    resolved_out = Path(output_dir).resolve()
    m8_root_parts = []
    for part in original_md_path.resolve().parts:
        m8_root_parts.append(part)
        if part.startswith("drl-selective-maintenance"):
            break
    m8_root = Path(*m8_root_parts) if m8_root_parts else None
    if m8_root is not None and (
        resolved_out == m8_root or resolved_out.is_relative_to(m8_root)
    ):
        raise ValueError(
            f"refusing to write normalized sidecar copies inside the M8 worktree: "
            f"{output_dir} (under {m8_root})"
        )
