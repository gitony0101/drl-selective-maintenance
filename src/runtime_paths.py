"""Neutral runtime-path resolution for external, git-ignored assets.

Large runtime artifacts (per-seed prediction caches, training checkpoints,
run outputs, raw C-MAPSS downloads) intentionally live outside version
control. Their location is machine-specific and therefore resolved from the
``DRL_EXTERNAL_ROOT`` environment variable instead of being hardcoded.

If ``DRL_EXTERNAL_ROOT`` is unset, a conventional sibling directory next to
the repository checkout is used.
"""
import os
from pathlib import Path


def repo_root() -> Path:
    """Absolute path of this repository checkout."""
    return Path(__file__).resolve().parents[1]


def external_root() -> Path:
    """Root directory holding git-ignored runtime assets.

    Resolution order:
      1. ``$DRL_EXTERNAL_ROOT`` if set;
      2. ``<repo>/../drl_external_assets`` otherwise.
    """
    configured = os.environ.get("DRL_EXTERNAL_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return repo_root().parent / "drl_external_assets"
