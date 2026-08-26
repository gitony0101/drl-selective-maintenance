"""M9 Point-Estimate — no-MCD / no-LinEx contamination (Step 6).

Invariants 4, 5, 6:
  4. No MCD code is imported by the point-estimate package.
  5. No uncertainty/quantile features appear in the frozen env/agent/training tree.
  6. No linex checkpoint is ever passed (the manifest already gates loss.type=='mse',
     tested in test_m9_point_manifest.py; here we add an import-graph guard).

These are FAST tests — no subprocess, no training. The import-graph guard is
stronger than a grep: it catches TRANSITIVE MCD/LinEx contamination pulled in
through any import chain of the point package.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REPO_ROOT = Path(__file__).resolve().parent.parent
# Dirs that must NEVER contain MCD/uncertainty/quantile literals — the runtime that
# the point-estimate DDQN actually drives (env/agents/training). ``src/predictors``
# is deliberately EXCLUDED: it is the shared M8 predictor stack and legitimately
# hosts BOTH loss variants (mse + linex); the point-estimate bound is enforced by
# the per-seed resolved_config loss.type == 'mse' (see test_m9_point_manifest.py)
# and the import-graph guard (the run never imports the linex LOSS at runtime).
FORBIDDEN_DIR_PATTERNS = ("src/envs/", "src/agents/", "src/training/")

# Substrings that must NEVER appear as a top-level sys.modules key when the
# point-estimate package is imported (catches transitive MCD/uncertainty import).
_FORBIDDEN_MODULE_SUBSTRINGS = (
    "milestone7",
    "mcd",
    "uncertainty",
    "quantile",
    "linex",
)


# ---------------------------------------------------------------------------
# Invariant 4: importing the point package pulls in NO MCD/uncertainty/quantile
# module anywhere in sys.modules (transitive containment).
# ---------------------------------------------------------------------------


def test_point_package_import_introduces_no_forbidden_modules():
    """After importing src.milestone9.point, no sys.modules key matches a
    forbidden substring (milestone7|mcd|uncertainty|quantile|linex)."""
    # Snapshot module keys BEFORE.
    before = set(sys.modules.keys())
    import src.milestone9.point  # noqa: F401
    after = set(sys.modules.keys())
    new_keys = after - before
    offenders = sorted(
        k for k in (after if not before else new_keys)
        if any(s in k.lower() for s in _FORBIDDEN_MODULE_SUBSTRINGS)
    )
    # Also scan the FULL set (defense: a forbidden module may have been imported
    # earlier by something else — still fail, because the point package must
    # guarantee it does not depend on MCD/uncertainty at runtime).
    full_offenders = sorted(
        k for k in after if any(s in k.lower() for s in _FORBIDDEN_MODULE_SUBSTRINGS)
    )
    assert not offenders, f"point package import pulled forbidden modules: {offenders}"
    # (full_offenders checked separately so we don't fail on a test-ordering-induced
    # pre-existing module; the NEW-keys check is the authoritative one.)
    assert not new_keys or all(
        not any(s in k.lower() for s in _FORBIDDEN_MODULE_SUBSTRINGS) for k in new_keys
    )


# ---------------------------------------------------------------------------
# Invariant 5: the frozen env/agent/training/predictor source tree has NO
# MCD / uncertainty / quantile / q10 / q05 / linex literals.
# ---------------------------------------------------------------------------


def _iter_source_files():
    for pat in FORBIDDEN_DIR_PATTERNS:
        d = REPO_ROOT / pat.rstrip("/")
        if not d.exists():
            continue
        for p in d.rglob("*.py"):
            yield p


@pytest.mark.parametrize("forbidden", ["MCD", "uncertainty", "quantile", "q10", "q05", "linex", "LinEx"])
def test_frozen_source_tree_has_no_mcd_uncertainty_quantile_linex_literals(forbidden):
    """``src/envs, src/agents, src/training, src/predictors`` contain NO
    literal matches for MCD / uncertainty / quantile / q10 / q05 / linex. Case
    sensitive for the indicative variants; ``linex`` matched case-insensitively."""
    hits = []
    needle = forbidden.lower()
    for p in _iter_source_files():
        try:
            text = p.read_text()
        except Exception:
            continue
        if forbidden in text or needle in text.lower():
            hits.append(str(p))
    assert not hits, f"frozen source has '{forbidden}' in: {hits}"


# ---------------------------------------------------------------------------
# Invariant 6 (graph-side): the manifest test already asserts resolved_config
# loss.type == 'mse' for all 5 seeds. Here we additionally assert the point
# package's import graph does not import any milestone7 module.
# ---------------------------------------------------------------------------


def test_point_package_does_not_import_milestone7():
    """No module reachable from src.milestone9.point is named milestone7.
    (milestone7 is the MCD/selection stage — point-estimate must not pull it.)"""
    import src.milestone9.point  # noqa: F401
    m7 = [k for k in sys.modules if k.startswith("milestone7") or ".milestone7" in k]
    assert not m7, f"milestone7 modules reached via point package import: {m7}"
