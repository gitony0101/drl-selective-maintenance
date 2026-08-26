"""Tests for the neutral external-runtime-path resolver (src/runtime_paths.py)."""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _fresh():
    import src.runtime_paths as rp
    return importlib.reload(rp)


def test_repo_root_is_repository(monkeypatch, tmp_path):
    rp = _fresh()
    root = rp.repo_root()
    assert (root / "src" / "runtime_paths.py").exists()


def test_env_var_overrides_default(monkeypatch, tmp_path):
    monkeypatch.setenv("DRL_EXTERNAL_ROOT", str(tmp_path))
    rp = _fresh()
    assert rp.external_root() == tmp_path.resolve()


def test_default_is_sibling_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("DRL_EXTERNAL_ROOT", raising=False)
    rp = _fresh()
    expected = rp.repo_root().parent / "drl_external_assets"
    # Do not require it to exist; only that resolution is deterministic and
    # outside the repository.
    assert rp.external_root() == expected.resolve()
    assert not rp.external_root().is_relative_to(rp.repo_root())


def test_milestone_modules_use_external_root(monkeypatch, tmp_path):
    """The sanitized milestone path modules must derive their roots from the
    environment variable, never from a hardcoded absolute path."""
    monkeypatch.setenv("DRL_EXTERNAL_ROOT", str(tmp_path))
    srcs = [
        "src/milestone9/point/pairing.py",
        "src/milestone9/point/cache_prep.py",
        "src/milestone10/e3/h2_context.py",
        "src/milestone10/e3/trajectories.py",
        "src/milestone11/e4/paths.py",
        "src/milestone11/e5/paths.py",
    ]
    for rel in srcs:
        text = (Path(__file__).resolve().parent.parent / rel).read_text()
        assert "/Use" + "rs/" not in text, f"{rel} contains an absolute home path"
        assert "runtime_paths" in text or "external_root" in text or \
            "REPO_ROOT" in text, f"{rel} does not use the neutral resolver"


def test_contract_json_has_no_absolute_paths():
    contract = (
        Path(__file__).resolve().parent.parent
        / "docs" / "milestone9" / "M9_POINT_ESTIMATE_CONTRACT.json"
    )
    import json
    data = json.loads(contract.read_text())
    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            assert "/Us" + "ers/" not in o
            # Course code removed from public version
    walk(data)
