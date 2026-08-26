"""M9 Point-Estimate -- prediction_cache_manifest_path config passthrough.

REGRESSION: the frozen V2 trainer's ``TrainerConfig.prediction_cache_manifest_path``
(field at ``src/training/ddqn_trainer.py:67``) defaults to the RELATIVE repo path
``data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json``. The
frozen ``parse_raw_config`` (``src/training/ddqn_config.py:parse_raw_config``) read
``prediction_cache_path`` from the env config but NOT ``prediction_cache_manifest_path``
— so a per-seed external cache (whose manifest lives at
``m9_point_caches/seed_<s>/.../prediction_cache_manifest_v2.json``) could NOT have its
manifest path pinned via the config; the trainer fell back to the relative repo
default and ``save_checkpoint`` raised ``FileNotFoundError``.

This test pins the fix: ``parse_raw_config`` reads ``environment.prediction_cache_manifest_path``
and flows it to ``TrainerConfig.prediction_cache_manifest_path`` so an M9 runtime
config (derived by ``config_runtime.derive_runtime_config`` with the manifest path
auto-set) pins the per-seed manifest path the trainer reads at checkpoint time.

This is infrastructure plumbing (a config field passthrough), NOT a scientific-
contract change: it adds a field read with a default equal to the existing
dataclass default. Budgets, metrics, data roles, reward, observation, actions,
and selection rules are untouched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_external_assets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REPO_ROOT = Path(__file__).resolve().parent.parent
PILOT_CFG = REPO_ROOT / "configs" / "agents" / "ddqn_pilot_k2.json"


def test_parse_raw_config_reads_prediction_cache_manifest_path_from_env():
    """parse_raw_config sources prediction_cache_manifest_path FROM
    environment.prediction_cache_manifest_path (not just the dataclass default).
    A raw config carrying an explicit manifest path flows it to the resolved
    TrainerConfig.prediction_cache_manifest_path."""
    from src.training.ddqn_config import parse_raw_config

    cache_dir = "/tmp/m9_test_cache/seed_6521/data/processed/fd001/v2/06_PREDICTIONS/seed_6521"
    explicit_manifest = f"{cache_dir}/prediction_cache_manifest_v2.json"
    raw = json.loads(PILOT_CFG.read_text())
    raw["environment"]["prediction_cache_path"] = cache_dir
    raw["environment"]["prediction_cache_manifest_path"] = explicit_manifest
    # Fix the stale pilot validation_split so TrainerConfig validation passes
    # (the pilot config's predictor_train is stale; this is the same in-memory
    # remediation config_runtime applies — orthogonal to the manifest path).
    raw["environment"]["validation_split"] = "rl_validation"
    cfg = parse_raw_config(raw)
    assert cfg.prediction_cache_manifest_path == explicit_manifest
    assert cfg.prediction_cache_path == cache_dir


def test_parse_raw_config_prediction_cache_manifest_path_defaults_to_repo_relative():
    """When the env config does NOT pin prediction_cache_manifest_path, the
    default is the relative repo path (the existing dataclass default) — so the
    change is backward-compatible for ALL existing M5/M8 configs that never set
    this field."""
    from src.training.ddqn_config import parse_raw_config

    raw = json.loads(PILOT_CFG.read_text())
    # The frozen pilot config has NO prediction_cache_manifest_path field.
    assert "prediction_cache_manifest_path" not in raw["environment"]
    # Fix the stale pilot validation_split so TrainerConfig validation passes.
    raw["environment"]["validation_split"] = "rl_validation"
    cfg = parse_raw_config(raw)
    assert cfg.prediction_cache_manifest_path == \
        "data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json"
