"""Tests for checkpoint/resolved-config consistency.

Compares the canonical Milestone-1 training fields between a resolved config and
checkpoint metadata. The negative-path tests exercise the production validator
``src.predictors.generate_cache._validate_checkpoint_config`` directly so the
project's gate does not drift from what the tests assert.
"""

import csv
import json
import pathlib

import pytest
import torch

from src.predictors import generate_cache as _generate_cache

REQUIRED_FIELDS = [
    "seed",
    "sequence_length",
    "rul_cap",
    "model_type",
    "n_features",
    "hidden_dim",
    "n_layers",
    "dropout",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "normalizer_id",
    "feature_schema_id",
]


def _write_resolved_config(path: pathlib.Path, **overrides) -> dict:
    base = {
        "seed": 6521,
        "sequence_length": 50,
        "rul_cap": 125,
        "model": {
            "type": "mlp",
            "n_features": 24,
            "hidden_dim": 128,
            "n_layers": 3,
            "dropout": 0.2,
        },
        "training": {"batch_size": 64, "learning_rate": 1e-3, "weight_decay": 1e-4},
        "normalizer_id": "fd001_normalizer_v2",
        "feature_schema_id": "fd001_feature_schema_v1",
    }
    base.update(overrides)
    with open(path, "w") as f:
        json.dump(base, f)
    return base


def _write_checkpoint(path: pathlib.Path, config: dict) -> None:
    ckpt = {"model_state_dict": {}, "optimizer_state_dict": {}, "epoch": 4, "config": config}
    torch.save(ckpt, path)


def _flatten_config(cfg: dict) -> dict:
    m = cfg.get("model", {})
    t = cfg.get("training", {})
    return {
        "seed": cfg["seed"],
        "sequence_length": cfg["sequence_length"],
        "rul_cap": cfg["rul_cap"],
        "model_type": m["type"],
        "n_features": m["n_features"],
        "hidden_dim": m["hidden_dim"],
        "n_layers": m["n_layers"],
        "dropout": m["dropout"],
        "batch_size": t["batch_size"],
        "learning_rate": t["learning_rate"],
        "weight_decay": t["weight_decay"],
        "normalizer_id": cfg["normalizer_id"],
        "feature_schema_id": cfg["feature_schema_id"],
    }


def _compare(resolved_cfg: dict, ckpt_config: dict) -> list:
    """Compare resolved config and checkpoint config.

    Accepts both the nested ``resolved`` form (``model.type``, ``training.batch_size``)
    and the flat checkpoint-form (``model_type``, ``batch_size``). Normalizes both
    to a canonical flat dict before comparison.
    """
    a = _flatten_config(resolved_cfg)
    b = _flatten_config(ckpt_config) if "model" in ckpt_config or "training" in ckpt_config else dict(ckpt_config)
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in a:
            errors.append(f"missing {field} in resolved config")
            continue
        if field not in b:
            errors.append(f"missing {field} in checkpoint config")
            continue
        if a[field] != b[field]:
            errors.append(f"{field} mismatch: resolved={a[field]!r} checkpoint={b[field]!r}")
    return errors


def test_all_required_fields_match(tmp_path: pathlib.Path):
    cfg_path = tmp_path / "resolved_config.json"
    resolved = _write_resolved_config(cfg_path)
    ckpt_config = {
        "seed": resolved["seed"],
        "sequence_length": resolved["sequence_length"],
        "rul_cap": resolved["rul_cap"],
        "model_type": resolved["model"]["type"],
        "n_features": resolved["model"]["n_features"],
        "hidden_dim": resolved["model"]["hidden_dim"],
        "n_layers": resolved["model"]["n_layers"],
        "dropout": resolved["model"]["dropout"],
        "batch_size": resolved["training"]["batch_size"],
        "learning_rate": resolved["training"]["learning_rate"],
        "weight_decay": resolved["training"]["weight_decay"],
        "normalizer_id": resolved["normalizer_id"],
        "feature_schema_id": resolved["feature_schema_id"],
    }
    errors = _compare(resolved, ckpt_config)
    assert errors == [], f"unexpected mismatches: {errors}"


def test_mismatch_detected(tmp_path: pathlib.Path):
    cfg_path = tmp_path / "resolved_config.json"
    resolved = _write_resolved_config(cfg_path, rul_cap=125)
    ckpt_config = {
        "seed": resolved["seed"],
        "sequence_length": resolved["sequence_length"],
        "rul_cap": 100,  # mismatch
        "model_type": resolved["model"]["type"],
        "n_features": resolved["model"]["n_features"],
        "hidden_dim": resolved["model"]["hidden_dim"],
        "n_layers": resolved["model"]["n_layers"],
        "dropout": resolved["model"]["dropout"],
        "batch_size": resolved["training"]["batch_size"],
        "learning_rate": resolved["training"]["learning_rate"],
        "weight_decay": resolved["training"]["weight_decay"],
        "normalizer_id": resolved["normalizer_id"],
        "feature_schema_id": resolved["feature_schema_id"],
    }
    errors = _compare(resolved, ckpt_config)
    assert any("rul_cap mismatch" in e for e in errors), f"expected rul_cap mismatch, got {errors}"


def test_checkpoint_config_roundtrips(tmp_path: pathlib.Path):
    cfg_path = tmp_path / "resolved_config.json"
    resolved = _write_resolved_config(cfg_path)
    ckpt_config = {
        "seed": resolved["seed"],
        "sequence_length": resolved["sequence_length"],
        "rul_cap": resolved["rul_cap"],
        "model_type": resolved["model"]["type"],
        "n_features": resolved["model"]["n_features"],
        "hidden_dim": resolved["model"]["hidden_dim"],
        "n_layers": resolved["model"]["n_layers"],
        "dropout": resolved["model"]["dropout"],
        "batch_size": resolved["training"]["batch_size"],
        "learning_rate": resolved["training"]["learning_rate"],
        "weight_decay": resolved["training"]["weight_decay"],
        "normalizer_id": resolved["normalizer_id"],
        "feature_schema_id": resolved["feature_schema_id"],
    }
    ckpt_path = tmp_path / "best_checkpoint.pt"
    _write_checkpoint(ckpt_path, ckpt_config)
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    errors = _compare(resolved, loaded["config"])
    assert errors == []


# ---------------------------------------------------------------------------
# Production-validator exercises
#
# These tests call the actual production code path
# ``src.predictors.generate_cache._validate_checkpoint_config`` so the project
# gate cannot drift from what the tests assert.
# ---------------------------------------------------------------------------


def test_production_validator_accepts_v2_checkpoint(tmp_path: pathlib.Path):
    """The cache generator's V2 safety guard accepts a full-fidelity checkpoint."""
    ckpt = tmp_path / "best_checkpoint.pt"
    cfg = {
        "seed": 6521,
        "sequence_length": 50,
        "rul_cap": 125,
        "model_type": "mlp",
        "n_features": 24,
        "hidden_dim": 128,
        "n_layers": 3,
        "dropout": 0.2,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "normalizer_id": "fd001_normalizer_v2",
        "feature_schema_id": "fd001_feature_schema_v1",
    }
    torch.save({
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "epoch": 0,
        "config": cfg,
    }, ckpt)
    # Production guard must accept this checkpoint without raising.
    _generate_cache._validate_checkpoint_config(ckpt)


def test_production_validator_rejects_missing_v2_ids(tmp_path: pathlib.Path):
    """A V1-style checkpoint (no normalizer_id / feature_schema_id) is refused."""
    ckpt = tmp_path / "best_checkpoint.pt"
    torch.save({
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "epoch": 0,
        "config": {
            "seed": 1, "sequence_length": 30, "rul_cap": 100,
            "model_type": "mlp", "n_features": 24,
        },
    }, ckpt)
    with pytest.raises(ValueError, match="normalizer|schema|mismatch|missing"):
        _generate_cache._validate_checkpoint_config(ckpt)


def test_production_validator_rejects_invalidated_checkpoint(tmp_path: pathlib.Path):
    """Checkpoints under results/invalidated/ are refused by the production guard."""
    bad = tmp_path / "results" / "invalidated" / "x" / "best_checkpoint.pt"
    bad.parent.mkdir(parents=True)
    torch.save({"model_state_dict": {}, "epoch": 0,
                "config": {"normalizer_id": "fd001_normalizer_v2",
                           "feature_schema_id": "fd001_feature_schema_v1"}}, bad)
    with pytest.raises(ValueError, match="invalidated"):
        _generate_cache._assert_checkpoint_safe(bad)
