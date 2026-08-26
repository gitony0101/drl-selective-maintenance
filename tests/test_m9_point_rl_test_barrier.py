"""
M9 Point-Estimate — rl_test information barrier (Step 4, invariant 10).

rl_test is SEALED for the entire point-estimate M9 pipeline. The barrier is on
consumption, not precomputation (rl_test rows exist in the cache parquet but
are never loaded by the DDQN run). Three fail-closed surfaces:

  a) the frozen preflight ``validate_row_asset_contract`` refuses rl_test splits
     and refuses ``allow_rl_test=True`` outright;
  b) ``TrainerConfig`` raises ValueError when ``split`` or ``validation_split``
     equals rl_test (the trainer NEVER constructs an rl_test env);
  c) the M9 wrapper rejects any rl_test split at its own level (defense-in-depth
     on top of the frozen barriers).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_external_assets

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Minimal valid bank paths for the preflight (it loads banks; if they don't
# exist the report records a different error and the rl_test assertion becomes
# ambiguous). Use the frozen production K=2 scenario banks.
REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_BANK = str(
    REPO_ROOT / "data" / "scenario_banks" / "m4_production"
    / "rl_validation_K2_failure-light-no-waste.json"
)
VAL_BANK = TRAIN_BANK  # same split name; the barrier fires before bank-split checks matter


def test_preflight_rejects_rl_test_validation_split():
    """Surface (a): validate_row_asset_contract with validation_split='rl_test'
    must return a report whose .ok is False (frozen barrier)."""
    from src.training.preflight import validate_row_asset_contract

    report = validate_row_asset_contract(
        training_scenario_bank_path=TRAIN_BANK,
        validation_scenario_bank_path=VAL_BANK,
        cost_regime_id="failure-light-no-waste",
        maintenance_capacity=2,
        prediction_cache_path="data/processed/fd001/v2/06_PREDICTIONS/",
        training_split="predictor_train",
        validation_split="rl_test",  # forbidden
        allow_rl_test=False,
    )
    assert report.ok is False
    joined = " | ".join(report.errors)
    assert "rl_test" in joined and "FORBIDDEN" in joined


def test_preflight_rejects_rl_test_training_split():
    """Surface (a): training_split='rl_test' must also make .ok False."""
    from src.training.preflight import validate_row_asset_contract

    report = validate_row_asset_contract(
        training_scenario_bank_path=TRAIN_BANK,
        validation_scenario_bank_path=VAL_BANK,
        cost_regime_id="failure-light-no-waste",
        maintenance_capacity=2,
        prediction_cache_path="data/processed/fd001/v2/06_PREDICTIONS/",
        training_split="rl_test",  # forbidden
        validation_split="rl_validation",
        allow_rl_test=False,
    )
    assert report.ok is False


def test_preflight_refuses_allow_rl_test_true():
    """Surface (a): the production preflight hard-refuses allow_rl_test=True
    even before checking splits — production paths must NEVER pass it."""
    from src.training.preflight import validate_row_asset_contract

    report = validate_row_asset_contract(
        training_scenario_bank_path=TRAIN_BANK,
        validation_scenario_bank_path=VAL_BANK,
        cost_regime_id="failure-light-no-waste",
        maintenance_capacity=2,
        prediction_cache_path="data/processed/fd001/v2/06_PREDICTIONS/",
        allow_rl_test=True,  # forbidden in production
    )
    assert report.ok is False
    assert any("allow_rl_test" in e and "REFUSED" in e for e in report.errors)


def test_trainer_config_rejects_rl_test_validation_split():
    """Surface (b): TrainerConfig raises ValueError when validation_split is
    rl_test (the frozen __post_init__ barrier)."""
    from src.training.ddqn_trainer import TrainerConfig

    with pytest.raises(ValueError, match="rl_test"):
        TrainerConfig(validation_split="rl_test")


def test_trainer_config_rejects_rl_test_training_split():
    """Surface (b): TrainerConfig raises ValueError when split is rl_test."""
    from src.training.ddqn_trainer import TrainerConfig

    with pytest.raises(ValueError, match="rl_test"):
        TrainerConfig(split="rl_test")


def test_wrapper_rejects_rl_test_split():
    """Surface (c): the M9 wrapper rejects rl_test at its own level before
    invoking any frozen CLI (defense-in-depth)."""
    from src.milestone9.point.wrapper import assert_no_rl_test_split

    with pytest.raises(ValueError, match="rl_test"):
        assert_no_rl_test_split({"split": "rl_test"})
    with pytest.raises(ValueError, match="rl_test"):
        assert_no_rl_test_split({"validation_split": "rl_test"})
    # The valid configuration does NOT raise.
    assert_no_rl_test_split({"split": "predictor_train",
                            "validation_split": "rl_validation"})
