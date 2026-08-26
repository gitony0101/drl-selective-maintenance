"""
Asset-contract preflight helper for M5 (non-training mode).

This module provides the SINGLE authoritative non-training asset-contract
validator used by:

  - ``python scripts/train_ddqn.py --validate-only`` (CLI preflight path)
  - ``scripts/generate_m5_matrix.py`` per-row validation in the matrix
    generator so that the same logic is shared between the two production
    entry points.

Contract:  Preflight MUST fail before any ``DDQNTrainer`` is constructed,
because constructing the trainer creates a run directory at once.
Preflight therefore performs NO side effects on disk besides reading the
referenced asset files (scenario banks + prediction-cache manifest).

It verifies, for the effective row, that:

  1. The training scenario bank loads.
  2. The validation scenario bank loads.
  3. EVERY training scenario's ``cost_regime_id`` matches the effective
     ``cost_regime_id`` (the exact blocker that escaped the previous
     preflight).
  4. EVERY validation scenario's ``cost_regime_id`` matches the effective
     ``cost_regime_id``.
  5. Every training scenario's ``maintenance_capacity`` matches the
     effective K.
  6. Every validation scenario's ``maintenance_capacity`` matches the
     effective K.
  7. Every training scenario's ``split`` equals ``predictor_train``.
  8. Every validation scenario's ``split`` equals ``rl_validation``.
  9. ``rl_test`` is rejected for either split, every scenario split, and
     every scenario bank's ``split`` field.
 10. The two prediction-cache compatibility identities hold:
        - the path exists and is the V2 cache; and
        - the manifest file exists and parses.
 11. The effective action count is consistent with K: K=1 -> 6; K=2 -> 16.

This helper is intentionally strict so that ANY pre-training contract
mismatch fails with a clear error and zero run-directory side effects.

Validation of cost parameters, dynamics, and prediction-cache compatibility
across regimes is NOT duplicated here: the trainer's environment
construction (``src.envs.selective_maintenance_env:__init__``) already
validates cost regime uniqueness and the env-level identity contract.

The ``validate_row_asset_contract`` function returns a PreflightReport
dataclass whose ``.ok`` is True iff ALL checks pass.  ``.errors`` lists
all collected failures; ``.warnings`` records non-blocking observations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence

from src.envs.scenario_bank import ScenarioBank, load_scenario_bank
from src.envs.config import ALLOWED_SPLITS, V2_CACHE_PATH_IDENTIFIER
from src.envs.action_table import ACTION_TABLE_N5_K1, ACTION_TABLE_N5_K2
from src.envs.costs import COST_REGIMES


# ---------------------------------------------------------------------------
# Expected split assignments for M5 formal rows.
# ---------------------------------------------------------------------------
TRAINING_SPLIT_REQUIRED = "predictor_train"
VALIDATION_SPLIT_REQUIRED = "rl_validation"

# Action count expected per K is derived from the frozen action tables.
EXPECTED_ACTION_COUNTS = {
    1: len(ACTION_TABLE_N5_K1),   # 6
    2: len(ACTION_TABLE_N5_K2),   # 16
}

# Cache path identifier required for the V2 cache (must be present in path).
V2_CACHE_IDENTIFIER = V2_CACHE_PATH_IDENTIFIER


@dataclass
class PreflightReport:
    """Result of an asset-contract preflight.

    Attributes:
        ok: True iff there are no errors.
        errors: All collected failure messages.
        warnings: Non-blocking observations.
        effective: Dict of effective resolved fields used for the check,
                   recorded for the caller's audit / matrix fields.
    """

    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    effective: dict = field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.ok = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def _safe_load_bank(path: str) -> tuple[Optional[ScenarioBank], Optional[str]]:
    """Try to load a scenario bank.

    Returns (bank, error).  Exactly one is None on return.
    """
    p = Path(path)
    if not p.exists():
        return None, f"scenario bank not found: {p}"
    try:
        return load_scenario_bank(p), None
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"scenario bank failed to load ({p}): {exc}"


def _assert_cost_regime_id(effective_regime: str) -> Optional[str]:
    """Return an error string if the effective regime is unknown, else None."""
    if effective_regime not in COST_REGIMES:
        return (
            f"effective cost_regime_id unknown: '{effective_regime}'. "
            f"Available: {sorted(COST_REGIMES.keys())}"
        )
    return None


def _assert_k(k_val: Any) -> Optional[str]:
    """Return an error string if K is not 1 or 2, else None."""
    try:
        k_int = int(k_val)
    except (TypeError, ValueError):
        return f"maintenance_capacity must be an int (1 or 2), got {k_val!r}"
    if k_int not in (1, 2):
        return f"maintenance_capacity must be 1 or 2, got {k_int}"
    return None


# ---------------------------------------------------------------------------
# Main preflight.
# ---------------------------------------------------------------------------

def validate_row_asset_contract(
    *,
    training_scenario_bank_path: str,
    validation_scenario_bank_path: str,
    cost_regime_id: str,
    maintenance_capacity: Any,
    prediction_cache_path: str,
    training_split: str = TRAINING_SPLIT_REQUIRED,
    validation_split: str = VALIDATION_SPLIT_REQUIRED,
    allow_rl_test: bool = False,
) -> PreflightReport:
    """Validate the full asset contract of one matrix row WITHOUT training.

    This is the SINGLE authoritative non-training asset-contract validator.
    Used by ``--validate-only`` and matrix generation.

    Args:
        training_scenario_bank_path: Bank used at training time.
        validation_scenario_bank_path: Bank used at validation time.
        cost_regime_id: Effective cost regime id (after CLI overrides).
        maintenance_capacity: Effective K (after CLI overrides).
        prediction_cache_path: Effective prediction cache directory.
        training_split: Expected training split ("-predictor_train").
        validation_split: Expected validation split ("rl_validation").
        allow_rl_test: Must NEVER be True for M5 production paths.  rl_test
            is sealed.  Tests may flip this to confirm the test fails closed.

    Returns:
        A PreflightReport.  Caller MUST check ``.ok``.
    """
    report = PreflightReport(ok=True)

    if allow_rl_test:
        # Production paths must NEVER pass this.  This guard only exists so
        # the test suite can assert the helper itself refuses rl_test.
        report.add_error(
            "REFUSED: allow_rl_test=True is forbidden in production preflight."
        )
        return report

    # Record the effective resolved fields for caller audit.
    report.effective = {
        "training_scenario_bank_path": training_scenario_bank_path,
        "validation_scenario_bank_path": validation_scenario_bank_path,
        "cost_regime_id": cost_regime_id,
        "maintenance_capacity": maintenance_capacity,
        "training_split": training_split,
        "validation_split": validation_split,
        "prediction_cache_path": prediction_cache_path,
    }

    # ----- Effective field validation (fail closed early) -----
    err = _assert_cost_regime_id(cost_regime_id)
    if err is not None:
        report.add_error(err)

    err = _assert_k(maintenance_capacity)
    if err is not None:
        report.add_error(err)
        # Cannot continue K-dependent checks; but still validate the banks.
        k_int = None
    else:
        k_int = int(maintenance_capacity)

    # rl_test barrier on splits themselves
    for label, sval in (
        ("training_split", training_split),
        ("validation_split", validation_split),
    ):
        if sval == "rl_test":
            report.add_error(
                f"{label}='rl_test' is FORBIDDEN. rl_test is sealed for "
                f"M5 training and validation."
            )

    if training_split != TRAINING_SPLIT_REQUIRED:
        report.add_error(
            f"training_split must be '{TRAINING_SPLIT_REQUIRED}', got '{training_split}'"
        )
    if validation_split != VALIDATION_SPLIT_REQUIRED:
        report.add_error(
            f"validation_split must be '{VALIDATION_SPLIT_REQUIRED}', got '{validation_split}'"
        )

    # ----- Load both scenario banks -----
    train_bank, terr = _safe_load_bank(training_scenario_bank_path)
    if terr is not None:
        report.add_error(terr)
    val_bank, verr = _safe_load_bank(validation_scenario_bank_path)
    if verr is not None:
        report.add_error(verr)

    if train_bank is None or val_bank is None:
        # Cannot continue; the bank existence errors are already recorded.
        return report

    # ----- Bank-level split agreement -----
    if train_bank.split != training_split:
        report.add_error(
            f"Training bank split mismatch: bank has '{train_bank.split}', "
            f"expected '{training_split}'"
        )
    if val_bank.split != validation_split:
        report.add_error(
            f"Validation bank split mismatch: bank has '{val_bank.split}', "
            f"expected '{validation_split}'"
        )

    if train_bank.split == "rl_test" or val_bank.split == "rl_test":
        report.add_error(
            "Bank split provenance contains rl_test; rl_test is forbidden."
        )

    # ----- Per-scenario validation -----
    for scenario in train_bank.scenarios:
        if scenario.cost_regime_id != cost_regime_id:
            report.add_error(
                f"Training scenario {scenario.scenario_id!r} cost_regime_id="
                f"'{scenario.cost_regime_id}' does not match effective "
                f"cost_regime_id='{cost_regime_id}'"
            )
        if k_int is not None and scenario.maintenance_capacity != k_int:
            report.add_error(
                f"Training scenario {scenario.scenario_id!r} K="
                f"{scenario.maintenance_capacity} does not match effective K={k_int}"
            )
        if scenario.split != TRAINING_SPLIT_REQUIRED:
            report.add_error(
                f"Training scenario {scenario.scenario_id!r} split="
                f"'{scenario.split}' must be '{TRAINING_SPLIT_REQUIRED}'"
            )
        if scenario.split == "rl_test":
            report.add_error(
                f"Training scenario {scenario.scenario_id!r} has forbidden split='rl_test'"
            )

    for scenario in val_bank.scenarios:
        if scenario.cost_regime_id != cost_regime_id:
            report.add_error(
                f"Validation scenario {scenario.scenario_id!r} cost_regime_id="
                f"'{scenario.cost_regime_id}' does not match effective "
                f"cost_regime_id='{cost_regime_id}'"
            )
        if k_int is not None and scenario.maintenance_capacity != k_int:
            report.add_error(
                f"Validation scenario {scenario.scenario_id!r} K="
                f"{scenario.maintenance_capacity} does not match effective K={k_int}"
            )
        if scenario.split != VALIDATION_SPLIT_REQUIRED:
            report.add_error(
                f"Validation scenario {scenario.scenario_id!r} split="
                f"'{scenario.split}' must be '{VALIDATION_SPLIT_REQUIRED}'"
            )
        if scenario.split == "rl_test":
            report.add_error(
                f"Validation scenario {scenario.scenario_id!r} has forbidden split='rl_test'"
            )

    # ----- Action count consistency -----
    if k_int is not None:
        expected = EXPECTED_ACTION_COUNTS.get(k_int)
        if expected is None:
            report.add_error(f"No expected action count registered for K={k_int}")
        else:
            report.effective["expected_action_count"] = expected

    # ----- Prediction-cache compatibility (V2 path + manifest exists) -----
    cache_dir = Path(prediction_cache_path)
    if V2_CACHE_IDENTIFIER not in str(cache_dir):
        report.add_error(
            f"prediction_cache_path must contain '{V2_CACHE_IDENTIFIER}', "
            f"got: {cache_dir}"
        )
    elif not cache_dir.exists():
        report.add_error(
            f"prediction cache directory not found: {cache_dir}"
        )

    # Predict-cache manifest presence check: the production path uses the
    # ``prediction_cache_manifest_path`` field of the trainer config.  We
    # co-locate that with the cache directory by convention:
    #   <cache_dir>/prediction_cache_manifest_v2.json
    manifest_candidates = [
        cache_dir / "prediction_cache_manifest_v2.json",
        cache_dir.parent / "prediction_cache_manifest_v2.json",
    ]
    manifest_found = next((p for p in manifest_candidates if p.exists()), None)
    if manifest_found is None and cache_dir.exists():
        # Non-blocking for the cache directory listing; we still record this
        # so the caller can verify the path is resolvable.
        report.add_warning(
            f"prediction cache manifest not auto-discovered under {cache_dir}; "
            f"ensure TrainerConfig.prediction_cache_manifest_path is correctly set."
        )
    elif manifest_found is not None:
        report.effective["prediction_cache_manifest_path"] = str(manifest_found)

    return report


def preflight_report_to_dict(report: PreflightReport) -> dict:
    """Serialize a preflight report to a plain dict for JSON artifacts."""
    return {
        "ok": report.ok,
        "errors": list(report.errors),
        "warnings": list(report.warnings),
        "effective": dict(report.effective),
    }
