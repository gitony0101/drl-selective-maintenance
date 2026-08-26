"""M9 Point-Estimate — wrapper validate-only dry-run (Step 12).

Invoke ``train_ddqn.py --validate-only`` for seed 6521 against the session-
scoped real seed-6521 cache (via the adapter) through the wrapper's argv
builder. ``validate-only`` runs ``validate_row_asset_contract`` (banks load,
cost_regime/K/split match, cache compatibility, no rl_test) BEFORE any
Trainer/output dir — so this proves the M9 wiring pipeline end-to-end with NO
training and NO new output dir.

This is a gated slow test (subprocess ~few seconds; no training). Required
green before pilot. Leaves the repository git-clean.

Pilot-config remediation (authorized 2026-08-06): the frozen pilot budget
source ``ddqn_pilot_k2.json`` carries ``validation_split="predictor_train"``,
stale by 16h relative to the trainer's hard ``rl_validation`` rule (committed
at d98cb41 06:40 vs rule at 3a28837 22:39). It was never run to completion
against the post-rule trainer. The CLI ``--validation-split`` flag cannot
rescue it (config-level validation fires before CLI overrides at
resolver.py:360-363). Remediation: ``config_runtime.derive_runtime_config``
overrides ``validation_split -> "rl_validation"`` in the IN-MEMORY runtime
copy ONLY (the frozen pilot file on disk is byte-identical/unmodified), and
the wrapper passes the two frozen regime-matched banks as explicit CLI flags.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.m9_slow

REPO_ROOT = Path(__file__).resolve().parents[2]
PILOT_CFG = REPO_ROOT / "configs" / "agents" / "ddqn_pilot_k2.json"


def test_wrapper_validate_only_for_seed_6521_real_cache(real_seed6521_cache_dir, tmp_path):
    """End-to-end: derive a seed-6521 runtime config from the frozen pilot config
    pointing at the real produced cache, with the pilot-only validation_split
    override to rl_validation, build the training argv with the two mandatory
    bank flags + --validate-only, run the subprocess, assert exit 0.

    Leaves no artifacts inside the repository (all outputs to tmp_path)."""
    import sys
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from src.milestone9.point import wrapper, config_runtime, pairing, paths

    # Derive runtime config: environment.prediction_cache_path -> real cache dir;
    # output.output_dir -> a tmp dir; pilot-only validation_split -> rl_validation
    # (the frozen pilot config's predictor_train split is stale; config-level
    # validation rejects it before any CLI override could land); AND
    # pilot_scenario_banks -> distinct regime-matched training/validation banks
    # (the frozen pilot config's legacy single scenario_bank_path makes
    # parse_raw_config default both banks equal, tripping the distinct-bank
    # rule before CLI overrides land). The cache dir contains the produced
    # cache (with the v2/06_PREDICTIONS path component baked into the temp
    # layout).
    runtime_cfg, diff = config_runtime.derive_runtime_config(
        frozen_config_path=PILOT_CFG,
        prediction_cache_path=str(real_seed6521_cache_dir),
        output_dir=str(tmp_path / "m9_run"),
        validation_split_override="rl_validation",
        pilot_scenario_banks=(
            paths.TRAINING_BANK,
            paths.PILOT_VALIDATION_BANK,
        ),
    )
    runtime_cfg_path = tmp_path / "seed_6521_runtime_config.json"
    runtime_cfg_path.write_text(json.dumps(runtime_cfg, indent=2))

    # The pilot remediation: validation_split override was applied.
    assert "environment.validation_split" in diff
    assert diff["environment.validation_split"]["old"] == "predictor_train"
    assert diff["environment.validation_split"]["new"] == "rl_validation"
    # The pilot remediation: distinct regime-matched banks were applied.
    assert "environment.training_scenario_bank_path" in diff
    assert diff["environment.training_scenario_bank_path"]["old"] is None
    assert diff["environment.training_scenario_bank_path"]["new"] == paths.TRAINING_BANK
    assert "environment.validation_scenario_bank_path" in diff
    assert diff["environment.validation_scenario_bank_path"]["old"] is None
    assert diff["environment.validation_scenario_bank_path"]["new"] == paths.PILOT_VALIDATION_BANK
    # The runtime config's validation_split is now rl_validation (not predictor_train).
    assert runtime_cfg["environment"]["validation_split"] == "rl_validation"
    # The runtime config carries distinct regime-matched bank paths.
    assert runtime_cfg["environment"]["training_scenario_bank_path"] == paths.TRAINING_BANK
    assert runtime_cfg["environment"]["validation_scenario_bank_path"] == paths.PILOT_VALIDATION_BANK

    rid = pairing.run_id_for_seed(6521)

    # Assert run_dir absent before launch.
    run_dir = Path(tmp_path / "m9_run") / rid
    wrapper.assert_run_dir_absent(run_dir)

    # Defense-in-depth: the runtime config's split is NOT rl_test.
    wrapper.assert_no_rl_test_split(runtime_cfg["environment"])

    # Frozen regime-matched bank paths (resolved absolute, asserted on disk).
    training_bank = paths.resolve_bank_path(paths.training_bank_path())
    pilot_validation_bank = paths.resolve_bank_path(paths.pilot_validation_bank_path())

    rec = wrapper.run_validate_only(
        6521, runtime_cfg_path, rid,
        training_scenario_bank=training_bank,
        validation_scenario_bank=pilot_validation_bank,
    )
    assert rec.returncode == 0, f"validate-only failed:\n{rec.stderr}"
    assert rec.executable == sys.executable
    assert rec.shell is False
    assert "--validate-only" in rec.command
    assert "--config" in rec.command and str(runtime_cfg_path) in rec.command
    assert "--run-id" in rec.command and rid in rec.command
    assert "--training-seed" in rec.command and "6521" in rec.command
    # The mandatory explicit-bank flags are present.
    assert "--training-scenario-bank" in rec.command
    assert "--validation-scenario-bank" in rec.command
    assert training_bank in rec.command
    assert pilot_validation_bank in rec.command
    # validate-only must NOT have created any run output dir.
    assert not run_dir.exists(), f"validate-only must not create run_dir: {run_dir}"
    # No training artifacts in the worktree.
    assert not Path(tmp_path / "m9_run").exists() or not any(
        Path(tmp_path / "m9_run").iterdir()
    ), "validate-only must not populate output_dir"
