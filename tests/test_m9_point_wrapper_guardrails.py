"""M9 Point-Estimate — no-overwrite / no-resume (Step 9, invariant 12).

The DDQN trainer constructs ``run_dir = Path(output_dir)/run_id`` and then
``mkdir(exist_ok=True)`` — silently overwriting an existing run_id on collision
(src/training/ddqn_trainer.py setup). The wrapper MUST prevent this:
  - assert the run_dir does NOT pre-exist before training;
  - never pass ``--resume`` to train_ddqn.py unless explicitly authorized.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_assert_run_dir_absent_raises_if_pre_existing(tmp_path):
    """Calling assert_run_dir_absent on a path that already exists raises ValueError."""
    from src.milestone9.point import wrapper

    pre = tmp_path / "existing_run"
    pre.mkdir()
    with pytest.raises(ValueError, match="already exists|refus|overwrite"):
        wrapper.assert_run_dir_absent(pre)


def test_assert_run_dir_absent_passes_when_absent(tmp_path):
    """assert_run_dir_absent is a no-op (no raise) when the path does not exist."""
    from src.milestone9.point import wrapper

    absent = tmp_path / "never_created"
    wrapper.assert_run_dir_absent(absent)  # must not raise


def test_assert_run_dir_absent_rejects_non_empty_even_if_created(tmp_path):
    """If the run_dir exists at all (empty or not), refusal is the safe default."""
    from src.milestone9.point import wrapper

    pre = tmp_path / "empty_existing"
    pre.mkdir()
    with pytest.raises(ValueError):
        wrapper.assert_run_dir_absent(pre)


def test_never_resume_unless_authorized_rejects_resume_flag():
    """The training argv builder REQUIRES an explicit authorization when the
    caller asks to resume (``resume=True``). Without it, ValueError; the M9
    wrapper never passes ``--resume`` itself."""
    from src.milestone9.point import wrapper

    # resume=True with no explicit authorization -> raise.
    with pytest.raises(ValueError, match="resume"):
        wrapper.build_training_argv(
            config_path=REPO_ROOT / "configs" / "agents" / "ddqn_pilot_k2.json",
            cache_path="/tmp/cache",
            output_dir="/tmp/out",
            run_id="m9_point_mse_control_seed_6521",
            seed=6521,
            training_scenario_bank="configs/scenarios/m5_pilot_k2.json",
            validation_scenario_bank="configs/scenarios/m5_validation_k2__light.json",
            resume=True,
            max_steps_override=None,
        )


def test_training_argv_never_includes_resume_by_default():
    """The default training argv NEVER contains --resume."""
    from src.milestone9.point import wrapper

    argv = wrapper.build_training_argv(
        config_path=REPO_ROOT / "configs" / "agents" / "ddqn_pilot_k2.json",
        cache_path="/tmp/cache",
        output_dir="/tmp/out",
        run_id="m9_point_mse_control_seed_6521",
        seed=6521,
        training_scenario_bank="configs/scenarios/m5_pilot_k2.json",
        validation_scenario_bank="configs/scenarios/m5_validation_k2__light.json",
        resume=False,
        max_steps_override=None,
    )
    assert "--resume" not in argv


def test_training_argv_unique_run_id_per_seed():
    """The wrapper's run_id is unique per seed (m9_point_mse_control_seed<s>)."""
    from src.milestone9.point import pairing, wrapper

    for s in (6521, 6522, 6523):
        rid = pairing.run_id_for_seed(s)
        argv = wrapper.build_training_argv(
            config_path=REPO_ROOT / "configs" / "agents" / "ddqn_pilot_k2.json",
            cache_path="/tmp/cache",
            output_dir="/tmp/out",
            run_id=rid,
            seed=s,
            training_scenario_bank="configs/scenarios/m5_pilot_k2.json",
            validation_scenario_bank="configs/scenarios/m5_validation_k2__light.json",
            resume=False,
            max_steps_override=None,
        )
        # --run-id value is rid, unique per seed.
        rid_idx = argv.index("--run-id")
        assert argv[rid_idx + 1] == rid
        assert str(s) in rid


def test_training_argv_includes_config_and_required_flags():
    """The argv carries --config (the per-seed runtime config), --run-id,
    --training-seed, and the two mandatory explicit-bank flags."""
    from src.milestone9.point import wrapper

    argv = wrapper.build_training_argv(
        config_path=REPO_ROOT / "configs" / "agents" / "ddqn_pilot_k2.json",
        cache_path="/tmp/cache",
        output_dir="/tmp/out",
        run_id="m9_point_mse_control_seed_6521",
        seed=6521,
        training_scenario_bank="configs/scenarios/m5_pilot_k2.json",
        validation_scenario_bank="configs/scenarios/m5_validation_k2__light.json",
        resume=False,
        max_steps_override=None,
    )
    assert "--config" in argv
    assert "--run-id" in argv
    assert "--training-seed" in argv
    # The --config value must be the passed config path.
    cidx = argv.index("--config")
    assert argv[cidx + 1] == str(REPO_ROOT / "configs" / "agents" / "ddqn_pilot_k2.json")
    # Mandatory explicit-bank flags (frozen resolver asserts_explicit_banks gate).
    assert "--training-scenario-bank" in argv
    assert "--validation-scenario-bank" in argv
    tb = argv.index("--training-scenario-bank")
    vb = argv.index("--validation-scenario-bank")
    assert argv[tb + 1] == "configs/scenarios/m5_pilot_k2.json"
    assert argv[vb + 1] == "configs/scenarios/m5_validation_k2__light.json"


def test_build_training_argv_rejects_missing_banks():
    """The frozen resolver's explicit-bank gate rejects any command missing the
    two bank flags; the wrapper mirrors this by raising if either bank path
    is empty/None."""
    from src.milestone9.point import wrapper

    with pytest.raises(ValueError, match="explicit-bank|bank|mandatory"):
        wrapper.build_training_argv(
            config_path=REPO_ROOT / "configs" / "agents" / "ddqn_pilot_k2.json",
            cache_path="/tmp/cache",
            output_dir="/tmp/out",
            run_id="m9_point_mse_control_seed_6521",
            seed=6521,
            training_scenario_bank="",
            validation_scenario_bank="configs/scenarios/m5_validation_k2__light.json",
            resume=False,
            max_steps_override=None,
        )
