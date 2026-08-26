"""M9 Point-Estimate -- wrapper.run_training.

``wrapper.run_training`` drives the frozen ``scripts/train_ddqn.py`` for ONE
seed as a subprocess (per the cache-generation / provenance-capture
discipline): ``sys.executable`` + explicit argv, ``shell=False``, fixed
fixed cwd, ``capture_output=True``, fail-closed on a non-zero exit. It
MUST refuse to launch when the run_dir already exists (the trainer's
``mkdir(exist_ok=True)`` would silently overwrite a collision), MUST reject
``--resume`` (formal runs are non-resumable), and MUST capture the full
command/exe/cwd/returncode/stdout/stderr into a record.

This test suite MOCKS the subprocess boundary (fast unit tests). The real
end-to-end training path is exercised by the slow pilot run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REPO_ROOT = Path(__file__).resolve().parent.parent
PILOT_CFG = REPO_ROOT / "configs" / "agents" / "ddqn_pilot_k2.json"


class _FakeCompleted:
    def __init__(self, returncode: int = 0, stdout: str = "ok", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_run_training_invokes_train_ddqn_subprocess_with_provenance(tmp_path, monkeypatch):
    """run_training shells out to scripts/train_ddqn.py with sys.executable,
    shell=False, fixed cwd, capture_output=True; the returned record
    carries the full command/exe/cwd/returncode/stdout/stderr."""
    from src.milestone9.point import wrapper

    captured = {}

    def fake_run(argv, **kw):
        captured["argv"] = argv
        captured["kw"] = kw
        return _FakeCompleted(stdout="trained", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    runtime_cfg = tmp_path / "runtime.json"
    runtime_cfg.write_text("{}")
    run_dir = tmp_path / "runs" / "m9_point_mse_control_seed_6521"
    rec = wrapper.run_training(
        seed=6521,
        runtime_config_path=runtime_cfg,
        run_dir=run_dir,
        run_id="m9_point_mse_control_seed_6521",
        training_scenario_bank="configs/scenarios/m5_pilot_k2.json",
        validation_scenario_bank="configs/scenarios/m5_validation_k2__light.json",
        max_steps_override=5000,
    )
    # Subprocess provenance captured.
    assert rec.executable == sys.executable
    assert rec.shell is False
    assert rec.cwd == str(REPO_ROOT)
    assert rec.returncode == 0
    assert rec.stdout == "trained"
    assert rec.stderr == ""
    kw = captured["kw"]
    assert kw["shell"] is False
    assert kw["cwd"] == str(REPO_ROOT)
    assert kw["capture_output"] is True
    argv = captured["argv"]
    assert argv[0] == sys.executable
    assert argv[1].endswith("scripts/train_ddqn.py")
    assert "--config" in argv and str(runtime_cfg) in argv
    assert "--training-seed" in argv and "6521" in argv
    assert "--run-id" in argv and "m9_point_mse_control_seed_6521" in argv
    # Mandatory explicit-bank flags.
    assert "--training-scenario-bank" in argv
    assert "--validation-scenario-bank" in argv
    assert "configs/scenarios/m5_pilot_k2.json" in argv
    assert "configs/scenarios/m5_validation_k2__light.json" in argv
    # max_steps override forwarded.
    ms_idx = argv.index("--max-steps")
    assert argv[ms_idx + 1] == "5000"
    # Never --resume.
    assert "--resume" not in argv
    # Never --validate-only.
    assert "--validate-only" not in argv


def test_run_training_refuses_when_run_dir_pre_existing(tmp_path, monkeypatch):
    """run_training asserts the run_dir is absent BEFORE launching (the trainer
    would silently overwrite a collision via mkdir(exist_ok=True))."""
    from src.milestone9.point import wrapper

    run_dir = tmp_path / "runs" / "already_here"
    run_dir.mkdir(parents=True)

    def fake_run(argv, **kw):
        raise AssertionError("subprocess must not run when run_dir pre-exists")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="already exists|refus|overwrite"):
        wrapper.run_training(
            seed=6521,
            runtime_config_path=tmp_path / "runtime.json",
            run_dir=run_dir,
            run_id="m9_point_mse_control_seed_6521",
            training_scenario_bank="configs/scenarios/m5_pilot_k2.json",
            validation_scenario_bank="configs/scenarios/m5_validation_k2__light.json",
        )


def test_run_training_fails_closed_on_nonzero_exit(tmp_path, monkeypatch):
    """A non-zero training exit raises subprocess.CalledProcessError (the
    wrapper does NOT swallow training failures — provenance is preserved in
    the exception so the caller can record it)."""
    from src.milestone9.point import wrapper

    def fake_run(argv, **kw):
        return _FakeCompleted(returncode=1, stdout="partial", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError) as ei:
        wrapper.run_training(
            seed=6521,
            runtime_config_path=tmp_path / "runtime.json",
            run_dir=tmp_path / "runs" / "rid",
            run_id="rid",
            training_scenario_bank="configs/scenarios/m5_pilot_k2.json",
            validation_scenario_bank="configs/scenarios/m5_validation_k2__light.json",
        )
    # The failing command is preserved on the exception.
    assert ei.value.returncode == 1
    assert sys.executable in ei.value.cmd[0]


def test_run_training_rejects_resume(tmp_path):
    """run_training exposes no resume path — formal runs are non-resumable."""
    from src.milestone9.point import wrapper

    sig_params = wrapper.run_training.__doc__ or ""
    # The function signature carries no resume parameter.
    import inspect
    params = set(inspect.signature(wrapper.run_training).parameters.keys())
    assert "resume" not in params


def test_run_training_device_forwardable(tmp_path, monkeypatch):
    """An optional device flag is forwarded as --device (CPU is the M9 default
    for a CPU-only laptop run; the caller pins it explicitly for reproducibility)."""
    from src.milestone9.point import wrapper

    captured = {}

    def fake_run(argv, **kw):
        captured["argv"] = argv
        return _FakeCompleted()

    monkeypatch.setattr(subprocess, "run", fake_run)
    wrapper.run_training(
        seed=6521,
        runtime_config_path=tmp_path / "runtime.json",
        run_dir=tmp_path / "runs" / "rid",
        run_id="rid",
        training_scenario_bank="configs/scenarios/m5_pilot_k2.json",
        validation_scenario_bank="configs/scenarios/m5_validation_k2__light.json",
        device="cpu",
    )
    argv = captured["argv"]
    d_idx = argv.index("--device")
    assert argv[d_idx + 1] == "cpu"
