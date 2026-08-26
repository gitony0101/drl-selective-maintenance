"""M9 point-estimate fail-closed execution wrapper.

Drives the frozen ``scripts/train_ddqn.py``, ``evaluate_ddqn.py`` and baseline
CLIs as subprocesses (per the cache-generation provenance discipline) with full
command/exe/cwd/exit/log capture, writes a result manifest, enforces serial
one-process execution, and adds M9-specific fail-closed guards on top of the
frozen stack.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Mapping, Optional

from . import manifest


_REPO_ROOT = manifest.REPO_ROOT
TRAIN_DDQN_REL = "scripts/train_ddqn.py"

_FORBIDDEN_SPLIT = "rl_test"


def assert_no_rl_test_split(effective_config: Mapping) -> None:
    """Reject any rl_test split at the wrapper level BEFORE invoking a frozen CLI.

    The frozen pipeline already rejects rl_test at
    ``validate_row_asset_contract`` (preflight) and ``TrainerConfig.__post_init__``;
    this is explicit defense-in-depth so a misconfigured M9 command fails closed
    before a subprocess is launched.

    Raises:
        ValueError: if effective_config's ``split`` or ``validation_split`` is
            rl_test.
    """
    for key in ("split", "validation_split"):
        if effective_config.get(key) == _FORBIDDEN_SPLIT:
            raise ValueError(
                f"M9 wrapper: {key}='rl_test' is FORBIDDEN. rl_test is sealed "
                f"for the entire point-estimate pipeline."
            )


def assert_run_dir_absent(run_dir: Path) -> None:
    """Invariant 12: the trainer's ``run_dir = output_dir/run_id`` must NOT
    pre-exist. The frozen trainer does ``mkdir(exist_ok=True)`` and silently
    overwrites a collision (src/training/ddqn_trainer.py), so the wrapper must
    refuse to launch when the run_dir already exists.

    Raises ValueError if the path exists (empty or otherwise). Pass-by-refuse:
    no creation here; just the assertion BEFORE the subprocess.
    """
    if Path(run_dir).exists():
        raise ValueError(
            f"M9 wrapper: run_dir already exists — refusing to silently "
            f"overwrite (trainer uses mkdir(exist_ok=True)): {run_dir}"
        )


def build_training_argv(
    config_path: Path,
    cache_path: str,
    output_dir: str,
    run_id: str,
    seed: int,
    training_scenario_bank: str,
    validation_scenario_bank: str,
    resume: bool = False,
    max_steps_override: Optional[int] = None,
    validate_only: bool = False,
    device: Optional[str] = None,
) -> List[str]:
    """Build the exact ``train_ddqn.py`` subprocess argv for one M9 point-estimate run.

    The returned argv is ``sys.executable + script + flags`` with:
      - ``--config <runtime config JSON>`` (the per-seed runtime config carrying
        the per-seed cache path + external output_dir; derived by
        ``config_runtime.derive_runtime_config``).
      - ``--training-seed <seed>`` (the resolver recognizes this CLI override).
      - ``--run-id <run_id>`` (the per-seed unique run id).
      - ``--training-scenario-bank <path>`` + ``--validation-scenario-bank
        <path>`` (MANDATORY: the frozen resolver's
        ``assert_explicit_banks`` gate at ``resolver.py:193-220`` rejects ANY
        command — dry-run, validate-only, smoke, real training, resume — that
        omits both flags. There is NO opt-out; the gate is source="argparse
        namespace" and refuses even a config that already carries the bank
        paths in its ``environment`` section. So the wrapper MUST forward both
        bank paths from the runtime config's environment section as explicit
        CLI flags. Banks are sourced from the frozen runtime config — NOT
        invented by the wrapper — and are regime-matched (K=2,
        failure-light-no-waste): formal uses ``m5_pilot_k2.json`` (predictor_train)
        + ``m5_validation_k2.json`` (rl_validation); pilot uses
        ``m5_pilot_k2.json`` + ``m5_validation_k2__light.json`` (rl_validation).

    ``--resume`` is REJECTED unless the caller explicitly sets ``resume=True``;
    even then, this builder NEVER emits ``--resume`` itself (the validation
    stage does not authorize resume — formal runs are non-resumable by design).

    ``--validate-only`` / ``--max-steps`` / ``--device`` may be passed; they
    are forwarded as bare/store-typed flags the resolver understands.
    """
    if resume:
        raise ValueError(
            "M9 wrapper: --resume is NOT authorized for the point-estimate "
            "pipeline (formal runs are non-resumable; on failure the "
            "matrix stops and evidence is preserved)."
        )
    if not training_scenario_bank or not validation_scenario_bank:
        raise ValueError(
            "M9 wrapper: the frozen resolver mandates explicit "
            "--training-scenario-bank and --validation-scenario-bank. "
            "Both bank paths (sourced from the runtime config's environment "
            "section) must be provided."
        )
    argv: List[str] = [
        sys.executable,
        str(_REPO_ROOT / TRAIN_DDQN_REL),
        "--config", str(config_path),
        "--training-seed", str(seed),
        "--run-id", str(run_id),
        "--training-scenario-bank", str(training_scenario_bank),
        "--validation-scenario-bank", str(validation_scenario_bank),
    ]
    if max_steps_override is not None:
        argv += ["--max-steps", str(int(max_steps_override))]
    if device is not None:
        argv += ["--device", str(device)]
    if validate_only:
        argv += ["--validate-only"]
    return argv


def run_validate_only(
    seed: int,
    runtime_config_path: Path,
    run_id: str,
    training_scenario_bank: str,
    validation_scenario_bank: str,
):
    """Validation stage: invoke ``train_ddqn.py --validate-only`` for one seed.

    Subprocess provenance captured (argv/exe/cwd/returncode/stdout/stderr) and
    fail-closed on a non-zero exit (the asset preflight must pass before any
    real training).

    The two mandatory scenario-bank paths are passed EXPLICITLY (the caller
    sources them from the frozen config's regime-matched banks): the formal
    run uses ``m5_pilot_k2.json`` (predictor_train) + ``m5_validation_k2.json``
    (rl_validation); the pilot uses ``m5_pilot_k2.json`` +
    ``m5_validation_k2__light.json`` (rl_validation). The frozen resolver's
    ``assert_explicit_banks`` gate rejects any command that omits these CLI
    flags — there is NO opt-out.

    Returns a small dataclass with the command + outputs.
    """
    import subprocess
    from dataclasses import dataclass

    argv = build_training_argv(
        config_path=runtime_config_path,
        cache_path="(via config)",
        output_dir="(via config)",
        run_id=run_id,
        seed=seed,
        training_scenario_bank=training_scenario_bank,
        validation_scenario_bank=validation_scenario_bank,
        resume=False,
        validate_only=True,
    )

    completed = subprocess.run(
        argv, shell=False, cwd=str(_REPO_ROOT),
        capture_output=True, text=True, check=False,
    )

    @dataclass
    class ValidateRecord:
        seed: int
        command: List[str]
        executable: str
        cwd: str
        shell: bool
        returncode: int
        stdout: str
        stderr: str

    rec = ValidateRecord(
        seed=seed, command=argv, executable=sys.executable,
        cwd=str(_REPO_ROOT), shell=False,
        returncode=completed.returncode,
        stdout=completed.stdout, stderr=completed.stderr,
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode, argv, output=completed.stdout, stderr=completed.stderr
        )
    return rec


def run_training(
    seed: int,
    runtime_config_path: Path,
    run_dir: Path,
    run_id: str,
    training_scenario_bank: str,
    validation_scenario_bank: str,
    max_steps_override: Optional[int] = None,
    device: Optional[str] = None,
):
    """Formal stage: invoke ``scripts/train_ddqn.py`` for ONE seed (real training).

    Runs the frozen trainer as a subprocess with the same provenance-capture
    discipline as cache generation: ``sys.executable`` + explicit argv,
    ``shell=False``, fixed worktree cwd, ``capture_output=True``. The two
    mandatory scenario-bank flags are passed explicitly (the frozen resolver's
    ``assert_explicit_banks`` gate rejects any command that omits them; there
    is NO opt-out). Fail-closed on a non-zero exit: the failing
    ``CalledProcessError`` carries the full command + captured stdout/stderr
    so the caller can record it before stopping the matrix (formal runs are
    NON-RESUMABLE — on failure the matrix stops and evidence is preserved).

    The run_dir MUST be asserted absent BEFORE launch (the trainer does
    ``mkdir(exist_ok=True)`` and silently overwrites a collision; this guard
    is the only protection). ``--resume`` is never emitted (formal runs
    are non-resumable by design). ``--validate-only`` is never emitted (use
    ``run_validate_only`` for the asset preflight gate).

    Returns a TrainingRecord (command/exe/cwd/shell/returncode/stdout/stderr).
    """
    import subprocess
    from dataclasses import dataclass

    assert_run_dir_absent(run_dir)

    argv = build_training_argv(
        config_path=runtime_config_path,
        cache_path="(via config)",
        output_dir="(via config)",
        run_id=run_id,
        seed=seed,
        training_scenario_bank=training_scenario_bank,
        validation_scenario_bank=validation_scenario_bank,
        resume=False,
        max_steps_override=max_steps_override,
        validate_only=False,
        device=device,
    )

    completed = subprocess.run(
        argv, shell=False, cwd=str(_REPO_ROOT),
        capture_output=True, text=True, check=False,
    )

    @dataclass
    class TrainingRecord:
        seed: int
        command: List[str]
        executable: str
        cwd: str
        shell: bool
        returncode: int
        stdout: str
        stderr: str

    rec = TrainingRecord(
        seed=seed, command=argv, executable=sys.executable,
        cwd=str(_REPO_ROOT), shell=False,
        returncode=completed.returncode,
        stdout=completed.stdout, stderr=completed.stderr,
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode, argv, output=completed.stdout, stderr=completed.stderr
        )
    return rec
