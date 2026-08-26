"""M9 point-estimate one-seed driver (pilot / formal modes).

Orchestrates ONE seed end-to-end through the tested M9 primitives:

  1. Load the frozen M9 contract + the per-seed manifest (manifest.load_frozen_checkpoints).
  2. Ensure the per-seed V2 cache exists at cache_env_path_for_seed(seed); regenerate
     via cache_prep.generate_for_seed if missing (provenance-captured subprocess).
  3. Derive the per-seed runtime config from the frozen ddqn config
     (config_runtime.derive_runtime_config); write it under run_dir.
  4. validate-only gate  (wrapper.run_validate_only) — the frozen asset preflight
     (banks load, K/cost_regime/split match, cache compatibility, NO rl_test).
  5. Real training  (wrapper.run_training) — scripts/train_ddqn.py, max_steps from
     the frozen pilot/formal budget, fail-closed on non-zero exit.
  6. Evaluate the best checkpoint on rl_validation (scripts/evaluate_ddqn.py) — subprocess.
  7. Baselines on rl_validation (M4 exact-myopic + M3 non-oracle families) — subprocesses.
  8. Write run_dir/result_manifest.json capturing EVERY subprocess's command/exe/cwd/
     exit/logs/identities + the training + eval + baseline metrics refs.

Every subprocess uses the provenance-capture discipline: sys.executable + explicit
argv, shell=False, fixed worktree cwd, capture_output=True, check=True (or
fail-closed on non-zero). The driver is a thin orchestrator over the tested
primitives (config_runtime, cache_prep, pairing, paths, wrapper).

USAGE:
  python scripts/run_m9_point_estimate.py --seed 6521 --phase pilot
  python scripts/run_m9_point_estimate.py --seed 6521 --phase formal --max-steps 2000

Pilot seeds: 6521, 6522 (serial; one heavy process at a time).
Formal seeds: 6521-6525 (serial; max_steps defaults to the frozen formal
budget 100000 unless --max-steps is explicitly passed for a debug-budget run).

External git-ignored roots:
  <container>/m9_point_caches/seed_<s>/...        — per-seed V2 caches
  <container>/m9_point_runs/<run_id>/             — training + eval + baselines + manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.milestone9.point import config_runtime, pairing, paths, wrapper
from src.milestone9.point import cache_prep
from src.milestone9.point.baselines import run_baselines, env_config_for_eval


_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONTAINER_ROOT = pairing._CONTAINER_ROOT
_RUNS_ROOT = _CONTAINER_ROOT / "m9_point_runs"

PILOT_CFG = _REPO_ROOT / "configs" / "agents" / "ddqn_pilot_k2.json"
FORMAL_CFG = _REPO_ROOT / "configs" / "agents" / "ddqn_v1.json"


@dataclass
class SubprocessRecord:
    """Minimal provenance record for an evaluate/baseline subprocess."""
    name: str
    command: List[str]
    executable: str
    cwd: str
    shell: bool
    returncode: int
    stdout: str
    stderr: str


def _run_subprocess(name: str, argv: List[str], cwd: Path, check: bool = True) -> SubprocessRecord:
    """Run a subprocess with full provenance capture; return a record."""
    completed = subprocess.run(
        argv, shell=False, cwd=str(cwd), capture_output=True, text=True, check=False,
    )
    rec = SubprocessRecord(
        name=name, command=argv, executable=sys.executable, cwd=str(cwd),
        shell=False, returncode=completed.returncode,
        stdout=completed.stdout, stderr=completed.stderr,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"{name} failed (rc={completed.returncode}):\nSTDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )
    return rec


def _ensure_cache(seed: int) -> Dict[str, Any]:
    """Ensure the per-seed V2 cache exists at cache_env_path_for_seed(seed) and
    binds to the frozen SHA256. Regenerate via cache_prep if missing."""
    env_path = pairing.cache_env_path_for_seed(seed)
    manifest_path = env_path / "prediction_cache_manifest_v2.json"
    if manifest_path.exists():
        pairing.validate_cache_pairing(seed, manifest_path)
        return {
            "cache_env_path": str(env_path),
            "cache_manifest_path": str(manifest_path),
            "regenerated": False,
            "provenance": None,
        }
    env_path.mkdir(parents=True, exist_ok=True)
    rec = cache_prep.generate_for_seed(seed, env_path)
    return {
        "cache_env_path": str(env_path),
        "cache_manifest_path": str(manifest_path),
        "regenerated": True,
        "provenance": {
            "command": rec.command, "executable": rec.executable, "cwd": rec.cwd,
            "shell": rec.shell, "check": rec.check, "returncode": rec.returncode,
            "stdout": rec.stdout, "stderr": rec.stderr,
            "checkpoint_sha256": rec.checkpoint_sha256,
            "cache_manifest_sha256": rec.cache_manifest_sha256,
            "training_summary": rec.training_summary,
            "adapter_provenance": rec.adapter_provenance,
        },
    }


def _run_evaluate(seed: int, runtime_cfg_path: Path, run_dir: Path) -> SubprocessRecord:
    """Evaluate the trained best checkpoint on rl_validation (no --validate-only;
    no --resume; rl_validation only; never rl_test)."""
    checkpoint = run_dir / "checkpoint_best.pt"
    if not checkpoint.exists():
        # Fall back to checkpoint_latest if best was never saved (validation never
        # beat the initial baseline — pilot with val_interval==max_steps may save
        # a best at the final validation; otherwise latest is the only artifact).
        latest = run_dir / "checkpoint_latest.pt"
        if not latest.exists():
            raise FileNotFoundError(
                f"no checkpoint to evaluate: {checkpoint} (and {latest} absent)"
            )
        checkpoint = latest
    return _run_subprocess(
        "evaluate_ddqn",
        [sys.executable, str(_REPO_ROOT / "scripts" / "evaluate_ddqn.py"),
         "--checkpoint", str(checkpoint), "--config", str(runtime_cfg_path),
         "--split", "rl_validation"],
        cwd=_REPO_ROOT, check=True,
    )


def _run_m4_baseline() -> SubprocessRecord:
    """DEPRECATED stub kept for provenance-history; the real baseline step is
    ``_run_baselines`` (the M9-owned runner reusing PolicyEvaluator +
    ExactMyopicOptimizer against the per-seed cache). M4's CLI hard-wires the
    production cache dir which does not exist in this worktree and which the
    contract forbids writing to -- so the M9-owned runner replaces it."""
    raise RuntimeError("M4 CLI baseline is disabled; use _run_baselines")


def _run_m3_baselines(validate_only: bool = False) -> SubprocessRecord:
    """DEPRECATED stub; M3's --mode formal_closeout demands a sealed
    formal_run_context.json + selected_thresholds.json the M9 pilot has not
    staged, and M3 hard-wires the production cache. The M9-owned runner
    (_run_baselines) replaces it."""
    raise RuntimeError("M3 CLI baseline is disabled; use _run_baselines")


# The frozen M3 reset seeds (scripts/run_m3_baselines.py:96). The DDQN
# validation_metrics.json reports num_episodes==5 over these 5 reset seeds.
_RESET_SEEDS = [6521, 6522, 6523, 6524, 6525]


def _run_baselines(runtime_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run the authoritative M9 baseline set (5 rule families + M4
    exact-myopic) against the per-seed cache the DDQN trained on, on the SAME
    rl_validation bank + horizon=100 + reset seeds the DDQN evaluated under.
    In-process (no subprocess): reuses PolicyEvaluator + ExactMyopicOptimizer
    (the frozen primitives M3/M4 CLIs use) -- see src/milestone9/point/baselines.py
    for why an M9-owned runner replaces the production-hardwired CLIs."""
    eval_env = env_config_for_eval(runtime_cfg)
    return run_baselines(
        env_config=eval_env,
        scenario_bank_path=eval_env.scenario_bank_path,
        reset_seeds=_RESET_SEEDS,
    )


def _write_staging_runtime_config(run_dir: Path, run_id: str, runtime: Dict[str, Any]) -> Path:
    """Write the per-seed runtime config to a SIBLING staging path
    (``<runs_root>/<phase>/<run_id>_runtime_config.json``), creating the phase
    runs dir if it does not yet exist. The staging path is a SIBLING of run_dir
    (NOT inside it) so run_dir stays ABSENT for the trainer to create itself --
    the wrapper's run_training asserts run_dir is absent before invoking the
    trainer's ``mkdir(exist_ok=True)`` which would silently overwrite a
    collision. The config is copied INTO run_dir after training succeeds."""
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = run_dir.parent / f"{run_id}_runtime_config.json"
    staging.write_text(json.dumps(runtime, indent=2))
    return staging


def run_one_seed(seed: int, phase: str, max_steps_override: Optional[int] = None,
                 device: Optional[str] = None) -> Dict[str, Any]:
    """Run ONE seed end-to-end. Returns the result manifest dict."""
    recc = {
        "seed": seed, "phase": phase, "max_steps_override": max_steps_override,
        "device": device,
    }

    cache = _ensure_cache(seed)
    recc["cache"] = cache

    if phase == "pilot":
        frozen_cfg = PILOT_CFG
        val_override = "rl_validation"
        pilot_banks = (paths.TRAINING_BANK, paths.PILOT_VALIDATION_BANK)
        training_bank = paths.resolve_bank_path(paths.training_bank_path())
        validation_bank = paths.resolve_bank_path(paths.pilot_validation_bank_path())
        if max_steps_override is None:
            max_steps_override = 5000
    elif phase == "formal":
        frozen_cfg = FORMAL_CFG
        val_override = None
        pilot_banks = None
        training_bank = paths.resolve_bank_path(paths.training_bank_path())
        validation_bank = paths.resolve_bank_path(paths.formal_validation_bank_path())
    else:
        raise ValueError(f"phase must be 'pilot' or 'formal'; got {phase!r}")

    runtime, diff = config_runtime.derive_runtime_config(
        frozen_config_path=frozen_cfg,
        prediction_cache_path=str(pairing.cache_env_path_for_seed(seed)),
        output_dir=str(_RUNS_ROOT / phase),
        validation_split_override=val_override,
        pilot_scenario_banks=pilot_banks,
    )
    recc["config_runtime_diff"] = diff

    run_id = pairing.run_id_for_seed(seed)
    run_dir = _RUNS_ROOT / phase / run_id

    # The runtime config is written to a SIBLING staging path (NOT inside run_dir)
    # so run_dir stays ABSENT for the trainer to create itself. The wrapper's
    # run_training asserts run_dir is absent (the trainer's mkdir(exist_ok=True)
    # would silently overwrite a collision); if the driver pre-created run_dir to
    # hold the config, run_training would refuse. So the config lives next to
    # run_dir during validate-only/training, then is copied INTO run_dir after
    # training succeeds (for run self-containment). If a prior failed attempt
    # left run_dir non-empty, refuse (preserve evidence).
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(
            f"run_dir exists and is non-empty (refusing to clobber evidence): {run_dir}"
        )
    if run_dir.exists() and not any(run_dir.iterdir()):
        run_dir.rmdir()
    staging_cfg_path = _write_staging_runtime_config(run_dir, run_id, runtime)
    recc["runtime_config_path_staging"] = str(staging_cfg_path)
    recc["run_dir"] = str(run_dir)

    recc["validate_only"] = _vodict(wrapper.run_validate_only(
        seed, staging_cfg_path, run_id,
        training_scenario_bank=training_bank,
        validation_scenario_bank=validation_bank,
    ))

    train_rec = wrapper.run_training(
        seed, staging_cfg_path, run_dir, run_id,
        training_scenario_bank=training_bank,
        validation_scenario_bank=validation_bank,
        max_steps_override=max_steps_override,
        device=device,
    )
    recc["training"] = _tdict(train_rec)

    # Training created run_dir; archive the runtime config INTO it for
    # self-containment (evaluate reads the staging path, but the run should
    # carry its own config for reproducibility/audit).
    runtime_cfg_in_run = run_dir / "runtime_config.json"
    if not runtime_cfg_in_run.exists():
        runtime_cfg_in_run.write_text(staging_cfg_path.read_text())
    runtime_cfg_path = staging_cfg_path  # evaluate/baselines read the staging path
    recc["runtime_config_path"] = str(runtime_cfg_path)

    recc["evaluate"] = _sdict(_run_evaluate(seed, runtime_cfg_path, run_dir))

    # Baselines: the authoritative M9 set (5 rule families + M4 exact-myopic)
    # reusing the frozen PolicyEvaluator + ExactMyopicOptimizer against the
    # per-seed cache, on the SAME rl_validation bank + horizon=100 + reset
    # seeds the DDQN evaluated under (apples-to-apples). See baselines.py for
    # why the M9-owned runner replaces the production-hardwired M3/M4 CLIs.
    runtime_cfg_dict = json.loads(runtime_cfg_path.read_text())
    recc["baselines"] = _run_baselines(runtime_cfg_dict)

    # The training run_manifest.json (trainer-written) + validation_metrics +
    # resolved_config are siblings in run_dir; record their paths + SHA256s.
    for artifact in ("run_manifest.json", "validation_metrics.json",
                     "resolved_config.json", "checkpoint_best.pt",
                     "checkpoint_latest.pt"):
        p = run_dir / artifact
        if p.exists():
            recc.setdefault("run_artifacts", {})[artifact] = {
                "path": str(p),
                "sha256": _file_sha256(p) if p.name.endswith(".pt") else None,
            }

    # Write the result manifest.
    result_manifest_path = run_dir / "m9_point_result_manifest.json"
    result_manifest_path.write_text(json.dumps(recc, indent=2, default=str))
    recc["result_manifest_path"] = str(result_manifest_path)
    result_manifest_path.write_text(json.dumps(recc, indent=2, default=str))
    return recc


def _vodict(rec) -> Dict[str, Any]:
    return {"command": rec.command, "executable": rec.executable, "cwd": rec.cwd,
            "shell": rec.shell, "returncode": rec.returncode,
            "stdout": rec.stdout, "stderr": rec.stderr}


def _tdict(rec) -> Dict[str, Any]:
    return {"command": rec.command, "executable": rec.executable, "cwd": rec.cwd,
            "shell": rec.shell, "returncode": rec.returncode,
            "stdout": rec.stdout, "stderr": rec.stderr}


def _sdict(rec: SubprocessRecord) -> Dict[str, Any]:
    return {"name": rec.name, "command": rec.command, "executable": rec.executable,
            "cwd": rec.cwd, "shell": rec.shell, "returncode": rec.returncode,
            "stdout": rec.stdout, "stderr": rec.stderr}


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="M9 point-estimate one-seed driver")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--phase", choices=["pilot", "formal"], required=True)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    result = run_one_seed(
        seed=args.seed, phase=args.phase,
        max_steps_override=args.max_steps, device=args.device,
    )
    print(f"\n=== M9 point-estimate {args.phase} seed {args.seed} COMPLETE ===")
    print(f"run_dir: {result['run_dir']}")
    print(f"result_manifest: {result['result_manifest_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
