"""M9 point-estimate BASELINE-ONLY formal rerun.

Re-evaluates the corrected M9 baseline set against the EXISTING five formal
per-seed DDQN run evaluation contexts -- WITHOUT retraining any DDQN and
WITHOUT regenerating any cache. For each formal seed, loads the existing
``runtime_config.json`` from the frozen DDQN run dir (read-only -- the DDQN
artifacts are NOT modified), builds the SAME evaluation ``EnvironmentConfig``
the DDQN validated under (rl_validation / K=2 / failure-light-no-waste /
per-seed V2 cache / the m5_validation_k2.json bank / horizon=100 / 5 paired
episodes via FIXED_RESET_SEEDS), and runs the corrected ``run_baselines``.

Results are written to a SEPARATE external git-ignored directory
(``<container>/m9_point_runs/baseline_repair/seed_/``) -- NOT into the
frozen DDQN run dirs -- so the DDQN result manifests stay byte-identical
(existing DDQN runs must not be overwritten, renamed, or regenerated).

Each seed is evaluated transactionally: verify inputs -> temp output -> run ->
validate episode count == 5, all families present, metrics finite, provenance
present -> finalize -> COMPLETED marker. Run serially (one heavy process at
a time).

USAGE:
    python scripts/run_m9_baseline_repair.py
    python scripts/run_m9_baseline_repair.py --seeds 6521
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.milestone9.point import pairing
from src.milestone9.point.baselines import (
    run_baselines,
    env_config_for_eval,
    M9_REGIME_K,
    M9_REGIME_COST,
    M9_REGIME_SPLIT,
)

_CONTAINER_ROOT = pairing._CONTAINER_ROOT
_FORMAL_RUNS = _CONTAINER_ROOT / "m9_point_runs" / "formal"
_OUT_ROOT = _CONTAINER_ROOT / "m9_point_runs" / "baseline_repair"

_SEEDS = [6521, 6522, 6523, 6524, 6525]
_RESET_SEEDS = [6521, 6522, 6523, 6524, 6525]  # frozen 5 paired episodes

_BASELINE_FAMILIES = [
    "corrective_only", "random_feasible", "age_threshold",
    "predicted_rul_threshold", "greedy_predicted_rul", "exact_myopic",
]


def _is_finite(x: Any) -> bool:
    if isinstance(x, (int, float)):
        return not (math.isnan(x) or math.isinf(x))
    return True


def _validate_finite(obj: Any, path: str = "") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _validate_finite(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _validate_finite(v, f"{path}[{i}]")
    else:
        if not _is_finite(obj):
            raise ValueError(f"non-finite value at {path}: {obj!r}")


def _run_one_seed(seed: int, out_root: Path) -> Dict[str, Any]:
    """Transactionally run the corrected baselines for one formal seed."""
    run_id = pairing.run_id_for_seed(seed)
    ddqn_run_dir = _FORMAL_RUNS / run_id
    runtime_cfg_path = ddqn_run_dir / "runtime_config.json"
    if not runtime_cfg_path.exists():
        raise FileNotFoundError(
            f"DDQN runtime config missing for seed {seed}: {runtime_cfg_path}"
        )

    runtime_cfg = json.loads(runtime_cfg_path.read_text())
    env_config = env_config_for_eval(runtime_cfg)

    # Fail-closed guards (mirrors run_baselines; repeated here for an early,
    # readable failure before any evaluation).
    if env_config.split != M9_REGIME_SPLIT:
        raise ValueError(f"seed {seed}: split={env_config.split!r} != {M9_REGIME_SPLIT!r}")
    if env_config.maintenance_capacity != M9_REGIME_K:
        raise ValueError(f"seed {seed}: K={env_config.maintenance_capacity} != {M9_REGIME_K}")
    if env_config.cost_regime_id != M9_REGIME_COST:
        raise ValueError(f"seed {seed}: cost={env_config.cost_regime_id!r} != {M9_REGIME_COST!r}")

    bank_path = env_config.scenario_bank_path

    out_dir = out_root / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "baseline_repair_results.json"

    print("=" * 70)
    print(f"[seed {seed}] corrected baseline rerun")
    print(f"  ddqn run dir: {ddqn_run_dir} (READ-ONLY, not modified)")
    print(f"  eval split: {env_config.split} | K={env_config.maintenance_capacity} "
          f"| cost={env_config.cost_regime_id} | horizon={env_config.episode_horizon}")
    print(f"  cache: {env_config.prediction_cache_path}")
    print(f"  bank: {bank_path} | reset_seeds={_RESET_SEEDS}")
    print(f"  output: {results_path}")

    start = time.time()
    results = run_baselines(
        env_config=env_config,
        scenario_bank_path=bank_path,
        reset_seeds=_RESET_SEEDS,
        ddqn_seed=seed,
    )
    elapsed = time.time() - start

    provenance = results.get("__provenance__", {})

    # Validation (transactional finalize).
    errors: List[str] = []
    families_present = set(k for k in results if k != "__provenance__")
    if families_present != set(_BASELINE_FAMILIES):
        errors.append(f"family set mismatch: {sorted(families_present)}")
    for fam in _BASELINE_FAMILIES:
        rec = results.get(fam, {})
        n = rec.get("num_episodes")
        if n != len(_RESET_SEEDS):
            errors.append(f"{fam}: num_episodes={n} != {len(_RESET_SEEDS)}")
        mc = rec.get("mean_total_cost")
        if not _is_finite(mc) or mc < 0:
            errors.append(f"{fam}: mean_total_cost={mc} not finite/non-negative")
        if fam == "exact_myopic":
            ad = rec.get("action_distribution", {})
            if ad.get("n_actions") != 16:
                errors.append(f"{fam}: n_actions={ad.get('n_actions')} != 16")
        if fam == "age_threshold":
            if rec.get("threshold") != provenance.get("m3_selected_thresholds", {}).get("age_threshold_k2_failure-light-no-waste"):
                errors.append(f"{fam}: threshold {rec.get('threshold')} != provenance M3 age threshold")
    if not results.get("__provenance__"):
        errors.append("missing __provenance__ block")
    # Provenance SHAs present
    m3prov = provenance.get("m3_selected_thresholds", {})
    m4prov = provenance.get("m4_scientific_selection", {})
    if not m3prov.get("sha256"):
        errors.append("provenance missing m3 selected_thresholds sha256")
    if not m4prov.get("decision_sha256"):
        errors.append("provenance missing m4 selection decision sha256")

    _validate_finite(results)

    payload = {
        "seed": seed,
        "ddqn_run_dir_read_only": str(ddqn_run_dir),
        "runtime_config_path": str(runtime_cfg_path),
        "ddqn_runtime_config_environment": runtime_cfg.get("environment"),
        "reset_seeds": _RESET_SEEDS,
        "elapsed_seconds": elapsed,
        "completed": not errors,
        "errors": errors,
        "results": results,
    }
    # Atomic finalize: write to temp then rename.
    tmp = results_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(results_path)

    print(f"  elapsed: {elapsed:.1f}s | completed: {payload['completed']}")
    if errors:
        print(f"  ERRORS: {errors}")
    print("  per-family summary:")
    for fam in _BASELINE_FAMILIES:
        rec = results.get(fam, {})
        print(f"    {fam:28s} mean_cost={rec.get('mean_total_cost'):>7.2f} "
              f"median={rec.get('median_total_cost'):>7.2f} "
              f"pm={rec.get('total_pm_actions')} "
              f"fail={rec.get('total_failures')} "
              f"cat={rec.get('catastrophic_episodes')} "
              f"eps={rec.get('num_episodes')}")
    # Compact DDQN cost alongside for convenience.
    vm = json.loads((ddqn_run_dir / "validation_metrics.json").read_text())
    last = vm[-1] if isinstance(vm, list) else vm
    print(f"    {'DDQN (frozen)':28s} mean_cost={last.get('mean_total_cost'):>7.2f} "
          f"pm={last.get('total_pm_actions')} fail={last.get('total_failures')} "
          f"eps={last.get('num_episodes')}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="M9 point-estimate corrected baseline-only formal rerun")
    ap.add_argument("--seeds", type=str, default=None,
                    help="comma-separated seeds (default: 6521-6525)")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else _SEEDS

    _OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"output root: {_OUT_ROOT}")
    print(f"seeds: {seeds} (serial)")

    all_payloads: List[Dict[str, Any]] = []
    any_errors = False
    for seed in seeds:
        payload = _run_one_seed(seed, _OUT_ROOT)
        all_payloads.append(payload)
        if not payload["completed"]:
            any_errors = True

    # Cross-seed index.
    index_path = _OUT_ROOT / "baseline_repair_index.json"
    index = {
        "seeds": seeds,
        "reset_seeds": _RESET_SEEDS,
        "baseline_families": _BASELINE_FAMILIES,
        "episode_design": "5 paired episodes per seed (1:1 scenario_id x reset_seed)",
        "per_seed": {
            str(p["seed"]): {
                "completed": p["completed"],
                "errors": p["errors"],
                "results_path": str(_OUT_ROOT / f"seed_{p['seed']}" / "baseline_repair_results.json"),
            } for p in all_payloads
        },
    }
    index_path.write_text(json.dumps(index, indent=2))

    print("=" * 70)
    print(f"index: {index_path}")
    if any_errors:
        print("STATUS: FAILED (one or more seeds had errors)")
        return 1
    print("STATUS: COMPLETED (all seeds validated: 5 episodes, 6 families, finite, provenance)")
    return 0


if __name__ == "__main__":
    sys.exit(main())