#!/usr/bin/env python3
"""
Milestone 3 Rule Baselines CLI.

Usage:
    python scripts/run_m3_baselines.py --help
    python scripts/run_m3_baselines.py --smoke
    python scripts/run_m3_baselines.py --tune
    python scripts/run_m3_baselines.py --evaluate
    python scripts/run_m3_baselines.py --all
    python scripts/run_m3_baselines.py --validate-artifacts

Barrier: rl_test split is rejected before data loading.

Formal --evaluate / --all / --tune are FAIL-CLOSED: the formal threshold
loader (load_formal_selected_thresholds) refuses to proceed if the
selected_thresholds.json file is missing, malformed, missing identities,
contains null/NaN/Inf, has a wrong run ID, wrong config hash, wrong
selected-threshold SHA, or runs Oracle without authorization. By design,
formal evaluation never falls back to "using defaults".

Smoke defaults are wired only inside run_smoke(); they MUST NOT be added
to the formal evaluation path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now import from src package
from src.baselines import (
    PolicyEvaluator,
    EvaluationConfig,
    EpisodeResult,
    tune_threshold,
    tune_all_thresholds,
    get_threshold_grid,
    THRESHOLD_POLICIES,
    NON_TUNED_POLICIES,
    AGE_THRESHOLDS,
    PREDICTED_RUL_THRESHOLDS,
    GREEDY_ACTIVATION_THRESHOLDS,
    ORACLE_THRESHOLDS,
    write_resolved_config,
    compute_canonical_config_sha256,
    read_resolved_config_sha256,
    write_threshold_search_results,
    write_threshold_search_summary,
    write_selected_thresholds,
    write_selected_thresholds_with_meta,
    write_episode_results,
    write_summary_by_policy,
    write_sanity_checks,
    write_run_provenance,
    write_scenario_bank_provenance,
    write_artifact_manifest,
    write_run_log,
    validate_artifacts,
    summarize_results,
    results_to_parquet,
    # Formal run context
    create_formal_run_context,
    seal_formal_run_context,
    load_formal_run_context,
    validate_formal_run_context,
)
from src.baselines.metrics import compute_summary_statistics

# Import M2 environment
from src.envs import (
    SelectiveMaintenanceEnv,
    EnvironmentConfig,
    get_default_config,
    list_cost_regimes,
)
from src.envs.scenario_bank import load_scenario_bank
from src.baselines.case_loader import get_scenario_bank_for_case, load_cases, CaseLoadResult


# Fixed reset seeds from M3 contract
FIXED_RESET_SEEDS = [6521, 6522, 6523, 6524, 6525]

# Policy families
POLICY_FAMILIES = [
    "corrective_only",
    "random_feasible",
    "age_threshold",
    "predicted_rul_threshold",
    "greedy_predicted_rul",
    "oracle_threshold",
]


def check_rl_test_barrier(split: str) -> None:
    """
    Reject rl_test split before data loading.

    Args:
        split: Requested split

    Raises:
        SystemExit: If split is rl_test
    """
    if split == "rl_test":
        print("ERROR: rl_test split is forbidden.", file=sys.stderr)
        print("Barrier: rl_test data must not be loaded for tuning or evaluation.", file=sys.stderr)
        print("Use predictor_train or rl_validation only.", file=sys.stderr)
        sys.exit(1)


# =============================================================================
# Formal selected-thresholds loader (FAIL-CLOSED)
# =============================================================================
#
# Formal evaluation MUST NOT print "using defaults" or substitute a default
# threshold value. A single strict loader is the only formal path; smoke may
# keep its own defaults inside run_smoke() because smoke is the only command
# that is allowed to substitute.
#
# Required identities for formal_closeout (4 threshold policies × 2 K × 4 regimes):
#     age_threshold              × {1,2} × 4 regimes
#     predicted_rul_threshold    × {1,2} × 4 regimes
#     greedy_predicted_rul       × {1,2} × 4 regimes
#     oracle_threshold           × {1,2} × 4 regimes
# Total: 32 identities.

FORMAL_POLICY_FAMILIES = (
    "age_threshold",
    "predicted_rul_threshold",
    "greedy_predicted_rul",
    "oracle_threshold",
)
FORMAL_K_VALUES = (1, 2)
FORMAL_COST_REGIMES = (
    "failure-light-no-waste",
    "failure-heavy-no-waste",
    "failure-light-waste-aware",
    "failure-heavy-waste-aware",
)
FORMAL_THRESHOLD_GRIDS: Dict[str, set] = {
    "age_threshold": {
        25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300,
    },
    "predicted_rul_threshold": {
        5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100,
    },
    "greedy_predicted_rul": {
        5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100,
    },
    "oracle_threshold": {
        1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50,
    },
}


def _identity_key(policy_family: str, k: int, cost_regime_id: str) -> str:
    """Build the canonical identity key matching what the writer uses."""
    return f"{policy_family}_k{k}_{cost_regime_id}"


# =============================================================================
# Formal Threshold Loader (typed, never touches os._exit)
# =============================================================================


class FormalThresholdError(RuntimeError):
    """Typed error raised by the formal threshold loader on any rejection.

    The loader is a pure validator: it NEVER calls ``os._exit``, NEVER
    prints ``LOADER_SUCCESS`` / ``LOADER_REJECTED`` markers (those are
    test-wrapping concerned only), NEVER substitutes a default
    threshold, and NEVER falls back to silent permissive parsing.

    All formal-threshold rejections must surface to the CLI boundary
    as a single raised ``FormalThresholdError`` carrying the canonical
    reason string; the CLI turns that into a 2-exit failure.
    """


def _revoke_formal(reason: str, errors: List[str]) -> None:
    """Accumulator helper used by the loader; keeps a list of reasons."""
    errors.append(reason)


def _abort_formal(reason: str) -> None:
    """Single-reason abort that raises a :class:`FormalThresholdError`.

    Used by every failure branch in :func:`load_formal_selected_thresholds`
    so test wrappers (or production code) catch a typed exception
    rather than relying on ``os._exit``. The CLI boundary catches this
    class and exits with code 2.
    """
    raise FormalThresholdError(reason)


def compute_sha256_file(path: Path) -> str:
    """Stream-compute SHA256 of an arbitrary file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def _revoke_formal(reason: str, errors: List[str]) -> None:
    """Accumulator-friendly abort for multi-error validation."""
    errors.append(reason)


def load_formal_selected_thresholds(
    selected_thresholds_path: Path,
    expected_run_id: str,
    expected_config_sha256: str,
    allow_oracle: bool,
    require_selected_sha: Optional[str] = None,
) -> Dict[str, int]:
    """
    Strict formal selected-thresholds loader.

    Refuses to proceed if any of the following is true:

      - file missing
      - JSON parse error
      - root is not a dict
      - any required identity (policy x K x regime) is missing
      - any extra or inconsistent identity is present
      - any threshold value is null, NaN, Inf, a malformed string, or
        outside the frozen formal threshold grid for that policy family
      - the recorded run_id, config_sha256, or selected_thresholds_sha256
        disagrees with what the caller expects
      - oracle_threshold identities are present but Oracle authorization
        is missing (`allow_oracle` is False)
      - allowed (oracle=True) but a non-oracle identity is incomplete

    Returns:
        Dict[identity_key, threshold_int] with exactly 32 entries.

    Raises:
        FormalThresholdError: on any rejection (no default substitution,
            no ``os._exit``). The CLI boundary catches this exception and
            exits the process with code 2.

    Notes:
        The loader is a pure validator. It NEVER calls ``os._exit``, NEVER
        prints ``LOADER_SUCCESS`` / ``LOADER_REJECTED`` markers (those
        concern only test wrappers), and NEVER substitutes a default
        threshold value. Selective maintenance evaluation must never run
        with a missing or unfilled threshold identity.
    """
    errors: List[str] = []

    if selected_thresholds_path is None:
        _revoke_formal("selected_thresholds_path is None", errors)
    elif not Path(selected_thresholds_path).exists():
        _revoke_formal(
            f"missing selected_thresholds.json: {selected_thresholds_path}", errors
        )

    if errors:
        for e in errors:
            _abort_formal(e)
        _abort_formal("unreachable")  # pragma: no cover

    path = Path(selected_thresholds_path)

    # Read raw bytes and (if a required SHA was supplied) compare SHA BEFORE
    # any JSON parsing. The strict loader never allows byte corruption to
    # reach the JSON parser; the SHA is computed on disk bytes verbatim,
    # so a single-byte mutation also changes the SHA, causing a fail-closed
    # rejection before parsing begins.
    actual_bytes_singleton = {"data": path.read_bytes()}
    if require_selected_sha is not None:
        actual_sha = hashlib.sha256(actual_bytes_singleton["data"]).hexdigest()
        if actual_sha != require_selected_sha:
            _abort_formal(
                f"selected_thresholds.json SHA256 mismatch: expected "
                f"{require_selected_sha}, got {actual_sha}"
            )

    # Parse JSON strictly.
    try:
        with open(path, "rb") as f:
            payload = json.loads(f.read().decode("utf-8"))
    except UnicodeDecodeError as e:
        _abort_formal(
            f"selected_thresholds.json is not valid UTF-8: {e}"
        )
    except json.JSONDecodeError as e:
        _abort_formal(f"selected_thresholds.json is not valid JSON: {e}")

    if not isinstance(payload, dict):
        _abort_formal(
            f"selected_thresholds.json root must be a dict, got {type(payload).__name__}"
        )

    # Top-level run-id, config-hash, and implementation-commit verification.
    # The writer is responsible for stamping a mandatory _meta envelope; a
    # missing envelope is a damaged formal run and must fail closed.
    if not isinstance(payload.get("_meta"), dict):
        _abort_formal(
            "selected_thresholds.json missing mandatory _meta envelope; "
            "the formal loader refuses any file without run-ID, config "
            "SHA, and implementation-commit provenance."
        )
    meta = payload["_meta"]

    # Required envelope keys (strict order: mandate run_id, config SHA,
    # and implementation commit).
    actual_run_id = meta.get("formal_run_id")
    if actual_run_id is None or str(actual_run_id) != str(expected_run_id):
        _revoke_formal(
            f"_meta.formal_run_id mismatch: expected {expected_run_id!r}, "
            f"got {actual_run_id!r}",
            errors,
        )
    actual_config_sha = meta.get("config_sha256")
    if actual_config_sha is None or str(actual_config_sha) != str(expected_config_sha256):
        _revoke_formal(
            f"_meta.config_sha256 mismatch: expected "
            f"{expected_config_sha256!r}, got {actual_config_sha!r}",
            errors,
        )
    actual_implementation_commit = meta.get("implementation_commit")
    expected_implementation_commit = os.environ.get("M3_FINAL_IMPLEMENTATION_COMMIT", "")
    if expected_implementation_commit:
        if actual_implementation_commit is None or str(actual_implementation_commit) != str(expected_implementation_commit):
            _revoke_formal(
                f"_meta.implementation_commit mismatch: expected "
                f"{expected_implementation_commit!r}, got {actual_implementation_commit!r}",
                errors,
            )

    if errors:
        for e in errors:
            _abort_formal(e)
        _abort_formal("unreachable")  # pragma: no cover

    # Required identity set depends on oracle authorization.
    # When --allow-oracle is granted the run must contain all 32 identities
    # (4 policy families × 2 K × 4 regimes). When --allow-oracle is NOT
    # granted only the 24 non-oracle identities are required and the file
    # must not contain any oracle_threshold_* entries either.
    policy_families = [p for p in FORMAL_POLICY_FAMILIES if p != "oracle_threshold" or allow_oracle]
    expected_keys = {
        _identity_key(p, k, r)
        for p in policy_families
        for k in FORMAL_K_VALUES
        for r in FORMAL_COST_REGIMES
    }
    actual_keys = set(payload.keys()) - {"_meta"}

    missing = expected_keys - actual_keys
    if missing:
        _revoke_formal(
            f"missing identities ({len(missing)}): {sorted(missing)[:5]}...",
            errors,
        )

    extra = actual_keys - expected_keys
    if extra:
        _revoke_formal(
            f"unexpected identities ({len(extra)}): {sorted(extra)[:5]}...",
            errors,
        )

    # If oracle identities are present, require explicit authorization.
    oracle_keys = {k for k in actual_keys if k.startswith("oracle_threshold_")}
    if oracle_keys and not allow_oracle:
        _revoke_formal(
            "oracle_threshold identities present but --allow-oracle authorization "
            "was not granted; refusing to evaluate Oracle",
            errors,
        )
    # If oracle is authorized but a non-oracle identity is missing, fail too.
    if not oracle_keys and allow_oracle:
        # Oracle authorized but no oracle threshold entries: that's a
        # damaged formal run, must fail rather than auto-reclassify.
        _revoke_formal(
            "--allow-oracle granted but no oracle_threshold identities in "
            "selected_thresholds.json (damaged formal run)",
            errors,
        )

    # Validate every threshold value.
    parsed: Dict[str, int] = {}
    for key in sorted(actual_keys):
        entry = payload[key]
        if not isinstance(entry, dict):
            _revoke_formal(f"{key}: entry must be a dict, got {type(entry).__name__}", errors)
            continue
        if "threshold" not in entry:
            _revoke_formal(f"{key}: missing 'threshold' field", errors)
            continue

        raw_value = entry["threshold"]
        policy_family = key.split("_k")[0]

        # Reject null, malformed strings, non-numbers.
        if raw_value is None:
            _revoke_formal(f"{key}: 'threshold' is None (no default substitution)", errors)
            continue
        # Reject booleans masquerading as numbers.
        if isinstance(raw_value, bool):
            _revoke_formal(f"{key}: 'threshold' is bool (not a number)", errors)
            continue
        # Reject malformed strings.
        if isinstance(raw_value, str):
            _revoke_formal(
                f"{key}: 'threshold' is a string {raw_value!r} (no default substitution)",
                errors,
            )
            continue
        # Numeric check (covers floats and ints).
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            _revoke_formal(
                f"{key}: 'threshold' is non-numeric {raw_value!r}", errors
            )
            continue
        # Reject NaN, Inf.
        if np.isnan(value):
            _revoke_formal(f"{key}: 'threshold' is NaN", errors)
            continue
        if np.isinf(value):
            _revoke_formal(f"{key}: 'threshold' is Inf", errors)
            continue
        # Must lie in frozen grid.
        grid = FORMAL_THRESHOLD_GRIDS.get(policy_family, set())
        if int(value) not in grid and value not in grid:
            _revoke_formal(
                f"{key}: threshold {value} not in frozen grid for {policy_family}",
                errors,
            )
            continue
        # Must be integer-valued (no fractional threshold values in formal grid).
        if not float(value).is_integer():
            _revoke_formal(
                f"{key}: threshold {value} is not integer-valued", errors
            )
            continue
        parsed[key] = int(value)

    if errors:
        # Aggregate all collected reasons into a single typed exception so
        # the CLI boundary can emit them as a fail-closed verdict without
        # ever calling ``os._exit``. The loader does not print any
        # test-only ``LOADER_REJECTED`` markers (those concerns live in
        # the test wrappers themselves).
        raise FormalThresholdError(
            "selected_thresholds.json failed "
            f"{len(errors)} checks; refusing to construct environment: "
            + "; ".join(errors)
        )

    # NOTE: The require_selected_sha check (when supplied) is performed
    # BEFORE JSON parsing further up, so we never reach this point with a
    # SHA mismatch waiting to be discovered here.

    return parsed


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load M3 configuration."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "baselines" / "m3_v1.json"
    else:
        config_path = Path(config_path)

    with open(config_path, "r") as f:
        return json.load(f)


def get_scenario_bank_path(
    config: Dict[str, Any],
    split: str,
    smoke_mode: bool = False,
) -> str:
    """
    Get scenario bank path from config or use smoke fallback.

    Args:
        config: M3 configuration dict
        split: Environment split
        smoke_mode: If True, use smoke banks as fallback

    Returns:
        Path to scenario bank

    Raises:
        CaseLoadError: If formal mode bank is missing
    """
    # Check if config has scenario_banks section
    if "scenario_banks" in config:
        bank_path = config["scenario_banks"].get(split)
        if bank_path:
            return bank_path

    # Formal mode: fail clearly if bank not configured
    if not smoke_mode:
        raise CaseLoadError(
            f"Formal mode: scenario bank for split '{split}' not configured in config. "
            f"Add 'scenario_banks.{split}' to config or use --smoke for smoke mode."
        )

    # Smoke mode fallback
    smoke_path = Path(__file__).parent.parent / "data" / "scenario_banks" / f"{split}_smoke.json"
    if smoke_path.exists():
        return str(smoke_path)

    raise CaseLoadError(
        f"Smoke mode: scenario bank not found at {smoke_path}. "
        f"Formal mode also has no configured bank for split '{split}'."
    )


def get_scenario_ids(
    split: str,
    k: int,
    cost_regime_id: str,
    scenario_bank_path: str,
    smoke_mode: bool = False,
) -> List[str]:
    """
    Load scenario bank using case loader and return scenario IDs.

    Uses case_loader to derive K=1 scenarios from K=2 source if needed.

    Args:
        split: Environment split
        k: Maintenance capacity (1 or 2)
        cost_regime_id: Cost regime ID
        scenario_bank_path: Path to source K=2 scenario bank
        smoke_mode: If True, allow fallback to direct loading

    Returns:
        List of derived scenario IDs

    Raises:
        CaseLoadError: If formal mode case derivation fails
    """
    from src.baselines.case_loader import get_scenario_bank_for_case

    # Formal mode: fail clearly on any derivation error
    if not smoke_mode:
        scenario_bank = get_scenario_bank_for_case(
            split=split,
            k=k,
            cost_regime_id=cost_regime_id,
            source_bank_path=scenario_bank_path,
        )
        return [s.scenario_id for s in scenario_bank.scenarios]

    # Smoke mode: allow fallback to direct loading
    try:
        scenario_bank = get_scenario_bank_for_case(
            split=split,
            k=k,
            cost_regime_id=cost_regime_id,
            source_bank_path=scenario_bank_path,
        )
        return [s.scenario_id for s in scenario_bank.scenarios]
    except Exception as e:
        print(f"Warning: Could not load scenarios via case loader: {e}", file=sys.stderr)
        # Fallback to direct load (no K/cost regime derivation)
        scenario_path = Path(scenario_bank_path)
        if not scenario_path.exists():
            return []
        scenario_bank = load_scenario_bank(scenario_path)
        return [s.scenario_id for s in scenario_bank.scenarios]


def run_smoke(
    config: Dict[str, Any],
    env_config: EnvironmentConfig,
    scenario_ids: List[str],
    output_dir: Path,
    scenario_bank_path: str = None,
    policy_filter: str = None,
    allow_oracle: bool = False,
) -> int:
    """
    Run smoke test: one episode per policy family.

    Args:
        policy_filter: If set, run only this policy family
        allow_oracle: If True and policy_filter is oracle_threshold, run oracle

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    print("=" * 60)
    print("M3 SMOKE TEST")
    print("=" * 60)

    log_messages = []
    log_messages.append(f"M3 Smoke Test started at {datetime.utcnow().isoformat()}")
    log_messages.append(f"Split: {env_config.split}")
    log_messages.append(f"K: {env_config.maintenance_capacity}")
    log_messages.append(f"Cost regime: {env_config.cost_regime_id}")
    log_messages.append(f"Scenarios: {scenario_ids[:3]}...")  # First 3

    # Load derived scenario bank for this K/regime (for environment)
    smoke_scenario_bank = None
    if scenario_bank_path:
        try:
            smoke_scenario_bank = get_scenario_bank_for_case(
                split=env_config.split,
                k=env_config.maintenance_capacity,
                cost_regime_id=env_config.cost_regime_id,
                source_bank_path=scenario_bank_path,
            )
        except Exception as e:
            print(f"Warning: Could not load derived scenario bank: {e}", file=sys.stderr)

    # Determine which policies to run
    if policy_filter is not None:
        # User specified a specific policy
        policies_to_run = [policy_filter]
        # Validate oracle request
        if policy_filter == "oracle_threshold" and not allow_oracle:
            print("ERROR: --policy oracle_threshold requires --allow-oracle flag", file=sys.stderr)
            return 1
    else:
        # Run all policies (excluding oracle unless explicitly allowed)
        policies_to_run = [p for p in POLICY_FAMILIES if p != "oracle_threshold"]

    # Create smoke-specific env config with correct scenario bank path
    if scenario_bank_path:
        smoke_env_config = EnvironmentConfig(
            environment_version=env_config.environment_version,
            split=env_config.split,
            fleet_size=env_config.fleet_size,
            maintenance_capacity=env_config.maintenance_capacity,
            delta_cycles=env_config.delta_cycles,
            episode_horizon=env_config.episode_horizon,
            age_scale_cycles=env_config.age_scale_cycles,
            rul_scale=env_config.rul_scale,
            cost_regime_id=env_config.cost_regime_id,
            scenario_bank_path=scenario_bank_path,
            prediction_cache_path=env_config.prediction_cache_path,
            info_mode=env_config.info_mode,
            seed=env_config.seed,
        )
    else:
        smoke_env_config = env_config

    evaluator = PolicyEvaluator(
        env_config=smoke_env_config,
        allow_oracle=allow_oracle and "oracle_threshold" in policies_to_run,
        diagnostic_mode="oracle_threshold" in policies_to_run,
    )

    results: List[EpisodeResult] = []

    for policy_family in policies_to_run:

        print(f"\nRunning {policy_family}...")
        log_messages.append(f"Running {policy_family}...")

        # Create policy
        if policy_family == "corrective_only":
            policy = evaluator.create_policy(policy_family)
            threshold = None
        elif policy_family == "random_feasible":
            policy = evaluator.create_policy(policy_family, policy_seed=42)
            threshold = None
        elif policy_family in ["age_threshold", "predicted_rul_threshold"]:
            threshold = 100  # Use middle threshold for smoke
            policy = evaluator.create_policy(policy_family, threshold=threshold, policy_seed=42)
        elif policy_family == "greedy_predicted_rul":
            threshold = 50
            policy = evaluator.create_policy(
                policy_family, activation_threshold=threshold, policy_seed=42
            )
        else:
            continue

        context = evaluator.create_context(policy_family, policy_seed=42)

        # Run one episode
        if scenario_ids:
            scenario_id = scenario_ids[0]
        else:
            print(f"  Warning: No scenarios available, skipping {policy_family}")
            continue

        env = SelectiveMaintenanceEnv(config=smoke_env_config, scenario_bank=smoke_scenario_bank)

        eval_config = EvaluationConfig(
            env_config=smoke_env_config,
            policy_id=f"{policy_family}_smoke",
            policy_family=policy_family,
            threshold=(
                None
                if policy_family in ("corrective_only", "random_feasible", "greedy_predicted_rul")
                else threshold
            ),
            activation_threshold=(
                threshold if policy_family == "greedy_predicted_rul" else None
            ),
            policy_seed=42,
        )

        run_id = f"smoke_{policy_family}_{env_config.split}_k{env_config.maintenance_capacity}"

        result = evaluator.evaluate_episode(
            env=env,
            policy=policy,
            context=context,
            scenario_id=scenario_id,
            reset_seed=FIXED_RESET_SEEDS[0],
            eval_config=eval_config,
            run_id=run_id,
        )

        results.append(result)

        # Validate result
        if result.completed:
            print(f"  Completed: {result.episode_steps} steps, return={result.episode_return:.2f}")
            log_messages.append(f"  {policy_family}: {result.episode_steps} steps, return={result.episode_return:.2f}")

            # Check for NaN/Inf
            if np.isnan(result.episode_return) or np.isinf(result.episode_return):
                print(f"  ERROR: NaN/Inf in return")
                log_messages.append(f"  ERROR: NaN/Inf in {policy_family} return")
                return 1

            # Check terminated (should be False)
            if result.terminated_count > 0:
                print(f"  ERROR: terminated=True occurred")
                log_messages.append(f"  ERROR: terminated=True in {policy_family}")
                return 1

            # Check truncated (should be True at horizon)
            if not result.truncated:
                print(f"  Warning: truncated=False (early termination?)")
                log_messages.append(f"  Warning: truncated=False in {policy_family}")
        else:
            print(f"  ERROR: {result.error}")
            log_messages.append(f"  ERROR: {policy_family} failed: {result.error}")
            return 1

    # Write smoke results
    if results:
        episode_path = write_episode_results(results, output_dir)
        log_messages.append(f"Episode results written to {episode_path}")

    log_messages.append(f"M3 Smoke Test completed at {datetime.utcnow().isoformat()}")
    log_messages.append("EXIT_CODE=0")

    # Write log
    write_run_log(log_messages, output_dir, 0)


def _sorted_scenario_ids_sha256(scenario_ids: List[str]) -> str:
    """Compute SHA256 over the sorted scenario IDs (newline-separated)."""
    payload = "\n".join(sorted(scenario_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _raw_source_scenario_ids(source_path: str) -> List[str]:
    """Re-open the source scenario-bank JSON and return the raw IDs.

    Independent of the running derivation: re-reads the file verbatim
    and walks ``scenarios[].scenario_id`` so the producer stamps the
    SAME raw IDs the independent recomputation hashes on disk.
    """
    p = Path(source_path)
    if not p.exists():
        raise RuntimeError(
            f"scenario-bank source file not found at {source_path}; cannot read raw scenario IDs"
        )
    with open(p, "r") as f:
        data = json.load(f)
    raw_ids = [s["scenario_id"] for s in data.get("scenarios", []) if s.get("scenario_id")]
    return raw_ids


def _raw_scenario_ids_sha256(source_path: str) -> str:
    """Compute sorted-scenario-IDs SHA256 over the RAW source-bank scenarios.

    Matches the auditor's recomputation semantics in
    ``scripts/independent_recompute_m3.py::verify_scenario_bank_sources``:
    re-open the source JSON, walk ``scenarios[].scenario_id``, sort
    lexicographically, join with newlines, and hash. The auditor does
    this independently; the producer stamps it from disk state.
    """
    return _sorted_scenario_ids_sha256(_raw_source_scenario_ids(source_path))


def _raw_id_from_derived(derived_id: str, k: int, cost_regime_id: str) -> str:
    """Map a derived scenario ID back to its raw source-bank ID.

    The case loader builds derived IDs by appending suffixes:

      * K=2 derivation: ``f"{raw_id}_{cost_regime_id}"``
      * K=1 derivation from K=2 source: ``f"{raw_id}_k1_{cost_regime_id}"``

    Both variants are deterministic and one-to-one. The truncated raw
    ID is whatever remains once the suffix is removed; if no known
    suffix matches, the input ID is returned unchanged so the caller
    can still see a stable form.
    """
    if not isinstance(derived_id, str):
        return derived_id
    suffix_k1 = f"_k1_{cost_regime_id}"
    if k == 1 and derived_id.endswith(suffix_k1):
        return derived_id[: -len(suffix_k1)]
    suffix_regime = f"_{cost_regime_id}"
    if derived_id.endswith(suffix_regime):
        return derived_id[: -len(suffix_regime)]
    return derived_id


def recompute_source_sha256(path: str) -> str:
    """Re-stream the source bank file and recompute its SHA256.

    Independent provenance consistency check: the recorded
    ``source_sha256`` must equal what we get by re-opening the file at
    ``source_path`` and reading its bytes verbatim. Internal
    cross-references between CaseLoadOutput records are not sufficient —
    an external watcher must reopen the file.
    """
    p = Path(path)
    if not p.exists():
        raise RuntimeError(
            f"scenario-bank source file not found at {path}; cannot "
            f"recompute SHA256"
        )
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def build_formal_scenario_bank_identities(
    config: Dict[str, Any],
    splits: List[str],
    k_values: List[int],
    cost_regimes: List[str],
) -> List[Dict[str, Any]]:
    """Build explicit scenario-bank identities for the formal run context.

    Each identity is derived from the actual on-disk source bank file via
    ``case_loader.load_cases``; no fabricated defaults are accepted. The
    returned dicts are the authoritative provenance payloads.

    Required fields per identity (per spec):
        split, K, cost_regime_id, source_path, source_sha256,
        scenario_count, sorted_scenario_ids_sha256

    The ``source_sha256`` is the SHA256 of the actual source bank file on
    disk at the time the identity was built. The caller may independently
    verify this by reopening the file at ``source_path`` and recomputing
    the SHA via :func:`recompute_source_sha256`.
    """
    identities: List[Dict[str, Any]] = []
    for split in splits:
        for k in k_values:
            for cost_regime_id in cost_regimes:
                scenario_bank_path = get_scenario_bank_path(
                    config=config,
                    split=split,
                    smoke_mode=False,
                )
                case_load_result = load_cases(
                    split=split,
                    k=k,
                    cost_regime_id=cost_regime_id,
                    source_bank_path=scenario_bank_path,
                )
                raw_scenario_ids = _raw_source_scenario_ids(
                    str(case_load_result.source_bank_path)
                )
                # Use the live source SHA from disk rather than relying
                # on internal provenance: the recorded value is what an
                # external auditor sees if they reopen the file.
                live_source_sha = recompute_source_sha256(
                    str(case_load_result.source_bank_path)
                )
                identity = {
                    "split": case_load_result.split,
                    "K": k,
                    "cost_regime_id": case_load_result.cost_regime_id,
                    "source_path": case_load_result.source_bank_path,
                    "source_sha256": live_source_sha,
                    "scenario_count": len(raw_scenario_ids),
                    "sorted_scenario_ids_sha256": _sorted_scenario_ids_sha256(
                        raw_scenario_ids
                    ),
                }
                identities.append(identity)
    expected_count = len(splits) * len(k_values) * len(cost_regimes)
    if len(identities) != expected_count:
        raise RuntimeError(
            f"scenario_bank_identities count mismatch: expected "
            f"{expected_count} (splits={len(splits)} x K={len(k_values)} "
            f"x regimes={len(cost_regimes)}) but produced {len(identities)}"
        )
    return identities


def derive_formal_run_id(output_dir: Path) -> str:
    """Derive formal_run_id from output_dir.name; must equal output_dir.name."""
    formal_run_id = output_dir.name
    if not formal_run_id or formal_run_id.startswith("."):
        raise RuntimeError(
            f"Invalid formal_run_id derived from output_dir.name: {formal_run_id!r}"
        )
    return formal_run_id

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)

    return 0


def run_tune(
    config: Dict[str, Any],
    env_config: EnvironmentConfig,
    output_dir: Path,
    allow_oracle: bool = False,
    policy_filter: str = None,
) -> int:
    """
    Run threshold tuning on rl_validation.

    Args:
        allow_oracle: If True, oracle policy is allowed
        policy_filter: If set, tune only this policy family

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    print("=" * 60)
    print("M3 THRESHOLD TUNING")
    print("=" * 60)

    # Barrier: must be rl_validation
    if env_config.split != "rl_validation":
        print(f"ERROR: Threshold tuning must use rl_validation split, got {env_config.split}", file=sys.stderr)
        return 1

    # Validate oracle request
    if policy_filter == "oracle_threshold" and not allow_oracle:
        print("ERROR: --policy oracle_threshold requires --allow-oracle flag", file=sys.stderr)
        return 1

    log_messages = []
    log_messages.append(f"M3 Threshold Tuning started at {datetime.utcnow().isoformat()}")
    log_messages.append(f"Split: {env_config.split}")
    # Scenario count logged per K/regime inside the loop

    # Step 1 of the formal chain: write resolved_config.json before any
    # tuning episode runs.  The SHA256 over canonical JSON bytes of this
    # file is the authoritative config identity for this run; the same
    # SHA is later embedded in selected_thresholds._meta.config_sha256 and
    # in formal_manifest.json.
    try:
        write_resolved_config(config, output_dir)
        resolved_config_sha = compute_canonical_config_sha256(config)
        log_messages.append(
            f"Wrote resolved_config.json (sha256={resolved_config_sha[:12]}...)"
        )
    except Exception as e:
        print(
            f"ERROR: failed to write resolved_config.json: {e}",
            file=sys.stderr,
        )
        return 1

    # Step 0a: Generate formal_run_id once in the orchestration layer.
    # Create_formal_run_context will derive the same value internally and
    # refuse if it does not match output_dir.name, so we record it here
    # and assert it ourselves first.
    try:
        formal_run_id = derive_formal_run_id(output_dir)
        if formal_run_id != output_dir.name:
            print(
                f"ERROR: formal_run_id {formal_run_id!r} does not match output_dir.name {output_dir.name!r}",
                file=sys.stderr,
            )
            return 1
    except Exception as e:
        print(
            f"ERROR: failed to derive formal_run_id: {e}",
            file=sys.stderr,
        )
        return 1

    # Step 0b-pre: Build scenario-bank identities BEFORE formal context.
    # These are derived from actual on-disk banks via case loader; no
    # fabricated defaults are accepted.
    #
    # The complete split × K × cost-regime identity set used by the
    # formal plan, including BOTH formal evaluation splits
    # (predictor_train + rl_validation) when applicable. Tuning always
    # runs on rl_validation but the identity set MUST include every
    # bank used by the unseen evaluation splits too, so an auditor
    # can replay the entire recorded identity surface.
    try:
        tuning_splits = [env_config.split]
        evaluation_splits = list(config.get("evaluation_splits", []) or [])
        # Defense-in-depth: rl_test is forbidden by the project barrier
        # and must never appear in the identity set.
        if "rl_test" in evaluation_splits:
            raise RuntimeError(
                "rl_test must never appear in evaluation_splits for a "
                "formal identity set; check config."
            )
        all_splits = sorted(set(tuning_splits) | set(evaluation_splits))
        all_k_values = [1, 2]
        all_cost_regimes = list(config["cost_regimes"])
        scenario_bank_identities: List[Dict[str, Any]] = build_formal_scenario_bank_identities(
            config,
            splits=all_splits,
            k_values=all_k_values,
            cost_regimes=all_cost_regimes,
        )
        log_messages.append(
            f"Built {len(scenario_bank_identities)} scenario-bank identities "
            f"from on-disk source banks (splits={all_splits})"
        )
    except Exception as e:
        print(
            f"ERROR: failed to build scenario-bank identities: {e}",
            file=sys.stderr,
        )
        return 1

    # Step 0b: Create immutable formal run context BEFORE tuning begins.
    # This seals: implementation commit (from git), config SHA, run ID,
    # scenario banks, reset seeds. The context will be updated exactly
    # ONCE after selected_thresholds.json is finalised, with the
    # selected-thresholds SHA. After that it is sealed against further
    # mutation.
    #
    # formal_run_id is generated ONCE here (in orchestration) and
    # threaded into the context creator so the same value is used by
    # both the orchestration step and the persistence step. The
    # creator refuses a divergent caller-supplied value.
    try:
        formal_context = create_formal_run_context(
            output_dir=output_dir,
            resolved_config=config,
            resolved_config_path=output_dir / "resolved_config.json",
            selected_thresholds_path=output_dir / "selected_thresholds.json",
            allow_oracle=allow_oracle,
            scenario_bank_identities=scenario_bank_identities,
            reset_seeds=FIXED_RESET_SEEDS,
            mode="formal_closeout" if allow_oracle else "diagnostic_non_oracle",
            formal_run_id=formal_run_id,
        )
        log_messages.append(
            f"Created formal_run_context.json (run_id={formal_context.formal_run_id})"
        )
    except Exception as e:
        print(
            f"ERROR: failed to create formal_run_context.json: {e}",
            file=sys.stderr,
        )
        return 1

    # Determine which policies to tune
    if policy_filter is not None:
        if policy_filter not in THRESHOLD_POLICIES:
            print(f"ERROR: Policy {policy_filter} is not a threshold policy", file=sys.stderr)
            print(f"Threshold policies: {list(THRESHOLD_POLICIES.keys())}", file=sys.stderr)
            return 1
        policies_to_tune = [policy_filter]
    else:
        policies_to_tune = list(THRESHOLD_POLICIES.keys())

    all_candidates = []
    # Episode-level rows captured per-(policy, threshold, scenario, seed);
    # written to canonical ``threshold_search_results.parquet`` so the
    # formal-closeout auditor can reconstruct the 9000 tuning identities.
    all_episode_rows = []
    selected_thresholds = {}
    scenario_bank_provenance = []  # Collect provenance for each K/regime
    seen_provenance_keys = set()  # Avoid duplicates

    for k in [1, 2]:
        for cost_regime_id in config["cost_regimes"]:
            print(f"\nTuning for K={k}, cost_regime={cost_regime_id}")
            log_messages.append(f"Tuning K={k}, {cost_regime_id}")

            # Load correct scenario bank and scenario IDs for this K/regime using case loader
            try:
                # Get scenario bank path from config (formal mode)
                scenario_bank_path = get_scenario_bank_path(
                    config=config,
                    split=env_config.split,
                    smoke_mode=False,  # Formal mode: fail if not configured
                )

                # Load with full provenance using load_cases
                case_load_result = load_cases(
                    split=env_config.split,
                    k=k,
                    cost_regime_id=cost_regime_id,
                    source_bank_path=scenario_bank_path,
                )

                # Get the derived scenario bank for this K/regime (single derivation)
                k_regime_scenario_bank = get_scenario_bank_for_case(
                    split=env_config.split,
                    k=k,
                    cost_regime_id=cost_regime_id,
                    source_bank_path=scenario_bank_path,
                )
                # Extract scenario IDs from the derived bank (no double-derivation)
                k_regime_scenario_ids = [s.scenario_id for s in k_regime_scenario_bank.scenarios]
                print(f"    Loaded {len(k_regime_scenario_ids)} scenarios for K={k}, {cost_regime_id}: {k_regime_scenario_ids[:3]}...")

                # Record provenance (avoid duplicates for same K/regime).
                # ``split`` / ``K`` / ``cost_regime_id`` are the canonical
                # fields used by the audit; ``sorted_scenario_ids_sha256``
                # is computed over the RAW source-bank IDs (not the
                # derived IDs) so it matches the auditor's independent
                # recomputation.
                prov_key = (env_config.split, k, cost_regime_id)
                if prov_key not in seen_provenance_keys:
                    seen_provenance_keys.add(prov_key)
                    raw_ids_sha = _raw_scenario_ids_sha256(
                        str(case_load_result.source_bank_path)
                    )
                    scenario_bank_provenance.append({
                        # Canonical contract fields (must be present and
                        # non-empty for every record on the formal path).
                        "split": env_config.split,
                        "K": int(k),
                        "cost_regime_id": cost_regime_id,
                        # Source provenance.
                        "source_path": case_load_result.source_bank_path,
                        "source_file_size": case_load_result.source_file_size,
                        "source_sha256": case_load_result.bank_sha256,
                        "scenario_count": int(case_load_result.bank_scenario_count),
                        "sorted_scenario_ids_sha256": raw_ids_sha,
                        # Derived-bank provenance (kept for diagnostic
                        # / regression visibility; not used by the
                        # identity hash).
                        "logical_bank_id": case_load_result.logical_bank_id,
                        "derived_k": case_load_result.k,
                        "derived_cost_regime_id": case_load_result.cost_regime_id,
                        "derived_scenario_count": case_load_result.derived_scenario_count,
                        "derived_scenario_ids": list(case_load_result.scenario_ids),
                        "derived_bank_sha256": case_load_result.derived_bank_sha256,
                    })

            except Exception as e:
                print(f"ERROR: Failed to load scenarios for K={k}, {cost_regime_id}: {e}", file=sys.stderr)
                return 1

            # Create env config for this K/regime
            k_config = EnvironmentConfig(
                environment_version=env_config.environment_version,
                split=env_config.split,
                fleet_size=env_config.fleet_size,
                maintenance_capacity=k,
                delta_cycles=env_config.delta_cycles,
                episode_horizon=env_config.episode_horizon,
                age_scale_cycles=env_config.age_scale_cycles,
                rul_scale=env_config.rul_scale,
                cost_regime_id=cost_regime_id,
                scenario_bank_path=env_config.scenario_bank_path,
                prediction_cache_path=env_config.prediction_cache_path,
                info_mode=env_config.info_mode,
                seed=env_config.seed,
            )

            for policy_family in policies_to_tune:
                if policy_family == "oracle_threshold" and not allow_oracle:
                    continue

                print(f"  Tuning {policy_family}...")

                # ``tune_threshold`` resolves scenarios through the
                # derived scenario bank (the env requires its scenario
                # IDs to come from the bank).  The env simulation
                # therefore uses derived IDs internally, but the
                # produced parquet's scenario_id column is rewritten
                # by post-processing the captured episode_rows so it
                # carries the RAW source-bank IDs the auditor hashes.
                selected, candidates = tune_threshold(
                    policy_family=policy_family,
                    k_capacity=k,
                    cost_regime_id=cost_regime_id,
                    env_config=k_config,
                    scenario_ids=k_regime_scenario_ids,  # derived (env-friendly)
                    scenario_bank=k_regime_scenario_bank,
                    reset_seeds=FIXED_RESET_SEEDS,
                    allow_oracle=allow_oracle and policy_family == "oracle_threshold",
                )

                # Rewrite episode rows' scenario_id from derived → raw.
                # Mapping is one-to-one by suffix: derived either equals
                # the raw ID (K=2) or is ``raw_k1_<regime>`` (K=1 derived
                # from a K=2 source).
                live_rows = getattr(selected, "episode_rows", None)
                if live_rows:
                    for row in live_rows:
                        row["scenario_id"] = _raw_id_from_derived(
                            row["scenario_id"], k, cost_regime_id
                        )

                all_candidates.extend(candidates)
                all_episode_rows.extend(live_rows or [])
                key = f"{policy_family}_k{k}_{cost_regime_id}"
                selected_thresholds[key] = selected

                print(f"    Best threshold: {selected.threshold}")
                print(f"    Mean cost: {selected.mean_total_cost:.2f}")
                log_messages.append(f"  {policy_family}: threshold={selected.threshold}, cost={selected.mean_total_cost:.2f}")

    # Write results
    if all_episode_rows:
        # Episode-level evidence (9000 rows for the formal oracle run)
        # goes into the canonical ``threshold_search_results.parquet``;
        # the auditor's candidate-identity reconstruction dedups the
        # 6-tuple to the 360-row candidate set and the 6-tuple episode
        # reconstruction builds the 9000-row tuning set from the same
        # parquet.
        write_threshold_search_results(all_episode_rows, output_dir)
        log_messages.append(f"Wrote {len(all_episode_rows)} tuning episode rows")
    if all_candidates:
        # Candidate summary is emitted as a CSV (360 rows) for human
        # inspection and not used by the auditor.
        write_threshold_search_summary(all_candidates, output_dir)
        log_messages.append(f"Wrote {len(all_candidates)} threshold candidates (summary)")

    if selected_thresholds:
        # Formal path: stamp _meta envelope so the formal loader can
        # verify run ID, config SHA, and implementation commit before
        # evaluation begins.  The config SHA is computed from the
        # canonical-JSON bytes of the resolved config we just wrote, not
        # from a directory hash or env-var fallback.
        if allow_oracle:
            meta_run_id = os.environ.get("M3_FORMAL_EXPECTED_RUN_ID") or output_dir.name
            meta_config_sha = resolved_config_sha
            # The context already obtained the authoritative full commit from
            # git after enforcing a clean worktree. Do not depend on a
            # caller-supplied environment variable for formal provenance.
            meta_commit = formal_context.implementation_commit
            write_selected_thresholds_with_meta(
                selected_thresholds,
                output_dir,
                formal_run_id=meta_run_id,
                config_sha256=meta_config_sha,
                implementation_commit=meta_commit,
            )
        else:
            # Non-formal (legacy/default) path keeps the old writer.
            write_selected_thresholds(selected_thresholds, output_dir)
        log_messages.append(f"Wrote {len(selected_thresholds)} selected thresholds")

        # Step 0c: Seal formal run context with selected_thresholds SHA.
        # This must happen exactly once after selected_thresholds.json is finalised.
        try:
            selected_path = output_dir / "selected_thresholds.json"
            seal_formal_run_context(output_dir, selected_thresholds_path=selected_path)
            log_messages.append(f"Sealed formal_run_context.json with selected_thresholds SHA")
        except Exception as e:
            print(
                f"ERROR: failed to seal formal_run_context.json: {e}",
                file=sys.stderr,
            )
            return 1

    # Write scenario-bank provenance
    if scenario_bank_provenance:
        write_scenario_bank_provenance(scenario_bank_provenance, output_dir)
        log_messages.append(f"Wrote {len(scenario_bank_provenance)} scenario-bank provenance records")

    # Write run provenance
    run_provenance = {
        "run_type": "threshold_tuning",
        "split": env_config.split,
        "k_values": [1, 2],
        "cost_regimes": config["cost_regimes"],
        "reset_seeds": FIXED_RESET_SEEDS,
        "policies_tuned": list(selected_thresholds.keys()),
        "total_candidates": len(all_candidates),
        "total_selected": len(selected_thresholds),
        "total_episodes": len(all_episode_rows),
        "completed_at": datetime.utcnow().isoformat(),
    }
    write_run_provenance(run_provenance, output_dir)
    log_messages.append("Wrote run_provenance.json")

    log_messages.append(f"M3 Threshold Tuning completed at {datetime.utcnow().isoformat()}")
    log_messages.append("EXIT_CODE=0")

    write_run_log(log_messages, output_dir, 0)

    print("\n" + "=" * 60)
    print("THRESHOLD TUNING COMPLETE")
    print("=" * 60)

    return 0


def run_evaluate(
    config: Dict[str, Any],
    env_config: EnvironmentConfig,
    selected_thresholds_path: Optional[Path],
    output_dir: Path,
    allow_oracle: bool = False,
    policy_filter: str = None,
    mode: str = "diagnostic_non_oracle",
) -> int:
    """
    Run full baseline evaluation.

    Args:
        allow_oracle: If True, oracle policy is allowed
        policy_filter: If set, evaluate only this policy family
        mode: Explicit mode. One of:
            - "formal_closeout": production-ready closeout; demands sealed
              formal context, Oracle authorization, 360/9000/32/2400 counts.
              Will NOT fall back to permissive interpretations of an
              unsealed or missing context, PENDING evidence, caller-supplied
              run IDs / commits, or absent selected files.
            - "diagnostic_non_oracle": explicit non-Oracle diagnostic mode.
            - "diagnostic_legacy": explicit diagnostic mode that may tolerate
              legacy context-shape fixtures (legacy `_sealed` alias).
              NEVER reachable from formal --all.

    Returns:
        Exit code (0 for success, 1 for failure, 2 for fail-closed)

    Raises:
        RuntimeError: if a non-explicit mode is implied from disk state.
    """
    print("=" * 60)
    print("M3 BASELINE EVALUATION")
    print("=" * 60)

    log_messages = []
    log_messages.append(f"M3 Baseline Evaluation started at {datetime.utcnow().isoformat()}")

    # Mode MUST be explicit; we no longer infer mode from artifacts on disk.
    if mode is None:
        print(
            "ERROR: --mode is required (formal_closeout, diagnostic_non_oracle, "
            "or diagnostic_legacy); mode will not be inferred from "
            "artifacts-on-disk.",
            file=sys.stderr,
        )
        return 2
    if mode not in ("formal_closeout", "diagnostic_non_oracle", "diagnostic_legacy"):
        print(
            f"ERROR: unknown mode {mode!r}; expected formal_closeout, "
            "diagnostic_non_oracle, or diagnostic_legacy.",
            file=sys.stderr,
        )
        return 2

    # Validate oracle request
    if policy_filter == "oracle_threshold" and not allow_oracle:
        print("ERROR: --policy oracle_threshold requires --allow-oracle flag", file=sys.stderr)
        return 1

    # Step 0d: Run the strict formal threshold loader.
    #
    # formal_closeout: a SEALED explicit context with recorded
    #   run_id, config SHA, and selected SHA is REQUIRED. Caller-supplied
    #   run IDs and config SHAs via environment variables are FORBIDDEN in
    #   formal_closeout mode; the sealed context is the only authority.
    #
    # diagnostic_non_oracle: strict parser; the legacy ``_sealed`` alias
    #   is rejected like in formal_closeout.
    #
    # diagnostic_legacy: the only mode that allows the legacy ``_sealed``
    #   alias on disk. NEVER reachable from --all.
    context_path = output_dir / "formal_run_context.json"
    formal_context_raw = None
    allow_legacy_sealed_alias = (mode == "diagnostic_legacy")
    if context_path.exists():
        try:
            formal_context_raw = load_formal_run_context(
                output_dir,
                allow_legacy_sealed_alias=allow_legacy_sealed_alias,
            )
        except Exception as e:
            print(
                f"ERROR (formal run context): {e}",
                file=sys.stderr,
            )
            return 2
    elif mode == "formal_closeout":
        # In formal_closeout mode a sealed context is REQUIRED; refuse to
        # fall back to a context-less evaluation. Mode must be explicit.
        print(
            f"ERROR (formal run context): formal_run_context.json is "
            f"missing at {context_path}; formal_closeout requires a "
            f"sealed formal run context created before evaluation.",
            file=sys.stderr,
        )
        return 2

    # In formal_closeout, a SEALED context is mandatory. An unsealed
    # context is treated as "context present but not yet sealed" — this
    # is a damaged formal run and must reject.
    if mode == "formal_closeout" and (
        formal_context_raw is None or not formal_context_raw.sealed
    ):
        print(
            "ERROR (formal run context): formal_closeout requires a sealed "
            "formal_run_context.json; the context is not sealed or "
            "selected_thresholds_sha256 is missing.",
            file=sys.stderr,
        )
        return 2

    # expected_run_id comes ONLY from output_dir.name. formal_closeout
    # forbids caller-supplied run IDs via env-var.
    expected_run_id = output_dir.name

    # expected_config_sha256 comes from the sealed context (formal_closeout),
    # OR from resolved_config.json on disk (diagnostic_*). Caller-supplied
    # env-var overrides are FORBIDDEN in formal_closeout.
    env_cfg_override = os.environ.get("M3_FORMAL_EXPECTED_CONFIG_SHA256", "")
    if env_cfg_override and mode == "formal_closeout":
        print(
            "ERROR (formal threshold loader, fail-closed): "
            "M3_FORMAL_EXPECTED_CONFIG_SHA256 env-var override is forbidden "
            "in formal_closeout mode; the sealed formal_run_context.json is "
            "the only authority.",
            file=sys.stderr,
        )
        return 2
    expected_config_sha256 = (
        (
            formal_context_raw.resolved_config_sha256
            if (formal_context_raw and formal_context_raw.resolved_config_sha256)
            else None
        )
        or read_resolved_config_sha256(output_dir)
        or ""
    )
    if not expected_config_sha256:
        print(
            "ERROR (formal threshold loader, fail-closed): "
            "resolved_config.json is missing from the output dir and the "
            "sealed context has no config SHA; refusing to evaluate a run "
            "whose canonical config SHA is unknown.",
            file=sys.stderr,
        )
        return 2

    expected_selected_sha: Optional[str] = (
        formal_context_raw.selected_thresholds_sha256
        if formal_context_raw else None
    )

    # Resolve selected_thresholds path; default to the file the context
    # points to (or the standard output-dir path when no context).
    if selected_thresholds_path is None:
        if formal_context_raw and formal_context_raw.selected_thresholds_path:
            selected_thresholds_path = Path(formal_context_raw.selected_thresholds_path)
        else:
            selected_thresholds_path = output_dir / "selected_thresholds.json"

    # Load the strict-formal thresholds. Rejection here always happens
    # before any environment construction. The loader raises a typed
    # ``FormalThresholdError`` instead of calling ``os._exit`` so the CLI
    # boundary can translate it into a 2-exit without bypassing Python's
    # exception machinery.
    try:
        thresholds_by_key: Dict[str, int] = load_formal_selected_thresholds(
            selected_thresholds_path,
            expected_run_id=expected_run_id,
            expected_config_sha256=expected_config_sha256,
            allow_oracle=allow_oracle,
            require_selected_sha=expected_selected_sha,
        )
    except FormalThresholdError as e:
        print(
            "ERROR (formal threshold loader, fail-closed): "
            f"{e}",
            file=sys.stderr,
        )
        return 2

    # Step 0e: Run the sealed-context + commit validation only when a
    # sealed context exists. A missing context is acceptable for
    # diagnostic / staging runs that have staged the thresholds payload
    # directly; the strict validation gate is reserved for closed-formal
    # productions.
    if formal_context_raw is not None:
        try:
            context_errors = validate_formal_run_context(output_dir)
            if context_errors:
                for err in context_errors:
                    print(f"ERROR (formal run context): {err}", file=sys.stderr)
                return 2
            formal_context = formal_context_raw
            log_messages.append(
                f"Validated formal_run_context.json (run_id={formal_context.formal_run_id})"
            )
        except Exception as e:
            print(
                f"ERROR (formal run context): {e}",
                file=sys.stderr,
            )
            return 2

    # Determine which policies to evaluate
    if policy_filter is not None:
        policies_to_eval = [policy_filter]
    else:
        policies_to_eval = [p for p in POLICY_FAMILIES]

    all_results: List[EpisodeResult] = []
    scenario_bank_provenance = []  # Collect provenance for each split/K/regime
    seen_provenance_keys = set()

    for split in config["evaluation_splits"]:
        check_rl_test_barrier(split)

        for k in config["k_values"]:
            for cost_regime_id in config["cost_regimes"]:
                print(f"\nEvaluating split={split}, K={k}, regime={cost_regime_id}")

                # Load correct scenario bank and scenario IDs for this K/regime
                try:
                    # Get scenario bank path from config (formal mode)
                    scenario_bank_path = get_scenario_bank_path(
                        config=config,
                        split=split,
                        smoke_mode=False,  # Formal mode: fail if not configured
                    )

                    # Load with full provenance using load_cases
                    case_load_result = load_cases(
                        split=split,
                        k=k,
                        cost_regime_id=cost_regime_id,
                        source_bank_path=scenario_bank_path,
                    )

                    # Get the derived scenario bank for this K/regime (single derivation)
                    k_regime_scenario_bank = get_scenario_bank_for_case(
                        split=split,
                        k=k,
                        cost_regime_id=cost_regime_id,
                        source_bank_path=scenario_bank_path,
                    )
                    # Extract scenario IDs from the derived bank
                    k_regime_scenario_ids = [s.scenario_id for s in k_regime_scenario_bank.scenarios]
                    # Raw source-bank IDs are used for raw IDs that the
                    # formal parquet records in scenario_id (matching the
                    # auditor's hash-of-source semantics).
                    k_regime_source_scenario_ids = _raw_source_scenario_ids(
                        str(case_load_result.source_bank_path)
                    )

                    # Record provenance (avoid duplicates). Same canonical-
                    # field shape as the tuning provenance record so the
                    # auditor sees one consistent format across runs.
                    prov_key = (split, k, cost_regime_id)
                    if prov_key not in seen_provenance_keys:
                        seen_provenance_keys.add(prov_key)
                        raw_ids_sha = _raw_scenario_ids_sha256(
                            str(case_load_result.source_bank_path)
                        )
                        scenario_bank_provenance.append({
                            "split": split,
                            "K": int(k),
                            "cost_regime_id": cost_regime_id,
                            "source_path": case_load_result.source_bank_path,
                            "source_file_size": case_load_result.source_file_size,
                            "source_sha256": case_load_result.bank_sha256,
                            "scenario_count": int(case_load_result.bank_scenario_count),
                            "sorted_scenario_ids_sha256": raw_ids_sha,
                            "logical_bank_id": case_load_result.logical_bank_id,
                            "derived_k": case_load_result.k,
                            "derived_cost_regime_id": case_load_result.cost_regime_id,
                            "derived_scenario_count": case_load_result.derived_scenario_count,
                            "derived_scenario_ids": list(case_load_result.scenario_ids),
                            "derived_bank_sha256": case_load_result.derived_bank_sha256,
                        })
                except Exception as e:
                    print(f"ERROR: Failed to load scenarios for split={split}, K={k}, {cost_regime_id}: {e}", file=sys.stderr)
                    return 1

                # Create env config and evaluator with correct K for this loop iteration
                k_config = EnvironmentConfig(
                    environment_version=env_config.environment_version,
                    split=split,
                    fleet_size=env_config.fleet_size,
                    maintenance_capacity=k,
                    delta_cycles=env_config.delta_cycles,
                    episode_horizon=env_config.episode_horizon,
                    age_scale_cycles=env_config.age_scale_cycles,
                    rul_scale=env_config.rul_scale,
                    cost_regime_id=cost_regime_id,
                    scenario_bank_path=env_config.scenario_bank_path,
                    prediction_cache_path=env_config.prediction_cache_path,
                    info_mode=env_config.info_mode,
                    seed=env_config.seed,
                )

                evaluator = PolicyEvaluator(
                    env_config=k_config,
                    allow_oracle=allow_oracle and "oracle_threshold" in policies_to_eval,
                    diagnostic_mode="oracle_threshold" in policies_to_eval,
                )

                # Evaluate each policy
                for policy_family in policies_to_eval:
                    if policy_family == "oracle_threshold" and not allow_oracle:
                        continue

                    # Get threshold for this policy from the strict
                    # formal loader.  corrective_only and random_feasible
                    # are not threshold-based and do not receive a value.
                    threshold_key = f"{policy_family}_k{k}_{cost_regime_id}"
                    threshold = thresholds_by_key.get(threshold_key)
                    if threshold is None and policy_family not in (
                        "corrective_only",
                        "random_feasible",
                    ):
                        print(
                            f"ERROR: formal loader returned no threshold for "
                            f"{threshold_key}; refusing to construct environment",
                            file=sys.stderr,
                        )
                        return 1

                    print(f"  {policy_family} (threshold={threshold})...")

                    # Create policy — strict path: no `or 50`, `or 100`,
                    # never substitutes a default.
                    if policy_family == "corrective_only":
                        policy = evaluator.create_policy(policy_family)
                    elif policy_family == "random_feasible":
                        policy = evaluator.create_policy(
                            policy_family, policy_seed=42
                        )
                    elif policy_family == "greedy_predicted_rul":
                        if threshold is None:
                            print(
                                f"ERROR: greedy_predicted_rul requires "
                                f"activation_threshold for {threshold_key}",
                                file=sys.stderr,
                            )
                            return 1
                        policy = evaluator.create_policy(
                            policy_family,
                            activation_threshold=threshold,
                            policy_seed=42,
                        )
                    else:
                        if threshold is None:
                            print(
                                f"ERROR: {policy_family} requires threshold for "
                                f"{threshold_key}",
                                file=sys.stderr,
                            )
                            return 1
                        policy = evaluator.create_policy(
                            policy_family,
                            threshold=threshold,
                            policy_seed=42,
                        )

                    context = evaluator.create_context(policy_family, policy_seed=42)

                    # Run episodes for all scenarios and seeds.
                    # Iterate over RAW source-bank scenario IDs so the
                    # parquet's scenario_id matches the IDs the auditor
                    # hashes against the source JSON.  The env still
                    # consumes the derived scenario bank; we resolve each
                    # raw ID to its derived ID inside ``k_regime_scenario_bank``
                    # by the deterministic suffix-strip rule.
                    raw_source_ids = _raw_source_scenario_ids(
                        str(case_load_result.source_bank_path)
                    )
                    # Build a derived -> raw lookup so we can find each
                    # derived ID's source ID even when the source-bank
                    # hash uses a different ordering.
                    raw_lookup: Dict[str, str] = {}
                    for sid in k_regime_scenario_bank.scenarios:
                        raw_id = _raw_id_from_derived(sid.scenario_id, k, cost_regime_id)
                        raw_lookup[sid.scenario_id] = raw_id

                    for raw_scenario_id in raw_source_ids:
                        # Find derived ID inside the bank (one-to-one).
                        derived_scenario_id = next(
                            (
                                d for d, r in raw_lookup.items() if r == raw_scenario_id
                            ),
                            None,
                        )
                        if derived_scenario_id is None:
                            print(
                                f"ERROR: no derived scenario for source ID "
                                f"{raw_scenario_id} in split={split}, K={k}, "
                                f"regime={cost_regime_id}",
                                file=sys.stderr,
                            )
                            return 1

                        for reset_seed in FIXED_RESET_SEEDS:
                            # Create environment with diagnostic mode for oracle
                            is_oracle = policy_family == "oracle_threshold"
                            env = SelectiveMaintenanceEnv(
                                config=k_config,
                                info_mode="diagnostic" if is_oracle else "normal",
                                scenario_bank=k_regime_scenario_bank,
                            )

                            eval_config = EvaluationConfig(
                                env_config=k_config,
                                policy_id=f"{policy_family}_k{k}_{cost_regime_id}",
                                policy_family=policy_family,
                                threshold=(
                                    None
                                    if policy_family in (
                                        "corrective_only",
                                        "random_feasible",
                                        "greedy_predicted_rul",
                                    )
                                    else threshold
                                ),
                                activation_threshold=(
                                    threshold
                                    if policy_family == "greedy_predicted_rul"
                                    else None
                                ),
                                policy_seed=42,
                            )

                            run_id = f"eval_{policy_family}_k{k}_{cost_regime_id}_{raw_scenario_id}_{reset_seed}"

                            # Pass the DERIVED scenario ID to evaluate_episode
                            # (env.reset() resolves the scenario by its bank
                            # key); record the RAW source-bank ID via the
                            # new ``source_scenario_id`` parameter so the
                            # writer emits it in the parquet.
                            result = evaluator.evaluate_episode(
                                env=env,
                                policy=policy,
                                context=context,
                                scenario_id=derived_scenario_id,
                                reset_seed=reset_seed,
                                eval_config=eval_config,
                                run_id=run_id,
                                source_scenario_id=raw_scenario_id,
                            )

                            all_results.append(result)

    # Write results
    if all_results:
        write_episode_results(all_results, output_dir)
        log_messages.append(f"Wrote {len(all_results)} episode results")

        # Compute and write summary
        summary_df = summarize_results(all_results)
        write_summary_by_policy(summary_df, output_dir)
        log_messages.append(f"Wrote summary for {len(summary_df)} groups")

    # Write scenario-bank provenance
    if scenario_bank_provenance:
        write_scenario_bank_provenance(scenario_bank_provenance, output_dir)
        log_messages.append(f"Wrote {len(scenario_bank_provenance)} scenario-bank provenance records")

    # Write run provenance
    run_provenance = {
        "run_type": "baseline_evaluation",
        "evaluation_splits": config["evaluation_splits"],
        "k_values": config["k_values"],
        "cost_regimes": config["cost_regimes"],
        "reset_seeds": FIXED_RESET_SEEDS,
        "policies_evaluated": policies_to_eval,
        "total_episodes": len(all_results),
        "scenario_banks_recorded": len(scenario_bank_provenance),
        "completed_at": datetime.utcnow().isoformat(),
    }
    write_run_provenance(run_provenance, output_dir)
    log_messages.append("Wrote run_provenance.json")

    log_messages.append(f"M3 Baseline Evaluation completed at {datetime.utcnow().isoformat()}")
    log_messages.append("EXIT_CODE=0")

    write_run_log(log_messages, output_dir, 0)

    print("\n" + "=" * 60)
    print("BASELINE EVALUATION COMPLETE")
    print("=" * 60)

    return 0


def run_validate_artifacts(
    output_dir: Path,
    mode: str,
) -> int:
    """Validate all artifacts under an explicitly declared mode.

    Args:
        output_dir: directory containing the artifacts
        mode: One of the explicit modes ("formal_closeout",
              "diagnostic_non_oracle", "diagnostic_legacy")
    """
    print("=" * 60)
    print("ARTIFACT VALIDATION")
    print(f"Mode (explicit): {mode}")
    print("=" * 60)

    if mode not in (
        "formal_closeout",
        "diagnostic_non_oracle",
        "diagnostic_legacy",
    ):
        print(
            f"ERROR: --mode is required (formal_closeout, diagnostic_non_oracle, "
            f"or diagnostic_legacy); got {mode!r}",
            file=sys.stderr,
        )
        return 2

    # Delegate to the real validator with an explicit mode.  This calls
    # the strict validator — the local validate_artifacts() helper only
    # does light schema checks and is intentionally NOT the formal path.
    import subprocess
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "validate_m3_artifacts.py"),
        str(output_dir),
        "--mode", mode,
    ]
    completed = subprocess.run(cmd)
    return completed.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Milestone 3 Rule Baselines CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_m3_baselines.py --help
    python scripts/run_m3_baselines.py --smoke
    python scripts/run_m3_baselines.py --tune --output-dir results/m3_test
    python scripts/run_m3_baselines.py --evaluate --output-dir results/m3_test
    python scripts/run_m3_baselines.py --all
        """,
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke test (one episode per policy)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run threshold tuning on rl_validation",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run full baseline evaluation",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run tune + evaluate",
    )
    parser.add_argument(
        "--validate-artifacts",
        action="store_true",
        help="Validate output artifacts",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="rl_validation",
        help="Environment split (default: rl_validation)",
    )
    parser.add_argument(
        "--k-capacity",
        type=int,
        default=2,
        help="Maintenance capacity K (default: 2)",
    )
    parser.add_argument(
        "--cost-regime",
        type=str,
        default="failure-light-no-waste",
        help="Cost regime (default: failure-light-no-waste)",
    )
    parser.add_argument(
        "--policy",
        type=str,
        choices=POLICY_FAMILIES,
        help="Specific policy family (default: all)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="6521,6522,6523,6524,6525",
        help="Comma-separated reset seeds",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config JSON (default: configs/baselines/m3_v1.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/milestone3/m3_baselines_<timestamp>)",
    )
    parser.add_argument(
        "--allow-oracle",
        action="store_true",
        help="Allow oracle policy (default: False)",
    )
    parser.add_argument(
        "--mode",
        choices=("formal_closeout", "diagnostic_non_oracle", "diagnostic_legacy"),
        default=None,
        help=(
            "Explicit mode for the production evaluation path. Required "
            "because we never infer mode from artifacts on disk. "
            "  - formal_closeout: production-ready closeout; demands sealed "
            "    formal context, Oracle authorization, 360/9000/32/2400 counts. "
            "  - diagnostic_non_oracle: explicit non-Oracle diagnostic mode "
            "    with 272/6800/24/2000 counts. "
            "  - diagnostic_legacy: explicit diagnostic mode that may tolerate "
            "    legacy context-shape fixtures (e.g. _sealed alias). "
            "    **NEVER reachable from --all.**"
        ),
    )

    args = parser.parse_args()

    # Default to --all if no command specified
    if not any([args.smoke, args.tune, args.evaluate, args.all, args.validate_artifacts]):
        parser.print_help()
        return 0

    # Load config
    config = load_config(args.config)

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent.parent / "results" / "milestone3" / f"m3_baselines_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Barrier: reject rl_test before any data loading
    if args.split == "rl_test":
        check_rl_test_barrier(args.split)

    # Create environment config
    env_config = get_default_config(
        split=args.split,
        maintenance_capacity=args.k_capacity,
        cost_regime_id=args.cost_regime,
        seed=int(args.seeds.split(",")[0]),
    )

    # Get scenario IDs
    scenario_ids = []
    try:
        # Use config-driven scenario bank path
        scenario_bank_path = get_scenario_bank_path(
            config=config,
            split=args.split,
            smoke_mode=True,  # Smoke mode: use smoke banks as fallback
        )
        scenario_ids = get_scenario_ids(
            split=args.split,
            k=args.k_capacity,
            cost_regime_id=args.cost_regime,
            scenario_bank_path=scenario_bank_path,
            smoke_mode=True,  # Allow fallback in smoke mode
        )
    except Exception as e:
        print(f"Warning: Could not load scenarios: {e}")

    # Run commands
    exit_code = 0

    if args.smoke:
        # Use config-driven scenario bank path for smoke testing
        smoke_path_str = get_scenario_bank_path(
            config=config,
            split=args.split,
            smoke_mode=True,
        )
        code = run_smoke(
            config, env_config, scenario_ids, output_dir,
            scenario_bank_path=smoke_path_str,
            policy_filter=args.policy,
            allow_oracle=args.allow_oracle,
        )
        if code != 0:
            exit_code = code

    if args.tune:
        code = run_tune(
            config, env_config, output_dir,
            allow_oracle=args.allow_oracle,
            policy_filter=args.policy,
        )
        if code != 0:
            exit_code = code

    if args.evaluate:
        # production --evaluate path requires an explicit --mode argument;
        # we no longer infer mode from artifacts-on-disk or default to
        # ``diagnostic_non_oracle``.
        if args.mode is None:
            print(
                "ERROR: --evaluate requires explicit --mode "
                "(formal_closeout, diagnostic_non_oracle, or diagnostic_legacy).",
                file=sys.stderr,
            )
            return 2
        selected_thresholds_path = output_dir / "selected_thresholds.json"
        code = run_evaluate(
            config, env_config, selected_thresholds_path, output_dir,
            allow_oracle=args.allow_oracle,
            policy_filter=args.policy,
            mode=args.mode,
        )
        if code != 0:
            exit_code = code

    if args.all:
        # Formal --all must use formal_closeout semantics:
        #  - --allow-oracle is required (Oracle is part of the formal contract)
        #  - --mode formal_closeout is required (no diagnostic-mode fallback
        #    from a formal --all chain; diagnostic_legacy modes are NEVER
        #    reachable from --all)
        #  - scenario-bank load failures must abort (no smoke fallback into
        #    the formal output directory)
        if not args.allow_oracle:
            print(
                "ERROR: --all requires --allow-oracle (formal_closeout). "
                "Formal directories may not silently substitute a non-Oracle run.",
                file=sys.stderr,
            )
            return 2
        if args.mode is not None and args.mode != "formal_closeout":
            print(
                f"ERROR: --all requires --mode formal_closeout (got {args.mode!r}); "
                "diagnostic modes (diagnostic_non_oracle, diagnostic_legacy) "
                "are NEVER reachable from a formal --all chain.",
                file=sys.stderr,
            )
            return 2

        # Always run the chain in explicit formal_closeout mode; never
        # fall back to permissive or diagnostic semantics.
        formal_mode = "formal_closeout"

        # Strict path: tune then evaluate in this same output dir.
        # No "smoke fall-back" — if scenario-bank load fails the formal
        # path aborts.
        tune_config = get_default_config(
            split="rl_validation",
            maintenance_capacity=args.k_capacity,
            cost_regime_id=args.cost_regime,
        )
        code = run_tune(
            config, tune_config, output_dir,
            allow_oracle=args.allow_oracle,
            policy_filter=args.policy,
        )
        if code != 0:
            print(
                "ERROR: formal --all tuning aborted; refusing to fall back "
                "to smoke (no smoke fallback in formal mode)",
                file=sys.stderr,
            )
            return code
        code = run_evaluate(
            config, env_config,
            output_dir / "selected_thresholds.json", output_dir,
            allow_oracle=args.allow_oracle,
            policy_filter=args.policy,
            mode=formal_mode,
        )
        if code != 0:
            exit_code = code
            return exit_code

        # Step 9: validator writes validation_report.json (must exist
        # BEFORE manifest sealing).  Validator does not touch the
        # formal manifest.
        validate_cmd = [
            sys.executable,
            str(Path(__file__).parent / "validate_m3_artifacts.py"),
            str(output_dir),
            "--mode", "formal_closeout",
        ]
        vrc = subprocess.run(validate_cmd).returncode
        if vrc != 0:
            print(
                "ERROR: formal --all validation_report.json step failed; "
                "refusing to seal manifest without a validator verdict.",
                file=sys.stderr,
            )
            return vrc

        # Step 10: independent recomputation runs as subprocess.  No
        # production-counter imports; reads parquets directly.
        recompute_script = Path(__file__).parent / "independent_recompute_m3.py"
        recompute_cmd = [
            sys.executable,
            str(recompute_script),
            str(output_dir),
        ]
        rc = subprocess.run(recompute_cmd).returncode
        if rc != 0:
            print(
                "ERROR: formal --all independent_recomputation.json step failed; "
                "refusing to seal manifest without a PASS verdict.",
                file=sys.stderr,
            )
            return rc

        # Step 11: formal_manifest.json is sealed LAST and may not be
        # re-touched.  All inputs above are sourced from disk state.
        from src.baselines.artifacts import generate_formal_manifest
        try:
            generate_formal_manifest(output_dir, mode="formal_closeout")
        except RuntimeError as exc:
            print(
                f"ERROR: formal --all manifest seal failed: {exc}",
                file=sys.stderr,
            )
            return 1

    if args.validate_artifacts:
        if args.mode is None:
            print(
                "ERROR: --validate-artifacts requires --mode "
                "(formal_closeout or diagnostic_non_oracle)",
                file=sys.stderr,
            )
            return 2
        code = run_validate_artifacts(output_dir, mode=args.mode)
        if code != 0:
            exit_code = code

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
