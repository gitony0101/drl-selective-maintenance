#!/usr/bin/env python3
"""
Independent recomputation for M3 formal runs.

Reconstructs all identity sets and scientific reconciliations WITHOUT
importing any production modules (no planners, threshold selectors,
identity counters, summary generators, or validators).

Uses only generic JSON/CSV/Parquet readers (pandas, stdlib json).

Exit code: 0 = PASS, 1 = FAIL (any mismatch)
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# FROZEN CONTRACT CONSTANTS (must match production exactly)
# =============================================================================

FORMAL_POLICY_FAMILIES = (
    "age_threshold",
    "predicted_rul_threshold",
    "greedy_predicted_rul",
    "oracle_threshold",
)
NON_TUNED_POLICIES = {"corrective_only", "random_feasible"}

FORMAL_K_VALUES = (1, 2)
FORMAL_COST_REGIMES = (
    "failure-light-no-waste",
    "failure-heavy-no-waste",
    "failure-light-waste-aware",
    "failure-heavy-waste-aware",
)

EVAL_SPLITS = ("predictor_train", "rl_validation")
EVAL_POLICIES = (
    "corrective_only",
    "random_feasible",
    "age_threshold",
    "predicted_rul_threshold",
    "greedy_predicted_rul",
    "oracle_threshold",
)

# Frozen threshold grids (from M3 contract)
AGE_THRESHOLDS = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
PREDICTED_RUL_THRESHOLDS = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100]
GREEDY_ACTIVATION_THRESHOLDS = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100]
ORACLE_THRESHOLDS = [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50]

THRESHOLD_GRIDS = {
    "age_threshold": set(AGE_THRESHOLDS),
    "predicted_rul_threshold": set(PREDICTED_RUL_THRESHOLDS),
    "greedy_predicted_rul": set(GREEDY_ACTIVATION_THRESHOLDS),
    "oracle_threshold": set(ORACLE_THRESHOLDS),
}

# Expected counts (formal_closeout with Oracle)
EXPECTED_TUNING_CANDIDATES = 360   # 45 thresholds × 2 K × 4 regimes
EXPECTED_TUNING_EPISODES = 9000    # 360 × 5 scenarios × 5 seeds
EXPECTED_SELECTED_THRESHOLDS = 32  # 4 policies × 2 K × 4 regimes
EXPECTED_EVALUATION_EPISODES = 2400  # 6 policies × 2 K × 4 regimes × 2 splits × 5 × 5

# The exact frozen formal reset-seed set. The independent checker requires
# the sealed context to carry reset_seeds that equal THIS set (as a set,
# unique, non-empty). No fallback is permitted.
FROZEN_FORMAL_RESET_SEEDS = (6521, 6522, 6523, 6524, 6525)

# Oracle terminology contract: the Oracle policy is a privileged-information
# diagnostic benchmark, NOT an optimal policy / upper bound. These forbidden
# phrases characterize it as an optimum and must not appear in any generated
# textual/JSON/CSV evidence field the checker scans. The required label (or an
# equivalent clearly non-optimal diagnostic label) must be present wherever the
# evidence speaks of the Oracle policy's role.
ORACLE_FORBIDDEN_PHRASES = (
    "optimal oracle",
    "optimal policy",
    "upper bound",
    "upper-bound",
)
ORACLE_REQUIRED_LABEL = "privileged-information diagnostic benchmark"
# Accepted equivalent diagnostic labels (any one satisfies the required-label
# check in the scanned evidence).
ORACLE_ACCEPTED_LABELS = (
    "privileged-information diagnostic benchmark",
    "privileged-information diagnostic",
    "diagnostic benchmark",
    "diagnostic baseline",
    "privileged diagnostic benchmark",
)


# =============================================================================
# UTILITIES
# =============================================================================

def identity_key(policy_family: str, k: int, cost_regime_id: str) -> str:
    """Build canonical identity key."""
    return f"{policy_family}_k{k}_{cost_regime_id}"


def bank_identity_key(split: str, k: int, cost_regime_id: str) -> str:
    """Canonical identity key for a scenario-bank record."""
    return f"{split}_k{k}_{cost_regime_id}"


def candidate_identity(
    policy_family: str, threshold: int, k: int, cost_regime_id: str,
) -> str:
    """Canonical identity for a tuning candidate cell."""
    return f"{policy_family}|t{threshold}|k{k}|{cost_regime_id}"


def tuning_episode_identity(
    policy_family: str,
    threshold: int,
    k: int,
    cost_regime_id: str,
    scenario_id: str,
    reset_seed: int,
) -> str:
    """Canonical identity for a tuning-episode row."""
    return (
        f"{policy_family}|t{threshold}|k{k}|{cost_regime_id}"
        f"|{scenario_id}|r{reset_seed}"
    )


def evaluation_identity(
    policy: str,
    split: str,
    k: int,
    cost_regime_id: str,
    scenario_id: str,
    reset_seed: int,
) -> str:
    """Canonical identity for an evaluation episode row."""
    return (
        f"{policy}|{split}|k{k}|{cost_regime_id}"
        f"|{scenario_id}|r{reset_seed}"
    )


def compute_set_sha(identities) -> str:
    """SHA256 of the sorted identity set (order-independent fingerprint)."""
    return hashlib.sha256(
        "\n".join(sorted(identities)).encode("utf-8")
    ).hexdigest()


def compute_sha256(file_path: Path) -> str:
    """Stream-compute SHA256 of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json_bytes(config: Dict[str, Any]) -> bytes:
    """Serialize a config dict to deterministically canonical JSON bytes.

    Independent (no production import) re-implementation of
    ``src.baselines.artifacts._canonical_json_bytes`` so the checker can
    verify resolved_config.json's SHA without importing production code.
    """
    def _normalize(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _normalize(obj[k]) for k in sorted(obj.keys())}
        if isinstance(obj, list):
            return [_normalize(v) for v in obj]
        if isinstance(obj, tuple):
            return [_normalize(v) for v in obj]
        if isinstance(obj, float):
            return float(obj)
        return obj
    return json.dumps(
        _normalize(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def compute_canonical_config_sha256(config: Dict[str, Any]) -> str:
    """Independent canonical-config SHA256 (mirrors production exactly)."""
    return hashlib.sha256(_canonical_json_bytes(config)).hexdigest()


def log(msg: str) -> None:
    print(f"  {msg}")


def _to_jsonable(obj: Any) -> Any:
    """Recursively coerce numpy / pandas scalars to JSON-native types."""
    import numpy as _np
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return [_to_jsonable(v) for v in sorted(obj, key=lambda x: str(x))]
    if isinstance(obj, _np.integer):
        return int(obj)
    if isinstance(obj, _np.floating):
        f = float(obj)
        if f != f or f in (float("inf"), float("-inf")):
            return None
        return f
    if isinstance(obj, _np.bool_):
        return bool(obj)
    if isinstance(obj, _np.ndarray):
        return [_to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj
    return obj


def error(errors: List[str], msg: str) -> None:
    errors.append(msg)
    print(f"  ✗ {msg}")


def ok(errors: List[str], msg: str) -> None:
    if not errors:
        print(f"  ✓ {msg}")


_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


# =============================================================================
# SEALED FORMAL CONTEXT VERIFICATION (no fallbacks)
# =============================================================================

def verify_sealed_formal_context(
    formal_context_path: Path,
    output_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """Verify formal_run_context.json is valid and strictly sealed.

    All of the following must hold (any failure is recorded; no fallback):
      - sealed == True
      - sealed_at non-null str
      - selected_thresholds_sha256 64-char lowercase hex
      - implementation_tree_clean == True
      - implementation_commit full 40 lowercase hex
      - oracle_authorized == True
      - reset_seeds non-empty, unique, exactly FROZEN_FORMAL_RESET_SEEDS
      - resolved_config.json exists & SHA matches context.resolved_config_sha256
      - selected_thresholds.json exists & SHA matches context.selected_thresholds_sha256

    Returns (context_dict, verification_evidence, errors).
    """
    errs: List[str] = []
    with open(formal_context_path, "r") as f:
        context = json.load(f)

    evidence: Dict[str, Any] = {
        "sealed": bool(context.get("sealed")),
        "sealed_at": context.get("sealed_at"),
        "selected_thresholds_sha256": context.get("selected_thresholds_sha256"),
        "implementation_tree_clean": context.get("implementation_tree_clean"),
        "implementation_commit": context.get("implementation_commit"),
        "oracle_authorized": context.get("oracle_authorized"),
        "reset_seeds": context.get("reset_seeds"),
        "resolved_config_sha256": context.get("resolved_config_sha256"),
    }

    if not context.get("sealed"):
        error(errs, "formal_run_context.json: sealed is not True")
    sealed_at = context.get("sealed_at")
    if sealed_at is None or not isinstance(sealed_at, str) or not sealed_at.strip():
        error(errs, f"formal_run_context.json: sealed_at invalid ({sealed_at!r})")
        sealed_at_ok = False
    else:
        sealed_at_ok = True
    evidence["sealed_at_valid"] = sealed_at_ok

    sel_sha = context.get("selected_thresholds_sha256")
    if not (isinstance(sel_sha, str) and _HEX64_RE.match(sel_sha)):
        error(errs, f"formal_run_context.json: selected_thresholds_sha256 invalid ({str(sel_sha)[:16]}…)")

    if context.get("implementation_tree_clean") is not True:
        error(errs, "formal_run_context.json: implementation_tree_clean is not True")

    impl_commit = context.get("implementation_commit")
    if not (isinstance(impl_commit, str) and _HEX40_RE.match(impl_commit)):
        error(errs, f"formal_run_context.json: implementation_commit not 40 lowercase hex ({str(impl_commit)[:16]}…)")

    if context.get("oracle_authorized") is not True:
        error(errs, "formal_run_context.json: oracle_authorized is not True")

    reset_seeds = context.get("reset_seeds")
    if reset_seeds is None:
        error(errs, "formal_run_context.json: reset_seeds missing (no fallback permitted)")
        reset_seeds = []
    if not isinstance(reset_seeds, list) or len(reset_seeds) == 0:
        error(errs, "formal_run_context.json: reset_seeds empty or not a list")
    else:
        if len(set(reset_seeds)) != len(reset_seeds):
            error(errs, f"formal_run_context.json: reset_seeds contains duplicates ({reset_seeds})")
        if set(reset_seeds) != set(FROZEN_FORMAL_RESET_SEEDS):
            error(errs, (
                f"formal_run_context.json: reset_seeds {reset_seeds} != "
                f"frozen formal set {list(FROZEN_FORMAL_RESET_SEEDS)}"
            ))

    # resolved_config.json exists & SHA matches
    rc_path_str = context.get("resolved_config_path")
    rc_path = Path(rc_path_str) if rc_path_str else (output_dir / "resolved_config.json")
    rc_exists = rc_path.exists()
    rc_match = None
    if not rc_exists:
        error(errs, f"resolved_config.json not found at {rc_path}")
    else:
        with open(rc_path, "r") as f:
            rc_data = json.load(f)
        actual_rc_sha = compute_canonical_config_sha256(rc_data)
        expected_rc_sha = context.get("resolved_config_sha256")
        rc_match = (actual_rc_sha == expected_rc_sha)
        if not rc_match:
            error(errs, (
                "resolved_config.json SHA mismatch: "
                f"context={str(expected_rc_sha)[:16]}… "
                f"actual={actual_rc_sha[:16]}…"
            ))
    evidence["resolved_config_path"] = str(rc_path)
    evidence["resolved_config_exists"] = rc_exists
    evidence["resolved_config_sha_match"] = rc_match

    # selected_thresholds.json exists & SHA matches
    sel_path_str = context.get("selected_thresholds_path")
    sel_path = Path(sel_path_str) if sel_path_str else (output_dir / "selected_thresholds.json")
    sel_exists = sel_path.exists()
    sel_match = None
    if not sel_exists:
        error(errs, f"selected_thresholds.json not found at {sel_path}")
    else:
        actual_sel_sha = compute_sha256(sel_path)
        expected_sel_sha = context.get("selected_thresholds_sha256")
        sel_match = (actual_sel_sha == expected_sel_sha)
        if not sel_match:
            error(errs, (
                "selected_thresholds.json SHA mismatch: "
                f"context={str(expected_sel_sha)[:16]}… "
                f"actual={actual_sel_sha[:16]}…"
            ))
    evidence["selected_thresholds_path"] = str(sel_path)
    evidence["selected_thresholds_exists"] = sel_exists
    evidence["selected_thresholds_sha_match"] = sel_match

    evidence["verdict"] = "PASS" if not errs else "FAIL"
    return context, evidence, errs


def verify_resolved_config_contract(context: Dict[str, Any], output_dir: Path) -> Tuple[Dict[str, Any], List[str]]:
    """Verify resolved_config.json's policy families / threshold grids / K /
    cost regimes / eval splits / reset seeds match the frozen formal contract.

    Returns (config_data, verification_subfields, errors).
    """
    errs: List[str] = []
    rc_path_str = context.get("resolved_config_path")
    rc_path = Path(rc_path_str) if rc_path_str else (output_dir / "resolved_config.json")
    if not rc_path.exists():
        error(errs, f"resolved_config.json not found at {rc_path} for contract verification")
        return {}, {"verdict": "FAIL", "errors": errs}, errs
    with open(rc_path, "r") as f:
        config_data = json.load(f)

    sub: Dict[str, Any] = {"verdict": "PASS"}

    fams = set(config_data.get("policy_families", []))
    # If policy_families not present, infer from threshold_grids keys.
    if not fams:
        fams = set(config_data.get("threshold_grids", {}).keys())
    expected_fams = set(FORMAL_POLICY_FAMILIES)
    sub["policy_families"] = {
        "expected": sorted(expected_fams),
        "actual": sorted(fams),
        "match": fams == expected_fams,
    }
    if fams != expected_fams:
        error(errs, f"resolved_config policy_families mismatch: {sorted(fams)} != {sorted(expected_fams)}")

    grids = config_data.get("threshold_grids", {})
    grid_match = True
    grid_detail = {}
    for pol, exp_grid in THRESHOLD_GRIDS.items():
        act = set(grids.get(pol, []))
        match = act == exp_grid
        grid_match = grid_match and match
        grid_detail[pol] = {"expected": sorted(exp_grid), "actual": sorted(act), "match": match}
        if not match:
            error(errs, f"threshold grid mismatch for {pol}: {sorted(act)} != {sorted(exp_grid)}")
    sub["threshold_grids"] = {"match": grid_match, "grids": grid_detail}

    k_vals = set(config_data.get("k_values", []))
    sub["k_values"] = {"expected": list(FORMAL_K_VALUES), "actual": sorted(k_vals), "match": k_vals == set(FORMAL_K_VALUES)}
    if k_vals != set(FORMAL_K_VALUES):
        error(errs, f"resolved_config k_values mismatch: {sorted(k_vals)} != {list(FORMAL_K_VALUES)}")

    regimes = set(config_data.get("cost_regimes", []))
    sub["cost_regimes"] = {"expected": list(FORMAL_COST_REGIMES), "actual": sorted(regimes), "match": regimes == set(FORMAL_COST_REGIMES)}
    if regimes != set(FORMAL_COST_REGIMES):
        error(errs, f"resolved_config cost_regimes mismatch: {sorted(regimes)} != {list(FORMAL_COST_REGIMES)}")

    eval_splits = config_data.get("evaluation_splits", [])
    sub["evaluation_splits"] = {"expected": list(EVAL_SPLITS), "actual": sorted(eval_splits), "match": set(eval_splits) == set(EVAL_SPLITS)}
    if set(eval_splits) != set(EVAL_SPLITS):
        error(errs, f"resolved_config evaluation_splits mismatch: {sorted(eval_splits)} != {list(EVAL_SPLITS)}")

    seeds = config_data.get("reset_seeds", [])
    sub["reset_seeds"] = {"expected": list(FROZEN_FORMAL_RESET_SEEDS), "actual": list(seeds), "match": set(seeds) == set(FROZEN_FORMAL_RESET_SEEDS)}
    if set(seeds) != set(FROZEN_FORMAL_RESET_SEEDS):
        error(errs, f"resolved_config reset_seeds mismatch: {seeds} != {list(FROZEN_FORMAL_RESET_SEEDS)}")

    sub["verdict"] = "PASS" if not errs else "FAIL"
    return config_data, sub, errs


# =============================================================================
# SCENARIO BANK VERIFICATION (independent reopening of source files)
# =============================================================================

def _bank_split(bank: Dict[str, Any]) -> str:
    return str(bank.get("split",
        bank.get("derived_split", "unknown")))


def _bank_k(bank: Dict[str, Any]) -> int:
    v = bank.get("K")
    if v is None:
        v = bank.get("k")
    if v is None:
        v = bank.get("derived_k")
    try:
        return int(v)
    except (TypeError, ValueError):
        raise ValueError(f"bank has no parseable K/derived_k: {bank}")


def _bank_regime(bank: Dict[str, Any]) -> str:
    v = bank.get("cost_regime_id")
    if v is None:
        v = bank.get("derived_cost_regime_id")
    if v is None:
        raise ValueError(f"bank has no cost_regime_id/derived_cost_regime_id: {bank}")
    return str(v)


def _bank_scenario_count(bank: Dict[str, Any]) -> Optional[int]:
    for key in ("scenario_count", "derived_scenario_count"):
        if key in bank and bank[key] is not None:
            try:
                return int(bank[key])
            except (TypeError, ValueError):
                return None
    return None


def verify_scenario_bank_sources(
    formal_context_path: Path,
    errors: List[str],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]], Dict[str, Any], Dict[str, Any]]:
    """Verify every scenario-bank source file independently.

    Returns:
        (bank_identities_map, scenario_ids_by_bank_key,
         scenario_bank_set_evidence, scenario_bank_file_evidence)
    """
    with open(formal_context_path, "r") as f:
        context = json.load(f)

    bank_identities = context.get("scenario_bank_identities", [])
    if not bank_identities:
        error(errors, "No scenario_bank_identities in formal_run_context.json")
        return {}, {}, {"verdict": "FAIL", "errors": ["no scenario_bank_identities"]}, []

    # ---- 16-bank identity SET verification (split × K × cost_regime_id) ----
    expected_set: Set[str] = set()
    for split in EVAL_SPLITS:
        for k in FORMAL_K_VALUES:
            for regime in FORMAL_COST_REGIMES:
                expected_set.add(bank_identity_key(split, k, regime))

    actual_keys: List[str] = []
    duplicate_keys: List[str] = []
    seen: Set[str] = set()
    for b in bank_identities:
        try:
            split = _bank_split(b)
            k = _bank_k(b)
            regime = _bank_regime(b)
            key = bank_identity_key(split, k, regime)
        except ValueError as e:
            error(errors, f"scenario_bank_identities: malformed record: {e}")
            continue
        actual_keys.append(key)
        if key in seen:
            duplicate_keys.append(key)
        seen.add(key)

    actual_set = set(actual_keys)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)

    for b in bank_identities:
        # reject forbidden splits (rl_test) in any record
        try:
            if _bank_split(b) == "rl_test":
                error(errors, "scenario_bank_identities: rl_test split present (forbidden in formal context)")
        except ValueError:
            pass

    bank_set_evidence: Dict[str, Any] = {
        "expected_count": len(expected_set),
        "actual_row_count": len(actual_keys),
        "actual_unique_count": len(actual_set),
        "expected_set_sha256": compute_set_sha(expected_set),
        "actual_set_sha256": compute_set_sha(actual_set),
        "missing_bank_identities": missing,
        "extra_bank_identities": extra,
        "duplicate_bank_identities": sorted(set(duplicate_keys)),
        "verdict": "PASS" if (not missing and not extra and not duplicate_keys
                              and len(actual_set) == len(expected_set)) else "FAIL",
    }
    if missing:
        error(errors, f"scenario_bank_set: missing bank identities ({len(missing)}): {missing[:5]}…")
    if extra:
        error(errors, f"scenario_bank_set: extra bank identities ({len(extra)}): {extra[:5]}…")
    if duplicate_keys:
        error(errors, f"scenario_bank_set: duplicate bank identities: {sorted(set(duplicate_keys))}")
    if len(actual_set) != len(expected_set):
        error(errors, f"scenario_bank_set: unique identity count {len(actual_set)} != {len(expected_set)}")

    # ---- per-bank FILE verification (independent reopening + SHA + IDs) ----
    bank_identities_map: Dict[str, Dict[str, Any]] = {}
    scenario_ids_by_bank: Dict[str, List[str]] = {}
    file_evidence: List[Dict[str, Any]] = []

    for b in bank_identities:
        try:
            split = _bank_split(b)
            k = _bank_k(b)
            regime = _bank_regime(b)
        except ValueError as e:
            error(errors, f"scenario_bank_identities: malformed record: {e}")
            continue
        bank_key = bank_identity_key(split, k, regime)

        src_path_str = b.get("source_path")
        if not src_path_str:
            error(errors, f"Bank {bank_key}: missing source_path")
            file_evidence.append({
                "bank_key": bank_key, "source_path": None,
                "source_file_exists": False, "parse_success": False,
                "source_sha_format_valid": False, "source_sha_match": False,
                "scenario_count_present": False, "scenario_count_match": False,
                "scenario_ids_sha_format_valid": False, "scenario_ids_sha_match": False,
                "actual_scenario_count": 0, "verdict": "FAIL",
            })
            continue

        src_path = Path(src_path_str)
        source_file_exists = src_path.exists()
        if not source_file_exists:
            error(errors, f"Bank {bank_key}: source file not found at {src_path_str}")
            file_evidence.append({
                "bank_key": bank_key, "source_path": src_path_str,
                "source_file_exists": False, "parse_success": False,
                "source_sha_format_valid": False, "source_sha_match": False,
                "scenario_count_present": False, "scenario_count_match": False,
                "scenario_ids_sha_format_valid": False, "scenario_ids_sha_match": False,
                "actual_scenario_count": 0, "verdict": "FAIL",
            })
            continue

        actual_src_sha = compute_sha256(src_path)
        expected_src_sha_raw = b.get("source_sha256")
        source_sha_format_valid = bool(
            isinstance(expected_src_sha_raw, str) and _HEX64_RE.match(expected_src_sha_raw)
        )
        source_sha_match = source_sha_format_valid and (actual_src_sha == expected_src_sha_raw)
        if not source_sha_format_valid:
            error(errors, f"Bank {bank_key}: source_sha256 missing or malformed ({str(expected_src_sha_raw)[:16]}…)")
        elif not source_sha_match:
            error(errors, (
                f"Bank {bank_key}: source_sha256 mismatch: "
                f"context={expected_src_sha_raw[:12]}… actual={actual_src_sha[:12]}…"
            ))

        parse_success = True
        try:
            with open(src_path, "r") as f:
                bank_data = json.load(f)
        except Exception as e:
            parse_success = False
            error(errors, f"Bank {bank_key}: failed to parse JSON: {e}")
            file_evidence.append({
                "bank_key": bank_key, "source_path": src_path_str,
                "source_file_exists": True, "parse_success": False,
                "source_sha_format_valid": source_sha_format_valid,
                "source_sha_match": source_sha_match,
                "scenario_count_present": False, "scenario_count_match": False,
                "scenario_ids_sha_format_valid": False, "scenario_ids_sha_match": False,
                "actual_scenario_count": 0, "verdict": "FAIL",
            })
            continue

        scenarios = bank_data.get("scenarios", [])
        actual_ids = sorted(s.get("scenario_id") for s in scenarios if s.get("scenario_id"))
        actual_count = len(actual_ids)

        expected_count_raw = _bank_scenario_count(b)
        scenario_count_present = expected_count_raw is not None
        scenario_count_match = scenario_count_present and (actual_count == expected_count_raw)
        if scenario_count_present and not scenario_count_match:
            error(errors, f"Bank {bank_key}: scenario count mismatch: expected {expected_count_raw}, got {actual_count}")

        # scenario-ID hash (sorted ids joined by newline — matches _sorted_scenario_ids_sha256)
        recomputed_ids_sha = hashlib.sha256(
            "\n".join(actual_ids).encode("utf-8")
        ).hexdigest()
        recorded_ids_sha_raw = b.get("sorted_scenario_ids_sha256")
        scenario_ids_sha_format_valid = bool(
            isinstance(recorded_ids_sha_raw, str) and _HEX64_RE.match(recorded_ids_sha_raw)
        )
        scenario_ids_sha_match = scenario_ids_sha_format_valid and (recomputed_ids_sha == recorded_ids_sha_raw)
        if not scenario_ids_sha_format_valid:
            error(errors, f"Bank {bank_key}: sorted_scenario_ids_sha256 missing or malformed ({str(recorded_ids_sha_raw)[:16]}…)")
        elif not scenario_ids_sha_match:
            error(errors, (
                f"Bank {bank_key}: sorted_scenario_ids_sha256 mismatch: "
                f"context={recorded_ids_sha_raw[:12]}… recomputed={recomputed_ids_sha[:12]}…"
            ))

        bank_identities_map[bank_key] = b
        scenario_ids_by_bank[bank_key] = actual_ids

        per_bank_pass = (
            source_file_exists and parse_success
            and source_sha_format_valid and source_sha_match
            and scenario_count_present and scenario_count_match
            and scenario_ids_sha_format_valid and scenario_ids_sha_match
        )
        file_evidence.append({
            "bank_key": bank_key,
            "source_path": src_path_str,
            "source_file_exists": source_file_exists,
            "parse_success": parse_success,
            "source_sha_format_valid": source_sha_format_valid,
            "source_sha_format": expected_src_sha_raw,
            "source_sha_match": source_sha_match,
            "expected_source_sha": expected_src_sha_raw,
            "actual_source_sha": actual_src_sha,
            "scenario_count_present": scenario_count_present,
            "scenario_count_match": scenario_count_match,
            "expected_scenario_count": expected_count_raw,
            "actual_scenario_count": actual_count,
            "scenario_ids_sha_format_valid": scenario_ids_sha_format_valid,
            "scenario_ids_sha_match": scenario_ids_sha_match,
            "expected_scenario_id_sha": recorded_ids_sha_raw,
            "actual_scenario_id_sha": recomputed_ids_sha,
            "verdict": "PASS" if per_bank_pass else "FAIL",
        })
        log(f"Bank {bank_key}: {actual_count} scenarios, SHA match={recomputed_ids_sha[:12]}…")

    return (bank_identities_map, scenario_ids_by_bank,
            bank_set_evidence, file_evidence)


# =============================================================================
# RECONSTRUCTION FUNCTIONS (read actual IDs from verified banks)
# =============================================================================

def reconstruct_candidate_identities(
    threshold_results_path: Path,
    errors: List[str],
) -> Tuple[Set[str], Dict[str, Any]]:
    """A. Candidate identity set (exact construction + comparison)."""
    expected_identities: Set[str] = set()
    for policy in FORMAL_POLICY_FAMILIES:
        grid = THRESHOLD_GRIDS[policy]
        for k in FORMAL_K_VALUES:
            for regime in FORMAL_COST_REGIMES:
                for thresh in sorted(grid):
                    expected_identities.add(
                        candidate_identity(policy, thresh, k, regime)
                    )

    actual_identities: Set[str] = set()
    actual_row_count = 0
    try:
        df = pd.read_parquet(threshold_results_path)
        actual_row_count = len(df)
        if df.empty:
            error(errors, "threshold_search_results.parquet is empty")
        else:
            for _, row in df.iterrows():
                actual_identities.add(
                    candidate_identity(
                        str(row["policy_family"]),
                        int(row["threshold"]),
                        int(row["k_capacity"]),
                        str(row["cost_regime_id"]),
                    )
                )
    except Exception as e:
        error(errors, f"Failed to read threshold_search_results.parquet: {e}")

    expected_count = EXPECTED_TUNING_CANDIDATES
    actual_unique_count = len(actual_identities)
    missing = expected_identities - actual_identities
    extra = actual_identities - expected_identities

    # The parquet is tuning-EPISODE level (one row per scenario × seed per
    # candidate); the candidate IDENTITY is the unique (policy, threshold,
    # K, regime) projection. Only the unique count must equal 360; row
    # count is informational (9000 in a valid formal run).
    if actual_unique_count != expected_count:
        error(errors, (
            f"Candidate identity: expected {expected_count} unique "
            f"(policy,threshold,K,regime) tuples, got "
            f"{actual_unique_count} (from {actual_row_count} rows).",
        ))
    else:
        ok(errors, f"Candidate identities: {actual_unique_count} == {expected_count} "
                   f"(from {actual_row_count} rows)")

    expected_sha = compute_set_sha(expected_identities)
    actual_sha = compute_set_sha(actual_identities)
    if expected_sha != actual_sha:
        error(errors, f"Candidate set SHA mismatch: expected={expected_sha[:12]}… actual={actual_sha[:12]}…")

    debug = {
        "expected_count": expected_count,
        "actual_row_count": actual_row_count,
        "actual_unique_count": actual_unique_count,
        "expected_set_sha256": expected_sha,
        "actual_set_sha256": actual_sha,
        "missing_identities": sorted(missing),
        "extra_identities": sorted(extra),
        "duplicate_identities": [],
        "verdict": "PASS" if (not missing and not extra
                              and actual_unique_count == expected_count) else "FAIL",
    }
    return actual_identities, debug


def reconstruct_tuning_episode_set(
    threshold_results_path: Path,
    scenario_ids_by_bank: Dict[str, List[str]],
    reset_seeds: List[int],
    errors: List[str],
) -> Tuple[Set[str], Dict[str, Any]]:
    """B. Tuning episode identity set (exact construction + comparison).

    The expected set maps each candidate (policy_family, threshold, K,
    cost_regime_id) to the EXACT (rl_validation, K, cost_regime_id) bank
    and uses ONLY that bank's actual scenario IDs — banks are NOT assumed
    to share scenario IDs across K/regime.
    """
    expected_identities: Set[str] = set()
    bank_missing: List[str] = []
    for policy in FORMAL_POLICY_FAMILIES:
        grid = THRESHOLD_GRIDS[policy]
        for k in FORMAL_K_VALUES:
            for regime in FORMAL_COST_REGIMES:
                bank_key = bank_identity_key("rl_validation", k, regime)
                scenario_ids = scenario_ids_by_bank.get(bank_key, [])
                if not scenario_ids:
                    if bank_key not in bank_missing:
                        error(errors, f"Tuning: no rl_validation bank for {bank_key}")
                        bank_missing.append(bank_key)
                    continue
                for thresh in sorted(grid):
                    for scenario_id in scenario_ids:
                        for seed in reset_seeds:
                            expected_identities.add(
                                tuning_episode_identity(
                                    policy, thresh, k, regime, scenario_id, int(seed)
                                )
                            )

    actual_identities: Set[str] = set()
    actual_row_count = 0
    try:
        df = pd.read_parquet(threshold_results_path)
        actual_row_count = len(df)
        if df.empty:
            error(errors, "threshold_search_results.parquet is empty")
        else:
            for _, row in df.iterrows():
                actual_identities.add(
                    tuning_episode_identity(
                        str(row["policy_family"]),
                        int(row["threshold"]),
                        int(row["k_capacity"]),
                        str(row["cost_regime_id"]),
                        str(row["scenario_id"]),
                        int(row["reset_seed"]),
                    )
                )
    except Exception as e:
        error(errors, f"Failed to read threshold_search_results.parquet: {e}")

    actual_unique_count = len(actual_identities)
    expected_count = len(expected_identities)
    missing = expected_identities - actual_identities
    extra = actual_identities - expected_identities

    expected_sha = compute_set_sha(expected_identities)
    actual_sha = compute_set_sha(actual_identities)

    # FORMAL-CONTRACT COUNT GATE: the per-bank universe must yield exactly
    # 9000 tuning identities. A smaller self-consistent universe (e.g. 4
    # scenarios/bank → 7200 rows) is internally consistent (row count matches
    # the bank-derived expected set) but violates the frozen formal contract.
    # The recompute must reject it regardless of internal consistency.
    formal_count_match = (expected_count == EXPECTED_TUNING_EPISODES)
    if not formal_count_match:
        error(errors, (
            f"Tuning formal-contract count: bank-derived expected={expected_count} "
            f"!= frozen formal contract {EXPECTED_TUNING_EPISODES} (wrong scenario "
            f"count per bank — a self-consistent smaller universe is still rejected)"
        ))

    if actual_row_count != expected_count:
        error(errors, f"Tuning episodes: expected {expected_count} rows, got {actual_row_count}")
    if actual_unique_count != expected_count:
        error(errors, f"Tuning unique identities: expected {expected_count}, got {actual_unique_count}")
    if missing:
        error(errors, f"Tuning identities missing ({len(missing)}): {sorted(missing)[:3]}…")
    if extra:
        error(errors, f"Tuning identities extra ({len(extra)}): {sorted(extra)[:3]}…")
    if expected_sha != actual_sha:
        error(errors, f"Tuning set SHA mismatch: expected={expected_sha[:12]}… actual={actual_sha[:12]}…")
    else:
        ok(errors, f"Tuning identities: {actual_unique_count} == {expected_count}")

    # Duplicate detection by row: rows > unique means duplicates on the 6-tuple.
    duplicate_identities: List[str] = []
    if actual_row_count != actual_unique_count:
        seen_local: Set[str] = set()
        try:
            df_dup = pd.read_parquet(threshold_results_path)
            for _, row in df_dup.iterrows():
                ident = tuning_episode_identity(
                    str(row["policy_family"]), int(row["threshold"]),
                    int(row["k_capacity"]), str(row["cost_regime_id"]),
                    str(row["scenario_id"]), int(row["reset_seed"]),
                )
                if ident in seen_local:
                    duplicate_identities.append(ident)
                else:
                    seen_local.add(ident)
        except Exception:
            pass

    debug = {
        "expected_count": expected_count,
        "actual_row_count": actual_row_count,
        "actual_unique_count": actual_unique_count,
        "formal_contract_count": EXPECTED_TUNING_EPISODES,
        "formal_contract_count_match": formal_count_match,
        "expected_set_sha256": expected_sha,
        "actual_set_sha256": actual_sha,
        "missing_identities": sorted(missing),
        "extra_identities": sorted(extra),
        "duplicate_identities": sorted(set(duplicate_identities)),
        "verdict": "PASS" if (formal_count_match
                              and not missing and not extra and not duplicate_identities
                              and actual_row_count == expected_count
                              and actual_unique_count == expected_count) else "FAIL",
    }
    return actual_identities, debug


def reconstruct_selected_winners(
    threshold_results_path: Path,
    selected_path: Path,
    errors: List[str],
) -> Tuple[Dict[str, int], List[str], Dict[str, Any]]:
    """C. Selected winners: independently recompute from
    threshold_search_results.parquet.

    Returns (winners, tie_break_evidence, winner_evidence_summary).
    """
    winners: Dict[str, int] = {}
    tie_break_records: List[Dict[str, Any]] = []

    try:
        df = pd.read_parquet(threshold_results_path)
    except Exception as e:
        error(errors, f"Cannot read threshold_search_results.parquet: {e}")
        return winners, [], {"actual_count": 0, "verdict": "FAIL"}

    try:
        with open(selected_path, "r") as f:
            selected = json.load(f)
        selected = {k: v for k, v in selected.items() if k != "_meta"}
    except Exception as e:
        error(errors, f"Cannot read selected_thresholds.json: {e}")
        return winners, [], {"actual_count": 0, "verdict": "FAIL"}

    required_cols = {
        "policy_family",
        "threshold",
        "k_capacity",
        "cost_regime_id",
        "total_cost",
        "failure_count",
        "wasted_life_cost",
        "completed",
    }
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        error(errors, f"Tuning episode evidence missing columns: {missing_cols}")
        return winners, [], {"actual_count": 0, "verdict": "FAIL"}

    incomplete_count = int((~df["completed"].astype(bool)).sum())
    if incomplete_count:
        error(errors, f"Tuning episode evidence has {incomplete_count} incomplete rows")

    completed = df[df["completed"].astype(bool)]
    candidate_cols = [
        "policy_family",
        "threshold",
        "k_capacity",
        "cost_regime_id",
    ]
    candidates = (
        completed.groupby(candidate_cols, as_index=False)
        .agg(
            mean_total_cost=("total_cost", "mean"),
            total_failures=("failure_count", "sum"),
            mean_wasted_life_cost=("wasted_life_cost", "mean"),
            episode_count=("total_cost", "size"),
        )
    )

    for policy in FORMAL_POLICY_FAMILIES:
        for k in FORMAL_K_VALUES:
            for regime in FORMAL_COST_REGIMES:
                key = identity_key(policy, k, regime)
                group = candidates[
                    (candidates["policy_family"] == policy) &
                    (candidates["k_capacity"] == k) &
                    (candidates["cost_regime_id"] == regime)
                ]
                if len(group) == 0:
                    error(errors, f"{key}: no tuning results found")
                    continue

                group_sorted = group.sort_values(
                    by=["mean_total_cost", "total_failures", "mean_wasted_life_cost", "threshold"],
                    ascending=[True, True, True, True],
                )
                best = group_sorted.iloc[0]
                recomputed = int(best["threshold"])
                winners[key] = recomputed

                tie_reason: Optional[str] = None
                if len(group) > 1:
                    second = group_sorted.iloc[1]
                    if best["mean_total_cost"] < second["mean_total_cost"]:
                        tie_reason = "lowest mean total cost"
                    elif best["total_failures"] < second["total_failures"]:
                        tie_reason = "fewest failures (tie on cost)"
                    elif best["mean_wasted_life_cost"] < second["mean_wasted_life_cost"]:
                        tie_reason = "lowest wasted-life cost (tie on cost and failures)"
                    else:
                        tie_reason = "lowest threshold (tie on all metrics)"

                if key not in selected:
                    error(errors, f"{key}: missing in selected_thresholds.json")
                    tie_break_records.append({
                        "key": key, "recomputed_threshold": recomputed,
                        "recorded_threshold": None, "tie_break_reason": tie_reason,
                        "recorded_tie_break_reason": None, "verdict": "FAIL",
                    })
                    continue

                recorded = selected[key].get("threshold")
                recorded_reason = selected[key].get("tie_break_reason", "")
                verdict = "PASS" if (recomputed == recorded and (tie_reason is None or tie_reason == recorded_reason)) else "FAIL"
                if recomputed != recorded:
                    error(errors, f"{key}: winner threshold {recomputed} ≠ recorded {recorded}")
                if tie_reason is not None and recorded_reason != tie_reason:
                    error(errors, f"{key}: tie_break_reason '{recorded_reason}' ≠ expected '{tie_reason}'")
                tie_break_records.append({
                    "key": key,
                    "recomputed_threshold": recomputed,
                    "recorded_threshold": recorded,
                    "tie_break_reason": tie_reason,
                    "recorded_tie_break_reason": recorded_reason,
                    "verdict": verdict,
                })

    expected_count = EXPECTED_SELECTED_THRESHOLDS
    actual_count = len(winners)
    if actual_count != expected_count:
        error(errors, f"Selected threshold count: expected {expected_count}, got {actual_count}")
    else:
        ok(errors, f"Selected thresholds: {actual_count} == {expected_count}")

    summary = {
        "expected_count": expected_count,
        "actual_count": actual_count,
        "verdict": "PASS" if (actual_count == expected_count and all(t["verdict"] == "PASS" for t in tie_break_records)) else "FAIL",
    }
    return winners, tie_break_records, summary


def reconcile_candidate_summary(
    threshold_results_path: Path,
    candidate_summary_path: Path,
    errors: List[str],
) -> Dict[str, Any]:
    """Independently aggregate episode rows and compare the 360-row CSV.

    The CSV is human-facing candidate evidence. It must be a faithful
    deterministic aggregation of the canonical episode-level parquet, never a
    separately fabricated source of metrics.
    """
    key_cols = [
        "policy_family",
        "threshold",
        "k_capacity",
        "cost_regime_id",
    ]
    metric_cols = [
        "mean_total_cost",
        "total_failures",
        "mean_wasted_life_cost",
        "episode_count",
    ]
    evidence: Dict[str, Any] = {
        "expected_count": EXPECTED_TUNING_CANDIDATES,
        "actual_count": 0,
        "missing_identities": [],
        "extra_identities": [],
        "metric_mismatch_count": 0,
        "verdict": "FAIL",
    }
    try:
        episodes = pd.read_parquet(threshold_results_path)
        summary = pd.read_csv(candidate_summary_path)
    except Exception as exc:
        error(errors, f"Cannot read candidate evidence: {exc}")
        return evidence

    required_episode = set(key_cols + [
        "total_cost", "failure_count", "wasted_life_cost", "completed"
    ])
    required_summary = set(key_cols + metric_cols)
    missing_episode_cols = sorted(required_episode - set(episodes.columns))
    missing_summary_cols = sorted(required_summary - set(summary.columns))
    if missing_episode_cols or missing_summary_cols:
        error(
            errors,
            "Candidate summary columns missing: "
            f"episode={missing_episode_cols}, summary={missing_summary_cols}",
        )
        return evidence

    completed = episodes[episodes["completed"].astype(bool)]
    aggregated = (
        completed.groupby(key_cols, as_index=False)
        .agg(
            mean_total_cost=("total_cost", "mean"),
            total_failures=("failure_count", "sum"),
            mean_wasted_life_cost=("wasted_life_cost", "mean"),
            episode_count=("total_cost", "size"),
        )
    )

    def _identity_set(frame: pd.DataFrame) -> Set[str]:
        return {
            candidate_identity(
                str(row["policy_family"]),
                int(row["threshold"]),
                int(row["k_capacity"]),
                str(row["cost_regime_id"]),
            )
            for _, row in frame.iterrows()
        }

    expected_ids = _identity_set(aggregated)
    actual_ids = _identity_set(summary)
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    duplicates = int(summary.duplicated(subset=key_cols).sum())
    mismatch_count = 0

    if len(summary) != EXPECTED_TUNING_CANDIDATES:
        error(
            errors,
            f"threshold_search_summary.csv rows={len(summary)} "
            f"!= {EXPECTED_TUNING_CANDIDATES}",
        )
    if duplicates:
        error(errors, f"threshold_search_summary.csv has {duplicates} duplicate identities")
    if missing:
        error(errors, f"Candidate summary missing identities ({len(missing)}): {missing[:3]}…")
    if extra:
        error(errors, f"Candidate summary extra identities ({len(extra)}): {extra[:3]}…")

    merged = aggregated.merge(
        summary,
        on=key_cols,
        how="inner",
        suffixes=("_recomputed", "_recorded"),
    )
    for field in metric_cols:
        left = pd.to_numeric(merged[f"{field}_recomputed"], errors="coerce")
        right = pd.to_numeric(merged[f"{field}_recorded"], errors="coerce")
        mismatches = (~np.isclose(left, right, rtol=0.0, atol=1e-9, equal_nan=False))
        count = int(mismatches.sum())
        mismatch_count += count
        if count:
            error(errors, f"Candidate summary {field} mismatches: {count}")

    passed = (
        len(summary) == EXPECTED_TUNING_CANDIDATES
        and len(aggregated) == EXPECTED_TUNING_CANDIDATES
        and not missing
        and not extra
        and duplicates == 0
        and mismatch_count == 0
    )
    evidence.update({
        "actual_count": int(len(summary)),
        "recomputed_count": int(len(aggregated)),
        "expected_set_sha256": compute_set_sha(expected_ids),
        "actual_set_sha256": compute_set_sha(actual_ids),
        "missing_identities": missing,
        "extra_identities": extra,
        "duplicate_identity_count": duplicates,
        "metric_mismatch_count": mismatch_count,
        "verdict": "PASS" if passed else "FAIL",
    })
    return evidence


def reconstruct_evaluation_identity_set(
    episode_path: Path,
    scenario_ids_by_bank: Dict[str, List[str]],
    reset_seeds: List[int],
    errors: List[str],
) -> Tuple[Set[str], Dict[str, Any]]:
    """D. Evaluation identity set (exact construction + comparison).

    Each (policy, split, K, cost_regime_id) maps to the EXACT
    (split, K, cost_regime_id) bank's scenario IDs — banks are NOT assumed
    to share scenario IDs across K/regime.
    """
    actual_identities: Set[str] = set()
    actual_row_count = 0
    try:
        df = pd.read_parquet(episode_path)
        actual_row_count = len(df)
        if df.empty:
            error(errors, "episode_results.parquet is empty")
        else:
            for _, row in df.iterrows():
                actual_identities.add(
                    evaluation_identity(
                        str(row["policy_family"]),
                        str(row["split"]),
                        int(row["maintenance_capacity"]),
                        str(row["cost_regime_id"]),
                        str(row["scenario_id"]),
                        int(row["reset_seed"]),
                    )
                )
    except Exception as e:
        error(errors, f"Failed to read episode_results.parquet: {e}")

    actual_unique_count = len(actual_identities)

    expected_identities: Set[str] = set()
    for policy in EVAL_POLICIES:
        for split in EVAL_SPLITS:
            for k in FORMAL_K_VALUES:
                for regime in FORMAL_COST_REGIMES:
                    bank_key = bank_identity_key(split, k, regime)
                    scenario_ids = scenario_ids_by_bank.get(bank_key, [])
                    if not scenario_ids:
                        error(errors, f"No scenario IDs for bank {bank_key}; cannot build expected evaluation set")
                        continue
                    for scenario_id in scenario_ids:
                        for seed in reset_seeds:
                            expected_identities.add(
                                evaluation_identity(
                                    policy, split, k, regime,
                                    scenario_id, int(seed),
                                )
                            )

    expected_count = len(expected_identities)
    missing = expected_identities - actual_identities
    extra = actual_identities - expected_identities
    expected_sha = compute_set_sha(expected_identities)
    actual_sha = compute_set_sha(actual_identities)

    # FORMAL-CONTRACT COUNT GATE: the bank universe must yield exactly 2400
    # evaluation identities. A smaller self-consistent universe (e.g. 4
    # scenarios/bank → 1920 rows) is internally consistent but violates the
    # frozen formal contract — the recompute rejects it regardless.
    formal_count_match = (expected_count == EXPECTED_EVALUATION_EPISODES)
    if not formal_count_match:
        error(errors, (
            f"Evaluation formal-contract count: bank-derived expected={expected_count} "
            f"!= frozen formal contract {EXPECTED_EVALUATION_EPISODES} (wrong scenario "
            f"count per bank — a self-consistent smaller universe is still rejected)"
        ))

    if actual_unique_count != expected_count:
        error(errors, f"Evaluation identity unique count: expected {expected_count}, got {actual_unique_count}")
    if actual_row_count != expected_count:
        error(errors, f"Evaluation row count: expected {expected_count}, got {actual_row_count}")
    if missing:
        error(errors, f"Evaluation identities missing ({len(missing)}): {sorted(missing)[:5]}…")
    if extra:
        error(errors, f"Evaluation identities extra ({len(extra)}): {sorted(extra)[:3]}…")
    if expected_sha != actual_sha:
        error(errors, f"Evaluation set SHA mismatch: expected={expected_sha[:12]}… actual={actual_sha[:12]}…")
    else:
        ok(errors, f"Evaluation identities: {actual_unique_count} == {expected_count}")

    duplicate_identities: List[str] = []
    if actual_row_count != actual_unique_count:
        seen_local: Set[str] = set()
        try:
            df_dup = pd.read_parquet(episode_path)
            for _, row in df_dup.iterrows():
                ident = evaluation_identity(
                    str(row["policy_family"]), str(row["split"]),
                    int(row["maintenance_capacity"]), str(row["cost_regime_id"]),
                    str(row["scenario_id"]), int(row["reset_seed"]),
                )
                if ident in seen_local:
                    duplicate_identities.append(ident)
                else:
                    seen_local.add(ident)
        except Exception:
            pass
        error(errors, f"Evaluation duplicate identities ({len(duplicate_identities)})")

    debug = {
        "expected_count": expected_count,
        "actual_row_count": actual_row_count,
        "actual_unique_count": actual_unique_count,
        "formal_contract_count": EXPECTED_EVALUATION_EPISODES,
        "formal_contract_count_match": formal_count_match,
        "expected_set_sha256": expected_sha,
        "actual_set_sha256": actual_sha,
        "missing_identities": sorted(missing),
        "extra_identities": sorted(extra),
        "duplicate_identities": sorted(set(duplicate_identities)),
        "verdict": "PASS" if (formal_count_match
                              and not missing and not extra and not duplicate_identities
                              and actual_row_count == expected_count
                              and actual_unique_count == expected_count) else "FAIL",
    }
    return actual_identities, debug


def validate_scientific_reconciliation(
    episode_path: Path,
    selected_path: Path,
    summary_path: Path,
    scenario_provenance_path: Path,
    errors: List[str],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """E. Scientific reconciliation → returns per-dimension evidences:

    threshold_use_evidence, non_threshold_policy_evidence,
    reward_cost_evidence, cost_decomposition_evidence,
    summary_recomputation_evidence (with scenario-provenance check folded
    into a separate evidence dict returned alongside).
    """
    try:
        episodes = pd.read_parquet(episode_path)
    except Exception as e:
        errors.append(f"Cannot read episode_results.parquet: {e}")
        return ({}, {}, {}, {}, {}, {})
    try:
        with open(selected_path, "r") as f:
            selected = json.load(f)
        selected = {k: v for k, v in selected.items() if k != "_meta"}
    except Exception as e:
        errors.append(f"Cannot read selected_thresholds.json: {e}")
        return ({}, {}, {}, {}, {}, {})
    try:
        summary = pd.read_csv(summary_path)
    except Exception as e:
        errors.append(f"Cannot read summary_by_policy.csv: {e}")
        return ({}, {}, {}, {}, {}, {})
    try:
        with open(scenario_provenance_path, "r") as f:
            prov = json.load(f)
        prov_banks = prov.get("scenario_banks", [])
    except Exception as e:
        errors.append(f"Cannot read scenario_bank_provenance.json: {e}")
        prov_banks = []

    local_errs: List[str] = []

    # 1. No rl_test
    rl_test_evidence: Dict[str, Any] = {"rl_test_rows": 0, "verdict": "PASS"}
    if "split" in episodes.columns:
        rl_test = episodes[episodes["split"] == "rl_test"]
        rl_test_evidence["rl_test_rows"] = int(len(rl_test))
        if len(rl_test) > 0:
            error(local_errs, f"Found {len(rl_test)} rl_test episode rows (forbidden)")
            rl_test_evidence["verdict"] = "FAIL"

    # 2. Threshold-use equality
    threshold_use_records: List[Dict[str, Any]] = []
    for policy in ["age_threshold", "predicted_rul_threshold", "greedy_predicted_rul", "oracle_threshold"]:
        if policy not in episodes["policy_family"].values:
            continue
        policy_rows = episodes[episodes["policy_family"] == policy]
        if "threshold" not in policy_rows.columns:
            error(local_errs, f"{policy}: missing threshold column")
            continue
        for (k, regime), group in policy_rows.groupby(["maintenance_capacity", "cost_regime_id"]):
            key = f"{policy}_k{int(k)}_{regime}"
            if key not in selected:
                error(local_errs, f"{key}: missing in selected_thresholds")
                continue
            expected = selected[key].get("threshold")
            actual = group["threshold"].dropna().unique()
            for at in actual:
                v = "PASS" if at == expected else "FAIL"
                threshold_use_records.append({
                    "key": key, "expected_threshold": expected,
                    "actual_threshold": int(at) if at is not None and not (isinstance(at, float) and pd.isna(at)) else None,
                    "verdict": v,
                })
                if at != expected:
                    error(local_errs, f"{key}: evaluation used {at}, selected was {expected}")
    threshold_use_evidence = {
        "records": threshold_use_records,
        "verdict": "PASS" if all(r["verdict"] == "PASS" for r in threshold_use_records) else "FAIL",
    }

    # 3. corrective_only / random_feasible have no threshold
    non_threshold_records: List[Dict[str, Any]] = []
    for policy in ["corrective_only", "random_feasible"]:
        if policy in episodes["policy_family"].values:
            policy_rows = episodes[episodes["policy_family"] == policy]
            if "threshold" in policy_rows.columns:
                non_null = int(policy_rows["threshold"].notna().sum())
                v = "PASS" if non_null == 0 else "FAIL"
                non_policy_records = {"policy": policy, "non_null_threshold_rows": non_null, "verdict": v}
                non_threshold_records.append(non_policy_records)
                if non_null > 0:
                    error(local_errs, f"{policy}: {non_null} rows have non-None threshold")
    non_threshold_policy_evidence = {
        "records": non_threshold_records,
        "verdict": "PASS" if all(r["verdict"] == "PASS" for r in non_threshold_records) else "FAIL",
    }

    # 4. Reward = -total_cost  (VIOLATION-ROW COUNTING, not residual summation)
    reward_cost_evidence: Dict[str, Any] = {
        "checked_rows": 0, "violation_count": 0, "max_abs_residual": 0.0,
        "sample_violating_identities": [], "verdict": "PASS",
    }
    if "episode_return" in episodes.columns and "total_cost" in episodes.columns:
        diff = (episodes["episode_return"] + episodes["total_cost"]).abs()
        checked_rows = int(len(diff))
        # Per-row boolean: True where |return + total_cost| > 1e-6.
        # Count the violating ROWS, not the residual values. A handful of rows
        # each individually > 1e-6 must FAIL even if their residual values
        # cancel/sum to a small number.
        violating = diff > 1e-6
        violation_count = int(violating.sum())
        max_abs_residual = float(diff.max()) if checked_rows else 0.0

        # Sample the identities of up to 10 violating rows for evidence.
        sample_identities: List[str] = []
        if violation_count > 0:
            bad_idx = diff.index[violating].tolist()
            id_cols_present = [c for c in (
                "policy_family", "split", "maintenance_capacity",
                "cost_regime_id", "scenario_id", "reset_seed",
            ) if c in episodes.columns]
            for idx in bad_idx[:10]:
                row = episodes.loc[idx]
                if id_cols_present:
                    parts = [f"{c}={row[c]}" for c in id_cols_present]
                    sample_identities.append("|".join(parts))
                else:
                    sample_identities.append(f"row={idx}")

        reward_cost_evidence.update({
            "checked_rows": checked_rows,
            "violation_count": violation_count,
            "max_abs_residual": max_abs_residual,
            "sample_violating_identities": sample_identities,
        })
        if violation_count > 0:
            error(local_errs, f"Reward ≠ -total_cost for {violation_count} episodes")
            reward_cost_evidence["verdict"] = "FAIL"

    # 5. Cost decomposition: total = preventive + failure + wasted_life
    cost_decomposition_evidence: Dict[str, Any] = {"max_abs_residual": 0.0, "verdict": "PASS"}
    req = ["total_cost", "preventive_cost", "failure_cost", "wasted_life_cost"]
    if all(c in episodes.columns for c in req):
        reconstructed = episodes["preventive_cost"] + episodes["failure_cost"] + episodes["wasted_life_cost"]
        diff = (episodes["total_cost"] - reconstructed).abs()
        max_res = float(diff.max())
        cost_decomposition_evidence["max_abs_residual"] = max_res
        if max_res > 1e-6:
            error(local_errs, "Cost decomposition mismatch")
            cost_decomposition_evidence["verdict"] = "FAIL"

    # 6. Summary metrics recompute from episode rows
    summary_records: List[Dict[str, Any]] = []
    group_cols = ["policy_id", "split", "maintenance_capacity", "cost_regime_id"]
    for group_keys, group_df in episodes.groupby(group_cols):
        if len(group_df) == 0:
            continue
        mean_cost = float(group_df["total_cost"].mean())
        mask = True
        for i, col in enumerate(group_cols):
            mask &= (summary[col] == group_keys[i])
        matching = summary[mask]
        if len(matching) == 0:
            error(local_errs, f"No summary row for group {dict(zip(group_cols, group_keys))}")
            summary_records.append({"group": dict(zip(group_cols, group_keys)), "verdict": "FAIL", "reason": "no summary row"})
        else:
            sum_mean = float(matching["mean"].iloc[0])
            v = "PASS" if abs(mean_cost - sum_mean) <= 1e-6 else "FAIL"
            summary_records.append({
                "group": dict(zip(group_cols, group_keys)),
                "episode_mean": mean_cost, "summary_mean": sum_mean, "verdict": v,
            })
            if v == "FAIL":
                error(local_errs, f"Summary mean mismatch for {dict(zip(group_cols, group_keys))}: "
                                 f"episode={mean_cost:.6f} summary={sum_mean:.6f}")
    summary_recomputation_evidence = {
        "records": summary_records,
        "verdict": "PASS" if all(r["verdict"] == "PASS" for r in summary_records) else "FAIL",
    }

    # 7. Scenario provenance: verify expected banks + SHA consistency
    expected_bank_keys = set()
    for split in EVAL_SPLITS:
        for k in FORMAL_K_VALUES:
            for regime in FORMAL_COST_REGIMES:
                expected_bank_keys.add(bank_identity_key(split, k, regime))

    prov_records: List[Dict[str, Any]] = []
    actual_bank_keys = set()
    for bank in prov_banks:
        try:
            split = str(bank.get("split", bank.get("derived_split", "unknown")))
            k = _bank_k(bank)
            regime = _bank_regime(bank)
        except (TypeError, ValueError):
            prov_records.append({"verdict": "FAIL", "reason": "malformed bank record"})
            continue
        key = bank_identity_key(split, k, regime)
        actual_bank_keys.add(key)
        recomputed = None
        recorded = bank.get("sorted_scenario_ids_sha256")
        source_path = bank.get("source_path")
        if source_path:
            try:
                with open(Path(source_path), "r") as source_file:
                    source_payload = json.load(source_file)
                raw_ids = sorted(
                    str(s["scenario_id"])
                    for s in source_payload.get("scenarios", [])
                    if s.get("scenario_id")
                )
                recomputed = hashlib.sha256(
                    "\n".join(raw_ids).encode("utf-8")
                ).hexdigest()
            except Exception as exc:
                error(local_errs, f"Cannot recompute raw scenario IDs for {key}: {exc}")
        if recorded is None:
            error(local_errs, "scenario_bank record missing sorted_scenario_ids_sha256")
        elif not _HEX64_RE.match(str(recorded)):
            error(local_errs, f"scenario_bank record has invalid SHA format: {str(recorded)[:16]}…")
        elif recomputed is not None and recomputed != recorded:
            error(local_errs, f"scenario_bank SHA mismatch ({key}): recomputed={recomputed[:12]}… recorded={str(recorded)[:12]}…")
            prov_records.append({
                "bank_key": key, "verdict": "FAIL",
                "recomputed_sorted_ids_sha256": recomputed,
                "recorded_sorted_ids_sha256": recorded,
            })
            continue
        bank_sha = bank.get("bank_sha256")
        if bank_sha is not None and not _HEX64_RE.match(str(bank_sha)):
            error(local_errs, f"scenario_bank record bank_sha256 invalid format ({key}): {str(bank_sha)[:16]}…")
        prov_records.append({
            "bank_key": key, "verdict": "PASS",
            "recomputed_sorted_ids_sha256": recomputed,
            "recorded_sorted_ids_sha256": recorded,
        })

    missing_b = sorted(expected_bank_keys - actual_bank_keys)
    extra_b = sorted(actual_bank_keys - expected_bank_keys)
    if missing_b:
        error(local_errs, f"Scenario provenance missing banks: {missing_b}")
    if extra_b:
        error(local_errs, f"Scenario provenance extra banks: {extra_b}")
    else:
        ok(local_errs, "Scenario provenance matches expected banks")

    scenario_provenance_reconciliation_evidence = {
        "records": prov_records,
        "expected_count": len(expected_bank_keys),
        "actual_unique_count": len(actual_bank_keys),
        "expected_set_sha256": compute_set_sha(expected_bank_keys),
        "actual_set_sha256": compute_set_sha(actual_bank_keys),
        "missing_bank_identities": missing_b,
        "extra_bank_identities": extra_b,
        "verdict": "PASS" if (not missing_b and not extra_b
                              and all(r["verdict"] == "PASS" for r in prov_records)) else "FAIL",
    }

    errors.extend(local_errs)
    return (threshold_use_evidence, non_threshold_policy_evidence,
            reward_cost_evidence, cost_decomposition_evidence,
            summary_recomputation_evidence, scenario_provenance_reconciliation_evidence)


# =============================================================================
# ORACLE TERMINOLOGY CHECK
# =============================================================================

def check_oracle_terminology(
    output_dir: Path,
    episode_path: Path,
    summary_path: Path,
    selected_path: Path,
) -> Dict[str, Any]:
    """Scan all generated textual/JSON/CSV evidence fields for Oracle naming.

    Reject forbidden phrases ('optimal oracle', 'optimal policy', 'upper
    bound' / 'upper-bound'). Require the diagnostic label (or an accepted
    equivalent) wherever the evidence references the Oracle policy role.
    """
    scanned_files: List[Dict[str, Any]] = []
    scanned_files_text: List[Dict[str, Any]] = []
    forbidden_matches: List[Dict[str, Any]] = []
    required_matches: List[Dict[str, Any]] = []

    # Files to scan: structured evidence the recompute itself emits plus the
    # formal artifact text/JSON/CSV that survives in the output dir.
    candidate_files = [
        ("episode_results.parquet", ["policy_family", "policy_id"]),
        ("summary_by_policy.csv", ["policy_id"]),
        ("selected_thresholds.json", None),
        ("scenario_bank_provenance.json", None),
        ("validation_report.json", None),
        ("formal_run_context.json", None),
    ]
    # Read each as text/bytes where feasible; skip binary-only parquet unless
    # it carries string columns we can scan (it does: policy_family/policy_id).
    for name, cols in candidate_files:
        fp = output_dir / name
        if not fp.exists():
            continue
        text_blob = ""
        try:
            if name.endswith(".parquet"):
                df = pd.read_parquet(fp)
                text_blob = " ".join(str(v) for v in df.to_dict(orient="records"))
                # include oracle_threshold / oracle policy family tokens explicitly
                for c in cols or []:
                    if c in df.columns:
                        text_blob += " " + " ".join(str(x) for x in df[c].unique())
            elif name.endswith(".csv"):
                text_blob = fp.read_text()
            else:
                text_blob = fp.read_text()
        except Exception:
            continue
        text_lower = text_blob.lower()
        forbidden_hits: List[str] = []
        for phrase in ORACLE_FORBIDDEN_PHRASES:
            if phrase in text_lower:
                forbidden_hits.append(phrase)
        required_present = any(label in text_lower for label in ORACLE_ACCEPTED_LABELS)
        oracle_referenced = "oracle" in text_lower
        scanned_files.append({
            "file": name,
            "forbidden_matches": forbidden_hits,
            "required_label_present": required_present,
            "oracle_referenced": oracle_referenced,
        })
        scanned_files_text.append({"file": name, "text": text_lower})
        for phrase in forbidden_hits:
            forbidden_matches.append({"file": name, "phrase": phrase})
        if required_present:
            required_matches.append({"file": name, "label": ORACLE_REQUIRED_LABEL})

    # The independent recomputation records ONE authoritative semantic-role
    # field for the Oracle policy. This field is recorded in the evidence so the
    # formal manifest can preserve it; it is NOT used to satisfy the
    # required-label check below (the spec forbids satisfying the required
    # semantic label "solely by inventing it inside the checker"). The
    # required-label satisfaction must come from the SCANNED evidence files
    # carrying a diagnostic label wherever they reference the Oracle role.
    oracle_semantic_role = ORACLE_REQUIRED_LABEL
    # The source of the diagnostic role is the ACTUAL generated artifact that
    # carries the required diagnostic label. The role must not be satisfied
    # solely by a checker-local constant; record the real path that produced a
    # required_label_match. When no scanned artifact carries the label, the
    # source remains None and the required-label check (if the Oracle role is
    # referenced anywhere) will FAIL below.
    oracle_role_source_path: Optional[str] = None
    if required_matches:
        # Prefer an authoritative formal artifact name over arbitrary scan
        # order: validation_report.json > formal_run_context.json >
        # scenario_bank_provenance.json > selected_thresholds.json >
        # summary_by_policy.csv > episode_results.parquet
        artifact_priority = (
            "validation_report.json",
            "formal_run_context.json",
            "scenario_bank_provenance.json",
            "selected_thresholds.json",
            "summary_by_policy.csv",
            "episode_results.parquet",
        )
        ranked = sorted(
            required_matches,
            key=lambda m: artifact_priority.index(m["file"])
            if m["file"] in artifact_priority else len(artifact_priority),
        )
        oracle_role_source_path = str(output_dir / ranked[0]["file"])

    verdict = "PASS"
    if forbidden_matches:
        verdict = "FAIL"
    # Determine whether any scanned file references the Oracle role at all.
    oracle_token_seen_in_any_file = any(
        sf.get("oracle_referenced") for sf in scanned_files
    )
    # The required diagnostic-label check fires only where the evidence
    # actually references the Oracle role; the checker's OWN emitted
    # oracle_semantic_role field does NOT satisfy it (the spec forbids
    # satisfying the required label "solely by inventing it inside the
    # checker"). The required label must appear in the scanned evidence.
    if oracle_token_seen_in_any_file:
        required_satisfied = bool(required_matches)
    else:
        # No scanned evidence speaks of the Oracle role — no label required.
        required_satisfied = True
    # The oracle_semantic_role_source MUST identify an actual generated
    # artifact carrying the diagnostic label whenever the Oracle role is
    # referenced anywhere. A checker-local constant alone does NOT satisfy the
    # contract: the role's provenance must be a real file path.
    if oracle_token_seen_in_any_file and oracle_role_source_path is None:
        required_satisfied = False
    if not required_satisfied:
        verdict = "FAIL"
    evidence = {
        "scanned_files": scanned_files,
        "forbidden_matches": forbidden_matches,
        "required_label_matches": required_matches,
        "oracle_semantic_role": oracle_semantic_role,
        "oracle_semantic_role_source": oracle_role_source_path,
        "required_label": ORACLE_REQUIRED_LABEL,
        "required_label_satisfied": required_satisfied,
        "oracle_role_referenced_in_scanned": oracle_token_seen_in_any_file,
        "verdict": verdict,
    }
    return evidence


def build_selected_threshold_file_verification(
    context_dict: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """Build the selected_threshold_file_verification AGGREGATE OBJECT.

    Contract A requires this section to be a typed object — never a bare
    boolean and never null. The object carries the actual file path, the
    expected SHA256 recorded in the sealed formal context, the actual SHA256
    recomputed from the on-disk file, the file-exists flag, the SHA-match
    flag, and an explicit aggregate verdict.

    Verdict is PASS only when the file exists AND the actual SHA256 equals the
    expected recorded SHA256. Any missing field, malformed SHA, missing file,
    or SHA mismatch yields a FAIL verdict.
    """
    sel_path_str = context_dict.get("selected_thresholds_path")
    sel_path = Path(sel_path_str) if sel_path_str else (output_dir / "selected_thresholds.json")
    expected_sha = context_dict.get("selected_thresholds_sha256")
    exists = sel_path.exists()
    actual_sha: Optional[str] = None
    sha_match = False
    if exists:
        try:
            actual_sha = compute_sha256(sel_path)
        except Exception:
            actual_sha = None
    if (
        isinstance(expected_sha, str)
        and actual_sha is not None
        and _HEX64_RE.match(expected_sha)
        and _HEX64_RE.match(actual_sha)
    ):
        sha_match = (actual_sha == expected_sha)
    verdict = "PASS" if (exists and sha_match) else "FAIL"
    return {
        "selected_thresholds_path": str(sel_path),
        "expected_sha256": expected_sha,
        "actual_sha256": actual_sha,
        "exists": bool(exists),
        "sha_match": bool(sha_match),
        "verdict": verdict,
    }


def build_deterministic_tie_break_evidence(
    tie_break_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the deterministic_tie_break_evidence AGGREGATE OBJECT.

    Contract A requires this section to be a typed object — never a bare
    list of per-record dicts and never null. The object wraps the per-bank
    tie-break records with checked_count, failed_count, and an explicit
    aggregate verdict. Any record-level FAIL forces aggregate FAIL; an
    inconsistent checked_count/failed_count is also a FAIL.
    """
    records = list(tie_break_records) if tie_break_records else []
    checked_count = len(records)
    failed_count = sum(
        1 for r in records if isinstance(r, dict) and r.get("verdict") == "FAIL"
    )
    verdict = "PASS" if (checked_count == 0 or failed_count == 0) else "FAIL"
    return {
        "records": records,
        "checked_count": checked_count,
        "failed_count": failed_count,
        "verdict": verdict,
    }


# =============================================================================
# MAIN
# =============================================================================

def main(output_dir: Path) -> int:
    output_dir = Path(output_dir)
    print("=" * 70)
    print(f"INDEPENDENT RECOMPUTATION: {output_dir}")
    print("=" * 70)

    all_errors: List[str] = []

    # Required files
    required_files = [
        "threshold_search_results.parquet",
        "threshold_search_summary.csv",
        "selected_thresholds.json",
        "episode_results.parquet",
        "summary_by_policy.csv",
        "scenario_bank_provenance.json",
        "validation_report.json",
        "formal_run_context.json",
        "resolved_config.json",
    ]
    missing_files: List[str] = []
    for rf in required_files:
        if not (output_dir / rf).exists():
            missing_files.append(rf)

    if missing_files:
        for e in missing_files:
            all_errors.append(f"Missing required file: {e}")
        # Still write a FAIL report with every structured section.
        report = _build_fail_report(output_dir, all_errors)
        with open(output_dir / "independent_recomputation.json", "w") as f:
            json.dump(report, f, indent=2)
        return 1

    threshold_results_path = output_dir / "threshold_search_results.parquet"
    candidate_summary_path = output_dir / "threshold_search_summary.csv"
    selected_path = output_dir / "selected_thresholds.json"
    episode_path = output_dir / "episode_results.parquet"
    summary_path = output_dir / "summary_by_policy.csv"
    scenario_prov_path = output_dir / "scenario_bank_provenance.json"
    formal_context_path = output_dir / "formal_run_context.json"

    # =========================================================================
    # STEP 0: Verify sealed formal context (no fallbacks)
    # =========================================================================
    print("\n[0a] Verifying sealed formal_run_context.json (strict, no fallback)...")
    context_dict, context_verification, ctx_errs = verify_sealed_formal_context(
        formal_context_path, output_dir,
    )
    all_errors.extend(ctx_errs)

    print("\n[0b] Verifying resolved_config.json matches the formal contract...")
    _, resolved_config_contract, rc_errs = verify_resolved_config_contract(
        context_dict, output_dir,
    )
    all_errors.extend(rc_errs)

    # reset_seeds: REQUIRED from sealed context. No fallback.
    reset_seeds = context_dict.get("reset_seeds")
    if reset_seeds is None:
        all_errors.append("formal_run_context.json: reset_seeds missing — no fallback permitted")
        reset_seeds = []
    else:
        reset_seeds = list(reset_seeds)

    # =========================================================================
    # STEP 1: Verify scenario bank sources independently (16-bank set + files)
    # =========================================================================
    print("\n[1] Verifying scenario-bank sources (16-bank set + per-file)...")
    _, scenario_ids_by_bank, bank_set_evidence, bank_file_evidence = \
        verify_scenario_bank_sources(formal_context_path, all_errors)

    # =========================================================================
    # A. Candidate identities (360-cell exact reconstruction)
    # =========================================================================
    print("\n[A] Reconstructing candidate identities (exact)...")
    _, candidate_evidence = reconstruct_candidate_identities(
        threshold_results_path, all_errors
    )
    candidate_summary_evidence = reconcile_candidate_summary(
        threshold_results_path, candidate_summary_path, all_errors
    )

    # =========================================================================
    # B. Tuning episodes (exact per-bank 6-tuple identity recomputation)
    # =========================================================================
    print("\n[B] Reconstructing tuning-episode identity set (exact, per-bank)...")
    _, tuning_evidence = reconstruct_tuning_episode_set(
        threshold_results_path, scenario_ids_by_bank, reset_seeds, all_errors,
    )

    # =========================================================================
    # C. Selected winners + deterministic tie-break
    # =========================================================================
    print("\n[C] Recomputing selected winners + tie-break...")
    _winners, tie_break_records, winner_summary = reconstruct_selected_winners(
        threshold_results_path, selected_path, all_errors,
    )

    # =========================================================================
    # D. Evaluation identities (2400-row exact reconstruction)
    # =========================================================================
    print("\n[D] Reconstructing evaluation identity set (exact, per-bank)...")
    _, eval_evidence = reconstruct_evaluation_identity_set(
        episode_path, scenario_ids_by_bank, reset_seeds, all_errors,
    )

    # =========================================================================
    # E. Scientific reconciliation
    # =========================================================================
    print("\n[E] Scientific reconciliation...")
    (thr_use_ev, non_thr_ev, reward_ev, cost_dec_ev,
     summary_ev, prov_rec_ev) = validate_scientific_reconciliation(
        episode_path, selected_path, summary_path, scenario_prov_path, all_errors,
    )

    # =========================================================================
    # F. Oracle terminology check
    # =========================================================================
    print("\n[F] Oracle terminology check...")
    oracle_term_evidence = check_oracle_terminology(
        output_dir, episode_path, summary_path, selected_path,
    )
    if oracle_term_evidence["verdict"] == "FAIL":
        all_errors.append(
            f"Oracle terminology contract violated: forbidden matches "
            f"{oracle_term_evidence['forbidden_matches']}"
        )

    # =========================================================================
    # Write report
    # =========================================================================
    verdict = "PASS" if len(all_errors) == 0 else "FAIL"
    # Build the contract-A AGGREGATE OBJECTS for the two sections that were
    # previously emitted as a bare boolean / bare list. These are required by
    # the spec to be typed objects with an explicit aggregate verdict.
    selected_threshold_file_verification = build_selected_threshold_file_verification(
        context_dict, output_dir,
    )
    if selected_threshold_file_verification["verdict"] != "PASS":
        all_errors.append(
            "selected_threshold_file_verification: verdict is "
            f"{selected_threshold_file_verification['verdict']} "
            f"(exists={selected_threshold_file_verification['exists']}, "
            f"sha_match={selected_threshold_file_verification['sha_match']})"
        )
        verdict = "FAIL"
    deterministic_tie_break_evidence = build_deterministic_tie_break_evidence(
        tie_break_records,
    )
    if deterministic_tie_break_evidence["verdict"] != "PASS":
        all_errors.append(
            "deterministic_tie_break_evidence: verdict is "
            f"{deterministic_tie_break_evidence['verdict']} "
            f"(failed_count={deterministic_tie_break_evidence['failed_count']})"
        )
        verdict = "FAIL"
    report = {
        "schema_version": "m3_independent_recompute_v2",
        "script_version": "3.0.0",
        "verdict": verdict,
        "executed_at": datetime.utcnow().isoformat(),
        "formal_run_id": context_dict.get("formal_run_id"),
        "implementation_commit": context_dict.get("implementation_commit"),
        "formal_run_context_verification": context_verification,
        "resolved_config_verification": resolved_config_contract,
        "selected_threshold_file_verification": selected_threshold_file_verification,
        "scenario_bank_set_evidence": bank_set_evidence,
        "scenario_bank_file_evidence": bank_file_evidence,
        "candidate_set_evidence": candidate_evidence,
        "candidate_summary_recomputation_evidence": candidate_summary_evidence,
        "tuning_set_evidence": tuning_evidence,
        "selected_winner_evidence": winner_summary,
        "deterministic_tie_break_evidence": deterministic_tie_break_evidence,
        "evaluation_set_evidence": eval_evidence,
        "threshold_use_evidence": thr_use_ev,
        "non_threshold_policy_evidence": non_thr_ev,
        "reward_cost_evidence": reward_ev,
        "cost_decomposition_evidence": cost_dec_ev,
        "summary_recomputation_evidence": summary_ev,
        "scenario_bank_provenance_reconciliation_evidence": prov_rec_ev,
        "oracle_terminology_evidence": oracle_term_evidence,
        "errors": all_errors,
    }
    with open(output_dir / "independent_recomputation.json", "w") as f:
        json.dump(_to_jsonable(report), f, indent=2)

    print("\n" + "=" * 70)
    if all_errors:
        print(f"✗ FAIL: {len(all_errors)} errors")
        for e in all_errors:
            print(f"  - {e}")
    else:
        print("✓ PASS: All reconstructions match")
    print("=" * 70)

    return 0 if verdict == "PASS" else 1


def _build_fail_report(output_dir: Path, all_errors: List[str]) -> Dict[str, Any]:
    """Minimal FAIL report when required files are missing — still carries
    every required top-level section (verdict FAIL)."""
    return {
        "schema_version": "m3_independent_recompute_v2",
        "script_version": "3.0.0",
        "verdict": "FAIL",
        "executed_at": datetime.utcnow().isoformat(),
        "formal_run_id": Path(output_dir).name,
        "implementation_commit": None,
        "formal_run_context_verification": {"verdict": "FAIL"},
        "resolved_config_verification": {"verdict": "FAIL"},
        "selected_threshold_file_verification": {
            "selected_thresholds_path": None,
            "expected_sha256": None,
            "actual_sha256": None,
            "exists": False,
            "sha_match": False,
            "verdict": "FAIL",
        },
        "scenario_bank_set_evidence": {"verdict": "FAIL"},
        "scenario_bank_file_evidence": [],
        "candidate_set_evidence": {"verdict": "FAIL"},
        "candidate_summary_recomputation_evidence": {"verdict": "FAIL"},
        "tuning_set_evidence": {"verdict": "FAIL"},
        "selected_winner_evidence": {"verdict": "FAIL"},
        "deterministic_tie_break_evidence": {
            "records": [],
            "checked_count": 0,
            "failed_count": 0,
            "verdict": "FAIL",
        },
        "evaluation_set_evidence": {"verdict": "FAIL"},
        "threshold_use_evidence": {"verdict": "FAIL"},
        "non_threshold_policy_evidence": {"verdict": "FAIL"},
        "reward_cost_evidence": {"verdict": "FAIL"},
        "cost_decomposition_evidence": {"verdict": "FAIL"},
        "summary_recomputation_evidence": {"verdict": "FAIL"},
        "oracle_terminology_evidence": {
            "scanned_files": [],
            "forbidden_matches": [],
            "required_label_matches": [],
            "oracle_semantic_role": ORACLE_REQUIRED_LABEL,
            "oracle_semantic_role_source": None,
            "required_label": ORACLE_REQUIRED_LABEL,
            "required_label_satisfied": False,
            "oracle_role_referenced_in_scanned": False,
            "verdict": "FAIL",
        },
        "errors": all_errors,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python independent_recompute_m3.py <output_dir>")
        sys.exit(1)
    sys.exit(main(Path(sys.argv[1])))
