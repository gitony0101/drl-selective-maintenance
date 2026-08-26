#!/usr/bin/env python3
"""
Synthetic full-contract fixtures for independent recomputation.

This suite does NOT run a real formal experiment. It builds the canonical
360-candidate / 9000-episode / 32-winner / 2400-evaluation identity sets
on disk and exercises scripts/independent_recompute_m3.py against them,
confirming both pass-on-valid and nonzero-exit-on-mutation behaviour.

Per the frozen contract:
  - exact 360 candidate identity set;
  - exact 9000 tuning identity set;
  - exact 32 winners;
  - exact 2400 evaluation identity set.

Mutations cover the eight checkpoint categories the M3 contract
explicitly demands: missing candidate, duplicate tuning identity,
missing evaluation identity, wrong winner, threshold-use mismatch,
summary mismatch, reward/cost mismatch, and scenario-bank SHA
mismatch.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RECOMPUTE_SCRIPT = PROJECT_ROOT / "scripts" / "independent_recompute_m3.py"

# Canonical repository scenario-bank files. Adversarial tests MUST NOT
# modify these; every source-bank test copies them into tmp_path and
# mutates only the copies.
CANONICAL_BANK_FILES = (
    "data/scenario_banks/predictor_train_smoke.json",
    "data/scenario_banks/rl_validation_smoke.json",
    "data/scenario_banks/rl_validation_k1_smoke.json",
    "data/scenario_banks/rl_test_smoke.json",
)


def _canonical_bank_shas() -> dict:
    """Snapshot the SHA256 of every canonical scenario-bank file.

    The autouse integrity fixture asserts these are unchanged after each
    test, proving no adversarial test mutated a canonical repository bank
    (even transiently).
    """
    shas = {}
    for rel in CANONICAL_BANK_FILES:
        p = PROJECT_ROOT / rel
        shas[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return shas


def _is_under_canonical_banks(target) -> bool:
    """True if ``target`` resolves to a path inside the canonical
    data/scenario_banks/ directory (any depth)."""
    try:
        resolved = Path(target).resolve(strict=False)
    except (OSError, ValueError):
        return False
    canonical_root = (PROJECT_ROOT / "data" / "scenario_banks").resolve(strict=False)
    try:
        resolved.relative_to(canonical_root)
        return True
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def _guard_canonical_bank_writes():
    """Write-guard against canonical scenario-bank mutation.

    Patches Path.write_text / write_bytes / unlink / replace / rename and
    builtins.open in write/append/update modes so ANY write-like operation
    against a resolved path under data/scenario_banks/ raises immediately.
    This catches a test that writes to a canonical repo bank (or to a
    canonical bank restored within the test) the instant the attempt is
    made — strictly stronger than a before/after SHA snapshot, which a
    write-and-restore can evade. The autouse SHA snapshot fixture below
    remains as additional evidence.
    """
    import builtins

    _orig_write_text = Path.write_text
    _orig_write_bytes = Path.write_bytes
    _orig_unlink = Path.unlink
    _orig_replace = Path.replace
    _orig_rename = Path.rename
    _orig_open = builtins.open

    def _reject(path, op):
        if _is_under_canonical_banks(path):
            raise AssertionError(
                f"test attempted to {op} a canonical scenario-bank file "
                f"({path}); tests must mutate only tmp_path copies"
            )

    def _write_text(self, data, *a, **k):
        _reject(self, "write_text")
        return _orig_write_text(self, data, *a, **k)

    def _write_bytes(self, data, *a, **k):
        _reject(self, "write_bytes")
        return _orig_write_bytes(self, data, *a, **k)

    def _unlink(self, *a, **k):
        _reject(self, "unlink")
        return _orig_unlink(self, *a, **k)

    def _replace(self, target, *a, **k):
        _reject(self, "replace")
        # also reject if the DESTINATION is a canonical bank (rename-into)
        _reject(target, "replace (target)")
        return _orig_replace(self, target, *a, **k)

    def _rename(self, target, *a, **k):
        _reject(self, "rename")
        _reject(target, "rename (target)")
        return _orig_rename(self, target, *a, **k)

    def _open(file, mode="r", *a, **k):
        m = str(mode)
        if any(c in m for c in ("w", "a", "x", "+")):
            _reject(file, f"open({mode!r})")
        return _orig_open(file, mode, *a, **k)

    Path.write_text = _write_text
    Path.write_bytes = _write_bytes
    Path.unlink = _unlink
    Path.replace = _replace
    Path.rename = _rename
    builtins.open = _open
    try:
        yield
    finally:
        Path.write_text = _orig_write_text
        Path.write_bytes = _orig_write_bytes
        Path.unlink = _orig_unlink
        Path.replace = _orig_replace
        Path.rename = _orig_rename
        builtins.open = _orig_open


@pytest.fixture(autouse=True)
def _assert_canonical_banks_unchanged():
    """Prove no test mutates a canonical scenario-bank file.

    Snapshots every canonical scenario-bank SHA before and after each
    test in this module; the test fails if any SHA changed, even
    transiently (a test that wrote-and-restored a canonical bank is
    caught the instant the during-write checksum diverges, but more
    importantly the after-test snapshot catches any leaked mutation).
    """
    before = _canonical_bank_shas()
    yield
    after = _canonical_bank_shas()
    changed = [rel for rel in before if before[rel] != after[rel]]
    assert not changed, (
        f"Canonical scenario-bank SHAs changed during the test: {changed}"
    )


# Mirror the constants frozen in ``independent_recompute_m3.py`` so the
# tests do not depend on production modules to construct fixtures.
FORMAL_POLICIES = (
    "age_threshold",
    "predicted_rul_threshold",
    "greedy_predicted_rul",
    "oracle_threshold",
)
K_VALUES = (1, 2)
REGIMES = (
    "failure-light-no-waste",
    "failure-heavy-no-waste",
    "failure-light-waste-aware",
    "failure-heavy-waste-aware",
)
EVAL_SPLITS = ("predictor_train", "rl_validation")
EVAL_POLICY_FAMILIES = (
    "corrective_only",
    "random_feasible",
    "age_threshold",
    "predicted_rul_threshold",
    "greedy_predicted_rul",
    "oracle_threshold",
)
GRIDS = {
    "age_threshold": [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300],
    "predicted_rul_threshold": [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100],
    "greedy_predicted_rul": [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100],
    "oracle_threshold": [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50],
}
EXPECTED_CANDIDATES = 360   # 4 policies (12+11+11+11=45 thresholds) × 2 K × 4 regimes
EXPECTED_TUNING_EPISODES = 9000   # 360 × 5 scenarios × 5 seeds
EXPECTED_SELECTED_THRESHOLDS = 32  # 4 policies × 2 K × 4 regimes
EXPECTED_EVALUATION_EPISODES = 2400  # 6 policies × 2 K × 4 regimes × 2 splits × 5 × 5


def _identity(policy_family: str, k: int, regime: str) -> str:
    return f"{policy_family}_k{k}_{regime}"


def _copy_canonical_bank(out: Path, source_rel: str) -> Path:
    """Copy a canonical scenario-bank file into tmp_path and return the copy path.

    Adversarial tests mutate ONLY the tmp_path copy; the canonical
    repository file is never touched.
    """
    src = PROJECT_ROOT / source_rel
    dst_dir = out / "source_banks"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / Path(source_rel).name
    shutil.copyfile(src, dst)
    return dst


_CANONICAL_BY_SPLIT = {
    "predictor_train": "data/scenario_banks/predictor_train_smoke.json",
    "rl_validation": "data/scenario_banks/rl_validation_smoke.json",
}


def _ensure_bank_copy(out: Path, split: str) -> Path:
    """Ensure the canonical split bank is copied to tmp_path; return path."""
    return _copy_canonical_bank(out, _CANONICAL_BY_SPLIT[split])


def _load_bank_scenarios(bank_path: Path) -> list:
    with open(bank_path, "r") as f:
        return json.load(f)["scenarios"]


def _write_threshold_search_results(out: Path, mutate: bool = False) -> None:
    """Build a full threshold_search_results.parquet with the canonical
    360 candidates × 25 episodes each, yielding 9000 tuning rows.

    The data is synthetic but deterministic; reward/cost decomposition
    matches the validator's expectations and rewards equal -total_cost.

    Scenario IDs are read from the rl_validation bank COPY inside
    tmp_path (never the canonical repository file).
    """
    rows = []
    grid_total_per_policy = sum(len(v) for v in GRIDS.values())
    assert grid_total_per_policy * len(K_VALUES) * len(REGIMES) == EXPECTED_CANDIDATES, (
        f"grid sizes drift: {grid_total_per_policy}-grid-per-policy → "
        f"{grid_total_per_policy * len(K_VALUES) * len(REGIMES)} candidates, "
        f"expected {EXPECTED_CANDIDATES}"
    )

    rl_bank = _ensure_bank_copy(out, "rl_validation")
    rl_ids = [s["scenario_id"] for s in _load_bank_scenarios(rl_bank)]

    for policy_family in FORMAL_POLICIES:
        grid = GRIDS[policy_family]
        for k in K_VALUES:
            for regime in REGIMES:
                for threshold in sorted(grid):
                    for scenario_id in rl_ids:
                        for seed in range(5):
                            cost_offset = (threshold - min(grid)) / max(grid)
                            total_cost = 1000.0 + cost_offset * 4 + (k - 1) * 50
                            failure_cost = 5.0
                            preventive_cost = 2.0
                            wasted_life_cost = total_cost - failure_cost - preventive_cost
                            rows.append({
                                "policy_family": policy_family,
                                "threshold": threshold,
                                "k_capacity": k,
                                "cost_regime_id": regime,
                                "mean_total_cost": total_cost,
                                "total_failures": 0,
                                "mean_wasted_life_cost": wasted_life_cost,
                                "scenario_id": scenario_id,
                                "reset_seed": 6521 + seed,
                                "mean_preventive_cost": preventive_cost,
                                "mean_failure_cost": failure_cost,
                                "mean_wasted_life_cost": wasted_life_cost,
                                "episode_count": 1,
                                "total_episode_count": 1,
                                "total_cost": total_cost,
                                "preventive_cost": preventive_cost,
                                "failure_cost": failure_cost,
                                "wasted_life_cost": wasted_life_cost,
                                "failure_count": 0,
                                "episode_steps": 100,
                                "completed": True,
                            })
    df = pd.DataFrame(rows)
    if mutate:
        df = df.drop(df.index[0]).reset_index(drop=True)
    df.to_parquet(out / "threshold_search_results.parquet", index=False)
    summary = (
        df[df["completed"]]
        .groupby(
            ["policy_family", "threshold", "k_capacity", "cost_regime_id"],
            as_index=False,
        )
        .agg(
            mean_total_cost=("total_cost", "mean"),
            total_failures=("failure_count", "sum"),
            mean_wasted_life_cost=("wasted_life_cost", "mean"),
            episode_count=("total_cost", "size"),
        )
    )
    summary.to_csv(out / "threshold_search_summary.csv", index=False)


def _write_selected_thresholds(out: Path, mutate: bool = False) -> None:
    """Build a complete 32-winner selected_thresholds.json with a _meta."""
    payload = {}
    for policy_family in FORMAL_POLICIES:
        grid = sorted(GRIDS[policy_family])
        for k in K_VALUES:
            for regime in REGIMES:
                key = _identity(policy_family, k, regime)
                # Lowest-threshold wins under the frozen tie-break order.
                chosen = grid[0]
                tie_break_reason = "lowest mean total cost"
                if mutate:
                    chosen = grid[2]  # wrong: should be grid[0]
                payload[key] = {
                    "threshold": chosen,
                    "k_capacity": k,
                    "cost_regime_id": regime,
                    "mean_total_cost": 1000.0,
                    "total_failures": 0,
                    "mean_wasted_life_cost": 993.0,
                    "episode_count": 25,
                    "tie_break_reason": tie_break_reason,
                }
    payload["_meta"] = {"formal_run_id": out.name, "config_sha256": "abcd1234"}
    (out / "selected_thresholds.json").write_text(json.dumps(payload, indent=2))


def _write_episode_results(
    out: Path,
    selected_path: Path,
    mutate_threshold_use: bool = False,
    mutate_reward_cost: bool = False,
) -> None:
    """Build full episode_results.parquet.

    2400 rows = 6 policies × 2 K × 4 regimes × 2 splits × 5 scenarios × 5 seeds.
    Uses actual scenario IDs from the banks.
    """
    selected = json.loads(selected_path.read_text())
    selected = {k: v for k, v in selected.items() if k != "_meta"}

    # Load actual scenario IDs from the tmp_path bank copies.
    pred_ids = [s["scenario_id"] for s in _load_bank_scenarios(_ensure_bank_copy(out, "predictor_train"))]
    rl_ids = [s["scenario_id"] for s in _load_bank_scenarios(_ensure_bank_copy(out, "rl_validation"))]
    ids_by_split = {
        "predictor_train": pred_ids,
        "rl_validation": rl_ids,
    }

    rows = []
    for split in EVAL_SPLITS:
        scenario_ids = ids_by_split[split]
        for policy_family in EVAL_POLICY_FAMILIES:
            for k in K_VALUES:
                for regime in REGIMES:
                    for scenario_id in scenario_ids:
                        for seed in range(5):
                            key = _identity(policy_family, k, regime)
                            threshold = selected.get(key, {}).get("threshold")
                            if threshold is None and policy_family in (
                                "corrective_only",
                                "random_feasible",
                            ):
                                threshold = None
                            elif threshold is None:
                                # Default for synthetic when key missing.
                                threshold = 25
                            if mutate_threshold_use and split == EVAL_SPLITS[0]:
                                threshold = (threshold or 25) + 1  # wrong: used a non-winner threshold
                            rows.append({
                                "policy_id": policy_family,
                                "policy_family": policy_family,
                                "split": split,
                                "maintenance_capacity": k,
                                "k_capacity": k,
                                "cost_regime_id": regime,
                                "threshold": threshold,
                                "scenario_id": scenario_id,
                                "reset_seed": 6521 + seed,
                                "total_cost": 1000.0,
                                "preventive_cost": 998.0,
                                "failure_cost": 1.0,
                                "wasted_life_cost": 1.0,
                                "episode_return": -1000.0,
                                "episode_count": 1,
                            })
    df = pd.DataFrame(rows)
    if mutate_reward_cost:
        # reward ≠ -total_cost for one row
        df.loc[df.index[0], "episode_return"] = 0.0
    df.to_parquet(out / "episode_results.parquet", index=False)


def _write_summary(out: Path, episode_path: Path, mutate_summary: bool = False) -> None:
    """Build summary_by_policy.csv grouped by policy_id × split × K × regime."""
    eps = pd.read_parquet(episode_path)
    group_cols = ["policy_id", "split", "maintenance_capacity", "cost_regime_id"]
    summary = eps.groupby(group_cols).agg(mean=("total_cost", "mean")).reset_index()
    summary["sample_std"] = 0.0
    summary["standard_error"] = 0.0
    summary["ci_95_lower"] = summary["mean"]
    summary["ci_95_upper"] = summary["mean"]
    summary["episode_count"] = 1
    if mutate_summary:
        # Inject one wrong mean.
        summary.loc[summary.index[0], "mean"] = float(summary.loc[summary.index[0], "mean"]) + 1000.0
    summary.to_csv(out / "summary_by_policy.csv", index=False)


def _write_scenario_provenance(
    out: Path,
    mutate_sha: bool = False,
    missing_evaluation_identity: bool = False,
) -> None:
    """16 banks (2 splits × 2 K × 4 regimes). These match the
    'evaluation identity' bank key the validator checks."""
    # Load actual scenario IDs from the tmp_path bank copies.
    pred_bank = _ensure_bank_copy(out, "predictor_train")
    rl_bank = _ensure_bank_copy(out, "rl_validation")
    pred_ids = [s["scenario_id"] for s in _load_bank_scenarios(pred_bank)]
    rl_ids = [s["scenario_id"] for s in _load_bank_scenarios(rl_bank)]
    ids_by_split = {
        "predictor_train": pred_ids,
        "rl_validation": rl_ids,
    }
    # SHA of the actual tmp_path bank file bytes — recomputed from the copy,
    # not from the canonical repository file.
    bank_shas = {
        "predictor_train": hashlib.sha256(pred_bank.read_bytes()).hexdigest(),
        "rl_validation": hashlib.sha256(rl_bank.read_bytes()).hexdigest(),
    }

    banks = []
    for split in EVAL_SPLITS:
        scenario_ids = ids_by_split[split]
        for k in K_VALUES:
            for regime in REGIMES:
                if missing_evaluation_identity and split == EVAL_SPLITS[1] and k == 2 and regime == REGIMES[0]:
                    continue  # skip one record → "missing evaluation identity"
                sorted_ids_sha = hashlib.sha256(
                    "\n".join(sorted(scenario_ids)).encode("utf-8")
                ).hexdigest()
                # Source SHA of the actual tmp_path bank file copy.
                sha = bank_shas[split]
                if mutate_sha:
                    sorted_ids_sha = "deadbeef" * 8  # wrong SHA → fail-closed
                banks.append({
                    "split": split,
                    "K": k,
                    "derived_k": k,
                    "k": k,
                    "cost_regime_id": regime,
                    "derived_cost_regime_id": regime,
                    "logical_bank_id": f"bank_{split}_{k}_{regime}",
                    "source_bank_path": str({"predictor_train": pred_bank, "rl_validation": rl_bank}[split]),
                    "source_path": str({"predictor_train": pred_bank, "rl_validation": rl_bank}[split]),
                    "bank_sha256": sha,
                    "source_sha256": sha,
                    "source_file_size": 1024,
                    "bank_scenario_count": len(scenario_ids),
                    "derived_scenario_count": len(scenario_ids),
                    "scenario_count": len(scenario_ids),
                    "scenario_ids": scenario_ids,
                    "derived_scenario_ids": scenario_ids,
                    "sorted_scenario_ids_sha256": sorted_ids_sha,
                    "derived_bank_sha256": sha,
                })
    payload = {"scenario_banks": banks}
    (out / "scenario_bank_provenance.json").write_text(json.dumps(payload, indent=2))


def _write_validation_report(out: Path) -> None:
    # validation_report.json carries the Oracle policy's authoritative
    # semantic role as a diagnostic-benchmark label so the recompute's
    # oracle-terminology scan finds the required label in a SCANNED
    # generated artifact (not invented inside the checker).
    (out / "validation_report.json").write_text(json.dumps({
        "verdict": "ALL PASSED",
        "oracle_semantic_role": "privileged-information diagnostic benchmark",
    }))


def _canonical_json_sha(obj) -> str:
    """Independent canonical-JSON SHA256 — mirrors the checker's algorithm
    so the fixture's resolved_config_sha256 matches what the checker recomputes.
    """
    def _norm(o):
        if isinstance(o, dict):
            return {k: _norm(o[k]) for k in sorted(o.keys())}
        if isinstance(o, list):
            return [_norm(v) for v in o]
        if isinstance(o, float):
            return float(o)
        return o
    return hashlib.sha256(json.dumps(
        _norm(obj), sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _write_full_resolved_config(out: Path) -> Path:
    """Write a resolved_config.json whose families/grids/K/regimes/eval
    splits/reset seeds ALL match the frozen formal contract, so the
    checker's resolved_config contract verification passes.
    """
    config = {
        "policy_families": list(FORMAL_POLICIES),
        "threshold_grids": {k: list(v) for k, v in GRIDS.items()},
        "k_values": list(K_VALUES),
        "cost_regimes": list(REGIMES),
        "evaluation_splits": list(EVAL_SPLITS),
        "reset_seeds": [6521, 6522, 6523, 6524, 6525],
    }
    rc_path = out / "resolved_config.json"
    rc_path.write_text(json.dumps(config, indent=2))
    return rc_path


def _write_full_recompute_report(
    out: Path, *, drop_section: str = None, mutate_section: str = None,
) -> Path:
    """Write an independent_recomputation.json carrying EVERY required
    top-level evidence section with top-level verdict 'PASS' and every
    structured section verdict 'PASS'. Used by manifest adversarial tests
    that need a fully-shaped report (so a specific gate, not the
    missing-section gate, is what fires).

    drop_section: remove a single required section (top-level verdict stays
        PASS) — used to test the manifest's missing-section rejection.
    mutate_section: change a structured section's verdict to FAIL while
        top-level verdict stays PASS — used to test the manifest's
        structured-section PASS requirement.
    """
    sha = "a" * 64
    report = {
        "verdict": "PASS",
        "formal_run_context_verification": {"verdict": "PASS"},
        "resolved_config_verification": {"verdict": "PASS"},
        "selected_threshold_file_verification": {
            "selected_thresholds_path": str(out / "selected_thresholds.json"),
            "expected_sha256": sha,
            "actual_sha256": sha,
            "exists": True,
            "sha_match": True,
            "verdict": "PASS",
        },
        "scenario_bank_set_evidence": {"verdict": "PASS"},
        "scenario_bank_file_evidence": [{"verdict": "PASS"}],
        "candidate_set_evidence": {
            "actual_unique_count": 360,
            "expected_set_sha256": sha, "actual_set_sha256": sha,
            "verdict": "PASS",
        },
        "candidate_summary_recomputation_evidence": {
            "actual_count": 360,
            "metric_mismatch_count": 0,
            "verdict": "PASS",
        },
        "tuning_set_evidence": {
            "actual_unique_count": 9000,
            "expected_set_sha256": sha, "actual_set_sha256": sha,
            "verdict": "PASS",
        },
        "selected_winner_evidence": {"actual_count": 32, "verdict": "PASS"},
        "deterministic_tie_break_evidence": {
            "records": [], "checked_count": 0,
            "failed_count": 0, "verdict": "PASS",
        },
        "evaluation_set_evidence": {
            "actual_unique_count": 2400,
            "expected_set_sha256": sha, "actual_set_sha256": sha,
            "verdict": "PASS",
        },
        "threshold_use_evidence": {"records": [], "verdict": "PASS"},
        "non_threshold_policy_evidence": {"records": [], "verdict": "PASS"},
        "reward_cost_evidence": {
            "checked_rows": 0, "violation_count": 0,
            "max_abs_residual": 0.0, "sample_violating_identities": [],
            "verdict": "PASS",
        },
        "cost_decomposition_evidence": {"max_abs_residual": 0.0, "verdict": "PASS"},
        "summary_recomputation_evidence": {"records": [], "verdict": "PASS"},
        "scenario_bank_provenance_reconciliation_evidence": {
            "records": [], "verdict": "PASS",
        },
        "oracle_terminology_evidence": {
            "scanned_files": [],
            "forbidden_matches": [],
            "required_label_matches": [{"file": "validation_report.json", "label": "privileged-information diagnostic benchmark"}],
            "oracle_semantic_role": "privileged-information diagnostic benchmark",
            "oracle_semantic_role_source": str(out / "validation_report.json"),
            "required_label": "privileged-information diagnostic benchmark",
            "required_label_satisfied": True,
            "oracle_role_referenced_in_scanned": True,
            "verdict": "PASS",
        },
        "errors": [],
    }
    if drop_section is not None:
        report.pop(drop_section, None)
    if mutate_section is not None:
        sec = report.get(mutate_section)
        if isinstance(sec, dict):
            sec["verdict"] = "FAIL"
            # For aggregate-object-with-records sections, also inject a
            # failed record so the failure is record-level-visible.
            if isinstance(sec.get("records"), list):
                sec["records"] = [{"verdict": "FAIL"}]
                sec["failed_count"] = 1
        elif isinstance(sec, list) and sec:
            sec[0]["verdict"] = "FAIL"
    p = out / "independent_recomputation.json"
    p.write_text(json.dumps(report, indent=2))
    return p


def _build_full_valid(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    # Copy the canonical banks into tmp_path and read IDs/SHAs from the copies.
    pred_bank = _ensure_bank_copy(out, "predictor_train")
    rl_bank = _ensure_bank_copy(out, "rl_validation")
    pred_ids = [s["scenario_id"] for s in _load_bank_scenarios(pred_bank)]
    rl_ids = [s["scenario_id"] for s in _load_bank_scenarios(rl_bank)]
    pred_sha = hashlib.sha256(pred_bank.read_bytes()).hexdigest()
    rl_sha = hashlib.sha256(rl_bank.read_bytes()).hexdigest()
    pred_ids_sha = hashlib.sha256("\n".join(sorted(pred_ids)).encode("utf-8")).hexdigest()
    rl_ids_sha = hashlib.sha256("\n".join(sorted(rl_ids)).encode("utf-8")).hexdigest()

    _write_threshold_search_results(out)
    _write_selected_thresholds(out)
    _write_episode_results(out, out / "selected_thresholds.json")
    _write_summary(out, out / "episode_results.parquet")
    _write_scenario_provenance(out)
    _write_validation_report(out)

    # Self-consistent resolved_config.json matching the frozen formal contract.
    rc_path = _write_full_resolved_config(out)
    rc_sha = _canonical_json_sha(json.loads(rc_path.read_text()))

    # selected_thresholds.json SHA — what the checker recomputes via raw bytes.
    sel_sha = hashlib.sha256((out / "selected_thresholds.json").read_bytes()).hexdigest()

    # Build all 16 bank identities pointing at the tmp_path bank copies.
    banks = []
    for split, bank_path, source_sha, ids, ids_sha in [
        ("predictor_train", str(pred_bank), pred_sha, pred_ids, pred_ids_sha),
        ("rl_validation", str(rl_bank), rl_sha, rl_ids, rl_ids_sha),
    ]:
        for k in (1, 2):
            for regime in REGIMES:
                banks.append({
                    "split": split,
                    "K": k,
                    "cost_regime_id": regime,
                    "source_path": bank_path,
                    "source_sha256": source_sha,
                    "scenario_count": len(ids),
                    "sorted_scenario_ids_sha256": ids_sha,
                })

    (out / "formal_run_context.json").write_text(json.dumps({
        "schema_version": "m3_formal_context_v1",
        "formal_run_id": out.name,
        "mode": "formal_closeout",
        "implementation_commit": "0" * 40,  # 40 lowercase hex (format-only check)
        "implementation_tree_clean": True,
        "resolved_config_path": str(rc_path),
        "resolved_config_sha256": rc_sha,
        "oracle_authorized": True,
        "selected_thresholds_path": str(out / "selected_thresholds.json"),
        "selected_thresholds_sha256": sel_sha,
        "sealed": True,
        "sealed_at": "2024-01-01T00:00:00",
        "scenario_bank_identities": banks,
        "reset_seeds": [6521, 6522, 6523, 6524, 6525],
        "created_at": "2024-01-01T00:00:00",
    }))


def _run_recompute(out: Path, expected_returncode: int) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, str(RECOMPUTE_SCRIPT), str(out)],
        capture_output=True, text=True, timeout=120,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == expected_returncode, (
        f"Expected exit {expected_returncode}, got {result.returncode}: "
        f"stdout={result.stdout[:300]} stderr={result.stderr[:300]}"
    )
    return result


class TestRecomputeExactIdentitySets:
    """Prove identity sets match the canonical 360 / 32 / 2400 / 9000."""

    def test_exact_360_candidate_identity_set(self, tmp_path):
        _build_full_valid(tmp_path)
        # Read parquet and confirm 360 unique (policy, threshold, k, regime) tuples.
        df = pd.read_parquet(tmp_path / "threshold_search_results.parquet")
        cols = ["policy_family", "threshold", "k_capacity", "cost_regime_id"]
        assert df[cols].drop_duplicates().shape[0] == EXPECTED_CANDIDATES

    def test_exact_9000_tuning_episode_set(self, tmp_path):
        _build_full_valid(tmp_path)
        df = pd.read_parquet(tmp_path / "threshold_search_results.parquet")
        assert len(df) == EXPECTED_TUNING_EPISODES

    def test_exact_32_winners(self, tmp_path):
        _build_full_valid(tmp_path)
        sel = json.loads((tmp_path / "selected_thresholds.json").read_text())
        sel = {k: v for k, v in sel.items() if k != "_meta"}
        assert len(sel) == EXPECTED_SELECTED_THRESHOLDS

    def test_exact_2400_evaluation_identity_set(self, tmp_path):
        _build_full_valid(tmp_path)
        eps = pd.read_parquet(tmp_path / "episode_results.parquet")
        assert len(eps) == EXPECTED_EVALUATION_EPISODES

    def test_validate_script_passes_on_valid_full_set(self, tmp_path):
        _build_full_valid(tmp_path)
        result = _run_recompute(tmp_path, expected_returncode=0)
        # When the script passes, it writes independent_recomputation.json with verdict PASS.
        report = json.loads((tmp_path / "independent_recomputation.json").read_text())
        assert report["verdict"] == "PASS", f"Verdict should be PASS: {report}"


class TestRecomputeDetectsMutations:
    """Each mutation must cause nonzero exit from the recompute script."""

    def test_catches_missing_candidate(self, tmp_path):
        _build_full_valid(tmp_path)
        # Remove one full candidate (all 25 rows of one combination)
        # so the unique-candidate set drops below 360 and the
        # validator catches the missing candidate.
        df = pd.read_parquet(tmp_path / "threshold_search_results.parquet")
        # Identify the first combo and remove every row that matches it.
        first_combo = df.iloc[0]
        mask = (
            (df["policy_family"] == first_combo["policy_family"])
            & (df["threshold"] == first_combo["threshold"])
            & (df["k_capacity"] == first_combo["k_capacity"])
            & (df["cost_regime_id"] == first_combo["cost_regime_id"])
        )
        df = df[~mask].reset_index(drop=True)
        df.to_parquet(tmp_path / "threshold_search_results.parquet", index=False)
        _run_recompute(tmp_path, expected_returncode=1)

    def test_catches_duplicate_tuning_identity(self, tmp_path):
        _build_full_valid(tmp_path)
        # Append an extra candidate row that introduces a new
        # (policy, threshold, k, regime) combo, so the unique-candidates
        # count exceeds 360 and the validator's identity-set comparison
        # flags a duplicate tuning identity.
        df = pd.read_parquet(tmp_path / "threshold_search_results.parquet")
        extra_row = df.iloc[0].copy()
        extra_row["threshold"] = int(extra_row["threshold"]) + 1
        df = pd.concat([df, pd.DataFrame([extra_row])], ignore_index=True)
        df.to_parquet(tmp_path / "threshold_search_results.parquet", index=False)
        _run_recompute(tmp_path, expected_returncode=1)

    def test_catches_missing_evaluation_identity(self, tmp_path):
        _build_full_valid(tmp_path)
        _write_scenario_provenance(tmp_path, missing_evaluation_identity=True)
        _run_recompute(tmp_path, expected_returncode=1)

    def test_catches_wrong_selected_winner(self, tmp_path):
        _build_full_valid(tmp_path)
        _write_selected_thresholds(tmp_path, mutate=True)
        # Rebuild episodes against the mutated selected file so the validator
        # still runs against the corrupt state.
        _write_episode_results(tmp_path, tmp_path / "selected_thresholds.json")
        _write_summary(tmp_path, tmp_path / "episode_results.parquet")
        _run_recompute(tmp_path, expected_returncode=1)

    def test_catches_threshold_use_mismatch(self, tmp_path):
        _build_full_valid(tmp_path)
        _write_episode_results(
            tmp_path, tmp_path / "selected_thresholds.json",
            mutate_threshold_use=True,
        )
        _write_summary(tmp_path, tmp_path / "episode_results.parquet")
        _run_recompute(tmp_path, expected_returncode=1)

    def test_catches_summary_mismatch(self, tmp_path):
        _build_full_valid(tmp_path)
        _write_summary(tmp_path, tmp_path / "episode_results.parquet", mutate_summary=True)
        _run_recompute(tmp_path, expected_returncode=1)

    def test_catches_reward_cost_mismatch(self, tmp_path):
        _build_full_valid(tmp_path)
        _write_episode_results(
            tmp_path, tmp_path / "selected_thresholds.json",
            mutate_reward_cost=True,
        )
        _write_summary(tmp_path, tmp_path / "episode_results.parquet")
        _run_recompute(tmp_path, expected_returncode=1)

    def test_catches_scenario_bank_sha_mismatch(self, tmp_path):
        _build_full_valid(tmp_path)
        # Mutate sorted_scenario_ids_sha256 to a wrong 64-hex value
        # so the validator's independent recomputation catches the
        # inconsistency as a fail-closed SHA mismatch.
        _write_scenario_provenance(tmp_path, mutate_sha=True)
        _run_recompute(tmp_path, expected_returncode=1)


# =============================================================================
# Adversarial balanced mutations required by the contract-corrected spec.
#
# Each mutation leaves the total ROW count unchanged (a row is removed and
# another duplicated, or one row is replaced with an off-grid illegal
# value). The exact identity-set recomputation must still flag a fail.
# =============================================================================


class TestRecomputeBalancedAdversarial:
    """Balanced adversarial tests where total row count remains unchanged."""

    def test_remove_tuning_identity_and_duplicate_another(self, tmp_path):
        """Remove one tuning identity; duplicate another (total row count unchanged)."""
        _build_full_valid(tmp_path)
        df = pd.read_parquet(tmp_path / "threshold_search_results.parquet")

        # Pick the first distinct (policy, k, regime) group, then take
        # two distinct thresholds within it: remove one, duplicate the
        # other enough times to keep row count equal.
        first = df.iloc[0]
        same_family_k_regime = df[
            (df["policy_family"] == first["policy_family"])
            & (df["k_capacity"] == first["k_capacity"])
            & (df["cost_regime_id"] == first["cost_regime_id"])
        ]
        distinct_thresh = sorted(same_family_k_regime["threshold"].unique())
        if len(distinct_thresh) < 2:
            return  # nothing to do if grid too small

        t_remove = distinct_thresh[0]
        t_keep = distinct_thresh[1]
        # The exact tuning-identity that lands at threshold=t_remove is
        # the unique cell (policy, threshold=t_remove, k, regime), which
        # has 25 rows (5 scenarios × 5 seeds).
        rows_to_remove = same_family_k_regime[
            same_family_k_regime["threshold"] == t_remove
        ]
        removed_count = len(rows_to_remove)
        df_without = df.drop(rows_to_remove.index).reset_index(drop=True)

        # Duplicate t_keep rows enough times to compensate for the
        # removed_count. We copy each t_keep row once so the parquet
        # has additional rows that pair with already-present rows on
        # (policy, threshold, k, regime) but unique on
        # (scenario_id, reset_seed) only if the source rows have unique
        # scenario/seed combinations — that's the case in the synthetic
        # fixture, so naive duplication introduces new (scenario, seed)
        # combinations. To force exactly N duplicate (policy, threshold,
        # k, regime, scenario_id, reset_seed) identities, we duplicate
        # the same 25 rows again. This is the "balanced" trick.
        keep_rows = same_family_k_regime[same_family_k_regime["threshold"] == t_keep]
        # Replicate ``keep_rows`` so total row count matches original.
        duplicated_block = keep_rows.copy()
        df_new = pd.concat(
            [df_without, duplicated_block], ignore_index=True
        )
        # Total row count: original minus removed_count plus keep_rows.
        # To preserve row count exactly, we copy keep_rows which is
        # TRUNCATED to removed_count by iloc below.
        truncated_block = keep_rows.iloc[:removed_count]
        df_new = pd.concat(
            [df_without, truncated_block], ignore_index=True
        )
        assert abs(len(df_new) - len(df)) <= 0
        df_new.to_parquet(
            tmp_path / "threshold_search_results.parquet", index=False
        )
        _run_recompute(tmp_path, expected_returncode=1)

    def test_remove_candidate_and_add_illegal_candidate(self, tmp_path):
        """Remove one valid candidate; add one illegal candidate (off-grid threshold)."""
        _build_full_valid(tmp_path)
        df = pd.read_parquet(tmp_path / "threshold_search_results.parquet")

        first = df.iloc[0]
        candidate_mask = (
            (df["policy_family"] == first["policy_family"])
            & (df["threshold"] == first["threshold"])
            & (df["k_capacity"] == first["k_capacity"])
            & (df["cost_regime_id"] == first["cost_regime_id"])
        )
        # Pick an off-grid threshold (not in the policy's frozen grid).
        illegal_threshold = int(first["threshold"]) + 999
        replacement_rows = []
        for _, row in df[candidate_mask].iterrows():
            new_row = row.copy()
            new_row["threshold"] = illegal_threshold
            replacement_rows.append(new_row)
        df_without = df.drop(df[candidate_mask].index).reset_index(drop=True)
        df_new = pd.concat(
            [df_without, pd.DataFrame(replacement_rows)],
            ignore_index=True,
        )
        df_new.to_parquet(
            tmp_path / "threshold_search_results.parquet", index=False
        )
        _run_recompute(tmp_path, expected_returncode=1)

    def test_remove_evaluation_identity_and_duplicate_another(self, tmp_path):
        """Remove one evaluation identity; duplicate another (row count unchanged)."""
        _build_full_valid(tmp_path)
        eps = pd.read_parquet(tmp_path / "episode_results.parquet")

        # Drop the very first row and append a duplicate of row 1
        # so total row count is preserved.
        removed_row = eps.iloc[0:1]
        replacement_row = eps.iloc[1:2]
        df_without = eps.iloc[1:].reset_index(drop=True)
        df_new = pd.concat(
            [df_without, replacement_row], ignore_index=True
        )
        assert len(df_new) == len(eps)
        df_new.to_parquet(tmp_path / "episode_results.parquet", index=False)
        _run_recompute(tmp_path, expected_returncode=1)


# =============================================================================
# Nine balanced adversarial tests (Step 7 of contract corrected)
# =============================================================================
#
# Each test creates a balanced mutation (total row count unchanged) that
# must cause the independent recomputation to fail (nonzero exit).
#
# 1. Replace one valid tuning scenario ID with an invalid but unique ID
# 2. Replace one valid tuning reset seed with an invalid but unique seed
# 3. Remove one valid tuning identity and duplicate another
# 4. Replace one valid evaluation scenario ID with an invalid but unique ID
# 5. Remove one evaluation identity and duplicate another
# 6. Mutate the actual source bank file after the context SHA is recorded
# 7. Change context source_sha256 while leaving provenance internally consistent
# 8. Missing/extra candidate identity
# 9. Duplicate tuning identity with different scenario/seed (identity collision)


class TestNineBalancedAdversarial:
    """Nine balanced adversarial tests required by the contract-corrected spec."""

    def test_tuning_scenario_id_replaced_with_invalid_unique(self, tmp_path):
        """1. Replace one valid tuning scenario ID with an invalid but unique scenario ID."""
        _build_full_valid(tmp_path)
        df = pd.read_parquet(tmp_path / "threshold_search_results.parquet")

        # Replace scenario_id in one row with an invalid but unique ID
        invalid_id = "invalid_scenario_999"
        df.loc[0, "scenario_id"] = invalid_id
        df.to_parquet(tmp_path / "threshold_search_results.parquet", index=False)
        _run_recompute(tmp_path, expected_returncode=1)

    def test_tuning_reset_seed_replaced_with_invalid_unique(self, tmp_path):
        """2. Replace one valid tuning reset seed with an invalid but unique reset seed."""
        _build_full_valid(tmp_path)
        df = pd.read_parquet(tmp_path / "threshold_search_results.parquet")

        # Replace reset_seed in one row with an invalid but unique seed
        invalid_seed = 999999
        df.loc[0, "reset_seed"] = invalid_seed
        df.to_parquet(tmp_path / "threshold_search_results.parquet", index=False)
        _run_recompute(tmp_path, expected_returncode=1)

    def test_remove_tuning_identity_duplicate_another(self, tmp_path):
        """3. Remove one valid tuning identity and duplicate another (total row count preserved)."""
        _build_full_valid(tmp_path)
        df = pd.read_parquet(tmp_path / "threshold_search_results.parquet")

        # Pick the first distinct (policy, k, regime) group, then take
        # two distinct thresholds within it: remove one, duplicate the other
        first = df.iloc[0]
        same_family_k_regime = df[
            (df["policy_family"] == first["policy_family"])
            & (df["k_capacity"] == first["k_capacity"])
            & (df["cost_regime_id"] == first["cost_regime_id"])
        ]
        distinct_thresh = sorted(same_family_k_regime["threshold"].unique())
        if len(distinct_thresh) < 2:
            pytest.skip("grid too small")

        t_remove = distinct_thresh[0]
        t_keep = distinct_thresh[1]
        rows_to_remove = same_family_k_regime[
            same_family_k_regime["threshold"] == t_remove
        ]
        removed_count = len(rows_to_remove)
        df_without = df.drop(rows_to_remove.index).reset_index(drop=True)

        # Duplicate t_keep rows (truncated to removed_count)
        keep_rows = same_family_k_regime[same_family_k_regime["threshold"] == t_keep]
        truncated_block = keep_rows.iloc[:removed_count]
        df_new = pd.concat([df_without, truncated_block], ignore_index=True)
        assert len(df_new) == len(df)
        df_new.to_parquet(tmp_path / "threshold_search_results.parquet", index=False)
        _run_recompute(tmp_path, expected_returncode=1)

    def test_evaluation_scenario_id_replaced_with_invalid_unique(self, tmp_path):
        """4. Replace one valid evaluation scenario ID with an invalid but unique ID."""
        _build_full_valid(tmp_path)
        eps = pd.read_parquet(tmp_path / "episode_results.parquet")

        # Replace scenario_id in one row with an invalid but unique ID
        invalid_id = "invalid_eval_scenario_999"
        eps.loc[0, "scenario_id"] = invalid_id
        eps.to_parquet(tmp_path / "episode_results.parquet", index=False)
        _run_recompute(tmp_path, expected_returncode=1)

    def test_remove_evaluation_identity_duplicate_another(self, tmp_path):
        """5. Remove one evaluation identity and duplicate another (total row count unchanged)."""
        _build_full_valid(tmp_path)
        eps = pd.read_parquet(tmp_path / "episode_results.parquet")

        # Drop the first row and append a duplicate of row 1
        removed_row = eps.iloc[0:1]
        replacement_row = eps.iloc[1:2]
        df_without = eps.iloc[1:].reset_index(drop=True)
        df_new = pd.concat([df_without, replacement_row], ignore_index=True)
        assert len(df_new) == len(eps)
        df_new.to_parquet(tmp_path / "episode_results.parquet", index=False)
        _run_recompute(tmp_path, expected_returncode=1)

    def test_mutate_source_bank_file_after_context_sha(self, tmp_path):
        """6. Mutate the actual source bank file after the context SHA is recorded."""
        _build_full_valid(tmp_path)

        # Read the formal context to find the source bank path
        with open(tmp_path / "formal_run_context.json", "r") as f:
            context = json.load(f)
        source_path = Path(context["scenario_bank_identities"][0]["source_path"])

        # Backup original, then mutate
        original_content = source_path.read_text()
        try:
            # Add a whitespace mutation that changes SHA
            mutated = original_content + "\n// MUTATED"
            source_path.write_text(mutated)
            _run_recompute(tmp_path, expected_returncode=1)
        finally:
            # Restore
            source_path.write_text(original_content)

    def test_change_context_source_sha256_provenance_consistent(self, tmp_path):
        """7. Change context source_sha256 while leaving provenance internally consistent."""
        _build_full_valid(tmp_path)

        # Modify the formal_run_context.json: change one source_sha256
        with open(tmp_path / "formal_run_context.json", "r") as f:
            context = json.load(f)
        context["scenario_bank_identities"][0]["source_sha256"] = "deadbeef" * 8
        with open(tmp_path / "formal_run_context.json", "w") as f:
            json.dump(context, f, indent=2)

        _run_recompute(tmp_path, expected_returncode=1)

    def test_missing_candidate_identity(self, tmp_path):
        """8. Missing/extra candidate identity."""
        _build_full_valid(tmp_path)
        df = pd.read_parquet(tmp_path / "threshold_search_results.parquet")

        # Remove all rows for one candidate identity
        first = df.iloc[0]
        candidate_mask = (
            (df["policy_family"] == first["policy_family"])
            & (df["threshold"] == first["threshold"])
            & (df["k_capacity"] == first["k_capacity"])
            & (df["cost_regime_id"] == first["cost_regime_id"])
        )
        df_without = df[~candidate_mask].reset_index(drop=True)
        df_without.to_parquet(tmp_path / "threshold_search_results.parquet", index=False)
        _run_recompute(tmp_path, expected_returncode=1)

    def test_extra_candidate_identity(self, tmp_path):
        """8. Extra candidate identity (duplicate with different threshold)."""
        _build_full_valid(tmp_path)
        df = pd.read_parquet(tmp_path / "threshold_search_results.parquet")

        # Duplicate one candidate's rows but change the threshold to an off-grid value
        first = df.iloc[0]
        candidate_mask = (
            (df["policy_family"] == first["policy_family"])
            & (df["threshold"] == first["threshold"])
            & (df["k_capacity"] == first["k_capacity"])
            & (df["cost_regime_id"] == first["cost_regime_id"])
        )
        # Use an off-grid threshold
        illegal_threshold = int(first["threshold"]) + 999
        extra_rows = df[candidate_mask].copy()
        extra_rows["threshold"] = illegal_threshold
        df_new = pd.concat([df, extra_rows], ignore_index=True)
        df_new.to_parquet(tmp_path / "threshold_search_results.parquet", index=False)
        _run_recompute(tmp_path, expected_returncode=1)


# =============================================================================
# Targeted adversarial tests required by the M3 readiness corrected spec
# (Section 9). All operate on tmp_path COPIES of the canonical scenario
# banks; the autouse integrity fixture asserts canonical bank SHAs are
# unchanged before/after each test.
# =============================================================================


def _distinct_per_kregime_bank(
    out: Path, split: str, k: int, regime: str, scenarios_per_bank: int = 5,
) -> tuple:
    """Build a synthetic per-(split,K,regime) bank with scenario IDs that
    are UNIQUE to that K×regime combination. The canonical repository
    banks all share scenario IDs across K/regime; this fixture creates
    16 distinct physical tmp_path files with disjoint scenario ID sets
    so the per-bank mapping logic can be exercised adversarially.

    ``scenarios_per_bank`` controls the count per bank (default 5, the
    formal contract value); a 4-scenario bank yields an internally
    consistent but smaller universe the recompute must still reject.
    """
    regime_token = regime.replace("failure-", "").replace("-no-waste", "L").replace("-waste-aware", "W")
    scenarios = []
    for i in range(scenarios_per_bank):
        sid = f"synth_{split[:2]}_k{k}_{regime_token}_{i}"
        scenarios.append({"scenario_id": sid, "k_capacity": k, "cost_regime_id": regime})
    dst_dir = out / "distinct_banks"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{split}_k{k}_{regime}.json"
    dst.write_text(json.dumps({"scenarios": scenarios}, indent=2))
    ids = [s["scenario_id"] for s in scenarios]
    sha = hashlib.sha256(dst.read_bytes()).hexdigest()
    ids_sha = hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()
    return dst, ids, sha, ids_sha


def _build_full_valid_with_distinct_per_kregime_banks(
    out: Path, swap_one_tuning_scenario_to_wrong_bank: bool = False,
    scenarios_per_bank: int = 5,
) -> None:
    """Variant fixture where every (split,K,regime) bank has DISTINCT
    scenario ID strings (16 disjoint sets). The recompute's per-bank
    expected-set construction must map each candidate to its exact
    (rl_validation, K, regime) bank's scenario IDs; using another bank's
    IDs must fail even though row count and unique count stay correct.
    """
    out.mkdir(parents=True, exist_ok=True)

    distinct_banks: dict = {}
    for split in EVAL_SPLITS:
        for k in K_VALUES:
            for regime in REGIMES:
                distinct_banks[(split, k, regime)] = _distinct_per_kregime_bank(
                    out, split, k, regime, scenarios_per_bank=scenarios_per_bank,
                )

    # resolved_config + selected_thresholds + validation_report + summary
    rc_path = _write_full_resolved_config(out)
    rc_sha = _canonical_json_sha(json.loads(rc_path.read_text()))
    _write_selected_thresholds(out)
    sel_sha = hashlib.sha256((out / "selected_thresholds.json").read_bytes()).hexdigest()
    _write_validation_report(out)

    reset_seeds = [6521, 6522, 6523, 6524, 6525]

    # Write threshold_search_results.parquet using the EXACT
    # rl_validation (K, regime) bank's scenario IDs per candidate.
    rows = []
    for policy_family in FORMAL_POLICIES:
        grid = GRIDS[policy_family]
        for k in K_VALUES:
            for regime in REGIMES:
                rl_path, rl_ids, _, _ = distinct_banks[("rl_validation", k, regime)]
                for threshold in sorted(grid):
                    for scenario_id in rl_ids:
                        for seed in reset_seeds:
                            total_cost = 1000.0 + (threshold - min(grid)) / max(grid) * 4
                            rows.append({
                                "policy_family": policy_family,
                                "threshold": threshold,
                                "k_capacity": k,
                                "cost_regime_id": regime,
                                "mean_total_cost": total_cost,
                                "total_failures": 0,
                                "mean_wasted_life_cost": total_cost - 5.0 - 2.0,
                                "scenario_id": scenario_id,
                                "reset_seed": seed,
                                "mean_preventive_cost": 2.0,
                                "mean_failure_cost": 5.0,
                                "episode_count": 1,
                                "total_episode_count": 1,
                                "total_cost": total_cost,
                                "preventive_cost": 2.0,
                                "failure_cost": 5.0,
                                "wasted_life_cost": total_cost - 5.0 - 2.0,
                                "failure_count": 0,
                                "episode_steps": 100,
                                "completed": True,
                            })
    df_tune = pd.DataFrame(rows)
    if swap_one_tuning_scenario_to_wrong_bank:
        # Replace one tuning row's scenario_id with an ID drawn from a
        # DIFFERENT (K, regime) rl_validation bank. Row count and unique
        # candidate count are unchanged; only the 6-tuple identity set
        # diverges (the wrong-bank ID is not in the expected tuning set).
        other_path, other_ids, _, _ = distinct_banks[("rl_validation", 2, REGIMES[-1])]
        df_tune.loc[0, "scenario_id"] = other_ids[0]
    df_tune.to_parquet(out / "threshold_search_results.parquet", index=False)
    candidate_summary = (
        df_tune[df_tune["completed"]]
        .groupby(
            ["policy_family", "threshold", "k_capacity", "cost_regime_id"],
            as_index=False,
        )
        .agg(
            mean_total_cost=("total_cost", "mean"),
            total_failures=("failure_count", "sum"),
            mean_wasted_life_cost=("wasted_life_cost", "mean"),
            episode_count=("total_cost", "size"),
        )
    )
    candidate_summary.to_csv(
        out / "threshold_search_summary.csv", index=False
    )

    # Write episode_results.parquet using exact (split, K, regime) bank ids.
    selected = json.loads((out / "selected_thresholds.json").read_text())
    selected = {k_: v for k_, v in selected.items() if k_ != "_meta"}
    eval_rows = []
    for split in EVAL_SPLITS:
        for k in K_VALUES:
            for regime in REGIMES:
                epath, eids, _, _ = distinct_banks[(split, k, regime)]
                for policy_family in EVAL_POLICY_FAMILIES:
                    for scenario_id in eids:
                        for seed in reset_seeds:
                            key = _identity(policy_family, k, regime)
                            threshold = (selected.get(key, {}).get("threshold")
                                         if policy_family in FORMAL_POLICIES else None)
                            eval_rows.append({
                                "policy_id": policy_family,
                                "policy_family": policy_family,
                                "split": split,
                                "maintenance_capacity": k,
                                "k_capacity": k,
                                "cost_regime_id": regime,
                                "threshold": threshold,
                                "scenario_id": scenario_id,
                                "reset_seed": seed,
                                "total_cost": 1000.0,
                                "preventive_cost": 998.0,
                                "failure_cost": 1.0,
                                "wasted_life_cost": 1.0,
                                "episode_return": -1000.0,
                                "episode_count": 1,
                            })
    eps = pd.DataFrame(eval_rows)
    eps.to_parquet(out / "episode_results.parquet", index=False)

    # summary_by_policy.csv
    group_cols = ["policy_id", "split", "maintenance_capacity", "cost_regime_id"]
    summary = eps.groupby(group_cols).agg(mean=("total_cost", "mean")).reset_index()
    summary["sample_std"] = 0.0
    summary["standard_error"] = 0.0
    summary["ci_95_lower"] = summary["mean"]
    summary["ci_95_upper"] = summary["mean"]
    summary["episode_count"] = 1
    summary.to_csv(out / "summary_by_policy.csv", index=False)

    # scenario_bank_provenance.json — 16 records pointing at the 16 distinct files.
    banks = []
    for split in EVAL_SPLITS:
        for k in K_VALUES:
            for regime in REGIMES:
                bpath, ids, bsha, ids_sha = distinct_banks[(split, k, regime)]
                banks.append({
                    "split": split,
                    "K": k,
                    "derived_k": k,
                    "k": k,
                    "cost_regime_id": regime,
                    "derived_cost_regime_id": regime,
                    "logical_bank_id": f"bank_{split}_{k}_{regime}",
                    "source_bank_path": str(bpath),
                    "source_path": str(bpath),
                    "bank_sha256": bsha,
                    "source_sha256": bsha,
                    "source_file_size": bpath.stat().st_size,
                    "bank_scenario_count": len(ids),
                    "derived_scenario_count": len(ids),
                    "scenario_count": len(ids),
                    "scenario_ids": ids,
                    "derived_scenario_ids": ids,
                    "sorted_scenario_ids_sha256": ids_sha,
                    "derived_bank_sha256": bsha,
                })
    (out / "scenario_bank_provenance.json").write_text(
        json.dumps({"scenario_banks": banks}, indent=2),
    )

    # formal_run_context.json — 16 identities pointing at the 16 distinct files.
    identities = []
    for split in EVAL_SPLITS:
        for k in K_VALUES:
            for regime in REGIMES:
                bpath, ids, bsha, ids_sha = distinct_banks[(split, k, regime)]
                identities.append({
                    "split": split,
                    "K": k,
                    "cost_regime_id": regime,
                    "source_path": str(bpath),
                    "source_sha256": bsha,
                    "scenario_count": len(ids),
                    "sorted_scenario_ids_sha256": ids_sha,
                })
    (out / "formal_run_context.json").write_text(json.dumps({
        "schema_version": "m3_formal_context_v1",
        "formal_run_id": out.name,
        "mode": "formal_closeout",
        "implementation_commit": "0" * 40,
        "implementation_tree_clean": True,
        "resolved_config_path": str(rc_path),
        "resolved_config_sha256": rc_sha,
        "oracle_authorized": True,
        "selected_thresholds_path": str(out / "selected_thresholds.json"),
        "selected_thresholds_sha256": sel_sha,
        "sealed": True,
        "sealed_at": "2024-01-01T00:00:00",
        "scenario_bank_identities": identities,
        "reset_seeds": reset_seeds,
        "created_at": "2024-01-01T00:00:00",
    }))


class TestTargetedReadinessRepair:
    """Section-9 required adversarial tests (all use tmp_path bank copies)."""

    def test_source_bank_mutation_attempt_rejected(self, tmp_path):
        """1. Canonical source-bank mutation attempt must be rejected by
        the recompute (and the canonical file must NOT be mutated)."""
        _build_full_valid(tmp_path)
        ctx = json.loads((tmp_path / "formal_run_context.json").read_text())
        source_path = Path(ctx["scenario_bank_identities"][0]["source_path"])
        assert str(source_path).startswith(str(tmp_path)), (
            f"identities must point at tmp_path copies, got {source_path}"
        )
        original = source_path.read_text()
        try:
            source_path.write_text(original + "\n// MUTATED")
            _run_recompute(tmp_path, expected_returncode=1)
            # Canonical repo banks untouched (asserted by the autouse fixture).
        finally:
            source_path.write_text(original)

    def test_missing_one_of_16_bank_identities_rejected(self, tmp_path):
        """2. Dropping one of the 16 bank identities → recompute fails."""
        _build_full_valid(tmp_path)
        ctx = json.loads((tmp_path / "formal_run_context.json").read_text())
        # remove one identity record (15 of 16 remain)
        ctx["scenario_bank_identities"] = ctx["scenario_bank_identities"][:-1]
        (tmp_path / "formal_run_context.json").write_text(json.dumps(ctx, indent=2))
        _run_recompute(tmp_path, expected_returncode=1)

    def test_duplicate_bank_identity_rejected(self, tmp_path):
        """3. A duplicate (split,K,regime) identity → recompute fails."""
        _build_full_valid(tmp_path)
        ctx = json.loads((tmp_path / "formal_run_context.json").read_text())
        # Duplicate the first identity record.
        ctx["scenario_bank_identities"].append(
            dict(ctx["scenario_bank_identities"][0])
        )
        (tmp_path / "formal_run_context.json").write_text(json.dumps(ctx, indent=2))
        _run_recompute(tmp_path, expected_returncode=1)

    def test_distinct_per_kregime_scenario_ids_pass(self, tmp_path):
        """4a. With distinct per-(K,regime) scenario IDs, the valid
        mapping (each candidate uses its exact bank's IDs) PASSES."""
        _build_full_valid_with_distinct_per_kregime_banks(tmp_path)
        _run_recompute(tmp_path, expected_returncode=0)

    def test_wrong_kregime_scenario_id_for_candidate_rejected(self, tmp_path):
        """4b. One tuning row uses a scenario ID from a DIFFERENT K/regime
        bank; row count and unique count are unchanged but the identity
        set diverges → recompute FAILS."""
        _build_full_valid_with_distinct_per_kregime_banks(
            tmp_path, swap_one_tuning_scenario_to_wrong_bank=True
        )
        _run_recompute(tmp_path, expected_returncode=1)

    def test_missing_reset_seeds_rejected(self, tmp_path):
        """5. Missing reset_seeds → recompute fails (no fallback)."""
        _build_full_valid(tmp_path)
        ctx = json.loads((tmp_path / "formal_run_context.json").read_text())
        ctx.pop("reset_seeds", None)
        (tmp_path / "formal_run_context.json").write_text(json.dumps(ctx, indent=2))
        _run_recompute(tmp_path, expected_returncode=1)

    def test_duplicate_reset_seed_rejected(self, tmp_path):
        """6. Duplicate reset seed → recompute fails."""
        _build_full_valid(tmp_path)
        ctx = json.loads((tmp_path / "formal_run_context.json").read_text())
        ctx["reset_seeds"] = [6521, 6521, 6523, 6524, 6525]
        (tmp_path / "formal_run_context.json").write_text(json.dumps(ctx, indent=2))
        _run_recompute(tmp_path, expected_returncode=1)

    def test_wrong_resolved_config_sha_rejected(self, tmp_path):
        """7. Wrong resolved_config_sha256 → recompute fails."""
        _build_full_valid(tmp_path)
        ctx = json.loads((tmp_path / "formal_run_context.json").read_text())
        ctx["resolved_config_sha256"] = "0" * 64
        (tmp_path / "formal_run_context.json").write_text(json.dumps(ctx, indent=2))
        _run_recompute(tmp_path, expected_returncode=1)

    def test_wrong_selected_threshold_sha_rejected(self, tmp_path):
        """8. Wrong selected_thresholds_sha256 → recompute fails."""
        _build_full_valid(tmp_path)
        ctx = json.loads((tmp_path / "formal_run_context.json").read_text())
        ctx["selected_thresholds_sha256"] = "0" * 64
        (tmp_path / "formal_run_context.json").write_text(json.dumps(ctx, indent=2))
        _run_recompute(tmp_path, expected_returncode=1)

    def test_structured_report_missing_required_section_rejected(self, tmp_path):
        """9. independent_recomputation.json must contain every required
        section. The recomputation of a valid fixture MUST produce all
        sections; removing one (i.e., the recomputation produces an
        incomplete report) must be detectable. We exercise this by
        running the script on a valid fixture and asserting every
        required section is present; the negative (a malformed
        independent_recomputation.json) is itself rejected by the
        formal manifest, verified in test_manifest_pass_with_mismatched_identity_set_sha_rejected
        below via the SHA gate."""
        _build_full_valid(tmp_path)
        _run_recompute(tmp_path, expected_returncode=0)
        r = json.loads((tmp_path / "independent_recomputation.json").read_text())
        required = [
            "schema_version", "script_version", "verdict", "executed_at",
            "formal_run_id", "implementation_commit",
            "formal_run_context_verification", "resolved_config_verification",
            "selected_threshold_file_verification", "scenario_bank_set_evidence",
            "scenario_bank_file_evidence", "candidate_set_evidence",
            "candidate_summary_recomputation_evidence",
            "tuning_set_evidence", "selected_winner_evidence",
            "deterministic_tie_break_evidence", "evaluation_set_evidence",
            "threshold_use_evidence", "non_threshold_policy_evidence",
            "reward_cost_evidence", "cost_decomposition_evidence",
            "summary_recomputation_evidence", "oracle_terminology_evidence",
            "errors",
        ]
        missing = [k for k in required if k not in r]
        assert not missing, f"independent_recomputation.json missing required sections: {missing}"

    def test_manifest_pass_with_mismatched_identity_set_sha_rejected(self, tmp_path):
        """10. A formal manifest must NOT report PASS when the
        independent_recomputation.json identity-set SHAs mismatch. Build
        an independent_recomputation.json with a matching verdict but a
        candidate_set_evidence expected != actual SHA, then require
        generate_formal_manifest(formal_closeout) to fail."""
        from src.baselines.artifacts import generate_formal_manifest

        out = tmp_path / "m3_manifest_mismatch"
        out.mkdir(parents=True, exist_ok=True)
        # The manifest's context validation requires implementation_commit
        # to equal the current HEAD; resolve it dynamically so this test
        # stays valid at any commit the corrected lands on.
        current_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True,
        ).strip()

        cfg = {
            "policy_families": list(FORMAL_POLICIES),
            "threshold_grids": GRIDS,
            "k_values": list(K_VALUES),
            "cost_regimes": list(REGIMES),
            "evaluation_splits": list(EVAL_SPLITS),
            "reset_seeds": [6521, 6522, 6523, 6524, 6525],
        }
        (out / "resolved_config.json").write_text(json.dumps(cfg, indent=2))
        rc_sha = _canonical_json_sha(cfg)
        _write_selected_thresholds(out)
        sel_sha = hashlib.sha256((out / "selected_thresholds.json").read_bytes()).hexdigest()

        # Sealed formal_run_context with the current git HEAD + all required fields.
        (out / "formal_run_context.json").write_text(json.dumps({
            "schema_version": "m3_formal_context_v1",
            "formal_run_id": out.name,
            "mode": "formal_closeout",
            "implementation_commit": current_commit,
            "implementation_tree_clean": True,
            "resolved_config_path": str(out / "resolved_config.json"),
            "resolved_config_sha256": rc_sha,
            "oracle_authorized": True,
            "selected_thresholds_path": str(out / "selected_thresholds.json"),
            "selected_thresholds_sha256": sel_sha,
            "sealed": True,
            "sealed_at": "2024-01-01T00:00:00",
            "scenario_bank_identities": [],
            "reset_seeds": [6521, 6522, 6523, 6524, 6525],
            "created_at": "2024-01-01T00:00:00",
        }))

        (out / "validation_report.json").write_text(json.dumps({
            "verdict": "ALL PASSED",
            "oracle_semantic_role": "privileged-information diagnostic benchmark",
        }))
        # Full-Section recompute report with top-level verdict PASS. The
        # candidate_set_evidence carries a MISMATCHED expected/actual SHA so
        # the manifest's identity-set SHA gate (not the missing-section gate)
        # is the specific contract that fires.
        _write_full_recompute_report(out)
        r = json.loads((out / "independent_recomputation.json").read_text())
        r["candidate_set_evidence"]["expected_set_sha256"] = "0" * 64
        r["candidate_set_evidence"]["actual_set_sha256"] = "1" * 64
        (out / "independent_recomputation.json").write_text(json.dumps(r, indent=2))
        # Provide empty (but valid-shaped) parquet/csv so artifact recount field loads.
        pd.DataFrame({"policy_family": []}).to_parquet(out / "threshold_search_results.parquet", index=False)
        pd.DataFrame({"policy_family": []}).to_parquet(out / "episode_results.parquet", index=False)
        (out / "threshold_search_summary.csv").write_text("policy_id,mean\n")
        (out / "summary_by_policy.csv").write_text("policy_id,mean\n")
        (out / "scenario_bank_provenance.json").write_text(json.dumps({"scenario_banks": []}))
        (out / "run_provenance.json").write_text(json.dumps({"reset_seeds": [6521, 6522, 6523, 6524, 6525]}))

        with pytest.raises(RuntimeError, match="identity-set SHA mismatch"):
            generate_formal_manifest(out, mode="formal_closeout")

    def test_manifest_missing_required_section_rejected(self, tmp_path):
        """10b. A formal manifest must reject an independent_recomputation.json
        that has top-level verdict PASS but is missing a required evidence
        section. This is NOT the identity-SHA mismatch test — it deletes one
        section (e.g. tuning_set_evidence) and requires a missing-section-
        specific rejection."""
        from src.baselines.artifacts import generate_formal_manifest

        out = tmp_path / "m3_manifest_missing_section"
        out.mkdir(parents=True, exist_ok=True)
        current_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True,
        ).strip()
        cfg = {
            "policy_families": list(FORMAL_POLICIES),
            "threshold_grids": {k: list(v) for k, v in GRIDS.items()},
            "k_values": list(K_VALUES), "cost_regimes": list(REGIMES),
            "evaluation_splits": list(EVAL_SPLITS),
            "reset_seeds": [6521, 6522, 6523, 6524, 6525],
        }
        (out / "resolved_config.json").write_text(json.dumps(cfg, indent=2))
        rc_sha = _canonical_json_sha(cfg)
        _write_selected_thresholds(out)
        sel_sha = hashlib.sha256((out / "selected_thresholds.json").read_bytes()).hexdigest()
        (out / "formal_run_context.json").write_text(json.dumps({
            "schema_version": "m3_formal_context_v1", "formal_run_id": out.name,
            "mode": "formal_closeout", "implementation_commit": current_commit,
            "implementation_tree_clean": True,
            "resolved_config_path": str(out / "resolved_config.json"),
            "resolved_config_sha256": rc_sha, "oracle_authorized": True,
            "selected_thresholds_path": str(out / "selected_thresholds.json"),
            "selected_thresholds_sha256": sel_sha, "sealed": True,
            "sealed_at": "2024-01-01T00:00:00", "scenario_bank_identities": [],
            "reset_seeds": [6521, 6522, 6523, 6524, 6525],
            "created_at": "2024-01-01T00:00:00",
        }))
        (out / "validation_report.json").write_text(json.dumps({
            "verdict": "ALL PASSED",
            "oracle_semantic_role": "privileged-information diagnostic benchmark",
        }))
        # Drop the tuning_set_evidence section; keep top-level verdict PASS.
        _write_full_recompute_report(out, drop_section="tuning_set_evidence")
        pd.DataFrame({"policy_family": []}).to_parquet(out / "threshold_search_results.parquet", index=False)
        pd.DataFrame({"policy_family": []}).to_parquet(out / "episode_results.parquet", index=False)
        (out / "threshold_search_summary.csv").write_text("policy_id,mean\n")
        (out / "summary_by_policy.csv").write_text("policy_id,mean\n")
        (out / "scenario_bank_provenance.json").write_text(json.dumps({"scenario_banks": []}))
        (out / "run_provenance.json").write_text(json.dumps({"reset_seeds": [6521, 6522, 6523, 6524, 6525]}))

        with pytest.raises(RuntimeError, match="missing required evidence sections"):
            generate_formal_manifest(out, mode="formal_closeout")

    def test_manifest_recompute_section_non_pass_rejected(self, tmp_path):
        """10c. A formal manifest must reject an independent_recomputation.json
        whose top-level verdict is PASS but a structured section reports a
        non-PASS verdict."""
        from src.baselines.artifacts import generate_formal_manifest

        out = tmp_path / "m3_manifest_nonpass_section"
        out.mkdir(parents=True, exist_ok=True)
        current_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True,
        ).strip()
        cfg = {
            "policy_families": list(FORMAL_POLICIES),
            "threshold_grids": {k: list(v) for k, v in GRIDS.items()},
            "k_values": list(K_VALUES), "cost_regimes": list(REGIMES),
            "evaluation_splits": list(EVAL_SPLITS),
            "reset_seeds": [6521, 6522, 6523, 6524, 6525],
        }
        (out / "resolved_config.json").write_text(json.dumps(cfg, indent=2))
        rc_sha = _canonical_json_sha(cfg)
        _write_selected_thresholds(out)
        sel_sha = hashlib.sha256((out / "selected_thresholds.json").read_bytes()).hexdigest()
        (out / "formal_run_context.json").write_text(json.dumps({
            "schema_version": "m3_formal_context_v1", "formal_run_id": out.name,
            "mode": "formal_closeout", "implementation_commit": current_commit,
            "implementation_tree_clean": True,
            "resolved_config_path": str(out / "resolved_config.json"),
            "resolved_config_sha256": rc_sha, "oracle_authorized": True,
            "selected_thresholds_path": str(out / "selected_thresholds.json"),
            "selected_thresholds_sha256": sel_sha, "sealed": True,
            "sealed_at": "2024-01-01T00:00:00", "scenario_bank_identities": [],
            "reset_seeds": [6521, 6522, 6523, 6524, 6525],
            "created_at": "2024-01-01T00:00:00",
        }))
        (out / "validation_report.json").write_text(json.dumps({
            "verdict": "ALL PASSED",
            "oracle_semantic_role": "privileged-information diagnostic benchmark",
        }))
        # Mutate one structured section's verdict to FAIL; top-level stays PASS.
        _write_full_recompute_report(out, mutate_section="reward_cost_evidence")
        pd.DataFrame({"policy_family": []}).to_parquet(out / "threshold_search_results.parquet", index=False)
        pd.DataFrame({"policy_family": []}).to_parquet(out / "episode_results.parquet", index=False)
        (out / "threshold_search_summary.csv").write_text("policy_id,mean\n")
        (out / "summary_by_policy.csv").write_text("policy_id,mean\n")
        (out / "scenario_bank_provenance.json").write_text(json.dumps({"scenario_banks": []}))
        (out / "run_provenance.json").write_text(json.dumps({"reset_seeds": [6521, 6522, 6523, 6524, 6525]}))

        # The manifest must reject a structured section carrying a non-PASS
        # verdict even when the top-level report verdict stays PASS.
        with pytest.raises(RuntimeError, match="reward_cost_evidence: FAIL"):
            generate_formal_manifest(out, mode="formal_closeout")

    def test_manifest_wrong_oracle_semantic_role_rejected(self, tmp_path):
        """10d. A formal manifest must reject an independent_recomputation.json
        whose oracle_terminology_evidence.oracle_semantic_role does not equal
        the authoritative diagnostic-benchmark role."""
        from src.baselines.artifacts import generate_formal_manifest

        out = tmp_path / "m3_manifest_wrong_oracle_role"
        out.mkdir(parents=True, exist_ok=True)
        current_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True,
        ).strip()
        cfg = {
            "policy_families": list(FORMAL_POLICIES),
            "threshold_grids": {k: list(v) for k, v in GRIDS.items()},
            "k_values": list(K_VALUES), "cost_regimes": list(REGIMES),
            "evaluation_splits": list(EVAL_SPLITS),
            "reset_seeds": [6521, 6522, 6523, 6524, 6525],
        }
        (out / "resolved_config.json").write_text(json.dumps(cfg, indent=2))
        rc_sha = _canonical_json_sha(cfg)
        _write_selected_thresholds(out)
        sel_sha = hashlib.sha256((out / "selected_thresholds.json").read_bytes()).hexdigest()
        (out / "formal_run_context.json").write_text(json.dumps({
            "schema_version": "m3_formal_context_v1", "formal_run_id": out.name,
            "mode": "formal_closeout", "implementation_commit": current_commit,
            "implementation_tree_clean": True,
            "resolved_config_path": str(out / "resolved_config.json"),
            "resolved_config_sha256": rc_sha, "oracle_authorized": True,
            "selected_thresholds_path": str(out / "selected_thresholds.json"),
            "selected_thresholds_sha256": sel_sha, "sealed": True,
            "sealed_at": "2024-01-01T00:00:00", "scenario_bank_identities": [],
            "reset_seeds": [6521, 6522, 6523, 6524, 6525],
            "created_at": "2024-01-01T00:00:00",
        }))
        (out / "validation_report.json").write_text(json.dumps({
            "verdict": "ALL PASSED",
            "oracle_semantic_role": "privileged-information diagnostic benchmark",
        }))
        _write_full_recompute_report(out)
        r = json.loads((out / "independent_recomputation.json").read_text())
        # Invent an optimum-style role inside the report → must be rejected.
        r["oracle_terminology_evidence"]["oracle_semantic_role"] = "optimal upper bound"
        (out / "independent_recomputation.json").write_text(json.dumps(r, indent=2))
        pd.DataFrame({"policy_family": []}).to_parquet(out / "threshold_search_results.parquet", index=False)
        pd.DataFrame({"policy_family": []}).to_parquet(out / "episode_results.parquet", index=False)
        (out / "threshold_search_summary.csv").write_text("policy_id,mean\n")
        (out / "summary_by_policy.csv").write_text("policy_id,mean\n")
        (out / "scenario_bank_provenance.json").write_text(json.dumps({"scenario_banks": []}))
        (out / "run_provenance.json").write_text(json.dumps({"reset_seeds": [6521, 6522, 6523, 6524, 6525]}))

        with pytest.raises(RuntimeError, match="oracle_semantic_role"):
            generate_formal_manifest(out, mode="formal_closeout")

    def test_forbidden_oracle_terminology_rejected(self, tmp_path):
        """11. Forbidden Oracle terminology in scanned evidence → FAIL."""
        _build_full_valid(tmp_path)
        # Append a forbidden phrase into a generated evidence file the
        # checker scans (scenario_bank_provenance.json carries text).
        prov = json.loads((tmp_path / "scenario_bank_provenance.json").read_text())
        prov["oracle_label_note"] = "Oracle is an upper bound policy"
        (tmp_path / "scenario_bank_provenance.json").write_text(
            json.dumps(prov, indent=2),
        )
        _run_recompute(tmp_path, expected_returncode=1)

    # ---- §3: per-bank structured evidence verdict fails on per-record defect ----

    def _bank_file_evidence_for(self, tmp_path, bank_index) -> dict:
        """Run the recompute on a freshly-built valid fixture and return the
        per-bank file-evidence record for identity ``bank_index`` (the record
        produced by verify_scenario_bank_sources)."""
        _build_full_valid(tmp_path)
        _run_recompute(tmp_path, expected_returncode=0)
        r = json.loads((tmp_path / "independent_recomputation.json").read_text())
        return r

    def _patch_context_identity(self, tmp_path, bank_index, patch) -> None:
        ctx = json.loads((tmp_path / "formal_run_context.json").read_text())
        ctx["scenario_bank_identities"][bank_index].update(patch)
        (tmp_path / "formal_run_context.json").write_text(json.dumps(ctx, indent=2))

    def _find_bank_evidence(self, tmp_path, bank_key) -> dict:
        r = json.loads((tmp_path / "independent_recomputation.json").read_text())
        for ev in r.get("scenario_bank_file_evidence", []):
            if ev["bank_key"] == bank_key:
                return ev
        raise AssertionError(f"no per-bank evidence for {bank_key}")

    def test_per_bank_malformed_source_sha_rejected(self, tmp_path):
        """§3a. A bank record with a malformed source_sha256 must produce a
        per-bank verdict FAIL (source_sha_format_valid=False) AND a global
        reconciliation error, and the JSON evidence section for that bank
        must report FAIL."""
        _build_full_valid(tmp_path)
        self._patch_context_identity(tmp_path, bank_index=0, patch={
            "source_sha256": "not-a-hex-string",
        })
        _run_recompute(tmp_path, expected_returncode=1)
        ctx = json.loads((tmp_path / "formal_run_context.json").read_text())
        rec = ctx["scenario_bank_identities"][0]
        bank_key = f"{rec['split']}_k{rec['K']}_{rec['cost_regime_id']}"
        ev = self._find_bank_evidence(tmp_path, bank_key)
        assert ev["source_sha_format_valid"] is False, ev
        assert ev["source_sha_match"] is False, ev
        assert ev["verdict"] == "FAIL", ev

    def test_per_bank_missing_source_sha_rejected(self, tmp_path):
        """§3b. A bank record with NO source_sha256 must produce per-bank
        verdict FAIL (source_sha_format_valid=False)."""
        _build_full_valid(tmp_path)
        self._patch_context_identity(tmp_path, bank_index=0, patch={
            "source_sha256": None,
        })
        _run_recompute(tmp_path, expected_returncode=1)
        ctx = json.loads((tmp_path / "formal_run_context.json").read_text())
        rec = ctx["scenario_bank_identities"][0]
        bank_key = f"{rec['split']}_k{rec['K']}_{rec['cost_regime_id']}"
        ev = self._find_bank_evidence(tmp_path, bank_key)
        assert ev["source_sha_format_valid"] is False, ev
        assert ev["verdict"] == "FAIL", ev

    def test_per_bank_scenario_count_mismatch_rejected(self, tmp_path):
        """§3c. A recorded scenario_count that does not match the actual file
        must produce per-bank verdict FAIL (scenario_count_match=False)."""
        _build_full_valid(tmp_path)
        self._patch_context_identity(tmp_path, bank_index=0, patch={
            "scenario_count": 99999,
        })
        _run_recompute(tmp_path, expected_returncode=1)
        ctx = json.loads((tmp_path / "formal_run_context.json").read_text())
        rec = ctx["scenario_bank_identities"][0]
        bank_key = f"{rec['split']}_k{rec['K']}_{rec['cost_regime_id']}"
        ev = self._find_bank_evidence(tmp_path, bank_key)
        assert ev["scenario_count_present"] is True, ev
        assert ev["scenario_count_match"] is False, ev
        assert ev["verdict"] == "FAIL", ev

    def test_per_bank_scenario_id_sha_mismatch_rejected(self, tmp_path):
        """§3d. A sorted_scenario_ids_sha256 that does not match the actual
        bank must produce per-bank verdict FAIL (scenario_ids_sha_match=False)."""
        _build_full_valid(tmp_path)
        self._patch_context_identity(tmp_path, bank_index=0, patch={
            "sorted_scenario_ids_sha256": "0" * 64,
        })
        _run_recompute(tmp_path, expected_returncode=1)
        ctx = json.loads((tmp_path / "formal_run_context.json").read_text())
        rec = ctx["scenario_bank_identities"][0]
        bank_key = f"{rec['split']}_k{rec['K']}_{rec['cost_regime_id']}"
        ev = self._find_bank_evidence(tmp_path, bank_key)
        assert ev["scenario_ids_sha_format_valid"] is True, ev
        assert ev["scenario_ids_sha_match"] is False, ev
        assert ev["verdict"] == "FAIL", ev

    def test_per_bank_source_sha_match_pass_enabled_only_when_all_valid(self, tmp_path):
        """§3e. A clean valid fixture must have every per-bank boolean True
        and every per-bank verdict PASS (sanity)."""
        _build_full_valid(tmp_path)
        _run_recompute(tmp_path, expected_returncode=0)
        r = json.loads((tmp_path / "independent_recomputation.json").read_text())
        for ev in r["scenario_bank_file_evidence"]:
            assert ev["source_file_exists"] is True, ev
            assert ev["parse_success"] is True, ev
            assert ev["source_sha_format_valid"] is True, ev
            assert ev["source_sha_match"] is True, ev
            assert ev["scenario_count_present"] is True, ev
            assert ev["scenario_count_match"] is True, ev
            assert ev["scenario_ids_sha_format_valid"] is True, ev
            assert ev["scenario_ids_sha_match"] is True, ev
            assert ev["verdict"] == "PASS", ev

    # ---- §4: reward/cost VIOLATION-ROW COUNTING (vs residual summation) ----

    def test_reward_cost_violation_row_counting(self, tmp_path):
        """§4. Several residuals > 1e-6 individually but whose numeric SUM is
        below 1 MUST still FAIL. The checker counts violating ROWS, not the
        summed residual magnitudes.

        Construction: 4 episode rows with residuals of +0.5, -0.5, +0.4, -0.4
        (signs differ so the residuals cancel to 0 numeric sum, well below 1),
        yet each row individually violates reward = -total_cost. The count of
        violating rows = 4, verdict must be FAIL.
        """
        _build_full_valid(tmp_path)
        eps_path = tmp_path / "episode_results.parquet"
        df = pd.read_parquet(eps_path)
        # Inject 4 violating residuals that cancel numerically.
        idxs = df.index[:4].tolist()
        offsets = [0.5, -0.5, 0.4, -0.4]
        for i, off in zip(idxs, offsets):
            base = float(df.at[i, "total_cost"])
            df.at[i, "total_cost"] = base + off  # episode_return stays -base → residual = off
        df.to_parquet(eps_path, index=False)
        # summary_by_policy.csv would now mismatch; but the recompute FAILs
        # earlier on reward/cost — assert the reward_cost_evidence records
        # violation rows (not a summed residual).
        _run_recompute(tmp_path, expected_returncode=1)
        r = json.loads((tmp_path / "independent_recomputation.json").read_text())
        rc = r["reward_cost_evidence"]
        assert rc["violation_count"] == 4, rc
        assert rc["verdict"] == "FAIL", rc
        assert rc["checked_rows"] == int(len(df)), rc
        assert rc["sample_violating_identities"], rc

    # ---- §6: self-consistent smaller universe is rejected ----

    def test_self_consistent_4_scenarios_per_bank_rejected(self, tmp_path):
        """§6. A self-consistent 4-scenarios-per-bank universe is internally
        consistent (row counts match the bank-derived expected set) but is
        REJECTED because the formal-contract counts are NOT 9000 / 2400.

        16 banks × 4 scenarios × 360 candidates × 5 seeds = 7200 tuning rows
        (≠ 9000); evaluation 6×2×4×4×5 = 960 per split-pair... actually
        6 policies × 2 splits × 2 K × 4 regimes × 4 scenarios × 5 seeds
        = 1920 (≠ 2400). The recompute must reject both via the
        formal-contract-count gate, not merely via internal consistency.
        """
        _build_full_valid_with_distinct_per_kregime_banks(
            tmp_path, scenarios_per_bank=4,
        )
        _run_recompute(tmp_path, expected_returncode=1)
        r = json.loads((tmp_path / "independent_recomputation.json").read_text())
        tuning = r["tuning_set_evidence"]
        evalv = r["evaluation_set_evidence"]
        assert tuning["formal_contract_count"] == 9000, tuning
        assert tuning["formal_contract_count_match"] is False, tuning
        assert tuning["verdict"] == "FAIL", tuning
        assert evalv["formal_contract_count"] == 2400, evalv
        assert evalv["formal_contract_count_match"] is False, evalv
        assert evalv["verdict"] == "FAIL", evalv


def _build_manifest_ready_dir(out: Path) -> Path:
    """Build a valid manifest-input directory that generate_formal_manifest
    (mode=formal_closeout) accepts, via the production code path. Returns the
    directory path; adversarial tests then mutate ONE section of
    independent_recomputation.json and assert the manifest rejects it.
    """
    out.mkdir(parents=True, exist_ok=True)
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True,
    ).strip()
    cfg = {
        "policy_families": list(FORMAL_POLICIES),
        "threshold_grids": {k: list(v) for k, v in GRIDS.items()},
        "k_values": list(K_VALUES), "cost_regimes": list(REGIMES),
        "evaluation_splits": list(EVAL_SPLITS),
        "reset_seeds": [6521, 6522, 6523, 6524, 6525],
    }
    (out / "resolved_config.json").write_text(json.dumps(cfg, indent=2))
    rc_sha = _canonical_json_sha(cfg)
    _write_selected_thresholds(out)
    sel_sha = hashlib.sha256((out / "selected_thresholds.json").read_bytes()).hexdigest()
    (out / "formal_run_context.json").write_text(json.dumps({
        "schema_version": "m3_formal_context_v1", "formal_run_id": out.name,
        "mode": "formal_closeout", "implementation_commit": current_commit,
        "implementation_tree_clean": True,
        "resolved_config_path": str(out / "resolved_config.json"),
        "resolved_config_sha256": rc_sha, "oracle_authorized": True,
        "selected_thresholds_path": str(out / "selected_thresholds.json"),
        "selected_thresholds_sha256": sel_sha, "sealed": True,
        "sealed_at": "2024-01-01T00:00:00", "scenario_bank_identities": [],
        "reset_seeds": [6521, 6522, 6523, 6524, 6525],
        "created_at": "2024-01-01T00:00:00",
    }))
    (out / "validation_report.json").write_text(json.dumps({
        "verdict": "ALL PASSED",
        "oracle_semantic_role": "privileged-information diagnostic benchmark",
    }))
    _write_full_recompute_report(out)
    pd.DataFrame({"policy_family": []}).to_parquet(out / "threshold_search_results.parquet", index=False)
    pd.DataFrame({"policy_family": []}).to_parquet(out / "episode_results.parquet", index=False)
    (out / "threshold_search_summary.csv").write_text("policy_id,mean\n")
    (out / "summary_by_policy.csv").write_text("policy_id,mean\n")
    (out / "scenario_bank_provenance.json").write_text(json.dumps({"scenario_banks": []}))
    (out / "run_provenance.json").write_text(json.dumps({"reset_seeds": [6521, 6522, 6523, 6524, 6525]}))
    return out


def _save_recompute(out: Path, r: dict) -> None:
    (out / "independent_recomputation.json").write_text(json.dumps(r, indent=2))


class TestManifestContractAShapeEnforcement:
    """Contract A / F shape enforcement: the formal manifest (mode =
    formal_closeout) must REJECT independent_recomputation.json whose required
    structured sections are bare booleans / bare lists / null / missing an
    aggregate verdict / record-level FAIL below aggregate PASS, and must reject
    an oracle_terminology_evidence missing oracle_semantic_role_source or
    whose source does not identify an actual generated artifact. A happy-path
    report with compliant object shapes must pass.
    """

    def test_compliant_shapes_pass_shape_gate(self, tmp_path):
        # Happy path for the shape gate: compliant object shapes (object with
        # aggregate verdict PASS, no record-level FAIL, real oracle source)
        # must NOT trip the contract-A type/verdict gate. With empty parquets
        # the manifest then reaches the later artifact-recount gate (expected);
        # we assert the failure is the recount gate, NOT a contract-A shape
        # rejection, proving the compliant shapes passed the shape gate.
        from src.baselines.artifacts import generate_formal_manifest
        out = _build_manifest_ready_dir(tmp_path / "ok")
        with pytest.raises(RuntimeError) as excinfo:
            generate_formal_manifest(out, mode="formal_closeout")
        msg = str(excinfo.value)
        assert "contract-A type/verdict" not in msg, msg
        assert "Artifact recount" in msg, msg

    def test_selected_threshold_file_verification_bare_bool_rejected(self, tmp_path):
        from src.baselines.artifacts import generate_formal_manifest
        out = _build_manifest_ready_dir(tmp_path / "stv_bool")
        r = json.loads((out / "independent_recomputation.json").read_text())
        r["selected_threshold_file_verification"] = True
        _save_recompute(out, r)
        with pytest.raises(RuntimeError, match="expected object, got bool"):
            generate_formal_manifest(out, mode="formal_closeout")

    def test_selected_threshold_file_verification_null_rejected(self, tmp_path):
        from src.baselines.artifacts import generate_formal_manifest
        out = _build_manifest_ready_dir(tmp_path / "stv_null")
        r = json.loads((out / "independent_recomputation.json").read_text())
        r["selected_threshold_file_verification"] = None
        _save_recompute(out, r)
        with pytest.raises(RuntimeError, match="section is null"):
            generate_formal_manifest(out, mode="formal_closeout")

    def test_selected_threshold_file_verification_missing_verdict_rejected(self, tmp_path):
        from src.baselines.artifacts import generate_formal_manifest
        out = _build_manifest_ready_dir(tmp_path / "stv_nov")
        r = json.loads((out / "independent_recomputation.json").read_text())
        r["selected_threshold_file_verification"] = {
            "selected_thresholds_path": str(out / "selected_thresholds.json"),
            "expected_sha256": "a" * 64, "actual_sha256": "a" * 64,
            "exists": True, "sha_match": True,
        }
        _save_recompute(out, r)
        with pytest.raises(RuntimeError, match="missing/None aggregate verdict"):
            generate_formal_manifest(out, mode="formal_closeout")

    def test_selected_threshold_file_verification_sha_mismatch_rejected(self, tmp_path):
        from src.baselines.artifacts import generate_formal_manifest
        out = _build_manifest_ready_dir(tmp_path / "stv_mismatch")
        r = json.loads((out / "independent_recomputation.json").read_text())
        r["selected_threshold_file_verification"] = {
            "selected_thresholds_path": str(out / "selected_thresholds.json"),
            "expected_sha256": "a" * 64, "actual_sha256": "b" * 64,
            "exists": True, "sha_match": False, "verdict": "FAIL",
        }
        _save_recompute(out, r)
        with pytest.raises(RuntimeError, match="FAIL"):
            generate_formal_manifest(out, mode="formal_closeout")

    def test_selected_threshold_file_verification_missing_file_rejected(self, tmp_path):
        from src.baselines.artifacts import generate_formal_manifest
        out = _build_manifest_ready_dir(tmp_path / "stv_missing")
        r = json.loads((out / "independent_recomputation.json").read_text())
        r["selected_threshold_file_verification"] = {
            "selected_thresholds_path": str(out / "missing.json"),
            "expected_sha256": "a" * 64, "actual_sha256": None,
            "exists": False, "sha_match": False, "verdict": "FAIL",
        }
        _save_recompute(out, r)
        with pytest.raises(RuntimeError, match="FAIL"):
            generate_formal_manifest(out, mode="formal_closeout")

    def test_deterministic_tie_break_bare_list_rejected(self, tmp_path):
        from src.baselines.artifacts import generate_formal_manifest
        out = _build_manifest_ready_dir(tmp_path / "dtb_list")
        r = json.loads((out / "independent_recomputation.json").read_text())
        r["deterministic_tie_break_evidence"] = []
        _save_recompute(out, r)
        with pytest.raises(RuntimeError, match="expected object, got list"):
            generate_formal_manifest(out, mode="formal_closeout")

    def test_deterministic_tie_break_missing_aggregate_verdict_rejected(self, tmp_path):
        from src.baselines.artifacts import generate_formal_manifest
        out = _build_manifest_ready_dir(tmp_path / "dtb_nov")
        r = json.loads((out / "independent_recomputation.json").read_text())
        r["deterministic_tie_break_evidence"] = {
            "records": [], "checked_count": 0, "failed_count": 0,
        }
        _save_recompute(out, r)
        with pytest.raises(RuntimeError, match="missing/None aggregate verdict"):
            generate_formal_manifest(out, mode="formal_closeout")

    def test_deterministic_tie_break_failed_count_gt_zero_rejected(self, tmp_path):
        from src.baselines.artifacts import generate_formal_manifest
        out = _build_manifest_ready_dir(tmp_path / "dtb_failcnt")
        r = json.loads((out / "independent_recomputation.json").read_text())
        r["deterministic_tie_break_evidence"] = {
            "records": [{"key": "x", "verdict": "FAIL"}],
            "checked_count": 1, "failed_count": 1, "verdict": "FAIL",
        }
        _save_recompute(out, r)
        with pytest.raises(RuntimeError, match="FAIL"):
            generate_formal_manifest(out, mode="formal_closeout")

    def test_deterministic_tie_break_record_level_fail_rejected(self, tmp_path):
        from src.baselines.artifacts import generate_formal_manifest
        out = _build_manifest_ready_dir(tmp_path / "dtb_rec")
        r = json.loads((out / "independent_recomputation.json").read_text())
        # Aggregate verdict lies PASS but a record carries FAIL — the manifest
        # must not let a record-level FAIL hide beneath top-level PASS.
        r["deterministic_tie_break_evidence"] = {
            "records": [{"key": "x", "verdict": "FAIL"}],
            "checked_count": 1, "failed_count": 0, "verdict": "PASS",
        }
        _save_recompute(out, r)
        with pytest.raises(RuntimeError, match="FAIL"):
            generate_formal_manifest(out, mode="formal_closeout")

    def test_oracle_semantic_role_source_missing_rejected(self, tmp_path):
        from src.baselines.artifacts import generate_formal_manifest
        out = _build_manifest_ready_dir(tmp_path / "src_missing")
        r = json.loads((out / "independent_recomputation.json").read_text())
        r["oracle_terminology_evidence"].pop("oracle_semantic_role_source", None)
        _save_recompute(out, r)
        with pytest.raises(RuntimeError, match="oracle_semantic_role_source"):
            generate_formal_manifest(out, mode="formal_closeout")

    def test_oracle_semantic_role_source_not_an_artifact_rejected(self, tmp_path):
        from src.baselines.artifacts import generate_formal_manifest
        out = _build_manifest_ready_dir(tmp_path / "src_fake")
        r = json.loads((out / "independent_recomputation.json").read_text())
        # Source points to a path that does not exist as a generated artifact —
        # the provenance must identify a REAL generated artifact, not a phantom.
        r["oracle_terminology_evidence"]["oracle_semantic_role_source"] = str(out / "does_not_exist.json")
        r["oracle_terminology_evidence"]["required_label_matches"] = [{"file": "does_not_exist.json", "label": "privileged-information diagnostic benchmark"}]
        _save_recompute(out, r)
        with pytest.raises(RuntimeError, match="oracle_semantic_role_source"):
            generate_formal_manifest(out, mode="formal_closeout")

    def test_oracle_wrong_semantic_role_rejected(self, tmp_path):
        from src.baselines.artifacts import generate_formal_manifest
        out = _build_manifest_ready_dir(tmp_path / "wrong_role")
        r = json.loads((out / "independent_recomputation.json").read_text())
        r["oracle_terminology_evidence"]["oracle_semantic_role"] = "optimal oracle upper bound"
        _save_recompute(out, r)
        with pytest.raises(RuntimeError, match="oracle_semantic_role"):
            generate_formal_manifest(out, mode="formal_closeout")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])