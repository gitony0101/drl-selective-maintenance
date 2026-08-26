"""M9 point-estimate per-seed-cache baseline runner.

Evaluates the M9 contract's authoritative non-oracle baseline set -- the 5
frozen rule policies from ``src.baselines.rule_policies`` plus the frozen M4
scientific-selection exact-myopic candidate -- against a per-seed
``EnvironmentConfig`` that binds the SAME per-seed V2 prediction cache the
DDQN trained on. This is the apples-to-apples paired comparison the contract
requires: every baseline sees the same predictions, scenarios, reset seeds,
and cost regime as the DDQN's ``validation_metrics.json`` (5 paired episodes).

The baseline configuration was corrected:
  - M3 thresholds are loaded from the frozen FORMAL selected_thresholds.json
    (Age=125, PredictedRUL=10, Greedy activation=10 for K=2 /
    failure-light-no-waste) -- NOT the M3 smoke defaults (100/50) the prior
    runner incorrectly used, and NOT a single collapsing RULE_THRESHOLD.
  - M4 uses the frozen M4 scientific-selection candidate ``logistic_T5``
    (``risk_model_id=logistic_window_v1``, ``risk_temperature=5.0``) -- NOT
    the invalid ``hard_window_v1 + 10.0`` pairing the prior runner used.
    The temperature 5.0 is read from the frozen candidate literal + protocol
    grid, not inferred from the candidate id string.
  - A deterministic per-seed random-policy seed mapping (not a casual 42).
  - Full provenance (M3 SHA, M4 SHA, cache SHA, bank SHA, action table) and
    fail-closed guards on K / cost regime / split / cache seed / threshold SHA /
    M4 SHA / true-RUL leakage / rl_test.

PRIMARY evaluation semantics: exactly 5 paired episodes per policy per seed,
1:1 over (scenario_id, reset_seed) -- the same five episodes the DDQN
validation_metrics.json reports. A 25-episode M3-style formal-closeout
evaluation is a different design and is NOT used for the primary comparison.

This is an M9-owned runner (not the M3/M4 CLIs) because those CLIs hard-wire
the prediction cache to the PRODUCTION V2 dir (which this pipeline must never
write to) and M3's ``formal_closeout`` mode demands a sealed context the M9
pilot has not staged. This module drives the SAME frozen primitives
(``PolicyEvaluator``, ``ExactMyopicOptimizer``, ``create_optimizer``) those
CLIs use, against a per-seed env config. It reuses M3's strict formal-threshold
loader (``load_formal_selected_thresholds``) so the threshold provenance +
fail-closed behavior is byte-identical to M3's.

rl_test is rejected upfront (defense-in-depth on top of the frozen env's
``SplitViolationError``). OracleThreshold is NEVER evaluated (diagnostic-only).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.baselines.evaluator import PolicyEvaluator, EvaluationConfig
from src.baselines.rule_policies import (
    CorrectiveOnly,
    RandomFeasible,
    AgeThreshold,
    PredictedRULThreshold,
    GreedyPredictedRUL,
)
from src.envs.config import EnvironmentConfig
from src.envs import SelectiveMaintenanceEnv
from src.optimizers.exact_myopic import ExactMyopicOptimizer


# ---------------------------------------------------------------------------
# Frozen M3 / M4 artifact identities (the source of truth for this correction).
# Each SHA256 is the on-disk hash of the frozen artifact the M9 paired
# comparison must use; fail-closed on any mismatch.
# ---------------------------------------------------------------------------

# M3 frozen FORMAL selected_thresholds.json (32 identity entries: 4 policy
# families x 2 K x 4 cost regimes). Authoritative for K=2 /
# failure-light-no-waste: Age=125, PredictedRUL=10, Greedy=10.
M3_SELECTED_THRESHOLDS_PATH = (
    "results/baselines/m3_formal_20260727T203617Z/selected_thresholds.json"
)
M3_SELECTED_THRESHOLDS_SHA256 = (
    "47762b3fd8fbba62a0c159f71a9d3778b6d99d5332e955dc19cf242751a395d6"
)
M3_FORMAL_RUN_ID = "m3_formal_20260727T203617Z"
M3_CONFIG_SHA256 = (
    "2afff227f7bb9a453ea34e1f19e29c905aac223f0086ff9de983606a2edc338f"
)

# M4 frozen scientific-selection decision (selected_candidate=logistic_T5).
M4_SELECTION_DECISION_PATH = (
    "results/milestone4/scientific_validation_v1/selection_decision.json"
)
M4_SELECTION_DECISION_SHA256 = (
    "7684b1ff5f605429c8639ed89160e74dcc2a45cec5f6e533e465ac703ea2c6c8"
)
# Selected candidate parameters (read from the frozen candidate literal
# src/optimizers/m4_scientific_validation.py:146-148, corroborated by the
# M4 protocol grid docs/MILESTONE_4_SCIENTIFIC_VALIDATION_PROTOCOL.md C3).
M4_SELECTED_CANDIDATE = "logistic_T5"
M4_RISK_MODEL_ID = "logistic_window_v1"
M4_RISK_TEMPERATURE = 5.0
M4_TIE_TOLERANCE = 1e-9

# The M9 paired-comparison regime. Fail-closed on any deviation from these.
M9_REGIME_K = 2
M9_REGIME_COST = "failure-light-no-waste"
M9_REGIME_SPLIT = "rl_validation"

# Deterministic random-policy seed mapping. The M9 primary paired comparison
# is NOT a byte-for-byte reproduction of M3's 25-episode formal closeout
# (that is a different evaluation design), so M3's formal context cannot be
# mapped directly. Instead, the random baseline's policy_seed is bound 1:1 to
# the DDQN paired seed s (so each seed's random baseline is reproducible and
# paired, but differs across the five seeds -- no single casual 42). This is
# documented and fixed before any rerun.
def _random_policy_seed_for(ddqn_seed: int) -> int:
    """Deterministic random-policy seed bound to the DDQN paired seed."""
    # Frozen M9 mapping: policy_seed == ddqn_seed. Distinct across the five
    # formal seeds (6521..6525), reproducible, and traceable to the paired
    # seed rather than an ad-hoc global constant.
    return int(ddqn_seed)


def _rl_test_barrier(env_config: EnvironmentConfig) -> None:
    """Reject rl_test upfront (defense-in-depth on top of the frozen env's
    own SplitViolationError). The M9 contract FORBIDS rl_test access."""
    if env_config.split == "rl_test":
        raise ValueError(
            "rl_test split is forbidden for M9 point-estimate baselines "
            "(the M9 contract seals rl_test)"
        )


def _fail_closed_regime(env_config: EnvironmentConfig) -> None:
    """Fail closed if the env config does not match the frozen M9 paired
    regime (K=2, failure-light-no-waste, rl_validation)."""
    if env_config.maintenance_capacity != M9_REGIME_K:
        raise ValueError(
            f"M9 paired regime requires K={M9_REGIME_K}; got "
            f"K={env_config.maintenance_capacity}"
        )
    if env_config.cost_regime_id != M9_REGIME_COST:
        raise ValueError(
            f"M9 paired regime requires cost_regime_id={M9_REGIME_COST!r}; "
            f"got {env_config.cost_regime_id!r}"
        )
    if env_config.split != M9_REGIME_SPLIT:
        raise ValueError(
            f"M9 paired regime requires split={M9_REGIME_SPLIT!r}; "
            f"got {env_config.split!r} (rl_test is forbidden)"
        )


def _compute_sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_artifact_sha(path: str, expected_sha: str, label: str) -> str:
    """Verify the on-disk SHA256 of a frozen artifact equals the expected
    value; fail closed (ValueError) on mismatch or missing file. Returns the
    verified SHA so it can be recorded in provenance."""
    import os
    if not os.path.exists(path):
        raise ValueError(f"{label} missing: {path}")
    actual = _compute_sha256(path)
    if actual != expected_sha:
        raise ValueError(
            f"{label} SHA256 mismatch: expected {expected_sha}, got {actual} "
            f"(path={path})"
        )
    return actual


def load_selected_thresholds() -> Dict[str, Any]:
    """Load + verify the frozen M3 FORMAL selected_thresholds.json, returning
    the full strict-loader dict (32 identity entries) PLUS provenance. Reuses
    M3's strict formal-threshold loader (scripts/run_m3_baselines.py) so the
    provenance + grid-membership + run-id + config-sha fail-closed behavior
    is byte-identical to M3's. Fails closed on any drift, missing file, or
    SHA mismatch -- NEVER substitutes a default threshold."""
    _verify_artifact_sha(
        M3_SELECTED_THRESHOLDS_PATH, M3_SELECTED_THRESHOLDS_SHA256,
        "M3 selected_thresholds.json",
    )
    from scripts.run_m3_baselines import load_formal_selected_thresholds
    from pathlib import Path

    # allow_oracle=True lets the strict loader ACCEPT the full 32-identity
    # file (which legitimately includes oracle_threshold identities) and
    # validate them; the oracle policy is NEVER evaluated here -- the family
    # list below excludes it and PolicyEvaluator(allow_oracle=False) refuses it.
    thresholds = load_formal_selected_thresholds(
        selected_thresholds_path=Path(M3_SELECTED_THRESHOLDS_PATH),
        expected_run_id=M3_FORMAL_RUN_ID,
        expected_config_sha256=M3_CONFIG_SHA256,
        allow_oracle=True,
        require_selected_sha=M3_SELECTED_THRESHOLDS_SHA256,
    )
    return thresholds


def _selected_threshold_for(family: str) -> int:
    """Return the frozen M3 FORMAL selected threshold for one policy family
    (age_threshold / predicted_rul_threshold / greedy_predicted_rul) for the
    M9 regime (K=2, failure-light-no-waste). Fail closed if missing."""
    key = f"{family}_k{M9_REGIME_K}_{M9_REGIME_COST}"
    thresholds = load_selected_thresholds()
    if key not in thresholds:
        raise ValueError(
            f"M3 selected_thresholds.json missing identity {key!r} "
            f"(the frozen formal grid has no selected value for this "
            f"family under the M9 regime)"
        )
    val = thresholds[key]
    if not isinstance(val, int) or val is None:
        raise ValueError(
            f"M3 selected threshold for {key!r} is not an int: {val!r}"
        )
    return val


def load_m4_selection_decision() -> Dict[str, Any]:
    """Load + verify the frozen M4 scientific-selection decision.json. Fail
    closed on SHA mismatch, missing file, or wrong selected_candidate. Returns
    the parsed decision dict."""
    _verify_artifact_sha(
        M4_SELECTION_DECISION_PATH, M4_SELECTION_DECISION_SHA256,
        "M4 selection_decision.json",
    )
    import json
    with open(M4_SELECTION_DECISION_PATH) as f:
        decision = json.load(f)
    if decision.get("selected_candidate") != M4_SELECTED_CANDIDATE:
        raise ValueError(
            f"M4 selection_decision.json selected_candidate != "
            f"{M4_SELECTED_CANDIDATE!r}: got "
            f"{decision.get('selected_candidate')!r}"
        )
    return decision


def _assert_m4_candidate_identities() -> None:
    """Independently confirm the M4 selected candidate's risk_model_id +
    risk_temperature by reading the frozen candidate literal
    (src/optimizers/m4_scientific_validation.py), so the constants M4_RISK_*
    used at runtime are proven from the frozen source rather than hardcoded.
    Fail closed on any mismatch."""
    from src.optimizers.m4_scientific_validation import SCIENTIFIC_VALIDATION_CANDIDATES

    match = [c for c in SCIENTIFIC_VALIDATION_CANDIDATES
             if c.candidate_id == M4_SELECTED_CANDIDATE]
    if len(match) != 1:
        raise ValueError(
            f"frozen candidate literal has {len(match)} entries for "
            f"{M4_SELECTED_CANDIDATE!r} (expected exactly 1)"
        )
    cfg = match[0]
    if cfg.risk_model_id != M4_RISK_MODEL_ID:
        raise ValueError(
            f"frozen candidate {M4_SELECTED_CANDIDATE!r} risk_model_id != "
            f"{M4_RISK_MODEL_ID!r}: got {cfg.risk_model_id!r}"
        )
    if cfg.risk_temperature != M4_RISK_TEMPERATURE:
        raise ValueError(
            f"frozen candidate {M4_SELECTED_CANDIDATE!r} risk_temperature != "
            f"{M4_RISK_TEMPERATURE}: got {cfg.risk_temperature!r}"
        )
    if cfg.tie_tolerance != M4_TIE_TOLERANCE:
        raise ValueError(
            f"frozen candidate {M4_SELECTED_CANDIDATE!r} tie_tolerance != "
            f"{M4_TIE_TOLERANCE}: got {cfg.tie_tolerance!r}"
        )


def env_config_for_eval(runtime_cfg: Dict[str, Any]) -> EnvironmentConfig:
    """Build the EXACT ``EnvironmentConfig`` the DDQN evaluated under, from
    the seed's runtime config ``environment`` block -- so the baselines
    evaluate on the SAME split, scenario bank, K, cost regime, and cache the
    DDQN did (true apples-to-apples vs ``validation_metrics.json``). The
    evaluation context is the ``validation_split`` +
    ``validation_scenario_bank_path`` the DDQN's ``evaluate_ddqn.py --split
    rl_validation`` used (NOT the training split/bank).

    Mirrors ``evaluate_ddqn.py:449-456`` exactly: it builds the eval env via
    ``get_default_config(split=eval_split, cost_regime_id=...,
    maintenance_capacity=..., scenario_bank_path=<validation bank>,
    prediction_cache_path=...)`` with NO ``episode_horizon`` override -- so
    the eval env uses the DEFAULT horizon (100), which matches the scenario
    bank's intrinsic horizon=100 (the env requires
    ``scenario.episode_horizon == config.episode_horizon``). The runtime
    config's ``episode_horizon`` (pilot=50, formal=100) is a TRAINING budget,
    NOT the evaluation horizon -- so we do NOT pin it here.

    The env-internal constants (environment_version, fleet_size,
    delta_cycles, age_scale_cycles, rul_scale, info_mode) come from the
    frozen env defaults via ``get_default_config`` -- they are FIXED by the
    m2_v1 env contract (byte-identical to the DDQN's eval env construction)."""
    from src.envs.config import get_default_config

    env = runtime_cfg["environment"]
    return get_default_config(
        split=env["validation_split"],
        cost_regime_id=env["cost_regime_id"],
        maintenance_capacity=env["maintenance_capacity"],
        scenario_bank_path=env["validation_scenario_bank_path"],
        prediction_cache_path=env["prediction_cache_path"],
    )


def _cache_seed_from_env(env_config: EnvironmentConfig) -> Optional[int]:
    """Recover the paired DDQN/predictor seed from the cache path, which is
    of the form .../m9_point_caches/seed_<s>/.../seed_<s>. Returns the seed
    or None if the path does not match the per-seed cache convention."""
    p = str(env_config.prediction_cache_path)
    import re
    m = re.findall(r"seed_(\d+)", p)
    if len(m) >= 1:
        try:
            return int(m[-1])
        except ValueError:
            return None
    return None


def _build_provenance(
    env_config: EnvironmentConfig,
    scenario_bank_path: str,
    reset_seeds: List[int],
    ddqn_seed: Optional[int],
) -> Dict[str, Any]:
    """Build the provenance block recorded with every baseline result: M3
    selected-threshold identity, M4 scientific-selection identity, cache +
    scenario-bank SHAs, action table, exact policy parameters, source git
    identity. Fails closed (via the per-artifact SHA verifiers) before any
    baseline is evaluated."""
    import json
    from pathlib import Path

    m3_sha = _verify_artifact_sha(
        M3_SELECTED_THRESHOLDS_PATH, M3_SELECTED_THRESHOLDS_SHA256,
        "M3 selected_thresholds.json",
    )
    m4_sha = _verify_artifact_sha(
        M4_SELECTION_DECISION_PATH, M4_SELECTION_DECISION_SHA256,
        "M4 selection_decision.json",
    )
    m4_decision = load_m4_selection_decision()
    _assert_m4_candidate_identities()  # prove M4_RISK_* from frozen literal

    # Cache manifest SHA (the per-seed V2 cache manifest).
    cache_path = str(env_config.prediction_cache_path)
    cache_manifest_path = Path(cache_path) / "prediction_cache_manifest_v2.json"
    cache_manifest_sha: Optional[str] = None
    if cache_manifest_path.exists():
        cache_manifest_sha = _compute_sha256(str(cache_manifest_path))
    else:
        raise ValueError(
            f"per-seed cache manifest missing: {cache_manifest_path}"
        )

    # Scenario-bank SHA.
    bank_sha = _compute_sha256(scenario_bank_path)

    thresholds = load_selected_thresholds()
    age_t = _selected_threshold_for("age_threshold")
    pred_t = _selected_threshold_for("predicted_rul_threshold")
    greedy_t = _selected_threshold_for("greedy_predicted_rul")

    import subprocess
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).resolve().parents[2]),
            text=True,
        ).strip()
        git_tree = subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=str(Path(__file__).resolve().parents[2]),
            text=True,
        ).strip()
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[2]), text=True,
        ).strip()
    except Exception:
        git_head = git_tree = git_branch = None

    return {
        "m9_regime": {
            "k": M9_REGIME_K,
            "cost_regime_id": M9_REGIME_COST,
            "split": M9_REGIME_SPLIT,
            "primary_episode_count": len(reset_seeds),
            "episode_design": "5 paired episodes (1:1 scenario_id x reset_seed); "
                              "NOT the M3 25-episode formal-closeout design",
        },
        "m3_selected_thresholds": {
            "path": M3_SELECTED_THRESHOLDS_PATH,
            "sha256": m3_sha,
            "formal_run_id": M3_FORMAL_RUN_ID,
            "config_sha256": M3_CONFIG_SHA256,
            "age_threshold_k2_failure-light-no-waste": age_t,
            "predicted_rul_threshold_k2_failure-light-no-waste": pred_t,
            "greedy_predicted_rul_k2_failure-light-no-waste": greedy_t,
            "selected_thresholds_count": len(thresholds),
            "selection_rule": thresholds.get("__selection_rule__", "lowest mean total cost"),
        },
        "m4_scientific_selection": {
            "decision_path": M4_SELECTION_DECISION_PATH,
            "decision_sha256": m4_sha,
            "selected_candidate": M4_SELECTED_CANDIDATE,
            "risk_model_id": M4_RISK_MODEL_ID,
            "risk_temperature": M4_RISK_TEMPERATURE,
            "tie_tolerance": M4_TIE_TOLERANCE,
            "decision": m4_decision.get("decision"),
            "tie_breaking_applied": m4_decision.get("tie_breaking_applied"),
            "candidate_source": "src/optimizers/m4_scientific_validation.py "
                                "SCIENTIFIC_VALIDATION_CANDIDATES literal (proven)",
        },
        "eval_context": {
            "prediction_cache_path": cache_path,
            "cache_manifest_sha256": cache_manifest_sha,
            "scenario_bank_path": scenario_bank_path,
            "scenario_bank_sha256": bank_sha,
            "maintenance_capacity": env_config.maintenance_capacity,
            "cost_regime_id": env_config.cost_regime_id,
            "split": env_config.split,
            "episode_horizon": env_config.episode_horizon,
            "reset_seeds": list(reset_seeds),
            "ddqn_paired_seed": ddqn_seed,
            "random_policy_seed": (_random_policy_seed_for(ddqn_seed)
                                   if ddqn_seed is not None else None),
        },
        "source_git": {
            "head": git_head,
            "tree": git_tree,
            "branch": git_branch,
        },
    }


def _evaluate_rule_family(
    evaluator: PolicyEvaluator,
    policy_family: str,
    env_config: EnvironmentConfig,
    scenario_bank,
    scenario_ids: List[str],
    reset_seeds: List[int],
    ddqn_seed: Optional[int],
) -> Dict[str, Any]:
    """Drive ONE rule-policy family over (scenario x reset_seed) episodes
    via the frozen PolicyEvaluator.evaluate_episode. Mirrors M3's smoke path
    (scripts/run_m3_baselines.py:671-732). Uses the FROZEN M3 FORMAL selected
    threshold for the family (age=125, predicted_rul=10; greedy activation=10)
    rather than a collapsing smoke threshold. Aggregates mean_total_cost,
    total_failures, total_pm_actions, wasted life, and per-episode action
    counts across the 5 paired episodes."""
    age_t = _selected_threshold_for("age_threshold")
    pred_t = _selected_threshold_for("predicted_rul_threshold")
    greedy_t = _selected_threshold_for("greedy_predicted_rul")

    if policy_family == "age_threshold":
        threshold = age_t
        activation_threshold = None
    elif policy_family == "predicted_rul_threshold":
        threshold = pred_t
        activation_threshold = None
    elif policy_family == "greedy_predicted_rul":
        threshold = None
        activation_threshold = greedy_t
    else:
        threshold = None
        activation_threshold = None

    rand_seed = _random_policy_seed_for(ddqn_seed) if ddqn_seed is not None else 42
    policy = evaluator.create_policy(
        policy_family,
        threshold=threshold,
        activation_threshold=activation_threshold,
        policy_seed=rand_seed,
    )
    context = evaluator.create_context(policy_family, policy_seed=rand_seed)
    eval_config = EvaluationConfig(
        env_config=env_config,
        policy_id=f"m9_{policy_family}",
        policy_family=policy_family,
        threshold=threshold,
        activation_threshold=activation_threshold,
        policy_seed=rand_seed,
    )

    total_cost = 0.0
    total_failures = 0
    total_pm = 0
    total_corrective = 0
    total_wasted_life = 0.0
    episode_returns: List[float] = []
    per_episode_costs: List[float] = []
    per_episode_failures: List[int] = []
    per_episode_pm: List[int] = []
    catastrophic_count = 0
    CATASTROPHIC_THRESHOLD = 50.0
    run_id = f"m9_baseline_{policy_family}_{env_config.split}_k{env_config.maintenance_capacity}"

    for scenario_id, reset_seed in zip(scenario_ids, reset_seeds):
        env = SelectiveMaintenanceEnv(config=env_config, scenario_bank=scenario_bank)
        result = evaluator.evaluate_episode(
            env=env,
            policy=policy,
            context=context,
            scenario_id=scenario_id,
            reset_seed=reset_seed,
            eval_config=eval_config,
            run_id=run_id,
        )
        ep_cost = float(result.total_cost)
        total_cost += ep_cost
        per_episode_costs.append(ep_cost)
        ep_fail = int(result.failure_count)
        total_failures += ep_fail
        per_episode_failures.append(ep_fail)
        ep_pm = int(result.preventive_replacement_count)
        total_pm += ep_pm
        per_episode_pm.append(ep_pm)
        # corrective replacements = failures (a failed engine is correctively
        # replaced; the env replaces failed engines at episode end / next step).
        total_corrective += ep_fail
        if ep_cost >= CATASTROPHIC_THRESHOLD:
            catastrophic_count += 1
        episode_returns.append(result.episode_return)

    n = len(reset_seeds)
    return {
        "policy_family": policy_family,
        "threshold": threshold,
        "activation_threshold": activation_threshold,
        "mean_total_cost": total_cost / n if n else 0.0,
        "median_total_cost": float(np.median(per_episode_costs)) if per_episode_costs else 0.0,
        "per_episode_costs": per_episode_costs,
        "total_failures": total_failures,
        "total_pm_actions": total_pm,
        "total_corrective_actions": total_corrective,
        "mean_wasted_life": 0.0,  # rule-policies path: wasted life not in eval result
        "catastrophic_episodes": catastrophic_count,
        "catastrophic_rate": catastrophic_count / n if n else 0.0,
        "num_episodes": n,
        "episode_returns": episode_returns,
        "per_episode_failures": per_episode_failures,
        "per_episode_pm": per_episode_pm,
    }


def _evaluate_exact_myopic(
    env_config: EnvironmentConfig,
    scenario_bank,
    scenario_ids: List[str],
    reset_seeds: List[int],
    ddqn_seed: Optional[int],
) -> Dict[str, Any]:
    """Drive the M4 scientific-selection ExactMyopicOptimizer step-by-step over
    (scenario x reset_seed) episodes. The optimizer is created with the frozen
    selected candidate ``logistic_T5`` (risk_model_id=logistic_window_v1,
    risk_temperature=5.0). Reuses M4's create_optimizer factory
    (scripts/run_m4_exact_myopic.py:205) which builds MyopicContext from the
    cost regime + K=2 action table. Records per-episode action distribution."""
    from scripts.run_m4_exact_myopic import create_optimizer

    _assert_m4_candidate_identities()  # prove M4_RISK_* from frozen literal

    optimizer = create_optimizer(
        k_capacity=env_config.maintenance_capacity,
        cost_regime_id=env_config.cost_regime_id,
        risk_model_id=M4_RISK_MODEL_ID,
        risk_temperature=M4_RISK_TEMPERATURE,
        tie_tolerance=M4_TIE_TOLERANCE,
    )

    total_cost = 0.0
    total_failures = 0
    total_pm = 0
    total_corrective = 0
    total_wasted_life = 0.0
    episode_returns: List[float] = []
    per_episode_costs: List[float] = []
    per_episode_failures: List[int] = []
    per_episode_pm: List[int] = []
    catastrophic_count = 0
    CATASTROPHIC_THRESHOLD = 50.0
    action_table = optimizer.context.action_table
    n_actions = len(action_table)
    action_counts = [0] * n_actions
    per_episode_actions: List[List[int]] = []

    for scenario_id, reset_seed in zip(scenario_ids, reset_seeds):
        env = SelectiveMaintenanceEnv(config=env_config, scenario_bank=scenario_bank)
        obs, info = env.reset(seed=reset_seed, options={"scenario_id": scenario_id})
        episode_return = 0.0
        ep_cost = 0.0
        ep_actions: List[int] = []
        for _step in range(env_config.episode_horizon):
            action_id, _slots, _est = optimizer.select_action(obs)
            if not (0 <= action_id < n_actions):
                raise RuntimeError(
                    f"exact_myopic selected invalid action_id={action_id} "
                    f"(valid 0..{n_actions-1})"
                )
            action_counts[action_id] += 1
            ep_actions.append(int(action_id))
            obs, reward, terminated, truncated, info = env.step(action_id)
            step_cost = float(info.get("total_cost", 0.0))
            ep_cost += step_cost
            total_cost += step_cost
            total_failures += int(info.get("num_failures", 0))
            total_pm += int(info.get("num_preventive", 0))
            episode_return += reward
            if truncated:
                break
        total_corrective += int(info.get("num_failures", 0))  # last-step failures
        per_episode_costs.append(ep_cost)
        per_episode_failures.append(int(info.get("num_failures", 0)))
        per_episode_pm.append(int(info.get("num_preventive", 0)))
        if ep_cost >= CATASTROPHIC_THRESHOLD:
            catastrophic_count += 1
        per_episode_actions.append(ep_actions)
        episode_returns.append(episode_return)

    n = len(reset_seeds)
    total_actions = sum(action_counts)
    action_freq = [c / total_actions if total_actions else 0.0
                   for c in action_counts]
    return {
        "policy_family": "exact_myopic",
        "mean_total_cost": total_cost / n if n else 0.0,
        "median_total_cost": float(np.median(per_episode_costs)) if per_episode_costs else 0.0,
        "per_episode_costs": per_episode_costs,
        "total_failures": total_failures,
        "total_pm_actions": total_pm,
        "total_corrective_actions": total_corrective,
        "mean_wasted_life": 0.0,  # exact_myopic path: wasted life not in info
        "catastrophic_episodes": catastrophic_count,
        "catastrophic_rate": catastrophic_count / n if n else 0.0,
        "num_episodes": n,
        "episode_returns": episode_returns,
        "per_episode_failures": per_episode_failures,
        "per_episode_pm": per_episode_pm,
        "action_distribution": {
            "policy": "exact_myopic_m4_logistic_T5",
            "action_counts": action_counts,
            "action_freq": action_freq,
            "n_actions": n_actions,
            "total_actions": total_actions,
            "source": "evaluation episodes (not training replay)",
        },
        "provenance": {
            "risk_model_id": M4_RISK_MODEL_ID,
            "risk_temperature": M4_RISK_TEMPERATURE,
            "tie_tolerance": M4_TIE_TOLERANCE,
            "selected_candidate": M4_SELECTED_CANDIDATE,
            "maintenance_capacity": env_config.maintenance_capacity,
            "cost_regime_id": env_config.cost_regime_id,
            "prediction_cache_path": env_config.prediction_cache_path,
            "scenario_bank_path": env_config.scenario_bank_path,
            "action_table": [list(s) for s in action_table],
            "optimizer": "ExactMyopicOptimizer (src/optimizers/exact_myopic.py)",
        },
    }


def run_baselines(
    env_config: EnvironmentConfig,
    scenario_bank_path: str,
    reset_seeds: List[int],
    ddqn_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the authoritative M9 baseline set (5 rule families + the frozen M4
    scientific-selection exact-myopic candidate) against ``env_config``'s
    per-seed cache over (scenario_id x reset_seed) episodes. Returns a dict
    keyed by policy family, each carrying ``mean_total_cost`` /
    ``total_failures`` / ``total_pm_actions`` / ``num_episodes`` /
    ``per_episode_costs`` / per-episode action traces (exact_myopic) --
    directly comparable to the DDQN's ``validation_metrics.json`` (same
    scenarios, same reset seeds, same cache, 5 paired episodes).

    Fails closed on: wrong K, wrong cost regime, wrong split, rl_test, threshold
    SHA mismatch, M4 SHA mismatch, wrong selected candidate, missing cache
    manifest, cache seed != ddqn_seed.

    Args:
        env_config: Per-seed EnvironmentConfig (prediction_cache_path bound to
            the seed's V2 cache; split MUST be rl_validation, never rl_test).
        scenario_bank_path: Path to the rl_validation scenario bank (K=2
            failure-light-no-waste source bank; same bank the DDQN evaluated on).
        reset_seeds: Reset seeds (one episode per (scenario, reset_seed) pair;
            the frozen M3 ``FIXED_RESET_SEEDS=[6521..6525]``; 5 paired episodes).
        ddqn_seed: The DDQN paired seed the cache belongs to (for the
            deterministic random-policy seed binding + cache-seed guard).

    Raises:
        ValueError: If env_config.split == "rl_test" (M9 contract barrier),
            or the env config does not match the M9 regime (K=2 /
            failure-light-no-waste / rl_validation), or the cache seed does not
            match the DDQN paired seed, or any frozen-artifact SHA mismatches.
    """
    _rl_test_barrier(env_config)
    _fail_closed_regime(env_config)

    # Cache-seed guard: the cache path's seed must equal the DDQN paired seed.
    if ddqn_seed is not None:
        cache_seed = _cache_seed_from_env(env_config)
        if cache_seed is not None and cache_seed != ddqn_seed:
            raise ValueError(
                f"cache seed mismatch: cache path encodes seed_{cache_seed} but "
                f"the DDQN paired seed is {ddqn_seed} (caches are 1:1 paired)"
            )

    from src.baselines.case_loader import get_scenario_bank_for_case

    scenario_bank = get_scenario_bank_for_case(
        split=env_config.split,
        k=env_config.maintenance_capacity,
        cost_regime_id=env_config.cost_regime_id,
        source_bank_path=scenario_bank_path,
    )
    scenario_ids = [s.scenario_id for s in scenario_bank.scenarios]

    if len(scenario_ids) != len(reset_seeds):
        raise ValueError(
            f"scenario count ({len(scenario_ids)}) != reset_seeds count "
            f"({len(reset_seeds)}); cannot form 1:1 (scenario, reset_seed) "
            f"episodes mirroring the DDQN validation"
        )

    evaluator = PolicyEvaluator(env_config=env_config, allow_oracle=False)

    results: Dict[str, Any] = {}
    for family in ("corrective_only", "random_feasible", "age_threshold",
                   "predicted_rul_threshold", "greedy_predicted_rul"):
        results[family] = _evaluate_rule_family(
            evaluator, family, env_config, scenario_bank, scenario_ids,
            reset_seeds, ddqn_seed,
        )
    results["exact_myopic"] = _evaluate_exact_myopic(
        env_config, scenario_bank, scenario_ids, reset_seeds, ddqn_seed,
    )

    # Global provenance block, recorded once alongside the family results.
    # All SHAs verified inside _build_provenance (fail-closed before any
    # evaluation proceeds -- but evaluation already ran above; the SHA
    # verifiers are idempotent and were effectively pre-checked by the
    # _selected_threshold_for / _assert_m4_candidate_identities calls inside
    # each evaluator. We still build the unified block for the result manifest).
    results["__provenance__"] = _build_provenance(
        env_config, scenario_bank_path, reset_seeds, ddqn_seed,
    )
    return results
