"""
Threshold tuning for Milestone 3 Baselines.

Searches threshold candidates on rl_validation split only.
Selects thresholds using frozen objective and deterministic tie-break.

Tuning split: rl_validation ONLY
Do not tune on predictor_train
Do not access rl_test
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..envs import EnvironmentConfig
from .evaluator import PolicyEvaluator, EvaluationConfig, EpisodeResult
from .protocols import PolicyContext


@dataclass
class ThresholdCandidate:
    """A threshold candidate result."""

    policy_family: str
    threshold: float
    k_capacity: int
    cost_regime_id: str
    mean_total_cost: float
    total_failures: int
    mean_wasted_life_cost: float
    episode_count: int


@dataclass
class SelectedThreshold:
    """Selected threshold for a policy/K/regime combination."""

    policy_family: str
    threshold: float
    k_capacity: int
    cost_regime_id: str
    mean_total_cost: float
    total_failures: int
    mean_wasted_life_cost: float
    episode_count: int
    tie_break_reason: str = "best"


# Frozen threshold grids from M3 contract
AGE_THRESHOLDS = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
PREDICTED_RUL_THRESHOLDS = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100]
GREEDY_ACTIVATION_THRESHOLDS = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100]
ORACLE_THRESHOLDS = [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50]

# Policies that need tuning
THRESHOLD_POLICIES = {
    "age_threshold": AGE_THRESHOLDS,
    "predicted_rul_threshold": PREDICTED_RUL_THRESHOLDS,
    "greedy_predicted_rul": GREEDY_ACTIVATION_THRESHOLDS,
    "oracle_threshold": ORACLE_THRESHOLDS,
}

# Policies that don't need tuning
NON_TUNED_POLICIES = {"corrective_only", "random_feasible"}


def get_threshold_grid(policy_family: str) -> List[float]:
    """
    Get frozen threshold grid for a policy family.

    Args:
        policy_family: One of age_threshold, predicted_rul_threshold,
                       greedy_predicted_rul, oracle_threshold

    Returns:
        List of threshold candidates

    Raises:
        ValueError: If policy family not in threshold policies
    """
    if policy_family not in THRESHOLD_POLICIES:
        raise ValueError(
            f"Policy family {policy_family} not in threshold policies. "
            f"Valid: {list(THRESHOLD_POLICIES.keys())}"
        )
    return THRESHOLD_POLICIES[policy_family]


def select_best_threshold(
    candidates: List[ThresholdCandidate],
) -> SelectedThreshold:
    """
    Select best threshold using frozen objective and deterministic tie-break.

    Tie-break order:
        1. Lower mean total cost
        2. Fewer total failures
        3. Lower mean wasted-life cost
        4. Lower threshold value (deterministic numeric order)

    Args:
        candidates: List of ThresholdCandidate instances

    Returns:
        SelectedThreshold with best threshold

    Raises:
        ValueError: If no candidates provided
    """
    if not candidates:
        raise ValueError("No threshold candidates provided")

    # Sort by tie-break order
    # Lower is better for all metrics
    sorted_candidates = sorted(
        candidates,
        key=lambda c: (
            c.mean_total_cost,
            c.total_failures,
            c.mean_wasted_life_cost,
            c.threshold,
        ),
    )

    best = sorted_candidates[0]

    # Determine tie-break reason
    if len(candidates) == 1:
        reason = "only candidate"
    else:
        second = sorted_candidates[1]
        if best.mean_total_cost < second.mean_total_cost:
            reason = "lowest mean total cost"
        elif best.total_failures < second.total_failures:
            reason = "fewest failures (tie on cost)"
        elif best.mean_wasted_life_cost < second.mean_wasted_life_cost:
            reason = "lowest wasted-life cost (tie on cost and failures)"
        else:
            reason = "lowest threshold (tie on all metrics)"

    return SelectedThreshold(
        policy_family=best.policy_family,
        threshold=best.threshold,
        k_capacity=best.k_capacity,
        cost_regime_id=best.cost_regime_id,
        mean_total_cost=best.mean_total_cost,
        total_failures=best.total_failures,
        mean_wasted_life_cost=best.mean_wasted_life_cost,
        episode_count=best.episode_count,
        tie_break_reason=reason,
    )


def tune_threshold(
    policy_family: str,
    k_capacity: int,
    cost_regime_id: str,
    env_config: EnvironmentConfig,
    scenario_ids: List[str],
    reset_seeds: List[int],
    allow_oracle: bool = False,
    policy_seed: int = 42,
    threshold_grid: Optional[List[float]] = None,
    scenario_bank = None,
) -> Tuple[SelectedThreshold, List[ThresholdCandidate]]:
    """
    Tune threshold for a single policy family, K, and cost regime.

    Args:
        policy_family: Policy family to tune
        k_capacity: K value (1 or 2)
        cost_regime_id: Cost regime ID
        env_config: Base environment config
        scenario_ids: List of scenario IDs to evaluate on
        reset_seeds: List of reset seeds
        allow_oracle: If True, oracle policy is allowed
        policy_seed: Base seed for policy RNG
        threshold_grid: Optional custom threshold grid. If None, uses frozen
            contract grid from THRESHOLD_POLICIES. Use for diagnostic tests
            without mutating module constants.
        scenario_bank: Optional ScenarioBank to use for environment. If None,
            loads from env_config.scenario_bank_path.

    Returns:
        Tuple of (selected_threshold, all_candidates)

    Raises:
        ValueError: If policy family not tunable or rl_test in scenario_ids
    """
    # Barrier: reject rl_test
    if any("rl_test" in sid for sid in scenario_ids):
        raise ValueError(
            "rl_test scenarios are forbidden in threshold tuning. "
            f"Provided scenario IDs: {scenario_ids}"
        )

    if policy_family not in THRESHOLD_POLICIES:
        raise ValueError(
            f"Policy family {policy_family} is not tunable. "
            f"Tunable policies: {list(THRESHOLD_POLICIES.keys())}"
        )

    # Use custom grid if provided, otherwise use frozen contract grid
    if threshold_grid is not None:
        grid = threshold_grid
    else:
        grid = get_threshold_grid(policy_family)

    candidates = []

    evaluator = PolicyEvaluator(
        env_config=env_config,
        allow_oracle=allow_oracle and policy_family == "oracle_threshold",
        diagnostic_mode=policy_family == "oracle_threshold",
    )

    # Per-episode rows (one per formal (policy, threshold, K, regime,
    # scenario_id, reset_seed)). The producer collects these into a
    # canonical ``threshold_search_results.parquet`` so the formal
    # contract's identity-set reconstruction can verify the full episode
    # universe from a single parquet. The aggregation
    # below still produces the same ``ThresholdCandidate`` records —
    # the existing winner-selection behavior is unchanged.
    episode_rows: List[Dict[str, Any]] = []

    for threshold in grid:
        # Evaluate this threshold
        episode_results: List[EpisodeResult] = []

        for scenario_id in scenario_ids:
            for reset_seed in reset_seeds:
                # Create policy and context
                if policy_family == "greedy_predicted_rul":
                    policy = evaluator.create_policy(
                        policy_family,
                        activation_threshold=threshold,
                        policy_seed=policy_seed,
                    )
                else:
                    policy = evaluator.create_policy(
                        policy_family,
                        threshold=threshold,
                        policy_seed=policy_seed,
                    )

                context = evaluator.create_context(policy_family, policy_seed=policy_seed)

                # Create environment with diagnostic mode for oracle
                from ..envs import SelectiveMaintenanceEnv
                is_oracle = policy_family == "oracle_threshold"
                env = SelectiveMaintenanceEnv(
                    config=env_config,
                    info_mode="diagnostic" if is_oracle else "normal",
                    scenario_bank=scenario_bank,
                )

                eval_config = EvaluationConfig(
                    env_config=env_config,
                    policy_id=f"{policy_family}_{threshold}",
                    policy_family=policy_family,
                    threshold=threshold if policy_family != "greedy_predicted_rul" else None,
                    activation_threshold=threshold if policy_family == "greedy_predicted_rul" else None,
                    policy_seed=policy_seed,
                )

                run_id = f"tune_{policy_family}_{threshold}_k{k_capacity}_{cost_regime_id}_{scenario_id}_{reset_seed}"

                result = evaluator.evaluate_episode(
                    env=env,
                    policy=policy,
                    context=context,
                    scenario_id=scenario_id,
                    reset_seed=reset_seed,
                    eval_config=eval_config,
                    run_id=run_id,
                )

                episode_results.append(result)

                # Capture per-episode row for the formal evidence.
                # ``scenario_id`` here is whatever the caller passed
                # — the formal producer passes raw source IDs; test
                # callers may pass derived IDs and the parquet will
                # match them.
                episode_rows.append({
                    "policy_family": policy_family,
                    "threshold": float(threshold),
                    "k_capacity": int(k_capacity),
                    "cost_regime_id": cost_regime_id,
                    "scenario_id": scenario_id,
                    "reset_seed": int(reset_seed),
                    "total_cost": float(result.total_cost),
                    "preventive_cost": float(result.preventive_cost),
                    "failure_cost": float(result.failure_cost),
                    "wasted_life_cost": float(result.wasted_life_cost),
                    "failure_count": int(result.failure_count),
                    "episode_steps": int(result.episode_steps),
                    "completed": bool(result.completed),
                })

        # Aggregate results for this threshold
        total_costs = [r.total_cost for r in episode_results if r.completed]
        failures = [r.failure_count for r in episode_results if r.completed]
        wasted_life_costs = [r.wasted_life_cost for r in episode_results if r.completed]

        # Check for failed episodes - threshold tuning must fail when episodes fail
        failed_episodes = [r for r in episode_results if not r.completed]
        if failed_episodes:
            # Collect error details for reporting
            error_details = []
            for r in failed_episodes[:3]:  # First 3 failures
                error_details.append(
                    f"scenario={r.scenario_id}, seed={r.reset_seed}: {r.error}"
                )
            raise RuntimeError(
                f"Threshold tuning failed for {policy_family} at threshold={threshold}, "
                f"K={k_capacity}, regime={cost_regime_id}: "
                f"{len(failed_episodes)}/{len(episode_results)} episodes failed. "
                f"Errors: {'; '.join(error_details)}"
            )

        mean_total_cost = float(np.mean(total_costs))
        total_failures = sum(failures)
        mean_wasted_life_cost = float(np.mean(wasted_life_costs))

        candidates.append(ThresholdCandidate(
            policy_family=policy_family,
            threshold=threshold,
            k_capacity=k_capacity,
            cost_regime_id=cost_regime_id,
            mean_total_cost=mean_total_cost,
            total_failures=total_failures,
            mean_wasted_life_cost=mean_wasted_life_cost,
            episode_count=len(total_costs),
        ))

    # Select best threshold
    selected = select_best_threshold(candidates)

    # Return per-episode rows along with the candidate summary; the
    # producer writes both, the test surfaces may verify the list.
    selected.episode_rows = episode_rows  # type: ignore[attr-defined]

    return selected, candidates


def tune_all_thresholds(
    k_capacity: int,
    cost_regime_id: str,
    env_config: EnvironmentConfig,
    scenario_ids: List[str],
    reset_seeds: List[int],
    allow_oracle: bool = False,
) -> Dict[str, SelectedThreshold]:
    """
    Tune all threshold policies for a given K and cost regime.

    Args:
        k_capacity: K value (1 or 2)
        cost_regime_id: Cost regime ID
        env_config: Base environment config
        scenario_ids: List of scenario IDs
        reset_seeds: List of reset seeds
        allow_oracle: If True, include oracle policy

    Returns:
        Dict mapping policy_family to SelectedThreshold
    """
    results = {}

    for policy_family in THRESHOLD_POLICIES.keys():
        if policy_family == "oracle_threshold" and not allow_oracle:
            continue

        selected, _ = tune_threshold(
            policy_family=policy_family,
            k_capacity=k_capacity,
            cost_regime_id=cost_regime_id,
            env_config=env_config,
            scenario_ids=scenario_ids,
            reset_seeds=reset_seeds,
            allow_oracle=allow_oracle and policy_family == "oracle_threshold",
        )

        results[policy_family] = selected

    return results


def candidates_to_dataframe(
    candidates: List[ThresholdCandidate],
) -> pd.DataFrame:
    """
    Convert threshold candidates to parquet-ready DataFrame.

    Args:
        candidates: List of ThresholdCandidate instances

    Returns:
        DataFrame with one row per candidate
    """
    records = []
    for c in candidates:
        records.append({
            "policy_family": c.policy_family,
            "threshold": c.threshold,
            "k_capacity": c.k_capacity,
            "cost_regime_id": c.cost_regime_id,
            "mean_total_cost": c.mean_total_cost,
            "total_failures": c.total_failures,
            "mean_wasted_life_cost": c.mean_wasted_life_cost,
            "episode_count": c.episode_count,
        })

    return pd.DataFrame(records)


def selected_thresholds_to_dict(
    selected: Dict[str, SelectedThreshold],
) -> Dict[str, Any]:
    """
    Convert selected thresholds to JSON-serializable dict.

    Args:
        selected: Dict mapping policy_family to SelectedThreshold

    Returns:
        JSON-serializable dictionary
    """
    result = {}
    for policy_family, thresh in selected.items():
        result[policy_family] = {
            "threshold": thresh.threshold,
            "k_capacity": thresh.k_capacity,
            "cost_regime_id": thresh.cost_regime_id,
            "mean_total_cost": thresh.mean_total_cost,
            "total_failures": thresh.total_failures,
            "mean_wasted_life_cost": thresh.mean_wasted_life_cost,
            "episode_count": thresh.episode_count,
            "tie_break_reason": thresh.tie_break_reason,
        }
    return result


# =============================================================================
# Formal Case Planning for M3 Experiments
# =============================================================================

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class TuningCandidateIdentity:
    """Unique identity of a tuning candidate."""
    policy_family: str
    threshold: float
    k_capacity: int
    cost_regime_id: str


@dataclass(frozen=True)
class TuningEpisodeIdentity:
    """Unique identity of a tuning episode."""
    policy_family: str
    threshold: float
    k_capacity: int
    cost_regime_id: str
    scenario_id: str
    reset_seed: int


@dataclass(frozen=True)
class SelectedThresholdIdentity:
    """Unique identity of a selected threshold."""
    policy_family: str
    k_capacity: int
    cost_regime_id: str


@dataclass(frozen=True)
class EvaluationEpisodeIdentity:
    """Unique identity of an evaluation episode."""
    policy_family: str
    k_capacity: int
    cost_regime_id: str
    split: str
    scenario_id: str
    reset_seed: int


def generate_formal_tuning_candidates(
    policy_families: List[str] = None,
    k_values: List[int] = None,
    cost_regimes: List[str] = None,
    include_oracle: bool = False,
) -> List[TuningCandidateIdentity]:
    """
    Generate all formal tuning candidate identities.

    Dynamically enumerates the complete Cartesian product of:
    - policy family (4 threshold families by default, or 3 if oracle excluded)
    - threshold (frozen grid per family: 12+11+11+11=45 total)
    - K capacity (2 values)
    - cost regime (4 values)

    Total: 45 × 2 × 4 = 360 unique candidates (or 34 × 2 × 4 = 272 without oracle)

    Args:
        policy_families: Optional list of policy families. If None, uses all
            threshold families from THRESHOLD_POLICIES (excluding oracle unless
            include_oracle=True).
        k_values: Optional list of K values. If None, uses [1, 2].
        cost_regimes: Optional list of cost regime IDs. If None, uses all four
            frozen regimes.
        include_oracle: If True, includes oracle_threshold policy family.

    Returns:
        List of TuningCandidateIdentity, deterministically ordered by:
        (policy_family, threshold, k_capacity, cost_regime_id)
    """
    if policy_families is None:
        policy_families = [
            "age_threshold",
            "predicted_rul_threshold",
            "greedy_predicted_rul",
        ]
        if include_oracle:
            policy_families.append("oracle_threshold")

    if k_values is None:
        k_values = [1, 2]

    if cost_regimes is None:
        cost_regimes = [
            "failure-light-no-waste",
            "failure-heavy-no-waste",
            "failure-light-waste-aware",
            "failure-heavy-waste-aware",
        ]

    candidates = []
    for policy_family in policy_families:
        grid = get_threshold_grid(policy_family)
        for threshold in grid:
            for k in k_values:
                for regime in cost_regimes:
                    candidates.append(TuningCandidateIdentity(
                        policy_family=policy_family,
                        threshold=threshold,
                        k_capacity=k,
                        cost_regime_id=regime,
                    ))

    # Sort for deterministic order
    candidates.sort(key=lambda c: (c.policy_family, c.threshold, c.k_capacity, c.cost_regime_id))

    return candidates


def generate_formal_tuning_episodes(
    candidates: List[TuningCandidateIdentity],
    scenario_ids: List[str],
    reset_seeds: List[int],
) -> List[TuningEpisodeIdentity]:
    """
    Generate all formal tuning episode identities.

    For each candidate, generates one episode per scenario per reset seed.

    Args:
        candidates: List of TuningCandidateIdentity from
            generate_formal_tuning_candidates().
        scenario_ids: List of scenario IDs for the tuning split.
        reset_seeds: List of reset seeds.

    Returns:
        List of TuningEpisodeIdentity, deterministically ordered.
    """
    episodes = []
    for candidate in candidates:
        for scenario_id in scenario_ids:
            for seed in reset_seeds:
                episodes.append(TuningEpisodeIdentity(
                    policy_family=candidate.policy_family,
                    threshold=candidate.threshold,
                    k_capacity=candidate.k_capacity,
                    cost_regime_id=candidate.cost_regime_id,
                    scenario_id=scenario_id,
                    reset_seed=seed,
                ))

    # Sort for deterministic order
    episodes.sort(key=lambda e: (
        e.policy_family, e.threshold, e.k_capacity, e.cost_regime_id,
        e.scenario_id, e.reset_seed,
    ))

    return episodes


def generate_formal_selected_thresholds(
    policy_families: List[str] = None,
    k_values: List[int] = None,
    cost_regimes: List[str] = None,
    include_oracle: bool = False,
) -> List[SelectedThresholdIdentity]:
    """
    Generate all formal selected threshold identities.

    Each selected threshold is the winner of a tuning search for one:
    - policy family
    - K capacity
    - cost regime

    Total: 4 × 2 × 4 = 32 (or 3 × 2 × 4 = 24 without oracle)

    Args:
        policy_families: Optional list of policy families. If None, uses
            non-oracle threshold families.
        k_values: Optional list of K values. If None, uses [1, 2].
        cost_regimes: Optional list of cost regime IDs. If None, uses all four.
        include_oracle: If True, includes oracle_threshold.

    Returns:
        List of SelectedThresholdIdentity, deterministically ordered.
    """
    if policy_families is None:
        policy_families = [
            "age_threshold",
            "predicted_rul_threshold",
            "greedy_predicted_rul",
        ]
        if include_oracle:
            policy_families.append("oracle_threshold")

    if k_values is None:
        k_values = [1, 2]

    if cost_regimes is None:
        cost_regimes = [
            "failure-light-no-waste",
            "failure-heavy-no-waste",
            "failure-light-waste-aware",
            "failure-heavy-waste-aware",
        ]

    identities = []
    for policy_family in policy_families:
        for k in k_values:
            for regime in cost_regimes:
                identities.append(SelectedThresholdIdentity(
                    policy_family=policy_family,
                    k_capacity=k,
                    cost_regime_id=regime,
                ))

    identities.sort(key=lambda i: (i.policy_family, i.k_capacity, i.cost_regime_id))

    return identities


def generate_formal_evaluation_episodes(
    policy_families: List[str],
    k_values: List[int] = None,
    cost_regimes: List[str] = None,
    splits: List[str] = None,
    scenario_ids_by_split: dict = None,
    reset_seeds: List[int] = None,
    include_oracle: bool = False,
) -> List[EvaluationEpisodeIdentity]:
    """
    Generate all formal evaluation episode identities.

    For each policy (6 total: 4 threshold + 2 non-threshold), generates one
    episode per K per regime per split per scenario per reset seed.

    Total with default config:
    6 policies × 2 K × 4 regimes × 2 splits × 5 scenarios × 5 seeds = 2400

    Args:
        policy_families: List of all 6 policy families to evaluate. If None,
            uses all six M3 policies (excluding oracle unless include_oracle).
        k_values: Optional list of K values. If None, uses [1, 2].
        cost_regimes: Optional list of cost regime IDs. If None, uses all four.
        splits: Optional list of evaluation splits. If None, uses both
            predictor_train and rl_validation.
        scenario_ids_by_split: Dict mapping split name to list of scenario IDs.
            If None, each split is assumed to have 5 placeholder scenarios.
        reset_seeds: Optional list of reset seeds. If None, uses 5 seeds.
        include_oracle: If True, includes oracle_threshold in policy_families.

    Returns:
        List of EvaluationEpisodeIdentity, deterministically ordered.
    """
    if policy_families is None:
        policy_families = [
            "corrective_only",
            "random_feasible",
            "age_threshold",
            "predicted_rul_threshold",
            "greedy_predicted_rul",
        ]
        if include_oracle:
            policy_families.append("oracle_threshold")

    if k_values is None:
        k_values = [1, 2]

    if cost_regimes is None:
        cost_regimes = [
            "failure-light-no-waste",
            "failure-heavy-no-waste",
            "failure-light-waste-aware",
            "failure-heavy-waste-aware",
        ]

    if splits is None:
        splits = ["predictor_train", "rl_validation"]

    if scenario_ids_by_split is None:
        # Placeholder: 5 scenarios per split
        scenario_ids_by_split = {
            split: [f"{split}_scenario_{i}" for i in range(5)]
            for split in splits
        }

    if reset_seeds is None:
        reset_seeds = [6521, 6522, 6523, 6524, 6525]

    episodes = []
    for policy_family in policy_families:
        for k in k_values:
            for regime in cost_regimes:
                for split in splits:
                    scenario_ids = scenario_ids_by_split.get(split, [])
                    for scenario_id in scenario_ids:
                        for seed in reset_seeds:
                            episodes.append(EvaluationEpisodeIdentity(
                                policy_family=policy_family,
                                k_capacity=k,
                                cost_regime_id=regime,
                                split=split,
                                scenario_id=scenario_id,
                                reset_seed=seed,
                            ))

    # Sort for deterministic order
    episodes.sort(key=lambda e: (
        e.policy_family, e.k_capacity, e.cost_regime_id, e.split,
        e.scenario_id, e.reset_seed,
    ))

    return episodes


def count_formal_tuning(
    include_oracle: bool = False,
) -> dict:
    """
    Count formal tuning experiment dimensions.

    Args:
        include_oracle: If True, includes oracle in counts.

    Returns:
        Dict with counts:
        - candidate_count: Total tuning candidates (360 or 272)
        - thresholds_per_k_regime: Thresholds per K/regime (45 or 34)
        - policy_count: Number of policy families (4 or 3)
    """
    policy_families = [
        "age_threshold",
        "predicted_rul_threshold",
        "greedy_predicted_rul",
    ]
    if include_oracle:
        policy_families.append("oracle_threshold")

    thresholds_per_k_regime = sum(
        len(get_threshold_grid(pf)) for pf in policy_families
    )

    candidate_count = thresholds_per_k_regime * 2 * 4  # 2 K, 4 regimes

    return {
        "candidate_count": candidate_count,
        "thresholds_per_k_regime": thresholds_per_k_regime,
        "policy_count": len(policy_families),
    }


def count_formal_evaluation(
    include_oracle: bool = False,
    scenario_count_per_split: int = 5,
    split_count: int = 2,
) -> dict:
    """
    Count formal evaluation experiment dimensions.

    Args:
        include_oracle: If True, includes oracle in policy count.
        scenario_count_per_split: Number of scenarios per split (default 5).
        split_count: Number of evaluation splits (default 2).

    Returns:
        Dict with counts:
        - policy_count: Number of policies (6 or 5)
        - episode_count: Total evaluation episodes
    """
    policy_count = 5  # corrective_only, random_feasible, 3 threshold (no oracle)
    if include_oracle:
        policy_count = 6

    episode_count = (
        policy_count *
        2 *  # K values
        4 *  # regimes
        split_count *
        scenario_count_per_split *
        5  # reset seeds
    )

    return {
        "policy_count": policy_count,
        "episode_count": episode_count,
    }
