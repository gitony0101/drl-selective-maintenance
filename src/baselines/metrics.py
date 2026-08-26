"""
Metrics computation and summarization for Milestone 3 Baselines.

Computes per-policy summary statistics:
- mean, sample_std, standard_error
- 95% confidence intervals
- episode_count
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .evaluator import EpisodeResult


@dataclass
class PolicySummary:
    """Summary statistics for a policy."""

    policy_id: str
    metric: str
    mean: float
    sample_std: float
    standard_error: float
    ci_95_lower: float
    ci_95_upper: float
    episode_count: int
    split: str
    maintenance_capacity: int
    cost_regime_id: str


def compute_summary_statistics(
    values: np.ndarray,
) -> Dict[str, float]:
    """
    Compute summary statistics for an array of values.

    Args:
        values: Array of metric values

    Returns:
        Dict with mean, sample_std, standard_error, ci_95_lower, ci_95_upper
    """
    n = len(values)
    if n == 0:
        return {
            "mean": np.nan,
            "sample_std": np.nan,
            "standard_error": np.nan,
            "ci_95_lower": np.nan,
            "ci_95_upper": np.nan,
            "episode_count": 0,
        }

    mean = float(np.mean(values))
    sample_std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    standard_error = sample_std / np.sqrt(n) if n > 0 else 0.0
    ci_95_lower = mean - 1.96 * standard_error
    ci_95_upper = mean + 1.96 * standard_error

    return {
        "mean": mean,
        "sample_std": sample_std,
        "standard_error": standard_error,
        "ci_95_lower": ci_95_lower,
        "ci_95_upper": ci_95_upper,
        "episode_count": n,
    }


def validate_no_nan_inf(values: np.ndarray, metric_name: str) -> None:
    """
    Validate that values contain no NaN or Inf.

    Args:
        values: Array of values to validate
        metric_name: Name of metric for error message

    Raises:
        ValueError: If NaN or Inf found
    """
    if np.isnan(values).any():
        raise ValueError(f"{metric_name} contains NaN values")
    if np.isinf(values).any():
        raise ValueError(f"{metric_name} contains Inf values")


def summarize_results(
    results: List[EpisodeResult],
    group_by: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Summarize episode results by policy and other dimensions.

    Args:
        results: List of EpisodeResult instances
        group_by: Columns to group by. Default: ["policy_id", "split", "maintenance_capacity", "cost_regime_id"]

    Returns:
        DataFrame with summary statistics per group
    """
    if group_by is None:
        group_by = ["policy_id", "split", "maintenance_capacity", "cost_regime_id"]

    # Convert results to DataFrame
    records = []
    for r in results:
        records.append({
            "run_id": r.run_id,
            "policy_id": r.policy_id,
            "policy_family": r.policy_family,
            "threshold": r.threshold,
            "split": r.split,
            "scenario_id": r.scenario_id,
            "cost_regime_id": r.cost_regime_id,
            "maintenance_capacity": r.maintenance_capacity,
            "reset_seed": r.reset_seed,
            "policy_seed": r.policy_seed,
            "episode_steps": r.episode_steps,
            "episode_return": r.episode_return,
            "discounted_return": r.discounted_return,
            "total_cost": r.total_cost,
            "preventive_cost": r.preventive_cost,
            "failure_cost": r.failure_cost,
            "wasted_life_cost": r.wasted_life_cost,
            "preventive_replacement_count": r.preventive_replacement_count,
            "failure_count": r.failure_count,
            "action_count": r.action_count,
            "empty_action_count": r.empty_action_count,
            "capacity_saturated_step_count": r.capacity_saturated_step_count,
            "mean_selected_predicted_rul": r.mean_selected_predicted_rul,
            "mean_selected_age": r.mean_selected_age,
            "nan_observation_count": r.nan_observation_count,
            "inf_observation_count": r.inf_observation_count,
            "terminated_count": r.terminated_count,
            "truncated": r.truncated,
            "completed": r.completed,
            "error": r.error,
        })

    df = pd.DataFrame(records)

    # Compute summary statistics per group
    summaries = []
    for group_keys, group_df in df.groupby(group_by):
        group_dict = dict(zip(group_by, group_keys))

        # Primary metric: total_cost
        cost_values = group_df["total_cost"].dropna().values
        cost_stats = compute_summary_statistics(cost_values)

        summary = {
            **group_dict,
            "metric": "total_cost",
            "mean": cost_stats["mean"],
            "sample_std": cost_stats["sample_std"],
            "standard_error": cost_stats["standard_error"],
            "ci_95_lower": cost_stats["ci_95_lower"],
            "ci_95_upper": cost_stats["ci_95_upper"],
            "episode_count": cost_stats["episode_count"],
        }
        summaries.append(summary)

    return pd.DataFrame(summaries)


def results_to_parquet(results: List[EpisodeResult]) -> pd.DataFrame:
    """
    Convert episode results to parquet-ready DataFrame.

    Args:
        results: List of EpisodeResult instances

    Returns:
        DataFrame with one row per episode
    """
    records = []
    for r in results:
        records.append({
            "run_id": r.run_id,
            "policy_id": r.policy_id,
            "policy_family": r.policy_family,
            "threshold": r.threshold,
            "split": r.split,
            "scenario_id": r.scenario_id,
            "cost_regime_id": r.cost_regime_id,
            "maintenance_capacity": r.maintenance_capacity,
            "reset_seed": r.reset_seed,
            "policy_seed": r.policy_seed,
            "episode_steps": r.episode_steps,
            "episode_return": r.episode_return,
            "discounted_return": r.discounted_return,
            "total_cost": r.total_cost,
            "preventive_cost": r.preventive_cost,
            "failure_cost": r.failure_cost,
            "wasted_life_cost": r.wasted_life_cost,
            "preventive_replacement_count": r.preventive_replacement_count,
            "failure_count": r.failure_count,
            "action_count": r.action_count,
            "empty_action_count": r.empty_action_count,
            "capacity_saturated_step_count": r.capacity_saturated_step_count,
            "mean_selected_predicted_rul": r.mean_selected_predicted_rul,
            "mean_selected_age": r.mean_selected_age,
            "nan_observation_count": r.nan_observation_count,
            "inf_observation_count": r.inf_observation_count,
            "terminated_count": r.terminated_count,
            "truncated": r.truncated,
            "completed": r.completed,
            "error": r.error if r.error else None,
            "mean_unused_true_rul_at_pm": r.mean_unused_true_rul_at_pm,
        })

    return pd.DataFrame(records)


def validate_json_serializable(obj: Any, path: str = "") -> None:
    """
    Validate that an object is JSON-serializable (no NaN, Inf, tensors).

    Args:
        obj: Object to validate
        path: Path string for error reporting

    Raises:
        ValueError: If not JSON-serializable
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            validate_json_serializable(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            validate_json_serializable(v, f"{path}[{i}]")
    elif isinstance(obj, (int, float, str, bool, type(None))):
        if isinstance(obj, float):
            if np.isnan(obj):
                raise ValueError(f"NaN at {path} is not JSON-serializable")
            if np.isinf(obj):
                raise ValueError(f"Inf at {path} is not JSON-serializable")
    elif hasattr(obj, "item"):  # numpy scalar
        val = obj.item()
        if np.isnan(val) or np.isinf(val):
            raise ValueError(f"Invalid numpy scalar at {path}")
    else:
        raise ValueError(f"Type {type(obj)} at {path} is not JSON-serializable")


def validate_artifact_values(df: pd.DataFrame, artifact_name: str) -> None:
    """
    Validate DataFrame for artifact export.

    Args:
        df: DataFrame to validate
        artifact_name: Name for error reporting

    Raises:
        ValueError: If invalid values found
    """
    for col in df.columns:
        values = df[col].dropna()
        if values.dtype.kind == 'f':  # float column
            if values.isna().any():
                pass  # Already handled by dropna
            if (values == np.inf).any() or (values == -np.inf).any():
                raise ValueError(f"{artifact_name} column {col} contains Inf values")