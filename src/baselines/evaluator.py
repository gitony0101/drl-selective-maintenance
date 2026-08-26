"""
Common evaluator for Milestone 3 Baselines.

Runs every policy through the same environment and scenarios:
- For practical policies: observation + PolicyContext only
- For oracle: OracleContext + diagnostic info
- Records actual cost components
- Verifies reward = -total_cost
- Enforces episode horizon
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..envs import SelectiveMaintenanceEnv, EnvironmentConfig
from ..envs.action_table import ActionSubset
from .protocols import PolicyContext, OracleContext, Observation, ActionId
from .rule_policies import (
    CorrectiveOnly,
    RandomFeasible,
    AgeThreshold,
    PredictedRULThreshold,
    GreedyPredictedRUL,
)
from .oracle_policy import OracleThreshold


def _result_threshold_value(eval_config: "EvaluationConfig") -> Optional[float]:
    """Resolve which threshold to record for an EpisodeResult.

    Each policy family declares exactly one:
      - corrective_only, random_feasible      -> None
      - age_threshold, predicted_rul_threshold,
        oracle_threshold                      -> eval_config.threshold
      - greedy_predicted_rul                  -> eval_config.activation_threshold

    No fallback substitution is allowed (Step 2). If both fields are set
    or both are None, that is a contract violation and we raise.
    """
    has_t = eval_config.threshold is not None
    has_at = eval_config.activation_threshold is not None
    if eval_config.policy_family in ("corrective_only", "random_feasible"):
        if has_t or has_at:
            raise ValueError(
                f"policy_family={eval_config.policy_family} does NOT use a threshold; "
                f"got threshold={eval_config.threshold} activation_threshold="
                f"{eval_config.activation_threshold}"
            )
        return None
    if eval_config.policy_family == "greedy_predicted_rul":
        if has_t:
            raise ValueError(
                f"greedy_predicted_rul must set activation_threshold, not threshold; "
                f"got threshold={eval_config.threshold}"
            )
        if not has_at:
            raise ValueError(
                "greedy_predicted_rul requires activation_threshold"
            )
        return eval_config.activation_threshold
    # Other threshold policies
    if has_at:
        raise ValueError(
            f"policy_family={eval_config.policy_family} must set threshold, not "
            f"activation_threshold; got activation_threshold="
            f"{eval_config.activation_threshold}"
        )
    if not has_t:
        raise ValueError(
            f"policy_family={eval_config.policy_family} requires threshold"
        )
    return eval_config.threshold


@dataclass
class EpisodeResult:
    """Results from a single episode evaluation."""

    run_id: str
    policy_id: str
    policy_family: str
    threshold: Optional[float]
    split: str
    scenario_id: str
    cost_regime_id: str
    maintenance_capacity: int
    reset_seed: int
    policy_seed: int
    episode_steps: int
    episode_return: float
    discounted_return: float
    total_cost: float
    preventive_cost: float
    failure_cost: float
    wasted_life_cost: float
    preventive_replacement_count: int
    failure_count: int
    action_count: int
    empty_action_count: int
    capacity_saturated_step_count: int
    mean_selected_predicted_rul: float
    mean_selected_age: float
    nan_observation_count: int
    inf_observation_count: int
    terminated_count: int
    truncated: bool
    completed: bool
    error: Optional[str] = None
    # Oracle-only diagnostic metrics
    mean_unused_true_rul_at_pm: Optional[float] = None
    # Raw source-bank scenario ID (the derived-ID used inside the env is
    # kept in scenario_id; source_scenario_id carries the raw ID the
    # auditor hashes directly against the source-bank JSON). Optional for
    # backward compatibility with callers that don't supply it; the
    # formal producer always populates this.
    source_scenario_id: Optional[str] = None


@dataclass
class EvaluationConfig:
    """Configuration for policy evaluation."""

    env_config: EnvironmentConfig
    policy_id: str
    policy_family: str
    threshold: Optional[float] = None
    activation_threshold: Optional[float] = None
    policy_seed: Optional[int] = None
    discount_factor: float = 1.0  # No discounting by default


class PolicyEvaluator:
    """
    Common evaluator for all policy families.

    Runs policies through the environment with proper information barriers:
    - Practical policies receive only observation + PolicyContext
    - Oracle receives OracleContext + diagnostic info
    """

    def __init__(
        self,
        env_config: EnvironmentConfig,
        allow_oracle: bool = False,
        diagnostic_mode: bool = False,
    ) -> None:
        """
        Initialize evaluator.

        Args:
            env_config: Environment configuration
            allow_oracle: If True, oracle policies may be evaluated
            diagnostic_mode: If True, diagnostic info is available
        """
        self.env_config = env_config
        self.allow_oracle = allow_oracle
        self.diagnostic_mode = diagnostic_mode

    def create_policy(
        self,
        policy_family: str,
        threshold: Optional[float] = None,
        activation_threshold: Optional[float] = None,
        policy_seed: Optional[int] = None,
    ):
        """
        Create a policy instance.

        Args:
            policy_family: One of:
                - corrective_only
                - random_feasible
                - age_threshold
                - predicted_rul_threshold
                - greedy_predicted_rul
                - oracle_threshold
            threshold: Threshold value for threshold-based policies
            activation_threshold: Activation threshold for greedy policy
            policy_seed: RNG seed for policy

        Returns:
            Policy instance
        """
        if policy_family == "corrective_only":
            return CorrectiveOnly()
        elif policy_family == "random_feasible":
            return RandomFeasible(seed=policy_seed)
        elif policy_family == "age_threshold":
            if threshold is None:
                raise ValueError("age_threshold requires threshold parameter")
            return AgeThreshold(threshold=threshold)
        elif policy_family == "predicted_rul_threshold":
            if threshold is None:
                raise ValueError("predicted_rul_threshold requires threshold parameter")
            return PredictedRULThreshold(threshold=threshold)
        elif policy_family == "greedy_predicted_rul":
            if activation_threshold is None:
                raise ValueError("greedy_predicted_rul requires activation_threshold parameter")
            return GreedyPredictedRUL(activation_threshold=activation_threshold)
        elif policy_family == "oracle_threshold":
            if not self.allow_oracle:
                raise ValueError("oracle_threshold requires allow_oracle=True")
            if threshold is None:
                raise ValueError("oracle_threshold requires threshold parameter")
            return OracleThreshold(threshold=threshold)
        else:
            raise ValueError(f"Unknown policy family: {policy_family}")

    def create_context(
        self,
        policy_family: str,
        policy_seed: Optional[int] = None,
    ) -> PolicyContext | OracleContext:
        """
        Create appropriate context for policy.

        Args:
            policy_family: Policy family name
            policy_seed: RNG seed for policy

        Returns:
            PolicyContext for practical policies, OracleContext for oracle
        """
        rng = np.random.default_rng(policy_seed)

        if policy_family == "oracle_threshold":
            return OracleContext(
                maintenance_capacity=self.env_config.maintenance_capacity,
                age_scale_cycles=self.env_config.age_scale_cycles,
                rul_scale=self.env_config.rul_scale,
                action_table=self._get_action_table(),
                cost_regime_id=self.env_config.cost_regime_id,
                policy_rng=rng,
                allow_oracle=True,
                diagnostic_mode=True,
            )
        else:
            return PolicyContext(
                maintenance_capacity=self.env_config.maintenance_capacity,
                age_scale_cycles=self.env_config.age_scale_cycles,
                rul_scale=self.env_config.rul_scale,
                action_table=self._get_action_table(),
                cost_regime_id=self.env_config.cost_regime_id,
                policy_rng=rng,
            )

    def _get_action_table(self) -> Tuple[Tuple[int, ...], ...]:
        """Get action table for current K."""
        if self.env_config.maintenance_capacity == 1:
            from ..envs.action_table import ACTION_TABLE_N5_K1
            return ACTION_TABLE_N5_K1
        else:
            from ..envs.action_table import ACTION_TABLE_N5_K2
            return ACTION_TABLE_N5_K2

    def evaluate_episode(
        self,
        env: SelectiveMaintenanceEnv,
        policy: Any,
        context: PolicyContext | OracleContext,
        scenario_id: str,
        reset_seed: int,
        eval_config: EvaluationConfig,
        run_id: str,
        source_scenario_id: Optional[str] = None,
    ) -> EpisodeResult:
        """
        Evaluate policy for one episode.

        Args:
            env: Environment instance (already reset)
            policy: Policy instance
            context: Policy context
            scenario_id: Current scenario ID (derived, env-compatible)
            reset_seed: Environment reset seed
            eval_config: Evaluation configuration
            run_id: Unique run identifier
            source_scenario_id: Optional raw source-bank scenario ID. When
                supplied, it is recorded on the EpisodeResult alongside
                the derived ``scenario_id`` so the writer can persist it
                into the parquet. Defaults to None for backward
                compatibility with callers that do not pass it.

        Returns:
            EpisodeResult with all metrics
        """
        is_oracle = isinstance(context, OracleContext)

        # Recording variables
        total_reward = 0.0
        discounted_reward = 0.0
        total_cost_accum = 0.0  # Initialize before loop for error handler
        preventive_cost_accum = 0.0
        failure_cost_accum = 0.0
        wasted_life_cost_accum = 0.0
        preventive_count = 0
        failure_count = 0
        action_count = 0
        empty_action_count = 0
        capacity_saturated_count = 0
        nan_obs_count = 0
        inf_obs_count = 0
        terminated_count = 0
        steps_completed = 0

        selected_ruls = []
        selected_ages = []
        unused_true_ruls_at_pm = []  # Oracle only

        try:
            obs, info = env.reset(seed=reset_seed, options={"scenario_id": scenario_id})

            for step in range(self.env_config.episode_horizon):
                # Validate observation
                if np.isnan(obs).any():
                    nan_obs_count += 1
                if np.isinf(obs).any():
                    inf_obs_count += 1

                # Select action
                if is_oracle:
                    # Oracle needs diagnostic info
                    action = policy.select_action(obs, context, diagnostic_info=info)
                else:
                    # Practical policy: observation + context only
                    action = policy.select_action(obs, context)

                # Validate action
                if action < 0 or action >= len(context.action_table):
                    raise ValueError(f"Invalid action ID: {action}")

                # Track action statistics
                if action == 0:
                    empty_action_count += 1
                else:
                    action_count += 1
                    selected_slots = context.action_table[action]
                    if len(selected_slots) == context.maintenance_capacity:
                        capacity_saturated_count += 1

                    # Track selected slot features (for reporting)
                    obs_reshaped = obs.reshape(5, 2)
                    for slot_idx in selected_slots:
                        selected_ages.append(obs_reshaped[slot_idx, 0])
                        selected_ruls.append(obs_reshaped[slot_idx, 1])

                # Execute step
                obs, reward, terminated, truncated, info = env.step(action)

                # Accumulate metrics from info
                # Note: reward = -total_cost per step, so episode_return = -episode_total_cost
                total_reward += reward
                discounted_reward += reward * (eval_config.discount_factor ** step)
                total_cost_accum += info.get("total_cost", 0.0)
                preventive_cost_accum += info.get("preventive_cost", 0.0)
                failure_cost_accum += info.get("failure_cost", 0.0)
                wasted_life_cost_accum += info.get("wasted_life_cost", 0.0)
                preventive_count += info.get("num_preventive", 0)
                failure_count += info.get("num_failures", 0)

                if terminated:
                    terminated_count += 1

                steps_completed = step + 1

                if truncated:
                    break

            return EpisodeResult(
                run_id=run_id,
                policy_id=eval_config.policy_id,
                policy_family=eval_config.policy_family,
                threshold=_result_threshold_value(eval_config),
                split=self.env_config.split,
                scenario_id=scenario_id,
                cost_regime_id=self.env_config.cost_regime_id,
                maintenance_capacity=self.env_config.maintenance_capacity,
                reset_seed=reset_seed,
                policy_seed=eval_config.policy_seed or 0,
                episode_steps=steps_completed,
                episode_return=total_reward,
                discounted_return=discounted_reward,
                total_cost=total_cost_accum,
                preventive_cost=preventive_cost_accum,
                failure_cost=failure_cost_accum,
                wasted_life_cost=wasted_life_cost_accum,
                preventive_replacement_count=preventive_count,
                failure_count=failure_count,
                action_count=action_count,
                empty_action_count=empty_action_count,
                capacity_saturated_step_count=capacity_saturated_count,
                mean_selected_predicted_rul=float(np.mean(selected_ruls)) if selected_ruls else 0.0,
                mean_selected_age=float(np.mean(selected_ages)) if selected_ages else 0.0,
                nan_observation_count=nan_obs_count,
                inf_observation_count=inf_obs_count,
                terminated_count=terminated_count,
                truncated=truncated,
                completed=True,
                error=None,
                source_scenario_id=source_scenario_id,
            )

        except Exception as e:
            # In case of error during episode, return partial results
            # Initialize accumulators that may not have been set
            if 'total_cost_accum' not in locals():
                total_cost_accum = 0.0
            if 'preventive_cost_accum' not in locals():
                preventive_cost_accum = 0.0
            if 'failure_cost_accum' not in locals():
                failure_cost_accum = 0.0
            if 'wasted_life_cost_accum' not in locals():
                wasted_life_cost_accum = 0.0

            return EpisodeResult(
                run_id=run_id,
                policy_id=eval_config.policy_id,
                policy_family=eval_config.policy_family,
                threshold=_result_threshold_value(eval_config),
                split=self.env_config.split,
                scenario_id=scenario_id,
                cost_regime_id=self.env_config.cost_regime_id,
                maintenance_capacity=self.env_config.maintenance_capacity,
                reset_seed=reset_seed,
                policy_seed=eval_config.policy_seed or 0,
                episode_steps=steps_completed,
                episode_return=total_reward,
                discounted_return=discounted_reward,
                total_cost=total_cost_accum,
                preventive_cost=preventive_cost_accum,
                failure_cost=failure_cost_accum,
                wasted_life_cost=wasted_life_cost_accum,
                preventive_replacement_count=preventive_count,
                failure_count=failure_count,
                action_count=action_count,
                empty_action_count=empty_action_count,
                capacity_saturated_step_count=capacity_saturated_count,
                mean_selected_predicted_rul=float(np.mean(selected_ruls)) if selected_ruls else 0.0,
                mean_selected_age=float(np.mean(selected_ages)) if selected_ages else 0.0,
                nan_observation_count=nan_obs_count,
                inf_observation_count=inf_obs_count,
                terminated_count=terminated_count,
                truncated=False,
                completed=False,
                error=str(e),
                source_scenario_id=source_scenario_id,
            )