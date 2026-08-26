"""
Formal Case Planner tests for Milestone 3.

Verifies the formal case planner produces correct counts:
- 360 tuning candidates (45 thresholds × 2 K × 4 regimes)
- 9000 tuning episodes (360 candidates × 5 scenarios × 5 seeds)
- 32 selected thresholds (4 policies × 2 K × 4 regimes)
- 2400 evaluation episodes (6 policies × 2 K × 4 regimes × 2 splits × 5 scenarios × 5 seeds)
"""

import pytest

from src.baselines.tuning import (
    generate_formal_tuning_candidates,
    generate_formal_tuning_episodes,
    generate_formal_selected_thresholds,
    generate_formal_evaluation_episodes,
    count_formal_tuning,
    count_formal_evaluation,
    TuningCandidateIdentity,
    TuningEpisodeIdentity,
    SelectedThresholdIdentity,
    EvaluationEpisodeIdentity,
)


class TestFormalTuningCandidates:
    """Test formal tuning candidate generation."""

    def test_candidate_count_without_oracle(self):
        """Should produce 272 candidates without oracle."""
        candidates = generate_formal_tuning_candidates(include_oracle=False)
        assert len(candidates) == 272, f"Expected 272, got {len(candidates)}"

    def test_candidate_count_with_oracle(self):
        """Should produce 360 candidates with oracle."""
        candidates = generate_formal_tuning_candidates(include_oracle=True)
        assert len(candidates) == 360, f"Expected 360, got {len(candidates)}"

    def test_candidate_identity_structure(self):
        """Each candidate should have correct identity fields."""
        candidates = generate_formal_tuning_candidates(include_oracle=True)
        c = candidates[0]
        assert hasattr(c, 'policy_family')
        assert hasattr(c, 'threshold')
        assert hasattr(c, 'k_capacity')
        assert hasattr(c, 'cost_regime_id')

    def test_no_duplicate_candidates(self):
        """All candidates should be unique."""
        candidates = generate_formal_tuning_candidates(include_oracle=True)
        unique_keys = set(
            (c.policy_family, c.threshold, c.k_capacity, c.cost_regime_id)
            for c in candidates
        )
        assert len(unique_keys) == len(candidates), "Duplicate candidates found"

    def test_deterministic_order(self):
        """Candidates should be deterministically ordered."""
        candidates1 = generate_formal_tuning_candidates(include_oracle=True)
        candidates2 = generate_formal_tuning_candidates(include_oracle=True)
        assert candidates1 == candidates2

    def test_policy_families_covered(self):
        """All four threshold policy families should be covered."""
        candidates = generate_formal_tuning_candidates(include_oracle=True)
        policy_families = set(c.policy_family for c in candidates)
        assert policy_families == {
            "age_threshold",
            "predicted_rul_threshold",
            "greedy_predicted_rul",
            "oracle_threshold",
        }

    def test_k_values_covered(self):
        """Both K=1 and K=2 should be covered."""
        candidates = generate_formal_tuning_candidates(include_oracle=True)
        k_values = set(c.k_capacity for c in candidates)
        assert k_values == {1, 2}

    def test_cost_regimes_covered(self):
        """All four cost regimes should be covered."""
        candidates = generate_formal_tuning_candidates(include_oracle=True)
        regimes = set(c.cost_regime_id for c in candidates)
        assert regimes == {
            "failure-light-no-waste",
            "failure-heavy-no-waste",
            "failure-light-waste-aware",
            "failure-heavy-waste-aware",
        }

    def test_threshold_counts_per_policy(self):
        """Each policy family should have correct threshold count."""
        candidates = generate_formal_tuning_candidates(include_oracle=True)

        # Count thresholds per policy (should be independent of K/regime)
        thresholds_by_policy = {}
        for c in candidates:
            if c.policy_family not in thresholds_by_policy:
                thresholds_by_policy[c.policy_family] = set()
            thresholds_by_policy[c.policy_family].add(c.threshold)

        # Verify counts (each threshold appears for each K×regime combo)
        # But we're counting unique thresholds per policy
        assert len(thresholds_by_policy["age_threshold"]) == 12
        assert len(thresholds_by_policy["predicted_rul_threshold"]) == 11
        assert len(thresholds_by_policy["greedy_predicted_rul"]) == 11
        assert len(thresholds_by_policy["oracle_threshold"]) == 11


class TestFormalTuningEpisodes:
    """Test formal tuning episode generation."""

    def test_episode_count_with_five_scenarios_five_seeds(self):
        """Should produce 9000 episodes with 5 scenarios and 5 seeds."""
        candidates = generate_formal_tuning_candidates(include_oracle=True)
        scenario_ids = [f"scenario_{i}" for i in range(5)]
        reset_seeds = [6521, 6522, 6523, 6524, 6525]

        episodes = generate_formal_tuning_episodes(
            candidates, scenario_ids, reset_seeds
        )

        # 360 candidates × 5 scenarios × 5 seeds = 9000
        assert len(episodes) == 9000, f"Expected 9000, got {len(episodes)}"

    def test_episode_identity_structure(self):
        """Each episode should have correct identity fields."""
        candidates = generate_formal_tuning_candidates(include_oracle=True)
        scenario_ids = ["scenario_0"]
        reset_seeds = [6521]

        episodes = generate_formal_tuning_episodes(
            candidates, scenario_ids, reset_seeds
        )

        e = episodes[0]
        assert hasattr(e, 'policy_family')
        assert hasattr(e, 'threshold')
        assert hasattr(e, 'k_capacity')
        assert hasattr(e, 'cost_regime_id')
        assert hasattr(e, 'scenario_id')
        assert hasattr(e, 'reset_seed')

    def test_no_duplicate_episodes(self):
        """All episodes should be unique."""
        candidates = generate_formal_tuning_candidates(include_oracle=True)
        scenario_ids = [f"scenario_{i}" for i in range(5)]
        reset_seeds = [6521, 6522, 6523, 6524, 6525]

        episodes = generate_formal_tuning_episodes(
            candidates, scenario_ids, reset_seeds
        )

        unique_keys = set(
            (e.policy_family, e.threshold, e.k_capacity, e.cost_regime_id,
             e.scenario_id, e.reset_seed)
            for e in episodes
        )
        assert len(unique_keys) == len(episodes), "Duplicate episodes found"


class TestFormalSelectedThresholds:
    """Test formal selected threshold generation."""

    def test_selected_count_without_oracle(self):
        """Should produce 24 selected thresholds without oracle."""
        selected = generate_formal_selected_thresholds(include_oracle=False)
        assert len(selected) == 24, f"Expected 24, got {len(selected)}"

    def test_selected_count_with_oracle(self):
        """Should produce 32 selected thresholds with oracle."""
        selected = generate_formal_selected_thresholds(include_oracle=True)
        assert len(selected) == 32, f"Expected 32, got {len(selected)}"

    def test_selected_identity_structure(self):
        """Each selected threshold should have correct identity fields."""
        selected = generate_formal_selected_thresholds(include_oracle=True)
        s = selected[0]
        assert hasattr(s, 'policy_family')
        assert hasattr(s, 'k_capacity')
        assert hasattr(s, 'cost_regime_id')

    def test_no_duplicate_selected(self):
        """All selected thresholds should be unique."""
        selected = generate_formal_selected_thresholds(include_oracle=True)
        unique_keys = set(
            (s.policy_family, s.k_capacity, s.cost_regime_id)
            for s in selected
        )
        assert len(unique_keys) == len(selected), "Duplicate selected thresholds found"


class TestFormalEvaluationEpisodes:
    """Test formal evaluation episode generation."""

    def test_episode_count_full_config(self):
        """Should produce 2400 episodes with full config."""
        policy_families = [
            "corrective_only",
            "random_feasible",
            "age_threshold",
            "predicted_rul_threshold",
            "greedy_predicted_rul",
            "oracle_threshold",
        ]
        scenario_ids_by_split = {
            "predictor_train": [f"predictor_train_{i}" for i in range(5)],
            "rl_validation": [f"rl_validation_{i}" for i in range(5)],
        }

        episodes = generate_formal_evaluation_episodes(
            policy_families=policy_families,
            scenario_ids_by_split=scenario_ids_by_split,
            include_oracle=True,
        )

        # 6 policies × 2 K × 4 regimes × 2 splits × 5 scenarios × 5 seeds = 2400
        assert len(episodes) == 2400, f"Expected 2400, got {len(episodes)}"

    def test_episode_count_without_oracle(self):
        """Should produce 2000 episodes without oracle."""
        policy_families = [
            "corrective_only",
            "random_feasible",
            "age_threshold",
            "predicted_rul_threshold",
            "greedy_predicted_rul",
        ]
        scenario_ids_by_split = {
            "predictor_train": [f"predictor_train_{i}" for i in range(5)],
            "rl_validation": [f"rl_validation_{i}" for i in range(5)],
        }

        episodes = generate_formal_evaluation_episodes(
            policy_families=policy_families,
            scenario_ids_by_split=scenario_ids_by_split,
            include_oracle=False,
        )

        # 5 policies × 2 K × 4 regimes × 2 splits × 5 scenarios × 5 seeds = 2000
        assert len(episodes) == 2000, f"Expected 2000, got {len(episodes)}"

    def test_episode_identity_structure(self):
        """Each episode should have correct identity fields."""
        policy_families = ["corrective_only"]
        scenario_ids_by_split = {"predictor_train": ["scenario_0"]}

        episodes = generate_formal_evaluation_episodes(
            policy_families=policy_families,
            scenario_ids_by_split=scenario_ids_by_split,
        )

        e = episodes[0]
        assert hasattr(e, 'policy_family')
        assert hasattr(e, 'k_capacity')
        assert hasattr(e, 'cost_regime_id')
        assert hasattr(e, 'split')
        assert hasattr(e, 'scenario_id')
        assert hasattr(e, 'reset_seed')

    def test_no_duplicate_evaluation_episodes(self):
        """All evaluation episodes should be unique."""
        policy_families = [
            "corrective_only",
            "random_feasible",
            "age_threshold",
            "predicted_rul_threshold",
            "greedy_predicted_rul",
            "oracle_threshold",
        ]
        scenario_ids_by_split = {
            "predictor_train": [f"predictor_train_{i}" for i in range(5)],
            "rl_validation": [f"rl_validation_{i}" for i in range(5)],
        }

        episodes = generate_formal_evaluation_episodes(
            policy_families=policy_families,
            scenario_ids_by_split=scenario_ids_by_split,
            include_oracle=True,
        )

        unique_keys = set(
            (e.policy_family, e.k_capacity, e.cost_regime_id, e.split,
             e.scenario_id, e.reset_seed)
            for e in episodes
        )
        assert len(unique_keys) == len(episodes), "Duplicate episodes found"


class TestCountHelpers:
    """Test count helper functions."""

    def test_count_formal_tuning_without_oracle(self):
        """count_formal_tuning should return correct counts without oracle."""
        counts = count_formal_tuning(include_oracle=False)
        assert counts["candidate_count"] == 272
        assert counts["thresholds_per_k_regime"] == 34
        assert counts["policy_count"] == 3

    def test_count_formal_tuning_with_oracle(self):
        """count_formal_tuning should return correct counts with oracle."""
        counts = count_formal_tuning(include_oracle=True)
        assert counts["candidate_count"] == 360
        assert counts["thresholds_per_k_regime"] == 45
        assert counts["policy_count"] == 4

    def test_count_formal_evaluation_without_oracle(self):
        """count_formal_evaluation should return correct counts without oracle."""
        counts = count_formal_evaluation(include_oracle=False)
        assert counts["policy_count"] == 5
        assert counts["episode_count"] == 2000

    def test_count_formal_evaluation_with_oracle(self):
        """count_formal_evaluation should return correct counts with oracle."""
        counts = count_formal_evaluation(include_oracle=True)
        assert counts["policy_count"] == 6
        assert counts["episode_count"] == 2400


class TestOracleCandidateDiagnostic:
    """Test oracle candidate handling."""

    def test_oracle_candidates_excluded_by_default(self):
        """Oracle candidates should be excluded by default."""
        candidates = generate_formal_tuning_candidates()
        oracle_candidates = [c for c in candidates if c.policy_family == "oracle_threshold"]
        assert len(oracle_candidates) == 0

    def test_oracle_candidates_included_with_flag(self):
        """Oracle candidates should be included with include_oracle=True."""
        candidates = generate_formal_tuning_candidates(include_oracle=True)
        oracle_candidates = [c for c in candidates if c.policy_family == "oracle_threshold"]
        # 11 thresholds × 2 K × 4 regimes = 88
        assert len(oracle_candidates) == 88

    def test_oracle_diagnostic_status_clear(self):
        """Oracle candidates should be clearly diagnostic."""
        candidates = generate_formal_tuning_candidates(include_oracle=True)
        oracle_candidates = [c for c in candidates if c.policy_family == "oracle_threshold"]

        # Verify oracle candidates have proper identity
        for c in oracle_candidates:
            assert c.policy_family == "oracle_threshold"
            assert c.threshold in [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50]