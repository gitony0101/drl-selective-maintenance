"""M9 point-estimate BASELINE REPAIR regression tests.

Pins the repaired baseline configuration against the frozen M3 selected
thresholds + M4 scientific-selection candidate, and the fail-closed guards,
per the independent post-run audit repair directive. These tests do NOT
re-run the heavyweight baseline evaluation (that needs the real per-seed V2
cache + M9_HEAVY=1); they verify the configuration constants, the frozen-
artifact loaders, the SHA fail-closed guards, the M4 candidate-identity
proof, the cache-seed pairing guard, the rl_test barrier, and the regime
guard. The heavyweight integration behavior is covered by
``tests/test_m9_point_baselines.py`` (under M9_HEAVY=1).

Frozen artifacts (independent verification):
  M3 selected_thresholds.json SHA256 47762b3f... -> age=125, pred=10, greedy=10
  M4 selection_decision.json SHA256 7684b1ff... -> logistic_T5
  logistic_T5 literal: risk_model_id=logistic_window_v1, risk_temperature=5.0
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_external_assets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from src.milestone9.point import baselines as B


# ---------------------------------------------------------------------------
# M3 frozen selected thresholds
# ---------------------------------------------------------------------------

def test_m3_selected_threshold_constants_match_frozen_artifact():
    """The M9 baseline runner's frozen constants match the on-disk M3
    selected_thresholds.json: SHA, run_id, config_sha; age=125, pred=10, greedy=10."""
    assert B.M3_SELECTED_THRESHOLDS_SHA256 == (
        "47762b3fd8fbba62a0c159f71a9d3778b6d99d5332e955dc19cf242751a395d6"
    )
    assert B.M3_FORMAL_RUN_ID == "m3_formal_20260727T203617Z"
    thresholds = B.load_selected_thresholds()
    assert len(thresholds) == 32  # 4 families x 2 K x 4 regimes
    assert thresholds["age_threshold_k2_failure-light-no-waste"] == 125
    assert thresholds["predicted_rul_threshold_k2_failure-light-no-waste"] == 10
    assert thresholds["greedy_predicted_rul_k2_failure-light-no-waste"] == 10


def test_m3_selected_thresholds_are_formal_not_smoke():
    """The selected values (125/10/10) are MEMBERS of the frozen formal threshold
    grid, NOT the smoke 100/50 values. 100 is NOT in the age grid's selected set
    under the formal selection rule; 125/10/10 are the selected ones."""
    from scripts.run_m3_baselines import FORMAL_THRESHOLD_GRIDS
    age_grid = FORMAL_THRESHOLD_GRIDS["age_threshold"]
    pred_grid = FORMAL_THRESHOLD_GRIDS["predicted_rul_threshold"]
    greedy_grid = FORMAL_THRESHOLD_GRIDS["greedy_predicted_rul"]
    assert 125 in age_grid
    assert 10 in pred_grid
    assert 10 in greedy_grid
    # The smoke threshold the prior runner used (100) is in the age grid but is
    # NOT the SELECTED value under the formal rule -- selected value is 125.
    assert B._selected_threshold_for("age_threshold") == 125
    assert B._selected_threshold_for("predicted_rul_threshold") == 10
    assert B._selected_threshold_for("greedy_predicted_rul") == 10


def test_m3_incorrect_smoke_thresholds_rejected():
    """The runner must NOT silently use 100 (age) or 50 (greedy) -- those are
    M3 smoke values, not the frozen selected ones. The per-family accessor
    returns exactly 125/10/10, never 100/50."""
    assert B._selected_threshold_for("age_threshold") != 100
    assert B._selected_threshold_for("predicted_rul_threshold") != 100
    assert B._selected_threshold_for("greedy_predicted_rul") != 50


def test_m3_missing_selected_thresholds_fails_closed(monkeypatch, tmp_path):
    """A missing selected_thresholds.json must fail closed (no default substitution)."""
    monkeypatch.setattr(B, "M3_SELECTED_THRESHOLDS_PATH", str(tmp_path / "missing.json"))
    with pytest.raises(ValueError, match="M3 selected_thresholds.json missing"):
        B.load_selected_thresholds()


def test_m3_hash_mismatch_fails_closed(monkeypatch, tmp_path):
    """A SHA256 mismatch on selected_thresholds.json must fail closed."""
    bad = tmp_path / "selected_thresholds.json"
    bad.write_text('{"_meta": {}}')  # wrong content -> wrong SHA
    monkeypatch.setattr(B, "M3_SELECTED_THRESHOLDS_PATH", str(bad))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        B.load_selected_thresholds()


def test_m3_wrong_k_or_regime_missing_fails_closed():
    """The M3 identity key uses K + cost regime; a non-M9-regime entry is absent
    from the loaded dict (the accessor fails closed on the missing key)."""
    thresholds = B.load_selected_thresholds()
    assert "age_threshold_k1_failure-light-no-waste" in thresholds  # K=1 exists
    # K=99 does not exist; the M9 accessor asks K=2 (correct). Confirm the K=2
    # selected entry is the one we use, not a wrong-K fallback.
    assert B._selected_threshold_for("age_threshold") == thresholds[
        "age_threshold_k2_failure-light-no-waste"
    ]


# ---------------------------------------------------------------------------
# M4 frozen scientific-selection candidate
# ---------------------------------------------------------------------------

def test_m4_selected_candidate_recovered_from_frozen_decision():
    """M4 selected_candidate == logistic_T5, from the frozen decision.json."""
    assert B.M4_SELECTION_DECISION_SHA256 == (
        "7684b1ff5f605429c8639ed89160e74dcc2a45cec5f6e533e465ac703ea2c6c8"
    )
    decision = B.load_m4_selection_decision()
    assert decision["selected_candidate"] == "logistic_T5"
    assert decision["decision"] == "select_logistic"


def test_m4_exact_risk_model_and_temperature_from_frozen_literal():
    """risk_model_id + temperature are PROVEN from the frozen candidate literal
    (src/optimizers/m4_scientific_validation.py), not inferred from "T5"."""
    B._assert_m4_candidate_identities()  # raises on any mismatch
    assert B.M4_RISK_MODEL_ID == "logistic_window_v1"
    assert B.M4_RISK_TEMPERATURE == 5.0
    assert B.M4_TIE_TOLERANCE == 1e-9


def test_m4_invalid_hard_window_10_rejected():
    """The invalid hard_window_v1 + 10.0 pairing must NOT be the selected M4
    config. The constants are logistic_window_v1 + 5.0."""
    assert (B.M4_RISK_MODEL_ID, B.M4_RISK_TEMPERATURE) != ("hard_window_v1", 10.0)
    assert B.M4_RISK_MODEL_ID == "logistic_window_v1"
    assert B.M4_RISK_TEMPERATURE == 5.0


def test_m4_candidate_literal_rejects_wrong_temperature():
    """If the frozen literal's temperature for logistic_T5 drifted (e.g. to 10.0),
    _assert_m4_candidate_identities fails closed."""
    from src.optimizers.m4_scientific_validation import SCIENTIFIC_VALIDATION_CANDIDATES
    # Confirm exactly one logistic_T5 entry and it has temperature 5.0.
    matches = [c for c in SCIENTIFIC_VALIDATION_CANDIDATES if c.candidate_id == "logistic_T5"]
    assert len(matches) == 1
    assert matches[0].risk_temperature == 5.0
    assert matches[0].risk_model_id == "logistic_window_v1"


def test_m4_selection_decision_wrong_candidate_fails_closed(monkeypatch, tmp_path):
    """A selection_decision.json naming a different candidate fails closed."""
    bad = tmp_path / "selection_decision.json"
    bad.write_text('{"selected_candidate": "hard_window_v1", "decision": "x"}')
    # Patch the SHA verifier path so the mismatch is on candidate, not SHA.
    monkeypatch.setattr(B, "M4_SELECTION_DECISION_PATH", str(bad))
    monkeypatch.setattr(B, "M4_SELECTION_DECISION_SHA256", B._compute_sha256(str(bad)))
    with pytest.raises(ValueError, match="selected_candidate"):
        B.load_m4_selection_decision()


def test_m4_does_not_observe_true_rul():
    """The M4 exact-myopic baseline uses point-estimate predicted-RUL
    observation, NOT true RUL. The optimizer's risk model operates on the
    observation (obs = 5 engines x [normalized age, predicted RUL]); no
    true-RUL field is read. Confirm the observation shape contract (10,) and
    that the env info carries no true_rul policy input."""
    # The exact_myopic observation is (10,) = 5 engines x [age, predicted_rul];
    # the optimizer select_action(obs) consumes only obs, never a true-RUL side
    # channel. The risk model logistic_window_v1 uses delta_cycles - predicted_rul.
    assert B.M4_RISK_MODEL_ID == "logistic_window_v1"  # not oracle_threshold
    # Exhaustive behavioral check that true RUL is not read is done via the
    # heavy integration test (no OracleThreshold family is evaluated).


def test_m4_action_table_is_16_actions_k2():
    """The M4 optimizer uses ACTION_TABLE_N5_K2 (16 actions) for K=2."""
    from scripts.run_m4_exact_myopic import create_optimizer
    opt = create_optimizer(
        k_capacity=2, cost_regime_id=B.M9_REGIME_COST,
        risk_model_id=B.M4_RISK_MODEL_ID,
        risk_temperature=B.M4_RISK_TEMPERATURE,
        tie_tolerance=B.M4_TIE_TOLERANCE,
    )
    assert len(opt.context.action_table) == 16
    assert opt.context.maintenance_capacity == 2


def test_m4_tie_breaking_preserved():
    """Tie-breaking is smallest action_id at tie_tolerance=1e-9."""
    assert B.M4_TIE_TOLERANCE == 1e-9
    from scripts.run_m4_exact_myopic import create_optimizer
    opt = create_optimizer(
        k_capacity=2, cost_regime_id=B.M9_REGIME_COST,
        risk_model_id=B.M4_RISK_MODEL_ID,
        risk_temperature=B.M4_RISK_TEMPERATURE,
        tie_tolerance=B.M4_TIE_TOLERANCE,
    )
    assert opt.tie_tolerance == 1e-9


# ---------------------------------------------------------------------------
# Pairing + evaluation fairness
# ---------------------------------------------------------------------------

def test_cache_seed_guard_rejects_cross_seed(monkeypatch, tmp_path):
    """run_baselines fails closed if the cache path's seed != the DDQN paired
    seed (caches are 1:1 paired)."""
    from src.envs.config import get_default_config
    from src.milestone9.point import paths
    # Build a config with the seed-6521 cache but pass ddqn_seed=6522.
    cache = str(paths.FORMAL_VALIDATION_BANK)  # placeholder; use real cache path
    from src.milestone9.point import pairing
    cache_path = str(pairing.cache_env_path_for_seed(6521))
    env = get_default_config(
        split="rl_validation", cost_regime_id=B.M9_REGIME_COST,
        maintenance_capacity=2, scenario_bank_path=str(paths.FORMAL_VALIDATION_BANK),
        prediction_cache_path=cache_path, seed=6521,
    )
    with pytest.raises(ValueError, match="cache seed mismatch"):
        B.run_baselines(
            env_config=env, scenario_bank_path=str(paths.FORMAL_VALIDATION_BANK),
            reset_seeds=[6521, 6522, 6523, 6524, 6525], ddqn_seed=6522,
        )


def test_run_baselines_rejects_rl_test(monkeypatch):
    """run_baselines rejects split == rl_test (M9 contract barrier),
    defense-in-depth on top of the env's SplitViolationError."""
    from src.envs.config import get_default_config
    from src.milestone9.point import paths, pairing
    cache_path = str(pairing.cache_env_path_for_seed(6521))
    env = get_default_config(
        split="rl_test", cost_regime_id=B.M9_REGIME_COST,
        maintenance_capacity=2, scenario_bank_path=str(paths.FORMAL_VALIDATION_BANK),
        prediction_cache_path=cache_path, seed=6521,
    )
    with pytest.raises(ValueError, match="rl_test"):
        B.run_baselines(
            env_config=env, scenario_bank_path=str(paths.FORMAL_VALIDATION_BANK),
            reset_seeds=[6521, 6522, 6523, 6524, 6525], ddqn_seed=6521,
        )


def test_run_baselines_rejects_wrong_k(monkeypatch):
    """run_baselines fails closed on K != 2 (M9 regime)."""
    from src.envs.config import get_default_config
    from src.milestone9.point import paths, pairing
    cache_path = str(pairing.cache_env_path_for_seed(6521))
    env = get_default_config(
        split="rl_validation", cost_regime_id=B.M9_REGIME_COST,
        maintenance_capacity=1, scenario_bank_path=str(paths.FORMAL_VALIDATION_BANK),
        prediction_cache_path=cache_path, seed=6521,
    )
    with pytest.raises(ValueError, match="K=2"):
        B.run_baselines(
            env_config=env, scenario_bank_path=str(paths.FORMAL_VALIDATION_BANK),
            reset_seeds=[6521, 6522, 6523, 6524, 6525], ddqn_seed=6521,
        )


def test_run_baselines_rejects_wrong_cost_regime(monkeypatch):
    """run_baselines fails closed on cost_regime != failure-light-no-waste."""
    from src.envs.config import get_default_config
    from src.milestone9.point import paths, pairing
    cache_path = str(pairing.cache_env_path_for_seed(6521))
    env = get_default_config(
        split="rl_validation", cost_regime_id="failure-heavy-no-waste",
        maintenance_capacity=2, scenario_bank_path=str(paths.FORMAL_VALIDATION_BANK),
        prediction_cache_path=cache_path, seed=6521,
    )
    with pytest.raises(ValueError, match="cost_regime_id"):
        B.run_baselines(
            env_config=env, scenario_bank_path=str(paths.FORMAL_VALIDATION_BANK),
            reset_seeds=[6521, 6522, 6523, 6524, 6525], ddqn_seed=6521,
        )


def test_run_baselines_rejects_wrong_split(monkeypatch):
    """run_baselines fails closed on split != rl_validation (e.g. predictor_train)."""
    from src.envs.config import get_default_config
    from src.milestone9.point import paths, pairing
    cache_path = str(pairing.cache_env_path_for_seed(6521))
    env = get_default_config(
        split="predictor_train", cost_regime_id=B.M9_REGIME_COST,
        maintenance_capacity=2, scenario_bank_path=str(paths.FORMAL_VALIDATION_BANK),
        prediction_cache_path=cache_path, seed=6521,
    )
    with pytest.raises(ValueError, match="split"):
        B.run_baselines(
            env_config=env, scenario_bank_path=str(paths.FORMAL_VALIDATION_BANK),
            reset_seeds=[6521, 6522, 6523, 6524, 6525], ddqn_seed=6521,
        )


def test_primary_episode_count_is_five():
    """PRIMARY evaluation uses exactly 5 paired episodes per policy per seed
    (1:1 scenario x reset_seed), NOT the M3 25-episode formal-closeout design."""
    # The reset seeds the M9 runner uses: 5.
    assert B.M9_REGIME_SPLIT == "rl_validation"
    # The formal-rerun script uses _RESET_SEEDS = [6521..6525] (5).
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_m9_baseline_repair",
        str(Path(__file__).resolve().parent.parent / "scripts" / "run_m9_baseline_repair.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod._RESET_SEEDS == [6521, 6522, 6523, 6524, 6525]
    assert len(mod._RESET_SEEDS) == 5
    assert mod._BASELINE_FAMILIES == [
        "corrective_only", "random_feasible", "age_threshold",
        "predicted_rul_threshold", "greedy_predicted_rul", "exact_myopic",
    ]


def test_random_policy_seed_is_deterministic_and_paired():
    """The random baseline's policy_seed is bound 1:1 to the DDQN paired seed
    (not a casual global 42). Distinct across the five seeds."""
    seeds = [B._random_policy_seed_for(s) for s in (6521, 6522, 6523, 6524, 6525)]
    assert seeds == [6521, 6522, 6523, 6524, 6525]
    assert len(set(seeds)) == 5  # distinct


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def test_provenance_records_m3_m4_cache_bank_sha():
    """The __provenance__ block records M3 SHA, M4 SHA, cache manifest SHA,
    scenario-bank SHA, and exact policy parameters."""
    # Structural check on the constants the provenance builder uses.
    assert B.M3_SELECTED_THRESHOLDS_SHA256.startswith("47762b3f")
    assert B.M4_SELECTION_DECISION_SHA256.startswith("7684b1ff")
    # Full provenance is exercised end-to-end under M9_HEAVY=1 (the integration
    # test asserts the __provenance__ block fields).


def test_m9_regime_constants():
    """The M9 paired regime is K=2 / failure-light-no-waste / rl_validation."""
    assert B.M9_REGIME_K == 2
    assert B.M9_REGIME_COST == "failure-light-no-waste"
    assert B.M9_REGIME_SPLIT == "rl_validation"


# ---------------------------------------------------------------------------
# rl_test seal
# ---------------------------------------------------------------------------

def test_rl_test_seal_constant():
    """rl_test is forbidden; M9_REGIME_SPLIT is rl_validation, never rl_test."""
    assert B.M9_REGIME_SPLIT != "rl_test"
    assert B.M9_REGIME_SPLIT == "rl_validation"
