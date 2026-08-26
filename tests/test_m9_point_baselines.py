"""M9 Point-Estimate -- per-seed-cache baseline runner.

The pilot/formal baselines must be evaluated on the SAME rl_validation
scenarios against the SAME per-seed V2 prediction cache the DDQN trained on
(apples-to-apples vs ``validation_metrics.json``'s mean_total_cost).

The frozen M3 CLI (``scripts/run_m3_baselines.py --mode formal_closeout``) and
the M4 CLI (``scripts/run_m4_exact_myopic.py --evaluate``) are hard-wired to
the PRODUCTION V2 prediction cache dir (M3 via ``get_default_config`` default;
M4 via ``run_m4_production_smoke_matrix``'s ``prediction_cache_dir =
repo_root / "data" / "processed" / "fd001" / "v2" / "06_PREDICTIONS"`` at
``run_m4_production_smoke.py:1194``). The production cache parquet does NOT
exist in a minimal checkout, and the contract FORBIDS writing to the production
V2 dir. M3's ``formal_closeout`` mode also demands a sealed
``formal_run_context.json`` + ``selected_thresholds.json`` payload that the
M9 pilot has not staged.

Rather than surgically threading a ``--prediction-cache-path`` override through
M3's 4 env-construction call sites + M4's production matrix (and around the
sealed-context gate), this module reuses the FROZEN baseline primitives
directly -- ``src.baselines.evaluator.PolicyEvaluator`` (the same evaluator
M3 uses), the 5 non-oracle rule policies from ``src.baselines.rule_policies``
(CorrectiveOnly, RandomFeasible, AgeThreshold, PredictedRULThreshold,
GreedyPredictedRUL), and ``src.optimizers.exact_myopic.ExactMyopicOptimizer``
(the same optimizer M4 uses) -- against a per-seed ``EnvironmentConfig``
built from ``get_default_config`` with the per-seed cache + scenario bank.

This honors the M9 contract's authoritative baseline set (§line 118: "from
``src/baselines/rule_policies.py`` ... and ``M4 exact-myopic`` via
``src/optimizers/exact_myopic.py``") AND the design directive to
"plumb baselines to the per-seed cache" without modifying frozen scripts or
touching the production V2 cache dir.

Risk model: ``hard_window_v1`` (the M4 default; deterministic point estimate,
NO uncertainty risk cache -- matches the point-estimate M9 contract).
OracleThreshold is NEVER run (diagnostic-only per the contract).

Every episode is driven by the SAME env config + SAME 5 scenarios + SAME
``FIXED_RESET_SEEDS=[6521..6525]`` used by the DDQN evaluation, so the
baseline ``mean_total_cost`` is directly comparable to the DDQN
``validation_metrics.json``'s ``mean_total_cost``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_external_assets
from src.runtime_paths import external_root as _EXTERNAL

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTAINER_ROOT = _EXTERNAL()
CACHE_ROOT = CONTAINER_ROOT / "m9_point_caches"

_HEAVY = os.environ.get("M9_HEAVY") == "1"
pytestmark = pytest.mark.skipif(
    not _HEAVY,
    reason="M9 baseline runner integration test needs the real per-seed V2 "
           "cache (set M9_HEAVY=1 to run); requires the seed-6521 cache at "
           "m9_point_caches/seed_6521/... to exist",
)

# The 6 baseline policy families this runner evaluates (the M9 contract's
# authoritative non-oracle set). OracleThreshold is NEVER included.
EXPECTED_POLICY_FAMILIES = [
    "corrective_only",
    "random_feasible",
    "age_threshold",
    "predicted_rul_threshold",
    "greedy_predicted_rul",
    "exact_myopic",
]

# The frozen M3 reset seeds (scripts/run_m3_baselines.py:96).
EXPECTED_RESET_SEEDS = [6521, 6522, 6523, 6524, 6525]


def _seed_6521_cache_env_path() -> Path:
    from src.milestone9.point import pairing
    return pairing.cache_env_path_for_seed(6521)


@pytest.fixture(scope="module")
def baseline_results():
    """Module-scoped: run the heavyweight run_baselines ONCE per session
    (drives 6 policies x 5 reset seeds x up to episode_horizon=100 steps),
    shared across the 3 assertions below. Builds the env config from the
    seed-6521 runtime config's environment block so the baselines evaluate
    on the SAME split/bank/horizon/K/cost-regime the DDQN did (apples-to-
    apples vs validation_metrics.json mean_total_cost=30.0)."""
    from src.milestone9.point.baselines import run_baselines, env_config_for_eval

    cache_path = str(_seed_6521_cache_env_path())
    assert (Path(cache_path) / "prediction_cache_manifest_v2.json").exists(), (
        f"seed-6521 cache manifest missing at {cache_path}; run "
        f"scripts/run_m9_point_estimate.py --seed 6521 --phase pilot first"
    )
    # Read the EXACT environment the seed-6521 DDQN evaluated under so the
    # baselines run on an identical evaluation context. The pilot used the
    # light validation bank (m5_validation_k2__light.json) + horizon=50.
    runtime_cfg_path = (
        CONTAINER_ROOT / "m9_point_runs" / "pilot"
        / "m9_point_mse_control_seed6521" / "runtime_config.json"
    )
    assert runtime_cfg_path.exists(), (
        f"seed-6521 runtime config missing at {runtime_cfg_path}; the pilot "
        f"training must have run first to produce the evaluation context"
    )
    import json as _json
    runtime_cfg = _json.loads(runtime_cfg_path.read_text())
    env_config = env_config_for_eval(runtime_cfg)
    # The baselines evaluate on the DDQN's validation bank (the light bank
    # for the pilot), NOT the formal bank.
    return run_baselines(
        env_config=env_config,
        scenario_bank_path=env_config.scenario_bank_path,
        reset_seeds=EXPECTED_RESET_SEEDS,
    )


def test_baseline_runner_returns_one_entry_per_authoritative_policy(baseline_results):
    """run_baselines returns a dict with one entry per authoritative
    non-oracle policy family (5 rule families + exact_myopic = 6), each
    carrying the aggregate metrics needed for the apples-to-apples DDQN
    comparison."""
    # One entry per authoritative policy family (plus the __provenance__ block).
    family_keys = {k for k in baseline_results if k != "__provenance__"}
    assert family_keys == set(EXPECTED_POLICY_FAMILIES), (
        f"baseline policy families mismatch: got {sorted(family_keys)}"
    )
    assert "__provenance__" in baseline_results, "missing __provenance__ block"

    # Each entry carries the aggregate metrics comparable to the DDQN's
    # validation_metrics.json (mean_total_cost, total_failures,
    # total_pm_actions, num_episodes).
    for family, rec in baseline_results.items():
        if family == "__provenance__":
            continue  # provenance is not a policy family
        assert "mean_total_cost" in rec, f"{family} missing mean_total_cost"
        assert "total_failures" in rec, f"{family} missing total_failures"
        assert "total_pm_actions" in rec, f"{family} missing total_pm_actions"
        assert "num_episodes" in rec, f"{family} missing num_episodes"
        assert isinstance(rec["mean_total_cost"], (int, float))
        assert rec["num_episodes"] == len(EXPECTED_RESET_SEEDS), (
            f"{family} num_episodes={rec['num_episodes']} != "
            f"{len(EXPECTED_RESET_SEEDS)} (one episode per reset seed)"
        )
        # Cost is non-negative (costs are non-negative in this env).
        assert rec["mean_total_cost"] >= 0, (
            f"{family} mean_total_cost={rec['mean_total_cost']} < 0"
        )

    # Frozen M3 FORMAL selected thresholds (NOT smoke 100/50): age=125,
    # predicted_rul=10, greedy activation=10 (K=2 / failure-light-no-waste).
    assert baseline_results["age_threshold"]["threshold"] == 125
    assert baseline_results["predicted_rul_threshold"]["threshold"] == 10
    assert baseline_results["greedy_predicted_rul"]["activation_threshold"] == 10

    # exact_myopic provenance: frozen M4 scientific-selection candidate.
    assert "provenance" in baseline_results["exact_myopic"]
    prov = baseline_results["exact_myopic"]["provenance"]
    assert prov["risk_model_id"] == "logistic_window_v1"
    assert prov["risk_temperature"] == 5.0
    assert prov["selected_candidate"] == "logistic_T5"
    assert prov["maintenance_capacity"] == 2
    assert prov["cost_regime_id"] == "failure-light-no-waste"
    # The runner records the per-seed cache path it was Driving (must match
    # the seed-6521 cache path it was constructed against).
    assert prov["prediction_cache_path"] == str(_seed_6521_cache_env_path())
    # The scenario bank is the validation bank from the seed-6521 runtime
    # config (the pilot's light validation bank), NOT the formal bank.
    assert prov["scenario_bank_path"].endswith("m5_validation_k2__light.json")
    assert "action_table" in prov  # identity of the frozen K=2

    # Unified __provenance__ block (M3 frozen selected thresholds + M4 selection).
    uprov = baseline_results["__provenance__"]
    m3 = uprov["m3_selected_thresholds"]
    assert m3["sha256"] == "47762b3fd8fbba62a0c159f71a9d3778b6d99d5332e955dc19cf242751a395d6"
    assert m3["age_threshold_k2_failure-light-no-waste"] == 125
    assert m3["predicted_rul_threshold_k2_failure-light-no-waste"] == 10
    assert m3["greedy_predicted_rul_k2_failure-light-no-waste"] == 10
    m4 = uprov["m4_scientific_selection"]
    assert m4["decision_sha256"] == "7684b1ff5f605429c8639ed89160e74dcc2a45cec5f6e533e465ac703ea2c6c8"
    assert m4["selected_candidate"] == "logistic_T5"
    assert m4["risk_model_id"] == "logistic_window_v1"
    assert m4["risk_temperature"] == 5.0
    assert uprov["m9_regime"]["k"] == 2
    assert uprov["m9_regime"]["split"] == "rl_validation"


def test_baseline_runner_corrective_only_matches_ddqn_pilot_no_pm(baseline_results):
    """SANITY CHECK vs the seed-6521 pilot DDQN validation_metrics.json:
    the pilot DDQN at 5000 steps never learned PM (total_pm_actions=0,
    total_failures=30, mean_total_cost=30.0 over 5 episodes -- it always
    picks the empty action and lets all engines fail). The
    CorrectiveOnly baseline ALSO does zero PM (it only ever acts on a
    failed engine), so its total_pm_actions MUST == 0. This is a
    mechanistic invariant, not a scientific claim about DDQN quality."""
    assert baseline_results["corrective_only"]["total_pm_actions"] == 0


def test_baseline_runner_rejects_rl_test_split():
    """The M9 contract FORBIDS rl_test access. run_baselines MUST refuse an
    env_config whose split is rl_test (defense-in-depth on top of the frozen
    env's own SplitViolationError)."""
    from src.milestone9.point.baselines import run_baselines
    from src.envs.config import get_default_config
    from src.milestone9.point import paths

    env_config = get_default_config(
        split="rl_test",
        cost_regime_id="failure-light-no-waste",
        maintenance_capacity=2,
        scenario_bank_path=str(paths.FORMAL_VALIDATION_BANK),
        prediction_cache_path=str(_seed_6521_cache_env_path()),
        seed=6521,
    )
    with pytest.raises(ValueError, match="rl_test"):
        run_baselines(
            env_config=env_config,
            scenario_bank_path=str(paths.FORMAL_VALIDATION_BANK),
            reset_seeds=EXPECTED_RESET_SEEDS,
        )
