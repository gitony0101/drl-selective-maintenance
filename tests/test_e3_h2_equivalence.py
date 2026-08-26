"""
E3 Part 2-3 tests: M9-seed H2 context adapter + behavioral equivalence.

Part 2: the M9PlannerContext adapter must not weaken canonical M6 provenance;
it must validate ``prediction_cache_manifest_sha256`` against the ACTUAL M9
per-seed cache manifest file SHA256 (truthful), and reject a fake/copied
``007c36af...`` value.

Part 3: given the SAME algorithmic state (same K/cost/risk/gamma/R1_hat), the
canonical H2Planner driven by a canonical context and by the M9 adapter must
return IDENTICAL selected action_id, per-action J2, ranking, and tie-break.
Differing only in the provenance-only manifest field, the H2 output must not
change.
"""

from __future__ import annotations

import numpy as np
import pytest
pytestmark = pytest.mark.requires_external_assets

from src.m6.contract import (
    M5_PREDICTION_CACHE_MANIFEST_SHA256,
    PlannerContext,
    IdentityMismatchError,
)
from src.m6.context import build_planner_context_h2
from src.m6.h2_planner import H2Planner
from src.milestone10.e3.h2_context import (
    FORMAL_SEEDS,
    build_m9_planner_context_h2,
    compute_r1_hat_cycles,
    manifest_file_sha256,
)


# Lazy cache SHA dictionary: computed on first access to avoid import-time I/O.
_CACHE_SHA_CACHE = {}

def _cache_sha_for_seed(seed: int) -> str:
    if seed not in _CACHE_SHA_CACHE:
        _CACHE_SHA_CACHE[seed] = manifest_file_sha256(seed)
    return _CACHE_SHA_CACHE[seed]


def _build_m9_ctx_with_manifest(seed: int, regime: str, manifest_sha: str):
    """
    Directly build an M9PlannerContext claiming ``manifest_sha`` as the cache
    manifest, with the ACTUAL seed manifest as the ground-truth expectation. Its
    ``__post_init__`` rejects any claimed value that is not the ACTUAL seed
    manifest.
    """
    from src.milestone10.e3.h2_context import M9PlannerContext
    from src.envs.action_table import ACTION_TABLE_N5_K2
    from src.m6.contract import (
        ACTION_TABLE_K2_SHA256,
        COST_REGIMES,
        ENVIRONMENT_CONTRACT_ID,
        M5_OBSERVATION_SCHEMA_ID,
        M4_RISK_MODEL_ID,
        M4_RISK_TEMPERATURE,
        M4_DELTA_CYCLES,
        M5_GAMMA,
    )

    r1 = compute_r1_hat_cycles(seed)
    reg = COST_REGIMES[regime]
    return M9PlannerContext(
        maintenance_capacity=2,
        delta_cycles=M4_DELTA_CYCLES,
        rul_scale=125.0,
        age_scale_cycles=341,
        action_table=ACTION_TABLE_N5_K2,
        action_table_sha256=ACTION_TABLE_K2_SHA256,
        cost_regime_id=regime,
        c_pm=reg["c_pm"],
        c_f=reg["c_f"],
        c_u=reg["c_u"],
        risk_model_id=M4_RISK_MODEL_ID,
        risk_temperature=M4_RISK_TEMPERATURE,
        gamma=M5_GAMMA,
        observation_schema_id=M5_OBSERVATION_SCHEMA_ID,
        environment_contract_id=ENVIRONMENT_CONTRACT_ID,
        prediction_cache_manifest_sha256=manifest_sha,
        horizon=2,
        forbid_rl_test=True,
        R1_hat_cycles=r1["R1_hat_cycles"],
        R1_hat_provenance={
            "predictor_train_manifest_sha256": manifest_sha,
            "computed_at_utc": r1["computed_at_utc"],
            "n_cycle1_records": r1["n_cycle1_records"],
        },
        expected_m9_manifest_sha256=_cache_sha_for_seed(seed),
    )


class _CanonicalContextFactory:
    """
    Build a canonical PlannerContext at an algorithmic state equal to a seed's
    M9 adapter (same R1_hat), carrying the canonical M6 manifest constant. This
    context ALWAYS exists: canonical __post_init__ does not cross-check the
    R1_hat_provenance manifest against the top-level manifest.
    """

    @staticmethod
    def for_seed(seed: int, regime: str) -> PlannerContext:
        r1 = compute_r1_hat_cycles(seed)
        return build_planner_context_h2(
            maintenance_capacity=2,
            cost_regime_id=regime,
            R1_hat_cycles=r1["R1_hat_cycles"],
            R1_hat_provenance={
                "predictor_train_manifest_sha256": r1["predictor_train_manifest_sha256"],
                "computed_at_utc": r1["computed_at_utc"],
                "n_cycle1_records": r1["n_cycle1_records"],
            },
        )


class TestPart2ManifestCompatibility:
    def test_adapter_uses_actual_m9_manifest_not_007c(self):
        # The adapter must truthfully carry the ACTUAL seed manifest SHA, never
        # the historical M6 007c36af constant or a copy of it.
        for seed in FORMAL_SEEDS:
            ctx = build_m9_planner_context_h2(seed, "failure-light-no-waste")
            actual = _cache_sha_for_seed(seed)
            assert len(ctx.prediction_cache_manifest_sha256) == 64
            assert ctx.prediction_cache_manifest_sha256 == actual
            assert ctx.prediction_cache_manifest_sha256 != M5_PREDICTION_CACHE_MANIFEST_SHA256
            assert (
                ctx.R1_hat_provenance["predictor_train_manifest_sha256"]
                == ctx.prediction_cache_manifest_sha256
            )

    def test_adapter_rejects_historical_m6_manifest(self):
        # The canonical M6 007c36af value (NOT the actual seed manifest) must be
        # rejected by the M9 adapter's construction path.
        seed = FORMAL_SEEDS[0]
        with pytest.raises(IdentityMismatchError):
            _build_m9_ctx_with_manifest(
                seed, "failure-light-no-waste", M5_PREDICTION_CACHE_MANIFEST_SHA256
            )

    def test_adapter_rejects_wrong_seed_manifest(self):
        # Cross-seed provenance must fail closed.
        s1, s2 = FORMAL_SEEDS[0], FORMAL_SEEDS[1]
        with pytest.raises(IdentityMismatchError):
            _build_m9_ctx_with_manifest(s1, "failure-light-no-waste", _cache_sha_for_seed(s2))

    def test_algorithms_per_seed_r1hat_are_seed_dependent(self):
        # R1_hat is a genuinely per-seed algorithmic input; at least two seeds
        # must differ so per-seed provenance is real, not a constant.
        rhats = {s: compute_r1_hat_cycles(s)["R1_hat_cycles"] for s in FORMAL_SEEDS}
        assert len(set(round(v, 6) for v in rhats.values())) >= 2


class TestPart3BehavioralEquivalence:
    @pytest.mark.parametrize("seed", FORMAL_SEEDS)
    @pytest.mark.parametrize("regime", ["failure-light-no-waste", "failure-heavy-waste-aware"])
    def test_identical_action_and_j2(self, seed, regime):
        canon = H2Planner(_CanonicalContextFactory.for_seed(seed, regime))
        m9 = H2Planner(build_m9_planner_context_h2(seed, regime))
        rng = np.random.default_rng(seed)
        for _ in range(40):
            o = rng.random(10).astype(np.float32)
            rc, rm = canon.plan(o), m9.plan(o)
            assert rc.action_id == rm.action_id, (seed, rc.action_id, rm.action_id)
            jc = [a.J2 for a in rc.per_action]
            jm = [a.J2 for a in rm.per_action]
            for c, m in zip(jc, jm):
                assert float(c) == float(m), (seed, c, m)
            # ranking identical => argmin order identical
            order_c = sorted(range(len(jc)), key=lambda i: (float(jc[i]), i))
            order_m = sorted(range(len(jm)), key=lambda i: (float(jm[i]), i))
            assert order_c == order_m
            assert rc.provenance["branch_prob_sum"] == rm.provenance["branch_prob_sum"]
            assert rc.selected_slots == rm.selected_slots