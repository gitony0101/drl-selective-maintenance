"""E3 (M10) M9-seed-cache H2 PlannerContext adapter.

Canonical M6 ``H2Planner`` requires a ``PlannerContext`` whose ``__post_init__``
hard-locks ``prediction_cache_manifest_sha256`` to the historical M5 cache
manifest SHA ``007c36af...`` (see ``src/m6/contract.py``). The M9 point-estimate
pipeline uses FIVE per-seed prediction caches, each with its OWN manifest file
SHA256 (e.g. seed 6521 = ``046e3318...``). Those actual values are deliberately
NOT ``007c36af...``.

This module supplies an explicit M9-specific ``M9PlannerContext`` that:

- keeps every *algorithmically-consumed* field identical to canonical M6
  (K/delta/rul scale/age scale/action table/cost regime/risk/gamma/R1_hat),
  so ``H2Planner``/``ForwardModel`` run unchanged; and
- validates ``prediction_cache_manifest_sha256`` against the ACTUAL per-seed
  M9 cache manifest file SHA256 (truthful provenance), never against a copy of
  ``007c36af...``.

It does NOT weaken the canonical ``PlannerContext`` contract: ``contract.py`` is
left byte-identical, the canonical validator is untouched, and only this new
M9 class overrides the manifest expectation for the genuine per-seed cache.

R1_hat is computed per-seed from that seed's ``predictor_train`` cache using the
same mathematical definition M6 uses (mean ``predicted_rul`` over all
``cycle == 1`` records).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from src.envs.action_table import ACTION_TABLE_N5_K2
from src.runtime_paths import external_root

from src.m6.contract import (
    PlannerContext,
    COST_REGIMES,
    ACTION_TABLE_K1_SHA256,
    ACTION_TABLE_K2_SHA256,
    ENVIRONMENT_CONTRACT_ID,
    M5_OBSERVATION_SCHEMA_ID,
    M4_RISK_MODEL_ID,
    M4_RISK_TEMPERATURE,
    M4_DELTA_CYCLES,
    M5_GAMMA,
    IdentityMismatchError,
    ContractViolationError,
)

# External git-ignored per-seed M9 cache layout (mirrors src/milestone9/point/pairing).
_CONTAINER_ROOT = external_root()
CACHE_ROOT = _CONTAINER_ROOT / "m9_point_caches"
# Nested per-seed dir mirrors src/milestone9/point/pairing.cache_env_path_for_seed.
_CACHE_REL = Path("data") / "processed" / "fd001" / "v2" / "06_PREDICTIONS"

FORMAL_SEEDS = (6521, 6522, 6523, 6524, 6525)


def seed_cache_dir(seed: int) -> Path:
    """The env-visible per-seed cache dir (nested ``seed_<s>`` under V2 path)."""
    return CACHE_ROOT / f"seed_{seed}" / _CACHE_REL / f"seed_{seed}"


def seed_cache_manifest_path(seed: int) -> Path:
    return seed_cache_dir(seed) / "prediction_cache_manifest_v2.json"


def seed_cache_parquet_path(seed: int) -> Path:
    return seed_cache_dir(seed) / "fd001_prediction_cache_v2.parquet"


def manifest_file_sha256(seed: int) -> str:
    """SHA256 (hex) of the actual per-seed ``prediction_cache_manifest_v2.json``.

    This is the truthful identity of the M9 seed cache referenced by the H2
    context's ``prediction_cache_manifest_sha256`` field.
    """
    p = seed_cache_manifest_path(seed)
    if not p.exists():
        raise FileNotFoundError(f"M9 seed cache manifest missing: {p}")
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compute_r1_hat_cycles(seed: int) -> Dict[str, Union[float, str, int]]:
    """Compute R1_hat for ``seed`` using the M6 mathematical definition.

    R1_hat = mean ``predicted_rul`` over all ``predictor_train`` cache records
    with ``cycle == 1``. Returns fields for the context's R1_hat_provenance.
    """
    import pandas as pd

    p = seed_cache_parquet_path(seed)
    if not p.exists():
        raise FileNotFoundError(f"M9 seed cache parquet missing: {p}")
    df = pd.read_parquet(p)
    tr = df[df["split"] == "predictor_train"]
    if "cycle" not in tr.columns or "predicted_rul" not in tr.columns:
        raise ContractViolationError("cache parquet lacks required columns")
    cycle1 = tr[tr["cycle"] == 1]
    n = len(cycle1)
    if n == 0:
        raise ContractViolationError(
            f"zero cycle==1 records in seed {seed} predictor_train; R1_hat fails closed"
        )
    r1 = float(np.mean(cycle1["predicted_rul"].astype(float)))
    return {
        "R1_hat_cycles": r1,
        "predictor_train_manifest_sha256": manifest_file_sha256(seed),
        "computed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_cycle1_records": int(n),
    }


@dataclass(frozen=True)
class M9PlannerContext(PlannerContext):
    """M9-seed-cache PlannerContext.

    Field-for-field same algorithmic contract as canonical ``PlannerContext``;
    the ONLY divergence is that ``prediction_cache_manifest_sha256`` must equal
    the ACTUAL M9 per-seed cache manifest file SHA256 (validated on
    construction). This keeps provenance truthful while leaving the canonical
    M6 contract and validator untouched.
    """

    # Expected ACTUAL per-seed manifest SHA256 (set by the factory).
    expected_m9_manifest_sha256: str = ""

    def __post_init__(self) -> None:
        """Faithful mirror of ``PlannerContext.__post_init__`` (contract.py:106)
        except ``prediction_cache_manifest_sha256`` is validated against the
        ACTUAL M9 per-seed manifest SHA256, not the historical M5 constant."""
        # --- Canonical field checks (unchanged) ---
        if self.maintenance_capacity not in (1, 2):
            raise IdentityMismatchError(
                f"maintenance_capacity must be 1 or 2, got {self.maintenance_capacity}"
            )
        if self.delta_cycles != 5:
            raise IdentityMismatchError(f"delta_cycles must be 5, got {self.delta_cycles}")
        if self.rul_scale != 125.0:
            raise IdentityMismatchError(f"rul_scale must be 125.0, got {self.rul_scale}")
        if self.age_scale_cycles != 341:
            raise IdentityMismatchError(
                f"age_scale_cycles must be 341, got {self.age_scale_cycles}"
            )
        if self.gamma != 0.95:
            raise IdentityMismatchError(f"gamma must be 0.95, got {self.gamma}")
        if self.risk_model_id != M4_RISK_MODEL_ID:
            raise IdentityMismatchError(
                f"risk_model_id must be '{M4_RISK_MODEL_ID}', got '{self.risk_model_id}'"
            )
        if self.risk_temperature != M4_RISK_TEMPERATURE:
            raise IdentityMismatchError(
                f"risk_temperature must be {M4_RISK_TEMPERATURE}, got {self.risk_temperature}"
            )
        if self.observation_schema_id != M5_OBSERVATION_SCHEMA_ID:
            raise IdentityMismatchError(
                f"observation_schema_id must be '{M5_OBSERVATION_SCHEMA_ID}'"
            )
        if self.environment_contract_id != ENVIRONMENT_CONTRACT_ID:
            raise IdentityMismatchError(
                f"environment_contract_id must be '{ENVIRONMENT_CONTRACT_ID}'"
            )

        # --- M9-manifest check (the intentional M9 divergence) ---
        if self.expected_m9_manifest_sha256 == "":
            raise IdentityMismatchError(
                "M9PlannerContext requires expected_m9_manifest_sha256"
            )
        if self.prediction_cache_manifest_sha256 != self.expected_m9_manifest_sha256:
            raise IdentityMismatchError(
                f"prediction_cache_manifest_sha256 mismatch: expected ACTUAL M9 "
                f"seed-cache manifest {self.expected_m9_manifest_sha256}, got "
                f"{self.prediction_cache_manifest_sha256}"
            )

        # --- Canonical cost-regime checks (unchanged) ---
        if self.cost_regime_id not in COST_REGIMES:
            raise IdentityMismatchError(
                f"cost_regime_id must be one of {list(COST_REGIMES.keys())}, "
                f"got '{self.cost_regime_id}'"
            )
        regime = COST_REGIMES[self.cost_regime_id]
        if self.c_pm != regime["c_pm"]:
            raise IdentityMismatchError(
                f"c_pm mismatch for regime {self.cost_regime_id}: expected "
                f"{regime['c_pm']}, got {self.c_pm}"
            )
        if self.c_f != regime["c_f"]:
            raise IdentityMismatchError(
                f"c_f mismatch for regime {self.cost_regime_id}: expected "
                f"{regime['c_f']}, got {self.c_f}"
            )
        if self.c_u != regime["c_u"]:
            raise IdentityMismatchError(
                f"c_u mismatch for regime {self.cost_regime_id}: expected "
                f"{regime['c_u']}, got {self.c_u}"
            )

        # --- Canonical action-table hash check (unchanged) ---
        expected_hash = (
            ACTION_TABLE_K1_SHA256 if self.maintenance_capacity == 1
            else ACTION_TABLE_K2_SHA256
        )
        if self.action_table_sha256 != expected_hash:
            raise IdentityMismatchError(
                f"action_table_sha256 mismatch for K={self.maintenance_capacity}: "
                f"expected {expected_hash}, got {self.action_table_sha256}"
            )

        # --- Canonical horizon check (unchanged) ---
        if self.horizon not in (0, 1, 2):
            raise IdentityMismatchError(
                f"horizon must be 0, 1, or 2, got {self.horizon}"
            )

        # --- Canonical H2-specific validation (unchanged) ---
        if self.horizon == 2:
            if self.R1_hat_cycles is None:
                raise IdentityMismatchError("R1_hat_cycles required for H2 (horizon=2)")
            if self.R1_hat_provenance is None:
                raise IdentityMismatchError("R1_hat_provenance required for H2 (horizon=2)")
            required_keys = {
                "predictor_train_manifest_sha256",
                "computed_at_utc",
                "n_cycle1_records",
            }
            if not required_keys.issubset(self.R1_hat_provenance.keys()):
                raise IdentityMismatchError(
                    f"R1_hat_provenance missing required keys: "
                    f"{required_keys - self.R1_hat_provenance.keys()}"
                )
            # R1_hat provenance manifest must also match the ACTUAL seed manifest.
            if (
                self.R1_hat_provenance["predictor_train_manifest_sha256"]
                != self.expected_m9_manifest_sha256
            ):
                raise IdentityMismatchError(
                    "R1_hat_provenance predictor_train_manifest_sha256 does not "
                    "match the ACTUAL M9 seed-cache manifest"
                )
        else:
            if self.R1_hat_cycles is not None:
                raise IdentityMismatchError(
                    f"R1_hat_cycles must be null for horizon={self.horizon}"
                )
            if self.R1_hat_provenance is not None:
                raise IdentityMismatchError(
                    f"R1_hat_provenance must be null for horizon={self.horizon}"
                )

        if not self.forbid_rl_test:
            raise IdentityMismatchError("forbid_rl_test must be True")


def build_m9_planner_context_h2(
    seed: int,
    cost_regime_id: str,
    cache_root: Path = CACHE_ROOT,
) -> M9PlannerContext:
    """Build a validated M9PlannerContext for ``seed``'s actual M9 cache.

    Reads the seed cache's manifest file SHA256 (truthful provenance) and
    computes R1_hat from the same cache, then returns a context ready for the
    canonical ``H2Planner``/``ForwardModel`` (which are unchanged).
    """
    if seed not in FORMAL_SEEDS:
        raise ValueError(f"seed {seed} not in formal seeds {list(FORMAL_SEEDS)}")
    if cost_regime_id not in COST_REGIMES:
        raise IdentityMismatchError(
            f"cost_regime_id must be one of {list(COST_REGIMES.keys())}"
        )

    manifest_sha = manifest_file_sha256(seed)
    r1 = compute_r1_hat_cycles(seed)

    # E3 frozen benchmark is always K=2 (configs/scenarios/m5_pilot_k2.json).
    action_table = ACTION_TABLE_N5_K2
    action_table_hash = ACTION_TABLE_K2_SHA256
    regime = COST_REGIMES[cost_regime_id]

    ctx = M9PlannerContext(
        maintenance_capacity=2,
        delta_cycles=M4_DELTA_CYCLES,
        rul_scale=125.0,
        age_scale_cycles=341,
        action_table=action_table,
        action_table_sha256=action_table_hash,
        cost_regime_id=cost_regime_id,
        c_pm=regime["c_pm"],
        c_f=regime["c_f"],
        c_u=regime["c_u"],
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
            "predictor_train_manifest_sha256": r1["predictor_train_manifest_sha256"],
            "computed_at_utc": r1["computed_at_utc"],
            "n_cycle1_records": r1["n_cycle1_records"],
        },
        expected_m9_manifest_sha256=manifest_sha,
    )
    return ctx


def cost_regime_for_m9(regime_id: str) -> Dict[str, float]:
    """Expose a frozen cost regime dict (identical to M6)."""
    if regime_id not in COST_REGIMES:
        raise IdentityMismatchError(
            f"unknown cost regime '{regime_id}'"
        )
    return COST_REGIMES[regime_id].copy()