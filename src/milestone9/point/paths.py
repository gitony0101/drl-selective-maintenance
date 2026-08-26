"""M9 point-estimate frozen regime-matched scenario-bank paths.

Single source of truth for the two scenario-bank paths every M9 training
command MUST pass as explicit CLI flags. The frozen V2 trainer resolver's
``assert_explicit_banks`` gate (``src/training/resolver.py:193-220``) rejects
ANY command — dry-run, validate-only, smoke, real training, resume — that
omits ``--training-scenario-bank`` and ``--validation-scenario-bank``. There
is NO opt-out; the gate is source="argparse namespace" and refuses even a
config whose ``environment`` section already carries the bank paths. So the
wrapper MUST forward both bank paths explicitly.

These bank paths are FROZEN (not derived): they are the regime-matched
(K=2, failure-light-no-waste) banks the M8 formal and M5 validated surface
already bind. The wrapper never invents banks; it consumes these frozen
literals and passes them through to the frozen CLIs as explicit flags.

  - TRAINING_BANK (predictor_train split, K=2, failure-light-no-waste):
    ``configs/scenarios/m5_pilot_k2.json`` — used by BOTH pilot and formal.
  - FORMAL_VALIDATION_BANK (rl_validation split, K=2, failure-light-no-waste):
    ``configs/scenarios/m5_validation_k2.json`` — the formal config
    ``ddqn_v1.json``'s environment-scoped validation bank. Used for formal runs.
  - PILOT_VALIDATION_BANK (rl_validation split, K=2, failure-light-no-waste):
    ``configs/scenarios/m5_validation_k2__light.json`` — the pilot validation
    bank. pilot. (Both validation banks carry rl_validation scenarios
    in the failure-light-no-waste regime at K=2; either is regime-correct.
    The pilot uses the ``__light`` variant by the preregistered pilot
    convention; the formal run binds the ``ddqn_v1.json``-named bank.)
"""

from __future__ import annotations

from .manifest import REPO_ROOT


TRAINING_BANK = "configs/scenarios/m5_pilot_k2.json"
FORMAL_VALIDATION_BANK = "configs/scenarios/m5_validation_k2.json"
PILOT_VALIDATION_BANK = "configs/scenarios/m5_validation_k2__light.json"


def training_bank_path() -> str:
    return str(TRAINING_BANK)


def formal_validation_bank_path() -> str:
    return str(FORMAL_VALIDATION_BANK)


def pilot_validation_bank_path() -> str:
    return str(PILOT_VALIDATION_BANK)


def resolve_bank_path(rel: str) -> str:
    """Resolve a frozen relative bank path to absolute, asserting it exists on
    disk at REPO_ROOT. Fail-closed if a required bank is missing — a missing
    bank would block the frozen resolver's preflight before any training."""
    abs_path = REPO_ROOT / rel
    if not abs_path.exists():
        raise FileNotFoundError(
            f"M9 paths: required scenario bank missing on disk: {abs_path}"
        )
    return str(abs_path)
