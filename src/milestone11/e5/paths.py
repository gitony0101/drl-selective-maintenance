"""E5 (M12) frozen constants and output namespace.

E5 is the final controlled experiment of this DRL selective-maintenance project.
It tests ONE mechanistic hypothesis: adding real failure-event coverage to the
model-training dataset improves the downstream decision performance of the SAME
learned-dynamics H=2 MPC as E4.

E5-A is EXACTLY the frozen E4 Learned-MPC result (never retrained).
E5-B retrains the SAME architecture on a failure-enriched training dataset.

E5 writes ONLY under ``m12_e5_outputs/`` at the project root. It never
overwrites ``m11_e4_outputs/``. The learned-dynamics H=2 MPC planner is reused
VERBATIM from the frozen E4 module ``src.milestone11.e4.mpc`` (Section 12:
"Use EXACTLY the frozen E4 Learned MPC. Do NOT modify it.").
"""

from __future__ import annotations

from pathlib import Path

from src.runtime_paths import external_root

# ---- Frozen benchmark constants (identical to E4) ----
OBSERVATION_DIM = 10
FLEET_SIZE = 5
MAINTENANCE_CAPACITY = 2   # K
NUM_ACTIONS = 16
GAMMA = 0.95
EPISODE_HORIZON = 100
TRAINING_SPLIT = "predictor_train"
EVALUATION_SPLIT = "rl_validation"
TRAINING_BANK = "configs/scenarios/m5_pilot_k2.json"
EVALUATION_BANK = "configs/scenarios/m5_validation_k2.json"
COST_REGIME_ID = "failure-light-no-waste"

# Formal seeds (each uses its own frozen M9 prediction cache).
FORMAL_SEEDS = (6521, 6522, 6523, 6524, 6525)
# Frozen paired rl_validation reset seeds (one per evaluation scenario).
EVAL_RESET_SEEDS = (6521, 6522, 6523, 6524, 6525)

# ---- E5-B dataset contract (Section 4) ----
# Per-seed behavior-source episode counts (all complete 100-step episodes).
EPISODES_PER_SOURCE = {
    "random_feasible": 50,   # 5000 transitions
    "exact_myopic": 50,      # 5000 transitions (M4)
    "h2": 20,                # 2000 transitions (canonical H2)
    "no_maintenance": 30,    # 3000 transitions (a_t = 0 for every step)
}
SOURCE_TRANSITIONS = {s: n * EPISODE_HORIZON for s, n in EPISODES_PER_SOURCE.items()}
TOTAL_EPISODES_PER_SEED = sum(EPISODES_PER_SOURCE.values())   # 150
TOTAL_TRANSITIONS_PER_SEED = sum(SOURCE_TRANSITIONS.values()) # 15000

# The no-maintenance coverage policy is a_t = 0 at every step. It exists ONLY to
# induce genuine failure trajectories and failure-cost events in dynamics_train.
NO_MAINTENANCE_ACTION_ID = 0

# ---- E5-B episode-level split targets (Section 5) ----
SPLIT_EPISODE_TARGETS = {
    "dynamics_train": 120,        # 12000 transitions
    "dynamics_validate": 15,      # 1500  transitions
    "dynamics_holdout": 15,       # 1500  transitions
}
# Per-source episode split allocation (sum to 120/15/15; 80/10/10).
#   random_feasible 50 -> 40/5/5   (E4-identical per-scenario 8/1/1)
#   exact_myopic    50 -> 40/5/5   (E4-identical per-scenario 8/1/1)
#   h2              20 -> 16/2/2
#   no_maintenance  30 -> 24/3/3
SPLIT_EPISODES_PER_SOURCE = {
    "random_feasible": (40, 5, 5),
    "exact_myopic": (40, 5, 5),
    "h2": (16, 2, 2),
    "no_maintenance": (24, 3, 3),
}

# ---- Model (Section 8, identical to E4) ----
ENSEMBLE_SIZE = 3
INPUT_DIM = OBSERVATION_DIM + NUM_ACTIONS   # 26
HIDDEN_DIMS = (128, 128)
OUTPUT_DIM = OBSERVATION_DIM + 1            # 10 delta-obs + 1 reward = 11

# ---- Training (Section 9, identical to E4) ----
BATCH_SIZE = 256
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 20
LEARNING_RATE = 1e-3

# ---- Root paths ----
_CONTAINER_ROOT = external_root()
E5_OUTPUT_ROOT = _CONTAINER_ROOT / "m12_e5_outputs"
# E4-A control result lives in the FROZEN E4 output namespace.
E4_OUTPUT_ROOT = _CONTAINER_ROOT / "m11_e4_outputs"
# Frozen E3 comparator results (C-DDQN / M4 / H2) are taken verbatim.
E3_OUTPUT_ROOT = _CONTAINER_ROOT / "m10_e3_outputs"
M9_CACHE_ROOT = _CONTAINER_ROOT / "m9_point_caches"


def seed_cache_dir(seed: int) -> Path:
    """Absolute path of the frozen M9 prediction cache for ``seed``."""
    return (
        M9_CACHE_ROOT
        / f"seed_{seed}"
        / "data"
        / "processed"
        / "fd001"
        / "v2"
        / "06_PREDICTIONS"
        / f"seed_{seed}"
    )


def seed_output_dir(seed: int) -> Path:
    return E5_OUTPUT_ROOT / f"seed_{seed}"