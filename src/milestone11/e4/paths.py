"""E4 (M11) frozen constants and output namespace.

Implements the frozen benchmark identity and the external (git-ignored) output
namespace for E4. E4 never overwrites E3 artifacts; it writes only under
``m11_e4_outputs/`` at the project root.
"""

from __future__ import annotations

from pathlib import Path

from src.runtime_paths import external_root

# ---- Frozen benchmark constants (from the E4 preregistration) ----
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

# ---- E4 dataset contract ----
N_EPISODES_PER_SOURCE = 50        # complete 100-step episodes per behavior source
TRANSITIONS_PER_SOURCE = 5000     # 50 * 100
N_SOURCES = 3                     # random_feasible, exact_myopic (M4), h2 (H2)
TOTAL_EPISODES_PER_SEED = 150     # 3 * 50
TOTAL_TRANSITIONS_PER_SEED = 15000

# Behavior-source episode split (Section 8).
N_TRAIN_EPISODES = 40
N_VAL_EPISODES = 5
N_HOLDOUT_EPISODES = 5

# ---- Model (Section 10) ----
ENSEMBLE_SIZE = 3
INPUT_DIM = OBSERVATION_DIM + NUM_ACTIONS   # 26
HIDDEN_DIMS = (128, 128)
OUTPUT_DIM = OBSERVATION_DIM + 1            # 10 delta-obs + 1 reward = 11

# ---- Training (Section 11) ----
BATCH_SIZE = 256
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 20
LEARNING_RATE = 1e-3

# ---- Root paths ----
_CONTAINER_ROOT = external_root()
E4_OUTPUT_ROOT = _CONTAINER_ROOT / "m11_e4_outputs"
M9_CACHE_ROOT = _CONTAINER_ROOT / "m9_point_caches"
E3_OUTPUT_ROOT = _CONTAINER_ROOT / "m10_e3_outputs"


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
    return E4_OUTPUT_ROOT / f"seed_{seed}"