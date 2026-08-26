"""
Shared pytest fixtures for Milestone 2 environment tests.

Provides reusable fixture helpers that:
- Read real unit IDs from PredictionStore (no unit_id + k inference)
- Verify all requested initial cycles exist
- Verify true_rul > 0 at every decision-boundary initial state
- Locate verified first failure cycle for a unit
- Never skip because of invalid unit assumptions

Non-fixture helper functions are in tests/m2_env_test_helpers.py.
"""

import pytest
from pathlib import Path
from typing import List

from src.predictors.prediction_store import load_default_prediction_store
from src.envs.scenario_bank import ScenarioBank

# Import non-fixture helpers from dedicated module
from tests.m2_env_test_helpers import (
    find_unit_with_failure_at_cycle,
    build_failure_fixture_scenario,
    build_scenario_bank_for_split,
    RecordingPredictionStore,
)


@pytest.fixture(scope="session")
def prediction_store():
    """Load the V2 prediction store (session-scoped for efficiency)."""
    cache_dir = Path("data/processed/fd001/v2/06_PREDICTIONS/")
    return load_default_prediction_store(cache_dir)


@pytest.fixture(scope="session")
def predictor_train_units(prediction_store) -> List[int]:
    """Get all unit IDs for predictor_train split."""
    return prediction_store.get_units("predictor_train")


@pytest.fixture(scope="session")
def rl_validation_units(prediction_store) -> List[int]:
    """Get all unit IDs for rl_validation split."""
    return prediction_store.get_units("rl_validation")


@pytest.fixture(scope="session")
def rl_test_units(prediction_store) -> List[int]:
    """Get all unit IDs for rl_test split."""
    return prediction_store.get_units("rl_test")


@pytest.fixture
def validation_k1_scenario_bank(prediction_store) -> ScenarioBank:
    """Load or build rl_validation K=1 smoke scenario bank."""
    return build_scenario_bank_for_split(
        split="rl_validation",
        prediction_store=prediction_store,
        k_capacity=1,
        num_scenarios=3,
        bank_id_suffix="_k1",
    )


@pytest.fixture
def predictor_train_k1_scenario_bank(prediction_store) -> ScenarioBank:
    """Load or build predictor_train K=1 smoke scenario bank."""
    return build_scenario_bank_for_split(
        split="predictor_train",
        prediction_store=prediction_store,
        k_capacity=1,
        num_scenarios=3,
        bank_id_suffix="_k1",
    )


@pytest.fixture
def rl_test_k1_scenario_bank(prediction_store) -> ScenarioBank:
    """Load or build rl_test K=1 smoke scenario bank."""
    return build_scenario_bank_for_split(
        split="rl_test",
        prediction_store=prediction_store,
        k_capacity=1,
        num_scenarios=3,
        bank_id_suffix="_k1",
    )