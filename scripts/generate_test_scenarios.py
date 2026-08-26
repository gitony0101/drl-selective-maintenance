"""
Generate minimal test scenario banks for Milestone 2 environment testing.

Creates smoke scenarios for predictor_train, rl_validation, and rl_test splits.
These are NOT scientific experiment banks - they are for testing environment mechanics only.
"""

import json
from pathlib import Path

from src.envs.scenario_bank import Scenario, ScenarioBank, save_scenario_bank
from src.envs.config import get_default_config
from src.envs.costs import list_cost_regimes


def get_valid_units_for_split(split: str) -> list[int]:
    """Get valid unit IDs for a split from the actual PredictionStore."""
    from src.predictors.prediction_store import load_default_prediction_store
    from pathlib import Path

    cache_dir = Path("data/processed/fd001/v2/06_PREDICTIONS/")
    store = load_default_prediction_store(cache_dir)
    return store.get_units(split)


def create_test_scenario_bank(
    split: str,
    output_path: Path,
    num_scenarios: int = 5,
    k_value: int = 2,
    horizon: int = 100,
):
    """Create a test scenario bank for a given split."""

    valid_units = get_valid_units_for_split(split)
    if len(valid_units) < 5:
        raise ValueError(f"Not enough units in split {split}: got {len(valid_units)}, need at least 5")

    scenarios = []

    for i in range(num_scenarios):
        # Select 5 distinct units deterministically
        start_idx = (i * 5) % len(valid_units)
        selected_units = []
        for j in range(5):
            idx = (start_idx + j) % len(valid_units)
            selected_units.append(valid_units[idx])

        # Vary initial cycles slightly for diversity
        initial_cycles = tuple(1 + (i * 10) % 50 for _ in range(5))

        scenario = Scenario(
            scenario_id=f"{split}_smoke_{i:03d}",
            split=split,
            initial_unit_ids=tuple(selected_units),
            initial_cycles=initial_cycles,
            replacement_seed=6521 + i,
            environment_seed=6521 + i,
            episode_horizon=horizon,
            maintenance_capacity=k_value,
            cost_regime_id="failure-light-no-waste",
        )
        scenarios.append(scenario)

    bank = ScenarioBank(
        bank_id=f"{split}_smoke_bank",
        split=split,
        scenarios=tuple(scenarios),
    )

    save_scenario_bank(bank, output_path)
    print(f"Created {output_path} with {len(scenarios)} scenarios")

    return bank


def main():
    """Generate all test scenario banks."""
    output_dir = Path("data/scenario_banks")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Main configuration: N=5, K=2, horizon=100
    create_test_scenario_bank(
        split="predictor_train",
        output_path=output_dir / "predictor_train_smoke.json",
        num_scenarios=5,
        k_value=2,
        horizon=100,
    )

    create_test_scenario_bank(
        split="rl_validation",
        output_path=output_dir / "rl_validation_smoke.json",
        num_scenarios=5,
        k_value=2,
        horizon=100,
    )

    create_test_scenario_bank(
        split="rl_test",
        output_path=output_dir / "rl_test_smoke.json",
        num_scenarios=5,
        k_value=2,
        horizon=100,
    )

    # K=1 sensitivity scenarios
    create_test_scenario_bank(
        split="rl_validation",
        output_path=output_dir / "rl_validation_k1_smoke.json",
        num_scenarios=3,
        k_value=1,
        horizon=100,
    )

    print("\nAll test scenario banks created successfully!")


if __name__ == "__main__":
    main()