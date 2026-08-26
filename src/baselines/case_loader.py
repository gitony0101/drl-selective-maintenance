"""
Centralized case loader for Milestone 3 experiments.

This module provides a single helper for loading and deriving scenario banks
used by:
- Smoke tests
- Threshold tuning
- Final evaluation
- Artifact validation probes

The loader:
1. Rejects rl_test split before loading any file
2. Loads source scenario banks (K=2) from JSON
3. Derives K=1 scenarios in-memory from K=2 source
4. Derives all four cost-regime variants in-memory
5. Ensures every derived Scenario has matching split, K, and cost_regime_id
6. Preserves source JSON files unchanged
7. Returns scenario IDs belonging to the derived bank
8. Never reuses scenario IDs across incompatible splits
9. Provides stable derived scenario IDs with explicit provenance

Usage:
    from src.baselines.case_loader import load_cases

    # Load for predictor_train, K=1, failure-light-no-waste
    scenario_ids = load_cases(
        split="predictor_train",
        k=1,
        cost_regime_id="failure-light-no-waste",
        source_bank_path="data/scenario_banks/predictor_train_smoke.json"
    )

    # Load for rl_validation, K=2, failure-heavy-waste-aware
    scenario_ids = load_cases(
        split="rl_validation",
        k=2,
        cost_regime_id="failure-heavy-waste-aware",
        source_bank_path="data/scenario_banks/rl_validation_smoke.json"
    )
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..envs.scenario_bank import Scenario, ScenarioBank, load_scenario_bank
from ..envs.costs import (
    FAILURE_LIGHT_NO_WASTE,
    FAILURE_HEAVY_NO_WASTE,
    FAILURE_LIGHT_WASTE_AWARE,
    FAILURE_HEAVY_WASTE_AWARE,
)

# All four frozen cost regimes
ALL_COST_REGIMES = frozenset({
    "failure-light-no-waste",
    "failure-heavy-no-waste",
    "failure-light-waste-aware",
    "failure-heavy-waste-aware",
})

# Allowed splits (excludes rl_test for tuning/evaluation)
ALLOWED_EXPERIMENT_SPLITS = frozenset({"predictor_train", "rl_validation"})


class RlTestBarrierError(Exception):
    """Raised when rl_test split is requested."""

    pass


class CaseLoadError(Exception):
    """Raised when case loading fails."""

    pass


@dataclass(frozen=True)
class CaseLoadResult:
    """Result of a case load operation."""

    split: str
    k: int
    cost_regime_id: str
    scenario_ids: Tuple[str, ...]
    source_bank_path: str
    derived_from_k: Optional[int] = None  # None if loaded directly, 2 if derived from K=2
    provenance: str = ""  # Human-readable provenance
    bank_sha256: Optional[str] = None  # SHA256 of source bank file
    bank_scenario_count: Optional[int] = None  # Number of scenarios in source bank
    # Additional formal provenance fields
    source_file_size: Optional[int] = None  # Source bank file size in bytes
    derived_scenario_count: Optional[int] = None  # Number of derived scenarios
    derived_bank_sha256: Optional[str] = None  # SHA256 of derived scenario content
    logical_bank_id: Optional[str] = None  # Derived bank logical ID


def _derive_k1_scenarios(
    source_bank: ScenarioBank,
) -> ScenarioBank:
    """
    Derive K=1 scenarios from a K=2 source bank.

    Creates new scenarios with:
    - scenario_id = original_id with "_k1" suffix
    - maintenance_capacity = 1
    - All other fields preserved

    Args:
        source_bank: Source scenario bank (assumed K=2)

    Returns:
        New ScenarioBank with K=1 derived scenarios
    """
    derived_scenarios = []
    for scenario in source_bank.scenarios:
        derived = Scenario(
            scenario_id=f"{scenario.scenario_id}_k1",
            split=scenario.split,
            initial_unit_ids=scenario.initial_unit_ids,
            initial_cycles=scenario.initial_cycles,
            replacement_seed=scenario.replacement_seed,
            environment_seed=scenario.environment_seed,
            episode_horizon=scenario.episode_horizon,
            maintenance_capacity=1,
            cost_regime_id=scenario.cost_regime_id,
        )
        derived_scenarios.append(derived)

    return ScenarioBank(
        bank_id=f"{source_bank.bank_id}_k1_derived",
        split=source_bank.split,
        scenarios=tuple(derived_scenarios),
    )


def _derive_cost_regime_scenarios(
    source_bank: ScenarioBank,
    target_cost_regime_id: str,
) -> ScenarioBank:
    """
    Derive scenarios with a specific cost regime.

    Creates new scenarios with:
    - scenario_id = original_id with "_<regime>" suffix
    - cost_regime_id = target_cost_regime_id
    - All other fields preserved

    Args:
        source_bank: Source scenario bank
        target_cost_regime_id: Target cost regime ID

    Returns:
        New ScenarioBank with derived cost regime scenarios
    """
    derived_scenarios = []
    for scenario in source_bank.scenarios:
        derived = Scenario(
            scenario_id=f"{scenario.scenario_id}_{target_cost_regime_id}",
            split=scenario.split,
            initial_unit_ids=scenario.initial_unit_ids,
            initial_cycles=scenario.initial_cycles,
            replacement_seed=scenario.replacement_seed,
            environment_seed=scenario.environment_seed,
            episode_horizon=scenario.episode_horizon,
            maintenance_capacity=scenario.maintenance_capacity,
            cost_regime_id=target_cost_regime_id,
        )
        derived_scenarios.append(derived)

    return ScenarioBank(
        bank_id=f"{source_bank.bank_id}_{target_cost_regime_id}_derived",
        split=source_bank.split,
        scenarios=tuple(derived_scenarios),
    )


def _derive_k1_and_cost_regime_scenarios(
    source_bank: ScenarioBank,
    target_cost_regime_id: str,
) -> ScenarioBank:
    """
    Derive scenarios with K=1 and a specific cost regime.

    Args:
        source_bank: Source scenario bank (assumed K=2)
        target_cost_regime_id: Target cost regime ID

    Returns:
        New ScenarioBank with K=1 and derived cost regime
    """
    derived_scenarios = []
    for scenario in source_bank.scenarios:
        derived = Scenario(
            scenario_id=f"{scenario.scenario_id}_k1_{target_cost_regime_id}",
            split=scenario.split,
            initial_unit_ids=scenario.initial_unit_ids,
            initial_cycles=scenario.initial_cycles,
            replacement_seed=scenario.replacement_seed,
            environment_seed=scenario.environment_seed,
            episode_horizon=scenario.episode_horizon,
            maintenance_capacity=1,
            cost_regime_id=target_cost_regime_id,
        )
        derived_scenarios.append(derived)

    return ScenarioBank(
        bank_id=f"{source_bank.bank_id}_k1_{target_cost_regime_id}_derived",
        split=source_bank.split,
        scenarios=tuple(derived_scenarios),
    )


def load_cases(
    split: str,
    k: int,
    cost_regime_id: str,
    source_bank_path: Optional[str] = None,
) -> CaseLoadResult:
    """
    Load and derive scenarios for a specific split, K, and cost regime.

    This is the centralized helper used by smoke, tuning, evaluation, and
    artifact validation.

    Args:
        split: Environment split ("predictor_train" or "rl_validation")
        k: Maintenance capacity (1 or 2)
        cost_regime_id: Cost regime ID (one of four frozen regimes)
        source_bank_path: Path to source K=2 JSON bank. If None, uses default path.

    Returns:
        CaseLoadResult with scenario IDs and provenance

    Raises:
        RlTestBarrierError: If rl_test split is requested
        CaseLoadError: If loading fails
    """
    # Barrier 1: Reject rl_test before any file loading
    if split == "rl_test":
        raise RlTestBarrierError(
            "rl_test split is forbidden. "
            "Use predictor_train or rl_validation only."
        )

    # Validate split
    if split not in ALLOWED_EXPERIMENT_SPLITS:
        raise CaseLoadError(
            f"Invalid split '{split}'. "
            f"Allowed: {sorted(ALLOWED_EXPERIMENT_SPLITS)}"
        )

    # Validate K
    if k not in {1, 2}:
        raise CaseLoadError(
            f"Invalid K={k}. Must be 1 or 2."
        )

    # Validate cost regime
    if cost_regime_id not in ALL_COST_REGIMES:
        raise CaseLoadError(
            f"Invalid cost regime '{cost_regime_id}'. "
            f"Allowed: {sorted(ALL_COST_REGIMES)}"
        )

    # Determine source bank path
    if source_bank_path is None:
        # Default path based on split
        source_bank_path = f"data/scenario_banks/{split}_smoke.json"

    source_path = Path(source_bank_path)
    if not source_path.exists():
        # Try alternate location
        alt_path = Path(__file__).parent.parent.parent / source_bank_path
        if alt_path.exists():
            source_path = alt_path
        else:
            raise CaseLoadError(
                f"Source bank not found at {source_bank_path}"
            )

    # Load source bank (K=2)
    source_bank = load_scenario_bank(source_path)

    # Validate source bank split
    if source_bank.split != split:
        raise CaseLoadError(
            f"Source bank split mismatch: expected '{split}', "
            f"got '{source_bank.split}'"
        )

    # Derive based on target K and cost regime
    if k == 2:
        # K=2: just derive cost regime
        derived_bank = _derive_cost_regime_scenarios(source_bank, cost_regime_id)
        derived_from_k = None
        provenance = (
            f"Loaded K=2 scenarios from {source_path}, "
            f"derived cost_regime='{cost_regime_id}'"
        )
    else:
        # K=1: derive both K and cost regime
        derived_bank = _derive_k1_and_cost_regime_scenarios(source_bank, cost_regime_id)
        derived_from_k = 2
        provenance = (
            f"Loaded K=2 scenarios from {source_path}, "
            f"derived K=1 and cost_regime='{cost_regime_id}'"
        )

    # Validate all derived scenarios
    for scenario in derived_bank.scenarios:
        if scenario.split != split:
            raise CaseLoadError(
                f"Derived scenario {scenario.scenario_id} has split mismatch: "
                f"expected '{split}', got '{scenario.split}'"
            )
        if scenario.maintenance_capacity != k:
            raise CaseLoadError(
                f"Derived scenario {scenario.scenario_id} has K mismatch: "
                f"expected {k}, got {scenario.maintenance_capacity}"
            )
        if scenario.cost_regime_id != cost_regime_id:
            raise CaseLoadError(
                f"Derived scenario {scenario.scenario_id} has cost regime mismatch: "
                f"expected '{cost_regime_id}', got '{scenario.cost_regime_id}'"
            )

    # Compute SHA256 of source bank file
    with open(source_path, "rb") as f:
        bank_sha256 = hashlib.sha256(f.read()).hexdigest()

    # Compute source file size
    source_file_size = source_path.stat().st_size

    # Compute SHA256 of derived scenario content (canonical form)
    # Hash the sorted scenario IDs and their key attributes
    derived_content = "".join(
        f"{s.scenario_id}|{s.split}|{s.maintenance_capacity}|{s.cost_regime_id}\n"
        for s in sorted(derived_bank.scenarios, key=lambda s: s.scenario_id)
    )
    derived_bank_sha256 = hashlib.sha256(derived_content.encode()).hexdigest()

    return CaseLoadResult(
        split=split,
        k=k,
        cost_regime_id=cost_regime_id,
        scenario_ids=tuple(s.scenario_id for s in derived_bank.scenarios),
        source_bank_path=str(source_path),
        derived_from_k=derived_from_k,
        provenance=provenance,
        bank_sha256=bank_sha256,
        bank_scenario_count=len(source_bank.scenarios),
        source_file_size=source_file_size,
        derived_scenario_count=len(derived_bank.scenarios),
        derived_bank_sha256=derived_bank_sha256,
        logical_bank_id=derived_bank.bank_id,
    )


def load_all_regimes(
    split: str,
    k: int,
    source_bank_path: Optional[str] = None,
) -> Dict[str, CaseLoadResult]:
    """
    Load scenarios for all four cost regimes.

    Args:
        split: Environment split
        k: Maintenance capacity
        source_bank_path: Path to source K=2 JSON bank

    Returns:
        Dict mapping cost_regime_id to CaseLoadResult
    """
    results = {}
    for regime_id in ALL_COST_REGIMES:
        results[regime_id] = load_cases(
            split=split,
            k=k,
            cost_regime_id=regime_id,
            source_bank_path=source_bank_path,
        )
    return results


def get_scenario_bank_for_case(
    split: str,
    k: int,
    cost_regime_id: str,
    source_bank_path: Optional[str] = None,
) -> ScenarioBank:
    """
    Load and derive a full ScenarioBank for a specific case.

    This is a convenience wrapper that returns the full ScenarioBank
    instead of just scenario IDs.

    Args:
        split: Environment split
        k: Maintenance capacity
        cost_regime_id: Cost regime ID
        source_bank_path: Path to source K=2 JSON bank

    Returns:
        ScenarioBank ready for environment use
    """
    result = load_cases(
        split=split,
        k=k,
        cost_regime_id=cost_regime_id,
        source_bank_path=source_bank_path,
    )

    # Load or derive the bank
    source_path = Path(result.source_bank_path)
    source_bank = load_scenario_bank(source_path)

    if k == 2:
        return _derive_cost_regime_scenarios(source_bank, cost_regime_id)
    else:
        return _derive_k1_and_cost_regime_scenarios(source_bank, cost_regime_id)


def get_k1_from_k2_source(
    source_bank: ScenarioBank,
) -> ScenarioBank:
    """
    Derive K=1 scenarios from a K=2 source bank.

    This is exposed for testing and for cases where the caller
    already has the source bank loaded.

    Args:
        source_bank: K=2 source ScenarioBank

    Returns:
        Derived K=1 ScenarioBank
    """
    return _derive_k1_scenarios(source_bank)


def verify_predictor_train_k1_derivation() -> str:
    """
    Verify that predictor_train K=1 can be derived from K=2 source.

    Returns:
        Provenance string if successful

    Raises:
        CaseLoadError: If derivation fails
    """
    try:
        result = load_cases(
            split="predictor_train",
            k=1,
            cost_regime_id="failure-light-no-waste",
            source_bank_path="data/scenario_banks/predictor_train_smoke.json",
        )
        if len(result.scenario_ids) == 0:
            raise CaseLoadError("predictor_train K=1 derivation produced 0 scenarios")
        return result.provenance
    except RlTestBarrierError:
        raise  # Re-raise barrier errors
    except Exception as e:
        raise CaseLoadError(f"predictor_train K=1 derivation failed: {e}")


def verify_rl_validation_k1_derivation() -> str:
    """
    Verify that rl_validation K=1 can be derived from K=2 source.

    Note: The rl_validation source bank may already be K=1.
    This function handles both cases.

    Returns:
        Provenance string if successful

    Raises:
        CaseLoadError: If derivation fails
    """
    source_path = Path("data/scenario_banks/rl_validation_smoke.json")
    if not source_path.exists():
        alt_path = Path(__file__).parent.parent.parent / source_path
        if alt_path.exists():
            source_path = alt_path
        else:
            raise CaseLoadError(f"Source bank not found at {source_path}")

    source_bank = load_scenario_bank(source_path)

    # Check if source is already K=1
    source_k = source_bank.scenarios[0].maintenance_capacity if source_bank.scenarios else 2

    if source_k == 1:
        # Source is already K=1, just derive cost regime
        derived = _derive_cost_regime_scenarios(source_bank, "failure-light-no-waste")
        if len(derived.scenarios) == 0:
            raise CaseLoadError("rl_validation K=1 produced 0 scenarios")
        return f"Loaded K=1 source from {source_path}, derived cost_regime"
    else:
        # Source is K=2, derive K=1
        result = load_cases(
            split="rl_validation",
            k=1,
            cost_regime_id="failure-light-no-waste",
            source_bank_path=str(source_path),
        )
        if len(result.scenario_ids) == 0:
            raise CaseLoadError("rl_validation K=1 derivation produced 0 scenarios")
        return result.provenance