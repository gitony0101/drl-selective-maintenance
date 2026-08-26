"""
Milestone 2 Selective Maintenance Environment Modules.

This package provides the full Gymnasium environment implementation:
- action_table: Deterministic direct-subset action enumeration
- config: Immutable environment configuration
- costs: Cost regime definitions and calculation
- errors: Custom environment exceptions
- scenario_bank: Scenario data structures and validation
- selective_maintenance_env: Main Gymnasium environment class
- state: Slot state representation

The full environment exposes:
- SelectiveMaintenanceEnv(gymnasium.Env)
- reset(seed, options) -> observation, info
- step(action_id) -> observation, reward, terminated, truncated, info
- get_action_mask() -> boolean array
- close()
"""

from .action_table import (
    ACTION_TABLE_N5_K1,
    ACTION_TABLE_N5_K2,
    ActionSubset,
    action_id_to_slots,
    build_action_table,
    get_action_table_config,
    slots_to_action_id,
    validate_action_table,
)
from .config import (
    DEFAULT_AGE_SCALE_CYCLES,
    DEFAULT_DELTA_CYCLES,
    DEFAULT_ENVIRONMENT_VERSION,
    DEFAULT_EPISODE_HORIZON,
    DEFAULT_FLEET_SIZE,
    DEFAULT_INFO_MODE,
    DEFAULT_MAINTENANCE_CAPACITY,
    DEFAULT_RUL_SCALE,
    DEFAULT_SEED,
    ALLOWED_SPLITS,
    INFO_MODES,
    EnvironmentConfig,
    get_default_config,
)
from .costs import (
    COST_REGIMES,
    DEFAULT_COST_REGIME_ID,
    FAILURE_HEAVY_NO_WASTE,
    FAILURE_HEAVY_WASTE_AWARE,
    FAILURE_LIGHT_NO_WASTE,
    FAILURE_LIGHT_WASTE_AWARE,
    CostRegime,
    calculate_total_cost,
    get_cost_regime,
    list_cost_regimes,
    validate_cost_regime,
)
from .errors import (
    ContractViolationError,
    InformationLeakageError,
    InvalidActionError,
    Milestone2EnvironmentError,
    MissingPredictionError,
    ScenarioValidationError,
    SplitViolationError,
)
from .scenario_bank import (
    Scenario,
    ScenarioBank,
    load_scenario_bank,
    save_scenario_bank,
    validate_full_scenario_bank,
    validate_scenario_against_config,
    validate_scenario_cycles_exist,
    validate_scenario_units_against_split,
)
from .selective_maintenance_env import SelectiveMaintenanceEnv
from .state import SlotState

__all__ = [
    # Action table
    "ACTION_TABLE_N5_K1",
    "ACTION_TABLE_N5_K2",
    "ActionSubset",
    "action_id_to_slots",
    "build_action_table",
    "get_action_table_config",
    "slots_to_action_id",
    "validate_action_table",
    # Config
    "DEFAULT_AGE_SCALE_CYCLES",
    "DEFAULT_DELTA_CYCLES",
    "DEFAULT_ENVIRONMENT_VERSION",
    "DEFAULT_EPISODE_HORIZON",
    "DEFAULT_FLEET_SIZE",
    "DEFAULT_INFO_MODE",
    "DEFAULT_MAINTENANCE_CAPACITY",
    "DEFAULT_RUL_SCALE",
    "DEFAULT_SEED",
    "ALLOWED_SPLITS",
    "INFO_MODES",
    "EnvironmentConfig",
    "get_default_config",
    # Costs
    "COST_REGIMES",
    "DEFAULT_COST_REGIME_ID",
    "FAILURE_HEAVY_NO_WASTE",
    "FAILURE_HEAVY_WASTE_AWARE",
    "FAILURE_LIGHT_NO_WASTE",
    "FAILURE_LIGHT_WASTE_AWARE",
    "CostRegime",
    "calculate_total_cost",
    "get_cost_regime",
    "list_cost_regimes",
    "validate_cost_regime",
    # Errors
    "ContractViolationError",
    "InformationLeakageError",
    "InvalidActionError",
    "Milestone2EnvironmentError",
    "MissingPredictionError",
    "ScenarioValidationError",
    "SplitViolationError",
    # Scenario bank
    "Scenario",
    "ScenarioBank",
    "load_scenario_bank",
    "save_scenario_bank",
    "validate_full_scenario_bank",
    "validate_scenario_against_config",
    "validate_scenario_cycles_exist",
    "validate_scenario_units_against_split",
    # Environment
    "SelectiveMaintenanceEnv",
    # State
    "SlotState",
]