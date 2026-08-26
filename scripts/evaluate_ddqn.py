#!/usr/bin/env python3
"""
DDQN Evaluation CLI for Milestone 5.

Usage:
    python scripts/evaluate_ddqn.py --checkpoint checkpoints/checkpoint_best.pt --config configs/agents/ddqn_v1.json
    python scripts/evaluate_ddqn.py --help

Provenance Validation:
    - CLI --split
    - config validation_split
    - checkpoint validation_split
    - validation scenario-bank declared split
    - scenario-bank content hash

Rules:
    - Any source containing rl_test causes rejection
    - Any disagreement among sources causes rejection
    - Rejection occurs before prediction cache load and environment construction
    - Scenario bank identity is SHA256 of file contents, not path string
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add repository root to path for src. imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.ddqn_trainer import DDQNTrainer, TrainerConfig, set_all_seeds
from src.agents.ddqn import DDQNAgent, DDQNAgentConfig
from src.agents.ddqn.checkpoint import load_checkpoint, compute_scenario_bank_content_hash
from src.envs.config import get_default_config, ALLOWED_SPLITS
from src.envs.selective_maintenance_env import SelectiveMaintenanceEnv
from src.envs.scenario_bank import load_scenario_bank
from src.training.ddqn_config import validate_config_for_evaluation, load_and_validate_config


FORBIDDEN_SPLITS = frozenset({"rl_test"})


def _reject_rl_test_split(source: str, value: str) -> None:
    """Reject rl_test split with clear FORBIDDEN message and exit nonzero."""
    print(f"ERROR: {source}='{value}' is FORBIDDEN for evaluation.", file=sys.stderr)
    print(
        "Milestone 5 evaluation must not access rl_test during development.",
        file=sys.stderr,
    )
    print(
        "rl_test is sealed and must only be used for final evaluation.",
        file=sys.stderr,
    )
    sys.exit(1)


def _check_split_barrier(source: str, value: str) -> None:
    """Check a single split source against rl_test barrier."""
    if value in FORBIDDEN_SPLITS:
        _reject_rl_test_split(source, value)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate DDQN checkpoint (Milestone 5)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint file",
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configuration JSON file",
    )

    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Override evaluation split",
    )

    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "mps", "cuda"],
        help="Device override",
    )

    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    """Load configuration from JSON file."""
    if not config_path.exists():
        print(f"ERROR: Configuration file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_and_validate_split(
    cli_split: str | None,
    config_dict: dict,
    checkpoint_metadata: dict,
    scenario_bank: dict | None,
) -> tuple[str, dict]:
    """
    Resolve evaluation split from multiple sources and validate provenance.

    Sources (priority order):
    1. CLI --split
    2. config validation_split (nested under environment or flat)
    3. checkpoint validation_split
    4. Default: rl_validation

    Validation:
    - Any source containing rl_test causes rejection (training AND validation splits)
    - Any disagreement among sources causes rejection
    - Scenario bank declared split must match resolved split
    - Rejection occurs before environment construction

    Returns:
        Tuple of (resolved_split, provenance_record)

    Raises:
        SystemExit: On rl_test or provenance disagreement
    """
    # Collect all sources explicitly BEFORE resolution
    # Handle both nested (environment.*) and flat config structures
    cli_split_val = cli_split

    # Config training_split: check flat field first (from TrainerConfig), then nested
    config_train_split = config_dict.get("training_split") or config_dict.get("environment", {}).get("split")

    # Config validation_split: check flat field first (from TrainerConfig), then nested
    config_val_split = config_dict.get("validation_split") or config_dict.get("environment", {}).get("validation_split")

    checkpoint_train_split = checkpoint_metadata.get("training_split")
    checkpoint_val_split = checkpoint_metadata.get("validation_split")
    scenario_bank_split = scenario_bank.get("split") if scenario_bank else None

    # Initialize provenance record
    provenance = {
        "cli_split": cli_split_val,
        "config_training_split": config_train_split,
        "config_validation_split": config_val_split,
        "checkpoint_training_split": checkpoint_train_split,
        "checkpoint_validation_split": checkpoint_val_split,
        "scenario_bank_declared_split": scenario_bank_split,
    }

    # Fail closed if checkpoint schema-v3 provenance is missing.
    if checkpoint_val_split is None:
        print(
            "ERROR: Checkpoint missing validation_split provenance (schema v3 required).",
            file=sys.stderr,
        )
        print(
            "Checkpoint must include validation_split from schema v3.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check ALL sources against rl_test barrier (both training and validation splits)
    if cli_split_val is not None:
        _check_split_barrier("CLI --split", cli_split_val)
    if config_train_split is not None:
        _check_split_barrier("config training_split", config_train_split)
    if config_val_split is not None:
        _check_split_barrier("config validation_split", config_val_split)
    if checkpoint_train_split is not None:
        _check_split_barrier("checkpoint training_split", checkpoint_train_split)
    if checkpoint_val_split is not None:
        _check_split_barrier("checkpoint validation_split", checkpoint_val_split)
    if scenario_bank_split is not None:
        _check_split_barrier("scenario bank declared split", scenario_bank_split)

    # Resolve: CLI takes priority, then config validation_split, then checkpoint validation_split, then default
    if cli_split_val is not None:
        resolved_split = cli_split_val
    elif config_val_split is not None:
        resolved_split = config_val_split
    elif checkpoint_val_split is not None:
        resolved_split = checkpoint_val_split
    else:
        resolved_split = "rl_validation"

    # Final barrier check on resolved split
    _check_split_barrier("resolved evaluation split", resolved_split)

    # Validate agreement among all present validation sources
    # Only validation splits should agree (training splits are separate provenance info)
    present_val_splits = [s for s in [
        cli_split_val,
        config_val_split,
        checkpoint_val_split,
        scenario_bank_split,
    ] if s is not None]

    if len(set(present_val_splits)) > 1:
        print(
            f"ERROR: Split provenance disagreement detected:",
            file=sys.stderr
        )
        for key, value in provenance.items():
            if value is not None:
                print(f"  {key}: {value}", file=sys.stderr)
        print(
            "All split sources must agree. Check CLI, config, checkpoint, and scenario bank.",
            file=sys.stderr
        )
        sys.exit(1)

    # Validate resolved split is rl_validation (M5 requirement)
    if resolved_split != "rl_validation":
        print(
            f"ERROR: Evaluation split must be 'rl_validation', got '{resolved_split}'.",
            file=sys.stderr,
        )
        print(
            "Milestone 5 evaluation must use rl_validation split.",
            file=sys.stderr,
        )
        sys.exit(1)

    provenance["resolved_split"] = resolved_split
    return resolved_split, provenance


def validate_scenario_bank_provenance(
    scenario_bank_path: str,
    checkpoint_metadata: dict,
    config_dict: dict,
) -> tuple[dict, int]:
    """
    Validate scenario bank provenance against checkpoint and config.

    Validates:
    - File exists
    - JSON parses
    - Declared split matches expected split
    - Maintenance capacity matches K
    - File content hash matches checkpoint provenance

    Returns:
        Tuple of (scenario_bank_dict, k_value)

    Raises:
        SystemExit: On validation failure
    """
    scenario_bank_path = Path(scenario_bank_path)

    # Check file exists
    if not scenario_bank_path.exists():
        print(
            f"ERROR: Scenario bank not found: {scenario_bank_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Parse JSON
    try:
        with scenario_bank_path.open("r", encoding="utf-8") as f:
            scenario_bank = json.load(f)
    except json.JSONDecodeError as e:
        print(
            f"ERROR: Scenario bank JSON parse error: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Get declared split and K
    declared_split = scenario_bank.get("split")
    declared_k = scenario_bank.get("maintenance_capacity")

    # Get expected values from checkpoint
    checkpoint_k = checkpoint_metadata.get("maintenance_capacity")
    checkpoint_bank_hash = checkpoint_metadata.get("validation_scenario_bank_identity")

    # Get expected K from config
    config_k = config_dict.get("maintenance_capacity")

    # Validate K consistency
    if checkpoint_k is None:
        print(
            "ERROR: Checkpoint missing maintenance_capacity (schema v3 required).",
            file=sys.stderr,
        )
        sys.exit(1)

    if declared_k is not None:
        if declared_k != checkpoint_k:
            print(
                f"ERROR: Scenario bank K mismatch: bank declares K={declared_k}, "
                f"checkpoint has K={checkpoint_k}.",
                file=sys.stderr,
            )
            sys.exit(1)

    if config_k is not None and config_k != checkpoint_k:
        print(
            f"ERROR: Config/checkpoint K mismatch: config has K={config_k}, "
            f"checkpoint has K={checkpoint_k}.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate content hash
    actual_hash = compute_scenario_bank_content_hash(scenario_bank_path)
    if checkpoint_bank_hash is not None:
        if actual_hash != checkpoint_bank_hash:
            print(
                f"ERROR: Scenario bank content hash mismatch:",
                file=sys.stderr,
            )
            print(f"  Checkpoint expects: {checkpoint_bank_hash}", file=sys.stderr)
            print(f"  Actual hash: {actual_hash}", file=sys.stderr)
            print(
                "Scenario bank file contents do not match checkpoint provenance.",
                file=sys.stderr,
            )
            print(
                "Tamper detection: file may have been modified since checkpoint.",
                file=sys.stderr,
            )
            sys.exit(1)

    return scenario_bank, checkpoint_k


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Validate checkpoint exists
    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    # Parse configuration through the standard config parser.
    # This ensures evaluation uses the same config parsing as training
    try:
        with args.config.open("r", encoding="utf-8") as f:
            raw_config = json.load(f)
        parsed = validate_config_for_evaluation(raw_config)
    except ValueError as e:
        print(f"ERROR: Config validation failed: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Config JSON parse error: {e}", file=sys.stderr)
        sys.exit(1)

    trainer_config = parsed.trainer_config
    config_dict = parsed.raw_config

    # BARRIER: Check ALL split sources against rl_test before loading checkpoint
    # Use trainer_config which has properly parsed fields
    if args.split is not None:
        _check_split_barrier("CLI --split", args.split)

    # Config splits (both training and validation) - from parsed TrainerConfig
    if trainer_config.split is not None:
        _check_split_barrier("config training_split", trainer_config.split)
    if trainer_config.validation_split is not None:
        _check_split_barrier("config validation_split", trainer_config.validation_split)

    # Load checkpoint metadata for provenance validation
    try:
        checkpoint_data, issues = load_checkpoint(args.checkpoint, agent=None)
    except ValueError as e:
        print(f"ERROR: Checkpoint load failed: {e}", file=sys.stderr)
        sys.exit(1)

    if issues.get("incompatibilities"):
        print("ERROR: Checkpoint incompatibilities:", file=sys.stderr)
        for issue in issues["incompatibilities"]:
            print(f"  - {issue}", file=sys.stderr)
        sys.exit(1)

    metadata = checkpoint_data.metadata
    checkpoint_metadata = metadata.to_dict()

    # Check checkpoint splits (both training and validation) barrier
    checkpoint_train_split = checkpoint_metadata.get("training_split")
    checkpoint_val_split = checkpoint_metadata.get("validation_split")
    if checkpoint_train_split is not None:
        _check_split_barrier("checkpoint training_split", checkpoint_train_split)
    if checkpoint_val_split is not None:
        _check_split_barrier("checkpoint validation_split", checkpoint_val_split)

    # Determine scenario bank path from TrainerConfig.
    # Schema-v3 evaluation must fail closed if validation_scenario_bank_path is missing
    config_val_bank_path = trainer_config.validation_scenario_bank_path
    if config_val_bank_path is None:
        # Fail closed: schema-v3 requires explicit validation_scenario_bank_path
        print(
            "ERROR: Config missing validation_scenario_bank_path (schema v3 required).",
            file=sys.stderr,
        )
        print(
            "Evaluation config must explicitly specify validation_scenario_bank_path.",
            file=sys.stderr,
        )
        print(
            "Fallback to K-based default paths is not allowed for schema v3.",
            file=sys.stderr,
        )
        sys.exit(1)

    scenario_bank_path = config_val_bank_path

    # Load and validate scenario bank
    scenario_bank, k_value = validate_scenario_bank_provenance(
        scenario_bank_path,
        checkpoint_metadata,
        config_dict,
    )

    # Scenario bank declared split is checked inside resolve_and_validate_split

    # Resolve and validate split with all sources including scenario bank
    eval_split, provenance = resolve_and_validate_split(
        cli_split=args.split,
        config_dict=config_dict,
        checkpoint_metadata=checkpoint_metadata,
        scenario_bank=scenario_bank,
    )

    # Build environment config from checkpoint metadata and production TrainerConfig
    env_config = get_default_config(
        split=eval_split,
        cost_regime_id=metadata.cost_regime_id,
        maintenance_capacity=metadata.maintenance_capacity,
        scenario_bank_path=scenario_bank_path,
        prediction_cache_path=trainer_config.prediction_cache_path,
    )

    # Create environment
    env = SelectiveMaintenanceEnv(config=env_config)

    # Create agent using production config values
    agent_config = DDQNAgentConfig(
        observation_dim=10,
        num_actions=metadata.action_count,
        hidden_dim=trainer_config.hidden_dim,
        num_hidden_layers=trainer_config.num_hidden_layers,
    )
    agent = DDQNAgent(config=agent_config)

    # Load checkpoint into agent
    load_checkpoint(args.checkpoint, agent=agent)

    # Evaluate
    print(f"Evaluating checkpoint: {args.checkpoint}")
    print(f"  K: {metadata.maintenance_capacity}")
    print(f"  Actions: {metadata.action_count}")
    print(f"  Cost regime: {metadata.cost_regime_id}")
    print(f"  Global step: {metadata.global_step}")
    print(f"  Split: {eval_split}")
    print(f"  Scenario bank: {scenario_bank_path}")
    print(f"  Scenario bank hash: {compute_scenario_bank_content_hash(Path(scenario_bank_path))}")
    print(f"  Split provenance:")
    for key, value in provenance.items():
        if value is not None:
            print(f"    {key}: {value}")

    episode_returns = []
    total_costs = []
    failure_counts = []
    pm_counts = []

    for ep in range(args.num_episodes):
        obs, _ = env.reset()
        episode_return = 0.0
        total_cost = 0.0
        failure_count = 0
        pm_count = 0

        for step in range(env.horizon):
            action = agent.evaluate_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_return += reward
            total_cost += info["total_cost"]
            failure_count += info["num_failures"]
            pm_count += info["num_preventive"]

            if truncated:
                break

        episode_returns.append(episode_return)
        total_costs.append(total_cost)
        failure_counts.append(failure_count)
        pm_counts.append(pm_count)

    # Print results
    import numpy as np

    print(f"\nResults ({args.num_episodes} episodes):")
    print(f"  Mean total cost: {np.mean(total_costs):.2f} (+/- {np.std(total_costs):.2f})")
    print(f"  Mean episode return: {np.mean(episode_returns):.2f}")
    print(f"  Total failures: {sum(failure_counts)}")
    print(f"  Total PM actions: {sum(pm_counts)}")


if __name__ == "__main__":
    main()