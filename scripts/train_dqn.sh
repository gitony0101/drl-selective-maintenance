#!/bin/bash
# DQN training launcher script
# Usage: bash scripts/train_dqn.sh

set -e

# Default parameters
N_ENGINES=10
K=2
NUM_EPISODES=500
WANDB_NAME="test-run"
SEED=42

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --n_engines)
            N_ENGINES="$2"
            shift 2
            ;;
        --K)
            K="$2"
            shift 2
            ;;
        --num_episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        # --wandb_name removed (no longer used)
            WANDB_NAME="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "DRL Selective Maintenance Training"
echo "========================================"
echo "  Number of engines: $N_ENGINES"
echo "  Capacity constraint K: $K"
echo "  Episodes: $NUM_EPISODES"
echo "  Run name: $WANDB_NAME"
echo "  Random seed: $SEED"
echo "========================================"

# Switch to the project root directory
cd "$(dirname "$0")/.."

# Run training
python trainer/train_dqn.py \
    --n_engines "$N_ENGINES" \
    --K "$K" \
    --num_episodes "$NUM_EPISODES" \
    # --wandb_name removed
    --seed "$SEED"

echo "========================================"
echo "Training complete!"
echo "========================================"