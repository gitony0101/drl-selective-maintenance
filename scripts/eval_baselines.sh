#!/bin/bash
# Baseline policy evaluation script
# Usage: bash scripts/eval_baselines.sh

set -e

echo "========================================"
echo "Baseline Policy Evaluation"
echo "========================================"

# Switch to the project root directory
cd "$(dirname "$0")/.."

# Run the baseline evaluation
python -c "
import sys
sys.path.insert(0, '.')

from environment.cmapss_env import CMAPSSMaintenanceEnv
from eval.baseline_rules import get_baseline_policy, evaluate_baseline

# Create the environment
env = CMAPSSMaintenanceEnv(
    n_engines=10,
    max_maintenance_per_step=2,
    max_steps=100,
)

# Baseline policies
policies = {
    'random': get_baseline_policy('random', n_engines=10, K=2),
    'threshold_20': get_baseline_policy('threshold', n_engines=10, K=2, rul_threshold=20),
    'threshold_30': get_baseline_policy('threshold', n_engines=10, K=2, rul_threshold=30),
    'threshold_40': get_baseline_policy('threshold', n_engines=10, K=2, rul_threshold=40),
    'priority_rul': get_baseline_policy('priority_rul', n_engines=10, K=2),
    'health': get_baseline_policy('health', n_engines=10, K=2),
}

print('Evaluating baseline policies (100 episodes per policy)...')
print('=' * 70)

results = {}
for name, policy in policies.items():
    metrics = evaluate_baseline(policy, env, n_episodes=100)
    results[name] = metrics
    print(f'{name}:')
    print(f'  Mean reward: {metrics[\"reward_mean\"]:.2f} ± {metrics[\"reward_std\"]:.2f}')
    print(f'  Mean failures: {metrics[\"n_failures_mean\"]:.2f}')
    print(f'  Success rate: {metrics[\"success_rate\"] * 100:.1f}%')
    print()

print('=' * 70)
print('Evaluation complete!')
"

echo "========================================"