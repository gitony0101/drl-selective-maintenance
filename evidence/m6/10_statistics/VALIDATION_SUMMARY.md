# M6 Validation Summary: DDQN vs H=2 Planner

## Comparison

The H=2 receding-horizon planner and the Point-DDQN policy were evaluated on the common `rl_validation` split (20 scenarios per cell) across eight capacity–cost regime cells. The primary metric is mean episodic total cost.

**Key finding:** In all eight cells, DDQN incurred higher mean total cost than H=2, with hierarchical bootstrap 95% confidence intervals excluding zero. The excess is localized primarily to realized failures.

## Per-Cell Results

| K | Cost Regime | DDQN Mean | H=2 Mean | Δ (DDQN−H=2) | 95% CI |
|---|-------------|-----------|----------|---------------|--------|
| 1 | failure-heavy / no-waste | 19.58 | 11.65 | +7.93 | [5.40, 10.80] |
| 1 | failure-heavy / waste-aware | 18.19 | 12.13 | +6.05 | [4.59, 7.61] |
| 1 | failure-light / no-waste | 15.68 | 11.55 | +4.13 | [2.95, 5.47] |
| 1 | failure-light / waste-aware | 15.04 | 11.96 | +3.08 | [2.33, 3.84] |
| 2 | failure-heavy / no-waste | 19.54 | 12.75 | +6.79 | [4.61, 9.01] |
| 2 | failure-heavy / waste-aware | 17.91 | 13.28 | +4.63 | [2.90, 6.59] |
| 2 | failure-light / no-waste | 15.94 | 12.50 | +3.44 | [2.46, 4.48] |
| 2 | failure-light / waste-aware | 15.31 | 12.96 | +2.35 | [1.57, 3.20] |

## Component Decomposition

The DDQN-H=2 cost gap is driven primarily by increased failure counts. In all eight cells, DDQN incurred 0.30–0.67 additional failures per episode relative to H=2. Preventive maintenance counts and wasted-life costs show smaller differences.

## Limitations

- Validation-only sample (20 scenarios per cell); no held-out test evaluation for this comparison.
- Per-cell 95% intervals are not familywise-adjusted.
- Results are specific to this benchmark and evaluation protocol; do not generalize to other domains.
- The H=2 planner encodes analytic transition/failure structure, so it is a structurally privileged comparator.

## Source of Record

Full per-cell statistics, bootstrap outputs, and per-scenario paired differences are available in `M6_STATISTICAL_RESULTS.json` and `M6_PER_CELL_RESULTS.csv`.