# M6 Preregistered Statistical Analysis

All results are per K × cost-regime cell. M5 denotes five independently trained policies, never an ensemble. No raw costs are pooled across regimes.

The preregistered analysis uses B=10,000 two-sided percentile-bootstrap resamples and base seed 652106. For cell index i (one-based, sorted by K and regime), H2−M4 begins at seed 652106+10i, the hierarchical M5−H2 bootstrap uses 652106+10i+1, and the five individual-policy paired bootstraps use 652106+10i+2 through +6. The M5 effect size is Cohen's d over 20 scenario-averaged differences after averaging the five policy differences within each scenario; hierarchical resampling, not that d, is the inferential treatment of seed and scenario variation.

Eight cells are characterized with per-cell 95% intervals. The preregistered plan applies no Bonferroni or Holm correction and tests no project-level global null.

## Primary total-cost comparisons

| K | Regime | H2−M4 mean [95% CI] | d | M5−H2 mean [hier. 95% CI] | d |
|---:|---|---:|---:|---:|---:|
| 1 | failure-heavy-no-waste | 0.6 [0.3, 0.9] | 0.8816 | 6.3 [4.21, 8.67] | 1.733 |
| 1 | failure-heavy-waste-aware | 0.8364 [0.567497, 1.1475] | 1.198 | 4.97656 [2.65627, 7.55118] | 1.014 |
| 1 | failure-light-no-waste | 0.45 [-0.1, 0.9] | 0.3927 | 3.3 [2.28, 4.39] | 1.766 |
| 1 | failure-light-waste-aware | 0.6402 [0.1457, 1.00052] | 0.6377 | 2.56526 [1.60614, 3.65229] | 1.298 |
| 2 | failure-heavy-no-waste | 0.35 [0.05, 0.7] | 0.4697 | 4.44 [2.92, 6.15] | 1.605 |
| 2 | failure-heavy-waste-aware | 0.6094 [0.293598, 0.970003] | 0.7704 | 2.79696 [1.33655, 4.44528] | 0.9631 |
| 2 | failure-light-no-waste | 0.55 [0.25, 0.9] | 0.7245 | 2.54 [1.77, 3.35] | 1.721 |
| 2 | failure-light-waste-aware | 0.577 [0.2709, 0.9403] | 0.7369 | 1.75016 [0.948481, 2.6005] | 1.064 |

## M5 individual-policy paired summaries

### K=1, failure-heavy-no-waste

| Training seed | Mean M5−H2 | 95% paired CI | d | Directions (−/0/+) |
|---:|---:|---:|---:|---:|
| 6521 | 1.35 | [0.4, 2.6] | 0.5231 | 0/11/9 |
| 6522 | 3.75 | [1.75, 6.1] | 0.7142 | 1/5/14 |
| 6523 | 1 | [0.35, 1.95] | 0.5 | 0/10/10 |
| 6524 | 14.05 | [10.3, 17.9] | 1.571 | 2/0/18 |
| 6525 | 11.35 | [6.55, 16.6] | 0.9588 | 0/5/15 |

### K=1, failure-heavy-waste-aware

| Training seed | Mean M5−H2 | 95% paired CI | d | Directions (−/0/+) |
|---:|---:|---:|---:|---:|
| 6521 | 6.7706 | [3.13732, 10.8376] | 0.7568 | 8/0/12 |
| 6522 | 5.7555 | [2.43318, 9.53491] | 0.6787 | 9/0/11 |
| 6523 | 1.4157 | [0.182188, 2.8514] | 0.4485 | 3/0/17 |
| 6524 | 7.6129 | [3.36213, 12.5029] | 0.6988 | 6/2/12 |
| 6525 | 3.3281 | [0.960938, 6.06441] | 0.544 | 12/0/8 |

### K=1, failure-light-no-waste

| Training seed | Mean M5−H2 | 95% paired CI | d | Directions (−/0/+) |
|---:|---:|---:|---:|---:|
| 6521 | 1.2 | [0.65, 1.8] | 0.8576 | 0/9/11 |
| 6522 | 2.35 | [1.35, 3.45] | 0.9253 | 1/5/14 |
| 6523 | 1.1 | [0.6, 1.65] | 0.8783 | 0/9/11 |
| 6524 | 6.4 | [4.7, 8.2] | 1.572 | 1/1/18 |
| 6525 | 5.45 | [3.2, 7.9] | 0.9925 | 0/5/15 |

### K=1, failure-light-waste-aware

| Training seed | Mean M5−H2 | 95% paired CI | d | Directions (−/0/+) |
|---:|---:|---:|---:|---:|
| 6521 | 3.1093 | [1.47418, 4.89044] | 0.7692 | 3/0/17 |
| 6522 | 2.8442 | [1.38533, 4.4642] | 0.7794 | 2/0/18 |
| 6523 | 1.2544 | [0.746395, 1.84082] | 0.9554 | 0/0/20 |
| 6524 | 3.7016 | [1.86629, 5.87875] | 0.7899 | 0/0/20 |
| 6525 | 1.9168 | [0.88984, 3.0623] | 0.7474 | 2/0/18 |

### K=2, failure-heavy-no-waste

| Training seed | Mean M5−H2 | 95% paired CI | d | Directions (−/0/+) |
|---:|---:|---:|---:|---:|
| 6521 | 4.05 | [1.55, 6.8] | 0.6477 | 2/8/10 |
| 6522 | 3.2 | [1.39875, 5.15] | 0.724 | 2/5/13 |
| 6523 | 1.5 | [1.1, 1.95] | 1.586 | 0/2/18 |
| 6524 | 7.5 | [4.4, 11.05] | 0.967 | 0/4/16 |
| 6525 | 5.95 | [2.9, 9.35] | 0.7881 | 0/8/12 |

### K=2, failure-heavy-waste-aware

| Training seed | Mean M5−H2 | 95% paired CI | d | Directions (−/0/+) |
|---:|---:|---:|---:|---:|
| 6521 | 3.1527 | [1.3224, 5.11633] | 0.6972 | 2/0/18 |
| 6522 | 1.8293 | [0.294978, 3.66192] | 0.4661 | 7/0/13 |
| 6523 | 1.1166 | [0.01409, 2.62872] | 0.3602 | 7/0/13 |
| 6524 | 7.1404 | [3.32069, 11.1919] | 0.7698 | 4/0/16 |
| 6525 | 0.7458 | [-0.2588, 2.04613] | 0.2715 | 8/2/10 |

### K=2, failure-light-no-waste

| Training seed | Mean M5−H2 | 95% paired CI | d | Directions (−/0/+) |
|---:|---:|---:|---:|---:|
| 6521 | 2.3 | [1, 3.7] | 0.7347 | 2/7/11 |
| 6522 | 1.95 | [1, 2.95] | 0.8633 | 1/5/14 |
| 6523 | 1.75 | [1.25, 2.25] | 1.503 | 0/3/17 |
| 6524 | 3.75 | [2.45, 5.15] | 1.174 | 0/4/16 |
| 6525 | 2.95 | [1.6, 4.45] | 0.8853 | 1/4/15 |

### K=2, failure-light-waste-aware

| Training seed | Mean M5−H2 | 95% paired CI | d | Directions (−/0/+) |
|---:|---:|---:|---:|---:|
| 6521 | 2.1059 | [1.01019, 3.23605] | 0.8078 | 2/0/18 |
| 6522 | 1.2825 | [0.55226, 2.07361] | 0.7261 | 3/0/17 |
| 6523 | 1.0698 | [0.407393, 1.90441] | 0.6125 | 0/0/20 |
| 6524 | 3.5936 | [1.7731, 5.5903] | 0.8022 | 1/0/19 |
| 6525 | 0.699 | [0.189198, 1.33155] | 0.5165 | 2/0/18 |

## Variation and direction counts

| K | Regime | M5 seed means | Seed SD | Scenario SD | H2−M4 dirs (−/0/+) | M5 scenario dirs (−/0/+) | M5 seed dirs (−/0/+) |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | failure-heavy-no-waste | 6521:1.35, 6522:3.75, 6523:1, 6524:14.05, 6525:11.35 | 6.014 | 3.636 | 0/10/10 | 0/0/20 | 0/0/5 |
| 1 | failure-heavy-waste-aware | 6521:6.771, 6522:5.755, 6523:1.416, 6524:7.613, 6525:3.328 | 2.558 | 4.91 | 0/0/20 | 4/0/16 | 0/0/5 |
| 1 | failure-light-no-waste | 6521:1.2, 6522:2.35, 6523:1.1, 6524:6.4, 6525:5.45 | 2.469 | 1.869 | 2/8/10 | 0/0/20 | 0/0/5 |
| 1 | failure-light-waste-aware | 6521:3.109, 6522:2.844, 6523:1.254, 6524:3.702, 6525:1.917 | 0.9752 | 1.977 | 1/0/19 | 1/0/19 | 0/0/5 |
| 2 | failure-heavy-no-waste | 6521:4.05, 6522:3.2, 6523:1.5, 6524:7.5, 6525:5.95 | 2.344 | 2.766 | 1/13/6 | 0/0/20 | 0/0/5 |
| 2 | failure-heavy-waste-aware | 6521:3.153, 6522:1.829, 6523:1.117, 6524:7.14, 6525:0.7458 | 2.596 | 2.904 | 1/0/19 | 2/0/18 | 0/0/5 |
| 2 | failure-light-no-waste | 6521:2.3, 6522:1.95, 6523:1.75, 6524:3.75, 6525:2.95 | 0.8158 | 1.476 | 0/11/9 | 0/0/20 | 0/0/5 |
| 2 | failure-light-waste-aware | 6521:2.106, 6522:1.283, 6523:1.07, 6524:3.594, 6525:0.699 | 1.152 | 1.645 | 0/0/20 | 3/0/17 | 0/0/5 |

## Cost components and behavior

Values are mean differences in the displayed comparison direction.

| K | Regime | Comparison | Preventive cost | Failure cost | Wasted-life cost | Preventive count | Failure count | Empty actions | Capacity-saturated steps |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | failure-heavy-no-waste | H2−M4 | 0.6 | 0 | 0 | 0.6 | 0 | -0.6 | 0.6 |
| 1 | failure-heavy-no-waste | M5−H2 | -0.4 | 6.7 | 0 | -0.4 | 0.67 | 0.4 | -0.4 |
| 1 | failure-heavy-waste-aware | H2−M4 | 0.65 | 0 | 0.1864 | 0.65 | 0 | -0.65 | 0.65 |
| 1 | failure-heavy-waste-aware | M5−H2 | -0.96 | 6 | -0.06344 | -0.96 | 0.6 | 0.96 | -0.96 |
| 1 | failure-light-no-waste | H2−M4 | 0.7 | -0.25 | 0 | 0.7 | -0.05 | -0.7 | 0.7 |
| 1 | failure-light-no-waste | M5−H2 | -0.05 | 3.35 | 0 | -0.05 | 0.67 | 0.05 | -0.05 |
| 1 | failure-light-waste-aware | H2−M4 | 0.75 | -0.25 | 0.1402 | 0.75 | -0.05 | -0.75 | 0.75 |
| 1 | failure-light-waste-aware | M5−H2 | -0.51 | 3 | 0.07526 | -0.51 | 0.6 | 0.51 | -0.51 |
| 2 | failure-heavy-no-waste | H2−M4 | 0.35 | 0 | 0 | 0.35 | 0 | -0.45 | -0.1 |
| 2 | failure-heavy-no-waste | M5−H2 | 0.14 | 4.3 | 0 | 0.14 | 0.43 | 0.66 | 0.8 |
| 2 | failure-heavy-waste-aware | H2−M4 | 0.45 | 0 | 0.1594 | 0.45 | 0 | -0.5 | -0.05 |
| 2 | failure-heavy-waste-aware | M5−H2 | -0.22 | 3 | 0.01696 | -0.22 | 0.3 | 1.09 | 0.87 |
| 2 | failure-light-no-waste | H2−M4 | 0.55 | 0 | 0 | 0.55 | 0 | -0.35 | 0.2 |
| 2 | failure-light-no-waste | M5−H2 | 0.39 | 2.15 | 0 | 0.39 | 0.43 | 0.21 | 0.6 |
| 2 | failure-light-waste-aware | H2−M4 | 0.45 | 0 | 0.127 | 0.45 | 0 | 0 | 0.45 |
| 2 | failure-light-waste-aware | M5−H2 | 0.13 | 1.5 | 0.1202 | 0.13 | 0.3 | 0.29 | 0.42 |

For K=1, failure-light-no-waste, the H2−M4 interval covers zero (mean 0.45, 95% CI [-0.10, 0.90], small d=0.393).

> No clear difference was detected under the current validation sample.

All per-scenario paired differences, component direction counts, and behavior direction counts are retained in M6_STATISTICAL_RESULTS.json.
