# Capacity-Constrained Selective Maintenance with Deep Reinforcement Learning

Comparing model-free value learning, structured planning, and learned-model control in a synthetic continuing fleet-maintenance benchmark built from NASA C-MAPSS FD001 degradation trajectories.

## Overview

A fleet of degrading assets must keep operating over a long horizon. At each maintenance window only `K` assets can be preventively replaced, so the decision maker faces a recurring trade-off: replacing an asset early wastes its remaining useful life and consumes scarce maintenance capacity, while deferring risks an expensive failure that also occupies a replacement slot. Which assets should be maintained now?

This is fundamentally a sequential decision problem. Today's replacement choice changes each asset's future degradation state and shifts future failures into different windows under a shared capacity constraint, so greedy per-window reasoning is not obviously sufficient. Reinforcement learning is a natural fit: a value-based policy can, in principle, account for how present actions shape future failure risk and future maintenance opportunities.

This repository implements that study end to end: data preparation from NASA C-MAPSS FD001 run-to-failure trajectories, a continuing fleet simulator with capacity-constrained preventive replacement, rule-based and optimization baselines, a Double DQN (DDQN) policy, a privileged short-horizon planner, learned-model MPC diagnostics, targeted ablations, and a frozen statistical evaluation with a one-time held-out replication.

## Research Question

> When rare failure events dominate long-term cost, can model-free value learning match short-horizon planning for capacity-constrained selective maintenance — and what do targeted ablations reveal about why or why not?

The question is deliberately benchmark-specific. The primary result is negative: **the tested Point-DDQN configuration incurred consistently higher total cost than a privileged analytic-model H=2 planner, on both the validation matrix and a sealed held-out split**, with the excess localized around realized failures.

## Problem Formulation

The environment is a synthetic continuing fleet of `N = 5` slots constructed from 100 NASA C-MAPSS FD001 run-to-failure trajectories, used as an empirical degradation library rather than a physical simulator.

- **Hidden simulator state** (per slot): trajectory ID, cycle index, age since replacement, true RUL, and last replacement type. True RUL is used only for hidden accounting — failure determination, unused-life cost, and oracle diagnostics — and is never exposed to any implementable policy.
- **Observation** (what policies see): a 10-dimensional vector of clipped normalized age and cached predicted RUL for each of the five slots. Policies act on predictions, not true remaining life.
- **Action**: a feasible subset of fleet slots selected for preventive replacement, `a_t ⊆ {0,…,4}` with `|a_t| ≤ K`. The full action table contains 16 subsets for `K = 2` and 6 for `K = 1`; the empty action is valid.
- **Capacity constraint**: at most `K` slots may be maintained per window (`K ∈ {1, 2}`).
- **Transition** per decision window (Δ = 5 cycles): selected slots are preventively replaced; unselected slots advance; any slot whose true RUL crosses zero fails, is charged a failure cost once, and is correctively replaced at the window's end. Failures never terminate the episode — this is a continuing fleet.
- **Cost / reward**: `r_t = −C_t` where `C_t = c_pm·|S_t| + c_f·N_fail,t + c_u·Σ_{i∈S_t} trueRUL_i,t/RUL_max`, with `c_f/c_pm ∈ {5, 10}` and `c_u/c_pm ∈ {0, 0.25}`, giving four cost regimes spanning failure-light/heavy × no-waste/waste-aware.
- **Episodes**: fixed horizons of decision windows (100 windows in the primary protocol). Evaluation is always paired on pre-generated, serialized scenario banks shared across all policies.

## Methods

| Method | Type | Notes |
|---|---|---|
| Rule-based baselines | Reference | Corrective-only, random feasible, age-threshold, predicted-RUL-threshold, greedy predicted-RUL |
| Exact myopic optimizer | Structured | Enumerates all feasible actions; minimizes estimated current-window cost using a predicted-risk surrogate. No true-RUL access |
| H=2 receding-horizon planner | Privileged comparator | Probability-weighted two-step planning with *encoded analytic transition/failure structure*; same observation as DDQN, but structurally privileged |
| Point-DDQN | Primary learner | Double DQN, MLP 10→128→128→\|A\|, Huber loss, replay 100k, γ=0.95, ε-greedy 1.0→0.05 |
| n-step DDQN variants | Ablation | Return horizon n∈{1,3} crossed with standard vs planner-seeded replay |
| Learned-model MPC | Diagnostic | Ensemble dynamics model + exhaustive two-step MPC under different training-data compositions |

The H=2 planner's privilege is explicit: it is given the benchmark's analytic transition and failure structure rather than learning it from experience. It does not see true RUL, identities, or realized future outcomes — it enumerates possible failure branches probabilistically. Because model knowledge and planning procedure differ simultaneously from DDQN's setting, this comparison does not isolate the value of planning itself; it asks whether the tested value learner matches a structurally advantaged short-horizon controller.

## Experimental Design

- **Dataset**: NASA C-MAPSS FD001 (single operating condition, single fault mode), 100 run-to-failure engine trajectories.
- **Engine-level partitioning** (frozen and disjoint): `predictor_train` = 60, `predictor_validation` = 15, `rl_validation` = 10, sealed `rl_test` = 15 engines. Sliding windows are never split randomly; scalers are fitted on predictor-training statistics only.
- **Leakage prevention**: true RUL never enters observations; test engines never contribute to predictor training; development and ablation decisions use `rl_validation` exclusively.
- **Held-out discipline**: after models, checkpoints, scenario banks, and primary statistics were frozen, the primary comparison alone was evaluated exactly once on the sealed `rl_test` split, with no test-informed tuning or selection. The n-step, risk-feature, and learned-MPC studies remain validation-only by design.
- **Statistical protocol**: five training seeds per cell ({6521–6525}), paired evaluation across cells, preregistered hierarchical (scenario-by-seed) bootstrap with 10,000 resamples and cellwise prespecified 95% intervals (no familywise-adjustment claims).

## Key Findings

**Primary held-out result.** In the one-time held-out evaluation, Point-DDQN minus H=2 mean episodic cost was positive in **all eight capacity–cost cells**, and every cellwise hierarchical bootstrap 95% CI excluded zero:

| K | Regime | DDQN | H=2 | Δ (DDQN−H=2) | 95% CI |
|---|---|---|---|---|---|
| 1 | failure-heavy / no-waste | 19.58 | 11.65 | +7.93 | [5.40, 10.80] |
| 1 | failure-heavy / waste-aware | 18.19 | 12.13 | +6.05 | [4.59, 7.61] |
| 1 | failure-light / no-waste | 15.68 | 11.55 | +4.13 | [2.95, 5.47] |
| 1 | failure-light / waste-aware | 15.04 | 11.96 | +3.08 | [2.33, 3.84] |
| 2 | failure-heavy / no-waste | 19.54 | 12.75 | +6.79 | [4.61, 9.01] |
| 2 | failure-heavy / waste-aware | 17.91 | 13.28 | +4.63 | [2.90, 6.59] |
| 2 | failure-light / no-waste | 15.94 | 12.50 | +3.44 | [2.46, 4.48] |
| 2 | failure-light / waste-aware | 15.31 | 12.96 | +2.35 | [1.57, 3.20] |

The same ordering held in all eight validation cells before the held-out replication, and the ordering reproduced on the sealed test split, reducing concern that the observed gap was specific to the validation scenarios. Cost decomposition localizes the excess primarily in **realized failures**: Point-DDQN incurred roughly 0.30–0.67 additional failures per episode than H=2 on validation.

*(Source of record: `evidence/heldout/RL_TEST_FINAL_RESULTS.json`, `evidence/heldout/final_test_evaluation/`.)*

**Validation-only ablations and diagnostics** (not held out):

- **n-step returns did not close the gap.** Switching from one-step to the tested n=3 return *increased* validation cost (+3.68 under standard replay, 95% CI [1.81, 5.55]; +3.00 under seeded replay, [0.56, 5.44]).
- **Planner-seeded replay was inconclusive**: −0.76 (95% CI [−1.55, 0.03], interval includes zero).

- **Learned MPC: prediction accuracy ≠ control quality.** With a failure-free training mixture, the ensemble achieved low next-observation RMSE (~0.02 vs ~0.095 persistence) yet selected the no-maintenance action at *every* decision, accumulating 54 failures across five evaluation episodes (mean cost 54.0).
- **Failure coverage partially recovered planning.** Enriching the mixture with failure transitions reduced mean cost to 34.6 (E5−E4 = −19.40, 95% CI [−28.28, −10.52]) and restored preventive behavior — but this intervention changes visitation, behavior-policy composition, and reward support simultaneously, so it does not isolate failure coverage as the sole cause.

No tested structured variant improved on the Point-DDQN policy either; the n-step numbers are cost increases relative to Point-DDQN.

## Why This Project Matters

The central mechanism-level insight is that **prediction accuracy alone is insufficient for control**. A learned dynamics model predicted common transitions accurately while missing exactly the rare, high-cost events that dominate closed-loop performance; similarly, a value learner trained on predominantly non-failure experience failed to internalize rare delayed failure costs. This connects three threads relevant to safety-critical sequential decision making:

- distribution coverage and data support in model-based RL (learned planners inherit the blind spots of their training mixtures);
- credit assignment through off-policy bootstrapping when rewards are sparse-but-large and delayed;
- the gap between upstream predictive metrics and downstream decision quality.

These observations are consistent with known concerns about distribution shift in learned-model planning; they are demonstrated here within one controlled benchmark, not claimed as general laws.

## Repository Structure

```
├── src/
│   ├── envs/            # Continuing-fleet selective-maintenance environment
│   ├── agents/ddqn/     # Double DQN agent, Q-network, replay buffer, checkpoints
│   ├── optimizers/      # Exact myopic optimizer + failure-risk surrogate
│   ├── m6/              # Privileged analytic H=2 planner
│   ├── milestone9/      # Point-estimate DDQN formal pipeline
│   ├── milestone10/e3/  # n-step return + seeded-replay ablations
│   ├── milestone11/e4/  # Learned-dynamics MPC (failure-free data composition)
│   ├── milestone11/e5/  # Failure-enriched composition intervention
│   ├── predictors/      # Frozen MLP RUL predictor + prediction cache generation
│   └── training/        # DDQN trainer, config identity, preflight checks
├── configs/             # Versioned experiment configurations (JSON)
├── scripts/             # Entry points: baselines, DDQN, E3–E5 runs & analyses
├── tests/               # Extensive unit/integration tests (env, baselines, DDQN, E3–E5)
├── report/figures/      # Selected result figures
├── evidence/            # Selected scientific artifacts (held-out results, M6 statistics)
├── docs/                # Reproducibility documentation
└── results/             # Result summaries for baseline/validation experiments
```

## Reproducibility

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for data acquisition, environment setup, configuration layout, and evaluation protocol, and [`docs/RESULTS.md`](docs/RESULTS.md) for a guide to the retained result artifacts. Large runtime assets (raw C-MAPSS downloads, processed prediction caches, per-seed checkpoints and rollouts) are intentionally not committed; their location is resolved through the `DRL_EXTERNAL_ROOT` environment variable (see `src/runtime_paths.py`).

Some portions of the test suite skip or error unless those external runtime assets are provisioned locally; heavy tests additionally require an explicit opt-in flag (`M9_HEAVY=1`). The committed evidence, statistics, and figures are self-contained.

## Limitations

- **Synthetic benchmark**: the fleet is constructed from C-MAPSS FD001 trajectories with synthetic replacement semantics; nothing here validates real aviation or industrial maintenance practice.
- **Privileged comparator**: the H=2 planner encodes analytic transition/failure structure. Model knowledge and planning are not separately identified, so the DDQN-vs-H=2 result does not generalize to "model-free vs planning".
- **Small scale**: N=5 slots, ≤16 actions, one dataset subset, 2 capacity levels × 4 cost regimes = 8 primary cells, five training seeds — seed-level estimates for ablations have limited precision.
- **Failure rarity dependence**: conclusions about failure-localized gaps and data-support effects are tied to this benchmark's failure frequency and reward magnitudes.
- **Bundled intervention**: the failure-coverage study improves control but changes several data-composition factors simultaneously.
- **Scope of held-out evidence**: only the primary comparison was evaluated on the sealed test split; all ablations remain validation-only.

## Technical Stack

Python · PyTorch · Gymnasium-style environment · NumPy · Pandas · scikit-learn · Matplotlib · pytest

## Project Status

Completed research prototype. The experimental campaign, statistical analysis, held-out replication, and reporting are finished; code and selected evidence artifacts are preserved as a record of the study. See `docs/RESULTS.md` for the artifact-by-artifact guide.
