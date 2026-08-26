# Results Guide

This repository ships a curated subset of the study's result artifacts. The
full frozen report and internal evidence archives are not part of this public
distribution. Every number quoted in the README traces to one of the artifacts
below.

## Held-out evaluation (primary result)

- `evidence/heldout/RL_TEST_FINAL_RESULTS.json`
  Status of the sealed held-out run: primary comparison complete; secondary
  comparisons were not tested on the held-out split (by protocol).
- `evidence/heldout/RL_TEST_M6_RESULTS.json`
  Per-cell held-out numbers for the primary comparison: Point-DDQN minus H=2
  mean episodic cost, hierarchical bootstrap 95% CIs, and cost-decomposition
  deltas (failure / preventive / unused-life) for all eight K×regime cells.
  **This is the source of record for the README's Key Findings table.**
- `evidence/heldout/final_test_evaluation/`
  Raw per-cell evaluation bundles (`rl_test_*_final_v1.json`,
  `core_episode_metrics.csv`, scenario manifests, run manifests, bank SHA256s)
  for the eight capacity–cost cells, plus the isolated evaluation driver
  `run_core_m6_test.py`.

## Primary-comparison validation statistics

- `evidence/m6/10_statistics/M6_STATISTICAL_RESULTS.json` — full per-cell
  descriptive statistics, paired contrasts, and bootstrap output on the
  validation matrix (8/8 ordering agreement with the held-out result).
- `evidence/m6/10_statistics/M6_STATISTICAL_ANALYSIS.md` — analysis narrative
  and interpretation of the same statistics.
- `evidence/m6/10_statistics/M6_PER_CELL_RESULTS.csv` — compact per-cell table.
- `evidence/m6/10_statistics/VALIDATION_SUMMARY.md` — validated summary
  of the primary comparison with hierarchical bootstrap CIs.

## Figures

- `report/figures/ddqn_vs_h2_heldout.pdf` — held-out primary
  comparison across all eight cells.
- `report/figures/learned_mpc_failure_coverage.pdf` /
  `.png` — learned-MPC control-quality diagnostic (E4 failure-free vs E5
  failure-enriched training mixture).

## Baseline and validation runs

- `results/baselines/*/` — formal rule-based baseline runs: episode-level
  parquet outputs, threshold-search summaries, selected thresholds, provenance
  manifests.
- `results/milestone4/scientific_validation_v1/` — exact-myopic optimizer
  scientific-validation outputs: paired episode metrics, candidate summaries,
  selection decision, bootstrap summary.
- `results/milestone5/experiment_matrix.json` — DDQN experiment-matrix
  definition.

## Validation-only ablation summaries

Validation-only ablation results (n-step returns, planner-seeded replay,
learned-MPC compositions) are summarized in the README Key Findings section.
Their full internal evidence archives are not redistributed here.

## Validation-only summaries (model-based control diagnostics)

- `evidence/ablations/nstep_replay_summary.json` — n-step return ablation
  summary (E3).
- `evidence/model_based/learned_mpc_failure_free_summary.json` — learned-MPC
  diagnostic under failure-free training mixture (E4).
- `evidence/model_based/learned_mpc_failure_enriched_summary.json` — learned-MPC
  diagnostic under failure-enriched training mixture (E5).

## Integrity

Retained JSON/CSV artifacts are verbatim copies of the originals that produced
the reported figures and tables. Scenario-bank SHA256 hashes inside the
run manifests pin the exact evaluation inputs.