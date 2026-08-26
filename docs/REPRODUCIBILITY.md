# Reproducibility

This document describes how to set up and re-run the pipeline. It covers data
acquisition, environment setup, configuration layout, and the evaluation
protocol.

## 1. Data acquisition

The benchmark is built from **NASA C-MAPSS FD001** run-to-failure trajectories.
The raw dataset is **not included** in this repository; obtain it from the
official NASA Prognostics Data Repository (C-MAPSS turbofan engine degradation
simulation data set) and place the three FD001 files in this layout:

```
data/raw/cmapss/FD001/
├── train_FD001.txt
├── test_FD001.txt
└── RUL_FD001.txt
```

Do not place any local machine paths inside tracked configuration files.

## 2. Environment setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

PyTorch runs on CPU by default; Apple MPS (`--device mps`) and CUDA are
supported where available. Device selection is explicit at every entry point.

## 3. External runtime assets

Large generated artifacts (per-seed prediction caches, frozen predictor
checkpoints, training/rollout outputs) live outside version control. Their root
is resolved by `src/runtime_paths.py`:

1. `$DRL_EXTERNAL_ROOT`, if set;
2. otherwise `<repo>/../drl_external_assets`.

Expected external layout (created by the pipeline scripts as needed):

```
$DRL_EXTERNAL_ROOT/
├── m9_point_caches/seed_<6521..6525>/   # per-seed V2 prediction caches
├── m10_e3_outputs/                      # E3 ablation outputs
├── m11_e4_outputs/                      # learned-MPC E4 outputs
└── m12_e5_outputs/                      # failure-coverage E5 outputs
```

## 4. Configuration

All experiment parameters are pinned in versioned JSON files:

- `configs/env/m2_v1.json` — environment contract (fleet size, capacity,
  decision interval, cost regimes).
- `configs/scenarios/*.json` — serialized scenario banks for paired evaluation
  (pilot and validation sets for K=1 and K=2 across all four cost regimes).
- `configs/baselines/m3_v1.json`, `configs/myopic/m4_*.json`,
  `configs/agents/ddqn_*.json`, `configs/predictor/mse_baseline.json` —
  baseline, optimizer, agent, and predictor settings.

Random seeds (6521–6525) and all reported hyperparameters are part of the
frozen experimental record and must not be changed retroactively.

## 5. Evaluation protocol

- Engine-level splits are frozen and disjoint:
  `predictor_train` (60) / `predictor_validation` (15) / `rl_validation` (10) /
  sealed `rl_test` (15). True RUL never enters observations.
- All policies are evaluated **paired** on identical scenario banks with fixed
  reset seeds, five training seeds per cell.
- Uncertainty is quantified by a preregistered hierarchical bootstrap
  (10,000 resamples) over scenarios-within-seeds, reporting cellwise 95%
  intervals.
- Only the primary comparison was evaluated once on the sealed test split;
  everything else is validation-only by design.

## 6. Test suite

```bash
# Fresh clone / public core:
pytest \
  -m "not legacy_v1 and not requires_v2_cache and not requires_external_assets"

# Full (needs runtime assets provisioned under DRL_EXTERNAL_ROOT):
pytest

# Heavy integration tests (explicit opt-in):
M9_HEAVY=1 pytest
```

### Test markers

- `legacy_v1` — tests that exercise invalidated V1 artifacts or require a
  frozen V1 reproduction environment; not part of the V2 canonical pipeline.
- `requires_v2_cache` — post-training integration tests that require the real
  generated V2 prediction cache to be present on disk.
- `requires_external_assets` — tests requiring external generated checkpoints,
  caches, or experiment assets (e.g., frozen M8 checkpoints, M9 baseline repair
  outputs, DDQN eval replay outputs) under `DRL_EXTERNAL_ROOT`.

### Frozen held-out driver

The evaluation driver under
`evidence/heldout/final_test_evaluation/run_core_m6_test.py` is a **standalone
evaluation driver** for the sealed held-out `rl_test` split. It is intentionally
excluded from ordinary pytest collection via a root `conftest.py` hook because
it requires the full external `rl_test` prediction cache and the frozen M6
worktree context. Run it directly with the required external assets provisioned.

### Heavy tests

Integration tests marked with the opt-in flag `M9_HEAVY=1` require the full
per-seed V2 prediction cache and are excluded from the public-core test suite.
They require explicit provisioning.

Tests that require provisioned runtime caches skip or fail-closed when those
assets are absent; this is intentional and documented in `pytest.ini`.

## 7. Entry points

Representative scripts under `scripts/`:

| Script | Purpose |
|---|---|
| `run_m2_environment_smoke.py` | Environment smoke check |
| `run_m3_baselines.py` | Rule-based baseline formal runs |
| `run_m4_exact_myopic.py` | Exact myopic optimizer |
| `train_ddqn.py` / `evaluate_ddqn.py` | Point-DDQN training/evaluation |
| `run_m9_point_estimate.py` | Point-estimate formal pipeline |
| `run_e3_formal.py`, `evaluate_e3.py`, `analyze_e3.py` | n-step/replay-seeding ablation |
| `run_e4_*.py`, `run_e5_*.py` | Learned-MPC studies |

Run each script with `--help` for its arguments.