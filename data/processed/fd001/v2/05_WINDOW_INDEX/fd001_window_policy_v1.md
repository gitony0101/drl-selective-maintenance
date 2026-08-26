# FD001 Window Policy V1

## Window Construction

Each sample window consists of exactly `sequence_length = 50` consecutive feature observations
ending at the `target_cycle` (the cycle being predicted).

## Early-Cycle Left Padding

For target cycles 1-49, insufficient history exists. These windows use **left padding** with
the first observed feature row for that unit.

- `left_pad_count = max(0, 50 - target_cycle)`
- `window_start_cycle = max(1, target_cycle - 49)`
- Padding content: First row's feature values repeated `left_pad_count` times

## RUL Labeling

- `true_rul_raw = max_cycle - target_cycle` (uncapped remaining useful life)
- `true_rul_capped = min(true_rul_raw, 125)` (capped for training stability)

## Stride

Windows are constructed for **every cycle** of every unit (stride = 1).
This produces one window index row per cycle table row.

## Protocol

All windows use:
- sequence_length: 50
- window_stride: 1
- early_cycle_policy: left_pad_with_first_observed_feature_row
- rul_cap: 125
- protocol_version: fd001_data_v1
