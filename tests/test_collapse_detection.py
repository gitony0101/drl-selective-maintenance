"""Tests for predictor collapse detection (src.predictors.collapse_detector)."""

import numpy as np

from src.predictors.collapse_detector import detect_collapse


def _true_rul(n=64):
    rng = np.random.default_rng(0)
    return rng.uniform(1.0, 125.0, size=n)


def test_constant_predictions_fail():
    preds = np.full(64, 42.0)
    res = detect_collapse(preds, _true_rul())
    assert res.is_collapsed
    assert res.failure_reason is not None
    assert "constant" in res.failure_reason.lower()


def test_near_constant_predictions_fail():
    rng = np.random.default_rng(1)
    true = _true_rul()
    preds = 40.0 + rng.normal(0.0, 0.05, size=len(true))  # tiny spread vs true std
    res = detect_collapse(preds, true, std_ratio_threshold=0.1)
    assert res.is_collapsed
    assert res.std_ratio < 0.1


def test_nan_inf_fail():
    true = _true_rul()
    preds_nan = np.full(64, 40.0)
    preds_nan[3] = np.nan
    res = detect_collapse(preds_nan, true)
    assert res.is_collapsed
    assert res.failure_reason is not None
    reason_lower = res.failure_reason.lower()
    assert "non-finite" in reason_lower or "nan" in reason_lower

    preds_inf = np.full(64, 40.0)
    preds_inf[5] = np.inf
    res2 = detect_collapse(preds_inf, true)
    assert res2.is_collapsed


def test_valid_varying_predictions_pass():
    rng = np.random.default_rng(2)
    true = _true_rul()
    preds = true + rng.normal(0.0, 3.0, size=len(true))  # strong correlation, full spread
    res = detect_collapse(preds, true)
    assert not res.is_collapsed, f"unexpected collapse: {res.failure_reason}"


def test_failure_messages_informative():
    preds = np.full(32, 7.0)
    res = detect_collapse(preds, _true_rul(32))
    assert res.is_collapsed
    # The reason should reference at least one diagnostic quantity that explains the failure.
    assert res.failure_reason and len(res.failure_reason) > 10
    assert any(tok in res.failure_reason.lower() for tok in ("std", "constant", "range", "unique", "correlation"))
