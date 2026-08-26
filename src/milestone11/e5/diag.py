"""E5-B model diagnostics (Sections 10-11).

Reuses the frozen E4 one-step / two-step diagnostic functions verbatim (identical
model family) and adds the failure-specific reward diagnostics required by
Section 10 because E5-B training data now contains genuine failure events:

    reward MAE/RMSE on failure transitions
    reward MAE/RMSE on non-failure transitions
    predicted vs actual failure-cost reward distribution

Two-step diagnostics also report the metrics separately for windows containing a
genuine failure event (Section 11), where sample size permits.
"""

from __future__ import annotations

import json
import numpy as np

from src.milestone11.e4.dataset import E4RawTransition
from src.milestone11.e4.model import DynamicsEnsemble
from src.milestone11.e4.diag import (
    one_step_metrics,
    two_step_metrics,
    _flat_obs,
    _one_hot_np,
    ensemble_predict,
    _two_step_windows,
)
from .dataset import load_dataset
from .paths import E5_OUTPUT_ROOT, FORMAL_SEEDS, GAMMA, OBSERVATION_DIM
from .split import SOURCES, partition_episodes


def _is_failure(t: E4RawTransition) -> bool:
    return t.cost_failure > 0


def failure_reward_metrics(ensemble: DynamicsEnsemble,
                           holdout: list[E4RawTransition],
                           device: str = "cpu") -> dict:
    """Failure-specific reward prediction errors on the held-out transitions.

    Rewards are predicted by the single-step ensemble (identical to one-step
    reward RMSE/MAE machinery in the E4 diagnostics) and grouped by whether the
    transition carries a genuine failure event (cost_failure > 0).
    """
    obs, nxt, act, rew = _flat_obs(holdout)
    d_mean, r_mean, d_std = ensemble_predict(ensemble, obs, act, device)
    f_mask = np.array([_is_failure(t) for t in holdout], dtype=bool)
    nf_mask = ~f_mask

    def _err(sel):
        if sel.sum() == 0:
            return None
        r_true = rew[sel]
        r_pred = r_mean[sel]
        return {
            "n": int(sel.sum()),
            "reward_mae": float(np.mean(np.abs(r_pred - r_true))),
            "reward_rmse": float(np.sqrt(np.mean((r_pred - r_true) ** 2))),
            "pred_mean": float(np.mean(r_pred)),
            "true_mean": float(np.mean(r_true)),
            "pred_min": float(np.min(r_pred)),
            "pred_max": float(np.max(r_pred)),
            "true_support": sorted([float(v) for v in np.unique(r_true)]),
        }

    fail_metrics = _err(f_mask)
    nfail_metrics = _err(nf_mask)
    return {
        "failure_transitions": {
            "n": int(f_mask.sum()) if len(f_mask) else 0,
            "metrics": fail_metrics,
        },
        "non_failure_transitions": {
            "n": int(nf_mask.sum()) if len(nf_mask) else 0,
            "metrics": nfail_metrics,
        },
    }


def two_step_failure_windows(ensemble: DynamicsEnsemble,
                             holdout: list[E4RawTransition],
                             device: str = "cpu") -> dict:
    """Two-step discounted-return / rollput error split by failure-window status.

    A window (t0, t1) is a "failure window" if either transition carries a
    genuine failure event. Reported where sample size permits.
    """
    windows = _two_step_windows(holdout)
    fail_win, nfail_win = [], []
    for (t0, t1) in windows:
        (fail_win if (_is_failure(t0) or _is_failure(t1)) else nfail_win).append(
            (t0, t1))

    def _metrics(win):
        if not win:
            return None
        one_rmse, two_rmse, rew1_err, disc_err = [], [], [], []
        for (t0, t1) in win:
            o0 = np.array(t0.observation_t, dtype=np.float32)
            o1 = np.array(t1.observation_t, dtype=np.float32)
            o2 = np.array(t1.next_observation_t, dtype=np.float32)
            a0 = np.array([t0.action_id_t], dtype=np.int64)
            a1 = np.array([t1.action_id_t], dtype=np.int64)
            r0t, r1t = t0.reward_t, t1.reward_t
            d1, r0h, _ = ensemble_predict(ensemble, o0[None], a0, device)
            o1_hat = o0 + d1[0]
            o1_hat_c = np.clip(o1_hat, 0.0, 1.0)
            d2, r1h, _ = ensemble_predict(ensemble, o1_hat_c[None], a1, device)
            o2_hat = o1_hat_c + d2[0]
            one_rmse.append(float(np.sqrt(np.mean((o1_hat - o1) ** 2))))
            two_rmse.append(float(np.sqrt(np.mean((o2_hat - o2) ** 2))))
            rew1_err.append(abs(float(r0h[0]) - r0t))
            pred_ret = float(r0h[0]) + GAMMA * float(r1h[0])
            true_ret = r0t + GAMMA * r1t
            disc_err.append(abs(pred_ret - true_ret))
        return {
            "n_windows": len(win),
            "one_step_state_rmse": float(np.mean(one_rmse)),
            "two_step_state_rmse": float(np.mean(two_rmse)),
            "one_step_reward_mae": float(np.mean(rew1_err)),
            "two_step_discounted_return_mae": float(np.mean(disc_err)),
        }

    return {
        "failure_windows": _metrics(fail_win),
        "non_failure_windows": _metrics(nfail_win),
    }


def run_diagnostics(ensemble: DynamicsEnsemble, seed: int,
                    out_root: Path = E5_OUTPUT_ROOT,
                    device: str = "cpu") -> dict:
    per_source = load_dataset(seed, out_root)
    holdout = []
    train = []
    for src in SOURCES:
        parts = partition_episodes(per_source[src], src)
        holdout.extend(parts["dynamics_holdout"])
        train.extend(parts["dynamics_train"])
    return {
        "formal_seed": seed,
        "one_step": one_step_metrics(ensemble, holdout, device),
        "two_step": two_step_metrics(ensemble, holdout, device),
        "failure_reward": failure_reward_metrics(ensemble, holdout, device),
        "two_step_failure_windows": two_step_failure_windows(ensemble, holdout, device),
        "n_holdout_transitions": len(holdout),
        "n_train_transitions": len(train),
    }