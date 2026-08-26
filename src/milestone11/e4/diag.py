"""E4 (M11) model diagnostics (Tasks 8-10, Sections 14-16).

Implements:
  - one-step held-out model test + persistence / per-action-mean baselines
  - two-step rollout test (compounding error)
  - partial-observability local-variance diagnostic

All diagnostics run on the frozen dynamics_holdout and use the frozen ensemble.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from .dataset import E4RawTransition, load_dataset
from .model import DynamicsEnsemble
from .paths import E4_OUTPUT_ROOT, FORMAL_SEEDS, GAMMA, NUM_ACTIONS, OBSERVATION_DIM
from .split import SOURCES, SPLITS, partition_episodes

FEATURE_LOW, FEATURE_HIGH = 0.0, 1.0


def _flat_obs(transitions: List[E4RawTransition]) -> Tuple[np.ndarray, np.ndarray,
                                                           np.ndarray, np.ndarray]:
    obs = np.array([t.observation_t for t in transitions], dtype=np.float32)
    nxt = np.array([t.next_observation_t for t in transitions], dtype=np.float32)
    act = np.array([t.action_id_t for t in transitions], dtype=np.int64)
    rew = np.array([t.reward_t for t in transitions], dtype=np.float32)
    return obs, nxt, act, rew


def _one_hot_np(actions: np.ndarray, num_actions: int = NUM_ACTIONS) -> np.ndarray:
    out = np.zeros((len(actions), num_actions), dtype=np.float32)
    out[np.arange(len(actions)), actions] = 1.0
    return out


def ensemble_predict(ensemble: DynamicsEnsemble, obs: np.ndarray,
                     actions: np.ndarray, device: str = "cpu") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ensemble-mean (delta_obs, reward) plus across-member std for a batch.

    obs: [B,10], actions: [B]. Returns (delta_mean[B,10], reward_mean[B],
    delta_std_ensemble[B,10]). Input obs are pre-clamped to [0,1].
    """
    x = torch.from_numpy(np.concatenate([obs, _one_hot_np(actions)], axis=1)).float()
    if device == "mps" or device == "cuda":
        x = x.to(device)
    deltas, rewards = [], []
    for m in range(ensemble.M):
        d, r = ensemble.predict_member(m, x)
        deltas.append(d.detach().cpu())
        rewards.append(r.detach().cpu())
    d_all = torch.stack(deltas, dim=0)      # [M,B,10]
    r_all = torch.stack(rewards, dim=0)     # [M,B]
    d_mean = d_all.mean(0).numpy()
    d_std = d_all.std(0).numpy()
    r_mean = r_all.mean(0).numpy()
    return d_mean, r_mean, d_std


def one_step_metrics(ensemble: DynamicsEnsemble, holdout: List[E4RawTransition],
                     device: str = "cpu") -> Dict[str, Any]:
    obs, nxt, act, rew = _flat_obs(holdout)
    delta_true = nxt - obs
    d_mean, r_mean, d_std = ensemble_predict(ensemble, obs, act, device)
    nxt_hat = obs + d_mean

    pred_err = nxt_hat - nxt                       # next-observation error
    rmse = float(np.sqrt(np.mean(pred_err ** 2)))
    mae = float(np.mean(np.abs(pred_err)))
    per_dim_rmse = [float(np.sqrt(np.mean(pred_err[:, i] ** 2))) for i in range(OBSERVATION_DIM)]
    per_dim_mae = [float(np.mean(np.abs(pred_err[:, i]))) for i in range(OBSERVATION_DIM)]

    rew_err = r_mean - rew
    rew_rmse = float(np.sqrt(np.mean(rew_err ** 2)))
    rew_mae = float(np.mean(np.abs(rew_err)))

    # predictive disagreement: mean across members of |member - ensemble-mean| for obs&rew
    with torch.no_grad():
        x = torch.from_numpy(
            np.concatenate([obs, _one_hot_np(act)], axis=1)).float()
        if device in ("mps", "cuda"):
            x = x.to(device)
        member_deltas, member_rews = [], []
        for m in range(ensemble.M):
            dm, rm = ensemble.predict_member(m, x)
            member_deltas.append(dm.detach().cpu().numpy())
            member_rews.append(rm.detach().cpu().numpy())
    md = np.stack(member_deltas)                   # [M,B,10]
    disagreement = float(np.mean(
        np.mean(np.abs(md - md.mean(0, keepdims=True)), axis=(0, 2))))

    # raw out-of-range prediction rate (before clamp): nxt_hat outside [0,1]
    oob = ((nxt_hat < FEATURE_LOW) | (nxt_hat > FEATURE_HIGH))
    raw_oor = float(oob.mean())
    per_dim_oor = [float(oob[:, i].mean()) for i in range(OBSERVATION_DIM)]

    # ---------- baselines ----------
    # persistence: nxt_hat = obs_t
    pers_err = obs - nxt
    pers_rmse = float(np.sqrt(np.mean(pers_err ** 2)))
    # per-action-mean reward baseline
    by_action = defaultdict(list)
    for a, r in zip(act, rew):
        by_action[int(a)].append(float(r))
    act_mean_reward = {a: float(np.mean(v)) for a, v in by_action.items()}
    baseline_rew_pred = np.array([act_mean_reward.get(int(a), 0.0) for a in act])
    base_rew_rmse = float(np.sqrt(np.mean((baseline_rew_pred - rew) ** 2)))

    return {
        "n": len(holdout),
        "next_obs_rmse": rmse, "next_obs_mae": mae,
        "per_dim_rmse": per_dim_rmse, "per_dim_mae": per_dim_mae,
        "reward_rmse": rew_rmse, "reward_mae": rew_mae,
        "disagreement": disagreement,
        "raw_out_of_range_rate": raw_oor, "per_dim_oor": per_dim_oor,
        "baselines": {
            "persistence_next_obs_rmse": pers_rmse,
            "per_action_mean_reward_rmse": base_rew_rmse,
            "per_action_mean_rewards": {f"{k}": v for k, v in
                                        sorted(act_mean_reward.items())},
        },
        "hard_gate": {
            "model_beats_persistence": rmse < pers_rmse,
            "model_beats_per_action_reward": rew_rmse < base_rew_rmse,
        },
    }


def _two_step_windows(transitions: List[E4RawTransition]) -> List[Tuple[E4RawTransition, E4RawTransition]]:
    """All valid consecutive-step pairs within the same episode (no crossing)."""
    per_ep: Dict[str, List[E4RawTransition]] = defaultdict(list)
    for t in transitions:
        per_ep[t.dataset_episode_id].append(t)
    windows = []
    for ep, trs in per_ep.items():
        ordered = sorted(trs, key=lambda t: t.step_index)
        for i in range(len(ordered) - 1):
            if (ordered[i].step_index + 1 == ordered[i + 1].step_index):
                windows.append((ordered[i], ordered[i + 1]))
    return windows


def two_step_metrics(ensemble: DynamicsEnsemble, holdout: List[E4RawTransition],
                     device: str = "cpu") -> Dict[str, Any]:
    windows = _two_step_windows(holdout)
    one_rmse_all, two_rmse_all = [], []
    rew1_err_all = []
    disc_return_err_all = []
    oob_one = 0.0
    oob_two = 0.0
    n_obs_total = 0.0
    n_win = 0
    for (t0, t1) in windows:
        n_win += 1
        o0 = np.array(t0.observation_t, dtype=np.float32)
        o1 = np.array(t1.observation_t, dtype=np.float32)
        o2 = np.array(t1.next_observation_t, dtype=np.float32)
        a0 = np.array([t0.action_id_t], dtype=np.int64)
        a1 = np.array([t1.action_id_t], dtype=np.int64)
        r0t, r1t = t0.reward_t, t1.reward_t

        # step 1: o0, a0 -> pred o1_hat
        d1, r0h, _ = ensemble_predict(ensemble, o0[None], a0, device)
        o1_hat = o0 + d1[0]
        o1_hat_c = np.clip(o1_hat, FEATURE_LOW, FEATURE_HIGH)
        one_rmse_all.append(float(np.sqrt(np.mean((o1_hat - o1) ** 2))))
        rew1_err_all.append(abs(float(r0h[0]) - r0t))
        n_obs_total += OBSERVATION_DIM
        oob_one += float(np.sum((o1_hat < FEATURE_LOW) | (o1_hat > FEATURE_HIGH)))

        # step 2: o1_hat_c, a1 -> pred o2_hat  (compounding: feed the prediction)
        d2, r1h, _ = ensemble_predict(ensemble, o1_hat_c[None], a1, device)
        o2_hat = o1_hat_c + d2[0]
        o2_hat_c = np.clip(o2_hat, FEATURE_LOW, FEATURE_HIGH)
        two_rmse_all.append(float(np.sqrt(np.mean((o2_hat - o2) ** 2))))
        oob_two += float(np.sum((o2_hat < FEATURE_LOW) | (o2_hat > FEATURE_HIGH)))

        # two-step discounted-return error
        pred_ret = float(r0h[0]) + GAMMA * float(r1h[0])
        true_ret = r0t + GAMMA * r1t
        disc_return_err_all.append(abs(pred_ret - true_ret))

    return {
        "n_windows": n_win,
        "one_step_state_rmse": float(np.mean(one_rmse_all)) if one_rmse_all else None,
        "two_step_state_rmse": float(np.mean(two_rmse_all)) if two_rmse_all else None,
        "one_step_reward_mae": float(np.mean(rew1_err_all)) if rew1_err_all else None,
        "two_step_discounted_return_mae": (float(np.mean(disc_return_err_all))
                                           if disc_return_err_all else None),
        "compounding_ratio": (float(np.mean(two_rmse_all) / np.mean(one_rmse_all))
                              if one_rmse_all and np.mean(one_rmse_all) > 0 else None),
        "raw_oob_one_step_frac": float(oob_one / n_obs_total) if n_obs_total else 0.0,
        "raw_oob_two_step_frac": float(oob_two / n_obs_total) if n_obs_total else 0.0,
        "no_cross_episode": True,
    }


def partial_observability_diag(transitions: List[E4RawTransition],
                               k: int = 20) -> Dict[str, Any]:
    """Nearest-neighbour local-variance diagnostic (Section 16).

    For a sample of (obs, action) points, find k nearest transitions sharing the
    same action in observation space and report the std of next-observation
    residuals — an estimate of irreducible conditional variance.
    """
    obs, nxt, act, _ = _flat_obs(transitions)
    # Sample a subset of points for tractability.
    rng = np.random.default_rng(1234)
    if len(obs) > 4000:
        idx = rng.choice(len(obs), 4000, replace=False)
    else:
        idx = np.arange(len(obs))
    sample_obs, sample_nxt, sample_act = obs[idx], nxt[idx], act[idx]

    residual_std = []
    # Only keep actions with reasonable support.
    act_counts = defaultdict(int)
    for a in act:
        act_counts[int(a)] += 1
    per_act_group = defaultdict(list)
    for (o, n, a) in zip(sample_obs, sample_nxt, sample_act):
        per_act_group[int(a)].append((o, n))
    for a, grp in per_act_group.items():
        if len(grp) < k:
            continue
        arr = np.array([o for o, _ in grp])
        nxt_arr = np.array([n for _, n in grp])
        for i, (o, _n) in enumerate(grp[:200]):   # sample within action
            dist = np.sum((arr - o) ** 2, axis=1)
            nbrs = np.argsort(dist)[1:k + 1]
            residual_std.append(float(np.std(nxt_arr[nbrs] - nxt_arr[i], axis=1).mean()))
    local_std = float(np.mean(residual_std)) if residual_std else None
    # Global marginal std as reference.
    global_next_std = float(np.std(nxt - obs))
    return {
        "local_residual_std_mean": local_std,
        "n_action_local_samples": len(residual_std),
        "global_delta_obs_std": global_next_std,
    }


def run_all_diags(ensemble: DynamicsEnsemble, seed: int, out_root: Path = E4_OUTPUT_ROOT,
                  device: str = "cpu") -> Dict[str, Any]:
    per_source = load_dataset(seed, out_root)
    holdout: List[E4RawTransition] = []
    train: List[E4RawTransition] = []
    for src in SOURCES:
        parts = partition_episodes(per_source[src])
        holdout.extend(parts["dynamics_holdout"])
        train.extend(parts["dynamics_train"])
    return {
        "formal_seed": seed,
        "one_step": one_step_metrics(ensemble, holdout, device),
        "two_step": two_step_metrics(ensemble, holdout, device),
        "partial_observability": partial_observability_diag(train, k=20),
        "n_holdout_transitions": len(holdout),
        "n_train_transitions": len(train),
    }