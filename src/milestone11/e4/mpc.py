"""E4 (M11) exhaustive H=2 learned MPC planner (Sections 17-19).

Given a 10-dim observation and an ensemble of learned dynamics models, exhaustively
enumerate all |A|^2 = 256 two-step action sequences, score each by the ensemble-mean
predicted two-step return G_bar = (1/M) sum_m (r_0^m + gamma r_1^m), and return the
first action of the argmax sequence (deterministic lowest-action-id tie-break).
Receding horizon: replan at every step. Only the learned models are queried; the
simulator dynamics are never touched.

Predicted observations are bounded to the canonical feature range [0,1] (Section 19):
the raw violation fraction is recorded before clamping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from .model import DynamicsEnsemble
from .paths import GAMMA, NUM_ACTIONS, OBSERVATION_DIM

FEATURE_LOW = 0.0
FEATURE_HIGH = 1.0


@dataclass
class MPCResult:
    action_id: int
    sequence: Tuple[int, int]
    predicted_return: float
    ensemble_std: float
    selected_sequence_std: float
    candidate_returns: Dict[Tuple[int, int], float] = field(default_factory=dict)
    raw_out_of_range_fraction_step1: float = 0.0
    raw_out_of_range_fraction_step2: float = 0.0
    horizon_used: int = 2


def _one_hot(actions, num_actions: int = NUM_ACTIONS, device=None) -> torch.Tensor:
    t = torch.tensor(actions).to(torch.long)
    if device is not None:
        t = t.to(device)
    return torch.nn.functional.one_hot(t, num_classes=num_actions).float()


def _mean_and_std(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return mean, var ** 0.5


def _pred_obs(o: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    return o + delta


def enumerate_sequences(horizon: int = 2) -> List[Tuple[int, int]]:
    """All action-sequence tuples for exhaustive planning."""
    if horizon == 1:
        return [(a0, -1) for a0 in range(NUM_ACTIONS)]
    if horizon == 2:
        return [(a0, a1) for a0 in range(NUM_ACTIONS) for a1 in range(NUM_ACTIONS)]
    raise ValueError(f"unsupported planning horizon {horizon}")


def score_sequence(ensemble: DynamicsEnsemble, o: torch.Tensor,
                   seq: Tuple[int, int], horizon: int) -> Tuple[float, float, Dict[str, float]]:
    """Score a sequence by ensemble-mean discounted return.

    Returns (G_bar, ensemble_std_of_returns, raw_oor_by_step) where raw_oor_by_step
    counts predicted-observation violations of [0,1] across members before clamp.
    """
    M = ensemble.M
    g_members: List[float] = []
    o_oor: Dict[str, float] = {"step1": 0.0, "step2": 0.0}
    # o is a single [10] observation -> work with a batch-of-1 row.
    if o.ndim == 1:
        o = o.unsqueeze(0)
    if o.shape != (1, OBSERVATION_DIM):
        raise ValueError(f"expected single [10] observation, got {tuple(o.shape)}")

    a0 = seq[0]
    a1_t = seq[1] if horizon >= 2 else None

    for m in range(M):
        x0 = torch.cat([o, _one_hot([a0], device=o.device)], dim=1)
        delta1, r0 = ensemble.predict_member(m, x0)
        pred_o1 = _pred_obs(o, delta1)
        # raw out-of-range before clamp
        o_oor["step1"] += float(torch.sum((pred_o1 < FEATURE_LOW) | (pred_o1 > FEATURE_HIGH)).item())
        o1_clamped = torch.clamp(pred_o1, FEATURE_LOW, FEATURE_HIGH)

        if a1_t is not None:
            x1 = torch.cat([o1_clamped, _one_hot([a1_t], device=o.device)], dim=1)
            delta2, r1 = ensemble.predict_member(m, x1)
            pred_o2 = _pred_obs(o1_clamped, delta2)
            o_oor["step2"] += float(torch.sum((pred_o2 < FEATURE_LOW) | (pred_o2 > FEATURE_HIGH)).item())
        else:
            r1 = torch.tensor([0.0])

        if r0.ndim == 1:
            r0v = float(r0[0].detach()) if r0.numel() else float(r0.detach())
        else:
            r0v = float(r0.detach())
        if r1.ndim == 1:
            r1v = float(r1[0].detach()) if r1.numel() else float(r1.detach())
        else:
            r1v = float(r1.detach())
        g = r0v + GAMMA * r1v
        g_members.append(g)

    n_rows = 1
    o_oor["step1"] = o_oor["step1"] / (M * OBSERVATION_DIM * n_rows)
    o_oor["step2"] = o_oor["step2"] / (M * OBSERVATION_DIM * n_rows)
    mean, std = _mean_and_std(g_members)
    return mean, std, o_oor


def plan(ensemble: DynamicsEnsemble, observation: torch.Tensor,
         steps_remaining_in_episode: int) -> MPCResult:
    """Exhaustive H=2 (or H=1 at the final step) learned MPC plan.

    observation: [10] tensor. Returns the MPCResult for execution.
    """
    if observation.ndim != 1 or observation.shape[0] != OBSERVATION_DIM:
        raise ValueError(f"expected [10] observation, got {tuple(observation.shape)}")
    horizon = 2 if steps_remaining_in_episode > 1 else 1
    sequences = enumerate_sequences(horizon)

    best_mean = -float("inf")
    best_seq: Optional[Tuple[int, int]] = None
    best_std = 0.0
    cand_returns: Dict[Tuple[int, int], float] = {}
    candidate_stds: List[float] = []
    max_oor1, max_oor2 = 0.0, 0.0

    # Exhaustive enumeration over sequences, deterministic lowest-id tie-break:
    # enumerate in lexicographic order, keep strictly-greater improvements.
    for seq in sequences:
        mean, std, oor = score_sequence(ensemble, observation, seq, horizon)
        cand_returns[seq] = mean
        candidate_stds.append(std)
        max_oor1 = max(max_oor1, oor["step1"])
        max_oor2 = max(max_oor2, oor["step2"])
        if mean > best_mean:  # strict: first (lowest-id) sequence wins ties
            best_mean = mean
            best_seq = seq
            best_std = std

    if best_seq is None:
        raise RuntimeError("no sequence scored")

    # Ensemble disagreement of the selected sequence vs candidate average.
    sel_std = best_std
    avg_cand_std = sum(candidate_stds) / len(candidate_stds) if candidate_stds else 0.0

    return MPCResult(
        action_id=int(best_seq[0]),
        sequence=best_seq,
        predicted_return=best_mean,
        ensemble_std=best_std,
        selected_sequence_std=sel_std,
        candidate_returns={f"{s}": v for s, v in cand_returns.items()},
        raw_out_of_range_fraction_step1=max_oor1,
        raw_out_of_range_fraction_step2=max_oor2,
        horizon_used=horizon,
    )


class LearnedMPCPolicy:
    """Receding-horizon H=2 learned MPC acting as an environment policy.

    Evaluates MPC on the frozen paired rl_validation protocol. H=1 used at the
    last remaining step. Deterministic lexicographic tie-break.
    """

    def __init__(self, ensemble: DynamicsEnsemble, device: str = "cpu") -> None:
        self.ensemble = ensemble.eval()
        self.device = device
        self.last_plan: Optional[MPCResult] = None
        self.n_plans = 0

    def select_action_id(self, observation) -> int:
        o = torch.tensor([float(x) for x in observation], dtype=torch.float32)
        if self.device == "mps":
            o = o.to("mps")
        res = plan(self.ensemble, o, steps_remaining_in_episode=2)
        self.last_plan = res
        self.n_plans += 1
        return res.action_id

    def select_action_id_last_step(self, observation) -> int:
        o = torch.tensor([float(x) for x in observation], dtype=torch.float32)
        if self.device == "mps":
            o = o.to("mps")
        res = plan(self.ensemble, o, steps_remaining_in_episode=1)
        self.last_plan = res
        return res.action_id