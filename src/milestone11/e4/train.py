"""E4 (M11) deterministic ensemble training (Tasks 6-7).

Trains all three members on dynamics_train with episode-level bootstrap
resampling and validation-based early stopping on dynamics_validate. All three
members share normalization computed from dynamics_train only. Deterministic
RNG seeding throughout so the training is exactly reproducible per seed.
"""

from __future__ import annotations

import json
import hashlib
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import E4RawTransition
from .model import DynamicsEnsemble, build_ensemble
from .normalize import build_inputs, compute_normalization, from_json, to_json
from .paths import (
    BATCH_SIZE,
    EARLY_STOP_PATIENCE,
    INPUT_DIM,
    LEARNING_RATE,
    MAX_EPOCHS,
    OBSERVATION_DIM,
)
from .split import SPLITS

MEMBER_INIT_SEED_BASE = 1100  # deterministic per-member init seeds
DEVICE = None  # resolved at train time


def pick_device(force: str | None = None) -> str:
    if force:
        return force
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def member_init_seed(m: int, formal_seed: int) -> int:
    """Deterministic init seed for member index m (0,1,2) of a formal seed."""
    return MEMBER_INIT_SEED_BASE + 1000 * m + formal_seed


def episode_groups(transitions: List[E4RawTransition]) -> List[List[E4RawTransition]]:
    groups: Dict[str, List[E4RawTransition]] = {}
    for t in transitions:
        groups.setdefault(t.dataset_episode_id, []).append(t)
    return list(groups.values())


def episode_bootstrap(train_groups: List[List[E4RawTransition]],
                      seed: int, n_boot=None,
                      rng=None) -> List[E4RawTransition]:
    """Episode-level bootstrap: sample N episodes with replacement from the
    training episodes, never crossing episode boundaries."""
    if rng is None:
        rng = np.random.default_rng(seed)
    if n_boot is None:
        n_boot = len(train_groups)
    drawn = [train_groups[i] for i in rng.integers(0, len(train_groups), n_boot)]
    out: List[E4RawTransition] = []
    for g in drawn:
        out.extend(g)
    return out


def _standardize_targets(data: Dict[str, torch.Tensor], norm):
    nd = (data["delta"] - norm["obs_mean"]) / norm["obs_std"]
    nr = (data["reward"] - norm["reward_mean"]) / norm["reward_std"]
    return data["x"], nd, nr


def _batch_loader(x, nd, nr, batch_size, rng: np.random.Generator):
    n = x.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    for start in range(0, n, batch_size):
        j = idx[start:start + batch_size]
        yield x[j].to(DEVICE), nd[j].to(DEVICE), nr[j].to(DEVICE)


def train_member(
    model: torch.nn.Module,
    train_transitions: List[E4RawTransition],
    val_transitions: List[E4RawTransition],
    norm: Dict[str, torch.Tensor],
    device: str,
    seed: int,
    max_epochs: int = MAX_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    patience: int = EARLY_STOP_PATIENCE,
    log: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train one member with episode-level bootstrap + validation early-stop."""
    global DEVICE
    DEVICE = device
    norm_device = {k: v.to(device) if torch.is_tensor(v) else v
                   for k, v in norm.items()}
    train_data = build_inputs(train_transitions, device=device)
    val_data = build_inputs(val_transitions, device=device)
    xv, nv_d, nv_r = _standardize_targets(val_data, norm_device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_groups = episode_groups(train_transitions)
    rng = np.random.default_rng(seed)
    best_val = float("inf")
    best_state = None
    best_epoch = -1
    epochs_since_improve = 0
    history = [] if log is None else log

    for epoch in range(1, max_epochs + 1):
        model.train()
        # Episode-level bootstrap once per epoch using a fresh RNG draw.
        boot = episode_bootstrap(train_groups, seed=seed + epoch, rng=rng)
        bd = build_inputs(boot, device=device)
        xb, nb_d, nb_r = _standardize_targets(bd, norm_device)
        ep_rng = np.random.default_rng(seed * 1000 + epoch)
        tot_state = 0.0
        tot_rew = 0.0
        for x, nd, nr in _batch_loader(xb, nb_d, nb_r, batch_size, ep_rng):
            opt.zero_grad()
            pd, pr = model(x)
            l_state = F.mse_loss(pd, nd)
            l_reward = F.mse_loss(pr, nr)
            loss = l_state + l_reward
            loss.backward()
            opt.step()
            tot_state += float(l_state.detach())
            tot_rew += float(l_reward.detach())
        nb = len(xb)
        tr_state = tot_state / max(1, nb // batch_size)
        tr_rew = tot_rew / max(1, nb // batch_size)

        # Validation (normalized) loss on dynamics_validate.
        model.eval()
        with torch.no_grad():
            pd_v, pr_v = model(xv)
            v_state = F.mse_loss(pd_v, nv_d)
            v_rew = F.mse_loss(pr_v, nv_r)
        v_loss = float(v_state + v_rew)
        history.append({
            "epoch": epoch,
            "train_state": tr_state, "train_reward": tr_rew,
            "val_state": float(v_state), "val_reward": float(v_rew),
            "val_loss": v_loss,
        })
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return {
        "best_epoch": best_epoch,
        "best_validation_loss": best_val,
        "train_state_loss": min((h["train_state"] for h in history), default=None),
        "train_reward_loss": min((h["train_reward"] for h in history), default=None),
        "val_state_loss": history[best_epoch - 1]["val_state"] if best_epoch >= 1 else None,
        "val_reward_loss": history[best_epoch - 1]["val_reward"] if best_epoch >= 1 else None,
        "epochs_trained": len(history),
        "history": history,
    }


def train_ensemble(
    formal_seed: int,
    train_transitions: List[E4RawTransition],
    val_transitions: List[E4RawTransition],
    out_root: Path,
    device: str | None = None,
    max_epochs: int = MAX_EPOCHS,
) -> Dict[str, Any]:
    """Train all three members for one formal seed and save artifacts."""
    device = pick_device(device)
    norm = compute_normalization(train_transitions)
    norm_j = to_json(norm)

    member_init_seeds = [member_init_seed(m, formal_seed) for m in range(3)]
    # Constructor-time manual_seed inside DynamicsNetwork makes init deterministic
    # per member regardless of global RNG, so no global seed lock is required.
    members, results = [], []
    log: Dict[str, Any] = {}
    for m in range(3):
        model = _make_member(INPUT_DIM, member_init_seeds[m], device)
        r = train_member(model, train_transitions, val_transitions, norm,
                         device, seed=member_init_seeds[m], max_epochs=max_epochs,
                         log=log.setdefault(f"member_{m}", []))
        members.append(model)
        results.append(r)

    ensemble = DynamicsEnsemble(torch.nn.ModuleList(members), norm)
    out_dir = out_root / "models" / f"seed_{formal_seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"members_state": [dict(m.state_dict()) for m in members],
                "member_init_seeds": list(member_init_seeds),
                "normalization": norm_j,
                "formal_seed": formal_seed,
                "arch": {"input_dim": INPUT_DIM, "hidden": "128,128",
                         "output_dim": OBSERVATION_DIM + 1}},
               out_dir / "ensemble.pt")
    (out_dir / "normalization.json").write_text(json.dumps(norm_j, indent=2))
    (out_dir / "training_results.json").write_text(json.dumps(
        {"member_results": results, "member_init_seeds": list(member_init_seeds),
         "normalization": norm_j,
         "config": {"batch_size": BATCH_SIZE, "max_epochs": max_epochs,
                    "patience": EARLY_STOP_PATIENCE, "lr": LEARNING_RATE,
                    "device": device}},
        indent=2))

    return {
        "formal_seed": formal_seed,
        "member_results": results,
        "member_init_seeds": list(member_init_seeds),
        "device": device,
        "model_path": str(out_dir / "ensemble.pt"),
    }


def _make_member(input_dim: int, init_seed: int, device: str) -> torch.nn.Module:
    from .model import DynamicsNetwork

    m = DynamicsNetwork(input_dim=input_dim, init_seed=init_seed)
    m.to(device)
    return m


def load_ensemble(seed: int, out_root: Path, device: str | None = None) -> DynamicsEnsemble:
    """Load a frozen trained ensemble for ``seed`` from disk."""
    device = pick_device(device)
    d = torch.load(out_root / "models" / f"seed_{seed}" / "ensemble.pt",
                   map_location=device, weights_only=False)
    norm = from_json(d["normalization"])
    members = torch.nn.ModuleList()
    for i, sd in enumerate(d["members_state"]):
        m = _make_member(INPUT_DIM, d["member_init_seeds"][i], device)
        m.load_state_dict(sd)
        m.eval()
        members.append(m)
    return DynamicsEnsemble(members, norm)