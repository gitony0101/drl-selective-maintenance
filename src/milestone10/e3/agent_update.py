"""E3 (M10) n-step Double DQN target + single gradient update.

Mirrors the frozen one-step ``DDQNAgent.update`` (network flow, optimizer,
Huber loss, gradient clipping, stop-gradient, online-only updates) but
generalises the TD target to an n-step bootstrap (Section 15):

    a_star = argmax_a Q_online(s_(t+k), a)       # online
    target_value = Q_target(s_(t+k), a_star)     # target (detached)
    y_t = R_t^(k) + gamma^k * (1 - terminated_for_bootstrap) * target_value

where ``R_t^(k) = discounted_reward_sum`` and ``k = bootstrap_steps``.

At ``k == 1`` with ``discounted_reward_sum = r_t`` and
``bootstrap_observation = next_obs_t`` the formula is numerically identical to
the frozen one-step Double DQN target, so n=1 reproduces the canonical DDQN.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from src.agents.ddqn.agent import DDQNAgent


def compute_nstep_td_target(
    agent: DDQNAgent,
    batch: Dict[str, object],
) -> torch.Tensor:
    """Compute n-step Double DQN TD target (detached, no grad through target).

    Args:
        agent: A ``DDQNAgent`` (online + target networks and config gamma).
        batch: n-step batch dict with keys:
            - discounted_reward_sum: (batch_size,) float32
            - bootstrap_observation: (batch_size, obs_dim) float32
            - bootstrap_steps: (batch_size,) int64
            - terminated_for_bootstrap: (batch_size,) bool

    Returns:
        TD target tensor (batch_size,), detached.
    """
    device = agent.device
    gamma = agent.config.gamma

    reward_sum = torch.as_tensor(
        batch["discounted_reward_sum"], dtype=torch.float32, device=device
    )
    bootstrap_obs = torch.as_tensor(
        batch["bootstrap_observation"], dtype=torch.float32, device=device
    )
    bootstrap_steps = torch.as_tensor(
        batch["bootstrap_steps"], dtype=torch.float32, device=device
    )
    term_for_bootstrap = torch.as_tensor(
        batch["terminated_for_bootstrap"], dtype=torch.float32, device=device
    )

    with torch.no_grad():
        next_q_values = agent.online_network(bootstrap_obs)
        next_actions = next_q_values.argmax(dim=1)

        target_q_values_all = agent.target_network(bootstrap_obs)
        target_q_selected = target_q_values_all.gather(
            dim=1, index=next_actions.unsqueeze(1)
        ).squeeze(1)

        bootstrap_mask = 1.0 - term_for_bootstrap
        gamma_power = torch.pow(gamma, bootstrap_steps)
        td_target = reward_sum + gamma_power * bootstrap_mask * target_q_selected

    return td_target.detach()


def update_nstep(agent: DDQNAgent, batch: Dict[str, object]) -> Dict[str, float]:
    """One gradient update over an n-step batch (online network only).

    Matches the frozen ``DDQNAgent.update`` in everything except the target
    (which uses the n-step formula). Returns the same metric dict.
    """
    td_target = compute_nstep_td_target(agent, batch)

    obs = torch.as_tensor(batch["observation"], dtype=torch.float32, device=agent.device)
    actions = torch.as_tensor(batch["action"], dtype=torch.int64, device=agent.device)

    q_values_all = agent.online_network(obs)
    q_selected = q_values_all.gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)

    if agent.config.use_huber_loss:
        td_error = q_selected - td_target
        td_loss = F.huber_loss(q_selected, td_target, delta=agent.config.huber_delta)
    else:
        td_error = q_selected - td_target
        td_loss = F.mse_loss(q_selected, td_target)
        td_error = td_error.detach()

    agent.optimizer.zero_grad()
    td_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        agent.online_network.parameters(), max_norm=agent.config.gradient_clip
    )
    agent.optimizer.step()
    agent.gradient_update_count += 1

    return {
        "td_loss": float(td_loss.item()),
        "td_error_mean": float(td_error.abs().mean().item()),
        "q_values_mean": float(q_selected.mean().item()),
        "grad_norm": float(grad_norm.item()),
    }