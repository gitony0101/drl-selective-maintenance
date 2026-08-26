"""E3 (M10) generic episode-aware n-step replay pipeline.

Converts raw ordered environment transitions into n-step replay transitions
(Section 12/13/14/6 of the E3 contract):

- ``R_t^(k) = sum_{i=0}^{k-1} gamma^i r_{t+i}``
- bootstrap state ``s_(t+k)``, bootstrap discount ``gamma^k``
- Double DQN target uses ``argmax_a Q_online(s_(t+k), a)`` evaluated by
  ``Q_target``, masked by ``(1 - terminated_for_bootstrap)``.

Frozen semantics preserved (Section 6):

- ``terminated=True`` stops the bootstrap mask.
- ``truncated=True`` ends the trajectory sequence and prevents n-step
  accumulation from crossing into the next episode, but ``truncated`` alone
  does NOT mask the bootstrap.

For ``n == 1`` every raw transition yields exactly one replay entry with
``discounted_reward_sum = r_t``, ``bootstrap_observation = next_obs_t``,
``bootstrap_steps = 1``, ``terminated_for_bootstrap = terminated_t`` and
``truncated_boundary = truncated_t`` -- numerically identical to the canonical
one-step DDQN transition. An ``NStepReplayBuffer`` storing these entries with
uniform minibatch sampling is provided so sampling stays uniform (no sequential
replay redesign).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class RawStep:
    """One ordered raw environment transition (time t -> t+1) within one episode."""

    observation: np.ndarray
    action_id: int
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class NStepTransition:
    """An aggregated n-step replay transition (one per raw transition, flushed)."""

    observation: np.ndarray      # (obs_dim,) float32 -- state at t
    action_id: int               # action at t
    discounted_reward_sum: float # sum_{i=0}^{k-1} gamma^i r_{t+i}
    bootstrap_observation: np.ndarray  # state reached after bootstrap_steps steps
    bootstrap_steps: int         # k, the actual number of steps accumulated (>= 1)
    terminated_for_bootstrap: bool     # True => mask bootstrap (1 - terminated)
    truncated_boundary: bool           # True => last accumulated step was truncated

    @property
    def gamma_power(self) -> float:
        # Need gamma at compute time; store separately in batch. Keep as field-less helper.
        raise NotImplementedError("gamma is applied at target time via bootstrap_steps")


def build_nstep_transitions(
    ordered_steps: Sequence[RawStep],
    n: int,
    gamma: float,
    *,
    validate_episode_boundary: bool = True,
) -> List[NStepTransition]:
    """Build n-step replay transitions from ordered raw steps of ONE episode.

    Args:
        ordered_steps: Chronological raw steps of a single episode (step_index
            monotonic, no cross-episode linkage).
        n: Effective n-step horizon (n >= 1).
        gamma: Discount factor in (0, 1).

    Returns:
        One ``NStepTransition`` per raw step, in the same order. A trailing window
        that cannot reach ``n`` steps before episode end is flushed with the
        number of steps actually accumulated (>= 1).

    Raises:
        ValueError: if ``n < 1``, ``not (0 < gamma < 1)``, or ordered steps are
            empty.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not (0.0 < gamma < 1.0):
        raise ValueError(f"gamma must be in (0,1), got {gamma}")
    if not ordered_steps:
        raise ValueError("ordered_steps must be non-empty")

    steps = list(ordered_steps)
    total = len(steps)
    out: List[NStepTransition] = []

    for t in range(total):
        base = steps[t]
        # Accumulate rewards over up to n raw steps, stopping early at a
        # terminal or truncated boundary (which both end the usable sequence).
        reward_sum = float(base.reward)
        k = 1
        bootstrap_obs = base.next_observation
        term_for_bootstrap = bool(base.terminated)
        trunc_boundary = bool(base.truncated)

        # If the base step is already terminal or truncated, k stays 1.
        hit_truncated = bool(base.truncated)
        hit_terminated = bool(base.terminated)
        while k < n and not hit_truncated and not hit_terminated:
            j = t + k
            if j >= total:
                # Exact episode end: n-step window runs out of steps.
                # Bootstrap state is the last next_observation; no cross-episode.
                break
            nxt = steps[j]
            reward_sum += (gamma ** k) * float(nxt.reward)
            k += 1
            bootstrap_obs = nxt.next_observation
            term_for_bootstrap = bool(nxt.terminated)
            trunc_boundary = bool(nxt.truncated)
            # Stop accumulating once this step's terminated/truncated ends the
            # usable sequence (a terminal/truncated step is the last of its
            # episode by construction).
            hit_truncated = bool(nxt.truncated)
            hit_terminated = bool(nxt.terminated)

        out.append(
            NStepTransition(
                observation=np.asarray(base.observation, dtype=np.float32),
                action_id=int(base.action_id),
                discounted_reward_sum=reward_sum,
                bootstrap_observation=np.asarray(bootstrap_obs, dtype=np.float32),
                bootstrap_steps=k,
                terminated_for_bootstrap=term_for_bootstrap,
                truncated_boundary=trunc_boundary,
            )
        )

    return out


class EpisodeNStepBuffer:
    """Streaming episode-aware n-step accumulator.

    Keeps raw steps of the CURRENT episode and emits an ``NStepTransition`` for
    the earliest pending raw step exactly when its n-step window is final:

    - ``n`` future raw steps are available (window capped at ``n``), or
    - a ``terminated``/``truncated`` raw step within the window is pushed
      (window cannot extend past it), or
    - the episode is complete (last pushed step is ``terminated``/``truncated``).

    The emitted sequence is IDENTICAL to ``build_nstep_transitions`` over the
    same episode's raw steps. Transitions are emitted incrementally (so replay
    insertion timing matches the canonical stream) without ever crossing an
    episode boundary. ``flush_episode()`` must be called when an episode ends.
    """

    def __init__(self, n: int, gamma: float) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if not (0.0 < gamma < 1.0):
            raise ValueError(f"gamma must be in (0,1), got {gamma}")
        self.n = n
        self.gamma = gamma
        self.pending: List[RawStep] = []

    def push(self, step: RawStep) -> List[NStepTransition]:
        """Append one raw step and return any newly-final n-step transitions."""
        self.pending.append(step)
        emitted: List[NStepTransition] = []
        while self._front_ready():
            # Compute the front transition exactly as the offline builder.
            tr = build_nstep_transitions(self.pending, self.n, self.gamma)[0]
            emitted.append(tr)
            self.pending.pop(0)
        return emitted

    def _front_ready(self) -> bool:
        # The window of pending[0] is final when:
        #  - we hold at least n pending raw steps (window capped at n; future
        #    steps cannot extend it), OR
        #  - any raw step within the first n pending steps is terminal/truncated
        #    (the window cannot extend past it).
        if len(self.pending) >= self.n:
            return True
        for j in range(1, len(self.pending)):
            if (self.pending[j].terminated or self.pending[j].truncated):
                return True
        return False

    def flush_episode(self) -> List[NStepTransition]:
        """Flush all remaining pending raw steps of the current episode."""
        if not self.pending:
            return []
        emitted = build_nstep_transitions(self.pending, self.n, self.gamma)
        self.pending = []
        return emitted


class NStepReplayBuffer:
    """Fixed-capacity ring buffer of ``NStepTransition`` with uniform seeded
    minibatch sampling (no sequential replay). Mirrors the frozen
    ``ReplayBuffer`` sampling discipline."""

    SCHEMA_VERSION = 1

    def __init__(self, capacity: int = 100_000, observation_dim: int = 10,
                 seed: Optional[int] = None) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity
        self.observation_dim = observation_dim

        self.observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.bootstrap_observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.discounted_reward_sums = np.zeros(capacity, dtype=np.float32)
        self.bootstrap_steps = np.ones(capacity, dtype=np.int64)
        self.terminated_for_bootstrap = np.zeros(capacity, dtype=bool)
        self.truncated_boundary = np.zeros(capacity, dtype=bool)

        self.write_index = 0
        self.current_size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.current_size

    def insert(self, tr: NStepTransition) -> None:
        obs = np.asarray(tr.observation, dtype=np.float32).reshape(-1)
        bobs = np.asarray(tr.bootstrap_observation, dtype=np.float32).reshape(-1)
        if obs.shape[0] != self.observation_dim:
            raise ValueError(f"observation shape mismatch: {obs.shape}")
        if bobs.shape[0] != self.observation_dim:
            raise ValueError(f"bootstrap_observation shape mismatch: {bobs.shape}")
        if not np.isfinite(obs).all() or not np.isfinite(bobs).all():
            raise ValueError("non-finite observation")
        if not np.isfinite(tr.discounted_reward_sum):
            raise ValueError("non-finite discounted_reward_sum")
        if tr.bootstrap_steps < 1:
            raise ValueError(f"bootstrap_steps must be >= 1, got {tr.bootstrap_steps}")
        if tr.action_id < 0:
            raise ValueError(f"action_id must be non-negative, got {tr.action_id}")

        idx = self.write_index
        self.observations[idx] = obs
        self.bootstrap_observations[idx] = bobs
        self.actions[idx] = tr.action_id
        self.discounted_reward_sums[idx] = float(tr.discounted_reward_sum)
        self.bootstrap_steps[idx] = int(tr.bootstrap_steps)
        self.terminated_for_bootstrap[idx] = bool(tr.terminated_for_bootstrap)
        self.truncated_boundary[idx] = bool(tr.truncated_boundary)
        self.write_index = (self.write_index + 1) % self.capacity
        if self.current_size < self.capacity:
            self.current_size += 1

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        if self.current_size < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} from buffer of size {self.current_size}"
            )
        indices = self.rng.choice(self.current_size, size=batch_size, replace=False)
        return self._gather(indices)

    def sample_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Uniformly sample a batch dict consumed by the E3 n-step target."""
        return self.sample(batch_size)

    def _gather(self, indices: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "observation": self.observations[indices],
            "action": self.actions[indices],
            "discounted_reward_sum": self.discounted_reward_sums[indices],
            "bootstrap_observation": self.bootstrap_observations[indices],
            "bootstrap_steps": self.bootstrap_steps[indices],
            "terminated_for_bootstrap": self.terminated_for_bootstrap[indices],
            "truncated_boundary": self.truncated_boundary[indices],
        }

    def clear(self) -> None:
        self.write_index = 0
        self.current_size = 0

    def state_dict(self) -> Dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "observation_dim": self.observation_dim,
            "capacity": self.capacity,
            "current_size": self.current_size,
            "write_index": self.write_index,
            "observations": self.observations.copy(),
            "bootstrap_observations": self.bootstrap_observations.copy(),
            "actions": self.actions.copy(),
            "discounted_reward_sums": self.discounted_reward_sums.copy(),
            "bootstrap_steps": self.bootstrap_steps.copy(),
            "terminated_for_bootstrap": self.terminated_for_bootstrap.copy(),
            "truncated_boundary": self.truncated_boundary.copy(),
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if state["schema_version"] != self.SCHEMA_VERSION:
            raise ValueError("n-step replay buffer schema version mismatch")
        if state["capacity"] != self.capacity:
            raise ValueError("n-step replay buffer capacity mismatch")
        if state["observation_dim"] != self.observation_dim:
            raise ValueError("n-step replay buffer observation_dim mismatch")
        for k in ("observations", "bootstrap_observations", "actions",
                  "discounted_reward_sums", "bootstrap_steps",
                  "terminated_for_bootstrap", "truncated_boundary"):
            if k not in state:
                raise ValueError(f"n-step replay state missing '{k}'")
        self.observations[:] = state["observations"]
        self.bootstrap_observations[:] = state["bootstrap_observations"]
        self.actions[:] = state["actions"]
        self.discounted_reward_sums[:] = state["discounted_reward_sums"]
        self.bootstrap_steps[:] = state["bootstrap_steps"]
        self.terminated_for_bootstrap[:] = state["terminated_for_bootstrap"]
        self.truncated_boundary[:] = state["truncated_boundary"]
        self.write_index = state["write_index"]
        self.current_size = state["current_size"]
        self.rng.bit_generator.state = state["rng_state"]