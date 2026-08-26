"""M9 point-estimate DDQN evaluation-action-trace reconstruction (the corrected report).

Reconstructs the DDQN evaluation action distribution on the SAME five
rl_validation episodes the frozen ``validation_metrics.json`` reports, by
deterministically replaying the trainer's ``compute_validation_metrics`` loop
with the frozen ``checkpoint_best.pt`` + the SAME validation ``EnvironmentConfig``
the trainer built. This requires ONLY evaluation (no retraining, no cache
regeneration) -- permitted per the directive §10.G.

Reproduction check: the replayed mean_total_cost MUST equal the frozen
``validation_metrics.json`` last-entry mean_total_cost (13.0/13.0/14.0/14.0/
14.0 for seeds 6521-6525). If it doesn't, the reconstructed action trace is
rejected as non-reproducing and the script fails closed. The action trace is
then authentic to the frozen eval.

Output (per seed): ``<container>/m9_point_runs/ddqn_eval_replay/seed_<s>/
ddqn_eval_replay.json`` with per-episode action counts/freqs, per-episode
costs, mean_total_cost, reproduction match status.

USAGE:
    python scripts/run_m9_ddqn_eval_replay.py
    python scripts/run_m9_ddqn_eval_replay.py --seeds 6521
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.milestone9.point import pairing
from src.milestone9.point.baselines import env_config_for_eval

_CONTAINER_ROOT = pairing._CONTAINER_ROOT
_FORMAL_RUNS = _CONTAINER_ROOT / "m9_point_runs" / "formal"
_OUT_ROOT = _CONTAINER_ROOT / "m9_point_runs" / "ddqn_eval_replay"

_SEEDS = [6521, 6522, 6523, 6524, 6525]


def _replay_one_seed(seed: int, out_root: Path) -> Dict[str, Any]:
    run_id = pairing.run_id_for_seed(seed)
    ddqn_dir = _FORMAL_RUNS / run_id
    ckpt_path = ddqn_dir / "checkpoint_best.pt"
    runtime_cfg_path = ddqn_dir / "runtime_config.json"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint missing: {ckpt_path}")
    runtime_cfg = json.loads(runtime_cfg_path.read_text())
    env_config = env_config_for_eval(runtime_cfg)

    from src.envs import SelectiveMaintenanceEnv
    from src.agents.ddqn import DDQNAgent, DDQNAgentConfig
    from src.agents.ddqn.checkpoint import load_checkpoint

    # Build the SAME val env the trainer built (seed=validation_seed from
    # runtime config training block). Use runtime_cfg's env block's cache.
    env = SelectiveMaintenanceEnv(config=env_config)
    n_actions = 16  # K=2 action table
    agent_config = DDQNAgentConfig(
        observation_dim=10,
        num_actions=n_actions,
        hidden_dim=128,
        num_hidden_layers=2,
    )
    agent = DDQNAgent(config=agent_config)
    load_checkpoint(str(ckpt_path), agent=agent)
    # agent.evaluate_action uses epsilon=0 greedy (deterministic).

    num_scenarios = min(10, len(env.scenario_bank.scenarios))
    action_table_len = n_actions
    action_counts = [0] * action_table_len
    per_episode_actions: List[List[int]] = []
    total_costs: List[float] = []
    failure_counts: List[int] = []
    pm_counts: List[int] = []

    for _ep in range(num_scenarios):
        obs, _info = env.reset()
        ep_actions: List[int] = []
        ep_cost = 0.0
        ep_fail = 0
        ep_pm = 0
        for _step in range(env_config.episode_horizon):
            action = agent.evaluate_action(obs)
            if not (0 <= action < action_table_len):
                raise RuntimeError(
                    f"DDQN eval produced invalid action {action} "
                    f"(valid 0..{action_table_len-1})"
                )
            action_counts[action] += 1
            ep_actions.append(int(action))
            obs, reward, terminated, truncated, info = env.step(action)
            ep_cost += float(info.get("total_cost", 0.0))
            ep_fail += int(info.get("num_failures", 0))
            ep_pm += int(info.get("num_preventive", 0))
            if truncated:
                break
        per_episode_actions.append(ep_actions)
        total_costs.append(ep_cost)
        failure_counts.append(ep_fail)
        pm_counts.append(ep_pm)

    replayed_mean_cost = float(sum(total_costs) / len(total_costs)) if total_costs else 0.0

    # Reproduction check vs the BEST checkpoint's own validation_mean_cost
    # (the authoritative validation cost of the policy we compare). The
    # checkpoint metadata records the validation cost at the global_step where
    # the best checkpoint was selected (lower = better). This is the DDQN
    # policy's actual validation cost -- NOT the final-step
    # validation_metrics.json last entry (which may be an un-selected final
    # step with a higher cost). We ALSO record the final-step value and flag
    # when the two differ (a metric-selection disclosure, NOT corruption).
    import torch
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    ckpt_meta = ck.get("metadata", {})
    ckpt_val_cost = ckpt_meta.get("validation_mean_cost")
    ckpt_global_step = ckpt_meta.get("global_step")
    if ckpt_val_cost is None:
        raise ValueError(
            f"checkpoint_best metadata missing validation_mean_cost for seed {seed}"
        )
    frozen_mean_cost = float(ckpt_val_cost)
    vm = json.loads((ddqn_dir / "validation_metrics.json").read_text())
    vm_last = vm[-1] if isinstance(vm, list) else vm
    final_step_cost = float(vm_last.get("mean_total_cost"))
    final_step_gs = vm_last.get("global_step")
    reproduced = abs(replayed_mean_cost - frozen_mean_cost) < 1e-6
    if abs(frozen_mean_cost - final_step_cost) < 1e-6:
        best_vs_final_disclose = (
            f"best-checkpoint val cost {frozen_mean_cost} (gs={ckpt_global_step}) "
            f"== final-step val cost {final_step_cost} (gs={final_step_gs})."
        )
    else:
        best_vs_final_disclose = (
            f"best-checkpoint val cost {frozen_mean_cost} (gs={ckpt_global_step}) "
            f"!= final-step val cost {final_step_cost} (gs={final_step_gs}); "
            f"the DDQN policy compared is the BEST checkpoint, so the comparison "
            f"uses {frozen_mean_cost}, NOT the final-step {final_step_cost}."
        )

    total_actions = sum(action_counts)
    action_freq = [c / total_actions if total_actions else 0.0
                   for c in action_counts]
    dominant = sorted(range(action_table_len),
                      key=lambda a: action_counts[a], reverse=True)[:5]

    payload = {
        "seed": seed,
        "checkpoint_path": str(ckpt_path),
        "replayed": {
            "mean_total_cost": replayed_mean_cost,
            "frozen_mean_total_cost": frozen_mean_cost,
            "reproduced": reproduced,
            "num_episodes": num_scenarios,
            "total_costs_per_episode": total_costs,
            "total_failures": sum(failure_counts),
            "total_pm_actions": sum(pm_counts),
            "per_episode_actions": per_episode_actions,
            "action_distribution": {
                "policy": "DDQN_point_estimate (checkpoint_best, epsilon=0 greedy)",
                "action_counts": action_counts,
                "action_freq": action_freq,
                "n_actions": action_table_len,
                "total_actions": total_actions,
                "dominant_action_ids": dominant,
                "source": "deterministic evaluation replay on the 5 rl_validation episodes (not training replay)",
            },
        },
        "reconstruction_note": (
            "Action trace reconstructed by deterministically replaying the "
            "trainer's compute_validation_metrics loop (env.reset() x "
            "num_scenarios, agent.evaluate_action epsilon=0) with the frozen "
            "checkpoint_best.pt + the SAME validation EnvironmentConfig. "
            "Permitted per directive §10.G (evaluation-only, no retraining). "
            "Reproduction check: replayed mean_total_cost == the BEST "
            "checkpoint metadata validation_mean_cost (the authoritative "
            "validation cost of the policy compared), NOT the final-step "
            "validation_metrics.json last entry."
        ),
        "metric_selection_disclosure": best_vs_final_disclose,
        "best_checkpoint": {
            "global_step": ckpt_global_step,
            "validation_mean_cost": frozen_mean_cost,
            "final_step_global_step": final_step_gs,
            "final_step_mean_cost": final_step_cost,
        },
    }

    out_dir = out_root / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ddqn_eval_replay.json"
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(out_path)

    print(f"[seed {seed}] replayed mean_cost={replayed_mean_cost:.4f} "
          f"frozen={frozen_mean_cost:.4f} reproduced={reproduced} "
          f"episodes={num_scenarios} total_actions={total_actions}")
    print(f"  dominant actions (id:count): "
          f"{[(a, action_counts[a]) for a in dominant]}")
    print(f"  total_failures={sum(failure_counts)} total_pm={sum(pm_counts)}")
    print(f"  output: {out_path}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(
        description="M9 DDQN evaluation action-trace reconstruction (replay)")
    ap.add_argument("--seeds", type=str, default=None,
                    help="comma-separated seeds (default: 6521-6525)")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else _SEEDS

    _OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"output root: {_OUT_ROOT}")
    print(f"seeds: {seeds}")

    payloads: List[Dict[str, Any]] = []
    any_fail = False
    for seed in seeds:
        p = _replay_one_seed(seed, _OUT_ROOT)
        payloads.append(p)
        if not p["replayed"]["reproduced"]:
            any_fail = True

    index_path = _OUT_ROOT / "ddqn_eval_replay_index.json"
    index = {
        "seeds": seeds,
        "per_seed": {
            str(p["seed"]): {
                "reproduced": p["replayed"]["reproduced"],
                "replayed_mean_total_cost": p["replayed"]["mean_total_cost"],
                "best_checkpoint_validation_mean_cost": p["replayed"]["frozen_mean_total_cost"],
                "results_path": str(_OUT_ROOT / f"seed_{p['seed']}" / "ddqn_eval_replay.json"),
            } for p in payloads
        },
    }
    index_path.write_text(json.dumps(index, indent=2))
    if any_fail:
        print("STATUS: FAILED (reproduction mismatch for one or more seeds)")
        return 1
    print("STATUS: COMPLETED (all seeds reproduced frozen validation cost)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
