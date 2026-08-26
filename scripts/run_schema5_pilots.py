"""Schema-v5 pilot runner.

Runs two pilots (K=1 and K=2) at exactly 6000 environment steps each,
starting from a clean committed code state.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.agents.ddqn.checkpoint import CHECKPOINT_SCHEMA_VERSION
from src.training.ddqn_trainer import DDQNTrainer, TrainerConfig


def run_pilot(k: int, run_id: str, output_dir: str) -> None:
    print(f"=== Pilot K={k} | run_id={run_id} | output_dir={output_dir}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_bank = "configs/scenarios/m5_pilot_k1.json" if k == 1 else "configs/scenarios/m5_pilot_k2.json"
    val_bank = "configs/scenarios/m5_validation_k1.json" if k == 1 else "configs/scenarios/m5_validation_k2.json"

    cfg = TrainerConfig(
        split="predictor_train",
        validation_split="rl_validation",
        maintenance_capacity=k,
        cost_regime_id="failure-light-no-waste",
        training_scenario_bank_path=train_bank,
        validation_scenario_bank_path=val_bank,
        max_steps=6000,
        warmup_transitions=500,
        batch_size=128,
        update_frequency=1,
        validation_interval=2000,  # 3 validations across 6000 steps
        checkpoint_interval=6000,
        training_seed=6521,
        output_dir=output_dir,
        run_id=run_id,
    )

    trainer = DDQNTrainer(config=cfg)
    t0 = time.time()
    metrics = trainer.train()
    elapsed = time.time() - t0

    final_step = trainer.global_step
    print(f"  trained {final_step} steps in {elapsed:.1f}s")
    print(f"  gradient_updates: {trainer.agent.gradient_update_count}")
    print(f"  episodes: {len(metrics.episode_returns)}")
    print(f"  validations: {len(metrics.validation_results)}")
    if metrics.validation_results:
        last_val = metrics.validation_results[-1]
        print(f"  last validation mean_total_cost: {last_val.get('mean_total_cost'):.3f}")

    # Verify run_manifest and checkpoints exist
    manifest_path = Path(output_dir) / run_id / "run_manifest.json"
    latest_ckpt = Path(output_dir) / run_id / "checkpoint_latest.pt"
    if metrics.validation_results:
        best_ckpt = Path(output_dir) / run_id / "checkpoint_best.pt"
    else:
        best_ckpt = None

    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"  manifest status: {manifest.get('status')}")
    print(f"  schema_version: {manifest.get('checkpoint_schema_version')}")
    print(f"  final_global_step: {manifest.get('final_global_step')}")
    print(f"  git_commit: {manifest.get('git_commit')}")
    print(f"  has checkpoint_latest: {latest_ckpt.exists()}")
    print(f"  has checkpoint_best: {best_ckpt.exists() if best_ckpt else 'N/A (no validation)'}")


if __name__ == "__main__":
    run_pilot(
        k=1,
        run_id="m5_k1_regimelight_seed6521_schema5_6000",
        output_dir="results/m5_schema5_pilots",
    )
    run_pilot(
        k=2,
        run_id="m5_k2_regimelight_seed6521_schema5_6000",
        output_dir="results/m5_schema5_pilots",
    )
