"""M9 point-estimate aggregate report builder -- regression test.

Pins ``scripts/m9_point_aggregate.py``'s paired five-seed report structure
(per-seed DDQN validation, cross-seed baseline stats, paired deltas) on the
EXACTING pilot seed-6521/6522 result manifests (the only completed runs at
test time). The aggregator reads each seed's ``m9_point_result_manifest.json``
+ ``run_manifest.json`` + ``validation_metrics.json`` + ``episode_metrics.csv``
+ ``training_metrics.jsonl`` siblings.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_external_assets
from src.runtime_paths import external_root as _EXTERNAL

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CONTAINER_ROOT = _EXTERNAL()
_PILOT_DIR = CONTAINER_ROOT / "m9_point_runs" / "pilot"

# The pilot's two seeds. The aggregator must read both.
_PILOT_SEEDS = [6521, 6522]


@pytest.mark.skipif(
    not (_PILOT_DIR / "m9_point_mse_control_seed6521" / "m9_point_result_manifest.json").exists(),
    reason="pilot seed-6521 result manifest missing; run the pilot first",
)
def test_aggregate_pilot_builds_paired_report_with_all_sections():
    """aggregate() returns a report with per-seed records, DDQN validation
    cross-seed stats, cross-seed baseline stats, and paired deltas."""
    from scripts.m9_point_aggregate import aggregate, _CATASTROPHIC_COST_THRESHOLD

    report = aggregate("pilot", _PILOT_SEEDS)

    assert report["phase"] == "pilot"
    assert report["n_seeds"] == 2
    assert set(report["seeds"]) == {6521, 6522}
    assert set(report["per_seed"].keys()) == {6521, 6522}

    # Per-seed: DDQN validation + baselines + provenance.
    for s in _PILOT_SEEDS:
        rec = report["per_seed"][s]
        assert "ddqn_validation" in rec
        assert rec["ddqn_validation"]["mean_total_cost"] == 30.0
        assert rec["ddqn_validation"]["total_pm_actions"] == 0
        assert rec["ddqn_validation"]["num_episodes"] == 5
        assert "baselines" in rec
        assert set(rec["baselines"].keys()) >= {
            "corrective_only", "random_feasible", "age_threshold",
            "predicted_rul_threshold", "greedy_predicted_rul", "exact_myopic",
        }
        assert "training_provenance" in rec
        assert rec["training_provenance"]["status"] == "COMPLETE"

    # Cross-seed DDQN stats: mean/median/std/min/max/n.
    assert report["ddqn_validation_stats"]["mean_total_cost"]["n"] == 2
    assert report["ddqn_validation_stats"]["mean_total_cost"]["mean"] == 30.0

    # Cross-seed baseline stats (one entry per family).
    assert set(report["baseline_stats"].keys()) >= {
        "corrective_only", "random_feasible", "age_threshold",
        "predicted_rul_threshold", "greedy_predicted_rul", "exact_myopic",
    }
    # CorrectiveOnly and exact_myopic have zero variance across pilot seeds.
    assert report["baseline_stats"]["corrective_only"]["mean"] == 54.0
    assert report["baseline_stats"]["exact_myopic"]["mean"] == 54.0

    # Paired deltas (DDQN - baseline): one entry per family, with stats.
    deltas = report["paired_deltas_ddqn_minus_baseline"]
    assert set(deltas.keys()) == set(report["baseline_stats"].keys())
    # DDQN (30.0) is WORSE than corrective_only (54.0) -> delta = -24.0.
    assert deltas["corrective_only"]["mean"] == -24.00


def test_aggregate_pilot_writes_report_json(tmp_path, monkeypatch):
    """main() writes aggregate_report.json to <runs_root>/<phase>/."""
    from scripts import m9_point_aggregate as agg

    monkeypatch.setattr(agg, "_RUNS_ROOT", tmp_path)
    # Point the per-seed loader at the real pilot runs.
    real_pilot = _PILOT_DIR
    monkeypatch.setattr(agg, "_RUNS_ROOT", real_pilot)

    out_path = real_pilot / "aggregate_report.json"
    if out_path.exists():
        out_path.unlink()

    import subprocess
    rc = subprocess.run(
        [sys.executable, str(Path(agg.__file__).resolve()),
         "--phase", "pilot", "--seeds", "6521,6522"],
        capture_output=True, text=True, check=False,
    )
    assert rc.returncode == 0, rc.stderr
    assert out_path.exists()
    report = json.loads(out_path.read_text())
    assert report["phase"] == "pilot"
