"""E3 Step 6 driver: build + validate frozen seeded-warmup manifests for seeds 6521..6525."""

from __future__ import annotations

import json
from pathlib import Path

from src.milestone10.e3.seeded_warmup import (
    FORMAL_SEEDS,
    E3_OUTPUT_ROOT,
    SEEDED_WARMUP_TOTAL,
    build_seeded_warmup_frames,
    load_seeded_warmup_frames,
    write_seeded_warmup_manifest,
)


def _integrity(frames) -> dict:
    from collections import Counter
    obs_ok = all(len(f.observation_t) == 10 and len(f.next_observation_t) == 10 for f in frames)
    act_ok = all(0 <= f.action_id_t < 16 for f in frames)
    fin_ok = all(__import__("math").isfinite(f.reward_t) for f in frames)
    counts = Counter(f.source_policy for f in frames)
    # successor check per episode
    eps = {}
    for f in frames:
        eps.setdefault(f.episode_id, []).append(f)
    succ = True
    for ep, trs in eps.items():
        ord_ = sorted(trs, key=lambda t: t.step_index)
        for i in range(len(ord_)-1):
            import numpy as np
            if not np.allclose(ord_[i].next_observation_t, ord_[i+1].observation_t, atol=1e-6):
                succ = False
    return {
        "total": len(frames),
        "obs_shape_ok": obs_ok, "action_k2_ok": act_ok, "reward_finite": fin_ok,
        "successor_ok": succ, "counts": dict(counts),
    }


def main() -> None:
    summary = {"all_valid": True, "seeds": {}}
    for seed in FORMAL_SEEDS:
        frames = build_seeded_warmup_frames(seed)
        integ = _integrity(frames)
        paths = write_seeded_warmup_manifest(seed, frames)
        # reload + integrity re-check
        reloaded = load_seeded_warmup_frames(seed)
        ri = _integrity(reloaded)
        valid = (
            len(frames) == SEEDED_WARMUP_TOTAL
            and all(integ[k] is True or isinstance(integ[k], dict) for k in
                    ("obs_shape_ok", "action_k2_ok", "reward_finite", "successor_ok"))
            and integ["counts"].get("exact_myopic", 0) == 1667
            and integ["counts"].get("h2", 0) == 1667
            and integ["counts"].get("exploration", 0) == 1666
            and ri["total"] == SEEDED_WARMUP_TOTAL
        )
        if not valid:
            summary["all_valid"] = False
        print(f"seed {seed}: total={integ['total']} counts={integ['counts']} "
              f"successor={integ['successor_ok']} -> {'VALID' if valid else 'INVALID'}")
        print(f"    {paths['raw']}")
        summary["seeds"][str(seed)] = {"valid": valid, "integrity": integ, "paths": {k: str(v) for k, v in paths.items()}}

    summary_path = E3_OUTPUT_ROOT / "seeded_warmup_manifests" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary: {summary_path}")


if __name__ == "__main__":
    main()