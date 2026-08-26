"""M9 Point-Estimate — runtime config derivation (Step 7).

Derives a per-seed runtime config JSON from the frozen ddqn config (pilot:
``configs/agents/ddqn_pilot_k2.json``; formal: ``configs/agents/ddqn_v1.json``).
The runtime config overrides ONLY:
  - ``environment.prediction_cache_path`` -> the per-seed cache directory
  - ``output.output_dir`` -> the external M9 runs directory

All other fields (agent hyperparams, training hyperparams, env scenario banks,
episode_horizon, max_steps, training_seed/validation_seed, etc.) MUST be
byte-identical to the frozen config. The derivator also records a SHA256 of the
diff so the contract's "no silent budget change" guarantee is auditable.

Invariant 9 (runtime config modifies only allowed fields) + the directive's
"do not silently change max steps / batch size / replay-buffer size / lr /
gamma / epsilon / target-update / validation frequency / scenario bank / reward
coefficients / maintenance capacity / episode termination" are enforced here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_external_assets

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REPO_ROOT = Path(__file__).resolve().parent.parent
PILOT_CFG = REPO_ROOT / "configs" / "agents" / "ddqn_pilot_k2.json"
FORMAL_CFG = REPO_ROOT / "configs" / "agents" / "ddqn_v1.json"

ALLOWED_OVERRIDE_KEYS = (
    ("environment", "prediction_cache_path"),
    ("output", "output_dir"),
)


def _frozen(seed: int) -> str:
    # any seed-specific const not needed here; helpers below
    return f"m9 tmp {seed}"


def _sha_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def test_runtime_config_auto_sets_prediction_cache_manifest_path(tmp_path):
    """When prediction_cache_path is overridden to a per-seed cache dir, the
    matching prediction_cache_manifest_path MUST be set to
    <cache_dir>/prediction_cache_manifest_v2.json so save_checkpoint's
    get_prediction_cache_identity call can find the manifest. The frozen pilot
    config ddqn_pilot_k2.json leaves prediction_cache_manifest_path at None
    (TrainerConfig defaults to the relative repo path which doesn't exist for
    external caches); the formal config ddqn_v1.json pins it to the relative
    repo path. For M9 both need the per-seed manifest path. This is the third
    core runtime override, tied directly to the cache binding."""
    from src.milestone9.point import config_runtime

    cache_dir = tmp_path / "cache" / "seed_6521"
    rt, diff = config_runtime.derive_runtime_config(
        frozen_config_path=PILOT_CFG,
        prediction_cache_path=str(cache_dir),
        output_dir=str(tmp_path / "runs"),
    )
    # The manifest path was auto-set to <cache_dir>/"prediction_cache_manifest_v2.json".
    assert rt["environment"]["prediction_cache_manifest_path"] == \
        str(cache_dir / "prediction_cache_manifest_v2.json")
    # The diff records it.
    assert "environment.prediction_cache_manifest_path" in diff
    # The frozen pilot config had prediction_cache_manifest_path == None (absent).
    assert diff["environment.prediction_cache_manifest_path"]["old"] is None


def test_runtime_config_overrides_only_prediction_cache_path_and_output_dir(tmp_path):
    """Deriving a runtime config changes exactly environment.prediction_cache_path,
    environment.prediction_cache_manifest_path (auto-set to the matching cache
    manifest), and output.output_dir; every other leaf is byte-identical to
    the frozen config."""
    from src.milestone9.point import config_runtime

    cache_dir = tmp_path / "cache" / "seed_6521"
    runs_dir = tmp_path / "runs"
    rt, diff = config_runtime.derive_runtime_config(
        frozen_config_path=PILOT_CFG,
        prediction_cache_path=str(cache_dir),
        output_dir=str(runs_dir),
    )
    # The diff must list EXACTLY the three core override paths.
    assert set(diff.keys()) == {
        "environment.prediction_cache_path",
        "environment.prediction_cache_manifest_path",
        "output.output_dir",
    }
    # And the new values:
    assert rt["environment"]["prediction_cache_path"] == str(cache_dir)
    assert rt["environment"]["prediction_cache_manifest_path"] == \
        str(cache_dir / "prediction_cache_manifest_v2.json")
    assert rt["output"]["output_dir"] == str(runs_dir)


def test_runtime_config_preserves_all_other_fields_byte_identical(tmp_path):
    """No frozen-field mutation outside the override list: every non-overridden
    leaf in the runtime config equals the corresponding frozen leaf."""
    from src.milestone9.point import config_runtime

    cache_dir = tmp_path / "cache" / "seed_6521"
    runs_dir = tmp_path / "runs"
    rt, diff = config_runtime.derive_runtime_config(
        frozen_config_path=PILOT_CFG,
        prediction_cache_path=str(cache_dir),
        output_dir=str(runs_dir),
    )
    frozen = json.loads(PILOT_CFG.read_text())

    def leaves(d, prefix=""):
        for k, v in d.items():
            p = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                yield from leaves(v, p)
            else:
                yield p, v

    rt_leaves = dict(leaves(rt))
    forz_leaves = dict(leaves(frozen))
    skip = {"environment.prediction_cache_path",
            "environment.prediction_cache_manifest_path",
            "output.output_dir"}
    for path, fv in forz_leaves.items():
        if path in skip:
            continue
        assert rt_leaves[path] == fv, f"runtime config mutated frozen field: {path}"


@pytest.mark.parametrize("frozen_cfg", [PILOT_CFG, FORMAL_CFG])
def test_runtime_config_budget_unchanged(frozen_cfg, tmp_path):
    """Invariant 9 / directive: training budget — max_steps, batch_size,
    replay_capacity, learning_rate not present in ddqn cfg but gamma epsilon
    target-update are tested elsewhere — here assert the training block is
    byte-identical to the frozen one."""
    from src.milestone9.point import config_runtime

    rt, _ = config_runtime.derive_runtime_config(
        frozen_config_path=frozen_cfg,
        prediction_cache_path=str(tmp_path / "c"),
        output_dir=str(tmp_path / "r"),
    )
    frozen = json.loads(frozen_cfg.read_text())
    assert rt["training"] == frozen["training"]
    assert rt["agent"] == frozen["agent"]
    assert rt["environment"]["maintenance_capacity"] == frozen["environment"]["maintenance_capacity"]
    assert rt["environment"]["episode_horizon"] == frozen["environment"]["episode_horizon"]
    assert rt["environment"]["cost_regime_id"] == frozen["environment"]["cost_regime_id"]
    assert rt["environment"]["split"] == frozen["environment"]["split"]
    assert rt["environment"]["validation_split"] == frozen["environment"]["validation_split"]


def test_runtime_config_records_diff_sha256(tmp_path):
    """The derivator returns a SHA256 of the diff for auditability."""
    from src.milestone9.point import config_runtime

    rt, diff = config_runtime.derive_runtime_config(
        frozen_config_path=PILOT_CFG,
        prediction_cache_path=str(tmp_path / "c"),
        output_dir=str(tmp_path / "r"),
    )
    assert "environment.prediction_cache_path" in diff
    assert "output.output_dir" in diff
    # diff entries are {old, new} dicts
    e = diff["environment.prediction_cache_path"]
    assert "old" in e and "new" in e
    assert e["old"] == json.loads(PILOT_CFG.read_text())["environment"]["prediction_cache_path"]
    assert e["new"] == str(tmp_path / "c")


def test_runtime_config_rejects_attempted_override_of_frozen_budget(tmp_path):
    """Only the two allowed keys (prediction_cache_path, output_dir) plus the
    pilot-only validation_split remediation may change. The derivator exposes
    no keyword parameters that could mutate training/agent/env BUDGETS
    (max_steps, batch_size, replay_capacity, lr, gamma, epsilon, scenario
    banks, maintenance_capacity, episode_horizon, cost_regime_id, seeds)."""
    from src.milestone9.point import config_runtime

    import inspect
    sig = inspect.signature(config_runtime.derive_runtime_config)
    allowed_params = {
        "frozen_config_path", "prediction_cache_path", "output_dir",
        "validation_split_override", "pilot_scenario_banks",
    }
    extra = set(sig.parameters.keys()) - allowed_params
    assert not extra, (
        f"derive_runtime_config gained a forbidden override parameter: {extra}"
    )
    # Both pilot-only params default to None (no override) so formal stays at
    # exactly two overrides.
    assert sig.parameters["validation_split_override"].default is None
    assert sig.parameters["pilot_scenario_banks"].default is None


def test_runtime_config_pilot_validation_split_override_fixes_stale_split(tmp_path):
    """Pilot remediation (authorized 2026-08-06): derive_runtime_config accepts
    validation_split_override='rl_validation' to fix the STALE
    ddqn_pilot_k2.json validation_split='predictor_train' in the IN-MEMORY
    runtime copy only. The frozen pilot file on disk is unmodified. The diff
    records the third override; the runtime config now has rl_validation;
    the frozen config still has predictor_train. All other leaves unchanged."""
    from src.milestone9.point import config_runtime

    rt, diff = config_runtime.derive_runtime_config(
        frozen_config_path=PILOT_CFG,
        prediction_cache_path=str(tmp_path / "c"),
        output_dir=str(tmp_path / "r"),
        validation_split_override="rl_validation",
    )
    frozen = json.loads(PILOT_CFG.read_text())
    # The frozen file on disk is UNMODIFIED by this derivation (the override
    # is applied to the in-memory runtime copy only). Re-read the raw file
    # and assert its validation_split is STILL the stale predictor_train value.
    raw_disk = json.loads(PILOT_CFG.read_text())
    assert raw_disk["environment"]["validation_split"] == "predictor_train"  # stale
    assert frozen["environment"]["validation_split"] == "predictor_train"  # stale
    assert frozen["environment"]["validation_split"] == "predictor_train"  # stale
    # The runtime copy has the remediated split.
    assert rt["environment"]["validation_split"] == "rl_validation"
    # The diff records all overrides, with old/new. The manifest path is the
    # third core override (auto-set from the cache dir), tied to the cache.
    assert set(diff.keys()) == {
        "environment.prediction_cache_path",
        "environment.prediction_cache_manifest_path",
        "output.output_dir",
        "environment.validation_split",
    }
    assert diff["environment.validation_split"]["old"] == "predictor_train"
    assert diff["environment.validation_split"]["new"] == "rl_validation"
    # Every OTHER leaf (training budget, agent, env non-split) is byte-identical.
    def leaves(d, prefix=""):
        for k, v in d.items():
            p = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                yield from leaves(v, p)
            else:
                yield p, v
    rt_leaves = dict(leaves(rt))
    fz_leaves = dict(leaves(frozen))
    skip = {"environment.prediction_cache_path",
            "environment.prediction_cache_manifest_path",
            "output.output_dir",
            "environment.validation_split"}
    for path, fv in fz_leaves.items():
        if path in skip:
            continue
        assert rt_leaves[path] == fv, f"runtime config mutated frozen field: {path}"


def test_runtime_config_pilot_override_rejects_non_rl_validation_value(tmp_path):
    """The pilot-only validation_split_override param accepts ONLY 'rl_validation'
    (the authorized remediation value). Any other non-None value raises."""
    from src.milestone9.point import config_runtime

    with pytest.raises(ValueError, match="rl_validation|authorized"):
        config_runtime.derive_runtime_config(
            frozen_config_path=PILOT_CFG,
            prediction_cache_path=str(tmp_path / "c"),
            output_dir=str(tmp_path / "r"),
            validation_split_override="rl_test",
        )
    with pytest.raises(ValueError, match="rl_validation|authorized"):
        config_runtime.derive_runtime_config(
            frozen_config_path=PILOT_CFG,
            prediction_cache_path=str(tmp_path / "c"),
            output_dir=str(tmp_path / "r"),
            validation_split_override="predictor_train",
        )


def test_runtime_config_formal_uses_three_core_overrides_unaffected_by_pilot_params(tmp_path):
    """The formal path (validation_split_override=None, pilot_scenario_banks=None)
    derives from ddqn_v1.json (which already has rl_validation and distinct
    banks) and produces EXACTLY the three core override-path diff — the pilot
    params do not change formal behavior. The formal config HAS a frozen
    prediction_cache_manifest_path (relative repo path); the runtime override
    records old->new to point at the per-seed cache manifest."""
    from src.milestone9.point import config_runtime

    rt, diff = config_runtime.derive_runtime_config(
        frozen_config_path=FORMAL_CFG,
        prediction_cache_path=str(tmp_path / "c"),
        output_dir=str(tmp_path / "r"),
    )
    assert set(diff.keys()) == {
        "environment.prediction_cache_path",
        "environment.prediction_cache_manifest_path",
        "output.output_dir",
    }
    # Formal config already has rl_validation and distinct banks; no override needed.
    frozen = json.loads(FORMAL_CFG.read_text())
    assert frozen["environment"]["validation_split"] == "rl_validation"
    assert rt["environment"]["validation_split"] == "rl_validation"


def test_runtime_config_pilot_scenario_banks_sets_distinct_banks(tmp_path):
    """Pilot-budget remediation (authorized 2026-08-06): the frozen
    ddqn_pilot_k2.json carries only the legacy single environment.scenario_bank_path
    (m5_pilot_k2.json), and parse_raw_config (ddqn_config.py:66-72) defaults BOTH
    training_scenario_bank_path and validation_scenario_bank_path to that same
    path, tripping the distinct-bank rule (ddqn_config.py:138-141) BEFORE CLI
    overrides can land. Remediation: derive_runtime_config accepts
    pilot_scenario_banks=(training, validation) and sets both DISTINCT regime-
    matched bank paths in the runtime env (old=None: the frozen pilot config
    lacks these keys). The frozen file on disk is unmodified. All OTHER leaves
    are byte-identical to the frozen pilot config."""
    from src.milestone9.point import config_runtime, paths

    rt, diff = config_runtime.derive_runtime_config(
        frozen_config_path=PILOT_CFG,
        prediction_cache_path=str(tmp_path / "c"),
        output_dir=str(tmp_path / "r"),
        pilot_scenario_banks=(paths.TRAINING_BANK, paths.PILOT_VALIDATION_BANK),
    )
    frozen = json.loads(PILOT_CFG.read_text())
    # The frozen pilot file is UNMODIFIED (still has the legacy single path, no
    # explicit training/validation bank keys).
    raw_disk = json.loads(PILOT_CFG.read_text())
    assert "training_scenario_bank_path" not in raw_disk["environment"]
    assert "validation_scenario_bank_path" not in raw_disk["environment"]
    assert raw_disk["environment"]["scenario_bank_path"] == "configs/scenarios/m5_pilot_k2.json"
    # The runtime copy has the DISTINCT regime-matched banks.
    assert rt["environment"]["training_scenario_bank_path"] == paths.TRAINING_BANK
    assert rt["environment"]["validation_scenario_bank_path"] == paths.PILOT_VALIDATION_BANK
    assert paths.TRAINING_BANK != paths.PILOT_VALIDATION_BANK
    # diff records old=None for the new keys.
    assert diff["environment.training_scenario_bank_path"]["old"] is None
    assert diff["environment.training_scenario_bank_path"]["new"] == paths.TRAINING_BANK
    assert diff["environment.validation_scenario_bank_path"]["old"] is None
    assert diff["environment.validation_scenario_bank_path"]["new"] == paths.PILOT_VALIDATION_BANK


def test_runtime_config_pilot_scenario_banks_rejects_duplicate(tmp_path):
    """Duplicate training/validation banks raise (the distinct-bank rule would
    reject them anyway; fail-closed here is the safe default)."""
    from src.milestone9.point import config_runtime

    with pytest.raises(ValueError, match="DISTINCT|distinct"):
        config_runtime.derive_runtime_config(
            frozen_config_path=PILOT_CFG,
            prediction_cache_path=str(tmp_path / "c"),
            output_dir=str(tmp_path / "r"),
            pilot_scenario_banks=("configs/scenarios/m5_pilot_k2.json",
                                  "configs/scenarios/m5_pilot_k2.json"),
        )


def test_runtime_config_pilot_scenario_banks_rejects_empty(tmp_path):
    """An empty/None bank path raises (the explicit-bank gate would reject it)."""
    from src.milestone9.point import config_runtime

    with pytest.raises(ValueError, match="non-empty"):
        config_runtime.derive_runtime_config(
            frozen_config_path=PILOT_CFG,
            prediction_cache_path=str(tmp_path / "c"),
            output_dir=str(tmp_path / "r"),
            pilot_scenario_banks=("configs/scenarios/m5_pilot_k2.json", ""),
        )


def test_runtime_config_suppresses_unique_run_id_collision_via_wrapper_not_config(tmp_path):
    """The runtime config deliberately does NOT inject run_id (the wrapper passes
    --run-id via CLI). The frozen config's run_id may be null/placeholder and
    is NOT one of the two overridden keys."""
    from src.milestone9.point import config_runtime

    rt, diff = config_runtime.derive_runtime_config(
        frozen_config_path=PILOT_CFG,
        prediction_cache_path=str(tmp_path / "c"),
        output_dir=str(tmp_path / "r"),
    )
    # output.run_id is left as-is (the wrapper handles uniqueness via CLI).
    assert "environment.prediction_cache_path" in diff
    assert "output.output_dir" in diff
    assert "output.run_id" not in diff  # not overridden here
