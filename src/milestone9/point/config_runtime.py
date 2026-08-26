"""M9 Point-Estimate — per-seed runtime config derivation.

Derives a runtime agent config JSON from a frozen ddqn config (pilot:
``configs/agents/ddqn_pilot_k2.json``; formal: ``configs/agents/ddqn_v1.json``)
overriding EXACTLY two paths:
  - ``environment.prediction_cache_path`` -> per-seed cache directory
  - ``output.output_dir`` -> external M9 runs directory

Every other field — agent hyperparams (gamma, epsilon, lr, target-update,
gradient-clip), training hyperparams (max_steps, batch_size, replay_capacity,
warmup_transitions, validation_interval, checkpoint_interval, update_frequency),
environment (maintenance_capacity, episode_horizon, cost_regime_id, split,
validation_split, scenario banks), training_seed/validation_seed — is preserved
byte-identical to the frozen config. The derivator has no keyword parameters
beyond the two override paths, so a caller cannot accidentally mutate any
budget or selection field.

Returned with a structured ``diff`` dict (only the two overridden keys, each as
``{"old": ..., "new": ...}``) for auditability.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


_FORBIDDEN_SPLIT = "rl_test"

_ALLOWED_OVERRIDES = {
    ("environment", "prediction_cache_path"),
    ("output", "output_dir"),
}

# PILOT-ONLY override: the frozen pilot budget source
# ``configs/agents/ddqn_pilot_k2.json`` carries ``validation_split:
# "predictor_train"``, a value frozen at d98cb41 (2026-07-22 06:40) BEFORE the
# M5 trainer's hard ``validation_split == "rl_validation"`` rule was added
# 16h later at 3a28837. No commit ever reconciled the pilot config, and it was
# never run to completion against the post-3a28837 trainer
# (``results/milestone5/pilot_k2/`` is absent). It cannot be rescued by the
# ``--validation-split`` CLI flag because the resolver runs
# ``load_and_validate_config`` (config-level validation) BEFORE applying CLI
# overrides (``resolver.py:360-363``), so the stale value raises before the
# override lands. The remediation — authorized 2026-08-06 — is to fix
# ``validation_split`` in the IN-MEMORY runtime copy ONLY (the frozen pilot
# file on disk is byte-identical/unmodified). This is applied ONLY for the
# pilot path; the formal path (``ddqn_v1.json``) already has the correct
# ``rl_validation`` split and uses exactly the two-path override set above.
_PILOT_VALIDATION_SPLIT_OVERRIDE = "rl_validation"


def derive_runtime_config(
    frozen_config_path: Path,
    prediction_cache_path: str,
    output_dir: str,
    validation_split_override: Optional[str] = None,
    pilot_scenario_banks: Optional[Tuple[str, str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Derive a runtime config from the frozen ddqn config.

    Overrides applied to the runtime config (in-memory copy ONLY; the frozen
    file on disk is byte-identical/unmodified):
      - ``environment.prediction_cache_path`` -> per-seed cache directory
      - ``output.output_dir`` -> external M9 runs directory

    Pilot-budget remediation overrides (applied ONLY when the caller passes
    them — the formal path's defaults keep them None):
      - ``validation_split_override``: when ``"rl_validation"``, sets
        ``environment.validation_split`` -> ``rl_validation``. Remediation for
        the stale ``ddqn_pilot_k2.json`` validation_split='predictor_train'
        (frozen before the trainer's hard rl_validation rule). The CLI
        ``--validation-split`` flag cannot rescue it because the resolver
        runs config-level validation BEFORE applying CLI overrides
        (resolver.py:360-363).
      - ``pilot_scenario_banks``: a ``(training_bank, validation_bank)`` tuple
        setting ``environment.training_scenario_bank_path`` and
        ``environment.validation_scenario_bank_path`` to DISTINCT regime-matched
        paths. Remediation for the legacy single
        ``ddqn_pilot_k2.json:environment.scenario_bank_path`` —
        ``parse_raw_config`` (ddqn_config.py:66-72) defaults both banks to the
        same legacy path, tripping the distinct-bank rule
        (ddqn_config.py:138-141) BEFORE CLI overrides land. The wrapper still
        passes ``--training-scenario-bank`` and ``--validation-scenario-bank``
        as explicit CLI flags (the gate mandates them regardless); the runtime
        override makes the config-level validation pass.

    No OTHER field — agent hyperparams, training hyperparams (max_steps,
    batch_size, warmup_transitions, replay_capacity, validation_interval,
    checkpoint_interval, update_frequency, epsilon_decay_steps), environment
    (maintenance_capacity, episode_horizon, cost_regime_id, split,
    training_seed/validation_seed) — is mutated. The derivator exposes no
    keyword parameters beyond the override paths + the two pilot-budget
    remediations, so a caller cannot accidentally mutate any budget or
    selection field.

    Raises ``ValueError`` for an unauthorized ``validation_split_override``
    value (only ``"rl_validation"`` permitted), for an rl_test value, or if
    ``pilot_scenario_banks`` carries a None/empty/duplicate bank or a bank
    equal to rl_test sentinel. Returns ``(runtime_config, diff)`` where
    ``diff`` maps dotted override paths to ``{"old", "new"}`` dicts; a bank
    absent from the frozen config records ``old=None``.
    """
    frozen = json.loads(Path(frozen_config_path).read_text())
    runtime = copy.deepcopy(frozen)

    # The manifest path is bound to the cache dir: the overridden
    # prediction_cache_path points at a per-seed cache dir whose
    # ``prediction_cache_manifest_v2.json`` is the cache-identity file the
    # generator writes (cache_prep._read_manifest_checkpoint_sha256) and the
    # trainer reads at save_checkpoint time
    # (ddqn_trainer.py: get_prediction_cache_identity(config.prediction_cache_manifest_path)).
    # The frozen pilot config leaves ``prediction_cache_manifest_path`` absent
    # (TrainerConfig defaults to the relative repo path
    # ``data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json``,
    # ddqn_config.py:67 — non-existent for an external per-seed cache); the
    # formal config pins that same relative repo path. For an M9 external
    # per-seed cache, BOTH must point at the per-seed manifest. Tie the manifest
    # path directly to the cache dir so the trainer can find it — this is the
    # third core runtime override, not a budget/metric/selection field.
    cache_manifest_path = f"{prediction_cache_path.rstrip('/')}/prediction_cache_manifest_v2.json"

    overrides: Dict[Tuple[str, str], Any] = {
        ("environment", "prediction_cache_path"): prediction_cache_path,
        ("environment", "prediction_cache_manifest_path"): cache_manifest_path,
        ("output", "output_dir"): output_dir,
    }
    if validation_split_override is not None:
        if validation_split_override != _PILOT_VALIDATION_SPLIT_OVERRIDE:
            raise ValueError(
                "M9 config_runtime: the only authorized validation_split "
                "override is 'rl_validation' (pilot-budget remediation for "
                "the stale ddqn_pilot_k2.json); got "
                f"{validation_split_override!r}."
            )
        if validation_split_override == _FORBIDDEN_SPLIT:
            raise ValueError(
                "M9 config_runtime: refusing to set validation_split to the "
                "forbidden rl_test split."
            )
        overrides[("environment", "validation_split")] = validation_split_override

    if pilot_scenario_banks is not None:
        training_bank, validation_bank = pilot_scenario_banks
        if not training_bank or not validation_bank:
            raise ValueError(
                "M9 config_runtime: pilot_scenario_banks must be a non-empty "
                f"(training_bank, validation_bank) tuple; got {pilot_scenario_banks!r}."
            )
        if training_bank == validation_bank:
            raise ValueError(
                "M9 config_runtime: pilot_scenario_banks must be DISTINCT "
                "training/validation banks (the trainer's distinct-bank rule "
                "at ddqn_config.py:138-141 rejects identical banks)."
            )
        overrides[("environment", "training_scenario_bank_path")] = training_bank
        overrides[("environment", "validation_scenario_bank_path")] = validation_bank

    diff: Dict[str, Dict[str, Any]] = {}
    for (section, key), new_val in overrides.items():
        if section not in runtime or not isinstance(runtime[section], dict):
            raise ValueError(
                f"frozen config has no '{section}' section to override "
                f"({frozen_config_path})"
            )
        old_val = runtime[section].get(key)
        runtime[section][key] = new_val
        diff[f"{section}.{key}"] = {"old": old_val, "new": new_val}

    # Defensive: assert no other key drifted (deep equality on the complement).
    _assert_only_overrides_changed(frozen, runtime, diff)
    return runtime, diff


def _assert_only_overrides_changed(
    frozen: Dict, runtime: Dict, diff: Dict
) -> None:
    def leaves(d, prefix=""):
        for k, v in d.items():
            p = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                yield from leaves(v, p)
            else:
                yield p, v

    f_leaves = dict(leaves(frozen))
    r_leaves = dict(leaves(runtime))
    overridden = set(diff.keys())
    drifted = []
    for p, fv in f_leaves.items():
        if p in overridden:
            continue
        if r_leaves.get(p) != fv:
            drifted.append(p)
    if drifted:
        raise ValueError(
            f"runtime config drifted from frozen on non-override fields: {drifted}"
        )
