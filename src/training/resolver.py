"""Shared authoritative command-to-effective-config resolver (M5 M5 provenance).

This is the SINGLE production resolver that turns an ``exact_training_command``
into the same effective ``TrainerConfig`` that ``scripts/train_ddqn.py`` would
construct at launch.  It exists to kill the synthetic-config-drift blocker that
escaped M5 reproducibility: the formal matrix generator hand-built a TrainerConfig that
could differ from the command actually executed.

It is used by:

  - ``scripts/train_ddqn.py`` (training CLI -- via the same override keys)
  - ``scripts/generate_m5_formal_matrix.py`` (matrix generation)
  - ``scripts/run_m5_smoke.py`` (smoke launch preparation)
  - ``src/training/preflight`` command-level preflight
  - ``tests`` (binding tests)

Contract:

    matrix row effective config
    == exact command effective config
    == --validate-only effective config
    == Trainer construction effective config

The resolver produces a ``ResolvedCommand`` containing the effective
``TrainerConfig`` and the effective identity dict (``config.to_dict()`` +
``num_actions``).  It applies NO training side effects and creates no output
directories.

FROZEN DECISION -- "Always require explicit banks":
  The mandatory explicit-bank gate lives HERE, inside the shared authoritative
  resolver, not only inside ``train_ddqn.py main()``.  Every entry point that
  resolves a command -- dry-run, validate-only, smoke, normal training, and
  resume -- therefore goes through ONE gate.  There is NO opt-out flag: the
  previous ``--allow-baseline-banks`` bypass is removed entirely.  A command
  that omits BOTH ``--training-scenario-bank <path>`` AND
  ``--validation-scenario-bank <path>`` fails closed with ``ExplicitBankError``
  before any TrainerConfig is materialised.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.training.ddqn_config import load_and_validate_config, apply_cli_overrides
from src.training.ddqn_trainer import TrainerConfig


class ExplicitBankError(ValueError):
    """Raised by the resolver when a command omits the mandatory explicit
    regime-specific scenario-bank flags.

    This is the fail-closed signal for the frozen "always require explicit
    banks" decision.  Callers translate it into a non-zero exit / failed
    preflight; they MUST NOT bypass it.
    """


# CLI flags this resolver recognises.  Kept in sync with
# ``scripts/train_ddqn.py``.  Flags that map to a TrainerConfig field use the
# TrainerConfig field name; the two bank flags are part of the binding contract.
SINGLE_VALUE_FLAGS: Dict[str, str] = {
    "--config": "_config",
    "--split": "split",
    "--validation-split": "validation_split",
    "--k-capacity": "maintenance_capacity",
    "--cost-regime": "cost_regime_id",
    "--training-seed": "training_seed",
    "--max-steps": "max_steps",
    "--output-dir": "output_dir",
    "--run-id": "run_id",
    "--device": "device",
    "--training-scenario-bank": "training_scenario_bank_path",
    "--validation-scenario-bank": "validation_scenario_bank_path",
}
BOOL_FLAGS: frozenset = frozenset({"--validate-only", "--dry-run", "--help"})

# The two flags whose joint presence is mandated by the frozen explicit-bank
# decision.  Both must be present for dry-run / validate-only / smoke / normal
# training / resume.  There is no bypass.
REQUIRED_BANK_FLAGS: Tuple[str, str] = (
    "--training-scenario-bank",
    "--validation-scenario-bank",
)


class ResolvedCommand:
    """Effective resolved production config for one command."""

    __slots__ = ("trainer_config", "config_path", "config_dict")

    def __init__(self, trainer_config: TrainerConfig, config_path: str,
                 config_dict: Dict[str, Any]):
        self.trainer_config = trainer_config
        self.config_path = config_path
        self.config_dict = config_dict

    @property
    def effective_identity_dict(self) -> Dict[str, Any]:
        """Dict over which ``resolved_config_identity`` is computed."""
        d = {**self.trainer_config.to_dict(), "num_actions": self.trainer_config.num_actions}
        return d

    @property
    def prediction_cache_split(self) -> str:
        """The scalar ``prediction_cache_split`` provenance field.

        The schema-6 trainer sets ``prediction_cache_split`` to the resolved
        ``validation_split`` at save time (see
        ``src/agents/ddqn/checkpoint.py:1059``).  We derive the same scalar from
        the resolved effective config here so that any caller that audits
        provenance (smoke driver, matrix preflight, tests) obtains the exact
        value the checkpoint would record -- NOT a value recovered from the
        manifest (a single cache manifest spans all splits).  Failing closed
        on provenance therefore starts from a resolver-derived split, not a
        path-only identity.
        """
        return self.trainer_config.validation_split

    def __getattr__(self, name: str) -> Any:
        # Convenience: expose TrainerConfig fields directly.
        return getattr(self.trainer_config, name)


def parse_command(tokens: List[str]) -> Tuple[str, Dict[str, Any], set, set]:
    """Parse a command token list into (config_path, overrides, flags_set, banks_set).

    ``banks_set`` is the subset of ``REQUIRED_BANK_FLAGS`` that appeared in the
    command, so the resolver's mandatory explicit-bank gate can inspect it.

    Tokens may include the leading ``python scripts/train_ddqn.py`` preamble;
    those tokens are ignored.  Unknown flags are collected but not raised
    here so callers can decide how strict to be.
    """
    overrides: Dict[str, Any] = {}
    flags: set = set()
    banks: set = set()
    config_path: Optional[str] = None
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in SINGLE_VALUE_FLAGS:
            field = SINGLE_VALUE_FLAGS[tok]
            if i + 1 >= len(tokens):
                raise ValueError(f"flag {tok} missing value")
            raw = tokens[i + 1]
            if field == "_config":
                config_path = raw
            else:
                overrides[field] = _coerce(field, raw)
            if tok in REQUIRED_BANK_FLAGS:
                banks.add(tok)
            i += 2
            continue
        if tok in BOOL_FLAGS:
            flags.add(tok)
            i += 1
            continue
        # Unknown single token (e.g. python, scripts/train_ddqn.py, a value
        # whose flag we already consumed).  Skip.
        i += 1
    if config_path is None:
        raise ValueError("command missing --config <path>")
    return config_path, overrides, flags, banks


def _coerce(field: str, raw: str) -> Any:
    if field == "maintenance_capacity":
        return int(raw)
    if field == "training_seed":
        return int(raw)
    if field == "max_steps":
        return int(raw)
    return raw


def _tokenize(command_or_tokens) -> List[str]:
    """Normalise a command argument into a token list.

    Accepts either an already-tokenised list of strings or a single command
    string.  A command string is split with :func:`shlex.split` (NOT
    ``str.split``), which correctly honours quoted paths containing spaces --
    the exact reason a naive ``command.split()`` is forbidden here.  A leading
    ``python scripts/train_ddqn.py`` preamble is preserved (it is ignored by
    the parser).
    """
    if isinstance(command_or_tokens, (list, tuple)):
        return list(command_or_tokens)
    return shlex.split(command_or_tokens)


def assert_explicit_banks(banks_present: set, *, source: str = "command") -> None:
    """The SINGLE mandatory explicit-bank gate.

    FROZEN DECISION -- "Always require explicit banks": every formal launch
    (dry-run, validate-only, smoke, normal training, resume) MUST pass BOTH
    ``--training-scenario-bank <path>`` and ``--validation-scenario-bank
    <path>``.  The base agent configs only point at the baseline
    light-no-waste banks, so a command that omits these flags would silently
    train/validate against the wrong regime -- the exact blocker this binding
    closes.  There is NO opt-out: the previous ``--allow-baseline-banks``
    bypass is removed entirely.

    Raises :class:`ExplicitBankError` (a ``ValueError``) when either flag is
    missing.  This is the shared authoritative gate; ``train_ddqn.py`` and
    every other entry point route through it so there is exactly one place
    where the decision is enforced.
    """
    missing = [f for f in REQUIRED_BANK_FLAGS if f not in banks_present]
    if missing:
        raise ExplicitBankError(
            f"explicit-bank gate FAILED for {source}: the mandatory regime-"
            f"specific scenario-bank flags are missing ({', '.join(missing)}). "
            f"Pass BOTH --training-scenario-bank <path> and "
            f"--validation-scenario-bank <path> so the effective config matches "
            f"the matrix row exactly.  The base agent config only points at "
            f"the baseline light-no-waste banks, so omitting these flags would "
            f"silently bind to the wrong regime.  There is no bypass."
        )


def resolve_command_to_effective(command_or_tokens,
                                 cwd: Optional[Path | str] = None) -> ResolvedCommand:
    """Resolve an exact training command to its effective TrainerConfig.

    This is the SINGLE authoritative resolver.  It mirrors exactly what
    ``scripts/train_ddqn.py`` does: ``load_and_validate_config`` then
    ``apply_cli_overrides`` over the CLI flag overrides, then the mandatory
    explicit-bank gate.  It performs no training side effects and creates no
    output directories.

    Args:
        command_or_tokens: Either a command string (split with
            :func:`shlex.split`) or a pre-tokenised list of strings.  A
            leading ``python scripts/train_ddqn.py`` preamble is allowed and
            ignored.
        cwd: Used only to resolve a relative ``--config`` path against the
            repository root when the path is not absolute and the file does
            not exist relative to the process CWD.  Typically the repo root.

    Returns:
        ResolvedCommand with the effective TrainerConfig and identity dict.

    Raises:
        ExplicitBankError: if the command omits a required bank flag.  There
            is no bypass: the frozen "always require explicit banks" decision
            is enforced here for every caller.
    """
    tokens = _tokenize(command_or_tokens)
    config_path, overrides, _flags, banks = parse_command(tokens)

    # MANDATORY GATE: enforce the frozen "always require explicit banks"
    # decision in the shared authoritative layer.  No bypass in production.
    assert_explicit_banks(banks, source="exact command")

    # Resolve config path against cwd (repo root) for robustness.
    p = Path(config_path)
    if not p.is_absolute() and not p.exists() and cwd is not None:
        alt = Path(cwd) / config_path
        if alt.exists():
            config_path = str(alt)

    parsed = load_and_validate_config(config_path, mode="training")
    tc = parsed.trainer_config
    if overrides:
        tc = apply_cli_overrides(tc, overrides)

    identity_dict = {**tc.to_dict(), "num_actions": tc.num_actions}
    return ResolvedCommand(tc, config_path, identity_dict)


def derive_prediction_cache_provenance(
        resolved: ResolvedCommand,
        *,
        manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute the prediction-cache provenance dict for a resolved command.

    Derives the scalar ``prediction_cache_split`` from the resolved effective
    config's ``validation_split`` (the exact value the schema-6 trainer writes
    at save time) and the manifest-derived fields from the on-disk prediction
    cache manifest.  This is the READ-ONLY form of the provenance the
    checkpoint records; auditors compare these recomputed values against the
    checkpoint's stored metadata and fail-closed on any mismatch.

    Args:
        resolved: A ResolvedCommand from resolve_command_to_effective.
        manifest_path: Path to the prediction cache manifest.  Defaults to
            the resolved config's ``prediction_cache_manifest_path``.

    Returns:
        Dict containing ``prediction_cache_split`` plus every
        manifest-derived provenance field.

    Raises:
        FileNotFoundError / ValueError: if the manifest is missing or any
            required provenance field cannot be computed.  Callers MUST NOT
            soften this into a warning -- "fail closed if required
            prediction-cache provenance cannot be computed."
    """
    from src.training.prediction_cache_identity import get_prediction_cache_identity

    if manifest_path is None:
        manifest_path = resolved.prediction_cache_manifest_path

    ident = get_prediction_cache_identity(manifest_path)
    # Scalar split is derived from the resolved effective config, NOT from
    # the manifest (a single cache manifest spans all splits).  This matches
    # the value save_checkpoint writes at checkpoint time.
    ident["prediction_cache_split"] = resolved.validation_split
    return ident


def resolve_argparse_namespace(args, mode: str = "training") -> TrainerConfig:
    """Mirror the load+override path for an argparse Namespace from
    ``scripts/train_ddqn.py``, then enforce the shared explicit-bank gate.

    This keeps the CLI and the resolver byte-identical in how CLI flags map to
    overrides.  ``scripts/train_ddqn.py`` builds the overrides dict from the
    argparse Namespace and calls this helper so that the binding tests, which
    go through ``resolve_command_to_effective``, exercise the SAME override
    semantics AND the SAME mandatory explicit-bank gate.

    Raises:
        ExplicitBankError: if the namespace omits a required bank flag.
    """
    # MANDATORY GATE: shared with resolve_command_to_effective.  No bypass.
    banks_present = set()
    if getattr(args, "training_scenario_bank", None) is not None:
        banks_present.add("--training-scenario-bank")
    if getattr(args, "validation_scenario_bank", None) is not None:
        banks_present.add("--validation-scenario-bank")
    assert_explicit_banks(banks_present, source="argparse namespace")

    overrides: Dict[str, Any] = {}
    if args.split is not None:
        overrides["split"] = args.split
    if args.validation_split is not None:
        overrides["validation_split"] = args.validation_split
    if args.k_capacity is not None:
        overrides["maintenance_capacity"] = args.k_capacity
    if args.cost_regime is not None:
        overrides["cost_regime_id"] = args.cost_regime
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.training_seed is not None:
        overrides["training_seed"] = args.training_seed
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.run_id is not None:
        overrides["run_id"] = args.run_id
    if args.device is not None:
        overrides["device"] = args.device
    if args.training_scenario_bank is not None:
        overrides["training_scenario_bank_path"] = args.training_scenario_bank
    if args.validation_scenario_bank is not None:
        overrides["validation_scenario_bank_path"] = args.validation_scenario_bank

    parsed = load_and_validate_config(args.config, mode=mode)
    tc = parsed.trainer_config
    if overrides:
        tc = apply_cli_overrides(tc, overrides)
    return tc


__all__ = [
    "ResolvedCommand",
    "ExplicitBankError",
    "parse_command",
    "resolve_command_to_effective",
    "resolve_argparse_namespace",
    "assert_explicit_banks",
    "derive_prediction_cache_provenance",
    "REQUIRED_BANK_FLAGS",
]
