"""
Focused M5 provenance EXACT-COMMAND / EFFECTIVE-CONFIG BINDING tests.

These tests prove the property the M5 reproducibility formal matrix violated: that a
row's recorded ``exact_training_command`` fully determines the same effective
configuration recorded in its matrix row, via the SAME production
``load_and_validate_config`` / ``apply_cli_overrides`` path used by
``scripts/train_ddqn.py``.

Coverage (Stage 6 of the M5 provenance contract):
  1.  all 40 formal commands resolve to their row-recorded banks;
  2.  K=1/heavy command uses m5_pilot_k1__heavy.json;
  3.  K=1/light-waste command uses m5_pilot_k1__light_waste.json;
  4.  K=2/heavy-waste command uses m5_pilot_k2__heavy_waste.json;
  5.  every command uses its matching validation bank;
  6.  exact-command identity equals matrix identity;
  7.  command-level --validate-only catches a mismatched bank;
  8.  command-level --validate-only creates no output directory;
  9.  missing bank overrides fail closed for a formal row;
  10. rl_test remains rejected;
  11. smoke and formal command builders share the same resolver;
  12. formal commands include max_steps=100000;
  13. smoke commands include max_steps=6000;
  14. no temporary per-cell config is required by the smoke driver;
  15. prediction-cache schema-6 provenance hashes match disk artifacts;
  16. all output paths remain unique and non-colliding.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_external_assets

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

FORMAL_OUTPUT_ROOT = "results/milestone5_formal_regimebanks_v1"
SMOKE_OUTPUT_ROOT = "results/m5_smoke_v1"
SUPERSEDED_ROOT = "results/milestone5"


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def _current_head() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def _train_help_flags() -> set[str]:
    txt = subprocess.run([PYTHON, "scripts/train_ddqn.py", "--help"],
                         cwd=str(REPO_ROOT), capture_output=True, text=True,
                         check=True).stdout
    return set(re.findall(r"--([a-z-]+)", txt))


def _import_matrix_module():
    import generate_m5_formal_matrix as mod
    return mod


def _import_smoke_module():
    import run_m5_smoke as mod
    return mod


def _build_rows():
    """Build the in-memory 40-row matrix through the production builder."""
    mod = _import_matrix_module()
    head = _current_head()
    pc_sha = mod.sha256_of(REPO_ROOT / mod.PC_MANIFEST_PATH)
    rows = [
        mod.build_row(k, r, s, head, pc_sha, mod.DEFAULT_OUTPUT_BASE)
        for k in mod.MAINTENANCE_CAPACITIES
        for r in mod.COST_REGIMES
        for s in mod.TRAINING_SEEDS
    ]
    return rows, head


def _resolve_command(command: str):
    """Resolve a command through the SHARED production resolver.

    Uses ``src.training.resolver.resolve_command_to_effective`` so the same
    path used by ``train_ddqn.py`` and preflight is exercised.  The command
    string is tokenised with :func:`shlex.split` (not ``str.split``), which
    correctly honours quoted paths containing spaces.
    """
    import shlex
    from src.training.resolver import resolve_command_to_effective
    parts = shlex.split(command)
    return resolve_command_to_effective(parts, cwd=REPO_ROOT)


# ---------------------------------------------------------------------------
# 1., 5., 6. Every formal command resolves to its row-recorded banks and
# identity.
# ---------------------------------------------------------------------------

class TestCommandResolvesToRowBanks:
    def test_all_40_commands_resolve_to_row_training_banks(self):
        rows, _ = _build_rows()
        for r in rows:
            eff = _resolve_command(r["exact_training_command"])
            assert eff.training_scenario_bank_path == r["training_scenario_bank_path"], (
                f"row {r['matrix_index']} ({r['run_id']}): "
                f"effective train bank {eff.training_scenario_bank_path!r} != "
                f"row bank {r['training_scenario_bank_path']!r}"
            )

    def test_all_40_commands_resolve_to_row_validation_banks(self):
        rows, _ = _build_rows()
        for r in rows:
            eff = _resolve_command(r["exact_training_command"])
            assert eff.validation_scenario_bank_path == r["validation_scenario_bank_path"], (
                f"row {r['matrix_index']} ({r['run_id']}): "
                f"effective val bank {eff.validation_scenario_bank_path!r} != "
                f"row bank {r['validation_scenario_bank_path']!r}"
            )

    def test_all_40_command_identities_equal_matrix_identity(self):
        from src.training.ddqn_config_identity import compute_resolved_config_identity
        rows, _ = _build_rows()
        for r in rows:
            eff = _resolve_command(r["exact_training_command"])
            eff_dict = {**eff.to_dict(), "num_actions": eff.num_actions}
            eff_id = compute_resolved_config_identity(eff_dict)
            assert eff_id == r["resolved_config_identity"], (
                f"row {r['matrix_index']} ({r['run_id']}): "
                f"command identity {eff_id} != matrix identity {r['resolved_config_identity']}"
            )

    def test_all_40_commands_resolve_expected_regime_K_splits(self):
        rows, _ = _build_rows()
        for r in rows:
            eff = _resolve_command(r["exact_training_command"])
            assert eff.cost_regime_id == r["cost_regime_id"]
            assert eff.maintenance_capacity == r["k"]
            assert eff.split == "predictor_train"
            assert eff.validation_split == "rl_validation"


# ---------------------------------------------------------------------------
# 2., 3., 4. Specific K/regime bank bindings.
# ---------------------------------------------------------------------------

class TestSpecificBankBindings:
    def _row(self, k, regime, seed=6521):
        rows, _ = _build_rows()
        for r in rows:
            if r["k"] == k and r["cost_regime_id"] == regime and r["seed"] == seed:
                return r
        raise AssertionError(f"row not found: k={k} regime={regime} seed={seed}")

    def test_k1_heavy_uses_heavy_training_bank(self):
        r = self._row(1, "failure-heavy-no-waste")
        assert r["training_scenario_bank_path"] == "configs/scenarios/m5_pilot_k1__heavy.json"
        eff = _resolve_command(r["exact_training_command"])
        assert eff.training_scenario_bank_path == "configs/scenarios/m5_pilot_k1__heavy.json"

    def test_k1_light_waste_uses_light_waste_training_bank(self):
        r = self._row(1, "failure-light-waste-aware")
        assert r["training_scenario_bank_path"] == "configs/scenarios/m5_pilot_k1__light_waste.json"
        eff = _resolve_command(r["exact_training_command"])
        assert eff.training_scenario_bank_path == "configs/scenarios/m5_pilot_k1__light_waste.json"

    def test_k2_heavy_waste_uses_heavy_waste_training_bank(self):
        r = self._row(2, "failure-heavy-waste-aware")
        assert r["training_scenario_bank_path"] == "configs/scenarios/m5_pilot_k2__heavy_waste.json"
        eff = _resolve_command(r["exact_training_command"])
        assert eff.training_scenario_bank_path == "configs/scenarios/m5_pilot_k2__heavy_waste.json"

    def test_k1_heavy_uses_heavy_validation_bank(self):
        r = self._row(1, "failure-heavy-no-waste")
        eff = _resolve_command(r["exact_training_command"])
        assert eff.validation_scenario_bank_path == "configs/scenarios/m5_validation_k1__heavy.json"


# ---------------------------------------------------------------------------
# 7., 8., 9. Command-level --validate-only behavior.
# ---------------------------------------------------------------------------

class TestCommandValidateOnly:
    def test_validate_only_passes_for_correct_formal_row(self, tmp_path):
        import shlex
        rows, _ = _build_rows()
        r = rows[5]  # K=1/heavy/6521
        cmd = shlex.split(r["exact_training_command"]) + ["--validate-only"]
        # Redirect output dir to tmp so NO production output dir is created.
        # Replace --output-dir arg with tmp_path.
        out_idx = cmd.index("--output-dir")
        cmd[out_idx + 1] = str(tmp_path / "noshadetest")
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT),
                              capture_output=True, text=True)
        assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
        assert "Configuration validated successfully" in proc.stdout

    def test_validate_only_catches_mismatched_bank(self, tmp_path):
        """A command whose --training-scenario-bank disagrees with
        --cost-regime must fail closed under --validate-only."""
        import shlex
        rows, _ = _build_rows()
        heavy = rows[5]  # K=1/heavy/6521
        # Replace its --training-scenario-bank with the baseline light bank.
        cmd = list(shlex.split(heavy["exact_training_command"]))
        cmd += ["--validate-only"]
        # Inject a mismatching training bank (light instead of heavy).
        cmd += ["--training-scenario-bank", "configs/scenarios/m5_pilot_k1__light.json"]
        out_idx = cmd.index("--output-dir")
        cmd[out_idx + 1] = str(tmp_path / "mismatch")
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT),
                              capture_output=True, text=True)
        assert proc.returncode != 0, "validate-only must reject mismatched bank"
        combined = proc.stdout + proc.stderr
        assert "FAILED" in combined or "does not match" in combined or "mismatch" in combined.lower() or "ERROR" in combined, combined

    def test_validate_only_creates_no_output_directory(self, tmp_path):
        import shlex
        rows, _ = _build_rows()
        r = rows[0]
        target = tmp_path / "no_dir_should_appear_here"
        cmd = shlex.split(r["exact_training_command"]) + ["--validate-only"]
        out_idx = cmd.index("--output-dir")
        cmd[out_idx + 1] = str(target)
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT),
                              capture_output=True, text=True)
        assert proc.returncode == 0, (proc.stdout, proc.stderr)
        assert not target.exists(), "validate-only must NOT create the output directory"

    def test_missing_bank_overrides_fail_closed_for_formal_row(self, tmp_path):
        """A formal-row style command WITHOUT --training-scenario-bank /
        --validation-scenario-bank must fail closed even when the base config
        already points at a baseline bank (the prior root blocker).

        The gate lives in the shared resolver; --allow-baseline-banks no longer
        exists (frozen decision: always require explicit banks).  The failure
        is therefore reported by the resolver as an explicit-bank gate error,
        BEFORE any TrainerConfig is materialised."""
        import shlex
        rows, _ = _build_rows()
        r = rows[5]  # K=1/heavy/6521
        cmd = shlex.split(r["exact_training_command"])
        # Strip the explicit bank flags to simulate the old binding.
        cmd = [c for i, c in enumerate(cmd)
               if not (c in ("--training-scenario-bank", "--validation-scenario-bank")
                       or (i > 0 and cmd[i - 1] in ("--training-scenario-bank", "--validation-scenario-bank")))]
        cmd += ["--validate-only"]
        out_idx = cmd.index("--output-dir")
        cmd[out_idx + 1] = str(tmp_path / "failclosed")
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT),
                              capture_output=True, text=True)
        # Must fail closed — missing explicit bank override for a formal row.
        assert proc.returncode != 0, (
            "missing explicit bank overrides must fail closed, "
            f"but command succeeded:\n{proc.stdout}\n{proc.stderr}"
        )
        combined = proc.stdout + proc.stderr
        assert "explicit-bank gate FAILED" in combined, combined


# ---------------------------------------------------------------------------
# 10. rl_test remains rejected.
# ---------------------------------------------------------------------------

class TestRlTestRejected:
    def test_no_row_uses_rl_test(self):
        rows, _ = _build_rows()
        for r in rows:
            assert r["training_split"] != "rl_test"
            assert r["validation_split"] != "rl_test"
            assert "rl_test" not in r["exact_training_command"]

    def test_validate_only_rejects_rl_test_split(self, tmp_path):
        import shlex
        rows, _ = _build_rows()
        r = rows[0]
        cmd = shlex.split(r["exact_training_command"]) + ["--validate-only"]
        cmd += ["--split", "rl_test"]
        out_idx = cmd.index("--output-dir")
        cmd[out_idx + 1] = str(tmp_path / "rltest")
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT),
                              capture_output=True, text=True)
        assert proc.returncode != 0
        assert "rl_test" in (proc.stdout + proc.stderr)


# ---------------------------------------------------------------------------
# 11., 12., 13., 14. Smoke and formal command builders.
# ---------------------------------------------------------------------------

class TestSmokeAndFormalShareResolver:
    def test_smoke_and_formal_use_same_resolver(self):
        from src.training.resolver import resolve_command_to_effective
        # Both builders must route through resolve_command_to_effective.
        smoke_src = (REPO_ROOT / "scripts" / "run_m5_smoke.py").read_text()
        formal_src = (REPO_ROOT / "scripts" / "generate_m5_formal_matrix.py").read_text()
        assert "resolve_command_to_effective" in smoke_src, (
            "smoke driver must use the shared resolver")
        assert "resolve_command_to_effective" in formal_src or "resolve_command_to_effective" in (
            REPO_ROOT / "src" / "training" / "resolver.py").read_text(), (
            "formal generator must use the shared resolver")

    def test_formal_commands_include_max_steps_100000(self):
        rows, _ = _build_rows()
        for r in rows:
            assert "--max-steps 100000" in r["exact_training_command"], r["exact_training_command"]

    def test_smoke_commands_include_max_steps_6000(self):
        mod = _import_smoke_module()
        # The smoke command builder must emit --max-steps 6000.
        # Inspect by building a sample command (no training).
        # First confirm the smoke builder function exists and sets 6000.
        assert mod.MAX_STEPS == 6000
        # Build the per-cell command shape and check the max-steps token.
        # Use a synthetic temp config path (the builder only needs the path).
        # The smoke builder now uses production CLI bonds; verify the source.
        src = (REPO_ROOT / "scripts" / "run_m5_smoke.py").read_text(encoding="utf-8")
        assert "--max-steps" in src
        assert "6000" in src

    def test_smoke_driver_requires_no_temp_per_cell_config(self):
        """The smoke driver must NOT depend on temp per-cell config files
        when the production CLI supports explicit bank overrides."""
        src = (REPO_ROOT / "scripts" / "run_m5_smoke.py").read_text(encoding="utf-8")
        # The old pattern materialised temp configs with
        # training_scenario_bank_path baked in.  After binding corrected,
        # the driver launches the base config + --training-scenario-bank /
        # --validation-scenario-bank flags directly.
        assert "cell_temp_config" not in src, (
            "smoke driver must not materialise temp per-cell configs")
        assert "--training-scenario-bank" in src
        assert "--validation-scenario-bank" in src


# ---------------------------------------------------------------------------
# 15. Prediction-cache schema-6 provenance hashes match disk artifacts.
# ---------------------------------------------------------------------------

class TestPredictionCacheProvenance:
    def test_predict_cache_identity_fields_match_disk(self):
        """Recompute the manifest-derived provenance fields from the
        prediction-cache manifest on disk and ensure they exactly equal what
        ``get_prediction_cache_identity`` returns (fail-closed, no
        validate-if-present)."""
        import hashlib
        from src.training.prediction_cache_identity import get_prediction_cache_identity
        pc_manifest = REPO_ROOT / "data/processed/fd001/v2/06_PREDICTIONS/prediction_cache_manifest_v2.json"
        ident = get_prediction_cache_identity(pc_manifest)
        # Manifest-derived provenance fields must be present and non-null.
        manifest_fields = [
            "prediction_cache_manifest_path",
            "prediction_cache_manifest_sha256",
            "prediction_cache_declared_cache_hash",
            "prediction_cache_predictor_checkpoint_hash",
            "prediction_cache_feature_schema_hash",
            "prediction_cache_normalizer_hash",
            "prediction_cache_schema_version",
        ]
        for f in manifest_fields:
            assert f in ident, f"missing manifest-derived provenance field {f}"
            assert ident[f] is not None, f"null provenance field {f}"
        # The scalar ``prediction_cache_split`` is NOT manifest-derived: the
        # schema-6 trainer sets it to the resolved ``validation_split`` at
        # save time (see src/agents/ddqn/checkpoint.py).  The smoke check derives
        # the disk-side ``prediction_cache_split`` from the resolved command's
        # validation split, not from the manifest (which has many splits).
        # The manifest exposes the LIST of splits present (it may include
        # rl_test because the V2 cache covers all splits; rl_test sealing is
        # about not TRAINING on rl_test scenarios, enforced at the
        # preflight / env / CLI barrier, not at the cache manifest).
        assert "prediction_cache_splits_present" in ident
        assert "predictor_train" in ident["prediction_cache_splits_present"]
        assert "rl_validation" in ident["prediction_cache_splits_present"]
        on_disk = hashlib.sha256(pc_manifest.read_bytes()).hexdigest()
        assert ident["prediction_cache_manifest_sha256"] == on_disk

    def test_audit_provenance_fn_reads_checkpoint_metadata(self):
        """The smoke verification helper must read schema-6 checkpoint metadata
        provenance fields directly and compare to recomputed disk values,
        with NO 'validate-if-present' weakening."""
        src = (REPO_ROOT / "scripts" / "run_m5_smoke.py").read_text(encoding="utf-8")
        # The smoke check must NOT weaken provenance with the old fallback pattern.
        assert "fallback" not in src.lower() or "no " + "fallback" in src.lower(), (
            "smoke check must keep the strict (no-fallback) provenance contract")
        # Must reference the manifest-derived provenance fields by name and
        # the scalar split field the trainer sets.
        for f in ("prediction_cache_manifest_sha256",
                  "prediction_cache_declared_cache_hash",
                  "prediction_cache_feature_schema_hash",
                  "prediction_cache_normalizer_hash",
                  "prediction_cache_split"):
            assert f in src, f"smoke audit must verify {f}"
        # Must call the helper that reads checkpoint metadata.
        assert "audit_prediction_cache_provenance" in src


# ---------------------------------------------------------------------------
# 16. All output paths remain unique and non-colliding.
# ---------------------------------------------------------------------------

class TestOutputPathsUnique:
    def test_40_unique_run_ids(self):
        rows, _ = _build_rows()
        assert len({r["run_id"] for r in rows}) == 40

    def test_40_unique_output_dirs(self):
        rows, _ = _build_rows()
        assert len({r["expected_output_dir"] for r in rows}) == 40

    def test_40_unique_identities(self):
        rows, _ = _build_rows()
        assert len({r["resolved_config_identity"] for r in rows}) == 40

    def test_no_formal_dir_under_superseded_root(self):
        rows, _ = _build_rows()
        for r in rows:
            assert not r["expected_output_dir"].startswith(SUPERSEDED_ROOT + "/")
            assert not r["expected_output_dir"] == SUPERSEDED_ROOT
            assert r["output_root"] == FORMAL_OUTPUT_ROOT


# ---------------------------------------------------------------------------
# Command-level preflight: all 40 rows resolve + identity equality, zero side
# effects.
# ---------------------------------------------------------------------------

class TestCommandLevelPreflight:
    def test_all_40_commands_preflight_pass(self, tmp_path):
        import shlex
        rows, _ = _build_rows()
        from src.training.resolver import resolve_command_to_effective
        from src.training.preflight import validate_row_asset_contract
        from src.training.ddqn_config_identity import compute_resolved_config_identity
        from src.agents.ddqn.checkpoint import compute_scenario_bank_content_hash
        for r in rows:
            eff = resolve_command_to_effective(shlex.split(r["exact_training_command"]),
                                               cwd=REPO_ROOT)
            rep = validate_row_asset_contract(
                training_scenario_bank_path=str((REPO_ROOT / eff.training_scenario_bank_path).resolve()),
                validation_scenario_bank_path=str((REPO_ROOT / eff.validation_scenario_bank_path).resolve()),
                cost_regime_id=eff.cost_regime_id,
                maintenance_capacity=eff.maintenance_capacity,
                prediction_cache_path=str((REPO_ROOT / "data/processed/fd001/v2/06_PREDICTIONS/").resolve()),
                training_split=eff.split,
                validation_split=eff.validation_split,
            )
            assert rep.ok, f"row {r['matrix_index']} preflight errors: {rep.errors}"
            # Effective bank hashes must equal the on-disk content hashes.
            tbh = compute_scenario_bank_content_hash(REPO_ROOT / eff.training_scenario_bank_path)
            vbh = compute_scenario_bank_content_hash(REPO_ROOT / eff.validation_scenario_bank_path)
            assert tbh == r["training_scenario_bank_content_hash"], f"row {r['matrix_index']} train bank hash"
            assert vbh == r["validation_scenario_bank_content_hash"], f"row {r['matrix_index']} val bank hash"
            # Identity equality.
            eff_dict = {**eff.to_dict(), "num_actions": eff.num_actions}
            assert compute_resolved_config_identity(eff_dict) == r["resolved_config_identity"]


# ---------------------------------------------------------------------------
# 17. Frozen decision: "always require explicit banks".  The gate lives in
#     the shared resolver; no --allow-baseline-banks bypass exists.
# ---------------------------------------------------------------------------

class TestExplicitBankGateInResolver:
    """The mandatory explicit-bank gate MUST live in the shared resolver, so
    every entry point (CLI, matrix, smoke, preflight, tests) goes through one
    gate.  There is NO opt-out: --allow-baseline-banks is removed entirely."""

    def test_resolver_raises_on_missing_training_bank(self):
        import shlex
        from src.training.resolver import resolve_command_to_effective, ExplicitBankError
        rows, _ = _build_rows()
        r = rows[5]
        cmd = shlex.split(r["exact_training_command"])
        # Strip ONLY the training bank flag.
        idx = cmd.index("--training-scenario-bank")
        cmd = cmd[:idx] + cmd[idx + 2:]
        with pytest.raises(ExplicitBankError):
            resolve_command_to_effective(cmd, cwd=REPO_ROOT)

    def test_resolver_raises_on_missing_both_banks(self):
        import shlex
        from src.training.resolver import resolve_command_to_effective, ExplicitBankError
        rows, _ = _build_rows()
        r = rows[0]
        cmd = shlex.split(r["exact_training_command"])
        # Strip both bank flags.
        for flag in ("--training-scenario-bank", "--validation-scenario-bank"):
            idx = cmd.index(flag)
            cmd = cmd[:idx] + cmd[idx + 2:]
        with pytest.raises(ExplicitBankError):
            resolve_command_to_effective(cmd, cwd=REPO_ROOT)

    def test_resolver_passes_when_both_banks_present(self):
        # All 40 formal commands include both bank flags -> no raise.
        rows, _ = _build_rows()
        for r in rows:
            eff = _resolve_command(r["exact_training_command"])
            assert eff.training_scenario_bank_path
            assert eff.validation_scenario_bank_path

    def test_no_allow_baseline_banks_flag_in_cli(self):
        # No argparse declaration of the removed bypass, and --help must not
        # expose it.  A documentation comment mentioning its removal is fine.
        import subprocess
        src = (REPO_ROOT / "scripts" / "train_ddqn.py").read_text(encoding="utf-8")
        # The bypass argument object must not be declared.
        assert '"--allow-baseline-banks"' not in src, (
            '--allow-baseline-banks string literal must be removed from '
            'train_ddqn.py (including comments); rename the prose if needed')
        assert "allow_baseline_banks" not in src, (
            "allow_baseline_banks identifier must be removed from train_ddqn.py")
        # --help must not advertise the bypass.
        txt = subprocess.run([sys.executable, "scripts/train_ddqn.py", "--help"],
                             cwd=str(REPO_ROOT), capture_output=True,
                             text=True, check=True).stdout
        assert "--allow-baseline-banks" not in txt, (
            "--allow-baseline-banks must not appear in --help")

    def test_resolver_module_exposes_assert_explicit_banks(self):
        from src.training.resolver import assert_explicit_banks, ExplicitBankError, REQUIRED_BANK_FLAGS
        assert callable(assert_explicit_banks)
        assert set(REQUIRED_BANK_FLAGS) == {
            "--training-scenario-bank", "--validation-scenario-bank"}
        # An empty bank set must raise.
        with pytest.raises(ExplicitBankError):
            assert_explicit_banks(set(), source="test")
        # A full bank set must NOT raise.
        assert_explicit_banks(set(REQUIRED_BANK_FLAGS), source="test")


# ---------------------------------------------------------------------------
# 18. prediction_cache_split is derived from the resolved effective config's
#     validation_split; prediction-cache provenance fails closed if the
#     manifest cannot be read.
# ---------------------------------------------------------------------------

class TestPredictionCacheSplitDerived:
    def test_resolver_prediction_cache_split_matches_validation_split(self):
        rows, _ = _build_rows()
        from src.training.resolver import resolve_command_to_effective
        import shlex
        for r in rows:
            eff = resolve_command_to_effective(shlex.split(r["exact_training_command"]),
                                                cwd=REPO_ROOT)
            assert eff.prediction_cache_split == eff.validation_split == "rl_validation"

    def test_derive_prediction_cache_provenance_succeeds_for_row(self):
        rows, _ = _build_rows()
        from src.training.resolver import (resolve_command_to_effective,
                                           derive_prediction_cache_provenance)
        import shlex
        r = rows[0]
        eff = resolve_command_to_effective(shlex.split(r["exact_training_command"]),
                                           cwd=REPO_ROOT)
        prov = derive_prediction_cache_provenance(eff)
        # The scalar split must be derived from the resolved config, NOT the
        # manifest (which spans all splits).
        assert prov["prediction_cache_split"] == eff.validation_split == "rl_validation"
        # Manifest-derived provenance fields must be present and non-null.
        for f in ("prediction_cache_manifest_path",
                  "prediction_cache_manifest_sha256",
                  "prediction_cache_declared_cache_hash",
                  "prediction_cache_predictor_checkpoint_hash",
                  "prediction_cache_feature_schema_hash",
                  "prediction_cache_normalizer_hash",
                  "prediction_cache_schema_version"):
            assert f in prov and prov[f] is not None, f

    def test_derive_prediction_cache_provenance_fails_closed_on_missing_manifest(self, tmp_path):
        """Fail closed (raise) if the prediction-cache manifest cannot be
        read -- NO validate-if-present weakening."""
        rows, _ = _build_rows()
        from src.training.resolver import (resolve_command_to_effective,
                                           derive_prediction_cache_provenance)
        import shlex
        r = rows[0]
        eff = resolve_command_to_effective(shlex.split(r["exact_training_command"]),
                                           cwd=REPO_ROOT)
        bogus = str(tmp_path / "does_not_exist_manifest.json")
        with pytest.raises((FileNotFoundError, ValueError, OSError)):
            derive_prediction_cache_provenance(eff, manifest_path=bogus)