"""
Focused M5 reproducibility reproducibility-tool tests.

These tests cover the OFFICIAL M5 reproducibility reproducibility scripts:

  - scripts/generate_m5_formal_matrix.py  (frozen 40-row package builder)
  - scripts/run_m5_smoke.py       (8-cell cross-regime smoke driver)

They are NON-TRAINING: they exercise the matrix generator in
``--validate-only`` mode and import the smoke driver's constants/module
functions directly so no training subprocess is launched.  They assert the
frozen contract the spec demands.

Coverage:
  1. matrix generator produces exactly 40 unique rows;
  2. 20 K=1 + 20 K=2;
  3. exactly the 4 cost regimes, each 10x;
  4. seeds 6521..6525, each 8x;
  5. 40 unique run_ids and 40 unique expected_output_dirs;
  6. every row references the right regime-specific banks;
  7. max_steps=100000 on every row and in every launch command;
  8. rl_test is forbidden in rows and launch commands;
  9. every row's output_root is the new non-colliding root and is NOT under
     the superseded results/milestone5 root;
  10. preflight passes 40/40 with zero side effects in --validate-only mode;
  11. every launch command flag is a valid train_ddqn.py flag;
  12. --validate-only does NOT require --out-dir and writes nothing;
  13. every row records the producing git commit (== current HEAD);
  14. smoke driver exposes 8 cells (2 K * 4 regimes), seed 6521,
      max_steps=6000, the unique smoke output root, and no rl_test.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_external_assets

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

# Make scripts importable as modules.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

FORMAL_OUTPUT_ROOT = "results/milestone5_formal_regimebanks_v1"
SMOKE_OUTPUT_ROOT = "results/m5_smoke_v1"
SUPERSEDED_ROOT = "results/milestone5"

REGIMES = {
    "failure-light-no-waste",
    "failure-heavy-no-waste",
    "failure-light-waste-aware",
    "failure-heavy-waste-aware",
}
SEEDS = {6521, 6522, 6523, 6524, 6525}
SUFS = {"light", "heavy", "light_waste", "heavy_waste"}


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def _current_head() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def _train_help_flags() -> set[str]:
    txt = subprocess.run([PYTHON, "scripts/train_ddqn.py", "--help"],
                         cwd=str(REPO_ROOT), capture_output=True, text=True).stdout
    return set(re.findall(r"--([a-z-]+)", txt))


def _matrix_validate_only() -> str:
    """Run the matrix generator in --validate-only; return its stdout."""
    r = subprocess.run(
        [PYTHON, str(REPO_ROOT / "scripts/generate_m5_formal_matrix.py"),
         "--validate-only"],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    return r.stdout


def _import_matrix_module():
    import generate_m5_formal_matrix as mod
    return mod


def _import_smoke_module():
    import run_m5_smoke as mod
    return mod


# ---------------------------------------------------------------------------
# 1. - 9., 13. Matrix generator build_row contracts (no training, no I/O).
# ---------------------------------------------------------------------------

class TestMatrixRowContracts:
    """build_row must produce the frozen 40-row contract in-memory."""

    def _rows(self):
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

    def test_exactly_40_rows(self):
        rows, _ = self._rows()
        assert len(rows) == 40

    def test_k_split_20_20(self):
        rows, _ = self._rows()
        assert sum(1 for r in rows if r["k"] == 1) == 20
        assert sum(1 for r in rows if r["k"] == 2) == 20

    def test_four_regimes_each_10(self):
        rows, _ = self._rows()
        from collections import Counter
        c = Counter(r["cost_regime_id"] for r in rows)
        assert set(c) == REGIMES
        assert all(v == 10 for v in c.values())

    def test_seeds_6521_to_6525_each_8(self):
        rows, _ = self._rows()
        from collections import Counter
        c = Counter(r["seed"] for r in rows)
        assert set(c) == SEEDS
        assert all(v == 8 for v in c.values())

    def test_unique_run_ids(self):
        rows, _ = self._rows()
        assert len({r["run_id"] for r in rows}) == 40

    def test_unique_output_dirs(self):
        rows, _ = self._rows()
        assert len({r["expected_output_dir"] for r in rows}) == 40

    def test_regime_specific_banks(self):
        rows, _ = self._rows()
        suf = {
            "failure-light-no-waste": "light",
            "failure-heavy-no-waste": "heavy",
            "failure-light-waste-aware": "light_waste",
            "failure-heavy-waste-aware": "heavy_waste",
        }
        for r in rows:
            s = suf[r["cost_regime_id"]]
            assert r["training_scenario_bank_path"] == f"configs/scenarios/m5_pilot_k{r['k']}__{s}.json"
            assert r["validation_scenario_bank_path"] == f"configs/scenarios/m5_validation_k{r['k']}__{s}.json"

    def test_max_steps_100000(self):
        rows, _ = self._rows()
        assert all(r["max_steps"] == 100_000 for r in rows)
        assert all("--max-steps 100000" in r["exact_training_command"] for r in rows)

    def test_no_rl_test(self):
        rows, _ = self._rows()
        for r in rows:
            assert r["training_split"] != "rl_test"
            assert r["validation_split"] != "rl_test"
            assert "rl_test" not in r["run_id"]
            assert "rl_test" not in r["exact_training_command"]

    def test_new_output_root_not_superseded(self):
        rows, _ = self._rows()
        for r in rows:
            assert r["output_root"] == FORMAL_OUTPUT_ROOT
            assert r["expected_output_dir"].startswith(FORMAL_OUTPUT_ROOT + "/")
            # NOT under the superseded root (segment-boundary check).
            assert r["output_root"] != SUPERSEDED_ROOT
            assert not r["output_root"].startswith(SUPERSEDED_ROOT + "/")
            assert not r["expected_output_dir"].startswith(SUPERSEDED_ROOT + "/")

    def test_producing_commit_is_current_head(self):
        rows, head = self._rows()
        assert all(r["expected_git_commit"] == head for r in rows)

    def test_action_counts_match_k(self):
        rows, _ = self._rows()
        assert all(r["action_count"] == 6 for r in rows if r["k"] == 1)
        assert all(r["action_count"] == 16 for r in rows if r["k"] == 2)

    def test_launch_command_flags_valid(self):
        rows, _ = self._rows()
        allowed = _train_help_flags()
        for r in rows:
            used = set(re.findall(r"--([a-z-]+)", r["exact_training_command"]))
            assert used <= allowed, f"unknown flag(s): {used - allowed}"

    def test_resolved_config_identity_unique_and_hex(self):
        rows, _ = self._rows()
        ids = {r["resolved_config_identity"] for r in rows}
        assert len(ids) == 40
        for i in ids:
            assert isinstance(i, str) and len(i) == 64
            int(i, 16)


# ---------------------------------------------------------------------------
# 10. + 12. --validate-only: 40/40 preflight, no side effects, no --out-dir.
# ---------------------------------------------------------------------------

class TestMatrixValidateOnly:
    def test_validate_only_40_pass_and_no_side_effects(self, tmp_path):
        out = _matrix_validate_only()
        assert "Preflight PASS:     40/40" in out
        assert "Side-effects clean: True" in out
        assert "All-pass:           True" in out
        # The formal in-repo output root must NOT be created by --validate-only.
        formal_root = REPO_ROOT / FORMAL_OUTPUT_ROOT
        if formal_root.exists():
            # If it exists at all (from a prior real run), it must be empty
            # OR unchanged by this call.  We assert it does not contain any
            # run outputs created during this test.
            assert not any(p.is_dir() and p.name.startswith("m5_formal_k") for p in formal_root.iterdir())

    def test_validate_only_does_not_require_out_dir(self):
        # No --out-dir; must exit 0 and print the validate-only banner.
        out = _matrix_validate_only()
        assert "VALIDATE-ONLY (no package written" in out


# ---------------------------------------------------------------------------
# Package-write smoke (exercises writers; still NO training).
# ---------------------------------------------------------------------------

class TestMatrixPackageWriters:
    def test_writes_full_package_with_40_pending_ledger(self, tmp_path):
        r = subprocess.run(
            [PYTHON, str(REPO_ROOT / "scripts/generate_m5_formal_matrix.py"),
             "--out-dir", str(tmp_path)],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr
        flp = tmp_path / "frozen_launch_package"
        for name in ("formal_matrix.json", "formal_matrix.csv",
                     "preflight_results.json", "FORMAL_RUNBOOK.md",
                     "launch_commands.txt", "SUPERSESSION_RECORD.md",
                     "M5_SMOKE_INVENTORY.json", "SHA256SUMS.txt"):
            assert (flp / name).exists(), name
        assert (tmp_path / "run_ledgers" / "formal_execution_ledger.json").exists()
        assert (tmp_path / "operator_logs").is_dir()
        assert (tmp_path / "postrun_audit").is_dir()
        assert (tmp_path / "validation_analysis").is_dir()
        # Ledger: 40 PENDING rows.
        ledger = json.loads((tmp_path / "run_ledgers" / "formal_execution_ledger.json").read_text())
        assert ledger["total_rows"] == 40
        assert ledger["expected_complete_count"] == 40
        assert all(row["status"] == "PENDING" for row in ledger["rows"])
        assert len(ledger["rows"]) == 40
        # Preflight 40/40.
        pre = json.loads((flp / "preflight_results.json").read_text())
        assert pre["total_rows"] == 40 and pre["passed"] == 40 and pre["all_pass"] is True
        # No rl_test in any row of the matrix.
        m = json.loads((flp / "formal_matrix.json").read_text())
        assert m["no_rl_test"] is True
        for row in m["rows"]:
            assert "rl_test" not in row["exact_training_command"]

    def test_package_csv_matches_json_row_for_row(self, tmp_path):
        r = subprocess.run(
            [PYTHON, str(REPO_ROOT / "scripts/generate_m5_formal_matrix.py"),
             "--out-dir", str(tmp_path)],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr
        import csv
        flp = tmp_path / "frozen_launch_package"
        m = json.loads((flp / "formal_matrix.json").read_text())
        rows = list(csv.DictReader((flp / "formal_matrix.csv").open()))
        assert len(m["rows"]) == len(rows) == 40
        for jr, cr in zip(m["rows"], rows):
            for k in ("matrix_index", "k", "cost_regime_id", "seed", "run_id",
                     "expected_output_dir", "training_scenario_bank_path",
                     "training_scenario_bank_content_hash",
                     "validation_scenario_bank_path",
                     "validation_scenario_bank_content_hash",
                     "resolved_config_identity", "expected_git_commit"):
                assert str(jr[k]) == str(cr[k]), (k, jr["run_id"], jr[k], cr[k])


# ---------------------------------------------------------------------------
# 14. Smoke driver static contracts (no training subprocess launched).
# ---------------------------------------------------------------------------

class TestSmokeDriverContracts:
    def test_eight_cells_seed_6521_max_steps_6000(self):
        mod = _import_smoke_module()
        cells = [(k, r) for k in (1, 2) for r in mod.REGIMES]
        assert len(cells) == 8
        assert mod.SEED == 6521
        assert mod.MAX_STEPS == 6000
        assert mod.DEFAULT_OUTPUT_ROOT == SMOKE_OUTPUT_ROOT

    def test_smoke_run_ids_unique_per_cell(self):
        mod = _import_smoke_module()
        ids = [mod.cell_run_id(k, r) for k in (1, 2) for r in mod.REGIMES]
        assert len(set(ids)) == 8

    def test_smoke_bank_paths_regime_specific(self):
        mod = _import_smoke_module()
        for k in (1, 2):
            for r in mod.REGIMES:
                tb, vb = mod.cell_bank_paths(k, r)
                assert "__" in tb and "__" in vb
                assert tb.startswith(f"configs/scenarios/m5_pilot_k{k}__")
                assert vb.startswith(f"configs/scenarios/m5_validation_k{k}__")

    def test_smoke_module_rejects_rl_test_source(self):
        # The smoke driver module source must not set any split to rl_test as
        # a goal.  It must contain rl_test only in rejection/audit/error paths
        # (the manifest audit that flags rl_test as FORBIDDEN) and never in
        # any actual launch command split argument.
        src = (REPO_ROOT / "scripts/run_m5_smoke.py").read_text(encoding="utf-8")
        assert "rl_test" in src and "FORBIDDEN" in src  # rejection present
        # No launch command ever sets a split to rl_test.
        assert '"split": "rl_test"' not in src
        assert "'split', 'rl_test'" not in src
        assert '"validation_split": "rl_test"' not in src
        assert '--validation-split" "rl_test"' not in src
        assert '--split" "rl_test"' not in src
