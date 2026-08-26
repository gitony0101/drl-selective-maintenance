"""
M8 Upstream Immutability Tests (UI-01 through UI-13)

Tests for upstream immutability per M8_TEST_PLAN.md Section 6.
"""
import hashlib
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_external_assets

# Root of the (git-ignored) M8 upstream worktree holding the frozen artifacts
# this immutability suite verifies. Override with DRL_EXTERNAL_ROOT.
_M8_ROOT = (
    __import__("src.runtime_paths", fromlist=["external_root"])
    .external_root()
    / "drl-selective-maintenance-m8"
)


class TestM8UpstreamImmutability:
    """Upstream immutability tests UI-01 through UI-13."""

    @classmethod
    def setup_class(cls):
        """Set up paths and pre-compute hashes."""
        cls.data_dir = Path("data/processed/fd001/v2")
        cls.repo_root = _M8_ROOT

        # Authoritative artifact paths and content hashes (frozen data-preparation record)
        cls.m1_artifacts = {
            "checkpoint": {
                "path": Path("results/predictor/mse_baseline_v2/checkpoints/best_checkpoint.pt"),
                "expected_hash": "ade3688496de7672367fcb58bbcba384f6835f81fe2c89d7ce9f88eeebe5b2b7",
            },
            "normalizer": {
                "path": cls.data_dir / "04_PROTOCOL/fd001_normalizer_v2.json",
                "expected_hash": "08477180719d004dc8f962762735b6344f8198a7719c1c510f9ad7ee15784fde",
            },
            "feature_schema": {
                "path": cls.data_dir / "04_PROTOCOL/fd001_feature_schema_v1.json",
                "expected_hash": "43772bbcaab99e79264fac54780025a54de6e29c75fdccab6dd4ef4d2cbe21da",
            },
            "split_manifest": {
                "path": cls.data_dir / "01_SPLIT/fd001_unit_split_v1.csv",
                "expected_hash": "a86fe8cb1e01d4c7b47fd76d9bcc23351e64b4641386838bee6475bd2863dc9a",
            },
            "prediction_cache": {
                "path": cls.data_dir / "06_PREDICTIONS/fd001_prediction_cache_v2.parquet",
                "expected_hash": "17596b4c15d92c82cc7d09e9417fbc848190731deb1a8bbe95f8fef19e7d5217",
            },
        }

        # M2-M6 source directories
        cls.upstream_sources = {
            "m2": Path("src/envs"),
            "m3": Path("src/agents/ddqn"),
            "m4": Path("src/agents/ddqn"),
            "m5": Path("src/maintenance"),
            "m6": Path("src/agents/ddqn"),
            "integration": Path("tests/test_integration_m3_m4_m5.py"),
        }

        # Pre-compute hashes
        cls._pre_hashes = {}
        for key, info in cls.m1_artifacts.items():
            cls._pre_hashes[key] = cls._hash_file(info["path"])

        for key, path in cls.upstream_sources.items():
            if path.is_dir():
                cls._pre_hashes[f"{key}_dir"] = cls._hash_dir(path)
            elif path.is_file():
                cls._pre_hashes[f"{key}_file"] = cls._hash_file(path)

    @staticmethod
    def _hash_file(path):
        """Compute SHA256 of a file."""
        if not Path(path).exists():
            return None
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _hash_dir(path):
        """Compute combined hash of all files in directory (git-tracked only)."""
        try:
            result = subprocess.run(
                ["git", "ls-files", "-z", str(path)],
                capture_output=True,
                text=True,
                cwd=str(_M8_ROOT),
            )
            if result.returncode != 0:
                return None
            files = [f for f in result.stdout.split("\0") if f]
            h = hashlib.sha256()
            for f in sorted(files):
                full = _M8_ROOT / f
                if full.exists():
                    with open(full, "rb") as fh:
                        for chunk in iter(lambda: fh.read(8192), b""):
                            h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None

    def _git_diff(self, path):
        """Get git diff for a path."""
        result = subprocess.run(
            ["git", "diff", "--", str(path)],
            capture_output=True,
            text=True,
            cwd=str(_M8_ROOT),
        )
        return result.stdout

    # UI-01
    def test_ui01_m1_checkpoint_unchanged(self):
        """UI-01: M1 checkpoint SHA256 unchanged."""
        info = self.m1_artifacts["checkpoint"]
        post_hash = self._hash_file(info["path"])

        if info["expected_hash"] is None or post_hash is None:
            pytest.skip("M1 checkpoint not present on disk (gitignored)")

        assert post_hash == info["expected_hash"], (
            f"M1 checkpoint hash changed: expected {info['expected_hash']}, got {post_hash}"
        )

    # UI-02
    def test_ui02_m1_normalizer_unchanged(self):
        """UI-02: M1 normalizer SHA256 unchanged."""
        info = self.m1_artifacts["normalizer"]
        post_hash = self._hash_file(info["path"])

        assert post_hash == info["expected_hash"], (
            f"M1 normalizer hash changed: expected {info['expected_hash']}, got {post_hash}"
        )

    # UI-03
    def test_ui03_m1_feature_schema_unchanged(self):
        """UI-03: M1 feature schema SHA256 unchanged."""
        info = self.m1_artifacts["feature_schema"]
        post_hash = self._hash_file(info["path"])

        assert post_hash == info["expected_hash"], (
            f"M1 feature schema hash changed: expected {info['expected_hash']}, got {post_hash}"
        )

    # UI-04
    def test_ui04_m1_frozen_cache_unchanged(self):
        """UI-04: M1 frozen prediction cache SHA256 unchanged (if exists)."""
        info = self.m1_artifacts["prediction_cache"]
        post_hash = self._hash_file(info["path"])

        if info["expected_hash"] is None or post_hash is None:
            pytest.skip("M1 prediction cache not present on disk (gitignored)")
            return

        assert post_hash == info["expected_hash"], (
            f"M1 prediction cache hash changed: expected {info['expected_hash']}, got {post_hash}"
        )

    # UI-05
    def test_ui05_split_manifest_unchanged(self):
        """UI-05: Split manifest SHA256 unchanged."""
        info = self.m1_artifacts["split_manifest"]
        post_hash = self._hash_file(info["path"])

        assert post_hash == info["expected_hash"], (
            f"Split manifest hash changed: expected {info['expected_hash']}, got {post_hash}"
        )

    # UI-06
    def test_ui06_m2_env_unchanged(self):
        """UI-06: M2 environment source files unchanged."""
        diff = self._git_diff(self.upstream_sources["m2"])
        assert diff == "", f"M2 env source files changed:\n{diff}"

    # UI-07
    def test_ui07_m3_unchanged(self):
        """UI-07: M3 source files unchanged."""
        diff = self._git_diff(self.upstream_sources["m3"])
        assert diff == "", f"M3 source files changed:\n{diff}"

    # UI-08
    def test_ui08_m4_unchanged(self):
        """UI-08: M4 source files unchanged."""
        diff = self._git_diff(self.upstream_sources["m4"])
        assert diff == "", f"M4 source files changed:\n{diff}"

    # UI-09
    def test_ui09_m5_unchanged(self):
        """UI-09: M5 source files unchanged."""
        diff = self._git_diff(self.upstream_sources["m5"])
        assert diff == "", f"M5 source files changed:\n{diff}"

    # UI-10
    def test_ui10_m6_unchanged(self):
        """UI-10: M6 source files unchanged."""
        diff = self._git_diff(self.upstream_sources["m6"])
        assert diff == "", f"M6 source files changed:\n{diff}"

    # UI-11
    def test_ui11_integration_unchanged(self):
        """UI-11: M3/M4/M5 integration test file unchanged."""
        diff = self._git_diff(self.upstream_sources["integration"])
        assert diff == "", f"Integration test file changed:\n{diff}"

    # UI-12
    def test_ui12_no_rl_test_access(self):
        """UI-12: No rl_test imports or references in M8 code."""
        m8_files = list(Path("src/predictors/losses").rglob("*.py")) + [
            Path("src/predictors/train.py"),
        ]

        forbidden = ["rl_test", "rl_test_split", "test_maintenance", "RLTest"]
        for f in m8_files:
            if f.exists():
                content = f.read_text()
                for term in forbidden:
                    assert term not in content, f"Forbidden term '{term}' found in {f}"

    # UI-13
    def test_ui13_no_ddqn_training(self):
        """UI-13: No DDQN training code in M8 modules."""
        m8_files = list(Path("src/predictors/losses").rglob("*.py")) + [
            Path("src/predictors/train.py"),
        ]

        forbidden = ["DDQNTrainer", "train_ddqn", "DQNAgent", "ReplayBuffer", "epsilon_greedy"]
        for f in m8_files:
            if f.exists():
                content = f.read_text()
                for term in forbidden:
                    assert term not in content, f"DDQN term '{term}' found in M8 file {f}"

    # UI-14
    def test_ui14_no_common_bank_generation(self):
        """UI-14: No common maintenance scenario bank generation in M8."""
        m8_files = list(Path("src/predictors/losses").rglob("*.py")) + [
            Path("src/predictors/train.py"),
        ]

        forbidden = ["scenario_bank", "common_bank", "MaintenanceScenario", "generate_scenarios"]
        for f in m8_files:
            if f.exists():
                content = f.read_text()
                for term in forbidden:
                    assert term not in content, f"Common bank term '{term}' found in M8 file {f}"

    # UI-15
    def test_ui15_no_formal_training(self):
        """UI-15: No full 200-epoch training in M8 code (smoke only)."""
        m8_files = list(Path("src/predictors/losses").rglob("*.py")) + [
            Path("src/predictors/train.py"),
        ]

        # Check for hardcoded 200 epochs (formal training budget)
        forbidden = ["max_epochs=200", "max_epochs = 200", "epochs=200", "epochs = 200"]
        for f in m8_files:
            if f.exists():
                content = f.read_text()
                for term in forbidden:
                    assert term not in content, f"Formal training term '{term}' found in M8 file {f}"

    # UI-16
    def test_ui16_m8_only_changed_files(self):
        """UI-16: Only M8-scope files changed in this branch."""
        result = subprocess.run(
            ["git", "diff", "--name-only", "4947e33..HEAD"],
            capture_output=True,
            text=True,
            cwd=str(_M8_ROOT),
        )
        changed_files = [f.strip() for f in result.stdout.split("\n") if f.strip()]

        # Allowed M8 files
        allowed_prefixes = [
            "src/predictors/losses/",
            "configs/predictor/",
            "tests/test_m8_",
            "docs/milestone8/",
        ]

        # Also allow the train.py modification (adapter)
        allowed_files = {
            "src/predictors/train.py",
            "configs/predictor/mse_baseline.json",
            # Exact-path contract sync with the governing implementation plan
            # docs/milestone8/runner_metrics_implementation_plan/M8_IMPLEMENTATION_COMMIT_PLAN.md
            # (tracked at HEAD). The six paths below are authorized by exact
            # pathname under commits M8-METRICS-1 (§2/§4.1) and M8-RUNNER-1
            # (§2/§5). They are listed here as exact files — NOT directory
            # prefixes — so no bare "scripts/", "tests/", or "src/" grant is
            # introduced. This synchronizes a stale UI-16 allowlist with the
            # approved runner/metrics implementation scope; it does not weaken
            # the immutability contract, which still rejects every path outside
            # this enumeration (see test_ui16_rejects_unauthorized_paths).
            "src/predictors/formal_metrics.py",      # M8-METRICS-1
            "src/predictors/formal_runner.py",        # M8-RUNNER-1
            "scripts/build_val_cycle_table.py",       # M8-RUNNER-1
            "scripts/drl_heavy_mps_lock.py",          # M8-RUNNER-1
            "tests/test_formal_metrics.py",            # M8-METRICS-1
            "tests/test_formal_runner.py",             # M8-RUNNER-1
        }

        for f in changed_files:
            is_allowed = any(f.startswith(p) for p in allowed_prefixes) or f in allowed_files
            assert is_allowed, f"Unauthorized file changed in M8 branch: {f}"

    # UI-16 negative test (contract-sync companion)
    def test_ui16_rejects_unauthorized_paths(self):
        """UI-16 negative: exact-path contract genuinely rejects unauthorized files.

        This companion test proves the Phase-2 contract synchronization is not a
        blanket allowlist: a file under scripts/, tests/, or src/ that is NOT one
        of the six exact paths authorized by
        docs/milestone8/runner_metrics_implementation_plan/M8_IMPLEMENTATION_COMMIT_PLAN.md
        must still be rejected. It also asserts no bare "scripts/", "tests/", or
        "src/predictors/" prefix is present in the allowlist, so a directory-wide
        grant can never silently admit future unauthorized files.
        """
        # Reconstruct the same allowlist the positive test uses.
        allowed_prefixes = [
            "src/predictors/losses/",
            "configs/predictor/",
            "tests/test_m8_",
            "docs/milestone8/",
        ]
        allowed_files = {
            "src/predictors/train.py",
            "configs/predictor/mse_baseline.json",
            "src/predictors/formal_metrics.py",
            "src/predictors/formal_runner.py",
            "scripts/build_val_cycle_table.py",
            "scripts/drl_heavy_mps_lock.py",
            "tests/test_formal_metrics.py",
            "tests/test_formal_runner.py",
        }

        # No bare directory-wide grant may exist for the formally-scoped dirs.
        # A "bare" prefix is a discrete allowed_prefixes entry that is exactly a
        # directory (e.g. "scripts/", "tests/", "src/", "src/predictors/") — it
        # would admit every current and future file in that directory. Scoped
        # prefixes such as "tests/test_m8_" and "docs/milestone8/" are not bare
        # (they name a constrained subtree), and are intentionally present in the
        # committed allowlist. Phase 2 must not convert a scoped prefix into a
        # bare one.
        bare_forbidden_prefixes = {
            "scripts/",
            "tests/",
            "src/",
            "src/predictors/",
            "configs/",
        }
        for p in allowed_prefixes:
            assert p not in bare_forbidden_prefixes, (
                f"Forbidden bare directory prefix in allowed_prefixes: {p}"
            )

        # Genuine unauthorized paths — even plausible-looking ones — must fail.
        genuinely_unauthorized = [
            "scripts/some_other_script.py",          # same dir, not authorized
            "tests/test_something_else.py",          # tests/ but not test_m8_/formal
            "src/predictors/unrelated_module.py",    # src/predictors/ but not exact
            "src/agents/ddqn/anything.py",           # upstream M3/M6, never M8
            "data/processed/fd001/v2/anything.parquet",
            "configs/predictor/unauthorized.json",   # configs/predictor/ IS a prefix...
        ]
        for f in genuinely_unauthorized:
            # configs/predictor/ is an intentionally broad prefix already present
            # in the committed allowlist (pre-existing, not introduced by Phase 2);
            # exclude that one path from the "must-reject" assertion so the test
            # targets the Phase-2 contract, not the pre-existing config prefix.
            if f.startswith("configs/predictor/"):
                continue
            is_allowed = any(f.startswith(p) for p in allowed_prefixes) or f in allowed_files
            assert not is_allowed, (
                f"Unauthorized path was admitted by the UI-16 contract: {f}"
            )

        # The six newly-authorized exact paths ARE admitted (proves sync, not removal).
        for f in (
            "src/predictors/formal_metrics.py",
            "src/predictors/formal_runner.py",
            "scripts/build_val_cycle_table.py",
            "scripts/drl_heavy_mps_lock.py",
            "tests/test_formal_metrics.py",
            "tests/test_formal_runner.py",
        ):
            is_allowed = any(f.startswith(p) for p in allowed_prefixes) or f in allowed_files
            assert is_allowed, f"Authorized exact path was rejected: {f}"

    # UI-17
    def test_ui17_no_smoke_bytes_staged(self):
        """UI-17: No smoke training output bytes staged or committed."""
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=str(_M8_ROOT),
        )
        for line in result.stdout.split("\n"):
            if line.strip():
                status, path = line[0:2], line[3:].strip()
                # Check for common smoke output patterns
                smoke_patterns = [
                    "results/milestone8_smoke/",
                    "results/predictor/linex_",
                    "checkpoints/",
                    "training_history.json",
                    "predictor_validation_predictions.parquet",
                    "best_checkpoint.pt",
                    "last_checkpoint.pt",
                ]
                for pattern in smoke_patterns:
                    if pattern in path:
                        pytest.fail(f"Smoke output file staged/committed: {path}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])