"""
Scenario-bank provenance tests for Milestone 3.

Verifies that all required provenance fields are recorded:
- logical_bank_id
- source_path (repository-relative)
- source_file_size
- source_sha256
- source_scenario_count
- derived_k
- derived_cost_regime_id
- derived_scenario_count
- derived_scenario_ids (ordered)
- derived_bank_sha256
"""

import pytest
from pathlib import Path

from src.baselines.case_loader import load_cases, CaseLoadResult


class TestScenarioBankProvenanceFields:
    """Test all required provenance fields are present."""

    def test_load_cases_returns_all_provenance_fields(self):
        """load_cases should return all required provenance fields."""
        result = load_cases(
            split="rl_validation",
            k=1,
            cost_regime_id="failure-light-no-waste",
            source_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )

        # Required fields from CaseLoadResult
        assert result.split == "rl_validation"
        assert result.k == 1
        assert result.cost_regime_id == "failure-light-no-waste"
        assert isinstance(result.scenario_ids, tuple)
        assert len(result.scenario_ids) > 0

        # Provenance fields
        assert result.source_bank_path is not None
        assert result.derived_from_k == 2  # K=1 derived from K=2 source

        # New formal provenance fields
        assert result.logical_bank_id is not None
        assert result.source_file_size is not None
        assert result.source_file_size > 0
        assert result.bank_sha256 is not None
        assert len(result.bank_sha256) == 64  # SHA256 hex length
        assert result.bank_scenario_count is not None
        assert result.bank_scenario_count > 0
        assert result.derived_scenario_count is not None
        assert result.derived_scenario_count > 0
        assert result.derived_bank_sha256 is not None
        assert len(result.derived_bank_sha256) == 64  # SHA256 hex length

    def test_provenance_fields_are_json_serializable(self):
        """All provenance fields should be JSON-serializable."""
        import json

        result = load_cases(
            split="rl_validation",
            k=1,
            cost_regime_id="failure-light-no-waste",
            source_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )

        # Convert to dict for serialization check
        provenance_dict = {
            "logical_bank_id": result.logical_bank_id,
            "source_path": result.source_bank_path,
            "source_file_size": result.source_file_size,
            "source_sha256": result.bank_sha256,
            "source_scenario_count": result.bank_scenario_count,
            "derived_k": result.k,
            "derived_cost_regime_id": result.cost_regime_id,
            "derived_scenario_count": result.derived_scenario_count,
            "derived_scenario_ids": list(result.scenario_ids),
            "derived_bank_sha256": result.derived_bank_sha256,
        }

        # Should not raise
        json.dumps(provenance_dict)

    def test_derived_scenario_ids_are_ordered(self):
        """Derived scenario IDs should be deterministically ordered."""
        result1 = load_cases(
            split="rl_validation",
            k=1,
            cost_regime_id="failure-light-no-waste",
            source_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )

        result2 = load_cases(
            split="rl_validation",
            k=1,
            cost_regime_id="failure-light-no-waste",
            source_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )

        # Should be identical
        assert result1.scenario_ids == result2.scenario_ids

        # Should be sorted
        assert list(result1.scenario_ids) == sorted(result1.scenario_ids)

    def test_derived_bank_sha256_is_deterministic(self):
        """Derived bank SHA256 should be deterministic."""
        result1 = load_cases(
            split="rl_validation",
            k=1,
            cost_regime_id="failure-light-no-waste",
            source_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )

        result2 = load_cases(
            split="rl_validation",
            k=1,
            cost_regime_id="failure-light-no-waste",
            source_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )

        assert result1.derived_bank_sha256 == result2.derived_bank_sha256

    def test_source_sha256_matches_file_content(self):
        """Source SHA256 should match actual file content."""
        import hashlib

        result = load_cases(
            split="rl_validation",
            k=1,
            cost_regime_id="failure-light-no-waste",
            source_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )

        # Compute SHA256 independently
        with open(result.source_bank_path, "rb") as f:
            computed_sha256 = hashlib.sha256(f.read()).hexdigest()

        assert result.bank_sha256 == computed_sha256

    def test_source_file_size_matches_actual(self):
        """Source file size should match actual file size."""
        result = load_cases(
            split="rl_validation",
            k=1,
            cost_regime_id="failure-light-no-waste",
            source_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )

        # Get actual size
        actual_size = Path(result.source_bank_path).stat().st_size

        assert result.source_file_size == actual_size

    def test_derived_scenario_count_matches_ids(self):
        """Derived scenario count should match scenario IDs length."""
        result = load_cases(
            split="rl_validation",
            k=1,
            cost_regime_id="failure-light-no-waste",
            source_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )

        assert result.derived_scenario_count == len(result.scenario_ids)

    def test_k2_load_has_derived_from_k_none(self):
        """K=2 load should have derived_from_k=None (not derived)."""
        result = load_cases(
            split="rl_validation",
            k=2,
            cost_regime_id="failure-light-no-waste",
            source_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )

        assert result.derived_from_k is None

    def test_k1_load_has_derived_from_k_2(self):
        """K=1 load should have derived_from_k=2 (derived from K=2)."""
        result = load_cases(
            split="rl_validation",
            k=1,
            cost_regime_id="failure-light-no-waste",
            source_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )

        assert result.derived_from_k == 2

    def test_provenance_for_all_k_regime_combinations(self):
        """All K/regime combinations should have complete provenance."""
        k_values = [1, 2]
        cost_regimes = [
            "failure-light-no-waste",
            "failure-heavy-no-waste",
            "failure-light-waste-aware",
            "failure-heavy-waste-aware",
        ]

        for k in k_values:
            for regime in cost_regimes:
                result = load_cases(
                    split="rl_validation",
                    k=k,
                    cost_regime_id=regime,
                    source_bank_path="data/scenario_banks/rl_validation_smoke.json",
                )

                # Verify all fields present
                assert result.logical_bank_id is not None
                assert result.source_file_size > 0
                assert len(result.bank_sha256) == 64
                assert result.bank_scenario_count > 0
                assert result.derived_scenario_count > 0
                assert len(result.derived_bank_sha256) == 64

    def test_logical_bank_id_reflects_derivation(self):
        """Logical bank ID should reflect K and cost regime derivation."""
        result = load_cases(
            split="rl_validation",
            k=1,
            cost_regime_id="failure-light-no-waste",
            source_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )

        # Should contain k1 and regime in the ID
        assert "k1" in result.logical_bank_id.lower()
        assert "failure-light-no-waste" in result.logical_bank_id


class TestProvenanceToDict:
    """Test converting CaseLoadResult to provenance dict."""

    def test_case_load_result_to_provenance_dict(self):
        """CaseLoadResult should convert to provenance dict format."""
        result = load_cases(
            split="rl_validation",
            k=1,
            cost_regime_id="failure-light-no-waste",
            source_bank_path="data/scenario_banks/rl_validation_smoke.json",
        )

        # Convert to dict as done in CLI
        provenance_dict = {
            "logical_bank_id": result.logical_bank_id,
            "source_path": result.source_bank_path,
            "source_file_size": result.source_file_size,
            "source_sha256": result.bank_sha256,
            "source_scenario_count": result.bank_scenario_count,
            "derived_k": result.k,
            "derived_cost_regime_id": result.cost_regime_id,
            "derived_scenario_count": result.derived_scenario_count,
            "derived_scenario_ids": list(result.scenario_ids),
            "derived_bank_sha256": result.derived_bank_sha256,
        }

        # All required keys present
        required_keys = [
            "logical_bank_id",
            "source_path",
            "source_file_size",
            "source_sha256",
            "source_scenario_count",
            "derived_k",
            "derived_cost_regime_id",
            "derived_scenario_count",
            "derived_scenario_ids",
            "derived_bank_sha256",
        ]

        for key in required_keys:
            assert key in provenance_dict, f"Missing key: {key}"