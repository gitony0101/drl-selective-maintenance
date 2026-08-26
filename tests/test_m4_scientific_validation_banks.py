"""
Tests for M4 Scientific Validation Bank Generator.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.requires_external_assets

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scripts.generate_m4_scientific_validation_banks import (
    generate_scientific_validation_banks,
    build_continuity_map,
    compute_quantile_strata,
    select_scenario_from_strata,
    load_prediction_cache,
    get_cache_sha256,
    ScientificValidationBankError,
    PROTOCOL_VERSION,
    SELECTION_BASIS,
    SPLITS,
    K_VALUES,
    COST_REGIMES,
    SEEDS,
    SCENARIOS_PER_BANK,
    UNITS_PER_SCENARIO,
    RUL_SCALE,
)
from envs.scenario_bank import load_scenario_bank, ScenarioBank


class TestBankGenerator:
    """Test the scientific validation bank generator."""

    @pytest.fixture
    def cache_path(self):
        """Path to prediction cache."""
        repo_root = Path(__file__).parent.parent
        return repo_root / "data" / "processed" / "fd001" / "v2" / "06_PREDICTIONS" / "fd001_prediction_cache_v2.parquet"

    def test_load_prediction_cache(self, cache_path):
        """Cache loading works and has required columns."""
        cache = load_prediction_cache(cache_path)
        assert len(cache) > 0
        required = ['split', 'unit_id', 'cycle', 'predicted_rul_normalized']
        for col in required:
            assert col in cache.columns
        assert 'predicted_rul_cycles' in cache.columns

    def test_get_cache_sha256(self, cache_path):
        """Cache SHA256 is computed correctly."""
        sha = get_cache_sha256(cache_path)
        assert len(sha) == 64
        assert all(c in '0123456789abcdef' for c in sha)

    def test_build_continuity_map(self, cache_path):
        """Continuity map built correctly for each split."""
        cache = load_prediction_cache(cache_path)

        for split in SPLITS:
            cont_map = build_continuity_map(cache, split)
            assert len(cont_map) >= UNITS_PER_SCENARIO
            for unit_id, cycles in cont_map.items():
                assert len(cycles) > 0
                for cycle, pred_rul in cycles:
                    assert isinstance(cycle, int)
                    assert isinstance(pred_rul, float)
                    assert np.isfinite(pred_rul)
                    assert pred_rul >= 0

    def test_compute_quantile_strata(self, cache_path):
        """Quantile stratification produces 5 strata with units."""
        cache = load_prediction_cache(cache_path)
        cont_map = build_continuity_map(cache, "rl_validation")
        strata, boundaries = compute_quantile_strata(cont_map)

        assert len(strata) == 5
        assert len(boundaries) == 6
        for i in range(5):
            assert len(strata[i]) > 0, f"Stratum {i} empty"
        for i in range(1, 6):
            assert boundaries[i] >= boundaries[i-1]

    def test_select_scenario_from_strata(self, cache_path):
        """Scenario selection picks one unit from each stratum."""
        cache = load_prediction_cache(cache_path)
        cont_map = build_continuity_map(cache, "rl_validation")
        strata, _ = compute_quantile_strata(cont_map)

        for seed in [6601, 6602, 6610, 6620]:
            selected = select_scenario_from_strata(strata, seed, 0)
            assert len(selected) == UNITS_PER_SCENARIO
            unit_ids = [s[0] for s in selected]
            assert len(set(unit_ids)) == UNITS_PER_SCENARIO  # All distinct

            # Verify one from each stratum
            strata_found = set()
            for unit_id, cycle, _ in selected:
                for s_idx in range(5):
                    for (uid, cyc, _) in strata[s_idx]:
                        if uid == unit_id and cyc == cycle:
                            strata_found.add(s_idx)
                            break
            assert len(strata_found) == 5, f"Missing strata: {set(range(5)) - strata_found}"

    def test_generate_all_banks(self, tmp_path):
        """Full generation produces all 16 banks with correct structure."""
        generate_scientific_validation_banks(tmp_path)

        # Check manifest
        manifest_path = tmp_path / "SCIENTIFIC_VALIDATION_BANK_MANIFEST.json"
        assert manifest_path.exists()
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest['protocol_version'] == PROTOCOL_VERSION
        assert manifest['selection_basis'] == SELECTION_BASIS
        assert manifest['no_rl_test_declaration'] is True
        assert manifest['splits'] == SPLITS
        assert manifest['k_values'] == K_VALUES
        assert manifest['cost_regimes'] == COST_REGIMES
        assert manifest['seeds'] == SEEDS
        assert manifest['scenarios_per_bank'] == SCENARIOS_PER_BANK
        assert manifest['units_per_scenario'] == UNITS_PER_SCENARIO
        assert len(manifest['banks']) == 16

        # Check each bank
        for bank_info in manifest['banks']:
            bank_file = tmp_path / bank_info['file']
            assert bank_file.exists()
            bank = load_scenario_bank(bank_file)
            assert bank.bank_id == bank_info['bank_id']
            assert bank.split == bank_info['split']
            assert len(bank.scenarios) == SCENARIOS_PER_BANK

            # Check metadata from manifest
            bank_metadata = manifest.get('bank_metadata', {}).get(bank_info['file'], {})
            assert bank_metadata.get('protocol_version') == PROTOCOL_VERSION
            assert bank_metadata.get('selection_basis') == SELECTION_BASIS
            assert bank_metadata.get('no_rl_test_declaration') is True
            assert bank_metadata.get('ordered_seeds') == SEEDS

            # Check scenarios
            for i, scenario in enumerate(bank.scenarios):
                assert scenario.scenario_id == f"{bank_info['split']}_K{bank_info['K']}_{bank_info['regime']}_{i:03d}"
                assert scenario.split == bank_info['split']
                assert scenario.maintenance_capacity == bank_info['K']
                assert scenario.cost_regime_id == bank_info['regime']
                assert scenario.replacement_seed == SEEDS[i]
                assert scenario.environment_seed == SEEDS[i]
                assert len(scenario.initial_unit_ids) == UNITS_PER_SCENARIO
                assert len(set(scenario.initial_unit_ids)) == UNITS_PER_SCENARIO

    def test_deterministic_generation(self, tmp_path):
        """Two generations produce identical files."""
        dir1 = tmp_path / "gen1"
        dir2 = tmp_path / "gen2"

        generate_scientific_validation_banks(dir1)
        generate_scientific_validation_banks(dir2)

        files1 = sorted([f for f in dir1.glob("*.json") if f.name != "SCIENTIFIC_VALIDATION_BANK_MANIFEST.json"])
        files2 = sorted([f for f in dir2.glob("*.json") if f.name != "SCIENTIFIC_VALIDATION_BANK_MANIFEST.json"])
        assert len(files1) == len(files2) == 16

        for f1, f2 in zip(files1, files2):
            with open(f1, 'rb') as fp1, open(f2, 'rb') as fp2:
                assert fp1.read() == fp2.read(), f"Non-deterministic: {f1.name}"

    def test_fail_closed_on_missing_cache(self, tmp_path):
        """Generator fails cleanly if cache missing."""
        bad_cache = tmp_path / "nonexistent.parquet"
        # This is tested via the internal error handling
        with pytest.raises(ScientificValidationBankError) as exc:
            from scripts.generate_m4_scientific_validation_banks import load_prediction_cache
            load_prediction_cache(bad_cache)
        assert "not found" in str(exc.value).lower()

    def test_fail_closed_on_rl_test(self, tmp_path):
        """Generator fails if asked for rl_test split."""
        repo_root = Path(__file__).parent.parent
        cache_path = repo_root / "data" / "processed" / "fd001" / "v2" / "06_PREDICTIONS" / "fd001_prediction_cache_v2.parquet"
        cache = load_prediction_cache(cache_path)

        with pytest.raises(ScientificValidationBankError) as exc:
            build_continuity_map(cache, "rl_test")
        assert "forbidden" in str(exc.value).lower()

    def test_fail_closed_insufficient_units(self, tmp_path):
        """Generator fails if insufficient units with continuity."""
        # This is tested by the validator - generator requires 5 units per scenario
        pass


class TestBankContents:
    """Test detailed bank content requirements."""

    @pytest.fixture
    def generated_banks(self, tmp_path):
        """Generate banks for testing."""
        generate_scientific_validation_banks(tmp_path)
        return tmp_path

    def test_pair_id_stability(self, generated_banks):
        """Same (unit_ids, cycles) produces same pair_id across banks."""
        from envs.scenario_bank import load_scenario_bank

        manifest_path = generated_banks / "SCIENTIFIC_VALIDATION_BANK_MANIFEST.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        pair_map = {}  # (unit_ids, cycles) -> set of pair_ids

        for bank_info in manifest['banks']:
            bank_file = generated_banks / bank_info['file']
            bank = load_scenario_bank(bank_file)
            metadata = manifest.get('bank_metadata', {}).get(bank_info['file'], {})
            pair_ids = metadata.get('pair_ids', [])
            for scenario, pair_id in zip(bank.scenarios, pair_ids):
                key = tuple(zip(scenario.initial_unit_ids, scenario.initial_cycles))
                if key not in pair_map:
                    pair_map[key] = set()
                pair_map[key].add(pair_id)

        # Each unique unit/cycle combination should map to exactly one pair_id
        for key, pair_ids in pair_map.items():
            assert len(pair_ids) == 1, f"Pair ID not stable for {key}: {pair_ids}"

    def test_same_units_cycles_across_k_regimes(self, generated_banks):
        """Same split should have same unit/cycle pairs across K and regimes."""
        from envs.scenario_bank import load_scenario_bank

        # Group by split
        split_pairs = {split: {} for split in SPLITS}

        for bank_file in generated_banks.glob("*.json"):
            if bank_file.name == "SCIENTIFIC_VALIDATION_BANK_MANIFEST.json":
                continue
            bank = load_scenario_bank(bank_file)
            split = bank.split
            for scenario in bank.scenarios:
                key = tuple(zip(scenario.initial_unit_ids, scenario.initial_cycles))
                config_key = (bank.split, bank.scenarios[0].maintenance_capacity, bank.scenarios[0].cost_regime_id)
                if key not in split_pairs[split]:
                    split_pairs[split][key] = []
                split_pairs[split][key].append(config_key)

        # For each split, same unit/cycle pairs should appear in all K×regime combos
        for split in SPLITS:
            pairs = split_pairs[split]
            # Should have 20 unique pairs (one per seed)
            assert len(pairs) == SCENARIOS_PER_BANK
            for key, configs in pairs.items():
                # Each pair should appear in all 8 configs (2 K × 4 regimes)
                assert len(configs) == 8, f"Pair {key} only in {len(configs)} configs: {configs}"

    def test_quantile_boundaries_recorded(self, generated_banks):
        """Quantile boundaries recorded in each bank."""
        from envs.scenario_bank import load_scenario_bank

        manifest_path = generated_banks / "SCIENTIFIC_VALIDATION_BANK_MANIFEST.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        for bank_info in manifest['banks']:
            bank_file = generated_banks / bank_info['file']
            bank = load_scenario_bank(bank_file)
            metadata = manifest.get('bank_metadata', {}).get(bank_info['file'], {})
            assert 'quantile_boundaries' in metadata
            boundaries = metadata['quantile_boundaries']
            assert len(boundaries) == 6
            for i in range(5):
                assert boundaries[i] <= boundaries[i+1]

    def test_generation_version(self, generated_banks):
        """Generation version recorded."""
        manifest_path = generated_banks / "SCIENTIFIC_VALIDATION_BANK_MANIFEST.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        for bank_info in manifest['banks']:
            metadata = manifest.get('bank_metadata', {}).get(bank_info['file'], {})
            assert metadata.get('generation_version') == PROTOCOL_VERSION


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])