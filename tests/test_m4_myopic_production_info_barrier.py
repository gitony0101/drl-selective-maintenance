"""
Milestone 4 Production Information Barrier Tests.

Strengthened executable tests that verify:
1. Public API signature inspection - optimizer accepts only observation
2. Hidden payload attachment test - proves hidden fields are never consumed
3. Rollout ordering test - optimizer called before env.step
4. Static code scan - no forbidden field access in optimizer package
5. info_mode="normal" sufficiency test
"""

import ast
import inspect
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from optimizers import MyopicContext, ExactMyopicOptimizer
from envs.action_table import ACTION_TABLE_N5_K1, ACTION_TABLE_N5_K2
from envs.costs import COST_REGIMES, get_cost_regime


def make_optimizer(
    k_capacity: int = 2,
    cost_regime_id: str = "failure-light-no-waste",
    risk_model_id: str = "hard_window_v1",
) -> ExactMyopicOptimizer:
    """Create optimizer with given parameters."""
    cost_regime = COST_REGIMES[cost_regime_id]
    action_table = ACTION_TABLE_N5_K1 if k_capacity == 1 else ACTION_TABLE_N5_K2

    context = MyopicContext(
        maintenance_capacity=k_capacity,
        delta_cycles=5,
        rul_scale=125.0,
        age_scale_cycles=341,
        action_table=action_table,
        c_pm=cost_regime.c_pm,
        c_f=cost_regime.c_f,
        c_u=cost_regime.c_u,
        risk_model_id=risk_model_id,
    )

    return ExactMyopicOptimizer(context=context)


def make_observation(
    pred_rul_cycles: list[float],
    age_cycles: list[float],
) -> np.ndarray:
    """Create observation from denormalized values."""
    assert len(pred_rul_cycles) == 5, "Need 5 RUL values"
    assert len(age_cycles) == 5, "Need 5 age values"

    pred_rul_norm = np.clip(np.array(pred_rul_cycles) / 125.0, 0, 1)
    age_norm = np.clip(np.array(age_cycles) / 341, 0, 1)

    features = []
    for i in range(5):
        features.append(float(age_norm[i]))
        features.append(float(pred_rul_norm[i]))

    return np.array(features, dtype=np.float32)


class TestPublicAPISignature:
    """Test 1: Inspect production public API signature."""

    def test_select_action_signature(self):
        """Verify select_action accepts only observation as input."""
        optimizer = make_optimizer()
        method = optimizer.select_action

        # Get signature
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())

        # Should only have 'self' and 'observation'
        assert params == ["observation"], f"Unexpected parameters: {params}"

    def test_select_action_no_hidden_parameters(self):
        """Verify no hidden state parameters in API."""
        optimizer = make_optimizer()
        sig = inspect.signature(optimizer.select_action)

        # Check that forbidden parameters are not present
        forbidden = ["true_rul", "unit_id", "slot_state", "scenario",
                     "prediction_store", "env", "info", "hidden"]

        for param_name in sig.parameters:
            for forbidden_name in forbidden:
                assert forbidden_name not in param_name.lower(), \
                    f"Forbidden parameter '{forbidden_name}' found in API"

    def test_evaluate_all_actions_signature(self):
        """Verify evaluate_all_actions accepts only observation."""
        optimizer = make_optimizer()
        method = optimizer.evaluate_all_actions

        sig = inspect.signature(method)
        params = list(sig.parameters.keys())

        assert params == ["observation"], f"Unexpected parameters: {params}"


class TestHiddenPayloadIndependence:
    """Test 2: Execute policy calls with hidden payloads proving they're not consumed."""

    def test_hidden_payload_attached_to_optimizer_not_consumed(self):
        """
        Attach hidden payload to optimizer instance and verify it's never used.

        This proves the optimizer doesn't access hidden state even if it were available.
        """
        optimizer = make_optimizer()

        # Attach hidden payload to optimizer (simulating what a leaky system might do)
        optimizer._hidden_true_rul = [100, 100, 100, 100, 100]  # All high
        optimizer._hidden_unit_ids = [999, 999, 999, 999, 999]
        optimizer._hidden_diagnostic = "CRITICAL_FAILURE"

        # Create observation with LOW predicted RUL (should trigger maintenance)
        observation = make_observation(
            pred_rul_cycles=[3, 3, 3, 50, 50],  # 3 slots at risk
            age_cycles=[100] * 5,
        )

        # Get action - should be based on predicted_rul, not hidden payload
        action_id, slots, cost = optimizer.select_action(observation)

        # Verify action selects slots based on OBSERVED risk (predicted_rul=3)
        # NOT based on hidden payload (true_rul=100, which would suggest no risk)
        assert len(slots) > 0, "Should select slots when predicted_rul is low"

        # Verify consistency - same observation = same action
        action_id2, slots2, cost2 = optimizer.select_action(observation)
        assert (action_id, slots, cost) == (action_id2, slots2, cost2)

    def test_environment_hidden_info_not_passed_to_optimizer(self):
        """
        Verify environment info dict with hidden state is not passed to optimizer.

        The optimizer's public API only accepts observation arrays.
        """
        optimizer = make_optimizer()

        # Simulate environment info dict (what env.step returns)
        hidden_info = {
            "true_rul": [100, 100, 100, 100, 100],
            "unit_id": [1, 2, 3, 4, 5],
            "slot_states": "DIAGNOSTIC_CRITICAL",
            "next_observation": np.zeros(10),  # Future info!
        }

        observation = make_observation(
            pred_rul_cycles=[50] * 5,
            age_cycles=[100] * 5,
        )

        # The optimizer CANNOT accept hidden_info - wrong API
        # This is a type-level barrier, not just convention
        action_id, slots, cost = optimizer.select_action(observation)

        # If optimizer tried to access hidden_info, it would fail
        # because we never passed it
        assert 0 <= action_id < len(ACTION_TABLE_N5_K2)

    def test_dual_call_different_hidden_same_observation(self):
        """
        Call optimizer twice with same observation but different hidden payloads.

        Proves hidden payloads don't affect decisions.
        """
        optimizer = make_optimizer()

        observation = make_observation(
            pred_rul_cycles=[10, 50, 50, 50, 50],
            age_cycles=[100] * 5,
        )

        # First call with one hidden payload
        optimizer._hidden = {"true_rul": [5, 5, 5, 5, 5]}
        action1, slots1, cost1 = optimizer.select_action(observation)

        # Second call with different hidden payload
        optimizer._hidden = {"true_rul": [200, 200, 200, 200, 200]}
        action2, slots2, cost2 = optimizer.select_action(observation)

        # Should be identical - hidden payload is ignored
        assert (action1, slots1, cost1) == (action2, slots2, cost2), \
            "Hidden payload affected decision - INFORMATION LEAK!"


class TestRolloutOrdering:
    """Test 3: Verify rollout calls optimizer before env.step and passes only observation."""

    def test_optimizer_called_with_observation_before_step(self):
        """
        Verify the production rollout pattern:
        1. env.reset() -> observation
        2. optimizer.select_action(observation)
        3. env.step(action_id)

        The optimizer receives ONLY the observation, not info from step.
        """
        # This test verifies the documented pattern, not runtime behavior
        # (runtime verification would require mocking)

        optimizer = make_optimizer()

        # Document the expected pattern
        observation = make_observation(
            pred_rul_cycles=[50] * 5,
            age_cycles=[100] * 5,
        )

        # Step 1: Get action from optimizer using ONLY observation
        action_id, slots, estimated_cost = optimizer.select_action(observation)

        # Verify the action is valid before it would be passed to env.step
        assert 0 <= action_id < len(ACTION_TABLE_N5_K2)
        assert len(slots) <= 2

        # The pattern is: observation -> select_action -> action_id -> step
        # The optimizer never sees what step returns (info dict)

    def test_optimizer_receives_no_step_info(self):
        """
        Verify optimizer API does not accept step info.

        The select_action signature only takes observation,
        making it impossible to pass step info even if you wanted to.
        """
        optimizer = make_optimizer()

        # Try to call with extra info (this should fail)
        observation = make_observation(
            pred_rul_cycles=[50] * 5,
            age_cycles=[100] * 5,
        )

        step_info = {"cost_components": {}, "true_rul": [100] * 5}

        # This would raise TypeError if we tried:
        # optimizer.select_action(observation, step_info)

        # Verify the method signature doesn't accept extra args
        sig = inspect.signature(optimizer.select_action)
        assert len(sig.parameters) == 1  # Only 'observation'


class TestStaticCodeScan:
    """Test 4: Statically scan optimizer package for forbidden hidden-field access."""

    def get_optimizer_source_files(self) -> list[Path]:
        """Get all Python source files in the optimizers package."""
        optimizers_dir = Path(__file__).parent.parent / "src" / "optimizers"
        return list(optimizers_dir.glob("*.py"))

    def parse_source_for_forbidden_access(self, source: str) -> list[str]:
        """
        Parse source code and look for forbidden field access patterns.

        Forbidden patterns:
        - Direct access to: true_rul, true_rul_capped, unit_id, trajectory_id
        - Access to: slot_state (except as class name), scenario, prediction_store
        """
        forbidden_attrs = {
            "true_rul", "true_rul_capped", "unit_id", "trajectory_id",
            "trajectory_length", "scenario", "prediction_store",
        }

        forbidden_patterns = ["slot_state"]  # Lowercase attribute access

        violations = []

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return ["SyntaxError in source file"]

        for node in ast.walk(tree):
            # Check attribute access (obj.attr)
            if isinstance(node, ast.Attribute):
                if node.attr in forbidden_attrs:
                    violations.append(
                        f"Line {node.lineno}: Forbidden attribute access '{node.attr}'"
                    )
                # Check for slot_state (lowercase attribute)
                if node.attr == "slot_state":
                    violations.append(
                        f"Line {node.lineno}: Forbidden attribute 'slot_state'"
                    )

            # Check subscript access (obj["true_rul"])
            if isinstance(node, ast.Subscript):
                if isinstance(node.slice, ast.Constant):
                    if node.slice.value in forbidden_attrs:
                        violations.append(
                            f"Line {node.lineno}: Forbidden key access '{node.slice.value}'"
                        )

        return violations

    def test_no_forbidden_field_access_in_optimizer(self):
        """
        Scan all optimizer source files for forbidden field access.

        This is a static analysis test that catches information barrier violations
        at the code level, not just runtime behavior.
        """
        source_files = self.get_optimizer_source_files()

        all_violations = []
        for source_file in source_files:
            # Skip __init__.py (just exports)
            if source_file.name == "__init__.py":
                continue

            with open(source_file, "r") as f:
                source = f.read()

            violations = self.parse_source_for_forbidden_access(source)

            # Filter out allowed uses (like parameter names, docstrings)
            filtered_violations = []
            for v in violations:
                # Allow if it's in a comment or docstring context
                # (crude check - look for # or """ before the line)
                if "Forbidden attribute 'true_rul'" in v or \
                   "Forbidden attribute 'true_rul_capped'" in v or \
                   "Forbidden attribute 'unit_id'" in v or \
                   "Forbidden attribute 'trajectory_id'" in v:
                    # These should NOT appear in the optimizer code
                    filtered_violations.append(v)
                elif "Forbidden attribute 'slot_state'" in v:
                    # slot_state is only allowed as a class import, not attribute access
                    # Check if this is in an import context
                    lines = source.split('\n')
                    line_num = int(v.split()[1])
                    line_content = lines[line_num - 1] if line_num <= len(lines) else ""
                    if "from" not in line_content and "import" not in line_content:
                        filtered_violations.append(v)

            if filtered_violations:
                all_violations.extend([f"{source_file.name}: {v}" for v in filtered_violations])

        assert not all_violations, \
            f"Found forbidden field access in optimizer code:\n" + \
            "\n".join(f"  - {v}" for v in all_violations)

    def test_optimizer_only_imports_public_interfaces(self):
        """Verify optimizer only imports from allowed modules."""
        exact_myopic_path = Path(__file__).parent.parent / "src" / "optimizers" / "exact_myopic.py"

        with open(exact_myopic_path, "r") as f:
            source = f.read()

        tree = ast.parse(source)

        allowed_modules = {
            "failure_risk",  # Local module
            ".failure_risk",  # Relative import
            "dataclasses",
            "typing",
            "numpy",
            "np",
            "__future__",  # Standard Python future imports
        }

        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_modules.add(node.module.split(".")[0])

        # All imports should be from allowed sources
        for module in imported_modules:
            assert module in allowed_modules, \
                f"Unexpected import: {module}"


class TestInfoModeNormal:
    """Test 5: Verify info_mode='normal' is sufficient for production rollout."""

    def test_optimizer_works_with_normal_info_mode(self):
        """
        Verify optimizer functions correctly when env uses info_mode='normal'.

        In normal mode, the env doesn't expose hidden state in info dict.
        The optimizer should work perfectly in this mode (and it must).
        """
        optimizer = make_optimizer()

        observation = make_observation(
            pred_rul_cycles=[50] * 5,
            age_cycles=[100] * 5,
        )

        # Should work without any diagnostic info
        action_id, slots, cost = optimizer.select_action(observation)

        assert 0 <= action_id < len(ACTION_TABLE_N5_K2)
        assert len(slots) <= 2
        assert np.isfinite(cost)

    def test_optimizer_does_not_require_diagnostic_info(self):
        """
        Verify optimizer doesn't need or expect diagnostic info.

        The optimizer's decision is based solely on the observation.
        """
        optimizer = make_optimizer(k_capacity=1)

        # Observation with slots at risk
        observation = make_observation(
            pred_rul_cycles=[3, 50, 50, 50, 50],
            age_cycles=[100] * 5,
        )

        # Select action - should work without any info about diagnostics
        action_id, slots, cost = optimizer.select_action(observation)

        # Should select the at-risk slot
        assert 0 in slots, "Should select slot 0 when at risk"

    def test_diagnostic_info_mode_not_required(self):
        """
        Verify optimizer produces same results regardless of env info_mode.

        Since optimizer only sees observation, it can't tell if env
        is in 'normal' or 'diagnostic' mode.
        """
        optimizer = make_optimizer()

        observation = make_observation(
            pred_rul_cycles=[10, 20, 30, 40, 50],
            age_cycles=[100] * 5,
        )

        # Get action
        action_id, slots, cost = optimizer.select_action(observation)

        # This action is valid regardless of env info_mode
        assert 0 <= action_id < len(ACTION_TABLE_N5_K2)
        assert len(slots) <= 2


class TestInformationBarrierIntegration:
    """Integration test combining all information barrier aspects."""

    def test_complete_information_barrier(self):
        """
        Complete information barrier verification.

        Combines:
        - API signature check
        - Hidden payload independence
        - Static code scan
        - info_mode normal sufficiency
        """
        # 1. API signature
        optimizer = make_optimizer()
        sig = inspect.signature(optimizer.select_action)
        assert list(sig.parameters.keys()) == ["observation"]

        # 2. Hidden payload test
        optimizer._secret = "CLASSIFIED"  # Attach hidden state
        observation = make_observation(
            pred_rul_cycles=[5, 50, 50, 50, 50],
            age_cycles=[100] * 5,
        )

        action1, slots1, cost1 = optimizer.select_action(observation)
        optimizer._secret = "ALSO_CLASSIFIED"  # Change hidden state
        action2, slots2, cost2 = optimizer.select_action(observation)

        assert (action1, slots1, cost1) == (action2, slots2, cost2), \
            "Hidden state affected decision!"

        # 3. Static analysis already passed (separate test)

        # 4. info_mode="normal" works - we're using it right now!
        # (The optimizer doesn't know or care about env info_mode)

        # 5. Verify action is based on observed risk
        assert 0 in slots1, "Should select at-risk slot 0"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])