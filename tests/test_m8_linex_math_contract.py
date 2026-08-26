"""
Mathematical contract verification tests for LinEx loss.

Verifies the exact mathematical properties defined in M8_LINEX_PREDICTOR_CONTRACT.
"""

import pytest
import torch
import numpy as np
from src.predictors.losses import LinExLoss


class TestLinExMathContract:
    """Mathematical contract verification (FROZEN per M8 contract)."""

    def test_zero_error_exact_zero_loss(self):
        """MC-01: L(0; a) = 0 exactly for all a > 0."""
        for a in [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
            loss_fn = LinExLoss(a=a, reduction="mean")
            y = torch.tensor([100.0, 100.0, 100.0])
            loss = loss_fn(y, y)
            assert loss.item() == 0.0, f"a={a}: L(0) = {loss.item()}, expected 0.0"

    def test_l_0_a_undefined_not_mse(self):
        """MC-02: L(error; 0) is UNDEFINED (a must be > 0). Not equal to MSE."""
        # LinExLoss constructor rejects a <= 0
        with pytest.raises(ValueError):
            LinExLoss(a=0.0)

        # Explicitly verify LinEx with small a is NOT MSE
        # MSE for error e is e^2
        # LinEx for small a: (a^2 * e^2)/2 + O(a^3)
        # These are NOT equal (differs by factor of a^2/2 and higher-order terms)
        a = 0.001
        loss_fn = LinExLoss(a=a, reduction="none")
        errors = torch.tensor([1.0, 2.0, 5.0, 10.0])
        y_pred = 100.0 + errors
        y_true = torch.full_like(y_pred, 100.0)

        linex_losses = loss_fn(y_pred, y_true)
        mse_losses = errors ** 2  # MSE with reduction='none'

        # They should NOT be equal (LinEx is not MSE even for small a)
        for i, (l, m) in enumerate(zip(linex_losses, mse_losses)):
            assert abs(l.item() - m.item()) > 1e-10, \
                f"error={errors[i].item()}: LinEx={l.item()}, MSE={m.item()} - should differ"

    def test_asymmetry_exact_formula(self):
        """MC-03: L(+|e|; a) > L(-|e|; a) for all a > 0, |e| > 0."""
        for a in [0.01, 0.1, 0.5, 1.0]:
            # Use higher overflow threshold for this test to avoid false failures
            loss_fn = LinExLoss(a=a, reduction="none", overflow_threshold=100.0)
            # Use errors that don't trigger overflow
            for e_mag in [0.1, 1.0, 5.0, 10.0, 20.0]:
                y_pred_pos = torch.tensor([100.0 + e_mag])
                y_pred_neg = torch.tensor([100.0 - e_mag])
                y_true = torch.tensor([100.0])

                loss_pos = loss_fn(y_pred_pos, y_true).item()
                loss_neg = loss_fn(y_pred_neg, y_true).item()

                assert loss_pos > loss_neg, \
                    f"a={a}, |e|={e_mag}: L(+|e|)={loss_pos} not > L(-|e|)={loss_neg}"

    def test_positive_error_heavier_than_negative(self):
        """MC-04: Positive error (overestimation) has heavier penalty."""
        loss_fn = LinExLoss(a=0.1, reduction="mean")
        # error = +10: dangerous overestimation
        # error = -10: conservative underestimation
        loss_over = loss_fn(torch.tensor([110.0]), torch.tensor([100.0])).item()
        loss_under = loss_fn(torch.tensor([90.0]), torch.tensor([100.0])).item()

        assert loss_over > loss_under, \
            f"Overestimation loss {loss_over} must exceed underestimation loss {loss_under}"

    def test_convexity(self):
        """MC-05: L(error; a) is strictly convex for a > 0."""
        # Second derivative: d²L/derror² = a² * exp(a * error) > 0 for all error
        a = 0.1
        loss_fn = LinExLoss(a=a, reduction="none")

        errors = torch.linspace(-20, 20, 401)
        y_true = torch.full_like(errors, 100.0)
        y_pred = 100.0 + errors

        losses = loss_fn(y_pred, y_true).numpy()

        # Check discrete convexity: f''(x) ≈ f(x+h) - 2f(x) + f(x-h) > 0
        h = errors[1] - errors[0]
        second_deriv = losses[2:] - 2*losses[1:-1] + losses[:-2]
        assert np.all(second_deriv > -1e-10), "Function not convex (numerical check failed)"

    def test_monotonic_in_abs_error(self):
        """MC-06: For fixed sign, loss increases with |error|."""
        loss_fn = LinExLoss(a=0.1, reduction="none")

        # Positive errors: larger error = larger loss
        pos_errors = torch.tensor([1.0, 5.0, 10.0, 20.0, 50.0])
        pos_losses = loss_fn(100.0 + pos_errors, torch.full_like(pos_errors, 100.0))
        assert torch.all(torch.diff(pos_losses) > 0), "Loss not monotonic in positive error"

        # Negative errors: larger |error| = larger loss
        neg_errors = torch.tensor([-1.0, -5.0, -10.0, -20.0, -50.0])
        neg_losses = loss_fn(100.0 + neg_errors, torch.full_like(neg_errors, 100.0))
        assert torch.all(torch.diff(neg_losses) > 0), "Loss not monotonic in |negative error|"

    def test_limit_a_to_zero_behavior(self):
        """MC-07: As a→0, L(error; a) / a² → error²/2 (local quadratic)."""
        # This is the local expansion, not identity with MSE
        a = 0.001
        loss_fn = LinExLoss(a=a, reduction="none")

        errors = torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0])
        y_pred = 100.0 + errors
        y_true = torch.full_like(errors, 100.0)

        linex_losses = loss_fn(y_pred, y_true).numpy()
        quadratic_approx = (a**2 * errors.numpy()**2) / 2

        # Relative error should be small for small a*error
        rel_errors = np.abs(linex_losses - quadratic_approx) / (np.abs(quadratic_approx) + 1e-12)
        assert np.all(rel_errors < 0.01), f"Expansion mismatch: max rel error = {np.max(rel_errors)}"

    def test_formula_matches_math(self):
        """MC-08: Implementation matches L(e; a) = exp(a*e) - a*e - 1 exactly."""
        a = 0.1
        loss_fn = LinExLoss(a=a, reduction="none")

        errors = torch.tensor([-50.0, -10.0, -1.0, 0.0, 1.0, 10.0, 50.0])
        y_pred = 100.0 + errors
        y_true = torch.full_like(errors, 100.0)

        computed = loss_fn(y_pred, y_true).numpy()

        # Manual formula - use float64 for expected to match computation precision
        expected = np.exp(a * errors.numpy().astype(np.float64)) - a * errors.numpy().astype(np.float64) - 1

        # Compare with appropriate tolerance for float32
        assert np.allclose(computed, expected, rtol=1e-5, atol=1e-6), \
            f"Formula mismatch: computed={computed}, expected={expected}"

    def test_expm1_equivalence(self):
        """MC-09: expm1(z) - z equals exp(z) - z - 1 for numerical stability."""
        a = 0.1
        loss_fn = LinExLoss(a=a, reduction="none")

        errors = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0])
        y_pred = 100.0 + errors
        y_true = torch.full_like(errors, 100.0)

        computed = loss_fn(y_pred, y_true).numpy()

        # Using expm1 (float64 for reference)
        z = a * errors.numpy().astype(np.float64)
        expm1_form = np.expm1(z) - z

        # Using exp (float64 for reference)
        exp_form = np.exp(z) - z - 1

        # Compare with appropriate tolerance for float32
        assert np.allclose(computed, expm1_form, rtol=1e-5, atol=1e-6)
        assert np.allclose(computed, exp_form, rtol=1e-5, atol=1e-6)

    def test_a_zero_not_mse_baseline(self):
        """MC-10: a=0 is explicitly NOT the MSE baseline (rejected at construction)."""
        # Contract: "Do not claim that a=0 equals MSE"
        # LinExLoss(a=0) raises ValueError - this is the correct behavior
        with pytest.raises(ValueError, match="must be > 0"):
            LinExLoss(a=0.0)

        # MSE is a SEPARATE loss function, not a limit case
        mse_fn = torch.nn.MSELoss()
        linex_fn = LinExLoss(a=0.001, reduction="mean")

        y_pred = torch.tensor([110.0, 90.0, 105.0, 95.0])
        y_true = torch.tensor([100.0, 100.0, 100.0, 100.0])

        mse_loss = mse_fn(y_pred, y_true).item()
        linex_loss = linex_fn(y_pred, y_true).item()

        # These are fundamentally different losses
        assert mse_loss != linex_loss, "MSE and LinEx should produce different values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])