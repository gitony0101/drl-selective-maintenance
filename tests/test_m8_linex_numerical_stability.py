"""
Numerical stability tests for LinEx loss.

Verifies edge cases, overflow handling, NaN/Inf behavior, and cross-device consistency.
"""

import pytest
import torch
import numpy as np
from src.predictors.losses import LinExLoss


class TestLinExNumericalStability:
    """Numerical stability tests (FROZEN per M8 contract)."""

    def test_large_positive_error_overflow_guard(self):
        """NS-01: Large positive error triggers overflow guard."""
        loss_fn = LinExLoss(a=1.0, overflow_threshold=20.0)
        # error = 30, a*error = 30 > 20 → should raise
        y_pred = torch.tensor([130.0])
        y_true = torch.tensor([100.0])

        with pytest.raises(RuntimeError, match="overflow"):
            loss_fn(y_pred, y_true)

    def test_large_negative_error_overflow_guard(self):
        """NS-02: Large negative error also triggers overflow guard."""
        loss_fn = LinExLoss(a=1.0, overflow_threshold=20.0)
        # error = -30, |a*error| = 30 > 20 → should raise
        y_pred = torch.tensor([70.0])
        y_true = torch.tensor([100.0])

        with pytest.raises(RuntimeError, match="overflow"):
            loss_fn(y_pred, y_true)

    def test_overflow_threshold_configurable(self):
        """NS-03: overflow_threshold is configurable."""
        # With higher threshold, same inputs should work
        loss_fn = LinExLoss(a=1.0, overflow_threshold=50.0)
        y_pred = torch.tensor([130.0])
        y_true = torch.tensor([100.0])

        loss = loss_fn(y_pred, y_true)
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_nan_in_y_pred_fails_closed(self):
        """NS-04: NaN in y_pred raises RuntimeError."""
        loss_fn = LinExLoss(a=0.1)
        y_pred = torch.tensor([float('nan')])
        y_true = torch.tensor([100.0])

        with pytest.raises(RuntimeError, match="NaN|Inf"):
            loss_fn(y_pred, y_true)

    def test_nan_in_y_true_fails_closed(self):
        """NS-05: NaN in y_true raises RuntimeError."""
        loss_fn = LinExLoss(a=0.1)
        y_pred = torch.tensor([110.0])
        y_true = torch.tensor([float('nan')])

        with pytest.raises(RuntimeError, match="NaN|Inf"):
            loss_fn(y_pred, y_true)

    def test_inf_in_y_pred_fails_closed(self):
        """NS-06: +Inf in y_pred raises RuntimeError."""
        loss_fn = LinExLoss(a=0.1)
        y_pred = torch.tensor([float('inf')])
        y_true = torch.tensor([100.0])

        with pytest.raises(RuntimeError, match="NaN|Inf"):
            loss_fn(y_pred, y_true)

    def test_neg_inf_in_y_pred_fails_closed(self):
        """NS-07: -Inf in y_pred raises RuntimeError."""
        loss_fn = LinExLoss(a=0.1)
        y_pred = torch.tensor([float('-inf')])
        y_true = torch.tensor([100.0])

        with pytest.raises(RuntimeError, match="NaN|Inf"):
            loss_fn(y_pred, y_true)

    def test_inf_in_y_true_fails_closed(self):
        """NS-08: Inf in y_true raises RuntimeError."""
        loss_fn = LinExLoss(a=0.1)
        y_pred = torch.tensor([110.0])
        y_true = torch.tensor([float('inf')])

        with pytest.raises(RuntimeError, match="NaN|Inf"):
            loss_fn(y_pred, y_true)

    def test_output_finite_for_valid_inputs(self):
        """NS-09: Valid inputs produce finite loss."""
        loss_fn = LinExLoss(a=0.1)
        y_pred = torch.linspace(50.0, 150.0, 100)
        y_true = torch.full_like(y_pred, 100.0)

        loss = loss_fn(y_pred, y_true)
        assert torch.isfinite(loss).all(), "Non-finite loss for valid inputs"

    def test_no_input_mutation(self):
        """NS-10: Input tensors are not modified."""
        loss_fn = LinExLoss(a=0.1)
        y_pred = torch.tensor([110.0, 90.0, 105.0])
        y_true = torch.tensor([100.0, 100.0, 100.0])
        y_pred_orig = y_pred.clone()
        y_true_orig = y_true.clone()

        loss_fn(y_pred, y_true)

        assert torch.equal(y_pred, y_pred_orig), "y_pred was modified"
        assert torch.equal(y_true, y_true_orig), "y_true was modified"

    def test_cpu_reproducibility(self):
        """NS-11: CPU results are deterministic."""
        torch.manual_seed(42)

        loss_fn = LinExLoss(a=0.1)
        y_pred = torch.randn(10) * 10 + 100
        y_true = torch.full_like(y_pred, 100.0)

        loss1 = loss_fn(y_pred, y_true)
        loss2 = loss_fn(y_pred, y_true)

        assert loss1.item() == loss2.item(), "CPU results not reproducible"

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_mps_cpu_consistency(self):
        """NS-12: MPS results match CPU within tolerance."""
        torch.manual_seed(42)

        # CPU
        loss_fn_cpu = LinExLoss(a=0.1).to("cpu")
        y_pred_cpu = torch.randn(10) * 10 + 100
        y_true_cpu = torch.full_like(y_pred_cpu, 100.0)
        loss_cpu = loss_fn_cpu(y_pred_cpu, y_true_cpu).item()

        # MPS
        loss_fn_mps = LinExLoss(a=0.1).to("mps")
        y_pred_mps = y_pred_cpu.to("mps")
        y_true_mps = y_true_cpu.to("mps")
        loss_mps = loss_fn_mps(y_pred_mps, y_true_mps).item()

        # Tolerance for MPS vs CPU
        assert abs(loss_mps - loss_cpu) < 1e-4, \
            f"MPS/CPU mismatch: MPS={loss_mps}, CPU={loss_cpu}, diff={abs(loss_mps - loss_cpu)}"

    def test_gradient_matches_analytical(self):
        """NS-13: Autograd gradient matches analytical gradient."""
        # Analytical gradient for LinEx with reduction="mean":
        # L = mean(exp(a*(y_pred - y_true)) - a*(y_pred - y_true) - 1)
        # dL/dy_pred = (1/N) * a * (exp(a*(y_pred - y_true)) - 1)
        loss_fn = LinExLoss(a=0.1, reduction="mean")

        y_pred = torch.tensor([105.0, 95.0, 110.0, 90.0], requires_grad=True)
        y_true = torch.tensor([100.0, 100.0, 100.0, 100.0])

        # Autograd
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        grad_auto = y_pred.grad.clone()

        # Analytical gradient
        error = y_pred.detach() - y_true
        a = 0.1
        N = len(y_pred)
        grad_analytical = (a / N) * (torch.exp(a * error) - 1)

        # Compare
        rel_diff = torch.abs(grad_auto - grad_analytical) / (torch.abs(grad_analytical) + 1e-8)
        assert torch.all(rel_diff < 1e-5), f"Gradient mismatch: auto={grad_auto}, analytical={grad_analytical}, rel_diff={rel_diff}"

    def test_small_a_small_error_expansion(self):
        """NS-14: Small a*error regime matches quadratic expansion."""
        # For |a*error| < 0.1, LinEx ≈ a²*error²/2
        a = 0.01
        loss_fn = LinExLoss(a=a, reduction="none")

        # Small errors such that |a*error| < 0.1
        errors = torch.linspace(-5.0, 5.0, 11)
        y_pred = 100.0 + errors
        y_true = torch.full_like(y_pred, 100.0)

        linex_losses = loss_fn(y_pred, y_true).numpy()
        quadratic = (a**2 * errors.numpy()**2) / 2

        # Relative error should be small (1% is reasonable for this range)
        # At |a*error| = 0.05, the cubic term is ~a³*error³/6 ≈ 1e-6 * 125 / 6 ≈ 2e-5
        # Quadratic term is ~a²*error²/2 ≈ 1e-4 * 25 / 2 = 1.25e-3
        # Relative error ~ 2e-5 / 1.25e-3 = 1.6%
        rel_error = np.abs(linex_losses - quadratic) / (np.abs(quadratic) + 1e-12)
        assert np.all(rel_error < 0.02), f"Quadratic expansion mismatch: max rel error = {np.max(rel_error)}"

    def test_dtype_float64(self):
        """NS-15: Float64 precision preserved."""
        loss_fn = LinExLoss(a=0.1)
        y_pred = torch.tensor([110.0], dtype=torch.float64)
        y_true = torch.tensor([100.0], dtype=torch.float64)

        loss = loss_fn(y_pred, y_true)
        assert loss.dtype == torch.float64

        # Higher precision check
        expected = np.exp(0.1 * 10.0) - 0.1 * 10.0 - 1
        assert abs(loss.item() - expected) < 1e-12

    def test_reduction_none_batch_processing(self):
        """NS-16: reduction='none' correctly processes full batch."""
        loss_fn = LinExLoss(a=0.1, reduction="none")

        batch_size = 1000
        y_pred = torch.randn(batch_size) * 20 + 100
        y_true = torch.full((batch_size,), 100.0)

        losses = loss_fn(y_pred, y_true)
        assert losses.shape == (batch_size,)
        assert torch.isfinite(losses).all()

        # Verify each element independently
        for i in range(10):  # Spot check
            single_loss = loss_fn(y_pred[i:i+1], y_true[i:i+1])
            assert abs(losses[i].item() - single_loss.item()) < 1e-6


class TestLinExEdgeCases:
    """Additional edge case tests."""

    def test_error_exactly_at_threshold(self):
        """Behavior at exact overflow threshold."""
        loss_fn = LinExLoss(a=1.0, overflow_threshold=10.0)
        # error = 10, a*error = 10 exactly at threshold
        y_pred = torch.tensor([110.0])
        y_true = torch.tensor([100.0])

        # Should NOT raise at exact threshold
        loss = loss_fn(y_pred, y_true)
        assert torch.isfinite(loss)

    def test_very_small_a(self):
        """Very small a behaves like scaled quadratic."""
        a = 1e-6
        loss_fn = LinExLoss(a=a, reduction="mean")

        y_pred = torch.tensor([110.0, 90.0, 105.0])
        y_true = torch.tensor([100.0, 100.0, 100.0])

        loss = loss_fn(y_pred, y_true)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_very_large_a(self):
        """Large a produces strong asymmetry."""
        a = 10.0
        loss_fn = LinExLoss(a=a, overflow_threshold=100.0)

        y_pred = torch.tensor([101.0])  # error = 1
        y_true = torch.tensor([100.0])

        loss = loss_fn(y_pred, y_true)
        assert torch.isfinite(loss)
        # With a=10, error=1: exp(10) - 10 - 1 ≈ 22026 - 11 = 22015
        assert loss.item() > 20000

    def test_gradient_clipping_compatibility(self):
        """Works with gradient clipping (as used in training)."""
        model = torch.nn.Linear(10, 1)
        loss_fn = LinExLoss(a=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randn(32, 10)
        y_true = torch.randn(32)

        optimizer.zero_grad()
        y_pred = model(x).squeeze()
        loss = loss_fn(y_pred, y_true)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        assert all(p.grad is not None for p in model.parameters())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])