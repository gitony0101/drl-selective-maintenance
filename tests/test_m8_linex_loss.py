"""
Core LinEx loss functionality tests.

Tests basic LinExLoss class behavior, factory function, and fundamental properties.
"""

import pytest
import torch
import numpy as np
from src.predictors.losses import LinExLoss, build_loss_fn


class TestLinExLossCore:
    """Core LinEx loss functionality tests."""

    def test_zero_error_zero_loss(self):
        """TL-01: Zero error gives zero loss for any a > 0."""
        for a in [0.01, 0.1, 0.5, 1.0, 2.0]:
            loss_fn = LinExLoss(a=a, reduction="mean")
            y_pred = torch.tensor([100.0, 100.0, 100.0])
            y_true = torch.tensor([100.0, 100.0, 100.0])
            loss = loss_fn(y_pred, y_true)
            assert loss.item() == 0.0, f"a={a}: expected 0.0, got {loss.item()}"

    def test_positive_a_required(self):
        """TL-02: a <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="a.*must be > 0"):
            LinExLoss(a=0.0)
        with pytest.raises(ValueError, match="a.*must be > 0"):
            LinExLoss(a=-0.1)
        with pytest.raises(ValueError, match="a.*must be > 0"):
            LinExLoss(a=-1.0)

    def test_asymmetric_penalties(self):
        """TL-03: Equal absolute errors receive asymmetric penalties."""
        loss_fn = LinExLoss(a=0.1, reduction="none")
        # error = +10 (overestimation) vs error = -10 (underestimation)
        y_pred_pos = torch.tensor([110.0])
        y_pred_neg = torch.tensor([90.0])
        y_true = torch.tensor([100.0])

        loss_pos = loss_fn(y_pred_pos, y_true).item()
        loss_neg = loss_fn(y_pred_neg, y_true).item()

        assert loss_pos > loss_neg, f"Overestimation loss {loss_pos} should be > underestimation loss {loss_neg}"

    def test_positive_error_heavier_penalty(self):
        """TL-04: Positive error receives larger loss than negative error of same magnitude."""
        loss_fn = LinExLoss(a=0.1, reduction="mean")
        # Test multiple magnitudes
        for error_mag in [1.0, 5.0, 10.0, 20.0, 50.0]:
            loss_pos = loss_fn(torch.tensor([100.0 + error_mag]), torch.tensor([100.0])).item()
            loss_neg = loss_fn(torch.tensor([100.0 - error_mag]), torch.tensor([100.0])).item()
            assert loss_pos > loss_neg, f"|error|={error_mag}: pos={loss_pos} not > neg={loss_neg}"

    def test_reduction_none_shape(self):
        """TL-05: reduction='none' preserves input shape."""
        loss_fn = LinExLoss(a=0.1, reduction="none")
        batch_size = 16
        y_pred = torch.randn(batch_size) * 10 + 100
        y_true = torch.full((batch_size,), 100.0)
        loss = loss_fn(y_pred, y_true)
        assert loss.shape == (batch_size,), f"Expected {(batch_size,)}, got {loss.shape}"

    def test_reduction_mean_value(self):
        """TL-06: reduction='mean' computes correct mean of elementwise losses."""
        loss_fn_none = LinExLoss(a=0.1, reduction="none")
        loss_fn_mean = LinExLoss(a=0.1, reduction="mean")

        y_pred = torch.tensor([110.0, 90.0, 105.0, 95.0])
        y_true = torch.tensor([100.0, 100.0, 100.0, 100.0])

        loss_none = loss_fn_none(y_pred, y_true)
        loss_mean = loss_fn_mean(y_pred, y_true)

        expected_mean = loss_none.mean().item()
        actual_mean = loss_mean.item()
        assert abs(actual_mean - expected_mean) < 1e-6, f"Mean mismatch: {actual_mean} vs {expected_mean}"

    def test_reduction_sum_value(self):
        """TL-07: reduction='sum' computes correct sum of elementwise losses."""
        loss_fn_none = LinExLoss(a=0.1, reduction="none")
        loss_fn_sum = LinExLoss(a=0.1, reduction="sum")

        y_pred = torch.tensor([110.0, 90.0, 105.0, 95.0])
        y_true = torch.tensor([100.0, 100.0, 100.0, 100.0])

        loss_none = loss_fn_none(y_pred, y_true)
        loss_sum = loss_fn_sum(y_pred, y_true)

        expected_sum = loss_none.sum().item()
        actual_sum = loss_sum.item()
        assert abs(actual_sum - expected_sum) < 1e-6, f"Sum mismatch: {actual_sum} vs {expected_sum}"

    def test_gradients_finite(self):
        """TL-08: Backward pass produces finite gradients."""
        loss_fn = LinExLoss(a=0.1, reduction="mean")
        y_pred = torch.tensor([110.0, 90.0, 105.0], requires_grad=True)
        y_true = torch.tensor([100.0, 100.0, 100.0])

        loss = loss_fn(y_pred, y_true)
        loss.backward()

        assert y_pred.grad is not None
        assert torch.isfinite(y_pred.grad).all(), "Gradients contain NaN or Inf"

    def test_invalid_reduction_fails(self):
        """TL-09: Invalid reduction mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid reduction"):
            LinExLoss(a=0.1, reduction="invalid")
        with pytest.raises(ValueError, match="Invalid reduction"):
            LinExLoss(a=0.1, reduction="avg")

    def test_invalid_a_fails(self):
        """TL-10: Invalid a (<=0) raises ValueError."""
        with pytest.raises(ValueError, match="must be > 0"):
            LinExLoss(a=0.0)
        with pytest.raises(ValueError, match="must be > 0"):
            LinExLoss(a=-0.1)


class TestBuildLossFn:
    """Factory function tests."""

    def test_build_mse(self):
        """build_loss_fn('mse') returns MSELoss."""
        loss_fn = build_loss_fn("mse")
        assert isinstance(loss_fn, torch.nn.MSELoss)

    def test_build_mse_no_kwargs(self):
        """build_loss_fn('mse', **kwargs) raises if kwargs provided."""
        with pytest.raises(ValueError, match="MSELoss takes no arguments"):
            build_loss_fn("mse", a=0.1)

    def test_build_linex(self):
        """build_loss_fn('linex', a=...) returns LinExLoss."""
        loss_fn = build_loss_fn("linex", a=0.1)
        assert isinstance(loss_fn, LinExLoss)
        assert loss_fn.a == 0.1

    def test_build_linex_requires_a(self):
        """build_loss_fn('linex') without 'a' raises ValueError."""
        with pytest.raises(ValueError, match="requires 'a' parameter"):
            build_loss_fn("linex")

    def test_build_unknown_type_fails(self):
        """build_loss_fn('unknown') raises ValueError."""
        with pytest.raises(ValueError, match="Unknown loss_type"):
            build_loss_fn("unknown")


class TestInputValidation:
    """Input validation tests."""

    def test_shape_mismatch_fails(self):
        """Shape mismatch raises ValueError."""
        loss_fn = LinExLoss(a=0.1)
        y_pred = torch.tensor([100.0, 100.0])
        y_true = torch.tensor([100.0])
        with pytest.raises(ValueError, match="Shape mismatch"):
            loss_fn(y_pred, y_true)

    def test_invalid_ndim_fails(self):
        """Invalid tensor dimensions raise ValueError."""
        loss_fn = LinExLoss(a=0.1)
        # 3D tensor
        y_pred = torch.randn(2, 3, 4)
        y_true = torch.randn(2, 3, 4)
        with pytest.raises(ValueError, match="Expected 1D|2D"):
            loss_fn(y_pred, y_true)

        # 2D with wrong second dimension
        y_pred = torch.randn(4, 2)
        y_true = torch.randn(4, 2)
        with pytest.raises(ValueError, match="Expected 1D|2D"):
            loss_fn(y_pred, y_true)

    def test_2d_squeeze_works(self):
        """2D (batch, 1) tensors are correctly squeezed."""
        loss_fn = LinExLoss(a=0.1, reduction="mean")
        y_pred = torch.tensor([[110.0], [90.0]])
        y_true = torch.tensor([[100.0], [100.0]])
        loss = loss_fn(y_pred, y_true)
        assert loss.dim() == 0  # scalar


class TestDeviceCompatibility:
    """Device compatibility tests."""

    def test_cpu_works(self):
        """LinExLoss works on CPU."""
        loss_fn = LinExLoss(a=0.1)
        y_pred = torch.tensor([110.0], device="cpu")
        y_true = torch.tensor([100.0], device="cpu")
        loss = loss_fn(y_pred, y_true)
        assert loss.device.type == "cpu"
        assert loss.item() > 0

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_mps_works(self):
        """LinExLoss works on MPS."""
        loss_fn = LinExLoss(a=0.1).to("mps")
        y_pred = torch.tensor([110.0], device="mps")
        y_true = torch.tensor([100.0], device="mps")
        loss = loss_fn(y_pred, y_true)
        assert loss.device.type == "mps"
        assert loss.item() > 0


class TestDtypeBehavior:
    """Dtype behavior tests."""

    def test_float32(self):
        """Float32 inputs produce float32 loss."""
        loss_fn = LinExLoss(a=0.1)
        y_pred = torch.tensor([110.0], dtype=torch.float32)
        y_true = torch.tensor([100.0], dtype=torch.float32)
        loss = loss_fn(y_pred, y_true)
        assert loss.dtype == torch.float32

    def test_float64(self):
        """Float64 inputs produce float64 loss."""
        loss_fn = LinExLoss(a=0.1)
        y_pred = torch.tensor([110.0], dtype=torch.float64)
        y_true = torch.tensor([100.0], dtype=torch.float64)
        loss = loss_fn(y_pred, y_true)
        assert loss.dtype == torch.float64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])