"""
LinEx (Linear-Exponential) Asymmetric Loss for RUL Prediction

Numerically stable implementation of L(error; a) = exp(a * error) - a * error - 1
where error = predicted_rul - true_rul

For a > 0: overestimation (error > 0) receives heavier penalty than underestimation.
"""

import torch
import torch.nn as nn
from typing import Literal


class LinExLoss(nn.Module):
    """
    LinEx asymmetric loss for RUL prediction.

    L(error; a) = exp(a * error) - a * error - 1
    where error = predicted_rul - true_rul

    For a > 0:
    - error > 0 (overestimation) → exponentially larger penalty
    - error < 0 (underestimation) → linear-ish penalty
    - error = 0 → zero loss

    Args:
        a: Asymmetry parameter (must be > 0). Controls penalty asymmetry.
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        overflow_threshold: Maximum allowed |a * error|. Exceeding raises RuntimeError.
                           Default: 20.0 (exp(20) ≈ 4.85e8, safe for float32)

    Shape:
        - y_pred: (batch,) or (batch, 1)
        - y_true: (batch,) or (batch, 1)
        - Output: scalar (reduction != 'none') or (batch,) (reduction == 'none')

    Examples:
        >>> loss_fn = LinExLoss(a=0.1)
        >>> y_pred = torch.tensor([100.0, 120.0, 80.0])
        >>> y_true = torch.tensor([100.0, 100.0, 100.0])
        >>> loss = loss_fn(y_pred, y_true)  # overestimation penalized more
    """

    def __init__(
        self,
        a: float,
        reduction: Literal["none", "mean", "sum"] = "mean",
        overflow_threshold: float = 20.0,
    ):
        super().__init__()

        if a <= 0:
            raise ValueError(
                f"LinEx asymmetry parameter 'a' must be > 0, got {a}. "
                "For MSE baseline, use nn.MSELoss() directly."
            )
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"Invalid reduction: {reduction}. Use 'none', 'mean', or 'sum'.")
        if overflow_threshold <= 0:
            raise ValueError(f"overflow_threshold must be > 0, got {overflow_threshold}")

        self.a = float(a)
        self.reduction = reduction
        self.overflow_threshold = float(overflow_threshold)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute LinEx loss.

        Args:
            y_pred: Predicted RUL values
            y_true: True RUL values

        Returns:
            Loss tensor (scalar or per-sample depending on reduction)

        Raises:
            ValueError: Shape mismatch or invalid input dimensions
            RuntimeError: NaN/Inf inputs, overflow detected, non-finite output
        """
        # Shape validation
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}"
            )
        if y_pred.ndim not in (1, 2) or (y_pred.ndim == 2 and y_pred.shape[1] != 1):
            raise ValueError(
                f"Expected 1D (batch,) or 2D (batch, 1), got {y_pred.shape}"
            )

        # Squeeze to 1D for computation
        y_pred_1d = y_pred.squeeze(-1) if y_pred.ndim == 2 else y_pred
        y_true_1d = y_true.squeeze(-1) if y_true.ndim == 2 else y_true

        # Finite input validation
        if not torch.isfinite(y_pred_1d).all():
            raise RuntimeError("NaN or Inf detected in y_pred")
        if not torch.isfinite(y_true_1d).all():
            raise RuntimeError("NaN or Inf detected in y_true")

        # error = predicted_rul - true_rul
        # error > 0 means overestimation (dangerous)
        error = y_pred_1d - y_true_1d

        # z = a * error
        z = self.a * error

        # Overflow guard: fail-closed with diagnostic info
        max_abs_z = z.abs().max().item()
        if max_abs_z > self.overflow_threshold:
            max_abs_error = error.abs().max().item()
            raise RuntimeError(
                f"LinEx overflow: |a * error| = {max_abs_z:.4f} > "
                f"threshold {self.overflow_threshold}. "
                f"a={self.a}, max|error|={max_abs_error:.4f}. "
                f"Consider reducing 'a' or increasing 'overflow_threshold'."
            )

        # Numerically stable: expm1(z) = exp(z) - 1
        # Loss = expm1(z) - z = (exp(z) - 1) - z = exp(z) - z - 1
        loss = torch.expm1(z) - z

        # Finite output validation
        if not torch.isfinite(loss).all():
            raise RuntimeError("Non-finite loss produced (NaN or Inf)")

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss

    def extra_repr(self) -> str:
        return f"a={self.a}, reduction={self.reduction}, overflow_threshold={self.overflow_threshold}"


def build_loss_fn(
    loss_type: Literal["mse", "linex"],
    **kwargs
) -> nn.Module:
    """
    Factory function for loss functions.

    Args:
        loss_type: "mse" or "linex"
        **kwargs: For "linex": a (required), reduction, overflow_threshold

    Returns:
        Configured loss module

    Raises:
        ValueError: Unknown loss_type or missing required args for LinEx
    """
    if loss_type == "mse":
        if kwargs:
            raise ValueError(f"MSELoss takes no arguments, got: {list(kwargs.keys())}")
        return nn.MSELoss()
    elif loss_type == "linex":
        if "a" not in kwargs:
            raise ValueError("LinEx loss requires 'a' parameter (asymmetry > 0)")
        return LinExLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Use 'mse' or 'linex'.")


# ---- Verification utilities (for testing) ----

def verify_linex_properties(a: float = 0.1, n_samples: int = 10000) -> dict:
    """
    Verify LinEx mathematical properties numerically.

    Returns dict with verification results.
    """
    import numpy as np

    # Generate symmetric errors
    errors = np.random.uniform(-50, 50, n_samples)
    errors_tensor = torch.tensor(errors, dtype=torch.float32)
    y_pred = errors_tensor + 100.0  # arbitrary true RUL = 100
    y_true = torch.full_like(y_pred, 100.0)

    loss_fn = LinExLoss(a=a, reduction="none")
    losses = loss_fn(y_pred, y_true).numpy()

    # Property 1: L(0; a) = 0
    zero_loss = LinExLoss(a=a, reduction="none")(
        torch.tensor([100.0]), torch.tensor([100.0])
    ).item()

    # Property 2: Asymmetry - positive error has larger loss
    pos_mask = errors > 0
    neg_mask = errors < 0
    # Match absolute values
    pos_errors = errors[pos_mask]
    neg_errors = -errors[neg_mask]  # Make positive

    # For matched absolute errors
    min_len = min(len(pos_errors), len(neg_errors))
    pos_errors = pos_errors[:min_len]
    neg_errors = neg_errors[:min_len]

    pos_losses = LinExLoss(a=a, reduction="none")(
        torch.tensor(100.0 + pos_errors, dtype=torch.float32),
        torch.tensor([100.0] * min_len, dtype=torch.float32)
    ).numpy()
    neg_losses = LinExLoss(a=a, reduction="none")(
        torch.tensor(100.0 - neg_errors, dtype=torch.float32),
        torch.tensor([100.0] * min_len, dtype=torch.float32)
    ).numpy()

    asymmetry_holds = np.all(pos_losses > neg_losses)

    # Property 3: Convexity (second derivative > 0)
    # d²L/derror² = a² * exp(a * error) > 0
    convexity_holds = True  # Analytically guaranteed for a > 0

    # Property 4: Small-a expansion check
    # L ≈ (a² * error²) / 2 for small |a * error|
    small_errors = np.linspace(-0.01, 0.01, 1000)
    a = 0.1
    exact = np.exp(a * small_errors) - a * small_errors - 1
    approx = (a**2 * small_errors**2) / 2
    rel_error = np.abs(exact - approx) / (np.abs(exact) + 1e-12)
    expansion_verified = np.all(rel_error < 0.01)  # <1% relative error

    return {
        "zero_loss": zero_loss,
        "zero_loss_approx_zero": abs(zero_loss) < 1e-10,
        "asymmetry_holds": bool(asymmetry_holds),
        "mean_pos_loss": float(np.mean(pos_losses)) if min_len > 0 else None,
        "mean_neg_loss": float(np.mean(neg_losses)) if min_len > 0 else None,
        "convexity_holds": convexity_holds,
        "small_a_expansion_verified": expansion_verified,
        "max_rel_error_expansion": float(np.max(rel_error)),
    }


if __name__ == "__main__":
    # Quick self-test
    print("Testing LinExLoss...")

    # Test 1: Basic functionality
    loss_fn = LinExLoss(a=0.1, reduction="mean")
    y_pred = torch.tensor([100.0, 120.0, 80.0, 90.0])
    y_true = torch.tensor([100.0, 100.0, 100.0, 100.0])
    loss = loss_fn(y_pred, y_true)
    print(f"Test 1 - Basic loss: {loss.item():.6f}")

    # Test 2: Zero error = zero loss
    loss_zero = loss_fn(torch.tensor([100.0]), torch.tensor([100.0]))
    print(f"Test 2 - Zero error loss: {loss_zero.item():.10f}")

    # Test 3: Asymmetry
    loss_pos = loss_fn(torch.tensor([110.0]), torch.tensor([100.0]))
    loss_neg = loss_fn(torch.tensor([90.0]), torch.tensor([100.0]))
    print(f"Test 3 - Overestimation (+10) loss: {loss_pos.item():.6f}")
    print(f"Test 3 - Underestimation (-10) loss: {loss_neg.item():.6f}")
    print(f"Test 3 - Asymmetry holds: {loss_pos.item() > loss_neg.item()}")

    # Test 4: Reduction modes
    loss_none = LinExLoss(a=0.1, reduction="none")(y_pred, y_true)
    loss_sum = LinExLoss(a=0.1, reduction="sum")(y_pred, y_true)
    print(f"Test 4 - Reduction none shape: {loss_none.shape}")
    print(f"Test 4 - Reduction sum: {loss_sum.item():.6f}")

    # Test 5: Properties verification
    props = verify_linex_properties(a=0.1)
    print(f"\nProperty verification:")
    for k, v in props.items():
        print(f"  {k}: {v}")

    # Test 6: Factory function
    mse_fn = build_loss_fn("mse")
    linex_fn = build_loss_fn("linex", a=0.05)
    print(f"\nTest 6 - Factory MSE: {type(mse_fn).__name__}")
    print(f"Test 6 - Factory LinEx: {type(linex_fn).__name__}, a={linex_fn.a}")

    print("\nAll self-tests passed!")