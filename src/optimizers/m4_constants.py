"""
Milestone 4 Constants - Single authoritative source for engineering thresholds.

This module provides the single frozen configuration source for all
engineering coverage thresholds. Changing the value here changes:
- Scenario generation (urgent candidate threshold)
- Scientific config (affects config_hash)
- Bank hashes (via scenario selection)
- Production runner behavior

It does NOT affect:
- Primary hard_window_v1 policy (uses delta_cycles=5)
"""

# Single authoritative engineering coverage threshold
# Used for scenario generation candidate selection
# NOT the primary policy failure window (delta_cycles=5)
ENGINEERING_COVERAGE_THRESHOLD_CYCLES = 6.0


def get_engineering_coverage_threshold_cycles() -> float:
    """
    Get the authoritative engineering coverage threshold.

    Returns:
        The frozen threshold value (6.0 cycles).

    This function ensures all callers use the same constant.
    """
    return ENGINEERING_COVERAGE_THRESHOLD_CYCLES