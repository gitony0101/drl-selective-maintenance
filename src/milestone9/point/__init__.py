"""M9 point-estimate fallback package.

Stage-0 returned MCD NO-GO; this package implements the contractually-mandated
point-estimate (non-uncertainty) DDQN run: mse_control predictors, no LinEx, no
MCD, no q10/q05/std features, no uncertainty risk cache. See
``docs/milestone9/M9_POINT_ESTIMATE_CONTRACT.{md,json}`` for the frozen contract.
"""

from __future__ import annotations
