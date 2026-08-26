"""M10 E3 (n-step DDQN) integration package.

Owns the E3-specific components that do NOT belong in the frozen milestone-9
point-estimate layer:

- ``h2_context``: the M9-seed-cache PlannerContext adapter (Part 2) that lets
  the canonical M6 H2Planner run against each official M9 per-seed prediction
  cache without weakening the canonical M6 PlannerContext contract.
- ``trajectories``: generation + integrity of training-only (predictor_train)
  raw M4 / H2 ordered trajectories (Parts 4-5).
- ``nstep``: the generic episode-aware n-step accumulator + replay integration
  (Parts 6-8).
"""

from __future__ import annotations

# Sub-namespaces are imported explicitly by callers; keep package import
# lightweight so importing e3 does not force heavy deps.