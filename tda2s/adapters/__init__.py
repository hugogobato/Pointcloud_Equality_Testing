"""Adapters over ``tcda_uq``: P1's import surface for the shared estimators.

See ``docs/reuse_from_tcda_uq.md`` for the full reuse audit and boundary
(CP_TATE owns bands; P1 owns p-values; the shared objects are the AIPW curve
and the per-unit EIF process).
"""

from .tcda_uq import aipw_curve, ctate_learner, silhouettes, tri_oracle

__all__ = ["aipw_curve", "silhouettes", "tri_oracle", "ctate_learner"]
