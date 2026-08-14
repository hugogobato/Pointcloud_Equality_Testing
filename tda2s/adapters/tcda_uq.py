"""Thin shim over ``tcda_uq``: import surface for P1, zero reimplemented math.

P1 reuses the released CP_TATE library (``tcda_uq``, installed from git) for
AIPW estimation, cross-fitting, the functional DR-learner, silhouettes, and the
tri-oracle simulation. This module is pure delegation and argument plumbing;
every function forwards to ``tcda_uq`` and only reshapes return values where the
P1 test statistic needs a different layout (e.g. stacking per-dimension score
lists into one array).

Boundary (see ``docs/reuse_from_tcda_uq.md``): tcda_uq owns confidence /
prediction bands; P1 owns p-values. The only shared objects are the AIPW curve
(``aipw[d]``) and the per-unit efficient-influence-function process
(``scores``), from which P1 computes its test statistic and multiplier-bootstrap
null law. Bands are deliberately not exposed here.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def aipw_curve(sample, tseq, n_basis: int, n_folds: int = 5, **cross_fit_kwargs) -> dict:
    """Cross-fitted AIPW estimate of the TATE curve(s) for one sample.

    Delegates to ``tcda_uq.estimators.cross_fit`` and reshapes the result for
    P1's test statistic:

    * ``aipw``: list, one ``[resolution]`` mean AIPW curve per homology dim.
    * ``scores``: ``(n, n_hom_dim, resolution)`` per-unit doubly-robust score
      process (the cross-fitted EIF; its mean over units is ``aipw``).
    * ``pi_hat``: ``(n,)`` cross-fitted propensity.
    * ``tseq``: the silhouette grid.

    Args:
        sample: observed triplet ``(phi, A, X)`` with ``phi`` ``[n, n_hom_dim,
            resolution]``, ``A`` ``[n]``, ``X`` ``[n, d]``.
        tseq: silhouette grid ``[resolution]``.
        n_basis: Fourier basis size for the outcome regression.
        n_folds: number of cross-fitting folds (``cross_fit``'s ``n_splits``).
        **cross_fit_kwargs: forwarded verbatim to ``cross_fit`` (e.g.
            ``propensity_estimator``, ``propensity_feature_fn``, ``stratify``,
            ``random_state``); default ``None`` reproduces tcda_uq defaults.
    """
    from tcda_uq.estimators import cross_fit

    result = cross_fit(sample, tseq, n_basis=n_basis, n_splits=n_folds,
                       **cross_fit_kwargs)
    return {
        "aipw": result.aipw,
        "scores": np.stack(result.scores, axis=1),
        "pi_hat": result.pi_hat,
        "tseq": np.asarray(result.tseq),
    }


def silhouettes(diagrams, interval=(0.0, 0.2), r: float = 3.0,
                resolution: int = 100) -> np.ndarray:
    """Power-weighted silhouettes of persistence diagrams.

    Delegates to ``tcda_uq.silhouette.compute_silhouette`` (defaults: interval
    ``(0, 0.2)``, ``r=3``, ``resolution=100``). Returns ``(n_hom_dim,
    resolution)``.
    """
    from tcda_uq.silhouette import compute_silhouette

    return compute_silhouette(diagrams, interval=interval, r=r,
                              resolution=resolution)


def tri_oracle(n: int, **kwargs) -> "SimulationSample":
    """Draw a sample from the tri-oracle simulation.

    Delegates to ``tcda_uq.datasets.TriOracleSimulation``: ``kwargs`` are
    passed to its constructor (``n_cov``, ``n_hom_dim``, ``resolution``,
    ``interval``, ``n_basis``, ``noise_scale``, ``seed``, ...), then ``sample(n)``
    is drawn. Returns a ``SimulationSample`` with ``oracle_tate``,
    ``oracle_ctate``, ``oracle_itte`` and the ``.observed`` triplet.
    """
    from tcda_uq.datasets import TriOracleSimulation

    return TriOracleSimulation(**kwargs).sample(n)


def ctate_learner(*args, **kwargs):
    """Functional DR-learner for the CTATE (same signature as ``CTATEDRLearner``).

    Pure delegation to ``tcda_uq.estimators.CTATEDRLearner``; the returned
    object is a ``CTATEDRLearner``: ``fit(sample, tseq, cross_fit_result=None,
    **cross_fit_kwargs)`` then ``predict(X_eval)``.
    """
    from tcda_uq.estimators import CTATEDRLearner

    return CTATEDRLearner(*args, **kwargs)
