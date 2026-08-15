"""Phase 2 prototype DR test of H0^out (task 2.4); polished in Phase 3.

The test statistic and its null follow the boundary statement of
``docs/reuse_from_tcda_uq.md``: P1's statistic is

    T_n = sqrt(n) * max_d || psi_hat_d ||_inf,

computed from ``cross_fit(...).aipw[d]`` (the cross-fitted AIPW TATE curve), and
its null is the multiplier bootstrap over the *centered* per-unit EIF process
``cross_fit(...).scores[d] - mean``:

    G_b(t) = n^{-1/2} sum_i xi_i (s_{i,d}(t) - mean_d(t)),  xi_i ~ N(0, 1),

compared at ``max_d sup_t |G_b(t)|`` (``tda2s.resample.multiplier_bootstrap``
returns exactly these sup draws). The multipliers ``xi_i`` are drawn once per
unit and shared across the homology degrees, since the degrees are dependent
functionals of the same units; see the note in
:func:`prototype_dr_from_phi`. This is a prototype: Phase 3 calibrates the
statistic (multiplicity handling, learner sweep, stratified-permutation null)
and relocates it to the Phase 3 deliverable; nothing here reimplements AIPW,
cross-fitting or the DR-learner, which are imported from ``tcda_uq`` through the
``tda2s.adapters.tcda_uq.aipw_curve`` shim.
"""

from __future__ import annotations

import numpy as np

from tda2s.resample import multiplier_bootstrap


def _default_rf(seed):
    """Seeded random-forest propensity estimator (default in tcda_uq)."""
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(random_state=int(seed))


def prototype_dr_from_phi(phi, A, X, tseq, n_basis=8, n_folds=2, n_draws=2000,
                          seed=0, **cross_fit_kwargs) -> float:
    """Prototype DR p-value from a silhouette triplet ``(phi, A, X)``.

    Splits ``prototype_dr_pvalue`` at the silhouette stage so sweeps that
    share the clouds across group assignments (Phase 2.3 common-random-numbers
    design) compute ``phi`` once and call this per assignment.

    Args:
        phi: ``(n, n_hom_dim, resolution)`` silhouette array (the
            ``phi`` returned by ``tda2s.dgp.to_silhouette_sample``).
        A: ``[n]`` group labels.
        X: ``[n, d_x]`` covariates.
        tseq: ``[resolution]`` grid underlying the silhouettes.
        n_basis: Fourier basis size of the outcome regression.
        n_folds: cross-fitting folds (``cross_fit``'s ``n_splits``).
        n_draws: multiplier-bootstrap draws.
        seed: RNG seed (cross_fit ``random_state`` and the bootstrap stream).
        **cross_fit_kwargs: forwarded to ``cross_fit`` (e.g. a
            ``propensity_estimator``); default reproduces tcda_uq defaults.

    Returns:
        float p-value in [0, 1] for ``H0^out: psi_d = 0``.
    """
    from tda2s.adapters.tcda_uq import aipw_curve

    cross_fit_kwargs.setdefault("propensity_estimator", _default_rf(seed))
    res = aipw_curve((phi, A, X), tseq, n_basis=n_basis, n_folds=n_folds,
                     random_state=seed, **cross_fit_kwargs)
    scores = res["scores"]                     # (n, n_hom_dim, resolution)
    aipw = res["aipw"]                         # list, per dim, [resolution]
    n = int(A.shape[0])

    T_obs = float(np.sqrt(n) * np.max([np.abs(np.asarray(psi)).max() for psi in aipw]))

    # One multiplier draw per *unit*, shared across homology degrees. The
    # degrees are two functionals of the same n units, so their EIF processes
    # are dependent; bootstrapping each degree with its own multipliers would
    # make the null's two components independent, and the maximum of two
    # independent copies stochastically dominates the maximum of positively
    # dependent ones with the same marginals. That inflates the null, and the
    # test would come out conservative -- which is exactly what the gate's size
    # criterion measures. Stacking the degrees along the curve axis and taking
    # one sup over the stack is ``max_d sup_t`` under shared multipliers.
    rng = np.random.default_rng(seed)
    centered = scores - scores.mean(axis=0, keepdims=True)   # (n, n_dim, res)
    stacked = centered.reshape(centered.shape[0], -1)        # (n, n_dim * res)
    nulls = multiplier_bootstrap(stacked, n_draws, rng)
    return float((1.0 + np.count_nonzero(nulls >= T_obs)) / (1.0 + n_draws))


def prototype_dr_pvalue(clouds, X, A, filtration="alpha", homology_dims=(0, 1),
                        interval=(0.0, 2.0), r: float = 3.0, resolution: int = 100,
                        n_basis: int = 8, n_folds: int = 2, n_draws: int = 2000,
                        seed: int = 0, **cross_fit_kwargs) -> float:
    """Prototype doubly-robust test of ``H0^out: psi_d = 0`` (full pipeline).

    Args:
        clouds: list of ``(m_i, d)`` point clouds (one per unit).
        X: ``[n, d_x]`` covariate matrix.
        A: ``[n]`` group labels.
        filtration, homology_dims: passed to ``tda2s.ph.compute_diagrams``.
        interval, r, resolution: silhouette domain, power weight and grid size
            (the same grid must cover the persistence scales of the clouds).
        n_basis: Fourier basis size of the outcome regression.
        n_folds: cross-fitting folds (``cross_fit``'s ``n_splits``).
        n_draws: multiplier-bootstrap draws.
        seed: RNG seed (cross_fit ``random_state`` and the bootstrap stream).
        **cross_fit_kwargs: forwarded to ``cross_fit`` (e.g. a
            ``propensity_estimator``); default reproduces tcda_uq defaults.

    Returns:
        float p-value in [0, 1]: the fraction of multiplier draws with
        ``max_d sup_t |G_b^{(d)}(t)| >= T_n`` (Phipson-Smyth convention).
    """
    from tda2s.dgp import to_silhouette_sample

    phi, A, X = to_silhouette_sample(clouds, X, A, filtration=filtration,
                                     homology_dims=homology_dims,
                                     interval=interval, r=r, resolution=resolution)
    tseq = np.linspace(interval[0], interval[1], resolution)
    return prototype_dr_from_phi(phi, A, X, tseq, n_basis=n_basis, n_folds=n_folds,
                                 n_draws=n_draws, seed=seed, **cross_fit_kwargs)