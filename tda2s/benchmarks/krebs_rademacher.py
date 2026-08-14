"""Krebs & Rademacher (2024) relevant-difference test on persistence diagrams.

Reference: J. Krebs and D. Rademacher, "Two-sample tests for relevant
differences in persistence diagrams", arXiv:2401.10349v1 (2024). Equation
numbers below are the paper's. Paper accessed in full; no author code was
released and the paper contains no simulation study, so this is a direct
transcription of Section 1.2 with no published figure or table to reproduce.

What the paper actually tests
-----------------------------
NOT a difference of mean summaries: the paper compares *dispersion* of the two
diagram populations under a Wasserstein metric, in a **relevant-difference**
framing with an externally set tolerance ``Delta >= 0`` (eq. 1.7)::

    H_0: (sigma^2_X - sigma^2_Y)^2 <= Delta   vs   H_1: (...)^2 > Delta

Section 1.1 does this with Frechet variances, which needs a Frechet mean in
``(D_r, W_r)`` -- a non-unique, expensive optimisation. Section 1.2 uses the
*independent copy* ("inco") variance instead, which needs no mean at all
(eq. 1.4)::

    sigma^2(X) := (1/2) E[ W_r^2( PD(X_0), PD(X') ) ]

estimated by the U-statistic with kernel ``h(.,.) = 2^{-1} W_r^2(PD(.), PD(.))``
(eqs. 1.5, 1.13). This module implements the Section 1.2 (inco-variance) test,
which the paper itself recommends as the practical route.

Statistic (eqs. 1.11-1.16)
--------------------------
Two-parameter partial-sum processes, normalised by the FULL sample size so that
``sigma^2_X(s, t) ~ s t sigma^2_X`` (eq. 1.12)::

    sigma^2_X(s, t) = 1/(m(m-1)) sum_{i<=floor(ms)} sum_{j<=floor(mt), j!=i} h(X_i, X_j)

with ``D(s, t) = sigma^2_X(s, t) - sigma^2_Y(s, t)`` (eq. 1.11) and
``D_hat = D(1, 1)``. The self-normaliser (eq. 1.14) is::

    V_hat = { int int [ D(s,t)^2 - (s t D_hat)^2 ]^2 nu(ds, dt) }^{1/2}

and the test statistic and rejection rule are::

    W_hat = (D_hat^2 - Delta) / V_hat ,      reject if W_hat > q_{1-alpha}

where ``q_{1-alpha}`` is the quantile of the pivotal limit (eq. 1.16)::

    W = 2 B(1) / { int int [ s t ( t B(s) + s B(t) - 2 s t B(1) ) ]^2 dnu(s,t) }^{1/2}

for a standard Brownian motion ``B``. The ``sqrt(m+n)`` rates of Theorem 1.4
cancel in the ratio, so no explicit scaling appears. ``nu`` is any probability
measure on ``[0, 1]^2``; the uniform measure on a regular grid is used here and
is exposed via ``grid``.

The Delta = 0 boundary
----------------------
The paper's framework is built for ``Delta > 0``. At ``Delta = 0`` its limit
degenerates: Theorem 1.4's scale factor is
``xi = 2 sqrt(Gamma_X/tau + Gamma_Y/(1-tau)) (sigma^2_X - sigma^2_Y)``, which is
zero exactly when the null holds with ``Delta = 0``, so ``W`` is not the right
reference law there. Because a benchmark suite needs every competitor evaluated
at a common classical null, ``delta=0`` (the default) instead calibrates
``D_hat^2`` by label permutation. That path is an extension, not the paper's
test, and is labelled as such; pass ``delta > 0`` for the published procedure.
"""
from __future__ import annotations

import numpy as np

from ._common import _perm_groups, _perm_pvalue, _points, _validate

__all__ = ["test_krebs_rademacher", "inco_variance", "pivotal_quantiles"]


def _pairwise_kernel(diagrams, metric):
    """``K[i, j] = (1/2) W_r^2(PD_i, PD_j)``, the kernel of eq. (1.13)."""
    n = len(diagrams)
    K = np.zeros((n, n))
    if metric == "wasserstein":
        import persim
        dist = lambda a, b: float(persim.wasserstein(a, b))
    elif metric == "bottleneck":
        import gudhi as gd
        dist = lambda a, b: float(gd.bottleneck_distance(a, b))
    else:
        raise ValueError(f"metric must be 'wasserstein' or 'bottleneck', got {metric!r}")
    pts = [_points(d) for d in diagrams]
    for i in range(n):
        for j in range(i + 1, n):
            K[i, j] = K[j, i] = 0.5 * dist(pts[i], pts[j]) ** 2
    return K


def inco_variance(K):
    """Inco-variance U-statistic ``sigma^2 = K.sum() / (n(n-1))`` (eq. 1.5)."""
    n = K.shape[0]
    if n < 2:
        return 0.0
    return float(K.sum() / (n * (n - 1)))


def _partial_sums(K, grid):
    """``sigma^2(s, t)`` of eq. (1.12) on a regular ``grid x grid`` mesh."""
    n = K.shape[0]
    if n < 2:
        return np.zeros((grid, grid))
    # cumulative block sums; K has a zero diagonal, so no diagonal correction
    C = np.zeros((n + 1, n + 1))
    C[1:, 1:] = K.cumsum(axis=0).cumsum(axis=1)
    ks = np.floor(n * np.arange(1, grid + 1) / grid).astype(int)
    return C[np.ix_(ks, ks)] / (n * (n - 1))


def _self_normaliser(D_st, D_hat, grid):
    """``V_hat`` of eq. (1.14) under uniform ``nu`` on the grid."""
    s = np.arange(1, grid + 1) / grid
    st = np.outer(s, s)
    integrand = (D_st ** 2 - (st * D_hat) ** 2) ** 2
    return float(np.sqrt(integrand.mean()))


def _simulate_W(grid, n_draws, rng):
    """Draws from the pivotal limit ``W`` of eq. (1.16)."""
    s = np.arange(1, grid + 1) / grid
    st = np.outer(s, s)
    incr = rng.normal(0.0, np.sqrt(1.0 / grid), size=(n_draws, grid))
    B = np.cumsum(incr, axis=1)                       # B(s_k), B(1) = B[:, -1]
    B1 = B[:, -1]
    # t B(s) + s B(t) - 2 s t B(1), then scaled by s t
    term = (s[None, None, :] * B[:, :, None]          # t B(s): rows = s, cols = t
            + s[None, :, None] * B[:, None, :]        # s B(t)
            - 2.0 * st[None] * B1[:, None, None])
    denom = np.sqrt(((st[None] * term) ** 2).mean(axis=(1, 2)))
    return 2.0 * B1 / denom


def pivotal_quantiles(q, grid=20, n_draws=20000, seed=0):
    """Quantiles of the pivotal limit ``W`` of eq. (1.16).

    Args:
        q: quantile level(s) in (0, 1).
        grid: mesh size of the uniform ``nu`` on ``[0, 1]^2`` (must match the
            grid used for ``V_hat``).
        n_draws: Monte Carlo Brownian paths.
        seed: RNG seed.

    Returns:
        Array of quantiles of ``W``.
    """
    return np.quantile(_simulate_W(grid, n_draws, np.random.default_rng(seed)), q)


def test_krebs_rademacher(diags0, diags1, delta=0.0, dim_index=-1,
                          metric="wasserstein", grid=20, n_perm=200, seed=None,
                          n_pivotal=20000):
    """Krebs-Rademacher inco-variance test (paper Section 1.2).

    Args:
        diags0, diags1: lists (over samples) of lists (per homology dim) of
            ``(k, 2)`` birth-death arrays. Sample ORDER matters: the partial-sum
            processes of eq. (1.12) are taken in the given order (irrelevant for
            i.i.d. samples, meaningful for the paper's time-series setting).
        delta: relevance tolerance ``Delta`` of eq. (1.7). ``> 0`` runs the
            published self-normalised test; ``0`` (default) falls back to a
            permutation null -- see the module docstring.
        dim_index: index into each sample's per-dimension diagram list
            (NOT the homology degree); ``-1`` is the last, i.e. the highest
            homology dimension present. The paper fixes one dimension.
        metric: ``"wasserstein"`` (``W_1`` via persim) or ``"bottleneck"``
            (``W_inf`` via gudhi).
        grid: mesh size for ``nu`` on ``[0, 1]^2``.
        n_perm: permutations used when ``delta == 0``.
        seed: RNG seed.
        n_pivotal: Brownian draws for the eq. (1.16) reference law.

    Returns:
        float p-value in [0, 1].
    """
    d0, d1, nd = _validate(diags0, diags1)
    if not -nd <= dim_index < nd:
        raise ValueError(
            f"dim_index {dim_index} out of range for {nd}-dim diagram lists")
    m, n = len(d0), len(d1)
    if m < 2 or n < 2:
        return 1.0

    pooled = [d[dim_index] for d in d0] + [d[dim_index] for d in d1]
    K = _pairwise_kernel(pooled, metric)
    idx0 = np.arange(m)
    idx1 = np.arange(m, m + n)

    def _d_hat(i0, i1):
        return inco_variance(K[np.ix_(i0, i0)]) - inco_variance(K[np.ix_(i1, i1)])

    if delta > 0:
        sx = _partial_sums(K[np.ix_(idx0, idx0)], grid)
        sy = _partial_sums(K[np.ix_(idx1, idx1)], grid)
        D_st = sx - sy
        D_hat = float(D_st[-1, -1])
        V_hat = _self_normaliser(D_st, D_hat, grid)
        if V_hat <= 0:
            return 1.0
        W_hat = (D_hat ** 2 - delta) / V_hat                  # eq. (1.8) form
        W = _simulate_W(grid, n_pivotal, np.random.default_rng(0 if seed is None else seed))
        return float((1.0 + np.sum(W >= W_hat)) / (1.0 + W.size))

    # delta == 0: permutation calibration of D_hat^2 (extension, see docstring)
    obs = _d_hat(idx0, idx1) ** 2
    rng = np.random.default_rng(seed)
    perms = _perm_groups(m + n, m, n_perm, rng)
    null = np.empty(n_perm)
    for b in range(n_perm):
        g = perms[b]
        null[b] = _d_hat(np.flatnonzero(g), np.flatnonzero(~g)) ** 2
    return _perm_pvalue(obs, null, "greater")
