"""Han, Kim & Kim (2026) kernel permutation test on persistence intensity functions.

Reference: Y. Han, I. Kim and J. Kim, "A two-sample test on weighted
persistence intensity functions in topological data analysis", arXiv:2607.20893
(2026).  Paper accessed in full on arXiv; this is the method as described
there (not the intensity-grid/sup-norm/bootstrapped version sketched in the
Phase 0.5 brief, which does not match the published procedure).

Method (paper Sections 2-4 and 6):
    * Diagram points z = (b, d) in Omega = {y > x >= 0} carry a weight
      ``w(z) = (d - b)^q`` (the paper also allows arctan; q = weight_power,
      default 1).
    * The kernel is a product of two 1D kernels with a bandwidth vector
      ``lambda = (lambda_1, lambda_2)``,
      ``k_lambda(x, y) = prod_i (1/(lambda_i sqrt(pi))) exp(-((x_i - y_i)/lambda_i)^2)``,
      and the weighted diagram kernel is
      ``K(X, Y) = sum_{x in X} sum_{y in Y} w(x) w(y) k_lambda(x, y)``.
    * The test statistic is the unbiased two-sample U-statistic estimator of
      ``||mu_p - mu_q||^2`` in the RKHS of ``k_{w, lambda}``:
      ``T = E_{i != i'} K(U_i, U_i') + E_{j != j'} K(U_j, U_j') - 2 E_{i, j} K(U_i, U_j)``.
    * The null distribution is the permutation distribution of T (the paper's
      Algorithm 1).  Because the optimal bandwidth is unknown, the default
      ``aggregate=True`` runs the bandwidth-aggregation test (Aggtest,
      Algorithm 2): per-bandwidth rank p-values are computed on the same
      permutations and combined as ``A_b = min_lambda p_b^lambda``; the final
      p-value counts how many of the B + 1 replicates (permutations plus the
      observation) attain ``A_b <= A_obs``.

The bandwidth grid is ``scale * m`` over ``scales``, where ``m`` is the
median pairwise absolute difference of the pooled diagram points along each
coordinate (a data-driven stand-in; the paper's simulation grid is reported
only in its supplementary material, which we could not fully retrieve).
"""
from __future__ import annotations

import numpy as np

from ._common import (_gaussian_gram_blocks, _median_pairwise_scale,
                      _membership, _perm_groups, _perm_pvalue, _points,
                      _validate)


def _weighted_diagram_kernel(X, Bw, lam):
    """Diagram-level weighted kernel matrix ``K(U_i, U_j)`` for bandwidth ``lam``.

    The paper's kernel is a product of one-dimensional Gaussians,
    ``prod_j (lam_j sqrt(pi))^-1 exp(-((x_j - y_j) / lam_j)^2)``, which is the
    isotropic ``exp(-||z - z'||^2)`` on rescaled coordinates ``z = x / lam``
    times the constant ``prod_j (lam_j sqrt(pi))^-1``. The point weights
    ``w(x) w(y)`` are already folded into the columns of ``Bw``, so the whole
    diagram-level matrix is one blockwise Gram accumulation and the point-level
    kernel is never materialised.
    """
    lam = np.asarray(lam, dtype=float)
    norm = 1.0 / np.prod(lam * np.sqrt(np.pi))
    return norm * _gaussian_gram_blocks(X / lam[None, :], Bw, gamma=1.0)


def _u_statistic(P, g):
    """Unbiased two-sample U-statistic on the diagram kernel matrix ``P``."""
    g = np.asarray(g, dtype=bool)
    i0, i1 = np.flatnonzero(g), np.flatnonzero(~g)
    n0, n1 = len(i0), len(i1)
    w0 = (P[np.ix_(i0, i0)].sum() - np.diag(P)[i0].sum()) / (n0 * (n0 - 1)) if n0 > 1 else 0.0
    w1 = (P[np.ix_(i1, i1)].sum() - np.diag(P)[i1].sum()) / (n1 * (n1 - 1)) if n1 > 1 else 0.0
    b = P[np.ix_(i0, i1)].sum() / (n0 * n1) if n0 and n1 else 0.0
    return float(w0 + w1 - 2.0 * b)


def test_han(diags0, diags1, weight="persistence", weight_power=1.0,
             scales=(0.5, 1.0, 2.0, 4.0), aggregate=True, n_perm=200,
             seed=None):
    """Han-Kim-Kim kernel permutation test on weighted persistence intensity.

    Args:
        diags0, diags1: lists (over samples) of lists (per homology dim) of
            ``(k, 2)`` birth-death arrays.
        weight: "persistence" (w(z) = (d - b)^weight_power, default) or None
            (unweighted).
        weight_power: exponent q of the persistence weight.
        scales: multipliers for the per-coordinate median pairwise scale that
            form the bandwidth grid (Aggtest).
        aggregate: if True (default), run the bandwidth-aggregated test
            (Aggtest); otherwise a single bandwidth ``(median, median)``.
        n_perm: number of label permutations (default 200).
        seed: RNG seed for the permutations.

    Returns:
        float p-value in [0, 1].
    """
    d0, d1, _ = _validate(diags0, diags1)
    n0, n1 = len(d0), len(d1)
    N = n0 + n1
    pooled = d0 + d1
    rng = np.random.default_rng(seed)

    # point-level data: (b, d) coordinates and persistence weights
    X = _flatten_bd(pooled)
    if len(X) == 0:
        return 1.0
    pers = X[:, 1] - X[:, 0]
    if weight == "persistence":
        weights = pers ** float(weight_power)
    elif weight is None:
        weights = np.ones(len(X))
    else:
        raise ValueError(f"unknown weight: {weight!r}")

    # per-diagram point membership
    counts = []
    for d in pooled:
        c = 0
        for dgm in d:
            c += len(_points(dgm))
        counts.append(c)
    Bw = _membership(counts, N, weights=weights)

    med = _median_pairwise_scale(X)
    if aggregate:
        grid = [med * s for s in scales]
    else:
        grid = [med]

    perms = _perm_groups(N, n0, n_perm, rng)
    stat_obs = []
    stat_null = np.zeros((len(grid), n_perm))
    for li, lam in enumerate(grid):
        P = _weighted_diagram_kernel(X, Bw, lam)
        g0 = np.zeros(N, dtype=bool)
        g0[:n0] = True
        stat_obs.append(_u_statistic(P, g0))
        for k in range(n_perm):
            stat_null[li, k] = _u_statistic(P, perms[k])
    stat_obs = np.asarray(stat_obs)

    if not aggregate:
        return _perm_pvalue(stat_obs[0], stat_null[0], "greater")

    # Aggtest: rank p-values per bandwidth over the B + 1 replicates
    all_stats = np.concatenate([stat_null, stat_obs[:, None]], axis=1)  # (n_bw, B+1)
    B1 = n_perm + 1
    ranks = (B1 - np.argsort(np.argsort(all_stats, axis=1), axis=1)) / B1
    A = ranks.min(axis=0)
    return float((1.0 + np.count_nonzero(A <= A[-1])) / B1)


def _flatten_bd(diags):
    pts = []
    for d in diags:
        for dgm in d:
            p = _points(dgm)
            if len(p):
                pts.append(p)
    return np.vstack(pts) if pts else np.empty((0, 2))