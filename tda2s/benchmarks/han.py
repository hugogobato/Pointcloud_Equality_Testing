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


def han_kernels(diags, weight="persistence", weight_power=1.0,
                scales=(0.5, 1.0, 2.0, 4.0), aggregate=True, epsilon=0.0):
    """Per-bandwidth diagram kernel matrices of a pooled sample, before labelling.

    Both the bandwidth grid (a multiple of the pooled median pairwise scale)
    and the kernel matrices depend only on the pooled diagrams, so a caller
    comparing several label draws over one sample (the Phase 2 imbalance
    sweep) builds them once here and reads them back per split through
    :func:`test_han_from_kernels`. This is the whole cost of the test: the
    Aggtest permutation loop is block sums on the matrices returned here.

    Args:
        diags: list over samples of lists (per homology dim) of ``(k, 2)``
            birth-death arrays, in whatever order the caller will label them.
        weight, weight_power, scales, aggregate, epsilon: as in
            :func:`test_han`. ``aggregate`` decides the grid size only; pass
            the same value to :func:`test_han_from_kernels`.

    Returns:
        list of ``(N, N)`` matrices, one per bandwidth, or ``None`` when the
        pooled sample has no diagram points above ``epsilon``.
    """
    pooled = [list(d) for d in diags]
    N = len(pooled)
    X, counts = _flatten_bd(pooled, epsilon=epsilon)
    if len(X) == 0:
        return None
    pers = X[:, 1] - X[:, 0]
    if weight == "persistence":
        weights = pers ** float(weight_power)
    elif weight is None:
        weights = np.ones(len(X))
    else:
        raise ValueError(f"unknown weight: {weight!r}")

    Bw = _membership(counts, N, weights=weights)
    med = _median_pairwise_scale(X)
    grid = [med * s for s in scales] if aggregate else [med]
    return [_weighted_diagram_kernel(X, Bw, lam) for lam in grid]


def test_han_from_kernels(Ps, group, aggregate=True, n_perm=200, seed=None):
    """Han-Kim-Kim p-value from precomputed per-bandwidth kernel matrices.

    Args:
        Ps: list of ``(N, N)`` matrices from :func:`han_kernels`, or ``None``
            (degenerate sample; the p-value is then 1).
        group: ``(N,)`` boolean mask in the matrices' row order, ``True`` =
            group 0.
        aggregate: must match the value passed to :func:`han_kernels`.
        n_perm: number of label permutations (default 200).
        seed: RNG seed for the permutations.

    Returns:
        float p-value in [0, 1].
    """
    if Ps is None:
        return 1.0
    g = np.asarray(group, dtype=bool)
    N = Ps[0].shape[0]
    if g.shape != (N,):
        raise ValueError(f"group mask has shape {g.shape}, expected ({N},)")
    n0 = int(g.sum())
    if n0 == 0 or n0 == N:
        raise ValueError("each group must contain at least one diagram")
    rng = np.random.default_rng(seed)

    perms = _perm_groups(N, n0, n_perm, rng)
    stat_obs = np.empty(len(Ps))
    stat_null = np.zeros((len(Ps), n_perm))
    for li, P in enumerate(Ps):
        stat_obs[li] = _u_statistic(P, g)
        for k in range(n_perm):
            stat_null[li, k] = _u_statistic(P, perms[k])

    if not aggregate:
        return _perm_pvalue(stat_obs[0], stat_null[0], "greater")

    # Aggtest: rank p-values per bandwidth over the B + 1 replicates
    all_stats = np.concatenate([stat_null, stat_obs[:, None]], axis=1)  # (n_bw, B+1)
    B1 = n_perm + 1
    ranks = (B1 - np.argsort(np.argsort(all_stats, axis=1), axis=1)) / B1
    A = ranks.min(axis=0)
    return float((1.0 + np.count_nonzero(A <= A[-1])) / B1)


def test_han(diags0, diags1, weight="persistence", weight_power=1.0,
             scales=(0.5, 1.0, 2.0, 4.0), aggregate=True, n_perm=200,
             seed=None, epsilon=0.0):
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
        epsilon: diagram points with persistence below this threshold are
            dropped before the kernel is built (default 0.0 = the published
            weighted intensity over all points).  Near-diagonal points carry
            weight ``(d - b)^q < epsilon^q``, so the perturbation of the
            diagram kernel is bounded by their kernel contribution and the
            permutation null remains exactly valid; ``epsilon`` only trades a
            negligible power shift against the O(n^2) point-level Gram cost
            (a sweep convenience).

    Returns:
        float p-value in [0, 1].
    """
    d0, d1, _ = _validate(diags0, diags1)
    n0, n1 = len(d0), len(d1)
    N = n0 + n1

    Ps = han_kernels(d0 + d1, weight=weight, weight_power=weight_power,
                     scales=scales, aggregate=aggregate, epsilon=epsilon)
    g0 = np.zeros(N, dtype=bool)
    g0[:n0] = True
    return test_han_from_kernels(Ps, g0, aggregate=aggregate, n_perm=n_perm,
                                 seed=seed)


def _flatten_bd(diags, epsilon=0.0):
    pts, counts = [], []
    for d in diags:
        n = 0
        for dgm in d:
            p = _points(dgm)
            if epsilon > 0.0 and len(p):
                p = p[p[:, 1] - p[:, 0] >= epsilon]
            if len(p):
                pts.append(p)
                n += len(p)
        counts.append(n)
    return (np.vstack(pts) if pts else np.empty((0, 2))), counts