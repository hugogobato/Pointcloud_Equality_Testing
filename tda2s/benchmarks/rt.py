"""Robinson & Turner (2017) permutation two-sample test for persistence diagrams.

Reference: A. Robinson and K. Turner, "Hypothesis testing for topological data
analysis", Journal of Applied and Computational Topology 1 (2017) 241-261
(arXiv:1310.7467).

The test is a randomization (label-permutation) test on a "joint loss
function".  For each homology dimension ``d`` a distance ``d_p`` between two
diagrams is fixed (bottleneck distance, p = infinity, by default; Wasserstein
W_1 via ``persim`` for ``metric="wasserstein"``).  The test statistic is the
sum over homology dimensions of the mean pairwise within-group distance over
the two groups (the paper's ``F_{p, q}`` family with q = 1):

    F(L) = sum_dims [ mean_{i<j, both in group 0} d_p(D_i, D_j)
                   + mean_{i<j, both in group 1} d_p(D_i, D_j) ].

If the grouping is sensible (alternative), within-group distances are small,
so we reject for *small* observed F: the p-value is the proportion of random
labelings whose loss is at most the observed loss, in the Phipson-Smyth
``(1 + count) / (1 + n_perm)`` convention used in the paper's Algorithm 2.

The pairwise distance matrix is precomputed once and read back under each
permutation, so the permutation loop is cheap.

``statistic="within"`` (default) reproduces the paper's joint loss.  The
``statistic="between"`` variant instead uses the mean pairwise distance
between the two groups' diagrams and rejects for large values; it is the
version described in the Phase 0.5 benchmark spec.
"""
from __future__ import annotations

import numpy as np

from ._common import _block_means, _perm_groups, _perm_pvalue, _points, _validate


def _diagram_distance(dgm1, dgm2, metric):
    a, b = _points(dgm1), _points(dgm2)
    if metric == "bottleneck":
        import gudhi as gd
        return gd.bottleneck_distance(a, b)
    if metric == "wasserstein":
        import persim
        return persim.wasserstein(a, b)
    raise ValueError(f"unknown metric: {metric!r}")


def test_rt_from_matrix(P, group, n_perm=200, statistic="within", seed=None,
                        observed=None):
    """Robinson-Turner p-value from a precomputed pairwise distance matrix.

    The pairwise matrix of a pooled sample does not depend on the labels, so
    sweeps that reuse one set of diagrams across many group assignments (e.g.
    the Phase 2 imbalance sweep, where the clouds are shared across propensity
    strengths) must compute ``P`` once and read it back here per split. This
    also keeps the expensive ``O(N^2)`` bottleneck loop out of the permutation
    count.

    Args:
        P: symmetric ``N x N`` matrix, ``P[i, j]`` = joint loss contribution
            of diagrams i, j (summed over homology dims by the caller).
        group: the observed labelling, **in the row order of** ``P``. Either an
            ``(N,)`` boolean mask with ``True`` = group 0, or an integer ``n0``
            meaning "the first ``n0`` rows are group 0". The integer form is
            only correct when ``P``'s rows are already sorted by label; a
            caller that shares one matrix across several label draws (the Phase
            2 sweep) must pass the mask, or the observed statistic is evaluated
            under an arbitrary labelling and the p-value degenerates to a draw
            from the null.
        n_perm: number of label permutations (default 200).
        statistic: ``"within"`` (joint loss; reject for small) or
            ``"between"`` (mean cross-group distance; reject for large).
        seed: RNG seed for the permutations.
        observed: optional precomputed observed statistic; computed when None.

    Returns:
        float p-value in [0, 1].
    """
    N = P.shape[0]
    rng = np.random.default_rng(seed)
    if np.ndim(group) == 0:
        n0 = int(group)
        obs_group = np.zeros(N, dtype=bool)
        obs_group[:n0] = True
    else:
        obs_group = np.asarray(group, dtype=bool)
        if obs_group.shape != (N,):
            raise ValueError(
                f"group mask has shape {obs_group.shape}, expected ({N},)")
        n0 = int(obs_group.sum())
    if observed is None:
        w0, w1, b = _block_means(P, obs_group)
        observed = (w0 + w1) if statistic == "within" else b
    perms = _perm_groups(N, n0, n_perm, rng)
    null = np.empty(n_perm)
    for k in range(n_perm):
        w0, w1, b = _block_means(P, perms[k])
        null[k] = (w0 + w1) if statistic == "within" else b
    direction = "less" if statistic == "within" else "greater"
    return _perm_pvalue(observed, null, direction)


def test_rt(diags0, diags1, metric="bottleneck", statistic="within",
            n_perm=200, seed=None):
    """Robinson-Turner permutation test between two groups of diagrams.

    Args:
        diags0, diags1: lists (over samples) of lists (per homology dim) of
            ``(k, 2)`` birth-death arrays.
        metric: pairwise diagram distance, "bottleneck" (default) or
            "wasserstein" (W_1).
        statistic: "within" (paper's joint loss; reject for small values) or
            "between" (mean cross-group distance; reject for large values).
        n_perm: number of label permutations (default 200).
        seed: RNG seed for the permutations.

    Returns:
        float p-value in [0, 1].
    """
    d0, d1, nd = _validate(diags0, diags1)
    n0, n1 = len(d0), len(d1)
    N = n0 + n1
    pooled = d0 + d1
    rng = np.random.default_rng(seed)

    P = np.zeros((N, N))
    for dim in range(nd):
        for i in range(N):
            for j in range(i + 1, N):
                v = _diagram_distance(pooled[i][dim], pooled[j][dim], metric)
                P[i, j] = P[j, i] = v

    obs_group = np.zeros(N, dtype=bool)
    obs_group[:n0] = True
    w0, w1, b = _block_means(P, obs_group)
    obs = (w0 + w1) if statistic == "within" else b

    perms = _perm_groups(N, n0, n_perm, rng)
    null = np.empty(n_perm)
    for k in range(n_perm):
        w0, w1, b = _block_means(P, perms[k])
        null[k] = (w0 + w1) if statistic == "within" else b

    direction = "less" if statistic == "within" else "greater"
    return _perm_pvalue(obs, null, direction)
