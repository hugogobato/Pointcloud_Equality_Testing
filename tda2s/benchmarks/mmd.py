"""Kernel maximum mean discrepancy (MMD) two-sample test on diagram points.

Reference: R. Kwitt, S. Huber, M. Niethammer, W. Lin and U. Bauer,
"Statistical topological data analysis - a kernel perspective", NeurIPS 2015.

Each diagram is embedded as the (unweighted) sum of Gaussian kernels over its
points in ``(b, d, persistence)`` (default) or ``(b, d)`` coordinates:

    k(p, q) = exp(-||p - q||^2 / (2 sigma^2)),
    K(D, D') = sum_{p in D} sum_{q in D'} k(p, q).

The two groups are compared through the biased MMD^2 estimate between the
empirical kernel-mean embeddings of the two diagram distributions; sigma is
set by the median heuristic over the pooled point set.  The null distribution
is obtained by permuting the diagram labels (200 permutations by default).

The diagram-level kernel matrix is precomputed once as block sums of the
point-level Gaussian kernel, so the permutation loop is cheap. Those block sums
are accumulated one column block at a time (``_gaussian_gram_blocks``): the
point-level Gram is ~18k x 18k at Phase 0 benchmark size, which is 2.6 GB that
the ``(n_samples, n_samples)`` result never requires.
"""
from __future__ import annotations

import numpy as np

from ._common import (_gaussian_gram_blocks, _median_euclidean, _membership,
                      _perm_groups, _perm_pvalue, _points, _validate)


def _point_features(diags, coords):
    """Per-sample coordinate matrix for all diagram points of ``diags``."""
    feats = []
    counts = []
    for d in diags:
        block = []
        for dgm in d:
            p = _points(dgm)
            if len(p):
                if coords == "bdp":
                    p = np.column_stack([p, p[:, 1] - p[:, 0]])
                block.append(p)
        x = np.vstack(block) if block else np.empty((0, 2 if coords == "bd" else 3))
        feats.append(x)
        counts.append(len(x))
    return feats, counts


def _mmd2(P, g):
    """Biased MMD^2 between the two groups given labels ``g`` (bool array)."""
    g = np.asarray(g, dtype=bool)
    n0, n1 = int(g.sum()), len(g) - int(g.sum())
    w0 = P[np.ix_(g, g)].sum() / (n0 * n0)
    w1 = P[np.ix_(~g, ~g)].sum() / (n1 * n1)
    b = P[np.ix_(g, ~g)].sum() / (n0 * n1)
    return float(w0 + w1 - 2.0 * b)


def test_mmd(diags0, diags1, sigma=None, coords="bdp", n_perm=200, seed=None):
    """Kernel MMD two-sample test between two groups of diagrams.

    Args:
        diags0, diags1: lists (over samples) of lists (per homology dim) of
            ``(k, 2)`` birth-death arrays.
        sigma: Gaussian bandwidth; ``None`` (default) = median heuristic over
            the pooled point set.
        coords: point coordinates, "bdp" = (birth, death, persistence) or
            "bd" = (birth, death).
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

    feats, counts = _point_features(pooled, coords)
    npts = sum(counts)
    if npts == 0:
        return 1.0

    X = np.vstack(feats)
    if sigma is None:
        sigma = _median_euclidean(X)

    # diagram-level kernel matrix, accumulated blockwise: the point-level Gram
    # is ~18k x 18k at benchmark size and is never needed in full.
    P = _gaussian_gram_blocks(X, _membership(counts, N),
                              gamma=1.0 / (2.0 * sigma * sigma))

    obs_group = np.zeros(N, dtype=bool)
    obs_group[:n0] = True
    obs = _mmd2(P, obs_group)

    perms = _perm_groups(N, n0, n_perm, rng)
    null = np.array([_mmd2(P, perms[k]) for k in range(n_perm)])
    return _perm_pvalue(obs, null, "greater")
