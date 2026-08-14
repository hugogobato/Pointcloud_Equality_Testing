"""Shared internal helpers for the competitor wrappers in ``tda2s.benchmarks``.

Implementation details of the wrappers, not part of the public API.

Diagram conventions (uniform across wrappers):
    * ``diags0`` / ``diags1``: lists over samples of lists (indexed by homology
      dim) of ``(k, 2)`` birth-death arrays.
    * All samples of a group share the same number of homology dimensions.
"""
from __future__ import annotations

import numpy as np


def _validate(diags0, diags1):
    """Validate the uniform diagram-list input format.

    Returns:
        ``(diags0, diags1, n_dims)`` with the diagrams as plain lists and the
        (common) number of homology dimensions.
    """
    d0 = [list(d) for d in diags0]
    d1 = [list(d) for d in diags1]
    if not d0 or not d1:
        raise ValueError("each group must contain at least one diagram")
    nd = len(d0[0])
    for d in d0 + d1:
        if len(d) != nd:
            raise ValueError(
                "all diagrams must contain the same number of homology dims")
    return d0, d1, nd


def _points(dgm):
    """``(k, 2)`` birth-death array -> float array, dropping empty/infinite rows."""
    a = np.asarray(dgm, dtype=float)
    if a.ndim != 2 or a.size == 0:
        return np.empty((0, 2))
    return a[np.isfinite(a).all(axis=1)]


def _persistence(dgm):
    a = _points(dgm)
    return a[:, 1] - a[:, 0]


def _flatten_points(diags):
    """All points (across samples and homology dims) -> ``(n, 2)`` array."""
    blocks = [_points(dgm) for d in diags for dgm in d]
    blocks = [b for b in blocks if len(b)]
    return np.vstack(blocks) if blocks else np.empty((0, 2))


def _median_pairwise_scale(points, coords=None, max_sample=2000, seed=0):
    """Median pairwise absolute difference along each coordinate of ``points``.

    Used to set per-coordinate bandwidths for kernel-based tests.  The
    subsample is deterministic (fixed internal seed) so results do not depend
    on the calling test's RNG.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return np.ones(max(1, pts.shape[1] if pts.ndim > 1 else 1))
    rng = np.random.default_rng(seed)
    if len(pts) > max_sample:
        pts = pts[rng.choice(len(pts), max_sample, replace=False)]
    if coords is None:
        coords = range(pts.shape[1])
    scales = []
    for j in coords:
        d = np.abs(pts[:, j, None] - pts[None, :, j])
        d = d[np.triu_indices(len(pts), 1)]
        s = np.median(d)
        scales.append(float(s) if np.isfinite(s) and s > 0 else 1.0)
    return np.asarray(scales)


def _median_euclidean(points, max_sample=2000, seed=0):
    """Median pairwise Euclidean distance (median-heuristic bandwidth)."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return 1.0
    rng = np.random.default_rng(seed)
    if len(pts) > max_sample:
        pts = pts[rng.choice(len(pts), max_sample, replace=False)]
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    d = d[np.triu_indices(len(pts), 1)]
    med = np.median(d)
    return float(med) if np.isfinite(med) and med > 0 else 1.0


#: Column-block width for :func:`_gaussian_gram_blocks`. ``n_points x CHUNK``
#: float64 is the peak allocation, so 1024 costs ~150 MB at 18k pooled points.
_GRAM_CHUNK = 1024


def _gaussian_gram_blocks(X, B, gamma=1.0, chunk=_GRAM_CHUNK):
    """``B @ exp(-gamma * ||x_i - x_j||^2) @ B.T`` without forming the full Gram.

    The kernel two-sample tests need only the ``(n_samples, n_samples)`` matrix
    of diagram-level kernel sums, but the naive route builds the point-level
    ``(n_points, n_points)`` Gram first. At the Phase 0 benchmark size (100
    clouds, H0 + H1, ~18k pooled points) that intermediate is 2.6 GB, and
    writing it as ``X[:, None, :] - X[None, :, :]`` costs a further 7.8 GB
    because the difference is materialised in 3-D before being reduced. This
    accumulates the same product one column block at a time instead, so peak
    memory is ``O(n_points * chunk)`` and the result is identical up to
    floating-point summation order.

    Args:
        X: ``(n_points, d)`` point coordinates.
        B: ``(n_samples, n_points)`` membership matrix, float. Per-point
            weights are folded in by scaling its columns, since a weighted
            kernel ``w(x) k(x, y) w(y)`` equals ``(B w) k (B w)^T``.
        gamma: exponent scale; pass ``1 / (2 sigma^2)`` for bandwidth ``sigma``.
        chunk: columns of the Gram per block.

    Returns:
        ``(n_samples, n_samples)`` array.
    """
    from scipy.spatial.distance import cdist

    npts = X.shape[0]
    S = np.empty((B.shape[0], npts))
    for start in range(0, npts, chunk):
        stop = min(start + chunk, npts)
        D2 = cdist(X, X[start:stop], "sqeuclidean")
        S[:, start:stop] = B @ np.exp(-gamma * D2)
    return S @ B.T


def _membership(counts, n_samples, weights=None):
    """``(n_samples, n_points)`` block-membership matrix for :func:`_gaussian_gram_blocks`.

    ``counts[i]`` points belong to sample ``i``, laid out contiguously in the
    order the samples were pooled. ``weights`` (per point) is folded into the
    columns when given.
    """
    npts = int(np.sum(counts))
    B = np.zeros((n_samples, npts))
    off = 0
    for i, c in enumerate(counts):
        B[i, off:off + c] = 1.0
        off += c
    if weights is not None:
        B *= np.asarray(weights, dtype=float)[None, :]
    return B


def _perm_groups(N, n0, n_perm, rng):
    """``(n_perm, N)`` boolean label arrays, exactly ``n0`` True entries per row."""
    return np.stack([rng.permutation(N) < n0 for _ in range(n_perm)])


def _perm_pvalue(obs, null, direction="greater"):
    """Phipson-Smyth permutation p-value: ``(1 + #extreme) / (1 + n_perm)``."""
    null = np.asarray(null, dtype=float)
    if direction == "greater":
        cnt = np.count_nonzero(null >= obs)
    else:
        cnt = np.count_nonzero(null <= obs)
    return float((1.0 + cnt) / (1.0 + len(null)))


def _block_means(P, g):
    """Group-split means of an ``N x N`` diagram-pair matrix ``P``.

    Args:
        P: symmetric matrix with ``P[i, j]`` = cost/kernel between diagrams i, j.
        g: ``(N,)`` bool labels (True = group 0).

    Returns:
        ``(within0, within1, between)``: mean off-diagonal ``P`` over pairs
        within group 0, within group 1, and across the two groups.
    """
    g = np.asarray(g, dtype=bool)
    i0, i1 = np.flatnonzero(g), np.flatnonzero(~g)
    n0, n1 = len(i0), len(i1)
    w0 = (P[np.ix_(i0, i0)].sum() - np.diag(P)[i0].sum()) / (n0 * (n0 - 1)) if n0 > 1 else 0.0
    w1 = (P[np.ix_(i1, i1)].sum() - np.diag(P)[i1].sum()) / (n1 * (n1 - 1)) if n1 > 1 else 0.0
    b = P[np.ix_(i0, i1)].sum() / (n0 * n1) if n0 and n1 else 0.0
    return float(w0), float(w1), float(b)


def _persistence_vectors(diags0, diags1, max_len=None):
    """Sorted (descending) zero-padded persistence vectors per diagram.

    Homology dims are concatenated; within each dim every diagram is padded
    with zeros to a common length (the max feature count across both groups,
    optionally capped at ``max_len``).

    Returns:
        ``(V0, V1)`` with shapes ``(n0, L)`` and ``(n1, L)``.
    """
    d0, d1, nd = _validate(diags0, diags1)
    parts0, parts1 = [], []
    for dim in range(nd):
        p0 = [_persistence(d[dim]) for d in d0]
        p1 = [_persistence(d[dim]) for d in d1]
        L = max([len(p) for p in p0 + p1] or [0])
        if max_len is not None:
            L = min(L, max_len)
        if L == 0:
            parts0.append(np.zeros((len(d0), 0)))
            parts1.append(np.zeros((len(d1), 0)))
            continue
        v0 = np.zeros((len(d0), L))
        v1 = np.zeros((len(d1), L))
        for i, p in enumerate(p0):
            p = np.sort(p)[::-1][:L]
            v0[i, :len(p)] = p
        for i, p in enumerate(p1):
            p = np.sort(p)[::-1][:L]
            v1[i, :len(p)] = p
        parts0.append(v0)
        parts1.append(v1)
    return (np.hstack(parts0) if parts0 else np.zeros((len(d0), 0)),
            np.hstack(parts1) if parts1 else np.zeros((len(d1), 0)))
