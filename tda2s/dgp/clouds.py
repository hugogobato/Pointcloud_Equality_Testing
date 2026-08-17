"""Controlled point-cloud generators (Phase 0.6).

Basic shape generators (circle, torus, sphere, cluster) plus the loop cloud
used by the covariate-driven DGP harness: ``n_loops`` circles arranged on a big
circle, each with its own radius, dialable noise and outlier fraction.

All generators return ``(n, d)`` float arrays (2-D for circle / cluster /
loops, 3-D for torus / sphere) and accept an ``rng`` argument that may be an
integer seed, ``None``, or a ``numpy`` Generator (a Generator is used as-is so
streams stay composable).

Memory discipline: generators are vectorised and allocate only ``O(n)``; they
are intended for small clouds (``n <= 300``).
"""

from __future__ import annotations

import numpy as np


def _as_rng(rng):
    """Normalise ``rng`` to a ``numpy`` Generator (Generator passthrough)."""
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def circle_cloud(n, radius=1.0, noise=0.05, rng=None):
    """Noisy circle: ``n`` points on a circle of radius ``radius`` plus Gaussian jitter.

    Args:
        n: number of points.
        radius: circle radius (the persistent ``H_1`` feature scale).
        noise: standard deviation of the isotropic Gaussian jitter.
        rng: seed or Generator.

    Returns:
        ``(n, 2)`` point cloud with one prominent ``H_1`` feature of persistence
        approximately ``radius``.
    """
    rng = _as_rng(rng)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    pts = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    return pts + rng.normal(scale=noise, size=pts.shape)


def torus_cloud(n, R=2.0, r=0.6, noise=0.05, rng=None):
    """Noisy torus: ``n`` points on a 3-D torus with major radius ``R`` and minor radius ``r``.

    Parametrised by ``u, v ~ Uniform[0, 2 pi)``:
    ``x = (R + r cos v) cos u``, ``y = (R + r cos v) sin u``, ``z = r sin v``.

    Returns:
        ``(n, 3)`` point cloud with one prominent ``H_2`` feature and one
        prominent ``H_1`` feature (the two independent cycles of the torus).
    """
    rng = _as_rng(rng)
    u = rng.uniform(0.0, 2.0 * np.pi, size=n)
    v = rng.uniform(0.0, 2.0 * np.pi, size=n)
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    pts = np.column_stack([x, y, z])
    return pts + rng.normal(scale=noise, size=pts.shape)


def sphere_cloud(n, radius=1.0, noise=0.05, rng=None):
    """Noisy 2-sphere: ``n`` points uniform on the sphere surface plus Gaussian jitter.

    Uses the standard ``z ~ Uniform(-1, 1)``, ``phi ~ Uniform(0, 2 pi)``
    parametrisation.

    Returns:
        ``(n, 3)`` point cloud with one prominent ``H_2`` feature of scale
        approximately ``radius``.
    """
    rng = _as_rng(rng)
    z = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    rho = np.sqrt(np.maximum(0.0, 1.0 - z**2))
    pts = radius * np.column_stack([rho * np.cos(phi), rho * np.sin(phi), z])
    return pts + rng.normal(scale=noise, size=pts.shape)


def cluster_cloud(n, n_clusters=3, spread=3.0, noise=0.2, rng=None):
    """Gaussian blob clusters: ``n`` points split across ``n_clusters`` blobs.

    Cluster centers sit on a lattice with spacing ``spread``; points within a
    cluster are Gaussian with standard deviation ``noise``.

    Returns:
        ``(n, 2)`` point cloud with ``n_clusters`` connected components at the
        ``H_0`` scale ``spread``.
    """
    rng = _as_rng(rng)
    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1")
    side = int(np.ceil(np.sqrt(n_clusters)))
    centers = []
    for k in range(n_clusters):
        i, j = k % side, k // side
        centers.append([(i - (side - 1) / 2) * spread, (j - (side - 1) / 2) * spread])
    centers = np.asarray(centers, dtype=float)
    sizes = np.full(n_clusters, n // n_clusters)
    sizes[: n % n_clusters] += 1
    pts = []
    for k, size in enumerate(sizes):
        pts.append(centers[k] + rng.normal(scale=noise, size=(size, 2)))
    return np.vstack(pts)


def split_cluster_cloud(n_per_blob, n_blobs, separation=3.0, noise=0.15,
                        deterministic=False, n_gon=12, rng=None):
    """Cloud of ``n_blobs`` Gaussian blobs on a regular polygon of side ``separation``.

    This is the WP1.1 / Phase 4.4 "cluster splitting" DGP: the H_0 diagram of a
    cloud with ``n_blobs`` well-separated blobs has ``n_blobs - 1`` finite
    classes, all dying at the blob-merge scale. Two blobs (one finite class)
    versus three blobs (two finite classes, equilateral arrangement so both
    classes share the merge law) give equal mean power-weighted silhouettes but
    different diagram laws; the mean is preserved because the silhouette is a
    normalized average and the merge-scale law is unchanged by the extra blob.

    With ``deterministic=True`` each blob is a fixed regular ``n_gon``-gon of
    radius ``noise`` (degenerate randomness), so under the radius-convention
    filtrations of ``tda2s.ph`` (alpha, Delaunay-Cech) the merge scale is
    exactly ``(separation - 2 * noise) / 2`` and the mean-silhouette equality
    holds realization by realization, not only in expectation. For the exact
    orientations used here, ``n_gon`` must be even for two blobs and divisible
    by 12 for three blobs, so a vertex sits exactly on each relevant inter-blob
    axis; with the default
    ``n_gon=12`` the within-blob classes all die at
    ``noise * sin(pi / n_gon)`` (0.0388 at the defaults), well below the
    persistence threshold used to isolate the merge classes.

    Note that the two arms differ in cardinality (``n_blobs * n_gon`` points),
    so any use of this generator as a null DGP must either state cloud size as
    part of the treatment or equalise it by subsampling.

    Args:
        n_per_blob: points per blob in the stochastic case (Gaussian around the
            blob centre with scale ``noise``). Ignored when
            ``deterministic=True``, which always emits ``n_gon`` vertices.
        n_blobs: number of blobs, placed at the vertices of a regular polygon
            with side length ``separation`` (2 blobs: a segment; 3 blobs: an
            equilateral triangle; more: the regular polygon, so consecutive
            neighbours are at distance ``separation``).
        separation: distance between neighbouring blob centres.
        noise: per-blob scale: Gaussian standard deviation (stochastic) or
            blob radius (deterministic).
        deterministic: use fixed regular polygons instead of Gaussian blobs.
        n_gon: vertices per blob in the deterministic case (default 12; use an
            even count for the two-blob line and a multiple of 12 for three
            blobs in the exact orientation).
        rng: seed or Generator.

    Returns:
        ``(n, 2)`` point cloud.
    """
    rng = _as_rng(rng)
    n_blobs = int(n_blobs)
    if n_blobs < 2:
        raise ValueError("n_blobs must be >= 2 (one blob has no finite H_0 class)")
    # Vertices of a regular n-gon with consecutive side length `separation`.
    R = separation / (2.0 * np.sin(np.pi / n_blobs))
    angles = 2.0 * np.pi * np.arange(n_blobs) / n_blobs
    centers = R * np.column_stack([np.cos(angles), np.sin(angles)])

    pts = []
    for k in range(n_blobs):
        if deterministic:
            theta = 2.0 * np.pi * np.arange(int(n_gon)) / int(n_gon)
            blob = centers[k] + noise * np.column_stack([np.cos(theta), np.sin(theta)])
        else:
            blob = centers[k] + rng.normal(scale=noise, size=(n_per_blob, 2))
        pts.append(blob)
    return np.vstack(pts)


def merge_staircase_cloud(merge_scales, noise=0.15, deterministic=True,
                          n_gon=12, rng=None):
    """Cloud whose H_0 classes die at the prescribed merge scales, one by one.

    This is the Phase 4 reverse-witness (W2') arm-0 DGP: ``len(merge_scales) + 1`` blobs on
    a line, placed so that consecutive blobs merge at exactly the given
    scales.  With ``deterministic=True`` each blob is a fixed regular
    ``n_gon``-gon of radius ``noise`` and the alpha-filtration merge scales
    are exactly ``(distance - 2 * noise) / 2``, so the H_0 diagram (filtered
    above the within-blob class scale ``noise * sin(pi / n_gon)``) is
    realization-invariant: ``{(0, s_1), ..., (0, s_k)}``.

    Args:
        merge_scales: the H_0 death scales, in the order the blobs merge
            (blob ``k+1`` sits ``2 * s_{k+1} + 2 * noise`` beyond blob ``k``).
        noise: blob radius in the deterministic case.
        deterministic: fixed regular polygons (exact merges); the stochastic
            Gaussian case is a diagnostic only and does not keep the deaths
            exact.
        n_gon: vertices per blob in the deterministic case (must be even so a
            vertex sits on the line's inter-blob axis).
        rng: seed or Generator.

    Returns:
        ``(n, 2)`` point cloud with one H_0 class per merge scale.
    """
    rng = _as_rng(rng)
    scales = [float(s) for s in merge_scales]
    if not scales or any(s <= 0.0 for s in scales):
        raise ValueError("merge_scales must be a non-empty list of positive scales")
    centers = [(0.0, 0.0)]
    p = 0.0
    for s in scales:
        p += 2.0 * s + 2.0 * float(noise)
        centers.append((p, 0.0))
    pts = []
    for cx, cy in centers:
        if deterministic:
            theta = 2.0 * np.pi * np.arange(int(n_gon)) / int(n_gon)
            blob = np.column_stack([cx + noise * np.cos(theta),
                                    cy + noise * np.sin(theta)])
        else:
            blob = np.column_stack([cx, cy]) + rng.normal(
                scale=noise, size=(int(n_gon), 2))
        pts.append(blob)
    return np.vstack(pts)


def loops_cloud(n, n_loops, radius=1.0, noise=0.05, outlier_fraction=0.0, rng=None):
    """Cloud of ``n_loops`` noisy circles arranged on a big circle.

    Each loop is a circle of its own radius; loop centers sit at equally spaced
    angles on a big circle whose radius guarantees the loops stay separated
    (adjacent centers at least 4 x the largest loop radius apart), so the alpha
    complex of the whole cloud has one prominent ``H_1`` feature per loop with
    persistence approximately equal to that loop's radius. A fraction
    ``outlier_fraction`` of the points are drawn uniformly over the bounding
    box of the arrangement (topological clutter).

    Args:
        n: total number of points.
        n_loops: number of loops (prominent ``H_1`` features).
        radius: loop radius; either a scalar (all loops equal) or a length
            ``n_loops`` array of per-loop radii.
        noise: standard deviation of the isotropic Gaussian jitter.
        outlier_fraction: fraction of points placed uniformly in the bounding
            box (default 0.0).
        rng: seed or Generator.

    Returns:
        ``(n, 2)`` point cloud.
    """
    rng = _as_rng(rng)
    n_loops = int(n_loops)
    if n_loops < 1:
        raise ValueError("n_loops must be >= 1")
    radii = np.full(n_loops, float(radius)) if np.isscalar(radius) else np.asarray(radius, dtype=float)
    if radii.shape[0] != n_loops:
        raise ValueError("radius array must have length n_loops")
    max_r = float(radii.max())
    n_outliers = int(round(outlier_fraction * n))
    n_loop_pts = n - n_outliers
    per_loop, remainder = divmod(n_loop_pts, n_loops)
    if per_loop < 8:
        raise ValueError("n too small for n_loops (need >= 8 points per loop)")

    big_R = 2.5 * max_r if n_loops > 1 else 0.0
    angles = 2.0 * np.pi * np.arange(n_loops) / n_loops
    centers = big_R * np.column_stack([np.cos(angles), np.sin(angles)])

    pts = []
    for k in range(n_loops):
        size = per_loop + (1 if k < remainder else 0)
        theta = rng.uniform(0.0, 2.0 * np.pi, size=size)
        circle = centers[k] + radii[k] * np.column_stack([np.cos(theta), np.sin(theta)])
        pts.append(circle + rng.normal(scale=noise, size=circle.shape))
    if n_outliers > 0:
        span = big_R + max_r
        pts.append(rng.uniform(-span, span, size=(n_outliers, 2)))
    return np.vstack(pts)
