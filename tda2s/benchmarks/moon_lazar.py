"""Moon & Lazar (2023) two-stage test on persistence images, as published.

Reference: C. Moon and N. A. Lazar, "Hypothesis testing for shapes using
vectorized persistence diagrams", J. R. Stat. Soc. Ser. C 72(3):628-648 (2023),
doi:10.1093/jrsssc/qlad024. Preprint arXiv:2006.05466, from which the section
and algorithm numbers below are taken. Paper accessed in full; no author code
was released, so this is a direct transcription of Algorithm 1.

Vectorisation (paper Section 2.2)
---------------------------------
A diagram ``PD = {(birth, death)}`` is transformed to ``PD_t = {(u = birth,
v = death - birth)}``. With a Gaussian smoothing function

    f_{(u,v)}(x, y | h) = 1/(2 pi h^2) exp(-((x-u)^2 + (y-v)^2) / (2 h^2))

and a weight ``w(u, v)``, the persistence surface is
``rho(x, y) = sum_{(u,v) in PD_t} f_{(u,v)}(x, y) w(u, v)`` and the persistence
image is the integral of ``rho`` over each pixel. Because ``f`` is a product of
two one-dimensional Gaussians, that integral is computed here in closed form as
a product of normal-CDF differences rather than by quadrature.

Weights offered by the paper: ``"constant"`` (``w = 1``), ``"arctan"``
(``w = arctan(R v^S)``, with the paper's ``R = S = 0.5``) and ``"linear"``
(``w = v``). Section 4.1's method comparison uses 40x40 pixels, ``h = 0.5`` and
the constant weight.

Two-stage procedure (paper Algorithm 1, Sections 3.1-3.2)
---------------------------------------------------------
1. Pre-filter: keep only pixels with ``v_x >= v_y`` (Algorithm 1 line 2). The
   complementary triangle corresponds to an empty region of the transformed
   diagram, leaving ``m(m+1)/2`` of the ``m^2`` pixels.
2. Stage I -- for each surviving pixel compute the *overall* (pooled-across-
   groups) sample standard deviation as the filter statistic (Section 3.1)::

       s^i = sqrt( sum_j sum_k (x^i_{(j,k)} - xbar^i)^2 / (n_1 + n_2 - 1) )

   and drop pixels whose filter statistic is at or below the ``C``-th
   percentile. The filter statistic is deliberately independent of the stage-II
   statistic (Bourgon et al. 2010), which is what keeps the conditional null
   valid.
3. Stage II -- a pooled-variance two-sample t-test per surviving pixel, then a
   multiple-testing adjustment (BH by default; BY also provided).

The returned scalar is the smallest adjusted p-value, so "reject at level
``alpha``" is exactly ``p <= alpha`` under the chosen FDR rule.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ._common import _points, _validate

__all__ = ["test_moon_lazar", "persistence_images"]

_WEIGHTS = {
    "constant": lambda v: np.ones_like(v),
    "arctan": lambda v: np.arctan(0.5 * np.power(np.abs(v), 0.5)),
    "linear": lambda v: v,
}


def _bh_adjusted(pvals):
    """Benjamini-Hochberg adjusted p-values (monotone), shape = input."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="mergesort")
    q = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.clip(q, 0.0, 1.0)
    return adj


def _by_adjusted(pvals):
    """Benjamini-Yekutieli adjusted p-values (BH scaled by the harmonic sum)."""
    m = len(pvals)
    c_m = np.sum(1.0 / np.arange(1, m + 1))
    return np.clip(_bh_adjusted(pvals) * c_m, 0.0, 1.0)


_ADJUST = {"bh": _bh_adjusted, "by": _by_adjusted}


def persistence_images(diagrams, resolution=40, bandwidth=0.5, weight="constant",
                       im_range=None):
    """Persistence images of a list of diagrams (Adams et al. parameterisation).

    Args:
        diagrams: list over samples of ``(k, 2)`` (birth, death) arrays -- ONE
            homology dimension.
        resolution: ``m``; the image is ``m x m`` pixels.
        bandwidth: Gaussian smoothing ``h``.
        weight: ``"constant"``, ``"arctan"`` (R = S = 0.5) or ``"linear"``.
        im_range: ``(lo, hi)`` shared by the birth and persistence axes;
            defaults to ``(0, max persistence-or-birth over the pooled input)``.
            A common square range is required because the paper's pre-filter
            compares the two pixel coordinates directly.

    Returns:
        ``(images, vx, vy)``: ``images`` of shape ``(n_samples, m*m)`` in
        row-major (x, y) order, plus the pixel-centre coordinate vectors.
    """
    try:
        wfn = _WEIGHTS[weight]
    except KeyError:
        raise ValueError(
            f"weight must be one of {sorted(_WEIGHTS)}, got {weight!r}") from None

    pts = [_points(d) for d in diagrams]
    trans = [np.column_stack([p[:, 0], p[:, 1] - p[:, 0]]) if len(p)
             else np.empty((0, 2)) for p in pts]

    if im_range is None:
        hi = 0.0
        for t in trans:
            if len(t):
                hi = max(hi, float(t.max()))
        im_range = (0.0, hi if hi > 0 else 1.0)
    lo, hi = float(im_range[0]), float(im_range[1])

    edges = np.linspace(lo, hi, resolution + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    vx = np.repeat(centres, resolution)      # x varies slowest (row-major)
    vy = np.tile(centres, resolution)

    out = np.zeros((len(trans), resolution * resolution))
    for i, t in enumerate(trans):
        if not len(t):
            continue
        u, v = t[:, 0], t[:, 1]
        w = wfn(v)
        # exact cell integral: product of 1-D normal-CDF differences
        cx = stats.norm.cdf(edges[None, :], loc=u[:, None], scale=bandwidth)
        cy = stats.norm.cdf(edges[None, :], loc=v[:, None], scale=bandwidth)
        px = np.diff(cx, axis=1)             # (n_pts, resolution)
        py = np.diff(cy, axis=1)
        out[i] = ((w[:, None] * px)[:, :, None] * py[:, None, :]).sum(axis=0).ravel()
    return out, vx, vy


def test_moon_lazar(diags0, diags1, dim_index=-1, resolution=40, bandwidth=0.5,
                    weight="constant", filter_threshold=80.0, adjust="bh",
                    im_range=None, alpha=0.05):
    """Two-stage (filter + pooled-variance t-test + FDR) test on persistence images.

    Args:
        diags0, diags1: lists (over samples) of lists (per homology dim) of
            ``(k, 2)`` birth-death arrays.
        dim_index: index into each sample's per-dimension diagram list
            (NOT the homology degree). ``-1`` is the last entry, which is the
            highest homology dimension present -- H1 for the usual ``(0, 1)``
            lists and for bare H1 lists alike. The paper's simulations use
            dimension one.
        resolution: persistence-image side length ``m`` (paper: 40).
        bandwidth: Gaussian smoothing ``h`` (paper Section 4.1: 0.5).
        weight: ``"constant"`` (paper Section 4.1), ``"arctan"`` or ``"linear"``.
        filter_threshold: ``C`` in percent; pixels at or below the ``C``-th
            percentile of the stage-I filter statistic are dropped (paper: 80).
        adjust: ``"bh"`` (paper's default) or ``"by"``.
        im_range: shared ``(lo, hi)`` for both image axes.
        alpha: retained for interface symmetry; the decision rule is
            ``returned p <= alpha``.

    Returns:
        float: smallest adjusted p-value over surviving pixels, in [0, 1].
        Returns 1.0 when no pixel survives the filter.
    """
    d0, d1, nd = _validate(diags0, diags1)
    if not -nd <= dim_index < nd:
        raise ValueError(
            f"dim_index {dim_index} out of range for {nd}-dim diagram lists")
    try:
        adjust_fn = _ADJUST[adjust]
    except KeyError:
        raise ValueError(
            f"adjust must be one of {sorted(_ADJUST)}, got {adjust!r}") from None

    pooled = [d[dim_index] for d in d0] + [d[dim_index] for d in d1]
    V, vx, vy = persistence_images(pooled, resolution=resolution,
                                   bandwidth=bandwidth, weight=weight,
                                   im_range=im_range)
    n0, n1 = len(d0), len(d1)

    keep = vx >= vy                                   # Algorithm 1, line 2
    V = V[:, keep]
    if V.shape[1] == 0:
        return 1.0

    # Stage I: overall (pooled) sample sd per pixel, Section 3.1
    xbar = V.mean(axis=0)
    s = np.sqrt(((V - xbar) ** 2).sum(axis=0) / (n0 + n1 - 1))
    t_C = np.percentile(s, filter_threshold)
    keep2 = s > t_C
    if not keep2.any():
        return 1.0
    V = V[:, keep2]

    # Stage II: pooled-variance two-sample t-test per surviving pixel
    with np.errstate(invalid="ignore", divide="ignore"):
        _, ps = stats.ttest_ind(V[:n0], V[n0:], axis=0, equal_var=True)
    ps = np.asarray(ps, dtype=float)
    ps[~np.isfinite(ps)] = 1.0                        # constant pixels

    return float(np.clip(adjust_fn(ps).min(), 0.0, 1.0))
