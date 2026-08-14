"""Betti-curve helpers for the smoothed bootstrap (Roycraft-Krebs-Polonik).

Re-exports the canonical ``betti_curve`` from ``tda2s.vec`` and adds the
sample-level aggregate used by smoothed-bootstrap statistics.
"""
from __future__ import annotations

import numpy as np

from tda2s.vec import betti_curve as _vec_betti_curve


def _max_death(diagrams):
    """Largest finite death across a list of per-dim (k, 2) arrays."""
    m = 0.0
    for dgm in diagrams:
        dgm = np.asarray(dgm, dtype=float)
        if dgm.ndim == 2 and dgm.size:
            finite = dgm[np.isfinite(dgm[:, 1]), 1]
            if finite.size:
                m = max(m, float(finite.max()))
    return m


def betti_curve(diagrams, interval=None, n_points=100):
    """Persistent Betti-number curve of ONE sample's diagram list.

    Args:
        diagrams: list of (k, 2) per-dim (birth, death) arrays (one sample).
        interval: (t_min, t_max) grid; defaults to (0, max death).
        n_points: grid resolution.

    Returns:
        ``(t_grid, betti_matrix)`` with ``betti_matrix[d]`` the B_d(t) curve.
    """
    iv = interval if interval is not None else (0.0, _max_death(diagrams) + 1e-9)
    grid = np.linspace(iv[0], iv[1], n_points)
    matrix = _vec_betti_curve(diagrams, interval=iv, n_points=n_points)
    return grid, matrix


def mean_betti_curve(sample_diagrams, interval=None, n_points=100):
    """Mean Betti curve over a sample of diagrams (for bootstrap statistics).

    All samples are evaluated on ONE shared grid. When ``interval`` is not
    given it is derived from the *pooled* diagrams, not per sample: averaging
    curves that were each sampled on their own grid would mix incomparable
    abscissae and silently distort the mean.

    Args:
        sample_diagrams: list over samples of list of (k, 2) per-dim arrays.
        interval: shared (t_min, t_max); defaults to (0, pooled max death).
        n_points: grid resolution.

    Returns:
        ``(t_grid, mean_betti)`` with ``mean_betti[d]`` a length-``n_points``
        curve averaged over the sample.
    """
    if interval is None:
        pooled = 0.0
        for diags in sample_diagrams:
            pooled = max(pooled, _max_death(diags))
        interval = (0.0, pooled + 1e-9) if pooled > 0 else (0.0, 1.0)
    grid = np.linspace(interval[0], interval[1], n_points)
    if not sample_diagrams:
        return grid, np.zeros((1, n_points))
    curves = [_vec_betti_curve(diags, interval=interval, n_points=n_points)
              for diags in sample_diagrams]
    return grid, np.mean(np.stack(curves), axis=0)