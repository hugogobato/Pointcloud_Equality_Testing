"""Vectorisation stack: persistence diagrams -> fixed-size feature vectors.

Uniform entry point ``vectorise(diags, representation, **kwargs)`` dispatches
to per-representation functions. Representations:

* ``silhouette``: power-weighted persistence silhouette (gudhi
  ``representations.Silhouette``), same parameterisation as
  ``tcda_uq.silhouette.core.compute_silhouette`` (interval (0.0, 0.2),
  resolution 100, keep_endpoints=True, power r=3).
* ``landscape``: persistence landscapes (gudhi ``representations.Landscape``).
* ``betti``: Betti curves ``B_d(t) = #{features: birth <= t < death}`` on a
  t-grid (implemented locally).
* ``euler``: Euler curves ``sum_d (-1)^d B_d(t)``.
* ``image``: persistence images (gudhi ``representations.PersistenceImage``).
* ``measure``: persistence measure (Divol-Lacombe style): the diagram is a
  measure ``mu = sum_p w_p * delta_{(b, (b+d)/2)}`` over the (birth, mid)
  plane, projected onto a fixed 2-D grid of bins.

Conventions
-----------
* ``diags``: list of ``(k, 2)`` (birth, death) arrays, one per homology dim
  (same format as ``tda2s.ph.compute_diagrams``); deaths must be finite
  (essential classes are dropped upstream).
* Each homology dim is vectorised separately, so outputs carry a leading
  per-dim axis; all outputs are float64 arrays.
* ``interval`` (or ``sample_range``) defaults, when not given, to ``(0.0,
  max death)`` derived from the diagrams.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


def _diagram_list(diags: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Coerce the per-dim diagram list to a list of (k, 2) float arrays."""
    out = []
    for dgm in diags:
        out.append(np.asarray(dgm, dtype=float).reshape(-1, 2))
    return out


def _default_interval(diags: Sequence[np.ndarray]) -> Tuple[float, float]:
    """Default sample range ``(0, max death)`` across all dims."""
    hi = 1.0
    for dgm in _diagram_list(diags):
        if len(dgm):
            hi = max(hi, float(dgm[:, 1].max()))
    return (0.0, hi)


def _power_weight(point: np.ndarray, r: float) -> float:
    """Power weight ``|death - birth|**r`` of a persistence point."""
    return float(np.abs(point[1] - point[0]) ** r)


def silhouette(diags: Sequence[np.ndarray], interval=(0.0, 0.2), r: float = 3.0,
               resolution: int = 100) -> np.ndarray:
    """Power-weighted persistence silhouette.

    Args:
        diags: list of ``(k, 2)`` (birth, death) arrays, one per homology dim.
        interval: sample range ``[t_min, t_max]`` of the silhouette.
        r: power-weight exponent ``w = (death - birth)**r``.
        resolution: number of grid points.

    Returns:
        ``(n_dims, resolution)`` array of silhouette values.
    """
    from gudhi.representations import Silhouette

    s = Silhouette(weight=lambda x: _power_weight(x, r), resolution=resolution,
                   sample_range=list(interval), keep_endpoints=True)
    return np.asarray(s.fit_transform(_diagram_list(diags)), dtype=float)


def landscape(diags: Sequence[np.ndarray], num_landscapes: int = 5,
              resolution: int = 100, interval: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Persistence landscapes of each dim's diagram.

    Args:
        diags: list of ``(k, 2)`` (birth, death) arrays, one per homology dim.
        num_landscapes: number of landscape functions per dim.
        resolution: number of grid points per landscape.
        interval: sample range; defaults to ``(0, max death)``.

    Returns:
        ``(n_dims, num_landscapes, resolution)`` array of landscape values.
    """
    from gudhi.representations import Landscape

    iv = interval if interval is not None else _default_interval(diags)
    l = Landscape(num_landscapes=num_landscapes, resolution=resolution,
                  sample_range=list(iv))
    out = np.asarray(l.fit_transform(_diagram_list(diags)), dtype=float)
    return out.reshape(len(diags), num_landscapes, resolution)


def betti_curve(diags: Sequence[np.ndarray], interval: Optional[Tuple[float, float]] = None,
                n_points: int = 100) -> np.ndarray:
    """Betti curves ``B_d(t) = #{features: birth <= t < death}``.

    Args:
        diags: list of ``(k, 2)`` (birth, death) arrays, one per homology dim.
        interval: t-grid range; defaults to ``(0, max death)``.
        n_points: number of t-grid points.

    Returns:
        ``(n_dims, n_points)`` array of Betti numbers on the t-grid.
    """
    iv = interval if interval is not None else _default_interval(diags)
    grid = np.linspace(iv[0], iv[1], n_points)
    rows = []
    for dgm in _diagram_list(diags):
        if len(dgm) == 0:
            rows.append(np.zeros(n_points))
            continue
        alive = (dgm[:, 0][:, None] <= grid[None, :]) & (grid[None, :] < dgm[:, 1][:, None])
        rows.append(alive.sum(axis=0, dtype=np.int64).astype(float))
    return np.stack(rows)


def euler_curve(diags: Sequence[np.ndarray], interval: Optional[Tuple[float, float]] = None,
                n_points: int = 100) -> np.ndarray:
    """Euler characteristic curve ``sum_d (-1)^d B_d(t)``.

    Args:
        diags: list of ``(k, 2)`` (birth, death) arrays, one per homology dim.
        interval: t-grid range; defaults to ``(0, max death)``.
        n_points: number of t-grid points.

    Returns:
        ``(n_points,)`` array of Euler characteristics on the t-grid.
    """
    b = betti_curve(diags, interval=interval, n_points=n_points)
    signs = np.array([(-1.0) ** d for d in range(len(diags))])
    return signs @ b


def persistence_image(diags: Sequence[np.ndarray], bandwidth: float = 0.1,
                      weight: Optional[Callable[[np.ndarray], float]] = None,
                      resolution=(10, 10),
                      interval: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Persistence images (gaussian kernels centred on diagram points).

    Args:
        diags: list of ``(k, 2)`` (birth, death) arrays, one per homology dim.
        bandwidth: gaussian kernel width.
        weight: point weight function; defaults to persistence ``death - birth``.
        resolution: ``(n_pixels_x, n_pixels_y)`` grid.
        interval: ``[t_min, t_max]`` shared by both axes; defaults to
            ``(0, max death)``.

    Returns:
        ``(n_dims, n_pixels_x, n_pixels_y)`` array of image intensities.
    """
    from gudhi.representations import PersistenceImage

    iv = interval if interval is not None else _default_interval(diags)
    w = weight if weight is not None else lambda x: x[1] - x[0]
    pi = PersistenceImage(bandwidth=bandwidth, weight=w,
                          resolution=list(resolution),
                          im_range=[iv[0], iv[1], iv[0], iv[1]])
    out = np.asarray(pi.fit_transform(_diagram_list(diags)), dtype=float)
    return out.reshape(len(diags), int(resolution[0]), int(resolution[1]))


def persistence_measure(diags: Sequence[np.ndarray],
                        weight: Optional[Callable[[np.ndarray], float]] = None,
                        interval: Optional[Tuple[float, float]] = None,
                        n_bins: int = 32) -> np.ndarray:
    """Persistence measure: ``mu = sum_p w_p * delta_{(b, (b+d)/2)}``.

    The diagram is represented as a weighted point measure on the (birth,
    mid) plane with ``mid = (b + d) / 2`` (Divol-Lacombe coordinates),
    projected onto a fixed regular grid of bins as a weighted 2-D histogram.

    Args:
        diags: list of ``(k, 2)`` (birth, death) arrays, one per homology dim.
        weight: point weight function; defaults to persistence ``death - birth``.
        interval: shared ``[t_min, t_max]`` for both the birth and mid axes;
            defaults to ``(0, max death)``.
        n_bins: number of bins along each axis.

    Returns:
        ``(n_dims, n_bins, n_bins)`` array of aggregated weights per bin.
    """
    iv = interval if interval is not None else _default_interval(diags)
    w = weight if weight is not None else lambda x: x[1] - x[0]
    edges = np.linspace(iv[0], iv[1], n_bins + 1)
    out = []
    for dgm in _diagram_list(diags):
        if len(dgm) == 0:
            out.append(np.zeros((n_bins, n_bins)))
            continue
        mid = (dgm[:, 0] + dgm[:, 1]) / 2.0
        ws = np.array([w(p) for p in dgm], dtype=float)
        hist, _, _ = np.histogram2d(dgm[:, 0], mid, bins=[edges, edges], weights=ws)
        out.append(hist)
    return np.stack(out)


def vectorise(diags: Sequence[np.ndarray], representation: str, **kwargs) -> np.ndarray:
    """Vectorise persistence diagrams under a named representation.

    Args:
        diags: list of ``(k, 2)`` (birth, death) arrays, one per homology dim.
        representation: one of {"silhouette", "landscape", "betti", "euler",
            "image", "measure"}.
        **kwargs: passed to the per-representation function (e.g. ``interval``,
            ``resolution``, ``r``).

    Returns:
        The representation vector; see the individual functions for shapes.
    """
    if representation == "silhouette":
        return silhouette(diags, **kwargs)
    if representation == "landscape":
        return landscape(diags, **kwargs)
    if representation == "betti":
        return betti_curve(diags, **kwargs)
    if representation == "euler":
        return euler_curve(diags, **kwargs)
    if representation == "image":
        return persistence_image(diags, **kwargs)
    if representation == "measure":
        return persistence_measure(diags, **kwargs)
    raise ValueError(
        f"unknown representation {representation!r}; expected one of "
        "['silhouette', 'landscape', 'betti', 'euler', 'image', 'measure']")
