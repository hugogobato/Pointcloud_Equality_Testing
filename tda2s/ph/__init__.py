"""PH pipeline: point cloud -> persistence diagrams, uniform API.

Filtrations: VR (gudhi), ripser (fast), Alpha, Cech, cubical-sublevel (grid
distance transform), DTM-Rips (weighted Rips with DTM vertex weights).

Conventions
-----------
* A diagram is a ``(k, 2)`` float array of (birth, death) pairs, one per
  homology dimension, returned as a list indexed by homology dim.
* Essential classes (death = inf) are dropped: with filtrations of compact
  point sets every class dies, and tcda_uq uses the same convention.
* All filtration values are radii. Alpha *and* Delaunay-Cech report squared
  circumradii internally, so both extractors take a square root; Rips/ripser
  and DTM-Rips are already on the radius scale.
* Diagrams are cached to disk keyed by (point-cloud hash, filtration, params):
  permutation tests must never recompute PH inside the permutation loop.
* gudhi pair format: ``st.persistence()`` yields ``(dim, (birth, death))``
  tuples with essential classes as ``(dim, (birth, inf))``. Extractors are
  defensive about non-tuple payloads anyway (see ``_drop_infinite``), and
  infinite deaths are always dropped, so downstream code never sees infs.
* ``dtm-rips`` with ``homology_dims`` including 2 and an unbounded
  ``max_edge_length`` enumerates every 3-simplex of the cloud (combinatorial
  blow-up); pass a bounded ``max_edge_length`` for that configuration.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HOMOLOGY_DIMS = (0, 1, 2)


@dataclass
class PhParams:
    """Filtration parameters (cached under the hash of these + the cloud)."""

    filtration: str = "alpha"
    homology_dims: Tuple[int, ...] = _HOMOLOGY_DIMS
    max_edge_length: Optional[float] = None
    grid_size: int = 64
    dtm_k: int = 20
    cache_dir: Optional[str] = None

    def key(self, points: np.ndarray) -> str:
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(points, dtype=np.float32).tobytes())
        h.update(repr((self.filtration, self.homology_dims, self.max_edge_length,
                       self.grid_size, self.dtm_k)).encode())
        return h.hexdigest()[:24]


def _points_to_float(points) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError(f"points must be (m, d), got {pts.shape}")
    return pts


def _drop_infinite(dgm: np.ndarray) -> np.ndarray:
    dgm = np.asarray(dgm, dtype=float)
    if dgm.ndim != 2:
        return dgm.reshape(0, 2)
    if dgm.size == 0:
        return dgm.reshape(0, 2)
    return dgm[np.isfinite(dgm[:, 1])]


def _alpha_diagrams(pts: np.ndarray, max_dim: int, max_edge_length) -> List[np.ndarray]:
    import gudhi as gd

    alpha = gd.AlphaComplex(points=pts)
    st = alpha.create_simplex_tree()
    if max_edge_length is not None:
        st.prune_above_filtration(max_edge_length**2)
    st.compute_persistence()
    out = []
    for d in range(max_dim + 1):
        finite = [p for p in st.persistence() if p[0] == d
                  and isinstance(p[1], tuple) and np.isfinite(p[1][1])]
        dgm = np.array([p[1] for p in finite], dtype=float).reshape(-1, 2)
        out.append(np.sqrt(dgm) if dgm.size else dgm)
    return out


def _vr_diagrams(pts: np.ndarray, max_dim: int, max_edge_length) -> List[np.ndarray]:
    import gudhi as gd

    rc = gd.RipsComplex(points=pts, max_edge_length=max_edge_length or np.inf)
    st = rc.create_simplex_tree(max_dimension=max_dim + 1)
    st.compute_persistence()
    return [_drop_infinite(np.array(
        [p[1] for p in st.persistence() if p[0] == d], dtype=float).reshape(-1, 2))
        for d in range(max_dim + 1)]


def _ripser_diagrams(pts: np.ndarray, max_dim: int, max_edge_length) -> List[np.ndarray]:
    import ripser

    dgms = ripser.ripser(pts, maxdim=max_dim, thresh=float(max_edge_length) if max_edge_length else np.inf)["dgms"]
    out = []
    for d in range(max_dim + 1):
        dgm = _drop_infinite(np.asarray(dgms[d], dtype=float))
        out.append(dgm)
    return out


def _cech_diagrams(pts: np.ndarray, max_dim: int, max_edge_length) -> List[np.ndarray]:
    import gudhi as gd

    cc = gd.DelaunayCechComplex(points=pts)
    st = cc.create_simplex_tree()
    if max_edge_length is not None:
        st.prune_above_filtration(max_edge_length**2)
    st.compute_persistence()
    out = []
    for d in range(max_dim + 1):
        dgm = _drop_infinite(np.array(
            [p[1] for p in st.persistence() if p[0] == d], dtype=float).reshape(-1, 2))
        # DelaunayCechComplex reports *squared* circumradii (same convention as
        # AlphaComplex); take the square root so every filtration in this module
        # is on the radius scale.
        out.append(np.sqrt(dgm) if dgm.size else dgm)
    return out


def _cubical_diagrams(pts: np.ndarray, max_dim: int, grid_size: int) -> List[np.ndarray]:
    """Cubical sublevel filtration of the distance-to-cloud function on a grid.

    The grid distance transform is a piecewise-Lipschitz proxy for the
    distance function; its sublevel sets reproduce the cloud's topology at
    scales above the grid resolution.
    """
    import gudhi as gd
    from scipy.spatial import cKDTree

    # Pad relative to the cloud's extent, not by an absolute epsilon: an
    # absolute pad would make the grid (and hence the filtration) depend on the
    # cloud's units, breaking scale equivariance at the 1e-6 level.
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = 1e-6 * float(np.max(hi - lo))
    pad = pad if pad > 0 else 1e-6
    lo, hi = lo - pad, hi + pad
    axes = [np.linspace(lo[j], hi[j], grid_size) for j in range(pts.shape[1])]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, pts.shape[1])
    dist = cKDTree(pts).query(grid, k=1)[0].reshape([grid_size] * pts.shape[1])
    cc = gd.CubicalComplex(dimensions=list(dist.shape), top_dimensional_cells=dist.ravel())
    cc.compute_persistence()
    return [_drop_infinite(np.array(cc.persistence_intervals_in_dimension(d), dtype=float).reshape(-1, 2))
            for d in range(min(max_dim, 2) + 1)]


def _dtm_rips_diagrams(pts: np.ndarray, max_dim: int, dtm_k: int, max_edge_length) -> List[np.ndarray]:
    """Weighted Rips with DTM vertex weights (Anai et al. 2019 construction)."""
    import numpy as np
    from scipy.spatial.distance import cdist
    from gudhi.point_cloud.dtm import DistanceToMeasure
    from gudhi.weighted_rips_complex import WeightedRipsComplex

    # DistanceToMeasure is not callable in gudhi 3.11 (dtm(pts) raises
    # TypeError); transform after fit.
    dtm = DistanceToMeasure(k=dtm_k)
    weights = np.asarray(dtm.fit_transform(pts), dtype=float)
    # gudhi's WeightedRipsComplex expects a pairwise distance matrix.
    dist = cdist(pts, pts)
    rc = WeightedRipsComplex(distance_matrix=dist, weights=weights,
                             max_filtration=max_edge_length if max_edge_length is not None else np.inf)
    st = rc.create_simplex_tree(max_dimension=max_dim + 1)
    st.compute_persistence()
    return [_drop_infinite(np.array(
        [p[1] for p in st.persistence() if p[0] == d], dtype=float).reshape(-1, 2))
        for d in range(max_dim + 1)]


def compute_diagrams(points, filtration: str = "alpha",
                     homology_dims: Sequence[int] = _HOMOLOGY_DIMS,
                     max_edge_length: Optional[float] = None,
                     grid_size: int = 64, dtm_k: int = 20,
                     standardise: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                     cache_dir: Optional[str] = None) -> List[np.ndarray]:
    """Compute persistence diagrams of a point cloud under ``filtration``.

    Args:
        points: ``(m, d)`` point cloud.
        filtration: one of {"vr", "ripser", "alpha", "cech", "cubical", "dtm-rips"}.
        homology_dims: homology dimensions to keep.
        max_edge_length: filtration cutoff (radius units; None = unbounded).
        grid_size: grid edge length for "cubical".
        dtm_k: DTM neighbourhood size for "dtm-rips".
        standardise: optional ``(mean, scale)`` pair of ``(d,)`` arrays applied
            to the points as ``(points - mean) / scale`` before filtration.
            Externally supplied by the caller (e.g. a fixed study-region
            background) instead of per-cloud scaling, so diagrams of different
            clouds live in one shared coordinate system.
        cache_dir: if set, diagrams are cached/loaded keyed by cloud+params hash.

    Returns:
        List of ``(k, 2)`` (birth, death) arrays, indexed by homology dim.
    """
    pts = _points_to_float(points)
    if standardise is not None:
        mean, scale = standardise
        pts = (pts - np.asarray(mean, dtype=float)) / np.asarray(scale, dtype=float)
    dims = tuple(int(d) for d in homology_dims)
    max_dim = max(dims)
    params = PhParams(filtration=filtration, homology_dims=dims,
                      max_edge_length=max_edge_length, grid_size=grid_size,
                      dtm_k=dtm_k, cache_dir=cache_dir)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"{params.key(pts)}.npz")
        if os.path.exists(path):
            with np.load(path, allow_pickle=False) as z:
                return [z[f"d{d}"] for d in dims]

    if filtration == "alpha":
        all_d = _alpha_diagrams(pts, max_dim, max_edge_length)
    elif filtration == "vr":
        all_d = _vr_diagrams(pts, max_dim, max_edge_length)
    elif filtration == "ripser":
        all_d = _ripser_diagrams(pts, max_dim, max_edge_length)
    elif filtration == "cech":
        all_d = _cech_diagrams(pts, max_dim, max_edge_length)
    elif filtration == "cubical":
        all_d = _cubical_diagrams(pts, max_dim, grid_size)
    elif filtration == "dtm-rips":
        all_d = _dtm_rips_diagrams(pts, max_dim, dtm_k, max_edge_length)
    else:
        raise ValueError(f"unknown filtration: {filtration}")

    out = [all_d[d] for d in dims]
    if cache_dir:
        np.savez(path, **{f"d{d}": arr for d, arr in zip(dims, out)})
    return out


def betti_numbers(diags: Sequence[np.ndarray], persistence_threshold: float) -> List[int]:
    """Count features with persistence strictly above a threshold, per dim.

    Args:
        diags: list of ``(k, 2)`` (birth, death) arrays (see ``compute_diagrams``).
        persistence_threshold: features with ``death - birth > threshold`` count.

    Returns:
        One count per homology dim.
    """
    counts = []
    for dgm in diags:
        dgm = np.asarray(dgm, dtype=float).reshape(-1, 2)
        counts.append(int((dgm[:, 1] - dgm[:, 0] > persistence_threshold).sum()))
    return counts