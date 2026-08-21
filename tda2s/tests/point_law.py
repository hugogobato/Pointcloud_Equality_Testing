"""Established point-law two-sample tests for the Phase 5AB benchmark.

This module implements the required raw-data competitors that use all
individual points (not disjoint blocks) for testing

    H0^law: P0 = P1

under Regime I (iid metric-measure sampling).  Every method:

* caches the pooled representation once (Gram, distance matrix, graph, …)
* calibrates by pooled point-label permutation preserving (n0, n1)
* is valid under iid exchangeability; dependence, overlapping blocks,
  or label-dependent tuning are outside the validity scope
* records the exact output fields demanded by
  docs/phase5ab_point_law_benchmark_plan.md Section 3.3

Method registry
---------------
* PointMMD-Gaussian  — characteristic Gaussian kernel MMD (Gretton et al. 2012)
* EnergyDistance      — Euclidean energy distance (Szekely & Rizzo 2013)
* FriedmanRafsky-MST  — MST cross-edge count (Friedman & Rafsky 1979)
* Schilling-kNN       — k-NN same-label edge count (Schilling 1986)
* Rosenbaum-CrossMatch — minimum-weight non-bipartite matching, cross pairs
                         (Rosenbaum 2005)
* SlicedWasserstein   — fixed-projection sliced W1 average (Ramdas et al. 2015
                         as secondary geometric baseline)
* ClassifierTwoSampleTest — sample-split logistic / RF accuracy with held-out
                            label permutation (Lopez-Paz & Oquab 2017)

Identification note (Section 1.2 audit)
----------------------------------------
1. A Gaussian point kernel on Euclidean space is characteristic (Simon-Gabriel &
   Scholkopf 2018, Gretton et al. 2012) so point-level Gaussian MMD identifies
   P0=P1 under the usual moment/measurability conditions.
2. Euclidean energy distance identifies equality of distributions under finite
   first-moment conditions (Szekely & Rizzo 2013; equivalence to MMD via
   Sejdinovic et al. 2013).
3. MST, k-NN and cross-match are valid permutation statistics under iid
   point-law null; their consistency requires graph-test conditions (Friedman &
   Rafsky 1979; Schilling 1986; Rosenbaum 2005) and they are NOT
   characteristic-kernel tests.
4. A finite collection of sliced-Wasserstein projections is a sensitivity
   statistic unless the projection family is shown to be identifying;
   the implementation fixes projections before labels and labels it honestly.
5. A classifier two-sample test targets P0=P1 only relative to its classifier
   class and split protocol (Lopez-Paz & Oquab 2017); not universally consistent.
6. The simple average pairwise kernel on raw bags (mean_pairwise) remains a
   downgraded sensitivity variant and must not replace the Hilbert-Gaussian
   unordered-bag kernel in the primary RawBlockMMD claim.
If any check fails for the selected implementation, the headline comparison
must exclude it and record the reason.
"""
from __future__ import annotations

import itertools
import math
import resource
import time
import tracemalloc
from typing import Optional, Sequence, Tuple

import numpy as np

from tda2s.resample import p_value
from tda2s.tests.single_cloud import REGIME_I, _label_masks, _mmd2_from_gram

__all__ = [
    "REGIME_I",
    "point_mmd_gaussian",
    "energy_distance_test",
    "friedman_rafsky_mst",
    "schilling_knn",
    "rosenbaum_crossmatch",
    "sliced_wasserstein_test",
    "classifier_two_sample_test",
    "METHOD_REGISTRY",
]

# ---------------------------------------------------------------------------
# Shared helpers (mirrors single_cloud.py but kept local for self-containment
# in Colab notebooks)

def _as_cloud(cloud, name: str) -> np.ndarray:
    points = np.asarray(cloud, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 1:
        raise ValueError(f"{name} must have shape (n, d) with n >= 2")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain only finite values")
    return points


def _validate_cloud_pair(cloud0, cloud1) -> Tuple[np.ndarray, np.ndarray]:
    x0, x1 = _as_cloud(cloud0, "cloud0"), _as_cloud(cloud1, "cloud1")
    if x0.shape[1] != x1.shape[1]:
        raise ValueError("cloud0 and cloud1 must have the same ambient dimension")
    return x0, x1


def _require_regime(candidate: str, regime: str) -> None:
    if regime != REGIME_I:
        raise ValueError(
            f"{candidate} is only valid for declared regime {REGIME_I!r}; received {regime!r}"
        )


def _current_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    scale = 1024 if __import__("sys").platform.startswith("linux") else 1
    return int(usage.ru_maxrss * scale)


def _finish(result: dict, started: float, peak_bytes: int) -> dict:
    result["runtime_seconds"] = float(time.perf_counter() - started)
    result["peak_memory_bytes"] = int(peak_bytes)
    result["peak_memory_measurement"] = "tracemalloc_python_allocations"
    return result


def _measure(callable_):
    tracemalloc.start()
    started = time.perf_counter()
    start_rss = _current_rss_bytes()
    try:
        result = callable_()
        _, peak = tracemalloc.get_traced_memory()
        result = _finish(result, started, peak)
        result["peak_rss_bytes"] = max(start_rss, _current_rss_bytes())
        result["peak_memory_measurement"] = (
            "tracemalloc_python_allocations; peak_rss_bytes is process high-water RSS"
        )
        return result
    finally:
        tracemalloc.stop()


def _permutation_pvalue_less(observed: float, null: np.ndarray) -> float:
    """Phipson-Smyth style p-value for left-tail (small is extreme)."""
    null = np.asarray(null, dtype=float)
    if null.size == 0:
        return 1.0
    cnt = np.count_nonzero(null <= observed)
    return float((1.0 + cnt) / (1.0 + len(null)))


def _median_heuristic_bandwidth(pooled: np.ndarray, max_sample: int = 2000, seed: int = 0) -> float:
    pts = np.asarray(pooled, dtype=float)
    if len(pts) < 2:
        return 1.0
    rng = np.random.default_rng(seed)
    if len(pts) > max_sample:
        pts = pts[rng.choice(len(pts), max_sample, replace=False)]
    # pairwise Euclidean distances, upper triangular
    from scipy.spatial.distance import pdist
    dists = pdist(pts, metric="euclidean")
    med = float(np.median(dists))
    return med if np.isfinite(med) and med > 0 else 1.0


def _gaussian_gram(pooled: np.ndarray, bandwidth: float) -> np.ndarray:
    if bandwidth <= 0:
        raise ValueError("kernel bandwidth must be positive")
    from scipy.spatial.distance import cdist
    D2 = cdist(pooled, pooled, metric="sqeuclidean")
    return np.exp(-D2 / (2.0 * float(bandwidth) ** 2))


def _euclidean_distance_matrix(pooled: np.ndarray) -> np.ndarray:
    from scipy.spatial.distance import cdist
    return cdist(pooled, pooled, metric="euclidean")


def _deterministic_mst_edges(distance_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return a deterministic MST edge list, including zero-distance ties.

    ``scipy.sparse.csgraph.minimum_spanning_tree`` treats zero entries in a
    dense/sparse matrix as absent edges.  That silently produces a forest for
    repeated points, which is common in the discrete four-atom diagnostic.
    Prim's algorithm below works directly with the complete finite distance
    matrix and resolves equal-distance choices by pooled point index.
    """
    distances = np.asarray(distance_matrix, dtype=float)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be square")
    if not np.isfinite(distances).all() or (distances < 0).any():
        raise ValueError("distance_matrix must contain finite non-negative distances")
    n = distances.shape[0]
    if n < 2:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    in_tree = np.zeros(n, dtype=bool)
    best = np.full(n, np.inf, dtype=float)
    parent = np.full(n, -1, dtype=int)
    best[0] = 0.0
    rows, cols = [], []

    for _ in range(n):
        candidates = np.flatnonzero(~in_tree)
        vertex = int(candidates[np.argmin(best[candidates])])
        if not np.isfinite(best[vertex]):
            raise ValueError("distance graph is disconnected")
        in_tree[vertex] = True
        if parent[vertex] >= 0:
            rows.append(parent[vertex])
            cols.append(vertex)

        remaining = ~in_tree
        candidate_distances = distances[vertex]
        strictly_better = remaining & (candidate_distances < best)
        tied_but_lower_index = (
            remaining
            & (candidate_distances == best)
            & ((parent < 0) | (vertex < parent))
        )
        update = strictly_better | tied_but_lower_index
        best[update] = candidate_distances[update]
        parent[update] = vertex

    return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)


# ---------------------------------------------------------------------------
# Method registry (machine-readable)

METHOD_REGISTRY = {
    "PointMMD-Gaussian": {
        "method": "point_mmd_gaussian",
        "target_null": "H0^law: P0=P1",
        "validity_regime": "iid_metric_measure",
        "sampling_unit": "individual iid point",
        "kernel_or_distance": "Gaussian point kernel",
        "is_characteristic": True,
        "requires": "pooled point Gram, frozen bandwidth",
        "literature": "Gretton et al. 2012 (A1); Simon-Gabriel & Scholkopf 2018 (A2)",
        "consistency_conditions": "characteristic kernel, bounded, moment/measurability",
        "is_full_point_law_test": True,
    },
    "EnergyDistance": {
        "method": "energy_distance_test",
        "target_null": "H0^law: P0=P1",
        "validity_regime": "iid_metric_measure",
        "sampling_unit": "individual iid point",
        "kernel_or_distance": "Euclidean distance",
        "is_characteristic": True,
        "requires": "pooled Euclidean distance matrix",
        "literature": "Szekely & Rizzo 2013 (B1); Sejdinovic et al. 2013 (B2)",
        "consistency_conditions": "finite first moment",
        "is_full_point_law_test": True,
    },
    "FriedmanRafsky-MST": {
        "method": "friedman_rafsky_mst",
        "target_null": "H0^law: P0=P1",
        "validity_regime": "iid_metric_measure",
        "sampling_unit": "individual iid point",
        "kernel_or_distance": "Euclidean MST cross-edge count",
        "is_characteristic": False,
        "requires": "pooled MST cached outside permutation loop",
        "literature": "Friedman & Rafsky 1979 (C1)",
        "consistency_conditions": "graph-test conditions; valid permutation procedure, consistency per source",
        "is_full_point_law_test": True,
    },
    "Schilling-kNN": {
        "method": "schilling_knn",
        "target_null": "H0^law: P0=P1",
        "validity_regime": "iid_metric_measure",
        "sampling_unit": "individual iid point",
        "kernel_or_distance": "k-NN same-label directed edge count",
        "is_characteristic": False,
        "requires": "pooled k-NN graph cached outside permutation loop",
        "literature": "Schilling 1986 (C2)",
        "consistency_conditions": "graph-test conditions; not characteristic-kernel",
        "is_full_point_law_test": True,
    },
    "Rosenbaum-CrossMatch": {
        "method": "rosenbaum_crossmatch",
        "target_null": "H0^law: P0=P1",
        "validity_regime": "iid_metric_measure",
        "sampling_unit": "individual iid point",
        "kernel_or_distance": "minimum-weight non-bipartite matching cross-pair count",
        "is_characteristic": False,
        "requires": "pooled optimal matching cached outside permutation loop; networkx",
        "literature": "Rosenbaum 2005 (C3)",
        "consistency_conditions": "matching graph conditions; valid permutation procedure",
        "is_full_point_law_test": True,
    },
    "SlicedWasserstein": {
        "method": "sliced_wasserstein_test",
        "target_null": "H0^law: P0=P1 (sensitivity)",
        "validity_regime": "iid_metric_measure",
        "sampling_unit": "individual iid point",
        "kernel_or_distance": "sliced Wasserstein-1 average (fixed projections)",
        "is_characteristic": False,
        "requires": "fixed projections, fixed seed; pooled label permutation",
        "literature": "Ramdas et al. 2015 (D1)",
        "consistency_conditions": "finite projection family is sensitivity statistic unless identifying family shown",
        "is_full_point_law_test": False,
        "note": "labelled as sensitivity, not universally identifying unless proven",
    },
    "ClassifierTwoSampleTest-logistic": {
        "method": "classifier_two_sample_test",
        "method_variant": "logistic",
        "target_null": "H0^law: P0=P1 (relative to classifier class)",
        "validity_regime": "iid_metric_measure",
        "sampling_unit": "individual iid point",
        "kernel_or_distance": "held-out accuracy (logistic regression)",
        "is_characteristic": False,
        "requires": "stratified train/test split, fixed hyperparameters, held-out label permutation",
        "literature": "Lopez-Paz & Oquab 2017 (E1)",
        "consistency_conditions": "depends on classifier class; not universally consistent if misspecified",
        "is_full_point_law_test": False,
    },
    "ClassifierTwoSampleTest-rf": {
        "method": "classifier_two_sample_test",
        "method_variant": "rf",
        "target_null": "H0^law: P0=P1 (relative to classifier class)",
        "validity_regime": "iid_metric_measure",
        "sampling_unit": "individual iid point",
        "kernel_or_distance": "held-out accuracy (random forest)",
        "is_characteristic": False,
        "requires": "stratified train/test split, fixed hyperparameters, held-out label permutation",
        "literature": "Lopez-Paz & Oquab 2017 (E1)",
        "consistency_conditions": "depends on classifier class; not universally consistent if misspecified",
        "is_full_point_law_test": False,
    },
}


# ---------------------------------------------------------------------------
# 2.1 Required: PointMMD-Gaussian

def point_mmd_gaussian(
    cloud0,
    cloud1,
    *,
    regime: str = REGIME_I,
    bandwidth: Optional[float] = None,
    bandwidth_is_median_heuristic: bool = False,
    median_heuristic_seed: int = 0,
    kernel: str = "gaussian",
    n_perm: int = 199,
    exact: bool = False,
    max_exact_permutations: int = 100_000,
    seed: Optional[int] = 0,
) -> dict:
    """Point-level Gaussian MMD with pooled kernel and point-label permutation.

    One pooled Gram matrix is built with a bounded Gaussian kernel whose
    bandwidth is fixed before treatment labels are used.  The statistic is
    the usual biased V-statistic MMD^2

        mean(K[X,X]) + mean(K[Y,Y]) - 2 mean(K[X,Y])

    and calibration is by pooled point-label permutation preserving n0, n1.

    If ``bandwidth is None`` the pooled median heuristic is computed once
    from the unlabeled pooled points and recorded.  Do not estimate bandwidth
    separately by arm or select it using rejection outcomes.
    """
    _require_regime("PointMMD-Gaussian", regime)
    x0, x1 = _validate_cloud_pair(cloud0, cloud1)
    if kernel.lower() != "gaussian":
        raise ValueError("only gaussian kernel is supported for the primary comparator")
    pooled = np.vstack([x0, x1])
    n0 = len(x0)
    n = len(pooled)

    # bandwidth handling
    if bandwidth is None:
        bandwidth_val = _median_heuristic_bandwidth(pooled, seed=median_heuristic_seed)
        bandwidth_is_median_heuristic = True
    else:
        bandwidth_val = float(bandwidth)
        if bandwidth_val <= 0:
            raise ValueError("bandwidth must be positive")
    # record original request for audit
    bandwidth_requested = bandwidth

    def run():
        gram = _gaussian_gram(pooled, bandwidth_val)
        group0 = np.zeros(n, dtype=bool)
        group0[:n0] = True
        observed = _mmd2_from_gram(gram, group0)
        masks, exact_used = _label_masks(
            n, n0, n_perm=n_perm, exact=exact,
            max_exact_permutations=max_exact_permutations, seed=seed,
        )
        null = np.asarray([_mmd2_from_gram(gram, m) for m in masks], dtype=float)
        # p_value uses (1+count)/(1+n_perm) convention
        if exact_used:
            pval = float(np.mean(null >= observed))
        else:
            pval = p_value(float(observed), null, alternative="greater")
        return {
            "candidate": "PointMMD-Gaussian",
            "method": "point_mmd_gaussian",
            "regime": regime,
            "inferential_target": "H0^law: P0=P1",
            "target_null": "H0^law: P0=P1",
            "validity_regime": regime,
            "sampling_unit": "individual iid point",
            "statistic": float(observed),
            "pvalue": float(pval),
            "null_statistics": null,
            "n_permutations": int(len(null)),
            "exact_enumeration": bool(exact_used),
            "permutation_group": f"all point-label splits with n0={n0}, n1={n - n0}",
            "n0": int(len(x0)),
            "n1": int(len(x1)),
            "d": int(pooled.shape[1]),
            "m": np.nan,
            "K0": np.nan,
            "K1": np.nan,
            "effective_sample_size_total": int(n),
            "unused_points0": 0,
            "unused_points1": 0,
            "kernel_or_distance": "Gaussian",
            "bandwidth_or_tuning": float(bandwidth_val),
            "bandwidth": float(bandwidth_val),
            "bandwidth_requested": bandwidth_requested,
            "bandwidth_is_median_heuristic": bool(bandwidth_is_median_heuristic),
            "median_heuristic_seed": int(median_heuristic_seed) if bandwidth_is_median_heuristic else None,
            "kernel": "Gaussian",
            "diagnostics": {
                "exchangeability_basis": "iid point-level exchangeability",
                "kernel": "Gaussian",
                "bandwidth": float(bandwidth_val),
                "bandwidth_fixed_before_labels": True,
                "bandwidth_is_median_heuristic": bool(bandwidth_is_median_heuristic),
                "bandwidth_computed_from_pooled_unlabeled": bool(bandwidth_is_median_heuristic),
                "gram_cached": True,
                "gram_recomputed_in_permutation_loop": False,
                "kernel_characteristicness": "Gaussian kernel is characteristic on Euclidean space (Gretton et al. 2012; Simon-Gabriel & Scholkopf 2018)",
                "point_law_identification": "identifies P0=P1 under characteristic kernel moment/measurability conditions",
                "statistic_type": "biased V-statistic MMD^2 (includes diagonal)",
                "is_full_point_law_test": True,
            },
        }

    return _measure(run)


# ---------------------------------------------------------------------------
# EnergyDistance

def energy_distance_test(
    cloud0,
    cloud1,
    *,
    regime: str = REGIME_I,
    n_perm: int = 199,
    exact: bool = False,
    max_exact_permutations: int = 100_000,
    seed: Optional[int] = 0,
) -> dict:
    """Euclidean energy distance with pooled distance matrix and permutation.

    Statistic (biased V-statistic / distance MMD):

        E = 2*mean(D[X,Y]) - mean(D[X,X]) - mean(D[Y,Y])

    calibrated by pooled point-label permutation.  Distances are computed
    once and reused.  Documented as V-statistic; an unbiased U-statistic
    variant would exclude the diagonal.
    """
    _require_regime("EnergyDistance", regime)
    x0, x1 = _validate_cloud_pair(cloud0, cloud1)
    pooled = np.vstack([x0, x1])
    n0 = len(x0)
    n = len(pooled)

    def _energy_from_dist(D: np.ndarray, mask: np.ndarray) -> float:
        mask = np.asarray(mask, dtype=bool)
        i0 = np.flatnonzero(mask)
        i1 = np.flatnonzero(~mask)
        if len(i0) == 0 or len(i1) == 0:
            raise ValueError("both groups must be non-empty")
        # V-statistic includes diagonal zeros
        within0 = D[np.ix_(i0, i0)].mean() if len(i0) else 0.0
        within1 = D[np.ix_(i1, i1)].mean() if len(i1) else 0.0
        between = D[np.ix_(i0, i1)].mean() if len(i0) and len(i1) else 0.0
        return float(2.0 * between - within0 - within1)

    def run():
        D = _euclidean_distance_matrix(pooled)
        group0 = np.zeros(n, dtype=bool)
        group0[:n0] = True
        observed = _energy_from_dist(D, group0)
        masks, exact_used = _label_masks(
            n, n0, n_perm=n_perm, exact=exact,
            max_exact_permutations=max_exact_permutations, seed=seed,
        )
        null = np.asarray([_energy_from_dist(D, m) for m in masks], dtype=float)
        if exact_used:
            pval = float(np.mean(null >= observed))
        else:
            pval = p_value(float(observed), null, alternative="greater")
        return {
            "candidate": "EnergyDistance",
            "method": "energy_distance",
            "regime": regime,
            "inferential_target": "H0^law: P0=P1",
            "target_null": "H0^law: P0=P1",
            "validity_regime": regime,
            "sampling_unit": "individual iid point",
            "statistic": float(observed),
            "pvalue": float(pval),
            "null_statistics": null,
            "n_permutations": int(len(null)),
            "exact_enumeration": bool(exact_used),
            "permutation_group": f"all point-label splits with n0={n0}, n1={n - n0}",
            "n0": int(len(x0)),
            "n1": int(len(x1)),
            "d": int(pooled.shape[1]),
            "m": np.nan,
            "K0": np.nan,
            "K1": np.nan,
            "effective_sample_size_total": int(n),
            "unused_points0": 0,
            "unused_points1": 0,
            "kernel_or_distance": "Euclidean",
            "bandwidth_or_tuning": np.nan,
            "distance": "Euclidean",
            "diagnostics": {
                "exchangeability_basis": "iid point-level exchangeability",
                "distance": "Euclidean",
                "distance_matrix_cached": True,
                "distance_recomputed_in_permutation_loop": False,
                "statistic_type": "biased V-statistic energy distance (includes diagonal; 2*between - within0 - within1)",
                "unbiased_variant": "U-statistic would exclude diagonal; not used here",
                "moment_assumptions": "finite first moment required for identification (Szekely & Rizzo 2013)",
                "kernel_characteristicness": "energy distance is characteristic under stated moment conditions (equivalence to MMD, Sejdinovic et al. 2013)",
                "point_law_identification": "identifies P0=P1 under finite-moment conditions",
                "is_full_point_law_test": True,
            },
        }

    return _measure(run)


# ---------------------------------------------------------------------------
# Friedman-Rafsky MST

def friedman_rafsky_mst(
    cloud0,
    cloud1,
    *,
    regime: str = REGIME_I,
    n_perm: int = 199,
    exact: bool = False,
    max_exact_permutations: int = 100_000,
    seed: Optional[int] = 0,
) -> dict:
    """Friedman-Rafsky MST test: pooled MST, cross-edge count, label permutation.

    The Euclidean MST is constructed once on the pooled points.  The statistic
    is the number of cross-arm MST edges (equivalently runs).  Direction is
    fixed before looking at results: fewer cross edges = more evidence against
    H0, so p-value is left-tail (small cross count is extreme).  Ties are
    resolved deterministically by pooled point index via the distance matrix
    ordering and scipy's deterministic MST.

    Reports vertices, edges, components, and exact statistic definition.
    MST is a representation of pooled geometry, not an identifying kernel
    embedding; claims use graph-test literature.
    """
    _require_regime("FriedmanRafsky-MST", regime)
    x0, x1 = _validate_cloud_pair(cloud0, cloud1)
    pooled = np.vstack([x0, x1])
    n0 = len(x0)
    n = len(pooled)

    def run():
        D = _euclidean_distance_matrix(pooled)
        # Use a complete-graph Prim implementation so duplicate points remain
        # connected by zero-length edges rather than being treated as absent.
        rows, cols = _deterministic_mst_edges(D)
        n_edges = len(rows)
        n_vertices = n
        # Observed cross count
        group0 = np.zeros(n, dtype=bool)
        group0[:n0] = True
        # Count cross edges: endpoints have different labels
        # mst edges are directed but we count undirected
        def cross_count(mask: np.ndarray) -> int:
            mask = np.asarray(mask, dtype=bool)
            cnt = 0
            for u, v in zip(rows, cols):
                if mask[u] != mask[v]:
                    cnt += 1
            return int(cnt)

        observed = cross_count(group0)
        masks, exact_used = _label_masks(
            n, n0, n_perm=n_perm, exact=exact,
            max_exact_permutations=max_exact_permutations, seed=seed,
        )
        null = np.asarray([cross_count(m) for m in masks], dtype=float)
        # left-tail: small cross count is extreme
        if exact_used:
            pval = float(np.mean(null <= observed))
        else:
            pval = _permutation_pvalue_less(float(observed), null)

        # The complete finite Euclidean graph is connected, including ties.
        n_components = 1
        # For reporting, also compute expected cross edges under null hypergeometric
        # (not used for p-value, just diagnostic)

        return {
            "candidate": "FriedmanRafsky-MST",
            "method": "friedman_rafsky_mst",
            "regime": regime,
            "inferential_target": "H0^law: P0=P1",
            "target_null": "H0^law: P0=P1",
            "validity_regime": regime,
            "sampling_unit": "individual iid point",
            "statistic": float(observed),
            "statistic_raw": int(observed),
            "pvalue": float(pval),
            "null_statistics": null,
            "n_permutations": int(len(null)),
            "exact_enumeration": bool(exact_used),
            "permutation_group": f"all point-label splits with n0={n0}, n1={n - n0}",
            "n0": int(len(x0)),
            "n1": int(len(x1)),
            "d": int(pooled.shape[1]),
            "m": np.nan,
            "K0": np.nan,
            "K1": np.nan,
            "effective_sample_size_total": int(n),
            "unused_points0": 0,
            "unused_points1": 0,
            "kernel_or_distance": "Euclidean MST",
            "bandwidth_or_tuning": np.nan,
            "n_vertices": int(n_vertices),
            "n_edges": int(n_edges),
            "n_components": int(n_components),
            "statistic_definition": "number of cross-arm MST edges (small is extreme; left-tail)",
            "statistic_direction": "reject for small cross-edge count",
            "diagnostics": {
                "exchangeability_basis": "iid point-level exchangeability; MST fixed under permutation",
                "graph": "Euclidean MST on pooled points (complete graph, Euclidean distances)",
                "graph_cached": True,
                "graph_recomputed_in_permutation_loop": False,
                "n_vertices": int(n_vertices),
                "n_edges": int(n_edges),
                "n_components": int(n_components),
                "tie_resolution": "deterministic via pooled point index ordering; distance matrix order",
                "statistic_definition": "number of cross-arm MST edges",
                "statistic_direction": "small cross count is extreme (left-tail)",
                "kernel_characteristicness": "none; graph statistic, not characteristic kernel (Friedman & Rafsky 1979)",
                "is_full_point_law_test": True,
                "consistency_note": "valid permutation test under iid null; consistency per graph-test literature",
            },
        }

    return _measure(run)


# ---------------------------------------------------------------------------
# Schilling kNN

def schilling_knn(
    cloud0,
    cloud1,
    *,
    regime: str = REGIME_I,
    k: int = 1,
    directed: bool = True,
    n_perm: int = 199,
    exact: bool = False,
    max_exact_permutations: int = 100_000,
    seed: Optional[int] = 0,
) -> dict:
    """Schilling k-NN test: pooled k-NN graph, same-label edge count.

    The pooled k-nearest-neighbour graph is constructed once.  Primary choice
    k=1; sensitivity panel uses k in {1,5,10}.  Fixed choice of directed vs
    symmetrised before running.  Uses same-label (or equivalently cross-label)
    edge count as statistic and calibrates with pooled point-label permutations.
    Ties resolved deterministically via index ordering with stable sort.
    """
    _require_regime("Schilling-kNN", regime)
    if int(k) < 1:
        raise ValueError("k must be >= 1")
    k = int(k)
    x0, x1 = _validate_cloud_pair(cloud0, cloud1)
    pooled = np.vstack([x0, x1])
    n0 = len(x0)
    n = len(pooled)
    if k >= n:
        raise ValueError(f"k={k} must be < pooled n={n}")

    def run():
        from scipy.spatial.distance import cdist

        D = cdist(pooled, pooled, metric="euclidean")
        # For each point, find k nearest neighbours excluding itself, stable sort
        # Use mergesort for deterministic tie handling, then sort by distance then index
        # argsort with kind='mergesort' is stable
        neighbours = np.empty((n, k), dtype=int)
        for i in range(n):
            dists = D[i]
            # set self distance to inf so it is not selected
            dists_i = dists.copy()
            dists_i[i] = np.inf
            # stable argsort
            order = np.argsort(dists_i, kind="mergesort")
            neighbours[i] = order[:k]

        # Directed edge list: (i, neighbours[i,j])
        directed_edges = [(int(i), int(neighbours[i, j])) for i in range(n) for j in range(k)]
        n_directed = len(directed_edges)

        if directed:
            edge_list = directed_edges
            n_edges_report = n_directed
            edge_type = "directed"
        else:
            # symmetrise: undirected edge if either direction exists
            # Build set of undirected edges
            und = set()
            for u, v in directed_edges:
                a, b = (u, v) if u < v else (v, u)
                und.add((a, b))
            edge_list = list(und)
            n_edges_report = len(edge_list)
            edge_type = "undirected_symmetrised"

        def same_label_count(mask: np.ndarray) -> int:
            mask = np.asarray(mask, dtype=bool)
            cnt = 0
            for u, v in edge_list:
                if mask[u] == mask[v]:
                    cnt += 1
            return int(cnt)

        group0 = np.zeros(n, dtype=bool)
        group0[:n0] = True
        observed = same_label_count(group0)
        masks, exact_used = _label_masks(
            n, n0, n_perm=n_perm, exact=exact,
            max_exact_permutations=max_exact_permutations, seed=seed,
        )
        null = np.asarray([same_label_count(m) for m in masks], dtype=float)
        if exact_used:
            pval = float(np.mean(null >= observed))
        else:
            pval = p_value(float(observed), null, alternative="greater")

        return {
            "candidate": f"Schilling-kNN-k{k}",
            "method": "schilling_knn",
            "regime": regime,
            "inferential_target": "H0^law: P0=P1",
            "target_null": "H0^law: P0=P1",
            "validity_regime": regime,
            "sampling_unit": "individual iid point",
            "statistic": float(observed),
            "statistic_raw": int(observed),
            "pvalue": float(pval),
            "null_statistics": null,
            "n_permutations": int(len(null)),
            "exact_enumeration": bool(exact_used),
            "permutation_group": f"all point-label splits with n0={n0}, n1={n - n0}",
            "n0": int(len(x0)),
            "n1": int(len(x1)),
            "d": int(pooled.shape[1]),
            "m": np.nan,
            "K0": np.nan,
            "K1": np.nan,
            "effective_sample_size_total": int(n),
            "unused_points0": 0,
            "unused_points1": 0,
            "kernel_or_distance": f"k-NN (k={k}, {edge_type})",
            "bandwidth_or_tuning": float(k),
            "k": int(k),
            "directed": bool(directed),
            "n_edges": int(n_edges_report),
            "n_directed_edges": int(n_directed),
            "statistic_definition": "number of same-label directed (or symmetrised) k-NN edges (large is extreme)",
            "statistic_direction": "reject for large same-label count",
            "diagnostics": {
                "exchangeability_basis": "iid point-level exchangeability; k-NN graph fixed under permutation",
                "graph": f"pooled k-NN graph (k={k}, {edge_type}, Euclidean)",
                "graph_cached": True,
                "graph_recomputed_in_permutation_loop": False,
                "k": int(k),
                "directed": bool(directed),
                "n_directed_edges": int(n_directed),
                "n_edges": int(n_edges_report),
                "tie_resolution": "deterministic stable sort (mergesort) with index tie-break",
                "statistic_definition": "same-label edge count",
                "statistic_direction": "large is extreme (greater tail)",
                "kernel_characteristicness": "none; graph statistic, not characteristic kernel (Schilling 1986)",
                "is_full_point_law_test": True,
                "consistency_note": "valid permutation test; consistency per Schilling conditions",
            },
        }

    return _measure(run)


# ---------------------------------------------------------------------------
# Rosenbaum CrossMatch

def rosenbaum_crossmatch(
    cloud0,
    cloud1,
    *,
    regime: str = REGIME_I,
    n_perm: int = 199,
    exact: bool = False,
    max_exact_permutations: int = 100_000,
    seed: Optional[int] = 0,
    max_n_for_exact_matching: int = 500,
) -> dict:
    """Rosenbaum cross-match test: minimum-weight matching, cross-pair count.

    Constructs a minimum-weight non-bipartite matching of the pooled points
    (Euclidean distances) and counts cross-arm pairs.  Matching is computed
    once per replication and never recomputed inside the label loop.
    Uses networkx.max_weight_matching with negative distances (blossom algorithm).
    If pooled n is odd, one point is left unmatched (documented).

    Dependency: networkx.  Placed in optional benchmark extra; failure to
    import is recorded as a refusal rather than silent skip.
    """
    _require_regime("Rosenbaum-CrossMatch", regime)
    x0, x1 = _validate_cloud_pair(cloud0, cloud1)
    pooled = np.vstack([x0, x1])
    n0 = len(x0)
    n = len(pooled)

    def run():
        try:
            import networkx as nx
            from networkx.algorithms.matching import max_weight_matching
        except Exception as exc:
            raise RuntimeError(f"networkx required for CrossMatch but failed to import: {exc}")

        if n > max_n_for_exact_matching:
            raise ValueError(
                f"pooled n={n} exceeds CrossMatch limit {max_n_for_exact_matching}; refused for budget (matching O(n^3))"
            )
        from scipy.spatial.distance import cdist, pdist, squareform
        # Use cdist symmetric; for matching we need complete graph edge weights = -dist
        # Build edge list for max_weight_matching: list of (u, v, weight)
        # For n=500, edges ~125k, feasible.
        # Optimize: use pdist to get condensed, then iterate.
        D = cdist(pooled, pooled, metric="euclidean")
        iu, ju = np.triu_indices(n, k=1)
        weights = -D[iu, ju]
        edge_tuples = list(zip(iu.tolist(), ju.tolist(), weights.tolist()))
        G = nx.Graph()
        G.add_weighted_edges_from(edge_tuples)
        matching = max_weight_matching(G, maxcardinality=True, weight="weight")
        # matching is set of (u, v) tuples (unordered)
        matching_list = list(matching)
        n_pairs = len(matching_list)
        n_unmatched = n - 2 * n_pairs
        # Observed cross pairs
        group0 = np.zeros(n, dtype=bool)
        group0[:n0] = True

        def cross_pair_count(mask: np.ndarray) -> int:
            mask = np.asarray(mask, dtype=bool)
            cnt = 0
            for u, v in matching_list:
                if mask[u] != mask[v]:
                    cnt += 1
            return int(cnt)

        observed = cross_pair_count(group0)
        masks, exact_used = _label_masks(
            n, n0, n_perm=n_perm, exact=exact,
            max_exact_permutations=max_exact_permutations, seed=seed,
        )
        null = np.asarray([cross_pair_count(m) for m in masks], dtype=float)
        # Under alternative, cross pairs decrease (within matches increase) => left-tail
        if exact_used:
            pval = float(np.mean(null <= observed))
        else:
            pval = _permutation_pvalue_less(float(observed), null)

        return {
            "candidate": "Rosenbaum-CrossMatch",
            "method": "rosenbaum_crossmatch",
            "regime": regime,
            "inferential_target": "H0^law: P0=P1",
            "target_null": "H0^law: P0=P1",
            "validity_regime": regime,
            "sampling_unit": "individual iid point",
            "statistic": float(observed),
            "statistic_raw": int(observed),
            "pvalue": float(pval),
            "null_statistics": null,
            "n_permutations": int(len(null)),
            "exact_enumeration": bool(exact_used),
            "permutation_group": f"all point-label splits with n0={n0}, n1={n - n0}",
            "n0": int(len(x0)),
            "n1": int(len(x1)),
            "d": int(pooled.shape[1]),
            "m": np.nan,
            "K0": np.nan,
            "K1": np.nan,
            "effective_sample_size_total": int(n),
            "unused_points0": 0,
            "unused_points1": 0,
            "kernel_or_distance": "minimum-weight non-bipartite matching (Euclidean)",
            "bandwidth_or_tuning": np.nan,
            "n_pairs": int(n_pairs),
            "n_unmatched": int(n_unmatched),
            "matching_algorithm": "networkx.max_weight_matching (Edmonds blossom) on complete graph with weight=-Euclidean distance, maxcardinality=True",
            "odd_n_rule": "if pooled n odd, one point left unmatched (maximum cardinality matching)",
            "statistic_definition": "number of cross-arm matched pairs (small is extreme; left-tail)",
            "statistic_direction": "reject for small cross-pair count",
            "diagnostics": {
                "exchangeability_basis": "iid point-level exchangeability; matching fixed under permutation",
                "matching": "minimum-weight non-bipartite matching (blossom) on complete Euclidean graph",
                "matching_cached": True,
                "matching_recomputed_in_permutation_loop": False,
                "n_pairs": int(n_pairs),
                "n_unmatched": int(n_unmatched),
                "odd_n_rule": "one point unmatched if n odd (maximum cardinality)",
                "statistic_definition": "cross-pair count",
                "statistic_direction": "small is extreme (left-tail)",
                "dependency": "networkx",
                "is_full_point_law_test": True,
                "consistency_note": "valid permutation test; consistency per Rosenbaum 2005",
            },
        }

    return _measure(run)


# ---------------------------------------------------------------------------
# Sliced Wasserstein

def sliced_wasserstein_test(
    cloud0,
    cloud1,
    *,
    regime: str = REGIME_I,
    n_projections: int = 100,
    projection_seed: int = 0,
    p: int = 1,
    n_perm: int = 199,
    exact: bool = False,
    max_exact_permutations: int = 100_000,
    seed: Optional[int] = 0,
) -> dict:
    """Sliced Wasserstein two-sample test: fixed projections, pooled permutation.

    Computational version: fixed-regularisation sliced-Wasserstein sensitivity.
    Regularisation = none (exact 1D Wasserstein); number of projections and
    projection seed are fixed before labels and recorded in every result.
    Uses Euclidean projections onto random unit directions (Gaussian then
    normalised), then exact 1D Wasserstein-1 (via scipy.stats.wasserstein_distance)
    per projection and averages.

    This is geometrically interpretable but finite projection family is a
    sensitivity statistic unless identifying family is proven; labelled honestly.
    """
    _require_regime("SlicedWasserstein", regime)
    if int(n_projections) < 1:
        raise ValueError("n_projections must be >= 1")
    if int(p) != 1:
        raise ValueError("only p=1 is implemented by scipy.stats.wasserstein_distance")
    p = 1
    x0, x1 = _validate_cloud_pair(cloud0, cloud1)
    pooled = np.vstack([x0, x1])
    n0 = len(x0)
    n = len(pooled)
    d = pooled.shape[1]
    n_projections = int(n_projections)

    def run():
        from scipy.stats import wasserstein_distance

        rng = np.random.default_rng(projection_seed)
        dirs = rng.normal(size=(n_projections, d))
        # normalise
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        dirs = dirs / norms

        # pooled projections: shape (n_projections, n)
        # pooled @ dirs.T -> (n, n_projections) then transpose
        proj_pooled = pooled @ dirs.T  # (n, n_projections)
        proj_pooled = proj_pooled.T  # (n_projections, n)

        def sliced_stat(mask: np.ndarray) -> float:
            mask = np.asarray(mask, dtype=bool)
            total = 0.0
            for pr in range(n_projections):
                vals = proj_pooled[pr]
                a = vals[mask]
                b = vals[~mask]
                # wasserstein_distance handles unequal sizes via sorting + interpolation
                total += wasserstein_distance(a, b)
            return float(total / n_projections)

        group0 = np.zeros(n, dtype=bool)
        group0[:n0] = True
        observed = sliced_stat(group0)
        masks, exact_used = _label_masks(
            n, n0, n_perm=n_perm, exact=exact,
            max_exact_permutations=max_exact_permutations, seed=seed,
        )
        null = np.asarray([sliced_stat(m) for m in masks], dtype=float)
        if exact_used:
            pval = float(np.mean(null >= observed))
        else:
            pval = p_value(float(observed), null, alternative="greater")

        return {
            "candidate": "SlicedWasserstein",
            "method": "sliced_wasserstein",
            "regime": regime,
            "inferential_target": "H0^law: P0=P1 (sensitivity; finite projections)",
            "target_null": "H0^law: P0=P1 (sensitivity)",
            "validity_regime": regime,
            "sampling_unit": "individual iid point",
            "statistic": float(observed),
            "pvalue": float(pval),
            "null_statistics": null,
            "n_permutations": int(len(null)),
            "exact_enumeration": bool(exact_used),
            "permutation_group": f"all point-label splits with n0={n0}, n1={n - n0}",
            "n0": int(len(x0)),
            "n1": int(len(x1)),
            "d": int(d),
            "m": np.nan,
            "K0": np.nan,
            "K1": np.nan,
            "effective_sample_size_total": int(n),
            "unused_points0": 0,
            "unused_points1": 0,
            "kernel_or_distance": f"sliced Wasserstein-1 (n_projections={n_projections})",
            "bandwidth_or_tuning": float(n_projections),
            "n_projections": int(n_projections),
            "projection_seed": int(projection_seed),
            "p": int(p),
            "transport_solver": "exact 1D Wasserstein via sorting (scipy.stats.wasserstein_distance)",
            "regularisation": "none (exact 1D)",
            "diagnostics": {
                "exchangeability_basis": "iid point-level exchangeability; projections fixed before labels",
                "distance": f"sliced Wasserstein-{p} average over {n_projections} random projections",
                "n_projections": int(n_projections),
                "projection_seed": int(projection_seed),
                "projection_distribution": "Gaussian then normalised to unit sphere (isotropic)",
                "transport_solver": "exact 1D Wasserstein via sorting (scipy)",
                "regularisation": "none",
                "is_identifying": False,
                "is_full_point_law_test": False,
                "honest_label": "finite-projection sensitivity statistic; not promoted to universally identifying unless proven",
                "reference": "Ramdas et al. 2015 (D1)",
            },
        }

    return _measure(run)


# ---------------------------------------------------------------------------
# Classifier two-sample test

def classifier_two_sample_test(
    cloud0,
    cloud1,
    *,
    regime: str = REGIME_I,
    classifier: str = "logistic",
    test_fraction: float = 0.5,
    split_seed: Optional[int] = 0,
    n_perm: int = 199,
    exact: bool = False,
    max_exact_permutations: int = 100_000,
    seed: Optional[int] = 0,
) -> dict:
    """Sample-split classifier two-sample test with held-out label permutation.

    Fixed model family: logistic regression or random forest.  Training/test
    split is stratified and independent of treatment labels via predeclared
    seed.  Hyperparameters are fixed.  Simplest valid implementation:
    fit on training subset and evaluate held-out accuracy; calibrate via
    held-out-label permutation conditional on fixed training fit (no retraining
    inside permutation loop).  Reports classifier class, split fraction,
    seeds, training time, and tuning info.

    Targets point-law equality only relative to classifier class and protocol;
    not automatically universally consistent.
    """
    _require_regime("ClassifierTwoSampleTest", regime)
    classifier = str(classifier).lower()
    if classifier not in ("logistic", "rf", "random_forest", "logreg"):
        raise ValueError("classifier must be 'logistic' or 'rf'")
    # normalise aliases
    if classifier in ("rf", "random_forest"):
        clf_name = "rf"
        clf_label = "ClassifierTwoSampleTest-rf"
    else:
        clf_name = "logistic"
        clf_label = "ClassifierTwoSampleTest-logistic"
    if not 0.0 < float(test_fraction) < 1.0:
        raise ValueError("test_fraction must be in (0,1)")
    test_fraction = float(test_fraction)
    x0, x1 = _validate_cloud_pair(cloud0, cloud1)
    pooled = np.vstack([x0, x1])
    n0 = len(x0)
    n1 = len(x1)
    n = len(pooled)
    y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])

    def run():
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        # stratified split preserving label proportions
        # train_test_split expects X, y
        split_seed_int = int(split_seed) if split_seed is not None else 0
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                pooled, y, test_size=test_fraction, random_state=split_seed_int, stratify=y
            )
        except ValueError as exc:
            raise ValueError(f"stratified split failed (n0={n0}, n1={n1}, test_fraction={test_fraction}): {exc}")

        n_train = len(X_train)
        n_test = len(X_test)
        # Count per group in test
        n0_test = int(np.sum(y_test == 0))
        n1_test = int(np.sum(y_test == 1))
        if n0_test == 0 or n1_test == 0:
            raise ValueError(f"test split has empty class: n0_test={n0_test}, n1_test={n1_test}")

        t_train_start = time.perf_counter()
        if clf_name == "logistic":
            clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=split_seed_int)
            clf_kwargs = {"max_iter": 1000, "solver": "lbfgs"}
        else:
            clf = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=split_seed_int)
            clf_kwargs = {"n_estimators": 100, "n_jobs": 1}
        clf.fit(X_train, y_train)
        train_time = time.perf_counter() - t_train_start

        # predictions on test
        y_pred = clf.predict(X_test)
        observed_acc = float(np.mean(y_pred == y_test))

        # held-out label permutation: permute y_test labels preserving counts
        # We have n_test positions, n0_test zeros. Generate masks for test permutation.
        masks, exact_used = _label_masks(
            n_test, n0_test, n_perm=n_perm, exact=exact,
            max_exact_permutations=max_exact_permutations, seed=seed,
        )
        # masks are bool arrays length n_test where True = group0 (label 0)
        # For each mask, create permuted y_test: 0 where mask True, 1 where False
        # Then compute accuracy against fixed y_pred
        null = np.empty(len(masks), dtype=float)
        for idx, mask in enumerate(masks):
            y_test_perm = np.where(mask, 0, 1)
            null[idx] = float(np.mean(y_pred == y_test_perm))

        if exact_used:
            pval = float(np.mean(null >= observed_acc))
        else:
            pval = p_value(float(observed_acc), null, alternative="greater")

        return {
            "candidate": clf_label,
            "method": "classifier_two_sample_test",
            "method_variant": clf_name,
            "regime": regime,
            "inferential_target": "H0^law: P0=P1 (relative to classifier class)",
            "target_null": "H0^law: P0=P1 (classifier-relative)",
            "validity_regime": regime,
            "sampling_unit": "individual iid point",
            "statistic": float(observed_acc),
            "statistic_raw": float(observed_acc),
            "pvalue": float(pval),
            "null_statistics": null,
            "n_permutations": int(len(null)),
            "exact_enumeration": bool(exact_used),
            "permutation_group": f"all held-out label splits with n0_test={n0_test}, n1_test={n1_test} (conditional on training fit)",
            "calibration": "held-out label permutation conditional on fixed training fit (no retraining inside loop)",
            "alternative_calibration": "retrain under permuted training labels would be exact but is not used here (computational cost)",
            "n0": int(n0),
            "n1": int(n1),
            "n": int(n),
            "d": int(pooled.shape[1]),
            "m": np.nan,
            "K0": np.nan,
            "K1": np.nan,
            "effective_sample_size_total": int(n),
            "unused_points0": 0,
            "unused_points1": 0,
            "kernel_or_distance": f"held-out accuracy ({clf_name})",
            "bandwidth_or_tuning": np.nan,
            "classifier": clf_name,
            "classifier_class": "LogisticRegression" if clf_name == "logistic" else "RandomForestClassifier",
            "classifier_params": clf_kwargs,
            "split_fraction": float(test_fraction),
            "split_seed": int(split_seed_int),
            "permutation_seed": int(seed) if seed is not None else None,
            "n_train": int(n_train),
            "n_test": int(n_test),
            "n0_test": int(n0_test),
            "n1_test": int(n1_test),
            "train_time_seconds": float(train_time),
            "hyperparameter_tuning": "none; fixed hyperparameters, selected only on training data if any",
            "diagnostics": {
                "exchangeability_basis": "iid point-level exchangeability; training fit fixed under held-out permutation",
                "classifier": clf_name,
                "classifier_class": "LogisticRegression" if clf_name == "logistic" else "RandomForestClassifier",
                "classifier_params": clf_kwargs,
                "split_fraction": float(test_fraction),
                "split_seed": int(split_seed_int),
                "stratified": True,
                "calibration": "held-out label permutation conditional on fixed training fit",
                "calibration_alternative": "retrain under permuted training labels (exact, not used; cost)",
                "n_train": int(n_train),
                "n_test": int(n_test),
                "n0_test": int(n0_test),
                "n1_test": int(n1_test),
                "train_time_seconds": float(train_time),
                "statistic_definition": "held-out classification accuracy (large is extreme)",
                "statistic_direction": "reject for large accuracy",
                "hyperparameter_tuning": "none; fixed before labels",
                "is_full_point_law_test": False,
                "limitation": "targets point-law equality only relative to classifier class; not universally consistent if misspecified (Lopez-Paz & Oquab 2017)",
            },
        }

    return _measure(run)
