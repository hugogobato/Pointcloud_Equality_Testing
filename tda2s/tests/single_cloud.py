"""Prototype tests for the exactly-two-cloud Regime-I problem.

This module implements the applicable Phase 5B candidates for the scoped
Regime-I benchmark:

``SC-A``
    pooled-point label permutation.  The persistence diagrams are recomputed
    for every label split, so this is a full point-law exchangeability
    baseline, not a barcode-law test.

``SC-B``
    a fixed disjoint partition into subclouds of size ``m`` followed by a
    diagram-level MMD permutation test.  The only barcode replicates are the
    disjoint blocks, and the returned ``K0`` and ``K1`` fields make that
    effective sample size explicit.

``SC-C``
    a finite-vector persistent-Betti contrast calibrated by a pooled
    point-level bootstrap.  The smoothed version follows the
    Roycraft--Krebs--Polonik bootstrap idea by resampling points and adding
    Gaussian kernel noise.  The ordinary bootstrap is exposed as a negative
    control.  This candidate is deliberately labelled as a finite-vector
    mean test and must not be reported as a test of the full fixed-``m``
    barcode law.

The common return object is a dictionary rather than a dataclass so that it
can be serialized by the Phase 5C fleet without a custom encoder.  Arrays are
kept in the object for diagnostics and reproducibility; callers that need
JSON should convert them explicitly.

The implementation is intentionally conservative about routing.  All three
methods accept only ``iid_metric_measure``.  A spatial process, a fixed cloud
without a sampling model, or a Bayesian generative model requires a new
observation-model lock and raises ``ValueError`` here.

The production entry point is ``sc_b_production_test``.  It freezes the
Phase-5 target at ``m=25``, the VR filtration, degrees ``(0, 1)``, and the
pre-registered kernel bandwidth.  The lower-level ``sc_b_disjoint_mmd``
function remains available for Phase-5 sensitivity work, including unlocked
values of ``m``.
"""
from __future__ import annotations

import itertools
import math
import resource
import time
import tracemalloc
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from tda2s.ph import compute_diagrams
from tda2s.resample import p_value

REGIME_I = "iid_metric_measure"
LOCKED_M = 25
LOCKED_HOMOLOGY_DIMS = (0, 1)
DEFAULT_KERNEL_BANDWIDTH = 0.10
DEFAULT_GRID = np.linspace(0.0, 1.0, 9)
#: Matches tda2s.ph.PhParams defaults so filtration-level options are never
#: silently replaced by a different default once the user selects them.
DEFAULT_GRID_SIZE = 64
DEFAULT_DTM_K = 20
PRODUCTION_MIN_BLOCKS = 5
PRODUCTION_API_VERSION = "sc-b-v1"
DEFAULT_RAW_POINT_KERNEL = "gaussian"
DEFAULT_RAW_POINT_BANDWIDTH = 0.10
DEFAULT_RAW_BAG_BANDWIDTH = 0.25
RAW_KERNEL_VARIANTS = ("gaussian_mean_embedding", "mean_pairwise")
HYBRID_ALPHA_GRID = (0.25, 0.50, 0.75, 1.00)

__all__ = [
    "REGIME_I",
    "LOCKED_M",
    "PRODUCTION_MIN_BLOCKS",
    "disjoint_partition",
    "persistent_betti_vector",
    "roycraft_reference_setting",
    "run_single_cloud_test",
    "sc_a_label_permutation",
    "sc_b_disjoint_mmd",
    "sc_b_production_test",
    "sc_b_repeated_partition_test",
    "raw_block_mmd",
    "hybrid_block_mmd",
    "sc_a_blockwise_label_permutation",
    "raw_block_repeated_partition_test",
    "sc_c_finite_vector",
    "sc_c_naive_bootstrap",
]


# ---------------------------------------------------------------------------
# Shared validation, timing, and permutation helpers


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
            f"{candidate} is only valid for declared regime {REGIME_I!r}; "
            f"received {regime!r}. Reclassify the observation model before inference."
        )


def _normalise_candidate(candidate: str) -> str:
    key = str(candidate).lower().replace("_", "-")
    aliases = {
        "a": "sc-a",
        "sc-a": "sc-a",
        "pooled-label-permutation": "sc-a",
        "b": "sc-b",
        "sc-b": "sc-b",
        "disjoint-mmd": "sc-b",
        "sc-b-production": "sc-b-production",
        "production-sc-b": "sc-b-production",
        "raw-block-mmd": "raw-block-mmd",
        "rawblockmmd": "raw-block-mmd",
        "hybrid-block-mmd": "hybrid-block-mmd",
        "hybridblockmmd": "hybrid-block-mmd",
        "sc-a-block": "sc-a-block",
        "sc-a-blockwise": "sc-a-block",
        "c": "sc-c",
        "sc-c": "sc-c",
        "finite-vector": "sc-c",
    }
    if key not in aliases:
        raise ValueError(
            "candidate must be one of {'SC-A', 'SC-B', 'SC-B-production', "
            "'RawBlockMMD', 'HybridBlockMMD', 'SC-A-Block', 'SC-C'}"
        )
    return aliases[key]


def _finish(result: dict, started: float, peak_bytes: int) -> dict:
    result["runtime_seconds"] = float(time.perf_counter() - started)
    result["peak_memory_bytes"] = int(peak_bytes)
    # This makes the memory measurement interpretable when numpy or GUDHI
    # allocates outside tracemalloc's Python allocator.
    result["peak_memory_measurement"] = "tracemalloc_python_allocations"
    return result


def _current_rss_bytes() -> int:
    """Return the process high-water RSS in bytes on Unix-like systems."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Linux and macOS expose ru_maxrss in KiB and bytes respectively.  The
    # repository's supported execution environments are Linux, but retaining
    # the branch keeps the helper interpretable on macOS.
    scale = 1024 if __import__("sys").platform.startswith("linux") else 1
    return int(usage.ru_maxrss * scale)


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


def _validate_ph_options(filtration: str, homology_dims: Sequence[int]) -> Tuple[int, ...]:
    dims = tuple(int(d) for d in homology_dims)
    if not dims or any(d < 0 for d in dims) or len(set(dims)) != len(dims):
        raise ValueError("homology_dims must be a non-empty sequence of distinct non-negative integers")
    if filtration not in {"vr", "ripser", "alpha", "cech", "cubical", "dtm-rips"}:
        raise ValueError(f"unknown filtration {filtration!r}")
    return dims


def _enumerated_masks(n: int, n0: int) -> Iterable[np.ndarray]:
    for indices in itertools.combinations(range(n), n0):
        mask = np.zeros(n, dtype=bool)
        mask[list(indices)] = True
        yield mask


def _label_masks(n: int, n0: int, *, n_perm: int, exact: bool,
                 max_exact_permutations: int, seed: Optional[int]) -> Tuple[list, bool]:
    if n0 <= 0 or n0 >= n:
        raise ValueError("both label groups must contain at least one observation")
    n_splits = math.comb(n, n0)
    if exact:
        if n_splits > max_exact_permutations:
            raise ValueError(
                f"exact enumeration needs {n_splits} splits, above the limit "
                f"max_exact_permutations={max_exact_permutations}")
        return list(_enumerated_masks(n, n0)), True
    if n_perm < 1:
        raise ValueError("n_perm must be >= 1")
    rng = np.random.default_rng(seed)
    masks = []
    for _ in range(int(n_perm)):
        mask = np.zeros(n, dtype=bool)
        mask[rng.permutation(n)[:n0]] = True
        masks.append(mask)
    return masks, False


def _permutation_pvalue(observed: float, null: np.ndarray, exact: bool) -> float:
    null = np.asarray(null, dtype=float)
    if null.size == 0:
        return 1.0
    if exact:
        return float(np.mean(null >= observed))
    return p_value(float(observed), null, alternative="greater")


def _ph(points: np.ndarray, *, filtration: str, homology_dims: Sequence[int],
        max_edge_length: Optional[float], grid_size: int, dtm_k: int,
        cache_dir: Optional[str]):
    return compute_diagrams(
        points,
        filtration=filtration,
        homology_dims=homology_dims,
        max_edge_length=max_edge_length,
        grid_size=grid_size,
        dtm_k=dtm_k,
        cache_dir=cache_dir,
    )


# ---------------------------------------------------------------------------
# Fixed persistence scale-space kernel and diagram-level MMD


def _pss_kernel(diagram0: np.ndarray, diagram1: np.ndarray,
                bandwidth: float) -> float:
    """Reininghaus persistence scale-space kernel on one degree.

    ``bandwidth`` is the locked raw metric bandwidth.  The reflected diagram
    term makes the kernel vanish on the diagonal of the persistence half
    plane, as in the source construction.  The joint kernel below combines
    the degree-specific universal kernels through a tensor product.
    """
    if bandwidth <= 0:
        raise ValueError("kernel bandwidth must be positive")
    f = np.asarray(diagram0, dtype=float).reshape(-1, 2)
    g = np.asarray(diagram1, dtype=float).reshape(-1, 2)
    if len(f) == 0 or len(g) == 0:
        return 0.0
    g_reflected = g[:, ::-1]
    d2 = ((f[:, None, :] - g[None, :, :]) ** 2).sum(axis=2)
    d2_reflected = ((f[:, None, :] - g_reflected[None, :, :]) ** 2).sum(axis=2)
    return float(
        (np.exp(-d2 / (8.0 * bandwidth))
         - np.exp(-d2_reflected / (8.0 * bandwidth))).sum()
        / (8.0 * np.pi * bandwidth)
    )


def _universal_diagram_kernel(diagrams0, diagrams1, bandwidth: float) -> float:
    """Characteristic joint kernel built from characteristic degree kernels.

    Kwitt et al.'s exponentiated persistence scale-space kernel is universal,
    and therefore characteristic, on the bounded diagram classes covered by
    their Proposition 2.  The locked object is a *joint* degree-0/degree-1
    diagram, so the correct product-space kernel is the tensor-product kernel
    ``prod_d k_d``.  Averaging the ``k_d`` values would retain only the two
    marginal diagram laws and would not identify their cross-degree dependence.
    """
    if len(diagrams0) != len(diagrams1):
        raise ValueError("joint diagram pairs must have the same number of degrees")
    if not diagrams0:
        raise ValueError("at least one homology degree is required")
    values = [math.exp(_pss_kernel(a, b, bandwidth))
              for a, b in zip(diagrams0, diagrams1)]
    # The tensor product is characteristic on a product of the bounded
    # per-degree diagram classes when every factor is characteristic.  It is
    # also positive definite, so the resulting MMD remains a valid RKHS
    # discrepancy.  Do not replace this with a sum or mean: those kernels can
    # be blind to changes in the dependence between homology degrees.
    return float(np.prod(values))


def _diagram_gram(diagrams: Sequence[Sequence[np.ndarray]], bandwidth: float) -> np.ndarray:
    n = len(diagrams)
    if n == 0:
        raise ValueError("at least one diagram is required")
    gram = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            value = _universal_diagram_kernel(diagrams[i], diagrams[j], bandwidth)
            gram[i, j] = gram[j, i] = value
    return gram


def _mmd2_from_gram(gram: np.ndarray, group0: np.ndarray) -> float:
    gram = np.asarray(gram, dtype=float)
    group0 = np.asarray(group0, dtype=bool)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or gram.shape[0] != len(group0):
        raise ValueError("gram and group0 have incompatible shapes")
    i0, i1 = np.flatnonzero(group0), np.flatnonzero(~group0)
    if len(i0) == 0 or len(i1) == 0:
        raise ValueError("both MMD groups must be non-empty")
    within0 = gram[np.ix_(i0, i0)].mean()
    within1 = gram[np.ix_(i1, i1)].mean()
    between = gram[np.ix_(i0, i1)].mean()
    return float(max(within0 + within1 - 2.0 * between, 0.0))


def _joint_discrepancy(diagrams0, diagrams1, bandwidth: float) -> float:
    """Squared RKHS distance between two joint degree-tagged diagrams."""
    value = (_universal_diagram_kernel(diagrams0, diagrams0, bandwidth)
             + _universal_diagram_kernel(diagrams1, diagrams1, bandwidth)
             - 2.0 * _universal_diagram_kernel(diagrams0, diagrams1, bandwidth))
    return float(max(value, 0.0))


# ---------------------------------------------------------------------------
# Raw and hybrid fixed-size block methods


def _validate_raw_kernel_options(point_kernel: str, raw_kernel: str,
                                 point_bandwidth: float,
                                 bag_bandwidth: float) -> tuple[str, str]:
    point_kernel = str(point_kernel).lower()
    raw_kernel = str(raw_kernel).lower()
    if point_kernel not in {"gaussian", "laplacian"}:
        raise ValueError("point_kernel must be 'gaussian' or 'laplacian'")
    if raw_kernel not in RAW_KERNEL_VARIANTS:
        raise ValueError(
            "raw_kernel must be one of {'gaussian_mean_embedding', 'mean_pairwise'}"
        )
    if float(point_bandwidth) <= 0 or float(bag_bandwidth) <= 0:
        raise ValueError("point and bag kernel bandwidths must be positive")
    return point_kernel, raw_kernel


def _point_kernel_matrix(points0: np.ndarray, points1: np.ndarray,
                         point_kernel: str, bandwidth: float) -> np.ndarray:
    """Bounded characteristic point kernel on Euclidean point coordinates."""
    delta = points0[:, None, :] - points1[None, :, :]
    if point_kernel == "gaussian":
        squared = np.sum(delta * delta, axis=2)
        return np.exp(-squared / (2.0 * float(bandwidth) ** 2))
    distances = np.abs(delta).sum(axis=2)
    return np.exp(-distances / float(bandwidth))


def _raw_block_features(blocks: Sequence[np.ndarray], *, point_kernel: str,
                        point_bandwidth: float) -> list[dict]:
    """Cache the raw feature needed by every block-kernel evaluation.

    The cached self inner product and the original points are sufficient to
    evaluate the whole raw Gram matrix before the permutation loop.  Point
    order does not appear in the feature definition.
    """
    features = []
    for block in blocks:
        points = np.asarray(block, dtype=float)
        self_mean = float(np.mean(_point_kernel_matrix(
            points, points, point_kernel, point_bandwidth)))
        features.append({"points": points, "self_mean": self_mean})
    return features


def _raw_block_kernel(feature0: dict, feature1: dict, *, point_kernel: str,
                      point_bandwidth: float, bag_bandwidth: float,
                      raw_kernel: str) -> float:
    cross_mean = float(np.mean(_point_kernel_matrix(
        feature0["points"], feature1["points"], point_kernel,
        point_bandwidth)))
    if raw_kernel == "mean_pairwise":
        return cross_mean
    distance2 = max(
        feature0["self_mean"] + feature1["self_mean"] - 2.0 * cross_mean,
        0.0,
    )
    return float(np.exp(-distance2 / (2.0 * float(bag_bandwidth) ** 2)))


def _raw_block_gram(features: Sequence[dict], *, point_kernel: str,
                    point_bandwidth: float, bag_bandwidth: float,
                    raw_kernel: str) -> np.ndarray:
    n = len(features)
    if n == 0:
        raise ValueError("at least one raw block is required")
    gram = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            value = _raw_block_kernel(
                features[i], features[j], point_kernel=point_kernel,
                point_bandwidth=point_bandwidth, bag_bandwidth=bag_bandwidth,
                raw_kernel=raw_kernel,
            )
            gram[i, j] = gram[j, i] = value
    return gram


def _block_target(m: int, *, barcode: bool = False,
                  raw_kernel: Optional[str] = None) -> str:
    if barcode:
        return (f"H0,{m}^bar: Phi^{m}_0:1(P0) = Phi^{m}_0:1(P1) "
                "(alpha=0 barcode-only diagnostic)")
    if raw_kernel == "mean_pairwise":
        return ("H0^raw-block-sensitivity: point-law-sensitive under the iid "
                f"product-block model, m={m}; not fully identified for "
                "unrestricted bag laws")
    return ("H0^law: P0 = P1 "
            f"(raw characteristic fixed-size block representation, m={m}, "
            "under iid point sampling)")


def _block_result_from_gram(
    *, candidate: str, method: str, target: str, regime: str,
    x0: np.ndarray, x1: np.ndarray, blocks0: Sequence[np.ndarray],
    blocks1: Sequence[np.ndarray], indices0: np.ndarray, indices1: np.ndarray,
    remainder0: np.ndarray, remainder1: np.ndarray, gram: np.ndarray,
    n_perm: int, exact: bool, max_exact_permutations: int,
    seed: Optional[int], diagnostics: dict,
) -> dict:
    K0, K1 = len(blocks0), len(blocks1)
    group0 = np.zeros(K0 + K1, dtype=bool)
    group0[:K0] = True
    observed = _mmd2_from_gram(gram, group0)
    masks, exact_used = _label_masks(
        K0 + K1, K0, n_perm=n_perm, exact=exact,
        max_exact_permutations=max_exact_permutations, seed=seed,
    )
    # All features are already represented in gram.  This loop performs only
    # label reassignment and MMD arithmetic, never point or PH recomputation.
    null = np.asarray([_mmd2_from_gram(gram, mask) for mask in masks], dtype=float)
    diagnostics = dict(diagnostics)
    diagnostics.update({
        "sampling_unit": "frozen disjoint m-point block",
        "point_sampling_unit": "individual iid point",
        "effective_sample_size": {
            "K0": int(K0), "K1": int(K1), "total": int(K0 + K1),
        },
        "unused_point_counts": {
            "arm0": int(len(remainder0)), "arm1": int(len(remainder1)),
        },
        "overlapping_blocks_used": False,
        "partition_frozen_for_call": True,
        "persistent_homology_recomputed_in_permutation_loop": False,
        "raw_features_recomputed_in_permutation_loop": False,
        "permutation_group": f"all block-label splits with K0={K0}, K1={K1}",
    })
    return {
        "candidate": candidate,
        "method": method,
        "regime": regime,
        "inferential_target": target,
        "statistic": float(observed),
        "pvalue": _permutation_pvalue(observed, null, exact_used),
        "posterior_quantity": None,
        "null_statistics": null,
        "n_permutations": int(len(null)),
        "exact_enumeration": bool(exact_used),
        "m": int(len(blocks0[0])),
        "sampling_unit": diagnostics["sampling_unit"],
        "kernel": diagnostics.get("kernel", ""),
        "bandwidth": diagnostics.get("bandwidth", diagnostics.get("bag_kernel_bandwidth")),
        "K0": int(K0),
        "K1": int(K1),
        "K_a": [int(K0), int(K1)],
        "n0": int(len(x0)),
        "n1": int(len(x1)),
        "remainder0": int(len(remainder0)),
        "remainder1": int(len(remainder1)),
        "unused_point_counts": [int(len(remainder0)), int(len(remainder1))],
        "block_indices0": indices0,
        "block_indices1": indices1,
        "diagnostics": diagnostics,
    }


def _validate_block_method_controls(partition_is_data_independent: bool,
                                    feature_tuning_is_label_independent: bool):
    if not partition_is_data_independent:
        raise ValueError(
            "block methods require a data-independent partition fixed without "
            "point coordinates"
        )
    if not feature_tuning_is_label_independent:
        raise ValueError(
            "label-dependent feature tuning is unsupported; freeze kernels and "
            "bandwidths before seeing treatment labels"
        )


def raw_block_mmd(
    cloud0, cloud1, *, regime: str = REGIME_I, m: int = LOCKED_M,
    partition_seed: Optional[int] = 0, partition0=None, partition1=None,
    point_kernel: str = DEFAULT_RAW_POINT_KERNEL,
    point_kernel_bandwidth: float = DEFAULT_RAW_POINT_BANDWIDTH,
    bag_kernel_bandwidth: float = DEFAULT_RAW_BAG_BANDWIDTH,
    raw_kernel: str = "gaussian_mean_embedding", n_perm: int = 999,
    exact: bool = False, max_exact_permutations: int = 100_000,
    seed: Optional[int] = 0, cache_dir: Optional[str] = None,
    partition_is_data_independent: bool = True,
    feature_tuning_is_label_independent: bool = True,
) -> dict:
    """Raw fixed-block MMD with a characteristic unordered-bag kernel.

    The default kernel is Gaussian on the point-kernel mean embedding of each
    bag.  ``raw_kernel='mean_pairwise'`` is available as a deliberately
    weaker diagnostic and is labelled non-characteristic for unrestricted bag
    laws in the returned diagnostics.
    """
    _require_regime("RawBlockMMD", regime)
    _validate_block_method_controls(
        partition_is_data_independent, feature_tuning_is_label_independent)
    x0, x1 = _validate_cloud_pair(cloud0, cloud1)
    if int(m) < 1:
        raise ValueError("m must be >= 1")
    m = int(m)
    point_kernel, raw_kernel = _validate_raw_kernel_options(
        point_kernel, raw_kernel, point_kernel_bandwidth, bag_kernel_bandwidth)

    def run():
        blocks0, indices0, remainder0 = _partition_or_draw(
            x0, m, partition_seed, partition0, "partition0")
        blocks1, indices1, remainder1 = _partition_or_draw(
            x1, m, None if partition_seed is None else int(partition_seed) + 1,
            partition1, "partition1")
        features = _raw_block_features(
            list(blocks0) + list(blocks1), point_kernel=point_kernel,
            point_bandwidth=point_kernel_bandwidth)
        gram = _raw_block_gram(
            features, point_kernel=point_kernel,
            point_bandwidth=point_kernel_bandwidth,
            bag_bandwidth=bag_kernel_bandwidth, raw_kernel=raw_kernel)
        return _block_result_from_gram(
            candidate="RawBlockMMD", method="raw_disjoint_block_mmd",
            target=_block_target(m, raw_kernel=raw_kernel), regime=regime, x0=x0, x1=x1,
            blocks0=blocks0, blocks1=blocks1, indices0=indices0,
            indices1=indices1, remainder0=remainder0, remainder1=remainder1,
            gram=gram, n_perm=n_perm, exact=exact,
            max_exact_permutations=max_exact_permutations, seed=seed,
            diagnostics={
                "exchangeability_basis": "iid point-law equality implies iid block exchangeability",
                "kernel": f"{raw_kernel} on unordered point bags",
                "point_kernel": point_kernel,
                "point_kernel_bandwidth": float(point_kernel_bandwidth),
                "bag_kernel_bandwidth": float(bag_kernel_bandwidth),
                "bandwidth_fixed_before_labels": True,
                "raw_features_cached": True,
                "persistent_homology_calls": 0,
                "kernel_characteristicness": (
                    "characteristic on fixed-size unordered bags via a Gaussian "
                    "kernel on the characteristic point mean embedding"
                    if raw_kernel == "gaussian_mean_embedding" else
                    "not characteristic for arbitrary bag laws; sensitivity only"
                ),
                "point_law_identification": (
                    "proved on the stated iid product-block model"
                    if raw_kernel == "gaussian_mean_embedding" else
                    "not claimed for unrestricted bag laws"
                ),
                "cache_dir": cache_dir,
            },
        )

    return _measure(run)


def hybrid_block_mmd(
    cloud0, cloud1, *, regime: str = REGIME_I, m: int = LOCKED_M,
    alpha: float = 0.50, partition_seed: Optional[int] = 0,
    partition0=None, partition1=None,
    point_kernel: str = DEFAULT_RAW_POINT_KERNEL,
    point_kernel_bandwidth: float = DEFAULT_RAW_POINT_BANDWIDTH,
    bag_kernel_bandwidth: float = DEFAULT_RAW_BAG_BANDWIDTH,
    raw_kernel: str = "gaussian_mean_embedding",
    barcode_kernel_bandwidth: float = DEFAULT_KERNEL_BANDWIDTH,
    filtration: str = "vr", homology_dims: Sequence[int] = LOCKED_HOMOLOGY_DIMS,
    max_edge_length: Optional[float] = None, grid_size: int = DEFAULT_GRID_SIZE,
    dtm_k: int = DEFAULT_DTM_K, n_perm: int = 999, exact: bool = False,
    max_exact_permutations: int = 100_000, seed: Optional[int] = 0,
    cache_dir: Optional[str] = None, partition_is_data_independent: bool = True,
    feature_tuning_is_label_independent: bool = True,
) -> dict:
    """Raw-plus-persistence fixed-block MMD with a predeclared weight.

    The persistence diagrams and raw block features are computed once before
    permutation.  For ``alpha>0`` the declared target remains ``P0=P1``;
    ``alpha=0`` is explicitly returned as a barcode-law diagnostic.
    """
    _require_regime("HybridBlockMMD", regime)
    _validate_block_method_controls(
        partition_is_data_independent, feature_tuning_is_label_independent)
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if float(barcode_kernel_bandwidth) <= 0:
        raise ValueError("barcode_kernel_bandwidth must be positive")
    x0, x1 = _validate_cloud_pair(cloud0, cloud1)
    if int(m) < 1:
        raise ValueError("m must be >= 1")
    m = int(m)
    dims = _validate_ph_options(filtration, homology_dims)
    point_kernel, raw_kernel = _validate_raw_kernel_options(
        point_kernel, raw_kernel, point_kernel_bandwidth, bag_kernel_bandwidth)

    def run():
        blocks0, indices0, remainder0 = _partition_or_draw(
            x0, m, partition_seed, partition0, "partition0")
        blocks1, indices1, remainder1 = _partition_or_draw(
            x1, m, None if partition_seed is None else int(partition_seed) + 1,
            partition1, "partition1")
        blocks = list(blocks0) + list(blocks1)
        features = _raw_block_features(
            blocks, point_kernel=point_kernel,
            point_bandwidth=point_kernel_bandwidth)
        raw_gram = _raw_block_gram(
            features, point_kernel=point_kernel,
            point_bandwidth=point_kernel_bandwidth,
            bag_bandwidth=bag_kernel_bandwidth, raw_kernel=raw_kernel)
        if alpha < 1.0:
            diagrams = [
                _ph(block, filtration=filtration, homology_dims=dims,
                    max_edge_length=max_edge_length, grid_size=grid_size,
                    dtm_k=dtm_k, cache_dir=cache_dir)
                for block in blocks
            ]
            barcode_gram = _diagram_gram(diagrams, barcode_kernel_bandwidth)
        else:
            barcode_gram = np.zeros_like(raw_gram)
        gram = float(alpha) * raw_gram + (1.0 - float(alpha)) * barcode_gram
        return _block_result_from_gram(
            candidate="HybridBlockMMD", method="hybrid_raw_barcode_block_mmd",
            target=_block_target(
                m, barcode=float(alpha) == 0.0,
                raw_kernel=raw_kernel if float(alpha) > 0 else None,
            ), regime=regime,
            x0=x0, x1=x1, blocks0=blocks0, blocks1=blocks1,
            indices0=indices0, indices1=indices1, remainder0=remainder0,
            remainder1=remainder1, gram=gram, n_perm=n_perm, exact=exact,
            max_exact_permutations=max_exact_permutations, seed=seed,
            diagnostics={
                "exchangeability_basis": "iid point-law equality implies iid block exchangeability",
                "kernel": "alpha*K_raw + (1-alpha)*K_barcode",
                "alpha": float(alpha),
                "alpha_fixed_before_labels": True,
                "raw_kernel": raw_kernel,
                "point_kernel": point_kernel,
                "point_kernel_bandwidth": float(point_kernel_bandwidth),
                "bag_kernel_bandwidth": float(bag_kernel_bandwidth),
                "barcode_kernel": "tensor-product degree-tagged persistence scale-space",
                "barcode_kernel_bandwidth": float(barcode_kernel_bandwidth),
                "raw_features_cached": True,
                "persistent_homology_calls": int(len(blocks)) if alpha < 1.0 else 0,
                "kernel_characteristicness": (
                    "characteristic for alpha>0 through the raw unordered-bag component"
                    if alpha > 0 else
                    "barcode characteristicness only on the locked bounded diagram class"
                ),
                "point_law_identification": (
                    "proved on the stated iid product-block model for alpha>0"
                    if alpha > 0 else "not claimed; barcode-law diagnostic"
                ),
                "cache_dir": cache_dir,
            },
        )

    return _measure(run)


def sc_a_blockwise_label_permutation(*args, **kwargs) -> dict:
    """SC-A-style pooled block-label permutation using the raw block Gram.

    Blocks are formed separately within the original arms and then pooled for
    label permutations. With the same partition and raw kernel this is
    algebraically identical to ``RawBlockMMD``. The distinction is retained
    so tournament reports can compare SC-A's pooled-label framing with the
    explicit block-kernel candidate without treating them as independent
    methods.
    """
    result = raw_block_mmd(*args, **kwargs)
    result["candidate"] = "SC-A-Block"
    result["method"] = "pooled_block_label_permutation_raw_mmd"
    result["diagnostics"] = dict(result["diagnostics"])
    result["diagnostics"].update({
        "pooled_label_framing": True,
        "block_construction": "separate within original arm, then pool labels",
        "equivalent_to": "RawBlockMMD with the same frozen partitions and kernel",
    })
    return result


def raw_block_repeated_partition_test(*args, **kwargs):
    """Refuse repeated or overlapping raw-block aggregation."""
    raise ValueError(
        "repeated-partition aggregation is not implemented for RawBlockMMD: "
        "use one frozen disjoint partition; overlapping or repeatedly reused "
        "points are not independent block observations"
    )


# ---------------------------------------------------------------------------
# SC-A: full point-law pooled label permutation


def sc_a_label_permutation(
    cloud0,
    cloud1,
    *,
    regime: str = REGIME_I,
    filtration: str = "vr",
    homology_dims: Sequence[int] = LOCKED_HOMOLOGY_DIMS,
    max_edge_length: Optional[float] = None,
    grid_size: int = DEFAULT_GRID_SIZE,
    dtm_k: int = DEFAULT_DTM_K,
    kernel_bandwidth: float = DEFAULT_KERNEL_BANDWIDTH,
    n_perm: int = 999,
    exact: bool = False,
    max_exact_permutations: int = 100_000,
    seed: Optional[int] = 0,
    cache_dir: Optional[str] = None,
) -> dict:
    """SC-A, the full point-law label-permutation baseline.

    The observed and every permuted split are passed through the PH extractor.
    Consequently this method is exact under point-level exchangeability for
    ``P0=P1`` but does not target the weaker fixed-``m`` barcode-law null.
    ``exact=True`` enumerates all ``choose(n0+n1, n0)`` splits and reports the
    unrandomized finite permutation p-value.
    """
    _require_regime("SC-A", regime)
    x0, x1 = _validate_cloud_pair(cloud0, cloud1)
    dims = _validate_ph_options(filtration, homology_dims)
    pooled = np.vstack([x0, x1])
    n0 = len(x0)

    def run():
        observed_mask = np.zeros(len(pooled), dtype=bool)
        observed_mask[:n0] = True

        def statistic(mask):
            d0 = _ph(pooled[mask], filtration=filtration, homology_dims=dims,
                     max_edge_length=max_edge_length, grid_size=grid_size,
                     dtm_k=dtm_k, cache_dir=cache_dir)
            d1 = _ph(pooled[~mask], filtration=filtration, homology_dims=dims,
                     max_edge_length=max_edge_length, grid_size=grid_size,
                     dtm_k=dtm_k, cache_dir=cache_dir)
            return _joint_discrepancy(d0, d1, kernel_bandwidth)

        observed = statistic(observed_mask)
        masks, exact_used = _label_masks(
            len(pooled), n0, n_perm=n_perm, exact=exact,
            max_exact_permutations=max_exact_permutations, seed=seed,
        )
        null = np.asarray([statistic(mask) for mask in masks], dtype=float)
        return {
            "candidate": "SC-A",
            "method": "pooled_point_label_permutation",
            "regime": regime,
            "inferential_target": "H0^law: P0 = P1",
            "statistic": float(observed),
            "pvalue": _permutation_pvalue(observed, null, exact_used),
            "posterior_quantity": None,
            "null_statistics": null,
            "n_permutations": int(len(null)),
            "exact_enumeration": bool(exact_used),
            "n0": int(len(x0)),
            "n1": int(len(x1)),
            "diagnostics": {
                "exchangeability_basis": "iid point-level exchangeability",
                "primary_phase5_target": "not H0,m^bar; this is the strongest-simple baseline",
                "filtration": filtration,
                "homology_dims": dims,
                "grid_size": int(grid_size),
                "dtm_k": int(dtm_k),
                "kernel": "tensor-product degree-tagged universal persistence scale-space",
                "kernel_bandwidth": float(kernel_bandwidth),
                "recomputed_diagrams_for_each_split": True,
                "pooled_split_count": int(math.comb(len(pooled), n0)),
                "cache_dir": cache_dir,
            },
        }

    return _measure(run)


# ---------------------------------------------------------------------------
# SC-B: fixed disjoint barcode blocks


def _validate_partition(indices, n: int, m: int, name: str) -> np.ndarray:
    out = np.asarray(indices, dtype=int)
    if out.ndim != 2 or out.shape[1] != m or out.shape[0] < 1:
        raise ValueError(f"{name} must have shape (K, m) with K >= 1")
    if np.any(out < 0) or np.any(out >= n):
        raise ValueError(f"{name} contains an out-of-range point index")
    if len(np.unique(out)) != out.size:
        raise ValueError(
            f"{name} contains repeated point indices; overlapping blocks are "
            "not allowed in the confirmatory SC-B path"
        )
    return out


def disjoint_partition(cloud, m: int = LOCKED_M, seed: Optional[int] = 0):
    """Return one frozen random disjoint partition and its unused remainder.

    The returned indices, rather than a collection of overlapping sampled
    subclouds, are the confirmatory replication record.  The remainder is
    intentionally discarded, so ``K=floor(n/m)`` is visible and honest.
    """
    points = _as_cloud(cloud, "cloud")
    if int(m) < 1:
        raise ValueError("m must be >= 1")
    m = int(m)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(points))
    K = len(points) // m
    indices = permutation[:K * m].reshape(K, m)
    remainder = permutation[K * m:]
    if K < 1:
        raise ValueError(f"cloud has n={len(points)} < m={m}; no barcode block exists")
    return points[indices], indices, remainder


def _partition_or_draw(cloud, m: int, seed: Optional[int], supplied, name: str):
    points = _as_cloud(cloud, name.replace("_", ""))
    if supplied is not None:
        indices = _validate_partition(supplied, len(points), m, name)
        used = np.unique(indices)
        remainder = np.setdiff1d(np.arange(len(points)), used, assume_unique=True)
        return points[indices], indices, remainder
    return disjoint_partition(points, m=m, seed=seed)


def sc_b_disjoint_mmd(
    cloud0,
    cloud1,
    *,
    regime: str = REGIME_I,
    m: int = LOCKED_M,
    partition_seed: Optional[int] = 0,
    partition0=None,
    partition1=None,
    filtration: str = "vr",
    homology_dims: Sequence[int] = LOCKED_HOMOLOGY_DIMS,
    max_edge_length: Optional[float] = None,
    grid_size: int = DEFAULT_GRID_SIZE,
    dtm_k: int = DEFAULT_DTM_K,
    kernel_bandwidth: float = DEFAULT_KERNEL_BANDWIDTH,
    n_perm: int = 999,
    exact: bool = False,
    max_exact_permutations: int = 100_000,
    seed: Optional[int] = 0,
    cache_dir: Optional[str] = None,
) -> dict:
    """SC-B, the fixed-disjoint-block barcode-law comparison.

    A single partition is frozen for the call.  Its blocks are independent
    barcode draws under Regime I, while any unused points are reported and
    discarded.  When a random partition is drawn, arm zero uses
    ``partition_seed`` and arm one uses ``partition_seed + 1``.  The MMD
    permutation loop consumes only the cached block diagrams, never
    overlapping subclouds and never new PH calculations.
    """
    _require_regime("SC-B", regime)
    x0, x1 = _validate_cloud_pair(cloud0, cloud1)
    if int(m) < 2:
        raise ValueError("m must be >= 2")
    m = int(m)
    dims = _validate_ph_options(filtration, homology_dims)

    def run():
        blocks0, indices0, remainder0 = _partition_or_draw(
            x0, m, partition_seed, partition0, "partition0")
        blocks1, indices1, remainder1 = _partition_or_draw(
            x1, m, None if partition_seed is None else int(partition_seed) + 1,
            partition1, "partition1")
        diagrams0 = [
            _ph(block, filtration=filtration, homology_dims=dims,
                max_edge_length=max_edge_length, grid_size=grid_size,
                dtm_k=dtm_k, cache_dir=cache_dir)
            for block in blocks0
        ]
        diagrams1 = [
            _ph(block, filtration=filtration, homology_dims=dims,
                max_edge_length=max_edge_length, grid_size=grid_size,
                dtm_k=dtm_k, cache_dir=cache_dir)
            for block in blocks1
        ]
        diagrams = diagrams0 + diagrams1
        K0, K1 = len(diagrams0), len(diagrams1)
        group0 = np.zeros(K0 + K1, dtype=bool)
        group0[:K0] = True
        gram = _diagram_gram(diagrams, kernel_bandwidth)
        observed = _mmd2_from_gram(gram, group0)
        masks, exact_used = _label_masks(
            K0 + K1, K0, n_perm=n_perm, exact=exact,
            max_exact_permutations=max_exact_permutations, seed=seed,
        )
        null = np.asarray([_mmd2_from_gram(gram, mask) for mask in masks], dtype=float)
        target = ("H0,25^bar: Phi^25_0:1(P0) = Phi^25_0:1(P1)"
                  if m == LOCKED_M else
                  f"H0,{m}^bar: Phi^{m}_0:1(P0) = Phi^{m}_0:1(P1)"
                  " (unlocked sensitivity size)")
        return {
            "candidate": "SC-B",
            "method": "disjoint_fixed_m_barcode_mmd",
            "regime": regime,
            "inferential_target": target,
            "statistic": float(observed),
            "pvalue": _permutation_pvalue(observed, null, exact_used),
            "posterior_quantity": None,
            "null_statistics": null,
            "n_permutations": int(len(null)),
            "exact_enumeration": bool(exact_used),
            "m": m,
            "K0": K0,
            "K1": K1,
            "K_a": [K0, K1],
            "n0": int(len(x0)),
            "n1": int(len(x1)),
            "remainder0": int(len(remainder0)),
            "remainder1": int(len(remainder1)),
            "block_indices0": indices0,
            "block_indices1": indices1,
            "diagnostics": {
                "barcode_replication_basis": "independent disjoint point blocks under iid sampling",
                "effective_sample_size": {"K0": K0, "K1": K1},
                "overlapping_blocks_used": False,
                "persistent_homology_calls": K0 + K1,
                "persistent_homology_recomputed_in_permutation_loop": False,
                "filtration": filtration,
                "homology_dims": dims,
                "grid_size": int(grid_size),
                "dtm_k": int(dtm_k),
                "kernel": "tensor-product degree-tagged universal persistence scale-space",
                "kernel_bandwidth": float(kernel_bandwidth),
                "partition_frozen_for_call": True,
                "partition_seed": partition_seed,
                "kernel_characteristicness": (
                    "tensor product of exponentiated persistence scale-space "
                    "kernels; characteristic claim applies to the locked VR "
                    "contract on bounded per-degree diagram classes"
                ),
                "cache_dir": cache_dir,
            },
        }

    return _measure(run)


def sc_b_production_test(
    cloud0,
    cloud1,
    *,
    regime: str = REGIME_I,
    partition_seed: Optional[int] = 0,
    partition0=None,
    partition1=None,
    n_perm: int = 999,
    exact: bool = False,
    max_exact_permutations: int = 100_000,
    seed: Optional[int] = 0,
    cache_dir: Optional[str] = None,
    partition_is_data_independent: bool = True,
) -> dict:
    """Run the frozen, target-matched SC-B production procedure.

    The production contract is deliberately narrower than the prototype:
    Regime I, Vietoris--Rips, degrees ``(0, 1)``, ``m=25``, bandwidth ``0.10``,
    and at least ``PRODUCTION_MIN_BLOCKS`` disjoint blocks per arm.  The
    supplied partition, when present, must have been fixed without looking at
    point coordinates.  This declaration is checked as a refusal mode but
    cannot certify a caller's external partition-construction history.

    Exactly one frozen partition is used.  Repeated partitions and any
    uncorrected aggregation are intentionally not part of the production
    API; see ``sc_b_repeated_partition_test`` and the Phase-5D note.
    """
    _require_regime("SC-B-production", regime)
    if not partition_is_data_independent:
        raise ValueError(
            "SC-B production requires a data-independent partition fixed "
            "without point coordinates; data-dependent partition selection is "
            "unsupported"
        )
    x0, x1 = _validate_cloud_pair(cloud0, cloud1)
    if len(x0) // LOCKED_M < PRODUCTION_MIN_BLOCKS:
        raise ValueError(
            f"cloud0 has only {len(x0) // LOCKED_M} disjoint m={LOCKED_M} blocks; "
            f"production SC-B requires at least {PRODUCTION_MIN_BLOCKS}"
        )
    if len(x1) // LOCKED_M < PRODUCTION_MIN_BLOCKS:
        raise ValueError(
            f"cloud1 has only {len(x1) // LOCKED_M} disjoint m={LOCKED_M} blocks; "
            f"production SC-B requires at least {PRODUCTION_MIN_BLOCKS}"
        )
    if partition0 is not None and int(np.asarray(partition0).shape[0]) < PRODUCTION_MIN_BLOCKS:
        raise ValueError(
            f"supplied partition0 has {int(np.asarray(partition0).shape[0])} blocks; "
            f"production SC-B requires at least {PRODUCTION_MIN_BLOCKS} disjoint "
            f"blocks per arm"
        )
    if partition1 is not None and int(np.asarray(partition1).shape[0]) < PRODUCTION_MIN_BLOCKS:
        raise ValueError(
            f"supplied partition1 has {int(np.asarray(partition1).shape[0])} blocks; "
            f"production SC-B requires at least {PRODUCTION_MIN_BLOCKS} disjoint "
            f"blocks per arm"
        )

    result = sc_b_disjoint_mmd(
        x0,
        x1,
        regime=regime,
        m=LOCKED_M,
        partition_seed=partition_seed,
        partition0=partition0,
        partition1=partition1,
        filtration="ripser",
        homology_dims=LOCKED_HOMOLOGY_DIMS,
        kernel_bandwidth=DEFAULT_KERNEL_BANDWIDTH,
        n_perm=n_perm,
        exact=exact,
        max_exact_permutations=max_exact_permutations,
        seed=seed,
        cache_dir=cache_dir,
    )
    result["production_api"] = PRODUCTION_API_VERSION
    result["diagnostics"].update({
        "production_api": PRODUCTION_API_VERSION,
        "target_lock": "H0,25^bar joint VR barcode law in degrees 0:1",
        "sampling_unit": "point",
        "partition_data_independent_declared": True,
        "minimum_blocks_per_arm": PRODUCTION_MIN_BLOCKS,
        "repeated_partition_aggregation": "refused",
        "unsupported_regimes": [
            "stationary_mixing_process",
            "fixed_cloud",
            "explicit_generative_model",
        ],
        "kernel_characteristicness": (
            "tensor product of exponentiated persistence scale-space kernels; "
            "characteristic claim applies to the locked VR contract on bounded "
            "per-degree diagram classes"
        ),
    })
    return result


def sc_b_repeated_partition_test(*args, **kwargs):
    """Refuse unsupported repeated-partition aggregation explicitly.

    A single fixed partition is the locked confirmatory procedure.  Reusing
    points across partitions creates dependent barcode summaries, so averaging
    statistics, pooling permutation draws, Fisher-combining p-values, or
    taking an uncorrected minimum p-value has no validity claim here.
    """
    raise ValueError(
        "repeated-partition aggregation is not implemented for production "
        "SC-B: use one frozen disjoint partition; overlapping or repeatedly "
        "reused points are not independent barcode replicates"
    )


# ---------------------------------------------------------------------------
# SC-C: finite persistent-Betti vector and smoothed point bootstrap


def persistent_betti_vector(
    diagrams: Sequence[np.ndarray],
    grid: Sequence[float] = DEFAULT_GRID,
    *,
    normalize_by: Optional[float] = None,
) -> np.ndarray:
    """Return a frozen finite vector of persistent Betti numbers.

    The coordinate indexed by ``(degree, r, s)`` is
    ``#{(birth, death): birth <= r and death > s}`` for grid values ``r <= s``.
    Coordinates with ``r > s`` are set to zero.  Passing an explicit grid is
    recommended because deriving it separately from the two clouds changes
    the estimand.
    """
    values = np.asarray(grid, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("grid must be a finite one-dimensional array with at least two values")
    if np.any(np.diff(values) < 0):
        raise ValueError("grid must be sorted in non-decreasing order")
    out = []
    for dgm in diagrams:
        dgm = np.asarray(dgm, dtype=float).reshape(-1, 2)
        if len(dgm):
            finite = dgm[np.isfinite(dgm).all(axis=1)]
            matrix = ((finite[:, 1, None, None] > values[None, None, :])
                      & (finite[:, 0, None, None] <= values[None, :, None]))
            matrix = matrix.sum(axis=0, dtype=float)
        else:
            matrix = np.zeros((len(values), len(values)), dtype=float)
        matrix[np.triu(np.ones_like(matrix, dtype=bool), k=0) == 0] = 0.0
        out.append(matrix.ravel())
    if not out:
        raise ValueError("at least one homology degree is required")
    vector = np.concatenate(out)
    if normalize_by is not None:
        if normalize_by <= 0:
            raise ValueError("normalize_by must be positive")
        vector = vector / float(normalize_by)
    return vector


def _bootstrap_cloud(base: np.ndarray, size: int, rng: np.random.Generator,
                     smoothing: bool, bandwidth: float) -> np.ndarray:
    indices = rng.integers(0, len(base), size=size)
    out = base[indices].copy()
    if smoothing:
        out += rng.normal(0.0, bandwidth, size=out.shape)
    return out


def _validate_grid(grid) -> np.ndarray:
    values = np.asarray(grid, dtype=float)
    # persistent_betti_vector performs the detailed validation; this helper
    # only ensures the returned metadata is an independent immutable snapshot.
    persistent_betti_vector([np.zeros((0, 2))], values)
    return values.copy()


def sc_c_finite_vector(
    cloud0,
    cloud1,
    *,
    regime: str = REGIME_I,
    filtration: str = "vr",
    homology_dims: Sequence[int] = LOCKED_HOMOLOGY_DIMS,
    max_edge_length: Optional[float] = None,
    grid_size: int = DEFAULT_GRID_SIZE,
    dtm_k: int = DEFAULT_DTM_K,
    grid: Sequence[float] = DEFAULT_GRID,
    bootstrap_bandwidth: float = 0.05,
    n_draws: int = 399,
    seed: Optional[int] = 0,
    smoothing: bool = True,
    cache_dir: Optional[str] = None,
) -> dict:
    """SC-C finite-vector bootstrap prototype.

    The null bootstrap draws both arms from the pooled empirical point law,
    with optional Gaussian kernel jitter.  The statistic is the scaled
    Euclidean contrast of normalized persistent-Betti vectors.  This is a
    finite-vector equality prototype under the Euclidean stabilizing-statistic
    conditions of Roycraft, Krebs, and Polonik; it is not a test of the full
    fixed-25 barcode law.

    ``smoothing=False`` is retained specifically as the naïve-bootstrap
    negative control.  It should not be promoted to a production method for
    persistent Betti statistics merely because its output is convenient.
    """
    _require_regime("SC-C", regime)
    x0, x1 = _validate_cloud_pair(cloud0, cloud1)
    dims = _validate_ph_options(filtration, homology_dims)
    grid_values = _validate_grid(grid)
    if bootstrap_bandwidth < 0:
        raise ValueError("bootstrap_bandwidth must be non-negative")
    if n_draws < 1:
        raise ValueError("n_draws must be >= 1")

    def run():
        d0 = _ph(x0, filtration=filtration, homology_dims=dims,
                 max_edge_length=max_edge_length, grid_size=grid_size,
                 dtm_k=dtm_k, cache_dir=cache_dir)
        d1 = _ph(x1, filtration=filtration, homology_dims=dims,
                 max_edge_length=max_edge_length, grid_size=grid_size,
                 dtm_k=dtm_k, cache_dir=cache_dir)
        v0 = persistent_betti_vector(d0, grid_values, normalize_by=len(x0))
        v1 = persistent_betti_vector(d1, grid_values, normalize_by=len(x1))
        scale = math.sqrt(len(x0) * len(x1) / (len(x0) + len(x1)))
        observed = float(scale * np.linalg.norm(v0 - v1))

        rng = np.random.default_rng(seed)
        pooled = np.vstack([x0, x1])
        null = np.empty(int(n_draws), dtype=float)
        for b in range(int(n_draws)):
            boot0 = _bootstrap_cloud(pooled, len(x0), rng, smoothing, bootstrap_bandwidth)
            boot1 = _bootstrap_cloud(pooled, len(x1), rng, smoothing, bootstrap_bandwidth)
            bd0 = _ph(boot0, filtration=filtration, homology_dims=dims,
                      max_edge_length=max_edge_length, grid_size=grid_size,
                      dtm_k=dtm_k, cache_dir=cache_dir)
            bd1 = _ph(boot1, filtration=filtration, homology_dims=dims,
                      max_edge_length=max_edge_length, grid_size=grid_size,
                      dtm_k=dtm_k, cache_dir=cache_dir)
            bv0 = persistent_betti_vector(bd0, grid_values, normalize_by=len(x0))
            bv1 = persistent_betti_vector(bd1, grid_values, normalize_by=len(x1))
            null[b] = scale * np.linalg.norm(bv0 - bv1)

        return {
            "candidate": "SC-C",
            "method": "smoothed_finite_persistent_betti_vector" if smoothing
                      else "naive_finite_persistent_betti_vector",
            "regime": regime,
            "inferential_target": "H0^finite-vector: equality of the frozen normalized persistent-Betti mean vector",
            "statistic": observed,
            "pvalue": p_value(observed, null, alternative="greater"),
            "posterior_quantity": None,
            "null_statistics": null,
            "n_draws": int(n_draws),
            "bootstrap": "smoothed" if smoothing else "naive_negative_control",
            "observed_vector0": v0,
            "observed_vector1": v1,
            "grid": grid_values,
            "diagnostics": {
                "point_level_replication": True,
                "bootstrap_null": "both arms resampled from pooled empirical law",
                "smoothing_bandwidth": float(bootstrap_bandwidth),
                "smoothed_bootstrap_source": "Roycraft, Krebs & Polonik (2023), DOI 10.1214/23-AOS2277",
                "persistent_homology_recomputed_in_bootstrap_loop": True,
                "filtration": filtration,
                "homology_dims": dims,
                "grid_size": int(grid_size),
                "dtm_k": int(dtm_k),
                "normalization": "persistent-Betti counts divided by arm cloud size",
                "primary_phase5_target": "not H0,25^bar; finite-vector prototype only",
                "cache_dir": cache_dir,
            },
        }

    return _measure(run)


def sc_c_naive_bootstrap(*args, **kwargs) -> dict:
    """SC-C's declared naïve-bootstrap negative control."""
    kwargs = dict(kwargs)
    kwargs["smoothing"] = False
    return sc_c_finite_vector(*args, **kwargs)


# ---------------------------------------------------------------------------
# Common interface and source-setting record


def run_single_cloud_test(candidate: str, cloud0, cloud1, *, regime: str = REGIME_I,
                          **kwargs) -> dict:
    """Dispatch one Phase 5B candidate through the common result interface."""
    key = _normalise_candidate(candidate)
    # Route before binding candidate-specific keyword arguments.  This keeps
    # an incompatible observation model a deterministic scientific error,
    # even when the caller supplied options belonging to another candidate.
    _require_regime(key.upper(), regime)
    if key == "sc-a":
        return sc_a_label_permutation(cloud0, cloud1, regime=regime, **kwargs)
    if key == "sc-b":
        return sc_b_disjoint_mmd(cloud0, cloud1, regime=regime, **kwargs)
    if key == "sc-b-production":
        return sc_b_production_test(cloud0, cloud1, regime=regime, **kwargs)
    if key == "raw-block-mmd":
        return raw_block_mmd(cloud0, cloud1, regime=regime, **kwargs)
    if key == "hybrid-block-mmd":
        return hybrid_block_mmd(cloud0, cloud1, regime=regime, **kwargs)
    if key == "sc-a-block":
        return sc_a_blockwise_label_permutation(
            cloud0, cloud1, regime=regime, **kwargs)
    return sc_c_finite_vector(cloud0, cloud1, regime=regime, **kwargs)


def roycraft_reference_setting() -> dict:
    """Return the pre-registered SC-C source-aligned pilot setting.

    The source studies persistent Betti numbers of binomial/Poisson point
    sets in Euclidean space and compares ordinary with smoothed bootstrap
    inference.  This record deliberately states what is reproduced and what
    is not: the P1 prototype uses its own VR-radius implementation and a
    two-cloud contrast, so it is not a claim to reproduce the source's full
    confidence-table fleet.
    """
    return {
        "source": "Roycraft, Krebs & Polonik (2023), Annals of Statistics 51, 1484-1509",
        "doi": "10.1214/23-AOS2277",
        "point_model": "binomial point samples from a Euclidean density",
        "statistic": "finite vector of persistent Betti numbers",
        "filtration": "Vietoris-Rips radius filtration",
        "comparison": "Gaussian smoothed bootstrap versus ordinary bootstrap",
        "p1_adaptation": "two independent clouds and a pooled-null finite-vector contrast",
        "not_claimed": "full reproduction of the source confidence-coverage tables",
    }
