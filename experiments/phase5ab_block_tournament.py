"""Phase 5AB tournament for raw and hybrid disjoint-block methods.

This runner reuses the Phase 5C DGP constructors, seeds, and reporting
conventions. It keeps the original SC-A and SC-B implementations unchanged
and adds the point-law-sensitive block candidates:

``RawBlockMMD``
    characteristic Gaussian kernel on raw point-kernel mean embeddings;
``HybridBlockMMD``
    a predeclared raw-plus-persistence kernel, with alpha fixed before labels;
``SC-A-Block``
    the pooled-label framing of the same cached raw block Gram matrix.

The default run is a compact reproducibility pilot. The same command accepts
``--replications 500`` for the registered Phase 5-sized run. Workers are used
only across independent original-cloud replications, never inside a
permutation loop. Outputs are a replication parquet, a summary parquet, a
comparison parquet, a figure, and a concise Markdown report.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from experiments.phase5_single_cloud_tournament import (
    ALPHA,
    BETTI_GRID,
    FAMILY_DESCRIPTION,
    FAMILY_ROLE,
    FILTRATION,
    HOMOLOGY_DIMS,
    KERNEL_BANDWIDTH,
    MAX_SC_A_POINT_N,
    M_GRID,
    N_GRID,
    _mc_interval,
    _project_cloud,
    _seed,
    make_cloud_pair,
    overlapping_negative_control,
)
from tda2s.tests.single_cloud import (
    DEFAULT_RAW_BAG_BANDWIDTH,
    DEFAULT_RAW_POINT_BANDWIDTH,
    REGIME_I,
    _label_masks,
    _mmd2_from_gram,
    _permutation_pvalue,
    _point_kernel_matrix,
    _raw_block_features,
    _raw_block_gram,
    _raw_block_kernel,
    _validate_partition,
    disjoint_partition,
    hybrid_block_mmd,
    raw_block_mmd,
    sc_a_blockwise_label_permutation,
    sc_a_label_permutation,
    sc_b_disjoint_mmd,
)


PRIMARY_M = 25
SIZE_BAND = (0.03, 0.08)
MC_CONFIDENCE = 0.95
DEFAULT_REPLICATIONS = 20
DEFAULT_PERMUTATIONS = 39
DEFAULT_WORKERS = 1
DEFAULT_FAMILIES = (
    "iid_null",
    "weak_barcode_null",
    "same_support_density",
    "same_square_four_atom_density",
    "topology_alt",
    "robust_contamination",
    "robust_unequal_cardinality",
    "robust_anisotropic_noise",
    "robust_boundary_truncation",
)
PRIMARY_CANDIDATES = ("SC-A", "SC-B", "RawBlockMMD", "HybridBlockMMD", "SC-A-Block")
HYBRID_ALPHAS = (0.25, 0.50, 0.75, 1.00)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
DEFAULT_REPLICATION_OUTPUT = os.path.join(RESULTS_DIR, "phase5ab_block_replications.parquet")
DEFAULT_SUMMARY_OUTPUT = os.path.join(RESULTS_DIR, "phase5ab_block_summary.parquet")
DEFAULT_COMPARISON_OUTPUT = os.path.join(RESULTS_DIR, "phase5ab_block_comparison.parquet")
DEFAULT_FIGURE = os.path.join(RESULTS_DIR, "phase5ab_block_comparison.png")
DEFAULT_REPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "phase5ab_block_report.md")
EXTRA_FAMILY_ROLE = "target_mismatch"
EXTRA_FAMILY_DESCRIPTION = (
    "four-atom square support with p=(.25,.25,.25,.25) versus "
    "q=(.70,.10,.10,.10)"
)


@dataclass(frozen=True)
class Cell:
    family: str
    n0: int
    n1: int
    m: int

    @property
    def cell_id(self) -> str:
        return f"{self.family}_n{self.n0}_n1{self.n1}_m{self.m}"


def _families(value: str | None) -> tuple[str, ...]:
    selected = DEFAULT_FAMILIES if value is None else tuple(v.strip() for v in value.split(","))
    known = set(FAMILY_ROLE) | {"same_square_four_atom_density"}
    unknown = sorted(set(selected) - known)
    if unknown:
        raise ValueError(f"unknown Phase 5 family/families: {unknown}")
    return selected


def _make_cells(families: Sequence[str], n_grid: Sequence[int], m_values: Sequence[int]) -> list[Cell]:
    cells = []
    for family in families:
        for n in n_grid:
            n1 = int(np.ceil(n * 1.25)) if family == "robust_unequal_cardinality" else int(n)
            for m in m_values:
                cells.append(Cell(family, int(n), n1, int(m)))
    return cells


def _common_raw_kwargs():
    return {
        "point_kernel": "gaussian",
        "point_kernel_bandwidth": DEFAULT_RAW_POINT_BANDWIDTH,
        "bag_kernel_bandwidth": DEFAULT_RAW_BAG_BANDWIDTH,
        "raw_kernel": "gaussian_mean_embedding",
    }


def _common_ph_kwargs():
    return {
        "filtration": FILTRATION,
        "homology_dims": HOMOLOGY_DIMS,
        "max_edge_length": None,
        "grid_size": 64,
        "dtm_k": 20,
        "kernel_bandwidth": KERNEL_BANDWIDTH,
    }


def _family_role(family: str) -> str:
    return FAMILY_ROLE.get(family, EXTRA_FAMILY_ROLE)


def _family_description(family: str) -> str:
    return FAMILY_DESCRIPTION.get(family, EXTRA_FAMILY_DESCRIPTION)


def _make_cloud_pair(family: str, n0: int, n1: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if family != "same_square_four_atom_density":
        return make_cloud_pair(family, n0, n1, seed)
    rng = np.random.default_rng(seed)
    square = np.asarray([
        [0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]
    ])
    p = np.full(4, 0.25)
    q = np.asarray([0.70, 0.10, 0.10, 0.10])
    return square[rng.choice(4, size=int(n0), p=p)], square[rng.choice(4, size=int(n1), p=q)]


def _candidate_label(candidate: str, alpha: float | None) -> str:
    return candidate if alpha is None else f"{candidate}-a{alpha:.2f}"


def _method_call(cell: Cell, replication: int, candidate: str, *, alpha: float | None,
                 n_permutations: int) -> tuple[dict, str]:
    cloud_seed = _seed("phase5ab-cloud", cell.cell_id, replication)
    cloud0, cloud1 = _make_cloud_pair(cell.family, cell.n0, cell.n1, cloud_seed)
    partition_seed = _seed("phase5ab-partition", cell.cell_id, replication)
    method_seed = _seed("phase5ab-method", cell.cell_id, replication, candidate, alpha)
    projection = False
    if candidate == "SC-A":
        cloud0 = _project_cloud(cloud0, MAX_SC_A_POINT_N, _seed("phase5ab-A0", cell.cell_id, replication))
        cloud1 = _project_cloud(cloud1, MAX_SC_A_POINT_N, _seed("phase5ab-A1", cell.cell_id, replication))
        projection = len(cloud0) < cell.n0 or len(cloud1) < cell.n1
        result = sc_a_label_permutation(
            cloud0, cloud1, regime=REGIME_I, n_perm=n_permutations,
            seed=method_seed, exact=False, **_common_ph_kwargs(),
        )
    elif candidate == "SC-B":
        result = sc_b_disjoint_mmd(
            cloud0, cloud1, regime=REGIME_I, m=cell.m,
            partition_seed=partition_seed, n_perm=n_permutations,
            seed=method_seed, exact=False, **_common_ph_kwargs(),
        )
    elif candidate == "RawBlockMMD":
        result = raw_block_mmd(
            cloud0, cloud1, regime=REGIME_I, m=cell.m,
            partition_seed=partition_seed, n_perm=n_permutations,
            seed=method_seed, exact=False, **_common_raw_kwargs(),
        )
    elif candidate == "SC-A-Block":
        result = sc_a_blockwise_label_permutation(
            cloud0, cloud1, regime=REGIME_I, m=cell.m,
            partition_seed=partition_seed, n_perm=n_permutations,
            seed=method_seed, exact=False, **_common_raw_kwargs(),
        )
    elif candidate == "HybridBlockMMD":
        hybrid_ph_kwargs = _common_ph_kwargs()
        hybrid_ph_kwargs["barcode_kernel_bandwidth"] = hybrid_ph_kwargs.pop("kernel_bandwidth")
        result = hybrid_block_mmd(
            cloud0, cloud1, regime=REGIME_I, m=cell.m,
            alpha=float(alpha), partition_seed=partition_seed,
            n_perm=n_permutations, seed=method_seed, exact=False,
            **_common_raw_kwargs(), **hybrid_ph_kwargs,
        )
    else:
        raise ValueError(f"unknown candidate {candidate!r}")
    return result, _candidate_label(candidate, alpha)


def _record(cell: Cell, replication: int, candidate: str, alpha: float | None,
            result: dict, *, projection: bool = False) -> dict:
    diagnostics = result.get("diagnostics", {})
    effective = diagnostics.get("effective_sample_size", {})
    is_block = candidate != "SC-A"
    sampling_unit = result.get("sampling_unit") or diagnostics.get("sampling_unit")
    if not sampling_unit:
        sampling_unit = "individual iid point" if candidate == "SC-A" else "disjoint m-point block"
    return {
        "record_type": "candidate",
        "status": "ok",
        "replication": int(replication),
        "cell_id": cell.cell_id,
        "family": cell.family,
        "family_role": _family_role(cell.family),
        "family_description": _family_description(cell.family),
        "candidate": candidate,
        "alpha": np.nan if alpha is None else float(alpha),
        "target": result.get("inferential_target", ""),
        "sampling_unit": sampling_unit,
        "m": int(result.get("m", cell.m)),
        "K0": float(result.get("K0", effective.get("K0", np.nan))) if is_block else np.nan,
        "K1": float(result.get("K1", effective.get("K1", np.nan))) if is_block else np.nan,
        "unused_points0": float(result.get("remainder0", np.nan)) if is_block else np.nan,
        "unused_points1": float(result.get("remainder1", np.nan)) if is_block else np.nan,
        "n_permutations": int(result.get("n_permutations", -1)),
        "statistic": float(result.get("statistic", np.nan)),
        "pvalue": float(result.get("pvalue", np.nan)),
        "reject": bool(float(result.get("pvalue", np.nan)) <= ALPHA),
        "kernel": result.get("kernel", diagnostics.get("kernel", "")),
        "bandwidth": json.dumps({
            key: value for key, value in diagnostics.items()
            if "bandwidth" in key
        }, sort_keys=True, default=str),
        "runtime_seconds": float(result.get("runtime_seconds", np.nan)),
        "peak_memory_bytes": int(result.get("peak_memory_bytes", -1)),
        "peak_rss_bytes": int(result.get("peak_rss_bytes", -1)),
        "overlapping_blocks_used": bool(diagnostics.get("overlapping_blocks_used", False)),
        "projection_used": bool(projection),
        "target_identification": diagnostics.get("point_law_identification", ""),
        "overlap_fraction": 0.0,
        "method_error": "",
    }


def _error_record(cell: Cell, replication: int, candidate: str, alpha: float | None,
                  exc: Exception) -> dict:
    return {
        "record_type": "candidate",
        "status": "failed",
        "replication": int(replication),
        "cell_id": cell.cell_id,
        "family": cell.family,
        "family_role": _family_role(cell.family),
        "family_description": _family_description(cell.family),
        "candidate": _candidate_label(candidate, alpha),
        "alpha": np.nan if alpha is None else float(alpha),
        "target": "",
        "sampling_unit": "",
        "m": int(cell.m), "K0": -1, "K1": -1,
        "unused_points0": -1, "unused_points1": -1,
        "n_permutations": -1, "statistic": np.nan, "pvalue": np.nan,
        "reject": False, "kernel": "", "bandwidth": "",
        "runtime_seconds": np.nan, "peak_memory_bytes": -1,
        "peak_rss_bytes": -1,
        "overlapping_blocks_used": False, "projection_used": False,
        "target_identification": "",
        "overlap_fraction": 0.0,
        "method_error": f"{type(exc).__name__}: {exc}",
    }


def _run_one(args: tuple[Cell, int, tuple[str, ...], int]) -> list[dict]:
    cell, replication, candidates, n_permutations = args
    rows = []
    for candidate in candidates:
        alphas = HYBRID_ALPHAS if candidate == "HybridBlockMMD" else (None,)
        for alpha in alphas:
            try:
                result, label = _method_call(
                    cell, replication, candidate, alpha=alpha,
                    n_permutations=n_permutations,
                )
                projection = candidate == "SC-A" and min(cell.n0, cell.n1) > MAX_SC_A_POINT_N
                rows.append(_record(cell, replication, label, alpha, result, projection=projection))
            except Exception as exc:
                rows.append(_error_record(cell, replication, candidate, alpha, exc))
    return rows


def _overlap_blocks(cloud: np.ndarray, m: int, overlap_fraction: float,
                    seed: int, K: int) -> tuple[list[np.ndarray], np.ndarray, float]:
    overlap = int(round(float(overlap_fraction) * m))
    step = max(1, m - overlap)
    permutation = np.random.default_rng(seed).permutation(len(cloud))
    indices = np.asarray(
        [permutation[k * step:k * step + m] for k in range(int(K))], dtype=int
    )
    blocks = [np.asarray(cloud[index], dtype=float) for index in indices]
    pairwise = []
    for i in range(len(indices)):
        for j in range(i):
            pairwise.append(
                len(set(indices[i]).intersection(indices[j])) / float(m)
            )
    mean_overlap = float(np.mean(pairwise)) if pairwise else 0.0
    return blocks, indices, mean_overlap


def _overlap_raw_negative_control(n: int, m: int, overlap_fraction: float,
                                  replication: int, n_permutations: int) -> dict:
    """Naive overlapping raw-block statistic, retained only as a negative control."""
    cloud0, cloud1 = make_cloud_pair(
        "iid_null", n, n, _seed("phase5ab-overlap-cloud", n, m, overlap_fraction, replication)
    )
    K0 = len(cloud0) // m
    K1 = len(cloud1) // m
    blocks0, indices0, mean0 = _overlap_blocks(
        cloud0, m, overlap_fraction,
        _seed("phase5ab-overlap-0", n, m, overlap_fraction, replication), K0)
    blocks1, indices1, mean1 = _overlap_blocks(
        cloud1, m, overlap_fraction,
        _seed("phase5ab-overlap-1", n, m, overlap_fraction, replication), K1)
    features = _raw_block_features(
        blocks0 + blocks1, point_kernel="gaussian",
        point_bandwidth=DEFAULT_RAW_POINT_BANDWIDTH)
    gram = _raw_block_gram(
        features, point_kernel="gaussian",
        point_bandwidth=DEFAULT_RAW_POINT_BANDWIDTH,
        bag_bandwidth=DEFAULT_RAW_BAG_BANDWIDTH,
        raw_kernel="gaussian_mean_embedding",
    )
    group0 = np.zeros(K0 + K1, dtype=bool)
    group0[:K0] = True
    observed = _mmd2_from_gram(gram, group0)
    masks, exact_used = _label_masks(
        K0 + K1, K0, n_perm=n_permutations, exact=False,
        max_exact_permutations=100_000,
        seed=_seed("phase5ab-overlap-perm", n, m, overlap_fraction, replication),
    )
    null = np.asarray([_mmd2_from_gram(gram, mask) for mask in masks])
    return {
        "statistic": float(observed),
        "pvalue": _permutation_pvalue(observed, null, exact_used),
        "K0": K0, "K1": K1,
        "unused_points0": int(n - len(np.unique(indices0))),
        "unused_points1": int(n - len(np.unique(indices1))),
        "unique_points0": int(len(np.unique(indices0))),
        "unique_points1": int(len(np.unique(indices1))),
        "mean_pairwise_overlap0": mean0,
        "mean_pairwise_overlap1": mean1,
        "n_permutations": len(null),
    }


def run_overlap_replicates(*, replications: int = DEFAULT_REPLICATIONS,
                           n: int = 250, m: int = PRIMARY_M,
                           n_permutations: int = DEFAULT_PERMUTATIONS,
                           overlap_fractions: Sequence[float] = (0.0, 0.25, 0.50, 0.75, 0.90)) -> pd.DataFrame:
    rows = []
    for replication in range(int(replications)):
        for overlap_fraction in overlap_fractions:
            started = time.perf_counter()
            try:
                raw = _overlap_raw_negative_control(
                    n, m, float(overlap_fraction), replication, n_permutations)
                row = {
                    "record_type": "negative_control",
                    "status": "ok",
                    "replication": int(replication),
                    "cell_id": f"overlap_n{n}_m{m}_omega{overlap_fraction:g}",
                    "family": "overlap_iid_null",
                    "family_role": "pseudo_replication_negative_control",
                    "family_description": "iid null with deliberately overlapping blocks",
                    "candidate": "RawBlockMMD-overlap-negative-control",
                    "alpha": np.nan,
                    "target": "invalid confirmatory path: overlapping blocks are not independent",
                    "sampling_unit": "overlapping m-point block (negative control only)",
                    "m": m, "K0": raw["K0"], "K1": raw["K1"],
                    "unused_points0": raw["unused_points0"],
                    "unused_points1": raw["unused_points1"],
                    "n_permutations": raw["n_permutations"],
                    "statistic": raw["statistic"], "pvalue": raw["pvalue"],
                    "reject": bool(raw["pvalue"] <= ALPHA),
                    "kernel": "gaussian_mean_embedding on raw bags",
                    "bandwidth": json.dumps({"point": DEFAULT_RAW_POINT_BANDWIDTH,
                                               "bag": DEFAULT_RAW_BAG_BANDWIDTH}),
                    "runtime_seconds": time.perf_counter() - started,
                    "peak_memory_bytes": -1,
                    "overlapping_blocks_used": True,
                    "projection_used": False,
                    "target_identification": "invalid because block observations are dependent",
                    "overlap_fraction": float(overlap_fraction),
                    "mean_pairwise_overlap0": raw["mean_pairwise_overlap0"],
                    "mean_pairwise_overlap1": raw["mean_pairwise_overlap1"],
                    "method_error": "",
                }
                rows.append(row)

                started = time.perf_counter()
                cloud0, cloud1 = make_cloud_pair(
                    "iid_null", n, n,
                    _seed("phase5ab-overlap-scb-cloud", n, m, overlap_fraction, replication),
                )
                scb = overlapping_negative_control(
                    cloud0, cloud1, m=m, overlap_fraction=float(overlap_fraction),
                    seed=_seed("phase5ab-overlap-scb", n, m, overlap_fraction, replication),
                    n_permutations=n_permutations, cache_dir=None,
                )
                rows.append({
                    **row,
                    "candidate": "SC-B-overlap-negative-control",
                    "target": "invalid confirmatory path: overlapping barcode blocks are not independent",
                    "sampling_unit": "overlapping m-point barcode block (negative control only)",
                    "K0": scb["K0"], "K1": scb["K1"],
                    "unused_points0": int(n - scb["unique_points0"]),
                    "unused_points1": int(n - scb["unique_points1"]),
                    "n_permutations": int(n_permutations),
                    "statistic": scb["statistic"], "pvalue": scb["pvalue"],
                    "reject": bool(scb["pvalue"] <= ALPHA),
                    "kernel": "tensor-product persistence scale-space",
                    "bandwidth": json.dumps({"barcode": KERNEL_BANDWIDTH}),
                    "runtime_seconds": time.perf_counter() - started,
                    "mean_pairwise_overlap0": scb["mean_pairwise_overlap0"],
                    "mean_pairwise_overlap1": scb["mean_pairwise_overlap1"],
                    "target_identification": "invalid because barcode blocks are dependent",
                })
            except Exception as exc:
                rows.append({
                    "record_type": "negative_control", "status": "failed",
                    "replication": int(replication),
                    "cell_id": f"overlap_n{n}_m{m}_omega{overlap_fraction:g}",
                    "family": "overlap_iid_null",
                    "family_role": "pseudo_replication_negative_control",
                    "family_description": "iid null with deliberately overlapping blocks",
                    "candidate": "overlap-negative-control", "alpha": np.nan,
                    "target": "", "sampling_unit": "",
                    "m": m, "K0": np.nan, "K1": np.nan,
                    "unused_points0": np.nan, "unused_points1": np.nan,
                    "n_permutations": n_permutations, "statistic": np.nan,
                    "pvalue": np.nan, "reject": False, "kernel": "",
                    "bandwidth": "", "runtime_seconds": time.perf_counter() - started,
                    "peak_memory_bytes": -1, "overlapping_blocks_used": True,
                    "projection_used": False, "target_identification": "",
                    "overlap_fraction": float(overlap_fraction),
                    "mean_pairwise_overlap0": np.nan,
                    "mean_pairwise_overlap1": np.nan,
                    "method_error": f"{type(exc).__name__}: {exc}",
                })
    return pd.DataFrame(rows)


def run_replicates(*, families: Sequence[str] = DEFAULT_FAMILIES,
                   n_grid: Sequence[int] = (250,), m_values: Sequence[int] = (PRIMARY_M,),
                   replications: int = DEFAULT_REPLICATIONS,
                   n_permutations: int = DEFAULT_PERMUTATIONS,
                   workers: int = DEFAULT_WORKERS,
                   candidates: Sequence[str] = PRIMARY_CANDIDATES) -> pd.DataFrame:
    if replications < 1 or n_permutations < 1:
        raise ValueError("replications and n_permutations must be positive")
    if workers < 1 or workers > 16:
        raise ValueError("workers must be in [1, 16]")
    cells = _make_cells(families, n_grid, m_values)
    candidate_tuple = tuple(candidates)
    args = [
        (cell, replication, candidate_tuple, int(n_permutations))
        for cell in cells for replication in range(int(replications))
    ]
    if workers == 1:
        nested = [_run_one(arg) for arg in args]
    else:
        with ProcessPoolExecutor(max_workers=min(int(workers), 16)) as pool:
            nested = list(pool.map(_run_one, args))
    return pd.DataFrame([row for rows in nested for row in rows])


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    groups = ["candidate", "family", "m", "alpha", "overlap_fraction", "target", "sampling_unit"]
    rows = []
    for keys, group in frame.groupby(groups, dropna=False, sort=True):
        ok = group[group["status"] == "ok"]
        successes = int(ok["reject"].sum())
        total = int(len(ok))
        low, high = _mc_interval(successes, total, MC_CONFIDENCE) if total else (np.nan, np.nan)
        values = dict(zip(groups, keys))
        values.update({
            "replications": total,
            "failed_replications": int(len(group) - total),
            "rejections": successes,
            "rejection_rate": successes / total if total else np.nan,
            "mc_low": low,
            "mc_high": high,
            "in_size_band": bool(SIZE_BAND[0] <= successes / total <= SIZE_BAND[1]) if total else False,
            "K0": float(ok["K0"].mean()) if total else np.nan,
            "K1": float(ok["K1"].mean()) if total else np.nan,
            "unused_points0": float(ok["unused_points0"].mean()) if total else np.nan,
            "unused_points1": float(ok["unused_points1"].mean()) if total else np.nan,
            "runtime_seconds": float(ok["runtime_seconds"].mean()) if total else np.nan,
            "peak_memory_bytes": float(ok["peak_memory_bytes"].mean()) if total else np.nan,
            "peak_rss_bytes": float(ok["peak_rss_bytes"].mean()) if total else np.nan,
        })
        rows.append(values)
    return pd.DataFrame(rows)


def _comparison(summary: pd.DataFrame) -> pd.DataFrame:
    """Create the requested side-by-side primary-m=25 comparison table."""
    primary = summary[summary["m"] == PRIMARY_M].copy()
    # alpha=.50 is the headline hybrid; all alpha values remain in summary.
    primary = primary[primary["candidate"].isin(
        ["SC-A", "SC-B", "RawBlockMMD", "HybridBlockMMD-a0.50", "SC-A-Block"]
    )]
    methods = sorted(primary["candidate"].dropna().unique())
    rows = []
    for method in methods:
        sub = primary[primary["candidate"] == method]
        target = sub["target"].dropna().iloc[0] if len(sub["target"].dropna()) else ""
        unit = sub["sampling_unit"].dropna().iloc[0] if len(sub["sampling_unit"].dropna()) else ""

        def rate(family):
            value = sub[(sub["family"] == family) & sub["overlap_fraction"].eq(0.0)]["rejection_rate"]
            return float(value.iloc[0]) if len(value) else np.nan

        overlap = summary[(summary["candidate"] == method + "-overlap-negative-control")
                          & summary["family"].eq("overlap_iid_null")
                          & summary["overlap_fraction"].eq(0.75)]

        rows.append({
            "method": method,
            "target": target,
            "sampling_unit": unit,
            "m": PRIMARY_M,
            "K0": float(sub["K0"].mean()) if len(sub) else np.nan,
            "K1": float(sub["K1"].mean()) if len(sub) else np.nan,
            "null_rejection_rate": rate("iid_null"),
            "density_power": rate("same_support_density"),
            "topology_power": rate("topology_alt"),
            "translated_null_rejection_rate": rate("weak_barcode_null"),
            "overlap_rejection_rate": float(overlap.iloc[0]["rejection_rate"]) if len(overlap) else np.nan,
            "runtime": float(sub["runtime_seconds"].mean()) if len(sub) else np.nan,
            "peak_memory": float(sub["peak_rss_bytes"].mean()) if len(sub) else np.nan,
        })
    return pd.DataFrame(rows)


def _plot(summary: pd.DataFrame, output: str) -> None:
    mpl_config = os.path.join("/tmp", "tda2s_phase5ab_mplconfig")
    os.makedirs(mpl_config, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_config)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    selected = ["SC-A", "SC-B", "RawBlockMMD", "HybridBlockMMD-a0.50", "SC-A-Block"]
    colors = {
        "SC-A": "#4477AA", "SC-B": "#228833", "RawBlockMMD": "#CC6677",
        "HybridBlockMMD-a0.50": "#AA3377", "SC-A-Block": "#117733",
    }
    validity = summary[(summary["family"].isin(["iid_null", "weak_barcode_null"]))
                       & summary["m"].eq(PRIMARY_M)]
    ax = axes[0]
    for method in selected:
        sub = validity[validity["candidate"] == method]
        if sub.empty:
            continue
        sub = sub.sort_values("family")
        x = np.arange(len(sub))
        ax.errorbar(x, sub["rejection_rate"],
                    yerr=[sub["rejection_rate"] - sub["mc_low"],
                          sub["mc_high"] - sub["rejection_rate"]],
                    marker="o", capsize=3, label=method, color=colors.get(method))
    ax.axhspan(*SIZE_BAND, color="#66AA55", alpha=0.12)
    ax.axhline(ALPHA, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks([0, 1], ["iid null", "translated weak null"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("rejection rate")
    ax.set_title("Target distinction and null calibration")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)

    ax = axes[1]
    sens = summary[(summary["candidate"] == "RawBlockMMD")
                   & summary["family"].isin(["iid_null", "same_support_density", "topology_alt"])]
    for family, group in sens.groupby("family"):
        group = group.sort_values("m")
        ax.plot(group["m"], group["rejection_rate"], marker="o", label=family)
    ax.axhline(ALPHA, color="black", linestyle="--", linewidth=0.8)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("block size m, primary m=25")
    ax.set_ylabel("rejection rate")
    ax.set_title("RawBlockMMD block-size sensitivity")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_report(summary: pd.DataFrame, comparison: pd.DataFrame, output: str,
                  *, replications: int, n_permutations: int) -> None:
    def cell(method, family):
        row = comparison[comparison["method"] == method]
        if row.empty:
            return "n/a"
        col = {
            "iid_null": "null_rejection_rate",
            "weak_barcode_null": "translated_null_rejection_rate",
            "same_support_density": "density_power",
            "topology_alt": "topology_power",
        }.get(family)
        return "n/a" if col not in row else f"{float(row.iloc[0][col]):.3f}"

    lines = [
        "# Phase 5AB block-method evaluation",
        "",
        f"This run used `{replications}` replications per cell and `{n_permutations}` block-label permutations. It reuses the Phase 5C DGP constructors and stable seed convention.",
        "",
        "## Design audit conclusion",
        "",
        "`RawBlockMMD`, `HybridBlockMMD` with alpha greater than zero, and `SC-A-Block` declare `H0^law: P0=P1`. Their sampling unit is a frozen disjoint block, and the effective sample size is `K0+K1`. `SC-B` remains a barcode-law test. Hybrid alpha equal to zero is reported only as a barcode-law diagnostic.",
        "",
        "The raw primary kernel is Gaussian on the characteristic point-kernel mean embedding of the unordered bag. Equality of its block embeddings identifies the iid block law and then the point law. The simple average pairwise kernel is not characteristic for unrestricted bag distributions, so it is not used as the primary identified kernel.",
        "",
        "## Required exact DGP validation",
        "",
        "For the existing four-atom square witness, the m=2 barcode-law total variation is `0.27`, and the m=25 all-atoms-seen lower bound is `0.201160`. These validate the DGP distinction and are not properties of the raw block method.",
        "",
        "## Primary comparison",
        "",
        "| method | target | sampling_unit | m | K0 | K1 | null_rejection_rate | density_power | topology_power | translated_null_rejection_rate | overlap_rejection_rate | runtime | peak_memory |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in comparison.iterrows():
        k0 = "n/a" if pd.isna(row["K0"]) else f"{row['K0']:.1f}"
        k1 = "n/a" if pd.isna(row["K1"]) else f"{row['K1']:.1f}"
        overlap = "n/a" if pd.isna(row["overlap_rejection_rate"]) else f"{row['overlap_rejection_rate']:.3f}"
        lines.append(
            f"| {row['method']} | {row['target']} | {row['sampling_unit']} | {int(row['m'])} | "
            f"{k0} | {k1} | {row['null_rejection_rate']:.3f} | "
            f"{row['density_power']:.3f} | {row['topology_power']:.3f} | "
            f"{row['translated_null_rejection_rate']:.3f} | {overlap} | "
            f"{row['runtime']:.3f} | {row['peak_memory']:.0f}"
        )
    lines.extend([
        "",
        "## Gate interpretation",
        "",
        f"Gate 1, target validity, passes by construction for the raw and alpha-positive hybrid candidates. Gate 2 requires the iid-null rate to lie in the predeclared Phase 5 band `{SIZE_BAND}`; the values above are empirical and require their Monte Carlo intervals in the summary parquet. Gate 3 is a comparison claim, not an assumption: density power must be judged against SC-A with intervals. Gate 4 asks whether the hybrid retains topology power. Gate 5 requires the translated cell to distinguish raw point-law sensitivity from SC-B barcode-law calibration. Gate 6 rejects overlap as a confirmatory construction.",
        "",
        "The primary m=25 effective counts are `floor(n_a/25)`, and unused remainders are stored in the replication parquet. The m-sensitivity panel includes m in {1, 2, 5, 10, 25, 50}; m=1 is the point-level raw-block baseline. No post-hoc m or alpha choice is used for the headline comparison.",
        "",
        "Claims about finite-sample permutation validity, disjoint-block independence under iid sampling, ordering invariance, and point-law identification of the primary raw kernel are mathematical or directly checkable. Relative power, runtime, memory, and robustness are empirical. Overlapping blocks, dependent point processes, and label-dependent tuning remain outside the validity claim.",
        "",
    ])
    four_atom = summary[(summary["family"] == "same_square_four_atom_density")
                        & summary["m"].eq(PRIMARY_M)
                        & summary["overlap_fraction"].eq(0.0)]
    if not four_atom.empty:
        lines.extend([
            "## Four-atom square density cell",
            "",
            "The evaluated discrete cell uses the existing square support with p=(0.25,0.25,0.25,0.25) and q=(0.70,0.10,0.10,0.10). The exact m=2 barcode-law TV is 0.27, and the m=25 all-atoms-seen lower bound is 0.201160. The observed m=25 rejection rates are:",
            "",
            "| method | rejection rate | 95% MC interval |",
            "|---|---:|---:|",
        ])
        for _, row in four_atom.iterrows():
            lines.append(
                f"| {row['candidate']} | {row['rejection_rate']:.3f} | "
                f"[{row['mc_low']:.3f}, {row['mc_high']:.3f}] |"
            )
        lines.append("")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def run_and_write(*, families: Sequence[str] = DEFAULT_FAMILIES,
                  n_grid: Sequence[int] = (250,), m_values: Sequence[int] = (PRIMARY_M,),
                  replications: int = DEFAULT_REPLICATIONS,
                  n_permutations: int = DEFAULT_PERMUTATIONS, workers: int = DEFAULT_WORKERS,
                  candidates: Sequence[str] = PRIMARY_CANDIDATES,
                  include_overlap: bool = True,
                  replication_output: str = DEFAULT_REPLICATION_OUTPUT,
                  summary_output: str = DEFAULT_SUMMARY_OUTPUT,
                  comparison_output: str = DEFAULT_COMPARISON_OUTPUT,
                  figure: str = DEFAULT_FIGURE, report: str = DEFAULT_REPORT) -> dict:
    frame = run_replicates(
        families=families, n_grid=n_grid, m_values=m_values,
        replications=replications, n_permutations=n_permutations,
        workers=workers, candidates=candidates,
    )
    if include_overlap and PRIMARY_M in tuple(int(value) for value in m_values):
        overlap = run_overlap_replicates(
            replications=replications, n=max(n_grid), m=PRIMARY_M,
            n_permutations=n_permutations,
        )
        frame = pd.concat([frame, overlap], ignore_index=True)
    summary = _summary(frame)
    comparison = _comparison(summary)
    for path in (replication_output, summary_output, comparison_output):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    frame.to_parquet(replication_output, index=False)
    summary.to_parquet(summary_output, index=False)
    comparison.to_parquet(comparison_output, index=False)
    _plot(summary, figure)
    _write_report(summary, comparison, report, replications=replications,
                  n_permutations=n_permutations)
    return {
        "replications": int(len(frame)),
        "summary_rows": int(len(summary)),
        "comparison_rows": int(len(comparison)),
        "replication_output": replication_output,
        "summary_output": summary_output,
        "comparison_output": comparison_output,
        "figure": figure,
        "report": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=DEFAULT_REPLICATIONS)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--families", default=None)
    parser.add_argument("--n-grid", default="250")
    parser.add_argument("--m-grid", default="25")
    parser.add_argument("--replication-output", default=DEFAULT_REPLICATION_OUTPUT)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--comparison-output", default=DEFAULT_COMPARISON_OUTPUT)
    parser.add_argument("--figure", default=DEFAULT_FIGURE)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--candidates", default=",".join(PRIMARY_CANDIDATES))
    parser.add_argument("--no-overlap", action="store_true")
    args = parser.parse_args()
    result = run_and_write(
        families=_families(args.families),
        n_grid=tuple(int(v) for v in args.n_grid.split(",")),
        m_values=tuple(int(v) for v in args.m_grid.split(",")),
        replications=args.replications, n_permutations=args.permutations,
        workers=args.workers, include_overlap=not args.no_overlap,
        candidates=tuple(v.strip() for v in args.candidates.split(",") if v.strip()),
        replication_output=args.replication_output,
        summary_output=args.summary_output, comparison_output=args.comparison_output,
        figure=args.figure, report=args.report,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
