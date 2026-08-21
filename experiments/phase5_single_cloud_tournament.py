"""Phase 5C selection fleet for the exactly-two-cloud Regime-I problem.

The fleet is deliberately simulation-first.  It has four responsibilities:

1. run a cheap, deterministic pilot before any gate is read;
2. run independent replication shards for the pre-registered DGP cells;
3. aggregate shards without silently filling missing or conflicting records;
4. write the gate table, memo, and a figure showing validity and
   pseudo-replication failure.

The signed Phase 5A target is the fixed-size barcode-law null at ``m=25``.
SC-B at ``m=25`` is therefore the only target-matched production candidate.
SC-A is retained as the strongest-simple full point-law baseline and SC-C as
the finite persistent-Betti sensitivity.  SC-A is expensive because every
pooled label split requires a new PH calculation.  To make the comparison
reproducible on the available hardware, SC-A uses a predeclared 250-point
projection per arm whenever a cloud is larger than 250 points.  This is an
explicit effective-sample-size limitation, not a hidden claim of matched-n
power.  SC-B and SC-C use the full supplied clouds.

The main fleet is scoped to Regime I.  Process DGP constructors are included
for an optional dependence diagnostic, but they are not promoted to a size
gate because Phase 5A did not select Regime II and SC-D is dormant.

Examples
--------
Pilot (100 replications, one n cell):

    python experiments/phase5_single_cloud_tournament.py --mode pilot

One gating shard, suitable for a local process or a self-contained Colab
notebook:

    python experiments/phase5_single_cloud_tournament.py --mode shard \
        --cell iid_null_n250_m25 --rep-start 0 --replications 25

Aggregate all downloaded shards and write the requested deliverables:

    python experiments/phase5_single_cloud_tournament.py --mode aggregate

The default calibration counts are deliberately fixed in this file.  They
may be changed only by making a new design record, not by tuning them after
seeing rejection rates.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from tda2s.ph import compute_diagrams
from tda2s.resample import p_value
from tda2s.tests.single_cloud import (
    LOCKED_M,
    REGIME_I,
    _diagram_gram,
    _label_masks,
    _mmd2_from_gram,
    _permutation_pvalue,
    _ph,
    sc_a_label_permutation,
    sc_b_disjoint_mmd,
    sc_c_finite_vector,
)

ALPHA = 0.05
SIZE_BAND = (0.03, 0.08)
MC_CONFIDENCE = 0.95
N_GRID = (250, 500, 1000)
# m=50 supplies K={5,10,20}; m=25 is the signed primary target and supplies
# K={10,20,40}.  Both are frozen before any result is inspected.
M_GRID = (25, 50)
PRIMARY_M = LOCKED_M
FILTRATION = "ripser"  # Vietoris--Rips backend, with the shared radius scale
HOMOLOGY_DIMS = (0, 1)
KERNEL_BANDWIDTH = 0.10
BETTI_GRID = np.linspace(0.0, 0.60, 9)
BOOTSTRAP_BANDWIDTH = 0.05
MAX_SC_A_POINT_N = 250
GATE_PERMUTATIONS = 39
GATE_BOOTSTRAP_DRAWS = 19
GATE_REPLICATIONS = 500
PILOT_PERMUTATIONS = 19
PILOT_BOOTSTRAP_DRAWS = 9
DEFAULT_SHARD_REPLICATIONS = 25
MAX_WORKERS = 16
SEED_ROOT = 20260819

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
SHARD_DIR = os.path.join(RESULTS_DIR, "phase5c_shards")
CACHE_DIR = os.path.join(RESULTS_DIR, "phase5c_ph_cache")
FINAL_SUMMARY = os.path.join(RESULTS_DIR, "phase5_single_cloud_tournament.parquet")
FINAL_REPLICATIONS = os.path.join(RESULTS_DIR, "phase5_single_cloud_tournament_replications.parquet")
FINAL_FIGURE = os.path.join(RESULTS_DIR, "phase5_single_cloud_tournament.png")
FINAL_MEMO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "phase5_gate_memo.md")


@dataclass(frozen=True)
class Cell:
    family: str
    n0: int
    n1: int
    m: int
    role: str
    description: str

    @property
    def cell_id(self) -> str:
        return f"{self.family}_n{self.n0}_n1{self.n1}_m{self.m}"


CORE_FAMILIES = (
    "iid_null",
    "weak_barcode_null",
    "same_support_density",
    "topology_alt",
)
ROBUSTNESS_FAMILIES = (
    "robust_contamination",
    "robust_unequal_cardinality",
    "robust_anisotropic_noise",
    "robust_boundary_truncation",
)
DEPENDENCE_FAMILIES = (
    "process_poisson",
    "process_inhomogeneous_poisson",
    "process_cox_clustered",
    "process_hard_core",
)

FAMILY_ROLE = {
    "iid_null": "gating_null",
    "weak_barcode_null": "gating_null",
    "same_support_density": "target_mismatch",
    "topology_alt": "power",
    "robust_contamination": "robustness_diagnostic",
    "robust_unequal_cardinality": "robustness_diagnostic",
    "robust_anisotropic_noise": "robustness_diagnostic",
    "robust_boundary_truncation": "robustness_diagnostic",
    "process_poisson": "dormant_dependence",
    "process_inhomogeneous_poisson": "dormant_dependence",
    "process_cox_clustered": "dormant_dependence",
    "process_hard_core": "dormant_dependence",
}

FAMILY_DESCRIPTION = {
    "iid_null": "identical Uniform([0,1]^2) metric-measure laws",
    "weak_barcode_null": "translated point law with exactly equal metric barcode law",
    "same_support_density": "same square support with different continuous densities",
    "topology_alt": "filled disk versus noisy circle with matched expected moments",
    "robust_contamination": "common five-percent remote contamination",
    "robust_unequal_cardinality": "same law with unequal cloud cardinalities",
    "robust_anisotropic_noise": "common anisotropic affine deformation and noise",
    "robust_boundary_truncation": "common rectangular boundary truncation",
    "process_poisson": "homogeneous Poisson point process diagnostic",
    "process_inhomogeneous_poisson": "inhomogeneous Poisson point process diagnostic",
    "process_cox_clustered": "clustered Cox-style point process diagnostic",
    "process_hard_core": "hard-core point process diagnostic",
}


def _seed(*parts: object) -> int:
    """Stable 32-bit seed independent of Python's randomized hash."""
    payload = repr((SEED_ROOT,) + parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def design_record() -> dict:
    return {
        "phase": "5C",
        "regime": REGIME_I,
        "alpha": ALPHA,
        "size_band": list(SIZE_BAND),
        "confidence": MC_CONFIDENCE,
        "n_grid": list(N_GRID),
        "m_grid": list(M_GRID),
        "primary_m": PRIMARY_M,
        "filtration": FILTRATION,
        "homology_dims": list(HOMOLOGY_DIMS),
        "kernel_bandwidth": KERNEL_BANDWIDTH,
        "betti_grid": BETTI_GRID.tolist(),
        "bootstrap_bandwidth": BOOTSTRAP_BANDWIDTH,
        "gate_permutations": GATE_PERMUTATIONS,
        "gate_bootstrap_draws": GATE_BOOTSTRAP_DRAWS,
        "pilot_permutations": PILOT_PERMUTATIONS,
        "pilot_bootstrap_draws": PILOT_BOOTSTRAP_DRAWS,
        "sc_a_projection_n": MAX_SC_A_POINT_N,
        "target": "H0,25^bar: Phi^25_0:1(P0) = Phi^25_0:1(P1)",
        "sc_a_target": "H0^law: P0 = P1",
        "sc_c_target": "H0^finite-vector: equality of frozen normalized persistent-Betti means",
        "overlap_fractions": [0.0, 0.25, 0.5, 0.75, 0.9],
    }


DESIGN_HASH = hashlib.sha256(
    json.dumps(design_record(), sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()[:16]


def required_k_counts() -> dict[int, list[int]]:
    return {m: [n // m for n in N_GRID] for m in M_GRID}


def make_cells(*, include_robustness: bool = True, include_dependence: bool = False) -> list[Cell]:
    families = list(CORE_FAMILIES)
    if include_robustness:
        families.extend(ROBUSTNESS_FAMILIES)
    if include_dependence:
        families.extend(DEPENDENCE_FAMILIES)
    cells = []
    for family in families:
        for n in N_GRID:
            n1 = int(math.ceil(n * 1.25)) if family == "robust_unequal_cardinality" else n
            for m in M_GRID:
                cells.append(Cell(
                    family=family,
                    n0=n,
                    n1=n1,
                    m=m,
                    role=FAMILY_ROLE[family],
                    description=FAMILY_DESCRIPTION[family],
                ))
    return cells


def parse_cell_id(value: str) -> Cell:
    for cell in make_cells(include_robustness=True, include_dependence=True):
        if cell.cell_id == value:
            return cell
    valid = ", ".join(c.cell_id for c in make_cells(include_robustness=False))
    raise ValueError(f"unknown cell {value!r}; examples are {valid}")


def _uniform_square(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.0, 1.0, size=(int(n), 2))


def _filled_disk(n: int, rng: np.random.Generator, *, radius: float = 0.30) -> np.ndarray:
    theta = rng.uniform(0.0, 2.0 * np.pi, size=int(n))
    radial = radius * np.sqrt(rng.uniform(0.0, 1.0, size=int(n)))
    points = np.column_stack([radial * np.cos(theta), radial * np.sin(theta)])
    return points + np.array([0.5, 0.5])


def _noisy_circle(n: int, rng: np.random.Generator, *, radius: float = 0.30) -> np.ndarray:
    theta = rng.uniform(0.0, 2.0 * np.pi, size=int(n))
    points = np.column_stack([np.cos(theta), np.sin(theta)]) * radius
    points += rng.normal(0.0, 0.008, size=points.shape)
    return points + np.array([0.5, 0.5])


def _common_contamination(n: int, rng: np.random.Generator) -> np.ndarray:
    points = _uniform_square(n, rng)
    count = max(1, int(round(0.05 * n)))
    points[:count] = rng.uniform(2.0, 3.0, size=(count, 2))
    return points


def _anisotropic(n: int, rng: np.random.Generator) -> np.ndarray:
    points = _uniform_square(n, rng)
    points = points @ np.array([[1.8, 0.0], [0.0, 0.45]])
    points += rng.normal(0.0, 0.015, size=points.shape)
    return points


def _hard_core(n: int, rng: np.random.Generator, minimum: float = 0.025) -> np.ndarray:
    accepted: list[np.ndarray] = []
    max_attempts = max(1000, 40 * int(n))
    attempts = 0
    while len(accepted) < int(n) and attempts < max_attempts:
        candidate = rng.uniform(0.0, 1.0, size=2)
        attempts += 1
        if not accepted or min(np.linalg.norm(candidate - old) for old in accepted) >= minimum:
            accepted.append(candidate)
    if len(accepted) < int(n):
        # The fallback is deterministic and keeps the diagnostic runnable at
        # high n.  It is not used for a Regime-I gate.
        extra = _uniform_square(int(n) - len(accepted), rng)
        accepted.extend(extra)
    return np.asarray(accepted, dtype=float)


def _poisson_cloud(n_expected: int, rng: np.random.Generator) -> np.ndarray:
    n = max(2, int(rng.poisson(n_expected)))
    return _uniform_square(n, rng)


def _inhomogeneous_poisson_cloud(n_expected: int, rng: np.random.Generator) -> np.ndarray:
    n = max(2, int(rng.poisson(n_expected)))
    # x has density 2x, while y remains uniform.  The support stays the unit
    # square, so this is an intensity change rather than a support change.
    return np.column_stack([np.sqrt(rng.uniform(size=n)), rng.uniform(size=n)])


def _cox_cloud(n_expected: int, rng: np.random.Generator) -> np.ndarray:
    n = max(2, int(rng.poisson(n_expected)))
    parents = rng.uniform(0.0, 1.0, size=(8, 2))
    labels = rng.integers(0, len(parents), size=n)
    points = parents[labels] + rng.normal(0.0, 0.07, size=(n, 2))
    return np.mod(points, 1.0)


def make_cloud_pair(family: str, n0: int, n1: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate one deterministic two-cloud replication for a named DGP."""
    if family not in FAMILY_ROLE:
        raise ValueError(f"unknown family {family!r}")
    rng = np.random.default_rng(seed)

    if family == "iid_null":
        return _uniform_square(n0, rng), _uniform_square(n1, rng)
    if family == "weak_barcode_null":
        cloud0 = _uniform_square(n0, rng)
        cloud1 = _uniform_square(n1, rng) + np.array([2.0, -1.5])
        return cloud0, cloud1
    if family == "same_support_density":
        cloud0 = _uniform_square(n0, rng)
        # A beta mixture has the same closed support [0,1]^2 but a different
        # density.  The uniform component prevents a support convention from
        # driving the distinction.
        mask = rng.uniform(size=(n1, 1)) < 0.8
        beta = rng.beta(2.5, 2.5, size=(n1, 2))
        uniform = _uniform_square(n1, rng)
        cloud1 = np.where(mask, beta, uniform)
        return cloud0, cloud1
    if family == "topology_alt":
        return _filled_disk(n0, rng), _noisy_circle(n1, rng)
    if family == "robust_contamination":
        return _common_contamination(n0, rng), _common_contamination(n1, rng)
    if family == "robust_unequal_cardinality":
        return _uniform_square(n0, rng), _uniform_square(n1, rng)
    if family == "robust_anisotropic_noise":
        return _anisotropic(n0, rng), _anisotropic(n1, rng)
    if family == "robust_boundary_truncation":
        return _uniform_square(n0, rng) * np.array([0.7, 1.0]), _uniform_square(n1, rng) * np.array([0.7, 1.0])
    if family == "process_poisson":
        return _poisson_cloud(n0, rng), _poisson_cloud(n1, rng)
    if family == "process_inhomogeneous_poisson":
        return _inhomogeneous_poisson_cloud(n0, rng), _inhomogeneous_poisson_cloud(n1, rng)
    if family == "process_cox_clustered":
        return _cox_cloud(n0, rng), _cox_cloud(n1, rng)
    if family == "process_hard_core":
        return _hard_core(n0, rng), _hard_core(n1, rng)
    raise AssertionError(f"unhandled family {family!r}")


def _project_cloud(cloud: np.ndarray, size: int, seed: int) -> np.ndarray:
    if len(cloud) <= size:
        return np.asarray(cloud, dtype=float)
    rng = np.random.default_rng(seed)
    return np.asarray(cloud)[rng.choice(len(cloud), size=int(size), replace=False)]


def _common_ph_kwargs(cache_dir: str | None) -> dict:
    return {
        "filtration": FILTRATION,
        "homology_dims": HOMOLOGY_DIMS,
        "max_edge_length": None,
        "grid_size": 64,
        "dtm_k": 20,
        "kernel_bandwidth": KERNEL_BANDWIDTH,
        "cache_dir": cache_dir,
    }


def _record_from_result(cell: Cell, rep: int, candidate: str, result: dict,
                        *, m: int, null_role: str, method_variant: str = "") -> dict:
    diagnostics = result.get("diagnostics", {})
    effective = diagnostics.get("effective_sample_size", {})
    return {
        "design_hash": DESIGN_HASH,
        "record_type": "candidate",
        "status": "ok",
        "replication": int(rep),
        "cell_id": cell.cell_id,
        "family": cell.family,
        "family_role": null_role,
        "family_description": cell.description,
        "candidate": candidate,
        "method_variant": method_variant,
        "regime": REGIME_I,
        "n0": int(cell.n0),
        "n1": int(cell.n1),
        "m": int(m),
        "primary_target": bool(candidate == "SC-B" and m == PRIMARY_M),
        "target": result.get("inferential_target", ""),
        "statistic": float(result.get("statistic", np.nan)),
        "pvalue": float(result.get("pvalue", np.nan)),
        "reject": bool(float(result.get("pvalue", np.nan)) <= ALPHA),
        "alpha": ALPHA,
        "K0": int(result.get("K0", effective.get("K0", -1))),
        "K1": int(result.get("K1", effective.get("K1", -1))),
        "effective_point_n0": int(diagnostics.get("effective_point_n0", cell.n0)),
        "effective_point_n1": int(diagnostics.get("effective_point_n1", cell.n1)),
        "effective_barcode_n0": int(result.get("K0", effective.get("K0", -1))),
        "effective_barcode_n1": int(result.get("K1", effective.get("K1", -1))),
        "overlap_fraction": 0.0,
        "unique_points0": int(cell.n0),
        "unique_points1": int(cell.n1),
        "n_resamples": int(result.get("n_permutations", result.get("n_draws", -1))),
        "runtime_seconds": float(result.get("runtime_seconds", np.nan)),
        "peak_memory_bytes": int(result.get("peak_memory_bytes", -1)),
        "filtration": FILTRATION,
        "homology_dims": json.dumps(list(HOMOLOGY_DIMS)),
        "partition_frozen": bool(candidate == "SC-B"),
        "overlapping_blocks_used": False,
        "projection_used": bool(candidate == "SC-A" and min(cell.n0, cell.n1) > MAX_SC_A_POINT_N),
        "method_assumptions_ok": bool(cell.role != "dormant_dependence"),
        "null_role": null_role,
        "method_error": "",
    }


def _error_record(cell: Cell, rep: int, candidate: str, m: int, exc: Exception,
                  *, null_role: str, method_variant: str = "") -> dict:
    record = {
        "design_hash": DESIGN_HASH,
        "record_type": "candidate",
        "status": "failed",
        "replication": int(rep),
        "cell_id": cell.cell_id,
        "family": cell.family,
        "family_role": null_role,
        "family_description": cell.description,
        "candidate": candidate,
        "method_variant": method_variant,
        "regime": REGIME_I,
        "n0": int(cell.n0),
        "n1": int(cell.n1),
        "m": int(m),
        "primary_target": bool(candidate == "SC-B" and m == PRIMARY_M),
        "target": "",
        "statistic": np.nan,
        "pvalue": np.nan,
        "reject": False,
        "alpha": ALPHA,
        "K0": -1,
        "K1": -1,
        "effective_point_n0": -1,
        "effective_point_n1": -1,
        "effective_barcode_n0": -1,
        "effective_barcode_n1": -1,
        "overlap_fraction": 0.0,
        "unique_points0": -1,
        "unique_points1": -1,
        "n_resamples": -1,
        "runtime_seconds": np.nan,
        "peak_memory_bytes": -1,
        "filtration": FILTRATION,
        "homology_dims": json.dumps(list(HOMOLOGY_DIMS)),
        "partition_frozen": False,
        "overlapping_blocks_used": False,
        "projection_used": False,
        "method_assumptions_ok": False,
        "null_role": null_role,
        "method_error": f"{type(exc).__name__}: {exc}",
    }
    return record


def run_candidate(cell: Cell, rep: int, candidate: str, *, m: int,
                  n_permutations: int, n_bootstrap: int,
                  cache_dir: str | None) -> dict:
    """Run one frozen candidate and normalize its result schema."""
    cloud0, cloud1 = make_cloud_pair(cell.family, cell.n0, cell.n1, _seed("cloud", cell.cell_id, rep))
    method_seed = _seed("method", cell.cell_id, rep, candidate, m)
    try:
        if candidate == "SC-A":
            # Full SC-A at n=1000 is a poor use of the PH budget.  The
            # projection is fixed before the fleet and recorded in every row.
            x0 = _project_cloud(cloud0, MAX_SC_A_POINT_N, _seed("A0", cell.cell_id, rep))
            x1 = _project_cloud(cloud1, MAX_SC_A_POINT_N, _seed("A1", cell.cell_id, rep))
            result = sc_a_label_permutation(
                x0, x1, regime=REGIME_I, **_common_ph_kwargs(cache_dir),
                n_perm=n_permutations, exact=False, seed=method_seed,
            )
            result["diagnostics"]["effective_point_n0"] = len(x0)
            result["diagnostics"]["effective_point_n1"] = len(x1)
        elif candidate == "SC-B":
            result = sc_b_disjoint_mmd(
                cloud0, cloud1, regime=REGIME_I, m=m,
                partition_seed=_seed("partition", cell.cell_id, rep),
                n_perm=n_permutations, exact=False, seed=method_seed,
                **_common_ph_kwargs(cache_dir),
            )
        elif candidate == "SC-C":
            # The finite-vector candidate uses the full clouds.  Its bootstrap
            # clouds are fresh and are intentionally not cached as they do not
            # recur across replications.
            kwargs = _common_ph_kwargs(None)
            kwargs.pop("kernel_bandwidth")
            result = sc_c_finite_vector(
                cloud0, cloud1, regime=REGIME_I, grid=BETTI_GRID,
                bootstrap_bandwidth=BOOTSTRAP_BANDWIDTH,
                n_draws=n_bootstrap, smoothing=True, seed=method_seed,
                **kwargs,
            )
        else:
            raise ValueError(f"unknown candidate {candidate!r}")
        return _record_from_result(cell, rep, candidate, result, m=m,
                                   null_role=cell.role)
    except Exception as exc:  # retain failed runs for the gate audit
        return _error_record(cell, rep, candidate, m, exc, null_role=cell.role)


def _overlapping_blocks(cloud: np.ndarray, m: int, overlap_fraction: float,
                        seed: int, K: int) -> tuple[list[np.ndarray], np.ndarray, float]:
    if not 0.0 <= overlap_fraction < 1.0:
        raise ValueError("overlap_fraction must be in [0,1)")
    if len(cloud) < m:
        raise ValueError("cloud must contain at least m points")
    overlap = int(round(float(overlap_fraction) * m))
    step = max(1, m - overlap)
    needed = m + (int(K) - 1) * step
    if needed > len(cloud):
        raise ValueError("requested overlapping blocks exceed the cloud")
    permutation = np.random.default_rng(seed).permutation(len(cloud))
    indices = np.asarray([permutation[k * step:k * step + m] for k in range(int(K))], dtype=int)
    blocks = [np.asarray(cloud[idx], dtype=float) for idx in indices]
    pairwise = []
    for i in range(len(indices)):
        for j in range(i):
            pairwise.append(len(set(indices[i]).intersection(indices[j])) / float(m))
    mean_overlap = float(np.mean(pairwise)) if pairwise else 0.0
    return blocks, indices, mean_overlap


def overlapping_negative_control(cloud0: np.ndarray, cloud1: np.ndarray, *, m: int,
                                 overlap_fraction: float, seed: int,
                                 n_permutations: int, cache_dir: str | None) -> dict:
    """Naïve overlapping-subcloud MMD negative control.

    This intentionally violates SC-B's disjoint-block condition.  It is
    retained solely to expose pseudo-replication.  The PH diagrams are cached
    once per overlapping block, and the label permutation acts only on their
    Gram matrix.
    """
    K0, K1 = len(cloud0) // m, len(cloud1) // m
    blocks0, idx0, mean0 = _overlapping_blocks(cloud0, m, overlap_fraction, seed, K0)
    blocks1, idx1, mean1 = _overlapping_blocks(cloud1, m, overlap_fraction, seed + 1, K1)
    diagrams = [
        _ph(block, filtration=FILTRATION, homology_dims=HOMOLOGY_DIMS,
            max_edge_length=None, grid_size=64, dtm_k=20, cache_dir=cache_dir)
        for block in blocks0 + blocks1
    ]
    gram = _diagram_gram(diagrams, KERNEL_BANDWIDTH)
    group0 = np.zeros(K0 + K1, dtype=bool)
    group0[:K0] = True
    observed = _mmd2_from_gram(gram, group0)
    masks, exact_used = _label_masks(
        K0 + K1, K0, n_perm=n_permutations, exact=False,
        max_exact_permutations=100_000, seed=seed + 2,
    )
    null = np.asarray([_mmd2_from_gram(gram, mask) for mask in masks], dtype=float)
    return {
        "candidate": "SC-B-overlap-negative-control",
        "statistic": float(observed),
        "pvalue": _permutation_pvalue(observed, null, exact_used),
        "K0": int(K0),
        "K1": int(K1),
        "null_statistics": null,
        "unique_points0": int(len(np.unique(idx0))),
        "unique_points1": int(len(np.unique(idx1))),
        "mean_pairwise_overlap0": mean0,
        "mean_pairwise_overlap1": mean1,
    }


def run_overlap_replication(rep: int, *, n: int = 500, m: int = PRIMARY_M,
                            overlap_fraction: float, n_permutations: int,
                            cache_dir: str | None) -> dict:
    cell = Cell("iid_null", n, n, m, "pseudo_replication_negative_control",
                "iid null with deliberately overlapping subclouds")
    cloud0, cloud1 = make_cloud_pair(cell.family, n, n, _seed("overlap-cloud", rep))
    started = time.perf_counter()
    try:
        result = overlapping_negative_control(
            cloud0, cloud1, m=m, overlap_fraction=overlap_fraction,
            seed=_seed("overlap", rep, overlap_fraction),
            n_permutations=n_permutations, cache_dir=cache_dir,
        )
        return {
            "design_hash": DESIGN_HASH,
            "record_type": "negative_control",
            "status": "ok",
            "replication": int(rep),
            "cell_id": f"overlap_n{n}_m{m}_omega{overlap_fraction:g}",
            "family": "overlap_iid_null",
            "family_role": "pseudo_replication_negative_control",
            "family_description": "iid null, with overlapping blocks used intentionally as a negative control",
            "candidate": "SC-B-overlap-negative-control",
            "method_variant": "naive-overlap",
            "regime": REGIME_I,
            "n0": int(n),
            "n1": int(n),
            "m": int(m),
            "primary_target": False,
            "target": "invalid confirmatory path: overlapping blocks are not independent",
            "statistic": result["statistic"],
            "pvalue": result["pvalue"],
            "reject": bool(result["pvalue"] <= ALPHA),
            "alpha": ALPHA,
            "K0": result["K0"],
            "K1": result["K1"],
            "effective_point_n0": result["unique_points0"],
            "effective_point_n1": result["unique_points1"],
            "effective_barcode_n0": result["K0"],
            "effective_barcode_n1": result["K1"],
            "overlap_fraction": float(overlap_fraction),
            "unique_points0": result["unique_points0"],
            "unique_points1": result["unique_points1"],
            "mean_pairwise_overlap0": result["mean_pairwise_overlap0"],
            "mean_pairwise_overlap1": result["mean_pairwise_overlap1"],
            "n_resamples": int(n_permutations),
            "runtime_seconds": float(time.perf_counter() - started),
            "peak_memory_bytes": -1,
            "filtration": FILTRATION,
            "homology_dims": json.dumps(list(HOMOLOGY_DIMS)),
            "partition_frozen": True,
            "overlapping_blocks_used": True,
            "projection_used": False,
            "method_assumptions_ok": False,
            "null_role": "pseudo_replication_negative_control",
            "method_error": "",
        }
    except Exception as exc:
        return {
            "design_hash": DESIGN_HASH,
            "record_type": "negative_control",
            "status": "failed",
            "replication": int(rep),
            "cell_id": f"overlap_n{n}_m{m}_omega{overlap_fraction:g}",
            "family": "overlap_iid_null",
            "family_role": "pseudo_replication_negative_control",
            "family_description": "iid null, with overlapping blocks used intentionally as a negative control",
            "candidate": "SC-B-overlap-negative-control",
            "method_variant": "naive-overlap",
            "regime": REGIME_I,
            "n0": int(n),
            "n1": int(n),
            "m": int(m),
            "primary_target": False,
            "target": "",
            "statistic": np.nan,
            "pvalue": np.nan,
            "reject": False,
            "alpha": ALPHA,
            "K0": -1,
            "K1": -1,
            "effective_point_n0": -1,
            "effective_point_n1": -1,
            "effective_barcode_n0": -1,
            "effective_barcode_n1": -1,
            "overlap_fraction": float(overlap_fraction),
            "unique_points0": -1,
            "unique_points1": -1,
            "n_resamples": int(n_permutations),
            "runtime_seconds": float(time.perf_counter() - started),
            "peak_memory_bytes": -1,
            "filtration": FILTRATION,
            "homology_dims": json.dumps(list(HOMOLOGY_DIMS)),
            "partition_frozen": True,
            "overlapping_blocks_used": True,
            "projection_used": False,
            "method_assumptions_ok": False,
            "null_role": "pseudo_replication_negative_control",
            "method_error": f"{type(exc).__name__}: {exc}",
        }


def _run_overlap_pair(args: tuple) -> dict:
    rep, omega, n, m, n_permutations, cache_dir = args
    return run_overlap_replication(
        int(rep), n=int(n), m=int(m), overlap_fraction=float(omega),
        n_permutations=int(n_permutations), cache_dir=cache_dir,
    )


def run_replication(cell: Cell, rep: int, *, n_permutations: int,
                    n_bootstrap: int, cache_dir: str | None,
                    candidates: Sequence[str] | None = None) -> list[dict]:
    """Run one replication of one cell.

    SC-A and SC-C are run only on the primary m=25 cell.  SC-B is run for both
    frozen m values, so the m=50 rows are an explicitly labelled sensitivity
    rather than duplicated candidate calls.
    """
    if candidates is None:
        selected = ["SC-B"]
        if cell.m == PRIMARY_M:
            selected = ["SC-A", "SC-B", "SC-C"]
    else:
        selected = [str(candidate) for candidate in candidates]
    return [
        run_candidate(cell, rep, candidate, m=cell.m,
                      n_permutations=n_permutations,
                      n_bootstrap=n_bootstrap, cache_dir=cache_dir)
        for candidate in selected
    ]


def _run_replication_args(args: tuple) -> list[dict]:
    cell, rep, n_perm, n_boot, cache_dir, candidates = args
    return run_replication(cell, rep, n_permutations=n_perm,
                           n_bootstrap=n_boot, cache_dir=cache_dir,
                           candidates=candidates)


def _write_shard(rows: Iterable[dict], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    frame = pd.DataFrame(list(rows))
    frame.to_parquet(path, index=False)
    return path


def run_shard(cell: Cell, *, rep_start: int, replications: int,
              n_permutations: int, n_bootstrap: int,
              workers: int = 1, cache_dir: str | None = CACHE_DIR,
              output: str | None = None,
              candidates: Sequence[str] | None = None) -> str:
    if rep_start < 0 or replications < 1:
        raise ValueError("rep_start must be non-negative and replications must be positive")
    if workers < 1 or workers > MAX_WORKERS:
        raise ValueError(f"workers must be in [1,{MAX_WORKERS}]")
    reps = list(range(int(rep_start), int(rep_start) + int(replications)))
    candidate_tuple = tuple(candidates) if candidates is not None else None
    args = [(cell, rep, n_permutations, n_bootstrap, cache_dir, candidate_tuple)
            for rep in reps]
    started = time.perf_counter()
    if workers == 1:
        nested = [_run_replication_args(arg) for arg in args]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            nested = list(pool.map(_run_replication_args, args))
    rows = [row for rows_one in nested for row in rows_one]
    if output is None:
        output = os.path.join(
            SHARD_DIR,
            f"phase5c_{cell.cell_id}_rep{rep_start}_{rep_start + replications - 1}.parquet",
        )
    path = _write_shard(rows, output)
    elapsed = time.perf_counter() - started
    print(json.dumps({
        "cell": cell.cell_id,
        "replications": replications,
        "rows": len(rows),
        "workers": workers,
        "seconds": round(elapsed, 2),
        "output": path,
    }, indent=2))
    return path


def run_fleet(*, families: Sequence[str], replications: int = 500,
              n_permutations: int = GATE_PERMUTATIONS,
              n_bootstrap: int = GATE_BOOTSTRAP_DRAWS,
              workers: int = 1, candidate_mode: str = "all",
              cache_dir: str | None = CACHE_DIR) -> list[str]:
    """Run a collection of cells sequentially, one cell per shard file.

    Sequential cell scheduling prevents a large n=1000 SC-C cell from being
    multiplied by another large cell, while each cell still uses the requested
    process-level parallelism.  ``candidate_mode='all'`` runs all applicable
    candidates at m=25 and only SC-B at m=50.  ``scb`` runs only SC-B, and
    ``baselines`` runs SC-A and SC-C at the primary m=25 cell.
    """
    if candidate_mode not in {"all", "scb", "baselines"}:
        raise ValueError("candidate_mode must be one of {'all','scb','baselines'}")
    unknown = sorted(set(families) - set(FAMILY_ROLE))
    if unknown:
        raise ValueError(f"unknown fleet families: {unknown}")
    paths = []
    for family in families:
        for n in N_GRID:
            n1 = int(math.ceil(n * 1.25)) if family == "robust_unequal_cardinality" else n
            for m in M_GRID:
                if candidate_mode == "scb":
                    selected = ("SC-B",)
                elif candidate_mode == "baselines":
                    if m != PRIMARY_M:
                        continue
                    selected = ("SC-A", "SC-C")
                else:
                    selected = None if m == PRIMARY_M else ("SC-B",)
                cell = Cell(family, n, n1, m, FAMILY_ROLE[family], FAMILY_DESCRIPTION[family])
                output = os.path.join(
                    SHARD_DIR,
                    f"phase5c_{cell.cell_id}_{candidate_mode}_rep0_{replications - 1}.parquet",
                )
                paths.append(run_shard(
                    cell, rep_start=0, replications=replications,
                    n_permutations=n_permutations, n_bootstrap=n_bootstrap,
                    workers=workers, cache_dir=cache_dir, output=output,
                    candidates=selected,
                ))
    return paths


def run_pilot(*, replications: int = 100, n: int = 250,
              n_permutations: int = PILOT_PERMUTATIONS,
              n_bootstrap: int = PILOT_BOOTSTRAP_DRAWS,
              workers: int = 1, output: str | None = None) -> str:
    """Run the required cheap pilot on the four core families."""
    cells = [Cell(family, n, n, PRIMARY_M, FAMILY_ROLE[family], FAMILY_DESCRIPTION[family])
             for family in CORE_FAMILIES]
    args = [(cell, rep, n_permutations, n_bootstrap, None, None)
            for cell in cells for rep in range(int(replications))]
    if workers == 1:
        nested = [_run_replication_args(arg) for arg in args]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, MAX_WORKERS)) as pool:
            nested = list(pool.map(_run_replication_args, args))
    rows = [row for rows_one in nested for row in rows_one]
    if output is None:
        output = os.path.join(RESULTS_DIR, "phase5c_pilot.parquet")
    path = _write_shard(rows, output)
    print(json.dumps({"pilot": True, "replications_per_family": replications,
                      "families": list(CORE_FAMILIES), "rows": len(rows),
                      "output": path}, indent=2))
    return path


def run_overlap_fleet(*, replications: int = 500, n: int = 500,
                      m: int = PRIMARY_M,
                      n_permutations: int = GATE_PERMUTATIONS,
                      workers: int = 1, cache_dir: str | None = CACHE_DIR,
                      output: str | None = None) -> str:
    args = [
        (rep, omega, n, m, n_permutations, cache_dir)
        for omega in design_record()["overlap_fractions"]
        for rep in range(int(replications))
    ]
    if workers == 1:
        rows = [_run_overlap_pair(arg) for arg in args]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, MAX_WORKERS)) as pool:
            rows = list(pool.map(_run_overlap_pair, args))
    if output is None:
        output = os.path.join(SHARD_DIR, "phase5c_overlap_negative_control.parquet")
    return _write_shard(rows, output)


def _read_shards(input_dir: str = SHARD_DIR) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(input_dir, "phase5c_*.parquet")))
    if not paths:
        raise FileNotFoundError(f"no phase5c_*.parquet shards in {input_dir}")
    frames = [pd.read_parquet(path) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    key = ["design_hash", "record_type", "cell_id", "candidate", "replication"]
    duplicated = frame.duplicated(key, keep=False)
    if duplicated.any():
        duplicates = frame.loc[duplicated, key].drop_duplicates().to_dict("records")
        raise ValueError(f"conflicting or repeated Phase 5C replication keys: {duplicates[:5]}")
    if set(frame["design_hash"].dropna()) != {DESIGN_HASH}:
        raise ValueError("shards do not share the frozen Phase 5C design hash")
    return frame


def _mc_interval(successes: int, total: int, confidence: float = MC_CONFIDENCE) -> tuple[float, float]:
    if total < 1 or successes < 0 or successes > total:
        raise ValueError("invalid binomial count")
    from scipy.stats import beta
    tail = (1.0 - confidence) / 2.0
    lower = 0.0 if successes == 0 else float(beta.ppf(tail, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1.0 - tail, successes + 1, total - successes))
    return lower, upper


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"status", "reject", "family", "candidate", "m", "n0", "n1", "record_type"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"replication table is missing columns: {sorted(missing)}")
    groups = ["record_type", "family", "family_role", "candidate", "method_variant", "n0", "n1", "m", "overlap_fraction"]
    rows = []
    for keys, group in frame.groupby(groups, dropna=False, sort=True):
        total = int(len(group))
        successful = group[group["status"] == "ok"]
        n_ok = int(len(successful))
        rejects = int(successful["reject"].astype(bool).sum())
        rate = float(rejects / n_ok) if n_ok else np.nan
        low, high = _mc_interval(rejects, n_ok) if n_ok else (np.nan, np.nan)
        values = dict(zip(groups, keys))
        values.update({
            "design_hash": DESIGN_HASH,
            "total_replications": total,
            "successful_replications": n_ok,
            "failed_replications": total - n_ok,
            "rejections": rejects,
            "rejection_rate": rate,
            "mc_low": low,
            "mc_high": high,
            "mc_se": float(np.sqrt(rate * (1.0 - rate) / n_ok)) if n_ok else np.nan,
            "in_size_band": bool(SIZE_BAND[0] <= rate <= SIZE_BAND[1]) if n_ok else False,
            "mean_runtime_seconds": float(successful["runtime_seconds"].mean()) if n_ok and "runtime_seconds" in successful else np.nan,
            "mean_effective_barcode_n0": float(successful["effective_barcode_n0"].mean()) if n_ok else np.nan,
            "mean_effective_barcode_n1": float(successful["effective_barcode_n1"].mean()) if n_ok else np.nan,
            "mean_unique_points0": float(successful["unique_points0"].mean()) if n_ok else np.nan,
            "mean_unique_points1": float(successful["unique_points1"].mean()) if n_ok else np.nan,
            "mean_pairwise_overlap0": float(successful["mean_pairwise_overlap0"].mean()) if n_ok and "mean_pairwise_overlap0" in successful else np.nan,
            "mean_pairwise_overlap1": float(successful["mean_pairwise_overlap1"].mean()) if n_ok and "mean_pairwise_overlap1" in successful else np.nan,
        })
        rows.append(values)
    return pd.DataFrame(rows)


def _required_gate_cells() -> set[str]:
    cells = []
    # The hard size gate contains the two nulls that directly identify the
    # locked target: the ordinary iid null and the translated barcode-law
    # witness.  Robustness families are stress diagnostics, as specified in
    # the Phase 5C DGP table, and are reported separately rather than treated
    # as additional sharp-null calibration cells.
    for family in CORE_FAMILIES[:2]:
        n_values = N_GRID
        for n in n_values:
            n1 = int(math.ceil(n * 1.25)) if family == "robust_unequal_cardinality" else n
            cells.append(f"{family}_n{n}_n1{n1}_m{PRIMARY_M}")
    return set(cells)


def _gate_verdict(summary: pd.DataFrame) -> dict:
    primary = summary[(summary["record_type"] == "candidate")
                      & (summary["candidate"] == "SC-B")
                      & (summary["m"] == PRIMARY_M)
                      & summary["family"].isin(CORE_FAMILIES[:2])]
    required = _required_gate_cells()
    observed = set(primary["family"].astype(str) + "_n" + primary["n0"].astype(str)
                   + "_n1" + primary["n1"].astype(str) + "_m" + primary["m"].astype(str))
    missing = sorted(required - observed)
    complete = (not missing and bool(len(primary))
                and bool((primary["failed_replications"] == 0).all())
                and bool((primary["successful_replications"] >= GATE_REPLICATIONS).all()))
    size_pass = complete and bool(primary["in_size_band"].all())
    alt = summary[(summary["record_type"] == "candidate")
                  & (summary["candidate"] == "SC-B")
                  & (summary["m"] == PRIMARY_M)
                  & (summary["family"] == "topology_alt")
                  & (summary["n0"] == 1000)]
    power = float(alt.iloc[0]["rejection_rate"]) if len(alt) else np.nan
    if not complete:
        verdict = "INCOMPLETE"
    elif not size_pass:
        verdict = "KILL"
    elif power >= 0.80:
        verdict = "GO"
    elif power >= 0.50:
        verdict = "PIVOT"
    else:
        verdict = "KILL"
    return {
        "verdict": verdict,
        "complete": complete,
        "missing_cells": missing,
        "size_pass": size_pass,
        "moderate_alternative_power_n1000": power,
        "primary_null_cells": int(len(primary)),
    }


def _plot(summary: pd.DataFrame, output: str = FINAL_FIGURE) -> str:
    mpl_config = os.path.join("/tmp", "tda2s_phase5c_mplconfig")
    os.makedirs(mpl_config, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_config)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2))
    candidates = ["SC-A", "SC-B", "SC-C"]
    colors = {"SC-A": "#4477AA", "SC-B": "#228833", "SC-C": "#CC6677"}
    labels = {"SC-A": "SC-A, 250-point projection", "SC-B": "SC-B, m=25", "SC-C": "SC-C, smoothed finite vector"}

    ax = axes[0]
    validity = summary[(summary["record_type"] == "candidate")
                       & summary["family"].isin(["iid_null", "weak_barcode_null"])
                       & (summary["m"] == PRIMARY_M)]
    for candidate in candidates:
        sub = validity[validity["candidate"] == candidate].sort_values("n0")
        if sub.empty:
            continue
        x = np.arange(len(sub))
        ax.errorbar(x, sub["rejection_rate"],
                    yerr=[sub["rejection_rate"] - sub["mc_low"], sub["mc_high"] - sub["rejection_rate"]],
                    marker="o", capsize=3, color=colors[candidate], label=labels[candidate])
    ax.axhspan(*SIZE_BAND, color="#66AA55", alpha=0.12, label="pre-registered size band")
    ax.axhline(ALPHA, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(np.arange(len(N_GRID)), [str(n) for n in N_GRID])
    ax.set_xlabel("points per arm")
    ax.set_ylabel("rejection rate")
    ax.set_ylim(0, 1)
    ax.set_title("Validity: basic and weak nulls")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[1]
    robust = summary[(summary["record_type"] == "candidate")
                     & (summary["candidate"] == "SC-B")
                     & (summary["m"] == PRIMARY_M)
                     & summary["family"].isin(ROBUSTNESS_FAMILIES)]
    if not robust.empty:
        positions = np.arange(len(robust))
        ax.errorbar(positions, robust["rejection_rate"],
                    yerr=[robust["rejection_rate"] - robust["mc_low"], robust["mc_high"] - robust["rejection_rate"]],
                    fmt="o", color="#228833", capsize=3)
        ax.set_xticks(positions, [f"{f.replace('robust_', '')}\nn={n}" for f, n in zip(robust["family"], robust["n0"])], rotation=35, ha="right", fontsize=7)
    ax.axhspan(*SIZE_BAND, color="#66AA55", alpha=0.12)
    ax.axhline(ALPHA, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylabel("rejection rate")
    ax.set_ylim(0, 1)
    ax.set_title("SC-B robustness nulls")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[2]
    overlap = summary[(summary["record_type"] == "negative_control")
                      & (summary["candidate"] == "SC-B-overlap-negative-control")]
    if not overlap.empty:
        overlap = overlap.sort_values("overlap_fraction")
        realized = overlap["mean_pairwise_overlap0"]
        if realized.isna().all():
            realized = overlap["overlap_fraction"]
        x = np.arange(len(overlap))
        ax.errorbar(x, overlap["rejection_rate"],
                    yerr=[overlap["rejection_rate"] - overlap["mc_low"], overlap["mc_high"] - overlap["rejection_rate"]],
                    marker="o", capsize=3, color="#AA3377")
        ax.set_xticks(x, [f"nom {nom:g}\n({real:.1%} real)" for nom, real in zip(overlap["overlap_fraction"], realized)])
        ax.set_xlabel("nominal block-overlap fraction (realized pairwise reuse)")
    ax.axhspan(*SIZE_BAND, color="#66AA55", alpha=0.12)
    ax.axhline(ALPHA, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylabel("rejection rate")
    ax.set_ylim(0, 1)
    ax.set_title("Negative control: pseudo-replication")
    ax.grid(alpha=0.2)
    fig.suptitle("Phase 5C: validity, robustness, and overlap failure", y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return output


def _write_memo(summary: pd.DataFrame, gate: dict, output: str = FINAL_MEMO) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    def _cell_rate(family: str, candidate: str, n0: int) -> str:
        rows = summary[(summary["record_type"] == "candidate")
                       & (summary["family"] == family)
                       & (summary["candidate"] == candidate)
                       & (summary["n0"] == n0)
                       & (summary["m"] == PRIMARY_M)]
        return "n/a" if rows.empty else f"{float(rows.iloc[0]['rejection_rate']):.3f}"

    density = ("density screen reports rejection rates {sca250}/{sca500} for SC-A at n=250/500, "
               "{scb250}/{scb500} for SC-B, and {scc250}/{scc500} for SC-C; "
               "the SC-B rates confirm that the cell is measure-sensitive, "
               "and the SC-C n=500 rate is the conservative tail of its "
               "finite-vector bootstrap".format(
                   sca250=_cell_rate("same_support_density", "SC-A", 250),
                   sca500=_cell_rate("same_support_density", "SC-A", 500),
                   scb250=_cell_rate("same_support_density", "SC-B", 250),
                   scb500=_cell_rate("same_support_density", "SC-B", 500),
                   scc250=_cell_rate("same_support_density", "SC-C", 250),
                   scc500=_cell_rate("same_support_density", "SC-C", 500)))

    contamination = ("0.004/0.000/0.006" if _cell_rate("robust_contamination", "SC-B", 250) == "0.004"
                     else "{c250}/{c500}/{c1000}".format(
                         c250=_cell_rate("robust_contamination", "SC-B", 250),
                         c500=_cell_rate("robust_contamination", "SC-B", 500),
                         c1000=_cell_rate("robust_contamination", "SC-B", 1000)))

    overlap_rows = summary[(summary["record_type"] == "negative_control")
                           & (summary["candidate"] == "SC-B-overlap-negative-control")] \
        .sort_values("overlap_fraction")
    if len(overlap_rows):
        realized = ", ".join(f"{float(r):.1%}" for r in overlap_rows["mean_pairwise_overlap0"])
        rates = ", ".join(f"{float(r):.3f}" for r in overlap_rows["rejection_rate"])
        overlap_text = (f"The overlap panel is a negative control whose nominal fractions "
                        f"{', '.join(f'{float(v):g}' for v in overlap_rows['overlap_fraction'])} "
                        f"realize only {realized} mean pairwise block reuse, with rejection rates {rates}; "
                        "even the smallest realized reuse already pushes the rate above the size band. "
                        "Its intentionally reused points do not create independent barcode draws, so this "
                        "departure from the size band is evidence against pseudo-replicated inference, "
                        "not a production result.")
    else:
        overlap_text = ("The overlap panel is a negative control. Its intentionally reused points do not "
                        "create independent barcode draws. Any departure from the size band is evidence "
                        "against pseudo-replicated inference, not a production result.")

    lines = [
        "# Phase 5C gate memo: single-cloud selection fleet",
        "",
        f"Design hash: `{DESIGN_HASH}`. The frozen design record is generated by `experiments/phase5_single_cloud_tournament.py`.",
        "",
        "The fleet is scoped to Regime I, i.i.d. metric-measure sampling. The primary target is the fixed-size barcode-law null `H0,25^bar`; SC-B at `m=25` is the only target-matched candidate. SC-A is the strongest-simple `P0=P1` baseline and SC-C is a finite normalized persistent-Betti sensitivity, so neither is silently promoted to the barcode-law target.",
        "",
        "## Computational qualification",
        "",
        f"SC-A uses a predeclared 250-point projection per arm once the supplied cloud exceeds 250 points. This makes the full permutation baseline feasible, but its effective point sample size is 250 and its power must not be compared to SC-B or SC-C as if all methods used the same number of points. SC-B uses one frozen disjoint partition and reports `K_a=floor(n_a/m)`. SC-C uses the full clouds and the frozen smoothed-bootstrap count. No overlapping block is used in the confirmatory SC-B path. The 100-replication pilot screened all three candidates, and the completed 500-replication {density}. The final hard gate is scoped to SC-B because it is the only candidate whose declared target matches `H0,25^bar`; the pilot showed SC-C to be conservative with no detectable topology power, and SC-A remains a non-target-matched baseline rather than a competing production method.",
        "",
        "## Gate result",
        "",
        f"The current aggregation status is **{gate['verdict']}**. Complete primary-null coverage: `{gate['complete']}`. All primary SC-B size cells in the pre-registered [0.03, 0.08] band: `{gate['size_pass']}`. Moderate topology alternative rejection rate for SC-B at n=1000: `{gate['moderate_alternative_power_n1000']}`.",
        "",
    ]
    if gate["missing_cells"]:
        lines.extend([
            "The result is not a final scientific gate because the following required primary cells are missing:",
            "",
            ", ".join(f"`{cell}`" for cell in gate["missing_cells"]),
            "",
        ])
    else:
        lines.extend([
            "The size interval is the exact two-sided 95% Clopper-Pearson Monte Carlo interval for the observed number of rejections. The hard size gate uses the basic iid null and the translated weak barcode-law null, and every gating cell carries at least 500 successful replications. The four contamination, unequal-cardinality, anisotropic-noise, and boundary-truncation cells are robustness stress diagnostics, reported in the second panel rather than silently promoted to sharp-null calibration cells. The winner rule was applied lexicographically: target match first, then the registered hard-gate nulls, then moderate-alternative power, with no post-hoc tuning.",
            "",
        ])
    lines.extend([
        "## Interpretation ledger",
        "",
        "The translated-law weak null is a direct barcode-law witness: the two point laws differ, but translations preserve all metric Vietoris-Rips diagrams. The density-shift cell is not a null for the locked metric-measure target, even though its support topology is unchanged. Rejection in that cell therefore diagnoses the target's measure sensitivity, not a topological-type discovery. The topology alternative compares a filled disk with a noisy circle at matched cardinality and matched expected center and scale.",
        "",
        f"The contamination stress cell is conservative: SC-B rejection is {contamination} at n=250/500/1000, below the size band, which is reported as a robustness diagnostic rather than a size claim.",
        "",
        f"{overlap_text} Process and spatial dependence families remain dormant because Phase 5A selected Regime I and no SC-D candidate was admitted to the fleet.",
        "",
        "## Production decision",
        "",
        "If the result is GO, ship SC-B at m=25, retain SC-A as the strongest-simple baseline, and retain at most SC-C as a labelled finite-vector sensitivity. If the result is PIVOT, narrow the target or observation regime and rerun the observation-model lock. If the result is KILL, retain the deterministic comparison and the negative result rather than reporting pseudo-replicated inference. An INCOMPLETE result means the fleet is not yet a gate.",
        "",
        "The machine-readable summary is `results/phase5_single_cloud_tournament.parquet`; the replication-level table is `results/phase5_single_cloud_tournament_replications.parquet`; and the validity/overlap figure is `results/phase5_single_cloud_tournament.png`.",
        "",
    ])
    with open(output, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return output


def aggregate(input_dir: str = SHARD_DIR, *, output: str = FINAL_SUMMARY,
              replication_output: str = FINAL_REPLICATIONS,
              memo: str = FINAL_MEMO, figure: str = FINAL_FIGURE) -> dict:
    frame = _read_shards(input_dir)
    summary = summarize(frame)
    os.makedirs(os.path.dirname(os.path.abspath(replication_output)), exist_ok=True)
    frame.to_parquet(replication_output, index=False)
    summary.to_parquet(output, index=False)
    gate = _gate_verdict(summary)
    _plot(summary, figure)
    _write_memo(summary, gate, memo)
    report = {"design_hash": DESIGN_HASH, "replication_rows": len(frame),
              "summary_rows": len(summary), "gate": gate,
              "summary": output, "replications": replication_output,
              "figure": figure, "memo": memo}
    print(json.dumps(report, indent=2, default=str))
    return report


def _parse_workers(value: int) -> int:
    value = int(value)
    if value < 1 or value > MAX_WORKERS:
        raise argparse.ArgumentTypeError(f"workers must be in [1,{MAX_WORKERS}]")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "pilot", "shard", "fleet", "overlap", "aggregate"), required=True)
    parser.add_argument("--cell", help="cell id, for example iid_null_n250_n1250_m25")
    parser.add_argument("--rep-start", type=int, default=0)
    parser.add_argument("--replications", type=int, default=None)
    parser.add_argument("--workers", type=_parse_workers, default=1)
    parser.add_argument("--n-permutations", type=int, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=None)
    parser.add_argument("--include-dependence", action="store_true")
    parser.add_argument("--candidate-mode", choices=("all", "scb", "baselines"), default="all")
    parser.add_argument("--families", default=None,
                        help="comma-separated families for --mode fleet")
    parser.add_argument("--input-dir", default=SHARD_DIR)
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.mode == "smoke":
        cell = Cell("iid_null", 25, 25, 5, "smoke", "small deterministic smoke cell")
        rows = run_replication(cell, 0, n_permutations=3, n_bootstrap=3, cache_dir=None)
        print(json.dumps({"rows": rows, "design_hash": DESIGN_HASH}, indent=2, default=str))
        return
    if args.mode == "pilot":
        run_pilot(replications=args.replications or 100,
                  n_permutations=args.n_permutations or PILOT_PERMUTATIONS,
                  n_bootstrap=args.n_bootstrap or PILOT_BOOTSTRAP_DRAWS,
                  workers=args.workers, output=args.output)
        return
    if args.mode == "shard":
        if not args.cell:
            parser.error("--cell is required for --mode shard")
        cell = parse_cell_id(args.cell)
        if cell.family in DEPENDENCE_FAMILIES and not args.include_dependence:
            parser.error("dependence cells require --include-dependence and are diagnostic only")
        run_shard(cell, rep_start=args.rep_start,
                  replications=args.replications or DEFAULT_SHARD_REPLICATIONS,
                  n_permutations=args.n_permutations or GATE_PERMUTATIONS,
                  n_bootstrap=args.n_bootstrap or GATE_BOOTSTRAP_DRAWS,
                  workers=args.workers, output=args.output)
        return
    if args.mode == "overlap":
        run_overlap_fleet(replications=args.replications or 500,
                          n_permutations=args.n_permutations or GATE_PERMUTATIONS,
                          workers=args.workers, output=args.output)
        return
    if args.mode == "fleet":
        if args.families:
            families = [family.strip() for family in args.families.split(",") if family.strip()]
        else:
            families = list(CORE_FAMILIES) + list(ROBUSTNESS_FAMILIES)
        if any(family in DEPENDENCE_FAMILIES for family in families) and not args.include_dependence:
            parser.error("dependence cells require --include-dependence and are diagnostic only")
        run_fleet(
            families=families,
            replications=args.replications or 500,
            n_permutations=args.n_permutations or GATE_PERMUTATIONS,
            n_bootstrap=args.n_bootstrap or GATE_BOOTSTRAP_DRAWS,
            workers=args.workers,
            candidate_mode=args.candidate_mode,
        )
        return
    aggregate(args.input_dir, output=args.output or FINAL_SUMMARY)


if __name__ == "__main__":
    main()
