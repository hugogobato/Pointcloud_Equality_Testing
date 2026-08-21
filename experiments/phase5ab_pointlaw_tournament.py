"""Phase 5AB point-law benchmark tournament — frozen protocol.

This runner extends the Phase 5 comparison so RawBlockMMD is evaluated against
established raw-data methods for H0^law: P0=P1 (not only SC-A/SC-B).

It reuses DGP constructors and seed conventions from
experiments/phase5_single_cloud_tournament.py and the block methods from
tda2s/tests/single_cloud.py, adding the point-level baselines from
tda2s/tests/point_law.py.

Locked design constants (Section 3.1):
  alpha=0.05 primary (0.01/0.10 diagnostics)
  m=25 primary, sensitivity {1,2,5,10,25,50}
  replications 500 primary, permutations 199 primary (39 reproduction, 999 cheap)
  hybrid alpha {0.25,0.50,0.75,1.00}
  d=2 primary, sensitivity {1,5,10,20,50} where DGP defined
  primary point bandwidth 0.30 (Gaussian), multipliers {0.25,0.5,1,2,4}
  primary bag kernel Hilbert-Gaussian (point bw 0.10, bag bw 0.25)

Methods dispatched:
  Required: PointMMD-Gaussian, EnergyDistance, FriedmanRafsky-MST, Schilling-kNN
  Secondary: Rosenbaum-CrossMatch, SlicedWasserstein, ClassifierTwoSample (logistic, rf)
  Retained: RawBlockMMD, SC-B, SC-A, HybridBlockMMD, SC-A-Block (for target-aware ranking)

Output schema: every replication record (including refusals) contains the 34
required fields per Section 3.3; additional method-specific fields are allowed.

Usage:
  Pilot (cheap):
    python experiments/phase5ab_pointlaw_tournament.py --mode pilot --replications 20 --permutations 39
  Single cell shard (for Colab):
    python experiments/phase5ab_pointlaw_tournament.py --mode shard --cell iid_null_n250_n250_m25_d2 --rep-start 0 --replications 25
  Full fleet locally (warning: long):
    python experiments/phase5ab_pointlaw_tournament.py --mode fleet --replications 500 --permutations 199 --workers 4
  Aggregate:
    python experiments/phase5ab_pointlaw_tournament.py --mode aggregate --input-dir results/phase5ab_pointlaw_shards
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Optional, Sequence

# Ensure project root is on sys.path when run as script `python experiments/...`
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from experiments.phase5_single_cloud_tournament import (
    ALPHA as PHASE5_ALPHA,
    BETTI_GRID,
    FILTRATION,
    HOMOLOGY_DIMS,
    KERNEL_BANDWIDTH,
    MAX_SC_A_POINT_N,
    FAMILY_ROLE,
    FAMILY_DESCRIPTION,
    _seed as _phase5_seed,
    make_cloud_pair,
    _project_cloud,
)
from tda2s.tests.single_cloud import (
    DEFAULT_RAW_BAG_BANDWIDTH,
    DEFAULT_RAW_POINT_BANDWIDTH,
    REGIME_I,
    hybrid_block_mmd,
    raw_block_mmd,
    sc_a_blockwise_label_permutation,
    sc_a_label_permutation,
    sc_b_disjoint_mmd,
)
from tda2s.tests.point_law import (
    point_mmd_gaussian,
    energy_distance_test,
    friedman_rafsky_mst,
    schilling_knn,
    rosenbaum_crossmatch,
    sliced_wasserstein_test,
    classifier_two_sample_test,
)

# ---------------------------------------------------------------------------
# Frozen design record
BENCHMARK_VERSION = "phase5ab-pointlaw-v1"
ALPHA = 0.05
SIZE_BAND = (0.03, 0.08)
MC_CONFIDENCE = 0.95
PRIMARY_M = 25
M_GRID = (1, 2, 5, 10, 25, 50)
N_GRID = (250, 500, 1000)
D_GRID = (2, 5, 10, 20, 50)
PRIMARY_N = 250
PRIMARY_D = 2
GATE_REPLICATIONS = 500
GATE_PERMUTATIONS = 199
REPRO_PERMUTATIONS = 39
CHEAP_PERMUTATIONS = 999
PILOT_REPLICATIONS = 20
PILOT_PERMUTATIONS = 39
DEFAULT_SHARD_REPLICATIONS = 25
MAX_WORKERS = 16
SEED_ROOT = 20260821
PRIMARY_POINT_BANDWIDTH = 0.30
BANDWIDTH_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)
HYBRID_ALPHAS = (0.25, 0.50, 0.75, 1.00)
SCHILLING_KS = (1, 5, 10)

CORE_FAMILIES = ("iid_null", "weak_barcode_null", "same_support_density", "same_square_four_atom_density", "topology_alt")
ROBUSTNESS_FAMILIES = ("robust_contamination", "robust_unequal_cardinality", "robust_anisotropic_noise", "robust_boundary_truncation")
DEPENDENCE_FAMILIES = ("process_poisson", "process_inhomogeneous_poisson", "process_cox_clustered", "process_hard_core")

# Map for registry
FAMILY_ROLE_EXT = dict(FAMILY_ROLE)
FAMILY_ROLE_EXT["same_square_four_atom_density"] = "target_mismatch"
FAMILY_DESCRIPTION_EXT = dict(FAMILY_DESCRIPTION)
FAMILY_DESCRIPTION_EXT["same_square_four_atom_density"] = "four-atom square p=(.25,.25,.25,.25) vs q=(.70,.10,.10,.10)"

# Results paths
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
SHARD_DIR = os.path.join(RESULTS_DIR, "phase5ab_pointlaw_shards")
FINAL_REPLICATIONS = os.path.join(RESULTS_DIR, "phase5ab_pointlaw_replications.parquet")
FINAL_SUMMARY = os.path.join(RESULTS_DIR, "phase5ab_pointlaw_summary.parquet")
FINAL_COMPARISON = os.path.join(RESULTS_DIR, "phase5ab_pointlaw_comparison.parquet")
FINAL_MANIFEST = os.path.join(RESULTS_DIR, "phase5ab_pointlaw_manifest.json")
FINAL_FIGURE = os.path.join(RESULTS_DIR, "phase5ab_pointlaw_comparison.png")
CACHE_DIR = os.path.join(RESULTS_DIR, "phase5ab_pointlaw_ph_cache")

# Candidate sets
PRIMARY_CANDIDATES = (
    "PointMMD-Gaussian",
    "EnergyDistance",
    "FriedmanRafsky-MST",
    "Schilling-kNN-k1",
    "RawBlockMMD",
    "SC-B",
    "SC-A",
)
SECONDARY_CANDIDATES = (
    "Rosenbaum-CrossMatch",
    "SlicedWasserstein",
    "ClassifierTwoSampleTest-logistic",
    "ClassifierTwoSampleTest-rf",
    "Schilling-kNN-k5",
    "Schilling-kNN-k10",
    "PointMMD-Gaussian-median",
)
ALL_CANDIDATES = PRIMARY_CANDIDATES + SECONDARY_CANDIDATES + ("HybridBlockMMD-a0.50", "SC-A-Block")

PROFILE_HEAVY_CANDIDATES = frozenset({
    "SC-A",
    "Rosenbaum-CrossMatch",
    "SlicedWasserstein",
    "ClassifierTwoSampleTest-logistic",
    "ClassifierTwoSampleTest-rf",
})

def design_record() -> dict:
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "phase": "5AB-pointlaw",
        "regime": REGIME_I,
        "alpha": ALPHA,
        "size_band": list(SIZE_BAND),
        "confidence": MC_CONFIDENCE,
        "n_grid": list(N_GRID),
        "m_grid": list(M_GRID),
        "d_grid": list(D_GRID),
        "primary_m": PRIMARY_M,
        "primary_d": 2,
        "filtration": FILTRATION,
        "homology_dims": list(HOMOLOGY_DIMS),
        "kernel_bandwidth": KERNEL_BANDWIDTH,
        "betti_grid": BETTI_GRID.tolist(),
        "primary_point_bandwidth": PRIMARY_POINT_BANDWIDTH,
        "bandwidth_multipliers": list(BANDWIDTH_MULTIPLIERS),
        "raw_point_bandwidth": DEFAULT_RAW_POINT_BANDWIDTH,
        "raw_bag_bandwidth": DEFAULT_RAW_BAG_BANDWIDTH,
        "hybrid_alphas": list(HYBRID_ALPHAS),
        "schilling_ks": list(SCHILLING_KS),
        "gate_replications": GATE_REPLICATIONS,
        "gate_permutations": GATE_PERMUTATIONS,
        "repro_permutations": REPRO_PERMUTATIONS,
        "pilot_permutations": PILOT_PERMUTATIONS,
        "sc_a_projection_n": MAX_SC_A_POINT_N,
        "seed_root": SEED_ROOT,
        "target_RawBlockMMD": "H0^law: P0=P1 (raw characteristic fixed-size block, iid)",
        "target_SC-B": "H0,25^bar: Phi^25_0:1(P0)=Phi^25_0:1(P1)",
        "target_SC-A": "H0^law: P0=P1 (persistence representation, not barcode law)",
        "seed_convention": {
            "cloud": ["benchmark_version", "family", "n0", "n1", "dimension", "replication"],
            "partition": ["benchmark_version", "cell_id", "replication"],
            "method": ["benchmark_version", "cell_id", "candidate", "replication"],
        },
    }

DESIGN_HASH = hashlib.sha256(
    json.dumps(design_record(), sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()[:16]

@dataclass(frozen=True)
class Cell:
    family: str
    n0: int
    n1: int
    m: int
    d: int
    role: str
    description: str

    @property
    def cell_id(self) -> str:
        return f"{self.family}_n{self.n0}_n1{self.n1}_m{self.m}_d{self.d}"


PROFILE_CELLS = (
    Cell("iid_null", 250, 250, PRIMARY_M, 2, "gating_null", "iid null"),
    Cell("same_support_density", 250, 250, PRIMARY_M, 2, "target_mismatch", "density"),
    Cell("topology_alt", 250, 250, PRIMARY_M, 2, "power", "topology"),
    Cell("iid_null", 50, 50, PRIMARY_M, 2, "gating_null", "small n"),
    Cell("iid_null", 250, 250, PRIMARY_M, 10, "gating_null", "high d"),
)

def _seed(*parts: object) -> int:
    payload = repr((SEED_ROOT,) + parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _cloud_seed(cell: Cell, replication: int) -> int:
    """Cloud seed independent of block size, so m-sensitivity shares clouds."""
    return _seed(BENCHMARK_VERSION, "cloud", cell.family, cell.n0, cell.n1, cell.d, replication)


def _partition_seed(cell: Cell, replication: int) -> int:
    return _seed(BENCHMARK_VERSION, "partition", cell.cell_id, replication)


def _method_seed(cell: Cell, replication: int, candidate: str) -> int:
    return _seed(BENCHMARK_VERSION, "method", cell.cell_id, candidate, replication)

def make_cells(
    families: Sequence[str] = CORE_FAMILIES,
    n_grid: Sequence[int] = N_GRID,
    m_values: Sequence[int] = (PRIMARY_M,),
    d_values: Sequence[int] = (2,),
    include_robustness: bool = False,
    include_dependence: bool = False,
) -> list[Cell]:
    fams = list(families)
    if include_robustness:
        fams.extend([f for f in ROBUSTNESS_FAMILIES if f not in fams])
    if include_dependence:
        fams.extend([f for f in DEPENDENCE_FAMILIES if f not in fams])
    cells = []
    for family in fams:
        for n in n_grid:
            n1_val = int(math.ceil(n * 1.25)) if family == "robust_unequal_cardinality" else int(n)
            for m in m_values:
                for d in d_values:
                    cells.append(Cell(
                        family=family, n0=int(n), n1=int(n1_val), m=int(m), d=int(d),
                        role=FAMILY_ROLE_EXT.get(family, "unknown"),
                        description=FAMILY_DESCRIPTION_EXT.get(family, family),
                    ))
    return cells

def _embed_cloud(cloud: np.ndarray, d: int, seed: int) -> np.ndarray:
    """Embed 2D cloud into d dimensions with independent irrelevant Gaussian noise."""
    cloud = np.asarray(cloud, dtype=float)
    if d == 2:
        return cloud
    if d < 2:
        raise ValueError("d must be >=2")
    n = len(cloud)
    rng = np.random.default_rng(seed)
    extra = rng.normal(0.0, 0.25, size=(n, int(d) - 2))
    return np.concatenate([cloud, extra], axis=1)

def _four_atom_cloud(n: int, rng: np.random.Generator, p: np.ndarray) -> np.ndarray:
    square = np.asarray([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]])
    return square[rng.choice(4, size=int(n), p=p)]

def make_cloud_pair_extended(family: str, n0: int, n1: int, seed: int, d: int = 2) -> tuple[np.ndarray, np.ndarray]:
    if family == "same_square_four_atom_density":
        rng = np.random.default_rng(seed)
        p = np.full(4, 0.25)
        q = np.asarray([0.70, 0.10, 0.10, 0.10])
        c0 = _four_atom_cloud(n0, rng, p)
        # use independent rng stream for second arm but deterministic from same seed
        rng1 = np.random.default_rng(_seed("four_atom_q", seed))
        c1 = _four_atom_cloud(n1, rng1, q)
        if d != 2:
            c0 = _embed_cloud(c0, d, _seed("embed0", seed, d))
            c1 = _embed_cloud(c1, d, _seed("embed1", seed, d))
        return c0, c1
    # default: use base DGP then embed
    c0, c1 = make_cloud_pair(family, n0, n1, seed)
    if d != 2:
        c0 = _embed_cloud(c0, d, _seed("embed0", seed, d))
        c1 = _embed_cloud(c1, d, _seed("embed1", seed, d))
    return c0, c1

def _mc_interval(successes: int, total: int, confidence: float = MC_CONFIDENCE):
    if total < 1 or successes < 0 or successes > total:
        raise ValueError("invalid binomial count")
    from scipy.stats import beta
    tail = (1.0 - confidence) / 2.0
    lower = 0.0 if successes == 0 else float(beta.ppf(tail, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1.0 - tail, successes + 1, total - successes))
    return lower, upper

# ---------------------------------------------------------------------------
# Method dispatch -> unified record

def _unified_record(
    *,
    cell: Cell,
    replication: int,
    candidate: str,
    method_variant: str,
    result: dict,
    cloud_seed: int,
    partition_seed: Optional[int],
    permutation_seed: int,
    status: str = "ok",
    failure_reason: str = "",
) -> dict:
    # result may be from point_law or single_cloud; normalize fields
    diagnostics = result.get("diagnostics", {}) if isinstance(result, dict) else {}
    # Determine target, validity, sampling_unit per registry or result
    target = result.get("inferential_target") or result.get("target_null") or result.get("target") or ""
    validity = result.get("validity_regime") or result.get("regime") or REGIME_I
    sampling_unit = result.get("sampling_unit") or diagnostics.get("sampling_unit") or ""
    # K, m, d, effective sample size
    is_block = candidate in ("RawBlockMMD","SC-B","SC-A-Block") or candidate.startswith("Hybrid")
    m_val = float(cell.m) if is_block else np.nan
    K0 = result.get("K0", np.nan)
    K1 = result.get("K1", np.nan)
    # For point methods K is nan; set effective total
    if status != "ok":
        effective_total = np.nan
        unused0 = np.nan
        unused1 = np.nan
    else:
        if candidate in ("RawBlockMMD","SC-B","HybridBlockMMD-a0.50","SC-A-Block","SC-B-production"):
            try:
                K0 = int(K0); K1 = int(K1)
            except Exception:
                K0 = np.nan; K1 = np.nan
            effective_total = int(K0+K1) if np.isfinite(K0) and np.isfinite(K1) else np.nan
            unused0 = int(result.get("remainder0", result.get("unused_points0", 0))) if "remainder0" in result else int(result.get("unused_points0", 0))
            unused1 = int(result.get("remainder1", result.get("unused_points1", 0))) if "remainder1" in result else int(result.get("unused_points1", 0))
        else:
            # point-level: total points
            n0 = int(result.get("n0", cell.n0))
            n1 = int(result.get("n1", cell.n1))
            effective_total = int(n0 + n1)
            K0 = np.nan; K1 = np.nan
            unused0 = 0; unused1 = 0
            m_val = np.nan
    # kernel/distance, bandwidth, alpha
    kernel_or_distance = result.get("kernel_or_distance") or result.get("kernel") or result.get("distance") or diagnostics.get("kernel") or diagnostics.get("distance") or ""
    bandwidth_or_tuning = result.get("bandwidth_or_tuning")
    if bandwidth_or_tuning is None:
        bandwidth_or_tuning = result.get("bandwidth", np.nan)
        if isinstance(bandwidth_or_tuning, dict):
            bandwidth_or_tuning = json.dumps(bandwidth_or_tuning)
    # alpha for hybrid
    alpha_val = result.get("alpha", np.nan)
    if "alpha" in diagnostics:
        alpha_val = diagnostics["alpha"]
    # Also check candidate parsing
    if candidate.startswith("Hybrid"):
        try:
            alpha_val = float(candidate.split("a")[-1])
        except Exception:
            pass
    # Permutation group
    perm_group = result.get("permutation_group") or result.get("perm_group") or ""
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "design_hash": DESIGN_HASH,
        "family": cell.family,
        "family_role": cell.role,
        "family_description": cell.description,
        "method": candidate,
        "method_variant": method_variant,
        "target_null": target,
        "validity_regime": validity,
        "sampling_unit": sampling_unit,
        "n0": int(cell.n0),
        "n1": int(cell.n1),
        "d": int(cell.d),
        "m": float(m_val) if isinstance(m_val, (int,float,np.floating)) and np.isfinite(float(m_val)) else np.nan,
        "K0": float(K0) if status=="ok" else np.nan,
        "K1": float(K1) if status=="ok" else np.nan,
        "effective_sample_size_total": float(effective_total) if status=="ok" and np.isfinite(effective_total) else (effective_total if status=="ok" else np.nan),
        "unused_points0": float(unused0) if status=="ok" else np.nan,
        "unused_points1": float(unused1) if status=="ok" else np.nan,
        "n_permutations": int(result.get("n_permutations", result.get("n_resamples", -1))) if status=="ok" else -1,
        "exact_enumeration": bool(result.get("exact_enumeration", result.get("exact", False))) if status=="ok" else False,
        "permutation_group": perm_group,
        "statistic": float(result.get("statistic", np.nan)) if status=="ok" else np.nan,
        "pvalue": float(result.get("pvalue", np.nan)) if status=="ok" else np.nan,
        "rejected": bool(float(result.get("pvalue", 1.0)) <= ALPHA) if status=="ok" else False,
        "kernel_or_distance": str(kernel_or_distance),
        "bandwidth_or_tuning": float(bandwidth_or_tuning) if isinstance(bandwidth_or_tuning, (int,float,np.floating)) and np.isfinite(float(bandwidth_or_tuning)) else (bandwidth_or_tuning if isinstance(bandwidth_or_tuning, str) else np.nan),
        "alpha": float(alpha_val) if isinstance(alpha_val, (int,float,np.floating)) and np.isfinite(float(alpha_val)) else np.nan,
        "cloud_seed": int(cloud_seed),
        "partition_seed": int(partition_seed) if partition_seed is not None else np.nan,
        "permutation_seed": int(permutation_seed),
        "runtime_seconds": float(result.get("runtime_seconds", np.nan)) if status=="ok" else np.nan,
        "peak_rss_bytes": int(result.get("peak_rss_bytes", -1)) if status=="ok" else -1,
        "peak_memory_bytes": int(result.get("peak_memory_bytes", -1)) if status=="ok" else -1,
        "status": status,
        "failure_reason": failure_reason,
        # extra diagnostics for traceability
        "cell_id": cell.cell_id,
        "replication": int(replication),
        "d_diagnostics": json.dumps({k: str(v) for k, v in diagnostics.items()}, sort_keys=True) if diagnostics else "",
    }

def _call_method(
    cell: Cell,
    replication: int,
    candidate: str,
    *,
    n_permutations: int,
    cache_dir: Optional[str],
    cloud_pair: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> tuple[dict, str]:
    cloud_seed = _cloud_seed(cell, replication)
    partition_seed = _partition_seed(cell, replication)
    method_seed = _method_seed(cell, replication, candidate)
    if cloud_pair is None:
        cloud_pair = make_cloud_pair_extended(cell.family, cell.n0, cell.n1, cloud_seed, d=cell.d)
    cloud0, cloud1 = cloud_pair

    # Dispatch
    try:
        if candidate == "PointMMD-Gaussian":
            result = point_mmd_gaussian(cloud0, cloud1, regime=REGIME_I, bandwidth=PRIMARY_POINT_BANDWIDTH, n_perm=n_permutations, seed=method_seed)
            variant = f"bw{PRIMARY_POINT_BANDWIDTH}"
        elif candidate == "PointMMD-Gaussian-median":
            result = point_mmd_gaussian(cloud0, cloud1, regime=REGIME_I, bandwidth=None, n_perm=n_permutations, seed=method_seed)
            variant = "median_heuristic"
        elif candidate.startswith("PointMMD-Gaussian-bw"):
            # e.g. PointMMD-Gaussian-bw0.15
            try:
                bw = float(candidate.split("bw")[1])
            except Exception:
                bw = PRIMARY_POINT_BANDWIDTH
            result = point_mmd_gaussian(cloud0, cloud1, regime=REGIME_I, bandwidth=bw, n_perm=n_permutations, seed=method_seed)
            variant = f"bw{bw}"
        elif candidate == "EnergyDistance":
            result = energy_distance_test(cloud0, cloud1, regime=REGIME_I, n_perm=n_permutations, seed=method_seed)
            variant = ""
        elif candidate == "FriedmanRafsky-MST":
            result = friedman_rafsky_mst(cloud0, cloud1, regime=REGIME_I, n_perm=n_permutations, seed=method_seed)
            variant = ""
        elif candidate.startswith("Schilling-kNN"):
            # parse k (candidate is e.g. Schilling-kNN-k1, contains two '-k' substrings)
            if "-k" in candidate:
                try:
                    k = int(candidate.rsplit("-k", 1)[1])
                except Exception:
                    k = 1
            else:
                k = 1
            result = schilling_knn(cloud0, cloud1, regime=REGIME_I, k=k, directed=True, n_perm=n_permutations, seed=method_seed)
            variant = f"k{k}"
        elif candidate == "Rosenbaum-CrossMatch":
            result = rosenbaum_crossmatch(cloud0, cloud1, regime=REGIME_I, n_perm=n_permutations, seed=method_seed)
            variant = ""
        elif candidate == "SlicedWasserstein":
            result = sliced_wasserstein_test(cloud0, cloud1, regime=REGIME_I, n_projections=100, projection_seed=0, n_perm=n_permutations, seed=method_seed)
            variant = "100proj"
        elif candidate == "ClassifierTwoSampleTest-logistic":
            result = classifier_two_sample_test(cloud0, cloud1, regime=REGIME_I, classifier="logistic", test_fraction=0.5, split_seed=method_seed, n_perm=n_permutations, seed=method_seed)
            variant = "logistic"
        elif candidate == "ClassifierTwoSampleTest-rf":
            result = classifier_two_sample_test(cloud0, cloud1, regime=REGIME_I, classifier="rf", test_fraction=0.5, split_seed=method_seed, n_perm=n_permutations, seed=method_seed)
            variant = "rf"
        elif candidate == "RawBlockMMD":
            result = raw_block_mmd(cloud0, cloud1, regime=REGIME_I, m=cell.m, partition_seed=partition_seed, n_perm=n_permutations, seed=method_seed)
            variant = f"m{cell.m}"
        elif candidate == "SC-B":
            result = sc_b_disjoint_mmd(cloud0, cloud1, regime=REGIME_I, m=cell.m, partition_seed=partition_seed, n_perm=n_permutations, seed=method_seed, filtration=FILTRATION, homology_dims=HOMOLOGY_DIMS, kernel_bandwidth=KERNEL_BANDWIDTH)
            variant = f"m{cell.m}"
        elif candidate == "SC-A":
            # SC-A uses pooled point permutation but expensive; apply projection cap like Phase 5
            x0p = _project_cloud(cloud0, MAX_SC_A_POINT_N, _seed(BENCHMARK_VERSION, "SC-A-project0", cell.cell_id, replication))
            x1p = _project_cloud(cloud1, MAX_SC_A_POINT_N, _seed(BENCHMARK_VERSION, "SC-A-project1", cell.cell_id, replication))
            result = sc_a_label_permutation(x0p, x1p, regime=REGIME_I, filtration=FILTRATION, homology_dims=HOMOLOGY_DIMS, kernel_bandwidth=KERNEL_BANDWIDTH, n_perm=n_permutations, seed=method_seed, cache_dir=cache_dir)
            # record projection in diagnostics
            result["diagnostics"]["projection_used"] = len(x0p) < len(cloud0) or len(x1p) < len(cloud1)
            variant = ""
        elif candidate.startswith("HybridBlockMMD"):
            # format HybridBlockMMD-a0.50
            try:
                alpha = float(candidate.split("a")[1])
            except Exception:
                alpha = 0.5
            result = hybrid_block_mmd(cloud0, cloud1, regime=REGIME_I, m=cell.m, alpha=alpha, partition_seed=partition_seed, n_perm=n_permutations, seed=method_seed, filtration=FILTRATION, homology_dims=HOMOLOGY_DIMS, barcode_kernel_bandwidth=KERNEL_BANDWIDTH)
            variant = f"a{alpha:.2f}"
        elif candidate == "SC-A-Block":
            result = sc_a_blockwise_label_permutation(cloud0, cloud1, regime=REGIME_I, m=cell.m, partition_seed=partition_seed, n_perm=n_permutations, seed=method_seed)
            variant = f"m{cell.m}"
        else:
            raise ValueError(f"unknown candidate {candidate!r}")
        return result, variant
    except Exception as exc:
        # bubble up for unified_record to catch as failure/refusal
        raise exc

def _run_one(args) -> list[dict]:
    cell, replication, candidates, n_permutations, cache_dir = args
    rows = []
    cloud_seed = _cloud_seed(cell, replication)
    partition_seed = _partition_seed(cell, replication)
    cloud_pair = make_cloud_pair_extended(cell.family, cell.n0, cell.n1, cloud_seed, d=cell.d)
    for candidate in candidates:
        method_seed = _method_seed(cell, replication, candidate)
        try:
            result, variant = _call_method(cell, replication, candidate, n_permutations=n_permutations, cache_dir=cache_dir, cloud_pair=cloud_pair)
            rec = _unified_record(cell=cell, replication=replication, candidate=candidate, method_variant=variant, result=result, cloud_seed=cloud_seed, partition_seed=partition_seed if candidate in ("RawBlockMMD","SC-B","HybridBlockMMD-a0.50","SC-A-Block") or candidate.startswith("Hybrid") else None, permutation_seed=method_seed, status="ok")
            # ensure rejected flag correct
            rec["rejected"] = bool(rec["pvalue"] <= ALPHA) if np.isfinite(rec["pvalue"]) else False
            rows.append(rec)
        except Exception as exc:
            # Refusal vs failed: check if expected refusal condition
            msg = f"{type(exc).__name__}: {exc}"
            # mark as refused if known condition phrases
            status = "failed"
            if any(k in str(exc).lower() for k in ["exceeds", "no barcode block", "m must be", "k must be", "cloud must contain", "pooled n", "refused", "k="]):
                status = "refused"
            # create dummy result for schema
            rec = _unified_record(cell=cell, replication=replication, candidate=candidate, method_variant="", result={"statistic": np.nan, "pvalue": np.nan, "n_permutations": -1, "exact_enumeration": False, "permutation_group": "", "diagnostics": {}, "kernel": "", "n0": cell.n0, "n1": cell.n1}, cloud_seed=cloud_seed, partition_seed=partition_seed if candidate in ("RawBlockMMD","SC-B","HybridBlockMMD-a0.50","SC-A-Block") or candidate.startswith("Hybrid") else None, permutation_seed=method_seed, status=status, failure_reason=msg)
            rows.append(rec)
    return rows

def run_replicates(
    *,
    families: Sequence[str] = CORE_FAMILIES,
    n_grid: Sequence[int] = (250,),
    m_values: Sequence[int] = (PRIMARY_M,),
    d_values: Sequence[int] = (2,),
    replications: int = PILOT_REPLICATIONS,
    n_permutations: int = PILOT_PERMUTATIONS,
    workers: int = 1,
    candidates: Sequence[str] = PRIMARY_CANDIDATES,
    cache_dir: Optional[str] = None,
    include_robustness: bool = False,
    include_dependence: bool = False,
) -> pd.DataFrame:
    if replications < 1 or n_permutations < 1:
        raise ValueError("replications and n_permutations must be positive")
    if workers < 1 or workers > MAX_WORKERS:
        raise ValueError(f"workers must be in [1,{MAX_WORKERS}]")
    cells = make_cells(families=families, n_grid=n_grid, m_values=m_values, d_values=d_values, include_robustness=include_robustness, include_dependence=include_dependence)
    cand_tuple = tuple(candidates)
    args = [(cell, rep, cand_tuple, int(n_permutations), cache_dir) for cell in cells for rep in range(int(replications))]
    if workers == 1:
        nested = [_run_one(a) for a in args]
    else:
        with ProcessPoolExecutor(max_workers=min(int(workers), MAX_WORKERS)) as pool:
            nested = list(pool.map(_run_one, args))
    return pd.DataFrame([r for rows in nested for r in rows])

def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    groups = ["method","family","family_role","n0","n1","m","d","alpha","kernel_or_distance","target_null","sampling_unit","validity_regime"]
    # ensure required columns exist
    for col in groups:
        if col not in frame.columns:
            frame[col] = np.nan
    rows = []
    for keys, group in frame.groupby(groups, dropna=False, sort=True):
        ok = group[group["status"]=="ok"]
        total = int(len(ok))
        fails = int(len(group)-total)
        refuses = int((group["status"]=="refused").sum())
        rejects = int(ok["rejected"].sum()) if total else 0
        rate = rejects/total if total else np.nan
        lo, hi = _mc_interval(rejects, total) if total else (np.nan, np.nan)
        values = dict(zip(groups, keys))
        values.update({
            "design_hash": DESIGN_HASH,
            "benchmark_version": BENCHMARK_VERSION,
            "replications": total,
            "failed_replications": fails - refuses,
            "refused_replications": refuses,
            "rejections": rejects,
            "rejection_rate": rate,
            "mc_low": lo,
            "mc_high": hi,
            "in_size_band": bool(SIZE_BAND[0] <= rate <= SIZE_BAND[1]) if total else False,
            "mean_runtime_seconds": float(ok["runtime_seconds"].mean()) if total and "runtime_seconds" in ok else np.nan,
            "mean_peak_rss_bytes": float(ok["peak_rss_bytes"].mean()) if total else np.nan,
            "mean_K0": float(ok["K0"].mean()) if total else np.nan,
            "mean_K1": float(ok["K1"].mean()) if total else np.nan,
            "effective_total_mean": float(ok["effective_sample_size_total"].mean()) if total else np.nan,
        })
        rows.append(values)
    return pd.DataFrame(rows)

def _comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    primary = summary[
        (summary["m"].isna() | summary["m"].eq(PRIMARY_M) | summary["m"].eq(float(PRIMARY_M)))
        & (summary["n0"] == PRIMARY_N)
        & (summary["n1"] == PRIMARY_N)
        & (summary["d"] == PRIMARY_D)
    ]
    # Keep both point and block methods at primary size
    # Select headline candidates
    keep = [c for c in ALL_CANDIDATES if c in summary["method"].unique()]
    primary = primary[primary["method"].isin(keep)]
    rows = []
    for method in sorted(primary["method"].unique()):
        sub = primary[primary["method"]==method]
        # pick target from first non-null
        try:
            target = sub["target_null"].dropna().iloc[0] if len(sub["target_null"].dropna()) else ""
        except Exception:
            target = ""
        try:
            unit = sub["sampling_unit"].dropna().iloc[0] if len(sub["sampling_unit"].dropna()) else ""
        except Exception:
            unit = ""
        def rate(fam):
            v = sub[sub["family"]==fam]["rejection_rate"]
            return float(v.iloc[0]) if len(v) else np.nan
        def mc(fam):
            v = sub[sub["family"]==fam]
            if len(v):
                return float(v.iloc[0]["mc_low"]), float(v.iloc[0]["mc_high"])
            return np.nan, np.nan
        iid_lo, iid_hi = mc("iid_null")
        rows.append({
            "method": method,
            "n0": PRIMARY_N,
            "n1": PRIMARY_N,
            "d": PRIMARY_D,
            "target_null": target,
            "sampling_unit": unit,
            "m": PRIMARY_M,
            "null_rejection_rate": rate("iid_null"),
            "null_mc_low": iid_lo,
            "null_mc_high": iid_hi,
            "density_power": rate("same_support_density"),
            "topology_power": rate("topology_alt"),
            "translated_pointlaw_power_or_barcode_null": rate("weak_barcode_null"),
            "four_atom_power": rate("same_square_four_atom_density"),
            "mean_runtime": float(sub["mean_runtime_seconds"].mean()) if len(sub) else np.nan,
            "mean_peak_rss": float(sub["mean_peak_rss_bytes"].mean()) if len(sub) else np.nan,
        })
    return pd.DataFrame(rows)

def _plot(summary: pd.DataFrame, output: str) -> None:
    mpl_config = os.path.join("/tmp", "tda2s_phase5ab_pointlaw_mplconfig")
    os.makedirs(mpl_config, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_config)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # The headline figure is explicitly the primary n=250, d=2 comparison.
    # The summary parquet retains all n and d cells separately.
    summary = summary[
        (summary["n0"] == PRIMARY_N)
        & (summary["n1"] == PRIMARY_N)
        & (summary["d"] == PRIMARY_D)
        & (summary["m"].isna() | summary["m"].eq(PRIMARY_M) | summary["m"].eq(float(PRIMARY_M)))
    ].copy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # Panel 1: calibration (iid null + weak barcode null)
    ax = axes[0,0]
    calibr = summary[summary["family"].isin(["iid_null","weak_barcode_null"])]
    methods = sorted(summary["method"].unique())
    colors = {"PointMMD-Gaussian":"#4477AA","EnergyDistance":"#228833","FriedmanRafsky-MST":"#CC6677","Schilling-kNN-k1":"#AA3377","RawBlockMMD":"#66CCEE","SC-B":"#228833","SC-A":"#4477AA","Rosenbaum-CrossMatch":"#332288","SlicedWasserstein":"#888888","ClassifierTwoSampleTest-logistic":"#CC3311","ClassifierTwoSampleTest-rf":"#EE7733"}
    for method in methods:
        sub = calibr[calibr["method"]==method].sort_values("family")
        if sub.empty:
            continue
        x = np.arange(len(sub))
        ax.errorbar(x, sub["rejection_rate"], yerr=[sub["rejection_rate"]-sub["mc_low"], sub["mc_high"]-sub["rejection_rate"]], marker="o", capsize=3, label=method, color=colors.get(method, None), linewidth=1)
    ax.axhspan(*SIZE_BAND, color="#66AA55", alpha=0.12)
    ax.axhline(ALPHA, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks([0,1], ["iid null","weak barcode"])
    ax.set_ylim(0,1)
    ax.set_ylabel("rejection rate")
    ax.set_title("Target separation and calibration (m=25, d=2)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.2)

    # Panel 2: power density vs topology by method
    ax = axes[0,1]
    power = summary[summary["family"].isin(["same_support_density","topology_alt"])]
    # pivot
    for method in methods:
        sub = power[power["method"]==method].sort_values("family")
        if sub.empty:
            continue
        if len(sub) == 2:
            ax.plot([0,1], sub["rejection_rate"].values, marker="o", label=method)
    ax.set_xticks([0,1], ["density","topology"])
    ax.set_ylim(0,1)
    ax.set_ylabel("rejection rate")
    ax.set_title("Density vs topology power")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.2)

    # Panel 3: runtime vs peak RSS
    ax = axes[1,0]
    for method in methods:
        sub = summary[summary["method"]==method]
        if sub.empty:
            continue
        ax.scatter(sub["mean_runtime_seconds"], sub["mean_peak_rss_bytes"]/1e6, label=method)
    ax.set_xlabel("mean runtime (s)")
    ax.set_ylabel("mean peak RSS (MB)")
    ax.set_title("Computation")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.2)

    # Panel 4: effective sample size diagnostic (K0+K1 vs n)
    ax = axes[1,1]
    valid = []
    labels = []
    vals = []
    for method in methods:
        sub = summary[summary["method"]==method]
        if sub.empty:
            continue
        eff = sub["effective_total_mean"].mean()
        if np.isfinite(eff):
            labels.append(method)
            vals.append(eff)
    if vals:
        ax.bar(np.arange(len(vals)), vals, tick_label=labels)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("effective sample size total")
    ax.set_title("Effective sample size (point vs block)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

def _write_report(summary: pd.DataFrame, comparison: pd.DataFrame, output: str, *, replications: int, n_permutations: int) -> None:
    lines = [
        "# Phase 5AB point-law benchmark report",
        "",
        f"Benchmark version `{BENCHMARK_VERSION}`, design hash `{DESIGN_HASH}`. Replications per cell `{replications}`, permutations `{n_permutations}`.",
        "",
        "## Design audit",
        "",
        "Frozen protocol per `docs/phase5ab_point_law_registry.md`. Primary `m=25` locked; point-level baselines use pooled kernels/distances/graphs cached outside permutation loop; block methods use frozen disjoint partition with `K_a=floor(n_a/m)` and remainders discarded.",
        "",
        "## Method registry (headline comparison distinguishes target)",
        "",
        "| method | target_null | sampling_unit | null_rejection | density | topology | translated | 4-atom | runtime | peak_RSS |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in comparison.iterrows():
        lines.append(f"| {row['method']} | {row['target_null']} | {row['sampling_unit']} | {row['null_rejection_rate']:.3f} | {row['density_power']:.3f} | {row['topology_power']:.3f} | {row['translated_pointlaw_power_or_barcode_null']:.3f} | {row['four_atom_power']:.3f} | {row['mean_runtime']:.3f} | {row['mean_peak_rss']:.0f} |")
    lines.extend([
        "",
        "## Gates (B1-B8)",
        "",
        "B1 target/literature validity: passes by construction for registered methods. SC-B remains under `H0,25^bar`; RawBlockMMD and required raw baselines under `H0^law` when assumptions hold.",
        "B2 iid size: primary H0^law methods should lie in [0.03,0.08] at 500 replications (with Clopper-Pearson intervals). Cells with K_a<5 are diagnostic only.",
        "B3 point-law power: RawBlockMMD gate passes only if competitive with strongest raw baseline on density/location/sparse/dense/rare alternatives (paired MC interval).",
        "B4 topology retention: useful power on filled-disk vs noisy-circle; hybrid predeclared alpha only.",
        "B5 barcode separation: translated cell is `translated_pointlaw_power` for RawBlockMMD vs `translated_barcode_null_rejection` for SC-B.",
        "B6 small-sample honesty: m=25 becomes unusable where K_a too small; report p-value grid coarseness.",
        "B7 robustness: overlap/dependence marked unsupported.",
        "B8 computation: runtime, peak RSS, failure rate, coverage reported; methods exceeding Colab budget flagged as computationally limited.",
        "",
        "## Effective sample size note",
        "",
        "Point-level methods use all points (`n0+n1` observations). Block methods use `K0+K1` blocks; the `m=1` RawBlock sensitivity is the bridge. Report contains both all-point and effective-sample-size comparisons.",
        "",
        "All headline numbers are regenerable from aggregated parquet without rerunning methods.",
        "",
    ])
    four = summary[summary["family"]=="same_square_four_atom_density"]
    if not four.empty:
        lines.extend(["## Four-atom diagnostic", "", "| method | rejection | 95% MC |", "|---|---:|---:|"])
        for _, r in four.iterrows():
            lines.append(f"| {r['method']} | {r['rejection_rate']:.3f} | [{r['mc_low']:.3f},{r['mc_high']:.3f}] |")
        lines.append("")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as h:
        h.write("\n".join(lines))

# ---------------------------------------------------------------------------
# Shard helpers (Colab-friendly)

def run_shard(
    cell: Cell,
    *,
    rep_start: int,
    replications: int,
    n_permutations: int,
    candidates: Sequence[str],
    workers: int = 1,
    cache_dir: Optional[str] = None,
    output: Optional[str] = None,
) -> str:
    if rep_start < 0 or replications < 1:
        raise ValueError("rep_start must be >=0 and replications positive")
    args = [(cell, rep, tuple(candidates), int(n_permutations), cache_dir) for rep in range(int(rep_start), int(rep_start)+replications)]
    if workers == 1:
        nested = [_run_one(a) for a in args]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, MAX_WORKERS)) as pool:
            nested = list(pool.map(_run_one, args))
    rows = [r for part in nested for r in part]
    frame = pd.DataFrame(rows)
    if output is None:
        output = os.path.join(SHARD_DIR, f"phase5ab_pointlaw_{cell.cell_id}_rep{rep_start}_{rep_start+replications-1}.parquet")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    frame.to_parquet(output, index=False)
    print(json.dumps({"cell": cell.cell_id, "replications": replications, "rows": len(rows), "workers": workers, "output": output}, indent=2))
    return output

def run_pilot(
    *,
    replications: int = PILOT_REPLICATIONS,
    n_permutations: int = PILOT_PERMUTATIONS,
    workers: int = 1,
    candidates: Sequence[str] = PRIMARY_CANDIDATES,
    families: Sequence[str] = CORE_FAMILIES,
    n_grid: Sequence[int] = (250,),
    output: Optional[str] = None,
) -> str:
    frame = run_replicates(families=families, n_grid=n_grid, m_values=(PRIMARY_M,), d_values=(2,), replications=replications, n_permutations=n_permutations, workers=workers, candidates=candidates)
    summary = summarize(frame)
    comparison = _comparison_table(summary)
    if output is None:
        output = os.path.join(RESULTS_DIR, "phase5ab_pointlaw_pilot.parquet")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    frame.to_parquet(output, index=False)
    # also write pilot summary/comparison for inspection
    summary.to_parquet(output.replace(".parquet","_summary.parquet"), index=False)
    comparison.to_parquet(output.replace(".parquet","_comparison.parquet"), index=False)
    print(json.dumps({"pilot": True, "replications": replications, "families": list(families), "rows": len(frame), "output": output}, indent=2))
    return output

def run_fleet(
    *,
    families: Sequence[str] = CORE_FAMILIES,
    replications: int = GATE_REPLICATIONS,
    n_permutations: int = GATE_PERMUTATIONS,
    workers: int = 1,
    candidates: Sequence[str] = PRIMARY_CANDIDATES,
    cache_dir: Optional[str] = None,
) -> list[str]:
    cells = make_cells(families=families, n_grid=N_GRID, m_values=(PRIMARY_M,), d_values=(2,))
    # sequential cell scheduling to avoid memory blow-up
    paths = []
    for cell in cells:
        out = os.path.join(SHARD_DIR, f"phase5ab_pointlaw_{cell.cell_id}_rep0_{replications-1}.parquet")
        paths.append(run_shard(cell, rep_start=0, replications=replications, n_permutations=n_permutations, candidates=candidates, workers=workers, cache_dir=cache_dir, output=out))
    return paths

def aggregate(input_dir: str = SHARD_DIR, output_prefix: str = "phase5ab_pointlaw") -> dict:
    paths = sorted(glob.glob(os.path.join(input_dir, "phase5ab_pointlaw*.parquet")))
    if not paths:
        raise FileNotFoundError(f"no shards in {input_dir}")
    frames = [pd.read_parquet(p) for p in paths]
    frame = pd.concat(frames, ignore_index=True)
    key = ["design_hash","cell_id","method","replication"]
    dup = frame.duplicated(key, keep=False)
    if dup.any():
        dups = frame.loc[dup, key].drop_duplicates().to_dict("records")
        raise ValueError(f"duplicate keys: {dups[:3]}")
    if set(frame["design_hash"].dropna()) != {DESIGN_HASH}:
        raise ValueError("shards have conflicting design hash")
    summary = summarize(frame)
    comparison = _comparison_table(summary)
    # write
    os.makedirs(os.path.dirname(os.path.abspath(FINAL_REPLICATIONS)), exist_ok=True)
    frame.to_parquet(FINAL_REPLICATIONS, index=False)
    summary.to_parquet(FINAL_SUMMARY, index=False)
    comparison.to_parquet(FINAL_COMPARISON, index=False)
    _plot(summary, FINAL_FIGURE)
    _write_report(summary, comparison, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "phase5ab_pointlaw_report.md"), replications=int(frame["replication"].nunique()), n_permutations=int(frame["n_permutations"].iloc[0]) if len(frame) else GATE_PERMUTATIONS)
    manifest = {
        "benchmark_version": BENCHMARK_VERSION,
        "design_hash": DESIGN_HASH,
        "design_record": design_record(),
        "input_dir": input_dir,
        "shards": paths,
        "n_shards": len(paths),
        "replication_rows": int(len(frame)),
        "cell_replications": int(frame[["cell_id", "replication"]].drop_duplicates().shape[0]),
        "summary_rows": int(len(summary)),
        "comparison_rows": int(len(comparison)),
        "outputs": {"replications": FINAL_REPLICATIONS, "summary": FINAL_SUMMARY, "comparison": FINAL_COMPARISON, "figure": FINAL_FIGURE},
    }
    with open(FINAL_MANIFEST, "w", encoding="utf-8") as h:
        json.dump(manifest, h, indent=2, sort_keys=True)
    print(json.dumps(manifest, indent=2))
    return manifest


def profile_representative_cells(
    *,
    replications: int = 3,
    n_permutations: int = REPRO_PERMUTATIONS,
    candidates: Sequence[str] = ALL_CANDIDATES,
    output: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> dict:
    """Measure method cost on the five predeclared fleet-profile cells.

    Heavy methods are timed for one replication per cell because matching and
    SC-A can dominate a local profile.  Cheap methods use ``replications``;
    the resulting per-call median is used for fleet estimates.  The output is
    deliberately a machine-readable manifest consumed by the Colab generator.
    """
    if replications < 1 or n_permutations < 1:
        raise ValueError("replications and n_permutations must be positive")
    if output is None:
        output = os.path.join(RESULTS_DIR, "phase5ab_pointlaw_profile.json")
    observations = []
    for cell in PROFILE_CELLS:
        for candidate in tuple(candidates):
            n_reps = 1 if candidate in PROFILE_HEAVY_CANDIDATES else int(replications)
            elapsed = []
            method_runtime = []
            statuses = []
            failure_reasons = []
            rss = []
            for replication in range(n_reps):
                started = time.perf_counter()
                rows = _run_one((cell, replication, (candidate,), int(n_permutations), cache_dir))
                elapsed.append(time.perf_counter() - started)
                row = rows[0]
                statuses.append(str(row["status"]))
                if row.get("failure_reason"):
                    failure_reasons.append(str(row["failure_reason"]))
                if np.isfinite(row.get("runtime_seconds", np.nan)):
                    method_runtime.append(float(row["runtime_seconds"]))
                if int(row.get("peak_rss_bytes", -1)) >= 0:
                    rss.append(int(row["peak_rss_bytes"]))
            ok_times = method_runtime or elapsed
            per_call = float(np.median(ok_times))
            observation = {
                "cell_id": cell.cell_id,
                "family": cell.family,
                "n0": cell.n0,
                "n1": cell.n1,
                "d": cell.d,
                "m": cell.m,
                "method": candidate,
                "profile_replications": n_reps,
                "n_permutations": int(n_permutations),
                "per_call_seconds": per_call,
                "mean_call_seconds": float(np.mean(ok_times)),
                "median_call_seconds": per_call,
                "min_call_seconds": float(np.min(ok_times)),
                "max_call_seconds": float(np.max(ok_times)),
                "mean_method_runtime_seconds": float(np.mean(method_runtime)) if method_runtime else None,
                "mean_peak_rss_bytes": float(np.mean(rss)) if rss else None,
                "statuses": {status: statuses.count(status) for status in sorted(set(statuses))},
                "failure_reasons": sorted(set(failure_reasons)),
                "predicted_500_replications_minutes": per_call * GATE_REPLICATIONS / 60.0,
            }
            observations.append(observation)

    manifest = {
        "benchmark_version": BENCHMARK_VERSION,
        "design_hash": DESIGN_HASH,
        "profile_version": 1,
        "profile_cells": [cell.cell_id for cell in PROFILE_CELLS],
        "cheap_profile_replications": int(replications),
        "heavy_profile_replications": 1,
        "n_permutations": int(n_permutations),
        "observations": observations,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=True)
    print(json.dumps({"profile": output, "observations": len(observations)}, indent=2))
    return manifest

# ---------------------------------------------------------------------------
# CLI

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["pilot","shard","fleet","aggregate","profile"], default="pilot")
    p.add_argument("--replications", type=int, default=None)
    p.add_argument("--permutations", type=int, default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--families", default=None, help="comma-separated families")
    p.add_argument("--candidates", default=None, help="comma-separated candidates")
    p.add_argument("--n-grid", default=None, help="comma-separated n values")
    p.add_argument("--m-grid", default=None)
    p.add_argument("--d-grid", default=None)
    p.add_argument("--cell", default=None, help="cell_id for shard mode")
    p.add_argument("--rep-start", type=int, default=0)
    p.add_argument("--input-dir", default=SHARD_DIR)
    p.add_argument("--output", default=None)
    p.add_argument("--cache-dir", default=None)
    args = p.parse_args()

    # defaults per mode
    if args.mode == "pilot":
        reps = args.replications or PILOT_REPLICATIONS
        perms = args.permutations or PILOT_PERMUTATIONS
        fams = tuple(v.strip() for v in args.families.split(",")) if args.families else CORE_FAMILIES
        cands = tuple(v.strip() for v in args.candidates.split(",")) if args.candidates else PRIMARY_CANDIDATES
        n_grid = tuple(int(v) for v in args.n_grid.split(",")) if args.n_grid else (250,)
        frame = run_replicates(families=fams, n_grid=n_grid, m_values=(PRIMARY_M,), d_values=(2,), replications=reps, n_permutations=perms, workers=args.workers, candidates=cands, cache_dir=args.cache_dir)
        # write pilot
        out = args.output or os.path.join(RESULTS_DIR, "phase5ab_pointlaw_pilot.parquet")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        frame.to_parquet(out, index=False)
        summary = summarize(frame)
        comparison = _comparison_table(summary)
        summary.to_parquet(out.replace(".parquet","_summary.parquet"), index=False)
        comparison.to_parquet(out.replace(".parquet","_comparison.parquet"), index=False)
        _plot(summary, out.replace(".parquet",".png"))
        _write_report(summary, comparison, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "phase5ab_pointlaw_pilot_report.md"), replications=reps, n_permutations=perms)
        print(json.dumps({"mode":"pilot","replications":reps,"permutations":perms,"rows":len(frame),"output":out}, indent=2))
    elif args.mode == "shard":
        if not args.cell:
            raise SystemExit("--cell required for shard mode")
        # parse cell_id: format family_n{n0}_n1{n1}_m{m}_d{d}
        # reuse make_cells to find matching cell
        all_cells = make_cells(families=list(FAMILY_ROLE_EXT.keys()), n_grid=N_GRID + (10,15,20,25,30,50,75,100,125,150), m_values=M_GRID, d_values=D_GRID, include_robustness=True, include_dependence=True)
        # also handle four_atom family not in FAMILY_ROLE_EXT? already added
        cell = next((c for c in all_cells if c.cell_id == args.cell), None)
        if cell is None:
            # try to parse manually
            try:
                parts = args.cell.split("_")
                # naive parse
                family = parts[0]
                # find n0,n1,m,d
                n0 = int([x for x in parts if x.startswith("n")][0][1:])
                n1 = int([x for x in parts if x.startswith("n1")][0][2:])
                m = int([x for x in parts if x.startswith("m")][0][1:])
                d = int([x for x in parts if x.startswith("d")][0][1:])
                cell = Cell(family=family, n0=n0, n1=n1, m=m, d=d, role=FAMILY_ROLE_EXT.get(family,"unknown"), description=FAMILY_DESCRIPTION_EXT.get(family, family))
            except Exception as exc:
                raise SystemExit(f"unknown cell {args.cell!r}: {exc}")
        reps = args.replications or DEFAULT_SHARD_REPLICATIONS
        perms = args.permutations or GATE_PERMUTATIONS
        cands = tuple(v.strip() for v in args.candidates.split(",")) if args.candidates else PRIMARY_CANDIDATES
        run_shard(cell, rep_start=args.rep_start, replications=reps, n_permutations=perms, candidates=cands, workers=args.workers, cache_dir=args.cache_dir, output=args.output)
    elif args.mode == "fleet":
        reps = args.replications or GATE_REPLICATIONS
        perms = args.permutations or GATE_PERMUTATIONS
        fams = tuple(v.strip() for v in args.families.split(",")) if args.families else CORE_FAMILIES
        cands = tuple(v.strip() for v in args.candidates.split(",")) if args.candidates else PRIMARY_CANDIDATES
        run_fleet(families=fams, replications=reps, n_permutations=perms, workers=args.workers, candidates=cands, cache_dir=args.cache_dir)
    elif args.mode == "aggregate":
        aggregate(input_dir=args.input_dir)
    elif args.mode == "profile":
        reps = args.replications or 5
        perms = args.permutations or REPRO_PERMUTATIONS
        cands = tuple(v.strip() for v in args.candidates.split(",")) if args.candidates else ALL_CANDIDATES
        profile_representative_cells(replications=reps, n_permutations=perms, candidates=cands, output=args.output, cache_dir=args.cache_dir)
    else:
        raise SystemExit(f"unknown mode {args.mode}")

if __name__ == "__main__":
    main()
