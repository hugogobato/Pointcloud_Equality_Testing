"""Phase 6 high-dimensional point-law benchmark — raw vs PCA pre-filtering.

Extends Phase 5AB to high ambient dimensions requested by ecologist
(Morimoto: raw ~40 dims, PCA to 3-4 dims for inference).

Motivation
----------
* Phase 5AB fixed d=2. Ecology has many ambient dimensions before reduction.
* Hypothesis: for high ambient d, frozen-block topology methods (SC-B,
  HybridBlockMMD) and RawBlockMMD may retain power better than point-level
  Gaussian MMD / energy / kNN with fixed bandwidth, due to curse of
  dimensionality and bandwidth mis-calibration.
* Ecological practice does PCA first (3-4 PCs) then inference. Need to test
  when PCA helps vs when it destroys signal (PCA-failure construction where
  signal lives in low-variance subspace).

Locked design (Phase 6)
-----------------------
* alpha=0.05, m=25 locked (as in Phase 5AB) but K_a=2 at n=50 is coarse — reported as floor.
* n0=n1 in {50, 500} (per request)
* d in {10, 20, 40} (per request); also allow d=2 as sanity bridge.
* core replications 500 per cell, permutations 199 (common random).
* filtrations for SC-B/Hybrid: "vr" (GUDHI) by default to avoid missing ripser
  in Colab images; "ripser" attempted if available, fallback to vr.
* point bandwidth 0.30 fixed (as before) + median heuristic variant.
* bag kernel Hilbert-Gaussian point bw 0.10 bag bw 0.25 (locked).

DGPs
----
Reuses Phase 5AB extended DGPs with high-d embedding:
  _embed_cloud_high: 2D signal + independent Gaussian noise N(0, noise_scale)
  in extra dims. For core families noise_scale=0.25 (low-noise, per-dim var
  0.0625 comparable to signal Uniform var 1/12 ~0.083 — PCA still recovers signal).
  For PCA-failure families noise_scale=1.5 (var 2.25 >> signal) so top PCs are
  pure noise and PCA-3 discards signal.

Families:
  core (same as Phase 5AB, but embedded):
    iid_null, weak_barcode_null (translation), same_support_density,
    same_square_four_atom_density, topology_alt (disk vs circle)
  added sparse/dense shift for high-d sensitivity:
    sparse_shift_08  : shift +0.8 in first coordinate only (sparse)
    dense_shift_08   : shift +0.8/sqrt(d) in all d coordinates (dense, same L2)
  PCA-failure families (high-noise embedding):
    pca_fail_sparse_shift , pca_fail_topology
  Each is instantiated at d=10,20,40 but meaningful only at high d.

Pre-processing variants
-----------------------
Each (family, n, d) cloud is evaluated under three preproc modes:
  raw   : test on d-dim clouds directly
  pca3  : pooled PCA to 3 dims (fit on pooled unlabeled points, then project both arms)
  pca4  : pooled PCA to 4 dims (ecologia's 3-or-4)

Implementation: candidate names encode preproc, e.g. "PointMMD-Gaussian-PCA3"
or cell preproc suffix. Here we encode as method_variant suffix and add columns
`preproc`, `pca_k`, `pca_variance_explained`. For aggregation, cell_id includes
preproc.

Methods
-------
Same as Phase 5AB but without SC-A (abandoned per user) and without
Rosenbaum for large n pooled >500 (refused). Included:
  PointMMD-Gaussian (0.30), PointMMD-Gaussian-median, EnergyDistance,
  FriedmanRafsky-MST, Schilling-kNN k=1,5,10,
  SlicedWasserstein (100 proj), ClassifierTwoSample logistic/rf,
  RawBlockMMD (m=25), SC-B (vr, m=25), HybridBlockMMD a=0.50, SC-A-Block.

SC-A is intentionally omitted (expensive, not needed).

Usage
-----
Pilot (cheap, 20 reps, 39 perms):
  python experiments/phase6_highdim_pointlaw_tournament.py --mode pilot --replications 20 --permutations 39

Single cell shard (Colab):
  python experiments/phase6_highdim_pointlaw_tournament.py --mode shard --cell iid_null_n500_n500_m25_d40_preproc_raw --rep-start 0 --replications 25

Full fleet locally:
  python experiments/phase6_highdim_pointlaw_tournament.py --mode fleet --replications 500 --permutations 199 --workers 4

Aggregate:
  python experiments/phase6_highdim_pointlaw_tournament.py --mode aggregate --input-dir results/phase6_highdim_shards

Profile:
  python experiments/phase6_highdim_pointlaw_tournament.py --mode profile

"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from experiments.phase5_single_cloud_tournament import (
    FAMILY_DESCRIPTION,
    FAMILY_ROLE,
    _seed as _phase5_seed,
    make_cloud_pair,
    _project_cloud,
    ALPHA as PHASE5_ALPHA,
    BETTI_GRID,
    FILTRATION as PHASE5_FILTRATION,
    HOMOLOGY_DIMS,
    KERNEL_BANDWIDTH,
    MAX_SC_A_POINT_N,
)
from tda2s.tests.single_cloud import (
    DEFAULT_RAW_BAG_BANDWIDTH,
    DEFAULT_RAW_POINT_BANDWIDTH,
    REGIME_I,
    hybrid_block_mmd,
    raw_block_mmd,
    sc_a_blockwise_label_permutation,
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
# Frozen design
BENCHMARK_VERSION = "phase6-highdim-v1"
ALPHA = 0.05
SIZE_BAND = (0.03, 0.08)
MC_CONFIDENCE = 0.95
PRIMARY_M = 25
M_GRID = (25,)
N_GRID = (50, 500)
D_GRID = (10, 20, 40)
# also allow d=2 for sanity but not primary
D_GRID_ALL = (2, 10, 20, 40)
PRIMARY_D = 10
PRIMARY_N = 500
GATE_REPLICATIONS = 500
GATE_PERMUTATIONS = 199
REPRO_PERMUTATIONS = 39
PILOT_REPLICATIONS = 20
PILOT_PERMUTATIONS = 39
DEFAULT_SHARD_REPLICATIONS = 25
MAX_WORKERS = 16
SEED_ROOT = 20260826
PRIMARY_POINT_BANDWIDTH = 0.30
BANDWIDTH_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)
HYBRID_ALPHAS = (0.50,)
SCHILLING_KS = (1, 5, 10)
# Use VR via GUDHI by default (ripser missing in many envs)
PH_FILTRATION = "vr"
PCA_KS = (3, 4)
NOISE_SCALE_CORE = 0.25
NOISE_SCALE_PCA_FAIL = 1.5

CORE_FAMILIES = ("iid_null", "weak_barcode_null", "same_support_density", "same_square_four_atom_density", "topology_alt")
SPARSE_DENSE_FAMILIES = ("sparse_shift_08", "dense_shift_08")
PCA_FAIL_FAMILIES = ("pca_fail_sparse_shift", "pca_fail_topology")
# Primary benchmark families (as requested)
PRIMARY_FAMILIES = CORE_FAMILIES
# Extended families for PCA-failure exploration (high-noise embedding)
EXTENDED_FAMILIES = CORE_FAMILIES + SPARSE_DENSE_FAMILIES + PCA_FAIL_FAMILIES

FAMILY_ROLE_EXT = dict(FAMILY_ROLE)
FAMILY_ROLE_EXT.update({
    "same_square_four_atom_density": "target_mismatch",
    "sparse_shift_08": "power_sparse",
    "dense_shift_08": "power_dense",
    "pca_fail_sparse_shift": "pca_failure_sparse",
    "pca_fail_topology": "pca_failure_topology",
})
FAMILY_DESCRIPTION_EXT = dict(FAMILY_DESCRIPTION)
FAMILY_DESCRIPTION_EXT.update({
    "same_square_four_atom_density": "four-atom square p=(.25,.25,.25,.25) vs q=(.70,.10,.10,.10)",
    "sparse_shift_08": "sparse mean shift +0.8 in first coordinate only (high-d)",
    "dense_shift_08": "dense mean shift +0.8/sqrt(d) in all d coordinates (same L2 as sparse)",
    "pca_fail_sparse_shift": "sparse shift +1.0 in low-variance dim with high-noise background (noise_scale=1.5) — PCA discards signal",
    "pca_fail_topology": "disk vs circle in low-variance subspace (2 dims var 0.1) with d-2 noise dims var 2.25 — PCA discards topology",
})

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
SHARD_DIR = os.path.join(RESULTS_DIR, "phase6_highdim_shards")
FINAL_REPLICATIONS = os.path.join(RESULTS_DIR, "phase6_highdim_replications.parquet")
FINAL_SUMMARY = os.path.join(RESULTS_DIR, "phase6_highdim_summary.parquet")
FINAL_COMPARISON = os.path.join(RESULTS_DIR, "phase6_highdim_comparison.parquet")
FINAL_MANIFEST = os.path.join(RESULTS_DIR, "phase6_highdim_manifest.json")
FINAL_FIGURE = os.path.join(RESULTS_DIR, "phase6_highdim_comparison.png")

PRIMARY_CANDIDATES = (
    "PointMMD-Gaussian",
    "PointMMD-Gaussian-median",
    "EnergyDistance",
    "FriedmanRafsky-MST",
    "Schilling-kNN-k1",
    "RawBlockMMD",
    "SC-B",
    "HybridBlockMMD-a0.50",
)
SECONDARY_CANDIDATES = (
    "Schilling-kNN-k5",
    "Schilling-kNN-k10",
    "SlicedWasserstein",
    "ClassifierTwoSampleTest-logistic",
    "ClassifierTwoSampleTest-rf",
    "SC-A-Block",
)
ALL_CANDIDATES = PRIMARY_CANDIDATES + SECONDARY_CANDIDATES
# Rosenbaum only for small n (pooled n <=500)
ROSENBAUM_CANDIDATE = "Rosenbaum-CrossMatch"

PREPROC_MODES = ("raw", "pca3", "pca4")

def design_record() -> dict:
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "phase": "6-highdim",
        "regime": REGIME_I,
        "alpha": ALPHA,
        "size_band": list(SIZE_BAND),
        "confidence": MC_CONFIDENCE,
        "n_grid": list(N_GRID),
        "m_grid": list(M_GRID),
        "d_grid": list(D_GRID),
        "primary_m": PRIMARY_M,
        "primary_d": PRIMARY_D,
        "primary_n": PRIMARY_N,
        "filtration": PH_FILTRATION,
        "homology_dims": list(HOMOLOGY_DIMS),
        "kernel_bandwidth": KERNEL_BANDWIDTH,
        "primary_point_bandwidth": PRIMARY_POINT_BANDWIDTH,
        "bandwidth_multipliers": list(BANDWIDTH_MULTIPLIERS),
        "raw_point_bandwidth": DEFAULT_RAW_POINT_BANDWIDTH,
        "raw_bag_bandwidth": DEFAULT_RAW_BAG_BANDWIDTH,
        "hybrid_alphas": list(HYBRID_ALPHAS),
        "schilling_ks": list(SCHILLING_KS),
        "pca_ks": list(PCA_KS),
        "noise_scale_core": NOISE_SCALE_CORE,
        "noise_scale_pca_fail": NOISE_SCALE_PCA_FAIL,
        "gate_replications": GATE_REPLICATIONS,
        "gate_permutations": GATE_PERMUTATIONS,
        "seed_root": SEED_ROOT,
        "preproc_modes": list(PREPROC_MODES),
        "families_core": list(CORE_FAMILIES),
        "families_extended": list(EXTENDED_FAMILIES),
        "target_RawBlockMMD": "H0^law: P0=P1 (raw characteristic block, iid)",
        "target_SC-B": "H0,25^bar: Phi^25_0:1(P0)=Phi^25_0:1(P1)",
        "seed_convention": {
            "cloud": ["benchmark_version", "family", "n0", "n1", "dimension", "noise_scale", "replication"],
            "partition": ["benchmark_version", "cell_id", "replication"],
            "method": ["benchmark_version", "cell_id", "candidate", "preproc", "replication"],
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
    preproc: str  # raw, pca3, pca4
    role: str
    description: str

    @property
    def cell_id(self) -> str:
        return f"{self.family}_n{self.n0}_n1{self.n1}_m{self.m}_d{self.d}_preproc_{self.preproc}"

    @property
    def pca_k(self) -> int:
        if self.preproc == "raw":
            return 0
        if self.preproc == "pca3":
            return 3
        if self.preproc == "pca4":
            return 4
        raise ValueError(f"unknown preproc {self.preproc}")

    @property
    def noise_scale(self) -> float:
        if self.family in PCA_FAIL_FAMILIES:
            return NOISE_SCALE_PCA_FAIL
        return NOISE_SCALE_CORE

PROFILE_CELLS = (
    Cell("iid_null", 500, 500, PRIMARY_M, 10, "raw", "gating_null", "iid null"),
    Cell("same_support_density", 500, 500, PRIMARY_M, 10, "raw", "target_mismatch", "density"),
    Cell("topology_alt", 500, 500, PRIMARY_M, 10, "raw", "power", "topology"),
    Cell("iid_null", 50, 50, PRIMARY_M, 40, "raw", "gating_null", "small n high d"),
    Cell("pca_fail_sparse_shift", 500, 500, PRIMARY_M, 40, "raw", "pca_failure_sparse", "pca fail raw"),
    Cell("pca_fail_sparse_shift", 500, 500, PRIMARY_M, 40, "pca3", "pca_failure_sparse", "pca fail pca3"),
)

def _seed(*parts: object) -> int:
    payload = repr((SEED_ROOT,) + parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")

def _cloud_seed(cell: Cell, replication: int) -> int:
    return _seed(BENCHMARK_VERSION, "cloud", cell.family, cell.n0, cell.n1, cell.d, cell.noise_scale, replication)

def _partition_seed(cell: Cell, replication: int) -> int:
    return _seed(BENCHMARK_VERSION, "partition", cell.cell_id, replication)

def _method_seed(cell: Cell, replication: int, candidate: str) -> int:
    return _seed(BENCHMARK_VERSION, "method", cell.cell_id, candidate, replication)

def _embed_cloud_high(cloud: np.ndarray, d: int, noise_scale: float, seed: int) -> np.ndarray:
    cloud = np.asarray(cloud, dtype=float)
    if d == 2:
        return cloud
    if d < 2:
        raise ValueError("d must be >=2")
    n = len(cloud)
    rng = np.random.default_rng(seed)
    extra = rng.normal(0.0, float(noise_scale), size=(n, int(d) - 2))
    return np.concatenate([cloud, extra], axis=1)

def _four_atom_cloud(n: int, rng: np.random.Generator, p: np.ndarray) -> np.ndarray:
    square = np.asarray([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.1]])
    # Actually original uses [[0,0],[0,1],[1,0],[1,1]]; keep same
    square = np.asarray([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]])
    return square[rng.choice(4, size=int(n), p=p)]

def make_cloud_pair_highdim(family: str, n0: int, n1: int, seed: int, d: int = 2, noise_scale: float = NOISE_SCALE_CORE) -> tuple[np.ndarray, np.ndarray]:
    # PCA-failure families need special construction with variance heterogeneity
    if family == "pca_fail_sparse_shift":
        # Base: d dims, first d-1 dims high var (2.0), last dim low var (0.1)
        # Signal: shift +1.0 in last dim (low-var) — PCA to 3 dims will discard it.
        rng = np.random.default_rng(seed)
        # high var dims
        c0_high = rng.normal(0.0, 2.0, size=(n0, max(d-1,0)))
        c0_low = rng.normal(0.0, 0.1, size=(n0, 1)) if d>=1 else np.empty((n0,0))
        c0 = np.concatenate([c0_high, c0_low], axis=1) if d>1 else c0_low
        # c1 same distribution but shifted in last dim
        rng1 = np.random.default_rng(_seed("pca_fail_q", seed))
        c1_high = rng1.normal(0.0, 2.0, size=(n1, max(d-1,0)))
        c1_low = rng1.normal(0.0, 0.1, size=(n1, 1)) + 1.0 if d>=1 else np.empty((n1,0))
        c1 = np.concatenate([c1_high, c1_low], axis=1) if d>1 else c1_low
        # If d==2, this matches construction: 1 high var dim +1 low var dim
        return c0, c1
    if family == "pca_fail_topology":
        # Topology lives in last 2 low-var dims, noise in remaining high-var dims
        rng = np.random.default_rng(seed)
        # Generate disk vs circle in 2D as before, then embed into low-var subspace
        # disk/circle points are uniform-ish with radius 0.3 centered at 0.5, variance ~0.02
        # We'll generate them centered at 0, then scale to low var
        from experiments.phase5_single_cloud_tournament import _filled_disk, _noisy_circle
        # Use deterministic seeds split
        s0 = _seed("pca_fail_topo_c0", seed)
        s1 = _seed("pca_fail_topo_c1", seed)
        r0 = np.random.default_rng(s0)
        r1 = np.random.default_rng(s1)
        # generate base 2D signals
        def filled_disk_low(n, rng_):
            theta = rng_.uniform(0.0, 2*np.pi, size=int(n))
            radial = 0.3 * np.sqrt(rng_.uniform(0.0, 1.0, size=int(n)))
            pts = np.column_stack([radial * np.cos(theta), radial * np.sin(theta)])
            # scale to low variance: disk points already small variance ~0.02, keep as is
            return pts
        def noisy_circle_low(n, rng_):
            theta = rng_.uniform(0.0, 2*np.pi, size=int(n))
            pts = np.column_stack([np.cos(theta), np.sin(theta)]) * 0.3
            pts += rng_.normal(0.0, 0.008, size=pts.shape)
            return pts
        base0 = filled_disk_low(n0, r0)
        base1 = noisy_circle_low(n1, r1)
        # Embed: first d-2 dims are high-var noise, last 2 dims are topology signal
        noise0 = rng.normal(0.0, 2.0, size=(n0, max(d-2,0)))
        noise1 = np.random.default_rng(_seed("pca_fail_topo_noise1", seed)).normal(0.0, 2.0, size=(n1, max(d-2,0)))
        if d >= 2:
            c0 = np.concatenate([noise0, base0], axis=1)
            c1 = np.concatenate([noise1, base1], axis=1)
        elif d==1:
            # degenerate
            c0 = base0[:, :1]
            c1 = base1[:, :1]
        else:
            c0 = base0
            c1 = base1
        return c0, c1
    if family == "sparse_shift_08":
        # Base iid uniform 2D then sparse shift in first coordinate only
        rng = np.random.default_rng(seed)
        # Use uniform for base then embed with noise_scale
        # For P0: uniform square + noise
        # For P1: same but first coordinate shifted by +0.8
        c0_base = rng.uniform(0.0, 1.0, size=(n0, 2))
        rng1 = np.random.default_rng(_seed("sparse_shift_q", seed))
        c1_base = rng1.uniform(0.0, 1.0, size=(n1, 2))
        c1_base[:, 0] += 0.8
        c0 = _embed_cloud_high(c0_base, d, noise_scale, _seed("embed0", seed, d))
        c1 = _embed_cloud_high(c1_base, d, noise_scale, _seed("embed1", seed, d))
        return c0, c1
    if family == "dense_shift_08":
        # Dense shift: +0.8/sqrt(d) in all d coordinates
        rng = np.random.default_rng(seed)
        c0_base = rng.uniform(0.0, 1.0, size=(n0, 2))
        c0 = _embed_cloud_high(c0_base, d, noise_scale, _seed("embed0", seed, d))
        # For c1: same base uniform then shift in all d dims
        rng1 = np.random.default_rng(_seed("dense_shift_q", seed))
        c1_base = rng1.uniform(0.0, 1.0, size=(n1, 2))
        c1 = _embed_cloud_high(c1_base, d, noise_scale, _seed("embed1", seed, d))
        shift = 0.8 / math.sqrt(d)
        c1 = c1 + shift
        c0 = c0 + 0.0  # no shift
        return c0, c1
    if family == "same_square_four_atom_density":
        rng = np.random.default_rng(seed)
        p = np.full(4, 0.25)
        q = np.asarray([0.70, 0.10, 0.10, 0.10])
        c0 = _four_atom_cloud(n0, rng, p)
        rng1 = np.random.default_rng(_seed("four_atom_q", seed))
        c1 = _four_atom_cloud(n1, rng1, q)
        if d != 2:
            c0 = _embed_cloud_high(c0, d, noise_scale, _seed("embed0", seed, d))
            c1 = _embed_cloud_high(c1, d, noise_scale, _seed("embed1", seed, d))
        return c0, c1
    # default: use base DGP then embed
    c0, c1 = make_cloud_pair(family, n0, n1, seed)
    if d != 2:
        c0 = _embed_cloud_high(c0, d, noise_scale, _seed("embed0", seed, d))
        c1 = _embed_cloud_high(c1, d, noise_scale, _seed("embed1", seed, d))
    return c0, c1

def _pca_project_pooled(cloud0: np.ndarray, cloud1: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, dict]:
    if k <= 0:
        raise ValueError("k must be positive")
    pooled = np.vstack([cloud0, cloud1])
    # Fit PCA on pooled unlabeled data (valid under H0)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=int(k))
    pooled_p = pca.fit_transform(pooled)
    c0_p = pooled_p[:len(cloud0)]
    c1_p = pooled_p[len(cloud0):]
    info = {
        "pca_k": int(k),
        "pca_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pca_variance_explained": float(np.sum(pca.explained_variance_ratio_)),
        "pca_singular_values": pca.singular_values_.tolist() if hasattr(pca, "singular_values_") else [],
        "pca_components_shape": pca.components_.shape,
    }
    return c0_p, c1_p, info

def _mc_interval(successes: int, total: int, confidence: float = MC_CONFIDENCE):
    if total < 1 or successes < 0 or successes > total:
        raise ValueError("invalid binomial count")
    from scipy.stats import beta
    tail = (1.0 - confidence) / 2.0
    lower = 0.0 if successes == 0 else float(beta.ppf(tail, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1.0 - tail, successes + 1, total - successes))
    return lower, upper

def make_cells(
    families: Sequence[str] = PRIMARY_FAMILIES,
    n_grid: Sequence[int] = N_GRID,
    m_values: Sequence[int] = M_GRID,
    d_values: Sequence[int] = D_GRID,
    preproc_modes: Sequence[str] = PREPROC_MODES,
    include_extended: bool = False,
) -> list[Cell]:
    fams = list(families)
    if include_extended:
        for f in EXTENDED_FAMILIES:
            if f not in fams:
                fams.append(f)
    cells = []
    for family in fams:
        for n in n_grid:
            for m in m_values:
                for d in d_values:
                    for preproc in preproc_modes:
                        # For low d=2, pca may be same as raw when k>=d, but keep for completeness
                        # For pca failure families, raw and pca both meaningful
                        # For core families, include all preproc
                        cells.append(Cell(
                            family=family, n0=int(n), n1=int(n), m=int(m), d=int(d),
                            preproc=preproc,
                            role=FAMILY_ROLE_EXT.get(family, "unknown"),
                            description=FAMILY_DESCRIPTION_EXT.get(family, family),
                        ))
    return cells

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
    pca_info: Optional[dict] = None,
    status: str = "ok",
    failure_reason: str = "",
) -> dict:
    diagnostics = result.get("diagnostics", {}) if isinstance(result, dict) else {}
    target = result.get("inferential_target") or result.get("target_null") or result.get("target") or ""
    validity = result.get("validity_regime") or result.get("regime") or REGIME_I
    sampling_unit = result.get("sampling_unit") or diagnostics.get("sampling_unit") or ""
    is_block = candidate in ("RawBlockMMD","SC-B","SC-A-Block") or candidate.startswith("Hybrid")
    m_val = float(cell.m) if is_block else np.nan
    K0 = result.get("K0", np.nan)
    K1 = result.get("K1", np.nan)
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
            n0 = int(result.get("n0", cell.n0))
            n1 = int(result.get("n1", cell.n1))
            effective_total = int(n0 + n1)
            K0 = np.nan; K1 = np.nan
            unused0 = 0; unused1 = 0
            m_val = np.nan
    kernel_or_distance = result.get("kernel_or_distance") or result.get("kernel") or result.get("distance") or diagnostics.get("kernel") or diagnostics.get("distance") or ""
    bandwidth_or_tuning = result.get("bandwidth_or_tuning")
    if bandwidth_or_tuning is None:
        bandwidth_or_tuning = result.get("bandwidth", np.nan)
        if isinstance(bandwidth_or_tuning, dict):
            bandwidth_or_tuning = json.dumps(bandwidth_or_tuning)
    alpha_val = result.get("alpha", np.nan)
    if "alpha" in diagnostics:
        alpha_val = diagnostics["alpha"]
    if candidate.startswith("Hybrid"):
        try:
            alpha_val = float(candidate.split("a")[-1])
        except Exception:
            pass
    perm_group = result.get("permutation_group") or result.get("perm_group") or ""
    # PCA info
    if pca_info is None:
        pca_info = {}
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
        "cell_id": cell.cell_id,
        "replication": int(replication),
        "preproc": cell.preproc,
        "pca_k": int(cell.pca_k),
        "noise_scale": float(cell.noise_scale),
        "pca_variance_explained": float(pca_info.get("pca_variance_explained", np.nan)) if pca_info else np.nan,
        "pca_variance_ratio": json.dumps(pca_info.get("pca_variance_ratio", [])) if pca_info else "",
        "d_diagnostics": json.dumps({k: str(v) for k, v in diagnostics.items()}, sort_keys=True) if diagnostics else "",
        "pca_info": json.dumps(pca_info, sort_keys=True) if pca_info else "",
    }

def _call_method(
    cell: Cell,
    replication: int,
    candidate: str,
    *,
    n_permutations: int,
    cache_dir: Optional[str],
    cloud_pair: Optional[tuple[np.ndarray, np.ndarray]] = None,
    pca_info_holder: Optional[list] = None,
) -> tuple[dict, str]:
    cloud_seed = _cloud_seed(cell, replication)
    partition_seed = _partition_seed(cell, replication)
    method_seed = _method_seed(cell, replication, candidate)
    if cloud_pair is None:
        cloud_pair = make_cloud_pair_highdim(cell.family, cell.n0, cell.n1, cloud_seed, d=cell.d, noise_scale=cell.noise_scale)
    cloud0, cloud1 = cloud_pair
    # Apply preproc if needed
    pca_info = {}
    if cell.preproc != "raw":
        k = cell.pca_k
        # Edge: if d <= k, no reduction needed but still do PCA
        cloud0, cloud1, pca_info = _pca_project_pooled(cloud0, cloud1, k)
        if pca_info_holder is not None:
            pca_info_holder.append(pca_info)
    else:
        if pca_info_holder is not None:
            pca_info_holder.append({})

    # Choose PH filtration with fallback
    filt = PH_FILTRATION
    # If ripser requested but not available, fallback to vr is handled inside compute_diagrams import error at call time
    # We'll just use vr

    try:
        if candidate == "PointMMD-Gaussian":
            result = point_mmd_gaussian(cloud0, cloud1, regime=REGIME_I, bandwidth=PRIMARY_POINT_BANDWIDTH, n_perm=n_permutations, seed=method_seed)
            variant = f"bw{PRIMARY_POINT_BANDWIDTH}_{cell.preproc}"
        elif candidate == "PointMMD-Gaussian-median":
            result = point_mmd_gaussian(cloud0, cloud1, regime=REGIME_I, bandwidth=None, n_perm=n_permutations, seed=method_seed)
            variant = f"median_{cell.preproc}"
        elif candidate.startswith("PointMMD-Gaussian-bw"):
            try:
                bw = float(candidate.split("bw")[1].split("_")[0])
            except Exception:
                bw = PRIMARY_POINT_BANDWIDTH
            result = point_mmd_gaussian(cloud0, cloud1, regime=REGIME_I, bandwidth=bw, n_perm=n_permutations, seed=method_seed)
            variant = f"bw{bw}_{cell.preproc}"
        elif candidate == "EnergyDistance":
            result = energy_distance_test(cloud0, cloud1, regime=REGIME_I, n_perm=n_permutations, seed=method_seed)
            variant = cell.preproc
        elif candidate == "FriedmanRafsky-MST":
            result = friedman_rafsky_mst(cloud0, cloud1, regime=REGIME_I, n_perm=n_permutations, seed=method_seed)
            variant = cell.preproc
        elif candidate.startswith("Schilling-kNN"):
            if "-k" in candidate:
                try:
                    k = int(candidate.rsplit("-k", 1)[1].split("_")[0])
                except Exception:
                    k = 1
            else:
                k = 1
            result = schilling_knn(cloud0, cloud1, regime=REGIME_I, k=k, directed=True, n_perm=n_permutations, seed=method_seed)
            variant = f"k{k}_{cell.preproc}"
        elif candidate == "Rosenbaum-CrossMatch":
            result = rosenbaum_crossmatch(cloud0, cloud1, regime=REGIME_I, n_perm=n_permutations, seed=method_seed)
            variant = cell.preproc
        elif candidate == "SlicedWasserstein":
            result = sliced_wasserstein_test(cloud0, cloud1, regime=REGIME_I, n_projections=100, projection_seed=0, n_perm=n_permutations, seed=method_seed)
            variant = f"100proj_{cell.preproc}"
        elif candidate == "ClassifierTwoSampleTest-logistic":
            result = classifier_two_sample_test(cloud0, cloud1, regime=REGIME_I, classifier="logistic", test_fraction=0.5, split_seed=method_seed, n_perm=n_permutations, seed=method_seed)
            variant = f"logistic_{cell.preproc}"
        elif candidate == "ClassifierTwoSampleTest-rf":
            result = classifier_two_sample_test(cloud0, cloud1, regime=REGIME_I, classifier="rf", test_fraction=0.5, split_seed=method_seed, n_perm=n_permutations, seed=method_seed)
            variant = f"rf_{cell.preproc}"
        elif candidate == "RawBlockMMD":
            result = raw_block_mmd(cloud0, cloud1, regime=REGIME_I, m=cell.m, partition_seed=partition_seed, n_perm=n_permutations, seed=method_seed)
            variant = f"m{cell.m}_{cell.preproc}"
        elif candidate == "SC-B":
            # Try ripser if available else vr; we locked to vr
            result = sc_b_disjoint_mmd(cloud0, cloud1, regime=REGIME_I, m=cell.m, partition_seed=partition_seed, n_perm=n_permutations, seed=method_seed, filtration=filt, homology_dims=HOMOLOGY_DIMS, kernel_bandwidth=KERNEL_BANDWIDTH)
            variant = f"m{cell.m}_{cell.preproc}"
        elif candidate.startswith("HybridBlockMMD"):
            try:
                alpha = float(candidate.split("a")[1].split("_")[0])
            except Exception:
                alpha = 0.5
            result = hybrid_block_mmd(cloud0, cloud1, regime=REGIME_I, m=cell.m, alpha=alpha, partition_seed=partition_seed, n_perm=n_permutations, seed=method_seed, filtration=filt, homology_dims=HOMOLOGY_DIMS, barcode_kernel_bandwidth=KERNEL_BANDWIDTH)
            variant = f"a{alpha:.2f}_{cell.preproc}"
        elif candidate == "SC-A-Block":
            result = sc_a_blockwise_label_permutation(cloud0, cloud1, regime=REGIME_I, m=cell.m, partition_seed=partition_seed, n_perm=n_permutations, seed=method_seed)
            variant = f"m{cell.m}_{cell.preproc}"
        else:
            raise ValueError(f"unknown candidate {candidate!r}")
        return result, variant, pca_info
    except Exception as exc:
        raise exc

def _run_one(args) -> list[dict]:
    cell, replication, candidates, n_permutations, cache_dir = args
    rows = []
    cloud_seed = _cloud_seed(cell, replication)
    partition_seed = _partition_seed(cell, replication)
    cloud_pair = make_cloud_pair_highdim(cell.family, cell.n0, cell.n1, cloud_seed, d=cell.d, noise_scale=cell.noise_scale)
    # For PCA preproc, we will transform inside _call_method per candidate but sharing same pooled PCA
    # To ensure same PCA projection for all candidates within replication, we need to compute PCA once per replication
    # So handle preproc caching here:
    pca_cache = {}
    # If cell.preproc != raw, compute PCA once
    if cell.preproc != "raw":
        # compute PCA projection for this replication's clouds (shared)
        k = cell.pca_k
        cloud0_raw, cloud1_raw = cloud_pair
        # Use deterministic PCA (already deterministic)
        c0_p, c1_p, pca_info = _pca_project_pooled(cloud0_raw, cloud1_raw, k)
        # Replace cloud_pair for this cell's candidates
        cloud_pair = (c0_p, c1_p)
        # store pca_info for rows
        cached_pca_info = pca_info
    else:
        cached_pca_info = {}

    for candidate in candidates:
        method_seed = _method_seed(cell, replication, candidate)
        try:
            # _call_method will re-apply PCA if we pass raw clouds, but we have already transformed
            # So we pass transformed cloud_pair and tell _call_method to not re-transform
            # We can call directly without going through _call_method's PCA branch by passing pre-transformed
            # Simpler: call _call_method with cloud_pair already transformed and temporarily set cell.preproc to raw for method internal?
            # Instead we create a helper that bypasses PCA re-application: we will inline dispatch here
            # For simplicity, we replicate dispatch without double PCA
            # So we call a special internal that assumes clouds already preproc'd
            # We'll directly dispatch using the already transformed clouds

            # Inline dispatch (duplicate of _call_method but without PCA)
            cloud0, cloud1 = cloud_pair
            filt = PH_FILTRATION
            pca_info = cached_pca_info if cell.preproc != "raw" else {}
            # dispatch
            if candidate == "PointMMD-Gaussian":
                result = point_mmd_gaussian(cloud0, cloud1, regime=REGIME_I, bandwidth=PRIMARY_POINT_BANDWIDTH, n_perm=n_permutations, seed=method_seed)
                variant = f"bw{PRIMARY_POINT_BANDWIDTH}_{cell.preproc}"
            elif candidate == "PointMMD-Gaussian-median":
                result = point_mmd_gaussian(cloud0, cloud1, regime=REGIME_I, bandwidth=None, n_perm=n_permutations, seed=method_seed)
                variant = f"median_{cell.preproc}"
            elif candidate == "EnergyDistance":
                result = energy_distance_test(cloud0, cloud1, regime=REGIME_I, n_perm=n_permutations, seed=method_seed)
                variant = cell.preproc
            elif candidate == "FriedmanRafsky-MST":
                result = friedman_rafsky_mst(cloud0, cloud1, regime=REGIME_I, n_perm=n_permutations, seed=method_seed)
                variant = cell.preproc
            elif candidate.startswith("Schilling-kNN"):
                if "-k" in candidate:
                    try:
                        k = int(candidate.rsplit("-k", 1)[1].split("_")[0])
                    except Exception:
                        k = 1
                else:
                    k = 1
                result = schilling_knn(cloud0, cloud1, regime=REGIME_I, k=k, directed=True, n_perm=n_permutations, seed=method_seed)
                variant = f"k{k}_{cell.preproc}"
            elif candidate == "Rosenbaum-CrossMatch":
                result = rosenbaum_crossmatch(cloud0, cloud1, regime=REGIME_I, n_perm=n_permutations, seed=method_seed)
                variant = cell.preproc
            elif candidate == "SlicedWasserstein":
                result = sliced_wasserstein_test(cloud0, cloud1, regime=REGIME_I, n_projections=100, projection_seed=0, n_perm=n_permutations, seed=method_seed)
                variant = f"100proj_{cell.preproc}"
            elif candidate == "ClassifierTwoSampleTest-logistic":
                result = classifier_two_sample_test(cloud0, cloud1, regime=REGIME_I, classifier="logistic", test_fraction=0.5, split_seed=method_seed, n_perm=n_permutations, seed=method_seed)
                variant = f"logistic_{cell.preproc}"
            elif candidate == "ClassifierTwoSampleTest-rf":
                result = classifier_two_sample_test(cloud0, cloud1, regime=REGIME_I, classifier="rf", test_fraction=0.5, split_seed=method_seed, n_perm=n_permutations, seed=method_seed)
                variant = f"rf_{cell.preproc}"
            elif candidate == "RawBlockMMD":
                result = raw_block_mmd(cloud0, cloud1, regime=REGIME_I, m=cell.m, partition_seed=partition_seed, n_perm=n_permutations, seed=method_seed)
                variant = f"m{cell.m}_{cell.preproc}"
            elif candidate == "SC-B":
                result = sc_b_disjoint_mmd(cloud0, cloud1, regime=REGIME_I, m=cell.m, partition_seed=partition_seed, n_perm=n_permutations, seed=method_seed, filtration=filt, homology_dims=HOMOLOGY_DIMS, kernel_bandwidth=KERNEL_BANDWIDTH)
                variant = f"m{cell.m}_{cell.preproc}"
            elif candidate.startswith("HybridBlockMMD"):
                try:
                    alpha = float(candidate.split("a")[1].split("_")[0])
                except Exception:
                    alpha = 0.5
                result = hybrid_block_mmd(cloud0, cloud1, regime=REGIME_I, m=cell.m, alpha=alpha, partition_seed=partition_seed, n_perm=n_permutations, seed=method_seed, filtration=filt, homology_dims=HOMOLOGY_DIMS, barcode_kernel_bandwidth=KERNEL_BANDWIDTH)
                variant = f"a{alpha:.2f}_{cell.preproc}"
            elif candidate == "SC-A-Block":
                result = sc_a_blockwise_label_permutation(cloud0, cloud1, regime=REGIME_I, m=cell.m, partition_seed=partition_seed, n_perm=n_permutations, seed=method_seed)
                variant = f"m{cell.m}_{cell.preproc}"
            else:
                raise ValueError(f"unknown candidate {candidate!r}")

            rec = _unified_record(cell=cell, replication=replication, candidate=candidate, method_variant=variant, result=result, cloud_seed=cloud_seed, partition_seed=partition_seed if candidate in ("RawBlockMMD","SC-B","HybridBlockMMD-a0.50","SC-A-Block") or candidate.startswith("Hybrid") else None, permutation_seed=method_seed, pca_info=pca_info, status="ok")
            rec["rejected"] = bool(rec["pvalue"] <= ALPHA) if np.isfinite(rec["pvalue"]) else False
            rows.append(rec)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            status = "failed"
            if any(k in str(exc).lower() for k in ["exceeds", "no barcode block", "m must be", "k must be", "cloud must contain", "pooled n", "refused", "k="]):
                status = "refused"
            rec = _unified_record(cell=cell, replication=replication, candidate=candidate, method_variant="", result={"statistic": np.nan, "pvalue": np.nan, "n_permutations": -1, "exact_enumeration": False, "permutation_group": "", "diagnostics": {}, "kernel": "", "n0": cell.n0, "n1": cell.n1}, cloud_seed=cloud_seed, partition_seed=partition_seed if candidate in ("RawBlockMMD","SC-B","HybridBlockMMD-a0.50","SC-A-Block") or candidate.startswith("Hybrid") else None, permutation_seed=method_seed, pca_info=cached_pca_info if cell.preproc != "raw" else {}, status=status, failure_reason=msg)
            rows.append(rec)
    return rows

def run_replicates(
    *,
    families: Sequence[str] = PRIMARY_FAMILIES,
    n_grid: Sequence[int] = N_GRID,
    m_values: Sequence[int] = M_GRID,
    d_values: Sequence[int] = D_GRID,
    preproc_modes: Sequence[str] = ("raw",),
    replications: int = PILOT_REPLICATIONS,
    n_permutations: int = PILOT_PERMUTATIONS,
    workers: int = 1,
    candidates: Sequence[str] = PRIMARY_CANDIDATES,
    cache_dir: Optional[str] = None,
    include_extended: bool = False,
) -> pd.DataFrame:
    if replications < 1 or n_permutations < 1:
        raise ValueError("replications and n_permutations must be positive")
    if workers < 1 or workers > MAX_WORKERS:
        raise ValueError(f"workers must be in [1,{MAX_WORKERS}]")
    cells = make_cells(families=families, n_grid=n_grid, m_values=m_values, d_values=d_values, preproc_modes=preproc_modes, include_extended=include_extended)
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
    groups = ["method","family","family_role","n0","n1","m","d","preproc","pca_k","noise_scale","alpha","kernel_or_distance","target_null","sampling_unit","validity_regime"]
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
            "mean_pca_variance_explained": float(ok["pca_variance_explained"].mean()) if total and "pca_variance_explained" in ok else np.nan,
        })
        rows.append(values)
    return pd.DataFrame(rows)

def _comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    # Headline: primary n=500, d=10, raw
    primary = summary[
        (summary["n0"] == PRIMARY_N)
        & (summary["n1"] == PRIMARY_N)
        & (summary["d"] == PRIMARY_D)
        & (summary["preproc"] == "raw")
    ]
    keep = [c for c in ALL_CANDIDATES if c in summary["method"].unique()]
    primary = primary[primary["method"].isin(keep)]
    rows = []
    for method in sorted(primary["method"].unique()):
        sub = primary[primary["method"]==method]
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
            "preproc": "raw",
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
            "sparse_shift_power": rate("sparse_shift_08"),
            "dense_shift_power": rate("dense_shift_08"),
            "mean_runtime": float(sub["mean_runtime_seconds"].mean()) if len(sub) else np.nan,
            "mean_peak_rss": float(sub["mean_peak_rss_bytes"].mean()) if len(sub) else np.nan,
        })
    return pd.DataFrame(rows)

def _plot(summary: pd.DataFrame, output: str) -> None:
    mpl_config = os.path.join("/tmp", "tda2s_phase6_mplconfig")
    os.makedirs(mpl_config, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_config)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Filter to primary n=500 for headline
    primary = summary[
        (summary["n0"] == PRIMARY_N)
        & (summary["n1"] == PRIMARY_N)
        & (summary["preproc"] == "raw")
    ].copy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0,0]
    calibr = primary[primary["family"].isin(["iid_null","weak_barcode_null"])]
    methods = sorted(summary["method"].unique())
    colors = {"PointMMD-Gaussian":"#4477AA","EnergyDistance":"#228833","FriedmanRafsky-MST":"#CC6677","Schilling-kNN-k1":"#AA3377","RawBlockMMD":"#66CCEE","SC-B":"#228833","SlicedWasserstein":"#888888","ClassifierTwoSampleTest-logistic":"#CC3311","ClassifierTwoSampleTest-rf":"#EE7733"}
    for method in methods:
        sub = calibr[calibr["method"]==method].sort_values("d")
        if sub.empty:
            continue
        x = np.arange(len(sub))
        ax.errorbar(x, sub["rejection_rate"], yerr=[sub["rejection_rate"]-sub["mc_low"], sub["mc_high"]-sub["rejection_rate"]], marker="o", capsize=3, label=method, color=colors.get(method, None), linewidth=1)
    ax.axhspan(*SIZE_BAND, color="#66AA55", alpha=0.12)
    ax.axhline(ALPHA, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(np.arange(len(primary["d"].unique())), [str(d) for d in sorted(primary["d"].unique())])
    ax.set_ylim(0,1)
    ax.set_ylabel("rejection rate")
    ax.set_title(f"Calibration vs d (raw, n={PRIMARY_N})")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.2)
    ax = axes[0,1]
    power = primary[primary["family"].isin(["same_support_density","topology_alt"])]
    for method in methods:
        sub = power[power["method"]==method].sort_values("d")
        if sub.empty:
            continue
        # For simplicity plot density vs topology averaged? We'll plot density only
        sub_d = sub[sub["family"]=="same_support_density"].sort_values("d")
        if len(sub_d):
            ax.plot(np.arange(len(sub_d)), sub_d["rejection_rate"].values, marker="o", label=method)
    ax.set_xticks(np.arange(len(primary["d"].unique())), [str(d) for d in sorted(primary["d"].unique())])
    ax.set_ylim(0,1)
    ax.set_ylabel("rejection rate")
    ax.set_title("Density power vs d (raw)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.2)
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
        "# Phase 6 high-dimensional point-law benchmark report",
        "",
        f"Benchmark version `{BENCHMARK_VERSION}`, design hash `{DESIGN_HASH}`. Replications per cell `{replications}`, permutations `{n_permutations}`.",
        "",
        "## Design audit",
        "",
        f"Frozen protocol: m={PRIMARY_M} locked; d in {list(D_GRID)}; n in {list(N_GRID)}; preproc in {list(PREPROC_MODES)}; PH filtration {PH_FILTRATION}; point bandwidth {PRIMARY_POINT_BANDWIDTH}; bag bandwidths {DEFAULT_RAW_POINT_BANDWIDTH}/{DEFAULT_RAW_BAG_BANDWIDTH}. SC-A omitted (expensive, not needed).",
        "",
        "## Method registry (headline comparison distinguishes target)",
        "",
        "| method | target_null | sampling_unit | null_rejection | density | topology | translated | 4-atom | sparse | dense | runtime | peak_RSS |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in comparison.iterrows():
        lines.append(f"| {row['method']} | {row['target_null']} | {row['sampling_unit']} | {row['null_rejection_rate']:.3f} | {row['density_power']:.3f} | {row['topology_power']:.3f} | {row['translated_pointlaw_power_or_barcode_null']:.3f} | {row['four_atom_power']:.3f} | {row['sparse_shift_power']:.3f} | {row['dense_shift_power']:.3f} | {row['mean_runtime']:.3f} | {row['mean_peak_rss']:.0f} |")
    lines.extend([
        "",
        "## Gates (B1-B8) adaptation for high-d",
        "",
        "B1 target/literature validity: passes by construction.",
        "B2 iid size: primary H0^law methods should lie in [0.03,0.08] at 500 replications.",
        "B3 point-law power vs d: evaluate dense vs sparse power decay with ambient dimension.",
        "B4 topology retention vs d and after PCA.",
        "B5 PCA separation: compare raw vs pca3 vs pca4 power; pca failure families should show collapse under pca3.",
        "B6 small-sample honesty: m=25 at n=50 (K=2) is floor; report p-value grid.",
        "B7 robustness: high-noise and pca failure diagnostics.",
        "B8 computation: runtime, peak RSS per d.",
        "",
        "## Effective sample size note",
        "",
        "Point-level methods use n0+n1 observations. Block methods use K0+K1 blocks; m=25 locked.",
        "",
        "## PCA note",
        "",
        "PCA fitted on pooled unlabeled points (valid under H0). Variance explained reported per replication. pca_fail families embed signal in low-var subspace so pca3 should lose power while raw retains some.",
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
        output = os.path.join(SHARD_DIR, f"phase6_highdim_{cell.cell_id}_rep{rep_start}_{rep_start+replications-1}.parquet")
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
    n_grid: Sequence[int] = (500,),
    d_grid: Sequence[int] = (10,),
    preproc_modes: Sequence[str] = ("raw",),
    output: Optional[str] = None,
) -> str:
    frame = run_replicates(families=families, n_grid=n_grid, m_values=M_GRID, d_values=d_grid, preproc_modes=preproc_modes, replications=replications, n_permutations=n_permutations, workers=workers, candidates=candidates, include_extended=False)
    summary = summarize(frame)
    comparison = _comparison_table(summary)
    if output is None:
        output = os.path.join(RESULTS_DIR, "phase6_highdim_pilot.parquet")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    frame.to_parquet(output, index=False)
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
    preproc_modes: Sequence[str] = ("raw", "pca3"),
    d_grid: Sequence[int] = D_GRID,
    n_grid: Sequence[int] = N_GRID,
    include_extended: bool = False,
    cache_dir: Optional[str] = None,
) -> list[str]:
    cells = make_cells(families=families, n_grid=n_grid, m_values=M_GRID, d_values=d_grid, preproc_modes=preproc_modes, include_extended=include_extended)
    paths = []
    for cell in cells:
        out = os.path.join(SHARD_DIR, f"phase6_highdim_{cell.cell_id}_rep0_{replications-1}.parquet")
        paths.append(run_shard(cell, rep_start=0, replications=replications, n_permutations=n_permutations, candidates=candidates, workers=workers, cache_dir=cache_dir, output=out))
    return paths

def aggregate(input_dir: str = SHARD_DIR, output_prefix: str = "phase6_highdim") -> dict:
    paths = sorted(glob.glob(os.path.join(input_dir, "phase6_highdim*.parquet")))
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
    os.makedirs(os.path.dirname(os.path.abspath(FINAL_REPLICATIONS)), exist_ok=True)
    frame.to_parquet(FINAL_REPLICATIONS, index=False)
    summary.to_parquet(FINAL_SUMMARY, index=False)
    comparison.to_parquet(FINAL_COMPARISON, index=False)
    _plot(summary, FINAL_FIGURE)
    _write_report(summary, comparison, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "phase6_highdim_report.md"), replications=int(frame["replication"].nunique()), n_permutations=int(frame["n_permutations"].iloc[0]) if len(frame) else GATE_PERMUTATIONS)
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
    if replications < 1 or n_permutations < 1:
        raise ValueError("replications and n_permutations must be positive")
    if output is None:
        output = os.path.join(RESULTS_DIR, "phase6_highdim_profile.json")
    observations = []
    for cell in PROFILE_CELLS:
        for candidate in tuple(candidates):
            n_reps = 1 if candidate in {"ClassifierTwoSampleTest-rf", "SlicedWasserstein"} else int(replications)
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
                "preproc": cell.preproc,
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
        "n_permutations": int(n_permutations),
        "observations": observations,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=True)
    print(json.dumps({"profile": output, "observations": len(observations)}, indent=2))
    return manifest

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["pilot","shard","fleet","aggregate","profile"], default="pilot")
    p.add_argument("--replications", type=int, default=None)
    p.add_argument("--permutations", type=int, default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--families", default=None, help="comma-separated families")
    p.add_argument("--candidates", default=None, help="comma-separated candidates")
    p.add_argument("--n-grid", default=None, help="comma-separated n values")
    p.add_argument("--d-grid", default=None, help="comma-separated d values")
    p.add_argument("--preproc", default=None, help="comma-separated preproc modes raw,pca3,pca4")
    p.add_argument("--cell", default=None, help="cell_id for shard mode")
    p.add_argument("--rep-start", type=int, default=0)
    p.add_argument("--input-dir", default=SHARD_DIR)
    p.add_argument("--output", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--include-extended", action="store_true", help="include sparse/dense and pca_fail families")
    args = p.parse_args()
    if args.mode == "pilot":
        reps = args.replications or PILOT_REPLICATIONS
        perms = args.permutations or PILOT_PERMUTATIONS
        fams = tuple(v.strip() for v in args.families.split(",")) if args.families else CORE_FAMILIES
        cands = tuple(v.strip() for v in args.candidates.split(",")) if args.candidates else PRIMARY_CANDIDATES
        n_grid = tuple(int(v) for v in args.n_grid.split(",")) if args.n_grid else (500,)
        d_grid = tuple(int(v) for v in args.d_grid.split(",")) if args.d_grid else (10,)
        preproc = tuple(v.strip() for v in args.preproc.split(",")) if args.preproc else ("raw",)
        frame = run_replicates(families=fams, n_grid=n_grid, m_values=M_GRID, d_values=d_grid, preproc_modes=preproc, replications=reps, n_permutations=perms, workers=args.workers, candidates=cands, cache_dir=args.cache_dir, include_extended=args.include_extended)
        out = args.output or os.path.join(RESULTS_DIR, "phase6_highdim_pilot.parquet")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        frame.to_parquet(out, index=False)
        summary = summarize(frame)
        comparison = _comparison_table(summary)
        summary.to_parquet(out.replace(".parquet","_summary.parquet"), index=False)
        comparison.to_parquet(out.replace(".parquet","_comparison.parquet"), index=False)
        _plot(summary, out.replace(".parquet",".png"))
        _write_report(summary, comparison, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "phase6_highdim_pilot_report.md"), replications=reps, n_permutations=perms)
        print(json.dumps({"mode":"pilot","replications":reps,"permutations":perms,"rows":len(frame),"output":out}, indent=2))
    elif args.mode == "shard":
        if not args.cell:
            raise SystemExit("--cell required for shard mode")
        all_cells = make_cells(families=list(FAMILY_ROLE_EXT.keys()), n_grid=N_GRID + (50,), m_values=M_GRID, d_values=D_GRID_ALL, preproc_modes=PREPROC_MODES, include_extended=True)
        cell = next((c for c in all_cells if c.cell_id == args.cell), None)
        if cell is None:
            try:
                # parse manually: family_n{n0}_n1{n1}_m{m}_d{d}_preproc_{preproc}
                # family may contain underscores
                # Find known family prefix
                found = None
                for fam in list(FAMILY_ROLE_EXT.keys()) + list(EXTENDED_FAMILIES):
                    if args.cell.startswith(fam + "_"):
                        found = fam
                        break
                if found is None:
                    raise ValueError("unknown family prefix")
                rest = args.cell[len(found)+1:]
                # rest format: n{n0}_n1{n1}_m{m}_d{d}_preproc_{preproc}
                import re
                m = re.search(r"_m(\d+)_", rest)
                d = re.search(r"_d(\d+)_", rest)
                n0 = re.search(r"n(\d+)_n1", rest)
                n1 = re.search(r"n1(\d+)_", rest)
                pre = re.search(r"preproc_(\w+)$", rest)
                if not all([m,d,n0,n1,pre]):
                    raise ValueError("parse failed")
                cell = Cell(family=found, n0=int(n0.group(1)), n1=int(n1.group(1)), m=int(m.group(1)), d=int(d.group(1)), preproc=pre.group(1), role=FAMILY_ROLE_EXT.get(found,"unknown"), description=FAMILY_DESCRIPTION_EXT.get(found, found))
            except Exception as exc:
                raise SystemExit(f"unknown cell {args.cell!r}: {exc}")
        reps = args.replications or DEFAULT_SHARD_REPLICATIONS
        perms = args.permutations or GATE_PERMUTATIONS
        cands = tuple(v.strip() for v in args.candidates.split(",")) if args.candidates else ALL_CANDIDATES
        run_shard(cell, rep_start=args.rep_start, replications=reps, n_permutations=perms, candidates=cands, workers=args.workers, cache_dir=args.cache_dir, output=args.output)
    elif args.mode == "fleet":
        reps = args.replications or GATE_REPLICATIONS
        perms = args.permutations or GATE_PERMUTATIONS
        fams = tuple(v.strip() for v in args.families.split(",")) if args.families else PRIMARY_FAMILIES
        cands = tuple(v.strip() for v in args.candidates.split(",")) if args.candidates else ALL_CANDIDATES
        d_grid = tuple(int(v) for v in args.d_grid.split(",")) if args.d_grid else D_GRID
        n_grid = tuple(int(v) for v in args.n_grid.split(",")) if args.n_grid else N_GRID
        preproc = tuple(v.strip() for v in args.preproc.split(",")) if args.preproc else ("raw","pca3")
        run_fleet(families=fams, replications=reps, n_permutations=perms, workers=args.workers, candidates=cands, preproc_modes=preproc, d_grid=d_grid, n_grid=n_grid, include_extended=args.include_extended, cache_dir=args.cache_dir)
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
