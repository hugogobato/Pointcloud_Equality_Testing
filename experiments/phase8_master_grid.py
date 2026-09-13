"""Phase 8 bake-off: replicated-cloud master grid (C4 empirical).

Replicated-cloud panel ONLY (one cloud = one unit). The single-cloud panel
(task 8.2's SC-A/SC-B/SC-C columns) is OUT OF SCOPE for this phase owner and
is skipped with a logged skip, never silently.

Grid
----
tests x imbalance x alternative-family x n, with 100-rep coarse everywhere
first and 500 reps only on gating cells (two-stage, RESEARCH_PLAN §5):

* tests: rt, mmd, han, strand, moon_lazar, frechet_anova (all H0^cond),
  krebs_rademacher (dispersion null; run on dispersion-shift + null cells
  only, logged skip elsewhere), dr_mult + dr_perm (H0^out, primary),
  dist_maxt (H0^dist-grid, production Phase 4.5 calibration),
  dist_mmd (H0^dist secondary, universal kernel), dr_equiv (null cells only).
* imbalance: randomized (lambda=0.0), mild (lambda=0.5), strong (lambda=1.0),
  where propensity = expit(1.5 * lambda * X @ beta) (Phase 2 convention).
* families: null (size), mean (loop-count shift), dispersion (loop-count /
  radius mixture, KR's turf), w1split (stochastic cluster splitting,
  H0^out approx true / H0^dist false), rare (rare extra loop).
* n (total clouds): 50, 100, 200, 500.

DGP reuse (no reinvented generators): CloudSampleDGP / loops_cloud /
split_cluster_cloud from tda2s.dgp (Phase 2 / Phase 4.5 recipes).

Discipline enforced here
------------------------
* No reimplemented AIPW: DR goes through tda2s.tests.dr_outcome.fit_dr
  (tcda_uq cross_fit underneath); dist-level through tda2s.tests.dist_level.
* Diagrams computed ONCE per replication; every test after that permutes
  cloud labels (never recomputes PH in a calibration loop). In-process
  diagram cache keyed by (cloud sha1, filtration, params).
* Hazard fix 1: exactly-equal diagram pairs short-circuit to distance 0 in
  the RT bottleneck matrix (CGAL degenerate path).
* Hazard fix 2: ONE fixed global persistence-measure grid
  (interval (0,2), 32 bins, R=3) for every dist-level call.
* Every reported cell carries reps + MC SE. Failed runs are retained as
  rows with ok=False, never hidden.

Modes
-----
* --smoke: tiny grid (<5 min), exercises every code path incl. skips.
* --coarse: 100 reps on every grid cell (the long pole; 16 workers max).
* --full: 500 reps on gating cells only (default: strongest-imbalance
  mean-shift column + W1 cluster-splitting column; override with --families
  / --imbalances / --ns).
* --robust: 100-rep robustness panel at n=100 (8.5).
* --aggregate: merge per-cell parquet shards into results/phase8_master.
* --figure: size/power panels by imbalance + cost panel.
* --cost: per-test wall-time / peak-RSS probe at n=200 (feeds Table 8.4).

Shard by alternative family: --families null,mean,dispersion,w1split,rare
(or a subset); each cell writes results/phase8_shards/cell_*.parquet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
import traceback

import numpy as np

# --------------------------------------------------------------------------
# frozen Phase 8 contract (chosen before the first shard; recorded here so
# the aggregate can refuse shards run under a different contract)
# --------------------------------------------------------------------------
BASE_SEED = 8100
ALPHA = 0.05
M_POINTS = 60               # points per cloud (loops_cloud needs >= 8/loop)
FILTRATION = "alpha"
HOM_DIMS = (0, 1)
SIL_INTERVAL = (0.0, 2.0)   # fixed global silhouette grid
SIL_R = 3.0
SIL_RES = 50
N_BASIS = 5
N_FOLDS = 2
N_PERM_COARSE = 199         # permutation draws, competitors (coarse/full)
N_PERM_SMOKE = 19
DR_DRAWS_COARSE = 399       # DR multiplier draws (Phase 3 convention)
DR_DRAWS_SMOKE = 39
DIST_DRAWS = 1999           # frozen Phase 4.5 contract (production cells)
DIST_DRAWS_SMOKE = 99
EPS_RT = 0.1                # near-diagonal filter for RT/MMD/Han kernels
                              # (Phase 2 value; all grid signals have
                              # persistence >= ~0.26, an order above the cut)
RT_APPROX = 0.01            # gudhi additive tolerance (Phase 2 convention)
PROP_SCALE = 1.5
BETA = np.array([-0.5, -0.1, 0.6])
IMBALANCES = {"randomized": 0.0, "mild": 0.5, "strong": 1.0}
FAMILIES = ("null", "mean", "dispersion", "w1split", "w1det", "rare")
TAU_W1DET = 0.3  # persistence threshold isolating the merge classes (Phase 4)
NS = (50, 100, 200, 500)
COARSE_REPS = 100
FULL_REPS = 500
N_BINS_DIST = 32            # frozen Phase 4.5 grid (do not change)
EQUIV_MARGIN = 0.25         # Phase 3 equivalence convention

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, "..", "results")
SHARDS = os.path.join(RESULTS, "phase8_shards")
MASTER_PARQUET = os.path.join(RESULTS, "phase8_master.parquet")
MASTER_JSON = os.path.join(RESULTS, "phase8_master.json")
FIG_PNG = os.path.join(RESULTS, "phase8_figure.png")

CONTRACT = {
    "base_seed": BASE_SEED, "alpha": ALPHA, "m_points": M_POINTS,
    "filtration": FILTRATION, "hom_dims": list(HOM_DIMS),
    "sil_interval": list(SIL_INTERVAL), "sil_r": SIL_R, "sil_res": SIL_RES,
    "n_basis": N_BASIS, "n_folds": N_FOLDS, "n_perm_coarse": N_PERM_COARSE,
    "dr_draws_coarse": DR_DRAWS_COARSE, "dist_draws": DIST_DRAWS,
    "eps_rt": EPS_RT, "rt_approx": RT_APPROX, "prop_scale": PROP_SCALE,
    "beta": list(BETA), "n_bins_dist": N_BINS_DIST,
}

# Null tested per column (printed in every table header; never rank across
# different nulls without it).
NULL_OF = {
    "rt": "H0^cond", "mmd": "H0^cond", "han": "H0^cond",
    "strand": "H0^cond", "moon_lazar": "H0^cond",
    "frechet_anova": "H0^cond", "krebs_rademacher": "H0^disp/equiv",
    "dr_mult": "H0^out", "dr_perm": "H0^out",
    "dist_maxt": "H0^dist-grid", "dist_mmd": "H0^dist",
    "dist_l1": "H0^dist (L1, w1det diagnostic)",
    "dr_equiv": "H0^equiv(non-equivalence null)",
}

# Fixed per-competitor seed offsets. These MUST be constants, never
# hash(name): Python string hashing is randomized per process
# (PYTHONHASHSEED), so hash-based offsets made the mmd/han/strand/moon_lazar
# columns irreproducible across worker processes (found 2026-09-10: 54-70 of
# 100 p-values differed coarse-vs-full on identical rep seeds, while every
# fixed-offset column was byte-identical). Fixed 2026-09-10; shards tagged
# coarse/robust/smoke predate the fix for those four columns only (valid
# Monte Carlo regardless: every p-value used its own independent seed).
_COMP_SEED_OFFSET = {"mmd": 101, "han": 202, "strand": 303,
                     "moon_lazar": 404, "frechet_anova": 11}


def _competitor_seed(seed_base: int, name: str) -> int:
    return seed_base + _COMP_SEED_OFFSET[name]


# Tests run in every cell (KR + equiv added selectively; see _one_rep).
ALWAYS_TESTS = ("rt", "mmd", "han", "strand", "moon_lazar", "frechet_anova",
                "dr_mult", "dr_perm", "dist_maxt", "dist_mmd")


def _seed(*parts) -> int:
    value = BASE_SEED
    for part in parts:
        if isinstance(part, str):
            part = sum((i + 1) * b for i, b in enumerate(part.encode()))
        value = (value * 7919 + int(part)) % (2 ** 31 - 1)
    return int(value)


# --------------------------------------------------------------------------
# diagram cache + hazard fixes
# --------------------------------------------------------------------------
_DIAG_CACHE: dict = {}
_CACHE_HITS = 0
_CACHE_MISSES = 0


def _cloud_key(cloud: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(cloud, dtype=np.float32).tobytes())
    h.update(repr((FILTRATION, HOM_DIMS)).encode())
    return h.hexdigest()[:24]


def diagrams_of(cloud: np.ndarray):
    """Diagrams for one cloud via the cache (never recomputed in loops)."""
    global _CACHE_HITS, _CACHE_MISSES
    from tda2s.ph import compute_diagrams
    key = _cloud_key(cloud)
    hit = _DIAG_CACHE.get(key)
    if hit is not None:
        _CACHE_HITS += 1
        return hit
    _CACHE_MISSES += 1
    d = compute_diagrams(cloud, filtration=FILTRATION, homology_dims=HOM_DIMS)
    _DIAG_CACHE[key] = d
    return d


def _rt_matrix_fast(diags, eps: float = EPS_RT, approx: float = RT_APPROX):
    """Joint-loss RT matrix with the exactly-equal short-circuit.

    Hazard fix (Phase 1 exit note): gudhi.bottleneck_distance is 20-60x
    slower on exactly-coincident diagrams (CGAL degenerate path), so pairs
    with array-equal filtered diagrams score 0 without a gudhi call.
    Near-diagonal points (persistence < eps) are dropped first (Phase 2
    convention); empty post-filter diagrams are kept (distance to empty =
    half the largest persistence), because loop-count contrasts live in
    exactly those diagrams.
    """
    import gudhi as gd
    n = len(diags)
    filt = [[np.asarray(d[d[:, 1] - d[:, 0] >= eps]).reshape(-1, 2)
             if len(d) else d.reshape(0, 2) for d in diag] for diag in diags]
    P = np.zeros((n, n))
    for dim in range(len(diags[0])):
        for i in range(n):
            Di = filt[i][dim]
            for j in range(i + 1, n):
                Dj = filt[j][dim]
                if Di.shape == Dj.shape and np.array_equal(Di, Dj):
                    continue  # short-circuit: distance is exactly 0
                v = float(gd.bottleneck_distance(Di, Dj, approx))
                P[i, j] = P[j, i] = P[i, j] + v
    return P


# --------------------------------------------------------------------------
# DGPs (all reuse tda2s.dgp generators; imbalance via Phase 2 propensity)
# --------------------------------------------------------------------------
def _universal_kernel_matrix(diags, sigma, block=100):
    """Vectorised universal PSS kernel matrix (same math as dist_level).

    ``DL.mmd_kernel_matrix`` loops over pairs in Python (fine for Phase 4's
    n<=200, prohibitive at n=500: 250k pairs). This pads diagrams to a common
    count and evaluates the Reininghaus k_sigma / Kwitt exp(k_sigma) with
    broadcasting, one row-block at a time. Padded entries are filled with
    1e18 so their Gaussian contributions underflow to exactly 0. Verified
    elementwise against ``DL.mmd_kernel_matrix`` in --mode smoke.
    """
    pts = [np.asarray(d, float).reshape(-1, 2) for d in diags]
    n = len(pts)
    pmax = max(1, max(len(p) for p in pts))
    F = np.full((n, pmax, 2), 1e18)
    for i, p in enumerate(pts):
        if len(p):
            F[i, :len(p)] = p
    G = F[:, :, ::-1]  # mirrored for the boundary correction term
    K = np.empty((n, n))
    denom = 8.0 * np.pi * sigma
    for lo in range(0, n, block):
        A = F[lo:lo + block]                      # (b, P, 2)
        d2 = ((A[:, None, :, None, :] - F[None, :, None, :, :]) ** 2).sum(-1)
        d2b = ((A[:, None, :, None, :] - G[None, :, None, :, :]) ** 2).sum(-1)
        with np.errstate(over="ignore", under="ignore"):
            k = (np.exp(-d2 / (8.0 * sigma)) - np.exp(-d2b / (8.0 * sigma))
                 ).sum(axis=(2, 3)) / denom
        K[lo:lo + block] = np.exp(k)
    return K


def _draw_labels(X, lam, seed):
    rng = np.random.default_rng(seed)
    pi = 1.0 / (1.0 + np.exp(-PROP_SCALE * lam * (np.asarray(X) @ BETA)))
    return rng.binomial(1, pi).astype(int), pi


def sample_cell(family, lam, n, rep):
    """One sample of n clouds. Returns dict(clouds, X, A, pi, truth)."""
    from tda2s.dgp import CloudSampleDGP
    from tda2s.dgp.clouds import loops_cloud, split_cluster_cloud
    rng = np.random.default_rng(_seed("sample", family, lam, n, rep))
    if family in ("null", "mean", "dispersion", "rare"):
        geff = 1 if family == "mean" else 0
        if family == "dispersion":
            # Loop-count mixture with matched mean (2 loops): control fixed
            # at 2, treated draws {1, 3} w.p. 1/2. The summary-mean is not
            # exactly preserved (silhouette is nonlinear), so this is a
            # dispersion-shift *alternative*, not a pure weak null.
            def knob(x, _rng=rng):
                return 2, 1.0, 0.05
            dgp = CloudSampleDGP(n_per_group=n // 2, m=M_POINTS, d_x=3,
                                 beta=BETA, prop_scale=PROP_SCALE * lam,
                                 topology_knob=knob, group_effect=0,
                                 seed=_seed("dgp", family, lam, n, rep))
            sample = dgp.sample(rng=rng)
            clouds, X, A = list(sample.clouds), sample.X, sample.A
            for i in range(len(clouds)):
                if A[i] == 1:
                    k = 1 if rng.random() < 0.5 else 3
                    clouds[i] = loops_cloud(M_POINTS, k, radius=1.0,
                                            noise=0.05, rng=rng)
            pi = dgp.propensity(X)
            truth = {"effect": "loop-count dispersion mixture"}
        elif family == "rare":
            dgp = CloudSampleDGP(n_per_group=n // 2, m=M_POINTS, d_x=3,
                                 beta=BETA, prop_scale=PROP_SCALE * lam,
                                 group_effect=0,
                                 seed=_seed("dgp", family, lam, n, rep))
            sample = dgp.sample(rng=rng)
            clouds, X, A = list(sample.clouds), sample.X, sample.A
            for i in range(len(clouds)):
                if A[i] == 1 and rng.random() < 0.15:
                    clouds[i] = loops_cloud(
                        M_POINTS, 2, radius=np.array([1.0, 0.35]),
                        noise=0.05, rng=rng)
            pi = dgp.propensity(X)
            truth = {"effect": "rare (15%) extra small loop r=0.35"}
        else:
            dgp = CloudSampleDGP(n_per_group=n // 2, m=M_POINTS, d_x=3,
                                 beta=BETA, prop_scale=PROP_SCALE * lam,
                                 group_effect=geff,
                                 seed=_seed("dgp", family, lam, n, rep))
            sample = dgp.sample(rng=rng)
            clouds, X, A, pi = sample.clouds, sample.X, sample.A, sample.propensity
            truth = {"effect": "none (size)" if geff == 0 else "+1 loop"}
        return {"clouds": clouds, "X": np.asarray(X, float),
                "A": np.asarray(A, int), "pi": np.asarray(pi, float),
                "truth": truth}
    if family == "w1split":
        # Stochastic cluster splitting (Phase 4.5 power recipe): equal
        # 36-point clouds, 2 vs 3 Gaussian blobs. With stochastic geometry
        # the merge-scale laws differ across arms, so H0^out is FALSE here
        # (everything fires) — this column is a gross-topology-change power
        # check, NOT the H0^out-true separation witness (that is 'w1det').
        X = rng.normal(size=(n, 1))
        # propensity logistic in X (same functional form as Phase 2, 1-D)
        pi = 1.0 / (1.0 + np.exp(-PROP_SCALE * lam * X[:, 0]))
        A = np.random.default_rng(
            _seed("labels", family, lam, n, rep)).binomial(1, pi).astype(int)
        clouds = []
        for a in A:
            if int(a) == 0:
                clouds.append(split_cluster_cloud(18, 2, separation=3.0,
                                                  noise=0.15, rng=rng))
            else:
                clouds.append(split_cluster_cloud(12, 3, separation=3.0,
                                                  noise=0.15, rng=rng))
        return {"clouds": clouds, "X": X, "A": A, "pi": pi,
                "truth": {"effect": "2-blob vs 3-blob split (stochastic)"}}
    if family == "w1det":
        # Deterministic W1 separation witness (Phase 4 recipe): 2-blob
        # deterministic 18-gons vs 3-blob deterministic 12-gons (36 points
        # each, one vs two finite H0 classes at the SAME merge scale 1.35).
        # After the TAU threshold the mean normalized silhouette is preserved
        # realization-by-realization (H0^out TRUE) while the expected
        # persistence measures differ by 1.35**3 (H0^dist FALSE). Adds the
        # imbalance dimension over Phase 4: DR size under confounding with
        # H0^out exactly true.
        from tda2s.dgp.clouds import split_cluster_cloud as _scc
        X = rng.normal(size=(n, 1))
        pi = 1.0 / (1.0 + np.exp(-PROP_SCALE * lam * X[:, 0]))
        A = np.random.default_rng(
            _seed("labels", family, lam, n, rep)).binomial(1, pi).astype(int)
        clouds = []
        for a in A:
            if int(a) == 0:
                clouds.append(_scc(18, 2, separation=3.0, noise=0.15,
                                   deterministic=True, n_gon=18, rng=rng))
            else:
                clouds.append(_scc(12, 3, separation=3.0, noise=0.15,
                                   deterministic=True, n_gon=12, rng=rng))
        return {"clouds": clouds, "X": X, "A": A, "pi": pi,
                "threshold_tau": TAU_W1DET,
                "truth": {"effect": "deterministic 2-vs-3 split (H0^out true)"}}
    raise ValueError(f"unknown family {family!r}")


def sample_robust(kind, n, rep):
    """Robustness DGPs (8.5), all under the null (size checks) except R4."""
    from tda2s.dgp import CloudSampleDGP
    from tda2s.dgp.clouds import loops_cloud
    rng = np.random.default_rng(_seed("robust", kind, n, rep))
    if kind == "outliers":
        dgp = CloudSampleDGP(n_per_group=n // 2, m=M_POINTS, d_x=3,
                             beta=np.zeros(3), prop_scale=0.0, group_effect=0,
                             outlier_fraction=0.10,
                             seed=_seed("dgp", kind, n, rep))
        s = dgp.sample(rng=rng)
        return {"clouds": s.clouds, "X": np.asarray(s.X, float),
                "A": np.asarray(s.A, int), "pi": np.asarray(s.propensity, float),
                "truth": {"defect": "10% uniform outliers"}}
    if kind == "unequal_size":
        dgp = CloudSampleDGP(n_per_group=n // 2, m=M_POINTS, d_x=3,
                             beta=np.zeros(3), prop_scale=0.0, group_effect=0,
                             seed=_seed("dgp", kind, n, rep))
        s = dgp.sample(rng=rng)
        clouds = list(s.clouds)
        A = np.asarray(s.A, int)
        for i in range(len(clouds)):  # cardinality confounded with arm
            m = 40 if A[i] == 0 else 160
            clouds[i] = loops_cloud(m, clouds[i].shape[0] and 2 or 2,
                                    radius=1.0, noise=0.05, rng=rng)
        return {"clouds": clouds, "X": np.asarray(s.X, float), "A": A,
                "pi": np.asarray(s.propensity, float),
                "truth": {"defect": "m=40 vs m=160 by arm"}}
    if kind == "heavy_tail":
        X = rng.normal(size=(n, 3))
        A = rng.binomial(1, 0.5, size=n).astype(int)
        clouds = []
        for _ in range(n):  # same heavy-tailed law both arms (size check)
            k = 1 if rng.random() < 0.8 else 8
            clouds.append(loops_cloud(max(64, 8 * k + 8), k, radius=1.0,
                                      noise=0.05, rng=rng))
        return {"clouds": clouds, "X": X, "A": A,
                "pi": np.full(n, 0.5),
                "truth": {"defect": "loop-count {1 w.p. .8, 8 w.p. .2}"}}
    if kind == "nuisance_misspec":
        dgp = CloudSampleDGP(n_per_group=n // 2, m=M_POINTS, d_x=3,
                             beta=BETA, prop_scale=PROP_SCALE, group_effect=0,
                             seed=_seed("dgp", kind, n, rep))
        s = dgp.sample(rng=rng)
        return {"clouds": s.clouds, "X": np.asarray(s.X, float),
                "A": np.asarray(s.A, int), "pi": np.asarray(s.propensity, float),
                "truth": {"defect": "both nuisances misspecified (neg control)"}}
    raise ValueError(f"unknown robustness kind {kind!r}")


# --------------------------------------------------------------------------
# one replication: diagrams once, all tests on labels
# --------------------------------------------------------------------------
def _silhouettes(diags_list):
    from tda2s.vec import silhouette
    phi = np.stack([
        silhouette(d, interval=SIL_INTERVAL, r=SIL_R, resolution=SIL_RES)
        for d in diags_list])
    return phi  # (n, n_dim, res)


def _dist_crossfit_nuisances(feats, A, X, seed):
    """Phase 4.5 confounded recipe: 5-fold LR propensity + per-arm linreg."""
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import KFold
    n = len(A)
    pi = np.empty(n)
    mu0 = np.empty_like(feats)
    mu1 = np.empty_like(feats)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        lr = LogisticRegression(max_iter=2000).fit(X[tr], A[tr])
        pi[te] = lr.predict_proba(X[te])[:, 1]
        for a, mu in ((0, mu0), (1, mu1)):
            m = (A[tr] == a)
            reg = LinearRegression().fit(X[tr][m], feats[tr][m])
            mu[te] = reg.predict(X[te])
    pi = np.clip(pi, 1e-2, 1 - 1e-2)
    return pi, mu0, mu1


def _one_rep(family, imb, n, rep, n_perm, dr_draws, dist_draws,
             robust_kind=None):
    from sklearn.linear_model import LogisticRegression
    from tda2s.benchmarks import run_competitor
    from tda2s.benchmarks.krebs_rademacher import test_krebs_rademacher
    from tda2s.benchmarks.rt import test_rt_from_matrix as _rt
    from tda2s.tests import dist_level as DL
    from tda2s.tests.dr_outcome import (equivalence_test, fit_dr,
                                        multiplier_test,
                                        propensity_strata,
                                        stratified_permutation_test)
    t_start = time.time()
    row = {"family": family, "imbalance": imb, "n": n, "rep": rep,
           "robust": robust_kind, "ok": True, "error": ""}
    timers = {}
    try:
        lam = IMBALANCES[imb] if imb in IMBALANCES else 0.0
        if robust_kind is not None:
            smp = sample_robust(robust_kind, n, rep)
            fam_key = f"robust:{robust_kind}"
        else:
            smp = sample_cell(family, lam, n, rep)
            fam_key = family
        clouds, X, A = smp["clouds"], smp["X"], smp["A"]
        n_tot = len(clouds)
        row["n0"] = int((A == 0).sum())
        row["n1"] = int((A == 1).sum())

        # Diagrams ONCE; everything downstream permutes labels only.
        t0 = time.time()
        diags_list = [diagrams_of(c) for c in clouds]
        if smp.get("threshold_tau") is not None:
            # Deterministic-witness filter (Phase 4 recipe): keep H0 classes
            # with persistence above TAU and round to 1e-6 so the merge
            # scales land exactly on the pinned grid bins. Required for the
            # realization-by-realization H0^out exactness of 'w1det'.
            tau = float(smp["threshold_tau"])
            screened = []
            for per_dim in diags_list:
                dgm = np.asarray(per_dim[0], dtype=float).reshape(-1, 2)
                screened.append([np.round(
                    dgm[dgm[:, 1] - dgm[:, 0] > tau], 6)])
            diags_list = screened
            row["witness_filter"] = f"TAU={tau}"
        timers["t_ph"] = time.time() - t0
        d0 = [d for d, a in zip(diags_list, A) if a == 0]
        d1 = [d for d, a in zip(diags_list, A) if a == 1]

        # ---- per-test isolation: one test's deterministic failure (e.g. the
        # studentized dist test on the zero-variance deterministic witness)
        # must not nuke the rep's other p-values. Failures are recorded per
        # test in row["fails"], NaN in the p-column; rep ok=False only when
        # the shared data pipeline itself fails (see the outer except).
        fails = {}

        def _run(key, fn, timer=None):
            t0 = time.time()
            try:
                row[f"p_{key}"] = float(fn())
            except Exception as exc:
                row[f"p_{key}"] = np.nan
                fails[key] = (f"{type(exc).__name__}: {exc} "
                              f"{traceback.format_exc()[-300:]}")
            if timer is not None:
                timers[timer] = time.time() - t0

        # ---- unadjusted competitors (H0^cond) ----
        seed_base = _seed("test", fam_key, imb, n, rep)
        for name in ("mmd", "han", "strand", "moon_lazar", "frechet_anova"):
            def _comp(name=name):
                kw = {"epsilon": EPS_RT} if name in ("mmd", "han") else {}
                kw = dict(kw, n_perm=n_perm,
                          seed=_competitor_seed(seed_base, name))
                return run_competitor(name, d0, d1, **kw)
            _run(name, _comp, timer=f"t_{name}")

        def _rt_run():
            P = _rt_matrix_fast(diags_list)
            return _rt(P, (A == 0), n_perm=n_perm,
                       statistic="within", seed=seed_base + 5)
        _run("rt", _rt_run, timer="t_rt")

        # ---- Krebs-Rademacher: dispersion turf only ----
        if (robust_kind is None and family in ("dispersion", "null")) or \
                (robust_kind in ("heavy_tail",)):
            def _kr():
                return test_krebs_rademacher(
                    d0, d1, delta=0.0, metric="bottleneck", n_perm=n_perm,
                    seed=seed_base + 7)
            _run("krebs_rademacher", _kr, timer="t_krebs_rademacher")
            row["skip_krebs"] = ""
        else:
            row["p_krebs_rademacher"] = np.nan
            row["skip_krebs"] = ("KR targets dispersion (blind to pure "
                                 "location by construction, eq. 1.7); run on "
                                 "dispersion-shift + null cells only")

        # ---- DR outcome-level (H0^out, primary) via tcda_uq path ----
        t0 = time.time()
        phi = _silhouettes(diags_list)
        tseq = np.linspace(SIL_INTERVAL[0], SIL_INTERVAL[1], SIL_RES)
        misspec = (robust_kind == "nuisance_misspec")
        if misspec:
            from sklearn.dummy import DummyClassifier
            prop_est = DummyClassifier(strategy="prior")
            n_basis = 2  # too few Fourier terms: mu misspecified too
        else:
            prop_est = LogisticRegression(max_iter=2000)
            n_basis = N_BASIS
        fit = fit_dr((phi, A, X), tseq, n_basis=n_basis, n_folds=N_FOLDS,
                     propensity_estimator=prop_est,
                     random_state=_seed("fold", fam_key, imb, n, rep))
        timers["t_dr_fit"] = time.time() - t0

        def _dr_mult():
            return multiplier_test(
                fit, n_draws=dr_draws,
                seed=_seed("mult", fam_key, imb, n, rep))["pvalue"]
        _run("dr_mult", _dr_mult, timer="t_dr_mult")

        def _dr_perm():
            strata = propensity_strata(_oos_pi(fit), n_bins=5)
            return stratified_permutation_test(
                fit, strata, n_perm=n_perm,
                seed=_seed("perm", fam_key, imb, n, rep))["pvalue"]
        _run("dr_perm", _dr_perm, timer="t_dr_perm")
        if (robust_kind is None and family == "null") or \
                (robust_kind in ("outliers", "unequal_size", "heavy_tail")):
            def _dr_equiv():
                return equivalence_test(
                    fit, margin=EQUIV_MARGIN, n_draws=dr_draws,
                    seed=_seed("equiv", fam_key, imb, n, rep))["pvalue"]
            _run("dr_equiv", _dr_equiv)
        else:
            row["p_dr_equiv"] = np.nan

        # ---- dist-level (H0^dist-grid production + MMD secondary) ----
        t0 = time.time()
        feats = DL.measure_features(
            diags_list, interval=DL.DEFAULT_INTERVAL,
            n_bins=N_BINS_DIST, weight_power=DL.DEFAULT_WEIGHT_POWER,
            homology_dim=0)
        pi_cf, mu0_cf, mu1_cf = _dist_crossfit_nuisances(
            feats, A, X, _seed("cf", fam_key, imb, n, rep))
        dfit = DL.fit_dist(diags_list, A, X, pi_cf, method="measure",
                           mu0_hat=mu0_cf, mu1_hat=mu1_cf,
                           interval=DL.DEFAULT_INTERVAL, n_bins=N_BINS_DIST,
                           weight_power=DL.DEFAULT_WEIGHT_POWER,
                           homology_dim=0)
        timers["t_dist_fit"] = time.time() - t0

        def _dmaxt():
            return DL.dist_multiplier_test(
                dfit, n_draws=dist_draws,
                seed=_seed("dmaxt", fam_key, imb, n, rep))["pvalue"]
        _run("dist_maxt", _dmaxt, timer="t_dist_maxt")
        if family == "w1det":
            # Deterministic witness: the AIPW scores are EXACTLY constant
            # (V is arm-measurable, so residuals vanish realization by
            # realization and studentization is 0/0 -> deterministic
            # failure, correctly refused). The Phase-4 L1 stratified
            # permutation is the defined dist readout here (power against
            # the separation; confounding caveat as in Phase 4.5 §4.5.4).
            def _dl1():
                dstrat = propensity_strata(pi_cf, n_bins=5)
                return DL.stratified_permutation_test(
                    dfit, dstrat, n_perm=n_perm,
                    seed=_seed("dl1", fam_key, imb, n, rep))["pvalue"]
            _run("dist_l1", _dl1, timer="t_dist_l1")
        else:
            row["p_dist_l1"] = np.nan
        t0 = time.time()
        # dist-MMD secondary: near-diagonal filter first. Stochastic-loop H0
        # diagrams carry ~m boundary points; the universal PSS kernel sums
        # over all of them, so unfiltered the kernel matrix is O(n^2) pairs
        # x O(m^2) points of pure noise. Dropping persistence < EPS_RT keeps
        # every grid signal (>= ~0.26) and perturbs each kernel entry by a
        # bounded amount; the permutation null stays exactly valid for any
        # fixed embedding (same argument as the Phase 2 sweep filter).
        def _filt_h0(dgm):
            dgm = np.asarray(dgm, float).reshape(-1, 2)
            if len(dgm) == 0:
                return np.zeros((0, 2))
            return dgm[dgm[:, 1] - dgm[:, 0] >= EPS_RT]
        filt_mmd = [[_filt_h0(dg[0])] for dg in diags_list]
        dfit_mmd = DL.fit_dist(
            filt_mmd, A, X, pi_cf, method="mmd", homology_dim=0)
        K = _universal_kernel_matrix(
            [f[0] for f in filt_mmd], dfit_mmd.kernel_sigma)
        dstrat = propensity_strata(pi_cf, n_bins=5)

        def _dmmd():
            return DL.stratified_permutation_test(
                dfit_mmd, dstrat, n_perm=n_perm,
                seed=_seed("dmmd", fam_key, imb, n, rep),
                kernel_matrix=K)["pvalue"]
        _run("dist_mmd", _dmmd, timer="t_dist_mmd")

        row["fails"] = json.dumps(fails) if fails else ""
        row["skip_single_cloud"] = ("single-cloud panel out of scope "
                                    "(Phase 5/6 machinery removed)")
    except Exception as exc:  # retained, never hidden
        row["ok"] = False
        row["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-800:]}"
        for name in list(ALWAYS_TESTS) + ["dist_l1"]:
            row.setdefault(f"p_{name}", np.nan)
        row.setdefault("p_krebs_rademacher", np.nan)
        row.setdefault("p_dr_equiv", np.nan)
        row.setdefault("fails", "")
        row.setdefault("skip_krebs", "")
        row.setdefault("skip_single_cloud", "single-cloud panel out of scope")
    row.update(timers)
    row["wall_rep"] = time.time() - t_start
    return row


def _oos_pi(fit):
    """Out-of-fold propensities in original sample order."""
    inv = np.argsort(fit.order)
    # pi_hat is stored in score/fold order alongside fold_ids
    return np.asarray(fit.pi_hat)[np.argsort(np.argsort(fit.order))][inv] \
        if False else np.asarray(fit.pi_hat)[_fold_to_original(fit)]


def _fold_to_original(fit):
    # fit.order maps score/fold position -> original row; pi_hat is in
    # score/fold order, so original-order pi = pi_hat[argsort(order)].
    return np.argsort(np.asarray(fit.order))


# --------------------------------------------------------------------------
# sharding / aggregation / figure
# --------------------------------------------------------------------------
def cell_path(family=None, imb=None, n=None, reps=None, robust_kind=None,
              tag="coarse"):
    if robust_kind is not None:
        name = f"cell_robust-{robust_kind}_n{n}_{reps}reps_{tag}.parquet"
    else:
        name = (f"cell_{family}_{imb}_n{n}_{reps}reps_{tag}.parquet")
    return os.path.join(SHARDS, name)


def run_cells(cells, reps, n_perm, dr_draws, dist_draws, workers, tag,
              skip_existing=False):
    from concurrent.futures import ProcessPoolExecutor
    import pandas as pd
    os.makedirs(SHARDS, exist_ok=True)
    t_all = time.time()
    done = 0
    for (family, imb, n, robust_kind) in cells:
        path = cell_path(family, imb, n, reps, robust_kind, tag)
        if skip_existing and os.path.exists(path):
            print(f"[phase8] exists, skipping {path}", flush=True)
            continue
        tasks = [(family, imb, n, r, n_perm, dr_draws, dist_draws,
                  robust_kind) for r in range(reps)]
        t0 = time.time()
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                rows = list(pool.map(_one_rep_star, tasks))
        else:
            rows = [_one_rep_star(t) for t in tasks]
        pd.DataFrame(rows).to_parquet(path, index=False)
        nok = sum(1 for r in rows if not r.get("ok"))
        dt = time.time() - t0
        done += 1
        print(f"[phase8] {done}/{len(cells)} {os.path.basename(path)}: "
              f"{len(rows)} reps, {nok} failed, {dt:.0f}s", flush=True)
    print(f"[phase8] all cells done in {time.time() - t_all:.0f}s")


def _one_rep_star(args):
    return _one_rep(*args)


def coarse_cells(families=FAMILIES, imbalances=tuple(IMBALANCES), ns=NS):
    out = []
    for f in families:
        for imb in imbalances:
            for n in ns:
                out.append((f, imb, n, None))
    return out


def robust_cells(n=100):
    return [(None, "randomized", n, k) for k in
            ("outliers", "unequal_size", "heavy_tail", "nuisance_misspec")]


def aggregate():
    import pandas as pd
    files = sorted(f for f in os.listdir(SHARDS) if f.endswith(".parquet"))
    if not files:
        raise SystemExit(f"no phase8 shards under {SHARDS}")
    frames = []
    contracts = set()
    for f in files:
        df = pd.read_parquet(os.path.join(SHARDS, f))
        df["shard"] = f
        frames.append(df)
    master = pd.concat(frames, ignore_index=True)
    master.to_parquet(MASTER_PARQUET, index=False)
    # summary table with reps + MC SE per (cell, test)
    pcols = [c for c in master.columns if c.startswith("p_")]
    summary = []
    for keys, grp in master.groupby(
            ["family", "imbalance", "n", "robust", "shard"], dropna=False):
        for pc in pcols:
            v = grp[pc].dropna().to_numpy()
            if len(v) == 0:
                continue
            rate = float((v <= ALPHA).mean())
            se = float(np.sqrt(max(rate * (1 - rate), 0.0) / len(v)))
            summary.append({"family": keys[0], "imbalance": keys[1],
                            "n": int(keys[2]),
                            "robust": None if pd.isna(keys[3]) else keys[3],
                            "shard": keys[4], "test": pc[2:],
                            "null": NULL_OF.get(pc[2:], "?"),
                            "reps": int(len(v)), "rate": rate, "mc_se": se,
                            "failures": int((~grp["ok"]).sum())})
    # timer means per (n, test)
    tcols = [c for c in master.columns if c.startswith("t_")]
    cost = []
    for n, grp in master[master["ok"]].groupby("n"):
        for tc in tcols:
            v = grp[tc].dropna().to_numpy()
            if len(v):
                cost.append({"n": int(n), "timer": tc,
                             "mean_s": float(v.mean()), "reps": int(len(v))})
    payload = {"contract": CONTRACT, "alpha": ALPHA,
               "null_of": NULL_OF, "summary": summary, "cost": cost,
               "n_rows": int(len(master)),
               "n_shards": len(files), "shards": files}
    with open(MASTER_JSON, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"[phase8] master: {len(master)} rows, {len(files)} shards")
    print(f"[phase8] wrote {MASTER_PARQUET} and {MASTER_JSON}")
    return payload


FAMILY_TITLES = {
    "null": "no effect ($H_0^{out}$ true)",
    "mean": "mean shift ($\\psi\\neq 0$)",
    "dispersion": "dispersion shift ($\\psi\\neq 0$)",
    "w1split": "stochastic split ($\\psi\\neq 0$)",
    "w1det": "deterministic split ($H_0^{out}$ true)",
    "rare": "rare extra loop ($\\psi\\neq 0$)",
}
TEST_COLORS = {
    "mmd": "#D55E00", "strand": "#009E73", "frechet_anova": "#E69F00",
    "dr_mult": "#000000", "dist_maxt": "#0072B2", "dist_mmd": "#CC79A7",
}
TEST_LABELS = {
    "mmd": "MMD", "strand": "STRAND", "frechet_anova": "Frechet ANOVA",
    "dr_mult": "DR multiplier", "dist_maxt": "dist-level max-t",
    "dist_mmd": "dist-level MMD",
}
NULL_PRETTY = {
    "H0^cond": "$H_0^{\\mathrm{cond}}$", "H0^out": "$H_0^{\\mathrm{out}}$",
    "H0^dist-grid": "$H_0^{\\mathrm{dist,grid}}$",
    "H0^dist": "$H_0^{\\mathrm{dist}}$",
    "H0^disp/equiv": "$H_0^{\\mathrm{disp/equiv}}$",
}


def figure(payload=None, out=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if payload is None:
        with open(MASTER_JSON) as fh:
            payload = json.load(fh)
    summ = payload["summary"]
    imbs = list(IMBALANCES)
    x = np.arange(len(imbs))
    tests_pow = ["mmd", "strand", "frechet_anova", "dr_mult", "dist_maxt",
                 "dist_mmd"]
    fams = ["null", "mean", "dispersion", "w1split", "w1det", "rare"]
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.8), sharey=True)
    handles = []
    for ax, fam in zip(axes.ravel(), fams):
        for t in tests_pow:
            rates, ses = [], []
            for imb in imbs:
                cells = [s for s in summ if s["family"] == fam and
                         s["imbalance"] == imb and s["n"] == 200 and
                         s["test"] == t and "coarse" in s["shard"]]
                if not cells:
                    cells = [s for s in summ if s["family"] == fam and
                             s["imbalance"] == imb and s["n"] == 200 and
                             s["test"] == t]
                if cells and cells[0]["rate"] == cells[0]["rate"]:
                    rates.append(cells[0]["rate"])
                    ses.append(cells[0]["mc_se"])
                else:
                    rates.append(np.nan)
                    ses.append(np.nan)
            rates = np.asarray(rates, dtype=float)
            ses = np.asarray(ses, dtype=float)
            style = "-" if t not in ("dr_mult", "dist_maxt") else "--"
            weight = 2.2 if t == "dr_mult" else 1.3
            h = ax.errorbar(x, rates, yerr=2 * ses, marker="o", ms=3.5,
                            lw=weight, ls=style, capsize=2, elinewidth=0.6,
                            color=TEST_COLORS[t],
                            label=f"{TEST_LABELS[t]} "
                                  f"({NULL_PRETTY[NULL_OF[t]]})")
            if fam == "null":
                handles.append(h.lines[0])
        ax.axhline(ALPHA, color="k", ls=":", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(imbs, rotation=18, fontsize=9)
        ax.set_title(FAMILY_TITLES[fam], fontsize=10.5)
        ax.set_ylim(-0.04, 1.04)
        ax.grid(alpha=0.25)
    for ax in axes[:, 0]:
        ax.set_ylabel("rejection rate at $\\alpha=0.05$ ($n=200$)")
    fig.legend(handles,
               [f"{TEST_LABELS[t]} ({NULL_PRETTY[NULL_OF[t]]})"
                for t in tests_pow],
               fontsize=9.5, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.05), frameon=False)
    fig.suptitle("Bake-off: rejection rates by alternative family at $n=200$ "
                 "(100 replications; bars are $\\pm 2$ MC SE)", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(out or FIG_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[phase8] figure -> {out or FIG_PNG}")


def cost_probe(n=200):
    """Per-test wall-time + peak-RSS probe, one rep, sequential (Table 8.4)."""
    base = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024
    print(f"[phase8 cost] base RSS {base:.0f} MB; running one n={n} rep "
          f"with per-test timers")
    row = _one_rep("mean", "randomized", n, 999, N_PERM_COARSE,
                   DR_DRAWS_COARSE, DIST_DRAWS)
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024
    for k in sorted(row):
        if k.startswith("t_") or k in ("wall_rep",):
            print(f"[phase8 cost] {k} = {row[k]:.2f}s")
    print(f"[phase8 cost] peak RSS after one rep: {peak:.0f} MB "
          f"(delta {peak - base:.0f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["smoke", "coarse", "full", "robust",
                                       "aggregate", "figure", "cost"],
                    required=True)
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--imbalances", default=",".join(IMBALANCES))
    ap.add_argument("--ns", default=",".join(str(v) for v in NS))
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    if args.mode == "aggregate":
        return aggregate()
    if args.mode == "figure":
        return figure()
    if args.mode == "cost":
        return cost_probe()

    fams = [f for f in args.families.split(",") if f]
    imbs = [v for v in args.imbalances.split(",") if v]
    ns = [int(v) for v in args.ns.split(",") if v]

    if args.mode == "smoke":
        # verify the vectorised MMD kernel against the Phase 4 reference
        from tda2s.tests import dist_level as _DL
        rng = np.random.default_rng(7)
        tiny = [rng.uniform(0, 1, size=(rng.integers(0, 6), 2))
                for _ in range(12)]
        ref = _DL.mmd_kernel_matrix(_DL.fit_dist(
            [[d] for d in tiny], np.arange(12) % 2,
            rng.normal(size=(12, 1)), np.full(12, 0.5), method="mmd",
            homology_dim=0))
        fast = _universal_kernel_matrix(
            tiny, _DL.DEFAULT_KERNEL_SIGMA)
        gap = float(np.max(np.abs(ref - fast)))
        print(f"[phase8] kernel check max abs gap = {gap:.2e}")
        assert gap < 1e-8, "vectorised MMD kernel disagrees with reference"
        cells = [("mean", "randomized", 50, None),
                 ("w1split", "strong", 50, None),
                 ("w1det", "strong", 50, None),
                 ("null", "randomized", 50, None)]
        run_cells(cells, reps=2, n_perm=N_PERM_SMOKE,
                  dr_draws=DR_DRAWS_SMOKE, dist_draws=DIST_DRAWS_SMOKE,
                  workers=1, tag="smoke")
    elif args.mode == "coarse":
        run_cells(coarse_cells(fams, imbs, ns), reps=args.reps or COARSE_REPS,
                  n_perm=N_PERM_COARSE, dr_draws=DR_DRAWS_COARSE,
                  dist_draws=DIST_DRAWS, workers=args.workers,
                  tag=args.tag or "coarse",
                  skip_existing=args.skip_existing)
    elif args.mode == "full":
        run_cells(coarse_cells(fams, imbs, ns), reps=args.reps or FULL_REPS,
                  n_perm=N_PERM_COARSE, dr_draws=DR_DRAWS_COARSE,
                  dist_draws=DIST_DRAWS, workers=args.workers,
                  tag=args.tag or "full",
                  skip_existing=args.skip_existing)
    elif args.mode == "robust":
        run_cells(robust_cells(), reps=args.reps or COARSE_REPS,
                  n_perm=N_PERM_COARSE, dr_draws=DR_DRAWS_COARSE,
                  dist_draws=DIST_DRAWS, workers=args.workers,
                  tag=args.tag or "robust",
                  skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
