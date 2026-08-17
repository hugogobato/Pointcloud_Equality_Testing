"""Phase 4.5 weak-null calibration fleet for the distribution-level test (C2).

Decision machinery for tasks 4.5.3 (studentized multiplier calibration),
4.5.4 (permutation audit) and 4.5.5/4.5.6 (size and power gates) of
``RESEARCH_PLAN_P1_TwoSample.md``.  The calibration contract is frozen in
``docs/phase45_weak_null_calibration.md`` before any fleet is run; the
constants below ARE that contract (they were recorded in the doc and in this
module before the first shard was executed).

Designs (every design is an exact weak or sharp null unless flagged as
power; see the calibration doc §4.5.5):

``w2p``         W2' unequal-law null: arm 0 draws the merge-staircase cloud
                {(0,1),(0,2)} exactly; arm 1 draws {(0,1),(0,1)} or
                {(0,2),(0,2)} with probability 1/2.  Expected persistence
                measures agree exactly in law (H0^dist,grid true) while the
                conditional laws differ.  No covariates, known propensity
                1/2, in-sample arm-mean outcome nuisances.  Cells: n in
                {50,100,200,500} x 500 replications.  Also runs the
                Rademacher and CCK Gaussian-path diagnostics.
``confounded``  Weak null through a binary confounder X in {0,1}: stratum 0
                treats arm 0 with the staircase S and arm 1 with the mixture
                M, stratum 1 swaps the arms, so the law-level target
                delta = (1/2)(v_M - v_S) + (1/2)(v_S - v_M) = 0 exactly while
                each stratum carries a contrast of magnitude 4.5 in L1.
                Propensity regimes (e(0), e(1)) in {(0.5,0.5), (0.4,0.6),
                (0.25,0.75)}.  Cross-fitted nuisances (5 folds: logistic
                propensity, per-arm linear outcome regression on X).  Cells:
                n x regime x 500.  Also runs the frozen and studentized
                within-stratum permutations (the 4.5.4 falsification case).
``sharp``       Conditional sharp null with imbalance: both arms draw S when
                X = 0 and M when X = 1, with the regime-1.0 propensities.
                Laws are equal per stratum, so every mechanism must hold
                level.  Cells: n x 500.
``w1``          W1 power design (task 4.5.6): arm 0 draws stochastic 2-blob
                clouds (18 points per blob), arm 1 draws stochastic 3-blob
                clouds (12 points per blob), separation 3.0, noise 0.15, so
                both arms have 36-point clouds and the expected-measure
                contrast is concentrated at the merge scale (observed L1
                contrast about 2.0).  The Phase 4 deterministic clouds have
                zero within-arm feature variance, which is a degenerate case
                for the studentized statistic; the stochastic geometry is the
                honest power version.  Cells: n in {50,100,200} x 500.
``local``       Mixture power curve at n = 200: arm 1 draws the 3-blob cloud
                with probability p and the 2-blob cloud otherwise, for p in
                {0, .25, .5, .75, 1}.  p = 0 is a randomized sharp null and
                doubles as a permutation agreement check.  Cells: p x 300.

Sharded exactly like the Phase 3/4 fleets; replication seeds are functions
of the replication index alone, so shards concatenate however the work is
split (``--workers`` parallelises within a shard and cannot change the
output).  No calibration draw recomputes persistent homology, features,
cross-fitting or a nuisance regression.

    rtk uv run python experiments/phase45_weak_null.py --mode shard \\
        --design confounded --shard-idx 0 --reps-per-shard 10 --workers 16
    rtk uv run python experiments/phase45_weak_null.py --mode aggregate
    rtk uv run python experiments/phase45_weak_null.py --mode figure
    rtk uv run python experiments/phase45_weak_null.py --mode check-dgp
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold

from tda2s.dgp.clouds import merge_staircase_cloud, split_cluster_cloud
from tda2s.ph import compute_diagrams
from tda2s.tests.dist_level import (
    dist_multiplier_test,
    fit_dist,
    measure_features,
    stratified_permutation_test as dist_permutation_test,
    studentized_permutation_test,
)
from tda2s.vec import persistence_measure

# -- the frozen Phase 4.5 calibration contract (task 4.5.1) ------------------
BASE_SEED = 4500
ALPHA = 0.05
N_DRAWS = 1999                 # multiplier draws (frozen)
N_CALIBRATION = 399            # permutation draws (Phase 4 frozen convention)
VARIANCE_FLOOR = 1e-8          # frozen; zero-variance bins are dropped
N_FOLDS = 5                    # cross-fitting folds for confounded/sharp
TAU = 0.3                      # persistence threshold (Phase 4 witness filter)
INTERVAL = (0.0, 2.0)          # shared pinned grid (Phase 4)
N_BINS = 32                    # bins per axis (Phase 4)
R = 3.0                        # weight power (Phase 4)
NOISE = 0.15
N_GON = 12
SAMPLE_SIZES = (50, 100, 200, 500)
POWER_SAMPLE_SIZES = (50, 100, 200)
REGIMES = (0.0, 0.5, 1.0)      # (e0, e1) in {(0.5,.5),(0.4,.6),(0.25,.75)}
P_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
N_SIZE_REPS = 500              # reps per size cell (>= 500 required)
N_POWER_REPS = 500
N_LOCAL_REPS = 300
CLOUD_POINTS = 36
W1_ARM0_N_GON = 18             # 2 blobs x 18 points
W1_ARM1_N_GON = N_GON          # 3 blobs x 12 points

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, "..", "results")
SHARDS = os.path.join(RESULTS, "phase45_shards")


def _seed(*parts) -> int:
    value = BASE_SEED
    for part in parts:
        if part is None:
            continue
        if isinstance(part, str):
            part = sum((i + 1) * byte for i, byte in enumerate(part.encode()))
        value = (value * 7919 + int(part)) % (2 ** 31 - 1)
    return int(value)


def _reps(design: str) -> int:
    return {"w2p": N_SIZE_REPS, "confounded": N_SIZE_REPS,
            "sharp": N_SIZE_REPS, "w1": N_POWER_REPS,
            "local": N_LOCAL_REPS}[design]


# ---------------------------------------------------------------------------
# cloud geometry and DGPs

def _threshold_diagrams(diagrams, tau: float = TAU) -> list:
    """Keep the H_0 classes with persistence above ``tau``, rounded to 1e-6.

    Same witness filter as Phase 4 (``experiments.phase4_separation.py``):
    deterministic blob geometry carries tiny within-blob classes below the
    threshold, and the 1e-6 rounding pins the merge scales onto the fixed
    bin edges so the exactness statements hold bin-for-bin.
    """
    out = []
    for per_dim in diagrams:
        dgm = np.asarray(per_dim[0], dtype=float).reshape(-1, 2)
        keep = dgm[(dgm[:, 1] - dgm[:, 0] > tau)]
        out.append([np.round(keep, 6)])
    return out


def _w1_stoch_cloud(arm: int, rng) -> np.ndarray:
    """Stochastic W1 cloud with 36 points in both arms."""
    n_blobs = 2 if arm == 0 else 3
    n_gon = W1_ARM0_N_GON if arm == 0 else W1_ARM1_N_GON
    return split_cluster_cloud(n_gon, n_blobs, separation=3.0, noise=NOISE,
                               deterministic=False, n_gon=n_gon, rng=rng)


def _w2p_cloud(arm: int, rng) -> np.ndarray:
    """W2' arm cloud: staircase {(0,1),(0,2)} or the two-location mixture."""
    if arm == 0:
        return merge_staircase_cloud([1.0, 2.0], noise=NOISE, n_gon=N_GON,
                                     rng=rng)
    separation = 2.3 if rng.random() < 0.5 else 4.3   # merges (1,1) or (2,2)
    return split_cluster_cloud(N_GON, 3, separation=separation, noise=NOISE,
                               deterministic=True, n_gon=N_GON, rng=rng)


def _confounded_cloud(x: int, a: int, rng) -> np.ndarray:
    """Stratum effect: S for (x=0, a=0) and (x=1, a=1), M otherwise.

    The pair (x, a) is the only input: the conditional law of the cloud is
    a deterministic function of (X, A), and the law-level contrast is
    E[v_{X,A} | A=1] - E[v_{X,A} | A=0] = 0 exactly (see the calibration doc
    for the bin-exact algebra).  S = staircase {(0,1),(0,2)} (mass 9 in L1);
    M draws {(0,1)} or {(0,2)} with probability 1/2 (mass 4.5 in L1), so
    each stratum carries an L1 contrast of exactly 4.5.
    """
    if (x == 0 and a == 0) or (x == 1 and a == 1):
        return merge_staircase_cloud([1.0, 2.0], noise=NOISE, n_gon=N_GON,
                                     rng=rng)
    separation = 2.3 if rng.random() < 0.5 else 4.3   # {(0,1)} or {(0,2)}
    return split_cluster_cloud(18, 2, separation=separation, noise=NOISE,
                               deterministic=True, n_gon=18, rng=rng)


def _sharp_cloud(x: int, rng) -> np.ndarray:
    """Conditional sharp null: identical law in both arms given X.

    X = 0 -> S, X = 1 -> M in both arms (the pair (x, 0) realizes exactly
    that: (0, 0) -> S and (1, 0) -> M).
    """
    return _confounded_cloud(x, 0, rng)


def _nuisance_models(feats, A, X, seed):
    """Cross-fitted propensity and per-arm outcome regressions (5 folds)."""
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    n = len(A)
    pi_hat = np.empty(n)
    mu0 = np.empty_like(feats)
    mu1 = np.empty_like(feats)
    for tr, te in kf.split(X):
        for a in (0, 1):
            model = LinearRegression().fit(X[tr][A[tr] == a],
                                           feats[tr][A[tr] == a])
            (mu0 if a == 0 else mu1)[te] = model.predict(X[te])
        pi = LogisticRegression(max_iter=2000, random_state=seed).fit(X[tr], A[tr])
        pi_hat[te] = pi.predict_proba(X[te])[:, 1]
    return pi_hat, mu0, mu1


def _in_sample_arm_means(feats, A):
    """No-covariate nuisances: per-arm feature means (the collapse limit)."""
    n = len(A)
    m0 = np.tile(feats[A == 0].mean(axis=0), (n, 1))
    m1 = np.tile(feats[A == 1].mean(axis=0), (n, 1))
    return np.full(n, 0.5), m0, m1


# ---------------------------------------------------------------------------
# one replication

def _one_rep(rep: int, n: int, design: str, regime: float = None,
             p: float = None) -> dict:
    """One sample of ``design`` at cell (n, regime, p), all calibrations.

    Row fields: multiplier p-value and statistic (the gating quantities),
    active/dropped coordinate counts, the observed L1 contrast, the
    Rademacher and CCK Gaussian-path p-values on ``w2p``, the frozen and
    studentized within-stratum permutation p-values (``confounded``,
    ``sharp``, ``w1``, ``local``), and fitted-propensity diagnostics.
    """
    rng = np.random.default_rng(_seed("sample", rep, n, design, regime, p))

    if design in ("w2p", "w1", "local"):
        n_arm = n // 2
        labels = np.concatenate([np.zeros(n_arm, dtype=int),
                                 np.ones(n - n_arm, dtype=int)])
        rng.shuffle(labels)
        X = np.zeros((n, 1))
        if design == "w2p":
            clouds = [_w2p_cloud(int(a), rng) for a in labels]
        elif design == "w1":
            clouds = [_w1_stoch_cloud(int(a), rng) for a in labels]
        else:
            clouds = [_w1_stoch_cloud(0, rng) for a in labels]
            for i, a in enumerate(labels):
                if a == 1 and rng.random() < p:
                    clouds[i] = _w1_stoch_cloud(1, rng)
    else:  # confounded / sharp
        X = rng.integers(0, 2, size=n)
        e0, e1 = _propensity_for(regime)
        pi_true = np.where(X == 0, e0, e1)
        labels = rng.binomial(1, pi_true).astype(int)
        if design == "confounded":
            clouds = [_confounded_cloud(int(x), int(a), rng)
                      for x, a in zip(X, labels)]
        else:
            clouds = [_sharp_cloud(int(x), rng) for x in X]

    diagrams = _threshold_diagrams(
        [compute_diagrams(c, filtration="alpha", homology_dims=(0,))
         for c in clouds])
    feats = measure_features(diagrams, interval=INTERVAL, n_bins=N_BINS,
                             weight_power=R, homology_dim=0)

    if design in ("w2p", "w1", "local"):
        pi_hat, mu0, mu1 = _in_sample_arm_means(feats, labels)
    else:
        pi_hat, mu0, mu1 = _nuisance_models(
            feats, labels, X.reshape(-1, 1), seed=_seed("nuis", rep, n, design,
                                                        regime))

    fit = fit_dist(diagrams, labels, X.reshape(-1, 1), pi_hat,
                   method="measure", interval=INTERVAL, n_bins=N_BINS,
                   weight_power=R, homology_dim=0, mu0_hat=mu0, mu1_hat=mu1)

    row = {
        "rep": int(rep), "n": int(n), "design": design,
        "regime": float(regime) if regime is not None else None,
        "p": float(p) if p is not None else None,
        "n_arm0": int((labels == 0).sum()), "n_arm1": int((labels == 1).sum()),
        "n_stratum0": int((X == 0).sum()),
        "min_pi": float(np.min(pi_hat)), "max_pi": float(np.max(pi_hat)),
    }

    try:
        mult = dist_multiplier_test(fit, n_draws=N_DRAWS, alpha=ALPHA,
                                    seed=_seed("mult", rep, n, design,
                                               regime, p),
                                    variance_floor=VARIANCE_FLOOR)
        row.update({
            "multiplier_p": float(mult["pvalue"]),
            "multiplier_statistic": float(mult["statistic"]),
            "estimate_l1": float(mult["estimate_l1"]),
            "n_active_coordinates": int(mult["n_active_coordinates"]),
            "n_dropped_coordinates": int(mult["n_dropped_coordinates"]),
            "n_coordinates": int(mult["n_coordinates"]),
            "sigma2_max": float(mult["sigma2"].max()),
        })
    except ValueError as exc:
        row["failure"] = f"deterministic: {exc}"

    if design == "w2p":
        rad = dist_multiplier_test(fit, n_draws=N_DRAWS, alpha=ALPHA,
                                   seed=_seed("rad", rep, n),
                                   variance_floor=VARIANCE_FLOOR,
                                   multiplier="rademacher")
        gau = dist_multiplier_test(fit, n_draws=N_DRAWS, alpha=ALPHA,
                                   seed=_seed("gau", rep, n),
                                   variance_floor=VARIANCE_FLOOR,
                                   gaussian_path=True)
        row["rademacher_p"] = float(rad["pvalue"])
        row["gaussian_path_p"] = float(gau["gaussian_path"]["pvalue"])
        row["gaussian_path_max_ecdf_gap"] = float(
            gau["gaussian_path"]["max_ecdf_gap"])

    if design in ("confounded", "sharp", "w1", "local"):
        strata = X if design in ("confounded", "sharp") else np.zeros(n, int)
        perm = dist_permutation_test(fit, strata, n_perm=N_CALIBRATION,
                                     seed=_seed("perm", rep, n, design,
                                                regime, p))
        row["permutation_p"] = float(perm["pvalue"])
        row["permutation_statistic"] = float(perm["statistic"])
    if design in ("confounded", "sharp", "w2p"):
        strata = X if design in ("confounded", "sharp") else np.zeros(n, int)
        sperm = studentized_permutation_test(
            fit, strata, n_perm=N_CALIBRATION,
            seed=_seed("sperm", rep, n, design, regime, p),
            variance_floor=VARIANCE_FLOOR)
        row["studentized_permutation_p"] = float(sperm["pvalue"])
    return row


def _propensity_for(regime: float):
    return {(0.0, 0.0): (0.5, 0.5),
            (0.5, 0.5): (0.4, 0.6),
            (1.0, 1.0): (0.25, 0.75)}[(regime, regime)]


def _cells(design: str):
    """(n, regime, p) cells of one design, in canonical order."""
    if design == "w2p":
        return [(n, None, None) for n in SAMPLE_SIZES]
    if design == "confounded":
        return [(n, r, None) for n in SAMPLE_SIZES for r in REGIMES]
    if design == "sharp":
        return [(n, 1.0, None) for n in SAMPLE_SIZES]
    if design == "w1":
        return [(n, None, None) for n in POWER_SAMPLE_SIZES]
    if design == "local":
        return [(200, None, p) for p in P_GRID]
    raise ValueError(f"unknown design {design!r}")


# ---------------------------------------------------------------------------
# sharding

def _one_rep_job(job):
    return _one_rep(*job)


def run_shard(shard_idx: int, reps_per_shard: int, *, design: str = "w2p",
              workers: int = 1) -> str:
    os.makedirs(SHARDS, exist_ok=True)
    lo = shard_idx * reps_per_shard
    hi = min(lo + reps_per_shard, _reps(design))
    jobs = []
    for rep in range(lo, hi):
        for n, regime, p in _cells(design):
            jobs.append((rep, n, design, regime, p))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(_one_rep_job, jobs))
    else:
        rows = [_one_rep(*j) for j in jobs]
    path = os.path.join(SHARDS, f"phase45_{design}_shard{shard_idx}.json")
    with open(path, "w") as fh:
        json.dump({"design": design, "shard": shard_idx,
                   "reps": [lo, hi], "rows": rows}, fh, indent=1)
    return path


def _load_rows(design: str):
    rows = []
    for path in sorted(glob.glob(os.path.join(SHARDS, f"phase45_{design}_shard*.json"))):
        with open(path) as fh:
            rows.extend(json.load(fh)["rows"])
    return rows


# ---------------------------------------------------------------------------
# aggregation and the 4.5.7 gate

def _rate(rows, key):
    vals = [r[key] for r in rows if key in r]
    if not vals:
        return None
    vals = np.asarray(vals)
    rate = float((vals < ALPHA).mean())
    return {"reps": int(vals.size), "rate": rate,
            "mc_se": float(np.sqrt(rate * (1 - rate) / vals.size)),
            "mean": float(vals.mean())}


def _cell_key(design: str, n: int, regime, p) -> str:
    return str((design, n, regime, p))


def aggregate(*, verbose: bool = True) -> dict:
    out = {"cells": {}, "verdict": {}}
    for design in ("w2p", "confounded", "sharp", "w1", "local"):
        rows = _load_rows(design)
        for n, regime, p in _cells(design):
            sel = [r for r in rows
                   if r["n"] == n
                   and (r["regime"] == regime or (r["regime"] is None and regime is None))
                   and (r["p"] == p or (r["p"] is None and p is None))]
            cell = {"design": design, "n": n, "regime": regime, "p": p}
            for key in ("multiplier_p", "permutation_p",
                        "studentized_permutation_p", "rademacher_p",
                        "gaussian_path_p"):
                cell[key] = _rate(sel, key)
            cell["estimate_l1_mean"] = float(np.mean(
                [r["estimate_l1"] for r in sel if "estimate_l1" in r])) \
                if any("estimate_l1" in r for r in sel) else None
            cell["n_active_mean"] = float(np.mean(
                [r["n_active_coordinates"] for r in sel
                 if "n_active_coordinates" in r])) \
                if any("n_active_coordinates" in r for r in sel) else None
            cell["deterministic_failures"] = int(
                sum(1 for r in sel if "failure" in r))
            out["cells"][_cell_key(design, n, regime, p)] = cell
            if verbose and cell["multiplier_p"]:
                line = (f"{design:11s} n={n:4d} regime={str(regime):4s} "
                        f"p={str(p):4s} reps={cell['multiplier_p']['reps']:4d} "
                        f"mult={cell['multiplier_p']['rate']:.3f} "
                        f"(se {cell['multiplier_p']['mc_se']:.3f})")
                if cell["permutation_p"]:
                    line += f" perm={cell['permutation_p']['rate']:.3f}"
                if cell["studentized_permutation_p"]:
                    line += f" sperm={cell['studentized_permutation_p']['rate']:.3f}"
                if cell["rademacher_p"]:
                    line += f" rad={cell['rademacher_p']['rate']:.3f}"
                if cell["gaussian_path_p"]:
                    line += f" gau={cell['gaussian_path_p']['rate']:.3f}"
                if cell["deterministic_failures"]:
                    line += f" FAILURES={cell['deterministic_failures']}"
                print(line)
    out["verdict"] = _gate(out["cells"])
    return out


def _gate(cells: dict) -> dict:
    """Task 4.5.7 gate, keeping strict and qualified decisions separate.

    The pre-registered size criterion is the closed interval [0.03, 0.08]
    on every primary multiplier cell with ``n >= 200``.  The fleet has one
    near miss, confounded ``r=1.0, n=200`` at 0.096.  It is recorded as an
    exemption for the proposed operating guidance, but it is not removed
    from the strict failure list and therefore cannot make the strict gate
    pass.
    """
    gating = ("w2p", "confounded", "sharp")
    size_failures = []
    missing_size_cells = []
    # Enumerate the pre-registered cells rather than iterating only over
    # files that happened to be loaded.  A missing or undersized shard must
    # not silently turn a gate into a pass.
    for design in gating:
        for n, regime, p in _cells(design):
            if n < 200:
                continue
            key = _cell_key(design, n, regime, p)
            cell = cells.get(key)
            mult = cell.get("multiplier_p") if cell else None
            if not mult or mult.get("reps", 0) < N_SIZE_REPS:
                missing_size_cells.append(
                    (key, int(mult["reps"]) if mult else 0))
                continue
            if not (0.03 <= mult["rate"] <= 0.08):
                size_failures.append((key, mult["rate"]))
    documented = [("('confounded', 200, 1.0, None)", 0.096)]
    documented_map = dict(documented)
    applied_exemptions = []
    qualified_failures = []
    for key, rate in size_failures:
        expected = documented_map.get(key)
        if expected is not None and np.isclose(rate, expected, atol=1e-12):
            applied_exemptions.append((key, rate))
        else:
            qualified_failures.append((key, rate))
    w1_200 = cells.get(_cell_key("w1", 200, None, None))
    local = {p: cells.get(_cell_key("local", 200, None, p)) for p in P_GRID}
    verdict = {
        # ``in_size_band`` retains its literal strict meaning.  The
        # exemption is reported separately and does not alter this field.
        "in_size_band": len(size_failures) == 0 and not missing_size_cells,
        "size_failures": [(k, round(r, 4)) for k, r in size_failures],
        "missing_size_cells": missing_size_cells,
        "qualified_operating_pass": (len(qualified_failures) == 0
                                      and not missing_size_cells),
        "qualified_size_failures": [
            (k, round(r, 4)) for k, r in qualified_failures],
        "documented_exemptions": [
            (k, round(r, 4)) for k, r in applied_exemptions],
        "w1_power_n200": (w1_200["multiplier_p"]["rate"]
                          if w1_200 and w1_200["multiplier_p"] else None),
        "w1_power_gate": bool(w1_200 and w1_200["multiplier_p"]
                              and w1_200["multiplier_p"]["rate"] >= 0.80),
        "local_power": {p: (local[p]["multiplier_p"]["rate"]
                            if local[p] and local[p]["multiplier_p"] else None)
                        for p in P_GRID},
    }
    return verdict


def make_figure() -> str:
    """Figure 4.5: size curves (panel A) and power curves (panel B)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = aggregate(verbose=False)
    cells = out["cells"]
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.4))
    band = (0.03, 0.08)
    for ax in (axa,):
        ax.axhspan(band[0], band[1], color="grey", alpha=0.25, lw=0)
        ax.axhline(ALPHA, color="black", lw=0.8, ls=":")
    series = [("w2p", "W2'", "o", "tab:blue", "multiplier_p", None),
              ("confounded", "confounded r=0.0", "s", "tab:green",
               "multiplier_p", 0.0),
              ("confounded", "confounded r=0.5", "s", "tab:olive",
               "multiplier_p", 0.5),
              ("confounded", "confounded r=1.0", "s", "tab:red",
               "multiplier_p", 1.0),
              ("sharp", "sharp mult", "d", "tab:purple", "multiplier_p", 1.0),
              ("sharp", "sharp perm", "d", "tab:orange", "permutation_p",
               1.0)]
    for design, label, marker, color, key, regime in series:
        xs, ys = [], []
        for n in SAMPLE_SIZES:
            cell = cells.get(_cell_key(design, n, regime, None))
            if cell and cell[key]:
                xs.append(n); ys.append(cell[key]["rate"])
        axa.plot(xs, ys, marker=marker, color=color, label=label)
    axa.set_xscale("log"); axa.set_xticks(list(SAMPLE_SIZES))
    axa.set_xticklabels([str(x) for x in SAMPLE_SIZES])
    axa.set_xlabel("sample size n")
    axa.set_ylabel("rejection rate at alpha = 0.05")
    axa.set_ylim(0.0, 0.25)
    axa.legend(fontsize=7, ncol=2)
    axa.set_title("(a) weak-null size, band [0.03, 0.08]")

    axb.axhline(ALPHA, color="black", lw=0.8, ls=":")
    xs = [n for n in POWER_SAMPLE_SIZES]
    ys = [cells.get(_cell_key("w1", n, None, None))["multiplier_p"]["rate"]
          if cells.get(_cell_key("w1", n, None, None)) else None for n in xs]
    ys = [y for y in ys if y is not None]
    axb.plot([n for n, y in zip(xs, ys) if y is not None], ys,
             marker="o", color="tab:blue", label="W1 power vs n")
    xs = list(P_GRID)
    ys = [local["multiplier_p"]["rate"] for local in
          [cells.get(_cell_key("local", 200, None, p)) for p in P_GRID]
          if local and local["multiplier_p"]]
    axb.plot(list(P_GRID), ys, marker="s", color="tab:red",
             label="local mixture power at n=200")
    axb.set_xlabel("p (mixture weight) / n (W1 line)")
    axb.set_ylabel("rejection rate")
    axb.legend(fontsize=7)
    axb.set_title("(b) power: W1 and the local mixture")
    fig.tight_layout()
    path = os.path.join(RESULTS, "phase45_weak_null_figure.png")
    fig.savefig(path, dpi=150)
    return path


def check_dgp() -> None:
    """Print the exactness statements the fleet relies on (task 4.5.2/4.5.5)."""
    rng = np.random.default_rng(_seed("check"))
    from tda2s.ph import compute_diagrams as cd

    def dgm(cloud):
        return _threshold_diagrams([cd(cloud, filtration="alpha",
                                       homology_dims=(0,))])[0][0]

    s = dgm(merge_staircase_cloud([1.0, 2.0], noise=NOISE, n_gon=N_GON))
    a = dgm(split_cluster_cloud(18, 2, separation=2.3, noise=NOISE,
                                deterministic=True, n_gon=18))
    b = dgm(split_cluster_cloud(18, 2, separation=4.3, noise=NOISE,
                                deterministic=True, n_gon=18))
    w = (lambda pp: float(abs(pp[1] - pp[0]) ** R))
    vS = persistence_measure([np.asarray(s)], weight=w, interval=INTERVAL,
                             n_bins=N_BINS)[0].ravel()
    vA = persistence_measure([np.asarray(a)], weight=w, interval=INTERVAL,
                             n_bins=N_BINS)[0].ravel()
    vB = persistence_measure([np.asarray(b)], weight=w, interval=INTERVAL,
                             n_bins=N_BINS)[0].ravel()
    vM = 0.5 * vA + 0.5 * vB
    print("S diagram:", s.tolist())
    print("M component A diagram:", a.tolist())
    print("M component B diagram:", b.tolist())
    print(f"||v_M - v_S||_1 = {np.abs(vM - vS).sum():.6f} (expect 4.5)")
    print(f"law-level target ||0.5(v_M-v_S)+0.5(v_S-v_M)||_1 = "
          f"{np.abs(0.5 * (vM - vS) + 0.5 * (vS - vM)).sum():.0f} (expect 0)")
    w0 = dgm(_w1_stoch_cloud(0, rng))
    w1 = dgm(_w1_stoch_cloud(1, rng))
    print("W1 stochastic diagrams:", w0.tolist(), "vs", w1.tolist())
    xs = []
    for rep in range(8):
        labels = np.array([0] * 50 + [1] * 50)
        rng2 = np.random.default_rng(_seed("stoch", rep))
        feats = np.stack([
            persistence_measure([np.asarray(dgm(_w1_stoch_cloud(int(a), rng2)))],
                                weight=w, interval=INTERVAL,
                                n_bins=N_BINS)[0].ravel()
            for a in labels])
        xs.append(np.abs(feats[labels == 1].mean(0)
                         - feats[labels == 0].mean(0)).sum())
    print(f"W1 stochastic sample L1 contrasts (8 reps at n=100): "
          f"[{', '.join(f'{v:.2f}' for v in xs)}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("shard", "aggregate", "figure",
                                           "check-dgp"), required=True)
    parser.add_argument("--design", choices=("w2p", "confounded", "sharp",
                                             "w1", "local"), default="w2p")
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--reps-per-shard", type=int, default=25)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shards", type=str, default="0")
    args = parser.parse_args()
    if args.mode == "shard":
        for idx in [int(s) for s in args.shards.split(",")]:
            path = run_shard(idx, args.reps_per_shard, design=args.design,
                             workers=args.workers)
            print("wrote", path)
    elif args.mode == "aggregate":
        out = aggregate()
        print("VERDICT:", out["verdict"])
        path = os.path.join(RESULTS, "phase45_aggregate.json")
        with open(path, "w") as fh:
            json.dump(out, fh, indent=1)
        print("wrote", path)
    elif args.mode == "figure":
        print("wrote", make_figure())
    elif args.mode == "check-dgp":
        check_dgp()


if __name__ == "__main__":
    main()
