"""WP7 T6 (``distgrid``) numerical falsification harness.

Owner work package: T6 ``distgrid`` (fixed-grid distribution-level max-t
validity, local power, grid-size dependence, and the strong-confounding
finite-n boundary).  Theory deliverable: ``theory/WP7_T6_distgrid.md``.

The harness never reimplements AIPW, cross-fitting, the multiplier test or
the persistence-measure vectorisation.  It calls the frozen production code in
``tda2s/tests/dist_level.py`` and the frozen Phase 4.5 DGP helpers in
``experiments/phase45_weak_null.py``.  The ``est_est`` arm of ``mechanism``
reuses the fleet's seeds exactly, so its multiplier p-values are a
reproduction check against ``results/phase45_shards/``.

Pre-registered predictions and falsification criteria
-----------------------------------------------------

P1 (mechanism, n=200, confounded r=1.0).  If the recorded 0.096 is driven by
physical overlap/positivity (H_overlap), the oracle-nuisance arm
``oracle_oracle`` must stay near the nominal level and the ``est_est`` excess
must collapse when either nuisance is replaced by its truth.  If it is driven
by the Gaussian-multiplier approximation to the non-Gaussian estimated
scores (H_mult), ``oracle_oracle`` is already anti-conservative (rate > 0.08)
and the excess is roughly invariant across the four nuisance arms.  Decision
rule: compute paired differences ``rate(est_est) - rate(oracle_oracle)`` with
its paired standard error.  H_overlap is supported if the paired difference
exceeds 0.02 and ``oracle_oracle`` has a 95% interval contained in [0, 0.08].
H_mult is supported if ``oracle_oracle``'s 95% interval lies above 0.05 and
the paired difference is below 0.02.  Both can hold; neither can.

Recorded outcome (2026-09-10 audit).  The literal rule returns *neither*:
the 95% interval for ``oracle_oracle`` is [0.048, 0.092], so the H-overlap
clause (contained in [0, 0.08]) fails and the H-mult clause (above 0.05,
difference below 0.02) fails.  The note reports the hybrid reading, not a
pre-registered H-mult verdict.

P2 (scaling).  Fit ``excess(n) = c_lambda / sqrt(n)`` to the confounded r=1.0
rates (existing n in {50,100,200,500} plus new mechanism cells at n=200 and
n=500).  The scaling restoring the upper band edge 0.08 at r=1.0 is
``n_back(lambda=1) = (c_1 / 0.03)**2``.  Check ``n_back <= 500`` against the
recorded n=500 rate 0.066 and report the MC-propagated uncertainty on
``n_back``.  Falsification: if the fitted ``c_1`` has the wrong sign or
``n_back``'s 95% interval excludes 500 while the observed n=500 rate is in
band, the claimed scaling is rejected.

P3 (q growth, synthetic).  With i.i.d. zero-mean score vectors of dimension
q and equicorrelation rho, the true size at the multiplier's 95% critical
value approaches 0.05 as n grows; for fixed n, the size gap grows with q, and
at fixed (n, q) it is larger for skewed two-point margins than for Gaussian
margins.  Falsification: a Gaussian-margin gap above 0.02 at q=2 for any n.

Recorded outcome (2026-09-10 audit).  The criterion fires on the recorded
configuration (``rho=0.3, R=500, R_mult=80, n_draws=199``): the Gaussian
``q=2, n=100`` gap is +0.025.  A faithful rerun at the script defaults
(``rho=0/0.5, R=1000, R_mult=150, n_draws=499``) gives +0.063.  The P3 verdict is FAIL on the Gaussian sanity criterion; the
skew-margin directional findings are descriptive only.

P4 (finite-grid non-implication).  Two deterministic diagrams that differ by
a within-bin rearrangement produce identical projected feature vectors while
their expected measures differ in total variation.  Falsification: the two
projected vectors differ in any coordinate.

P5 (local power).  Under the W1 design perturbed at mixture rate
``p_n = h / (sqrt(n) S)``, the empirical rejection curve at n in {200, 500}
tracks the Gaussian-shift prediction ``P(max_J |Z + h/S * (mu3-mu2)| > c)``
within 2 Monte Carlo standard errors.  Falsification: a gap above
0.06 with MC SE below 0.03.

Run::

    uv run python scripts/theory_dev/t6_distgrid_validation.py --mode existing
    uv run python scripts/theory_dev/t6_distgrid_validation.py --mode mechanism-aggregate
    uv run python scripts/theory_dev/t6_distgrid_validation.py --mode qcheck
    uv run python scripts/theory_dev/t6_distgrid_validation.py --mode witness
    uv run python scripts/theory_dev/t6_distgrid_validation.py --mode local-aggregate
    uv run python scripts/theory_dev/t6_distgrid_validation.py --mode summary
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from experiments.phase45_weak_null import (
    ALPHA,
    INTERVAL,
    N_BINS,
    N_DRAWS,
    N_GON,
    NOISE,
    R,
    SAMPLE_SIZES,
    TAU,
    VARIANCE_FLOOR,
    _confounded_cloud,
    _in_sample_arm_means,
    _nuisance_models,
    _propensity_for,
    _seed,
    _threshold_diagrams,
    _w1_stoch_cloud,
)
from tda2s.ph import compute_diagrams
from tda2s.tests.dist_level import (
    dist_multiplier_test,
    dist_scores,
    fit_dist,
    measure_features,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(ROOT, "results", "theory_validation")
SHARDS45 = os.path.join(ROOT, "results", "phase45_shards")
os.makedirs(RESULTS, exist_ok=True)

ARMS = ("est_est", "oracle_oracle", "oracle_est_m", "est_oracle_m")
REGIME = 1.0
N_MECH_REPS = 400
N_MECH_REPS_500 = 300
N_LOCAL_REPS = 300


# ---------------------------------------------------------------------------
# helpers

def _jsonify(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _jsonify(obj.tolist())
    return obj


def _rate_summary(pvals):
    p = np.asarray([v for v in pvals if v is not None], dtype=float)
    if p.size == 0:
        return None
    rate = float((p < ALPHA).mean())
    return {
        "reps": int(p.size),
        "rate": rate,
        "mc_se": float(np.sqrt(max(rate * (1.0 - rate), 1e-12) / p.size)),
        "mean_p": float(p.mean()),
    }


def _moment_diagnostics(scores, active):
    """Per-active-coordinate sample moments of the raw scores."""
    out = []
    for j in np.flatnonzero(active):
        s = scores[:, j]
        mu = float(s.mean())
        sd = float(s.std(ddof=0))
        z = (s - mu) / sd if sd > 0 else np.zeros_like(s)
        out.append({
            "mean": mu,
            "sigma": sd,
            "bias_in_stat_units": float(np.sqrt(s.size) * mu / sd) if sd > 0 else 0.0,
            "skew": float(np.mean(z ** 3)),
            "excess_kurtosis": float(np.mean(z ** 4) - 3.0),
        })
    return out


def _bin_index(x, n_bins=N_BINS, interval=INTERVAL):
    edges = np.linspace(interval[0], interval[1], n_bins + 1)
    return int(np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_bins - 1))


_MID1 = _bin_index(0.0) * N_BINS + _bin_index(0.5)
_MID2 = _bin_index(0.0) * N_BINS + _bin_index(1.0)


def _oracle_means(X, q):
    """Conditional means of the confounded features given (X, A) exactly.

    The conditional laws are the deterministic staircase S = (1, 8) at
    (mid 0.5, mid 1.0) and the 50/50 mixture M of (1, 0) and (0, 8), so
    m_0(0) = (1, 8), m_0(1) = (0.5, 4), m_1(0) = (0.5, 4), m_1(1) = (1, 8).
    """
    m0 = np.zeros((X.shape[0], q))
    m1 = np.zeros((X.shape[0], q))
    table0 = {0: (1.0, 8.0), 1: (0.5, 4.0)}
    table1 = {0: (0.5, 4.0), 1: (1.0, 8.0)}
    for x in (0, 1):
        sel = X == x
        m0[sel, _MID1] = table0[x][0]
        m0[sel, _MID2] = table0[x][1]
        m1[sel, _MID1] = table1[x][0]
        m1[sel, _MID2] = table1[x][1]
    return m0, m1


# ---------------------------------------------------------------------------
# existing fleet summary

def existing_summary():
    rows = []
    for path in sorted(glob.glob(os.path.join(SHARDS45, "phase45_confounded_shard*.json"))):
        with open(path) as fh:
            rows.extend(json.load(fh)["rows"])
    cells = {}
    for n in SAMPLE_SIZES:
        for reg in (0.0, 0.5, 1.0):
            sel = [r for r in rows if r["n"] == n and r["regime"] == reg
                   and "multiplier_p" in r]
            stat = np.array([r["multiplier_statistic"] for r in sel])
            pv = np.array([r["multiplier_p"] for r in sel])
            l1 = np.array([r["estimate_l1"] for r in sel])
            minpi = np.array([r["min_pi"] for r in sel])
            cells[f"confounded|n={n}|r={reg}"] = {
                "rate": _rate_summary(pv),
                "stat_quantiles": {
                    "0.5": float(np.quantile(stat, 0.5)),
                    "0.9": float(np.quantile(stat, 0.9)),
                    "0.95": float(np.quantile(stat, 0.95)),
                    "0.99": float(np.quantile(stat, 0.99)),
                },
                "estimate_l1_mean": float(l1.mean()),
                "min_pi_quantiles": {
                    "0.50": float(np.quantile(minpi, 0.5)),
                    "0.01": float(np.quantile(minpi, 0.01)),
                    "min": float(minpi.min()),
                },
                "n_active_unique": np.unique(
                    [r["n_active_coordinates"] for r in sel],
                    return_counts=True)[0].tolist(),
            }
    out = {"source": "results/phase45_shards/phase45_confounded_shard*.json",
           "cells": cells}
    path = os.path.join(RESULTS, "t6_existing_summary.json")
    with open(path, "w") as fh:
        json.dump(_jsonify(out), fh, indent=1)
    print("wrote", path)
    return out


# ---------------------------------------------------------------------------
# mechanism shards (nuisance decomposition, paired by DGP seed)

def _surrogate_diagrams(X, labels, rng):
    """Exact diagram surrogate for the confounded DGP (no persistent homology).

    ``check_dgp`` shows that the thresholded H0 diagram of the staircase S is
    exactly ``[[0,2],[0,1]]`` and that of the mixture component M is exactly
    ``[[0,1]]`` or ``[[0,2]]`` with probability 1/2.  The map
    ``(X, A) -> diagram`` is therefore a deterministic function plus one coin
    flip, and this function reproduces its law without computing persistence.
    The estimator still runs through ``fit_dist``/``dist_multiplier_test``.
    """
    diags = []
    for x, a in zip(X, labels):
        if (x == 0 and a == 0) or (x == 1 and a == 1):
            diags.append([np.array([[0.0, 2.0], [0.0, 1.0]])])
        elif rng.random() < 0.5:
            diags.append([np.array([[0.0, 1.0]])])
        else:
            diags.append([np.array([[0.0, 2.0]])])
    return diags


def _mechanism_rep(rep, n, arms=ARMS, n_draws=N_DRAWS, surrogate=False,
                   regime=REGIME):
    rng = np.random.default_rng(_seed("sample", rep, n, "confounded", regime, None))
    X = rng.integers(0, 2, size=n)
    e0, e1 = _propensity_for(regime)
    pi_true = np.where(X == 0, e0, e1)
    labels = rng.binomial(1, pi_true).astype(int)
    if surrogate:
        diagrams = _surrogate_diagrams(X, labels, rng)
    else:
        clouds = [_confounded_cloud(int(x), int(a), rng)
                  for x, a in zip(X, labels)]
        diagrams = _threshold_diagrams(
            [compute_diagrams(c, filtration="alpha", homology_dims=(0,))
             for c in clouds])
    feats = measure_features(diagrams, interval=INTERVAL, n_bins=N_BINS,
                             weight_power=R, homology_dim=0)
    pi_est, mu0_est, mu1_est = _nuisance_models(
        feats, labels, X.reshape(-1, 1),
        seed=_seed("nuis", rep, n, "confounded", regime))
    mu0_or, mu1_or = _oracle_means(X, feats.shape[1])
    nuisance = {
        "est_est": (pi_est, mu0_est, mu1_est),
        "oracle_oracle": (pi_true, mu0_or, mu1_or),
        "oracle_est_m": (pi_true, mu0_est, mu1_est),
        "est_oracle_m": (pi_est, mu0_or, mu1_or),
    }
    mult_seed = _seed("mult", rep, n, "confounded", regime, None)
    row = {"rep": int(rep), "n": int(n), "regime": float(regime)}
    for arm in arms:
        pi_hat, mu0, mu1 = nuisance[arm]
        fit = fit_dist(diagrams, labels, X.reshape(-1, 1), pi_hat,
                       method="measure", interval=INTERVAL, n_bins=N_BINS,
                       weight_power=R, homology_dim=0, mu0_hat=mu0, mu1_hat=mu1)
        mult = dist_multiplier_test(fit, n_draws=n_draws, alpha=ALPHA,
                                    seed=mult_seed, variance_floor=VARIANCE_FLOOR)
        scores = dist_scores(fit)
        active = np.asarray(mult["active"], dtype=bool)
        row[arm] = {
            "p": float(mult["pvalue"]),
            "stat": float(mult["statistic"]),
            "crit": float(np.quantile(mult["null"], 1.0 - ALPHA)),
            "null_mean": float(mult["null"].mean()),
            "n_active": int(active.sum()),
            "diagnostics": _moment_diagnostics(scores, active),
        }
    return row


def _mechanism_job(job):
    return _mechanism_rep(*job)


def mechanism_shard(shard_idx, reps_per_shard, *, n, reps_total, workers=1,
                    arms=ARMS):
    lo = shard_idx * reps_per_shard
    hi = min(lo + reps_per_shard, reps_total)
    jobs = [(rep, n, tuple(arms)) for rep in range(lo, hi)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(_mechanism_job, jobs))
    else:
        rows = [_mechanism_rep(*j) for j in jobs]
    path = os.path.join(RESULTS, f"t6_mechanism_n{n}_shard{shard_idx}.json")
    with open(path, "w") as fh:
        json.dump(_jsonify({"n": n, "reps": [lo, hi], "rows": rows,
                            "arms": list(arms)}), fh, indent=1)
    print("wrote", path)
    return path


def mechanism_aggregate():
    out = {"arms": list(ARMS), "cells": {}}
    for n in (200, 500):
        rows = []
        for path in sorted(glob.glob(
                os.path.join(RESULTS, f"t6_mechanism_n{n}_shard*.json"))):
            with open(path) as fh:
                rows.extend(json.load(fh)["rows"])
        cell = {"reps": len(rows)}
        for arm in ARMS:
            pvals = [r[arm]["p"] for r in rows if arm in r]
            crit = [r[arm]["crit"] for r in rows if arm in r]
            stats = [r[arm]["stat"] for r in rows if arm in r]
            diag = [r[arm]["diagnostics"] for r in rows if arm in r]
            bias = [d["bias_in_stat_units"] for dd in diag for d in dd]
            skew = [d["skew"] for dd in diag for d in dd]
            cell[arm] = {
                "rate": _rate_summary(pvals),
                "crit_mean": float(np.mean(crit)),
                "stat_mean": float(np.mean(stats)),
                "stat_q95": float(np.quantile(stats, 0.95)) if stats else None,
                "mean_abs_bias_stat_units": float(np.mean(np.abs(bias))),
                "mean_skew": float(np.mean(skew)),
                "mean_abs_skew": float(np.mean(np.abs(skew))),
            }
        out["cells"][f"n={n}"] = cell
    path = os.path.join(RESULTS, "t6_mechanism_summary.json")
    with open(path, "w") as fh:
        json.dump(_jsonify(out), fh, indent=1)
    print("wrote", path)
    return out


def _surrogate_job(job):
    rep, n, arms, regime = job
    return _mechanism_rep(rep, n, arms=arms, surrogate=True, regime=regime)


def surrogate_shard(shard_idx, reps_per_shard, *, n, reps_total, workers=1,
                    arms=ARMS, regime=REGIME):
    """Feature-level (no-PH) mechanism/scaling shards for the confounded DGP."""
    lo = shard_idx * reps_per_shard
    hi = min(lo + reps_per_shard, reps_total)
    jobs = [(rep, n, tuple(arms), float(regime)) for rep in range(lo, hi)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(_surrogate_job, jobs))
    else:
        rows = [_surrogate_job(j) for j in jobs]
    tag = "" if regime == 1.0 else f"_r{regime}"
    path = os.path.join(RESULTS,
                        f"t6_surrogate_n{n}{tag}_shard{shard_idx}.json")
    with open(path, "w") as fh:
        json.dump(_jsonify({"n": n, "regime": float(regime), "reps": [lo, hi],
                            "rows": rows, "arms": list(arms),
                            "surrogate": True}), fh, indent=1)
    print("wrote", path)
    return path


def surrogate_aggregate():
    out = {"arms": list(ARMS), "cells": {}}
    for n in (100, 200, 350, 500, 800, 1200):
        rows = []
        for path in sorted(glob.glob(
                os.path.join(RESULTS, f"t6_surrogate_n{n}_shard*.json"))):
            with open(path) as fh:
                rows.extend(json.load(fh)["rows"])
        if not rows:
            continue
        cell = {"reps": len(rows)}
        for arm in ARMS:
            sel = [r for r in rows if arm in r]
            if not sel:
                continue
            pvals = [r[arm]["p"] for r in sel]
            crit = [r[arm]["crit"] for r in sel]
            stats = [r[arm]["stat"] for r in sel]
            diag = [r[arm]["diagnostics"] for r in sel]
            bias = [d["bias_in_stat_units"] for dd in diag for d in dd]
            skew = [d["skew"] for dd in diag for d in dd]
            kurt = [d["excess_kurtosis"] for dd in diag for d in dd]
            cell[arm] = {
                "rate": _rate_summary(pvals),
                "crit_mean": float(np.mean(crit)),
                "stat_mean": float(np.mean(stats)),
                "stat_q95": float(np.quantile(stats, 0.95)),
                "mean_abs_bias_stat_units": float(np.mean(np.abs(bias))),
                "mean_skew": float(np.mean(skew)),
                "mean_abs_skew": float(np.mean(np.abs(skew))),
                "mean_excess_kurtosis": float(np.mean(kurt)),
            }
        out["cells"][f"n={n}"] = cell
    path = os.path.join(RESULTS, "t6_surrogate_summary.json")
    with open(path, "w") as fh:
        json.dump(_jsonify(out), fh, indent=1)
    print("wrote", path)
    return out


# ---------------------------------------------------------------------------
# synthetic q-growth check

def _score_block(rng, R, n, q, kind, rho):
    if kind == "gauss":
        z = rng.standard_normal((R, n, q))
    elif kind == "skew":
        a = (rng.random((R, n, q)) < 0.9).astype(float)
        z = np.where(a > 0, 1.0, -3.0)
        z = (z - 0.6) / 1.2
    elif kind == "twopoint":
        z = rng.choice(np.array([-1.0, 1.0]), size=(R, n, q))
    elif kind == "kurt":
        # symmetric four-point law with standardised excess kurtosis ~0.54,
        # matching the oracle confounded score law of Lemma T6.11
        u = rng.random((R, n, q))
        z = np.empty_like(u)
        z[u < 0.47] = 1.0
        z[(u >= 0.47) & (u < 0.94)] = -1.0
        z[(u >= 0.94) & (u < 0.97)] = 3.5
        z[u >= 0.97] = -3.5
        z = z / np.sqrt(1.675)
    else:
        raise ValueError(kind)
    if rho > 0:
        f = rng.standard_normal((R, n, 1))
        z = np.sqrt(rho) * f + np.sqrt(1.0 - rho) * z
    return z


def qcheck(*, n_grid=(100, 500), q_grid=(2, 10, 100, 1000),
           rho_grid=(0.0, 0.5), kinds=("gauss", "skew"),
           R=1000, R_mult=150, n_draws=499, seed0=90210):
    out = {"config": {"n_grid": list(n_grid), "q_grid": list(q_grid),
                      "rho_grid": list(rho_grid), "kinds": list(kinds),
                      "R": int(R), "R_mult": int(R_mult),
                      "n_draws": int(n_draws)}, "cells": {}}
    kind_code = {"gauss": 1, "skew": 2, "twopoint": 3, "kurt": 4}
    for kind in kinds:
        for rho in rho_grid:
            for q in q_grid:
                for n in n_grid:
                    rng = np.random.default_rng(
                        _seed("qcheck", kind_code[kind], int(rho * 100),
                              q, n, seed0))
                    z = _score_block(rng, R, n, q, kind, rho)
                    mu = z.mean(axis=1, keepdims=True)
                    centered = z - mu
                    sd = np.sqrt((centered ** 2).mean(axis=1, keepdims=True))
                    t_obs = np.max(np.sqrt(n) * np.abs(mu) / sd, axis=(1, 2))
                    q95_true = float(np.quantile(t_obs, 0.95))
                    Rm = min(R, R_mult)
                    crit = np.empty(Rm)
                    for b in range(Rm):
                        xi = rng.standard_normal((n_draws, n))
                        draws = (xi @ centered[b]) / np.sqrt(n)
                        per_draw = np.max(np.abs(draws) / sd[b, 0][None, :],
                                          axis=1)
                        crit[b] = np.quantile(per_draw, 1.0 - ALPHA)
                    rate_true = float((t_obs[:Rm] > crit).mean())
                    mc_se = float(np.sqrt(max(rate_true * (1 - rate_true), 1e-12) / Rm))
                    out["cells"][f"{kind}|rho={rho}|q={q}|n={n}"] = {
                        "q95_true": q95_true,
                        "q95_mult_mean": float(crit.mean()),
                        "size_at_mult_q95": rate_true,
                        "size_mc_se": mc_se,
                        "gap_from_alpha": rate_true - ALPHA,
                        "expected_max_scale": float(np.sqrt(2.0 * np.log(q))),
                    }
                    print(f"qcheck {kind} rho={rho} q={q:5d} n={n:5d} "
                          f"q95_true={q95_true:.3f} q95_mult={crit.mean():.3f} "
                          f"size={rate_true:.3f} (se {mc_se:.3f})")
    path = os.path.join(RESULTS, "t6_qcheck.json")
    with open(path, "w") as fh:
        json.dump(_jsonify(out), fh, indent=1)
    print("wrote", path)
    return out


# ---------------------------------------------------------------------------
# finite-grid non-implication witness

def witness():
    from tda2s.vec import persistence_measure
    w = (lambda p: float(abs(p[1] - p[0]) ** R))

    def feats(dgm):
        return persistence_measure([np.asarray(dgm, dtype=float).reshape(-1, 2)],
                                   weight=w, interval=INTERVAL,
                                   n_bins=N_BINS)[0].ravel()

    b, p = 0.10, 0.36
    delta = 0.02
    d1 = np.array([[b, b + p]])
    d2 = np.array([[b + delta, b + delta + p]])
    v1, v2 = feats(d1), feats(d2)
    same = bool(np.allclose(v1, v2, atol=0.0, rtol=0.0))
    # total-variation distance between the two atomic expected measures
    tv = 2.0 * p ** R
    # Euclidean (birth, mid) displacement between the atoms
    mid1, mid2 = (b + p / 2.0), (b + delta + p / 2.0)
    displacement = float(np.hypot(delta, mid2 - mid1))
    out = {
        "diagram_1": d1.tolist(),
        "diagram_2": d2.tolist(),
        "persistence": p,
        "weight": p ** R,
        "grid_bin_birth": _bin_index(b),
        "grid_bin_mid": _bin_index(b + p / 2.0),
        "projected_vectors_equal": same,
        "projected_l1": float(np.abs(v1 - v2).sum()),
        "tv_distance_between_measures": tv,
        "tv_convention": ("unnormalised total variation, 2*w for the two "
                          "one-atom measures; the normalised TV distance "
                          "between the one-atom probability measures is w"),
        "tv_distance_normalised": float(p ** R),
        "euclidean_atom_displacement": displacement,
        "note": ("bin width = 2/32 = 0.0625; birth drift 0.02 and mid drift "
                 "0.02 keep both atoms inside the same (birth, mid) cells"),
    }
    path = os.path.join(RESULTS, "t6_grid_witness.json")
    with open(path, "w") as fh:
        json.dump(_jsonify(out), fh, indent=1)
    print(json.dumps(_jsonify(out), indent=1))
    print("wrote", path)
    return out


# ---------------------------------------------------------------------------
# local (Pitman) power for the fixed-grid test

def _w1_contrast_pilot(reps=400, seed=1234):
    """Estimate S = ||E[V(3-blob)] - E[V(2-blob)]||_1 and the mean features."""
    rng = np.random.default_rng(seed)
    w = (lambda pp: float(abs(pp[1] - pp[0]) ** R))
    acc = {0: [], 1: []}
    for arm in (0, 1):
        for _ in range(reps):
            c = _w1_stoch_cloud(arm, rng)
            dgm = _threshold_diagrams([compute_diagrams(
                c, filtration="alpha", homology_dims=(0,))])[0][0]
            from tda2s.vec import persistence_measure
            v = persistence_measure([np.asarray(dgm, dtype=float).reshape(-1, 2)],
                                    weight=w, interval=INTERVAL,
                                    n_bins=N_BINS)[0].ravel()
            acc[arm].append(v)
    mu0 = np.mean(acc[0], axis=0)
    mu1 = np.mean(acc[1], axis=0)
    S = float(np.abs(mu1 - mu0).sum())
    return {"mu0": mu0, "mu1": mu1, "S": S}


def _local_rep(rep, n, h, S, n_draws=N_DRAWS):
    rng = np.random.default_rng(_seed("localpitman", rep, n, int(100 * h)))
    n_arm = n // 2
    labels = np.concatenate([np.zeros(n_arm, dtype=int),
                             np.ones(n - n_arm, dtype=int)])
    rng.shuffle(labels)
    p_n = h / (np.sqrt(n) * S)
    clouds = []
    n_switch = 0
    for a in labels:
        if a == 1 and rng.random() < p_n:
            clouds.append(_w1_stoch_cloud(1, rng))
            n_switch += 1
        else:
            clouds.append(_w1_stoch_cloud(0, rng))
    diagrams = _threshold_diagrams(
        [compute_diagrams(c, filtration="alpha", homology_dims=(0,)) for c in clouds])
    feats = measure_features(diagrams, interval=INTERVAL, n_bins=N_BINS,
                             weight_power=R, homology_dim=0)
    pi_hat, mu0, mu1 = _in_sample_arm_means(feats, labels)
    fit = fit_dist(diagrams, labels, np.zeros((n, 1)), pi_hat,
                   method="measure", interval=INTERVAL, n_bins=N_BINS,
                   weight_power=R, homology_dim=0, mu0_hat=mu0, mu1_hat=mu1)
    mult = dist_multiplier_test(fit, n_draws=n_draws, alpha=ALPHA,
                                seed=_seed("localpitman_mult", rep, n, int(100 * h)),
                                variance_floor=VARIANCE_FLOOR)
    scores = dist_scores(fit)
    active = np.asarray(mult["active"], dtype=bool)
    return {"rep": int(rep), "n": int(n), "h": float(h),
            "p_n": float(p_n), "n_switch": int(n_switch),
            "p": float(mult["pvalue"]), "stat": float(mult["statistic"]),
            "crit": float(np.quantile(mult["null"], 1.0 - ALPHA)),
            "scores": scores[:, active], "active": np.flatnonzero(active)}


def _local_job(job):
    return _local_rep(*job)


def local_shard(shard_idx, reps_per_shard, *, n, h, S, reps_total, workers=1):
    lo = shard_idx * reps_per_shard
    hi = min(lo + reps_per_shard, reps_total)
    jobs = [(rep, n, h, S) for rep in range(lo, hi)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(_local_job, jobs))
    else:
        rows = [_local_rep(*j) for j in jobs]
    path = os.path.join(RESULTS, f"t6_local_n{n}_h{h}_shard{shard_idx}.json")
    with open(path, "w") as fh:
        json.dump(_jsonify({"n": n, "h": h, "reps": [lo, hi], "rows": rows}), fh,
                  indent=1)
    print("wrote", path)
    return path


def _predicted_local_power(d_shift, cov, n_gauss=20000, seed=555):
    """Gaussian-shift prediction of the rejection probability.

    ``d_shift`` is the local shift vector in score units on the active
    coordinates; ``cov`` is their covariance.  The studentized vector has
    correlation ``R``; the null critical value is the Gaussian max quantile on
    the same correlation.
    """
    d_shift = np.asarray(d_shift, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    sd = np.sqrt(np.diag(cov))
    d_std = d_shift / sd
    Rmat = cov / np.outer(sd, sd)
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(Rmat + 1e-12 * np.eye(len(d_std)))
    Z = rng.standard_normal((n_gauss, len(d_std))) @ L.T
    T_null = np.max(np.abs(Z), axis=1)
    crit = float(np.quantile(T_null, 1.0 - ALPHA))
    power = float(np.mean(np.max(np.abs(Z + d_std[None, :]), axis=1) > crit))
    return {"shift_std": d_std, "crit": crit, "power": power,
            "n_shift_coords": int(len(d_std))}


def local_aggregate():
    out = {"pilot": None, "cells": {}}
    pilot_path = os.path.join(RESULTS, "t6_local_pilot.json")
    if os.path.exists(pilot_path):
        with open(pilot_path) as fh:
            pilot = json.load(fh)
    else:
        pilot = None
    if pilot is not None:
        out["pilot"] = {"source": "t6_local_pilot.json",
                        "S": float(pilot["S"])}
    for n in (200, 500):
        for h in (1.0, 2.0, 6.0):
            rows = []
            for path in sorted(glob.glob(
                    os.path.join(RESULTS, f"t6_local_n{n}_h{h}_shard*.json"))):
                with open(path) as fh:
                    rows.extend(json.load(fh)["rows"])
            if not rows:
                continue
            pv = np.array([r["p"] for r in rows])
            rate = float((pv < ALPHA).mean())
            mc_se = float(np.sqrt(max(rate * (1 - rate), 1e-12) / len(rows)))
            cell = {"reps": len(rows), "rate": rate, "mc_se": mc_se}
            # prediction from the pooled score covariance on the coordinates
            # that are active in every replication (the active set varies a
            # little across W1 draws).
            if pilot is not None:
                mu0 = np.asarray(pilot["mu0"])
                mu1 = np.asarray(pilot["mu1"])
                S = float(pilot["S"])
                usable = [r for r in rows if np.asarray(r["scores"]).size]
                if usable:
                    sets = [set(np.asarray(r["active"], dtype=int).tolist())
                            for r in usable]
                    ref = sorted(set.intersection(*sets))
                    covs = []
                    for r in usable:
                        positions = [list(np.asarray(r["active"],
                                                     dtype=int)).index(j)
                                     for j in ref]
                        covs.append(np.cov(
                            np.asarray(r["scores"])[:, positions].T, ddof=1))
                    cov = np.mean(covs, axis=0)
                    if cov.ndim == 0:
                        cov = np.array([[float(cov)]])
                    ref_arr = np.asarray(ref, dtype=int)
                    d_shift = h / S * (mu1[ref_arr] - mu0[ref_arr])
                    pred = _predicted_local_power(d_shift, cov)
                    cell["prediction"] = pred
                    cell["gap"] = rate - pred["power"]
                    cell["active_set_intersection"] = ref
            out["cells"][f"n={n}|h={h}"] = cell
    path = os.path.join(RESULTS, "t6_local_summary.json")
    with open(path, "w") as fh:
        json.dump(_jsonify(out), fh, indent=1)
    print(json.dumps(_jsonify(out), indent=1))
    print("wrote", path)
    return out


# ---------------------------------------------------------------------------
# driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=("existing", "mechanism-shard", "mechanism-aggregate",
                             "surrogate-shard", "surrogate-aggregate",
                             "qcheck", "witness", "local-pilot", "local-shard",
                             "local-aggregate", "summary"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--h", type=float, default=1.0)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--reps-per-shard", type=int, default=50)
    ap.add_argument("--reps-total", type=int, default=400)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--arms", type=str, default=",".join(ARMS))
    ap.add_argument("--regime", type=float, default=REGIME)
    args = ap.parse_args()
    arms = tuple(a for a in args.arms.split(",") if a)
    if args.mode == "existing":
        existing_summary()
    elif args.mode == "mechanism-shard":
        mechanism_shard(args.shard_idx, args.reps_per_shard, n=args.n,
                        reps_total=args.reps_total, workers=args.workers)
    elif args.mode == "mechanism-aggregate":
        mechanism_aggregate()
    elif args.mode == "surrogate-shard":
        surrogate_shard(args.shard_idx, args.reps_per_shard, n=args.n,
                        reps_total=args.reps_total, workers=args.workers,
                        arms=arms, regime=args.regime)
    elif args.mode == "surrogate-aggregate":
        surrogate_aggregate()
    elif args.mode == "qcheck":
        qcheck()
    elif args.mode == "witness":
        witness()
    elif args.mode == "local-pilot":
        pilot = _w1_contrast_pilot()
        path = os.path.join(RESULTS, "t6_local_pilot.json")
        with open(path, "w") as fh:
            json.dump(_jsonify(pilot), fh, indent=1)
        print("S =", pilot["S"], "-> wrote", path)
    elif args.mode == "local-shard":
        with open(os.path.join(RESULTS, "t6_local_pilot.json")) as fh:
            S = float(json.load(fh)["S"])
        local_shard(args.shard_idx, args.reps_per_shard, n=args.n, h=args.h,
                    S=S, reps_total=args.reps_total, workers=args.workers)
    elif args.mode == "local-aggregate":
        local_aggregate()
    elif args.mode == "summary":
        out = {}
        for name in ("t6_existing_summary", "t6_surrogate_summary",
                     "t6_mechanism_analysis", "t6_oracle_score_moments",
                     "t6_qcheck", "t6_grid_witness", "t6_local_summary"):
            path = os.path.join(RESULTS, name + ".json")
            if os.path.exists(path):
                with open(path) as fh:
                    out[name] = json.load(fh)
        path = os.path.join(RESULTS, "t6_summary.json")
        with open(path, "w") as fh:
            json.dump(out, fh, indent=1)
        print("wrote", path)


if __name__ == "__main__":
    main()
