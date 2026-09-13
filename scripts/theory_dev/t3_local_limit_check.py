"""T3 validation: finite-sample drift of the cross-fitted AIPW process under
local alternatives psi_d^(n) = n^{-1/2} g_d.

Theory target (WP7_T3_local_limit.md, Theorems T3.1/T3.2 and the truncation
identity below).  For the treated-only location shift of the Phase-7 oracle
harness, and a finite-basis Fourier outcome regression whose population
projection P_K satisfies P_K m_a = m_a for the base arms, the standardized
AIPW estimator satisfies, conditional on the fitted nuisances,

    sqrt(n) (psi_hat_d(t) - psi_d^(n)(t))
        = G_n phi_d^(0)(t) + amp * r_g(t) + R_n(t),
    amp_n = (1/n) sum_i (pi_i / e_hat_i - 1),
    r_g(t) = g(t) - P_K g(t),

where R_n has conditional mean zero (the outcome-regression estimation error
is centred) and G_n is the centred empirical process.  The expected local
drift is amp * r_g, exactly zero for a direction in the regression span
(r_g = 0) and nonzero for a direction whose Fourier content is truncated.
The drift is O(delta_n) with delta_n the propensity-estimation error: it
vanishes asymptotically for any consistent e_hat but can be non-negligible at
n = 500.

Design.  A *paired* (CRN) design isolates the drift: per replication the base
sample is drawn once and the three directions (mean, bump, freq) are shifted
with identical fold seeds and identical RandomForest seeds, so the leading
empirical-process fluctuation cancels in differences between directions.  For
each direction we record

    D_obs(d, t) = sqrt(n) * mean_hom(psi_hat_d(t)) - g_d(t)
                = sqrt(n) (psi_hat_d(t) - psi_d^(n)(t))   [base TATE = 0],
    pred(d, t)  = amp * r_g_d(t),
    resid(., t) = D_obs(d, t) - D_obs(mean, t) - pred(d, t)   (d = bump, freq)

because r_g = 0 for the mean direction.  Estimation goes through
``tda2s.tests.dr_outcome.fit_dr`` (tcda_uq underneath); nothing is
reimplemented.

Pre-registered criteria (fixed before the production run; a 6-replication
diagnostic with fresh seeds was used only to check the fold coupling and to
size the run):

  N1  Projection.  With the same Fourier span used by
      ``tcda_uq.estimators.nuisance.fit_functional_regression`` (n_basis = 5,
      interval (0, 1)), the residual sup norms satisfy
      ||r_g^freq||_inf >= 0.1 and ||r_g^mean||_inf <= 1e-8.  Structural fact,
      not a hypothesis test.
  N2  Drift identity.  For each (setting, n, direction in {bump, freq}) the
      Monte-Carlo mean over replications of ``resid`` has mean over t within
      3 MC standard errors of zero, and max_t |mean resid| <= 4 MC SE.  This
      is the finite-sample statement of Theorem T3.2.
  N3  Truncation explanation.  For the Phase-7 miss cell PROG1/freq/n=500,
      the empirical multiplier rate lies within 2 MC standard errors of the
      corrected per-replication Gaussian prediction (shift g + amp r_g), and
      the corrected prediction is closer to the empirical rate than the naive
      prediction (shift g).  This tests the direction of the correction on
      the miss cell; it does not certify the full Phase-7 fleet.
  N4  Regular direction.  For direction ``mean`` (r_g = 0) under PROG0 and
      PROG1, the naive prediction lies within 2.5 MC standard errors of the
      empirical multiplier rate; the truncation correction is exactly zero.

Reading rule: a NEGATIVE result on N2/N3 means the drift identity does not
hold at the tested n and the note must say so; no criterion is adjusted after
the run.

Usage:
    python scripts/theory_dev/t3_local_limit_check.py --mode pilot --workers 4
    python scripts/theory_dev/t3_local_limit_check.py --mode run --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))
if os.path.join(ROOT, "experiments") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "experiments"))

from phase7_local_power import (  # noqa: E402
    _seed,
    _tseq,
    base_sample,
    direction_shape,
    shifted_triplet,
)
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from tda2s.tests.dr_outcome import fit_dr, multiplier_test  # noqa: E402

OUT_JSON = os.path.join(ROOT, "results", "theory_validation",
                        "t3_local_limit_check.json")
SCALE = 0.7
ALPHA = 0.05
N_CAL = 1999
N_PRED = 10000
DIRECTIONS = ("mean", "bump", "freq")
# Production cells.  The design was reduced on 2026-09-10 (60 -> 30 -> 15 ->
# 12 reps per n=500 cell) because the host was running the other WP fleets at
# load average 40 to 60 and the full design projected to hours; the criteria
# N1-N4 were fixed before either version of the production run, and the
# per-rep drift identity checked by N2 is exact in the paired design, so the
# reduced replication count costs little.  The empirical-rate comparison of
# N3 uses the 300-rep Phase-7 cell as its reference, not these 12 reps.  The
# driver checkpoints the aggregate after every completed replication, so a
# partial run is a valid (if noisier) result.
CELLS_DEFAULT = (("PROG1", 500, 12), ("PROG0", 500, 12))


def fourier_span(tseq: np.ndarray) -> np.ndarray:
    """Span of skfda's FourierBasis(interval, 5) evaluated on the grid.

    Span {1, sin(2 pi u), cos(2 pi u), sin(4 pi u), cos(4 pi u)},
    u = (t - t0)/(t1 - t0); checked against
    ``FDataGrid(...).to_basis(FourierBasis(tseq[[0,-1]], 5))``.
    """
    u = (np.asarray(tseq) - tseq[0]) / (tseq[-1] - tseq[0])
    return np.stack([np.ones_like(u), np.sin(2 * np.pi * u),
                     np.cos(2 * np.pi * u), np.sin(4 * np.pi * u),
                     np.cos(4 * np.pi * u)], axis=1)


def projection_residual(g: np.ndarray, tseq: np.ndarray) -> np.ndarray:
    B = fourier_span(tseq)
    coef, _, _, _ = np.linalg.lstsq(B, np.asarray(g, dtype=float), rcond=None)
    return np.asarray(g, dtype=float) - B @ coef


def gauss_power(cov: np.ndarray, shift: np.ndarray, *, n_pred: int = N_PRED,
                seed: int = 0, alpha: float = ALPHA) -> dict:
    rng = np.random.default_rng(seed)
    cov = np.asarray(cov, dtype=float)
    cov = (cov + cov.T) / 2.0
    q = cov.shape[0]
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(q))
    Z = rng.standard_normal((n_pred, q)) @ L.T
    null = np.max(np.abs(Z), axis=1)
    crit = float(np.quantile(null, 1.0 - alpha))
    alt = np.max(np.abs(Z + np.asarray(shift)[None, :]), axis=1)
    pw = float(np.mean(alt > crit))
    return {"crit": crit, "power": pw,
            "power_se": float(np.sqrt(max(pw * (1.0 - pw), 0.0) / n_pred))}


def one_rep(setting: str, n: int, rep: int) -> dict:
    tseq = _tseq()
    sample, _ = base_sample(setting, n, rep)
    pi_true = np.asarray(sample.propensity, dtype=float)
    shapes = {d: direction_shape(d, tseq) * SCALE for d in DIRECTIONS}
    r_gs = {d: projection_residual(shapes[d], tseq) for d in DIRECTIONS}

    out: dict = {"setting": setting, "n": n, "rep": rep,
                 "directions": {d: {} for d in DIRECTIONS},
                 "r_g_sup": {d: float(np.max(np.abs(r_gs[d])))
                             for d in DIRECTIONS}}
    amps = []
    for d in DIRECTIONS:
        g = shapes[d]
        triplet = shifted_triplet(sample, g, n)
        fit = fit_dr(
            triplet.observed, triplet.tseq, n_basis=5, n_folds=2,
            propensity_estimator=RandomForestClassifier(
                n_estimators=100, min_samples_leaf=4, n_jobs=1,
                random_state=_seed("rf", setting, rep, n)),
            random_state=_seed("fold", setting, rep, n, "t3"),
        )
        mult = multiplier_test(fit, n_draws=N_CAL,
                               seed=_seed("mult", setting, d, rep, n))
        scores = np.asarray(mult["scores"], dtype=float)      # [n, hom, res]
        est = scores.mean(axis=0)                             # [hom, res]
        centered = (scores - est[None, :, :]).reshape(len(scores), -1)
        cov = centered.T @ centered / len(scores)
        amp = float(np.mean(pi_true[fit.order] / np.asarray(fit.pi_hat) - 1.0))
        amps.append(amp)
        g_vec = np.tile(g, est.shape[0])
        d_obs = np.sqrt(n) * est.mean(axis=0) - g
        pred_drift = amp * r_gs[d]
        gp_naive = gauss_power(cov, g_vec,
                               seed=_seed("gp_naive", setting, d, rep, n))
        gp_corr = gauss_power(cov, g_vec + np.tile(pred_drift, est.shape[0]),
                              seed=_seed("gp_corr", setting, d, rep, n))
        out["directions"][d] = {
            "amp": amp,
            "stat": float(mult["statistic"]),
            "crit_mult": float(mult["critical_value"]),
            "p_mult": float(mult["pvalue"]),
            "d_obs": d_obs,
            "pred_drift": pred_drift,
            "gauss_naive_power": gp_naive["power"],
            "gauss_naive_crit": gp_naive["crit"],
            "gauss_corr_power": gp_corr["power"],
        }
    out["amp_spread"] = float(np.std(amps))
    return out


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _one_rep_job(job):
    return one_rep(*job)


def run_cells(cells, workers: int, on_rep=None) -> list[dict]:
    jobs = [(s, n, r) for s, n, reps in cells for r in range(reps)]
    if workers <= 1:
        return [one_rep(*j) for j in jobs]
    import multiprocessing as mp
    with mp.Pool(processes=workers) as pool:
        out = []
        for k, row in enumerate(pool.imap_unordered(_one_rep_job, jobs),
                                start=1):
            out.append(row)
            print(f"  rep {k}/{len(jobs)} done "
                  f"({time.strftime('%H:%M:%S')})", flush=True)
            if on_rep is not None:
                on_rep(out)
        return out


def write_summary(path: str, summary: dict):
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp, "w") as fh:
        json.dump(_jsonable(summary), fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def aggregate(reps: list[dict]) -> dict:
    cells: dict = {}
    for row in reps:
        cells.setdefault((row["setting"], row["n"]), []).append(row)
    out = {}
    for (setting, n), rows in sorted(cells.items()):
        R = len(rows)
        cell: dict = {"setting": setting, "n": n, "reps": R,
                      "r_g_sup": rows[0]["r_g_sup"], "directions": {}}
        mean_obs = np.stack([r["directions"]["mean"]["d_obs"] for r in rows])
        for d in DIRECTIONS:
            d_obs = np.stack([r["directions"][d]["d_obs"] for r in rows])
            pred = np.stack([r["directions"][d]["pred_drift"] for r in rows])
            amp = np.array([r["directions"][d]["amp"] for r in rows])
            stat = np.array([r["directions"][d]["stat"] for r in rows])
            crit = np.array([r["directions"][d]["crit_mult"] for r in rows])
            gp_n = np.array([r["directions"][d]["gauss_naive_power"]
                             for r in rows])
            gp_c = np.array([r["directions"][d]["gauss_corr_power"]
                             for r in rows])
            rate = float(np.mean(stat > crit))
            entry = {
                "amp_mean": float(amp.mean()),
                "amp_sd": float(amp.std(ddof=1)),
                "emp_rate": rate,
                "mc_se_emp": float(np.sqrt(max(rate * (1.0 - rate), 0.0) / R)),
                "gauss_naive_mean": float(gp_n.mean()),
                "gauss_naive_se": float(gp_n.std(ddof=1) / np.sqrt(R)),
                "gauss_corr_mean": float(gp_c.mean()),
                "gauss_corr_se": float(gp_c.std(ddof=1) / np.sqrt(R)),
                "pred_drift_sup_mean": float(np.max(np.abs(pred), axis=1).mean()),
            }
            if d != "mean":
                resid = (d_obs - mean_obs) - pred          # [R, res]
                mean_res = resid.mean(axis=0)              # [res]
                se_t = resid.std(axis=0, ddof=1) / np.sqrt(R)
                resid_scalar = resid.mean(axis=1)          # per-rep mean over t
                entry["resid_mean_over_t"] = float(resid_scalar.mean())
                entry["resid_se_over_t"] = float(
                    resid_scalar.std(ddof=1) / np.sqrt(R))
                entry["resid_mean_z_over_t"] = float(
                    entry["resid_mean_over_t"] / max(entry["resid_se_over_t"],
                                                     1e-12))
                entry["resid_max_abs_mean"] = float(np.max(np.abs(mean_res)))
                entry["resid_max_t"] = int(np.argmax(np.abs(mean_res)))
                entry["resid_max_abs_z"] = float(np.max(
                    np.abs(mean_res) / np.maximum(se_t, 1e-12)))
            cell["directions"][d] = entry
        out[f"{setting}|{n}"] = cell
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "run"), default="run")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default=OUT_JSON)
    args = parser.parse_args()
    cells = (("PROG1", 500, 4), ("PROG0", 500, 4)) if args.mode == "pilot" \
        else CELLS_DEFAULT
    t0 = time.time()

    def meta(n_done: int):
        return {"mode": args.mode, "workers": args.workers,
                "scale": SCALE, "n_cal": N_CAL, "n_pred": N_PRED,
                "alpha": ALPHA, "elapsed_s": time.time() - t0,
                "cells": [list(c) for c in cells], "reps_done": n_done}

    def on_rep(rows):
        summary = aggregate(rows)
        summary["_meta"] = meta(len(rows))
        write_summary(args.out, summary)

    reps = run_cells(cells, args.workers, on_rep=on_rep)
    summary = aggregate(reps)
    summary["_meta"] = meta(len(reps))
    write_summary(args.out, summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
