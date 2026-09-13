"""Phase 7.5 — numerical falsification for local power (C4).

Simulates the local-alternative sequence  psi_d^{(n)} = n^{-1/2} g  on the
Phase-3 oracle harness (``tcda_uq.datasets.TriOracleSimulation`` with known
TATE), for three fixed directions g (mean-shift, scale-localized bump,
high-frequency oscillation), n in {100, 200, 500}, 300 replications per cell,
in three settings:

  * PROG0    : prognostic outcomes, randomised assignment  (regime/lambda = 0)
  * PROG1    : prognostic outcomes, confounded assignment  (regime/lambda = 1)
  * NONPROG1 : X-independent outcomes, confounded assignment (regime = 1)

plus two null (h = 0) regression cells at n = 200 (PROG0, PROG1).

Per replication the script records rejection indicators of
  (i)   the DR max-statistic with shared-multiplier calibration,
  (ii)  the DR max-statistic with frozen stratified-permutation calibration,
  (iii) the unadjusted difference-of-means max-statistic with a free-label
        permutation null (the honest-relabelling comparator of task 7.2),
and accumulates the cross-rep outer products of the centred DR scores and of
the unadjusted influence vectors.  Those covariances give the *predicted*
Gaussian-shift limit power  P(||G + g||_inf > c_{1-alpha})  by direct Monte
Carlo simulation of the limiting Gaussian vector — computed from null
structure only, never fitted to the empirical rejection rates.

Falsification rule (pre-registered, see ``aggregate``): the Gaussian-shift
prediction is confirmed iff every DR cell at n = 500 agrees with its
prediction within 0.06 absolute, the unadjusted test agrees within 0.06
where it is valid (PROG0, NONPROG1), the unadjusted PROG1-null cell breaks
size (> 0.20), and the DR PROG1-null cell holds size (<= 0.08).  Any
violation is reported as MISMATCH: the theory note must then NOT be written
to fit the numerics.

Estimator settings match the Phase-3 oracle fleet exactly (N_BASIS = 5,
N_FOLDS = 2, RF propensity, resolution 50, interval (0, 1), B = 399, known-
propensity strata with 8 bins), so Phase-3 null rates are a valid external
anchor.  No AIPW / cross-fitting code is reimplemented here: estimation goes
through ``tda2s.tests.dr_outcome.fit_dr`` (tcda_uq underneath).

Usage:
    python experiments/phase7_local_power.py --mode smoke      # fast end-to-end check
    python experiments/phase7_local_power.py --mode pilot      # null pilot -> scale choice
    python experiments/phase7_local_power.py --mode run --workers 8 [--reps 300 --resume]
    python experiments/phase7_local_power.py --mode aggregate  # -> results/phase7_local_power.json
    python experiments/phase7_local_power.py --mode figure     # -> results/phase7_local_power.png
"""

from __future__ import annotations

import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np

from tda2s.resample import p_value
from tda2s.tests.dr_outcome import (
    fit_dr,
    multiplier_test,
    propensity_strata,
    stratified_permutation_test,
)

BASE_SEED = 7100
ALPHA = 0.05
N_GRID = (100, 200, 500)
DIRECTIONS = ("mean", "bump", "freq")
N_BASIS = 5
N_FOLDS = 2
N_CAL = 399
INTERVAL = (0.0, 1.0)
RESOLUTION = 50
REPS_DEFAULT = 300
NSTRATA = 8
N_DRAWS_PRED = 20000
TOL_CONFIRM = 0.06
_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, "..", "results")
CHECKPOINT = os.path.join(RESULTS, "phase7_checkpoint.json")
SUMMARY = os.path.join(RESULTS, "phase7_local_power.json")
FIGURE = os.path.join(RESULTS, "phase7_local_power.png")

SETTINGS = ("PROG0", "PROG1", "NONPROG1", "NONPROG0")
REGIME_OF = {"PROG0": 0.0, "PROG1": 1.0, "NONPROG1": 1.0, "NONPROG0": 0.0}
# NONPROG0 = randomised assignment + X-independent outcomes: the exact
# ARE=1 design of Thm 7.2(a) (identical limit laws, no IPW inflation since
# e = 1/2, no prognostic signal). Added after the NONPROG1 partials showed
# the IPW-variance tax under imbalance (WP7 §5).


def _seed(*parts) -> int:
    value = BASE_SEED
    for part in parts:
        if isinstance(part, str):
            part = sum((i + 1) * b for i, b in enumerate(part.encode()))
        value = (value * 7919 + int(part)) % (2 ** 31 - 1)
    return int(value)


def _tseq() -> np.ndarray:
    return np.linspace(INTERVAL[0], INTERVAL[1], RESOLUTION)


def direction_shape(name: str, tseq: np.ndarray) -> np.ndarray:
    """Fixed unit-sup-norm direction g(t) on the silhouette grid."""
    t = np.asarray(tseq, dtype=float)
    if name == "mean":
        g = np.ones_like(t)
    elif name == "bump":
        g = np.exp(-((t - 0.5) ** 2) / (2.0 * 0.08 ** 2))
    elif name == "freq":
        g = np.cos(2.0 * np.pi * 3.0 * t)
    else:
        raise ValueError(f"unknown direction {name}")
    g = g / np.max(np.abs(g))
    return g


def _null_sim(regime: float, rep: int):
    """TriOracleSimulation under the sharp null (equal arm coefficients)."""
    from tcda_uq.datasets import TriOracleSimulation

    sim = TriOracleSimulation(
        n_cov=3, n_hom_dim=2, resolution=RESOLUTION, interval=INTERVAL,
        n_basis=N_BASIS, coef_scale=0.6, noise_scale=0.15,
        prop_scale=1.5 * regime, seed=_seed("model", rep),
    )
    for d in range(sim.n_hom_dim):
        sim.Gamma[1][d] = sim.Gamma[0][d].copy()
    return sim


def base_sample(setting: str, n: int, rep: int):
    """Draw one null base sample; NONPROG1 strips the X-dependence of outcomes.

    Returns (sample, sim) where sample has .observed/.tseq/.propensity.
    The returned potential outcomes have exactly zero TATE by construction.
    """
    regime = REGIME_OF[setting]
    sim = _null_sim(regime, rep)
    sample = sim.sample(n, rng=_seed("sample", setting, rep, n))
    if setting in ("NONPROG1", "NONPROG0"):
        # X-INDEPENDENT potential outcomes: common constant mean plus fresh
        # noise draws from sim._noise (which ignores X at hetero_scale=0).
        # An earlier version subtracted only the pooled sample mean, leaving
        # mu_a(X)-structure intact (bogus non-prognostic design); the cells
        # below replace those runs (see §5 docstring / WP7 §5).
        pot = np.asarray(sample.potential_outcomes, dtype=float)  # [n,2,hom,res]
        mbar = pot.mean(axis=(0, 1))                              # [hom,res]
        rng_n = np.random.default_rng(_seed("nonprog-noise", rep, n))
        pot_new = np.empty_like(pot)
        pot_new[:, 0] = mbar[None, :, :] + sim._noise(n, rng_n, None)
        pot_new[:, 1] = mbar[None, :, :] + sim._noise(n, rng_n, None)
        sample = SimpleNamespace(
            tseq=sample.tseq, X=sample.X, A=sample.A,
            propensity=sample.propensity, potential_outcomes=pot_new,
            observed=None,
        )
        phi = pot_new[np.arange(n), sample.A]
        sample.observed = (phi, sample.A, sample.X)
    return sample, sim


def shifted_triplet(sample, shift_curve: np.ndarray | None, n: int):
    """Apply the local shift n^{-1/2} g to the treated potential outcome.

    shift_curve is [res] (added to every homology dim) or None for the null.
    """
    pot = np.asarray(sample.potential_outcomes, dtype=float).copy()
    if shift_curve is not None:
        g = np.asarray(shift_curve, dtype=float)
        pot[:, 1, :, :] = pot[:, 1, :, :] + g[None, None, :] / np.sqrt(n)
    A = np.asarray(sample.A)
    phi = pot[np.arange(len(A)), A]
    return SimpleNamespace(
        observed=(phi, A, np.asarray(sample.X)),
        tseq=np.asarray(sample.tseq),
        propensity=np.asarray(sample.propensity),
    )


def unadjusted_permutation_test(phi: np.ndarray, A: np.ndarray, *,
                                n_perm: int = N_CAL, seed: int = 0) -> dict:
    """Unadjusted max-statistic with a free-label permutation null.

    T = sqrt(n) max_d sup_t |bar Z_1 - bar Z_0|.  Under randomisation
    (PROG0 null) the labels are exchangeable so the test is exact; under
    confounding it is the intentionally naive comparator of task 7.2.
    """
    phi = np.asarray(phi, dtype=float)
    A = np.asarray(A, dtype=int)
    n = len(A)
    rng = np.random.default_rng(seed)
    m1 = phi[A == 1].mean(axis=0)
    m0 = phi[A == 0].mean(axis=0)
    stat = float(np.sqrt(n) * np.max(np.abs(m1 - m0)))
    n1 = int((A == 1).sum())
    null = np.empty(n_perm)
    Acopy = A.copy()
    for b in range(n_perm):
        perm = rng.permutation(n)
        p1 = phi[perm[:n1]].mean(axis=0)
        p0 = phi[perm[n1:]].mean(axis=0)
        null[b] = float(np.sqrt(n) * np.max(np.abs(p1 - p0)))
    return {"statistic": stat, "pvalue": p_value(stat, null), "null": null}


def _one_local_fit(triplet, rep: int, n: int, setting: str, tag: str,
                   n_calibration: int = N_CAL) -> dict:
    """Fit DR + all three calibrations on one (possibly shifted) sample."""
    from sklearn.ensemble import RandomForestClassifier

    fit = fit_dr(
        triplet.observed, triplet.tseq, n_basis=N_BASIS, n_folds=N_FOLDS,
        propensity_estimator=RandomForestClassifier(
            n_estimators=100, min_samples_leaf=4, n_jobs=1,
            random_state=_seed("rf", setting, rep, n)),
        random_state=_seed("fold", setting, rep, n, tag),
    )
    strata = propensity_strata(triplet.propensity, n_bins=NSTRATA)
    mult = multiplier_test(
        fit, n_draws=n_calibration,
        seed=_seed("mult", setting, tag, rep, n))
    perm = stratified_permutation_test(
        fit, strata, n_perm=n_calibration,
        seed=_seed("perm", setting, tag, rep, n))
    phi, A, _ = triplet.observed
    unadj = unadjusted_permutation_test(
        np.asarray(phi), np.asarray(A), n_perm=n_calibration,
        seed=_seed("unadj", setting, tag, rep, n))
    scores = np.asarray(mult["scores"], dtype=float)          # [n,hom,res] fold order
    est = np.asarray(mult["estimate"], dtype=float)           # [hom,res]
    centered = (scores - est[None, :, :]).reshape(len(scores), -1)  # [n,q]
    cov_dr = centered.T @ centered                             # [q,q] (unnormalised)
    # Unadjusted influence vectors: n * (A/n1 - (1-A)/n0) * (Z - arm mean).
    Z = np.asarray(phi, dtype=float).reshape(len(A), -1)
    A1 = (np.asarray(A) == 1)
    n1, n0 = int(A1.sum()), int((~A1).sum())
    infl = np.empty_like(Z)
    infl[A1] = (Z[A1] - Z[A1].mean(axis=0)) * (len(A) / n1)
    infl[~A1] = -(Z[~A1] - Z[~A1].mean(axis=0)) * (len(A) / n0)
    # Unnormalised outer product: aggregate() divides the rep-sum by total n,
    # exactly as for cov_dr above. (A previous version divided by n here AND
    # in aggregate, shrinking Sigma_unadj by n and sending the predicted
    # unadjusted power to ~1; fixed before the production fleet.)
    cov_unadj = infl.T @ infl
    return {
        "multiplier_p": float(mult["pvalue"]),
        "permutation_p": float(perm["pvalue"]),
        "unadjusted_p": float(unadj["pvalue"]),
        "statistic": float(mult["statistic"]),
        "estimate_sup": float(np.max(np.abs(est))),
        "cov_dr": cov_dr, "cov_unadj": cov_unadj, "n": int(len(A)),
    }


def cells_for(reps: int, n_grid=N_GRID, directions=DIRECTIONS):
    """Full cell list: (setting, direction-or-None, n). None = null cell."""
    cells = []
    for setting in SETTINGS:
        if setting in ("NONPROG1", "NONPROG0"):
            ns = (200, 500)
        else:
            ns = tuple(n_grid)
        for n in ns:
            for d in directions:
                cells.append((setting, d, n))
    cells.append(("PROG0", None, 200))
    cells.append(("PROG1", None, 200))
    cells.append(("NONPROG1", None, 200))
    cells.append(("NONPROG0", None, 200))
    return cells


def _task_key(setting, direction, n, lo, hi):
    return f"{setting}|{direction}|{n}|{lo}-{hi}"


def run_task(payload: dict) -> dict:
    """Worker: one (setting, n, rep-chunk); CRN across directions via base draws."""
    setting = payload["setting"]
    n = payload["n"]
    directions = payload["directions"]
    want_null = payload["want_null"]
    scale = payload["scale"]
    n_calibration = payload["n_calibration"]
    tseq = _tseq()
    shapes = {d: direction_shape(d, tseq) * scale for d in directions}
    out = {}
    for tag in directions + ([None] if want_null else []):
        out[_task_key(setting, tag, n, payload["lo"], payload["hi"])] = {
            "mult": [], "perm": [], "unadj": [],
            "cov_dr": np.zeros((2 * RESOLUTION, 2 * RESOLUTION)),
            "cov_unadj": np.zeros((2 * RESOLUTION, 2 * RESOLUTION)),
            "count_n": 0, "n": n,
        }
    for rep in range(payload["lo"], payload["hi"]):
        sample, _ = base_sample(setting, n, rep)
        jobs = [(d, shapes[d]) for d in directions] + ([(None, None)] if want_null else [])
        for tag, shape in jobs:
            triplet = shifted_triplet(sample, shape, n)
            r = _one_local_fit(triplet, rep, n, setting, str(tag),
                               n_calibration=n_calibration)
            key = _task_key(setting, tag, n, payload["lo"], payload["hi"])
            acc = out[key]
            acc["mult"].append(r["multiplier_p"])
            acc["perm"].append(r["permutation_p"])
            acc["unadj"].append(r["unadjusted_p"])
            acc["cov_dr"] += r["cov_dr"]
            acc["cov_unadj"] += r["cov_unadj"]
            acc["count_n"] += r["n"]
    for acc in out.values():
        acc["cov_dr"] = acc["cov_dr"].tolist()
        acc["cov_unadj"] = acc["cov_unadj"].tolist()
    return out


def build_tasks(reps: int, chunk: int, scale: float, n_calibration: int,
                directions=DIRECTIONS):
    tasks = []
    for setting in SETTINGS:
        # NONPROG1 runs at n=200 (ARE finite-n record) and n=500 (tax decay).
        # NONPROG0 (exact ARE=1 design) runs at both n for the same reason.
        if setting in ("NONPROG1", "NONPROG0"):
            ns = (200, 500)
        else:
            ns = tuple(N_GRID)
        for n in ns:
            want_null = (n == 200)  # all three settings carry a null cell
            for lo in range(0, reps, chunk):
                tasks.append({
                    "setting": setting, "n": n, "lo": lo,
                    "hi": min(lo + chunk, reps), "directions": list(directions),
                    "want_null": want_null, "scale": scale,
                    "n_calibration": n_calibration,
                })
    return tasks


def _merge_into(agg: dict, partial: dict):
    for key, acc in partial.items():
        dest = agg.setdefault(key, {
            "mult": [], "perm": [], "unadj": [],
            "cov_dr": np.zeros((2 * RESOLUTION, 2 * RESOLUTION)),
            "cov_unadj": np.zeros((2 * RESOLUTION, 2 * RESOLUTION)),
            "count_n": 0, "n": acc["n"],
        })
        dest["mult"].extend(acc["mult"])
        dest["perm"].extend(acc["perm"])
        dest["unadj"].extend(acc["unadj"])
        dest["cov_dr"] += np.asarray(acc["cov_dr"])
        dest["cov_unadj"] += np.asarray(acc["cov_unadj"])
        dest["count_n"] += acc["count_n"]


def run_fleet(reps: int, workers: int, chunk: int, scale: float,
              n_calibration: int, resume: bool, directions=DIRECTIONS):
    import multiprocessing as mp

    os.makedirs(RESULTS, exist_ok=True)
    tasks = build_tasks(reps, chunk, scale, n_calibration, directions)

    def task_id(t):
        return f'{t["setting"]}|{t["n"]}|{t["lo"]}-{t["hi"]}'

    agg: dict = {}
    done_ids: set = set()
    if resume and os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as fh:
            payload = json.load(fh)
        done_ids = set(payload.get("done", []))
        for key, acc in payload.get("agg", {}).items():
            agg[key] = {
                "mult": acc["mult"], "perm": acc["perm"],
                "unadj": acc["unadj"],
                "cov_dr": np.asarray(acc["cov_dr"]),
                "cov_unadj": np.asarray(acc["cov_unadj"]),
                "count_n": acc["count_n"], "n": acc["n"],
            }
    pending = [t for t in tasks if task_id(t) not in done_ids]
    print(f"fleet: {len(tasks)} tasks, {len(pending)} pending, "
          f"{workers} workers, scale={scale}", flush=True)
    t0 = time.time()
    finished = 0

    def save():
        serial = {k: {**v, "cov_dr": np.asarray(v["cov_dr"]).tolist(),
                      "cov_unadj": np.asarray(v["cov_unadj"]).tolist()}
                  for k, v in agg.items()}
        tmp = CHECKPOINT + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"done": sorted(done_ids), "agg": serial,
                       "meta": {"scale": scale, "reps": reps,
                                "n_calibration": n_calibration,
                                "directions": list(directions)}}, fh)
        os.replace(tmp, CHECKPOINT)

    with mp.Pool(processes=workers) as pool:
        asyncs = [(t, pool.apply_async(run_task, (t,))) for t in pending]
        for t, a in asyncs:
            try:
                partial = a.get()
            except Exception as exc:  # keep the checkpoint; report the failure
                print(f"TASK FAILED {task_id(t)}: {exc!r}", flush=True)
                continue
            _merge_into(agg, partial)
            done_ids.add(task_id(t))
            finished += 1
            save()
            el = time.time() - t0
            print(f"[{finished}/{len(pending)}] {task_id(t)} "
                  f"elapsed={el/60:.1f}min", flush=True)
    print(f"fleet done in {(time.time()-t0)/60:.1f} min", flush=True)
    return agg


def predicted_power(cov: np.ndarray, g_vec: np.ndarray, alpha: float = ALPHA,
                    n_draws: int = N_DRAWS_PRED, seed: int = 0):
    """Gaussian-shift limit power P(||G+g||_inf > c) for G ~ N(0, cov)."""
    rng = np.random.default_rng(seed)
    cov = np.asarray(cov, dtype=float)
    cov = (cov + cov.T) / 2.0
    q = cov.shape[0]
    L = np.linalg.cholesky(cov + 1e-10 * np.eye(q))
    Z = rng.standard_normal((n_draws, q)) @ L.T
    stats0 = np.max(np.abs(Z), axis=1)
    crit = float(np.quantile(stats0, 1.0 - alpha))
    stats1 = np.max(np.abs(Z + g_vec[None, :]), axis=1)
    phat = float(np.mean(stats1 > crit))
    se = float(np.sqrt(phat * (1.0 - phat) / n_draws))
    return {"power": phat, "se": se, "crit": crit,
            "null_mean": float(stats0.mean())}


def aggregate(checkpoint: str = CHECKPOINT, output: str = SUMMARY,
              n_draws: int = N_DRAWS_PRED):
    with open(checkpoint) as fh:
        payload = json.load(fh)
    agg = payload["agg"]
    meta = payload.get("meta", {})
    scale = meta.get("scale")
    tseq = _tseq()
    # Merge checkpoint chunks: several keys can share (setting, direction, n).
    merged: dict = {}
    for key, acc in sorted(agg.items()):
        parts = key.split("|")
        setting, tag, n = parts[0], parts[1], int(parts[2])
        mkey = (setting, tag, n)
        dest = merged.setdefault(mkey, {
            "mult": [], "perm": [], "unadj": [],
            "cov_dr": np.zeros((2 * RESOLUTION, 2 * RESOLUTION)),
            "cov_unadj": np.zeros((2 * RESOLUTION, 2 * RESOLUTION)),
            "count_n": 0, "n": n,
        })
        dest["mult"].extend(acc["mult"])
        dest["perm"].extend(acc["perm"])
        dest["unadj"].extend(acc["unadj"])
        dest["cov_dr"] += np.asarray(acc["cov_dr"])
        dest["cov_unadj"] += np.asarray(acc["cov_unadj"])
        dest["count_n"] += acc["count_n"]
    cells = []
    for (setting, tag, n), acc in sorted(merged.items(),
                                        key=lambda kv: str(kv[0])):
        mult = np.asarray(acc["mult"])
        perm = np.asarray(acc["perm"])
        unadj = np.asarray(acc["unadj"])
        reps = len(mult)
        R = len(mult)
        tot_n = acc["count_n"]
        sigma_dr = np.asarray(acc["cov_dr"]) / tot_n
        sigma_unadj = np.asarray(acc["cov_unadj"]) / tot_n
        cell = {
            "setting": setting, "direction": tag, "n": n, "reps": R,
            "dr_multiplier_rate": float(np.mean(mult < ALPHA)),
            "dr_permutation_rate": float(np.mean(perm < ALPHA)),
            "unadjusted_rate": float(np.mean(unadj < ALPHA)),
            "mc_se_dr": float(np.sqrt(max(
                float(np.mean(mult < ALPHA)) *
                (1.0 - float(np.mean(mult < ALPHA))), 0.0) / max(R, 1))),
        }
        if tag != "None":
            g = direction_shape(tag, tseq) * scale
            g_vec = np.tile(g, 2)
            pred = predicted_power(sigma_dr, g_vec, n_draws=n_draws,
                                   seed=_seed("pred", setting, tag))
            cell["predicted_dr"] = pred["power"]
            cell["predicted_dr_se"] = pred["se"]
            cell["predicted_crit"] = pred["crit"]
            if setting != "PROG1":
                pred_u = predicted_power(sigma_unadj, g_vec, n_draws=n_draws,
                                         seed=_seed("predU", setting, tag))
                cell["predicted_unadj"] = pred_u["power"]
                cell["predicted_unadj_se"] = pred_u["se"]
            else:
                cell["predicted_unadj"] = None  # bias-divergent; no limit law
                cell["predicted_unadj_se"] = None
        cells.append(cell)
    verdict = judge(cells)
    summary = {"meta": {**meta, "alpha": ALPHA, "tol": TOL_CONFIRM,
                        "n_draws_pred": n_draws},
               "cells": cells, "verdict": verdict}
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print(json.dumps(verdict, indent=2))
    return summary


def judge(cells):
    """Pre-registered falsification rule (see module docstring)."""
    notes = []
    ok = True
    missing = []

    def get(**kw):
        hit = [c for c in cells
               if all(c[k] == v for k, v in kw.items())]
        if not hit:
            missing.append(str(kw))
            return None
        return hit[0]

    for setting in ("PROG0", "PROG1"):
        for d in DIRECTIONS:
            c500 = get(setting=setting, direction=d, n=500)
            if c500 is None:
                continue
            gap = abs(c500["dr_multiplier_rate"] - c500["predicted_dr"])
            status = "OK" if gap <= TOL_CONFIRM else "MISS"
            if gap > TOL_CONFIRM:
                ok = False
            notes.append(f"DR {setting}/{d}/n=500: emp={c500['dr_multiplier_rate']:.3f} "
                         f"pred={c500['predicted_dr']:.3f} gap={gap:.3f} {status}")
    for setting, nlist in (("PROG0", (500,)), ("NONPROG1", (200, 500)),
                            ("NONPROG0", (200, 500))):
        for d in DIRECTIONS:
            for n in nlist:
                c = get(setting=setting, direction=d, n=n)
                if c is None:
                    continue
                if c["predicted_unadj"] is None:
                    continue
                gap = abs(c["unadjusted_rate"] - c["predicted_unadj"])
                status = "OK" if gap <= TOL_CONFIRM else "MISS"
                if gap > TOL_CONFIRM:
                    ok = False
                notes.append(f"UNADJ {setting}/{d}/n={n}: emp={c['unadjusted_rate']:.3f} "
                             f"pred={c['predicted_unadj']:.3f} gap={gap:.3f} {status}")
    null1 = get(setting="PROG1", direction="None", n=200)
    if null1 is not None:
        notes.append(f"NULL PROG1/n=200: dr={null1['dr_multiplier_rate']:.3f} "
                     f"unadj={null1['unadjusted_rate']:.3f}")
        if not (null1["unadjusted_rate"] > 0.20):
            ok = False
            notes.append("MISS: unadjusted PROG1 null did not break size (>0.20 required)")
        if not (null1["dr_multiplier_rate"] <= 0.08):
            ok = False
            notes.append("MISS: DR PROG1 null exceeded 0.08")
    null0 = get(setting="PROG0", direction="None", n=200)
    if null0 is not None:
        notes.append(f"NULL PROG0/n=200: dr={null0['dr_multiplier_rate']:.3f} "
                     f"unadj={null0['unadjusted_rate']:.3f}")
        if not (null0["dr_multiplier_rate"] <= 0.08 and null0["unadjusted_rate"] <= 0.08):
            ok = False
            notes.append("MISS: a PROG0 null cell exceeded 0.08")
    nullN = get(setting="NONPROG1", direction="None", n=200)
    if nullN is not None:
        notes.append(f"NULL NONPROG1/n=200: dr={nullN['dr_multiplier_rate']:.3f} "
                     f"unadj={nullN['unadjusted_rate']:.3f}")
        if not (nullN["dr_multiplier_rate"] <= 0.08 and nullN["unadjusted_rate"] <= 0.08):
            ok = False
            notes.append("MISS: a NONPROG1 null cell exceeded 0.08")
    nullN0 = get(setting="NONPROG0", direction="None", n=200)
    if nullN0 is not None:
        notes.append(f"NULL NONPROG0/n=200: dr={nullN0['dr_multiplier_rate']:.3f} "
                     f"unadj={nullN0['unadjusted_rate']:.3f}")
        if not (nullN0["dr_multiplier_rate"] <= 0.08 and nullN0["unadjusted_rate"] <= 0.08):
            ok = False
            notes.append("MISS: a NONPROG0 null cell exceeded 0.08")
    for d in DIRECTIONS:  # ARE=1: matched limit laws, matched finite-n rates
        # n=200 is the pre-registered record; n=500 was added after the
        # partials showed a finite-n nuisance tax (WP7 §5), to test decay.
        # NONPROG0 (exact ARE=1 design) added on the same occasion.
        for n in (200, 500):
            for setting in ("NONPROG1", "NONPROG0"):
                c = get(setting=setting, direction=d, n=n)
                if c is None:
                    continue
                gap = abs(c["dr_multiplier_rate"] - c["unadjusted_rate"])
                status = "OK" if gap <= TOL_CONFIRM else "MISS"
                if gap > TOL_CONFIRM:
                    ok = False
                notes.append(f"ARE1 {setting}/{d}/n={n}: dr={c['dr_multiplier_rate']:.3f} "
                             f"unadj={c['unadjusted_rate']:.3f} gap={gap:.3f} {status}")
    if missing:
        ok = False
        notes.append(f"INCOMPLETE: {len(missing)} cells missing "
                     f"(e.g. {missing[0]}); verdict pending full fleet")
    return {"confirmed": bool(ok) and not missing, "notes": notes}


def make_figure(summary_path: str = SUMMARY, out: str = FIGURE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(summary_path) as fh:
        summary = json.load(fh)
    cells = summary["cells"]

    def get(**kw):
        hit = [c for c in cells
               if all(c[k] == v for k, v in kw.items())]
        return hit[0] if hit else None

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    for j, d in enumerate(DIRECTIONS):
        for i, setting in enumerate(("PROG0", "PROG1")):
            ax = axes[i][j]
            rows = [(n, get(setting=setting, direction=d, n=n))
                    for n in N_GRID]
            rows = [(n, c) for n, c in rows if c is not None]
            if not rows:
                ax.set_title(f"{setting} / {d} (pending)")
                continue
            ns = [n for n, _ in rows]
            emp = [c["dr_multiplier_rate"] for _, c in rows]
            emp_p = [c["dr_permutation_rate"] for _, c in rows]
            una = [c["unadjusted_rate"] for _, c in rows]
            pred = [c["predicted_dr"] for _, c in rows]
            ax.plot(ns, emp, "o-", label="DR multiplier (emp)")
            ax.plot(ns, emp_p, "s--", label="DR frozen-perm (emp)")
            ax.plot(ns, una, "x:", label="unadjusted (emp)")
            ax.plot(ns, pred, "k-", alpha=0.4, label="Gaussian-shift limit")
            for nn in (200, 500):
                for setting, mk, col in (("NONPROG1", "D", "C0"),
                                         ("NONPROG0", "o", "C2")):
                    c = get(setting=setting, direction=d, n=nn)
                    if c is None:
                        continue
                    ax.plot([nn], [c["dr_multiplier_rate"]], mk,
                            mfc="none", mec=col, mew=1.5,
                            label=f"DR {setting}" if nn == 200 else None)
                    ax.plot([nn], [c["unadjusted_rate"]], "x", mec="C3",
                            mew=1.5,
                            label=f"unadj {setting}" if nn == 200 else None)
            ax.set_xscale("log")
            ax.set_xticks(list(N_GRID), [str(n) for n in N_GRID])
            ax.set_ylim(-0.03, 1.03)
            ax.grid(alpha=0.3)
            ax.set_title(f"{setting} / {d}")
            if j == 0:
                ax.set_ylabel("rejection rate")
            if i == 1:
                ax.set_xlabel("n")
            if i == 0 and j == 2:
                ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Phase 7.5 local power: psi^(n) = n^-1/2 g  (alpha=0.05)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    try:
        from google.colab import files  # noqa
        files.download(out)
    except Exception:
        pass


def pilot(reps: int = 60, n: int = 500, n_calibration: int = N_CAL):
    """Null pilot at PROG0/n: estimate limit covariances, choose the scale.

    Scale rule (pre-registered): unit-sup-norm shapes times a common s with
    the mean-direction DR predicted power closest to 0.60 (grid 0.2..3.0).
    Uses null data only — no alternative information enters the choice.
    """
    tseq = _tseq()
    cov = np.zeros((2 * RESOLUTION, 2 * RESOLUTION))
    tot = 0
    for rep in range(reps):
        sample, _ = base_sample("PROG0", n, 90000 + rep)
        triplet = shifted_triplet(sample, None, n)
        r = _one_local_fit(triplet, 90000 + rep, n, "PROG0", "pilot",
                           n_calibration=n_calibration)
        cov += r["cov_dr"]
        tot += r["n"]
    sigma = cov / tot
    print(f"pilot: {reps} null reps at PROG0 n={n}", flush=True)
    best = None
    for s in np.arange(0.2, 3.01, 0.1):
        g_vec = np.tile(direction_shape("mean", tseq) * s, 2)
        pred = predicted_power(sigma, g_vec, seed=1)
        print(f"  s={s:.1f} pred_mean={pred['power']:.3f} crit={pred['crit']:.3f}",
              flush=True)
        if best is None or abs(pred["power"] - 0.60) < abs(best[1] - 0.60):
            best = (s, pred["power"])
    print(f"PILOT SCALE: s={best[0]:.2f} (pred_mean={best[1]:.3f})", flush=True)
    for d in DIRECTIONS:
        g_vec = np.tile(direction_shape(d, tseq) * best[0], 2)
        pred = predicted_power(sigma, g_vec, seed=2)
        print(f"  dir={d}: pred={pred['power']:.3f}", flush=True)
    return best[0]


def diag(out_path: str = "/tmp/phase7_diag.json", reps: int = 40,
         scale: float = 0.7, n_calibration: int = N_CAL,
         pred_summary: str = SUMMARY):
    """Locate statistic-side vs calibration-side deviations (follow-up only).

    For each (setting, direction, n in {(PROG1,freq,500),(PROG0,freq,500),
    (PROG1,bump,500)}) on FRESH seeds: per-rep multiplier crit, frozen-perm
    crit, observed statistic, versus the fleet Gaussian predicted crit.
    boot_crit ~= perm_crit < pred_crit  -> calibration-side (quantiles low);
    boot_crit ~= perm_crit ~= pred_crit -> statistic-side (law off).
    """
    from sklearn.ensemble import RandomForestClassifier

    with open(pred_summary) as fh:
        summary = json.load(fh)

    def pred_crit_for(setting, direction, n):
        hit = [c for c in summary["cells"] if c["setting"] == setting
               and c["direction"] == direction and c["n"] == n]
        return hit[0]["predicted_crit"] if hit else None

    tseq = _tseq()
    out = []
    for setting, direction, n in (("PROG1", "freq", 500),
                                  ("PROG0", "freq", 500),
                                  ("PROG1", "bump", 500)):
        g = direction_shape(direction, tseq) * scale
        pc = pred_crit_for(setting, direction, n)
        mc, pc_, st = [], [], []
        for k in range(reps):
            rep = 60000 + k
            sample, _ = base_sample(setting, n, rep)
            triplet = shifted_triplet(sample, g, n)
            fit = fit_dr(
                triplet.observed, triplet.tseq, n_basis=N_BASIS,
                n_folds=N_FOLDS,
                propensity_estimator=RandomForestClassifier(
                    n_estimators=100, min_samples_leaf=4, n_jobs=1,
                    random_state=_seed("rf", setting, rep, n)),
                random_state=_seed("fold", setting, rep, n, "diag"),
                propensity_feature_fn=None,
            )
            m = multiplier_test(fit, n_draws=n_calibration,
                                seed=_seed("mult", setting, "diag", rep, n))
            p = stratified_permutation_test(
                fit, propensity_strata(triplet.propensity, n_bins=NSTRATA),
                n_perm=n_calibration,
                seed=_seed("perm", setting, "diag", rep, n))
            mc.append(float(m["critical_value"]))
            pc_.append(float(p["critical_value"]))
            st.append(float(m["statistic"]))
        row = {"setting": setting, "direction": direction, "n": n,
               "reps": reps, "pred_crit": pc,
               "mult_crit_mean": float(np.mean(mc)),
               "perm_crit_mean": float(np.mean(pc_)),
               "stat_mean": float(np.mean(st)),
               "reject_mult": float(np.mean(np.array(st) > np.array(mc)))}
        print(json.dumps(row, indent=2), flush=True)
        out.append(row)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    return out


def smoke():
    """Bounded end-to-end check: 2 dirs x n=100 x 4 reps, tiny calibration."""
    t0 = time.time()
    out = run_fleet_mini()
    el = time.time() - t0
    print(f"SMOKE OK in {el:.1f}s: {json.dumps(out, indent=2)}")
    return out


def run_fleet_mini():
    tseq = _tseq()
    rows = []
    for setting in ("PROG0", "PROG1"):
        for d in ("mean", "bump"):
            for rep in range(4):
                sample, _ = base_sample(setting, 100, 81000 + rep)
                triplet = shifted_triplet(
                    sample, direction_shape(d, tseq) * 1.0, 100)
                r = _one_local_fit(triplet, 81000 + rep, 100, setting, d,
                                   n_calibration=39)
                rows.append((setting, d, round(r["multiplier_p"], 4),
                             round(r["unadjusted_p"], 4)))
    return rows


def main():
    global CHECKPOINT, SUMMARY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "pilot", "run", "aggregate",
                                           "figure", "diag"), required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reps", type=int, default=REPS_DEFAULT)
    parser.add_argument("--chunk", type=int, default=10)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--n-calibration", type=int, default=N_CAL)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--output", default=SUMMARY)
    args = parser.parse_args()
    if args.workers > 16:
        raise ValueError("workers capped at 16 per task spec")
    CHECKPOINT, SUMMARY = args.checkpoint, args.output
    if args.mode == "smoke":
        smoke()
    elif args.mode == "pilot":
        pilot()
    elif args.mode == "run":
        if args.scale is None:
            raise ValueError("--run requires --scale (from --mode pilot)")
        run_fleet(args.reps, args.workers, args.chunk, args.scale,
                  args.n_calibration, args.resume)
    elif args.mode == "aggregate":
        aggregate(args.checkpoint, args.output)
    elif args.mode == "figure":
        make_figure(args.output, os.path.join(
            os.path.dirname(os.path.abspath(args.output)),
            "phase7_local_power.png"))
    elif args.mode == "diag":
        diag(scale=(args.scale if args.scale is not None else 0.7),
             n_calibration=args.n_calibration)


if __name__ == "__main__":
    main()
