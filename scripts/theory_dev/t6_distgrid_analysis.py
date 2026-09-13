"""WP7 T6 (``distgrid``) analysis of the surrogate mechanism/scaling runs.

Reads the feature-level surrogate shards (``t6_surrogate_n*_shard*.json``),
the frozen fleet summary (``t6_existing_summary.json``) and writes
``t6_mechanism_analysis.json`` with

* per-arm rejection rates and MC standard errors;
* paired nuisance-arm differences (same DGP seed across arms);
* per-arm score diagnostics (mean absolute bias in statistic units, mean
  signed and absolute skewness, mean excess kurtosis);
* weighted fits of the est-arm excess rejection probability to
  ``a n^{-1/2}``, ``a n^{-1/2} + b n^{-1}`` and ``c n^{-1}``, with a
  nonparametric bootstrap over the per-cell Bernoulli draws;
* the restoring sample size ``n_restore = min{n : fitted excess <= 0.03}`` for
  each model and its bootstrap interval.

Run::

    uv run python scripts/theory_dev/t6_distgrid_analysis.py
"""

from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(ROOT, "results", "theory_validation")
ARMS = ("est_est", "oracle_oracle", "oracle_est_m", "est_oracle_m")
FLEET_N = (50, 100, 200, 500)
ALLOWANCE = 0.03          # upper band edge 0.08 minus nominal 0.05
N_BOOT = 4000
SEED = 260910


def _load_surrogate():
    rows = []
    pattern = re.compile(r"^t6_surrogate_n\d+_shard\d+\.json$")
    for path in sorted(glob.glob(os.path.join(RESULTS, "t6_surrogate_n*_shard*.json"))):
        if not pattern.match(os.path.basename(path)):
            continue
        with open(path) as fh:
            rows.extend(json.load(fh)["rows"])
    return rows


def oracle_score_moments(lam):
    """Exact support moments of the oracle confounded score at imbalance ``lam``.

    Enumerates the six support points of section 6.1 for the general regime
    ``e(0) = 0.5 - 0.25 lam``, ``e(1) = 0.5 + 0.25 lam``.
    """
    v_s = np.array([1.0, 8.0])
    v_m = np.array([0.5, 4.0])
    m = {(0, 0): v_s, (0, 1): v_m, (1, 0): v_m, (1, 1): v_s}
    probs, phis = [], []
    for x in (0, 1):
        e = 0.5 - 0.25 * lam if x == 0 else 0.5 + 0.25 * lam
        for a in (0, 1):
            p_xa = 0.5 * (e if a == 1 else 1.0 - e)
            vals = [v_s] if ((x == 0 and a == 0) or (x == 1 and a == 1)) \
                else [np.array([1.0, 0.0]), np.array([0.0, 8.0])]
            for v in vals:
                phi = (m[(x, 1)] - m[(x, 0)]
                       + (a / e) * (v - m[(x, 1)])
                       - ((1 - a) / (1 - e)) * (v - m[(x, 0)]))
                probs.append(p_xa / len(vals))
                phis.append(phi)
    probs = np.asarray(probs)
    phis = np.asarray(phis)
    mean = (probs[:, None] * phis).sum(axis=0)
    dev = phis - mean
    cov = (probs[:, None] * dev).T @ dev
    sd = np.sqrt(np.diag(cov))
    std = dev / sd
    return {
        "lambda": float(lam),
        "support": phis.tolist(),
        "probabilities": probs.tolist(),
        "mean": mean.tolist(),
        "var": np.diag(cov).tolist(),
        "corr": float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])) if cov[0, 0] * cov[1, 1] > 0 else None,
        "skew": (probs @ std ** 3).tolist(),
        "excess_kurtosis": (probs @ std ** 4 - 3.0).tolist(),
    }


def _load_lambda():
    out = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, "t6_surrogate_n200_r*_shard*.json"))):
        with open(path) as fh:
            data = json.load(fh)
        out.setdefault(float(data["regime"]), []).extend(data["rows"])
    return out


def _load_fleet():
    path = os.path.join(RESULTS, "t6_existing_summary.json")
    with open(path) as fh:
        return json.load(fh)


def _cell_stats(rows, arm):
    sel = [r for r in rows if arm in r]
    if not sel:
        return None
    p = np.array([r[arm]["p"] for r in sel])
    diag = [d for r in sel for d in r[arm]["diagnostics"]]
    return {
        "reps": int(len(sel)),
        "rate": float((p < 0.05).mean()),
        "mc_se": float(np.sqrt(max((p < 0.05).mean() * (1 - (p < 0.05).mean()), 1e-12) / len(p))),
        "crit_mean": float(np.mean([r[arm]["crit"] for r in sel])),
        "stat_q95": float(np.quantile([r[arm]["stat"] for r in sel], 0.95)),
        "stat_mean": float(np.mean([r[arm]["stat"] for r in sel])),
        "mean_abs_bias_stat_units": float(np.mean([abs(d["bias_in_stat_units"]) for d in diag])),
        "mean_skew": float(np.mean([d["skew"] for d in diag])),
        "mean_abs_skew": float(np.mean([abs(d["skew"]) for d in diag])),
        "mean_excess_kurtosis": float(np.mean([d["excess_kurtosis"] for d in diag])),
    }


def _paired(rows, arm_a, arm_b):
    pairs = [(r[arm_a]["p"], r[arm_b]["p"]) for r in rows
             if arm_a in r and arm_b in r]
    if not pairs:
        return None
    a = np.array([1.0 * (x < 0.05) for x, _ in pairs])
    b = np.array([1.0 * (y < 0.05) for _, y in pairs])
    d = a - b
    return {
        "reps": int(len(pairs)),
        "diff": float(d.mean()),
        "mc_se": float(d.std(ddof=1) / np.sqrt(len(d))),
        "n_a": int(a.sum()),
        "n_b": int(b.sum()),
    }


def _fit_models(ns, rates, ses, rng, n_boot=N_BOOT):
    """Bootstrap the three decay models for the est-arm excess."""
    ns = np.asarray(ns, dtype=float)
    rates = np.asarray(rates, dtype=float)
    ses = np.asarray(ses, dtype=float)
    excess = rates - 0.05

    def design(n, model):
        if model == "inv_sqrt":
            return np.column_stack([1.0 / np.sqrt(n)])
        if model == "inv_sqrt_plus_inv":
            return np.column_stack([1.0 / np.sqrt(n), 1.0 / n])
        if model == "inv":
            return np.column_stack([1.0 / n])
        raise ValueError(model)

    def restore(coef, model, allowance=ALLOWANCE, n_max=10 ** 7):
        n_grid = np.exp(np.linspace(np.log(2.0), np.log(float(n_max)), 4000))
        pred = design(n_grid, model) @ coef
        ok = pred <= allowance
        return float(n_grid[ok][0]) if ok.any() else float("inf")

    out = {}
    for model in ("inv_sqrt", "inv_sqrt_plus_inv", "inv"):
        X = design(ns, model)
        w = 1.0 / np.sqrt(ses ** 2 + 1e-8)
        Xw = X * w[:, None]
        yw = excess * w
        coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        pred = X @ coef
        ss_res = float(np.sum((pred - excess) ** 2))
        out[model] = {
            "coef": coef.tolist(),
            "restore_n": restore(coef, model),
            "ss_res": ss_res,
        }
        boot = []
        for _ in range(n_boot):
            resampled = excess + rng.normal(0.0, ses)
            cb, *_ = np.linalg.lstsq(Xw, resampled * w, rcond=None)
            boot.append(restore(cb, model))
        boot = np.asarray(boot)
        out[model]["restore_n_q05"] = float(np.quantile(boot, 0.05))
        out[model]["restore_n_q50"] = float(np.quantile(boot, 0.50))
        out[model]["restore_n_q95"] = float(np.quantile(boot, 0.95))
    return out


def main():
    rows = _load_surrogate()
    fleet = _load_fleet()
    out = {"surrogate_files": sorted(glob.glob(
        os.path.join(RESULTS, "t6_surrogate_n*_shard*.json"))),
        "n_surrogate_rows": len(rows), "cells": {}, "paired": {},
        "fleet_est": {}, "fits": {}}

    ns = sorted({r["n"] for r in rows})
    for n in ns:
        sub = [r for r in rows if r["n"] == n]
        out["cells"][f"n={n}"] = {arm: _cell_stats(sub, arm) for arm in ARMS}

    for n in ns:
        sub = [r for r in rows if r["n"] == n]
        out["paired"][f"n={n}"] = {
            "est_minus_oracle": _paired(sub, "est_est", "oracle_oracle"),
            "propensity_estimated_channel": _paired(sub, "est_est",
                                                    "oracle_est_m"),
            "outcome_model_estimated_channel": _paired(sub, "est_est",
                                                       "est_oracle_m"),
        }

    for n in FLEET_N:
        cell = fleet["cells"].get(f"confounded|n={n}|r=1.0")
        if cell:
            out["fleet_est"][f"n={n}"] = cell["rate"]

    lam = _load_lambda()
    out["lambda_sweep"] = {}
    for regime, sub in sorted(lam.items()):
        out["lambda_sweep"][f"lambda={regime}"] = {
            "reps": len(sub),
            "arms": {arm: _cell_stats(sub, arm) for arm in ARMS
                     if _cell_stats(sub, arm) is not None},
            "paired_est_minus_oracle": _paired(sub, "est_est",
                                               "oracle_oracle"),
        }
    # fleet est-arm rates at n=200 for the same regime (external anchor)
    out["fleet_est_vs_lambda"] = {
        f"lambda={regime}": fleet["cells"].get(
            f"confounded|n=200|r={regime}", {}).get("rate")
        for regime in (0.0, 0.5, 1.0)}

    # The canonical est-arm scaling uses the frozen 500-replication fleet; the
    # surrogate est arm reproduces those seeds exactly, so mixing the two would
    # double count.  Surrogate cells are used only where the fleet has no cell.
    fit_ns, fit_rates, fit_ses = [], [], []
    covered = set()
    for n in FLEET_N:
        cell = fleet["cells"].get(f"confounded|n={n}|r=1.0")
        if cell:
            fit_ns.append(n)
            fit_rates.append(cell["rate"]["rate"])
            fit_ses.append(cell["rate"]["mc_se"])
            covered.add(n)
    for n in ns:
        cell = out["cells"].get(f"n={n}", {}).get("est_est")
        if cell and cell["reps"] >= 100 and n not in covered:
            fit_ns.append(n)
            fit_rates.append(cell["rate"])
            fit_ses.append(cell["mc_se"])
    out["fits"]["source"] = "fleet_500_reps_plus_surrogate_extras" \
        if covered == set(FLEET_N) else "mixed"
    out["fits"]["points"] = [{"n": int(n), "rate": float(r), "mc_se": float(s)}
                             for n, r, s in zip(fit_ns, fit_rates, fit_ses)]
    rng = np.random.default_rng(SEED)
    out["fits"]["models"] = _fit_models(fit_ns, fit_rates, fit_ses, rng)

    out["oracle_score_moments"] = {f"lambda={lam}": oracle_score_moments(lam)
                                   for lam in (0.0, 0.25, 0.5, 0.75, 1.0)}
    with open(os.path.join(RESULTS, "t6_oracle_score_moments.json"), "w") as fh:
        json.dump(out["oracle_score_moments"], fh, indent=1)

    path = os.path.join(RESULTS, "t6_mechanism_analysis.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
