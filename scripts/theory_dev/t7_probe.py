"""T7 mechanism probes for the Phase-7 local-power mismatch.

The Phase-7 fleet (``experiments/phase7_local_power.py``) compared the DR
max-statistic rejection rate with a Gaussian-shift prediction built from the
pooled covariance of the *local* centred scores.  One cell missed at n=500
(PROG1/freq, 0.730 vs 0.628), with the same sign of gap in other freq cells.

This script isolates three candidate mechanisms without curve fitting:

(a) finite-basis truncation of the nuisance outcome regression
    (regression ``n_basis`` K vs the shift's Fourier content);
(b) IPW tax under imbalance / propensity estimation error
    (``prop_scale`` sweep; true vs estimated propensity);
(c) max-over-grid and frozen-nuisance/multiplier effects
    (per-rep versus pooled Gaussian calibration; multiplier draw count;
    small-grid synthetic checks with an exactly known score law).

Nothing is reimplemented: estimation goes through ``fit_dr``.

Outputs: JSON under ``results/theory_validation/t7_*.json``.  Each run records
the per-replication scalar diagnostics (statistic, per-rep multiplier and
Gaussian critical values, ESS, clipping fraction, score skewness) plus the
pooled covariance per score variant.  Aggregation never fits a curve; the
Gaussian-shift power is always computed from the recorded covariance by direct
simulation.

Pre-registered reading rules (fixed before production runs):
  R1. Mechanism (a) is CONFIRMED for a cell if raising the regression basis
      from K=5 to K>=7 (which contains cos(6 pi t)) moves
      |empirical - pooled Gaussian| below the 0.06 fleet tolerance and the
      gap was above it at K=5, with the same seeds.
  R2. Mechanism (b) is CONFIRMED if using the true propensity in the score
      (same fitted outcome regression) moves the gap below 0.06, or if the
      gap scales monotonically with prop_scale and tracks
      Var(e(X))/ESS diagnostics.
  R3. Mechanism (c) is CONFIRMED if the per-rep Gaussian calibration
      reproduces the empirical multiplier rate within tolerance while the
      pooled Gaussian prediction does not: the gap is an aggregation
      (convolution) effect, not a score-law effect.
  R4. A gap that survives oracle nuisances (true propensity AND true outcome
      means) at n=500 is classified UNEXPLAINED by (a)-(c) and is reported
      as such.
  R5. Differences below the MC tolerance (0.06 at 60--300 reps; MC SE is
      reported per cell) are NOT interpreted.

Usage:
    python scripts/theory_dev/t7_probe.py --mode core --workers 3 --reps 60
    python scripts/theory_dev/t7_probe.py --mode basis --workers 3 --reps 60
    python scripts/theory_dev/t7_probe.py --mode strength --workers 3 --reps 50
    python scripts/theory_dev/t7_probe.py --mode synthetic
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))
if os.path.join(ROOT, "experiments") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "experiments"))

from phase7_local_power import (  # noqa: E402
    INTERVAL,
    NSTRATA,
    RESOLUTION,
    _seed,
    _tseq,
    direction_shape,
    shifted_triplet,
)
from tda2s.tests.dr_outcome import fit_dr, multiplier_test  # noqa: E402

from sklearn.ensemble import RandomForestClassifier  # noqa: E402

Q = 2 * RESOLUTION
ALPHA = 0.05
REGIME_OF = {"PROG0": 0.0, "PROG1": 1.0, "NONPROG1": 1.0, "NONPROG0": 0.0}


# --------------------------------------------------------------------------- IO


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


def save_json(path: str, payload: dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(_jsonable(payload), fh, indent=2, sort_keys=True)
    print(f"wrote {path}", flush=True)


# --------------------------------------------------------------- data plumbing


def draw_base(setting: str, n: int, rep: int, prop_scale: float | None = None):
    """Sample generator mirroring ``phase7_local_power.base_sample``.

    The only difference is that ``prop_scale`` can be overridden (mechanism b)
    while all random streams stay keyed by the phase-7 seeds, so a fixed rep
    gives the same model and sample draws for every override.
    """
    from tcda_uq.datasets import TriOracleSimulation

    regime = REGIME_OF[setting]
    ps = 1.5 * regime if prop_scale is None else float(prop_scale)
    sim = TriOracleSimulation(
        n_cov=3, n_hom_dim=2, resolution=RESOLUTION, interval=INTERVAL,
        n_basis=5, coef_scale=0.6, noise_scale=0.15,
        prop_scale=ps, seed=_seed("model", rep),
    )
    for d in range(sim.n_hom_dim):
        sim.Gamma[1][d] = sim.Gamma[0][d].copy()
    sample = sim.sample(n, rng=_seed("sample", setting, rep, n))
    if setting in ("NONPROG1", "NONPROG0"):
        pot = np.asarray(sample.potential_outcomes, dtype=float)
        mbar = pot.mean(axis=(0, 1))
        rng_n = np.random.default_rng(_seed("nonprog-noise", rep, n))
        pot_new = np.empty_like(pot)
        pot_new[:, 0] = mbar[None, :, :] + sim._noise(n, rng_n, None)
        pot_new[:, 1] = mbar[None, :, :] + sim._noise(n, rng_n, None)
        sample = SimpleNamespace(
            tseq=sample.tseq, X=sample.X, A=sample.A,
            propensity=sample.propensity, potential_outcomes=pot_new,
            observed=None, mbar=mbar,
        )
    return sim, sample


def shifted(sample, g: np.ndarray | None, n: int):
    """``shifted_triplet`` re-implementation that keeps the potential outcomes."""
    pot = np.asarray(sample.potential_outcomes, dtype=float).copy()
    if g is not None:
        pot[:, 1, :, :] = pot[:, 1, :, :] + np.asarray(g)[None, None, :] / np.sqrt(n)
    A = np.asarray(sample.A)
    phi = pot[np.arange(len(A)), A]
    return pot, (phi, A, np.asarray(sample.X))


# ------------------------------------------------------------------ statistics


def score_stats(score_tensor: np.ndarray, g_vec: np.ndarray | None):
    """Observed statistic, per-unit covariance, centring diagnostics."""
    n = score_tensor.shape[0]
    est = np.asarray(score_tensor, dtype=float).mean(axis=0)
    stat = float(np.sqrt(n) * np.max(np.abs(est)))
    centered = (score_tensor - est[None, :, :]).reshape(n, -1)
    cov = centered.T @ centered / n
    jstar = int(np.argmax(np.abs(est).reshape(-1)))
    col = centered[:, jstar]
    out = {
        "stat": stat,
        "cov": cov,
        "centered": centered,
        "skew_max_coord": float(_skew(col)),
        "kurt_max_coord": float(_kurt(col)),
    }
    if g_vec is not None:
        proj = centered @ (g_vec / np.linalg.norm(g_vec))
        out["skew_g_proj"] = float(_skew(proj))
        out["kurt_g_proj"] = float(_kurt(proj))
    return out


def _skew(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    m = x.mean()
    s = x.std()
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def _kurt(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    m = x.mean()
    s = x.std()
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4))


def gauss_crit_power(cov: np.ndarray, g_vec: np.ndarray, *, n_draws: int = 20000,
                     seed: int = 0, alpha: float = ALPHA):
    """Gaussian-shift crit/power from a supplied covariance (no fitting)."""
    rng = np.random.default_rng(seed)
    cov = np.asarray(cov, dtype=float)
    cov = (cov + cov.T) / 2.0
    q = cov.shape[0]
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(q))
    Z = rng.standard_normal((n_draws, q)) @ L.T
    null = np.max(np.abs(Z), axis=1)
    crit = float(np.quantile(null, 1.0 - alpha))
    alt = np.max(np.abs(Z + g_vec[None, :]), axis=1)
    pw = float(np.mean(alt > crit))
    return {"crit": crit, "power": pw,
            "power_se": float(np.sqrt(max(pw * (1 - pw), 0) / n_draws))}


def boot_crit(centered: np.ndarray, *, n_boot: int = 2000, seed: int = 0,
              alpha: float = ALPHA) -> float:
    """Unit-resampling (non-parametric) critical value for the score mean.

    Draws ``counts ~ Multinomial(n, 1/n)`` so that ``counts @ centered`` is
    the bootstrap sum with integer multiplicities; no n_boot x n x q tensor
    is materialised.
    """
    n = centered.shape[0]
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot)
    draws = counts @ centered / np.sqrt(n)
    null = np.max(np.abs(draws), axis=1)
    return float(np.quantile(null, 1.0 - alpha))


def true_means(sample, sim, X: np.ndarray, g: np.ndarray | None, n: int,
               setting: str):
    """True conditional means (mu0, mu1) including the local shift in arm 1."""
    if setting in ("NONPROG1", "NONPROG0"):
        mbar = np.asarray(sample.mbar, dtype=float)
        mu0 = np.broadcast_to(mbar[None, :, :],
                              (X.shape[0],) + mbar.shape).copy()
        mu1 = mu0.copy()
        if g is not None:
            mu1 = mu1 + np.asarray(g)[None, None, :] / np.sqrt(n)
        return mu0, mu1
    mu0 = sim._mean(0, X)
    mu1 = sim._mean(1, X)
    if g is not None:
        mu1 = mu1 + np.asarray(g)[None, None, :] / np.sqrt(n)
    return mu0, mu1


def oracle_scores(phi, A, pi, mu0, mu1):
    """AIPW score tensor from supplied outcomes and nuisances."""
    inv = (A / pi - (1.0 - A) / (1.0 - pi))[:, None, None]
    mu_obs = A[:, None, None] * mu1 + (1.0 - A)[:, None, None] * mu0
    return (mu1 - mu0) + inv * (phi - mu_obs)


# ---------------------------------------------------------------- one-cell run


def _job_core(job: dict) -> dict:
    """One (setting, n, direction, K, prop_scale) cell over a rep chunk."""
    setting = job["setting"]
    n = job["n"]
    direction = job["direction"]
    lo, hi = job["lo"], job["hi"]
    K = job["K"]
    prop_scale = job.get("prop_scale")
    B = job["B"]
    B_hi = job.get("B_hi")
    n_draws = job["n_draws"]
    n_boot = job["n_boot"]
    tseq = _tseq()
    scale = job["scale"]
    g = None if direction is None else direction_shape(direction, tseq) * scale
    g_vec = None if g is None else np.tile(g, 2)
    cell_key = f"{setting}|{n}|{direction}"
    cell = {
        "meta": {"setting": setting, "n": n, "direction": direction,
                 "K": K, "prop_scale": prop_scale, "scale": scale},
        "variants": {},
    }

    def acc(variant):
        if variant not in cell["variants"]:
            cell["variants"][variant] = {
                "variant": variant, "reps": 0,
                "stat": [], "skew_max_coord": [], "kurt_max_coord": [],
                "skew_g_proj": [], "kurt_g_proj": [], "cov_sum": np.zeros((Q, Q)),
                "ess_treated": [], "ess_control": [], "frac_clip": [],
                "min_pi": [], "max_pi": [], "var_e": [],
                "max_w_treated": [], "max_w_control": [],
                "mean_w_treated": [], "mean_e_over_ehat": [],
                "mean_one_minus_e_over_one_minus_ehat": [],
                "proj_g_scale": [],
            }
        return cell["variants"][variant]

    for rep in range(lo, hi):
        sim, sample = draw_base(setting, n, rep, prop_scale=prop_scale)
        pot, observed = shifted(sample, g, n)
        phi, A, X = observed
        pi_true = np.asarray(sample.propensity, dtype=float)
        tag = str(direction)
        fit = fit_dr(
            observed, tseq, n_basis=K, n_folds=2,
            propensity_estimator=RandomForestClassifier(
                n_estimators=100, min_samples_leaf=4, n_jobs=1,
                random_state=_seed("rf", setting, rep, n)),
            random_state=_seed("fold", setting, rep, n, tag),
        )
        pi_hat = np.asarray(fit.pi_hat, dtype=float)
        A_ord = np.asarray(fit.labels_order, dtype=float)
        X_ord = np.asarray(fit.X)[fit.order]
        phi_ord = np.asarray(fit.phi_order, dtype=float)

        mu0_true, mu1_true = true_means(sample, sim, X_ord, g, n, setting)
        variants_scores = {
            "fit": np.stack(fit.result.scores, axis=1),
            "true_pi": oracle_scores(phi_ord, A_ord, pi_true[fit.order],
                                     fit.mu0, fit.mu1),
            "true_mu": oracle_scores(phi_ord, A_ord, pi_hat, mu0_true, mu1_true),
            "oracle": oracle_scores(phi_ord, A_ord, pi_true[fit.order],
                                    mu0_true, mu1_true),
        }
        ratio_e = float(np.mean(pi_true[fit.order] / pi_hat))
        ratio_1me = float(np.mean((1.0 - pi_true[fit.order]) / (1.0 - pi_hat)))
        for name, tensor in variants_scores.items():
            st = score_stats(tensor, g_vec)
            a = acc(name)
            a["reps"] += 1
            a["stat"].append(st["stat"])
            a["skew_max_coord"].append(st["skew_max_coord"])
            a["kurt_max_coord"].append(st["kurt_max_coord"])
            a["skew_g_proj"].append(st.get("skew_g_proj", 0.0))
            a["kurt_g_proj"].append(st.get("kurt_g_proj", 0.0))
            a["cov_sum"] += st["cov"]
            a["mean_w_treated"].append(float(np.mean(A_ord / pi_hat)))
            a["mean_e_over_ehat"].append(ratio_e)
            a["mean_one_minus_e_over_one_minus_ehat"].append(ratio_1me)
            if g_vec is not None:
                proj = float(np.sqrt(n) * (tensor.mean(axis=0).reshape(-1)
                                           @ g_vec)
                             / np.linalg.norm(g_vec))
                a["proj_g_scale"].append(proj)

        w_t = A_ord / pi_hat
        w_c = (1.0 - A_ord) / (1.0 - pi_hat)
        ess_t = _ess(w_t)
        ess_c = _ess(w_c)
        for name in variants_scores:
            a = acc(name)
            a["ess_treated"].append(ess_t)
            a["ess_control"].append(ess_c)
            a["max_w_treated"].append(float(w_t.max()))
            a["max_w_control"].append(float(w_c.max()))
            a["frac_clip"].append(float(np.mean(
                (pi_hat <= 0.0100001) | (pi_hat >= 0.9899999))))
            a["min_pi"].append(float(pi_hat.min()))
            a["max_pi"].append(float(pi_hat.max()))
            a["var_e"].append(float(np.var(pi_true)))

        # Multiplier calibration of the deployed (fitted) score.
        if B:
            m399 = multiplier_test(
                fit, n_draws=B, seed=_seed("mult", setting, tag, rep, n))
            acc("fit")["crit_mult"] = acc("fit").get("crit_mult", [])
            acc("fit")["crit_mult"].append(float(m399["critical_value"]))
            acc("fit")["p_mult"] = acc("fit").get("p_mult", [])
            acc("fit")["p_mult"].append(float(m399["pvalue"]))
        if B_hi:
            mhi = multiplier_test(
                fit, n_draws=B_hi, seed=_seed("multH", setting, tag, rep, n))
            acc("fit")["crit_mult_hi"] = acc("fit").get("crit_mult_hi", [])
            acc("fit")["crit_mult_hi"].append(float(mhi["critical_value"]))

        # Per-rep Gaussian and unit-bootstrap calibrations of `fit` and `oracle`.
        if B:
            st_fit = score_stats(variants_scores["fit"], g_vec)
            st_or = score_stats(variants_scores["oracle"], g_vec)
            gp_fit = gauss_crit_power(st_fit["cov"], g_vec, n_draws=n_draws,
                                      seed=_seed("g", setting, tag, rep, n))
            gp_or = gauss_crit_power(st_or["cov"], g_vec, n_draws=n_draws,
                                     seed=_seed("gO", setting, tag, rep, n))
            bc_fit = boot_crit(st_fit["centered"], n_boot=n_boot,
                               seed=_seed("np", setting, tag, rep, n))
            a = acc("fit")
            a.setdefault("crit_gauss", []).append(gp_fit["crit"])
            a.setdefault("p_gauss", []).append(gp_fit["power"])
            a.setdefault("crit_np", []).append(bc_fit)
            if g is not None:
                Pg = regression_projection(g, K)
                g_eff = g + (ratio_e - 1.0) * (g - Pg)
                gp_corr = gauss_crit_power(
                    st_fit["cov"], np.tile(g_eff, 2), n_draws=n_draws,
                    seed=_seed("gc", setting, tag, rep, n))
                a.setdefault("p_gauss_corr", []).append(gp_corr["power"])
            a = acc("oracle")
            a.setdefault("crit_gauss", []).append(gp_or["crit"])
            a.setdefault("p_gauss", []).append(gp_or["power"])

    return {"cells": {cell_key: cell}}


def regression_projection(g: np.ndarray, K: int) -> np.ndarray:
    """L2 projection of a curve onto the span of the K-function Fourier basis."""
    from skfda.representation.basis import FourierBasis

    tseq = _tseq()
    B = np.asarray(FourierBasis((float(tseq[0]), float(tseq[-1])), K)(tseq))
    if B.ndim == 3:
        B = B[..., 0]
    Q, _ = np.linalg.qr(B.T)
    return Q @ (Q.T @ np.asarray(g, dtype=float))


def _ess(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    w = w[w > 0]
    if w.size == 0:
        return 0.0
    return float(w.sum() ** 2 / np.sum(w ** 2))


def merge_chunks(chunks: list[dict]) -> dict:
    merged: dict = {}
    for ch in chunks:
        for key, cell in ch["cells"].items():
            dest = merged.setdefault(key, {"meta": cell["meta"], "variants": {}})
            for name, a in cell["variants"].items():
                dst = dest["variants"].setdefault(name, {})
                for k, v in a.items():
                    if k == "cov_sum":
                        dst[k] = dst.get(k, np.zeros((Q, Q))) + np.asarray(v)
                    elif k == "reps":
                        dst[k] = dst.get(k, 0) + v
                    elif isinstance(v, list):
                        dst.setdefault(k, []).extend(v)
                    else:
                        dst[k] = v
    return {"cells": merged}


def summarize(merged: dict, alpha: float = ALPHA) -> dict:
    """Per-cell empirical rates, pooled Gaussian predictions, decompositions."""
    out = {}
    for key, cell in merged["cells"].items():
        meta = cell["meta"]
        g = None
        if meta.get("direction") is not None:
            g = direction_shape(meta["direction"], _tseq()) * meta["scale"]
        g_vec = None if g is None else np.tile(g, 2)
        out[key] = {}
        for name, a in cell["variants"].items():
            R = len(a["stat"])
            cov_pool = np.asarray(a["cov_sum"]) / max(R, 1)
            row = {
                "variant": name, "reps": R,
                "stat_mean": float(np.mean(a["stat"])),
                "stat_sd": float(np.std(a["stat"], ddof=1)),
                "skew_max_coord_mean": float(np.mean(a["skew_max_coord"])),
                "kurt_max_coord_mean": float(np.mean(a["kurt_max_coord"])),
                "ess_treated_mean": float(np.mean(a["ess_treated"])),
                "ess_control_mean": float(np.mean(a["ess_control"])),
                "frac_clip_mean": float(np.mean(a["frac_clip"])),
                "var_e_mean": float(np.mean(a["var_e"])),
                "max_w_treated_mean": float(np.mean(a["max_w_treated"])),
                "max_w_control_mean": float(np.mean(a["max_w_control"])),
                "mean_w_treated_mean": float(np.mean(a["mean_w_treated"])),
                "mean_e_over_ehat_mean": float(np.mean(a["mean_e_over_ehat"])),
                "mean_1me_over_1mehat_mean": float(
                    np.mean(a["mean_one_minus_e_over_one_minus_ehat"])),
            }
            if a.get("proj_g_scale"):
                row["proj_g_scale_mean"] = float(np.mean(a["proj_g_scale"]))
                row["proj_g_scale_sd"] = float(np.std(a["proj_g_scale"], ddof=1))
            if g_vec is not None:
                pp = gauss_crit_power(
                    cov_pool, g_vec,
                    seed=_seed("pooled", meta.get("setting", ""),
                               name, meta.get("n", 0)))
                row["pooled_gauss_crit"] = pp["crit"]
                row["pooled_gauss_power"] = pp["power"]
                row["pooled_gauss_power_se"] = pp["power_se"]
                # Theory-based local-bias correction for the deployed fit:
                #   g_eff = g + delta * (g - P_K g),  delta = E[e/ ehat] - 1.
                K = int(meta.get("K", 5))
                Pg = regression_projection(g, K)
                delta = row["mean_e_over_ehat_mean"] - 1.0
                g_eff = g + delta * (g - Pg)
                row["delta_mean_ratio"] = delta
                row["shift_out_of_span_frac"] = float(
                    np.linalg.norm(g - Pg) / max(np.linalg.norm(g), 1e-12))
                if name == "fit":
                    cp = gauss_crit_power(
                        cov_pool, np.tile(g_eff, 2),
                        seed=_seed("pooledCorr", meta.get("setting", ""),
                                   name, meta.get("n", 0)))
                    row["corrected_gauss_crit"] = cp["crit"]
                    row["corrected_gauss_power"] = cp["power"]
                    row["corrected_gauss_power_se"] = cp["power_se"]
                    row["perrep_corrected_gauss_power_mean"] = float(
                        np.mean(a["p_gauss_corr"]))
                stat = np.asarray(a["stat"])
                row["stat_gt_pooled_gauss"] = float(np.mean(stat > pp["crit"]))
                if "crit_gauss" in a:
                    cg = np.asarray(a["crit_gauss"])
                    row["perrep_gauss_power_mean"] = float(np.mean(a["p_gauss"]))
                    row["stat_gt_perrep_gauss"] = float(np.mean(stat > cg))
                    row["crit_gauss_mean"] = float(cg.mean())
                if "crit_np" in a:
                    cn = np.asarray(a["crit_np"])
                    row["stat_gt_np"] = float(np.mean(stat > cn))
                    row["crit_np_mean"] = float(cn.mean())
                if "crit_mult" in a:
                    cm = np.asarray(a["crit_mult"])
                    row["mult_rate"] = float(np.mean(
                        np.asarray(a["p_mult"]) < alpha))
                    row["mc_se_mult"] = float(np.sqrt(max(
                        row["mult_rate"] * (1 - row["mult_rate"]), 0) / R))
                    row["crit_mult_mean"] = float(cm.mean())
                    row["stat_gt_mult_crit"] = float(np.mean(stat > cm))
                if "crit_mult_hi" in a:
                    chi = np.asarray(a["crit_mult_hi"])
                    row["crit_mult_hi_mean"] = float(chi.mean())
                    row["stat_gt_mult_hi"] = float(np.mean(stat > chi))
            out[key][name] = row
    return out


# --------------------------------------------------------------------- drivers


def run_cells(jobs: list[dict], workers: int) -> dict:
    chunks = []
    if workers <= 1:
        for i, j in enumerate(jobs):
            chunks.append(_job_core(j))
            print(f"  chunk {i + 1}/{len(jobs)} done", flush=True)
    else:
        import multiprocessing as mp
        with mp.Pool(processes=workers) as pool:
            for i, ch in enumerate(pool.imap_unordered(_job_core, jobs)):
                chunks.append(ch)
                print(f"  chunk {i + 1}/{len(jobs)} done", flush=True)
    merged = merge_chunks(chunks)
    return {"cells": merged["cells"], "summary": summarize(merged)}


def build_jobs(cells, reps: int, chunk: int, *, K=5, prop_scale=None,
               scale=0.7, B=399, B_hi=None, n_draws=20000, n_boot=2000):
    jobs = []
    for setting, n, direction in cells:
        for lo in range(0, reps, chunk):
            jobs.append({
                "setting": setting, "n": n, "direction": direction,
                "lo": lo, "hi": min(lo + chunk, reps), "K": K,
                "prop_scale": prop_scale, "scale": scale, "B": B,
                "B_hi": B_hi, "n_draws": n_draws, "n_boot": n_boot,
            })
    return jobs


def mode_core(args):
    cells = [tuple(c.split(":")) for c in args.cells]
    cells = [(s, int(n), (d if d != "None" else None)) for s, n, d in cells]
    jobs = build_jobs(cells, args.reps, args.chunk, K=args.K,
                      prop_scale=args.prop_scale, scale=args.scale,
                      B=args.B, B_hi=args.B_hi, n_draws=args.n_draws,
                      n_boot=args.n_boot)
    print(f"core: {len(jobs)} chunk-jobs, {args.workers} workers", flush=True)
    t0 = time.time()
    out = run_cells(jobs, args.workers)
    out["elapsed_s"] = time.time() - t0
    save_json(args.out, out)
    for key, variants in out["summary"].items():
        for name, row in variants.items():
            print(key, name, json.dumps(row, sort_keys=True), flush=True)
    return out


def mode_basis(args):
    """Mechanism (a): vary the outcome-regression Fourier basis K."""
    directions = args.directions.split(",")
    settings = args.settings.split(",")
    ns = [int(x) for x in args.ns.split(",")]
    all_json = {}
    for K in [int(x) for x in args.K_list.split(",")]:
        cells = [(s, n, d) for s in settings for n in ns for d in directions]
        jobs = build_jobs(cells, args.reps, args.chunk, K=K,
                          scale=args.scale, B=args.B, B_hi=args.B_hi,
                          n_draws=args.n_draws, n_boot=args.n_boot)
        print(f"basis K={K}: {len(jobs)} chunk-jobs", flush=True)
        t0 = time.time()
        out = run_cells(jobs, args.workers)
        out["elapsed_s"] = time.time() - t0
        out["K"] = K
        all_json[str(K)] = out
        for key, variants in out["summary"].items():
            for name, row in variants.items():
                print(f"K={K} {key} {name}",
                      json.dumps(row, sort_keys=True), flush=True)
    save_json(args.out, all_json)
    return all_json


def mode_strength(args):
    """Mechanism (b): vary propensity strength with all else fixed."""
    directions = args.directions.split(",")
    settings = args.settings.split(",")
    ns = [int(x) for x in args.ns.split(",")]
    all_json = {}
    for ps in [float(x) for x in args.prop_scale_list.split(",")]:
        cells = [(s, n, d) for s in settings for n in ns for d in directions]
        jobs = build_jobs(cells, args.reps, args.chunk, K=args.K,
                          prop_scale=ps, scale=args.scale, B=args.B,
                          B_hi=args.B_hi, n_draws=args.n_draws,
                          n_boot=args.n_boot)
        print(f"strength prop_scale={ps}: {len(jobs)} chunk-jobs", flush=True)
        t0 = time.time()
        out = run_cells(jobs, args.workers)
        out["elapsed_s"] = time.time() - t0
        out["prop_scale"] = ps
        all_json[str(ps)] = out
        for key, variants in out["summary"].items():
            for name, row in variants.items():
                print(f"ps={ps} {key} {name}",
                      json.dumps(row, sort_keys=True), flush=True)
    save_json(args.out, all_json)
    return all_json


# ------------------------------------------------------------------ synthetic


def _draw_law(rng, n, q, L, law, contam_p=0.02, contam_s=10.0):
    """One score matrix with covariance L L' and the requested marginal law.

    ``heavy`` is a variance-mixture ("contaminated") law tuned to mimic the
    kurtosis of the fitted IPW score: S = contam_s with probability
    contam_p and 1 otherwise, standardised to unit variance.  It has the same
    covariance as ``gaussian`` but kurtosis
    (1-p+p s^4)/(1-p+p s^2)^2, about 23 at the defaults.
    """
    if law == "gaussian":
        return rng.standard_normal((n, q)) @ L.T
    if law == "skew":
        e = rng.exponential(1.0, size=(n, q)) - 1.0  # unit variance, skew 2
        return e @ L.T
    if law == "heavy":
        scale = np.sqrt(1.0 - contam_p + contam_p * contam_s ** 2)
        s = np.where(rng.random((n, q)) < contam_p, contam_s, 1.0)
        return (rng.standard_normal((n, q)) * s) @ L.T / scale
    raise ValueError(law)


def _multiplier_crit(centered: np.ndarray, n_draws: int, rng,
                     alpha: float = ALPHA) -> float:
    n = centered.shape[0]
    null = np.empty(n_draws)
    for lo in range(0, n_draws, 512):
        hi = min(lo + 512, n_draws)
        xi = rng.standard_normal((hi - lo, n))
        null[lo:hi] = np.max(np.abs(xi @ centered), axis=1) / np.sqrt(n)
    return float(np.quantile(null, 1.0 - alpha))


def _true_stat_pool(n, q, L, g_vec, law, n_pool, rng, batch=1000,
                    contam_p=0.02, contam_s=10.0):
    """Pool of sqrt(n)-scaled score means + g from the true law (batched)."""
    out = np.empty(n_pool)
    done = 0
    while done < n_pool:
        b = min(batch, n_pool - done)
        Z = _draw_law(rng, b * n, q, L, law, contam_p, contam_s).reshape(b, n, q)
        out[done:done + b] = np.max(
            np.abs(Z.mean(axis=1) * np.sqrt(n) + g_vec), axis=1)
        done += b
    return out


def mode_synthetic(args):
    """Mechanism (c): small grids with an exactly known score law.

    For each (q, n, law) the *true* law of the statistic is a calibrated
    simulation: the normalized sum of n iid score vectors from the stated law,
    plus the shift, maximized over q coordinates.  Each of R test datasets
    produces a critical value from (i) its own Gaussian covariance, (ii) the
    Gaussian multiplier bootstrap, (iii) the unit bootstrap; the true
    rejection probability of each method is the share of the true-law pool
    above that critical value, averaged over datasets.  The pooled
    Gaussian-shift prediction (the fleet's prediction) is also recorded.
    Because the covariance is identical across laws, any law-dependence in
    the calibration columns is pure finite-sample skew/kurtosis.
    """
    rng = np.random.default_rng(12345)
    rows = []
    n_pool = args.n_pool
    for q in [int(x) for x in args.qs.split(",")]:
        for n in [int(x) for x in args.ns.split(",")]:
            rho = args.rho
            Sigma = rho ** np.abs(np.subtract.outer(np.arange(q), np.arange(q)))
            L = np.linalg.cholesky(Sigma + 1e-12 * np.eye(q))
            g_vec = np.zeros(q)
            g_vec[0] = args.scale
            for law in args.laws.split(","):
                R = args.reps
                pool = _true_stat_pool(n, q, L, g_vec, law, n_pool, rng,
                                       contam_p=args.contam_prob,
                                       contam_s=args.contam_scale)
                cov_sum = np.zeros((q, q))
                rej_mult = rej_np = rej_g = 0.0
                rej_mult_self = rej_np_self = 0
                crits = []
                for _ in range(R):
                    rng_law = np.random.default_rng(rng.integers(1 << 31))
                    Z = _draw_law(rng_law, n, q, L, law,
                                  args.contam_prob, args.contam_scale)
                    scores = Z + g_vec[None, :] / np.sqrt(n)
                    est = scores.mean(axis=0)
                    centered = scores - est[None, :]
                    cov_hat = centered.T @ centered / n
                    cov_sum += cov_hat
                    stat = float(np.sqrt(n) * np.max(np.abs(est)))
                    cg = gauss_crit_power(cov_hat, g_vec, n_draws=4000,
                                          seed=int(rng_law.integers(1 << 31)))["crit"]
                    cm = _multiplier_crit(centered, args.n_cal, rng_law)
                    cb = boot_crit(centered, n_boot=1000, seed=int(rng_law.integers(1 << 31)))
                    crits.append((cg, cm, cb))
                    rej_g += float(np.mean(pool > cg))
                    rej_mult += float(np.mean(pool > cm))
                    rej_np += float(np.mean(pool > cb))
                    rej_mult_self += int(stat > cm)
                    rej_np_self += int(stat > cb)
                cov_pool = cov_sum / R
                pooled = gauss_crit_power(cov_pool, g_vec, n_draws=100000,
                                          seed=3)
                cg_all = np.array([c[0] for c in crits])
                cm_all = np.array([c[1] for c in crits])
                cb_all = np.array([c[2] for c in crits])
                rows.append({
                    "q": q, "n": n, "law": law, "reps": R,
                    "pooled_gauss_power": pooled["power"],
                    "true_rej_perrep_gauss": rej_g / R,
                    "true_rej_multiplier": rej_mult / R,
                    "true_rej_unit_bootstrap": rej_np / R,
                    "self_multiplier_rate": rej_mult_self / R,
                    "self_unit_bootstrap_rate": rej_np_self / R,
                    "crit_perrep_gauss_mean": float(cg_all.mean()),
                    "crit_multiplier_mean": float(cm_all.mean()),
                    "crit_unit_bootstrap_mean": float(cb_all.mean()),
                })
    save_json(args.out, {"rows": rows})
    for row in rows:
        print(json.dumps(row, sort_keys=True), flush=True)
    return rows


# ----------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    def common(p):
        p.add_argument("--workers", type=int, default=3)
        p.add_argument("--reps", type=int, default=60)
        p.add_argument("--chunk", type=int, default=10)
        p.add_argument("--scale", type=float, default=0.7)
        p.add_argument("--K", type=int, default=5)
        p.add_argument("--prop-scale", type=float, default=None)
        p.add_argument("--B", type=int, default=399)
        p.add_argument("--B-hi", type=int, default=None)
        p.add_argument("--n-draws", type=int, default=20000)
        p.add_argument("--n-boot", type=int, default=2000)
        p.add_argument("--out", required=True)

    p_core = sub.add_parser("core")
    common(p_core)
    p_core.add_argument("--cells", nargs="+", required=True,
                        help="setting:n:direction e.g. PROG1:500:freq")
    p_core.set_defaults(func=mode_core)

    p_basis = sub.add_parser("basis")
    common(p_basis)
    p_basis.add_argument("--K-list", default="5,7,9,15")
    p_basis.add_argument("--settings", default="PROG1")
    p_basis.add_argument("--ns", default="500")
    p_basis.add_argument("--directions", default="freq,bump,mean")
    p_basis.set_defaults(func=mode_basis)

    p_strength = sub.add_parser("strength")
    common(p_strength)
    p_strength.add_argument("--prop-scale-list", default="0.0,0.75,1.5,3.0")
    p_strength.add_argument("--settings", default="PROG1")
    p_strength.add_argument("--ns", default="500")
    p_strength.add_argument("--directions", default="freq")
    p_strength.set_defaults(func=mode_strength)

    p_syn = sub.add_parser("synthetic")
    p_syn.add_argument("--reps", type=int, default=200)
    p_syn.add_argument("--scale", type=float, default=0.7)
    p_syn.add_argument("--qs", default="4,8")
    p_syn.add_argument("--ns", default="100,500")
    p_syn.add_argument("--laws", default="gaussian,skew")
    p_syn.add_argument("--rho", type=float, default=0.5)
    p_syn.add_argument("--contam-prob", type=float, default=0.02)
    p_syn.add_argument("--contam-scale", type=float, default=10.0)
    p_syn.add_argument("--n-pool", type=int, default=20000)
    p_syn.add_argument("--n-cal", type=int, default=1999)
    p_syn.add_argument("--out", required=True)
    p_syn.set_defaults(func=mode_synthetic)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
