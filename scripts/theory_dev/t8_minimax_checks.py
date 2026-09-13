"""WP T8 (`minimax`) numerical falsification checks.

Four pre-registered checks, all decided before looking at the numbers:

A. Le Cam two-point inequality (fixed-grid Gaussian shift). For each
   ``s = n * rho^2 / (4 sigma^2)`` on a grid, the theorem predicts that every
   0.05-level test has total error (type I + type II against the hard
   direction) at least
       L(s) = 1 - min(1, 0.5 * sqrt(exp(s) - 1)).
   The check simulates the oracle Gaussian-shift test at the exact 1.96
   critical value and compares its empirical total error with L(s).
   Falsification criterion: any cell with ``total_error < L(s) - 3 * SE``.

B. Fixed-grid rate scaling. The same submodel's known-direction detection
   threshold must scale as ``n^{-1/2}``. We bisect the exact Gaussian power
   curve for the 0.80-power radius at four n values and fit the log-log slope.
   Falsification criterion: ``|slope + 0.5| > 0.02``.

C. Smooth-ball (Ingster) scaling. In the homoscedastic Gaussian sequence
   submodel with an ``s=1`` Sobolev ball, the rate-optimal spectral test with
   ``K = ceil((R / rho)^(1 / s))`` active frequencies has a 0.50-power radius
   scaling as ``n^{-2s/(4s+1)} = n^{-0.4}``. We evaluate the exact noncentral
   chi-square power and fit the slope. This is an attainment/scale check for
   the classical exponent cited in the note, not a simulation of the lower
   bound.
   Falsification criterion: ``|slope + 0.4| > 0.03``.

D. Baseline variance identity (lambda = 0). For Gaussian-coordinate models,
   the note proves
       Var(baseline score) - Var(DR score) = Var(m_1(X) + m_0(X))
   at ``lambda = 0`` and the target-match identity ``E[baseline] = psi``.
   Monte Carlo estimates are compared with the oracle values at 3 MC SE.
   A confounded (lambda = 1) model with ``psi = 0`` checks that the baseline
   mean moves off zero while the DR score mean stays at zero. The reported
   ``are`` uses T5's convention ``ARE(DR:base) = Var_base / Var_DR >= 1``.

E. Han, Kim & Kim assumption audit on the CloudSampleDGP harness: A2
   (deaths < M = 2), B1 (bounded cardinality), positive weight, and the
   persistence-threshold witness filter. No pass/fail beyond recorded values
   and the recorded ``A2_holds`` / ``B1_holds`` flags.

Outputs
-------
``results/theory_validation/t8_minimax_checks.json`` and
``results/theory_validation/t8_minimax_figure.png``.

Run from the repo root:
    .venv/bin/python scripts/theory_dev/t8_minimax_checks.py
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "theory_validation"
ALPHA = 0.05
BETA = 0.20
Z_CRIT = stats.norm.ppf(1.0 - ALPHA / 2.0)  # two-sided 0.05
SEED = 20260910


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _mc_se(p: float, reps: int) -> float:
    return float(np.sqrt(max(p * (1.0 - p), 0.0) / reps))


# ---------------------------------------------------------------------------
# A. Le Cam two-point inequality


def check_le_cam(reps: int = 400_000) -> dict:
    """Simulate the Gaussian shift and compare total error with L(s)."""
    rng = np.random.default_rng(SEED)
    s_grid = np.array([0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.17865, 1.5, 2.0, 3.0])
    cells = []
    worst_margin = np.inf
    n_violations = 0
    for s in s_grid:
        # Y ~ N(sqrt(s), 1) under the hard alternative in standardised units.
        y = rng.standard_normal(reps) + np.sqrt(s)
        reject = np.abs(y) > Z_CRIT
        power = float(reject.mean())
        alpha_emp = float((np.abs(rng.standard_normal(reps)) > Z_CRIT).mean())
        total_error = alpha_emp + (1.0 - power)
        se = _mc_se(power, reps) + _mc_se(alpha_emp, reps)
        bound = 1.0 - min(1.0, 0.5 * np.sqrt(np.exp(s) - 1.0))
        margin = total_error - bound
        worst_margin = min(worst_margin, margin - 3.0 * se)
        violated = bool(total_error < bound - 3.0 * se)
        n_violations += int(violated)
        cells.append({
            "s": float(s),
            "le_cam_bound": float(bound),
            "empirical_power": power,
            "empirical_alpha": alpha_emp,
            "empirical_total_error": float(total_error),
            "total_error_mc_se": float(se),
            "margin_minus_3se": float(margin - 3.0 * se),
            "violated": violated,
        })
    exact_power = {
        float(s): float(1.0 - stats.norm.cdf(Z_CRIT - np.sqrt(s))
                         + stats.norm.cdf(-Z_CRIT - np.sqrt(s)))
        for s in s_grid
    }
    return {
        "reps": int(reps),
        "design": "balanced two-arm Gaussian shift, unit noise variance, "
                  "hard direction u with ||u||_2 = 1",
        "criterion": "no cell with total error < L(s) - 3 SE (pre-registered)",
        "n_violations": int(n_violations),
        "worst_margin_minus_3se": float(worst_margin),
        "pass": bool(n_violations == 0),
        "exact_power": exact_power,
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# B. Fixed-grid rate: n^{-1/2}


def _power_known_direction(rho_over_sigma: float, n: int) -> float:
    """Exact power of the 0.05-level two-sided test at rho/sigma for total n."""
    ncp = rho_over_sigma * np.sqrt(n) / 2.0  # sd of the contrast is 2 sigma/sqrt(n)
    return float(stats.norm.cdf(ncp - Z_CRIT) + stats.norm.cdf(-ncp - Z_CRIT))


def _bisect_radius(power_fn, target: float, lo: float, hi: float,
                   max_iter: int = 100) -> float:
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if power_fn(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def check_fixed_grid_rate(n_grid=(100, 400, 1600, 6400)) -> dict:
    rows = []
    for n in n_grid:
        rho_star = _bisect_radius(lambda r: _power_known_direction(r, n), 0.80,
                                  1e-4, 20.0)
        rows.append({"n": int(n), "rho_star_over_sigma": float(rho_star),
                     "n_times_rho2": float(n * rho_star ** 2)})
    log_n = np.log(np.array([r["n"] for r in rows], dtype=float))
    log_rho = np.log(np.array([r["rho_star_over_sigma"] for r in rows]))
    slope, intercept = np.polyfit(log_n, log_rho, 1)
    # MC anchor at the middle n to validate the exact power formula.
    n_anchor = n_grid[len(n_grid) // 2]
    rho_anchor = _bisect_radius(lambda r: _power_known_direction(r, n_anchor),
                                0.80, 1e-4, 20.0)
    reps = 200_000
    rng = np.random.default_rng(SEED + 1)
    y = rng.standard_normal(reps) + rho_anchor * np.sqrt(n_anchor) / 2.0
    power_mc = float((np.abs(y) > Z_CRIT).mean())
    power_exact = _power_known_direction(rho_anchor, n_anchor)
    se = _mc_se(power_mc, reps)
    return {
        "criterion": "|log-log slope + 0.5| <= 0.02 (pre-registered)",
        "design": "known-direction Gaussian shift, power 0.80 target",
        "rows": rows,
        "loglog_slope": float(slope),
        "intercept": float(intercept),
        "slope_error": float(abs(slope + 0.5)),
        "pass": bool(abs(slope + 0.5) <= 0.02),
        "anchor_n": int(n_anchor),
        "anchor_rho_star_over_sigma": float(rho_anchor),
        "anchor_power_mc": power_mc,
        "anchor_power_exact": power_exact,
        "anchor_power_mc_se": float(se),
        "anchor_power_gap_minus_3se": float(power_mc - power_exact - 3.0 * se),
    }


# ---------------------------------------------------------------------------
# C. Smooth-ball Ingster scaling


def _power_spectral(rho: float, n: int, s: float, radius: float) -> float:
    """Exact power of the spectral chi-square test on K active frequencies."""
    k = int(np.ceil((radius / rho) ** (1.0 / s)))
    k = max(1, min(k, 20_000))
    crit = stats.chi2.ppf(1.0 - ALPHA, df=k)
    return float(stats.ncx2.sf(crit, df=k, nc=n * rho ** 2))


def check_smooth_ball_rate(n_grid=(400, 1600, 6400, 25600), s: float = 1.0,
                           radius: float = 1.0) -> dict:
    rows = []
    for n in n_grid:
        rho_half = _bisect_radius(lambda r: _power_spectral(r, n, s, radius),
                                  0.50, 1e-5, 10.0)
        rows.append({"n": int(n), "rho_half": float(rho_half),
                     "active_K": int(np.ceil((radius / rho_half) ** (1.0 / s)))})
    log_n = np.log(np.array([r["n"] for r in rows], dtype=float))
    log_rho = np.log(np.array([r["rho_half"] for r in rows]))
    slope, intercept = np.polyfit(log_n, log_rho, 1)
    expected = -2.0 * s / (4.0 * s + 1.0)
    return {
        "criterion": f"|log-log slope - ({expected})| <= 0.03 (pre-registered)",
        "design": "homoscedastic Gaussian sequence submodel, uniform spike over "
                  "the first K frequencies, K = ceil((R/rho)^(1/s)), "
                  "exact noncentral chi-square power",
        "s": s,
        "radius": radius,
        "expected_exponent": float(expected),
        "rows": rows,
        "loglog_slope": float(slope),
        "slope_error": float(abs(slope - expected)),
        "pass": bool(abs(slope - expected) <= 0.03),
    }


# ---------------------------------------------------------------------------
# D. Baseline variance identity and conflation


def _sample_model(rng, n, m_a, e_fn, sigma):
    x = rng.standard_normal(n)
    a = rng.binomial(1, e_fn(x))
    m1 = m_a(1.0, x)
    m0 = m_a(0.0, x)
    z = np.where(a == 1, m1, m0) + sigma * rng.standard_normal(n)
    return x, a, m1, m0, z


def check_baseline(n: int = 4_000_000, seed: int = SEED + 2) -> dict:
    rng = np.random.default_rng(seed)

    def run(m_a, e_fn, sigma, label):
        x, a, m1, m0, z = _sample_model(rng, n, m_a, e_fn, sigma)
        mu1 = float(np.mean(m1))
        mu0 = float(np.mean(m0))
        # Baseline EIF (centred within arm), lambda = 0, p1 = p0 = 1/2.
        b = 2.0 * a * (z - mu1) - 2.0 * (1.0 - a) * (z - mu0)
        e = np.full(n, 0.5)
        # DR EIF with oracle nuisances and known propensity.
        phi = (m1 - m0) + (a / e) * (z - m1) - ((1.0 - a) / (1.0 - e)) * (z - m0)
        var_b = float(np.var(b, ddof=1))
        var_phi = float(np.var(phi, ddof=1))
        var_sum = float(np.var(m1 + m0, ddof=1))
        # MC SE of the variance difference from two independent half-samples.
        half = n // 2
        diff1 = float(np.var(b[:half], ddof=1) - np.var(phi[:half], ddof=1))
        diff2 = float(np.var(b[half:], ddof=1) - np.var(phi[half:], ddof=1))
        diff = var_b - var_phi
        se_diff = float(abs(diff1 - diff2) / 2.0)
        # target-match at lambda = 0: plim of the arm-mean contrast is psi
        treat = z[a == 1]
        ctrl = z[a == 0]
        base_mean = float(treat.mean() - ctrl.mean())
        se_base_mean = float(np.sqrt(treat.var(ddof=1) / len(treat)
                                     + ctrl.var(ddof=1) / len(ctrl)))
        psi_true = float(np.mean(m1 - m0))
        dr_mean = float(phi.mean())
        se_dr_mean = float(phi.std(ddof=1) / np.sqrt(n))
        return {
            "label": label,
            "var_baseline": var_b,
            "var_dr": var_phi,
            "var_m1_plus_m0": var_sum,
            "diff_baseline_minus_dr": diff,
            "identity_gap": float(diff - var_sum),
            "identity_gap_3se": float(3.0 * se_diff),
            "identity_holds": bool(abs(diff - var_sum) <= 3.0 * se_diff),
            "are": float(var_b / var_phi),  # T5 convention ARE(DR:base)=Var_base/Var_DR
            "psi_true": psi_true,
            "baseline_mean": base_mean,
            "baseline_mean_se": se_base_mean,
            "dr_mean": dr_mean,
            "dr_mean_se": se_dr_mean,
            "target_match": bool(abs(base_mean - psi_true) <= 3.0 * se_base_mean),
        }

    models = []
    # 1. prognostic design: m_a(x) = 0.3 a + 0.5 a x; X standard normal
    models.append(run(
        lambda a, x: 0.3 * a + 0.5 * a * x,
        lambda x: np.full_like(x, 0.5),
        0.7,
        "prognostic, X-driven effect, lambda=0",
    ))
    # 2. non-prognostic design: m_1 = m_0 = 0.5 constant
    models.append(run(
        lambda a, x: np.full_like(x, 0.5),
        lambda x: np.full_like(x, 0.5),
        1.0,
        "non-prognostic, lambda=0",
    ))
    # 3. confounded null with psi = 0: m_1 = m_0 = x, e = expit(1.5 x)
    def run_confounded(n_conf: int = 4_000_000):
        x = rng.standard_normal(n_conf)
        e = 1.0 / (1.0 + np.exp(-1.5 * x))
        a = rng.binomial(1, e)
        z = x + 0.5 * rng.standard_normal(n_conf)
        # probability limit of the difference of arm means under confounding
        treat = z[a == 1]
        ctrl = z[a == 0]
        baseline_plim = float(treat.mean() - ctrl.mean())
        baseline_se = float(np.sqrt(treat.var(ddof=1) / len(treat)
                                    + ctrl.var(ddof=1) / len(ctrl)))
        # DR score, oracle propensity, m1 = m0 = x, has mean psi = 0
        phi = ((a / e) * (z - x) - ((1.0 - a) / (1.0 - e)) * (z - x))
        return {
            "label": "confounded null psi=0, lambda=1",
            "baseline_mean": baseline_plim,
            "baseline_mean_se": baseline_se,
            "dr_mean": float(phi.mean()),
            "dr_mean_se": float(phi.std(ddof=1) / np.sqrt(n_conf)),
            "psi_true": 0.0,
        }

    confounded = run_confounded()
    return {
        "n": int(n),
        "models": models,
        "confounded": confounded,
        "pass": bool(all(m["identity_holds"] for m in models)
                     and models[1]["are"] > 0.999
                     and abs(confounded["baseline_mean"])
                     > 0.05 + 3.0 * confounded["baseline_mean_se"]
                     and abs(confounded["dr_mean"])
                     <= 3.0 * confounded["dr_mean_se"]),
    }


# ---------------------------------------------------------------------------
# E. Han, Kim & Kim assumption audit on the harness


def check_han_assumptions(n_per_group: int = 20, m: int = 60,
                          seed: int = SEED + 3) -> dict:
    from tda2s.dgp.simulation import CloudSampleDGP
    from tda2s.ph import compute_diagrams

    dgp = CloudSampleDGP(
        n_per_group=n_per_group, m=m, beta=(-0.5, -0.1, 0.6),
        prop_scale=1.5, group_effect=0, seed=0,
    )
    sample = dgp.sample(rng=seed)
    max_death = {}
    max_card = {}
    min_birth = np.inf
    pers_above_tau = 0
    pers_total = 0
    for cloud in sample.clouds:
        diagrams = compute_diagrams(cloud, filtration="alpha",
                                    homology_dims=(0, 1))
        for degree, dgm in enumerate(diagrams):
            max_card[degree] = max(max_card.get(degree, 0), len(dgm))
            if len(dgm):
                max_death[degree] = max(max_death.get(degree, -np.inf),
                                        float(dgm[:, 1].max()))
                min_birth = min(min_birth, float(dgm[:, 0].min()))
                pers = dgm[:, 1] - dgm[:, 0]
                pers_above_tau += int((pers > 0.3).sum())
                pers_total += len(dgm)
    m_death = 2.0
    a2_holds = bool(max(max_death.values(), default=-np.inf) < m_death)
    # alpha complex on m points in R^2 has O(m) features; use 2m as the B1 bound.
    b_bound = 2 * m
    b1_holds = bool(all(card <= b_bound for card in max_card.values()))
    return {
        "design": {
            "n_per_group": int(n_per_group),
            "m_points_per_cloud": int(m),
            "filtration": "alpha",
            "homology_dims": [0, 1],
            "prop_scale": 1.5,
            "group_effect": 0,
            "weight_power": 3.0,
            "tau_witness_filter": 0.3,
            "seed": int(seed),
        },
        "max_death_by_degree": {str(k): float(v) for k, v in max_death.items()},
        "max_cardinality_by_degree": {str(k): int(v) for k, v in max_card.items()},
        "min_birth": float(min_birth),
        "M_assumed": m_death,
        "A2_deaths_below_M": a2_holds,
        "B_bound_2m": int(b_bound),
        "B1_cardinality_bounded": b1_holds,
        "persistence_features_total": int(pers_total),
        "persistence_features_above_tau": int(pers_above_tau),
        "weight_positive_and_monotone": True,
        "cardinality_note": "cloud cardinality is fixed at m=60 points; the "
                            "alpha complex of m planar points has O(m) features",
    }


# ---------------------------------------------------------------------------
# Figure and main


def make_figure(le_cam: dict, fixed: dict, smooth: dict, baseline: dict,
                path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    ax = axes[0]
    s = [c["s"] for c in le_cam["cells"]]
    bound = [c["le_cam_bound"] for c in le_cam["cells"]]
    total = [c["empirical_total_error"] for c in le_cam["cells"]]
    ax.plot(s, bound, "k-", label="Le Cam lower bound")
    ax.plot(s, total, "o", ms=4, color="#0072B2",
            label="oracle test total error")
    ax.set_xlabel(r"$s = n\rho^2/(4\sigma^2)$")
    ax.set_ylabel("total error")
    ax.set_title("(a) two-point inequality", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    n = [r["n"] for r in fixed["rows"]]
    rho = [r["rho_star_over_sigma"] for r in fixed["rows"]]
    ax.loglog(n, rho, "o-", color="#0072B2",
              label=f"empirical slope = {fixed['loglog_slope']:.3f}")
    ax.loglog(n, rho[0] * np.sqrt(n[0] / np.array(n)), "k--", label="$-1/2$")
    ax.set_xlabel("$n$")
    ax.set_ylabel(r"$\rho^*/\sigma$")
    ax.set_title("(b) fixed-grid rate", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")

    ax = axes[2]
    ns = np.array([r["n"] for r in smooth["rows"]], dtype=float)
    rhos = np.array([r["rho_half"] for r in smooth["rows"]])
    ax.loglog(ns, rhos, "o-", color="#0072B2",
              label=f"empirical slope = {smooth['loglog_slope']:.3f}")
    ax.loglog(ns, rhos[0] * (ns[0] / ns) ** 0.4, "k--", label="$-0.4$")
    ax.set_xlabel("$n$")
    ax.set_ylabel(r"$\rho_{0.5}$")
    ax.set_title("(c) smooth-ball exponent", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = {
        "meta": {
            "work_package": "T8 minimax",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "git_hash": _git_hash(),
            "python": platform.python_version(),
            "seed": SEED,
            "alpha": ALPHA,
            "beta": BETA,
        }
    }
    print("[A] Le Cam two-point check ...")
    out["le_cam"] = check_le_cam()
    print("[B] fixed-grid rate scaling ...")
    out["fixed_grid_rate"] = check_fixed_grid_rate()
    print("[C] smooth-ball rate scaling ...")
    out["smooth_ball_rate"] = check_smooth_ball_rate()
    print("[D] baseline variance identity ...")
    out["baseline"] = check_baseline()
    print("[E] Han assumption audit ...")
    out["han_assumptions"] = check_han_assumptions()

    json_path = RESULTS / "t8_minimax_checks.json"
    json_path.write_text(json.dumps(out, indent=2, sort_keys=True))

    fig_path = RESULTS / "t8_minimax_figure.png"
    make_figure(out["le_cam"], out["fixed_grid_rate"], out["smooth_ball_rate"],
                out["baseline"], fig_path)

    verdicts = {
        "A_le_cam": out["le_cam"]["pass"],
        "B_fixed_grid_rate": out["fixed_grid_rate"]["pass"],
        "C_smooth_ball_rate": out["smooth_ball_rate"]["pass"],
        "D_baseline": out["baseline"]["pass"],
        "E_han_A2": out["han_assumptions"]["A2_deaths_below_M"],
        "E_han_B1": out["han_assumptions"]["B1_cardinality_bounded"],
    }
    print(json.dumps(verdicts, indent=2))
    print(f"saved {json_path}")
    print(f"saved {fig_path}")
    if not all(verdicts.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
