"""WP T9 numerical falsification: multiplicity for the cross-degree max-statistic.

Pre-registered criteria (fixed before the run; see theory/WP7_T9_multiplicity.md):

C1 (shared-multiplier FWER).  At alpha = 0.05, B = 399, K = 2 degrees,
r = 25 grid points, the shared-Gaussian-multiplier max-statistic rejects at a
rate inside [alpha - 3*MC_SE, alpha + 3*MC_SE] in the four dependent cells
(n in {100, 400}, Gaussian and standardized non-Gaussian scores, rho_deg = 0.9)
and in the independent cell (n = 200, Gaussian, rho_deg = 0.0).
MC_SE = sqrt(p(1-p)/reps).

C2 (dependence preservation).  The per-degree-independent multiplier variant
(source VJM convention) rejects at a rate no larger than the shared variant
plus 3*MC_SE; the mapping-note conjecture is that positive cross-degree
dependence makes the independent-draw maximum stochastically larger
(conservative).  A one-sided violation is reported as a falsified direction.

S1 (exact scale invariance).  With degree 0's scores multiplied by a, the
pooled-studentized comparator p-value is exactly unchanged for every
replication (max absolute difference 0), because pooled centering and scaling
are 1-homogeneous.

S2 (collapse of the unstudentized max).  Against a fixed single-degree signal
with a degree-0 scale a in {1, 2, 4, 8, 16, 32, 64}, the unstudentized power at
a = 64 is at most alpha + 3*MC_SE, while the unstudentized power at a = 1 is
above alpha + 5*MC_SE (the signal is detectable at balanced scales).

S3 (power restored by studentization).  The pooled-studentized power at a = 64
is within 3*MC_SE of the a = 1 value, and at a = 1 the unstudentized and
studentized powers agree within 3*MC_SE (no material cost at balanced scales).

P1 (pooled exactness).  Under exact exchangeability of the B + 1 augmented
values, the pooled convention's rejection rate equals alpha within 3*MC_SE for
B in {19, 39, 99, 199, 399} at alpha = 0.05 (alpha(B+1) integer in each cell).

P2 (source convention defect).  The source (null-only) standardization's
absolute size error at B = 19 is larger than at B = 399, and its B = 399 error
is within 3*MC_SE of zero; an ordering reversal is reported as falsified.

The simulation model is the Gaussian-limit idealisation of the score process:
i.i.d. unit score vectors with an AR(1) within-degree covariance (rho_t = 0.6)
and, in the dependent cells, a positive cross-degree covariance
(rho_deg = 0.9); the n = 200 Gaussian cell sets rho_deg = 0.0 by construction.
The multiplier null uses one shared Gaussian draw per unit, as in
``tda2s/tests/dr_outcome.py`` and CCK (2013).
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tda2s.resample import p_value  # noqa: E402
from tda2s.tests.dr_outcome import _standardize_family  # noqa: E402

OUT = os.path.join(_ROOT, "results", "theory_validation")
os.makedirs(OUT, exist_ok=True)

ALPHA = 0.05
B_DRAWS = 399
K = 2
R_GRID = 25
MC_Z = 3.0
SCALES = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)


def mc_se(rate: float, reps: int) -> float:
    return float(np.sqrt(max(rate * (1.0 - rate), 1e-12) / reps))


def ar1_cov(r: int, rho: float) -> np.ndarray:
    idx = np.arange(r)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def cross_degree_cov(rho_t: float, rho_deg: float, k: int = K,
                     r: int = R_GRID) -> np.ndarray:
    within = ar1_cov(r, rho_t)
    blocks = np.empty((k, k, r, r), dtype=float)
    for a in range(k):
        for b in range(k):
            blocks[a, b] = within if a == b else rho_deg * within
    return blocks.transpose(0, 2, 1, 3).reshape(k * r, k * r)


def cov_cholesky(rho_deg: float = 0.9) -> np.ndarray:
    cov = cross_degree_cov(0.6, rho_deg)
    return np.linalg.cholesky(cov + 1e-10 * np.eye(K * R_GRID))


def score_draws(rng: np.random.Generator, n: int, chol: np.ndarray,
                kind: str) -> np.ndarray:
    z = rng.standard_normal((n, chol.shape[0]))
    if kind == "chisq":
        z = (z ** 2 - 1.0) / np.sqrt(2.0)
    return z @ chol.T


def max_pvalue(observed: np.ndarray, null: np.ndarray) -> float:
    return float(p_value(float(np.max(observed)), np.max(null, axis=1)))


def pooled_studentized_pvalue(observed: np.ndarray, null: np.ndarray,
                              convention: str = "pooled") -> float:
    _, _, z_obs, z_null = _standardize_family(observed, null, convention, 1e-10)
    return float(p_value(float(np.max(z_obs)), np.max(z_null, axis=1)))


def fwer_replication(rng: np.random.Generator, n: int, chol: np.ndarray,
                     kind: str):
    scores = score_draws(rng, n, chol, kind)
    observed = np.sqrt(n) * np.abs(scores.mean(axis=0)).reshape(K, R_GRID).max(axis=1)
    centered = scores - scores.mean(axis=0)

    xi = rng.standard_normal((B_DRAWS, n))
    shared = np.abs(xi @ centered).reshape(B_DRAWS, K, R_GRID)
    shared = shared.max(axis=2) / np.sqrt(n)
    p_shared = max_pvalue(observed, shared)

    per_degree = np.empty((B_DRAWS, K))
    for d in range(K):
        xi_d = rng.standard_normal((B_DRAWS, n))
        block = xi_d @ centered[:, d * R_GRID:(d + 1) * R_GRID]
        per_degree[:, d] = np.abs(block).max(axis=1) / np.sqrt(n)
    p_indep = max_pvalue(observed, per_degree)
    return p_shared, p_indep


def run_fwer(n: int, reps: int, seed: int, kind: str, rho_deg: float = 0.9):
    rng = np.random.default_rng(seed)
    chol = cov_cholesky(rho_deg)
    shared_rej = 0
    indep_rej = 0
    for _ in range(reps):
        p_shared, p_indep = fwer_replication(rng, n, chol, kind)
        shared_rej += p_shared <= ALPHA
        indep_rej += p_indep <= ALPHA
    rate_shared = shared_rej / reps
    rate_indep = indep_rej / reps
    return {
        "n": n,
        "kind": kind,
        "rho_deg": rho_deg,
        "reps": reps,
        "alpha": ALPHA,
        "shared_rate": rate_shared,
        "shared_mc_se": mc_se(rate_shared, reps),
        "indep_rate": rate_indep,
        "indep_mc_se": mc_se(rate_indep, reps),
    }


def power_replication(rng: np.random.Generator, n: int, chol: np.ndarray,
                      signal: float, scales=SCALES):
    scores = score_draws(rng, n, chol, "gaussian")
    scores[:, R_GRID:2 * R_GRID] += signal / np.sqrt(n)
    xi = rng.standard_normal((B_DRAWS, n))
    out = {}
    for a in scales:
        scaled = scores.copy()
        scaled[:, :R_GRID] *= a
        observed = np.sqrt(n) * np.abs(scaled.mean(axis=0)).reshape(K, R_GRID).max(axis=1)
        centered = scaled - scaled.mean(axis=0)
        null = np.abs(xi @ centered).reshape(B_DRAWS, K, R_GRID).max(axis=2)
        null = null / np.sqrt(n)
        out[a] = {
            "unstud": max_pvalue(observed, null),
            "pooled": pooled_studentized_pvalue(observed, null, "pooled"),
            "source": pooled_studentized_pvalue(observed, null, "source"),
        }
    return out


def run_power(n: int, reps: int, seed: int, signal: float, scales=SCALES):
    rng = np.random.default_rng(seed)
    chol = cov_cholesky()
    rej = {a: {"unstud": 0, "pooled": 0, "source": 0} for a in scales}
    exact_mismatch = 0.0
    for _ in range(reps):
        out = power_replication(rng, n, chol, signal, scales)
        ref = out[scales[0]]["pooled"]
        for a in scales:
            for key in ("unstud", "pooled", "source"):
                rej[a][key] += out[a][key] <= ALPHA
            exact_mismatch = max(exact_mismatch, abs(out[a]["pooled"] - ref))
    rows = []
    for a in scales:
        row = {"scale": a, "n": n, "signal": signal, "reps": reps}
        for key in ("unstud", "pooled", "source"):
            rate = rej[a][key] / reps
            row[key + "_rate"] = rate
            row[key + "_mc_se"] = mc_se(rate, reps)
        rows.append(row)
    return rows, exact_mismatch


def pool_size_check(rng: np.random.Generator, reps: int, B: int, convention: str):
    cov = np.array([[1.0, 0.6], [0.6, 1.0]])
    chol = np.linalg.cholesky(cov)
    rej = 0
    for _ in range(reps):
        values = rng.standard_normal((B + 1, 2)) @ chol.T
        values = np.abs(values) + 0.1 * values ** 2
        rej += pooled_studentized_pvalue(values[0], values[1:], convention) <= ALPHA
    rate = rej / reps
    return {"B": B, "convention": convention, "reps": reps, "rate": rate,
            "mc_se": mc_se(rate, reps)}


def run_pool(reps: int, seed: int, Bs=(19, 39, 99, 199, 399)):
    rng = np.random.default_rng(seed)
    rows = []
    for B in Bs:
        for convention in ("pooled", "source"):
            rows.append(pool_size_check(rng, reps, B, convention))
    return rows


def _task_fwer(spec):
    return run_fwer(*spec)


def _task_power(spec):
    return run_power(*spec)


def _task_pool(spec):
    return run_pool(*spec)


def evaluate_checks(fwer_rows, power_rows, pool_rows):
    checks = {}
    checks["C1_all_shared_in_band"] = all(
        abs(r["shared_rate"] - ALPHA) <= MC_Z * r["shared_mc_se"] for r in fwer_rows)
    dependent = [r for r in fwer_rows if r["rho_deg"] > 0.0]
    checks["C2_all_indep_direction"] = all(
        r["indep_rate"] <= r["shared_rate"] + MC_Z * max(r["shared_mc_se"], r["indep_mc_se"])
        for r in dependent)
    checks["S1_exact_scale_invariance"] = all(
        row["pooled_exact_mismatch"] == 0.0 for row in power_rows)
    for n in sorted({r["n"] for r in power_rows}):
        rows = [r for r in power_rows if r["n"] == n]
        top = max(rows, key=lambda r: r["scale"])
        one = min(rows, key=lambda r: r["scale"])
        checks[f"S2_n{n}_collapse"] = bool(
            top["unstud_rate"] <= ALPHA + MC_Z * top["unstud_mc_se"] + 1e-12
            and one["unstud_rate"] >= ALPHA + 5 * one["unstud_mc_se"])
        checks[f"S3_n{n}_retained"] = bool(
            abs(top["pooled_rate"] - one["pooled_rate"])
            <= MC_Z * max(top["pooled_mc_se"], one["pooled_mc_se"])
            and abs(one["unstud_rate"] - one["pooled_rate"])
            <= MC_Z * max(one["unstud_mc_se"], one["pooled_mc_se"]))
    for row in pool_rows:
        row["pooled_target"] = float(np.floor(ALPHA * (row["B"] + 1)) / (row["B"] + 1))
    pooled_rows = [r for r in pool_rows if r["convention"] == "pooled"]
    checks["P1_pooled_exact"] = all(
        abs(r["rate"] - r["pooled_target"]) <= MC_Z * r["mc_se"] for r in pooled_rows)
    source_rows = sorted([r for r in pool_rows if r["convention"] == "source"],
                         key=lambda r: r["B"])
    if source_rows:
        small, large = source_rows[0], source_rows[-1]
        checks["P2_source_error_decays"] = bool(
            abs(small["rate"] - ALPHA) >= abs(large["rate"] - ALPHA)
            and abs(large["rate"] - ALPHA) <= MC_Z * large["mc_se"])
    return checks


def write_figure(power_rows, out_dir=None, filename="t9_power_allocation.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.6, 4.0))
        colors = {200: ("#D55E00", "#E69F00"), 500: ("#0072B2", "#56B4E9")}
        for n, marker in ((200, "o"), (500, "s")):
            rows = sorted([r for r in power_rows if r["n"] == n],
                          key=lambda r: r["scale"])
            scales = [r["scale"] for r in rows]
            rate = np.asarray([r["unstud_rate"] for r in rows])
            se = np.asarray([r["unstud_mc_se"] for r in rows])
            ax.errorbar(scales, rate, yerr=2 * se, marker=marker,
                        color=colors[n][0], capsize=2, elinewidth=0.7,
                        label=f"unstudentized shared max, $n={n}$")
            rate = np.asarray([r["pooled_rate"] for r in rows])
            se = np.asarray([r["pooled_mc_se"] for r in rows])
            ax.errorbar(scales, rate, yerr=2 * se, marker=marker,
                        linestyle="--", color=colors[n][1], capsize=2,
                        elinewidth=0.7, alpha=0.95,
                        label=f"pooled-studentized comparator, $n={n}$")
        ax.axhline(ALPHA, color="black", linewidth=0.8, linestyle=":")
        ax.text(1.05, ALPHA + 0.012, r"$\alpha=0.05$", fontsize=7.5,
                color="0.3")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("degree-0 scale multiplier $a$")
        ax.set_ylabel("power (rejection rate) at $\\alpha=0.05$")
        ax.set_ylim(-0.03, 0.78)
        ax.set_title("Power under degree-0 scale heterogeneity", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = os.path.join(out_dir or OUT, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path
    except Exception as exc:  # pragma: no cover
        print("figure skipped:", exc)
        return None


def main():
    quick = "--quick" in sys.argv
    workers = 1
    for arg in sys.argv:
        if arg.startswith("--workers="):
            workers = max(1, min(7, int(arg.split("=", 1)[1])))
    reps_fwer = 600 if quick else 2500
    reps_power = 400 if quick else 1500
    reps_pool = 4000 if quick else 12000
    signal = 1.7

    fwer_specs = [(n, reps_fwer, 1000 + 17 * c, kind, rho_deg)
                  for c, (n, kind, rho_deg) in enumerate(
                      [(n, kind, 0.9) for n in (100, 400)
                       for kind in ("gaussian", "chisq")]
                      + [(200, "gaussian", 0.0)])]
    power_specs = [(n, reps_power, 7000 + n, signal) for n in (200, 500)]
    pool_specs = [(reps_pool, 31337)]

    if workers > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=workers) as pool:
            fwer_rows = pool.map(_task_fwer, fwer_specs)
            power_payload = pool.map(_task_power, power_specs)
            pool_rows = pool.map(_task_pool, pool_specs)[0]
    else:
        fwer_rows = [_task_fwer(spec) for spec in fwer_specs]
        power_payload = [_task_power(spec) for spec in power_specs]
        pool_rows = _task_pool(pool_specs[0])

    power_rows = []
    for rows, mismatch in power_payload:
        for row in rows:
            row["pooled_exact_mismatch"] = mismatch
        power_rows.extend(rows)

    checks = evaluate_checks(fwer_rows, power_rows, pool_rows)
    summary = {
        "alpha": ALPHA,
        "B_draws": B_DRAWS,
        "scales": list(SCALES),
        "criteria": {
            "C1": "shared FWER within 3 MC SE of alpha for every cell",
            "C2": "independent-per-degree FWER <= shared + 3 MC SE",
            "S1": "max |pooled p(a) - pooled p(1)| == 0",
            "S2": "unstud power(a=64) <= alpha + 3 MC SE < unstud power(a=1)",
            "S3": "|pooled power(a=64) - pooled power(a=1)| <= 3 MC SE and "
                  "|unstud - pooled| at a=1 <= 3 MC SE",
            "P1": "pooled size == floor(alpha(B+1))/(B+1) within 3 MC SE",
            "P2": "source size error at B=19 >= source error at B=399, "
                  "B=399 within 3 MC SE of alpha",
        },
        "fwer": fwer_rows,
        "power": power_rows,
        "pool": pool_rows,
        "checks": checks,
    }
    path = os.path.join(OUT, "t9_multiplicity_checks.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    fig = write_figure(power_rows)
    print(json.dumps(checks, indent=1, sort_keys=True))
    print("wrote", path)
    if fig:
        print("wrote", fig)


if __name__ == "__main__":
    main()
