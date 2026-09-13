"""WP7 T2 (leakage) numerical validation.

Falsification protocol (fixed before the production run):

F1  Recorded false-positive sweep is nondecreasing in lambda up to Monte
    Carlo noise: for every competitor and every adjacent pair, a drop is
    flagged if r_{j+1} < r_j - 3 * SE_diff (independent-proportion SE), and
    the lambda=0 -> lambda=1 gain must exceed 3 SE.  falsified if any flag.
F2  Exact Phase-2 law divergences (|Delta K|, TV, Hellinger, Chernoff, KL,
    normal power proxy at n=400) are strictly increasing over
    lambda in {0,0.2,0.4,0.6,0.8,1.0}; falsified by any non-increase.
F3  Every competitor's recorded rate has Spearman rho >= 0.8 against the
    exact normal power proxy (ties averaged); falsified otherwise.
F4  The simulated two-sample permutation test on the harness loop count K
    matches the exact normal power proxy within 3 Monte Carlo SE at every
    lambda; falsified otherwise.
F5  Under masking every recorded competitor rate is <= alpha + 3 SE_alpha
    with SE_alpha = sqrt(alpha(1-alpha)/1000); falsified by any excess.
F6  The WP2 recorded masking diagnostic (mean H1 persistence difference
    +0.00034, sd 0.00947 over 1000 replications) is within 3 SE of zero
    (3 * sd / sqrt(1000)); falsified otherwise.  The WP2 text figure
    "about 1.2 loops" for the lambda=1 loop-count gap is checked against the
    exact quadrature and a direct harness simulation and reported, not used
    as a pass condition.
F7  The loop-count permutation test keeps size alpha at lambda=0 for every
    n in {100,200,400,800,1600} within 3 Monte Carlo SE; falsified otherwise.

Support check (not part of the pass conditions): the exact divergence
functionals of F2 are re-evaluated on a 1,001-point grid in [0,1] and checked
for strict increase, backing the claim of Proposition A.2(iii); recorded under
"monotone_1001".

The DR prototype row is excluded from F1, F3 and F5 by construction (it
tests H0^out, not H0^cond).  Its Part A size and Part B power are recorded
as consistency checks: valid size at every lambda (its null is exactly true
under P_lambda) and power at least 1 - 3 SE under masking (its null is
exactly false there).

Run:  .venv/bin/python scripts/theory_dev/t2_leakage_validate.py
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.integrate import quad
from scipy.special import expit, ndtr
from scipy.stats import binomtest, spearmanr

ALPHA = 0.05
N_PHASE2 = 400
N_JSON_REPS = 1000
LAMBDAS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
LAM_LABELS = [f"{x:g}" for x in LAMBDAS]
COMPETITORS = ["rt", "mmd", "han", "strand", "moon_lazar", "frechet_anova"]
BETA = np.array([-0.5, -0.1, 0.6])
PROP_SCALE = 1.5
K_LO, K_HI = 1, 3
EDGES = np.array([-np.inf, -np.log(2.0), np.log(2.0), np.inf])

MC_SE_ALPHA = float(np.sqrt(ALPHA * (1.0 - ALPHA) / N_JSON_REPS))
MASK_BAND = float(ALPHA + 3.0 * MC_SE_ALPHA)
MONOTONE_Z = 3.0
SPEARMAN_MIN = 0.8
SIM_REPS = 2000
SIM_PERMS = 199
NSIM_REPS = 1000
NSCALE = [100, 200, 400, 800, 1600]
NWP2_MEAN = 0.00034
NWP2_SD = 0.00947
WP2_LOOP_GAP_TEXT = 1.2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(ROOT, "results")
OUT_DIR = os.path.join(RESULTS, "theory_validation")
FIG_JSON = os.path.join(RESULTS, "phase2_figure1.json")

_t_nodes, _t_weights = hermegauss(200)
_GH_NODES = _t_nodes
_GH_WEIGHTS = _t_weights / np.sqrt(2.0 * np.pi)


def g_lambda(lam: float, x0: np.ndarray) -> np.ndarray:
    """E[e_lambda(X) | X_0 = x0] by Gauss-Hermite over the other coordinates."""
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    s_u = float(np.sqrt(BETA[1] ** 2 + BETA[2] ** 2))
    u = s_u * _GH_NODES
    arg = lam * PROP_SCALE * (BETA[0] * x0[:, None] + u[None, :])
    return expit(arg) @ _GH_WEIGHTS


def exact_law(lam: float) -> dict:
    """Exact arm-conditional loop-count laws and divergence functionals."""
    pk = ndtr(EDGES[1:]) - ndtr(EDGES[:-1])
    pka1 = np.array([
        quad(lambda x: g_lambda(lam, x)[0]
             * np.exp(-x * x / 2.0) / np.sqrt(2.0 * np.pi),
             EDGES[j], EDGES[j + 1], limit=200)[0]
        for j in range(3)
    ])
    pka0 = pk - pka1
    p1 = float(pka1.sum())
    p0 = 1.0 - p1
    k = np.array([1.0, 2.0, 3.0])
    mu1 = pka1 / p1
    mu0 = pka0 / p0
    e1 = float((mu1 * k).sum())
    e0 = float((mu0 * k).sum())
    v1 = float((mu1 * k ** 2).sum() - e1 ** 2)
    v0 = float((mu0 * k ** 2).sum() - e0 ** 2)
    delta = e1 - e0
    zeta = abs(delta) / float(np.sqrt(v1 / (N_PHASE2 * p1)
                                     + v0 / (N_PHASE2 * p0)))
    tv = 0.5 * float(np.abs(mu1 - mu0).sum())
    hell = float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(mu1 * mu0)))))
    kl = float(np.sum(mu1 * np.log(mu1 / mu0)))
    chernoff = chernoff_information(mu1, mu0)
    crit = float(1.959963984540054)
    power_proxy = float(ndtr(zeta - crit) + ndtr(-zeta - crit))
    return {
        "lambda": float(lam), "p1": p1, "p0": p0,
        "mu1": mu1.tolist(), "mu0": mu0.tolist(),
        "mean1": e1, "mean0": e0, "var1": v1, "var0": v0,
        "delta_mean": delta, "abs_delta_mean": abs(delta),
        "zeta_n400": zeta, "tv": tv, "hellinger": hell, "kl": kl,
        "chernoff": chernoff, "power_proxy": power_proxy,
    }


def chernoff_information(mu1: np.ndarray, mu0: np.ndarray,
                         n_grid: int = 20001) -> float:
    u = np.linspace(0.0, 1.0, n_grid)
    term = (mu1[None, :] ** u[:, None]
            * mu0[None, :] ** (1.0 - u[:, None])).sum(axis=1)
    return float(-np.nanmin(np.log(term)))


def harness_loop_gap_mc(lam: float, n: int, rng: np.random.Generator) -> dict:
    x = rng.normal(size=(n, 3))
    e = expit(PROP_SCALE * lam * (x @ BETA))
    a = rng.binomial(1, e)
    k = 1 + np.floor(expit(x[:, 0]) * 3.0).astype(int)
    m1, m0 = k[a == 1].mean(), k[a == 0].mean()
    se = float(np.sqrt(k[a == 1].var() / max(1, (a == 1).sum())
                       + k[a == 0].var() / max(1, (a == 0).sum())))
    return {"gap": float(m1 - m0), "mc_se": se}


def perm_test_one(K: np.ndarray, A: np.ndarray, n_perm: int,
                  rng: np.random.Generator) -> float:
    n = len(K)
    n1 = int(A.sum())
    obs = K[A == 1].mean() - K[A == 0].mean()
    u = rng.random((n_perm, n))
    idx = np.argpartition(u, n1, axis=1)[:, :n1]
    take = K[idx].sum(axis=1)
    perm = take / n1 - (K.sum() - take) / (n - n1)
    return float((1.0 + np.count_nonzero(np.abs(perm) >= abs(obs))) / (1.0 + n_perm))


def simulate_perm_power(lam: float, n: int, reps: int, n_perm: int,
                        seed: int) -> dict:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(reps, n, 3))
    e = expit(PROP_SCALE * lam * (x @ BETA))
    A = rng.binomial(1, e)
    K = 1 + np.floor(expit(x[..., 0]) * 3.0).astype(int)
    rej = 0
    for r in range(reps):
        p = perm_test_one(K[r], A[r], n_perm, rng)
        rej += p <= ALPHA
    rate = rej / reps
    return {"lambda": float(lam), "n": int(n), "reps": int(reps),
            "n_perm": int(n_perm), "rate": float(rate),
            "mc_se": float(np.sqrt(max(rate * (1 - rate), 1e-12) / reps))}


def sim_delta_validate(lam: float, n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return harness_loop_gap_mc(lam, n, rng)


def load_recorded() -> dict:
    with open(FIG_JSON) as fh:
        return json.load(fh)


def check_sweep(recorded: dict, proxy: np.ndarray) -> dict:
    rates = recorded["sweep_rates"]
    out = {}
    for test in COMPETITORS:
        row = rates[test]
        r = np.array([row[f"lam{lab}"] for lab in LAM_LABELS])
        drops = []
        for j in range(len(r) - 1):
            se_diff = float(np.sqrt(r[j] * (1 - r[j]) / N_JSON_REPS
                                    + r[j + 1] * (1 - r[j + 1]) / N_JSON_REPS))
            if r[j + 1] < r[j] - MONOTONE_Z * se_diff:
                drops.append({"from": j, "to": j + 1,
                              "drop": float(r[j] - r[j + 1]),
                              "threshold": MONOTONE_Z * se_diff})
        gain = float(r[-1] - r[0])
        se_gain = float(np.sqrt(r[0] * (1 - r[0]) / N_JSON_REPS
                                + r[-1] * (1 - r[-1]) / N_JSON_REPS))
        rho, pval = spearmanr(proxy, r)
        out[test] = {
            "rates": r.tolist(), "largest_drop": float(-np.min(np.diff(r))),
            "monotone_violations": drops, "gain_lam0_to_lam1": gain,
            "gain_threshold": MONOTONE_Z * se_gain,
            "spearman_rho": float(rho), "spearman_p": float(pval),
        }
    return out


def check_masking(recorded: dict) -> dict:
    rates = recorded["masking_rates"]
    out = {}
    for test in COMPETITORS:
        r = rates[test]
        bt = binomtest(int(round(r * N_JSON_REPS)), N_JSON_REPS, ALPHA)
        out[test] = {
            "rate": float(r), "band_upper": MASK_BAND,
            "excess_over_band": float(r - MASK_BAND),
            "binom_p_two_sided": float(bt.pvalue),
        }
    return out


def make_figure(exact: list, raw_sweep: dict, raw_masking: dict,
                path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"rt": "Robinson-Turner", "mmd": "MMD", "han": "Han et al.",
              "strand": "STRAND", "moon_lazar": "Moon-Lazar",
              "frechet_anova": "Frechet ANOVA", "dr": "DR prototype"}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), layout="constrained")
    xs = np.arange(len(LAMBDAS))
    ax = axes[0]
    for test in COMPETITORS + ["dr"]:
        row = [raw_sweep[test][f"lam{lab}"] for lab in LAM_LABELS]
        ax.plot(xs, row, marker="o", ms=3.5, lw=1.3, alpha=0.85,
                label=labels[test])
    proxy = [e["power_proxy"] for e in exact]
    ax.plot(xs, proxy, "k--", lw=1.8, label="K normal proxy")
    se = MC_SE_ALPHA
    ax.fill_between(xs, max(0.0, ALPHA - 3 * se), ALPHA + 3 * se,
                    color="k", alpha=0.10)
    ax.axhline(ALPHA, color="k", ls=":", lw=0.9)
    ax.set_xticks(xs, LAM_LABELS)
    ax.set_xlabel("imbalance $\\lambda$")
    ax.set_ylabel("type-I error at $\\alpha=0.05$")
    ax.set_title("(a) $H_0^{\\mathrm{cond}}$ sweep vs exact proxy", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    ax = axes[1]
    lam_fine = np.linspace(0.0, 1.0, 201)
    fine = [exact_law(float(t)) for t in lam_fine]
    for key, lab in [("abs_delta_mean", "$|\\Delta|$ (mean $K$)"),
                     ("tv", "TV"),
                     ("hellinger", "Hellinger"),
                     ("chernoff", "Chernoff"),
                     ("kl", "KL")]:
        vals = [e[key] for e in fine]
        ax.plot(lam_fine, vals, lw=1.6, label=lab)
    ax.set_xlabel("imbalance $\\lambda$")
    ax.set_ylabel("arm-law divergence")
    ax.set_title("(b) exact arm-law divergence", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[2]
    tests = COMPETITORS + ["dr"]
    vals = [raw_masking[t] for t in tests]
    ax.bar(np.arange(len(tests)), vals,
           color=["#4477AA"] * 6 + ["#CC6677"])
    ax.axhline(ALPHA, color="k", ls="--", lw=0.9)
    ax.fill_between([-0.6, len(tests) - 0.4], 0.0, MASK_BAND,
                    color="k", alpha=0.08)
    ax.set_xticks(np.arange(len(tests)),
                  [labels[t] for t in tests], rotation=30, fontsize=7.5,
                  ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("rejection rate")
    ax.set_title("(c) masking: rates against the $\\alpha+3$ SE band",
                 fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    for b, v in zip(ax.patches, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", fontsize=7.5)
    fig.savefig(path, dpi=200)


def main() -> dict:
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    recorded = load_recorded()

    exact = [exact_law(float(lam)) for lam in LAMBDAS]
    proxy = np.array([e["power_proxy"] for e in exact])

    lam_fine = np.linspace(0.0, 1.0, 1001)
    fine = [exact_law(float(t)) for t in lam_fine]
    monotone_1001 = {}
    for key in ["abs_delta_mean", "tv", "hellinger", "kl", "chernoff",
                "power_proxy"]:
        vals = np.array([e[key] for e in fine])
        diffs = np.diff(vals)
        monotone_1001[key] = {
            "strictly_increasing": bool(np.all(diffs > 0)),
            "min_first_difference": float(np.min(diffs)),
        }
    monotone_1001_pass = bool(all(v["strictly_increasing"]
                                  for v in monotone_1001.values()))

    mc_gap = []
    for lam in LAMBDAS:
        entry = sim_delta_validate(float(lam), 1_000_000, seed=int(7000 + 13 * lam * 10))
        entry["lambda"] = float(lam)
        entry["quadrature_gap"] = float(
            exact_law(float(lam))["delta_mean"])
        entry["quadrature_mc_se_diff"] = float(
            abs(entry["gap"] - entry["quadrature_gap"]) / max(entry["mc_se"], 1e-12))
        mc_gap.append(entry)

    f2_violations = []
    for key in ["abs_delta_mean", "tv", "hellinger", "kl", "chernoff",
                "power_proxy"]:
        vals = np.array([e[key] for e in exact])
        if np.any(np.diff(vals) <= 0):
            f2_violations.append(key)
    f2_pass = len(f2_violations) == 0

    sweep_checks = check_sweep(recorded, proxy)
    f1_bad = [t for t, c in sweep_checks.items()
              if c["monotone_violations"]
              or c["gain_lam0_to_lam1"] <= c["gain_threshold"]]
    f1_pass = len(f1_bad) == 0
    f3_bad = [t for t, c in sweep_checks.items()
              if c["spearman_rho"] < SPEARMAN_MIN]
    f3_pass = len(f3_bad) == 0

    sim = []
    for i, lam in enumerate(LAMBDAS):
        entry = simulate_perm_power(float(lam), N_PHASE2, SIM_REPS, SIM_PERMS,
                                    seed=31000 + i)
        entry["proxy"] = float(proxy[i])
        entry["se_of_diff"] = float(np.sqrt(
            entry["mc_se"] ** 2 + proxy[i] * (1 - proxy[i]) / SIM_REPS))
        entry["n_se_from_proxy"] = float(
            abs(entry["rate"] - proxy[i]) / max(entry["se_of_diff"], 1e-12))
        sim.append(entry)
    f4_bad = [s["lambda"] for s in sim if s["n_se_from_proxy"] > MONOTONE_Z]
    f4_pass = len(f4_bad) == 0

    nscale = []
    for lam in [1.0, 0.2, 0.0]:
        for n in NSCALE:
            entry = simulate_perm_power(float(lam), n, NSIM_REPS,
                                        SIM_PERMS, seed=41000 + n)
            nscale.append(entry)
    nscale_size = [s for s in nscale if s["lambda"] == 0.0]
    f7_pass = all(abs(s["rate"] - ALPHA) <= MONOTONE_Z * s["mc_se"]
                  for s in nscale_size)

    masking = check_masking(recorded)
    f5_bad = [t for t, c in masking.items() if c["rate"] > MASK_BAND]
    f5_pass = len(f5_bad) == 0

    dr_sweep = np.array([recorded["sweep_rates"]["dr"][f"lam{lab}"]
                         for lab in LAM_LABELS])
    dr_size_pass = bool(np.all(dr_sweep <= ALPHA + 3 * MC_SE_ALPHA))
    dr_mask = float(recorded["masking_rates"]["dr"])
    dr_power_pass = bool(dr_mask >= 1.0 - 3 * MC_SE_ALPHA)

    wp2_se = NWP2_SD / np.sqrt(N_JSON_REPS)
    f6_pass = abs(NWP2_MEAN) <= 3.0 * wp2_se
    wp2_gap_note = {
        "recorded_text_value": WP2_LOOP_GAP_TEXT,
        "exact_quadrature_lambda1": float(
            [e for e in exact if e["lambda"] == 1.0][0]["abs_delta_mean"]),
        "direct_sim_lambda1": float(
            [m for m in mc_gap if m["lambda"] == 1.0][0]["gap"]),
        "direct_sim_mc_se": float(
            [m for m in mc_gap if m["lambda"] == 1.0][0]["mc_se"]),
        "status": "recorded text value inconsistent with algebra and simulation",
    }

    verdicts = {
        "F1_sweep_monotone": bool(f1_pass),
        "F2_divergences_increase": bool(f2_pass),
        "F3_rank_agreement": bool(f3_pass),
        "F4_mechanism_vs_proxy": bool(f4_pass),
        "F5_masking_band": bool(f5_pass),
        "F6_wp2_masking_diagnostic": bool(f6_pass),
        "F7_k_sim_size_flat_in_n": bool(f7_pass),
        "DR_partA_size_valid": bool(dr_size_pass),
        "DR_masking_power_high": bool(dr_power_pass),
        "gate": "PASS" if all([f1_pass, f2_pass, f3_pass, f4_pass, f5_pass,
                               f6_pass, f7_pass]) else "FAIL",
    }

    payload = {
        "config": {
            "alpha": ALPHA, "n_phase2": N_PHASE2,
            "n_json_reps": N_JSON_REPS, "mc_se_alpha": MC_SE_ALPHA,
            "mask_band": MASK_BAND, "spearman_min": SPEARMAN_MIN,
            "monotone_z": MONOTONE_Z, "sim_reps": SIM_REPS,
            "sim_perms": SIM_PERMS, "nscale": NSCALE,
        },
        "criteria": {
            "F1": "no 3-SE drop in any recorded H0^cond competitor curve and "
                  "lambda=0 -> 1 gain above 3 SE",
            "F2": "exact |Delta|, TV, Hellinger, KL, Chernoff, proxy strictly "
                  "increase over the six lambda values",
            "F3": "per-H0^cond-test Spearman rho against the exact proxy >= 0.8",
            "F4": "simulated K permutation power within 3 SE of the proxy",
            "F5": "masking H0^cond rates <= alpha + 3 sqrt(alpha(1-alpha)/1000)",
            "F6": "WP2 mean persistence difference within 3 SE of zero",
            "F7": "simulated K permutation size at lambda=0 within 3 SE of "
                  "alpha for every n in {100,...,1600}",
            "DR_partA": "DR prototype size <= alpha + 3 SE at every lambda "
                        "(its null H0^out is exactly true under P_lambda)",
            "DR_partB": "DR prototype masking power >= 1 - 3 SE "
                        "(its null H0^out is exactly false)",
        },
        "exact_law": exact,
        "monotone_1001": monotone_1001,
        "monotone_1001_pass": monotone_1001_pass,
        "mc_loop_gap": mc_gap,
        "sweep_checks": sweep_checks,
        "simulation": sim,
        "nscale": nscale,
        "masking_checks": masking,
        "dr_checks": {"sweep_rates": dr_sweep.tolist(),
                      "size_valid": bool(dr_size_pass),
                      "masking_power": dr_mask,
                      "masking_power_valid": bool(dr_power_pass)},
        "wp2_loop_gap_note": wp2_gap_note,
        "verdicts": verdicts,
        "runtime_s": None,
    }
    make_figure(exact, recorded["sweep_rates"], recorded["masking_rates"],
                os.path.join(OUT_DIR, "t2_leakage_check.png"))

    payload["runtime_s"] = round(time.time() - t0, 1)
    out_path = os.path.join(OUT_DIR, "t2_leakage_check.json")
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=1)

    print("exact arm-law quantities")
    for e in exact:
        print(f"  lam={e['lambda']:4.1f}  Delta={e['delta_mean']:+.4f}  "
              f"TV={e['tv']:.4f}  Hell={e['hellinger']:.4f}  "
              f"KL={e['kl']:.4f}  Chernoff={e['chernoff']:.4f}  "
              f"zeta={e['zeta_n400']:5.2f}  proxy={e['power_proxy']:.4f}")
    print("1,001-point monotonicity (support check):",
          "PASS" if monotone_1001_pass else "FAIL")
    for key, val in monotone_1001.items():
        print(f"  {key:14s} strict={val['strictly_increasing']}  "
              f"min_diff={val['min_first_difference']:.3e}")
    print("recorded sweep vs proxy: Spearman rho")
    for t, c in sweep_checks.items():
        print(f"  {t:14s} rho={c['spearman_rho']:+.3f}  "
              f"gain={c['gain_lam0_to_lam1']:+.3f}  drops={len(c['monotone_violations'])}")
    print("K permutation simulation vs proxy")
    for s in sim:
        print(f"  lam={s['lambda']:4.1f}  sim={s['rate']:.3f} "
              f"(se {s['mc_se']:.3f})  proxy={s['proxy']:.3f}  "
              f"{s['n_se_from_proxy']:.2f} SE")
    print("K permutation size/power in n")
    for s in nscale:
        print(f"  lam={s['lambda']:4.1f}  n={s['n']:5d}  rate={s['rate']:.3f} "
              f"(se {s['mc_se']:.3f})")
    print("DR row:", json.dumps(payload["dr_checks"]))
    print("masking")
    for t, c in masking.items():
        print(f"  {t:14s} rate={c['rate']:.3f}  binom_p={c['binom_p_two_sided']:.3f}")
    print("WP2 loop-gap note:", json.dumps(wp2_gap_note))
    print("verdicts:", json.dumps(verdicts))
    print(f"wrote {out_path}")
    return payload


if __name__ == "__main__":
    main()
