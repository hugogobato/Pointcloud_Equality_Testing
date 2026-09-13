#!/usr/bin/env python
"""WP T1 numerical witness: target-relative non-dominance on the Phase-2 DGPs.

Pre-registered falsification criteria (fixed before the first run; the
deliverable ``theory/WP7_T1_target_relative.md`` reports them verbatim):

A1 (Phase-2 leakage anchor).  In ``results/phase2_figure1.json``, at
    ``lambda=1`` the minimum sweep rejection rate over
    {rt, mmd, han, strand, moon_lazar} is >= 0.96, and the DR prototype rate
    is <= 0.02 at every lambda.
A2 (Phase-2 masking anchor).  The maximum masking rejection rate over the six
    H0^cond competitors is <= 0.10 and the DR prototype masking rate is 1.0.
A3 (Phase-2 randomisation control).  Every competitor rate at ``lambda=0``
    lies in [0.02, 0.08].
B1 (Harness construction).  The oracle loop count equals the deterministic
    knob ``1 + floor(expit(gamma * x0) * k_max)`` for every unit, so the
    conditional diagram law given X does not depend on the arm and
    ``psi_d = 0`` exactly.
B2 (Harness H0^cond violation).  ``mean(k | A=0) - mean(k | A=1) >= 0.5``
    with a two-sided label-permutation p-value <= 0.01 (2000 draws).
B3 (Harness target null, exact).  With a set of covariate rows repeated so that
    each distinct row occurs in both arms, the arm-conditional oracle mean loop
    counts agree exactly at every matched row (max absolute conditional-mean
    difference = 0), so the covariate-standardised contrast is identically
    zero.  The true-propensity IPW resampling distribution is reported
    descriptively (mean, sd, z), not gated.
C1 (Masking algebra).  The two arm type-frequency vectors are equal exactly
    (total variation distance 0 to 1e-12) at the Phase-2 masking parameters,
    and an independent Monte Carlo draw of the masking DGP shows no arm-type
    association (permutation p >= 0.05).
C2 (Masking H0^out violation).  ``sup_t |(8/45)(mbar_C - (mbar_A+mbar_B)/2)|``
    over the stacked silhouette grid is >= 0.5 and >= 5 times its batch Monte
    Carlo standard error (8 batches of 9 clouds per type, interval (0,2),
    r=3, resolution 100).
C3 (Masking H0^dist,grid violation).  The same contrast on the stacked
    fixed-grid expected persistence measures (interval (0,2), 32 bins, weight
    power 3, degrees 0 and 1) has L1 norm >= 0.5 and maximum absolute
    coordinate >= 0.05.
D  (Divergence-silent compensation).  The exact family
    ``Q_1(2) = L_C + eps R``, ``Q_0(2) = q* + (7/15) eps R`` with
    ``R = (1/2, 0, -1/2)`` keeps the two arm marginals equal for
    ``eps in [0, 1/3]`` (exact rational arithmetic) while the finite-support
    target moves by ``(8/45) eps R != 0``.

A failed criterion is reported, not refit.  JSON output:
``results/theory_validation/t1_target_relative_witness.json``.

POST-HOC diagnostics (added after the first run, which missed B2 and C2 on
their magnitude legs while their significance legs passed; these are not part
of the primary gate and are reported as such):
PH1. Population harness gap ``E[k|A=0] - E[k|A=1]``, computed from 4,000,000
     covariate draws with the exact propensity, with its Monte Carlo SE.
PH2. Harness seed robustness: 25 independent samples at the same parameters,
     each with a 1,000-draw permutation p-value; report the minimum gap and
     the maximum p-value across seeds.
PH3. Masking batch positivity: the 8 batch silhouette-contrast suprema,
     reported individually, with the count of positive batches.
"""

from __future__ import annotations

import json
import os
import sys
import time
from fractions import Fraction

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scipy.special import expit  # noqa: E402

from tda2s.dgp import CloudSampleDGP, loops_cloud, masking_stratum_sample  # noqa: E402
from tda2s.ph import compute_diagrams  # noqa: E402
from tda2s.tests.dist_level import measure_features  # noqa: E402
from tda2s.vec import silhouette  # noqa: E402

OUT = os.path.join(_ROOT, "results", "theory_validation")
os.makedirs(OUT, exist_ok=True)

SEED = 20260910
INTERVAL = (0.0, 2.0)
RESOLUTION = 100
SIL_R = 3.0
N_BINS = 32
WEIGHT_POWER = 3.0
ALPHA = 0.05
N_REPS_PHASE2 = 1000
N_PERM = 2000
N_BATCH = 8
N_PER_BATCH = 9
M_POINTS = 120
NOISE = 0.05
RADII = {"A": 1.0, "B": 2.0, "C": 4.0}
LOOPS = {"A": 1, "B": 1, "C": 2}
#: WP2 Theorem 2.2 effect factor psi = (8/45)(m_C - (m_A + m_B)/2).
MASK_FACTOR = 8.0 / 45.0

F_ZERO = Fraction(0)
F_ONE = Fraction(1)
F_HALF = Fraction(1, 2)
F_QUARTER = Fraction(1, 4)
F_FOUR_15 = Fraction(4, 15)
F_SEVEN_15 = Fraction(7, 15)
F_EIGHT_45 = Fraction(8, 45)


def _perm_pvalue_gap(values: np.ndarray, labels: np.ndarray, rng,
                     n_perm: int = N_PERM) -> float:
    """Two-sided permutation p for the absolute mean gap in ``values``."""
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    obs = abs(values[labels == 1].mean() - values[labels == 0].mean())
    draws = 0
    for _ in range(n_perm):
        perm = rng.permutation(labels)
        gap = abs(values[perm == 1].mean() - values[perm == 0].mean())
        draws += int(gap >= obs)
    return (1.0 + draws) / (1.0 + n_perm)


def _perm_pvalue_tv(counts: np.ndarray, labels: np.ndarray, rng) -> float:
    """Two-sided permutation p for the total-variation distance of type laws."""
    counts = np.asarray(counts, dtype=float)
    labels = np.asarray(labels)

    def tv(lab):
        p1 = counts[lab == 1].sum(axis=0)
        p0 = counts[lab == 0].sum(axis=0)
        p1 = p1 / max(p1.sum(), 1.0)
        p0 = p0 / max(p0.sum(), 1.0)
        return 0.5 * float(np.abs(p1 - p0).sum())

    obs = tv(labels)
    draws = 0
    for _ in range(N_PERM):
        draws += int(tv(rng.permutation(labels)) >= obs)
    return (1.0 + draws) / (1.0 + N_PERM)


def check_phase2_anchors() -> dict:
    with open(os.path.join(_ROOT, "results", "phase2_figure1.json")) as fh:
        data = json.load(fh)
    sweep = data["sweep_rates"]
    masking = data["masking_rates"]
    reps = data["n_reps"]["sweep"]
    mc_se = float(np.sqrt(ALPHA * (1.0 - ALPHA) / reps))
    cond_tests = ["rt", "mmd", "han", "strand", "moon_lazar"]
    lam1 = [sweep[t]["lam1"] for t in cond_tests]
    dr_all = [sweep["dr"][key] for key in sweep["dr"]]
    cond_mask = [masking[t] for t in cond_tests]
    return {
        "mc_se_at_alpha": mc_se,
        "lambda1_min_cond_rate": float(np.min(lam1)),
        "lambda1_cond_rates": {t: sweep[t]["lam1"] for t in cond_tests},
        "frechet_lambda1": sweep["frechet_anova"]["lam1"],
        "dr_rates": {k: sweep["dr"][k] for k in sweep["dr"]},
        "lambda0_competitor_rates": {
            t: sweep[t]["lam0"] for t in cond_tests + ["frechet_anova"]
        },
        "masking_cond_rates": {t: masking[t] for t in cond_tests},
        "masking_frechet": masking["frechet_anova"],
        "masking_dr": masking["dr"],
        "A1_pass": bool(np.min(lam1) >= 0.96 and max(dr_all) <= 0.02),
        "A2_pass": bool(max(cond_mask) <= 0.10 and masking["dr"] == 1.0),
        "A3_pass": bool(all(
            0.02 <= sweep[t]["lam0"] <= 0.08
            for t in cond_tests + ["frechet_anova"]
        )),
    }


def harness_witness() -> dict:
    beta = np.array([-0.5, -0.1, 0.6])
    dgp = CloudSampleDGP(
        n_per_group=400, m=M_POINTS, d_x=3, beta=beta, prop_scale=1.5,
        gamma=1.0, k_max=3, radius=1.0, noise=NOISE, group_effect=0,
        seed=SEED,
    )
    sample = dgp.sample(rng=SEED)
    X, A = sample.X, sample.A
    k = sample.true_n_loops.astype(float)
    knob = 1.0 + np.floor(expit(1.0 * X[:, 0]) * 3.0)
    b1 = float(np.max(np.abs(k - knob)))
    # B3: genuine arm-conditional oracle comparison. A set of covariate rows is
    # repeated within one sample so each distinct row occurs in both arms; with
    # group_effect = 0 the conditional oracle means must agree exactly.
    beta_dir = dgp.beta / float(dgp.beta @ dgp.beta)
    X_base = np.outer(np.linspace(-0.8, 0.8, 20), beta_dir)
    n_rep = 10
    X_rep = np.tile(X_base, (n_rep, 1))
    sample_rep = dgp.sample(n_per_group=X_rep.shape[0] // 2, X=X_rep,
                            rng=SEED + 6)
    k_rep = sample_rep.true_n_loops.astype(float)
    cells = {}
    for xi, ai, ki in zip(sample_rep.X, sample_rep.A, k_rep):
        cells.setdefault(tuple(np.round(xi, 10)), {0: [], 1: []})[int(ai)].append(ki)
    matched = [v for v in cells.values() if v[0] and v[1]]
    b3 = float(max(abs(np.mean(v[1]) - np.mean(v[0])) for v in matched))
    gap = float(k[A == 0].mean() - k[A == 1].mean())
    rng = np.random.default_rng(SEED + 1)
    b2_p = _perm_pvalue_gap(k, A, rng)

    e = dgp.propensity(X)
    n = len(k)
    redraws = 4000
    rng_ipw = np.random.default_rng(SEED + 2)
    a_draws = rng_ipw.random((redraws, n)) < e[None, :]
    psi_ipw = (
        (a_draws * (k / e)[None, :]).mean(axis=1)
        - (((1.0 - a_draws)) * (k / (1.0 - e))[None, :]).mean(axis=1)
    )
    z = float(psi_ipw.mean() / (psi_ipw.std(ddof=1) / np.sqrt(redraws)))

    # PH1: population gap by exact-propensity integration on 4e6 covariates.
    rng_pop = np.random.default_rng(SEED + 4)
    n_pop = 4_000_000
    X_pop = rng_pop.normal(size=(n_pop, 3))
    k_pop = 1.0 + np.floor(expit(X_pop[:, 0]) * 3.0)
    e_pop = expit(1.5 * (X_pop @ beta))
    m1_pop = float((e_pop * k_pop).mean() / e_pop.mean())
    m0_pop = float(((1.0 - e_pop) * k_pop).mean() / (1.0 - e_pop).mean())
    se1_pop = float((e_pop * k_pop).std(ddof=1) / np.sqrt(n_pop) / e_pop.mean())
    se0_pop = float(((1.0 - e_pop) * k_pop).std(ddof=1) / np.sqrt(n_pop)
                    / (1.0 - e_pop).mean())
    population_gap = m0_pop - m1_pop

    # PH2: seed robustness, 25 independent samples, 1000 permutations each.
    robustness = []
    for rep in range(25):
        s = dgp.sample(rng=SEED + 1000 + rep)
        kk = s.true_n_loops.astype(float)
        aa = s.A
        g = float(kk[aa == 0].mean() - kk[aa == 1].mean())
        pp = _perm_pvalue_gap(kk, aa, np.random.default_rng(SEED + 2000 + rep),
                              n_perm=1000)
        robustness.append({"rep": rep, "gap": g, "perm_pvalue": pp})
    return {
        "post_hoc_population_gap": population_gap,
        "post_hoc_population_gap_se": float(np.hypot(se0_pop, se1_pop)),
        "post_hoc_population_mean_k_treated": m1_pop,
        "post_hoc_population_mean_k_control": m0_pop,
        "post_hoc_seed_robustness": robustness,
        "post_hoc_seed_min_gap": float(min(r["gap"] for r in robustness)),
        "post_hoc_seed_max_perm_p": float(max(r["perm_pvalue"]
                                              for r in robustness)),
        "n_units": int(n),
        "n_treated": int(A.sum()),
        "n_control": int((1 - A).sum()),
        "mean_k_treated": float(k[A == 1].mean()),
        "mean_k_control": float(k[A == 0].mean()),
        "gap_control_minus_treated": gap,
        "perm_pvalue": b2_p,
        "B1_max_abs_k_minus_knob": b1,
        "B3_matched_covariate_rows": int(len(matched)),
        "B3_max_abs_conditional_mean_diff": b3,
        "ipw_redraw_mean": float(psi_ipw.mean()),
        "ipw_redraw_sd": float(psi_ipw.std(ddof=1)),
        "ipw_redraw_z": z,
        "B1_pass": bool(b1 == 0.0),
        "B2_pass": bool(gap >= 0.5 and b2_p <= 0.01),
        "B3_pass": bool(b3 == 0.0),
    }


def masking_algebra() -> dict:
    one, zero = F_ONE, F_ZERO
    e = (F_HALF, F_HALF, F_QUARTER)
    pX = one / 3
    z1 = sum(e[x] * pX for x in range(3))
    z0 = sum((one - e[x]) * pX for x in range(3))
    p1 = tuple(e[x] * pX / z1 for x in range(3))
    p0 = tuple((one - e[x]) * pX / z0 for x in range(3))
    q = (F_FOUR_15, F_FOUR_15, F_SEVEN_15)
    eA, eB, eC = (one, zero, zero), (zero, one, zero), (zero, zero, one)

    def mixture(p, qs):
        out = [zero, zero, zero]
        for x in range(3):
            for t in range(3):
                out[t] += p[x] * qs[x][t]
        return tuple(out)

    m1 = mixture(p1, (eA, eB, eC))
    m0 = mixture(p0, (eA, eB, q))
    tv = sum(abs(a - b) for a, b in zip(m1, m0)) / 2

    R = (F_HALF, zero, -F_HALF)
    levels = []
    for eps in (zero, one / 5, one / 3):
        t2 = tuple(eC[t] + eps * R[t] for t in range(3))
        q2 = tuple(q[t] + F_SEVEN_15 * eps * R[t] for t in range(3))
        m1e = mixture(p1, (eA, eB, t2))
        m0e = mixture(p0, (eA, eB, q2))
        max_diff = max(abs(a - b) for a, b in zip(m1e, m0e))
        levels.append({
            "eps": str(eps),
            "treated_stratum2": [str(v) for v in t2],
            "control_stratum2": [str(v) for v in q2],
            "max_abs_arm_marginal_diff": str(max_diff),
            "target_shift_vector": [str(F_EIGHT_45 * eps * R[t])
                                    for t in range(3)],
        })
    return {
        "arm1_type_frequencies": [str(v) for v in m1],
        "arm0_type_frequencies": [str(v) for v in m0],
        "tv_arm_type_frequencies": str(tv),
        "compensation_levels": levels,
        "C1_algebra_pass": bool(tv == 0),
        "D_algebra_pass": bool(
            all(entry["max_abs_arm_marginal_diff"] == "0"
                for entry in levels)
        ),
    }


def _type_of(oracle_entry) -> str:
    radii = np.asarray(oracle_entry["radii"], dtype=float)
    n_loops = int(oracle_entry["n_loops"])
    r = float(np.round(radii[0], 6))
    for name in ("A", "B", "C"):
        if n_loops == LOOPS[name] and abs(r - RADII[name]) < 1e-9:
            return name
    raise RuntimeError(f"unclassifiable oracle entry: {oracle_entry}")


def masking_monte_carlo() -> dict:
    sample = masking_stratum_sample(n_per_group=300, m=M_POINTS, noise=NOISE,
                                    seed=SEED)
    n = len(sample.clouds)
    labels = np.asarray(sample.A)
    counts = np.zeros((n, 3))
    for i in range(n):
        counts[i, ("A", "B", "C").index(_type_of(sample.oracle[i]))] = 1.0
    rng = np.random.default_rng(SEED + 3)
    p_mc = _perm_pvalue_tv(counts, labels, rng)
    arm1 = counts[labels == 1].sum(axis=0)
    arm0 = counts[labels == 0].sum(axis=0)
    arm1 = arm1 / arm1.sum()
    arm0 = arm0 / arm0.sum()
    tv_mc = 0.5 * float(np.abs(arm1 - arm0).sum())

    # Dynamic witness clouds: S clouds per type, alpha filtration, stacked
    # (H0, H1) silhouettes on the Phase-2 grid and fixed-grid measures.
    s_total = N_BATCH * N_PER_BATCH
    diagrams_by_type = {"A": [], "B": [], "C": []}
    sil_by_type = {"A": [], "B": [], "C": []}
    for name in ("A", "B", "C"):
        local = np.random.default_rng(SEED + 10 + ("ABC".index(name)))
        for j in range(s_total):
            cloud = loops_cloud(M_POINTS, LOOPS[name], radius=RADII[name],
                                noise=NOISE, rng=local)
            diags = compute_diagrams(cloud, homology_dims=(0, 1))
            diagrams_by_type[name].append(diags)
            sil_by_type[name].append(
                silhouette(diags, interval=INTERVAL, r=SIL_R,
                           resolution=RESOLUTION))
    all_diags = (diagrams_by_type["A"] + diagrams_by_type["B"]
                 + diagrams_by_type["C"])
    index = {"A": slice(0, s_total), "B": slice(s_total, 2 * s_total),
             "C": slice(2 * s_total, 3 * s_total)}
    feats0 = measure_features(all_diags, interval=INTERVAL, n_bins=N_BINS,
                              weight_power=WEIGHT_POWER, homology_dim=0)
    feats1 = measure_features(all_diags, interval=INTERVAL, n_bins=N_BINS,
                              weight_power=WEIGHT_POWER, homology_dim=1)
    feats = np.hstack([feats0, feats1])

    def contrast(values):
        c = values[index["C"]].mean(axis=0)
        a = values[index["A"]].mean(axis=0)
        b = values[index["B"]].mean(axis=0)
        return MASK_FACTOR * (c - 0.5 * (a + b))

    sil_all = np.stack([sil_by_type[name][j] for name in ("A", "B", "C")
                        for j in range(s_total)])
    sil_contrast = contrast(sil_all)
    feat_contrast = contrast(feats)

    batch_stats = []
    batch_l1 = []
    for bb in range(N_BATCH):
        lo, hi = bb * N_PER_BATCH, (bb + 1) * N_PER_BATCH
        s_bb = []
        f_bb = []
        for name in ("A", "B", "C"):
            idx = index[name]
            s_bb.append(sil_all[idx][lo:hi].mean(axis=0))
            f_bb.append(feats[idx][lo:hi].mean(axis=0))
        c_s = MASK_FACTOR * (s_bb[2] - 0.5 * (s_bb[0] + s_bb[1]))
        c_f = MASK_FACTOR * (f_bb[2] - 0.5 * (f_bb[0] + f_bb[1]))
        batch_stats.append(float(np.max(np.abs(c_s))))
        batch_l1.append(float(np.abs(c_f).sum()))
    batch_stats = np.asarray(batch_stats)
    batch_l1 = np.asarray(batch_l1)
    sup_sil = float(np.max(np.abs(sil_contrast)))
    l1_feat = float(np.abs(feat_contrast).sum())
    max_feat = float(np.max(np.abs(feat_contrast)))
    return {
        "mc_type_counts_arm1": arm1.tolist(),
        "mc_type_counts_arm0": arm0.tolist(),
        "mc_tv": tv_mc,
        "mc_perm_pvalue": p_mc,
        "s_clouds_per_type": s_total,
        "sup_silhouette_contrast": sup_sil,
        "batch_sup_mean": float(batch_stats.mean()),
        "batch_sup_se": float(batch_stats.std(ddof=1) / np.sqrt(N_BATCH)),
        "post_hoc_batch_sup_values": [float(v) for v in batch_stats],
        "post_hoc_batch_sup_all_positive": bool(np.all(batch_stats > 0.0)),
        "feat_contrast_l1": l1_feat,
        "feat_contrast_max_abs": max_feat,
        "batch_l1_mean": float(batch_l1.mean()),
        "batch_l1_se": float(batch_l1.std(ddof=1) / np.sqrt(N_BATCH)),
        "C1_mc_pass": bool(p_mc >= 0.05),
        "C2_pass": bool(sup_sil >= 0.5
                        and sup_sil >= 5.0 * float(batch_stats.std(ddof=1)
                                                  / np.sqrt(N_BATCH))),
        "C3_pass": bool(l1_feat >= 0.5 and max_feat >= 0.05),
    }


def main() -> dict:
    t0 = time.time()
    anchors = check_phase2_anchors()
    harness = harness_witness()
    masking = masking_algebra()
    masking.update(masking_monte_carlo())
    passes = {
        "A1": anchors["A1_pass"],
        "A2": anchors["A2_pass"],
        "A3": anchors["A3_pass"],
        "B1": harness["B1_pass"],
        "B2": harness["B2_pass"],
        "B3": harness["B3_pass"],
        "C1": bool(masking["C1_algebra_pass"] and masking["C1_mc_pass"]),
        "C2": masking["C2_pass"],
        "C3": masking["C3_pass"],
        "D": masking["D_algebra_pass"],
    }
    out = {
        "script": "scripts/theory_dev/t1_target_relative_witness.py",
        "seed": SEED,
        "alpha": ALPHA,
        "phase2_anchors": anchors,
        "harness_witness": harness,
        "masking_witness": masking,
        "criteria_pass": passes,
        "all_pass": bool(all(passes.values())),
        "wall_seconds": None,
    }
    out["wall_seconds"] = round(time.time() - t0, 2)
    path = os.path.join(OUT, "t1_target_relative_witness.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({"criteria_pass": passes, "all_pass": out["all_pass"],
                      "wall_seconds": out["wall_seconds"]}, indent=1))
    return out


if __name__ == "__main__":
    main()
