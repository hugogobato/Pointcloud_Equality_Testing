# Phase 8 bake-off report (replicated-cloud panel, C4 empirical)

Status: RESULTS COMPLETE on the coarse fleet (100 reps x n in {50,100,200});
full 500-rep stage in flight on 4 decision-relevant cells; n=500 coarse
deferred with measured cost (see Fleet box). Gate recommendation in §7.

## 0. Scope

Replicated-cloud panel ONLY (one cloud = one unit). The single-cloud panel
(task 8.2: SC-A/SC-B/SC-C, subgroup/heterogeneity/BLP columns) is OUT OF
SCOPE: Phase 5 (single-cloud) and Phase 6 (conditional/heterogeneity) are
removed, so any cell needing that machinery is skipped and the skip is
logged per replication (`skip_single_cloud` reads "single-cloud panel out of
scope (Phase 5/6 machinery removed)"). No SC-A/SC-B/SC-C numbers appear
anywhere in this report. Krebs-Rademacher runs on dispersion-shift + null
cells only (its dispersion null is blind to pure location by construction,
eq. 1.7); elsewhere `p_krebs_rademacher` is NaN with `skip_krebs` logged.

## 1. Methods

Grid: {rt, mmd, han, strand, moon_lazar, frechet_anova, krebs_rademacher
(selective), dr_mult, dr_perm, dist_maxt, dist_mmd, dr_equiv (null cells),
dist_l1 (w1det diagnostic)} x {randomized (λ=0), mild (λ=0.5), strong (λ=1.0)}
x {null, mean, dispersion, w1split, w1det, rare} x n={50,100,200} coarse
(+w1det n=500 pilot). Two-stage: 100 reps coarse everywhere at n<=200,
500 reps full on decision-relevant cells only.

Nulls under test (printed in every table header; no cross-null ranking
without the header): competitors (rt, mmd, han, strand, moon_lazar,
frechet_anova) test H0^cond; dr_mult/dr_perm test H0^out (primary);
dist_maxt tests H0^dist-grid (production Phase 4.5 calibration); dist_mmd
tests H0^dist (universal-kernel secondary); krebs_rademacher tests
H0^disp/equiv; dr_equiv tests H0^equiv (the non-equivalence null, so
rejection = declaring equivalence); dist_l1 is the Phase-4 L1 stratified
permutation diagnostic on the deterministic witness only.

DGPs (all reuse tda2s.dgp; nothing reinvented): null/mean via CloudSampleDGP
(group_effect 0/1, propensity expit(1.5λXβ), Phase 2 recipe); dispersion via
loop-count mixture {1,3} w.p. 1/2 vs fixed 2 (mean-matched count, summary
mean not exact, so this is a dispersion-shift *alternative*, not a pure weak
null); w1split via stochastic split_cluster_cloud 2-vs-3 blobs at 36
points/arm (Phase 4.5 power recipe; H0^out FALSE here, a gross-topology power
check, not the separation witness); w1det via deterministic 18-gon/12-gon
2-vs-3 split with the TAU=0.3 persistence filter (Phase 4 recipe; H0^out TRUE
realization-by-realization, H0^dist FALSE); rare via 15% extra r=0.35 loop
(loops_cloud). Clouds: m=60 points, alpha filtration, H0+H1. Silhouettes on
one fixed global grid ((0,2), r=3, res 50); dist-level on the frozen Phase
4.5 grid ((0,2), 32 bins, R=3, H0 only).

DR path: tda2s.tests.dr_outcome.fit_dr over tcda_uq cross_fit (no
reimplemented AIPW); propensity LogisticRegression (correctly specified for
these DGPs), n_basis=5, 2 folds; multiplier (399 draws, primary) + frozen
stratified permutation (199). Dist-level: 5-fold LR + per-arm linreg
cross-fit (Phase 4.5 confounded recipe) → dist_multiplier_test (1999 draws,
frozen contract) + universal-kernel MMD permutation on the cached kernel
matrix (near-diagonal EPS=0.1 pre-filter, permutation-exact for any fixed
embedding). Competitor perms: 199. RT matrix: exactly-equal short-circuit
(CGAL degenerate-path hazard, Phase 1 exit note) + EPS=0.1 + gudhi approx
0.01 (Phase 2 values); empty post-filter diagrams kept (loop-count signal
lives there). Diagrams computed once per replication and cached in-process
keyed by (cloud sha1, filtration, params); all calibrations permute cloud
labels only (never recompute PH in a loop). Workers capped at 16 total. Seed discipline: every replication seed derives
from `_seed` (integer arithmetic, process-independent). Seed fix 2026-09-10:
the mmd/han/strand/moon_lazar competitor offsets used `hash(name) % 997`,
and Python string hashing is randomized per process, so those four columns
were valid Monte Carlo but not byte-reproducible across workers (54-70 of
100 p-values differed on identical rep seeds; every fixed-offset column was
byte-identical). Replaced by fixed offsets (`_COMP_SEED_OFFSET`,
frechet_anova keeps its historic +11); regression test
`test_competitor_seeds_deterministic` forbids `hash(` in `_one_rep`.
Coarse/robust/smoke shards predate the fix for those four columns only;
post-fix cross-process check (PYTHONHASHSEED=1 vs 99, all 13 p-columns)
is byte-identical.

Fleet box (honest accounting). Coarse: 45 DGP x imbalance x n cells at
n in {50,100,200} x 100 reps + 7 w1det cells (incl. one n=500 pilot) + 4
robustness cells x 100 reps + 4 full cells x 500 reps = 64 shards, 7608
rows, 0 failed reps.
Wall: ~6h coarse + ~29min robust at 16 workers (phase8_coarse.log), plus the
w1det stage2. Full (500 reps): COMPLETE on strongest-imbalance mean-shift
(mean/strong n=50, n=100) + W1 columns (w1split/strong n=100, n=200), 8
workers each (16 total); 0 failed reps. Mean/strong n=200 full deferred as a
>2h single cell (1825s per 100 reps measured → ~2.5h per 500). Coarse n=500 for the
loops_cloud families deferred: measured 284s/rep sequential at n=500 (Han
74s + RT 31s + MMD/dist_mmd ~20s each) → ~30-45min per 100-rep cell at 16
workers x 15 cells ≈ 8-11h; operating guidance below is stated for n<=200
with the w1det n=500 pilot as the only n=500 evidence. Every rate below
carries reps + MC SE; failed runs are retained as ok=False rows (none
occurred on coarse).

## 2. Tables (100 reps coarse; MC SE in parentheses; α=0.05)

### Table 8.0. Size: null family (H0^cond cols vs H0^out vs H0^dist-grid vs H0^dist vs H0^disp/equiv; dr_equiv rejects = declares equivalence)

| n | imbalance | rt [cond] | mmd [cond] | han [cond] | strand [cond] | moon [cond] | frechet [cond] | krebs [disp] | dr_mult [out] | dr_perm [out] | dist_maxt [dgrid] | dist_mmd [dist] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | randomized | 0.020 (0.014) | 0.040 (0.020) | 0.030 (0.017) | 0.030 (0.017) | 0.030 (0.017) | 0.020 (0.014) | 0.020 (0.014) | 0.040 (0.020) | 0.040 (0.020) | 0.020 (0.014) | 0.010 (0.010) |
| 50 | mild | 0.090 (0.029) | 0.140 (0.035) | 0.120 (0.032) | 0.110 (0.031) | 0.050 (0.022) | 0.090 (0.029) | 0.060 (0.024) | 0.070 (0.026) | 0.030 (0.017) | 0.060 (0.024) | 0.060 (0.024) |
| 50 | strong | 0.250 (0.043) | 0.430 (0.050) | 0.390 (0.049) | 0.470 (0.050) | 0.180 (0.038) | 0.160 (0.037) | 0.070 (0.026) | 0.040 (0.020) | 0.020 (0.014) | 0.070 (0.026) | 0.020 (0.014) |
| 100 | strong | 0.510 (0.050) | 0.740 (0.044) | 0.740 (0.044) | 0.830 (0.038) | 0.420 (0.049) | 0.290 (0.045) | 0.120 (0.032) | 0.120 (0.032) | 0.040 (0.020) | 0.020 (0.014) | 0.030 (0.017) |
| 200 | randomized | 0.040 (0.020) | 0.020 (0.014) | 0.030 (0.017) | 0.050 (0.022) | 0.040 (0.020) | 0.040 (0.020) | 0.020 (0.014) | 0.070 (0.026) | 0.060 (0.024) | 0.030 (0.017) | 0.050 (0.022) |
| 200 | mild | 0.390 (0.049) | 0.590 (0.049) | 0.570 (0.050) | 0.630 (0.048) | 0.360 (0.048) | 0.100 (0.030) | 0.130 (0.034) | 0.070 (0.026) | 0.080 (0.027) | 0.060 (0.024) | 0.010 (0.010) |
| 200 | strong | 0.840 (0.037) | 0.950 (0.022) | 0.950 (0.022) | 0.940 (0.024) | 0.700 (0.046) | 0.430 (0.050) | 0.180 (0.038) | 0.030 (0.017) | 0.000 (0.000) | 0.060 (0.024) | 0.020 (0.014) |

Reading: under randomization every column holds size. Under mild imbalance
at n=200 the six H0^cond columns already reject at 0.10-0.63 while the
H0^out/H0^dist-grid columns stay at 0.01-0.08. Under strong imbalance at
n=200 the H0^cond columns reject at 0.43-0.95 (five of six at >=0.70) while
dr_mult/dr_perm/dist_maxt/dist_mmd hold 0.00-0.06. This reproduces the Phase
2 C1 false-positive finding inside the bake-off DGP. Two flags: dr_mult at
strong n=100 reads 0.120 (0.032), ~1.2 SE above the 0.08 band edge,
consistent with the Phase 4.5 small-n strong-confounding wobble; dr_perm at
strong n=200 reads 0.000 (conservative, frozen-strata discreteness), not
anti-conservative. dr_equiv (equivalence-declaration rate under the null):
randomized n=200 0.970 (0.017), mild n=200 0.890 (0.031), strong n=200 0.650
(0.048); at n=50 it is 0.000-0.010 everywhere (margin 0.25 unreachable at
n=50). Equivalence needs n>=200 and degrades under strong confounding.

### Table 8.1. Power: mean-shift (+1 loop; H0^cond vs H0^out vs H0^dist-grid vs H0^dist)

| n | imbalance | rt | mmd | han | strand | moon | frechet | dr_mult [out] | dr_perm [out] | dist_maxt [dgrid] | dist_mmd [dist] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | randomized | 0.800 (0.040) | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 0.940 (0.024) | 0.970 (0.017) | 0.110 (0.031) | 0.090 (0.029) | 0.970 (0.017) | 0.230 (0.042) |
| 50 | strong | 0.370 (0.048) | 0.800 (0.040) | 0.800 (0.040) | 0.920 (0.027) | 0.480 (0.050) | 0.470 (0.050) | 0.130 (0.034) | 0.070 (0.026) | 0.830 (0.038) | 0.440 (0.050) |
| 100 | randomized | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 0.450 (0.050) | 0.280 (0.045) | 1.000 (0.000) | 0.310 (0.046) |
| 100 | mild | 0.960 (0.020) | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 0.980 (0.014) | 0.960 (0.020) | 0.310 (0.046) | 0.260 (0.044) | 1.000 (0.000) | 0.520 (0.050) |
| 100 | strong | 0.670 (0.047) | 0.980 (0.014) | 0.970 (0.017) | 1.000 (0.000) | 0.810 (0.039) | 0.800 (0.040) | 0.340 (0.047) | 0.120 (0.032) | 0.980 (0.014) | 0.590 (0.049) |
| 200 | randomized | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 0.990 (0.010) | 0.930 (0.026) | 1.000 (0.000) | 0.570 (0.050) |
| 200 | mild | 0.980 (0.014) | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 0.910 (0.029) | 0.770 (0.042) | 1.000 (0.000) | 0.800 (0.040) |
| 200 | strong | 0.980 (0.014) | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 0.950 (0.022) | 0.760 (0.043) | 0.360 (0.048) | 1.000 (0.000) | 0.710 (0.045) |

Reading: the DR outcome test is late to the mean-shift signal at small n
and under confounding (see LOSSES in §8.3), catches up at n=200 randomized
(0.99/0.93), and the distribution-level max-t column fires at >=0.83
everywhere including n=50. Competitor "power" under imbalance is not
like-for-like: Table 8.0 shows they reject 0.39-0.95 on the null under the
same imbalance, so their mean-column rates mix signal with confounding.

### Table 8.2. Power: dispersion-shift (loop-count mixture; H0^cond vs H0^out vs H0^dist-grid vs H0^dist vs H0^disp/equiv for KR)

| n | imbalance | rt | mmd | han | strand | moon | frechet | krebs [disp] | dr_mult [out] | dr_perm [out] | dist_maxt [dgrid] | dist_mmd [dist] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | randomized | 1.000 (0.000) | 0.720 (0.045) | 1.000 (0.000) | 0.090 (0.029) | 0.950 (0.022) | 1.000 (0.000) | 0.930 (0.026) | 0.970 (0.017) | 0.910 (0.029) | 1.000 (0.000) | 0.580 (0.049) |
| 50 | strong | 1.000 (0.000) | 0.780 (0.041) | 1.000 (0.000) | 0.040 (0.020) | 0.930 (0.026) | 1.000 (0.000) | 0.970 (0.017) | 0.940 (0.024) | 0.740 (0.044) | 0.980 (0.014) | 0.410 (0.049) |
| 200 | randomized | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 0.230 (0.042) | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 0.900 (0.030) |
| 200 | strong | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 0.240 (0.043) | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 0.990 (0.010) | 0.940 (0.024) | 1.000 (0.000) | 0.550 (0.050) |

Reading: KR holds its turf (0.93-1.00) as designed. STRAND is weak on this
mixture everywhere (0.04-0.24, a named loss). dist_mmd is the soft column
here (0.41-0.90), trailing dist_maxt by ~0.4.

### Table 8.3a. Power: stochastic cluster splitting w1split (H0^out FALSE here; gross-topology power check)

| n | imbalance | rt | mmd | han | strand | moon | frechet | dr_mult [out] | dr_perm [out] | dist_maxt [dgrid] | dist_mmd [dist] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | randomized | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 0.970 (0.017) | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) |
| 200 | mild | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 0.990 (0.010) | 1.000 (0.000) | 1.000 (0.000) | 0.990 (0.010) | 1.000 (0.000) | 1.000 (0.000) |
| 200 | strong | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 0.960 (0.020) | 0.760 (0.043) | 1.000 (0.000) | 1.000 (0.000) |

(Every w1split coarse cell saturates at 0.96-1.00 except dr_perm at strong
n=200 with 0.760 (0.043). Full 500-rep tightenings in flight.)

### Table 8.3b. Separation: deterministic witness w1det (H0^out TRUE, H0^dist FALSE; H0^cond vs H0^out vs H0^dist vs dist-L1 diagnostic)

| n | imbalance | rt [cond] | mmd [cond] | han [cond] | strand [cond] | moon [cond] | frechet [cond] | dr_mult [out] | dr_perm [out] | dist_mmd [dist] | dist_l1 [diag] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 200 | randomized | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 0.000 (0.000) | 1.000 (0.000) | 0.570 (0.050) | 0.000 (0.000) | 0.040 (0.020) | 1.000 (0.000) | 1.000 (0.000) |
| 200 | mild | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 0.000 (0.000) | 1.000 (0.000) | 0.540 (0.050) | 0.000 (0.000) | 0.040 (0.020) | 1.000 (0.000) | 1.000 (0.000) |
| 500 | randomized | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 0.000 (0.000) | 1.000 (0.000) | 0.590 (0.049) | 0.000 (0.000) | 0.010 (0.010) | 1.000 (0.000) | 1.000 (0.000) |

Reading: the separation the paper needs. With H0^out exactly true,
dr_perm holds 0.010-0.040 across imbalance and n (the H0^out size line);
dr_mult sits at 0.000 (the documented degenerate-conservative multiplier on
the realization-invariant outcome, Phase 4 note, not a size claim). The
dist columns have power 1.000 at every n including 500. The H0^cond columns
fire at 1.000 (RT/MMD/Han/Moon) because the unit-level laws differ; STRAND
is blind here (0.000, a competitor loss that cuts the other way); Frechet
sees about half (0.54-0.59). dist_maxt is not defined on w1det (AIPW scores
exactly constant → studentization 0/0, correctly refused as a
deterministic failure and recorded per-test, never crashing the rep);
dist_l1 is the defined dist readout on this witness with the Phase 4.5 §4
confounding caveat.

### Table 8.4. Power: rare-feature (15% extra small loop; H0^cond vs H0^out vs H0^dist-grid vs H0^dist)

| n | imbalance | rt | mmd | han | strand | moon | frechet | dr_mult [out] | dr_perm [out] | dist_maxt [dgrid] | dist_mmd [dist] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | strong | 0.230 (0.042) | 0.600 (0.049) | 0.620 (0.049) | 0.650 (0.048) | 0.310 (0.046) | 0.280 (0.045) | 0.130 (0.034) | 0.080 (0.027) | 0.180 (0.038) | 0.030 (0.017) |
| 100 | randomized | 0.200 (0.040) | 0.090 (0.029) | 0.120 (0.032) | 0.070 (0.026) | 0.100 (0.030) | 0.100 (0.030) | 0.280 (0.045) | 0.280 (0.045) | 0.340 (0.047) | 0.020 (0.014) |
| 200 | randomized | 0.400 (0.049) | 0.170 (0.038) | 0.280 (0.045) | 0.110 (0.031) | 0.160 (0.037) | 0.030 (0.017) | 0.530 (0.050) | 0.480 (0.050) | 0.940 (0.024) | 0.060 (0.024) |
| 200 | mild | 0.570 (0.050) | 0.840 (0.037) | 0.890 (0.031) | 0.830 (0.038) | 0.650 (0.048) | 0.180 (0.038) | 0.390 (0.049) | 0.380 (0.049) | 0.900 (0.030) | 0.030 (0.017) |
| 200 | strong | 0.920 (0.027) | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 0.960 (0.020) | 0.560 (0.050) | 0.220 (0.041) | 0.110 (0.031) | 0.770 (0.042) | 0.050 (0.022) |

Reading: the rare feature is hard for everyone at n=50 and at randomized
n=100 (all columns 0.02-0.34; ours lead slightly). At randomized n=200 the
dist_maxt column (0.940) and DR (0.48-0.53) beat every H0^cond column (best
RT 0.40); Han/MMD/STRAND sit at 0.11-0.28 on their non-turf subtle signal.
Under strong confounding the H0^cond columns read 0.92-1.00 but Table 8.0
shows they reject 0.70-0.95 on the null under the same imbalance, so that
"power" is not evidence of topological sensitivity. dist_mmd is blind to
the rare feature everywhere (0.02-0.06, a named secondary-method loss).

## 2b. Full stage (500 reps, strongest-imbalance mean-shift + W1 columns)

| cell | rt | mmd | han | strand | moon | frechet | dr_mult [out] | dr_perm [out] | dist_maxt [dgrid] | dist_mmd [dist] |
|---|---|---|---|---|---|---|---|---|---|---|
| mean/strong n=50 | 0.366 (0.022) | 0.806 (0.018) | 0.802 (0.018) | 0.900 (0.013) | 0.518 (0.022) | 0.498 (0.022) | 0.212 (0.018) | 0.120 (0.015) | 0.908 (0.013) | 0.382 (0.022) |
| w1split/strong n=100 | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 0.974 (0.007) | 1.000 (0.000) | 0.926 (0.012) | 0.692 (0.021) | 0.978 (0.007) | 0.994 (0.003) |
| w1split/strong n=200 | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 0.972 (0.007) | 0.746 (0.019) | 0.998 (0.002) | 1.000 (0.000) |
| mean/strong n=100 | 0.752 (0.019) | 0.986 (0.005) | 0.992 (0.004) | 0.996 (0.003) | 0.868 (0.015) | 0.818 (0.017) | 0.332 (0.021) | 0.144 (0.016) | 0.994 (0.003) | 0.546 (0.022) |

Full-vs-coarse consistency: mean/strong n=50 full dr_mult 0.212 (0.018) vs
coarse 0.130 (0.034) (coarse 95% interval covers 0.06-0.20; wider than the
full estimate by honest Monte Carlo noise, same qualitative loss to STRAND
0.900 vs 0.920). mean/strong n=100 full dr_mult 0.332 (0.021) vs coarse
0.340 (0.047), dead-on with SE halved; the STRAND/Han loss (0.996/0.992 vs
0.332) is now tight. w1split/strong n=100 full dr_mult 0.926 (0.012) vs
coarse 1.000, n=200 full 0.972 (0.007) vs coarse 0.960 (0.020): saturation
confirmed within noise; dr_perm stays the weaker calibration (0.692/0.746
vs 0.926/0.972).
| w1split/strong n=200 | in flight | | | | | | | | | |

Full-vs-coarse consistency: mean/strong n=50 full dr_mult 0.212 (0.018) vs
coarse 0.130 (0.034) (coarse 95% interval covers 0.06-0.20; wider than the
full estimate by honest Monte Carlo noise, same qualitative loss to STRAND
0.900 vs 0.920). w1split/strong n=100 full dr_mult 0.926 (0.012) vs coarse
1.000 (saturation confirmed within noise).

## 3. Regime analysis WITH LOSSES (8.3)

Wins first, briefly. Under any imbalance the DR outcome (multiplier) and
the dist-level max-t hold size while all six H0^cond columns break
(Table 8.0, strong n=200: 0.03/0.00/0.06 vs 0.43-0.95); that is C1 inside
the bake-off. The dist-level max-t column is the most uniformly powerful
new column (1.000 on w1split/w1det/dispersion, 0.77-1.00 on mean/rare at
n=200). On the rare-randomized subtle signal at n=200, ours win outright
(dist_maxt 0.94, DR 0.48-0.53 vs best competitor RT 0.40).

Losses (the credible part).

1. Small-n randomized mean-shift: RT and STRAND win on simplicity.
   Mean/randomized n=50: STRAND/MMD/Han 1.000, RT 0.800 vs dr_mult 0.110
   (0.031), dr_perm 0.090 (0.029). Mean/randomized n=100: competitors 1.000
   vs dr 0.450/0.280. The cross-fitted AIPW pays a variance tax the
   unadjusted permutation tests do not pay when there is nothing to adjust
   for. The DR outcome test should not be the default at n=50 randomized
   for a gross mean shift.
2. Confounded mean-shift at n<=100: STRAND/Han/MMD win big. Mean/strong
   n=50: STRAND 0.920 vs dr 0.130/0.070. Mean/strong n=100: STRAND 1.000,
   Han/MMD 0.97-0.98 vs dr 0.340/0.120. At n=200 the gap narrows for the
   multiplier (0.760) but dr_perm stays at 0.360 (0.048) against 0.95-1.00.
   The frozen-strata permutation is the weaker DR calibration under strong
   confounding; the multiplier is the primary for a reason.
3. Han's turf (rare-feature under confounding) and intensity signals: Han
   wins. Rare/strong n=200: Han/MMD/STRAND 1.000 vs dr 0.220/0.110 (with
   the Table 8.0 confounding caveat on the competitor rates). Rare/mild
   n=200: Han 0.890 vs dr 0.390/0.380, a cleaner Han win. Do not claim to
   beat the minimax-optimal intensity test on a faint intensity signal.
4. STRAND losses that cut our way (reported symmetrically): STRAND is weak
   on the dispersion mixture (0.04-0.24, Table 8.2) and blind on the
   deterministic split (0.000, Table 8.3b) where RT/MMD/Han fire at 1.000.
5. Secondary-method losses: dist_mmd trails dist_maxt by ~0.3-0.5 on
   mean/dispersion and is blind on rare (0.02-0.06); it is a "did the mean
   integrate something away" check, not a primary. dr_perm trails dr_mult
   on every confounded power cell (e.g. mean/strong n=200: 0.360 vs 0.760)
   and should be reported as the conservative second calibration, not
   pooled with the multiplier.

## 4. Cost table (8.4): wall-time per test at matched n + peak RSS

Means over coarse reps (in-worker timers; 16-worker fleet). Single-rep
probe at n=200/mean/randomized for the RSS high-water mark.

| timer (per rep) | n=50 | n=100 | n=200 | probe n=200 (s) |
|---|---|---|---|---|
| t_ph (diagrams, once) | 0.29 | 0.59 | 1.09 | 0.45 |
| t_rt (H0^cond) | 0.79 | 2.97 | 10.86 | 4.51 |
| t_mmd (H0^cond) | 0.87 | 2.21 | 5.94 | 3.61 |
| t_han (H0^cond) | 1.00 | 3.48 | 17.54 | 11.07 |
| t_strand (H0^cond) | 1.04 | 1.78 | 2.94 | 1.02 |
| t_moon_lazar (H0^cond) | 0.11 | 0.20 | 0.39 | 0.12 |
| t_frechet_anova (H0^cond) | 0.37 | 0.50 | 0.75 | 0.19 |
| t_krebs (H0^disp, selective) | 1.87 | 6.47 | 24.16 | — |
| t_dr_fit (cross-fit, once) | 11.48 | 11.69 | 11.22 | 5.31 |
| t_dr_mult (H0^out calib) | 0.02 | 0.03 | 0.04 | 0.01 |
| t_dr_perm (H0^out calib) | 0.04 | 0.04 | 0.06 | 0.02 |
| t_dist_fit (cross-fit, once) | 0.27 | 0.46 | 0.68 | 0.23 |
| t_dist_maxt (H0^dgrid calib) | 0.28 | 0.34 | 0.66 | 0.02 |
| t_dist_mmd (H0^dist calib) | 2.52 | 8.62 | 48.56 | 7.38 |
| wall_rep total | — | — | — | 48.95 |

Peak RSS (single n=200 rep, sequential, same process): base 33 MB →
1548 MB high-water (delta ~1515 MB; includes interpreter + GUDHI + sklearn
+ cached kernel blocks, not a worker-pool artifact). At matched n=200 the
bill is dominated by Han (17.5s), RT (10.9s), dist_mmd (48.6s mean; the
vectorized universal kernel is O(n^2) pairs x diagram points and fans out
on high-cardinality reps), and the one-time DR cross-fit (~11.4s, flat in
n over 50-200 because the silhouette-regression cost dominates the sample
size). All calibrations after the fits are milliseconds (no PH, no refit
inside any loop). At matched inferential target the DR multiplier adds
~0.04s over its fit; the 11s fit is the price of the confounding
adjustment, and Table 8.0 is what it buys. dist_mmd at n=200 is the cost
outlier and a further reason it stays secondary; n=500 loops_cloud cells
were costed out of the coarse fleet on this evidence (284s/rep sequential:
Han 74s, RT 31s, MMD/dist_mmd ~20s each).

## 5. Robustness panel (8.5, 100 reps, n=100, randomized unless noted)

| defect (all under the null except R4) | rt | mmd | han | strand | moon | frechet | dr_mult [out] | dr_perm [out] | dist_maxt [dgrid] | dist_mmd [dist] |
|---|---|---|---|---|---|---|---|---|---|---|
| R1 10% uniform outliers | 0.060 (0.024) | 0.090 (0.029) | 0.050 (0.022) | 0.050 (0.022) | 0.040 (0.020) | 0.060 (0.024) | 0.090 (0.029) | 0.040 (0.020) | 0.020 (0.014) | 0.040 (0.020) |
| R2 m=40 vs 160 by arm (cardinality confounded) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 [out] | 1.000 [out] | 1.000 [dgrid] | 1.000 [dist] |
| R3 heavy-tailed loop count {1 w.p. .8, 8 w.p. .2}, same law both arms | 0.020 (0.014) | 0.020 (0.014) | 0.030 (0.017) | 0.020 (0.014) | 0.020 (0.014) | 0.030 (0.017) | 0.030 (0.017) | 0.030 (0.017) | 0.000 (0.000) | 0.010 (0.010) |
| R4 both nuisances misspecified, confounded null (DummyClassifier + n_basis=2) | 0.510 (0.050) | 0.650 (0.048) | 0.640 (0.048) | 0.680 (0.047) | 0.380 (0.049) | 0.270 (0.044) | 0.080 (0.027) | 0.090 (0.029) | 0.050 (0.022) | 0.060 (0.024) |

Reading: R1/R3 hold size everywhere (no fragility to outliers or to
heavy-tailed cardinality when the law is shared). R2 rejects at 1.000 in
every column including ours: cloud cardinality is part of the diagram
signal (more points → more/sharper classes), so an arm-confounded m is a
confounded contrast, not power; equalise m or declare size part of the
treatment (Phase 1 exit note 5). R4 was built as a negative control and did
not break the DR columns (0.08/0.09 at n=100): the misspecification
(DummyClassifier propensity + 2 Fourier terms) is evidently too mild at
this n to separate DR from luck, so this is reported as an uninformative
control, not as DR evidence; the H0^cond columns reject at 0.27-0.68 on the
same confounded null. dr_equiv on R1/R3 declares equivalence at 0.91/0.99;
on R2 it is 0.000 (correctly refuses).

## 6. Decision table (8.6): "use X when …", from the sampling unit

| # | Your situation (start here) | Use | Null you are actually testing |
|---|---|---|---|
| 1 | Replicated clouds (one cloud = one unit), randomized groups, gross mean shift, n<=100 | STRAND or Han/MMD (simplest that fires); DR as sensitivity | H0^cond (then Table 8.1) |
| 2 | Replicated clouds + covariate imbalance suspected (propensity varies with X) | DR multiplier (primary) + dist-level max-t | H0^out / H0^dist-grid (Tables 8.0-8.1; competitors test the wrong null here) |
| 3 | Replicated clouds + question is the interventional law, not the mean (split/merge, multiplicity) | dist-level max-t (headline) + dist_mmd secondary | H0^dist-grid / H0^dist (Table 8.3b) |
| 4 | Replicated clouds + faint/diffuse intensity signal, randomized | Han et al. (minimax turf); do not expect DR to win | H0^cond (Table 8.4 rare/randomized) |
| 5 | Replicated clouds + dispersion/scale question | KR on its turf + RT/MMD alongside; STRAND will likely miss | H0^disp/equiv + H0^cond (Table 8.2) |
| 6 | Replicated clouds + need to certify sameness | DR equivalence (margin 0.25) at n>=200 randomized/mild; not at n=50, not under strong confounding | H0^equiv (§2, Table 8.0 note) |
| 7 | Exactly two clouds, no replication | None of the above; route to the single-cloud regime (OUT OF SCOPE here, Phase 5 machinery removed) | — (this bake-off does not cover it) |
| 8 | Cloud sizes differ by arm | Fix the design first (equalise m); no test survives R2 | — (Table R2) |

## 7. Can-claim / cannot-claim + gate

Can claim. (i) Under covariate imbalance the six H0^cond columns break
(strong n=200 null: 0.43-0.95) while dr_mult/dr_perm/dist_maxt hold
0.00-0.06: the bake-off reproduces C1. (ii) The distribution-level max-t is
the most uniformly powerful new column (1.00 on w1split/w1det/dispersion;
0.77-1.00 on mean/rare at n=200) with size 0.01-0.07 across all 9 null
cells. (iii) The deterministic separation holds under imbalance: H0^out
size 0.01-0.04 (dr_perm) with dist power 1.00 (Table 8.3b). (iv) Costs are
measured: calibrations are ms after one-time fits; Han/RT/dist_mmd scale
steeply in n.

Cannot claim. (i) No small-n DR power story: at n=50-100 the DR outcome
test loses to STRAND/Han/MMD on mean-shift (gaps up to 0.85) and to Han on
rare; dr_perm additionally trails dr_mult under strong confounding (0.36 vs
0.76 at mean/strong n=200). (ii) No uniform weak-null theorem: carries
forward Phase 4.5's qualification (strongest confounding supported at
n>=500; dr_mult strong-n=100 wobble 0.12 here is consistent). (iii) No R4
robustness credit: the both-misspecified control did not break DR, hence is
uninformative. (iv) No n=500 loops_cloud operating point: coarse n=500
deferred on measured cost; only the w1det n=500 pilot exists. (v) No
cross-null ranking: every comparison above names its nulls; competitor
"power" under imbalance mixes signal with size distortion.

Gate: READY-WITH-QUALIFICATIONS. The coarse fleet (100 reps, n=50/100/200,
0 failures, every cell reps + MC SE), the 4-cell 500-rep full stage (0
failures, decision-relevant SEs halved, qualitative verdicts unchanged),
the w1det separation, the cost probe, and the 4-cell robustness panel
support the §7 claims. Full n=500 loops_cloud coverage and a stronger R4
control remain future work and are not claimed.

## 8. Provenance of every file

- Driver: experiments/phase8_master_grid.py (contract CONTRACT; modes
  smoke/coarse/full/robust/aggregate/figure/cost; --workers capped at 16).
- Shards: results/phase8_shards/cell_*.parquet (one per cell, tagged
  smoke/coarse/full/robust; 64 shards at aggregation).
- Master: results/phase8_master.parquet + results/phase8_master.json
  (contract + null_of + per-(cell,test) reps/rate/mc_se + cost means).
- Figure: results/phase8_figure.png (coarse rejection-by-imbalance at n=200
  across families; H0^cond vs H0^out/H0^dist-grid distinguished in title).
- Tests: tests/test_phase8.py (schema + fixed-seed determinism + null
  headers + contract keys; 4 passed).
- Wall time: phase8_coarse.log (~6h coarse 42 cells + ~29min robust at 16
  workers) + phase8_stage2.log (w1det) + phase8_full_mean.log /
  phase8_full_w1split.log (full stage COMPLETE, 4 cells x 500 reps at 8+8
  workers, 0 failures).
