# Phase 4.5: Weak-null calibration of the distribution-level (C2) test

Status: **CLOSED WITH QUALIFICATION**. Tasks 4.5.1-4.5.7 are implemented.
The strict pre-registered size gate has one failure at confounded `r=1.0`,
`n=200` (0.096). A qualified operating recommendation is retained, with the
exception and its scope stated explicitly in section 6.

Fleet driver: `experiments/phase45_weak_null.py`; production code:
`tda2s/tests/dist_level.py`; unit tests: `tests/test_phase45.py`;
results: `results/phase45_shards/*.json`, `results/phase45_aggregate.json`,
`results/phase45_weak_null_figure.png`.

---

## 1. Task 4.5.1: the frozen calibration contract (locked)

Finite-grid target and statistic are frozen exactly as specified in the plan,
`RESEARCH_PLAN_P1_TwoSample.md` Phase 4.5:

- Grid: `N_BINS = 32` bins on the interval `(0, 2)`, coordinates
  (birth, mid = (b+d)/2) of the H0 persistence measure with weight
  `R = 3.0`; 1024 coordinates in total.
- Witness filter: only classes with persistence `> TAU = 0.3` survive;
  classes are rounded to 1e-6 so deterministic blob geometry lands exactly
  on bin edges.
- Scores: cross-fitted AIPW influence scores `phi_hat_i` (5 folds,
  per-arm linear outcome regressions, logistic propensity regression),
  or in-sample arm means with known propensity 1/2 where the design
  has no covariates.
- Variance: `sigma_hat_j^2 = (1/n) * sum_i (phi_ij - delta_hat_j)^2`
  (1/n normalization, not 1/(n-1)).
- Variance floor: `VARIANCE_FLOOR = 1e-8`; coordinates with
  `sigma_hat_j^2 <= VARIANCE_FLOOR` are dropped from the max over J and
  reported (`n_dropped_coordinates`). If every coordinate is dropped
  (all scores constant), the test raises a deterministic-failure error
  (`ValueError`), which the fleet records, never silently passes.
- Statistic: `T_n = max_{j in J} sqrt(n) |delta_hat_j| / sigma_hat_j`,
  null via Gaussian multipliers (`n_draws = 1999`, `alpha = 0.05`),
  Phipson-Smyth correction for p-value draws.
- Diagnostics recorded in every replication: Rademacher multiplier
  p-value, CCK-style Gaussian-path max-t p-value and max ECDF gap,
  frozen-label studentized permutation p-value (sharp-null diagnostic
  only, see section 4).
- Scope: H0dist,grid `delta = E{m1(X)} - E{m0(X)} = 0`, the weak
  expected-measure null. The sharp null (identical unit-level laws
  given X) is a strict subset. Exchangeability is not assumed.

## 2. Task 4.5.2: the finite-dimensional AIPW linearization (verified)

For the fixed-grid vector V_i in R^q with target
delta = E{m1(X)} - E{m0(X)}, m_a(x) = E[V(a) | X = x], the score

    phi_i = (m1_hat - m0_hat)(X_i)
            + (A_i / e_hat(X_i)) (V_i - m1_hat(X_i))
            - ((1 - A_i) / (1 - e_hat(X_i))) (V_i - m0_hat(X_i))

satisfies E[phi_i] = delta at the oracle nuisances under consistency,
conditional exchangeability, positivity, and integrability, and the sample
average of the scores equals the AIPW estimate of delta (unit test
`test_score_mean_is_the_aipw_contrast`, exact to 1e-12). This is the
fixed-grid specialization of the Banach-valued double-robust
representation; see the Souto-Diamantis adaptation in the Phase 4
prototype (`docs/phase4_tdist_decision.md`) and the classic source
[1, 2].

The expectation check is coordinatewise: conditional on X, the treated
residual has mean zero because E[A/e(X) | X] = 1 and E[V(1) | A=1,X] =
m_1(X), and the control residual is analogous. Cross-fitting makes the
estimated nuisance functions fixed with respect to each held-out residual;
it does not make the finite-sample estimated score exactly unbiased.

The W2' and power designs use the known propensity 1/2 only because their
design has no covariates and balanced labels. The confounded and sharp designs
estimate the propensity and outcome regressions by cross-fitting. The fleet
does not establish robustness for arbitrary nuisance learners or convergence
rates.

Oracle-score exactness: with in-sample arm-mean nuisances on W1 clouds,
`dist_scores(fit).mean(0)` equals the sample feature contrast
`t1 - t0` exactly, and the effect concentrates at the merge-scale bin
(`test_oracle_scores_are_exact_on_w1_clouds`).

## 3. Task 4.5.3: the two-sample collapse identity (verified)

For balanced arms, known propensity e = 1/2, and in-sample arm-mean
nuisances, the AIPW score reduces to the centered features
`phi_i = (V_i - mean_V_arm) * (2A_i - 1)`-style contrast and the
studentized max-t statistic collapses exactly to the ordinary pooled
two-sample t-test on a single active coordinate:

    T_n = sqrt(n / (n - 2)) * |t_pooled|

verified numerically to 1e-9 relative error at n_arm in {20, 50}
(`test_two_sample_collapse_identity_exact`). In a 60-replication check, the
multiplier p-value differs from the scipy two-sample t-test p-value by a mean
absolute 0.0137, with six rejections for each procedure
(`test_two_sample_collapse_multiplier_matches_t_test`). This is Monte Carlo
agreement, not an identity between the two p-value algorithms.
This identity requires the AIPW version with in-sample arm-mean
nuisances: the IPW plug-in form has a different score variance and does
not collapse.

## 4. Task 4.5.4: the permutation audit (diagnostic only)

For a balanced split with fixed scores `y_i`, write `s_i = 2L_i - 1` for a
uniform fixed-count relabeling and let `S^2 = (n-1)^(-1) sum_i (y_i - ybar)^2`
be the finite-population variance around the pooled mean. Then
the scalar mean contrast is
`D* = (2/n) sum_i s_i y_i`, and the finite-population calculation gives
`Var_perm(D*) = 4 S^2 / n`. This identity is for one scalar linear contrast,
not for the variance of the maximum over coordinates. It also describes the
variance of the relabeling law, not automatically the sampling variance of a
covariate-standardized AIPW contrast.

The implementation used here is more limited still: the AIPW nuisances and
the observed denominator are frozen while labels are relabeled. Therefore
the routine is a frozen-score diagnostic, not an exact permutation test for
the fitted estimator. General studentized permutation results such as
Chung and Romano [3] and Janssen [4] provide context for particular
non-identically distributed mean problems, but they do not validate this
covariate-adjusted, frozen-nuisance construction.

The confounded DGP supplies a concrete failure witness. Its conditional arm
laws differ by stratum while the target-standardized mean contrast cancels
exactly. Under strong propensity imbalance, the frozen relabeling law is not
centered and scaled as the weak-null sampling law, and the test rejects far
too often. The fleet therefore retains both permutation paths as diagnostics
only. Their agreement on the sharp design is empirical evidence, not an
exact finite-sample guarantee.

## 5. Task 4.5.5 + 4.5.6: fleet results (13,000 replications)

Rejection rates at alpha = 0.05 of the multiplier test (band [0.03, 0.08]):
"perm" = frozen studentized label permutation, "sperm" = studentized
stratified permutation, both diagnostics. There are 500 replications per size
cell and 300 per local-power cell. The table reports the cell-specific Monte
Carlo standard error; it is approximately 0.010 for a 0.05 rate with 500
replications, but can be as high as 0.016 in the displayed size cells and
0.028 in the local-power cells.

| design | n | mult | perm | sperm |
|---|---|---|---|---|
| W2' | 50 | 0.056 | - | 0.070 |
| W2' | 100 | 0.058 | - | 0.036 |
| W2' | 200 | 0.070 | - | 0.056 |
| W2' | 500 | 0.042 | - | 0.032 |
| confounded r=0.0 | 50 | 0.076 | 0.068 | 0.062 |
| confounded r=0.0 | 100 | 0.046 | 0.046 | 0.046 |
| confounded r=0.0 | 200 | 0.054 | 0.036 | 0.046 |
| confounded r=0.0 | 500 | 0.058 | 0.066 | 0.062 |
| confounded r=0.5 | 50 | 0.094 | 0.104 | 0.118 |
| confounded r=0.5 | 100 | 0.058 | 0.092 | 0.096 |
| confounded r=0.5 | 200 | 0.050 | 0.096 | 0.102 |
| confounded r=0.5 | 500 | 0.050 | 0.086 | 0.086 |
| confounded r=1.0 | 50 | 0.154 | 0.198 | 0.228 |
| confounded r=1.0 | 100 | 0.090 | 0.208 | 0.244 |
| confounded r=1.0 | 200 | 0.096 | 0.220 | 0.214 |
| confounded r=1.0 | 500 | 0.066 | 0.228 | 0.248 |
| sharp r=1.0 | 50 | 0.118 | 0.054 | 0.044 |
| sharp r=1.0 | 100 | 0.056 | 0.038 | 0.042 |
| sharp r=1.0 | 200 | 0.072 | 0.056 | 0.058 |
| sharp r=1.0 | 500 | 0.058 | 0.056 | 0.056 |
| W1 | 50 | 0.996 | 1.000 | - |
| W1 | 100 | 1.000 | 1.000 | - |
| W1 | 200 | 1.000 | 1.000 | - |
| local p=0.0 (randomized sharp null) | 200 | 0.033 | 0.040 | - |
| local p=0.25 | 200 | 0.380 | 0.203 | - |
| local p=0.5 | 200 | 0.997 | 1.000 | - |
| local p=0.75 | 200 | 1.000 | 1.000 | - |
| local p=1.0 | 200 | 1.000 | 1.000 | - |

Reading. (1) Under the pure null (W2') and the weak null without
confounding (r=0.0) the multiplier is in band at every n. (2) Under
strong confounding (r=1.0) the multiplier is mildly anti-conservative
at n <= 200 (0.09-0.15) and enters the band by n = 500 (0.066); the
remaining marginal cell is n=200 at 0.096 (SE 0.013). (3) The
diagnostic permutation rates confirm the audit: at r=1.0 the frozen
permutation rejects 20-25% under the weak null at every n, consistent with
the predicted failure of exchangeability, while under the conditional sharp
null (sharp cells) permutation and multiplier agree and are both in
band at n >= 100. (4) W1 power is 1.000 already at n = 50 (gate: >= 0.80
at n = 200), and the local-mixture power curve rises monotonically
0.033 -> 0.38 (p=0.25) -> 1.00 (p >= 0.5), recorded rather than hidden.
(5) The Gaussian-path diagnostic is close to the Gaussian multiplier in the
W2' cells (rates differ by at most 0.004). The Rademacher path is also close
for n >= 100, but is 0.106 versus 0.056 at n=50, so it is retained as a
diagnostic rather than treated as interchangeable. The deterministic-failure
field never triggered across the fleet.

## 6. Task 4.5.7: production decision (qualified gate)

Decision. C2 ships as an empirical studentized multiplier procedure on the
frozen finite-grid contract, with differentiated operating guidance. The
strict gate rule is multiplier size in [0.03, 0.08] for every gating cell with
n >= 200 and >= 500 replications, together with W1 power >= 0.80 at n=200.

Strict result: `in_size_band = False`. Every n >= 200 size cell is in band
except confounded r=1.0 at n=200, which is 0.096 (about 1.2 Monte Carlo SE
above the upper edge); the corresponding n=500 cell is 0.066. The code and
aggregate JSON retain this as a failure, while reporting a separate
`qualified_operating_pass` that records the explicitly documented near-miss as
an exemption. This is a qualified operating recommendation, not a
post-hoc claim that the strict gate passed. The strongest-confounding result
supports n >= 500; n >= 200 is supported for the other tested null designs,
subject to the finite-grid and DGP scope.

Can-claim / cannot-claim paragraph (manuscript-ready). On the frozen 32x32
grid, the studentized Gaussian-multiplier max-t procedure had rejection rates
inside [0.03, 0.08] for all tested weak-null cells with n >= 200 except the
confounded r=1.0, n=200 cell (0.096), and the same strong-confounding design
was inside the band at n=500 (0.066). This fleet supports using n >= 500 as
the conservative operating point under the strongest tested confounding and
does not establish a finite-sample theorem or uniform weak-null control for
all n >= 200. The procedure retains power 1.0 against the W1 separation at
n=50 and the local-mixture curve reaches 1.0 at mixture weight 0.5. No
type-I-error claim is made for the frozen relabeling diagnostics under the
weak null; in the confounded r=1.0 design they reject at roughly 20-25%, while
the sharp-null agreement is empirical only.

## 7. Source map

- [1] Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E.,
  Hansen, C., Newey, W., Robins, J. (2018). Double/debiased machine
  learning for treatment and structural parameters. The Econometrics
  Journal, 21(1), C1-C68. https://doi.org/10.1111/ectj.12097
- [2] Souto-Diamantis: Banach-valued double-robust representation
  (internal adaptation; see `docs/phase4_tdist_decision.md` for the
  Phase 4 source mapping).
- [3] Chung, E., Romano, J.P. (2013). Exact and asymptotically robust
  permutation tests. Annals of Statistics, 41(2), 484-507.
  https://doi.org/10.1214/13-AOS1090
- [4] Janssen, A. (1997). Studentized permutation tests for
  non-i.i.d. hypotheses and the generalized Behrens-Fisher problem.
  Statistics & Probability Letters, 36(1), 9-21.
  https://doi.org/10.1016/S0167-7152(97)00043-6
