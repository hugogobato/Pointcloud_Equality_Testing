# Phase 0 — completion record and audit

Status as of 2026-08-13. Plan: `RESEARCH_PLAN_P1_TwoSample.md` §Phase 0.

## Exit criterion — PASS

> generate 100 clouds → compute diagrams → vectorise → run all competitors →
> get p-values, in under 10 min on 20 threads

`scripts/phase0_exit_benchmark.py`, 100 clouds × 150 pts, alpha filtration,
H(0,1), 16 workers, 200 permutations:

| stage | cold | warm (cached diagrams) |
|---|---|---|
| generate 100 clouds | 0.00 s | 0.02 s |
| diagrams (16 workers) | 0.08 s | 0.24 s |
| vectorise × 6 representations | 0.35 s | 0.17 s |
| all 7 competitors | 166.7 s | 170.6 s |
| **total** | **167.1 s** | **171.0 s** |

Budget 600 s. Diagram computation is negligible; `han` (bandwidth-aggregated
kernel test) is 83% of the whole run at 138 s and is the cost to watch when
Phases 3–5 need thousands of permutations.

Sanity: with `group_effect=1` (group A gets one extra loop), six of seven
competitors reject at 5% — rt 0.015, mmd 0.005, han 0.010, strand 0.005,
moon_lazar ~0, krebs_rademacher 0.005; frechet_anova 0.134 is the outlier,
consistent with its coordinate-wise-median proxy being coarse (see below).

## Test suite

85 tests, all green (78 core + 7 published reproductions).

| file | covers |
|---|---|
| `test_smoke.py` | 0.1 scaffold |
| `test_ph.py` | 0.2 pipeline, torus Betti (1,2,1), filtration scale |
| `test_vec.py` | 0.3 representations, silhouette W₁-Lipschitz |
| `test_resample.py` | 0.4 permutation / multiplier / paired / smoothed / folds |
| `test_benchmarks.py` | 0.5 size and power of all 7 wrappers |
| `test_published_reproductions.py` | 0.5 published figures — Dubey–Müller Fig. 1, Moon–Lazar Fig. 5, Robinson–Turner, Krebs–Rademacher pivotal law |
| `test_dgp.py` | 0.6 knob independence, oracle recovery |
| `test_tcda_uq_shim.py` | 0.8 delegation, AIPW-vs-oracle |
| `test_forward_compat.py` | 0.9 unequal sizes, dim 20, external standardise, DTM-Rips |

The reproduction file is the slow one (≈25 min: 1200 Monte Carlo replications of
the Moon–Lazar design). Everything else runs in a few minutes.

## Defects found and fixed during the audit

1. **Čech filtration was on a squared-radius axis.** `gudhi.DelaunayCechComplex`
   reports squared circumradii (like `AlphaComplex`), but `_cech_diagrams` did
   not take the square root, so `cech` diagrams were incomparable with every
   other filtration. A unit-radius circle hid it (1² = 1). Fixed; regression
   test `test_filtration_values_are_radii_not_squared_radii` now asserts every
   filtration is 1-homogeneous in the cloud scale.
2. **Cubical grid padding was an absolute `1e-6`,** so the cubical filtration
   was not scale-equivariant. Padding is now relative to the cloud's extent.
3. **`smoothed_bootstrap` pooled instead of resampling.** Each bootstrap unit
   `vstack`-ed all *n* diagrams into one, inflating every Betti count *n*-fold
   (observed statistic 2.6 vs bootstrap draws ~160). It now draws one index per
   bootstrap unit and repairs jittered points that fall below the diagonal.
4. **`mean_betti_curve` averaged curves sampled on different grids** — each
   sample derived its own `(0, max death)` interval. The interval is now
   derived once from the pooled diagrams.
5. **The silhouette Lipschitz check tested the wrong metric.** Kim–Lee's
   Lemma 2.1 bounds `‖ΔΛ‖_∞` by **W₁**; the test used the bottleneck distance
   W∞ (mislabelled "W1 proxy") and passed only because it moved a *single*
   diagram point, where W₁ = W∞. Under a many-point perturbation the W∞ ratio
   reaches **7.0 at r = 1**, far outside the asserted factor of 2. Against W₁ the
   constant is stable at ≈0.5–0.6 for r ∈ {1,2,3}. The test is now a randomised
   sweep over r and over random subsets, plus a guardrail test that pins the
   W∞ failure so the metric is not quietly swapped back.
6. **`COMPETITORS` was not callable with one uniform kwargs dict** — `moon_lazar`
   is analytic and takes no `n_perm`/`seed`, so sweeping the registry raised
   `TypeError`. Added `run_competitor(name, ...)`, which passes only the kwargs
   a given wrapper accepts, and corrected the module's "uniform signature" claim.
7. **`tda2s/dgp/simulation.py` never used its intended silhouette path** —
   `from tda2s.vec import compute_silhouette` always raised `ImportError` (the
   function is named `silhouette`) and silently fell through to a gudhi
   fallback. Now calls `tda2s.vec.silhouette` directly.
8. **Three wrappers did not implement their source paper** — `frechet_anova`,
   `moon_lazar` and `krebs_rademacher`. All three rewritten; see §0.5.
9. Minor: unused `np.random.default_rng(seed)` in `CloudSampleDGP.__init__`
   (the `seed` argument does not affect sampling — `sample(rng=...)` does);
   `n_per_group` documented as if groups were balanced, when labels are drawn
   `A ~ Bern(π(X))` and deliberately are not; stale comment in
   `krebs_rademacher.py`. All corrected in place.

`betti_curve` in `tda2s/vec` was also vectorised (it looped over grid points in
Python), which the smoothed-bootstrap coverage test needs to run in seconds.

## Verified acceptance checks

| # | Check | Status |
|---|---|---|
| 0.1 | smoke test on noisy circle; `tcda_uq` installs in same env | PASS — installed from git, commit `71203a3` (`direct_url.json` verified) |
| 0.2 | torus Betti (1,2,1) | PASS — alpha and VR |
| 0.3 | silhouette Lipschitz (Kim–Lee Lemma 2.1) | PASS — **restated against W₁**, see defect 5 |
| 0.4 | multiplier bootstrap nominal coverage on a GP toy | PASS — 200 MC reps, coverage in [0.90, 0.98] |
| 0.5 | each wrapper reproduces a published figure or table | PASS — Dubey–Müller Fig. 1 (both panels), Moon–Lazar Fig. 5 (all four σ), Robinson–Turner via Fig. 5b; Krebs–Rademacher has no simulation study, its pivotal law is checked instead. See §0.5 below |
| 0.6 | oracle diagrams recoverable; knobs independent | PASS |
| 0.7 | zero unresolved load-bearing refs | PASS — 31 entries, every one carries a DOI or URL |
| 0.8 | zero reimplemented AIPW / cross-fitting / DR-learner | PASS — `tda2s/adapters/tcda_uq.py` is pure delegation |
| 0.9 | four forward-compat tests; no ecological data in repo | PASS — grep-verified, only docstring mentions |

## 0.5 — provenance and published reproductions

### Author code: none exists

Checked each source before writing anything. **No author implementation is
released for any of the seven methods** (no repository linked from the paper,
the journal supplement, or the authors' pages). Dubey & Müller's Fréchet ANOVA
has third-party R implementations, but calling R would require `rpy2`, which is
outside the approved dependency set. So every wrapper is a replication from the
paper, with the section and equation numbers cited in its module docstring, and
each is checked against a published figure where the paper has one.

### The three deviating wrappers were rewritten

The audit found three wrappers implementing something other than their source.
All three are now direct transcriptions:

* **`frechet_anova`** — was a between/within ratio on coordinate-wise medians of
  persistence vectors. Now Dubey & Müller eqs. (6)–(11) exactly: pooled and
  within-group Fréchet variances give `F_n` (eq. 7), the Levene-type `U_n`
  (eq. 8) supplies the variance-contrast term, and `T_n` (eq. 11) combines them.
  The `U_n` term is precisely what the old proxy was missing, and it is why the
  scale panel of Figure 1 below is reproducible at all.
* **`moon_lazar`** — was sorted zero-padded persistence vectors. Now Algorithm 1
  exactly: persistence images (Adams et al. parameterisation, with cell
  integrals in closed form as products of normal-CDF differences), the
  `v_x ≥ v_y` pre-filter, the stage-I pooled-sd variance filter at the C-th
  percentile, pooled-variance t-tests, then BH (or BY).
* **`krebs_rademacher`** — was a sup-norm Betti-curve statistic. Now the paper's
  Section 1.2 inco-variance test: the U-statistic kernel `h = ½W_r²` (eq. 1.13),
  the two-parameter partial-sum processes (eq. 1.12), the self-normaliser
  (eq. 1.14) and the pivotal Brownian limit (eq. 1.16).

### Reproductions — `tests/test_published_reproductions.py`

| method | published target | published | ours |
|---|---|---|---|
| `frechet_anova` | Dubey & Müller Fig. 1 **left** (location, sd 0.5) | ≈0.05 at δ=0, →1 by \|δ\|≈0.5 | 0.073 / 0.227 / 0.947 at δ = 0, 0.25, 0.5 |
| `frechet_anova` | Dubey & Müller Fig. 1 **right** (scale, sd 0.2) | ≈0.05 at r=1, →1 by r≈1.5 | 0.047 / 0.967 / 1.000 at r = 1, 1.5, 2 |
| `moon_lazar` | Moon & Lazar Fig. 5a/5b, σ = 0.05 | FPR 0.022, power 0.98 | FPR 0.047, power 0.987 |
| `moon_lazar` | … σ = 0.10 | FPR 0.040, power 0.62 | FPR 0.040, power 0.600 |
| `moon_lazar` | … σ = 0.15 | FPR 0.035, power 0.24 | FPR 0.020, power 0.213 |
| `moon_lazar` | … σ = 0.20 | FPR 0.025, power 0.10 | FPR 0.020, power 0.073 |
| `rt` | the "PD" curve of Moon & Lazar Fig. 5b | power ≈0.97 at σ=0.05, ≈0.05 at σ=0.20 | 1.000 at σ=0.05, 0.275 at σ=0.20, FPR 0.075 |
| `krebs_rademacher` | **none exists** — see below | — | eq. (1.16) law calibrated: rejects at 0.0485 at its own q₉₅ |

Papers use 500–1000 replications; these use 150 (40 for `rt`), so the Monte
Carlo standard error is ≈0.04 and every assertion is stated as a tolerance, not
a digit match. The one visible gap is `rt` at σ = 0.20 (0.275 vs ≈0.05): Moon &
Lazar do not state which loss or how many permutations they used when re-running
Robinson & Turner, so the high-noise tail of that curve is not pinned down by
the published text. It is bounded in the test at ≤0.30 and flagged here rather
than tuned to match.

`mmd`, `han` and `strand` publish no simulation figure on a design that can be
reconstructed from the text, and are validated by the size/power tests in
`test_benchmarks.py`.

### Krebs & Rademacher has no figure to reproduce

arXiv:2401.10349 is 43 pages of theory with **zero** occurrences of
"Simulation", "Table" or "Monte Carlo" (checked by extracting the PDF text).
There is nothing to reproduce. What is checkable is the reference law itself, so
the test asserts that the eq. (1.16) pivotal limit is mesh-stable (q₉₅ = 27.84
at grid 20 vs 27.93 at grid 40) and self-calibrated (rejects at 0.0485 at its
own 95th percentile).

Two properties of this method must be carried into Phase 6's table, because they
are the published method's behaviour and not implementation defects:

1. **It tests dispersion, not location.** Eq. (1.7) is
   `H₀: (σ²_X − σ²_Y)² ≤ Δ`, so it is blind to a pure location shift by design.
   It scores 0/40 on the circle-vs-two-circles power design and 20/20 on a
   random-radius (dispersion) alternative; pinned by
   `test_krebs_rademacher_targets_dispersion_not_location`.
2. **Δ = 0 is outside the paper's framework.** Theorem 1.4's scale factor
   `ξ ∝ (σ²_X − σ²_Y)` vanishes exactly on the classical null, so `W` is not the
   right reference law there. The wrapper's default `delta=0` therefore
   calibrates by permutation and is labelled an extension in its docstring;
   `delta > 0` runs the published self-normalised procedure.
