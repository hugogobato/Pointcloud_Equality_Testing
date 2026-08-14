# Reuse audit: what P1 imports from `tcda_uq` (Phase 0.8)

`tcda_uq` is the released `CP_TATE` library (github.com/hugogobato/tcda_uq, MIT,
Python 3.10), installed into the project venv from git. P1 imports its point
estimators, cross-fitting, the functional DR-learner, the power-weighted
silhouette, and the tri-oracle simulation; it does **not** reimplement any of
their mathematics. This document is the precise inventory: exact import paths,
calling conventions, what P1 builds on top, and the boundary against the parts
of `tcda_uq` that P1 deliberately does not use.

## Boundary statement (CP_TATE owns bands, P1 owns p-values)

`tcda_uq` quantifies *uncertainty*: simultaneous confidence bands for the TATE
and CTATE and conformal prediction bands for the ITTE. P1 does *hypothesis
testing*. The only shared objects are the AIPW curve and the per-unit
efficient-influence-function (EIF) process, and they should be named as such in
the manuscript:

* P1's statistic is `T_n = sqrt(n) * ||psi_hat_d||_inf`, computed from
  `cross_fit(...).aipw[d]` (the mean AIPW curve for homology dim `d`).
* Its null law is calibrated by a **multiplier bootstrap over the centered
  per-unit EIF process**, `cross_fit(...).scores[d]` (equivalently
  `cross_fit(...).influence()[d]`): draw iid mean-zero unit-1 multipliers
  `xi_i`, form `G_b(t) = n^{-1/2} sum_i xi_i (s_{i,d}(t) - mean)`, and compare
  `sup_t |sqrt(n) psi_hat_d(t)|` against the bootstrap law of `sup_t |G_b(t)|`.
  The p-value itself is P1's (the `tda2s` resampling engine); the influence
  process it reweights is tcda_uq's.

Everything downstream of that point (band construction, Liebl-Reimherr,
Pini-Vantini, conformal prediction) belongs to `CP_TATE`, and P1 must not claim
it. The shim in `tda2s/adapters/tcda_uq.py` therefore exposes only the
estimators and the EIF; banding entry points are documented below but not
imported.

## 1. `tcda_uq.estimators` — point estimation and cross-fitting

### 1.1 `cross_fit` — the P1 workhorse

Import path: `from tcda_uq.estimators import cross_fit` (module
`tcda_uq/estimators/nuisance.py`).

```python
cross_fit(sample, tseq, n_basis=3, propensity_estimator=None, n_splits=2,
          stratify=True, random_state=0, propensity_feature_fn=None)
    -> CrossFitResult
```

`sample` is the observed triplet `(phi, A, X)`:

* `phi`: silhouettes, shape `[n, n_hom_dim, resolution]`
* `A`:   treatment/group label, shape `[n]`
* `tseq`: silhouette grid `[resolution]`
* `n_basis`: Fourier basis size of the function-on-scalar outcome regression
* `propensity_estimator`: sklearn classifier (default `RandomForestClassifier`)
* `propensity_feature_fn`: optional `X -> features` map for the propensity model

`CrossFitResult` fields (dataclass, `tcda_uq/estimators/nuisance.py`):

| Field | Shape | Meaning |
|---|---|---|
| `tseq` | `[resolution]` | the grid |
| `aipw` | list, per hom dim, `[resolution]` | cross-fitted mean AIPW TATE curve `psi_hat_d` |
| `ipw` | list, per hom dim, `[resolution]` | mean IPW curve (no outcome model) |
| `plugin` | list, per hom dim, `[resolution]` | mean plug-in `mu_1 - mu_0` curve |
| `scores` | list, per hom dim, `[n, resolution]` | per-unit doubly-robust score (EIF) process; `scores[d].mean(0) == aipw[d]` |
| `pi_hat` | `[n]` | cross-fitted propensity (each unit scored by out-of-fold nuisances) |
| `order` | `[n]` | index into the original sample for each row of `scores` |
| `folds` | list of `NuisanceFit` | one per fold, with `predict_mu` / `predict_pi` |

Methods: `influence()` returns the *centered* EIF process
(`scores - mean`, one `[n, resolution]` per hom dim) — the object the
multiplier bootstrap reweights; `tate()` is an alias for `aipw`.

What P1 builds on top: the statistic `T_n = sqrt(n) ||psi_hat_d||_inf` from
`result.aipw[d]`, calibrated by multiplier bootstrap over `result.influence()[d]`
(or `result.scores[d]`, centered). The permutation-style null distribution of
the two-sample test (Phase 2+) also reuses `aipw[d]` and `scores[d]`; diagram
recomputation is never inside a bootstrap loop (diagrams are cached in
`tda2s.ph`).

### 1.2 Nuisance helpers

Import path: `from tcda_uq.estimators import fit_functional_regression,
predict_functional_regression, fit_propensity` (same module).

```python
fit_functional_regression(sample, tseq, n_basis)      # -> [f_reg0, f_reg1] per hom dim
predict_functional_regression(reg, X_eval, tseq)      # -> [(mu0, mu1) ...], [n_eval, res] each
fit_propensity(X, A, estimator=None)                  # -> fitted sklearn classifier
```

The outcome model is a Fourier function-on-scalar regression via scikit-fda
(`LinearRegression` on `FourierBasis`), one model per arm per homology dim; the
propensity defaults to a random forest. P1 does not call these directly in its
own math: they are the nuisances that `cross_fit` fits per fold, and P1 accesses
them only through `CrossFitResult.folds` if ever needed. They matter to P1
through the *well-specified-ness* of the simulation: the tri-oracle DGP (below)
generates outcomes in exactly this model family, so `cross_fit` with matching
`n_basis` has well-specified nuisances.

### 1.3 Per-unit score construction (`aipw` module)

Import path: `from tcda_uq.estimators import aipw_estimator, aipw_scores,
ipw_estimator, plugin_estimator` (module `tcda_uq/estimators/aipw.py`).

These are the mean estimators and the per-unit score decomposition that
`cross_fit` uses internally; `aipw_scores(pi_hat, mu_hats, sample)` returns the
per-unit DR score process `phi_hat_{i,d}(t) = mu1-mu0 + A/pi*phi - ...`
whose sample mean is the AIPW estimate. P1 consumes them only through
`CrossFitResult` (never by re-evaluating `aipw_scores` itself), which keeps the
single source of truth in tcda_uq.

### 1.4 `CTATEDRLearner` — the functional DR-learner (C3, conditional null)

Import path: `from tcda_uq.estimators import CTATEDRLearner` (module
`tcda_uq/estimators/ctate_dr_learner.py`).

```python
CTATEDRLearner(n_basis=5, *, feature_fn=None, stage1_n_basis=None)
learner.fit(sample, tseq, *, cross_fit_result=None, **cross_fit_kwargs)
learner.predict(X_eval)      # -> [m, n_hom_dim, resolution]  (or [n_hom, res] for 1-D x)
```

The DR-learner regresses the cross-fitted EIF pseudo-outcome curves on
`feature_fn(X)` (default linear in `X`) with a Fourier second stage. Fitted
fields: `cross_fit_result_` (the Stage-1 `CrossFitResult`), `X_`, `pseudo_`
(the per-unit pseudo-outcome curves), `tseq`, `n_hom_dim`, and `stages_` — a
list of `SecondStageFit` per hom dim with `B` (basis), `M` (`pinv(Z)`), `gamma`
(coefficients), `coef`, `resid` (second-stage residual curves). Convenience
methods `predict_dim`, `weights(x, d)` (smoother weights `a(x)`), `residuals(d)`.

What P1 builds on top: the conditional two-sample test on
`tau_hat_d(t, x) = learner.predict(X_eval)[:, d, :]` (C3), whose null is
`tau_d(t, x) = 0` for all `(t, x)`; the pointwise curves and the smoother
weights come from the learner, P1 only attaches its p-value calibration.

## 2. `tcda_uq.silhouette` — persistence to vectorisation

Import path: `from tcda_uq.silhouette import compute_silhouette,
diagrams_from_pointcloud, power_weight` (module `tcda_uq/silhouette/core.py`,
re-exported in `tcda_uq/silhouette/__init__.py`).

```python
power_weight(point, r=3.0)                              # |death - birth| ** r
compute_silhouette(diags, interval=(0.0, 0.2), r=3.0, resolution=100)
    # -> (n_hom_dim, resolution)
diagrams_from_pointcloud(points, homology_dims=(0, 1))
    # alpha-complex diagrams in radii, essential classes dropped
```

Defaults: interval `(0, 0.2)`, `r=3`, `resolution=100` (the Kim-Lee
parameterisation `w_p = (b_p - a_p)^r`). The module also exposes
`s silhouette_from_pointcloud` and `silhouette_from_image` (cubical, for
SARS-CoV-2 CT scans) — the latter is a CP_TATE-applied-data modality, out of
P1's scope.

What P1 builds on top: the observed silhouettes `phi` fed to `cross_fit` can be
produced either by `tda2s.vec` (P1's own vectorisation stack, Phase 0.3, with
the same convention: radii, essential classes dropped) or, for direct
comparability with CP_TATE runs, by delegating to this module through the shim.
`tda2s` and `tcda_uq` agree on the diagram conventions (radii, `inf` deaths
dropped), which is what makes the two pipelines interchangeable.

## 3. `tcda_uq.datasets.TriOracleSimulation` — the tri-oracle DGP

Import path: `from tcda_uq.datasets import TriOracleSimulation, SimulationSample`
(module `tcda_uq/datasets/simulation.py`).

```python
TriOracleSimulation(n_cov=5, n_hom_dim=2, resolution=100, interval=(0.0, 1.0),
                    n_basis=5, mu1=None, mu2=None, sigma2=0.5, beta=None,
                    coef_scale=1.0, coef_decay=0.5, noise_scale=0.3,
                    noise_df=None, prop_scale=1.0, hetero_scale=0.0, seed=0)
sim.sample(n, rng=None) -> SimulationSample
```

`SimulationSample` fields: `tseq`, `X` `[n, d]`, `A` `[n]`, `propensity` `[n]`
(true `pi(X)`), `potential_outcomes` `[n, 2, n_hom, res]`, and the three oracles
`oracle_tate` `[n_hom, res]`, `oracle_ctate` `[n, n_hom, res]`,
`oracle_itte` `[n, n_hom, res]`; property `.observed` gives the factual triplet
`(phi, A, X)` in the exact shape `cross_fit` consumes.

Caveat for P1: the propensity model hardcodes the interaction terms
`X1 * X2` and `X0 * X2`, so **`n_cov` must be at least 3**. The Phase 0.8 test
uses `n_cov=3` as the minimal well-defined configuration.

What P1 builds on top: the oracle harness for validating P1's own DGP
(Phase 0.6) and the estimator stack: on a `TriOracleSimulation` sample with
well-specified nuisances, `cross_fit(...).aipw[d]` must land within sampling
noise of `oracle_tate[d]` (pinned by `tests/test_tcda_uq_shim.py`; measured
sup-norm distance ~0.08 at `n=100`, `n_cov=3`, `coef_scale=0.5`,
`noise_scale=0.15`, seeded random-forest propensity, seed 2).

## 4. Multiplier draws in `tcda_uq.uq.asymptotic`

The bootstrap reweighting law is generated by a single helper, shared by every
band construction in the package:

```python
# tcda_uq/uq/asymptotic/multiplier_bootstrap.py  (private, prefixed "_")
_draw_multipliers(kind, shape, rng)
    # kind in {"gaussian", "rademacher", "mammen"}; iid mean-0 variance-1
    # returns [shape] array of multipliers xi
```

Its consumers (all in `tcda_uq.uq.asymptotic`, re-exported in
`tcda_uq/uq/asymptotic/__init__.py`):

```python
multiplier_bootstrap_band(influence, tseq, estimate, *, alpha=0.05, n_boot=2000,
                          multiplier="rademacher", standardize=True,
                          variance=None, rng=None) -> tcda_uq.metrics.Band
topological_effect_test(influence, estimate, *, n_boot=2000,
                        multiplier="rademacher", standardize=False,
                        variance=None, rng=None)  # -> {statistic, pvalue, crit}
iwt_pvalues(...)   # Pini-Vantini interval-wise testing (module pini_vantini.py)
```

What P1 builds on top: P1's p-value calibration uses the same multiplier
families (Rademacher default, Gaussian/Mammen options) over the *shared* EIF
process `cross_fit(...).influence()[d]`. The draws themselves are trivially
three lines, but the convention (family, studentization default) is CP_TATE's;
P1's `tda2s.resample` engine reproduces the same laws so that p-values and
CP_TATE bands are calibrated on identical null processes. The band objects
themselves (`tcda_uq.metrics.Band`) and the banding callables are **not** part
of P1's import surface (boundary statement above).

## 5. What P1 does not import (deliberately out of scope)

* `tcda_uq.uq.asymptotic.multiplier_bootstrap_band` / `_bands` — TATE confidence band
* `tcda_uq.uq.asymptotic.ctate_confidence_band` — CTATE confidence band
* `tcda_uq.uq.asymptotic.liebl_reimherr_*`, `pini_vantini_*` — fair pivotal / IWT bands
* `tcda_uq.uq.conformal.*` — ITTE prediction bands (adaptive/conformal machinery)
* `tcda_uq.metrics.coverage` / `plotting` — band evaluation helpers
* `tcda_uq.datasets.sarscov2` / `orbit` / `topological` loaders — applied-data
  and Level-B modalities P1 does not depend on

## 6. The shim and its mapping

`tda2s/adapters/tcda_uq.py` is pure delegation plus argument plumbing:

| P1 need | tcda_uq import | shim call |
|---|---|---|
| Cross-fitted AIPW curve + EIF | `tcda_uq.estimators.cross_fit` | `aipw_curve(sample, tseq, n_basis, n_folds=5, **kwargs)` -> `{aipw, scores (n, n_hom, res), pi_hat, tseq}` |
| Power-weighted silhouette | `tcda_uq.silhouette.compute_silhouette` | `silhouettes(diags, interval=(0, 0.2), r=3, resolution=100)` |
| Oracle DGP | `tcda_uq.datasets.TriOracleSimulation` | `tri_oracle(n, **sim_kwargs)` -> `SimulationSample` |
| CTATE DR-learner | `tcda_uq.estimators.CTATEDRLearner` | `ctate_learner(*args, **kwargs)` |

The only reshaping in the shim is `np.stack(result.scores, axis=1)` so P1 gets
one `[n, n_hom_dim, resolution]` array instead of a per-dimension list; all
mathematics lives in `tcda_uq`. A grep for AIPW / cross-fitting / DR-learner
math in `tda2s/` finds only this adapter and its tests.
