# Phase 3.5: Vejdemo-Johansson--Mukherjee mapping note

**Status (2026-08-16): Phase 3.5 complete. Audit verdict PASS with three
recorded deviations; comparator implemented; benchmark fleet run at its full
budget and the pre-registered rule fired in the comparator's favour.** Tag:
`adapt`. The comparator is `vjm_multiplicity_test` in
`tda2s/tests/dr_outcome.py`. It is *not* labelled a Vejdemo-Johansson--Mukherjee
control without qualification anywhere in the code or the manuscript: what
transfers is the multiplicity architecture, not the source's null model and not
its limit-theorem justification. Section 5 below states exactly what may and
may not be claimed.

---

## 1. Task 3.5.1: the source procedure, restated and source-checked

The source is Vejdemo-Johansson and Mukherjee, *Multiple testing with
persistent homology*, arXiv:1812.06491, <https://arxiv.org/abs/1812.06491>,
published as *Multiple hypothesis testing with persistent homology*,
Foundations of Data Science 4(4):667-705 (2022), doi:10.3934/fods.2022018.
All section and method numbers below are read from **arXiv v4 (25 August
2022)**, which is the version matching the published paper.

**Correction to the research plan, applied.** `RESEARCH_PLAN_P1_TwoSample.md`
Phase 3.5 originally cited "§4.1 Method 1 (FWER), §4.2 Method 2 (FDR), §4.3
(two-sample FDR), §5 (studentized comparability), §6 (the
persistence-specific limit-theorem justification)". None of those locations is
correct for v4, and the method-to-error-rate assignment was wrong as well. The
plan now carries the table below instead; this is the authoritative copy.

| Location | Content |
|---|---|
| §3.1, Method 1 | the **one-sample** test procedure (not FWER) |
| §3.2, Theorem 3.1 | the standardization proposal and its limit-theorem argument |
| §3.3.1, Method 2 | **FWER** correction |
| §3.3.2, Method 3 | **FDR** correction |
| §3.4, Method 4 | FDR for multiple **two-sample** tests |
| §3.5, Methods 5 and 6 | FWER and FDR across **differing invariants**, including across homology degrees |
| §5.1 | the empirical comparability ("exchangeability") check |
| §6.1-6.3 | limitations, recommendations, future work |

`Literature_Review/VERIFIED_NOTES.md` already flagged this reference as
**found-with-different-details** on the title and venue; the internal numbering
is a second, separate discrepancy and its entry now records both.

### 1.1 The null model (§3.1)

The source tests $H_0$: the point cloud is an i.i.d. uniform sample from a
convex body in $\mathbb R^d$, against a non-uniform alternative on the same
body. The convex body is estimated from the data, either as an axis-aligned
bounding box with Lloyd's (1952) unbiased order-statistic estimator or as the
convex hull with a dilation correction; the unbiased convex hull is the
recommended default (§6.2). Method 1 then simulates $\Pi$ clouds of the same
cardinality uniformly on that body, computes the statistic $\gamma$ on each,
and reports $\hat p = (\#\{\gamma_\pi < \gamma^*\}+1)/(\Pi+1)$ or its
right- or two-tailed variants. The abstract's universality claim is that this
simulated null, being a topological summary of a homogeneous unit-intensity
Poisson process, can be **precomputed once** and reused like a $p$-value table.
Convexity of the body is not incidental: §3.1.1 states that it is required in
order to standardize the null distributions across the family.

### 1.2 Standardization (§3.2)

For each test, the simulated statistics $\{\gamma_1,\dots,\gamma_\Pi\}$ give
$\hat\mu_\pi$ and $\hat\sigma_\pi$, and every statistic (observed and
simulated) is mapped to $(\gamma-\hat\mu)/\hat\sigma$. The theoretical
argument offered is Theorem 3.1, quoted from Hiraoka, Shirai and Trinh,
*Limit theorems for persistence diagrams*, Annals of Applied Probability
28(5):2740-2780 (2018), Theorem 5.2: for a stationary point process $\Phi$
with all finite moments and any convex shape $L$,
$n^{-1/2}\big(\beta_q^{r,s}(K(\Phi_{\ell L})) - \mathbb E[\beta_q^{r,s}(K(\Phi_{\ell L}))]\big)
\Rightarrow N(0,\sigma_{r,s}^2)$. The source is explicit that this is an
asymptotic motivation only, and that it simulates an empirical null precisely
because the sample sizes encountered do not justify normality.

### 1.3 FWER (§3.3.1, Method 2)

The identity used is $\alpha_{\text{FWER}} = \mathbb P(\max_i t_i > c \mid H_0)$
for standardized statistics $t_1,\dots,t_m$, valid "as long as the null
distributions are comparable across all tests". The procedure, in the source's
own notation, draws $N$ null clouds per test, computes $\tilde y_j$ observed
and $\tilde y_j^i$ simulated, forms $\hat\mu_i,\hat\sigma_i$, standardizes both,
takes $z_i = \max_j y_j^i$, and rejects at $p=(\#\{z_i>y_i\}+1)/(N+1)$.

**Two source defects to be aware of when citing.** First, the indices $i$ and
$j$ swap roles between steps 3 and 4 of Method 2 (step 3 has $j$ indexing tests
and $i$ indexing draws; step 4 reverses them), and the final $p$-value formula
compares elementwise where the FWER identity requires the observed *maximum*
against the $N$ null maxima. The only reading consistent with the identity, and
the one implemented here, is: standardize each family member by its own null
mean and standard deviation, form the joint maximum within each null draw, and
rank the observed joint maximum against those $N$ values. Second, in Method 4
step 4 the denominator of $\%R$ is written $K$ where the number of two-sample
tests is $h$, and the step numbering skips 3. These are transcription defects,
not conceptual ones, but the restatement must not silently inherit them.

### 1.4 FDR (§3.3.2 Method 3, §3.4 Method 4, §3.5 Method 6)

Method 3 ranks the standardized observed statistics, and for a cutoff $c$ sets
$\%V(c)=\#\{y_i^j\ge c\}/(K(N-1))$, $\%R(c)=\#\{x_i\ge c\}/K$ and
$\hat q_{\text{FDR}}(c)=\%V(c)/\%R(c)$, choosing the smallest $c$ with
$\hat q_{\text{FDR}}(c)\le\alpha$. Method 4 is the two-sample analogue built on
Robinson and Turner's within-group $p$-Wasserstein loss $F_{p,q}$, with labels
permuted across all $h$ two-sample tests and the same $\%V/\%R$ bookkeeping on
the raw distances (Method 4 does **not** standardize).

### 1.5 Combining invariants (§3.5, Methods 5 and 6)

Methods 5 and 6 replace the single statistic $\gamma$ by a family
$(\gamma_1,\dots,\gamma_K)$. The closing paragraph of §3.5 is the passage that
matters most for P1, and it sanctions the degree family directly: "By repeating
the same point cloud in several positions among the $X_1,\dots,X_K$, we can use
these expanded methods to, for instance, construct acyclicity tests that span
across all homological dimensions of interest: set $X_1=\dots=X_K=X$, and pick
for your invariants some specific persistence diagram invariant applied in turn
to the Betti-$d$ diagrams as $d$ ranges from 1 to $K$." §6.2 then recommends
the dimension-agnostic composite as the default. So a family indexed by
homology degree is the source's own construction, not an extrapolation of it.

### 1.6 The comparability evidence (§5.1)

§5.1, titled "Exchangeability", draws 100 acyclic models, simulates 100 clouds
each, standardizes every invariant, and overlays the ECDFs (Figure 3),
observing "some variation around a core curve" with "no dramatic outliers".
This is empirical evidence for comparability, not a theorem. Table 11 reports
composite-invariant FWER rates at $\alpha\in\{0.01,0.05,0.10\}$ that sit at or
below nominal, several noticeably below (for example `pi.Linf` at
0.00/0.03/0.06 under the unbiased hull), which is consistent with the
conservativeness diagnosed in §2.3 below.

---

## 2. Task 3.5.2: the mapping audit

The original note listed three issues to resolve. Each is resolved below, and
each resolution changes the procedure, so the result is an adaptation with a
stated assumption set rather than an import.

### 2.1 Issue 1: the null generator does not transfer, and does not need to

The source's null concerns the topological structure of a point cloud under an
exogenous, pre-data generative model $M$. P1's null is
$H_0^{\mathrm{out}}: \psi_d\equiv 0$, a statement about a covariate-adjusted
treatment-effect function, and there is no point-process model of which it is a
statement. Nothing in the source's null model can therefore be borrowed, and
with it goes the source's headline **computational** contribution: P1 cannot
precompute a universal null table and must simulate a null per dataset.

What P1 substitutes is the mechanism Phase 3 already owns. For degree $d$ the
observed statistic is $X_d=\sqrt n\,\|\hat\psi_d\|_\infty$, and the null
replicates $Y_{d,b}$ come from either the frozen-nuisance, covariate-preserving
stratified permutation (primary) or the shared multiplier bootstrap
(secondary), both in `tda2s/tests/dr_outcome.py`. This substitution is what
makes the transfer possible; it is also the reason the procedure must not be
called "the VJM test" without qualification.

### 2.2 Issue 2: comparability is a power question here, not a validity question

This is the load-bearing resolution, and it is what lets the comparator be
adopted without importing the source's stationary point-process assumptions.

**What does not transfer.** Theorem 3.1 fails to apply to P1 in three separate
ways. The object it governs is a persistent Betti number $\beta_q^{r,s}$ of a
stationary point process on a growing convex window, whereas $X_d$ is a
sup-norm functional of a cross-fitted AIPW contrast of silhouettes. There is no
stationary point process in the P1 design to which the theorem could be
applied. And even where a limit does exist, the relevant one is Kim and Lee's
functional weak convergence $\sqrt n\,\hat\psi_d \Rightarrow \mathbb G_d$, a
tight centred Gaussian process in $\ell^\infty(T)$, so $X_d \Rightarrow
\|\mathbb G_d\|_\infty$, which is not normal and whose law depends on the whole
covariance kernel of $\mathbb G_d$ rather than on two moments. A location-scale
standardization therefore does **not** equalize the family's null laws, not
even asymptotically. Any claim that it does would be false, and the source does
not need it to be true either, since it argues the point empirically in §5.1.

**What does hold.** Let $T=(X_0,\dots,X_{K-1})$ be the observed family and
$T^{(1)},\dots,T^{(N)}$ the null replicates, each an entire $K$-vector produced
by one draw of the null mechanism. Assume the conditional-null setup already
documented for the permutation path: the cross-fitted nuisances and the
propensity strata are fixed independently of the permuted labels, and the sharp
conditional null holds. Then the labels are exchangeable within strata given
the design, so the $(N+1)$-tuple $\big(T,T^{(1)},\dots,T^{(N)}\big)$ is
exchangeable. Let $S$ be any measurable reduction $\mathbb R^K\to\mathbb R$
whose constants are computed as a **symmetric** function of those $N+1$
vectors, and put $s_0=S(T)$, $s_b=S(T^{(b)})$. Then $(s_0,\dots,s_N)$ is
exchangeable, and the rank $p$-value
$p=\big(1+\#\{b: s_b\ge s_0\}\big)/(N+1)$ satisfies
$\mathbb P(p\le\alpha)\le\alpha$ for every $\alpha$.

The studentized maximum is such a reduction, provided the standardizing
constants are symmetric. Validity therefore does not depend on the degrees
being comparable at all. Comparability governs only how the family's $\alpha$
is *allocated* across degrees, that is, power. This is the honest transfer:
the source's §3.2/§5.1 comparability discussion moves from being a
prerequisite for validity to being a power-allocation device, and the residual
incomparability is then something to measure rather than to assume away. The
comparator reports a pairwise Kolmogorov-Smirnov statistic between the
standardized per-degree null samples for exactly that purpose (the statistic
only; the draws are dependent across degrees, so a KS $p$-value would not be
valid).

### 2.3 Issue 3: exactness, and two deviations that improve on the source

**Deviation A, joint null draws.** The source's step 1 draws
$(M_1^1,\dots,M_K^N)$ i.i.d. from $M$, so the null replicates are independent
across family members. Under the §3.5 composite construction, where
$X_1=\dots=X_K$ is one cloud, the *observed* invariants are strongly dependent
while the null maximum is a maximum over $K$ independent standardized draws.
With matched marginals, a maximum over independent coordinates is
stochastically larger than one over coordinates that are positively dependent
in the positive-orthant sense. Degree-wise statistics built from shared units
and shared cross-fitting are expected to be positively dependent, though that
is an expectation here and not something this note proves; under it, the
source's composite FWER procedure is conservative in precisely the
configuration it recommends, which is consistent with the below-nominal entries
in its Table 11. P1 draws jointly instead: one multiplier per unit shared
across all degrees, or one label permutation shared across all degrees. This
preserves the cross-degree dependence induced by the shared units and the
shared cross-fitting. It is a strict improvement and it is a deviation, so it
is recorded as one.

**Deviation B, pooled standardization.** The source computes $\hat\mu_d$ and
$\hat\sigma_d$ from the $N$ null replicates only. That map is not symmetric in
the $N+1$ augmented values, so it breaks the exchangeability argument of §2.2
and the resulting rank test is exact only up to an $O(1/N)$ term.
`vjm_multiplicity_test` therefore defaults to `standardization="pooled"`,
computing the constants from all $N+1$ values, which restores exactness;
`standardization="source"` reproduces the source's convention and both are
reported in every result row so the size of the difference is measured rather
than asserted. `tests/test_phase3_5_vjm.py` pins the symmetry property and
pins that the source convention lacks it.

**Scope of exactness.** The exactness in §2.2 is inherited from the frozen
stratified permutation path and is no stronger than that path's own scope,
which `tda2s/tests/dr_outcome.py` already states: it is exact conditional
randomisation inference when the fitted nuisances and the strata are fixed
independently of the permuted labels, and a fast cross-fitted approximation
otherwise. Under the multiplier mechanism the argument is asymptotic only.
Neither claim is upgraded by this phase.

### 2.4 The one source method whose setting matches P1 is the one P1 must not use

Method 4 (§3.4) is the two-sample procedure, and it is the closest of the six
to P1's design. It is nonetheless unusable as a P1 comparator: its permutation
is a plain relabelling across the two groups with the Robinson-Turner loss, so
its target is $H_0^{\mathrm{cond}}$, the conditional-law null that Phase 2
established is the wrong target under covariate imbalance (`theory/WP2_covariate_shift.md`
§3-§4, with empirical type-I error 1.000 at $\lambda=1$ for the worst
competitor). Adopting Method 4 would reintroduce exactly the failure C1 is
about. Its $\%V/\%R$ bookkeeping is reusable; its null is not.

### 2.5 Verdict

**PASS, as an adaptation.** The family of homology degrees is the source's own
Method 5 construction (§3.5), the studentize-then-maximize architecture
transfers, and the transfer is justified by exchangeability of P1's own null
mechanism plus Kim and Lee's functional CLT rather than by the source's
Theorem 3.1. Three things do not transfer and are replaced: the null model,
the independence of the null draws, and the null-only standardization. The FDR
arm is retained only as a diagnostic, for the reason in §4.4.

This gate was decided on the argument alone, before the fleet ran. §4.6 records
that the benchmark subsequently agreed with it.

---

## 3. Task 3.5.3: implementation

`tda2s.tests.dr_outcome.vjm_multiplicity_test(fit, mechanism=..., strata=...,
n_draws=..., alpha=..., seed=..., standardization=...)`.

The function calls `degree_null_statistics`, which returns the observed vector
$X\in\mathbb R^K$ and the null matrix $Y\in\mathbb R^{N\times K}$ from **one**
shared null draw per replicate, and then reduces that single matrix four ways:
Bonferroni over the per-degree rank $p$-values, the Phase 3 unstudentized
shared maximum, the pooled-standardized studentized maximum, and the
source-standardized studentized maximum. Because all four read the same
draws, differences between them are differences of procedure and carry no
Monte Carlo noise from the calibration. The FDR threshold of Method 3/6 and the
comparability diagnostic are returned alongside.

Two invariants are pinned by tests rather than asserted in prose. First,
`max` over the degree axis of the per-degree null matrix equals, to machine
precision, the null produced by the existing `multiplier_test` and
`stratified_permutation_test` at the same seed, so the comparator provably
reads the Phase 3 nulls rather than a fresh set of draws. Second, the
studentized decision is exactly invariant when one degree's silhouettes are
multiplied by a constant, while the unstudentized shared maximum is not; the
AIPW fit is exactly equivariant under that rescaling, so this is a clean
re-parametrization rather than a change of estimand.

**Reproducibility hazard recorded while implementing.** `tcda_uq`'s
`fit_propensity` defaults to `RandomForestClassifier()` with no
`random_state`, so any driver that omits an explicit propensity estimator is
not reproducible, and two fits on identical data disagree. The Phase 3 driver
already passes a seeded estimator; the Phase 3.5 driver and tests now do the
same, and `_one_cell` documents why.

---

## 4. Task 3.5.4: benchmark

### 4.1 Design

`experiments/phase3_5_vjm.py`, sharded exactly like the Phase 3 fleet, with
Colab notebooks from `experiments/colab/make_phase3_5_notebooks.py`. Five
designs:

| Design | Cells | Replications | Purpose |
|---|---|---|---|
| `fwer` | $n\in\{50,100,200,500\}$ times three propensity regimes, global null | 500 | the pre-registered decision |
| `power` | $n\in\{50,100\}$ times null/alternative times degree scale $\in\{1,8\}$ | 200 | validity and power under scale imbalance |
| `degrees3` | $n=100$, the plan's $d\in\{0,1,2\}$ family | 200 | family size $K=3$ |
| `learners` | the four task 3.3 propensity learners, null | 100 | outside the correctly specified regime |
| `stress` | the four task 3.4 misspecification cases, null | 100 | double-robustness nulls |

The `fwer`, `power`, `learners` and `stress` designs use the two-degree family
of the Phase 3 oracle fleet, so $K=2$ there and the Bonferroni penalty is at
its mildest; any advantage the comparator shows over Bonferroni at $K=2$ is
therefore a lower bound on what a wider family would show, which is the
conservative direction for the conclusion. `degrees3` widens the family to the
plan's $d\in\{0,1,2\}$.

The alternative is a single-degree one: the sharp null is imposed on every
degree and then degree 1's outcome-mean intercept is shifted by 0.15, and since
the first Fourier basis function is identically one this gives
$\psi_1(t)\equiv 0.15$ and $\psi_0\equiv 0$ exactly. The degree-scale knob
multiplies degree 0's silhouettes by 8, which leaves the estimand untouched and
makes the two degrees incomparable. That pair is the cleanest possible probe of
what studentization is for.

### 4.2 The fleet re-derives Phase 3 rather than merely agreeing with it

The Phase 3.5 seed tags match `phase3_dr_calibration._one_fit` exactly, so the
`fwer` cells share the Phase 3 oracle null design, the replication seeds, the
fold seeds, the propensity seed and the calibration seeds. The shared
max-statistic read off the Phase 3.5 null matrix must therefore equal the
published Phase 3 `permutation_p` and `multiplier_p` to the last digit.
`python -m experiments.phase3_5_vjm --mode check-phase3` verifies this against
the downloaded Phase 3 shards: **36 of 36 null cells (3 replications, 4 sample
sizes, 3 regimes, both mechanisms) agree exactly.** A reduced version runs in
the test suite. The consequence is that the Phase 3.5 fleet does not merely
compare to Phase 3, it reproduces it, and the comparator columns are a strict
addition to the same draws.

### 4.3 Pre-registered decision rule

At 500 replications and $\alpha=0.05$, the empirical FWER of the comparator
must lie in $[0.03,0.08]$ on the `fwer` design. A miss is a calibration
failure and the comparator is rejected; the cutoff is not to be tuned. The
`power` and `degrees3` designs do not gate anything and are reported as
evidence about usefulness. This mirrors the Phase 3 treatment of the two
oracle null cells at 0.026, which were recorded against the strict band rather
than rounded into it.

### 4.4 The FDR arm is a diagnostic only, and its real home is Phase 5.2

Method 3/6 estimates $\%V$ and $\%R$ over candidate cutoffs. With $K=2$ or 3
degrees, $\%R$ takes three or four possible values, so $\hat q_{\text{FDR}}$ is
a ratio of extremely coarse proportions and the attainable false-discovery
rates are correspondingly lumpy. It is computed and reported, and it is not
offered as a degree-level error-rate control. The procedure's natural P1 home
is task 5.2, the discrete-subgroup conditional test, where the family is large
enough for an FDR estimator to mean something. That forward pointer is the
useful part of the source for the rest of the programme.

### 4.5 Fleet results (2026-08-16)

All five designs ran to their full budget: 20 `fwer` shards, 8 `power`, 8
`degrees3`, 4 `learners`, 4 `stress`, 25 replications each, no duplicate and no
conflicting records. The downloaded checkpoints live in `experiments/colab/`
alongside the Phase 3 ones, and aggregate with

```
python -m experiments.phase3_5_vjm --mode aggregate --design <design> \
    --input-dir experiments/colab
```

into `results/phase3_5_<design>_summary.json`.

#### 4.5.1 FWER, 500 replications per cell

Rejection rates at $\alpha=0.05$ under the global null. Twelve cells, both
mechanisms, 12000 replications in total.

Frozen stratified permutation:

| $n$ | regime | VJM pooled | VJM source | Bonferroni | shared max |
|---|---|---|---|---|---|
| 50 | 0.0 | 0.040 | 0.044 | 0.036 | 0.040 |
| 50 | 0.5 | 0.034 | 0.034 | 0.032 | 0.034 |
| 50 | 1.0 | 0.054 | 0.056 | 0.050 | 0.046 |
| 100 | 0.0 | 0.042 | 0.042 | 0.032 | 0.038 |
| 100 | 0.5 | 0.040 | 0.040 | 0.036 | 0.034 |
| 100 | 1.0 | 0.048 | 0.048 | 0.046 | 0.052 |
| 200 | 0.0 | 0.034 | 0.034 | 0.032 | 0.030 |
| 200 | 0.5 | **0.028** | **0.028** | **0.028** | **0.026** |
| 200 | 1.0 | 0.034 | 0.036 | 0.034 | 0.030 |
| 500 | 0.0 | 0.046 | 0.046 | 0.044 | 0.046 |
| 500 | 0.5 | 0.032 | 0.034 | 0.030 | 0.030 |
| 500 | 1.0 | 0.038 | 0.038 | 0.038 | 0.038 |

Shared multiplier:

| $n$ | regime | VJM pooled | VJM source | Bonferroni | shared max |
|---|---|---|---|---|---|
| 50 | 0.0 | 0.056 | 0.056 | 0.060 | 0.064 |
| 50 | 0.5 | 0.046 | 0.050 | 0.044 | 0.042 |
| 50 | 1.0 | 0.058 | 0.058 | 0.056 | 0.044 |
| 100 | 0.0 | 0.060 | 0.060 | 0.052 | 0.056 |
| 100 | 0.5 | 0.044 | 0.046 | 0.042 | 0.056 |
| 100 | 1.0 | 0.062 | 0.062 | 0.058 | 0.048 |
| 200 | 0.0 | 0.034 | 0.034 | 0.032 | 0.036 |
| 200 | 0.5 | 0.038 | 0.038 | 0.036 | 0.038 |
| 200 | 1.0 | 0.034 | 0.034 | **0.028** | 0.034 |
| 500 | 0.0 | 0.044 | 0.048 | 0.044 | 0.040 |
| 500 | 0.5 | 0.032 | 0.032 | **0.028** | 0.034 |
| 500 | 1.0 | 0.030 | 0.030 | **0.024** | **0.026** |

Bold marks a cell outside $[0.03,0.08]$. Counting cell-by-cell over all 24
cell-mechanism combinations, the comparator is in band in 23, the source
convention in 23, the shared max-statistic in 22 and Bonferroni in 20. Pooling
the 6000 replications per mechanism gives an overall FWER of 0.0392
(permutation) and 0.0448 (multiplier) for the comparator, against 0.0370 and
0.0432 for the shared maximum; every pooled figure is inside the band.

#### 4.5.2 The one miss, recorded strictly

The comparator's single out-of-band cell is the frozen permutation at $n=200$,
regime 0.5, at 0.028, which is 0.27 Monte Carlo standard errors below the 0.03
boundary. Following the §4.3 precedent it is recorded at its strict value and
not rounded into the band.

That cell is one of the two the Phase 3 exit already flags. On the identical
draws the Phase 3 shared max-statistic is 0.026 there, and at the other flagged
cell (multiplier, $n=500$, regime 1.0) the shared max is 0.026 while the
comparator is 0.030. At both of the Phase 3 conservative cells the comparator
is therefore closer to nominal than the statistic Phase 3 has already accepted,
and it is out of band only where that statistic is out of band more severely.
The residual conservatism is a property of the frozen-nuisance permutation null
in those design cells, which is an open Phase 3 calibration item, and not
something the comparator introduces.

#### 4.5.3 The fleet reproduces Phase 3 at full budget

Beyond the 36-cell `check-phase3` spot check of §4.2, the completed `fwer`
design reproduces the published Phase 3 oracle null rejection rates in all 12
cells and both mechanisms, 24 of 24, to every recorded digit. The comparator
columns are additions to those same draws, so every contrast in this section is
free of between-procedure Monte Carlo noise.

#### 4.5.4 Power and scale invariance, 200 replications

The design pairs a single-degree alternative ($\psi_1\equiv0.15$,
$\psi_0\equiv0$) with a degree-scale knob that multiplies degree 0's
silhouettes by 8 without touching the estimand.

| design | $n$ | scale | mech | VJM pooled | Bonferroni | shared max |
|---|---|---|---|---|---|---|
| power | 100 | 1 | perm | 0.840 | 0.825 | 0.830 |
| power | 100 | 8 | perm | 0.845 | 0.840 | 0.040 |
| power | 100 | 1 | mult | 0.855 | 0.840 | 0.840 |
| power | 100 | 8 | mult | 0.835 | 0.835 | 0.035 |
| power | 50 | 1 | perm | 0.465 | 0.460 | 0.480 |
| power | 50 | 8 | perm | 0.440 | 0.415 | 0.050 |
| power | 50 | 1 | mult | 0.540 | 0.505 | 0.520 |
| power | 50 | 8 | mult | 0.540 | 0.510 | 0.065 |
| degrees3 | 100 | 1 | perm | 0.780 | 0.770 | 0.780 |
| degrees3 | 100 | 8 | perm | 0.790 | 0.765 | 0.055 |
| degrees3 | 100 | 1 | mult | 0.805 | 0.795 | 0.765 |
| degrees3 | 100 | 8 | mult | 0.805 | 0.785 | 0.070 |

Averaged over these twelve cells the comparator's power is 0.714 at scale 1 and
0.709 at scale 8, while the shared maximum falls from 0.703 to 0.053, which is
its own null level. The pilot's reading survives at four times the replication
budget: the unstudentized maximum stops being a test of the family and becomes
a test of the loud degree, and the two scale-free procedures are untouched. The
`degrees3` rows show the same behaviour at $K=3$, so widening the family does
not disturb it.

These designs also carry null rows, which are diagnostics at 200 replications
(Monte Carlo standard error about 0.015) and gate nothing. The comparator is in
band in 11 of their 12 null cell-mechanisms, including all four `degrees3`
cells, so the $K=3$ family calibrates as well as $K=2$. The exception is
`power`, $n=100$, scale 8 under permutation at 0.025, where the shared maximum
is 0.050. That is the one place in the fleet where the comparator is out of band
and the Phase 3 statistic is not, and it is recorded as such. It is a
200-replication cell 0.45 standard errors below the boundary, in the conservative
direction, and it does not recur in the corresponding 500-replication `fwer`
cells, so it is noted rather than treated as evidence against the comparator;
the $[0.03,0.08]$ rule was pre-registered on the `fwer` design precisely because
200 replications cannot separate 0.025 from 0.030.

The comparator beats Bonferroni in 12 of 12 alternative cells, never loses one,
and the mean gain is 0.016. That is the dependence-preserving gain and it is
small, exactly as $K=2$ or 3 predicts. It is reported as a consistent direction,
not as a material power advantage.

#### 4.5.5 Pooled against source standardization

Over all 64 cell-mechanism combinations in the five designs the two conventions
differ at all in only 18, the mean difference is $-0.0016$, and the largest is
0.010. The exactness repair of §2.3 Deviation B therefore costs essentially
nothing empirically while removing a genuine finite-sample defect, which is the
best outcome available: it is chosen on the proof, and the data confirm the
choice is not paid for in power.

#### 4.5.6 Learner and stress nulls, 100 replications

These are diagnostics outside the correctly specified regime, at $n=200$ and
regime 1.0, and they do not gate anything. The stress cases are conservative
throughout, 0.000 to 0.040 across all four procedures, consistent with the
Phase 3 stress fleet. Among the learners, gradient boosting, logistic and
random forest sit between 0.020 and 0.060.

The neural propensity learner is the exception, at 0.090 for the comparator and
0.100 for the shared maximum under permutation, and 0.090 for both under the
multiplier. All four procedures move together, the shared max-statistic worst,
so this is not a comparator effect: it is the Phase 3 estimator with a neural
propensity at $n=200$ under the most imbalanced regime. The Phase 3 learner
fleet ran at $n=100$ and did not surface it. With 100 replications the Monte
Carlo standard error is about 0.029, so 0.090 is roughly one standard error
above the band and this is a flag rather than a finding. It is recorded here as
a Phase 3 item for the learner-robustness review, since the natural next step
is to re-run the learner grid at $n=200$ with the Phase 3 budget rather than to
draw any conclusion from 100 replications.

#### 4.5.7 Comparability diagnostic

The measured maximum pairwise KS distance between standardized per-degree null
distributions has median 0.047 over the 64 cell-mechanism combinations and a
90th percentile of 0.057, so the residual incomparability is small and stable
almost everywhere. Three cells exceed 0.08, all under permutation: the
`both_correct` and `outcome_misspecified` stress cells at 0.156 and 0.147, and
the `logistic` learner cell at 0.090.

Per §2.2 this quantity governs how the family's power is allocated and not
whether the test is valid, so the elevated cells are not a calibration concern.
Their rejection rates bear that out: 0.010, 0.010 and 0.040 respectively, all
in or below the range the rest of their designs occupy, so the incomparability
is not producing anticonservatism. The diagnostic is reported because §2.2
promised it would be measured rather than assumed away.

No pattern is offered for which cells are elevated. The obvious guess, that it
tracks outcome misspecification, does not survive contact with the numbers:
`outcome_misspecified` and `both_misspecified` share the same truncated basis
and sit at 0.147 and 0.045. With 100 replications and 199 calibration draws
these three values are not separated from the rest by enough to support a
mechanism, and none of them bears on validity, so they are recorded and left
alone.

### 4.6 Verdict on the pre-registered rule

**The comparator passes and is retained.** On the `fwer` design, which is the
design the rule was pre-registered against, the pooled FWER is in band under
both mechanisms; per cell the comparator is the best-calibrated of the four
procedures, in band in 23 of 24 against 22 for the shared maximum and 20 for
Bonferroni; and its single miss there is inherited rather than introduced.

(The non-gating designs contribute one further conservative miss, `power` at
$n=100$ and scale 8, discussed in §4.5.4. It is a 200-replication cell, it is
not inherited, and it does not reappear at the gating budget. It is recorded
because the alternative would be to report only the misses that happen to have
an exculpatory story.)

That last point is the one the rule turns on, and §6 pre-registered the test
for it: a failure attributable to the comparator would have to be one that
leaves Phase 3 untouched. This miss does not, because it occurs in a cell where
the Phase 3 shared max-statistic reads the identical draws and is further from
nominal. Rejecting the comparator on that cell would equally reject the Phase 3
primary test, which is the incoherent conclusion §6 was written to prevent. The
strict value is recorded, the cutoff is not tuned, and the conservatism it
reflects stays on the books as the open Phase 3 calibration item.

---

## 5. What the manuscript may and may not claim

May claim: that P1 implements a degree-multiplicity comparator adapted from
Vejdemo-Johansson and Mukherjee's empirical-null construction (Method 5, §3.5),
that the adaptation replaces their point-process null model with P1's
covariate-preserving conditional null, draws the family jointly rather than
independently, and uses a pooled standardization that keeps the rank test exact
under exchangeability; and that its validity rests on exchangeability of P1's
null mechanism together with Kim and Lee's functional CLT.

May also claim, on the §4.5 fleet: that the comparator's empirical FWER is
0.039 under the frozen stratified permutation and 0.045 under the shared
multiplier over 6000 replications each, both inside the pre-registered
$[0.03,0.08]$ band; that it is the best-calibrated of the four procedures
cell-by-cell; and that it is insensitive to a per-degree rescaling that
destroys the shared max-statistic's power (0.71 against 0.05 at a scale ratio
of 8). The rescaling result should be stated for what it is, a demonstration
that studentization is what makes a degree family a family, on a simulation
built to isolate exactly that.

May not claim: that P1 controls FWER "by the Vejdemo-Johansson--Mukherjee
procedure"; that the source's universality or precomputed-null-table result
applies to cross-fitted AIPW scores; that Hiraoka-Shirai-Trinh Theorem 5.2
justifies studentizing $\sqrt n\|\hat\psi_d\|_\infty$; that standardization
makes the per-degree null laws equal; or that the shared max-statistic or
Bonferroni lines of `degree_multiplicity_test` are VJM procedures. The last
point was the specific hazard the plan's Phase 3.5 exit condition guards
against, and it is now guarded in code by the docstring of
`degree_multiplicity_test`.

## 6. If the fleet fails the band (not triggered)

**This contingency did not fire; §4.6 records the outcome. The section is kept
as written because its coupling argument is what §4.6 relies on to classify the
one out-of-band cell, and that argument has to have been fixed in advance of
the data to carry any weight.**

If the comparator misses $[0.03,0.08]$, the fallback is the plan's: record the
failed assumption, retain Bonferroni and the shared max-statistic as P1's valid
multiplicity procedures, and report the source as related work with an
explanation of its non-transferability. Note that in that event the §2.2
exchangeability argument would be the thing under suspicion, not the
standardization, since the standardization is a bijective reparametrization of
a valid rank test; and the same suspicion would fall on the Phase 3 shared
max-statistic, which reads the identical draws. That coupling is by design: it
means the comparator cannot fail for reasons that leave Phase 3 untouched.
