# Phase 3.5: Vejdemo-Johansson--Mukherjee mapping note

**Status (2026-08-16): audit complete, verdict PASS with three recorded
deviations, comparator implemented, benchmark fleet generated and pending
execution.** Tag: `adapt`. The comparator is `vjm_multiplicity_test` in
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

### 4.5 Pilot evidence, and what is still pending

A 50-replication pilot at $n_{\text{calibration}}=199$, run locally before the
fleet was sized, gives the intended separation. Rejection rates at
$\alpha=0.05$ under the frozen stratified permutation, single-degree
alternative with shift 0.15:

| $n$ | degree scale | shared max | VJM pooled | Bonferroni |
|---|---|---|---|---|
| 50 | 1 | 0.50 | 0.50 | 0.44 |
| 50 | 8 | 0.08 | 0.50 | 0.44 |
| 100 | 1 | 0.76 | 0.76 | 0.76 |
| 100 | 8 | 0.06 | 0.80 | 0.78 |

The reading is that the unstudentized shared maximum collapses to near its
level as soon as the degrees are on different scales, because it becomes a test
of the loud degree only, while both scale-free procedures are unaffected.
Bonferroni is also scale-free, so the comparator's advantage over Bonferroni is
the small dependence-preserving gain visible in the table (0.50 against 0.44,
0.80 against 0.78) rather than anything dramatic, and at $K=2$ that is exactly
what should be expected. **This is a pilot at 50 replications, with Monte Carlo
standard errors around 0.07, and it decides nothing.** The pre-registered
decision waits on the fleet.

**Pending:** run the eight notebooks in `experiments/colab/`
(`phase3_5_fwer_nb_00..03`, `phase3_5_power_nb_00`, `phase3_5_degrees3_nb_00`,
`phase3_5_learners_nb_00`, `phase3_5_stress_nb_00`), drop the downloaded shard
JSON files into `results/phase3_5_shards/`, and aggregate with
`python -m experiments.phase3_5_vjm --mode aggregate --design <design>`. Then
record the FWER table here and fire the §4.3 rule.

---

## 5. What the manuscript may and may not claim

May claim: that P1 implements a degree-multiplicity comparator adapted from
Vejdemo-Johansson and Mukherjee's empirical-null construction (Method 5, §3.5),
that the adaptation replaces their point-process null model with P1's
covariate-preserving conditional null, draws the family jointly rather than
independently, and uses a pooled standardization that keeps the rank test exact
under exchangeability; and that its validity rests on exchangeability of P1's
null mechanism together with Kim and Lee's functional CLT.

May not claim: that P1 controls FWER "by the Vejdemo-Johansson--Mukherjee
procedure"; that the source's universality or precomputed-null-table result
applies to cross-fitted AIPW scores; that Hiraoka-Shirai-Trinh Theorem 5.2
justifies studentizing $\sqrt n\|\hat\psi_d\|_\infty$; that standardization
makes the per-degree null laws equal; or that the shared max-statistic or
Bonferroni lines of `degree_multiplicity_test` are VJM procedures. The last
point was the specific hazard the plan's Phase 3.5 exit condition guards
against, and it is now guarded in code by the docstring of
`degree_multiplicity_test`.

## 6. If the fleet fails the band

If the comparator misses $[0.03,0.08]$, the fallback is the plan's: record the
failed assumption, retain Bonferroni and the shared max-statistic as P1's valid
multiplicity procedures, and report the source as related work with an
explanation of its non-transferability. Note that in that event the §2.2
exchangeability argument would be the thing under suspicion, not the
standardization, since the standardization is a bijective reparametrization of
a valid rank test; and the same suspicion would fall on the Phase 3 shared
max-statistic, which reads the identical draws. That coupling is by design: it
means the comparator cannot fail for reasons that leave Phase 3 untouched.
