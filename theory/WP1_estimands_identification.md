# WP1 — Estimands and Identification (Phase 1 deliverable)

Status: drafted 2026-08-14, after Phase 0 PASS. Plan: `RESEARCH_PLAN_P1_TwoSample.md` §Phase 1.

Exit audit of Phase 1: every null has a stated identification condition and a stated failure
mode; task 1.3 (the marginal-identification question) is answered. **Gate 1.3 → PASS**: the
paper's targets need no change, but the headline estimand must be named as the
*covariate-standardized* topological effect (§4).

Sources read in full for this work package: Souto & Diamantis, *A Mathematical Framework for
TCDA*, arXiv:2607.28161 (identification, distribution level, topological ignorability, Theorem
5.9); Saki & Faghihi, arXiv:2603.14169v2 (non-identification of the marginal effect, Propositions
3 and 6, Theorem 5, Remarks 4 and 7, Example 8); Faghihi, arXiv:2606.01184 (injectivity); Kim &
Lee, arXiv:2603.02289 (silhouette definition, Lemma 2.1, identification conditions C1–C3). All
citations carry DOI/URL per `Literature_Review/VERIFIED.bib`.

---

## 1. Task 1.1 — The six nulls, stated with measure-theoretic care

### 1.1 Setup and notation

Let $(X_i, A_i, Y_i)_{i=1}^n$ be i.i.d. copies of $(X, A, Y)$ on $\mathcal X \times \{0,1\}
\times \mathcal Y$, where $\mathcal Y$ is a Polish space of point clouds, $X$ are covariates,
$A = \mathbb 1\{\text{group 1}\}$ with propensity $e(x) = P(A=1 \mid X=x)$, and $Y^a$ are
potential outcomes satisfying consistency $Y = Y^A$. Fix a filtration $F$, a homology degree
$d$, and a representation $\Phi$ with values in a separable Banach space $B$; the default is
the power-weighted silhouette on a fixed compact interval $T \subset \mathbb R_+$,

$$\phi(t; D) = \frac{\sum_{p \in D} w_p \Lambda_p(t)}{\sum_{p \in D} w_p}, \qquad w_p = (b_p - a_p)^r, \qquad \Lambda_p(t) = \max\{0, \min\{t - a_p, b_p - t\}\},$$

with $p = (a_p, b_p)$ running over the diagram $D = \mathrm{Dgm}_d(F(Y))$ (Kim & Lee, eq. 2).
Write $D^a_{i,d} = \mathrm{Dgm}_d(F(Y^a_i))$, $Z^a_i = \Phi(D^a_{i,d})$, and let $P^a_Y$ denote
the interventional law of $Y^a$. For a distribution-level representation $T_{\mathrm{dist}}$
(mapping laws of clouds to a summary space), set $\Theta^a_{\mathrm{dist}} = T_{\mathrm{dist}}
(P^a_Y)$ and $\delta_{\mathrm{dist}} = d(\Theta^1_{\mathrm{dist}}, \Theta^0_{\mathrm{dist}})$
for a fixed metric $d$. Write the mean functional $A(P) = \int \Phi(\mathrm{Dgm}_d(F(y))) \,
dP(y)$, so that $A(P^a_Y) = \mathbb E[Z^a]$; $A$ is affine in $P$ (Bochner expectation).

### 1.2 The six nulls

| Label | Statement (measurable form) | Object being compared |
|---|---|---|
| $H_0^{\mathrm{cond}}$ | $\mathcal L(D \mid A=1) = \mathcal L(D \mid A=0)$ | Observational conditional laws of diagrams; the field's null, confounded by $X$ |
| $H_0^{\mathrm{out}}$ | $\psi_d(t) := A(P^1_Y)(t) - A(P^0_Y)(t) = 0$ for all $t \in T$ | Outcome-level TATE: means of potential silhouettes (Kim & Lee) |
| $H_0^{\mathrm{dist}}$ | $\delta_{\mathrm{dist}} = d(T_{\mathrm{dist}}(P^1_Y), T_{\mathrm{dist}}(P^0_Y)) = 0$ | Distribution-level: topological representations of the interventional laws |
| $H_0^{\mathrm{ctate}}$ | $\tau_d(t,x) := \mathbb E[\delta_{i,d}(t) \mid X=x] = 0$ for $P_X$-a.e. $x$, all $t$ | Conditional effect; $\delta_{i,d} = Z^1_i - Z^0_i$ is the individual topological treatment effect |
| $H_0^{\mathrm{het}}$ | $x \mapsto \tau_d(t,\cdot)$ is constant in $x$ ($P_X$-a.e.), all $t$ | Heterogeneity omnibus |
| $H_0^{\mathrm{equiv}}$ | $\sup_{t \in T} \lvert \psi_d(t) \rvert \ge \delta$ for a prespecified margin $\delta > 0$ | Equivalence: certifies topological sameness up to $\delta$ |

Each null is a statement about laws, not about the sample. $H_0^{\mathrm{cond}}$ is directly
testable from observational data with no identification step. The other five are statements
about potential-outcome quantities, so their testability is governed by the identification
results of §3; the failure mode of each null is recorded in the exit table of §7.

### 1.3 The relation between $H_0^{\mathrm{dist}}$ and $H_0^{\mathrm{out}}$ is pairing-conditional

The plan's §1 claims $H_0^{\mathrm{dist}} \Rightarrow H_0^{\mathrm{out}}$ with false converse.
With full care, that claim is true only for specific pairings of $(\Phi, T_{\mathrm{dist}})$, and
false for the default silhouette pairing. This is a genuine correction to the plan, made in
advance of the manuscript so the paper's Figure 1 and its "two tests" framing are stated
correctly.

**Theorem 1 (the implication is pairing-conditional).** Let $A$ be the mean functional and
suppose $A(P) = L(T_{\mathrm{dist}}(P))$ for a fixed measurable map $L$ (the outcome-level mean
factors through the distribution-level representation). Then $H_0^{\mathrm{dist}} \Rightarrow
H_0^{\mathrm{out}}$. Two concrete regimes:

1. *Canonical measure-valued pairing.* Take $\Phi(D) = \mu_D = \sum_{p \in D} w_p \,
\delta_{\mathrm{coord}(p)}$ (un-normalized weighted persistence measure, e.g. in Divol–Lacombe
coordinates $(b, (b+d)/2)$) and $T_{\mathrm{dist}}(P) = \mathbb E_P[\mu_{D(F(Y))}]$ (expected
persistence measure). Then $A = T_{\mathrm{dist}}$ identically, so $H_0^{\mathrm{dist}}$ and
$H_0^{\mathrm{out}}$ coincide. This is the "linear-friendly" regime advertised in Phase 4.1:
expectation commutes with the un-normalized measure summary, and there is no normalization
ratio to break linearity.
2. *General factorization.* If $A = L \circ T_{\mathrm{dist}}$, the implication holds by
composition; the converse fails whenever $L$ or $T_{\mathrm{dist}}$ is non-injective (equal
representations do not determine equal means in general).

**Theorem 2 (default silhouette pairing: the two nulls are logically independent).** Take
$\Phi$ = normalized power-weighted silhouette and $T_{\mathrm{dist}}(P)$ = expected persistence
measure. Then neither implication holds:

*W1, the counterexample to the converse (the plan's cluster-splitting witness).* Let
$d > 0$ be fixed. Under $L(D^0) = \delta_{(0,d)}$ (one persistent $H_0$ class) and $L(D^1) =
\delta_{(0,d)} + \delta_{(0,d)}$ (two classes, same merge scale), the mean silhouettes are equal
exactly, because the silhouette is a normalized average and the two tents coincide, while the
expected persistence measures differ ($\delta_{(0,d)}$ vs $2\delta_{(0,d)}$). Hence $H_0^{\mathrm{out}}$
holds and $H_0^{\mathrm{dist}}$ fails.

*W2, the counterexample to the implication (the multiplicity mixture).* Let $L(D^0) =
\delta_{(0,1)}$ and $L(D^1)$ draw $\{(0,1),(0,1)\}$ with probability $1/2$ and the empty
diagram with probability $1/2$. Then the expected persistence measures are equal (both
$\delta_{(0,1)}$), while the mean silhouettes differ by half a tent. Hence $H_0^{\mathrm{dist}}$
holds and $H_0^{\mathrm{out}}$ fails. W2 is the normalized-summary manifestation of the
covariate-mixing non-commutation of Saki & Faghihi (Remark 4): topology is applied per cloud
inside the expectation, and the normalized ratio prevents the expectation from commuting with
the mixture.

**Proof of Theorem 1.** Case 1 is the identity $A(P^a) = \mathbb E[\mu_{D^a}] = T_{\mathrm{dist}}
(P^a)$. Case 2: $A(P^1) = L(T_{\mathrm{dist}}(P^1)) = L(T_{\mathrm{dist}}(P^0)) = A(P^0)$.
The W1 and W2 witnesses are verified by direct computation of the tent average and the measure
expectation; both are instantiated numerically in §1.4.

**Connection to Souto & Diamantis Theorem 5.9.** The two *contrasts* $\Delta_{\mathrm{out}} =
A(P^1) - A(P^0)$ and $\Delta_{\mathrm{dist}} = T_{\mathrm{dist}}(P^1) - T_{\mathrm{dist}}(P^0)$
agree for all laws if and only if $T_{\mathrm{dist}} - A$ is constant. In the default pairing
$T_{\mathrm{dist}} - A$ is not constant (W1 and W2 are both divergence pairs), which is the
framework's own certificate that the two nulls are distinct objects. The manuscript should
state Theorems 1 and 2 with this citation and should not print the unhedged sentence
"$H_0^{\mathrm{dist}} \Rightarrow H_0^{\mathrm{out}}$" from the plan's §1; the correct,
still-unhedged sentence is pairing-conditional as above.

### 1.4 Numerical instantiation (exit check for task 1.1)

`scripts/wp1_counterexample.py` (runs the PH pipeline, silhouettes, persistence measures and
permutation tests from `tda2s`); the DGP `split_cluster_cloud` is added to `tda2s/dgp/clouds.py`
and is the Phase 4.4 separation DGP. Results at $n=300$ clouds per group, 300 permutations,
10 Monte Carlo replications:

| Check | Statistic | Result | Reading |
|---|---|---|---|
| W1, law level | mean-silhouette gap | 0.0096 (1.9 null SDs) | $H_0^{\mathrm{out}}$ holds; gap is group-mean Monte Carlo noise |
| W1, law level | expected-measure distance | 1.350 | $H_0^{\mathrm{dist}}$ fails, at a scale far above noise |
| W1, law level | outcome-level test rejection | 1/10 at 5% | size held (expect 0.5) |
| W1, law level | distribution-level test rejection | 10/10 at 5% | power held |
| W2, law level | mean-silhouette gap | 0.356 (24 null SDs) | $H_0^{\mathrm{out}}$ fails |
| W2, law level | expected-measure distance | 0.05 | $H_0^{\mathrm{dist}}$ holds; distance is Monte Carlo noise |
| W2, law level | outcome-level rejection | 10/10 | power held |
| W2, law level | distribution-level rejection | 1/10 | size held |
| Cloud exact (deterministic blobs + threshold $\tau = 0.3$) | H0 counts | 1 vs 2 | the split is realized by the PH pipeline |
| Cloud exact | H0 deaths | 1.350 both groups | equal merge scale exactly |
| Cloud exact | mean-silhouette gap | 0.000e+00 | mean preserved, realization by realization |
| Cloud exact | expected-measure distance | 1.350 | distribution-level separation |
| Cloud Gaussian (diagnostic) | gap / null SD | 3.2 | noisy geometry breaks exactness, see remark |

The deterministic cloud construction: each blob is a regular 12-gon of radius 0.15; two blobs
(one $H_0$ class, death $(3.0 - 0.3)/2 = 1.35$) versus three blobs in an equilateral
arrangement (two classes, both dying at 1.35). Persistence thresholding at $\tau = 0.3$ removes
the tiny within-blob classes (death 0.039) that exist in both groups; without it the mean
silhouette equality is only approximate at the $10^{-4}$ level. With noisy Gaussian blobs the
two classes become the two smallest order statistics of three pair-merge scales, biasing the
mean silhouette (gap 3.2 null SDs in the diagnostic), so Phase 4.4 must use the deterministic
DGP or tightly-clustered thresholded diagrams for its "power equals alpha" claim. The
diagnostic's RT p-value 0.003 confirms that Robinson–Turner-style tests see this bias, not the
multiplicity.

---

## 2. Task 1.2 — g-formula identification of $\psi_d$ and $\delta_{\mathrm{dist}}$

All results of this section are reused from Souto & Diamantis §§4–5 (Theorems 4.1, 5.3,
Corollary 5.4, Proposition 5.6), which in turn restate and extend Kim & Lee (C1–C3). Define the
outcome regression $m_a(x) = \mathbb E[Z \mid A = a, X = x]$ (Bochner-valued) and $e_a(x) =
P(A = a \mid X = x)$.

**Outcome level (identification of $H_0^{\mathrm{out}}$).** Assume consistency, conditional
exchangeability $Y^a \perp A \mid X$ for $a \in \{0,1\}$ (arm-specific, i.e. *weak* conditional
exchangeability, suffices), and strict positivity $0 < e(x) < 1$. If $\mathbb E \lVert Z^a
\rVert_B < \infty$ then (Theorem 4.1)

$$\mathbb E[Z^a] = \mathbb E[m_a(X)] = \mathbb E\Big[\frac{\mathbb 1\{A=a\}}{e_a(X)} Z\Big],$$

so $\psi_d = \mathbb E[m_1(X) - m_0(X)]$ is identified from the observational law. Strict
positivity suffices for the population identities; uniform inverse-weight bounds are needed
only for the asymptotic arguments of Phase 3. The augmented representation $\Psi_a(\hat m_a,
\hat e_a)$ (Proposition 4.3) satisfies the double-robustness identity $\Psi_a(\hat m_a, \hat e_a) -
\mathbb E[Z^a] = \mathbb E[(\hat e_a - e_a)(\hat m_a - m_a)/\hat e_a]$, so the AIPW pseudo-outcome
is exact if either nuisance is correct, and its remainder is the product rate $\lVert \hat e - e
\rVert \cdot \lVert \hat m - m \rVert$, the load-bearing object for Phase 3.4's stress test.

**Distribution level (identification of $H_0^{\mathrm{dist}}$).** Let $Q_a(x, \cdot) =
\mathcal L(Y \mid A = a, X = x)$ be the observed conditional outcome kernel, defined $P_X$-a.e.
under strict positivity. Under consistency, joint conditional exchangeability $(Y^0, Y^1) \perp
A \mid X$, and strict positivity (Theorem 5.3), the interventional laws satisfy the law-valued
g-formula

$$P^a_Y = \int_{\mathcal X} Q_a(x, \cdot) \, dP_X(x), \qquad a \in \{0,1\}.$$

If $T_{\mathrm{dist}}$ is known and well-defined at $P^0_Y$ and $P^1_Y$, then $\Theta^a_{\mathrm{dist}}$,
$\Delta_{\mathrm{dist}}$ and $\delta_{\mathrm{dist}}$ are identified (Corollary 5.4). Two
remarks are load-bearing for the manuscript. First, topology is applied *after* the mixture:
$\mathrm{Dgm}(F(\cdot))$ does not commute with mixing over $X$ (Remark 5.7; Saki & Faghihi
Remark 4), so the plug-in estimator must form the mixture first and then transform, never the
reverse; W2 of §1.3 is the normalized-summary case of this failure. Second, continuity of
$T_{\mathrm{dist}}$ is not needed for identification but is needed for stable estimation
(Corollary 5.4 note); Phase 4's plugin consistency argument therefore requires a continuity
or stability-transfer condition, exactly the object Souto & Diamantis license.

**Conditional level (identification of $H_0^{\mathrm{ctate}}$ and $H_0^{\mathrm{het}}$).**
Under the same conditions, the conditional effect is identified as $\tau_d(t,x) = \mathbb E[
m_1(X) - m_0(X) \mid X = x]$, and equivalently as the conditional mean of the DR pseudo-outcome
$\rho_d(t, Z; \eta) = m_1(t,X) - m_0(t,X) + (A/e(X))(Z(t) - m_1(t,X)) - ((1-A)/(1-e(X)))(Z(t)
- m_0(t,X))$: $\mathbb E[\rho_d(t, Z; \eta) \mid X = x] = \tau_d(t,x)$. This is the DR-learner
property (Kennedy 2023) that Phase 5's pseudo-outcome construction consumes; the
distribution-level conditional target $\tau_{\mathrm{dist}}$ is identified by the conditional
g-formula $P_{a,x} = Q_a(x,\cdot)$ $P_X$-a.s. (Proposition 5.6).

---

## 3. Task 1.3 — the marginal-identification question (the gate)

**Answer: no. The marginal persistence-diagram effect is not identified from conditional
topological ignorability alone.** Verified in Phase 0.7 (abstract verbatim) and re-read in full
text for this work package: Saki & Faghihi's abstract states exactly this, their Remark 7
("Proposition 6 identifies the transformed conditional laws, not the marginal interventional
law itself... conditional topological ignorability does not identify $\Theta_\Psi$"), their
Example 8 (two mixtures with identical conditional summaries but different marginal topology),
and Souto & Diamantis Proposition 8.2(3) and §8. The mechanism is the same non-commutation of
persistent homology with covariate mixtures that appears throughout this document: conditional
topological ignorability fixes $T_{\mathrm{dist}}(P_{a,x}) = T_{\mathrm{dist}}(P^{\mathrm{fact}}_{a,x})$
stratum by stratum, and the marginal law is a $P_X$-mixture of the stratum laws, whose topology
depends on the geometric alignment of the strata, which the identifying condition does not
constrain.

**Consequence for the paper's headline estimand.** The paper must target the
*covariate-standardized* topological effect and say so plainly:

1. $H_0^{\mathrm{out}}$ already is a standardization: $\psi_d = \mathbb E[m_1(X) - m_0(X)]$
   (Theorem 4.1), so no change to the outcome-level null; it is identified under the standard
   A1 conditions (consistency, weak conditional exchangeability, positivity), which the paper
   and Kim & Lee assume.
2. $H_0^{\mathrm{dist}}$ is identified under *joint* conditional exchangeability via the
   g-formula (Corollary 5.4); that is the paper's working assumption and must be stated as
   such. Where a reader grants only conditional topological ignorability, the identified
   distribution-level object is the standardized within-stratum contrast
   $\tau_{\mathrm{dist}} = \mathbb E_X[d(T_{\mathrm{dist}}(Q_1(X,\cdot)), T_{\mathrm{dist}}
   (Q_0(X,\cdot)))]$ (Proposition 5.6), and the paper must not present $\delta_{\mathrm{dist}}$
   as identified in that regime. One honest sentence in the assumptions subsection discharges
   this, and the unmeasured-confounding sensitivity of Phase 9.3 is its natural home.
3. $H_0^{\mathrm{ctate}}$ and $H_0^{\mathrm{het}}$ are conditional targets and remain
   identified under topological ignorability at the summary level (Saki & Faghihi Proposition
   6), with the semantic caveat of §4.

**Gate verdict: PASS.** Phases 3–5 estimate $\psi_d$, $\delta_{\mathrm{dist}}$ and $\tau_d$
under A1 exactly as planned; no estimand change, only a naming obligation.

---

## 4. Task 1.4 — injectivity and the identification regimes of the six nulls

Faghihi (Theorem 5): if the chosen summary is injective on the model class, conditional
topological ignorability is *equivalent* to weak conditional exchangeability; for non-injective
summaries it is strictly weaker, and then only the structural feature of interest is
identified, not the full interventional law. Regime per null:

| Null | Identification regime | What is identified under the regime's assumptions |
|---|---|---|
| $H_0^{\mathrm{cond}}$ | observational | The conditional laws themselves; no causal burden. This is the field's null and the object Phase 2 shows to be the wrong target under imbalance |
| $H_0^{\mathrm{out}}$ | mean summary in $B$; weak conditional exchangeability + consistency + positivity | $\psi_d$, the mean of the chosen summary; injectivity of $\Phi$ is *not* required (a mean of a non-injective summary is still identified; only the mean is claimed) |
| $H_0^{\mathrm{dist}}$ | law level; joint conditional exchangeability + positivity, $T_{\mathrm{dist}}$ well-defined on $P^a_Y$ | $\Theta^a_{\mathrm{dist}}$ and hence $\delta_{\mathrm{dist}}$; under topological ignorability alone only $\tau_{\mathrm{dist}}$ (see §3). Semantic caveat: for non-injective $T_{\mathrm{dist}}$, $\delta_{\mathrm{dist}} = 0$ means equality of the selected representations, not equality of the interventional laws |
| $H_0^{\mathrm{ctate}}$ | conditional; weak exchangeability (pointwise $x$) | $\tau_d(t,x)$; under topological ignorability, the conditional contrast of the *transformed* laws (Saki & Faghihi Proposition 6), which is a summary-level quantity and must be labelled as such |
| $H_0^{\mathrm{het}}$ | as $H_0^{\mathrm{ctate}}$ | dependence of $\tau_d$ on $x$; the null is a statement about a function, so its testability additionally requires consistent estimation of $\tau_d(t,\cdot)$ over $x$ (Phase 5) |
| $H_0^{\mathrm{equiv}}$ | as $H_0^{\mathrm{out}}$ | $\sup_t \lvert \psi_d(t) \rvert$ against a prespecified margin |

---

## 5. Task 1.5 — scope limits: why not $(\mathcal D_p, W_p)$ directly

The referee objection this section pre-empts: "your two-sample problem lives naturally in
diagram space, why do you move to Banach summaries?" The answer is four-fold, each item cited:

1. $(\mathcal D_p, W_p)$ is complete and separable (Mileyko, Mukherjee & Harer 2011), so a
   Fréchet two-sample apparatus exists in principle, but Fréchet means in diagram space are
   non-unique (Mileyko et al. 2011; Turner et al. 2014 find only local minima), which makes
   mean-based tests ambiguous.
2. The geometry of $(\mathcal D_2, W_2)$ is nonnegatively curved and infinite-dimensional in
   every standard sense (Che, Galaz-García, Guijarro & Membrillo Solis, JACT 2024). The
   Fréchet-ANOVA CLT of Dubey & Müller (2019) assumes curvature conditions this space
   violates, a remark worth printing when the competitor appears in Phase 7's table.
3. Naive bootstrap resampling is inconsistent for persistent Betti numbers (Roycraft, Krebs &
   Polonik, AoS 2023); any Betti- or Euler-curve statistic in this paper must use the smoothed
   bootstrap, and Phase 4.3 will say so wherever a Betti count enters.
4. Banach summary space and persistence-measure space carry the machinery the paper needs:
   the power-weighted silhouette is 1-Lipschitz in $W_1$ (Kim & Lee Lemma 2.1 and Theorem 5.3),
   averages of landscapes and silhouettes satisfy SLLN and CLT (Bubenik 2015; Chazal, Fasy,
   Lecci, Rinaldo & Wasserman 2014), and persistence measures (Divol & Lacombe 2021) give a
   linear-friendly, Fréchet-pathology-free home for the distribution-level null (Theorem 1,
   case 1).

Consequently the paper does inference in $B$ (silhouettes, landscapes) or in persistence-measure
space (expected measure), never in $(\mathcal D_p, W_p)$ directly; Fréchet means in diagram
space are at most a robustness check, and only where their non-uniqueness does not affect the
conclusion.

---

## 6. Implementation remarks for Phases 2–5

1. **gudhi bottleneck degeneracy.** `gudhi.bottleneck_distance` is pathologically slow (tens
   of ms per call) on identical diagrams, where the optimal distance is zero (CGAL degenerate
   path). Robinson–Turner's O(N²) pairwise loop then hangs for minutes, as discovered while
   instantiating the deterministic witness. Phase 2.3's imbalance sweep and Phase 7's grid can
   hit near-identical diagrams (e.g. small effects, low $n$); benchmark wrappers must either
   skip zero-cost pairs or cap diagram sizes before this call. Recorded in the WP1 script
   docstring.
2. **Exact mean-preserving DGPs need deterministic geometry or thresholding.** Noisy blob
   geometry turns the two $H_0$ classes of the split group into order statistics of three
   pair-merge scales, biasing the mean silhouette (3.2 null SDs at the diagnostic's settings).
   Phase 4.4's "Phase 3 has power = $\alpha$" claim must use `split_cluster_cloud(...,
   deterministic=True)` with the persistence threshold, exactly as in `wp1_counterexample.py`.
3. **The RT wrapper is blind to multiplicity in this regime.** With equal merge scales,
   bottleneck distances are zero within groups and positive across, so the observed within-loss
   is the minimal possible value and the Robinson–Turner within-statistic cannot reject
   (analytically p = 1; it was not run on the exact witness because of remark 1). This is
   itself an argument for C2 and should appear in Phase 4.4's narrative.
4. **W2 is the Phase 2.2 masking template.** The multiplicity-mixture witness is a diagram-law
   version of the covariate-mixture masking that Phase 2.2 builds at the cloud level; the two
   phases share the mechanism (normalization ratios inside the expectation).

---

## 7. Exit audit — identification conditions and failure modes per null

| Null | Identification condition | Failure mode (when the test is misleading) |
|---|---|---|
| $H_0^{\mathrm{cond}}$ | none (observational laws) | fires on covariate shift even with no topological effect; silent under masking. This is the Phase 2 gate's target |
| $H_0^{\mathrm{out}}$ | consistency + weak conditional exchangeability + strict positivity + $\mathbb E \lVert Z^a \rVert < \infty$ | hidden confounding (the g-formula mixture is the wrong one); non-overlap (unidentified on deterministic-treatment regions); representation drift (only the chosen summary's mean is claimed) |
| $H_0^{\mathrm{dist}}$ | joint conditional exchangeability + positivity; $T_{\mathrm{dist}}$ well-defined on $P^a_Y$; continuity of $T_{\mathrm{dist}}$ for estimation | under topological ignorability alone the marginal object is unidentified (§3); non-continuous $T_{\mathrm{dist}}$ gives identification without stable estimation; non-injective $T_{\mathrm{dist}}$ makes the null about representations, not laws |
| $H_0^{\mathrm{ctate}}$ | as $H_0^{\mathrm{out}}$, pointwise; fixed-$x$ positivity | weak overlap at $x$; under topological ignorability the target downgrades to the transformed-law contrast (§4) |
| $H_0^{\mathrm{het}}$ | as $H_0^{\mathrm{ctate}}$ | nuisance estimator of $\tau_d(t,\cdot)$ inconsistent in $x$ (the omnibus then fires on estimation error); multiplicity across $x$ and $t$ uncontrolled |
| $H_0^{\mathrm{equiv}}$ | as $H_0^{\mathrm{out}}$ | margin $\delta$ chosen post hoc; multiplicity across $t$ and degrees |

**Deliverable met:** `theory/WP1_estimands_identification.md` (this file), with numerical
evidence in `scripts/wp1_counterexample.py` and the reusable DGP in `tda2s/dgp/clouds.py`.
Gate 1.3 answered: the headline is the covariate-standardized topological effect; no estimand
changes for Phases 3–5.
