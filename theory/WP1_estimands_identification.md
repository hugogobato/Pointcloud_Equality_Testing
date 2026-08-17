# WP1 — Estimands and Identification (Phase 1 deliverable)

Status: drafted 2026-08-14, after Phase 0 PASS; audited and revised 2026-08-14.
Plan: `RESEARCH_PLAN_P1_TwoSample.md` §Phase 1.

Exit audit of Phase 1: every null has a stated identification condition and a stated failure
mode; task 1.3 (the marginal-identification question) is answered. **Gate 1.3 → PASS**: the
paper's targets need no change, but the headline estimand must be named as the
*covariate-standardized* topological effect (§4).

Sources read for this work package: Souto & Diamantis, *A Mathematical Framework for TCDA*,
arXiv:2607.28161 (identification, distribution level, topological ignorability; Theorems 4.1,
5.3, 5.9, Corollary 5.4, Definition 5.5, Propositions 4.3, 5.6, 8.2, Remarks 5.7) — statements
and hypotheses checked against the manuscript source, not from memory; Saki & Faghihi,
arXiv:2603.14169v2 (non-identification of the marginal effect; Propositions 3 and 6, Theorem 5,
Remarks 4 and 7, Example 8, Table 1) — full text, internal numbering verified item by item;
Faghihi, arXiv:2606.01184 (injectivity) — abstract only, see the attribution note in §4; Kim &
Lee, arXiv:2603.02289 (silhouette definition, Lemma 2.1, identification conditions C1–C3). All
citations carry DOI/URL per `Literature_Review/VERIFIED.bib`.

**Revisions made in the 2026-08-14 audit** (each is argued in place): the $H_0^{\mathrm{dist}}
\Rightarrow H_0^{\mathrm{out}}$ analysis is now stated per Phase-4.1 candidate and the appeal to
Souto & Diamantis Theorem 5.9 carries its shared-codomain hypothesis (§1.3); the empty-diagram
witness W2 is demoted in favour of W2′, which needs no degenerate convention (§1.3, §1.4); the
claimed weak-versus-joint exchangeability asymmetry between the outcome and distribution levels
is withdrawn — both levels are identified under the same A1 (§2, §3, §4, §7); $\tau_d$ and
$\tau_{\mathrm{dist}}$ are separated (§2); the Remark 7 quotation is restored to full, with its
"unless weak ignorability holds" qualifier, and Saki & Faghihi's Table 1 is what makes the gate
verdict coherent (§3); Theorem 5 is re-attributed to Saki & Faghihi (§4); the claim that
Robinson–Turner cannot reject on the exact witness is **reversed** — it rejects at the minimum
attainable $p$-value, measured (§6.3); and a persistence-measure binning bug that silently
collapses the distribution-level statistic is documented and fixed (§6.5). Numerics in §1.4 were
re-run after the fix.

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
(P^a_Y)$ and $\delta_{\mathrm{dist}} = \varrho(\Theta^1_{\mathrm{dist}},
\Theta^0_{\mathrm{dist}})$ for a fixed metric $\varrho$ on the distribution-level summary space.
Write the mean functional $\mathcal A(P) = \int \Phi(\mathrm{Dgm}_d(F(y))) \, dP(y)$, so that
$\mathcal A(P^a_Y) = \mathbb E[Z^a]$; $\mathcal A$ is affine in $P$ (Bochner expectation).

**Notation discipline (three collisions the manuscript must avoid).** (i) $A$ is the treatment
indicator; the mean functional is $\mathcal A$, following Souto & Diamantis §5.4. (ii) $d$ is the
homology degree only; the distribution-level metric is $\varrho$, and death coordinates in the
witnesses of §1.3 are written $\ell$. (iii) $\delta$ is the equivalence margin and the Dirac mass;
the individual topological effect is written $\Delta_{i,d} = Z^1_i - Z^0_i$, never $\delta_{i,d}$.

**Implementation note (the $\sqrt2$).** The displayed $\phi$ is the Kim & Lee normalisation.
`tda2s.vec.silhouette` wraps `gudhi.representations.Silhouette`, which returns $\sqrt2\,\phi$
(gudhi multiplies the weighted tent average by $\sqrt2$; verified against the source, gudhi
3.11.0). Every null in §1.2 is scale-invariant except $H_0^{\mathrm{equiv}}$, whose margin
$\delta$ is not, so the margin must be declared against a stated normalisation; and any numeric
silhouette value quoted in this document or in the manuscript is on the gudhi scale, which is
$\sqrt2$ times the displayed formula.

### 1.2 The six nulls

| Label | Statement (measurable form) | Object being compared |
|---|---|---|
| $H_0^{\mathrm{cond}}$ | $\mathcal L(D \mid A=1) = \mathcal L(D \mid A=0)$ | Observational conditional laws of diagrams; the field's null, confounded by $X$ |
| $H_0^{\mathrm{out}}$ | $\psi_d(t) := \mathcal A(P^1_Y)(t) - \mathcal A(P^0_Y)(t) = 0$ for all $t \in T$ | Outcome-level TATE: means of potential silhouettes (Kim & Lee) |
| $H_0^{\mathrm{dist}}$ | $\delta_{\mathrm{dist}} = \varrho(T_{\mathrm{dist}}(P^1_Y), T_{\mathrm{dist}}(P^0_Y)) = 0$ | Distribution-level: topological representations of the interventional laws |
| $H_0^{\mathrm{ctate}}$ | $\tau_d(t,x) := \mathbb E[\Delta_{i,d}(t) \mid X=x] = 0$ for $P_X$-a.e. $x$, all $t$ | Conditional effect; $\Delta_{i,d} = Z^1_i - Z^0_i$ is the individual topological treatment effect |
| $H_0^{\mathrm{het}}$ | $x \mapsto \tau_d(t,x)$ is $P_X$-a.e. constant, for every $t$ | Heterogeneity omnibus |
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

**Theorem 1 (the implication is pairing-conditional).** Let $\mathcal A$ be the mean functional,
let $\varrho$ separate points (a genuine metric, not a pseudometric, on the range of
$T_{\mathrm{dist}}$), and suppose $\mathcal A = L \circ T_{\mathrm{dist}}$ for some map $L$
defined on that range (the outcome-level mean factors through the distribution-level
representation). Then $H_0^{\mathrm{dist}} \Rightarrow H_0^{\mathrm{out}}$. Three regimes matter,
and they are exactly the three $T_{\mathrm{dist}}$ candidates of plan task 4.1:

1. *Canonical measure-valued pairing (plan candidate (i)).* Take $\Phi(D) = \mu_D = \sum_{p \in D}
w_p \, \delta_{\mathrm{coord}(p)}$, the **un-normalized** weighted persistence measure, and
$T_{\mathrm{dist}}(P) = \mathbb E_P[\mu_{D(F(Y))}]$, the expected persistence measure. Then
$\mathcal A = T_{\mathrm{dist}}$ identically ($L = \mathrm{id}$), so $H_0^{\mathrm{dist}}$ and
$H_0^{\mathrm{out}}$ coincide. This is the "linear-friendly" regime advertised in Phase 4.1:
expectation commutes with the un-normalized measure summary, and there is no normalization ratio
to break linearity. *Technical caveat, load-bearing for Phase 4.2:* §1.1 asks $\Phi$ to take
values in a **separable** Banach space, and finite Radon measures under the total-variation norm
are not separable, so $\mathbb E_P[\mu_D]$ must not be introduced as a Bochner integral in
$(\mathcal M, \|\cdot\|_{TV})$. State it instead in the dual pairing against a separable predual
($\mu \mapsto \int f\,d\mu$ for $f$ in a fixed separable test class — the tent family of the
codomain-match repair below is exactly such a class), or embed into a separable RKHS. Divol &
Chazal's LLN for expected persistence measures is stated in the vague/weak topology for this
reason, not in TV.
2. *Kernel mean embedding (plan candidate (ii)) — and such a kernel exists.* If $T_{\mathrm{dist}}$
is the mean embedding of $\mathcal L(D)$ under a **characteristic** kernel on diagram space,
$T_{\mathrm{dist}}$ is injective on laws, so $T_{\mathrm{dist}}(P^1) = T_{\mathrm{dist}}(P^0)$
forces $\mathcal L(D^1) = \mathcal L(D^0)$ and hence equality of the mean of *any* integrable
$\Phi$. The plan's original unhedged sentence is therefore **true** for this candidate, and
$H_0^{\mathrm{dist}}$ is then the strictly stronger null. This regime is **not hypothetical**:
Kwitt, Huber, Niethammer, Lin & Bauer (NIPS 2015), Proposition 2, prove that
$k^U_\sigma(F,G) = \exp(k_\sigma(F,G))$ — the exponentiated Reininghaus persistence scale-space
kernel — is *universal with respect to $d_{W,1}$* on the set $S$ of diagrams whose birth/death
coordinates are bounded by a fixed $R$ and whose total multiplicity is bounded by a fixed $N$.
Universal implies characteristic, so the mean embedding is injective on laws over $S$. Their
proof routes through Christmann & Steinwart's Taylor-kernel theorem, using injectivity and
continuity of the PSS feature map $\Phi_\sigma : \mathcal D \to L^2(\Omega)$ and compactness of
$(S, d_{W,1})$; the two boundedness restrictions are what buy that compactness, and they are
mild for data from a finite process. Consequently the manuscript cannot treat the
$H_0^{\mathrm{dist}}$ / $H_0^{\mathrm{out}}$ relation as a fact about topology — it is a
consequence of a representation choice that Phase 4.1 makes and must state.
3. *Expected persistence intensity (plan candidate (iii)).* The intensity is a smoothing of the
expected measure, so it inherits case 1 against the un-normalized $\Phi$ and fails, exactly as
in Theorem 2, against the normalized silhouette.

**Theorem 2 (default silhouette pairing: the two nulls are logically independent).** Take
$\Phi$ = **normalized** power-weighted silhouette and $T_{\mathrm{dist}}(P)$ = expected
persistence measure. Then neither implication holds. Write $\ell$ for a death coordinate and
$\delta_{(0,\ell)}$ for the Dirac mass at the diagram point $(0, \ell)$; a diagram is identified
with the counting measure of its points.

*W1, the counterexample to the converse (the plan's cluster-splitting witness).* Fix
$\ell > 0$. Let $D^0 = \delta_{(0,\ell)}$ (one persistent $H_0$ class) and $D^1 =
2\,\delta_{(0,\ell)}$ (two classes, same merge scale), both deterministic. The mean silhouettes
are equal exactly, because the silhouette is a normalized average and the two tents coincide,
while the expected persistence measures differ ($w_\ell\delta_{(0,\ell)}$ vs
$2w_\ell\delta_{(0,\ell)}$). Hence $H_0^{\mathrm{out}}$ holds and $H_0^{\mathrm{dist}}$ fails.

*W2′, the counterexample to the implication (multiplicity mixture over two locations).* Let
$D^0 = \delta_{(0,1)} + \delta_{(0,2)}$ deterministically, and let $D^1$ equal
$2\,\delta_{(0,1)}$ or $2\,\delta_{(0,2)}$ with probability $1/2$ each. Each location carries
multiplicity $1$ in expectation in both arms, so the expected persistence measures agree
**exactly, for any point weight**, while the mean silhouettes weight the two tents
$(1/3, 2/3)$ against $(1/2, 1/2)$ (for $r = 1$; the analogous mismatch holds for every $r$) and
therefore differ. Hence $H_0^{\mathrm{dist}}$ holds and $H_0^{\mathrm{out}}$ fails.

*W2, the shorter but weaker variant.* The two-point-versus-empty witness — $D^0 =
\delta_{(0,1)}$; $D^1 = 2\,\delta_{(0,1)}$ or the empty diagram with probability $1/2$ each —
also works, and the mean silhouettes then differ by half a tent. It is **not** the version to
print: it relies entirely on the convention $\phi(\cdot;\varnothing) \equiv 0$, which gudhi
supplies through an empty sum rather than by definition, and a referee can dismiss it as a
degenerate-convention artefact. W2′ has no empty diagram anywhere. Both are instantiated in
§1.4; W2′ is the manuscript's witness and W2 is at most a footnote.

The mechanism in both is the non-affineness of the *normalization ratio*: topology is applied
per cloud inside the expectation, and dividing by $\sum_p w_p$ prevents the expectation from
commuting with the mixture. This is the same algebra as the covariate-mixture non-commutation of
Saki & Faghihi (Remark 4), but it is not literally that statement: W2 and W2′ mix over
realizations *within* an arm, with no covariate present. The manuscript should say "the same
non-commutation mechanism", not "the covariate-mixing non-commutation".

**Proof of Theorem 1.** Case 1 is the identity $\mathcal A(P^a) = \mathbb E[\mu_{D^a}] =
T_{\mathrm{dist}}(P^a)$. Case 2 follows from injectivity of the embedding. In general, if
$\mathcal A = L \circ T_{\mathrm{dist}}$ and $\varrho$ separates points, then
$\delta_{\mathrm{dist}} = 0$ gives $T_{\mathrm{dist}}(P^1) = T_{\mathrm{dist}}(P^0)$, hence
$\mathcal A(P^1) = \mathcal A(P^0)$. The converse fails whenever $L$ or $T_{\mathrm{dist}}$ is
non-injective. The W1 and W2′ witnesses are verified by direct computation of the tent average
and the measure expectation; both are instantiated numerically in §1.4.

**Connection to Souto & Diamantis Theorem 5.9 — and the codomain caveat.** That theorem states:
for $\mathcal A, T_{\mathrm{dist}}: \mathcal Q \to \mathcal B$ **into a common Banach space
$\mathcal B$**, the two contrasts $\Delta_{\mathrm{out}} = \mathcal A(P^1) - \mathcal A(P^0)$ and
$\Delta_{\mathrm{dist}} = T_{\mathrm{dist}}(P^1) - T_{\mathrm{dist}}(P^0)$ agree for every pair
of laws iff $T_{\mathrm{dist}} - \mathcal A$ is constant on $\mathcal Q$. The shared codomain is
a hypothesis, not a formality: in the default pairing the mean silhouette lives in $C(T)$ and the
expected persistence measure in a space of Radon measures, so the difference
$T_{\mathrm{dist}} - \mathcal A$ is **not defined** and Theorem 5.9 does not apply as stated.
Two repairs, either of which is honest:

- *Codomain-match first.* Replace the expected measure by the linear functional it induces on
  the same grid of tents, $T_{\mathrm{dist}}(P) = \int \Lambda_p(\cdot)\, d(\mathbb E_P[\mu_D])(p)
  = \mathbb E_P\big[\sum_p w_p \Lambda_p(\cdot)\big]$, i.e. the expected **un-normalized**
  weighted silhouette, which lies in $C(T)$. Now both maps are affine into $C(T)$, Theorem 5.9
  applies verbatim, and W1 exhibits a pair on which the contrasts differ, so
  $T_{\mathrm{dist}} - \mathcal A$ is non-constant. (Theorem 5.9's final clause covers this: two
  *distinct affine* maps can also give different contrasts; non-affineness is sufficient, not
  necessary.)
- *Do not invoke it.* W1 and W2′ are proved by direct computation and need no appeal to Theorem
  5.9 at all.

What Theorem 5.9 does **not** deliver, in either repair, is the logical independence of the two
nulls: non-agreement of *contrasts* on some pair of laws is weaker than the existence of
witnesses in both directions. That is what W1 and W2′ are for, and the manuscript should cite
Theorem 5.9 as context and the witnesses as proof. The plan's §1 should not print the unhedged
sentence "$H_0^{\mathrm{dist}} \Rightarrow H_0^{\mathrm{out}}$"; the correct, still-unhedged
sentence names the pairing, per Theorem 1.

### 1.4 Numerical instantiation (exit check for task 1.1)

`scripts/wp1_counterexample.py` (runs the PH pipeline, silhouettes, persistence measures and
permutation tests from `tda2s`); the DGP `split_cluster_cloud` lives in `tda2s/dgp/clouds.py`
and is the Phase 4.4 separation DGP. Results at $n=300$ clouds per group, 300 permutations,
10 Monte Carlo replications. Silhouette values are on the gudhi scale ($\sqrt2 \phi$, §1.1).

| Check | Statistic | Result | Reading |
|---|---|---|---|
| W1, law level | mean-silhouette gap | 0.0096 (1.9 null SDs) | $H_0^{\mathrm{out}}$ holds; gap is group-mean Monte Carlo noise |
| W1, law level | expected-measure distance | 1.355 | $H_0^{\mathrm{dist}}$ fails, at a scale far above noise |
| W1, law level | outcome-level test rejection | 1/10 at 5% | consistent with size (0.5 rejections expected of 10) |
| W1, law level | distribution-level test rejection | 10/10 at 5% | power held |
| W2′, law level | mean-silhouette gap | 0.560 (19 null SDs) | $H_0^{\mathrm{out}}$ fails |
| W2′, law level | expected-measure distance | 0.12 | $H_0^{\mathrm{dist}}$ holds; population value is exactly 0, this is MC noise |
| W2′, law level | outcome-level rejection | 10/10 | power held |
| W2′, law level | distribution-level rejection | 1/10 | consistent with size |
| W2 (weaker variant) | gap / measure distance | 0.356 (24 SDs) / 0.05 | same conclusion, but rests on $\phi(\cdot;\varnothing) \equiv 0$ |
| Cloud exact (deterministic blobs + threshold $\tau = 0.3$) | H0 counts | 1 vs 2 | the split is realized by the PH pipeline |
| Cloud exact | H0 deaths | 1.350 both groups | equal merge scale exactly |
| Cloud exact | mean-silhouette gap | 0.000e+00 | mean preserved, realization by realization |
| Cloud exact | expected-measure distance | 2.460375 | distribution-level separation, weight power $r=3$ |
| Cloud exact | Robinson–Turner "within" $p$, $N=16$ | 0.005 | **RT rejects**, at the minimum attainable $p$; see §6.3 |
| Cloud Gaussian (diagnostic) | gap / null SD | 3.2 | noisy geometry breaks exactness, see remark |

**Three caveats on this table, so it is not over-read.** First, 10 Monte Carlo replications
establish *existence* of the two witnesses, nothing about calibration: a test with true size 0.20
yields $\ge 1/10$ rejections with probability 0.89, so "1/10" is not evidence of size control and
must not be quoted as such. Phase 4.4's 300-replication fleet supplies an empirical exit check, not a
formal calibration theorem. Second, the Phase 4 deterministic cloud witness has equal total cardinality in both arms.
The 18-gon versus 12-gon choice equalises total cardinality at 36 points; the different within-blob
resolutions are part of the construction and their short classes are removed by the threshold.
Third, the persistence-measure grid is now pinned (see §6.5): binning each diagram
against its own range silently collapses the statistic to total weighted mass.

The deterministic cloud construction: the two-blob arm uses regular 18-gons and the three-blob
arm regular 12-gons, all of radius 0.15; two blobs
(one $H_0$ class, death $(3.0 - 0.3)/2 = 1.35$) versus three blobs in an equilateral
arrangement (two classes, both dying at 1.35). Persistence thresholding at $\tau = 0.3$ removes
the tiny within-blob classes (deaths below 0.039) that exist in both groups; without it the mean
silhouette equality is only approximate at the $10^{-4}$ level. With noisy Gaussian blobs the
two classes become the two smallest order statistics of three pair-merge scales, biasing the
mean silhouette (gap 3.2 null SDs in the diagnostic), so Phase 4.4 must use the deterministic
DGP or tightly-clustered thresholded diagrams for its low-outcome-rejection claim. The 18-gon
two-blob arm has vertices on the $0°/180°$ axis, while the 12-gon three-blob arm has vertices on
the relevant $150°/330°$ axes, so the merge scale is identical to machine precision.

---

## 2. Task 1.2 — g-formula identification of $\psi_d$ and $\delta_{\mathrm{dist}}$

All results of this section are reused from Souto & Diamantis §§4–5 (Theorems 4.1, 5.3,
Corollary 5.4, Proposition 5.6), which in turn restate and extend Kim & Lee (C1–C3). Define the
outcome regression $m_a(x) = \mathbb E[Z \mid A = a, X = x]$ (Bochner-valued) and $e_a(x) =
P(A = a \mid X = x)$.

**A1, the assumption set, is the same at both levels.** Souto & Diamantis state Theorems 4.1 and
5.3 under a single Assumption 2.2, *joint* conditional exchangeability $(Y^0, Y^1) \perp A \mid
X$. Both proofs use only the arm-specific consequence $Y^a \perp A \mid X$, and the paper says so
itself at the head of its §8: "For identification of treatment-specific marginal laws, the weaker
arm-specific condition $Y^a \perp A \mid X$ is sufficient; we call this *weak conditional
exchangeability*." So **A1 = consistency + weak conditional exchangeability + strict positivity
identifies the outcome level and the distribution level alike.** There is no assumption
asymmetry between $\psi_d$ and $\delta_{\mathrm{dist}}$, and the manuscript must not manufacture
one: claiming C2 needs joint exchangeability while C1/C3 need only the arm-specific form would
hand a referee a free objection against the paper's newest contribution. (Joint exchangeability
becomes load-bearing only for joint-distribution targets — individual effects, quantiles of
$Z^1 - Z^0$ — which this paper does not estimate.)

**Outcome level (identification of $H_0^{\mathrm{out}}$).** Assume A1. If $\mathbb E \lVert Z^a
\rVert_B < \infty$ then (Theorem 4.1)

$$\mathbb E[Z^a] = \mathbb E[m_a(X)] = \mathbb E\Big[\frac{\mathbb 1\{A=a\}}{e_a(X)} Z\Big],$$

so $\psi_d = \mathbb E[m_1(X) - m_0(X)]$ is identified from the observational law. Strict
positivity suffices for the population identities; uniform inverse-weight bounds are needed
only for the asymptotic arguments of Phase 3. The augmented representation $\Psi_a(\hat m_a,
\hat e_a)$ (Proposition 4.3) satisfies the double-robustness identity $\Psi_a(\hat m_a, \hat e_a) -
\mathbb E[Z^a] = \mathbb E[(\hat e_a - e_a)(\hat m_a - m_a)/\hat e_a]$, so the AIPW pseudo-outcome
is exact if either nuisance is correct. Its remainder is the product rate $\varepsilon^{-1}
\lVert \hat e_a - e_a \rVert_{L^2(P_X)} \cdot (\mathbb E \lVert \hat m_a - m_a \rVert_B^2)^{1/2}$,
where $\varepsilon$ is the lower bound imposed on $\hat e_a$; the $\varepsilon^{-1}$ factor is
part of the statement, not decoration, and is what Phase 3.6's positivity diagnostics are
guarding. This is the load-bearing object for Phase 3.4's stress test.

**Distribution level (identification of $H_0^{\mathrm{dist}}$).** Let $Q_a(x, \cdot) =
\mathcal L(Y \mid A = a, X = x)$ be the observed conditional outcome kernel, defined $P_X$-a.e.
under strict positivity. Under A1 (Theorem 5.3, whose proof again needs only the arm-specific
form), the interventional laws satisfy the law-valued g-formula

$$P^a_Y = \int_{\mathcal X} Q_a(x, \cdot) \, dP_X(x), \qquad a \in \{0,1\}.$$

If $T_{\mathrm{dist}}$ is known and well-defined at $P^0_Y$ and $P^1_Y$, then $\Theta^a_{\mathrm{dist}}$,
$\Delta_{\mathrm{dist}}$ and $\delta_{\mathrm{dist}}$ are identified (Corollary 5.4). Two
remarks are load-bearing for the manuscript. First, topology is applied *after* the mixture:
$\mathrm{Dgm}(F(\cdot))$ does not commute with mixing over $X$ (Remark 5.7; Saki & Faghihi
Remark 4), so the plug-in estimator must form the mixture first and then transform, never the
reverse; W2′ of §1.3 exhibits the same algebra with the mixture taken over realizations rather
than over covariates. Second, continuity of
$T_{\mathrm{dist}}$ is not needed for identification but is needed for stable estimation
(Corollary 5.4 note); Phase 4's plugin consistency argument therefore requires a continuity
or stability-transfer condition, exactly the object Souto & Diamantis license.

**Conditional level (identification of $H_0^{\mathrm{ctate}}$ and $H_0^{\mathrm{het}}$).**
Under A1, the conditional effect is identified as $\tau_d(t,x) = m_1(t,x) - m_0(t,x)$, and
equivalently as the conditional mean of the DR pseudo-outcome
$\rho_d(t, Z; \eta) = m_1(t,X) - m_0(t,X) + (A/e(X))(Z(t) - m_1(t,X)) - ((1-A)/(1-e(X)))(Z(t)
- m_0(t,X))$: $\mathbb E[\rho_d(t, Z; \eta) \mid X = x] = \tau_d(t,x)$. This is the DR-learner
property (Kennedy 2023) that Phase 5's pseudo-outcome construction consumes.

**Do not conflate $\tau_d$ with $\tau_{\mathrm{dist}}$.** They are different objects and the
manuscript must keep them apart. $\tau_d(t,x) = m_1(t,x) - m_0(t,x)$ is *outcome-level*: signed,
Banach-valued, a function of $(t,x)$, and the target of C3. $\tau_{\mathrm{dist}} = \mathbb E_X
[\varrho(T_{\mathrm{dist}}(P_{1,X}), T_{\mathrm{dist}}(P_{0,X}))]$ (Souto & Diamantis Definition
5.5, identified by Proposition 5.6 via $P_{a,x} = Q_a(x,\cdot)$ $P_X$-a.s.; Saki & Faghihi's
$\tau_\Psi$) is *distribution-level*: a non-negative scalar, already averaged over $x$, and
therefore useless as a heterogeneity target — a null of $\tau_{\mathrm{dist}} = 0$ says the
within-stratum representations coincide a.e., not that the effect is homogeneous. C3's tests are
about $\tau_d$; $\tau_{\mathrm{dist}}$ enters this paper only as the fallback estimand of §3.

---

## 3. Task 1.3 — the marginal-identification question (the gate)

**Answer: no. The marginal persistence-diagram effect is not identified from conditional
topological ignorability alone.** Verified against the full text of arXiv:2603.14169v2 (local
copy at `../CP_TATE/theory/source_pdfs/`), whose internal numbering was checked item by item.
Their Remark 7 reads, **in full** (the qualifier matters and must not be elided):

> Proposition 6 identifies the transformed conditional laws, not the marginal interventional law
> itself. **Unless weak ignorability holds or $\Psi$ commutes with mixing**, conditional
> topological ignorability does not identify $\Theta_\Psi = \lVert \Psi(P_1) - \Psi(P_0)
> \rVert_E$. This is the fundamental reason to target $\tau_\Psi$ rather than $\Theta_\Psi$ in
> the noninjective regime.

Their Example 8 supplies the witness (two mixtures with identical conditional summaries but
different marginal topology, per alignment of the modes across strata); their Proposition 6 is
the positive half; Souto & Diamantis Proposition 8.2(3) and §8 restate it. The mechanism is the
non-commutation of persistent homology with covariate mixtures: conditional topological
ignorability fixes $T_{\mathrm{dist}}(P_{a,x}) = T_{\mathrm{dist}}(P^{\mathrm{fact}}_{a,x})$
stratum by stratum, and the marginal law is a $P_X$-mixture of the stratum laws, whose topology
depends on the geometric alignment of the strata, which the identifying condition does not
constrain.

**The bracketed qualifier is what makes the gate verdict PASS rather than FAIL, so it must be
stated, not buried.** Saki & Faghihi's own Table 1 is explicit: under *weak ignorability +
positivity with an arbitrary measurable $\Psi$*, the identified targets are "the interventional
laws $P_t$, the marginal contrast $\Theta_\Psi$, the conditional contrast $\theta_\Psi(z)$, and
the standardized effect $\tau_\Psi$" — all four. The non-identification bites only in the strictly
weaker regime where a reader grants conditional topological ignorability *instead of* weak
exchangeability. This paper assumes A1, which includes weak conditional exchangeability, so
$\delta_{\mathrm{dist}}$ is identified under the paper's own assumptions. What the result costs
the paper is therefore a *robustness* claim, not an estimand: the paper cannot say "our
distribution-level target survives on topological ignorability alone."

**Consequence for the paper's headline estimand.** The paper must target the
*covariate-standardized* topological effect and say so plainly:

1. $H_0^{\mathrm{out}}$ already is a standardization: $\psi_d = \mathbb E[m_1(X) - m_0(X)]$
   (Theorem 4.1), so no change to the outcome-level null; it is identified under the standard
   A1 conditions (consistency, weak conditional exchangeability, positivity), which the paper
   and Kim & Lee assume.
2. $H_0^{\mathrm{dist}}$ is identified under the **same** A1 via the g-formula (Theorem 5.3,
   Corollary 5.4) — see §2's assumption note; no stronger exchangeability is required. Where a
   reader grants only conditional topological ignorability, the identified distribution-level
   object drops to the standardized within-stratum contrast $\tau_{\mathrm{dist}} = \mathbb E_X
   [\varrho(T_{\mathrm{dist}}(Q_1(X,\cdot)), T_{\mathrm{dist}}(Q_0(X,\cdot)))]$ (Proposition
   5.6; Saki & Faghihi Proposition 6), and the paper must not present $\delta_{\mathrm{dist}}$
   as identified in that regime. One honest sentence in the assumptions subsection discharges
   this, and the unmeasured-confounding sensitivity of Phase 9.3 is its natural home.
3. $H_0^{\mathrm{ctate}}$ and $H_0^{\mathrm{het}}$ target $\tau_d(t,x) = m_1(t,x) - m_0(t,x)$,
   which is identified under A1. They do **not** inherit Saki & Faghihi Proposition 6: that
   proposition identifies $\theta_\Psi(z)$, a norm of a difference of *distribution-level*
   transforms, which is a different object from the signed outcome-level CTATE (§2). The one
   way to route C3 through topological ignorability is to instantiate $T_{\mathrm{dist}} =
   \mathcal A$, the mean functional read as a map on stratum laws; ignorability relative to that
   $T_{\mathrm{dist}}$ says exactly $\mathbb E[\Phi(D^a) \mid X=x] = m_a(x)$, which is what
   $\tau_d$ needs. If the manuscript wants that route it must name the representation; otherwise
   it should simply cite A1.

**Gate verdict: PASS.** Phases 3–5 estimate $\psi_d$, $\delta_{\mathrm{dist}}$ and $\tau_d$
under A1 exactly as planned; no estimand change, only a naming obligation — every target is a
covariate-standardized effect, and the paper says so in the assumptions subsection rather than
claiming a marginal null. The plan's risk item 3 ("if the marginal topological null is genuinely
unidentified under conditional ignorability, the paper's headline becomes the *standardized*
effect") is discharged, and it was already the right instinct: $\psi_d$ was a standardization
from the start.

---

## 4. Task 1.4 — injectivity and the identification regimes of the six nulls

**Saki & Faghihi, arXiv:2603.14169v2, Theorem 5** ("Injective transforms collapse topological
ignorability to weak ignorability"): if the chosen summary is injective on the model class,
conditional topological ignorability is *equivalent* to weak conditional exchangeability; for
non-injective summaries it is strictly weaker, and then only the structural feature of interest
is identified, not the full interventional law. Restated as Souto & Diamantis Proposition 8.2(2).
*Attribution note:* the numbered Theorem 5 is Saki & Faghihi's, not the single-author Faghihi
paper's — arXiv:2606.01184 states the same conclusion in its abstract (verified verbatim in
Phase 0.7) but its internal numbering has not been checked, so cite it for the claim and Saki &
Faghihi Theorem 5 for the theorem. Souto & Diamantis add the sharpening that non-injectivity is
*necessary but not sufficient* for topological ignorability to be strictly weaker: the model
class must actually contain distinct relevant laws in the same fiber. Regime per null:

| Null | Identification regime | What is identified under the regime's assumptions |
|---|---|---|
| $H_0^{\mathrm{cond}}$ | observational | The conditional laws themselves; no causal burden. This is the field's null and the object Phase 2 shows to be the wrong target under imbalance |
| $H_0^{\mathrm{out}}$ | mean summary in $B$; A1 (consistency + weak conditional exchangeability + strict positivity) | $\psi_d$, the mean of the chosen summary; injectivity of $\Phi$ is *not* required (a mean of a non-injective summary is still identified; only the mean is claimed) |
| $H_0^{\mathrm{dist}}$ | law level; **the same A1**, plus $T_{\mathrm{dist}}$ well-defined on $P^a_Y$ | $\Theta^a_{\mathrm{dist}}$ and hence $\delta_{\mathrm{dist}}$; under conditional topological ignorability *instead of* A1, only $\tau_{\mathrm{dist}}$ (see §3). Semantic caveat: for non-injective $T_{\mathrm{dist}}$, $\delta_{\mathrm{dist}} = 0$ means equality of the selected representations, not equality of the interventional laws |
| $H_0^{\mathrm{ctate}}$ | conditional; A1 with positivity holding at $x$ | $\tau_d(t,x) = m_1(t,x) - m_0(t,x)$. Topological ignorability relative to a distribution-level $T_{\mathrm{dist}}$ does *not* deliver this (§2, §3.3); it delivers $\theta_\Psi(x)$, a different, norm-valued object |
| $H_0^{\mathrm{het}}$ | as $H_0^{\mathrm{ctate}}$ | dependence of $\tau_d$ on $x$; the null is a statement about a function, so its testability additionally requires consistent estimation of $\tau_d(t,\cdot)$ over $x$ (Phase 5) |
| $H_0^{\mathrm{equiv}}$ | as $H_0^{\mathrm{out}}$ | $\sup_t \lvert \psi_d(t) \rvert$ against a prespecified margin, on a declared silhouette normalisation (§1.1) |

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
   case 1) — subject to the separability caveat recorded there, which is why the persistence
   measure enters through a separable test class rather than through the total-variation norm.

Consequently the paper does inference in $B$ (silhouettes, landscapes) or in persistence-measure
space (expected measure), never in $(\mathcal D_p, W_p)$ directly; Fréchet means in diagram
space are at most a robustness check, and only where their non-uniqueness does not affect the
conclusion.

---

## 6. Implementation remarks for Phases 2–5

1. **gudhi bottleneck degeneracy.** `gudhi.bottleneck_distance` is 20–60× slower on *identical*
   diagrams, where the optimal distance is exactly zero (CGAL degenerate path), than on generic
   ones. Measured on this machine, per call: 3.2 / 8.8 / 54 / 148 ms at diagram sizes
   $k = 1 / 25 / 100 / 200$, against 0.08 / 0.22 / 1.0 / 2.4 ms generic. Diagrams that merely
   share *ties* but differ are fast (0.27 / 1.0 / 2.8 / 10.7 ms), so the trigger is exact
   coincidence, not degeneracy in general. At $N = 600$ diagrams Robinson–Turner's O($N^2$) loop
   is then ~9 minutes of pure bottleneck cost. Phase 2.3's imbalance sweep and Phase 7's grid can
   hit identical diagrams (small effects, low $n$, thresholded $H_0$ diagrams with one or two
   points); benchmark wrappers must short-circuit exactly-equal diagrams to zero before the call.
   Recorded in the WP1 script docstring.
2. **Exact mean-preserving DGPs need deterministic geometry or thresholding.** Noisy blob
   geometry turns the two $H_0$ classes of the split group into order statistics of three
   pair-merge scales, biasing the mean silhouette (3.2 null SDs at the diagnostic's settings).
   Phase 4.4's low-outcome-rejection diagnostic must use
   `split_cluster_cloud(..., deterministic=True)` with the persistence threshold, exactly as in
   `wp1_counterexample.py`, and must additionally equalise cloud cardinality across arms (§1.4).
3. **Robinson–Turner detects W1 decisively; the earlier "p = 1" reading was inverted.** With
   equal merge scales, within-group bottleneck distances are zero and cross-group distances are
   $\ell/2 = 0.675$, so the observed within-loss is the *minimum* attainable — and RT rejects for
   **small** within-loss (`tda2s/benchmarks/rt.py:89` uses `direction="less"`, matching the
   paper's Algorithm 2). Minimum observed loss therefore gives the *smallest* attainable
   $p$-value, not the largest. Measured on the exact witness at $N = 16$, 199 permutations:
   $p = 0.005 = 1/(199+1)$, the floor. The run costs 0.2 s, so remark 1 was never a reason to
   skip it. The correct narrative for Phase 4.4 is therefore **not** "RT is blind here": RT is a
   test of $H_0^{\mathrm{cond}}$, W1 violates $H_0^{\mathrm{cond}}$, and RT rightly fires. What
   W1 shows is that the *outcome-level* mean-silhouette test is blind to it (gap exactly 0), so
   the argument for C2 is against the outcome-level contrast, not against RT. This also corrects
   §1.4's earlier gloss that RT's Gaussian-diagnostic $p = 0.003$ reflects the order-statistic
   bias rather than the multiplicity: RT sees the multiplicity, as the exact witness proves.
4. **W2′ is the Phase 2.2 masking template.** The multiplicity-mixture witness is a diagram-law
   version of the covariate-mixture masking that Phase 2.2 builds at the cloud level; the two
   phases share the mechanism (normalization ratios inside the expectation) though not the
   mixing variable (realizations here, covariates there).
5. **Persistence measures must be binned on one fixed, shared grid.**
   `tda2s.vec.persistence_measure` derives its bin edges from the diagrams handed to it, so
   calling it one diagram at a time without an explicit `interval` rescales the grid per cloud.
   Because every degree-0 diagram has birth 0, a single-class diagram $\{(0,c)\}$ then lands in
   the *same* bin whatever $c$ is, and the "expected persistence measure" collapses to total
   weighted mass — a scale-invariant scalar. Measured cost: against two $H_0$ laws with equal
   expected mass but different expected measures (arm 0 death $\equiv 2$; arm 1 death $1$ w.p.
   $2/3$ and $4$ w.p. $1/3$), the collapsed statistic gives $L^1 = 0.02$ where the true distance
   is $4.02$ — zero power. W1 and W2′ survived it only because their separation *is* mass. Fixed
   in `scripts/wp1_counterexample.py` (`_meas` now pins `interval=(0, 2)`); Phase 4.2/4.3 must
   pin the grid once, globally, before any group contrast, and Phase 2.3 must do the same.
   Accordingly, the implemented Phase 4 target is the fixed-grid projection of the expected
   persistence measure. Equality of the unprojected measures is not certified by a finite grid
   unless a resolution or injectivity argument is added.

---

## 7. Exit audit — identification conditions and failure modes per null

| Null | Identification condition | Failure mode (when the test is misleading) |
|---|---|---|
| $H_0^{\mathrm{cond}}$ | none (observational laws) | fires on covariate shift even with no topological effect; silent under masking. This is the Phase 2 gate's target |
| $H_0^{\mathrm{out}}$ | consistency + weak conditional exchangeability + strict positivity + $\mathbb E \lVert Z^a \rVert < \infty$ | hidden confounding (the g-formula mixture is the wrong one); non-overlap (unidentified on deterministic-treatment regions); representation drift (only the chosen summary's mean is claimed) |
| $H_0^{\mathrm{dist}}$ | **the same A1** as $H_0^{\mathrm{out}}$; $T_{\mathrm{dist}}$ well-defined on $P^a_Y$; continuity of $T_{\mathrm{dist}}$ for estimation | if only conditional topological ignorability is granted, the marginal object is unidentified and the target drops to $\tau_{\mathrm{dist}}$ (§3); non-continuous $T_{\mathrm{dist}}$ gives identification without stable estimation; non-injective $T_{\mathrm{dist}}$ makes the null about representations, not laws; an unpinned binning grid silently collapses the estimator (§6.5) |
| $H_0^{\mathrm{ctate}}$ | as $H_0^{\mathrm{out}}$, pointwise; fixed-$x$ positivity | weak overlap at $x$; conflation with the norm-valued $\theta_\Psi(x)$ of the topological-ignorability literature, which is a different estimand (§2, §4) |
| $H_0^{\mathrm{het}}$ | as $H_0^{\mathrm{ctate}}$ | nuisance estimator of $\tau_d(t,\cdot)$ inconsistent in $x$ (the omnibus then fires on estimation error); multiplicity across $x$ and $t$ uncontrolled |
| $H_0^{\mathrm{equiv}}$ | as $H_0^{\mathrm{out}}$ | margin $\delta$ chosen post hoc; margin declared against an unstated silhouette normalisation (§1.1's $\sqrt2$); multiplicity across $t$ and degrees |

**Deliverable met:** `theory/WP1_estimands_identification.md` (this file), with numerical
evidence in `scripts/wp1_counterexample.py` and the reusable DGP `split_cluster_cloud` in
`tda2s/dgp/clouds.py` (exported from `tda2s.dgp`, covered by `tests/test_dgp.py`).
Gate 1.3 answered: the headline is the covariate-standardized topological effect; no estimand
changes for Phases 3–5.

**Both obligations this document handed forward are now discharged in the plan (2026-08-14).**
The plan's §1 bullet list was internally inconsistent — its first bullet asserted
$H_0^{\mathrm{dist}} \Rightarrow H_0^{\mathrm{out}}$ unhedged while its second warned that the two
levels can disagree — and has been replaced by the pairing-conditional statement of §1.3, with
task 1.1 rewritten so it no longer instructs a proof of a false claim. Task 4.1 is now marked a
logical decision and carries the witness table below, computed exactly in
`scripts/wp1_pairing_separation.py` (the witness laws are finitely supported, so no Monte Carlo
enters; $0$ means the statistic cannot see the witness):

| Statistic | W1 | W2′ | Induced relation to $H_0^{\mathrm{out}}$ |
|---|---|---|---|
| $\mathcal A$: mean normalized silhouette | $0$ | $0.544$ | *is* $H_0^{\mathrm{out}}$ |
| (i) expected persistence measure, $L^1$ (weight power $r=3$) | $2.460375$ | $0$ | **logically independent** |
| (ii) squared MMD under the universal kernel $k^U_\sigma$ | $1.916989$ | $0.398762$ | **strictly stronger** |

The middle row is the paper's headline pairing and is what keeps C2 a genuinely second test; the
bottom row is available off the shelf (the kernel is already wrapped by task 0.5) and should be
reported as a secondary. What must not happen is for the manuscript to print a relation between
the two nulls without naming the pairing that makes it true.
