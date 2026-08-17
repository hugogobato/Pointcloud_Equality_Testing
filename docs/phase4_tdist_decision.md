# Phase 4.1 — The $T_{\mathrm{dist}}$ decision (C2)

Status: **FROZEN (2026-08-16)**. The decision the plan and WP1 already
recommend is recorded here as a Phase-4 deliverable with the justification the
manuscript must absorb. The companion code is `tda2s/tests/dist_level.py`.

**Decision.** The headline distribution-level representation is

$$T_{\mathrm{dist}}(P) \;=\; \mathbb{E}_P[\mu_{D(F(Y))}],$$

the **expected persistence measure** (Divol & Lacombe 2021; plan candidate
(i)), with the point weight $w_p = (\mathrm{death}_p - \mathrm{birth}_p)^r$
matched to the silhouette exponent so that the codomain-match repair of WP1
§1.3 is available when a referee asks. The contrast is the $L^1$ distance of
expected measures on a fixed shared grid,

$$\delta_{\mathrm{dist}} \;=\; \big\lVert\, T_{\mathrm{dist}}(P^1_Y)
- T_{\mathrm{dist}}(P^0_Y)\,\big\rVert_1.$$

**Finite-grid qualification.** The implementation tests the projected target
$\Pi_G T_{\mathrm{dist}}$, where $\Pi_G$ is the fixed 32-by-32 histogram on
the $(\mathrm{birth},\mathrm{midpoint})$ plane. Thus the shipped p-values concern
equality of the projected expected measures. Equality of the unprojected
measures would require a resolution limit or an injectivity assumption for
$\Pi_G$, neither of which is asserted here.

A secondary variant — the MMD under the **universal** persistence scale-space
kernel $k^U_\sigma = \exp(k_\sigma)$ of Kwitt, Huber, Niethammer, Lin & Bauer
(NIPS 2015, Prop. 2) — is **reported alongside** as `method="mmd"` in
`fit_dist`. It
is a strictly stronger null ($H_0^{\mathrm{dist}} \Rightarrow H_0^{\mathrm{out}}$,
converse false), so it is the natural "did we miss anything that the mean
integrates away?" check; it is already wrapped by `tda2s.benchmarks.mmd`.

## Why this and not candidate (ii) as headline

The choice is *logical*, not computational. Each candidate induces a different
relation between $H_0^{\mathrm{dist}}$ and $H_0^{\mathrm{out}}$, and the
manuscript's headline sentence is a consequence of it (plan risk 9).

| $T_{\mathrm{dist}}$ | Relation to $H_0^{\mathrm{out}}$ (normalized silhouette) | W1 | W2′ | Verdict |
|---|---|---:|---:|---|
| $\mathcal A$ — the mean normalized silhouette | *is* $H_0^{\mathrm{out}}$ | $0$ | $0.544$ | baseline row |
| **(i) expected persistence measure, $L^1$ (weight power $r=3$)** | **logically independent** — witnesses both ways | $2.460375$ | $0$ | **headline.** Existing LLN, kernel estimator, linear-friendly, no Fréchet pathology. Keeps C2 a genuinely *second* test |
| (ii) squared MMD under the universal kernel $k^U_\sigma$ | **strictly stronger** — $H_0^{\mathrm{dist}} \Rightarrow H_0^{\mathrm{out}}$, converse false | $1.916989$ | $0.398762$ | secondary. Collapses the "two independent tests" framing into a nesting |

The table uses the Phase 4 conventions, namely persistence weight power
$r=3$ and squared MMD. The corresponding balanced W2′ root-MMD diagnostic is
$0.631476$, as reported in the witness tests.

The numbers are **exact** (the witness laws are finitely supported) and are
reproduced by `scripts/wp1_pairing_separation.py`; $0$ means the statistic
cannot see the witness at all. See
`theory/WP1_estimands_identification.md` §1.3–§1.4, Theorems 1–2 and §7 for the
proof that the implication is pairing-conditional and that the default
pairing gives logical independence. The two implications of the decision are
load-bearing:

1. **The headline pairing keeps C2 as a second test rather than a strictly
   stronger one.** With $T_{\mathrm{dist}} = $ expected measure, W1 witnesses
   $H_0^{\mathrm{out}}$ true and $H_0^{\mathrm{dist}}$ false (one class splits
   into two; mean preserved by the normalization ratio, expected measure
   doubled); W2′ witnesses the reverse (a multiplicity mixture over two
   locations; expected measures agree for any weight, mean silhouettes differ).
   This is the content of plan tasks 4.4 + 4.R, and it is what the abstract
   sentence "*the outcome-level and distribution-level nulls are logically
   independent, with explicit witnesses in both directions*" depends on.
2. **Under (ii) the "two independent tests" framing collapses into a
   nesting.** $k^U_\sigma$ is universal with respect to $d_{W,1}$ on the set
   $S$ of diagrams with birth/death $\le R$ and total multiplicity $\le N$
   (Kwitt et al. Prop. 2; Christmann–Steinwart Taylor-kernel theorem).
   Universal implies characteristic, so the mean embedding is injective on laws
   over $S$ and $H_0^{\mathrm{dist}}$ becomes literally "the interventional
   diagram laws are equal", the covariate-adjusted version of the MMD test the
   field already runs on observational laws (plan §2 lists MMD as a
   competitor). That is still novel (nobody runs it on interventional laws)
   but it is a *different* claim, and the plan's C1 (the covariate-shift
   failure) is then absorbed into C2 rather than complemented by it.

The paper reports both. If (ii) rejects where (i) does not, the difference
localizes the effect in a part of the law the expected measure integrates
away; if they agree, the headline pairing is doing the work.

## Mathematical caveats the manuscript must print

1. **Separability of the codomain.** Finite Radon measures under the
   total-variation norm are not separable, so $\mathbb E_P[\mu_D]$ must **not**
   be introduced as a Bochner integral in $(\mathcal M, \|\cdot\|_{TV})$. The
   headline implementation takes the codomain-match route of WP1 §1.3: it
   evaluates $T_{\mathrm{dist}}(P)$ through its dual pairing against the
   separable tent family — the bin counts on one **fixed, shared grid** are a
   dual-pairing evaluation on a finite separable test class — so the LLN of
   Divol & Chazal (stated in the vague/weak topology, not in TV) is what
   licenses consistency. The grid is pinned once, globally, before any group
   contrast. Per-diagram binning without an explicit `interval` rescales the
   grid per cloud and silently collapses the statistic to total weighted mass
   (measured: $L^1 = 0.02$ where the true distance is $4.02$; WP1 §6.5).
   `tda2s.vec.persistence_measure` derives its bin edges from the diagrams it
   is handed, so every call site must pass `interval=`; the Phase-4 module
   does so once and passes the shared grid to every subsequent call.
2. **The codomain-match repair is harmless.** The expected *un-normalized*
   weighted silhouette $T_{\mathrm{dist}}(P)(t) = \mathbb E_P\big[\sum_p w_p
   \Lambda_p(t)\big]$ lies in $C(T)$ and equals the expected persistence
   measure read through the tent family. Souto & Diamantis Theorem 5.9 then
   applies verbatim with $\mathcal A$ the *mean normalized silhouette* and
   $T_{\mathrm{dist}}$ this un-normalized analogue; W1 witnesses that
   $T_{\mathrm{dist}} - \mathcal A$ is non-constant, so the two contrasts
   disagree on some pair of laws (Theorem 5.9's "if and only if"). But
   non-agreement of contrasts is weaker than logical independence of the
   nulls: W2′ is what provides the reverse implication failure. The
   manuscript cites Theorem 5.9 as context and the two witnesses as proof,
   exactly as WP1 §1.3 prescribes.
3. **Affine-law obstruction.** The honest reason (i) is linear-friendly is
   that $\mu_D$ is the **un-normalized** measure: it is affine in the diagram
   law $P$, so the $g$-formula mixture over covariates commutes with it,

   $$T_{\mathrm{dist}}(P^a_Y) \;=\; \int_\mathcal X \mathbb E[\mu_{D(F(Y))} \mid X = x, A = a]\,dP_X(x),$$

   and the IPW reweighting is exact (no normalization ratio breaks linearity,
   no Fréchet mean is taken, no diagram-space geometry is entered). This is the
   **same obstruction that makes the *normalized* silhouette non-affine**
   (WP1 §1.3, Theorem 2; Saki & Faghihi Remark 4): the normalization ratio
   $\sum_p w_p$ is inside the expectation and does not commute with mixing,
   which is exactly what W2′ exploits. The consequence for the *estimator*
   (task 4.2) is that the AIPW form is the efficient-influence-function
   estimator when the conditional mean nuisance is supplied; plain IPW is not
   generally efficient. The code exposes the conditional Souto–Diamantis stability-transfer
   formula $L\varepsilon$, but does not pretend to estimate the unknown empirical
   $W_1$ error from one sample. Continuity is needed for stable estimation, not
   for identification.
4. **Permutation calibration is not a weak-null theorem.** For the expected
   measure, $H_0^{\mathrm{dist}}$ is equality of expected feature vectors,
   not equality of the full unit-level diagram laws. Frozen-label permutation
   is conditionally exact under a sharp/exchangeable null, but that assumption
   is violated by W2′. Its observed rejection rate is therefore a diagnostic
   for this prototype, not a finite-sample size guarantee. A production test
   of the weak mean null needs studentized multiplier or bootstrap calibration.

## What does not change if the pairing changes (re-check rule)

WP1 §1.3 and plan risk 9 require that *every* time $(\Phi, T_{\mathrm{dist}})$
changes, the W1/W2′ witness table is re-run. The Phase 4 module exposes
`method` in `fit_dist` so a reviewer can run the same separation experiment under
each candidate without touching the DGP or the null:

* `method="measure"` (default, candidate (i)) — the headline pairing.
  The W1 separation should fire; the W2′ reverse should not (no reverse case
  exists; the reverse-witness subsection reduces to "no reverse case exists, and here is why").
* `method="mmd"` (candidate (ii)) — the secondary pairing. Both W1 and
  W2′ should fire, because $H_0^{\mathrm{dist}}$ is strictly stronger than
  $H_0^{\mathrm{out}}$. This is itself a sanity check: if under `"mmd"` the W2′
  reverse fails, the MMD implementation is wrong.

## Source map used for this decision

- **Souto & Diamantis**, arXiv:2607.28161 — Definition 5.5, Prop. 5.6
  (identification of $\delta_{\mathrm{dist}}$), Theorem 5.9
  (shared-codomain agreement iff $T_{\mathrm{dist}} - \mathcal A$ is
  constant), §5.4 (the affine-law obstruction), §8 (weak exchangeability
  suffices). Statements checked against the manuscript source. **[A]**
- **Saki & Faghihi**, arXiv:2603.14169v2 — Remarks 4 and 7, Example 8,
  Table 1, Theorem 5; with Remarks 4 and 7 supplying the covariate-mixture
  non-commutation that mirrors W2′. Full text, internal numbering verified.
  **[A]**
- **Kwitt, Huber, Niethammer, Lin & Bauer**, NIPS 2015 — Proposition 2:
  $k^U_\sigma$ is universal wrt $d_{W,1}$ on the bounded-multiplicity set $S$,
  hence characteristic, hence the mean embedding is injective on laws over
  $S$. Verified against the paper 2026-08-14 (plan §2). **[A]**
- **Divol & Lacombe**, JACT 2021 — the LLN for expected persistence measures
  in the vague/weak topology (the codomain caveat). **[A]**
- **Divol & Chazal**, JoCG 2021 — persistence measures via optimal partial
  transport; the cleanest home for $H_0^{\mathrm{dist}}$, cited as the
  candidate-(i) home in plan §2 ("Geometry hazards"). **[A]**

All four are in `Literature_Review/VERIFIED.bib` with DOI/URL per the global
instructions.
