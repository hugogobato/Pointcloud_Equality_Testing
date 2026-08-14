# Research Plan P2 — Certified Filtration Substitution

**Working title:** *How Coarse Can You Go? Certified Statistical Equivalence of Cubical and Geometric
Filtrations for Large Point Clouds*

**One-line thesis.** Substituting a cheap cubical filtration for an expensive geometric one is currently
justified by asymptotic hand-waving. We give (i) an explicit bottleneck bound in the cube side length $h$,
(ii) a proof that the target must be Čech and not Rips because the Rips gap is multiplicative and does not
vanish, and (iii) a finite-sample **equivalence test** that certifies the coarsest admissible resolution
with a stated tolerance and stated error control.

---

## 0. The correction that reshapes this project

Your stated premise is that cubical and Vietoris–Rips filtrations converge as the cubes get small. **Against
Rips this is false.** The correct chain:

| Step | Statement | Error |
|---|---|---|
| 1 | Sublevel sets of $d_{X_n}(y)=\min_{x\in X_n}\|y-x\|$ are unions of balls; by the nerve theorem their PH **equals** Čech PH. | **exact** |
| 2 | $f_h$ = $d_{X_n}$ sampled on a grid of cell size $h$. Since $d_{X_n}$ is 1-Lipschitz, $\|f_h-d_{X_n}\|_\infty\le c_d h$; stability gives $d_B(\mathrm{Dgm}(f_h),\mathrm{Dgm}_{\text{Čech}})\le c_d h$. | $O(\sqrt d\, h)$, **vanishes** |
| 3 | Čech vs Rips are **multiplicatively** interleaved with constant $\vartheta_d=\sqrt{2d/(d+1)}$ in $\mathbb{R}^d$ (Jung), $2$ in general metric spaces. | **does not vanish with $h$** |

So `cubical(h) → Čech` and `Čech ↛ Rips`. The induced bottleneck floor against Rips is *multiplicative*,
hence proportional to feature death time: relatively worse for large-scale features, negligible for short
ones.

**This is not a problem, it is the paper's sharpest result.** It says:

- The equivalence margin $\delta$ has a computable lower bound $\delta_{\mathrm{floor}}(\text{reference})$.
- $\delta_{\mathrm{floor}}=0$ for a Čech/Alpha reference and $>0$ for a Rips reference.
- **Therefore Alpha is the correct reference filtration**, which is also the computationally sane choice in
  $d\le3$.

The plan below is built around this. Lemma A (step 2) and Lemma B (step 3, with a matching lower bound) are
the two headline theorems.

---

## 1. Contributions

| | Contribution | Status |
|---|---|---|
| **C1** | **Lemma A.** Explicit closed-form bottleneck (and $W_p$, via Skraba–Turner) bound between the cubical sublevel-set diagram at resolution $h$ and the Čech diagram, with the constant $c_d$ made explicit for both the V- and T-constructions. | `Claude.md` §6.1 identifies exactly this as absent from the literature. Provable. |
| **C2** | **Lemma B.** The Rips reference carries an irreducible multiplicative floor; construct point sets attaining it (sharpness). Corollary: an achievable-margin theorem. | New, and it corrects a premise the whole side-idea literature states loosely. |
| **C3** | **The noise-dominance margin.** Choose $\delta$ so that discretisation bias is dominated by the intrinsic sampling variability of the diagram. Turns an arbitrary tolerance into a principled one. | New. The most defensible answer to "where does $\delta$ come from?". |
| **C4** | **The paired equivalence test** and the resolution-selection procedure: finite-sample upper confidence bound, fixed-sequence testing up the $h$-ladder (FWER-free), honest treatment of non-monotonicity. | No calibrated filtration-equivalence test exists (`Concensus.md` grades this cell **GAP**). |
| **C5** | **Selection validity.** $\hat h$ is data-dependent; establish that downstream inference on fresh data is unaffected under sample splitting, and quantify the cost of reuse. | New; the thing practitioners will get wrong. |
| **C6** | Software + a machine-readable **"resolution certificate"** artifact. | — |
| **C7** | **The second ladder: PCA truncation** (Track 2D, added 2026-08-13). PH is invariant under full-rank PCA, so all topological loss comes from truncation; the truncation error is bounded by the **max** PCA residual while the field selects $k$ by **mean** variance explained. Same equivalence machinery, new ladder. | New. Directly answers a question an applied collaboration is already asking, and it is **independent of the Phase 0 gate**, so it survives even if cubical loses to Alpha. |

---

## 2. Source map (per CLAUDE.md §4)

**For Lemma A**
- Cohen-Steiner, Edelsbrunner, Harer, *DCG* 2007 — bottleneck stability $d_B(\mathrm{Dgm}(f),\mathrm{Dgm}(g))\le\|f-g\|_\infty$. **[A]** The engine.
- Skraba & Turner, arXiv:2006.16824 — $p$-Wasserstein stability in terms of the $p$-norm of the perturbation. **[A]** Gives the $W_p$ version of Lemma A, much tighter than the $\infty$-norm route.
- Wagner, Chen, Vuçini (2011/2012) — cubical PH computation. **[A]**
- Bleile, Garin, Heiss, Maggs, Robins, *Research in Comp. Topology 2* (2022) — **the two dual cubical constructions (V vs T) and how to convert between their diagrams.** **[A]** Load-bearing: "the cubical diagram" is not unique, and $c_d$ differs between constructions. Read before writing Lemma A.
- Kaji, Sudo, Ahara, arXiv:2005.12692 — Cubical Ripser. **[A]**
- Edelsbrunner & Harer, nerve theorem / union-of-balls. **[A]**

**For Lemma B**
- Chazal, de Silva, Oudot, *Geom. Dedicata* 2014 (arXiv:1207.3885) — persistence stability for geometric complexes, Rips/Čech/witness interleaving. **[A]** The primary source.
- de Silva & Ghrist — Rips–Čech interleaving with the Jung constant. **[B]** Verify the exact convention.
- Bauer & Edelsbrunner, *TAMS* 2017 — Morse theory of Čech and Delaunay complexes; Čech = Alpha at the level of PH. **[A]** This is what licenses using Alpha as the reference.
- Chazal, de Silva, Glisse, Oudot, *The Structure and Stability of Persistence Modules* (2016) — the interleaving formalism. **[A]**

**For C3 (margin) and C4 (test)**
- Schuirmann (1987) — TOST. **[A]**
- Lakens (2017) — modern primer with power analysis. **[A]**
- Lauzon & Caffo, *TAS* 2009 — multiplicity control for TOST. **[A]**
- Boulaguiem et al. (2024), $\alpha$-TOST — uniformly more powerful than standard TOST. **[B]**
- Krebs & Rademacher, arXiv:2401.10349 — two-sample tests for **relevant differences** in persistence diagrams under Wasserstein geometry. **[A]** **The closest existing work; the main comparator for C4.**
- Fasy, Lecci, Rinaldo, Wasserman, Balakrishnan, Singh, *AoS* 2014 — confidence sets for diagrams. **[A]**
- Roycraft, Krebs, Polonik, *AoS* 2023 — **naïve bootstrap fails**; smoothed bootstrap restores validity for persistent Betti numbers and Euler characteristics. **[A]** Mandatory if those statistics enter.
- Chazal, Glisse, Labruère, Michel, *JMLR* 2015 — minimax rates for diagram estimation. **[A]** Feeds the noise-dominance margin.
- Arnal, Cohen-Steiner, Divol (2024) — Čech diagrams converge in $W_p$ iff $p>m$ (manifold dimension). **[B]** A sharp caveat if $W_p$ is the discrepancy.

**For C5**
- Fixed-sequence / hierarchical testing (Maurer–Hommel–Bofinger). **[A]**
- Sample splitting and post-selection inference (Berk et al.; Fithian–Sun–Taylor). **[A]**

**Competitors for the Phase-0 viability gate (this is the list that decides the project)**
- Alpha complexes / Delaunay (GUDHI, CGAL) — **the real threat**: same PH as Čech, near-linear in practice in $d\le3$. **[A]**
- Sheehy, *DCG* 2013 — linear-size sparse Rips with explicit interleaving. **[A]**
- Koyama, Robins, Turner, arXiv:2412.07805 — distilled Rips, same PH, lower memory. **[A]**
- Graf et al., NeurIPS 2025 — the **flood complex**, PH on millions of points. **[B]** Verify; if real, it is a direct competitor to the entire premise.
- Leitão, arXiv:2602.22784 — cover refinements with $\log^3$ interleaving. **[B]**
- Otter, Porter, Tillmann, Grindrod, Harrington, *EPJ Data Science* 2017 — the standard complexity benchmark. **[A]**
- Dłotko & Gurnari, *GigaScience* 2023 — Euler curves/profiles for big data. **[B]**

---

## 3. Phases

Legend: **[SEQ]** · **[PAR-k]** · **[GATE]**.

### Phase 0 — **[GATE]** Viability: does cubical actually win? **(run this first, before anything else)**

*Goal:* two weeks, decisive. Either the premise holds in some regime, or the project is redirected before
any theory is written.

**The threat.** In $d\le3$, Alpha complexes give *exactly* Čech PH and run near-linearly in practice.
Cubical cost is $(L/h)^d$ cells, growing as $h^{-d}$. If Alpha dominates cubical at every realistic $(n,d)$,
the motivation for certified cubical substitution evaporates and the project must move to a regime where it
survives.

| # | Task | Deliverable |
|---|---|---|
| 0.1 | Benchmark harness: wall-clock, peak RSS, and diagram accuracy for {VR (Ripser), Alpha, sparse Rips, distilled Rips, flood complex, cubical at a ladder of $h$} × $d\in\{2,3,4,5\}$ × $n\in\{10^3,10^4,10^5,10^6,10^7\}$. | `bench/filtration_bench.py` + a runtime/memory atlas |
| 0.2 | Accuracy axis: for each method, bottleneck/$W_2$ distance to the Alpha (= Čech) ground truth where computable. | Accuracy-vs-cost Pareto frontier per $(n,d)$ |
| 0.3 | **Identify the regime where cubical wins**, if any. Candidates to check explicitly: (a) $d\ge4$ where Delaunay blows up; (b) natively gridded or functional data; (c) **DTM / density / KDE sublevel filtrations where Alpha does not apply at all**; (d) out-of-core and streaming regimes; (e) GPU-friendly workloads. | `docs/phase0_regime_verdict.md` |
| 0.4 | Verify the flood complex (Graf et al. 2025) claim empirically. If it does millions of points well, it must be the reference or a co-competitor. | Verdict note |
| 0.5 | **[GATE] decision.** | See below |

**Gate criterion.** There must exist at least one $(n,d,\text{filtration-function})$ regime where cubical at
a *useful* resolution is $\ge5\times$ faster or $\ge5\times$ lighter than the best available exact method at
comparable accuracy.

- **Pass, and the regime is DTM/density filtrations** → *most likely outcome.* Alpha is defined for
  distance-to-set only; DTM and KDE sublevel filtrations have no Alpha analogue, so cubical is the natural
  cheap route and the reference becomes "cubical at $h\to0$" or "weighted Alpha / DTM-filtration on a
  subsample". Reframe the paper around **DTM-cubical vs DTM-geometric**, which is also the more robust,
  more modern filtration anyway. Lemma A survives essentially unchanged ($\mathrm{DTM}$ is also Lipschitz).
- **Pass with plain distance filtration in $d\ge4$** → proceed exactly as written.
- **Fail** → the project pivots to *approximation-equivalence testing as a general method*, with cubical as
  one instance among sparse Rips, subsampling, cover refinements, and **PCA truncation (Track 2D)**. The
  statistical contributions (C3, C4, C5, C7) survive intact; only C1/C2 need re-targeting to whichever
  approximation you certify. **Track 2D is the ready-made landing site for this branch:** it needs no
  cubical complexes, it has an applied collaboration already asking for it, and its bound is sharper than
  Lemma A's.

**Deliverable:** the regime verdict + the cost/accuracy atlas. The atlas is a paper figure regardless of
outcome.

**Parallelisation:** **[PAR-0]** = {0.1 (compute agent, long-running, launch first), 0.2, 0.4} and 0.3 after
0.1–0.2 land. This phase is mostly one big compute job — start it and work on Phase 1 meanwhile.

---

### Phase 1 — Fix the objects **[SEQ, after 0]**

*Goal:* remove every ambiguity before proving anything. Half the confusion in the reviewed literature comes
from unstated conventions.

| # | Task | Notes |
|---|---|---|
| 1.1 | **Which filtration function?** $d_{X_n}$ (distance to point set), DTM$_m$, or KDE. Fix one primary (Phase-0 verdict decides; DTM is the likely and better answer) and state Lipschitz constants for each. | DTM is 1-Lipschitz too — Lemma A transfers. |
| 1.2 | **Which cubical construction?** V-construction (values on vertices) vs T-construction (values on top cells). They give *different* diagrams; Bleile et al. give the conversion. Fix one, state the other's constant. | $c_d$ depends on this. Getting it wrong invalidates Lemma A. |
| 1.3 | **Which reference?** Alpha (= Čech PH, Bauer–Edelsbrunner) as primary; Rips as a secondary reference where Lemma B's floor applies. | |
| 1.4 | **Which discrepancy?** $d_B$, $W_p$, or a functional $\sup_t|\Phi(D)(t)-\Phi(D')(t)|$. Recommend reporting $d_B$ (theory-native), $W_2$ (statistically better behaved), and the silhouette sup-norm (comparable to P1). | Note Arnal et al.: $W_p$ convergence for Čech needs $p>m$. Check whether it binds. |
| 1.5 | Conventions table (radius vs diameter for Rips, closed vs open balls, grid alignment/offset). | Put it in an appendix; referees will check it. |

**Deliverable:** `theory/WP1_conventions.md`. **Exit:** a single unambiguous object per slot.

---

### Phase 2 — Theory: Lemma A and Lemma B **[PAR-2, three tracks]**

*Goal:* the mathematical core. Per CLAUDE.md §5, every track runs its numerical falsification script
**before** the proof is finalised.

#### Track 2A — Lemma A (the $h$-bound)

| # | Task |
|---|---|
| 2A.1 | Bound $\|f_h - d_{X_n}\|_\infty$ for the chosen construction. For the V-construction with vertex sampling and a 1-Lipschitz $f$ on a cubic grid of side $h$, the natural constant is the cell circumradius $\sqrt d\,h/2$; verify whether the T-construction gives $\sqrt d\,h$ or a different constant. |
| 2A.2 | Apply CEH stability → $d_B(\mathrm{Dgm}(f_h),\mathrm{Dgm}_{\text{Čech}})\le c_d h$. State $c_d$ explicitly. |
| 2A.3 | Skraba–Turner route → the $W_p$ version, which is much tighter and less outlier-pessimistic. Requires a total-persistence bound; supply one or state the condition. |
| 2A.4 | **Sharpness:** construct point configurations where the bound is attained up to a constant, or prove the constant cannot be improved. |
| 2A.5 | **Numerical falsification:** sweep $h$, measure $d_B$ empirically on ≥6 shape families, plot against the bound. **The empirical curve must sit below the bound at every $h$ and scale linearly.** If it does not, the proof is wrong; find out here. |
| 2A.6 | Extend to DTM and KDE filtration functions. |

#### Track 2B — Lemma B (the Rips floor)

| # | Task |
|---|---|
| 2B.1 | State the Čech↔Rips multiplicative interleaving with the exact convention fixed in 1.5; derive the induced bottleneck bound on the linear scale (it scales with death time, so express it as a *relative* floor). |
| 2B.2 | **Lower bound / sharpness:** construct explicit point sets in $\mathbb{R}^d$ where the Čech–Rips bottleneck gap is $\Theta(\vartheta_d - 1)\times$ feature scale, proving the floor is real and not an artefact of the bound. |
| 2B.3 | **Corollary (achievable-margin theorem):** for a Rips reference, no resolution $h$ can certify equivalence at margin $\delta<\delta_{\mathrm{floor}}$; give $\delta_{\mathrm{floor}}$ in closed form as a function of $d$ and the largest feature scale of interest. |
| 2B.4 | **Numerical falsification:** compute Čech-vs-Rips bottleneck across $d=2,\dots,6$ and confirm it plateaus (does **not** decay) as $n\to\infty$ and as any grid is refined. **This is the single most important sanity check in the project — it either confirms the correction in §0 or overturns it.** Run it in week one. |
| 2B.5 | Practical corollary: which reference to use, and the guidance table. |

#### Track 2C — Error decomposition

| # | Task |
|---|---|
| 2C.1 | Decompose total error into (i) discretisation $O(c_d h)$, (ii) sampling error of the cloud as an estimate of the underlying shape, (iii) the Čech–Rips offset. |
| 2C.2 | Sampling-error scale: use minimax rates for diagram estimation (Chazal et al. 2015) to get the order of (ii) in terms of $n$ and the manifold dimension. |
| 2C.3 | **The crossover:** the $h$ at which (i) drops below (ii). **This is the theoretical prediction that Phase 4's noise-dominance margin operationalises, and that Phase 6 must recover empirically.** |

#### Track 2D — The second ladder: PCA truncation **(added 2026-08-13)**

*Why this is here.* P2 is not really about cube size. It is about **any approximation-parameter ladder that
carries a stability bound**, tested for statistical equivalence against an exact reference. Grid resolution
$h$ is the first instance. **The number of retained principal components $k$ is the second**, and for the
ecological collaboration (P1 Phases 11–12) it is the one that is actually asked about: the standard niche
pipeline reduces $\sim20$ environmental variables to PC1–PC3 before computing anything, and the group has
said explicitly that they are working on understanding the effects of that step.

The mathematics is short and worth stating exactly, because it is sharper than the $h$ case:

1. Persistent homology of a Rips or Čech filtration depends only on the pairwise distance matrix.
2. Full-rank PCA is a centring plus an orthogonal rotation, so it **preserves all pairwise distances
   exactly** and the diagram is *identical*. **Every topological consequence of "doing PCA" comes from
   truncation alone.**
3. Truncation is a contraction. With $V_k$ the retained loadings and $V_\perp$ the discarded ones,
   $$0 \;\le\; \|x_i-x_j\| - \|V_k^\top(x_i-x_j)\| \;\le\; \|V_\perp^\top(x_i-x_j)\| \;\le\; 2\max_i\|V_\perp^\top x_i^c\|.$$
4. A uniform sup-norm metric distortion $r$ gives an $r$-interleaving of the Rips filtrations, hence
   $d_B(\mathrm{Dgm}_{\text{full}},\mathrm{Dgm}_{\mathrm{PC}k})\le r$, computable from the PCA residuals
   alone without ever running PH in full dimension.

| # | Task |
|---|---|
| 2D.1 | Prove the chain above properly, including the interleaving step and the $W_p$ analogue via Skraba–Turner. State it for Rips, Čech and DTM-Rips; note that Alpha and cubical are **infeasible** at $d\approx20$, which is why this track lives on the Rips side. |
| 2D.2 | **The corollary that makes it publishable:** the bound is driven by the **maximum** residual norm, whereas the conventional stopping rule (variance explained) controls the **mean squared** residual $\sum_{l>k}\lambda_l$. So a dataset can sit at 95% variance explained and still have its topology moved by a handful of extreme points. **Variance-explained is the wrong criterion for topological faithfulness.** Give a witness dataset. |
| 2D.3 | Tightness study: how loose is the max-residual bound in practice? Report the ratio (empirical $d_B$) / (bound) across DGPs and real data. **Expect it to be loose**; the honest framing is a cheap screen, with the empirical $d_B$ carrying the applied claims. |
| 2D.4 | Port Phases 3–5 from the $h$-ladder to the $k$-ladder verbatim: paired equivalence test on $R_i(k)=d(D_i^{\text{full}}, D_{i,k})$, noise-dominance margin, fixed-sequence testing **up** the ladder. **The $k$-ladder is deterministically monotone in $k$ (residuals shrink), which makes the multiplicity scheme strictly cleaner than the non-monotone $h$-ladder.** |
| 2D.5 | Intrinsic-dimension connection: if the environmental covariates are functionally dependent (temperature is predictable from latitude/longitude, etc.), the cloud sits near a low-dimensional manifold and the effective $k$ is set by intrinsic dimension, not by a variance threshold. Compare a $k$ chosen by an intrinsic-dimension estimator against $k$ chosen by 80/90/95% variance. |
| 2D.6 | **Numerical falsification** before writing any proof (CLAUDE.md §5): sample from a known manifold embedded in $\mathbb{R}^{20}$ with a known hole, truncate to $k=2,3,\dots,20$, and check that empirical $d_B$ is (a) monotone non-increasing in $k$ and (b) below the bound at every $k$. |
| 2D.7 | Deliver the **truncation certificate** consumed by P1 task 12.5: given a dataset and a $k$, return the bound, the empirical $d_B$, the equivalence verdict at margin $\delta$, and the $k^\star$ at which certification first succeeds. |

**Why this track de-risks the whole project.** Phase 0's gate asks whether cubical beats Alpha, and it may
not. Track 2D involves no cubical complexes at all, so **it survives every branch of the Phase 0 gate**,
including the branch where cubical loses outright. If Phase 0 fails, 2D plus the DTM pivot is the paper.

**Deliverables:** `theory/WP2A_h_bound.md`, `theory/WP2B_rips_floor.md`, `theory/WP2C_decomposition.md`,
`theory/WP2D_truncation.md` + four falsification scripts.
**Exit:** both lemmas proved with sharpness examples; all four numerical checks consistent with the theory.

**Parallelisation:** **[PAR-2]** = {2A, 2B, 2C, 2D} — four fully independent agents. **But run 2B.4 first, as
a one-day standalone job, before committing agents**: if it overturns §0, tracks 2A and 2C change. 2D is
unaffected by 2B.4's outcome and by the Phase 0 gate, so it is the safest agent slot to fill first if you
want work proceeding while those two decisions are pending.

**Sequencing against P1.** 2D.7's certificate is an input to P1 task 12.5, which sits after P1 Phase 10.
There is a lot of slack, so do not let 2D pull P2 out of order. If P2 is behind, P1 Phase 12.5 degrades
gracefully to reporting the empirical $d_B$ without the certificate.

---

### Phase 3 — The paired equivalence test (C4) **[PAR-A, after 1]**

*Goal:* a finite-sample-honest test of $H_0:\mathbb{E}[R(h)]\ge\delta$ vs $H_1:<\delta$.

| # | Task | Notes |
|---|---|---|
| 3.1 | Setup: validation clouds $i=1..m$, paired discrepancies $R_i(h)=d(D_i^{\mathrm{ref}},D_{i,h}^{\mathrm{cub}})\ge0$, bounded above by the Lemma-A bound. | Boundedness is a gift: it enables finite-sample concentration. |
| 3.2 | **Upper confidence bound on $\mathbb{E}[R(h)]$.** Three routes, implement all: (a) empirical Bernstein / Hoeffding using the Lemma-A range (finite-sample, conservative); (b) paired bootstrap-$t$ (asymptotic, tighter); (c) $\alpha$-TOST style finite-sample correction. Report all three. | (a) is the one that lets you claim a *guarantee* rather than an approximation. Lead with it. |
| 3.3 | **Functional version:** $\sup_t|\Delta_h(t)|\le\delta$ via a simultaneous band on the paired silhouette difference (multiplier bootstrap). Localises *at which scales* the filtrations disagree. | High interpretive value: "they agree for features below persistence $p$, disagree above". |
| 3.4 | Per-degree versions $d=0,1,2$ with multiplicity control (Lauzon–Caffo for TOST multiplicity; a multivariate-TOST variant for simultaneous degrees). | |
| 3.5 | Compare against Krebs–Rademacher's relevant-difference test, which is the closest existing object. State clearly what is different: theirs is two-sample and unpaired, ours is paired and margin-anchored to an approximation bound. | Mandatory positioning; do not let a referee find this comparison first. |
| 3.6 | Power analysis: $m$ required to certify at margin $\delta$ given the discrepancy distribution. Give a design formula. | Practitioners need "how many validation clouds do I need?". |

**Deliverable:** `tda2s/equiv/paired_tost.py`; size/power tables.
**Exit:** finite-sample route (a) never exceeds nominal size across ≥5 DGP families; bootstrap route (b)
within $[0.03,0.08]$.

---

### Phase 4 — The margin (C3) **[PAR-A, after 1; needs 2C for the theoretical variant]**

*Goal:* answer "where does $\delta$ come from?" three ways, and recommend one.

| # | Task | Notes |
|---|---|---|
| 4.1 | **Theory margin:** $\delta = c_d h$ from Lemma A. Self-consistent but circular for *selecting* $h$; useful as a sanity ceiling. | |
| 4.2 | **Noise-dominance margin (recommended, C3).** Set $\delta$ = the intrinsic sampling variability of the reference diagram itself, estimated by resampling the cloud (subsample or smoothed-bootstrap) and measuring the spread of $d(D^{\mathrm{ref}}, D^{\mathrm{ref},*})$. **Reading:** "discretisation error at resolution $h$ is no larger than the noise you already have." | This is the paper's most quotable idea. It makes $\delta$ data-driven and interpretable rather than arbitrary. |
| 4.3 | Validity of 4.2: $\delta$ is now estimated, so the equivalence test inherits its uncertainty. Either (i) use a lower confidence bound on $\delta$ (conservative, valid), or (ii) derive a joint procedure. **Implement (i); attempt (ii).** | The obvious referee attack. Handle it explicitly. |
| 4.4 | **Scientific margin:** $\delta$ = the smallest persistence the user cares about. Trivial to state, and the right default when domain knowledge exists. | |
| 4.5 | Sensitivity: how does the selected $\hat h$ move across the three margins? Report the full curve $\hat h(\delta)$ rather than a single number. | The honest deliverable is a curve, not a scalar. |
| 4.6 | **Numerical falsification:** on DGPs with known ground-truth topology, check that $h$ chosen by the noise-dominance margin preserves the true Betti numbers and the true persistent features. | If it does not, 4.2 is wrong and must be re-derived. |

**Deliverable:** `tda2s/equiv/margins.py` + the $\hat h(\delta)$ sensitivity curve.

---

### Phase 5 — Resolution selection and its validity (C5) **[SEQ, after 3+4]**

| # | Task | Notes |
|---|---|---|
| 5.1 | The $h$-ladder $h_1<h_2<\dots<h_K$ and the selection rule $\hat h=\max\{h_k: \mathrm{UCB}_{1-\alpha}(\mathbb{E}R(h_k))\le\delta\}$. | |
| 5.2 | **Multiplicity.** Primary: **fixed-sequence testing from the finest $h$ upward, stopping at the first non-rejection — FWER control at $\alpha$ with no correction at all.** Secondary: a simultaneous bootstrap band over $h$, for when the fixed sequence is inappropriate. | The fixed-sequence trick is elegant and free; make it the default. |
| 5.3 | **Non-monotonicity.** $\mathbb{E}[R(h)]$ need not be monotone in $h$ (`Perplexity.md` §5 is right). Fixed-sequence testing implicitly assumes it. Options: (a) state monotonicity as an assumption and **test it empirically**; (b) fall back to simultaneous bands when it fails. Quantify how often it fails in practice. | Do not paper over this. |
| 5.4 | **Selection validity (C5).** $\hat h$ is data-dependent. Prove that using $\hat h$ on *fresh* data leaves downstream inference valid (sample splitting), and empirically quantify the type-I inflation if the validation set is reused. | Give the inflation number. It is the thing practitioners need to be scared by. |
| 5.5 | The **resolution certificate**: a machine-readable artifact recording $(\hat h, \delta, \alpha, m$, reference filtration, construction, degree, discrepancy, the UCB, the monotonicity check$)$. | Turns the method into something citable in someone else's methods section. Small effort, disproportionate impact. |

**Deliverable:** `tda2s/equiv/select.py`; certificate schema; the selection-validity result.

---

### Phase 6 — Simulation study **[SEQ, after 3,4,5]**

| # | Task |
|---|---|
| 6.1 | Size and power of the equivalence test across ≥6 shape families (circles, tori, spheres, linked tori, Klein bottle embedding, Swiss roll, clustered/multimodal, real-derived) × $d\in\{2,3,4\}$ × $m\in\{20,50,100,200\}$ × noise and outlier levels. |
| 6.2 | **Recovery of the theoretical crossover.** Does the selected $\hat h$ match Phase 2C.3's predicted crossover? **This is the study that validates the whole theory-to-practice chain.** |
| 6.3 | Coverage of the certificate: across repeated validation sets, how often does the certified $\hat h$ actually satisfy $\mathbb{E}[R(\hat h)]\le\delta$ on fresh data? Target $\ge1-\alpha$. |
| 6.4 | Downstream fidelity: at the certified $\hat h$, do scientific conclusions (Betti numbers, a downstream classifier, a P1-style two-sample test) match those at the reference? |
| 6.5 | Failure-mode catalogue: where does certification fail or mislead? Outliers, non-uniform density, thin features near the grid scale, anisotropy, features smaller than $h$. |
| 6.6 | Cost/accuracy Pareto curve at the certified $\hat h$ vs the Phase-0 competitors. **The practical payoff figure.** |

**Deliverable:** master tables; the crossover-recovery figure; the failure catalogue.
**Parallelisation:** **[PAR-6]** shards by shape family — 6 agents. Checkpoint per cell.

---

### Phase 7 — Large-scale application **[SEQ, after 6]**

| # | Task |
|---|---|
| 7.1 | Pick 2–3 genuinely large datasets in the Phase-0 winning regime. Candidates: LiDAR / airborne point clouds ($10^7$ points, $d=3$), 3D medical volumes (natively gridded, so the cubical side is free), cosmological large-scale structure catalogues, molecular dynamics trajectories. |
| 7.2 | Run the full pipeline: certify $\hat h$ on a validation subsample, apply at scale, report the runtime and memory saving against the best exact method. |
| 7.3 | Where an exact reference is computable on a subsample, verify the certificate holds out of sample. |
| 7.4 | Report at least one case where certification **refuses** (no admissible $h$ at the requested margin). A method that never says no is not a test. |

---

### Phase 8 — Software and manuscript **[SEQ, last]**

8.1 `tda2s.equiv` release: API, tests, docs, a one-command "certify my resolution" entry point ·
8.2 Assemble theory (Phases 2, 4, 5) · 8.3 Related work: interleaving/approximation × equivalence testing ×
scalable PH · 8.4 Explicit correction section on the Rips-vs-Čech premise (a service to the field; write it
generously, not combatively) · 8.5 Open problems: multiparameter filtrations, adaptive/non-uniform grids,
certification under distribution shift · 8.6 Target: **SIAM J. Mathematics of Data Science**; alternatives
*Journal of Applied and Computational Topology*, *Annals of Applied Statistics*.

---

## 4. Dependency graph and parallelisation

```
0 [GATE] viability  ◄── RUN FIRST, 2 weeks, decides the whole project
 │      (2B.4 sanity check runs alongside, 1 day)
 ▼
1 (fix conventions) ──────────────┐
 │                                │
 ▼                                ▼
┌──── PAR-2: three theory agents ────┐   ┌──── PAR-A: two agents ────┐
│  2A (h-bound)                      │   │  3 (equivalence test)     │
│  2B (Rips floor)                   │   │  4 (margins) ◄─ needs 2C  │
│  2C (error decomposition)          │   └───────────┬───────────────┘
└────────────────┬───────────────────┘               │
                 └──────────────┬────────────────────┘
                                ▼
                        5 (selection + validity)
                                │
                                ▼
                        6 (simulation)  ◄── PAR-6: 6 sharded agents
                                │
                                ▼
                        7 (large-scale) ──► 8 (manuscript + software)
```

**Parallelises well:**
- Phase 2's three theory tracks are fully independent — three agents.
- Phase 3 ∥ Phase 4 (except 4.1, which needs 2A, and 4.2's validation, which needs 2C).
- Phase 6 shards by shape family — six agents.
- Phase 0's benchmark grid shards by $(n,d)$ cell.

**Must not be parallelised:**
- **2B.4 before everything.** One day of compute confirms or overturns §0's correction. If Čech-vs-Rips
  bottleneck turns out to decay, Lemma B disappears and the project's framing changes. Run it in week one,
  alone.
- **Phase 1 before Phase 2.** Three agents proving lemmas under three different cubical conventions produce
  three mutually incompatible constants. This has to be locked first.
- **Phase 5 after both 3 and 4.** Selection depends on both the test and the margin; building it early
  guarantees a rewrite.
- **Phase 6 after 5 is frozen.** Same reasoning as P1 Phase 7: re-running the master grid is the most
  expensive available mistake.

---

## 5. Compute budget (per CLAUDE.md §2)

Cap at 16 workers. Cubical PH memory scales as $(L/h)^d$ — this is the binding constraint.

| Job | Cost | Strategy |
|---|---|---|
| Phase 0 benchmark grid | ~24–48 h | Shard by $(n,d)$; **hard RSS guard: skip any cell projected over 24 GB**; checkpoint per cell |
| Phase 2 falsification sweeps | ~8 h | Embarrassingly parallel over $h$ and shape family |
| Phase 6 master grid | ~30 h | Shard by shape family; two-stage coarse-then-fine |
| Phase 7 at $10^7$ points | memory-bound, not CPU-bound | Out-of-core / chunked cubical; run overnight, single job, low worker count |

**Memory discipline (critical here, more so than in P1):**
- Fine grids in $d=3$ at $h$ small enough to matter can exceed RAM. Compute a projected cell count
  **before** allocating and refuse cells over budget.
- Cache all reference (Alpha/VR) diagrams to disk keyed by cloud hash — they are computed once and reused
  across every $h$ in the ladder and every bootstrap replicate.
- The paired bootstrap resamples *clouds*, not diagrams; never recompute PH inside the bootstrap loop.

---

## 6. Cross-cutting risks

1. **Phase-0 gate failure (highest).** Alpha may simply dominate in $d\le3$. Mitigation: the gate is
   deliberately first and cheap, and the DTM/density-filtration escape route is already identified. Expect
   to reframe toward DTM; that is a *better* paper anyway.
2. **The cubical construction ambiguity.** V vs T give different diagrams. Getting this wrong invalidates
   Lemma A. Mitigation: Phase 1.2, Bleile et al. read in full before proving anything.
3. **Sharpness may be hard.** If 2A.4 or 2B.2 does not close, ship the upper bounds with numerical evidence
   of tightness and state sharpness as open. Acceptable; do not let it block Phase 3.
4. **Estimated margin (4.2/4.3).** The noise-dominance margin is the best idea in the project and also its
   softest statistical point. The conservative lower-confidence-bound version (4.3(i)) must work even if the
   joint procedure does not.
5. **Non-monotonicity in $h$.** Handle explicitly in 5.3; do not assume it away silently.
6. **$W_p$ caveat.** Arnal et al.: Čech diagrams converge in $W_p$ only for $p>m$. Check whether this binds
   for the chosen $p$ and manifold dimension; if it does, use $d_B$ or $p$ large.
7. **Referee objection: "this is just stability plus TOST."** It partly is. The defence is Lemma B (the
   non-obvious correction), the noise-dominance margin, and selection validity. Make those three prominent
   in the abstract.

---

## 7. First five concrete actions

1. **Run 2B.4 alone, this week.** Compute Čech-vs-Rips bottleneck across $d=2..6$ and increasing $n$;
   confirm it plateaus. One day. It validates or kills §0's central correction.
2. Launch Phase 0.1's benchmark grid as a long-running background job (it is the gate and it takes days).
3. Read Bleile et al. (2022) on the dual cubical constructions before writing a single line of Lemma A.
4. Read Krebs & Rademacher arXiv:2401.10349 in full — it is the closest prior art to C4 and determines how
   you must position the test.
5. Decide, from Phase 0.3, whether the paper is about distance filtrations or DTM filtrations. Everything
   downstream branches on that answer.
