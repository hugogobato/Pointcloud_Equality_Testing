# Decision Memo: One Project or Two?

**Date:** 2026-08-09
**Input:** `Literature_Review/{Claude,Perplexity,Genspark,Grok,Deepseek,Manus,Concensus}.md`
**Question:** Should (1) TATE-based topological two-sample testing and (2) filtration-equivalence /
cube-size calibration be one research project or two?

---

## Verdict

**Two papers, one shared codebase.** Plus an optional third short paper that composes them.

| | **P1 — Topological Two-Sample Testing** | **P2 — Certified Filtration Substitution** |
|---|---|---|
| Plan file | `RESEARCH_PLAN_P1_TwoSample.md` | `RESEARCH_PLAN_P2_FiltrationEquivalence.md` |
| Null | difference: contrast $=0$ | **equivalence**: discrepancy $\ge \delta$ |
| Design | two independent groups | **paired** within-unit |
| Causal content | central (propensity, DR, ignorability, TCATE) | **vacuous** (deterministic, unconfounded assignment) |
| New mathematics | identification + local power | **approximation theory** (interleaving in $h$) |
| Dominant risk | "Kim & Lee already did it" | "cubical doesn't actually win vs Alpha" |
| Venue | JMLR / NeurIPS / ICML / *Biometrika*-adjacent | SIMODS / JACT / AoAS |

---

## Why not one project

**1. The nulls are not reparameterisations of each other.**
P1 tests $H_0:\Delta = 0$. P2 tests $H_0:\mathbb{E}[R(h)] \ge \delta$ against $H_1: <\delta$. An equivalence
test needs a margin, a one-sided upper confidence bound, its own power curve, and its own multiplicity
scheme over the resolution ladder. None of that is machinery P1 supplies. Every review that thought about
it carefully reached this same split (`Perplexity.md` §Procedure A/B; `Genspark.md` §6.3; `Claude.md` §7).

**2. The designs differ.**
P1: $n_1$ clouds in group 1, $n_0$ in group 0, independent. P2: $m$ validation clouds, each yielding a
*pair* $(D_i^{\mathrm{ref}}, D_{i,h}^{\mathrm{cub}})$. Pairing is not a detail — it is where most of P2's
power comes from, and it changes the resampling scheme, the variance estimator, and the asymptotics.

**3. The causal apparatus is the entire selling point of P1 and contributes nothing to P2.**
In P2 the "treatment" (which filtration) is assigned deterministically to the *same* unit. Propensity is
constant, there is no confounding, AIPW degenerates to a paired difference, and topological ignorability
is trivially satisfied. Wrapping P2 in causal language would be decoration that a referee will strip out.

**4. P2 requires approximation-theory results P1 never touches.**
The load-bearing object is an explicit bottleneck bound in the cube side length $h$, plus the
Čech-vs-Rips floor. That is a computational-topology contribution with its own literature (nerve theorems,
interleaving, sparse Rips). It has no counterpart in P1.

**5. Risk isolation.**
P1's failure mode is a novelty problem; P2's is a theorem-or-benchmark problem. They are uncorrelated.
Coupling them means either failure sinks a single large paper; splitting them means one survives.

---

## Why they still share a codebase

Both need: the PH pipeline (GUDHI / Ripser / Cubical Ripser / Alpha), the vectorisation stack
(power-weighted silhouettes, landscapes, Betti/Euler curves, persistence images), a resampling engine
(permutation, multiplier bootstrap, paired bootstrap), a controlled point-cloud DGP harness, and wrappers
around the competitor tests. Building this twice is waste. **Phase 0 is shared and gates both projects.**

---

## Two corrections to the premises, before you start

### (a) "Cubical and VR converge as cube size $\to 0$" is false against Rips

The correct chain is:

1. Sublevel sets of the distance-to-point-set function $d_{X_n}$ are unions of balls; by the nerve theorem
   their PH equals **Čech** PH. Exact, no error.
2. Sampling $d_{X_n}$ on a grid of cell size $h$: since $d_{X_n}$ is 1-Lipschitz,
   $\|f_h - d_{X_n}\|_\infty \le c_d h$, so by the stability theorem
   $d_B(\mathrm{Dgm}(f_h), \mathrm{Dgm}_{\text{Čech}}) \le c_d h \to 0$. **This is the provable lemma
   `Claude.md` §6.1 correctly identified as missing from the literature.**
3. **Čech vs Rips are only multiplicatively interleaved**, with constant $\vartheta_d=\sqrt{2d/(d+1)}$
   in $\mathbb{R}^d$ (Jung's theorem) and $2$ in a general metric space. **This gap does not shrink
   with $h$.**

So `cubical(h) → Čech` as $h\to0$, and `Čech ≠ Rips` by a non-vanishing, *multiplicative* factor. On the
linear scale the induced bottleneck floor is proportional to feature death time, so it is relatively worse
for large-scale features.

**Consequence, and it is a good one:** the equivalence margin $\delta$ has a computable lower bound
$\delta_{\text{floor}}$ whenever the reference is Rips, and $\delta_{\text{floor}}=0$ when the reference is
Čech/Alpha. That is a sharp, publishable statement, and it makes Alpha the correct reference filtration.
It is now **P2 Lemma B**, a headline result rather than a footnote.

### (b) Without covariates, the causal reframing of P1 is a relabelling

With $A$ assigned by group and no confounders, the propensity is constant, the DR estimator collapses to a
difference of group summaries, and $H_0:\mathrm{TATE}=0$ coincides exactly with the Robinson–Turner and
Fréchet-ANOVA nulls. `Claude.md` §8.2 states this plainly and is right.

The causal machinery earns its keep **only under covariate imbalance**. P1 is therefore built around that,
with an explicit kill-gate (Phase 2): if the DR test does not measurably beat permutation and MMD under
covariate shift, the causal wrapper is dropped and the paper becomes the distribution-level +
conditional-testing contribution alone.

---

## Scope boundary against your existing `CP_TATE` project

These are close cousins and the delineation must be explicit in both manuscripts, or a referee (or Ioannis)
will read it as salami slicing.

| | `CP_TATE` | `Pointcloud_Equality_Testing` P1 |
|---|---|---|
| Question | **How uncertain** is the effect? | **Is there** an effect? |
| Output | confidence / prediction **bands** | **p-values**, size, power |
| Estimands | TATE, CTATE, ITTE | standardized two-sample null, distribution-level null, conditional null |
| Novelty engine | conformal prediction, functional bands | covariate-shift diagnosis, distribution-level test, local power |
| Shared | AIPW estimator, silhouette pipeline, DR-learner | reuse from `tcda_uq`, do not rebuild |

**Rule:** P1 imports `CP_TATE`'s functional DR-learner and AIPW code rather than reimplementing it, and cites
it. P1 never claims a banding contribution; `CP_TATE` never claims a testing contribution.

**The import surface is now concrete.** `CP_TATE` is released as
[`tcda_uq`](https://github.com/hugogobato/tcda_uq) (MIT, Python 3.10):

```
uv pip install "tcda_uq @ git+https://github.com/hugogobato/tcda_uq.git"
```

P1 takes `tcda_uq.estimators` (plug-in / IPW / AIPW, `cross_fit`, functional nuisance regression, CTATE
DR-learner), `tcda_uq.silhouette`, and `tcda_uq.datasets.TriOracleSimulation` (an oracle DGP with known
TATE, CTATE and ITTE, useful for validating P1's own harness). The precise handoff: P1's statistic is
$T_n=\sqrt n\,\|\hat\psi_d\|_\infty$ read off `cross_fit(...).aipw[d]`, calibrated by multiplier bootstrap
over `cross_fit(...).influence()[d]`. Everything downstream of that (band construction, Liebl–Reimherr,
Pini–Vantini, conformal prediction) is `CP_TATE`'s and P1 must not claim it. See P1 Phase 0.8.

---

## Optional Paper 3 (do not start until P1 and P2 both land)

*Filtration-aware topological two-sample testing.* Compose P2's certified $\hat h$ into P1's test and prove
the two-stage procedure controls type I error accounting for resolution selection (sample-split, or a
selective-inference correction). Short, clean, and only worth writing if both parents exist. Roughly 6 weeks
of work; it is a *Journal of Applied and Computational Topology* note, not a flagship.

---

## The ecological collaboration (added 2026-08-13)

Morimoto, Jānis and Joseph brought an application: species **niche hypervolumes**. Occurrence points for a
species, environmental covariates extracted at each location, PCA, PC1–PC3 as the coordinate system, and the
resulting cloud is the hypervolume.

**Decision: methods first, application second.** It becomes **P1 Phases 11 and 12**, a separate companion
manuscript, and it does not enter Phases 0–10 except through a one-day forward-compatibility check (P1 task
0.9). The reason is directional: the applied analysis cannot say anything until the DR test exists and is
calibrated, whereas the methods paper never needs the ecology. Running them together lets one dataset's
idiosyncrasies drive the estimand, and niche data has no ground truth against which a size-power claim could
be checked.

Three things the application changes, all upgrades:

1. **P1's Phase 2 gate stops being hypothetical.** Sampling effort (GBIF record counts differ by orders of
   magnitude between well-surveyed and poorly-surveyed regions) is a confounder that is correlated with
   essentially any treatment you can define *and* mechanically alters the persistence diagram. That is
   exactly the structure C1 predicts existing tests fail on.
2. **P1's bake-off gains named incumbents.** The Warren–Glor–Turelli niche-equivalency permutation test and
   Broennimann's PCA-env overlap statistics test the $H_0^{\mathrm{cond}}$ null and adjust for nothing.
   Verify both in P1 task 0.7.
3. **P2 gains a second, safer ladder.** PH is invariant under full-rank PCA, so every topological effect of
   dimensionality reduction comes from truncation, and the truncation error is controlled by the **maximum**
   PCA residual while the field selects $k$ by **mean** variance explained. That is P2 Track 2D / C7, and
   unlike the cubical story it is independent of P2's Phase 0 viability gate.

**One thing to settle with the co-authors now, at zero cost:** which contrast. P1's causal machinery needs
cloud-level replication ($\ge30$ species per arm). A two-species comparison, which is what the ecology
literature defaults to, has $n=1$ vs $n=1$ and routes to P1 Phase 8 instead, where the causal apparatus is
decoration. This is gate 11.1, and it has weeks of data-acquisition lead time riding on it.

---

## Suggested calendar

```
Phase 0 (shared, ~3 wks)
   │
   ├──► P2 Phase 0 viability gate (2 wks, CHEAP + DECISIVE — run this early)
   │        ├──► P2 Phases 1-8
   │        └──► P2 Track 2D (k-ladder) — gate-independent, safe to start anytime
   │
   └──► P1 Phases 1-2 (incl. kill-gate)
            └──► P1 Phases 3-10  ── METHODS PAPER
                     │
                     ├──► P1 Phase 11 (ecological data track, starts during Phase 7)
                     │
                     └──► P1 Phase 12 (ecological analysis) ── COMPANION PAPER
```

Run **P2's Phase 0 viability gate first**, even though P1 is the higher-profile paper. It costs two weeks
and it either validates P2's entire motivation or redirects it before you spend months on theory.

Two conversations cost nothing and should happen before either gate: **gate 11.1** with the ecology
co-authors, and a check that P2 Track 2D is not already in the dimensionality-reduction literature they are
reading.
