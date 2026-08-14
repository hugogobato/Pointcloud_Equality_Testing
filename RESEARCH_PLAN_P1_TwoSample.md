# Research Plan P1 — Covariate-Adjusted Topological Two-Sample Testing

**Working title:** *Do These Point Clouds Differ Topologically? Covariate-Adjusted, Doubly Robust, and
Conditional Two-Sample Tests for Persistent Homology*

**One-line thesis.** Every existing topological two-sample test asks whether
$\mathcal{L}(D\mid A=1)=\mathcal{L}(D\mid A=0)$. Under covariate imbalance that null is the wrong target:
it fires on covariate shift and it can be silent when a real topological difference exists. Recasting the
comparison as a topological causal contrast fixes the target, buys double robustness and semiparametric
efficiency, and unlocks two tests nobody currently has: a **distribution-level** test and a **conditional
(TCATE)** test.

---

## 0. Honest positioning (read this before writing any code)

**What already exists.** Kim & Lee (arXiv:2603.02289) define
$\psi_d(t)=\mathbb{E}[\phi(t;D^1_{i,d})-\phi(t;D^0_{i,d})]$ on power-weighted silhouettes, build an
efficient doubly robust estimator, prove functional weak convergence in $\ell^\infty$, and construct a
multiplier-bootstrap test of $H_0:\psi_d\equiv0$. **In the group-comparison setting, that is already the
test.** Do not reinvent it.

**What is a relabelling.** With $A$ an exogenous group indicator and no covariates, exchangeability holds
by design, the propensity is constant, AIPW degenerates to a difference of group means, and the TATE null
coincides with Robinson–Turner and with Fréchet ANOVA. Any claim of novelty in that regime is false and
will be caught.

**What is genuinely open, and is therefore this paper.**

| | Contribution | Status in literature |
|---|---|---|
| **C1** | Covariate shift breaks the conditional-law null: existing topological two-sample tests neither control size nor retain power under imbalance. Stated as a theorem plus witness DGPs. | **Unremarked.** No TDA testing paper adjusts for covariates. |
| **C2** | The **distribution-level** test of $\delta_{\mathrm{dist}}=0$ (topologize the *interventional law*, not the individual outcomes). | Estimand defined in Souto & Diamantis (2607.28161); **no test exists.** |
| **C3** | **Conditional / TCATE** testing: is there *any* subgroup with a topological difference, and is the effect heterogeneous in $X$? | **Absent entirely** from TDA testing. |
| **C4** | Local power theory for the DR functional test + first honest bake-off against RT, MMD, STRAND, Han et al., Fréchet ANOVA, Moon–Lazar. | Power analysis is rare (only Han et al. 2607.20893 have minimax results). |
| **C5** | Software: `tda2s`, importing `tcda_uq` (the released `CP_TATE` library) for AIPW, cross-fitting and the DR-learner. | — |

**Scope discipline.** C1 is the paper's spine. C2 and C3 are the methodological payload. If Phase 2's
kill-gate fails, C1 collapses and the paper retreats to C2+C3+C4 as a "two tests TDA does not have" paper,
which is still publishable but at a lower tier.

**Sequencing decision (2026-08-13).** The methods paper is built first, on synthetic DGPs and a
methodological real-data example (Phase 9). The ecological collaboration with Morimoto, Jānis and Joseph
(species niche hypervolumes) is a **deferred, downstream track**: Phases 11 and 12, after the methods
paper is written. The reason is directional. The applied analysis cannot say anything until the DR test
exists and is calibrated, whereas the methods paper never needs the ecology. Running them together would
let one dataset's idiosyncrasies drive the estimand. **A cold agent working Phases 0–10 should treat the
ecology as out of scope**, with the single exception of task 0.9, which is a one-day forward-compatibility
check, and Phase 11, which starts early only because data acquisition is slow.

---

## 1. Estimands, stated precisely

Data $\{(Y_i, A_i, X_i)\}_{i=1}^n$, $Y_i$ a point cloud, $A_i\in\{0,1\}$ the group label, $X_i$ covariates.
Fix a filtration $\mathcal{F}$, degree $d$, representation $\Phi$ (default: power-weighted silhouette).
Write $D_{i,d}=\mathrm{Dgm}_d(\mathcal{F}(Y_i))$ and $Z_i=\Phi(D_{i,d})$.

| Label | Null | Reads as |
|---|---|---|
| $H_0^{\mathrm{cond}}$ | $\mathcal{L}(D\mid A=1)=\mathcal{L}(D\mid A=0)$ | *Everything the field currently tests.* Confounded by $X$. |
| $H_0^{\mathrm{out}}$ | $\psi_d(t):=\mathbb{E}[\Phi(D^1)(t)]-\mathbb{E}[\Phi(D^0)(t)]=0\ \forall t$ | Outcome-level TATE. Kim & Lee. |
| $H_0^{\mathrm{dist}}$ | $\delta_{\mathrm{dist}}:=d(T_{\mathrm{dist}}(P^1_Y),T_{\mathrm{dist}}(P^0_Y))=0$ | Distribution-level. **New test (C2).** |
| $H_0^{\mathrm{ctate}}$ | $\tau_d(t,x):=\mathbb{E}[\delta_{i,d}(t)\mid X=x]=0\ \forall t,x$ | Conditional. **New (C3).** |
| $H_0^{\mathrm{het}}$ | $\tau_d(t,x)$ does not depend on $x$ | Heterogeneity omnibus. **New (C3).** |
| $H_0^{\mathrm{equiv}}$ | $\sup_t\|\psi_d(t)\|\ge\delta$ | Equivalence variant, for certifying *sameness*. |

**Two facts that must appear in the manuscript, unhedged:**

- $H_0^{\mathrm{dist}}\Rightarrow H_0^{\mathrm{out}}$, and the converse is false. Two groups can share a
  mean silhouette while differing in variance, multimodality, or rare persistent features.
- $H_0^{\mathrm{out}}$ and $H_0^{\mathrm{dist}}$ can disagree because persistent homology does not commute
  with mixing over covariates. Souto & Diamantis prove outcome-level and distribution-level contrasts agree
  for all laws iff $T_{\mathrm{dist}}-\mathcal{A}$ is constant, so a **non-affine** $T_{\mathrm{dist}}$
  *cannot* universally agree with the outcome-level construction. Cite your own paper here; it is the
  cleanest statement of the point and it is what licenses C2 as a distinct test rather than a variant.

---

## 2. Source map (per CLAUDE.md §4)

Confidence: **[A]** cross-corroborated by ≥2 reviews with consistent arXiv ID; **[B]** single-source, verify;
**[C]** title/attribution uncertain, must be located.

**Foundations to build on**
- Kim & Lee, *Topological Causal Effects*, arXiv:2603.02289, ICLR 2026 **[A]** — TATE, DR estimator,
  functional weak convergence, Lemma 2.1 (1-Lipschitz silhouette), Thm 5.3 (silhouette↔$W_1$ stability).
  PDF is already local at `../CP_TATE/2603.02289v1.pdf`; code at `../CP_TATE/top-causal-effect-main/`.
- Souto & Diamantis, *A Mathematical Framework for TCDA*, arXiv:2607.28161 **[A]** — four-layer
  architecture, outcome- vs distribution-level, $g$-formula identification, DR representations for
  Banach-valued summaries, stability-transfer bounds, affine-law functional. **Your own paper; C2's
  estimand comes from here.**
- Chazal, Fasy, Lecci, Rinaldo, Wasserman (SoCG 2014) — weak convergence of average landscapes/silhouettes,
  bootstrap consistency. **[A]**
- Bubenik (JMLR 2015) — landscapes in Banach space, SLLN + CLT. **[A]**
- Petersen & Müller (AoS 2019) — Fréchet regression. **[A]**
- Kennedy (2023) — DR-Learner, for C3's pseudo-outcome construction. **[A]**

**Competitors to benchmark (mandatory, all of them)**
- Robinson & Turner, arXiv:1310.7467, JACT 2017 **[A]** — the canonical permutation test.
- Kwitt, Huber, Niethammer, Lin, Bauer (NeurIPS 2015) **[A]** — kernel MMD on diagrams.
- Han, Kim & Kim, arXiv:2607.20893 **[A]** — weighted persistence intensity functions, minimax optimal.
  **The strongest competitor; if you beat nothing else, you must engage this.**
- Murris, Stolz & Borgwardt, arXiv:2606.11911 (STRAND) **[A]** — survival framing, calibrated type I,
  high power from few diagrams.
- Moon & Lazar, arXiv:2006.05466, JRSS-C 2023 **[A]** — vectorized diagrams, two-stage + FDR.
- Dubey & Müller, *Biometrika* 2019 **[A]** — Fréchet ANOVA. Applies to $(\mathcal{D}_p,W_p)$ since it is
  complete and separable (Mileyko et al. 2011). **Caveat to state: its CLT assumes curvature conditions
  diagram space violates (Che et al. arXiv:2109.14697) — this is itself a remark worth making.**
- Krebs & Rademacher, arXiv:2401.10349 **[A]** — relevant-difference tests; the natural comparator for
  $H_0^{\mathrm{equiv}}$.
- Cericola et al., *Involve* 2018 **[A]** — $k$-sample extension of RT.
- Berry, Chen, Cisewski-Kehe, Fasy, JACT 2020 **[A]** — functional summaries, explicit two-sample tests.

**Geometry hazards to cite as scope limits**
- Mileyko, Mukherjee, Harer (2011); Turner et al. (2014) — Fréchet means non-unique. **[A]**
- Che, Galaz-García, Guijarro, Membrillo Solis, arXiv:2109.14697 — $(\mathcal{D}_2,W_2)$ is nonnegatively
  curved but infinite-dimensional in every standard sense. **[A]**
- Roycraft, Krebs, Polonik, AoS 2023 — **naïve bootstrap fails** for persistent Betti numbers; use the
  smoothed bootstrap. **[A]** Load-bearing if any Betti/Euler statistic enters.
- Divol & Lacombe, JACT 2021; Divol & Chazal, JoCG 2021 — persistence measures via optimal partial
  transport. **[A]** The cleanest home for $H_0^{\mathrm{dist}}$; consider it as the default representation
  for C2 instead of trying to do Fréchet means in $(\mathcal{D}_p,W_p)$.

**To locate and verify in Phase 0**
- Saki & Faghihi, arXiv:2603.14169 **[B]** — persistent-homology ignorability; the claim that a *marginal*
  PD effect is **not identified** from conditional topological ignorability alone. **If true this is
  load-bearing for §1 and constrains C2's identification. Verify first.**
- Faghihi, arXiv:2606.01184 **[B]** — topological ignorability = weak ignorability when the summary is
  injective.
- Vejdemo-Johansson & Mukherjee, arXiv:1812.06491 **[B]** — multiplicity across barcode comparisons.
- Kumar & Dhar, arXiv:2211.13959 **[B]** — Betti-number homological equivalence test.
- Nakayama, arXiv:2511.00938 **[C]**, Islambekov et al. (FoDS 2023) **[B]**.

---

## 3. Phases

Legend: **[SEQ]** must run in order · **[PAR-k]** all tasks tagged `PAR-k` can run as independent agents
in parallel · **[GATE]** produces a go/no-go decision.

### Phase 0 — Shared infrastructure and scope lock **[SEQ]** *(shared with P2)*

*Goal:* one repo that both projects build on, and a verified bibliography.

| # | Task | Deliverable | Acceptance check |
|---|---|---|---|
| 0.1 | Repo scaffold `tda2s/`, `uv` env on **Python 3.10** (matches `tcda_uq`, so the two install side by side), GUDHI + Ripser + Cubical Ripser + Alpha, pinned. | `pyproject.toml`, `environment.yaml` | `pytest` green on a smoke test computing $H_0,H_1$ on a noisy circle; `uv pip install "tcda_uq @ git+https://github.com/hugogobato/tcda_uq.git"` succeeds in the same env |
| 0.2 | PH pipeline: cloud → diagram, for VR / Alpha / Čech / cubical-sublevel / DTM. Uniform API. | `tda2s/ph/` | Diagrams for a torus match published Betti numbers $(1,2,1)$ |
| 0.3 | Vectorisation stack: power-weighted silhouette (Kim–Lee parameterisation, weight $w_p=(b_p-a_p)^r$), landscapes, Betti curves, Euler curves, persistence images, persistence measures (Divol–Lacombe). | `tda2s/vec/` | Lipschitz property of the silhouette (Kim–Lee Lemma 2.1) verified numerically |
| 0.4 | Resampling engine: label permutation, multiplier bootstrap, paired bootstrap, **smoothed** bootstrap (Roycraft et al.), cross-fitting folds. | `tda2s/resample/` | Multiplier bootstrap recovers nominal coverage on a Gaussian-process toy |
| 0.5 | **Competitor wrappers**, all of §2: RT (or `inphr` via `rpy2`), MMD, Han et al., STRAND, Moon–Lazar, Fréchet ANOVA, Krebs–Rademacher. | `tda2s/benchmarks/` | Each reproduces a published figure or table from its source paper |
| 0.6 | **Controlled DGP harness.** Point-cloud generators with independently dialable knobs: number of loops/voids, feature persistence, noise, outlier fraction, $n$ per cloud, number of clouds, **covariate $X$, propensity $\pi(X)$, covariate-driven topology**. | `tda2s/dgp/` | Oracle diagrams recoverable; knobs verified independent |
| 0.7 | **Citation verification sweep.** Resolve every **[B]**/**[C]** reference in §2. Priority: Saki & Faghihi 2603.14169 (identification claim), Faghihi 2606.01184, Han et al. 2607.20893. | `Literature_Review/VERIFIED.bib` with DOI/URL per CLAUDE.md §3 | Zero unresolved load-bearing refs; flag missing URLs as `.bib` comments |
| 0.8 | **Reuse audit of `tcda_uq`** (the released `CP_TATE` library, `github.com/hugogobato/tcda_uq`, MIT). Map exactly what P1 imports instead of rebuilding: `tcda_uq.estimators` (plug-in / IPW / **AIPW**, `cross_fit`, functional nuisance regression, **CTATE DR-learner**), `tcda_uq.silhouette` (diagram / cloud / image → power-weighted silhouette), `tcda_uq.datasets.TriOracleSimulation` (an oracle DGP with known TATE, CTATE and ITTE — use it to validate P1's own harness in 0.6), and the multiplier draws inside `tcda_uq.uq.asymptotic`. | `docs/reuse_from_tcda_uq.md` + a thin `tda2s/adapters/tcda_uq.py` shim | P1 contains **zero** reimplemented AIPW / cross-fitting / DR-learner code; a test shows `tda2s` silhouettes match `tcda_uq.silhouette` on a fixed diagram |
| 0.9 | **Forward-compatibility freeze for the deferred ecological track (1 day; no ecology work yet).** Confirm the 0.2 / 0.3 / 0.6 APIs will accept, without rework: (i) clouds of wildly unequal cardinality, $10$ to $10^5$ points; (ii) ambient dimension up to $\sim20$; (iii) an **externally supplied** standardisation (mean/scale vector) rather than per-cloud scaling; (iv) DTM-Rips. Write the four tests now, let them fail, make them pass. Build nothing else. | `tests/test_forward_compat.py` | Four passing tests; **no** ecological data, loaders or dependencies in the repo |

**Exit:** end-to-end run — generate 100 clouds, compute diagrams, vectorise, run all competitors, get
p-values — in under 10 min on 20 threads.

**Parallelisation:** 0.1 first **[SEQ]**. Then **[PAR-0]** = {0.2+0.3 (one agent, they are coupled), 0.4,
0.5, 0.6, 0.7, 0.8+0.9 (one agent; 0.9 is a rider on the API knowledge from 0.8)} — six agents. 0.5 is the
longest pole; start it first.

**Boundary note.** `tcda_uq` quantifies *uncertainty* (confidence bands for TATE/CTATE, conformal
prediction bands for ITTE). P1 does *hypothesis testing*. The shared objects are narrow and should be
named as such in the manuscript: P1's statistic is $T_n=\sqrt n\,\|\hat\psi_d\|_\infty$ computed from
`cross_fit(...).aipw[d]`, calibrated by a multiplier bootstrap over `cross_fit(...).influence()[d]`.
Everything downstream of that (band construction, Liebl–Reimherr, Pini–Vantini, conformal) belongs to
`CP_TATE` and P1 must not claim it.

---

### Phase 1 — Identification and estimand formalisation **[SEQ, after 0]**

*Goal:* say exactly what is being tested and under what assumptions, before writing an estimator.

| # | Task | Notes |
|---|---|---|
| 1.1 | Write the six nulls of §1 with full measure-theoretic care; prove $H_0^{\mathrm{dist}}\Rightarrow H_0^{\mathrm{out}}$ and give an explicit counterexample to the converse. | Counterexample: one persistent cluster splitting into two with mean preserved. Numerically instantiate it. |
| 1.2 | $g$-formula identification of $\psi_d$ and of $\delta_{\mathrm{dist}}$ under consistency + conditional exchangeability + positivity. Reuse Souto–Diamantis §identification. | Distribution-level identification is the harder one; PH does not commute with mixtures. |
| 1.3 | **Resolve the marginal-identification question.** Is the marginal PD effect identified from conditional topological ignorability alone? Verify Saki & Faghihi's claimed negative result. If it holds, state the paper's target as the **covariate-standardized** topological effect and say so plainly; do not sell a marginal null you cannot identify. | **This determines the paper's headline estimand. Do not skip.** |
| 1.4 | Injectivity remark: for injective $\Phi$, topological ignorability $=$ weak ignorability (Faghihi); for non-injective $\Phi$ only the structural feature is identified. State which regime each of the six nulls lives in. | |
| 1.5 | Scope-limit section: diagram-space geometry (non-unique Fréchet means, Che et al. curvature), why the paper works in Banach/Hilbert summary space or in persistence-measure space and **not** in $(\mathcal{D}_p,W_p)$ directly. | Pre-empts the obvious referee objection. |

**Deliverable:** `theory/WP1_estimands_identification.md` — a self-contained section draft.
**Exit:** every null has a stated identification condition and a stated failure mode. §1.3 answered.

**Parallelisation:** 1.1 and 1.5 are **[PAR-1]** (two agents). 1.2→1.3→1.4 is a chain **[SEQ]** and 1.3
gates Phases 3–5, so put your strongest agent on it.

---

### Phase 2 — **[GATE]** The covariate-shift failure of existing tests (C1)

*Goal:* establish, theoretically and empirically, that the field's current null is the wrong target. This
is the paper's reason to exist. **If this phase fails, the causal wrapper is dropped.**

| # | Task | Notes |
|---|---|---|
| 2.1 | **Theorem (false positive).** Construct a DGP class where $\psi_d\equiv0$ and $\delta_{\mathrm{dist}}=0$ (no topological effect) but $\mathcal{L}(D\mid A=1)\ne\mathcal{L}(D\mid A=0)$ because $X\not\perp A$. Prove RT / MMD / Han reject with probability $\to1$. | Mechanism: $X$ drives both group assignment and the number of loops; conditionally on $X$ the groups are identical. |
| 2.2 | **Theorem (false negative / masking).** DGP where $\psi_d\not\equiv0$ but the covariate mixture cancels it marginally, so all existing tests have power $\to\alpha$. | Simpson-type cancellation. This is the more striking of the two; lead with it. |
| 2.3 | Empirical size/power curves for RT, MMD, Han, STRAND, Moon–Lazar, Fréchet ANOVA across an **imbalance sweep** $\lambda\in[0,1]$ interpolating from randomised to strongly confounded. | Single figure: type-I error of every competitor climbing away from $\alpha$ as $\lambda$ grows. This is the paper's Figure 1. |
| 2.4 | Prototype DR test (rough, uncalibrated) on the same sweep; show flat size. | Enough to fire the gate; polish in Phase 3. |
| 2.5 | **[GATE] decision.** | See below. |

**Gate criterion.** At the strongest imbalance setting, with $n=200$ clouds per group and $\alpha=0.05$:
existing tests must show empirical type-I error $\ge 0.20$ (2.1) **or** power $\le 0.10$ against a
$\psi_d$ that the DR test detects with power $\ge 0.70$ (2.2), while the prototype DR test holds size in
$[0.03,0.08]$.

- **Pass** → C1 is the spine; proceed to Phases 3–7 as written.
- **Fail** → drop C1, rewrite the abstract around C2+C3 ("two tests TDA does not have"), keep Phase 4 and
  Phase 5, compress Phase 3 to a reproduction, and downgrade the venue target.

**Deliverable:** `theory/WP2_covariate_shift.md` + `experiments/phase2_imbalance_sweep.py` + Figure 1.

**Parallelisation:** **[PAR-2]** = {2.1, 2.2} theory agents; {2.3} a compute agent (long-running, launch
first, this is the 20-thread job); {2.4} an implementation agent. 2.5 is **[SEQ]** and yours to call.

---

### Phase 3 — The outcome-level DR test, properly **[PAR-A]**

*Goal:* a calibrated, cross-fitted test of $H_0^{\mathrm{out}}$ under covariates. Mostly adaptation, not
invention — say so in the paper.

| # | Task | Notes |
|---|---|---|
| 3.1 | AIPW estimator of $\psi_d(t)$ with cross-fitting: call `tcda_uq.estimators.cross_fit(sample, tseq, n_basis=...)` and read `.aipw[d]` and `.influence()[d]`. **Do not rebuild it.** | $\hat\psi_d(t)=\frac1n\sum_i[\hat\mu_1-\hat\mu_0+(\frac{A_i}{\hat e}-\frac{1-A_i}{1-\hat e})(Z_i-\hat\mu_{A_i})]$. Validate the shim against `TriOracleSimulation`, whose true TATE is known. |
| 3.2 | Test statistic $T_n=\sqrt n\|\hat\psi_d\|_\infty$; null via multiplier bootstrap **and** via a covariate-preserving permutation scheme (permute within propensity strata). | Reuse `tcda_uq`'s multiplier *draws* over the influence function, but build the null distribution of $T_n$ here — that is a test, not a band. The stratified-permutation variant is a small but real contribution: it gives a finite-sample-exact option under a stronger assumption. |
| 3.3 | Nuisance estimation: propensity $\hat e(X)$ and outcome regressions $\hat\mu_a(t,X)$ (function-on-scalar). Sweep learners: parametric, random forest, gradient boosting, neural. | Tests the "adaptive nuisance" regime where Donsker conditions fail. |
| 3.4 | **Double-robustness stress test:** misspecify $\hat e$ only, $\hat\mu$ only, both. Confirm size held in the first two, lost in the third. | The empirical proof of the DR claim. Must be in the paper. |
| 3.5 | Multiplicity across degrees $d\in\{0,1,2\}$: Bonferroni, a sharper max-statistic over $d$, and Vejdemo-Johansson–Mukherjee. | |
| 3.6 | Positivity diagnostics: overlap plots, effective sample size, behaviour as $\hat e\to0$. | Reuse `CP_TATE` Phase 4.5 diagnostics and `tcda_uq`'s positivity-stabilised weighting (`make_weight_fn("overlap")`). |
| 3.7 | Equivalence variant $H_0^{\mathrm{equiv}}$: TOST on $\sup_t|\psi_d(t)|$ with margin $\delta$, multiplicity per Lauzon–Caffo. | Light touch; the serious equivalence work is P2. |

**Deliverable:** `tda2s/tests/dr_outcome.py`; size/power tables over $n\in\{50,100,200,500\}$, ≥500 reps.
**Exit:** size in $[0.03,0.08]$ across the imbalance sweep; DR behaviour confirmed in 3.4.

---

### Phase 4 — The distribution-level test (C2) **[PAR-A]**

*Goal:* the first test of $\delta_{\mathrm{dist}}=0$. This is new and it is the most technically
interesting piece.

| # | Task | Notes |
|---|---|---|
| 4.1 | Choose $T_{\mathrm{dist}}$. Candidates: (i) **expected persistence measure** (Divol–Chazal) — recommended, it is linear-friendly and dodges Fréchet-mean pathology; (ii) kernel mean embedding of the diagram distribution; (iii) expected persistence intensity function (connects to Han et al.). Justify against the affine-law obstruction in Souto–Diamantis. | **Recommendation: start with (i).** It has an existing LLN and a kernel-based estimator. |
| 4.2 | Weighted / standardized estimator of $T_{\mathrm{dist}}(P^a_Y)$ under confounding: reweight by $\hat e(X)$ to form the interventional law's topological representation, then contrast. Derive the influence function if one exists; otherwise state the estimator as plug-in with a stability-transfer error bound (Souto–Diamantis). | The honest fallback (plug-in + stability transfer) is fine and is what your own framework licenses. |
| 4.3 | Test statistic $\hat\delta_{\mathrm{dist}}$ and its null. Weighted permutation, or bootstrap with the smoothed correction if any Betti/Euler statistic enters. | Roycraft et al.: naïve bootstrap is **inconsistent** here. Do not skip the smoothing. |
| 4.4 | **Separation experiment.** DGP where $H_0^{\mathrm{out}}$ holds exactly but $H_0^{\mathrm{dist}}$ fails (cluster splitting with preserved mean silhouette). Show Phase 3's test has power $=\alpha$ and Phase 4's has power $\to1$. | This single figure justifies C2's existence. High priority. |
| 4.5 | Reverse experiment: where the outcome-level test is more powerful. Report both honestly. | |

**Deliverable:** `tda2s/tests/dist_level.py` + the separation figure.
**Exit:** the separation in 4.4 demonstrated at $n=200$ with power $\ge0.8$ vs $\le0.07$.
**Risk:** if the influence function in 4.2 does not exist in closed form, this phase downgrades from
"efficient" to "consistent plug-in". That is acceptable; state it.

---

### Phase 5 — Conditional and heterogeneity testing (C3) **[PAR-A]**

*Goal:* the tests you specifically want. Be realistic about what is provable.

| # | Task | Notes |
|---|---|---|
| 5.1 | Functional DR-learner for $\tau_d(t,x)$ (TATE EIF as pseudo-outcome, regressed on $X$). **Import the fitted learner from `tcda_uq.estimators` (`CP_TATE` Phase 3).** | Not a novelty claim; it is infrastructure. `tcda_uq` gives you $\hat\tau_d(t,x)$ and a *band* for it; P1 adds the *tests* in 5.2–5.4. |
| 5.2 | **Discrete-subgroup conditional test.** Pre-specified subgroups $\{S_j\}$: test $H_0: \psi_d^{(j)}=0$ per subgroup with FWER/FDR control. Cheap, defensible, immediately usable. | Ship this first; it is the version practitioners will actually use. |
| 5.3 | **Heterogeneity omnibus** $H_0^{\mathrm{het}}$: $\tau_d(t,x)$ constant in $x$. Statistic: a projection / quadratic-form of the DR-learner residual on a basis in $x$, calibrated by multiplier bootstrap. | This is the genuinely new test and it is *easier* than uniform $(t,x)$ inference because it is a null of no dependence. **Recommended headline for C3.** |
| 5.4 | **Best-linear-projection variant:** project $\tau_d(t,\cdot)$ on a low-dimensional $x$-basis and test coefficient significance (Chernozhukov-style BLP). Gives an interpretable "which covariate drives the topological difference" readout. | High practical value, low theoretical risk. |
| 5.5 | **Explicitly deferred:** honest simultaneous bands over $(t,x)$ at nonparametric rate. State as open; do not attempt. | This is the same problem `CP_TATE` Phase 3 deferred. Deferring it twice is consistent, not evasive — cite the deferral. |
| 5.6 | Validation: power of 5.3 against dialable heterogeneity strength; size under $\tau_d$ constant. | |

**Deliverable:** `tda2s/tests/{subgroup,heterogeneity,blp}.py`; heterogeneity power curves.
**Exit:** 5.3 holds size and has power $\ge0.8$ against moderate heterogeneity at $n=300$.

---

### Phase 6 — Local power theory (C4) **[SEQ, after 3]**

*Goal:* replace "ours is better empirically" with "ours is better, here is the rate".

| # | Task | Notes |
|---|---|---|
| 6.1 | Local alternatives $\psi_d^{(n)}=n^{-1/2}g$ for fixed direction $g$; derive the limiting distribution of $T_n$ under the sequence and hence the local power function. | Standard route via the functional CLT + Le Cam. Kim & Lee give consistency against *fixed* alternatives only, so this is new. |
| 6.2 | Asymptotic relative efficiency of the DR test vs the permutation test, as a function of the covariate-imbalance parameter $\lambda$. Show ARE $=1$ at $\lambda=0$ (the honest relabelling statement) and $>1$ for $\lambda>0$. | **This is the theorem that makes the paper's claim precise instead of rhetorical.** |
| 6.3 | Semiparametric efficiency bound for $\psi_d$ in the nonparametric model; confirm the AIPW estimator attains it. | May already be in Kim & Lee; check first, and cite rather than re-derive if so. |
| 6.4 | Compare against Han et al.'s minimax rate for the intensity-function test: same regime? different regime? Be explicit about which alternatives each is optimal against. | Do not claim to beat a minimax-optimal test on its own turf. |
| 6.5 | **Numerical falsification** (CLAUDE.md §5): simulate the local-alternative sequence and check the empirical power curve matches the theoretical one before finalising the proof. | Run this *before* writing the proof, not after. |

**Deliverable:** `theory/WP6_local_power.md`.
**Exit:** 6.2's ARE curve derived and numerically confirmed to within Monte Carlo error.

---

### Phase 7 — The bake-off **[SEQ, after 3,4,5]**

*Goal:* one table a referee cannot argue with.

| # | Task | Notes |
|---|---|---|
| 7.1 | Master grid: {8 tests} × {randomised, mild imbalance, strong imbalance} × {mean-shift, variance-shift, cluster-splitting, rare-feature alternatives} × {$n=50,100,200,500$} × 500 reps. Report size and power. | The full grid is expensive; budget it (see §5). |
| 7.2 | Regime analysis: which test wins where. Include the cases where **we lose** (small $n$ randomised: RT and STRAND likely win on simplicity; against Han's target alternatives: Han likely wins). | Reporting losses is what makes the wins credible. |
| 7.3 | Computational cost comparison at matched power. | |
| 7.4 | Robustness: outliers, unequal cloud sizes, heavy-tailed diagram cardinality (Han et al.'s variance-bound issue), model misspecification. | |
| 7.5 | Decision table: "use test X when …". | The most-cited table in the paper, if you write it well. |

**Deliverable:** master tables + the decision table.
**Parallelisation:** the grid shards perfectly by DGP family. **[PAR-7]** = one agent per alternative type
(4 agents), each writing to a shared results store. Cap at 16 workers (see §5).

---

### Phase 8 — The single-cloud regime **[PAR-B, optional, runs anytime after Phase 0]**

*Goal:* handle the case your directory name implies (two point clouds, not two *samples* of clouds),
honestly.

| # | Task | Notes |
|---|---|---|
| 8.1 | State the obstruction: one cloud yields one diagram; there is no replication, so any test imposes a resampling model. | Say this in the abstract if this regime is in the paper. |
| 8.2 | Blumberg et al. (FoCM 2014) subsampling: fixed-size sub-clouds → an empirical distribution of diagrams per cloud. Apply Phases 3–5 to these. | The most principled and most cited route. |
| 8.3 | **Be explicit about the changed null:** this tests the metric-measure-space null (are the two mm-spaces topologically equal), *conditional on the observed clouds*, not a population null over clouds. | Pseudo-replication is the trap; Perplexity's review flags it correctly. |
| 8.4 | Validate: does the subsampling test hold size when the two clouds are drawn from the same mm-space? Does it inflate when sub-clouds overlap? Sweep the overlap fraction. | Expect inflation; quantify it and give a sub-sample-size rule. |
| 8.5 | Compare against Fasy et al. confidence sets + overlap, and Glenn et al. single-image confidence regions. | |

**Deliverable:** `tda2s/tests/single_cloud.py` + a scope-limits subsection.
**Decision:** if 8.4 shows uncontrollable inflation, this becomes a two-paragraph "why we do not do this"
remark rather than a section. That is a fine outcome.

---

### Phase 9 — Real data **[SEQ, after 7]**

| # | Task | Notes |
|---|---|---|
| 9.1 | A dataset with genuine covariate imbalance and a topological question. Candidates: neuroimaging cohorts (the RT paper's ADHD fMRI setting, with demographics as $X$), single-cell / spatial transcriptomics, materials point clouds, or `tcda_uq`'s ORBIT / SARS-CoV-2 CT loaders (`[data]` extra) as a fast start. **Pick one where imbalance is real** or C1 has no application. | Choose in Phase 0 so data access does not block Phase 9. **This is *not* the ecological dataset**: niche hypervolumes are Phases 11–12 and a separate manuscript. Do not merge them; the methods paper needs a clean, already-curated example, not a multi-week data build. |
| 9.2 | Full analysis: unadjusted vs adjusted, all three nulls, subgroup and heterogeneity readouts. | The money figure: unadjusted rejects, adjusted does not (or vice versa). |
| 9.3 | Sensitivity to unmeasured confounding for the topological effect. | Connects to `../Sensitivity_Idea` if that exists. |
| 9.4 | Pre-registration of the analysis plan before unblinding. | Cheap credibility. |

---

### Phase 10 — Manuscript and software **[SEQ, last for the methods paper]**

10.1 Assemble theory (Phases 1, 2, 6) · 10.2 Related work: TDA testing × causal inference × functional data
· 10.3 Can-claim / cannot-claim table (the honest §0 stack) · 10.4 `tda2s` release: API, tests, docs,
tutorials · 10.5 Pre-submission novelty search · 10.6 Target: **JMLR** or **NeurIPS**; fall back to
*Journal of Applied and Computational Topology* if the gate failed.

**The methods paper is done here.** Phases 11 and 12 are a second, applied manuscript.

---

### Phase 11 — Ecological data track: niche hypervolumes **[PAR-C, long lead — start during Phase 7, finish whenever]**

*Goal:* reach an analysis-ready dataset and a pre-registered analysis plan for the collaboration with
Morimoto, Jānis and Joseph, **while running no inference and letting nothing here reshape the methods
paper.** This track is separated out because occurrence-data acquisition and cleaning take weeks of
wall-clock time and are completely independent of Phases 3–10, so it should run in the background rather
than block anything.

*Context, as stated by the collaborators (2026-08-11).* A niche hypervolume is built as: gather all
observation points of a species → extract environmental covariates at each location (WorldClim, elevation,
…) → PCA → use PC1, PC2 and sometimes PC3 as the coordinate system → the resulting point cloud is the
hypervolume. **Take the PCA step at face value for now**, per Morimoto's explicit instruction; the
dimensionality-reduction question is quarantined into task 11.6 and P2 Phase 2D and does not gate anything
here.

| # | Task | Deliverable | Notes |
|---|---|---|---|
| 11.1 | **[GATE] Design lock.** Agree the contrast with the co-authors *before any data pull*. See the table below. | One-page design memo, countersigned | The whole applicability of Phases 3–5 turns on this. |
| 11.2 | Data acquisition: GBIF occurrences for the agreed taxa; WorldClim v2 or CHELSA bioclim (19 variables) plus elevation, sampled at each occurrence. | `data/raw/` + a provenance manifest (version, download date, query) | Long lead. Launch as soon as 11.1 passes. |
| 11.3 | Cleaning: coordinate deduplication, country-centroid and institution-coordinate removal, spatial thinning, temporal window. **Log every filter, because thinning changes cloud cardinality and cardinality changes persistent homology.** | `data/clean/` + a filter-decision log | The filter choices are themselves a sensitivity axis in 12.2. |
| 11.4 | **Confounder construction. This is why the application exists.** At minimum: record count $n_i$; a sampling-effort proxy (target-group background density at the range); geographic range extent; latitude of range centroid; year span of records; taxonomic family or phylogenetic eigenvectors; available trait covariates. | `data/covariates.parquet` + a covariate dictionary | Sampling effort is the archetypal confounder: it is correlated with essentially any treatment you can define, and it mechanically alters the diagram. |
| 11.5 | **Standardisation protocol.** $z$-score each environmental variable against a **fixed background** (the study region's environmental grid), never per-species. | `docs/standardisation.md` + the frozen mean/scale vector | Per-species scaling makes the metric itself group-dependent, which confounds the contrast at the level of the distance matrix. This is what task 0.9(iii) was reserving. |
| 11.6 | Representation ladder, recorded but not yet analysed: (a) full scaled environmental space, $d\approx20$, and (b) PC1–PC3. Store PCA loadings, eigenvalues, and **per-point residual norms off the retained subspace**. | `data/representations/` | The residual norms are the input to P2 Phase 2D's truncation certificate. Compute and store them; do not interpret them yet. |
| 11.7 | Pre-registration of the analysis plan, written against the frozen Phase 3–5 estimators. | `preregistration.md`, timestamped | Cheap credibility, and it stops the applied analysis from drifting into estimand-shopping. |

**11.1 design options.**

| Contrast | Unit / design | Routes to |
|---|---|---|
| Invasive vs native congeners, $\ge30$ species per arm | species-level replication, species covariates | **Phases 3–5. The intended target.** |
| Occupied vs available environment (presence / target-group background) | control = background clouds | Good fit; the propensity has a direct ecological reading as sampling-bias correction |
| Historical vs recent records, same species | **paired** within-species | P2's design, not P1's |
| Species A vs species B | $n=1$ vs $n=1$ | **Phase 8** single-cloud regime; the causal apparatus is decoration here |

**Gate criterion (11.1).** The agreed design must yield $\ge30$ clouds per arm with measurable covariates.
Below that, cross-fitting and nuisance estimation are not worth their variance.
- **Pass** → proceed to 11.2 and, later, Phase 12 as written.
- **Fail (two-species comparison)** → the analysis routes to Phase 8, not Phases 3–5. Say so to the
  co-authors immediately, because it changes what the collaboration can claim: a two-species contrast tests
  a metric-measure-space null conditional on the observed clouds, not a population null over species, and
  permuting occurrence points between two species is pseudo-replication given their spatial
  autocorrelation.

**Exit:** a frozen, documented, pre-registered dataset. Zero inference run.
**Parallelisation:** 11.1 **[SEQ]** and yours to negotiate. Then **[PAR-C]** = {11.2→11.3 (one agent,
chained), 11.4, 11.5, 11.6} once 11.1 passes. 11.7 last.
**Risk:** the co-authors may want a two-species comparison, which is the fail branch. Raise it in the
first conversation rather than after the data build.

---

### Phase 12 — Ecological analysis and companion paper **[SEQ, after 10 and 11]**

*Goal:* the applied paper. Framing is ecological; the method is cited from P1, not re-derived.

| # | Task | Notes |
|---|---|---|
| 12.1 | Ecological baselines, wrapped like any other competitor: the Warren–Glor–Turelli niche-equivalency permutation test, Broennimann's PCA-env overlap statistics (Schoener's $D$, Hellinger $I$), and Blonder's hypervolume overlap. **All three are [B]: verify in the Phase 0.7 citation sweep.** | These are the incumbents. Warren's test is a permutation of occurrences between species, i.e. exactly the $H_0^{\mathrm{cond}}$ null that Phase 2 shows is the wrong target, and it adjusts for nothing. If C1 holds, this is where it bites in the real world. |
| 12.2 | Run the P1 stack: unadjusted (Warren-style permutation, RT, MMD) versus adjusted (Phase 3 DR test), on the identical data. | **The money figure:** unadjusted rejects and adjusted does not, or the reverse, with sampling effort identified as the driver. Report sensitivity to the 11.3 filter choices. |
| 12.3 | Distribution-level readout (Phase 4) and heterogeneity / BLP readout (Phase 5.4). | The BLP is the ecologically interpretable one: *which* covariate drives the topological difference. Expect this to be the most-read result in the applied paper. |
| 12.4 | **The availability confound.** A hole in an occupied-environment cloud can be inherited from a hole in the *available* environment: planetary climate space is neither convex nor simply connected, so $H_1$ features may be properties of the background rather than of the species. Contrast against the target-group background, or condition on it. | If this cannot be fully resolved, state it as a scope limit in plain language. It is the objection an ecologist referee will raise first, and pre-empting it is worth more than a partial fix. |
| 12.5 | Pre-PCA versus post-PCA robustness: run the full analysis in the scaled $d\approx20$ environmental space (DTM-Rips only; Alpha and cubical are infeasible at that dimension) and in PC1–PC3. Consume the truncation certificate from **P2 Phase 2D**. | If the two agree, the paper certifies standard practice, which is an easy sell. If they disagree, that disagreement *is* the applied paper's headline and speaks directly to what the group says it is working on. Either outcome is publishable. |
| 12.6 | Manuscript. Target *Methods in Ecology and Evolution*, *Ecography*, or *Global Ecology and Biogeography*. | **P1's own abstract does not change because of this phase.** The applied paper cites P1 for the method; the methods paper does not need the ecology to stand up. |

**Deliverable:** the applied manuscript plus a reproducible analysis repo depending on released `tda2s`.
**Exit:** 12.2's comparison produced and interpreted, with 12.4 either resolved or explicitly scoped out.
**Risk:** venue drift. An ecology co-author pulls the framing toward ecology; that is correct for *this*
paper and wrong for P1. Keep them separate documents from day one.

---

## 4. Dependency graph and parallelisation

```
0 (shared infra) ──[GATE-free, but blocks all]
 │
 ├─► 1 (identification) ──┐
 │      ▲                 │
 │      └─ 1.3 gates 3,4,5│
 │                        │
 └─► 2 (covariate shift) ─┤ [GATE]
        │                 │
        ▼                 ▼
     ┌──────────── PAR-A (three independent agents) ────────────┐
     │  3 (DR outcome test)   4 (dist-level test)   5 (TCATE)   │
     └───────┬──────────────────────┬────────────────┬──────────┘
             │                      │                │
             ▼                      │                │
          6 (local power)           │                │
             │                      │                │
             └──────────┬───────────┴────────────────┘
                        ▼
                     7 (bake-off)  ◄── PAR-7: 4 sharded agents
                        │
                        ├──────────────────► 11 (ecological data track) ──┐  [PAR-C, long lead]
                        ▼                          [GATE at 11.1]         │
                     9 (real data)                                        │
                        │                                                 │
                        ▼                                                 │
                    10 (manuscript — METHODS PAPER SUBMITTED)             │
                        │                                                 │
                        └──────────────────┬──────────────────────────────┘
                                           ▼
                                 12 (ecological analysis
                                    + companion paper)

 8 (single-cloud)  ── PAR-B: independent, any time after 0
                        └── also the fallback landing site if gate 11.1 fails
```

**What parallelises well (safe to hand to independent agents):**
- **Phase 0:** six agents. Longest pole is 0.5 (competitor wrappers) — launch first.
- **Phase 1:** 1.1 ∥ 1.5. The 1.2→1.3→1.4 chain is sequential and is the highest-value agent slot.
- **Phase 2:** 2.1 ∥ 2.2 (theory) ∥ 2.3 (compute, long) ∥ 2.4 (implementation).
- **Phases 3 ∥ 4 ∥ 5:** three agents, fully independent. **This is the main parallel window.**
- **Phase 7:** shard by alternative family, 4 agents.
- **Phase 8:** an entire independent track.
- **Phase 11:** an entire independent track, and the only one that is wall-clock- rather than
  compute-bound. Start it during Phase 7 precisely because it will otherwise be the thing that delays the
  companion paper by a month.

**What must not be parallelised:**
- **1.3 → everything.** The identification answer determines what Phases 3–5 are even estimating. Running
  them before 1.3 lands risks three agents building the wrong estimand.
- **The Phase 2 gate.** Do not start Phases 3–5 in earnest before firing it; a failed gate rewrites the
  paper's frame and changes what Phase 3 needs to be.
- **Phase 6 needs Phase 3's estimator fixed.** Local power theory on a moving target is wasted work.
- **Phase 7 needs 3, 4, 5 all frozen.** Re-running the master grid because one test changed is the single
  most expensive mistake available here.
- **Phase 12 must not start before Phase 10.** Not a resource constraint, a discipline constraint: an
  applied analysis running alongside an unfinished methods paper will pull the estimand toward whatever
  the data happens to support. Phase 11 is exempt because it produces no inference.
- **Gate 11.1 before any data pull.** A GBIF and WorldClim build for the wrong design is weeks of wasted
  wall-clock, and the wrong design (two species) is the one the ecology literature defaults to.

---

## 5. Compute budget (per CLAUDE.md §2)

Machine: i9-13900H, 10P+20L cores. Assume other experiments are running: **cap at 16 workers**, leave 4.

| Job | Cost | Strategy |
|---|---|---|
| Phase 2 imbalance sweep | ~6 h | `joblib` over reps, 16 workers |
| Phase 3 size/power tables | ~12 h | shard by $n$; checkpoint per shard |
| **Phase 7 master grid** | **8 tests × 3 regimes × 4 alts × 4 $n$ × 500 reps ≈ 200k test runs** | **Two-stage: 100 reps for a coarse pass, then 500 reps only on cells that matter. Shard by alternative family. Checkpoint to parquet after every cell.** |
| VR on large clouds | memory blow-up | Cap points per cloud at 1–2k for VR; use Alpha above that |

**Memory discipline:** permutation tests recompute diagrams; cache diagrams to disk keyed by
(cloud hash, filtration, params) and permute *labels*, never recompute PH inside the permutation loop.
This is the single biggest speedup available and it applies to every phase.

---

## 6. Cross-cutting risks

1. **Novelty (highest).** Kim & Lee own the outcome-level test. Mitigation: Phase 2's gate, and lead with
   C1/C2/C3 rather than C-nothing. If the gate fails, the paper is smaller — accept that early rather than
   late.
2. **Self-overlap with `CP_TATE`.** Mitigation: the boundary table in `RESEARCH_DECISION.md`; import
   `tcda_uq`, do not reimplement; cite `CP_TATE` explicitly. Concretely: `CP_TATE` owns bands, P1 owns
   p-values, and the only shared objects are the AIPW curve and the influence function (Phase 0.8's
   boundary note).
3. **Identification (1.3).** If the marginal topological null is genuinely unidentified under conditional
   ignorability, the paper's headline becomes the *standardized* effect. Not fatal, but it must be
   discovered in Phase 1, not in review.
4. **Diagram-space geometry.** Never do inference in $(\mathcal{D}_p,W_p)$ directly. Functional summaries
   or persistence measures only. Fréchet means as a robustness check at most.
5. **Bootstrap validity.** Naïve bootstrap is inconsistent for persistent Betti numbers. Use the smoothed
   bootstrap anywhere those enter, and say so.
6. **Multiplicity.** Degrees × scales × subgroups. Control it, and report the uncontrolled version too so
   readers can see the cost.
7. **Data access for Phase 9.** Resolve in Phase 0.
8. **Scope creep from the ecological collaboration.** The application is genuinely well matched to C1, and
   that is exactly what makes it dangerous: it is tempting to promote it into the methods paper. Do not.
   The methods paper's claims must be provable on synthetic DGPs with known ground truth, which niche data
   never has. Phases 11–12 are a separate manuscript with a separate venue, and the only permitted
   backflow into Phases 0–10 is task 0.9's four forward-compatibility tests.

---

## 7. First five concrete actions

1. Run Phase 0.7 (citation verification), especially Saki & Faghihi 2603.14169 — it may reshape §1.
2. Launch Phase 0.5 (competitor wrappers) as a long-running agent; it is the critical path.
3. Read `../CP_TATE/2603.02289v1.pdf` in full, specifically the DR estimator, Lemma 2.1, and Theorem 5.3;
   then install `tcda_uq` and run `scripts/reproduce_coverage.py --quick` to confirm the import surface
   works before Phase 3 depends on it.
4. Build the Phase 0.6 DGP harness with the covariate knob — Phase 2's gate depends entirely on it, and
   validate it against `tcda_uq.datasets.TriOracleSimulation`, whose true TATE/CTATE/ITTE are known.
5. Draft the two witness DGPs for Phase 2.1/2.2 on paper before implementing; if you cannot construct the
   masking example (2.2), the paper's strongest claim is in trouble and you want to know now.

**And one thing not to do yet.** Do not touch GBIF, WorldClim, or any niche data until Phase 7. The single
exception is a conversation with Morimoto, Jānis and Joseph to settle gate 11.1 (which contrast, how many
species per arm), which costs nothing and has weeks of lead time riding on it.
