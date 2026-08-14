# Statistical Two-Sample Testing of Point-Cloud Topology via Persistence Diagrams
## A Literature Review and Research Blueprint, Anchored in Topological Causal Effects

**Research idea under review:** develop a statistical test for whether two point-cloud datasets (or two single observations) are significantly different in topology, given a chosen filtration and persistence-diagram (PD) representation. The framing proposed by the user: encode group membership as a binary covariate A ∈ {0,1} and test whether the topological contrast — analogous to the Topological Average Treatment Effect (TATE) — is zero. A side idea: use the same test to compare filtrations (Vietoris–Rips vs. cubical) and calibrate the largest cube size usable for very large point clouds.

---

## 1. Positioning summary

The proposed test sits at the intersection of two mature but so-far disjoint literatures:

1. **Topological causal inference** — where Kim & Lee's *Topological Causal Effects* (arXiv:2603.02289) and Diamantis & Souto's *A Mathematical Framework for Topological Causal Data Analysis* (arXiv:2607.28161) define treatment effects on persistence-diagram summaries (TATE / TATE_out / Δ_dist / δ_dist).
2. **Topological two-sample testing** — where a rich toolbox already exists: permutation tests on diagrams (Robinson & Turner 2017), functional summaries with bootstrap confidence bands (Bubenik 2015; Chazal et al. 2014; Fasy et al. 2014), kernel/MMD tests (Gretton et al. 2012; Kwitt et al. 2015), and — crucially for the side idea — *equivalence / relevant-difference* tests (Krebs & Rademacher 2024).

**The core thesis of this review:** the user's TATE-based framing is not just a metaphor — when A is an *exogenous* group indicator (i.i.d. samples from two populations), the causal assumptions collapse (exchangeability holds by design), and the Kim–Lee "test of no topological effect" becomes exactly a functional two-sample test on the silhouette contrast. The genuinely novel contributions would be (a) making this reduction explicit and rigorous, and (b) building a *filtration-equivalence test* with a cube-size calibration objective, which does not exist in the literature.

---

## 2. The two anchor papers

### 2.1 Topological Causal Effects — Kim & Lee (arXiv:2603.02289)

**What it does.** Defines causal effects through differences in the topological structure of potential outcomes, summarized by **power-weighted silhouette functions of persistence diagrams**. It develops an efficient, **doubly robust estimator** in a fully nonparametric model, establishes **functional weak convergence**, and constructs a **formal test of the null hypothesis of no topological effect**, with asymptotically valid size and consistency. It also derives new stability bounds for weighted silhouettes under Wasserstein perturbations of diagrams.

**Key objects (from the paper).** For a persistence diagram D with points p = (a_p, b_p):
- Tent function: Λ_p(t) = max{0, min{t − a_p, b_p − t}}, t ∈ T (a fixed compact interval).
- Power-weighted silhouette: φ(t; D) = Σ_p w_p Λ_p(t) / Σ_p w_p, with power weight w_p = (b_p − a_p)^r (r > 0 emphasizes long-lived features).
- Lemma 2.1 (Lipschitz stability): sup_{|s−t|≤δ} |φ(s; D) − φ(t; D)| ≤ δ — the silhouette is 1-Lipschitz, which is what makes functional inference tractable.

**What it lacks for your purpose.** It is built for *observational* causal inference: it requires consistency, conditional exchangeability, positivity, and propensity-score estimation. It does not provide a ready-made two-sample test for arbitrary point-cloud groups, and its test is specific to the silhouette functional rather than a general PD-distance test.

### 2.2 A Mathematical Framework for Topological Causal Data Analysis — Diamantis & Souto (arXiv:2607.28161)

**What it does.** Separates the observation space, causal-model class, topological representation, and causal query into a four-layer architecture. It distinguishes:
- **Outcome-level TCDA:** TATE_out = E[T_out(Y¹)] − E[T_out(Y⁰)] — the contrast of *expected topological summaries of individual outcomes* (e.g., expected silhouette/landscape).
- **Distribution-level TCDA:** Δ_dist = T_dist(P¹_Y) − T_dist(P⁰_Y) applied to *interventional outcome laws*, and the scalar discrepancy δ_dist = d_dist(T_dist(P¹_Y), T_dist(P⁰_Y)) for metric-valued targets.

It establishes g-formula identification, doubly robust representations for Banach-space-valued summaries, stability-transfer bounds for plug-in estimation, and the affine law functional A(P) = ∫ T_out(y) dP(y). A key structural result: **outcome-level and distribution-level contrasts agree for all laws iff T_dist − A is constant on the relevant class; a non-affine T_dist cannot agree universally with the outcome-level construction** — i.e., the order of "topologize then aggregate" vs. "aggregate then topologize" is part of the definition of the target.

**Why it matters for your idea.** The two-sample test you propose has *two possible targets*, and this paper shows they are not the same:
- A **silhouette/landscape contrast** (mean of per-observation diagrams) is *outcome-level* — it detects shifts in the expected topological summary.
- An **MMD / Wasserstein test on the distribution of diagrams** is *distribution-level* — it can detect changes (e.g., one persistent cluster splitting into two) even when the means coincide.

The paper itself provides no concrete test statistic or permutation machinery — it is a framework, which is exactly the gap your proposal fills.

---

## 3. The existing two-sample testing landscape on persistence diagrams

### 3.1 Permutation / randomization tests on diagrams (the workhorse)

| Paper | Method | Notes |
|---|---|---|
| **Robinson & Turner 2017**, *Hypothesis testing for topological data analysis* (arXiv:1310.7467; JACT) | Randomization/permutation NHST on sets of PDs; loss = pairwise distances between elements of each sample and all elements of the other sample | The canonical reference; applied to fMRI (ADHD) data; empirically explores power |
| **Bubenik 2015**, *Statistical TDA using persistence landscapes* (JMLR 16) | Two-sample z-test on landscape norms; permutation test (10,000 reps) | Landscapes are Banach-valued, enabling classical machinery |
| **Berry et al. 2020**, *Functional summaries of persistence diagrams* (JACT) | Permutation tests on functional summaries; bootstrap confidence bands | Compares human vs. monkey data |
| **Kumar & Dhar 2022**, *Testing Homological Equivalence Using Betti Numbers* (arXiv:2211.13959) | Betti-number-based homological equivalence test | Benchmarked against Robinson–Turner and landscape tests |

### 3.2 Functional summaries + bootstrap confidence bands

| Paper | Method |
|---|---|
| **Chazal et al. 2014**, *Stochastic convergence of persistence landscapes and silhouettes* (SoCG) | Bootstrap gives valid confidence bands for the average landscape/silhouette |
| **Fasy et al. 2014**, *Confidence sets for persistence diagrams* (Ann. Statist. 42) | Bootstrap confidence sets for PDs via stability |
| **Chazal et al. 2018**, *Robust topological inference* (JMLR 18) | Distance-to-measure and kernel distance; asymptotically valid bootstrap bands |
| **Moon & Lazar 2023**, *Hypothesis testing for shapes using vectorized persistence diagrams* (JRSS-C 72) | Two-stage test; pooled-variance t-test per pixel of a vectorized diagram |
| **Rieck et al. 2017**, *Topological ML with persistence indicator functions* | Persistence indicator functions + standard two-sample z-test |

### 3.3 Kernel-based tests (MMD)

| Paper | Method |
|---|---|
| **Gretton et al. 2012**, *A kernel two-sample test* (JMLR 13) | Maximum Mean Discrepancy (MMD); asymptotic null distribution; the foundation |
| **Kwitt et al. 2015**, *Statistical TDA — a kernel perspective* (NeurIPS) | Kernel mean embeddings of PDs; two-sample hypothesis testing |
| **Han, Kim & Kim 2026**, *A Two-Sample Test on Weighted Persistence Intensity Functions* (arXiv:2607.20893) | Kernel-based permutation test on weighted persistence intensity functions; MMD-based |
| **Murris, Stolz & Borgwardt 2026**, *From Persistence to Survival* (arXiv:2606.11911) | Non-parametric two-sample test with calibrated Type I error and high power from few diagrams; effect sizes; vectorisation |

### 3.4 Equivalence / relevant-difference testing (critical for the side idea)

| Paper | Method |
|---|---|
| **Krebs & Rademacher 2024**, *Two-sample tests for relevant differences in persistence diagrams* (arXiv:2401.10349) | Tests whether the difference between two diagram populations exceeds a *pre-specified relevant threshold* under Wasserstein metrics — i.e., equivalence-style testing, not exact-equality testing |

This is the single most relevant reference for the cube-size calibration idea: it formalizes "how different is different enough to matter," which is precisely what a filtration-calibration decision needs.

### 3.5 Multiple testing

| Paper | Method |
|---|---|
| **Vejdemo-Johansson & Mukherjee 2018**, *Multiple testing with persistent homology* (arXiv:1812.06491) | Permutation tests + barcode distances for two-sample testing; pairwise testing across diagram groups; multiplicity control |

---

## 4. Gap statement

Three concrete gaps justify the proposed research:

1. **No explicit bridge between TATE and two-sample testing.** The observation that "group indicator = binary treatment" has not been made rigorous in the literature. When A is an exogenous group label, exchangeability holds by design, so the Kim–Lee test of "no topological effect" (H0: TATE = 0) reduces to a *functional two-sample test* on the silhouette contrast. Making this reduction explicit — and deriving its null distribution under permutation vs. the Kim–Lee functional weak-convergence machinery — is a clean, publishable contribution.

2. **Outcome-level vs. distribution-level targets are conflated in practice.** The TCDA framework shows these are different estimands that can disagree. A proposed test should state which target it tests (mean-summary contrast vs. distribution-of-diagrams contrast) and justify the choice — most existing tests are implicitly distribution-level (MMD) or outcome-level (landscape means) without saying so.

3. **No filtration-equivalence test with a resolution-calibration objective.** The stability literature bounds the *error* of grid/cubical approximations (see §6), but no one has built a *hypothesis test* that decides "the largest cube size h such that the cubical PD is statistically indistinguishable from the reference (VR) PD." This is the side idea, and it is genuinely open.

---

## 5. Methodological blueprint for the proposed test

### 5.1 Formal setup
- **Data regime A (datasets):** n₁ point-clouds in group 1, n₀ in group 0; each point-cloud → one PD via the chosen filtration. This is the regime the existing literature handles.
- **Data regime B (single observation per group):** only one PD per group. A distributional two-sample test is *impossible without replication*. Options: (i) subsample points within each cloud, recompute PDs, and treat these as pseudo-replicates — but this tests the *sampling distribution of the PD estimator*, not the population of point-clouds; (ii) fall back to a distance threshold on the two diagrams (no p-value). This distinction must be stated explicitly in the paper.

### 5.2 Null hypotheses
- **Outcome-level (TATE-style):** H0: E[φ(D | A=1)] = E[φ(D | A=0)] for all t ∈ T, i.e., the silhouette contrast function is identically zero.
- **Distribution-level (MMD-style):** H0: P(D | A=1) = P(D | A=0).
- **Equivalence version (for calibration):** H0: d(P₁, P₀) ≤ δ for a pre-specified tolerance δ (Krebs–Rademacher style).

### 5.3 Test statistics
- **TATE-style:** sup-norm (or L²-norm) of the silhouette contrast, with null distribution from (a) permutation of group labels (Robinson–Turner) or (b) functional weak convergence + simultaneous confidence bands (Kim–Lee).
- **MMD on diagrams** (Kwitt et al. 2015) for the distribution-level target.
- **Wasserstein distance between mean diagrams** (Krebs & Rademacher 2024) for the equivalence target.

### 5.4 Permutation and bootstrap scheme
- Permute group labels, recompute the statistic, build the empirical null (Robinson–Turner).
- Bootstrap-resample point-clouds within groups for confidence bands (Fasy/Chazal machinery).
- For the TATE-style functional test, the Kim–Lee weak-convergence result provides an asymptotic alternative to permutation.

### 5.5 Power and multiplicity
- **Power is governed by the number of point-clouds per group (replication), not the number of points per cloud.** This is the dominant practical constraint and should drive the experimental design.
- Multiple homology dimensions (d = 0, 1, 2, …) and filtration scales t induce a multiple-testing problem → use simultaneous bands (Kim–Lee) or explicit multiplicity control (Vejdemo-Johansson & Mukherjee 2018).

---

## 6. Side idea: filtration comparison and cube-size calibration

### 6.1 Why VR vs. cubical, and what "convergence" really means
- **Vietoris–Rips** builds simplicial complexes on the point set; its persistent homology is stable under Gromov–Hausdorff perturbations (Chazal, de Silva & Oudot 2014), making it the theoretical reference for point-cloud topology. Its cost grows combinatorially with the number of points.
- **Cubical complexes** approximate the sublevel sets of a function sampled on a grid and are markedly cheaper for large data (Wagner, Chen & Vuçini 2011).
- **The honest convergence statement:** both are *consistent estimators of the persistent homology of the underlying shape*, but under different approximation schemes. By the stability theorem (Cohen-Steiner, Edelsbrunner & Harer 2005), the bottleneck distance between the cubical PD and the true PD is bounded by the sup-norm error of the grid sampling of the filtration function — O(grid size) for Lipschitz functions. By GH stability (Chazal, de Silva & Oudot 2014), the VR PD converges to the true PD as sampling density grows. Hence the *discrepancy between VR-PD and cubical-PD is bounded by the sum of the two approximation errors*, and shrinks as cube size → 0 and density → ∞. They converge to the *same limit*; they do not formally "converge to each other" except through that shared limit. The correct correspondence is to filter the *distance function to the point set* on the cubical grid (robust to outliers via distance-to-measure; Chazal, Cohen-Steiner & Mérigot 2011).

### 6.2 Calibration experiment design
1. **Reference:** compute VR PDs (or high-resolution cubical PDs) on a validation set of point-clouds.
2. **Candidates:** cubical PDs at cube sizes h₁ < h₂ < … < h_k.
3. **For each h:** run the *equivalence* test (Krebs–Rademacher relevant-difference test, or a TATE-style test with tolerance δ) comparing the cubical PD distribution against the reference.
4. **Selection:** pick the largest h for which the test does not reject (equivalently, estimated discrepancy ≤ δ).
5. **Validation:** confirm the chosen h on a held-out set (avoids overfitting the resolution to the validation data).

### 6.3 Honest caveats
- **"Fails to reject" ≠ "equal."** A plain two-sample test cannot certify equivalence; the calibration must use an equivalence/relevant-difference test with a pre-specified tolerance δ (Krebs & Rademacher 2024). This is the single most important methodological point for the side idea.
- **Tolerance must be scale-aware:** features with persistence below δ are not resolvable at cube size h anyway; δ should be chosen relative to the feature scales of scientific interest.
- **The test's power depends on the number of validation point-clouds**, so the calibration budget (how many clouds, how many cube sizes) is a design parameter.

---

## 7. Open design decisions (for the next step)

1. **Target level:** outcome-level (silhouette/landscape contrast) vs. distribution-level (MMD on diagrams) — the TCDA framework says these can disagree; which is scientifically meaningful for your application?
2. **Data regime:** replicated point-cloud datasets, or single observations (requiring subsampling pseudo-replicates)?
3. **Vectorization choice:** silhouette (TATE-native), landscape, persistence image, or persistence indicator function — each has different stability and power properties.
4. **Testing philosophy:** exact-equality (point null) vs. equivalence (tolerance δ) — the latter is mandatory for the cube-size calibration.

---

## 8. References

1. K. Kim, H. Lee. *Topological Causal Effects.* arXiv:2603.02289. https://arxiv.org/abs/2603.02289
2. I. Diamantis, H. G. Souto. *A Mathematical Framework for Topological Causal Data Analysis.* arXiv:2607.28161. https://arxiv.org/abs/2607.28161
3. A. Robinson, K. Turner. *Hypothesis testing for topological data analysis.* J. Appl. Comput. Topol. 1, 2017. arXiv:1310.7467. https://arxiv.org/abs/1310.7467
4. P. Bubenik. *Statistical topological data analysis using persistence landscapes.* JMLR 16, 2015. https://www.jmlr.org/papers/volume16/bubenik15a/bubenik15a.pdf
5. F. Chazal, B. T. Fasy, F. Lecci, A. Rinaldo, L. Wasserman. *Stochastic convergence of persistence landscapes and silhouettes.* SoCG 2014. https://dl.acm.org/doi/abs/10.1145/2582112.2582128
6. B. T. Fasy, F. Lecci, A. Rinaldo, L. Wasserman, S. Balakrishnan, A. Singh. *Confidence sets for persistence diagrams.* Ann. Statist. 42(6), 2014. https://projecteuclid.org/journals/annals-of-statistics/volume-42/issue-6/Confidence-sets-for-persistence-diagrams/10.1214/14-AOS1252.pdf
7. F. Chazal, B. T. Fasy, F. Lecci, B. Michel, A. Rinaldo, L. Wasserman. *Robust topological inference: distance to a measure and kernel distance.* JMLR 18, 2018. https://www.jmlr.org/papers/volume18/15-484/15-484.pdf
8. E. Berry, Y.-C. Chen, J. Cisewski-Kehe, B. T. Fasy. *Functional summaries of persistence diagrams.* J. Appl. Comput. Topol. 4, 2020. arXiv:1804.01618. https://arxiv.org/pdf/1804.01618
9. C. Moon, N. A. Lazar. *Hypothesis testing for shapes using vectorized persistence diagrams.* JRSS-C 72(3), 2023. arXiv:2006.05466. https://arxiv.org/pdf/2006.05466
10. B. Rieck, F. Sadlo, H. Leitte. *Topological machine learning with persistence indicator functions.* Topological Methods in Data Analysis and Visualization V, 2017. arXiv:1907.13496. https://arxiv.org/pdf/1907.13496
11. A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Schölkopf, A. Smola. *A kernel two-sample test.* JMLR 13, 2012. https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf
12. R. Kwitt, S. Huber, M. Niethammer, W. Lin, U. Bauer. *Statistical topological data analysis — a kernel perspective.* NeurIPS 2015. https://proceedings.neurips.cc/paper_files/paper/2015/file/74563ba21a90da13dacf2a73e3ddefa7-Paper.pdf
13. Y. Han, I. Kim, J. Kim. *A Two-Sample Test on Weighted Persistence Intensity Functions in Topological Data Analysis.* arXiv:2607.20893. https://arxiv.org/abs/2607.20893
14. J. Murris, B. Stolz, K. Borgwardt. *From Persistence to Survival: Hypothesis Testing, Effect Sizes and Vectorisation for Topological Features.* arXiv:2606.11911. https://arxiv.org/abs/2606.11911
15. J. Krebs, D. Rademacher. *Two-sample tests for relevant differences in persistence diagrams.* arXiv:2401.10349. https://arxiv.org/abs/2401.10349
16. M. Vejdemo-Johansson, S. Mukherjee. *Multiple testing with persistent homology.* arXiv:1812.06491. https://arxiv.org/abs/1812.06491
17. S. Kumar, S. S. Dhar. *Testing Homological Equivalence Using Betti Numbers.* arXiv:2211.13959. https://arxiv.org/abs/2211.13959
18. D. Cohen-Steiner, H. Edelsbrunner, J. Harer. *Stability of persistence diagrams.* SoCG 2005. https://dl.acm.org/doi/abs/10.1145/1064092.1064133
19. F. Chazal, V. de Silva, S. Oudot. *Persistence stability for geometric complexes.* Geom. Dedicata 173, 2014. arXiv:1207.3885. https://arxiv.org/pdf/1207.3885
20. F. Chazal, D. Cohen-Steiner, Q. Mérigot. *Geometric inference for probability measures.* Found. Comput. Math. 11, 2011. https://link.springer.com/article/10.1007/s10208-011-9098-0
21. H. Wagner, C. Chen, E. Vuçini. *Efficient computation of persistent homology for cubical data.* Topological Methods in Data Analysis and Visualization II, 2011. https://link.springer.com/chapter/10.1007/978-3-642-23175-9_7
