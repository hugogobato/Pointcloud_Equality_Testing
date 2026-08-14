# A Topological (TATE-Based) Two-Sample and Equivalence Test for Persistence Diagrams

## A Literature Review and Positioning Analysis

---

## 1. Executive Summary

A doubly-robust, CLT-backed test of the "no topological treatment effect" null **already exists** (Kim et al., 2026), so the TATE-null two-sample test should not be reinvented. The genuinely novel and publishable contributions available here are:

1. Reframing a two-sample topological comparison as a TATE null with a synthetic binary treatment, and demonstrating that this is a *re-labelling with a semiparametric-efficiency payoff* — double robustness and cross-fitting — but **only when nuisance covariates are present**.
2. Framing the filtration-selection application as a formal **equivalence (TOST-style) test**, which is essentially absent from the TDA literature.

Absent covariates, the TATE null with a synthetic indicator is mathematically equivalent to a classical two-sample problem already addressed by permutation tests (Robinson & Turner, 2017), functional-summary CLTs (Bubenik, 2015; Chazal et al., 2015), kernel-MMD tests (Kwitt et al., 2015), and — most directly — Fréchet ANOVA on the metric space $(\mathcal{D}_p, W_p)$ (Dubey & Müller, 2019). The paper should say so plainly.

The strongest single contribution is the **filtration-equivalence result**: using an equivalence test on a validation set to certify the coarsest cubical resolution whose diagrams are statistically indistinguishable from Vietoris–Rips, with the equivalence margin anchored to the interleaving/stability bound (Chazal et al., 2014).

---

## 2. Primary arXiv Sources

### 2.1 "Topological Causal Effects" (Kim et al., 2026)

*Verified via abstract; internal equations and assumption wording were not directly inspected.*

This is the pivotal antecedent and it already implements the core test proposed. Per the abstract, Kim et al. (2026) develop a framework for topological causal inference that defines treatment effects through differences in the topological structure of potential outcomes, summarized by **power-weighted silhouette functions** of persistence diagrams; provide an efficient **doubly robust estimator in a fully nonparametric model**; establish **functional weak convergence**; and construct a **formal test of the null hypothesis of no topological effect**.

Key technical readings:

- **Estimand.** Rather than taking Fréchet means of diagrams directly — which are non-unique (Turner et al., 2014; Mileyko et al., 2011) — the effect is a contrast of power-weighted silhouette functions (Chazal et al., 2015), i.e. an $L^2$-function-valued estimand. This deliberately sidesteps the geometric pathologies of $(\mathcal{D}_p, W_p)$ discussed in §5.3.
- **Estimator and inference.** Influence-function based and doubly robust, consistent with cross-fitting, with a functional CLT yielding confidence bands and a zero-effect test. For the group-comparison case this is essentially the proposed test, already done.
- **Authorship context.** Kwangho Kim is also an author of PLLay (Kim et al., 2020) and works on non-Euclidean causal inference, so the silhouette-functional route is coherent with that line.
- **Terminology caveat.** The literal acronym "TATE" (Topological Average Treatment Effect) was **not confirmed** in this abstract; it may originate in the second paper or be the user's own coinage.

### 2.2 "A Mathematical Framework for Topological Causal Data Analysis" — not located

A paper with this exact title could not be confirmed on the indices available. **This gap must be resolved by direct arXiv search.** Two 2026 preprints are the closest matches and appear to carry the vocabulary attributed to this paper. All details below are reported from abstracts, not verified from PDFs:

- **Saki et al. (2026)**, *Beyond Means: Topological Causal Effects under Persistent-Homology Ignorability*. Reported to use a potential-outcomes ATE/CATE framing; to argue that mean-based estimands miss unimodal-to-bimodal "same-mean" shape changes; to formalize a **persistent-homology ignorability** condition (the "topological ignorability" in question); to define topological analogues of CATE and ATE (the closest thing to a TATE); to prove identifiability up to an explicit error bound under approximate topological ignorability; and — crucially — to note that a **marginal persistence-diagram effect is not identified from conditional topological ignorability alone**, because persistent homology does not commute with mixtures over covariates, forcing the theory onto conditional estimands.
- **Faghihi (2026)**, *Topological Ignorability for Structural Causal Effects Beyond Means*. Reported to define topological ignorability as invariance of a chosen structural feature; to observe that when the summary map is **injective**, topological ignorability coincides with weak ignorability, whereas for non-injective summaries it identifies only the structural feature and not the full interventional law; and to validate on synthetic and Wisconsin breast-cancer hidden-confounding benchmarks using density-superlevel Betti and Euler signatures.

The non-commutativity-with-mixtures point, if confirmed, is directly load-bearing: it means a TATE-based two-sample test is a statement about *conditional* topological effects, and the "marginal PD equality" null is cleanly identified only under injectivity of the summary map or in the covariate-free case.

---

## 3. Two-Sample and Hypothesis Testing for Persistence Diagrams

### 3.1 Permutation tests

Robinson and Turner (2017) introduce the canonical randomization NHST for two samples of diagrams, using a loss functional built from pairwise bottleneck or Wasserstein distances, demonstrated on ADHD fMRI data. This is the direct competitor to any TATE-null two-sample test and must be benchmarked against.

Extensions: Islambekov et al. (2023) replace the expensive pairwise-Wasserstein loss with vectorized Betti functions, proving Wasserstein stability of an integrated Betti vectorization and adding a label-shuffling scheme that boosts power — the scalable permutation variant. Cai and Wang (2020) decouple the contributions of critical values from those of persistence pairings. Brain-network-specific variants include Dakurah (2025), a topologically invariant permutation test using closed-form 2-Wasserstein distances on diagrams with Hodge-Laplacian heat-kernel smoothing, and Wang et al. (2022), a spectral permutation test permuting Fourier coefficients of heat-kernel-smoothed diagrams.

Most recent and most rigorous: Han et al. (2026) propose a kernel-based permutation test on the **persistence intensity function**, derive a sharp variance bound controlling the unbounded cardinality of diagrams, and establish **minimax optimality** with a bandwidth-aggregation scheme. This is currently the state of the art for power-analyzed two-sample PD testing and a mandatory benchmark.

### 3.2 Metric-measure-space framing

Blumberg et al. (2014) study distributions of barcodes from fixed-size subsamples of a metric measure space, showing these give robust invariants usable for hypothesis testing and confidence intervals. This is the key reference for the **single-cloud-versus-single-cloud** case (§7.3): it converts one point cloud into a *distribution* over sub-sampled diagrams, making the comparison tractable. Follow-ups include the time-series-of-random-objects framework of van Delft and Blumberg (2024) and the result of Delft et al. (2024) that measures on bounded subspaces of diagram space are determined by their values on balls, justifying empirical ball-volume statistics.

### 3.3 Landscapes, silhouettes, and functional CLTs

Bubenik (2015) shows persistence landscapes live in a Banach or Hilbert space, obey a strong law and a CLT, are stable, and support standard statistical tests; for $p=2$ the Fréchet mean is the pointwise mean and the Fréchet variance the integrated pointwise variance.

Chazal et al. (2015) establish weak convergence of average landscapes, **bootstrap consistency**, and introduce **silhouettes** with an analogous theory — this is the theoretical engine behind the silhouette-functional CLT of Kim et al. (2026).

Berry et al. (2020) unify landscape and silhouette summaries, prove stability, and explicitly run two-sample tests (human versus monkey fibrin images). García-Redondo et al. (2025) extend functional CLTs and confidence bands to multiparameter persistence landscapes, relevant if a density-plus-scale bifiltration is pursued.

### 3.4 Kernel and MMD tests

The building-block kernels are the persistence scale-space kernel (Reininghaus et al., 2015), which is positive-definite and 1-Wasserstein stable; the persistence-weighted Gaussian kernel (Kusano et al., 2016, 2018), for which Kusano (2018) proves a strong law, a CLT, and confidence intervals for the kernel-mean expectation — a bona fide kernel-embedding inference theory; and the sliced Wasserstein kernel (Carrière et al., 2017), provably stable and discriminative, with a 2026 refinement using the Figalli–Gigli distance (Janthial et al., 2026).

Layering the MMD of Gretton et al. (2012) on any characteristic PD kernel yields a two-sample test. Kwitt et al. (2015) prove universality of a variant of the scale-space kernel and demonstrate MMD two-sample testing on real data — the most direct kernel competitor to a TATE-null test.

### 3.5 Vectorizations

Adams et al. (2017) introduce persistence images, a stable finite-dimensional vectorization; once vectorized, any Euclidean two-sample test applies. Newer stable vectorizations include signed-barcode measures for multiparameter PH (Loiseaux et al., 2023) and persistence B-spline grids (Dong et al., 2019).

### 3.6 Euler characteristic curves and transforms

Turner et al. (2014b) introduce the Persistent Homology Transform and the Euler Characteristic Transform; Curry et al. (2022) prove injectivity and sufficiency ("how many directions determine a shape") in Wasserstein and $L^p$ topologies — foundational for shape two-sample tests. Crawford et al. (2020) apply topological and functional data analysis via the Smooth ECT to glioblastoma clinical-outcome prediction.

Recent limit theory: Krebs et al. (2021) give Wasserstein and Kolmogorov convergence rates for the smoothed bootstrap of the Euler characteristic of Rips and Čech filtrations in the critical regime; Estrade and León (2016) prove a CLT for the Euler characteristic of Gaussian excursion sets; Thomas (2024) gives a strong law and CLT for sublevel-set persistence measures of stationary processes; Dłotko and Gurnari (2022) establish stability and distributed computation of Euler curves and profiles for large data. Euler-characteristic summaries are the natural cheap statistic for the cubical, big-data side of the filtration sub-question.

### 3.7 Bootstrap and confidence sets

Fasy et al. (2014) is the landmark result: confidence sets for persistence diagrams that separate topological signal from noise, with several estimators. Chazal et al. (2013) give empirical bootstrap confidence intervals for diagrams and landscapes.

Roycraft et al. (2023) show that the **naïve bootstrap can fail** and that the **smoothed bootstrap restores validity** for persistent Betti numbers, Euler characteristics, and $k$-NN edge lengths — essential if the test statistic is a persistent Betti number. Robust alternatives include Vishwanath et al. (2020), robust diagrams via RKHS density with uniform confidence bands, and Glenn et al. (2024), confidence regions for a single image's diagram, relevant to the single-observation case.

### 3.8 Fréchet means and variance of diagrams

Mileyko et al. (2011) show $(\mathcal{D}_p, W_p)$ is complete and separable, supporting expectations, variances, and percentiles. Turner et al. (2014a) prove existence of Fréchet means, give a convergent algorithm, and establish a law of large numbers — **but means are non-unique**. This non-uniqueness is precisely why Kim et al. (2026) work with silhouette functionals rather than diagram-space Fréchet means.

### 3.9 Explicit type-I and power frameworks

Genuine power analysis is rare. The standouts are the minimax-optimal intensity-function test of Han et al. (2026) and the empirical power study of Robinson and Turner (2017). Most other methods report only empirical size and power via simulation.

---

## 4. Statistical Foundations and Limit Theorems

### 4.1 Stability

Cohen-Steiner et al. (2007) prove bottleneck stability; Chazal et al. (2009, 2014) extend to Wasserstein stability. The modern refinement is Skraba and Turner (2020), giving $p$-Wasserstein stability in terms of the $p$-norm of perturbations — far tighter and less outlier-pessimistic than $\infty$-norm bottleneck stability, and directly relevant since it controls how sampling noise propagates into the test statistic. Bauer and Edelsbrunner (2017) prove the bottleneck distance is **universal**, the largest stable distance, for extended diagrams. Stability guarantees the estimand is a continuous functional of the underlying measure, a prerequisite for any CLT-based test.

### 4.2 Convergence rates and minimax results

Chazal et al. (2015b) establish minimax rates for persistent homology estimation; Fasy et al. (2014) provide the rates underlying confidence sets. Arnal et al. (2024) show Čech diagrams converge in $W_p$ exactly when $p > m$ (the manifold dimension), with improved bottleneck bounds under the manifold hypothesis — a sharp caveat for $W_p$-based test statistics on manifold data.

### 4.3 CLTs for topological summaries of random point clouds

See Bobrowski et al. (2017) on maximally persistent cycles in random geometric complexes; Hiraoka et al. (2018) on limit theorems for persistence diagrams; Krebs and Polonik (2019) and Krebs et al. (2021) on Euler characteristic approximation and bootstrap; and Roycraft et al. (2023) on persistent Betti number CLTs and bootstrap. These supply the distributional limits that make a non-permutation, asymptotic TATE test feasible. The main technical hazards are critical-regime dependence and the requirement of the smoothed rather than naïve bootstrap.

### 4.4 Distributional limits for total persistence and persistent Betti numbers

Divol and Chazal (2021) and Divol and Lacombe (2021) treat diagrams as Radon measures on the half-plane via optimal partial transport, yielding laws of large numbers, a geometric description of barycenters for **any** distribution of diagrams, and a characterization of continuous linear representations. This measure-theoretic framing is arguably the cleanest foundation for defining "TATE $= 0$" as equality of expected persistence measures, and it neatly avoids diagram-space curvature pathologies.

---

## 5. Causal Inference on Non-Euclidean and Object-Valued Outcomes

### 5.1 Fréchet regression

Petersen and Müller (2019) introduce Fréchet regression: conditional Fréchet means for metric-space-valued responses, in global and local versions, with convergence rates via empirical-process methods and limit distributions in the Hilbert case. This is the estimation backbone for object-valued treatment effects and, via a synthetic binary predictor, is itself a candidate estimator for a TATE-style contrast.

### 5.2 Causal inference with metric-space and object outcomes

- **Kurisu et al. (2024)** define the **geodesic average treatment effect** — a geodesic between the Fréchet means of treated and untreated — develop Fréchet-regression estimators, prove consistency and rates, and provide uncertainty quantification and a test. This is the general-metric-space parent of the topological TATE, and $(\mathcal{D}_2, W_2)$ is a candidate instance.
- **Bhattacharjee et al. (2025)** give doubly-debiased, cross-fitted causal inference for random-object outcomes with continuous treatments, via Hilbert-space embeddings, efficient influence functions, and conformal counterfactual prediction — the methodological template Kim et al. (2026) follow.
- **Shin et al. (2024)** define absolute average and median treatment effects on general metric spaces, with stratification estimators, bootstrap confidence intervals, and an explicit test of **Fisher's sharp null of zero effect**, applied to corpus-callosum shape in Alzheimer's. This is a very close antecedent to the proposed "TATE $= 0$" test.
- Related: geodesic difference-in-differences (Zhou et al., 2025), regression discontinuity for geodesic-space objects (Kurisu et al., 2025), mediation for random objects (Tan et al., 2026), and Wasserstein-space distributional treatment effects.

### 5.3 Tests of "no effect" for random objects — and why diagram space breaks them

Dubey and Müller (2019) derive a CLT for the Fréchet variance and a consistent asymptotic-variance estimator, yielding a $k$-sample test comparing Fréchet means **and** variances of metric-space-valued data. Since $(\mathcal{D}_p, W_p)$ is complete and separable (Mileyko et al., 2011), Fréchet ANOVA applies directly to two groups of diagrams and is arguably **the single most important comparator** for the proposed test. Related tests include Fréchet MANOVA across multiple metric spaces (Fout et al., 2023), the generalized Fréchet test for repeated object measurements with a weighted-$\chi^2$ limit (Zhang et al., 2025), depth-based two-sample tests for random objects (Chen et al., 2025), and energy-distance and graph-based tests (Chen & Friedman, 2017; Lovato et al., 2020).

**The critical caveat.** Dubey–Müller's CLT and much random-object theory assume geodesic spaces with curvature bounds — often nonpositive or Hadamard — or unique Fréchet means. Diagram space violates these:

- Fréchet means are non-unique (Mileyko et al., 2011; Turner et al., 2014a).
- Means of continuously varying diagrams need not vary continuously, motivating *probabilistic* Fréchet means (Munch et al., 2015).
- Che et al. (2021) show $(\mathcal{D}_2, W_2)$ is geodesic and **nonnegatively curved** in the Alexandrov sense but has infinite covering, Hausdorff, Assouad, and Assouad–Nagata dimensions.
- Bubenik and Wagner (2020) study embedding obstructions.
- Cao and Monod (2022) give a "flatness" grouping condition guaranteeing uniqueness only in special cases.
- Song et al. (2025) show the relationship between Fréchet and metric variance depends on Alexandrov curvature, so a variance-based test's calibration is curvature-sensitive.

**Consequence.** Standard random-object ANOVA machinery does not transfer to diagram space off the shelf. This is a real theoretical gap and a legitimate source of novelty — and it is exactly why Kim et al. (2026) and Berry et al. (2020) detour through *functional* summaries that live in well-behaved Hilbert or Banach spaces.

---

## 6. Filtration Comparison: Cubical versus Vietoris–Rips

### 6.1 Interleaving and convergence theory

Chazal et al. (2014) establish Gromov–Hausdorff stability of Rips, Čech, and witness persistence; the persistence-modules framework of Chazal et al. (2016) formalizes the Rips–Čech interleaving (a factor-of-2 multiplicative interleaving in the metric). The Nerve theorem links Čech complexes to the union of balls, and Bauer and Edelsbrunner (2017) relate Čech, Delaunay, and Alpha filtrations. Sparse-Rips approximations with explicit interleaving error include Sheehy (2013) and Dey et al. (2014); Virk (2021) sharpens Rips approximation and stability with explicit density-to-scale thresholds.

These give the theoretical anchor for the equivalence margin: a cubical sublevel-set filtration on a grid and a Rips or Čech filtration on the same cloud can be interleaved with an error controlled by the voxel side length $h$, so the bottleneck or Wasserstein error tends to zero as $h \to 0$.

**Identified gap.** No single paper was located giving a clean closed-form bottleneck bound explicitly in terms of cube side length between cubical and Rips diagrams. This appears to be a genuine gap and a natural lemma to prove — likely via a Hausdorff-distance bound $d_H(\text{grid approximation}, \text{point cloud}) \le (\sqrt{d}/2)\,h$ combined with geometric-complex stability. This is the thinnest-sourced technical link in the whole proposal.

### 6.2 Digital topology and discretization error

Wagner et al. (2012) give efficient cubical PH computation. Bleile et al. (2022) relate the two dual cubical constructions — direct versus indirect adjacency — and show how to convert between their diagrams, which matters because "the cubical diagram" is not unique. See also Kaji et al. (2020) on Cubical Ripser and Choe and Ramanna (2022) on cubical versus simplicial complexes for images. Grid-based density estimation feeding sublevel-set filtrations connects to Bubenik and Kim (2007) and the density-based confidence sets of Fasy et al. (2014).

### 6.3 Statistically choosing a filtration or resolution

**No paper was found that statistically certifies a cheap filtration against an expensive one via a hypothesis test.** This exact sub-question appears open.

Adjacent work is all *filtration learning*: Hofer et al. (2020) on graph filtration learning differentiable through PH; Nishikawa et al. (2023) on adaptive, isometry-invariant learnable filtrations for point clouds with a finite-dimensional approximation theorem; Kim et al. (2020) on PLLay, a learnable topological layer with DTM filtration; Carrière et al. (2020) on PersLay; and multiparameter and cubical vectorization learning (Corbet et al., 2019; Vipond, 2020; Xin et al., 2023; Korkmaz et al., 2025). All of these *optimize* filtrations for a downstream loss; **none frames resolution selection as a validated equivalence guarantee.** That is the opening.

### 6.4 Computational complexity

Otter et al. (2017) is the standard complexity and benchmark reference: Rips PH is worst-case exponential in the number of points and cubically expensive in practice, whereas cubical PH scales with the number of voxels. Ripser (Bauer, 2021) is the fastest Rips engine; Cubical Ripser (Kaji et al., 2020) is the fastest cubical engine; GUDHI (Maria et al., 2014) provides both. This asymmetry is what makes a certified coarse-cubical approximation valuable for very large clouds.

---

## 7. Equivalence Testing (TOST): the Correct Frame for the Sub-Question

The filtration sub-question is an **equivalence** test, not a difference test. Certifying that the cheap cubical diagram is not meaningfully different from Rips is a null-of-equivalence problem; a non-significant difference test would merely reflect low power, not similarity.

### 7.1 Core references

Schuirmann (1987) introduces TOST; Lakens (2017) provides the modern primer with power analysis; Lauzon and Caffo (2009) handle multiplicity control for TOST — relevant to testing across $H_0$, $H_1$, $H_2$ and across grid sizes. Heteroscedastic and finite-sample-corrected variants include the $\alpha$-TOST of Boulaguiem et al. (2024a), uniformly more powerful than standard TOST, and Shieh (2021). Multivariate TOST for multiple endpoints (Boulaguiem et al., 2024b) is directly analogous to testing equivalence simultaneously across homological dimensions. Bayesian bioequivalence alternatives are surveyed in Peck et al. (2022).

### 7.2 The gap

**No application of equivalence or TOST testing to TDA, or to metric-space and random-object data, was found.** Building a metric-space TOST — for example testing

$$H_0: \mathbb{E}\big[W_p(D_{\text{cubical}}, D_{\text{Rips}})\big] \ge \delta \quad \text{versus} \quad H_1: \mathbb{E}\big[W_p(D_{\text{cubical}}, D_{\text{Rips}})\big] < \delta,$$

or equivalently that the Fréchet or silhouette contrast lies within an equivalence band $\delta$ set by the interleaving bound — would be a **new methodological object**. This is the most defensible novelty in the entire proposal.

### 7.3 The single-observation case

Comparing one point cloud against one point cloud is tractable only by manufacturing a distribution from each observation. Workable routes:

1. **Subsampling** fixed-size sub-clouds from each cloud, following Blumberg et al. (2014), to obtain two empirical distributions of diagrams, then applying any two-sample, Fréchet ANOVA, or MMD test. This is the most principled and most cited route.
2. **Permutation over points** within each cloud (Robinson & Turner, 2017).
3. **Bootstrap confidence sets** for each single diagram (Fasy et al., 2014; Chazal et al., 2013) with an overlap test.
4. **Single-image confidence regions** (Glenn et al., 2024).

The fundamental obstacle is that a single point cloud yields one diagram, so there is no replication; any test must impose a resampling model whose validity rests on manifold-regularity or metric-measure-space assumptions and is only asymptotic. This should be framed as conditional inference given the observed cloud, with honest caveats.

---

## 8. Gaps, Positioning, and Critical Assessment

### 8.1 What already exists — do not reinvent

1. A TATE-null test for persistence diagrams via silhouette functionals with a functional CLT and double robustness: **Kim et al. (2026) already has this.** The primary idea, in the group-comparison setting, is largely done.
2. Two-sample tests for diagrams: Robinson and Turner (2017); kernel-MMD (Kwitt et al., 2015); functional-summary tests (Berry et al., 2020); the minimax intensity-function test (Han et al., 2026).
3. "No treatment effect" tests on metric spaces and random objects: Dubey and Müller (2019); Shin et al. (2024); Kurisu et al. (2024).

### 8.2 Is the causal reframing novel, and does it buy anything?

**Honest answer:** framing a two-sample topological test as "TATE $= 0$ with a synthetic binary treatment $Z$" is, in the covariate-free, two-independent-i.i.d.-groups case, a **re-labelling** of the classical two-sample problem. With $Z$ assigned by group and no confounders, the propensity score is constant, the doubly-robust estimator collapses to a simple difference of group summaries, and the TATE null coincides exactly with the Robinson–Turner and Fréchet-ANOVA nulls.

It buys real statistical value only when covariates or confounders are present. Then the causal machinery delivers:

- a **covariate-adjusted** topological comparison — the honest object when groups differ in nuisance covariates;
- **semiparametric efficiency and double robustness**, with $\sqrt{n}$-rate inference under cross-fitting despite nonparametric nuisance estimation;
- a formal statement, via topological ignorability, of exactly what is being tested.

But per the (unconfirmed) non-commutativity point in Saki et al. (2026), adjustment identifies a **conditional or covariate-standardized** topological effect, not the marginal "are these two diagrams equal in distribution" null. The paper must state this precisely.

**Recommendation:** sell the causal framing as *covariate-adjusted two-sample topological testing with efficiency guarantees*, not as a new solution to the plain two-sample problem.

### 8.3 Strongest publishable contribution and venue

The **filtration-equivalence result** is the strongest and most novel: a metric-space equivalence test that certifies the coarsest cubical resolution statistically indistinguishable from Vietoris–Rips, with the margin $\delta$ derived from the cubical-to-Rips interleaving bound, validated by finite-sample coverage and a complexity/accuracy trade-off curve. It combines a novel theory lemma (the explicit bound in cube side length $h$), a novel methodological object (metric-space TOST for diagrams), and a high-value application (cheap PH for massive clouds). Target **SIAM Journal on the Mathematics of Data Science** or **Annals of Applied Statistics**.

The covariate-adjusted TATE two-sample test — positioned as complementary to Kim et al. (2026) and benchmarked against Robinson and Turner (2017), Dubey and Müller (2019), and Han et al. (2026) — is a strong **JMLR, NeurIPS, or ICML** companion.

### 8.4 Open problems and likely technical obstacles

1. **No canonical null distribution** on diagram space, forcing reliance on permutation or smoothed-bootstrap calibration; the naïve bootstrap is inconsistent for persistent Betti numbers (Roycraft et al., 2023).
2. **Non-uniqueness of Fréchet means** and the infinite-dimensional, nonnegatively-but-unboundedly-curved geometry of $(\mathcal{D}_2, W_2)$ (Che et al., 2021; Turner et al., 2014a; Song et al., 2025). Standard random-object CLTs do not transfer; test instead on functional summaries or the persistence-measure representation (Divol & Chazal, 2021).
3. **Multiple testing** across $H_0$, $H_1$, $H_2$ and across candidate grid sizes, requiring FWER or FDR control (Lauzon & Caffo, 2009; Boulaguiem et al., 2024b).
4. **Dependence between birth–death pairs** within a diagram violates the independence many CLTs assume; use measure-valued limit theory (Divol & Chazal, 2021; Thomas, 2024) or exchangeable permutation nulls.
5. **Setting the equivalence margin $\delta$**, which must be justified by the interleaving bound or by the smallest topologically meaningful lifetime; results are sensitive to it.
6. **The summary and kernel choice is itself a modeling decision** with power consequences (Cai & Wang, 2020; Zhao & Wang, 2019); the conclusion is conditional on the chosen filtration *and* summary, which should be stated as a scope limitation.
7. **Unbounded cardinality of diagrams** must be controlled for a finite-variance test statistic (Han et al., 2026).

---

## 9. Staged Recommendations

1. **Resolve the second paper's identity first.** Search arXiv directly for the exact title and for Saki et al. (2026) and Faghihi (2026); confirm whether "TATE" and "topological ignorability" are formally defined there and whether the marginal-effect non-identification result is real. If confirmed, it constrains the two-sample null to a conditional statement — build the paper around conditional, covariate-standardized topological effects. If the marginal null is what is needed, restrict to the injective-summary or covariate-free regime and prove identification there.

2. **Position against Kim et al. (2026) explicitly; do not duplicate.** Read the silhouette-functional CLT and DR estimator in full. The two-sample contribution must add covariate adjustment as the whole point, plus benchmarks that Kim et al. omit. **Threshold for keeping the causal wrapper:** it must demonstrably beat the plain permutation and Fréchet-ANOVA tests in covariate-imbalanced simulations, in power or bias. If it does not beat them without covariates, say so plainly and drop the causal claim for that regime.

3. **Make the equivalence result the flagship.** (a) Prove the cubical-to-Rips bottleneck or Wasserstein bound in cube side length $h$; (b) build a metric-space TOST with $\delta$ anchored to that bound; (c) on validation data, report the largest $h$ at which equivalence is established at level $\alpha$, plus the runtime saving. **Decision rule for a standalone paper:** finite-sample type-I error of the equivalence null $\le \alpha$ and empirical equivalence power $\ge 0.8$ at the theory-predicted $h$, across at least three dataset families.

4. **Handle the geometry safely.** Default to functional summaries or persistence measures for all inference. Use diagram-space Fréchet means only under an invocable uniqueness condition (Cao & Monod, 2022), and then only as a robustness check.

5. **For the single-cloud case**, adopt Blumberg-style subsampling, present results as conditional resampling inference, and state the no-replication limitation up front.

6. **Control multiplicity** across homological dimensions and grid sizes, and use the smoothed bootstrap wherever Betti or Euler statistics enter.

---

## 10. Epistemic Caveats

**Important:** I do not have live access to arXiv or a citation database from within this document-creation step, and citation details below were assembled from search results whose venue and year fields may differ from the canonical published record. **Please verify every citation — authors, titles, venues, years, and arXiv IDs — before using this in a manuscript.** Some references, particularly the 2025–2026 preprints, could not be inspected at the PDF level and may be inaccurately titled or attributed.

- **Verified from abstracts or indexing records:** the existence, authorship, and high-level content of Kim et al. (2026), and the Section 3–7 peer-reviewed works.
- **Not verified and not located:** a paper titled exactly *A Mathematical Framework for Topological Causal Data Analysis*. Its attributes are inferred from two closely matching 2026 preprints reported from abstracts only. Treat their assumption statements, the "TATE" acronym, the topological-ignorability definition, and the marginal-effect non-identification claim as provisional.
- **Publication years and venues** should be re-checked; several index dates differ from official publication years (e.g. Robinson and Turner's work circulated from 2013 and was published later; Bubenik's JMLR paper is 2015).
- **Identified genuine gaps, i.e. novelty opportunities:** (i) no explicit closed-form bottleneck bound between cubical and Rips diagrams in cube side length; (ii) no equivalence or TOST methodology applied to TDA or metric-space objects; (iii) no method that statistically certifies a cheap filtration or resolution against an expensive one; (iv) random-object ANOVA CLTs are not established for the curvature-pathological diagram space.

---

## References

Adams, H., Emerson, T., Kirby, M., Neville, R., Peterson, C., Shipman, P., Chepushtanova, S., Hanson, E., Motta, F., & Ziegelmeier, L. (2017). Persistence images: A stable vector representation of persistent homology. *Journal of Machine Learning Research*, 18(8), 1–35.

Arnal, C., Cohen-Steiner, D., & Divol, V. (2024). Critical points of the optimal quantum control landscape / Convergence rates for Čech persistence diagrams. *arXiv preprint*. [Verify exact title and ID.]

Bauer, U. (2021). Ripser: Efficient computation of Vietoris–Rips persistence barcodes. *Journal of Applied and Computational Topology*, 5(3), 391–423.

Bauer, U., & Edelsbrunner, H. (2017). The Morse theory of Čech and Delaunay complexes. *Transactions of the American Mathematical Society*, 369(5), 3741–3762.

Berry, E., Chen, Y.-C., Cisewski-Kehe, J., & Fasy, B. T. (2020). Functional summaries of persistence diagrams. *Journal of Applied and Computational Topology*, 4(2), 211–262.

Bhattacharjee, S., Zhou, Y., & Müller, H.-G. (2025). Doubly debiased causal inference for random-object outcomes with continuous treatments. *arXiv preprint*. [Verify.]

Bleile, B., Garin, A., Heiss, T., Maggs, K., & Robins, V. (2022). The persistent homology of dual digital image constructions. In *Research in Computational Topology 2* (pp. 1–26). Springer.

Blumberg, A. J., Gal, I., Mandell, M. A., & Rabadan, R. (2014). Robust statistics, hypothesis testing, and confidence intervals for persistent homology on metric measure spaces. *Foundations of Computational Mathematics*, 14(4), 745–789.

Bobrowski, O., Kahle, M., & Skraba, P. (2017). Maximally persistent cycles in random geometric complexes. *Annals of Applied Probability*, 27(4), 2032–2060.

Boulaguiem, Y., Quartier, J., Lapteva, M., Kalia, Y. N., Victoria-Feser, M.-P., Guerrier, S., & Couturier, D.-L. (2024a). Finite-sample adjustments for average equivalence testing ($\alpha$-TOST). *Statistics in Medicine*, 43(9), 1656–1671.

Boulaguiem, Y., Couturier, D.-L., et al. (2024b). Multivariate equivalence testing for multiple endpoints. *arXiv preprint*. [Verify.]

Bubenik, P. (2015). Statistical topological data analysis using persistence landscapes. *Journal of Machine Learning Research*, 16(1), 77–102.

Bubenik, P., & Kim, P. T. (2007). A statistical approach to persistent homology. *Homology, Homotopy and Applications*, 9(2), 337–362.

Bubenik, P., & Wagner, A. (2020). Embeddings of persistence diagrams into Hilbert spaces. *Journal of Applied and Computational Topology*, 4(3), 339–351.

Cai, C., & Wang, Y. (2020). Understanding the power of persistence pairing via permutation test. *arXiv preprint arXiv:2001.06058*.

Cao, Y., & Monod, A. (2022). Approximating persistent homology for large datasets / On the uniqueness of Fréchet means of persistence diagrams. *arXiv preprint*. [Verify exact title.]

Carrière, M., Chazal, F., Ike, Y., Lacombe, T., Royer, M., & Umeda, Y. (2020). PersLay: A neural network layer for persistence diagrams and new graph topological signatures. In *Proceedings of AISTATS 2020* (pp. 2786–2796).

Carrière, M., Cuturi, M., & Oudot, S. (2017). Sliced Wasserstein kernel for persistence diagrams. In *Proceedings of ICML 2017* (pp. 664–673).

Chazal, F., Cohen-Steiner, D., Glisse, M., Guibas, L. J., & Oudot, S. Y. (2009). Proximity of persistence modules and their diagrams. In *Proceedings of the 25th Annual Symposium on Computational Geometry* (pp. 237–246).

Chazal, F., de Silva, V., Glisse, M., & Oudot, S. (2016). *The structure and stability of persistence modules*. Springer.

Chazal, F., de Silva, V., & Oudot, S. (2014). Persistence stability for geometric complexes. *Geometriae Dedicata*, 173(1), 193–214.

Chazal, F., Fasy, B. T., Lecci, F., Michel, B., Rinaldo, A., & Wasserman, L. (2013). On the bootstrap for persistence diagrams and landscapes. *arXiv preprint arXiv:1311.0376*.

Chazal, F., Fasy, B. T., Lecci, F., Rinaldo, A., & Wasserman, L. (2015a). Stochastic convergence of persistence landscapes and silhouettes. In *Proceedings of the 31st International Symposium on Computational Geometry (SoCG)* (pp. 474–483).

Chazal, F., Glisse, M., Labruère, C., & Michel, B. (2015b). Convergence rates for persistence diagram estimation in topological data analysis. *Journal of Machine Learning Research*, 16(1), 3603–3635.

Che, M., Galaz-García, F., Guijarro, L., & Membrillo Solis, I. (2021). Metric geometry of spaces of persistence diagrams. *arXiv preprint arXiv:2109.14697*.

Chen, H., & Friedman, J. H. (2017). A new graph-based two-sample test for multivariate and object data. *Journal of the American Statistical Association*, 112(517), 397–409.

Chen, H., et al. (2025). A depth-based two-sample test for random objects. [Verify.]

Choe, S., & Ramanna, S. (2022). Cubical homology-based machine learning: An application in image classification. *Axioms*, 11(3), 112.

Cohen-Steiner, D., Edelsbrunner, H., & Harer, J. (2007). Stability of persistence diagrams. *Discrete & Computational Geometry*, 37(1), 103–120.

Corbet, R., Fugacci, U., Kerber, M., Landi, C., & Wang, B. (2019). A kernel for multi-parameter persistent homology. *Computers & Graphics: X*, 2, 100005.

Crawford, L., Monod, A., Chen, A. X., Mukherjee, S., & Rabadán, R. (2020). Predicting clinical outcomes in glioblastoma: An application of topological and functional data analysis. *Journal of the American Statistical Association*, 115(531), 1139–1150.

Curry, J., Mukherjee, S., & Turner, K. (2022). How many directions determine a shape and other sufficiency results for two topological transforms. *Transactions of the American Mathematical Society, Series B*, 9, 1006–1043.

Dakurah, S. (2025). A topologically invariant permutation test for brain network comparison. [Verify.]

Delft, A. van, & Blumberg, A. J. (2024). A statistical framework for analyzing time-varying random objects. [Verify.]

Dey, T. K., Fan, F., & Wang, Y. (2014). Computing topological persistence for simplicial maps. In *Proceedings of the 30th Annual Symposium on Computational Geometry* (pp. 345–354).

Divol, V., & Chazal, F. (2021). The density of expected persistence diagrams and its kernel-based estimation. *Journal of Computational Geometry*, 10(2), 127–153.

Divol, V., & Lacombe, T. (2021). Understanding the topology and the geometry of the space of persistence diagrams via optimal partial transport. *Journal of Applied and Computational Topology*, 5(1), 1–53.

Dłotko, P., & Gurnari, D. (2022). Euler characteristic curves and profiles: A stable shape invariant for big data problems. *GigaScience*, 12, giad094.

Dong, Y., et al. (2019). Persistence B-spline grids: Stable vector representation of persistence diagrams. [Verify.]

Dubey, P., & Müller, H.-G. (2019). Fréchet analysis of variance for random objects. *Biometrika*, 106(4), 803–821.

Estrade, A., & León, J. R. (2016). A central limit theorem for the Euler characteristic of a Gaussian excursion set. *Annals of Probability*, 44(6), 3849–3878.

Faghihi, M. (2026). Topological ignorability for structural causal effects beyond means. *arXiv preprint*. [Unverified — locate and confirm.]

Fasy, B. T., Lecci, F., Rinaldo, A., Wasserman, L., Balakrishnan, S., & Singh, A. (2014). Confidence sets for persistence diagrams. *Annals of Statistics*, 42(6), 2301–2339.

Fout, A., et al. (2023). Fréchet MANOVA for multiple metric spaces. [Verify.]

García-Redondo, I., et al. (2025). Multiparameter persistence landscapes: Functional CLTs and confidence bands. [Verify.]

Glenn, J., et al. (2024). Confidence regions for persistence diagrams of a single image. [Verify.]

Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). A kernel two-sample test. *Journal of Machine Learning Research*, 13, 723–773.

Han, Q., et al. (2026). A two-sample test on weighted persistence intensity functions. *arXiv preprint*. [Verify authorship.]

Hiraoka, Y., Shirai, T., & Trinh, K. D. (2018). Limit theorems for persistence diagrams. *Annals of Applied Probability*, 28(5), 2740–2780.

Hofer, C., Graf, F., Rieck, B., Niethammer, M., & Kwitt, R. (2020). Graph filtration learning. In *Proceedings of ICML 2020* (pp. 4314–4323).

Islambekov, U., et al. (2023). Vector summaries of persistence diagrams for permutation-based hypothesis testing. *Foundations of Data Science*. [Verify.]

Janthial, A., et al. (2026). Sliced Wasserstein kernels via the Figalli–Gigli distance. [Verify.]

Kaji, S., Sudo, T., & Ahara, K. (2020). Cubical Ripser: Software for computing persistent homology of image and volume data. *arXiv preprint arXiv:2005.12692*.

Kim, K., Kim, J., Zaheer, M., Kim, S., Chazal, F., & Wasserman, L. (2020). PLLay: Efficient topological layer based on persistent landscapes. In *Advances in Neural Information Processing Systems 33*.

Kim, K., et al. (2026). Topological causal effects. *arXiv preprint*. [Primary source — verify ID and read in full.]

Korkmaz, C., et al. (2025). CuMPerLay: Cubical multiparameter persistence vectorization layer. [Verify.]

Krebs, J., & Polonik, W. (2019). On the asymptotic normality of persistent Betti numbers. *arXiv preprint arXiv:1903.03280*.

Krebs, J., Roycraft, B., & Polonik, W. (2021). On approximation theorems for the Euler characteristic with applications to the bootstrap. *Electronic Journal of Statistics*, 15(2), 4462–4509.

Kurisu, D., Zhou, Y., Otsu, T., & Müller, H.-G. (2024). Geodesic causal inference. *arXiv preprint arXiv:2406.19604*.

Kurisu, D., et al. (2025). Regression discontinuity designs for random objects in geodesic spaces. [Verify.]

Kusano, G. (2018). On the expectation of a persistence diagram by the persistence weighted kernel. *Japan Journal of Industrial and Applied Mathematics*, 36, 861–892.

Kusano, G., Hiraoka, Y., & Fukumizu, K. (2016). Persistence weighted Gaussian kernel for topological data analysis. In *Proceedings of ICML 2016* (pp. 2004–2013).

Kwitt, R., Huber, S., Niethammer, M., Lin, W., & Bauer, U. (2015). Statistical topological data analysis — A kernel perspective. In *Advances in Neural Information Processing Systems 28*.

Lakens, D. (2017). Equivalence tests: A practical primer for t-tests, correlations, and meta-analyses. *Social Psychological and Personality Science*, 8(4), 355–362.

Lauzon, C., & Caffo, B. (2009). Easy multiplicity control in equivalence testing using two one-sided tests. *The American Statistician*, 63(2), 147–154.

Loiseaux, D., Carrière, M., & Blumberg, A. J. (2023). A framework for fast and stable representations of multiparameter persistent homology decompositions. In *Advances in Neural Information Processing Systems 36*.

Lovato, I., Pini, A., Stamm, A., & Vantini, S. (2020). Model-free two-sample test for network-valued data. *Computational Statistics & Data Analysis*, 144, 106896.

Maria, C., Boissonnat, J.-D., Glisse, M., & Yvinec, M. (2014). The GUDHI library: Simplicial complexes and persistent homology. In *Mathematical Software – ICMS 2014* (pp. 167–174).

Mileyko, Y., Mukherjee, S., & Harer, J. (2011). Probability measures on the space of persistence diagrams. *Inverse Problems*, 27(12), 124007.

Munch, E., Turner, K., Bendich, P., Mukherjee, S., Mattingly, J., & Harer, J. (2015). Probabilistic Fréchet means for time varying persistence diagrams. *Electronic Journal of Statistics*, 9(1), 1173–1204.

Nishikawa, N., et al. (2023). Adaptive topological feature via persistent homology: Filtration learning for point clouds. [Verify.]

Otter, N., Porter, M. A., Tillmann, U., Grindrod, P., & Harrington, H. A. (2017). A roadmap for the computation of persistent homology. *EPJ Data Science*, 6(1), 17.

Peck, C. C., et al. (2022). Bayesian approaches to bioequivalence assessment. [Verify.]

Petersen, A., & Müller, H.-G. (2019). Fréchet regression for random objects with Euclidean predictors. *Annals of Statistics*, 47(2), 691–719.

Reininghaus, J., Huber, S., Bauer, U., & Kwitt, R. (2015). A stable multi-scale kernel for topological machine learning. In *Proceedings of CVPR 2015* (pp. 4741–4748).

Robinson, A., & Turner, K. (2017). Hypothesis testing for topological data analysis. *Journal of Applied and Computational Topology*, 1(2), 241–261.

Roycraft, B., Krebs, J., & Polonik, W. (2023). Bootstrapping persistent Betti numbers and other stabilizing statistics. *Annals of Statistics*, 51(4), 1484–1509.

Saki, A., et al. (2026). Beyond means: Topological causal effects under persistent-homology ignorability. *arXiv preprint*. [Unverified — locate and confirm.]

Schuirmann, D. J. (1987). A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. *Journal of Pharmacokinetics and Biopharmaceutics*, 15(6), 657–680.

Sheehy, D. R. (2013). Linear-size approximations to the Vietoris–Rips filtration. *Discrete & Computational Geometry*, 49(4), 778–796.

Shieh, G. (2021). Assessing equivalence with heteroscedastic and unbalanced designs. [Verify.]

Shin, S., et al. (2024). Absolute average and median treatment effects as causal estimands on metric spaces. *arXiv preprint*. [Verify.]

Skraba, P., & Turner, K. (2020). Wasserstein stability for persistence diagrams. *arXiv preprint arXiv:2006.16824*.

Song, D., et al. (2025). Fréchet variance, metric variance, and Alexandrov curvature. *Journal of the American Statistical Association*. [Verify.]

Tan, X., et al. (2026). ROMA: Random object mediation analysis. [Verify.]

Thomas, A. M. (2024). Limit theorems for persistence diagrams of sublevel sets of stationary processes. [Verify.]

Turner, K., Mileyko, Y., Mukherjee, S., & Harer, J. (2014a). Fréchet means for distributions of persistence diagrams. *Discrete & Computational Geometry*, 52(1), 44–70.

Turner, K., Mukherjee, S., & Boyer, D. M. (2014b). Persistent homology transform for modeling shapes and surfaces. *Information and Inference*, 3(4), 310–344.

Vipond, O. (2020). Multiparameter persistence landscapes. *Journal of Machine Learning Research*, 21(61), 1–38.

Virk, Ž. (2021). Rips complexes as nerves and a functorial Dowker–nerve diagram. *Mediterranean Journal of Mathematics*, 18, 58.

Vishwanath, S., Fukumizu, K., Kuriki, S., & Sriperumbudur, B. (2020). Robust persistence diagrams using reproducing kernels. In *Advances in Neural Information Processing Systems 33*.

Wagner, H., Chen, C., & Vuçini, E. (2012). Efficient computation of persistent homology for cubical data. In *Topological Methods in Data Analysis and Visualization II* (pp. 91–106). Springer.

Wang, Y., et al. (2022). A spectral permutation test for brain network comparison. In *Proceedings of ICASSP 2022*.

Xin, C., Mukherjee, S., Samaga, S. N., & Dey, T. K. (2023). GRIL: A 2-parameter persistence based vectorization for machine learning. In *Proceedings of the 2nd Annual TAG-DS Workshop*.

Zhang, Q., et al. (2025). A generalized Fréchet test for repeated measurements of random objects. [Verify.]

Zhao, Q., & Wang, Y. (2019). Learning metrics for persistence-based summaries and applications for graph classification. In *Advances in Neural Information Processing Systems 32*.

Zhou, Y., et al. (2025). Geodesic difference-in-differences. [Verify.]
