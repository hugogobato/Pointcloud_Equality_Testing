# Literature review and research assessment

Your idea is promising, but it combines three distinct statistical questions that should not be conflated:

1. **Two-sample equality:** do two populations of point clouds have the same topological distribution?
2. **Topological mean/effect:** does group membership change the expected persistence-based representation?
3. **Filtration equivalence:** does a cubical filtration with resolution $h$ produce a statistically equivalent representation to a Vietoris–Rips filtration?

The two TCDA papers are especially relevant to the second question, but existing persistence-diagram two-sample testing is already fairly developed for the first. The strongest research opportunity is therefore likely the integration of these ideas into a unified, filtration-aware test with explicit equivalence guarantees and a principled resolution-selection procedure.

## 1. Relevant existing literature

### Topological causal effects

*Topological Causal Effects* defines a functional topological treatment effect using the difference between expected power-weighted silhouettes of persistence diagrams:

$$
\psi_d(t)
=
\mathbb{E}\left[
\phi_d(\mathcal D_d^1)(t)
-
\phi_d(\mathcal D_d^0)(t)
\right],
$$

where $d$ is the homology degree and $t$ is the filtration parameter. The resulting object is a function over filtration scales rather than a scalar. The paper develops plug-in, inverse-probability-weighted, and augmented inverse-probability-weighted estimators, establishes functional weak convergence, and proposes a multiplier-bootstrap test based on

$$
T_n=\sqrt{n}\,
\left\|
\widehat{\psi}_d
\right\|_{\infty}.
$$

Under the null of no topological effect, the population effect is zero; under fixed alternatives, the test is consistent. The paper also gives a stability bound connecting the supremum difference between silhouettes to a Wasserstein distance between persistence diagrams.[^1]

This is directly applicable to your proposal if each point cloud is treated as one structured observation $Y_i$, and the binary indicator

$$
A_i =
\begin{cases}
1,&\text{point cloud from group 1},\\
0,&\text{point cloud from group 0}
\end{cases}
$$

is used as the treatment variable.

However, in an observational two-group comparison, $A_i$ should initially be interpreted as a **group label**, not automatically as a causal treatment. A causal interpretation requires consistency, conditional exchangeability, and positivity. Without these assumptions, the procedure is a functional two-sample test or an associational topological contrast.

### Mathematical framework for TCDA

The second paper gives an important conceptual refinement. It separates:

- the observation space;
- the causal model;
- the topological representation;
- the causal query.

It distinguishes **outcome-level TCDA**, which first maps each individual point cloud to a diagram or functional summary and then averages, from **distribution-level TCDA**, which first considers an interventional probability law and then applies a topological representation to that law.[^2]

For your problem, outcome-level TCDA would be

$$
Z_i=T_{\mathcal F}(Y_i)
=
\Phi\!\left(
\operatorname{Dgm}_d(\mathcal F(Y_i))
\right),
$$

where $\mathcal F$ is the chosen filtration and $\Phi$ could be a silhouette, landscape, Betti curve, persistence image, or another representation. The group contrast is

$$
\Delta_{\mathcal F}
=
\mathbb E[Z_i\mid A_i=1]
-
\mathbb E[Z_i\mid A_i=0].
$$

The null hypothesis is then

$$
H_0^{\text{mean}}:
\Delta_{\mathcal F}=0.
$$

This is not equivalent to equality of the two distributions of persistence diagrams. Two groups may have the same mean silhouette while differing substantially in variance, multimodality, or rare persistent features. The TCDA framework explicitly emphasizes that outcome-level and distribution-level contrasts need not agree because topological transformation and population aggregation generally do not commute.[^2]

### Existing two-sample tests for persistence diagrams

Several earlier methods address the non-causal two-sample problem.

Robinson and Turner proposed a randomization-based test using pairwise distances between persistence diagrams. Its null hypothesis is that two samples of diagrams arise from the same population or process. The test can use bottleneck or Wasserstein-type distances and obtains significance through permutations.[^3]

Krebs and Rademacher study two-sample tests for *relevant differences* in persistence diagrams. Their methods compare Fréchet variances or independent-copy variances under Wasserstein geometry and establish consistency using functional central limit theorems and $U$-statistical arguments. This is especially relevant if your scientific question concerns variability, rather than only a mean topological effect.[^4]

Kernel methods provide another route. Persistence diagrams can be embedded into a reproducing-kernel Hilbert space, after which standard kernel two-sample tests such as MMD can be applied. This tests equality of the embedded distributions and can detect differences beyond the mean. The key limitation is that validity depends on the chosen kernel and on whether the embedding is sufficiently characteristic for the scientific purpose.[^5]

There are also tests based on vectorized persistence diagrams, persistence landscapes, persistence images, silhouettes, persistence intensities, and persistence-survival functions. These are computationally convenient, but every vectorization changes the null hypothesis. For example,

$$
\mathbb E[\text{silhouette}_1]
=
\mathbb E[\text{silhouette}_0]
$$

is weaker than equality of the full diagram distributions.

## 2. A precise formulation of your test

Suppose the data consist of independent structured observations

$$
\{(Y_i,A_i,X_i)\}_{i=1}^{n},
$$

where $Y_i$ is a point cloud, $A_i\in\{0,1\}$ is the group indicator, and $X_i$ contains optional covariates.

For a fixed filtration $\mathcal F$, homology degree $d$, and representation $\Phi$, define

$$
Z_i^{\mathcal F}
=
\Phi\left(
D_{i,d}^{\mathcal F}
\right),
\qquad
D_{i,d}^{\mathcal F}
=
\operatorname{Dgm}_d(\mathcal F(Y_i)).
$$

### Mean-representation test

If $\Phi$ maps into a Banach space, such as $C([t_{\min},t_{\max}])$ for silhouettes or landscapes, define

$$
\Delta_{\mathcal F}(t)
=
\mathbb E[Z_i^{\mathcal F}(t)\mid A_i=1]
-
\mathbb E[Z_i^{\mathcal F}(t)\mid A_i=0].
$$

The global null is

$$
H_0^{\text{mean}}:
\Delta_{\mathcal F}(t)=0
\quad
\text{for all }t\text{ and all selected homology degrees }d.
$$

A natural test statistic is

$$
T_{\mathcal F}
=
\sqrt{n_{\mathrm{eff}}}
\,
\left\|
\widehat{\Delta}_{\mathcal F}
\right\|_{\infty},
$$

where $n_{\mathrm{eff}}$ reflects the two group sizes. Critical values can be obtained using:

- a group-label permutation test in a randomized or exchangeable two-sample design;
- a multiplier bootstrap;
- an influence-function bootstrap if covariate adjustment is used;
- a block or cluster bootstrap if point clouds are dependent.

For an observational study with covariates, the doubly robust estimator suggested by *Topological Causal Effects* is appropriate:

$$
\widehat{\Delta}_{\mathcal F}(t)
=
\frac{1}{n}
\sum_{i=1}^{n}
\left[
\widehat{\mu}_1(t,X_i)
-
\widehat{\mu}_0(t,X_i)
+
\left\{
\frac{A_i}{\widehat e(X_i)}
-
\frac{1-A_i}{1-\widehat e(X_i)}
\right\}
\left\{
Z_i(t)-\widehat{\mu}_{A_i}(t,X_i)
\right\}
\right].
$$

This gives a causal TATE only under the relevant causal assumptions. Otherwise, it is a covariate-adjusted group contrast.

### Full distributional test

If your intended null is genuinely

$$
H_0^{\text{dist}}:
\mathcal L(D^{\mathcal F}\mid A=1)
=
\mathcal L(D^{\mathcal F}\mid A=0),
$$

then testing equality of mean silhouettes is insufficient.

Better candidates include:

- Wasserstein-distance permutation tests on diagrams;
- distance-based randomization tests;
- kernel MMD on persistence diagrams;
- energy tests using diagram distances;
- persistence-image or landscape-based multivariate tests;
- tests comparing Fréchet means and variances.

A useful hierarchy is:

$$
H_0^{\text{dist}}
\Longrightarrow
H_0^{\text{mean}},
$$

but the reverse implication generally does not hold. Thus, a non-significant TATE test should not be interpreted as evidence that the two complete point-cloud populations are topologically identical.

### Important unit-of-analysis issue

The method requires multiple approximately independent units. If you have one large point cloud for group 0 and one large point cloud for group 1, computing one diagram per group does not provide enough replication for a conventional two-sample test.

Possible units include:

- one point cloud per subject;
- one spatial window per subject, with cluster-aware inference;
- independently generated point clouds;
- bootstrap point clouds, provided the resampling scheme represents the population-sampling mechanism;
- repeated scans or repeated observations.

Randomly splitting one point cloud into many pieces can produce pseudo-replication and severely inflated significance unless the sampling design justifies treating those pieces as independent.

## 3. What would be novel?

The basic idea “compute persistence diagrams, measure their distances, and use permutations” is not new. The potential novelty lies in combining four components:

1. a functional TATE-style test for topological representations;
2. a full distributional two-sample test in diagram space;
3. a formal comparison of filtrations;
4. a resolution-selection rule for scalable computation.

A strong contribution could be a **filtration-aware two-sample testing framework** that reports several distinct estimands:

$$
\Delta_{\mathcal F}^{\text{mean}},
\qquad
\Delta_{\mathcal F}^{\text{dist}},
\qquad
\Delta_{\mathcal F}^{\text{local}}(t,d).
$$

Here:

- $\Delta_{\mathcal F}^{\text{mean}}$ measures the difference in expected summaries;
- $\Delta_{\mathcal F}^{\text{dist}}$ measures the difference between diagram distributions;
- $\Delta_{\mathcal F}^{\text{local}}(t,d)$ identifies the homology degree and filtration scales responsible for the difference.

The contribution would be strongest if it included finite-sample calibration, power comparisons, robustness to point-cloud size, dependence handling, and computational complexity analysis.

## 4. Comparing Vietoris–Rips and cubical filtrations

Your second idea is also worthwhile, but the target must be defined carefully.

Let

$$
D_{i}^{\mathrm{VR}}
=
\operatorname{Dgm}_d(\mathcal F_{\mathrm{VR}}(Y_i))
$$

and

$$
D_{i,h}^{\mathrm{Cub}}
=
\operatorname{Dgm}_d(\mathcal F_{\mathrm{Cub},h}(Y_i)),
$$

where $h$ is the cubical grid size. Apply a common representation $\Phi$, or compare diagrams directly:

$$
R_i(h)
=
d_{\mathsf D}
\left(
D_i^{\mathrm{VR}},
D_{i,h}^{\mathrm{Cub}}
\right).
$$

A paired test can target

$$
H_0(h):
\mathbb E[R_i(h)]=0,
$$

but exact equality is usually unrealistic because the two filtrations are different constructions. A more useful formulation is an **equivalence or relevant-difference test**:

$$
H_0^{\text{non-equivalence}}(h):
\mathbb E[R_i(h)]>\varepsilon,
$$

versus

$$
H_1^{\text{equivalence}}(h):
\mathbb E[R_i(h)]\leq\varepsilon,
$$

where $\varepsilon$ is a scientifically or computationally chosen tolerance.

This is preferable to “fail to reject equality.” A non-significant test does not establish that the methods are equivalent. Equivalence requires a tolerance and a confidence interval or confidence bound showing that the discrepancy is sufficiently small.

A paired bootstrap or paired permutation procedure is appropriate because the two diagrams are computed from the same point cloud. Pairing usually provides substantially more power than treating the VR and cubical diagrams as independent samples.

You could also compare the full representations:

$$
\Delta_h(t)
=
\mathbb E[
\Phi(D_i^{\mathrm{VR}})(t)
-
\Phi(D_{i,h}^{\mathrm{Cub}})(t)
],
$$

and test

$$
\left\|\Delta_h\right\|_{\infty}\leq \varepsilon.
$$

This would identify not only whether the methods differ, but also the scales at which they differ.

### Caveat about convergence

The statement that cubical and Vietoris–Rips filtrations “converge when cubes are sufficiently small” needs qualification. They are not automatically the same filtration, and decreasing the grid size does not by itself guarantee convergence to the same persistence module. Convergence requires a common underlying object, compatible filtration functions, and a theorem controlling the approximation error.

The relevant stability principle is that perturbations of filtration functions can induce bounded perturbations of persistence diagrams, typically through inequalities of the form

$$
d_B(D(f),D(g))
\leq
\|f-g\|_{\infty}.
$$

[^6]

For point-cloud discretization, the approximation error may involve grid quantization, Hausdorff error, density estimation error, boundary effects, and the difference between the metric used by the cubical construction and the metric underlying the VR filtration. Recent work gives explicit bounds for certain comparisons involving enrichment, sparsification, grid alignment, and cubical constructions, but these bounds are filtration- and homology-specific rather than a universal VR-to-cubical equivalence theorem.[^7]

Therefore, your paper should not assume convergence. It should either:

- prove a bound for the specific VR and cubical constructions you use; or
- define statistical equivalence empirically through a tolerance $\varepsilon$, while separately discussing theoretical approximation error.


## 5. Selecting the largest cube size

A principled workflow would be:

1. Choose a validation sample of independent point clouds.
2. Compute $D_i^{\mathrm{VR}}$ once for each validation cloud.
3. For candidate resolutions $h_1,\ldots,h_K$, compute $D_{i,h_k}^{\mathrm{Cub}}$.
4. Calculate paired discrepancies $R_i(h_k)$, preferably using Wasserstein distance or a stable functional representation.
5. Construct simultaneous confidence intervals or upper confidence bounds over $h$.
6. Select the largest $h$ satisfying the equivalence condition.

For example, select

$$
\widehat h_{\max}
=
\max
\left\{
h_k:
U_{1-\alpha}\bigl(\mathbb E[R(h_k)]\bigr)
\leq \varepsilon
\right\},
$$

where $U_{1-\alpha}$ is an upper confidence bound.

The selection should be based on a validation set and then locked before evaluating the final scientific hypothesis. Otherwise, choosing $h$ because it produces the most favorable group comparison can cause selection bias and inflate the type-I error.

You should also account for multiple candidate resolutions. A simultaneous bootstrap band over $h$, or a multiplicity correction, is preferable to testing each grid size independently. Furthermore, the discrepancy need not be monotone in $h$, so selecting the largest acceptable resolution should not rely only on a binary search unless monotonicity has been established theoretically or empirically.

### Recommended final methodology

A publishable version of the project could contain two procedures:

**Procedure A: Topological two-sample test**

$$
H_0:
\mathcal L(D^{\mathcal F}\mid A=0)
=
\mathcal L(D^{\mathcal F}\mid A=1),
$$

using a Wasserstein-distance permutation/MMD test, supplemented by a functional TATE test based on silhouettes or landscapes.

**Procedure B: Filtration-equivalence calibration**

$$
H_0(h):
\mathbb E[
d_{\mathsf D}(D^{\mathrm{VR}},D_{h}^{\mathrm{Cub}})
]
>
\varepsilon
$$

against equivalence within tolerance. Use paired bootstrap inference and choose the largest validated $h$ whose upper confidence bound is below $\varepsilon$.

This separation is important: Procedure A tests a scientific difference between groups, whereas Procedure B calibrates whether a computational approximation is sufficiently faithful to a reference filtration. The TCDA papers supply the causal and functional-inference foundation, while the persistence-diagram two-sample literature supplies the distributional testing tools.[^1][^3][^4]

Overall, the idea is technically feasible and potentially valuable. The central conceptual decision is whether you want to test equality of **expected topological summaries**, equality of **full persistence-diagram distributions**, or **approximate equivalence between filtrations**. These should be presented as separate hypotheses rather than as interchangeable versions of “topological difference equals zero.”

[^1]: https://arxiv.org/html/2603.02289v1

[^2]: https://arxiv.org/html/2607.28161v1

[^3]: https://arxiv.org/abs/1310.7467

[^4]: https://arxiv.org/abs/2401.10349

[^5]: http://wwwx.cs.unc.edu/~mn/sites/default/files/kwitt-statistical-topological-data-analysis-a-kernel-perspective.pdf

[^6]: https://arxiv.org/pdf/2006.16824v5.pdf

[^7]: https://arxiv.org/abs/2511.07093

[^8]: https://iclr.cc/virtual/2026/poster/10008398

[^9]: https://arxiv.org/abs/2006.10012

[^10]: https://openaccess.thecvf.com/content_cvpr_2016_workshops/w23/papers/Anirudh_A_Riemannian_Framework_CVPR_2016_paper.pdf

[^11]: https://pmc.ncbi.nlm.nih.gov/articles/PMC5692542/

[^12]: https://ui.adsabs.harvard.edu/abs/2026arXiv260302289K/abstract

[^13]: https://arxiv.org/abs/1901.03048

[^14]: https://arxiv.org/abs/2304.03382

[^15]: https://arxiv.org/pdf/2603.14169.pdf

[^16]: https://arxiv.org/pdf/2606.11911v1.pdf

[^17]: https://arxiv.org/abs/2412.17482

[^18]: https://arxiv.org/pdf/2607.28161.pdf

[^19]: https://pmc.ncbi.nlm.nih.gov/articles/PMC5763311/

[^20]: https://cran.r-project.org/web/packages/TDAstats/vignettes/inference.html

[^21]: https://arxiv.org/html/2404.18194v1

[^22]: https://openreview.net/pdf?id=dYaos1ITw4

[^23]: https://www.mdpi.com/1099-4300/25/11/1509/pdf?version=1698835989

[^24]: https://www.aimsciences.org/article/doi/10.3934/fods.2022014

[^25]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10669999/

[^26]: https://arxiv.org/abs/2607.28161

[^27]: https://arxiv.org/abs/2001.06058

[^28]: https://arxiv.org/abs/2006.05466

[^29]: https://arxiv.org/pdf/2607.20893.pdf

[^30]: https://arxiv.org/abs/2003.01352

[^31]: https://cran.r-project.org/web/packages/inphr/inphr.pdf

[^32]: https://arxiv.org/abs/2207.03926

[^33]: https://jocg.org/index.php/jocg/article/download/2982/2688

[^34]: https://arxivtldr.org/abs/2607.28161

[^35]: https://arxiv.org/abs/2305.08999

[^36]: https://arxiv.deeppaper.ai/papers/2607.28161v1

[^37]: https://arxiv.org/abs/1904.07768

[^38]: https://r-consortium.org/posts/statistical-inference-for-persistence-diagrams/

[^39]: https://aipapers.ai/paper/26917756

[^40]: https://arxiv.org/abs/2301.07191

[^41]: https://arxiv.org/pdf/2105.05151v1.pdf

[^42]: https://arxiv.org/abs/2111.06502

[^43]: https://studenttheses.uu.nl/bitstream/handle/20.500.12932/36446/thesis.pdf?sequence=1

[^44]: https://arxiv.org/abs/1902.05911

[^45]: https://arxiv.org/abs/2307.16333

[^46]: https://www.epfl.ch/labs/hessbellwald-lab/wp-content/uploads/2020/02/Poster_geocow_AdlieGarin.pdf

[^47]: https://arxiv.org/abs/2209.02791

[^48]: https://en.wikipedia.org/wiki/Persistent_homology

[^49]: https://arxiv.org/abs/2411.08201

[^50]: https://par.nsf.gov/servlets/purl/10358772

[^51]: https://arxiv.org/abs/2001.07588

[^52]: https://arxiv.org/pdf/1602.03760.pdf

[^53]: https://arxiv.org/abs/1810.10144

[^54]: https://publikationen.sulb.uni-saarland.de/bitstream/20.500.11880/26911/1/aruni_choudhary_camera_ready_thesis.pdf

[^55]: https://arxiv.org/html/2408.11450v1

[^56]: https://pure.mpg.de/rest/items/item_3024390_2/component/file_3024391/content

[^57]: https://arxiv.org/abs/2309.14211

[^58]: https://en.papernotes.org/ICLR2026/causal_inference/topological_causal_effects/

[^59]: https://persim.scikit-tda.org/en/latest/notebooks/Persistence barcode measure.html

[^60]: https://people.willamette.edu/~ijohnson/courses/19-20review_files/research/JohnsonPurdy_ExtendingHypothTestwPersisHom.pdf

[^61]: https://pages.stat.wisc.edu/~mchung/papers/wang.2022.ICASSP.pdf

[^62]: https://mlanthology.org/iclr/2026/kim2026iclr-topological/

[^63]: https://data.math.au.dk/publications/csgb/2016/math-csgb-2016-15.pdf

[^64]: https://chatpaper.com/paper/315869

[^65]: https://www.youtube.com/watch?v=x9ihVVV-bzc

[^66]: https://mrzv.org/publications/persistent-homology-handbook-dcg/handbook-dcg/

[^67]: https://pub.ista.ac.at/~edels/Papers/2012-11-PHTheoryPractice.pdf

[^68]: https://morfismos.cinvestav.mx/sites/default/files/Upload/vol21-n1-2.pdf

[^69]: https://dioscuri-tda.org/documents/paris_2021_data_science_school/slides/TDA_Tutorial.pdf

