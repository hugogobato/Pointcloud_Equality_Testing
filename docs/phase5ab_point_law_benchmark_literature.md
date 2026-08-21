# Phase 5AB point-law benchmark literature list

## Coverage note

The benchmark has strong established foundations for MMD, energy distance, graph-based multivariate two-sample tests, and exact permutation calibration. The method-specific fixed-disjoint-block construction is less standard: the project combines a fixed-size raw-bag representation, a characteristic kernel on that representation, and block-label permutation. That interface should be described as a proposed construction built from known ingredients, not as an entirely new theory of two-sample testing. The literature list is deliberately small and is intended to support implementation and claim checking, not to serve as an exhaustive survey.

## Group A: MMD, characteristic kernels, and distribution embeddings

Serves WP-B0, WP-B1, WP-B4, and the RawBlockMMD identification audit.

| # | Citation | Need it serves | Open-access URL | DOI / identifier | Access | Metadata confidence | Suggested filename |
|---|---|---|---|---|---|---|---|
| A1 | Gretton, Borgwardt, Rasch, Schölkopf, and Smola (2012), “A Kernel Two-Sample Test,” *JMLR* 13, 723--773 | Standard MMD statistic, permutation and asymptotic calibration, characteristic-kernel two-sample testing | [JMLR article](https://www.jmlr.org/papers/v13/gretton12a.html), [PDF](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf) | JMLR 13(25) | Open access | VERIFIED | `A_Gretton2012_MMD.pdf` |
| A2 | Simon-Gabriel and Schölkopf (2018), “Kernel Distribution Embeddings: Universal Kernels, Characteristic Kernels and Kernel Metrics on Distributions,” *JMLR* 19 | Conditions under which kernel embeddings and kernel metrics identify probability laws | [JMLR article](https://www.jmlr.org/papers/v19/16-291.html), [PDF](https://www.jmlr.org/papers/volume19/16-291/16-291.pdf) | JMLR 19 | Open access | VERIFIED | `A_SimonGabriel2018_embeddings.pdf` |
| A3 | Szabó, Gretton, Póczos, and Sriperumbudur (2015), “Two-stage Sampled Learning Theory on Distributions,” *PMLR* 38 | Embedding distributions or sets from finite samples and the two-stage sampling distinction | [PMLR article](https://proceedings.mlr.press/v38/szabo15.html), [PDF](https://proceedings.mlr.press/v38/szabo15.pdf) | PMLR 38 | Open access | VERIFIED | `A_Szabo2015_two_stage.pdf` |
| A4 | Buathong, Ginsbourger, and Krityakierne (2020), “Kernels over Sets of Finite Sets using RKHS Embeddings,” *PMLR* 108 | Related finite-set kernels and radial kernels applied to RKHS set embeddings | [PMLR article](https://proceedings.mlr.press/v108/buathong20a.html), [PDF](https://proceedings.mlr.press/v108/buathong20a/buathong20a.pdf) | PMLR 108 | Open access | VERIFIED | `A_Buathong2020_finite_sets.pdf` |

The agent must check whether A3 or A4 already supplies the exact characteristicness result needed for the empirical mean-embedding image of fixed-size unordered bags. If not, label the project-specific extension as `adapt`, not `provable from citation alone`.

## Group B: energy and distance-based two-sample testing

Serves WP-B1 and the comparison interpretation for EnergyDistance.

| # | Citation | Need it serves | Open-access URL | DOI / identifier | Access | Metadata confidence | Suggested filename |
|---|---|---|---|---|---|---|---|
| B1 | Székely and Rizzo (2013), “Energy Statistics: A Class of Statistics Based on Distances,” *Journal of Statistical Planning and Inference* 143, 1249--1272 | Energy distance, distance-based two-sample statistics, and permutation implementation | [ScienceDirect article](https://www.sciencedirect.com/science/article/pii/S0378375813000633) | `10.1016/j.jspi.2013.03.018` | Abstract/publisher page | VERIFIED | `B_Szekely2013_energy_statistics.pdf` |
| B2 | Sejdinovic, Sriperumbudur, Gretton, and Fukumizu (2013), “Equivalence of Distance-Based and RKHS-Based Statistics in Hypothesis Testing,” *Annals of Statistics* | Relationship between energy/distance tests and MMD, including identifying classes | [arXiv record](https://arxiv.org/abs/1207.6076) | arXiv:1207.6076 | Open preprint | VERIFIED | `B_Sejdinovic2013_distance_RKHS.pdf` |

The implementation must still state whether it uses a biased V-statistic or an unbiased U-statistic. The source does not remove the need to document the exact finite-sample statistic used in the code.

## Group C: graph and matching two-sample tests

Serves WP-B2 and the graph-test target and consistency audit.

| # | Citation | Need it serves | Open-access URL | DOI / identifier | Access | Metadata confidence | Suggested filename |
|---|---|---|---|---|---|---|---|
| C1 | Friedman and Rafsky (1979), “Multivariate Generalizations of the Wald--Wolfowitz and Smirnov Two-Sample Tests,” *The Annals of Statistics* 7, 697--717 | MST/run-based multivariate two-sample test and null results | [DOI landing page](https://doi.org/10.1214/aos/1176344722) | `10.1214/aos/1176344722` | DOI/abstract page | VERIFIED | `C_Friedman1979_Rafsky_MST.pdf` |
| C2 | Schilling (1986), “Multivariate Two-Sample Tests Based on Nearest Neighbors,” *JASA* 81, 799--806 | k-nearest-neighbour two-sample statistic, weighting choices, and consistency | [DOI landing page](https://doi.org/10.2307/2289012), [author PDF](https://www.csun.edu/~hcmth031/mtstbonn.pdf) | `10.2307/2289012` | Author PDF open; DOI page | VERIFIED | `C_Schilling1986_kNN.pdf` |
| C3 | Rosenbaum (2005), “An Exact Distribution-Free Test Comparing Two Multivariate Distributions Based on Adjacency,” *JRSS B* 67, 515--530 | Minimum-distance matching, cross-match statistic, and exact distribution-free calibration | [Wiley article](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9868.2005.00513.x), [PDF](https://watermark02.silverchair.com/jrsssb_67_4_515.pdf) | `10.1111/j.1467-9868.2005.00513.x` | Abstract page; PDF availability may vary | VERIFIED | `C_Rosenbaum2005_crossmatch.pdf` |

The graph methods should be reported as valid permutation procedures under the iid null, with consistency conditions inherited from these sources. They should not be described as characteristic in the MMD sense.

## Group D: Wasserstein and related distance tests

Serves WP-B3 and the OT sensitivity panel.

| # | Citation | Need it serves | Open-access URL | DOI / identifier | Access | Metadata confidence | Suggested filename |
|---|---|---|---|---|---|---|---|
| D1 | Ramdas, Garcia, and Cuturi (2015), “On Wasserstein Two Sample Testing and Related Families of Nonparametric Tests” | Wasserstein two-sample testing and relationships among Wasserstein, energy, and kernel methods | [arXiv record](https://arxiv.org/abs/1509.02237) | arXiv:1509.02237 | Open preprint | VERIFIED | `D_Ramdas2015_Wasserstein_testing.pdf` |

The project should not claim that a finite-projection or regularised approximation is universally identifying without checking the exact representation. This group is primarily for an interpretable secondary baseline.

## Group E: classifier two-sample tests

Serves WP-B3 and the classifier calibration audit.

| # | Citation | Need it serves | Open-access URL | DOI / identifier | Access | Metadata confidence | Suggested filename |
|---|---|---|---|---|---|---|---|
| E1 | Lopez-Paz and Oquab (2017), “Revisiting Classifier Two-Sample Tests,” ICLR | Classifier-based two-sample testing, held-out accuracy, learned representations, and interpretation | [arXiv record](https://arxiv.org/abs/1610.06545), [reference implementation](https://github.com/lopezpaz/classifier_tests) | arXiv:1610.06545 | Open preprint and code | VERIFIED | `E_LopezPaz2017_C2ST.pdf` |

The benchmark must state the classifier class and sample-splitting rule. A finite classifier is a restricted test and is not automatically a universal equality test.

## Group F: fixed-size barcode laws, subsampling, and block validity

Serves the target separation between SC-B and RawBlockMMD and WP-B0, WP-B4, and WP-B5.

| # | Citation | Need it serves | Open-access URL | DOI / identifier | Access | Metadata confidence | Suggested filename |
|---|---|---|---|---|---|---|---|
| F1 | Blumberg, Gal, Mandell, and Pancia (2014), “Robust Statistics, Hypothesis Testing, and Confidence Intervals for Persistent Homology on Metric Measure Spaces,” *Foundations of Computational Mathematics* 14, 745--789 | Fixed-size barcode-law construction and its metric-measure interpretation | [arXiv record](https://arxiv.org/abs/1206.4581), [author PDF](https://www.math.columbia.edu/~thaddeus/blumberg/12.pdf), [DOI landing page](https://doi.org/10.1007/s10208-014-9201-4) | `10.1007/s10208-014-9201-4` | Open preprint and author PDF | VERIFIED | `F_Blumberg2014_barcode_law.pdf` |
| F2 | Chazal, Fasy, Lecci, Michel, Rinaldo, and Wasserman (2015), “Subsampling Methods for Persistent Homology,” *PMLR* 37 | Risk and stability considerations for persistence estimators obtained by subsampling | [PMLR article](https://proceedings.mlr.press/v37/chazal15.html), [PDF](https://proceedings.mlr.press/v37/chazal15.pdf) | PMLR 37:2143--2151 | Open access | VERIFIED | `F_Chazal2015_subsampling_PH.pdf` |

These sources support the barcode representation and its assumptions. They do not justify calling overlapping subclouds independent observations. The project-specific block permutation argument remains a separate result.

## Group G: ecological application comparators, separate from the generic point-law benchmark

Serves the later ecology application only. These methods must not be ranked as generic tests of `P0=P1` unless their ecological target is explicitly redefined.

| # | Citation | Need it serves | Open-access URL | DOI / identifier | Access | Metadata confidence | Suggested filename |
|---|---|---|---|---|---|---|---|
| G1 | Warren, Glor, and Turelli (2008), “Environmental Niche Equivalency versus Conservatism: Quantitative Approaches to Niche Evolution,” *Evolution* 62, 2868--2883 | Niche-equivalency permutation comparator for the ecology panel | [Author PDF](https://www.danwarren.net/Warren%20et%20al%202008%20Niche%20conservatism.pdf), [Wiley record](https://doi.org/10.1111/j.1558-5646.2008.00482.x) | `10.1111/j.1558-5646.2008.00482.x` | Open author PDF | VERIFIED | `G_Warren2008_niche_equivalency.pdf` |
| G2 | Broennimann et al. (2012), “Measuring Ecological Niche Overlap from Occurrence and Spatial Environmental Data,” *Global Ecology and Biogeography* 21, 481--497 | PCA-env, Schoener's D, Hellinger I, and background-environment overlap | [Wiley record](https://onlinelibrary.wiley.com/doi/10.1111/j.1466-8238.2011.00698.x), [PDF](https://bprc-ecoinformatics.github.io/assets/scripts/2_NicheAnalysis/GlobalEcolBiogeogr_Broennimann_2012.pdf) | `10.1111/j.1466-8238.2011.00698.x` | Open PDF available | VERIFIED | `G_Broennimann2012_PCA_env.pdf` |
| G3 | Blonder (2014), “The n-dimensional hypervolume,” *Global Ecology and Biogeography* 23, 595--609 | Hypervolume representation and overlap context | [DOI landing page](https://doi.org/10.1111/geb.12146) | `10.1111/geb.12146` | Open publisher record | VERIFIED | `G_Blonder2014_hypervolume.pdf` |

## Known but no link found

No method required by the current plan is placed in this section. If an agent discovers a method with uncertain metadata, it must be added here until a resolvable source is verified.

## Acquisition snippet

The following is a suggested acquisition command. It is intentionally not executed by this plan.

```bash
mkdir -p literature/phase5ab_pointlaw
curl -L -o literature/phase5ab_pointlaw/A_Gretton2012_MMD.pdf "https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf"
curl -L -o literature/phase5ab_pointlaw/A_SimonGabriel2018_embeddings.pdf "https://www.jmlr.org/papers/volume19/16-291/16-291.pdf"
curl -L -o literature/phase5ab_pointlaw/A_Szabo2015_two_stage.pdf "https://proceedings.mlr.press/v38/szabo15.pdf"
curl -L -o literature/phase5ab_pointlaw/A_Buathong2020_finite_sets.pdf "https://proceedings.mlr.press/v108/buathong20a/buathong20a.pdf"
curl -L -o literature/phase5ab_pointlaw/B_Sejdinovic2013_distance_RKHS.pdf "https://arxiv.org/pdf/1207.6076"
curl -L -o literature/phase5ab_pointlaw/C_Schilling1986_kNN.pdf "https://www.csun.edu/~hcmth031/mtstbonn.pdf"
curl -L -o literature/phase5ab_pointlaw/C_Rosenbaum2005_crossmatch.pdf "https://watermark02.silverchair.com/jrsssb_67_4_515.pdf"
curl -L -o literature/phase5ab_pointlaw/D_Ramdas2015_Wasserstein_testing.pdf "https://arxiv.org/pdf/1509.02237"
curl -L -o literature/phase5ab_pointlaw/E_LopezPaz2017_C2ST.pdf "https://arxiv.org/pdf/1610.06545"
curl -L -o literature/phase5ab_pointlaw/F_Chazal2015_subsampling_PH.pdf "https://proceedings.mlr.press/v37/chazal15.pdf"
curl -L -o literature/phase5ab_pointlaw/G_Warren2008_niche_equivalency.pdf "https://www.danwarren.net/Warren%20et%20al%202008%20Niche%20conservatism.pdf"
curl -L -o literature/phase5ab_pointlaw/G_Broennimann2012_PCA_env.pdf "https://bprc-ecoinformatics.github.io/assets/scripts/2_NicheAnalysis/GlobalEcolBiogeogr_Broennimann_2012.pdf"
```

Several verified sources are listed in the tables but omitted from the download snippet because only a publisher or DOI page was verified. The agent should download those manually if access is available. Do not treat a failed download as evidence that the citation is false.

## Source cheatsheet requirement

After acquisition, create `docs/phase5ab_point_law_source_cheatsheet.md` containing, for every source used in a claim, the exact theorem, proposition, definition, or algorithm section and the assumptions that transfer to this project. The cheatsheet must explicitly distinguish results that apply to individual iid points from results about distributions of sets, persistence diagrams, or dependent spatial processes.
