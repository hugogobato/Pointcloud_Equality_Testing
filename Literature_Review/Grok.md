# Literature Review: Statistical Tests for Differences Between Point-Cloud Datasets via Filtrations and Persistence Diagrams (with Links to Topological Causal Effects)

## 1. Core Research Idea and Framing via Topological Causal Effects

The proposed research develops a statistical test for whether two point-cloud datasets (or single observations) differ significantly in topological structure under a chosen filtration and persistence-diagram (PD) representation. Framing the problem by introducing a binary indicator covariate \(A \in \{0,1\}\) that labels the two groups and testing whether the topological average treatment effect (TATE) is zero aligns closely with recent work on topological causal inference.

Kim and Lee (2026) define the TATE as a functional estimand based on power-weighted silhouette functions of PDs:

\[
\psi_d(t) := \mathbb{E}\bigl[\phi(t;D_{i,d}^1)\bigr] - \mathbb{E}\bigl[\phi(t;D_{i,d}^0)\bigr],
\]

where \(\phi(\cdot;D)\) is the power-weighted silhouette of the \(d\)-dimensional PD arising from a filtration of the (potential) outcome. They construct a doubly-robust (AIPW-style) nonparametric estimator, prove functional weak convergence in \(\ell^\infty\), and obtain an asymptotically valid level-\(\alpha\) test of the null \(H_0: \psi_d \equiv 0\) (no topological effect) via a multiplier bootstrap on the supremum norm. Stability bounds relating the silhouette difference to the Wasserstein distance between diagrams are also derived. Empirical illustrations include point-cloud (ORBIT), molecular-graph and image data (Kim & Lee, 2026).

Souto and Diamantis (2026) embed this construction in a broader four-layer architecture (observation space, causal-model class, topological representation, causal query). They distinguish *outcome-level* TCDA (apply topology to individual potential outcomes then average) from *distribution-level* TCDA (apply topology to the interventional laws \(P_a = \mathcal{L}(Y^a)\)). The silhouette-based TATE of Kim and Lee is recovered as a concrete instance of the outcome-level estimand in a Banach-space setting. Identification proceeds via the usual \(g\)-formula under consistency, conditional exchangeability and positivity; doubly-robust representations and stability-transfer bounds are given for general Banach-valued summaries. The framework also discusses target-specific “topological ignorability,” which can be weaker than classical conditional exchangeability when the topological map is non-injective (Souto & Diamantis, 2026).

Taken together, these two papers supply a ready-made, rigorously justified route for the primary idea: treat the two point-cloud collections as the two arms of a binary treatment, estimate the silhouette (or other functional) contrast, and test whether it is identically zero. The same machinery immediately yields confidence bands and power analysis.

## 2. Classical and Contemporary Two-Sample Tests for Persistence Diagrams

Independent of the causal framing, a substantial literature already addresses the two-sample problem for PDs or their functional summaries.

- **Permutation / randomization tests on diagram distances.**  
  Robinson and Turner (2017) introduce a loss-function based on pairwise \(p\)-Wasserstein (or bottleneck) distances within and between two samples of diagrams; a permutation test yields a \(p\)-value for the null that the two collections arise from the same process. Cericola et al. (2018) extend the procedure to three or more groups. These tests are distribution-free, easy to implement, and have been applied to point-cloud shape data, silhouette data and fMRI concurrence filtrations. Their power depends on the chosen distance and on the separation of the underlying geometric measures (Robinson & Turner, 2017; Cericola et al., 2018).

- **Functional summaries + classical or permutation tests.**  
  Persistence landscapes (Bubenik, 2015) and silhouettes (Chazal et al., 2014) map diagrams into \(L^p\) function spaces. Average landscapes converge weakly to Gaussian processes; bootstrap confidence bands are available. Two-sample \(z\)-tests, Hotelling’s \(T^2\) tests on vectors of landscape functionals, and permutation tests on the \(L^p\) distance between mean landscapes are standard (Bubenik, 2015; Chazal et al., 2014). Moon and Lazar (2020) propose a two-stage procedure for persistence images: a filtering step that discards uninformative pixels followed by FDR-controlled multiple testing. Krebs and Rademacher (2024) develop tests for *relevant* differences (i.e., Fréchet variances differing by more than a tolerance \(\Delta\)) under weak dependence, again using Wasserstein geometry on the space of diagrams (Moon & Lazar, 2020; Krebs & Rademacher, 2024).

- **Other vectorizations and divergences.**  
  Kernel two-sample tests in RKHS, survival-function representations of persistence (e.g., STRAND), and persistence-landscape + Jensen–Shannon divergences have also been proposed. All of these can be paired with permutation or asymptotic tests (Nakayama, 2025).

Collectively these methods already allow one to decide whether two collections of PDs (hence of point clouds under a fixed filtration) are statistically distinguishable. The causal/TATE route of Kim–Lee and Souto–Diamantis adds identification under confounding, efficient estimation, and a functional null that is natural when the scientific question is “does treatment change topology?” rather than the pure two-sample null.

## 3. Side Idea: Comparing Filtrations and Choosing Cube Size for Large Point Clouds

Vietoris–Rips (VR) filtrations are the most common for point clouds but become computationally prohibitive on large data. Cubical complexes (or approximations of VR by cubical towers) are substantially cheaper, especially on grids or after spatial quantization. Theoretical approximation results show that scaled cubical towers can 2-approximate (or \(2d^{1/4}\)-approximate) the \(L^\infty\)-Rips or Euclidean Rips modules. When the cube diameter tends to zero the persistence modules converge in the interleaving (or bottleneck) distance; quantitative rates depend on the density of the underlying measure and the Lipschitz constants of the filtration functions.

A statistical test that declares two filtrations “equivalent” (their induced PD distributions, or the distributions of any stable functional summary, are indistinguishable) therefore supplies a practical stopping rule: on a validation subsample one can increase cube size until the test against the reference VR (or \(\alpha\)-complex) diagram fails to reject, thereby obtaining the coarsest admissible cubical filtration for the full data set. Because the same functional summaries and permutation/AIPW machinery already developed for the two-sample problem apply directly to “VR versus cubical” pairs, the side idea is a natural next step rather than a wholly separate research programme.

## 4. Gaps, Opportunities and Suggested Next Steps

- **Single-observation vs. sample-of-diagrams.** Most existing tests assume multiple independent diagrams per group. Extending the TATE framework (or a bootstrap/permutation procedure) to the extreme case of two single large point clouds remains largely open; concentration inequalities for empirical PDs under sampling from a common measure would be needed.
- **Choice of summary and power.** Silhouettes, landscapes, persistence images and Betti curves emphasize different features. Systematic power studies under controlled topological alternatives would guide practitioners.
- **Computational–statistical trade-off.** Formal rates that relate cube diameter, approximation error in the interleaving distance, and the power of the subsequent two-sample test are missing; deriving them would turn the “largest admissible cube” heuristic into a theoretically justified procedure.
- **Confounding and observational data.** When the two point-cloud collections are observational rather than randomized, the full TCDA identification theory becomes essential; topological ignorability may allow weaker assumptions than classical exchangeability for non-injective summaries.

In summary, the two cited arXiv papers already furnish a mature, asymptotically valid testing framework for the primary research idea via the TATE null. The classical TDA two-sample literature supplies complementary, often simpler, non-causal procedures. The filtration-comparison side idea follows almost immediately once a reliable test for diagram (or summary) equality is in hand, and is especially attractive for scaling TDA to very large point clouds.

## References

Bubenik, P. (2015). Statistical topological data analysis using persistence landscapes. *Journal of Machine Learning Research*, 16, 77–102.

Cericola, C., Johnson, I., Kiers, J., Krock, M., Purdy, J., & Torrence, J. (2018). Extending hypothesis testing with persistence homology to three or more groups. *Involve*, 11(1), 27–51.

Chazal, F., Fasy, B. T., Lecci, F., Rinaldo, A., & Wasserman, L. (2014). Stochastic convergence of persistence landscapes and silhouettes. *Proceedings of the Thirtieth Annual Symposium on Computational Geometry*, 474–483. (Also arXiv:1312.0308)

Kim, K., & Lee, H. (2026). Topological causal effects. *arXiv preprint arXiv:2603.02289*. (Also in *Proceedings of the Fourteenth International Conference on Learning Representations (ICLR 2026)*)

Krebs, J., & Rademacher, D. (2024). Two-sample tests for relevant differences in persistence diagrams. *arXiv preprint arXiv:2401.10349*.

Moon, C., & Lazar, N. A. (2020). Hypothesis testing for shapes using vectorized persistence diagrams. *arXiv preprint arXiv:2006.05466*.

Nakayama, T. (2025). Persistence-based statistics for detecting structural changes in high-dimensional point clouds. *arXiv preprint arXiv:2511.00938*.

Robinson, A., & Turner, K. (2017). Hypothesis testing for topological data analysis. *Journal of Applied and Computational Topology*, 1, 241–261. (Originally arXiv:1310.7467)

Souto, H. G., & Diamantis, I. (2026). A mathematical framework for topological causal data analysis. *arXiv preprint arXiv:2607.28161*.