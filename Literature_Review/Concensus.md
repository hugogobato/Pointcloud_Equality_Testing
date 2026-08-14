# A **Topological Two-Sample** Test Roadmap

The literature supports **your core idea**: a formal test of whether two point-cloud samples differ topologically under a fixed filtration and persistence representation is feasible, and it sits at the intersection of existing persistence-diagram hypothesis testing and the newer causal framing of topological effects  (Kim & Lee, 2026; Robinson & Turner, 2013; Moon & Lazar, 2020). The same literature also makes clear that the null hypothesis is only scientifically well defined after fixing the filtration and vectorization, because those choices determine the target topological feature being compared  (Saki & Faghihi, 2026; Faghihi, 2026).

## Causal Framing

A causal-style formulation is plausible if group membership is encoded as a binary indicator and the estimand is defined as a difference in topological summaries, rather than a difference in coordinate means  (Kim & Lee, 2026; Saki & Faghihi, 2026; Faghihi, 2026). This is attractive precisely because mean effects can vanish when topology changes, such as when a unimodal distribution becomes bimodal without changing its average  (Saki & Faghihi, 2026; Faghihi, 2026).  

The strongest caveat is that **filtration choice is part of the estimand**: a Vietoris–Rips filtration on raw points, a density-level filtration, and a cubical filtration on a gridded field target different scientific objects, so a “topological difference” is never filtration-free  (Saki & Faghihi, 2026; Faghihi, 2026; Wang & Wei, 2016). Likewise, replacing landscapes with images, silhouettes, Betti curves, or Euler summaries changes the inferential target rather than merely the numerical encoding  (Saki & Faghihi, 2026; Faghihi, 2026; Debnath et al., 2025; Saki & Faghihi, 2026).

- **Topological causal effects** define intervention effects directly on persistence-based summaries and provide nonparametric inference for a null of no topological effect  (Kim & Lee, 2026).
- Persistent homology summaries can be estimated from outcome clouds, density level sets, or distance-to-measure filtrations  (Saki & Faghihi, 2026).
- Overlap and weighting diagnostics matter if the binary group indicator is treated causally rather than as a simple two-sample label  (Faghihi, 2026).

## Existing Testing Methods

The direct predecessor of your proposal is the randomization-style two-sample test on persistence diagrams, which was designed to assess whether two samples arise from the same process using a loss built from pairwise diagram distances  (Robinson & Turner, 2013). That line was later extended from two groups to three or more groups and validated in simulations with varying sample sizes and measurement error  (Cericola et al., 2016).  

A second branch tests **vectorized diagrams** instead of diagrams in their native metric space, using persistence images or related summaries with filtering and multiple testing control  (Moon & Lazar, 2020). These methods are cheaper and more interpretable, but the topological causal literature explicitly warns that vectorized-summary tests generally do not by themselves yield valid inference in the metric space of persistence diagrams  (Moon & Lazar, 2020; Kim & Lee, 2026).

- **Permutation tests** already exist for persistence landscapes and sampled point-cloud studies  (Kovačev-Nikolić, 2012).
- A newer framework proposes **feature-level significance** via a universal null law for normalized persistence diagrams  (Bobrowski & Skraba, 2023).
- Weight and threshold tuning remain unresolved in vectorized-diagram testing, which creates researcher-degree-of-freedom concerns  (Moon & Lazar, 2020).

## Filtration Comparison

The side idea—testing whether two filtrations give statistically similar persistence outputs—is also well motivated. Vietoris–Rips is widely used but computationally expensive, with worst-case exponential growth and substantial time and memory costs  (Leitão, 2026; Koyama et al., 2024; Bauer et al., 2023). Several papers therefore study cheaper constructions or approximations that preserve or closely approximate the same persistent homology  (Leitão, 2026; Koyama et al., 2024; Wang & Wei, 2016).  

Your proposed validation strategy—choose the coarsest cubical resolution whose diagrams are not detectably different from a high-fidelity reference—fits this literature well. Cubical methods have documented convergence and reliability across mesh sizes, and one object-oriented cubical framework reports consistency with Vietoris–Rips across many numerical tests  (Wang & Wei, 2016). What is still missing is a principled inferential rule that turns “close enough” interleaving or empirical similarity into a calibrated acceptance region for mesh size selection  (Saki & Faghihi, 2026; Leitão, 2026).

| Filtration Question | What Literature Supports | Main Limitation |
|---|---|---|
| Rips vs cheaper approximation | Orders-of-magnitude smaller approximation with log³-interleaving  (Leitão, 2026)| Approximation guarantee is not itself a hypothesis test |
| Standard vs distilled Rips | Same persistent homology, lower memory footprint  (Koyama et al., 2024)| Still tied to pairwise distances |
| Cubical vs Rips-style targets | Numerical consistency and mesh convergence reported  (Wang & Wei, 2016)| Target depends on representation and flow construction |
| Specialized fast Rips cases | Near-quadratic algorithms in restricted geometric settings  (Coldren, 2019)| Narrow applicability |

**Figure 1:** Filtration comparison questions and current evidence limits

## Evidence Gaps

## Evidence Coverage Across This Idea
|  | **Two-sample testing** | **Filtration specificity** | **Computation** | **Validation for tuning** |
|---|---|---|---|---|
| Point-cloud group difference | **6** | **5** | **4** | **2** |
| Native diagram inference | **4** | **3** | **2** | **1** |
| Vectorized summary testing | **5** | **4** | **4** | **2** |
| Filtration equivalence testing | **1** | **5** | **6** | **1** |
| Mesh-size selection | **GAP** | **3** | **4** | **GAP** |

**Figure 2:** Evidence coverage across testing and filtration questions

The biggest gap is **filtration equivalence as a statistical hypothesis test**. The corpus contains stability, interleaving, convergence, and isomorphism results for alternative filtrations, but not a calibrated null test that says when two filtration pipelines are statistically indistinguishable on finite samples  (Leitão, 2026; Koyama et al., 2024; Wang & Wei, 2016).  

A second gap is **decision-theoretic tuning** of representation parameters such as persistence-image weights, filtering thresholds, or cubical mesh size. Existing testing papers note unresolved parameter selection, while the causal papers insist those choices define the estimand itself, so tuning cannot be separated from the scientific question  (Moon & Lazar, 2020; Saki & Faghihi, 2026; Faghihi, 2026).  

Overall, the most defensible next project is a **filtration-conditional two-sample test** with null “topological contrast equals zero under a prespecified filtration and summary,” followed by a second validation test for whether a cheaper filtration is inferentially indistinguishable from a reference one on held-out data. That is very close to what the current literature enables, and it addresses a real gap rather than duplicating existing permutation or vectorized-diagram tests.
 
_These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at https://consensus.app. © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent._
 
## References
 
Bauer, U., Lenzen, F., & Lesnick, M. (2023). Efficient Two-Parameter Persistence Computation via Cohomology. 15:1-15:17. https://doi.org/10.4230/lipics.socg.2023.15
 
Bobrowski, O., & Skraba, P. (2023). A universal null-distribution for topological data analysis. *Scientific Reports, 13*. https://doi.org/10.1038/s41598-023-37842-2
 
Cericola, C., Johnson, I., Kiers, J., Krock, M., Purdy, J., & Torrence, J. (2016). Extending hypothesis testing with persistent homology to three or more groups. *Involve, A Journal of Mathematics, 11*, 27-51. https://doi.org/10.2140/involve.2018.11.27
 
Coldren, E. (2019). On Vietoris – Rips complexes : the persistent homology of cyclic graphs.
 
Debnath, M., Salvia, V. K., Siddiqui, A., Qidwai, K. A., Durga, P., & Bavankumar, S. (2025). Mathematical Foundations of Explainable AI: A Framework based on Topological Data Analysis. *Communications on Applied Nonlinear Analysis*. https://doi.org/10.52783/cana.v32.4650
 
Faghihi, U. (2026). Topological Ignorability for Structural Causal Effects Beyond Means. https://doi.org/10.48550/arxiv.2606.01184
 
Kim, K., & Lee, H. (2026). Topological Causal Effects. *ArXiv, abs/2603.02289*. https://doi.org/10.48550/arxiv.2603.02289
 
Kovačev-Nikolić, V. (2012). Persistent Homology in Analysis of Point-Cloud Data. https://doi.org/10.7939/r3rp90
 
Koyama, M. A., Robins, V., & Turner, K. (2024). The distilled Vietoris Rips filtration for persistent homology and a new memory efficient algorithm. https://doi.org/10.48550/arxiv.2412.07805
 
Leitão, A. (2026). It's All About Covers: Persistent Homology of Cover Refinements. https://doi.org/10.48550/arxiv.2602.22784
 
Moon, C., & Lazar, N. (2020). Hypothesis testing for shapes using vectorized persistence diagrams. *Journal of the Royal Statistical Society Series C: Applied Statistics*. https://doi.org/10.1093/jrsssc/qlad024
 
Robinson, A., & Turner, K. (2013). Hypothesis testing for topological data analysis. *Journal of Applied and Computational Topology, 1*, 241-261. https://doi.org/10.1007/s41468-017-0008-7
 
Saki, A., & Faghihi, U. (2026). Beyond Means: Topological Causal Effects under Persistent-Homology Ignorability. *ArXiv, abs/2603.14169*. https://doi.org/10.48550/arxiv.2603.14169
 
Wang, B., & Wei, G. (2016). Object-oriented Persistent Homology. *Journal of computational physics, 305*, 276 - 299. https://doi.org/10.1016/j.jcp.2015.10.036
 
