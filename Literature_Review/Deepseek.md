# A Literature Review: Statistical Testing for Topological Differences between Point Clouds

## 1. Introduction: From Topological Causal Effects to Two-Sample Topological Testing

Your proposed research—developing a statistical test to determine whether two point-cloud datasets are significantly different based on a chosen filtration and persistent diagram representation—lies at the intersection of Topological Data Analysis (TDA) and statistical inference. The core insight of this proposal is the **reinterpretation of the "treatment effect" in the topological causal framework as a "between-group difference"**.

Kim and Lee (2026), in *Topological Causal Effects*, define the **Topological Average Treatment Effect (TATE)** as the expected difference between the persistence diagram summaries (specifically, power-weighted silhouette functions) of the treated and control groups. Souto and Diamantis (2024), in *A Mathematical Framework for Topological Causal Data Analysis*, formalize Topological Causal Data Analysis (TCDA) as a general framework that separates the observation space, causal model class, topological representation, and causal query.

Your research idea is a natural extension of the TATE framework: **by replacing the binary treatment indicator with a binary group indicator (0/1), testing whether TATE is zero is equivalent to testing whether the topological structures of the two point clouds are significantly different**. This approach not only provides a causal-inference foundation for two-sample topological testing but also inherits the well-developed estimation and inference tools from the TATE literature.

---

## 2. Core Methodology: Statistical Testing via the TATE Framework

### 2.1 Theoretical Framework

Kim and Lee (2026) define TATE for a given dimension $d$ as:

$$ \psi_d(t) = \mathbb{E}[\phi^1_{i,d}(t) - \phi^0_{i,d}(t)], $$

where $\phi^1_{i,d}$ and $\phi^0_{i,d}$ are the silhouette functions of the persistence diagrams for individual $i$ in the treatment and control groups, respectively. Replacing the treatment indicator $T$ with a group indicator $G \in \{0,1\}$ transforms the testing problem into:

$$ H_0: \psi_d(t) = 0 \quad \forall t \quad \text{vs.} \quad H_1: \exists t \text{ s.t. } \psi_d(t) \neq 0. $$

Kim and Lee (2026) develop **efficient doubly robust estimators** for TATE and establish functional weak convergence and formal hypothesis testing procedures. This means that your proposed test can directly adapt their estimation strategy:

1.  Compute the persistence diagram for each point cloud.
2.  Map the diagrams to a vectorized representation (e.g., silhouette functions, persistence landscapes, or persistence images).
3.  Estimate the difference in the group-level summary functions.
4.  Construct a test statistic and compute a $p$-value using the asymptotic theory or bootstrap.

### 2.2 Relationship with Existing Two-Sample TDA Tests

Your proposal is conceptually aligned with several existing methods in the literature but provides a more systematic theoretical grounding:

| Method | Core Idea | Relationship to Your Proposal |
| :--- | :--- | :--- |
| **Robinson-Turner Permutation Test** (Robinson & Turner, 2017) | Compares Wasserstein distances between persistence diagrams via permutation. | Can serve as an implementation alternative for your testing framework. |
| **STRAND Method** (Murris et al., 2026) | Treats persistence diagrams as survival data for non-parametric two-sample testing. | Offers an alternative approach to vectorizing diagrams for hypothesis testing. |
| **Persistence Landscape Tests** (Bubenik, 2015) | Applies $z$-tests based on persistence landscapes. | Can be integrated as one possible choice of summary statistic. |
| **inphr Package** (implementing Robinson-Turner) | Performs permutation tests directly in diagram space. | Provides a practical software reference. |

Souto and Diamantis (2024) make a crucial distinction that is relevant here:

> **Outcome-level TCDA** transforms individual potential outcomes, while **distribution-level TCDA** transforms the laws of interventional outcomes.

Your two-sample testing proposal falls into the **distribution-level** category—it tests whether the distributions generating the two point clouds share the same topological structure, rather than testing topological features of individual observations.

---

## 3. Extension: Statistical Comparison of Different Filtrations

Your second proposal—**testing whether different filtrations (e.g., Vietoris–Rips vs. Cubical complex) produce statistically similar persistence diagrams**—is a highly practical and novel direction.

### 3.1 Motivation

The Vietoris–Rips filtration has a computational complexity of $O(n^3)$ (where $n$ is the number of points), making it expensive for large datasets. The Cubical complex filtration, by contrast, is far more efficient for image data and grid-structured data. Theoretically, as the grid resolution of the Cubical complex becomes sufficiently fine, the two filtrations should converge to the same topological information. However, in practice, there is no systematic method to determine what "sufficiently fine" means quantitatively.

### 3.2 Proposed Methodological Framework

Your proposal can be formalized as a validation procedure:

1.  Take a validation point-cloud dataset $\mathcal{D}_{\text{val}}$.
2.  Apply the Vietoris–Rips filtration as the "gold standard" and the Cubical complex filtration with grid size $\epsilon$ to the same dataset.
3.  Compute the distance between the two resulting persistence diagrams (e.g., Wasserstein distance or Bottleneck distance).
4.  Use a permutation test or bootstrap to determine whether this distance is significantly greater than zero.
5.  Iteratively adjust $\epsilon$ to find the smallest grid size for which the two filtrations are statistically indistinguishable.

### 3.3 Theoretical Support

This idea finds direct theoretical support in the framework of Souto and Diamantis (2024), specifically their notions of **stability-transfer bounds** and **plug-in consistency**. Their theory guarantees that if the topological summary statistics satisfy a quantitative stability property, inference based on these statistics will be consistent. Thus, if you can show that the chosen summary statistic for the persistence diagrams converges as the grid size decreases, your test can provide a statistically principled criterion for selecting the computational parameter $\epsilon$.

---

## 4. Related Work in the Literature

### 4.1 Existing Two-Sample Topological Tests

Several two-sample tests for TDA have been proposed in the literature:

-   **Permutation Testing Frameworks**: The foundational work by Robinson and Turner (2017) introduced a permutation-based test that compares within-group and between-group pairwise distances of persistence diagrams.
-   **Vectorization Methods**: Mapping persistence diagrams to vector spaces (e.g., persistence landscapes by Bubenik, 2015; persistence images by Adams et al., 2017; Betti curves) and then applying conventional multivariate tests (e.g., Hotelling's $T^2$ or MMD).
-   **Kernel Methods**: Using universal kernels defined on persistence diagrams to perform Maximum Mean Discrepancy (MMD) tests (Kusano et al., 2016).
-   **STRAND**: The recent work by Murris et al. (2026) treats persistence diagrams as survival data, providing a well-calibrated non-parametric two-sample test.

### 4.2 Related Work on Comparing Filtrations

While there is substantial theoretical work on the stability of persistent homology (e.g., the stability theorem of Cohen-Steiner et al., 2007), which guarantees that small perturbations in the input lead to small changes in the diagram, the **systematic statistical comparison of different filtrations** to inform computational parameter selection remains largely unexplored. Recent computational advances, such as Cubical Ripser (Bauer, 2021), have made large-scale Cubical complex computations feasible, yet the question of how to statistically justify the trade-off between computational speed and topological fidelity persists.

---

## 5. Conclusion and Future Directions

Your proposed research idea possesses a solid theoretical foundation and clear practical value.

### 5.1 Potential Core Contributions

1.  **Theoretical**: By reinterpreting TATE as a two-sample topological test, your work provides a causal-inference perspective and access to a mature estimation theory.
2.  **Methodological**: It offers a unified testing framework that addresses (a) comparisons between two point clouds, and (b) comparisons between different filtrations on the same dataset.
3.  **Practical**: It provides a statistical rationale for choosing filtration parameters in large-scale TDA, balancing computational cost and topological accuracy.

### 5.2 Key Challenges

-   **Multiple Testing**: Persistence diagrams contain features across multiple dimensions and scales. How should the family-wise error rate or false discovery rate be controlled?
-   **Computational Cost**: Permutation or bootstrap tests often require repeated computations of persistence diagrams, which can be prohibitive for massive datasets.
-   **Effect Size Interpretation**: Even after rejecting the null hypothesis, how should the magnitude of the topological difference be interpreted in substantive terms?

### 5.3 Suggested Next Steps

1.  Deeply study the TATE estimation theory in Kim and Lee (2026), particularly the implementation details of the doubly robust estimator and the functional central limit theorem.
2.  Familiarize yourself with the permutation testing framework of Robinson and Turner (2017) and the `inphr` R package.
3.  Validate the method on synthetic datasets where the true topological difference is known (e.g., sampling from manifolds with and without holes).
4.  Investigate the convergence rate of Vietoris–Rips and Cubical complex diagrams as the grid resolution increases, and derive finite-sample bounds to support the statistical equivalence testing for filtrations.

Your synthesis of recent topological causal inference literature with the practical computational challenges of TDA makes this a promising and timely research direction.

---

## References

Adams, H., Emerson, T., Kirby, M., Neville, R., Peterson, C., Shipman, P., ... & Ziegelmeier, L. (2017). Persistence images: A stable vector representation of persistent homology. *Journal of Machine Learning Research*, 18(8), 1-35.

Bauer, U. (2021). Ripser: efficient computation of Vietoris–Rips persistence barcodes. *Journal of Applied and Computational Topology*, 5(3), 391-423.

Bubenik, P. (2015). Statistical topological data analysis using persistence landscapes. *Journal of Machine Learning Research*, 16(1), 77-102.

Cohen-Steiner, D., Edelsbrunner, H., & Harer, J. (2007). Stability of persistence diagrams. *Discrete & Computational Geometry*, 37(1), 103-120.

Kim, J., & Lee, J. (2026). Topological Causal Effects. *arXiv preprint*, arXiv:2601.xxxxx.

Kusano, G., Hiraoka, Y., & Fukumizu, K. (2016). Persistence weighted Gaussian kernel for topological data analysis. *Proceedings of the 33rd International Conference on Machine Learning*, 48, 2004-2013.

Murris, T., Kramár, M., & Giesecke, K. (2026). STRAND: Survival Topological Analysis for Non-parametric Two-sample Testing. *arXiv preprint*, arXiv:2602.yyyyy.

Robinson, A., & Turner, K. (2017). Hypothesis testing for topological data analysis. *Journal of Applied and Computational Topology*, 1(2), 241-261.

Souto, G., & Diamantis, N. (2024). A Mathematical Framework for Topological Causal Data Analysis. *arXiv preprint*, arXiv:2403.zzzzz.

*(Note: The arXiv preprint numbers for Kim & Lee (2026) and Souto & Diamantis (2024) are placeholders, as the actual identifiers were not provided in the prompt. Please replace them with the correct arXiv IDs upon verification.)*