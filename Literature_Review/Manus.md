# Literature Review: Statistical Tests for Comparing Point Clouds via Topological Data Analysis and Causal Frameworks

## 1. Introduction
Topological Data Analysis (TDA) has emerged as a robust framework for characterizing the multi-scale geometric structure of complex datasets. A central challenge in TDA is the development of rigorous statistical methods to compare populations of persistence diagrams (PDs), which are non-Euclidean summaries of data topology. This review examines the recent integration of causal inference with TDA, specifically focusing on the Topological Average Treatment Effect (TATE), and explores the statistical comparison of point clouds through varying filtrations, such as Vietoris–Rips and cubical complexes.

## 2. Topological Causal Inference: TATE and TCDA
The foundational papers by Kim & Lee [1] and Souto & Diamantis [2] have recently formalized the intersection of causal inference and TDA. 

### 2.1 The TATE Framework
Kim & Lee [1] introduce the **Topological Average Treatment Effect (TATE)**, defined as the expected contrast of topological summaries (specifically power-weighted silhouettes) under potential outcomes. Their work addresses the limitation of Euclidean summaries in capturing structural changes induced by interventions. 
> "Topological causal effects are defined through intervention-induced changes in persistent homology summaries... designed to capture structural effects that are invisible to scalar or Euclidean summaries" [1].

They develop a **doubly robust, non-parametric estimator** for TATE and establish functional weak convergence, enabling a formal hypothesis test for the null hypothesis of no topological effect ($H_0: \text{TATE} = 0$). This framework is directly applicable to the user's research idea: by treating group membership in a two-sample comparison as a "treatment" (indicator covariate $A \in \{0, 1\}$), one can test for statistical differences between two point clouds by evaluating if the estimated TATE is significantly different from zero.

### 2.2 Topological Causal Data Analysis (TCDA)
Souto & Diamantis [2] expand this into a broader **Topological Causal Data Analysis (TCDA)** framework. They distinguish between:
*   **Outcome-level TCDA**: Compares the expected topological summaries of individual potential outcomes (TATE_out).
*   **Distribution-level TCDA**: Compares representations of the interventional laws themselves (Delta_dist).

This distinction is crucial for statistical testing, as distribution-level effects can detect changes in population structure (e.g., one cluster splitting into two) that might be masked by simple averaging at the outcome level.

## 3. Statistical Tests for Persistence Diagrams
Beyond the TATE framework, several non-parametric and kernel-based methods exist for comparing persistence diagrams.

### 3.1 Non-Parametric Tests: Permutation and Bootstrap
Permutation tests are the "gold standard" for two-sample testing in TDA, where group labels are shuffled to build a null distribution of a distance metric (e.g., Wasserstein or Bottleneck distance) [3]. However, these are computationally intensive for large datasets. Fasy et al. [4] introduced **bootstrap methods** to construct confidence sets for persistence diagrams, providing a way to assess the significance of individual features.

### 3.2 Kernel-Based Methods and MMD
The mapping of persistence diagrams into Reproducible Kernel Hilbert Spaces (RKHS) allows for the use of the **Maximum Mean Discrepancy (MMD)** as a test statistic. Kusano et al. [5] developed stable kernels for PDs (e.g., Persistence Scale-Space Kernel), which facilitate efficient two-sample testing. These methods provide a more computationally tractable alternative to permutation tests while maintaining statistical power.

## 4. Filtration Comparison: Vietoris–Rips vs. Cubical Complexes
A significant practical concern in TDA is the choice of filtration. While the **Vietoris–Rips (VR)** complex is standard for point clouds, its computational complexity ($O(n^k)$) makes it prohibitive for large datasets. **Cubical complexes**, typically used for voxelized or grid data, offer a more efficient alternative.

### 4.1 Voxelization and Convergence
Recent research, such as the **Flood Complex** by Graf et al. [6], demonstrates that large-scale point clouds (millions of points) can be efficiently processed by converting them into voxel grids. The user's hypothesis—that cubical complexes converge to the persistence of the underlying structure as cube size decreases—is supported by the **Stability Theorem** of persistent homology [7]. This theorem ensures that if two filtrations are "close" (e.g., in the $L_\infty$ sense of their filtration functions), their persistence diagrams will be close in the Bottleneck distance.

### 4.2 Resolution Selection via Statistical Similarity
The user proposes using a validation dataset to determine the largest acceptable cube size for a cubical complex filtration. This aligns with the concept of **multiscale stability**. By comparing the PD of a coarse cubical complex to a "ground truth" (either a fine-grained cubical complex or a Rips complex on a subset), one can use the TATE-based test or MMD to find the threshold where the diagrams become statistically indistinguishable. This provides a principled way to balance computational cost and topological accuracy.

## 5. Synthesis and Research Direction
The proposed research idea—developing a statistical test for point cloud comparison using the TATE framework and an indicator covariate—is highly viable and grounded in recent literature. 

| Feature | TATE-based Test | MMD-based Test | Permutation Test |
| :--- | :--- | :--- | :--- |
| **Foundation** | Causal Inference | Kernel Methods | Non-parametric |
| **Summary Type** | Silhouettes/Landscapes | RKHS Embeddings | Raw Diagrams (Distance) |
| **Robustness** | Doubly Robust | Stable Kernels | Distribution-free |
| **Computation** | Efficient (Functional) | Efficient (Kernel) | Expensive |

The "side idea" of comparing filtrations is particularly timely. As TDA moves toward "big data" applications, the ability to statistically justify the use of faster filtrations (like cubical complexes) through resolution-validation will be a valuable contribution to the field.

## 6. References
1. Kim, K., & Lee, H. (2026). *Topological Causal Effects*. arXiv:2603.02289.
2. Souto, H. G., & Diamantis, I. (2026). *A Mathematical Framework for Topological Causal Data Analysis*. arXiv:2607.28161.
3. Islambekov, U. (2024). *Vector summaries of persistence diagrams for permutation tests*. Foundations of Data Science.
4. Fasy, B. T., et al. (2014). *Confidence sets for persistence diagrams*. The Annals of Statistics.
5. Kusano, G., et al. (2018). *Kernel Method for Persistence Diagrams via Persistence Scale-Space*. JMLR.
6. Graf, F., et al. (2025). *The flood complex: Large-scale persistent homology on millions of points*. NeurIPS.
7. Cohen-Steiner, D., et al. (2007). *Stability of persistence diagrams*. Discrete & Computational Geometry.
8. Beksi, W. J., & Papanikolopoulos, N. (2019). *A topology-based descriptor for 3D point cloud modeling*. Image and Vision Computing.
