# Implementation note — identification audit (Phase 5AB Section 1.2)

This note freezes the identification claims for the benchmark's point-law methods. It is written **before** large experiments (WP-B0) so that no bandwidth, block size, or hybrid weight is chosen after seeing rejection rates.

## 1. Gaussian point kernel MMD identifies P0=P1

*Claim:* Under the iid metric-measure model on a common Euclidean ambient space, the Gaussian point kernel is characteristic, so equality of its mean embeddings identifies `P0=P1` under the usual moment and measurability conditions.  
*Source:* Gretton, Borgwardt, Rasch, Scholkopf, Smola (2012), JMLR 13, Theorem 5 and discussion of characteristic kernels; Simon-Gabriel & Scholkopf (2018), JMLR 19 (conditions for characteristic and universal kernels and kernel metrics on distributions).  
*Implementation:* `tda2s/tests/point_law.py:point_mmd_gaussian` uses a bounded Gaussian kernel `k(x,y)=exp(-||x-y||^2 / 2 sigma^2)` with bandwidth fixed before labels (primary locked `sigma=0.30` on unit-square scale; sensitivity multipliers {0.25,0.5,1,2,4}; median-heuristic variant computes median Euclidean distance once from pooled unlabeled points and records it). Statistic is biased V-statistic MMD^2 `mean K[X,X]+mean K[Y,Y]-2 mean K[X,Y]`. Calibration is pooled label permutation preserving `n0,n1`. This matches the source's permutation construction.

## 2. Euclidean energy distance identifies equality of distributions

*Claim:* Euclidean energy distance identifies equality of distributions under finite-moment conditions.  
*Source:* Szekely & Rizzo (2013), JSPi; Sejdinovic, Sriperumbudur, Gretton & Fukumizu (2013), AoS (equivalence of distance-based and RKHS statistics, including identifying classes).  
*Implementation:* `energy_distance_test` computes Euclidean distance matrix once: `E = 2*mean(D[X,Y]) - mean(D[X,X]) - mean(D[Y,Y])` (biased V-statistic, documented; unbiased U-statistic would exclude diagonal and is not used). Calibrated by pooled permutation with same distances. Reported with distance convention, moment assumption (finite first moment), and V-statistic flag.

## 3. MST, kNN, cross-match are valid permutation statistics, not characteristic kernels

*Claim:* Friedman-Rafsky MST, Schilling kNN, and Rosenbaum cross-match statistics are valid permutation statistics under the iid point-law null, but their consistency and power claims require the graph-test literature. They must not be described as characteristic-kernel tests.  
*Source:* Friedman & Rafsky (1979), AoS (MST/run test); Schilling (1986), JASA (kNN); Rosenbaum (2005), JRSS B (cross-match).  
*Implementation:*  
- MST: `friedman_rafsky_mst` builds the Euclidean MST once with deterministic Prim on the complete pooled distance matrix, statistic = number of cross-arm edges (small is extreme, left-tail). The complete-graph implementation preserves zero-distance edges between duplicate points, reports vertices=N, edges=N-1, components=1, and resolves ties by pooled index.
- kNN: `schilling_knn` builds pooled k-NN graph once (stable mergesort tie resolution), statistic = same-label directed (primary) edge count, with `k=1` primary and `{1,5,10}` panel.  
- CrossMatch: `rosenbaum_crossmatch` builds minimum-weight non-bipartite matching once (networkx `max_weight_matching` blossom on complete graph with weight=-Euclidean, maxcardinality). Counts cross pairs (small is extreme). If pooled N>500, method **refuses** the cell (`status=refused`) rather than silently switching to an approximate matching; odd N leaves one unmatched per maximum-cardinality rule. All report permutation group `all point-label splits with n0,n1` and `validity_regime=iid_metric_measure`.

## 4. Sliced Wasserstein finite projections are a sensitivity statistic

*Claim:* A finite collection of sliced-Wasserstein projections is a sensitivity statistic unless the implementation and theory establish an identifying projection family. An exact or sufficiently rich construction may be point-law-sensitive but finite approximations must be labelled honestly.  
*Source:* Ramdas, Garcia & Cuturi (2015), arXiv:1509.02237.  
*Implementation:* `sliced_wasserstein_test` uses fixed-regularisation exact sliced-Wasserstein-1: `n_projections=100`, isotropic Gaussian directions normalised to unit sphere, `projection_seed` fixed before labels, recorded in every result, `transport_solver = exact 1D Wasserstein via sorting (scipy.stats.wasserstein_distance)`, regularisation `none`. Statistic = mean W1 over projections. Calibrated by pooled permutation. Method is **labelled** `target_null = H0^law (sensitivity; finite projections)` and `is_full_point_law_test=False` in the registry. It is geometrically interpretable but not promoted to universally identifying.

## 5. Classifier two-sample test is classifier-relative

*Claim:* A classifier two-sample test targets point-law equality only relative to its classifier class and sample-splitting/calibration protocol. It is not automatically universally consistent because the classifier may be misspecified.  
*Source:* Lopez-Paz & Oquab (2017), ICLR.  
*Implementation:* `classifier_two_sample_test` uses sample-split with fixed family: `logistic` (sklearn `LogisticRegression(max_iter=1000,solver=lbfgs)`) and `rf` (`RandomForest(n_estimators=100,n_jobs=1)`) as separate variants. Training/test split is stratified `test_fraction=0.5`, `split_seed` predeclared, independent of treatment labels. Hyperparameters fixed or selected only on training data. Calibration: fit once on training, evaluate held-out accuracy, then **held-out label permutation conditional on fixed training fit** (no retraining inside loop; exact retrain variant noted as alternative). Reports classifier class, split fraction, seeds, train time, and tuning flag `none`. Pooled hyperparameter search on test split is prohibited in the confirmatory benchmark.

## 6. Average pairwise bag kernel is a downgraded sensitivity variant

*Claim:* The simple average pairwise kernel on raw bags (`mean_pairwise`) remains a downgraded sensitivity variant and must not replace the Hilbert-Gaussian unordered-bag kernel in the primary RawBlockMMD claim.  
*Source:* Buathong et al. (2020), PMLR 108; Szabo et al. (2015), PMLR 38 (finite-set kernels discussion); project-specific characteristicness audit in `tda2s/tests/single_cloud.py:_block_target`.  
*Implementation:* `RawBlockMMD` primary uses `raw_kernel=gaussian_mean_embedding` (Gaussian on characteristic point-kernel mean embedding, characteristic on fixed-size unordered bags via Gaussian kernel on mean embedding; see `single_cloud.py: _raw_block_kernel` and diagnostics). `mean_pairwise` is exposed only as `H0^raw-block-sensitivity` diagnostic with `kernel_characteristicness = not characteristic for arbitrary bag laws` and is never used as primary.

## Consequence for headline comparison

If any of the checks above fails for the selected implementation (e.g. missing dependency, theoretical mismatch, or label-dependent tuning), that method is **excluded from the headline `H0^law` ranking** and a refusal/failure record with `failure_reason` is emitted (Section 3.3). CrossMatch exceeding the `n>500` budget is an expected refusal, not a silent omission.

## Verification hooks implemented

* Preservation of unequal arm sizes in every permutation.
* Exact small-sample enumeration test (`exact=True` branch) for all point-level methods; verified against independent brute-force on `n<=10`.
* Bandwidth/median-heuristic invariance to point ordering; `point_mmd_gaussian` at `bandwidth` computed from pooled unlabeled points is permutation-invariant.
* Graph/matching methods: permutation preserves `n0,n1` and keeps graph/matching fixed; tiny exact enumeration agrees with independent counting.
* Block vs point bridge: `RawBlockMMD` at `m=1` uses Gaussian mean-embedding kernel, distinct from `PointMMD-Gaussian` point kernel; both reported separately per plan.
