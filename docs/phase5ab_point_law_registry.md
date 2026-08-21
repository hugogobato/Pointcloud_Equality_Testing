# Phase 5AB point-law benchmark registry (WP-B0 freeze)

**Benchmark version:** `phase5ab-pointlaw-v1`  
**Design hash:** derived from `experiments/phase5ab_pointlaw_tournament.py:design_record()` SHA256[:16]  
**Regime:** `iid_metric_measure` (Regime I) only. No size claim for spatial processes, overlapping blocks, or data-dependent partitions.  
**Significance level (primary):** `alpha = 0.05` (diagnostics at 0.01, 0.10)  
**Monte Carlo interval:** two-sided 95% Clopper-Pearson (beta quantiles)  
**Replications (primary):** 500 per cell; 200 exploratory; 1000 for selected size cells  
**Permutations (primary):** 199 per replication; 39 reproduction panel; 999 cheap-method precision panel  

## Shared design constants

| Quantity | Primary value | Sensitivity values |
|---|---|---:|
| Block size for RawBlockMMD/SC-B | m=25 | m in {1,2,5,10,25,50} |
| Hybrid weight | alpha in {0.25,0.50,0.75,1.00} | none chosen post hoc |
| Primary ambient dimension | d=2 | d in {1,5,10,20,50} where DGP defined |
| Primary point kernel | Gaussian | bandwidth multipliers {0.25,0.5,1,2,4} |
| Primary bag kernel | Hilbert-Gaussian unordered-bag | point bandwidth 0.10, bag bandwidth 0.25 |
| Primary graph k | k=1 | k in {1,5,10} |
| Primary OT projections | 100 sliced | seed fixed |
| Primary C2ST split | 0.5 stratified | logistic + RF variants |

## Method registry

| Method | Declared target | Sampling unit | Observation used | Validity basis | Full point-law? |
|---|---|---|---|---|---|
| PointMMD-Gaussian (locked bw) | H0^law: P0=P1 | individual iid point | pooled Gaussian Gram (bounded characteristic) | Gretton et al. 2012 (A1); Simon-Gabriel 2018 (A2) | yes |
| PointMMD-Gaussian (median heuristic) | H0^law: P0=P1 | individual iid point | pooled median-heuristic bandwidth (unlabeled pooled) | same | yes |
| EnergyDistance (V-statistic) | H0^law: P0=P1 | individual iid point | pooled Euclidean distance matrix | Szekely & Rizzo 2013 (B1); Sejdinovic 2013 (B2) | yes |
| FriedmanRafsky-MST | H0^law: P0=P1 | individual iid point | pooled MST fixed, cross-edge count (left-tail) | Friedman & Rafsky 1979 (C1) | yes (graph) |
| Schilling-kNN k=1 | H0^law: P0=P1 | individual iid point | pooled k-NN graph fixed, same-label edges (greater) | Schilling 1986 (C2) | yes (graph) |
| Schilling-kNN k=5,10 | H0^law: P0=P1 | individual iid point | same | same | yes |
| Rosenbaum-CrossMatch | H0^law: P0=P1 | individual iid point | pooled minimum-weight matching fixed (networkx blossom) | Rosenbaum 2005 (C3) | yes (graph) |
| SlicedWasserstein (100 proj) | H0^law sensitivity | individual iid point | pooled projections fixed, W1 average | Ramdas et al. 2015 (D1) | no — sensitivity only |
| ClassifierTwoSample (logistic, RF) | H0^law relative to classifier | individual iid point | stratified 50/50 split, held-out accuracy, held-out label permutation | Lopez-Paz & Oquab 2017 (E1) | no — classifier-relative |
| RawBlockMMD m=25 | H0^law: P0=P1 | frozen disjoint m-point block | Hilbert-Gaussian bag kernel, K=floor(n/m) per arm | characteristic bag kernel + iid block exchangeability | yes (block) |
| HybridBlockMMD alpha>0 | H0^law: P0=P1 | frozen disjoint block | alpha*K_raw+(1-alpha)*K_barcode | same | yes |
| HybridBlockMMD alpha=0 | H0,m^bar | frozen disjoint barcode block | persistence only | barcode-law diagnostic | no |
| SC-B | H0,25^bar | frozen disjoint block | joint degree 0:1 persistence diagrams | Blumberg et al. 2014 (F1) | no — barcode law |
| SC-A | H0^law: P0=P1 | individual iid point | persistence recomputed per split | point-law permutation baseline (persistence representation) | point-law null but persistence-only |

Guardrail: the word `point-cloud` never silently means persistence diagram. Registry distinguishes `individual iid point`, `frozen disjoint raw block`, `frozen disjoint barcode block`, `diagram observation`.

## Output schema (Section 3.3)

Every replication record (including refusals) must contain:

```
benchmark_version, design_hash, family, family_role, family_description,
method, method_variant, target_null, validity_regime, sampling_unit,
n0, n1, d, m, K0, K1, effective_sample_size_total, unused_points0, unused_points1,
n_permutations, exact_enumeration, permutation_group,
statistic, pvalue, rejected,
kernel_or_distance, bandwidth_or_tuning, alpha,
cloud_seed, partition_seed, permutation_seed,
runtime_seconds, peak_rss_bytes, status, failure_reason
```

Additional method-specific fields (e.g. k, n_projections, classifier, n_vertices) are allowed.

## Sampling-unit and effective-sample-size reporting

* Point-level methods: `K0=K1=NA`, `effective_sample_size_total = n0+n1`, `unused=0`, `m=NA`.
* Block methods: `K_a=floor(n_a/m)`, `effective_sample_size_total=K0+K1`, `unused = n_a - K_a*m`, `m` recorded, `d` recorded.
* Report must contain **both** an all-point comparison and an effective-sample-size comparison. Never equate 10 raw blocks with 250 points.
* `m=1` RawBlockMMD is the direct bridge between point-level and block-level constructions.

## Seed convention

For each replication, generate each cloud once from a deterministic seed keyed by `(benchmark_version, family, n0, n1, dimension, replication)`. The cloud seed is independent of `m`, so block-size sensitivity cells reuse the same generated clouds. All methods in the cell receive the same clouds. Method-specific permutation seeds are separate and recorded. Partition seed is `(benchmark_version, cell_id, replication)` and is recorded for block methods.

The runner records `peak_rss_bytes` as the process high-water RSS observed during the call, while `peak_memory_bytes` is the Python-allocation peak from `tracemalloc`. These are resource diagnostics, not claims of an isolated allocator peak for every method.

## Refusal policy

A method that refuses a cell because `n_a<m`, `K_a<1`, `K_a<5` diagnostic, `k>=n`, matching infeasible (`n>500` CrossMatch limit or `n odd` handled), or classifier split undefined (`test split empty class`) must produce a refusal record (`status` in {`ok`,`failed`,`refused`}) rather than silently dropping the cell. Cells with `K_a<5` are diagnostic only because block permutation distribution is too coarse. Report exact minimum attainable p-value grid for each cell.

## DGP families (core)

* `iid_null` — Uniform([0,1]^2) point-law size gate
* `weak_barcode_null` — translated point law with equal barcode law (point-law power vs barcode-law calibration diagnostic)
* `same_support_density` — same square support, different continuous density
* `same_square_four_atom_density` — discrete four-atom square occupancy
* `topology_alt` — filled disk vs noisy circle
* Robustness: `robust_contamination`, `robust_unequal_cardinality`, `robust_anisotropic_noise`, `robust_boundary_truncation`
* Dependence (diagnostic only, `validity_status=unsupported_dependence`): `process_poisson` etc.

## Identification audit (Section 1.2)

All six checks passed; see `docs/phase5ab_point_law_implementation_note.md`.

## Literature cheatsheet requirement

After acquisition, `docs/phase5ab_point_law_source_cheatsheet.md` will contain exact theorem/proposition numbers per source.

## Verification

Rerunning a cell with different shard assignment must reproduce same cloud and statistic given same method seed. `m=1` RawBlockMMD compared with independently implemented point-kernel statistic must agree (WP-B1 verification).
