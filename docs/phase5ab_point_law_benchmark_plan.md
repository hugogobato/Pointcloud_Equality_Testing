# Phase 5AB point-law benchmark plan

## Handoff objective

Extend the Phase 5 comparison so that `RawBlockMMD` is evaluated against established methods for testing equality of multivariate point distributions, not only against SC-A and SC-B. The benchmark must preserve the distinction between

\[
H_0^{\mathrm{law}}:P_0=P_1
\]

and

\[
H_{0,m}^{\mathrm{bar}}:\Phi^m_{0:1}(P_0)=\Phi^m_{0:1}(P_1).
\]

The main headline method remains `RawBlockMMD` with the locked disjoint-block construction and `m=25`. The benchmark must determine whether its improved calibration and density sensitivity survive comparison with raw-data methods that use all individual points.

This document is an execution plan for another agent. It is not permission to change the target, silently replace the existing API, or select a method, bandwidth, block size, or hybrid weight after inspecting the final rejection results.

## 1. Preflight audit and current repository state

The current implementation is in `tda2s/tests/single_cloud.py`. The relevant APIs are `sc_a_label_permutation`, `sc_b_disjoint_mmd`, `raw_block_mmd`, `hybrid_block_mmd`, and `sc_a_blockwise_label_permutation`. The Phase 5 runner is `experiments/phase5ab_block_tournament.py`, which reuses the DGP constructors and seed conventions from `experiments/phase5_single_cloud_tournament.py`.

The existing methods have the following status.

| Method | Declared target | Observation used for inference | Current interpretation |
|---|---|---|---|
| SC-A | `H0^law: P0=P1` | individual points, but persistence statistic recomputed after every pooled point-label split | point-law null with a persistence-only representation; it is not a fully identified raw point-law procedure |
| SC-B | `H0,25^bar` | frozen disjoint blocks of 25 points, represented by joint degree-0/degree-1 persistence diagrams | barcode-law production test |
| RawBlockMMD | `H0^law: P0=P1` | frozen disjoint raw blocks, `K_a=floor(n_a/m)` per arm | proposed point-law method under the iid metric-measure model |
| HybridBlockMMD, `alpha>0` | `H0^law: P0=P1` | the same raw blocks plus cached persistence features | point-law method with a topological auxiliary term |
| HybridBlockMMD, `alpha=0` | `H0,m^bar` | the same blocks, persistence only | barcode-law diagnostic, not a point-law test |
| SC-A-Block | `H0^law: P0=P1` | frozen disjoint raw blocks and pooled block labels | algebraically equivalent to RawBlockMMD under shared inputs |

The existing Phase 5 results are not sufficient as a general point-law benchmark because they do not yet include the main classical raw-data competitors: point-level MMD, energy distance, graph-based MST and nearest-neighbour tests, cross-match, Wasserstein-type tests, and classifier two-sample tests. The benchmark must add these methods under the same result schema rather than comparing incompatible ad hoc outputs.

### 1.1 Observation model and validity scope

The primary model is Regime I: the two clouds are independent iid samples from probability laws `P0` and `P1` on a common metric-measure space. Under this model, individual point labels are exchangeable under `P0=P1`, and the frozen disjoint blocks used by RawBlockMMD and SC-B are independent observations with laws `P0^m` and `P1^m`.

No primary size claim is allowed for spatially dependent point processes, clustered occurrence records, overlapping blocks, or a data-dependent partition. The existing Poisson, inhomogeneous Poisson, Cox-clustered, and hard-core cells remain diagnostics only. If an agent wants a valid spatial benchmark, it must define a new sampling unit and a separate dependence-aware calibration argument.

### 1.2 Identification audit before implementation

The agent must write a short implementation note confirming the following.

1. A Gaussian point kernel on a Euclidean ambient space is characteristic, so point-level Gaussian MMD identifies `P0=P1` under the usual moment and measurability conditions.
2. Euclidean energy distance identifies equality of distributions under its stated finite-moment conditions.
3. MST, k-nearest-neighbour, and cross-match statistics are valid permutation statistics under the iid point-law null, but their consistency and power claims require the conditions of the corresponding graph-test literature. They must not be described as characteristic-kernel tests.
4. A finite collection of sliced-Wasserstein projections is a sensitivity statistic unless the implementation and theory establish an identifying projection family. An exact or sufficiently rich Wasserstein construction may be described as point-law-sensitive, but finite computational approximations must be labelled honestly.
5. A classifier two-sample test targets point-law equality only relative to its classifier class and sample-splitting/calibration protocol. It is not automatically universally consistent because the classifier may be misspecified.
6. The simple average pairwise kernel on raw bags remains a downgraded sensitivity variant. It must not replace the Hilbert-Gaussian unordered-bag kernel in the primary RawBlockMMD claim.

If any of these checks fails for the selected implementation, stop the corresponding method from entering the headline comparison and record the reason in the machine-readable output.

## 2. Established point-law benchmark methods

The following methods should be implemented in priority order. The first four are required for the primary benchmark. The remaining methods are required if the profile run shows that they can fit the Colab budget, otherwise they become clearly labelled secondary benchmarks.

### 2.1 Required methods

#### PointMMD-Gaussian

Use all individual points. Construct one pooled point Gram matrix with a bounded Gaussian kernel whose bandwidth is fixed before treatment labels are used. Compute the usual two-sample MMD statistic and calibrate by pooled point-label permutation preserving `n0` and `n1`.

This is the closest established raw-data comparator. It must be implemented separately from RawBlockMMD because `RawBlockMMD` at `m=1` uses a Gaussian kernel on the empirical mean-embedding representation, whereas this comparator uses the ordinary point Gaussian kernel directly.

Required variants are a locked primary bandwidth and a predeclared bandwidth sensitivity panel. If the pooled median heuristic is used, compute it once from the pooled unlabeled points and record it. Do not estimate a bandwidth separately by arm or select it using rejection outcomes.

#### EnergyDistance

Use all points and the Euclidean distance matrix. The statistic is the usual two-sample energy statistic, calibrated by pooled point-label permutation. Implement the statistic directly from the pooled distance matrix so that the observed and permuted values use exactly the same distances.

This is a full point-law comparator under its finite-moment assumptions, not a topological method. Report its distance convention, moment assumptions, and whether the implementation uses the biased V-statistic or the unbiased U-statistic.

#### FriedmanRafsky-MST

Construct the Euclidean minimum spanning tree on the pooled points once. The statistic is the number of cross-arm MST edges, or equivalently the corresponding runs statistic, with a fixed direction chosen before looking at the result. Calibrate labels by permutation while keeping the MST fixed.

Ties must be resolved deterministically using the pooled point index. The method must report the number of vertices, edges, connected components if the implementation uses a forest, and the exact statistic definition. The MST is a representation of the pooled geometry, not an identifying kernel embedding, so the theory and claims must use the graph-test literature rather than the MMD argument.

#### Schilling-kNN

Construct the pooled k-nearest-neighbour graph once. The primary choice is `k=1`; the sensitivity panel uses `k in {1,5,10}`. Fix whether the graph is directed or symmetrised before running the benchmark. Use the same-label or cross-label edge count as the statistic and calibrate with pooled point-label permutations.

Do not choose `k` from the final rejection results. If the graph has ties, use deterministic index ordering. Report the number of directed or undirected edges and the chosen k value.

### 2.2 Secondary methods

#### Rosenbaum-CrossMatch

Construct a minimum-weight non-bipartite matching of the pooled points and count cross-arm pairs. Use the exact conditional permutation distribution when available or Monte Carlo label permutations otherwise. The matching is computed once per replication and never recomputed inside the label loop.

Use a pinned, documented matching implementation. If the required matching dependency is not already present, place it in an optional benchmark extra rather than changing the core package silently. If odd pooled cardinality is supported, document the unmatched-point rule.

#### Wasserstein or OT two-sample statistic

Implement one clearly defined computational version, preferably a fixed-regularisation entropic Wasserstein statistic or a fixed-projection sliced-Wasserstein sensitivity. The regularisation, number of projections, projection seed, and transport solver must be fixed before labels and recorded in every result.

This method is useful because it is geometrically interpretable, but it must not be promoted to a universally identifying point-law test if the selected finite approximation does not have that property. Use pooled point-label permutation for calibration.

#### Classifier two-sample test

Use a sample-split classifier test with a fixed model family, such as logistic regression and a small random forest as separate variants. The training/test split must be independent of the treatment labels or be part of a predeclared stratified protocol. Hyperparameters are fixed or selected only on training data.

The simplest valid implementation is to fit on a training subset and evaluate held-out classification accuracy. For exact calibration, retrain under the permuted training labels, or use a carefully justified held-out-label permutation conditional on the fixed training fit. The chosen calibration argument must be written down before the benchmark. Report the classifier class, split fraction, random seed, training time, and whether tuning was performed.

### 2.3 Methods that must not be ranked as full point-law tests

Hotelling's two-sample statistic, coordinate-wise t-tests, mean-distance tests, and finite moment tests may be useful diagnostic baselines, but they target restricted nulls such as equality of means or covariances. They may be included in an appendix labelled `restricted-moment-null`, but they must not enter the main `H0^law` ranking.

The existing persistence-diagram competitors in `tda2s/benchmarks/` also must not be mixed into the point-law ranking merely because their input started as a point cloud. Unless a method retains and tests the raw point law, it belongs in a separate diagram-law or topological-summary panel.

## 3. Frozen comparison protocol

### 3.1 Shared design constants

The following values are locked before the final fleet is generated.

| Quantity | Primary value | Sensitivity values |
|---|---:|---:|
| significance level | `0.05` | `0.01`, `0.10` for calibration diagnostics only |
| block size for RawBlockMMD and SC-B | `m=25` | `m in {1,2,5,10,25,50}` |
| core replications | `500` per cell | `200` for broad exploratory grids, `1000` for selected size cells |
| common random permutations | `199` per replication | `39` reproduction panel, `999` cheap-method precision panel |
| hybrid weights | `alpha in {0.25,0.50,0.75,1.00}` | none chosen post hoc |
| primary ambient dimension | `d=2` | `d in {1,5,10,20,50}` where the DGP is defined |
| primary block kernel | Hilbert-Gaussian unordered-bag kernel | point and bag bandwidth multipliers `{0.25,0.5,1,2,4}` |

The existing Phase 5 reproduction panel must first be rerun with the original seeds, families, and `39` permutations so that the new benchmark has a direct compatibility check with Table 8. The main comparison then uses the common `199`-permutation protocol. Exact enumeration is used for tiny samples whenever the group size is manageable, and the result records `exact_enumeration=True`.

The Monte Carlo interval for a rejection rate is a two-sided 95 percent Clopper-Pearson interval unless the report explicitly declares another interval. A difference in rejection rates must also receive a paired or independent Monte Carlo interval, using the shared cloud seed to exploit paired comparisons wherever possible.

### 3.2 Shared randomisation and fairness rules

For each replication, generate each cloud once from a deterministic seed keyed by `(benchmark_version, family, n0, n1, dimension, effect_size, replication)`. All methods in the cell receive the same clouds. Method-specific permutation seeds are separate and recorded.

Point-level methods use all points unless a method's definition requires otherwise. RawBlockMMD, HybridBlockMMD, SC-A-Block, and SC-B use the one frozen disjoint partition per arm, discard and report the remainder, and expose `K0`, `K1`, and `K0+K1`.

The report must contain both an all-point comparison and an effective-sample-size comparison. It must not pretend that ten raw blocks and 250 individual points are the same number of observations. The `m=1` RawBlockMMD sensitivity is the direct bridge between the point-level and block-level constructions.

Every feature or tuning parameter must be label-independent. In particular, the following are prohibited in the confirmatory benchmark: bandwidth chosen to maximise rejection; alpha chosen after seeing power; a partition selected using arm labels; classifier hyperparameters selected on the test split; and persistence parameters changed for different treatment arms.

### 3.3 Required output fields

Every replication record, including refusals and skipped cells, must contain at least:

```text
benchmark_version
design_hash
family
family_role
family_description
method
method_variant
target_null
validity_regime
sampling_unit
n0
n1
d
m
K0
K1
effective_sample_size_total
unused_points0
unused_points1
n_permutations
exact_enumeration
permutation_group
statistic
pvalue
rejected
kernel_or_distance
bandwidth_or_tuning
alpha
cloud_seed
partition_seed
permutation_seed
runtime_seconds
peak_rss_bytes
status
failure_reason
```

Additional method-specific fields are allowed, but the common fields cannot be renamed. A method that refuses a cell because `n_a<m`, `K_a<1`, a matching is infeasible, or a classifier split is undefined must produce a refusal record rather than silently dropping the cell.

## 4. DGP benchmark matrix

The existing Phase 5 DGP constructors and seed conventions are reused. Do not redesign the core matrix unless an implementation incompatibility is documented.

### 4.1 Core cells

The following cells are required for every primary method that can run at the indicated sample size.

| Cell | True relation | Interpretation |
|---|---|---|
| iid null | `P0=P1=Uniform([0,1]^2)` | point-law size gate |
| translated weak barcode null | `P1` is a translation of `P0` | `P0 != P1` but the locked Vietoris--Rips barcode law agrees; point-law power versus barcode-law calibration diagnostic |
| same-support density | same square support, different continuous density | density sensitivity without a support-topology change |
| four-atom square | `p=(.25,.25,.25,.25)` versus `q=(.70,.10,.10,.10)` | discrete occupancy and collision validation |
| topology alternative | filled disk versus noisy circle with matched gross moments | moderate topological alternative |

For the four-atom cell, print the existing exact `m=2` barcode-law total variation `0.27` and the `m=25` all-atoms-seen lower bound `0.201160`. These are DGP validation facts, not performance claims about RawBlockMMD.

The translated cell must be reported with target labels. For SC-B, its rejection rate is a barcode-null size diagnostic. For RawBlockMMD and the other point-law methods, its rejection rate is power against a point-law alternative. It must never be labelled simply `null_rejection_rate` without the target column beside it.

### 4.2 Existing robustness cells

Reuse contamination, unequal cardinality, anisotropic noise, boundary truncation, and the deliberate overlapping-block negative control. The original Poisson, inhomogeneous Poisson, Cox-clustered, and hard-core process cells may be included as a separate dependence panel.

The equal-law versions of contamination, anisotropic noise, boundary transformation, and unequal sample cardinality are diagnostics for whether the permutation implementations behave as expected under a shared iid law. The process cells are not size gates because the iid exchangeability argument does not apply to dependent point processes.

The overlap panel must remain a negative control. The agent must not call the number of overlapping blocks an effective sample size, and the confirmatory APIs must continue to refuse overlapping partitions and repeated-partition aggregation.

### 4.3 New difficulty cells

Add the following controlled sensitivity dimensions after the core reproduction passes.

#### Small sample sizes

Use

```text
n0=n1 in {10,15,20,25,30,50,75,100,125,150,250,500}
```

For the locked `m=25` method, values below `n=25` are explicit refusals. Values with `K_a<5` are diagnostic only because the block permutation distribution is too coarse to support a useful 0.05 test. Report the exact minimum attainable p-value or p-value grid for each cell. Do not call a zero-rejection rate at `K0=K1=1` evidence of calibration.

For smaller `n`, compare point-level methods, RawBlockMMD at valid smaller block sizes, and the `m=1` raw sensitivity. The goal is to quantify the power loss caused by the locked block size, not to hide it.

#### Ambient dimension and irrelevant noise

Use embedded two-dimensional alternatives in dimensions `d in {2,5,10,20}` and, if memory permits, `d=50`. Add independent irrelevant coordinates with fixed variance. Include sparse mean or density changes in one coordinate and dense changes across all coordinates.

The raw Gaussian bandwidth must be either fixed by the design scale or selected using the pooled unlabeled sample only. The benchmark must include a bandwidth sensitivity plot because MMD and energy methods can lose power as dimension increases.

#### Effect-size curves

Replace single alternatives by parameterised curves. Use translation magnitude `delta`, density-mixture weight `epsilon`, mean shift `delta`, topological noise level `sigma`, circle radius, and rare-component mass as separate axes. Use at least five values per axis, including a null or near-null endpoint. Report rejection curves with Monte Carlo intervals rather than a single power number.

#### Unequal sample sizes and unequal cloud cardinalities

Use arm-size ratios `1:1`, `1:2`, and `1:4`, both under `P0=P1` and under selected alternatives. Preserve the original arm counts in every permutation. For RawBlockMMD report `K0`, `K1`, and both remainders. For point-level methods report `n0`, `n1`, and the exact permutation group.

#### Outliers and measurement noise

Vary contamination fractions and outlier radius, and add isotropic or anisotropic measurement noise. Include both common contamination, which should preserve the iid null if generated identically in both arms, and differential contamination, which is an alternative. DTM-Rips may be included for the persistence diagnostic, but it must not be introduced into RawBlockMMD's raw kernel without a separate target statement.

#### Dependence and spatial structure

Use the existing process cells only to show where iid point-label or block-label permutation becomes unsupported. The expected result is not a calibrated size guarantee. Add a `validity_status=unsupported_dependence` field and keep these rows out of the main size/power ranking.

## 5. Implementation work packages

### WP-B0: freeze the benchmark contract

Objective: create the design hash, method registry, target table, seed convention, schema, and refusal policy.

Tasks: inspect existing APIs; preserve current SC-A and SC-B behavior; add target-aware method metadata; add the common output record; add exact tiny-sample tests; and write a one-page method registry before running large experiments.

Guardrail: do not let the word `point-cloud` silently mean a persistence diagram. The registry must distinguish `individual iid point`, `frozen disjoint raw block`, `frozen disjoint barcode block`, and `diagram observation`.

Verification: rerunning a cell with a different shard assignment must reproduce the same cloud and statistic given the same method seed.

### WP-B1: implement raw point-law baselines

Objective: implement PointMMD-Gaussian and EnergyDistance with the same permutation and result APIs as the existing methods.

Tasks: cache pooled point kernels or distances; verify invariance to point ordering; preserve unequal arm sizes; test exact small samples; test bandwidth and distance metadata; and compare PointMMD at `m=1` with its independently implemented point-kernel statistic.

Guardrail: do not use the block partition in these point-level methods. That would turn a baseline into another version of RawBlockMMD.

Verification: under the iid null, the rejection rate must lie within the registered size band after accounting for p-value discreteness; under a controlled mean or density shift, the statistic must increase monotonically on average as the effect size increases.

### WP-B2: implement graph and matching baselines

Objective: implement FriedmanRafsky-MST, Schilling-kNN, and, if dependencies permit, Rosenbaum-CrossMatch.

Tasks: define the graph or matching exactly; cache it outside the permutation loop; resolve ties deterministically; verify that permutations preserve `n0` and `n1`; and add a small exact enumeration test.

Guardrail: a graph statistic is not an MMD and does not inherit characteristic-kernel identification. Use the appropriate consistency language.

Verification: under a pure label permutation of one fixed pooled point set, every implementation must agree with an independently computed enumeration on a tiny example.

### WP-B3: implement optional OT and classifier baselines

Objective: add one fixed Wasserstein/OT variant and one sample-split C2ST only after profiling their runtime and calibration.

Tasks: freeze solver and tuning parameters; write the calibration argument; record training/test split and classifier; test label-independent tuning; and refuse unsupported small cells.

Guardrail: do not report an arbitrary finite projection or classifier as a universally identifying point-law test.

Verification: compare the method against PointMMD and EnergyDistance on translation, sparse shift, dense shift, and high-dimensional noise cells, with runtime and failure rates.

### WP-B4: extend the common tournament

Objective: extend `experiments/phase5ab_block_tournament.py` or create a thin point-law companion runner that reuses its DGP constructors, seeds, summary functions, and output conventions.

Tasks: add method registry dispatch; generate common clouds once per cell; support local smoke runs; write replication and summary parquet; calculate paired Monte Carlo intervals; and produce the target-aware comparison table.

Guardrail: do not create a second incompatible result schema. If a separate runner is necessary for optional methods, it must emit the same schema and use the same design hash.

Verification: the original Phase 5 reproduction cells reproduce the existing SC-B and RawBlockMMD rates within Monte Carlo uncertainty, and the new methods appear beside them with explicit target columns.

### WP-B5: run the Colab fleet

Objective: execute the predeclared core and sensitivity matrix using independent, self-contained notebooks.

Tasks: profile all method-family cells locally; estimate cost; construct a balanced manifest for up to 40 shards; generate notebooks with embedded or cloned source; checkpoint each shard; download one parquet per shard; and provide a local aggregation command.

Guardrail: do not use nested workers inside a permutation loop. Do not allow two notebooks to write the same output file.

Verification: a shard rerun produces byte-equivalent replication records after sorting by the stable key, and aggregation refuses duplicate or conflicting design hashes.

### WP-B6: analyse and report

Objective: create the final machine-readable tables, plots, and a concise report that distinguishes target, effective sample size, calibration, power, and unsupported regimes.

Required plots: size and power curves versus `n`; effect-size curves; topology versus density power; translated-cell target separation; `m` versus effective sample size; dimension versus power; bandwidth sensitivity; runtime and peak RSS; and overlap/dependence failure diagnostics.

Guardrail: never rank methods in a table without printing their null target beside the method name.

Verification: every headline number in the report can be regenerated from the aggregated parquet files without rerunning the methods.

## 6. Colab execution design

The long benchmark should use independent CPU notebooks. The proposed fleet is `N_SHARDS=40`, with each notebook assigned a deterministic subset of cells. The notebooks should target approximately 60 to 180 minutes of actual computation, not artificial waiting. If profiling shows a shard below one hour, merge it with an adjacent shard; if it exceeds eight hours, split it. A notebook may stop early only at a checkpoint and must be resumable.

Each notebook must be self-contained. It must install or verify dependencies, obtain the project source, set `OMP_NUM_THREADS=1` and equivalent BLAS limits before importing NumPy, define its shard parameters in the first code cell, and write only to its own output path. GPU use is not required for the core methods. Use at most the runtime's available CPU workers and do not create nested process pools.

The first cell should expose:

```python
SHARD_ID = 0
N_SHARDS = 40
SEED_ROOT = "phase5ab-pointlaw-v1"
WALL_BUDGET_MIN = 450
N_WORKERS = 2
```

The notebook must checkpoint every 10 to 25 completed replications to a shard-specific parquet file and write a manifest containing the design hash, code commit or source hash, package versions, and completed cell keys. If a cell fails, write a refusal or error record and continue only when the failure is declared nonfatal by the manifest.

The final notebook cell must use the safe Colab download fallback:

```python
output_file = SHARD_OUT
try:
    from google.colab import files
    files.download(output_file)
    print("Downloaded:", output_file)
except Exception as exc:
    print("(Not on Colab / download skipped):", exc)
```

The local aggregator must concatenate shard files, reject duplicate or conflicting design hashes, and write:

```text
results/phase5ab_pointlaw_replications.parquet
results/phase5ab_pointlaw_summary.parquet
results/phase5ab_pointlaw_comparison.parquet
results/phase5ab_pointlaw_manifest.json
results/phase5ab_pointlaw_*.png
```

The manifest should report which of the 40 shards are present and which cells remain incomplete. Partial aggregation is allowed for debugging but cannot trigger a decision gate.

## 7. Decision gates

### Gate B1: target and literature validity

Pass only if every method has an explicit target, sampling unit, calibration argument, and limitation. SC-B must remain under `H0,m^bar`; RawBlockMMD and the required raw-data baselines must be under `H0^law` when their stated assumptions hold.

### Gate B2: basic iid size

Pass only if the primary `H0^law` methods have rejection rates compatible with `alpha=0.05` in the uniform-square iid null. Use the existing Phase 5 band `[0.03,0.08]` for the 500-replication gate, together with Monte Carlo intervals. Cells with too few blocks or too-coarse exact permutation groups are diagnostic and cannot be used to claim or reject general calibration.

### Gate B3: point-law power

RawBlockMMD passes the comparative gate only if it is competitive with the strongest raw-data baseline on density, location, sparse, dense, and rare-feature alternatives, and if any claimed improvement has a Monte Carlo interval. A win over SC-A alone is not sufficient because SC-A is a persistence-only representation despite its point-law null label.

### Gate B4: topology retention

The method must retain useful power on the existing filled-disk versus noisy-circle alternative. The hybrid may be reported as useful if it improves or preserves power relative to RawBlockMMD, but no hybrid winner is declared if its advantage occurs only after post-hoc alpha selection.

### Gate B5: barcode-law separation

On the translated cell, SC-B should remain near its barcode-law size while raw point-law methods may reject. The table must call this `translated_pointlaw_power` for RawBlockMMD and `translated_barcode_null_rejection` for SC-B. This gate fails if the report collapses the two interpretations.

### Gate B6: small-sample honesty

Pass only if the report shows where `m=25` becomes unusable because `K_a` is too small, rather than silently switching to a smaller `m`. The primary method may be recommended only with a stated minimum effective block count or a warning that the p-value grid is coarse.

### Gate B7: robustness scope

Pass only if overlap and dependence cells are explicitly marked unsupported or negative controls. No method receives a validity claim there without a separate dependence-aware theorem and implementation.

### Gate B8: computation

Report runtime, peak RSS, failure rate, and completed-cell coverage. A method that is statistically attractive but repeatedly exceeds the Colab budget must be presented as computationally limited, not silently omitted.

## 8. Claims register

| Claim | Status | Required evidence |
|---|---|---|
| Conditional pooled-label permutation is valid for any measurable point statistic under iid `P0=P1`. | `provable` | exact exchangeability argument and tiny enumeration test |
| Frozen disjoint block-label permutation is valid under iid `P0=P1`. | `provable` | block exchangeability argument, remainder accounting, and null simulation |
| Equality of the primary RawBlockMMD block embeddings identifies `P0=P1` under the stated characteristic-kernel model. | `adapt` | characteristic-kernel argument plus finite bag-kernel counterexample for the simple average kernel |
| Point-level Gaussian MMD and Euclidean energy distance are full point-law-sensitive methods under their assumptions. | `provable/adapt` | source-grounded theory note and controlled alternatives |
| MST, kNN, and cross-match are useful raw-data equality tests. | `adapt` | exact statistic definitions, source conditions, and benchmark results |
| RawBlockMMD materially improves density power over SC-A. | `empirical only` | paired Monte Carlo interval against SC-A, with the caveat that SC-A is persistence-only |
| RawBlockMMD materially improves density power over the strongest established raw-data baseline. | `empirical only` | primary gate B3 |
| Topological augmentation improves point-law testing. | `conjecture/empirical only` | predeclared alpha panel; no claim if the benchmark does not support it |
| RawBlockMMD is valid for overlapping blocks or dependent occurrence processes. | `OPEN, not claimed` | separate theory required |
| RawBlockMMD directly answers the ecological niche-equivalency question. | `OPEN, not claimed` | ecological sampling, covariate, and target audit required |

## 9. Ecology-specific handoff

The ecology application described in `RESEARCH_DECISION.md` constructs a species niche hypervolume from occurrence locations, environmental covariates, PCA, and PC1--PC3. That application cannot automatically be treated as the Regime-I iid benchmark.

If there are at least 30 independent species-level clouds per arm and the points within each cloud can reasonably be treated as iid draws after a frozen external standardisation, the point-law methods can be used as a methodological comparison. If the scientific unit is one species versus another, the data are `n=1` versus `n=1` at the cloud level and the current iid two-sample interpretation is not justified. Spatial dependence, sampling effort, temporal clustering, and background availability must be addressed first.

The Warren--Glor--Turelli niche-equivalency permutation and Broennimann PCA-env overlap measures should be included only in a separate applied panel. They answer an ecological niche-equivalency or overlap question, not automatically `P0=P1` for raw environmental occurrence points. Their presence in an ecology figure must not be used to rank the generic point-law methods.

## 10. Immediate execution order

1. Freeze WP-B0 and create the method registry, design hash, common schema, and refusal records.
2. Run the original Phase 5 reproduction panel unchanged.
3. Implement PointMMD-Gaussian and EnergyDistance, then run tiny exact tests and a 20-replication local profile.
4. Implement FriedmanRafsky-MST and Schilling-kNN, then add CrossMatch if the matching dependency is stable.
5. Run a cost profile over one representative PH cell, one raw cell, one graph cell, one small-`n` cell, and one high-dimensional cell.
6. Generate the 40-shard manifest so every notebook is independently reproducible and targets at least one hour but remains below the runtime cap.
7. Run the core 500-replication fleet, aggregate, and inspect only calibration and failure diagnostics before running the broad sensitivity fleet.
8. Run small-`n`, dimension, bandwidth, effect-size, unequal-cardinality, contamination, and dependence panels.
9. Produce the target-aware comparison table and plots, then update the Phase 5 report only from aggregated machine-readable results.
10. Make the recommendation conditional on Gate B3. If RawBlockMMD does not beat the strongest raw-data baseline, report it as a valid blockwise point-law test with an effective-sample-size tradeoff, not as a universal winner.

## 11. Literature and novelty stance

The exact name `RawBlockMMD` and the project API are project-specific. The statistical ingredients are not new. The benchmark must therefore describe the contribution as a proposed fixed-disjoint-block point-law test and test whether that construction is useful relative to established raw-data tests. A novelty claim requires a broader literature search than the sources listed in the companion reading list.

The companion file `docs/phase5ab_point_law_benchmark_literature.md` maps every method in this plan to a verified source and provides acquisition links. Before writing a paper claim, the agent should extract the exact theorem or consistency conditions used by each baseline into a short source cheatsheet.
