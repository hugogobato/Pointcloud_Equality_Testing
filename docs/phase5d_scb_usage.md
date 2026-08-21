# Phase 5D SC-B production usage

The production procedure applies only when the two arrays are iid point-level
samples from two metric-measure laws. The method does not infer this regime
from the shape of an array. Spatial clouds, fixed objects, and Bayesian
generative models require a separate observation-model lock.

Use the frozen entry point:

```python
from tda2s.tests.single_cloud import sc_b_production_test

result = sc_b_production_test(
    cloud0,                         # shape (n0, dimension)
    cloud1,                         # shape (n1, dimension)
    partition_seed=20260819,        # fixed before inspecting rejections
    n_perm=999,
    seed=20260820,
)

print(result["pvalue"])
print(result["inferential_target"])
print(result["diagnostics"]["effective_sample_size"])
```

The production contract fixes the Vietoris--Rips radius filtration, degrees
zero and one jointly, subcloud size `m=25`, and kernel bandwidth `0.10`. Each
arm must contain at least five complete disjoint blocks, so each cloud must
have at least 125 points. The actual effective sample sizes are returned as
`K0` and `K1`, with unused remainder counts returned as `remainder0` and
`remainder1`.

The returned p-value is an exact permutation p-value when `exact=True` and
the complete split set is below `max_exact_permutations`. Otherwise it uses
the corrected Monte Carlo rank p-value. Persistent homology is computed once
per disjoint block. The permutation loop reuses only the cached diagram Gram
matrix.

The partition seed or supplied block indices must be chosen independently of
the point coordinates. A supplied partition is accepted for reproducibility,
but the API cannot verify how it was constructed. Passing
`partition_is_data_independent=False` refuses the call. When a random
partition is drawn, arm zero uses `partition_seed` and arm one uses
`partition_seed + 1`; a supplied partition must itself contain at least five
complete 25-point blocks, matching the minimum enforced for drawn
partitions.

The following calls are refused: non-iid regimes, fewer than five complete
25-point blocks in either arm, data-dependent partitions, repeated-partition
aggregation, and any request to alter `m`, the filtration, the joint degree
set, or the confirmatory bandwidth through the production entry point. The
lower-level `sc_b_disjoint_mmd` function can run unlocked sensitivity values,
but those results must be labelled as sensitivity analyses and must not be
reported as the locked `H0,25^bar` test.

The result diagnostics record the target lock, sampling unit, partition
status, effective sample size, filtration, kernel, bandwidth, PH call count,
and unsupported regimes. The method is not a test of `P0=P1`; that stronger
point-law equality is the SC-A baseline. Nor is it a test of an unspecified
support-topology notion. It tests equality of the fixed-size joint barcode
law.

The implementation intentionally does not average results from multiple
partitions. A future pre-registered extension could use Bonferroni correction
over a fixed number of data-independent partitions, but an uncorrected
minimum, pooled permutation distribution, or Fisher combination is not
supported.
