# Phase 5AB final report

## Scope and target

The primary locked construction uses one frozen, data-independent, disjoint
partition into blocks of size `m=25` per cloud. The inferential observations
are the retained blocks, with `K_a=floor(n_a/25)` and the unused remainder
discarded and reported. The permutation group contains all pooled block-label
splits preserving `K0` and `K1`. Under the Regime-I iid point model, block
labels are exchangeable conditional on the frozen partition.

`RawBlockMMD`, `SC-A-Block`, and `HybridBlockMMD` with `alpha>0` declare
`H0^law: P0=P1`. SC-B remains the barcode-law test
`H0,25^bar: Phi^25_{0:1}(P0)=Phi^25_{0:1}(P1)`. No result is collapsed across
these targets.

The primary raw representation is a Gaussian kernel on the point-kernel mean
embedding of each unordered block. This is characteristic on fixed-size bags
under the stated Euclidean iid model. The simple average pairwise kernel is
available only as a downgraded sensitivity variant because it is not
characteristic for unrestricted bag distributions.

SC-A-Block forms blocks separately within the original arms and then applies
the pooled block-label permutation. With the same partition and kernel it is
algebraically equivalent to RawBlockMMD, so it is a framing comparison rather
than an additional independent source of evidence.

## 500-replication core gate, n=250 per arm

The high-replication block run used 39 random block-label permutations per
replication. Values below are rejection rates with exact two-sided 95%
Clopper-Pearson Monte Carlo intervals.

| method | target | K0 | K1 | iid null | density | topology | translated weak null | overlap negative control | runtime (s) | peak RSS (MB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SC-B | barcode law | 10 | 10 | 0.042 [0.026, 0.063] | 0.988 | 0.998 | 0.054 [0.036, 0.078] | 0.200 | 0.045 | 162.7 |
| RawBlockMMD | point law | 10 | 10 | 0.064 [0.044, 0.089] | 0.998 | 1.000 | 1.000 [0.993, 1.000] | 1.000 | 0.013 | 162.7 |
| SC-A-Block | point law | 10 | 10 | 0.060 [0.041, 0.085] | 0.996 | 1.000 | 1.000 [0.993, 1.000] | n/a | 0.013 | 162.7 |
| HybridBlockMMD, alpha=0.50 | point law | 10 | 10 | 0.046 [0.029, 0.068] | 0.996 | 1.000 | 0.504 [0.459, 0.549] | n/a | 0.052 | 162.7 |
| SC-A baseline, 20-rep RSS run | point law | n/a | n/a | 0.050 [0.001, 0.249] | 0.500 | 1.000 | 0.000 [0.000, 0.168] | n/a | 2.7 | 168.6 |

The hybrid alpha sensitivity was informative: translated-null rejection was
0.096 for alpha `0.25`, 0.504 for alpha `0.50`, 0.988 for alpha `0.75`, and
1.000 for alpha `1.00`. This is expected because the translated construction
preserves metric persistence diagrams but changes raw coordinates. It confirms
that alpha-positive hybrids are not barcode-only tests.

The basic iid-null rates for all primary block methods lie in the predeclared
Phase 5 size band `[0.03, 0.08]`. The continuous same-support density cell
shows a large gain over the 20-replication SC-A baseline, whose rejection rate
was 0.50 [0.272, 0.728], compared with 0.998 [0.989, 1.000] for RawBlockMMD.
The baseline has a much smaller replication count and should not be treated as
a same-precision comparison. The SC-A RSS baseline was rerun separately with
20 replications to put its memory measurement on the same instrumentation
scale.

## Four-atom square validation

The evaluated discrete cell uses the existing square support with
`p=(0.25,0.25,0.25,0.25)` and `q=(0.70,0.10,0.10,0.10)`. The exact `m=2`
barcode-law total variation is `0.27`, and the `m=25` all-atoms-seen lower
bound is `0.201160`. These are DGP validation facts, not properties of the
raw method.

In the separate 100-replication `m=25` run, RawBlockMMD, SC-A-Block, and all
hybrid weights rejected in 1.00 of replications, SC-B rejected in 0.04, and
SC-A rejected in 0.00. The SC-A result is a useful warning that point-level
exchangeability alone does not guarantee power when its statistic uses only
persistence diagrams and the block-level occupancy signal is not retained.

## Effective sample size and m sensitivity

For `n_a=250`, the raw method’s effective arm counts were 250, 125, 50, 25,
10, and 5 for `m=1,2,5,10,25,50`, respectively. In the 100-replication raw
sensitivity run, iid-null rejection rates were 0.06, 0.08, 0.04, 0.04, 0.04,
and 0.09 in that order. Density power was 0.88, 0.94, 1.00, 1.00, 1.00,
and 0.96; topology power was 1.00, 1.00, 1.00, 1.00, 1.00, and 0.97.
Translated-null rejection was 1.00, 1.00, 1.00, 1.00, 1.00, and 0.98.
The headline choice remains `m=25`; the other values are sensitivity results,
not post-hoc tuning.

## Overlap and robustness

Overlapping blocks were evaluated only as negative controls. At nominal overlap
fractions `0, .25, .50, .75, .90`, RawBlockMMD negative-control rejection was
`0.05, 0.55, 0.85, 1.00, 1.00`; SC-B’s corresponding rates were
`0.05, 0.20, 0.20, 0.20, 0.35`. These reused blocks are dependent and are not
additional effective observations. The confirmatory APIs refuse overlapping
partitions and repeated-partition aggregation.

The 20-replication robustness panel reuses contamination, unequal cardinality,
anisotropic noise, and boundary truncation. It is diagnostic rather than a new
sharp-null validity gate. Dependent point processes and data-dependent feature
tuning remain unsupported.

## Decision gates and claim status

Gate 1, target validity, passes for the raw and alpha-positive hybrid methods.
Gate 2, iid calibration, passes for the 500-replication primary block rates.
Gate 3, density sensitivity, passes empirically against the available SC-A
baseline, with the replication-count qualification above. Gate 4, topology
retention, passes in this benchmark for both RawBlockMMD and the hybrid. Gate 5,
the translated diagnostic, passes because SC-B remains calibrated while raw
and hybrid methods react according to alpha. Gate 6 passes only when overlap
and dependence are treated as unsupported or negative-control regimes.

The following are proved or directly checkable: iid disjoint-block
independence, conditional exact permutation validity for any measurable block
statistic, remainder accounting, ordering invariance, and
`P0^m=P1^m => P0=P1`. The characteristicness of the Hilbert-Gaussian bag
kernel is an adaptation of the standard characteristic-kernel argument to the
empirical mean-embedding image of fixed-size bags. Relative power, robustness,
runtime, and memory are empirical. No claim is made for dependent point
processes, overlapping blocks, or arbitrary non-Euclidean observation models.

## Conclusion

RawBlockMMD is a valid point-law test under the declared iid metric-measure
model, with an honest loss from `n_a` points to `floor(n_a/25)` block
observations at the locked setting. It materially improves density sensitivity
over the existing SC-A persistence-statistic baseline in the continuous cell
and remains powerful on the registered topology alternative. HybridBlockMMD
retains the point-law target for every alpha greater than zero while exposing a
controlled raw-versus-persistence tradeoff. SC-B remains the valid barcode-law
test and should not be described as a full point-law test.
