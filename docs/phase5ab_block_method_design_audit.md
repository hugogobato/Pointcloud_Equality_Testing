# Phase 5AB block-method design audit

This note is written before implementation. The scope is Regime I, in which
the two observed clouds are independent iid samples from probability laws
`P0` and `P1` on a common separable metric space. The primary locked block
size is `m=25`.

## 1. Observation unit

The inferential observations are the retained, fixed-size, disjoint blocks
`B_{ak}=(Y_{a,(k-1)m+1},...,Y_{a,km})`, with
`K_a=floor(n_a/m)`. Under the iid model these blocks are independent within
and across arms. The unused `n_a-m K_a` points are discarded and reported.
The point coordinates inside a block are treated as an unordered bag by both
the raw and persistence representations.

The original sampling unit remains an individual point. The block is the
replicate used by the permutation test. This distinction matters for the
effective sample size and for the exchangeability argument.

## 2. Null hypothesis

The raw and hybrid candidates declare

`H0^law: P0 = P1`.

The persistence representation is auxiliary. For a hybrid weight
`alpha>0`, the raw component remains present, so the method is not silently
changed to the barcode-law null. The special case `alpha=0` is reported only
as the SC-B-style diagnostic target

`H0,m^bar: Phi^m_{0:1}(P0) = Phi^m_{0:1}(P1)`.

The mean-pairwise raw kernel is retained only as an explicitly labelled
sensitivity variant. The primary raw kernel is a characteristic kernel on
unordered fixed-size bags.

## 3. Permutation group

After one frozen disjoint partition per arm, the pooled block sample has
`K=K0+K1` observations. The exact permutation group is the set of all
subsets of `K0` pooled blocks, equivalently all binary label vectors with
exactly `K0` zeros and `K1` ones. Randomized calibration samples from this
group without recomputing raw features or persistence diagrams.

## 4. Exchangeability assumption

Under `P0=P1`, iid point sampling and a partition mechanism independent of
point coordinates imply that, conditional on the realized partition, all
retained blocks are iid from the same law `P^m`. Therefore block labels are
exchangeable over the stated group. This argument does not apply to
overlapping blocks, spatially dependent point processes, fixed clouds with
no sampling model, or data-dependent partition or bandwidth selection.

## 5. Why the raw representation identifies the point law

Let `k` be a bounded characteristic point kernel and let
`mu_B = m^{-1} sum_{x in B} phi(x)` be the empirical kernel mean embedding of
the bag. The primary raw block kernel is

`K_raw(B,B') = exp(-||mu_B-mu_B'||^2/(2 tau^2))`.

The Gaussian kernel on the Hilbert-space embedding is characteristic on the
law of `mu_B`. Since `k` is characteristic, equality of empirical embeddings
identifies the empirical measures, hence identifies unordered bags including
multiplicity. Consequently equality of the raw block-kernel embeddings
implies equality of the block laws `P0^m=P1^m`; equality of the first
marginals then implies `P0=P1`.

The simple average pairwise kernel

`K_mean(B,B') = m^{-2} sum_{x in B,y in B'} k(x,y)`

is not characteristic for arbitrary distributions of bags. For `m=2`, let
one bag law put probability one half on `{a,a}` and `{b,b}`, while another
puts probability one on `{a,b}`. Both have the same expected empirical
measure, so their `K_mean` embeddings agree although the bag laws differ.
Within the restricted iid product family, however, the population MMD under
`K_mean` reduces to the point-law MMD, so it remains point-law sensitive. The
implementation therefore uses the Hilbert-Gaussian bag kernel for the fully
identified primary claim and labels `K_mean` as a sensitivity statistic.

For `alpha>0`, a nonnegative sum of the characteristic raw kernel and the
positive-definite barcode kernel remains characteristic because zero hybrid
MMD forces zero raw MMD. The barcode term can add topological power without
replacing the raw point-law target.

## 6. Status of claims

The following are direct or standard arguments: disjoint iid blocks are
independent; conditional exact permutation calibration is valid for any
measurable block statistic; the marginal implication
`P0^m=P1^m => P0=P1`; the unorderedness of the raw construction; and the
finite counterexample to characteristicness of the simple average kernel.

The characteristicness of the Hilbert-Gaussian kernel is an adaptation of
the standard characteristic-kernel argument to the empirical mean-embedding
image of fixed-size bags. The finite-sample permutation implementation and
the no-persistence-in-the-loop invariant are directly verifiable software
properties. Power comparisons, relative stability, and whether the hybrid
recovers topology power at the cost of raw sensitivity remain empirical
claims. No claim is made for dependent point processes or overlapping blocks.

## 7. Information lost

The method discards the unused remainder points, so its confirmatory sample
size is `K0+K1`, not `n0+n1`. It discards within-block ordering, intentionally,
and compresses each bag through a fixed-bandwidth kernel mean embedding. The
raw embedding is injective on fixed-size bags in the stated kernel class, but
finite computation and finite samples still limit power. Persistence adds an
isometry-invariant geometric summary and therefore cannot by itself recover
ambient-coordinate information. With `alpha>0`, that barcode information loss
does not determine the hybrid target because the raw characteristic term is
still present.

## Design decision

The confirmatory candidates are `RawBlockMMD` with the characteristic
Hilbert-Gaussian bag kernel and `HybridBlockMMD` with predeclared
`alpha in {0.25, 0.50, 0.75, 1.00}`. `SC-A-Block` is the pooled block-label
implementation using the same cached raw Gram matrix as `RawBlockMMD`; it is
expected to be algebraically equivalent, and that equivalence is reported
rather than treated as a new source of independent evidence.
