# Phase 5D: SC-B hardening

## Scope and production target

Phase 5C selected SC-B as the only target-matched candidate for the scoped
Regime-I benchmark. The production target is the fixed-size joint barcode-law
null

\[
H_{0,25}^{\mathrm{bar}}:
  \Phi^{25}_{0:1}(P_0)=\Phi^{25}_{0:1}(P_1),
\]

where \(\Phi^{25}_{0:1}(P)\) is the law of the pair of Vietoris--Rips
diagrams in degrees zero and one obtained from 25 independent draws from
\(P\). The fixed partition is the source of replication. If \(n_a\) points
are observed in arm \(a\), the effective barcode sample size is

\[
K_a=\lfloor n_a/25\rfloor,
\]

and the remaining points are discarded for the confirmatory statistic. A
Monte Carlo permutation count is not an additional barcode replicate.

The production implementation is `tda2s.tests.single_cloud.sc_b_production_test`.
The lower-level `sc_b_disjoint_mmd` function remains available for explicitly
labelled sensitivity values of \(m\), but only the production entry point is
the frozen Phase 5D procedure.

## Assumptions

Let \(Y_{a,1},\ldots,Y_{a,n_a}\) be the points in arm \(a\), and let
\(B_m(y_{1:m})\) denote the measurable pair of finite Vietoris--Rips
diagrams in degrees zero and one. The finite-sample result uses the following
assumptions.

**A1, point-level replication.** Within each arm the points are iid from
\(P_a\), and the two arms are independent. The synthetic benchmark satisfies
this assumption. A spatial point process, a fixed cloud, and a Bayesian model
are not substitutes for A1.

**A2, fixed data-independent partition.** For each arm, a partition
\(I_{a1},\ldots,I_{aK_a}\) consists of disjoint sets of size 25. The
partition may be deterministic or randomly generated, but its mechanism is
independent of all point coordinates and labels. Conditioning on a realized
partition is therefore legitimate. Choosing a partition after inspecting
point geometry is outside the validity claim.

**A3, common fixed-size null.** Under the null,
\(Q_0=Q_1\), where \(Q_a=\mathcal L\{B_{25}(Y_{a,1:25})\}\). This is the
barcode-law null, not the stronger point-law null \(P_0=P_1\).

**A4, measurable fixed statistic.** The filtration, degrees, metric units,
kernel bandwidth, and MMD statistic are fixed before the permutation. The
implementation computes all block diagrams once and applies permutations only
to the resulting diagram-level Gram matrix.

For finite-sample validity, no characteristicness or bounded-support
assumption is needed. Those conditions enter only the consistency statement.

## Finite-sample permutation validity

For a realized partition, define

\[
Z_{ak}=B_{25}\bigl((Y_{a,j})_{j\in I_{ak}}\bigr),
\qquad k=1,\ldots,K_a.
\]

By A1 and A2, the \(Z_{ak}\) are iid from \(Q_a\) within each arm. Under
A3, the pooled vector \((Z_{01},\ldots,Z_{0K_0},Z_{11},\ldots,Z_{1K_1})\)
is iid from the same law. Consequently, conditional on the realized
partition, its labels are exchangeable over all \(\binom{K_0+K_1}{K_0}\)
splits.

Let \(T(z,g)\) be any measurable statistic, including the biased squared MMD
used by SC-B, and let \(G\) be the set of all label splits with \(K_0\)
labels in arm zero. The exact p-value is

\[
p_{\mathrm{ex}}(z,g)=
\frac{1}{|G|}\sum_{g'\in G}
  \mathbf 1\{T(z,g')\geq T(z,g)\}.
\]

The rank of the observed statistic among the exchangeable orbit is
uniform up to ties. Therefore

\[
\Pr_{H_{0,25}^{\mathrm{bar}}}
  \{p_{\mathrm{ex}}\leq\alpha\mid I_{01:K_0},I_{11:K_1}\}
\leq\alpha.
\]

The same inequality holds without conditioning after averaging over the
data-independent partition mechanism. This proves finite-sample level control
for the one-partition procedure. It does not require \(K_0=K_1\), and it
continues to hold for any fixed measurable discrepancy. The role of the
kernel is power and target separation, not the exchangeability argument.

The production Monte Carlo p-value uses \(B\) independently sampled label
permutations and the Phipson--Smyth correction

\[
p_{\mathrm{MC}}=
\frac{1+\sum_{b=1}^{B}
  \mathbf 1\{T(z,g_b)\geq T(z,g)\}}{B+1}.
\]

The observed split is thus treated as one member of the exchangeable rank
sample. Conditional on the realized block diagrams, this randomized-rank
construction is conservative under the same null. Exact enumeration is
available only when the complete split set is below the configured safety
limit.

## The joint characteristic kernel

For one homology degree, the implementation uses

\[
k_d(F,G)=\exp\{k_{\sigma}(F,G)\},
\]

where \(k_{\sigma}\) is the reflected persistence scale-space kernel and the
bandwidth is fixed at \(h=0.10\). Kwitt, Huber, Niethammer, Lin, and Bauer,
Proposition 2, establish universality with respect to the bottleneck/Wasserstein
diagram metric on bounded diagram classes with bounded coordinates and bounded
multiplicity. In particular, the mean embedding is injective on laws over
such a class.

The production kernel for the joint diagram is

\[
K\bigl((F_0,F_1),(G_0,G_1)\bigr)
  = k_0(F_0,G_0)\,k_1(F_1,G_1).
\]

This is the tensor-product kernel. If each factor is characteristic on its
diagram class, the tensor-product mean embedding is injective on the product
law under the corresponding bounded-support conditions. The fixed finite
cloud size bounds the number of possible simplices and hence the diagram
multiplicity. A common bounded metric support supplies a uniform coordinate
bound. For unbounded point laws, the production test remains finite-sample
valid, but this note makes no automatic characteristicness claim unless the
chosen kernel is separately known to be characteristic on the actual support.

The distinction from an averaged degree kernel is load-bearing. If

\[
K_{\mathrm{avg}}((F_0,F_1),(G_0,G_1))
  =\tfrac12\{k_0(F_0,G_0)+k_1(F_1,G_1)\},
\]

then its mean embedding records the two marginal laws but not their
dependence. Two joint laws can have identical degree marginals and different
cross-degree coupling, giving zero population MMD under \(K_{\mathrm{avg}}\)
while remaining distinct. The implementation uses the product kernel to avoid
this target failure.

## Consistency

Assume A1--A4, a common bounded per-degree diagram support on which the two
factor kernels are characteristic, and

\[
K_0,K_1\longrightarrow\infty,
\qquad
\frac{K_1}{K_0+K_1}\longrightarrow\lambda\in(0,1),
\]

with the subcloud size fixed at 25. The product kernel is bounded on the
locked support. Under any fixed alternative \(Q_0\neq Q_1\), characteristicness
gives

\[
\operatorname{MMD}^2_K(Q_0,Q_1)>0.
\]

The biased empirical MMD converges to this positive quantity, whereas the
permutation distribution is calibrated to equal-label assignments and its
critical values converge to zero on the corresponding null scale. Hence the
permutation test is consistent against every fixed alternative in the
characteristic support class. The asymptotic sequence is specifically
\(n_a\to\infty\) with fixed \(m=25\); a growing \(m\) changes the estimand and
requires a new argument.

The result does not imply consistency for equality of the full point laws.
It also does not imply detection of support-topology differences that leave
the fixed-size barcode law unchanged. Those are target limitations, not
calibration failures.

## Repeated partitions: decision and safe boundary

The production API deliberately uses one frozen partition and refuses a
repeated-partition aggregation. Reusing points across partitions creates
dependent diagram summaries, so averaging MMD statistics, pooling permutation
draws, Fisher-combining the resulting p-values, or taking an uncorrected
minimum p-value has no validity claim in this package.

A conservative future extension is possible. If \(L\) partitions are fixed
before looking at the point coordinates, and each produces a valid marginal
p-value \(p_\ell\), then

\[
p_{\mathrm{Bonf}}=\min\{1,L\min_{1\leq\ell\leq L}p_\ell\}
\]

is valid by the union bound, regardless of the dependence between partitions.
This is not implemented because it was not part of the signed Phase 5C
design, and it would trade the production method's single frozen partition for
a new multiplicity contract. A future repeated-partition method must freeze
the number and construction of partitions, report the realized overlap and
effective block counts, and use a predeclared aggregation rule.

## Verification record

The implementation and focused checks are in
`tda2s/tests/single_cloud.py`, `tests/test_single_cloud.py`,
`scripts/phase5d_scb_hardening.py`, and
`tests/test_phase5d_scb.py`. The checks include a two-point joint-law witness
that separates the old averaged kernel from the product kernel, exact and
Monte Carlo null calibration, effective block-count accounting, fixed-target
refusals, and explicit refusal of repeated-partition aggregation.

The Phase 5C robustness and overlap results remain evidence about the original
registered fleet. The Phase 5D numerical check reruns the target-matched SC-B
cells with the corrected joint kernel before the production claim is carried
forward.
