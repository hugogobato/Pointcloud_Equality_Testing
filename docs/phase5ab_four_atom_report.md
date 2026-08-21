# Phase 5AB block-method evaluation

This run used `100` replications per cell and `39` block-label permutations. It reuses the Phase 5C DGP constructors and stable seed convention.

## Design audit conclusion

`RawBlockMMD`, `HybridBlockMMD` with alpha greater than zero, and `SC-A-Block` declare `H0^law: P0=P1`. Their sampling unit is a frozen disjoint block, and the effective sample size is `K0+K1`. `SC-B` remains a barcode-law test. Hybrid alpha equal to zero is reported only as a barcode-law diagnostic.

The raw primary kernel is Gaussian on the characteristic point-kernel mean embedding of the unordered bag. Equality of its block embeddings identifies the iid block law and then the point law. The simple average pairwise kernel is not characteristic for unrestricted bag distributions, so it is not used as the primary identified kernel.

## Required exact DGP validation

For the existing four-atom square witness, the m=2 barcode-law total variation is `0.27`, and the m=25 all-atoms-seen lower bound is `0.201160`. These validate the DGP distinction and are not properties of the raw block method.

## Primary comparison

| method | target | sampling_unit | m | K0 | K1 | null_rejection_rate | density_power | topology_power | translated_null_rejection_rate | overlap_rejection_rate | runtime | peak_memory |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HybridBlockMMD-a0.50 | H0^law: P0 = P1 (raw characteristic fixed-size block representation, m=25, under iid point sampling) | frozen disjoint m-point block | 25 | 10.0 | 10.0 | nan | nan | nan | nan | n/a | 0.049 | 55990
| RawBlockMMD | H0^law: P0 = P1 (raw characteristic fixed-size block representation, m=25, under iid point sampling) | frozen disjoint m-point block | 25 | 10.0 | 10.0 | nan | nan | nan | nan | n/a | 0.015 | 55545
| SC-A | H0^law: P0 = P1 | individual iid point | 25 | n/a | n/a | nan | nan | nan | nan | n/a | 10.651 | 3359124
| SC-A-Block | H0^law: P0 = P1 (raw characteristic fixed-size block representation, m=25, under iid point sampling) | frozen disjoint m-point block | 25 | 10.0 | 10.0 | nan | nan | nan | nan | n/a | 0.015 | 55337
| SC-B | H0,25^bar: Phi^25_0:1(P0) = Phi^25_0:1(P1) | disjoint m-point block | 25 | 10.0 | 10.0 | nan | nan | nan | nan | n/a | 0.037 | 43594

## Gate interpretation

Gate 1, target validity, passes by construction for the raw and alpha-positive hybrid candidates. Gate 2 requires the iid-null rate to lie in the predeclared Phase 5 band `(0.03, 0.08)`; the values above are empirical and require their Monte Carlo intervals in the summary parquet. Gate 3 is a comparison claim, not an assumption: density power must be judged against SC-A with intervals. Gate 4 asks whether the hybrid retains topology power. Gate 5 requires the translated cell to distinguish raw point-law sensitivity from SC-B barcode-law calibration. Gate 6 rejects overlap as a confirmatory construction.

The primary m=25 effective counts are `floor(n_a/25)`, and unused remainders are stored in the replication parquet. The m-sensitivity panel includes m in {1, 2, 5, 10, 25, 50}; m=1 is the point-level raw-block baseline. No post-hoc m or alpha choice is used for the headline comparison.

Claims about finite-sample permutation validity, disjoint-block independence under iid sampling, ordering invariance, and point-law identification of the primary raw kernel are mathematical or directly checkable. Relative power, runtime, memory, and robustness are empirical. Overlapping blocks, dependent point processes, and label-dependent tuning remain outside the validity claim.

## Four-atom square density cell

The evaluated discrete cell uses the existing square support with p=(0.25,0.25,0.25,0.25) and q=(0.70,0.10,0.10,0.10). The exact m=2 barcode-law TV is 0.27, and the m=25 all-atoms-seen lower bound is 0.201160. The observed m=25 rejection rates are:

| method | rejection rate | 95% MC interval |
|---|---:|---:|
| HybridBlockMMD-a0.25 | 1.000 | [0.964, 1.000] |
| HybridBlockMMD-a0.50 | 1.000 | [0.964, 1.000] |
| HybridBlockMMD-a0.75 | 1.000 | [0.964, 1.000] |
| HybridBlockMMD-a1.00 | 1.000 | [0.964, 1.000] |
| RawBlockMMD | 1.000 | [0.964, 1.000] |
| SC-A | 0.000 | [0.000, 0.036] |
| SC-A-Block | 1.000 | [0.964, 1.000] |
| SC-B | 0.040 | [0.011, 0.099] |
