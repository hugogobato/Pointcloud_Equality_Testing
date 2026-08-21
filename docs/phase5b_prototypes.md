# Phase 5B prototype record

Phase 5B is implemented for the scoped Regime-I benchmark. The common entry
point is `tda2s.tests.single_cloud.run_single_cloud_test`. Each result reports
the declared regime, inferential target, statistic, p-value, null draws,
diagnostics, runtime, and Python-tracked peak memory.

SC-A is the strongest-simple pooled-point baseline. It computes the joint
degree-0 and degree-1 Vietoris--Rips diagrams for the observed split and for
every requested label split, then compares the two diagrams with the fixed
degree-tagged universal persistence scale-space kernel. Exact tiny-sample
enumeration is available through `exact=True`. Its target is `P0 = P1`, not
the fixed-size barcode-law null.

SC-B is the primary fixed-size barcode prototype. It freezes one random
partition per cloud, computes one joint diagram per disjoint block, and
calibrates a diagram-level MMD by permuting the block labels. The result
exposes `m`, `K0`, `K1`, unused-point counts, and the block indices. The
permutation loop performs no PH calculations. Overlapping blocks are rejected
as a confirmatory input. The joint kernel is the tensor product of the
degree-specific exponentiated persistence scale-space kernels. An additive or
averaged degree kernel would test only equality of the two degree marginals,
not equality of their joint law. The default `m=25` is the signed Phase 5A
target; other values are returned as explicitly unlocked sensitivity sizes.
The frozen production contract is documented in the Phase 5D usage guide.

SC-C is a finite-vector prototype. Its frozen vector contains persistent
Betti coordinates on one shared `(r,s)` grid for degrees 0 and 1, normalized
by cloud size. The pooled point bootstrap is calibrated under the finite
vector null. `sc_c_finite_vector(..., smoothing=True)` adds Gaussian kernel
jitter after point resampling, following the smoothed-bootstrap route of
Roycraft, Krebs, and Polonik (Annals of Statistics, 2023, DOI
`10.1214/23-AOS2277`). `sc_c_naive_bootstrap` is retained as a negative
control. Neither SC-C path is reported as a test of the full barcode law.

The source-aligned pilot is `scripts/phase5b_roycraft_pilot.py`. It uses two
independent binomial samples from the uniform law on the unit square, a
Vietoris--Rips radius filtration, and compares the smoothed and ordinary
bootstrap paths. Its output is a prototype implementation check, not a claim
to reproduce the source paper's full confidence-coverage tables.

The following checks are now available:

`pytest -q tests/test_single_cloud.py` verifies incompatible-regime refusal,
complete exact enumeration, disjointness and effective block counts, the
shared-grid Betti vector, candidate dispatch, and the naïve negative-control
label.

`python scripts/phase5b_roycraft_pilot.py` runs a small deterministic source-
aligned pilot. Full 500-replication Phase 5C selection remains gated on this
prototype layer and is not included in Phase 5B.

`python scripts/phase5b_iid_null_pilot.py` runs the SC-B basic i.i.d.-null
calibration pilot. It enumerates all block-label splits in each replication,
so the reported rejection rate is not confounded by random permutation draws.
Each replication also draws a fresh frozen partition, so the rate averages
over partition structures rather than conditioning on one fixed partition.
The 100-replication run at `alpha=0.05` reported a rejection rate of 0.04
(min p-value 1/70, the exact-enumeration lattice), inside the `[0.03, 0.08]`
Phase 5C size band. This is a prototype calibration, not the 500-replication
gate.
