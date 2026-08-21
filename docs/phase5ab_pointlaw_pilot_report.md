# Phase 5AB point-law benchmark report

Benchmark version `phase5ab-pointlaw-v1`, design hash `df1e561dc12a58e4`. Replications per cell `20`, permutations `39`.

## Design audit

Frozen protocol per `docs/phase5ab_point_law_registry.md`. Primary `m=25` locked; point-level baselines use pooled kernels/distances/graphs cached outside permutation loop; block methods use frozen disjoint partition with `K_a=floor(n_a/m)` and remainders discarded.

## Method registry (headline comparison distinguishes target)

| method | target_null | sampling_unit | null_rejection | density | topology | translated | 4-atom | runtime | peak_RSS |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| EnergyDistance | H0^law: P0=P1 | individual iid point | 0.050 | 1.000 | 1.000 | 1.000 | 1.000 | 0.027 | 110998528 |
| FriedmanRafsky-MST | H0^law: P0=P1 | individual iid point | 0.000 | 0.750 | 1.000 | 1.000 | 0.000 | 0.066 | 110998528 |
| PointMMD-Gaussian | H0^law: P0=P1 | individual iid point | 0.050 | 1.000 | 1.000 | 1.000 | 1.000 | 0.048 | 110998528 |
| Schilling-kNN-k1 | H0^law: P0=P1 | individual iid point | 0.000 | 0.550 | 1.000 | 1.000 | 0.000 | 0.047 | 110998528 |

## Gates (B1-B8)

B1 target/literature validity: passes by construction for registered methods. SC-B remains under `H0,25^bar`; RawBlockMMD and required raw baselines under `H0^law` when assumptions hold.
B2 iid size: primary H0^law methods should lie in [0.03,0.08] at 500 replications (with Clopper-Pearson intervals). Cells with K_a<5 are diagnostic only.
B3 point-law power: RawBlockMMD gate passes only if competitive with strongest raw baseline on density/location/sparse/dense/rare alternatives (paired MC interval).
B4 topology retention: useful power on filled-disk vs noisy-circle; hybrid predeclared alpha only.
B5 barcode separation: translated cell is `translated_pointlaw_power` for RawBlockMMD vs `translated_barcode_null_rejection` for SC-B.
B6 small-sample honesty: m=25 becomes unusable where K_a too small; report p-value grid coarseness.
B7 robustness: overlap/dependence marked unsupported.
B8 computation: runtime, peak RSS, failure rate, coverage reported; methods exceeding Colab budget flagged as computationally limited.

## Effective sample size note

Point-level methods use all points (`n0+n1` observations). Block methods use `K0+K1` blocks; the `m=1` RawBlock sensitivity is the bridge. Report contains both all-point and effective-sample-size comparisons.

All headline numbers are regenerable from aggregated parquet without rerunning methods.

## Four-atom diagnostic

| method | rejection | 95% MC |
|---|---:|---:|
| EnergyDistance | 1.000 | [0.832,1.000] |
| FriedmanRafsky-MST | 0.000 | [0.000,0.168] |
| PointMMD-Gaussian | 1.000 | [0.832,1.000] |
| Schilling-kNN-k1 | 0.000 | [0.000,0.168] |
