# Phase 6 high-dimensional point-law benchmark report

Benchmark version `phase6-highdim-v1`, design hash `401912b22817e1af`. Replications per cell `30`, permutations `99`.

## Design audit

Frozen protocol: m=25 locked; d in [10, 20, 40]; n in [50, 500]; preproc in ['raw', 'pca3', 'pca4']; PH filtration vr; point bandwidth 0.3; bag bandwidths 0.1/0.25. SC-A omitted (expensive, not needed).

## Method registry (headline comparison distinguishes target)

| method | target_null | sampling_unit | null_rejection | density | topology | translated | 4-atom | sparse | dense | runtime | peak_RSS |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EnergyDistance | H0^law: P0=P1 | individual iid point | 0.133 | 0.333 | 0.167 | nan | nan | 1.000 | 1.000 | 0.567 | 135645949 |
| FriedmanRafsky-MST | H0^law: P0=P1 | individual iid point | 0.067 | 0.600 | 0.967 | nan | nan | 1.000 | 1.000 | 0.716 | 135645949 |
| HybridBlockMMD-a0.50 | H0^law: P0 = P1 (raw characteristic fixed-size block representation, m=25, under iid point sampling) | frozen disjoint m-point block | 0.000 | 0.567 | 0.767 | nan | nan | 0.033 | 0.033 | 0.386 | 135645949 |
| PointMMD-Gaussian | H0^law: P0=P1 | individual iid point | 0.100 | 1.000 | 1.000 | nan | nan | 1.000 | 1.000 | 0.570 | 135645949 |
| RawBlockMMD | H0^law: P0 = P1 (raw characteristic fixed-size block representation, m=25, under iid point sampling) | frozen disjoint m-point block | 0.100 | 0.033 | 0.367 | nan | nan | 0.833 | 1.000 | 0.098 | 135645949 |
| SC-B | H0,25^bar: Phi^25_0:1(P0) = Phi^25_0:1(P1) |  | 0.000 | 0.600 | 0.700 | nan | nan | 0.033 | 0.033 | 0.304 | 135645949 |
| Schilling-kNN-k1 | H0^law: P0=P1 | individual iid point | 0.133 | 0.567 | 0.933 | nan | nan | 1.000 | 1.000 | 0.487 | 135645949 |

## Gates (B1-B8) adaptation for high-d

B1 target/literature validity: passes by construction.
B2 iid size: primary H0^law methods should lie in [0.03,0.08] at 500 replications.
B3 point-law power vs d: evaluate dense vs sparse power decay with ambient dimension.
B4 topology retention vs d and after PCA.
B5 PCA separation: compare raw vs pca3 vs pca4 power; pca failure families should show collapse under pca3.
B6 small-sample honesty: m=25 at n=50 (K=2) is floor; report p-value grid.
B7 robustness: high-noise and pca failure diagnostics.
B8 computation: runtime, peak RSS per d.

## Effective sample size note

Point-level methods use n0+n1 observations. Block methods use K0+K1 blocks; m=25 locked.

## PCA note

PCA fitted on pooled unlabeled points (valid under H0). Variance explained reported per replication. pca_fail families embed signal in low-var subspace so pca3 should lose power while raw retains some.

All headline numbers are regenerable from aggregated parquet without rerunning methods.
