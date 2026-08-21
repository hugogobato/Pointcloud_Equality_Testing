# Phase 5A design lock: exactly two observed point clouds

**Status:** GO for the synthetic methods-paper benchmark under Regime I.
The future real-data application remains PENDING CLASSIFICATION until its
sampling metadata are recorded. This lock does not authorize a spatial,
ecological, or Bayesian inference claim by analogy.

**Decision record:** the repository contains declared i.i.d. synthetic
point-cloud generators, but no real-data observation protocol. The Regime-I
decision is therefore scoped to the methods-paper synthetic benchmark. A
co-author must sign off on any extension to an external data set.

## Locked target table

| Item | Locked choice | Scope and refusal rule |
|---|---|---|
| Observation regime | Regime I, i.i.d. metric-measure sampling | $Y_{a,j}\stackrel{\mathrm{iid}}{\sim}P_a$ within arm, independent arms, or a declared randomized design. Reject a call that supplies only one fixed array with no sampling model. |
| Sampling unit | Point | Disjoint blocks of $m$ points are the only barcode replicates. Overlapping subclouds are exploratory output, never confirmatory replication. |
| Primary null | $H_{0,25}^{\mathrm{bar}}:\Phi^{25}_{0:1}(P_0)=\Phi^{25}_{0:1}(P_1)$ | Equality of the joint degree-0 and degree-1 Vietoris--Rips barcode law for a fixed 25-point i.i.d. subcloud. |
| Secondary null | $H_0^{\mathrm{law}}:P_0=P_1$ | Strongest-simple full point-law baseline. It is not relabelled as the primary topological null. |
| Subcloud size | $m=25$ | Frozen before any rejection. At $n_a\in\{250,500,1000\}$, the confirmatory disjoint-block counts are $K_a\in\{10,20,40\}$. |
| Filtration | Vietoris--Rips, radius parameter | The ambient metric and raw distance units are fixed. Radius convention: an edge between two points at distance $d$ enters the filtration at radius $d/2$, and every filtration in the shared PH module reports this radius scale. Essential classes are dropped and diagrams use the standard birth/death convention. |
| Degrees | $d\in\{0,1\}$, jointly | A degree-specific sensitivity may be reported only as a predeclared secondary analysis. |
| Primary object | Probability law on the joint diagram space $\mathcal D_0\times\mathcal D_1$ | No claim of equality of empirical clouds, full point laws, supports, or unspecified topology. |
| SC-B diagnostic kernel | Tensor-product degree-tagged persistence scale-space embedding with Gaussian RKHS factors, $h=0.10$ | No median-heuristic bandwidth in confirmatory inference. Characteristicness is asserted on bounded per-degree diagram classes under the Phase 5D conditions. |
| Discrepancy norm | RKHS norm for the SC-B diagnostic; equality for the primary null | A finite-vector norm is a different target and must be labelled as such. |
| Equivalence margin | None for the primary equality test | Equivalence is not silently substituted for equality. A future margin $\varepsilon$ requires a new lock and a scientific scale argument. |
| Nuisances | Point order, common ambient isometries, and representation of the same metric cloud | Density, intensity, anisotropic deformation, boundary truncation, and global scale are differences under this raw-metric lock. |
| Nuisance estimation | None in the Regime-I synthetic benchmark | Measurement error, thinning, registration, or covariate adjustment require a new observation model and cannot be hidden inside SC-A or SC-B. |
| Asymptotics | $n_0,n_1\to\infty$, $n_1/(n_0+n_1)\to\lambda\in(0,1)$, $m=25$ fixed | Effective barcode sample size is $K_a=\lfloor n_a/25\rfloor$, not the number of overlapping draws or Monte Carlo resamples. |
| Calibration scope | To be selected in Phase 5B from methods compatible with Regime I | No Phase 5B resampling or Bayesian model is authorized by this document alone. |

The primary estimand is

\[
  \Phi^{25}_{0:1}(P_a)
  = \mathcal L\left(\operatorname{Dgm}_0(\mathsf{VR}(Y_{a,1:25})),
                       \operatorname{Dgm}_1(\mathsf{VR}(Y_{a,1:25}))\right).
\]

The secondary point-law baseline is $P_0=P_1$. The logical relation is one
way: $P_0=P_1$ implies the barcode-law null, while the converse fails in
general. The translation witness in
scripts/phase5_estimand_witnesses.py is an exact example of a point-law
difference invisible to any metric-invariant barcode summary. The equilateral
occupancy witness shows that a finite summary can agree at one subcloud size
while the fixed-$m$ barcode law differs at the locked size.

## Regime routing contract

Every future candidate method must declare one of the following before it is
run:

1. iid_metric_measure, with point-level replication and the primary target
   above;
2. stationary_mixing_process, with an increasing window, spatial blocks, and
   a named process-level vector;
3. fixed_cloud, which returns only a deterministic discrepancy and no
   inferential quantity; or
4. explicit_generative_model, which returns model-relative posterior
   quantities only after a likelihood and prior-predictive audit.

An incompatible declaration is an error, not a warning. In particular,
iid_metric_measure cannot be inferred from an array's shape, and a large
number of overlapping subclouds cannot upgrade fixed_cloud to an i.i.d.
sample.

## Gate 5A decision

The gate passes for the current synthetic scope because the data-generating
mechanism supplies genuine i.i.d. point replication and the target is
well-defined with fixed $m$, filtration, degree set, and nuisance policy.
The decision is:

> **GO: Regime I for the synthetic methods-paper benchmark.**

For a future real-data application, the gate is not inherited. If the points
come from one stationary or mixing spatial process, rerun the lock under
Regime II and replace the primary null by the frozen normalized process
vector. If the clouds are fixed objects with no sampling model, record
KILL inference and report only deterministic distances and perturbation
sensitivity. If domain experts provide a defensible likelihood and prior
predictive constraints, record a new Regime-IV conditional lock before any
Bayesian result.

## Claim ledger for Phase 5A

| Claim | Tag | Verification or limitation |
|---|---|---|
| The fixed-$m$ barcode law is a well-defined target under the Regime-I model. | provable | Definition plus measurability assumptions; stability is supported by Blumberg et al. |
| Disjoint blocks of i.i.d. points are independent barcode draws. | provable | Direct consequence of independence of disjoint point sets. |
| Overlapping subclouds are not additional independent replicates. | provable | Shared points induce dependence; the script and the effective-size formula make the distinction explicit. |
| $P_0=P_1\Rightarrow H_{0,25}^{\mathrm{bar}}$. | provable | Pushforward of equal laws through the fixed-size PH map. |
| $H_{0,25}^{\mathrm{bar}}\Rightarrow P_0=P_1$. | conjecture, rejected | The translation witness gives a direct counterexample, so this implication must not be claimed. |
| Equality of a finite topological summary implies the barcode-law null. | conjecture, rejected | The equilateral occupancy construction has equal $m=2$ summary and unequal locked-$m$ barcode laws. |
| A fixed cloud supports a frequentist p-value without a sampling model. | conjecture, rejected | Two generative families assign positive likelihood to the same arrays, with arbitrarily different tails. |

The conjecture labels in the last three rows make the rejected claims
visible. They are not permitted claims of the project.

## Phase 5 boundary

The implemented Regime-I methods attach this target and regime declaration to
every result. SC-A is the full point-law label-permutation baseline, SC-B is
the hardened disjoint fixed-$m$ barcode comparison, and SC-C is finite-summary
inference that must not be presented as a test of $H_{0,25}^{\mathrm{bar}}$.
SC-D and SC-E remain out of scope until a separate Regime-II or Regime-IV lock
is signed.
