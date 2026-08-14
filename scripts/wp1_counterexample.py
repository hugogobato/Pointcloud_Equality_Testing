"""WP1.1 numerical instantiation: the six-null implication and its witnesses.

Instantiates the two structural facts of
``theory/WP1_estimands_identification.md``:

  W1 -- the plan's counterexample to the converse: H0^out holds, H0^dist fails.
        One persistent H0 cluster (a 2-blob cloud) splits into two (a 3-blob
        cloud) while the mean power-weighted silhouette is preserved.
  W2 -- independence witness: H0^dist holds, H0^out fails.
        Two diagram laws with equal *expected persistence measure* but
        different mean silhouettes (a multiplicity mixture; the covariate-
        mixing non-commutation of Saki & Faghihi 2603.14169, Remark 4).

Three layers, in increasing realism:

  1. Law-level W1/W2 with random merge scale ``d``: the exact objects of the
     theorem, sampled directly as diagrams.  Here the permutation tests are
     well-posed (within-group variability is present), so size and power are
     measured by Monte Carlo.
  2. Cloud-level exact witness: deterministic 12-gon blobs plus persistence
     thresholding ``tau`` (kills the tiny within-blob classes present in both
     groups).  Realization-by-realization: 2-blob clouds give H0 = {(0, d)}
     and 3-blob clouds give H0 = {(0, d), (0, d)} with d = (sep - 2 rho) / 2
     exactly, so the mean-silhouette gap is exactly zero.  Point estimates
     only (permutation tests are degenerate here: zero within-group
     variability makes the null mis-centered).
  3. Gaussian-blob diagnostic: noisy geometry moves the two H0 classes to the
     two smallest order statistics of three pair-merge scales, so exact mean
     preservation is lost through a second-order bias.  Quantified here;
     Phase 4.4 must use layer 2 (or thresholded tight clusters) for its
     "power = alpha" claim.

Implementation remark (Phase 2.3 / 7 must know): ``gudhi.bottleneck_distance``
is pathologically slow (tens of ms per call) on *identical* diagrams, where
the optimal distance is exactly zero (CGAL degenerate path).  RT's pairwise
O(N^2) loop then hangs for minutes.  This script therefore runs the RT wrapper
only on diagrams that are all distinct (the Gaussian diagnostic).

Usage
-----
    rtk uv run python scripts/wp1_counterexample.py

Reused by Phase 4.4: the ``split_cluster_cloud`` generator lives in
``tda2s/dgp/clouds.py``.
"""
from __future__ import annotations

import argparse

import numpy as np

from tda2s.benchmarks.rt import test_rt
from tda2s.dgp.clouds import split_cluster_cloud
from tda2s.ph import compute_diagrams
from tda2s.resample import p_value, permutation_test
from tda2s.vec import persistence_measure, silhouette

D_MERGE = (3.0 - 2 * 0.15) / 2.0  # exact merge scale of the 12-gon blobs
TAU = 0.3  # persistence threshold: kills within-blob classes (death ~0.04)


def _sil(diagrams, interval=(0.0, 2.0)):
    return np.stack([silhouette([d], interval=interval, r=3.0)[0] for d in diagrams])


def _meas(diagrams, n_bins=32):
    return np.stack([persistence_measure([d], n_bins=n_bins)[0].ravel() for d in diagrams])


def _perm_tests(sil, meas, labels, n_perm, rng):
    """Silhouette-sup and expected-measure permutation tests.

    Returns ``(p_sil, null_sd_sil, p_meas)``.
    """
    def stat_sil(lbl):
        m = sil[lbl == 1].mean(axis=0) - sil[lbl == 0].mean(axis=0)
        return float(np.max(np.abs(m)))

    def stat_meas(lbl):
        m = meas[lbl == 1].mean(axis=0) - meas[lbl == 0].mean(axis=0)
        return float(np.abs(m).sum())

    obs_sil, null_sil = permutation_test(stat_sil, labels, n_perm, rng)
    obs_meas, null_meas = permutation_test(stat_meas, labels, n_perm, rng)
    return (p_value(obs_sil, null_sil, "greater"), float(np.std(null_sil)),
            p_value(obs_meas, null_meas, "greater"))


def _mc_rep_w1(rng, n_per_group, n_perm, seed):
    """One replication of law-level W1 (random d, per-cloud equal deaths)."""
    rng = np.random.default_rng(seed)
    ds = rng.normal(D_MERGE, 0.08, size=2 * n_per_group)
    ds = np.clip(ds, 0.9, 1.8)
    d0 = [np.array([[0.0, d]]) for d in ds[:n_per_group]]
    d1 = [np.array([[0.0, d], [0.0, d]]) for d in ds[n_per_group:]]
    sil = np.vstack([_sil(d0), _sil(d1)])
    meas = np.vstack([_meas(d0), _meas(d1)])
    labels = np.concatenate([np.zeros(n_per_group, dtype=int),
                             np.ones(n_per_group, dtype=int)])
    p_sil, null_sd, p_meas = _perm_tests(sil, meas, labels, n_perm, rng)
    gap = float(np.max(np.abs(sil[:n_per_group].mean(0) - sil[n_per_group:].mean(0))))
    dist = float(np.abs(meas[:n_per_group].mean(0) - meas[n_per_group:].mean(0)).sum())
    return p_sil, null_sd, p_meas, gap, dist


def _mc_rep_w2(rng, n_per_group, n_perm, seed):
    """One replication of law-level W2 (equal expected measure)."""
    rng = np.random.default_rng(seed)
    dgm1 = np.array([[0.0, 1.0], [0.0, 1.0]])
    d0 = [np.array([[0.0, 1.0]]) for _ in range(n_per_group)]
    d1 = [dgm1.copy() if rng.random() < 0.5 else np.zeros((0, 2)) for _ in range(n_per_group)]
    sil = np.vstack([_sil(d0), _sil(d1)])
    meas = np.vstack([_meas(d0), _meas(d1)])
    labels = np.concatenate([np.zeros(n_per_group, dtype=int),
                             np.ones(n_per_group, dtype=int)])
    p_sil, null_sd, p_meas = _perm_tests(sil, meas, labels, n_perm, rng)
    gap = float(np.max(np.abs(sil[:n_per_group].mean(0) - sil[n_per_group:].mean(0))))
    dist = float(np.abs(meas[:n_per_group].mean(0) - meas[n_per_group:].mean(0)).sum())
    return p_sil, null_sd, p_meas, gap, dist


def cloud_exact_witness(rng):
    """Deterministic-blob point estimates: gap exactly zero, measure apart."""
    rng = np.random.default_rng(rng)
    c0 = [split_cluster_cloud(n_per_blob=100, n_blobs=2, noise=0.15,
                              deterministic=True, rng=rng) for _ in range(200)]
    c1 = [split_cluster_cloud(n_per_blob=100, n_blobs=3, noise=0.15,
                              deterministic=True, rng=rng) for _ in range(200)]
    d0 = [np.array([[0.0, d] for d in compute_diagrams(c, homology_dims=(0,))[0][:, 1]
                    if d > TAU]).reshape(-1, 2) for c in c0]
    d1 = [np.array([[0.0, d] for d in compute_diagrams(c, homology_dims=(0,))[0][:, 1]
                    if d > TAU]).reshape(-1, 2) for c in c1]
    counts0 = sorted({len(d) for d in d0})
    counts1 = sorted({len(d) for d in d1})
    deaths0 = np.unique(np.concatenate([d[:, 1] for d in d0]))
    deaths1 = np.unique(np.concatenate([d[:, 1] for d in d1]))
    gap = float(np.max(np.abs(_sil(d0).mean(0) - _sil(d1).mean(0))))
    dist = float(np.abs(_meas(d0).mean(0) - _meas(d1).mean(0)).sum())
    return counts0, counts1, deaths0, deaths1, gap, dist


def gaussian_diagnostic(rng, n_per_group, n_perm):
    """Noisy-geometry variant: report the order-statistic bias."""
    rng = np.random.default_rng(rng)
    c0 = [split_cluster_cloud(n_per_blob=100, n_blobs=2, rng=rng) for _ in range(n_per_group)]
    c1 = [split_cluster_cloud(n_per_blob=100, n_blobs=3, rng=rng) for _ in range(n_per_group)]
    d0 = [np.array([[0.0, d] for d in compute_diagrams(c, homology_dims=(0,))[0][:, 1]
                    if d > TAU]).reshape(-1, 2) for c in c0]
    d1 = [np.array([[0.0, d] for d in compute_diagrams(c, homology_dims=(0,))[0][:, 1]
                    if d > TAU]).reshape(-1, 2) for c in c1]
    sil = np.vstack([_sil(d0), _sil(d1)])
    meas = np.vstack([_meas(d0), _meas(d1)])
    labels = np.concatenate([np.zeros(n_per_group, dtype=int),
                             np.ones(n_per_group, dtype=int)])
    p_sil, null_sd, p_meas = _perm_tests(sil, meas, labels, n_perm, rng)
    gap = float(np.max(np.abs(sil[:n_per_group].mean(0) - sil[n_per_group:].mean(0))))
    p_rt = test_rt([[d] for d in d0], [[d] for d in d1], metric="bottleneck",
                   statistic="within", n_perm=n_perm, seed=7)
    return dict(gap=gap, null_sd=null_sd, p_sil=p_sil, p_rt=p_rt, p_meas=p_meas)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-per-group", type=int, default=300)
    ap.add_argument("--n-perm", type=int, default=300)
    ap.add_argument("--mc-reps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print("== law-level W1: H0^out holds, H0^dist fails (random d) ==")
    w1 = [_mc_rep_w1(rng, args.n_per_group, args.n_perm, 1000 + k)
          for k in range(args.mc_reps)]
    p_sil, null_sd, p_meas, gap, dist = zip(*w1)
    print(f"  mean-silhouette gap / null SD : "
          f"{np.mean([g / max(s, 1e-12) for g, s in zip(gap, null_sd)]):.3f}"
          f"  (gap {np.mean(gap):.4f})")
    print(f"  expected-measure distance     : {np.mean(dist):.3f}")
    print(f"  outcome-level test rejection  : {np.sum(np.array(p_sil) < 0.05)}/{args.mc_reps}"
          f"  (size: expect ~{0.05 * args.mc_reps:.1f})")
    print(f"  distribution-level rejection  : {np.sum(np.array(p_meas) < 0.05)}/{args.mc_reps}"
          f"  (power: expect {args.mc_reps})")

    print("\n== law-level W2: H0^dist holds, H0^out fails (multiplicity mixture) ==")
    w2 = [_mc_rep_w2(rng, args.n_per_group, args.n_perm, 2000 + k)
          for k in range(args.mc_reps)]
    p_sil, null_sd, p_meas, gap, dist = zip(*w2)
    print(f"  mean-silhouette gap / null SD : "
          f"{np.mean([g / max(s, 1e-12) for g, s in zip(gap, null_sd)]):.3f}"
          f"  (gap {np.mean(gap):.4f})")
    print(f"  expected-measure distance     : {np.mean(dist):.3e}")
    print(f"  outcome-level test rejection  : {np.sum(np.array(p_sil) < 0.05)}/{args.mc_reps}"
          f"  (power: expect {args.mc_reps})")
    print(f"  distribution-level rejection  : {np.sum(np.array(p_meas) < 0.05)}/{args.mc_reps}"
          f"  (size: expect ~{0.05 * args.mc_reps:.1f})")

    print("\n== cloud-level exact witness (deterministic blobs + threshold "
          f"tau={TAU}) ==")
    counts0, counts1, d0u, d1u, gap, dist = cloud_exact_witness(rng)
    print(f"  H0 feature counts per cloud: group A {counts0}, group B {counts1}")
    print(f"  H0 death values: group A {np.round(d0u, 6)}, group B {np.round(d1u, 6)}")
    print(f"  mean-silhouette gap (exact)       : {gap:.3e}")
    print(f"  expected-measure distance (exact) : {dist:.3f}")

    print("\n== cloud-level Gaussian diagnostic (exactness is lost) ==")
    g = gaussian_diagnostic(rng, args.n_per_group, args.n_perm)
    print(f"  mean-silhouette gap / null SD : {g['gap'] / max(g['null_sd'], 1e-12):.2f}"
          f"  (gap {g['gap']:.4f}, null SD {g['null_sd']:.4f})")
    print(f"  outcome-level p / RT p / dist-level p : "
          f"{g['p_sil']:.3f} / {g['p_rt']:.3f} / {g['p_meas']:.3f}")
    print("  -> noisy geometry biases the two H0 classes toward the smaller")
    print("     pair-merge order statistics; use the deterministic DGP for the")
    print("     Phase 4.4 'power = alpha' claim.")


if __name__ == "__main__":
    main()