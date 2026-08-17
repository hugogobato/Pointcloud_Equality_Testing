"""Phase 4 tests: the distribution-level test (C2).

Pins the four things the Phase 4 experiment relies on:

* the W1 and W2' witnesses are exact at the cloud level (mean silhouette
  preserved vs expected persistence measures apart, and the reverse), after
  the TAU threshold and the coordinate rounding that restores the exact
  merge scales from the alpha filtration's floating-point output;
* the estimator: the IPW plug-in of the expected measure is exact for the
  linear functional on the witness laws, and the AIPW pseudo-outcome
  coincides with it when the nuisances are the true conditional means;
* the statistic and the null: the measure contrast is blind to W2' in law,
  the MMD variant is blind to neither, the vectorised kernel-path null agrees
  with the direct pass, and the stratified permutation preserves the
  per-stratum label composition with frozen weights;
* a W2' calibration guardrail: at the known-propensity design the
  distribution-level permutation diagnostic is not strongly anti-conservative;
  this is not a size theorem for the weak expected-measure null.

The cloud-level witnesses are the same objects the experiment runs
(``experiments.phase4_separation``), so this file and the fleet cannot drift
apart.
"""

import numpy as np

from tda2s.dgp.clouds import merge_staircase_cloud, split_cluster_cloud
from tda2s.ph import compute_diagrams
from tda2s.tests.dist_level import (
    _arm_estimates,
    _draw_stratified_labels,
    _permutation_stats,
    dist_statistic,
    fit_dist,
    measure_features,
    mmd_kernel_matrix,
    stratified_permutation_test,
)
from tda2s.vec import persistence_measure, silhouette

from experiments.phase4_separation import (
    _threshold_diagrams,
    _w1_cloud,
    _w2p_cloud,
    INTERVAL,
    N_BINS,
    N_GON,
    NOISE,
    R,
    TAU,
)

L3 = 1.35 ** 3                     # the W1 expected-measure separation
W1_SIL_GAP = 0.0                   # mean normalized silhouette preserved exactly
W2P_SIL_GAP = 0.5444166687923362   # balanced W2' mean-silhouette sup gap (res 100)
W2P_MMD = 0.6314761786996302       # balanced W2' universal-kernel MMD


def _w1_sample(n_per_arm: int, seed: int = 0):
    """Deterministic W1 clouds with sorted labels (arm 0 first)."""
    rng = np.random.default_rng(seed)
    clouds = ([_w1_cloud(0, rng) for _ in range(n_per_arm)]
              + [_w1_cloud(1, rng) for _ in range(n_per_arm)])
    diagrams = _threshold_diagrams(
        [compute_diagrams(c, filtration="alpha", homology_dims=(0,))
         for c in clouds])
    labels = np.concatenate([np.zeros(n_per_arm, dtype=int),
                             np.ones(n_per_arm, dtype=int)])
    return diagrams, labels


def _w2p_balanced_diagrams():
    """The W2' laws as exact diagram arrays (the law-level witnesses)."""
    m0 = np.array([[0.0, 1.0], [0.0, 2.0]])
    lo = np.array([[0.0, 1.0], [0.0, 1.0]])
    hi = np.array([[0.0, 2.0], [0.0, 2.0]])
    return m0, lo, hi


# ---------------------------------------------------------------------------
# witness exactness at the cloud level

def test_w1_cloud_witness_is_exact():
    """2-blob vs 3-blob deterministic clouds: gap exactly 0, measure apart."""
    rng = np.random.default_rng(0)
    assert _w1_cloud(0, rng).shape == _w1_cloud(1, rng).shape == (36, 2)
    diagrams, labels = _w1_sample(20, seed=1)
    d0 = [d[0] for d, a in zip(diagrams, labels) if a == 0]
    d1 = [d[0] for d, a in zip(diagrams, labels) if a == 1]
    assert sorted({len(d) for d in d0}) == [1]
    assert sorted({len(d) for d in d1}) == [2]
    assert {round(float(v), 9) for d in d0 for v in d[:, 1]} == {1.35}
    assert {round(float(v), 9) for d in d1 for v in d[:, 1]} == {1.35}

    sil0 = np.stack([silhouette([d], interval=INTERVAL, r=R)[0] for d in d0])
    sil1 = np.stack([silhouette([d], interval=INTERVAL, r=R)[0] for d in d1])
    assert float(np.max(np.abs(sil1.mean(0) - sil0.mean(0)))) == 0.0

    meas0 = np.stack([persistence_measure([d], interval=INTERVAL, n_bins=N_BINS,
                                          weight=lambda p: abs(p[1] - p[0]) ** R)[0].ravel()
                      for d in d0])
    meas1 = np.stack([persistence_measure([d], interval=INTERVAL, n_bins=N_BINS,
                                          weight=lambda p: abs(p[1] - p[0]) ** R)[0].ravel()
                      for d in d1])
    dist = float(np.abs(meas1.mean(0) - meas0.mean(0)).sum())
    assert np.isclose(dist, L3, rtol=1e-9)


def test_w2p_cloud_witness_is_exact_in_law():
    """Staircase vs two-location mixture: measures equal, silhouettes apart."""
    rng = np.random.default_rng(2)
    counts = {0: [], 1: []}
    for _ in range(12):
        d0 = _threshold_diagrams(
            [compute_diagrams(_w2p_cloud(0, rng), filtration="alpha",
                              homology_dims=(0,))])[0][0]
        d1 = _threshold_diagrams(
            [compute_diagrams(_w2p_cloud(1, rng), filtration="alpha",
                              homology_dims=(0,))])[0][0]
        counts[0].append(len(d0))
        counts[1].append(len(d1))
        assert {round(float(v), 9) for v in d0[:, 1]} == {1.0, 2.0}
        assert set(np.round(d1[:, 1], 9)) <= {1.0, 2.0}
    assert sorted(set(counts[0])) == [2]
    assert sorted(set(counts[1])) == [2]

    m0, lo, hi = _w2p_balanced_diagrams()
    w = (lambda p: abs(p[1] - p[0]) ** R)

    def _meas(dgm):
        return persistence_measure([dgm], weight=w, interval=INTERVAL,
                                   n_bins=N_BINS)[0].ravel()

    balanced = 0.5 * _meas(lo) + 0.5 * _meas(hi)
    np.testing.assert_array_equal(balanced, _meas(m0))  # exact, bin for bin

    s0 = silhouette([m0], interval=INTERVAL, r=R)[0]
    s1 = 0.5 * silhouette([lo], interval=INTERVAL, r=R)[0] \
        + 0.5 * silhouette([hi], interval=INTERVAL, r=R)[0]
    assert np.isclose(float(np.max(np.abs(s1 - s0))), W2P_SIL_GAP, rtol=1e-9)

    from tda2s.tests.dist_level import _k_universal

    def _mmd(law0, law1, sigma=0.10):
        def _cross(a, b):
            return sum(pa * pb * _k_universal(da, db, sigma)
                       for pa, da in a for pb, db in b)
        return float(np.sqrt(max(_cross(law0, law0) - 2 * _cross(law0, law1)
                                 + _cross(law1, law1), 0.0)))

    assert np.isclose(_mmd([(1.0, m0)], [(0.5, lo), (0.5, hi)]), W2P_MMD,
                      rtol=1e-9)


def test_merge_staircase_cloud_has_exact_merge_scales():
    """The W2' arm-0 generator: deaths exactly (1, 2), realization-invariant."""
    rng = np.random.default_rng(3)
    deaths = set()
    for _ in range(8):
        cloud = merge_staircase_cloud([1.0, 2.0], noise=NOISE, n_gon=N_GON,
                                      rng=rng)
        dgm = _threshold_diagrams(
            [compute_diagrams(cloud, filtration="alpha", homology_dims=(0,))])[0][0]
        assert len(dgm) == 2
        deaths |= {round(float(v), 9) for v in dgm[:, 1]}
    assert deaths == {1.0, 2.0}


# ---------------------------------------------------------------------------
# grid discipline

def test_unpinned_per_diagram_binning_collapses_the_contrast():
    """The WP1 §6.5 collapse exists in the raw vec API and is blocked here.

    Two diagrams {(0,1),(0,1)} and {(0,2)} have equal total mass under the
    linear weight.  Binning each on its own derived grid puts the class(es)
    in the same relative bin, so the L1 contrast is zero; on the pinned grid
    the bins differ and the contrast is 4.
    """
    a = np.array([[0.0, 1.0], [0.0, 1.0]])
    b = np.array([[0.0, 2.0]])
    w = (lambda p: float(p[1] - p[0]))
    collapsed = persistence_measure([a], weight=w, n_bins=N_BINS)[0].ravel()
    collapsed_b = persistence_measure([b], weight=w, n_bins=N_BINS)[0].ravel()
    assert np.isclose(np.abs(collapsed - collapsed_b).sum(), 0.0, atol=1e-12)

    pinned = persistence_measure([a], weight=w, interval=INTERVAL,
                                 n_bins=N_BINS)[0].ravel()
    pinned_b = persistence_measure([b], weight=w, interval=INTERVAL,
                                   n_bins=N_BINS)[0].ravel()
    assert np.isclose(np.abs(pinned - pinned_b).sum(), 4.0, atol=1e-9)


def test_measure_features_are_pinned_to_the_shared_grid():
    """Features of one diagram are unchanged by the co-diagrams present.

    Every public entry point takes the explicit ``interval``; without it the
    grid would rescale per call and the same diagram would move bins when
    other units with larger deaths are present.
    """
    small = [np.array([[0.0, 1.0]])]
    large_set = [np.array([[0.0, 1.0]]), np.array([[0.0, 3.0], [0.0, 5.0]])]
    alone = measure_features(small, interval=INTERVAL, n_bins=N_BINS,
                             weight_power=R, homology_dim=0)
    among = measure_features(large_set, interval=INTERVAL, n_bins=N_BINS,
                             weight_power=R, homology_dim=0)
    np.testing.assert_array_equal(alone[0], among[0])

    # the same comparison through the raw vec API without an interval shifts
    alone_raw = persistence_measure([small[0]], n_bins=N_BINS)[0].ravel()
    among_raw = persistence_measure(large_set, n_bins=N_BINS)[0].ravel()
    assert not np.allclose(alone_raw, among_raw)


# ---------------------------------------------------------------------------
# estimator

def test_ipw_estimator_is_exact_on_the_w1_law():
    """IPW plug-in: T1 - T0 = 1.35**3 exactly at the oracle propensity 1/2."""
    diagrams, labels = _w1_sample(25, seed=4)
    n = len(labels)
    fit = fit_dist(diagrams, labels, np.zeros((n, 1)), np.full(n, 0.5),
                   method="measure", interval=INTERVAL, n_bins=N_BINS,
                   weight_power=R, homology_dim=0)
    assert np.isclose(dist_statistic(fit), L3, rtol=1e-9)
    t0, t1 = _arm_estimates(fit, fit.A.astype(float))
    assert np.isclose(float(np.abs(t1 - t0).sum()), L3, rtol=1e-9)
    # the estimated arms are the exact population means at n1 = n0
    assert np.isclose(t0.sum(), L3, rtol=1e-9)
    assert np.isclose(t1.sum(), 2.0 * L3, rtol=1e-9)


def test_aipw_with_correct_nuisances_coincides_with_the_estimand():
    """With the true conditional means the AIPW pseudo-outcome is exact.

    On the deterministic W1 sample the arm-conditional means are the arm
    means themselves, so the residuals vanish and the AIPW estimate equals
    the g-formula value 1.35**3 exactly.
    """
    diagrams, labels = _w1_sample(25, seed=5)
    n = len(labels)
    mu = measure_features(diagrams, interval=INTERVAL, n_bins=N_BINS,
                          weight_power=R, homology_dim=0)
    m0_true = np.tile(mu[labels == 0].mean(0), (n, 1))
    m1_true = np.tile(mu[labels == 1].mean(0), (n, 1))
    fit = fit_dist(diagrams, labels, np.zeros((n, 1)), np.full(n, 0.5),
                   method="measure", interval=INTERVAL, n_bins=N_BINS,
                   weight_power=R, homology_dim=0,
                   mu0_hat=m0_true, mu1_hat=m1_true)
    assert fit.is_aipw
    assert np.isclose(dist_statistic(fit), L3, rtol=1e-9)

    # Also pin the non-cancelling nuisance-contrast case, which catches a
    # sign error in the cached permutation formula that the exact-W1 case can
    # hide.
    dgm = np.array([[0.0, 1.0]])
    tiny_diagrams = [[dgm] for _ in range(4)]
    tiny_labels = np.array([0, 0, 1, 1])
    mu0_hat = np.full((4, 1), 0.2)
    mu1_hat = np.full((4, 1), 0.8)
    tiny_fit = fit_dist(tiny_diagrams, tiny_labels, np.zeros((4, 1)),
                        np.full(4, 0.5), method="measure", interval=INTERVAL,
                        n_bins=1, weight_power=1.0, mu0_hat=mu0_hat,
                        mu1_hat=mu1_hat)
    draws = np.array([tiny_labels, [0, 1, 0, 1]], dtype=float)
    direct = np.array([dist_statistic(tiny_fit, draw) for draw in draws])
    fast = _permutation_stats(tiny_fit, draws)
    np.testing.assert_allclose(fast, direct, rtol=1e-12, atol=1e-12)


def test_stability_transfer_bound_applies_a_supplied_w1_error():
    """The module applies, but does not estimate, the §5.4 transfer bound."""
    from tda2s.tests.dist_level import stability_transfer_bound

    assert np.isclose(stability_transfer_bound(0.25, lipschitz=2.0), 0.5)


# ---------------------------------------------------------------------------
# statistic and null mechanism

def test_dist_statistic_is_blind_to_w2p_in_law():
    """The measure contrast of the balanced W2' laws is exactly zero."""
    m0, lo, hi = _w2p_balanced_diagrams()
    diagrams = [[m0], [m0], [m0], [m0], [lo], [lo], [hi], [hi]]
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    fit = fit_dist(diagrams, labels, np.zeros((8, 1)), np.full(8, 0.5),
                   method="measure", interval=INTERVAL, n_bins=N_BINS,
                   weight_power=R, homology_dim=0)
    assert float(dist_statistic(fit)) == 0.0


def test_permutation_null_preserves_strata_and_frozen_weights():
    diagrams, labels = _w1_sample(20, seed=7)
    n = len(labels)
    fit = fit_dist(diagrams, labels, np.zeros((n, 1)), np.full(n, 0.5),
                   method="measure", interval=INTERVAL, n_bins=N_BINS,
                   weight_power=R, homology_dim=0)
    strata = np.zeros(n, dtype=int)
    draws = _draw_stratified_labels(labels.astype(float), strata, 64,
                                    np.random.default_rng(0))
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        assert np.all(draws[:, idx].sum(axis=1) == labels[idx].sum())
    # frozen weights: the propensity vector cached at fit time is the only
    # label-dependent input; a redraw of the same labels gives the same stats
    w0, w1 = fit._weights_for(labels)
    np.testing.assert_array_equal(w0, fit._weights_for(labels)[0])
    np.testing.assert_array_equal(w1, fit._weights_for(labels)[1])
    null = _permutation_stats(fit, draws, batch_size=16)
    assert null.shape == (64,)
    assert np.all(null >= 0.0)


def test_dist_permutation_pvalue_mechanics():
    diagrams, labels = _w1_sample(20, seed=8)
    n = len(labels)
    fit = fit_dist(diagrams, labels, np.zeros((n, 1)), np.full(n, 0.5),
                   method="measure", interval=INTERVAL, n_bins=N_BINS,
                   weight_power=R, homology_dim=0)
    out = stratified_permutation_test(fit, np.zeros(n, dtype=int),
                                      n_perm=199, seed=9)
    assert 1.0 / 200.0 <= out["pvalue"] <= 1.0
    assert np.isclose(out["statistic"], dist_statistic(fit))
    assert out["null"].shape == (199,)
    assert int(out["n_strata"]) == 1


def test_vectorized_mmd_null_agrees_with_the_direct_pass():
    """The cached-kernel quadratic-form null equals per-draw kernel calls."""
    rng = np.random.default_rng(10)
    diagrams = []
    for _ in range(24):
        arm = int(rng.random() < 0.5)
        diagrams.append(_threshold_diagrams(
            [compute_diagrams(_w1_cloud(arm, rng), filtration="alpha",
                              homology_dims=(0,))])[0])
    labels = np.zeros(24, dtype=int)
    labels[::2] = 1
    fit = fit_dist(diagrams, labels, np.zeros((24, 1)), np.full(24, 0.5),
                   method="mmd", homology_dim=0)
    draws = _draw_stratified_labels(labels.astype(float), np.zeros(24, dtype=int),
                                    12, np.random.default_rng(11))
    direct = np.array([dist_statistic(fit, draws[b]) for b in range(12)])
    from tda2s.tests.dist_level import _mmd_stats_from_kernel

    fast = _mmd_stats_from_kernel(fit, draws, mmd_kernel_matrix(fit))
    np.testing.assert_allclose(fast, direct, rtol=1e-10, atol=1e-12)


def test_mmd_kernel_path_is_used_by_the_permutation_test():
    diagrams, labels = _w1_sample(20, seed=12)
    n = len(labels)
    fit = fit_dist(diagrams, labels, np.zeros((n, 1)), np.full(n, 0.5),
                   method="mmd", homology_dim=0)
    out = stratified_permutation_test(fit, np.zeros(n, dtype=int),
                                      n_perm=99, seed=13,
                                      kernel_matrix=mmd_kernel_matrix(fit))
    assert 1.0 / 100.0 <= out["pvalue"] <= 1.0
    assert out["statistic"] > 0.0          # MMD sees the W1 multiplicity
    assert np.all(out["null"] >= 0.0)


# ---------------------------------------------------------------------------
# oracle size on the W2' null

def test_w2p_permutation_diagnostic_is_not_anti_conservative():
    """The W2' permutation diagnostic is not strongly anti-conservative.

    The W2' arm-1 law is a mixture over two locations, so H0^dist holds in
    law but the units are not exchangeable; the frozen-nuisance permutation
    test is only approximately calibrated.  Sixty replications at n = 40
    enforce only an anti-conservatism guardrail (rate <= 0.12); this is not a
    two-sided Monte Carlo size certification.
    """
    n = 40
    n_perm = 99
    rates = []
    for rep in range(60):
        rng = np.random.default_rng(7000 + rep)
        n_arm = n // 2
        labels = np.concatenate([np.zeros(n_arm, dtype=int),
                                 np.ones(n - n_arm, dtype=int)])
        rng.shuffle(labels)
        clouds = [_w2p_cloud(int(a), rng) for a in labels]
        diagrams = _threshold_diagrams(
            [compute_diagrams(c, filtration="alpha", homology_dims=(0,))
             for c in clouds])
        fit = fit_dist(diagrams, labels, np.zeros((n, 1)), np.full(n, 0.5),
                       method="measure", interval=INTERVAL, n_bins=N_BINS,
                       weight_power=R, homology_dim=0)
        out = stratified_permutation_test(fit, np.zeros(n, dtype=int),
                                          n_perm=n_perm, seed=8000 + rep)
        rates.append(out["pvalue"] < 0.05)
    rate = float(np.mean(rates))
    assert rate <= 0.12, f"W2' null rejection rate {rate} far above alpha"


def test_w1_dist_power_smoke():
    """On W1 the distribution-level test fires at small n."""
    for rep in range(3):
        diagrams, labels = _w1_sample(20, seed=100 + rep)
        n = len(labels)
        fit = fit_dist(diagrams, labels, np.zeros((n, 1)), np.full(n, 0.5),
                       method="measure", interval=INTERVAL, n_bins=N_BINS,
                       weight_power=R, homology_dim=0)
        out = stratified_permutation_test(fit, np.zeros(n, dtype=int),
                                          n_perm=99, seed=200 + rep)
        assert out["pvalue"] < 0.05
        assert out["statistic"] > L3 * 0.9
