"""Phase 4.5 unit tests: the studentized multiplier weak-null calibration.

Covers the frozen calibration contract (task 4.5.1), the finite-dimensional
AIPW linearization (task 4.5.2), the two-sample collapse identity of task
4.5.3, the zero-variance deterministic failure report, the multiplier/
Gaussian-path/Rademacher agreement diagnostics, and the cloud-level
guardrails behind the fleet (``experiments/phase45_weak_null.py``).

Run with ``rtk uv run pytest tests/test_phase45.py -q``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps

from tda2s.tests.dist_level import (
    DistFit,
    VARIANCE_FLOOR,
    DEFAULT_N_DRAWS,
    _arm_estimates,
    _max_t_statistic,
    _score_variance,
    dist_multiplier_test,
    dist_scores,
    fit_dist,
    stratified_permutation_test,
    studentized_permutation_test,
)
from experiments.phase45_weak_null import (
    _confounded_cloud,
    _one_rep,
    _propensity_for,
    _sharp_cloud,
    _threshold_diagrams,
    _w1_stoch_cloud,
    _w2p_cloud,
    INTERVAL,
    N_BINS,
    R,
    TAU,
    _cells,
    _gate,
)
from tda2s.dgp.clouds import merge_staircase_cloud, split_cluster_cloud
from tda2s.ph import compute_diagrams
from tda2s.vec import persistence_measure

ALPHA = 0.05


# ---------------------------------------------------------------------------
# helpers

def _feature_fit(n_arm, *, seed=0, q=5, mu0_val=1.0, mu1_val=1.5,
                 sigma0=2.0, sigma1=1.0):
    """Feature-level DistFit: balanced arms, known propensity 1/2.

    ``V`` is supplied directly (no diagrams are needed for the score layer;
    dummy diagrams keep the cache valid).  Nuisances are in-sample arm means.
    """
    rng = np.random.default_rng(seed)
    labels = np.concatenate([np.zeros(n_arm, dtype=int),
                             np.ones(n_arm, dtype=int)])
    rng.shuffle(labels)
    n = 2 * n_arm
    V = np.full((n, q), 0.5)
    V[labels == 0, 0] = rng.normal(mu0_val, sigma0, size=n_arm)
    V[labels == 1, 0] = rng.normal(mu1_val, sigma1, size=n_arm)
    diagrams = [[np.array([[0.0, 1.0]])] for _ in range(n)]
    X = np.zeros((n, 1))
    pi = np.full(n, 0.5)
    m0 = np.tile(V[labels == 0].mean(0), (n, 1))
    m1 = np.tile(V[labels == 1].mean(0), (n, 1))
    return DistFit(diagrams=diagrams, A=labels, X=X, pi_hat=pi,
                   measure_features=V, mu0_hat=m0, mu1_hat=m1,
                   method="measure", interval=INTERVAL, n_bins=N_BINS,
                   weight_power=R, homology_dim=0, kernel_sigma=0.1)


def _sample(diagrams, labels, X=None, pi_hat=None, mu0_hat=None, mu1_hat=None):
    n = len(labels)
    if X is None:
        X = np.zeros((n, 1))
    if pi_hat is None:
        pi_hat = np.full(n, 0.5)
    return fit_dist(diagrams, labels, X, pi_hat, method="measure",
                    interval=INTERVAL, n_bins=N_BINS, weight_power=R,
                    homology_dim=0, mu0_hat=mu0_hat, mu1_hat=mu1_hat)


def _cloud_sample(n, cloud_fn, seed):
    rng = np.random.default_rng(seed)
    labels = np.concatenate([np.zeros(n // 2, dtype=int),
                             np.ones(n - n // 2, dtype=int)])
    rng.shuffle(labels)
    clouds = [cloud_fn(int(a), rng) for a in labels]
    diagrams = _threshold_diagrams(
        [compute_diagrams(c, filtration="alpha", homology_dims=(0,))
         for c in clouds])
    return diagrams, labels


def _cloud_fit(n, cloud_fn, seed):
    diagrams, labels = _cloud_sample(n, cloud_fn, seed)
    feats = fit_dist(diagrams, labels, np.zeros((n, 1)), np.full(n, 0.5),
                     method="measure", interval=INTERVAL, n_bins=N_BINS,
                     weight_power=R, homology_dim=0).measure_features
    m0 = np.tile(feats[labels == 0].mean(0), (n, 1))
    m1 = np.tile(feats[labels == 1].mean(0), (n, 1))
    return _sample(diagrams, labels, mu0_hat=m0, mu1_hat=m1)


# ---------------------------------------------------------------------------
# task 4.5.1: the frozen contract

def test_frozen_contract_constants():
    assert DEFAULT_N_DRAWS == 1999
    assert VARIANCE_FLOOR == 1e-8


def test_variance_floor_boundary_is_dropped():
    with pytest.raises(ValueError, match="variance"):
        _max_t_statistic(np.array([0.0]), np.array([VARIANCE_FLOOR]),
                         10, VARIANCE_FLOOR)


def test_fit_dist_rejects_mismatched_nuisance_shape():
    diagrams = [[np.array([[0.0, 1.0]])] for _ in range(4)]
    labels = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="mu0_hat"):
        fit_dist(diagrams, labels, np.zeros((4, 1)), np.full(4, 0.5),
                 mu0_hat=np.zeros((4, 2)), mu1_hat=np.zeros((4, 1)))


def test_fit_dist_rejects_partial_aipw_nuisances():
    diagrams = [[np.array([[0.0, 1.0]])] for _ in range(4)]
    labels = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="supplied together"):
        fit_dist(diagrams, labels, np.zeros((4, 1)), np.full(4, 0.5),
                 mu0_hat=np.zeros((4, 1024)))


def test_gate_keeps_documented_near_miss_out_of_strict_pass():
    cells = {}
    for design in ("w2p", "confounded", "sharp"):
        for n, regime, p in _cells(design):
            if n < 200:
                continue
            key = str((design, n, regime, p))
            cells[key] = {
                "design": design, "n": n,
                "multiplier_p": {"reps": 500, "rate": 0.05},
            }
    key = "('confounded', 200, 1.0, None)"
    cells[key]["multiplier_p"]["rate"] = 0.096
    verdict = _gate(cells)
    assert verdict["in_size_band"] is False
    assert verdict["qualified_operating_pass"] is True
    assert verdict["size_failures"] == [(key, 0.096)]


# ---------------------------------------------------------------------------
# task 4.5.2: the finite-dimensional AIPW linearization

def test_score_mean_is_the_aipw_contrast():
    for seed in range(5):
        fit = _feature_fit(30, seed=seed)
        scores = dist_scores(fit)
        t0, t1 = _arm_estimates(fit, fit.A)
        np.testing.assert_allclose(scores.mean(0), t1 - t0, atol=1e-12)


def test_oracle_scores_are_exact_on_w1_clouds():
    """Stochastic W1: in-sample means still reproduce the feature contrast."""
    n = 100
    rng = np.random.default_rng(0)
    labels = np.concatenate([np.zeros(n // 2, dtype=int),
                             np.ones(n // 2, dtype=int)])
    rng.shuffle(labels)
    clouds = [_w1_stoch_cloud(int(a), rng) for a in labels]
    diagrams = _threshold_diagrams(
        [compute_diagrams(c, filtration="alpha", homology_dims=(0,))
         for c in clouds])
    # oracle nuisances are the in-sample arm means of the features
    fit0 = _sample(diagrams, labels)
    feats = fit0.measure_features
    fit = _sample(diagrams, labels,
                  mu0_hat=np.tile(feats[labels == 0].mean(0), (n, 1)),
                  mu1_hat=np.tile(feats[labels == 1].mean(0), (n, 1)))
    est = dist_scores(fit).mean(0)
    t0, t1 = _arm_estimates(fit, fit.A)
    np.testing.assert_allclose(est, t1 - t0, atol=1e-12)
    # the stochastic W1 keeps the effect at the merge scale: L1 contrast
    # concentrates in a few bins near mid = 1.35 / 2
    assert np.abs(est).sum() > 1.0


# ---------------------------------------------------------------------------
# the weak-null algebra of the confounded DGP (fleet exactness)

def test_confounded_weak_null_target_is_zero_in_law():
    w = (lambda p: float(abs(p[1] - p[0]) ** R))
    def feature(cloud):
        diagram = _threshold_diagrams([compute_diagrams(
            cloud, filtration="alpha", homology_dims=(0,))])[0][0]
        return persistence_measure(
            [diagram], weight=w, interval=INTERVAL, n_bins=N_BINS)[0].ravel()

    # Use the two fixed mixture components, rather than two random draws from
    # the mixture.  This checks the law-level identity itself and is robust to
    # changes in the random-number stream.
    vS = feature(merge_staircase_cloud([1.0, 2.0], noise=0.15, n_gon=12))
    vA = feature(split_cluster_cloud(
        18, 2, separation=2.3, noise=0.15, deterministic=True, n_gon=18))
    vB = feature(split_cluster_cloud(
        18, 2, separation=4.3, noise=0.15, deterministic=True, n_gon=18))
    vM = 0.5 * vA + 0.5 * vB
    # The staircase measure is the sum of the two one-merge components, while
    # the mixture is their average.  Hence each stratum contrast has L1 mass
    # 4.5 and the two standardized stratum contrasts cancel bin-for-bin.
    np.testing.assert_allclose(vS, vA + vB, atol=1e-12)
    np.testing.assert_allclose(np.abs(vM - vS).sum(), 4.5, atol=1e-9)
    np.testing.assert_allclose(vS - vM, -(vM - vS), atol=1e-12)
    np.testing.assert_allclose(0.5 * (vM - vS) + 0.5 * (vS - vM), 0,
                               atol=1e-12)


# ---------------------------------------------------------------------------
# task 4.5.3: two-sample collapse identity

def test_two_sample_collapse_identity_exact():
    """Balanced arms, known e = 1/2, in-sample means:
    T_n == sqrt(n / (n - 2)) * |t_pooled| exactly."""
    for n_arm in (20, 50):
        fit = _feature_fit(n_arm, seed=3)
        scores = dist_scores(fit)
        est = scores.mean(0)[0]
        V1 = fit.measure_features[:, 0]
        m1 = V1[fit.A == 1].mean()
        m0 = V1[fit.A == 0].mean()
        s2 = _score_variance(scores)[0]
        stat, active = _max_t_statistic(est * np.ones(1), np.array([s2]),
                                        fit.n, VARIANCE_FLOOR)
        S = ((V1[fit.A == 1] - m1) ** 2).sum() + ((V1[fit.A == 0] - m0) ** 2).sum()
        t_pooled = abs(m1 - m0) / np.sqrt(S * 4 / (fit.n * (fit.n - 2)))
        assert np.isclose(est, m1 - m0, atol=1e-12)
        ratio = stat / (t_pooled * np.sqrt(fit.n / (fit.n - 2)))
        assert abs(ratio - 1.0) < 1e-9


def test_two_sample_collapse_multiplier_matches_t_test():
    diffs, rej_mult, rej_t = [], [], []
    for rep in range(60):
        fit = _feature_fit(20, seed=1000 + rep)
        out = dist_multiplier_test(fit, n_draws=1999, seed=2000 + rep)
        p_t = sps.ttest_ind(fit.measure_features[fit.A == 1, 0],
                            fit.measure_features[fit.A == 0, 0],
                            equal_var=True).pvalue
        diffs.append(abs(out["pvalue"] - p_t))
        rej_mult.append(out["pvalue"] < ALPHA)
        rej_t.append(p_t < ALPHA)
    assert np.mean(diffs) < 0.03
    assert abs(np.mean(rej_mult) - np.mean(rej_t)) <= 0.10


# ---------------------------------------------------------------------------
# multiplier null and variance diagnostics

def test_multiplier_null_draws_are_unit_scale():
    """Single active coordinate: the null draws are |Z|-distributed
    (E = sqrt(2/pi), Var = 1 - 2/pi); 5 active coordinates inflate the max."""
    fit = _feature_fit(60, seed=7)
    out = dist_multiplier_test(fit, n_draws=4999, seed=8)
    assert out["n_active_coordinates"] == 1
    assert abs(out["null"].mean() - np.sqrt(2 / np.pi)) < 0.02
    assert abs(out["null"].var() - (1 - 2 / np.pi)) < 0.02


def test_zero_variance_coordinates_dropped_and_reported():
    fit = _feature_fit(25, seed=11)
    out = dist_multiplier_test(fit, n_draws=999, seed=12)
    assert out["n_coordinates"] == 5
    assert out["n_active_coordinates"] == 1
    assert out["n_dropped_coordinates"] == 4
    assert not out["active"][1:].any()


def test_all_constant_scores_raise_deterministic_failure():
    fit = _feature_fit(20, seed=0)
    fit.measure_features = np.full((fit.n, 5), 2.0)
    fit.mu0_hat = np.full((fit.n, 5), 2.0)
    fit.mu1_hat = np.full((fit.n, 5), 2.0)
    with pytest.raises(ValueError, match="variance"):
        dist_multiplier_test(fit, n_draws=99, seed=0)


def test_gaussian_path_agrees_with_multiplier():
    """CCK-style Gaussian path vs multiplier null on correlated scores."""
    diffs = []
    for rep in range(30):
        rng = np.random.default_rng(rep)
        n = 60
        labels = np.concatenate([np.zeros(n // 2, dtype=int),
                                 np.ones(n // 2, dtype=int)])
        rng.shuffle(labels)
        V = np.zeros((n, 4))
        V[:, 0] = rng.normal(0, 1.5, n)
        V[:, 1] = V[:, 0] + rng.normal(0, 0.5, n)
        V[:, 2] = rng.normal(0, 0.2, n)
        V[:, 3] = 0.0
        diagrams = [[np.array([[0.0, 1.0]])] for _ in range(n)]
        fit = DistFit(diagrams=diagrams, A=labels, X=np.zeros((n, 1)),
                      pi_hat=np.full(n, 0.5), measure_features=V,
                      mu0_hat=np.tile(V[labels == 0].mean(0), (n, 1)),
                      mu1_hat=np.tile(V[labels == 1].mean(0), (n, 1)),
                      method="measure", interval=INTERVAL, n_bins=N_BINS,
                      weight_power=R, homology_dim=0, kernel_sigma=0.1)
        out = dist_multiplier_test(fit, n_draws=1999, seed=100 + rep,
                                   gaussian_path=True)
        diffs.append(abs(out["pvalue"]
                         - out["gaussian_path"]["pvalue"]))
    assert np.mean(diffs) < 0.02


def test_rademacher_and_gaussian_multipliers_agree():
    rates_g, rates_r = [], []
    for rep in range(30):
        fit = _cloud_fit(100, _w2p_cloud, 300 + rep)
        g = dist_multiplier_test(fit, n_draws=1999, seed=400 + rep)
        r = dist_multiplier_test(fit, n_draws=1999, seed=400 + rep,
                                 multiplier="rademacher")
        rates_g.append(g["pvalue"] < ALPHA)
        rates_r.append(r["pvalue"] < ALPHA)
    assert abs(np.mean(rates_g) - np.mean(rates_r)) <= 0.06


# ---------------------------------------------------------------------------
# cloud-level guardrails (the fleet cells, at small rep counts)

def test_w2p_multiplier_size_guardrail():
    rates = []
    for rep in range(60):
        fit = _cloud_fit(100, _w2p_cloud, 500 + rep)
        out = dist_multiplier_test(fit, n_draws=1999, seed=600 + rep)
        rates.append(out["pvalue"] < ALPHA)
    assert 0.0 <= np.mean(rates) <= 0.11


def test_w2p_gaussian_path_diagnostic_ran():
    fit = _cloud_fit(100, _w2p_cloud, 0)
    out = dist_multiplier_test(fit, n_draws=999, seed=0, gaussian_path=True)
    assert out["gaussian_path"] is not None
    assert 0.0 < out["gaussian_path"]["max_ecdf_gap"] < 0.2


def test_sharp_null_permutation_and_multiplier_agree():
    from experiments.phase45_weak_null import _nuisance_models
    rates_m, rates_p, rates_s = [], [], []
    for rep in range(60):
        n = 100
        rng = np.random.default_rng(700 + rep)
        X = rng.integers(0, 2, size=n)
        e0, e1 = _propensity_for(1.0)
        labels = rng.binomial(1, np.where(X == 0, e0, e1)).astype(int)
        clouds = [_sharp_cloud(int(x), rng) for x in X]
        diagrams = _threshold_diagrams(
            [compute_diagrams(c, filtration="alpha", homology_dims=(0,))
             for c in clouds])
        feats = fit_dist(diagrams, labels, X.reshape(-1, 1), np.full(n, 0.5),
                         method="measure", interval=INTERVAL, n_bins=N_BINS,
                         weight_power=R, homology_dim=0).measure_features
        pi_hat, mu0, mu1 = _nuisance_models(
            feats, labels, X.reshape(-1, 1), seed=800 + rep)
        fit = _sample(diagrams, labels, X.reshape(-1, 1), pi_hat,
                      mu0_hat=mu0, mu1_hat=mu1)
        out = dist_multiplier_test(fit, n_draws=999, seed=900 + rep)
        perm = stratified_permutation_test(fit, X, n_perm=399, seed=950 + rep)
        sperm = studentized_permutation_test(fit, X, n_perm=399, seed=960 + rep)
        rates_m.append(out["pvalue"] < ALPHA)
        rates_p.append(perm["pvalue"] < ALPHA)
        rates_s.append(sperm["pvalue"] < ALPHA)
    assert all(0.0 <= r <= 0.12 for r in (np.mean(rates_m), np.mean(rates_p),
                                          np.mean(rates_s)))
    assert abs(np.mean(rates_m) - np.mean(rates_p)) <= 0.07
    assert abs(np.mean(rates_m) - np.mean(rates_s)) <= 0.07


def test_w1_multiplier_power_smoke():
    for rep in range(3):
        fit = _cloud_fit(50, _w1_stoch_cloud, 1000 + rep)
        out = dist_multiplier_test(fit, n_draws=999, seed=1100 + rep)
        assert out["pvalue"] < 0.10


def test_local_sharp_null_at_p_zero():
    rates_m, rates_p = [], []
    for rep in range(20):
        row = _one_rep(rep, 200, "local", p=0.0)
        rates_m.append(row["multiplier_p"] < ALPHA)
        rates_p.append(row["permutation_p"] < ALPHA)
    assert abs(np.mean(rates_m) - np.mean(rates_p)) <= 0.08
    assert 0.0 <= np.mean(rates_m) <= 0.12


def test_confounded_row_shape():
    row = _one_rep(0, 100, "confounded", regime=0.5)
    for key in ("multiplier_p", "permutation_p", "studentized_permutation_p",
                "n_active_coordinates"):
        assert key in row
    assert row["n_active_coordinates"] >= 1
