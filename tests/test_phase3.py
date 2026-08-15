"""Phase 3 outcome-level DR estimator and calibration tests."""

import numpy as np

from tcda_uq.datasets import TriOracleSimulation

from tda2s.tests.dr_outcome import (
    _draw_stratified_labels,
    _permutation_stats,
    _scores_for_labels,
    equivalence_test,
    fit_dr,
    degree_multiplicity_test,
    multiplier_test,
    positivity_diagnostics,
    propensity_strata,
    stratified_permutation_test,
)


def _fit_small(seed=0):
    sim = TriOracleSimulation(
        n_cov=3, n_hom_dim=2, resolution=24, n_basis=5,
        coef_scale=0.5, noise_scale=0.08, seed=3,
    )
    sample = sim.sample(48, rng=seed)
    fit = fit_dr(sample.observed, sample.tseq, n_basis=5, n_folds=2,
                 random_state=seed)
    return sample, fit


def test_cached_fit_reproduces_tcda_uq_aipw_and_score_order():
    sample, fit = _fit_small(seed=1)
    np.testing.assert_allclose(fit.estimate, np.asarray(fit.result.aipw), atol=1e-12)
    observed_scores = _scores_for_labels(fit, fit.labels_order)
    np.testing.assert_allclose(observed_scores,
                               np.stack(fit.result.scores, axis=1), atol=1e-10)
    np.testing.assert_allclose(observed_scores.mean(axis=0), fit.estimate, atol=1e-12)
    assert np.array_equal(np.sort(fit.order), np.arange(len(sample.A)))
    assert fit.mu0.shape == fit.phi_order.shape == (48, 2, 24)


def test_stratified_draws_preserve_counts_without_refitting():
    rng = np.random.default_rng(4)
    labels = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    strata = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    draws = _draw_stratified_labels(labels, strata, 80, rng)
    for s in np.unique(strata):
        idx = strata == s
        assert np.all(draws[:, idx].sum(axis=1) == labels[idx].sum())


def test_vectorized_permutation_statistic_matches_direct_score_evaluation():
    _, fit = _fit_small(seed=5)
    rng = np.random.default_rng(6)
    draws = rng.integers(0, 2, size=(7, fit.n))
    fast = _permutation_stats(fit, draws, batch_size=3)
    slow = []
    for labels in draws:
        curves = _scores_for_labels(fit, labels).mean(axis=0)
        slow.append(np.sqrt(fit.n) * np.max(np.abs(curves)))
    np.testing.assert_allclose(fast, slow, atol=1e-12)


def test_phase3_calibrations_and_diagnostics_smoke():
    _, fit = _fit_small(seed=2)
    strata_order = propensity_strata(fit.pi_hat, n_bins=4)
    strata = np.empty(fit.n, dtype=int)
    strata[fit.order] = strata_order

    multi = multiplier_test(fit, n_draws=80, seed=5)
    perm = stratified_permutation_test(fit, strata, n_perm=80, seed=5)
    assert 0.0 <= multi["pvalue"] <= 1.0
    assert 0.0 <= perm["pvalue"] <= 1.0
    assert multi["null"].shape == perm["null"].shape == (80,)

    diagnostics = positivity_diagnostics(fit)
    assert 0.0 < diagnostics["min_pi"] < diagnostics["max_pi"] < 1.0
    assert diagnostics["ess_treated"] > 0.0
    assert diagnostics["ess_control"] > 0.0

    equiv = equivalence_test(fit, margin=2.0, n_draws=80, seed=5)
    assert 0.0 <= equiv["pvalue"] <= 1.0
    assert equiv["upper_bound"] >= equiv["observed_norm"]

    multiplicity = degree_multiplicity_test(fit, n_draws=80, seed=5)
    assert 0.0 <= multiplicity["bonferroni_pvalue"] <= 1.0
    assert 0.0 <= multiplicity["max_statistic_pvalue"] <= 1.0


def test_shared_degree_multipliers_are_reproducible():
    _, fit = _fit_small(seed=3)
    a = multiplier_test(fit, n_draws=60, seed=11)
    b = multiplier_test(fit, n_draws=60, seed=11)
    np.testing.assert_array_equal(a["null"], b["null"])
    assert a["pvalue"] == b["pvalue"]
