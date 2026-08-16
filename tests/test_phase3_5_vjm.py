"""Phase 3.5 studentized empirical-null degree comparator.

The tests pin the three properties the mapping audit relies on: the per-degree
null draws are the *same* draws the Phase 3 statistics already use, the pooled
standardization is exactly exchangeable while the source convention is not, and
the comparator is invariant to a per-degree rescaling of the summaries, which
is the property the unstudentized shared max lacks.
"""

import numpy as np

from sklearn.linear_model import LogisticRegression
from tcda_uq.datasets import TriOracleSimulation

from tda2s.tests.dr_outcome import (
    _ks_statistic,
    _standardize_family,
    degree_null_statistics,
    fit_dr,
    multiplier_test,
    propensity_strata,
    stratified_permutation_test,
    vjm_multiplicity_test,
)


def _fit_small(seed=0, degree_scale=1.0):
    # tcda_uq's default propensity model is an unseeded random forest, so the
    # estimator is passed explicitly: without it two fits on identical data
    # disagree and no invariance below is testable.
    sim = TriOracleSimulation(
        n_cov=3, n_hom_dim=2, resolution=24, n_basis=5,
        coef_scale=0.5, noise_scale=0.08, seed=3,
    )
    sample = sim.sample(48, rng=seed)
    phi, A, X = sample.observed
    phi = np.array(phi, dtype=float)
    phi[:, 0, :] *= degree_scale
    fit = fit_dr((phi, A, X), sample.tseq, n_basis=5, n_folds=2,
                 propensity_estimator=LogisticRegression(max_iter=1000,
                                                         random_state=seed),
                 random_state=seed)
    strata = propensity_strata(sample.propensity, n_bins=4)
    return sample, fit, strata


def test_per_degree_null_reduces_to_the_phase3_shared_max_null():
    """The comparator must read the Phase 3 nulls, not a fresh set of draws."""
    _, fit, strata = _fit_small(seed=1)

    family = degree_null_statistics(fit, mechanism="multiplier", n_draws=64,
                                    seed=7)
    reference = multiplier_test(fit, n_draws=64, seed=7)
    np.testing.assert_allclose(np.max(family["null"], axis=1), reference["null"],
                               atol=1e-12)
    assert np.isclose(np.max(family["observed"]), reference["statistic"])

    family = degree_null_statistics(fit, mechanism="permutation", strata=strata,
                                    n_draws=64, seed=7)
    reference = stratified_permutation_test(fit, strata, n_perm=64, seed=7)
    np.testing.assert_allclose(np.max(family["null"], axis=1), reference["null"],
                               atol=1e-12)
    assert np.isclose(np.max(family["observed"]), reference["statistic"])


def test_pooled_standardization_is_exchangeable_and_source_is_not():
    """Pooled constants are symmetric in the n_draws + 1 augmented values.

    Swapping the observed statistic with one null replicate must leave the
    pooled standardized *set* unchanged, which is what makes the rank p-value
    exact under exchangeability.  The source convention centres on the null
    replicates only, so the same swap moves the constants.
    """
    rng = np.random.default_rng(0)
    observed = np.array([1.4, 0.7])
    null = rng.gamma(3.0, 0.5, size=(50, 2))

    mu_p, sd_p, z_obs, z_null = _standardize_family(observed, null, "pooled", 1e-10)
    swapped_obs = null[0].copy()
    swapped_null = null.copy()
    swapped_null[0] = observed
    mu_s, sd_s, z_obs_s, z_null_s = _standardize_family(
        swapped_obs, swapped_null, "pooled", 1e-10)
    np.testing.assert_allclose(mu_p, mu_s, atol=1e-12)
    np.testing.assert_allclose(sd_p, sd_s, atol=1e-12)
    np.testing.assert_allclose(np.sort(np.vstack([z_obs[None, :], z_null]), axis=0),
                               np.sort(np.vstack([z_obs_s[None, :], z_null_s]), axis=0),
                               atol=1e-12)

    mu_src, _, _, _ = _standardize_family(observed, null, "source", 1e-10)
    mu_src_swapped, _, _, _ = _standardize_family(swapped_obs, swapped_null,
                                                  "source", 1e-10)
    assert not np.allclose(mu_src, mu_src_swapped)


def test_comparator_is_invariant_to_a_per_degree_rescaling():
    """A degree rescaling must not move the studentized decision.

    Multiplying one degree's summaries by a constant is exactly the
    incomparability the source's standardization exists to remove.  The
    unstudentized shared max is not invariant to it; the comparator is.
    """
    _, fit, strata = _fit_small(seed=2, degree_scale=1.0)
    _, fit_scaled, strata_scaled = _fit_small(seed=2, degree_scale=6.0)

    base = vjm_multiplicity_test(fit, strata=strata, n_draws=199, seed=4)
    scaled = vjm_multiplicity_test(fit_scaled, strata=strata_scaled,
                                   n_draws=199, seed=4)

    np.testing.assert_allclose(base["conventions"]["pooled"]["standardized_observed"],
                               scaled["conventions"]["pooled"]["standardized_observed"],
                               rtol=1e-8, atol=1e-8)
    assert base["pvalue"] == scaled["pvalue"]
    # The scaling is a genuine change of the raw family, so the invariance
    # above is not vacuous: degree 0 is scaled exactly, degree 1 is untouched,
    # and the unstudentized max switches which degree it reads.
    np.testing.assert_allclose(scaled["observed"][0], 6.0 * base["observed"][0],
                               rtol=1e-8)
    np.testing.assert_allclose(scaled["observed"][1], base["observed"][1],
                               rtol=1e-8)
    assert int(np.argmax(base["observed"])) != int(np.argmax(scaled["observed"]))
    assert scaled["shared_max_statistic"] != base["shared_max_statistic"]


def test_comparator_reports_all_three_procedures_on_one_null_matrix():
    _, fit, strata = _fit_small(seed=5)
    out = vjm_multiplicity_test(fit, strata=strata, n_draws=199, alpha=0.05,
                                seed=6)

    assert out["null"].shape == (199, fit.n_hom_dim)
    for key in ("pvalue", "bonferroni_pvalue", "shared_max_pvalue"):
        assert 1.0 / 200.0 <= out[key] <= 1.0
    # Bonferroni cannot be sharper than the smallest per-degree p-value.
    assert out["bonferroni_pvalue"] >= out["per_degree_pvalue"].min()
    assert set(out["conventions"]) == {"pooled", "source"}
    assert 0.0 <= out["comparability"]["max_pairwise_ks"] <= 1.0
    fdr = out["fdr"]
    assert fdr["cutoff"] is None or all(
        d in range(fit.n_hom_dim) for d in fdr["rejected_degrees"])

    multiplier_arm = vjm_multiplicity_test(fit, mechanism="multiplier",
                                           n_draws=199, seed=6)
    assert 1.0 / 200.0 <= multiplier_arm["pvalue"] <= 1.0


def test_ks_statistic_matches_a_direct_ecdf_computation():
    rng = np.random.default_rng(3)
    a = rng.normal(size=200)
    b = rng.normal(loc=0.8, size=150)
    grid = np.concatenate([a, b])
    expected = np.max(np.abs(
        np.mean(a[None, :] <= grid[:, None], axis=1)
        - np.mean(b[None, :] <= grid[:, None], axis=1)))
    assert np.isclose(_ks_statistic(a, b), expected)
    assert _ks_statistic(a, a) == 0.0


def test_phase3_5_driver_reproduces_the_published_phase3_pvalues_exactly():
    """The comparator is a strict addition to the Phase 3 draws, not a rerun.

    The Phase 3.5 ``fwer`` design shares the Phase 3 oracle null design and
    every seed, so the shared max-statistic it reads off its own null matrix
    must equal the published Phase 3 p-values to the last digit.  This pins
    the estimator, the calibration draws and the seed derivation in one check.
    """
    import os

    from experiments.phase3_5_vjm import check_phase3_agreement

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    compared = check_phase3_agreement(os.path.join(repo, "experiments", "colab"),
                                      reps=(0,), sizes=(50,))
    assert len(compared) == 3  # three propensity regimes at n = 50
