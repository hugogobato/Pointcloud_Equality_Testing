"""Tests for the resampling engine (Phase 0.4 acceptance).

1. Multiplier bootstrap recovers nominal coverage on a GP toy.
2. Permutation test holds size; strata-preserving permutation never crosses
   strata and deflates rejection in a Simpson-type design.
3. Smoothed bootstrap power/size sanity on diagram classes.
4. cross_fit_folds partitions exactly; paired_bootstrap preserves split.
"""
import numpy as np
import pytest

from tda2s.resample import (
    cross_fit_folds,
    multiplier_bootstrap,
    paired_bootstrap,
    permutation_test,
    p_value,
    smoothed_bootstrap,
)
from tda2s.resample.smoothing import mean_betti_curve


def _gp_curves(n=100, n_basis=50, res=200, sigma=0.1, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, res)
    u = (t - t[0]) / (t[-1] - t[0])
    basis = np.stack([np.cos(2 * np.pi * j * u) for j in range(1, n_basis + 1)], axis=1)
    coef = rng.normal(0, sigma / np.arange(1, n_basis + 1), size=(n, n_basis))
    return t, (basis @ coef.T).T


def _sup_diff_stat(labels, curves):
    """Sup-norm of the group-mean difference on the full curve array."""
    return np.abs(np.mean(curves[labels == 0], axis=0) - np.mean(curves[labels == 1], axis=0)).max()


def _marginal_diff_stat(labels, curves):
    """Sup-norm of the unadjusted (marginal) group-mean difference."""
    return np.abs(np.mean(curves[labels == 1], axis=0)
                  - np.mean(curves[labels == 0], axis=0)).max()


def test_multiplier_bootstrap_coverage():
    """95% sup-band from multiplier draws covers the true mean curve ~95%."""
    rng = np.random.default_rng(0)
    n = 100
    res = 200
    sigma = 0.1
    n_draws = 2000
    n_mc = 200
    covered = 0
    for rep in range(n_mc):
        t, curves = _gp_curves(n=n, res=res, sigma=sigma, seed=rep)
        influence = curves - curves.mean(axis=0)
        null = multiplier_bootstrap(influence, n_draws, rng)
        band = np.quantile(null, 0.95)
        observed = np.sqrt(n) * np.abs(curves.mean(axis=0)).max()
        covered += observed <= band
    rate = covered / n_mc
    assert 0.90 <= rate <= 0.98, f"coverage {rate:.3f} outside [0.90, 0.98]"


def test_permutation_size():
    rng = np.random.default_rng(1)
    n = 60
    res = 100
    alpha = 0.05
    n_mc = 200
    rejects = 0
    for rep in range(n_mc):
        t, curves = _gp_curves(n=n, res=res, seed=rep)
        labels = np.zeros(n, dtype=int)
        labels[: n // 2] = 1
        stat_fn = lambda lab: _sup_diff_stat(lab, curves)
        obs, nulls = permutation_test(stat_fn, labels, n_perm=999, rng=rng)
        rejects += p_value(obs, nulls, "greater") < alpha
    rate = rejects / n_mc
    assert 0.03 <= rate <= 0.08, f"size {rate:.3f} outside [0.03, 0.08]"


def test_strata_never_cross():
    """Within-stratum permutation preserves each stratum's treated count."""
    rng = np.random.default_rng(2)
    strata = np.repeat([0, 1], 30)
    labels = np.zeros(60, dtype=int)
    labels[:25] = 1        # 25 treated in stratum 0
    labels[30:35] = 1      # 5 treated in stratum 1

    def coded_counts(lab):
        """Both stratum treated-counts packed into one scalar (100*n0 + n1)."""
        return 100 * int(lab[strata == 0].sum()) + int(lab[strata == 1].sum())

    for _ in range(50):
        obs, nulls = permutation_test(coded_counts, labels, n_perm=20, rng=rng,
                                      strata=strata)
        assert obs == 2505
        assert np.all(nulls == 2505), "labels crossed strata"


def test_simpson_unstratified_permutation_is_anticonservative():
    """The Phase-2 failure mode, in miniature.

    Design: two strata; stratum 1's curves are shifted up by 1.0; treatment is
    confounded with stratum (25/30 treated in stratum 0, 5/30 in stratum 1).
    Within each stratum the conditional null holds exactly -- there is no
    treatment effect anywhere.

    A *marginal* mean-difference statistic is nonetheless large, purely from the
    covariate shift. Full-label permutation destroys the label/stratum
    association, so its null is centred at zero and the test rejects the true
    null almost always. Within-stratum permutation keeps each stratum's treated
    count fixed, so the null carries the same confounding as the data and the
    test holds its size.
    """
    rng = np.random.default_rng(2)
    n = 60
    strata = np.repeat([0, 1], 30)
    labels = np.zeros(n, dtype=int)
    labels[:25] = 1
    labels[30:35] = 1

    rej_unstrat = rej_strat = 0
    n_mc = 100
    for rep in range(n_mc):
        _, curves = _gp_curves(n=n, res=100, seed=3 + rep)
        curves = curves.copy()
        curves[strata == 1] += 1.0          # covariate effect, not a treatment effect
        stat = lambda lab: _marginal_diff_stat(lab, curves)

        obs, nulls = permutation_test(stat, labels, n_perm=200, rng=rng)
        rej_unstrat += p_value(obs, nulls, "greater") < 0.05

        obs, nulls = permutation_test(stat, labels, n_perm=200, rng=rng, strata=strata)
        rej_strat += p_value(obs, nulls, "greater") < 0.05

    assert rej_unstrat / n_mc >= 0.90, (
        f"unstratified rate {rej_unstrat/n_mc:.3f}; expected near 1 (anticonservative)")
    assert rej_strat / n_mc <= 0.15, (
        f"stratified rate {rej_strat/n_mc:.3f}; expected near alpha=0.05")


# --- smoothed bootstrap (Roycraft-Krebs-Polonik) -----------------------------
#
# The smoothed bootstrap approximates the *sampling* distribution of a
# persistent-Betti functional, so the right checks are (i) it is centred on the
# observed statistic and (ii) its percentile interval attains nominal coverage
# of the population value. Comparing the observed statistic against its own
# bootstrap distribution, as if the latter were a null, is a category error:
# the bootstrap distribution is centred at the statistic, not at zero.

_H1_GRID = (0.0, 1.0)   # radius scale: a circle of radius r has its H1 class die near sqrt(3) r / 2
_T0_INDEX = 40          # t ~ 0.404 on a 100-point grid over [0, 1]


def _h1_diagrams(n_samples, n=80, seed=0):
    """List over samples of a ONE-element diagram list holding only H1.

    The circle radius varies over samples (U[0.35, 0.75]), so the loop's death
    time sqrt(3) r straddles t0: B_1(t0) is a genuinely random 0/1 per sample
    rather than a constant, which is what makes a coverage check meaningful.
    """
    from tda2s.ph import compute_diagrams

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_samples):
        radius = rng.uniform(0.35, 0.75)
        th = rng.uniform(0, 2 * np.pi, n)
        pts = np.stack([radius * np.cos(th), radius * np.sin(th)], axis=1)
        pts = pts + rng.normal(0, 0.02, size=pts.shape)
        h1 = compute_diagrams(pts, filtration="ripser", homology_dims=(0, 1))[1]
        out.append([h1])
    return out


def _mean_h1_at_t0(sample_diagrams):
    """Mean persistent Betti number B_1(t0) over the sample, on a fixed grid."""
    _, mean = mean_betti_curve(sample_diagrams, interval=_H1_GRID, n_points=100)
    return float(mean[0][_T0_INDEX])


def test_smoothed_bootstrap_is_centred_on_the_statistic():
    """Bootstrap draws must be centred at the observed statistic.

    This is the check that catches pooling bugs: if a draw concatenated the k
    resampled diagrams into a single diagram, every Betti count would be
    inflated k-fold and the draws would sit far above the observed value.
    """
    rng = np.random.default_rng(5)
    diags = _h1_diagrams(30, seed=0)
    obs, draws = smoothed_bootstrap(diags, n_draws=300, rng=rng, sigma=0.02,
                                    stat_fn=_mean_h1_at_t0)
    sd = draws.std(ddof=1)
    assert sd > 0, "bootstrap draws are degenerate"
    assert abs(draws.mean() - obs) < 0.5 * sd, (
        f"bootstrap mean {draws.mean():.3f} not centred on obs {obs:.3f} (sd {sd:.3f})")


def test_smoothed_bootstrap_scale_shrinks_with_n():
    """Bootstrap spread must fall like 1/sqrt(n)."""
    rng = np.random.default_rng(6)
    _, small = smoothed_bootstrap(_h1_diagrams(20, seed=1), n_draws=300, rng=rng,
                                  sigma=0.02, stat_fn=_mean_h1_at_t0)
    _, big = smoothed_bootstrap(_h1_diagrams(80, seed=2), n_draws=300, rng=rng,
                                sigma=0.02, stat_fn=_mean_h1_at_t0)
    ratio = small.std(ddof=1) / big.std(ddof=1)
    assert 1.3 <= ratio <= 3.0, f"sd ratio {ratio:.2f}, expected ~sqrt(4) = 2"


def test_smoothed_bootstrap_coverage():
    """Percentile intervals cover the population B_1(t0) at ~nominal rate."""
    rng = np.random.default_rng(7)
    truth = _mean_h1_at_t0(_h1_diagrams(400, seed=999))

    covered = 0
    n_rep = 30
    for rep in range(n_rep):
        diags = _h1_diagrams(25, seed=2000 + rep)
        _, draws = smoothed_bootstrap(diags, n_draws=200, rng=rng, sigma=0.02,
                                      stat_fn=_mean_h1_at_t0)
        lo, hi = np.quantile(draws, [0.025, 0.975])
        covered += lo <= truth <= hi
    assert covered / n_rep >= 0.80, f"coverage {covered/n_rep:.2f} at nominal 0.95"


def test_cross_fit_folds_partition_and_stratify():
    rng = np.random.default_rng(7)
    n = 100
    folds = cross_fit_folds(n, 5, rng)
    test_idx = np.concatenate([t for _, t in folds])
    assert sorted(test_idx) == list(range(n))
    train, test = folds[0]
    assert set(train).isdisjoint(set(test))

    labels = np.repeat([0, 1], n // 2)
    folds = cross_fit_folds(n, 5, rng, stratify_labels=labels)
    for _, t in folds:
        assert (labels[t] == 1).sum() == 10  # 50 treated / 5 folds
        assert (labels[t] == 0).sum() == 10


def test_paired_bootstrap_preserves_split():
    rng = np.random.default_rng(11)
    n = 60
    labels = np.zeros(n, dtype=int)
    labels[:20] = 1
    obs, nulls = paired_bootstrap(lambda lab: int(lab.sum()), labels, n_draws=100, rng=rng)
    assert obs == 20
    assert np.all(nulls == 20)


def test_p_value_bounds():
    assert 0.0 <= p_value(1.0, [0.5, 0.7, 0.9]) <= 1.0
    assert p_value(2.0, [0.5, 0.7, 0.9]) == pytest.approx(1.0 / 4.0)
    assert p_value(-2.0, [-0.5, -0.7, -0.9], "less") == pytest.approx(1.0 / 4.0)