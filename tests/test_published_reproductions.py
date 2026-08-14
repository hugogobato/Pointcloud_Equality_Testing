"""Phase 0.5 acceptance: reproduce a published figure or table per competitor.

The plan requires each competitor wrapper to "reproduce a published figure or
table from its source paper". These tests do that where the source paper
contains one, and pin the method's defining behaviour where it does not. The
simulation designs live in :mod:`tda2s.repro` so that these tests and the
full-fidelity Colab notebooks in ``notebooks/`` run the same code.

Reproduced here
---------------
* ``frechet_anova``  -- Dubey & Muller (2019) Figure 1, BOTH panels.
* ``moon_lazar``     -- Moon & Lazar (2023) Figure 5, false-positive rate and
                        power, all four noise levels.
* ``rt``             -- the Robinson & Turner curve of Moon & Lazar Figure 5b,
                        which is that paper's independent reproduction of the
                        Robinson-Turner test on the same design.
* ``krebs_rademacher`` -- the source paper (arXiv:2401.10349) is a 43-page
                        theory paper with NO simulation study, so there is no
                        figure or table to reproduce. Its pivotal null law
                        (eq. 1.16) is checked against the paper's Theorem 1.5
                        statement instead.

Monte Carlo budgets here are reduced from the papers' 500 replications; each
assertion states a tolerance consistent with the resulting standard error, so
these are reproductions with honest error bars rather than exact digit matches.
``notebooks/01_moon_lazar_figure5.ipynb`` and
``notebooks/02_dubey_muller_figure1.ipynb`` run the same designs at the
published budget. Runs sequentially, small clouds only.
"""
import numpy as np
import pytest

from tda2s.benchmarks.krebs_rademacher import pivotal_quantiles
from tda2s.repro import (ALPHA, MOON_LAZAR_FIG5, MOON_LAZAR_SETTINGS,
                         dubey_muller_rejections, moon_lazar_rejections)

_ML_REPS = 150            # paper uses 500; MC se ~0.04 at these rates
_DM_REPS = 150
_RT_REPS = 40             # rt is the slow one (pairwise Wasserstein per perm)


def _dm_rate(**kw):
    return float(dubey_muller_rejections(_DM_REPS, **kw)[1].mean())


def _ml_rate(name, sigma, scenario, reps, base_seed, **kw):
    return float(moon_lazar_rejections(name, sigma, scenario, reps, base_seed,
                                       **kw)[1].mean())


# --------------------------------------------------------------------------
# Dubey & Muller (2019), Figure 1
# --------------------------------------------------------------------------
# n1 = n2 = 100, alpha = 0.05. The paper's curves sit at the 0.05 line at
# delta = 0 (resp. r = 1) and reach 1 by |delta| ~ 0.5 (resp. r ~ 1.5).

def test_dubey_muller_figure1_left_location():
    """Fig. 1 left: level at delta = 0, power ~1 by |delta| = 0.5."""
    size = _dm_rate(base_seed=1, delta=0.0)
    mid = _dm_rate(base_seed=2, delta=0.25)
    far = _dm_rate(base_seed=3, delta=0.5)
    print(f"[DM fig1 L] delta=0: {size:.3f}  0.25: {mid:.3f}  0.5: {far:.3f}")
    assert size <= 0.10, f"level {size:.3f} exceeds nominal 0.05 + MC error"
    assert far >= 0.90, f"power at delta=0.5 is {far:.3f}, paper's curve is ~1"
    assert size < mid < far, "power must increase with |delta|"


def test_dubey_muller_figure1_right_scale():
    """Fig. 1 right: level at r = 1, power ~1 by r = 1.5.

    This panel is the one a between/within-ratio proxy cannot reproduce: the two
    groups have equal Frechet means and differ only in Frechet variance, which
    is what the Levene-type U_n term of eq. (8) exists to detect.
    """
    size = _dm_rate(base_seed=4, r=1.0, sd=0.2)
    mid = _dm_rate(base_seed=5, r=1.5, sd=0.2)
    far = _dm_rate(base_seed=6, r=2.0, sd=0.2)
    print(f"[DM fig1 R] r=1: {size:.3f}  1.5: {mid:.3f}  2.0: {far:.3f}")
    assert size <= 0.10, f"level {size:.3f} exceeds nominal 0.05 + MC error"
    assert mid >= 0.85, f"power at r=1.5 is {mid:.3f}, paper's curve is ~1"
    assert far >= 0.95, f"power at r=2 is {far:.3f}, paper's curve is 1"


# --------------------------------------------------------------------------
# Moon & Lazar (2023), Figure 5
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sigma", sorted(MOON_LAZAR_FIG5))
def test_moon_lazar_figure5(sigma):
    """Fig. 5a/5b: two-stage test's false-positive rate and power."""
    kw = dict(MOON_LAZAR_SETTINGS)
    tag = int(sigma * 1000)
    fpr = _ml_rate("moon_lazar", sigma, "fpr", _ML_REPS, tag, **kw)
    power = _ml_rate("moon_lazar", sigma, "power", _ML_REPS, 5000 + tag, **kw)
    ref = MOON_LAZAR_FIG5[sigma]
    print(f"[ML fig5] sigma={sigma:.2f}  fpr {fpr:.3f} (paper {ref['fpr']:.3f})"
          f"  power {power:.3f} (paper {ref['power']:.2f})")
    # Fig. 5a: the paper's claim is that the two-stage test keeps the false
    # positive rate below the nominal level at every noise level.
    assert fpr <= 0.10, f"FPR {fpr:.3f} far above the paper's <=0.05 curve"
    # Fig. 5b: match the published power to within ~3 MC standard errors.
    assert abs(power - ref["power"]) <= 0.15, (
        f"power {power:.3f} vs published {ref['power']:.2f} at sigma={sigma}")


def test_robinson_turner_matches_moon_lazar_figure5b():
    """The 'PD' curve of Moon & Lazar Fig. 5b is the Robinson-Turner test.

    Moon & Lazar re-run Robinson & Turner (2017) on their own design with
    1-Wasserstein pairwise distances, giving an independent published power
    curve for our ``rt`` wrapper: ~0.97 at sigma = 0.05 falling to ~0.05 at
    sigma = 0.20. The high-noise tail is the loosest part of the reproduction --
    the paper does not state which loss or how many permutations it used when
    re-running that test -- so it is bounded rather than matched.
    """
    kw = dict(metric="wasserstein", statistic="within", n_perm=200, seed=0)
    hi = _ml_rate("rt", 0.05, "power", _RT_REPS, 11, **kw)
    lo = _ml_rate("rt", 0.20, "power", _RT_REPS, 12, **kw)
    fpr = _ml_rate("rt", 0.05, "fpr", _RT_REPS, 13, **kw)
    print(f"[RT fig5b] power sigma=0.05: {hi:.3f} (paper ~0.97), "
          f"sigma=0.20: {lo:.3f} (paper ~0.05), fpr: {fpr:.3f}")
    assert hi >= 0.85, f"power {hi:.3f} at sigma=0.05, paper's PD curve is ~0.97"
    assert lo <= 0.30, f"power {lo:.3f} at sigma=0.20, paper's PD curve is ~0.05"
    assert fpr <= 0.15, f"false positive rate {fpr:.3f} at sigma=0.05"


# --------------------------------------------------------------------------
# Krebs & Rademacher (2024): no simulation study exists in the paper
# --------------------------------------------------------------------------

def test_krebs_rademacher_pivotal_law_is_calibrated():
    """Theorem 1.5: the eq. (1.16) reference law gives an exact level at D^2 = Delta.

    arXiv:2401.10349 contains no simulations (43 pages, zero tables), so there
    is no published figure to reproduce. What can be checked is the object the
    test is calibrated against: ``W`` of eq. (1.16) must be a genuine pivotal
    law, so its own upper quantiles reject at exactly the nominal rate, and it
    must not depend on the mesh used for ``nu``.
    """
    q95_a = pivotal_quantiles(0.95, grid=20, n_draws=40000, seed=0)
    q95_b = pivotal_quantiles(0.95, grid=40, n_draws=40000, seed=1)
    print(f"[KR eq1.16] q95 grid=20: {q95_a:.3f}  grid=40: {q95_b:.3f}")
    assert np.isfinite(q95_a) and q95_a > 0
    # mesh-stability: nu's discretisation must not move the critical value much
    assert abs(q95_a - q95_b) / abs(q95_a) < 0.15, (
        f"critical value moved from {q95_a:.3f} to {q95_b:.3f} with the mesh")

    # self-consistency: an independent sample of W rejects at ~alpha
    from tda2s.benchmarks.krebs_rademacher import _simulate_W
    W = _simulate_W(20, 40000, np.random.default_rng(99))
    rate = float(np.mean(W > q95_a))
    print(f"[KR eq1.16] empirical rejection at q95: {rate:.4f}")
    assert 0.04 <= rate <= 0.06, f"pivotal law not calibrated: {rate:.4f}"
