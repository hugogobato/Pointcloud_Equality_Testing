"""Sanity tests for the Phase 0.5 competitor benchmark wrappers.

Every wrapper takes ``(diags0, diags1)`` and returns a p-value; they are called
through ``run_competitor``, which drops the kwargs a given wrapper does not
accept.  Three checks per competitor:

    * ``test_p_value_range``: the returned value is a finite number in [0, 1]
      on a small example.
    * ``test_size_control``: under the null (two independent draws of the same
      class) the rejection rate at alpha = 0.05 over 40 Monte Carlo
      replications is at most 0.12.
    * ``test_power``: class A (one circle) vs class B (two circles) is
      rejected at alpha = 0.05 in at least 50% of 40 replications.

Test DGP: 80-point unit circle with Gaussian noise (sigma = 0.04) for class A;
class B adds a 40-point circle of radius 0.3.  Persistence diagrams are the
alpha-complex H1 diagrams, and features with persistence below ``MIN_PERS``
are dropped (they are numerical noise of order 1e-2 or less, and the filter is
applied identically to both groups so the null is preserved).
"""
import numpy as np
import pytest

from tda2s.benchmarks import COMPETITORS, run_competitor
from tda2s.ph import compute_diagrams

ALPHA = 0.05
N_PERM = 200
MIN_PERS = 0.05
N_REP = 40
DGP_SEED_BASE = 100000
WRAPPER_SEED_BASE = 200000


def _circle(rng, radius, n, noise=0.04):
    th = rng.uniform(0.0, 2.0 * np.pi, n)
    pts = np.stack([radius * np.cos(th), radius * np.sin(th)], axis=1)
    return pts + rng.normal(0.0, noise, size=pts.shape)


def _class_a(rng, n=25):
    return [_circle(rng, 1.0, 80) for _ in range(n)]


def _class_b(rng, n=25):
    return [np.vstack([_circle(rng, 1.0, 80), _circle(rng, 0.3, 40)]) for _ in range(n)]


def _class_c(rng, n=25):
    """Same expected shape as class A but a RANDOM radius: a pure dispersion
    alternative (the diagram cloud is more spread, its centre is unchanged)."""
    return [_circle(rng, rng.uniform(0.5, 1.5), 80) for _ in range(n)]


def _diagrams(clouds):
    out = []
    for c in clouds:
        d = compute_diagrams(c, filtration="alpha", homology_dims=(1,))[0]
        d = d[d[:, 1] - d[:, 0] > MIN_PERS]
        out.append([d])
    return out


def _call(name, d0, d1, seed, n_perm=N_PERM):
    """One uniform call site; run_competitor drops kwargs a wrapper lacks."""
    return run_competitor(name, d0, d1, n_perm=n_perm, seed=seed, alpha=ALPHA)


@pytest.mark.parametrize("name", COMPETITORS)
def test_p_value_range(name):
    rng = np.random.default_rng(1)
    d0 = _diagrams(_class_a(rng, 6))
    d1 = _diagrams(_class_a(rng, 6))
    p = _call(name, d0, d1, seed=0, n_perm=50)
    assert np.isfinite(p)
    assert 0.0 <= p <= 1.0


@pytest.mark.parametrize("name", COMPETITORS)
def test_size_control(name):
    rej = 0
    for rep in range(N_REP):
        rng = np.random.default_rng(DGP_SEED_BASE + rep)
        d0 = _diagrams(_class_a(rng))
        d1 = _diagrams(_class_a(rng))
        p = _call(name, d0, d1, seed=WRAPPER_SEED_BASE + rep)
        rej += int(p <= ALPHA)
    rate = rej / N_REP
    print(f"[size]  {name:18s}: {rej}/{N_REP} rejections ({rate:.2f})")
    assert rate <= 0.12


#: Competitors that target the *dispersion* of the diagram population rather
#: than its location, and so are not expected to have power against class A vs
#: class B (a pure location alternative). See
#: ``test_krebs_rademacher_targets_dispersion_not_location``.
DISPERSION_TESTS = {"krebs_rademacher"}


@pytest.mark.parametrize("name", [n for n in COMPETITORS if n not in DISPERSION_TESTS])
def test_power(name):
    rej = 0
    for rep in range(N_REP):
        rng = np.random.default_rng(DGP_SEED_BASE + rep)
        d0 = _diagrams(_class_a(rng))
        d1 = _diagrams(_class_b(rng))
        p = _call(name, d0, d1, seed=WRAPPER_SEED_BASE + rep)
        rej += int(p <= ALPHA)
    rate = rej / N_REP
    print(f"[power] {name:18s}: {rej}/{N_REP} rejections ({rate:.2f})")
    assert rate >= 0.5


def test_krebs_rademacher_targets_dispersion_not_location():
    """Krebs-Rademacher compares inco-variances, so it is blind to location.

    This is a property of the published method (arXiv:2401.10349 eq. 1.7 tests
    ``(sigma^2_X - sigma^2_Y)^2``), not a defect of the wrapper, and it is
    pinned here so the benchmark table reports it rather than mistaking it for
    an implementation bug. Class A (fixed-radius circle) vs class B (extra inner
    loop) shifts the *centre* of the diagram population; class A vs class C
    (random radius) shifts its *spread*.
    """
    rej_loc = rej_disp = 0
    n_rep = 20
    for rep in range(n_rep):
        rng = np.random.default_rng(DGP_SEED_BASE + rep)
        a = _diagrams(_class_a(rng))
        b = _diagrams(_class_b(rng))
        c = _diagrams(_class_c(rng))
        rej_loc += int(_call("krebs_rademacher", a, b, seed=rep) <= ALPHA)
        rej_disp += int(_call("krebs_rademacher", a, c, seed=rep) <= ALPHA)
    print(f"[kr] location {rej_loc}/{n_rep}, dispersion {rej_disp}/{n_rep}")
    assert rej_disp / n_rep >= 0.8, "no power against the alternative it targets"
    assert rej_loc / n_rep <= 0.4, "unexpectedly sensitive to a location shift"