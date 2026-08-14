"""Phase 0.3: vectorisation stack tests.

Covers the power-weighted silhouette Lipschitz stability (Kim-Lee), output
shapes/finiteness for all representations, the Betti curves of a noisy
circle, and agreement with tcda_uq.silhouette.compute_silhouette.
"""
import numpy as np
import pytest
from gudhi.bottleneck import bottleneck_distance

from tda2s.ph import compute_diagrams
from tda2s.vec import (betti_curve, euler_curve, landscape,
                       persistence_image, persistence_measure, silhouette,
                       vectorise)


def _noisy_circle(n=200, radius=1.0, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, size=n)
    pts = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    return pts + rng.normal(0, noise, size=pts.shape)


def _circle_diags():
    pts = _noisy_circle()
    return compute_diagrams(pts, filtration="alpha", homology_dims=(0, 1))


@pytest.mark.parametrize("r", [1.0, 2.0, 3.0])
def test_silhouette_lipschitz_stability(r):
    """Power-weighted silhouette is Lipschitz in W_1 (Kim-Lee Lemma 2.1).

        || Lambda_D - Lambda_D' ||_inf  <=  C * W_1(D, D')

    The metric matters. Kim-Lee's bound is against W_1 (optimal matching, total
    displacement), NOT against the bottleneck distance W_inf, which only charges
    the single largest displacement. Under W_1 the measured constant is stable
    at ~0.5-0.6 across r in {1, 2, 3}; under W_inf the same perturbations give
    ratios up to 7.0 at r = 1, because moving many points at once costs W_inf
    nothing extra. A single-point perturbation cannot tell the two apart (there
    W_1 = W_inf), so the sweep below moves a random *subset* of the diagram.
    """
    import persim

    h1 = _circle_diags()[1]
    rng = np.random.default_rng(0)

    worst = 0.0
    for _ in range(40):
        p = h1.copy()
        k = rng.integers(1, len(p) + 1)
        idx = rng.choice(len(p), k, replace=False)
        p[idx] += rng.uniform(-0.05, 0.05, size=(k, 2))
        p[:, 1] = np.maximum(p[:, 1], p[:, 0])      # stay above the diagonal

        w1 = float(persim.wasserstein(h1, p))
        if w1 <= 1e-12:
            continue
        s1 = silhouette([h1], interval=(0.0, 2.0), r=r, resolution=200)
        s2 = silhouette([p], interval=(0.0, 2.0), r=r, resolution=200)
        worst = max(worst, float(np.abs(s1 - s2).max()) / w1)

    assert worst > 0.0, "no perturbation moved the silhouette"
    assert worst <= 1.5, f"r={r}: sup|dSil| / W_1 reached {worst:.3f}, expected <~0.6"


def test_silhouette_is_not_bottleneck_lipschitz_at_low_r():
    """Guardrail for the metric above: W_inf is the wrong denominator.

    Documents *why* the test uses W_1 -- at r = 1 a many-point perturbation
    drives sup|dSil| / W_inf well past any small constant, so a bottleneck-based
    stability claim for the silhouette would be false.
    """
    h1 = _circle_diags()[1]
    rng = np.random.default_rng(0)

    worst = 0.0
    for _ in range(40):
        p = h1.copy()
        p += rng.uniform(-0.05, 0.05, size=p.shape)
        p[:, 1] = np.maximum(p[:, 1], p[:, 0])
        wb = bottleneck_distance(h1, p)
        if wb <= 1e-12:
            continue
        s1 = silhouette([h1], interval=(0.0, 2.0), r=1.0, resolution=200)
        s2 = silhouette([p], interval=(0.0, 2.0), r=1.0, resolution=200)
        worst = max(worst, float(np.abs(s1 - s2).max()) / wb)

    assert worst > 2.0, (
        f"bottleneck ratio only reached {worst:.3f}; if this ever holds below a "
        "small constant, revisit which metric the stability claim is stated in")


def test_representation_shapes_and_finiteness():
    """Every representation returns a finite array of the expected shape."""
    diags = _circle_diags()
    n_dims = len(diags)

    s = silhouette(diags)
    assert s.shape == (n_dims, 100) and np.isfinite(s).all()

    l = landscape(diags, num_landscapes=3, resolution=50)
    assert l.shape == (n_dims, 3, 50) and np.isfinite(l).all()

    b = betti_curve(diags, interval=(0.0, 2.0), n_points=64)
    assert b.shape == (n_dims, 64) and np.isfinite(b).all()
    assert (b >= 0).all() and b.dtype == float

    e = euler_curve(diags, interval=(0.0, 2.0), n_points=64)
    assert e.shape == (64,) and np.isfinite(e).all()

    img = persistence_image(diags, resolution=(8, 8), interval=(0.0, 2.0))
    assert img.shape == (n_dims, 8, 8) and np.isfinite(img).all()

    m = persistence_measure(diags, n_bins=16, interval=(0.0, 2.0))
    assert m.shape == (n_dims, 16, 16) and np.isfinite(m).all()
    assert (m >= 0).all()


def test_vectorise_dispatches():
    """vectorise routes to the per-representation functions."""
    diags = _circle_diags()
    assert np.array_equal(vectorise(diags, "silhouette"), silhouette(diags))
    assert np.array_equal(vectorise(diags, "landscape"), landscape(diags))
    assert np.array_equal(vectorise(diags, "betti"), betti_curve(diags))
    assert np.array_equal(vectorise(diags, "euler"), euler_curve(diags))
    assert np.array_equal(vectorise(diags, "image"), persistence_image(diags))
    assert np.array_equal(vectorise(diags, "measure"), persistence_measure(diags))
    with pytest.raises(ValueError):
        vectorise(diags, "nope")


def test_betti_curve_noisy_circle():
    """The H1 Betti curve of a noisy circle plateaus at 1 around the radius scale.

    The persistent H1 class is alive on a wide t-interval near the circle
    radius (birth ~ 0.03, death ~ 0.86 for this cloud), and no noise feature
    survives on (0.2, 0.8), so the H1 curve's sup over that plateau is 1. The
    H0 curve counts component merges and decays to 0 at large t because the
    essential class is dropped (tda2s convention); it is non-increasing and
    starts at the number of H0 features.
    """
    diags = _circle_diags()
    grid = np.linspace(0.0, 2.0, 201)
    b = betti_curve(diags, interval=(0.0, 2.0), n_points=201)
    mask = (grid >= 0.2) & (grid <= 0.8)
    assert b[1][mask].min() == 1
    assert b[1][mask].max() == 1
    assert b[0][0] == len(diags[0])
    assert b[0][-1] == 0
    assert (np.diff(b[0]) <= 0).all()


def test_silhouette_matches_tcda_uq():
    """tda2s silhouettes match tcda_uq.silhouette.compute_silhouette (~1e-6).

    Both wrap gudhi's Silhouette with the same weight, interval, resolution
    and keep_endpoints convention, so they agree to float precision. The
    multi-dim case (torus H0/H1/H2) is exercised as well.
    """
    from tcda_uq.silhouette.core import compute_silhouette as tcda_sil

    diags = _circle_diags()
    kwargs = dict(interval=(0.0, 0.2), r=3.0, resolution=100)
    ours = silhouette(diags, **kwargs)
    theirs = np.asarray(tcda_sil(diags, **kwargs), dtype=float)
    assert np.abs(ours - theirs).max() <= 1e-6

    rng = np.random.default_rng(28)
    u = np.linspace(0, 2 * np.pi, 14, endpoint=False)
    v = np.linspace(0, 2 * np.pi, 14, endpoint=False)
    U, V = np.meshgrid(u, v)
    x = (2 + np.cos(V)) * np.cos(U)
    y = (2 + np.cos(V)) * np.sin(U)
    z = np.sin(V)
    torus = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    torus = torus + rng.normal(0, 0.05, size=torus.shape)
    tdiags = compute_diagrams(torus, filtration="alpha", homology_dims=(0, 1, 2))
    ours3 = silhouette(tdiags, **kwargs)
    theirs3 = np.asarray(tcda_sil(tdiags, **kwargs), dtype=float)
    assert np.abs(ours3 - theirs3).max() <= 1e-6
