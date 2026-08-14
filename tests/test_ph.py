"""Phase 0.2: PH pipeline tests.

Covers the DTM-Rips fix (gudhi's DistanceToMeasure is fit/transform-based,
not callable), the extraction loop for homology_dims=(0, 1, 2), torus Betti
recovery under alpha and VR filtrations, cross-filtration agreement between
gudhi VR and ripser, the on-disk caching round-trip, and ``betti_numbers``.

All point clouds are kept small (<= 500 points) and tests run sequentially.
"""
import numpy as np
import pytest

from tda2s.ph import betti_numbers, compute_diagrams


def _noisy_circle(n=200, radius=1.0, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, size=n)
    pts = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    return pts + rng.normal(0, noise, size=pts.shape)


def _torus_grid(nu=20, nv=20, R=2.0, r=1.0, noise=0.12, seed=28):
    """Torus parameterization x=(R+r cos v)cos u, y=(R+r cos v)sin u, z=r sin v.

    Grid-sampled in (u, v) with small gaussian noise. The seed is fixed so the
    test is deterministic.
    """
    rng = np.random.default_rng(seed)
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    v = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    U, V = np.meshgrid(u, v)
    x = (R + r * np.cos(V)) * np.cos(U)
    y = (R + r * np.cos(V)) * np.sin(U)
    z = r * np.sin(V)
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    return pts + rng.normal(0, noise, size=pts.shape)


def _betti_above_clear_gap(diags, floor=0.1):
    """Count features above the largest persistence gap per dim.

    Per homology dim: restrict to features with persistence > floor, then
    split the sorted persistence values at their largest gap and count the
    features above it. A single surviving feature (no gap needed) counts 1.
    """
    out = []
    for dgm in diags:
        pers = np.sort(dgm[:, 1] - dgm[:, 0]) if len(dgm) else np.array([])
        idx = np.where(pers > floor)[0]
        if len(idx) == 0:
            out.append(0)
        elif len(idx) == 1:
            out.append(1)
        else:
            gaps = np.diff(pers[idx])
            out.append(int(len(idx) - gaps.argmax() - 1))
    return out


def test_dtm_rips_noisy_circle_prominent_h1():
    """DTM-Rips (Anai et al.) on a noisy circle recovers a prominent H1 class.

    gudhi 3.11's DistanceToMeasure is not callable; the module uses
    fit_transform and the weighted Rips complex. The circle has radius 1, so
    the H1 class persisting for more than half the radius scale is prominent.
    """
    pts = _noisy_circle(n=200, noise=0.05)
    diags = compute_diagrams(pts, filtration="dtm-rips", homology_dims=(0, 1),
                             dtm_k=20, max_edge_length=3.0)
    h1 = diags[1]
    assert h1.shape[1] == 2
    assert len(h1) > 0
    pers = h1[:, 1] - h1[:, 0]
    assert pers.max() > 0.5 * 1.0
    assert np.isfinite(h1).all()


@pytest.mark.parametrize("filtration", ["vr", "ripser", "alpha", "cech"])
def test_homology_dims_012_extraction(filtration):
    """The extraction loop returns per-requested-dim arrays for dims (0, 1, 2)."""
    pts = _noisy_circle(n=150, noise=0.05)
    diags = compute_diagrams(pts, filtration=filtration, homology_dims=(0, 1, 2),
                             max_edge_length=1.2)
    assert len(diags) == 3
    for dim, dgm in enumerate(diags):
        assert dgm.ndim == 2 and dgm.shape[1] == 2
        assert dgm.dtype == float
        assert np.isfinite(dgm).all()
        if dim == 0:
            assert len(dgm) > 0
    assert len(diags[2]) == 0 or (diags[2][:, 1] > diags[2][:, 0]).all()


@pytest.mark.parametrize("filtration", ["vr", "ripser", "alpha", "cech", "dtm-rips",
                                        "cubical"])
def test_filtration_values_are_radii_not_squared_radii(filtration):
    """Every filtration must be 1-homogeneous in the scale of the cloud.

    Scaling the cloud by ``c`` must scale all birth/death values by ``c``. A
    filtration that reports *squared* radii (AlphaComplex and
    DelaunayCechComplex both do, internally) would scale by ``c**2`` instead,
    silently putting one filtration on a different axis from the others. This
    is a regression test: ``cech`` previously leaked squared values.
    """
    c = 3.0
    pts = _noisy_circle(n=80, radius=1.0, noise=0.03)
    kw = dict(homology_dims=(0, 1), dtm_k=10, grid_size=32)
    base = compute_diagrams(pts, filtration=filtration, **kw)
    scaled = compute_diagrams(c * pts, filtration=filtration, **kw)

    for d, (b, s) in enumerate(zip(base, scaled)):
        assert len(b) == len(s), f"dim {d}: feature count changed under scaling"
        if not len(b):
            continue
        order_b = np.lexsort((b[:, 1], b[:, 0]))
        order_s = np.lexsort((s[:, 1], s[:, 0]))
        assert np.allclose(c * b[order_b], s[order_s], rtol=1e-6, atol=1e-6), (
            f"dim {d}: {filtration} is not 1-homogeneous "
            f"(max death {b[:, 1].max():.4f} -> {s[:, 1].max():.4f}, "
            f"expected {c * b[:, 1].max():.4f})")


def test_cech_and_alpha_share_a_scale():
    """Delaunay-Cech and Alpha must land on the same axis on the same cloud."""
    pts = _noisy_circle(n=120, radius=1.0, noise=0.03)
    cech = compute_diagrams(pts, filtration="cech", homology_dims=(0, 1))
    alpha = compute_diagrams(pts, filtration="alpha", homology_dims=(0, 1))
    d_cech = float(cech[1][:, 1].max())
    d_alpha = float(alpha[1][:, 1].max())
    # Both are radius-scale, so the dominant H1 deaths agree to within 25%;
    # under the squared-radius bug the ratio was ~1/d_alpha, i.e. way off.
    assert 0.75 <= d_cech / d_alpha <= 1.33, (
        f"cech H1 death {d_cech:.4f} vs alpha {d_alpha:.4f} -- scale mismatch")


def test_torus_betti_alpha_and_vr():
    """Alpha and VR filtrations on a noisy grid torus recover Betti (1, 2, 1).

    Features are counted above the largest persistence gap per dim (with a
    floor of 0.1 relative to the torus minor radius r=1). For this fixed cloud
    the gaps are: H0 0.289 -> 0.326 (alpha) / 0.578 -> 0.652 (vr), H1
    0.324 -> 0.52 (alpha) / 0.62 -> 0.96 (vr), H2 0.108 -> 0.326 (alpha) /
    0.204 -> 0.538 (vr).
    """
    pts = _torus_grid()
    assert len(pts) <= 500
    alpha = compute_diagrams(pts, filtration="alpha", homology_dims=(0, 1, 2))
    vr = compute_diagrams(pts, filtration="vr", homology_dims=(0, 1, 2),
                          max_edge_length=2.0)
    assert _betti_above_clear_gap(alpha) == [1, 2, 1]
    assert _betti_above_clear_gap(vr) == [1, 2, 1]
    # every counted feature is well above the noise floor
    for diags in (alpha, vr):
        counts = _betti_above_clear_gap(diags)
        for dim, count in enumerate(counts):
            pers = diags[dim][:, 1] - diags[dim][:, 0]
            assert (np.sort(pers)[-count:] > 0.1).all()


def test_vr_ripser_agreement():
    """gudhi VR and ripser compute the same filtration; persistence values agree."""
    pts = _noisy_circle(n=200, noise=0.05)
    vr = compute_diagrams(pts, filtration="vr", homology_dims=(0, 1),
                          max_edge_length=2.0)
    ripser = compute_diagrams(pts, filtration="ripser", homology_dims=(0, 1),
                              max_edge_length=2.0)
    for dim in (0, 1):
        pv = np.sort(vr[dim][:, 1] - vr[dim][:, 0])
        pr = np.sort(ripser[dim][:, 1] - ripser[dim][:, 0])
        k = min(len(pv), len(pr))
        assert k > 0
        assert np.abs(pv[-k:] - pr[-k:]).max() <= 0.05


def test_caching_returns_identical_arrays(tmp_path):
    """Repeated calls with the same cache_dir return bit-identical arrays."""
    pts = _noisy_circle(n=150, noise=0.05)
    cache_dir = str(tmp_path / "ph_cache")
    d1 = compute_diagrams(pts, filtration="alpha", homology_dims=(0, 1),
                          cache_dir=cache_dir)
    d2 = compute_diagrams(pts, filtration="alpha", homology_dims=(0, 1),
                          cache_dir=cache_dir)
    assert len(d1) == len(d2) == 2
    for a, b in zip(d1, d2):
        assert np.array_equal(a, b)
    assert list(tmp_path.glob("ph_cache/*.npz"))


def test_betti_numbers():
    """betti_numbers counts features with persistence above the threshold."""
    pts = _noisy_circle(n=200, noise=0.05)
    diags = compute_diagrams(pts, filtration="alpha", homology_dims=(0, 1))
    counts = betti_numbers(diags, persistence_threshold=0.5)
    assert isinstance(counts, list)
    assert counts == [0, 1]
    assert betti_numbers(diags, persistence_threshold=-1.0) == [
        len(diags[0]), len(diags[1])]
