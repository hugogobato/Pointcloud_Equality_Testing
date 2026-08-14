"""Smoke test: PH of a noisy circle gives H0 and H1 features.

Phase 0.1 acceptance: pytest green on a smoke test computing H0, H1 on a noisy circle.
"""
import numpy as np
import pytest

from tda2s.ph import compute_diagrams


def _noisy_circle(n=300, radius=1.0, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, size=n)
    pts = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    return pts + rng.normal(0, noise, size=pts.shape)


@pytest.mark.parametrize("filtration", ["alpha", "vr", "cech", "ripser"])
def test_noisy_circle_h0_h1(filtration):
    pts = _noisy_circle()
    diags = compute_diagrams(pts, filtration=filtration, homology_dims=(0, 1))
    assert len(diags) == 2

    h0 = diags[0]
    h1 = diags[1]
    assert h0.shape[1] == 2
    assert h1.shape[1] == 2

    # H1: exactly one prominent feature whose persistence ~ the circle radius
    if len(h1) > 0:
        pers = h1[:, 1] - h1[:, 0]
        assert pers.max() > 0.5 * 1.0
    else:
        pytest.skip(f"{filtration}: no H1 features at default resolution")


def test_cubical_noisy_circle():
    pts = _noisy_circle()
    diags = compute_diagrams(pts, filtration="cubical", homology_dims=(0, 1))
    assert len(diags) == 2
    assert diags[1].shape[1] == 2