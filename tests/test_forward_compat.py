"""Phase 0.9: forward-compatibility freeze for the deferred ecological track.

The ecological collaboration (species niche hypervolumes, Phases 11-12 of
RESEARCH_PLAN_P1_TwoSample.md) is deferred, but the Phase 0.2 / 0.3 / 0.6 APIs
must accept, without rework, its four load-bearing shapes:

1. clouds of wildly unequal cardinality (10 up to 10^5 points; the test uses
   10 and 2000 to stay light),
2. ambient dimension up to ~20,
3. an externally supplied standardisation (mean/scale vector) rather than
   per-cloud scaling -- realised as the ``standardise=(mean, scale)`` parameter
   of ``tda2s.ph.compute_diagrams``,
4. DTM-Rips as a filtration option.

No ecological data, loaders or dependencies live in this repository; these
tests only freeze the API capabilities the deferred track will need. Build
nothing else.
"""
import numpy as np


def _noisy_circle(n=300, radius=1.0, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, size=n)
    pts = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    return pts + rng.normal(0, noise, size=pts.shape)


def _h1_persistence(diags):
    h1 = diags[1]
    if len(h1) == 0:
        return 0.0
    return float(np.max(h1[:, 1] - h1[:, 0]))


def test_unequal_cloud_sizes():
    """Clouds of wildly unequal cardinality both yield valid diagrams.

    The API contract is that ``compute_diagrams`` accepts clouds of any
    per-cloud cardinality; there is no cross-cloud size bookkeeping. The small
    cloud (n=10) is computed with ripser, the large cloud (n=2000) with alpha:
    ripser/VR are O(n^2) in memory and blow up on large clouds, while alpha is
    the recommended filtration there -- the point is that the same API call
    handles both without rework.
    """
    from tda2s.ph import compute_diagrams

    small = np.random.default_rng(0).normal(size=(10, 2))
    large = _noisy_circle(n=2000, seed=1)

    d_small = compute_diagrams(small, filtration="ripser", homology_dims=(0, 1))
    d_large = compute_diagrams(large, filtration="alpha", homology_dims=(0, 1))

    for diags in (d_small, d_large):
        assert len(diags) == 2
        for dgm in diags:
            assert dgm.ndim == 2 and dgm.shape[1] == 2
    assert _h1_persistence(d_large) > 0.5


def test_ambient_dimension_20():
    """A 20-D cloud with an embedded circle has a prominent H1 feature.

    Ambient dimension is not a rework trigger: ``compute_diagrams`` runs in
    d=20 and H1 persistence of the embedded noisy circle is detected. The
    filtration is VR with ``max_edge_length``: alpha is not feasible at d=20
    (gudhi AlphaComplex targets low ambient dimension), and a filtration cutoff
    cleanly separates the circle (pairwise distances <= 2R) from the 20-D
    random clutter (whose pairwise distances are >> cutoff), so the circle's
    H1 class survives.
    """
    from tda2s.ph import compute_diagrams

    rng = np.random.default_rng(7)
    n_circ, n_clutter = 200, 100
    theta = rng.uniform(0, 2 * np.pi, size=n_circ)
    pts = np.zeros((n_circ, 20))
    pts[:, 0] = np.cos(theta) + rng.normal(0, 0.03, n_circ)
    pts[:, 1] = np.sin(theta) + rng.normal(0, 0.03, n_circ)
    clutter = rng.uniform(-3, 3, size=(n_clutter, 20))
    cloud = np.vstack([pts, clutter])

    diags = compute_diagrams(cloud, filtration="vr", homology_dims=(0, 1),
                             max_edge_length=2.0)
    assert len(diags) == 2
    assert _h1_persistence(diags) > 0.5


def test_external_standardisation():
    """Externally supplied (mean, scale) standardisation, not per-cloud scaling.

    ``compute_diagrams(..., standardise=(mean, scale))`` applies a caller-
    supplied coordinate transform (points - mean) / scale before filtration,
    e.g. a fixed study-region background rather than each cloud's own mean and
    spread. With a shared external standardisation the H1 persistence of two
    noisy circles tracks their radii (both live in one coordinate system);
    per-cloud z-scoring instead maps both to ~sqrt(2) persistence, i.e. it
    changes the diagrams and makes the metric group-dependent. The external
    standardisation is deterministic: the same input yields the same diagram.
    """
    from tda2s.ph import compute_diagrams

    cloud_a = _noisy_circle(radius=1.0, seed=1)
    cloud_b = _noisy_circle(radius=2.0, seed=2)
    external = (np.zeros(2), np.ones(2))

    h1a = compute_diagrams(cloud_a, filtration="alpha", homology_dims=(1,),
                           standardise=external)[0]
    h1b = compute_diagrams(cloud_b, filtration="alpha", homology_dims=(1,),
                           standardise=external)[0]
    h1a_again = compute_diagrams(cloud_a, filtration="alpha", homology_dims=(1,),
                                 standardise=external)[0]

    pa = float(np.max(h1a[:, 1] - h1a[:, 0]))
    pb = float(np.max(h1b[:, 1] - h1b[:, 0]))
    np.testing.assert_array_equal(h1a, h1a_again)  # external is stable/reproducible
    assert pa > 0.3 and pb > 0.3                    # H1 preserved under the shared frame
    assert pb > 1.5 * pa                            # persistence tracks the shared scale

    pc_a = (cloud_a.mean(axis=0), cloud_a.std(axis=0))
    pc_b = (cloud_b.mean(axis=0), cloud_b.std(axis=0))
    h1a_pc = compute_diagrams(cloud_a, filtration="alpha", homology_dims=(1,),
                              standardise=pc_a)[0]
    h1b_pc = compute_diagrams(cloud_b, filtration="alpha", homology_dims=(1,),
                              standardise=pc_b)[0]
    pa_pc = float(np.max(h1a_pc[:, 1] - h1a_pc[:, 0]))
    pb_pc = float(np.max(h1b_pc[:, 1] - h1b_pc[:, 0]))
    assert abs(pa_pc - pb_pc) < 0.2    # per-cloud scaling equalises both radii
    assert abs(pb - pb_pc) > 0.2        # per-cloud scaling changes the diagram


def test_dtm_rips_is_an_option():
    """DTM-Rips is a filtration option and is robust to outliers.

    ``filtration="dtm-rips"`` yields a prominent H1 feature on a noisy circle
    with 10% outliers: DTM vertex weights downweight the outliers, so the main
    loop's persistence stays at least half of the clean circle's. Both clouds
    use DTM-Rips so the comparison is apples-to-apples.
    """
    from tda2s.ph import compute_diagrams

    clean = _noisy_circle(seed=3)
    rng = np.random.default_rng(4)
    outliers = rng.uniform(-3, 3, size=(30, 2))
    contaminated = np.vstack([clean, outliers])

    pers_clean = _h1_persistence(
        compute_diagrams(clean, filtration="dtm-rips", homology_dims=(0, 1),
                         dtm_k=20))
    pers_out = _h1_persistence(
        compute_diagrams(contaminated, filtration="dtm-rips",
                         homology_dims=(0, 1), dtm_k=20))
    assert pers_clean > 0.0
    assert pers_out >= 0.5 * pers_clean
