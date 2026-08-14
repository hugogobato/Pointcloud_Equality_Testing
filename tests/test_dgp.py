"""Phase 0.6 acceptance tests: controlled DGP harness.

Covers knob independence (loop count / radius / noise), covariate-driven
topology, conditional-identical groups given X, group effects, propensity
recovery, and tcda_uq-format silhouette export.
"""

import numpy as np
import pytest
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from tda2s.dgp import (
    CloudSampleDGP,
    circle_cloud,
    cluster_cloud,
    loops_cloud,
    sphere_cloud,
    torus_cloud,
    to_silhouette_sample,
)
from tda2s.ph import compute_diagrams


def _max_h1_persistence(cloud, filtration="alpha"):
    """Largest H1 persistence in the diagram of ``cloud``."""
    dgm = compute_diagrams(cloud, filtration=filtration, homology_dims=(1,))[0]
    if len(dgm) == 0:
        return 0.0
    return float((dgm[:, 1] - dgm[:, 0]).max())


def _recovered_loop_count(cloud, threshold=0.4, filtration="alpha"):
    """Number of H1 features with persistence above ``threshold``."""
    dgm = compute_diagrams(cloud, filtration=filtration, homology_dims=(1,))[0]
    if len(dgm) == 0:
        return 0
    return int(((dgm[:, 1] - dgm[:, 0]) > threshold).sum())


def test_generators_shapes():
    rng = np.random.default_rng(0)
    assert circle_cloud(100, radius=1.0, noise=0.05, rng=rng).shape == (100, 2)
    assert torus_cloud(100, R=2.0, r=0.6, noise=0.05, rng=rng).shape == (100, 3)
    assert sphere_cloud(100, radius=1.0, noise=0.05, rng=rng).shape == (100, 3)
    assert cluster_cloud(100, n_clusters=3, spread=3.0, noise=0.2, rng=rng).shape == (100, 2)
    assert loops_cloud(120, n_loops=3, radius=1.0, noise=0.05, rng=rng).shape == (120, 2)


# ------------------------------------------------------------------ test 1
def test_knob_independence_oracle_and_persistence():
    radius, noise = 1.0, 0.02
    for n_loops in (1, 2, 3):
        knob = lambda x, nl=n_loops: (nl, radius, noise)
        dgp = CloudSampleDGP(n_per_group=6, m=240, d_x=3, beta=np.zeros(3),
                             topology_knob=knob, group_effect=0, seed=0)
        s = dgp.sample(rng=1)
        assert np.all(s.true_n_loops == n_loops)
        assert np.all(s.true_noise == noise)

        pers = [_max_h1_persistence(c) for c in s.clouds]
        assert abs(np.mean(pers) - radius) <= 0.3 * radius


def test_noise_does_not_change_recovered_loop_count():
    for noise in (0.02, 0.1):
        counts = []
        for seed in range(6):
            c = loops_cloud(240, 2, radius=1.0, noise=noise, rng=seed)
            counts.append(_recovered_loop_count(c))
        assert counts == [2] * 6


# ------------------------------------------------------------------ test 2
def test_covariate_driven_topology():
    dgp = CloudSampleDGP(n_per_group=25, m=120, d_x=3, beta=np.zeros(3),
                         gamma=1.5, k_max=3, group_effect=0, seed=0)
    s = dgp.sample(rng=1)
    assert pearsonr(s.X[:, 0], s.true_n_loops).statistic > 0.8
    assert abs(pearsonr(s.X[:, 1], s.true_n_loops).statistic) < 0.2


# ------------------------------------------------------------------ test 3
def test_conditional_identical_groups_given_x():
    n_per_group = 8
    rng = np.random.default_rng(0)
    X0 = rng.normal(size=(n_per_group, 3))
    X = np.vstack([X0, X0])  # every row of group B repeats a row of group A
    dgp = CloudSampleDGP(n_per_group=n_per_group, m=120, d_x=3, beta=np.zeros(3),
                         gamma=1.5, k_max=3, group_effect=0, seed=0)
    s = dgp.sample(X=X, rng=1)
    for i in range(n_per_group):
        a = s.oracle[i]
        b = s.oracle[i + n_per_group]
        assert a["n_loops"] == b["n_loops"]
        assert np.array_equal(a["radii"], b["radii"])
        assert a["noise"] == b["noise"]


# ------------------------------------------------------------------ test 4
def test_group_effect_more_loops_in_A():
    dgp = CloudSampleDGP(n_per_group=25, m=120, d_x=3, beta=np.zeros(3),
                         gamma=1.5, k_max=3, group_effect=1, seed=0)
    s = dgp.sample(rng=1)
    mean_a = s.true_n_loops[s.A == 1].mean()
    mean_b = s.true_n_loops[s.A == 0].mean()
    assert mean_a > mean_b


# ------------------------------------------------------------------ test 5
def test_propensity_auc():
    dgp_strong = CloudSampleDGP(n_per_group=25, m=120, d_x=3,
                                beta=np.array([1.5, 1.0, 0.5]), group_effect=0, seed=0)
    s = dgp_strong.sample(rng=3)
    lr = LogisticRegression(max_iter=2000).fit(s.X, s.A)
    auc = roc_auc_score(s.A, lr.predict_proba(s.X)[:, 1])
    assert auc > 0.8

    dgp_null = CloudSampleDGP(n_per_group=30, m=120, d_x=3,
                              beta=np.zeros(3), group_effect=0, seed=0)
    s = dgp_null.sample(rng=0)
    lr = LogisticRegression(max_iter=2000).fit(s.X, s.A)
    auc = roc_auc_score(s.A, lr.predict_proba(s.X)[:, 1])
    assert abs(auc - 0.5) < 0.15


# ------------------------------------------------------------------ test 6
def test_silhouette_compatibility_with_tcda_uq():
    tc = pytest.importorskip("tcda_uq.silhouette")
    n_per_group = 4
    dgp = CloudSampleDGP(n_per_group=n_per_group, m=100, d_x=3, beta=np.zeros(3),
                         group_effect=0, seed=0)
    s = dgp.sample(rng=0)
    phi, A, X = to_silhouette_sample(s.clouds, s.X, s.A,
                                     interval=(0.0, 1.0), r=3.0, resolution=60)
    assert phi.shape == (2 * n_per_group, 2, 60)
    assert A.shape == (2 * n_per_group,)
    assert X.shape == (2 * n_per_group, 3)

    for i in range(len(s.clouds)):
        diags = compute_diagrams(s.clouds[i], filtration="alpha", homology_dims=(0, 1))
        ref = tc.compute_silhouette(diags, interval=(0.0, 1.0), r=3.0, resolution=60)
        assert np.allclose(phi[i], ref, atol=1e-6)


def test_observed_format_matches_trioracle():
    tri = pytest.importorskip("tcda_uq.datasets")
    n_per_group = 4
    dgp = CloudSampleDGP(n_per_group=n_per_group, m=100, d_x=3, beta=np.zeros(3),
                         group_effect=0, seed=0)
    s = dgp.sample(rng=0)
    phi, A, X = s.observed(interval=(0.0, 1.0), r=3.0, resolution=60)
    ref = tri.TriOracleSimulation(n_cov=3, n_hom_dim=2, resolution=60).sample(8, rng=0)
    ref_phi, ref_A, ref_X = ref.observed
    assert phi.shape == ref_phi.shape
    assert A.shape == ref_A.shape
    assert X.shape == ref_X.shape
    assert isinstance(A[0], (int, np.integer))