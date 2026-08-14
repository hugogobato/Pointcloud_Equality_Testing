"""Phase 0.8 acceptance: the tcda_uq shim is pure delegation, and it works.

The shim (``tda2s/adapters/tcda_uq.py``) exposes tcda_uq's estimators to P1
with zero reimplemented AIPW / cross-fitting / DR-learner math. These tests pin
the delegation contract:

* tda2s silhouettes equal ``tcda_uq.silhouette.compute_silhouette`` on a fixed
  diagram (noisy circle, computed via ``tda2s.ph.compute_diagrams``);
* ``cross_fit`` run through the shim on a tri-oracle sample recovers the oracle
  TATE within tolerance (sup-norm distance of ``aipw[d]`` vs ``oracle_tate[d]``);
* the shim returns per-unit scores shaped ``(n, n_hom_dim, resolution)``;
* ``CTATEDRLearner`` imports and fits through the shim without error.

Note on covariates: the tri-oracle DGP's propensity model hardcodes the
interaction terms ``X1 * X2`` and ``X0 * X2`` (tcda_uq
``datasets/simulation.py``), so ``n_cov`` must be at least 3. The Phase 0.8
spec's "2 covariates" is realised as the minimal ``n_cov=3``.

Nuisance specification for the oracle-recovery test: tcda_uq's default
propensity model (``RandomForestClassifier()``) carries no ``random_state``,
so per-run EIF draws would be nondeterministic. The test pins a seeded random
forest and modest ``coef_scale``/``noise_scale`` so the sampling noise of the
TATE estimate at n=100 stays well below the 0.2 tolerance; the outcome
regression is in the exact model family of the DGP, so nuisances are
well-specified.
"""
import numpy as np
import pytest

from sklearn.ensemble import RandomForestClassifier

from tda2s.adapters.tcda_uq import aipw_curve, ctate_learner, silhouettes, tri_oracle
from tda2s.ph import compute_diagrams
from tcda_uq.datasets import TriOracleSimulation
from tcda_uq.silhouette import compute_silhouette

N = 100
N_HOM = 2
RES = 50


def _noisy_circle(n=300, radius=1.0, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, size=n)
    pts = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    return pts + rng.normal(0, noise, size=pts.shape)


@pytest.fixture(scope="module")
def tri_sample():
    sim = TriOracleSimulation(n_cov=3, n_hom_dim=N_HOM, resolution=RES,
                              noise_scale=0.15, coef_scale=0.5)
    return sim.sample(N, rng=2)


@pytest.fixture(scope="module")
def aipw_out(tri_sample):
    return aipw_curve(tri_sample.observed, tri_sample.tseq, n_basis=5, n_folds=5,
                      propensity_estimator=RandomForestClassifier(random_state=0))


def test_silhouettes_match_tcda_uq_on_fixed_diagram():
    diags = compute_diagrams(_noisy_circle(), filtration="alpha",
                             homology_dims=(0, 1))
    ours = silhouettes(diags, interval=(0.0, 0.2), r=3, resolution=100)
    theirs = compute_silhouette(diags, interval=(0.0, 0.2), r=3, resolution=100)
    assert ours.shape == theirs.shape == (2, 100)
    np.testing.assert_allclose(ours, theirs, atol=1e-6)


def test_aipw_recovers_oracle_tate(tri_sample, aipw_out):
    dist = max(
        np.max(np.abs(aipw_out["aipw"][d] - tri_sample.oracle_tate[d]))
        for d in range(N_HOM)
    )
    print("AIPW-vs-oracle TATE sup-norm distance:", dist)
    assert dist < 0.2


def test_scores_shape(aipw_out):
    assert aipw_out["scores"].shape == (N, N_HOM, RES)
    assert aipw_out["pi_hat"].shape == (N,)
    assert aipw_out["tseq"].shape == (RES,)
    for d in range(N_HOM):
        np.testing.assert_allclose(
            aipw_out["scores"][:, d, :].mean(axis=0), aipw_out["aipw"][d], atol=1e-10
        )


def test_ctate_learner_fits_through_shim(tri_sample):
    learner = ctate_learner(n_basis=5)
    learner.fit(tri_sample.observed, tri_sample.tseq)
    assert len(learner.stages_) == N_HOM
    pred = learner.predict(tri_sample.X[:3])
    assert pred.shape == (3, N_HOM, RES)


def test_tri_oracle_returns_simulation_sample():
    sample = tri_oracle(50, n_cov=3, n_hom_dim=2, resolution=RES, seed=0)
    assert sample.oracle_tate.shape == (N_HOM, RES)
    assert sample.oracle_ctate.shape == (50, N_HOM, RES)
    assert sample.oracle_itte.shape == (50, N_HOM, RES)
    phi, A, X = sample.observed
    assert phi.shape == (50, N_HOM, RES) and A.shape == (50,) and X.shape == (50, 3)
