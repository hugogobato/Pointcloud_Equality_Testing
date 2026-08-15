"""Phase 2.2 acceptance tests: masking DGP, the DR prototype, and the sweep.

Covers the exact Simpson-cancellation construction (marginal observational
laws coincide between arms while the covariate-standardized topological effect
is non-zero), the DGP's oracle, a smoke test of the DR prototype
statistic/null pipeline on the tri-oracle-shaped data, and the invariants the
imbalance sweep relies on when it shares one pooled sample across six group
splits: the precomputed matrices must be read back under the *observed* labels
and must reproduce the direct wrappers exactly.
"""

import numpy as np
import pytest
from scipy.stats import ks_2samp

from tda2s.adapters.dr_test import prototype_dr_from_phi, prototype_dr_pvalue
# NB: imported as a module, not by name -- the wrappers are called ``test_*``
# and pytest would collect them as (fixture-less, erroring) test functions.
from tda2s import benchmarks as bm
from tda2s.benchmarks import han_kernels, mmd_gram
from tda2s.benchmarks._common import _block_means
from tda2s.dgp import CloudSampleDGP, masking_stratum_sample, to_silhouette_sample
from tda2s.ph import compute_diagrams
from tda2s.vec import silhouette


def _pooled_h1_persistence(diags):
    """All H1 persistences across the samples of ``diags``."""
    out = []
    for d in diags:
        p = d[1][:, 1] - d[1][:, 0]
        if len(p):
            out.append(p)
    return np.concatenate(out) if out else np.empty(0)


def _silhouette_triplet(sample, interval=(0.0, 2.0), resolution=100):
    return to_silhouette_sample(sample.clouds, sample.X, sample.A,
                                filtration="alpha", homology_dims=(0, 1),
                                interval=interval, r=3.0, resolution=resolution)


# ---------------------------------------------------------------- DGP tests
def test_masking_law_equality_h1_persistence():
    """The marginal H1-persistence law must coincide between arms."""
    pers1, pers0 = [], []
    for seed in range(6):
        sample = masking_stratum_sample(60, seed=seed)
        diags = [compute_diagrams(c, homology_dims=(0, 1)) for c in sample.clouds]
        mask = sample.A.astype(bool)
        pers1.append(_pooled_h1_persistence([d for d, m in zip(diags, mask) if m]))
        pers0.append(_pooled_h1_persistence([d for d, m in zip(diags, mask) if not m]))
    pers1, pers0 = np.concatenate(pers1), np.concatenate(pers0)
    assert len(pers1) > 200 and len(pers0) > 200
    assert abs(pers1.mean() - pers0.mean()) < 0.05
    assert ks_2samp(pers1, pers0).pvalue > 0.05


def test_masking_effect_present_in_oracle_and_standardized_contrast():
    """H0^out is false: treated and control conditional laws differ at x=2."""
    sample = masking_stratum_sample(60, seed=0)
    t_loops = [o["n_loops"] for i, o in enumerate(sample.oracle.values())
               if o["stratum"] == 2 and sample.A[i]]
    # treat: 2 loops; control: mixture of 1-loop/2-loop types at x=2
    c_loops = []
    for i, o in enumerate(sample.oracle.values()):
        if o["stratum"] == 2 and not sample.A[i]:
            c_loops.append(o["n_loops"])
    assert set(t_loops) == {2}
    assert 1 in set(c_loops)


def test_masking_stratum_mixture_weights():
    """The control mixture at stratum 2 is (4/15, 4/15, 7/15)."""
    n = 2000
    sample = masking_stratum_sample(n, seed=0)
    from tda2s.dgp.simulation import _MASK_W0, _MASK_W1
    idx = np.flatnonzero((sample.X[:, 0] == 2) & (sample.A == 0))
    loops = np.array([sample.oracle[i]["n_loops"] for i in idx])
    frac_a = float(np.mean(loops == 1))
    frac_c = float(np.mean(loops == 2))
    assert abs(frac_a - 2 * _MASK_W0) < 0.03
    assert abs(frac_c - _MASK_W1) < 0.03


def test_masking_dgp_x_and_propensity():
    sample = masking_stratum_sample(90, seed=1)
    assert sample.X.shape == (180, 1)
    assert set(np.unique(sample.X)) == {0.0, 1.0, 2.0}
    # propensity matches the (1/2, 1/2, 1/4) design
    for x in (0.0, 1.0):
        ii = np.flatnonzero(sample.X[:, 0] == x)
        assert abs(sample.propensity[ii].mean() - 0.5) < 1e-12
    ii = np.flatnonzero(sample.X[:, 0] == 2.0)
    assert abs(sample.propensity[ii].mean() - 0.25) < 1e-12
    # observed arm fractions match propensity (binomial noise)
    assert 0.30 < sample.A.mean() < 0.45


def test_masking_silhouette_contrast_nonzero():
    """The standardized silhouette effect psi_d must be non-zero.

    With m_A = Lambda_{r_A}, m_B = Lambda_{r_B}, m_C = Lambda_{r_C} (r_C far
    from both r_A, r_B), ``psi_d = (1/3)(8/15)(m_C - (m_A + m_B)/2)`` does not
    vanish on the grid, so a mean contrast between the arms' silhouettes
    weighted by the true propensity is bounded away from zero.
    """
    sample = masking_stratum_sample(120, seed=0)
    phi, _, _ = _silhouette_triplet(sample)
    tseq = np.linspace(0.0, 2.0, 100)
    e = sample.propensity
    n = len(sample.clouds)
    # IPW contrast of the TATE curve (true propensity): nonzero if psi_d != 0
    inv = sample.A / e - (1 - sample.A) / (1 - e)
    tate = (inv[:, None, None] * phi).mean(axis=0)          # (2, 100)
    sup = float(np.abs(tate).max())
    assert sup > 0.05, "expected a visible standardized effect at n=120"


# ------------------------------------------------------------- DR prototype
def test_dr_prototype_pipeline_smoke():
    """End-to-end smoke: p-value in [0, 1] and deterministic in the seed."""
    sample = masking_stratum_sample(50, seed=2)
    phi, _, _ = _silhouette_triplet(sample, resolution=60)
    tseq = np.linspace(0.0, 2.0, 60)
    p = prototype_dr_from_phi(phi, sample.A, sample.X, tseq, n_basis=6,
                              n_folds=2, n_draws=200, seed=7)
    assert 0.0 <= p <= 1.0
    p2 = prototype_dr_from_phi(phi, sample.A, sample.X, tseq, n_basis=6,
                               n_folds=2, n_draws=200, seed=7)
    assert p == p2


def test_dr_prototype_null_behaviour_randomised():
    """Under a randomised design (beta = 0) the DR prototype does not fire."""
    dgp = CloudSampleDGP(n_per_group=30, m=120, d_x=3, beta=np.zeros(3),
                         group_effect=0, seed=0)
    s = dgp.sample(rng=3)
    p = prototype_dr_pvalue(s.clouds, s.X, s.A, n_basis=6, n_folds=2,
                            n_draws=200, seed=1)
    assert p > 0.05


# ------------------------------------------------- sweep precompute invariants
@pytest.fixture(scope="module")
def _split_sample():
    """A small pooled sample with *interleaved* (unsorted) group labels.

    Group 1 clouds carry two loops, group 0 one loop, so the two arms are
    genuinely distinguishable; the labels alternate, so any code path that
    assumes "the first n0 rows are group 0" sees an essentially random split.
    """
    from tda2s.dgp.simulation import loops_cloud

    rng = np.random.default_rng(0)
    A = np.tile([0, 1], 14)
    clouds = [loops_cloud(70, 2 if a else 1, radius=1.0, noise=0.05, rng=rng)
              for a in A]
    diags = [compute_diagrams(c, homology_dims=(0, 1)) for c in clouds]
    return diags, A


def test_readback_matches_direct_wrapper(_split_sample):
    """mmd_gram/han_kernels + read-back reproduce the wrappers exactly.

    Same pooled row order on both sides, so the permutation draws coincide and
    the two paths must agree to the last bit, not merely in distribution.
    """
    diags, A = _split_sample
    mask = A.astype(bool)
    d0 = [d for d, m in zip(diags, mask) if not m]
    d1 = [d for d, m in zip(diags, mask) if m]
    pooled = d0 + d1
    g_sorted = np.zeros(len(pooled), dtype=bool)
    g_sorted[:len(d0)] = True

    P, _ = mmd_gram(pooled, epsilon=0.1)
    assert bm.test_mmd_from_gram(P, g_sorted, n_perm=100, seed=5) == \
        bm.test_mmd(d0, d1, n_perm=100, seed=5, epsilon=0.1)

    Ps = han_kernels(pooled, epsilon=0.1)
    assert bm.test_han_from_kernels(Ps, g_sorted, n_perm=100, seed=5) == \
        bm.test_han(d0, d1, n_perm=100, seed=5, epsilon=0.1)


def test_rt_from_matrix_reads_the_observed_labels(_split_sample):
    """A shared matrix must be scored under the labels, not under row position.

    Regression test: ``bm.test_rt_from_matrix(P, n0)`` on a matrix whose rows are
    *not* sorted by label evaluates the observed statistic on an arbitrary
    split, so the p-value is a draw from the null however strong the true group
    difference is. The mask form must recover the same observed statistic as
    sorting the rows, and must reject where the integer form does not.
    """
    import experiments.phase2_imbalance_sweep as sweep

    diags, A = _split_sample
    mask = A.astype(bool)
    P = sweep._rt_matrix(diags)                    # replication's own row order
    d0 = [d for d, m in zip(diags, mask) if not m]
    d1 = [d for d, m in zip(diags, mask) if m]
    P_sorted = sweep._rt_matrix(d0 + d1)           # rows sorted by label
    g_sorted = np.zeros(len(diags), dtype=bool)
    g_sorted[:len(d0)] = True

    w0, w1, _ = _block_means(P, ~mask)
    s0, s1, _ = _block_means(P_sorted, g_sorted)
    assert w0 + w1 == pytest.approx(s0 + s1, rel=1e-12)

    p_mask = bm.test_rt_from_matrix(P, ~mask, n_perm=300, seed=1)
    p_positional = bm.test_rt_from_matrix(P, len(d0), n_perm=300, seed=1)
    assert p_mask < 0.05 < p_positional


def test_rt_from_matrix_rejects_a_mask_of_the_wrong_length():
    P = np.zeros((6, 6))
    with pytest.raises(ValueError):
        bm.test_rt_from_matrix(P, np.ones(5, dtype=bool), n_perm=10)


def test_sweep_replication_is_reproducible(monkeypatch):
    """Two calls of the same replication index must return identical p-values.

    The group labels used to be drawn from a mutable-default RNG shared across
    calls, so a replication's answer depended on how many replications had
    already run in the process: shard *i* on Colab and replication *i* here
    disagreed, and a shard was not reproducible from its index alone.
    """
    import experiments.phase2_imbalance_sweep as sweep

    monkeypatch.setattr(sweep, "N_PER_GROUP", 8)
    monkeypatch.setattr(sweep, "M", 60)
    monkeypatch.setattr(sweep, "N_PERM", 30)
    monkeypatch.setattr(sweep, "N_DRAWS", 100)
    monkeypatch.setattr(sweep, "LAMBDAS", np.array([0.0, 1.0]))

    first = sweep.rep_false_positive(0)
    sweep.rep_false_positive(1)          # advance any shared RNG state
    second = sweep.rep_false_positive(0)
    assert first == second
