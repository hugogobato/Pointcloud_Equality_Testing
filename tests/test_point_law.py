import numpy as np
import pytest

from experiments.phase5ab_pointlaw_tournament import Cell, _cloud_seed, run_replicates, summarize
from tda2s.tests.point_law import friedman_rafsky_mst, rosenbaum_crossmatch, sliced_wasserstein_test


def test_mst_keeps_duplicate_points_connected():
    cloud = np.zeros((6, 2))
    result = friedman_rafsky_mst(cloud[:3], cloud[3:], n_perm=9, seed=3)
    assert result["n_edges"] == 5
    assert result["n_components"] == 1


def test_cloud_seed_is_independent_of_block_size():
    cell_m1 = Cell("iid_null", 50, 50, 1, 2, "gating_null", "iid null")
    cell_m25 = Cell("iid_null", 50, 50, 25, 2, "gating_null", "iid null")
    assert _cloud_seed(cell_m1, 7) == _cloud_seed(cell_m25, 7)


def test_summary_does_not_pool_different_sample_sizes():
    frame = run_replicates(
        families=("iid_null",),
        n_grid=(50, 250),
        replications=1,
        n_permutations=3,
        workers=1,
        candidates=("PointMMD-Gaussian",),
    )
    summary = summarize(frame)
    assert sorted(summary["n0"].astype(int).unique()) == [50, 250]
    assert len(summary) == 2


def test_crossmatch_refuses_only_above_pooled_500_by_default():
    rng = np.random.default_rng(5)
    x0 = rng.normal(size=(251, 2))
    x1 = rng.normal(size=(250, 2))
    with pytest.raises(ValueError, match="exceeds CrossMatch limit 500"):
        rosenbaum_crossmatch(x0, x1, n_perm=1, seed=1)


def test_sliced_wasserstein_rejects_unimplemented_p2():
    rng = np.random.default_rng(6)
    x0 = rng.normal(size=(8, 2))
    x1 = rng.normal(size=(8, 2))
    with pytest.raises(ValueError, match="only p=1"):
        sliced_wasserstein_test(x0, x1, p=2, n_projections=2, n_perm=1, seed=1)
