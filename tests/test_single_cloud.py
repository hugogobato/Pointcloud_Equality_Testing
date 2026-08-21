"""Phase 5B prototype and routing tests."""

import math

import numpy as np
import pytest

from tda2s.tests.single_cloud import (
    REGIME_I,
    disjoint_partition,
    persistent_betti_vector,
    roycraft_reference_setting,
    run_single_cloud_test,
    sc_a_label_permutation,
    sc_b_disjoint_mmd,
    sc_c_finite_vector,
    sc_c_naive_bootstrap,
)


def _clouds(n0=6, n1=6, seed=4):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n0, 2)), rng.normal(size=(n1, 2))


def test_incompatible_regime_is_rejected_before_inference():
    x0, x1 = _clouds()
    for candidate in ("SC-A", "SC-B", "SC-C"):
        with pytest.raises(ValueError, match="only valid"):
            run_single_cloud_test(
                candidate, x0, x1, regime="stationary_mixing_process", n_perm=1,
            )


def test_sc_a_exact_enumeration_is_complete_and_has_common_fields():
    x0, x1 = _clouds(3, 3)
    result = sc_a_label_permutation(
        x0, x1, homology_dims=(0,), exact=True, max_exact_permutations=100,
    )
    assert result["candidate"] == "SC-A"
    assert result["inferential_target"] == "H0^law: P0 = P1"
    assert result["exact_enumeration"]
    assert len(result["null_statistics"]) == math.comb(6, 3)
    assert result["pvalue"] == pytest.approx(
        np.mean(result["null_statistics"] >= result["statistic"])
    )
    assert result["runtime_seconds"] >= 0.0
    assert result["peak_memory_bytes"] >= 0


def test_sc_b_exposes_only_disjoint_effective_replicates():
    x0, x1 = _clouds(7, 8)
    result = sc_b_disjoint_mmd(
        x0, x1, m=2, homology_dims=(0,), exact=True,
    )
    assert result["K0"] == 7 // 2
    assert result["K1"] == 8 // 2
    assert result["K_a"] == [result["K0"], result["K1"]]
    assert result["remainder0"] == 1
    assert result["remainder1"] == 0
    assert not result["diagnostics"]["overlapping_blocks_used"]
    assert len(np.unique(result["block_indices0"])) == 6
    assert len(result["null_statistics"]) == math.comb(7, 3)
    assert "unlocked sensitivity" in result["inferential_target"]


def test_sc_b_rejects_an_overlapping_confirmatory_partition():
    x0, x1 = _clouds(6, 6)
    overlapping = np.array([[0, 1], [1, 2], [3, 4]])
    with pytest.raises(ValueError, match="overlapping"):
        sc_b_disjoint_mmd(
            x0, x1, m=2, homology_dims=(0,), partition0=overlapping,
            partition1=np.array([[0, 1], [2, 3], [4, 5]]), n_perm=2,
        )


def test_disjoint_partition_returns_floor_count_and_unused_points():
    x0, _ = _clouds(11, 11)
    blocks, indices, remainder = disjoint_partition(x0, m=3, seed=12)
    assert blocks.shape == (3, 3, 2)
    assert indices.shape == (3, 3)
    assert len(remainder) == 2
    assert len(np.unique(indices)) == 9
    assert set(indices.ravel()).isdisjoint(set(remainder))


def test_persistent_betti_vector_uses_one_shared_grid():
    diagram = [np.array([[0.0, 0.5]])]
    vector = persistent_betti_vector(diagram, grid=[0.0, 0.25, 0.5])
    assert vector.shape == (9,)
    # (r,s) = (0,0), (0,.25), and (.25,.25) are the surviving coordinates.
    assert vector.tolist() == [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_sc_c_smoothed_and_naive_negative_control_are_labelled():
    x0, x1 = _clouds(8, 8, seed=22)
    kwargs = dict(
        homology_dims=(0,), grid=np.linspace(0.0, 1.0, 4), n_draws=4,
        bootstrap_bandwidth=0.03,
    )
    smooth = sc_c_finite_vector(x0, x1, **kwargs)
    naive = sc_c_naive_bootstrap(x0, x1, **kwargs)
    assert smooth["bootstrap"] == "smoothed"
    assert naive["bootstrap"] == "naive_negative_control"
    assert "finite-vector" in smooth["inferential_target"]
    assert "H0,25^bar" not in smooth["inferential_target"]
    assert np.isfinite(smooth["null_statistics"]).all()


def test_dispatch_and_source_setting_record():
    x0, x1 = _clouds(4, 4)
    result = run_single_cloud_test("sc-b", x0, x1, m=2,
                                  homology_dims=(0,), n_perm=3)
    assert result["candidate"] == "SC-B"
    source = roycraft_reference_setting()
    assert source["doi"] == "10.1214/23-AOS2277"
    assert source["comparison"] == "Gaussian smoothed bootstrap versus ordinary bootstrap"
    assert REGIME_I == "iid_metric_measure"
