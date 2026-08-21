"""Focused Phase 5D checks for the hardened SC-B production procedure."""

import numpy as np
import pytest

from tda2s.tests.single_cloud import (
    DEFAULT_KERNEL_BANDWIDTH,
    PRODUCTION_MIN_BLOCKS,
    LOCKED_M,
    _mmd2_from_gram,
    _permutation_pvalue,
    _universal_diagram_kernel,
    sc_b_production_test,
    sc_b_repeated_partition_test,
)


def _joint_kernel_witness():
    low = np.array([[0.0, 0.20]])
    high = np.array([[0.0, 0.50]])
    aligned = [(low, low), (high, high)]
    crossed = [(low, high), (high, low)]

    def additive(left, right):
        return 0.5 * sum(
            _universal_diagram_kernel([a], [b], DEFAULT_KERNEL_BANDWIDTH)
            for a, b in zip(left, right)
        )

    def product(left, right):
        return _universal_diagram_kernel(left, right, DEFAULT_KERNEL_BANDWIDTH)

    def mmd2(left, right, kernel):
        within_left = np.mean([[kernel(a, b) for b in left] for a in left])
        within_right = np.mean([[kernel(a, b) for b in right] for a in right])
        between = np.mean([[kernel(a, b) for b in right] for a in left])
        return max(float(within_left + within_right - 2.0 * between), 0.0)

    return mmd2(aligned, crossed, additive), mmd2(aligned, crossed, product)


def _clouds(n=LOCKED_M * PRODUCTION_MIN_BLOCKS, seed=32):
    rng = np.random.default_rng(seed)
    return rng.uniform(size=(n, 2)), rng.uniform(size=(n, 2))


def test_tensor_product_kernel_sees_cross_degree_dependence():
    additive, product = _joint_kernel_witness()
    assert additive == pytest.approx(0.0, abs=1e-12)
    assert product > 1e-8


def test_production_api_freezes_target_and_reports_effective_sample_size():
    cloud0, cloud1 = _clouds()
    result = sc_b_production_test(
        cloud0, cloud1, n_perm=7, seed=7, partition_seed=11,
    )
    assert result["production_api"] == "sc-b-v1"
    assert result["m"] == LOCKED_M
    assert result["K_a"] == [PRODUCTION_MIN_BLOCKS, PRODUCTION_MIN_BLOCKS]
    assert result["diagnostics"]["sampling_unit"] == "point"
    assert result["diagnostics"]["repeated_partition_aggregation"] == "refused"
    assert result["diagnostics"]["overlapping_blocks_used"] is False
    assert result["diagnostics"]["persistent_homology_recomputed_in_permutation_loop"] is False
    assert "H0,25^bar" in result["inferential_target"]


def test_production_api_refuses_unsupported_regime_and_small_clouds():
    cloud0, cloud1 = _clouds()
    with pytest.raises(ValueError, match="only valid"):
        sc_b_production_test(cloud0, cloud1, regime="fixed_cloud", n_perm=1)
    with pytest.raises(ValueError, match="at least"):
        sc_b_production_test(cloud0[:LOCKED_M], cloud1, n_perm=1)
    with pytest.raises(ValueError, match="data-independent"):
        sc_b_production_test(cloud0, cloud1, partition_is_data_independent=False, n_perm=1)


def test_repeated_partition_aggregation_is_an_explicit_refusal():
    with pytest.raises(ValueError, match="repeated-partition aggregation"):
        sc_b_repeated_partition_test(None, None)


def test_production_api_refuses_supplied_partition_with_too_few_blocks():
    cloud0, cloud1 = _clouds()
    part = np.zeros((PRODUCTION_MIN_BLOCKS - 1, LOCKED_M), dtype=int)
    part[:, :] = np.arange((PRODUCTION_MIN_BLOCKS - 1) * LOCKED_M).reshape(
        PRODUCTION_MIN_BLOCKS - 1, LOCKED_M
    )
    with pytest.raises(ValueError, match="at least"):
        sc_b_production_test(
            cloud0, cloud1, partition0=part, partition1=part,
            partition_seed=None, n_perm=1,
        )


def test_mmd_gram_helper_is_zero_for_constant_kernel():
    gram = np.ones((4, 4), dtype=float)
    group0 = np.array([True, True, False, False])
    assert _mmd2_from_gram(gram, group0) == pytest.approx(0.0)


def test_exact_permutation_pvalue_handles_ties_conservatively():
    observed = 0.0
    null = np.zeros(10, dtype=float)
    assert _permutation_pvalue(observed, null, exact=True) == 1.0
