"""Focused checks for the raw and hybrid disjoint-block candidates."""

import math

import numpy as np
import pytest

import tda2s.tests.single_cloud as single_cloud
from tda2s.tests.single_cloud import (
    hybrid_block_mmd,
    raw_block_mmd,
    raw_block_repeated_partition_test,
    sc_a_blockwise_label_permutation,
)


def _clouds(n=6, seed=91):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 2)), rng.normal(size=(n, 2))


def test_raw_block_partition_and_remainder_accounting():
    x0, x1 = _clouds(5)
    result = raw_block_mmd(x0, x1, m=2, exact=True, max_exact_permutations=100)
    assert result["K0"] == result["K1"] == 2
    assert result["remainder0"] == result["remainder1"] == 1
    assert result["diagnostics"]["effective_sample_size"]["total"] == 4
    assert result["unused_point_counts"] == [1, 1]
    assert result["diagnostics"]["overlapping_blocks_used"] is False


def test_raw_block_exact_tiny_permutation_group_and_schema():
    x0, x1 = _clouds(4)
    result = raw_block_mmd(x0, x1, m=2, exact=True, max_exact_permutations=100)
    assert len(result["null_statistics"]) == math.comb(4, 2)
    assert result["pvalue"] == pytest.approx(
        np.mean(result["null_statistics"] >= result["statistic"])
    )
    required = {
        "candidate", "inferential_target", "sampling_unit", "m", "K0", "K1",
        "statistic", "pvalue", "n_permutations", "runtime_seconds",
        "peak_memory_bytes", "diagnostics", "block_indices0", "block_indices1",
    }
    assert required - set(result) == set()
    assert "H0^law: P0 = P1" in result["inferential_target"]
    assert "H0,2^bar" not in result["inferential_target"]


def test_raw_kernel_is_invariant_to_within_block_ordering():
    x0, x1 = _clouds(6)
    part0 = np.array([[0, 1], [2, 3], [4, 5]])
    part1 = np.array([[0, 1], [2, 3], [4, 5]])
    ordered = raw_block_mmd(
        x0, x1, m=2, partition0=part0, partition1=part1,
        exact=True, max_exact_permutations=100,
    )
    reversed_within = raw_block_mmd(
        x0, x1, m=2, partition0=part0[:, ::-1], partition1=part1[:, ::-1],
        exact=True, max_exact_permutations=100,
    )
    assert ordered["statistic"] == pytest.approx(reversed_within["statistic"])
    assert np.allclose(ordered["null_statistics"], reversed_within["null_statistics"])


def test_hybrid_computes_persistence_once_before_permutations(monkeypatch):
    x0, x1 = _clouds(6)
    calls = []
    original = single_cloud._ph

    def counting_ph(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(single_cloud, "_ph", counting_ph)
    result = hybrid_block_mmd(
        x0, x1, m=2, alpha=0.5, homology_dims=(0,), n_perm=7, seed=5,
    )
    assert len(calls) == result["K0"] + result["K1"]
    assert result["diagnostics"]["persistent_homology_recomputed_in_permutation_loop"] is False
    assert result["diagnostics"]["raw_features_recomputed_in_permutation_loop"] is False


def test_hybrid_target_is_barcode_only_only_at_alpha_zero():
    x0, x1 = _clouds(4)
    positive = hybrid_block_mmd(x0, x1, m=2, alpha=0.5, n_perm=3, seed=2)
    zero = hybrid_block_mmd(x0, x1, m=2, alpha=0.0, n_perm=3, seed=2)
    assert "H0^law: P0 = P1" in positive["inferential_target"]
    assert "H0,2^bar" not in positive["inferential_target"]
    assert "H0,2^bar" in zero["inferential_target"]


def test_block_label_permutation_preserves_k0_k1_and_is_reproducible():
    x0, x1 = _clouds(8)
    first = raw_block_mmd(x0, x1, m=2, n_perm=25, seed=12, partition_seed=13)
    second = raw_block_mmd(x0, x1, m=2, n_perm=25, seed=12, partition_seed=13)
    assert first["K_a"] == [4, 4]
    assert first["diagnostics"]["permutation_group"] == (
        "all block-label splits with K0=4, K1=4"
    )
    assert first["statistic"] == pytest.approx(second["statistic"])
    assert first["pvalue"] == pytest.approx(second["pvalue"])
    assert np.array_equal(first["null_statistics"], second["null_statistics"])


def test_overlapping_and_unsupported_controls_are_refused():
    x0, x1 = _clouds(6)
    overlapping = np.array([[0, 1], [1, 2], [3, 4]])
    disjoint = np.array([[0, 1], [2, 3], [4, 5]])
    with pytest.raises(ValueError, match="overlapping"):
        raw_block_mmd(x0, x1, m=2, partition0=overlapping, partition1=disjoint)
    with pytest.raises(ValueError, match="repeated-partition aggregation"):
        raw_block_repeated_partition_test(x0, x1)
    with pytest.raises(ValueError, match="data-independent"):
        raw_block_mmd(x0, x1, m=2, partition_is_data_independent=False)
    with pytest.raises(ValueError, match="label-dependent"):
        raw_block_mmd(x0, x1, m=2, feature_tuning_is_label_independent=False)


def test_sc_a_block_is_same_cached_raw_statistic_with_same_partitions():
    x0, x1 = _clouds(6)
    part = np.array([[0, 1], [2, 3], [4, 5]])
    raw = raw_block_mmd(
        x0, x1, m=2, partition0=part, partition1=part,
        exact=True, max_exact_permutations=100,
    )
    pooled = sc_a_blockwise_label_permutation(
        x0, x1, m=2, partition0=part, partition1=part,
        exact=True, max_exact_permutations=100,
    )
    assert pooled["candidate"] == "SC-A-Block"
    assert pooled["diagnostics"]["equivalent_to"].startswith("RawBlockMMD")
    assert pooled["statistic"] == pytest.approx(raw["statistic"])
    assert np.allclose(pooled["null_statistics"], raw["null_statistics"])


def test_m_one_is_a_point_level_sensitivity_baseline():
    x0, x1 = _clouds(3)
    result = raw_block_mmd(x0, x1, m=1, exact=True, max_exact_permutations=100)
    assert result["K0"] == result["K1"] == 3
    assert result["m"] == 1
