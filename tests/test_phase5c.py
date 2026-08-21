"""Mechanical checks for the Phase 5C selection fleet."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from experiments.phase5_single_cloud_tournament import (
    BETTI_GRID,
    CORE_FAMILIES,
    FAMILY_ROLE,
    M_GRID,
    N_GRID,
    PRIMARY_M,
    ROBUSTNESS_FAMILIES,
    SIZE_BAND,
    _mc_interval,
    _overlapping_blocks,
    _required_gate_cells,
    design_record,
    make_cells,
    make_cloud_pair,
    parse_cell_id,
    required_k_counts,
)


def test_frozen_m_grid_spans_required_effective_block_counts():
    counts = required_k_counts()
    assert counts[50] == [5, 10, 20]
    assert counts[25] == [10, 20, 40]
    assert set(M_GRID) == {25, 50}
    assert PRIMARY_M == 25
    assert set(N_GRID) == {250, 500, 1000}


def test_design_record_freezes_gate_and_target_parameters():
    record = design_record()
    assert record["alpha"] == pytest.approx(0.05)
    assert record["size_band"] == list(SIZE_BAND)
    assert record["primary_m"] == PRIMARY_M
    assert record["target"].startswith("H0,25")
    assert record["sc_a_projection_n"] == 250
    assert record["overlap_fractions"] == [0.0, 0.25, 0.5, 0.75, 0.9]


def test_cell_ids_round_trip_and_unequal_cardinality_is_explicit():
    cell = parse_cell_id("robust_unequal_cardinality_n250_n1313_m25")
    assert cell.n0 == 250
    assert cell.n1 == 313
    assert cell.m == 25
    assert cell.role == "robustness_diagnostic"
    assert any(c.cell_id == cell.cell_id for c in make_cells())


def test_core_generators_are_deterministic_and_have_requested_cardinality():
    for family in CORE_FAMILIES:
        x0, x1 = make_cloud_pair(family, 32, 37, 91)
        y0, y1 = make_cloud_pair(family, 32, 37, 91)
        assert x0.shape == (32, 2)
        assert x1.shape == (37, 2)
        assert np.array_equal(x0, y0)
        assert np.array_equal(x1, y1)


def test_overlap_constructor_reports_real_reuse_without_exceeding_cloud():
    cloud = np.arange(500 * 2, dtype=float).reshape(500, 2)
    blocks, indices, average = _overlapping_blocks(
        cloud, m=25, overlap_fraction=0.75, seed=7, K=20,
    )
    assert len(blocks) == 20
    assert indices.shape == (20, 25)
    assert len(np.unique(indices)) < indices.size
    assert 0.0 < average < 1.0


def test_monte_carlo_interval_is_bounded_and_contains_observed_rate():
    low, high = _mc_interval(25, 500)
    assert 0.0 <= low <= 0.05 <= high <= 1.0
    assert _mc_interval(0, 500)[0] == 0.0
    assert _mc_interval(500, 500)[1] == 1.0


def test_betti_grid_is_frozen_before_the_fleet():
    assert np.array_equal(BETTI_GRID, np.linspace(0.0, 0.60, 9))


def test_overlap_negative_control_schema_is_aggregatable(monkeypatch):
    import experiments.phase5_single_cloud_tournament as fleet

    def fake_control(cloud0, cloud1, *, m, overlap_fraction, seed,
                     n_permutations, cache_dir):
        return {
            "candidate": "SC-B-overlap-negative-control",
            "statistic": 1.0,
            "pvalue": 0.5,
            "K0": 4,
            "K1": 4,
            "null_statistics": np.array([0.5]),
            "unique_points0": 100,
            "unique_points1": 100,
            "mean_pairwise_overlap0": float(overlap_fraction),
            "mean_pairwise_overlap1": float(overlap_fraction),
        }

    monkeypatch.setattr(fleet, "overlapping_negative_control", fake_control)
    with tempfile.TemporaryDirectory() as directory:
        out = os.path.join(directory, "phase5c_overlap_negative_control.parquet")
        fleet.run_overlap_fleet(replications=2, workers=1, cache_dir=None, output=out)
        frame = fleet._read_shards(directory)
        needed = {"statistic", "target", "effective_barcode_n0", "n_resamples",
                  "runtime_seconds", "overlapping_blocks_used", "null_role"}
        assert needed <= set(frame.columns)
        summary = fleet.summarize(frame)
        overlap = summary[summary["record_type"] == "negative_control"]
        assert overlap["rejection_rate"].notna().all()
        assert overlap["mean_pairwise_overlap0"].notna().all()


def test_gate_verdict_requires_plan_replication_count():
    import experiments.phase5_single_cloud_tournament as fleet

    def make_summary(successful):
        rows = []
        for family in ("iid_null", "weak_barcode_null"):
            for n in N_GRID:
                rows.append({
                    "record_type": "candidate", "family": family, "candidate": "SC-B",
                    "m": PRIMARY_M, "n0": n, "n1": n, "overlap_fraction": 0.0,
                    "successful_replications": successful, "failed_replications": 0,
                    "rejection_rate": 0.05, "in_size_band": True,
                })
        rows.append({
            "record_type": "candidate", "family": "topology_alt", "candidate": "SC-B",
            "m": PRIMARY_M, "n0": 1000, "n1": 1000, "overlap_fraction": 0.0,
            "successful_replications": successful, "failed_replications": 0,
            "rejection_rate": 1.0, "in_size_band": False,
        })
        return pd.DataFrame(rows)

    assert fleet._gate_verdict(make_summary(500))["verdict"] == "GO"
    assert fleet._gate_verdict(make_summary(300))["verdict"] == "INCOMPLETE"


def test_robustness_families_are_labeled_diagnostics_not_gating_nulls():
    for family in ROBUSTNESS_FAMILIES:
        assert FAMILY_ROLE[family] == "robustness_diagnostic"
    assert not any("robust_" in cell for cell in _required_gate_cells())
