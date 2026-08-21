"""Mechanical checks for the Phase 5A estimand witnesses and design lock."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from phase5_estimand_witnesses import run_witnesses  # noqa: E402


def test_phase5a_witness_certificate_passes():
    result = run_witnesses()
    assert all(result["checks"].values())
    assert result["primary_lock"]["m"] == 25
    assert result["primary_lock"]["filtration"] == "vr"


def test_same_support_density_witness_is_not_a_support_topology_witness():
    witness = run_witnesses()["witnesses"]["same_support_different_density"]
    assert witness["support_equal"]
    assert witness["same_support_topology"]
    assert witness["point_law_total_variation"] > 0
    assert witness["barcode_law_tv_m2"] > 0


def test_finite_summary_does_not_define_the_locked_barcode_law():
    witness = run_witnesses()["witnesses"]["same_finite_summary_different_point_laws"]
    assert witness["finite_summary_equal"]
    assert witness["barcode_law_tv_m2"] < 1e-10
    assert witness["barcode_law_tv_m25_lower_bound"] > 0


def test_fixed_cloud_has_no_default_p_value():
    witness = run_witnesses()["witnesses"]["fixed_cloud_ambiguity"]
    assert witness["positive_likelihood_for_all_tail_scales"]
    rows = witness["rows"]
    assert rows[0]["shared_model_mean_norm"] < rows[-1]["shared_model_mean_norm"]
