"""Phase 8 schema + determinism tests (fast: smoke-scale cells only)."""

import os

import numpy as np
import pandas as pd
import pytest

from experiments.phase8_master_grid import (
    ALWAYS_TESTS,
    CONTRACT,
    NULL_OF,
    _COMP_SEED_OFFSET,
    _competitor_seed,
    _one_rep,
    cell_path,
)

SHARDS = os.path.join(os.path.dirname(__file__), "..", "results",
                      "phase8_shards")
ALPHA = 0.05


def _smoke_cell():
    path = cell_path("mean", "randomized", 50, 2, None, "smoke")
    assert os.path.exists(path), f"run --mode smoke first ({path})"
    return pd.read_parquet(path)


def test_smoke_schema():
    df = _smoke_cell()
    for col in ("family", "imbalance", "n", "rep", "ok", "error", "fails",
                "skip_single_cloud", "wall_rep", "n0", "n1"):
        assert col in df.columns, f"missing column {col}"
    for t in ALWAYS_TESTS:
        assert f"p_{t}" in df.columns, f"missing p_{t}"
    assert "p_krebs_rademacher" in df.columns
    assert "p_dr_equiv" in df.columns
    assert "p_dist_l1" in df.columns
    # no hidden failures
    assert df["ok"].all(), df.loc[~df["ok"], "error"].tolist()
    # p-values in [0, 1] (NaN allowed only for documented skips)
    for c in [c for c in df.columns if c.startswith("p_")]:
        v = df[c].dropna().to_numpy()
        assert ((v >= 0.0) & (v <= 1.0)).all(), c
    # single-cloud skip logged, never silent
    assert df["skip_single_cloud"].str.contains("out of scope").all()
    # timers present
    assert any(c.startswith("t_") for c in df.columns)


def test_determinism_byte_identical():
    """One coarse-scale rep rerun from fixed seed reproduces p-values."""
    kw = dict(family="mean", imb="randomized", n=50, rep=7,
              n_perm=19, dr_draws=39, dist_draws=99)
    r1 = _one_rep(**kw)
    r2 = _one_rep(**kw)
    assert r1["ok"] and r2["ok"]
    for c in [k for k in r1 if k.startswith("p_")]:
        a, b = r1[c], r2[c]
        if isinstance(a, float) and np.isnan(a):
            assert isinstance(b, float) and np.isnan(b), c
        else:
            assert a == b, f"{c}: {a} != {b}"


def test_null_headers_complete():
    for t in list(ALWAYS_TESTS) + ["krebs_rademacher", "dr_equiv", "dist_l1"]:
        assert t in NULL_OF, f"no null header for {t}"
    # competitors test H0^cond, ours test H0^out / H0^dist-grid
    for t in ("rt", "mmd", "han", "strand", "moon_lazar", "frechet_anova"):
        assert NULL_OF[t] == "H0^cond"
    assert NULL_OF["dr_mult"] == "H0^out"
    assert NULL_OF["dist_maxt"] == "H0^dist-grid"


def test_contract_keys():
    for k in ("base_seed", "n_perm_coarse", "eps_rt", "n_bins_dist",
              "sil_interval"):
        assert k in CONTRACT, k
    assert CONTRACT["n_bins_dist"] == 32  # frozen Phase 4.5 grid


def test_competitor_seeds_deterministic():
    """Competitor seeds must be fixed constants, never hash(name).

    Regression test for the 2026-09-10 find: hash() is randomized per
    process (PYTHONHASHSEED), which made the mmd/han/strand/moon_lazar
    columns irreproducible across worker processes while every
    fixed-offset column was byte-identical.
    """
    assert set(_COMP_SEED_OFFSET) == {"mmd", "han", "strand", "moon_lazar",
                                      "frechet_anova"}
    assert _COMP_SEED_OFFSET["frechet_anova"] == 11  # historic offset kept
    assert len(set(_COMP_SEED_OFFSET.values())) == 5  # all distinct
    for name, off in _COMP_SEED_OFFSET.items():
        assert _competitor_seed(1000, name) == 1000 + off
    import inspect
    import experiments.phase8_master_grid as grid
    src = inspect.getsource(grid._one_rep)
    assert "hash(" not in src, "hash() must not seed any test"
