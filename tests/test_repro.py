"""The published designs in ``tda2s.repro`` must not depend on how they shard.

The Colab notebooks split 500 replications across many worker processes and
concatenate the indicator arrays. That is only sound if replication ``r`` is a
function of ``(base_seed, r)`` and nothing else -- in particular not of how many
replications preceded it in the same process. A sequential RNG would break this
silently: the numbers would still look plausible, but they would depend on
``chunk`` and on the order shards happened to complete.

These tests run sequentially and at tiny budgets; the parallel driver itself is
exercised by the notebooks, not here (the suite stays single-process).
"""
import numpy as np
import pytest

from tda2s.repro import (MOON_LAZAR_FIG5, MOON_LAZAR_FIG5B_RT,
                         MOON_LAZAR_SETTINGS, dubey_muller_rejections,
                         moon_lazar_cloud, moon_lazar_rejections)

_SPLITS = [(0, 3), (3, 4), (4, 9)]      # deliberately uneven
_N = 9


def _concat(parts):
    return np.concatenate([p[1] for p in parts])


def test_moon_lazar_sharding_is_invariant():
    kw = dict(MOON_LAZAR_SETTINGS)
    whole = moon_lazar_rejections("moon_lazar", 0.10, "power", _N, 5, **kw)[1]
    parts = [moon_lazar_rejections("moon_lazar", 0.10, "power", range(a, b), 5, **kw)
             for a, b in _SPLITS]
    assert np.array_equal(whole, _concat(parts))


def test_dubey_muller_sharding_is_invariant():
    whole = dubey_muller_rejections(_N, 5, delta=0.4, n_perm=50)[1]
    parts = [dubey_muller_rejections(range(a, b), 5, delta=0.4, n_perm=50)
             for a, b in _SPLITS]
    assert np.array_equal(whole, _concat(parts))


def test_replication_indices_are_returned_for_reassembly():
    """Shards must report which replications they computed, not just how many."""
    ids, rej = dubey_muller_rejections(range(4, 9), 5, delta=0.4, n_perm=20)
    assert np.array_equal(ids, np.arange(4, 9))
    assert rej.shape == (5,)


def test_different_base_seeds_give_different_streams():
    a = dubey_muller_rejections(30, 1, delta=0.3, n_perm=50)[1]
    b = dubey_muller_rejections(30, 2, delta=0.3, n_perm=50)[1]
    assert not np.array_equal(a, b), "base_seed does not change the design draw"


def test_moon_lazar_shapes_differ_as_the_paper_specifies():
    """Shape 1 is one radius-1 circle; shape 2 is radii 0.9 and 1.1."""
    rng = np.random.default_rng(0)
    one = moon_lazar_cloud(1, 0.0, rng, n=200)
    rng = np.random.default_rng(0)
    two = moon_lazar_cloud(2, 0.0, rng, n=200)
    r1 = np.linalg.norm(one, axis=1)
    r2 = np.linalg.norm(two, axis=1)
    assert np.allclose(r1, 1.0)
    assert np.allclose(np.unique(np.round(r2, 6)), [0.9, 1.1])


@pytest.mark.parametrize("table", [MOON_LAZAR_FIG5, MOON_LAZAR_FIG5B_RT])
def test_published_reference_tables_cover_the_same_noise_levels(table):
    assert sorted(table) == [0.05, 0.10, 0.15, 0.20]
