"""Published simulation designs, reproduced exactly as their papers specify.

None of the competitor methods in :mod:`tda2s.benchmarks` ships author code, so
each wrapper is a replication from the paper. This module holds the *designs*
those papers simulate, so that the pytest reproductions
(``tests/test_published_reproductions.py``) and the full-fidelity Colab
notebooks (``notebooks/``) run one implementation rather than two that drift.

Every runner is expressed as a function of an explicit list of replication
indices, and replication ``r`` draws from ``default_rng([base_seed, r])``. That
makes a run reproducible *independently of how it is sharded*: replication 7 is
the same experiment whether it is computed alone, in a block of 50, or as part
of a sequential sweep. Shards are therefore concatenated, not averaged, and a
notebook that splits 500 replications over 40 cores gets exactly the result a
single-process run would have produced.

Designs
-------
``moon_lazar_*``
    Moon & Lazar (2023) Section 4.1, itself the design of Robinson & Turner
    (2017): shape 1 is one circle of radius 1, shape 2 is two circles of radii
    0.9 and 1.1; 50 points per cloud plus ``N(0, sigma^2)`` noise; Rips
    complex; dimension-one diagrams. The false-positive scenario draws 20
    clouds from shape 2 and splits them at random into two groups of 10; the
    power scenario draws 10 from each shape. Figure 5 reports both against
    ``sigma``, and its "PD" curve is the same design run through the
    Robinson-Turner test.

``dubey_muller_*``
    Dubey & Muller (2019) Section 5: the random objects are ``N(mu, 1)``
    distributions under the L2-Wasserstein metric. Because
    ``W_2(N(a,1), N(b,1)) = |a - b|``, the space collapses to the real line and
    the Frechet mean is the arithmetic mean, so the statistic is exact and no
    barycentre machinery is involved. Figure 1 left varies a location shift
    ``delta``; Figure 1 right varies a scale ratio ``r`` at equal means, which
    is the panel that only the Levene-type ``U_n`` term of eq. (8) can detect.
"""
from __future__ import annotations

import numpy as np

from tda2s.benchmarks import run_competitor
from tda2s.benchmarks.frechet_anova import frechet_anova_statistic
from tda2s.ph import compute_diagrams

__all__ = [
    "ALPHA",
    "MOON_LAZAR_FIG5",
    "MOON_LAZAR_FIG5B_RT",
    "moon_lazar_cloud",
    "moon_lazar_diagram",
    "moon_lazar_rejections",
    "dubey_muller_statistic",
    "dubey_muller_rejections",
]

#: Nominal level used by both papers.
ALPHA = 0.05

#: Moon & Lazar (2023) Figure 5, read off the published panels (500 reps each):
#: false-positive rate (5a) and power (5b) of the two-stage test.
MOON_LAZAR_FIG5 = {
    0.05: {"fpr": 0.022, "power": 0.98},
    0.10: {"fpr": 0.040, "power": 0.62},
    0.15: {"fpr": 0.035, "power": 0.24},
    0.20: {"fpr": 0.025, "power": 0.10},
}

#: The "PD" curve of Moon & Lazar Figure 5b: the same design run through the
#: Robinson & Turner (2017) test, which is an independent published power curve
#: for the ``rt`` wrapper.
MOON_LAZAR_FIG5B_RT = {0.05: 0.97, 0.10: 0.55, 0.15: 0.20, 0.20: 0.05}

#: Two-stage settings of the paper's Section 4.1 method comparison.
MOON_LAZAR_SETTINGS = dict(resolution=40, bandwidth=0.5, weight="constant",
                           filter_threshold=80.0, adjust="bh", im_range=(0.0, 2.0))


def _rep_rng(base_seed, rep):
    """RNG for replication ``rep``, independent of how replications are sharded."""
    return np.random.default_rng([int(base_seed), int(rep)])


# --------------------------------------------------------------------------
# Moon & Lazar (2023) Figure 5 / Robinson & Turner design
# --------------------------------------------------------------------------

def moon_lazar_cloud(shape, sigma, rng, n=50):
    """One noisy point cloud: ``shape=1`` is one circle, ``shape=2`` is two."""
    th = rng.uniform(0, 2 * np.pi, n)
    r = np.ones(n) if shape == 1 else np.where(np.arange(n) < n // 2, 0.9, 1.1)
    pts = np.column_stack([r * np.cos(th), r * np.sin(th)])
    return pts + rng.normal(0, sigma, pts.shape)


def moon_lazar_diagram(pts):
    """Dimension-one Rips diagram of one cloud, as a one-element dim list."""
    return [compute_diagrams(pts, filtration="ripser", homology_dims=(0, 1))[1]]


def _moon_lazar_groups(scenario, sigma, rng):
    if scenario == "fpr":
        ds = [moon_lazar_diagram(moon_lazar_cloud(2, sigma, rng)) for _ in range(20)]
        idx = rng.permutation(20)
        return [ds[i] for i in idx[:10]], [ds[i] for i in idx[10:]]
    if scenario == "power":
        return ([moon_lazar_diagram(moon_lazar_cloud(1, sigma, rng)) for _ in range(10)],
                [moon_lazar_diagram(moon_lazar_cloud(2, sigma, rng)) for _ in range(10)])
    raise ValueError(f"scenario must be 'fpr' or 'power', got {scenario!r}")


def moon_lazar_rejections(name, sigma, scenario, reps, base_seed, alpha=ALPHA,
                          **kwargs):
    """Reject/accept indicators over ``reps`` replications of the Fig. 5 design.

    Args:
        name: competitor key, e.g. ``"moon_lazar"`` (Fig. 5a/5b) or ``"rt"``
            (the "PD" curve of Fig. 5b).
        sigma: noise level.
        scenario: ``"fpr"`` (both groups from shape 2) or ``"power"``.
        reps: number of replications, or an explicit iterable of replication
            indices -- pass a slice of ``range(n)`` to compute one shard.
        base_seed: replication ``r`` uses ``default_rng([base_seed, r])``.
        alpha: nominal level.
        **kwargs: forwarded to the competitor (filtered by ``run_competitor``).

    Returns:
        ``(rep_ids, rejections)``: the replication indices computed and a
        ``uint8`` array of indicators, so shards can be concatenated.
    """
    rep_ids = np.arange(reps) if np.isscalar(reps) else np.asarray(list(reps))
    out = np.empty(len(rep_ids), dtype=np.uint8)
    for k, rep in enumerate(rep_ids):
        rng = _rep_rng(base_seed, rep)
        g0, g1 = _moon_lazar_groups(scenario, sigma, rng)
        out[k] = run_competitor(name, g0, g1, **kwargs) <= alpha
    return rep_ids, out


# --------------------------------------------------------------------------
# Dubey & Muller (2019) Figure 1
# --------------------------------------------------------------------------

def dubey_muller_statistic(mu, labels):
    """``T_n`` of eq. (11) on the real line, where the Frechet mean is the mean."""
    mu = np.asarray(mu, dtype=float)
    within = np.empty(len(mu))
    for g in np.unique(labels):
        m = labels == g
        within[m] = (mu[m] - mu[m].mean()) ** 2
    pooled = (mu - mu.mean()) ** 2
    return frechet_anova_statistic(within, pooled, labels)


def dubey_muller_rejections(reps, base_seed, delta=0.0, r=1.0, sd=0.5, n=100,
                            n_perm=200, alpha=ALPHA):
    """Reject/accept indicators over ``reps`` replications of the Fig. 1 design.

    Args:
        reps: number of replications, or an explicit iterable of indices.
        base_seed: replication ``rep`` uses ``default_rng([base_seed, rep])``.
        delta: location shift of group 2 (Fig. 1 left sweeps this).
        r: scale ratio of group 2 (Fig. 1 right sweeps this at ``delta = 0``).
        sd: base standard deviation (paper: 0.5 left, 0.2 right).
        n: per-group sample size (paper: 100).
        n_perm: permutations per replication.
        alpha: nominal level.

    Returns:
        ``(rep_ids, rejections)``, concatenable across shards.
    """
    rep_ids = np.arange(reps) if np.isscalar(reps) else np.asarray(list(reps))
    labels = np.r_[np.ones(n, int), np.zeros(n, int)]
    out = np.empty(len(rep_ids), dtype=np.uint8)
    for k, rep in enumerate(rep_ids):
        rng = _rep_rng(base_seed, rep)
        mu = np.r_[np.clip(rng.normal(0.0, sd, n), -10, 10),
                   np.clip(rng.normal(delta, sd * r, n), -10, 10)]
        obs = dubey_muller_statistic(mu, labels)
        null = np.array([dubey_muller_statistic(mu, labels[rng.permutation(2 * n)])
                         for _ in range(n_perm)])
        out[k] = (1 + np.sum(null >= obs)) / (1 + n_perm) < alpha
    return rep_ids, out
