"""STRAND: survival-framing two-sample test for collections of persistence
diagrams (Murris, Stolz & Borgwardt 2026).

Reference: J. Murris, B. Stolz and K. Borgwardt, "From persistence to
survival: hypothesis testing, effect sizes and vectorisation for topological
features", arXiv:2606.11911 (2026).  Paper accessed in full on arXiv.

Method (paper Section 3.2 and Appendix E):
    * Each topological feature with persistence p = d - b is treated as a
      fully observed survival time.  Persistence values are pooled within each
      group, and the persistence survival function S(t) = P(p > t) is
      compared between groups with the log-rank test.
    * The log-rank statistic is computed per homology dimension (stratum) and
      combined into the stratified statistic
      ``chi^2 = (sum_dims (O_d - E_d))^2 / sum_dims V_d``, where ``O_d - E_d``
      is the observed-minus-expected event count and ``V_d`` the
      hypergeometric variance in dimension d (paper Eq. 6-7).
    * The null distribution is obtained by permuting the group labels at the
      *diagram* level (paper Appendix E), which preserves within-diagram
      dependence exactly and is exchangeable under the null; the paper's
      asymptotic chi-square p-value is also available via ``asymptotic=True``.

Degenerate cases (no features in either group, zero variance) are mapped to
p = 1 or to an infinite statistic (reject), respectively.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ._common import _perm_groups, _perm_pvalue, _persistence, _validate


def _logrank(e0, e1):
    """Log-rank (O - E, V) for two event-time samples (persistence values)."""
    if len(e0) == 0 and len(e1) == 0:
        return 0.0, 0.0
    e = np.concatenate([e0, e1])
    g = np.concatenate([np.zeros(len(e0)), np.ones(len(e1))])
    if len(e) == 0:
        return 0.0, 0.0
    order = np.argsort(e, kind="mergesort")
    e, g = e[order], g[order]
    _, inv = np.unique(e, return_inverse=True)
    ntimes = int(inv.max()) + 1
    tot = np.bincount(inv, minlength=ntimes)
    d0 = np.bincount(inv[g == 0], minlength=ntimes)
    d1 = tot - d0
    # at risk just before t_j: features with persistence >= t_j
    c0 = np.concatenate([[0], np.cumsum(d0)[:-1]])
    c1 = np.concatenate([[0], np.cumsum(d1)[:-1]])
    n0 = len(e0) - c0
    n1 = len(e1) - c1
    nj = n0 + n1
    dj = tot
    with np.errstate(divide="ignore", invalid="ignore"):
        E = n0 * dj / nj
        V = n0 * n1 * dj * (nj - dj) / (nj * nj * (nj - 1))
    V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
    Z = float(np.sum(d0 - E))
    Vsum = float(np.sum(V))
    return Z, Vsum


def _stratified_stat(e0_list, e1_list):
    """Stratified log-rank statistic ``Z^2 / V`` over homology dims.

    Returns ``(Z, V)`` so callers can detect the degenerate case V = 0.
    """
    Z = V = 0.0
    for e0, e1 in zip(e0_list, e1_list):
        z, v = _logrank(e0, e1)
        Z += z
        V += v
    return Z, V


def _stat_value(Z, V):
    if V > 0:
        return float(Z * Z / V)
    return 0.0 if Z == 0 else np.inf


def test_strand(diags0, diags1, n_perm=200, seed=None, asymptotic=False):
    """STRAND log-rank two-sample test on pooled feature lifetimes.

    Args:
        diags0, diags1: lists (over samples) of lists (per homology dim) of
            ``(k, 2)`` birth-death arrays.
        n_perm: number of diagram-level label permutations (default 200).
        seed: RNG seed for the permutations.
        asymptotic: if True, use the asymptotic chi-square(1) p-value of the
            stratified log-rank statistic instead of the permutation null.

    Returns:
        float p-value in [0, 1].
    """
    d0, d1, nd = _validate(diags0, diags1)
    n0, n1 = len(d0), len(d1)
    N = n0 + n1
    pooled = d0 + d1
    rng = np.random.default_rng(seed)

    # per-sample, per-dim feature lifetimes
    events = [[_persistence(d[dim]) for d in pooled] for dim in range(nd)]
    total = sum(len(e) for lst in events for e in lst)
    if total == 0:
        return 1.0

    Z, V = _stratified_stat(
        [np.concatenate(ev[:n0]) for ev in events],
        [np.concatenate(ev[n0:]) for ev in events])
    obs = _stat_value(Z, V)

    if asymptotic:
        p = 1.0 - stats.chi2.cdf(obs, df=1)
        if V == 0:
            return 0.0 if Z != 0 else 1.0
        return float(np.clip(p, 0.0, 1.0))

    perms = _perm_groups(N, n0, n_perm, rng)
    null = np.empty(n_perm)
    for k in range(n_perm):
        g = perms[k]
        Z, V = _stratified_stat(
            [np.concatenate([ev[i] for i in np.flatnonzero(g)]) for ev in events],
            [np.concatenate([ev[i] for i in np.flatnonzero(~g)]) for ev in events])
        null[k] = _stat_value(Z, V)
    return _perm_pvalue(obs, null, "greater")
