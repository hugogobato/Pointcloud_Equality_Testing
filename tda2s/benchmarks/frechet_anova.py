"""Frechet ANOVA test of Dubey & Muller (2019), as published.

Reference: P. Dubey and H.-G. Muller, "Frechet analysis of variance for random
objects", Biometrika 106(4):803-821 (2019), doi:10.1093/biomet/asz052.
Preprint arXiv:1710.02761v3, from which the equation numbers below are taken.
Paper accessed in full; no author code was released, so this is a direct
transcription of Section 4.

Method (paper Section 4, equations 6-13)
----------------------------------------
For groups ``G_1, ..., G_k`` of sizes ``n_j`` in a bounded metric space
``(Omega, d)``, with ``lambda_j = n_j / n``:

* group Frechet mean and variance (page 6)::

      mu_j   = argmin_w (1/n_j) sum_{i in G_j} d^2(w, Y_i)
      V_j    = (1/n_j) sum_{i in G_j} d^2(mu_j, Y_i)

* the variance estimate of eq. (3), applied per group::

      s2_j   = (1/n_j) sum_{i in G_j} d^4(mu_j, Y_i)
               - [ (1/n_j) sum_{i in G_j} d^2(mu_j, Y_i) ]^2

* pooled Frechet mean and variance, eq. (6)::

      mu_p   = argmin_w (1/n) sum_j sum_{i in G_j} d^2(w, Y_i)
      V_p    = (1/n) sum_j sum_{i in G_j} d^2(mu_p, Y_i)

* the two auxiliary statistics, eqs. (7) and (8)::

      F_n    = V_p - sum_j lambda_j V_j
      U_n    = sum_{j < l} (lambda_j lambda_l) / (s2_j s2_l) * (V_j - V_l)^2

* the test statistic, eq. (11)::

      T_n    = n U_n / sum_j (lambda_j / s2_j)
               + n F_n^2 / sum_j (lambda_j^2 s2_j)

Under the null of equal Frechet means *and* variances, ``T_n -> chi^2_(k-1)``
(Theorem 2, eq. 12) and the level-alpha rejection region is
``T_n > chi^2_{k-1, alpha}`` (eq. 13). ``F_n`` targets mean differences and
``U_n`` variance differences, so the test has power against both -- the paper
notes ``U_n`` is a Levene-type term, which is why a pure between/within ratio
would be blind to the scale alternatives of its Figure 1 (right panel).

The paper adds (end of Section 4) that "asymptotic tests may not work very well
... where the group sample sizes are small", and that permutation tests using
``T_n`` give more accurate level-alpha tests; ``n_perm`` selects that route and
is the default here, because two-sample topology problems are small-n.

Metric space
------------
Dubey-Muller is stated for any bounded metric space, so the caller chooses one:

* ``space="summary"`` (default) embeds each diagram in ``L^2`` via a functional
  summary (Betti curves by default, concatenated over homology dimensions).
  There the Frechet mean is the arithmetic mean -- exact, unique, closed-form.
  This is also the space P1's own scope-limit section argues for, since
  ``(D_p, W_p)`` has non-unique Frechet means (Turner et al.; Che et al.).
* ``space="diagram"`` uses ``W_2`` between diagrams and needs a true Wasserstein
  barycentre. That requires the ``POT`` package via ``gudhi.wasserstein``; if it
  is missing the call raises rather than silently substituting a proxy.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ._common import _perm_groups, _perm_pvalue, _points, _validate

__all__ = ["test_frechet_anova", "frechet_anova_statistic"]


def frechet_anova_statistic(sq_dists_to_group_mean, sq_dists_to_pooled_mean, labels):
    """Dubey-Muller ``T_n`` from precomputed squared distances (eqs. 6-11).

    Args:
        sq_dists_to_group_mean: ``(n,)`` array; entry ``i`` is ``d^2(mu_{g(i)},
            Y_i)``, the squared distance from unit ``i`` to *its own* group's
            Frechet mean.
        sq_dists_to_pooled_mean: ``(n,)`` array of ``d^2(mu_p, Y_i)``.
        labels: ``(n,)`` integer group labels.

    Returns:
        float ``T_n``. ``inf`` if a group's variance estimate ``s2_j`` is zero
        (degenerate group: reject).
    """
    a = np.asarray(sq_dists_to_group_mean, dtype=float)
    b = np.asarray(sq_dists_to_pooled_mean, dtype=float)
    labels = np.asarray(labels)
    groups = np.unique(labels)
    n = labels.size

    lam, V, s2 = [], [], []
    for g in groups:
        m = labels == g
        d2 = a[m]
        lam.append(m.sum() / n)
        V.append(d2.mean())
        s2.append((d2 ** 2).mean() - d2.mean() ** 2)     # eq. (3) per group
    lam = np.asarray(lam)
    V = np.asarray(V)
    s2 = np.asarray(s2)

    if np.any(s2 <= 0) or not np.all(np.isfinite(s2)):
        return np.inf

    V_p = b.mean()                                        # eq. (6)
    F_n = V_p - float(lam @ V)                            # eq. (7)

    U_n = 0.0                                             # eq. (8)
    for j in range(len(groups)):
        for l in range(j + 1, len(groups)):
            U_n += (lam[j] * lam[l]) / (s2[j] * s2[l]) * (V[j] - V[l]) ** 2

    term_u = n * U_n / float(np.sum(lam / s2))            # eq. (11), first term
    term_f = n * F_n ** 2 / float(np.sum(lam ** 2 * s2))  # eq. (11), second term
    return float(term_u + term_f)


def _l2_summary_stat(X, labels):
    """``T_n`` in ``L^2``: the Frechet mean is the arithmetic mean."""
    X = np.asarray(X, dtype=float)
    a = np.empty(len(X))
    for g in np.unique(labels):
        m = labels == g
        a[m] = ((X[m] - X[m].mean(axis=0)) ** 2).sum(axis=1)
    b = ((X - X.mean(axis=0)) ** 2).sum(axis=1)
    return frechet_anova_statistic(a, b, labels)


def _summary_vectors(diags0, diags1, representation, n_points, interval):
    """Embed every diagram list in ``R^p`` via a functional summary."""
    from tda2s.vec import vectorise

    d0, d1, _ = _validate(diags0, diags1)
    pooled = d0 + d1
    if interval is None:
        hi = 0.0
        for d in pooled:
            for dgm in d:
                p = _points(dgm)
                if len(p):
                    hi = max(hi, float(p[:, 1].max()))
        interval = (0.0, hi if hi > 0 else 1.0)
    rows = []
    for d in pooled:
        clean = [_points(dgm) for dgm in d]
        v = vectorise(clean, representation, interval=interval, n_points=n_points)
        rows.append(np.ravel(np.asarray(v, dtype=float)))
    return np.stack(rows), len(d0), len(d1)


def _diagram_barycentre_stat(diags0, diags1, labels, order):
    """``T_n`` in ``(D_2, W_2)`` using a true Wasserstein barycentre."""
    try:
        from gudhi.wasserstein import wasserstein_distance
        from gudhi.wasserstein.barycenter import lagrangian_barycenter
    except ImportError as exc:                            # pragma: no cover
        raise ImportError(
            "space='diagram' needs a Wasserstein barycentre, which requires the "
            "optional POT package (`pip install pot`). Dubey-Muller's Frechet "
            "mean has no closed form in (D_p, W_p); this wrapper will not "
            "substitute a proxy. Use space='summary' instead."
        ) from exc

    d0, d1, nd = _validate(diags0, diags1)
    pooled = d0 + d1

    def _mean_and_sq(idx):
        """Barycentre over the given units, and each unit's squared W_2 to it."""
        sq = np.zeros(len(pooled))
        for dim in range(nd):
            dgms = [_points(pooled[i][dim]) for i in idx]
            bary = lagrangian_barycenter(pdiagset=dgms, init=0)
            if bary is None:
                bary = np.empty((0, 2))
            for i in range(len(pooled)):
                w = wasserstein_distance(_points(pooled[i][dim]), bary,
                                         order=order, internal_p=2)
                sq[i] += float(w) ** 2
        return sq

    a = np.empty(len(pooled))
    for g in np.unique(labels):
        idx = np.flatnonzero(labels == g)
        a[idx] = _mean_and_sq(idx)[idx]
    b = _mean_and_sq(np.arange(len(pooled)))
    return frechet_anova_statistic(a, b, labels)


def test_frechet_anova(diags0, diags1, space="summary", representation="betti",
                       n_points=100, interval=None, n_perm=1000, seed=None,
                       order=2):
    """Dubey-Muller Frechet ANOVA two-sample test on persistence diagrams.

    Args:
        diags0, diags1: lists (over samples) of lists (per homology dim) of
            ``(k, 2)`` birth-death arrays.
        space: ``"summary"`` (L^2 functional summary, exact Frechet mean) or
            ``"diagram"`` (W_2 on diagrams, needs a barycentre via POT).
        representation: summary used when ``space="summary"``; any name accepted
            by ``tda2s.vec.vectorise`` (default ``"betti"``).
        n_points, interval: grid for the summary.
        n_perm: permutations for the label-permutation null. ``None`` uses the
            asymptotic ``chi^2_{k-1}`` of Theorem 2 instead -- the paper warns
            that route is unreliable at small group sizes.
        seed: RNG seed for the permutations.
        order: Wasserstein order when ``space="diagram"``.

    Returns:
        float p-value in [0, 1].
    """
    if space == "summary":
        X, n0, n1 = _summary_vectors(diags0, diags1, representation, n_points,
                                     interval)
        stat = lambda lab: _l2_summary_stat(X, lab)
    elif space == "diagram":
        d0, d1, _ = _validate(diags0, diags1)
        n0, n1 = len(d0), len(d1)
        stat = lambda lab: _diagram_barycentre_stat(diags0, diags1, lab, order)
    else:
        raise ValueError(f"space must be 'summary' or 'diagram', got {space!r}")

    N = n0 + n1
    labels = np.zeros(N, dtype=int)
    labels[:n0] = 1
    obs = stat(labels)

    if n_perm is None:
        if not np.isfinite(obs):
            return 0.0
        return float(stats.chi2.sf(obs, df=1))            # eq. (12), k = 2

    rng = np.random.default_rng(seed)
    perms = _perm_groups(N, n0, n_perm, rng)
    null = np.array([stat(perms[b].astype(int)) for b in range(n_perm)])
    return _perm_pvalue(obs, null, "greater")
