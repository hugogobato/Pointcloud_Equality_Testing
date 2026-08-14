"""Resampling engine: null distributions for the Phase 3-5 tests.

Schemes
-------
* ``permutation_test`` -- label permutation (exact null under exchangeability),
  optionally restricted to permutation *within propensity strata* (the
  covariate-preserving variant).
* ``multiplier_bootstrap`` -- Gaussian/Rademacher multiplier bootstrap over a
  per-unit influence-function matrix, calibrated to the weak limit of the
  empirical process: null draws are ``sqrt(n) * sup_t | n^{-1/2} sum_i g_i
  inf_i(t) |``.
* ``paired_bootstrap`` -- unit resampling with replacement preserving the
  treated/control split (paired designs, two-sample mean differences).
* ``smoothed_bootstrap`` -- Roycraft-Krebs-Polonik smoothed bootstrap for
  persistent Betti numbers: resample diagrams with replacement, jitter
  (birth, death) coordinates by N(0, sigma^2), recompute the statistic.
* ``cross_fit_folds`` -- k-fold cross-fitting index splits.
* ``p_value`` -- Monte Carlo p-value helper.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .smoothing import betti_curve

__all__ = [
    "permutation_test",
    "multiplier_bootstrap",
    "paired_bootstrap",
    "smoothed_bootstrap",
    "cross_fit_folds",
    "p_value",
]


def p_value(observed: float, null_stats: Sequence[float], alternative: str = "greater") -> float:
    """Monte Carlo p-value with Phipson-Smyth correction.

    ``alternative="greater"``: P(T* >= T_obs); ``"less"``: P(T* <= T_obs);
    ``"two-sided"``: P(|T*| >= |T_obs|) with the same correction.
    """
    null_stats = np.asarray(null_stats, dtype=float)
    if null_stats.size == 0:
        return 1.0
    if alternative == "greater":
        return (1.0 + (null_stats >= observed).sum()) / (1.0 + null_stats.size)
    if alternative == "less":
        return (1.0 + (null_stats <= observed).sum()) / (1.0 + null_stats.size)
    if alternative == "two-sided":
        return (1.0 + (np.abs(null_stats) >= abs(observed)).sum()) / (1.0 + null_stats.size)
    raise ValueError(f"unknown alternative: {alternative}")


def permutation_test(stat_fn: Callable, group_labels: np.ndarray, n_perm: int,
                     rng: np.random.Generator,
                     strata: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
    """Label-permutation test of ``stat_fn``.

    Args:
        stat_fn: callable taking ``(labels)`` and returning the statistic.
            The observed statistic is ``stat_fn(group_labels)``.
        group_labels: length-n binary treatment labels.
        n_perm: number of permutations.
        rng: numpy Generator.
        strata: optional length-n stratum ids; labels are permuted only within
            each stratum (covariate-preserving null).

    Returns:
        ``(observed_stat, null_stats)``.
    """
    labels = np.asarray(group_labels).copy()
    n = labels.size
    null_stats = np.empty(n_perm)
    observed = stat_fn(labels)

    if strata is None:
        for b in range(n_perm):
            null_stats[b] = stat_fn(labels[rng.permutation(n)])
    else:
        strata = np.asarray(strata)
        permuted = labels.copy()
        for s in np.unique(strata):
            idx = np.flatnonzero(strata == s)
            permuted[idx] = labels[idx][rng.permutation(idx.size)]
        for b in range(n_perm):
            for s in np.unique(strata):
                idx = np.flatnonzero(strata == s)
                permuted[idx] = labels[idx][rng.permutation(idx.size)]
            null_stats[b] = stat_fn(permuted)
    return float(observed), null_stats


def multiplier_bootstrap(influence_matrix: np.ndarray, n_draws: int,
                         rng: np.random.Generator,
                         kind: str = "gaussian") -> np.ndarray:
    """Multiplier bootstrap over a per-unit influence matrix.

    Args:
        influence_matrix: ``(n, resolution)`` per-unit influence-function
            values (row = unit). Convention matches tcda_uq's ``scores``.
        n_draws: number of null draws.
        rng: numpy Generator.
        kind: multiplier law: "gaussian" (N(0,1)) or "rademacher" (+-1).

    Returns:
        ``n_draws`` null statistics ``sup_t | n^{-1/2} sum_i g_i inf_i(t) |``.
        This is on the same scale as ``sqrt(n) * sup_t |mean_i inf_i(t)|``, the
        studentised statistic ``T_n`` of the plan, so observed values may be
        compared with these draws directly.
    """
    inf = np.asarray(influence_matrix, dtype=float)
    n = inf.shape[0]
    n_res = inf.shape[1]
    null_stats = np.empty(n_draws)
    scaled = inf / np.sqrt(n)
    for b in range(n_draws):
        if kind == "gaussian":
            g = rng.standard_normal(n)
        elif kind == "rademacher":
            g = rng.choice([-1.0, 1.0], size=n)
        else:
            raise ValueError(f"unknown multiplier kind: {kind}")
        curve = g @ scaled
        null_stats[b] = np.max(np.abs(curve))
    return null_stats


def paired_bootstrap(stat_fn: Callable, group_labels: np.ndarray, n_draws: int,
                     rng: np.random.Generator) -> Tuple[float, np.ndarray]:
    """Unit resampling with replacement preserving the treated/control split.

    Args:
        stat_fn: callable taking ``(labels)``.
        group_labels: length-n binary labels.
        n_draws: number of bootstrap draws.

    Returns:
        ``(observed_stat, bootstrap_stats)``.
    """
    labels = np.asarray(group_labels)
    n = labels.size
    n1 = int((labels == 1).sum())
    observed = stat_fn(labels)
    null_stats = np.empty(n_draws)
    for b in range(n_draws):
        idx = np.concatenate([
            rng.choice(np.flatnonzero(labels == 1), size=n1, replace=True),
            rng.choice(np.flatnonzero(labels == 0), size=n - n1, replace=True),
        ])
        null_stats[b] = stat_fn(labels[idx])
    return float(observed), null_stats


def smoothed_bootstrap(diagrams: Sequence[Sequence[np.ndarray]], n_draws: int,
                       rng: np.random.Generator, sigma: float,
                       stat_fn: Callable) -> Tuple[float, np.ndarray]:
    """Smoothed (jittered) bootstrap for persistent Betti numbers.

    Roycraft-Krebs-Polonik: the naive bootstrap is inconsistent for persistent
    Betti numbers; jittering (birth, death) coordinates by N(0, sigma^2) fixes
    the boundary effects.

    Args:
        diagrams: list over samples of list of (k, 2) per-dim diagrams.
        n_draws: number of bootstrap samples.
        rng: numpy Generator.
        sigma: jitter bandwidth (e.g. bandwidth / 2 of the kernel density).
        stat_fn: callable taking a list of diagram-lists (one per sample) and
            returning the statistic.

    Returns:
        ``(observed_stat, bootstrap_stats)``.
    """
    diagrams = [[np.asarray(d, dtype=float).reshape(-1, 2) for d in per_dim]
                for per_dim in diagrams]
    observed = stat_fn(diagrams)
    null_stats = np.empty(n_draws)
    n_samples = len(diagrams)
    for b in range(n_draws):
        # Resample WHOLE samples (diagram lists) with replacement -- one drawn
        # index per bootstrap unit. Pooling several units into a single diagram
        # would multiply every feature count by the pool size.
        idx = rng.integers(0, n_samples, size=n_samples)
        resampled = []
        for i in idx:
            per_dim = []
            for dgm in diagrams[i]:
                if dgm.size == 0:
                    per_dim.append(np.zeros((0, 2)))
                    continue
                jittered = dgm + rng.normal(0.0, sigma, size=dgm.shape)
                # keep the diagram above the diagonal after jittering
                jittered[:, 1] = np.maximum(jittered[:, 1], jittered[:, 0])
                per_dim.append(jittered)
            resampled.append(per_dim)
        null_stats[b] = stat_fn(resampled)
    return float(observed), null_stats


def cross_fit_folds(n: int, k_folds: int, rng: np.random.Generator,
                    stratify_labels: Optional[np.ndarray] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """k-fold cross-fitting index splits.

    Returns:
        List of ``(train_idx, test_idx)`` pairs covering all n indices exactly
        once as test indices. If ``stratify_labels`` is given, class balance is
        preserved within folds.
    """
    if stratify_labels is None:
        perm = rng.permutation(n)
        fold_of = np.zeros(n, dtype=int)
        for f in range(k_folds):
            fold_of[perm[f::k_folds]] = f
    else:
        labels = np.asarray(stratify_labels)
        fold_of = np.empty(n, dtype=int)
        for lab in np.unique(labels):
            idx = np.flatnonzero(labels == lab)
            perm = rng.permutation(idx.size)
            for f in range(k_folds):
                fold_of[idx[perm[f::k_folds]]] = f
    folds = []
    for f in range(k_folds):
        test_idx = np.flatnonzero(fold_of == f)
        train_idx = np.flatnonzero(fold_of != f)
        folds.append((train_idx, test_idx))
    return folds