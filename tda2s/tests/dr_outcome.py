"""Calibrated outcome-level doubly robust tests for Phase 3.

The AIPW point estimate and cross-fitting are delegated to ``tcda_uq``.  This
module adds the testing layer that P1 owns:

* the max-over-scale and max-over-degree statistic;
* a shared-multiplier null over the cross-fitted score process;
* a fast covariate-preserving permutation null;
* the Phase 3.5 studentized empirical-null degree comparator adapted from
  Vejdemo-Johansson and Mukherjee (:func:`vjm_multiplicity_test`).

The permutation implementation is deliberately explicit about its scope.  A
full permutation test that refits the nuisances for every label draw is
prohibitively expensive.  ``fit_dr`` fits cross-fitted nuisances once,
reconstructs their out-of-fold predictions in original sample order, and
``stratified_permutation_test`` evaluates the AIPW score with those predictions
held fixed.  This is exact conditional randomisation inference when the fitted
nuisances and propensity strata are fixed independently of the permuted labels,
for example under a sharp conditional null with externally fixed design
information.  With estimated, label-dependent nuisances it is a fast
cross-fitted calibration approximation, not a finite-sample exact theorem.
The distinction is part of the Phase 3 report and is not hidden by the API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import numpy as np
from sklearn.ensemble import (HistGradientBoostingClassifier,
                               RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, StratifiedKFold

from tda2s.adapters.tcda_uq import aipw_curve
from tda2s.resample import multiplier_bootstrap, p_value


def _clip_propensity(pi: np.ndarray) -> np.ndarray:
    """Match tcda_uq's propensity clipping convention."""
    out = np.asarray(pi, dtype=float).copy()
    out[out <= 0.0] = 1e-2
    out[out >= 1.0] = 1.0 - 1e-2
    return out


def _split_test_indices(A: np.ndarray, X: np.ndarray, n_folds: int,
                        stratify: bool, random_state: Optional[int]):
    """Reproduce the test-fold order used by tcda_uq.cross_fit."""
    if stratify:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                   random_state=random_state)
        iterator = splitter.split(X, A)
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True,
                         random_state=random_state)
        iterator = splitter.split(X)
    return list(iterator)


def _as_score_tensor(values: Iterable[np.ndarray]) -> np.ndarray:
    """Stack tcda_uq's per-dimension curves as ``[n, d, t]``."""
    return np.stack([np.asarray(v, dtype=float) for v in values], axis=1)


@dataclass
class DRFit:
    """Cached cross-fitted nuisances and data for Phase 3 calibration."""

    phi: np.ndarray                 # original order, [n, d, t]
    A: np.ndarray                   # original order, [n]
    X: np.ndarray                   # original order, [n, p]
    tseq: np.ndarray
    result: Any                     # tcda_uq CrossFitResult
    order: np.ndarray               # score/fold order -> original row index
    labels_order: np.ndarray        # labels in score/fold order
    phi_order: np.ndarray           # outcomes in score/fold order
    mu0: np.ndarray                 # fixed out-of-fold predictions [n, d, t]
    mu1: np.ndarray                 # fixed out-of-fold predictions [n, d, t]
    pi_hat: np.ndarray              # fixed out-of-fold propensities [n]
    fold_ids: np.ndarray            # fold id in score/fold order
    n_basis: int
    n_folds: int
    stratify: bool
    random_state: Optional[int]
    propensity_feature_fn: Optional[Callable]

    @property
    def n(self) -> int:
        return int(self.phi.shape[0])

    @property
    def n_hom_dim(self) -> int:
        return int(self.phi.shape[1])

    @property
    def resolution(self) -> int:
        return int(self.phi.shape[2])

    @property
    def estimate(self) -> np.ndarray:
        """AIPW curve in dimension-major shape ``[d, t]``."""
        return _as_score_tensor(self.result.scores).mean(axis=0)

    @property
    def centered_scores(self) -> np.ndarray:
        """Centered cross-fitted score process in score/fold order."""
        return _as_score_tensor(self.result.scores) - self.estimate[None, :, :]


def fit_dr(sample, tseq, *, n_basis: int = 8, n_folds: int = 2,
           propensity_estimator=None, stratify: bool = True,
           random_state: Optional[int] = 0,
           propensity_feature_fn: Optional[Callable] = None) -> DRFit:
    """Fit tcda_uq's cross-fitted AIPW estimator and cache fold predictions.

    No estimator is reimplemented here.  The fold reconstruction is only an
    adapter around the public ``CrossFitResult.folds`` objects, allowing the
    permutation layer to evaluate many label assignments without refitting.
    """
    phi, A, X = (np.asarray(sample[0]), np.asarray(sample[1]),
                 np.asarray(sample[2]))
    if phi.ndim != 3 or A.ndim != 1 or X.ndim != 2:
        raise ValueError("sample must have phi[n,d,t], A[n], and X[n,p]")
    if len(A) != len(phi) or len(X) != len(phi):
        raise ValueError("phi, A, and X must have the same number of rows")
    A = A.astype(int, copy=False)
    if not np.all(np.isin(A, (0, 1))):
        raise ValueError("A must contain only 0/1 labels")

    wrapped = aipw_curve(
        (phi, A, X), np.asarray(tseq), n_basis=n_basis, n_folds=n_folds,
        propensity_estimator=propensity_estimator, stratify=stratify,
        random_state=random_state, propensity_feature_fn=propensity_feature_fn,
    )
    result = wrapped["raw_result"]
    folds = _split_test_indices(A, X, n_folds, stratify, random_state)
    order = np.concatenate([test_idx for _, test_idx in folds])
    if not np.array_equal(order, np.asarray(result.order)):
        raise RuntimeError("could not reproduce tcda_uq cross-fit fold order")
    if len(folds) != len(result.folds):
        raise RuntimeError("tcda_uq returned an unexpected number of folds")

    n, n_dim, res = phi.shape
    mu0 = np.empty((n, n_dim, res), dtype=float)
    mu1 = np.empty_like(mu0)
    pi_parts = []
    fold_id_parts = []
    for fold_id, ((_, test_idx), fitted) in enumerate(zip(folds, result.folds)):
        mu_hats = fitted.predict_mu(X[test_idx])
        if len(mu_hats) != n_dim:
            raise RuntimeError("tcda_uq returned an unexpected dimension count")
        for d, (m0, m1) in enumerate(mu_hats):
            mu0[test_idx, d, :] = np.asarray(m0, dtype=float)
            mu1[test_idx, d, :] = np.asarray(m1, dtype=float)
        features = (propensity_feature_fn(X[test_idx])
                    if propensity_feature_fn is not None else X[test_idx])
        pi_parts.append(np.asarray(
            fitted.prop_model.predict_proba(features)[:, 1], dtype=float))
        fold_id_parts.append(np.full(len(test_idx), fold_id, dtype=int))

    # ``result.scores`` is concatenated in fold order, whereas the nuisance
    # arrays above are assigned in original order.  Put all cached quantities
    # into one unambiguous order for fast matrix calculations.
    pi_order = _clip_propensity(np.concatenate(pi_parts))
    fold_ids = np.concatenate(fold_id_parts)
    return DRFit(
        phi=phi, A=A, X=X, tseq=np.asarray(tseq), result=result,
        order=order, labels_order=A[order], phi_order=phi[order],
        mu0=mu0[order], mu1=mu1[order], pi_hat=pi_order,
        fold_ids=fold_ids, n_basis=n_basis, n_folds=n_folds,
        stratify=stratify, random_state=random_state,
        propensity_feature_fn=propensity_feature_fn,
    )


def _scores_for_labels(fit: DRFit, labels_order: np.ndarray) -> np.ndarray:
    """Evaluate the AIPW score using cached nuisances and new labels."""
    labels = np.asarray(labels_order, dtype=float)
    if labels.shape != (fit.n,):
        raise ValueError("labels must have one entry per fitted unit")
    if not np.all(np.isin(labels, (0.0, 1.0))):
        raise ValueError("labels must contain only 0/1 values")
    inv_treat = (labels / fit.pi_hat)[:, None, None]
    inv_control = ((1.0 - labels) / (1.0 - fit.pi_hat))[:, None, None]
    return (fit.mu1 - fit.mu0 + inv_treat * (fit.phi_order - fit.mu1)
            - inv_control * (fit.phi_order - fit.mu0))


def _select_curves(curves: np.ndarray, degrees=None) -> np.ndarray:
    curves = np.asarray(curves, dtype=float)
    if curves.ndim != 2:
        raise ValueError("curves must have shape [dimension, resolution]")
    if degrees is None:
        return curves
    idx = np.asarray(list(degrees), dtype=int)
    if idx.ndim != 1 or len(idx) == 0 or np.any(idx < 0) or np.any(idx >= len(curves)):
        raise ValueError("degrees must be valid non-empty dimension indices")
    return curves[idx]


def _per_degree_norm(curves: np.ndarray, norm: str) -> np.ndarray:
    """Return one scale norm per degree, keeping the degree axis."""
    curves = np.asarray(curves, dtype=float)
    if norm == "sup":
        return np.max(np.abs(curves), axis=-1)
    if norm == "l2":
        # Root-mean-square on the common grid.  The common 1/resolution factor
        # makes this comparable across the Phase 3 grids.
        return np.sqrt(np.mean(curves ** 2, axis=-1))
    raise ValueError("norm must be 'sup' or 'l2'")


def _curve_norm(curves: np.ndarray, norm: str) -> np.ndarray:
    """Return one max-over-degree norm for each leading draw."""
    return np.max(_per_degree_norm(curves, norm), axis=-1)


def dr_statistic(curves: np.ndarray, n: int, *, degrees=None,
                 studentize: bool = False, score_values: Optional[np.ndarray] = None,
                 sd_floor: float = 1e-10, norm: str = "sup") -> float:
    """Compute ``sqrt(n) max_d sup_t |psi_hat_d(t)|``.

    If ``studentize=True``, ``score_values`` must contain the per-unit score
    process in the same order as ``curves`` and the statistic uses its
    pointwise empirical standard deviation.  The raw statistic is the primary
    Phase 3 specification; studentization is a diagnostic/robustness option.
    """
    selected = _select_curves(curves, degrees)
    if studentize:
        if score_values is None:
            raise ValueError("score_values required for studentization")
        scores = np.asarray(score_values, dtype=float)
        sd = scores.std(axis=0, ddof=1)
        sd = np.maximum(_select_curves(sd, degrees), sd_floor)
        selected = selected / sd
    return float(np.sqrt(n) * _curve_norm(selected[None, :, :], norm)[0])


def _bootstrap_stats(centered: np.ndarray, n_draws: int, rng: np.random.Generator,
                     *, degrees=None, studentize: bool = False,
                     sd_floor: float = 1e-10, multiplier: str = "gaussian",
                     norm: str = "sup", reduce: str = "max") -> np.ndarray:
    """Shared-multiplier statistics with one multiplier per unit.

    ``reduce="max"`` returns one max-over-degree statistic per draw, which is
    the Phase 3 primary specification.  ``reduce="per_degree"`` keeps the
    degree axis, returning ``[n_draws, n_selected_degrees]``; the multiplier
    stream is identical either way, so the two reductions describe the same
    null draws.  Phase 3.5 needs the per-degree form because the
    Vejdemo-Johansson--Mukherjee comparator studentizes each degree against
    its own null before taking the joint maximum.
    """
    centered = np.asarray(centered, dtype=float)
    n, n_dim, res = centered.shape
    if reduce not in ("max", "per_degree"):
        raise ValueError("reduce must be 'max' or 'per_degree'")
    if reduce == "max" and not studentize and norm == "sup":
        flat = centered.reshape(n, n_dim * res)
        return multiplier_bootstrap(flat, n_draws, rng, kind=multiplier)

    sd = centered.std(axis=0, ddof=1) if studentize else None
    if sd is not None:
        sd = np.maximum(_select_curves(sd, degrees), sd_floor)
    d_idx = list(range(n_dim)) if degrees is None else list(np.asarray(list(degrees), dtype=int))
    out = (np.empty((n_draws, len(d_idx)), dtype=float) if reduce == "per_degree"
           else np.empty(n_draws, dtype=float))
    for b in range(n_draws):
        if multiplier == "gaussian":
            xi = rng.standard_normal(n)
        elif multiplier == "rademacher":
            xi = rng.choice(np.array([-1.0, 1.0]), size=n)
        else:
            raise ValueError("multiplier must be 'gaussian' or 'rademacher'")
        draw = (xi[:, None, None] * centered).sum(axis=0) / np.sqrt(n)
        if studentize:
            draw = draw[d_idx] / sd
        else:
            draw = draw[d_idx]
        if reduce == "per_degree":
            out[b] = _per_degree_norm(draw, norm)
        else:
            out[b] = _curve_norm(draw[None, :, :], norm)[0]
    return out


def multiplier_test(fit: DRFit, *, n_draws: int = 2000, alpha: float = 0.05,
                    degrees=None, studentize: bool = False, multiplier: str = "gaussian",
                    seed: Optional[int] = 0, norm: str = "sup") -> dict:
    """Calibrate the outcome-level test by a shared multiplier process."""
    observed_scores = _scores_for_labels(fit, fit.labels_order)
    observed = observed_scores.mean(axis=0)
    centered = observed_scores - observed[None, :, :]
    stat = dr_statistic(observed, fit.n, degrees=degrees, studentize=studentize,
                        score_values=observed_scores, norm=norm)
    null = _bootstrap_stats(centered, n_draws, np.random.default_rng(seed),
                            degrees=degrees, studentize=studentize,
                            multiplier=multiplier, norm=norm)
    return {
        "method": "multiplier",
        "statistic": stat,
        "pvalue": p_value(stat, null),
        "null": null,
        "critical_value": float(np.quantile(null, 1.0 - alpha)),
        "estimate": observed,
        "scores": observed_scores,
    }


def degree_multiplicity_test(fit: DRFit, *, n_draws: int = 2000,
                             alpha: float = 0.05, studentize: bool = False,
                             multiplier: str = "gaussian",
                             seed: Optional[int] = 0) -> dict:
    """Bonferroni and max-statistic calibration across homology degrees.

    The max-statistic uses one multiplier per unit shared across all degrees,
    preserving their dependence.  The Bonferroni line is included as the
    transparent conservative comparator.  Neither line is a
    Vejdemo-Johansson--Mukherjee procedure; that comparator is
    :func:`vjm_multiplicity_test`, which studentizes each degree against its
    own empirical null before taking the joint maximum, and whose mapping to
    P1 is audited in ``docs/phase3_5_vjm_mapping.md``.
    """
    per_degree = [
        multiplier_test(fit, n_draws=n_draws, alpha=alpha, degrees=[d],
                        studentize=studentize, multiplier=multiplier,
                        seed=seed)
        for d in range(fit.n_hom_dim)
    ]
    max_test = multiplier_test(
        fit, n_draws=n_draws, alpha=alpha, degrees=range(fit.n_hom_dim),
        studentize=studentize, multiplier=multiplier, seed=seed,
    )
    raw_p = min((entry["pvalue"] for entry in per_degree), default=1.0)
    return {
        "per_degree": per_degree,
        "bonferroni_pvalue": float(min(1.0, fit.n_hom_dim * raw_p)),
        "max_statistic": max_test,
        "max_statistic_pvalue": float(max_test["pvalue"]),
    }


def propensity_strata(pi_hat: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Create deterministic quantile propensity strata for diagnostics.

    Quantile bins preserve propensity approximately.  For an exact conditional
    randomisation claim, pass externally fixed strata to
    :func:`stratified_permutation_test` instead.
    """
    pi = np.asarray(pi_hat, dtype=float)
    if pi.ndim != 1 or len(pi) == 0 or n_bins < 1:
        raise ValueError("pi_hat must be non-empty 1-D and n_bins >= 1")
    edges = np.unique(np.quantile(pi, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) <= 1:
        return np.zeros(len(pi), dtype=int)
    return np.digitize(pi, edges[1:-1], right=True).astype(int)


def _draw_stratified_labels(labels: np.ndarray, strata: np.ndarray,
                            n_draws: int, rng: np.random.Generator) -> np.ndarray:
    labels = np.asarray(labels, dtype=float)
    strata = np.asarray(strata)
    if labels.shape != strata.shape:
        raise ValueError("labels and strata must have the same shape")
    draws = np.broadcast_to(labels, (n_draws, len(labels))).copy()
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        for b in range(n_draws):
            draws[b, idx] = labels[idx][rng.permutation(len(idx))]
    return draws


def _permutation_stats(fit: DRFit, label_draws: np.ndarray, *, degrees=None,
                       studentize: bool = False, batch_size: int = 64,
                       sd_floor: float = 1e-10, norm: str = "sup",
                       reduce: str = "max") -> np.ndarray:
    """Vectorized permutation statistics with a bounded temporary footprint.

    ``reduce`` behaves as in :func:`_bootstrap_stats`: ``"max"`` collapses the
    degree axis, ``"per_degree"`` keeps it for the Phase 3.5 comparator.  The
    label draws are identical either way, so both reductions read the same
    null.
    """
    draws = np.asarray(label_draws, dtype=float)
    if draws.ndim != 2 or draws.shape[1] != fit.n:
        raise ValueError("label_draws must have shape [n_draws, n]")
    n_draws = draws.shape[0]
    base = fit.mu1 - fit.mu0
    residual1 = fit.phi_order - fit.mu1
    residual0 = fit.phi_order - fit.mu0
    base_mean = base.mean(axis=0)
    inv_pi = 1.0 / fit.pi_hat
    inv_one_minus_pi = 1.0 / (1.0 - fit.pi_hat)
    if studentize:
        observed_scores = _scores_for_labels(fit, fit.labels_order)
        sd = observed_scores.std(axis=0, ddof=1)
        sd = np.maximum(_select_curves(sd, degrees), sd_floor)
    d_idx = list(range(fit.n_hom_dim)) if degrees is None else list(np.asarray(list(degrees), dtype=int))
    if reduce not in ("max", "per_degree"):
        raise ValueError("reduce must be 'max' or 'per_degree'")
    out = (np.empty((n_draws, len(d_idx)), dtype=float) if reduce == "per_degree"
           else np.empty(n_draws, dtype=float))
    for lo in range(0, n_draws, batch_size):
        hi = min(lo + batch_size, n_draws)
        a = draws[lo:hi]
        treated = np.einsum("bn,nhr->bhr", a * inv_pi, residual1)
        control = np.einsum("bn,nhr->bhr", (1.0 - a) * inv_one_minus_pi, residual0)
        curves = base_mean[None, :, :] + (treated - control) / fit.n
        if studentize:
            curves = curves[:, d_idx, :] / sd[None, :, :]
        else:
            curves = curves[:, d_idx, :]
        if reduce == "per_degree":
            out[lo:hi] = _per_degree_norm(curves, norm) * np.sqrt(fit.n)
        else:
            out[lo:hi] = _curve_norm(curves, norm) * np.sqrt(fit.n)
    return out


def stratified_permutation_test(fit: DRFit, strata: np.ndarray, *, n_perm: int = 999,
                                alpha: float = 0.05, degrees=None,
                                studentize: bool = False, seed: Optional[int] = 0,
                                batch_size: int = 64, norm: str = "sup") -> dict:
    """Fast covariate-preserving permutation calibration.

    ``strata`` is supplied in original sample order.  Labels are permuted only
    within each stratum, while the cached cross-fitted nuisances are held fixed.
    No persistent homology, regression, or cross-fitting is performed inside
    the permutation loop.
    """
    strata = np.asarray(strata)
    if strata.shape != (fit.n,):
        raise ValueError("strata must be in original sample order with length n")
    strata_order = strata[fit.order]
    observed_scores = _scores_for_labels(fit, fit.labels_order)
    observed = observed_scores.mean(axis=0)
    stat = dr_statistic(observed, fit.n, degrees=degrees, studentize=studentize,
                        score_values=observed_scores, norm=norm)
    rng = np.random.default_rng(seed)
    label_draws = _draw_stratified_labels(fit.labels_order, strata_order,
                                          n_perm, rng)
    null = _permutation_stats(fit, label_draws, degrees=degrees,
                              studentize=studentize, batch_size=batch_size,
                              norm=norm)
    return {
        "method": "stratified_permutation_frozen_nuisance",
        "statistic": stat,
        "pvalue": p_value(stat, null),
        "null": null,
        "critical_value": float(np.quantile(null, 1.0 - alpha)),
        "estimate": observed,
        "scores": observed_scores,
        "strata": strata,
        "n_strata": int(len(np.unique(strata))),
    }


def degree_null_statistics(fit: DRFit, *, mechanism: str = "permutation",
                           strata: Optional[np.ndarray] = None,
                           n_draws: int = 399, seed: Optional[int] = 0,
                           norm: str = "sup", multiplier: str = "gaussian",
                           batch_size: int = 64) -> dict:
    """Observed and null per-degree statistics from one shared null draw.

    Every degree is evaluated on the *same* multiplier draw or the *same*
    label permutation, so the returned ``null`` matrix carries the dependence
    between degrees induced by the shared units and the shared cross-fitting.
    This is the object every Phase 3/3.5 multiplicity procedure consumes:
    Bonferroni, the shared max-statistic, and the studentized empirical-null
    comparator all reduce this one matrix differently, which makes their
    comparison exact rather than Monte-Carlo noisy.

    Returns a dict with ``observed`` of shape ``[n_degrees]`` and ``null`` of
    shape ``[n_draws, n_degrees]``, both on the ``sqrt(n)`` scale.
    """
    observed_scores = _scores_for_labels(fit, fit.labels_order)
    estimate = observed_scores.mean(axis=0)
    observed = np.sqrt(fit.n) * _per_degree_norm(estimate, norm)
    rng = np.random.default_rng(seed)
    if mechanism == "multiplier":
        centered = observed_scores - estimate[None, :, :]
        null = _bootstrap_stats(centered, n_draws, rng, multiplier=multiplier,
                                norm=norm, reduce="per_degree")
    elif mechanism == "permutation":
        if strata is None:
            raise ValueError("the permutation mechanism requires strata")
        strata = np.asarray(strata)
        if strata.shape != (fit.n,):
            raise ValueError("strata must be in original sample order with length n")
        label_draws = _draw_stratified_labels(fit.labels_order, strata[fit.order],
                                              n_draws, rng)
        null = _permutation_stats(fit, label_draws, norm=norm,
                                  batch_size=batch_size, reduce="per_degree")
    else:
        raise ValueError("mechanism must be 'multiplier' or 'permutation'")
    return {
        "mechanism": mechanism,
        "norm": norm,
        "observed": observed,
        "null": null,
        "estimate": estimate,
        "scores": observed_scores,
    }


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (no p-value).

    Only the statistic is reported: the null draws are dependent across
    degrees, so a KS *p-value* would not be valid here.  The statistic is used
    as a descriptive comparability measure, the P1 analogue of the source's
    own empirical check that standardized invariants share a distribution.
    """
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _standardize_family(observed: np.ndarray, null: np.ndarray,
                        convention: str, sd_floor: float):
    """Location-scale standardization of a family of scalar statistics.

    ``convention="source"`` reproduces the source's step 4 literally: the
    centering and scaling constants come from the null replicates only.
    ``convention="pooled"`` computes them from the ``n_draws + 1`` values that
    include the observed statistic.  Only the pooled version is a symmetric
    function of the augmented sample, so only the pooled version leaves the
    rank test exact when the observed statistic and the null replicates are
    exchangeable.  See ``docs/phase3_5_vjm_mapping.md``.
    """
    if convention == "source":
        mu = null.mean(axis=0)
        sd = null.std(axis=0, ddof=1)
    elif convention == "pooled":
        pooled = np.vstack([observed[None, :], null])
        mu = pooled.mean(axis=0)
        sd = pooled.std(axis=0, ddof=1)
    else:
        raise ValueError("standardization must be 'pooled' or 'source'")
    sd = np.maximum(sd, sd_floor)
    return mu, sd, (observed - mu) / sd, (null - mu[None, :]) / sd[None, :]


def _fdr_threshold(z_obs: np.ndarray, z_null: np.ndarray, alpha: float) -> dict:
    """Source Method 3/6 false-discovery threshold on standardized statistics.

    The proportions are the source's ``%V`` and ``%R``; the only deviation is
    bookkeeping, since all ``n_draws`` null replicates are used in ``%V``
    rather than the source's ``N - 1`` (their first column holds the observed
    value).  With only three homology degrees the estimator is very coarse
    (``%R`` takes four possible values), which is why Phase 3.5 reports it as
    a diagnostic and forwards the procedure to the many-hypothesis families of
    Phase 5.2 rather than treating it as a degree-level control.
    """
    candidates = np.unique(z_obs)[::-1]
    chosen, chosen_q = None, None
    trace = []
    for c in candidates:
        pV = float(np.mean(z_null >= c))
        pR = float(np.mean(z_obs >= c))
        q = float(pV / pR) if pR > 0 else 0.0
        trace.append({"cutoff": float(c), "prop_null": pV, "prop_observed": pR,
                      "q_hat": q})
        if q <= alpha and (chosen is None or c < chosen):
            chosen, chosen_q = float(c), q
    rejected = ([int(d) for d in np.flatnonzero(z_obs >= chosen)]
                if chosen is not None else [])
    return {
        "alpha": float(alpha),
        "cutoff": chosen,
        "q_hat": chosen_q,
        "rejected_degrees": rejected,
        "attainable_q_hat": float(min(entry["q_hat"] for entry in trace)) if trace else 1.0,
        "trace": trace,
    }


def vjm_multiplicity_test(fit: DRFit, *, mechanism: str = "permutation",
                          strata: Optional[np.ndarray] = None,
                          n_draws: int = 399, alpha: float = 0.05,
                          seed: Optional[int] = 0, norm: str = "sup",
                          multiplier: str = "gaussian",
                          standardization: str = "pooled",
                          sd_floor: float = 1e-10,
                          batch_size: int = 64) -> dict:
    """Studentized empirical-null degree multiplicity comparator (Phase 3.5).

    This is the P1 transfer of the Vejdemo-Johansson--Mukherjee empirical-null
    multiple-testing construction (arXiv:1812.06491v4, Method 5 in section 3.5,
    which is Method 2 of section 3.3.1 applied across homological dimensions of
    one object).  Three things are deliberately *not* imported from the source
    and the change is recorded in ``docs/phase3_5_vjm_mapping.md``:

    1. the null replicates come from P1's frozen-nuisance stratified
       permutation or shared-multiplier mechanism, not from the source's
       uniform-on-a-convex-body point-process null, which has no meaning for a
       covariate-adjusted treatment-effect null;
    2. the replicates are drawn *jointly* across degrees, whereas the source
       simulates each family member independently;
    3. standardization defaults to the pooled convention, which keeps the rank
       test exact under exchangeability; ``standardization="source"`` restores
       the source's null-only constants for comparison.

    Bonferroni and the unstudentized shared max-statistic are returned from the
    same null matrix, so the three procedures are compared on identical draws.
    """
    family = degree_null_statistics(
        fit, mechanism=mechanism, strata=strata, n_draws=n_draws, seed=seed,
        norm=norm, multiplier=multiplier, batch_size=batch_size)
    observed, null = family["observed"], family["null"]
    n_degrees = len(observed)

    per_degree_p = np.array([p_value(observed[d], null[:, d])
                             for d in range(n_degrees)])
    shared_max_stat = float(np.max(observed))
    shared_max_p = p_value(shared_max_stat, np.max(null, axis=1))

    conventions = {}
    for name in ("pooled", "source"):
        mu, sd, z_obs, z_null = _standardize_family(observed, null, name, sd_floor)
        joint_null = np.max(z_null, axis=1)
        joint_obs = float(np.max(z_obs))
        conventions[name] = {
            "mu": mu, "sigma": sd,
            "standardized_observed": z_obs,
            "statistic": joint_obs,
            "pvalue": p_value(joint_obs, joint_null),
            "critical_value": float(np.quantile(joint_null, 1.0 - alpha)),
            "joint_null": joint_null,
            "fdr": _fdr_threshold(z_obs, z_null, alpha),
            "standardized_null_mean": z_null.mean(axis=0),
            "standardized_null_sd": z_null.std(axis=0, ddof=1),
            "standardized_null_q95": np.quantile(z_null, 0.95, axis=0),
        }

    _, _, _, z_null_pooled = _standardize_family(observed, null, "pooled", sd_floor)
    pairwise_ks = [
        _ks_statistic(z_null_pooled[:, i], z_null_pooled[:, j])
        for i in range(n_degrees) for j in range(i + 1, n_degrees)
    ]
    primary = conventions[standardization]
    return {
        "method": "vjm_studentized_empirical_null",
        "mechanism": mechanism,
        "standardization": standardization,
        "n_draws": int(n_draws),
        "observed": observed,
        "null": null,
        "estimate": family["estimate"],
        "statistic": primary["statistic"],
        "pvalue": primary["pvalue"],
        "critical_value": primary["critical_value"],
        "fdr": primary["fdr"],
        "conventions": conventions,
        "per_degree_pvalue": per_degree_p,
        "bonferroni_pvalue": float(min(1.0, n_degrees * per_degree_p.min())),
        "shared_max_statistic": shared_max_stat,
        "shared_max_pvalue": shared_max_p,
        "comparability": {
            "raw_null_mean": null.mean(axis=0),
            "raw_null_sd": null.std(axis=0, ddof=1),
            "max_pairwise_ks": float(max(pairwise_ks)) if pairwise_ks else 0.0,
        },
    }


def positivity_diagnostics(fit: DRFit) -> dict:
    """Overlap and effective-sample-size diagnostics for a fitted DR test."""
    pi = fit.pi_hat
    A = fit.labels_order.astype(bool)
    wt1 = A / pi
    wt0 = (~A) / (1.0 - pi)

    def ess(weights):
        weights = weights[weights > 0]
        return float(weights.sum() ** 2 / np.sum(weights ** 2)) if len(weights) else 0.0

    return {
        "min_pi": float(pi.min()),
        "max_pi": float(pi.max()),
        "q01_pi": float(np.quantile(pi, 0.01)),
        "q99_pi": float(np.quantile(pi, 0.99)),
        "fraction_pi_at_clip": float(np.mean((pi <= 0.0100001) | (pi >= 0.9899999))),
        "n_treated": int(A.sum()),
        "n_control": int((~A).sum()),
        "ess_treated": ess(wt1),
        "ess_control": ess(wt0),
    }


def equivalence_test(fit: DRFit, margin: float, *, alpha: float = 0.05,
                     n_draws: int = 2000, degrees=None, multiplier: str = "gaussian",
                     seed: Optional[int] = 0) -> dict:
    """Sup-norm TOST-style equivalence test for ``||psi||_inf < margin``.

    The test rejects the non-equivalence null when the simultaneous upper
    confidence bound for the selected max norm is at most ``margin``.  This is
    the functional analogue of the two one-sided tests and controls degree
    multiplicity through the selected max statistic.
    """
    observed_scores = _scores_for_labels(fit, fit.labels_order)
    estimate = observed_scores.mean(axis=0)
    centered = observed_scores - estimate[None, :, :]
    selected = _select_curves(estimate, degrees)
    observed_norm = float(np.max(np.abs(selected)))
    errors = _bootstrap_stats(centered, n_draws, np.random.default_rng(seed),
                              degrees=degrees, multiplier=multiplier)
    q = float(np.quantile(errors, 1.0 - alpha)) / np.sqrt(fit.n)
    threshold = np.sqrt(fit.n) * (margin - observed_norm)
    p_equiv = p_value(threshold, errors, alternative="greater")
    return {
        "method": "supnorm_equivalence",
        "margin": float(margin),
        "observed_norm": observed_norm,
        "upper_bound": observed_norm + q,
        "pvalue": p_equiv,
        "reject_non_equivalence": bool(observed_norm + q <= margin),
        "null_error": errors,
    }


def propensity_learner_grid(seed: int = 0) -> dict:
    """Small, reproducible propensity learner grid for task 3.3."""
    return {
        "logistic": LogisticRegression(max_iter=2000, random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, n_jobs=1, random_state=seed),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
            random_state=seed),
        "neural": MLPClassifier(hidden_layer_sizes=(32,), max_iter=400,
                                early_stopping=True, random_state=seed),
    }
