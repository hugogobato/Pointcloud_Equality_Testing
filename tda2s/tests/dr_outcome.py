"""Calibrated outcome-level doubly robust tests for Phase 3.

The AIPW point estimate and cross-fitting are delegated to ``tcda_uq``.  This
module adds the testing layer that P1 owns:

* the max-over-scale and max-over-degree statistic;
* a shared-multiplier null over the cross-fitted score process;
* a fast covariate-preserving permutation null.

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


def _curve_norm(curves: np.ndarray, norm: str) -> np.ndarray:
    """Return one max-over-degree norm for each leading draw."""
    curves = np.asarray(curves, dtype=float)
    if norm == "sup":
        return np.max(np.abs(curves), axis=(-2, -1))
    if norm == "l2":
        # Root-mean-square on the common grid.  The common 1/resolution factor
        # makes this comparable across the Phase 3 grids.
        return np.max(np.sqrt(np.mean(curves ** 2, axis=-1)), axis=-1)
    raise ValueError("norm must be 'sup' or 'l2'")


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
                     norm: str = "sup") -> np.ndarray:
    """Shared-multiplier max statistics with one multiplier per unit."""
    centered = np.asarray(centered, dtype=float)
    n, n_dim, res = centered.shape
    if not studentize and norm == "sup":
        flat = centered.reshape(n, n_dim * res)
        return multiplier_bootstrap(flat, n_draws, rng, kind=multiplier)

    sd = centered.std(axis=0, ddof=1) if studentize else None
    if sd is not None:
        sd = np.maximum(_select_curves(sd, degrees), sd_floor)
    d_idx = list(range(n_dim)) if degrees is None else list(np.asarray(list(degrees), dtype=int))
    out = np.empty(n_draws, dtype=float)
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
    transparent conservative comparator.  A Vejdemo-Johansson--Mukherjee
    implementation is not silently substituted here: its published
    persistence-specific multiple-testing construction remains a separate
    benchmark to add once its exact finite-sample convention is frozen.
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
                       sd_floor: float = 1e-10, norm: str = "sup") -> np.ndarray:
    """Vectorized permutation statistics with a bounded temporary footprint."""
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
    out = np.empty(n_draws, dtype=float)
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
