"""Phase 4 — the distribution-level test (C2).

The first test of

.. math::

    H_0^{\\mathrm{dist}}:\\;\\delta_{\\mathrm{dist}}\\;
        = \\varrho\\bigl(T_{\\mathrm{dist}}(P^1_Y),\\;
                       T_{\\mathrm{dist}}(P^0_Y)\\bigr) = 0,

the covariate-adjusted, distribution-level topological two-sample null of
Souto & Diamantis (arXiv:2607.28161).  This is contribution C2 of the plan,
and is the most technically interesting piece of P1.

Two ``T_dist`` representations are implemented (see ``docs/phase4_tdist_decision.md``):

* ``method="measure"`` (headline, candidate (i)) — the *expected persistence
  measure* $T_{\\mathrm{dist}}(P) = \\mathbb E_P[\\mu_{D(F(Y))}]$ of Divol &
  Lacombe, contrasted by the $L^1$ distance on one fixed shared grid.  The
  measure is **affine** in $P$, so inverse-propensity-weighting is exact:
  $\\frac{1}{n}\\sum_i\\frac{\\mathbb 1\\{A_i=a\\}}{\\hat e_a(X_i)}
  \\mu_{D_i}$ is unbiased for $\\mathbb E[\\mu_D^a]$ when the propensity is
  correct, with no
  normalization ratio inside the expectation (the same ratio that breaks the
  *normalized* silhouette in WP1 Theorem 2 / W2').  This linearity is the
  reason candidate (i) is linear-friendly and is what the plan's Phase-4 risk
  note ("honest fallback = plug-in + stability transfer") cashes out.  The
  AIPW pseudo-outcome is the EIF form when the conditional mean nuisance is
  supplied; plain IPW is unbiased under a correct propensity but is not in
  general the efficient estimator.

* ``method="mmd"`` (secondary, candidate (ii)) — the MMD under the universal
  persistence scale-space kernel $k^U_\\sigma = \\exp(k_\\sigma)$ of Kwitt,
  Huber, Niethammer, Lin & Bauer (NIPS 2015, Prop. 2).  Universal implies
  characteristic, so $H_0^{\\mathrm{dist}}$ is then *strictly stronger* than
  $H_0^{\\mathrm{out}}$ and the "two independent tests" framing collapses into
  a nesting.  It is reported alongside the headline as the "did we miss
  anything the mean integrated away?" check.

Estimator (task 4.2)
-------------------

Let $\\mu_i = \\mu_{D(F(Y_i))}$ be the (vectorized) persistence measure of
unit $i$'s d-degree diagram, and $m_a(x) = \\mathbb E[\\mu \\mid A = a, X = x]$
the outcome regression in the measure's dual-pairing space.  Under A1
(consistency + weak conditional exchangeability + strict positivity) the
$g$-formula identifies $T_{\\mathrm{dist}}(P^a_Y)$ as $\\mathbb E[m_a(X)]$, and
the AIPW pseudo-outcome

.. math::

   \\rho^{\\mathrm{dist}}_{i,a} \\;=\\; \\hat m_a(X_i)
       + \\frac{\\mathbb 1\\{A_i=a\\}}{\\hat e_a(X_i)}
              \\bigl(\\mu_i - \\hat m_a(X_i)\\bigr)

satisfies $\\mathbb E[\\rho^{\\mathrm{dist}}_{i,a}] = T_{\\mathrm{dist}}(P^a_Y)$
whenever either $\\hat m_a$ or $\\hat e_a$ is correct (the standard
double-robustness identity, with $\\mu_i$ a Banach-space-valued outcome read
through the separable tent family — WP1 §1.3 codomain caveat).  The point
estimate is $\\hat T_a = \\frac 1 n \\sum_i \\rho^{\\mathrm{dist}}_{i,a}$ and
the (linear-functional) influence function is

.. math::

   \\mathrm{EIF}_a(Z) \\;=\\; \\frac{\\mathbb 1\\{A = a\\}}{e_a(X)}
       \\bigl(\\mu - m_a(X)\\bigr) \\;+\\; m_a(X) \\;-\\; \\mathbb E[m_a(X)].

This module implements the IPW plug-in and the AIPW estimator when fitted
outcome-regression nuisances are supplied through ``mu0_hat`` and
``mu1_hat``.  With no fitted means it degrades to the IPW plug-in, which is
the plan's licensed plug-in fallback.  The stability-transfer statement from
Souto & Diamantis §5.4 is conditional: if $T_{\\mathrm{dist}}$ is
$L$-Lipschitz in $W_1$ and a separate argument supplies
$\\|\\hat P^a-P^a\\|_{W_1}\\leq\\varepsilon$, then the transfer error is at
most $L\\varepsilon$.  This module does not estimate that unknown population
$W_1$ error or assert a universal $L=1$ bound for the weighted, binned
measure.

Null distribution (task 4.3)
---------------------------

The statistic is $\\hat\\delta_{\\mathrm{dist}} = \\|\\hat T_1 - \\hat T_0\\|_1$
(``"measure"``) or $\\widehat{\\mathrm{MMD}}_w^2$ under the universal kernel
(``"mmd"``).  The null is a **weighted permutation within propensity strata**:
labels are permuted only within each stratum (preserving the
covariate-driven propensity exactly), and the *frozen* IPW / AIPW weights
are held fixed across draws.  For ``method="measure"`` this is conditionally
exact only when the full score/statistic map is invariant under relabeling
under a sharp or exchangeable null.  Holding fitted AIPW nuisances fixed does
not by itself guarantee that invariance.  Equality of expected measure
vectors alone is a weak moment null and does not imply permutation validity.
Consequently the W2' rejection-rate check below is a calibration diagnostic,
not a finite-sample size theorem.  A production test for the weak mean null
needs a studentized multiplier or bootstrap calibration.  No Betti or Euler
statistic enters under the expected-measure representation, so the
Roycraft–Krebs–Polonik smoothed-bootstrap correction is not required for this
particular statistic; the smoothed bootstrap remains available through
``tda2s.resample.smoothed_bootstrap`` for statistics that do take a Betti
count.

Weak-null calibration (Phase 4.5)
---------------------------------

The production test of the weak null :math:`\\delta = 0` (equality of the
covariate-standardized expected-measure vectors on the fixed grid, with
possibly unequal unit-level laws) is the studentized max-:math:`t` multiplier
test of :func:`dist_multiplier_test`.  The per-unit AIPW contrast score
:func:`dist_scores` is the influence-function linearization of the contrast
(the difference of the two arm EIFs; the Phase 4.5 certificate), the
statistic is :math:`T_n = \\max_{j\\in J}\\sqrt n\\,|\\hat\\delta_j|/
\\hat\\sigma_j` with :math:`\\hat\\sigma_j^2 = n^{-1}\\sum_i(\\varphi_{ij} -
\\hat\\delta_j)^2`, and the null draws are the studentized multiplier sums
over the centered scores with one multiplier per unit.  Coordinates whose
sample variance is at or below ``variance_floor`` are excluded from ``J``
both for the observed statistic and the null, and a deterministic failure is
reported if no coordinate survives.  The calibration contract, the derivation
of the score, the audit of the permutation alternative, and the size fleet
are recorded in ``docs/phase45_weak_null_calibration.md``.  The frozen-label
permutation (:func:`stratified_permutation_test` and the studentized variant
:func:`studentized_permutation_test`) remains a sharp-null diagnostic only:
under the weak null with imbalance the permuted score mean is centered at
the observational contrast rather than at zero, which is the Phase-4.5 audit
finding (task 4.5.4).  The Gaussian diagnostic path of
:func:`dist_multiplier_test` (``gaussian_path=True``) simulates the CCK-style
max of the limiting Gaussian vector directly from the estimated score
correlation, as a model check on the multiplier draws.

Grid discipline (WP1 §6.5, plan hazard)
---------------------------------------

The persistence-measure grid must be **pinned once, globally**, before any
group contrast.  ``tda2s.vec.persistence_measure`` derives its bin edges from
the diagrams it is handed, so binning one diagram at a time without an
explicit ``interval`` rescales the grid per cloud: every degree-0 diagram has
birth 0, so a single-class diagram $\\{(0, c)\\}$ then lands in the same bin
whatever $c$ is, and the expected measure collapses to total weighted mass
(measured cost of the collapse: $L^1 = 0.02$ where the true distance is
$4.02$).  Every public entry point in this module takes an explicit
``interval`` and passes it to every ``persistence_measure`` call, so the
collapse cannot be reintroduced by a caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np

from tda2s.resample import p_value

# ---------------------------------------------------------------------------
# defaults shared with the separation experiment and the test suite

DEFAULT_INTERVAL: Tuple[float, float] = (0.0, 2.0)
DEFAULT_N_BINS: int = 32
DEFAULT_WEIGHT_POWER: float = 3.0          # match the silhouette exponent r=3
DEFAULT_KERNEL_SIGMA: float = 0.10         # PSS kernel bandwidth, plan §4.1

# Phase 4.5 frozen calibration contract (docs/phase45_weak_null_calibration.md).
# These constants are frozen before the weak-null fleet; changing any of them
# after the fleet ran would invalidate the pre-registered rejection band.
VARIANCE_FLOOR: float = 1e-8   # coordinates with sample variance <= this are
                               # dropped from the max (zero-variance handling)
DEFAULT_N_DRAWS: int = 1999    # multiplier draws; p-value resolution 1/2000


# ---------------------------------------------------------------------------
# diagram-measure vectorisation

def _clip_propensity(pi: np.ndarray, lo: float = 1e-2,
                     hi: float = 1.0 - 1e-2) -> np.ndarray:
    """Match tcda_uq's propensity clipping convention so weights stay finite."""
    out = np.asarray(pi, dtype=float).copy()
    out[out <= 0.0] = lo
    out[out >= 1.0] = hi
    return out


def measure_features(
    diagrams: Sequence[Sequence[np.ndarray]],
    *,
    interval: Tuple[float, float] = DEFAULT_INTERVAL,
    n_bins: int = DEFAULT_N_BINS,
    weight_power: float = DEFAULT_WEIGHT_POWER,
    homology_dim: int = 0,
) -> np.ndarray:
    """Vectorised expected-measure features for a list of diagrams.

    Each unit's d-th diagram is read as the Divol–Lacombe persistence measure
    on the (birth, mid) plane, binned on ONE fixed grid shared by every unit.
    The point weight is $w_p = (\\mathrm{death} - \\mathrm{birth})^r$ with
    ``r = weight_power``, matched to the silhouette exponent so the codomain
    of $T_{\\mathrm{dist}}$ and the codomain of the mean silhouette live in
    the same tent family (WP1 §1.3 codomain-match repair).

    Args:
        diagrams: list of per-unit diagram lists (each unit's diagram list is
            indexed by homology dim, the ``tda2s.ph.compute_diagrams``
            convention).  Only ``homology_dim`` is read; a degree-aware
            caller can stack several degrees by calling once per degree.
        interval: the ONE fixed grid shared by every unit; load-bearing, see
            the module docstring "Grid discipline".
        n_bins: bins per axis of the (birth, mid) plane.
        weight_power: power-weight exponent $r$.
        homology_dim: which homology dimension to vectorise (default 0, the
            W1 / W2′ witness degree).

    Returns:
        ``(n_units, n_bins * n_bins)`` float array of binned weighted masses.
    """
    from tda2s.vec import persistence_measure

    w = (lambda p: float(abs(p[1] - p[0]) ** weight_power))
    feats = []
    for diags in diagrams:
        dgm = diags[homology_dim] if len(diags) > homology_dim else np.zeros((0, 2))
        # Pass the EXPLICIT interval every time: per-diagram default binning
        # silently rescales the grid and collapses the statistic (WP1 §6.5).
        feat = persistence_measure(
            [np.asarray(dgm, dtype=float).reshape(-1, 2)],
            weight=w, interval=interval, n_bins=n_bins,
        )[0].ravel()
        feats.append(feat)
    return np.stack(feats) if feats else np.empty((0, n_bins * n_bins))


# ---------------------------------------------------------------------------
# kernel-MMD features (candidate (ii))

def _k_pss(f: np.ndarray, g: np.ndarray, sigma: float) -> float:
    """Reininghaus et al. persistence scale-space kernel k_sigma(F, G)."""
    f = np.asarray(f, float).reshape(-1, 2)
    g = np.asarray(g, float).reshape(-1, 2)
    gbar = g[:, ::-1]
    d2 = ((f[:, None, :] - g[None, :, :]) ** 2).sum(-1)
    d2bar = ((f[:, None, :] - gbar[None, :, :]) ** 2).sum(-1)
    return float((np.exp(-d2 / (8 * sigma)) - np.exp(-d2bar / (8 * sigma))).sum()
                / (8 * np.pi * sigma))


def _k_universal(f: np.ndarray, g: np.ndarray, sigma: float) -> float:
    """Kwitt et al. NIPS 2015 Prop. 2: k^U_sigma = exp(k_sigma), universal."""
    return float(np.exp(_k_pss(f, g, sigma)))


def mmd2_weighted(mu0: np.ndarray, w0: np.ndarray,
                  mu1: np.ndarray, w1: np.ndarray,
                  diagrams0: Sequence[np.ndarray],
                  diagrams1: Sequence[np.ndarray],
                  sigma: float) -> float:
    """Weighted MMD^2 between two arms under the universal PSS kernel.

    The mean embedding of arm $a$ enters with the IPW weights
    $w_i = \\mathbb 1\\{A_i = a\\}/\\hat e_a(X_i)$ (normalised to sum to one
    within the arm), so the contrast targets the interventional embedding
    rather than the observational one.  The kernel is evaluated on the
    *point-level* diagrams, so the Radon mean embeddings enter as weighted
    sums of diagram-level kernel blocks; this is the same construction the
    field's MMD competitor uses, but reweighted by $\\hat e(X)$.

    Args:
        mu0, mu1: not used by the kernel itself; kept for API symmetry with
            the L1 path (callers that already have the measure features can
            pass them; the kernel is recomputed from diagrams).
        w0, w1: per-unit IPW weights for arms 0 and 1 (will be normalised
            to sum to one internally).
        diagrams0, diagrams1: per-unit diagram lists for the two arms,
            each a ``(k, 2)`` birth-death array for the chosen homology dim.
        sigma: PSS kernel bandwidth.

    Returns:
        The (squared, biased) weighted MMD between the two arms.
    """
    w0 = np.asarray(w0, float)
    w1 = np.asarray(w1, float)
    if w0.sum() <= 0 or w1.sum() <= 0:
        return 0.0
    w0 = w0 / w0.sum()
    w1 = w1 / w1.sum()

    def _arm_matrix(diags, weights):
        n = len(diags)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                k = _k_universal(diags[i], diags[j], sigma)
                K[i, j] = k
                K[j, i] = k
        return weights @ K @ weights

    a = _arm_matrix(diagrams0, w0)
    b = _arm_matrix(diagrams1, w1)
    # cross term: weighted sum over i in arm 0, j in arm 1 (universal kernel).
    cross = 0.0
    for i, wi in enumerate(w0):
        for j, wj in enumerate(w1):
            cross += wi * wj * _k_universal(diagrams0[i], diagrams1[j], sigma)
    return float(max(a + b - 2.0 * cross, 0.0))


# ---------------------------------------------------------------------------
# fitted distribution-level test object

@dataclass
class DistFit:
    """Cached diagrams, IPW weights and (optional) out-of-fold nuisance means.

    Mirrors ``DRFit`` of ``tda2s.tests.dr_outcome`` but for the
    distribution-level contrast.  nuisances ``mu0_hat`` and ``mu1_hat`` are the
    cross-fitted outcome regressions (the persistence-measure feature vectors
    regressed on ``X``); when ``None``, the estimator degrades to the IPW
    plug-in (Plan: "honest fallback = plug-in + stability transfer").
    """

    diagrams: list                       # per-unit diagram lists (original order)
    A: np.ndarray                        # original order, [n]
    X: np.ndarray                        # original order, [n, p]
    pi_hat: np.ndarray                   # cross-fitted propensity [n]
    measure_features: np.ndarray         # (n, n_bins^2) measure features
    mu0_hat: Optional[np.ndarray]        # cross-fitted m_0(X) features
    mu1_hat: Optional[np.ndarray]        # cross-fitted m_1(X) features
    method: str                          # "measure" or "mmd"
    interval: Tuple[float, float]
    n_bins: int
    weight_power: float
    homology_dim: int
    kernel_sigma: float

    @property
    def n(self) -> int:
        return int(self.A.shape[0])

    @property
    def is_aipw(self) -> bool:
        return self.mu0_hat is not None and self.mu1_hat is not None

    def _weights_for(self, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """IPW weights ``1/e_a`` for the two arms under one label assignment.

        Weights are the frozen, clipping-stabilised inverse propensities, so
        they are identical across permutation draws — the covariate-preserving
        part of the null.
        """
        labels = np.asarray(labels, dtype=float)
        a1 = labels
        a0 = 1.0 - labels
        w1 = a1 / self.pi_hat
        w0 = a0 / (1.0 - self.pi_hat)
        return w0, w1


def fit_dist(
    diagrams: list,
    A: np.ndarray,
    X: np.ndarray,
    pi_hat: np.ndarray,
    *,
    method: str = "measure",
    mu0_hat: Optional[np.ndarray] = None,
    mu1_hat: Optional[np.ndarray] = None,
    interval: Tuple[float, float] = DEFAULT_INTERVAL,
    n_bins: int = DEFAULT_N_BINS,
    weight_power: float = DEFAULT_WEIGHT_POWER,
    homology_dim: int = 0,
    kernel_sigma: float = DEFAULT_KERNEL_SIGMA,
) -> DistFit:
    """Cache the fitted nuisance state of the distribution-level test.

    No cross-fitting is reimplemented here.  ``pi_hat`` and (optionally)
    ``mu0_hat`` / ``mu1_hat`` come from the caller's cross-fitted nuisance
    models — the same objects ``tda2s.tests.dr_outcome.fit_dr`` produces for
    the outcome-level test, but with the persistence-measure feature vector
    replacing the silhouette as the outcome.  A caller that does not supply
    outcome-regression nuisance curves gets the plain IPW plug-in, which is
    the plan's licensed fallback for candidate (i) (linear-friendly).
    """
    A = np.asarray(A, dtype=int)
    X = np.asarray(X, dtype=float)
    pi_hat = _clip_propensity(pi_hat)
    if A.shape != (len(diagrams),) or X.shape[0] != len(diagrams):
        raise ValueError("diagrams, A, X must have matching first dimension")
    if pi_hat.shape != (len(diagrams),):
        raise ValueError("pi_hat must have length n")
    if (mu0_hat is None) != (mu1_hat is None):
        raise ValueError("mu0_hat and mu1_hat must be supplied together")
    feats = None
    if method == "measure":
        feats = measure_features(
            diagrams, interval=interval, n_bins=n_bins,
            weight_power=weight_power, homology_dim=homology_dim,
        )
        expected_shape = feats.shape
        for name, nuisance in (("mu0_hat", mu0_hat), ("mu1_hat", mu1_hat)):
            if nuisance is not None:
                nuisance = np.asarray(nuisance, dtype=float)
                if nuisance.shape != expected_shape:
                    raise ValueError(
                        f"{name} must have shape {expected_shape}, got "
                        f"{nuisance.shape}")
                if not np.isfinite(nuisance).all():
                    raise ValueError(f"{name} must contain only finite values")
        if not np.isfinite(pi_hat).all():
            raise ValueError("pi_hat must contain only finite values")
    elif method != "mmd":
        raise ValueError("method must be 'measure' or 'mmd'")
    return DistFit(
        diagrams=diagrams, A=A, X=X, pi_hat=pi_hat,
        measure_features=feats,
        mu0_hat=None if mu0_hat is None else np.asarray(mu0_hat, float),
        mu1_hat=None if mu1_hat is None else np.asarray(mu1_hat, float),
        method=method, interval=interval, n_bins=n_bins,
        weight_power=weight_power, homology_dim=homology_dim,
        kernel_sigma=kernel_sigma,
    )


# ---------------------------------------------------------------------------
# statistic and its per-arm pseudo-outcome

def _arm_estimates(fit: DistFit, labels: np.ndarray):
    """Return (T0_hat, T1_hat) under one label assignment.

    For ``method="measure"`` these are the AIPW (or IPW plug-in if no
    nuisance means) estimates of the interventional expected persistence
    measure vectors.  For ``method="mmd"`` they are the diagrams and weights
    themselves, returned so :func:`mmd2_weighted` can take over.
    """
    labels = np.asarray(labels, dtype=float)
    w0, w1 = fit._weights_for(labels)
    if fit.method == "measure":
        mu = fit.measure_features
        if fit.is_aipw:
            # AIPW: rho_a = m_hat_a + (1{A=a}/e_a) (mu - m_hat_a)
            # E[rho_a] = T_a when either nuisance is correct.
            t0 = fit.mu0_hat + w0[:, None] * (mu - fit.mu0_hat)
            t1 = fit.mu1_hat + w1[:, None] * (mu - fit.mu1_hat)
        else:
            # IPW plug-in: T_a_hat = (1/n) sum_i 1{A_i=a}/e_a(X_i) mu_i.
            # The 1/n (not 1/n_a) matches the g-formula standardisation.
            t0 = w0[:, None] * mu
            t1 = w1[:, None] * mu
        return t0.mean(axis=0), t1.mean(axis=0)
    # mmd: return diagrams split by labels, plus weights, for the kernel pass
    i0 = np.flatnonzero(labels == 0.0)
    i1 = np.flatnonzero(labels == 1.0)
    d0 = [fit.diagrams[i][fit.homology_dim] for i in i0]
    d1 = [fit.diagrams[i][fit.homology_dim] for i in i1]
    # weight-normalised pseudo-outcome: (n_arm diagrams, weights)
    return (d0, w0[i0], fit.kernel_sigma), (d1, w1[i1], fit.kernel_sigma)


def dist_statistic(fit: DistFit, labels: Optional[np.ndarray] = None) -> float:
    """Distribution-level test statistic under one label assignment.

    The statistic is $\\|\\hat T_1 - \\hat T_0\\|_1$ (candidate (i)) or
    $\\widehat{\\mathrm{MMD}}_w^2$ under the universal kernel (candidate (ii));
    both target the covariate-adjusted interventional contrast.  The default
    ``labels=None`` reads the observed labels, so ``dist_statistic(fit)`` is
    the observed test statistic.
    """
    labels = fit.A.astype(float) if labels is None else np.asarray(labels, float)
    if fit.method == "measure":
        t0, t1 = _arm_estimates(fit, labels)
        return float(np.abs(t1 - t0).sum())
    # mmd: `t0` and `t1` carry (diagrams, weights, sigma)
    (d0, w0, sig), (d1, w1, _) = _arm_estimates(fit, labels)
    if len(d0) == 0 or len(d1) == 0:
        return 0.0
    return mmd2_weighted(None, w0, None, w1, d0, d1, sig)


# ---------------------------------------------------------------------------
# null: weighted stratified permutation with frozen nuisances

def _mmd_stats_from_kernel(fit: DistFit, label_draws: np.ndarray,
                           kernel_matrix: np.ndarray) -> np.ndarray:
    """Vectorised MMD null statistics from a cached universal-kernel matrix.

    The weighted MMD under a label vector ``a`` is a quadratic form in the
    per-arm IPW weights, so once ``K = [k^U_sigma(dgm_i, dgm_j)]`` is
    precomputed every permutation draw reduces to three einsums.  The weights
    are the frozen, clipping-stabilised inverse propensities of
    ``DistFit._weights_for`` (normalised to sum to one within the arm, the
    same convention as ``mmd2_weighted``); empty arms score 0, matching the
    direct path.
    """
    K = np.asarray(kernel_matrix, dtype=float)
    if K.shape != (fit.n, fit.n):
        raise ValueError("kernel_matrix must have shape [n, n]")
    a = np.asarray(label_draws, dtype=float)
    s1 = a * (1.0 / fit.pi_hat)[None, :]
    s0 = (1.0 - a) * (1.0 / (1.0 - fit.pi_hat))[None, :]
    n1 = s1.sum(axis=1, keepdims=True)
    n0 = s0.sum(axis=1, keepdims=True)
    w1 = s1 / np.where(n1 > 0, n1, 1.0)
    w0 = s0 / np.where(n0 > 0, n0, 1.0)
    out = (np.einsum("bi,ij,bj->b", w1, K, w1)
           + np.einsum("bi,ij,bj->b", w0, K, w0)
           - 2.0 * np.einsum("bi,ij,bj->b", w1, K, w0))
    out = np.maximum(out, 0.0)
    out[np.asarray(n1)[:, 0] <= 0.0] = 0.0
    out[np.asarray(n0)[:, 0] <= 0.0] = 0.0
    return out


def mmd_kernel_matrix(fit: DistFit) -> np.ndarray:
    """Full ``(n, n)`` matrix of the universal kernel between unit diagrams.

    Computed once per fit so the MMD permutation null becomes a sequence of
    quadratic-form evaluations on cached blocks, the same
    no-PH-inside-the-permutation-loop discipline as the L1 path.  The direct
    per-draw kernel evaluation in :func:`dist_statistic` is retained as the
    reference the vectorised path is tested against.
    """
    ds = [np.asarray(d[fit.homology_dim], dtype=float).reshape(-1, 2)
          for d in fit.diagrams]
    n = len(ds)
    K = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            k = _k_universal(ds[i], ds[j], fit.kernel_sigma)
            K[i, j] = k
            K[j, i] = k
    return K


def _draw_stratified_labels(labels: np.ndarray, strata: np.ndarray,
                            n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """Permute labels within each stratum; identical to ``dr_outcome``'s helper."""
    labels = np.asarray(labels, dtype=float)
    strata = np.asarray(strata)
    draws = np.broadcast_to(labels, (n_draws, len(labels))).copy()
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        for b in range(n_draws):
            draws[b, idx] = labels[idx][rng.permutation(len(idx))]
    return draws


def stratified_permutation_test(
    fit: DistFit, strata: np.ndarray, *, n_perm: int = 999,
    alpha: float = 0.05, seed: Optional[int] = 0, batch_size: int = 128,
    kernel_matrix: Optional[np.ndarray] = None,
) -> dict:
    """Weighted, covariate-preserving permutation diagnostic.

    Labels are permuted only within each propensity stratum, while the cached
    IPW / AIPW weights are held fixed.  No persistent homology or nuisance
    regression is recomputed inside the permutation loop.  This is not a
    weak-null calibration unless the complete frozen score map is invariant
    under relabeling.  The ``strata`` array must be in original sample order.
    For ``method="mmd"`` a cached
    universal-kernel matrix (see :func:`mmd_kernel_matrix`) turns the null
    into quadratic-form evaluations; the direct per-draw kernel pass is used
    when none is given.
    """
    strata = np.asarray(strata)
    if strata.shape != (fit.n,):
        raise ValueError("strata must be in original sample order with length n")
    observed = dist_statistic(fit, fit.A)
    rng = np.random.default_rng(seed)
    label_draws = _draw_stratified_labels(fit.A.astype(float), strata, n_perm, rng)
    null = _permutation_stats(fit, label_draws, batch_size=batch_size,
                              kernel_matrix=kernel_matrix)
    return {
        "method": "stratified_permutation_frozen_weights",
        "T_dist": fit.method,
        "statistic": observed,
        "pvalue": p_value(observed, null),
        "null": null,
        "critical_value": float(np.quantile(null, 1.0 - alpha)),
        "strata": strata,
        "n_strata": int(len(np.unique(strata))),
        "n_perm": int(n_perm),
    }


def _permutation_stats(fit: DistFit, label_draws: np.ndarray,
                       batch_size: int = 128,
                       kernel_matrix: Optional[np.ndarray] = None) -> np.ndarray:
    """Vectorised null statistics for the measure contrast.

    Computed entirely from cached features + frozen weights: the AIPW
    pseudo-outcome is linear in the labels once the nuisances are fixed, so a
    batch of label draws reduces to an einsum (the same structure as
    ``dr_outcome._permutation_stats``).  For ``method="mmd"`` the null is a
    sequence of quadratic forms on the precomputed universal-kernel matrix
    ``kernel_matrix`` (see :func:`mmd_kernel_matrix`); without it the direct
    per-draw kernel pass is used (kept for small ``n`` and as the reference
    the vectorised path is tested against).
    """
    draws = np.asarray(label_draws, dtype=float)
    if draws.ndim != 2 or draws.shape[1] != fit.n:
        raise ValueError("label_draws must have shape [n_draws, n]")
    n_draws = draws.shape[0]
    if fit.method == "mmd":
        if kernel_matrix is None:
            # The MMD pass re-evaluates the universal kernel under each draw,
            # so it is intentionally not vectorised; the secondary variant's
            # cost is acceptable because it is the diagnostic, not the
            # headline.  Callers with a cached kernel matrix skip this loop.
            return np.array([dist_statistic(fit, draws[b]) for b in range(n_draws)])
        return _mmd_stats_from_kernel(fit, draws, kernel_matrix)
    # measure: the L1 contrast of the frozen-nuisance score means
    curves = _permutation_curves(fit, draws, batch_size)
    return np.abs(curves).sum(axis=1)


# ---------------------------------------------------------------------------
# Phase 4.5: studentized weak-null calibration (tasks 4.5.2 / 4.5.3)

def dist_scores(fit: DistFit, labels: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-unit AIPW contrast scores ``phi`` of shape ``(n, q)``.

    The Phase 4.5 influence-score linearization of the distribution-level
    contrast on the fixed measure grid:

    .. math::

        \\varphi_i = \\hat m_1(X_i) - \\hat m_0(X_i)
            + \\frac{A_i}{\\hat e(X_i)}\\{V_i - \\hat m_1(X_i)\\}
            - \\frac{1 - A_i}{1 - \\hat e(X_i)}\\{V_i - \\hat m_0(X_i)\\},

    whose sample mean is exactly the AIPW contrast estimate
    ``_arm_estimates(fit, A)[1] - _arm_estimates(fit, A)[0]`` (verified by
    ``tests/test_phase45.py``).  With no fitted outcome nuisances it degrades
    to the IPW plug-in contrast score
    ``(A/e) V - ((1-A)/(1-e)) V``.  Only the ``method="measure"`` fit has a
    coordinatewise score; the MMD variant is not calibrated by this path.
    """
    if fit.method != "measure":
        raise ValueError("dist_scores is defined for method='measure' only")
    labels = fit.A if labels is None else np.asarray(labels, dtype=int)
    if labels.shape != (fit.n,):
        raise ValueError("labels must have one entry per unit")
    a1 = labels.astype(float)
    a0 = 1.0 - a1
    w1 = a1 / fit.pi_hat
    w0 = a0 / (1.0 - fit.pi_hat)
    mu = fit.measure_features
    if fit.is_aipw:
        return (fit.mu1_hat - fit.mu0_hat
                + w1[:, None] * (mu - fit.mu1_hat)
                - w0[:, None] * (mu - fit.mu0_hat))
    return w1[:, None] * mu - w0[:, None] * mu


def _score_variance(scores: np.ndarray) -> np.ndarray:
    """Coordinatewise score variance ``n^{-1} sum_i (phi_ij - delta_j)^2``.

    The ``1/n`` (not ``1/(n-1)``) convention is part of the frozen Phase 4.5
    contract; it is what makes the no-covariate balanced collapse reduce to
    the two-sample pooled-``t`` statistic up to the documented
    ``sqrt(n / (n - 2))`` factor (see the calibration doc).
    """
    centered = np.asarray(scores, dtype=float) - np.asarray(scores, dtype=float).mean(axis=0)
    return np.mean(centered ** 2, axis=0)


def _max_t_statistic(estimate: np.ndarray, sigma2: np.ndarray, n: int,
                     variance_floor: float) -> Tuple[float, np.ndarray]:
    """``max_{j in J} sqrt(n) |delta_j| / sigma_j`` with ``J`` the active set.

    Returns ``(statistic, active_mask)``; raises ``ValueError`` when no
    coordinate has variance above ``variance_floor`` (the deterministic
    failure report of task 4.5.3).
    """
    estimate = np.asarray(estimate, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    active = sigma2 > variance_floor
    if not active.any():
        raise ValueError(
            f"no coordinate has sample variance above variance_floor="
            f"{variance_floor}: every bin is constant in the sample")
    sigma = np.sqrt(sigma2[active])
    stat = float(np.max(np.sqrt(int(n)) * np.abs(estimate[active]) / sigma))
    return stat, active


def _multiplier_max_t_null(centered: np.ndarray, sigma: np.ndarray,
                           n_draws: int, rng: np.random.Generator,
                           multiplier: str, batch_size: int) -> np.ndarray:
    """Studentized max-``t`` multiplier null on the active coordinates.

    One multiplier per unit, shared across coordinates (preserving their
    dependence); draws are ``max_{j in J} |n^{-1/2} sum_i xi_i c_ij| / sigma_j``
    on the centered scores ``c``.  Batched so the multiplier matrix stays
    bounded (``batch_size x n`` floats per chunk).
    """
    centered = np.asarray(centered, dtype=float)
    n = centered.shape[0]
    sigma = np.asarray(sigma, dtype=float)
    null = np.empty(int(n_draws), dtype=float)
    for lo in range(0, int(n_draws), int(batch_size)):
        hi = min(lo + int(batch_size), int(n_draws))
        m = hi - lo
        if multiplier == "gaussian":
            xi = rng.standard_normal((m, n))
        elif multiplier == "rademacher":
            xi = rng.choice(np.array([-1.0, 1.0]), size=(m, n))
        else:
            raise ValueError("multiplier must be 'gaussian' or 'rademacher'")
        draws = (xi @ centered) / np.sqrt(n)
        null[lo:hi] = np.max(np.abs(draws) / sigma[None, :], axis=1)
    return null


def gaussian_max_t_null(centered: np.ndarray, n_draws: int,
                        rng: np.random.Generator,
                        jitter: float = 1e-12) -> np.ndarray:
    """CCK-style Gaussian diagnostic path: max of the limiting Gaussian vector.

    Draws ``max_{j in J} |Z_j|`` with ``Z ~ N(0, R_hat)`` and ``R_hat`` the
    estimated correlation matrix of the centered scores.  This is the
    Chernozhukov--Chetverikov--Kato (2013) Gaussian approximation reference;
    under the null it must agree with the multiplier null up to Monte Carlo
    error, so a systematic disagreement is a small-``n`` calibration flag,
    not a replacement for the multiplier test.
    """
    centered = np.asarray(centered, dtype=float)
    if centered.ndim != 2:
        raise ValueError("centered must be a (n, q) array of active scores")
    R = np.corrcoef(centered, rowvar=False)
    R = (R + R.T) / 2.0
    L = np.linalg.cholesky(R + jitter * np.eye(R.shape[0]))
    Z = rng.standard_normal((int(n_draws), R.shape[0])) @ L.T
    return np.max(np.abs(Z), axis=1)


def dist_multiplier_test(
    fit: DistFit, *,
    n_draws: int = DEFAULT_N_DRAWS, alpha: float = 0.05, seed: Optional[int] = 0,
    variance_floor: float = VARIANCE_FLOOR, multiplier: str = "gaussian",
    gaussian_path: bool = False, batch_size: int = 64,
) -> dict:
    """Production weak-null calibration: studentized max-``t`` multiplier test.

    The primary Phase 4.5 test of ``H_0^dist,grid: delta = 0``.  Scores are
    the cross-fitted AIPW contrast scores of :func:`dist_scores` (the caller
    supplies the frozen nuisances through ``fit_dist``); no persistent
    homology or nuisance regression is recomputed inside the calibration
    loop, only the multiplier draws.  Coordinates with sample variance at or
    below ``variance_floor`` are dropped from ``J`` (both statistic and null)
    and reported; a sample in which every coordinate is constant raises
    ``ValueError`` (the deterministic failure report).  ``gaussian_path=True``
    adds the CCK-style Gaussian reference of :func:`gaussian_max_t_null` on an
    independent RNG stream, so enabling it never perturbs the multiplier
    draws.
    """
    if fit.method != "measure":
        raise ValueError("dist_multiplier_test is defined for method='measure' only")
    scores = dist_scores(fit)
    estimate = scores.mean(axis=0)
    sigma2 = _score_variance(scores)
    stat, active = _max_t_statistic(estimate, sigma2, fit.n, variance_floor)
    sigma = np.sqrt(sigma2[active])
    centered_active = scores[:, active] - estimate[active][None, :]
    rng = np.random.default_rng(seed)
    null = _multiplier_max_t_null(centered_active, sigma, n_draws, rng,
                                  multiplier, batch_size)
    out = {
        "method": "studentized_multiplier_max_t",
        "statistic": stat,
        "pvalue": p_value(stat, null),
        "null": null,
        "critical_value": float(np.quantile(null, 1.0 - alpha)),
        "estimate": estimate,
        "estimate_l1": float(np.abs(estimate).sum()),
        "sigma2": sigma2,
        "active": active,
        "n_coordinates": int(active.size),
        "n_active_coordinates": int(active.sum()),
        "n_dropped_coordinates": int(active.size - active.sum()),
        "variance_floor": float(variance_floor),
        "multiplier": multiplier,
        "n_draws": int(n_draws),
        "gaussian_path": None,
    }
    if gaussian_path:
        g_rng = np.random.default_rng((int(seed) + 7919) % (2 ** 31 - 1)
                                      if seed is not None else None)
        g_null = gaussian_max_t_null(centered_active, n_draws, g_rng)
        out["gaussian_path"] = {
            "pvalue": p_value(stat, g_null),
            "null": g_null,
            "null_mean": float(g_null.mean()),
            "null_q95": float(np.quantile(g_null, 0.95)),
            "max_ecdf_gap": float(_ks_statistic(g_null, null)),
        }
    return out


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov--Smirnov statistic (descriptive; no p-value)."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


# ---------------------------------------------------------------------------
# Phase 4.5 task 4.5.4 subject: studentized permutation (sharp-null only)

def _permutation_curves(fit: DistFit, label_draws: np.ndarray,
                        batch_size: int = 128) -> np.ndarray:
    """Coordinatewise frozen-nuisance score means under label draws.

    The per-draw contrast curve is the frozen-nuisance analogue of
    ``dist_scores(fit, labels).mean(axis=0)`` for every draw, computed by the
    linear reduction (labels enter only through the IPW weight mask).  This
    is the shared engine of ``_permutation_stats`` and
    :func:`studentized_permutation_test`.
    """
    draws = np.asarray(label_draws, dtype=float)
    if draws.ndim != 2 or draws.shape[1] != fit.n:
        raise ValueError("label_draws must have shape [n_draws, n]")
    mu = fit.measure_features
    if fit.is_aipw:
        base = (fit.mu1_hat - fit.mu0_hat).mean(axis=0)
        resid1 = mu - fit.mu1_hat
        resid0 = mu - fit.mu0_hat
        inv1 = 1.0 / fit.pi_hat
        inv0 = 1.0 / (1.0 - fit.pi_hat)
    else:
        base = np.zeros_like(mu[0])
        resid1 = mu
        resid0 = mu
        inv1 = 1.0 / fit.pi_hat
        inv0 = 1.0 / (1.0 - fit.pi_hat)
    chunks = []
    n_draws = draws.shape[0]
    for lo in range(0, n_draws, int(batch_size)):
        hi = min(lo + int(batch_size), n_draws)
        a = draws[lo:hi]
        coeff = a[:, :, None] * inv1[None, :, None] * resid1[None, :, :] \
            - (1.0 - a[:, :, None]) * inv0[None, :, None] * resid0[None, :, :]
        chunks.append(base[None, :] + coeff.mean(axis=1))
    return np.concatenate(chunks, axis=0)


def studentized_permutation_test(
    fit: DistFit, strata: np.ndarray, *, n_perm: int = 999, alpha: float = 0.05,
    seed: Optional[int] = 0, variance_floor: float = VARIANCE_FLOOR,
    batch_size: int = 128,
) -> dict:
    """Frozen-denominator within-stratum permutation diagnostic.

    Labels are permuted within the supplied strata and the studentized max-
    ``t`` numerator is recomputed on each draw, while the observed variance
    ``sigma_hat^2`` and active coordinate set are frozen (the same ``J`` and
    floor as the multiplier test, so the two diagnostics are comparable).
    This is not the ordinary studentized permutation statistic with a
    draw-specific denominator.  **Scope: diagnostic only.**  Task 4.5.4
    audits why this cannot be a weak-null calibration: under ``delta = 0``
    with unequal conditional laws the permuted score mean is centered at the
    observational contrast, not at zero (see
    ``docs/phase45_weak_null_calibration.md`` §4.5.4).  It is reported
    alongside the multiplier test on the sharp design as an empirical
    agreement check of task 4.5.6.  Even under a sharp or exchangeable null,
    freezing fitted nuisances can break exact finite-sample permutation
    invariance.
    """
    if fit.method != "measure":
        raise ValueError("studentized_permutation_test is for method='measure' only")
    strata = np.asarray(strata)
    if strata.shape != (fit.n,):
        raise ValueError("strata must be in original sample order with length n")
    scores = dist_scores(fit)
    estimate = scores.mean(axis=0)
    sigma2 = _score_variance(scores)
    stat, active = _max_t_statistic(estimate, sigma2, fit.n, variance_floor)
    sigma = np.sqrt(sigma2[active])
    rng = np.random.default_rng(seed)
    label_draws = _draw_stratified_labels(fit.A.astype(float), strata, n_perm, rng)
    curves = _permutation_curves(fit, label_draws, batch_size)
    null = np.max(np.sqrt(fit.n) * np.abs(curves[:, active]) / sigma[None, :],
                  axis=1)
    return {
        "method": "studentized_permutation_max_t",
        "statistic": stat,
        "pvalue": p_value(stat, null),
        "null": null,
        "critical_value": float(np.quantile(null, 1.0 - alpha)),
        "strata": strata,
        "n_strata": int(len(np.unique(strata))),
        "n_perm": int(n_perm),
        "n_active_coordinates": int(active.sum()),
    }


# ---------------------------------------------------------------------------
# stability-transfer bound interface (task 4.2 fallback note)

def stability_transfer_bound(w1_error: float, *, lipschitz: float) -> float:
    """Apply a separately established §5.4 stability-transfer bound.

    If an external argument gives ``W1(P_hat, P) <= w1_error`` and the chosen
    representation is known to be ``lipschitz``-Lipschitz in that metric, the
    transfer bound is ``lipschitz * w1_error``.  The function deliberately
    does not infer ``w1_error`` from the observed diagrams: empirical total
    persistence is only a scale diagnostic, not a finite-sample upper bound
    on empirical Wasserstein error.
    """
    w1_error = float(w1_error)
    lipschitz = float(lipschitz)
    if not np.isfinite(w1_error) or w1_error < 0.0:
        raise ValueError("w1_error must be a finite non-negative number")
    if not np.isfinite(lipschitz) or lipschitz < 0.0:
        raise ValueError("lipschitz must be a finite non-negative number")
    return float(lipschitz * w1_error)
