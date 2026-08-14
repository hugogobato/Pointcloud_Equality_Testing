"""Covariate-driven point-cloud DGP harness (Phase 0.6).

The harness generates *datasets* of point clouds for two-sample topological
testing. It exposes independently dialable knobs:

  * covariates ``X ~ N(0, I)`` or a two-component Gaussian mixture;
  * propensity ``pi(X) = expit(prop_scale * X @ beta)`` (imbalance dialled via
    ``|beta|`` and ``prop_scale``);
  * covariate-driven topology: a deterministic function ``topology_knob(x) ->
    (n_loops, radius, noise)`` maps each covariate row to the generator
    parameters of its cloud;
  * an optional direct group effect that shifts the topology of group A
    regardless of ``X`` (power experiments).

Key design property: with ``group_effect=0`` the topology of a cloud is a
deterministic function of its covariate vector ``X_i``, so conditional on ``X``
the two groups have the *identical* topological law; only the propensity
differs between groups. This is the exact structure Phase 2's covariate-shift
gate experiments require. With ``group_effect > 0`` the group-A loop count is
shifted regardless of ``X``.

The harness records, per cloud, the true generator parameters (``n_loops``,
per-loop ``radii``, ``noise``, ``outlier_fraction``) in ``CloudSample.oracle``,
so tests and experiments can verify that dialled knobs are recovered from the
oracle.

``to_silhouette_sample`` converts a sample of clouds to the observed triplet
``(phi, A, X)`` in the ``tcda_uq`` silhouette convention (``phi`` of shape
``[n, n_hom_dim, resolution]``), so Phase 3 can feed both harnesses through the
same downstream estimators.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from .clouds import loops_cloud

_MU1 = np.array([1.0, 0.6, -0.7, 2.2, -1.0])
_MU2 = np.array([0.4, -0.4, -0.6, 3.3, 3.0])
_BETA = np.array([-0.5, -0.1, 0.6, 0.1, 0.1])


@dataclass
class CloudSample:
    """One realised dataset of point clouds with per-cloud topology oracles.

    Attributes:
        clouds: list of ``(m_i, 2)`` point clouds, one per unit.
        X: ``[n, d_x]`` covariate matrix.
        A: ``[n]`` group labels (0 = B, 1 = A).
        propensity: ``[n]`` true propensity ``pi(X)``.
        oracle: dict mapping unit index to ``{"n_loops", "radii", "noise",
            "outlier_fraction"}``, the exact generator parameters used.
    """

    clouds: list
    X: np.ndarray
    A: np.ndarray
    propensity: np.ndarray
    oracle: dict

    @property
    def true_n_loops(self) -> np.ndarray:
        """``[n]`` oracle loop counts."""
        return np.array([self.oracle[i]["n_loops"] for i in range(len(self.clouds))])

    @property
    def true_radii(self) -> list:
        """Per-cloud oracle loop radii (list of per-loop arrays)."""
        return [self.oracle[i]["radii"] for i in range(len(self.clouds))]

    @property
    def true_noise(self) -> np.ndarray:
        """``[n]`` oracle point jitter scales."""
        return np.array([self.oracle[i]["noise"] for i in range(len(self.clouds))])

    def observed(self, **sil_kwargs):
        """Observed triplet ``(phi, A, X)`` in tcda_uq silhouette format."""
        return to_silhouette_sample(self.clouds, self.X, self.A, **sil_kwargs)


def _default_topology_knob(gamma, k_max, radius, noise):
    """Default knob: ``n_loops = 1 + floor(expit(gamma * x0) * k_max)``."""

    def knob(x):
        n_loops = 1 + int(np.floor(expit(gamma * x[0]) * k_max))
        return n_loops, radius, noise

    return knob


class CloudSampleDGP:
    """Two-group point-cloud DGP with covariate-driven topology.

    Args:
        n_per_group: half the dataset size; ``n = 2 * n_per_group`` clouds are
            drawn in total. Labels are then drawn as ``A_i ~ Bern(pi(X_i))``, so
            the two groups are *not* forced to be equal-sized -- that imbalance
            is exactly the confounding this harness exists to create. Pass
            ``beta=np.zeros(d_x)`` for a balanced, unconfounded design.
        m: number of points per cloud.
        d_x: covariate dimension.
        covariate: ``"gaussian"`` (standard normal) or ``"mixture"``
            (two-component Gaussian mixture, tcda_uq conventions).
        beta: propensity coefficients (``None`` uses the tcda_uq default).
        prop_scale: multiplier of the propensity logit (``> 1`` pushes
            ``pi(X)`` toward ``{0, 1}``).
        topology_knob: callable ``x -> (n_loops, radius, noise)`` mapping one
            covariate row to generator parameters. ``None`` uses
            ``n_loops = 1 + floor(expit(gamma * x[0]) * k_max)`` with fixed
            ``radius`` and ``noise``. A deterministic knob is what makes the
            two groups conditionally identical given ``X``.
        gamma, k_max: slope and max extra loops of the default knob.
        radius, noise, outlier_fraction: fixed generator parameters used when
            ``topology_knob`` is ``None``.
        group_effect: integer loop-count shift applied to group A regardless
            of ``X`` (0 = conditionally identical groups).
        seed: recorded on the instance for provenance; the model coefficients
            (mixture means, default ``beta``) are fixed constants, so sampling
            randomness is controlled entirely by ``sample(rng=...)``.
    """

    def __init__(
        self,
        n_per_group: int = 25,
        m: int = 120,
        d_x: int = 3,
        covariate: str = "gaussian",
        beta=None,
        prop_scale: float = 1.0,
        topology_knob=None,
        gamma: float = 1.0,
        k_max: int = 3,
        radius: float = 1.0,
        noise: float = 0.05,
        outlier_fraction: float = 0.0,
        group_effect: int = 0,
        seed: int = 0,
    ):
        if n_per_group < 1:
            raise ValueError("n_per_group must be >= 1")
        if m < 40:
            raise ValueError("m must be >= 40 to keep loop persistence recoverable")
        if covariate not in ("gaussian", "mixture"):
            raise ValueError("covariate must be 'gaussian' or 'mixture'")

        self.n_per_group = int(n_per_group)
        self.m = int(m)
        self.d_x = int(d_x)
        self.covariate = covariate
        self.prop_scale = float(prop_scale)
        self.gamma = float(gamma)
        self.k_max = int(k_max)
        self.radius = float(radius)
        self.noise = float(noise)
        self.outlier_fraction = float(outlier_fraction)
        self.group_effect = int(group_effect)

        self.seed = seed
        self.mu1 = _MU1[: self.d_x]
        self.mu2 = _MU2[: self.d_x]
        self.Sigma = np.eye(self.d_x) * 0.5
        self.beta = np.asarray(beta if beta is not None else _BETA[: self.d_x], dtype=float)
        if self.beta.shape[0] != self.d_x:
            raise ValueError("beta must have length d_x")

        self.topology_knob = topology_knob if topology_knob is not None else _default_topology_knob(
            self.gamma, self.k_max, self.radius, self.noise
        )
        self._max_loops = max(1, self.m // 8)

    def propensity(self, X):
        """True propensity ``pi(X) = expit(prop_scale * X @ beta)``, ``[n]``."""
        return expit(self.prop_scale * (np.asarray(X, dtype=float) @ self.beta))

    def topology(self, x):
        """Topology tuple ``(n_loops, radius, noise)`` for one covariate row."""
        return self.topology_knob(np.asarray(x, dtype=float))

    def _sample_covariates(self, n, rng):
        if self.covariate == "gaussian":
            return rng.normal(size=(n, self.d_x))
        n1 = n // 2
        X1 = rng.multivariate_normal(self.mu1, self.Sigma, size=n1)
        X2 = rng.multivariate_normal(self.mu2, self.Sigma, size=n - n1)
        return np.vstack([X1, X2])

    def sample(self, n_per_group=None, X=None, rng=None) -> CloudSample:
        """Draw a dataset of ``2 * n_per_group`` point clouds.

        Args:
            n_per_group: overrides the constructor default.
            X: optional fixed covariate matrix ``[n, d_x]`` (e.g. repeated rows
                for conditional-identical-group checks); drawn otherwise.
            rng: seed or Generator.

        Returns:
            :class:`CloudSample` with ``clouds``, ``X``, ``A``, ``propensity``
            and the per-cloud ``oracle`` of true generator parameters.
        """
        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        n_per_group = self.n_per_group if n_per_group is None else int(n_per_group)
        n = 2 * n_per_group
        if X is None:
            X = self._sample_covariates(n, rng)
        else:
            X = np.asarray(X, dtype=float)
            if X.shape != (n, self.d_x):
                raise ValueError(f"X must have shape ({n}, {self.d_x})")

        pi = self.propensity(X)
        A = rng.binomial(1, pi).astype(int)

        clouds, oracle = [], {}
        for i in range(n):
            n_loops, radius, noise = self.topology(X[i])
            if A[i] == 1:
                n_loops = int(n_loops) + self.group_effect
            n_loops = int(np.clip(n_loops, 1, self._max_loops))
            radii = np.full(n_loops, float(radius))
            cloud = loops_cloud(self.m, n_loops, radius=radii, noise=noise,
                                outlier_fraction=self.outlier_fraction, rng=rng)
            clouds.append(cloud)
            oracle[i] = {
                "n_loops": n_loops,
                "radii": radii.copy(),
                "noise": float(noise),
                "outlier_fraction": self.outlier_fraction,
            }
        return CloudSample(clouds=clouds, X=X, A=A, propensity=pi, oracle=oracle)


def _silhouette_from_diagrams(diags, interval, r, resolution):
    """Power-weighted silhouette of a diagram list (tcda_uq convention).

    Delegates to ``tda2s.vec.silhouette``, which uses the same power-weight
    convention (``|death - birth| ** r``, ``keep_endpoints=True``) as
    ``tcda_uq.silhouette.compute_silhouette``.
    """
    from tda2s.vec import silhouette

    return silhouette(diags, interval=interval, r=r, resolution=resolution)


def to_silhouette_sample(clouds, X, A, filtration="alpha", homology_dims=(0, 1),
                         interval=(0.0, 1.0), r=3.0, resolution=100, **ph_kwargs):
    """Convert clouds to the observed ``(phi, A, X)`` silhouette triplet.

    Each cloud is mapped to persistence diagrams via
    ``tda2s.ph.compute_diagrams`` and then to a power-weighted silhouette, so
    the output matches ``tcda_uq``'s observed format: ``phi`` has shape
    ``[n, n_hom_dim, resolution]``, ``A`` is ``[n]`` and ``X`` is ``[n, d_x]``.

    Args:
        clouds: iterable of ``(m_i, 2)`` point clouds.
        X: ``[n, d_x]`` covariate matrix.
        A: ``[n]`` group labels.
        filtration, homology_dims: passed to ``tda2s.ph.compute_diagrams``.
        interval, r, resolution: silhouette domain, power-weight exponent and
            grid size (tcda_uq conventions).
        **ph_kwargs: extra ``compute_diagrams`` keyword arguments.

    Returns:
        Tuple ``(phi, A, X)`` with ``phi`` of shape ``[n, n_hom_dim, resolution]``.
    """
    from tda2s.ph import compute_diagrams

    n_hom = len(homology_dims)
    n = len(clouds)
    phi = np.empty((n, n_hom, resolution))
    for i, cloud in enumerate(clouds):
        diags = compute_diagrams(cloud, filtration=filtration,
                                 homology_dims=homology_dims, **ph_kwargs)
        phi[i] = _silhouette_from_diagrams(diags, interval=interval, r=r,
                                           resolution=resolution)
    return phi, np.asarray(A, dtype=int), np.asarray(X, dtype=float)