"""Controlled DGP harness: point-cloud generators and two-group datasets.

Public API:
  * generators: ``circle_cloud``, ``torus_cloud``, ``sphere_cloud``,
    ``cluster_cloud``, ``loops_cloud``;
  * harness: ``CloudSampleDGP`` (two-group datasets with covariate-driven
    topology, propensity, and group-effect knobs), ``CloudSample`` (the
    realised dataset with per-cloud oracles);
  * export: ``to_silhouette_sample`` (clouds -> tcda_uq-format ``(phi, A, X)``).
"""

from .clouds import circle_cloud, cluster_cloud, loops_cloud, sphere_cloud, torus_cloud
from .simulation import CloudSample, CloudSampleDGP, to_silhouette_sample

__all__ = [
    "circle_cloud",
    "torus_cloud",
    "sphere_cloud",
    "cluster_cloud",
    "loops_cloud",
    "CloudSampleDGP",
    "CloudSample",
    "to_silhouette_sample",
]