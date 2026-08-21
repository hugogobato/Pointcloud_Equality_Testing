"""Small source-aligned pilot for the Phase 5B SC-C prototype.

Roycraft, Krebs, and Polonik study persistent Betti statistics for binomial
and Poisson point sets in Euclidean space and compare ordinary and smoothed
bootstrap procedures.  This script fixes one simple binomial setting, a
uniform law on the unit square, and records the two bootstrap paths under the
P1 two-cloud contrast.  It is a reproducible implementation check, not a
claim to reproduce the paper's full confidence-coverage tables.

The default is intentionally cheap.  Increase ``--replications`` and
``--draws`` only after the prototype passes this check.  Every bootstrap draw
recomputes PH, so large runs should be sharded over independent replications
before the Phase 5C fleet.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from tda2s.tests.single_cloud import (
    sc_c_finite_vector,
    sc_c_naive_bootstrap,
    roycraft_reference_setting,
)


def run_pilot(*, replications: int = 4, n_points: int = 32,
              n_draws: int = 19, seed: int = 20260819) -> dict:
    if replications < 1 or n_points < 4 or n_draws < 1:
        raise ValueError("replications, n_points, and n_draws must be positive")
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 0.75, 7)
    rows = []
    for replication in range(replications):
        # Binomial point samples from one continuous Euclidean density, which
        # is the source-aligned setting used for this prototype check.
        cloud0 = rng.uniform(0.0, 1.0, size=(n_points, 2))
        cloud1 = rng.uniform(0.0, 1.0, size=(n_points, 2))
        common = dict(
            filtration="vr",
            homology_dims=(0, 1),
            max_edge_length=0.75,
            grid=grid,
            bootstrap_bandwidth=0.05,
            n_draws=n_draws,
            seed=seed + replication,
        )
        smooth = sc_c_finite_vector(cloud0, cloud1, **common)
        naive = sc_c_naive_bootstrap(cloud0, cloud1, **common)
        rows.append({
            "replication": replication,
            "smoothed_pvalue": float(smooth["pvalue"]),
            "naive_pvalue": float(naive["pvalue"]),
            "smoothed_statistic": float(smooth["statistic"]),
            "naive_statistic": float(naive["statistic"]),
            "smoothed_runtime_seconds": float(smooth["runtime_seconds"]),
            "naive_runtime_seconds": float(naive["runtime_seconds"]),
        })
    return {
        "setting": roycraft_reference_setting(),
        "pilot": {
            "point_law": "Uniform([0,1]^2), independent binomial clouds",
            "n_points_per_cloud": int(n_points),
            "replications": int(replications),
            "bootstrap_draws": int(n_draws),
            "alpha": 0.05,
            "seed": int(seed),
            "filtration": "vr radius scale",
            "grid": grid.tolist(),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=4)
    parser.add_argument("--n-points", type=int, default=32)
    parser.add_argument("--draws", type=int, default=19)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()
    print(json.dumps(run_pilot(
        replications=args.replications,
        n_points=args.n_points,
        n_draws=args.draws,
        seed=args.seed,
    ), indent=2))


if __name__ == "__main__":
    main()
