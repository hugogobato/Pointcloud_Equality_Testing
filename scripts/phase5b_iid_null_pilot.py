"""Small SC-B null-calibration pilot under an i.i.d. point-law null.

This is a prototype check, not the Phase 5C 500-replication gate.  Each arm
contains 20 i.i.d. points and SC-B uses four disjoint blocks of size five.
The block-label null is enumerated exactly, so the pilot isolates the
calibration of the fixed-disjoint-block construction from Monte Carlo error
in its p-values.  The frozen partition is redrawn for every replication
(``partition_seed = base + replication``), so the reported rejection rate
averages over partition draws rather than conditioning on one fixed one.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from tda2s.tests.single_cloud import sc_b_disjoint_mmd


def run_null_pilot(*, replications: int = 100, n_points: int = 20,
                   m: int = 5, alpha: float = 0.05,
                   seed: int = 20260819) -> dict:
    if replications < 1 or n_points < m or m < 2:
        raise ValueError("replications >= 1, n_points >= m, and m >= 2 are required")
    rng = np.random.default_rng(seed)
    pvalues = []
    for replication in range(replications):
        # Same continuous law in both arms.  The two clouds are still drawn
        # independently, as required by the Regime-I lock.
        cloud0 = rng.uniform(0.0, 1.0, size=(n_points, 2))
        cloud1 = rng.uniform(0.0, 1.0, size=(n_points, 2))
        # A fresh frozen partition per replication (base seed + index), so
        # the rejection rate is not conditional on one partition structure.
        result = sc_b_disjoint_mmd(
            cloud0,
            cloud1,
            m=m,
            homology_dims=(0, 1),
            exact=True,
            seed=seed,
            partition_seed=seed + replication,
        )
        pvalues.append(float(result["pvalue"]))
    pvalues = np.asarray(pvalues)
    return {
        "candidate": "SC-B",
        "null": "P0 = P1 = Uniform([0,1]^2)",
        "replications": int(replications),
        "n_points_per_cloud": int(n_points),
        "m": int(m),
        "K0": int(n_points // m),
        "K1": int(n_points // m),
        "alpha": float(alpha),
        "seed": int(seed),
        "rejection_rate": float(np.mean(pvalues <= alpha)),
        "pvalue_min": float(pvalues.min()),
        "pvalue_max": float(pvalues.max()),
        "pvalues": pvalues.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--n-points", type=int, default=20)
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()
    print(json.dumps(run_null_pilot(
        replications=args.replications,
        n_points=args.n_points,
        m=args.m,
        alpha=args.alpha,
        seed=args.seed,
    ), indent=2))


if __name__ == "__main__":
    main()
