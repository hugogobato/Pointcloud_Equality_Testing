"""Numerical hardening checks for the Phase 5D SC-B production kernel.

The fleet is intentionally limited to the two registered primary-null
families and the registered topology alternative.  It reruns SC-B only, with
the corrected tensor-product joint kernel.  Independent original-cloud
replications may be parallelized; permutations remain inside each replication
and never create additional workers.

Examples:

    python scripts/phase5d_scb_hardening.py --replications 100 --workers 4
    python scripts/phase5d_scb_hardening.py --replications 500 --workers 16 \
        --output results/phase5d_scb_hardening.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import beta

from experiments.phase5_single_cloud_tournament import make_cloud_pair
from tda2s.tests.single_cloud import sc_b_production_test


FAMILIES = ("iid_null", "weak_barcode_null", "topology_alt")
N_GRID = (250, 500, 1000)
ALPHA = 0.05
PERMUTATIONS = 39
SEED_ROOT = 20260819


def _seed(family: str, n: int, replication: int) -> int:
    payload = repr((SEED_ROOT, "phase5d", family, n, replication)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _run_one(args: tuple[str, int, int]) -> dict:
    family, n, replication = args
    cloud_seed = _seed(family, n, replication)
    partition_seed = _seed("partition-" + family, n, replication)
    method_seed = _seed("method-" + family, n, replication)
    cloud0, cloud1 = make_cloud_pair(family, n, n, cloud_seed)
    result = sc_b_production_test(
        cloud0,
        cloud1,
        partition_seed=partition_seed,
        n_perm=PERMUTATIONS,
        seed=method_seed,
    )
    return {
        "family": family,
        "n0": n,
        "n1": n,
        "replication": replication,
        "pvalue": float(result["pvalue"]),
        "reject": bool(result["pvalue"] <= ALPHA),
        "statistic": float(result["statistic"]),
        "K0": int(result["K0"]),
        "K1": int(result["K1"]),
        "m": int(result["m"]),
        "n_permutations": int(result["n_permutations"]),
        "kernel": result["diagnostics"]["kernel"],
        "production_api": result["production_api"],
        "runtime_seconds": float(result["runtime_seconds"]),
    }


def _interval(successes: int, total: int) -> tuple[float, float]:
    if successes == 0:
        low = 0.0
    else:
        low = float(beta.ppf(0.025, successes, total - successes + 1))
    if successes == total:
        high = 1.0
    else:
        high = float(beta.ppf(0.975, successes + 1, total - successes))
    return low, high


def run(replications: int, workers: int, output: str) -> pd.DataFrame:
    if replications < 1:
        raise ValueError("replications must be positive")
    if workers < 1 or workers > 16:
        raise ValueError("workers must be in [1, 16]")
    args = [
        (family, n, replication)
        for family in FAMILIES
        for n in N_GRID
        for replication in range(replications)
    ]
    if workers == 1:
        rows = [_run_one(arg) for arg in args]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_run_one, args))
    frame = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    frame.to_parquet(output, index=False)
    summary = []
    for (family, n), group in frame.groupby(["family", "n0"]):
        successes = int(group["reject"].sum())
        low, high = _interval(successes, len(group))
        summary.append({
            "family": family,
            "n": int(n),
            "replications": int(len(group)),
            "rejections": successes,
            "rejection_rate": successes / len(group),
            "mc_low": low,
            "mc_high": high,
        })
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"Wrote {len(frame)} replication rows to {output}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--output", default="results/phase5d_scb_hardening.parquet"
    )
    args = parser.parse_args()
    run(args.replications, args.workers, args.output)


if __name__ == "__main__":
    main()
