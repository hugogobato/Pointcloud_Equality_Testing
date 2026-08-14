"""Phase 0 exit criterion: end-to-end run under 10 minutes on 20 threads.

    generate 100 clouds -> compute diagrams -> vectorise -> run all competitors
    -> get p-values

Usage
-----
    rtk uv run python scripts/phase0_exit_benchmark.py
    rtk uv run python scripts/phase0_exit_benchmark.py --workers 8 --no-cache

Notes
-----
* Diagram computation is the only parallel stage; it is embarrassingly parallel
  over clouds and is the dominant cost. Workers are capped at 16 (of the box's
  20 logical CPUs) so a concurrent experiment is not starved, and clouds are
  small, so peak RSS stays in the low hundreds of MB.
* Diagrams are cached on disk by (cloud hash, filtration, params). The timing
  reported as the headline is the COLD run (cache cleared); the warm run is
  reported alongside because that is what the permutation loops of Phases 3-5
  will actually see.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tda2s.benchmarks import COMPETITORS, run_competitor
from tda2s.dgp.simulation import CloudSampleDGP
from tda2s.ph import compute_diagrams
from tda2s.vec import vectorise

REPRESENTATIONS = ("silhouette", "landscape", "betti", "euler", "image", "measure")


def _diagram_job(args):
    """Top-level so it pickles for ProcessPoolExecutor."""
    points, filtration, homology_dims, cache_dir = args
    return compute_diagrams(points, filtration=filtration,
                            homology_dims=homology_dims, cache_dir=cache_dir)


class Stopwatch:
    def __init__(self):
        self.laps = []

    def lap(self, name, seconds):
        self.laps.append((name, seconds))
        print(f"  {name:<34s} {seconds:8.2f} s", flush=True)

    @property
    def total(self):
        return sum(s for _, s in self.laps)


def run(n_clouds, m, filtration, homology_dims, workers, n_perm, cache_dir, seed):
    sw = Stopwatch()

    # ---- 1. generate clouds -------------------------------------------------
    t0 = time.perf_counter()
    dgp = CloudSampleDGP(n_per_group=n_clouds // 2, m=m, d_x=3, gamma=1.0, k_max=3,
                         radius=1.0, noise=0.05, group_effect=1)
    sample = dgp.sample(rng=np.random.default_rng(seed))
    sw.lap(f"generate {len(sample.clouds)} clouds ({m} pts)", time.perf_counter() - t0)

    # ---- 2. diagrams (parallel) --------------------------------------------
    jobs = [(c, filtration, homology_dims, cache_dir) for c in sample.clouds]
    t0 = time.perf_counter()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            diagrams = list(pool.map(_diagram_job, jobs, chunksize=4))
    else:
        diagrams = [_diagram_job(j) for j in jobs]
    sw.lap(f"diagrams [{filtration}, {workers} workers]", time.perf_counter() - t0)

    n_feat = sum(len(d) for diags in diagrams for d in diags)
    print(f"    ({n_feat} finite features across {len(diagrams)} diagrams)", flush=True)

    # ---- 3. vectorise -------------------------------------------------------
    t0 = time.perf_counter()
    vectors = {}
    for rep in REPRESENTATIONS:
        vectors[rep] = np.stack([np.ravel(vectorise(d, rep)) for d in diagrams])
    sw.lap(f"vectorise x{len(REPRESENTATIONS)} representations", time.perf_counter() - t0)
    for rep in REPRESENTATIONS:
        print(f"    {rep:<12s} {vectors[rep].shape}", flush=True)

    # ---- 4. competitors -----------------------------------------------------
    A = sample.A
    diags1 = [d for d, a in zip(diagrams, A) if a == 1]
    diags0 = [d for d, a in zip(diagrams, A) if a == 0]
    print(f"  groups: n1={len(diags1)}, n0={len(diags0)}", flush=True)

    pvalues = {}
    t_all = time.perf_counter()
    for name in COMPETITORS:
        t0 = time.perf_counter()
        pvalues[name] = run_competitor(name, diags0, diags1, n_perm=n_perm, seed=0)
        sw.laps.append((f"competitor {name}", time.perf_counter() - t0))
        print(f"  competitor {name:<24s} {time.perf_counter() - t0:8.2f} s  "
              f"p = {pvalues[name]:.4f}", flush=True)
    print(f"  {'all competitors':<34s} {time.perf_counter() - t_all:8.2f} s", flush=True)

    return sw, pvalues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clouds", type=int, default=100)
    ap.add_argument("--m", type=int, default=150, help="points per cloud")
    ap.add_argument("--filtration", default="alpha")
    ap.add_argument("--homology-dims", default="0,1")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--cache-dir", default=".cache/phase0_bench")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--budget", type=float, default=600.0, help="exit budget, seconds")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dims = tuple(int(x) for x in args.homology_dims.split(","))
    cache_dir = None if args.no_cache else args.cache_dir
    if cache_dir and os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)     # headline run is COLD

    print(f"Phase 0 exit benchmark -- {args.n_clouds} clouds, {args.m} pts, "
          f"{args.filtration}, H{dims}, {args.workers} workers, "
          f"{args.n_perm} permutations\n")

    print("COLD run (no cached diagrams)")
    sw, pvalues = run(args.n_clouds, args.m, args.filtration, dims, args.workers,
                      args.n_perm, cache_dir, args.seed)
    cold = sw.total

    warm = None
    if cache_dir:
        print("\nWARM run (diagrams read from cache)")
        sw2, _ = run(args.n_clouds, args.m, args.filtration, dims, args.workers,
                     args.n_perm, cache_dir, args.seed)
        warm = sw2.total

    print("\n" + "=" * 62)
    print(f"{'COLD total':<34s} {cold:8.2f} s   (budget {args.budget:.0f} s)")
    if warm is not None:
        print(f"{'WARM total':<34s} {warm:8.2f} s")
    print("=" * 62)
    print("p-values:", {k: round(v, 4) for k, v in pvalues.items()})

    ok = cold < args.budget
    print(f"\nPhase 0 exit criterion: {'PASS' if ok else 'FAIL'} "
          f"({cold:.1f} s {'<' if ok else '>='} {args.budget:.0f} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
