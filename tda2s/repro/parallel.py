"""Sharded execution of the published designs in :mod:`tda2s.repro`.

Used by the Colab notebooks to run the papers' full 500-replication budgets.
Because replication ``r`` is seeded from ``(base_seed, r)`` alone, shards are
independent and their indicator arrays concatenate into exactly the result a
single sequential process would have produced -- so ``n_jobs`` and ``chunk``
change only the wall clock, never the numbers.

Memory model
------------
Workers are forked, not spawned, so the parent's already-imported numpy, gudhi,
ripser and persim pages are shared copy-on-write rather than duplicated per
worker. Each worker's own working set for these designs is a few MB (20 clouds
of 50 points, 40x40 persistence images), so peak RSS is roughly
``parent + n_jobs * small`` and 32 workers stay comfortably inside 8 GB.

Set the BLAS thread limit to 1 *before importing numpy* when using this module,
or ``n_jobs`` processes will each spin up a full thread pool and oversubscribe
the machine::

    import os
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from . import dubey_muller_rejections, moon_lazar_rejections

__all__ = ["default_workers", "run_moon_lazar_grid", "run_dubey_muller_sweep"]


def default_workers(cap=32):
    """Worker count: all cores, capped (default 32) to bound peak memory."""
    return max(1, min(cap, os.cpu_count() or 1))


def _chunks(n, size):
    return [np.arange(s, min(s + size, n)) for s in range(0, n, size)]


def _ml_task(args):
    key, name, sigma, scenario, rep_ids, base_seed, kwargs = args
    _, rej = moon_lazar_rejections(name, sigma, scenario, rep_ids, base_seed,
                                   **kwargs)
    return key, rep_ids, rej


def _dm_task(args):
    key, rep_ids, base_seed, kwargs = args
    _, rej = dubey_muller_rejections(rep_ids, base_seed, **kwargs)
    return key, rep_ids, rej


def _run(task_fn, tasks, n_jobs, label, progress):
    """Fork a pool, run ``tasks``, and merge indicators per key."""
    acc = {}
    t0 = time.perf_counter()
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as pool:
        futures = [pool.submit(task_fn, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            key, rep_ids, rej = fut.result()
            acc.setdefault(key, []).append((rep_ids, rej))
            if progress and (i % max(1, len(futures) // 20) == 0 or i == len(futures)):
                el = time.perf_counter() - t0
                print(f"  {label}: {i}/{len(futures)} shards  {el:6.1f}s "
                      f"(eta {el * (len(futures) - i) / i:6.1f}s)", flush=True)

    out = {}
    for key, parts in acc.items():
        # sort by replication index so the concatenation is shard-order-invariant
        parts.sort(key=lambda p: p[0][0])
        rej = np.concatenate([p[1] for p in parts])
        out[key] = {"n": int(rej.size), "rejections": int(rej.sum()),
                    "rate": float(rej.mean()),
                    "se": float(np.sqrt(rej.mean() * (1 - rej.mean()) / rej.size))}
    return out


def run_moon_lazar_grid(name, sigmas, scenarios, reps, base_seed=0, n_jobs=None,
                        chunk=25, progress=True, **kwargs):
    """Run the Moon & Lazar Fig. 5 design over a sigma x scenario grid.

    Args:
        name: competitor key (``"moon_lazar"`` for Fig. 5a/5b, ``"rt"`` for the
            "PD" curve of Fig. 5b).
        sigmas: iterable of noise levels.
        scenarios: iterable of ``"fpr"`` / ``"power"``.
        reps: replications per cell (the papers use 500).
        base_seed: offset; each cell uses a distinct derived seed so that no two
            cells share a replication stream.
        n_jobs: worker processes (default :func:`default_workers`).
        chunk: replications per shard.
        progress: print shard-completion progress.
        **kwargs: forwarded to the competitor.

    Returns:
        ``{(sigma, scenario): {"n", "rejections", "rate", "se"}}``.
    """
    n_jobs = n_jobs or default_workers()
    tasks = []
    for si, sigma in enumerate(sigmas):
        for ci, scenario in enumerate(scenarios):
            seed = int(base_seed) + 1000 * (si + 1) + 17 * (ci + 1)
            for ids in _chunks(reps, chunk):
                tasks.append(((float(sigma), scenario), name, float(sigma),
                              scenario, ids, seed, kwargs))
    print(f"{name}: {len(tasks)} shards over {n_jobs} workers "
          f"({len(sigmas) * len(scenarios)} cells x {reps} reps)", flush=True)
    return _run(_ml_task, tasks, n_jobs, name, progress)


def run_dubey_muller_sweep(values, param, reps, base_seed=0, n_jobs=None,
                           chunk=50, progress=True, **kwargs):
    """Run the Dubey & Muller Fig. 1 design over a sweep of one parameter.

    Args:
        values: parameter grid (``delta`` for the left panel, ``r`` for the right).
        param: ``"delta"`` or ``"r"``.
        reps: replications per point (the paper uses 500).
        base_seed: offset; each grid point uses a distinct derived seed.
        n_jobs: worker processes (default :func:`default_workers`).
        chunk: replications per shard.
        progress: print shard-completion progress.
        **kwargs: other design arguments (``sd``, ``n``, ``n_perm``).

    Returns:
        ``{value: {"n", "rejections", "rate", "se"}}``.
    """
    if param not in ("delta", "r"):
        raise ValueError(f"param must be 'delta' or 'r', got {param!r}")
    n_jobs = n_jobs or default_workers()
    tasks = []
    for vi, v in enumerate(values):
        seed = int(base_seed) + 1000 * (vi + 1)
        kw = dict(kwargs)
        kw[param] = float(v)
        for ids in _chunks(reps, chunk):
            tasks.append((float(v), ids, seed, kw))
    print(f"dubey_muller[{param}]: {len(tasks)} shards over {n_jobs} workers "
          f"({len(values)} points x {reps} reps)", flush=True)
    return _run(_dm_task, tasks, n_jobs, f"dm-{param}", progress)
