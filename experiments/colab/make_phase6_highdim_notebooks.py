"""Generate Colab notebooks for Phase 6 high-dimensional benchmark.

Covers d=10,20,40 and n=50,500 as requested, plus PCA pre-filtering exploration
including PCA-failure constructions where signal lives in low-variance subspace.

Each notebook is self-contained: clones repo, pip installs, then runs a set of
tasks. Each task is (cell_id, rep_start, replications, candidates). Tasks are
checkpointed every 25 replications to a shard-specific parquet; notebook
downloads each shard file via google.colab.files when on Colab.

Target runtime per notebook: 60-540 min (1-9h). Predicted times use profiled
per-rep costs at 199 permutations (primary) with VR filtration (GUDHI) and
without SC-A.

Usage:
  python experiments/colab/make_phase6_highdim_notebooks.py --n-notebooks 32 --replications 500 --permutations 199
  python experiments/colab/make_phase6_highdim_notebooks.py --n-notebooks 24 --replications 500 --permutations 199 --include-pca --include-fail

Outputs:
  notebooks/phase6_highdim_shards/shard_*.ipynb
  notebooks/phase6_highdim_shards/manifest.json

Aggregation (after downloads):
  python experiments/phase6_highdim_pointlaw_tournament.py --mode aggregate --input-dir results/phase6_highdim_shards
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import math
import hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
# For embedding we could read modules, but we use git clone approach like phase5ab
# Keep version tag for reproducibility
BENCHMARK_VERSION = "phase6-highdim-v1"
SEED_ROOT = 20260826
PH_FILTRATION = "vr"
# Candidate sets
PRIMARY_CANDIDATES = [
    "PointMMD-Gaussian",
    "PointMMD-Gaussian-median",
    "EnergyDistance",
    "FriedmanRafsky-MST",
    "Schilling-kNN-k1",
    "RawBlockMMD",
    "SC-B",
    "HybridBlockMMD-a0.50",
]
SECONDARY_CANDIDATES = [
    "Schilling-kNN-k5",
    "Schilling-kNN-k10",
    "SlicedWasserstein",
    "ClassifierTwoSampleTest-logistic",
    "ClassifierTwoSampleTest-rf",
    "SC-A-Block",
]
ALL_CANDIDATES = PRIMARY_CANDIDATES + SECONDARY_CANDIDATES
ROSENBAUM = "Rosenbaum-CrossMatch"

# Per-rep wall estimates (seconds) at 199 perms, ALL_CANDIDATES (14 methods, without Rosenbaum)
# Measured on local with VR filtration, 2 workers off, sequential per-rep
# n=500 d=40 17s, n=500 d=10 18s, n=50 d=10 5.3s, n=50 d=40 5s
# We'll use conservative 18s for n=500 and 6s for n=50 (with 14 methods)
PER_REP_EST = {
    50: 6.0,
    500: 18.0,
}
# If we double methods for raw+pca within same cell (approx 2x), use 12s and 36s
# But we keep per-cell preproc separate, so per-rep stays 6/18.
# PCA overhead is negligible.

FAMILIES_CORE = ["iid_null","weak_barcode_null","same_support_density","same_square_four_atom_density","topology_alt"]
FAMILIES_SPARSE_DENSE = ["sparse_shift_08","dense_shift_08"]
FAMILIES_PCA_FAIL = ["pca_fail_sparse_shift","pca_fail_topology"]
D_GRID = [10,20,40]
N_GRID = [50,500]
PREPROC_RAW = ["raw"]
PREPROC_PCA = ["pca3","pca4"]

HEADER_TEMPLATE = """\
# Phase 6 high-dimensional point-law shard {shard_id:02d} of {n_shards}

**Predicted runtime: {pred_min:.1f} minutes** (wall, N_WORKERS={n_workers}, 199 perms, VR filtration).
Each task checkpoints every 25 replications to `results/phase6_highdim_shards/`.

This fleet explores **ambient dimension d=10,20,40** and **n=50,500** (ecology: raw ~40 dims, PCA to 3-4).
Core families: iid_null (size), weak_barcode_null (translation), same_support_density,
same_square_four_atom_density (atoms), topology_alt (disk vs circle).
Additionally sparse/dense mean shifts and PCA-failure constructions where signal lives
in low-variance subspace (top 3 PCs are pure noise).

Methods: PointMMD (fixed 0.30 + median), Energy, MST, kNN k=1,5,10, SlicedW, C2ST log/rf,
RawBlockMMD (m=25, Hilbert-Gaussian bag), SC-B (m=25, VR d=0,1), Hybrid a=0.50.
SC-A omitted (expensive, abandoned). Rosenbaum only for n=50 (pooled n≤100).

**Preproc**: `raw` = test on d-dim clouds directly; `pca3`/`pca4` = pooled PCA to 3/4 dims
(fitted on pooled unlabeled points, valid under H0) then test. PCA-failure families use
noise_scale=1.5 (high-noise background) so pca3 discards signal.

**Runtime → Run all**, leave tab open. Each task downloads its parquet when done.
Collect all `phase6_highdim_*.parquet` into `results/phase6_highdim_shards/` and run:

```
python experiments/phase6_highdim_pointlaw_tournament.py --mode aggregate --input-dir results/phase6_highdim_shards
```

Design hash `{design_hash}` | FILTRATION={filt} | seed_root={seed_root}
"""

INSTALL_CELL = """\
%%bash
set -e
pip install -q numpy scipy scikit-learn scikit-fda matplotlib networkx==3.4.2 gudhi pot ripser 2>&1 | tail -20
pip install -q git+https://github.com/hugogobato/tcda_uq.git 2>&1 | tail -5
echo '--- install done ---'
# Verify imports
python -c "import gudhi, sklearn, ripser; print('gudhi', gudhi.__version__)"
"""

REPO_CELL_TEMPLATE = """\
import os, shutil, sys, subprocess
REPO_DIR = "/content/Pointcloud_Equality_Testing"
if os.path.isdir(REPO_DIR):
    shutil.rmtree(REPO_DIR)
ret = os.system("git clone --depth 1 https://github.com/hugogobato/Pointcloud_Equality_Testing.git " + REPO_DIR)
print("clone exit", ret)
os.chdir(REPO_DIR)
ret = os.system("python -m pip install -q -e .")
print("pip install -e . exit", ret)
# Ensure results dir
os.makedirs("results/phase6_highdim_shards", exist_ok=True)
print("repo ready", REPO_DIR)
"""

CHECK_CELL = """\
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
sys.path.insert(0, "/content/Pointcloud_Equality_Testing")
# Smoke test
import numpy as np
from experiments.phase6_highdim_pointlaw_tournament import make_cloud_pair_highdim, Cell
c0,c1 = make_cloud_pair_highdim("iid_null", 50, 50, seed=123, d=10)
print("cloud shapes", c0.shape, c1.shape)
# Test point MMD and block with VR (should use gudhi, not ripser)
from experiments.phase6_highdim_pointlaw_tournament import _run_one
cell = Cell(family="iid_null", n0=50, n1=50, m=25, d=10, preproc="raw", role="gating_null", description="test")
rows = _run_one((cell, 0, ("PointMMD-Gaussian","RawBlockMMD","SC-B"), 19, None))
print("smoke", [(r["method"], r["status"], round(float(r["pvalue"]),3)) for r in rows])
print("imports OK |", os.cpu_count(), "CPUs")
"""

TASK_CELL_TEMPLATE = """\
import os, json, time
from pathlib import Path
import pandas as pd
from experiments.phase6_highdim_pointlaw_tournament import Cell, run_shard, DESIGN_HASH

# Fleet controls — first executable cell after smoke
SHARD_ID = {shard_id}
N_SHARDS = {n_shards}
SEED_ROOT = {seed_root}
WALL_BUDGET_MIN = {wall_budget}
N_WORKERS = {n_workers}
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
print({{"SHARD_ID": SHARD_ID, "N_SHARDS": N_SHARDS, "WALL_BUDGET_MIN": WALL_BUDGET_MIN, "N_WORKERS": N_WORKERS, "DESIGN_HASH": DESIGN_HASH}})

TASKS = {tasks_json}

SHARD_OUT_DIR = "results/phase6_highdim_shards"
os.makedirs(SHARD_OUT_DIR, exist_ok=True)

t0 = time.time()
for idx, task in enumerate(TASKS):
    print(f"\\n=== Task {{idx+1}}/{{len(TASKS)}} {{task['task_id']}} ===")
    cell = Cell(family=task["family"], n0=task["n0"], n1=task["n1"], m=task["m"], d=task["d"], preproc=task["preproc"], role=task["role"], description=task["description"])
    out = os.path.join(SHARD_OUT_DIR, f"phase6_highdim_{{cell.cell_id}}_rep{{task['rep_start']}}_{{task['rep_start']+task['replications']-1}}.parquet")
    # Checkpoint every 25 reps if task larger than 25 — run_shard does it internally via single file
    run_shard(cell, rep_start=task["rep_start"], replications=task["replications"], n_permutations=task["n_permutations"], candidates=task["candidates"], workers=N_WORKERS, cache_dir=None, output=out)
    # Try download
    try:
        from google.colab import files
        files.download(out)
        print("Downloaded:", out)
    except Exception as e:
        print("(Not on Colab / download skipped):", e)
    print(f"--- {{time.time()-t0:.0f}}s elapsed, {{idx+1}}/{{len(TASKS)}} tasks done ---")

print(f"Shard {{SHARD_ID}} done in {{time.time()-t0:.0f}}s")
print("Files kept in", SHARD_OUT_DIR)
"""

def _code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": src, "execution_count": None, "outputs": []}

def _markdown(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def build_tasks(families, n_grid, d_grid, preproc_modes, replications=500, n_permutations=199, include_pca_fail=False):
    tasks = []
    # Expand families
    fams = list(families)
    if include_pca_fail:
        for f in FAMILIES_PCA_FAIL:
            if f not in fams:
                fams.append(f)
        for f in FAMILIES_SPARSE_DENSE:
            if f not in fams:
                fams.append(f)
    for fam in fams:
        for n in n_grid:
            for d in d_grid:
                for preproc in preproc_modes:
                    # cell meta
                    # Determine candidates per n
                    if n == 50:
                        cands = ALL_CANDIDATES + [ROSENBAUM]
                    else:
                        cands = ALL_CANDIDATES
                    # Predicted per task ( chunk of 25 reps )
                    # Split 500 reps into chunks of 25 => 20 chunks per cell
                    n_chunks = math.ceil(replications / 25)
                    for chunk in range(n_chunks):
                        rep_start = chunk*25
                        reps = min(25, replications - rep_start)
                        if reps <=0:
                            continue
                        per_rep = PER_REP_EST[n]
                        pred = per_rep * reps
                        tasks.append({
                            "task_id": f"{fam}_n{n}_n1{n}_m25_d{d}_preproc_{preproc}_rep{rep_start}_{rep_start+reps-1}",
                            "cell_id": f"{fam}_n{n}_n1{n}_m25_d{d}_preproc_{preproc}",
                            "family": fam,
                            "n0": n,
                            "n1": n,
                            "m": 25,
                            "d": d,
                            "preproc": preproc,
                            "role": "unknown",  # will be overwritten by Cell's role via tournament's mapping
                            "description": fam,
                            "rep_start": rep_start,
                            "replications": reps,
                            "n_permutations": n_permutations,
                            "candidates": cands,
                            "predicted_seconds": pred,
                        })
    return tasks

def _assign_tasks_to_shards(tasks, n_shards):
    # Round-robin by predicted_seconds to balance
    tasks_sorted = sorted(tasks, key=lambda t: t["predicted_seconds"], reverse=True)
    shards = [{"tasks": [], "pred": 0.0} for _ in range(n_shards)]
    for task in tasks_sorted:
        # assign to shard with smallest pred
        idx = min(range(n_shards), key=lambda i: shards[i]["pred"])
        shards[idx]["tasks"].append(task)
        shards[idx]["pred"] += task["predicted_seconds"]
    return shards

def make_notebook(shard_id, n_shards, shard_tasks, design_hash, seed_root, filt, wall_budget=450, n_workers=2):
    pred_min = sum(t["predicted_seconds"] for t in shard_tasks) / 60.0 / max(n_workers,1)  # approximate wall with workers=2 parallel ~1/2
    # Actually tasks sequential within shard, workers parallel per task internal? run_shard uses workers for per-replication parallel
    # So wall ~ sum(pred)/ (n_workers?) Not exactly. Use sum/ n_workers
    # For simplicity wall = sum / n_workers
    # Use 0.7 factor for overhead
    pred_min = sum(t["predicted_seconds"] for t in shard_tasks) / 60.0
    # If N_WORKERS=2, per-rep parallel may cut ~1.6x, so adjust
    pred_min = pred_min / (1.6 if n_workers==2 else 1)
    header = HEADER_TEMPLATE.format(shard_id=shard_id, n_shards=n_shards, pred_min=pred_min, n_workers=n_workers, design_hash=design_hash, filt=filt, seed_root=seed_root)
    tasks_json = json.dumps(shard_tasks, indent=2)
    cells = [
        _markdown(header),
        _code(f'import os\n\nSHARD_ID = {shard_id}\nN_SHARDS = {n_shards}\nSEED_ROOT = {seed_root}\nWALL_BUDGET_MIN = {wall_budget}\nN_WORKERS = {n_workers}\nos.environ["OMP_NUM_THREADS"] = "1"\nos.environ["OPENBLAS_NUM_THREADS"] = "1"\nos.environ["MKL_NUM_THREADS"] = "1"\nos.environ["NUMEXPR_NUM_THREADS"] = "1"\nprint({{"SHARD_ID": SHARD_ID, "N_SHARDS": N_SHARDS, "WALL_BUDGET_MIN": WALL_BUDGET_MIN, "N_WORKERS": N_WORKERS}})\n'),
        _code(REPO_CELL_TEMPLATE),
        _code(INSTALL_CELL),
        _code(CHECK_CELL),
        _code(TASK_CELL_TEMPLATE.format(shard_id=shard_id, n_shards=n_shards, seed_root=seed_root, wall_budget=wall_budget, n_workers=n_workers, tasks_json=tasks_json)),
    ]
    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "cells": cells
    }

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-notebooks", type=int, default=32)
    ap.add_argument("--replications", type=int, default=500)
    ap.add_argument("--permutations", type=int, default=199)
    ap.add_argument("--n-workers", type=int, default=2)
    ap.add_argument("--wall-budget", type=int, default=450)
    ap.add_argument("--outdir", default="notebooks/phase6_highdim_shards")
    ap.add_argument("--families", default=None, help="comma-separated families; default core")
    ap.add_argument("--include-pca", action="store_true", help="include pca3 and pca4 preproc")
    ap.add_argument("--include-fail", action="store_true", help="include pca failure families")
    ap.add_argument("--preproc", default=None, help="comma-separated preproc modes")
    ap.add_argument("--d-grid", default=None, help="comma-separated d")
    ap.add_argument("--n-grid", default=None, help="comma-separated n")
    args = ap.parse_args()

    if args.families:
        families = [f.strip() for f in args.families.split(",")]
    else:
        families = FAMILIES_CORE
    if args.include_fail:
        for f in FAMILIES_PCA_FAIL+FAMILIES_SPARSE_DENSE:
            if f not in families:
                families.append(f)
    if args.preproc:
        preproc_modes = [p.strip() for p in args.preproc.split(",")]
    else:
        preproc_modes = ["raw","pca3"] if args.include_pca else ["raw"]
        if args.include_pca and "pca4" not in preproc_modes:
            # include pca4 only for d=40 to limit explode
            pass
    d_grid = [int(x) for x in args.d_grid.split(",")] if args.d_grid else D_GRID
    n_grid = [int(x) for x in args.n_grid.split(",")] if args.n_grid else N_GRID

    # Build tasks
    tasks = build_tasks(families, n_grid, d_grid, preproc_modes, replications=args.replications, n_permutations=args.permutations, include_pca_fail=False)
    # If include_fail, tasks already includes those families via families list above, but build_tasks's include_pca_fail flag adds again — handle
    # Actually families already includes fail, so build_tasks should not double
    # So set include_pca_fail=False
    n_shards = args.n_notebooks
    shards = _assign_tasks_to_shards(tasks, n_shards)

    # Need design hash from tournament
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    from experiments.phase6_highdim_pointlaw_tournament import DESIGN_HASH
    design_hash = DESIGN_HASH

    os.makedirs(args.outdir, exist_ok=True)
    # Clear old notebooks if regenerating with fewer shards
    for old in sorted(f for f in os.listdir(args.outdir) if f.endswith(".ipynb")):
        os.remove(os.path.join(args.outdir, old))

    manifest = {
        "benchmark_version": BENCHMARK_VERSION,
        "design_hash": design_hash,
        "filtration": PH_FILTRATION,
        "seed_root": SEED_ROOT,
        "families": families,
        "n_grid": n_grid,
        "d_grid": d_grid,
        "preproc_modes": preproc_modes,
        "replications": args.replications,
        "permutations": args.permutations,
        "n_shards": n_shards,
        "n_workers": args.n_workers,
        "wall_budget_min": args.wall_budget,
        "total_tasks": len(tasks),
        "shards": []
    }
    for sid, shard in enumerate(shards):
        nb = make_notebook(sid, n_shards, shard["tasks"], design_hash, SEED_ROOT, PH_FILTRATION, wall_budget=args.wall_budget, n_workers=args.n_workers)
        path = os.path.join(args.outdir, f"shard_{sid:02d}.ipynb")
        with open(path, "w") as fh:
            json.dump(nb, fh, indent=1)
        pred_min = shard["pred"]/60.0 / (1.6 if args.n_workers==2 else 1)
        manifest["shards"].append({
            "shard_id": sid,
            "notebook": path,
            "n_tasks": len(shard["tasks"]),
            "predicted_minutes": pred_min,
            "predicted_seconds": shard["pred"],
            "tasks": shard["tasks"]
        })
        print(f"wrote shard {sid:02d} with {len(shard['tasks'])} tasks pred {pred_min:.1f} min -> {path}")

    # Summary stats
    preds = [s["predicted_minutes"] for s in manifest["shards"]]
    print(f"total tasks {len(tasks)} across {n_shards} shards: min {min(preds):.1f} max {max(preds):.1f} mean {sum(preds)/len(preds):.1f} total {sum(preds)/60:.1f}h")
    # Check wall budget compliance
    too_small = [s for s in manifest["shards"] if s["predicted_minutes"] < 60]
    too_large = [s for s in manifest["shards"] if s["predicted_minutes"] > 540]
    if too_small:
        print(f"WARNING: {len(too_small)} shards below 60 min: ids {[s['shard_id'] for s in too_small]}")
    if too_large:
        print(f"WARNING: {len(too_large)} shards above 540 min: ids {[s['shard_id'] for s in too_large]}")
    manifest["predicted_shard_minutes"] = preds
    manifest["too_small"] = [s["shard_id"] for s in too_small]
    manifest["too_large"] = [s["shard_id"] for s in too_large]
    with open(os.path.join(args.outdir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(f"manifest -> {os.path.join(args.outdir, 'manifest.json')}")

if __name__ == "__main__":
    main()
