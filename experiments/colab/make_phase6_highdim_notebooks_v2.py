"""Generate Colab notebooks for Phase 6 high-dimensional benchmark v2.

Fixes:
- single zip download per shard (browser limit, one download/tab)
- skip existing parquet (idempotent reruns)
- configurable tasks per shard (default 20, avoids timeouts: some 67-task shards only did 30)
- 2-phase support: pilot (100 reps, 99 perms, primary candidates) = explanatory,
                 thorough (500 reps, 199 perms, all candidates) = dense power/size
Usage:
  # Pilot core raw (explanatory, ~50 min/shard)
  python experiments/colab/make_phase6_highdim_notebooks_v2.py --outdir notebooks/phase6_highdim_core_raw_pilot --families iid_null,weak_barcode_null,same_support_density,same_square_four_atom_density,topology_alt --replications 100 --permutations 99 --tasks-per-shard 20 --candidates primary --preproc raw --d-grid 10,20,40 --n-grid 50,500 --n-workers 2

  # Thorough core raw (only after pilot promising)
  python experiments/colab/make_phase6_highdim_notebooks_v2.py --outdir notebooks/phase6_highdim_core_raw_thorough --families iid_null,weak_barcode_null,same_support_density,same_square_four_atom_density,topology_alt --replications 500 --permutations 199 --tasks-per-shard 20 --candidates all --preproc raw

  # Pilot extra dense (dense signal in many dims)
  python experiments/colab/make_phase6_highdim_notebooks_v2.py --outdir notebooks/phase6_highdim_extra_dense_pilot --families dense_same_support_density,dense_topology_rotated,covariance_shift --replications 100 --permutations 99 --tasks-per-shard 20 --candidates primary --preproc raw,pca3

  # Thorough extra dense + remaining shards for phase6_highdim_shards
  python experiments/colab/make_phase6_highdim_notebooks_v2.py --outdir notebooks/phase6_highdim_shards_remaining --families iid_null,weak_barcode_null,same_support_density,same_square_four_atom_density,topology_alt,pca_fail_sparse_shift,pca_fail_topology,sparse_shift_08,dense_shift_08 --replications 500 --permutations 199 --tasks-per-shard 20 --candidates all --preproc raw,pca3

All notebooks zip results/ph-<fleet>/ into /tmp/<fleet>_shardXX.zip and download ONE file per tab.
Aggregation: python experiments/phase6_highdim_pointlaw_tournament.py --mode aggregate --input-dir results/<fleet>
"""
from __future__ import annotations
import argparse, json, os, sys, math, shutil, glob

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
BENCHMARK_VERSION = "phase6-highdim-v1"
SEED_ROOT = 20260826
PH_FILTRATION = "vr"

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

PER_REP_EST = {50: 6.0, 500: 18.0}
FAMILIES_CORE = ["iid_null","weak_barcode_null","same_support_density","same_square_four_atom_density","topology_alt"]
FAMILIES_SPARSE_DENSE = ["sparse_shift_08","dense_shift_08"]
FAMILIES_PCA_FAIL = ["pca_fail_sparse_shift","pca_fail_topology"]
FAMILIES_EXTRA_DENSE = ["dense_same_support_density","dense_topology_rotated","covariance_shift"]
D_GRID = [10,20,40]
N_GRID = [50,500]

HEADER_TEMPLATE = """\
# Phase 6 high-dimensional point-law shard {shard_id:02d} of {n_shards} — {fleet_name} [{phase_label}]

**Predicted runtime: {pred_min:.1f} minutes** (wall, N_WORKERS={n_workers}, {perms} perms, {cands_label}, VR filtration).
Tasks in shard: {n_tasks} (≤{tasks_per_shard} cap) → each task = 25 replications checkpointed to `{results_dir}/`.

This is **{phase_desc}**. {phase_note}

**Fleet**: d={d_grid}, n={n_grid}, preproc={preproc_modes}, families={families}
Methods: {cands_label}. SC-A omitted. Rosenbaum only for n=50.
Preproc `raw` vs `pca3` (pooled PCA to 3 dims, valid under H0). PCA-failure families use high-noise background.

**Download**: at end of shard ONE zip is created (`/tmp/{fleet_name}_shardXX.zip`) containing all parquets in `{results_dir}/` and downloaded via `google.colab.files.download` (avoids per-tab multi-download limit). Keep tab open until “Downloaded zip” prints. Then collect all downloaded zips locally, unzip into `results/{fleet_name}/` and run:

```
python experiments/phase6_highdim_pointlaw_tournament.py --mode aggregate --input-dir results/{fleet_name}
```

Design hash `{design_hash}` | FILTRATION={filt} | seed_root={seed_root}
"""

REPO_CELL_TEMPLATE = """\
import os, shutil, sys, subprocess
REPO_DIR = "/content/Pointcloud_Equality_Testing"
FLEET_NAME = "{fleet_name}"
RESULTS_DIR = "{results_dir}"
if os.path.isdir(REPO_DIR):
    shutil.rmtree(REPO_DIR)
ret = os.system("git clone --depth 1 https://github.com/hugogobato/Pointcloud_Equality_Testing.git " + REPO_DIR)
print("clone exit", ret)
os.chdir(REPO_DIR)
ret = os.system("python -m pip install -q -e .")
print("pip install -e . exit", ret)
os.makedirs(RESULTS_DIR, exist_ok=True)
print("repo ready", REPO_DIR, "results", RESULTS_DIR)
"""

INSTALL_CELL = """\
%%bash
set -e
pip install -q numpy scipy scikit-learn scikit-fda matplotlib networkx==3.4.2 gudhi pot ripser 2>&1 | tail -20
pip install -q git+https://github.com/hugogobato/tcda_uq.git 2>&1 | tail -5
echo '--- install done ---'
python -c "import gudhi, sklearn, ripser; print('gudhi', gudhi.__version__)"
"""

CHECK_CELL = """\
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
sys.path.insert(0, "/content/Pointcloud_Equality_Testing")
import numpy as np
from experiments.phase6_highdim_pointlaw_tournament import make_cloud_pair_highdim, Cell
c0,c1 = make_cloud_pair_highdim("iid_null", 50, 50, seed=123, d=10)
print("cloud shapes", c0.shape, c1.shape)
from experiments.phase6_highdim_pointlaw_tournament import _run_one
cell = Cell(family="iid_null", n0=50, n1=50, m=25, d=10, preproc="raw", role="gating_null", description="test")
rows = _run_one((cell, 0, ("PointMMD-Gaussian","RawBlockMMD","SC-B"), 19, None))
print("smoke", [(r["method"], r["status"], round(float(r["pvalue"]),3)) for r in rows])
print("imports OK |", os.cpu_count(), "CPUs")
"""

TASK_CELL_TEMPLATE = """\
import os, json, time, glob, shutil
from pathlib import Path
import pandas as pd
from experiments.phase6_highdim_pointlaw_tournament import Cell, run_shard, DESIGN_HASH

SHARD_ID = {shard_id}
N_SHARDS = {n_shards}
FLEET_NAME = "{fleet_name}"
RESULTS_DIR = "{results_dir}"
SEED_ROOT = {seed_root}
WALL_BUDGET_MIN = {wall_budget}
N_WORKERS = {n_workers}
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
print({{"SHARD_ID": SHARD_ID, "N_SHARDS": N_SHARDS, "FLEET": FLEET_NAME, "WALL_BUDGET_MIN": WALL_BUDGET_MIN, "N_WORKERS": N_WORKERS, "DESIGN_HASH": DESIGN_HASH}})

TASKS = {tasks_json}

os.makedirs(RESULTS_DIR, exist_ok=True)

t0 = time.time()
skipped = 0
ran = 0
for idx, task in enumerate(TASKS):
    print(f"\\n=== Task {{idx+1}}/{{len(TASKS)}} {{task['task_id']}} ({{task['n_permutations']}} perms, {{task['replications']}} reps) ===")
    cell = Cell(family=task["family"], n0=task["n0"], n1=task["n1"], m=task["m"], d=task["d"], preproc=task["preproc"], role=task["role"], description=task["description"])
    out = os.path.join(RESULTS_DIR, f"phase6_highdim_{{cell.cell_id}}_rep{{task['rep_start']}}_{{task['rep_start']+task['replications']-1}}.parquet")
    if os.path.exists(out):
        # quick integrity check: can read?
        try:
            import pandas as pd
            df = pd.read_parquet(out)
            if len(df) > 0:
                print(f"  -> Skipping, already exists: {{out}} ({{len(df)}} rows)")
                skipped += 1
                continue
        except Exception as e:
            print(f"  -> Existing file unreadable, will rerun: {{e}}")
    # checkpoint every 25 reps if task larger than 25 — run_shard does it internally via single file
    run_shard(cell, rep_start=task["rep_start"], replications=task["replications"], n_permutations=task["n_permutations"], candidates=task["candidates"], workers=N_WORKERS, cache_dir=None, output=out)
    ran += 1
    print(f"  -> done {{out}}")
    print(f"--- {{time.time()-t0:.0f}}s elapsed, {{idx+1}}/{{len(TASKS)}} tasks done (ran {{ran}}, skipped {{skipped}}) ---")
    # Optional: show partial zip progress but do NOT download per-task (hits browser limit)

print(f"\\nShard {{SHARD_ID}} tasks done in {{time.time()-t0:.0f}}s (ran {{ran}}, skipped {{skipped}})")
# Create single zip for download (one download per tab)
import shutil, glob, os
parquets = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.parquet")))
print(f"Shard produced {{len(parquets)}} parquets in {{RESULTS_DIR}}")
zip_base = f"/tmp/{{FLEET_NAME}}_shard{{SHARD_ID:02d}}"
# make_archive expects base without extension, root_dir is RESULTS_DIR's parent but we want archive containing files flat
# We'll create zip with files flat via shutil.make_archive from RESULTS_DIR
# Use temporary staging: archive the RESULTS_DIR folder
try:
    # Remove old zip if exists
    if os.path.exists(zip_base + ".zip"):
        os.remove(zip_base + ".zip")
    shutil.make_archive(zip_base, "zip", RESULTS_DIR)
    zip_path = zip_base + ".zip"
    sz = os.path.getsize(zip_path) / 1e6
    print(f"Archive created: {{zip_path}} ({{sz:.1f}} MB) with {{len(parquets)}} files")
    try:
        from google.colab import files
        files.download(zip_path)
        print("Downloaded zip:", zip_path)
    except Exception as e:
        print("(Not on Colab / download skipped - manual download needed):", e)
        print("If on Colab, run: from google.colab import files; files.download('"+zip_path+"')")
    print("Files kept in", RESULTS_DIR, "and", zip_path)
except Exception as e:
    print("Zip creation failed:", e)
    # fallback: try to download individual files if zip fails (but will hit limit)
    print("Parquets:", parquets[:5])

try:
    from google.colab import files
    files.download(zip_path)
    print("Downloaded:", zip_path)
except Exception as e:
    print("(Not on Colab / download skipped):", e)
print("Done.")
"""

def _code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": src, "execution_count": None, "outputs": []}
def _markdown(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def build_tasks(families, n_grid, d_grid, preproc_modes, replications=500, n_permutations=199, candidates=None):
    if candidates is None:
        candidates = ALL_CANDIDATES
    tasks=[]
    for fam in families:
        for n in n_grid:
            for d in d_grid:
                for preproc in preproc_modes:
                    if n==50:
                        cands = list(candidates) + [ROSENBAUM] if ROSENBAUM not in candidates else list(candidates)
                    else:
                        cands = list(candidates)
                    n_chunks = math.ceil(replications/25)
                    for chunk in range(n_chunks):
                        rep_start = chunk*25
                        reps = min(25, replications-rep_start)
                        if reps<=0:
                            continue
                        # Adjust per-rep estimate for perms: scale linearly with perms
                        base = PER_REP_EST[n]
                        # 199 perms -> base, scale for other perms
                        per_rep = base * (n_permutations/199.0)
                        # candidates scaling: primary 8 vs all 14 ~1.75x, but per_rep already for all; scale down for primary
                        if len(cands) <= 8:
                            per_rep = per_rep * 0.6
                        pred = per_rep * reps
                        tasks.append({
                            "task_id": f"{fam}_n{n}_n1{n}_m25_d{d}_preproc_{preproc}_rep{rep_start}_{rep_start+reps-1}",
                            "cell_id": f"{fam}_n{n}_n1{n}_m25_d{d}_preproc_{preproc}",
                            "family": fam, "n0": n, "n1": n, "m": 25, "d": d, "preproc": preproc,
                            "role": "unknown", "description": fam,
                            "rep_start": rep_start, "replications": reps,
                            "n_permutations": n_permutations, "candidates": cands,
                            "predicted_seconds": pred,
                        })
    return tasks

def _assign_tasks_to_shards(tasks, n_shards):
    # round-robin by predicted_seconds
    tasks_sorted = sorted(tasks, key=lambda t: t["predicted_seconds"], reverse=True)
    shards = [{"tasks": [], "pred": 0.0} for _ in range(n_shards)]
    for task in tasks_sorted:
        idx = min(range(n_shards), key=lambda i: shards[i]["pred"])
        shards[idx]["tasks"].append(task)
        shards[idx]["pred"] += task["predicted_seconds"]
    return shards

def _assign_tasks_to_shards_capped(tasks, tasks_per_shard=20):
    # Sort by predicted_seconds descending, then chunk sequentially to respect cap while balancing
    tasks_sorted = sorted(tasks, key=lambda t: t["predicted_seconds"], reverse=True)
    n_shards = math.ceil(len(tasks_sorted)/tasks_per_shard)
    shards = [{"tasks": [], "pred": 0.0} for _ in range(n_shards)]
    # Round-robin capped: assign largest tasks round-robin but stop when shard reaches cap
    # We'll do greedy: iterate tasks sorted, assign to shard with smallest pred that isn't full
    for task in tasks_sorted:
        candidates = [i for i, s in enumerate(shards) if len(s["tasks"]) < tasks_per_shard]
        if not candidates:
            # fallback (should not happen)
            candidates = list(range(n_shards))
        idx = min(candidates, key=lambda i: shards[i]["pred"])
        shards[idx]["tasks"].append(task)
        shards[idx]["pred"] += task["predicted_seconds"]
    return shards

def make_notebook(shard_id, n_shards, shard_tasks, design_hash, seed_root, filt, wall_budget=450, n_workers=2, fleet_name="phase6_highdim", results_dir="results/phase6_highdim", phase_label="thorough", phase_desc="thorough", phase_note="", tasks_per_shard=20, perms=199, cands_label="all"):
    # pred_min
    pred_min = sum(t["predicted_seconds"] for t in shard_tasks) / 60.0
    pred_min = pred_min / (1.6 if n_workers==2 else 1)
    d_grid = sorted(set(t["d"] for t in shard_tasks))
    n_grid = sorted(set(t["n0"] for t in shard_tasks))
    preproc_modes = sorted(set(t["preproc"] for t in shard_tasks))
    families = sorted(set(t["family"] for t in shard_tasks))
    header = HEADER_TEMPLATE.format(shard_id=shard_id, n_shards=n_shards, pred_min=pred_min, n_workers=n_workers, design_hash=design_hash, filt=filt, seed_root=seed_root, fleet_name=fleet_name, results_dir=results_dir, perms=perms, cands_label=cands_label, n_tasks=len(shard_tasks), tasks_per_shard=tasks_per_shard, phase_label=phase_label, phase_desc=phase_desc, phase_note=phase_note, d_grid=d_grid, n_grid=n_grid, preproc_modes=preproc_modes, families=families)
    tasks_json = json.dumps(shard_tasks, indent=2)
    cells = [
        _markdown(header),
        _code(f'import os\n\nSHARD_ID = {shard_id}\nN_SHARDS = {n_shards}\nFLEET_NAME = "{fleet_name}"\nRESULTS_DIR = "{results_dir}"\nSEED_ROOT = {seed_root}\nWALL_BUDGET_MIN = {wall_budget}\nN_WORKERS = {n_workers}\nos.environ["OMP_NUM_THREADS"] = "1"\nos.environ["OPENBLAS_NUM_THREADS"] = "1"\nos.environ["MKL_NUM_THREADS"] = "1"\nos.environ["NUMEXPR_NUM_THREADS"] = "1"\nprint({{"SHARD_ID": SHARD_ID, "N_SHARDS": N_SHARDS, "FLEET": FLEET_NAME, "WALL_BUDGET_MIN": WALL_BUDGET_MIN, "N_WORKERS": N_WORKERS}})\n'),
        _code(REPO_CELL_TEMPLATE.format(fleet_name=fleet_name, results_dir=results_dir)),
        _code(INSTALL_CELL),
        _code(CHECK_CELL),
        _code(TASK_CELL_TEMPLATE.format(shard_id=shard_id, n_shards=n_shards, seed_root=seed_root, wall_budget=wall_budget, n_workers=n_workers, tasks_json=tasks_json, fleet_name=fleet_name, results_dir=results_dir)),
    ]
    return {"nbformat":4,"nbformat_minor":0,"metadata":{"colab":{"provenance":[]},"kernelspec":{"name":"python3","display_name":"Python 3"}},"cells":cells}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--families", default=None, help="comma-separated")
    ap.add_argument("--replications", type=int, default=500)
    ap.add_argument("--permutations", type=int, default=199)
    ap.add_argument("--n-workers", type=int, default=2)
    ap.add_argument("--wall-budget", type=int, default=450)
    ap.add_argument("--preproc", default=None)
    ap.add_argument("--d-grid", default=None)
    ap.add_argument("--n-grid", default=None)
    ap.add_argument("--tasks-per-shard", type=int, default=20, help="max tasks per notebook (20 recommended to avoid timeout)")
    ap.add_argument("--candidates", default="all", help="all|primary or comma-separated")
    ap.add_argument("--phase-label", default=None, help="custom phase label for header")
    args = ap.parse_args()

    if args.families:
        families = [f.strip() for f in args.families.split(",") if f.strip()]
    else:
        families = FAMILIES_CORE

    if args.preproc:
        preproc_modes = [p.strip() for p in args.preproc.split(",")]
    else:
        preproc_modes = ["raw"]

    if args.candidates == "all":
        candidates = ALL_CANDIDATES
        cands_label = "all (14 incl SC-A)"
    elif args.candidates == "primary":
        candidates = PRIMARY_CANDIDATES
        cands_label = "primary (8)"
    else:
        candidates = [c.strip() for c in args.candidates.split(",")]
        cands_label = ",".join(candidates)

    d_grid = [int(x) for x in args.d_grid.split(",")] if args.d_grid else D_GRID
    n_grid = [int(x) for x in args.n_grid.split(",")] if args.n_grid else N_GRID

    # phase description
    if args.replications<=100 and args.permutations<=99:
        phase_label = args.phase_label or "pilot-explanatory"
        phase_desc = "pilot / explanatory"
        phase_note = "Smaller budget (100 reps, 99 perms, primary candidates) to check calibration (size in [0.03,0.08]), point-law power retention vs d, and PCA vs raw separation. If promising, run thorough fleet."
    else:
        phase_label = args.phase_label or "thorough"
        phase_desc = "thorough / high-precision"
        phase_note = "Full budget (500 reps, 199 perms, all candidates) for size/power gates and computation. Run only after pilot promising."

    tasks = build_tasks(families, n_grid, d_grid, preproc_modes, replications=args.replications, n_permutations=args.permutations, candidates=candidates)

    # shard assignment capped
    shards = _assign_tasks_to_shards_capped(tasks, tasks_per_shard=args.tasks_per_shard)
    n_shards = len(shards)

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    from experiments.phase6_highdim_pointlaw_tournament import DESIGN_HASH
    design_hash = DESIGN_HASH

    os.makedirs(args.outdir, exist_ok=True)
    for old in sorted(f for f in os.listdir(args.outdir) if f.endswith(".ipynb")):
        os.remove(os.path.join(args.outdir, old))
    if os.path.exists(os.path.join(args.outdir, "manifest.json")):
        os.remove(os.path.join(args.outdir, "manifest.json"))

    fleet_name = os.path.basename(args.outdir.rstrip("/"))
    results_dir = f"results/{fleet_name}"

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
        "candidates": candidates,
        "candidates_label": cands_label,
        "tasks_per_shard": args.tasks_per_shard,
        "n_shards": n_shards,
        "n_workers": args.n_workers,
        "wall_budget_min": args.wall_budget,
        "total_tasks": len(tasks),
        "phase_label": phase_label,
        "fleet_name": fleet_name,
        "results_dir": results_dir,
        "shards": []
    }
    for sid, shard in enumerate(shards):
        nb = make_notebook(sid, n_shards, shard["tasks"], design_hash, SEED_ROOT, PH_FILTRATION, wall_budget=args.wall_budget, n_workers=args.n_workers, fleet_name=fleet_name, results_dir=results_dir, phase_label=phase_label, phase_desc=phase_desc, phase_note=phase_note, tasks_per_shard=args.tasks_per_shard, perms=args.permutations, cands_label=cands_label)
        path = os.path.join(args.outdir, f"shard_{sid:02d}.ipynb")
        with open(path, "w") as fh:
            json.dump(nb, fh, indent=1)
        pred_min = shard["pred"]/60.0 / (1.6 if args.n_workers==2 else 1)
        manifest["shards"].append({"shard_id": sid, "notebook": path, "n_tasks": len(shard["tasks"]), "predicted_minutes": pred_min, "predicted_seconds": shard["pred"], "tasks": shard["tasks"]})
        print(f"wrote shard {sid:02d} with {len(shard['tasks'])} tasks pred {pred_min:.1f} min -> {path}")
    preds = [s["predicted_minutes"] for s in manifest["shards"]]
    print(f"total tasks {len(tasks)} across {n_shards} shards (cap {args.tasks_per_shard}): min {min(preds):.1f} max {max(preds):.1f} mean {sum(preds)/len(preds):.1f} total {sum(preds)/60:.1f}h")
    too_small = [s for s in manifest["shards"] if s["predicted_minutes"] < 45]
    too_large = [s for s in manifest["shards"] if s["predicted_minutes"] > 540]
    if too_small:
        print(f"NOTE: {len(too_small)} shards below 45 min (pilot may be short): ids {[s['shard_id'] for s in too_small]}")
    if too_large:
        print(f"WARNING: {len(too_large)} shards above 540 min: ids {[s['shard_id'] for s in too_large]}")
    manifest["predicted_shard_minutes"] = preds
    manifest["too_small"] = [s["shard_id"] for s in too_small]
    manifest["too_large"] = [s["shard_id"] for s in too_large]
    with open(os.path.join(args.outdir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(f"manifest -> {os.path.join(args.outdir, 'manifest.json')}")
    # Also write consolidation helper
    helper = os.path.join(args.outdir, "CONSOLIDATE.sh")
    with open(helper, "w") as fh:
        fh.write(f"""#!/bin/bash
# After downloading all /tmp/{fleet_name}_shardXX.zip files, unzip into results/{fleet_name}/
set -e
mkdir -p results/{fleet_name}
echo "Unzip all downloaded zips into results/{fleet_name}/"
# Example: if zips are in ~/Downloads/
# for z in ~/Downloads/{fleet_name}_shard*.zip; do unzip -o "$z" -d results/{fleet_name}/; done
# or if already in {args.outdir}/
for z in {args.outdir}/*.zip; do [ -f "$z" ] && unzip -o "$z" -d results/{fleet_name}/ || true; done
ls -lh results/{fleet_name}/ | head
echo "Then aggregate: python experiments/phase6_highdim_pointlaw_tournament.py --mode aggregate --input-dir results/{fleet_name}"
""")
    print(f"helper -> {helper}")

if __name__ == "__main__":
    main()
