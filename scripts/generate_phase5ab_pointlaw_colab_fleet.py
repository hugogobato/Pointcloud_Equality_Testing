"""Generate the 40-notebook Colab fleet for the Phase 5AB point-law benchmark.

The fleet is built from the local frozen runner and its machine-readable cost
profile.  Each task contains one cell, a 25-replication range, and a fixed
candidate set.  Tasks are greedily balanced by predicted cost across exactly
40 notebooks.  CrossMatch is included only at pooled n <= 500 because the
profile shows that its exact blossom implementation is computationally
limited above that threshold; those rows remain secondary benchmark rows.

Usage::

    python scripts/generate_phase5ab_pointlaw_colab_fleet.py \
        --profile results/phase5ab_pointlaw_profile.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass

from experiments.phase5ab_pointlaw_tournament import (
    ALL_CANDIDATES,
    BENCHMARK_VERSION,
    CORE_FAMILIES,
    DESIGN_HASH,
    GATE_PERMUTATIONS,
    N_GRID,
    PRIMARY_M,
    PRIMARY_CANDIDATES,
    SECONDARY_CANDIDATES,
    SEED_ROOT,
    make_cells,
)

N_SHARDS = 40
WALL_BUDGET_MIN = 450
TASK_REPLICATIONS = 25
DEFAULT_PROFILE = "results/phase5ab_pointlaw_profile.json"
DEFAULT_NOTEBOOK_DIR = "notebooks/phase5ab_pointlaw_shards"
DEFAULT_MANIFEST = "experiments/colab/phase5ab_pointlaw_shards/manifest.json"
REPO_URL = "https://github.com/hugogobato/Pointcloud_Equality_Testing.git"


@dataclass(frozen=True)
class Task:
    task_id: str
    cell_id: str
    family: str
    n0: int
    n1: int
    m: int
    d: int
    rep_start: int
    replications: int
    candidates: tuple[str, ...]
    predicted_seconds: float


def _source_hash(repo_root: str) -> str:
    paths = (
        "tda2s/tests/point_law.py",
        "tda2s/tests/single_cloud.py",
        "experiments/phase5_single_cloud_tournament.py",
        "experiments/phase5ab_pointlaw_tournament.py",
    )
    digest = hashlib.sha256()
    for rel in paths:
        digest.update(rel.encode("utf-8"))
        with open(os.path.join(repo_root, rel), "rb") as handle:
            digest.update(handle.read())
    return digest.hexdigest()


def _profile_lookup(profile: dict) -> dict[tuple[str, int, int, str], dict]:
    return {
        (row["family"], int(row["n0"]), int(row["d"]), row["method"]): row
        for row in profile["observations"]
    }


def _candidate_set(n0: int) -> tuple[str, ...]:
    # CrossMatch is retained only where the profile shows it is feasible in a
    # useful shard.  At n>=500 per arm the runner records an explicit refusal.
    if n0 <= 250:
        return tuple(PRIMARY_CANDIDATES + SECONDARY_CANDIDATES)
    return tuple(c for c in ALL_CANDIDATES if c != "Rosenbaum-CrossMatch" and c not in ("HybridBlockMMD-a0.50", "SC-A-Block"))


def _method_seconds(
    lookup: dict[tuple[str, int, int, str], dict],
    *,
    family: str,
    n0: int,
    n1: int,
    d: int,
    method: str,
    n_permutations: int,
) -> float:
    # The implementation refuses CrossMatch above pooled n=500, so do not
    # assign runtime to those intentional refusal records.
    if method == "Rosenbaum-CrossMatch" and n0 + n1 > 500:
        return 0.01
    reference_n = n0 if n0 in (50, 250) else 250
    reference_d = d if (family, reference_n, d, method) in lookup else 2
    row = lookup.get((family, reference_n, reference_d, method))
    if row is None:
        row = lookup.get(("iid_null", reference_n, reference_d, method))
    if row is None:
        raise KeyError(f"profile has no observation for {family}, n={n0}, d={d}, {method}")
    base = float(row["per_call_seconds"])

    # Most methods recompute a dense pooled representation or evaluate it over
    # permutations.  These exponents are conservative planning estimates, not
    # inferential results.  SC-A is capped at 250 points per arm by the locked
    # runner, while CrossMatch is dominated by its one matching computation.
    if method == "SC-A":
        size_factor = 1.0
    elif method == "Rosenbaum-CrossMatch":
        size_factor = 1.0
    elif method.startswith("ClassifierTwoSampleTest"):
        size_factor = max(1.0, (n0 / reference_n) ** 1.5)
    elif method == "SlicedWasserstein":
        size_factor = max(1.0, (n0 / reference_n) ** 1.5)
    else:
        size_factor = max(1.0, (n0 / reference_n) ** 2)
    permutation_factor = {
        "Rosenbaum-CrossMatch": 1.10,
        "ClassifierTwoSampleTest-logistic": 2.0,
        "ClassifierTwoSampleTest-rf": 2.0,
    }.get(method, max(1.0, (n_permutations + 1) / (int(row["n_permutations"]) + 1)))
    dimension_factor = max(1.0, d / reference_d)
    return max(0.01, base * size_factor * permutation_factor * dimension_factor)


def _build_tasks(profile: dict, *, replications: int, n_permutations: int) -> list[Task]:
    lookup = _profile_lookup(profile)
    cells = make_cells(
        families=CORE_FAMILIES,
        n_grid=tuple(N_GRID) + (50,),
        m_values=(PRIMARY_M,),
        d_values=(2,),
    )
    tasks = []
    for cell in cells:
        candidates = _candidate_set(cell.n0)
        for rep_start in range(0, replications, TASK_REPLICATIONS):
            count = min(TASK_REPLICATIONS, replications - rep_start)
            per_rep = sum(
                _method_seconds(
                    lookup,
                    family=cell.family,
                    n0=cell.n0,
                    n1=cell.n1,
                    d=cell.d,
                    method=method,
                    n_permutations=n_permutations,
                )
                for method in candidates
            )
            tasks.append(Task(
                task_id=f"{cell.cell_id}_rep{rep_start}_{rep_start + count - 1}",
                cell_id=cell.cell_id,
                family=cell.family,
                n0=cell.n0,
                n1=cell.n1,
                m=cell.m,
                d=cell.d,
                rep_start=rep_start,
                replications=count,
                candidates=candidates,
                predicted_seconds=per_rep * count,
            ))
    return tasks


def _balance(tasks: list[Task]) -> list[list[Task]]:
    shards: list[list[Task]] = [[] for _ in range(N_SHARDS)]
    loads = [0.0] * N_SHARDS
    for task in sorted(tasks, key=lambda item: item.predicted_seconds, reverse=True):
        shard = min(range(N_SHARDS), key=loads.__getitem__)
        shards[shard].append(task)
        loads[shard] += task.predicted_seconds
    return shards


def _write_notebook(path: str, shard_id: int, tasks: list[Task], source_files: dict[str, str], source_hash: str) -> None:
    task_payload = [asdict(task) for task in tasks]
    for task in task_payload:
        task["candidates"] = list(task["candidates"])
    first_cell = f'''import os\n\n# Fleet controls are intentionally the first executable cell.\nSHARD_ID = {shard_id}\nN_SHARDS = {N_SHARDS}\nSEED_ROOT = {BENCHMARK_VERSION!r}\nWALL_BUDGET_MIN = {WALL_BUDGET_MIN}\nN_WORKERS = 2\nos.environ["OMP_NUM_THREADS"] = "1"\nos.environ["OPENBLAS_NUM_THREADS"] = "1"\nos.environ["MKL_NUM_THREADS"] = "1"\nos.environ["NUMEXPR_NUM_THREADS"] = "1"\nprint({{"SHARD_ID": SHARD_ID, "N_SHARDS": N_SHARDS, "WALL_BUDGET_MIN": WALL_BUDGET_MIN, "N_WORKERS": N_WORKERS}})\n'''
    clone_cell = f'''import os\nimport shutil\n\nREPO_DIR = "/content/Pointcloud_Equality_Testing"\nif os.path.isdir(REPO_DIR):\n    shutil.rmtree(REPO_DIR)\nos.system("git clone --depth 1 {REPO_URL} " + REPO_DIR)\nos.chdir(REPO_DIR)\nos.system("python -m pip install -q -e .")\nos.system("python -m pip install -q networkx==3.4.2")\nprint("Repository ready:", REPO_DIR)\n'''
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": f"# Phase 5AB point-law shard {shard_id:02d}\n\nPredicted runtime: {sum(t.predicted_seconds for t in tasks) / 60:.1f} minutes. Each task checkpoints after {TASK_REPLICATIONS} replications."},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": first_cell},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": clone_cell},
    ]
    for rel, content in source_files.items():
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": f"%%writefile {rel}\n{content}",
        })
    run_cell = f'''import json\nimport os\nimport time\nfrom pathlib import Path\n\nimport pandas as pd\n\nfrom experiments.phase5ab_pointlaw_tournament import (\n    DESIGN_HASH, GATE_PERMUTATIONS, Cell, run_shard,\n)\n\nEXPECTED_DESIGN_HASH = {DESIGN_HASH!r}\nSOURCE_HASH = {source_hash!r}\nTASKS = {task_payload!r}\nSHARD_OUT = os.path.join(REPO_DIR, "phase5ab_pointlaw_shard_{{:02d}}.parquet".format(SHARD_ID))\nMANIFEST_OUT = os.path.join(REPO_DIR, "phase5ab_pointlaw_shard_{{:02d}}_manifest.json".format(SHARD_ID))\nCACHE_DIR = os.path.join(REPO_DIR, "phase5ab_pointlaw_cache_{{:02d}}".format(SHARD_ID))\nos.makedirs(CACHE_DIR, exist_ok=True)\n\nif os.path.exists(SHARD_OUT):\n    accumulated = pd.read_parquet(SHARD_OUT)\nelse:\n    accumulated = pd.DataFrame()\ncompleted = set()\nif not accumulated.empty:\n    completed = set(zip(accumulated["cell_id"], accumulated["method"], accumulated["replication"]))\ncompleted_tasks = []\nstarted = time.time()\nfor task in TASKS:\n    expected_cell = task["cell_id"]\n    expected_reps = range(task["rep_start"], task["rep_start"] + task["replications"])\n    if all((expected_cell, method, rep) in completed for method in task["candidates"] for rep in expected_reps):\n        completed_tasks.append(task["task_id"])\n        continue\n    parts = task["cell_id"].rsplit("_n", 1)\n    family = parts[0]\n    suffix = "_n" + parts[1]\n    n0_text, suffix = suffix.split("_n1", 1)\n    n1_text, suffix = suffix.split("_m", 1)\n    m_text, d_text = suffix.split("_d", 1)\n    cell = Cell(family, int(n0_text), int(n1_text), int(m_text), int(d_text), "unknown", family)\n    task_file = os.path.join(REPO_DIR, "phase5ab_task_{{}}_{{}}.parquet".format(SHARD_ID, task["task_id"]))\n    run_shard(cell, rep_start=task["rep_start"], replications=task["replications"], n_permutations=GATE_PERMUTATIONS, candidates=task["candidates"], workers=N_WORKERS, cache_dir=CACHE_DIR, output=task_file)\n    part = pd.read_parquet(task_file)\n    accumulated = pd.concat([accumulated, part], ignore_index=True)\n    accumulated = accumulated.drop_duplicates(["design_hash", "cell_id", "method", "replication"], keep="last")\n    accumulated = accumulated.sort_values(["cell_id", "method", "replication"]).reset_index(drop=True)\n    accumulated.to_parquet(SHARD_OUT, index=False)\n    completed = set(zip(accumulated["cell_id"], accumulated["method"], accumulated["replication"]))\n    completed_tasks.append(task["task_id"])\n    with open(MANIFEST_OUT, "w", encoding="utf-8") as handle:\n        json.dump({{"benchmark_version": {BENCHMARK_VERSION!r}, "design_hash": DESIGN_HASH, "expected_design_hash": EXPECTED_DESIGN_HASH, "source_hash": SOURCE_HASH, "shard_id": SHARD_ID, "n_shards": N_SHARDS, "completed_tasks": completed_tasks, "elapsed_seconds": time.time() - started}}, handle, indent=2, sort_keys=True)\n    print("Checkpoint:", task["task_id"], "rows:", len(accumulated), "elapsed_min:", round((time.time() - started) / 60, 1))\n\nassert DESIGN_HASH == EXPECTED_DESIGN_HASH\nprint("Shard complete:", SHARD_OUT, "rows:", len(accumulated), "elapsed_min:", round((time.time() - started) / 60, 1))\n\noutput_file = SHARD_OUT\ntry:\n    from google.colab import files\n    files.download(output_file)\n    print("Downloaded:", output_file)\nexcept Exception as exc:\n    print("(Not on Colab / download skipped):", exc)\n'''
    # The cell-id parser is written as a compact escaped string above.  Add
    # the family/n0 split after the legacy suffix parsing so names containing
    # underscores (for example same_support_density) remain unambiguous.
    run_cell = run_cell.replace(
        '    m_text, d_text = suffix.split("_d", 1)\n',
        '    m_text, d_text = suffix.split("_d", 1)\n    family, n0_text = family.rsplit("_n", 1)\n',
    )
    run_cell = run_cell.replace(
        '    DESIGN_HASH, GATE_PERMUTATIONS, Cell, run_shard,\n',
        '    DESIGN_HASH, FAMILY_DESCRIPTION_EXT, FAMILY_ROLE_EXT, GATE_PERMUTATIONS, Cell, run_shard,\n',
    )
    run_cell = run_cell.replace(
        '    cell = Cell(family, int(n0_text), int(n1_text), int(m_text), int(d_text), "unknown", family)\n',
        '    cell = Cell(family, int(n0_text), int(n1_text), int(m_text), int(d_text), FAMILY_ROLE_EXT.get(family, "unknown"), FAMILY_DESCRIPTION_EXT.get(family, family))\n',
    )
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": run_cell})
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}},
        "cells": cells,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(notebook, handle, indent=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--repo-root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    parser.add_argument("--notebook-dir", default=DEFAULT_NOTEBOOK_DIR)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--replications", type=int, default=500)
    parser.add_argument("--permutations", type=int, default=GATE_PERMUTATIONS)
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    with open(os.path.join(repo_root, args.profile), encoding="utf-8") as handle:
        profile = json.load(handle)
    if profile.get("design_hash") != DESIGN_HASH:
        raise SystemExit(f"profile design hash {profile.get('design_hash')} != runner {DESIGN_HASH}")
    tasks = _build_tasks(profile, replications=args.replications, n_permutations=args.permutations)
    shards = _balance(tasks)
    loads = [sum(task.predicted_seconds for task in shard) for shard in shards]
    if max(loads) / 60 > WALL_BUDGET_MIN:
        raise SystemExit(f"predicted shard exceeds {WALL_BUDGET_MIN} minutes: {max(loads) / 60:.1f}")

    source_files = {
        rel: open(os.path.join(repo_root, rel), encoding="utf-8").read()
        for rel in (
            "tda2s/tests/point_law.py",
            "tda2s/tests/single_cloud.py",
            "experiments/phase5_single_cloud_tournament.py",
            "experiments/phase5ab_pointlaw_tournament.py",
        )
    }
    source_hash = _source_hash(repo_root)
    notebook_dir = os.path.join(repo_root, args.notebook_dir)
    manifest_path = os.path.join(repo_root, args.manifest)
    os.makedirs(notebook_dir, exist_ok=True)
    for old in os.listdir(notebook_dir):
        if old.startswith("shard_") and old.endswith(".ipynb"):
            os.remove(os.path.join(notebook_dir, old))
    for shard_id, shard_tasks in enumerate(shards):
        _write_notebook(os.path.join(notebook_dir, f"shard_{shard_id:02d}.ipynb"), shard_id, shard_tasks, source_files, source_hash)

    manifest = {
        "benchmark_version": BENCHMARK_VERSION,
        "design_hash": DESIGN_HASH,
        "source_hash": source_hash,
        "repo_url": REPO_URL,
        "n_shards": N_SHARDS,
        "replications_per_cell": args.replications,
        "permutations": args.permutations,
        "task_replications": TASK_REPLICATIONS,
        "wall_budget_min": WALL_BUDGET_MIN,
        "n_workers": 2,
        "seed_root": SEED_ROOT,
        "cells": sorted({task.cell_id for task in tasks}),
        "candidate_policy": "PRIMARY+SECONDARY for n<=250; CrossMatch excluded above pooled n=500; retained hybrids excluded from fleet",
        "predicted_shard_minutes": [load / 60 for load in loads],
        "predicted_total_hours": sum(loads) / 3600,
        "present_shards": [],
        "incomplete_shards": list(range(N_SHARDS)),
        "shards": [
            {
                "shard_id": shard_id,
                "notebook": os.path.join(args.notebook_dir, f"shard_{shard_id:02d}.ipynb"),
                "predicted_minutes": loads[shard_id] / 60,
                "tasks": [asdict(task) for task in shard_tasks],
            }
            for shard_id, shard_tasks in enumerate(shards)
        ],
    }
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(json.dumps({"notebooks": N_SHARDS, "tasks": len(tasks), "min_minutes": min(loads) / 60, "max_minutes": max(loads) / 60, "total_hours": sum(loads) / 3600, "manifest": manifest_path}, indent=2))


if __name__ == "__main__":
    main()
