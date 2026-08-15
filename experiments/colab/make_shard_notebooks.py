"""Generate the self-contained Colab notebooks that run the Phase 2 sweep.

Usage:  python experiments/colab/make_shard_notebooks.py
            [--n-notebooks 20] [--shards-per-notebook 5] [--reps-per-shard 10]
            [--outdir experiments/colab]

The sweep is split into *shards* of ``--reps-per-shard`` replications, which is
the atomic unit: shard ``i`` runs replications ``[i*R, (i+1)*R)`` of both parts
and writes ``results/shards/phase2_shard{i}.json``. Replication indices are
derived from the shard index alone and every RNG draw is keyed off them, so a
shard is reproducible from its number and shards computed anywhere are
interchangeable.

Each notebook owns a contiguous *block* of shards and downloads each shard file
as soon as it is written, which is what makes a dropped Colab session cheap: a
disconnect costs the shard in flight, not the notebook's whole block. The
defaults (20 notebooks x 5 shards x 10 replications) give the 1000 replications
per part that the Phase 2 gate is powered for, at roughly 1.5-2 h per notebook.

Each notebook: creates the package tree, embeds every ``tda2s`` module the
sweep imports and the sweep driver itself (``%%writefile`` cells), installs the
third-party dependencies (gudhi, ripser, persim, scikit-fda, tcda_uq from git),
checks the imports before committing to a long run, then runs its shards over
however many CPUs the Colab VM turns out to have.

Because the module sources are embedded at generation time, regenerate the
fleet after any edit to ``tda2s/`` or to the sweep driver.
"""

from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SWEEP_REL = "experiments/phase2_imbalance_sweep.py"

MODULES = [
    "tda2s/__init__.py",
    "tda2s/dgp/__init__.py",
    "tda2s/dgp/clouds.py",
    "tda2s/dgp/simulation.py",
    "tda2s/ph/__init__.py",
    "tda2s/vec/__init__.py",
    "tda2s/resample/__init__.py",
    "tda2s/resample/smoothing.py",
    "tda2s/benchmarks/__init__.py",
    "tda2s/benchmarks/_common.py",
    "tda2s/benchmarks/rt.py",
    "tda2s/benchmarks/mmd.py",
    "tda2s/benchmarks/han.py",
    "tda2s/benchmarks/strand.py",
    "tda2s/benchmarks/moon_lazar.py",
    "tda2s/benchmarks/frechet_anova.py",
    "tda2s/benchmarks/krebs_rademacher.py",
    "tda2s/adapters/__init__.py",
    "tda2s/adapters/tcda_uq.py",
    "tda2s/adapters/dr_test.py",
]

HEADER_TEMPLATE = """\
# Phase 2 imbalance sweep - notebook {nb:02d} of {n_notebooks}

Runs shards **{first}-{last}** ({n_reps} replications, indices
`{rep_lo}`-`{rep_hi}`) of both Phase 2 experiments:

* **Part A** - false positives under covariate shift: the causal nulls hold
  exactly while `L(D|A=1) != L(D|A=0)`, swept over six propensity strengths.
* **Part B** - Simpson masking: `L(D|A=1) = L(D|A=0)` exactly while
  `psi_d != 0`.

**Runtime -> Run all**, then leave the tab open. Expect roughly 1.5-2 h. Each
shard downloads a `phase2_shard<i>.json` the moment it finishes, so a dropped
session costs at most the shard in flight. Drop every downloaded file into
`results/shards/` in the repo and run

```
python experiments/phase2_imbalance_sweep.py --mode aggregate
```

once the whole fleet is in.
"""

# %%bash so a failed install is visible in the cell output rather than swallowed
INSTALL_CELL = (
    "%%bash\n"
    "pip install -q numpy scipy scikit-learn joblib matplotlib scikit-fda \\\n"
    "              gudhi ripser persim pot\n"
    "pip install -q git+https://github.com/hugogobato/tcda_uq.git\n"
    "echo '--- install done ---'\n"
)

# %%writefile does not create parent directories, so the tree comes first.
MKDIR_CELL_TEMPLATE = """\
import os

for d in {dirs!r}:
    os.makedirs(d, exist_ok=True)
print("package tree ready")
"""

CHECK_CELL = """\
import os, sys

sys.path.insert(0, "/content")
os.environ["PYTHONPATH"] = "/content"   # joblib workers inherit this

import gudhi, numpy, tcda_uq                      # noqa: F401
from experiments.phase2_imbalance_sweep import LAMBDAS, N_PER_GROUP, run_shard

print("imports OK |", N_PER_GROUP, "units per arm |", len(LAMBDAS), "lambdas |",
      os.cpu_count(), "CPUs")
"""

RUN_CELL_TEMPLATE = """\
SHARD_IDXS = {shard_idxs!r}
REPS_PER_SHARD = {reps_per_shard}

import os, time

# one worker per CPU, capped: the per-replication peak is ~1 GB and Colab's
# free VM has ~12 GB, so the cap is about leaving the VM responsive.
WORKERS = max(1, min(os.cpu_count() or 1, 4))
print(f"running shards {{SHARD_IDXS}} on {{WORKERS}} worker(s)")

t0 = time.time()
for shard in SHARD_IDXS:
    path = run_shard(shard, REPS_PER_SHARD, workers=WORKERS)
    try:
        from google.colab import files
        files.download(path)
        print("Downloaded:", path)
    except Exception as e:
        print("(Not on Colab / download skipped):", e)
    print(f"--- {{time.time() - t0:.0f}}s elapsed, "
          f"{{SHARD_IDXS.index(shard) + 1}}/{{len(SHARD_IDXS)}} shards done ---")

print("all shards done in", round(time.time() - t0), "s")
print("files also kept in /content/results/shards/ if a download was missed")
"""


def _code(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": source,
            "execution_count": None, "outputs": []}


def _markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def _dirs_for(paths):
    """Every directory the ``%%writefile`` cells will need, plus the outputs."""
    dirs = {os.path.dirname(p) for p in paths if os.path.dirname(p)}
    return sorted(dirs) + ["results/shards"]


def _notebook(nb, n_notebooks, shard_idxs, reps_per_shard) -> dict:
    n_reps = len(shard_idxs) * reps_per_shard
    cells = [_markdown(HEADER_TEMPLATE.format(
        nb=nb, n_notebooks=n_notebooks, first=shard_idxs[0],
        last=shard_idxs[-1], n_reps=n_reps,
        rep_lo=shard_idxs[0] * reps_per_shard,
        rep_hi=(shard_idxs[-1] + 1) * reps_per_shard - 1))]
    cells.append(_code(INSTALL_CELL))
    cells.append(_code(MKDIR_CELL_TEMPLATE.format(
        dirs=_dirs_for(MODULES + [SWEEP_REL]))))
    for rel in MODULES + [SWEEP_REL]:
        with open(os.path.join(REPO, rel)) as fh:
            code = fh.read()
        cells.append(_code(f"%%writefile {rel}\n{code}"))
    cells.append(_code(CHECK_CELL))
    cells.append(_code(RUN_CELL_TEMPLATE.format(shard_idxs=list(shard_idxs),
                                                reps_per_shard=reps_per_shard)))
    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {"colab": {"provenance": []},
                     "kernelspec": {"name": "python3",
                                    "display_name": "Python 3"}},
        "cells": cells,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-notebooks", type=int, default=20)
    ap.add_argument("--shards-per-notebook", type=int, default=5)
    ap.add_argument("--reps-per-shard", type=int, default=10)
    ap.add_argument("--outdir", default=HERE)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    # Clear the previous generation first. Regenerating with fewer notebooks
    # would otherwise leave the tail of the old fleet behind, and a stale
    # notebook is indistinguishable from a current one until it is run.
    for old in sorted(f for f in os.listdir(args.outdir)
                      if f.endswith(".ipynb")
                      and (f.startswith("phase2_nb_")
                           or f.startswith("phase2_shard_"))):
        os.remove(os.path.join(args.outdir, old))

    k = args.shards_per_notebook
    for nb in range(args.n_notebooks):
        shard_idxs = list(range(nb * k, (nb + 1) * k))
        path = os.path.join(args.outdir, f"phase2_nb_{nb:02d}.ipynb")
        with open(path, "w") as fh:
            json.dump(_notebook(nb, args.n_notebooks, shard_idxs,
                                args.reps_per_shard), fh, indent=1)

    n_shards = args.n_notebooks * k
    total = n_shards * args.reps_per_shard
    print(f"wrote {args.n_notebooks} notebooks covering {n_shards} shards "
          f"x {args.reps_per_shard} reps = {total} replications per part "
          f"-> {args.outdir}")
    print("then: put the downloaded phase2_shard*.json into results/shards/ "
          "and run 'python experiments/phase2_imbalance_sweep.py --mode aggregate'")


if __name__ == "__main__":
    main()
