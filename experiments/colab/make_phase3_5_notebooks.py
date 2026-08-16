"""Generate self-contained Colab notebooks for the Phase 3.5 fleet.

The notebooks embed the exact local source, so a Colab session needs no
repository mount.  Sizing is set from a measured local replication cost of
about 12 s (``fwer``), 4 s (``power``), 5 s (``learners``) and 6 s (``stress``)
per replication, which puts every notebook comfortably inside a ~1 h session:

    fwer      4 notebooks x 5 shards x 25 reps = 500 replications
    power     1 notebook  x 8 shards x 25 reps = 200 replications
    degrees3  1 notebook  x 8 shards x 25 reps = 200 replications
    learners  1 notebook  x 4 shards x 25 reps = 100 replications
    stress    1 notebook  x 4 shards x 25 reps = 100 replications

Shards are coarse on purpose: each is one downloadable checkpoint, so a
dropped session costs only the shard in flight without producing dozens of
files to reassemble.
"""

from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

MODULES = [
    "tda2s/__init__.py",
    "tda2s/adapters/__init__.py",
    "tda2s/adapters/tcda_uq.py",
    "tda2s/resample/__init__.py",
    "tda2s/resample/smoothing.py",
    "tda2s/tests/__init__.py",
    "tda2s/tests/dr_outcome.py",
    "tda2s/dgp/__init__.py",
    "tda2s/dgp/clouds.py",
    "tda2s/dgp/simulation.py",
    "tda2s/ph/__init__.py",
    "tda2s/vec/__init__.py",
    "experiments/phase3_dr_calibration.py",
    "experiments/phase3_5_vjm.py",
]

INSTALL = """%%bash
set -e
pip install -q numpy scipy scikit-learn matplotlib scikit-fda gudhi ripser persim
pip install -q git+https://github.com/hugogobato/tcda_uq.git
# Colab may start with multimethod 2.1, which conflicts with scikit-fda's
# dispatch metaclass. Pin it last, as in the Phase 2 and Phase 3 fleets.
pip install -q 'multimethod==2.0.2'
echo '--- Phase 3.5 dependencies installed ---'
"""

# reps per shard, shards per notebook, number of notebooks
FLEET = {
    "fwer": (25, 5, 4),
    "power": (25, 8, 1),
    "degrees3": (25, 8, 1),
    "learners": (25, 4, 1),
    "stress": (25, 4, 1),
}

DESCRIPTIONS = {
    "fwer": ("The primary task 3.5.4 decision: the global null at "
             "n in {50,100,200,500} and three propensity regimes. Every "
             "replication reads Bonferroni, the shared max-statistic and both "
             "standardization conventions of the studentized comparator off "
             "one null matrix per mechanism."),
    "power": ("Validity and power under a deliberate per-degree scale "
              "imbalance. Degree 0's silhouettes are multiplied by 8, which "
              "leaves the estimand untouched but makes the degrees "
              "incomparable, and the alternative lives in degree 1 only."),
    "degrees3": ("The same validity-and-power design widened from two "
                 "homology degrees to the plan's d in {0,1,2}, which is where "
                 "the Bonferroni penalty and the cross-degree dependence the "
                 "comparator preserves both start to bite."),
    "learners": ("The task 3.3 propensity learner grid, null only, so the "
                 "comparator is checked outside the correctly specified "
                 "parametric regime."),
    "stress": ("The task 3.4 double-robustness misspecification grid, null "
               "only."),
}


def _code(source):
    return {"cell_type": "code", "metadata": {}, "source": source,
            "execution_count": None, "outputs": []}


def _markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def _write_tree():
    dirs = sorted({os.path.dirname(path) for path in MODULES if os.path.dirname(path)})
    dirs += ["results/phase3_5_shards"]
    return "import os\nfor d in %r:\n    os.makedirs(d, exist_ok=True)\nprint('source tree ready')\n" % dirs


def _source_cells():
    cells = []
    for rel in MODULES:
        with open(os.path.join(REPO, rel)) as fh:
            code = fh.read()
        cells.append(_code("%%writefile %s\n%s" % (rel, code)))
    return cells


def _check_cell():
    return """import os, sys
sys.path.insert(0, '/content')
os.environ['MPLCONFIGDIR'] = '/tmp/mplconfig'
os.makedirs('/tmp/mplconfig', exist_ok=True)
import numpy as np
from experiments.phase3_5_vjm import _one_cell, _vjm_sample

row = _one_cell(_vjm_sample(40, 0, 1.0, alternative=False), 0, 40, 1.0,
                alternative=False, n_calibration=39)
assert 0.0 < row['permutation_vjm_pooled_p'] <= 1.0
assert 0.0 < row['multiplier_shared_max_p'] <= 1.0
print('Phase 3.5 imports and comparator smoke passed')
"""


def _run_cell(shards, reps_per_shard, design, n_calibration):
    return """import time
SHARDS = %r
REPS_PER_SHARD = %d
DESIGN = %r
N_CALIBRATION = %d
from experiments.phase3_5_vjm import run_shard

t0 = time.time()
for shard in SHARDS:
    path = run_shard(shard, REPS_PER_SHARD, design=DESIGN,
                     n_calibration=N_CALIBRATION)
    try:
        from google.colab import files
        files.download(path)
        print('Downloaded:', path)
    except Exception as exc:
        print('(Not on Colab / download skipped):', exc)
    print('completed shard', shard, 'elapsed seconds', round(time.time() - t0))
print('all shards done; files remain under /content/results/phase3_5_shards/')
""" % (list(shards), reps_per_shard, design, n_calibration)


def build_notebook(nb, n_notebooks, shards, reps_per_shard, design, n_calibration):
    cells = [_markdown(
        "# Phase 3.5 multiplicity comparator, %s notebook %02d of %02d\n\n%s\n\n"
        "Runs shards **%d-%d** (%d replications). Each completed shard is "
        "downloaded immediately, so a disconnected session loses at most the "
        "shard in flight.\n\n"
        "Nothing here refits persistent homology, cross-fitting or a nuisance "
        "regression inside a calibration loop: one cross-fitted fit per cell "
        "produces one shared null matrix, and all four multiplicity "
        "procedures are read off it."
        % (design, nb, n_notebooks, DESCRIPTIONS[design], shards[0], shards[-1],
           len(shards) * reps_per_shard))]
    cells += [_code(INSTALL), _code(_write_tree())]
    cells += _source_cells()
    cells += [_code(_check_cell()),
              _code(_run_cell(shards, reps_per_shard, design, n_calibration))]
    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {"colab": {"provenance": []},
                     "kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "cells": cells,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", choices=tuple(FLEET) + ("all",), default="all")
    parser.add_argument("--n-calibration", type=int, default=399)
    parser.add_argument("--outdir", default=HERE)
    args = parser.parse_args()
    designs = tuple(FLEET) if args.design == "all" else (args.design,)
    for design in designs:
        reps_per_shard, shards_per_nb, n_notebooks = FLEET[design]
        prefix = f"phase3_5_{design}_nb_"
        for old in os.listdir(args.outdir):
            if old.startswith(prefix) and old.endswith(".ipynb"):
                os.remove(os.path.join(args.outdir, old))
        for nb in range(n_notebooks):
            first = nb * shards_per_nb
            shards = list(range(first, first + shards_per_nb))
            out = os.path.join(args.outdir, f"{prefix}{nb:02d}.ipynb")
            with open(out, "w") as fh:
                json.dump(build_notebook(nb, n_notebooks, shards, reps_per_shard,
                                         design, args.n_calibration), fh, indent=1)
            print(out)


if __name__ == "__main__":
    main()
