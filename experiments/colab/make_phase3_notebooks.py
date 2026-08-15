"""Generate self-contained Colab notebooks for the Phase 3 shards.

The notebooks embed the exact local source used by the experiment driver, so a
Colab session needs no repository mount. The default fleet has ten oracle
notebooks, each carrying five ten-replication shards, which totals 500
replications at every configured sample-size/regime/alternative cell. The
``clouds`` fleet is intentionally smaller because persistent homology is the
expensive part.
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
]

INSTALL = """%%bash
set -e
pip install -q numpy scipy scikit-learn matplotlib scikit-fda gudhi ripser persim
pip install -q git+https://github.com/hugogobato/tcda_uq.git
# Colab may start with multimethod 2.1, which conflicts with scikit-fda's
# dispatch metaclass. Pin it last, as in the Phase 2 fleet.
pip install -q 'multimethod==2.0.2'
echo '--- Phase 3 dependencies installed ---'
"""


def _code(source):
    return {"cell_type": "code", "metadata": {}, "source": source,
            "execution_count": None, "outputs": []}


def _markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def _write_tree():
    dirs = sorted({os.path.dirname(path) for path in MODULES if os.path.dirname(path)})
    dirs += ["results/phase3_shards"]
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
from experiments.phase3_dr_calibration import run_shard
from tda2s.tests.dr_outcome import fit_dr
from tcda_uq.datasets import TriOracleSimulation

sim = TriOracleSimulation(n_cov=3, n_hom_dim=2, resolution=16, n_basis=5, seed=0)
s = sim.sample(24, rng=0)
fit = fit_dr(s.observed, s.tseq, n_basis=5, n_folds=2, random_state=0)
assert fit.estimate.shape == (2, 16)
print('Phase 3 imports and cached-fit smoke passed')
"""


def _run_cell(shards, reps_per_shard, design, n_calibration, cloud_n):
    return """import time
SHARDS = %r
REPS_PER_SHARD = %d
DESIGN = %r
N_CALIBRATION = %d
CLOUD_N = %d
from experiments.phase3_dr_calibration import run_shard

t0 = time.time()
for shard in SHARDS:
    path = run_shard(shard, REPS_PER_SHARD, design=DESIGN,
                     n_calibration=N_CALIBRATION, cloud_n=CLOUD_N)
    try:
        from google.colab import files
        files.download(path)
        print('Downloaded:', path)
    except Exception as exc:
        print('(Not on Colab / download skipped):', exc)
    print('completed shard', shard, 'elapsed seconds', round(time.time() - t0))
print('all shards done; files remain under /content/results/phase3_shards/')
""" % (list(shards), reps_per_shard, design, n_calibration, cloud_n)


def build_notebook(nb, n_notebooks, shards, reps_per_shard, design,
                   n_calibration, cloud_n):
    if design == "oracle":
        description = (
            "The oracle design is the long Phase 3 table: each shard covers "
            "all n in {50,100,200,500}, three propensity regimes, and null "
            "and alternative outcomes."
        )
    else:
        description = (
            "The cloud design computes alpha-complex persistence once per "
            "replication, then uses the cached DR fit for both null calibrations."
        )
    cells = [_markdown(
        "# Phase 3 DR calibration, notebook %02d of %02d\n\n%s\n\n"
        "Runs shards **%d-%d**, with %d total replications. Each completed "
        "shard is downloaded immediately, so a disconnected session loses at "
        "most the shard in flight.\n\n"
        "The stratified-permutation path freezes the cross-fitted nuisances and "
        "evaluates all label draws by matrix algebra. It does not rerun PH, "
        "cross-fitting, or nuisance regression inside the permutation loop."
        % (nb, n_notebooks, description, shards[0], shards[-1],
           len(shards) * reps_per_shard))]
    cells += [_code(INSTALL), _code(_write_tree())]
    cells += _source_cells()
    cells += [_code(_check_cell()), _code(_run_cell(
        shards, reps_per_shard, design, n_calibration, cloud_n))]
    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {"colab": {"provenance": []},
                     "kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "cells": cells,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", choices=("oracle", "clouds", "learners", "stress"), default="oracle")
    parser.add_argument("--n-notebooks", type=int, default=None)
    parser.add_argument("--shards-per-notebook", type=int, default=5)
    parser.add_argument("--reps-per-shard", type=int, default=10)
    parser.add_argument("--n-calibration", type=int, default=399)
    parser.add_argument("--cloud-n", type=int, default=100)
    parser.add_argument("--outdir", default=HERE)
    args = parser.parse_args()
    if args.n_notebooks is not None:
        n_notebooks = args.n_notebooks
    else:
        n_notebooks = {"oracle": 10, "clouds": 4, "learners": 2, "stress": 2}[args.design]
    prefix = f"phase3_{args.design}_nb_"
    for old in os.listdir(args.outdir):
        if old.startswith(prefix) and old.endswith(".ipynb"):
            os.remove(os.path.join(args.outdir, old))
    for nb in range(n_notebooks):
        first = nb * args.shards_per_notebook
        shards = list(range(first, first + args.shards_per_notebook))
        out = os.path.join(args.outdir, f"{prefix}{nb:02d}.ipynb")
        with open(out, "w") as fh:
            json.dump(build_notebook(
                nb, n_notebooks, shards, args.reps_per_shard, args.design,
                args.n_calibration, args.cloud_n), fh, indent=1)
        print(out)


if __name__ == "__main__":
    main()
