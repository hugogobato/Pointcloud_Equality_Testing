# Phase 2 Colab fleet

`phase2_nb_00.ipynb` ... `phase2_nb_19.ipynb` are self-contained notebooks.
Each one installs the dependencies (gudhi, ripser, persim, scikit-fda,
`tcda_uq` from git), embeds this repo's `tda2s` sources and the sweep
driver, runs five shards of ten replications each, and downloads one
`phase2_shard<i>.json` per shard. 20 notebooks x 5 shards x 10
replications = **1000 replications per part**, which is what gate 2.5's
size band is powered for (at 1000 replications a correctly sized test
clears `[0.03, 0.08]` with probability ~0.999 per lambda; at 200 it would
fail somewhere in the sweep about a quarter of the time on noise alone).

Both experiments run in every notebook: Part A is the covariate-shift
false-positive sweep over six propensity strengths, Part B the Simpson
masking experiment.

**Current state (2026-08-15): the fleet has run and Phase 2 is closed.** All
100 shards are in `results/shards/`, gate 2.5 fired from them, and Figure 1
is `results/phase2_figure1.{png,json}`. Nothing here needs running again
unless the sweep's configuration changes; the instructions below are kept
for that case and for Phase 3, which re-fires the size criterion against
these same shards.

## Run

1. Upload the notebooks to Colab (one per account/session is fine, and
   they are completely independent of each other). Regenerate first if
   `tda2s/` or the sweep driver has changed since the notebooks were
   written; notebooks generated before 2026-08-15 also crash partway into
   the first shard (see "The multimethod pin" below).
2. **Runtime -> Run all.** Expect roughly 1.5-2 h per notebook. The
   import cell finishes with `DR smoke p = ...`; if that line prints, the
   environment is sound and the run cell will not hit an import error.
3. Each shard downloads the moment it finishes, so a dropped session
   costs at most the shard in flight rather than the notebook's block.
   Anything already finished also stays in the VM's
   `/content/results/shards/` until the session is recycled.
4. Drop every downloaded `phase2_shard<i>.json` into `results/shards/`
   in this repo. Order does not matter and duplicates are harmless.

Notebook `nb` covers shards `5*nb .. 5*nb+4`, so before running the whole
fleet, check what is already on disk — `--mode aggregate` prints the
missing shard indices, and any shard already present does not need its
notebook run again:

```bash
python experiments/phase2_imbalance_sweep.py --mode aggregate
# [phase2] part A: 100/1000 replications; 90 shard(s) missing at 10 reps/shard: 0, 1, 2, ...
```

Shard `i` is exactly replications `[10i, 10i+10)` and every RNG draw is
keyed off the replication index, so a shard is reproducible from its
number alone and it does not matter where it ran. To fill a gap in the
fleet locally:

```bash
python experiments/phase2_imbalance_sweep.py --mode local --shards 90-99 \
       --reps-per-shard 10 --workers 4 --skip-existing
```

## Aggregate

```bash
python experiments/phase2_imbalance_sweep.py --mode aggregate
```

Merges every `results/shards/phase2_shard*.json` (deduplicated by
replication index, refusing to pool shards whose sampling config
disagrees), writes `results/phase2_figure1.png` +
`results/phase2_figure1.json`, and prints the Phase 2 GATE summary. It is
safe to run on a partial fleet to watch the numbers settle; just do not
read the verdict off an under-powered subset.

## The multimethod pin

`scikit-fda` registers multiple-dispatch hints on `ABCMeta`-based classes.
`multimethod` 2.1 builds its `subtype` metaclass off `type` rather than
`ABCMeta`, so that registration raises

```
TypeError: metaclass conflict: the metaclass of a derived class must be
a (non-strict) subclass of the metaclasses of all its bases
```

the first time `skfda.ml.regression` is imported. Nothing in the
dependency graph constrains the version, so Colab's preinstalled 2.1
survived every install line; and because `tcda_uq` imports `skfda`
lazily from inside `cross_fit`, the crash landed on the first DR
replication (minutes into a shard) rather than at import. The install
cell now pins `multimethod==2.0.2` last, and the import cell exercises
the full chain plus a few-second DR smoke test, so a broken environment
fails in the first minute.

Verified on Python 3.12 (Colab's version) in a venv built from the
install cell starting at multimethod 2.1: the chain imports, and
replication 900 of both parts reproduces `results/shards/phase2_shard90.json`
exactly, so Colab shards remain interchangeable with the locally
computed 90-99.

## Regenerate

The notebooks embed the module sources at generation time, so regenerate
after any edit to `tda2s/` or to the sweep driver:

```bash
python experiments/colab/make_shard_notebooks.py \
       --n-notebooks 20 --shards-per-notebook 5 --reps-per-shard 10
```

Fewer, longer notebooks (`--n-notebooks 10 --shards-per-notebook 10`)
means less uploading and a longer session to keep alive; more, shorter
ones is the reverse. The replication total is the product of the three
numbers and should stay at 1000.

## Phase 3 DR calibration fleet

`phase3_oracle_nb_00.ipynb` through `phase3_oracle_nb_09.ipynb` are independent,
self-contained notebooks for the Phase 3 functional tables. The default fleet
runs five ten-replication shards per notebook, giving 500 replications at
every `n in {50, 100, 200, 500}`, propensity regime, and null or alternative
cell. Each shard downloads `phase3_oracle_shard<i>.json` immediately on
completion.

The oracle fleet isolates DR calibration from persistent-homology runtime. The
four `phase3_clouds_nb_*.ipynb` notebooks run the same cached calibration on
the point-cloud DGP for a smaller topological confirmation. The two learner
and two stress notebooks cover the propensity-learner sweep and the
double-robustness misspecification configurations. All fleets freeze the
fitted cross-fit nuisances, so permutation draws do not refit `cross_fit` or
recompute diagrams.

Regenerate the fleets after changing the Phase 3 module or driver:

```bash
python experiments/colab/make_phase3_notebooks.py --design oracle
python experiments/colab/make_phase3_notebooks.py --design clouds
python experiments/colab/make_phase3_notebooks.py --design learners
python experiments/colab/make_phase3_notebooks.py --design stress
```

Run the notebooks in Colab. The download cell includes a safe
`google.colab.files.download` fallback, so a non-Colab run keeps output in
`results/phase3_shards/`. Aggregate downloaded shards with:

```bash
python experiments/phase3_dr_calibration.py --mode aggregate --design oracle
python experiments/phase3_dr_calibration.py --mode aggregate --design clouds
python experiments/phase3_dr_calibration.py --mode aggregate --design learners
python experiments/phase3_dr_calibration.py --mode aggregate --design stress
```

The oracle fleet stays within the memory limit by using one bounded-memory job
per notebook and vectorized permutation batches instead of storing all draws
at once.

## Phase 3.5 multiplicity comparator fleet

`make_phase3_5_notebooks.py` writes eight self-contained notebooks for the
Phase 3.5 benchmark of the four degree-multiplicity procedures: Bonferroni,
the Phase 3 unstudentized shared max-statistic, and the studentized
empirical-null comparator adapted from Vejdemo-Johansson and Mukherjee under
both its pooled and its source standardization. Every replication produces one
cross-fitted fit and one shared null matrix per mechanism, and all four
procedures are read off that single matrix, so the differences between them
carry no Monte Carlo noise from the calibration.

Four `phase3_5_fwer_nb_*.ipynb` notebooks carry the 500-replication
pre-registered decision (FWER in [0.03, 0.08] at alpha = 0.05). One
`phase3_5_power_nb_00.ipynb` measures validity and power under a deliberate
per-degree scale imbalance, `phase3_5_degrees3_nb_00.ipynb` repeats it with the
three-degree family, and `phase3_5_learners_nb_00.ipynb` and
`phase3_5_stress_nb_00.ipynb` cover the propensity-learner and
double-robustness nulls. Each notebook is roughly 25 minutes at measured local
rates, and each shard is 25 replications, so a dropped session costs one
checkpoint.

```bash
python experiments/colab/make_phase3_5_notebooks.py            # all designs
python experiments/phase3_5_vjm.py --mode aggregate --design fwer
python experiments/phase3_5_vjm.py --mode aggregate --design power
python experiments/phase3_5_vjm.py --mode aggregate --design degrees3
python experiments/phase3_5_vjm.py --mode aggregate --design learners
python experiments/phase3_5_vjm.py --mode aggregate --design stress
```

Downloaded shards belong in `results/phase3_5_shards/`. The `fwer` design
reuses the Phase 3 oracle null design and every Phase 3 seed, so its shared-max
column reproduces the published Phase 3 p-values exactly; verify with

```bash
python experiments/phase3_5_vjm.py --mode check-phase3
```

which recomputes 36 null cells from the downloaded Phase 3 shards and fails on
any digit of disagreement. The audit that licenses the comparator, and the
list of what does not transfer from the source, is `docs/phase3_5_vjm_mapping.md`.
