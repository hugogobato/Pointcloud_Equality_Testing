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

**Current state (2026-08-14): shards 90-99 are already done** (run locally,
in `results/shards/`), so **notebooks 18 and 19 do not need running** --- the
outstanding fleet is **`phase2_nb_00` through `phase2_nb_17`**, covering
shards 0-89. Confirm with the coverage report below before starting.

## Run

1. Upload the notebooks to Colab (one per account/session is fine, and
   they are completely independent of each other). **Re-upload after
   2026-08-15** --- the fleet generated before that date crashes partway
   into the first shard; see "The multimethod pin" below.
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
