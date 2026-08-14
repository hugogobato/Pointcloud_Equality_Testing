# Reproduction notebooks

Full-budget reproductions of the competitors' published figures, sized for a
Colab (or any Linux) box with many cores. The pytest versions in
`tests/test_published_reproductions.py` run the same designs at 150
replications so they stay usable locally; these run the papers' 500.

| notebook | reproduces | cost at 500 reps |
|---|---|---|
| `01_moon_lazar_figure5.ipynb` | Moon & Lazar (2023) Fig. 5a and 5b, all four noise levels, plus the "PD" (Robinson–Turner) curve of Fig. 5b | 8.9 core-min (0.14 core-h) |
| `02_dubey_muller_figure1.ipynb` | Dubey & Müller (2019) Fig. 1, both panels, as dense power curves | 6.4 core-min (0.11 core-h) |

Costs are measured single-core on an i9-13900H, so divide by `N_JOBS` and allow
1.5–2× for a slower Colab CPU. Both are small: at two workers each notebook is
under ten minutes of compute, and the `pip install` of the persistent-homology
stack is a meaningful fraction of the wall clock. Peak RSS is a few hundred MB.

Completed runs are in `results/` (`*_reps500.json` plus a PNG each); the
numbers are tabulated in `docs/phase0_completion.md` §0.5.

## Running one

Open the notebook in Colab and run the cells top to bottom. The first cell
pins BLAS to one thread per process (this must happen **before** numpy is
imported, or every worker spins up its own thread pool and oversubscribes the
machine); the second clones this repo and installs the persistent-homology
stack.

The install deliberately uses `--no-deps` and keeps Colab's own numpy and scipy
rather than the pins in `pyproject.toml`. Installing the pinned numpy would
force a runtime restart, and nothing in these designs depends on that exact
version. If you want the pinned environment, use `pip install -e .` instead and
restart the runtime when Colab asks.

## Sharding

Replication `r` is seeded from `(base_seed, r)` alone, so a replication is the
same experiment however the work is divided. Shards are concatenated, not
averaged, and `N_JOBS` / `CHUNK` change only the wall clock:

```python
run_moon_lazar_grid("moon_lazar", sigmas, scenarios, reps=500,
                    n_jobs=32, chunk=25, **MOON_LAZAR_SETTINGS)
```

`tests/` pins this property directly, so a refactor that breaks it fails there
rather than silently producing shard-order-dependent numbers.

Workers are **forked**, so the parent's already-imported numpy, gudhi, ripser
and persim pages are shared copy-on-write instead of duplicated per process.
Each worker's own working set is a few MB (20 clouds of 50 points, 40×40
persistence images), which is what keeps 32 workers inside 8 GB. Raise
`default_workers(cap=...)` if you have more RAM headroom than cores.

## Output

Each notebook writes a JSON summary and a PNG to `results/`, then offers both
as browser downloads. `results/*.json` and `results/*.csv` are gitignored, so
committing a run is a deliberate `git add -f`.
