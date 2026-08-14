# tda2s

Topological two-sample testing: covariate-adjusted, doubly robust, and conditional tests for persistent homology (P1), sharing infrastructure with certified filtration substitution (P2).

## Install

```bash
uv venv --python 3.10 .venv
uv pip install -e ".[dev,tate]"
pytest -q
```

The `tate` extra installs `tcda_uq` (the released `CP_TATE` library) from GitHub; P1 imports its AIPW / cross-fitting / DR-learner rather than reimplementing them.

## Layout

| Path | Contents |
|---|---|
| `tda2s/ph/` | cloud → diagram pipeline (VR, Alpha, Čech, cubical-sublevel, DTM-Rips) |
| `tda2s/vec/` | vectorisation: power-weighted silhouette, landscapes, Betti/Euler curves, persistence images, persistence measures |
| `tda2s/resample/` | label permutation, multiplier bootstrap, paired bootstrap, smoothed bootstrap, cross-fitting folds |
| `tda2s/benchmarks/` | competitor wrappers: RT, MMD, Han et al., STRAND, Moon–Lazar, Fréchet ANOVA, Krebs–Rademacher |
| `tda2s/dgp/` | controlled point-cloud generators with covariate / propensity / covariate-driven-topology knobs |
| `tda2s/adapters/` | thin shims over `tcda_uq` (AIPW, cross-fitting, DR-learner, silhouettes) |
| `tda2s/repro/` | the competitors' published simulation designs, plus a sharded runner |
| `notebooks/` | Colab notebooks reproducing published figures at the papers' full budgets |
| `tests/` | pytest suite (incl. `test_forward_compat.py` for the deferred ecological track) |

## Reproducing the competitors

None of the seven competitor methods ships author code, so every wrapper in
`tda2s/benchmarks/` is a replication from its paper, with section and equation
numbers cited in the module docstring. Each is checked against a published
figure where the source paper has one:

```bash
pytest tests/test_published_reproductions.py -q -s     # ~25 min, 150 reps
```

The designs live in `tda2s/repro/` so the tests and the notebooks share one
implementation. `notebooks/` runs the same designs at the papers' 500
replications; see `notebooks/README.md`.