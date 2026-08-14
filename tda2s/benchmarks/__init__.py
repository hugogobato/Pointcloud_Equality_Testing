"""Competitor two-sample tests on persistence diagrams (Phase 0.5).

Every wrapper takes ``(diags0, diags1)`` first and returns a p-value in
[0, 1], where ``diags0``/``diags1`` are lists over samples of lists (per
homology dim) of ``(k, 2)`` birth-death arrays.

Their *keyword* arguments are NOT uniform: six of the seven are permutation
tests taking ``n_perm``/``seed``, but ``moon_lazar`` is analytic (pooled-variance
t-tests + Benjamini-Hochberg) and accepts neither. Call through
:func:`run_competitor` to sweep the registry with one kwargs dict; it drops the
arguments a given wrapper does not accept instead of raising ``TypeError``.

None of these methods ship author code, so every wrapper is a transcription of
its source paper with the section and equation numbers cited in the module
docstring. ``tests/test_published_reproductions.py`` reproduces a published
figure for each method that has one.

Competitors:
    * ``rt``              - Robinson & Turner (2017) permutation test on
                            pairwise Wasserstein distances.
    * ``mmd``             - Kwitt et al. (2015) kernel MMD on diagram points.
    * ``han``             - Han, Kim & Kim (2026) kernel permutation test on
                            weighted persistence intensity functions.
    * ``strand``          - Murris, Stolz & Borgwardt (2026) log-rank test on
                            feature lifetimes, stratified by homology dim.
    * ``moon_lazar``      - Moon & Lazar (2023) Algorithm 1: persistence images,
                            variance pre-filter, pooled-variance t-tests, FDR.
    * ``frechet_anova``   - Dubey & Muller (2019) Frechet ANOVA, eqs. (6)-(11),
                            including the Levene-type ``U_n`` term.
    * ``krebs_rademacher``- Krebs & Rademacher (2024) Section 1.2 relevant-
                            difference test on Wasserstein inco-variances.
                            Targets DISPERSION, not location: it is blind to a
                            pure location shift by construction (eq. 1.7).
"""
from .frechet_anova import test_frechet_anova
from .han import test_han
from .krebs_rademacher import test_krebs_rademacher
from .mmd import test_mmd
from .moon_lazar import test_moon_lazar
from .rt import test_rt
from .strand import test_strand

COMPETITORS = {
    "rt": test_rt,
    "mmd": test_mmd,
    "han": test_han,
    "strand": test_strand,
    "moon_lazar": test_moon_lazar,
    "frechet_anova": test_frechet_anova,
    "krebs_rademacher": test_krebs_rademacher,
}

#: Wrappers that are analytic rather than permutation-based.
ANALYTIC = frozenset({"moon_lazar"})


def run_competitor(name, diags0, diags1, **kwargs):
    """Run competitor ``name``, passing only the kwargs it actually accepts.

    Lets a caller sweep the whole registry with a single kwargs dict (e.g.
    ``n_perm=200, seed=0``) even though the analytic wrappers take no such
    arguments.

    Args:
        name: key of :data:`COMPETITORS`.
        diags0, diags1: the two groups of diagram lists.
        **kwargs: candidate keyword arguments; unsupported ones are dropped.

    Returns:
        float p-value in [0, 1].
    """
    import inspect

    try:
        fn = COMPETITORS[name]
    except KeyError:
        raise ValueError(
            f"unknown competitor {name!r}; expected one of {sorted(COMPETITORS)}") from None
    accepted = inspect.signature(fn).parameters
    return float(fn(diags0, diags1,
                    **{k: v for k, v in kwargs.items() if k in accepted}))


__all__ = ["COMPETITORS", "ANALYTIC", "run_competitor", "test_rt", "test_mmd",
           "test_han", "test_strand", "test_moon_lazar", "test_frechet_anova",
           "test_krebs_rademacher"]
