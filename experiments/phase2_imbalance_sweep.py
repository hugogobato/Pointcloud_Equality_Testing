"""Phase 2 gate: covariate-shift failure of the field's tests (tasks 2.1-2.4).

Two experiments and Figure 1.

Part A -- false-positive sweep (task 2.3), Figure 1a:
  DGP: ``CloudSampleDGP(group_effect=0)``; X ~ N(0, I_3); propensity
  ``expit(PROP_SCALE * lambda * X @ beta)``; topology ``k(x) =
  1 + floor(expit(x_0) * 3)`` loops.  For lambda = 0 the groups are randomised
  and every test has exact size alpha; as lambda grows, X drives both the group
  assignment and the loop count, so ``L(D|A=1) != L(D|A=0)`` while the causal
  nulls hold *exactly*: ``psi_d = 0`` (groups conditionally identical given X)
  and ``delta_dist = 0``.  Reported: rejection rate per (test, lambda): the
  type-I error of the six competitors climbing away from alpha (Theorem 2.1),
  and the DR prototype (task 2.4) holding its level.

Part B -- masking (task 2.2), Figure 1b:
  DGP: ``masking_stratum_sample``: ``L(D|A=1) = L(D|A=0)`` exactly (verified
  algebra and in tests), while ``psi_d != 0`` (Theorem 2.2).  The six
  competitors sit at alpha (their null is exactly true); the DR prototype has
  power.  Diagnostics for the writeup: the per-cloud max-H1-persistence
  difference between arms (centered at 0) and the KS test on pooled
  persistences (uniform under the null).

Modes
-----
* ``--mode shard --shard-idx i --reps-per-shard R``: run replications
  ``[i*R, (i+1)*R)`` of *both* parts in-process (no joblib) and write
  ``results/shards/phase2_shard{i}.json``.  The Colab fleet (see
  ``experiments/colab/``) runs 100 shards x 10 reps = 1000 reps per part,
  five shards per notebook, downloading each shard as it lands.
* ``--mode local --shards 0-3 --workers W``: the same shards, with joblib over
  W workers and one checkpoint file per shard.  Replication indices are keyed
  off the shard, so a shard run here and the same shard run on Colab are
  interchangeable; ``--skip-existing`` resumes an interrupted run.
* ``--mode aggregate``: merge every ``results/shards/phase2_shard*.json``
  (deduplicated by replication index), compute rejection rates, draw
  ``results/phase2_figure1.png`` and the Figure-1 data JSON, and print the
  Phase 2 GATE summary.

Design notes (Section 5 of RESEARCH_PLAN_P1_TwoSample.md):
  * clouds depend on X only (``group_effect=0``), so one set of clouds and
    diagrams serves every lambda: a common-random-numbers design whose group
    splits differ only through the labels.  The DR silhouette triplet is also
    shared.
  * every label-independent structure is built once per replication and read
    back per split (``_precompute``): RT's pairwise bottleneck matrix, MMD's
    diagram Gram matrix and Han's four per-bandwidth kernel matrices.  Those
    three are ~85% of a replication's cost and none of them depends on the
    labels, so rebuilding them for each of the six lambdas buys nothing.
    Group membership enters only as a boolean mask over the pooled rows.
  * near-diagonal diagram points (persistence < ``EPS_RT``) are dropped before
    the RT bottleneck matrix and the MMD/Han kernels.  By stability of the
    bottleneck distance and the boundedness of the Gaussian kernel every entry
    changes by at most ``2 * EPS_RT``, the permutation null remains exactly
    valid for any statistic, and the sweep's signals (loop persistences >=
    ~0.65, loop-count gaps >= ~0.3) sit an order of magnitude above the filter
    error.  The published wrappers are untouched; the filter lives only here
    (see WP2, Section 4).
  * replications are embarrassingly parallel and each shard is a checkpoint,
    so an interrupted run loses at most the shard in flight.  ``--workers``
    caps the pool (default 16) to leave headroom for concurrent work on this
    machine; the Colab notebooks size it from the VM's CPU count.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from joblib import Parallel, delayed

from tda2s.adapters.dr_test import prototype_dr_from_phi
from tda2s.benchmarks import (han_kernels, mmd_gram, run_competitor,
                              test_han_from_kernels, test_mmd_from_gram,
                              test_rt_from_matrix)
from tda2s.dgp import CloudSampleDGP, masking_stratum_sample, to_silhouette_sample
from tda2s.ph import compute_diagrams

# ----------------------------------------------------------------- config ----
N_PER_GROUP = 200
M = 120
PROP_SCALE = 1.5                       # propensity logit scale at lambda = 1
BETA = np.array([-0.5, -0.1, 0.6])     # tcda_uq default coefficients (d_x = 3)
LAMBDAS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
D_X = 3
N_PERM = 200
ALPHA = 0.05
INTERVAL = (0.0, 2.0)
RESOLUTION = 100
SIL_R = 3.0
N_BASIS = 8
N_FOLDS = 2
N_DRAWS = 2000
BASE_SEED = 2100
TARGET_REPS = 1000   # the fleet's replication budget; see WP2 section 6
EPS_RT = 0.1     # near-diagonal filter for the RT matrix / MMD / Han kernels
RT_APPROX = 0.01  # gudhi additive tolerance for the bottleneck matrix
COMPETITORS = ["rt", "mmd", "han", "strand", "moon_lazar", "frechet_anova"]
#: competitors with no label-independent precompute, re-run per group split
PER_SPLIT = ["strand", "moon_lazar", "frechet_anova"]
_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, "..", "results")
SHARDS = os.path.join(RESULTS, "shards")
FIG_PNG = os.path.join(RESULTS, "phase2_figure1.png")
FIG_JSON = os.path.join(RESULTS, "phase2_figure1.json")


def _seed(*parts):
    """Deterministic per-(part, rep, lambda, test) RNG seed."""
    s = BASE_SEED
    for p in parts:
        if isinstance(p, str):
            h = 0
            for ch in p.encode():
                h = (h * 31 + ch) % (2 ** 31)
            p = h
        s = (s * 7919 + int(p)) % (2 ** 31)
    return s


def _diagrams_of(clouds):
    return [compute_diagrams(c, homology_dims=(0, 1)) for c in clouds]


def _rt_matrix(diags):
    """Robinson-Turner pairwise joint-loss matrix, shared across all splits.

    Two implementation notes, both of which leave the permutation null exactly
    valid because the matrix is fixed before any label is drawn:

    * diagram points with persistence < ``EPS_RT`` are dropped (module
      docstring), and
    * the bottleneck calls ask gudhi for an additive ``RT_APPROX``
      approximation instead of the exact CGAL path, which is ~4x faster here.
      Every entry moves by at most ``2 * RT_APPROX = 0.02`` against a signal
      (loop persistences >= ~0.65) an order of magnitude larger.

    Empty (post-filter) diagrams are *not* skipped: ``d_B(varnothing, D)`` is
    half the largest persistence of ``D``, and in this sweep the group contrast
    is precisely a difference in loop *count*, so a diagram that filters down
    to nothing is the informative case, not a missing one.
    """
    n = len(diags)
    P = np.zeros((n, n))
    gd = __import__("gudhi", fromlist=["bottleneck_distance"])
    filt = [[d[d[:, 1] - d[:, 0] >= EPS_RT] for d in diag] for diag in diags]
    for dim in range(2):
        for i in range(n):
            Di = filt[i][dim]
            for j in range(i + 1, n):
                v = float(gd.bottleneck_distance(Di, filt[j][dim], RT_APPROX))
                P[i, j] = P[j, i] = P[i, j] + v
    return P


def _precompute(diags):
    """Everything about one replication that does not depend on the labels.

    RT's pairwise bottleneck matrix, MMD's diagram Gram matrix and Han's four
    per-bandwidth kernel matrices are functions of the *pooled* diagrams alone,
    so a replication that compares six propensity strengths over one sample
    builds them once here and reads them back per split. Rebuilding them per
    lambda is ~85% of the replication's cost and buys nothing.
    """
    return {"rt": _rt_matrix(diags),
            "mmd": mmd_gram(diags, epsilon=EPS_RT)[0],
            "han": han_kernels(diags, epsilon=EPS_RT)}


def _competitor_pvalues(diags, pre, A, seed):
    """All six competitor p-values for one group split of a replication.

    Args:
        diags: the pooled diagrams, in the replication's own order.
        pre: the label-independent structures from :func:`_precompute`.
        A: ``(N,)`` treatment labels in that same order.
        seed: base RNG seed for this split.

    The three precomputed tests read their matrices back under the mask
    ``A == 0``; the remaining three have no label-independent structure worth
    caching and are re-run on the split diagram lists.
    """
    mask = np.asarray(A).astype(bool)
    g0 = ~mask                       # True = group 0, in the pooled row order
    d0 = [d for d, m in zip(diags, mask) if not m]
    d1 = [d for d, m in zip(diags, mask) if m]

    out = {name: run_competitor(name, d0, d1, n_perm=N_PERM, seed=seed,
                                epsilon=EPS_RT)
           for name in PER_SPLIT}
    out["rt"] = float(test_rt_from_matrix(pre["rt"], g0, n_perm=N_PERM,
                                          statistic="within", seed=seed))
    out["mmd"] = float(test_mmd_from_gram(pre["mmd"], g0, n_perm=N_PERM,
                                          seed=seed))
    out["han"] = float(test_han_from_kernels(pre["han"], g0, n_perm=N_PERM,
                                             seed=seed))
    return out


# ---------------------------------------------------------------- part A -----
def expit_lambda(dgp, X, lam):
    return 1.0 / (1.0 + np.exp(-PROP_SCALE * lam * (np.asarray(X) @ dgp.beta)))


def rep_false_positive(rep):
    """One replication of the Part A sweep; returns per-lambda p-values.

    Every RNG draw is keyed off ``rep`` (and, for the labels, off lambda), so a
    replication is reproducible in isolation: shard *i* run on Colab and the
    same replication index run here give the same numbers.
    """
    seed = _seed("A", rep)
    dgp = CloudSampleDGP(n_per_group=N_PER_GROUP, m=M, d_x=D_X, beta=BETA,
                         prop_scale=PROP_SCALE, group_effect=0, seed=seed)
    sample = dgp.sample(rng=seed)
    clouds, X = sample.clouds, sample.X
    n = len(clouds)

    diags = _diagrams_of(clouds)
    pre = _precompute(diags)
    phi, _, _ = to_silhouette_sample(clouds, X, np.zeros(n), filtration="alpha",
                                     homology_dims=(0, 1), interval=INTERVAL,
                                     r=SIL_R, resolution=RESOLUTION)
    tseq = np.linspace(INTERVAL[0], INTERVAL[1], RESOLUTION)

    out = {}
    for lam in LAMBDAS:
        key = f"lam{lam:g}"
        pi = expit_lambda(dgp, X, lam)
        labels_rng = np.random.default_rng(_seed("A", rep, key, "labels"))
        A = labels_rng.binomial(1, pi).astype(int)
        pvals = _competitor_pvalues(diags, pre, A, _seed("A", rep, key, "t"))
        pvals["dr"] = prototype_dr_from_phi(phi, A, X, tseq, n_basis=N_BASIS,
                                            n_folds=N_FOLDS, n_draws=N_DRAWS,
                                            seed=_seed("A", rep, key, "dr"))
        out[key] = pvals
    return {"rep": rep, "pvals": out}


# ---------------------------------------------------------------- part B -----
def rep_masking(rep):
    """One replication of the Part B masking experiment."""
    seed = _seed("B", rep)
    sample = masking_stratum_sample(N_PER_GROUP, seed=seed)
    clouds, X, A = sample.clouds, sample.X, sample.A

    diags = _diagrams_of(clouds)
    pre = _precompute(diags)
    phi, _, _ = to_silhouette_sample(clouds, X, A, filtration="alpha",
                                     homology_dims=(0, 1), interval=INTERVAL,
                                     r=SIL_R, resolution=RESOLUTION)
    tseq = np.linspace(INTERVAL[0], INTERVAL[1], RESOLUTION)

    pvals = _competitor_pvalues(diags, pre, A, _seed("B", rep, "t"))
    pvals["dr"] = prototype_dr_from_phi(phi, A, X, tseq, n_basis=N_BASIS,
                                        n_folds=N_FOLDS, n_draws=N_DRAWS,
                                        seed=_seed("B", rep, "dr"))

    pers1 = np.concatenate([d[1][:, 1] - d[1][:, 0] for d, m in zip(diags, A)
                            if m and d[1].size])
    pers0 = np.concatenate([d[1][:, 1] - d[1][:, 0] for d, m in zip(diags, A)
                            if not m and d[1].size])
    mean_diff = float(pers1.mean() - pers0.mean()) if pers1.size and pers0.size else None
    ks_p = None
    if pers1.size and pers0.size:
        from scipy import stats
        ks_p = float(stats.ks_2samp(pers1, pers0).pvalue)
    return {"rep": rep, "pvals": pvals, "mean_pers_diff": mean_diff, "ks_p": ks_p}


# ---------------------------------------------------------------- shards -----
def run_shard(shard_idx, reps_per_shard, workers=1):
    """Run replications ``[shard_idx*R, (shard_idx+1)*R)`` of both parts.

    The file is named after the *replication range*, never after where it ran,
    so a shard computed locally and the same shard computed on Colab are
    interchangeable and the aggregate deduplicates them cleanly.

    Returns the path of the written ``results/shards/phase2_shard<i>.json``.
    """
    start = int(shard_idx) * int(reps_per_shard)
    reps = list(range(start, start + int(reps_per_shard)))
    t0 = time.time()
    if workers > 1:
        part_a = Parallel(n_jobs=workers, verbose=1)(
            delayed(rep_false_positive)(r) for r in reps)
        part_b = Parallel(n_jobs=workers, verbose=1)(
            delayed(rep_masking)(r) for r in reps)
    else:
        part_a = [rep_false_positive(r) for r in reps]
        part_b = [rep_masking(r) for r in reps]
    name = f"shard{shard_idx}"
    path = os.path.join(SHARDS, f"phase2_{name}.json")
    os.makedirs(SHARDS, exist_ok=True)
    payload = {
        "name": name,
        "config": {"n_per_group": N_PER_GROUP, "m": M, "prop_scale": PROP_SCALE,
                   "beta": list(BETA), "lambdas": list(LAMBDAS),
                   "n_perm": N_PERM, "n_draws": N_DRAWS, "epsilon": EPS_RT,
                   "reps_per_shard": int(reps_per_shard),
                   "start": start, "reps": reps},
        "part_a": part_a, "part_b": part_b,
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh)
    os.replace(tmp, path)
    dt = time.time() - t0
    print(f"[phase2] shard {name}: {len(reps)} reps x 2 parts in {dt:.0f}s "
          f"-> {path}  (per-rep avg {dt / (2 * len(reps)):.1f}s)")
    return path


#: keys of ``payload["config"]`` that must agree across shards to be poolable.
_POOLABLE = ("n_per_group", "m", "prop_scale", "beta", "lambdas", "n_perm",
             "n_draws", "epsilon")


def _load_shards():
    """Merge every ``phase2_shard<i>.json``, deduplicated by replication index.

    Only files named for a replication range are read: anything else in the
    directory (a scratch run, a smoke test) is reported and skipped rather than
    silently pooled, since a stray file whose replication indices overlap a
    real shard would win the deduplication and quietly replace it.

    Shards whose sampling configuration disagrees with the first are refused
    outright: pooling replications run under different ``n_per_group`` or
    ``lambdas`` would produce a rejection rate that estimates nothing.
    """
    all_files = sorted(f for f in os.listdir(SHARDS) if f.endswith(".json"))
    files = [f for f in all_files if f.startswith("phase2_shard")]
    skipped = [f for f in all_files if f not in files]
    if skipped:
        print(f"[phase2] ignoring {len(skipped)} non-shard file(s) in {SHARDS}: "
              f"{', '.join(skipped)}")
    if not files:
        raise SystemExit(f"no phase2_shard*.json under {SHARDS}; "
                         f"run --mode shard/local first")
    config, part_a, part_b = None, {}, {}
    for fn in files:
        with open(os.path.join(SHARDS, fn)) as fh:
            data = json.load(fh)
        cfg = data["config"]
        if config is None:
            config = cfg
        else:
            bad = [k for k in _POOLABLE if cfg.get(k) != config.get(k)]
            if bad:
                raise SystemExit(
                    f"{fn} disagrees with the other shards on {bad}; "
                    f"re-run it against the current config before aggregating")
        for r in data["part_a"]:
            part_a.setdefault(r["rep"], r)
        for r in data["part_b"]:
            part_b.setdefault(r["rep"], r)
    print(f"[phase2] merged {len(files)} shard file(s): "
          f"{len(part_a)} part-A and {len(part_b)} part-B replications")
    _report_coverage(part_a, part_b)
    return config, part_a, part_b


def _report_coverage(part_a, part_b, target=TARGET_REPS):
    """Report which replication indices are missing from the merged fleet.

    With a hundred shard files arriving from a notebook fleet, a dropped
    download is easy to miss and silently costs power rather than raising
    anything, so the gap is named explicitly.
    """
    for label, part in (("part A", part_a), ("part B", part_b)):
        have = set(part)
        if not have:
            continue
        missing = sorted(set(range(target)) - have)
        extra = sorted(r for r in have if r >= target)
        if not missing and not extra:
            print(f"[phase2] {label}: complete, replications 0-{target - 1}")
            continue
        # collapse the missing indices into the shard ranges they came from
        shards = sorted({r // 10 for r in missing})
        head = ", ".join(str(s) for s in shards[:12])
        tail = "" if len(shards) <= 12 else f", ... (+{len(shards) - 12} more)"
        print(f"[phase2] {label}: {len(have)}/{target} replications; "
              f"{len(shards)} shard(s) missing at 10 reps/shard: {head}{tail}")
        if extra:
            print(f"[phase2] {label}: {len(extra)} replication(s) beyond "
                  f"index {target - 1} were also merged")


# ---------------------------------------------------------------- figure -----
def _rates_a(part_a, keys):
    tests = list(COMPETITORS) + ["dr"]
    rates = {t: {k: 0.0 for k in keys} for t in tests}
    for rep in part_a.values():
        for k in keys:
            for t in tests:
                rates[t][k] += (rep["pvals"][k][t] <= ALPHA)
    n = max(1, len(part_a))
    for t in tests:
        for k in keys:
            rates[t][k] /= n
    return rates


def _rates_b(part_b):
    tests = list(COMPETITORS) + ["dr"]
    rates = {t: 0.0 for t in tests}
    for rep in part_b.values():
        for t in tests:
            rates[t] += (rep["pvals"][t] <= ALPHA)
    n = max(1, len(part_b))
    for t in tests:
        rates[t] /= n
    return rates


def make_figure(sweep_rates, masking_rates, n_reps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    tests = list(COMPETITORS) + ["dr"]
    labels = {"rt": "Robinson-Turner", "mmd": "MMD", "han": "Han et al.",
              "strand": "STRAND", "moon_lazar": "Moon-Lazar",
              "frechet_anova": "Frechet ANOVA", "dr": "DR prototype"}
    lam_keys = [f"lam{lam:g}" for lam in LAMBDAS]
    xs = np.arange(len(lam_keys))
    for t in tests:
        ax1.plot(xs, [sweep_rates[t][k] for k in lam_keys], marker="o",
                 ms=4, lw=1.4, label=labels[t])
    ax1.axhline(ALPHA, color="k", ls="--", lw=0.9)
    # a rejection rate cannot be negative, so clip the band rather than let it
    # run below the axis at small replication counts
    se = np.sqrt(ALPHA * (1 - ALPHA) / max(1, n_reps))
    # annotate at the right edge, above the band: the left edge is where every
    # curve starts at alpha, so a label there sits on top of the data
    ax1.text(xs[-1], ALPHA + 3 * se + 0.015, r"$\alpha = 0.05$", fontsize=8,
             ha="right", va="bottom")
    ax1.fill_between(xs, max(0.0, ALPHA - 3 * se), ALPHA + 3 * se, color="k",
                     alpha=0.12,
                     label=f"$\\alpha \\pm 3\\,\\mathrm{{SE}}$ ({n_reps} reps)")
    ax1.set_xticks(xs, [f"{lam:g}" for lam in LAMBDAS])
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_xlabel("imbalance $\\lambda$ (propensity logit scale)")
    ax1.set_ylabel("type-I error rate at $\\alpha = 0.05$")
    ax1.set_title("(a) false positives under covariate shift ($\\psi_d \\equiv 0$)")
    ax1.legend(fontsize=7.5, loc="upper left")
    ax1.grid(alpha=0.25)

    y = [masking_rates[t] for t in tests]
    bars = ax2.bar(np.arange(len(tests)), y, color=["#4477AA"] * 6 + ["#CC6677"])
    ax2.axhline(ALPHA, color="k", ls="--", lw=0.9)
    ax2.set_xticks(np.arange(len(tests)), [labels[t] for t in tests], rotation=28, fontsize=8)
    ax2.set_ylabel("rejection rate at $\\alpha = 0.05$")
    ax2.set_ylim(0, 1.12)   # headroom for the bar value labels
    ax2.set_title("(b) Simpson masking ($L(D|A{=}1)=L(D|A{=}0)$, $\\psi_d \\neq 0$)")
    ax2.grid(axis="y", alpha=0.25)
    for b, v in zip(bars, y):
        # clear the alpha line: a near-alpha bar's label lands on it at +0.02
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.045, f"{v:.3f}",
                 ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_PNG, dpi=200)
    return fig


def aggregate(print_gate=True):
    """Merge shards, write Figure 1 + data JSON, print the gate report."""
    config, part_a, part_b = _load_shards()
    n_a, n_b = len(part_a), len(part_b)
    sweep_rates = _rates_a(part_a, [f"lam{lam:g}" for lam in LAMBDAS])
    masking_rates = _rates_b(part_b)

    make_figure(sweep_rates, masking_rates, n_a)
    fig_data = {
        "n_reps": {"sweep": n_a, "masking": n_b},
        "alpha": ALPHA,
        "config": config,
        "sweep_rates": sweep_rates,
        "masking_rates": masking_rates,
    }
    with open(FIG_JSON, "w") as fh:
        json.dump(fig_data, fh, indent=1)
    print(f"[phase2] figure written to {FIG_PNG}, data to {FIG_JSON}")
    if print_gate:
        print_gate_report(sweep_rates, masking_rates, part_b, n_a)
    return fig_data


def print_gate_report(sweep_rates, masking_rates, part_b, n_reps):
    """The task 2.5 verdict, as stated in RESEARCH_PLAN_P1_TwoSample.md.

    The plan's criterion is a *disjunction* over the two failure modes, both
    conditioned on the DR prototype keeping its level:

        (2.1 false positives at the strongest imbalance  OR  2.2 masking)
        AND  the DR prototype holds size across the sweep.

    The labels below are the plan's *task* numbers, not its contribution
    labels: both failure modes are evidence for contribution C1, which is why
    either one of them passes the gate. 2.1 is read at ``lambda = 1``
    specifically, not as a maximum over the sweep:
    the claim being certified is that the competitors fail *at the strongest
    imbalance*, and a maximum over six settings would also fire on a single
    lucky interior point.

    One refinement of the plan's binary verdict. The plan's FAIL branch says
    "drop C1, rewrite the abstract around C2+C3", which is the right response
    when the *competitors* turn out to be fine under imbalance. It is not the
    right response when the competitor evidence is overwhelming and the only
    unmet condition is the size of the deliberately uncalibrated Phase 2.4
    prototype (task 2.4's own words: "rough, uncalibrated ... polish in Phase
    3"). That combination says the prototype needs Phase 3's calibration (3.2:
    proper multiplier bootstrap and the stratified-permutation variant), not
    that C1 is false. It is reported as INCONCLUSIVE and left as the caller's
    call, rather than silently counted as either outcome.
    """
    lam_keys = [f"lam{lam:g}" for lam in LAMBDAS]
    worst_at_1 = max(sweep_rates[t]["lam1"] for t in COMPETITORS)
    worst_over_sweep = max(max(sweep_rates[t].values()) for t in COMPETITORS)
    dr_size = [sweep_rates["dr"][k] for k in lam_keys]
    max_comp_mask = max(masking_rates[t] for t in COMPETITORS)
    dr_mask = masking_rates["dr"]

    fp_ok = worst_at_1 >= 0.20
    mask_ok = max_comp_mask <= 0.10 and dr_mask >= 0.70
    size_ok = all(0.03 <= v <= 0.08 for v in dr_size)
    se = np.sqrt(ALPHA * (1 - ALPHA) / max(1, n_reps))

    provisional = n_reps < TARGET_REPS
    banner = ("PROVISIONAL (partial fleet)" if provisional else "")
    print(f"\n===== Phase 2 GATE summary {banner} =====")
    print(f"  replications: {n_reps}  (MC se at alpha: {se:.4f})")
    if provisional:
        print(f"  ** {n_reps} of {TARGET_REPS} replications. The size band "
              f"[0.03, 0.08] is +-2 MC se at 200 replications, so a verdict "
              f"read off this subset is not the gate. **")
    print(f"  2.1 false positives: worst competitor at lambda=1 = {worst_at_1:.3f} "
          f"(need >= 0.20) -> {'MET' if fp_ok else 'not met'}")
    print(f"                       (worst anywhere in the sweep: {worst_over_sweep:.3f})")
    print(f"  2.2 masking        : competitor max = {max_comp_mask:.3f} "
          f"(need <= 0.10), DR = {dr_mask:.3f} (need >= 0.70) "
          f"-> {'MET' if mask_ok else 'not met'}")
    print(f"  2.4 DR size        : {min(dr_size):.3f}-{max(dr_size):.3f} across lambda "
          f"(need every lambda in [0.03, 0.08]) -> {'MET' if size_ok else 'not met'}")
    if not size_ok:
        bad = [f"{k}={v:.3f}" for k, v in zip(lam_keys, dr_size)
               if not 0.03 <= v <= 0.08]
        print(f"                       outside the band: {', '.join(bad)}")
    diffs = np.array([r["mean_pers_diff"] for r in part_b.values()
                      if r["mean_pers_diff"] is not None])
    if diffs.size:
        print(f"  diagnostic         : E[H1 pers | A=1] - E[H1 pers | A=0] = "
              f"{diffs.mean():+.4f} (sd {diffs.std():.4f}, want ~0)")
    evidence = fp_ok or mask_ok
    if evidence and size_ok:
        verdict, action = "PASS", ("C1 is the spine; proceed to Phases 3-7 "
                                   "as written")
    elif evidence:
        verdict, action = "INCONCLUSIVE", (
            "the competitor failure is established, but the uncalibrated "
            "prototype does not hold its level; this is Phase 3.2's job, not "
            "a refutation of C1. Calibrate, then re-fire")
    else:
        verdict, action = "FAIL", ("drop C1, rewrite around C2+C3 "
                                   "(see plan section 3, Phase 2)")
    print(f"  GATE{' (provisional)' if provisional else ''}: {verdict} -> {action}")
    return verdict


# ----------------------------------------------------------------- main ------
def _parse_shards(spec):
    """``"3"`` / ``"0-7"`` / ``"0,2,5-7"`` -> a list of shard indices."""
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["shard", "local", "aggregate"], required=True)
    ap.add_argument("--shard-idx", type=int, default=0,
                    help="shard index for --mode shard (one notebook = one shard)")
    ap.add_argument("--shards", default=None,
                    help="shard indices for --mode local, e.g. '0-7' or '0,3,5'; "
                         "defaults to --shard-idx")
    ap.add_argument("--reps-per-shard", type=int, default=25)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip-existing", action="store_true",
                    help="--mode local: leave already-written shard files alone")
    args = ap.parse_args()

    if args.mode == "shard":
        run_shard(args.shard_idx, args.reps_per_shard, workers=1)
    elif args.mode == "local":
        idxs = _parse_shards(args.shards) if args.shards else [args.shard_idx]
        for i in idxs:
            path = os.path.join(SHARDS, f"phase2_shard{i}.json")
            if args.skip_existing and os.path.exists(path):
                print(f"[phase2] shard {i} already at {path}, skipping")
                continue
            run_shard(i, args.reps_per_shard, workers=args.workers)
    else:
        aggregate()


if __name__ == "__main__":
    main()
