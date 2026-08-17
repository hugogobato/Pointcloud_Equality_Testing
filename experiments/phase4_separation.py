"""Phase 4 witness experiments: W1 separation and the W2' reverse case (C2).

Two designs, both on the exact deterministic witness DGPs, both running the
outcome-level Phase 3 test and the distribution-level Phase 4 test (expected
persistence measure, candidate (i), and the universal-kernel MMD, candidate
(ii)) on the same replications:

``separation`` (W1)  -- cluster splitting with preserved mean silhouette.
    Arm 0 clouds are 2-blob deterministic 18-gons (36 points, one finite H_0
    class at the merge scale), arm 1 clouds are 3-blob deterministic 12-gons
    (36 points, two finite H_0 classes at the same merge scale).  The polygon
    resolutions differ so that total cloud cardinality is equal across arms;
    both within-blob classes remain below the persistence threshold.
    After the TAU=0.3 persistence threshold the per-arm diagrams are exactly
    {(0, 1.35)} and {(0, 1.35), (0, 1.35)}: the mean normalized silhouette is
    preserved realization-by-realization (H0^out true, the outcome permutation
    line stays near alpha while the multiplier line is degenerate) while the expected persistence measures differ by
    1.35**3 (H0^dist false, Phase 4 power -> 1).

``reverse`` (W2')  -- the multiplicity mixture over two locations, lifted to
    clouds.  Arm 0 draws the merge-staircase cloud {(0,1),(0,2)} exactly;
    arm 1 draws {(0,1),(0,1)} or {(0,2),(0,2)} with probability 1/2 each
    (equilateral 12-gons at merge scales 1 and 2; equal 36-point clouds).
    In law the expected persistence measures agree exactly for any point
    weight (H0^dist true); the frozen-label expected-measure permutation is
    only a calibration diagnostic here because the unit-level laws differ.
    The mean normalized silhouettes differ by 0.544 on the sup scale
    (H0^out false: Phase 3 power -> 1).  The MMD variant fires on W2' too,
    which is the honest
    "no reverse case exists under candidate (ii)" note: under (ii) the null
    is strictly stronger, so the distribution-level test is *not* blind.

Both designs run the outcome-level test through the full Phase 3 pipeline
(``tda2s.tests.dr_outcome.fit_dr`` + frozen-nuisance stratified permutation
and shared multiplier) with a null covariate X ~ N(0,1) and a randomized
design (fixed-count complete randomization with known marginal propensity), so
the outcome-level arm is literally the Phase 3 test and the permutation null
is the corresponding relabelling statement at lambda = 0 (plan 6.2).  The
distribution-level arm runs
``tda2s.tests.dist_level.fit_dist`` + ``stratified_permutation_test`` on the
same sample with the same strata and frozen weights.

Sharded exactly like the Phase 3/3.5 fleets; replication seeds are functions
of the replication index alone, so shards concatenate whatever way the work
is split (``--workers`` parallelises within a shard and cannot change the
output).  No permutation draw recomputes persistent homology, cross-fitting,
or a nuisance regression.

    rtk uv run python experiments/phase4_separation.py --mode shard \
        --design separation --shard-idx 0 --reps-per-shard 10 --workers 10
    rtk uv run python experiments/phase4_separation.py --mode aggregate
    rtk uv run python experiments/phase4_separation.py --mode figure
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.linear_model import LogisticRegression

from tda2s.dgp.clouds import merge_staircase_cloud, split_cluster_cloud
from tda2s.ph import compute_diagrams
from tda2s.tests.dist_level import (
    fit_dist,
    mmd_kernel_matrix,
    stratified_permutation_test as dist_permutation_test,
)
from tda2s.tests.dr_outcome import (
    fit_dr,
    multiplier_test,
    stratified_permutation_test as outcome_permutation_test,
)
from tda2s.vec import persistence_measure, silhouette

BASE_SEED = 4100
ALPHA = 0.05
SAMPLE_SIZES = (25, 50, 100, 200)
N_CALIBRATION = 399
TAU = 0.3                # persistence threshold isolating the merge classes
INTERVAL = (0.0, 2.0)    # ONE grid shared by every silhouette and measure
RESOLUTION = 50          # silhouette grid (Phase 3 convention)
N_BINS = 32              # measure bins per axis
R = 3.0                  # silhouette power weight; the measure weight matches it
NOISE = 0.15             # deterministic blob radius
N_GON = 12               # vertices per deterministic blob
W1_ARM1_N_GON = N_GON    # 3 blobs x 12 points = 36 points
W1_ARM0_N_GON = 3 * N_GON // 2  # 2 blobs x 18 points = 36 points
N_BASIS = 5
N_FOLDS = 2
_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, "..", "results")
SHARDS = os.path.join(RESULTS, "phase4_shards")


def _seed(*parts) -> int:
    value = BASE_SEED
    for part in parts:
        if isinstance(part, str):
            part = sum((i + 1) * byte for i, byte in enumerate(part.encode()))
        value = (value * 7919 + int(part)) % (2 ** 31 - 1)
    return int(value)


# ---------------------------------------------------------------------------
# witness DGPs (exact at the diagram level after the TAU threshold)

def _threshold_diagrams(diagrams, tau: float = TAU) -> list:
    """Keep the H_0 classes with persistence above ``tau`` (the witness filter).

    The deterministic blobs carry tiny within-blob classes (persistence
    ``NOISE * sin(pi / N_GON) ~ 0.039``) that are present in both arms in
    unequal numbers; the W1/W2' witnesses are statements about the filtered
    diagrams, so the threshold is applied before any silhouette or measure is
    computed.  The kept (birth, death) coordinates are then rounded to 1e-6:
    the alpha filtration returns the exact merge scales up to floating-point
    noise (e.g. ``1.0`` as ``0.9999999999999999``), and on the pinned
    measure grid that 1e-16 displacement is enough to move a class across a
    bin edge (measured: mid 0.5 vs 0.49999999999999994 land in bins 8 and 7,
    which destroyed the W2' expected-measure equality).  The witness scales
    are separated by orders of magnitude more than the rounding, so this is
    witness-exactness restoration, not information loss.
    """
    out = []
    for per_dim in diagrams:
        dgm = np.asarray(per_dim[0], dtype=float).reshape(-1, 2)
        keep = dgm[(dgm[:, 1] - dgm[:, 0] > tau)]
        out.append([np.round(keep, 6)])
    return out


def _w1_cloud(arm: int, rng) -> np.ndarray:
    """W1 arm cloud with equal total cardinality in both arms."""
    n_blobs = 2 if arm == 0 else 3
    n_gon = W1_ARM0_N_GON if arm == 0 else W1_ARM1_N_GON
    return split_cluster_cloud(n_gon, n_blobs, separation=3.0, noise=NOISE,
                               deterministic=True, n_gon=n_gon, rng=rng)


def _w2p_cloud(arm: int, rng) -> np.ndarray:
    """W2' arm cloud: staircase {(0,1),(0,2)} or the two-location mixture."""
    if arm == 0:
        return merge_staircase_cloud([1.0, 2.0], noise=NOISE, n_gon=N_GON,
                                     rng=rng)
    separation = 2.3 if rng.random() < 0.5 else 4.3   # merges (1,1) or (2,2)
    return split_cluster_cloud(N_GON, 3, separation=separation, noise=NOISE,
                               deterministic=True, n_gon=N_GON, rng=rng)


def _silhouette_of(diagrams: list) -> np.ndarray:
    """Silhouette (1, RESOLUTION) of a thresholded H_0 diagram list."""
    return silhouette([diagrams[0]], interval=INTERVAL, r=R,
                      resolution=RESOLUTION)[0]


def _measure_of(diagrams: list) -> np.ndarray:
    """Expected-measure features (N_BINS**2,) on the pinned grid."""
    w = (lambda p: float(abs(p[1] - p[0]) ** R))
    return persistence_measure([diagrams[0]], weight=w, interval=INTERVAL,
                               n_bins=N_BINS)[0].ravel()


# ---------------------------------------------------------------------------
# one replication

def _one_rep(rep: int, n: int, design: str, n_calibration: int = N_CALIBRATION,
             tau: float = TAU) -> dict:
    """One sample, both tests, both mechanisms, plus the MMD variant.

    The design is complete randomization with a fixed treatment count,
    independent of the null covariate ``X ~ N(0, 1)``.  The supplied known
    propensity is the corresponding marginal treatment probability
    ``n_treated / n`` (1/2 for even ``n``); permutation strata collapse to a
    single stratum.
    """
    if design not in ("separation", "reverse"):
        raise ValueError("design must be 'separation' or 'reverse'")
    cloud_fn = _w1_cloud if design == "separation" else _w2p_cloud
    rng = np.random.default_rng(_seed("sample", rep, n, design))
    n_arm = n // 2
    n_treated = n - n_arm
    labels = np.concatenate([np.zeros(n_arm, dtype=int),
                             np.ones(n_treated, dtype=int)])
    rng.shuffle(labels)
    X = rng.normal(size=(n, 1))
    pi_known = np.full(n, n_treated / n)
    clouds = []
    for a in labels:
        clouds.append(cloud_fn(int(a), rng))

    diagrams = _threshold_diagrams(
        [compute_diagrams(c, filtration="alpha", homology_dims=(0,)) for c in clouds],
        tau=tau)
    phi = np.stack([_silhouette_of(d) for d in diagrams])[:, None, :]
    meas = np.stack([_measure_of(d) for d in diagrams])

    # diagnostic gaps at the sample level
    gap_sup = float(np.max(np.abs(meas[labels == 1].mean(0)
                                  - meas[labels == 0].mean(0))))
    sil_gap = float(np.max(np.abs(phi[labels == 1, 0].mean(0)
                                  - phi[labels == 0, 0].mean(0))))

    # -- Phase 3 leg: the outcome-level test through the real pipeline
    strata = np.zeros(n, dtype=int)          # fixed-count design -> one stratum
    tseq = np.linspace(INTERVAL[0], INTERVAL[1], RESOLUTION)
    fit = fit_dr(
        (phi, labels, X), tseq, n_basis=N_BASIS, n_folds=N_FOLDS,
        propensity_estimator=LogisticRegression(max_iter=2000,
                                                random_state=_seed("logit", rep)),
        random_state=_seed("fold", rep, n, design),
    )
    outcome_perm = outcome_permutation_test(
        fit, strata, n_perm=n_calibration, seed=_seed("perm", rep, n, design))
    outcome_mult = multiplier_test(
        fit, n_draws=n_calibration, seed=_seed("mult", rep, n, design))

    # -- Phase 4 leg: the distribution-level test (candidate (i), then (ii))
    dfit = fit_dist(
        diagrams, labels, X, pi_known, method="measure",
        interval=INTERVAL, n_bins=N_BINS, weight_power=R, homology_dim=0)
    dist_perm = dist_permutation_test(
        dfit, strata, n_perm=n_calibration,
        seed=_seed("dist-perm", rep, n, design))
    dfit_mmd = fit_dist(
        diagrams, labels, X, pi_known, method="mmd", homology_dim=0)
    K = mmd_kernel_matrix(dfit_mmd)
    mmd_perm = dist_permutation_test(
        dfit_mmd, strata, n_perm=n_calibration,
        seed=_seed("mmd-perm", rep, n, design), kernel_matrix=K)

    return {
        "rep": int(rep), "n": int(n), "design": design,
        "tau": float(tau),
        "n_arm0": int((labels == 0).sum()), "n_arm1": int((labels == 1).sum()),
        "silhouette_gap_sup": sil_gap,
        "measure_l1": gap_sup,
        "outcome_statistic": float(outcome_perm["statistic"]),
        "outcome_permutation_p": float(outcome_perm["pvalue"]),
        "outcome_multiplier_p": float(outcome_mult["pvalue"]),
        "dist_statistic": float(dist_perm["statistic"]),
        "dist_permutation_p": float(dist_perm["pvalue"]),
        "mmd_statistic": float(mmd_perm["statistic"]),
        "mmd_permutation_p": float(mmd_perm["pvalue"]),
    }


# ---------------------------------------------------------------------------
# sharding

def run_shard(shard_idx: int, reps_per_shard: int, *, design: str = "separation",
              n_calibration: int = N_CALIBRATION, workers: int = 1,
              sample_sizes=SAMPLE_SIZES) -> str:
    if design not in ("separation", "reverse"):
        raise ValueError("design must be 'separation' or 'reverse'")
    os.makedirs(SHARDS, exist_ok=True)
    lo = int(shard_idx) * int(reps_per_shard)
    tasks = [(rep, n, design, n_calibration)
             for rep in range(lo, lo + int(reps_per_shard))
             for n in sample_sizes]
    rows = []
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futures = [pool.submit(_one_rep, *task) for task in tasks]
            for f in futures:
                rows.append(f.result())
    else:
        rows = [_one_rep(*task) for task in tasks]
    payload = {
        "design": design, "shard_idx": int(shard_idx),
        "reps": [lo, lo + int(reps_per_shard)],
        "n_calibration": int(n_calibration), "rows": rows,
    }
    path = os.path.join(SHARDS, f"phase4_{design}_shard{shard_idx}.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return path


# ---------------------------------------------------------------------------
# aggregation and figure

def _load_rows(pattern: str):
    rows, seen = [], {}
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            payload = json.load(fh)
        for row in payload["rows"]:
            # Colab appends ``(1)`` when a checkpoint is downloaded twice.
            key = (row.get("rep"), row.get("n"), row.get("design"))
            fingerprint = json.dumps(row, sort_keys=True, separators=(",", ":"))
            if key in seen:
                if seen[key] != fingerprint:
                    raise ValueError(f"conflicting Phase 4 rows for cell {key}")
                continue
            seen[key] = fingerprint
            rows.append(row)
    return rows


PROCEDURES = (
    ("outcome_permutation", "outcome_permutation_p"),
    ("outcome_multiplier", "outcome_multiplier_p"),
    ("dist_permutation", "dist_permutation_p"),
    ("mmd_permutation", "mmd_permutation_p"),
)


def aggregate(design: str = "separation", input_dir: str = SHARDS,
              output: str | None = None) -> str:
    rows = _load_rows(os.path.join(input_dir, f"phase4_{design}_shard*.json"))
    if not rows:
        raise FileNotFoundError(f"no Phase 4 {design} shard files in {input_dir}")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["n"], row["tau"])].append(row)
    summary = []
    for (n, tau), cells in sorted(grouped.items()):
        out = {"n": n, "tau": tau, "replications": len(cells)}
        for name, key in PROCEDURES:
            vals = np.array([c[key] for c in cells])
            rate = float(np.mean(vals < ALPHA))
            out[f"{name}_rejection_rate"] = rate
            out[f"{name}_mc_se"] = float(
                np.sqrt(max(rate * (1.0 - rate), 0.0) / len(vals)))
        for key in ("silhouette_gap_sup", "measure_l1", "dist_statistic",
                    "mmd_statistic"):
            out[key + "_mean"] = float(np.mean([c[key] for c in cells]))
        summary.append(out)
    if output is None:
        output = os.path.join(RESULTS, f"phase4_{design}_summary.json")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w") as fh:
        json.dump({"design": design, "alpha": ALPHA, "rows": summary}, fh,
                  indent=2, sort_keys=True)
    print(json.dumps({"design": design, "cells": len(summary), "output": output},
                     indent=2))
    return output


def figure(separation: str | None = None, reverse: str | None = None,
           output: str | None = None) -> str:
    """Two-panel power curve figure: W1 (left) and W2' (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _load(path):
        with open(path) as fh:
            return json.load(fh)

    sep = _load(separation or os.path.join(RESULTS, "phase4_separation_summary.json"))
    rev = _load(reverse or os.path.join(RESULTS, "phase4_reverse_summary.json"))
    data = {"separation": sep, "reverse": rev, "alpha": ALPHA,
            "note": ("W1: outcome-level power ~ alpha (H0^out true exactly), "
                     "with the multiplier line conservative (0 at all n: the "
                     "realization-invariant outcome leaves the test statistic "
                     "in the lower tail of its own bootstrap null); "
                     "distribution-level power -> 1 (H0^dist false).  W2': "
                     "the expected-measure permutation diagnostic is near alpha "
                     "on this DGP (H0^dist true in law), but is not a weak-null "
                     "size guarantee because the unit-level laws differ; "
                     "outcome-level power -> 1.  The MMD arm fires on both "
                     "witnesses: under candidate (ii) H0^dist is strictly "
                     "stronger, so no reverse case exists there.")}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    styles = [
        ("outcome_permutation", "o-", "#c0392b", "outcome-level (permutation)"),
        ("outcome_multiplier", "s--", "#e67e22", "outcome-level (multiplier)"),
        ("dist_permutation", "^-", "#2980b9", "dist-level: expected measure"),
        ("mmd_permutation", "d-.", "#27ae60", "dist-level: universal-kernel MMD"),
    ]
    titles = {
        "separation": "W1: H$_0^{out}$ true, H$_0^{dist}$ false\n"
                      "cluster splitting, mean silhouette preserved",
        "reverse": "W2': H$_0^{dist}$ true, H$_0^{out}$ false\n"
                   "multiplicity mixture over two locations",
    }
    for ax, (design, payload) in zip(axes, (("separation", sep), ("reverse", rev))):
        rows = sorted(payload["rows"], key=lambda r: r["n"])
        ns = [r["n"] for r in rows]
        for name, fmt, color, label in styles:
            ax.plot(ns, [r[f"{name}_rejection_rate"] for r in rows], fmt,
                    color=color, label=label, markersize=5)
        ax.axhline(ALPHA, color="0.45", ls=":", lw=1)
        ax.text(ns[-1], ALPHA + 0.012, r"$\alpha=0.05$", ha="right", fontsize=8,
                color="0.3")
        ax.set_xlabel("clouds per group $n/2$")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(titles[design], fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("empirical rejection rate")
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    if output is None:
        output = os.path.join(RESULTS, "phase4_figure.png")
    fig.savefig(output, dpi=150)
    plt.close(fig)
    fig_json = os.path.join(RESULTS, "phase4_figure.json")
    with open(fig_json, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    print(json.dumps({"figure": output, "json": fig_json}, indent=2))
    return output


def check_dgp(n: int = 200) -> None:
    """Print the exact witness-law checks the experiment relies on."""
    rng = np.random.default_rng(_seed("check-dgp"))
    for design in ("separation", "reverse"):
        cloud_fn = _w1_cloud if design == "separation" else _w2p_cloud
        labels = np.concatenate([np.zeros(n // 2, dtype=int),
                                 np.ones(n - n // 2, dtype=int)])
        rng.shuffle(labels)
        clouds = [cloud_fn(int(a), rng) for a in labels]
        diagrams = _threshold_diagrams(
            [compute_diagrams(c, filtration="alpha", homology_dims=(0,))
             for c in clouds])
        deaths = {0: [], 1: []}
        counts = {0: [], 1: []}
        for a, d in zip(labels, diagrams):
            deaths[int(a)].extend(d[0][:, 1].tolist())
            counts[int(a)].append(len(d[0]))
        print(f"== {design} ==")
        print("  feature counts per cloud:", {k: sorted(set(v)) for k, v in counts.items()})
        print("  H_0 death values:", {k: sorted(set(v)) for k, v in deaths.items()})
        sil0 = np.stack([_silhouette_of(d) for a, d in zip(labels, diagrams) if a == 0])
        sil1 = np.stack([_silhouette_of(d) for a, d in zip(labels, diagrams) if a == 1])
        meas0 = np.stack([_measure_of(d) for a, d in zip(labels, diagrams) if a == 0])
        meas1 = np.stack([_measure_of(d) for a, d in zip(labels, diagrams) if a == 1])
        print(f"  mean-silhouette sup gap   : {np.max(np.abs(sil1.mean(0) - sil0.mean(0))):.6f}")
        print(f"  expected-measure L1       : {np.abs(meas1.mean(0) - meas0.mean(0)).sum():.6f}")
        if design == "reverse":
            # balanced two-location arm: exact equality of the expected measures
            lo = [np.array([[0.0, 1.0], [0.0, 1.0]])]
            hi = [np.array([[0.0, 2.0], [0.0, 2.0]])]
            m0 = _measure_of([np.array([[0.0, 1.0], [0.0, 2.0]])])
            m1 = 0.5 * _measure_of(lo) + 0.5 * _measure_of(hi)
            s0 = _silhouette_of([np.array([[0.0, 1.0], [0.0, 2.0]])])
            s1 = 0.5 * _silhouette_of(lo) + 0.5 * _silhouette_of(hi)
            print("  balanced W2' measure L1 (exact):",
                  f"{np.abs(m1 - m0).sum():.3e}")
            print("  balanced W2' silhouette gap (exact):",
                  f"{np.max(np.abs(s1 - s0)):.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("shard", "aggregate", "figure",
                                           "smoke", "check-dgp"), required=True)
    parser.add_argument("--design", choices=("separation", "reverse"),
                        default="separation")
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--reps-per-shard", type=int, default=10)
    parser.add_argument("--n-calibration", type=int, default=N_CALIBRATION)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--input-dir", default=SHARDS)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.mode == "shard":
        print(run_shard(args.shard_idx, args.reps_per_shard,
                        design=args.design, n_calibration=args.n_calibration,
                        workers=args.workers))
    elif args.mode == "aggregate":
        aggregate(args.design, args.input_dir, args.output)
    elif args.mode == "figure":
        figure(output=args.output)
    elif args.mode == "check-dgp":
        check_dgp()
    else:
        rows = [_one_rep(0, 25, "separation", n_calibration=39),
                _one_rep(0, 25, "reverse", n_calibration=39)]
        print(json.dumps(rows, indent=2, sort_keys=True))
        print("Phase 4 separation smoke OK")


if __name__ == "__main__":
    main()
