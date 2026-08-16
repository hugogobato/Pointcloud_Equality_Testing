"""Phase 3.5 benchmark: degree multiplicity procedures on a shared null.

Task 3.5.4 compares four multiplicity procedures for the family of homology
degrees:

* Bonferroni over the per-degree permutation/multiplier p-values;
* the Phase 3 unstudentized shared max-statistic;
* the Vejdemo-Johansson--Mukherjee studentized empirical-null comparator with
  the exchangeability-preserving pooled standardization;
* the same comparator with the source's null-only standardization.

All four read *one* null matrix per replication, produced by
``tda2s.tests.dr_outcome.degree_null_statistics``, so the differences between
them are differences of procedure and not Monte Carlo noise.  Both null
mechanisms (frozen-nuisance stratified permutation and shared multiplier) are
recorded for every cell.

Designs
-------
``fwer``     the primary decision: null cells at n in {50,100,200,500} and
             three propensity regimes, 500 replications.
``power``    validity and power under a deliberate per-degree scale imbalance,
             with a single-degree alternative that only degree 1 carries.
``learners`` the four propensity learners of task 3.3, null only.
``stress``   the four double-robustness misspecification cases of task 3.4,
             null only.

Sharded exactly like the Phase 3 fleet::

    python experiments/phase3_5_vjm.py --mode shard --design fwer \
        --shard-idx 0 --reps-per-shard 10

Replication seeds are functions of the replication index alone, so shards
concatenate into the sequential result however the work is split.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from experiments.phase3_dr_calibration import (
    ALPHA,
    INTERVAL,
    N_BASIS,
    N_CALIBRATION,
    N_FOLDS,
    RESOLUTION,
    _interaction_features,
    _seed,
)
from tda2s.tests.dr_outcome import (
    fit_dr,
    propensity_learner_grid,
    propensity_strata,
    vjm_multiplicity_test,
)

SAMPLE_SIZES = (50, 100, 200, 500)
REGIMES = (0.0, 0.5, 1.0)
POWER_SAMPLE_SIZES = (50, 100)
# Multiplies degree 0's silhouettes.  The AIPW fit is exactly equivariant under
# this rescaling, so it changes nothing about the estimand and everything about
# whether the degrees are on a comparable scale -- which is the incomparability
# the source's standardization exists to remove.
DEGREE_SCALES = (1.0, 8.0)
# Constant shift added to degree 1's outcome-mean intercept under the
# alternative.  Psi[0] is identically 1, so psi_1(t) = ALT_SHIFT and psi_0 == 0
# exactly: the alternative lives in one degree only.  0.15 was chosen from a
# 50-replication pilot so that balanced-scale power is about 0.50 at n = 50 and
# 0.76 at n = 100, well away from the ceiling where procedures stop separating.
ALT_SHIFT = 0.15
MECHANISMS = ("permutation", "multiplier")
# Seed tags deliberately match ``phase3_dr_calibration._one_fit``.  The Phase 3
# oracle null cells and the Phase 3.5 ``fwer`` cells then share the fitted
# nuisances *and* the calibration draws, so this driver's shared-max column
# reproduces the published Phase 3 p-values exactly rather than merely
# agreeing in distribution.  See ``check_phase3_agreement``.
SEED_TAGS = {"permutation": "perm", "multiplier": "mult"}
_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, "..", "results")
SHARDS = os.path.join(RESULTS, "phase3_5_shards")


def _vjm_sample(n: int, rep: int, regime: float, *, alternative: bool,
                true_n_basis: int = N_BASIS, shift: float = ALT_SHIFT,
                n_hom_dim: int = 2):
    """Tri-oracle sample under the sharp null or a single-degree alternative.

    ``shift`` is an explicit argument rather than a module global so that a
    worker process cannot silently run a different effect size from the one
    the caller set.
    """
    sim = __import__("tcda_uq.datasets", fromlist=["TriOracleSimulation"]).TriOracleSimulation(
        n_cov=3, n_hom_dim=n_hom_dim, resolution=RESOLUTION, interval=INTERVAL,
        n_basis=true_n_basis, coef_scale=0.6, noise_scale=0.15,
        prop_scale=1.5 * regime, seed=_seed("model", rep),
    )
    for d in range(sim.n_hom_dim):
        sim.Gamma[1][d] = sim.Gamma[0][d].copy()
    if alternative:
        sim.Gamma[1][1][0, 0] += shift
    return sim.sample(n, rng=_seed("sample", rep, n, int(100 * regime),
                                   int(alternative)))


def _scaled_observed(sample, degree_scale: float):
    phi, A, X = sample.observed
    phi = np.array(phi, dtype=float)
    if degree_scale != 1.0:
        phi[:, 0, :] *= degree_scale
    return phi, A, X


def _one_cell(sample, rep: int, n: int, regime: float, *, alternative: bool,
              degree_scale: float = 1.0, n_calibration: int = N_CALIBRATION,
              n_bins: int = 8, n_basis: int = N_BASIS,
              propensity_estimator=None, propensity_feature_fn=None) -> dict:
    """One fit, one null matrix per mechanism, four procedures read off both."""
    if propensity_estimator is None:
        # tcda_uq's default propensity model is an unseeded random forest, so
        # the estimator is always passed explicitly here; otherwise a shard
        # would not reproduce.
        propensity_estimator = RandomForestClassifier(
            n_estimators=100, min_samples_leaf=4, n_jobs=1,
            random_state=_seed("rf", rep, n))
    fit = fit_dr(
        _scaled_observed(sample, degree_scale), sample.tseq, n_basis=n_basis,
        n_folds=N_FOLDS, propensity_estimator=propensity_estimator,
        random_state=_seed("fold", rep, n, int(100 * regime), int(alternative)),
        propensity_feature_fn=propensity_feature_fn,
    )
    strata = propensity_strata(sample.propensity, n_bins=n_bins)
    row = {
        "rep": int(rep), "n": int(n), "regime": float(regime),
        "alternative": bool(alternative), "degree_scale": float(degree_scale),
        "n_hom_dim": int(fit.n_hom_dim), "strata": "known_propensity",
    }
    for mechanism in MECHANISMS:
        # The unscaled cells reuse the Phase 3 seed exactly; only a rescaled
        # family needs its own stream, and it gets one that cannot collide.
        seed_parts = ["scale" + str(int(10 * degree_scale))] if degree_scale != 1.0 else []
        out = vjm_multiplicity_test(
            fit, mechanism=mechanism,
            strata=(strata if mechanism == "permutation" else None),
            n_draws=n_calibration, alpha=ALPHA,
            seed=_seed(SEED_TAGS[mechanism], *seed_parts, rep, n,
                       int(100 * regime), int(alternative)),
        )
        prefix = f"{mechanism}_"
        row[prefix + "vjm_pooled_p"] = float(out["conventions"]["pooled"]["pvalue"])
        row[prefix + "vjm_source_p"] = float(out["conventions"]["source"]["pvalue"])
        row[prefix + "bonferroni_p"] = float(out["bonferroni_pvalue"])
        row[prefix + "shared_max_p"] = float(out["shared_max_pvalue"])
        row[prefix + "max_pairwise_ks"] = float(out["comparability"]["max_pairwise_ks"])
        row[prefix + "vjm_fdr_n_rejected"] = int(
            len(out["conventions"]["pooled"]["fdr"]["rejected_degrees"]))
        for d in range(fit.n_hom_dim):
            row[f"{prefix}per_degree_p{d}"] = float(out["per_degree_pvalue"][d])
            row[f"{prefix}null_sd{d}"] = float(out["comparability"]["raw_null_sd"][d])
            row[f"{prefix}observed{d}"] = float(out["observed"][d])
    return row


def _fwer_replication(rep: int, n_calibration: int = N_CALIBRATION):
    return [
        _one_cell(_vjm_sample(n, rep, regime, alternative=False), rep, n, regime,
                  alternative=False, n_calibration=n_calibration)
        for n in SAMPLE_SIZES for regime in REGIMES
    ]


def _power_replication(rep: int, n_calibration: int = N_CALIBRATION,
                       n_hom_dim: int = 2, sample_sizes=POWER_SAMPLE_SIZES):
    rows = []
    for n in sample_sizes:
        for alternative in (False, True):
            sample = _vjm_sample(n, rep, 1.0, alternative=alternative,
                                 n_hom_dim=n_hom_dim)
            for degree_scale in DEGREE_SCALES:
                rows.append(_one_cell(
                    sample, rep, n, 1.0, alternative=alternative,
                    degree_scale=degree_scale, n_calibration=n_calibration))
    return rows


def _degrees3_replication(rep: int, n_calibration: int = N_CALIBRATION):
    """The plan's d in {0,1,2} family.

    The other designs use the two-degree family of the Phase 3 oracle fleet so
    that their shared-max column reproduces the published Phase 3 numbers.
    This design widens the family to three degrees, which is where the
    Bonferroni penalty and the dependence the comparator preserves both start
    to matter.
    """
    return _power_replication(rep, n_calibration=n_calibration, n_hom_dim=3,
                              sample_sizes=(100,))


def _learner_replication(rep: int, n: int = 200, regime: float = 1.0,
                         n_calibration: int = N_CALIBRATION):
    sample = _vjm_sample(n, rep, regime, alternative=False)
    rows = []
    for name, estimator in propensity_learner_grid(seed=_seed("learner", rep)).items():
        feature_fn = _interaction_features if name == "logistic" else None
        row = _one_cell(sample, rep, n, regime, alternative=False,
                        n_calibration=n_calibration,
                        propensity_estimator=estimator,
                        propensity_feature_fn=feature_fn)
        row["learner"] = name
        rows.append(row)
    return rows


def _stress_replication(rep: int, n: int = 200, regime: float = 1.0,
                        n_calibration: int = N_CALIBRATION):
    """Task 3.4's misspecification grid, reused as a Phase 3.5 null."""
    sample = _vjm_sample(n, rep, regime, alternative=False, true_n_basis=9)
    logistic = LogisticRegression(max_iter=2000, C=1e6, random_state=_seed("stress", rep))
    cases = {
        "both_correct": (9, logistic, _interaction_features),
        "propensity_misspecified": (9, logistic, None),
        "outcome_misspecified": (3, logistic, _interaction_features),
        "both_misspecified": (3, logistic, None),
    }
    rows = []
    for name, (n_basis, estimator, feature_fn) in cases.items():
        row = _one_cell(sample, rep, n, regime, alternative=False,
                        n_calibration=n_calibration, n_basis=n_basis,
                        propensity_estimator=estimator,
                        propensity_feature_fn=feature_fn)
        row["stress_case"] = name
        rows.append(row)
    return rows


DESIGNS = {
    "fwer": _fwer_replication,
    "power": _power_replication,
    "degrees3": _degrees3_replication,
    "learners": _learner_replication,
    "stress": _stress_replication,
}


def run_shard(shard_idx: int, reps_per_shard: int, *, design: str = "fwer",
              n_calibration: int = N_CALIBRATION):
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {sorted(DESIGNS)}")
    os.makedirs(SHARDS, exist_ok=True)
    lo = int(shard_idx) * int(reps_per_shard)
    rows = []
    for rep in range(lo, lo + int(reps_per_shard)):
        rows.extend(DESIGNS[design](rep, n_calibration=n_calibration))
    payload = {
        "design": design, "shard_idx": int(shard_idx),
        "reps": [lo, lo + int(reps_per_shard)],
        "n_calibration": int(n_calibration), "rows": rows,
    }
    path = os.path.join(SHARDS, f"phase3_5_{design}_shard{shard_idx}.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return path


def _cell_key(row):
    return (row.get("rep"), row.get("n"), row.get("regime"),
            row.get("alternative"), row.get("degree_scale"),
            row.get("n_hom_dim"), row.get("learner"), row.get("stress_case"))


def _load_rows(pattern: str):
    rows, seen = [], {}
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            payload = json.load(fh)
        for row in payload["rows"]:
            # Colab appends ``(1)`` when a checkpoint is downloaded twice.
            # Identical records are the same result; disagreeing ones are a bug.
            key = _cell_key(row)
            fingerprint = json.dumps(row, sort_keys=True, separators=(",", ":"))
            if key in seen:
                if seen[key] != fingerprint:
                    raise ValueError(f"conflicting Phase 3.5 rows for cell {key}")
                continue
            seen[key] = fingerprint
            rows.append(row)
    return rows


PROCEDURES = ("vjm_pooled", "vjm_source", "bonferroni", "shared_max")


def aggregate(design: str = "fwer", input_dir: str = SHARDS,
              output: str | None = None):
    rows = _load_rows(os.path.join(input_dir, f"phase3_5_{design}_shard*.json"))
    if not rows:
        raise FileNotFoundError(f"no Phase 3.5 {design} shard files in {input_dir}")
    grouped = defaultdict(list)
    for row in rows:
        grouped[_cell_key(row)[1:]].append(row)
    summary = []
    for (n, regime, alternative, degree_scale, n_hom_dim, learner,
         stress_case), cells in sorted(grouped.items(), key=str):
        out = {
            "n": n, "regime": regime, "alternative": alternative,
            "degree_scale": degree_scale, "n_hom_dim": n_hom_dim,
            "learner": learner, "stress_case": stress_case,
            "replications": len(cells),
        }
        for mechanism in MECHANISMS:
            for procedure in PROCEDURES:
                key = f"{mechanism}_{procedure}_p"
                if key not in cells[0]:
                    continue
                vals = np.array([c[key] for c in cells])
                rate = float(np.mean(vals < ALPHA))
                out[f"{mechanism}_{procedure}_rejection_rate"] = rate
                out[f"{mechanism}_{procedure}_mc_se"] = float(
                    np.sqrt(max(rate * (1.0 - rate), 0.0) / len(vals)))
            keys = ["max_pairwise_ks"] + [f"null_sd{d}" for d in range(n_hom_dim or 0)]
            for key in keys:
                full = f"{mechanism}_{key}"
                if full in cells[0]:
                    out[f"{full}_mean"] = float(np.mean([c[full] for c in cells]))
        summary.append(out)
    if output is None:
        output = os.path.join(RESULTS, f"phase3_5_{design}_summary.json")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w") as fh:
        json.dump({"design": design, "alpha": ALPHA, "band": [0.03, 0.08],
                   "rows": summary}, fh, indent=2, sort_keys=True)
    print(json.dumps({"design": design, "cells": len(summary), "output": output},
                     indent=2))
    return output


def check_phase3_agreement(phase3_dir: str, reps=(0, 1, 2), sizes=None,
                           atol: float = 0.0):
    """Reproduce published Phase 3 p-values from this driver, exactly.

    The Phase 3.5 ``fwer`` cells share the Phase 3 oracle null design, the
    replication seeds, the fold seeds, the propensity seed and the calibration
    seeds.  The shared max-statistic read off the Phase 3.5 null matrix must
    therefore equal the Phase 3 fleet's ``permutation_p`` and ``multiplier_p``
    exactly, not merely in distribution.  Any drift in the estimator, the
    calibration draws or the seed derivation breaks this check, which makes it
    the cheapest available regression test over the whole pipeline.

    Returns the list of compared cells; raises on any mismatch.
    """
    published = {}
    for path in sorted(glob.glob(os.path.join(phase3_dir, "phase3_oracle_shard*.json"))):
        with open(path) as fh:
            payload = json.load(fh)
        for row in payload["rows"]:
            if row["alternative"] or row["rep"] not in reps:
                continue
            if sizes is not None and row["n"] not in sizes:
                continue
            published[(row["rep"], row["n"], row["regime"])] = row
    if not published:
        raise FileNotFoundError(f"no Phase 3 oracle shards with reps {reps} in {phase3_dir}")

    compared = []
    for (rep, n, regime), reference in sorted(published.items()):
        row = _one_cell(_vjm_sample(n, rep, regime, alternative=False), rep, n,
                        regime, alternative=False)
        for mechanism, key in (("permutation", "permutation_p"),
                               ("multiplier", "multiplier_p")):
            got = row[f"{mechanism}_shared_max_p"]
            want = reference[key]
            if abs(got - want) > atol:
                raise AssertionError(
                    f"Phase 3 disagreement at rep={rep} n={n} regime={regime} "
                    f"{mechanism}: phase3.5 {got} vs phase3 {want}")
        compared.append({"rep": rep, "n": n, "regime": regime,
                         "permutation_p": row["permutation_shared_max_p"],
                         "multiplier_p": row["multiplier_shared_max_p"]})
    print(json.dumps({"cells_compared": len(compared), "all_exact": True},
                     indent=2))
    return compared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",
                        choices=("shard", "aggregate", "smoke", "check-phase3"),
                        required=True)
    parser.add_argument("--design", choices=tuple(DESIGNS), default="fwer")
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--reps-per-shard", type=int, default=10)
    parser.add_argument("--n-calibration", type=int, default=N_CALIBRATION)
    parser.add_argument("--input-dir", default=SHARDS)
    parser.add_argument("--phase3-dir",
                        default=os.path.join(_HERE, "colab"),
                        help="directory holding the Phase 3 oracle shard JSON files")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.mode == "shard":
        print(run_shard(args.shard_idx, args.reps_per_shard, design=args.design,
                        n_calibration=args.n_calibration))
    elif args.mode == "aggregate":
        aggregate(args.design, args.input_dir, args.output)
    elif args.mode == "check-phase3":
        check_phase3_agreement(args.phase3_dir)
    else:
        rows = [_one_cell(_vjm_sample(50, 0, 1.0, alternative=False), 0, 50, 1.0,
                          alternative=False, n_calibration=39)]
        print(json.dumps(rows, indent=2, sort_keys=True))
        print("Phase 3.5 smoke OK")


if __name__ == "__main__":
    main()
