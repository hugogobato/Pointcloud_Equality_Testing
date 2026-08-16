"""Phase 3 experiments for the outcome-level doubly robust test.

The default ``oracle`` design is the controlled functional benchmark from
``tcda_uq.datasets.TriOracleSimulation``.  It is intentionally fast enough to
produce the required size/power tables at n in {50, 100, 200, 500}.  The
``clouds`` design runs the same cached testing layer on the project's
covariate-driven point-cloud DGP, but is substantially slower because
persistent homology is computed once per replication.

Long runs are sharded by replication index:

    python experiments/phase3_dr_calibration.py --mode shard \
        --shard-idx 0 --reps-per-shard 10

The shard JSON files are independent checkpoints.  After the fleet is
complete, aggregate them with ``--mode aggregate``.  No permutation draw
recomputes persistent homology, cross-fitting, or a nuisance regression.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from tda2s.dgp import CloudSampleDGP, to_silhouette_sample
from tda2s.tests.dr_outcome import (
    equivalence_test,
    fit_dr,
    multiplier_test,
    positivity_diagnostics,
    propensity_learner_grid,
    propensity_strata,
    stratified_permutation_test,
)

BASE_SEED = 3100
ALPHA = 0.05
SAMPLE_SIZES = (50, 100, 200, 500)
REGIMES = (0.0, 0.5, 1.0)
N_BASIS = 5
N_FOLDS = 2
N_CALIBRATION = 399
INTERVAL = (0.0, 1.0)
RESOLUTION = 50
_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, "..", "results")
SHARDS = os.path.join(RESULTS, "phase3_shards")


def _seed(*parts) -> int:
    value = BASE_SEED
    for part in parts:
        if isinstance(part, str):
            part = sum((i + 1) * byte for i, byte in enumerate(part.encode()))
        value = (value * 7919 + int(part)) % (2 ** 31 - 1)
    return int(value)


def _interaction_features(X):
    """Features matching TriOracleSimulation's logistic propensity law."""
    X = np.asarray(X)
    return np.column_stack([X, X[:, 1] * X[:, 2], X[:, 0] * X[:, 2]])


def _tri_sample(n: int, rep: int, regime: float, *, alternative: bool,
                true_n_basis: int = N_BASIS):
    """Draw a tri-oracle sample, optionally imposing the sharp outcome null."""
    sim = __import__("tcda_uq.datasets", fromlist=["TriOracleSimulation"]).TriOracleSimulation(
        n_cov=3, n_hom_dim=2, resolution=RESOLUTION, interval=INTERVAL,
        n_basis=true_n_basis, coef_scale=0.6, noise_scale=0.15,
        prop_scale=1.5 * regime, seed=_seed("model", rep),
    )
    if not alternative:
        # The model class is unchanged, but both potential-outcome mean
        # coefficients are made equal before sampling.  This is a controlled
        # sharp null with known TATE == 0 and preserves covariate-driven noise.
        for d in range(sim.n_hom_dim):
            sim.Gamma[1][d] = sim.Gamma[0][d].copy()
    return sim.sample(n, rng=_seed("sample", rep, n, int(100 * regime), int(alternative)))


def _one_fit(sample, rep: int, n: int, regime: float, *, alternative: bool,
             n_calibration: int = N_CALIBRATION, n_bins: int = 8,
             n_basis: int = N_BASIS, propensity_estimator=None,
             propensity_feature_fn=None, include_forms: bool = False):
    fit = fit_dr(
        sample.observed, sample.tseq, n_basis=n_basis, n_folds=N_FOLDS,
        propensity_estimator=(propensity_estimator if propensity_estimator is not None else
                              RandomForestClassifier(
                                  n_estimators=100, min_samples_leaf=4,
                                  n_jobs=1, random_state=_seed("rf", rep, n))),
        random_state=_seed("fold", rep, n, int(100 * regime), int(alternative)),
        propensity_feature_fn=propensity_feature_fn,
    )
    # The strata are formed from the known design propensity in the oracle
    # benchmark.  This is deliberately stronger than estimated quantile
    # strata, and is the clean calibration reference.  In the cloud design
    # the fitted propensity strata are used instead and reported as such.
    strata_original = propensity_strata(sample.propensity, n_bins=n_bins)
    multiplier = multiplier_test(
        fit, n_draws=n_calibration, seed=_seed("mult", rep, n, int(100 * regime), int(alternative)),
    )
    permutation = stratified_permutation_test(
        fit, strata_original, n_perm=n_calibration,
        seed=_seed("perm", rep, n, int(100 * regime), int(alternative)),
    )
    equivalence = equivalence_test(
        fit, margin=0.25, n_draws=n_calibration,
        seed=_seed("equiv", rep, n, int(100 * regime), int(alternative)),
    )
    pos = positivity_diagnostics(fit)
    row = {
        "rep": int(rep), "n": int(n), "regime": float(regime),
        "alternative": bool(alternative),
        "multiplier_p": float(multiplier["pvalue"]),
        "permutation_p": float(permutation["pvalue"]),
        "statistic": float(multiplier["statistic"]),
        "estimate_sup": float(np.max(np.abs(multiplier["estimate"]))),
        "equivalence_p": float(equivalence["pvalue"]),
        "equivalence_reject_non_equivalence": bool(equivalence["reject_non_equivalence"]),
        "strata": "known_propensity",
        **{f"positivity_{k}": float(v) for k, v in pos.items()
           if isinstance(v, (int, float, np.integer, np.floating))},
    }
    # The raw sup statistic is the preregistered primary test.  The optional
    # diagnostics address the Phase 2 observation that n and statistic form
    # matter: L2, raw sup studentization, and L2 studentization use the same
    # fitted nuisances and strata, not fresh data.  They are off for the main
    # 500-replication fleet to keep the long run focused and bounded.
    if include_forms:
        for norm, studentized in (("l2", False), ("sup", True), ("l2", True)):
            tag = f"{norm}_{'studentized' if studentized else 'raw'}"
            m = multiplier_test(
                fit, n_draws=n_calibration, studentize=studentized, norm=norm,
                seed=_seed("mult", tag, rep, n, int(100 * regime), int(alternative)),
            )
            p = stratified_permutation_test(
                fit, strata_original, n_perm=n_calibration, studentize=studentized,
                norm=norm,
                seed=_seed("perm", tag, rep, n, int(100 * regime), int(alternative)),
            )
            row[f"multiplier_{tag}_p"] = float(m["pvalue"])
            row[f"permutation_{tag}_p"] = float(p["pvalue"])
    return row


def _oracle_replication(rep: int, sample_sizes=SAMPLE_SIZES,
                        regimes=REGIMES, n_calibration=N_CALIBRATION,
                        include_forms: bool = False):
    rows = []
    for n in sample_sizes:
        for regime in regimes:
            for alternative in (False, True):
                sample = _tri_sample(n, rep, regime, alternative=alternative)
                rows.append(_one_fit(sample, rep, n, regime,
                                     alternative=alternative,
                                     n_calibration=n_calibration,
                                     include_forms=include_forms))
    return rows


def _cloud_replication(rep: int, n: int = 100, regime: float = 1.0,
                       alternative: bool = False, n_calibration: int = 399,
                       include_forms: bool = False):
    dgp = CloudSampleDGP(
        n_per_group=n // 2, m=120, d_x=3,
        beta=np.array([-0.5, -0.1, 0.6]), prop_scale=1.5 * regime,
        group_effect=1 if alternative else 0,
        seed=_seed("cloud-model", rep, n),
    )
    sample = dgp.sample(rng=_seed("cloud-sample", rep, n, int(100 * regime), int(alternative)))
    phi, A, X = to_silhouette_sample(
        sample.clouds, sample.X, sample.A, filtration="alpha",
        homology_dims=(0, 1), interval=(0.0, 2.0), r=3.0, resolution=RESOLUTION,
    )
    # Propensity strata use the DGP's known design probability.  A production
    # analysis should pre-register how these are obtained when e(X) is unknown.
    observed = type("Observed", (), {
        "observed": (phi, A, X), "tseq": np.linspace(0.0, 2.0, RESOLUTION),
        "propensity": sample.propensity,
    })()
    return _one_fit(observed, rep, n, regime, alternative=alternative,
                    n_calibration=n_calibration, n_bins=8,
                    include_forms=include_forms)


def learner_sweep(rep: int, n: int = 200, regime: float = 1.0,
                  n_calibration: int = N_CALIBRATION,
                  include_forms: bool = False):
    """Task 3.3 benchmark: four propensity learners on one oracle design."""
    sample = _tri_sample(n, rep, regime, alternative=False)
    rows = []
    for name, estimator in propensity_learner_grid(seed=_seed("learner", rep)).items():
        # The grid is deterministic; cloning inside tcda_uq makes each fold
        # receive a fresh estimator.  Use the matching interaction features for
        # the logistic reference, while the flexible learners use raw X.
        feature_fn = _interaction_features if name == "logistic" else None
        row = _one_fit(sample, rep, n, regime, alternative=False,
                       n_calibration=n_calibration,
                       propensity_estimator=estimator,
                       propensity_feature_fn=feature_fn,
                       include_forms=include_forms)
        row["learner"] = name
        rows.append(row)
    return rows


def double_robustness_stress(rep: int, n: int = 200, regime: float = 1.0,
                             n_calibration: int = N_CALIBRATION):
    """Task 3.4: correct/misspecified e and mu configurations.

    The data-generating outcome mean has nine Fourier terms, while the main
    benchmark uses five.  The stress suite therefore has a deliberate
    outcome-regression misspecification when it fits three terms.  The
    propensity law contains two interactions, so raw-X logistic regression is
    the deliberate propensity misspecification; the interaction feature map is
    the correctly specified parametric reference.
    """
    sample = _tri_sample(n, rep, regime, alternative=False, true_n_basis=9)
    logistic = LogisticRegression(max_iter=2000, C=1e6, random_state=_seed("stress", rep))
    cases = {
        "both_correct": (9, logistic, _interaction_features),
        "propensity_misspecified": (9, logistic, None),
        "outcome_misspecified": (3, logistic, _interaction_features),
        "both_misspecified": (3, logistic, None),
    }
    rows = []
    for name, (n_basis, estimator, feature_fn) in cases.items():
        row = _one_fit(
            sample, rep, n, regime, alternative=False,
            n_calibration=n_calibration, n_basis=n_basis,
            propensity_estimator=estimator,
            propensity_feature_fn=feature_fn,
        )
        row["stress_case"] = name
        rows.append(row)
    return rows


def run_shard(shard_idx: int, reps_per_shard: int, *, design: str = "oracle",
              n_calibration: int = N_CALIBRATION, cloud_n: int = 100,
              include_forms: bool = False):
    os.makedirs(SHARDS, exist_ok=True)
    lo = int(shard_idx) * int(reps_per_shard)
    hi = lo + int(reps_per_shard)
    rows = []
    for rep in range(lo, hi):
        if design == "oracle":
            rows.extend(_oracle_replication(
                rep, n_calibration=n_calibration, include_forms=include_forms))
        elif design == "clouds":
            for alternative in (False, True):
                rows.append(_cloud_replication(
                    rep, n=cloud_n, regime=1.0, alternative=alternative,
                    n_calibration=n_calibration, include_forms=include_forms))
        elif design == "learners":
            rows.extend(learner_sweep(rep, n=cloud_n,
                                      n_calibration=n_calibration,
                                      include_forms=include_forms))
        elif design == "stress":
            rows.extend(double_robustness_stress(
                rep, n=cloud_n, n_calibration=n_calibration))
        else:
            raise ValueError("design must be oracle, clouds, learners, or stress")
    payload = {
        "design": design, "shard_idx": int(shard_idx),
        "reps": [lo, hi], "n_calibration": int(n_calibration),
        "rows": rows,
    }
    path = os.path.join(SHARDS, f"phase3_{design}_shard{shard_idx}.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return path


def _load_rows(pattern: str):
    import glob
    rows = []
    seen = {}
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            payload = json.load(fh)
        for row in payload["rows"]:
            # Colab appends ``(1)`` when a checkpoint is downloaded twice.
            # Treat an identical replication-cell record as the same result,
            # but fail loudly if two files disagree for that cell.
            key = (row.get("rep"), row.get("n"), row.get("regime"),
                   row.get("alternative"), row.get("learner"),
                   row.get("stress_case"))
            fingerprint = json.dumps(row, sort_keys=True, separators=(",", ":"))
            previous = seen.get(key)
            if previous is not None:
                if previous != fingerprint:
                    raise ValueError(
                        f"conflicting Phase 3 rows for replication cell {key}"
                    )
                continue
            seen[key] = fingerprint
            rows.append(row)
    return rows


def aggregate(design: str = "oracle", input_dir: str = SHARDS,
              output: str | None = None):
    rows = _load_rows(os.path.join(input_dir, f"phase3_{design}_shard*.json"))
    if not rows:
        raise FileNotFoundError(f"no Phase 3 {design} shard files in {input_dir}")
    grouped = defaultdict(list)
    for row in rows:
        key = (row.get("n"), row.get("regime"), row.get("alternative"),
               row.get("learner"), row.get("stress_case"))
        grouped[key].append(row)
    summary = []
    for (n, regime, alternative, learner, stress_case), cells in sorted(
            grouped.items(), key=str):
        out = {
            "n": n, "regime": regime, "alternative": alternative,
            "learner": learner, "stress_case": stress_case,
            "replications": len(cells),
        }
        for method in ("multiplier", "permutation"):
            vals = np.array([c[f"{method}_p"] for c in cells])
            out[f"{method}_rejection_rate"] = float(np.mean(vals < ALPHA))
            out[f"{method}_pvalue_mean"] = float(np.mean(vals))
            out[f"{method}_pvalue_mc_se"] = float(np.sqrt(max(out[f"{method}_rejection_rate"] *
                                                               (1 - out[f"{method}_rejection_rate"]), 0.0) /
                                                         len(vals)))
        if "equivalence_reject_non_equivalence" in cells[0]:
            out["equivalence_rejection_rate"] = float(np.mean([
                c["equivalence_reject_non_equivalence"] for c in cells]))
            out["equivalence_pvalue_mean"] = float(np.mean([
                c["equivalence_p"] for c in cells]))
        for key in ("positivity_min_pi", "positivity_max_pi",
                    "positivity_ess_treated", "positivity_ess_control"):
            if key in cells[0]:
                out[key + "_mean"] = float(np.mean([c[key] for c in cells]))
        summary.append(out)
    if output is None:
        output = os.path.join(RESULTS, f"phase3_{design}_summary.json")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w") as fh:
        json.dump({"design": design, "alpha": ALPHA, "rows": summary}, fh,
                  indent=2, sort_keys=True)
    print(json.dumps({"design": design, "rows": len(summary), "output": output}, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("shard", "aggregate", "smoke"), required=True)
    parser.add_argument("--design", choices=("oracle", "clouds", "learners", "stress"), default="oracle")
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--reps-per-shard", type=int, default=10)
    parser.add_argument("--n-calibration", type=int, default=N_CALIBRATION)
    parser.add_argument("--cloud-n", type=int, default=100)
    parser.add_argument("--include-forms", action="store_true",
                        help="also record L2 and studentized calibration diagnostics")
    parser.add_argument("--input-dir", default=SHARDS)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.mode == "shard":
        path = run_shard(args.shard_idx, args.reps_per_shard,
                          design=args.design, n_calibration=args.n_calibration,
                          cloud_n=args.cloud_n, include_forms=args.include_forms)
        print(path)
    elif args.mode == "aggregate":
        aggregate(args.design, args.input_dir, args.output)
    else:
        # A bounded, end-to-end smoke that exercises every calibration path.
        rows = _oracle_replication(0, sample_sizes=(50,), regimes=(0.0,),
                                    n_calibration=39)
        print(json.dumps(rows, indent=2, sort_keys=True))
        print("Phase 3 smoke OK")


if __name__ == "__main__":
    main()
