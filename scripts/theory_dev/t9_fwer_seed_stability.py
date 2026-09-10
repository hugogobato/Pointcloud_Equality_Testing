"""Seed stability for the one T9 FWER cell outside the two-sided screen.

The main T9 run records the shared-multiplier FWER at n = 100 with
standardised chi-square scores at 0.0364 (MC SE 0.0037), which is 3.63 exact
MC SE below alpha = 0.05 (3.68 with the rounded MC SE) and so misses the
pre-registered two-sided 3 SE screen (conservative direction).  This script repeats that cell, and its Gaussian
counterpart, on four fresh seeds and writes
``results/theory_validation/t9_fwer_seed_stability.json``.

Reported quantity: per-seed rejection rate and MC SE, the pooled rate over
four seeds, and the fraction of seeds whose two-sided 3 SE screen around
alpha covers the rate.  A stable pooled rate below alpha is finite-sample
conservativeness (consistent with the upper-control theorem); a rate that
crosses alpha across seeds is Monte Carlo noise.  Neither outcome is used to
edit the main criterion.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.theory_dev.t9_multiplicity_checks import ALPHA, mc_se, run_fwer  # noqa: E402

OUT = os.path.join(_ROOT, "results", "theory_validation")


def main():
    reps = 2500
    seeds = (4242, 5252, 6262, 7272)
    rows = []
    for kind in ("gaussian", "chisq"):
        for seed in seeds:
            row = run_fwer(100, reps, seed=seed, kind=kind, rho_deg=0.9)
            rows.append(row)
    summary = {"alpha": ALPHA, "reps": reps, "rows": rows}
    for kind in ("gaussian", "chisq"):
        cells = [r for r in rows if r["kind"] == kind]
        pooled = float(np.mean([r["shared_rate"] for r in cells]))
        summary[f"{kind}_pooled_rate"] = pooled
        summary[f"{kind}_pooled_mc_se"] = mc_se(pooled, reps * len(cells))
        summary[f"{kind}_in_two_sided_3se"] = int(sum(
            abs(r["shared_rate"] - ALPHA) <= 3.0 * r["shared_mc_se"] for r in cells))
    path = os.path.join(OUT, "t9_fwer_seed_stability.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    print(json.dumps(summary, indent=1, sort_keys=True))
    print("wrote", path)


if __name__ == "__main__":
    main()
