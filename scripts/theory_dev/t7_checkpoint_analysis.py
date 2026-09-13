"""T7: evaluate the local-bias correction against the full Phase-7 checkpoint.

Reads ``results/phase7_checkpoint.json`` (full 300-rep core cells), rebuilds the
per-cell pooled DR covariance exactly as ``phase7_local_power.aggregate`` does,
and compares three Gaussian-shift predictions with the recorded multiplier
rejection rate:

  naive      : shift g                       (the fleet's prediction);
  corrected  : shift g + delta (g - P_K g)   (delta = E[e/ehat]-1 from the
               T7 probe, P_K the projection onto the regression Fourier basis);
  unadjusted : not relevant here.

No parameter is fitted to any rejection rate: delta comes from a separate
probe run with the same phase-7 seeds, and P_K is known from the basis.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))
if os.path.join(ROOT, "experiments") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase7_local_power import (  # noqa: E402
    ALPHA,
    RESOLUTION,
    _seed,
    _tseq,
    direction_shape,
    predicted_power,
)
from t7_probe import gauss_crit_power, regression_projection  # noqa: E402

Q = 2 * RESOLUTION


def load_checkpoint(path: str) -> dict:
    t0 = time.time()
    with open(path) as fh:
        payload = json.load(fh)
    print(f"checkpoint loaded in {time.time() - t0:.1f}s", flush=True)
    return payload


def merge_cells(agg: dict) -> dict:
    merged: dict = {}
    for key, acc in sorted(agg.items()):
        parts = key.split("|")
        setting, tag, n = parts[0], parts[1], int(parts[2])
        mkey = f"{setting}|{n}|{tag}"
        dest = merged.setdefault(mkey, {
            "setting": setting, "direction": None if tag == "None" else tag,
            "n": n, "mult": [], "perm": [], "unadj": [],
            "cov_dr": np.zeros((Q, Q)), "cov_unadj": np.zeros((Q, Q)),
            "count_n": 0,
        })
        dest["mult"].extend(acc["mult"])
        dest["perm"].extend(acc["perm"])
        dest["unadj"].extend(acc["unadj"])
        dest["cov_dr"] += np.asarray(acc["cov_dr"], dtype=float)
        dest["cov_unadj"] += np.asarray(acc["cov_unadj"], dtype=float)
        dest["count_n"] += acc["count_n"]
    return merged


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=os.path.join(
        ROOT, "results", "phase7_checkpoint.json"))
    ap.add_argument("--probe", nargs="+", default=[
        os.path.join(ROOT, "results", "theory_validation", "t7_core_n500b.json"),
        os.path.join(ROOT, "results", "theory_validation", "t7_ndecay_prog1_freq.json")])
    ap.add_argument("--scale", type=float, default=0.7)
    ap.add_argument("--out", default=os.path.join(
        ROOT, "results", "theory_validation", "t7_checkpoint_corrected.json"))
    args = ap.parse_args()

    payload = load_checkpoint(args.checkpoint)
    merged = merge_cells(payload["agg"])
    probe: dict = {}
    for path in args.probe:
        if os.path.exists(path):
            data = json.load(open(path))
            for key, variants in data.get("summary", {}).items():
                probe.setdefault(key, {})
                probe[key].update(variants)

    tseq = _tseq()
    rows = []
    for key, cell in sorted(merged.items()):
        setting, n, direction = cell["setting"], cell["n"], cell["direction"]
        if direction is None:
            continue
        mult = np.asarray(cell["mult"])
        R = len(mult)
        sigma = cell["cov_dr"] / cell["count_n"]
        g = direction_shape(direction, tseq) * args.scale
        g_vec = np.tile(g, 2)
        naive = predicted_power(sigma, g_vec, alpha=ALPHA,
                                seed=_seed("pred", setting, direction))
        row = {
            "cell": key, "setting": setting, "direction": direction,
            "n": n, "reps": R,
            "mult_rate": float(np.mean(mult < ALPHA)),
            "mc_se": float(np.sqrt(max(
                float(np.mean(mult < ALPHA))
                * (1.0 - float(np.mean(mult < ALPHA))), 0.0) / R)),
            "naive_gauss_power": naive["power"],
            "naive_gap": float(np.mean(mult < ALPHA)) - naive["power"],
        }
        pcell = probe.get(key, {}).get("fit")
        if pcell is not None:
            delta = pcell["mean_e_over_ehat_mean"] - 1.0
            Pg = regression_projection(g, 5)
            g_eff = np.tile(g + delta * (g - Pg), 2)
            corr = predicted_power(sigma, g_eff, alpha=ALPHA,
                                   seed=_seed("corr", setting, direction))
            row.update({
                "delta_hat": delta,
                "out_of_span_frac": float(np.linalg.norm(g - Pg)
                                          / max(np.linalg.norm(g), 1e-12)),
                "corrected_gauss_power": corr["power"],
                "corrected_gap": float(np.mean(mult < ALPHA)) - corr["power"],
                "probe_mult_rate": pcell.get("mult_rate"),
                "probe_reps": pcell.get("reps"),
            })
        rows.append(row)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"rows": rows}, fh, indent=2, sort_keys=True)
    print(f"wrote {args.out}")
    for row in rows:
        if row["n"] < 500:
            continue
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in row.items()}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
