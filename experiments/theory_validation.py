"""WP T10: complete the Phase-7 local-power fleet and validate predicted curves.

Deliverables (under ``results/theory_validation/``):
  * ``t10_shards/*.json.gz``           per-task checkpoints of the new replicates
  * ``t10_full_fleet.json``            full 34-cell fleet (observed + predicted)
  * ``t10_local_power_full.png``       DR vs unadjusted vs Gaussian-shift curves
  * ``t10_mismatch_diagnostics.json``  gap table with MC SEs and 2-SE flags
  * ``t10_mismatch_table.md``          the same table, for the WP T7 agent
  * ``t10_validator_phase7.json``      validator self-test on the Phase-7 summary
  * ``t10_validator_full_fleet.json``  validator self-test on the full fleet
  * ``t10_fleet_timing.json``          wall-time accounting

CLI
---
    python experiments/theory_validation.py fleet --workers 8 [--reps 300]
        [--chunk 10] [--scale 0.7] [--limit N] [--dry-run]
    python experiments/theory_validation.py report [--n-draws 20000]
    python experiments/theory_validation.py validate --input PATH [--tol 0.06]
        [--k-se 2.0] [--output PATH]
    python experiments/theory_validation.py all --workers 8

``fleet`` reuses ``experiments/phase7_local_power.py`` unchanged (import, no
copy): it lists the tasks of ``build_tasks(reps, chunk, scale, n_cal)`` that
are not already complete in ``results/phase7_checkpoint.json`` or in the T10
shard directory, and runs the rest, writing one atomic gzip shard per task.
``results/phase7_local_power.json`` and ``results/phase7_checkpoint.json``
are only ever read. Workers are capped at 16 and default to the 8 CPUs
visible in this environment.

Seed protocol
-------------
All seeds derive from ``phase7_local_power.BASE_SEED = 7100`` through
``value -> (value * 7919 + part) % (2**31 - 1)`` with the same part strings
as the original fleet: model ``("model", rep)``, sample
``("sample", setting, rep, n)``, non-prognostic noise
``("nonprog-noise", rep, n)``, random forest ``("rf", setting, rep, n)``,
folds ``("fold", setting, rep, n, tag)``, multiplier draws
``("mult", setting, tag, rep, n)``, frozen permutations
``("perm", setting, tag, rep, n)``, unadjusted permutations
``("unadj", setting, tag, rep, n)``. New replicates therefore carry exactly
the seeds the original run would have used at those replicate indices, with
the shared base draw preserving CRN across directions.

Validator schema
----------------
Input is a JSON object with a ``cells`` list. Native field names (as written
by ``phase7_local_power.aggregate`` and by this module)::

    {
      "cells": [
        {
          "setting": "PROG0",            # required, str
          "direction": "mean",           # "mean"|"bump"|"freq"|"None"
          "n": 500,                      # required, int
          "reps": 300,                   # optional, int
          "dr_multiplier_rate": 0.623,   # observed rate
          "mc_se_dr": 0.028,             # optional; recomputed from reps if absent
          "predicted_dr": 0.628,         # Gaussian-shift prediction, or null
          "predicted_dr_se": 0.0034,     # MC SE of the Gaussian-shift prediction
          "unadjusted_rate": 0.047,
          "mc_se_unadjusted": 0.012,
          "predicted_unadj": 0.076,
          "predicted_unadj_se": 0.002
        }
      ]
    }

A generic cell with ``"observed"``, ``"observed_se"``, ``"predicted"`` and
``"predicted_se"`` is also accepted (reported as test ``generic``). Cells
whose observed or predicted value is missing/null, and null cells without a
predicted limit, are skipped and counted. Per scored row the validator
computes ``gap = observed - predicted``,
``se_gap = sqrt(se_obs**2 + se_pred**2)``, ``z = gap / se_gap``, a 2-SE flag
(``|z| > k_se``) and a tolerance flag (``|gap| > tol``); the run passes iff
no row exceeds ``tol``.

Validator scope vs the pre-registered judge
-------------------------------------------
The validator's all-rows criterion is deliberately wider than the
pre-registered Phase-7 ``judge`` rule (``phase7_local_power.judge``). The
validator scores every directional cell x {DR, unadjusted} row that has a
prediction (51 rows on the full fleet, 39 on the partial Phase-7 summary).
The pre-registered rule scores a subset of those prediction gaps (DR
PROG0/PROG1 at n=500, plus selected unadjusted rows), and additionally
applies the four null-cell size checks (PROG1 unadjusted > 0.20; DR and
unadjusted <= 0.08 for PROG0, NONPROG1 and NONPROG0) and the twelve ARE1
rows (``|dr - unadjusted| <= 0.06`` for NONPROG1/NONPROG0). Consequently:
1. the validator row set is a superset for prediction gaps and can fail on
   rows the pre-registered rule ignores;
2. the validator never evaluates the null-cell size checks or the ARE1
   rows, so it can pass while the pre-registered rule fails; and
3. validator ``passed`` is not the pre-registered verdict.

``validate`` therefore also reports ``pre_registered_confirmed``, copied
from the input's own ``verdict`` block when present, so the two criteria are
recorded together. See ``results/theory_validation/README.md``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import phase7_local_power as p7  # noqa: E402

RESULTS = os.path.normpath(os.path.join(_HERE, "..", "results"))
T10_DIR = os.path.join(RESULTS, "theory_validation")
SHARDS_DIR = os.path.join(T10_DIR, "t10_shards")
PHASE7_CHECKPOINT = os.path.join(RESULTS, "phase7_checkpoint.json")
PHASE7_SUMMARY = os.path.join(RESULTS, "phase7_local_power.json")
FULL_FLEET = os.path.join(T10_DIR, "t10_full_fleet.json")
FULL_FIGURE = os.path.join(T10_DIR, "t10_local_power_full.png")
DIAG_JSON = os.path.join(T10_DIR, "t10_mismatch_diagnostics.json")
DIAG_MD = os.path.join(T10_DIR, "t10_mismatch_table.md")
VALIDATOR_SELFTEST = os.path.join(T10_DIR, "t10_validator_phase7.json")
TIMING_JSON = os.path.join(T10_DIR, "t10_fleet_timing.json")

DEFAULT_SCALE = 0.7
DEFAULT_TOL = 0.06
DEFAULT_K_SE = 2.0
MAX_WORKERS = 16
SETTING_ORDER = ("PROG0", "PROG1", "NONPROG1", "NONPROG0")
DIRECTION_ORDER = ("mean", "bump", "freq")
SEED_PROTOCOL = {
    "base_seed": p7.BASE_SEED,
    "mixer": "value -> (value * 7919 + part) % (2**31 - 1)",
    "model": '("model", rep)',
    "sample": '("sample", setting, rep, n)',
    "nonprog_noise": '("nonprog-noise", rep, n)',
    "random_forest": '("rf", setting, rep, n)',
    "folds": '("fold", setting, rep, n, tag)',
    "multiplier": '("mult", setting, tag, rep, n)',
    "frozen_permutation": '("perm", setting, tag, rep, n)',
    "unadjusted_permutation": '("unadj", setting, tag, rep, n)',
    "prediction_dr": '("pred", setting, tag)',
    "prediction_unadj": '("predU", setting, tag)',
}


def _read_json(path: str):
    with open(path) as fh:
        return json.load(fh)


def _atomic_json(path: str, obj, indent=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=indent, sort_keys=True)
    os.replace(tmp, path)


def _binom_se(p: float, reps: int) -> float:
    return float(math.sqrt(max(p * (1.0 - p), 0.0) / max(int(reps), 1)))


def _task_id(task: dict) -> str:
    return f'{task["setting"]}|{task["n"]}|{task["lo"]}-{task["hi"]}'


def _shard_path(task: dict) -> str:
    return os.path.join(
        SHARDS_DIR,
        f'{task["setting"]}_{task["n"]}_{task["lo"]}-{task["hi"]}.json.gz')


def _shard_task_from_name(name: str) -> str:
    stem = name[: -len(".json.gz")]
    setting, n, rng = stem.rsplit("_", 2)
    return f"{setting}|{n}|{rng}"


def _existing_task_ids() -> set:
    done: set = set()
    if os.path.exists(PHASE7_CHECKPOINT):
        with open(PHASE7_CHECKPOINT) as fh:
            done |= set(json.load(fh).get("done", []))
    if os.path.isdir(SHARDS_DIR):
        done |= {_shard_task_from_name(n) for n in os.listdir(SHARDS_DIR)
                 if n.endswith(".json.gz")}
    return done


def _run_and_save(task: dict) -> dict:
    tid = _task_id(task)
    t0 = time.time()
    try:
        partial = p7.run_task(task)
    except Exception as exc:  # noqa: BLE001 - report, keep the fleet alive
        return {"task_id": tid, "error": repr(exc), "seconds": time.time() - t0}
    tmp = _shard_path(task) + ".tmp"
    with gzip.open(tmp, "wt") as fh:
        json.dump({"task_id": tid, "payload": partial,
                   "seconds": time.time() - t0}, fh,
                  separators=(",", ":"))
    os.replace(tmp, _shard_path(task))
    return {"task_id": tid, "seconds": time.time() - t0,
            "keys": sorted(partial)}


def fleet(workers: int = 8, reps: int = p7.REPS_DEFAULT, chunk: int = 10,
          scale: float = DEFAULT_SCALE, n_calibration: int | None = None,
          limit: int | None = None, dry_run: bool = False):
    """Run every fleet task not already covered by the Phase-7 checkpoint."""
    if workers > MAX_WORKERS:
        raise ValueError(f"workers capped at {MAX_WORKERS} per task spec")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    n_cal = p7.N_CAL if n_calibration is None else int(n_calibration)
    done = _existing_task_ids()
    tasks = [t for t in p7.build_tasks(reps, chunk, scale, n_cal)
             if _task_id(t) not in done]
    if limit is not None:
        tasks = tasks[: int(limit)]
    print(f"fleet: {len(tasks)} pending tasks (phase-7/T10 done: {len(done)}), "
          f"workers={workers}, reps={reps}, chunk={chunk}, scale={scale}, "
          f"n_cal={n_cal}", flush=True)
    if dry_run:
        for t in tasks:
            tags = list(t["directions"]) + (["null"] if t["want_null"] else [])
            print(f"  {_task_id(t)}: {tags}", flush=True)
        return tasks
    if not tasks:
        print("fleet: no pending tasks; leaving the existing timing "
              "artifact untouched "
              f"({os.path.relpath(TIMING_JSON, RESULTS)})", flush=True)
        return {
            "started": None, "workers": workers, "reps": reps, "chunk": chunk,
            "scale": scale, "n_calibration": n_cal, "tasks_total": 0,
            "tasks": [], "failures": [], "timing_written": False,
        }
    os.makedirs(SHARDS_DIR, exist_ok=True)
    timing = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "workers": workers, "reps": reps, "chunk": chunk, "scale": scale,
        "n_calibration": n_cal, "tasks_total": len(tasks),
        "tasks": [], "failures": [],
    }
    t0 = time.time()
    if tasks:
        with mp.Pool(processes=min(workers, len(tasks))) as pool:
            for k, res in enumerate(pool.imap_unordered(_run_and_save, tasks),
                                    start=1):
                if "error" in res:
                    timing["failures"].append(res)
                    print(f"[{k}/{len(tasks)}] FAILED {res['task_id']}: "
                          f"{res['error']}", flush=True)
                else:
                    timing["tasks"].append(res)
                    el = time.time() - t0
                    eta = (el / k) * (len(tasks) - k)
                    print(f"[{k}/{len(tasks)}] {res['task_id']} "
                          f"{res['seconds']:.1f}s elapsed={el/60:.1f}min "
                          f"eta={eta/60:.1f}min", flush=True)
                timing["elapsed_seconds"] = time.time() - t0
                _atomic_json(TIMING_JSON, timing, indent=2)
    timing["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
    timing["elapsed_seconds"] = time.time() - t0
    timing["wall_minutes"] = timing["elapsed_seconds"] / 60.0
    _atomic_json(TIMING_JSON, timing, indent=2)
    print(f"fleet wall time: {timing['wall_minutes']:.1f} min "
          f"({len(timing['failures'])} failures)", flush=True)
    return timing


def _load_agg_sources():
    old = _read_json(PHASE7_CHECKPOINT)
    old_agg = old.get("agg", {})
    shards = {}
    if os.path.isdir(SHARDS_DIR):
        for name in sorted(os.listdir(SHARDS_DIR)):
            if not name.endswith(".json.gz"):
                continue
            with gzip.open(os.path.join(SHARDS_DIR, name), "rt") as fh:
                shard = json.load(fh)
            shards[shard["task_id"]] = shard["payload"]
    return old, old_agg, shards


def _aggregate_sources(old_agg: dict, shards: dict):
    agg = dict(old_agg)
    overlap = []
    for tid, payload in shards.items():
        for key, acc in payload.items():
            if key in agg:
                overlap.append(key)
            agg[key] = acc
    return agg, overlap


def _cells_from_agg(agg: dict, scale: float, n_draws: int):
    merged: dict = {}
    for key, acc in sorted(agg.items()):
        parts = key.split("|")
        setting, tag, n = parts[0], parts[1], int(parts[2])
        dest = merged.setdefault((setting, tag, n), {
            "mult": [], "perm": [], "unadj": [],
            "cov_dr": np.zeros((2 * p7.RESOLUTION, 2 * p7.RESOLUTION)),
            "cov_unadj": np.zeros((2 * p7.RESOLUTION, 2 * p7.RESOLUTION)),
            "count_n": 0, "n": n,
        })
        dest["mult"].extend(acc["mult"])
        dest["perm"].extend(acc["perm"])
        dest["unadj"].extend(acc["unadj"])
        dest["cov_dr"] += np.asarray(acc["cov_dr"], dtype=float)
        dest["cov_unadj"] += np.asarray(acc["cov_unadj"], dtype=float)
        dest["count_n"] += acc["count_n"]

    tseq = p7._tseq()
    cells = []
    for (setting, tag, n), acc in sorted(merged.items(),
                                         key=lambda kv: str(kv[0])):
        mult = np.asarray(acc["mult"], dtype=float)
        perm = np.asarray(acc["perm"], dtype=float)
        unadj = np.asarray(acc["unadj"], dtype=float)
        reps = len(mult)
        tot_n = acc["count_n"]
        sigma_dr = np.asarray(acc["cov_dr"], dtype=float) / tot_n
        sigma_unadj = np.asarray(acc["cov_unadj"], dtype=float) / tot_n
        dr_rate = float(np.mean(mult < p7.ALPHA))
        perm_rate = float(np.mean(perm < p7.ALPHA))
        unadj_rate = float(np.mean(unadj < p7.ALPHA))
        cell = {
            "setting": setting, "direction": tag, "n": n, "reps": reps,
            "dr_multiplier_rate": dr_rate,
            "dr_permutation_rate": perm_rate,
            "unadjusted_rate": unadj_rate,
            "mc_se_dr": _binom_se(dr_rate, reps),
            "mc_se_perm": _binom_se(perm_rate, reps),
            "mc_se_unadj": _binom_se(unadj_rate, reps),
        }
        if tag != "None":
            g = p7.direction_shape(tag, tseq) * scale
            g_vec = np.tile(g, 2)
            pred = p7.predicted_power(
                sigma_dr, g_vec, n_draws=n_draws,
                seed=p7._seed("pred", setting, tag))
            cell["predicted_dr"] = pred["power"]
            cell["predicted_dr_se"] = pred["se"]
            cell["predicted_crit"] = pred["crit"]
            if setting != "PROG1":
                pred_u = p7.predicted_power(
                    sigma_unadj, g_vec, n_draws=n_draws,
                    seed=p7._seed("predU", setting, tag))
                cell["predicted_unadj"] = pred_u["power"]
                cell["predicted_unadj_se"] = pred_u["se"]
            else:
                cell["predicted_unadj"] = None
                cell["predicted_unadj_se"] = None
        cells.append(cell)
    return cells


def _expected_keys() -> set:
    keys = set()
    for setting, direction, n in p7.cells_for(p7.REPS_DEFAULT):
        keys.add((setting, "None" if direction is None else direction, n))
    return keys


def _gap_row(cell: dict, test: str, tol: float, k_se: float):
    if test == "dr":
        obs = cell["dr_multiplier_rate"]
        pred = cell.get("predicted_dr")
        se_obs = cell["mc_se_dr"]
        se_pred = cell.get("predicted_dr_se")
    else:
        obs = cell["unadjusted_rate"]
        pred = cell.get("predicted_unadj")
        se_obs = cell["mc_se_unadj"]
        se_pred = cell.get("predicted_unadj_se")
    if pred is None:
        return None
    se_pred = 0.0 if se_pred is None else float(se_pred)
    se_obs = 0.0 if se_obs is None else float(se_obs)
    gap = float(obs) - float(pred)
    se_gap = math.sqrt(se_obs ** 2 + se_pred ** 2)
    z = (gap / se_gap) if se_gap > 0 else None
    if z is not None:
        flag_2se = bool(abs(z) > k_se)
    else:
        flag_2se = bool(abs(gap) > tol)
    return {
        "setting": cell["setting"], "direction": cell["direction"],
        "n": cell["n"], "reps": cell["reps"], "test": test,
        "observed": float(obs), "predicted": float(pred),
        "se_observed": se_obs, "se_predicted": se_pred, "se_gap": se_gap,
        "gap": gap, "z": z, "flag_2se": flag_2se,
        "within_tol": bool(abs(gap) <= tol),
    }


def _diagnostics(cells, tol: float, k_se: float) -> dict:
    rows = []
    for cell in cells:
        if cell["direction"] == "None":
            continue
        for test in ("dr", "unadjusted"):
            row = _gap_row(cell, test, tol, k_se)
            if row is not None:
                rows.append(row)
    over = [r for r in rows if not r["within_tol"]]
    flagged = [r for r in rows if r["flag_2se"]]
    worst = max(rows, key=lambda r: abs(r["gap"])) if rows else None
    return {
        "tolerance": tol, "k_se": k_se, "n_rows": len(rows),
        "n_over_tolerance": len(over), "n_flagged_2se": len(flagged),
        "max_abs_gap": abs(worst["gap"]) if worst else 0.0,
        "worst_row": worst,
        "rows": rows,
    }


def _write_mismatch_md(diag: dict, verdict: dict, path: str,
                       verdict_change: dict):
    over = [r for r in diag["rows"] if not r["within_tol"]]
    flagged = [r for r in diag["rows"] if r["flag_2se"]]
    lines = [
        "# WP T10 mismatch diagnostics (handoff to WP T7)",
        "",
        f"Gap = observed rejection rate - Gaussian-shift prediction. "
        f"SE(gap) = sqrt(SE_obs^2 + SE_pred^2), where SE_obs is the binomial "
        f"MC SE of the empirical rate and SE_pred is the MC SE of the "
        f"20,000-draw Gaussian-shift power. A row is flagged when "
        f"|gap| > {diag['k_se']:.1f} SE(gap); the pre-registered tolerance is "
        f"{diag['tolerance']:.2f} absolute.",
        "",
        f"Scored rows: {diag['n_rows']}; over tolerance: "
        f"{diag['n_over_tolerance']}; flagged at 2 SE: {diag['n_flagged_2se']}; "
        f"max |gap| = {diag['max_abs_gap']:.3f}.",
        "",
        f"Full-fleet verdict: confirmed={verdict.get('confirmed')}.",
        f"Recorded (partial) verdict confirmed="
        f"{verdict_change.get('recorded_confirmed')}.",
        "",
        "| Setting | Direction | n | reps | Test | Observed | Predicted | "
        "Gap | SE(gap) | z | 2SE? |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in diag["rows"]:
        z = "n/a" if r["z"] is None else f"{r['z']:.2f}"
        lines.append(
            f"| {r['setting']} | {r['direction']} | {r['n']} | {r['reps']} | "
            f"{r['test']} | {r['observed']:.3f} | {r['predicted']:.3f} | "
            f"{r['gap']:+.3f} | {r['se_gap']:.4f} | {z} | "
            f"{'YES' if r['flag_2se'] else 'no'} |")
    if over:
        lines += ["", "Rows over the 0.06 absolute tolerance:"]
        for r in over:
            lines.append(
                f"- {r['setting']}/{r['direction']}/n={r['n']} "
                f"({r['test']}): observed {r['observed']:.3f}, predicted "
                f"{r['predicted']:.3f}, gap {r['gap']:+.3f}, SE "
                f"{r['se_gap']:.4f}.")
    if flagged:
        lines += ["", "Rows beyond 2 SE(gap):"]
        for r in flagged:
            lines.append(
                f"- {r['setting']}/{r['direction']}/n={r['n']} "
                f"({r['test']}): z={r['z']:.2f}, gap {r['gap']:+.3f}.")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _recorded_consistency(cells, recorded_cells) -> dict:
    """Deviation of unchanged cells from the recorded Phase-7 summary."""
    rec = {(c["setting"], c["direction"], c["n"]): c
           for c in (recorded_cells or [])}

    def _dev(field):
        vals = []
        for c in cells:
            r = rec.get((c["setting"], c["direction"], c["n"]))
            if r is None or r.get("reps") != c.get("reps"):
                continue
            a, b = c.get(field), r.get(field)
            if a is None or b is None:
                continue
            vals.append(abs(float(a) - float(b)))
        return max(vals) if vals else None

    return {
        "cells_compared": sum(
            1 for c in cells
            if (c["setting"], c["direction"], c["n"]) in rec
            and rec[(c["setting"], c["direction"], c["n"])].get("reps")
            == c.get("reps")),
        "max_abs_dev_dr_rate": _dev("dr_multiplier_rate"),
        "max_abs_dev_dr_pred": _dev("predicted_dr"),
        "max_abs_dev_unadj_rate": _dev("unadjusted_rate"),
        "max_abs_dev_unadj_pred": _dev("predicted_unadj"),
    }


def _compare_verdicts(old_verdict, new_verdict) -> dict:
    old_notes = list((old_verdict or {}).get("notes", []))
    new_notes = list(new_verdict.get("notes", []))
    return {
        "recorded_confirmed": (old_verdict or {}).get("confirmed"),
        "full_fleet_confirmed": new_verdict.get("confirmed"),
        "same_confirmed": (old_verdict or {}).get("confirmed")
        == new_verdict.get("confirmed"),
        "notes_added": [n for n in new_notes if n not in old_notes],
        "notes_removed": [n for n in old_notes if n not in new_notes],
    }


def report(n_draws: int | None = None, tol: float = DEFAULT_TOL,
           k_se: float = DEFAULT_K_SE):
    """Aggregate Phase-7 checkpoint + T10 shards into the full fleet."""
    t0 = time.time()
    n_draws = p7.N_DRAWS_PRED if n_draws is None else int(n_draws)
    old, old_agg, shards = _load_agg_sources()
    scale = float(old.get("meta", {}).get("scale", DEFAULT_SCALE))
    agg, overlap = _aggregate_sources(old_agg, shards)
    if overlap:
        raise RuntimeError(
            f"{len(overlap)} shard keys duplicate the Phase-7 checkpoint; "
            f"refusing to double count (e.g. {overlap[:3]})")
    cells = _cells_from_agg(agg, scale, n_draws)
    present = {(c["setting"], c["direction"], c["n"]) for c in cells}
    expected = _expected_keys()
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    verdict = p7.judge(cells)
    recorded = _read_json(PHASE7_SUMMARY) if os.path.exists(PHASE7_SUMMARY) else {}
    diagnostics = _diagnostics(cells, tol, k_se)
    verdict_change = _compare_verdicts(recorded.get("verdict"), verdict)
    consistency = _recorded_consistency(cells, recorded.get("cells"))
    out = {
        "meta": {
            "wp": "T10",
            "source_checkpoint": PHASE7_CHECKPOINT,
            "source_summary": PHASE7_SUMMARY,
            "n_shard_files": len(shards),
            "scale": scale, "alpha": p7.ALPHA,
            "tol": tol, "k_se": k_se, "n_draws_pred": n_draws,
            "reps": p7.REPS_DEFAULT, "chunk": 10,
            "missing_cells": [list(m) for m in missing],
            "extra_cells": [list(e) for e in extra],
            "report_seconds": None,
        },
        "seed_protocol": SEED_PROTOCOL,
        "cells": cells,
        "diagnostics": diagnostics,
        "verdict": verdict,
        "recorded_verdict": recorded.get("verdict"),
        "verdict_change": verdict_change,
        "recorded_consistency": consistency,
    }
    out["meta"]["report_seconds"] = time.time() - t0
    _atomic_json(FULL_FLEET, out, indent=2)
    _atomic_json(DIAG_JSON, diagnostics, indent=2)
    _write_mismatch_md(diagnostics, verdict, DIAG_MD, verdict_change)
    make_full_figure(FULL_FLEET, FULL_FIGURE)
    print(f"report: {len(cells)} cells, missing={len(missing)}, "
          f"verdict confirmed={verdict.get('confirmed')}, "
          f"over_tol={diagnostics['n_over_tolerance']}, "
          f"flagged_2se={diagnostics['n_flagged_2se']}", flush=True)
    return out


def make_full_figure(summary_path: str = FULL_FLEET, out: str = FULL_FIGURE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    summary = _read_json(summary_path)
    cells = summary["cells"]

    def get(setting, direction, n):
        for c in cells:
            if (c["setting"] == setting and c["direction"] == direction
                    and c["n"] == n):
                return c
        return None

    fig = plt.figure(figsize=(13.5, 11.0), layout="constrained")
    gs = fig.add_gridspec(5, 3, height_ratios=(1, 1, 1, 1, 0.9))
    for i, setting in enumerate(SETTING_ORDER):
        for j, direction in enumerate(DIRECTION_ORDER):
            ax = fig.add_subplot(gs[i, j])
            rows = [(n, get(setting, direction, n))
                    for n in (100, 200, 500)]
            rows = [(n, c) for n, c in rows if c is not None]
            if not rows:
                ax.set_title(f"{setting} / {direction} (pending)", fontsize=10)
                continue
            ns = [n for n, _ in rows]
            dr = [c["dr_multiplier_rate"] for _, c in rows]
            dr_se = [c["mc_se_dr"] for _, c in rows]
            un = [c["unadjusted_rate"] for _, c in rows]
            un_se = [c["mc_se_unadj"] for _, c in rows]
            pm = [c["dr_permutation_rate"] for _, c in rows]
            pred = [c["predicted_dr"] for _, c in rows]
            pred_u = [c.get("predicted_unadj") for _, c in rows]
            ax.errorbar(ns, dr, yerr=dr_se, fmt="o-", color="C0", capsize=3,
                        label="DR multiplier (emp)")
            ax.errorbar(ns, un, yerr=un_se, fmt="x:", color="C1", capsize=3,
                        label="unadjusted (emp)")
            ax.plot(ns, pred, "k--", marker="s", ms=3, lw=1.2, alpha=0.75,
                    label="Gaussian-shift (DR)")
            if all(v is not None for v in pred_u):
                ax.plot(ns, pred_u, ":", color="gray", marker="^", ms=3,
                        lw=1.2, label="Gaussian-shift (unadj)")
            ax.plot(ns, pm, "-.", color="C2", lw=1.0, alpha=0.55,
                    label="DR frozen-perm (emp)")
            ax.set_xscale("log")
            ax.set_xticks(ns)
            ax.set_xticklabels([str(n) for n in ns])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_ylim(-0.04, 1.04)
            ax.grid(alpha=0.3)
            ax.set_title(f"{setting} / {direction}", fontsize=10)
            if j == 0:
                ax.set_ylabel("rejection rate")
            if i == 3:
                ax.set_xlabel("n")
            if i == 0 and j == 0:
                ax.legend(fontsize=6.5, loc="upper left")

    sub = gs[4, :].subgridspec(1, 4, wspace=0.3)
    for j, setting in enumerate(SETTING_ORDER):
        ax = fig.add_subplot(sub[0, j])
        c = get(setting, "None", 200)
        if c is None:
            ax.set_title(f"{setting} null (pending)", fontsize=9)
            continue
        dr, dr_se = c["dr_multiplier_rate"], c["mc_se_dr"]
        un, un_se = c["unadjusted_rate"], c["mc_se_unadj"]
        ax.bar([0], [dr], width=0.55, yerr=[dr_se], capsize=3, color="C0",
               label="DR")
        ax.bar([1], [un], width=0.55, yerr=[un_se], capsize=3, color="C1",
               label="unadjusted")
        ax.axhline(0.05, color="0.3", lw=0.8, ls=":")
        ax.axhline(0.08, color="0.3", lw=0.8, ls="--")
        ax.axhline(0.20, color="0.3", lw=0.8, ls="-.")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["DR", "unadj"])
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.3, axis="y")
        ax.set_title(f"{setting} null n=200", fontsize=9)
        if j == 0:
            ax.set_ylabel("size")
    fig.suptitle(
        "Local-power fleet: rejection rate under local alternatives "
        "$\\psi^{(n)} = n^{-1/2} g$; error bars are $\\pm1$ MC SE, dashed/dotted "
        "black curves are Gaussian-shift predictions",
        fontsize=10)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    try:
        from google.colab import files  # noqa: F401
        files.download(out)
    except Exception:
        pass


SETTING_LABEL = {
    "PROG0": "prognostic, randomized",
    "PROG1": "prognostic, confounded",
    "NONPROG0": "non-prognostic, randomized",
    "NONPROG1": "non-prognostic, confounded",
}
DIRECTION_LABEL = {"mean": "mean shift", "bump": "bump shift",
                   "freq": "frequency shift"}
MAIN_SETTING_COLORS = {
    "PROG0": "#0072B2", "PROG1": "#D55E00",
    "NONPROG0": "#009E73", "NONPROG1": "#E69F00",
}


def make_main_figure(summary_path: str = FULL_FLEET, out: str | None = None):
    """Compact main-text local-power figure (size, power, prediction check)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = _read_json(summary_path)
    cells = summary["cells"]

    def get(setting, direction, n):
        for c in cells:
            if (c["setting"] == setting and c["direction"] == direction
                    and c["n"] == n):
                return c
        return None

    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.6))

    ax = axes[0, 0]
    settings = ("PROG0", "PROG1", "NONPROG0", "NONPROG1")
    x = np.arange(len(settings))
    width = 0.36
    for i, (key, lab, color) in enumerate((("dr_multiplier_rate", "DR multiplier",
                                            "#0072B2"),
                                           ("unadjusted_rate", "unadjusted",
                                            "#D55E00"))):
        vals, errs = [], []
        for s in settings:
            c = get(s, "None", 200)
            vals.append(c[key])
            errs.append(c["mc_se_dr"] if "dr" in key else c["mc_se_unadj"])
        ax.bar(x + (i - 0.5) * width, vals, width, color=color, label=lab,
               yerr=2 * np.asarray(errs), capsize=2,
               error_kw={"elinewidth": 0.7})
    ax.axhline(0.05, color="0.3", lw=0.8, ls=":")
    ax.axhline(0.08, color="0.3", lw=0.8, ls="--")
    ax.text(3.42, 0.115, "nominal $\\alpha=0.05$\nupper edge $0.08$", fontsize=7,
            ha="right", va="bottom", color="0.3")
    ax.set_xticks(x)
    ax.set_xticklabels([SETTING_LABEL[s].replace(", ", ",\n") for s in settings],
                       fontsize=8)
    ax.set_ylabel("rejection rate at $\\alpha=0.05$")
    ax.set_ylim(0, 1.0)
    ax.set_title("(a) size under the global null ($n=200$)", fontsize=10.5)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.25)

    handles = {}
    for ax, setting in ((axes[0, 1], "PROG1"), (axes[1, 0], "PROG0")):
        for direction, color in (("mean", "#0072B2"), ("bump", "#009E73"),
                                 ("freq", "#D55E00")):
            ns, dr, drse, un, unse = [], [], [], [], []
            for n in (100, 200, 500):
                c = get(setting, direction, n)
                if c is None:
                    continue
                ns.append(n)
                dr.append(c["dr_multiplier_rate"])
                drse.append(c["mc_se_dr"])
                un.append(c["unadjusted_rate"])
                unse.append(c["mc_se_unadj"])
            h = ax.errorbar(ns, dr, yerr=2 * np.asarray(drse), marker="o",
                            color=color, capsize=2, elinewidth=0.7,
                            label=DIRECTION_LABEL[direction])
            ax.errorbar(ns, un, yerr=2 * np.asarray(unse), marker="s",
                        ls=":", color=color, alpha=0.75, capsize=2,
                        elinewidth=0.7)
            handles.setdefault(direction, h.lines[0])
        ax.axhline(0.05, color="0.4", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks([100, 200, 500])
        ax.set_xticklabels(["100", "200", "500"])
        ax.set_xlabel("clouds per group $n$")
        ax.set_ylabel("rejection rate at $\\alpha=0.05$")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"(b) power vs $n$: {SETTING_LABEL[setting]}"
                     if setting == "PROG1" else
                     f"(c) power vs $n$: {SETTING_LABEL[setting]}", fontsize=10.5)
        ax.grid(alpha=0.25)

    ax = axes[1, 1]
    markers = {"mean": "o", "bump": "s", "freq": "^"}
    for setting in ("PROG0", "PROG1", "NONPROG0", "NONPROG1"):
        for direction, marker in markers.items():
            px, py, pe = [], [], []
            for n in (100, 200, 500):
                c = get(setting, direction, n)
                if c is None or c.get("predicted_dr") is None:
                    continue
                px.append(c["predicted_dr"])
                py.append(c["dr_multiplier_rate"])
                pe.append(c["mc_se_dr"])
            ax.errorbar(px, py, yerr=2 * np.asarray(pe), fmt=marker, ms=4,
                        color=MAIN_SETTING_COLORS[setting], alpha=0.85,
                        capsize=2, elinewidth=0.6,
                        label=SETTING_LABEL[setting] if direction == "mean"
                        else None)
    ax.plot([0, 1.05], [0, 1.05], "k--", lw=0.9)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Gaussian-shift predicted power")
    ax.set_ylabel("DR empirical power")
    ax.set_title("(d) prediction fidelity (all cells)", fontsize=10.5)
    ax.text(0.98, 0.04, "markers: circle = mean, square = bump, "
            "triangle = frequency", fontsize=8, ha="right", va="bottom",
            color="0.35", transform=ax.transAxes)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.25)

    fig.legend([handles[d] for d in ("mean", "bump", "freq")],
               [DIRECTION_LABEL[d] for d in ("mean", "bump", "freq")],
               ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               frameon=False, fontsize=8.5,
               title="solid: DR multiplier, dotted: unadjusted",
               title_fontsize=8)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    out = out or FULL_FIGURE.replace("t10_local_power_full",
                                     "t10_local_power_main")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def _cell_label(cell: dict) -> str:
    return (f"{cell.get('setting')}/{cell.get('direction')}/"
            f"n={cell.get('n')}")


def _score_cells(cells: list, tol: float, k_se: float):
    rows, skipped = [], []
    for c in cells:
        if "dr_multiplier_rate" in c:
            specs = [
                ("dr", "dr_multiplier_rate", "mc_se_dr",
                 "predicted_dr", "predicted_dr_se"),
                ("unadjusted", "unadjusted_rate", "mc_se_unadjusted",
                 "predicted_unadj", "predicted_unadj_se"),
            ]
        elif "observed" in c:
            specs = [("generic", "observed", "observed_se",
                      "predicted", "predicted_se")]
        else:
            skipped.append({"cell": _cell_label(c),
                            "reason": "unrecognised schema"})
            continue
        for test, obs_key, se_key, pred_key, pred_se_key in specs:
            obs = c.get(obs_key)
            pred = c.get(pred_key)
            if obs is None or pred is None:
                skipped.append({"cell": _cell_label(c), "test": test,
                                "reason": "observed or predicted missing"})
                continue
            se_obs = c.get(se_key)
            if se_obs is None:
                reps = c.get("reps")
                se_obs = _binom_se(float(obs), int(reps)) if reps else 0.0
            se_pred = c.get(pred_se_key)
            se_pred = 0.0 if se_pred is None else float(se_pred)
            gap = float(obs) - float(pred)
            se_gap = math.sqrt(float(se_obs) ** 2 + se_pred ** 2)
            z = (gap / se_gap) if se_gap > 0 else None
            flag_2se = (abs(z) > k_se) if z is not None else (
                abs(gap) > tol)
            rows.append({
                "setting": c.get("setting"), "direction": c.get("direction"),
                "n": c.get("n"), "reps": c.get("reps"), "test": test,
                "observed": float(obs), "predicted": float(pred),
                "se_observed": float(se_obs), "se_predicted": se_pred,
                "se_gap": se_gap, "gap": gap, "z": z,
                "flag_2se": bool(flag_2se),
                "within_tol": bool(abs(gap) <= tol),
            })
    return rows, skipped


def validate(input_path: str, tol: float = DEFAULT_TOL,
             k_se: float = DEFAULT_K_SE, output: str | None = None):
    """Score observed-vs-predicted gaps of a predicted-curves JSON.

    ``passed`` applies this module's all-rows criterion (every scored
    prediction-gap row must be within ``tol``); it is not the pre-registered
    ``phase7_local_power.judge`` verdict. When the input carries a ``verdict``
    block, its ``confirmed`` value is emitted as ``pre_registered_confirmed``
    alongside the validator result. See the module docstring for the scope
    split.
    """
    payload = _read_json(input_path)
    cells = payload.get("cells", [])
    verdict = payload.get("verdict")
    pre_confirmed = (
        bool(verdict.get("confirmed"))
        if isinstance(verdict, dict) and "confirmed" in verdict else None)
    rows, skipped = _score_cells(cells, tol, k_se)
    over = [r for r in rows if not r["within_tol"]]
    flagged = [r for r in rows if r["flag_2se"]]
    result = {
        "input": os.path.abspath(input_path),
        "validator_scope": "all_scored_prediction_gap_rows",
        "pre_registered_confirmed": pre_confirmed,
        "pre_registered_criterion": "phase7_local_power.judge",
        "tol": tol, "k_se": k_se,
        "n_cells": len(cells), "n_scored": len(rows),
        "n_skipped": len(skipped),
        "passed": len(over) == 0,
        "n_over_tolerance": len(over),
        "n_flagged_2se": len(flagged),
        "max_abs_gap": max((abs(r["gap"]) for r in rows), default=0.0),
        "rows": rows, "skipped": skipped,
    }
    if output:
        _atomic_json(output, result, indent=2)
    print(f"validate {input_path}: scored={len(rows)} "
          f"over_tol={len(over)} flagged_2se={len(flagged)} "
          f"PASS={'yes' if result['passed'] else 'no'} "
          f"pre_registered_confirmed={pre_confirmed}", flush=True)
    for r in over:
        z = "n/a" if r["z"] is None else f"{r['z']:.2f}"
        print(f"  OVER  {r['setting']}/{r['direction']}/n={r['n']} "
              f"({r['test']}): obs={r['observed']:.3f} "
              f"pred={r['predicted']:.3f} gap={r['gap']:+.3f} "
              f"se={r['se_gap']:.4f} z={z}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fleet", help="complete missing Phase-7 cells")
    pf.add_argument("--workers", type=int, default=8)
    pf.add_argument("--reps", type=int, default=p7.REPS_DEFAULT)
    pf.add_argument("--chunk", type=int, default=10)
    pf.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    pf.add_argument("--n-calibration", type=int, default=p7.N_CAL)
    pf.add_argument("--limit", type=int, default=None)
    pf.add_argument("--dry-run", action="store_true")

    pr = sub.add_parser("report", help="aggregate full fleet + figure + tables")
    pr.add_argument("--n-draws", type=int, default=p7.N_DRAWS_PRED)
    pr.add_argument("--tol", type=float, default=DEFAULT_TOL)
    pr.add_argument("--k-se", type=float, default=DEFAULT_K_SE)

    pv = sub.add_parser(
        "validate",
        help="score observed-vs-predicted rows under the all-rows criterion",
        description=(
            "Score every observed-vs-predicted row that has a prediction and "
            "pass iff none exceeds --tol. This all-rows criterion is not the "
            "pre-registered phase7_local_power.judge rule: the validator row "
            "set is a superset for prediction gaps (it can fail on rows the "
            "judge ignores) and it omits the judge's null-cell size checks "
            "and twelve ARE1 checks (so it can pass while the judge fails). "
            "The input's own verdict, when present, is echoed as "
            "pre_registered_confirmed in the output JSON."))
    pv.add_argument("--input", required=True)
    pv.add_argument("--tol", type=float, default=DEFAULT_TOL)
    pv.add_argument("--k-se", type=float, default=DEFAULT_K_SE)
    pv.add_argument("--output", default=None)

    pa = sub.add_parser("all", help="fleet + report + validator self-test")
    pa.add_argument("--workers", type=int, default=8)
    pa.add_argument("--reps", type=int, default=p7.REPS_DEFAULT)
    pa.add_argument("--chunk", type=int, default=10)
    pa.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    pa.add_argument("--tol", type=float, default=DEFAULT_TOL)
    pa.add_argument("--k-se", type=float, default=DEFAULT_K_SE)

    args = parser.parse_args()
    if args.cmd == "fleet":
        if args.workers > MAX_WORKERS:
            parser.error(f"--workers must be <= {MAX_WORKERS}")
        fleet(workers=args.workers, reps=args.reps, chunk=args.chunk,
              scale=args.scale, n_calibration=args.n_calibration,
              limit=args.limit, dry_run=args.dry_run)
    elif args.cmd == "report":
        report(n_draws=args.n_draws, tol=args.tol, k_se=args.k_se)
    elif args.cmd == "validate":
        result = validate(args.input, tol=args.tol, k_se=args.k_se,
                          output=args.output)
        sys.exit(0 if result["passed"] else 1)
    elif args.cmd == "all":
        if args.workers > MAX_WORKERS:
            parser.error(f"--workers must be <= {MAX_WORKERS}")
        fleet(workers=args.workers, reps=args.reps, chunk=args.chunk,
              scale=args.scale, n_calibration=p7.N_CAL)
        report(tol=args.tol, k_se=args.k_se)
        validate(FULL_FLEET, tol=args.tol, k_se=args.k_se,
                 output=os.path.join(T10_DIR, "t10_validator_full_fleet.json"))
        validate(PHASE7_SUMMARY, tol=args.tol, k_se=args.k_se,
                 output=VALIDATOR_SELFTEST)


if __name__ == "__main__":
    main()
