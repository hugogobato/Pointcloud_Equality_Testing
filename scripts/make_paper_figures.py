"""Render the submission figure set into ``paper/figures/``.

Every panel is produced from the JSON payloads already committed under
``results/``; no experiment is re-run and no quantity is recomputed from
shards.  The renderers are the plotting functions that live next to each
experiment (``experiments/phase*.py`` and ``scripts/theory_dev/t*.py``), so a
later change to the workflow propagates to the paper figures.  The two
appendix figures for T6/T7 are rendered here because those artifacts never
had a plotting function.

Usage:
    rtk uv run python scripts/make_paper_figures.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "paper" / "figures"
RESULTS = ROOT / "results"
T10 = RESULTS / "theory_validation"

for _p in (ROOT / "experiments", ROOT / "scripts" / "theory_dev",
           ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _say(name, path):
    print(f"[paper-figures] {name} -> {path}")


def render_fig1():
    import phase2_imbalance_sweep as p2
    data = _load(RESULTS / "phase2_figure1.json")
    out = FIGDIR / "fig1_imbalance_validity.png"
    p2.make_figure(data["sweep_rates"], data["masking_rates"],
                   data["n_reps"]["sweep"], out=out)
    _say("fig1_imbalance_validity", out)


def render_fig2():
    import phase4_separation as p4
    out = FIGDIR / "fig2_c2_separation.png"
    p4.figure(separation=str(RESULTS / "phase4_separation_summary.json"),
              reverse=str(RESULTS / "phase4_reverse_summary.json"),
              output=str(out))
    _say("fig2_c2_separation", out)


def render_fig3():
    import phase45_weak_null as p45
    payload = _load(RESULTS / "phase45_aggregate.json")
    out = FIGDIR / "fig3_calibration_scope.png"
    p45.make_figure(payload=payload, out=str(out))
    _say("fig3_calibration_scope", out)


def render_fig4():
    import phase8_master_grid as p8
    payload = _load(RESULTS / "phase8_master.json")
    out = FIGDIR / "fig4_bakeoff.png"
    p8.figure(payload=payload, out=str(out))
    _say("fig4_bakeoff", out)


def render_fig5_and_appendix_t10():
    import theory_validation as tv
    out_main = FIGDIR / "fig5_local_power.png"
    tv.make_main_figure(tv.FULL_FLEET, out=str(out_main))
    _say("fig5_local_power", out_main)
    out_full = FIGDIR / "appendix_t10_full_fleet.png"
    tv.make_full_figure(tv.FULL_FLEET, out=str(out_full))
    _say("appendix_t10_full_fleet", out_full)


def render_appendix_t2():
    import t2_leakage_validate as t2
    recorded = _load(T10 / "t2_leakage_check.json")
    raw_sweep = {}
    for test, c in recorded["sweep_checks"].items():
        raw_sweep[test] = {f"lam{lab}": rate for lab, rate
                           in zip(t2.LAM_LABELS, c["rates"])}
    raw_sweep["dr"] = {f"lam{lab}": rate for lab, rate in
                       zip(t2.LAM_LABELS, recorded["dr_checks"]["sweep_rates"])}
    raw_masking = {t: c["rate"]
                   for t, c in recorded["masking_checks"].items()}
    raw_masking["dr"] = recorded["dr_checks"]["masking_power"]
    exact = [t2.exact_law(float(lam)) for lam in t2.LAMBDAS]
    out = FIGDIR / "appendix_c1_masking_mechanism.png"
    t2.make_figure(exact, raw_sweep, raw_masking, str(out))
    _say("appendix_c1_masking_mechanism", out)


def render_appendix_t3():
    import t3_local_limit_figure as t3
    run = _load(T10 / "t3_local_limit_check.json")
    p7 = _load(RESULTS / "phase7_local_power.json")
    out = FIGDIR / "appendix_t3_local_limit.png"
    t3.make_figure(run, p7, out=str(out))
    _say("appendix_t3_local_limit", out)


def render_appendix_t4():
    import t4_efficiency_checks as t4
    payload = _load(T10 / "t4_efficiency_checks.json")
    out = FIGDIR / "appendix_t4_efficiency.png"
    t4.make_figure(payload, out)
    _say("appendix_t4_efficiency", out)


def render_appendix_t5():
    import t5_are_validation as t5
    payload = _load(T10 / "t5_are_validation.json")
    out = FIGDIR / "appendix_t5_are.png"
    t5.make_figure(payload["C3"]["table"], payload["C5"], str(out))
    _say("appendix_t5_are", out)


def render_appendix_t6():
    """T6 local-grid power: empirical vs Gaussian-shift prediction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = _load(T10 / "t6_local_summary.json")
    cells = payload["cells"]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for n, color, marker in ((200, "#0072B2", "o"), (500, "#D55E00", "s")):
        hs, emp, ese, pred = [], [], [], []
        for h in (1.0, 2.0, 6.0):
            c = cells.get(f"n={n}|h={h}")
            if not c:
                continue
            hs.append(h)
            emp.append(c["rate"])
            ese.append(c["mc_se"])
            pred.append(c["prediction"]["power"])
        ax.errorbar(hs, emp, yerr=2 * np.asarray(ese), marker=marker,
                    color=color, capsize=3, elinewidth=0.7,
                    label=f"empirical, $n={n}$")
        ax.plot(hs, pred, ls="--", marker=marker, mfc="none", color=color,
                label=f"predicted, $n={n}$")
    ax.set_xscale("log")
    ax.set_xticks([1.0, 2.0, 6.0])
    ax.set_xticklabels(["1", "2", "6"])
    ax.set_xlabel("local shift size $h$")
    ax.set_ylabel("rejection rate at $\\alpha=0.05$")
    ax.set_ylim(-0.02, 0.45)
    ax.set_title("Local-grid power: empirical vs Gaussian-shift prediction",
                 fontsize=9)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = FIGDIR / "appendix_t6_distgrid_local.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    _say("appendix_t6_distgrid_local", out)


def render_appendix_t7():
    """T7 mismatch: naive vs corrected Gaussian-shift predictions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = _load(T10 / "t7_checkpoint_corrected.json")
    rows = payload["rows"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    colors = {"PROG0": "#0072B2", "PROG1": "#D55E00",
              "NONPROG0": "#009E73", "NONPROG1": "#E69F00"}
    markers = {"mean": "o", "bump": "s", "freq": "^"}
    y = []
    seen = set()
    for r in rows:
        lab = r["setting"] if r["setting"] not in seen else None
        seen.add(r["setting"])
        ax.errorbar(r["naive_gauss_power"], r["mult_rate"],
                    yerr=2 * r["mc_se"], fmt=markers[r["direction"]],
                    color=colors[r["setting"]], alpha=0.85, capsize=2,
                    elinewidth=0.6, label=lab)
        if "corrected_gauss_power" in r:
            ax.plot([r["corrected_gauss_power"]], [r["mult_rate"]], marker="x",
                    ms=6, color=colors[r["setting"]])
            ax.annotate("", xy=(r["corrected_gauss_power"], r["mult_rate"]),
                        xytext=(r["naive_gauss_power"], r["mult_rate"]),
                        arrowprops=dict(arrowstyle="->", lw=0.7,
                                        color=colors[r["setting"]]))
        y.append((f'{r["setting"]} {r["direction"]} n={r["n"]}',
                  r["naive_gap"], r.get("corrected_gap")))
    ax.plot([0, 1], [0, 1], "k--", lw=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("predicted power (naive; arrow tip = corrected)")
    ax.set_ylabel("empirical DR power")
    ax.set_title("(a) prediction fidelity", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    labels = [t[0] for t in y]
    naive_gap = np.asarray([t[1] for t in y])
    corr = np.asarray([np.nan if t[2] is None else t[2] for t in y])
    idx = np.arange(len(y))
    ax2.barh(idx - 0.2, naive_gap, height=0.4, color="#D55E00",
             label="naive")
    ax2.barh(idx + 0.2, corr, height=0.4, color="#0072B2", label="corrected")
    ax2.axvline(0.0, color="k", lw=0.8, ls="--")
    ax2.set_yticks(idx)
    ax2.set_yticklabels(labels, fontsize=6.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("empirical $-$ predicted rejection rate")
    ax2.set_title("(b) local-bias gap", fontsize=9)
    ax2.legend(fontsize=7.5)
    ax2.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out = FIGDIR / "appendix_t7_mismatch.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    _say("appendix_t7_mismatch", out)


def render_appendix_t8():
    import t8_minimax_checks as t8
    payload = _load(T10 / "t8_minimax_checks.json")
    out = FIGDIR / "appendix_t8_minimax.png"
    t8.make_figure(payload["le_cam"], payload["fixed_grid_rate"],
                   payload["smooth_ball_rate"], payload["baseline"], out)
    _say("appendix_t8_minimax", out)


def render_appendix_t9():
    import t9_multiplicity_checks as t9
    payload = _load(T10 / "t9_multiplicity_checks.json")
    out = t9.write_figure(payload["power"], out_dir=str(FIGDIR),
                          filename="appendix_t9_multiplicity.png")
    _say("appendix_t9_multiplicity", out)


def render_appendix_explainer():
    import generate_phase4_distribution_explainer as ex
    out = FIGDIR / "appendix_distribution_explainer.png"
    ex.OUT = out
    ex.main()
    _say("appendix_distribution_explainer", out)


RENDERERS = (
    ("fig1", render_fig1),
    ("fig2", render_fig2),
    ("fig3", render_fig3),
    ("fig4", render_fig4),
    ("fig5+t10", render_fig5_and_appendix_t10),
    ("t2", render_appendix_t2),
    ("t3", render_appendix_t3),
    ("t4", render_appendix_t4),
    ("t5", render_appendix_t5),
    ("t6", render_appendix_t6),
    ("t7", render_appendix_t7),
    ("t8", render_appendix_t8),
    ("t9", render_appendix_t9),
    ("explainer", render_appendix_explainer),
)


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    failures = []
    for name, fn in RENDERERS:
        try:
            fn()
        except Exception as exc:  # keep going: one bad figure must not hide
            failures.append((name, repr(exc)))
            print(f"[paper-figures] FAILED {name}: {exc!r}")
    if failures:
        raise SystemExit(f"figure failures: {failures}")


if __name__ == "__main__":
    main()
