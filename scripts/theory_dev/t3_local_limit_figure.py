"""T3 summary and figure for the local-limit validation run.

Reads ``results/theory_validation/t3_local_limit_check.json`` (produced by
``t3_local_limit_check.py``) and, when present,
``results/phase7_local_power.json`` (the 300-replication Phase-7 fleet), and
prints the mechanism table used in ``theory/WP7_T3_local_limit.md`` section 7
without refitting anything.  Writes the two-panel figure
``results/theory_validation/t3_local_limit_check.png``:

  left  : per-cell per-direction rejection rates, naive Gaussian shift, the
          drift-corrected shift g + amp r_g, this run's empirical multiplier
          rate, and the Phase-7 300-rep empirical rate where available;
  right : the drift-stationarity residual diagnostics max_t |mean resid| and
          max_t z against the pre-registered 4-SE line, for the four
          (setting, direction) combinations of the production cells.

Usage:
    python scripts/theory_dev/t3_local_limit_figure.py
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))
OUT = os.path.join(ROOT, "results", "theory_validation",
                   "t3_local_limit_check.json")
P7 = os.path.join(ROOT, "results", "phase7_local_power.json")
PNG = os.path.join(ROOT, "results", "theory_validation",
                   "t3_local_limit_check.png")
DIRECTIONS = ("mean", "bump", "freq")
SETTINGS = ("PROG1", "PROG0")


def make_figure(run, p7=None, out=None, verbose=False):
    """Three-panel local-limit figure (power + the two N2 diagnostics)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def p7_cell(setting, direction, n=500):
        if p7 is None:
            return None
        for c in p7["cells"]:
            if (c["setting"] == setting and c["direction"] == direction
                    and c["n"] == n):
                return c
        return None

    rows = []
    for setting in SETTINGS:
        cell = run[f"{setting}|500"]
        for d in DIRECTIONS:
            e = cell["directions"][d]
            ref = p7_cell(setting, d)
            ref_rate = None if ref is None else ref["dr_multiplier_rate"]
            ref_se = None if ref is None else ref["mc_se_dr"]
            combined = None
            if ref_rate is not None:
                combined = float(np.sqrt(e["gauss_corr_se"] ** 2 + ref_se ** 2))
            rows.append({
                "setting": setting, "direction": d, "reps": cell["reps"],
                "naive": e["gauss_naive_mean"], "naive_se": e["gauss_naive_se"],
                "corrected": e["gauss_corr_mean"], "corrected_se": e["gauss_corr_se"],
                "emp_run": e["emp_rate"], "emp_run_se": e["mc_se_emp"],
                "phase7_emp": ref_rate, "phase7_se": ref_se,
                "gap_corrected": (None if ref_rate is None
                                  else e["gauss_corr_mean"] - ref_rate),
                "gap_naive": (None if ref_rate is None
                              else e["gauss_naive_mean"] - ref_rate),
                "resid_max_abs_mean": e.get("resid_max_abs_mean"),
                "resid_max_abs_z": e.get("resid_max_abs_z"),
            })
    if verbose:
        print(json.dumps({"rows": rows}, indent=2, sort_keys=True))

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
    ax = axes[0]
    x = np.arange(len(DIRECTIONS))
    width = 0.13
    method_colors = {"naive": "#0072B2", "corrected": "#E69F00",
                     "emp_run": "#D55E00"}
    for i, setting in enumerate(SETTINGS):
        data = [r for r in rows if r["setting"] == setting]
        base = (i - 0.5) * 0.56
        hatch = None if setting == "PROG1" else "///"
        for shift, key in ((-1.5, "naive"), (-0.5, "corrected"),
                           (0.5, "emp_run")):
            vals = [r[key] for r in data]
            errs = [2 * r[f"{key}_se"] for r in data]
            ax.bar(x + base + shift * width, vals, width,
                   color=method_colors[key], alpha=0.9, hatch=hatch,
                   edgecolor="white", linewidth=0.4,
                   label=f"{setting} {key.replace('_', ' ')}",
                   yerr=errs, capsize=2, error_kw={"elinewidth": 0.7})
        for j, r in enumerate(data):
            if r["phase7_emp"] is not None:
                ax.errorbar(x[j] + base + 1.5 * width, r["phase7_emp"],
                            yerr=2 * r["phase7_se"], fmt="k_", capsize=3,
                            label=("300-rep empirical, $\\pm2$ SE"
                                   if (i == 0 and j == 0) else None))
    ax.set_xticks(x)
    ax.set_xticklabels(DIRECTIONS)
    ax.set_ylabel("rejection rate")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("(a) local power vs shifted-signal prediction", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.4, loc="upper left", ncol=1)

    labels, max_mean, max_z = [], [], []
    for setting in SETTINGS:
        cell = run[f"{setting}|500"]
        for d in ("bump", "freq"):
            e = cell["directions"][d]
            labels.append(f"{setting}\n{d}")
            max_mean.append(e["resid_max_abs_mean"])
            max_z.append(e["resid_max_abs_z"])
    x = np.arange(len(labels))
    ax = axes[1]
    ax.plot(x, max_mean, marker="o", color="#0072B2",
            label="$\\max_t|$mean residual$|$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("raw residual scale")
    ax.set_title("(b) raw drift residual", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)

    ax = axes[2]
    ax.plot(x, max_z, marker="s", mfc="none", color="#D55E00",
            label="$\\max_t z$")
    ax.axhline(4.0, color="r", ls="--", lw=1.0, label="limit (4 SE)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("standardized residual (SE units)")
    ax.set_title("(c) standardized drift statistic", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)

    fig.tight_layout()
    out = out or PNG
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main():
    with open(OUT) as fh:
        run = json.load(fh)
    p7 = None
    if os.path.exists(P7):
        with open(P7) as fh:
            p7 = json.load(fh)
    print(f"wrote {make_figure(run, p7, verbose=True)}")


if __name__ == "__main__":
    main()
