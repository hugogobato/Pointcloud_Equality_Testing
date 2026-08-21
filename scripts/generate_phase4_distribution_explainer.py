"""Create a schematic explaining the Phase 4 distribution-level target.

The figure is conceptual rather than an empirical result.  It illustrates that
the probability law is over whole specimen point clouds, that each cloud is
mapped to a persistence diagram, and that the expected persistence measure is
an average weighted mass on the birth--midpoint plane.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase4_distribution_explainer.png"


def draw_cloud_group(ax, label, rng, variant):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(label, fontsize=11, fontweight="bold", pad=8)
    centers = (0.17, 0.50, 0.83)
    for j, cx in enumerate(centers, start=1):
        ax.add_patch(
            Rectangle((cx - 0.12, 0.16), 0.24, 0.62, facecolor="#f8fafc",
                      edgecolor="#cbd5e1", linewidth=1.0, zorder=0)
        )
        theta = np.linspace(0, 2 * np.pi, 55, endpoint=False)
        radius = 0.11 + 0.008 * rng.normal(size=theta.size)
        x = cx + radius * np.cos(theta) + 0.006 * rng.normal(size=theta.size)
        y = 0.47 + 0.20 * np.sin(theta) + 0.012 * rng.normal(size=theta.size)
        if variant == 1 and j == 3:
            # A small second component makes the picture visibly schematic.
            x = np.r_[x, cx + 0.035 + 0.025 * rng.normal(size=18)]
            y = np.r_[y, 0.31 + 0.025 * rng.normal(size=18)]
        ax.scatter(x, y, s=5, color="#2563eb" if variant == 0 else "#b45309",
                   alpha=0.75, linewidths=0)
        ax.text(cx, 0.08, rf"$Y_{{{j}}}$", ha="center", va="center", fontsize=9)
    ax.text(0.5, 0.91, "one point cloud per specimen", ha="center", va="center",
            fontsize=8.5, color="#475569")


def draw_diagrams(ax, label, rng, variant):
    ax.set_title(label, fontsize=10, fontweight="bold", pad=7)
    ax.set_xlim(0, 1.15)
    ax.set_ylim(0, 1.45)
    ax.set_xlabel("birth", fontsize=8)
    ax.set_ylabel("midpoint", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.18, linewidth=0.5)
    for j in range(7):
        if variant == 0:
            centers = [(0.22, 0.48), (0.55, 0.95)]
        else:
            centers = [(0.23, 0.50), (0.70, 1.13)]
        count = 1 + (j % 2 == 0)
        for k in range(count):
            bx, mx = centers[k % len(centers)]
            ax.scatter(bx + 0.035 * rng.normal(), mx + 0.035 * rng.normal(),
                       s=13, color="#2563eb" if variant == 0 else "#b45309",
                       alpha=0.65)
    ax.text(0.03, 1.34, "individual $D_i$", fontsize=8, color="#475569")


def weighted_measure(variant, rng, n=6000, bins=28):
    if variant == 0:
        centers = [(0.22, 0.50, 0.62), (0.55, 0.96, 0.26)]
    else:
        centers = [(0.23, 0.51, 0.58), (0.70, 1.13, 0.52)]
    chunks = []
    weights = []
    per = n // len(centers)
    for birth, midpoint, mass in centers:
        b = birth + 0.07 * rng.normal(size=per)
        m = midpoint + 0.08 * rng.normal(size=per)
        p = np.maximum(m - b, 0.05)
        chunks.append(np.column_stack([b, m]))
        weights.append(mass * p**3)
    points = np.vstack(chunks)
    weights = np.concatenate(weights)
    H, bx, mx = np.histogram2d(points[:, 0], points[:, 1],
                                bins=bins, range=((0, 1.15), (0, 1.45)),
                                weights=weights)
    return H.T, bx, mx


def main():
    rng = np.random.default_rng(18)
    fig = plt.figure(figsize=(11.2, 8.2), facecolor="white")
    gs = fig.add_gridspec(3, 4, height_ratios=[1.05, 1.02, 1.15],
                          hspace=0.56, wspace=0.52)

    ax_cloud0 = fig.add_subplot(gs[0, :2])
    ax_cloud1 = fig.add_subplot(gs[0, 2:])
    draw_cloud_group(ax_cloud0, "Reference collection ($a=0$)", rng, 0)
    draw_cloud_group(ax_cloud1, "Assessment collection ($a=1$)", rng, 1)

    ax_d0 = fig.add_subplot(gs[1, :2])
    ax_d1 = fig.add_subplot(gs[1, 2:])
    draw_diagrams(ax_d0, "Persistence diagrams across reference specimens", rng, 0)
    draw_diagrams(ax_d1, "Persistence diagrams across assessment specimens", rng, 1)

    H0, bx, mx = weighted_measure(0, rng)
    H1, _, _ = weighted_measure(1, rng)
    diff = H1 - H0
    extent = (bx[0], bx[-1], mx[0], mx[-1])
    vmax = max(H0.max(), H1.max())
    ax_m0 = fig.add_subplot(gs[2, 0])
    ax_m1 = fig.add_subplot(gs[2, 1])
    ax_md = fig.add_subplot(gs[2, 2:])
    for ax, H, title, cmap, norm in [
        (ax_m0, H0, r"$\mathbb{E}[\mu_D]$ for $a=0$", "Blues", None),
        (ax_m1, H1, r"$\mathbb{E}[\mu_D]$ for $a=1$", "YlOrBr", None),
        (ax_md, diff, r"difference: $\mathbb{E}[\mu_D^1]-\mathbb{E}[\mu_D^0]$",
         "RdBu_r", TwoSlopeNorm(vcenter=0, vmin=-max(abs(diff.min()), abs(diff.max())),
                                vmax=max(abs(diff.min()), abs(diff.max())))),
    ]:
        im = ax.imshow(H, origin="lower", extent=extent, aspect="auto", cmap=cmap,
                       norm=norm, interpolation="nearest")
        ax.set_xlabel("birth", fontsize=8)
        ax.set_ylabel("midpoint", fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
        ax.tick_params(labelsize=7)
        ax.text(0.03, 0.08, "weighted mass\n(not a PDF)", transform=ax.transAxes,
                fontsize=7, color="#334155", va="bottom",
                bbox=dict(facecolor="white", alpha=0.72, edgecolor="none", pad=2))
    ax_md.text(0.03, 0.92, r"$\delta_{\mathrm{dist}}=\|\cdot\|_1$ on a shared grid",
               transform=ax_md.transAxes, fontsize=8, color="#334155", va="top")

    fig.suptitle(
        "What is distributed in distribution-level testing?",
        fontsize=15, fontweight="bold", y=0.985)
    fig.text(
        0.5, 0.012,
        "The law is over whole specimen point clouds (and therefore their diagrams), not over the points within one cloud. "
        "The bottom row is a schematic expected weighted persistence measure.",
        ha="center", va="bottom", fontsize=9, color="#475569")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
