"""WP1.3 / Phase 4.1: which (Phi, T_dist) pairing separates which witness.

The choice of distribution-level representation ``T_dist`` in Phase 4.1 is a
*logical* commitment, not only a computational one: it decides whether
H0^dist and H0^out are independent nulls, nested nulls, or the same null.
This script computes the three candidate statistics on the two WP1 witnesses
exactly (the laws are finitely supported, so no Monte Carlo is needed) and
prints the separation table that Phase 4.1's decision rests on.

Statistics
----------
  A      -- the mean functional of the NORMALIZED power-weighted silhouette
            (Kim & Lee's estimand; what H0^out compares).
  T(i)   -- expected persistence measure, L1 on a fixed grid (Divol-Lacombe;
            plan candidate (i)).
  T(ii)  -- MMD under the universal persistence scale-space kernel
            k^U_sigma(F, G) = exp(k_sigma(F, G)) of Kwitt, Huber, Niethammer,
            Lin & Bauer, NIPS 2015, Proposition 2: universal with respect to
            d_{W,1} on the set S of diagrams with birth/death bounded by R and
            total multiplicity bounded by N.  Universal implies characteristic,
            so the mean embedding is injective on laws over S and T(ii)
            separates *any* two distinct diagram laws (plan candidate (ii)).

  k_sigma(F, G) = 1/(8 pi sigma) sum_{p in F} sum_{q in G}
                    [ exp(-|p - q|^2 / (8 sigma))
                    - exp(-|p - qbar|^2 / (8 sigma)) ],  qbar = (q_y, q_x)

Witnesses (see theory/WP1_estimands_identification.md 1.3)
----------------------------------------------------------
  W1  : D^0 = {(0, l)}          ; D^1 = {(0, l), (0, l)}       (deterministic)
  W2' : D^0 = {(0,1), (0,2)}    ; D^1 = {(0,1),(0,1)} or
                                        {(0,2),(0,2)} w.p. 1/2 each

Expected reading: A is blind to W1, T(i) is blind to W2', T(ii) is blind to
neither.  Hence candidate (i) makes the two nulls logically independent and
candidate (ii) makes H0^dist strictly stronger than H0^out.

Usage
-----
    rtk uv run python scripts/wp1_pairing_separation.py
"""
from __future__ import annotations

import numpy as np

from tda2s.vec import persistence_measure, silhouette

IV = (0.0, 2.5)          # one fixed grid, shared by every statistic
R_POW = 3.0              # silhouette power weight
N_BINS = 32
SIGMA = 0.10             # PSS kernel bandwidth


# ---------------------------------------------------------------- statistics
def mean_silhouette(law):
    """Mean NORMALIZED power-weighted silhouette of a finitely supported law.

    ``law`` is a list of ``(probability, diagram)`` pairs.
    """
    out = 0.0
    for prob, dgm in law:
        out = out + prob * silhouette([np.asarray(dgm, float).reshape(-1, 2)],
                                      interval=IV, r=R_POW)[0]
    return out


def expected_measure(law):
    """Expected persistence measure on the fixed grid, weight = persistence."""
    out = 0.0
    for prob, dgm in law:
        out = out + prob * persistence_measure(
            [np.asarray(dgm, float).reshape(-1, 2)],
            interval=IV, n_bins=N_BINS)[0].ravel()
    return out


def _k_pss(f, g, sigma=SIGMA):
    """Reininghaus et al. persistence scale-space kernel k_sigma(F, G)."""
    f = np.asarray(f, float).reshape(-1, 2)
    g = np.asarray(g, float).reshape(-1, 2)
    if len(f) == 0 or len(g) == 0:
        return 0.0
    gbar = g[:, ::-1]
    d2 = ((f[:, None, :] - g[None, :, :]) ** 2).sum(-1)
    d2bar = ((f[:, None, :] - gbar[None, :, :]) ** 2).sum(-1)
    return float((np.exp(-d2 / (8 * sigma)) - np.exp(-d2bar / (8 * sigma))).sum()
                 / (8 * np.pi * sigma))


def _k_universal(f, g, sigma=SIGMA):
    """Kwitt et al. NIPS 2015 Prop. 2: k^U_sigma = exp(k_sigma), universal."""
    return float(np.exp(_k_pss(f, g, sigma)))


def mmd_universal(law0, law1, sigma=SIGMA):
    """Exact MMD between two finitely supported diagram laws under k^U_sigma."""
    def cross(a, b):
        return sum(pa * pb * _k_universal(da, db, sigma)
                   for pa, da in a for pb, db in b)
    mmd2 = cross(law0, law0) - 2 * cross(law0, law1) + cross(law1, law1)
    return float(np.sqrt(max(mmd2, 0.0)))


# ------------------------------------------------------------------ witnesses
def witnesses(ell=1.35):
    w1 = (
        [(1.0, [[0.0, ell]])],
        [(1.0, [[0.0, ell], [0.0, ell]])],
    )
    w2p = (
        [(1.0, [[0.0, 1.0], [0.0, 2.0]])],
        [(0.5, [[0.0, 1.0], [0.0, 1.0]]), (0.5, [[0.0, 2.0], [0.0, 2.0]])],
    )
    return {"W1": w1, "W2'": w2p}


def main():
    rows = []
    for name, (law0, law1) in witnesses().items():
        a = float(np.abs(mean_silhouette(law1) - mean_silhouette(law0)).max())
        t1 = float(np.abs(expected_measure(law1) - expected_measure(law0)).sum())
        t2 = mmd_universal(law0, law1)
        rows.append((name, a, t1, t2))

    print(f"{'witness':>8} | {'A: sup|mean sil gap|':>21} | "
          f"{'T(i): L1 exp. measure':>22} | {'T(ii): MMD, universal':>22}")
    print("-" * 84)
    for name, a, t1, t2 in rows:
        print(f"{name:>8} | {a:21.6f} | {t1:22.6f} | {t2:22.6f}")
    print()
    print("Reading (0 = the statistic cannot see the witness):")
    print("  A     blind to W1  -> H0^out holds there; C2 exists because of this.")
    print("  T(i)  blind to W2' -> the two nulls are LOGICALLY INDEPENDENT.")
    print("  T(ii) blind to neither -> H0^dist is STRICTLY STRONGER than H0^out,")
    print("        so under candidate (ii) the 'two independent tests' framing")
    print("        collapses into a nesting.  Phase 4.1 must choose deliberately.")


if __name__ == "__main__":
    main()
