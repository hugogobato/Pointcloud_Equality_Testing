"""Executable Phase 5A nonidentification witnesses.

The script deliberately separates three objects that are easy to conflate:

* equality of the full point laws;
* equality of a fixed-size barcode law; and
* equality of one finite topological summary.

The examples are finite and deterministic apart from the probability laws, so
the reported checks do not depend on Monte Carlo error.  Run it from the
repository root with::

    python scripts/phase5_estimand_witnesses.py

The primary Phase-5 lock uses m=25, VR-radius diagrams in degrees 0 and 1.
"Radius" is used in the standard sense: an edge between two points at
distance d enters the filtration at radius d/2, and the shared PH module
reports every filtration on this radius scale.  The small exhaustive
enumeration uses m=4 only where it is needed to certify the isometry witness;
the equality there holds for every m.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from tda2s.ph import compute_diagrams


SEED = 20260819
PRIMARY_M = 25
FILTRATION = "vr"
HOMOLOGY_DIMS = (0, 1)
MAX_EDGE_LENGTH = 4.0


def _as_points(points: Sequence[Sequence[float]]) -> np.ndarray:
    return np.asarray(points, dtype=float).reshape(-1, 2)


def _validate_probabilities(probabilities: Sequence[float], n: int) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (n,) or np.any(p < 0) or not np.isclose(p.sum(), 1.0):
        raise ValueError("probabilities must be nonnegative and sum to one")
    return p


def total_variation(p: Sequence[float], q: Sequence[float]) -> float:
    """Total variation for two laws on the same finite support."""
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    return float(0.5 * np.abs(p_arr - q_arr).sum())


def expected_two_point_h0_persistence(
    support: Sequence[Sequence[float]], probabilities: Sequence[float]
) -> float:
    """E total finite H0 persistence for m=2 in the radius-scale VR filtration.

    Two i.i.d. draws at distance d merge into one component at radius d/2, so
    the finite H0 persistence of the two-point cloud is d/2.
    """
    points = _as_points(support)
    p = _validate_probabilities(probabilities, len(points))
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    return float(0.5 * (p @ distances @ p))


def _diagram_key(diagrams: Iterable[np.ndarray], decimals: int = 8) -> tuple:
    """Canonical, JSON-friendly key for a joint persistence diagram."""
    result = []
    for diagram in diagrams:
        dgm = np.asarray(diagram, dtype=float).reshape(-1, 2)
        if len(dgm):
            rounded = np.round(dgm, decimals=decimals)
            order = np.lexsort((rounded[:, 1], rounded[:, 0]))
            dgm = rounded[order]
        result.append(tuple(tuple(float(x) for x in row) for row in dgm))
    return tuple(result)


def _diagram_total_h0(diagrams: Sequence[np.ndarray]) -> float:
    dgm = np.asarray(diagrams[0], dtype=float).reshape(-1, 2)
    return float(np.maximum(dgm[:, 1] - dgm[:, 0], 0.0).sum()) if len(dgm) else 0.0


def enumerate_barcode_law(
    support: Sequence[Sequence[float]],
    probabilities: Sequence[float],
    m: int,
    *,
    homology_dims: Sequence[int] = HOMOLOGY_DIMS,
) -> Mapping[tuple, float]:
    """Enumerate the fixed-m barcode law for a small finite support.

    This is used only for support sizes and m values small enough for exact
    enumeration.  The primary m=25 occupancy check below avoids an exponential
    enumeration.
    """
    points = _as_points(support)
    p = _validate_probabilities(probabilities, len(points))
    law: defaultdict[tuple, float] = defaultdict(float)
    for indices in itertools.product(range(len(points)), repeat=int(m)):
        mass = float(np.prod(p[list(indices)]))
        if mass == 0.0:
            continue
        sample = points[list(indices)]
        diagrams = compute_diagrams(
            sample,
            filtration=FILTRATION,
            homology_dims=tuple(homology_dims),
            max_edge_length=MAX_EDGE_LENGTH,
        )
        law[_diagram_key(diagrams)] += mass
    return dict(law)


def law_total_variation(p: Mapping[tuple, float], q: Mapping[tuple, float]) -> float:
    keys = set(p) | set(q)
    return float(0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys))


def _law_mean_h0_persistence(law: Mapping[tuple, float]) -> float:
    total = 0.0
    for key, mass in law.items():
        dgm = np.asarray(key[0], dtype=float).reshape(-1, 2)
        total += float(mass) * (float(np.maximum(dgm[:, 1] - dgm[:, 0], 0.0).sum()) if len(dgm) else 0.0)
    return float(total)


def occupancy_law_three_point(probabilities: Sequence[float], m: int) -> np.ndarray:
    """Law of the number of distinct support atoms in m categorical draws."""
    p = _validate_probabilities(probabilities, 3)
    m = int(m)
    one = float(np.sum(p**m))
    three = float(1.0 - np.sum((1.0 - p) ** m) + np.sum(p**m))
    two = float(1.0 - one - three)
    out = np.array([one, two, three], dtype=float)
    out[out < 0.0] = 0.0
    return out / out.sum()


def probability_all_atoms_drawn(probabilities: Sequence[float], m: int) -> float:
    """P(all support atoms are drawn at least once in m i.i.d. categorical draws).

    Inclusion-exclusion over the missing-atom set S::

        P = sum_S (-1)^|S| (1 - p_S)^m,   p_S = sum_{j in S} p_j.

    The number of distinct atoms drawn is a deterministic function of the
    stored H0 diagram (finite H0 classes + 1 for a connected cloud), so
    |P(P0) - P(P1)| is a lower bound on the total variation between the two
    fixed-m barcode laws.
    """
    p = np.asarray(probabilities, dtype=float)
    n = p.size
    total = 0.0
    for s in range(1 << n):
        missing = [j for j in range(n) if (s >> j) & 1]
        total += (-1.0) ** len(missing) * (1.0 - p[missing].sum()) ** int(m)
    return float(total)


def _point_key(point: Sequence[float]) -> tuple[float, float]:
    x = np.asarray(point, dtype=float)
    return float(x[0]), float(x[1])


def _cloud_likelihood(cloud: np.ndarray, masses: Mapping[tuple[float, float], float]) -> float:
    likelihood = 1.0
    for point in cloud:
        likelihood *= float(masses.get(_point_key(point), 0.0))
    return float(likelihood)


def _shared_mixture_masses(
    cloud0: np.ndarray, cloud1: np.ndarray, tail_point: Sequence[float], epsilon: float
) -> dict[tuple[float, float], float]:
    atoms = np.unique(np.vstack([cloud0, cloud1]), axis=0)
    masses = {_point_key(point): (1.0 - epsilon) / len(atoms) for point in atoms}
    masses[_point_key(tail_point)] = float(epsilon)
    return masses


def _separate_mixture_masses(
    cloud: np.ndarray, tail_point: Sequence[float], epsilon: float
) -> dict[tuple[float, float], float]:
    atoms = np.unique(cloud, axis=0)
    masses = {_point_key(point): (1.0 - epsilon) / len(atoms) for point in atoms}
    masses[_point_key(tail_point)] = float(epsilon)
    return masses


def _mean_norm(masses: Mapping[tuple[float, float], float]) -> float:
    return float(sum(mass * np.linalg.norm(np.asarray(point)) for point, mass in masses.items()))


def witness_same_support_different_density() -> dict:
    """Same geometric support, different density, and changed barcode law."""
    square = _as_points(
        [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
    )
    uniform = np.full(4, 0.25)
    concentrated = np.array([0.70, 0.10, 0.10, 0.10])
    law_uniform_m2 = enumerate_barcode_law(square, uniform, 2, homology_dims=(0,))
    law_concentrated_m2 = enumerate_barcode_law(square, concentrated, 2, homology_dims=(0,))
    analytic_uniform = expected_two_point_h0_persistence(square, uniform)
    analytic_concentrated = expected_two_point_h0_persistence(square, concentrated)
    all_atoms_uniform = probability_all_atoms_drawn(uniform, PRIMARY_M)
    all_atoms_concentrated = probability_all_atoms_drawn(concentrated, PRIMARY_M)
    return {
        "support_equal": bool(np.array_equal(square, square)),
        "same_support_topology": True,
        "point_law_total_variation": total_variation(uniform, concentrated),
        "collision_probability_m2": {
            "uniform": float(np.sum(uniform**2)),
            "concentrated": float(np.sum(concentrated**2)),
        },
        "expected_h0_persistence_m2": {
            "uniform": analytic_uniform,
            "concentrated": analytic_concentrated,
        },
        "enumerated_mean_h0_persistence_m2": {
            "uniform": _law_mean_h0_persistence(law_uniform_m2),
            "concentrated": _law_mean_h0_persistence(law_concentrated_m2),
        },
        "barcode_law_tv_m2": law_total_variation(law_uniform_m2, law_concentrated_m2),
        "all_atoms_drawn_prob_m25": {
            "uniform": all_atoms_uniform,
            "concentrated": all_atoms_concentrated,
        },
        "barcode_law_tv_m25_lower_bound": abs(all_atoms_uniform - all_atoms_concentrated),
        "conclusion": "same support topology does not imply the locked barcode-law null",
    }


def witness_same_finite_summary_different_point_laws() -> dict:
    """Different laws agree on a finite summary but not at the locked m."""
    triangle = _as_points(
        [(0.0, 0.0), (1.0, 0.0), (0.5, np.sqrt(3.0) / 2.0)]
    )
    p = np.array([0.60, 0.20, 0.20])
    root = np.sqrt(0.13)
    q = np.array([0.50, (0.50 + root) / 2.0, (0.50 - root) / 2.0])
    m2_law_p = enumerate_barcode_law(triangle, p, 2, homology_dims=(0,))
    m2_law_q = enumerate_barcode_law(triangle, q, 2, homology_dims=(0,))
    occupancy_p = occupancy_law_three_point(p, PRIMARY_M)
    occupancy_q = occupancy_law_three_point(q, PRIMARY_M)
    return {
        "support_equal": True,
        "point_law_total_variation": total_variation(p, q),
        "finite_summary": "E total H0 persistence for m=2 (radius-scale VR)",
        "finite_summary_p": expected_two_point_h0_persistence(triangle, p),
        "finite_summary_q": expected_two_point_h0_persistence(triangle, q),
        "finite_summary_equal": bool(
            np.isclose(
                expected_two_point_h0_persistence(triangle, p),
                expected_two_point_h0_persistence(triangle, q),
                atol=1e-12,
            )
        ),
        "collision_probability_m2": {
            "p": float(np.sum(p**2)),
            "q": float(np.sum(q**2)),
        },
        "enumerated_mean_h0_persistence_m2": {
            "p": _law_mean_h0_persistence(m2_law_p),
            "q": _law_mean_h0_persistence(m2_law_q),
        },
        "barcode_law_tv_m2": law_total_variation(m2_law_p, m2_law_q),
        "occupancy_law_m25_p": occupancy_p.tolist(),
        "occupancy_law_m25_q": occupancy_q.tolist(),
        "barcode_law_tv_m25_lower_bound": float(0.5 * np.abs(occupancy_p - occupancy_q).sum()),
        "conclusion": "a finite summary and even a small-m barcode law need not identify the locked m=25 law",
    }


def witness_isometric_barcode_collision() -> dict:
    """Distinct ambient point laws with identical metric barcode laws."""
    base = _as_points([(0.0, 0.0), (1.0, 0.0), (0.5, np.sqrt(3.0) / 2.0)])
    translated = base + np.array([10.0, -7.0])
    probabilities = np.array([0.20, 0.50, 0.30])
    law_base = enumerate_barcode_law(base, probabilities, 4)
    law_translated = enumerate_barcode_law(translated, probabilities, 4)
    return {
        "ambient_translation": [10.0, -7.0],
        # The two laws have disjoint supports, so as laws on the ambient
        # coordinate space they are mutually singular: TV is exactly 1.
        "point_law_total_variation": 1.0,
        "fixed_m_checked": 4,
        "barcode_law_tv_m4": law_total_variation(law_base, law_translated),
        "mean_h0_persistence_base": _law_mean_h0_persistence(law_base),
        "mean_h0_persistence_translated": _law_mean_h0_persistence(law_translated),
        "conclusion": "the metric barcode target is invariant to ambient isometries, so it is not a full coordinate-law test",
    }


def witness_fixed_cloud_ambiguity() -> dict:
    """The same fixed pair has positive likelihood under very different laws."""
    cloud0 = _as_points([(-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)])
    cloud1 = _as_points([(0.0, -1.0), (0.0, 0.0), (0.0, 1.0)])
    epsilon = 0.10
    tail_scales = (10.0, 1000.0, 100000.0)
    rows = []
    for scale in tail_scales:
        shared_tail = (scale, scale)
        separate_tail0 = (scale, 0.0)
        separate_tail1 = (0.0, scale)
        shared = _shared_mixture_masses(cloud0, cloud1, shared_tail, epsilon)
        separate0 = _separate_mixture_masses(cloud0, separate_tail0, epsilon)
        separate1 = _separate_mixture_masses(cloud1, separate_tail1, epsilon)
        shared_likelihood = _cloud_likelihood(cloud0, shared) * _cloud_likelihood(cloud1, shared)
        separate_likelihood = _cloud_likelihood(cloud0, separate0) * _cloud_likelihood(cloud1, separate1)
        rows.append(
            {
                "tail_scale": scale,
                "shared_model_joint_likelihood": shared_likelihood,
                "separate_model_joint_likelihood": separate_likelihood,
                "shared_model_mean_norm": _mean_norm(shared),
                "separate_model_mean_norm_sum": _mean_norm(separate0) + _mean_norm(separate1),
            }
        )
    return {
        "cloud_sizes": [len(cloud0), len(cloud1)],
        "epsilon_remote_tail": epsilon,
        "models": "shared mixture versus separate mixtures, each with atoms at every observed point",
        "positive_likelihood_for_all_tail_scales": all(
            row["shared_model_joint_likelihood"] > 0.0
            and row["separate_model_joint_likelihood"] > 0.0
            for row in rows
        ),
        "rows": rows,
        "conclusion": "a fixed pair of arrays does not identify a population law or a shared-versus-separate claim",
    }


def run_witnesses() -> dict:
    same_support = witness_same_support_different_density()
    finite_collision = witness_same_finite_summary_different_point_laws()
    isometric_collision = witness_isometric_barcode_collision()
    fixed_cloud = witness_fixed_cloud_ambiguity()
    checks = {
        "same_support_is_exact": same_support["support_equal"],
        "same_support_changes_barcode_law": same_support["barcode_law_tv_m2"] > 1e-8,
        "same_support_changes_locked_m25_law": same_support["barcode_law_tv_m25_lower_bound"] > 1e-8,
        "analytic_h0_m2_matches_enumerated": all(
            np.isclose(
                same_support["expected_h0_persistence_m2"][k],
                same_support["enumerated_mean_h0_persistence_m2"][k],
                atol=1e-12,
            )
            for k in ("uniform", "concentrated")
        )
        and all(
            np.isclose(
                finite_collision["finite_summary_p" if k == "p" else "finite_summary_q"],
                finite_collision["enumerated_mean_h0_persistence_m2"][k],
                atol=1e-12,
            )
            for k in ("p", "q")
        ),
        "finite_summary_is_equal": finite_collision["finite_summary_equal"],
        "finite_summary_m2_barcode_is_equal": finite_collision["barcode_law_tv_m2"] < 1e-10,
        "locked_m25_barcode_differs": finite_collision["barcode_law_tv_m25_lower_bound"] > 1e-8,
        "isometry_preserves_barcode_law": isometric_collision["barcode_law_tv_m4"] < 1e-10,
        "fixed_cloud_models_both_possible": fixed_cloud["positive_likelihood_for_all_tail_scales"],
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"Phase 5A witness check(s) failed: {failed}")
    return {
        "seed": SEED,
        "primary_lock": {
            "m": PRIMARY_M,
            "filtration": FILTRATION,
            "homology_dims": list(HOMOLOGY_DIMS),
            "filtration_scale": "radius (edge between points at distance d enters at d/2)",
        },
        "witnesses": {
            "same_support_different_density": same_support,
            "same_finite_summary_different_point_laws": finite_collision,
            "isometric_barcode_collision": isometric_collision,
            "fixed_cloud_ambiguity": fixed_cloud,
        },
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="also write the JSON certificate to this path",
    )
    args = parser.parse_args()
    result = run_witnesses()
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
