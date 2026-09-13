"""Independent spot-check of paper-figure values against the raw shards.

Re-aggregates the committed shards/parquet inputs with the experiment
aggregation functions and compares the resulting numbers with the JSON
payloads that feed ``paper/figures/``.  Run before regenerating figures:

    rtk uv run python scripts/audit_figure_values.py
"""
import json, os, sys, glob
import numpy as np

from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "theory_dev"))
os.chdir(ROOT)

checks = []
def chk(name, got, want, tol=5e-4):
    ok = abs(float(got) - float(want)) <= tol
    checks.append((name, ok, round(float(got), 4), round(float(want), 4)))
    print(("PASS" if ok else "FAIL"), name, got, want)

# ---- phase 2: raw shards -> rates vs committed figure JSON
import phase2_imbalance_sweep as p2
config, part_a, part_b = p2._load_shards()
sweep = p2._rates_a(part_a, [f"lam{lam:g}" for lam in p2.LAMBDAS])
mask = p2._rates_b(part_b)
fig = json.load(open("results/phase2_figure1.json"))
for t, k, v in [("rt","lam0.6",None), ("strand","lam1",None), ("dr","lam1",None),
                ("moon_lazar","lam0",None)]:
    chk(f"phase2 sweep {t}@{k}", sweep[t][k], fig["sweep_rates"][t][k])
for t in ("dr","moon_lazar","mmd"):
    chk(f"phase2 masking {t}", mask[t], fig["masking_rates"][t])
print("phase2 merged reps", len(part_a), len(part_b))

# ---- phase 4: re-aggregate shards -> compare to committed summaries
import phase4_separation as p4
for design, committed in (("separation","phase4_separation_summary.json"),
                          ("reverse","phase4_reverse_summary.json")):
    import tempfile
    tmp = os.path.join(tempfile.gettempdir(), f"phase4_{design}_tmp.json")
    p4.aggregate(design=design, input_dir=os.path.join(ROOT,"results","phase4_shards"), output=tmp)
    new = json.load(open(tmp))
    old = json.load(open(os.path.join(ROOT,"results",committed)))
    for design_row, (nr, orow) in enumerate(zip(new["rows"], old["rows"])):
        if nr["n"] != orow["n"]: raise SystemExit("row mismatch")
    for n_target, key in ((200,"outcome_permutation_rejection_rate"),
                          (25,"dist_permutation_rejection_rate"),
                          (50,"outcome_multiplier_rejection_rate")):
        nrow = next(r for r in new["rows"] if r["n"]==n_target)
        orow = next(r for r in old["rows"] if r["n"]==n_target)
        chk(f"phase4 {design} n={n_target} {key}", nrow[key], orow[key])

# ---- phase 45: re-aggregate shards -> compare cells
import phase45_weak_null as p45
agg = p45.aggregate(verbose=False)
committed45 = json.load(open(os.path.join(ROOT,"results","phase45_aggregate.json")))
for key in ["('confounded', 50, 1.0, None)", "('w2p', 500, None, None)",
            "('local', 200, None, 0.25)"]:
    chk(f"phase45 {key}", agg["cells"][key]["multiplier_p"]["rate"],
        committed45["cells"][key]["multiplier_p"]["rate"])

# ---- phase 8: raw parquet -> master summary
import pandas as pd
shard = os.path.join(ROOT,"results","phase8_shards")
for fam, imb, n, test, pcol in [("null","mild",200,"mmd","p_mmd"),
                                ("dispersion","randomized",200,"strand","p_strand"),
                                ("w1det","mild",200,"dr_mult","p_dr_mult"),
                                ("null","strong",200,"dr_mult","p_dr_mult")]:
    f = os.path.join(shard, f"cell_{fam}_{imb}_n{n}_100reps_coarse.parquet")
    if not os.path.exists(f):
        print("missing", f); continue
    df = pd.read_parquet(f)
    rate = float((df[pcol].dropna() <= 0.05).mean())
    master = json.load(open(os.path.join(ROOT,"results","phase8_master.json")))
    cell = next(c for c in master["summary"] if c["family"]==fam and c["imbalance"]==imb
                and c["n"]==n and c["test"]==test and "coarse" in c["shard"])
    chk(f"phase8 {fam}/{imb}/n{n}/{test}", rate, cell["rate"])

# ---- T10: n=500 production cells vs phase7 summary
t10 = json.load(open(os.path.join(ROOT,"results","theory_validation","t10_full_fleet.json")))
p7 = json.load(open(os.path.join(ROOT,"results","phase7_local_power.json")))
for setting, direction in (("PROG1","freq"), ("PROG0","mean"), ("NONPROG1","freq")):
    c10 = next(c for c in t10["cells"] if c["setting"]==setting and c["direction"]==direction and c["n"]==500)
    c7 = next(c for c in p7["cells"] if c["setting"]==setting and c["direction"]==direction and c["n"]==500)
    if setting == "NONPROG1":
        # t10 completed this cell past the Phase-7 partial (190/300 reps);
        # the estimate legitimately moves, so only assert both are in range.
        cond = (c10["reps"] >= 300 and abs(c10["dr_multiplier_rate"]
                                           - c7["dr_multiplier_rate"]) < 0.05)
        chk(f"t10 {setting}/{direction}/n500 dr (completed reps)",
            1 if cond else 0, 1, tol=0)
    else:
        chk(f"t10 {setting}/{direction}/n500 dr", c10["dr_multiplier_rate"], c7["dr_multiplier_rate"])

# ---- t3 / t5 / t8 / t9 / t4: figure sources are the JSON itself; check identities
t3 = json.load(open(os.path.join(ROOT,"results","theory_validation","t3_local_limit_check.json")))
e = t3["PROG1|500"]["directions"]["freq"]
chk("t3 PROG1/freq naive", e["gauss_naive_mean"], 0.6442)
chk("t3 PROG1/freq corrected", e["gauss_corr_mean"], 0.7766)
chk("t3 PROG1/bump emp", t3["PROG1|500"]["directions"]["bump"]["emp_rate"], 0.0833)

t5 = json.load(open(os.path.join(ROOT,"results","theory_validation","t5_are_validation.json")))
row1 = next(r for r in t5["C3"]["table"]["rows"] if r["lambda"]==1.0)
chk("t5 ARE lambda=1", row1["are_ipw_over_dr"], 4.5315)
chk("t5 first-order lambda=1", 4*(1+4*row1["var_e"]), 4.8734)
chk("t5 overlap lambda=1", row1["overlap_quad"], 6.0174)

t8 = json.load(open(os.path.join(ROOT,"results","theory_validation","t8_minimax_checks.json")))
chk("t8 fixed slope", t8["fixed_grid_rate"]["loglog_slope"], -0.5)
chk("t8 smooth slope", t8["smooth_ball_rate"]["loglog_slope"], -0.4147)
chk("t8 oracle total s=0.5", t8["le_cam"]["cells"][5]["empirical_total_error"], 0.9411)

t9 = json.load(open(os.path.join(ROOT,"results","theory_validation","t9_multiplicity_checks.json")))
r = next(x for x in t9["power"] if x["n"]==200 and x["scale"]==2.0)
chk("t9 unstud n200 a=2", r["unstud_rate"], 0.048)
r = next(x for x in t9["power"] if x["n"]==500 and x["scale"]==1.0)
chk("t9 unstud n500 a=1", r["unstud_rate"], 0.65267)
r = next(x for x in t9["power"] if x["n"]==200 and x["scale"]==64.0)
chk("t9 pooled n200 a=64", r["pooled_rate"], 0.66933)

t4 = json.load(open(os.path.join(ROOT,"results","theory_validation","t4_efficiency_checks.json")))
chk("t4 oracle ratio mean", float(np.mean(t4["main"]["oracle"]["ratio"])), 1.0386, tol=2e-3)
chk("t4 ss_knn ratio mean-0", float(np.mean(t4["main"]["ss_knn"]["ratio"])), 1.2512, tol=2e-3)
chk("t4 cf_knn ratio mean-0", float(np.mean(t4["main"]["cf_knn"]["ratio"])), 1.9411, tol=2e-3)

# ---- t2: recorded sweep/masking must agree with the Phase-2 payload
import t2_leakage_validate as t2m
t2 = json.load(open(os.path.join(ROOT,"results","theory_validation","t2_leakage_check.json")))
for test in ("rt", "mmd", "frechet_anova"):
    for i, lam in enumerate(t2m.LAMBDAS):
        chk(f"t2 sweep {test}@lam{lam:g}",
            t2["sweep_checks"][test]["rates"][i],
            fig["sweep_rates"][test][f"lam{lam:g}"])
for test in ("rt", "moon_lazar", "mmd"):
    chk(f"t2 masking {test}", t2["masking_checks"][test]["rate"],
        fig["masking_rates"][test])
# ---- t6: re-aggregate the local-grid shards for one cell
t6 = json.load(open(os.path.join(ROOT,"results","theory_validation","t6_local_summary.json")))
pv = []
for path in sorted(glob.glob(os.path.join(
        ROOT,"results","theory_validation","t6_local_n200_h1.0_shard*.json"))):
    with open(path) as fh:
        pv.extend(r["p"] for r in json.load(fh)["rows"])
rate = float((np.asarray(pv) < 0.05).mean())
chk("t6 local n200 h1.0 raw", rate, t6["cells"]["n=200|h=1.0"]["rate"])
chk("t6 local n200 h6.0", t6["cells"]["n=200|h=6.0"]["rate"], 0.32)
chk("t6 local n500 h1.0 predicted",
    t6["cells"]["n=500|h=1.0"]["prediction"]["power"], 0.06045)

# ---- t7: complete T7 rows must match the Phase-7 fleet rates
t7 = json.load(open(os.path.join(ROOT,"results","theory_validation",
                                 "t7_checkpoint_corrected.json")))
for setting, direction, n in (("PROG0","mean",500), ("PROG0","freq",100),
                              ("PROG1","mean",200)):
    row = next(r for r in t7["rows"] if r["setting"]==setting
               and r["direction"]==direction and r["n"]==n)
    ref = next(c for c in p7["cells"] if c["setting"]==setting
               and c["direction"]==direction and c["n"]==n)
    chk(f"t7 {setting}/{direction}/n{n} vs phase7", row["mult_rate"],
        ref["dr_multiplier_rate"])

# summary
fails = [c for c in checks if not c[1]]
print(f"\n{len(checks)-len(fails)}/{len(checks)} checks PASS")
if fails:
    print("FAILURES:", fails)
    sys.exit(1)
