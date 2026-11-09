"""
Print every number cited in the manuscript, read from results/*.json.

    python summarize_results.py > results/summary.txt
"""

import json

from run_experiments import s_at_fraction


def load(n):
    with open(f"results/{n}.json") as f:
        return json.load(f)


def pct(a):
    return f"{a['mean']:.1f}% (SD {a['sd']:.1f})"


cal = load("scar_calibration")
print("== Scar calibration ==")
print(f"reference params R2 (32-grid) {cal['used_r2']:.4f}; 64-grid {cal['used_r2_64grid']:.4f}")
print("reference predictions (0,1,3,7,28 d):", [round(v, 2) for v in cal["used_pred"]])
print("residuals:", [round(v, 2) for v in cal["used_residuals"]])
print("2025 grid optimum on edge of grid:", cal["grid_2025"]["on_grid_edge"])
print(f"extended grid R2 {cal['grid_extended']['r2']:.4f}; params {cal['grid_extended']['params']}")
print(f"least-squares R2 {cal['least_squares']['r2']:.4f}; params {cal['least_squares']['params']}")
print("least-squares fold change vs reference:", {k: round(v, 3) for k, v in cal["least_squares"]["fold_change_vs_used"].items()})
print("least-squares predictions:", [round(v, 2) for v in cal["least_squares"]["pred"]])

e1 = load("e1_uninjured_calibration")
print("\n== E1 uninjured calibration ==")
for r in e1["rows"]:
    print(f"B={r['budget']:>5}: crossing {r['crossing']:.1f}% (SD {r['sd']:.1f}); with 200-step cap {r['crossing_200step_cap']:.1f}%")
print("unlimited walk:", e1["unlimited_walk"])

e2 = load("e2_crossing_vs_scar")
print("\n== E2 crossing vs scar (calibrated B=200 unless noted) ==")
print("uninjured at B=200:", pct(e2["uninjured_calibrated"]["crossing"]))
for route, rows in e2["routes"].items():
    tot = [r["calibrated"]["crossing"]["mean"] for r in rows]
    print(f"-- {route}: total crossing range {min(tot):.1f}-{max(tot):.1f}%")
    for r in rows:
        c = r["calibrated"]
        print(f"  level {r['level']:<6} core {r['core_S']:.2f} healthy {r['healthy_S']:.2f} | initial settings crossing "
              f"{r['settings_2025']['crossing']['mean']:.1f}% | calibrated all {pct(c['crossing'])}, through core "
              f"{pct(c['through_core'])}, through zone {c['through_zone']['mean']:.1f}%, bypass {pct(c['bypass'])}")

e5 = load("e5_therapy_scope")
print("\n== E5 therapy scope ==")
for r in e5["rows"]:
    g, l = r["global"], r["lesion"]
    print(f"gamma {r['gamma']:<6} core {l['core_S']:.2f} | healthy (global) {g['healthy_S']:.2f} | initial: global "
          f"{g['settings_2025']['crossing']['mean']:.1f}% (through core {g['settings_2025']['through_core']['mean']:.1f}) "
          f"lesion {l['settings_2025']['crossing']['mean']:.1f}% | calibrated: global {g['calibrated']['crossing']['mean']:.1f}% "
          f"(core {g['calibrated']['through_core']['mean']:.1f}) lesion {l['calibrated']['crossing']['mean']:.1f}% "
          f"(core {l['calibrated']['through_core']['mean']:.1f})")
    gi = g["settings_2025"]
    if gi["crossing"]["mean"] > 0:
        print(f"      initial settings, whole-grid: through penumbra only {gi['through_zone']['mean'] - gi['through_core']['mean']:.1f}% "
              f"of {gi['crossing']['mean']:.1f}% crossing; around the zone {gi['bypass']['mean']:.1f}%")

e3 = load("e3_budget_sensitivity")
print("\n== E3 budget sensitivity (idealized disk fields) ==")
print("path stats (unlimited walk, s~1):", e3["path_stats"])
for r in e3["rows"]:
    s = r["simulated"]; p = r["predicted"]
    pr = r["predicted_through_core"]
    same = s_at_fraction(e3["s_grid"], pr, 0.5 * s["y_at_baseline"] / pr[0]) if pr[0] > 0 else None
    print(f"B={r['budget']}: through-core at s=1 {s['y_at_baseline']:.1f}% (s=1.2: {r['through_core_mean'][1]:.1f}%); "
          f"S50 {s['s50']} CI {r['s50_ci']}; 90->10 width {s['width_90_10']}; predicted S50 own-baseline {p['s50']}, "
          f"same-baseline {same}; predicted through-core at s=1: {pr[0]:.1f}%")

e4 = load("e4_lesion_size")
print("\n== E4 lesion size (B=200, idealized disk fields) ==")
for vname in ("scaled_geometry", "fixed_geometry"):
    print(f"-- {vname}")
    for r in e4[vname]:
        s, ps = r["simulated"], r["path_stats"]
        print(f"  core radius {r['core_radius']} (zone {r['zone_radius']}), rows {r['start_y']}->{r['target_y']} "
              f"({r['target_y'] - r['start_y']} rows): uninjured {r['uninjured_crossing']['mean']:.1f}%; "
              f"through-core at s=1 {s['y_at_baseline']:.1f}%; S50 {s['s50']} CI {r['s50_ci']}; "
              f"predicted S50 (own baseline) {r['predicted']['s50']}; median moves of through-core paths: "
              f"zone {ps['median_zone_moves']}, healthy {ps['median_healthy_moves']}")

e6 = load("e6_treatments")
print("\n== E6 treatments (B=200) ==")
print("uninjured:", pct(e6["uninjured"]["crossing"]))
for pset, rows in e6["parameter_sets"].items():
    print(f"-- {pset}")
    for r in rows:
        print(f"  {r['name']:44s} core d28 {r['core_S_d28']:.2f} | all {pct(r['crossing'])} | through core "
              f"{pct(r['through_core'])} | bypass {r['bypass']['mean']:.1f}%")

for pset, rows in e6["parameter_sets"].items():
    for r in rows:
        if r["name"].startswith("chABC, days"):
            t, cs = r["trajectory"]["t_h"], r["trajectory"]["core_S"]
            end = 168 if "1-7" in r["name"] else 336
            print(f"[{pset}] {r['name']}: core S at end of window ({end} h) {cs[t.index(end)]:.2f}; day 28 {r['core_S_d28']:.2f}")
        if r.get("scale"):
            print(f"[{pset}] {r['name']}: multipliers {r['scale']}; production params {r['production_params']}")

print("\n== Audit of the initial implementation (audit/results) ==")
sw = json.load(open("audit/results/sweep_results.json"))
for label in ("original_conditional", "fixed_default"):
    print(f"-- sweep, {label}")
    for r in sw[label]:
        print(f"  gamma {r['gamma']:<6} core {r['core_S']:.2f} healthy {r['healthy_S']:.3f} crossing {r['cross_mean']:.1f}% "
              f"(SD {r['cross_sd']:.1f}); fraction of crossers through core {r['fraction_of_crossers_through_core']}")
ct = json.load(open("audit/results/control_results.json"))
print("-- lesion-only therapy:", [(round(r["core_S"], 2), r["cross"]) for r in ct["lesion_only_therapy"]])
print("-- no lesion, uniform S:", [(r["uniform_S"], round(r["cross"], 1)) for r in ct["no_lesion_uniform"]])
mm = json.load(open("audit/results/mmp_results.json"))
print("-- MMP removed: max change in calibrated core trajectory",
      round(mm["core_trajectory_days_1_3_7_28"]["max_abs_difference"], 4))
for r in mm["sweep"]:
    a, b = r["with_mmp"], r["without_mmp"]
    print(f"  gamma {r['gamma']:<6} core {a['core_S']:.2f}: crossing with MMP {a['cross_mean']:.1f}%, without {b['cross_mean']:.1f}%; "
          f"healthy-tissue S range with MMP {a['healthy_min']:.5f}-{a['healthy_max']:.5f}, without {b['healthy_min']:.5f}-{b['healthy_max']:.5f}")

# Field differences between time-limited arms and control (Limitations)
import numpy as np
from scar_model import FITTED_PARAMS, ScarModel, Therapy
print("\n== Day-28 field difference from control, time-limited chABC arms ==")
for pset, base in [("reference", {}), ("least_squares", cal["least_squares"]["params"])]:
    p = {**FITTED_PARAMS, **base}
    ctrl = ScarModel(params=p); ctrl.run(672)
    for lab, th in [("days 1-7", Therapy(0.012, 24, 168)), ("days 7-14", Therapy(0.012, 168, 336))]:
        m = ScarModel(params=p, therapies=[th]); m.run(672)
        print(f"[{pset}] chABC {lab}: max |S - S_control| over the grid {np.abs(m.S - ctrl.S).max():.2e}")
