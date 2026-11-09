"""
Run all experiments reported in the manuscript.

    python run_experiments.py            # everything (about 10 minutes on 2 cores)
    python run_experiments.py e2 e3      # a subset

Each experiment writes results/<name>.json. All random draws use
numpy Generators seeded from (BASE_SEED, experiment, condition, replicate),
so results are reproducible bit-for-bit on the same numpy version.

E1  uninjured_calibration   crossing in an uninjured cord vs budget B
E2  crossing_vs_scar        crossing vs lesion CSPG, two independent routes
                            to lowering scar (lesion-scoped therapy dose,
                            reduced production), at the 2025 settings and at
                            the calibrated budget
E3  budget_sensitivity      location of the through-lesion threshold (S50)
                            vs budget B, simulated and predicted from
                            path-cost statistics
E4  lesion_size             S50 vs lesion size at the calibrated budget, with
                            start/target rows that move with the lesion and
                            with start/target rows fixed
E5  therapy_scope           lesion-scoped vs whole-grid therapy (the 2025
                            implementation) at the 2025 settings and at the
                            calibrated budget
E6  treatments              treatment arms with identical axon parameters,
                            using the reference scar parameters and the
                            least-squares parameter set from calibrate_scar.py;
                            production changes are fractional, so each arm
                            makes the same relative change under both sets
"""

import json
import os
import sys
import time

import numpy as np

from axon_model import MAX_STEPS, MAX_STEPS_2025, AxonGeometry, simulate_axons, summarize
from scar_model import FITTED_PARAMS, ScarModel, Therapy

BASE_SEED = 20261108
N_AXONS = 250
B_2025 = 25.0
B_REF = 200.0       # calibrated budget (see E1); ~94% crossing in uninjured tissue
HOURS = 672.0       # 28 days
os.makedirs("results", exist_ok=True)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def rng_for(*keys):
    return np.random.default_rng([BASE_SEED, *(int(round(float(k))) for k in keys)])


def geometry(core_radius=8, center_y=32):
    return AxonGeometry(start_y=int(center_y - core_radius - 10), target_y=int(center_y + core_radius + 5))


def disk_field(s, core_radius=8, penumbra=5, N=64, center=(32, 32)):
    """Idealized scar: S = s inside the scar zone, 1 elsewhere. Returns (S, core, zone)."""
    yy, xx = np.mgrid[:N, :N]
    r2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    core = r2 <= core_radius ** 2
    zone = r2 <= (core_radius + penumbra) ** 2
    S = np.ones((N, N))
    S[zone] = s
    return S, core, zone


def replicate(S, core, zone, keys, n_rep, budget, max_steps=MAX_STEPS, geom=None):
    """Run n_rep populations; return list of per-replicate summaries."""
    geom = geom or geometry()
    out = []
    for r in range(n_rep):
        res = simulate_axons(S, rng_for(*keys, r), N_AXONS, {"cost_budget": budget},
                             geometry=geom, max_steps=max_steps, core_mask=core, zone_mask=zone)
        out.append(summarize(res))
    return out


def agg(reps):
    keys = reps[0].keys()
    return {k: {"mean": float(np.mean([r[k] for r in reps])),
                "sd": float(np.std([r[k] for r in reps], ddof=1)),
                "values": [float(r[k]) for r in reps]} for k in keys}


def s_at_fraction(s_vals, y, frac):
    """Scar level at which y first falls to `frac` of its value at s_vals[0] (= 1, baseline).

    Linear interpolation between grid points; None if y never falls that far
    within the grid, or if there is nothing to fall from.
    """
    s_vals, y = np.asarray(s_vals, float), np.asarray(y, float)
    if y[0] <= 0:
        return None
    target = frac * y[0]
    below = np.flatnonzero(y <= target)
    if below.size == 0:
        return None
    i = below[0]
    if i == 0:
        return float(s_vals[0])
    s0, s1, y0, y1 = s_vals[i - 1], s_vals[i], y[i - 1], y[i]
    return float(s0 + (y0 - target) * (s1 - s0) / (y0 - y1))


def threshold_summary(s_vals, y):
    """S50 and the 90%->10% transition width of a declining curve."""
    s50 = s_at_fraction(s_vals, y, 0.5)
    s90, s10 = s_at_fraction(s_vals, y, 0.9), s_at_fraction(s_vals, y, 0.1)
    return {"s50": s50, "s90": s90, "s10": s10,
            "width_90_10": (s10 - s90) if (s10 is not None and s90 is not None) else None,
            "y_at_baseline": float(y[0])}


def bootstrap_s50(s_vals, rep_matrix, n_boot=500, seed=0):
    """rep_matrix: (n_s, n_rep). Percentile 95% CI for S50 over replicate resamples."""
    rng = np.random.default_rng([BASE_SEED, 999, seed])
    est = []
    n_rep = rep_matrix.shape[1]
    for _ in range(n_boot):
        idx = rng.integers(0, n_rep, (rep_matrix.shape[0], n_rep))
        v = s_at_fraction(s_vals, np.take_along_axis(rep_matrix, idx, axis=1).mean(axis=1), 0.5)
        if v is not None:
            est.append(v)
    if len(est) < 0.95 * n_boot:
        return None
    return [float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))]


def save(name, obj):
    with open(f"results/{name}.json", "w") as f:
        json.dump(obj, f, indent=1)


def model_field(params=None, therapies=(), core_radius=8):
    m = ScarModel(params=params, therapies=therapies, core_radius=core_radius)
    m.run(HOURS)
    return m


# ----------------------------------------------------------------------------
# experiments
# ----------------------------------------------------------------------------
def e1():
    """Uninjured cord: S = 1 everywhere, no lesion."""
    S = np.ones((64, 64))
    none = np.zeros_like(S, dtype=bool)
    rows = []
    for i, B in enumerate([25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400]):
        a = agg(replicate(S, none, none, (1, i), 20, B))
        a2025 = agg(replicate(S, none, none, (1, 100 + i), 20, B, max_steps=MAX_STEPS_2025))
        rows.append({"budget": B, "crossing": a["crossing"]["mean"], "sd": a["crossing"]["sd"],
                     "crossing_200step_cap": a2025["crossing"]["mean"]})
        print(f"E1 B={B:5.0f}  uninjured crossing {a['crossing']['mean']:5.1f}%  (200-step cap: {a2025['crossing']['mean']:5.1f}%)", flush=True)
    # Unlimited walk: path statistics in uninjured tissue
    res = simulate_axons(S, rng_for(1, 999), 4000, {"cost_budget": 1e12}, max_steps=10000)
    n = res["n_steps"][res["crossed"]] - 1
    save("e1_uninjured_calibration", {
        "rows": rows, "B_ref": B_REF, "healthy_step_cost": 0.85,
        "straight_path_steps": 31,
        "unlimited_walk": {"crossing": float(100 * res["crossed"].mean()), "steps_mean": float(n.mean()),
                           "steps_median": float(np.median(n)), "steps_p90": float(np.percentile(n, 90)),
                           "rows_per_step": float(31 / n.mean())}})


GAMMAS = [0.0, 0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.010, 0.012, 0.016, 0.022, 0.035, 0.05, 0.08]
PROD_MULT = [1.0, 0.8, 0.6, 0.45, 0.3, 0.2, 0.15, 0.1, 0.05, 0.02, 0.0]


def e2():
    """Crossing vs lesion CSPG at the 2025 settings and at B_REF."""
    settings = {"settings_2025": (B_2025, MAX_STEPS_2025), "calibrated": (B_REF, MAX_STEPS)}
    out = {"routes": {}}
    for route, levels in [("lesion_therapy", GAMMAS), ("reduced_production", PROD_MULT)]:
        rows = []
        for i, v in enumerate(levels):
            if route == "lesion_therapy":
                m = model_field(therapies=[Therapy(v, scope="lesion")] if v > 0 else ())
            else:
                p = {"alpha_peak": 0.028 * v, "alpha_acute": 0.012 * v, "eta_secondary": 0.018 * v}
                m = model_field(params=p)
            row = {"level": v, "core_S": m.core_mean(), "healthy_S": m.healthy_mean()}
            for sname, (B, ms) in settings.items():
                row[sname] = agg(replicate(m.S, m.core, m.zone, (2, route == "lesion_therapy", i, B), 30, B, ms))
            rows.append(row)
            print(f"E2 {route:18s} level {v:<6} core S {row['core_S']:.2f}  "
                  f"2025: {row['settings_2025']['crossing']['mean']:5.1f}%  "
                  f"calibrated: {row['calibrated']['crossing']['mean']:5.1f}% "
                  f"(through core {row['calibrated']['through_core']['mean']:5.1f}%, bypass {row['calibrated']['bypass']['mean']:5.1f}%)", flush=True)
        out["routes"][route] = rows
    S1 = np.ones((64, 64)); none = np.zeros_like(S1, dtype=bool)
    out["uninjured_calibrated"] = agg(replicate(S1, none, none, (2, 9), 30, B_REF))
    save("e2_crossing_vs_scar", out)


S_GRID = [round(v, 3) for v in np.linspace(1.0, 6.0, 26)]


def path_cost_prediction(budgets, s_grid, core_radius=8, geom=None):
    """Predict through-core crossing from path statistics of an unlimited walk.

    With S = 1 outside the scar zone and s inside it, a path that makes n_h
    moves in healthy tissue and n_z moves in the zone costs
        n_h * c(1) + n_z * c(s),   c(S) = c0 + c_S * S.
    Paths are sampled once without a budget (on the disk field at s = 1.0001,
    so the zone boundary still biases direction), and an axon is predicted to
    cross through the core if that cost is within B.
    """
    S, core, zone = disk_field(1.0001, core_radius)
    seed = rng_for(3, 777, core_radius) if geom is None else rng_for(3, 778, core_radius, geom.start_y, geom.target_y)
    res = simulate_axons(S, seed, 8000, {"cost_budget": 1e12},
                         geometry=geom or geometry(core_radius), max_steps=10000, core_mask=core, zone_mask=zone)
    ok = res["crossed"] & res["entered_core"]
    n_tot = res["n_steps"] - 1
    n_z = res["zone_steps"].astype(float)
    n_h = n_tot - n_z
    pred = {}
    for B in budgets:
        curve = []
        for s in s_grid:
            c = n_h * 0.85 + n_z * (0.05 + 0.8 * s)
            curve.append(float(100 * np.mean(ok & (c <= B))))
        pred[str(B)] = curve
    stats = {"through_core_unlimited": float(100 * ok.mean()),
             "median_zone_moves": float(np.median(n_z[ok])), "median_healthy_moves": float(np.median(n_h[ok]))}
    return pred, stats


def e3():
    budgets = [100, 150, 200, 300, 400, 600]
    pred, stats = path_cost_prediction(budgets, S_GRID)
    out = {"s_grid": S_GRID, "budgets": budgets, "path_stats": stats, "rows": []}
    for j, B in enumerate(budgets):
        mats = {"through_core": [], "crossing": [], "bypass": []}
        for i, s in enumerate(S_GRID):
            S, core, zone = disk_field(s)
            reps = replicate(S, core, zone, (3, j, i), 20, B)
            for k in mats:
                mats[k].append([r[k] for r in reps])
        tc = np.array(mats["through_core"])
        means = tc.mean(axis=1)
        sim = threshold_summary(S_GRID, means)
        prd = threshold_summary(S_GRID, pred[str(B)])
        row = {"budget": B,
               "through_core_mean": means.tolist(),
               "through_core_reps": tc.tolist(),
               "crossing_mean": np.array(mats["crossing"]).mean(axis=1).tolist(),
               "bypass_mean": np.array(mats["bypass"]).mean(axis=1).tolist(),
               "simulated": sim, "s50_ci": bootstrap_s50(S_GRID, tc, seed=j),
               "predicted_through_core": pred[str(B)], "predicted": prd}
        out["rows"].append(row)
        print(f"E3 B={B:4d}  through-core at baseline {sim['y_at_baseline']:5.1f}%  S50 sim {sim['s50']}  "
              f"CI {row['s50_ci']}  pred {prd['s50']}  width90-10 {sim['width_90_10']}", flush=True)
    save("e3_budget_sensitivity", out)


E4_RADII = [5, 8, 11]


def e4():
    """S50 vs lesion size at B_REF, two ways.

    scaled_geometry  start row 10 rows above the core, target row at the zone's
                     lower edge (as for the reference lesion), so the straight-line
                     distance grows with the lesion: 25, 31, 37 rows.
    fixed_geometry   the start and target rows of the largest lesion (11 and 48)
                     for every lesion size, so only the lesion changes.
    """
    fixed = geometry(max(E4_RADII))
    S1 = np.ones((64, 64)); none = np.zeros_like(S1, dtype=bool)
    out = {"s_grid": S_GRID, "budget": B_REF, "radii": E4_RADII}
    for v, vname in enumerate(("scaled_geometry", "fixed_geometry")):
        rows = []
        for j, r in enumerate(E4_RADII):
            geom = geometry(r) if vname == "scaled_geometry" else fixed
            jj = j + 10 * v                       # v = 0 keeps the original seeds
            mat = []
            for i, s in enumerate(S_GRID):
                S, core, zone = disk_field(s, core_radius=r)
                reps = replicate(S, core, zone, (4, jj, i), 20, B_REF, geom=geom)
                mat.append([x["through_core"] for x in reps])
            mat = np.array(mat)
            sim = threshold_summary(S_GRID, mat.mean(axis=1))
            unc = agg(replicate(S1, none, none, (4, 50 + jj), 20, B_REF, geom=geom))["crossing"]
            pred, stats = path_cost_prediction([B_REF], S_GRID, core_radius=r, geom=geom)
            rows.append({"core_radius": r, "zone_radius": r + 5, "start_y": geom.start_y, "target_y": geom.target_y,
                         "uninjured_crossing": {"mean": unc["mean"], "sd": unc["sd"]},
                         "through_core_mean": mat.mean(axis=1).tolist(), "through_core_reps": mat.tolist(),
                         "simulated": sim, "s50_ci": bootstrap_s50(S_GRID, mat, seed=100 + jj),
                         "path_stats": stats, "predicted": threshold_summary(S_GRID, pred[str(B_REF)])})
            print(f"E4 {vname} core radius {r} rows {geom.start_y}->{geom.target_y}: uninjured {unc['mean']:.1f}%  "
                  f"through-core at baseline {sim['y_at_baseline']:.1f}%  S50 {sim['s50']}  CI {rows[-1]['s50_ci']}  "
                  f"zone/healthy moves {stats['median_zone_moves']}/{stats['median_healthy_moves']}", flush=True)
        out[vname] = rows
    save("e4_lesion_size", out)


def e5():
    settings = {"settings_2025": (B_2025, MAX_STEPS_2025), "calibrated": (B_REF, MAX_STEPS)}
    out = {"rows": []}
    for i, g in enumerate(GAMMAS):
        row = {"gamma": g}
        for scope in ("lesion", "global"):
            m = model_field(therapies=[Therapy(g, scope=scope)] if g > 0 else ())
            row[scope] = {"core_S": m.core_mean(), "healthy_S": m.healthy_mean()}
            for sname, (B, ms) in settings.items():
                row[scope][sname] = agg(replicate(m.S, m.core, m.zone, (5, scope == "global", i, B), 20, B, ms))
        out["rows"].append(row)
        print(f"E5 gamma {g:<5}  healthy S global {row['global']['healthy_S']:.2f} | 2025 settings: "
              f"global {row['global']['settings_2025']['crossing']['mean']:5.1f}% lesion {row['lesion']['settings_2025']['crossing']['mean']:5.1f}% | "
              f"calibrated: global {row['global']['calibrated']['crossing']['mean']:5.1f}% "
              f"(through core {row['global']['calibrated']['through_core']['mean']:4.1f}) "
              f"lesion {row['lesion']['calibrated']['crossing']['mean']:5.1f}% (through core {row['lesion']['calibrated']['through_core']['mean']:4.1f})",
              flush=True)
    save("e5_therapy_scope", out)


# Production changes are multipliers on the parameter set in use. Under the
# reference set they give eta = 0.004 and 0.006, alpha_acute = 0.003 and
# alpha_peak = 0.020.
TREATMENTS = {
    "Control": {},
    "chABC, continuous, low dose": {"therapies": [Therapy(0.006, 0, np.inf)]},
    "chABC, continuous, high dose": {"therapies": [Therapy(0.012, 0, np.inf)]},
    "chABC, days 1-7": {"therapies": [Therapy(0.012, 24, 168)]},
    "chABC, days 7-14": {"therapies": [Therapy(0.012, 168, 336)]},
    "Reduced secondary wave": {"scale": {"eta_secondary": 0.004 / 0.018}},
    "Reduced acute + astrocytic production": {"scale": {"alpha_acute": 0.25, "alpha_peak": 0.020 / 0.028}},
    "chABC (continuous) + reduced secondary wave": {"therapies": [Therapy(0.010, 0, np.inf)],
                                                    "scale": {"eta_secondary": 0.006 / 0.018}},
}


def e6():
    with open("results/scar_calibration.json") as f:
        lsq = json.load(f)["least_squares"]["params"]
    out = {"treatments": list(TREATMENTS), "parameter_sets": {}}
    for pset_name, base in [("reference", {}), ("least_squares", lsq)]:
        rows = []
        for i, (name, spec) in enumerate(TREATMENTS.items()):
            params = {**FITTED_PARAMS, **base}
            for k, f in spec.get("scale", {}).items():
                params[k] = params[k] * f
            m = ScarModel(params=params, therapies=spec.get("therapies", ()))
            traj = m.run(HOURS, record_hours=list(np.arange(0, HOURS + 1, 12)))
            a = agg(replicate(m.S, m.core, m.zone, (6, pset_name == "reference", i), 30, B_REF))
            rows.append({"name": name, "core_S_d28": m.core_mean(), "trajectory": traj,
                         "scale": spec.get("scale", {}),
                         "production_params": {k: params[k] for k in ("alpha_acute", "alpha_peak", "eta_secondary")},
                         **{k: a[k] for k in ("crossing", "through_core", "bypass")}})
            print(f"E6 [{pset_name}] {name:44s} core S {m.core_mean():.2f}  crossing {a['crossing']['mean']:5.1f}%  "
                  f"through core {a['through_core']['mean']:5.1f}%", flush=True)
        out["parameter_sets"][pset_name] = rows
    S1 = np.ones((64, 64)); none = np.zeros_like(S1, dtype=bool)
    out["uninjured"] = agg(replicate(S1, none, none, (6, 9), 30, B_REF))
    save("e6_treatments", out)


EXPERIMENTS = {"e1": e1, "e2": e2, "e3": e3, "e4": e4, "e5": e5, "e6": e6}

if __name__ == "__main__":
    chosen = sys.argv[1:] or list(EXPERIMENTS)
    for name in chosen:
        t0 = time.time()
        EXPERIMENTS[name]()
        print(f"[{name} done in {time.time() - t0:.0f}s]", flush=True)
