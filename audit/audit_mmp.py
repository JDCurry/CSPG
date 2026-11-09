"""
Audit of the initial implementation: effect of its auxiliary MMP/TIMP fields.

The initial scar model carried MMP (M) and TIMP (T) fields. M is set to 0.5 in
the lesion core at injury and decays within hours; T only inhibits M. Their only
effect on CSPG is the term -beta_mmp * M * S, so beta_mmp = 0 removes both.

This script checks two things:
  1. the calibrated core trajectory (32x32 grid, dt 1 h, core radius 5, as in
     the calibration) with and without MMP;
  2. the initial threshold sweep (whole-grid therapy, budget 25, 200-step cap,
     the same axon parameters at every level, the same seeds as the
     fixed_default sweep in audit_threshold.py) with and without MMP.

MMP diffuses into healthy tissue before it decays and leaves a very small CSPG
gradient there. The direction rule responds to the sign of a CSPG difference,
not its size, so this gradient changes crossing even though it does not change
the core trajectory.

    python audit/audit_mmp.py      (about 3 minutes)
Output: audit/results/mmp_results.json
"""
import contextlib
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "initial_implementation"))
import numpy as np
from scipy.ndimage import binary_dilation
from sci_scar_simulator import SCIScarSimulator
from axon_growth_simulator import AxonGrowthSimulator

BASE = dict(alpha_peak=0.028, alpha_onset=3, alpha_width=6, S_max=5.0, alpha_acute=0.012,
            tau_acute=24, eta_secondary=0.018, eta_onset=36, eta_width=12.0, delta_feedback=0.00025)
GAMMA = {4.5: 0.000, 4.0: 0.003, 3.5: 0.006, 3.0: 0.009, 2.5: 0.012, 2.0: 0.016, 1.5: 0.022}
BETA_DEFAULT = 0.01


def core_trajectory(beta):
    sim = SCIScarSimulator(grid_size=32, dt=1.0)
    p = BASE.copy(); p["beta_mmp"] = beta
    sim.set_parameters(**p)
    sim.create_injury(center=(16, 16), radius=5, injury_type="contusion")
    with contextlib.redirect_stdout(io.StringIO()):
        r = sim.run_to_timepoints([24, 72, 168, 672], sim.injury_mask)
    return [float(v) for v in r["S_values"]]


def scar_field(gamma, beta):
    sim = SCIScarSimulator(grid_size=64, dt=0.5)
    p = BASE.copy(); p.update(gamma_therapy=gamma, beta_mmp=beta)
    sim.set_parameters(**p)
    sim.create_injury(center=(32, 32), radius=8, injury_type="contusion")
    sim.run(duration_hours=672, verbose=False)
    zone = binary_dilation(sim.injury_mask, iterations=5)
    return sim.S.copy(), sim.S[~zone]


def crossing(S, seed):
    np.random.seed(seed)
    a = AxonGrowthSimulator(scar_field=S, grid_size=64, lesion_center=(32, 32), lesion_radius=8)
    a.add_population("CST", n_axons=250, tract_type="descending")
    with contextlib.redirect_stdout(io.StringIO()):
        r = a.run(max_steps=200, verbose=False)
    return 100 * r["crossing_rate"]


if __name__ == "__main__":
    with_mmp, without = core_trajectory(BETA_DEFAULT), core_trajectory(0.0)
    out = {"core_trajectory_days_1_3_7_28": {"with_mmp": with_mmp, "without_mmp": without,
                                             "max_abs_difference": float(np.max(np.abs(np.subtract(with_mmp, without))))},
           "sweep": []}
    print("core trajectory max |difference|:", out["core_trajectory_days_1_3_7_28"]["max_abs_difference"], flush=True)
    for target, g in GAMMA.items():
        row = {"target": target, "gamma": g}
        for label, beta in [("with_mmp", BETA_DEFAULT), ("without_mmp", 0.0)]:
            S, healthy = scar_field(g, beta)
            cr = [crossing(S, 1000 * k + int(target * 10)) for k in range(15)]
            row[label] = {"core_S": float(S[24:40, 24:40].mean()), "healthy_min": float(healthy.min()),
                          "healthy_max": float(healthy.max()), "cross_mean": float(np.mean(cr)), "cross_sd": float(np.std(cr))}
        out["sweep"].append(row)
        print(target, g, {k: (round(row[k]["core_S"], 3), round(row[k]["cross_mean"], 1),
                              f"{row[k]['healthy_min']:.5f}-{row[k]['healthy_max']:.5f}") for k in ("with_mmp", "without_mmp")}, flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "mmp_results.json"), "w") as f:
        json.dump(out, f, indent=1)
