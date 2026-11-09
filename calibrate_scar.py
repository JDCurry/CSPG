"""
Calibrate the scar model against the stylized CSPG trajectory.

The target trajectory (fold change 1.0, 1.5, 3.0, 4.0, 4.2 at 0 h, 1 d, 3 d,
7 d, 28 d) is a stylized aggregate synthesized from qualitative descriptions
of CSPG accumulation after rodent SCI. It is not a measured dataset. With five
target points (four informative, because S(0) is fixed by the initial
condition) and five swept parameters, the fit is a calibration, not a
validation, and the parameters are not identifiable. This script reports:

1. The 2025 grid search (48 combinations), reproduced on the rewritten model.
2. The same grid extended beyond the edge on which each optimum lay (one
   step for four parameters, two for delta_feedback), to check whether the
   optimum is interior.
3. A bounded least-squares refit (each parameter within 0.05-20 times its
   reference value), to show how far the parameters can move while matching
   the target equally well or better. alpha_acute ends at the lower bound.

Output: results/scar_calibration.json
"""

import itertools
import json
import os

import numpy as np
from scipy.optimize import least_squares

from scar_model import FITTED_PARAMS, ScarModel

T_H = np.array([0, 24, 72, 168, 672], dtype=float)
TARGET = np.array([1.0, 1.5, 3.0, 4.0, 4.2])
FREE = ["alpha_peak", "alpha_acute", "eta_secondary", "eta_onset", "delta_feedback"]


def trajectory(params, grid_size=32, dt=1.0, radius=5):
    """Core-mean trajectory at the target times (the 2025 fit used this small grid)."""
    m = ScarModel(grid_size=grid_size, dt=dt, center=(16, 16), core_radius=radius, params=params)
    return np.array(m.run(672, record_hours=T_H)["core_S"])


def r_squared(obs, pred):
    return float(1 - np.sum((obs - pred) ** 2) / np.sum((obs - obs.mean()) ** 2))


def grid_search(grid):
    best = None
    for combo in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), combo))
        pred = trajectory(params)
        r2 = r_squared(TARGET, pred)
        if best is None or r2 > best["r2"]:
            best = {"params": params, "r2": r2, "pred": pred.tolist()}
    return best


def main():
    os.makedirs("results", exist_ok=True)
    out = {"t_h": T_H.tolist(), "target": TARGET.tolist(),
           "note": "Target is a stylized aggregate trajectory, not measured data."}

    grid_2025 = {
        "alpha_peak": [0.025, 0.028],
        "alpha_acute": [0.010, 0.012],
        "eta_secondary": [0.012, 0.015, 0.018],
        "eta_onset": [36.0, 42.0],
        "delta_feedback": [0.0002, 0.00025],
    }
    best = grid_search(grid_2025)
    best["on_grid_edge"] = {k: (best["params"][k] in (min(v), max(v))) for k, v in grid_2025.items()}
    out["grid_2025"] = best

    grid_ext = {
        "alpha_peak": [0.025, 0.028, 0.031],
        "alpha_acute": [0.010, 0.012, 0.014],
        "eta_secondary": [0.015, 0.018, 0.021],
        "eta_onset": [30.0, 36.0, 42.0],
        "delta_feedback": [0.00025, 0.0003, 0.00035],
    }
    out["grid_extended"] = grid_search(grid_ext)

    x0 = np.array([FITTED_PARAMS[k] for k in FREE])
    scale = x0.copy()

    def resid(z):
        return trajectory(dict(zip(FREE, z * scale))) - TARGET

    lsq = least_squares(resid, np.ones_like(x0), bounds=(0.05, 20), diff_step=1e-3)
    p_lsq = dict(zip(FREE, (lsq.x * scale).tolist()))
    pred = trajectory(p_lsq)
    out["least_squares"] = {"params": p_lsq, "r2": r_squared(TARGET, pred), "pred": pred.tolist(),
                            "fold_change_vs_used": {k: p_lsq[k] / FITTED_PARAMS[k] for k in FREE},
                            "bounds_relative_to_reference": [0.05, 20],
                            "at_bound": {k: bool(min(abs(z - 0.05), abs(z - 20)) < 1e-3) for k, z in zip(FREE, lsq.x)}}

    used = trajectory(None)
    out["used_params"] = {k: FITTED_PARAMS[k] for k in FITTED_PARAMS}
    out["used_pred"] = used.tolist()
    out["used_r2"] = r_squared(TARGET, used)
    out["used_residuals"] = (used - TARGET).tolist()
    # The same parameters on the 64-grid used for all downstream experiments
    full = ScarModel().run(672, record_hours=T_H)["core_S"]
    out["used_pred_64grid"] = full
    out["used_r2_64grid"] = r_squared(TARGET, np.array(full))
    full_lsq = ScarModel(params=p_lsq).run(672, record_hours=T_H)["core_S"]
    out["least_squares"]["pred_64grid"] = full_lsq
    out["least_squares"]["r2_64grid"] = r_squared(TARGET, np.array(full_lsq))

    with open("results/scar_calibration.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("grid_2025", "grid_extended", "least_squares", "used_r2", "used_r2_64grid")}, indent=1))


if __name__ == "__main__":
    main()
