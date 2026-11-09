"""
Audit of the initial implementation (files in initial_implementation/, unmodified).

Reruns the initial threshold sweep (whole-grid therapy, budget 25, 200-step cap)
with fixed seeds, three ways:
  original_conditional  axon parameters switched by scar target, as in threshold_analysis.py
  fixed_default         the same axon parameters for every level
  fixed_permissive      the more permissive parameters for every level
and records, for each level, lesion-core CSPG, healthy-tissue CSPG, crossing,
and the fraction of crossing axons whose path entered the lesion core.

    python audit/audit_threshold.py     (about 5 minutes)
Output: audit/results/sweep_results.json
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "initial_implementation"))
import numpy as np
from sci_scar_simulator import SCIScarSimulator
from axon_growth_simulator import AxonGrowthSimulator

BASE = dict(alpha_peak=0.028, alpha_onset=3, alpha_width=6, S_max=5.0, alpha_acute=0.012,
            tau_acute=24, eta_secondary=0.018, eta_onset=36, eta_width=12.0, delta_feedback=0.00025)
GAMMA = {4.5: 0.000, 4.0: 0.003, 3.5: 0.006, 3.0: 0.009, 2.5: 0.012, 2.0: 0.016, 1.5: 0.022}
yy, xx = np.ogrid[:64, :64]
CORE = (xx - 32) ** 2 + (yy - 32) ** 2 <= 8 ** 2
_cache = {}


def scar_field(gamma):
    if gamma not in _cache:
        sim = SCIScarSimulator(grid_size=64, dt=0.5)
        p = BASE.copy(); p["gamma_therapy"] = gamma
        sim.set_parameters(**p)
        sim.create_injury(center=(32, 32), radius=8, injury_type="contusion")
        sim.run(duration_hours=672, verbose=False)
        _cache[gamma] = sim.S.copy()
    return _cache[gamma]


def run_axons(S, n_axons, seed, axon_params=None):
    np.random.seed(seed)
    a = AxonGrowthSimulator(scar_field=S, grid_size=64, lesion_center=(32, 32), lesion_radius=8)
    a.add_population("CST", n_axons=n_axons, tract_type="descending")
    if axon_params:
        a.set_parameters(**axon_params)
    r = a.run(max_steps=200, verbose=False)
    pop = a.populations["CST"]
    idx = np.where(pop.crossed)[0]
    through = sum(any(CORE[y, x] for (x, y) in pop.paths[i]) for i in idx)
    return r["crossing_rate"] * 100, (through / len(idx) if len(idx) else None)


def original_params(target):
    if target < 2.5:
        return dict(cost_threshold=28.0, growth_rate=0.96)
    if target < 3.5:
        return dict(cost_threshold=26.0, growth_rate=0.95)
    return None


if __name__ == "__main__":
    out = {}
    for label, pfun in [("original_conditional", original_params), ("fixed_default", lambda t: None),
                        ("fixed_permissive", lambda t: dict(cost_threshold=28.0, growth_rate=0.96))]:
        rows = []
        for target, g in GAMMA.items():
            S = scar_field(g)
            res = [run_axons(S, 250, 1000 * k + int(target * 10), pfun(target)) for k in range(15)]
            cr = [c for c, _ in res]; th = [f for _, f in res if f is not None]
            rows.append(dict(target=target, gamma=g, core_S=float(S[24:40, 24:40].mean()),
                             healthy_S=float(S[:6, :6].mean()), cross_mean=float(np.mean(cr)),
                             cross_sd=float(np.std(cr)), fraction_of_crossers_through_core=float(np.mean(th)) if th else None))
            print(label, rows[-1], flush=True)
        out[label] = rows
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "sweep_results.json"), "w") as f:
        json.dump(out, f, indent=1)
