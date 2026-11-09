"""
Controls on the initial implementation (budget 25, 200-step cap).

  lesion_only_therapy  therapy restricted to the production zone (healthy tissue untouched)
  production_scaled    scar lowered by reducing production, no therapy
  no_lesion_uniform    no lesion; uniform CSPG everywhere

    python audit/audit_controls.py      (about 3 minutes)
Output: audit/results/control_results.json
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "initial_implementation"))
sys.path.insert(0, HERE)
import numpy as np
from scipy.ndimage import binary_dilation
from sci_scar_simulator import SCIScarSimulator
from audit_threshold import BASE, run_axons


class LesionOnlyTherapy(SCIScarSimulator):
    def compute_dS(self):
        dS = super().compute_dS()
        zone = binary_dilation(self.injury_mask, iterations=5)
        dS[~zone] += self.get_gamma_therapy() * self.S[~zone]
        return dS


def field(cls, params):
    s = cls(grid_size=64, dt=0.5); s.set_parameters(**params)
    s.create_injury(center=(32, 32), radius=8, injury_type="contusion"); s.run(duration_hours=672, verbose=False)
    return s.S.copy()


if __name__ == "__main__":
    out = {}
    rows = []
    for g in [0.0, 0.003, 0.006, 0.009, 0.012, 0.016, 0.022, 0.035]:
        p = BASE.copy(); p["gamma_therapy"] = g; S = field(LesionOnlyTherapy, p)
        cr = [run_axons(S, 250, 500 + k)[0] for k in range(10)]
        rows.append(dict(gamma=g, core_S=float(S[24:40, 24:40].mean()), healthy_S=float(S[:6, :6].mean()),
                         cross=float(np.mean(cr)), sd=float(np.std(cr))))
        print("lesion-only therapy", rows[-1], flush=True)
    out["lesion_only_therapy"] = rows
    rows = []
    for m in [1.0, 0.8, 0.6, 0.45, 0.3, 0.15]:
        p = BASE.copy(); p.update(alpha_peak=0.028 * m, alpha_acute=0.012 * m, eta_secondary=0.018 * m, gamma_therapy=0.0)
        S = field(SCIScarSimulator, p)
        cr = [run_axons(S, 250, 600 + k)[0] for k in range(10)]
        rows.append(dict(prod_mult=m, core_S=float(S[24:40, 24:40].mean()), healthy_S=float(S[:6, :6].mean()),
                         cross=float(np.mean(cr)), sd=float(np.std(cr))))
        print("production-scaled", rows[-1], flush=True)
    out["production_scaled"] = rows
    rows = []
    for hS in [1.0, 0.86, 0.6, 0.4, 0.2, 0.1]:
        S = np.full((64, 64), hS)
        cr = [run_axons(S, 250, 700 + k)[0] for k in range(10)]
        rows.append(dict(uniform_S=hS, cross=float(np.mean(cr)), sd=float(np.std(cr))))
        print("no lesion, uniform S", rows[-1], flush=True)
    out["no_lesion_uniform"] = rows
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "control_results.json"), "w") as f:
        json.dump(out, f, indent=1)
