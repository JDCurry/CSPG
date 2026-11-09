"""
CSPG scar accumulation model
============================

S(x, y, t) is the inhibitory CSPG level of spinal cord tissue, expressed as
fold change relative to uninjured tissue (S = 1). The lesion is a disk on a
2-D grid: a core of radius `core_radius` plus a penumbra rim of width
`penumbra`. Together they form the scar zone Z.

The model is a pointwise ODE evaluated on the grid. S does not diffuse
(CSPGs are matrix-bound), so the "field" is a set of independent ODEs that
differ only by whether a pixel is inside the scar zone and by any therapy
mask:

    dS/dt = Z(x) * [ alpha(t) * (1 - S/S_max)_+  -  delta * S^2 ]
            - sum_k  gamma_k * W_k(t) * R_k(x) * S

    alpha(t) = a_acute * exp(-t / tau)                 (acute phase)
             + a_peak  * sig((t - t1) / w1)             (astrocytic phase)
             + eta     * sig((t - t2) / w2)             (secondary wave)

Z(x) is 1 inside the scar zone and 0 elsewhere, so tissue outside the zone
stays at baseline unless a therapy is applied to it. Each therapy k is a
first-order degradation term with rate gamma_k, a time window W_k(t), and a
spatial scope R_k(x): either the scar zone ("lesion") or the whole grid
("global"). The global scope exists only as a negative control; see
CHANGES.md.

Changes from the 2025 version are listed in CHANGES.md.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

# Parameters selected by the grid search in calibrate_scar.py (all rates per hour).
FITTED_PARAMS: Dict[str, float] = {
    "alpha_acute": 0.012,     # acute-phase amplitude
    "tau_acute": 24.0,        # acute-phase decay constant (h)
    "alpha_peak": 0.028,      # astrocytic-phase amplitude
    "alpha_onset": 3.0,       # astrocytic-phase midpoint t1 (h)
    "alpha_width": 6.0,       # astrocytic-phase width w1 (h)
    "eta_secondary": 0.018,   # secondary-wave amplitude
    "eta_onset": 36.0,        # secondary-wave midpoint t2 (h)
    "eta_width": 12.0,        # secondary-wave width w2 (h)
    "delta_feedback": 0.00025,  # quadratic self-limitation
    "S_max": 5.0,             # production saturation ceiling
}

S_FLOOR = 0.1  # numerical floor on S (only reachable under strong therapy)


@dataclass
class Therapy:
    """First-order CSPG degradation (e.g. chondroitinase ABC)."""

    gamma: float                 # degradation rate (1/h)
    start_h: float = 0.0         # window start (h post-injury), inclusive
    end_h: float = np.inf        # window end (h post-injury), exclusive
    scope: str = "lesion"        # "lesion" (scar zone only) or "global" (whole grid)

    def active(self, t: float) -> bool:
        return self.start_h <= t < self.end_h


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


class ScarModel:
    """CSPG accumulation on a square grid with a circular lesion."""

    def __init__(
        self,
        grid_size: int = 64,
        dt: float = 0.5,
        center: tuple = (32, 32),
        core_radius: float = 8,
        penumbra: float = 5,
        initial_core_S: float = 0.8,
        params: Optional[Dict[str, float]] = None,
        therapies: Iterable[Therapy] = (),
    ):
        self.N = grid_size
        self.dt = dt
        self.p = dict(FITTED_PARAMS)
        if params:
            unknown = set(params) - set(self.p)
            if unknown:
                raise KeyError(f"Unknown parameters: {sorted(unknown)}")
            self.p.update(params)
        self.therapies: List[Therapy] = list(therapies)
        for th in self.therapies:
            if th.scope not in ("lesion", "global"):
                raise ValueError(f"Unknown therapy scope: {th.scope}")

        yy, xx = np.mgrid[: self.N, : self.N]
        r2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
        self.core = r2 <= core_radius ** 2
        self.zone = r2 <= (core_radius + penumbra) ** 2
        self._everywhere = np.ones((self.N, self.N), dtype=bool)

        # Contusion: lesion core starts slightly below baseline (initial tissue loss).
        self.S = np.ones((self.N, self.N))
        self.S[self.core] = initial_core_S
        self.t = 0.0

    # ------------------------------------------------------------------
    def alpha(self, t: float) -> float:
        p = self.p
        acute = p["alpha_acute"] * np.exp(-t / p["tau_acute"])
        astro = p["alpha_peak"] * _sigmoid((t - p["alpha_onset"]) / p["alpha_width"])
        wave = p["eta_secondary"] * _sigmoid((t - p["eta_onset"]) / p["eta_width"])
        return acute + astro + wave

    def step(self) -> None:
        p = self.p
        S = self.S
        dS = np.zeros_like(S)
        Sz = S[self.zone]
        production = np.maximum(self.alpha(self.t) * (1.0 - Sz / p["S_max"]), 0.0)
        dS[self.zone] = production - p["delta_feedback"] * Sz ** 2
        for th in self.therapies:
            if th.active(self.t):
                mask = self.zone if th.scope == "lesion" else self._everywhere
                dS[mask] -= th.gamma * S[mask]
        self.S = np.maximum(S + self.dt * dS, S_FLOOR)
        self.t += self.dt

    def run(self, hours: float, record_hours: Optional[Iterable[float]] = None) -> Dict[str, list]:
        """Advance `hours`; optionally record the core mean at given times (h)."""
        record = sorted(record_hours) if record_hours is not None else []
        out = {"t_h": [], "core_S": []}
        n_steps = int(round(hours / self.dt))
        ri = 0
        while ri < len(record) and record[ri] <= self.t + 1e-9:
            out["t_h"].append(self.t); out["core_S"].append(self.core_mean()); ri += 1
        for _ in range(n_steps):
            self.step()
            while ri < len(record) and record[ri] <= self.t + 1e-9:
                out["t_h"].append(self.t); out["core_S"].append(self.core_mean()); ri += 1
        return out

    # ------------------------------------------------------------------
    def core_mean(self) -> float:
        return float(self.S[self.core].mean())

    def zone_mean(self) -> float:
        return float(self.S[self.zone].mean())

    def healthy_mean(self) -> float:
        return float(self.S[~self.zone].mean())


def scar_field(hours: float = 672.0, **kwargs) -> ScarModel:
    """Convenience: build a model, run it for `hours`, return it."""
    m = ScarModel(**kwargs)
    m.run(hours)
    return m
