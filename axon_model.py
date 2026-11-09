"""
Cost-budget axon growth model
=============================

Growth cones perform a caudally biased random walk over a static CSPG field
S(x, y) (fold change; healthy tissue = 1). The rules are those of the 2025
version (see CHANGES.md); this implementation is vectorized over axons and
takes an explicit random generator so runs are reproducible.

At each step, for each active growth cone at (x, y):

1. Extension attempt. The cone extends with probability
       p = clip(g * exp(-k * S(x, y)), 0.01, 0.99);
   otherwise it stays put this step (no cost).
2. Direction. One of the 8 neighbours is drawn with weight
       w = (1 + b)        if the move is caudal (dy > 0)
         * (1 + s)        if S decreases along the move
         * (1 - s/2)      if S increases along the move,
   with w = 0 for moves off the grid.
3. Cost. The cone pays c = c0 + c_S * S at the destination. If the
   accumulated cost exceeds the budget B, the cone collapses without moving.
4. Success. A cone that reaches row `target_y` has crossed.

The budget rule is a hard cutoff on summed path cost, so crossing probability
falls steeply once the cost of the cheapest likely path exceeds B. Any
"threshold" in scar density therefore has a location set by B, c_S, c0 and
the path geometry; see the analysis in run_experiments.py.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

# (dx, dy) for the 8-connected neighbourhood, same order as the 2025 version.
DIRS = np.array([(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)])

DEFAULT_AXON_PARAMS: Dict[str, float] = {
    "growth_rate": 0.95,          # g
    "S_inhibition": 0.3,          # k
    "baseline_cost": 0.05,        # c0
    "cost_per_S": 0.8,            # c_S
    "cost_budget": 25.0,          # B (2025 value; recalibrated in run_experiments.py)
    "caudal_bias": 0.7,           # b
    "gradient_sensitivity": 0.4,  # s
}

# The 2025 version stopped every walk after 200 steps. Because caudal drift is
# weak (about 0.21 rows per move), that cap alone blocks about half of all
# axons in uninjured tissue. The experiments use a non-binding cap instead.
MAX_STEPS_2025 = 200
MAX_STEPS = 2000


@dataclass
class AxonGeometry:
    grid_size: int = 64
    start_y: int = 14             # lesion centre y (32) - core radius (8) - 10
    target_y: int = 45            # lesion centre y (32) + core radius (8) + 5
    x_low: int = 10               # starting columns drawn from [x_low, x_high)
    x_high: int = 54


def simulate_axons(
    S: np.ndarray,
    rng: np.random.Generator,
    n_axons: int = 250,
    params: Optional[Dict[str, float]] = None,
    geometry: AxonGeometry = AxonGeometry(),
    max_steps: int = MAX_STEPS,
    core_mask: Optional[np.ndarray] = None,
    zone_mask: Optional[np.ndarray] = None,
    record_paths: bool = False,
) -> Dict[str, np.ndarray]:
    """Grow one population of descending axons through field S.

    Returns per-axon arrays: crossed, start_x, entered_core, entered_zone,
    cost, and n_steps (positions visited).
    """
    p = dict(DEFAULT_AXON_PARAMS)
    if params:
        unknown = set(params) - set(p)
        if unknown:
            raise KeyError(f"Unknown axon parameters: {sorted(unknown)}")
        p.update(params)
    N = S.shape[0]
    g = geometry
    if core_mask is None:
        core_mask = np.zeros_like(S, dtype=bool)
    if zone_mask is None:
        zone_mask = np.zeros_like(S, dtype=bool)

    x = rng.integers(g.x_low, g.x_high, n_axons)
    y = np.full(n_axons, g.start_y)
    start_x = x.copy()
    active = np.ones(n_axons, dtype=bool)
    crossed = np.zeros(n_axons, dtype=bool)
    cost = np.zeros(n_axons)
    n_pos = np.ones(n_axons, dtype=int)
    entered_core = core_mask[y, x].copy()
    entered_zone = zone_mask[y, x].copy()
    zone_steps = np.zeros(n_axons, dtype=int)  # moves that landed inside the scar zone
    hist_x, hist_y = ([x.copy()], [y.copy()]) if record_paths else (None, None)

    dx_all = DIRS[:, 0][None, :]
    dy_all = DIRS[:, 1][None, :]
    b, s = p["caudal_bias"], p["gradient_sensitivity"]

    for _ in range(max_steps):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        done = y[idx] >= g.target_y
        crossed[idx[done]] = True
        active[idx[done]] = False
        idx = idx[~done]
        if idx.size == 0:
            continue

        xs, ys = x[idx], y[idx]
        S_here = S[ys, xs]
        p_grow = np.clip(p["growth_rate"] * np.exp(-S_here * p["S_inhibition"]), 0.01, 0.99)
        grow = rng.random(idx.size) <= p_grow
        idx, xs, ys, S_here = idx[grow], xs[grow], ys[grow], S_here[grow]
        if idx.size == 0:
            continue

        nx = xs[:, None] + dx_all
        ny = ys[:, None] + dy_all
        inside = (nx >= 0) & (nx < N) & (ny >= 0) & (ny < N)
        S_next = S[np.clip(ny, 0, N - 1), np.clip(nx, 0, N - 1)]
        w = np.ones(nx.shape)
        caudal = ((g.target_y > ys[:, None]) & (dy_all > 0)) | ((g.target_y < ys[:, None]) & (dy_all < 0))
        w = np.where(caudal, w * (1 + b), w)
        w = np.where(S_next < S_here[:, None], w * (1 + s),
                     np.where(S_next > S_here[:, None], w * (1 - 0.5 * s), w))
        w = np.where(inside, w, 0.0)
        total = w.sum(axis=1)
        stuck = total <= 0
        active[idx[stuck]] = False
        idx, xs, ys, w, total = idx[~stuck], xs[~stuck], ys[~stuck], w[~stuck], total[~stuck]
        if idx.size == 0:
            continue
        cdf = np.cumsum(w / total[:, None], axis=1)
        choice = np.minimum((rng.random(idx.size)[:, None] >= cdf).sum(axis=1), 7)
        nxs = np.clip(xs + DIRS[choice, 0], 0, N - 1)
        nys = np.clip(ys + DIRS[choice, 1], 0, N - 1)

        cost[idx] += p["baseline_cost"] + p["cost_per_S"] * S[nys, nxs]
        collapse = cost[idx] > p["cost_budget"]
        active[idx[collapse]] = False
        move = ~collapse
        mi = idx[move]
        x[mi], y[mi] = nxs[move], nys[move]
        n_pos[mi] += 1
        entered_core[mi] |= core_mask[nys[move], nxs[move]]
        entered_zone[mi] |= zone_mask[nys[move], nxs[move]]
        zone_steps[mi] += zone_mask[nys[move], nxs[move]]
        arrived = nys[move] >= g.target_y
        crossed[mi[arrived]] = True
        active[mi[arrived]] = False
        if record_paths:
            hist_x.append(x.copy()); hist_y.append(y.copy())

    extra = {"path_x": np.array(hist_x), "path_y": np.array(hist_y)} if record_paths else {}
    return {
        **extra,
        "crossed": crossed,
        "start_x": start_x,
        "entered_core": entered_core,
        "entered_zone": entered_zone,
        "cost": cost,
        "n_steps": n_pos,
        "zone_steps": zone_steps,
    }


def summarize(res: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Crossing fractions (percent of all axons in the population)."""
    n = res["crossed"].size
    c = res["crossed"]
    return {
        "crossing": 100.0 * c.mean(),
        "through_core": 100.0 * (c & res["entered_core"]).sum() / n,
        "through_zone": 100.0 * (c & res["entered_zone"]).sum() / n,
        "bypass": 100.0 * (c & ~res["entered_zone"]).sum() / n,
    }
