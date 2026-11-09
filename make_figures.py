"""
Build all manuscript figures from results/*.json (run run_experiments.py first).

    python make_figures.py

Writes figures/fig1..fig6 as PDF (for LaTeX) and PNG (300 dpi).
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from axon_model import simulate_axons
from scar_model import FITTED_PARAMS, ScarModel, Therapy
from run_experiments import s_at_fraction

os.makedirs("figures", exist_ok=True)

# Light-mode palette (reference categorical slots 1-3, blue sequential ramp, neutral inks).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]  # ordinal, steps 250-700
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.5, "lines.markersize": 5, "legend.frameon": False, "legend.fontsize": 7.5,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def load(name):
    with open(f"results/{name}.json") as f:
        return json.load(f)


def load_audit(name):
    with open(f"audit/results/{name}.json") as f:
        return json.load(f)


def panel_label(ax, s):
    ax.text(-0.14, 1.06, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def save(fig, name):
    fig.savefig(f"figures/{name}.pdf")
    fig.savefig(f"figures/{name}.png")
    plt.close(fig)
    print("wrote", name)


# ----------------------------------------------------------------------------
def fig1():
    cal = load("scar_calibration")
    t_d = np.array(cal["t_h"]) / 24
    hours = np.arange(0, 673, 2)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axs[0]
    for params, color, lab in [(None, BLUE, f"Reference parameters ($R^2$ = {cal['used_r2']:.3f})"),
                               (cal["least_squares"]["params"], ORANGE,
                                f"Least-squares refit ($R^2$ = {cal['least_squares']['r2']:.3f})")]:
        m = ScarModel(params=params)
        tr = m.run(672, record_hours=hours)
        ax.plot(np.array(tr["t_h"]) / 24, tr["core_S"], color=color, label=lab)
    ax.plot(t_d, cal["target"], "o", color=INK, ms=5, zorder=5, label="Stylized target (not measured data)")
    ax.set_xlabel("Days after injury"); ax.set_ylabel("Lesion-core CSPG (fold change)")
    ax.set_xlim(0, 28.5); ax.set_ylim(0, 5)
    ax.legend(loc="lower right")
    panel_label(ax, "A")

    ax = axs[1]
    names = {"alpha_acute": r"acute amplitude $\alpha_{acute}$", "alpha_peak": r"astrocytic amplitude $\alpha_{peak}$",
             "eta_secondary": r"secondary-wave amplitude $\eta$", "eta_onset": r"secondary-wave onset $t_2$",
             "delta_feedback": r"self-limitation $\delta$"}
    fc = cal["least_squares"]["fold_change_vs_used"]
    keys = list(names)
    vals = [fc[k] for k in keys]
    y = np.arange(len(keys))
    ax.barh(y, np.log2(vals), color=BLUE, height=0.55)
    ax.axvline(0, color=MUTED, lw=0.8)
    for yi, v in zip(y, vals):
        ax.text(np.log2(v) + (0.15 if v > 1 else -0.15), yi, f"{v:.2g}×", va="center",
                ha="left" if v > 1 else "right", fontsize=7.5, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels([names[k] for k in keys])
    ax.set_xlim(-7.2, 3.4)
    ax.set_xlabel("Refit ÷ reference (log$_2$ scale)")
    ax.grid(axis="y", visible=False)
    panel_label(ax, "B")
    fig.tight_layout()
    save(fig, "fig1_scar_calibration")


# ----------------------------------------------------------------------------
def draw_paths(ax, model, budget, n=70, seed=11, title=""):
    rng = np.random.default_rng(seed)
    res = simulate_axons(model.S, rng, n, {"cost_budget": budget}, core_mask=model.core,
                         zone_mask=model.zone, record_paths=True)
    ax.imshow(model.S, cmap="Blues", vmin=0, vmax=5, origin="upper")
    for i in range(n):
        px, py = res["path_x"][:, i], res["path_y"][:, i]
        if res["crossed"][i]:
            ax.plot(px, py, color=ORANGE, lw=0.9, alpha=0.9)
        else:
            ax.plot(px, py, color=MUTED, lw=0.5, alpha=0.5)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(32 + 8 * np.cos(th), 32 + 8 * np.sin(th), color=INK, lw=0.8, ls="--")
    ax.axhline(45, color=INK2, lw=0.8, ls=":")
    ax.set_xlim(0, 63); ax.set_ylim(63, 0)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(title, fontsize=8)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(GRID)


def fig2():
    e1 = load("e1_uninjured_calibration")
    B = [r["budget"] for r in e1["rows"]]
    fig = plt.figure(figsize=(7.2, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1])
    ax = fig.add_subplot(gs[0])
    ax.plot(B, [r["crossing"] for r in e1["rows"]], "o-", color=BLUE, label="cap 2,000 steps (used here)")
    ax.plot(B, [r["crossing_200step_cap"] for r in e1["rows"]], "s--", color=ORANGE, label="cap 200 steps (initial)")
    for b, lab in [(25, "initial\nbudget"), (e1["B_ref"], "calibrated\nbudget")]:
        ax.axvline(b, color=MUTED, lw=0.8, ls=":")
        ax.text(b + 6, 62, lab, fontsize=7, color=INK2)
    ax.set_xlabel("Cost budget $B$"); ax.set_ylabel("Crossing, uninjured cord (%)")
    ax.set_ylim(-3, 103); ax.legend(loc="lower right", fontsize=6.8)
    panel_label(ax, "A")

    ctrl = ScarModel(); ctrl.run(672)
    trt = ScarModel(therapies=[Therapy(0.012, scope="lesion")]); trt.run(672)
    ax = fig.add_subplot(gs[1])
    draw_paths(ax, ctrl, e1["B_ref"], title=f"Control, core {ctrl.core_mean():.1f}×")
    panel_label(ax, "B")
    ax = fig.add_subplot(gs[2])
    draw_paths(ax, trt, e1["B_ref"], title=f"Lesion-scoped therapy, core {trt.core_mean():.1f}×")
    panel_label(ax, "C")
    handles = [Line2D([], [], color=ORANGE, lw=1.2, label="crossed"), Line2D([], [], color=MUTED, lw=1, label="collapsed"),
               Line2D([], [], color=INK, lw=0.8, ls="--", label="lesion core"), Line2D([], [], color=INK2, lw=0.8, ls=":", label="target row")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.66, -0.06))
    fig.tight_layout()
    save(fig, "fig2_budget_and_paths")


# ----------------------------------------------------------------------------
def fig3():
    e5 = load("e5_therapy_scope")
    rows = e5["rows"]
    core = [r["lesion"]["core_S"] for r in rows]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for ax, setting, title, lab in [(axs[0], "settings_2025", "Initial settings ($B$ = 25, 200-step cap)", "A"),
                                    (axs[1], "calibrated", "Calibrated budget ($B$ = 200)", "B")]:
        for scope, color, name in [("global", BLUE, "whole-grid therapy"), ("lesion", ORANGE, "lesion-scoped therapy")]:
            ax.plot(core, [r[scope][setting]["crossing"]["mean"] for r in rows], "o-", color=color, ms=4,
                    label=f"{name}: all crossing")
            ax.plot(core, [r[scope][setting]["through_core"]["mean"] for r in rows], "--", color=color,
                    label=f"{name}: through core")
        ax.set_xlabel("Lesion-core CSPG (fold change)"); ax.set_title(title, fontsize=8)
        ax.set_ylim(-3, 103); ax.set_xlim(0.3, 4.7)
        panel_label(ax, lab)
    # Initial implementation itself (audit/), whole-grid therapy, initial settings
    sw = load_audit("sweep_results")["original_conditional"]
    mm = load_audit("mmp_results")["sweep"]
    axs[0].plot([r["core_S"] for r in sw], [r["cross_mean"] for r in sw], "D", color=INK, ms=3.8,
                label="initial code as run: all crossing")
    axs[0].plot([r["without_mmp"]["core_S"] for r in mm], [r["without_mmp"]["cross_mean"] for r in mm], "D",
                color=INK, mfc="white", ms=3.8, mew=0.9, label="initial code, MMP field removed")
    axs[0].set_ylabel("Axons crossing (%)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.4, -0.2), fontsize=7)
    ax = axs[2]
    ax.plot(core, [r["global"]["healthy_S"] for r in rows], "o-", color=BLUE, ms=4)
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=":")
    ax.text(4.6, 1.04, "uninjured level", fontsize=7, color=INK2, ha="right", va="bottom")
    ax.set_xlabel("Lesion-core CSPG (fold change)"); ax.set_ylabel("Healthy-tissue CSPG (fold change)")
    ax.set_title("Whole-grid therapy also\nclears healthy tissue", fontsize=8)
    ax.set_ylim(0, 1.15); ax.set_xlim(0.3, 4.7)
    panel_label(ax, "C")
    fig.tight_layout()
    save(fig, "fig3_therapy_scope")


# ----------------------------------------------------------------------------
def fig4():
    e2 = load("e2_crossing_vs_scar")
    fig, ax = plt.subplots(figsize=(4.8, 3.9))
    for route, marker in [("lesion_therapy", "o"), ("reduced_production", "^")]:
        rows = e2["routes"][route]
        x = [r["core_S"] for r in rows]
        for metric, color in [("crossing", BLUE), ("through_core", ORANGE), ("bypass", AQUA)]:
            m = np.array([r["calibrated"][metric]["mean"] for r in rows])
            sd = np.array([r["calibrated"][metric]["sd"] for r in rows])
            ax.errorbar(x, m, yerr=sd, fmt=marker, color=color, ms=5, capsize=0, elinewidth=0.8,
                        mfc=color if marker == "o" else "white", mew=1.0)
    u = e2["uninjured_calibrated"]["crossing"]["mean"]
    ax.axhline(u, color=MUTED, lw=0.8, ls="--")
    ax.text(4.65, u + 1.5, f"uninjured cord ({u:.0f}%)", fontsize=7, color=INK2, ha="right")
    ax.axvline(3.0, color=MUTED, lw=0.8, ls=":")
    ax.text(3.07, 88, "3×: apparent threshold\nin the initial\nimplementation", fontsize=6.8, color=INK2, va="top")
    handles = [Line2D([], [], color=BLUE, lw=0, marker="s", label="all crossing"),
               Line2D([], [], color=ORANGE, lw=0, marker="s", label="crossing through lesion core"),
               Line2D([], [], color=AQUA, lw=0, marker="s", label="crossing around the scar zone"),
               Line2D([], [], color=INK2, lw=0, marker="o", label="scar lowered by lesion-scoped therapy"),
               Line2D([], [], color=INK2, lw=0, marker="^", mfc="white", label="scar lowered by reduced production")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=7)
    ax.set_xlabel("Lesion-core CSPG at day 28 (fold change)"); ax.set_ylabel("Axons crossing (%)")
    ax.set_ylim(-3, 103); ax.set_xlim(0.3, 4.7)
    fig.tight_layout()
    save(fig, "fig4_crossing_vs_scar")


# ----------------------------------------------------------------------------
def fig5():
    e3, e4 = load("e3_budget_sensitivity"), load("e4_lesion_size")
    s = np.array(e3["s_grid"])
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1.3, 1, 0.75]})
    ax = axs[0]
    for row, color in zip(e3["rows"], RAMP):
        ax.plot(s, row["through_core_mean"], "o", color=color, ms=3.5)
        ax.plot(s, row["predicted_through_core"], "-", color=color, lw=1.2, label=f"$B$ = {row['budget']}")
    ax.set_xlabel("Scar-zone CSPG (fold change)"); ax.set_ylabel("Crossing through lesion core (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=6.8,
              title="points: simulated; lines: path-cost prediction", title_fontsize=6.8)
    ax.set_xlim(0.9, 6.1); ax.set_xticks(range(1, 7))
    panel_label(ax, "A")

    ax = axs[1]
    rows = [r for r in e3["rows"] if r["simulated"]["s50"] is not None]
    Bs = [r["budget"] for r in rows]
    s50 = [r["simulated"]["s50"] for r in rows]
    lo = [r["simulated"]["s50"] - r["s50_ci"][0] if r["s50_ci"] else 0 for r in rows]
    hi = [r["s50_ci"][1] - r["simulated"]["s50"] if r["s50_ci"] else 0 for r in rows]
    ax.errorbar(Bs, s50, yerr=[lo, hi], fmt="o", color=BLUE, ms=5, capsize=2, elinewidth=0.9, label="simulated (95% CI)")
    pb, p_same, p_own = [], [], []
    for r in e3["rows"]:
        sim0, pr = r["through_core_mean"][0], np.array(r["predicted_through_core"])
        v = s_at_fraction(e3["s_grid"], pr, 0.5 * sim0 / pr[0]) if pr[0] > 0 else None
        if v is not None and r["predicted"]["s50"] is not None:
            pb.append(r["budget"]); p_same.append(v); p_own.append(r["predicted"]["s50"])
    ax.plot(pb, p_same, "-", color=ORANGE, label="prediction, same baseline")
    ax.plot(pb, p_own, ":", color=ORANGE, label="prediction, own baseline")
    ax.set_xlabel("Cost budget $B$"); ax.set_ylabel("$S_{50}$ (fold change)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=6.8)
    panel_label(ax, "B")

    ax = axs[2]
    for vname, color, dx, lab in [("scaled_geometry", BLUE, -0.09, "start/target move"),
                                  ("fixed_geometry", ORANGE, 0.09, "start/target fixed")]:
        r4 = [r for r in e4[vname] if r["simulated"]["s50"] is not None]
        x = np.array([e4["radii"].index(r["core_radius"]) for r in r4]) + dx
        v = [r["simulated"]["s50"] for r in r4]
        lo = [r["simulated"]["s50"] - r["s50_ci"][0] if r["s50_ci"] else 0 for r in r4]
        hi = [r["s50_ci"][1] - r["simulated"]["s50"] if r["s50_ci"] else 0 for r in r4]
        ax.errorbar(x, v, yerr=[lo, hi], fmt="o", color=color, ms=4.5, capsize=2, elinewidth=0.9, label=lab)
    ax.set_xticks(range(len(e4["radii"]))); ax.set_xticklabels([str(r) for r in e4["radii"]])
    ax.set_xlabel("Lesion-core radius (grid units)"); ax.set_ylabel(f"$S_{{50}}$ at $B$ = {int(e4['budget'])}")
    ax.set_xlim(-0.5, len(e4["radii"]) - 0.5)
    ax.set_ylim(1.5, 3.45)
    ax.legend(loc="upper right", fontsize=6.5, handletextpad=0.2, borderaxespad=0.2)
    panel_label(ax, "C")
    fig.tight_layout()
    save(fig, "fig5_threshold_location")


# ----------------------------------------------------------------------------
def nice(name):
    return name.replace("days 1-7", "days 1–7").replace("days 7-14", "days 7–14")


def fig6():
    e6 = load("e6_treatments")
    ref = e6["parameter_sets"]["reference"]
    fig = plt.figure(figsize=(7.2, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.45])
    ax = fig.add_subplot(gs[0])
    show = {"Control": (INK2, "-"), "chABC, continuous, high dose": (BLUE, "-"),
            "chABC, days 1-7": (ORANGE, "-"), "chABC, days 7-14": (AQUA, "-")}
    for row in ref:
        if row["name"] in show:
            c, ls = show[row["name"]]
            t = np.array(row["trajectory"]["t_h"]) / 24
            ax.plot(t, row["trajectory"]["core_S"], color=c, ls=ls, label=nice(row["name"]))
    for d0, d1, lab in [(1, 7, "d1–7"), (7, 14, "d7–14")]:
        ax.axvspan(d0, d1, color=GRID if d0 == 1 else "#f2f1ed", alpha=0.6, lw=0)
        ax.text((d0 + d1) / 2, 4.75, lab, ha="center", fontsize=6.8, color=INK2)
    ax.set_xlabel("Days after injury"); ax.set_ylabel("Lesion-core CSPG (fold change)")
    ax.set_xlim(0, 28.5); ax.set_ylim(0, 5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.8)
    panel_label(ax, "A")

    ax = fig.add_subplot(gs[1])
    order = list(reversed(range(len(ref))))
    y = np.arange(len(ref))
    tot = [ref[i]["crossing"]["mean"] for i in order]
    thr = [ref[i]["through_core"]["mean"] for i in order]
    ax.barh(y + 0.19, tot, height=0.36, color=BLUE, label="all crossing")
    ax.barh(y - 0.19, thr, height=0.36, color=ORANGE, label="through lesion core")
    for yi, i in zip(y, order):
        ax.text(101, yi, f"{ref[i]['core_S_d28']:.2f}×", va="center", fontsize=7, color=INK2)
    ax.text(101, len(ref) - 0.35, "core\nCSPG", fontsize=6.5, color=INK2, va="bottom")
    u = e6["uninjured"]["crossing"]["mean"]
    ax.axvline(u, color=MUTED, lw=0.8, ls="--")
    ax.text(u - 1, -0.75, "uninjured", fontsize=6.5, color=INK2, ha="right", va="center")
    ax.set_yticks(y); ax.set_yticklabels([nice(ref[i]["name"]) for i in order], fontsize=7)
    ax.set_xlim(0, 100); ax.set_xlabel("Axons crossing at day 28 (%)")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-1.1, len(ref) - 0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2, fontsize=6.8)
    panel_label(ax, "B")
    fig.tight_layout()
    save(fig, "fig6_treatments")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        try:
            f()
        except FileNotFoundError as e:
            print(f"skipped {f.__name__}: {e}")
