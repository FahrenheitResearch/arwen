"""Evidence charts for the stale-guard remediation lane (ledger #348).

ANALYSIS charts, not weather fields -- matplotlib is allowed here under
the render law; the weather-field PNGs in the same gallery come from
rw_wrfbatch.

Conventions held across all four panels:
  blue   = the behaviour this lane SHIPS (or the new measurement)
  orange = the retired / alternative behaviour it is compared against
  gray   = a reference bound, never a series
Status colors (green/red) appear only on the verdict panel, always with
a word beside them so identity is never colour-alone.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "evidence" / "2026-08-25-stale-guards-engine"
OUT.mkdir(parents=True, exist_ok=True)

SHIP = "#1f6fb4"      # current / shipped / new measurement
ALT = "#e07b16"       # retired / alternative
REF = "#6b7280"       # reference bound
INK = "#1f2328"
INK2 = "#57606a"
GRID = "#e6e8eb"
GOOD = "#1a7f37"
BAD = "#bc4c00"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8,
})


def frame(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- 1. tropical
hours = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
cfl30 = [0.129, 0.727, 0.879, 0.910, 0.950, 0.966, 0.976,
         0.957, 0.939, 0.929, 0.949, 0.946, 0.941]
cfl60 = [0.258, 1.614, 1.827, 1.890, 1.956, 1.973, 1.988,
         1.957, 1.997, 1.965, 1.931, 1.917, 1.842]

fig, ax = plt.subplots(figsize=(8, 4.6))
frame(ax)
ax.axhline(1.0, color=REF, linewidth=2, linestyle=(0, (5, 4)), zorder=1)
ax.text(0.06, 1.03, "vertical Courant limit", color=REF, fontsize=9, va="bottom")
ax.plot(hours, cfl60, color=ALT, linewidth=2, marker="o", markersize=5,
        markeredgecolor="white", markeredgewidth=1.2, zorder=3)
ax.plot(hours, cfl30, color=SHIP, linewidth=2, marker="o", markersize=5,
        markeredgecolor="white", markeredgewidth=1.2, zorder=3)
ax.annotate("60 s clock (un-halved)\npeak 2.00", (6.0, 1.842),
            xytext=(-8, 14), textcoords="offset points",
            color=ALT, fontsize=9, ha="right", fontweight="bold")
ax.annotate("30 s clock (the guard)\npeak 0.98", (6.0, 0.941),
            xytext=(-8, -30), textcoords="offset points",
            color=SHIP, fontsize=9, ha="right", fontweight="bold")
ax.set_xlabel("forecast hour")
ax.set_ylabel("vertical CFL  (co-located |w|/dz, v1.1 monitor)")
ax.set_title("The tropical clock guard, re-measured on the corrected instrument",
             fontsize=12, pad=12, loc="left")
ax.set_ylim(0, 2.25)
ax.set_xlim(-0.15, 6.35)
fig.tight_layout()
fig.savefig(OUT / "chart-tropical-clock-cfl.png", dpi=160)
plt.close(fig)

# ------------------------------------------------------- 2. admitted child size
arms = ["rte-rrtmgp parent\n(386x308)", "legacy-RRTMG parent\n(290x232)"]
retired = [282, 324]
affine = [342, 258]

fig, ax = plt.subplots(figsize=(8, 4.6))
frame(ax)
x = [0, 1]
w = 0.34
b1 = ax.bar([i - w / 2 for i in x], retired, w, color=ALT,
            edgecolor="white", linewidth=2, zorder=3)
b2 = ax.bar([i + w / 2 for i in x], affine, w, color=SHIP,
            edgecolor="white", linewidth=2, zorder=3)
for bars, vals in ((b1, retired), (b2, affine)):
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 6, f"{v}",
                ha="center", color=INK, fontsize=10, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, v - 24, "ran whole",
                ha="center", color="white", fontsize=8.5)
ax.set_xticks(x)
ax.set_xticklabels(arms)
ax.set_ylabel("largest child admitted  (cells per side)")
ax.set_title("Standalone-child fit: two retired constants vs the live envelope",
             fontsize=12, pad=12, loc="left")
ax.legend([b1, b2], ["retired: flat reserve + 1.75x floor",
                     "live affine envelope (ships)"],
          frameon=False, loc="upper center", ncol=2,
          bbox_to_anchor=(0.5, -0.16), labelcolor=INK2)
ax.set_ylim(0, 400)
fig.tight_layout()
fig.savefig(OUT / "chart-downscale-child-fit.png", dpi=160)
plt.close(fig)

# ------------------------------------------------------------- 3. pool slack
pre_cells = [60 * 48, 110 * 88]
pre_slack = [0.269, 0.468]
post_cells = [60 * 48, 110 * 88, 290 * 232]
post_slack = [0.213, 0.150, 0.019]

fig, ax = plt.subplots(figsize=(8, 4.6))
frame(ax)
ax.axhline(0.20, color=REF, linewidth=2, linestyle=(0, (5, 4)), zorder=1)
ax.plot(pre_cells, pre_slack, color=ALT, linewidth=2, marker="o",
        markersize=8, markeredgecolor="white", markeredgewidth=1.5, zorder=3)
ax.plot(post_cells, post_slack, color=SHIP, linewidth=2, marker="o",
        markersize=8, markeredgecolor="white", markeredgewidth=1.5, zorder=3)
ax.annotate("pre-#310 calibration (6 h)", (110 * 88, 0.468),
            xytext=(14, 0), textcoords="offset points", color=ALT,
            fontsize=9.5, fontweight="bold", va="center")
ax.text(0.035, 0.14, "post-#310, this lane (2 h)", transform=ax.transAxes,
        color=SHIP, fontsize=9.5, fontweight="bold", va="center")
ax.text(0.99, 0.99, "POOL_SLACK_FRACTION = 0.20  (shipped, unchanged)",
        transform=ax.transAxes, color=REF, fontsize=9.5, va="top",
        ha="right")
ax.set_xscale("log")
ax.set_xticks(pre_cells + [290 * 232])
ax.set_xticklabels(["60x48\n2,880", "110x88\n9,680", "290x232\n67,280"])
ax.minorticks_off()
ax.set_xlabel("root domain, horizontal grid cells (log scale)")
ax.set_ylabel("measured pool slack\n(pool peak - estimate) / estimate")
ax.set_title("Legacy-RRTMG pool retention: measured, not re-fitted",
             fontsize=12, pad=12, loc="left")
ax.set_ylim(0, 0.55)
ax.set_xlim(2200, 90000)
fig.tight_layout()
fig.savefig(OUT / "chart-pool-slack.png", dpi=160)
plt.close(fig)

# --------------------------------------------------- 4. finding 3 gate verdict
labels = ["RETIRED gate\nrtol 2e-5, atol 0.0",
          "LIVE gate\n1.73 m storage atol"]
frac = [4.64e-5 / 2.0e-5, 0.634 / 1.7320508]
colors = [BAD, GOOD]
verdicts = ["would REFUSE the pair", "ACCEPTS the pair"]
raw = ["worst 4.64e-5 against a 2e-5 bound",
       "worst 0.634 m against a 1.73 m bound"]

fig, ax = plt.subplots(figsize=(8, 3.6))
frame(ax)
bars = ax.barh(labels, frac, height=0.42, color=colors,
               edgecolor="white", linewidth=2, zorder=3)
ax.axvline(1.0, color=REF, linewidth=2, linestyle=(0, (5, 4)), zorder=1)
ax.text(1.0, 0.985, "  each gate's own bound", transform=ax.get_xaxis_transform(),
        color=REF, fontsize=9.5, va="top")
for bar, f, v, r in zip(bars, frac, verdicts, raw):
    mid = bar.get_y() + bar.get_height() / 2
    ax.text(f + 0.06, mid + 0.10, v, va="center", color=INK,
            fontsize=10.5, fontweight="bold")
    ax.text(f + 0.06, mid - 0.11, r, va="center", color=INK2, fontsize=9)
ax.set_xlabel("measured disagreement, as a multiple of that gate's bound")
ax.set_title("One generated 654,432-cell pair, judged by both contracts",
             fontsize=12, pad=12, loc="left")
ax.set_xlim(0, 4.3)
ax.set_ylim(1.7, -0.7)
fig.tight_layout()
fig.savefig(OUT / "chart-dual-edge-gate.png", dpi=160)
plt.close(fig)

print(json.dumps(sorted(p.name for p in OUT.glob("chart-*.png")), indent=2))
