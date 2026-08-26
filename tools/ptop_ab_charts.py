#!/usr/bin/env python3
"""Analysis charts for a p_top A/B: skill, structure, and cost.

Reads the score files tools/obs_battery_score.py wrote and the probes
tools/ptop_ab_probes.py wrote, and draws the five evidence charts as
PNGs.  Analysis charts only -- weather-field imagery stays with the
Rust renderer; nothing here plots a field on a map.

Usage::

    python tools/ptop_ab_charts.py --scores DIR --probes DIR --out DIR \
        [--control-label "control (100 hPa)"] \
        [--treatment-label "treatment (50 hPa)"] \
        [--cost control=354.9,3666 --cost treatment=309.1,3628]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Two-arm categorical pair (Tol bright): CVD-separable, fixed order.
CONTROL_COLOR = "#4477AA"
TREATMENT_COLOR = "#EE7733"
GRID = dict(color="#d5d5d5", linewidth=0.6)


def _style(ax):
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _lead_map(score: dict) -> dict[int, float]:
    by_lead = score["reflectivity"]["primary_by_lead"]
    if isinstance(by_lead, dict):
        return {int(k): v for k, v in by_lead.items() if v is not None}
    return {int(row["lead_hour"]): row["fss"] for row in by_lead
            if row.get("fss") is not None}


def chart_fss(scores: dict[str, dict], labels: dict[str, str], out: Path):
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    for arm, color in (("control", CONTROL_COLOR),
                       ("treatment", TREATMENT_COLOR)):
        leads = _lead_map(scores[arm])
        hours = sorted(leads)
        ax.plot(hours, [leads[h] for h in hours], color=color, lw=2,
                marker="o", ms=5, label=labels[arm])
        scalar = scores[arm]["reflectivity"]["primary_scalar"]
        ax.annotate(f"mean {scalar:.4f}", xy=(hours[-1], leads[hours[-1]]),
                    xytext=(6, 14 if arm == "treatment" else -16),
                    textcoords="offset points", color="#444444", fontsize=9)
    refl = scores["control"]["reflectivity"]
    ax.set_xlabel("forecast lead (h)")
    ax.set_ylabel(f"FSS, {refl['primary_threshold_dbz']:g} dBZ, "
                  f"{refl['primary_box_length_m'] / 1000:g} km box")
    ax.set_title("Composite-reflectivity FSS vs MRMS, by lead")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "fss_by_lead.png")
    plt.close(fig)


def _frame(probes: dict, index: int) -> dict:
    return probes["frames"][index]


def chart_wmax_profile(probes: dict[str, dict], labels: dict[str, str],
                       index: int, out: Path):
    fig, ax = plt.subplots(figsize=(5.4, 6.2), dpi=150)
    for arm, color in (("control", CONTROL_COLOR),
                       ("treatment", TREATMENT_COLOR)):
        frame = _frame(probes[arm], index)
        prof = frame["w_max_profile"]
        z_km = [z / 1000.0 for z in prof["z_mean_m_msl"]]
        ax.plot(prof["w_max_ms"], z_km, color=color, lw=2,
                label=labels[arm])
        base_km = (frame["sponge_base_m_agl"]["mean"]
                   + (frame["column_top_m_msl"]["mean"]
                      - frame["column_depth_m"]["mean"])) / 1000.0
        ax.axhline(base_km, color=color, lw=1.2, ls="--", alpha=0.8)
        ax.annotate(f"sponge base {labels[arm].split()[0]}",
                    xy=(ax.get_xlim()[1] * 0.4, base_km),
                    xytext=(0, 4), textcoords="offset points",
                    color=color, fontsize=8)
    valid = _frame(probes["control"], index)["frame"].replace(
        "wrfout_d01_", "")
    ax.set_xlabel("domain-max updraft (m/s)")
    ax.set_ylabel("height (km MSL)")
    ax.set_title(f"Max-updraft profile, {valid}")
    ax.legend(frameon=False, loc="upper right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "wmax_profile.png")
    plt.close(fig)


def chart_refl_tail(probes: dict[str, dict], labels: dict[str, str],
                    index: int, out: Path):
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    width = 2.0
    for k, (arm, color) in enumerate((("control", CONTROL_COLOR),
                                      ("treatment", TREATMENT_COLOR))):
        hist = _frame(probes[arm], index)["comp_refl_histogram"]
        edges, counts = hist["edges_dbz"], hist["counts"]
        centers, values = [], []
        for j, count in enumerate(counts):
            if edges[j] >= 35.0:
                centers.append((edges[j] + edges[j + 1]) / 2.0)
                values.append(count)
        ax.bar([c + (k - 0.5) * width for c in centers], values, width=width,
               color=color, label=labels[arm], edgecolor="white",
               linewidth=0.5)
    valid = _frame(probes["control"], index)["frame"].replace(
        "wrfout_d01_", "")
    ax.set_xlabel("composite reflectivity (dBZ)")
    ax.set_ylabel("grid cells")
    ax.set_title(f"High-reflectivity core cells, {valid}")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "refl_tail.png")
    plt.close(fig)


def chart_tops(probes: dict[str, dict], labels: dict[str, str], out: Path):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), dpi=150, sharex=True)
    for ax, key, title in ((axes[0], "cloud_top_m_agl",
                            "Cloud-top height, 90th percentile"),
                           (axes[1], "echo_top_m_agl",
                            "18 dBZ echo-top height, 90th percentile")):
        for arm, color in (("control", CONTROL_COLOR),
                           ("treatment", TREATMENT_COLOR)):
            hours, tops = [], []
            for lead, frame in enumerate(probes[arm]["frames"]):
                value = frame.get(key, {}).get("p90")
                if value is not None:
                    hours.append(lead)
                    tops.append(value / 1000.0)
            ax.plot(hours, tops, color=color, lw=2, marker="o", ms=4,
                    label=labels[arm])
        ax.set_ylabel("km AGL")
        ax.set_title(title, fontsize=10)
        _style(ax)
    axes[0].legend(frameon=False)
    axes[1].set_xlabel("forecast lead (h)")
    fig.tight_layout()
    fig.savefig(out / "tops_by_lead.png")
    plt.close(fig)


def chart_cost(costs: dict[str, tuple[float, float]],
               labels: dict[str, str], out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), dpi=150)
    arms = ("control", "treatment")
    colors = (CONTROL_COLOR, TREATMENT_COLOR)
    for ax, index, ylabel, title in (
            (axes[0], 0, "seconds", "Forecast wall (4320 steps)"),
            (axes[1], 1, "MiB", "Peak VRAM (sampled)")):
        values = [costs[arm][index] for arm in arms]
        bars = ax.bar([labels[a] for a in arms], values, color=colors,
                      width=0.55)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:g}", xy=(bar.get_x() + bar.get_width() / 2,
                                          value),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9, color="#444444")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        _style(ax)
    fig.tight_layout()
    fig.savefig(out / "cost.png")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=6,
                        help="frame used for the profile/histogram charts")
    parser.add_argument("--control-label", default="control (100 hPa)")
    parser.add_argument("--treatment-label", default="treatment (50 hPa)")
    parser.add_argument("--cost", action="append", default=[],
                        metavar="ARM=WALL_S,VRAM_MIB")
    arguments = parser.parse_args()

    labels = {"control": arguments.control_label,
              "treatment": arguments.treatment_label}
    scores = {arm: json.loads(
        (arguments.scores / f"score_{arm}.json").read_text())
        for arm in ("control", "treatment")}
    probes = {arm: json.loads(
        (arguments.probes / f"probes_{arm}.json").read_text())
        for arm in ("control", "treatment")}
    arguments.out.mkdir(parents=True, exist_ok=True)

    chart_fss(scores, labels, arguments.out)
    chart_wmax_profile(probes, labels, arguments.frame_index, arguments.out)
    chart_refl_tail(probes, labels, arguments.frame_index, arguments.out)
    chart_tops(probes, labels, arguments.out)
    costs = {}
    for spec in arguments.cost:
        arm, _, pair = spec.partition("=")
        wall, _, vram = pair.partition(",")
        costs[arm] = (float(wall), float(vram))
    if len(costs) == 2:
        chart_cost(costs, labels, arguments.out)
    print(f"charts written to {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
