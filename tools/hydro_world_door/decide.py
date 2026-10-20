"""Print the default-on hydro door's decision for experiment files (CPU only).

  python -m tools.hydro_world_door.decide EXPERIMENT.toml [...] [--png coverage.png]

Each decision is exactly what ``gpuwm go`` / ``gpuwm ensemble`` record under
``hydro_activation.door``; with --png it also draws the provider coverage and
each parent-grid footprint (an analysis chart, not a weather field).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from gpuwm.hydro import door


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", nargs="+", type=Path)
    parser.add_argument("--png", type=Path)
    args = parser.parse_args(argv)
    rows = []
    for path in args.experiments:
        started = time.perf_counter()
        geometry = door.domain_geometry(path)
        geometry_s = time.perf_counter() - started
        decision = door.decide(settings={}, requested=None, config_path=path, geometry=geometry)
        rows.append({"experiment": str(path), "geometry_seconds": geometry_s,
                     "decision_seconds": time.perf_counter() - started,
                     "parallelism": "1 thread, startup only: projection of a 33x33 parent lattice "
                                    "and one subprocess probe per covering provider",
                     "decision": door.public(decision), "_points": geometry["points"]})
    print(json.dumps([{k: v for k, v in row.items() if not k.startswith("_")} for row in rows],
                     indent=2))
    if args.png:
        _draw(rows, args.png)


def _draw(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    table = door.load_providers()
    fig, ax = plt.subplots(figsize=(12, 7.2), dpi=130)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 80)
    ax.set_facecolor("#f4f6f8")
    ax.grid(color="white", linewidth=0.8)
    for row in table["providers"]:
        if row["coverage"] == "global":
            continue
        for west, south, east, north in row["coverage"]:
            ax.add_patch(Rectangle((west, south), east - west, north - south, fill=False,
                                   edgecolor="#3366aa", linewidth=1.6, linestyle="--"))
            ax.text(west, north + 1.5, f"{row['id']} (priority {row['priority']})",
                    color="#3366aa", fontsize=8)
    ax.text(-178, 75, "hydro-world-global covers every domain (priority 20)", color="#3366aa",
            fontsize=8)
    colours = {"fabric_covered": "#2a9d4b", "no_covering_fabric": "#d07a00",
               "parent_lsm_unsupported": "#b8323a", "no_domain_geometry": "#666666"}
    legend, seen = [], {}
    for number, row in enumerate(rows, 1):
        lat = [p[0] for p in row["_points"]]
        lon = [p[1] for p in row["_points"]]
        decision = row["decision"]
        colour = colours.get(decision["reason"], "black")
        ax.scatter(lon, lat, s=2, color=colour, alpha=0.6)
        key = (round(max(lat), 1), round(max(lon), 1))
        offset = seen.get(key, 0)
        seen[key] = offset + 1
        ax.annotate(str(number), (max(lon), max(lat)), xytext=(4 + 10 * offset, 2),
                    textcoords="offset points", fontsize=9, color=colour, weight="bold")
        gap = decision["gaps"][0] if decision["gaps"] else {"owner": "", "gap": decision.get("provider", "")}
        legend.append((colour, f"{number}  {Path(row['experiment']).stem}: hydro "
                       f"{'ON' if decision['enabled'] else 'OFF'} ({decision['reason']}); "
                       f"first gap [{gap['owner']}]: {gap['gap'][:110]}"))
    for index, (colour, text) in enumerate(legend):
        fig.text(0.01, 0.135 - 0.03 * index, text, fontsize=7.5, color=colour)
    fig.subplots_adjust(bottom=0.30)
    ax.set_xlabel("longitude (degrees east)")
    ax.set_ylabel("latitude (degrees north)")
    ax.set_title("Default-on hydro door: provider coverage (dashed) and parent-grid footprints "
                 "(dots, 33x33 lattice)", fontsize=10)
    fig.savefig(out)


if __name__ == "__main__":
    main()
