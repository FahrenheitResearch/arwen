"""tables.py CANDIDATE MAP : markdown tables of the candidate rows, and the generated arm against the shipped map.
Every number printed comes from the two JSON files named on the command line."""
import json, math, sys
from collections import defaultdict

cand = json.load(open(sys.argv[1], encoding="utf-8"))
shipped = json.load(open(sys.argv[2], encoding="utf-8"))
winds = [20.0, 30.0, 40.0, 50.0, 60.0]


def cell(row, i):
    e = row["stable_s_per_km"][i]
    t = row["top_s_per_km"][i]
    if t is None:
        return "-"
    runs = [r for r in row["per_wind"][i]["runs"] if not r.get("below_floor")]
    stops = [r["stop"] for r in runs if not r["held"]]
    mark = ""
    if "runaway" in stops:
        mark = "r"
    elif "near-bound" in stops:
        mark = "n"
    if e is None:
        return f"none{mark}"
    return f"{e:g}{mark}"


groups = defaultdict(list)
for row in cand["rows"]:
    groups[(row["settings"], row["dx_m"], row["crest_m"], row["sound_steps"])].append(row)

print("## Entries (s/km held three hours; r = a longer rung ran away (w over 10x bound or non-finite at the stop), n = a longer rung stopped near the bound; none = no rung down to the floor held; - = not run)\n")
for key in sorted(groups, key=lambda k: (k[1], k[2], k[3], k[0])):
    settings, dx, crest, sub = key
    print(f"### {dx/1000:g} km, crest {crest:g} m, {sub} substeps, settings {settings}\n")
    print("| ridge slope | grid slope | " + " | ".join(f"{w:g} m/s" for w in winds) + " | peak WRF vert. Courant (held runs) | peak near-ground geometric Courant (held runs) |")
    print("|---|---|" + "---|" * len(winds) + "---|---|")
    for row in sorted(groups[key], key=lambda r: (r["slope"], r["ridge_slope"])):
        held = [r for pw in row["per_wind"] for r in pw["runs"] if r["held"] and not r.get("below_floor")]
        vc = max((r["peak_wrf_vertical_courant"] or 0.0) for r in held) if held else None
        ng = max((r["peak_near_ground_vertical_courant"] or 0.0) for r in held) if held else None
        print(f"| {row['ridge_slope']:g} | {row['slope']:.4f} | " + " | ".join(cell(row, i) for i in range(len(winds)))
              + f" | {'' if vc is None else f'{vc:.2f}'} | {'' if ng is None else f'{ng:.1f}'} |")
    print()

# stop classes
counts = defaultdict(int)
ratios = {"runaway": [], "near-bound": []}
for row in cand["rows"]:
    for pw in row["per_wind"]:
        for r in pw["runs"]:
            if r.get("below_floor"):
                continue
            if r["held"]:
                counts[(row["settings"], "held")] += 1
            else:
                counts[(row["settings"], r["stop"])] += 1
                ratios[r["stop"]].append(math.inf if r["peak_w"] is None else r["peak_w"] / r["bound"])
print("## Runs by outcome\n")
print("| settings | held | stopped: runaway | stopped: near-bound |")
print("|---|---|---|---|")
for s in ("generated", "ncar"):
    print(f"| {s} | {counts[(s,'held')]} | {counts[(s,'runaway')]} | {counts[(s,'near-bound')]} |")
fin = lambda xs: [x for x in xs if math.isfinite(x)]
print()
if ratios["near-bound"]:
    print(f"Near-bound stops: peak w / bound from {min(ratios['near-bound']):.3f} to {max(ratios['near-bound']):.3f} ({len(ratios['near-bound'])} runs).")
if ratios["runaway"]:
    f = fin(ratios["runaway"])
    print(f"Runaway stops: {len(ratios['runaway'])} runs, {len(ratios['runaway'])-len(f)} non-finite, finite peak w / bound from {min(f) if f else float('nan'):.1f} to {max(f) if f else float('nan'):.3g}.")
print()

# generated arm against the shipped map, on rows the map carries
print("## Generated arm against the shipped map (same spacing, crest, ridge slope, substeps), 20-60 m/s\n")
print("Shipped entries are the map's own (half hour, three-hour checks only where its what-text says); new entries held three hours. Only rungs at or under the shipped row's longest step tried are comparable.\n")
print("| spacing | crest | ridge slope | substeps | shipped entries | shipped tried | new entries |")
print("|---|---|---|---|---|---|---|")
idx = {(r["dx_m"], r["crest_m"], round(r["ridge_slope"], 6), r["sound_steps"]): r for r in shipped["rows"]}
for row in sorted(cand["rows"], key=lambda r: (r["dx_m"], r["crest_m"], r["sound_steps"], r["ridge_slope"])):
    if row["settings"] != "generated":
        continue
    m = idx.get((row["dx_m"], row["crest_m"], round(row["ridge_slope"], 6), row["sound_steps"]))
    if m is None:
        continue
    top = m.get("top_s_per_km") or [shipped["ladder_s_per_km"][0]] * len(shipped["winds_m_s"])
    fmt = lambda xs: ", ".join("null" if x is None else f"{x:g}" for x in xs)
    print(f"| {row['dx_m']/1000:g} km | {row['crest_m']:g} | {row['ridge_slope']:g} | {row['sound_steps']} | {fmt(m['stable_s_per_km'][:5])} | {fmt(top[:5])} | {fmt(row['stable_s_per_km'])} |")
