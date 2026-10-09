"""recheck_summary.py RECHECK_DIR : the blow-up-criterion re-runs of cells the map's bound stopped near it.
Per cell and step: ran three hours without w passing 10 x the bound (finite) or ran away, the peak w over the bound,
whether w was still growing (last hour's peak more than 10 percent over the hour before), peak WRF vertical Courant."""
import json, math, sys
from collections import defaultdict
from pathlib import Path
recs = []
for p in sorted(Path(sys.argv[1]).glob("*.jsonl")):
    for line in p.read_text().splitlines():
        if line.strip(): recs.append(json.loads(line))
def growing(r):
    h = r["hourly_peak_w"]
    return len(h) >= 3 and h[-1] > 1.10 * h[-2]
tab = defaultdict(lambda: defaultdict(int))
for r in recs:
    k = (r["dx_m"], r["settings"], r["per_km"])
    if not r["held"]:
        tab[k]["ran away"] += 1
    elif growing(r):
        tab[k]["finite, still growing"] += 1
    else:
        tab[k]["finite, steady"] += 1
print("| spacing | settings | step s/km | cells | ran away (w non-finite or > 10x bound) | finite, still growing in hour 3 | finite, steady |")
print("|---|---|---|---|---|---|---|")
for k in sorted(tab):
    t = tab[k]; n = sum(t.values())
    print(f"| {k[0]/1000:g} km | {k[1]} | {k[2]:g} | {n} | {t['ran away']} | {t['finite, still growing']} | {t['finite, steady']} |")
print()
fin = [r for r in recs if r["held"]]
if fin:
    ratios = [r["peak_w"] / r["bound"] for r in fin]
    print(f"Finite runs: {len(fin)}; peak w / map bound {min(ratios):.2f} to {max(ratios):.2f}; peak WRF vertical Courant {min(r['peak_wrf_vertical_courant'] for r in fin):.2f} to {max(r['peak_wrf_vertical_courant'] for r in fin):.2f}")
ra = [r for r in recs if not r["held"]]
if ra:
    print(f"Ran away: {len(ra)}; stopped at {min(r['stopped_at_s'] for r in ra):.0f} to {max(r['stopped_at_s'] for r in ra):.0f} s")
print()
print("Per cell (dx crest ridge_slope wind settings): result at 6.0 s/km | at 3.0 s/km")
cells = defaultdict(dict)
for r in recs:
    cells[(r["dx_m"], r["crest_m"], r["ridge_slope"], r["wind"], r["settings"])][r["per_km"]] = r
def word(r):
    if r is None: return "-"
    if not r["held"]: return f"RAN AWAY at {r['stopped_at_s']:.0f} s"
    return f"finite, peak {r['peak_w']/r['bound']:.2f}x bound{' growing' if growing(r) else ''}, vc {r['peak_wrf_vertical_courant']:.2f}"
for k in sorted(cells):
    c = cells[k]
    print(f"  {k[0]/1000:g} km {k[1]:g} m r{k[2]:g} {k[3]:g} m/s {k[4]}: {word(c.get(6.0))} | {word(c.get(3.0))}")
