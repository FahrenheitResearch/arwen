"""recheck_cells.py ROWS_DIR... : the cells to re-run under the blow-up criterion.  A cell (row, wind) qualifies when a
run at NCAR's 6.0 s/km, or the first rung tried there if 6.0 was never reached, stopped NEAR THE BOUND (finite w at most
10 x the map's held bound at the stop).  Each qualifying cell is re-run at 6.0 s/km and at 3.0 s/km (the default
clock's halved step on both NCAR grids) for the full three hours.  Prints one line per run: dx crest ridge_slope wind
sound_steps settings step_s."""
import json, math, sys
from pathlib import Path
seen = set()
for d in sys.argv[1:]:
    for p in sorted(Path(d).glob("*.json")):
        if p.name.startswith("provenance"):
            continue
        r = json.loads(p.read_text())
        for wind in r["winds_m_s"]:
            runs = [x for x in r["runs"] if x["wind"] == wind]
            if not runs:
                continue
            at6 = [x for x in runs if abs(x["per_km"] - 6.0) < 1e-9]
            probe = at6[0] if at6 else runs[0]
            if probe["held"]:
                continue
            if not math.isfinite(probe["peak_w"]) or probe["peak_w"] > 10.0 * probe["bound"]:
                continue
            for per_km in (6.0, 3.0):
                key = (r["dx_m"], r["crest_m"], r["ridge_slope"], wind, r["sound_steps"], r["settings"], per_km)
                if key in seen:
                    continue
                seen.add(key)
                print(f"{r['dx_m']:g} {r['crest_m']:g} {r['ridge_slope']:g} {wind:g} {r['sound_steps']} {r['settings']} {per_km * r['dx_m'] / 1000.0:g}")
