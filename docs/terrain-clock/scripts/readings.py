"""readings.py CANDIDATE : the engine's own read_map (gpuwm/terrain_clock.py, lane tree) over the candidate rows alone,
one settings arm at a time, at the NCAR v4.4 CONUS readings from FINDINGS.md / REVIEW.md (domain-wide and face-local).
Prints the reading and what it does to NCAR's step.  Run with PYTHONPATH=<lane tree>, CPU only."""
import json, math, sys
from gpuwm.terrain_clock import MapRow, StableStepMap, read_map

cand = json.load(open(sys.argv[1], encoding="utf-8"))
WINDS = (20.0, 30.0, 40.0, 50.0, 60.0)

CASES = [
    # label, dx, crest, slope, wind, NCAR step s
    ("12 km, domain-wide reading (clock today)", 12000.0, 3615.0, 0.0915, 48.4, 72.0),
    ("12 km, face-local crest, start/boundary face wind", 12000.0, 3398.0, 0.0915, 11.0, 72.0),
    ("12 km, face-local crest, f012 across-face wind", 12000.0, 3398.0, 0.0915, 25.0, 72.0),
    ("12 km, face-local crest, f012 wind x 2.3 margin", 12000.0, 3398.0, 0.0915, 57.5, 72.0),
    ("2.5 km, domain-wide reading (clock today)", 2500.0, 3971.0, 0.337, 52.4, 15.0),
    ("2.5 km, face-local crest, face wind 15 m/s", 2500.0, 2590.0, 0.337, 15.0, 15.0),
    ("2.5 km, face-local crest, face wind 15 x 2.3 = 34.5 m/s", 2500.0, 2590.0, 0.337, 34.5, 15.0),
]

for settings in ("generated", "ncar"):
    for sub in (4, 6):
        rows = []
        for r in cand["rows"]:
            if r["settings"] != settings or r["sound_steps"] != sub:
                continue
            stable = tuple(r["stable_s_per_km"])
            tried = tuple(r["top_s_per_km"])
            # an untried wind (past one that held none) carries no entry and no range
            tried = tuple(t if t is not None else 0.0 for t in tried)
            rows.append(MapRow(r["dx_m"], r["crest_m"], r["slope"], sub, stable, tried))
        if not rows:
            continue
        table = StableStepMap(WINDS, (6.5, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0), tuple(rows), 10800.0)
        print(f"## settings {settings}, {sub} substeps ({len(rows)} rows)")
        for label, dx, crest, slope, wind, step in CASES:
            try:
                rd = read_map(dx, crest, slope, wind, sub, table=table)
            except Exception as exc:  # a reading the candidate rows cannot give
                print(f"  {label}: no reading ({exc})")
                continue
            per_km = step / (dx / 1000.0)
            limit = rd.stopped_per_km if rd.stopped_per_km is not None else rd.per_km
            if rd.per_km is None:
                verdict = "no rung held at this wind (entry null)"
            elif rd.stopped_per_km is None:
                verdict = f"every cell read held every rung tried (to {rd.top_per_km:g} s/km): NCAR's {per_km:g} s/km admitted" if per_km <= rd.top_per_km + 1e-9 else f"held to {rd.top_per_km:g} s/km, NCAR's {per_km:g} s/km past what was measured"
            else:
                held = min(rd.per_km, rd.stopped_per_km)
                div = math.ceil(per_km / held - 1e-9)
                verdict = (f"a longer step stopped; shortest held entry under a stop {rd.stopped_per_km:g} s/km "
                           f"(crest {rd.stopped_crest_m:g}); NCAR's {per_km:g} s/km "
                           + ("admitted" if per_km <= held + 1e-9 else f"divided by {div} -> {step/div:g} s"))
            print(f"  {label}: rows dx {rd.dx_rows} crest {rd.crest_row:g} slope {rd.slope_row} wind {rd.wind_row:g}"
                  f" beyond {list(rd.beyond)} | held {rd.per_km} stopped {rd.stopped_per_km} | {verdict}")
        print()
