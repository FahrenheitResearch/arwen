"""rederive6.py RESULTS_DIR LANE_ROOT : step 6, the candidate map re-derived under the 100 m/s peak rule (lead decision
3) and LA d01's 80 m/s column (read at the 1.5 input-wind floor, lead decision 2).

* fixed rows (gpuwm/terrain_clock_map_local_face.json): the two entries that rested on a run held at peak |w| of
  100 m/s or more (every run behind the map re-read: 3 h sweep, 12 h long probes, the step-5 70 m/s cells) take the
  rung their walk now reaches (results/cells/fx6-*.json, 12 h, the probe's own 100 m/s stop); 80 m/s joins the winds,
  null on every row (not run, not read) except the ncar cells LA d01 reads there (results/cells/la80-*.json), each
  walked down from its 70 m/s entry under the rule, 12 h;
* adaptive rows (gpuwm/terrain_clock_adaptive_local_face.json): every row whose walk counted such a run, re-walked
  (results/adaptive/ad6-*.json: the old runs reused under the rule, the rungs the old count never reached run now);
* docs/terrain-clock/step6/: every run behind both, compacted."""
import glob
import json
import math
import sys
from pathlib import Path

res, lane = Path(sys.argv[1]), Path(sys.argv[2])
MAP = lane / "gpuwm/terrain_clock_map_local_face.json"
ADA = lane / "gpuwm/terrain_clock_adaptive_local_face.json"
OUT = lane / "docs/terrain-clock/step6"
OUT.mkdir(parents=True, exist_ok=True)
PEAK = 100.0
SRC = "CLOCK-CHECK-NCAR-2026-10-06/fix/step6 (box W1, lane tip 0a52e0a7f + the step-6 probe, 2026-10-07)"


def dump_map(doc):
    out = ["{"]
    keys = list(doc)
    for n, k in enumerate(keys):
        v = doc[k]
        end = "," if n < len(keys) - 1 else ""
        if k == "rows":
            out.append(' "rows": [')
            out += ["  " + json.dumps(r) + ("," if m < len(v) - 1 else "") for m, r in enumerate(v)]
            out.append(" ]" + end)
        elif isinstance(v, dict) and k in ("long_probes", "wind_70_probes", "wind_80_probes", "peak_rule"):
            out.append(" " + json.dumps(k) + ": {")
            sub = list(v)
            for m, kk in enumerate(sub):
                vv = v[kk]
                e2 = "," if m < len(sub) - 1 else ""
                if isinstance(vv, list) and vv and isinstance(vv[0], dict):
                    out.append("  " + json.dumps(kk) + ": [")
                    out += ["   " + json.dumps(x) + ("," if q < len(vv) - 1 else "") for q, x in enumerate(vv)]
                    out.append("  ]" + e2)
                else:
                    out.append("  " + json.dumps(kk) + ": " + json.dumps(vv) + e2)
            out.append(" }" + end)
        else:
            out.append(" " + json.dumps(k) + ": " + json.dumps(v) + end)
    out.append("}")
    return "\n".join(out) + "\n"


def dump_rows(doc):
    head = {k: v for k, v in doc.items() if k != "rows"}
    out = ["{"] + [f" {json.dumps(k)}: {json.dumps(v)}," for k, v in head.items()]
    out.append(' "rows": [')
    out.append(",\n".join("  " + json.dumps(r) for r in doc["rows"]))
    out += [" ]", "}"]
    return "\n".join(out) + "\n"


def num(v):
    if isinstance(v, float):
        return round(v, 3) if math.isfinite(v) else None
    return v


def slim(run):
    keep = ("per_km", "held", "peak_w", "stopped_at_s", "peak_wrf_vertical_courant", "hourly_peak_w", "nx", "wall_s")
    return {k: num(run.get(k)) for k in keep if k in run}


raw = MAP.read_text(encoding="utf-8")
doc = json.loads(raw)
assert dump_map(doc) == raw, "map layout differs from the file's own"
assert doc["winds_m_s"] == [20.0, 30.0, 40.0, 50.0, 60.0, 70.0], doc["winds_m_s"]
cells = {Path(p).stem: json.loads(Path(p).read_text()) for p in glob.glob(str(res / "cells/*.json"))}
fx = {k: v for k, v in cells.items() if k.startswith("fx6-")}
la80 = {k: v for k, v in cells.items() if k.startswith("la80-")}
jobs80 = [json.loads(line) for line in (res / "jobs-80.jsonl").read_text().splitlines() if line.strip()]
assert len(fx) == 2 and len(la80) == len(jobs80) == 192, (len(fx), len(la80), len(jobs80))


def row_of(r):
    hit = [x for x in doc["rows"] if x["settings"] == r["settings"] and x["dx_m"] == r["dx"]
           and x["crest_m"] == r["crest"] and x["ridge_slope"] == r["ridge_slope"]
           and x["sound_steps"] == r["sound_steps"]]
    assert len(hit) == 1, r["key"]
    return hit[0]


def rungs_run(r):
    return [[x["per_km"], x["held"], num(x["peak_w"]), x["stopped_at_s"]] for x in r["runs"]]


# 1. the two entries the peak rule moves.
changed = []
for key, r in sorted(fx.items()):
    row = row_of(r)
    wi = doc["winds_m_s"].index(float(r["wind"]))
    was = row["stable_s_per_km"][wi]
    assert r["entry_12h"] is not None and r["entry_12h"] < was, (key, was, r["entry_12h"])
    assert r["rungs"][0] < was
    row["stable_s_per_km"][wi] = r["entry_12h"]
    changed.append({"settings": r["settings"], "dx_m": r["dx"], "crest_m": r["crest"], "ridge_slope": r["ridge_slope"],
                    "sound_steps": r["sound_steps"], "wind_m_s": float(r["wind"]), "was_s_per_km": was,
                    "was_held_at_peak_w_m_s": {"generated": 117.6, "ncar": 119.3}[r["settings"]],
                    "entry_s_per_km": r["entry_12h"], "top_s_per_km": row["top_s_per_km"][wi],
                    "rungs_run": rungs_run(r)})
# 2. the 80 m/s column.
doc["winds_m_s"] = doc["winds_m_s"] + [80.0]
for row in doc["rows"]:
    row["stable_s_per_km"] = list(row["stable_s_per_km"]) + [None]
    row["top_s_per_km"] = list(row["top_s_per_km"]) + [None]
cells80 = []
for key, r in sorted(la80.items()):
    row = row_of(r)
    e70 = row["stable_s_per_km"][5]
    assert e70 is not None, key
    entry, top = r["entry_12h"], float(r["rungs"][0])
    # The walk starts where the weaker wind held (only the deciding cell's walk started before its own 70 m/s
    # re-walk landed; that landed on 4.0, the rung this walk started from).
    assert top <= e70 + 1e-9, (key, top, e70)
    row["stable_s_per_km"][6] = entry
    row["top_s_per_km"][6] = top
    cells80.append({"settings": r["settings"], "dx_m": r["dx"], "crest_m": r["crest"], "ridge_slope": r["ridge_slope"],
                    "slope": r["slope"], "sound_steps": r["sound_steps"], "wind_m_s": 80.0, "entry_s_per_km": entry,
                    "top_s_per_km": top, "rungs_run": rungs_run(r)})
doc["held_peak_w_m_s"] = PEAK
doc["peak_rule"] = {
    "held_peak_w_m_s": PEAK,
    "why": ("Lead decision 3 after step 5 (2026-10-07): a probe run counts as held only if it stayed finite AND its "
            "peak vertical velocity stayed under 100 m/s; runs above that count as stopped.  Every run behind this map "
            "was re-read under it (the 3,814 three-hour sweep runs, the 1,387 twelve-hour long-probe runs and the 395 "
            "step-5 70 m/s runs): two held entries rested on a run past 100 m/s, and each walk was taken on down, "
            "12 h under the probe's own 100 m/s stop."),
    "source": SRC + "; every run in docs/terrain-clock/step6/terrain_clock_step6_cells-2026-10-07.json",
    "cells_moved": changed}
doc["wind_80_probes"] = {
    "seconds": 43200.0, "cells": len(cells80),
    "why": ("LA Santa Ana 2025-01-07 12Z d01 (2.25 km) is read at the 1.5 x input-wind floor (lead decision 2) at up "
            "to 73.9 m/s, the 80 m/s column, where the candidate had run nothing and the engine read the shipped map's "
            "cells, measured under the retired old bound.  These are every cell its faces read there: the ncar arm's "
            "2 and 3 km rows of crests 3000, 3500 and 4500 m, ridges 0.05 to 0.40, at 4 and 6 substeps, each run 12 h "
            "under the blow-up test with the 100 m/s peak rule on the 12-hour domain, walked down from its 70 m/s "
            "entry."),
    "source": SRC + "; every run in docs/terrain-clock/step6/terrain_clock_step6_cells-2026-10-07.json",
    "cells_run": cells80}
doc["what"] = doc["what"] + (
    "  PEAK RULE (2026-10-07, fix/step6): a run holds only where it stayed finite AND its peak |w| stayed under "
    "held_peak_w_m_s (100 m/s); the two entries that rested on a run past it moved down (peak_rule).  80 M/S "
    "(fix/step6): the ncar arm's 2 and 3 km rows that LA Santa Ana's parent grid reads at 80 m/s under the 1.5 "
    "input-wind floor (192 cells, wind_80_probes) were run 12 hours under that test, each walked down from its "
    "70 m/s entry; every other row carries null at 80 m/s (not run, not read).")
MAP.write_text(dump_map(doc), encoding="utf-8", newline="\n")

# 3. adaptive rows.
araw = ADA.read_text(encoding="utf-8")
ada = json.loads(araw)
assert dump_rows(ada) == araw, "adaptive layout differs from the file's own"
rows6 = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(res / "adaptive/ad6-*.json")))]
assert len(rows6) == 15 and all(r["complete"] for r in rows6), len(rows6)
moved = []
for r in rows6:
    hit = [x for x in ada["rows"] if x["arm"] == r["settings"] and x["dx_m"] == r["dx"]
           and x["crest_m"] == r["crest"] and x["ridge_slope"] == r["ridge_slope"]]
    assert len(hit) == 1, r["key"]
    x = hit[0]
    assert x["adaptive_s_per_km"][:5] == r["was_adaptive_s_per_km"], r["key"]
    was = (list(x["adaptive_s_per_km"][:5]), list(x["adaptive_top_s_per_km"][:5]))
    x["adaptive_s_per_km"][:5] = r["adaptive_s_per_km"]
    x["adaptive_top_s_per_km"][:5] = r["adaptive_top_s_per_km"]
    moved.append({"arm": r["settings"], "dx_m": r["dx"], "crest_m": r["crest"], "ridge_slope": r["ridge_slope"],
                  "was_s_per_km": was[0], "now_s_per_km": r["adaptive_s_per_km"],
                  "was_top_s_per_km": was[1], "now_top_s_per_km": r["adaptive_top_s_per_km"],
                  "new_runs": r["new_runs"]})
ada["held_peak_w_m_s"] = PEAK
ada["step6_peak_rule"] = {"rows": len(moved), "source": SRC + "; every run in docs/terrain-clock/step6/"
                          "terrain_clock_adaptive_rows_step6_evidence-2026-10-07.json", "moved": moved}
ada["what"] = ada["what"] + (
    "  PEAK RULE (2026-10-07, fix/step6): a run holds only where it stayed finite AND its peak |w| stayed under "
    "100 m/s (held_peak_w_m_s); the 15 rows whose walk counted a run past it (46 runs, 101 to 388 m/s) were "
    "re-walked, the old runs reused under the rule and the rungs the old count never reached run 12 h now "
    "(step6_peak_rule).")
ADA.write_text(dump_rows(ada), encoding="utf-8", newline="\n")

# 4. evidence.
ev = {"schema": "gpuwm-terrain-clock-step6-cells-v1", "held_peak_w_m_s": PEAK, "seconds": 43200.0,
      "what": ("Step 6 fixed-step probe cells, 12 h each under the blow-up test with the 100 m/s peak rule: fx6 the "
               "two entries the rule moved, la80 LA d01's 80 m/s cells.  Every rung run, with peak |w|, stop time, "
               "peak WRF vertical Courant and hourly peaks."),
      "cells": [{**{k: v for k, v in r.items() if k != "runs"}, "runs": [slim(x) for x in r["runs"]]}
                for _k, r in sorted(cells.items())]}
(OUT / "terrain_clock_step6_cells-2026-10-07.json").write_text(json.dumps(ev, separators=(",", ":")) + "\n",
                                                                encoding="utf-8", newline="\n")
aev = {"schema": "gpuwm-terrain-clock-adaptive-rows-evidence-v1", "held_peak_w_m_s": PEAK,
       "rows": [{**{k: v for k, v in r.items() if k != "runs"},
                 "runs": [{kk: num(vv) for kk, vv in x.items() if kk != "sound_steps_taken"} for x in r["runs"]]}
                for r in rows6]}
(OUT / "terrain_clock_adaptive_rows_step6_evidence-2026-10-07.json").write_text(
    json.dumps(aev, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
print("fixed moved", [(c["settings"], c["dx_m"], c["crest_m"], c["ridge_slope"], c["wind_m_s"], c["was_s_per_km"],
                       c["entry_s_per_km"]) for c in changed])
print("80 m/s cells", len(cells80), "none held", sum(1 for c in cells80 if c["entry_s_per_km"] is None))
for m in moved:
    print("adaptive", m["arm"], m["dx_m"], m["crest_m"], m["ridge_slope"], m["was_s_per_km"], "->", m["now_s_per_km"],
          "new runs", m["new_runs"])
