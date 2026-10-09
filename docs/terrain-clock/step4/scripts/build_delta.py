"""build_delta.py ROWS_DIR ROWS2030_DIR SHIPPED_MAP OUT_DELTA OUT_EVIDENCE OUT_TABLE : the step-4 adaptive rows (12 h, blow-up
held test; 40-60 m/s in ROWS_DIR, 20-30 m/s in ROWS2030_DIR) as a candidate map delta on the shipped map's nine winds,
every run as an evidence document, and a plain-text comparison with the shipped adaptive entries."""
import glob, json, sys
from pathlib import Path

rows_dir, rows2030_dir, shipped_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out_delta, out_ev, out_table = Path(sys.argv[4]), Path(sys.argv[5]), Path(sys.argv[6])
shipped = json.loads(shipped_path.read_text(encoding="utf-8"))
WINDS = [float(w) for w in shipped["winds_m_s"]]
srows = {}
for r in shipped["rows"]:
    if "adaptive_s_per_km" in r:
        srows[(float(r["dx_m"]), float(r["crest_m"]), round(float(r["ridge_slope"]), 6))] = r


def load(d):
    out = {}
    for f in sorted(glob.glob(str(d / "*.json"))):
        m = json.loads(Path(f).read_text(encoding="utf-8"))
        if not m.get("complete"):
            raise SystemExit(f"incomplete row {f}")
        out[(m["settings"], m["dx_m"], m["crest_m"], m["ridge_slope"])] = m
    return out


hi, lo = load(rows_dir), load(rows2030_dir)
if set(hi) != set(lo):
    raise SystemExit(f"phase rows differ: {len(hi)} vs {len(lo)}")
first = next(iter(hi.values()))
ladder = first["ladder_s_per_km"]
for m in list(hi.values()) + list(lo.values()):
    assert m["ladder_s_per_km"] == ladder and m["seconds"] == 43200.0 and m["criterion"] == "blowup"
    assert m["targets"] == first["targets"] and m["increase_pct"] == first["increase_pct"]
    assert m["provenance"]["probe_sha256"] == first["provenance"]["probe_sha256"]


def on_map(pairs):
    out = [None] * len(WINDS)
    for w, v in pairs:
        out[WINDS.index(float(w))] = v
    return out


delta_rows, evidence, lines = [], [], []
for key in sorted(hi):
    a, b = lo[key], hi[key]
    winds = a["winds_m_s"] + b["winds_m_s"]
    ent = a["adaptive_s_per_km"] + b["adaptive_s_per_km"]
    top = a["adaptive_top_s_per_km"] + b["adaptive_top_s_per_km"]
    assert a["slope"] == b["slope"] and a["etac"] == b["etac"]
    row = {"arm": b["settings"], "dx_m": b["dx_m"], "crest_m": b["crest_m"], "ridge_slope": b["ridge_slope"],
           "slope": b["slope"], "etac": b["etac"], "sound_steps": 4,
           "adaptive_s_per_km": on_map(zip(winds, ent)), "adaptive_top_s_per_km": on_map(zip(winds, top))}
    s = srows.get((float(b["dx_m"]), float(b["crest_m"]), round(float(b["ridge_slope"]), 6)))
    if s is not None:
        row["shipped_adaptive_s_per_km"] = s["adaptive_s_per_km"]
        row["shipped_adaptive_top_s_per_km"] = s["adaptive_top_s_per_km"]
    delta_rows.append(row)
    runs = a["runs"] + b["runs"]
    evidence.append({**{k: v for k, v in b.items() if k not in ("runs", "winds_m_s", "adaptive_s_per_km",
                                                                  "adaptive_top_s_per_km", "provenance")},
                     "winds_m_s": winds, "adaptive_s_per_km": ent, "adaptive_top_s_per_km": top,
                     "provenance": [a["provenance"], b["provenance"]], "runs": runs})

delta = {
    "schema": "gpuwm-terrain-clock-adaptive-delta-local-face-v1",
    "what": ("Adaptive-clock entries for the terrain clock's local-face candidate (terrain_clock = \"local_face\"), "
             "re-measured under the blow-up held test for 12 hours, on the 2 and 3 km four-substep rows of crests "
             "3000, 3500 and 4500 m, ridge slopes 0.05 to 0.50 (to 0.40 at 3000 and 3500 m), at each candidate arm's "
             "dynamics (generated, ncar).  No code reads this file: it is the delta a ruling would apply to the "
             "adaptive entries the candidate reads (today the shipped map's, measured three hours under the old "
             "bound, at the generated dynamics only, and none at 3500 m).  Each row: the bell ridge of the shipped "
             "probe on the production adaptive clock (tools/terrain_clock_probe.py adaptive-sweep: the controller "
             "fed the dycore's own WRF CFL after every step, WRF's substep count from the live step, min_time_step "
             "3 x dx, landing on every hour, first step 5 s/km), at every CFL target pair in targets with growth "
             "bounded at max_step_increase_pct.  Per wind, max_time_step walks ladder_s_per_km (the shipped adaptive "
             "ladder, 15 to 5 s/km, then 4.5 to 3 s/km, the 3 x dx floor) longest first; 20 and 30 m/s were walked "
             "as one chain and 40, 50 and 60 m/s as another, each wind from the step the weaker wind of its chain "
             "held, and past a wind that held none the stronger winds of its chain were not run.  HELD: the run "
             "stayed finite with peak |w| under blowup_factor x (4 x wind x slope + 20 m/s) for the whole 12 h, at "
             "both target pairs.  adaptive_s_per_km is the first rung that held (null where none did down to 3 "
             "s/km); adaptive_top_s_per_km the first rung tried at that wind (null where that wind was not run; 70 "
             "m/s and up were not run).  The clock reads every wind up to the one it reads at and takes the least, "
             "so the two chains read as one.  shipped_* are the shipped map's entries on the same ridge, where it "
             "has one (generated dynamics, three hours, old bound)."),
    "adaptive": {"targets": first["targets"], "max_step_increase_pct": first["increase_pct"],
                 "ladder_s_per_km": ladder, "seconds": first["seconds"], "criterion": first["criterion"],
                 "blowup_factor": 10.0, "winds_run_m_s": [20.0, 30.0, 40.0, 50.0, 60.0]},
    "winds_m_s": WINDS,
    "probe_sha256": first["provenance"]["probe_sha256"],
    "source": ("CLOCK-CHECK-NCAR-2026-10-06/fix/step4 (box W1, 2026-10-07 09:25-10:44Z, 448 row walks, every run in "
               "terrain_clock_adaptive_rows_evidence-2026-10-07.json)"),
    "rows": delta_rows}


def dump_rows(doc, path):
    head = {k: v for k, v in doc.items() if k != "rows"}
    out = ["{"] + [f" {json.dumps(k)}: {json.dumps(v)}," for k, v in head.items()]
    out.append(' "rows": [')
    out.append(",\n".join("  " + json.dumps(r) for r in doc["rows"]))
    out += [" ]", "}"]
    path.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")


dump_rows(delta, out_delta)
dump_rows({"schema": "gpuwm-terrain-clock-adaptive-rows-evidence-v1", "rows": evidence}, out_ev)
json.loads(out_delta.read_text(encoding="utf-8")); json.loads(out_ev.read_text(encoding="utf-8"))
print(len(delta_rows), "rows;", sum(len(e["runs"]) for e in evidence), "runs")
