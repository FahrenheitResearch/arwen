"""clock4.py DELTA.json VARIANT PREPARED_ROOT TOML OUT.json : clock3.py (the tree runner's own CPU preflight) with the
local-face candidate's map carrying the step-4 adaptive rows (DELTA.json, terrain_clock_adaptive_delta_local_face.json)
in place of the shipped adaptive entries on the rows it covers.  VARIANT "as-is" reads them with the lane's code
unchanged; "none-held-cap" also applies finding 1's proposed fix (an adaptive domain with no fixed limit, where a cell
read held no longest step tried, is capped at the shortest step tried there).  CPU only, no card.  Nothing here changes
the lane's code; the overlay is a monkeypatch for this reading only."""
import functools, json, os, sys
from dataclasses import replace
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("GPUWM_NO_LOCAL_GPU", "1")
import gpuwm.terrain_clock as tc
import gpuwm.terrain_clock_local as tl

delta_path, variant = Path(sys.argv[1]), sys.argv[2]
root, cfg, outp = Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
DELTA = json.loads(delta_path.read_text(encoding="utf-8"))
WINDS = [float(w) for w in tc.measured_map().winds]
BLOCK = DELTA["adaptive"]
MEAS = tc.AdaptiveMeasurement(targets=tuple((float(a), float(b)) for a, b in BLOCK["targets"]),
                              max_step_increase_pct=int(BLOCK["max_step_increase_pct"]),
                              ladder=tuple(float(v) for v in BLOCK["ladder_s_per_km"]),
                              seconds=float(BLOCK["seconds"]))
NEW = {}
for row in DELTA["rows"]:
    NEW.setdefault((row["arm"], float(row["dx_m"]), float(row["crest_m"])), []).append(row)


def lookup(arm, row):
    for cand in NEW.get((arm, float(row.dx_m), float(row.crest_m)), ()):
        if abs(float(cand["slope"]) - float(row.slope)) <= 5e-5:
            return cand
    return None


def cells(arms, row):
    found = [lookup(arm, row) for arm in arms]
    if not found or any(f is None for f in found):
        return None
    entries, tops = [], []
    for i in range(len(WINDS)):
        e = [f["adaptive_s_per_km"][i] for f in found]
        t = [f["adaptive_top_s_per_km"][i] for f in found]
        if any(v is None for v in t):
            entries.append(None); tops.append(None); continue
        entries.append(None if any(v is None for v in e) else min(e)); tops.append(min(t))
    return tuple(entries), tuple(tops)


@functools.lru_cache(maxsize=8)
def overlay(arms):
    """tl.candidate_map(arms) with the delta's adaptive cells on the rows it measured (per arm; a shipped row no arm
    re-measured, read by every arm, takes the least of the arms' cells)."""
    base = tc.measured_map()
    rows = []
    seen = set()
    for arm in arms:
        merged, added = tl._arm_rows(arm)
        for position, (item, shipped_row) in enumerate(zip(merged, base.rows)):
            if item["arm"] is None:
                if position in seen:
                    continue
                seen.add(position)
                row, who = shipped_row, tuple(arms)
            else:
                row, who = replace(shipped_row, stable=tuple(item["stable"]), tried=tuple(item["tried"])), (arm,)
            got = cells(who, row) if row.sound_steps == 4 else None
            rows.append(row if got is None else replace(row, adaptive=got[0], adaptive_tried=got[1]))
        for item in added:
            r = item["row"]
            row = tc.MapRow(float(r["dx_m"]), float(r["crest_m"]), float(r["slope"]), int(r["sound_steps"]),
                            tuple(item["stable"]), tuple(item["tried"]))
            got = cells((arm,), row) if row.sound_steps == 4 else None
            rows.append(row if got is None else replace(row, adaptive=got[0], adaptive_tried=got[1]))
    return replace(base, rows=tuple(rows), adaptive=MEAS)


tl.candidate_map = overlay
if variant == "none-held-cap":
    _orig_cap = tc._adaptive_cap

    def _cap(reading, limit, run, table):
        cap, source = _orig_cap(reading, limit, run, table)
        if cap is None and reading.adaptive_none_held and reading.adaptive_none_held_below_per_km is not None:
            return reading.adaptive_none_held_below_per_km, "adaptive none held"
        return cap, source
    tc._adaptive_cap = _cap
elif variant != "as-is":
    raise SystemExit(f"variant {variant}")

from gpuwm.fetch import sha256_file
from gpuwm.go_cli import _hierarchy_document
from gpuwm.prepared_domain_tree_forecast import preflight_prepared_tree
doc = _hierarchy_document(root)
inputs = preflight_prepared_tree(prepared_root=root, preparation_receipt_sha256=sha256_file(doc),
                                 experiment_config=cfg, experiment_config_sha256=sha256_file(cfg))
tcr = dict(inputs.terrain_clock)
json.dump({"prepared_root": str(root), "config": str(cfg), "delta": str(delta_path), "variant": variant,
           "terrain_clock": tcr}, open(outp, "w"), indent=1, default=str)
for d in tcr.get("domains", ()):
    lf = d.get("local_faces") or {}
    g = lf.get("governing") or {}
    mr = d.get("map_rows") or {}
    print(cfg.name, variant, "d%02d" % int(d["grid_id"]), d.get("status"), "dt", (d.get("dt_s") or {}).get("seconds"),
          "max_step", (d.get("max_time_step_s") or {}).get("seconds") if isinstance(d.get("max_time_step_s"), dict)
          else d.get("max_time_step_s"), "from", d.get("max_time_step_from"), "arms", lf.get("arms"),
          "| adaptive:", mr.get("adaptive_clock"))
