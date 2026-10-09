"""Assemble ``tools/terrain_clock_probe.py sweep`` rows into a candidate map.

The candidate is NOT read by the engine: ``gpuwm/terrain_clock.py`` reads
only ``gpuwm/terrain_clock_map.json``.  It carries the rows measured for
the terrain clock's local-face rework (2026-10-07) at two dynamics
settings arms, each row with its settings, provenance and every run it
rests on, so the code change can be made against measured rows instead
of a slope tolerance.

Usage::

    python tools/terrain_clock_candidate.py ROWS_DIR [ROWS_DIR ...] --out FILE

A key measured on more than one host keeps the row that finished first
(both are listed under ``duplicates``)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

MAP_PATH = Path(__file__).resolve().parents[1] / "gpuwm" / "terrain_clock_map.json"

#: A stop is RUNAWAY when the peak vertical velocity at the stop is
#: non-finite or more than this many times the held bound (no less than
#: 200 m/s), and NEAR-BOUND otherwise.  A near-bound stop is not by itself
#: evidence of instability or of its absence: the map's bound stops a run
#: at the first step past it, so a slowly growing instability and a finite
#: wave grown past the bound both stop near 1x.  Named breakage: in
#: three-hour runs the bound stopped 2 km cells (4.5 km crest, slope
#: 0.09-0.15, 40-50 m/s) at every step down to 3 s/km at 1.00-1.06 x the
#: bound; ``terrain_clock_probe.py fixed --criterion blowup`` re-runs such
#: cells to tell the two apart (MEASURE.md).
BLOWUP_FACTOR = 10.0

WHAT = (
    "CANDIDATE rows, not read by the engine.  Each row: a bell ridge across "
    "a periodic 8-row grid in a uniform cross-ridge wind, through the "
    "production step(), generated 49-level ladder, etac from the vertical "
    "survey, Rayleigh lid from 5 km below the 20 km top, at the dynamics "
    "settings named on the row (settings: generated = epssm 0.5, "
    "sixth-order filter 0.12 slope-tapered, the map's own; ncar = epssm "
    "0.1, no sixth-order filter, damp_opt 3, w_damping 1, NCAR's v4.4 "
    "CONUS benchmark as WOOF imports it).  Each wind (20 to 60 m/s) walked "
    "rungs_s_per_km from the step the weaker wind held; every run lasted "
    "three hours (10800 s) on a domain the flow cannot wrap, unless it "
    "stopped.  Held means the run finished with peak |w| under 4 x wind x "
    "slope + 20 m/s, the map's criterion.  stable_s_per_km is the first "
    "rung that held, null where none did (and past such a wind, stronger "
    "winds were not run: top_s_per_km null).  top_s_per_km is the longest "
    "rung tried at that wind.  per_wind lists every run with held, peak w, "
    "the bound, the peak WRF-form vertical Courant number (the quantity "
    "w_damping acts on above 1), the w_damping cell visits, the peak "
    "geometric |w| dt / dz Courant number over the lowest four full levels, "
    "and the stop class (runaway: non-finite or more than 10 x the bound at "
    "the stop; near-bound: otherwise, which a finite wave grown past the "
    "bound and a slowly growing instability both give).")


def _stop_class(run: dict) -> str | None:
    if run["held"]:
        return None
    if run.get("stop") == "growing":
        # criterion "steady": finite and under the blow-up bound, but its
        # last hour's peak w still growing (terrain_clock_probe.STEADY_GROWTH).
        return "growing"
    peak = float(run["peak_w"])
    if not math.isfinite(peak) or peak > BLOWUP_FACTOR * float(run["bound"]):
        return "runaway"
    return "near-bound"


def _slim(run: dict) -> dict:
    return {"per_km": run["per_km"], "dt_s": run["dt"],
            "held": run["held"], "stop": _stop_class(run),
            "stopped_at_s": run["stopped_at_s"],
            "peak_w": (None if not math.isfinite(run["peak_w"])
                       else round(float(run["peak_w"]), 3)),
            "bound": round(float(run["bound"]), 3),
            "peak_wrf_vertical_courant": (
                None if not math.isfinite(run["peak_wrf_vertical_courant"])
                else round(float(run["peak_wrf_vertical_courant"]), 4)),
            "w_damping_cell_visits": run["w_damping_cell_visits"],
            "peak_near_ground_vertical_courant": (
                None if not math.isfinite(
                    run["peak_near_ground_vertical_courant"])
                else round(float(run["peak_near_ground_vertical_courant"]),
                           3)),
            "nx": run["nx"], "steps": run["steps"], "wall_s": run["wall_s"]}


def clip_to_floor(row: dict, floor: float) -> dict:
    """A row walked past ``floor`` (s/km) read as if its rungs stopped
    there: an entry under the floor becomes null and stronger winds
    untried, exactly what the shorter walk would have recorded; runs
    under the floor are kept, marked ``below_floor``."""
    entries, tops = [], []
    cut = False
    for entry, top in zip(row["stable_s_per_km"], row["top_s_per_km"]):
        if cut:
            entries.append(None)
            tops.append(None)
            continue
        if entry is not None and entry < floor - 1e-9:
            entry, cut = None, True
        elif entry is None:
            cut = True
        entries.append(entry)
        tops.append(top)
    row["stable_s_per_km"], row["top_s_per_km"] = entries, tops
    row["rungs_s_per_km"] = [r for r in row["rungs_s_per_km"]
                             if r >= floor - 1e-9]
    for run in row["runs"]:
        run["below_floor"] = bool(run["per_km"] < floor - 1e-9)
    return row


def assemble(dirs, floor: float | None = None) -> dict:
    rows = {}
    duplicates = []
    for directory in dirs:
        for path in sorted(Path(directory).glob("*.json")):
            if path.name.startswith("provenance"):
                continue
            row = json.loads(path.read_text(encoding="utf-8"))
            if floor is not None:
                row = clip_to_floor(row, floor)
            key = row["provenance"]["key"]
            if key in rows:
                duplicates.append({"key": key,
                                   "kept": rows[key]["provenance"],
                                   "other": row["provenance"]})
                if row["provenance"]["finished_utc"] >= rows[key][
                        "provenance"]["finished_utc"]:
                    continue
            rows[key] = row
    out = []
    for key in sorted(rows, key=lambda k: (
            rows[k]["settings"], rows[k]["dx_m"], rows[k]["crest_m"],
            rows[k]["sound_steps"], rows[k]["ridge_slope"])):
        row = rows[key]
        per_wind = []
        for wind in row["winds_m_s"]:
            runs = [{**_slim(r), "below_floor": r.get("below_floor", False)}
                    for r in row["runs"] if float(r["wind"]) == float(wind)]
            per_wind.append({"wind": wind, "runs": runs})
        out.append({
            "dx_m": row["dx_m"], "crest_m": row["crest_m"],
            "ridge_slope": row["ridge_slope"], "slope": row["slope"],
            "sound_steps": row["sound_steps"], "etac": row["etac"],
            "thinnest_layer_fraction": row["thinnest_layer_fraction"],
            "settings": row["settings"], "dynamics": row["dynamics"],
            "criterion": row.get("criterion", "map"),
            "seconds": row["seconds"], "winds_m_s": row["winds_m_s"],
            "stable_s_per_km": row["stable_s_per_km"],
            "top_s_per_km": row["top_s_per_km"],
            "provenance": row["provenance"], "per_wind": per_wind})
    return {"schema": "gpuwm-terrain-clock-map-candidate-v1", "what": WHAT,
            "rungs_s_per_km": (rows[next(iter(rows))]["rungs_s_per_km"]
                               if rows else []),
            "blowup_factor": BLOWUP_FACTOR, "rows": out,
            "duplicates": duplicates}


#: The engine's candidate map (``gpuwm/terrain_clock_map_local_face.json``,
#: read only under ``terrain_clock = "local_face"``): the rows alone.
ENGINE_WHAT = (
    "Rows the terrain clock's local-face candidate (terrain_clock = "
    "\"local_face\", gpuwm/terrain_clock_local.py) reads beside the shipped "
    "map; no other clock reads them.  Each row: a bell ridge across a "
    "periodic 8-row grid in a uniform cross-ridge wind, through the "
    "production step(), the generated 49-level ladder, etac from the "
    "vertical survey, Rayleigh lid from 5 km below the 20 km top, at the "
    "dynamics of its arm (arms).  Each wind walked rungs_s_per_km from the "
    "step the weaker wind held; each run lasted the stated seconds on a "
    "domain the flow cannot wrap.  The criterion says what held: 'steady' "
    "= finite, peak |w| under blowup_factor x (4 x wind x slope + 20 m/s) "
    "the whole run, and the last hour's peak w no more than steady_growth "
    "x the hour before.  stable_s_per_km is the first rung that held, null "
    "where none did down to the last rung; top_s_per_km the longest rung "
    "tried at that wind, null where that wind was not run (past a wind "
    "that held none), which the engine does not read.  Every run behind "
    "each entry is in the candidate document named under source.")


def engine_map(document: dict, *, source: str, steady_growth: float
               ) -> dict:
    """The engine's slim candidate map from an assembled document."""
    rows = document["rows"]
    criteria = sorted({row["criterion"] for row in rows})
    if len(criteria) != 1:
        raise SystemExit(f"rows measured under {criteria}: one criterion "
                         "per engine map")
    winds = sorted({tuple(row["winds_m_s"]) for row in rows})
    if len(winds) != 1:
        raise SystemExit(f"rows at different winds: {winds}")
    seconds = sorted({row["seconds"] for row in rows})
    arms = {}
    for row in rows:
        dyn = arms.setdefault(row["settings"], row["dynamics"])
        if dyn != row["dynamics"]:
            raise SystemExit(f"arm {row['settings']} carries two dynamics")
    # The blow-up criterion (lead decision (i), 2026-10-07) holds a run
    # that stayed finite and under blowup_factor x the old bound for the
    # whole run; it reads no growth threshold.
    head = ({"what": DIRECT_WHAT, "criterion": "blowup",
             "blowup_factor": BLOWUP_FACTOR} if criteria[0] == "blowup" else
            {"what": ENGINE_WHAT, "criterion": criteria[0],
             "blowup_factor": BLOWUP_FACTOR, "steady_growth": steady_growth})
    return {
        "schema": "gpuwm-terrain-clock-map-local-face-v1", **head,
        "seconds": seconds[0] if len(seconds) == 1 else seconds,
        "winds_m_s": list(winds[0]),
        "rungs_s_per_km": document["rungs_s_per_km"],
        "arms": arms, "source": source,
        "rows": [{key: row[key] for key in (
            "dx_m", "crest_m", "ridge_slope", "slope", "sound_steps",
            "settings", "stable_s_per_km", "top_s_per_km")}
            for row in rows]}


#: The engine's candidate map when every row was swept directly under the
#: blow-up criterion (``terrain_clock_probe.py sweep --criterion blowup``).
DIRECT_WHAT = (
    "Rows the terrain clock's local-face candidate (terrain_clock = "
    "\"local_face\", gpuwm/terrain_clock_local.py) reads; no other clock "
    "reads them.  Each row: a bell ridge across a periodic 8-row grid in a "
    "uniform cross-ridge wind, through the production step(), the "
    "generated 49-level ladder, etac from the vertical survey, Rayleigh "
    "lid from 5 km below the 20 km top, at the dynamics of its arm "
    "(arms).  Each wind walked rungs_s_per_km from the step the weaker "
    "wind held; each run lasted the stated seconds on a domain the flow "
    "cannot wrap.  Criterion 'blowup' (lead decision 2026-10-07): a step "
    "HELD where its run stayed finite with peak |w| under blowup_factor x "
    "(4 x wind x slope + 20 m/s) for the whole run; the old bound alone "
    "stopped finite mountain waves on gentle 2 and 3 km ridges.  "
    "stable_s_per_km is the first rung that held, null where none did down "
    "to the last rung; top_s_per_km the longest rung tried at that wind, "
    "null where that wind was not run (past a wind that held none), which "
    "the engine does not read.  An entry under its top saw every longer "
    "rung tried run away, and is read as a stop.  Where a row's ridge is "
    "one the shipped map ran, the engine merges the two "
    "(terrain_clock_local.merge_cell).  Every run behind each entry is in "
    "the candidate document named under source.")


def _write_rows(path: Path, document: dict) -> None:
    head = {k: v for k, v in document.items() if k != "rows"}
    lines = ["{"]
    for key, value in head.items():
        lines.append(f" {json.dumps(key)}: {json.dumps(value)},")
    lines.append(' "rows": [')
    lines.append(",\n".join("  " + json.dumps(r) for r in document["rows"]))
    lines.append(" ]")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8",
                    newline="\n")




# ---------------------------------------------------------------------------
# The blow-up criterion, from every run measured.
# ---------------------------------------------------------------------------

#: What a run says about its rung under the blow-up criterion (finite and
#: under BLOWUP_FACTOR x the map's bound for the whole run, or not).
FINITE, RUNAWAY, AT_BOUND = "finite", "runaway", "stopped at the bound"


def _evidence_of(run: dict, source: str) -> str | None:
    """``finite``, ``runaway`` or ``stopped at the bound``.

    * a run the map's bound held, or the blow-up or steady criterion ran
      to the end (held, or stopped only as still growing), ran the whole
      three hours finite and under the blow-up bound: ``finite``;
    * a run stopped non-finite or past BLOWUP_FACTOR x the bound:
      ``runaway``;
    * a run the map's bound stopped near it says nothing about blow-up
      (it was stopped at the bound): tried, neither held nor run away."""
    if run["held"]:
        return FINITE
    if run.get("stop") == "growing":
        return FINITE
    peak = float(run["peak_w"])
    if not math.isfinite(peak) or peak > BLOWUP_FACTOR * float(run["bound"]):
        return RUNAWAY
    if source == "map":
        return AT_BOUND
    raise SystemExit(f"a {source} run stopped under the blow-up bound: {run}")


def evidence_rows(map_dirs, recheck_dirs, sweep_dirs, *, rungs, winds,
                  floor: float) -> dict:
    """Engine rows under the blow-up criterion from every run measured.

    Per cell (row, wind) and rung (s/km): every run's evidence; a rung seen
    both finite and running away is refused (the probe is deterministic,
    S and G agree to the bit).  The entry is the longest rung seen finite,
    ``None`` where none was and one ran away, and ``top`` the longest rung
    with any evidence; a cell with none is not measured (both ``None``).
    Rungs under ``floor`` are not read."""
    cells: dict[tuple, dict[float, str]] = {}
    meta: dict[tuple, dict] = {}

    def add(key, wind, per_km, verdict):
        if per_km < floor - 1e-9:
            return
        rung = round(float(per_km), 6)
        seen = cells.setdefault(key + (float(wind),), {})
        old = seen.get(rung)
        if verdict == AT_BOUND:
            seen.setdefault(rung, verdict)
            return
        if old not in (None, AT_BOUND, verdict):
            raise SystemExit(f"{key} {wind} m/s {rung} s/km: finite and "
                             "running away")
        seen[rung] = verdict

    def row_key(row):
        return (row["settings"], float(row["dx_m"]), float(row["crest_m"]),
                float(row["ridge_slope"]), int(row["sound_steps"]))

    for kind, dirs in (("map", map_dirs), ("steady", sweep_dirs)):
        for directory in dirs:
            for path in sorted(Path(directory).glob("*.json")):
                if path.name.startswith("provenance"):
                    continue
                row = json.loads(path.read_text(encoding="utf-8"))
                key = row_key(row)
                meta.setdefault(key, {k: row[k] for k in (
                    "dx_m", "crest_m", "ridge_slope", "slope", "sound_steps",
                    "settings", "dynamics")})
                for run in row["runs"]:
                    add(key, run["wind"], run["per_km"],
                        _evidence_of(run, kind))
    for directory in recheck_dirs:
        for path in sorted(Path(directory).glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                run = json.loads(line)
                key = (run["settings"], float(run["dx_m"]),
                       float(run["crest_m"]), float(run["ridge_slope"]),
                       int(run["sound_steps"]))
                add(key, run["wind"], run["per_km"],
                    _evidence_of(run, "blowup"))
    rows = []
    for key in sorted(meta):
        stable, tried = [], []
        for wind in winds:
            seen = cells.get(key + (float(wind),), {})
            if not seen:
                stable.append(None)
                tried.append(None)
                continue
            finite = [r for r, v in seen.items() if v == FINITE]
            stable.append(max(finite) if finite else None)
            tried.append(max(seen))
            if not finite and RUNAWAY not in seen.values():
                # Tried only to the bound: nothing held, nothing ran away.
                stable[-1] = None
        rows.append({**meta[key], "stable_s_per_km": stable,
                     "top_s_per_km": tried,
                     "rungs_seen": {str(w): {str(r): v for r, v in sorted(
                         cells.get(key + (float(w),), {}).items(),
                         reverse=True)} for w in winds}})
    return {"rows": rows, "rungs_s_per_km": list(rungs)}


def evidence_engine_map(document: dict, *, source: str) -> dict:
    """The engine's slim candidate map under the blow-up criterion."""
    rows = document["rows"]
    arms = {}
    for row in rows:
        if arms.setdefault(row["settings"], row["dynamics"]) != row[
                "dynamics"]:
            raise SystemExit(f"arm {row['settings']} carries two dynamics")
    return {
        "schema": "gpuwm-terrain-clock-map-local-face-v1",
        "what": EVIDENCE_WHAT, "criterion": "blowup",
        "blowup_factor": BLOWUP_FACTOR, "seconds": 10800.0,
        "winds_m_s": [20.0, 30.0, 40.0, 50.0, 60.0],
        "rungs_s_per_km": document["rungs_s_per_km"], "arms": arms,
        "source": source,
        "rows": [{key: row[key] for key in (
            "dx_m", "crest_m", "ridge_slope", "slope", "sound_steps",
            "settings", "stable_s_per_km", "top_s_per_km")}
            for row in rows]}


EVIDENCE_WHAT = (
    "Rows the terrain clock's local-face candidate (terrain_clock = "
    "\"local_face\", gpuwm/terrain_clock_local.py) reads beside the shipped "
    "map; no other clock reads them.  Each row: a bell ridge across a "
    "periodic 8-row grid in a uniform cross-ridge wind, through the "
    "production step(), the generated 49-level ladder, etac from the "
    "vertical survey, Rayleigh lid from 5 km below the 20 km top, at the "
    "dynamics of its arm (arms), each run three hours on a domain the flow "
    "cannot wrap.  Criterion 'blowup': a step HELD where a run at it stayed "
    "finite with peak |w| under blowup_factor x (4 x wind x slope + 20 m/s) "
    "the whole three hours.  Every run of the three measurements named "
    "under source counts: a run the map's bound held ran finite; a run "
    "stopped non-finite or past the blow-up bound ran away; a run the "
    "map's bound stopped near it was tried and neither held nor ran away.  "
    "stable_s_per_km is the longest rung (rungs_s_per_km, down to 3 s/km) "
    "seen to hold, null where none did and one ran away; top_s_per_km the "
    "longest rung with any evidence, null where the cell was never run "
    "(the engine does not read such a cell).  An entry under its top saw "
    "a longer step run away or go unmeasured at the bound, and is read as "
    "a stop.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dirs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--engine-out", type=Path, default=None,
                        help="also write the engine's slim candidate map")
    parser.add_argument("--source", default="",
                        help="where the full candidate document lives, "
                             "recorded in the engine map")
    parser.add_argument("--floor-per-km", type=float, default=None,
                        help="read every row as walked no further than this")
    parser.add_argument("--evidence", action="store_true",
                        help="the blow-up criterion from every run: DIRS are "
                             "map-criterion row dirs, --recheck and --sweep "
                             "add the others; --engine-out is then written "
                             "from them")
    parser.add_argument("--recheck", type=Path, nargs="*", default=[])
    parser.add_argument("--sweep", type=Path, nargs="*", default=[])
    args = parser.parse_args(argv)
    if args.evidence:
        document = evidence_rows(
            args.dirs, args.recheck, args.sweep,
            rungs=[6.5, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0],
            winds=[20.0, 30.0, 40.0, 50.0, 60.0],
            floor=args.floor_per_km or 3.0)
        _write_rows(args.out, document)
        if args.engine_out is not None:
            _write_rows(args.engine_out, evidence_engine_map(
                document, source=args.source))
        print(json.dumps({"rows": len(document["rows"])}))
        return 0
    document = assemble(args.dirs, args.floor_per_km)
    document["floor_per_km"] = args.floor_per_km
    _write_rows(args.out, document)
    if args.engine_out is not None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "terrain_clock_probe", Path(__file__).with_name(
                "terrain_clock_probe.py"))
        probe = importlib.util.module_from_spec(spec)
        import sys
        sys.modules[spec.name] = probe
        spec.loader.exec_module(probe)
        _write_rows(args.engine_out, engine_map(
            document, source=args.source,
            steady_growth=probe.STEADY_GROWTH))
    print(json.dumps({"rows": len(document["rows"]),
                      "duplicates": len(document["duplicates"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
