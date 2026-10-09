"""Measure and check the terrain clock's local-face candidate on real inputs.

``gpuwm/terrain_clock_local.py`` reads every terrain face with the crest and
crest-level wind in a neighbourhood around it, times a wind margin.  This
tool runs it through the WRF-input door the way a forecast does, on the CPU
(no card), and keeps what a claim about it rests on.

Subcommands:

* ``door DIR --run-seconds S --out OUT``: the door's own derivation
  (``gpuwm.wrfinput_door.resolve_wrfinput_run``, ``prepare_wrf_run``, the
  substep and terrain-clock derivations ``_with_terrain_acoustics`` makes),
  for the clock the namelist's selector comment names.  Writes the run's
  ``terrain-clock.json`` and ``acoustic-substeps.json`` receipts and, for
  each domain read face by face, the fields its reading read
  (``faces-dNN.npz``: terrain, map factors, every column's crest-band wind,
  which input carried it) as a fixture, captured from the door itself.
* ``reproduce FIXTURE RECEIPT``: the reading again from a saved fixture, on
  the engine's own functions, compared with the door's receipt.
* ``margin FIXTURE WRFOUT``: every face's crest-level wind in a reference
  forecast's output (the same window and band), against what its inputs
  carried: the factor each face needs for its input wind, times the
  factor, to read a map column at or above the column of the wind the
  forecast carried over it.
* ``slim FIXTURE --out JSON``: the face groups a fixture reads (each
  group's leading face and its values), small enough for the test suite.
* ``moved --out JSON``: every shipped map row a candidate arm's three-hour
  row on the same ridge changes, shipped and merged entries side by side
  (``gpuwm.terrain_clock_local.moved_rows``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _save_fixture(path: Path, fields, meta: dict) -> None:
    np.savez_compressed(
        path, terrain=np.asarray(fields.terrain, dtype=np.float32),
        msfu=(np.zeros(0, np.float32) if fields.msfu is None
              else np.asarray(fields.msfu, dtype=np.float32)),
        msfv=(np.zeros(0, np.float32) if fields.msfv is None
              else np.asarray(fields.msfv, dtype=np.float32)),
        wind=np.asarray(fields.wind, dtype=np.float32),
        which=np.asarray(fields.which, dtype=np.int8),
        meta=np.frombuffer(json.dumps({
            **meta, "sources": list(fields.sources),
            "radius": int(fields.radius), "dx": float(fields.dx),
            "dy": float(fields.dy)}).encode("utf-8"), dtype=np.uint8))


def load_fixture(path):
    """``(FaceFields, meta)`` from a saved fixture."""
    from gpuwm.terrain_clock_local import FaceFields

    data = np.load(path)
    meta = json.loads(bytes(data["meta"]).decode("utf-8"))
    msfu = data["msfu"] if data["msfu"].size else None
    msfv = data["msfv"] if data["msfv"].size else None
    fields = FaceFields(
        terrain=np.asarray(data["terrain"], dtype=np.float64),
        msfu=None if msfu is None else np.asarray(msfu, dtype=np.float64),
        msfv=None if msfv is None else np.asarray(msfv, dtype=np.float64),
        wind=np.asarray(data["wind"], dtype=np.float64),
        which=np.asarray(data["which"], dtype=np.int16),
        sources=tuple(meta["sources"]), radius=int(meta["radius"]),
        dx=float(meta["dx"]), dy=float(meta["dy"]))
    return fields, meta


def door(directory: Path, run_seconds: float, out: Path) -> dict:
    import gpuwm.terrain_clock_local as local
    from gpuwm.prepared_domain_tree_forecast import _with_terrain_acoustics
    from gpuwm.wrfinput_door import resolve_wrfinput_run
    from gpuwm.wrfinput_forecast import prepare_wrf_run

    out.mkdir(parents=True, exist_ok=True)
    captured = []
    original = local.local_readings

    def recording(exp, **kwargs):
        result = original(exp, **kwargs)
        captured.append((exp, kwargs, result))
        return result

    local.local_readings = recording
    try:
        run = resolve_wrfinput_run(directory, rrtmg_variant="rrtmg_legacy")
        inputs = prepare_wrf_run(run, out / "prep", run_seconds=run_seconds)
        inputs = _with_terrain_acoustics(inputs)
    finally:
        local.local_readings = original
    receipt = dict(inputs.terrain_clock)
    (out / "terrain-clock.json").write_text(
        json.dumps(receipt, indent=1, default=str), encoding="utf-8")
    (out / "acoustic-substeps.json").write_text(
        json.dumps(dict(inputs.acoustic_substeps), indent=1, default=str),
        encoding="utf-8")
    for exp, kwargs, result in captured:
        for gid, faces in result.items():
            if isinstance(faces, str):
                continue
            dc = next(d for d in exp.domains if int(d.grid_id) == gid)
            _save_fixture(out / f"faces-d{gid:02d}.npz", faces.fields, {
                "grid_id": gid, "domain_wind_m_s": faces.domain_wind,
                "arms": list(faces.arms), "radius_m": faces.radius_m,
                "margin": faces.margin, "steepest_slope": float(
                    kwargs["slopes"][gid]),
                "crest": kwargs["winds"][gid].receipt(),
                "run": {name: getattr(dc.run, name) for name in (
                    "dx", "dy", "dt", "time_step_sound", "epssm", "smdiv",
                    "emdiv", "damp_opt", "zdamp", "dampcoef", "w_damping",
                    "diff_6th_opt", "diff_6th_factor", "diff_6th_slopeopt",
                    "use_adaptive_time_step", "terrain_clock")}})
    rows = []
    for row in receipt.get("domains", ()):
        rows.append({key: row.get(key) for key in (
            "grid_id", "status", "clock", "configured_dt_s", "dt_s",
            "time_step_sound", "held_s_per_km", "limit_s_per_km")})
    summary = {"directory": str(directory), "run_seconds": run_seconds,
               "domains": rows}
    print(json.dumps(summary, indent=1))
    return summary


class _Run:
    """The run fields the derivation reads, from a fixture's meta."""

    def __init__(self, fields: dict):
        for key, value in fields.items():
            setattr(self, key, value)


def reproduce(fixture: Path, receipt_path: Path) -> int:
    from fractions import Fraction

    from gpuwm.terrain_clock import CrestWind
    from gpuwm.terrain_clock_local import (DomainFaces, candidate_map,
                                           derive_local)

    fields, meta = load_fixture(fixture)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    row = next(r for r in receipt["domains"]
               if int(r["grid_id"]) == int(meta["grid_id"]))
    arms = tuple(meta["arms"])
    faces = DomainFaces(fields=fields, table=candidate_map(arms), arms=arms,
                        radius_m=float(meta["radius_m"]),
                        domain_wind=float(meta["domain_wind_m_s"]),
                        dx=float(meta["run"]["dx"]),
                        margin=float(meta["margin"]))
    crest = meta["crest"]
    wind = CrestWind(label=f"d{meta['grid_id']:02d}",
                     crest_height_m=crest["crest_height_m"],
                     wind_m_s=crest["crest_level_wind_m_s"],
                     when=crest["when"], source=crest["source"],
                     height_m=crest["height_m"])
    dt = Fraction(row["configured_dt_s"]["numerator"],
                  row["configured_dt_s"]["denominator"])
    run = _Run({**meta["run"], "terrain_clock": "local_face"})
    adaptation = derive_local(int(meta["grid_id"]), run, dt,
                              float(meta["steepest_slope"]), wind, faces)
    again = adaptation.receipt()
    same = (json.dumps(again["local_faces"], sort_keys=True)
            == json.dumps(row["local_faces"], sort_keys=True)
            and again["dt_s"] == row["dt_s"]
            and again["time_step_sound"] == row["time_step_sound"])
    print(json.dumps({"fixture": str(fixture), "dt_s": again["dt_s"],
                      "status": again["status"],
                      "reproduces_door_receipt": same}, indent=1))
    return 0 if same else 1


def _wrfout_column_band(path: Path, tops: np.ndarray) -> np.ndarray:
    import netCDF4

    from gpuwm.terrain_clock import StartWinds

    with netCDF4.Dataset(path) as nc:
        nc.set_auto_mask(False)
        arrays = {"u": nc["U"][0], "v": nc["V"][0], "php": nc["PH"][0],
                  "phb": nc["PHB"][0]}
    start = StartWinds("reference forecast", arrays.__getitem__)
    return start.column_band(tops)


def margin(fixture: Path, wrfout: Path, out: Path | None) -> dict:
    from gpuwm.terrain_clock_local import (_faces, _window_max, band_tops,
                                           candidate_map)

    fields, meta = load_fixture(fixture)
    table = candidate_map(tuple(meta["arms"]))
    winds = np.asarray(table.winds)
    tops = band_tops(fields.terrain, fields.radius)
    end_column = _wrfout_column_band(wrfout, tops)
    axis, j, i, slope, crest, wind_in = _faces(fields)
    end = np.where(np.isfinite(end_column), end_column, -np.inf)
    wend = _window_max(end, fields.radius)
    w_end = np.concatenate([np.maximum(wend[:, :-1], wend[:, 1:]).ravel(),
                            np.maximum(wend[:-1, :], wend[1:, :]).ravel()])
    domain = float(meta["domain_wind_m_s"])
    column = np.searchsorted(winds, w_end - 1e-9, side="left")
    # The column the end wind reads; the input wind times the factor must
    # pass the column below it (read_map takes the first column at or
    # above the wind).
    below = np.where(column > 0, winds[np.clip(column - 1, 0,
                                                winds.size - 1)], 0.0)
    need = np.where(column > 0, below / np.maximum(wind_in, 1e-6), 0.0)
    over_domain = w_end > domain + 1e-9
    worst = int(np.argmax(need))
    order = np.argsort(-need)
    top = [{"face": {"axis": "x" if int(axis[k]) == 0 else "y",
                     "j": int(j[k]), "i": int(i[k])},
            "slope": round(float(slope[k]), 4),
            "crest_m": round(float(crest[k]), 1),
            "input_wind_m_s": round(float(wind_in[k]), 2),
            "end_wind_m_s": round(float(w_end[k]), 2),
            "factor_needed_over": round(float(need[k]), 4)}
           for k in order[:25]]
    pct = {f"p{q}": round(float(np.percentile(need, q)), 4)
           for q in (50, 90, 99, 99.9)}
    ratio = np.where(wind_in > 0, w_end / np.maximum(wind_in, 1e-6), 0.0)
    # THE DECISION, not the column: a face's input wind read a column
    # below its end wind's harms only where the map holds a longer step
    # there than at the end wind's column.  The reading only shortens as
    # the wind column rises (every weaker wind is read too), so per face
    # the factor needed is the one that lifts its input wind past the
    # column below the weakest column whose limit is the end wind's.
    decide = _decision_needs(fields, meta, table, slope, crest, wind_in,
                             w_end, domain)
    hindsight = _domain_limits(fields, meta, table, slope, crest, wind_in,
                               w_end, domain)
    result = {"fixture": str(fixture), "wrfout": str(wrfout),
              "decision": decide, "domain_limits": hindsight,
              "faces": int(need.size),
              "factor_needed_over": round(float(need[worst]), 4),
              "factor_needed_percentiles": pct,
              "end_over_input_ratio_max": round(float(ratio.max()), 4),
              "faces_end_wind_over_domain_reading": int(over_domain.sum()),
              "domain_wind_m_s": domain,
              "end_wind_max_m_s": round(float(w_end.max()), 2),
              "largest_needs": top}
    print(json.dumps({k: v for k, v in result.items()
                      if k != "largest_needs"}, indent=1))
    if out is not None:
        out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def _domain_limits(fields, meta, table, slope, crest, wind_in, w_end,
                   domain) -> dict:
    """THE CHECK ON THE MARGIN THAT DECIDES: the domain's limit (s/km, the
    least over every face, :func:`gpuwm.terrain_clock_local.combine`)
    with each face read at (a) the margin, as the door reads it; (b) the
    larger of that and the wind the reference forecast carried over the
    face at the end hour; (c) the end-hour wind alone.  Where (b) equals
    (a), no face's growth in the reference forecast passed what the margin
    covers in a way that moves the domain's step."""
    from gpuwm.terrain_clock import read_map
    from gpuwm.terrain_clock_local import _keys, _limit, combine

    dx = float(meta["run"]["dx"])
    margin = float(meta["margin"])
    local = np.where(np.isfinite(wind_in), wind_in, domain)
    at_margin = np.minimum(margin * local, domain)
    arms = {"margin": at_margin,
            "margin_or_end_wind": np.maximum(at_margin, w_end),
            "end_wind": np.where(np.isfinite(w_end), w_end, 0.0)}
    out = {}
    for count in (4, 6):
        here = {}
        for name, wind in arms.items():
            keys = _keys(table, dx, slope, crest, wind, count)
            _uniq, first = np.unique(keys, return_index=True)
            readings = [read_map(dx, float(crest[k]), float(slope[k]),
                                 float(wind[k]), count, table)
                        for k in first]
            combined, governing = combine(readings)
            k = int(first[readings.index(governing)])
            here[name] = {
                "limit_s_per_km": _limit(combined),
                "governing_face": {"index": k,
                                   "slope": round(float(slope[k]), 4),
                                   "crest_m": round(float(crest[k]), 1),
                                   "wind_read_m_s": round(float(wind[k]), 2),
                                   "input_wind_m_s": round(float(
                                       wind_in[k]), 2),
                                   "end_wind_m_s": round(float(w_end[k]),
                                                         2)}}
        out[str(count)] = here
    return out


def _decision_needs(fields, meta, table, slope, crest, wind_in, w_end,
                    domain) -> dict:
    """Per substep count, the factor and the added wind each face needs
    for its input wind to read a limit no longer than its end wind's,
    over every face, and the faces that need the most."""
    from gpuwm.terrain_clock import read_map
    from gpuwm.terrain_clock_local import _keys, _limit

    dx = float(meta["run"]["dx"])
    winds = np.asarray(table.winds)
    out = {}
    for count in (4, 6):
        # The bin each face reads, wind column aside.
        base_keys = _keys(table, dx, slope, crest, np.zeros_like(slope),
                          count)
        uniq, inverse = np.unique(base_keys, return_inverse=True)
        # One representative face per bin; its limit at every column.
        first = np.zeros(uniq.size, dtype=np.int64)
        first[inverse[::-1]] = np.arange(inverse.size)[::-1]
        limits = np.empty((uniq.size, winds.size))
        for b, k in enumerate(first):
            for c, w in enumerate(winds):
                limits[b, c] = _limit(read_map(dx, float(crest[k]),
                                               float(slope[k]), float(w),
                                               count, table))
        end_col = np.minimum(np.searchsorted(winds, w_end - 1e-9), winds.size
                             - 1)
        end_limit = limits[inverse, end_col]
        # The weakest column whose limit is no longer than the end wind's.
        ok = limits[inverse] <= end_limit[:, None] + 1e-12
        weakest = np.argmax(ok, axis=1)
        below = np.where(weakest > 0, winds[np.clip(weakest - 1, 0, None)],
                         0.0)
        factor = np.where(weakest > 0, below / np.maximum(wind_in, 1e-6),
                          0.0)
        added = np.where(weakest > 0, below - wind_in, 0.0)
        within = np.minimum(np.searchsorted(
            winds, np.minimum(domain, np.inf) - 1e-9), winds.size - 1)
        cap_short = int((weakest > within).sum())
        order = np.argsort(-factor)[:15]
        # Growth where the wind decides at all: faces whose ground holds a
        # different limit at some wind column (a stop seen somewhere in
        # 20 to 100 m/s), with an end wind past the first column.
        sensitive = (limits[inverse].max(axis=1) > limits[inverse].min(
            axis=1) + 1e-12) & (w_end > winds[0] + 1e-9)
        ratio = np.where(sensitive, w_end / np.maximum(wind_in, 1e-6), 0.0)
        top_ratio = np.argsort(-ratio)[:15]
        out[str(count)] = {
            "wind_sensitive_faces": int(sensitive.sum()),
            "growth_on_wind_sensitive_faces_max": round(float(ratio.max()),
                                                        4),
            "growth_on_wind_sensitive_faces_p99": (round(float(
                np.percentile(ratio[sensitive], 99)), 4)
                if sensitive.any() else None),
            "largest_growth": [{"face_index": int(k),
                                "slope": round(float(slope[k]), 4),
                                "crest_m": round(float(crest[k]), 1),
                                "input_wind_m_s": round(float(wind_in[k]),
                                                        2),
                                "end_wind_m_s": round(float(w_end[k]), 2),
                                "ratio": round(float(ratio[k]), 4)}
                               for k in top_ratio],
            "factor_needed_over": round(float(factor.max()), 4),
            "added_wind_needed_over_m_s": round(float(added.max()), 3),
            "faces_needing_more_than_the_input": int((weakest > np.minimum(
                np.searchsorted(winds, wind_in - 1e-9), winds.size - 1)
            ).sum()),
            "faces_the_domain_cap_cannot_cover": cap_short,
            "faces_needing_more_than_the_margin": int(
                (factor > float(meta["margin"]) + 1e-12).sum()),
            "wind_sensitive_faces_needing_more_than_the_margin": int(
                ((factor > float(meta["margin"]) + 1e-12)
                 & sensitive).sum()),
            "largest": [{"face_index": int(k), "slope": round(float(slope[k]), 4),
                         "crest_m": round(float(crest[k]), 1),
                         "input_wind_m_s": round(float(wind_in[k]), 2),
                         "end_wind_m_s": round(float(w_end[k]), 2),
                         "factor": round(float(factor[k]), 4),
                         "added_m_s": round(float(added[k]), 2)}
                        for k in order]}
    return out


def slim(fixture: Path, out: Path, receipt_path: Path | None) -> None:
    """Each face group a fixture's reading takes, at both substep counts,
    as the test suite keeps it."""
    from gpuwm.terrain_clock_local import DomainFaces, candidate_map

    fields, meta = load_fixture(fixture)
    arms = tuple(meta["arms"])
    faces = DomainFaces(fields=fields, table=candidate_map(arms), arms=arms,
                        radius_m=float(meta["radius_m"]),
                        domain_wind=float(meta["domain_wind_m_s"]),
                        dx=float(meta["run"]["dx"]),
                        margin=float(meta["margin"]))
    document = {"meta": {k: v for k, v in meta.items()}, "counts": {}}
    for count in (4, 6):
        reading = faces.reading(count)
        document["counts"][str(count)] = [
            {"faces": g.count, "axis": g.axis, "j": g.j, "i": g.i,
             "slope": g.slope, "crest_m": g.crest_m,
             "input_wind_m_s": g.input_wind_m_s,
             "wind_read_m_s": g.wind_read_m_s,
             "wind_capped": g.wind_capped, "source": g.source}
            for g in reading.groups]
    if receipt_path is not None:
        document["door_receipt"] = json.loads(
            receipt_path.read_text(encoding="utf-8"))
    out.write_text(json.dumps(document, indent=1), encoding="utf-8")


def moved(out: Path) -> dict:
    from gpuwm.terrain_clock_local import candidate_document, moved_rows

    arms = sorted(candidate_document()["arms"])
    document = {"arms": arms, "rows": {arm: moved_rows(arm) for arm in arms}}
    out.write_text(json.dumps(document, indent=1), encoding="utf-8")
    print(json.dumps({arm: len(rows) for arm, rows in
                      document["rows"].items()}))
    return document


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("door")
    d.add_argument("directory", type=Path)
    d.add_argument("--run-seconds", type=float, required=True)
    d.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("reproduce")
    r.add_argument("fixture", type=Path)
    r.add_argument("receipt", type=Path)
    m = sub.add_parser("margin")
    m.add_argument("fixture", type=Path)
    m.add_argument("wrfout", type=Path)
    m.add_argument("--out", type=Path, default=None)
    s = sub.add_parser("slim")
    s.add_argument("fixture", type=Path)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--receipt", type=Path, default=None)
    v = sub.add_parser("moved")
    v.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "moved":
        moved(args.out)
        return 0
    if args.command == "door":
        door(args.directory, args.run_seconds, args.out)
        return 0
    if args.command == "reproduce":
        return reproduce(args.fixture, args.receipt)
    if args.command == "margin":
        margin(args.fixture, args.wrfout, args.out)
        return 0
    slim(args.fixture, args.out, args.receipt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
