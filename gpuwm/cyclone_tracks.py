"""Track guidance for setup only. No external path enters the live follower."""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
from pathlib import Path

SCHEMA = "gpuwm.tc-tracks.v1"
CSV_COLUMNS = ("member_id", "track_id", "weight", "valid", "lead_hours", "lat",
               "lon", "mslp_pa", "max_wind10_ms", "flags")
NEAR_TIE_PA = 100.0


def _time(value):
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if moment.utcoffset() != timedelta(0):
            raise ValueError()
        return moment.astimezone(timezone.utc).replace(tzinfo=None)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"track-valid-time: expected UTC timestamp, got {value!r}") from error


def _number(value, field, *, optional=False):
    if optional and value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"track-number: {field} must be finite") from error
    if isinstance(value, bool) or not math.isfinite(result):
        raise ValueError(f"track-number: {field} must be finite")
    return result


def load_tracks(path):
    path = Path(path).expanduser().resolve()
    data = path.read_bytes()
    if path.suffix.lower() == ".csv":
        reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise ValueError("track-csv-columns: expected " + ",".join(CSV_COLUMNS))
        tracks, members = {}, {}
        for row in reader:
            mid, tid = row["member_id"], row["track_id"]
            key = (mid, tid)
            members.setdefault(mid, {"member_id": mid})
            track = tracks.setdefault(key, {"member_id": mid, "track_id": tid,
                                            "weight": _number(row["weight"], "weight"), "fixes": []})
            if track["weight"] != _number(row["weight"], "weight"):
                raise ValueError("track-csv-weight: inconsistent weight within track")
            flags = row["flags"]
            try:
                flags = json.loads(flags) if flags else []
            except ValueError as error:
                raise ValueError("track-csv-flags: expected a JSON array") from error
            track["fixes"].append({k: (flags if k == "flags" else row[k])
                                   for k in CSV_COLUMNS[3:]})
        doc = {"schema": SCHEMA, "tracker": {}, "members": list(members.values()),
               "tracks": list(tracks.values())}
        # The CSV twin has no tracker/member metadata columns. Bind the JSON
        # twin when available, but never pretend standalone CSV supplied them.
        twin = path.with_suffix(".json")
        if twin.exists():
            other, _, _ = load_tracks(twin)
            def rows(tracks):
                return sorted((t["member_id"], t["track_id"], t["weight"],
                               json.dumps(t["fixes"], sort_keys=True)) for t in tracks)
            _validate(doc)
            if rows(doc["tracks"]) != rows(other["tracks"]):
                raise ValueError("track-csv-twin-mismatch: JSON fixes differ from CSV")
            doc["tracker"], doc["members"] = other["tracker"], other["members"]
            doc["metadata_input"] = {"path": str(twin), "sha256": hashlib.sha256(twin.read_bytes()).hexdigest()}
    else:
        try:
            doc = json.loads(data)
        except ValueError as error:
            raise ValueError("track-json: invalid JSON") from error
    _validate(doc)
    return doc, str(path), hashlib.sha256(data).hexdigest()


def _validate(doc):
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise ValueError(f"track-schema: expected {SCHEMA}")
    if not isinstance(doc.get("tracker"), dict) or not isinstance(doc.get("members"), list) or not isinstance(doc.get("tracks"), list):
        raise ValueError("track-schema: tracker, members and tracks are required")
    mids = [m.get("member_id") for m in doc["members"]]
    if any(not isinstance(m, str) or not m for m in mids) or len(set(mids)) != len(mids):
        raise ValueError("track-member-identity: member ids must be unique nonempty strings")
    seen = set()
    for track in doc["tracks"]:
        mid, tid = track.get("member_id"), track.get("track_id")
        if mid not in mids or not isinstance(tid, str) or not tid or (mid, tid) in seen:
            raise ValueError("track-identity: unknown member or duplicate/empty track id")
        seen.add((mid, tid))
        track["weight"] = _number(track.get("weight"), "weight")
        if not isinstance(track.get("fixes"), list):
            raise ValueError("track-fixes: expected a list")
        previous = None
        for fix in track["fixes"]:
            if fix is None:
                continue
            if not isinstance(fix, dict):
                raise ValueError("track-fix: expected an object or null")
            moment = _time(fix.get("valid"))
            if previous is not None and moment <= previous:
                raise ValueError("track-fix-order: valid times must strictly increase")
            previous = moment
            for field in ("lead_hours", "lat", "lon", "mslp_pa", "max_wind10_ms"):
                fix[field] = _number(fix.get(field), field, optional=field != "lead_hours")
            if (fix["lat"] is None) != (fix["lon"] is None):
                raise ValueError("track-position: latitude and longitude must both be present or missing")
            if fix["lat"] is not None and not (-90 <= fix["lat"] <= 90 and -180 <= fix["lon"] <= 180):
                raise ValueError("track-position: latitude/longitude outside signed degree ranges")
            if fix["mslp_pa"] is not None and fix["mslp_pa"] <= 0:
                raise ValueError("track-pressure: mslp_pa must be positive")
            if not isinstance(fix.get("flags"), list):
                raise ValueError("track-flags: expected an array")


def fix_at(fixes, moment):
    """Exact fix or short great-circle interpolation; never bridge null fixes."""
    previous = None
    for fix in fixes:
        if fix is None or fix["lat"] is None:
            previous = None
            continue
        valid = _time(fix["valid"])
        if valid == moment:
            return dict(fix), {"method": "exact", "valid": [fix["valid"]]}
        if previous is not None and _time(previous["valid"]) < moment < valid:
            fraction = (moment - _time(previous["valid"])).total_seconds() / (valid - _time(previous["valid"])).total_seconds()
            def vector(f):
                lat, lon = math.radians(f["lat"]), math.radians(f["lon"])
                return (math.cos(lat)*math.cos(lon), math.cos(lat)*math.sin(lon), math.sin(lat))
            a, b = vector(previous), vector(fix)
            angle = math.acos(max(-1., min(1., sum(x*y for x, y in zip(a, b)))))
            if math.pi - angle < 1e-8:
                raise ValueError("track-antipodal-fixes: great-circle interpolation is ambiguous")
            weights = ((1.-fraction, fraction) if angle < 1e-8 else
                       (math.sin((1.-fraction)*angle)/math.sin(angle), math.sin(fraction*angle)/math.sin(angle)))
            x, y, z = [weights[0]*p + weights[1]*q for p, q in zip(a, b)]
            result = {"valid": moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
                      "lat": math.degrees(math.atan2(z, math.hypot(x, y))),
                      "lon": (math.degrees(math.atan2(y, x))+180.) % 360.-180.,
                      "flags": list(dict.fromkeys(previous["flags"] + fix["flags"]))}
            for field in ("lead_hours", "mslp_pa", "max_wind10_ms"):
                p, q = previous[field], fix[field]
                result[field] = None if p is None or q is None else p + fraction*(q-p)
            return result, {"method": "great-circle", "valid": [previous["valid"], fix["valid"]], "fraction": fraction}
        previous = fix
    return None, {}


def select_track(path, *, storm, member, source, start, hours, search_area=None):
    from gpuwm.cyclone_sources import source_adapter, validate_center
    from gpuwm.source_coverage import points_outside

    doc, filename, digest = load_tracks(path)
    mids = [m["member_id"] for m in doc["members"]]
    if member is None:
        if len(mids) != 1:
            raise ValueError("track-member-required: select --member from " + ", ".join(mids))
        member = mids[0]
    if member not in mids:
        raise ValueError(f"track-member-not-found: {member!r} is not in the tracks file")
    end = start + timedelta(hours=hours)
    window = source_adapter(source).coverage_window
    from gpuwm.cyclone_search import window as search_window
    search = search_window(search_area) if search_area is not None else None
    ranked, rejected = [], []
    for track in doc["tracks"]:
        if track["member_id"] != member:
            continue
        center, interpolation = fix_at(track["fixes"], start)
        fixes = [f for f in track["fixes"] if f is not None and f["lat"] is not None
                 and start <= _time(f["valid"]) <= end]
        last, _ = fix_at(track["fixes"], end)
        if center is not None and not any(f["valid"] == center["valid"] for f in fixes):
            fixes.insert(0, center)
        if last is not None and not any(f["valid"] == last["valid"] for f in fixes):
            fixes.append(last)
        reason = None
        if center is None:
            reason = "track-missing-start-fix"
        elif search is not None and bool(points_outside(search, center["lat"], center["lon"])):
            reason = "track-outside-search-area"
        elif any(bool(points_outside(window, f["lat"], f["lon"])) for f in fixes):
            reason = "track-outside-source-coverage"
        pressures = [f["mslp_pa"] for f in fixes if f["mslp_pa"] is not None]
        cycle_pressure = center.get("mslp_pa") if center else None
        if reason is None and cycle_pressure is None and storm == "auto":
            reason = "track-no-cycle-pressure"
        if reason:
            rejected.append({"track_id": track["track_id"], "reason": reason})
            continue
        validate_center(source, (center["lat"], center["lon"]))
        # Identity labels belong to the generic track data, never a model or
        # case branch. Unlabelled tracks remain selectable by their track id.
        aliases = {str(track["track_id"]).casefold()}
        for key in ("atcf_id", "storm_id", "name", "storm_name"):
            if track.get(key):
                aliases.add(str(track[key]).casefold())
        aliases.update(str(value).casefold() for value in track.get("aliases", []))
        ranked.append({"track_id": track["track_id"], "minimum_mslp_pa": min(pressures) if pressures else None,
                       "cycle_mslp_pa": cycle_pressure, "aliases": aliases,
                       "center_fix": center, "interpolation": interpolation, "fixes": fixes})
    ranked.sort(key=lambda t: (t["cycle_mslp_pa"] is None, t["cycle_mslp_pa"] or 0., t["track_id"]))
    catalogue_identity = None
    if storm == "auto":
        if not ranked:
            raise ValueError("track-no-eligible-system: " + json.dumps(rejected))
        chosen = ranked[0]
    else:
        matches = [t for t in ranked if storm.casefold() in t["aliases"]]
        if len(matches) > 1:
            raise ValueError(f"track-storm-ambiguous: {storm!r}; select a unique track id")
        chosen = matches[0] if matches else None
        if chosen is None:
            rejected_reason = next((r["reason"] for r in rejected if r["track_id"].casefold() == storm.casefold()), None)
            if rejected_reason:
                raise ValueError(f"{rejected_reason}: {storm!r} for member {member!r}")
        if chosen is None and ranked:
            from gpuwm.cyclone_catalog import match_explicit
            matched = match_explicit(storm, start, ranked)
            if matched is not None:
                chosen, catalogue_identity = matched
        if chosen is None:
            reason = next((r["reason"] for r in rejected if r["track_id"] == storm), "track-id-not-found")
            raise ValueError(f"{reason}: {storm!r} for member {member!r}")
    competitors = [t for t in ranked if t is not chosen]
    def summary(t):
        return {k: t[k] for k in ("track_id", "cycle_mslp_pa", "minimum_mslp_pa") if t[k] is not None}
    score = chosen["cycle_mslp_pa"]
    metadata = next(m for m in doc["members"] if m["member_id"] == member)
    lineage = {"tracks_file": filename, "tracks_sha256": digest,
               "tracker": doc["tracker"], "member": metadata, "member_id": member,
               "track_id": chosen["track_id"], "center_fix": chosen["center_fix"],
               "interpolation": chosen["interpolation"],
               "selection_rule": ("lowest-cycle-mslp-pa; track-id lexical tie break" if storm == "auto" else "explicit-track-id"),
               "window_start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
               "window_end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
               "runner_up": summary(competitors[0]) if competitors else {},
               "near_tie_pa": NEAR_TIE_PA,
               "near_ties": [summary(t) for t in competitors if score is not None and t["cycle_mslp_pa"] is not None
                             and abs(t["cycle_mslp_pa"]-score) <= NEAR_TIE_PA],
               "rejected": rejected,
               "metadata_status": "json" if doc["tracker"] else "not-supplied-by-standalone-csv"}
    if catalogue_identity is not None:
        lineage["catalogue_identity"] = catalogue_identity
        lineage["selection_rule"] = "explicit-catalogue-cycle-position-match"
    if search is not None:
        lineage["search_area"] = list(search_area)
        lineage["search_area_basis"] = "position at window_start; source coverage also enforced"
    if "metadata_input" in doc:
        lineage["metadata_input"] = doc["metadata_input"]
    return {"lineage": lineage, "fixes": chosen["fixes"]}


def toml_lineage(value):
    """TOML has no null; retain missing values in the JSON receipt only."""
    if isinstance(value, dict):
        return {k: toml_lineage(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [toml_lineage(v) for v in value if v is not None]
    return value


def validate_lineage(table):
    """This table records setup provenance and has no integration controls."""
    if not isinstance(table, dict) or set(table) != {"cyclone_tracks"}:
        raise ValueError("lineage: expected the cyclone_tracks provenance block")
    row = table["cyclone_tracks"]
    if not isinstance(row, dict) or not {"tracks_file", "tracks_sha256", "tracker", "member_id", "track_id", "center_fix", "selection_rule"} <= row.keys():
        raise ValueError("lineage: incomplete cyclone track provenance")


def layout(*, point, fixes, dimensions, dx_m, clearance):
    """Place a projected parent around the path and reserve the full follower."""
    from gpuwm import domain_wizard as dw
    from gpuwm.cyclone_setup import RATIO
    import numpy as np

    parent, child = dimensions
    projection = dw._projection_entries(*point, "auto")
    grid = dw._root_grid(projection, *parent, dx_m)
    lat = np.array([f["lat"] for f in fixes])
    lon = np.array([f["lon"] for f in fixes])
    x, y = grid.latlon_to_ij(lat, lon)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("track-projection: path cannot be represented on the start projection")
    # One extra cell covers the initial nest's whole-parent-cell registration.
    padding = [child[i]/RATIO/2. + clearance + 1 for i in range(2)]
    required = tuple(max(parent[i], 2*math.ceil((float(np.ptp(axis))+2*padding[i])/2.))
                     for i, axis in enumerate((x, y)))
    center = grid.ij_to_latlon((float(x.min())+float(x.max()))/2.,
                               (float(y.min())+float(y.max()))/2.)
    projection.update(ref_lat=float(center[0]), ref_lon=float(center[1]))
    shifted = dw._root_grid(projection, *parent, dx_m)
    sx, sy = shifted.latlon_to_ij(lat, lon)
    fits = all(bool(np.all((axis >= .5+padding[i]) & (axis <= parent[i]+.5-padding[i])))
               for i, axis in enumerate((sx, sy)))
    if not fits:
        projection = dw._projection_entries(*point, "auto")
    return projection, required


def placement_report(text, selection):
    from gpuwm import domain_wizard as dw
    from gpuwm import cyclone_setup as tc
    import tomllib

    raw = tomllib.loads(text)
    exp = dw.experiment_from_text(text, source="cyclone-track-placement.toml")
    parent, child = exp.domains
    from gpuwm.static.projection import grids_from_projection_config
    child_grid = grids_from_projection_config(exp)[1]
    actual_seed = child_grid.ij_to_latlon(child_grid.e_we/2., child_grid.e_sn/2.)
    grid = dw._root_grid(raw["projection"], parent.run.nx, parent.run.ny, parent.run.dx)
    clearance = tc._nest_clearance_cells(exp, (child.run.nx, child.run.ny))
    padding = [child.run.nx/tc.RATIO/2.+clearance, child.run.ny/tc.RATIO/2.+clearance]
    checked = []
    for fix in selection["fixes"]:
        x, y = (float(v) for v in grid.latlon_to_ij(fix["lat"], fix["lon"]))
        inside = .5 <= x <= parent.run.nx+.5 and .5 <= y <= parent.run.ny+.5
        reserved = (.5+padding[0] <= x <= parent.run.nx+.5-padding[0]
                    and .5+padding[1] <= y <= parent.run.ny+.5-padding[1])
        checked.append({"valid": fix["valid"], "lat": fix["lat"], "lon": fix["lon"],
                        "parent_ij": [x, y], "inside_parent": inside,
                        "follower_clearance": reserved})
    return {"initial_nest_center": [float(v) for v in actual_seed],
            "seed_registration": "nearest whole parent cell to center_fix",
            "parent_center": [raw["projection"]["ref_lat"], raw["projection"]["ref_lon"]],
            "clearance_parent_cells": clearance, "fixes": checked,
            "all_fixes_inside_parent": all(f["inside_parent"] for f in checked),
            "all_fixes_with_follower_clearance": all(f["follower_clearance"] for f in checked),
            "leaves_parent": [f for f in checked if not f["inside_parent"]],
            "leaves_follower_clearance": [f for f in checked if not f["follower_clearance"]],
            "policy": "path envelope plus follower search, movement and boundary/blend clearance; start-centered fallback when it does not fit"}
