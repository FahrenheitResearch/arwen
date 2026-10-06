#!/usr/bin/env python3
"""Build pinned regional rain input manifests from saved WRF frames or MRMS packs."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def utc(value):
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00").replace("_", "T"))
    return stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp.astimezone(timezone.utc)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def frames(paths, variable, **options):
    return {"paths": [str(p.resolve()) for p in paths], "variable": variable, **options}


def model(args):
    from gpuwm import netcdf_bridge
    origin = utc(args.analysis_end)
    selected, shape, echo_name = [], None, None
    for path in sorted(args.frames.glob(args.pattern)):
        if not path.is_file():
            continue
        with netcdf_bridge.open_dataset(path) as ds:
            absent = [name for name in ("Times", "RAINC", "RAINNC", "XLAT", "XLONG") if name not in ds.variables]
            if absent:
                raise ValueError(f"{path}: missing native scoring fields {absent}; reflectivity-only archives cannot provide rain")
            strings = [row.tobytes().decode("ascii").rstrip("\x00") for row in ds.variables["Times"][:]]
            if len(strings) != 1:
                raise ValueError(f"{path}: model manifest builder needs one Time frame per file")
            stamp = utc(strings[0])
            seconds = (stamp-origin).total_seconds()
            if seconds < -1e-6 or seconds > 21600+1e-6:
                continue
            current = tuple(ds.variables["RAINNC"].shape[-2:])
            if shape is not None and current != shape:
                raise ValueError("saved model geometry changes during free forecast")
            shape = current
            field = "REFL_COMPOSITE" if "REFL_COMPOSITE" in ds.variables else "REFL_10CM"
            if field not in ds.variables:
                raise ValueError(f"{path}: no native composite reflectivity")
            if echo_name is not None and echo_name != field:
                raise ValueError("saved reflectivity variable changes between frames")
            echo_name = field
            reset = getattr(ds, "GPUWM_RAIN_RESET_ID", None)
            if reset is None and not args.assert_no_resets:
                raise ValueError(f"{path}: no rain reset identity; --assert-no-resets requires a documented non-resetting lineage")
            reset = 0 if reset is None else int(reset)
            selected.append((seconds, stamp.isoformat().replace("+00:00", "Z"), path.resolve(), reset))
    selected.sort(key=lambda v: v[0])
    if not selected:
        raise ValueError("no selected free-forecast frames")
    # At the forecast fork the producer can retain both the pre-analysis
    # leg-end frame and the explicit free-forecast issuance frame. The
    # issuance carries applied pending increments and owns the zero-hour
    # forecast state. It is selected only at the exact analysis clock.
    groups = {}
    for entry in selected:
        groups.setdefault(entry[0], []).append(entry)
    resolved, choices = [], []
    for seconds, entries in sorted(groups.items()):
        if len(entries) > 1:
            issuance = [e for e in entries if e[2].stem.startswith("wrfout_start")]
            if abs(seconds) > 1e-6 or len(issuance) != 1:
                raise ValueError("duplicate saved valid times; narrow --pattern to one member and domain")
            resolved.append(issuance[0])
            choices.append({"time_seconds": seconds, "selected": str(issuance[0][2]),
                            "omitted": [str(e[2]) for e in entries if e is not issuance[0]],
                            "reason": "explicit free-forecast issuance supersedes the pre-analysis frame at the fork"})
        else:
            resolved.append(entries[0])
    selected = resolved
    paths = [v[2] for v in selected]
    if len({v[3] for v in selected}) != 1:
        raise ValueError("resetting archives need a manually pinned reset_carry_mm manifest")
    first = paths[0]
    echo_options = {"reduction": "max_z"} if echo_name == "REFL_10CM" else {}
    mask = {"constant": 1, "shape": [len(paths), *shape]}
    return {"schema": "regional-rain/input.v1", "analysis_end": args.analysis_end,
            "times_utc": [v[1] for v in selected],
            "grid": {"wrf_projection": {"path": str(first)}, "equal_area_center": [args.center_lon, args.center_lat]},
            "rain_accum_mm": {"sum": [frames(paths, "RAINC", units="mm"), frames(paths, "RAINNC", units="mm")]},
            "rain_valid": mask, "reset_ids": [v[3] for v in selected],
            "echo_dbz": frames(paths, echo_name, **echo_options), "echo_valid": mask,
            "rain_reset_policy": "native GPUWM_RAIN_RESET_ID" if not args.assert_no_resets else "explicit non-resetting lineage assertion",
            "native_geometry_sha256": sha(first), "frame_selection": choices, "archive_grade": "archive-rich-research"}


def packs(folder, product, quantity, units):
    from gpuwm.obs.obspack import read_pack, GRID_SCHEMA
    selected = []
    for path in sorted(folder.glob("*.obspack")):
        pack = read_pack(path)
        if pack.schema != GRID_SCHEMA:
            continue
        meta = pack.meta
        p = meta.get("provenance", {}).get("product")
        # Native product metadata owns quantity and units, not filenames.
        if meta.get("quantity") != quantity or meta.get("units") != units:
            raise ValueError(f"{path}: expected {quantity} {units}, got {meta.get('quantity')} {meta.get('units')}")
        if p is not None and p != product:
            raise ValueError(f"{path}: wrong native MRMS product {p!r}")
        selected.append((utc(meta["valid_time"]), path.resolve(), pack.array("values").shape, meta.get("grid")))
    selected.sort(key=lambda v: v[0])
    if not selected or len({v[0] for v in selected}) != len(selected):
        raise ValueError(f"{folder}: no packs or duplicate native product times")
    return selected


def truth(args):
    rain = packs(args.rate_packs, "PrecipRate_00.00", "precipitation_rate", "mm/hr")
    echo = packs(args.echo_packs, "MergedReflectivityQCComposite_00.50", "composite_reflectivity", "dBZ")
    grid = rain[0][3]
    if any(v[3] != grid or v[2] != rain[0][2] for v in (*rain, *echo)):
        raise ValueError("MRMS products do not share the identical pinned bbox/grid")
    rp, ep = [v[1] for v in rain], [v[1] for v in echo]
    return {"schema": "regional-rain/input.v1", "analysis_end": args.analysis_end,
            "times_utc": [v[0].isoformat().replace("+00:00", "Z") for v in rain],
            "echo_times_utc": [v[0].isoformat().replace("+00:00", "Z") for v in echo],
            "grid": {"latitude": {"path": str(args.geo_pack.resolve()), "variable": "latitude"},
                     "longitude": {"path": str(args.geo_pack.resolve()), "variable": "longitude"},
                     "lat_lon_regular": True, "equal_area_center": [args.center_lon, args.center_lat]},
            "rate_mm_h": frames(rp, "values", units="mm/hr"), "rain_valid": frames(rp, "valid"),
            "echo_dbz": frames(ep, "values", units="dBZ"), "echo_valid": frames(ep, "valid"),
            "truth_revision": "pinned native MRMS PrecipRate and QC composite archives",
            "native_quality_policy": "product sentinel/bitmap masks; no replacement of missing cells by dry values",
            "archive_grade": "archive-rich-research"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("model", "truth"))
    parser.add_argument("--analysis-end", required=True)
    parser.add_argument("--center-lon", type=float, required=True)
    parser.add_argument("--center-lat", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=Path)
    parser.add_argument("--pattern", default="wrfout_*.nc", help="select exactly one member/domain")
    parser.add_argument("--assert-no-resets", action="store_true")
    parser.add_argument("--rate-packs", type=Path)
    parser.add_argument("--echo-packs", type=Path)
    parser.add_argument("--geo-pack", type=Path)
    args = parser.parse_args(argv)
    required = ("frames",) if args.kind == "model" else ("rate_packs", "echo_packs", "geo_pack")
    if any(getattr(args, name) is None for name in required):
        parser.error(f"{args.kind} requires " + ", ".join("--"+name.replace("_", "-") for name in required))
    try:
        manifest = model(args) if args.kind == "model" else truth(args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n", encoding="utf-8")
        print(json.dumps({"manifest": str(args.out), "rain_frames": len(manifest["times_utc"]),
                          "echo_frames": len(manifest.get("echo_times_utc", manifest["times_utc"]))}, sort_keys=True))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print(f"regional rain manifest refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
