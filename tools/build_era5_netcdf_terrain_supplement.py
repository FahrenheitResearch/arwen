"""Build a lossless NetCDF view of a hash-bound ERA5 GRIB1 terrain product."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path

import netCDF4
import numpy as np

from gpuwm.mapped_composition import _partition_mapping
from gpuwm.mapped_source import (
    _array_sha256,
    _decode_grib,
    _sha256,
    load_mapping,
)


RECEIPT_SCHEMA = "rw-wps-era5-netcdf-terrain-conversion-v1"


def _primary_contract(path: Path) -> tuple[datetime, np.ndarray]:
    with netCDF4.Dataset(path) as dataset:
        time = dataset.variables["time"]
        raw = netCDF4.num2date(
            np.asarray(time[:]), time.units,
            calendar=getattr(time, "calendar", "standard"),
            only_use_cftime_datetimes=False,
        )
        values = np.ravel(raw)
        if values.size != 1 or not isinstance(values[0], datetime):
            raise ValueError(f"{path} does not have one standard-calendar time")
        level = dataset.variables["level"]
        if getattr(level, "units", None) != "hPa":
            raise ValueError(f"{path} pressure coordinate is not hPa")
        levels = np.asarray(level[:], dtype=np.float64)
    return values[0], levels


def build(
    source_grib: Path,
    grib1_bridge: Path,
    grib1_mapping: Path,
    primary_files: tuple[Path, ...],
    output: Path,
    receipt_path: Path,
) -> dict[str, object]:
    source_grib = source_grib.resolve()
    grib1_bridge = grib1_bridge.resolve()
    grib1_mapping = grib1_mapping.resolve()
    primary_files = tuple(path.resolve() for path in primary_files)
    output = output.resolve()
    receipt_path = receipt_path.resolve()
    for path in (source_grib, grib1_bridge, grib1_mapping, *primary_files):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not primary_files or len(set(primary_files)) != len(primary_files):
        raise ValueError("primary file inventory must be non-empty and unique")
    if output.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite terrain output or receipt")
    output.parent.mkdir(parents=True, exist_ok=True)

    contracts = tuple(_primary_contract(path) for path in primary_files)
    times = tuple(item[0] for item in contracts)
    if tuple(sorted(times)) != times or len(set(times)) != len(times):
        raise ValueError("primary times must be unique and increasing")
    levels = contracts[0][1]
    if any(not np.array_equal(item[1], levels) for item in contracts[1:]):
        raise ValueError("primary pressure coordinates differ")

    mapping = load_mapping(grib1_mapping)
    if mapping["format"] != "grib1":
        raise ValueError("terrain conversion requires a GRIB1 mapping")
    collection = _decode_grib(
        _partition_mapping(mapping, terrain_only=True),
        (source_grib,), grib1_bridge=grib1_bridge,
        grib2_inventory=None, grib2_dump=None,
    )
    terrain = {
        valid_time: value.values
        for (valid_time, member, field), value in collection.direct.items()
        if member is None and field == "terrain_height"
    }
    missing = [time for time in times if time not in terrain]
    if missing:
        raise ValueError(f"GRIB terrain lacks primary valid times {missing}")
    selected = tuple(np.asarray(terrain[time], dtype=np.float64) for time in times)
    if any(not np.array_equal(selected[0], value) for value in selected[1:]):
        raise ValueError("GRIB terrain is not invariant across primary times")
    # Undo the mapping's geopotential-to-height conversion.  The NetCDF
    # mapping owns and re-applies that exact conversion, preserving one
    # semantic authority for the final canonical field.
    terrain_mapping = mapping["fields"]["terrain_height"]
    scale = float(terrain_mapping["units"].get("scale", 1.0))
    offset = float(terrain_mapping["units"].get("offset", 0.0))
    if scale == 0.0:
        raise ValueError("terrain mapping has a zero unit scale")
    geopotential = np.stack([(value - offset) / scale for value in selected])
    if not np.isfinite(geopotential).all():
        raise ValueError("terrain inverse unit conversion produced non-finite values")

    staging = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        with netCDF4.Dataset(staging, "w", format="NETCDF4") as dataset:
            dataset.createDimension("time", len(times))
            dataset.createDimension("level", len(levels))
            dataset.createDimension("latitude", collection.latitude.size)
            dataset.createDimension("longitude", collection.longitude.size)
            time_variable = dataset.createVariable("time", "f8", ("time",))
            time_variable.standard_name = "time"
            time_variable.units = "hours since 1900-01-01 00:00:00"
            time_variable.calendar = "gregorian"
            time_variable[:] = netCDF4.date2num(
                times, time_variable.units, calendar=time_variable.calendar,
            )
            level_variable = dataset.createVariable("level", "f8", ("level",))
            level_variable.standard_name = "air_pressure"
            level_variable.units = "hPa"
            level_variable[:] = levels
            latitude = dataset.createVariable("latitude", "f8", ("latitude",))
            latitude.standard_name = "latitude"
            latitude.units = "degrees_north"
            latitude[:] = collection.latitude
            longitude = dataset.createVariable("longitude", "f8", ("longitude",))
            longitude.standard_name = "longitude"
            longitude.units = "degrees_east"
            longitude[:] = collection.longitude
            orography = dataset.createVariable(
                "OROG", "f8", ("time", "latitude", "longitude"),
            )
            orography.long_name = "Invariant surface geopotential"
            orography.units = "m**2 s**-2"
            orography[:] = geopotential
            dataset.source_grib_sha256 = _sha256(source_grib)
            dataset.grib1_decoder_sha256 = _sha256(grib1_bridge)
            dataset.grib1_mapping_sha256 = _sha256(grib1_mapping)
            dataset.conversion_schema = RECEIPT_SCHEMA
        os.replace(staging, output)
    except BaseException:
        if staging.exists():
            staging.unlink()
        raise

    with netCDF4.Dataset(output) as dataset:
        output_geopotential = np.asarray(dataset.variables["OROG"][:])
        output_times = netCDF4.num2date(
            np.asarray(dataset.variables["time"][:]),
            dataset.variables["time"].units,
            calendar=dataset.variables["time"].calendar,
            only_use_cftime_datetimes=False,
        )
    if not np.array_equal(output_geopotential, geopotential) \
            or tuple(np.ravel(output_times)) != times:
        output.unlink(missing_ok=True)
        raise ValueError("written terrain NetCDF differs from its source arrays")
    reconverted = output_geopotential * scale + offset
    if not np.array_equal(reconverted, np.stack(selected)):
        output.unlink(missing_ok=True)
        raise ValueError("NetCDF terrain conversion is not bit-exact after remapping")

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "PASS",
        "source_grib": {
            "path": str(source_grib), "bytes": source_grib.stat().st_size,
            "sha256": _sha256(source_grib),
        },
        "grib1_bridge": {
            "path": str(grib1_bridge), "bytes": grib1_bridge.stat().st_size,
            "sha256": _sha256(grib1_bridge),
        },
        "grib1_mapping": {
            "path": str(grib1_mapping), "bytes": grib1_mapping.stat().st_size,
            "sha256": _sha256(grib1_mapping),
        },
        "primary_files": [{
            "path": str(path), "bytes": path.stat().st_size,
            "sha256": _sha256(path), "valid_time": time.isoformat(),
        } for path, time in zip(primary_files, times)],
        "output": {
            "path": str(output), "bytes": output.stat().st_size,
            "sha256": _sha256(output),
        },
        "shape": list(geopotential.shape),
        "valid_times": [time.isoformat() for time in times],
        "invariant_across_valid_times": True,
        "latitude_sha256": _array_sha256(collection.latitude),
        "longitude_sha256": _array_sha256(collection.longitude),
        "height_sha256": _array_sha256(np.stack(selected)),
        "geopotential_sha256": _array_sha256(geopotential),
        "roundtrip_height_bit_exact": True,
    }
    receipt["receipt_content_sha256"] = hashlib.sha256(json.dumps(
        receipt, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    staging_receipt = receipt_path.with_name(
        f".{receipt_path.name}.tmp-{os.getpid()}"
    )
    staging_receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(staging_receipt, receipt_path)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-grib", type=Path, required=True)
    parser.add_argument("--grib1-bridge", type=Path, required=True)
    parser.add_argument("--grib1-mapping", type=Path, required=True)
    parser.add_argument("--primary", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = build(
        args.source_grib, args.grib1_bridge, args.grib1_mapping,
        tuple(args.primary), args.output, args.receipt,
    )
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
