"""Write genuine constant GRIB2 fields for the production prep CLI replay.

The packaged source grids and native level ladders are retained exactly.
Only field values are synthetic. Constant simple packing uses zero bits,
so generation never allocates a native-grid array. The real Rust decoder
still validates and expands every record that the production mapping uses.
Section layouts follow tools/rustwx/vendor/wx-core/src/grib2/writer.rs.
This script is a fixture author, never a replacement decoder or transform.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import struct


def u32(value):
    return struct.pack(">I", int(value))


def signed32(value):
    number = int(round(abs(value) * 1_000_000))
    return u32(number | (0x80000000 if value < 0 else 0))


def section(number, payload):
    return u32(5 + len(payload)) + bytes([number]) + payload


def grid_section(parameters):
    p = parameters
    data = bytes([0]) + u32(p["nx"] * p["ny"]) + bytes([0, 0]) + struct.pack(">H", 30)
    data += bytes([p["shape_of_earth"], 0]) + u32(0) + bytes([0]) + u32(0) + bytes([0]) + u32(0)
    data += u32(p["nx"]) + u32(p["ny"])
    data += signed32(p["lat1"]) + signed32(p["lon1"] % 360) + bytes([0x38])
    data += signed32(p["latin1"]) + signed32(p["lov"] % 360)
    data += u32(round(p["dx_m"] * 1000)) + u32(round(p["dy_m"] * 1000))
    data += bytes([0, 0x40]) + signed32(p["latin1"]) + signed32(p["latin2"])
    data += signed32(-90) + u32(0)
    result = section(3, data)
    assert len(result) == 81
    return result


def level_octets(value):
    for scale in range(10):
        number = value * 10 ** scale
        if abs(number - round(number)) < 1e-6:
            return bytes([scale]) + u32(round(number))
    raise ValueError(f"cannot encode level {value}")


def message(parameters, selector, value, cycle, lead):
    # WMO FM 92 sections 1, 3, 4.0, 5.0, 6 and 7.
    ident = struct.pack(">HHBBB HBBBBBB", selector.get("center", 7),
        selector.get("subcenter", 0), selector.get("master_table_version", 2),
        selector.get("local_table_version", 1), 1, cycle.year,
        cycle.month, cycle.day, cycle.hour, cycle.minute, cycle.second, 0) + bytes([1])
    sec1 = section(1, ident)
    assert len(sec1) == 21
    product = struct.pack(">HH", 0, 0)
    product += bytes([selector["category"], selector["parameter"], 2, 0, 0])
    product += struct.pack(">HBBI", 0, 0, 1, lead)
    product += bytes([selector["level_type"]]) + level_octets(selector.get("level_value", 0))
    if "second_level_type" in selector:
        product += bytes([selector["second_level_type"]]) + level_octets(selector["second_level_value"])
    else:
        product += bytes([255]) * 6
    sec4 = section(4, product)
    assert len(sec4) == 34
    sec5 = section(5, u32(parameters["nx"] * parameters["ny"]) + struct.pack(">HfHHBB", 0, value, 0, 0, 0, 0))
    assert len(sec5) == 21
    body = sec1 + grid_section(parameters) + sec4 + sec5 + section(6, bytes([255])) + section(7, b"") + b"7777"
    return b"GRIB" + bytes([0, 0, selector["discipline"], 2]) + struct.pack(">Q", 16 + len(body)) + body


def values(name, level, mapping):
    vertical = mapping["coordinates"]["vertical"]
    if vertical["kind"] == "model_level":
        fraction = (level - 1) / (len(vertical["levels"]) - 1) if level else 0.0
        top = vertical["model_top_pressure_pa"]
        pressure = 99_000.0 * math.exp(math.log(top / 99_000.0) * fraction)
    else:
        pressure = level * 100 if level else 99_000
    height = 287.0 * 280.0 / 9.81 * math.log(100_000.0 / pressure)
    if name == "air_pressure":
        return pressure
    if name == "geopotential_height":
        return height
    if name == "air_temperature":
        return max(210, 290 - 0.005 * height)
    if name in ("specific_humidity", "specific_humidity_2m"):
        return 0.001
    if name in ("eastward_wind", "eastward_wind_10m"):
        return 5.0
    if name in ("northward_wind", "northward_wind_10m"):
        return 2.0
    if name == "surface_pressure":
        return 100_000.0
    if name == "terrain_height":
        return 0.0
    if name in ("skin_temperature", "air_temperature_2m", "soil_temperature"):
        return 290.0
    if name == "volumetric_soil_moisture":
        return 0.3
    if name == "land_fraction":
        return 1.0
    if name == "vegetation_fraction":
        return 60.0
    if name == "water_friendly_aerosol_number":
        return 100_000_000.0
    if name == "ice_friendly_aerosol_number":
        return 1_000_000.0
    return 0.0


def write_fields(path, mapping, cycle, lead, only=None):
    count = 0
    field_records = {}
    with path.open("wb") as stream:
        for name, spec in mapping["fields"].items():
            if only is not None and name not in only:
                continue
            selectors = spec.get("selectors", [])
            if not selectors:
                continue
            # Multiple volume selectors are alternatives, while soil selectors
            # are distinct depth nodes required by the composition contract.
            use = selectors[:1] if "vertical" in spec.get("source_axes", []) else selectors
            written = 0
            for selector in use:
                levels = mapping["coordinates"]["vertical"]["levels"] if "vertical" in spec.get("source_axes", []) else [None]
                for level in levels:
                    selector = dict(selector)
                    if level is not None:
                        selector["level_value"] = level
                    val = values(name, level, mapping)
                    stream.write(message(mapping["grid"]["parameters"], selector, val, cycle, lead))
                    count += 1
                    written += 1
            field_records[name] = written
    data = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "messages": count, "field_records": field_records,
            "grid": mapping["grid"]["parameters"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", default="2026-10-03T18:00:00")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cycle = datetime.fromisoformat(args.start)
    authority = args.engine / "gpuwm" / "authorities"
    load = lambda name: json.loads((authority / f"rw-wps-{name}-grib2.mapping.json").read_text())
    records = []
    for lead in (0, 3):
        records.append(write_fields(args.output / f"rap-native-f{lead:02}.grib2", load("rap-native"), cycle, lead))
    records.append(write_fields(args.output / "hrrr-native-f00.grib2", load("hrrr-native"), cycle, 0))
    records.append(write_fields(args.output / "hrrr-soil-f00.grib2", load("hrrr-prs"), cycle, 0,
                                {"terrain_height", "soil_temperature", "volumetric_soil_moisture"}))
    records.append(write_fields(args.output / "hrrr-vegetation-f00.grib2", load("hrrr-surface-vegetation"), cycle, 0,
                                {"vegetation_fraction"}))
    request = {"schema": "gpuwm-initial-source-v1", "source": "hrrr-native",
               "input_files": ["hrrr-native-f00.grib2"],
               "supplements": {"soil_surface_data": ["hrrr-soil-f00.grib2"],
                               "vegetation_surface_data": ["hrrr-vegetation-f00.grib2"]}}
    (args.output / "initial-inputs.json").write_text(json.dumps(request, indent=2) + "\n")
    receipt = {"schema": "static-prep-real-cli-synthetic-grib-fixture-v1", "start": args.start,
               "synthetic_values": True, "production_source_grids": True,
               "decode_and_transform_mocked": False, "records": records}
    (args.output / "input-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "input_bytes": sum(r["bytes"] for r in records),
                      "messages": sum(r["messages"] for r in records)}))


if __name__ == "__main__":
    main()
