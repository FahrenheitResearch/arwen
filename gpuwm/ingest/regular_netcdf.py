"""Bind native regular-grid NetCDF fields to the shared forcing snapshots.

The file's explicit schema, units and missing-mask variables are authoritative.
All NetCDF and CF-time decoding is performed by the existing Rust bridge.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from gpuwm.netcdf_bridge import Dataset
from gpuwm.ingest.grib import Era5DecodeResult, Era5Snapshot
from gpuwm.zarr_bridge import regular_netcdf_record
from gpuwm.ingest.lake_temperature import LAKE_FIELD_UNITS, source_lake_fields

REGULAR_SCHEMA = "arwen.regular-forcing.v1"
REGULAR_FIELD_UNITS = {
    "Z": "m2 s-2", "T": "K", "U": "m s-1", "V": "m s-1",
    "SPFH": "kg kg-1", "PRES": "Pa", "Q2": "kg kg-1",
    "U10": "m s-1", "V10": "m s-1", "T2": "K", "D2": "K",
    "LANDSEA": "1", "SOILGEO": "m2 s-2", "PSFC": "Pa", "PMSL": "Pa",
    "SKINTEMP": "K", "SEAICE": "1", "SST": "K", "SNOW_EC": "m",
    "ST000007": "K", "ST007028": "K", "ST028100": "K", "ST100289": "K",
    "SM000007": "m3 m-3", "SM007028": "m3 m-3",
    "SM028100": "m3 m-3", "SM100289": "m3 m-3",
    **LAKE_FIELD_UNITS,
}


class RegularNetcdfSnapshot(Era5Snapshot):
    """Explicit direct specific-humidity authority for native schema files."""

    specific_humidity_authority = True
    analyzed_species = ()


def _schema(dataset):
    if getattr(dataset, "arwen_regular_forcing_schema", None) != REGULAR_SCHEMA:
        raise ValueError(f"{dataset.path} lacks the declared regular forcing NetCDF schema")
    for name, units in (("time", "seconds since 1970-01-01 00:00:00"),
                        ("level", "hPa"), ("latitude", "degrees_north"),
                        ("longitude", "degrees_east")):
        var = dataset.variables.get(name)
        if var is None or var.dimensions != (name,) or getattr(var, "units", None) != units:
            raise ValueError(f"{dataset.path} has invalid {name} coordinate metadata")


def inspect_regular_netcdf_times(paths) -> tuple[datetime, ...]:
    """Read only the file metadata and native-decoded CF time coordinate."""
    times = []
    for path in paths:
        with Dataset(path) as dataset:
            _schema(dataset)
            times.extend(dataset.variables["time"].times())
    if len(times) != len(set(times)):
        raise ValueError("regular forcing files contain duplicate valid times")
    return tuple(sorted(times))


def decode_regular_netcdf_files(paths, *, valid_times=None,
                                excluded_valid_times=()) -> Era5DecodeResult:
    """Decode declared files without changing the catalog's selected times.

    Actual file paths may be extensionless content-addressed snapshots. Schema
    verification, not suffix, establishes their native container binding here.
    """
    selection = None if valid_times is None else tuple(valid_times)
    exclusions = set(excluded_valid_times or ())
    if selection is not None:
        if len(set(selection)) != len(selection) or selection != tuple(sorted(selection)):
            raise ValueError("regular forcing selection must be unique and increasing")
        if set(selection) & exclusions:
            raise ValueError("regular forcing selected and excluded valid times overlap")
    snapshots = {}
    sources = {}
    missing = {}
    for raw_path in paths:
        path = Path(raw_path).resolve()
        with Dataset(path) as dataset:
            _schema(dataset)
            times = dataset.variables["time"].times()
            if len(times) != len(set(times)) or times != tuple(sorted(times)):
                raise ValueError(f"{path} has repeated or unordered valid times")
            if getattr(dataset, "specific_humidity_authority", None) != "direct":
                raise ValueError(f"{path} does not declare its specific-humidity authority")
            fields = {}
            for name, var in dataset.variables.items():
                semantic = getattr(var, "arwen_field", None)
                if semantic is None:
                    continue
                if semantic != name or name not in REGULAR_FIELD_UNITS:
                    raise ValueError(f"{path} has unknown or inconsistent field binding {name}")
                if getattr(var, "units", None) != REGULAR_FIELD_UNITS[name]:
                    raise ValueError(f"{path} field {name} has unexpected units")
                if any(key in var.attributes for key in ("scale_factor", "add_offset", "_FillValue", "missing_value")):
                    raise ValueError(f"{path} regular forcing must use unpacked values and explicit native masks")
                if var.dimensions not in (("time", "latitude", "longitude"),
                                          ("time", "level", "latitude", "longitude")):
                    raise ValueError(f"{path} field {name} has incompatible dimensions")
                mask = dataset.variables.get(getattr(var, "arwen_missing_mask", ""))
                if mask is None or mask.dimensions != var.dimensions or mask.shape != var.shape:
                    raise ValueError(f"{path} field {name} lacks its native missing-value mask")
                fields[name] = (var, mask)
            if not {"PRES", "SPFH", "Q2"}.issubset(fields) or "RH" in fields:
                raise ValueError(f"{path} direct humidity requires PRES, SPFH and Q2")
            levels = dataset.variables["level"][:]
            latitude = dataset.variables["latitude"][:]
            longitude = dataset.variables["longitude"][:]
            for index, valid_time in enumerate(times):
                if valid_time in exclusions or (selection is not None and valid_time not in selection):
                    continue
                if valid_time in snapshots:
                    raise ValueError(f"duplicate regular forcing snapshot at {valid_time}")

                requested = [var.name for pair in fields.values() for var in pair]
                with regular_netcdf_record(path, index, requested) as record:
                    def array(variable):
                        payload, shape = record[variable.name]
                        if shape != variable.shape[1:]:
                            raise ValueError(f"{path} field {variable.name} has an inconsistent record shape")
                        values = np.fromfile(payload, dtype="<f8")
                        return values.reshape(shape)

                    def items():
                        for name, (variable, mask) in fields.items():
                            # Raw views of native-decoded buffers: no Python
                            # physical conversions or mask reconstruction.
                            native_mask = array(mask)
                            if not np.isin(native_mask, (0.0, 1.0)).all():
                                raise ValueError(f"{path} field {name} has invalid mask values")
                            missing[(valid_time, name)] = native_mask.astype(bool)
                            values = array(variable)
                            sources[(valid_time, name)] = (path,)
                            yield name, values

                    snapshots[valid_time] = RegularNetcdfSnapshot.from_field_items(
                        valid_time=valid_time, levels_hpa=levels,
                        latitude=latitude, longitude=longitude, field_items=items())
                    source_lake_fields(snapshots[valid_time].fields)
    ordered = tuple(sorted(snapshots))
    if selection is not None and ordered != selection:
        raise ValueError(f"regular forcing decode does not match catalog selection: {ordered}")
    if not ordered:
        raise ValueError("regular forcing selection contains no snapshots")
    inventories = {tuple(sorted(snapshot.fields)) for snapshot in snapshots.values()}
    if len(inventories) != 1:
        raise ValueError("regular forcing field inventory changes between valid times")
    return Era5DecodeResult(tuple(snapshots[t] for t in ordered), sources, missing)
