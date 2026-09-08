"""Google ARCO ERA5 provider metadata and acquisition orchestration.

The generic Rust Zarr bridge performs every array read, crop, transform and
NetCDF write. This module defines the provider's existing forcing semantics.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from gpuwm.ingest.regular_netcdf import REGULAR_FIELD_UNITS

ARCO_STORE = ("https://storage.googleapis.com/gcp-public-data-arco-era5/ar/"
              "full_37-1h-0p25deg-chunk-1.zarr-v3")
ARCO_COMBINED_NAME = "era5-combined.nc"
_RECEIPT = "era5-arco-acquisition.json"
_SCHEMA = "arwen.era5-arco-acquisition.v1"
_PRESSURE = {
    "geopotential": "Z", "temperature": "T", "u_component_of_wind": "U",
    "v_component_of_wind": "V", "specific_humidity": "SPFH",
}
_SURFACE = {
    "10m_u_component_of_wind": "U10", "10m_v_component_of_wind": "V10",
    "2m_temperature": "T2", "2m_dewpoint_temperature": "D2",
    "land_sea_mask": "LANDSEA", "geopotential_at_surface": "SOILGEO",
    "surface_pressure": "PSFC", "mean_sea_level_pressure": "PMSL",
    "skin_temperature": "SKINTEMP", "sea_ice_cover": "SEAICE",
    "sea_surface_temperature": "SST", "snow_depth": "SNOW_EC",
    "lake_mix_layer_temperature": "LAKE_WATER_TEMP",
    "lake_ice_temperature": "LAKE_ICE_TEMP",
    "lake_ice_depth": "LAKE_ICE_DEPTH",
    "soil_temperature_level_1": "ST000007", "soil_temperature_level_2": "ST007028",
    "soil_temperature_level_3": "ST028100", "soil_temperature_level_4": "ST100289",
    "volumetric_soil_water_layer_1": "SM000007", "volumetric_soil_water_layer_2": "SM007028",
    "volumetric_soil_water_layer_3": "SM028100", "volumetric_soil_water_layer_4": "SM100289",
}


def _validate(path, *, times, area):
    from gpuwm import fetch
    from gpuwm.ingest.horiz import global_longitude_period_columns
    from gpuwm.ingest.regular_netcdf import decode_regular_netcdf_files

    result = decode_regular_netcdf_files((path,), valid_times=times)
    required = set(_PRESSURE.values()) | set(_SURFACE.values()) | {"PRES", "Q2"}
    for snapshot in result.snapshots:
        if set(snapshot.fields) != required:
            raise ValueError("ARCO output does not contain its complete forcing inventory")
        if set(snapshot.levels_hpa.tolist()) != set(fetch.ERA5_PRESSURE_LEVELS_HPA):
            raise ValueError("ARCO output must contain all 37 declared pressure levels")
        if (min(snapshot.latitude) > area.lat_south or max(snapshot.latitude) < area.lat_north):
            raise ValueError("ARCO output does not cover the requested latitude range")
        west = area.lon_west
        while west < snapshot.longitude[0]:
            west += 360
        if (global_longitude_period_columns(snapshot.longitude) is None
                and (west > snapshot.longitude[-1]
                     or west + area.longitude_span_degrees > snapshot.longitude[-1] + 0.251)):
            raise ValueError("ARCO output does not cover the requested longitude range")
        for name in (*_PRESSURE.values(), "PRES", "Q2", "T2", "D2", "U10", "V10", "PSFC", "PMSL", "SOILGEO", "LANDSEA", "SKINTEMP"):
            if not np.isfinite(snapshot.fields[name]).all():
                raise ValueError(f"ARCO output field {name} contains unavailable values at {snapshot.valid_time}")
    return {"checks": ["exact selected times", "all 37 pressure levels", "complete forcing inventory",
                       "native missing-mask provenance", "coordinate coverage", "required finite fields"],
            "failures": []}


def retrieve_era5_arco(*, cycle: datetime | str, hours: int, area,
                       out: str | Path, cadence: int = 6,
                       force: bool = False, progress=print) -> Path:
    """Acquire and publish one native NetCDF forcing file plus its receipt."""
    from gpuwm import fetch, fetch_guard
    from gpuwm.zarr_bridge import extract_regular_zarr

    if isinstance(cycle, str):
        cycle = fetch.parse_cycle(cycle, "era5")
    if not isinstance(cycle, datetime):
        raise ValueError("ERA5 cycle must be an explicit UTC date and hour")
    if cycle.tzinfo is not None:
        cycle = cycle.astimezone(timezone.utc).replace(tzinfo=None)
    if cycle.minute or cycle.second or cycle.microsecond:
        raise ValueError("ERA5 cycle must fall on an exact UTC hour")
    if isinstance(hours, bool) or not isinstance(hours, int) or hours <= 0:
        raise ValueError("ERA5 hours must be a positive integer")
    if isinstance(cadence, bool) or not isinstance(cadence, int) or cadence not in (1, 3, 6):
        raise ValueError("ERA5 cadence must be 1, 3 or 6 hours")
    if isinstance(area, str):
        area = fetch.parse_area(area)
    if not isinstance(area, fetch.Area):
        raise ValueError("ERA5 needs an explicit geographic area")
    times = fetch._era5_times(cycle, hours, cadence)
    request = {"store": ARCO_STORE, "times": [t.isoformat() + "Z" for t in times],
        "area": [area.lat_south, area.lon_west, area.lat_north, area.lon_east],
        "expected_levels_hpa": list(fetch.ERA5_PRESSURE_LEVELS_HPA),
        "specific_humidity_authority": True,
        "fields": [{"source": source, "output": name, "units": REGULAR_FIELD_UNITS[name],
                    "pressure": pressure} for mapping, pressure in ((_PRESSURE, True), (_SURFACE, False))
                   for source, name in mapping.items()],
        "derived": [{"kind": "pressure_from_levels", "output": "PRES", "units": "Pa"},
                    {"kind": "specific_humidity_from_dewpoint", "output": "Q2", "units": "kg kg-1",
                     "inputs": ["D2", "T2", "PSFC"]}]}
    identity = {"source": "era5", "provider": "arco", "cycle": cycle.isoformat() + "Z",
                "hours": hours, "cadence_hours": cadence, "area": area.as_manifest(), "native_request": request}
    out = Path(out).expanduser().resolve()
    target = out / ARCO_COMBINED_NAME
    receipt_path = out / _RECEIPT
    with fetch_guard.hold("fetch-out", out, progress=progress):
        for path in (target, receipt_path):
            if os.path.lexists(path) and (path.is_symlink() or not path.is_file()):
                raise FileExistsError(f"ARCO retrieval preserves the non-regular path: {path}")
        if not force and (target.exists() or receipt_path.exists()):
            try:
                prior = json.loads(receipt_path.read_text(encoding="utf-8"))
                artifact = prior["artifact"]
                if (prior["schema"] == _SCHEMA and prior["request"] == identity
                        and target.stat().st_size == artifact["bytes"]
                        and fetch.sha256_file(target) == artifact["sha256"]):
                    _validate(target, times=times, area=area)
                    if fetch.sha256_file(target) == artifact["sha256"]:
                        progress(f"fetch era5: reused verified ARCO inputs in {target}")
                        return target
            except (OSError, ValueError, KeyError, TypeError):
                pass
            raise FileExistsError(f"ARCO output is not a verified match for this request: {target}. "
                "Choose another output directory or explicitly force a refetch; existing files were preserved.")
        out.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".era5-arco-", dir=out) as temporary:
            stage = Path(temporary)
            combined = stage / ARCO_COMBINED_NAME
            progress("fetch era5: reading Google ARCO source metadata and native Zarr chunks")
            native = extract_regular_zarr(request, request_path=stage / "request.json",
                                         output=combined, progress=progress)
            progress("fetch era5: validating native NetCDF forcing inventory and coverage")
            validation = _validate(combined, times=times, area=area)
            receipt = {"schema": _SCHEMA, "status": "validated", "request": identity,
                "artifact": {"name": target.name, "bytes": combined.stat().st_size,
                             "sha256": fetch.sha256_file(combined)},
                "native": native, "validation": validation, "forecast_started": False}
            staged_receipt = stage / _RECEIPT
            fetch_guard.atomic_write_text(staged_receipt,
                json.dumps(receipt, indent=2, sort_keys=True) + "\n", tag="era5-arco")
            if force:
                for path in (receipt_path, target):
                    if os.path.lexists(path):
                        if path.is_symlink() or not path.is_file():
                            raise FileExistsError(f"ARCO retrieval preserves the changed path: {path}")
                        aside = fetch_guard.quarantine(path, tag="era5-arco-refetch")
                        progress(f"fetch era5: preserved previous {path.name} as {aside.name}")
            os.link(combined, target)
            try:
                os.link(staged_receipt, receipt_path)
            except BaseException:
                if target.is_file() and os.path.samestat(combined.stat(), target.stat()):
                    target.unlink()
                raise
            fetch_guard._fsync_dir(out)
        progress(f"fetch era5: published validated ARCO inputs in {target}")
        return target
