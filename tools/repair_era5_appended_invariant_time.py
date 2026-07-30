"""Repair the known NCO-appended invariant-time overwrite in ERA5 NetCDF.

This is not a filename-trust shim.  It refuses to act unless the source's
embedded NCO histories independently bind the requested ERA5 day/hour and
also show the later 1979 invariant land-mask append that caused the overwrite.
Every variable other than ``time`` and ``utc_date`` must remain byte-identical
at the decoded-array level, and the repair emits a hash-bound JSON receipt.
"""

from __future__ import annotations

import argparse
import calendar
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil

import netCDF4
import numpy as np


RECEIPT_SCHEMA = "rw-wps-era5-netcdf-time-repair-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(variable) -> str:
    variable.set_auto_maskandscale(False)
    value = np.ascontiguousarray(variable[:])
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(value.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _variable_hashes(path: Path) -> dict[str, str]:
    with netCDF4.Dataset(path) as dataset:
        return {
            name: _array_sha256(variable)
            for name, variable in dataset.variables.items()
        }


def repair(
    source: Path,
    output: Path,
    receipt_path: Path,
    valid_time: datetime,
) -> dict[str, object]:
    source = source.resolve()
    output = output.resolve()
    receipt_path = receipt_path.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite repair output or receipt")
    if output.parent != receipt_path.parent:
        raise ValueError("repair output and receipt must share one staging directory")
    output.parent.mkdir(parents=True, exist_ok=True)

    with netCDF4.Dataset(source) as dataset:
        if "time" not in dataset.variables or "utc_date" not in dataset.variables:
            raise ValueError("source lacks scalar time/utc_date evidence")
        time_variable = dataset.variables["time"]
        utc_variable = dataset.variables["utc_date"]
        if time_variable.size != 1 or utc_variable.size != 1:
            raise ValueError("repair supports exactly one source time")
        units = getattr(time_variable, "units", None)
        calendar_name = getattr(time_variable, "calendar", "standard")
        if units != "hours since 1900-01-01 00:00:00" \
                or calendar_name != "gregorian":
            raise ValueError("source time coordinate differs from the ERA5 contract")
        observed_time = float(np.asarray(time_variable[:]).reshape(-1)[0])
        observed_utc_date = int(np.asarray(utc_variable[:]).reshape(-1)[0])
        observed_datetime = netCDF4.num2date(
            observed_time, units, calendar=calendar_name,
            only_use_cftime_datetimes=False,
        )
        if observed_datetime != datetime(1979, 1, 1) \
                or observed_utc_date != 1979010100:
            raise ValueError("source does not contain the known 1979 overwrite signature")
        history = "\n".join(
            str(getattr(dataset, name))
            for name in ("history", "history_of_appended_files")
            if name in dataset.ncattrs()
        )
        history_sha256 = hashlib.sha256(history.encode("utf-8")).hexdigest()
        day = valid_time.strftime("%Y%m%d")
        daily_token = f"{day}00_{day}23"
        daily_index = f"-d time,{valid_time.hour},{valid_time.hour}"
        month_end = calendar.monthrange(valid_time.year, valid_time.month)[1]
        month = valid_time.strftime("%Y%m")
        monthly_token = f"{month}0100_{month}{month_end:02d}23"
        monthly_index_value = (valid_time.day - 1) * 24 + valid_time.hour
        monthly_index = f"-d time,{monthly_index_value},{monthly_index_value}"
        invariant_token = "1979010100_1979010100"
        evidence = {
            "pressure_level_day": daily_token in history,
            "pressure_level_hour_index": daily_index in history,
            "surface_month": monthly_token in history,
            "surface_hour_index": monthly_index in history,
            "appended_invariant_1979": invariant_token in history,
            "nco_append": "ncks -A" in history,
        }
        if not all(evidence.values()):
            raise ValueError(f"embedded history does not prove requested time: {evidence}")

    before_variables = _variable_hashes(source)
    staging = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    try:
        shutil.copy2(source, staging)
        with netCDF4.Dataset(staging, "r+") as dataset:
            expected_numeric = netCDF4.date2num(
                valid_time, units, calendar=calendar_name,
            )
            dataset.variables["time"][:] = [expected_numeric]
            dataset.variables["utc_date"][:] = [int(valid_time.strftime("%Y%m%d%H"))]
            dataset.setncattr(
                "rw_wps_time_repair",
                json.dumps({
                    "schema": RECEIPT_SCHEMA,
                    "source_sha256": _sha256(source),
                    "history_sha256": history_sha256,
                    "valid_time": valid_time.isoformat(),
                }, sort_keys=True, separators=(",", ":")),
            )
        os.replace(staging, output)
    except BaseException:
        if staging.exists():
            staging.unlink()
        raise

    after_variables = _variable_hashes(output)
    protected = sorted(set(before_variables) - {"time", "utc_date"})
    changed = [
        name for name in protected
        if before_variables[name] != after_variables.get(name)
    ]
    if changed or set(before_variables) != set(after_variables):
        output.unlink(missing_ok=True)
        raise ValueError(f"repair changed protected variable arrays: {changed}")
    expected_time_hash = after_variables["time"]
    expected_utc_hash = after_variables["utc_date"]
    if expected_time_hash == before_variables["time"] \
            or expected_utc_hash == before_variables["utc_date"]:
        output.unlink(missing_ok=True)
        raise ValueError("repair did not change both corrupted time variables")

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "PASS",
        "source": {
            "path": str(source), "bytes": source.stat().st_size,
            "sha256": _sha256(source),
        },
        "output": {
            "path": str(output), "bytes": output.stat().st_size,
            "sha256": _sha256(output),
        },
        "requested_valid_time": valid_time.isoformat(),
        "observed_corrupt_time": observed_datetime.isoformat(),
        "observed_corrupt_utc_date": observed_utc_date,
        "history_sha256": history_sha256,
        "history_evidence": evidence,
        "protected_variable_count": len(protected),
        "protected_variable_hashes_unchanged": True,
        "protected_variable_sha256": {
            name: before_variables[name] for name in protected
        },
        "before_time_sha256": before_variables["time"],
        "after_time_sha256": expected_time_hash,
        "before_utc_date_sha256": before_variables["utc_date"],
        "after_utc_date_sha256": expected_utc_hash,
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
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--valid-time", required=True)
    args = parser.parse_args()
    valid_time = datetime.fromisoformat(args.valid_time)
    if valid_time.tzinfo is not None or valid_time.second or valid_time.microsecond:
        raise ValueError("valid time must be naive UTC at whole-minute precision")
    receipt = repair(args.input, args.output, args.receipt, valid_time)
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
