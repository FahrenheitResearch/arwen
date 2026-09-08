"""ERA5 selection metadata and transport to the native GRIB1 member checker."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess

SCHEMA = "arwen.era5-member-selection.v1"


def validate_selection(*, product_type="reanalysis", member=None, cadence=6, provider="cds", cycle=None):
    if product_type not in ("reanalysis", "ensemble_members"):
        raise ValueError("ERA5 product must be reanalysis or ensemble_members")
    if provider not in ("cds", "arco"):
        raise ValueError("ERA5 provider must be cds or arco")
    if isinstance(cadence, bool) or not isinstance(cadence, int) or cadence not in (1, 3, 6):
        raise ValueError("ERA5 boundary cadence must be 1, 3 or 6 hours")
    if product_type == "reanalysis":
        if member is not None:
            raise ValueError("ERA5 reanalysis has no EDA member; select ensemble_members explicitly")
        return None
    if provider != "cds" or cadence != 3:
        raise ValueError("ERA5 EDA requires CDS and its native 3-hour boundary cadence")
    if isinstance(member, str) and len(member) == 1 and member.isascii() and member.isdigit():
        member = int(member)
    if isinstance(member, bool) or not isinstance(member, int) or not 0 <= member <= 9:
        raise ValueError("ERA5 EDA requires an explicit member number 0..9")
    if cycle is not None and (cycle.hour % 3 or cycle.minute or cycle.second or cycle.microsecond):
        raise ValueError("ERA5 EDA initialization must be an exact 3-hourly UTC analysis time")
    return member


def _invoke(bridge, arguments, *, timeout=300):
    process = subprocess.run([str(bridge), *map(str, arguments)], capture_output=True,
        timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0)
    if process.returncode:
        detail = process.stderr[:8192].decode("utf-8", errors="replace").strip()
        raise ValueError("Native ERA5 member verification refused the payload: " + detail)
    if len(process.stdout) > 16384:
        raise ValueError("Native ERA5 member checker returned oversized metadata")
    report = json.loads(process.stdout)
    if report.get("schema") != SCHEMA or report.get("byte_preserving") is not True:
        raise ValueError("Native bridge does not implement the required ERA5 member selection contract")
    return report


def require_bridge():
    from gpuwm.bridges import find_bridge
    bridge = find_bridge("grib1_bridge")
    if bridge is None:
        raise ValueError("ERA5 EDA needs the native grib1_bridge with --era5-member support; rebuild/install the matching ArWen bridge")
    report = _invoke(bridge, ["--era5-member-capabilities"], timeout=15)
    if report.get("members") != list(range(10)) or report.get("complete_input_census") is not True:
        raise ValueError("The selected native bridge lacks complete ten-member ERA5 EDA verification")
    if report.get("local_definitions") != [1, 17, 36]:
        raise ValueError("The selected native bridge lacks the ERA5 surface/soil/SST/sea-ice local definitions; rebuild/install the matching bridge")
    expected_lake = {"center": 98, "table": 228, "parameters": [8, 13, 14], "surface_only": True}
    if (report.get("table_qualified_census") is not True
            or report.get("lake_surface_parameters") != expected_lake):
        raise ValueError("The selected native bridge lacks table-qualified ERA5 lake-field member verification; rebuild/install the matching bridge")
    return Path(bridge)


def check_member(path, member, *, bridge=None):
    report = _invoke(bridge or require_bridge(), ["--check-era5-member", member, path])
    if report.get("member") != member or report.get("selected_only") is not True or report.get("messages", 0) <= 0:
        raise ValueError("Native ERA5 member receipt does not match the selected member")
    return report


def select_member(path, output, member, *, bridge=None):
    report = _invoke(bridge or require_bridge(), ["--era5-member", member, path, output])
    if report.get("member") != member or report.get("selected_only") is not True or report.get("input_messages") != 10 * report.get("messages", 0):
        raise ValueError("Native ERA5 selection did not verify all ten members for every field")
    return report
