#!/usr/bin/env python3
"""Fail-closed seal for the native direct-export stock-WRF proof."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re

import netCDF4
import numpy as np


_BAD_LOG_PATTERN = re.compile(
    r"(^|[^A-Za-z])(NaN|Inf(?:inity)?|CFL|FATAL|SIGSEGV|"
    r"segmentation fault)([^A-Za-z]|$)",
    re.IGNORECASE | re.MULTILINE,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _timing(text: str, pattern: str) -> float:
    match = re.search(pattern, text)
    if not match:
        raise SystemExit(f"missing timing evidence: {pattern}")
    return float(match.group(1))


def _time_value(text: str, label: str) -> str:
    match = re.search(
        rf"^\s*{re.escape(label)}:\s*(.+)$", text, re.MULTILINE)
    if not match:
        raise SystemExit(f"missing /usr/bin/time field: {label}")
    return match.group(1).strip()


def _positive_fraction(value: object) -> Fraction:
    try:
        result = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as error:
        raise argparse.ArgumentTypeError(
            "time step must be a positive rational number of seconds"
        ) from error
    if result <= 0:
        raise argparse.ArgumentTypeError(
            "time step must be a positive rational number of seconds"
        )
    return result


def _logged_step_times(
    start: datetime, run_seconds: int, time_step_seconds: Fraction,
) -> list[str]:
    if run_seconds <= 0 or time_step_seconds <= 0:
        raise SystemExit(
            "run seconds must be a positive multiple of the positive time step"
        )
    step_count = Fraction(run_seconds, 1) / time_step_seconds
    if step_count.denominator != 1 or step_count <= 0:
        raise SystemExit(
            "run seconds must be a positive multiple of the positive time step"
        )
    result = []
    for ordinal in range(1, step_count.numerator + 1):
        offset = ordinal * time_step_seconds
        logged_second = offset.numerator // offset.denominator
        result.append((
            start + timedelta(seconds=logged_second)
        ).strftime("%Y-%m-%d_%H:%M:%S"))
    return result


def _expected_step_times(
    valid_time: str,
    run_seconds: int,
    time_step_seconds: Fraction | int,
) -> list[str]:
    try:
        start = datetime.strptime(valid_time, "%Y-%m-%d_%H:%M:%S")
    except ValueError as error:
        raise SystemExit(
            "valid time must use YYYY-MM-DD_HH:MM:SS"
        ) from error
    return _logged_step_times(
        start, run_seconds, Fraction(time_step_seconds),
    )


def _expected_hierarchy_step_times(
    valid_time: str,
    run_seconds: int,
    time_step_seconds: Fraction | int,
    hierarchy: object,
) -> dict[int, list[str]]:
    """Derive each WRF domain's logged integer-second timestamps."""

    root_times = _expected_step_times(
        valid_time, run_seconds, time_step_seconds,
    )
    if not isinstance(hierarchy, list) or not hierarchy:
        raise SystemExit("hierarchy export lacks a non-empty hierarchy")
    try:
        start = datetime.strptime(valid_time, "%Y-%m-%d_%H:%M:%S")
    except ValueError as error:  # already checked above; retain local authority
        raise SystemExit(
            "valid time must use YYYY-MM-DD_HH:MM:SS"
        ) from error
    expected_ids = list(range(1, len(hierarchy) + 1))
    actual_ids = [row.get("grid_id") for row in hierarchy]
    if actual_ids != expected_ids:
        raise SystemExit(
            "export hierarchy grid ids are not contiguous parent-first: "
            f"{actual_ids}"
        )

    exact_dt: dict[int, Fraction] = {}
    expected: dict[int, list[str]] = {}
    for row in hierarchy:
        grid_id = row["grid_id"]
        parent_id = row.get("parent_id")
        ratio = row.get("parent_time_step_ratio")
        if grid_id == 1:
            if parent_id != 0 or ratio != 1:
                raise SystemExit("hierarchy root has invalid parent/time ratio")
            domain_dt = Fraction(time_step_seconds)
        else:
            if (parent_id not in exact_dt or isinstance(ratio, bool)
                    or not isinstance(ratio, int) or ratio < 2):
                raise SystemExit(
                    f"d{grid_id:02d} has an invalid parent/time ratio"
                )
            domain_dt = exact_dt[parent_id] / ratio
        observed_dt = row.get("dt_s")
        if (isinstance(observed_dt, bool)
                or not isinstance(observed_dt, (int, float))
                or not np.isfinite(observed_dt)
                or not np.isclose(
                    float(observed_dt), float(domain_dt),
                    rtol=1e-7, atol=1e-9,
                )):
            raise SystemExit(
                f"d{grid_id:02d} manifest dt {observed_dt!r} differs from "
                f"the parent-ratio value {float(domain_dt)!r}"
            )
        exact_dt[grid_id] = domain_dt
        expected[grid_id] = _logged_step_times(
            start, run_seconds, domain_dt,
        )
    if expected[1] != root_times:
        raise SystemExit("hierarchy root timing differs from the requested step")
    return expected


def _hash_manifest(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1,
    ):
        match = re.fullmatch(r"([0-9a-f]{64})\s+[ *]?(.+)", line)
        if not match:
            raise SystemExit(
                f"invalid SHA-256 manifest row {path}:{line_number}"
            )
        name = Path(match.group(2)).name
        if name in rows:
            raise SystemExit(f"duplicate basename {name!r} in {path}")
        rows[name] = match.group(1)
    if not rows:
        raise SystemExit(f"empty SHA-256 manifest {path}")
    return rows


def _domain_paths(values: list[str]) -> dict[int, Path]:
    result = {}
    for value in values:
        match = re.fullmatch(r"d(\d{2})=(.+)", value)
        if not match or int(match.group(1)) < 1:
            raise SystemExit(
                "domain history paths must use dNN=/path/to/wrfout"
            )
        domain_id = int(match.group(1))
        if domain_id in result:
            raise SystemExit(f"duplicate history path for d{domain_id:02d}")
        candidate = Path(match.group(2)).resolve()
        aliases_existing = any(
            candidate == existing or (
                candidate.exists() and existing.exists()
                and candidate.samefile(existing)
            )
            for existing in result.values()
        )
        if aliases_existing:
            raise SystemExit(
                "multiple history domains resolve to the same file"
            )
        result[domain_id] = candidate
    return result


def _validate_export_file_inventory(
    manifest_files: object, domain_ids: tuple[int, ...],
) -> dict[str, object]:
    expected_files = {
        "wrfbdy_d01",
        *(f"wrfinput_d{domain_id:02d}" for domain_id in domain_ids),
    }
    if not isinstance(manifest_files, dict) \
            or set(manifest_files) != expected_files:
        observed_files = (
            sorted(manifest_files)
            if isinstance(manifest_files, dict) else repr(manifest_files)
        )
        raise SystemExit(
            "direct export file inventory differs from its domain hierarchy: "
            f"expected={sorted(expected_files)}, got={observed_files}"
        )
    return manifest_files


def _validate_export_authority(
    export_manifest: object, valid_time: str,
) -> list[dict[str, object]] | None:
    if not isinstance(export_manifest, dict) \
            or export_manifest.get("status") != "READY":
        raise SystemExit("direct export manifest is not READY")
    hierarchy = export_manifest.get("hierarchy")
    expected_schema = (
        "gpuwm-native-direct-wrf-export-v2"
        if hierarchy is None
        else "gpuwm-native-direct-wrf-hierarchy-export-v1"
    )
    if export_manifest.get("schema") != expected_schema:
        raise SystemExit(
            "direct export manifest schema differs from its hierarchy shape"
        )
    if export_manifest.get("valid_time") != valid_time:
        raise SystemExit(
            "direct export valid time differs from requested stock-WRF gate"
        )
    return hierarchy


def _history_identity(
    path: Path,
    *,
    domain_id: int,
    valid_time: str,
    hierarchy_row: dict[str, object] | None,
) -> dict[str, object]:
    """Bind a finite history readback to its requested domain and time."""

    with netCDF4.Dataset(path) as dataset:
        observed_domain = int(dataset.getncattr("GRID_ID"))
        if observed_domain != domain_id:
            raise SystemExit(
                f"d{domain_id:02d} history has GRID_ID={observed_domain}"
            )
        if "Times" not in dataset.variables:
            raise SystemExit(f"d{domain_id:02d} history lacks Times")
        decoded = np.asarray(netCDF4.chartostring(
            np.asarray(dataset.variables["Times"][:])
        )).reshape(-1)
        times = [str(value) for value in decoded]
        if valid_time not in times:
            raise SystemExit(
                f"d{domain_id:02d} history does not contain {valid_time}"
            )
        identity = {
            "grid_id": observed_domain,
            "times": times,
        }
        if hierarchy_row is not None:
            expected_integer = {
                "PARENT_ID": int(hierarchy_row["parent_id"]),
                "I_PARENT_START": int(hierarchy_row["i_parent_start"]),
                "J_PARENT_START": int(hierarchy_row["j_parent_start"]),
                "PARENT_GRID_RATIO": int(
                    hierarchy_row["parent_grid_ratio"]
                ),
                "WEST-EAST_GRID_DIMENSION": int(hierarchy_row["nx"]) + 1,
                "SOUTH-NORTH_GRID_DIMENSION": int(hierarchy_row["ny"]) + 1,
                "BOTTOM-TOP_GRID_DIMENSION": int(hierarchy_row["nz"]) + 1,
            }
            observed_integer = {
                name: int(dataset.getncattr(name))
                for name in expected_integer
            }
            if observed_integer != expected_integer:
                raise SystemExit(
                    f"d{domain_id:02d} history geometry differs from export "
                    "hierarchy: "
                    f"observed={observed_integer}, "
                    f"expected={expected_integer}"
                )
            expected_float = {
                "DX": float(hierarchy_row["dx_m"]),
                "DY": float(hierarchy_row["dy_m"]),
                "DT": float(hierarchy_row["dt_s"]),
            }
            observed_float = {
                name: float(dataset.getncattr(name))
                for name in expected_float
            }
            if any(
                not np.isfinite(observed_float[name]) or not np.isclose(
                    observed_float[name], expected_float[name],
                    rtol=1e-7, atol=1e-9,
                )
                for name in expected_float
            ):
                raise SystemExit(
                    f"d{domain_id:02d} history spacing/time step differs "
                    "from export hierarchy: "
                    f"observed={observed_float}, expected={expected_float}"
                )
            identity.update({
                "parent_id": observed_integer["PARENT_ID"],
                "i_parent_start": observed_integer["I_PARENT_START"],
                "j_parent_start": observed_integer["J_PARENT_START"],
                "parent_grid_ratio": observed_integer[
                    "PARENT_GRID_RATIO"
                ],
                "nx": observed_integer["WEST-EAST_GRID_DIMENSION"] - 1,
                "ny": observed_integer["SOUTH-NORTH_GRID_DIMENSION"] - 1,
                "nz": observed_integer["BOTTOM-TOP_GRID_DIMENSION"] - 1,
                "dx_m": observed_float["DX"],
                "dy_m": observed_float["DY"],
                "dt_s": observed_float["DT"],
            })
    return identity


def _finite_ranges(path: Path) -> dict[str, object]:
    names = (
        "U", "V", "W", "PH", "PHB", "T", "P", "PB", "MU", "MUB",
        "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP",
        "PSFC", "T2", "Q2", "TSK",
    )
    result = {}
    with netCDF4.Dataset(path) as dataset:
        missing = sorted(set(names) - set(dataset.variables))
        if missing:
            raise SystemExit(f"wrfout missing validation fields: {missing}")
        for name in names:
            value = np.asarray(dataset.variables[name][:])
            if not np.isfinite(value).all():
                raise SystemExit(f"wrfout contains non-finite {name}")
            result[name] = {
                "minimum": float(value.min()),
                "maximum": float(value.max()),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path)
    parser.add_argument("--time-log", type=Path, required=True)
    parser.add_argument("--wrf-exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--valid-time", default="2026-07-18_00:00:00")
    parser.add_argument("--run-seconds", type=int, default=10)
    parser.add_argument(
        "--time-step-seconds", type=_positive_fraction,
        default=Fraction(5, 1),
        help="root time step in seconds; decimal and rational values are accepted",
    )
    parser.add_argument(
        "--wrfout", type=Path,
        help="explicit history file path (useful on hosts that cannot name colons)",
    )
    parser.add_argument(
        "--wrfout-domain", action="append", default=[],
        help="repeat as dNN=/path/to/wrfout for a hierarchy",
    )
    parser.add_argument("--pre-run-hashes", type=Path)
    parser.add_argument("--post-run-hashes", type=Path)
    args = parser.parse_args()

    if (args.pre_run_hashes is None) != (args.post_run_hashes is None):
        raise SystemExit("pre/post run hash manifests must be supplied together")

    export_manifest = json.loads(
        (args.export / "manifest.json").read_text(encoding="utf-8"))
    hierarchy_rows = _validate_export_authority(
        export_manifest, args.valid_time,
    )
    if hierarchy_rows is None:
        domain_ids = (1,)
        expected_by_domain = {
            1: _expected_step_times(
                args.valid_time, args.run_seconds, args.time_step_seconds,
            ),
        }
    else:
        expected_by_domain = _expected_hierarchy_step_times(
            args.valid_time,
            args.run_seconds,
            args.time_step_seconds,
            hierarchy_rows,
        )
        domain_ids = tuple(expected_by_domain)
    manifest_files = _validate_export_file_inventory(
        export_manifest.get("files"), domain_ids,
    )
    for name, spec in manifest_files.items():
        exported = args.export / name
        consumed = args.run / name
        if _sha256(exported) != spec["sha256"]:
            raise SystemExit(f"exported {name} digest drift")
        if _sha256(consumed) != spec["sha256"]:
            raise SystemExit(f"stock WRF did not consume sealed {name}")

    stdout = args.stdout.read_text(encoding="utf-8", errors="replace")
    stderr = (
        "" if args.stderr is None
        else args.stderr.read_text(encoding="utf-8", errors="replace")
    )
    time_text = args.time_log.read_text(encoding="utf-8", errors="replace")
    if "wrf: SUCCESS COMPLETE WRF" not in stdout:
        raise SystemExit("stock WRF success marker is absent")
    if "FATAL CALLED" in stdout:
        raise SystemExit("stock WRF log contains a fatal marker")
    bad_log_matches = [match.group(2) for match in _BAD_LOG_PATTERN.finditer(
        stdout + "\n" + stderr + "\n" + time_text
    )]
    if bad_log_matches:
        raise SystemExit(f"stock WRF log contains blocking patterns: {bad_log_matches}")
    if _time_value(time_text, "Exit status") != "0":
        raise SystemExit("stock WRF process exit status is nonzero")
    for name in manifest_files:
        marker = f"Input data is acceptable to use: {name}"
        if marker not in stdout:
            raise SystemExit(f"stock WRF acceptance marker is absent for {name}")
    parsed_steps = [
        {
            "valid_time": match.group(1),
            "domain_id": int(match.group(2)),
            "elapsed_seconds": float(match.group(3)),
        }
        for match in re.finditer(
            r"Timing for main: time (\S+) on domain\s+(\d+):\s+([0-9.]+)",
            stdout,
        )
    ]
    observed_domain_ids = {item["domain_id"] for item in parsed_steps}
    if observed_domain_ids != set(domain_ids):
        raise SystemExit(
            "completed WRF steps cover unexpected domains: "
            f"{sorted(observed_domain_ids)}"
        )
    for domain_id, expected_steps in expected_by_domain.items():
        observed_times = [
            item["valid_time"] for item in parsed_steps
            if item["domain_id"] == domain_id
        ]
        if observed_times != expected_steps:
            raise SystemExit(
                f"unexpected completed WRF steps for d{domain_id:02d}: "
                f"{observed_times}"
            )
    main_steps = (
        [
            {
                "valid_time": item["valid_time"],
                "elapsed_seconds": item["elapsed_seconds"],
            }
            for item in parsed_steps
        ]
        if domain_ids == (1,)
        else parsed_steps
    )

    explicit_wrfout = _domain_paths(args.wrfout_domain)
    if args.wrfout is not None and explicit_wrfout:
        raise SystemExit("--wrfout and --wrfout-domain are mutually exclusive")
    if args.wrfout is not None and domain_ids != (1,):
        raise SystemExit("--wrfout is only valid for a single-domain export")
    if explicit_wrfout and set(explicit_wrfout) != set(domain_ids):
        raise SystemExit(
            "explicit history paths differ from export domains: "
            f"expected={list(domain_ids)}, got={sorted(explicit_wrfout)}"
        )
    wrfout_paths = explicit_wrfout or {
        domain_id: (
            args.wrfout
            if domain_id == 1 and args.wrfout is not None
            else args.run / f"wrfout_d{domain_id:02d}_{args.valid_time}"
        )
        for domain_id in domain_ids
    }
    hash_evidence: dict[str, object] = {}
    if args.pre_run_hashes is not None:
        pre_run = _hash_manifest(args.pre_run_hashes)
        post_run = _hash_manifest(args.post_run_hashes)
        if pre_run != post_run:
            raise SystemExit("stock WRF mutated a sealed input or authority")
        expected_hashes = {
            "wrf.exe": _sha256(args.wrf_exe),
            "namelist.input": _sha256(args.run / "namelist.input"),
            **{
                name: str(spec["sha256"])
                for name, spec in manifest_files.items()
            },
        }
        if pre_run != expected_hashes:
            raise SystemExit(
                "pre/post run hash manifest differs from consumed authorities"
            )
        hash_evidence = {
            "inputs_unchanged": True,
            "pre_run_manifest_sha256": _sha256(args.pre_run_hashes),
            "post_run_manifest_sha256": _sha256(args.post_run_hashes),
            "sealed_sha256": pre_run,
        }
    wrfout_domains = {}
    hierarchy_by_domain = (
        {} if hierarchy_rows is None
        else {int(row["grid_id"]): row for row in hierarchy_rows}
    )
    for domain_id, wrfout in wrfout_paths.items():
        wrfout_domains[f"d{domain_id:02d}"] = {
            "path": wrfout.name,
            "bytes": wrfout.stat().st_size,
            "sha256": _sha256(wrfout),
            "finite_ranges": _finite_ranges(wrfout),
            "identity": _history_identity(
                wrfout,
                domain_id=domain_id,
                valid_time=args.valid_time,
                hierarchy_row=hierarchy_by_domain.get(domain_id),
            ),
            "time": args.valid_time,
            "note": (
                "initial history readback; advancement is bound by log steps"
            ),
        }
    wrfinput_timings = {
        f"d{domain_id:02d}": _timing(
            stdout,
            rf"Timing for processing wrfinput file .*? for domain\s+"
            rf"{domain_id}:\s+([0-9.]+)",
        )
        for domain_id in domain_ids
    }
    readback = {"wrfout": wrfout_domains["d01"]}
    if domain_ids != (1,):
        readback["wrfout_domains"] = wrfout_domains
    payload = {
        "schema": (
            "gpuwm-native-direct-wrf-stock-oracle-v1"
            if domain_ids == (1,)
            else "gpuwm-native-direct-wrf-hierarchy-stock-oracle-v1"
        ),
        "status": "PASS",
        "oracle": {
            "identity": "unchanged stock WRF v4.6.1 wrf.exe",
            "wrf_exe_sha256": _sha256(args.wrf_exe),
            "command": str(args.wrf_exe),
            "note": (
                "The stock oracle uses RRTM longwave because stock WRF "
                "rejects ra_lw_physics=0; the exported IC/LBC are unchanged."
            ),
        },
        "consumed_export": export_manifest,
        "acceptance": {
            "domain_count": len(domain_ids),
            "wrfinput_elapsed_seconds": wrfinput_timings["d01"],
            **(
                {} if domain_ids == (1,)
                else {"wrfinput_elapsed_seconds_by_domain": wrfinput_timings}
            ),
            "wrfbdy_elapsed_seconds": _timing(
                stdout,
                r"Timing for processing lateral boundary .*?:\s+([0-9.]+)"),
            "completed_main_steps": main_steps,
            "success_marker": "wrf: SUCCESS COMPLETE WRF",
            "blocking_log_pattern_count": 0,
            "process_wall": _time_value(time_text, "Elapsed (wall clock) time (h:mm:ss or m:ss)"),
            "maximum_rss_kib": int(_time_value(
                time_text, "Maximum resident set size (kbytes)")),
            "exit_status": 0,
        },
        "readback": readback,
        "evidence": {
            "namelist_sha256": _sha256(args.run / "namelist.input"),
            "stdout_sha256": _sha256(args.stdout),
            **(
                {} if args.stderr is None
                else {"stderr_sha256": _sha256(args.stderr)}
            ),
            "time_log_sha256": _sha256(args.time_log),
            **hash_evidence,
        },
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = args.output.with_name(args.output.name + f".tmp-{os.getpid()}")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, args.output)
    print(encoded, end="")


if __name__ == "__main__":
    main()
