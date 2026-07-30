#!/usr/bin/env python3
"""Run a fail-closed unchanged-stock-WRF oracle for one HRRR export.

The helper deliberately builds a fresh run directory from only the symlinks in
an existing stock-WRF table/template directory.  Previous IC/LBC, namelists,
history, restart, log, and executable artifacts are never inherited.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable, Mapping

import netCDF4
import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gpuwm.ingest.hrrr_target import (  # noqa: E402
    HrrrTargetDomain,
    load_hrrr_target_domain,
)
from gpuwm.wrf_direct import _dimensions, _load_contract  # noqa: E402


EXPORT_SCHEMA = "gpuwm-native-direct-wrf-export-v2"
EVIDENCE_SCHEMA = "gpuwm-hrrr-stock-wrf-acceptance-v1"
REQUIRED_EXPORT_FILES = ("wrfinput_d01", "wrfbdy_d01")
REQUIRED_ACCEPTANCE_LINES = {
    "wrfinput_d01": "Input data is acceptable to use: wrfinput_d01",
    "wrfbdy_d01": "Input data is acceptable to use: wrfbdy_d01",
}
SUCCESS_LINE = "wrf: SUCCESS COMPLETE WRF"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_MAIN_TIMING_RE = re.compile(
    r"Timing for main: time\s+(\S+)\s+on domain\s+1:\s+([0-9.]+)")
_EXCLUDED_TEMPLATE_PATTERNS = (
    "namelist.*",
    "wrf.exe",
    "real.exe",
    "ideal.exe",
    "ndown.exe",
    "wrfinput_d*",
    "wrfbdy_d*",
    "wrflowinp_d*",
    "wrffdda_d*",
    "wrfout_d*",
    "wrfrst_d*",
    "met_em.*",
    "rsl.*",
    "*.log",
    "*.json",
    "core",
    "core.*",
)
_WRFOUT_FINITE_FIELDS = (
    "U", "V", "W", "PH", "PHB", "T", "P", "PB", "MU", "MUB",
    "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP",
    "PSFC", "T2", "Q2", "TSK",
)


class AcceptanceFailure(RuntimeError):
    """An acceptance prerequisite or oracle gate failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _is_excluded_template_name(name: str) -> bool:
    lowered = name.lower()
    return any(fnmatch.fnmatchcase(lowered, pattern)
               for pattern in _EXCLUDED_TEMPLATE_PATTERNS)


def template_symlink_plan(template: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Return the clean symlink inventory and deliberately skipped names."""

    if not template.is_dir():
        raise AcceptanceFailure(f"stock-WRF template is not a directory: {template}")
    links: list[dict[str, str]] = []
    skipped: list[str] = []
    for source in sorted(template.iterdir(), key=lambda item: item.name):
        if _is_excluded_template_name(source.name):
            skipped.append(source.name)
            continue
        if not source.is_symlink():
            raise AcceptanceFailure(
                "stock-WRF template contains a non-symlink table artifact: "
                f"{source.name}")
        try:
            resolved = source.resolve(strict=True)
        except FileNotFoundError as error:
            raise AcceptanceFailure(
                f"stock-WRF template contains a broken symlink: {source.name}") \
                from error
        links.append({
            "name": source.name,
            "template_target": os.readlink(source),
            "resolved_target": str(resolved),
            "target_kind": "directory" if resolved.is_dir() else "file",
        })
    if not links:
        raise AcceptanceFailure("stock-WRF template has no usable table symlinks")
    return links, skipped


def _copy_verified_export(export: Path, destination: Path,
                          manifest: Mapping[str, object]) -> None:
    files = manifest["files"]
    for name in REQUIRED_EXPORT_FILES:
        source = export / name
        if not source.is_file() or source.is_symlink():
            raise AcceptanceFailure(
                f"direct export {name} must be a regular file")
        expected = files[name]
        observed = sha256_file(source)
        if observed != expected["sha256"]:
            raise AcceptanceFailure(f"direct export {name} digest drift")
        if source.stat().st_size != expected["bytes"]:
            raise AcceptanceFailure(f"direct export {name} size drift")
        shutil.copy2(source, destination / name)
        if sha256_file(destination / name) != observed:
            raise AcceptanceFailure(f"copy verification failed for {name}")


def assemble_run_directory(*, template: Path, export: Path, run_dir: Path,
                           wrf_exe: Path,
                           export_manifest: Mapping[str, object],
                           domain_spec: Path, gpuwm_namelist: Path,
                           valid_time: str,
                           run_seconds: int) -> dict[str, object]:
    """Atomically assemble a new stock-WRF run directory."""

    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite run directory: {run_dir}")
    template_resolved = template.resolve(strict=True)
    run_parent = run_dir.parent.resolve()
    try:
        run_parent.relative_to(template_resolved)
    except ValueError:
        pass
    else:
        raise AcceptanceFailure("run directory may not be inside the template")
    links, skipped = template_symlink_plan(template)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{run_dir.name}.tmp-", dir=run_dir.parent))
    try:
        for item in links:
            destination = staging / item["name"]
            destination.symlink_to(
                item["resolved_target"],
                target_is_directory=item["target_kind"] == "directory",
            )
        _copy_verified_export(export, staging, export_manifest)
        (staging / "wrf.exe").symlink_to(wrf_exe.resolve(strict=True))
        command = [
            sys.executable,
            str(REPO / "tools" / "write_hrrr_stock_wrf_namelist.py"),
            "--domain-spec", str(domain_spec),
            "--gpuwm-namelist", str(gpuwm_namelist),
            "--valid-time", valid_time,
            "--run-seconds", str(run_seconds),
            "--output", str(staging / "namelist.input"),
        ]
        completed = subprocess.run(
            command, check=False, text=True, capture_output=True)
        if completed.returncode != 0:
            raise AcceptanceFailure(
                "stock-WRF namelist generation failed: "
                f"{completed.stderr.strip()}")
        os.replace(staging, run_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "template": str(template_resolved),
        "included_symlinks": links,
        "skipped_artifacts": skipped,
        "namelist_sha256": sha256_file(run_dir / "namelist.input"),
    }


def _read_times(dataset: netCDF4.Dataset) -> list[str]:
    if "Times" not in dataset.variables:
        raise AcceptanceFailure(f"{Path(dataset.filepath()).name} has no Times")
    values = dataset.variables["Times"][:]
    return [b"".join(np.asarray(row).tolist()).decode("ascii")
            for row in values]


def _finite_variable(variable: netCDF4.Variable,
                     *, budget_bytes: int = 64 * 1024 * 1024) -> int:
    """Check a floating variable in bounded slabs; return the largest slab."""

    if np.dtype(variable.dtype).kind != "f":
        return 0
    shape = tuple(int(size) for size in variable.shape)
    total = math.prod(shape) * np.dtype(variable.dtype).itemsize
    if not shape or total <= budget_bytes:
        selections: Iterable[object] = (Ellipsis,)
    else:
        axis = max(range(len(shape)), key=lambda index: shape[index])
        per_index = max(1, total // shape[axis])
        count = max(1, budget_bytes // per_index)
        slices = []
        for start in range(0, shape[axis], count):
            selection = [slice(None)] * len(shape)
            selection[axis] = slice(start, min(shape[axis], start + count))
            slices.append(tuple(selection))
        selections = slices
    largest = 0
    for selection in selections:
        value = variable[selection]
        if np.ma.isMaskedArray(value) and np.ma.getmaskarray(value).any():
            raise AcceptanceFailure(f"{variable.name} contains masked data")
        array = np.asarray(value)
        largest = max(largest, array.nbytes)
        if not np.isfinite(array).all():
            raise AcceptanceFailure(f"{variable.name} contains non-finite data")
    return largest


def _assert_close(label: str, actual: object, expected: float) -> None:
    try:
        value = float(actual)
    except (TypeError, ValueError) as error:
        raise AcceptanceFailure(f"invalid {label} attribute") from error
    if not math.isclose(value, expected, rel_tol=2.0e-6, abs_tol=1.0e-6):
        raise AcceptanceFailure(f"{label} mismatch: {value} != {expected}")


def _expected_identity_attrs(path_name: str, target: HrrrTargetDomain,
                             valid_time: str) -> dict[str, object]:
    attrs: dict[str, object] = {
        "START_DATE": valid_time,
        "WEST-EAST_GRID_DIMENSION": target.nx + 1,
        "SOUTH-NORTH_GRID_DIMENSION": target.ny + 1,
        "BOTTOM-TOP_GRID_DIMENSION": target.nz + 1,
    }
    if path_name == "wrfinput_d01":
        attrs["SIMULATION_START_DATE"] = valid_time
    elif path_name != "wrfbdy_d01":
        raise AcceptanceFailure(f"unsupported direct-export file {path_name}")
    return attrs


def _validate_dataset(path: Path, contract: Mapping[str, object], *,
                      target: HrrrTargetDomain, valid_time: str,
                      expected_times: list[str]) -> dict[str, object]:
    dimensions = _dimensions(
        contract, nx=target.nx, ny=target.ny, nz=target.nz)
    largest_slab = 0
    float_count = 0
    with netCDF4.Dataset(path) as dataset:
        expected_variables = [item["name"] for item in contract["variables"]]
        if list(dataset.variables) != expected_variables:
            raise AcceptanceFailure(f"{path.name} variable inventory/order drift")
        if set(dataset.dimensions) != set(dimensions):
            raise AcceptanceFailure(f"{path.name} dimension inventory drift")
        for name, expected in dimensions.items():
            dimension = dataset.dimensions[name]
            if expected is None:
                if not dimension.isunlimited():
                    raise AcceptanceFailure(
                        f"{path.name} {name} is not unlimited")
            elif len(dimension) != expected:
                raise AcceptanceFailure(
                    f"{path.name} {name} length {len(dimension)} != {expected}")
        if _read_times(dataset) != expected_times:
            raise AcceptanceFailure(f"{path.name} Times identity mismatch")
        expected_attrs = _expected_identity_attrs(
            path.name, target, valid_time)
        for name, expected in expected_attrs.items():
            if dataset.getncattr(name) != expected:
                raise AcceptanceFailure(f"{path.name} {name} mismatch")
        grid = target.grid()
        float_attrs = {
            "DX": target.dx_m,
            "DY": target.dy_m,
            "DT": target.time_step_seconds,
            "CEN_LAT": float(grid.cen_lat),
            "CEN_LON": float(grid.cen_lon),
            "MOAD_CEN_LAT": target.ref_lat,
            "TRUELAT1": target.truelat1,
            "TRUELAT2": target.truelat2,
            "STAND_LON": target.stand_lon,
        }
        for name, expected in float_attrs.items():
            _assert_close(f"{path.name} {name}", dataset.getncattr(name), expected)
        for variable in dataset.variables.values():
            if np.dtype(variable.dtype).kind == "f":
                float_count += 1
                largest_slab = max(largest_slab, _finite_variable(variable))
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "times": expected_times,
        "dimension_count": len(dimensions),
        "variable_count": len(contract["variables"]),
        "finite_float_variable_count": float_count,
        "largest_validation_slab_bytes": largest_slab,
    }


def validate_export(export: Path, domain_spec: Path,
                    valid_time: str) -> tuple[dict[str, object], dict[str, object]]:
    """Reopen an arbitrary-domain direct export and bind it to its target."""

    target = load_hrrr_target_domain(domain_spec)
    manifest_path = export / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AcceptanceFailure("cannot read direct-export manifest") from error
    if manifest.get("schema") != EXPORT_SCHEMA or manifest.get("status") != "READY":
        raise AcceptanceFailure("direct-export manifest is not a READY v2 export")
    if manifest.get("valid_time") != valid_time:
        raise AcceptanceFailure("direct-export valid time mismatch")
    if manifest.get("dimensions") != {
            "nx": target.nx, "ny": target.ny, "nz": target.nz}:
        raise AcceptanceFailure("direct-export target dimensions mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(REQUIRED_EXPORT_FILES):
        raise AcceptanceFailure("direct-export file inventory mismatch")
    for name in REQUIRED_EXPORT_FILES:
        spec = files[name]
        if (not isinstance(spec, dict)
                or isinstance(spec.get("bytes"), bool)
                or not isinstance(spec.get("bytes"), int)
                or spec["bytes"] <= 0
                or not isinstance(spec.get("sha256"), str)
                or not _SHA256_RE.fullmatch(spec["sha256"])):
            raise AcceptanceFailure(f"invalid direct-export manifest entry {name}")
    boundary_times = manifest.get("boundary_times")
    count = manifest.get("boundary_record_count")
    if (not isinstance(boundary_times, list)
            or not boundary_times
            or isinstance(count, bool)
            or not isinstance(count, int)
            or len(boundary_times) != count
            or any(not isinstance(value, str) for value in boundary_times)):
        raise AcceptanceFailure("invalid direct-export boundary time inventory")
    if boundary_times[0] != valid_time:
        raise AcceptanceFailure("direct-export boundary times do not begin at f00")
    interval = manifest.get("boundary_interval_seconds")
    if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
        raise AcceptanceFailure("invalid direct-export boundary interval")
    try:
        parsed_boundary_times = [
            datetime.strptime(value, "%Y-%m-%d_%H:%M:%S")
            for value in boundary_times]
    except ValueError as error:
        raise AcceptanceFailure("invalid direct-export boundary timestamp") from error
    if any(
            later - earlier != timedelta(seconds=interval)
            for earlier, later in zip(
                parsed_boundary_times, parsed_boundary_times[1:])):
        raise AcceptanceFailure("direct-export boundary cadence mismatch")
    next_boundary_times = manifest.get("next_boundary_times")
    expected_next = [
        (value + timedelta(seconds=interval)).strftime("%Y-%m-%d_%H:%M:%S")
        for value in parsed_boundary_times]
    if next_boundary_times != expected_next:
        raise AcceptanceFailure("direct-export next-boundary identity mismatch")
    contract = _load_contract()
    reopened = {
        "wrfinput_d01": _validate_dataset(
            export / "wrfinput_d01", contract["wrfinput"],
            target=target, valid_time=valid_time,
            expected_times=[valid_time]),
        "wrfbdy_d01": _validate_dataset(
            export / "wrfbdy_d01", contract["wrfbdy"],
            target=target, valid_time=valid_time,
            expected_times=boundary_times),
    }
    for name in REQUIRED_EXPORT_FILES:
        if reopened[name]["sha256"] != files[name]["sha256"]:
            raise AcceptanceFailure(f"reopened {name} digest drift")
        if reopened[name]["bytes"] != files[name]["bytes"]:
            raise AcceptanceFailure(f"reopened {name} size drift")
    evidence = {
        "status": "PASS",
        "target_domain_sha256": target.identity_sha256(),
        "target_domain": target.to_payload(),
        "domain_spec_sha256": sha256_file(domain_spec),
        "manifest_sha256": sha256_file(manifest_path),
        "files": reopened,
    }
    return manifest, evidence


def _matching_lines(texts: Mapping[str, str], needle: str) -> list[dict[str, str]]:
    matches = []
    for source, text in texts.items():
        for line in text.splitlines():
            if needle in line:
                matches.append({"source": source, "line": line.strip()})
    return matches


def validate_wrf_logs(*, run_dir: Path, stdout_path: Path,
                      valid_time: str, run_seconds: int,
                      returncode: int) -> dict[str, object]:
    log_paths = {stdout_path.name: stdout_path}
    for name in ("rsl.out.0000", "rsl.error.0000"):
        path = run_dir / name
        if path.is_file():
            log_paths[name] = path
    texts = {
        name: path.read_text(encoding="utf-8", errors="replace")
        for name, path in log_paths.items()
    }
    if returncode != 0:
        raise AcceptanceFailure(f"stock WRF exited with status {returncode}")
    gates = {}
    for name, exact in REQUIRED_ACCEPTANCE_LINES.items():
        matches = _matching_lines(texts, exact)
        if not matches:
            raise AcceptanceFailure(f"stock WRF did not accept {name}")
        gates[name] = {"exact_marker": exact, "matches": matches}
    success_matches = _matching_lines(texts, SUCCESS_LINE)
    if not success_matches:
        raise AcceptanceFailure("stock WRF success marker is absent")
    combined = "\n".join(texts.values())
    if "FATAL CALLED" in combined:
        raise AcceptanceFailure("stock WRF log contains FATAL CALLED")
    steps = [
        {"valid_time": match.group(1),
         "elapsed_seconds": float(match.group(2))}
        for match in _MAIN_TIMING_RE.finditer(combined)
    ]
    if not steps:
        raise AcceptanceFailure("stock WRF log has no completed main step")
    start = datetime.strptime(valid_time, "%Y-%m-%d_%H:%M:%S")
    expected_end = (start + timedelta(seconds=run_seconds)).strftime(
        "%Y-%m-%d_%H:%M:%S")
    if steps[-1]["valid_time"] != expected_end:
        raise AcceptanceFailure(
            f"stock WRF ended at {steps[-1]['valid_time']} not {expected_end}")
    return {
        "input_acceptance": gates,
        "success": {
            "exact_marker": SUCCESS_LINE,
            "matches": success_matches,
        },
        "completed_main_steps": steps,
        "expected_end_time": expected_end,
        "exit_status": returncode,
        "log_sha256": {
            name: sha256_file(path) for name, path in log_paths.items()},
    }


def _validate_wrfout(run_dir: Path, target: HrrrTargetDomain,
                     valid_time: str) -> dict[str, object]:
    path = run_dir / f"wrfout_d01_{valid_time}"
    if not path.is_file():
        raise AcceptanceFailure(f"stock WRF did not emit {path.name}")
    largest_slab = 0
    ranges = {}
    with netCDF4.Dataset(path) as dataset:
        missing = sorted(set(_WRFOUT_FINITE_FIELDS) - set(dataset.variables))
        if missing:
            raise AcceptanceFailure(f"wrfout missing fields: {missing}")
        expected = {
            "west_east": target.nx,
            "south_north": target.ny,
            "bottom_top": target.nz,
        }
        for name, length in expected.items():
            if name not in dataset.dimensions or len(dataset.dimensions[name]) != length:
                raise AcceptanceFailure(f"wrfout {name} dimension mismatch")
        for name in _WRFOUT_FINITE_FIELDS:
            variable = dataset.variables[name]
            largest_slab = max(largest_slab, _finite_variable(variable))
            value = np.asarray(variable[(0,) + (slice(None),) * (variable.ndim - 1)])
            ranges[name] = {
                "minimum": float(value.min()),
                "maximum": float(value.max()),
            }
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "finite_ranges": ranges,
        "largest_validation_slab_bytes": largest_slab,
    }


def _parse_time_log(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    fields = {}
    for label in (
            "Elapsed (wall clock) time (h:mm:ss or m:ss)",
            "Maximum resident set size (kbytes)",
            "Exit status"):
        match = re.search(rf"^\s*{re.escape(label)}:\s*(.+)$", text, re.MULTILINE)
        if not match:
            raise AcceptanceFailure(f"missing /usr/bin/time field: {label}")
        fields[label] = match.group(1).strip()
    if fields["Exit status"] != "0":
        raise AcceptanceFailure("/usr/bin/time recorded nonzero exit status")
    return {
        "wall_clock": fields["Elapsed (wall clock) time (h:mm:ss or m:ss)"],
        "maximum_rss_kib": int(fields["Maximum resident set size (kbytes)"]),
        "exit_status": 0,
        "sha256": sha256_file(path),
    }


def run_acceptance(args: argparse.Namespace) -> dict[str, object]:
    if args.run_seconds <= 0:
        raise AcceptanceFailure("run-seconds must be positive")
    expected_wrf_sha = args.expected_wrf_sha256.lower()
    if not _SHA256_RE.fullmatch(expected_wrf_sha):
        raise AcceptanceFailure("expected-wrf-sha256 must be 64 lowercase hex digits")
    wrf_exe = args.wrf_exe.resolve(strict=True)
    if not wrf_exe.is_file():
        raise AcceptanceFailure("wrf.exe does not resolve to a regular file")
    actual_wrf_sha = sha256_file(wrf_exe)
    if actual_wrf_sha != expected_wrf_sha:
        raise AcceptanceFailure("wrf.exe SHA-256 does not match pinned identity")
    time_exe = args.time_exe.resolve(strict=True)
    if not time_exe.is_file() or not os.access(time_exe, os.X_OK):
        raise AcceptanceFailure("time executable is not an executable file")

    manifest, reopen = validate_export(
        args.export, args.domain_spec, args.valid_time)
    assembly = assemble_run_directory(
        template=args.template_run_dir,
        export=args.export,
        run_dir=args.run_dir,
        wrf_exe=wrf_exe,
        export_manifest=manifest,
        domain_spec=args.domain_spec,
        gpuwm_namelist=args.gpuwm_namelist,
        valid_time=args.valid_time,
        run_seconds=args.run_seconds,
    )
    stdout_path = args.run_dir / "wrf.stdout.log"
    time_path = args.run_dir / "wrf.time.log"
    command = [
        str(time_exe), "-v", "-o", str(time_path.resolve()), "./wrf.exe"]
    with stdout_path.open("wb") as stdout:
        completed = subprocess.run(
            command,
            cwd=args.run_dir,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            check=False,
        )
    logs = validate_wrf_logs(
        run_dir=args.run_dir,
        stdout_path=stdout_path,
        valid_time=args.valid_time,
        run_seconds=args.run_seconds,
        returncode=completed.returncode,
    )
    timing = _parse_time_log(time_path)
    target = load_hrrr_target_domain(args.domain_spec)
    wrfout = _validate_wrfout(args.run_dir, target, args.valid_time)
    consumed = {}
    for name in REQUIRED_EXPORT_FILES:
        digest = sha256_file(args.run_dir / name)
        if digest != manifest["files"][name]["sha256"]:
            raise AcceptanceFailure(f"stock WRF mutated sealed {name}")
        consumed[name] = digest
    if sha256_file(wrf_exe) != expected_wrf_sha:
        raise AcceptanceFailure("wrf.exe identity changed during acceptance run")
    return {
        "schema": EVIDENCE_SCHEMA,
        "status": "PASS",
        "oracle": {
            "identity": args.wrf_identity,
            "resolved_wrf_exe": str(wrf_exe),
            "expected_wrf_exe_sha256": expected_wrf_sha,
            "observed_wrf_exe_sha256": actual_wrf_sha,
            "command": command,
        },
        "export_reopen": reopen,
        "direct_export_manifest": manifest,
        "assembly": assembly,
        "acceptance": logs,
        "timing": timing,
        "consumed_input_sha256": consumed,
        "wrfout_readback": wrfout,
        "evidence_inputs": {
            "gpuwm_namelist_sha256": sha256_file(args.gpuwm_namelist),
            "stdout_sha256": sha256_file(stdout_path),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--domain-spec", type=Path, required=True)
    parser.add_argument("--gpuwm-namelist", type=Path, required=True)
    parser.add_argument("--valid-time", required=True)
    parser.add_argument("--run-seconds", type=int, default=10)
    parser.add_argument("--template-run-dir", type=Path, required=True)
    parser.add_argument("--wrf-exe", type=Path, required=True)
    parser.add_argument("--expected-wrf-sha256", required=True)
    parser.add_argument(
        "--wrf-identity", default="unchanged stock WRF wrf.exe")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--time-exe", type=Path, default=Path("/usr/bin/time"))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.evidence.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {args.evidence}")
    payload = run_acceptance(args)
    _atomic_json(args.evidence, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
