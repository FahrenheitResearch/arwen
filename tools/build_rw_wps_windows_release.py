#!/usr/bin/env python3
"""Build the complete Windows x86_64 CPU RW-WPS runtime from one checkout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.build_native_wrf_windows_distribution import (  # noqa: E402
    build_windows_distribution,
)
from tools.build_rw_wps_release import (  # noqa: E402
    _cargo_release_environment,
    _run,
    _stage_rw_wps_python_project,
)


def build_windows_release(args: argparse.Namespace) -> dict[str, object]:
    if platform.system() != "Windows" or platform.machine() not in {
            "x86_64", "AMD64"}:
        raise RuntimeError(
            "RW-WPS Windows release assembly requires Windows x86_64")
    if sys.version_info < (3, 11):
        raise RuntimeError("RW-WPS release assembly requires Python 3.11+")

    output = args.output_dir.resolve()
    archive = args.archive.resolve()
    if output.exists() or archive.exists():
        raise FileExistsError("output directory and archive must not exist")
    if archive.suffix.lower() != ".zip":
        raise ValueError("Windows RW-WPS archive must use a .zip suffix")
    output.parent.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)

    dirty = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain"], text=True)
    if dirty:
        raise RuntimeError("refusing to build RW-WPS from a dirty source tree")

    manifest = REPO / "tools" / "grib1_bridge" / "Cargo.toml"
    source_date_epoch = subprocess.check_output(
        ["git", "-C", str(REPO), "show", "-s", "--format=%ct", "HEAD"],
        text=True,
    ).strip()
    with tempfile.TemporaryDirectory(prefix="rw-wps-windows-release-build-") as raw:
        temporary = Path(raw)
        python_project = temporary / "python-project"
        python_inventory = _stage_rw_wps_python_project(python_project)
        wheel_dir = temporary / "wheel"
        wheel_dir.mkdir()
        wheel_environment = os.environ.copy()
        wheel_environment["SOURCE_DATE_EPOCH"] = source_date_epoch
        wheel_environment["PYTHONHASHSEED"] = "0"
        _run([
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--disable-pip-version-check",
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
            str(python_project),
        ], env=wheel_environment)
        wheels = sorted(wheel_dir.glob("rw_wps-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(
                f"wheel build produced {len(wheels)} RW-WPS wheels: {wheels}")

        cargo_target = temporary / "cargo-target"
        environment = _cargo_release_environment(
            source_root=REPO,
            target_dir=cargo_target,
            source_date_epoch=source_date_epoch,
        )
        _run([
            "cargo",
            "build",
            "--manifest-path",
            str(manifest),
            "--release",
            "--locked",
            "--offline",
        ], cwd=manifest.parent, env=environment)
        native = cargo_target / "release"
        distribution_args = argparse.Namespace(
            wheel=wheels[0],
            grib1_bridge=native / "grib1_bridge.exe",
            grib2_inventory=native / "grib2_inventory.exe",
            grib2_dump=native / "grib2_dump.exe",
            gfs_bridge=native / "gfs_grib2_bridge.exe",
            hrrr_bridge=native / "hrrr_grib2_bridge.exe",
            cpu_backend=native / "gpuwm_preprocess_cpu.dll",
            output_dir=output,
            archive=archive,
        )
        result = build_windows_distribution(distribution_args)

    result["build_interface"] = "tools/build_rw_wps_windows_release.py"
    result["rust_build"] = "cargo-release-locked-offline-path-remapped"
    result["python_build"] = "pip-wheel-no-build-isolation-no-deps"
    result["python_package_inventory"] = python_inventory
    result["source_date_epoch"] = source_date_epoch
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(
        build_windows_release(args),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
