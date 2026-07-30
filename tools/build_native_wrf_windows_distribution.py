#!/usr/bin/env python3
"""Build a hash-bound Windows x86_64 CPU RW-WPS runtime distribution."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import zipfile


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gpuwm import __version__  # noqa: E402
from gpuwm.native_wrf_distribution import (  # noqa: E402
    BRIDGE_NAMES,
    PYTHON_DISTRIBUTION,
    RUNTIME_SCHEMA,
    WINDOWS_CPU_BACKEND_LIBRARY,
    bridge_identity,
    cpu_backend_identity,
    distribution_contract,
)
from tools.build_native_wrf_distribution import (  # noqa: E402
    CONFIGS,
    PACKAGED_AUTHORITIES,
    _copy_file,
    _git,
    _payload_records,
    _require_tracked,
    _sha256,
    _verify_wheel_matches_source,
    _wheel_metadata,
)


_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_DYNAMIC_MSVC_RUNTIME_MARKERS = (
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "ucrtbase.dll",
    "api-ms-win-crt-",
)


def _dynamic_msvc_runtime_imports(path: Path) -> list[str]:
    """Return forbidden dynamic CRT import markers embedded in a PE."""

    payload = path.read_bytes().lower()
    return [
        marker for marker in _DYNAMIC_MSVC_RUNTIME_MARKERS
        if marker.encode("ascii") in payload
    ]


def _require_static_msvc_runtime(path: Path) -> list[str]:
    imports = _dynamic_msvc_runtime_imports(path)
    if imports:
        raise RuntimeError(
            f"Windows native artifact imports dynamic MSVC CRT: "
            f"{path}: {imports}")
    return imports


def _write_deterministic_zip(source: Path, archive: Path) -> None:
    """Write a byte-reproducible ZIP with one explicit top-level root."""

    temporary = archive.with_name(archive.name + f".partial-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    paths = [source, *sorted(source.rglob("*"))]
    with zipfile.ZipFile(
        temporary,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as output:
        for path in paths:
            relative = (Path(source.name) / path.relative_to(source)).as_posix()
            is_directory = path.is_dir()
            if not is_directory and not path.is_file():
                raise ValueError(
                    f"distribution contains a non-regular path: {path}")
            name = relative + ("/" if is_directory else "")
            info = zipfile.ZipInfo(name, date_time=_ZIP_TIMESTAMP)
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = 0o755 if is_directory else 0o644
            info.external_attr = ((0o040000 if is_directory else 0o100000) | mode) << 16
            info.flag_bits = 0x800
            output.writestr(info, b"" if is_directory else path.read_bytes())
    os.replace(temporary, archive)


def build_windows_distribution(args: argparse.Namespace) -> dict[str, object]:
    wheel = args.wheel.resolve()
    output = args.output_dir.resolve()
    archive = args.archive.resolve()
    if output.exists() or archive.exists():
        raise FileExistsError("output directory and archive must not exist")
    if archive.suffix.lower() != ".zip":
        raise ValueError("Windows RW-WPS archive must use a .zip suffix")
    if platform.system() != "Windows" or platform.machine() not in {
            "x86_64", "AMD64"}:
        raise RuntimeError(
            "Windows RW-WPS distribution must be built on Windows x86_64")
    if _git("status", "--porcelain"):
        raise RuntimeError("refusing to package a dirty source tree")

    commit = _git("rev-parse", "HEAD^{commit}")
    tree = _git("rev-parse", "HEAD^{tree}")
    name, version, requirements = _wheel_metadata(wheel)
    normalized_name = name.lower().replace("_", "-")
    if normalized_name != PYTHON_DISTRIBUTION or version != __version__:
        raise ValueError(
            f"wheel identity mismatch: name={name!r} version={version!r}, "
            f"expected {PYTHON_DISTRIBUTION} {__version__}")
    wheel_source = _verify_wheel_matches_source(wheel)

    staging = output.with_name(output.name + f".partial-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    _copy_file(wheel, staging / "wheel" / wheel.name)

    bridge_inputs = {
        "grib1_bridge": args.grib1_bridge.resolve(),
        "grib2_inventory": args.grib2_inventory.resolve(),
        "grib2_dump": args.grib2_dump.resolve(),
        "gfs_grib2_bridge": args.gfs_bridge.resolve(),
        "hrrr_grib2_bridge": args.hrrr_bridge.resolve(),
        "rw_fetch": args.rw_fetch.resolve(),
    }
    bridge_build_identity = {}
    for bridge_name in BRIDGE_NAMES:
        source = bridge_inputs[bridge_name]
        dynamic_crt_imports = _require_static_msvc_runtime(source)
        identity = bridge_identity(source, bridge_name)
        if identity["binary_format"] != "pe":
            raise ValueError(f"Windows bridge is not PE: {source}")
        identity["path"] = source.name
        identity["dynamic_msvc_runtime_imports"] = dynamic_crt_imports
        bridge_build_identity[bridge_name] = identity
        _copy_file(
            source,
            staging / "libexec" / "bridges" / f"{bridge_name}.exe",
        )

    cpu_backend_source = args.cpu_backend.resolve()
    dynamic_crt_imports = _require_static_msvc_runtime(cpu_backend_source)
    cpu_backend_build_identity = cpu_backend_identity(cpu_backend_source)
    if cpu_backend_build_identity["binary_format"] != "pe":
        raise ValueError(f"Windows CPU backend is not PE: {cpu_backend_source}")
    cpu_backend_build_identity["path"] = cpu_backend_source.name
    cpu_backend_build_identity[
        "dynamic_msvc_runtime_imports"] = dynamic_crt_imports
    _copy_file(
        cpu_backend_source,
        staging / "libexec" / "bridges" / WINDOWS_CPU_BACKEND_LIBRARY,
    )

    installer = REPO / "tools" / "install_gpuwm_native_wrf_windows.ps1"
    launcher = REPO / "tools" / "gpuwm_native_wrf_launcher_windows.ps1"
    command_launcher = REPO / "tools" / "gpuwm_native_wrf_launcher_windows.cmd"
    docs = (
        REPO / "docs" / "native-wrf-distribution.md",
        REPO / "docs" / "native-mapped-source-authoring.md",
    )
    for source in (installer, launcher, command_launcher, *docs):
        _require_tracked(source)
    _copy_file(installer, staging / "install.ps1")
    for name in ("rw-wps.ps1", "gpuwm-wrf-init.ps1"):
        _copy_file(launcher, staging / "bin" / name)
    for name in ("rw-wps.cmd", "gpuwm-wrf-init.cmd"):
        _copy_file(command_launcher, staging / "bin" / name)
    for source in docs:
        _copy_file(source, staging / "share" / "docs" / source.name)
    for config_name in CONFIGS:
        source = REPO / "configs" / config_name
        _require_tracked(source)
        _copy_file(source, staging / "share" / "configs" / config_name)
    for authority_name in PACKAGED_AUTHORITIES:
        source = REPO / "gpuwm" / "authorities" / authority_name
        _require_tracked(source)
        _copy_file(source, staging / "share" / "configs" / authority_name)

    manifest = {
        "schema": RUNTIME_SCHEMA,
        "status": "READY",
        "artifact": {
            "name": output.name,
            "platform": "windows-x86_64-cpu",
            "gpuwm_version": version,
            "python_distribution": normalized_name,
            "wheel": wheel.name,
            "wheel_requires_dist": requirements,
        },
        "source": {
            "commit": commit,
            "tree": tree,
            "worktree_clean": True,
            "tracked_worktree_clean": True,
            "wheel_source_match": wheel_source,
        },
        "bridge_build_identity": bridge_build_identity,
        "cpu_backend_build_identity": cpu_backend_build_identity,
        "contract": distribution_contract("windows-x86_64"),
        "payload": _payload_records(staging),
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    sums = _payload_records(staging)
    (staging / "SHA256SUMS").write_text(
        "".join(
            f"{record['sha256']}  {name}\n"
            for name, record in sorted(sums.items())
        ),
        encoding="ascii",
        newline="\n",
    )
    os.replace(staging, output)
    _write_deterministic_zip(output, archive)
    return {
        "schema": RUNTIME_SCHEMA,
        "status": "PASS",
        "platform": "windows-x86_64-cpu",
        "output_dir": str(output),
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": _sha256(archive),
        "manifest_sha256": _sha256(output / "manifest.json"),
        "sha256s_sha256": _sha256(output / "SHA256SUMS"),
        "source_commit": commit,
        "source_tree": tree,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--grib1-bridge", type=Path, required=True)
    parser.add_argument("--grib2-inventory", type=Path, required=True)
    parser.add_argument("--grib2-dump", type=Path, required=True)
    parser.add_argument("--gfs-bridge", type=Path, required=True)
    parser.add_argument("--hrrr-bridge", type=Path, required=True)
    parser.add_argument("--rw-fetch", type=Path, required=True)
    parser.add_argument("--cpu-backend", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(
        build_windows_distribution(args),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
