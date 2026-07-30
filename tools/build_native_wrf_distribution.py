#!/usr/bin/env python3
"""Build a hash-bound Linux gpuwm native-WRF runtime distribution."""

from __future__ import annotations

import argparse
from email.parser import Parser
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gpuwm import __version__  # noqa: E402
from gpuwm.native_wrf_distribution import (  # noqa: E402
    BRIDGE_NAMES,
    CPU_BACKEND_LIBRARY,
    CUDA_KERNEL_SOURCES,
    HRRR_HELPERS,
    PYTHON_DISTRIBUTION,
    RUNTIME_SCHEMA,
    bridge_identity,
    cpu_backend_identity,
    distribution_contract,
)


CONFIGS = (
    "era5_wrf_direct_proof.toml",
    "gfs_wrf_direct_proof.toml",
    "gfs_wrf_hierarchy_proof.namelist.input",
    "gfs_wrf_hierarchy_proof.namelist.wps",
    "gfs_wrf_hierarchy_proof.toml",
    "hrrr_target_ohio_192x160_3km.json",
    "hrrr_target_oklahoma_192x160_3km.json",
    "hrrr_target_oklahoma_1000x1000_1km.json",
    "rw-wps-era5-1974-probe.mapping.json",
    "rw-wps-era5-1974-terrain.composition.json",
    "rw-wps-era5-netcdf.mapping.json",
    "rw-wps-era5-netcdf-terrain.composition.json",
    "rw-wps-gfs-pressure-grib2.mapping.json",
    "rw-wps-gfs-terrain.composition.json",
)
PACKAGED_AUTHORITIES = (
    "rw-wps-20crv3-member-grib2.mapping.json",
    "rw-wps-20crv3-member-grib2.composition.json",
    "rw-wps-20crv3-member-grib2.provenance.json",
)
REQUIRED_WHEEL_PAYLOADS = {
    "gpuwm/era5_direct.py",
    "gpuwm/gfs_direct.py",
    "gpuwm/ingest/backend_contract.py",
    "gpuwm/ingest/cpu_backend.py",
    "gpuwm/ingest/preprocess_backend.py",
    "gpuwm/hrrr_hierarchy_direct.py",
    "gpuwm/hrrr_native_static.py",
    "gpuwm/mapped_composition.py",
    "gpuwm/mapped_authoring.py",
    "gpuwm/mapped_direct.py",
    "gpuwm/mapped_source.py",
    "gpuwm/native_domain_artifacts.py",
    "gpuwm/native_hierarchy.py",
    "gpuwm/native_wrf_contract.py",
    "gpuwm/native_wrf_distribution.py",
    "gpuwm/source_cli.py",
    "gpuwm/source_authorities.py",
    "gpuwm/source_hierarchy.py",
    "gpuwm/twentycrv3.py",
    "gpuwm/twentycrv3_direct.py",
    "gpuwm/twentycrv3_wrf.py",
    "gpuwm/wrf_direct.py",
    "gpuwm/wrf_direct_v461_contract.json",
    "gpuwm/core/constants.py",
    "gpuwm/core/diagnostics.py",
    "gpuwm/core/grid.py",
    "gpuwm/core/landuse.py",
    "gpuwm/core/nest_interp.py",
    "gpuwm/core/noah.py",
    "gpuwm/core/state.py",
    "gpuwm/data/noah_tables/LANDUSE.TBL",
    "gpuwm/data/noah_tables/VEGPARM.TBL",
    "gpuwm/data/noah_tables/SOILPARM.TBL",
    "gpuwm/data/noah_tables/GENPARM.TBL",
} | {
    f"gpuwm/core/kernels/{name}" for name in CUDA_KERNEL_SOURCES
} | {
    f"gpuwm/authorities/{name}" for name in PACKAGED_AUTHORITIES
} | {
    f"tools/{name}" for name in HRRR_HELPERS
}
FORBIDDEN_WHEEL_PAYLOADS = {
    "gpuwm/cli.py",
    "gpuwm/runtime.py",
    "gpuwm/supervisor.py",
    "gpuwm/core/model.py",
    "gpuwm/core/dycore.py",
    "gpuwm/core/physics.py",
}
FORBIDDEN_WHEEL_PREFIXES = (
    "gpuwm/verify/",
    "gpuwm/data/rrtmgp/",
    "gpuwm/data/thompson/",
    "gpuwm/data/wrf_radiation/",
)

_PUBLIC_TEXT_SUFFIXES = {
    ".cfg", ".csv", ".f", ".f90", ".json", ".md", ".py", ".sh",
    ".tbl", ".toml", ".txt",
}
_PRIVATE_PATH_MARKERS = {
    b"c:/" + b"users/": "Windows user profile",
    b"c:\\" + b"users\\": "Windows user profile",
    b"/mnt/c/" + b"users/": "WSL-mounted user profile",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), *args], text=True).strip()


def _wheel_metadata(wheel: Path) -> tuple[str, str, list[str]]:
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist()
                 if name.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise ValueError("wheel must contain exactly one METADATA file")
        parsed = Parser().parsestr(archive.read(names[0]).decode("utf-8"))
    return (
        str(parsed["Name"]),
        str(parsed["Version"]),
        list(parsed.get_all("Requires-Dist", [])),
    )


def _wheel_public_path_violations(wheel: Path) -> list[dict[str, str]]:
    """Locate developer-specific absolute paths in installed text payloads."""

    violations = []
    with zipfile.ZipFile(wheel) as archive:
        for info in archive.infolist():
            if (
                info.is_dir()
                or Path(info.filename).suffix.lower() not in _PUBLIC_TEXT_SUFFIXES
            ):
                continue
            payload = archive.read(info).lower()
            for marker, label in _PRIVATE_PATH_MARKERS.items():
                if marker in payload:
                    violations.append({
                        "path": info.filename,
                        "marker": marker.decode("ascii"),
                        "kind": label,
                    })
    return violations


def _verify_wheel_matches_source(wheel: Path) -> dict[str, object]:
    compared: list[tuple[str, str]] = []
    private_paths = _wheel_public_path_violations(wheel)
    if private_paths:
        raise ValueError(
            "wheel contains developer-specific absolute paths: "
            f"{private_paths}"
        )
    with zipfile.ZipFile(wheel) as archive:
        wheel_names = set(archive.namelist())
        missing = sorted(REQUIRED_WHEEL_PAYLOADS - wheel_names)
        if missing:
            raise ValueError(f"wheel lacks required native-WRF payloads: {missing}")
        forbidden = sorted(FORBIDDEN_WHEEL_PAYLOADS & wheel_names)
        forbidden.extend(sorted(
            name for name in wheel_names
            if name.startswith(FORBIDDEN_WHEEL_PREFIXES)
        ))
        if forbidden:
            raise ValueError(
                f"RW-WPS wheel contains forecast-only payloads: {forbidden}"
            )
        entry_points = [name for name in wheel_names
                        if name.endswith(".dist-info/entry_points.txt")]
        if len(entry_points) != 1:
            raise ValueError("wheel must contain exactly one entry_points.txt")
        entry_text = archive.read(entry_points[0]).decode("utf-8")
        for expected in (
            "gpuwm-wrf-init = gpuwm.source_cli:main",
            "rw-wps = gpuwm.source_cli:main",
            "gpuwm-wrf-runtime-check = gpuwm.native_wrf_distribution:main",
        ):
            if expected not in entry_text:
                raise ValueError(f"wheel lacks entry point: {expected}")
        for info in archive.infolist():
            name = info.filename
            if name.endswith("/") or ".dist-info/" in name:
                continue
            source = REPO / Path(name)
            if not source.is_file():
                raise FileNotFoundError(
                    f"wheel payload has no matching source file: {name}")
            wheel_digest = hashlib.sha256(archive.read(info)).hexdigest()
            source_digest = _sha256(source)
            if wheel_digest != source_digest:
                raise ValueError(f"wheel/source mismatch for {name}")
            if name.startswith("tools/") and name.endswith(".sh") \
                    and b"\r" in archive.read(info):
                raise ValueError(f"wheel shell payload is not LF-only: {name}")
            compared.append((name, wheel_digest))
    if not compared:
        raise ValueError("wheel contains no source payloads")
    aggregate = hashlib.sha256()
    for name, digest in sorted(compared):
        aggregate.update(name.encode() + b"\0" + digest.encode() + b"\n")
    return {
        "file_count": len(compared),
        "aggregate_sha256": aggregate.hexdigest(),
    }


def _copy_file(source: Path, destination: Path, *, executable: bool = False) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    destination.chmod(0o755 if executable else 0o644)


def _require_tracked(source: Path) -> None:
    relative = source.resolve().relative_to(REPO).as_posix()
    subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "--error-unmatch", relative],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _require_lf_shell(source: Path) -> None:
    if b"\r" in source.read_bytes():
        raise ValueError(f"bundled shell script is not LF-only: {source}")


def _payload_records(root: Path) -> dict[str, dict[str, object]]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            relative = path.relative_to(root).as_posix()
            result[relative] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "executable": bool(path.stat().st_mode & 0o111),
            }
    return result


def _write_deterministic_tar(source: Path, archive: Path) -> None:
    temporary = archive.with_name(archive.name + f".partial-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open("wb") as raw:
        with gzip.GzipFile(
                filename="", fileobj=raw, mode="wb", mtime=0,
                compresslevel=9) as compressed:
            with tarfile.open(
                    fileobj=compressed, mode="w",
                    format=tarfile.PAX_FORMAT) as tar:
                for path in [source, *sorted(source.rglob("*"))]:
                    relative = Path(source.name) / path.relative_to(source)
                    info = tarfile.TarInfo(relative.as_posix())
                    info.uid = info.gid = 0
                    info.uname = info.gname = "root"
                    info.mtime = 0
                    if path.is_dir():
                        info.type = tarfile.DIRTYPE
                        info.mode = 0o755
                    elif path.is_file():
                        info.type = tarfile.REGTYPE
                        info.mode = (
                            0o755 if path.stat().st_mode & 0o111 else 0o644)
                        info.size = path.stat().st_size
                    else:
                        raise ValueError(
                            f"distribution contains a non-regular path: {path}")
                    if path.is_file():
                        with path.open("rb") as stream:
                            tar.addfile(info, stream)
                    else:
                        tar.addfile(info)
    os.replace(temporary, archive)


def build_distribution(args: argparse.Namespace) -> dict[str, object]:
    wheel = args.wheel.resolve()
    output = args.output_dir.resolve()
    archive = args.archive.resolve()
    if output.exists() or archive.exists():
        raise FileExistsError("output directory and archive must not exist")
    if platform.system() != "Linux" or platform.machine() not in {
            "x86_64", "AMD64"}:
        raise RuntimeError("native-WRF distribution must be built on Linux x86_64")
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
        identity = bridge_identity(source, bridge_name)
        identity["path"] = source.name
        bridge_build_identity[bridge_name] = identity
        _copy_file(
            source, staging / "libexec" / "bridges" / bridge_name,
            executable=True)
    cpu_backend_source = args.cpu_backend.resolve()
    cpu_backend_build_identity = cpu_backend_identity(cpu_backend_source)
    cpu_backend_build_identity["path"] = cpu_backend_source.name
    _copy_file(
        cpu_backend_source,
        staging / "libexec" / "bridges" / CPU_BACKEND_LIBRARY)
    installer = REPO / "tools" / "install_gpuwm_native_wrf.sh"
    launcher = REPO / "tools" / "gpuwm_native_wrf_launcher.sh"
    docs = (
        REPO / "docs" / "native-wrf-distribution.md",
        REPO / "docs" / "native-mapped-source-authoring.md",
    )
    for source in (installer, launcher, *docs):
        _require_tracked(source)
    for source in (installer, launcher):
        _require_lf_shell(source)
    _copy_file(
        installer,
        staging / "install.sh", executable=True)
    _copy_file(
        launcher,
        staging / "bin" / "gpuwm-wrf-init", executable=True)
    _copy_file(
        launcher,
        staging / "bin" / "rw-wps", executable=True)
    for source in docs:
        _copy_file(source, staging / "share" / "docs" / source.name)
    for name in CONFIGS:
        _require_tracked(REPO / "configs" / name)
        _copy_file(
            REPO / "configs" / name,
            staging / "share" / "configs" / name)
    for name in PACKAGED_AUTHORITIES:
        source = REPO / "gpuwm" / "authorities" / name
        _require_tracked(source)
        _copy_file(source, staging / "share" / "configs" / name)

    manifest = {
        "schema": RUNTIME_SCHEMA,
        "status": "READY",
        "artifact": {
            "name": output.name,
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
        "contract": distribution_contract("linux-x86_64"),
        "payload": _payload_records(staging),
    }
    manifest_path = staging / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    sums = _payload_records(staging)
    (staging / "SHA256SUMS").write_text(
        "".join(f"{record['sha256']}  {name}\n"
                for name, record in sorted(sums.items())),
        encoding="ascii")
    os.replace(staging, output)
    _write_deterministic_tar(output, archive)
    return {
        "schema": RUNTIME_SCHEMA,
        "status": "PASS",
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
        build_distribution(args), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
