#!/usr/bin/env python3
"""Clean-install an RW-WPS archive and exercise its CPU backend without CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> Path:
    """Extract only regular files/directories below one archive root."""

    roots: set[str] = set()
    with tarfile.open(archive, mode="r:gz") as source:
        members = source.getmembers()
        if not members:
            raise ValueError("RW-WPS archive is empty")
        for member in members:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe archive path: {member.name!r}")
            if not relative.parts or relative.parts[0] in {"", "."}:
                raise ValueError(f"invalid archive path: {member.name!r}")
            if not (member.isdir() or member.isfile()):
                raise ValueError(
                    f"archive contains a non-regular entry: {member.name!r}")
            roots.add(relative.parts[0])
        if len(roots) != 1:
            raise ValueError(
                f"RW-WPS archive must contain one top-level root, got {roots}")

        destination.mkdir(parents=True, exist_ok=False)
        destination_root = destination.resolve()
        for member in members:
            relative = PurePosixPath(member.name)
            target = destination.joinpath(*relative.parts)
            resolved = target.resolve()
            if destination_root not in resolved.parents and resolved != destination_root:
                raise ValueError(f"archive path escapes destination: {member.name!r}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = source.extractfile(member)
            if payload is None:
                raise ValueError(f"archive file has no payload: {member.name!r}")
            with payload, target.open("xb") as output:
                shutil.copyfileobj(payload, output)
            target.chmod(member.mode & 0o777)
    return destination / next(iter(roots))


def _safe_extract_zip(archive: Path, destination: Path) -> Path:
    """Extract a regular-file-only ZIP below one case-insensitive root."""

    roots: set[str] = set()
    seen: set[str] = set()
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        if not members:
            raise ValueError("RW-WPS archive is empty")
        for member in members:
            relative = PurePosixPath(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe archive path: {member.filename!r}")
            if not relative.parts or relative.parts[0] in {"", "."}:
                raise ValueError(f"invalid archive path: {member.filename!r}")
            if "\\" in member.filename or ":" in member.filename:
                raise ValueError(f"invalid archive path: {member.filename!r}")
            key = "/".join(relative.parts).casefold()
            if key in seen:
                raise ValueError(f"duplicate archive path: {member.filename!r}")
            seen.add(key)
            mode = (member.external_attr >> 16) & 0xFFFF
            kind = stat.S_IFMT(mode)
            if member.is_dir():
                if kind not in {0, stat.S_IFDIR}:
                    raise ValueError(
                        f"archive contains a non-directory entry: {member.filename!r}")
            elif kind not in {0, stat.S_IFREG}:
                raise ValueError(
                    f"archive contains a non-regular entry: {member.filename!r}")
            roots.add(relative.parts[0].casefold())
        if len(roots) != 1:
            raise ValueError(
                f"RW-WPS archive must contain one top-level root, got {roots}")

        destination.mkdir(parents=True, exist_ok=False)
        destination_root = destination.resolve()
        for member in members:
            relative = PurePosixPath(member.filename)
            target = destination.joinpath(*relative.parts)
            resolved = target.resolve()
            if destination_root not in resolved.parents and resolved != destination_root:
                raise ValueError(f"archive path escapes destination: {member.filename!r}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as payload, target.open("xb") as output:
                shutil.copyfileobj(payload, output)
    root_name = PurePosixPath(members[0].filename).parts[0]
    return destination / root_name


def _extract_archive(archive: Path, destination: Path) -> Path:
    if archive.suffix.lower() == ".zip":
        return _safe_extract_zip(archive, destination)
    return _safe_extract(archive, destination)


def _run(
    argv: list[str], *, cwd: Path, env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        argv, cwd=cwd, env=env, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        rendered = " ".join(argv)
        raise RuntimeError(
            f"clean-install command failed with {completed.returncode}: "
            f"{rendered}\nstdout:\n{completed.stdout[-8000:]}\n"
            f"stderr:\n{completed.stderr[-8000:]}")
    return completed


def _write_receipt(path: Path, value: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing existing receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def smoke_install(args: argparse.Namespace) -> dict[str, object]:
    archive = args.archive.resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    started = time.perf_counter()
    context = (
        tempfile.TemporaryDirectory(prefix="rw-wps-clean-install-")
        if args.work_dir is None else None
    )
    work = Path(context.name) if context is not None else args.work_dir.resolve()
    if context is None:
        work.mkdir(parents=True, exist_ok=False)
    try:
        extracted = _extract_archive(archive, work / "extracted")
        venv = work / "venv"
        _run([str(args.python), "-m", "venv", str(venv)], cwd=work)
        python = (
            venv / "Scripts" / "python.exe"
            if os.name == "nt" else venv / "bin" / "python"
        )
        dependency_command = [
            str(python), "-m", "pip", "install",
            "--disable-pip-version-check",
        ]
        if args.wheelhouse is not None:
            dependency_command.extend((
                "--no-index", "--find-links", str(args.wheelhouse.resolve())))
        dependency_command.extend(("numpy>=1.26", "netCDF4>=1.6"))
        _run(dependency_command, cwd=work)

        environment = os.environ.copy()
        environment["GPUWM_PYTHON"] = str(python)
        if os.name == "nt":
            powershell = shutil.which("powershell.exe")
            if powershell is None:
                raise FileNotFoundError("powershell.exe is required")
            install_command = [
                powershell,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(extracted / "install.ps1"),
                "-SkipGpu",
            ]
            launcher = [
                powershell,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(extracted / "bin" / "rw-wps.ps1"),
            ]
        else:
            install_command = [
                "bash", str(extracted / "install.sh"), "--skip-gpu"]
            launcher = [str(extracted / "bin" / "rw-wps")]
        install = _run(install_command, cwd=work, env=environment)
        version = _run(
            [*launcher, "--version"], cwd=work, env=environment)
        sources = json.loads(_run(
            [*launcher, "--list-sources"], cwd=work,
            env=environment).stdout)
        support = json.loads(_run(
            [*launcher, "--show-support-matrix"], cwd=work,
            env=environment).stdout)
        installed = extracted / "runtime"
        runtime_receipt = json.loads(
            (installed / "native-wrf-runtime-receipt.json").read_text(
                encoding="utf-8"))
        self_test = runtime_receipt["cpu_preprocess_backend"]["self_test"]
        if self_test.get("status") != "PASS":
            raise RuntimeError("installed CPU preprocessing self-test did not pass")
        optional = _run([
            str(python), "-c",
            "import importlib.util; "
            "assert importlib.util.find_spec('cupy') is None; "
            "assert importlib.util.find_spec('matplotlib') is None",
        ], cwd=work)
        freeze = _run(
            [str(python), "-m", "pip", "freeze", "--all"], cwd=work)
        manifest = json.loads(
            (extracted / "manifest.json").read_text(encoding="utf-8"))
        return {
            "schema": "rw-wps.clean-cpu-install.v1",
            "status": "PASS",
            "platform": {
                "system": platform.system(),
                "machine": platform.machine(),
            },
            "archive": {
                "bytes": archive.stat().st_size,
                "sha256": _sha256(archive),
            },
            "source": manifest["source"],
            "runtime": {
                "version_stdout": version.stdout.strip(),
                "source_count": sources["source_count"],
                "runnable_source_count": sources["runnable_source_count"],
                "support_schema": support["schema"],
                "cpu_preprocess_backend": runtime_receipt[
                    "cpu_preprocess_backend"],
                "gpu": runtime_receipt["gpu"],
                "cupy_and_matplotlib_absent": optional.returncode == 0,
                "installed_packages": sorted(
                    line for line in freeze.stdout.splitlines() if line),
            },
            "install_stdout": install.stdout.strip(),
            "outside_source_checkout": True,
            "wall_seconds": time.perf_counter() - started,
        }
    finally:
        if context is not None:
            context.cleanup()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args(argv)
    result = smoke_install(args)
    if args.receipt is not None:
        _write_receipt(args.receipt.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
