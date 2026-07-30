"""Distribution contract and fail-closed runtime check for native WRF input.

The dedicated ``rw-wps`` wheel contains preprocessing/export modules and the
installed HRRR orchestration helpers, not gpuwm's forecast executor.  A
versioned Linux or Windows runtime bundle supplies the platform Rust bridges
and a launcher.  Source GRIBs, static geography, namelists, and stock ``wrf.exe``
remain explicit, hash-bound case inputs rather than package data.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any

from gpuwm import __version__


RUNTIME_SCHEMA = "gpuwm-native-wrf-runtime-v1"
PYTHON_DISTRIBUTION = "rw-wps"
BRIDGE_NAMES = (
    "grib1_bridge",
    "grib2_inventory",
    "grib2_dump",
    "gfs_grib2_bridge",
    "hrrr_grib2_bridge",
    # Built from tools/rustwx, not tools/grib1_bridge -- the vendored
    # wx-core download stack and its offline vendor closure live there.
    # The distribution takes every bridge by explicit path, so the two
    # cargo workspaces cost nothing here beyond two build commands.
    "rw_fetch",
)
CPU_BACKEND_LIBRARY = "libgpuwm_preprocess_cpu.so"
WINDOWS_CPU_BACKEND_LIBRARY = "gpuwm_preprocess_cpu.dll"
HRRR_HELPERS = (
    "download_hrrr_native_subset.py",
    "prepare_hrrr_cpu_wrf.sh",
    "prepare_hrrr_domain_cpu_wrf.sh",
    "prepare_hrrr_500_native.sh",
    "prepare_hrrr_wrf.py",
    "hrrr_build_native_static.py",
    "hrrr_pipeline.py",
    "hrrr_single_domain_benchmark.py",
    "seal_hrrr_native_bridge.py",
    "write_hrrr_native_geometry_receipt.py",
)
CUDA_KERNEL_SOURCES = (
    "common.cuh",
    "diagnostics.cu",
    "lbc_flow.cu",
    "lbc_state.cu",
    "spec_bdy.cu",
    "vert_interp.cu",
)

_MINIMUM_VERSIONS = {
    "numpy": (1, 26),
    "netCDF4": (1, 6),
    "cupy-cuda12x": (13, 0),
}
_BRIDGE_USAGE_MARKERS = {
    "grib1_bridge": "usage: grib1_bridge",
    "grib2_inventory": "usage: grib2_inventory",
    "grib2_dump": "usage: grib2_dump",
    "gfs_grib2_bridge": "usage: gfs_grib2_bridge",
    "hrrr_grib2_bridge": "usage: hrrr_grib2_bridge",
    "rw_fetch": "usage: rw_fetch",
}
# The generic GRIB2 pair is an internal tabular ABI, not just any executable
# with the right basename and usage string.  These adjacent-column markers
# deliberately cover fields consumed unconditionally by mapped_source.py.
# A stale bridge without ``member`` previously passed the usage-only check and
# then failed after a distribution had already been built and installed.
_BRIDGE_ABI_MARKERS = {
    "grib2_inventory": (
        b"parameter\tcenter\tsubcenter\tmaster_table_version\t"
        b"local_table_version\tname"
    ),
    "grib2_dump": (
        b"parameter\tcenter\tsubcenter\tmaster_table_version\t"
        b"local_table_version\tlevel_type"
    ),
    # The fetch record is a JSON ABI rather than a tabular one, but the
    # failure mode is identical: a stale rw_fetch that still prints its
    # usage line while emitting a record missing, say, ``mode_reason``
    # would pass a usage-only probe and then break the manifest author
    # after a distribution had already been built and installed.
    "rw_fetch": (
        b"gpuwm-rw-fetch-record-v1\tmode\tmode_reason\tsource\tgrib_url\t"
        b"idx_url\tidx_sha256\tidx_record_count\tselected_record_count\t"
        b"ranges\tsha256"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _native_platform_name() -> str:
    system = platform.system()
    machine = platform.machine()
    if machine not in {"x86_64", "AMD64"}:
        raise RuntimeError(f"unsupported native-WRF architecture: {machine}")
    if system == "Linux":
        return "linux-x86_64"
    if system == "Windows":
        return "windows-x86_64"
    raise RuntimeError(f"unsupported native-WRF operating system: {system}")


def distribution_contract(
        target_platform: str | None = None) -> dict[str, Any]:
    """Return the machine-readable install and scientific-control contract."""

    supported_platforms = {
        "linux-x86_64": {
            "operating_system": "Linux",
            "architecture": "x86_64",
            "shell": "bash",
            "python": ">=3.11",
            "cuda_runtime_family": "12.x",
            "backends": ["cpu", "cuda"],
            "hrrr_shell_pipeline": True,
        },
        "windows-x86_64": {
            "operating_system": "Windows",
            "architecture": "x86_64",
            "shell": "PowerShell 5.1+",
            "python": ">=3.11",
            "cuda_runtime_family": None,
            "backends": ["cpu"],
            "hrrr_shell_pipeline": False,
            "msvc_runtime": "statically linked",
            "status": (
                "sealed CPU runtime; public CLI/introspection and pure-"
                "Python/Rust preprocessing paths"
            ),
        },
    }
    selected_name = target_platform or _native_platform_name()
    if selected_name not in supported_platforms:
        raise ValueError(
            f"unsupported native-WRF target platform: {selected_name}")
    selected_platform = {
        "identifier": selected_name,
        **supported_platforms[selected_name],
    }
    libraries_by_platform = {
        "linux-x86_64": CPU_BACKEND_LIBRARY,
        "windows-x86_64": WINDOWS_CPU_BACKEND_LIBRARY,
    }
    return {
        "schema": RUNTIME_SCHEMA,
        "gpuwm_version": __version__,
        "python_distribution": PYTHON_DISTRIBUTION,
        "platform": selected_platform,
        "supported_platforms": supported_platforms,
        "python_dependencies": {
            "required": {
                "numpy": ">=1.26",
                "netCDF4": ">=1.6",
            },
            "gpu_extra": {"cupy-cuda12x": ">=13.0"},
        },
        "bundled_native_bridges": list(BRIDGE_NAMES),
        "bundled_native_libraries": [libraries_by_platform[selected_name]],
        "bundled_native_libraries_by_platform": libraries_by_platform,
        "preprocess_backends": {
            "cuda": "cupy-fp32-v1",
            "cpu": "rust-scoped-threads-fp32-v1",
            "parity_contract": "gpuwm-preprocess-backend-parity-v1",
        },
        "runtime_forbidden": ["WPS", "real.exe"],
        "external_case_inputs": [
            "source GRIB payloads and their SHA-256 manifest",
            "a filename-bound exact-member manifest for 20CRv3",
            "WPS_GEOG or a sealed native static cache and receipt",
            "source-specific Vtable/orography where required",
            "hash-bound WPS geometry and gpuwm experiment configuration",
            "a supported WRF namelist for the HRRR route",
        ],
        "public_controls": {
            "hrrr": {
                "source": "ordered f00..f12 wrfnat+soil GRIB2 pairs",
                "cadence_seconds": 3600,
                "domain": (
                    "one CONUS Lambert gpuwm-hrrr-target-domain-v1; target "
                    "nz must match explicit WRF e_vert-1; positive dx=dy; "
                    "spec width 5=1+4; complete HRRR halo"
                ),
                "physics": "WSM6 + YSU + classic-MM5 option 91 + Noah",
                "mutable": [
                    "validated Lambert center/extent/resolution",
                    "positive integer target timestep in seconds",
                    "run-seconds within source coverage",
                    "pipeline workers 1..13",
                    "positive preparation worker count",
                ],
            },
            "era5": {
                "source": "uniform combined GRIB1 series beginning at f00",
                "domain": (
                    "one static one-way Lambert hierarchy; WPS geometry must "
                    "match every experiment domain; nested initialization "
                    "requires WPS_GEOG and uses source invariant SOILGEO or "
                    "an exact per-domain orography declaration; "
                    "explicit strictly decreasing eta grid; hybrid_opt=2; "
                    "model top covered by the source pressure levels"
                ),
                "physics": "WSM6 + YSU + classic-MM5 option 91 + Noah",
                "preprocessing": (
                    "CUDA default; deterministic parallel CPU or automatic "
                    "selection; positive explicit CPU worker count"
                ),
            },
            "gfs": {
                "source": (
                    "uniform pgrb2.0p25 series beginning at f000; one- or "
                    "three-hour cadence; complete 1000..100-hPa and Noah soil"
                ),
                "domain": (
                    "one static one-way Lambert hierarchy; WPS geometry must "
                    "match every experiment domain; nested initialization "
                    "requires WPS_GEOG; "
                    "explicit strictly decreasing eta grid; hybrid_opt=2; "
                    "model top no higher than the 100-hPa source top"
                ),
                "physics": "WSM6 + YSU + classic-MM5 option 91 + Noah",
                "preprocessing": (
                    "CUDA default; deterministic parallel CPU or automatic "
                    "selection; positive explicit CPU worker count"
                ),
            },
            "20crv3": {
                "source": (
                    "one filename-identified every-member pressure/surface "
                    "GRIB2 series with an exact custom SHA-256 manifest"
                ),
                "authorities": (
                    "immutable mapping, composition, and provenance JSON "
                    "packaged in the wheel and runtime archive"
                ),
                "domain": (
                    "one-way Lambert hierarchy through the packaged mapping's "
                    "declared max_dom=4; unchanged-stock-WRF gate pending"
                ),
                "preprocessing": (
                    "CUDA default; deterministic parallel CPU or automatic "
                    "selection; bounded child initialization workers"
                ),
            },
            "mapped": {
                "source": (
                    "strict rw-wps.mapping.v1 plus hash-bound composition, "
                    "primary/supplement/provenance inventories, and GRIB1, "
                    "GRIB2, or NetCDF decoder identity"
                ),
                "domain": (
                    "one-way Lambert hierarchy up to the mapping's max_dom; "
                    "one shared explicit eta coordinate; exact WPS/experiment "
                    "topology agreement"
                ),
                "physics": "WSM6 + YSU + classic-MM5 option 91 + Noah",
                "preprocessing": (
                    "CUDA default; deterministic parallel CPU or automatic "
                    "selection; bounded child initialization workers"
                ),
                "authoring": (
                    "explicit source-family-independent descriptor/Vtable "
                    "compilation plus create-only exact input manifests; "
                    "validated mappings do not inherit stock-WRF certification"
                ),
                "real_gates": (
                    "ERA5 GRIB1, GFS GRIB2, ERA5 NetCDF single domains plus "
                    "GFS GRIB2 d01 through d04 accepted by unchanged WRF v4.6.1"
                ),
            },
        },
    }


def _numeric_version(value: str) -> tuple[int, ...]:
    numbers = tuple(int(item) for item in re.findall(r"\d+", value))
    if not numbers:
        raise RuntimeError(f"dependency has an unparseable version: {value!r}")
    return numbers


def cpu_backend_library_name(system: str | None = None) -> str:
    """Return the release CPU-library basename for one operating system."""

    observed = platform.system() if system is None else system
    if observed == "Linux":
        return CPU_BACKEND_LIBRARY
    if observed == "Windows":
        return WINDOWS_CPU_BACKEND_LIBRARY
    raise RuntimeError(f"unsupported native-WRF runtime system: {observed}")


def _bridge_status_identity(value: os.stat_result) -> tuple[int, ...]:
    """Return a stable open-handle/path identity on POSIX and Windows."""

    mode = stat.S_IFMT(value.st_mode) if os.name == "nt" else value.st_mode
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        mode,
    )


def _bridge_binary_format(payload: bytes, path: Path) -> str:
    if payload[:4] == b"\x7fELF":
        return "elf"
    if payload[:2] == b"MZ" and len(payload) >= 0x40:
        pe_offset = int.from_bytes(payload[0x3c:0x40], "little")
        if pe_offset <= len(payload) - 4 \
                and payload[pe_offset:pe_offset + 4] == b"PE\0\0":
            return "pe"
    raise RuntimeError(
        f"native bridge is not an ELF or PE executable: {path}"
    )


def bridge_identity(path: Path, name: str) -> dict[str, Any]:
    """Prove a bundled bridge is the expected native executable."""

    path = Path(path).resolve()
    if name not in _BRIDGE_USAGE_MARKERS:
        raise ValueError(f"unknown native bridge identity: {name}")
    if not path.is_file():
        raise FileNotFoundError(f"native bridge is missing: {path}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) \
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"native bridge is not a regular file: {path}")
        payload = stream.read()
        after = os.fstat(stream.fileno())
    observed = _bridge_status_identity(after)
    current = path.stat()
    if observed != _bridge_status_identity(before) \
            or observed != _bridge_status_identity(current):
        raise RuntimeError(f"native bridge changed while identifying it: {path}")
    binary_format = _bridge_binary_format(payload, path)
    if binary_format == "elf" and not os.access(path, os.X_OK):
        raise FileNotFoundError(
            f"native bridge is not executable: {path}")
    abi_marker = _BRIDGE_ABI_MARKERS.get(name)
    if abi_marker is not None and abi_marker not in payload:
        raise RuntimeError(
            f"native bridge has an incompatible tabular ABI: {path}")
    with tempfile.TemporaryDirectory(prefix=f"gpuwm-{name}-identity-") as directory:
        frozen = Path(directory) / (
            f"{name}.exe" if binary_format == "pe" else name
        )
        frozen.write_bytes(payload)
        frozen.chmod(after.st_mode)
        try:
            completed = subprocess.run(
                [str(frozen)],
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"native bridge identity probe timed out: {path}"
            ) from error
        except OSError as error:
            raise RuntimeError(
                "native bridge frozen-copy identity probe could not execute; "
                f"the temporary filesystem may be mounted noexec: {path}"
            ) from error
    output = completed.stdout + completed.stderr
    marker = _BRIDGE_USAGE_MARKERS[name]
    if completed.returncode == 0 or marker not in output.lower():
        raise RuntimeError(
            f"native bridge failed its no-argument identity check: {path}")
    with path.open("rb") as stream:
        final_status = os.fstat(stream.fileno())
        final_payload = stream.read()
    final_current = path.stat()
    if final_payload != payload \
            or _bridge_status_identity(final_status) != observed \
            or _bridge_status_identity(final_current) != observed:
        raise RuntimeError(f"native bridge changed during identity probe: {path}")
    result = {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "binary_format": binary_format,
        "no_arg_returncode": completed.returncode,
        "usage_output_sha256": hashlib.sha256(
            output.encode("utf-8")).hexdigest(),
    }
    if abi_marker is not None:
        result["tabular_abi_marker_sha256"] = hashlib.sha256(
            abi_marker).hexdigest()
    return result


def cpu_backend_identity(path: Path) -> dict[str, Any]:
    """Load and identify the bundled deterministic CPU transform library."""

    from gpuwm.ingest.cpu_backend import (
        CPU_BACKEND_ABI,
        CpuPreprocessBackend,
    )

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"native CPU preprocessing backend is missing: {path}")
    payload = path.read_bytes()
    binary_format = _bridge_binary_format(payload, path)
    expected_format = "pe" if platform.system() == "Windows" else "elf"
    if binary_format != expected_format:
        raise RuntimeError(
            "native CPU preprocessing backend format does not match host: "
            f"{binary_format} != {expected_format}: {path}")
    backend = CpuPreprocessBackend(path)
    try:
        return {
            "path": str(path),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "binary_format": binary_format,
            "abi": CPU_BACKEND_ABI,
            "backend": backend.name,
            "arithmetic": backend.arithmetic,
            "self_test": _cpu_backend_self_test(backend),
        }
    finally:
        backend.close()


def _cpu_backend_self_test(backend: Any) -> dict[str, Any]:
    """Execute a deterministic native transform, not just an ABI probe.

    Loading a shared library and reading its ABI symbol does not prove its
    worker path can consume and produce FP32 arrays on the installation host.
    The clean CPU installer therefore performs the smallest useful bilinear
    interpolation with both one and three workers and binds the output bytes.
    No CuPy import or CUDA device is involved.
    """

    import numpy as np

    source = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    index = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    plan = backend.indexed_plan(source.shape, index, index)
    serial = np.ascontiguousarray(
        plan.apply(source, method="bilinear", workers=1), dtype=np.float32)
    parallel = np.ascontiguousarray(
        plan.apply(source, method="bilinear", workers=3), dtype=np.float32)
    expected = np.array([[1.0, 2.5, 4.0]], dtype=np.float32)
    if not np.array_equal(serial, expected):
        raise RuntimeError(
            "CPU preprocessing self-test produced unexpected bilinear values: "
            f"{serial.tolist()} != {expected.tolist()}")
    if not np.array_equal(parallel, serial):
        raise RuntimeError(
            "CPU preprocessing self-test changed with worker count")
    return {
        "status": "PASS",
        "operation": "indexed_bilinear_fp32",
        "worker_counts": [1, 3],
        "output_values": serial.tolist(),
        "output_sha256": hashlib.sha256(serial.tobytes()).hexdigest(),
    }


def _installed_record_receipt() -> dict[str, Any]:
    dist = metadata.distribution(PYTHON_DISTRIBUTION)
    observed_version = dist.version
    if observed_version != __version__:
        raise RuntimeError(
            f"{PYTHON_DISTRIBUTION} version mismatch: "
            f"metadata={observed_version}, "
            f"module={__version__}")
    checked: list[tuple[str, str]] = []
    for item in dist.files or ():
        expected = item.hash
        if expected is None:
            continue
        if expected.mode != "sha256":
            raise RuntimeError(
                f"unsupported wheel RECORD digest {expected.mode!r} for {item}")
        path = Path(dist.locate_file(item))
        if not path.is_file():
            raise FileNotFoundError(f"installed wheel file is missing: {path}")
        actual = _sha256(path)
        encoded = base64.urlsafe_b64encode(bytes.fromhex(actual)).decode().rstrip("=")
        if encoded != expected.value:
            raise RuntimeError(f"installed wheel RECORD mismatch for {item}")
        checked.append((str(item).replace("\\", "/"), actual))
    if not checked:
        raise RuntimeError(
            f"installed {PYTHON_DISTRIBUTION} distribution has no hashed "
            "RECORD files"
        )
    aggregate = hashlib.sha256()
    for name, digest in sorted(checked):
        aggregate.update(name.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\n")
    return {
        "distribution_name": PYTHON_DISTRIBUTION,
        "distribution_version": observed_version,
        "record_file_count": len(checked),
        "record_aggregate_sha256": aggregate.hexdigest(),
    }


def _runtime_dependency_modules(require_gpu: bool) -> dict[str, str]:
    modules = {
        "numpy": "numpy",
        "netCDF4": "netCDF4",
    }
    if require_gpu:
        modules["cupy-cuda12x"] = "cupy"
    return modules


def verify_runtime(
    *, bridge_dir: Path, require_gpu: bool = True,
) -> dict[str, Any]:
    """Verify an installed wheel, helper scripts, bridges, and CUDA runtime."""

    system = platform.system()
    machine = platform.machine()
    if system not in {"Linux", "Windows"} or machine not in {
            "x86_64", "AMD64"}:
        raise RuntimeError(
            "native-WRF runtime requires Linux or Windows x86_64")
    if system == "Windows" and require_gpu:
        raise RuntimeError(
            "the sealed Windows x86_64 runtime is CPU-only; use --skip-gpu")
    if sys.version_info < (3, 11):
        raise RuntimeError("native-WRF runtime requires Python 3.11 or newer")
    bash = shutil.which("bash") if system == "Linux" else None
    if system == "Linux" and bash is None:
        raise FileNotFoundError("bash is required by the installed HRRR pipeline")

    dependencies: dict[str, str] = {}
    dependency_modules = _runtime_dependency_modules(require_gpu)
    for package, module_name in dependency_modules.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            raise RuntimeError(
                f"required runtime module is missing: {module_name}") from error
        dependencies[package] = str(getattr(module, "__version__", ""))
        if _numeric_version(dependencies[package]) < _MINIMUM_VERSIONS[package]:
            minimum = ".".join(str(value) for value in _MINIMUM_VERSIONS[package])
            raise RuntimeError(
                f"{package} {dependencies[package]} is older than {minimum}")

    # Import public entrypoint modules explicitly.  Dependency-only imports do
    # not prove the wheel contains the application modules an installed
    # ``rw-wps`` launcher will execute.
    for module_name in (
        "gpuwm.mapped_authoring",
        "gpuwm.source_authorities",
        "gpuwm.source_cli",
        "gpuwm.twentycrv3_direct",
        "gpuwm.twentycrv3_wrf",
    ):
        try:
            importlib.import_module(module_name)
        except ImportError as error:
            raise RuntimeError(
                f"required runtime entrypoint module is missing: {module_name}"
            ) from error

    bridge_dir = Path(bridge_dir).resolve()
    bridges: dict[str, dict[str, Any]] = {}
    bridge_suffix = ".exe" if system == "Windows" else ""
    for name in BRIDGE_NAMES:
        bridges[name] = bridge_identity(
            bridge_dir / f"{name}{bridge_suffix}", name)
    cpu_backend = cpu_backend_identity(
        bridge_dir / cpu_backend_library_name(system))

    tools_root = Path(__file__).resolve().parent.parent / "tools"
    helpers: dict[str, str] = {}
    for name in HRRR_HELPERS:
        path = tools_root / name
        if not path.is_file():
            raise FileNotFoundError(f"installed HRRR helper is missing: {path}")
        helpers[name] = _sha256(path)

    kernel_root = Path(__file__).resolve().parent / "core" / "kernels"
    observed_kernels = {
        path.name for path in kernel_root.iterdir()
        if path.is_file() and path.suffix in {".cu", ".cuh"}
    }
    if observed_kernels != set(CUDA_KERNEL_SOURCES):
        raise RuntimeError(
            "installed CUDA kernel-source inventory differs from contract: "
            f"missing={sorted(set(CUDA_KERNEL_SOURCES) - observed_kernels)}, "
            f"extra={sorted(observed_kernels - set(CUDA_KERNEL_SOURCES))}")
    kernels = {
        name: _sha256(kernel_root / name) for name in CUDA_KERNEL_SOURCES}

    gpu: dict[str, Any] = {"required": require_gpu, "status": "NOT_CHECKED"}
    if require_gpu:
        import cupy as cp

        count = int(cp.cuda.runtime.getDeviceCount())
        if count < 1:
            raise RuntimeError("CuPy reports no CUDA devices")
        device = cp.cuda.Device(0)
        with device:
            value = cp.arange(16, dtype=cp.float32)
            checksum = float(cp.asnumpy(value.sum()))
            properties = cp.cuda.runtime.getDeviceProperties(0)
            runtime_version = int(cp.cuda.runtime.runtimeGetVersion())
        if not 12_000 <= runtime_version < 13_000:
            raise RuntimeError(
                "native-WRF runtime requires a CUDA 12.x runtime, got "
                f"{runtime_version}")
        raw_name = properties.get("name", b"")
        if isinstance(raw_name, bytes):
            raw_name = raw_name.decode("utf-8", errors="replace")
        gpu = {
            "required": True,
            "status": "PASS",
            "device_count": count,
            "device_0_name": str(raw_name),
            "driver_version": int(cp.cuda.runtime.driverGetVersion()),
            "runtime_version": runtime_version,
            "test_checksum": checksum,
        }

    return {
        "schema": RUNTIME_SCHEMA,
        "status": "PASS",
        "contract": distribution_contract(),
        "platform_observed": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "python_executable": str(Path(sys.executable).resolve()),
            "bash": bash,
        },
        "dependencies": dependencies,
        "installed_wheel": _installed_record_receipt(),
        "bridges": bridges,
        "cpu_preprocess_backend": cpu_backend,
        "hrrr_helpers": helpers,
        "cuda_kernel_sources": kernels,
        "gpu": gpu,
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite runtime receipt: {path}")
    temporary = path.with_name(path.name + f".partial-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-dir", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--contract", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    args = parser.parse_args(argv)
    if args.contract:
        if args.bridge_dir is not None or args.receipt is not None:
            parser.error("--contract cannot be combined with runtime verification inputs")
        print(json.dumps(distribution_contract(), indent=2, sort_keys=True))
        return 0
    if args.bridge_dir is None:
        parser.error("--bridge-dir is required unless --contract is used")
    result = verify_runtime(
        bridge_dir=args.bridge_dir, require_gpu=not args.skip_gpu)
    if args.receipt is not None:
        _atomic_json(args.receipt, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
