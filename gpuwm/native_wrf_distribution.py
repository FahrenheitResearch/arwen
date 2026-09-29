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
from dataclasses import dataclass
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
from gpuwm.bridges import BRIDGE_ABI_MARKERS as _SHARED_BRIDGE_ABI_MARKERS
from gpuwm.bridges import BRIDGE_ENV as _SHARED_BRIDGE_ENV
from gpuwm.bridges import CRATE_RELATIVE as _DECODER_WORKSPACE
from gpuwm.bridges import RUSTWX_CRATE_RELATIVE as _RENDERER_WORKSPACE
from gpuwm.bridges import quiet_loader_errors
from gpuwm.gpu_stack_identity import gpu_cuda_stack_identity
from gpuwm.runtime_manifest import RUNTIME_SCHEMA
# The fetch backbone's environment variable comes from the module that
# READS it, for the same reason the six decoders' come from
# :data:`gpuwm.bridges.BRIDGE_ENV`.  It cannot live in that map -- its
# consumers resolve through :func:`gpuwm.bridges.crate_dir`, the
# decoder crate, and rw_fetch builds in ``tools/rustwx`` -- so this is
# the one row whose authority is elsewhere, and naming it as a literal
# here would have been a second copy inside the declaration that exists
# to end second copies.
from gpuwm.rustwx_fetch import FETCH_ENV as _FETCH_BACKBONE_ENV
PYTHON_DISTRIBUTION = "rw-wps"


@dataclass(frozen=True)
class BundledBridge:
    """One executable the standalone rw-wps bundle carries, declared once.

    Everything a builder needs about a bridge is on the row: the cargo
    workspace that produces it, the option that names an already-built
    copy, the environment variable the installed launcher binds it to,
    the no-argument usage marker its identity probe demands, and the
    route that refuses without it.

    THE ROW IS THE WHOLE DECLARATION, because the shape it replaces cost
    a release.  A bridge used to be a name in a tuple plus a hand-written
    ``parser.add_argument`` plus a hand-written entry in each builder's
    ``bridge_inputs`` dict plus a hand-written ``export`` in each
    launcher -- five places, none of which any test held together.  Two
    of the five went stale: ``tools/build_rw_wps_release.py`` never
    passed ``rw_fetch`` at all, so the standalone release builder raised
    ``AttributeError`` before staging a byte, and both launchers bound
    five of the six bridges.  Then 2.7.5 added ``gdt101_remap`` to the
    bridge bundle for ICON global and the standalone bundle silently did
    not grow it, which shipped as a known limit with ``gpuwm doctor``
    naming the gap.

    A new bridge is now one row.  Nothing else in this file or in either
    builder is written per bridge, and
    ``tests/test_native_wrf_distribution.py`` fails when a row and the
    surfaces that must honour it disagree.
    """

    name: str
    workspace: str
    option: str
    env_var: str
    usage_marker: str
    consumer: str

    @property
    def dest(self) -> str:
        """The parsed-option attribute this row's path arrives on."""

        return self.option.removeprefix("--").replace("-", "_")


#: Every bridge the standalone rw-wps bundle carries.  The order is the
#: order the manifest and the contract report them in.
#:
#: ``rw_fetch`` builds from the renderer workspace and not the decoder
#: one -- the vendored wx-core download stack and its offline vendor
#: closure live there -- which is why the workspace is a column rather
#: than an assumption.  ``gdt101_remap`` is the icosahedral remapper the
#: ICON global route writes its regional intermediates with; it builds
#: beside the GRIB decoders in the same workspace.
BUNDLED_BRIDGES: tuple[BundledBridge, ...] = (
    BundledBridge(
        "grib1_bridge", _DECODER_WORKSPACE, "--grib1-bridge",
        _SHARED_BRIDGE_ENV["grib1_bridge"], "usage: grib1_bridge",
        "the ERA5 route (rw-wps --source era5)"),
    BundledBridge(
        "grib2_inventory", _DECODER_WORKSPACE, "--grib2-inventory",
        _SHARED_BRIDGE_ENV["grib2_inventory"], "usage: grib2_inventory",
        "every generic GRIB2 prep route (rw-wps --source gfs/20crv3/mapped)"),
    BundledBridge(
        "grib2_dump", _DECODER_WORKSPACE, "--grib2-dump",
        _SHARED_BRIDGE_ENV["grib2_dump"], "usage: grib2_dump",
        "every generic GRIB2 prep route (rw-wps --source gfs/20crv3/mapped)"),
    BundledBridge(
        "gfs_grib2_bridge", _DECODER_WORKSPACE, "--gfs-bridge",
        _SHARED_BRIDGE_ENV["gfs_grib2_bridge"], "usage: gfs_grib2_bridge",
        "the GFS front door (rw-wps --source gfs)"),
    BundledBridge(
        "hrrr_grib2_bridge", _DECODER_WORKSPACE, "--hrrr-bridge",
        _SHARED_BRIDGE_ENV["hrrr_grib2_bridge"], "usage: hrrr_grib2_bridge",
        "the HRRR front door (rw-wps --source hrrr)"),
    BundledBridge(
        "gdt101_remap", _DECODER_WORKSPACE, "--gdt101-remap",
        _SHARED_BRIDGE_ENV["gdt101_remap"], "usage: gdt101_remap",
        "every source whose native grid is a GDT-101 unstructured mesh "
        "(rw-wps --source icon-global or icon-d2), whose input-normalization "
        "stage "
        "writes its regional intermediates with this binary and refuses "
        "rather than falling back without it"),
    BundledBridge(
        "rw_fetch", _RENDERER_WORKSPACE, "--rw-fetch",
        _FETCH_BACKBONE_ENV, "usage: rw_fetch",
        "the rust fetch backbone (rw-wps --source ... over NOMADS/S3)"),
)
BRIDGE_NAMES = tuple(bridge.name for bridge in BUNDLED_BRIDGES)
#: The cargo workspaces a standalone bundle must build, in build order
#: and without repeats: the column above, read as the build plan it is.
BRIDGE_WORKSPACES = tuple(dict.fromkeys(
    bridge.workspace for bridge in BUNDLED_BRIDGES))
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
    **{bridge.name: bridge.usage_marker for bridge in BUNDLED_BRIDGES},
    # The mapped decode engine is a decoder of record: on the Rust route
    # it is the one binary an input manifest seals, so it needs the same
    # no-argument identity probe the subprocess tools get.  Without a row
    # here `bridge_identity` refuses the name outright and no manifest
    # could be authored for that route at all.  It is NOT in
    # BUNDLED_BRIDGES because the standalone bundle does not carry it:
    # the mapped route builds its own copy, and a row here would promise
    # a file the bundle has no build step for.
    "gpuwm_mapped_engine": "usage: gpuwm_mapped_engine",
}
# The generic GRIB2 pair is an internal tabular ABI, not just any executable
# with the right basename and usage string.  These adjacent-column markers
# deliberately cover fields consumed unconditionally by mapped_source.py.
# A stale bridge without ``member`` previously passed the usage-only check and
# then failed after a distribution had already been built and installed.
_BRIDGE_ABI_MARKERS = {
    # The GRIB bridges' markers live in gpuwm.bridges, because
    # `gpuwm doctor` applies the same handshake one step earlier -- at
    # the report a user reads before burning a preparation run, rather
    # than at the sealing of a distribution.  Two copies is how the
    # gfs_grib2_bridge series-contract change came to be caught here and
    # not there.
    **dict(_SHARED_BRIDGE_ABI_MARKERS),
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


def add_bridge_options(parser: argparse.ArgumentParser) -> None:
    """Give a distribution builder one option per declared bridge.

    Both builders call this instead of writing a line per bridge, so a
    new row in :data:`BUNDLED_BRIDGES` reaches the Linux and the Windows
    builder in the same commit that declares it.  ``dest`` is spelled
    out rather than left to argparse so the option text and the attribute
    the builders read cannot drift apart.
    """

    for bridge in BUNDLED_BRIDGES:
        parser.add_argument(
            bridge.option, type=Path, required=True, dest=bridge.dest,
            help=f"the built {bridge.name}, needed by {bridge.consumer}")


def bridge_inputs(args: argparse.Namespace) -> dict[str, Path]:
    """Resolve every declared bridge from one builder's parsed options.

    A caller that assembles its own namespace (the release builder does,
    from a cargo target directory) and forgets a row is refused BY NAME
    here, naming the route a bundle without that binary would refuse.
    That refusal is the whole point: the shipped release builder dropped
    ``rw_fetch`` from its namespace and the failure was an
    ``AttributeError`` inside the packager, three screens from the
    omission.
    """

    resolved: dict[str, Path] = {}
    for bridge in BUNDLED_BRIDGES:
        value = getattr(args, bridge.dest, None)
        if value is None:
            raise ValueError(
                f"the standalone rw-wps bundle declares {bridge.name} and "
                f"this build named no {bridge.option}: a bundle without it "
                f"refuses {bridge.consumer}")
        resolved[bridge.name] = Path(value).resolve()
    return resolved


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
            # Both backends map soil, snow, skin temperature and sea ice
            # through these entries of the bundled CPU library: the WPS
            # masked chain, the native HRRR route's soil stencil, and the
            # lake skin search with the water-temperature blends, the
            # water repairs and the per-body assembly.
            "masked_surface_chain": "rust-wps-masked-chain-f64-v2",
            "masked_bilinear_stencil": "rust-masked-bilinear-stencil-f64-v1",
            "water_blend": "rust-water-blend-f64-v3",
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
            # The format was proven above, so this launch cannot be the
            # unbounded CreateProcess of a non-image; the error mode
            # covers the remaining loader dialog (a missing DLL), which
            # hangs the same way and which no header can predict.
            with quiet_loader_errors():
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
    marker = _BRIDGE_ABI_MARKERS["gpuwm_preprocess_cpu"]
    if marker not in payload:
        raise RuntimeError(
            "native CPU preprocessing backend predates the masked surface "
            "chain, soil stencil and water blends "
            f"({marker.decode('ascii')}), which every preparation with a "
            f"land-sea mask needs under both backends: {path}")
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
            "masked_chain_self_test": _cpu_masked_chain_self_test(backend),
            "masked_stencil_self_test": _cpu_masked_stencil_self_test(
                backend),
            "water_blend_self_test": _cpu_water_blend_self_test(backend),
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


def _cpu_masked_chain_self_test(backend: Any) -> dict[str, Any]:
    """Run the masked surface chain on the host, not just look it up.

    Two cases, each at one and three workers: WPS's queue-limited search
    (a target at (4.49, 4.49) takes 11, the donor dequeued first, not 22,
    the globally nearer one), and a sea-ice sheet stored at 1.0003 that
    the range repair puts on its bound and counts.  The outputs and
    counts are bound by hash, so the cut proves the Windows library the
    same way as the Linux one.
    """

    import numpy as np

    field = np.zeros((1, 10, 10), dtype=np.float64)
    donors = np.zeros((10, 10), dtype=bool)
    field[0, 4, 0], donors[4, 0] = 11.0, True
    field[0, 6, 7], donors[6, 7] = 22.0, True
    ice = np.full((1, 6, 6), 1.0003, dtype=np.float64)
    everywhere = np.ones((6, 6), dtype=bool)
    cases = (
        ("search_queue_limited", field, donors,
         np.array([4.49]), np.array([4.49]), ("search",), None,
         [11.0]),
        ("range_roundoff_at_bound", ice, everywhere,
         np.array([2.25, 3.0]), np.array([1.5, 4.0]),
         ("four_pt", "average_4pt"), (0.0, 1.0), [1.0, 1.0]),
    )
    digest = hashlib.sha256()
    for name, source, valid, ty, tx, chain, bounds, expected in cases:
        results = []
        for workers in (1, 3):
            values, counts = backend.wps_masked_chain(
                source, valid, None, ty, tx, np.ones(ty.shape, dtype=bool),
                chain, mode="plain", fill_value=-1.0,
                physical_range=bounds, workers=workers)
            results.append((values.tobytes(), counts.tobytes()))
        if results[0] != results[1]:
            raise RuntimeError(
                f"CPU masked-chain self-test {name} changed with worker count")
        observed = np.frombuffer(results[0][0], dtype=np.float64).tolist()
        if observed != expected:
            raise RuntimeError(
                f"CPU masked-chain self-test {name} produced {observed}, "
                f"not {expected}")
        digest.update(results[0][0] + results[0][1])
    return {
        "status": "PASS",
        "operation": "wps_masked_chain_f64",
        "cases": [case[0] for case in cases],
        "worker_counts": [1, 3],
        "output_sha256": digest.hexdigest(),
    }


def _cpu_masked_stencil_self_test(backend: Any) -> dict[str, Any]:
    """Build and apply the native HRRR route's soil stencil on the host.

    One window, at one and three workers: a target with land corners
    (renormalised weights), a sea-covered target with two land cells at
    the same distance inside the radius (the radius scan, whose first
    nearest cell in scan order wins the tie) and a sea-covered target
    whose nearest land lies past the radius in a window closed on every
    edge (the nearest-cell search).  The indices,
    weights, report counts and the applied soil values are bound by hash,
    so the cut proves the Windows library the same way as the Linux one.
    """

    import numpy as np

    valid = np.zeros((12, 12), dtype=bool)
    valid[2, 2] = valid[2, 3] = True
    valid[5, 9] = valid[9, 5] = True
    x = np.array([2.25, 7.0, 10.4])
    y = np.array([1.5, 7.0, 10.6])
    apply = np.ones(3, dtype=bool)
    field = np.arange(144, dtype=np.float32).reshape(1, 12, 12) / 7.0
    digest = hashlib.sha256()
    results = []
    for workers in (1, 3):
        code, raw = backend.masked_bilinear_stencil(
            x, y, valid, apply, fallback_radius=3, closed_edges=15,
            edges_unknown=False, distant_cells=8.0, listed=4,
            workers=workers)
        if code:
            raise RuntimeError(
                f"CPU masked-stencil self-test refused with code {code}")
        applied = backend.masked_stencil_apply(
            field, raw["indices_y"], raw["indices_x"], raw["weights"],
            workers=workers)
        results.append(b"".join((
            raw["indices_y"].tobytes(), raw["indices_x"].tobytes(),
            raw["weights"].tobytes(), raw["counts"].tobytes(),
            applied.tobytes())))
    if results[0] != results[1]:
        raise RuntimeError(
            "CPU masked-stencil self-test changed with worker count")
    donors = (raw["indices_y"][0].tolist(), raw["indices_x"][0].tolist())
    if donors != ([1, 5, 9], [2, 9, 5]):
        raise RuntimeError(
            f"CPU masked-stencil self-test chose donors {donors}, not "
            "([1, 5, 9], [2, 9, 5])")
    digest.update(results[0])
    return {
        "status": "PASS",
        "operation": "masked_bilinear_stencil_f64",
        "cases": ["renormalized_corners", "radius_scan_tie",
                  "nearest_past_radius"],
        "worker_counts": [1, 3],
        "output_sha256": digest.hexdigest(),
    }


def _cpu_water_blend_self_test(backend: Any) -> dict[str, Any]:
    """Run the lake skin search and the three water blends on the host.

    One small source at one and three workers: a lake target whose nearest
    water lies past the first search window, with two water cells tied in
    distance (the first in row-major order wins); a blend with one donor
    corner missing (renormalised weights); a component fill that closes a
    hole from its neighbours; an overlay sample with one invalid
    corner; a water repair where one body closes from its own water and
    another takes the nearest water; and a per-body assembly with one body
    on analysis and one on skin; the labelling of two cells that touch
    only at a corner (one body, eight-connected) beside a separate one;
    the source owner of a cell two bodies claim equally (the higher label
    wins); and the surface-nearest search with two water cells tied in
    distance (the first scanned wins).  Every output is bound by hash, so
    the cut proves the Windows library the same way as the Linux one.
    """

    import numpy as np

    skin = np.arange(400, dtype=np.float64).reshape(20, 20) / 3.0 + 270.0
    water = np.zeros((20, 20), dtype=bool)
    water[10, 0] = water[10, 19] = True
    field = np.arange(16, dtype=np.float64).reshape(4, 4) + 280.0
    donors = np.ones((4, 4), dtype=bool)
    donors[1, 2] = False
    rows0 = np.array([[1]])
    cols0 = np.array([[1]])
    corners = ((rows0, cols0, np.array([[0.25]])),
               (rows0, cols0 + 1, np.array([[0.25]])),
               (rows0 + 1, cols0, np.array([[0.25]])),
               (rows0 + 1, cols0 + 1, np.array([[0.25]])))
    holes = field.copy()
    holes[1, 1] = holes[2, 2] = np.nan
    component = np.ones((4, 4), dtype=bool)
    component[3, 3] = False
    valid = np.ones((4, 4), dtype=bool)
    valid[0, 1] = False
    # Two bodies on a 6x6 grid: the left one has a bad hole among good
    # water, the right one is wholly bad and takes the nearest water.
    repair_labels = np.zeros((6, 6), dtype=np.int32)
    repair_labels[1:5, 0:2] = 1
    repair_labels[1:3, 4:6] = 2
    repair_water = repair_labels > 0
    repair_values = np.where(
        repair_water, 280.0 + np.arange(36).reshape(6, 6) / 7.0, 300.0)
    repair_values[2, 1] = 0.0
    repair_values[1:3, 4:6] = np.nan
    repair_source = np.where(repair_water, 1, 0).astype(np.int8)
    body_labels = repair_labels.copy()
    body_owner = np.zeros((4, 4), dtype=np.int32)
    body_owner[1:3, 0:2] = 1
    body_rows, body_cols = np.meshgrid(
        np.arange(6) // 2, np.arange(6) // 2, indexing="ij")
    body_corners = ((body_rows, body_cols, np.full((6, 6), 0.5)),
                    (np.minimum(body_rows + 1, 3), body_cols,
                     np.full((6, 6), 0.5)))
    label_mask = np.zeros((5, 5), dtype=bool)
    label_mask[0, 0] = label_mask[1, 1] = label_mask[4, 4] = True
    owner_labels = np.array([1, 2, 2, 1], dtype=np.int32)
    owner_lat = np.array([0.0, 0.0, 1.0, 1.0])
    owner_lon = np.array([0.0, 0.0, 1.0, 1.0])
    digest = hashlib.sha256()
    results = []
    for workers in (1, 3):
        nearest = backend.lake_water_nearest(
            skin, water, np.array([10.0]), np.array([9.5]), workers=workers)
        blend = backend.masked_bilinear_blend(
            field, donors, corners, (1, 1), workers=workers)
        filled = backend.component_fill(holes, component, workers=workers)
        sampled, covered = backend.overlay_bilinear_sample(
            field, valid, np.array([0]), np.array([0]), np.array([0.5]),
            np.array([0.25]), np.array([True]), workers=workers)
        repaired, repaired_source, tallies, repaired_mask = (
            backend.water_repair(
                repair_values, repair_source, repair_water, repair_labels,
                minimum=170.0, maximum=400.0, nearest_water_code=5,
                surrounding_skin_code=6, workers=workers))
        body_values = np.full((6, 6), 290.0)
        body_source = np.zeros((6, 6), dtype=np.int8)
        stats, coverage, _ = backend.water_bodies(
            labels=body_labels, lake_class=np.array([False, True, False]),
            skin=np.full((6, 6), 290.0), values=body_values,
            source=body_source, codes=(1, 2, 4), sst=field,
            owner=body_owner, corners=body_corners, min_coverage=0.5,
            minimum=170.0, maximum=400.0, max_listed=4, workers=workers)
        labelled, bodies = backend.label_components(label_mask)
        owned = backend.component_owner(
            owner_labels, np.array([0.0, 1.0]), np.array([0.0, 1.0]),
            owner_lat, owner_lon, (2, 2), workers=workers)
        searched, unmatched = backend.masked_nearest(
            field.astype(np.float32), ~donors, np.array([[1.0]]),
            np.array([[2.0]]), np.array([[False]]), surface="water",
            fill_value=-1.0, radius=1, workers=workers)
        results.append(b"".join((
            nearest.tobytes(), blend.tobytes(), filled.tobytes(),
            sampled.tobytes(), covered.tobytes(), repaired.tobytes(),
            repaired_source.tobytes(), repaired_mask.tobytes(),
            repr(sorted(tallies.items())).encode("ascii"),
            body_values.tobytes(), body_source.tobytes(), stats.tobytes(),
            coverage.tobytes(), labelled.tobytes(), bytes([bodies]),
            owned.tobytes(), searched.tobytes(), bytes([unmatched]))))
    if results[0] != results[1]:
        raise RuntimeError(
            "CPU water-blend self-test changed with worker count")
    if float(nearest[0]) != float(skin[10, 0]):
        raise RuntimeError(
            f"CPU water-blend self-test chose {float(nearest[0])!r}, not "
            f"the skin at row 10, column 0 ({float(skin[10, 0])!r})")
    if bodies != 2 or owned.tolist() != [[2, 0], [0, 2]]:
        raise RuntimeError(
            f"CPU water-blend self-test labelled {bodies} bodies and gave "
            f"owners {owned.tolist()}, not 2 bodies and [[2, 0], [0, 2]]")
    if float(searched[0, 0]) != float(field[0, 2]) or unmatched:
        raise RuntimeError(
            f"CPU water-blend self-test searched {float(searched[0, 0])!r}, "
            f"not the water cell at row 0, column 2 ({float(field[0, 2])!r})")
    digest.update(results[0])
    return {
        "status": "PASS",
        "operation": "water_blend_f64",
        "cases": ["lake_search_past_window", "renormalized_blend",
                  "component_fill", "overlay_invalid_corner",
                  "water_repair_own_body_and_nearest",
                  "water_bodies_analysis_and_skin",
                  "labelling_eight_connected", "owner_tie_higher_label",
                  "surface_nearest_first_of_tie"],
        "worker_counts": [1, 3],
        "output_sha256": digest.hexdigest(),
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

    gpu: dict[str, Any] = gpu_cuda_stack_identity(require_gpu=require_gpu)

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


def build_parser() -> argparse.ArgumentParser:
    """This console script's parser, built without parsing anything.

    Exposed so the docs/CLI parity test can read the option surface of a
    documented door without running it.
    """

    parser = argparse.ArgumentParser(prog="gpuwm-wrf-runtime-check",
                                     description=__doc__)
    parser.add_argument("--bridge-dir", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--contract", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # A runtime check whose own runtime is unnamed is not a check.
    from gpuwm.provenance_gate import announce_for_main

    refusal = announce_for_main("gpuwm-wrf-runtime-check")
    if refusal is not None:
        print(f"gpuwm-wrf-runtime-check: {refusal}", file=sys.stderr)
        return 2
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
