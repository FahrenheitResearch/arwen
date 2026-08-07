"""Identity of the toolchain that turns this project's ``.cu`` sources into SASS.

Every acceptance gate stated in ULP against a Fortran oracle is a measurement
of *the compiled kernel*, not of the source.  The source is pinned by
:mod:`gpuwm.certify.kernel_manifest` (``source_sha256``); this module pins the
other half -- which compiler produced the image the numbers were measured on.

Why it exists.  On 2026-08-04 this project's kernel compiler changed underneath
a certified table without a single tracked input moving: ``cupy`` 14.0.1,
``numpy`` 2.2.6, CPython 3.13.7 and the display driver were all still on their
pinned versions, and the ``.cu`` file was byte-identical, yet the Shin-Hong
CUDA column's distance from WRF moved on two fields.  The mover was NVRTC:
CuPy had been resolving the system CUDA v13.0 toolkit (NVRTC/ptxas 13.0.48)
and, once ``nvidia-cuda-nvrtc-cu12`` 12.9.86 appeared in ``site-packages``, it
resolved that instead.  The published pin row for that item reported only
``nvrtc.getVersion()`` -- ``(12, 9)`` -- which cannot separate 12.9.41 from
12.9.86, and the failure surfaced as a bare number mismatch with no named
cause.  What follows is the missing resolution.

Nothing here needs a device: NVRTC compiles an empty translation unit without
one, and the reported build string is exactly what CuPy folds into its own
kernel-cache key, so it is the same discriminator the cache already trusts.

Every item carries a status, the same contract :mod:`gpuwm.certify.pins` uses:
``resolved`` means something measured it in this process, ``unavailable``
means nothing did and says why.  A fingerprint is never filled with a
plausible value.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys

#: Status of an item this process measured.
STATUS_RESOLVED = "resolved"

#: Status of an item nothing measured, carrying the reason.
STATUS_UNAVAILABLE = "unavailable"

#: Value stored for an item that could not be resolved.  Deliberately a
#: string, not ``None``: a fingerprint round-trips through JSON receipts and
#: must compare equal to itself after that trip.
UNRESOLVED = "unavailable"

_BUILD_RE = re.compile(
    r"Cuda compilation tools, release [\d.]+, V(?P<build>[\d.]+)")
_BUILD_ID_RE = re.compile(r"Compiler Build ID: (?P<id>\S+)")


def nvrtc_banner() -> str:
    """The comment banner NVRTC puts at the head of the PTX it emits.

    Compiling the empty translation unit is the cheapest complete statement of
    compiler identity available: it names the release, the four-part build and
    the internal changelist, and needs no device.  Returns ``""`` if NVRTC is
    not importable or refuses the empty program.
    """
    try:
        from cupy_backends.cuda.libs import nvrtc
    except Exception:
        return ""
    program = None
    try:
        program = nvrtc.createProgram("", "compile_platform_probe.cu", [], [])
        nvrtc.compileProgram(program, [])
        ptx = nvrtc.getPTX(program)
    except Exception:
        return ""
    finally:
        if program is not None:
            try:
                nvrtc.destroyProgram(program)
            except Exception:
                pass
    if isinstance(ptx, bytes):
        ptx = ptx.decode("ascii", "replace")
    # The banner is the leading comment block; the body is irrelevant here and
    # would make the fingerprint depend on the empty program's codegen.
    return "\n".join(
        line for line in ptx.splitlines()[:12] if line.startswith("//"))


def nvrtc_build(banner: str | None = None) -> str:
    """The four-part NVRTC build, e.g. ``"12.9.86"`` or ``"13.0.48"``.

    This is the discriminator ``nvrtc.getVersion()`` is too coarse to be:
    12.9.41 and 12.9.86 both report ``(12, 9)`` and do not generate the same
    code.  ``UNRESOLVED`` when NVRTC did not answer.
    """
    text = nvrtc_banner() if banner is None else banner
    match = _BUILD_RE.search(text)
    return match.group("build") if match else UNRESOLVED


def nvrtc_build_id(banner: str | None = None) -> str:
    """NVRTC's internal changelist, e.g. ``"CL-36037853"``, or ``UNRESOLVED``."""
    text = nvrtc_banner() if banner is None else banner
    match = _BUILD_ID_RE.search(text)
    return match.group("id") if match else UNRESOLVED


def nvrtc_library() -> dict[str, str]:
    """Path and SHA-256 of the NVRTC shared library this process loaded.

    The build string above is what the compiler says about itself; this is
    what the filesystem says about the file, and the two are recorded together
    so a same-version rebuild -- or a second copy of the same version shadowing
    the first -- cannot pass as the image the numbers were measured on.
    """
    path = _loaded_library_path()
    if not path:
        return {"status": STATUS_UNAVAILABLE,
                "reason": "no NVRTC shared library is loaded in this process",
                "path": UNRESOLVED, "sha256": UNRESOLVED}
    try:
        with open(path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
    except OSError as error:
        return {"status": STATUS_UNAVAILABLE,
                "reason": f"{type(error).__name__}: {error}",
                "path": path, "sha256": UNRESOLVED}
    return {"status": STATUS_RESOLVED, "path": path, "sha256": digest}


def _loaded_library_path() -> str:
    """Absolute path of the loaded NVRTC library, or ``""``.

    Windows and Linux keep this in different places and neither exposes it
    through CuPy, so both are read directly.  Failure is a plain ``""``: this
    is provenance, and provenance must never be able to break a run.
    """
    try:
        if sys.platform == "win32":
            return _loaded_library_path_win32()
        with open("/proc/self/maps", encoding="ascii", errors="replace") as fh:
            for line in fh:
                path = line.rsplit(" ", 1)[-1].strip()
                name = os.path.basename(path)
                if name.startswith("libnvrtc.so") and "builtins" not in name:
                    return path
    except Exception:
        return ""
    return ""


def _loaded_library_path_win32() -> str:
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetModuleHandleW.restype = ctypes.c_void_p
    kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
    kernel32.GetModuleFileNameW.argtypes = [
        ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
    # CuPy's CUDA 12 and 13 builds resolve different SONAMEs; ask for both
    # rather than assuming which wheel is installed.
    for name in ("nvrtc64_120_0.dll", "nvrtc64_130_0.dll"):
        handle = kernel32.GetModuleHandleW(name)
        if not handle:
            continue
        buffer = ctypes.create_unicode_buffer(4096)
        if kernel32.GetModuleFileNameW(handle, buffer, 4096):
            return buffer.value
    return ""


def compile_platform_fingerprint() -> dict[str, str]:
    """Everything that decides what SASS this process runs, in one dict.

    Ordered so a diff reads outward from the compiler: the two items that
    actually generate code first, then the runtime that loads it, then the
    host libraries that feed it.  Values are strings so the dict survives a
    JSON round trip into a receipt and still compares equal.
    """
    banner = nvrtc_banner()
    library = nvrtc_library()
    fingerprint = {
        "nvrtc_build": nvrtc_build(banner),
        "nvrtc_build_id": nvrtc_build_id(banner),
        "nvrtc_library_sha256": library["sha256"],
        "cuda_driver_version": _driver_version(),
        "device_compute_capability": _compute_capability(),
        "cupy_version": _cupy_version(),
        "numpy_version": _package_version("numpy"),
    }
    return fingerprint


def _cupy_version() -> str:
    """Reuse the pin table's resolver rather than re-deriving it.

    CuPy ships one wheel name per CUDA major, and a second enumeration of
    those names is how two receipts end up disagreeing about the same box.
    :mod:`gpuwm.certify.pins` already owns that enumeration.
    """
    try:
        from gpuwm.certify.pins import _installed_cupy_version
        return _installed_cupy_version()
    except Exception:
        return UNRESOLVED


def _driver_version() -> str:
    try:
        from cupy.cuda import runtime
        return str(runtime.driverGetVersion())
    except Exception:
        return UNRESOLVED


def _compute_capability() -> str:
    try:
        import cupy
        return str(cupy.cuda.Device().compute_capability)
    except Exception:
        return UNRESOLVED


def _package_version(name: str) -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version(name)
    except Exception:
        return UNRESOLVED


def describe_drift(recorded, measured) -> list[str]:
    """Human-readable ``key: recorded -> measured`` lines for what differs.

    Empty when the two agree on every key they share.  Keys present in only
    one side are reported too: a fingerprint that grew a field is itself a
    change worth naming rather than silently ignoring.
    """
    lines = []
    for key in sorted(set(recorded) | set(measured)):
        was = recorded.get(key, "(absent)")
        now = measured.get(key, "(absent)")
        if was != now:
            lines.append(f"  {key}: recorded {was} -> measured {now}")
    return lines
