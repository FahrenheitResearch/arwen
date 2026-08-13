"""``gpuwm doctor``: verify the runtime estate, print exact remedies.

The pip package deliberately splits its runtime across four estates the
installer cannot see from ``pip install`` alone: the GPU runtime (CuPy),
the render extra (wrf-rust + matplotlib), the compiled Rust artifacts
(never shipped in the wheel; ``gpuwm fetch-bridges`` stages a release's
prebuilt bundle where one exists for the platform), and the data roots
(``WPS_GEOG``/``GPUWM_CASE_DATA_ROOT``).  Doctor checks each one for
real, not by presence: it imports CuPy/wrf/matplotlib in short-lived
subprocesses, probe-executes every bridge executable, loads the CPU
preprocessing library through ctypes and reads its ABI version,
sha256-validates the packaged Thompson tables with the same routine the
model uses at launch, parses the Noah/landuse tables with the model's
own parsers, and requires each WPS_GEOG dataset's ``index`` file.  No
cargo builds, no network; the only device work is two deliberate
short-lived subprocess probes (the CuPy-wheel/box cuBLAS pairing and
the radar-DA eigensolver), isolated so a wedged runtime cannot poison
this process.  Every gap prints a remedy whose every line is either a
command that runs as printed in this platform's own shell or a ``#``
comment -- never prose fused onto a command -- instead of letting the
user meet a raw traceback three commands later.

Statuses distinguish what was proven: ``verified`` means the deep check
ran and passed; ``present`` is reserved for the few items where nothing
deeper than existence can honestly be checked (and says so); ``missing``
is a gap with a remedy; ``info`` is context.

The report is layered (:mod:`gpuwm.explain`).  By default every finding
is one line -- status, subject, and THE command that closes it -- and
adjacent findings that share a remedy fold into one, because a fresh pip
install gaps every Rust artifact at once and six identical remedies read
as six problems.  ``--explain`` prints what this module always printed:
the evidence behind each check and the whole pasteable remedy block,
verbatim.  Nothing was shortened; the long form moved one flag away.

Exit status: 0 when nothing actionable is missing, 1 otherwise.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import importlib.metadata
from importlib.util import find_spec
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from gpuwm import bridges
from gpuwm import rustwx
from gpuwm import rustwx_fetch
from gpuwm.explain import explain_enabled

# The pip extras exactly as the README installs them.
#
# Each hint is a command first and explanation second, on separate
# lines, because a remedy is read under stress and pasted whole: the
# parenthetical that used to trail the command on the same line --
# `pip install 'gpuwm[gpu]'   (or: pip install cupy-cuda12x)` -- is a
# shell error the moment anyone does the obvious thing with it.
#
# The CuPy hint is the one remedy in this module that must NOT lead with
# a single command, and it is the exception that proves the rule: CuPy
# ships one wheel per CUDA major, pip cannot detect the major, and this
# text is only reached when the box's own major could not be read
# either.  A hint that led with `pip install 'gpuwm[gpu]'` -- as it did
# through 1.8.0 -- handed a CUDA-13 box the cu12 wheel and called it the
# default.  So when the major is unknown, BOTH are printed and neither
# is the default; when it is known, `_gpu_extra_hint` below replaces
# this text with the one line that matches the box.
GPU_EXTRA_HINT = ("nvidia-smi\n"
                  "  # read the CUDA version out of that header.  CuPy\n"
                  "  # ships one wheel per CUDA major and pip cannot\n"
                  "  # detect which one this box serves, so the extra\n"
                  "  # has to name it.  Then, for a CUDA 12.x box:\n"
                  "pip install 'gpuwm[gpu-cu12]'\n"
                  "  # or, for a box whose CUDA is 13-only:\n"
                  "pip install 'gpuwm[gpu-cu13]'")

#: The brief report has room for exactly one command.  It carries the
#: cu12 wheel with the cu13 alternative fused on as a shell comment --
#: which runs as printed in sh and in PowerShell alike -- because the
#: one thing this line may never do is imply cu12 is simply correct.
GPU_EXTRA_UNKNOWN_ACTION = ("pip install 'gpuwm[gpu-cu12]'  "
                            "# CUDA 12.x; cu13 box: gpuwm[gpu-cu13]")
RENDER_EXTRA_HINT = ("pip install 'gpuwm[render]'\n"
                     "  # installs wrf-rust + matplotlib")
GEOG_HINT = (
    "gpuwm fetch-geog\n"
    "  # downloads the nine required WPS_GEOG datasets (~1.3 GB\n"
    "  # compressed, ~16 GB unpacked) into $GPUWM_CASE_DATA_ROOT/WPS_GEOG,\n"
    "  # or --root DIR.  Resumable, SHA-256-verified, re-run safe.")
REINSTALL_HINT = (
    "pip install -e .\n"
    "  # the installed package is incomplete; reinstall from a clone\n"
    "  # (or from a rebuilt wheel)")

#: Bridge executables the real-data routes launch, with the consumer
#: that fails without each one.
_BRIDGE_CONSUMERS = {
    "grib1_bridge": "ERA5 route (gpuwm check/run, rw-wps --source era5)",
    "gfs_grib2_bridge": "GFS front door (rw-wps --source gfs)",
    "hrrr_grib2_bridge": "HRRR front door (rw-wps --source hrrr)",
    "grib2_inventory": "20CRv3/mapped GRIB2 routes",
    "grib2_dump": "20CRv3/mapped GRIB2 routes",
}

_PROBE_TIMEOUT_S = 30


@dataclass(frozen=True)
class Check:
    """One doctor line: verified/present/MISSING/info plus the remedy.

    Four fields carry the report; three more carry its *layering*.

    ``detail`` and ``remedy`` are the full text -- the evidence and the
    whole pasteable remedy block -- and ``--explain`` prints them
    unchanged.  ``brief`` and ``action`` are the same finding at one
    line: the short evidence token, and THE single command to run.
    They are declared here rather than sliced out of ``detail`` and
    ``remedy`` by a parser, because most remedy blocks open with a ``#``
    comment and several have no command in them at all (an unset
    ``GPUWM_CASE_DATA_ROOT`` needs a path only the reader knows).  A
    parser would have had to guess, and a guessed next command is the
    one thing this report cannot afford to get wrong.

    ``group`` lets the terse report fold repeats: a pip install gaps
    five bridges at once and prints ``gpuwm fetch-bridges`` five times,
    which reads as five problems.  Grouping is a *presentation* of the
    same five checks -- ``--explain`` and ``--json`` still carry each
    one by name.
    """

    name: str
    status: str  # "verified" | "present" | "missing" | "info"
    detail: str
    remedy: str | None = None
    #: The single next command, or a short imperative when there is no
    #: command to print.  ``None`` for a check with nothing to do.
    action: str | None = None
    #: Short evidence for the one-line form; ``None`` prints the name
    #: alone, which is the right answer for a check whose only news is
    #: that it passed.
    brief: str | None = None
    #: Fold key for the terse report; ``None`` never folds.
    group: str | None = None
    #: Does this gap justify a nonzero exit?  ``True`` for anything
    #: broken or integrity-suspect (a named executable that is absent,
    #: a table that fails its hash, a manifest that fails revalidation).
    #: ``False`` for an *absent optional* piece with a documented
    #: fallback or a documented opt-in (CuPy on the base install, the
    #: render extra, the rust renderer, the CPU preprocess library, the
    #: WPS_GEOG tree nobody has fetched yet).  The report text is
    #: identical either way -- MISSING stays MISSING and the remedy
    #: still prints; only the exit code reads this, so a base install
    #: that did everything its documentation asked stops failing
    #: installers and `gpuwm setup` for gaps it was told are optional.
    blocking: bool = True


#: Terse-report fold keys.  Both name a crate, because that is what
#: makes the members share one remedy: everything under
#: ``tools/grib1_bridge`` is staged (or built) together, and so is
#: everything under ``tools/rustwx``.
_GROUP_BRIDGES = "bridges"
_GROUP_ENGINES = "rust engines"

#: The two ``gpuwm setup`` steps, by the command each one runs.  Doctor
#: names ``gpuwm setup`` in its summary exactly when it would otherwise
#: print more than one of these, which is the definition of a fresh
#: install: the wrapper is a shorter true answer than the list.
SETUP_ACTIONS = ("gpuwm fetch-bridges", "gpuwm fetch-tables")


def _short(text: str, limit: int = 64) -> str:
    """One line of at most ``limit`` characters, ellipsis when cut.

    The terse report shows evidence, not the whole finding: a failed
    import's last traceback line can be any length, and a line that
    wraps three times is the wall this report exists to stop being.  The
    untruncated text is one flag away and the ellipsis says so.
    """

    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[:limit - 3].rstrip() + "..."


def _build_action(crate_relative: str = bridges.CRATE_RELATIVE) -> str:
    """THE single next command for a missing Rust artifact, here.

    Three installs, three true one-liners.  In a checkout it is the
    cargo build.  On a wheel with a bundle published for this platform
    it is the download.  On a wheel without one the honest answer is
    that there is no single command -- it is a clone and a build -- so
    the action is the flag that prints those steps rather than a
    fabricated one-liner naming a directory this install does not have.
    """

    if bridges.sources_present(crate_relative):
        return bridges.cargo_build_one_liner(crate_relative)
    if bridges.prebuilt_bundle_offer() is not None:
        return "gpuwm fetch-bridges"
    return "gpuwm doctor --explain"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Python packages: actual imports, in a subprocess
# ---------------------------------------------------------------------------

def _import_probe(module: str,
                  distribution: str | None = None) -> tuple[bool, str]:
    """Actually import ``module`` in a subprocess; (ok, evidence).

    ``find_spec`` alone lies green for a package whose install is broken
    (ABI mismatch, missing native dependency, half-removed dist-info).
    The subprocess keeps a failing import from poisoning this process
    and allocates nothing beyond the import itself.

    The version reported is the *installed distribution's*, read from
    package metadata, falling back to the module's ``__version__``
    attribute.  A module attribute is whatever the author last edited by
    hand and can lag the release it shipped in -- a field report had
    doctor announcing wrf-rust 0.2.34 on a machine with 0.2.35
    installed, which is the wrong number to hand someone debugging a
    version-sensitive problem.  ``distribution`` is the pip name when it
    differs from the import name (``wrf`` is installed as ``wrf-rust``).
    """

    if find_spec(module) is None:
        return False, "not installed"
    code = (
        "import sys, importlib.metadata as md\n"
        f"import {module} as m\n"
        "try:\n"
        f"    version = md.version({(distribution or module)!r})\n"
        "except Exception:\n"
        "    version = str(getattr(m, '__version__', 'imported'))\n"
        "sys.stdout.write(version)\n")
    try:
        probe = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            errors="replace", timeout=_PROBE_TIMEOUT_S * 4)
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, f"installed but the import probe failed to run: {error}"
    if probe.returncode == 0:
        return True, (probe.stdout or "").strip() or "imported"
    tail = [line for line in (probe.stderr or "").strip().splitlines()
            if line.strip()]
    reason = tail[-1] if tail else f"exit {probe.returncode}"
    return False, f"installed but failed to import: {reason}"


#: CuPy ships one wheel per CUDA major (``cupy-cuda12x``, ``cupy-cuda13x``)
#: and the wrong one is invisible to an import probe: the wheel hard-codes
#: its CUDA major's library names (``libcublas.so.12``) and loads cuBLAS
#: lazily, so on a CUDA-13-only box ``import cupy`` succeeds, kernels
#: compile, and the first matmul of a real run is what dies.  Proven on a
#: rented CUDA 13.2 node (2026-08-05): import probes, certification, and a
#: 57-test GPU slice all passed honestly before the first cuBLAS load
#: killed the campaign.  So this probe performs that first load on
#: purpose, in a fresh subprocess, and reports the CUDA majors of both
#: sides so the check can name the extra that matches the box.
#:
#: The judgment is the LOAD, not the majors: a newer driver serves older
#: wheels wherever the older runtime libraries exist (the reference box
#: runs a 13.3 driver over a working cupy-cuda12x install), so refusing on
#: a bare major mismatch would refuse machines that work.  The majors are
#: read for the diagnosis and the remedy, after the load has failed.
_CUBLAS_PAIRING_PROBE = """
import json, sys
out = {}
try:
    import cupy
except Exception as error:
    sys.stdout.write(json.dumps({"cupy": repr(error)}))
    raise SystemExit(0)
for key, call in (("wheel_runtime", "runtimeGetVersion"),
                  ("driver", "driverGetVersion")):
    try:
        out[key] = int(getattr(cupy.cuda.runtime, call)())
    except Exception:
        out[key] = 0
try:
    out["devices"] = int(cupy.cuda.runtime.getDeviceCount())
except Exception as error:
    out["devices"] = 0
    out["device_error"] = f"{type(error).__name__}: {error}"
if out["devices"] > 0:
    try:
        a = cupy.arange(4.0, dtype=cupy.float32).reshape(2, 2)
        b = a @ a
        float(b[0, 0])
        out["cublas"] = "ok"
    except Exception as error:
        out["cublas"] = f"{type(error).__name__}: {error}"
sys.stdout.write(json.dumps(out))
"""

_CUPY_WHEEL_NAME = re.compile(r"^cupy-cuda(\d+)x$")

#: CUDA major -> the pip extra that installs its wheel.  A major outside
#: this table gets the bare wheel name instead of an extra.
#:
#: 12 names ``gpu-cu12``, not the ``gpu`` alias: a remedy that says which
#: major it chose and why is the whole point, and ``gpu`` says neither.
#: The alias still resolves for anyone who already types it.
_GPU_EXTRA_BY_MAJOR = {12: "gpu-cu12", 13: "gpu-cu13"}


def _driver_library_names() -> tuple[str, ...]:
    """The CUDA driver library this platform would load, by name."""

    if sys.platform == "win32":
        return ("nvcuda.dll",)
    if sys.platform == "darwin":
        return ()
    return ("libcuda.so.1", "libcuda.so")


def _driver_cuda_major() -> int | None:
    """The CUDA major this box's DRIVER serves, or ``None`` if unknown.

    Read with ``ctypes`` straight off the driver library rather than
    through CuPy, because the case that needs the answer most is the box
    that has NO CuPy yet: that is where the remedy has to name an extra,
    and where every CuPy-based probe is by definition unavailable.  Until
    1.8.1 that case got a static hint whose only command was the cu12
    extra, so a CUDA-13 box asked doctor what to install and was told to
    install the wheel that cannot load cuBLAS on it.

    ``cuDriverGetVersion`` is the one entry point that answers this
    without ``cuInit``: no context, no device open, nothing that could
    disturb a card another process is using.  It is still driver contact,
    so ``GPUWM_NO_LOCAL_GPU`` suppresses it and the caller falls back to
    naming both extras.  Every failure is a ``None`` -- a box with no
    NVIDIA driver is the ordinary case here, not an error.
    """

    if os.environ.get("GPUWM_NO_LOCAL_GPU", "") not in ("", "0"):
        return None
    for name in _driver_library_names():
        try:
            library = ctypes.CDLL(name)
            version = ctypes.c_int(0)
            if library.cuDriverGetVersion(ctypes.byref(version)) != 0:
                continue
        except (OSError, AttributeError, ValueError):
            continue
        if version.value > 0:
            return version.value // 1000
    return None


def _gpu_extra_hint(box_major: int | None) -> tuple[str, str]:
    """``(remedy, action)`` for a box with no working CuPy.

    With the box's CUDA major in hand the remedy leads with the ONE
    extra that matches it and says which major it read.  Without it,
    both extras are printed and neither is presented as the default,
    because a silent default is exactly how a CUDA-13 box ends up
    running cupy-cuda12x.
    """

    extra = _GPU_EXTRA_BY_MAJOR.get(box_major) if box_major else None
    if extra is None:
        return GPU_EXTRA_HINT, GPU_EXTRA_UNKNOWN_ACTION
    install = f"pip install 'gpuwm[{extra}]'"
    remedy = (f"# this box's driver serves CUDA {box_major}, and CuPy ships\n"
              f"# one wheel per CUDA major; this extra is the matching one\n"
              f"{install}")
    return remedy, install


def _installed_cupy_wheels() -> list[tuple[str, int | None]]:
    """Every installed CuPy distribution as ``(name, CUDA major)``.

    The major comes from the distribution NAME, because that is the
    thing pip resolved and the thing an uninstall must name; a
    source-built ``cupy`` has no major in its name and reports ``None``.
    """

    found = []
    for dist in importlib.metadata.distributions():
        name = (dist.metadata["Name"] or "").strip().lower()
        match = _CUPY_WHEEL_NAME.match(name)
        if match:
            found.append((name, int(match.group(1))))
        elif name == "cupy":
            found.append((name, None))
    return sorted(set(found))


def _cublas_pairing_probe() -> dict:
    """Run :data:`_CUBLAS_PAIRING_PROBE` out of process; ``{}`` if not."""

    try:
        probe = subprocess.run(
            [sys.executable, "-c", _CUBLAS_PAIRING_PROBE],
            capture_output=True, text=True, errors="replace",
            timeout=_PROBE_TIMEOUT_S * 4)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"probe": f"did not run: {error}"}
    try:
        return json.loads(probe.stdout or "{}")
    except ValueError:
        tail = [line for line in (probe.stderr or "").strip().splitlines()
                if line.strip()]
        return {"probe": tail[-1] if tail else f"exit {probe.returncode}"}


def _wrong_wheel_remedy(wheels: list[tuple[str, int | None]],
                        box_major: int) -> tuple[str, str]:
    """(remedy, action) that moves the install onto ``box_major``.

    Every line is a command as printed or a ``#`` comment, identical in
    both shells: ``pip`` spells the same everywhere.
    """

    lines = [f"# this box serves CUDA {box_major}; the installed CuPy "
             "cannot load cuBLAS on it"]
    for name, _ in wheels:
        lines.append(f"pip uninstall -y {name}")
    extra = _GPU_EXTRA_BY_MAJOR.get(box_major)
    if extra is not None:
        install = f"pip install 'gpuwm[{extra}]'"
    else:
        install = f"pip install cupy-cuda{box_major}x"
    lines.append(install)
    return "\n".join(lines), install


def _cupy_check() -> Check:
    ok, evidence = _import_probe("cupy")
    if ok:
        wheels = _installed_cupy_wheels()
        named = ", ".join(name for name, _ in wheels) or "cupy"
        if os.environ.get("GPUWM_NO_LOCAL_GPU", "") not in ("", "0"):
            # The documented never-open-the-local-device switch (the
            # rented-GPU workflow leaves it set on the user's own
            # machine).  The pairing probe is real device contact, so
            # under this flag it does not run, and the report says which
            # question went unanswered rather than pretending it passed.
            return Check(
                "cupy (GPU runtime)", "present",
                f"imported in a subprocess (cupy {evidence}, {named}); "
                "GPUWM_NO_LOCAL_GPU is set, so the CuPy-wheel/box CUDA "
                "pairing was not judged (no device contact)",
                brief=f"cupy {evidence}, device not touched",
                blocking=False)
        result = _cublas_pairing_probe()
        cublas = result.get("cublas")
        driver = int(result.get("driver") or 0)
        wheel_runtime = int(result.get("wheel_runtime") or 0)
        wheel_majors = sorted({major for _, major in wheels
                               if major is not None})
        if not wheel_majors and wheel_runtime:
            wheel_majors = [wheel_runtime // 1000]
        if cublas == "ok":
            return Check(
                "cupy (GPU runtime)", "verified",
                f"imported in a subprocess (cupy {evidence}, {named}); "
                f"cuBLAS loaded on this box (wheel CUDA "
                f"{wheel_runtime // 1000}.{wheel_runtime % 1000 // 10}, "
                f"driver CUDA {driver // 1000}.{driver % 1000 // 10})",
                brief=f"cupy {evidence}, cuBLAS ok")
        if result.get("devices", 0) == 0:
            # No device to load cuBLAS against, so the pairing cannot be
            # judged -- and a box without a CUDA device must not fail
            # doctor for it.  ``present``, honestly: import proven,
            # pairing not.
            reason = result.get("device_error") or result.get(
                "cupy") or result.get("probe") or "no CUDA device visible"
            return Check(
                "cupy (GPU runtime)", "present",
                f"imported in a subprocess (cupy {evidence}, {named}), "
                f"but the CuPy-wheel/box CUDA pairing was not judged: "
                f"{_short(str(reason), 96)}",
                brief=f"cupy {evidence}, no device", blocking=False)
        # A device exists and the first cuBLAS load failed.  This is the
        # CUDA-major trap when the majors disagree, and a broken install
        # either way; both refuse, because the next real run dies at its
        # first matmul.
        box_major = driver // 1000
        detail = (f"cupy {evidence} ({named}) imports, but cuBLAS failed "
                  f"its first load on this box: {_short(str(cublas), 120)}.")
        if box_major and wheel_majors and box_major not in wheel_majors:
            targets = ", ".join(f"CUDA {major}" for major in wheel_majors)
            detail += (f"  The wheel targets {targets}; the box serves "
                       f"CUDA {driver // 1000}.{driver % 1000 // 10}.  "
                       "An import probe cannot see this: the wheel "
                       "hard-codes its own major's library names and "
                       "loads cuBLAS lazily, so everything short of a "
                       "real matmul passes")
        if box_major:
            remedy, action = _wrong_wheel_remedy(wheels, box_major)
        else:
            remedy, action = _gpu_extra_hint(_driver_cuda_major())
        return Check("cupy (GPU runtime)", "missing", detail, remedy,
                     action=action, brief=_short(str(cublas)))
    # No usable CuPy.  The extra to name is a property of the BOX, so it
    # is read from the driver here rather than defaulted: this is the
    # branch a fresh CUDA-13 machine lands on, and the branch that used
    # to hand it the cu12 wheel.
    box_major = _driver_cuda_major()
    remedy, action = _gpu_extra_hint(box_major)
    served = "" if box_major is None else f"; this box serves CUDA {box_major}"
    if evidence == "not installed":
        return Check(
            "cupy (GPU runtime)", "missing",
            "not installed -- gpuwm check/run and the domain wizard's "
            "sizing estimator need it; fetch/import-namelist/render "
            f"do not{served}", remedy,
            action=action, brief="not installed", blocking=False)
    return Check("cupy (GPU runtime)", "missing", evidence + served, remedy,
                 action=action, brief=_short(evidence),
                 blocking=False)


#: What to install when the device has no cuSOLVER.  Every wheel here is a
#: dependency OF cusolver, not a nicety: cusolver links cublas and cusparse,
#: and on Windows a wheel's DLLs are only visible to a compiled extension
#: through ``os.add_dll_directory``, never through PATH.
CUSOLVER_HINT = (
    "# only needed for eigensolver='library', or for an ensemble larger\n"
    "  # than this project's own kernel supports.  The default analysis\n"
    "  # does not use cuSOLVER at all, so this is optional.\n"
    "  #\n"
    "  # FIRST check whether it is installed but merely unreachable, which\n"
    "  # is the commoner fault and which installing again will not fix:\n"
    "  python -c \"import sys,pathlib;print([str(p) for p in pathlib.Path(sys.prefix).rglob('*cusolver*') if p.is_dir()])\"\n"
    "  # If that prints a directory, the library is present and the loader\n"
    "  # cannot see it.  On Linux put that directory on $LD_LIBRARY_PATH;\n"
    "  # on Windows PATH is not enough -- since Python 3.8 a compiled\n"
    "  # extension resolves its dependants through os.add_dll_directory()\n"
    "  # only.  If it prints nothing, install it:\n"
    "  pip install nvidia-cusolver-cu12 nvidia-cublas-cu12 nvidia-cusparse-cu12")


#: Solves one tiny symmetric eigenproblem two ways and reports which worked.
#: Deliberately the SAME two calls the analysis makes, because the fault this
#: check exists for is invisible to anything cheaper: cupy's core loads
#: independently of cuBLAS and cuSOLVER, so "cupy imports" and even "cupy
#: multiplies arrays" are both true on a box where the factorisation cannot
#: run at all.
#:
#: TWO PROPERTIES OF THIS SCRIPT ARE LOAD-BEARING.  Do not tidy them away.
#:
#: 1. It runs in a FRESH PROCESS (see :func:`_eigensolver_probe`), and
#: 2. it calls NOTHING from ``cupy.linalg`` before ``eigh``.
#:
#: Measured on a rented Ada node, 2026-08-05, cupy 14.1.1, where the wheel's
#: ``libcusolver.so.11`` was installed but off the loader path:
#:
#:     $ python -c "import cupy; cupy.linalg.eigh(A)"
#:     ImportError: libcusolver.so.11: cannot open shared object file
#:
#:     $ python -c "import cupy; cupy.linalg.inv(A); cupy.linalg.eigh(A)"
#:     ok
#:
#: ``inv`` reaches cuSOLVER through ``cupy_backends.cuda.libs.cusolver``,
#: whose own link resolution finds the wheel; batched ``eigh`` reaches it
#: through ``cupyx.cusolver``, whose does not -- and once the first call has
#: pulled the library into the process, the second one finds it already
#: loaded.  So cuSOLVER's availability on such a box is a property of what
#: the process happened to do first, which is why "it worked yesterday" is a
#: real thing people say about this fault.  A probe that shares a process
#: with anything else, or that warms up with a different factorisation,
#: reports green on a box where the analysis will fail.
_EIGENSOLVER_PROBE = """
import json, numpy, sys
out = {}
try:
    import cupy
except Exception as error:
    sys.stdout.write(json.dumps({"cupy": repr(error)}))
    raise SystemExit(0)
rs = numpy.random.RandomState(0)
b = rs.standard_normal((8, 6, 6))
a = b @ b.transpose(0, 2, 1) + 6.0 * numpy.eye(6)
a = 0.5 * (a + a.transpose(0, 2, 1))
try:
    from gpuwm.core.jacobi_eigh import batched_eigh
    w, v = batched_eigh(cupy.asarray(a))
    got = cupy.asnumpy(v @ (w[:, :, None] * v.transpose(0, 2, 1)))
    residual = float(numpy.abs(got - a).max() / numpy.abs(a).max())
    out["jacobi"] = "ok" if residual < 1e-12 else f"residual {residual:.2e}"
except Exception as error:
    out["jacobi"] = f"{type(error).__name__}: {error}"
try:
    cupy.linalg.eigh(cupy.asarray(a))
    out["library"] = "ok"
except Exception as error:
    out["library"] = f"{type(error).__name__}: {error}"
sys.stdout.write(json.dumps(out))
"""


def _eigensolver_probe() -> dict:
    """Run :data:`_EIGENSOLVER_PROBE` out of process; ``{}`` if it could not."""
    if find_spec("cupy") is None:
        return {"cupy": "not installed"}
    try:
        probe = subprocess.run(
            [sys.executable, "-c", _EIGENSOLVER_PROBE],
            capture_output=True, text=True, errors="replace",
            timeout=_PROBE_TIMEOUT_S * 4)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"probe": f"did not run: {error}"}
    try:
        return json.loads(probe.stdout or "{}")
    except ValueError:
        tail = [line for line in (probe.stderr or "").strip().splitlines()
                if line.strip()]
        return {"probe": tail[-1] if tail else f"exit {probe.returncode}"}


def _da_eigensolver_check() -> Check:
    """Can the radar-DA analysis factor its matrices, and with what?

    This is the one line in the report that exists because of a specific
    field failure rather than a category of them.  Twice -- once on a box
    whose NVIDIA wheels were installed but invisible to a compiled extension,
    once on a rented node whose CUDA install simply shipped without it -- the
    DA path died inside cuSOLVER, and both times the surrounding evidence
    said the GPU was healthy, because for everything except the
    factorisation it was.  ``gpuwm doctor`` now answers the question
    directly instead of leaving it to be inferred.

    Not blocking: an install that never runs data assimilation never touches
    either solver, and an install that does has this project's own kernel.
    """
    name = "radar-DA eigensolver"
    result = _eigensolver_probe()
    if "cupy" in result:
        return Check(
            name, "info",
            "not judged -- data assimilation runs on the device and cupy is "
            f"unavailable here ({result['cupy']})",
            brief="needs cupy", blocking=False)
    if "probe" in result:
        return Check(
            name, "missing",
            f"the probe could not be run: {result['probe']}",
            CUSOLVER_HINT, action="pip install nvidia-cusolver-cu12",
            brief=_short(result["probe"]), blocking=False)

    jacobi = result.get("jacobi", "did not report")
    library = result.get("library", "did not report")
    if jacobi == "ok":
        # The supported path works, so cuSOLVER is genuinely optional and
        # its absence is news rather than a gap.
        extra = ("cuSOLVER also available"
                 if library == "ok"
                 else f"cuSOLVER unavailable ({_short(library)}), which the "
                      "default analysis does not need")
        return Check(
            name, "verified",
            f"this project's batched Jacobi kernel solved and reconstructed "
            f"a test batch; {extra}",
            None if library == "ok" else CUSOLVER_HINT,
            action=None if library == "ok" else "pip install nvidia-cusolver-cu12",
            brief=("jacobi kernel ok, cuSOLVER ok" if library == "ok"
                   else "jacobi kernel ok, no cuSOLVER"),
            blocking=False)
    if library == "ok":
        return Check(
            name, "present",
            f"cuSOLVER works, but this project's own kernel did not: "
            f"{_short(jacobi)}.  Analyses still run, through "
            "eigensolver='library'",
            "# report this: the bundled kernel should build wherever cupy\n"
            "  # does.  Meanwhile set LetkfConfig(eigensolver='library')\n"
            "  # or run the analysis on the host",
            action="set eigensolver='library'",
            brief=_short(jacobi), blocking=False)
    return Check(
        name, "missing",
        f"neither solver works here -- bundled kernel: {_short(jacobi)}; "
        f"cuSOLVER: {_short(library)}.  Data assimilation cannot run on "
        "this device; forecasts are unaffected",
        CUSOLVER_HINT, action="pip install nvidia-cusolver-cu12",
        brief="no device eigensolver", blocking=False)


def _render_extra_check() -> Check:
    results = {"wrf-rust": _import_probe("wrf", "wrf-rust"),
               "matplotlib": _import_probe("matplotlib")}
    broken = {name: evidence for name, (ok, evidence) in results.items()
              if not ok}
    if not broken:
        versions = ", ".join(
            f"{name} {evidence}" for name, (_, evidence) in results.items())
        return Check("render extra (wrf-rust + matplotlib)", "verified",
                     f"imported in subprocesses ({versions})",
                     brief=_short(versions))
    if all(evidence == "not installed" for evidence in broken.values()):
        return Check(
            "render extra (wrf-rust + matplotlib)", "missing",
            f"{' and '.join(sorted(broken))} not installed -- gpuwm "
            "render needs the render extra", RENDER_EXTRA_HINT,
            action="pip install 'gpuwm[render]'",
            brief=f"{' and '.join(sorted(broken))} not installed",
            blocking=False)
    detail = "; ".join(f"{name}: {evidence}"
                       for name, evidence in sorted(broken.items()))
    return Check("render extra (wrf-rust + matplotlib)", "missing",
                 detail, RENDER_EXTRA_HINT,
                 action="pip install 'gpuwm[render]'", brief=_short(detail),
                 blocking=False)


# ---------------------------------------------------------------------------
# Bridge executables: probe-execute, not stat()
# ---------------------------------------------------------------------------

def _exec_probe(path: Path) -> tuple[bool, str]:
    """Launch ``path`` once; can this binary execute at all?

    The bridges are austere fail-closed CLIs without ``--version``:
    given a lone probe argument each prints its usage/error diagnostic
    and exits 1 or 2.  That observable -- the process launches, emits a
    diagnostic, exits with an orderly code -- separates a runnable
    executable from an empty, truncated, or wrong-platform file, which
    refuses to launch (OSError) or dies with an abnormal status
    (Windows NTSTATUS / signal) and no diagnostic of its own.

    The header is checked BEFORE anything is launched, because on
    Windows the expensive way of asking this question can never answer:
    ``subprocess.run``'s timeout bounds the wait and not
    ``CreateProcess``, and a file with a corrupt image header can raise
    a modal loader dialog inside that call which no timeout reaches.
    Two release batteries froze there, on a file containing sixteen
    ASCII characters.  :func:`gpuwm.bridges.launchable` answers the same
    question from the bytes; the launch that follows is bounded by the
    timeout AND by an error mode that makes the loader fail rather than
    prompt.  Nothing about a genuine bridge's verdict changed.
    """

    ok, evidence = bridges.launchable(path)
    if not ok:
        return False, f"{evidence} -- corrupt, truncated, or built for " \
                      "another platform"
    try:
        with bridges.quiet_loader_errors():
            probe = subprocess.run(
                [str(path), "--version"], capture_output=True, text=True,
                errors="replace", timeout=_PROBE_TIMEOUT_S)
    except OSError as error:
        return False, f"exists but failed to execute: {error}"
    except subprocess.TimeoutExpired:
        return False, (f"probe invocation did not exit within "
                       f"{_PROBE_TIMEOUT_S} s")
    if probe.returncode == 0:
        return True, "probe invocation exited 0"
    diagnostic = bool((probe.stderr or probe.stdout or "").strip())
    if probe.returncode in (1, 2) and diagnostic:
        return True, (f"executes (probe exit {probe.returncode} with its "
                      "usage diagnostic)")
    return False, (f"probe invocation exited {probe.returncode} without a "
                   "usage diagnostic -- corrupt or built for another "
                   "platform")


def _bridge_checks() -> list[Check]:
    checks: list[Check] = []
    crate = bridges.crate_dir() / "Cargo.toml"
    for name, consumer in _BRIDGE_CONSUMERS.items():
        try:
            found = bridges.find_bridge(name)
        except FileNotFoundError as error:
            checks.append(Check(
                f"bridge {name}", "missing", str(error),
                f"# {bridges.BRIDGE_ENV[name]} names a missing "
                "executable: point it at a real build, or unset it "
                "and build one --\n"
                + bridges.bridge_remedy(name),
                action=f"unset {bridges.BRIDGE_ENV[name]}, or point it "
                       "at a real build",
                brief=f"{bridges.BRIDGE_ENV[name]} names a missing file",
                group=_GROUP_BRIDGES))
            continue
        if found is not None:
            ok, evidence = _exec_probe(found)
            if ok:
                # It launches.  That was the whole check, and it is not
                # enough: the wheel ships no Rust, so an upgrade of the
                # Python half leaves yesterday's binaries in place, and
                # a bridge built before a contract change still launches
                # and still prints its usage line.  1.1.0 moved the GFS
                # series file to three columns and doctor blessed every
                # 1.0.1 bridge as `ok`, after which each preparation
                # died blaming the series file gpuwm had just written.
                ok, evidence = bridges.bridge_abi_matches(name, found)
            if ok:
                checks.append(Check(
                    f"bridge {name}", "verified", f"{found} -- {evidence}",
                    brief=_short(evidence), group=_GROUP_BRIDGES))
            else:
                checks.append(Check(
                    f"bridge {name}", "missing", f"{found} -- {evidence}",
                    f"# this one has to be replaced -- needed by: "
                    f"{consumer}\n"
                    + bridges.install_aware_build_hint(
                        bridges.CARGO_BUILD_HINT),
                    action=_build_action(), brief=_short(evidence),
                    group=_GROUP_BRIDGES))
        elif crate.is_file():
            checks.append(Check(
                f"bridge {name}", "missing",
                f"not built yet (checkout crate: {crate.parent})",
                f"# needed by: {consumer}\n{bridges.CARGO_BUILD_HINT}",
                action=bridges.CARGO_BUILD_HINT, brief="not built yet",
                group=_GROUP_BRIDGES))
        else:
            checks.append(Check(
                f"bridge {name}", "missing",
                "no checkout crate and no prebuilt executable "
                f"(searched {', '.join(str(c) for c in bridges.bridge_candidates(name))})",
                bridges.bridge_remedy(name)
                + f"\n  # needed by: {consumer}",
                action=_build_action(), brief="not staged",
                group=_GROUP_BRIDGES))
    return checks


def _fetch_backbone_check() -> Check:
    """The vendored Rust fetch backbone: probe-execute, not stat().

    ``gpuwm fetch --engine auto`` routes HRRR through this binary
    exactly when the check passes.  A missing one is not a gap in the
    estate -- the stdlib Python transport is the documented, always
    available fallback -- so an unbuilt backbone is ``info`` rather than
    ``missing``.  A binary that *exists* but reports a different
    fetch-record ABI is a different matter: that one fails after the
    download rather than before it, so it is reported ``missing``.
    """

    name = f"fetch backbone {rustwx_fetch.FETCH_NAME} (rust download engine)"
    try:
        found = rustwx_fetch.find_fetch_bin()
    except FileNotFoundError as error:
        return Check(
            name, "missing", str(error),
            f"# {rustwx_fetch.FETCH_ENV} names a missing executable: "
            "point it at a real build, or unset it and build one --\n"
            + rustwx_fetch.fetch_remedy(),
            action=f"unset {rustwx_fetch.FETCH_ENV}, or point it at a "
                   "real build",
            brief=f"{rustwx_fetch.FETCH_ENV} names a missing file",
            group=_GROUP_ENGINES)
    if found is None:
        crate = rustwx_fetch.crate_dir() / "Cargo.toml"
        if crate.is_file():
            detail = f"not built yet (checkout crate: {crate.parent})"
        else:
            detail = (
                "no checkout crate and no prebuilt executable (searched "
                + ", ".join(str(c)
                            for c in rustwx_fetch.fetch_candidates()) + ")")
        return Check(
            name, "info",
            detail + " -- gpuwm fetch falls back to the Python transport",
            bridges.install_aware_build_hint(
                rustwx_fetch.CARGO_BUILD_HINT, "tools/rustwx")
            + "\n  # enables gpuwm fetch --engine rust: parallel range "
            "GETs,\n  # the cross-process NOMADS rate governor, and "
            "--mode full-file",
            action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
            brief="not built; gpuwm fetch uses the Python transport",
            group=_GROUP_ENGINES)
    ok, evidence = rustwx_fetch.probe_fetch_bin(found)
    if not ok:
        return Check(
            name, "missing", f"{found} -- {evidence}",
            "# it has to be replaced:\n" + bridges.install_aware_build_hint(
                rustwx_fetch.CARGO_BUILD_HINT, "tools/rustwx"),
            action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
            brief=_short(evidence), group=_GROUP_ENGINES)
    return Check(name, "verified", f"{found} -- {evidence}",
                 brief=_short(evidence), group=_GROUP_ENGINES)


def _nexrad_front_door_check() -> Check:
    """The radar front door: is it here, and is it current?

    The one bundled binary with **no fallback**.  A missing fetch
    backbone leaves the Python transport and a missing renderer leaves
    matplotlib, so both are ``info``; there is no second way to turn a
    radar volume into observations, so an absent ``rw_nexrad`` is the
    difference between a box that can assimilate and a box that cannot.
    It is reported ``missing`` and it blocks, for the same reason the
    GRIB bridges do: it ships in the bundle ``gpuwm fetch-bridges``
    stages, so its absence means an incomplete install rather than an
    unexercised option.

    This check exists because the report used to pass without it.
    ``rw_nexrad`` was outside both audited sets -- not a GRIB bridge,
    not a render engine -- so ``gpuwm doctor`` printed a clean estate on
    installs where every radar route was dead, and the first news came
    from the launcher refusing to print a plan.

    Staleness is judged, not just presence.  The wrapper pins the exact
    ``--abi`` contract line it was written against, so a binary built
    before the live-chunk route fails the probe here rather than at the
    first live fetch -- and the remedy for that one is *rebuild*, never
    re-point, which is why the stale branch says so in those words.
    """

    # Imported here rather than at module scope, and the reason is what
    # this module IS.  `gpuwm.obs.nexrad` needs only the standard library,
    # but reaching it executes `gpuwm/obs/__init__.py`, which imports the
    # gridding stack and therefore numpy.  doctor is the tool for
    # diagnosing a broken or partial install, so a doctor that cannot be
    # imported without the full scientific stack cannot diagnose the
    # installs it exists for -- and the release pipeline proved it: the
    # bridges job imports this module for `_exec_probe` alone, installs no
    # dependencies, and died on `import numpy` three jobs deep.
    #
    # The failure branch is NOT a quiet skip.  This check was added
    # because the report used to pass without it, so a version that can
    # vanish when an import fails would reintroduce exactly the
    # silent-green hole it closes.  An unimportable obs stack is itself a
    # broken install, and it is reported as one.
    try:
        from gpuwm.obs import nexrad
    except ImportError as error:
        return Check(
            "radar front door (radar observation ingest)", "missing",
            f"gpuwm.obs is not importable ({error}) -- blocks ALL radar "
            "observation ingest, and means this install is incomplete "
            "rather than merely missing the binary",
            "# reinstall so the observation stack imports:\n"
            f"{REINSTALL_HINT}",
            action="reinstall gpuwm", brief="obs stack not importable",
            group=_GROUP_BRIDGES)

    name = f"radar front door {nexrad.NEXRAD_NAME} (radar observation ingest)"
    blocks = ("blocks ALL radar observation ingest -- every DA nowcast "
              "route, live and archived")
    try:
        found = nexrad.find_nexrad_bin()
    except FileNotFoundError as error:
        return Check(
            name, "missing", f"{error} -- {blocks}",
            f"# {nexrad.NEXRAD_ENV} names a missing executable: point it "
            "at a real build, or unset it and get one --\n"
            + nexrad.nexrad_remedy(),
            action=f"unset {nexrad.NEXRAD_ENV}, or point it at a real build",
            brief=f"{nexrad.NEXRAD_ENV} names a missing file",
            group=_GROUP_ENGINES)
    if found is None:
        crate = nexrad.crate_dir() / "Cargo.toml"
        if crate.is_file():
            detail = f"not built yet (checkout crate: {crate.parent})"
        else:
            detail = ("no checkout crate and no prebuilt executable "
                      "(searched "
                      + ", ".join(str(c)
                                  for c in nexrad.nexrad_candidates()) + ")")
        return Check(
            name, "missing", f"{detail} -- {blocks}",
            nexrad.nexrad_remedy()
            + "\n  # without it nothing can read a radar volume: no "
            "superobs,\n  # no analysis, no nowcast",
            action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
            brief="not staged; no radar ingest", group=_GROUP_ENGINES)
    ok, evidence = nexrad.probe_nexrad_bin(found)
    if not ok:
        return Check(
            name, "missing", f"{found} -- {evidence}; {blocks}",
            "# REBUILD it -- do not re-point GPUWM_RW_NEXRAD at another\n"
            "  # copy: this is a contract change, so every binary older\n"
            "  # than it fails the same way --\n"
            + bridges.install_aware_build_hint(
                nexrad.CARGO_BUILD_HINT, "tools/rustwx"),
            action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
            brief=_short(evidence), group=_GROUP_ENGINES)
    return Check(name, "verified", f"{found} -- {evidence}",
                 brief=_short(evidence), group=_GROUP_ENGINES)


def _region_dealias_check() -> Check:
    """The region-global dealiasing engine, which the default now needs.

    ``missing``, and the change of status from ``info`` is the finding:
    ``region-global`` became the shipped ``--dealias-engine`` on
    2026-08-12, so an absent library no longer costs an OPTION -- it
    costs every dealiased velocity in the DA nowcast, on the path a run
    reaches an hour in.  A default that needs a build is a default whose
    build blocks, the same rule that puts ``rw_nexrad`` here while the
    renderer and the fetch backbone stay informational.

    What it does NOT do is send the operator to a workaround: the remedy
    is the build, and ``--dealias-engine vad-region`` is named as what it
    is, a different solver with different decisions, not as a way around
    a missing file.
    """

    name = "region-global dealiasing engine (default --dealias-engine)"
    blocks = ("blocks the default velocity dealiasing path -- every "
              "--dealias run in the DA nowcast, live and archived")
    try:
        from gpuwm.obs import dealias_region
    except ImportError as error:                 # pragma: no cover - partial
        return Check(name, "missing",
                     f"gpuwm.obs is not importable ({error}) -- {blocks}",
                     "# reinstall so the observation stack imports:\n"
                     f"{REINSTALL_HINT}",
                     action="reinstall gpuwm",
                     brief="obs stack not importable",
                     group=_GROUP_ENGINES)
    try:
        found = dealias_region.find_region_bridge()
    except FileNotFoundError as error:
        return Check(
            name, "missing", str(error),
            f"# {dealias_region.REGION_DEALIAS_ENV} names a missing "
            "library: point it at a real build, or unset it --\n"
            + dealias_region.region_bridge_remedy(),
            action=(f"unset {dealias_region.REGION_DEALIAS_ENV}, or point it "
                    "at a real build"),
            brief=f"{dealias_region.REGION_DEALIAS_ENV} names a missing file",
            group=_GROUP_ENGINES)
    if found is None:
        return Check(
            name, "missing", f"not staged or built -- {blocks}",
            dealias_region.region_bridge_remedy()
            + "\n  # or `gpuwm fetch-bridges`, which stages it with the\n"
            "  # other prebuilt artifacts.  --dealias-engine vad-region\n"
            "  # runs without it, but it is a DIFFERENT solver: it\n"
            "  # abstains where this one resolves, and the engine that\n"
            "  # ran is recorded with the velocities it made",
            action=_build_action(dealias_region.CRATE_RELATIVE),
            brief="not staged; default dealiasing blocked",
            group=_GROUP_ENGINES)
    try:
        engine = dealias_region.load_region_dealiaser(found)
    except Exception as error:                   # noqa: BLE001 - reported
        return Check(
            name, "missing", f"{found} -- {error}",
            "# REBUILD it: this is a contract mismatch, so re-pointing\n"
            f"  # {dealias_region.REGION_DEALIAS_ENV} at another copy of the\n"
            "  # same vintage fails identically --\n"
            + dealias_region.region_bridge_remedy(),
            action=_build_action(dealias_region.CRATE_RELATIVE),
            brief=_short(str(error)), group=_GROUP_ENGINES)
    evidence = (f"ABI {engine.abi_version}, refinement API "
                f"{engine.rift_api_version}, upstream "
                f"{dealias_region.UPSTREAM_COMMIT[:12]}")
    return Check(name, "verified", f"{found} -- {evidence}",
                 brief=_short(evidence), group=_GROUP_ENGINES)


def _rust_renderer_check() -> Check:
    """The vendored Rusty Weather renderer: probe-execute, not stat().

    ``gpuwm render`` defaults to this engine exactly when the check
    passes; a missing or unrunnable binary is not a gap in the estate
    (matplotlib remains the documented fallback), so the statuses are
    ``verified``/``info`` rather than ``missing``.
    """

    name = f"renderer {rustwx.RENDERER_NAME} (rust render engine)"
    try:
        found = rustwx.find_renderer()
    except FileNotFoundError as error:
        return Check(
            name, "missing", str(error),
            f"# {rustwx.RENDERER_ENV} names a missing executable: "
            "point it at a real build, or unset it and build one --\n"
            + rustwx.renderer_remedy(),
            action=f"unset {rustwx.RENDERER_ENV}, or point it at a real "
                   "build",
            brief=f"{rustwx.RENDERER_ENV} names a missing file",
            group=_GROUP_ENGINES)
    if found is None:
        crate = rustwx.crate_dir() / "Cargo.toml"
        if crate.is_file():
            detail = f"not built yet (checkout crate: {crate.parent})"
        else:
            detail = ("no checkout crate and no prebuilt executable "
                      f"(searched {', '.join(str(c) for c in rustwx.renderer_candidates())})")
        return Check(
            name, "info",
            detail + " -- gpuwm render falls back to matplotlib",
            bridges.install_aware_build_hint(
                rustwx.CARGO_BUILD_HINT, "tools/rustwx")
            + "\n  # enables --engine rust and makes it the default",
            action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
            brief="not built; gpuwm render uses matplotlib",
            group=_GROUP_ENGINES)
    ok, evidence = rustwx.probe_renderer(found)
    if not ok:
        # Non-blocking on the exit code, per this function's own
        # contract: matplotlib remains the documented fallback.
        return Check(
            name, "missing", f"{found} -- {evidence}",
            "# it has to be replaced:\n" + bridges.install_aware_build_hint(
                rustwx.CARGO_BUILD_HINT, "tools/rustwx"),
            action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
            brief=_short(evidence), group=_GROUP_ENGINES,
            blocking=False)
    # Ask the question the renderer answers, in the renderer's own
    # order.  Probing gpuwm's checkout path alone reported "NO basemap
    # assets found" on every pip install -- including the ones where
    # rw_wrfbatch resolves the assets from its own build directory and
    # draws the coastlines the report says are missing.
    basemap = rustwx.resolve_basemap_dir(found)
    if basemap is None:
        basemap_note = ("NO basemap assets found -- charts render "
                        "without coast/state/county lines; set "
                        "RUSTWX_BASEMAP_DIR to a checkout's "
                        "tools/rustwx/assets/basemap")
    elif os.environ.get("RUSTWX_BASEMAP_DIR") or os.environ.get(
            "RUSTWX_ASSETS_DIR"):
        basemap_note = f"basemaps {basemap} (RUSTWX_* environment override)"
    else:
        basemap_note = f"basemaps {basemap}"
    return Check(name, "verified", f"{found} -- {evidence}; {basemap_note}",
                 brief=("built; no basemap assets" if basemap is None
                        else "built, with basemap assets"),
                 group=_GROUP_ENGINES)


def _provenance_check() -> Check:
    """Which tree is executing, and do its version claims agree?

    The estate report's own subject.  Every other check here asks
    whether some artifact beside gpuwm is usable; this one asks whether
    the gpuwm doing the asking can name itself -- which is the question
    that went unasked while a checkout sat 2,370 commits behind with an
    editable install reporting a version from a different tree.

    ``missing`` and blocking only for a genuine contradiction, which is
    :func:`gpuwm.provenance_gate.version_identity_refusal`'s definition
    and not "anything unusual": a wheel with no git, a clone nobody
    installed, and a checkout with a borrowed-but-agreeing number are
    all reported ``verified``, because none of them is wrong about
    anything.
    """

    from gpuwm.explain import split
    from gpuwm.provenance import resolve
    from gpuwm.provenance_gate import (executing_version,
                                       version_identity_refusal)

    name = "install provenance (which tree is executing)"
    try:
        prov = resolve()
        refusal = version_identity_refusal(prov)
    except Exception as error:                          # noqa: BLE001
        return Check(name, "missing",
                     f"provenance could not be resolved: {error}",
                     REINSTALL_HINT, action="pip install -e .",
                     brief=_short(str(error)))
    if refusal is not None:
        action, _ = split(refusal)
        return Check(
            name, "missing", action,
            f"# the two claims have to be re-bound:\n"
            f"  pip install -e {prov.source_root}\n"
            "  # (`gpuwm version` prints the same finding with the\n"
            "  #  upgrade path for a wheel install)",
            action=f"pip install -e {prov.source_root}",
            brief="version claims disagree")
    detail = f"{prov.install_kind} {executing_version()} at {prov.source_root}"
    git = prov.git or {}
    if git.get("commit"):
        detail += (f", git {git['commit']} on "
                   f"{git.get('branch') or 'a detached HEAD'}"
                   f" ({'dirty' if git.get('dirty') else 'clean'})")
    if prov.metadata_is_borrowed:
        # Not a refusal -- the digits agree -- but a reader comparing
        # receipts has to know the number came from elsewhere.
        detail += ("; NOTE its version string is read from another "
                   "install's metadata (no distribution provides this "
                   "code), and happens to agree")
    return Check(name, "verified", detail, brief=_short(detail))


def _renderer_tree_check() -> Check:
    """Does the resolved render engine belong to THIS tree?

    Separate from :func:`_rust_renderer_check`, which asks whether a
    renderer exists and runs.  Existence was the whole gate until now,
    and existence is what let a checkout borrow the engine some other
    tree had staged in the shared bridge directory -- a substitution
    with no message anywhere and a different product catalog at the far
    end of it.

    Non-blocking, deliberately.  ``gpuwm render --engine rust``
    REFUSES a foreign engine and ``--engine auto`` degrades to
    matplotlib, so the estate is not broken; this line is how a reader
    finds out before they run a render rather than after.
    """

    from gpuwm.provenance_gate import bridge_tree_match

    name = f"renderer tree match ({rustwx.RENDERER_NAME} vs this checkout)"
    try:
        found = rustwx.find_renderer()
    except FileNotFoundError as error:
        # _rust_renderer_check already reports this one in full.
        return Check(name, "info", str(error), brief=_short(str(error)),
                     group=_GROUP_ENGINES)
    match = bridge_tree_match(found, env_var=rustwx.RENDERER_ENV)
    if match.verdict == "absent":
        return Check(name, "info",
                     "no renderer resolved, so there is nothing to match",
                     brief="no renderer resolved", group=_GROUP_ENGINES)
    if match.matched:
        return Check(name, "verified", f"{match.verdict}: {match.basis}",
                     brief=match.verdict, group=_GROUP_ENGINES)
    return Check(
        name, "missing",
        f"{match.bridge} is from another tree -- {match.basis}, while "
        f"this checkout is at {(match.engine_commit or '')[:12]}",
        # Every non-comment line here must be a pasteable command --
        # `tests/test_doctor.py` enforces it, and a bare `VAR=value` is
        # neither a command nor portable between PowerShell and sh.  So
        # the declaration route is printed the way `bridges
        # .artifact_remedy` prints its own env-var route: commented.
        "# build the engine this tree matches:\n"
        + bridges.install_aware_build_hint(
            rustwx.CARGO_BUILD_HINT, "tools/rustwx")
        + "\n  # ... or, if that binary IS the one you mean, declare it:\n"
        f"  #   {rustwx.RENDERER_ENV}={match.bridge}",
        action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
        brief="renderer built from another commit",
        group=_GROUP_ENGINES, blocking=False)


def _cpu_library_check() -> Check:
    from gpuwm.ingest.cpu_backend import (
        CPU_BACKEND_ABI, CpuPreprocessBackend)

    # The tail is a `#` block, not prose fused onto the build command.
    # It used to read `... --offline  then copy it into <dir> or set
    # GPUWM_CPU_PREPROCESS_BRIDGE`, which a reader pastes whole and the
    # shell hands to cargo as arguments -- the exact failure the remedy
    # contract exists to stop, and a node-7 field finding.
    remedy = (bridges.install_aware_build_hint(bridges.CARGO_BUILD_HINT)
              + "\n  # then copy the built library into "
              f"{bridges.default_bridge_dir()},\n"
              "  # or set GPUWM_CPU_PREPROCESS_BRIDGE to its full path")
    try:
        backend = CpuPreprocessBackend()
    except FileNotFoundError as error:
        return Check(
            "cpu preprocess library", "missing",
            "gpuwm_preprocess_cpu shared library not found "
            "(--preprocess-backend cpu needs it; the CUDA backend does "
            f"not): {error}", remedy,
            action=_build_action(), brief="not staged",
            group=_GROUP_BRIDGES, blocking=False)
    except (OSError, RuntimeError, AttributeError) as error:
        return Check(
            "cpu preprocess library", "missing",
            f"found but not loadable as ABI v{CPU_BACKEND_ABI}: {error}",
            "# it has to be replaced:\n" + remedy,
            action=_build_action(),
            brief=f"not loadable as ABI v{CPU_BACKEND_ABI}",
            group=_GROUP_BRIDGES)
    path, abi = backend.path, backend.abi_version
    backend.close()
    # The same library carries the dealiaser's coarse VAD search.  A
    # library built before that entry point existed still serves every
    # interpolation call, so this is a note on the line and not a second
    # verdict -- but it is the difference between a radar volume
    # dealiased in seconds and one dealiased in half a minute, and a
    # user who cannot see which one they have cannot ask why.
    from gpuwm.obs.coarse_cost import unavailable_reason

    reason = unavailable_reason()
    search = ("radar coarse VAD search: native"
              if reason is None else
              f"radar coarse VAD search: NumPy ({reason})")
    return Check("cpu preprocess library", "verified",
                 f"{path} loaded via ctypes, ABI v{abi}; {search}",
                 brief=f"ABI v{abi}", group=_GROUP_BRIDGES)


# ---------------------------------------------------------------------------
# Packaged tables: the model's own validators, not directory counts
# ---------------------------------------------------------------------------

def _thompson_tables_check() -> Check:
    from gpuwm.core.thompson_contract import validate_table_assets
    from gpuwm.physics_compat import thompson_table_root
    from gpuwm.table_assets import (
        classify_assets, missing_externalized_assets)

    root = thompson_table_root()
    try:
        assets = validate_table_assets(root)
    except (FileNotFoundError, ValueError, OSError) as error:
        # The externalized assets (gpuwm.table_assets: the two largest
        # Thompson tables) are published as release assets rather than
        # shipped in the wheel; their absence has a one-command fix
        # that is not "reinstall".
        fetchable = missing_externalized_assets(Path(root))
        _valid, invalid, absent = classify_assets(Path(root))
        if fetchable and not invalid and len(absent) == len(fetchable):
            total_mib = sum(a.bytes for a in fetchable) / (1024 * 1024)
            return Check(
                "thompson tables", "missing",
                f"{', '.join(a.filename for a in fetchable)} not staged "
                f"at {root} (externalized: published as a release asset, "
                "not shipped in the package)",
                "gpuwm fetch-tables\n"
                f"  # one {total_mib:.0f} MiB download, SHA-256 verified "
                "against the\n"
                "  # packaged pins before install; --from stages it from a\n"
                "  # local directory instead, offline",
                action="gpuwm fetch-tables",
                brief=f"{len(fetchable)} externalized table(s) not staged "
                      f"({total_mib:.0f} MiB)")
        return Check(
            "thompson tables", "missing", str(error),
            REINSTALL_HINT
            + "\n  # if GPUWM_THOMPSON_TABLE_ROOT is set, point it at a"
            "\n  # byte-identical mirror of the packaged tables, or unset it",
            action="pip install -e .", brief=_short(str(error)))
    return Check(
        "thompson tables", "verified",
        f"{len(assets)} assets at {root} byte-validated (exact size + "
        f"SHA-256, {sum(asset.bytes for asset in assets):,} B), the same "
        "validation every mp8 run performs at load",
        brief=f"{len(assets)} assets byte-validated")


def _noah_tables_check() -> Check:
    try:
        from gpuwm.core.landuse import load_landuse_table
        from gpuwm.core.noah import load_tables

        tables = load_tables()
        landuse = load_landuse_table()
    except Exception as error:  # any parse/read failure is the finding
        return Check("noah tables", "missing",
                     f"packaged tables failed to parse: {error}",
                     REINSTALL_HINT, action="pip install -e .",
                     brief=_short(f"failed to parse: {error}"))
    return Check(
        "noah tables", "verified",
        "VEGPARM/SOILPARM/GENPARM parsed by the SOIL_VEG_GEN_PARM "
        f"transcription ({tables.lucats} vegetation / {tables.slcats} "
        f"soil categories); LANDUSE.TBL parsed ({landuse.lucats} "
        "categories)",
        brief="VEGPARM/SOILPARM/GENPARM/LANDUSE parsed")


# ---------------------------------------------------------------------------
# Data roots
# ---------------------------------------------------------------------------

def geography_gaps(geog: Path) -> list[Check]:
    """The reasons a staged WPS_GEOG tree cannot build static fields.

    Empty means usable.  Published because ``gpuwm go`` needs the same
    answer doctor gives, and the alternative -- a second existence test
    written at the call site -- is how the two came to disagree in the
    first place: doctor named the remedy while ``go`` passed
    ``--geog-root`` through unexamined and let ``rw-wps`` fail on a
    missing ``index`` file, after the fetch stage had downloaded the
    forcing.  One check, two readers.
    """

    return [check for check in _geog_tree_checks(geog)
            if check.status == "missing"]


def _geog_tree_checks(geog: Path) -> list[Check]:
    """Check one staged WPS_GEOG tree, wherever it was resolved from.

    The dataset list comes from :mod:`gpuwm.geog_assets` -- the module
    that stages the tree -- rather than from the domain wizard, which
    also declares it.  The two are one list: ``geog_assets`` derives its
    order from :data:`gpuwm.geog_assets.GEOG_ARCHIVES` and
    ``tests/test_fetch_geog.py`` asserts it equals
    ``gpuwm.domain_wizard.GEOG_DATASETS`` exactly, so this is the same
    nine names read from the half doctor's remedy names.

    Reading them from the wizard instead was also the one thing that
    made this report unreachable from a preprocessing-only install: the
    wizard imports the memory preflight and ``gpuwm.cli``, so the
    standalone RW-WPS wheel could not stage a module that named it, and
    its package-boundary scan said so.
    """

    from gpuwm.geog_assets import geog_datasets

    GEOG_DATASETS = geog_datasets()

    checks: list[Check] = []
    if not geog.is_dir():
        # A tree nobody has fetched yet is the documented state of a
        # fresh install (the ~16 GB download is an explicit opt-in), so
        # it reports but does not fail the exit code.  A *partial* tree
        # below stays blocking -- that one is a corrupted download.
        return [Check(
            "WPS_GEOG", "missing",
            f"{geog} does not exist (the default geog_root).  Nothing "
            "that builds static fields can run without it",
            GEOG_HINT, action="gpuwm fetch-geog",
            brief=f"{geog} does not exist", blocking=False)]
    absent = sorted(name for name in GEOG_DATASETS
                    if not (geog / name).is_dir())
    unindexed = sorted(
        name for name in GEOG_DATASETS
        if (geog / name).is_dir() and not (geog / name / "index").is_file())
    if absent or unindexed:
        problems = []
        if absent:
            problems.append(f"missing dataset directorie(s): "
                            f"{', '.join(absent)}")
        if unindexed:
            problems.append(
                "dataset(s) without their WPS `index` file (empty or "
                f"partial download): {', '.join(unindexed)}")
        checks.append(Check(
            "WPS_GEOG", "missing", f"{geog}: " + "; ".join(problems),
            GEOG_HINT, action="gpuwm fetch-geog",
            brief=f"{len(absent) + len(unindexed)} of "
                  f"{len(GEOG_DATASETS)} dataset(s) absent or unindexed"))
    else:
        checks.append(Check(
            "WPS_GEOG", "verified",
            f"{geog} carries all {len(GEOG_DATASETS)} required datasets, "
            "each with its WPS index file",
            brief=f"all {len(GEOG_DATASETS)} datasets indexed"))
    return checks


def _case_data_root_check() -> list[Check]:
    """The case-data root AND the static geography under it.

    v1.0.0 returned a single ``info`` when GPUWM_CASE_DATA_ROOT was
    unset and never looked for WPS_GEOG at all, so doctor printed "no
    gaps; every check verified" on a machine with zero static geography
    -- contradicting the README, which says doctor requires each
    dataset's index file, and greenlighting a box on which nothing
    downstream could run.  An unset variable is not an excuse to skip
    the check: ``fetch-geog`` and every config default resolve to the
    same place through :func:`gpuwm.geog_assets.default_geog_root`, so
    doctor looks exactly there.
    """

    from gpuwm.geog_assets import default_geog_root

    raw = os.environ.get("GPUWM_CASE_DATA_ROOT")
    if not raw:
        return [
            Check(
                "GPUWM_CASE_DATA_ROOT", "info",
                "not set.  Layout when you set it: the root is the "
                "directory that CONTAINS your case bundles and (by "
                "default) WPS_GEOG -- configs reference "
                "${GPUWM_CASE_DATA_ROOT}/<bundle>/... and geog_root "
                "defaults to ${GPUWM_CASE_DATA_ROOT}/WPS_GEOG",
                brief=f"not set; geog_root defaults to "
                      f"{default_geog_root()}"),
            *_geog_tree_checks(default_geog_root()),
        ]
    root = Path(raw)
    if not root.is_dir():
        return [Check(
            "GPUWM_CASE_DATA_ROOT", "missing",
            f"set to {raw} but that directory does not exist",
            "# point GPUWM_CASE_DATA_ROOT at the directory that CONTAINS\n"
            "  # your case bundles and WPS_GEOG -- there is no command to\n"
            "  # print here, only your path",
            action="point GPUWM_CASE_DATA_ROOT at an existing directory",
            brief=f"set to {raw}, which does not exist")]
    return [
        Check("GPUWM_CASE_DATA_ROOT", "present",
              f"{root} (directory exists; its datasets are checked "
              "individually below)", brief=str(root)),
        *_geog_tree_checks(root / "WPS_GEOG"),
    ]


def _distribution_manifest_check() -> Check:
    """The sealed manifest: the WHOLE schema, then the artifacts.

    This used to check ``schema`` and ``status`` and stop.  A field user
    hand-authored a two-key document to get past an unrelated bug, got a
    green line here, and died several minutes into a preparation at
    ``contract.platform.backends`` -- a key doctor had never looked at.
    The validator it calls now is the same one every consumer calls, and
    it names every missing field at once.
    """

    from gpuwm.runtime_manifest import RUNTIME_SCHEMA, manifest_defects

    name = "GPUWM_NATIVE_DISTRIBUTION_MANIFEST"
    raw = os.environ.get(name)
    if not raw:
        return Check(
            name, "info",
            "not set (normal for source clones and pip installs; only "
            "sealed runtime archives set it to bind their decoder "
            "inventory)",
            brief="not set (normal for clones and pip installs)")
    path = Path(raw)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as error:
        return Check(
            name, "missing",
            f"set to {raw} but not a readable JSON document: {error}",
            f"# unset {name} unless you are running a sealed runtime\n"
            "  # archive, whose installer sets it correctly",
            action=f"unset {name}", brief=_short(f"unreadable: {error}"))
    defects = manifest_defects(payload)
    if defects:
        shown = "; ".join(defects[:4])
        more = f" (+{len(defects) - 4} more)" if len(defects) > 4 else ""
        return Check(
            name, "missing",
            f"{path}: not a complete {RUNTIME_SCHEMA} document -- "
            f"{len(defects)} problem(s): {shown}{more}",
            f"# unset {name} unless you are running a sealed runtime\n"
            "  # archive, whose installer writes this document beside its\n"
            "  # artifacts.  A clone or a pip install binds its identity\n"
            "  # from git or from the installed wheel and needs no manifest.",
            action=f"unset {name}",
            brief=f"{len(defects)} schema problem(s): {_short(defects[0], 40)}")
    # A non-empty per-artifact inventory is part of the required schema
    # now, so the old "READY but declares no hashes -- presence only"
    # branch is unreachable: such a document is refused above, by name.
    declared = payload["payload"]
    root = path.resolve().parent
    failures: list[str] = []
    verified = 0
    for relative, record in sorted(declared.items()):
        expected = record.get("sha256") if isinstance(record, dict) else None
        if not isinstance(expected, str):
            failures.append(f"{relative}: malformed manifest record")
            continue
        artifact = root / relative
        if not artifact.is_file():
            failures.append(f"{relative}: missing")
            continue
        expected_bytes = record.get("bytes")
        if (isinstance(expected_bytes, int)
                and artifact.stat().st_size != expected_bytes):
            failures.append(f"{relative}: size mismatch")
            continue
        if _sha256(artifact) != expected:
            failures.append(f"{relative}: sha256 mismatch")
            continue
        verified += 1
    if failures:
        shown = "; ".join(failures[:5])
        more = f" (+{len(failures) - 5} more)" if len(failures) > 5 else ""
        return Check(
            name, "missing",
            f"{path}: {len(failures)} of {len(declared)} declared "
            f"artifacts failed revalidation: {shown}{more}",
            "# re-extract the sealed runtime archive (its installer wrote\n"
            f"  # this manifest beside its artifacts), or unset {name}",
            action="re-extract the sealed runtime archive, or unset "
                   f"{name}",
            brief=f"{len(failures)} of {len(declared)} artifacts failed "
                  "revalidation")
    return Check(
        name, "verified",
        f"{path}: READY; all {verified} declared artifacts re-hashed "
        "and match",
        brief=f"READY; {verified} artifacts re-hashed")


# ---------------------------------------------------------------------------
# The paths a RUN resolves: provenance, importability, per-source decoders
# ---------------------------------------------------------------------------

def _install_identity_check() -> Check:
    """Can this install name itself?  Every run's receipt asks it.

    Not a nicety.  ``_source_identity`` runs before any data is read and
    used to be a bare ``git rev-parse`` with the working directory set
    to the directory holding the package -- ``site-packages`` on a pip
    install, where git exits 128.  A field user hit
    ``CalledProcessError: returned non-zero exit status 128`` on the
    first preparation of a green install; nothing in the estate report
    had asked the question that fails.  So doctor asks it, here, by
    calling the exact resolver the run calls.
    """

    from gpuwm.runtime_manifest import IdentityError, provenance

    root = Path(__file__).resolve().parent.parent
    name = "run provenance (identity every receipt binds)"
    try:
        identity = provenance(root)
    except IdentityError as error:
        return Check(
            name, "missing", str(error),
            REINSTALL_HINT
            + "\n  # a run records what produced it; with no distribution\n"
            "  # metadata and no checkout there is nothing to record",
            action="pip install -e .", brief=_short(str(error)))
    except Exception as error:  # a manifest defect, reported in full below
        return Check(
            name, "missing",
            f"the bound distribution manifest is unusable: {error}",
            "# see the GPUWM_NATIVE_DISTRIBUTION_MANIFEST line below",
            action="unset GPUWM_NATIVE_DISTRIBUTION_MANIFEST",
            brief=_short(str(error)))
    source = identity["identity_source"]
    if source == "git":
        evidence = f"git checkout {str(identity['git_commit'])[:12]} at {root}"
    elif source == "gpuwm-native-distribution-manifest":
        evidence = ("sealed runtime manifest "
                    f"{str(identity['git_commit'])[:12]}")
    else:
        wheel = identity["installed_wheel"] or {}
        evidence = (f"installed wheel {wheel.get('distribution_name')} "
                    f"{wheel.get('distribution_version')} "
                    f"({wheel.get('record_file_count')} RECORD entries, "
                    f"aggregate "
                    f"{str(wheel.get('record_aggregate_sha256'))[:12]})")
    return Check(name, "verified", f"{source}: {evidence}",
                 brief=_short(f"{source}: {evidence}"))


#: Modules a data route launches in a fresh interpreter, from whatever
#: directory the user happens to be standing in.
_ROUTE_ENTRY_MODULES = (
    "tools.prepare_hrrr_wrf",
    "tools.hrrr_single_domain_benchmark",
)


def _non_git_import_check() -> Check:
    """Do the route entry points import from a directory that is not a repo?

    The wrapper scripts are packaged modules, but they resolve paths and
    provenance relative to the installed tree, and one of them used to
    do that with a subprocess whose working directory decided the
    answer.  Importing them from a scratch directory -- which is where a
    user actually runs -- is the cheapest possible proof that neither
    the current directory nor the absence of a repository around it
    changes whether they load.
    """

    import tempfile

    name = "route entry points (import from a non-repository directory)"
    code = "".join(f"import {module}\n" for module in _ROUTE_ENTRY_MODULES)
    # The probe's import path is the directory that holds the ``gpuwm``
    # package, which is what BOTH installs give a run: site-packages is
    # already on ``sys.path`` for a wheel, and a checkout root is what
    # every documented developer invocation supplies.  Seeding it keeps
    # the question about the working directory -- the thing that broke
    # -- rather than about whether a clone happens to be pip-installed.
    root = str(Path(__file__).resolve().parent.parent)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        root + os.pathsep + environment.get("PYTHONPATH", "")).rstrip(
            os.pathsep)
    with tempfile.TemporaryDirectory(prefix="gpuwm-doctor-cwd-") as scratch:
        try:
            probe = subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True,
                errors="replace", cwd=scratch, env=environment,
                timeout=_PROBE_TIMEOUT_S * 4)
        except (OSError, subprocess.TimeoutExpired) as error:
            return Check(
                name, "missing", f"the import probe failed to run: {error}",
                REINSTALL_HINT, action="pip install -e .",
                brief=_short(str(error)))
    if probe.returncode == 0:
        return Check(
            name, "verified",
            f"{', '.join(_ROUTE_ENTRY_MODULES)} imported from a scratch "
            "directory outside any git repository",
            brief=f"{len(_ROUTE_ENTRY_MODULES)} modules import cleanly")
    tail = [line for line in (probe.stderr or "").strip().splitlines()
            if line.strip()]
    reason = tail[-1] if tail else f"exit {probe.returncode}"
    return Check(
        name, "missing", f"failed to import: {reason}",
        REINSTALL_HINT
        + "\n  # the preparation wrappers must load from any directory;\n"
        "  # an install that only works inside a checkout is incomplete",
        action="pip install -e .", brief=_short(reason))


#: Doctor's per-source route checks, by the ``--source`` name.  These are
#: public data products, not cases.
DOCTOR_SOURCES = ("gfs", "hrrr")

#: Fold key for one source's route findings.
_GROUP_ROUTE = "route"


def _decoder_route_check(source: str) -> Check:
    """THE decoder that source's preparation will launch, resolved here.

    Resolved by calling :func:`gpuwm.bridges.resolve_source_decoder` --
    the function the preparation wrapper itself calls -- rather than by
    re-deriving a path.  Doctor reporting one path while preparation
    used another is not a hypothetical: the wrapper resolved a source
    checkout's cargo workspace under ``site-packages``, so a wheel
    install got "no gaps" here and ``HRRR decoder is missing`` there.
    """

    name = f"{source} route decoder"
    try:
        found = bridges.resolve_source_decoder(source)
    except FileNotFoundError as error:
        message = str(error)
        headline, _, remedy = message.partition("\n")
        return Check(
            name, "missing", headline,
            remedy or bridges.bridge_remedy(bridges.SOURCE_DECODERS[source]),
            action=_build_action(), brief="not resolvable",
            group=_GROUP_ROUTE)
    except bridges.DecoderContractError as error:
        # The resolver owns the contract check now, so this arm reports
        # what preparation would refuse rather than re-deriving it.  A
        # stale binary reaches exactly one verdict, from one place.
        headline, _, remedy = str(error).partition("\n")
        return Check(
            name, "missing", headline,
            "# this one has to be replaced before the "
            f"{source} route can run:\n" + remedy,
            action=_build_action(),
            brief=_short("does not speak this release's contract"),
            group=_GROUP_ROUTE)
    executable = bridges.SOURCE_DECODERS[source]
    # Still asked, still reported: the resolver guarantees this is True,
    # and the evidence string is what the verified line says out loud.
    _, evidence = bridges.bridge_abi_matches(executable, found)
    return Check(
        name, "verified",
        f"preparation will launch {found} ({evidence})",
        brief=_short(str(found)), group=_GROUP_ROUTE)


def _hrrr_fetch_path_check() -> Check:
    """Which byte transport ``gpuwm fetch --source hrrr`` will use.

    Reported because the difference is 16x on a real link and was
    invisible until after the download: the Python fallback can only do
    ``.idx`` range subsets, and one 419 MB file took 560 s that way
    against 27-35 s taken whole.  Not a gap -- the fallback is the
    documented always-available route and the fetch-backbone line above
    already carries the remedy -- so this is ``info`` when it is slow
    and ``verified`` when it is not.
    """

    from gpuwm import fetch, rustwx_fetch

    name = "hrrr route fetch transport"
    try:
        found = rustwx_fetch.find_fetch_bin()
        usable = found is not None and rustwx_fetch.probe_fetch_bin(found)[0]
    except FileNotFoundError:
        found, usable = None, False
    if usable:
        return Check(
            name, "verified",
            f"engine rust ({found}), default --mode {fetch.HRRR_DEFAULT_MODE}"
            " (whole objects in parallel range GETs)",
            brief=f"rust, {fetch.HRRR_DEFAULT_MODE}", group=_GROUP_ROUTE)
    return Check(
        name, "info",
        "engine python (the rust backbone is not usable here), which can "
        "only do .idx range subsets -- correct, and roughly an order of "
        "magnitude slower per file",
        "gpuwm setup\n"
        "  # stages the rust fetch backbone; whole-file transfers need it",
        action="gpuwm setup", brief="python transport (.idx subsets only)",
        group=_GROUP_ROUTE)


def _gfs_fetch_path_check() -> Check:
    """Which byte transports ``gpuwm fetch --source gfs`` can use.

    Two first-class transports, reported the way the HRRR line is: the
    default NOMADS grib-filter crop (governed stdlib HTTP, always
    available) and ``--mode full-file`` (whole pgrb2.0p25 objects from
    the S3 archive, preferring the rust backbone's parallel range
    GETs).  The report that prompted this line showed doctor naming the
    hrrr route while ``gfs_grib2_bridge`` sat verified two rows up with
    no route reported around it.
    """

    from gpuwm import rustwx_fetch

    name = "gfs route fetch transport"
    try:
        found = rustwx_fetch.find_fetch_bin()
        usable = found is not None and rustwx_fetch.probe_fetch_bin(found)[0]
    except FileNotFoundError:
        found, usable = None, False
    if usable:
        return Check(
            name, "verified",
            "default: NOMADS grib-filter crop (governed stdlib HTTP); "
            f"--mode full-file: whole S3 objects, engine rust ({found})",
            brief="cgi-subset default; full-file via rust",
            group=_GROUP_ROUTE)
    return Check(
        name, "verified",
        "default: NOMADS grib-filter crop (governed stdlib HTTP); "
        "--mode full-file: whole S3 objects through the stdlib "
        "transport (the rust backbone is not usable here; `gpuwm "
        "setup` stages the faster parallel-range-GET engine)",
        brief="cgi-subset default; full-file via python",
        group=_GROUP_ROUTE)


def _source_route_checks(source: str) -> list[Check]:
    """Everything one data route resolves before it reads a byte."""

    if source not in DOCTOR_SOURCES:
        raise ValueError(f"unknown doctor source {source!r}; known: "
                         f"{list(DOCTOR_SOURCES)}")
    checks = [_decoder_route_check(source)]
    if source == "hrrr":
        checks.append(_hrrr_fetch_path_check())
    if source == "gfs":
        checks.append(_gfs_fetch_path_check())
    return checks


def collect_checks(sources: tuple[str, ...] | None = None) -> list[Check]:
    """The estate, plus every named data route's own resolution.

    ``sources=None`` means every route this build knows, which is what
    a bare ``gpuwm doctor`` runs: the report that said "no gaps" before
    a route died on a path it had never resolved is the reason these
    are in the default estate rather than behind a flag.
    """

    selected = DOCTOR_SOURCES if sources is None else tuple(sources)
    checks: list[Check] = []
    version = ".".join(str(v) for v in sys.version_info[:3])
    if sys.version_info >= (3, 11):
        checks.append(Check("python", "verified", f"{version} (>= 3.11)",
                            brief=f"{version} (>= 3.11)"))
    else:
        checks.append(Check(
            "python", "missing", f"{version} is below the 3.11 floor",
            "# install Python 3.11 or newer -- which installer is right\n"
            "  # here depends on how this Python was installed",
            action="install Python 3.11 or newer",
            brief=f"{version} is below the 3.11 floor"))
    checks.append(_cupy_check())
    checks.append(_da_eigensolver_check())
    checks.append(_render_extra_check())
    checks.append(_rust_renderer_check())
    checks.append(_renderer_tree_check())
    checks.append(_fetch_backbone_check())
    checks.append(_nexrad_front_door_check())
    checks.append(_region_dealias_check())
    checks.extend(_bridge_checks())
    checks.append(_cpu_library_check())
    checks.append(_thompson_tables_check())
    checks.append(_noah_tables_check())
    checks.extend(_case_data_root_check())
    checks.append(_distribution_manifest_check())
    checks.append(_provenance_check())
    checks.append(_install_identity_check())
    checks.append(_non_git_import_check())
    for source in selected:
        checks.extend(_source_route_checks(source))
    return checks


#: Column the first remedy line starts at: ten spaces of gutter plus
#: the ``remedy: `` label itself.  Continuation lines match it.
_REMEDY_LABEL = "          remedy: "

#: Status -> the label both reports print.  One vocabulary, so a reader
#: who runs the terse form and then ``--explain`` is reading the same
#: four words in the same column.
_LABELS = {"verified": "ok     ", "present": "present",
           "missing": "MISSING", "info": "info   "}


def _remedy_block(remedy: str) -> list[str]:
    """The remedy, every physical line aligned under ``remedy:``.

    Only the first line used to be indented; the rest arrived with
    whatever leading whitespace the composer happened to give them (0,
    2 or 4 spaces, three different composers), so a report showed
    commands hanging at column 0 under a label at column 18 and read as
    if the block had ended.  Both shells ignore leading whitespace, so
    aligning the whole block costs the reader nothing on paste.
    """

    lines = remedy.splitlines() or [remedy]
    block = [_REMEDY_LABEL + lines[0].strip()]
    block += [" " * len(_REMEDY_LABEL) + line.strip() if line.strip() else ""
              for line in lines[1:]]
    return block


def _install_headline() -> str:
    """Which copy of gpuwm produced this report.

    A report that diagnoses an estate has to say whose estate, and the
    version number alone does not: an editable install shadows every
    wheel pip writes, so a reader can be looking at findings from code
    that is months older than the version they think they upgraded to.
    No network here -- `gpuwm version` owns the index comparison; this
    is the local identity only, and it never fails a report it is only
    the header of.
    """

    try:
        from gpuwm.version_cli import headline

        return headline()
    except Exception as error:                          # noqa: BLE001
        return f"(install identity unavailable: {type(error).__name__})"


def format_report(checks: list[Check]) -> str:
    """The full report: every finding's evidence and whole remedy block.

    This is what ``gpuwm doctor --explain`` prints, unchanged from when
    it was the only thing doctor printed.  The remedy blocks in
    particular are not summarized anywhere: they are the text a reader
    pastes, and a paraphrase of a command is not a command.
    """

    lines = ["gpuwm doctor: runtime estate"]
    for check in checks:
        lines.append(f"  {_LABELS[check.status]} {check.name}: "
                     f"{check.detail}")
        if check.remedy:
            lines.extend(_remedy_block(check.remedy))
    gaps = sum(1 for check in checks if check.status == "missing")
    blocking = len(blocking_gaps(checks))
    opt_in = gaps - blocking
    presence_only = sum(1 for check in checks if check.status == "present")
    if gaps:
        # Not "every remedy is copy-pasteable": a remedy can be a
        # sequence of steps, and on a pip install the bridge remedy is a
        # clone-and-build rather than a single line.  Claiming one line
        # when six were printed is the kind of small lie that costs the
        # reader their trust in the other five.
        #
        # And the count is SPLIT, because the exit code splits it.  A
        # summary reading "1 gap(s)" out of a process that exits 0 is a
        # report disagreeing with itself, and a fleet node comparing
        # 1.3.1 (exit 1) with 1.4.0 (exit 0) on the identical gap text
        # reasonably read the pair as a silent regression.  The exit code
        # is the right one -- an opt-in nobody opted into is not a fault
        # -- so the severity is what has to be visible.
        lines.append(
            f"gpuwm doctor: {gaps} gap(s), {blocking} of them blocking "
            f"(the exit code is {1 if blocking else 0})"
            + (f"; the other {opt_in} is/are opt-in pieces this install "
               "has not staged, each a documented choice with its own "
               "command above" if opt_in else "")
            + ".  Every remedy line above is either a command to run as "
              "printed, in the order printed, or a '#' comment.")
    elif presence_only:
        lines.append(f"gpuwm doctor: no gaps ({presence_only} check(s) "
                     "presence-only as labeled, the rest verified).")
    else:
        lines.append("gpuwm doctor: no gaps; every check verified.")
    return "\n".join(lines)


def _fold(checks: list[Check]) -> list[tuple[Check, int]]:
    """Collapse runs of same-status, same-group, same-action checks.

    A pip install gaps every Rust artifact at once, and the untidy
    consequence is six identical ``-> gpuwm fetch-bridges`` lines that
    read as six separate problems needing six separate fixes.  Folding
    is presentation only: the fold is keyed on the three things that
    make two lines say the same thing, and each member survives intact
    in ``--explain`` and ``--json``.

    Only ADJACENT checks fold, so the report keeps the order
    :func:`collect_checks` chose -- a group interrupted by an unrelated
    check stays two entries rather than silently reordering the report.
    """

    folded: list[tuple[Check, int]] = []
    for check in checks:
        if folded and check.group is not None:
            last, count = folded[-1]
            if (last.group == check.group and last.status == check.status
                    and last.action == check.action):
                folded[-1] = (last, count + 1)
                continue
        folded.append((check, 1))
    return folded


def _setup_summary(gaps: list[Check]) -> str | None:
    """``gpuwm setup`` as the next step, when it is genuinely shorter.

    Named only when more than one gap would be closed by it -- which is
    what a fresh install looks like from here.  A single gap already
    printed its own one command above; sending that reader to a wrapper
    that runs two steps to fix one is longer, not shorter.
    """

    wanted = {check.action for check in gaps
              if check.action in SETUP_ACTIONS}
    if len(wanted) < 2:
        return None
    return ("gpuwm setup runs " + " then ".join(
        action for action in SETUP_ACTIONS if action in wanted)
        + " in order.")


def format_brief(checks: list[Check]) -> str:
    """The default report: one line per finding, and the next command.

    Everything :func:`format_report` prints is still true and still
    reachable; this is the same findings at the width of a glance.  The
    rule for each line is what a reader in a hurry needs: the status,
    what it is about, and -- when there is something to do -- the single
    command that does it.
    """

    lines = ["gpuwm doctor"]
    for check, count in _fold(checks):
        name = check.name if count == 1 else f"{check.group} ({count})"
        line = f"  {_LABELS[check.status]} {name}"
        if count == 1 and check.brief:
            line += f": {check.brief}"
        if check.action:
            line += f"  -> {check.action}"
        lines.append(line)
    gaps = [check for check in checks if check.status == "missing"]
    blocking = blocking_gaps(checks)
    if gaps:
        # Say how many of them the exit code is about.  See the note in
        # format_report: "1 gap(s)" beside exit 0 is the whole of the
        # reported 1.3.1 -> 1.4.0 "regression".
        summary = (f"gpuwm doctor: {len(gaps)} gap(s), {len(blocking)} "
                   f"of them blocking (exit "
                   f"{1 if blocking else 0}).")
        setup = _setup_summary(gaps)
        if setup:
            summary += " " + setup
        lines.append(summary)
        lines.append("Run `gpuwm doctor --explain` for the full remedy "
                     "for each line, with the evidence behind it.")
    else:
        lines.append("gpuwm doctor: no gaps.")
    return "\n".join(lines)


def blocking_gaps(checks: list[Check]) -> list[Check]:
    """The gaps that justify a nonzero exit: broken or integrity-suspect
    findings, never the absent-optional ones (see :class:`Check`).

    This split is the whole answer to the reported ``gpuwm setup`` exit
    regression (1.3.1 exited 1 on an unfetched WPS_GEOG tree; 1.4.0 exits
    0 on the identical gap text).  Exiting 0 is correct -- that download
    is an explicit ~16 GB opt-in, and an install that did everything its
    documentation asked must not fail an installer.  What was wrong was
    that both reports printed the same "1 gap(s)" line, so the only
    visible difference between the two versions was the exit code.  The
    severity is printed now; the exit code did not move."""

    return [check for check in checks
            if check.status == "missing" and check.blocking]


def doctor_main(args) -> int:
    sources = getattr(args, "source", None) or None
    checks = collect_checks(tuple(sources) if sources else None)
    if getattr(args, "json", False):
        print(json.dumps(
            [check.__dict__ for check in checks], indent=2))
        return 1 if blocking_gaps(checks) else 0
    # WHICH copy of gpuwm produced this report.  Printed here rather
    # than folded into format_report/format_brief because those two are
    # pinned verbatim by a golden test, and rightly: --explain's promise
    # is that the long form comes back unchanged.  A header is a
    # property of the printed command, not of the report text.
    #
    # It goes FIRST because it reframes everything under it: an estate
    # diagnosed from an editable install months behind the version the
    # reader thinks they upgraded to is a different report.
    print(_install_headline())
    print(format_report(checks) if explain_enabled(args)
          else format_brief(checks))
    return 1 if blocking_gaps(checks) else 0


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "doctor",
        help="verify the runtime estate for real (subprocess imports of "
             "cupy/wrf/matplotlib, bridge probe executions, ctypes load "
             "of the CPU library, table hash/parse validation, WPS_GEOG "
             "index files) and print one line per item with the command "
             "that closes each gap (--explain for the full remedies)")
    parser.add_argument("--json", action="store_true",
                        help="emit the checks as JSON")
    parser.add_argument(
        "--source", action="append", choices=sorted(DOCTOR_SOURCES),
        metavar="SOURCE",
        help="report only this data route's own resolution (repeatable) "
             "alongside the shared estate: the exact decoder its "
             "preparation will launch, and the byte transport its fetch "
             "will use.  Omitted, every route this build knows is "
             f"reported ({', '.join(sorted(DOCTOR_SOURCES))})")
    parser.set_defaults(func=doctor_main)
    return parser


__all__ = ["Check", "DOCTOR_SOURCES", "SETUP_ACTIONS", "collect_checks",
           "doctor_main", "format_brief", "format_report", "geography_gaps",
           "register_cli"]
