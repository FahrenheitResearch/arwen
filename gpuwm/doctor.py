"""``gpuwm doctor``: verify the runtime estate, print exact remedies.

The pip package deliberately splits its runtime across estates the
installer cannot see from ``pip install`` alone: the GPU runtime
(CuPy), the optional pip extras (``render``, ``obs``, ``dealias`` and
whatever else the installed distribution declares), the compiled Rust
artifacts (never shipped in the wheel; ``gpuwm fetch-bridges`` stages a
release's prebuilt bundle where one exists for the platform), the
packaged physics tables, and the data roots
(``WPS_GEOG``/``GPUWM_CASE_DATA_ROOT``).  Doctor checks each one for
real, not by presence: it imports every declared package in short-lived
subprocesses, probe-executes every bridge and front-door executable,
loads the CPU preprocessing library through ctypes and reads its ABI
version, sha256-validates the packaged Thompson tables with the same
routine the model uses at launch, parses the Noah/landuse tables with
the model's own parsers, re-hashes every artifact a sealed manifest
declares, resolves each data route's decoder and byte transport through
the resolvers the route itself calls, and requires each WPS_GEOG
dataset's ``index`` file.  No cargo builds, no network; the device work
is three deliberate short-lived subprocess probes (the CuPy-wheel/box
cuBLAS pairing, a cold-cache NVRTC compile, and the radar-DA
eigensolver), isolated so a wedged runtime cannot poison this process.
Every gap prints a remedy whose every line is either a command that
runs as printed in this platform's own shell or a ``#`` comment --
never prose fused onto a command -- instead of letting the user meet a
raw traceback three commands later.

**The extras are read from the INSTALLED distribution's metadata**, not
from a checkout's ``pyproject.toml``.  What a box can do is decided by
what pip resolved onto it, and a checkout sitting beside a wheel is
exactly the configuration where a transcription and the truth part
company.

Statuses distinguish what was proven: ``verified`` means the deep check
ran and passed; ``present`` is for the few items where nothing deeper
than existence can honestly be checked, plus the one deep check that
can half-pass and says which half (the radar-DA eigensolver);
``untested`` is for a question this module deliberately does not answer
-- it never runs cargo, so it cannot say whether a build would succeed
-- and its detail always opens with "not tested"; ``missing`` is a gap
with a remedy; ``info`` is context.  ``verified`` is never printed over
a question nobody asked: two binaries once carried ``ok`` on a box
where the command that needed them could not find them, and that is
what ``untested`` exists to stop.

The report is layered (:mod:`gpuwm.explain`).  By default every finding
is one line -- status, subject, and THE command that closes it -- and
adjacent findings that share a remedy fold into one, because a fresh pip
install gaps every Rust artifact at once and six identical remedies read
as six problems.  ``--explain`` prints what this module always printed:
the evidence behind each check and the whole pasteable remedy block,
verbatim.  Nothing was shortened; the long form moved one flag away.

Exit status: 1 when any finding is ``broken`` or ``unreachable``, 0
otherwise.  A ``degraded`` or ``opt-in`` finding still prints MISSING
and still prints its command; it does not fail the process.  The rule
that assigns those four words is stated once, in :func:`blocking_gaps`.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import importlib.metadata
from importlib.util import find_spec

from gpuwm.local_gpu import no_local_gpu
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from gpuwm import bridges
from gpuwm import rustwx
from gpuwm import rustwx_fetch
from gpuwm import science_core
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
#: NOT "wrf-rust + matplotlib".  matplotlib is a BASE dependency, on
#: every install already; the extra's second package is pyshp, the
#: shapefile reader every basemap needs, and naming matplotlib here left
#: a reader no way to learn that.  Measured wrong by the 2026-08-14
#: reachability audit against the shipped METADATA.
RENDER_EXTRA_HINT = ("pip install 'gpuwm[render]'\n"
                     "  # installs wrf-rust (the derived-quantity core\n"
                     "  # gpuwm enprod and --engine matplotlib import) and\n"
                     "  # pyshp (the shapefile reader the DA nowcast's\n"
                     "  # basemaps read).  matplotlib is NOT in this extra:\n"
                     "  # it is a base dependency, installed already.")
GEOG_STACK_HINT = (
    "pip install --upgrade 'gpuwm[geog]'\n"
    "  # rasterio and pyproj are the PARITY FALLBACK for the\n"
    "  # high-resolution warp, not what a default run reads: the\n"
    "  # default engine is the Rust static-fields library (the\n"
    "  # `static builder` row).  Install these only to run\n"
    "  # GPUWM_STATIC_PYTHON=1.  To add just the two libraries:\n"
    "  #   pip install --upgrade rasterio pyproj")
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
#:
#: The last two entries are worded the way they are because the obvious
#: wording was measured false.  They used to read "20CRv3/mapped GRIB2
#: routes", which a reader takes to mean the staged binary this report
#: probes is what those routes use.  It is not: the mapped route and
#: ``gpuwm adapt`` build their own copies with cargo and never consult
#: the resolver ladder, and the 20CRv3 direct route makes the paths
#: ``required=True``.  :func:`_mapped_grib2_route_check` reports that
#: separately; these lines now claim only what they prove, which is that
#: the staged file exists and runs.
_BRIDGE_CONSUMERS = {
    "grib1_bridge": "ERA5 route (gpuwm check/run, rw-wps --source era5)",
    "gfs_grib2_bridge": "GFS front door (rw-wps --source gfs, and every "
                        "stage of the gpuwm go chain)",
    "hrrr_grib2_bridge": "HRRR front door (rw-wps --source hrrr)",
    "grib2_inventory": "the 20CRv3, mapped and packaged-profile GRIB2 "
                       "prep routes (gpuwm prep/rw-wps), which resolve it "
                       "from here (--grib2-inventory overrides) -- see the "
                       "mapped/20CRv3 route line",
    "grib2_dump": "the 20CRv3, mapped and packaged-profile GRIB2 prep "
                  "routes (gpuwm prep/rw-wps), which resolve it from here "
                  "(--grib2-dump overrides) -- see the mapped/20CRv3 "
                  "route line",
}

#: The doors a box without CuPy cannot open, and the half it keeps.
#: Declared once because two lines report it -- the deep ``cupy (GPU
#: runtime)`` check and the ``[gpu-cu12]``/``[gpu-cu13]`` extras lines --
#: and a reader who found them disagreeing would be right to distrust
#: both.  Every entry was traced to a real refusal or a real import.
_GPU_DOORS = ("gpuwm run", "gpuwm go", "gpuwm check", "gpuwm domain",
              "gpuwm resume", "gpuwm verify", "gpuwm stream",
              "gpuwm multi-run", "gpuwm downscale", "gpuwm ingest",
              "gpuwm-prepared-forecast", "gpuwm-prepared-tree-forecast",
              "the DA nowcast")
_GPU_STILL_WORKS = ("the whole preprocessing half -- gpuwm fetch, "
                    "import-namelist, adapt, render, report and rw-wps")

_PROBE_TIMEOUT_S = 30


#: The four severities, and the two of them that move the exit code.
#: ``blocking_gaps`` states the rule; these are its vocabulary.
#:
#: They exist because a two-valued model could not tell a reader the
#: difference between "the ~16 GB terrain download you deliberately
#: skipped" and "this box cannot run a forecast", and doctor printed
#: the same ``0 of them blocking (exit 0)`` over both.
SEVERITY_BROKEN = "broken"
SEVERITY_UNREACHABLE = "unreachable"
SEVERITY_DEGRADED = "degraded"
SEVERITY_OPT_IN = "opt-in"

#: There is deliberately NO ``BLOCKING_SEVERITIES`` constant.  ``broken``
#: always blocks; ``unreachable`` blocks for a README/FIRST-LIGHT door and
#: not for a CLI-reference-only one, so a constant equating severity with
#: the exit code would be a false claim in the one module that exists to
#: stop those.  ``Check.blocking`` is the authority and
#: :func:`blocking_gaps` states the rule.

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

    ``status`` has five values and one of them is newer than the rest.
    ``untested`` exists because a 2026-08-14 audit caught this report
    printing ``ok`` for two binaries on a box where the command that
    needs them could not find them: doctor had proven the FILES were
    runnable and said nothing about the RESOLUTION, and ``ok`` is not a
    word for a question nobody asked.  Where a check cannot exercise the
    thing it reports on, it says ``untested`` and its detail opens with
    "not tested".  Never ``verified``, and never a quiet omission.
    """

    name: str
    status: str  # "verified" | "present" | "untested" | "missing" | "info"
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
    #: Does this gap justify a nonzero exit?  The rule is stated once,
    #: in :func:`blocking_gaps`, and ``severity`` below is the word for
    #: WHY.  The report text is identical either way -- MISSING stays
    #: MISSING and the remedy still prints; only the exit code and the
    #: summary line read these two.
    blocking: bool = True
    #: Which of the four severities this finding carries:
    #: ``"broken"`` (something present and wrong), ``"unreachable"``
    #: (a documented command cannot run at all), ``"degraded"`` (an
    #: optional mode is gone, a documented default still works), or
    #: ``"opt-in"`` (an explicit choice the user has not made).  The
    #: first two block; the last two do not.  ``None`` on a finding
    #: that is not a gap.
    #:
    #: Declared as a field rather than inferred from ``blocking``
    #: because "why does this fail my install" and "does this fail my
    #: install" are different questions, and a boolean answering both
    #: is how ``0 of them blocking`` came to sit over a box that could
    #: not run a forecast.
    #:
    #: A gap that does not declare one gets the word its ``blocking``
    #: flag has always meant (see ``__post_init__``), so no finding is
    #: ever unclassified and no call site has to repeat itself.
    severity: str | None = None

    def __post_init__(self) -> None:
        # The severities are new; ``blocking`` is not, and every legacy
        # call site already carried the distinction in it.  Its
        # docstring through 2.3.3 read: True for "anything broken or
        # integrity-suspect", False for "an absent optional piece with
        # a documented fallback or a documented opt-in".  That IS the
        # broken/opt-in pair, so the default is a translation rather
        # than a guess -- and the two words the boolean cannot express,
        # `unreachable` and `degraded`, are passed explicitly by the
        # checks that mean them.
        if self.status == "missing" and self.severity is None:
            object.__setattr__(
                self, "severity",
                SEVERITY_BROKEN if self.blocking else SEVERITY_OPT_IN)


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

    if no_local_gpu():
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
        if no_local_gpu():
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
        if (_names_missing_cuda_headers(cublas)
                and (not wheel_majors or box_major in wheel_majors
                     or not box_major)):
            # THE CIRCULAR REMEDY, closed.  CuPy 14 raises its
            # headers error from the first device call of any kind, so
            # the right wheel on the right box reaches HERE -- and this
            # branch used to answer "the installed CuPy cannot load
            # cuBLAS on it: uninstall cupy-cuda13x, install
            # gpuwm[gpu-cu13]", which removes the correct wheel and
            # installs the same correct wheel, forever.  The wheel is
            # fine; what is missing is a CUDA toolkit.  The
            # majors-disagree case is deliberately NOT folded in: there
            # the wheel really is wrong, and its own remedy replaces it
            # with one whose gpuwm extra carries the toolkit too.
            detail += ("  The wheel matches this box; what is absent is the "
                       "CUDA toolkit HEADER tree, which no cupy wheel ships "
                       "and no wheel reinstall supplies")
            remedy, action = _cuda_headers_remedy(box_major or None)
            return Check("cupy (GPU runtime)", "missing", detail, remedy,
                         action=action, brief=_short(str(cublas)),
                         severity=SEVERITY_BROKEN)
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
                     action=action, brief=_short(str(cublas)),
                     severity=SEVERITY_BROKEN)
    # No usable CuPy.  The extra to name is a property of the BOX, so it
    # is read from the driver here rather than defaulted: this is the
    # branch a fresh CUDA-13 machine lands on, and the branch that used
    # to hand it the cu12 wheel.
    box_major = _driver_cuda_major()
    remedy, action = _gpu_extra_hint(box_major)
    served = "" if box_major is None else f"; this box serves CUDA {box_major}"
    if evidence == "not installed":
        # BLOCKING, and it was not until 2026-08-14.  This line used to
        # be an "absent optional" beside `0 of them blocking (exit 0)`,
        # on a box where `gpuwm run` -- the product's headline command,
        # and the whole point of installing it -- dies in a raw
        # ModuleNotFoundError after fetching gigabytes and preparing
        # them.  A green light over that is worse than no light: an
        # installer script that trusts the exit code ships the box, and
        # the user meets the traceback instead of this remedy.
        return Check(
            "cupy (GPU runtime)", "missing",
            "not installed -- without it this box cannot run "
            + ", ".join(_GPU_DOORS)
            + f".  It can still run {_GPU_STILL_WORKS}{served}", remedy,
            action=action, brief="not installed",
            severity=SEVERITY_UNREACHABLE)
    # CuPy IS installed and would not import.  The extra cannot be the
    # answer to every version of that -- see _cupy_import_failure_remedy,
    # which reads the import's own message and routes accordingly.
    remedy, action = _cupy_import_failure_remedy(evidence, box_major)
    return Check("cupy (GPU runtime)", "missing", evidence + served, remedy,
                 action=action, brief=_short(evidence),
                 severity=SEVERITY_BROKEN)


#: Compiles two kernels and reports which one built.  The pair is the whole
#: point, because the two failures it separates need OPPOSITE remedies:
#:
#: 1. a SELF-CONTAINED kernel with no ``#include`` at all -- how every gpuwm
#:    kernel is written.  It exercises NVRTC itself, which ships INSIDE the
#:    CuPy wheel.  If this fails, the wheel is the problem.
#: 2. a CuPy reduction, which compiles through CuPy's own cub/jitify
#:    preamble.  That preamble ``#include``s the CUDA runtime headers, and
#:    those come from a TOOLKIT, found through ``CUDA_PATH``.  If this fails
#:    while (1) passes, the headers are the problem and no wheel supplies them.
#:
#: THE COLD CACHE IS LOAD-BEARING.  ``CUPY_CACHE_DIR`` is redirected to an
#: empty directory because a warm kernel cache is exactly what hides this
#: fault: the box that produced this check compiled its reductions once under
#: CUDA 12, moved to a CUDA-13 toolkit, and kept serving the cached cubins for
#: weeks while every UNCACHED reduction failed.  A probe that reuses the
#: cache reports green on a box where the next new kernel dies.
_NVRTC_HEADER_PROBE = """
import json, sys
out = {}
try:
    import cupy
except Exception as error:
    sys.stdout.write(json.dumps({"cupy": repr(error)}))
    raise SystemExit(0)
try:
    if cupy.cuda.runtime.getDeviceCount() < 1:
        sys.stdout.write(json.dumps({"devices": 0}))
        raise SystemExit(0)
except Exception as error:
    sys.stdout.write(json.dumps({"devices": 0, "device_error": repr(error)}))
    raise SystemExit(0)
try:
    module = cupy.RawModule(code=(
        'extern "C" __global__ '
        'void gpuwm_probe(float* out) { out[0] = 1.0f; }'))
    module.get_function("gpuwm_probe")
    out["self_contained"] = "ok"
except Exception as error:
    out["self_contained"] = f"{type(error).__name__}: {error}"
try:
    total = int(cupy.arange(64, dtype=cupy.int64).sum())
    out["toolkit_headers"] = "ok" if total == 2016 else f"wrong sum {total}"
except Exception as error:
    out["toolkit_headers"] = f"{type(error).__name__}: {error}"
import os
out["cuda_path"] = os.environ.get("CUDA_PATH") or ""
sys.stdout.write(json.dumps(out))
"""


def _nvrtc_header_probe() -> dict:
    """Run :data:`_NVRTC_HEADER_PROBE` cold, out of process; ``{}`` if not."""

    if find_spec("cupy") is None:
        return {"cupy": "not installed"}
    with tempfile.TemporaryDirectory(prefix="gpuwm-nvrtc-probe-") as cache:
        environment = dict(os.environ, CUPY_CACHE_DIR=cache)
        try:
            probe = subprocess.run(
                [sys.executable, "-c", _NVRTC_HEADER_PROBE],
                capture_output=True, text=True, errors="replace",
                env=environment, timeout=_PROBE_TIMEOUT_S * 6)
        except subprocess.TimeoutExpired:
            # NOT a verdict.  A cold NVRTC compile on a slow or busy box
            # can outrun this budget while being perfectly healthy, and
            # reporting that as a missing header tree would send a reader
            # to install a toolkit they already have.
            return {"slow": f"no answer within {_PROBE_TIMEOUT_S * 6} s"}
        except OSError as error:
            return {"probe": f"did not run: {error}"}
    try:
        return json.loads(probe.stdout or "{}")
    except ValueError:
        tail = [line for line in (probe.stderr or "").strip().splitlines()
                if line.strip()]
        return {"probe": tail[-1] if tail else f"exit {probe.returncode}"}


#: The CuPy wheel extra that installs a matching CUDA toolkit, as wheels.
#:
#: CuPy 14.0 added it; 13.x has none, which is why the gpu extras floor at
#: 14.0 (pip only WARNS on an extra the resolved version does not provide,
#: so a lower floor would let the toolkit go silently missing).
_CUPY_TOOLKIT_EXTRA = "ctk"


def _cupy_ctk_install(major: int) -> str:
    """The one command that gives a driver-only box a working CuPy."""

    return f"pip install 'cupy-cuda{major}x[{_CUPY_TOOLKIT_EXTRA}]'"


def _cuda_headers_remedy(box_major: int | None) -> tuple[str, str]:
    """``(remedy, action)`` for a box whose CuPy cannot find CUDA headers.

    NOT A WHEEL PROBLEM, SO NOT A WHEEL REMEDY.  This is the correction
    that matters most on a fresh box: the gap used to be reported -- when
    it was reported at all -- with ``pip install 'gpuwm[gpu-cu13]'``,
    which reinstalls the CuPy wheel that is already present and supplies
    no headers, because no CuPy wheel has ever carried a header tree.  A
    buyer following it watches pip succeed and the fault survive.

    WHAT LEADS NOW, and why it changed.  MEASURED on weather-node-1
    (Ubuntu, CUDA 13 driver, no ``nvcc``, no ``/usr/local/cuda``, no
    conda), 2026-08-17: a bare ``pip install cupy-cuda13x`` gives a 274
    MiB environment in which cuBLAS, a self-contained ``RawModule`` and a
    ``cupy.arange().sum()`` reduction ALL fail with "Failed to find CUDA
    headers", and ``pip install 'cupy-cuda13x[ctk]'`` gives an 1867 MiB
    environment in which all three pass.  CuPy publishes the toolkit as
    an extra of its own wheel, pinned to that wheel's CUDA major, so the
    fix is one line, needs no conda, no administrator and no
    ``CUDA_PATH``.  The old lead -- a conda toolkit -- assumed an estate
    the failing box did not have.

    CuPy's own message hardcodes ``cupy-cuda12x[ctk]`` at every major.
    Pasted on a CUDA-13 box that installs the wheel that cannot load
    cuBLAS there, which is the trap this whole module was rewritten
    around, so the major printed here is the one read off the driver.

    The external-toolkit routes survive as commented alternatives, for an
    estate that wants a real system toolkit or holds a pinned CuPy older
    than 14.0 (which has no ``[ctk]`` extra to ask for).  Commented, not
    live: a remedy is pasted whole, and a paste that installs a toolkit
    two different ways is a worse install than the one it repaired.
    """

    major = box_major if box_major in _GPU_EXTRA_BY_MAJOR else None
    pin = f"={major}" if major else ""
    if bridges.WINDOWS_SHELL:
        point_at_conda = "#   $env:CUDA_PATH = $env:CONDA_PREFIX"
    else:
        point_at_conda = '#   export CUDA_PATH="$CONDA_PREFIX"'
    lines = [
        "# the CuPy wheel ships NVRTC -- the COMPILER -- and no headers.",
        "# CuPy's own kernels include the CUDA runtime headers, so on a",
        "# box with no CUDA toolkit the first device call of any kind --",
        "# a reduction, a sort, the first cuBLAS load behind a matmul --",
        "# dies with 'Failed to find CUDA headers'.  Reinstalling the",
        "# cupy wheel or the gpuwm GPU extra CANNOT fix that: it",
        "# reinstalls the compiler, which is already here.",
        "#",
        "# The header tree has to MATCH that compiler's major, and CuPy",
        "# publishes exactly that as an extra of its own wheel:",
    ]
    if major is None:
        lines += [
            "# this box's CUDA major could not be read, and the wheel is",
            "# chosen by it, so read it first:",
            "nvidia-smi",
            "# ...then run the line that matches what it prints:",
            f"#   {_cupy_ctk_install(12)}",
            f"#   {_cupy_ctk_install(13)}",
        ]
        action = ("nvidia-smi  # then pip install "
                  "'cupy-cuda12x[ctk]' or 'cupy-cuda13x[ctk]' to match")
    else:
        lines += [
            f"# this box's driver serves CUDA {box_major}, so it is this one",
            _cupy_ctk_install(major),
        ]
        action = _cupy_ctk_install(major)
    lines += [
        "#",
        "# That pulls the CUDA runtime and its headers, NVRTC, cuBLAS,",
        "# cuFFT, cuRAND, cuSOLVER and cuSPARSE as wheels: about 1.6 GiB",
        "# installed, measured on a driver-only box.  gpuwm's own GPU",
        "# extras ask for the same set from 2.5.0 on, so a FRESH install",
        "# needs nothing beyond the documented pip line; this repairs an",
        "# environment that already exists, or one holding a CuPy older",
        "# than 14.0, which has no [ctk] extra to ask for.",
        "#",
        "# ALTERNATIVES, for an estate that installs a real toolkit",
        "# instead.  conda puts one into the active environment on both",
        "# platforms without administrator rights:",
        f"#   conda install -c nvidia cuda-toolkit{pin}",
        f"#   conda install -c conda-forge cuda-toolkit{pin}",
        "# ...and if CUDA_PATH did not get set on activation, or the",
        "# toolkit came from NVIDIA's own installer, point it at the",
        "# toolkit root yourself:",
        point_at_conda,
        "# ...or take the NVIDIA wheels on their own, which carry the",
        "# same header tree:",
        "#   " + _cuda_wheel_install(_CUDA_TOOLKIT_PACKAGES, major),
        "#   python -c \"import nvidia.cuda_runtime as r,pathlib;"
        "print(pathlib.Path(r.__file__).parent)\"",
    ]
    return "\n".join(lines), action


#: Markers that name the TOOLKIT HEADERS as the missing piece, wherever
#: the message surfaced -- an import, the cuBLAS pairing probe, or a
#: kernel compile.  CuPy says "Failed to find CUDA headers" in those
#: words, and names the ``CUDA_PATH``/``nvcc``/header file it consulted
#: looking for them.
#:
#: This is a FAILURE CLASS, not a probe: CuPy 14 raises the same error
#: from the first device call of any kind, so the branch that owns the
#: cuBLAS probe meets it too and used to call it a wrong wheel.
_HEADERS_FAILURE_MARKERS = ("cuda headers", "cuda_path", "cuda_home",
                            "nvcc", "cuda toolkit", "cuda_fp8.hpp",
                            "cuda_runtime.h")


def _names_missing_cuda_headers(text: object) -> bool:
    """Does this failure message name the toolkit-headers class?"""

    lowered = str(text or "").lower()
    return any(marker in lowered for marker in _HEADERS_FAILURE_MARKERS)


#: The markers that name the WHEEL: a shared library the wheel
#: hard-codes and the loader could not produce.  (Their opposite number,
#: the toolkit-headers class, is :data:`_HEADERS_FAILURE_MARKERS`, which
#: lives beside the remedy it selects because three checks read it now.)
_WHEEL_IMPORT_MARKERS = ("no module named", "cannot open shared object",
                         "libcublas", "libnvrtc", "libcudart",
                         "dll load failed")


def _cupy_import_failure_remedy(evidence: str,
                                box_major: int | None) -> tuple[str, str]:
    """``(remedy, action)`` for a CuPy that is INSTALLED and will not import.

    THE 2.2.1 FIELD CASE, on Ubuntu.  Doctor printed ``MISSING cupy --
    ... Failed to find CUDA headers ...`` and offered ``pip install
    'gpuwm[gpu-cu13]'``.  The wheel was already installed; what was absent
    was the CUDA TOOLKIT, and no gpuwm extra has ever carried one, so the
    reader watched pip report success and the fault survive.  The line that
    actually fixed that box was a toolkit install.

    The two faults are separable from the import's own message, cheaply
    and without a second probe: CuPy names the header tree or the
    ``CUDA_PATH``/``nvcc`` it consulted when the toolkit is what is
    missing, and names a shared library it could not load when the wheel
    is.  Where neither signature appears the symptom really is ambiguous,
    so BOTH remedies print, each LABELLED by the symptom it answers --
    never one of them silently chosen, which is how this bug read to the
    user who hit it.  The single action stays the toolkit line, because
    that is the branch a fresh box lands on and the branch the wheel
    remedy cannot fix.
    """

    lowered = evidence.lower()
    if _names_missing_cuda_headers(evidence):
        return _cuda_headers_remedy(box_major)
    if any(marker in lowered for marker in _WHEEL_IMPORT_MARKERS):
        return _gpu_extra_hint(box_major)
    headers_remedy, headers_action = _cuda_headers_remedy(box_major)
    wheel_remedy, _ = _gpu_extra_hint(box_major)
    return "\n".join([
        "# This import failed for one of two reasons and its message names",
        "# neither, so both remedies are printed.  Match yours by symptom.",
        "#",
        "# SYMPTOM: the message names a header, CUDA_PATH or nvcc.  The CUDA",
        "# TOOLKIT is missing, and no wheel or gpuwm extra supplies one:",
        headers_remedy,
        "#",
        "# SYMPTOM: the message names a shared library or a missing module.",
        "# The CuPy wheel itself is absent or wrong for this box:",
        wheel_remedy,
    ]), headers_action


def _cuda_headers_check() -> Check:
    """Can CuPy COMPILE on this box, and if not, is it the wheel or headers?

    A check rather than a footnote because the gap is silent by
    construction: cupy imports, cuBLAS loads, a matmul returns the right
    answer, and doctor called that verified -- while the first uncached
    reduction of a real run died on a missing header.  Everything cheaper
    than a cold compile passes on a box that cannot compile.

    Never blocking.  Most of gpuwm's own kernels are self-contained
    source strings compiled without jitify, and fetch, import-namelist
    and render do not touch CUDA at all.  "Never read the toolkit tree"
    is what this docstring used to say, and it is too strong: two
    microphysics kernels carry a ``#include <cmath>``.  What the probe
    below separates is still the right pair -- NVRTC itself, which ships
    inside the wheel, against the toolkit include tree CuPy's own cub
    preamble needs -- and the finding is reported either way.
    """

    name = "CUDA kernel headers"
    if no_local_gpu():
        return Check(
            name, "info",
            "not judged -- compiling a kernel is device contact and "
            "GPUWM_NO_LOCAL_GPU is set",
            brief="device not touched", blocking=False)
    result = _nvrtc_header_probe()
    if "cupy" in result:
        return Check(
            name, "info",
            f"not judged -- kernels compile through cupy, which is "
            f"unavailable here ({result['cupy']})",
            brief="needs cupy", blocking=False)
    if "devices" in result:
        reason = result.get("device_error") or "no CUDA device visible"
        return Check(
            name, "info",
            f"not judged -- compiling needs a device: {_short(str(reason), 96)}",
            brief="no device", blocking=False)
    if "slow" in result:
        return Check(
            name, "info",
            f"not judged -- the cold compile did not finish "
            f"({result['slow']}); a first compile on a busy box is slow",
            brief="compile timed out", blocking=False)
    if "probe" in result:
        return Check(
            name, "missing",
            f"the compile probe could not be run: {result['probe']}",
            *_cuda_headers_remedy(_driver_cuda_major()),
            brief=_short(result["probe"]), blocking=False)

    self_contained = result.get("self_contained", "did not report")
    headers = result.get("toolkit_headers", "did not report")
    cuda_path = result.get("cuda_path") or "unset"
    if self_contained == "ok" and headers == "ok":
        return Check(
            name, "verified",
            f"a self-contained kernel and a cupy reduction both compiled "
            f"from a COLD cache (CUDA_PATH {cuda_path})",
            brief="kernels compile", blocking=False)
    if self_contained != "ok" and _names_missing_cuda_headers(self_contained):
        # CuPy 14 raises the headers error BEFORE it compiles anything,
        # so a kernel with no ``#include`` at all fails too and the
        # two-kernel split cannot separate the pair on its own.  Falling
        # through to the wheel branch here is how a driver-only box was
        # told "NVRTC ships inside the cupy wheel, so this is the wheel"
        # and sent to reinstall the wheel that is present and correct.
        # The message names the class; the class picks the remedy.
        box_major = _driver_cuda_major()
        remedy, action = _cuda_headers_remedy(box_major)
        return Check(
            name, "missing",
            f"no kernel compiles on this box, including one with no "
            f"#include at all: {_short(str(self_contained), 120)}.  That "
            f"names the toolkit HEADER tree, which no cupy wheel ships and "
            f"no wheel reinstall supplies (CUDA_PATH {cuda_path})",
            remedy, action=action, brief="toolkit headers missing",
            blocking=False)
    if self_contained == "ok":
        # THE DISTINCTION THIS CHECK EXISTS FOR.  NVRTC works, so the
        # wheel is fine and reinstalling it is wasted advice; what is
        # missing is the header tree NVRTC was asked to read.
        box_major = _driver_cuda_major()
        remedy, action = _cuda_headers_remedy(box_major)
        return Check(
            name, "missing",
            f"NVRTC works -- a self-contained kernel compiled -- but a cupy "
            f"reduction did not: {_short(str(headers), 120)}.  That is the "
            f"toolkit HEADER tree, which no cupy wheel ships and no wheel "
            f"reinstall supplies (CUDA_PATH {cuda_path})",
            remedy, action=action, brief="toolkit headers missing",
            blocking=False)
    # NVRTC itself did not build the simplest kernel there is, so this is
    # the wheel, and the wheel remedy is the honest one.
    box_major = _driver_cuda_major()
    remedy, action = _gpu_extra_hint(box_major)
    return Check(
        name, "missing",
        f"cupy could not compile even a self-contained kernel: "
        f"{_short(str(self_contained), 120)}.  NVRTC ships inside the cupy "
        f"wheel, so this is the wheel rather than the toolkit headers",
        remedy, action=action, brief="nvrtc unusable", blocking=False)


#: The CUDA libraries cuSOLVER needs present TOGETHER.  Every one is a
#: dependency OF cusolver, not a nicety: cusolver links cublas and cusparse,
#: and on Windows a wheel's DLLs are only visible to a compiled extension
#: through ``os.add_dll_directory``, never through PATH.
_CUSOLVER_PACKAGES = ("nvidia-cusolver", "nvidia-cublas", "nvidia-cusparse")

#: The toolkit pieces a header gap needs from pip when the estate has no
#: conda: the CUDA runtime, whose wheel carries the header tree CuPy's own
#: reduction kernels include, and NVRTC beside it so the compiler and the
#: headers it reads come from the same major.
_CUDA_TOOLKIT_PACKAGES = ("nvidia-cuda-runtime", "nvidia-cuda-nvrtc")


def _cuda_wheel_install(packages: tuple[str, ...], major: int | None) -> str:
    """The pip line that installs these CUDA libraries for ``major``.

    NVIDIA HAS DEPRECATED THE ``-cuXX`` SUFFIXED NAMES -- ``-cu12`` as well
    as ``-cu13`` -- and the suffixed distributions survive as tombstones:
    ``pip install nvidia-cusolver-cu13`` resolves, downloads, reports
    success, and installs a package whose entire content is a notice saying
    to use ``nvidia-cusolver`` instead.  A remedy that prints one leaves the
    box exactly as broken as before while the transcript says it was fixed
    -- the NVRTC shadow trap with extra steps.  So every name printed here
    is the unsuffixed one, at EVERY major; a remedy that spelled the suffix
    for 12 and dropped it for 13 was one NVIDIA deprecation away from being
    wrong again, and that deprecation has now landed.

    The major does not disappear with the suffix, it MOVES: out of the
    package name and into a version pin, ``nvidia-cusolver=="12.*"``, so
    the line still matches the box it was generated for.  The pin is
    derived from the detected driver and never hardcoded; with no major in
    hand the line carries no pin and the caller prints the lookup that
    finds one.  Double quotes, because ``12.*`` is a glob to sh and a
    string to PowerShell only when it is quoted, and the same spelling has
    to survive both.  ``--no-deps`` is what keeps the unsuffixed packages
    from dragging those same tombstones back in as dependencies.
    """

    if major is None:
        return "pip install --no-deps " + " ".join(packages)
    pins = " ".join(f'"{name}=={major}.*"' for name in packages)
    return "pip install --no-deps " + pins


def _cusolver_hint(box_major: int | None) -> tuple[str, str]:
    """``(remedy, action)`` for a device whose cuSOLVER is missing.

    The install line is a property of the BOX's CUDA major, read from the
    driver, for the same reason the CuPy extra is: until this release the
    line was a frozen ``-cu12`` spelling, so a CUDA-13 box asking doctor
    how to get its eigensolver back was handed three tombstone packages.
    The names are unsuffixed now at every major (see
    :func:`_cuda_wheel_install`), so the major rides in the version pin;
    with no major in hand both pins are printed and neither is presented
    as the default, because a silent default is exactly how a CUDA-13 box
    ends up installing CUDA 12 libraries.
    """

    preamble = (
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
        "  # only.  If it prints nothing, install it:")
    tombstone = (
        "  # (the -cu12 and -cu13 spellings of these are deprecation\n"
        "  # tombstones: they install cleanly and supply nothing)")
    if box_major is None:
        return ("\n".join([
            preamble,
            "  # NVIDIA has deprecated the -cuXX suffixes, so the package",
            "  # names no longer carry the major -- the version pin does,",
            "  # and it has to match the major this box serves.  Read it:",
            "  nvidia-smi",
            "  # then, for a CUDA 12.x box:",
            f"  {_cuda_wheel_install(_CUSOLVER_PACKAGES, 12)}",
            "  # or, for a box whose CUDA is 13-only:",
            f"  {_cuda_wheel_install(_CUSOLVER_PACKAGES, 13)}",
            tombstone,
        ]), "nvidia-smi  # then install the CUDA libraries for that major")
    install = _cuda_wheel_install(_CUSOLVER_PACKAGES, box_major)
    return ("\n".join([
        preamble,
        f"  # this box's driver serves CUDA {box_major}, so:",
        tombstone,
        f"  {install}",
    ]), install)


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
    # The install spelling is the BOX's property, not a constant: see
    # _cusolver_hint.  Read once, so all four branches agree.
    cusolver_remedy, cusolver_action = _cusolver_hint(_driver_cuda_major())
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
            cusolver_remedy, action=cusolver_action,
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
            None if library == "ok" else cusolver_remedy,
            action=None if library == "ok" else cusolver_action,
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
        cusolver_remedy, action=cusolver_action,
        brief="no device eigensolver", blocking=False)


# ---------------------------------------------------------------------------
# The pip extras: enumerated from the INSTALLED distribution, judged by import
# ---------------------------------------------------------------------------
#
# Doctor probed cupy, wrf and matplotlib and NOTHING ELSE on the Python
# side.  A 2026-08-14 reachability audit ran `gpuwm doctor --explain` on
# a bare install and measured the consequence: 40,351 characters of
# report in which the strings `scipy`, `pyshp`, `shapefile`, `rasterio`
# and `pyproj` appeared ZERO times.  Four extras -- `obs`, `dealias`,
# `geog`, and half of `render` -- were invisible to the one command
# whose whole job is telling a user what their install cannot do.  The
# same report called `[render]` "wrf-rust + matplotlib" while the extra
# is wrf-rust + pyshp and matplotlib is a BASE dependency, so the one
# line a reader could have followed to the shapefile reader said the
# wrong package.
#
# The enumeration below is therefore NOT a transcription of
# pyproject.toml.  It is read from the installed distribution's own
# metadata, because that is what pip resolved onto this box, and a
# checkout sitting beside a wheel is exactly the configuration where a
# transcription and the truth part company.  What this module still
# declares by hand is the thing metadata cannot carry: which documented
# command each extra is the difference between running and not.

#: The distribution whose metadata declares the extras.  Named once.
_DISTRIBUTION = "gpuwm"

#: The leading name of a ``Requires-Dist`` line, and the ``extra ==``
#: marker that assigns it to one.
_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_EXTRA_MARKER = re.compile(r"""extra\s*==\s*['"]([^'"]+)['"]""")


def _canonical(name: str) -> str:
    """PEP 503 normalisation: one spelling two tables can agree on."""

    return re.sub(r"[-_.]+", "-", name).strip().lower()


#: pip distribution -> the name ``import`` actually uses, for every
#: package this project declares.  The two differ often enough that
#: guessing is how a check reports green on an absent package:
#: ``pyshp`` imports as ``shapefile``, ``wrf-rust`` as ``wrf``,
#: ``cupy-cuda12x`` as ``cupy``.  A distribution missing from this table
#: is still CHECKED -- by its installed metadata -- but the report says
#: that is all it was, rather than claiming an import nobody attempted.
#: ``tests/test_doctor_extras.py`` fails if this project declares a
#: requirement this table does not name.
_IMPORT_NAME = {
    "cupy-cuda12x": "cupy",
    "cupy-cuda13x": "cupy",
    "wrf-rust": "wrf",
    "pyshp": "shapefile",
    "scipy": "scipy",
    "rasterio": "rasterio",
    "pyproj": "pyproj",
    "numpy": "numpy",
    "netcdf4": "netCDF4",
    "matplotlib": "matplotlib",
    "jsonschema": "jsonschema",
    "pytest": "pytest",
    "psutil": "psutil",
    "huggingface-hub": "huggingface_hub",
    "h5py": "h5py",
    # The companion distribution: pure reference tables (RRTMGP,
    # Thompson), pinned `==` and installed by a bare `pip install
    # gpuwm`.  The probe imports it like any other, which is exactly
    # what a reader needs -- an importable gpuwm_data is what
    # gpuwm.data_assets resolves against.
    "gpuwm-data": "gpuwm_data",
}


#: Distribution -> the check that already judges it, deeply.  Listed
#: here so the extras/base block NAMES the package and points at the
#: verdict instead of publishing a second one.  Two lines disagreeing
#: about one package is worse than one line saying less.
#: The CuPy wheels are deliberately NOT here.  Their extras report
#: presence without a verdict (:func:`_mutually_exclusive_extra_check`),
#: so naming which wheel pip resolved is the whole content of those two
#: lines; filtering them out left the line saying " not installed" about
#: nothing at all.
_DEEP_CHECKED = {
    "rasterio": "geography stack (rasterio + pyproj)",
    "pyproj": "geography stack (rasterio + pyproj)",
}


@dataclass(frozen=True)
class _Requirement:
    """One ``Requires-Dist`` line, split into the parts a check needs."""

    #: PEP 503 name, which is what pip installed it under.
    distribution: str
    #: The requirement without its environment marker -- the half a
    #: reader recognises (``scipy>=1.11``).
    specifier: str
    #: ``gpuwm[render]`` and friends: an extra that pulls another extra
    #: rather than a package.  Reported as the alias it is.
    alias: bool
    #: The environment marker MINUS its ``extra ==`` clause -- the half
    #: that decides whether pip installs this requirement on THIS
    #: interpreter at all.  ``""`` when there is none.
    marker: str = ""


#: The ``extra == "..."`` clause and the boolean glue around it.  Removed
#: rather than evaluated: the extra is how the requirement was filed in
#: the first place, so what is left is the part that answers "would pip
#: install this here".
_EXTRA_CLAUSE = re.compile(
    r"""\s*(?:and\s+|or\s+)?extra\s*==\s*['"][^'"]+['"]\s*""")


def _parse_requirement(text: str) -> _Requirement | None:
    match = _REQUIREMENT_NAME.match(text)
    if match is None:
        return None
    distribution = _canonical(match.group(1))
    head, _, marker = text.partition(";")
    return _Requirement(distribution, head.strip(),
                        distribution == _canonical(_DISTRIBUTION),
                        _EXTRA_CLAUSE.sub(" ", marker).strip(" and or\t"))


def _marker_gap(requirement: "_Requirement") -> str | None:
    """The sentence that names why this build excludes ``requirement``.

    A marker tells a reader that pip skipped something; it does not tell
    them WHY, and "why" is the whole content of a gap report.  This is
    how the project says it out loud.  Anything excluded with no answer
    here still reports the marker verbatim -- an unnamed exclusion is a
    defect to surface, not one to hide.

    Asked, not tabled.  The one gap this project has ever recorded is the
    science core's interpreter ceiling, and that ceiling is an
    availability fact about an index that moves: wrf-rust 0.2.39 closed
    the 3.14 gap that 0.2.38 opened.  A table entry frozen at import
    would have gone on printing the closed one, so the answer is computed
    from :data:`science_core.SCIENCE_CORE_PYTHON_CEILING` at the moment a
    reader asks, and only when this interpreter is genuinely over it.
    """

    if requirement.distribution != _canonical(
            science_core.SCIENCE_CORE_DISTRIBUTION):
        return None
    found = _python_version()
    if science_core.python_supports_science_core(found):
        # Excluded for some reason that is NOT the ceiling; naming the
        # ceiling here would be a confident wrong answer.
        return None
    return science_core.python_gap_sentence(found)


def _python_version() -> tuple[int, ...]:
    """This interpreter's version, as one patchable seam.

    A function so the marker branches can be exercised from a suite that
    is not running on the interpreter under test; patching ``sys``
    itself would reach every other consumer in the process.
    """

    return tuple(sys.version_info[:3])


def _marker_excludes_this_interpreter(requirement: _Requirement) -> bool:
    """Would pip SKIP this requirement here, marker and all?

    ``packaging`` answers it exactly and is present on every install
    (matplotlib, a base dependency, requires it), so it is tried first.
    The fallback covers only ``python_version`` comparisons, which is the
    one marker this project authors, and is documented as partial rather
    than presented as an evaluator: a marker it cannot read reports
    "not excluded", which is the pre-marker behaviour and never invents
    an exclusion that pip did not make.
    """

    marker = requirement.marker
    if not marker:
        return False
    found = _python_version()
    try:
        from packaging.markers import Marker

        # The interpreter comes from the same seam the rest of this
        # module reports, never from ``packaging``'s own view of the
        # process: a check that judges one interpreter and reports
        # another is the class of defect this file exists to catch.
        return not Marker(marker).evaluate({
            "python_version": f"{found[0]}.{found[1]}",
            "python_full_version": ".".join(str(part) for part in found)})
    except ImportError:
        pass
    except Exception:
        return False
    found = found[:2]
    for operator, literal in re.findall(
            r"""python_version\s*(<=|>=|==|!=|<|>)\s*['"]([0-9.]+)['"]""",
            marker):
        want = tuple(int(part) for part in literal.split(".") if part.isdigit())
        here = found[:len(want)]
        satisfied = {
            "<": here < want, "<=": here <= want, ">": here > want,
            ">=": here >= want, "==": here == want, "!=": here != want,
        }[operator]
        if not satisfied:
            return True
    return False


def declared_requirements() -> tuple[tuple[_Requirement, ...],
                                     dict[str, tuple[_Requirement, ...]]] | None:
    """``(base requirements, extra -> requirements)`` for THIS install.

    ``None`` when the interpreter has no installed ``gpuwm``
    distribution to read -- a source tree on ``PYTHONPATH`` and nothing
    else.  That case is reported as unanswered rather than guessed at,
    because the alternative is reading a checkout's ``pyproject.toml``
    and calling it the estate, which is the exact substitution this
    module exists to refuse.
    """

    try:
        declared = importlib.metadata.requires(_DISTRIBUTION) or []
        metadata = importlib.metadata.metadata(_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return None
    base: list[_Requirement] = []
    extras: dict[str, list[_Requirement]] = {
        _canonical(name): []
        for name in (metadata.get_all("Provides-Extra") or [])}
    for text in declared:
        requirement = _parse_requirement(text)
        if requirement is None:
            continue
        marker = _EXTRA_MARKER.search(text)
        if marker is None:
            base.append(requirement)
            continue
        extras.setdefault(_canonical(marker.group(1)), []).append(requirement)
    return tuple(base), {name: tuple(items)
                         for name, items in extras.items()}


@dataclass(frozen=True)
class _ExtraFacts:
    """What one extra is the difference between running and not.

    ``doors`` are the documented commands a box WITHOUT the extra
    cannot run at all.  ``still_works`` is what such a box can still do,
    and it is not decoration: a reader who sees ``[render]`` reported
    MISSING beside a working ``gpuwm render`` deserves to be told why
    both are true (the rust engine needs no Python package; the
    matplotlib engine and ``gpuwm enprod`` do).

    ``blocking`` follows ONE rule, stated in :func:`blocking_gaps`: an
    extra blocks exactly when a command the product's own README or
    FIRST-LIGHT tells a user to run cannot run without it.
    """

    doors: tuple[str, ...]
    still_works: str | None
    blocking: bool
    severity: str
    #: Named when another check owns the verdict, so the extras section
    #: reports the install line without double-counting the gap.
    deferred_to: str | None = None
    #: One of a set where installing every member would be the fault:
    #: the CuPy pair, one wheel per CUDA major.  Such a line never
    #: carries a gap, because a correctly installed box is missing the
    #: other one by definition.
    mutually_exclusive: bool = False
    #: Absent is this extra's NORMAL state, so its absence is context
    #: rather than a gap.  `[dev]` is the case: a user install is
    #: supposed not to have pytest, and printing MISSING at every reader
    #: who never intends to run the suite spends the word MISSING on
    #: nothing.  The line still prints, with its install command.
    expected_absent: bool = False


#: Which extra unlocks what.  Hand-declared because no metadata carries
#: it, and audited against the code in ``tests/test_doctor_extras.py``:
#: every door named here must be a real door, and every extra the
#: installed metadata declares must appear here.
_EXTRA_FACTS: dict[str, _ExtraFacts] = {
    "gpu-cu12": _ExtraFacts(
        doors=_GPU_DOORS, still_works=_GPU_STILL_WORKS,
        blocking=True, severity=SEVERITY_UNREACHABLE,
        deferred_to="cupy (GPU runtime)", mutually_exclusive=True),
    "gpu-cu13": _ExtraFacts(
        doors=_GPU_DOORS, still_works=_GPU_STILL_WORKS,
        blocking=True, severity=SEVERITY_UNREACHABLE,
        deferred_to="cupy (GPU runtime)", mutually_exclusive=True),
    # wrf-rust and pyshp gate DIFFERENT doors, which is why the old
    # single "render extra" line could not tell the truth about either:
    # wrf-rust is what `gpuwm enprod` and `gpuwm render --engine
    # matplotlib` import, and pyshp is the shapefile reader every DA
    # nowcast basemap draws with.  Nothing under `gpuwm/` imports
    # pyshp at all -- the rust renderer reads the shapefiles itself --
    # so a reader who is told "the render extra" and watches `gpuwm
    # render` work anyway is entitled to think the report is wrong.
    "render": _ExtraFacts(
        doors=("gpuwm enprod", "gpuwm render --engine matplotlib",
               "gpuwm go's render stage (it gates on the wrf package)",
               "python -m tools.da_nowcast render and the launcher's "
               "basemap endpoint (pyshp)",
               "python -m tilestream.realcase_render (pyshp)"),
        still_works="gpuwm render's DEFAULT rust engine, which draws from "
                    "the rw_wrfbatch binary and needs no Python package "
                    "from this extra",
        blocking=True, severity=SEVERITY_UNREACHABLE),
    "obs": _ExtraFacts(
        doors=("python -m tools.obs_battery_score",
               "python -m tools.obs_precampaign_controls",
               "python -m tools.obs_battery_registration"),
        still_works="every forecast and preprocessing route; only scoring "
                    "a run against observations needs it",
        blocking=False, severity=SEVERITY_DEGRADED),
    # NOT the default engine.  region-global became the shipped
    # --dealias-engine on 2026-08-12 and is the Rust library this report
    # checks on its own line; scipy labels the gate regions for the OTHER
    # engine only.  Saying otherwise would send a reader to pip for a
    # library their default path never loads.
    "dealias": _ExtraFacts(
        doors=("--dealias-engine vad-region, on python -m "
               "tools.obs_radar_grid_build / tools.obs_radar_grid_from_pack "
               "/ tools.da_nowcast run",),
        still_works="the shipped default --dealias-engine region-global, "
                    "which is the Rust library reported above",
        blocking=False, severity=SEVERITY_DEGRADED),
    # EMPTY from 2.3.3: rasterio and pyproj became runtime dependencies,
    # and the extra is retained so `pip install 'gpuwm[geog]'` -- written
    # down in 2.3.2's own docs and in other people's scripts -- keeps
    # resolving.  It is still reported, because "this extra now installs
    # nothing, and here is where its packages went" is exactly what a
    # reader following an older doc needs to be told.
    "geog": _ExtraFacts(
        doors=("a run whose config sets [static.highres] enabled = true "
               "(gpuwm run / resume / static and both prepared runners)",),
        still_works="the standard WPS_GEOG static-field route, which is "
                    "what every config without that table uses",
        blocking=False, severity=SEVERITY_OPT_IN,
        deferred_to="geography stack (rasterio + pyproj)"),
    "dev": _ExtraFacts(
        doors=("pytest -- this project's own test suite",),
        still_works="everything a user runs; this extra is for developing "
                    "gpuwm, not for using it",
        blocking=False, severity=SEVERITY_OPT_IN, expected_absent=True),
    # Maintainer-only, and absent is its normal state for the same reason
    # `[dev]`'s is: the one command behind it needs write credentials on a
    # dataset repository nobody but the maintainers owns, so no user path
    # can reach it.  It is reported rather than hidden because it is a
    # declared extra, and an inventory that quietly skips one is an
    # inventory a reader cannot trust.
    "publish": _ExtraFacts(
        doors=("python -m tools.publish_geog_mirror upload -- pushing the "
               "WPS_GEOG mirror snapshot to Hugging Face",),
        still_works="every route that READS the mirror, which is every "
                    "user-facing one; this extra is for writing it",
        blocking=False, severity=SEVERITY_OPT_IN, expected_absent=True),
}

#: Fold keys for the extras block.  Two, not one: a run of aliases
#: and a run of real extras fold to the same word otherwise, and a
#: report that prints "pip extras (3)" three times reads as one thing
#: said three times rather than three different things.
_GROUP_EXTRAS = "pip extras"
_GROUP_EXTRA_ALIASES = "pip extra aliases"

#: Groups the brief folds into ONE line wherever their members appear,
#: rather than only when they happen to be adjacent.  Exactly one group
#: qualifies, on one property: every line in it carries no information
#: a reader can act on (an alias that resolves, no gap, no command), so
#: the position of each member is not a fact worth three lines.  Nothing
#: with a remedy belongs here -- folding those would hide WHERE in the
#: estate the gap is.
_SCATTERED_FOLD_GROUPS = frozenset({_GROUP_EXTRA_ALIASES})


def _extra_install_line(extra: str) -> str:
    """The one command that installs ``extra``, as it must be typed."""

    return f"pip install 'gpuwm[{extra}]'"


def _package_evidence(requirement: _Requirement,
                      probe: dict[str, tuple[bool, str]]
                      ) -> tuple[str, str, str]:
    """``(state, label, evidence)`` for one required package.

    ``state`` is one of ``"ok"``, ``"absent"``, ``"broken"``,
    ``"untested"`` or ``"excluded"``.  ``untested`` is not a euphemism
    for ok: it is the answer for a distribution whose import name this
    module does not know, where the honest report is that its metadata
    was read and nothing was imported.

    ``excluded`` is the state pip put the package in, not one this box
    can leave: an environment marker on the requirement means pip never
    tried to install it here.  Reporting that as ``absent`` produces the
    circular remedy in its second home -- ``wrf-rust ... not installed``
    over ``pip install 'gpuwm[render]'``, on an interpreter where that
    command installs nothing, reports success, and leaves doctor saying
    MISSING forever.
    """

    module = _IMPORT_NAME.get(requirement.distribution)
    try:
        version: str | None = importlib.metadata.version(
            requirement.distribution)
    except importlib.metadata.PackageNotFoundError:
        version = None
    if version is None and _marker_excludes_this_interpreter(requirement):
        # Only when it is genuinely absent: a marker is about what pip
        # WOULD install, and a package that is here anyway (a source
        # build, a newer upstream release) is judged on what it does.
        gap = _marker_gap(requirement)
        return ("excluded", requirement.specifier,
                gap or (f"this build's metadata excludes it on this "
                        f"interpreter ({requirement.marker}), and the "
                        f"project has not recorded why"))
    if module is None:
        if version is None:
            return ("absent", requirement.specifier, "not installed")
        return ("untested", requirement.specifier,
                f"{version} installed per distribution metadata; NOT "
                "imported (doctor does not know this package's import "
                "name)")
    if module not in probe:
        probe[module] = _import_probe(module, requirement.distribution)
    ok, evidence = probe[module]
    label = (requirement.specifier if module == requirement.distribution
             else f"{requirement.specifier} (imports as {module})")
    if version is None:
        # The DISTRIBUTION is what pip resolved, and an import cannot
        # see it: cupy-cuda12x and cupy-cuda13x both import as `cupy`,
        # so a box with the wrong wheel for its CUDA major answers
        # `import cupy` perfectly and is still the failure this whole
        # module was rewritten around.  Metadata first, import second.
        if ok:
            return ("absent", label,
                    f"not installed -- `import {module}` does succeed, but "
                    "from a different distribution; this one is not what "
                    "pip resolved here")
        return ("absent", label, "not installed")
    if ok:
        return ("ok", label, f"{evidence} (distribution {version})")
    return ("broken", label, evidence)


#: CUDA major -> the extra whose wheel serves it.  The inverse of
#: :data:`_GPU_EXTRA_BY_MAJOR`, used to say which of a mutually
#: exclusive pair matches THIS box.
_MAJOR_BY_GPU_EXTRA = {extra: major
                       for major, extra in _GPU_EXTRA_BY_MAJOR.items()}


def _mutually_exclusive_extra_check(name: str, extra: str, install: str,
                                    facts: _ExtraFacts,
                                    states: list) -> Check:
    """One of a pair where installing BOTH would be the fault.

    ``[gpu-cu12]`` and ``[gpu-cu13]`` are alternatives, one per CUDA
    major, and a healthy CUDA-12 box has exactly one of them.  Reporting
    the other as a MISSING gap would fail every correctly installed
    machine, so these lines never carry a verdict: they say which wheel
    this extra installs, whether pip resolved it here, which of the pair
    matches the CUDA major read off this box's own driver, and the exact
    install line.  The verdict, the remedy and the exit code stay on the
    one deep check that judges the wheel against the box.
    """

    installed = [label for _i, state, label, _w in states if state == "ok"]
    absent = [label for _i, state, label, _w in states if state != "ok"]
    major = _MAJOR_BY_GPU_EXTRA.get(extra)
    box_major = _driver_cuda_major()
    if box_major is None:
        fit = ("this box's CUDA major could not be read, so neither of "
               "the pair can be called the matching one here")
    elif major == box_major:
        fit = f"this box's driver serves CUDA {box_major}: THIS is the pair's matching extra"
    else:
        fit = (f"this box's driver serves CUDA {box_major}, so the "
               f"matching extra is "
               f"[{_GPU_EXTRA_BY_MAJOR.get(box_major, f'cuda-{box_major}')}], "
               "not this one")
    held = (f"{', '.join(installed)} installed" if installed
            else f"{', '.join(absent)} not installed")
    return Check(
        name, "info",
        f"{held}.  One extra per CUDA major, and a box needs exactly one: "
        f"{fit}.  Install line: {install}.  Needed by: "
        + ", ".join(facts.doors)
        + f".  The verdict and the remedy are on the `{facts.deferred_to}` "
          "line, which judges the installed wheel against this box",
        brief=_short(f"{held}; {fit}"), group=_GROUP_EXTRA_ALIASES)


def _extra_check(extra: str, requirements: tuple[_Requirement, ...],
                 probe: dict[str, tuple[bool, str]]) -> Check:
    """One extra: what it holds, what it unlocks, and how to install it."""

    name = f"pip extra [{extra}]"
    install = _extra_install_line(extra)
    aliases = [item for item in requirements if item.alias]
    if aliases and len(aliases) == len(requirements):
        resolved = ", ".join(sorted(item.specifier for item in aliases))
        return Check(
            name, "info",
            f"an alias: {install} resolves to {resolved}.  The verdict is "
            "on those extras' own lines",
            brief=f"alias for {resolved}", group=_GROUP_EXTRA_ALIASES)
    facts = _EXTRA_FACTS.get(extra)
    # A package a deeper check already judges is NAMED here and judged
    # there.  An older install's metadata can still carry rasterio and
    # pyproj inside [geog]; the geography-stack line is the one verdict
    # about them either way, so this block never publishes a second.
    owned = sorted({f"{item.specifier} -> `{_DEEP_CHECKED[item.distribution]}`"
                    for item in requirements
                    if not item.alias and item.distribution in _DEEP_CHECKED})
    states = [(item, *_package_evidence(item, probe))
              for item in requirements
              if not item.alias and item.distribution not in _DEEP_CHECKED]
    packages = ", ".join(label for _item, _state, label, _why in states)
    if facts is not None and facts.mutually_exclusive:
        return _mutually_exclusive_extra_check(
            name, extra, install, facts, states)
    if not states:
        # An extra with nothing left for this block to judge.  `[geog]`
        # became one in 2.3.3, deliberately: rasterio and pyproj moved
        # into the runtime dependencies and the empty extra was kept so
        # the install line printed in 2.3.2's own documentation keeps
        # resolving.  Silence here would leave a reader following that
        # documentation with no way to learn where its packages went.
        if owned:
            held = (f"its package(s) are judged elsewhere: "
                    + ", ".join(owned))
        else:
            held = (f"declares no packages on this build, so {install} "
                    "installs nothing and cannot fail")
        where = (f"  The verdict is on the `{facts.deferred_to}` line"
                 if facts is not None and facts.deferred_to and not owned
                 else "")
        return Check(
            name, "info",
            f"{held}.  The extra is retained so install lines that name "
            f"it keep resolving.{where}",
            brief=_short(held), group=_GROUP_EXTRA_ALIASES)
    absent = [(label, why) for _i, state, label, why in states
              if state == "absent"]
    broken = [(label, why) for _i, state, label, why in states
              if state == "broken"]
    untested = [(label, why) for _i, state, label, why in states
                if state == "untested"]
    excluded = [(label, why) for _i, state, label, why in states
                if state == "excluded"]

    if facts is None:
        # An extra this build declares and this module has never been
        # taught.  Reported loudly rather than skipped: a silent extra is
        # the whole defect this section closes.
        unlocks = ("what it unlocks is NOT RECORDED in gpuwm doctor -- "
                   "this build declares an extra doctor has not been "
                   "taught")
        blocking, severity = False, SEVERITY_DEGRADED
    else:
        unlocks = "needed by: " + ", ".join(facts.doors)
        blocking, severity = facts.blocking, facts.severity

    if broken:
        detail = ("; ".join(f"{label}: {why}" for label, why in broken)
                  + f".  {unlocks}")
        return Check(
            name, "missing", detail,
            f"# an installed package of [{extra}] does not import; "
            "reinstall it:\n" + install,
            action=install, brief=_short(detail), group=_GROUP_EXTRAS,
            blocking=True, severity=SEVERITY_BROKEN)

    if excluded:
        # A gap this box cannot close, so it prints no command.  MISSING
        # over an install line that installs nothing is worse than
        # silence: it sends a reader round a loop the project already
        # knows has no exit, which is why the sentence names the exit
        # (an interpreter this requirement HAS wheels for) and who owns
        # it.  Non-blocking for the same reason: `pip install gpuwm`
        # did everything it was asked, and an installer that trusts the
        # exit code must not fail a box for an upstream wheel matrix.
        detail = ("; ".join(f"{label}: {why}" for label, why in excluded)
                  + f".  {unlocks}")
        if facts is not None and facts.still_works:
            detail += f".  This box still has {facts.still_works}"
        return Check(
            name, "info", "not installable on this interpreter -- " + detail,
            brief=_short("; ".join(why for _label, why in excluded)),
            group=_GROUP_EXTRAS, blocking=False)

    if absent:
        missing = ", ".join(label for label, _why in absent)
        detail = f"{missing} not installed.  {unlocks}"
        if facts is not None and facts.still_works:
            detail += f".  Without it this box still has {facts.still_works}"
        remedy = install
        if facts is not None and facts.deferred_to:
            remedy = (f"# the verdict and the CUDA-major-specific remedy are "
                      f"on the `{facts.deferred_to}` line above\n{install}")
        if facts is not None and facts.expected_absent:
            return Check(
                name, "info",
                f"{missing} not installed, which is the normal state of an "
                f"install that is not developing gpuwm.  {unlocks}.  "
                f"Install line: {install}",
                brief=f"{missing} not installed (expected)",
                group=_GROUP_EXTRAS)
        return Check(
            name, "missing", detail, remedy, action=install,
            brief=f"{missing} not installed", group=_GROUP_EXTRAS,
            blocking=blocking, severity=severity)

    if untested:
        detail = ("; ".join(f"{label}: {why}" for label, why in untested)
                  + f".  {unlocks}")
        return Check(name, "untested", "not tested -- " + detail,
                     brief=_short(f"metadata only: {packages}"),
                     group=_GROUP_EXTRAS)

    return Check(name, "verified",
                 f"{packages} imported in subprocesses.  {unlocks}",
                 brief=_short(packages), group=_GROUP_EXTRAS)


def _base_dependency_check(base: tuple[_Requirement, ...],
                           probe: dict[str, tuple[bool, str]]) -> Check:
    """The dependencies pip installs with no extra asked for.

    Here because ``matplotlib`` used to be reported as half of the
    ``[render]`` extra, which it has never been: it is a base
    requirement, already installed on every box, and naming it in the
    render remedy hid the package that extra actually carries.  A base
    dependency that is absent or unimportable is not an opt-in -- it is
    a broken install, and it blocks.

    Requirements a DEEPER check already owns are named here and judged
    there.  ``rasterio``/``pyproj`` became runtime dependencies in
    2.3.3 and have their own geography-stack line; probing them twice
    would put two verdicts about one fact in one report, which is how a
    reader learns to believe neither.
    """

    name = "base dependencies (installed by `pip install gpuwm`)"
    states = [(item, *_package_evidence(item, probe))
              for item in base
              if not item.alias and item.distribution not in _DEEP_CHECKED]
    elsewhere = sorted({f"{item.specifier} -> `{_DEEP_CHECKED[item.distribution]}`"
                        for item in base
                        if item.distribution in _DEEP_CHECKED})
    faults = [(label, why) for _i, state, label, why in states
              if state in ("absent", "broken")]
    if faults:
        detail = "; ".join(f"{label}: {why}" for label, why in faults)
        return Check(
            name, "missing", detail,
            "# these are not optional and not extras: they are what\n"
            "  # `pip install gpuwm` itself installs, so an absent one\n"
            "  # means a damaged install rather than a choice --\n"
            "pip install --force-reinstall --no-deps gpuwm\n"
            "  # then, if it persists, reinstall with its dependencies:\n"
            "pip install --force-reinstall gpuwm",
            action="pip install --force-reinstall gpuwm",
            brief=_short(detail), blocking=True, severity=SEVERITY_BROKEN)
    packages = ", ".join(label for _i, _s, label, _w in states)
    detail = f"{packages} imported in subprocesses"
    if elsewhere:
        detail += ("; also required and reported on its own line: "
                   + ", ".join(elsewhere))
    return Check(name, "verified", detail, brief=_short(packages))


#: Packages no extra and no base requirement NAMES, which a documented
#: door nonetheless imports.  Each arrives transitively, and that is
#: exactly why it needs reporting: a transitive dependency is nobody's
#: declared contract, so nothing fails when it goes away.
#:
#: Pillow is here because `gpuwm render --pair`'s own refusal calls it
#: "installed with the render extra", and it is not: `[render]` is
#: wrf-rust + pyshp, and Pillow rides in with matplotlib, a BASE
#: dependency.  A reader following that message installs an extra that
#: cannot supply the package they are missing.
_TRANSITIVE_CONSUMERS = (
    ("PIL", "Pillow", "gpuwm render --pair",
     "arrives with matplotlib (a base dependency), NOT with the "
     "[render] extra"),
)


def _transitive_dependency_check() -> Check:
    """Packages a documented door imports that no requirement names."""

    name = "transitive dependencies (named by no requirement)"
    faults = []
    evidence = []
    for module, distribution, door, provenance in _TRANSITIVE_CONSUMERS:
        ok, why = _import_probe(module, distribution)
        if ok:
            evidence.append(f"{distribution} {why} ({provenance})")
        else:
            faults.append(f"{distribution}: {why} -- {door} needs it; "
                          f"it {provenance}")
    if faults:
        detail = "; ".join(faults)
        return Check(
            name, "missing", detail,
            "# these are not in any gpuwm extra, so no extra of any\n"
            "  # name installs them.  Reinstall the base dependency that\n"
            "  # carries them, or name the package directly:\n"
            "pip install --force-reinstall matplotlib",
            action="pip install --force-reinstall matplotlib",
            brief=_short(detail), blocking=False,
            severity=SEVERITY_DEGRADED)
    return Check(name, "verified", "; ".join(evidence),
                 brief=_short(", ".join(
                     distribution
                     for _m, distribution, _d, _p in _TRANSITIVE_CONSUMERS)))


def _extras_checks() -> list[Check]:
    """Every extra this INSTALLED distribution declares, plus the base.

    Order is the metadata's own, which is pyproject's declaration order
    through the wheel: the GPU pair, render, geog, obs, dealias, dev,
    then the aliases.  A reader comparing this report against the
    install lines in the documentation is reading them in the same
    order the packaging declares them.
    """

    declared = declared_requirements()
    if declared is None:
        return [Check(
            "pip extras", "untested",
            "not tested -- this interpreter has no installed gpuwm "
            "distribution to read extras from (a source tree on "
            "PYTHONPATH declares none).  Which extras exist, and which "
            "of them are installed, is a property of an INSTALL",
            "# install the package, even from the checkout, so its\n"
            "  # metadata exists to be read:\n"
            "pip install -e .",
            action="pip install -e .",
            brief="no installed distribution metadata")]
    base, extras = declared
    probe: dict[str, tuple[bool, str]] = {}
    checks = [_base_dependency_check(base, probe),
              _transitive_dependency_check()]
    for extra in extras:
        checks.append(_extra_check(extra, extras[extra], probe))
    return checks


def _geog_stack_check() -> Check:
    """Is the high-resolution path's PARITY FALLBACK available here?

    The check `gpuwm doctor` did not have in 2.3.2, which is why the
    failure had to be discovered by running the feature: rasterio and
    pyproj lived in an extra nothing documented, and the only thing that
    reported their absence was a traceback after a 160.7 MiB download.
    Doctor's whole job is to answer that before anything is run.

    What the row reports moved when the warp substrate flipped onto the
    Rust ``static-fields`` library.  The DEFAULT engine is that library,
    reported by the ``static builder`` row; these two are what
    ``GPUWM_STATIC_PYTHON=1`` runs on.  So the row is ``info`` rather
    than ``missing`` when they are absent: a wheel install with the
    library staged builds high-resolution terrain perfectly without
    them, and calling that MISSING would be doctor crying about a thing
    the shipped configuration does not read.  It becomes a real problem
    only when the library is also unavailable, and the ``static
    builder`` row is what says so.

    Deliberately a REAL import in a subprocess, not ``find_spec``.  Both
    libraries are thin Python over large native stacks (GDAL, PROJ), and
    the interesting failure on a box that has them installed is the one
    where the shared library will not load -- an ABI mismatch, a conda
    and pip GDAL fighting, a half-removed dist-info.  ``find_spec`` calls
    all of those green.  The front-door refusal uses the cheap probe
    because it runs on every build; doctor is where the expensive, honest
    answer belongs.
    """
    from gpuwm.static.geog_stack import GEOG_EXTRA, GEOG_MODULES

    results = {name: _import_probe(name) for name, _role in GEOG_MODULES}
    broken = {name: evidence for name, (ok, evidence) in results.items()
              if not ok}
    title = "geography stack (rasterio + pyproj, the highres fallback)"
    if not broken:
        versions = ", ".join(
            f"{name} {evidence}" for name, (_, evidence) in results.items())
        return Check(title, "verified",
                     f"imported in subprocesses ({versions}); "
                     "GPUWM_STATIC_PYTHON=1 can run the pure-Python "
                     "high-resolution fallback",
                     brief=_short(versions))
    detail = "; ".join(f"{name}: {evidence}"
                       for name, evidence in sorted(broken.items()))
    return Check(
        title, "info",
        f"{detail} -- the pure-Python high-resolution fallback "
        "(GPUWM_STATIC_PYTHON=1) cannot run here.  The DEFAULT engine is "
        "the Rust static-fields library; see the `static builder` row",
        GEOG_STACK_HINT, action=f"pip install 'gpuwm[{GEOG_EXTRA}]'",
        brief=_short(detail), blocking=False)


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


#: Why an ``unreachable`` finding can still exit 0.  Printed on the two
#: that do, so the summary's severity census and its blocking count are
#: never in unexplained disagreement.
_REFERENCE_DOOR_NOTE = (
    "This door is not one of the README's or FIRST-LIGHT's own first-run "
    "steps, so it reports UNREACHABLE without failing the exit code")

#: The two GRIB2 tools, and the directory the mapped/20CRv3 routes
#: build them in.  ``mapped_source._build_grib2_tools`` spells this
#: path as ``Path(__file__).resolve().parents[1] / "tools" /
#: "grib1_bridge"`` from a module that lives in the same package
#: directory as this one, so the two expressions are the same path by
#: construction -- and ``tests/test_doctor_mapped_route.py`` binds them
#: by capturing the ``cwd`` that function would hand to cargo.
_GRIB2_ROUTE_TOOLS = ("grib2_inventory", "grib2_dump")
_GRIB2_ROUTE_CRATE_RELATIVE = ("tools", "grib1_bridge")


def _grib2_route_crate() -> Path:
    """The directory the mapped/20CRv3 routes run ``cargo build`` in."""

    return Path(__file__).resolve().parents[1].joinpath(
        *_GRIB2_ROUTE_CRATE_RELATIVE)


def _packaged_grib2_source_ids() -> tuple[str, ...]:
    """The packaged-profile GRIB2 sources, named for the line above.

    Derived from the adapter table and the packaged profiles rather than
    hand-listed, because the list GROWS: every new GRIB2 model landed as
    a packaged profile joins it on the day its rows land, and a
    hand-written enumeration would still be naming the first two.  Empty
    on any read problem -- this feeds a report line, never a gate.

    Since the compose port these sources are the ones that do NOT launch
    the two executables on a bare run: their preparation composes inside
    ``gpuwm_mapped_engine``.  They are still enumerated because a reader
    of this line is asking "does MY source need these tools?", and the
    answer for every one of them is worth printing rather than implying.
    """

    try:
        from gpuwm.source_adapters import source_adapters
        from gpuwm.source_authorities import packaged_profile

        ids = []
        for adapter in source_adapters():
            if (adapter.packaged_profile is None
                    or adapter.runner != "mapped_composition_v1"):
                continue
            try:
                declared = packaged_profile(str(adapter.packaged_profile))
            except (KeyError, FileNotFoundError, RuntimeError):
                continue
            if declared["source_format"] == "grib2":
                ids.append(adapter.source_id)
        return tuple(ids)
    except Exception:  # a broken adapter table is its own finding
        return ()


def _mapped_grib2_route_check() -> Check:
    """Can the mapped/20CRv3 GRIB2 routes reach the tools they need?

    THIS IS THE ``ok`` THIS MODULE WAS CAUGHT PRINTING, and it is now the
    line that reports the fix.  Through 2.3.3 ``gpuwm fetch-bridges``
    staged ``grib2_inventory`` and ``grib2_dump``, ``_bridge_checks``
    probe-executed both and printed ``ok bridge grib2_inventory``, and
    then::

        $ gpuwm-mapped-inspect --mapping <the wheel's own authority> \\
              --input FILE.grib2
        NotADirectoryError: [WinError 267] The directory name is invalid

    Both of doctor's statements about the FILES were true.  Neither was a
    statement about the ROUTE, and the route is what the reader was
    asking about: ``mapped_source._build_grib2_tools()`` never consulted
    the staged copies, it shelled ``cargo build`` in ``<package
    parent>/tools/grib1_bridge``, a directory no wheel contains.

    That function consults :func:`gpuwm.bridges.find_bridge` FIRST now,
    the way ``ingest/grib.py`` always did, so the staged copies ARE what
    the default route uses.  And it is the resolver of RECORD: the
    ``rw-wps``/``gpuwm prep`` front door resolves the same two tools
    through the same function for every GRIB2 route it dispatches --
    mapped, 20CRv3, and each packaged-profile source -- so a bare
    ``gpuwm prep --source hrrr-prs`` needs no tool flags on a box this
    line calls verified.  The check applies the resolver's own static
    contract gate too, because "resolves" that hands the route a stale
    decoder is the same green-over-a-hole one release later.  This
    check still answers the route's question rather than the file's --
    it just has a third answer it did not have before:

    * both tools resolve through the ladder, and then the default route
      reaches them.  ``verified``.  This is the state a wheel install
      lands in after ``gpuwm fetch-bridges``, and reporting it as a gap
      would be the same defect in the opposite direction.
    * neither resolves, and a crate is present (a source checkout), and
      then doctor CANNOT answer: judging the build would mean running
      cargo, which this module never does.  ``untested``, and it says so.
    * neither resolves and there is no crate to build in.  ``missing``,
      and the remedy is ``gpuwm fetch-bridges`` with nothing appended to
      it, because staging is now sufficient on its own.

    It does not block.  See :func:`blocking_gaps`: the doors it closes
    (``gpuwm-mapped-inspect``, ``gpuwm adapt --descriptor``, the 20CRv3
    direct route) live in the CLI reference and the examples, not in the
    README's or FIRST-LIGHT's own first-run path.
    """

    name = "mapped/20CRv3 GRIB2 route (the tools it resolves)"
    # WHICH doors, exactly.  Before the compose port this line could say
    # "every packaged-profile GRIB2 source", because every mapped prep
    # ran the Python engine and launched these two executables.  It no
    # longer does: `compose` is declared for grib2, so a BARE prep of a
    # packaged-profile source resolves the engine and no subprocess tool
    # at all (`gpuwm prep --source hrrr-prs ... --dry-run` prints neither
    # flag).  Saying otherwise here would tell a reader their bytes are
    # decoded somewhere they are not, which is the same defect
    # `ENGINE_GAPS` exists to prevent, in a report line instead of a
    # table.  The doors below are the ones measured to still resolve
    # them: `gpuwm adapt` (adapt.py), `gpuwm-member-prep`'s identity
    # verification (member_prep.py), and any mapped run that asks for
    # the Python engine or names a tool.  The `--source 20crv3` runner
    # left this list when it moved onto `decode_composed_source`: a bare
    # member prep composes in the engine and reads its raw
    # record-inventory surface in process, so it resolves neither tool.
    packaged = _packaged_grib2_source_ids()
    doors = ("gpuwm adapt --descriptor/--input, gpuwm-member-prep, and "
             "any mapped run that asks for the Python engine with "
             "--mapped-engine python or names a tool"
             + (f" (the packaged GRIB2 profiles -- {', '.join(packaged)} -- "
                "20crv3's member route included, compose in the engine on "
                "a bare run and resolve neither)"
                if packaged else ""))
    staged: dict[str, Path | None] = {}
    override_faults: list[str] = []
    stale_notes: list[str] = []
    for tool in _GRIB2_ROUTE_TOOLS:
        try:
            resolved = bridges.find_bridge(tool)
        except FileNotFoundError as error:
            # A set environment override naming a missing file.  The
            # route raises on it too, by design, so it is a gap here
            # whatever else resolves.
            staged[tool] = None
            override_faults.append(str(error))
            continue
        if resolved is not None:
            # The same static contract gate the resolver of record
            # (mapped_source._build_grib2_tools) applies: a stale hit
            # is a hit the route will not accept, so reporting it
            # resolved would be the green-over-a-hole this check exists
            # to stop -- in its newest costume.
            ok, evidence = bridges.bridge_abi_matches(tool, resolved)
            if not ok:
                stale_notes.append(f"{tool} at {resolved} {evidence}")
                resolved = None
        staged[tool] = resolved
    crate = _grib2_route_crate()
    have_crate = (crate / "Cargo.toml").is_file()
    found = {tool: path for tool, path in staged.items() if path is not None}

    if (len(found) == len(_GRIB2_ROUTE_TOOLS) and not override_faults
            and not stale_notes):
        return Check(
            name, "verified",
            f"{', '.join(str(path) for path in found.values())} resolve "
            f"through the same ladder {doors} use, so the default route "
            "reaches them with no flags",
            brief="resolves both tools", group=_GROUP_BRIDGES)

    if have_crate:
        absent = [tool for tool in _GRIB2_ROUTE_TOOLS if tool not in found]
        note = (f"{', '.join(absent)} do not resolve through the ladder, and "
                f"the route falls back to a cargo build in {crate}.  Doctor "
                "never runs cargo (see this module's docstring), so whether "
                "that build succeeds here is unknown")
        if stale_notes:
            note += "; " + "; ".join(stale_notes)
        if override_faults:
            note += "; " + "; ".join(override_faults)
        return Check(name, "untested", f"not tested -- {note}",
                     brief="not tested; the route falls back to cargo",
                     group=_GROUP_BRIDGES)

    absent = [tool for tool in _GRIB2_ROUTE_TOOLS if tool not in found]
    detail = (
        f"{', '.join(absent)} not resolvable, and this install has no "
        f"tools/grib1_bridge crate to build them in ({crate}), so "
        f"{doors} cannot resolve the tools")
    if stale_notes:
        detail += "; " + "; ".join(stale_notes)
    if override_faults:
        detail += "; " + "; ".join(override_faults)
    remedy_lines = [
        "gpuwm fetch-bridges",
        "  # stages both tools where the mapped route already looks;",
        "  # --grib2-inventory / --grib2-dump override the resolved paths",
        "  # and are not needed for the default route."]
    if override_faults:
        remedy_lines.insert(0, "# unset the environment override(s) below, "
                               "or point them at a real build:")
    return Check(name, "missing", f"{detail}.  {_REFERENCE_DOOR_NOTE}",
                 "\n".join(remedy_lines),
                 action="gpuwm fetch-bridges",
                 brief=f"{', '.join(absent)} not resolvable by this route",
                 group=_GROUP_BRIDGES,
                 blocking=False, severity=SEVERITY_UNREACHABLE)


#: One next command for every obs front door the bundle cannot supply:
#: they share a clone and a cargo build, so a copy per door reads as
#: several problems.  Reached only when an artifact is absent from the
#: bundle; every one of them is IN it as of this release, so the live
#: remedy is `gpuwm fetch-bridges`.  Kept because the branch that needs
#: it is the branch a future front door lands on first.
_OBS_FRONT_DOOR_ACTION = ("build the obs front doors from a clone "
                          "of the renderer workspace")


#: Every bundled artifact, and the check that reports it.  A SET, kept
#: by hand and guarded by a test, because the alternative is the defect
#: it exists to stop: each of these checks is written per artifact, so
#: an artifact added to the bundle is invisible here until someone
#: remembers to write its check -- which is exactly how ``rw_nexrad``
#: and then the three observation front doors came to be absent from a
#: report that called the estate green.
#:
#: :func:`_bundle_coverage_checks` turns the set into a live sweep, so a
#: new artifact is REPORTED (as untested, never as ok) from the moment
#: the bundle carries it, rather than waiting for this map to catch up.
#: ``tests/test_doctor_route_honesty.py`` fails when they disagree.
_CHECKED_ARTIFACTS = {
    "grib1_bridge": "the `bridge ...` lines",
    "gfs_grib2_bridge": "the `bridge ...` lines",
    "hrrr_grib2_bridge": "the `bridge ...` lines",
    "grib2_inventory": "the `bridge ...` and mapped/20CRv3 route lines",
    "grib2_dump": "the `bridge ...` and mapped/20CRv3 route lines",
    "gpuwm_preprocess_cpu": "the `cpu preprocess library` line",
    "rw_fetch": "the `fetch backbone` line",
    "rw_wrfbatch": "the `renderer` line",
    "rw_nexrad": "the `radar front door` line",
    "region_global_dealias": "the `region-global dealiasing engine` line",
    "rw_odim": "the `obs front door` lines",
    "rw_mrms": "the `obs front door` lines",
    "rw_stage4": "the `obs front door` lines",
    "rw_asos": "the `obs front door` lines",
    "rw_goes": "the `obs front door` lines",
    "rw_opera": "the `obs front door` lines",
    "rw_netcdf": "the `NetCDF decoder` line",
    "netcdf_writer": "the `NetCDF writer` line",
    "gpuwm_mapped_engine": "the `mapped decode engine` line",
    "static_fields": "the `static builder` line",
    "obs_regrid": "the `observation remap` line",
}


def _bundle_coverage_checks() -> list[Check]:
    """Is every artifact the bundle carries reported by some check?

    The one check in this module that is ABOUT the report.  Each of the
    others is written per artifact, so the estate's completeness has
    always depended on somebody remembering -- and twice it did not:
    ``rw_nexrad`` shipped outside both audited sets and doctor passed on
    boxes where every radar route was dead, and the three observation
    front doors did the same thing again.  A bundle that grows a
    fourteenth artifact must not need a third incident.

    An uncovered artifact is resolved through the same ladder its
    consumer would use and then reported ``untested`` when it is there
    -- present, unprobed, and saying so -- because doctor knows no
    contract for a binary nobody has taught it about.  Never ``ok``.
    """

    try:
        from gpuwm import bridge_assets

        bundled = tuple(bridge_assets.BUNDLED_ARTIFACTS)
    except Exception as error:                   # noqa: BLE001 - reported
        return [Check(
            "bundled artifact coverage", "untested",
            f"not tested -- the bundle manifest could not be read "
            f"({type(error).__name__}: {error}), so doctor cannot say "
            "whether every artifact it carries is reported above",
            "# reinstall so the packaged bundle manifest loads:\n"
            + REINSTALL_HINT,
            action="reinstall gpuwm", brief="bundle manifest unreadable")]

    uncovered = [artifact for artifact in bundled
                 if artifact.name not in _CHECKED_ARTIFACTS]
    if not uncovered:
        owners = sorted({_CHECKED_ARTIFACTS[artifact.name]
                         for artifact in bundled})
        return [Check(
            "bundled artifact coverage", "verified",
            f"all {len(bundled)} artifact(s) this release's bundle carries "
            f"are reported by a check above ({'; '.join(owners)})",
            brief=f"{len(bundled)} of {len(bundled)} bundled artifacts "
                  "reported")]

    checks: list[Check] = []
    for artifact in uncovered:
        name = f"bundled artifact {artifact.name} (no check of its own)"
        try:
            filename = bridge_assets.artifact_filename(
                artifact, bridge_assets.host_platform() or "linux-x86_64")
            found = bridges.find_artifact(artifact.env_var, filename)
        except Exception as error:               # noqa: BLE001 - reported
            checks.append(Check(
                name, "untested",
                f"not tested -- it is in this release's bundle and doctor "
                f"has no check for it; resolving it also failed "
                f"({type(error).__name__}: {error})",
                brief="in the bundle, unresolvable, unchecked"))
            continue
        if found is None:
            checks.append(Check(
                name, "missing",
                f"in this release's bundle and not staged here; needed by: "
                f"{artifact.consumer}.  Doctor has no contract probe for "
                "this artifact, so staging it is all this line can ask for",
                "gpuwm fetch-bridges\n"
                "  # stages every pinned artifact, this one included",
                action="gpuwm fetch-bridges",
                brief="bundled, not staged, and unchecked",
                group=_GROUP_BRIDGES, blocking=False,
                severity=SEVERITY_DEGRADED))
            continue
        checks.append(Check(
            name, "untested",
            f"not tested -- {found} is staged, and doctor has no contract "
            f"probe for this artifact (needed by: {artifact.consumer}).  "
            "Its presence was checked; its contract was not",
            brief="staged; contract not probed", group=_GROUP_BRIDGES))
    return checks


def _obs_front_door_checks() -> list[Check]:
    """Every observation front door, reported by name.

    ``gpuwm/obs/frontdoor.py`` resolves each of these through the same
    ladder as ``rw_nexrad`` and refuses by name when one is absent.
    Through 2.3.3 none of them was in ``BUNDLED_ARTIFACTS``, so ``gpuwm
    fetch-bridges`` printed "all artifacts already staged and pin-valid"
    and the refusal repeated verbatim -- and doctor, which audited
    exactly the bundled set, reported a fully green estate on a box
    where every obs front door was absent.

    Both halves of that are fixed now and this function is where they
    meet.  The binaries are REPORTED, so the estate is no longer green
    over them; and the remedy is a FUNCTION of the bundle manifest
    rather than an assumption, so it reads ``gpuwm fetch-bridges``
    exactly when that command can supply the artifact.  As of this
    release every door below is bundled, so it always can -- but the
    branch that says otherwise is kept, because the next front door to
    land will land on it before it lands in a bundle.

    Non-blocking: these doors are named in no README and no FIRST-LIGHT
    step.
    """

    try:
        from gpuwm.obs import frontdoor
    except ImportError as error:                 # pragma: no cover - partial
        return [Check(
            "observation front doors (MRMS / Stage-IV / ASOS)", "missing",
            f"gpuwm.obs is not importable ({error}) -- blocks every "
            "observation front door and means this install is incomplete",
            "# reinstall so the observation stack imports:\n"
            + REINSTALL_HINT,
            action="reinstall gpuwm", brief="obs stack not importable",
            group=_GROUP_ENGINES, severity=SEVERITY_BROKEN)]
    try:
        from gpuwm.bridge_assets import BUNDLED_ARTIFACTS

        bundled = {artifact.name for artifact in BUNDLED_ARTIFACTS}
    except Exception:                            # noqa: BLE001 - reported
        bundled = set()

    # The door NAMED per instrument is the product subcommand, because
    # that is the one a reader can type after `pip install gpuwm`.  The
    # campaign drivers under tools/ are the same binaries reached the
    # other way and are named beside the three that have one; `gpuwm
    # obs <instrument>` is what every one of them has.
    doors = {"mrms": "gpuwm obs mrms (and python -m tools.obs_fetch_mrms)",
             "stage4": "gpuwm obs stage4 (and python -m "
                       "tools.obs_fetch_stage4)",
             "asos": "gpuwm obs asos (and python -m tools.obs_fetch_asos)",
             "goes": "gpuwm obs goes",
             "opera": "gpuwm obs opera",
             "odim": "gpuwm obs odim, and every `gpuwm obs radar` "
                     "subcommand"}
    # Read from the resolver rather than enumerated here, so a door added
    # to FRONT_DOORS and not to `doors` above is a loud KeyError in the
    # test suite instead of a silently unreported binary -- which is the
    # exact failure this whole function exists to have stopped.
    unnamed = sorted(set(frontdoor.FRONT_DOORS) - set(doors))
    assert not unnamed, (
        f"gpuwm.obs.frontdoor gained {unnamed} and gpuwm.doctor does not "
        "name the door(s) they unlock")
    checks: list[Check] = []
    for instrument, door in sorted(doors.items()):
        front = frontdoor.FRONT_DOORS[instrument]
        name = f"obs front door {front.name} ({front.subject})"
        in_bundle = front.name in bundled
        try:
            found = front.find()
        except FileNotFoundError as error:
            checks.append(Check(
                name, "missing", f"{error} -- {door} cannot run",
                f"# {front.env_var} names a missing executable: point it "
                "at a real build, or unset it --\n"
                + bridges.install_aware_build_hint(
                    frontdoor.CARGO_BUILD_HINT, bridges.RUSTWX_CRATE_RELATIVE),
                action=f"unset {front.env_var}, or point it at a real build",
                brief=f"{front.env_var} names a missing file",
                group=_GROUP_ENGINES, blocking=False,
                severity=SEVERITY_UNREACHABLE))
            continue
        if found is None:
            # The remedy is a FUNCTION of whether the bundle can supply
            # this artifact, read from the bundle manifest rather than
            # assumed -- and assuming is the defect being fixed.  The
            # resolver's own refusal opens with `gpuwm fetch-bridges`
            # because *a* bundle exists for the platform, without ever
            # asking whether THIS artifact is in it, so a reader ran it,
            # was told all pinned artifacts were staged, and met the
            # identical refusal again.  When a release does start
            # bundling these, this line becomes that one command with no
            # edit here.
            if in_bundle:
                bundle_note = ("it IS in this release's prebuilt bundle, so "
                               "one command stages it")
                remedy = ("gpuwm fetch-bridges\n"
                          "  # stages every pinned artifact for this "
                          "platform, this one included")
                action = "gpuwm fetch-bridges"
            else:
                bundle_note = (
                    "and `gpuwm fetch-bridges` will NOT supply it: this "
                    "artifact is not in the bundle at all, so that command "
                    "reports every pinned artifact staged and this door "
                    "stays shut")
                remedy = (
                    f"# {front.name} is not a bundled artifact.  The only\n"
                    "  # route to it is a clone and a build of the renderer\n"
                    "  # workspace:\n"
                    + bridges.install_aware_build_hint(
                        frontdoor.CARGO_BUILD_HINT,
                        bridges.RUSTWX_CRATE_RELATIVE)
                    + f"\n  # then copy "
                    f"{bridges.executable_name(front.name)} "
                    f"into {bridges.default_bridge_dir()},\n"
                    f"  # or set {front.env_var} to its full path")
                # One action for all three, so the terse report folds
                # them into a single line: they share one clone, one
                # cargo build, and one reason.
                action = _OBS_FRONT_DOOR_ACTION
            checks.append(Check(
                name, "missing",
                f"not built and not staged -- {door} cannot run, "
                f"{bundle_note}.  {_REFERENCE_DOOR_NOTE}",
                remedy, action=action,
                brief=f"not staged; {door} cannot run",
                group=_GROUP_ENGINES, blocking=False,
                severity=SEVERITY_UNREACHABLE))
            continue
        ok, evidence = front.probe(found)
        if not ok:
            checks.append(Check(
                name, "missing", f"{found} -- {evidence}; {door} cannot run",
                "# REBUILD it -- this is a record-contract change, so\n"
                "  # another copy of the same vintage fails identically --\n"
                + bridges.install_aware_build_hint(
                    frontdoor.CARGO_BUILD_HINT, bridges.RUSTWX_CRATE_RELATIVE),
                action=f"rebuild {front.name}", brief=_short(evidence),
                group=_GROUP_ENGINES, blocking=False,
                severity=SEVERITY_UNREACHABLE))
            continue
        checks.append(Check(name, "verified", f"{found} -- {evidence}",
                            brief=_short(evidence), group=_GROUP_ENGINES))
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
            detail + " -- gpuwm fetch falls back to the Python transport, "
            "which has no whole-file branch: every object arrives as "
            "hundreds of serial .idx range GETs, measured at 560 s for "
            "one 419 MB HRRR file against 27-35 s for the same file "
            "taken whole (~16x).  A run that pays it records "
            "engine_selection='python-fallback' in its fetch manifest",
            bridges.install_aware_build_hint(
                rustwx_fetch.CARGO_BUILD_HINT, "tools/rustwx")
            + "\n  # enables gpuwm fetch --engine rust: parallel range "
            "GETs,\n  # the cross-process NOMADS rate governor, and "
            "--mode full-file",
            action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
            brief="not built; gpuwm fetch pays a measured ~16x transport "
                  "tax",
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

    A bundled binary with **no fallback**.  A missing fetch backbone
    leaves the Python transport, so that one is ``info``; there is no
    second way to turn a radar volume into observations, so an absent
    ``rw_nexrad`` is the difference between a box that can assimilate
    and a box that cannot.  (It is not the ONLY such binary, which this
    docstring used to claim: ``region_global_dealias`` has no fallback
    either -- ``vad-region`` is a different solver, not a substitute --
    and the renderer's "matplotlib fallback" turned out to need the
    ``[render]`` extra as well.  Both were measured, 2026-08-14.)
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


def _netcdf_decoder_check() -> Check:
    """The NetCDF decoder, which every NetCDF source now needs.

    ``missing`` rather than ``info``, for the same reason ``rw_nexrad``
    is: since NetCDF decode moved off ``netCDF4.Dataset`` and onto the
    vendored ``netcrust`` reader, an absent ``rw_netcdf`` does not cost
    an option -- it costs every NetCDF source the product accepts, and
    there is deliberately no Python decoder behind it to quietly take
    over.  A prerequisite of a shipped path belongs in the blocking
    group, and the remedy is the build, never a workaround flag.
    """

    name = "NetCDF decoder (every NetCDF source)"
    blocks = ("blocks every NetCDF source -- gpuwm adapt, the mapped "
              "routes, rw-wps --source netcdf")
    try:
        from gpuwm import netcdf_bridge
    except ImportError as error:                 # pragma: no cover - partial
        return Check(name, "missing",
                     f"gpuwm.netcdf_bridge is not importable ({error}) -- "
                     f"{blocks}",
                     "# reinstall so the NetCDF front door imports:\n"
                     "  pip install --force-reinstall gpuwm",
                     brief="netcdf_bridge not importable",
                     group=_GROUP_ENGINES)
    try:
        found = netcdf_bridge.find_netcdf_bin()
    except FileNotFoundError as error:
        return Check(
            name, "missing", f"{error} -- {blocks}",
            f"# {netcdf_bridge.NETCDF_ENV} names a missing executable: "
            "point it at a real build, or unset it and get one --\n"
            + netcdf_bridge.netcdf_remedy(),
            action=f"unset {netcdf_bridge.NETCDF_ENV}, or point it at a "
                   f"real build",
            brief=f"{netcdf_bridge.NETCDF_ENV} names a missing file",
            group=_GROUP_ENGINES)
    if found is None:
        return Check(
            name, "missing",
            "no prebuilt executable (searched "
            + ", ".join(str(c) for c in netcdf_bridge.netcdf_candidates())
            + f") -- {blocks}",
            netcdf_bridge.netcdf_remedy()
            + "\n  # without it no NetCDF source can be read at all: gpuwm\n"
            "  # decodes NetCDF in Rust and keeps no Python fallback",
            action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
            brief="not staged; no NetCDF source can be read",
            group=_GROUP_ENGINES)
    return Check(name, "verified", str(found),
                 brief="staged", group=_GROUP_ENGINES)


def _mapped_engine_check() -> Check:
    """The mapped decode engine, and whether a bare mapped run needs it.

    Two verdicts from one probe, because the answer genuinely depends on
    which engine is the default: while the default is the Python engine
    an absent binary costs nothing and saying ``missing`` would be a
    false alarm, and the moment the default flips an absent binary costs
    every mapped source and anything softer than ``missing`` would be
    the ``ok``-on-a-dead-route defect this report exists to prevent.
    The check reads :data:`gpuwm.mapped_engine_bridge.DEFAULT_ENGINE`
    rather than restating it, so the flip moves this line with it.
    """

    name = "mapped decode engine (gpuwm prep --source mapped)"
    try:
        from gpuwm import mapped_engine_bridge as engine_bridge
    except ImportError as error:                 # pragma: no cover - partial
        return Check(name, "missing",
                     f"gpuwm.mapped_engine_bridge is not importable ({error})",
                     "# reinstall so the mapped seam imports:\n"
                     "  pip install --force-reinstall gpuwm",
                     brief="mapped_engine_bridge not importable",
                     group=_GROUP_ENGINES)
    default = engine_bridge.DEFAULT_ENGINE
    blocker = engine_bridge.DEFAULT_ENGINE_BLOCKER
    required = default == engine_bridge.ENGINE_RUST
    try:
        found = engine_bridge.find_engine()
    except FileNotFoundError as error:
        return Check(
            name, "missing", str(error),
            f"# {engine_bridge.ENGINE_PATH_ENV} names a missing "
            "executable: point it at a real build, or unset it --\n"
            + engine_bridge.engine_remedy_lines(
                "the override names no file"),
            action=f"unset {engine_bridge.ENGINE_PATH_ENV}, or point it "
                   "at a real build",
            brief=f"{engine_bridge.ENGINE_PATH_ENV} names a missing file",
            group=_GROUP_ENGINES)
    if found is None:
        detail = (
            "no prebuilt executable (searched "
            + ", ".join(str(c) for c in engine_bridge.engine_candidates())
            + ")")
        if required:
            return Check(
                name, "missing",
                f"{detail} -- blocks every mapped source, which is the "
                "default decode path",
                engine_bridge.engine_remedy_lines("it is not staged"),
                action=_build_action(engine_bridge.ENGINE_CRATE_RELATIVE),
                brief="not staged; no mapped source can be read",
                group=_GROUP_ENGINES)
        return Check(
            name, "info",
            f"{detail}. Mapped sources decode on the {default} engine on "
            f"this release, so nothing is blocked: {blocker}",
            engine_bridge.engine_remedy_lines("it is not staged"),
            action=_build_action(engine_bridge.ENGINE_CRATE_RELATIVE),
            brief=f"not staged; mapped decode runs on {default}",
            group=_GROUP_ENGINES, blocking=False)
    matches, reason = bridges.bridge_abi_matches(
        engine_bridge.ENGINE_NAME, found)
    if not matches:
        return Check(
            name, "missing" if required else "info",
            f"{found} {reason}",
            engine_bridge.engine_remedy_lines(reason),
            action=_build_action(engine_bridge.ENGINE_CRATE_RELATIVE),
            brief="staged build predates this release's frameset contract",
            group=_GROUP_ENGINES, blocking=required)
    if required:
        # The paths still routed to Python are named HERE rather than
        # left for a reader to discover from a receipt: this line is
        # what a user reads to learn where their bytes are decoded, and
        # "the Rust engine is the default" without the exceptions would
        # be a true sentence that leaves a false impression.
        suffix = "".join(f"; {gap}" for gap in engine_bridge.ENGINE_GAPS)
    else:
        suffix = (f"; mapped decode runs on the {default} engine by default "
                  f"on this release -- {blocker}")
    return Check(
        name, "verified", f"{found} speaks {engine_bridge.FRAMESET_SCHEMA}"
        + suffix,
        brief=f"staged; default engine is {default}",
        group=_GROUP_ENGINES)


def _ncwrite_check() -> Check:
    """The NetCDF WRITER, which two default engines now need.

    ``missing`` rather than ``info``, for the same reason ``rw_netcdf``
    is: since the product tape and then the wrfinput/wrfbdy pair flipped
    onto the Rust classic writer, an absent ``netcdf_writer`` library
    does not cost an option -- it costs every history write AND every
    WRF-input export, and both refuse loudly rather than silently
    downgrading a container with build state.  The remedy is the build;
    ``GPUWM_WRFOUT_WRITER=python`` and ``GPUWM_WRFINPUT_WRITER=python``
    are named as what they are, documented workarounds onto the netCDF4
    writer, not fallbacks the product takes on its own.
    """

    name = "NetCDF writer (default wrfout and wrfinput engine)"
    blocks = ("blocks every history write and every WRF-input export -- "
              "gpuwm sim, gpuwm go, gpuwm prep, gpuwm-wrf-init, rw-wps, "
              "the tilestream history path")
    try:
        from gpuwm.io import nc_writer_bridge
    except ImportError as error:                 # pragma: no cover - partial
        return Check(name, "missing",
                     f"gpuwm.io.nc_writer_bridge is not importable "
                     f"({error}) -- {blocks}",
                     "# reinstall so the writer seam imports:\n"
                     "  pip install --force-reinstall gpuwm",
                     brief="nc_writer_bridge not importable",
                     group=_GROUP_ENGINES)
    # Install-aware: `gpuwm fetch-bridges` first on a wheel, the in-tree
    # build for a checkout -- then name the workaround AS a workaround.
    remedy = bridges.artifact_remedy(
        env_var=nc_writer_bridge.NCWRITE_BRIDGE_ENV,
        filename=nc_writer_bridge.library_names()[0],
        subject="the NetCDF writer",
        crate_relative=bridges.RUSTWX_CRATE_RELATIVE) + (
        "\n  # GPUWM_WRFOUT_WRITER=python (history) and\n"
        "  # GPUWM_WRFINPUT_WRITER=python (the WRF input pair) select the\n"
        "  # netCDF4 writer as an explicit workaround, each onto its own\n"
        "  # HDF5 container (NETCDF4_CLASSIC / NETCDF4)")
    try:
        path = nc_writer_bridge.resolve_ncwrite_bridge()
    except FileNotFoundError as error:
        return Check(
            name, "missing", f"{error} -- {blocks}", remedy,
            action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
            brief="not staged; default history/WRF-input writes blocked",
            group=_GROUP_ENGINES)
    reason = nc_writer_bridge.unavailable_reason()
    if reason is not None:
        return Check(
            name, "missing", f"{path} -- {reason} -- {blocks}", remedy,
            action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
            brief="not loadable; default history/WRF-input writes blocked",
            group=_GROUP_ENGINES)
    return Check(name, "verified",
                 f"{path} -- ABI {nc_writer_bridge.NCWRITE_ABI}",
                 brief="staged", group=_GROUP_ENGINES)


def _static_builder_check() -> Check:
    """The static-field builder, which the default build path now needs.

    ``missing`` rather than ``info``, the ``netcdf_writer`` posture:
    since ``build_static`` and the ProjectedGrid array methods flipped
    onto the Rust crate by default (the static-rust-port lanes), an
    absent ``static_fields`` library does not cost an option -- it turns
    every static build into a reported WORKAROUND onto the numpy
    fallback.  The fallback still produces byte-identical fields (the
    dual-run parity record in docs/dev/static-rust-port.md), so the
    breakage this row names is the degradation itself: a wheel whose
    every prepare prints a workaround line is not running the shipped
    configuration, and at corridor scale the fallback is measured ~11x
    slower.
    """

    name = "static builder (default static-field engine)"
    degrades = ("every static build -- gpuwm go's prepare stage, nest "
                "initialization, the statics corridor, and the whole "
                "[static.highres] overlay including its GeoTIFF decode "
                "and warp -- runs the pure-Python fallback as a reported "
                "WORKAROUND, which needs the [geog] extra installed")
    try:
        from gpuwm.static import rust_bridge
    except ImportError as error:                 # pragma: no cover - partial
        return Check(name, "missing",
                     f"gpuwm.static.rust_bridge is not importable "
                     f"({error}) -- {degrades}",
                     "# reinstall so the builder seam imports:\n"
                     "  pip install --force-reinstall gpuwm",
                     brief="rust_bridge not importable",
                     group=_GROUP_ENGINES)
    remedy = bridges.artifact_remedy(
        env_var=rust_bridge.STATIC_BRIDGE_ENV,
        filename=rust_bridge.library_names()[0],
        subject="the static-field builder",
        crate_relative=bridges.RUSTWX_CRATE_RELATIVE) + (
        "\n  # GPUWM_STATIC_PYTHON=1 selects the numpy fallback as an\n"
        "  # explicit workaround (byte-identical fields, measured ~11x\n"
        "  # slower at corridor scale)")
    try:
        path = rust_bridge.resolve_static_bridge()
    except FileNotFoundError as error:
        return Check(
            name, "missing", f"{error} -- {degrades}", remedy,
            action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
            brief="not staged; static builds degrade to the fallback",
            group=_GROUP_ENGINES)
    reason = rust_bridge.unavailable_reason()
    if reason is not None:
        return Check(
            name, "missing", f"{path} -- {reason} -- {degrades}", remedy,
            action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
            brief="not loadable; static builds degrade to the fallback",
            group=_GROUP_ENGINES)
    return Check(name, "verified",
                 f"{path} -- ABI {rust_bridge.STATIC_ABI}",
                 brief="staged", group=_GROUP_ENGINES)


def _obs_regrid_check() -> Check:
    """The observation remap, which the battery's default now needs.

    ``missing`` rather than ``info``, the ``static_fields`` posture:
    since `gpuwm.verify.obs.regrid` flipped onto the Rust crate by
    default, an absent ``obs_regrid`` library does not cost an option --
    every observation score becomes a reported WORKAROUND onto the scipy
    fallback.

    The breakage this row names is not slowness.  On the scipy path two
    source cells exactly equidistant from one destination cell are
    resolved by whichever the cKDTree traversal reached first (measured:
    232 of 400 trials answered the lowest index, 168 answered another),
    so a remap plan stops being a function of its two grids alone.  The
    battery's premise is that every arm is remapped by the identical
    integer array, which is what makes arm scores comparable; a plan
    whose ties turn on traversal order is that premise, unenforced.
    """

    name = "observation remap (default battery remap engine)"
    degrades = ("every observation score -- python -m "
                "tools.obs_battery_score, tools.obs_precampaign_controls, "
                "tools.obs_battery_registration -- runs the scipy fallback "
                "as a reported WORKAROUND, where exactly-tied nearest "
                "neighbours are resolved by cKDTree traversal order rather "
                "than by lowest source index")
    try:
        from gpuwm import obs_regrid_bridge as regrid_bridge
    except ImportError as error:                 # pragma: no cover - partial
        return Check(name, "missing",
                     f"gpuwm.obs_regrid_bridge is not importable "
                     f"({error}) -- {degrades}",
                     "# reinstall so the remap seam imports:\n"
                     "  pip install --force-reinstall gpuwm",
                     brief="regrid_bridge not importable",
                     group=_GROUP_ENGINES)
    remedy = bridges.artifact_remedy(
        env_var=regrid_bridge.OBSREGRID_BRIDGE_ENV,
        filename=regrid_bridge.library_names()[0],
        subject="the observation remap engine",
        crate_relative=bridges.RUSTWX_CRATE_RELATIVE) + (
        "\n  # GPUWM_OBSREGRID_PYTHON=1 selects the scipy fallback as an\n"
        "  # explicit workaround (bit-identical everywhere except exact\n"
        "  # neighbour ties, where scipy has no defined answer)")
    try:
        path = regrid_bridge.resolve_obsregrid_bridge()
    except FileNotFoundError as error:
        return Check(
            name, "missing", f"{error} -- {degrades}", remedy,
            action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
            brief="not staged; observation scores degrade to the fallback",
            group=_GROUP_ENGINES)
    reason = regrid_bridge.unavailable_reason()
    if reason is not None:
        return Check(
            name, "missing", f"{path} -- {reason} -- {degrades}", remedy,
            action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
            brief="not loadable; observation scores degrade to the fallback",
            group=_GROUP_ENGINES)
    return Check(name, "verified",
                 f"{path} -- ABI {regrid_bridge.OBSREGRID_ABI}",
                 brief="staged", group=_GROUP_ENGINES)


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


def _matplotlib_engine_note() -> str:
    """What ``gpuwm render --engine matplotlib`` would do, on THIS install.

    This function exists because the sentence it replaces was false.
    Doctor said "matplotlib remains the documented fallback" in three
    places, and the matplotlib engine imports the ``wrf`` package at the
    top of its render loop -- every product it draws is a ``wrf.getvar``
    call -- so on an install without ``[render]`` there was no fallback
    at all.  The sentence has to be a question, asked here.

    It is no longer a FALLBACK in either arm.  ``--engine auto``
    refuses when ``rw_wrfbatch`` is unusable rather than degrading,
    because the render law (CLAUDE.md, Drew 2026-08-06) reserves
    weather-field product plots for that binary and names exactly one
    permitted fallback, which is not this engine (audit F7).  So the
    honest doctor line is about a WORKAROUND a reader can type, never
    about a safety net that catches them.
    """

    ok, _evidence = _import_probe("wrf", "wrf-rust")
    if ok:
        return ("`gpuwm render --engine matplotlib` is available here as "
                "a NAMED WORKAROUND (the wrf package imports); the "
                "default --engine auto refuses rather than degrading to "
                "it, because weather fields come from rw_wrfbatch")
    if not science_core.python_supports_science_core(_python_version()):
        # Not "not installed": not INSTALLABLE.  Telling a reader over
        # the ceiling to run pip for a package with no wheel for their
        # interpreter is the same dead loop the extras line used to
        # print.  The sentence is derived, so the day upstream publishes
        # the wheels this branch stops firing on its own -- which is
        # what 0.2.39 did for 3.14.
        return ("`gpuwm render --engine matplotlib` cannot run on this "
                "interpreter, and pip cannot supply what it needs: "
                + science_core.python_gap_sentence(_python_version())
                + ".  The DEFAULT rust engine is unaffected -- it draws "
                  "from the rw_wrfbatch binary and imports no Python "
                  "science core at all")
    return ("gpuwm render has NO engine left here: the matplotlib "
            "workaround imports the wrf package from the [render] "
            "extra, which is not installed (see the `pip extra "
            "[render]` line)")


def _rust_renderer_check() -> Check:
    """The vendored Rusty Weather renderer: probe-execute, not stat().

    ``gpuwm render`` defaults to this engine exactly when the check
    passes.  Whether its absence is a gap depends on something this
    check cannot see on its own -- the matplotlib engine needs the
    ``[render]`` extra too -- so the status stays ``info``/``missing``
    and non-blocking here, and the sentence about the fallback is
    computed (:func:`_matplotlib_engine_note`) rather than asserted.
    The blocking verdict on an absent ``[render]`` belongs to the extras
    block, which owns that package; two lines blocking on one fact would
    count one problem twice.
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
        note = _matplotlib_engine_note()
        return Check(
            name, "info", f"{detail} -- {note}",
            bridges.install_aware_build_hint(
                rustwx.CARGO_BUILD_HINT, "tools/rustwx")
            + "\n  # enables --engine rust and makes it the default",
            action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
            brief=_short(f"not built; {note}"),
            group=_GROUP_ENGINES)
    ok, evidence = rustwx.probe_renderer(found)
    if not ok:
        # Non-blocking here on purpose, and NOT because "matplotlib
        # remains the fallback" -- that sentence was false on a base
        # install.  The extras block owns the [render] verdict; this
        # line says what is true of the binary and names the state of
        # the other engine.
        note = _matplotlib_engine_note()
        return Check(
            name, "missing", f"{found} -- {evidence}; {note}",
            "# it has to be replaced:\n" + bridges.install_aware_build_hint(
                rustwx.CARGO_BUILD_HINT, "tools/rustwx"),
            action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
            brief=_short(evidence), group=_GROUP_ENGINES,
            blocking=False, severity=SEVERITY_DEGRADED)
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
            "(--preprocess-backend cpu needs it, and so does "
            "--preprocess-backend auto wherever CUDA is unusable; the "
            "radar dealiaser's coarse VAD search falls back to NumPy "
            f"without it): {error}", remedy,
            action=_build_action(), brief="not staged",
            group=_GROUP_BRIDGES, blocking=False,
            severity=SEVERITY_DEGRADED)
    except (OSError, RuntimeError, AttributeError) as error:
        return Check(
            "cpu preprocess library", "missing",
            f"found but not loadable as ABI v{CPU_BACKEND_ABI}: {error}",
            "# it has to be replaced:\n" + remedy,
            action=_build_action(),
            brief=f"not loadable as ABI v{CPU_BACKEND_ABI}",
            group=_GROUP_BRIDGES)
    path, abi = backend.path, backend.abi_version
    indexed = backend.indexed_donor_interp
    backend.close()
    # The same library carries the dealiaser's coarse VAD search.  A
    # library built before that entry point existed still serves every
    # interpolation call, so this is a note on the line and not a second
    # verdict -- but it is the difference between a radar volume
    # dealiased in seconds and one dealiased in half a minute, and a
    # user who cannot see which one they have cannot ask why.  It is a
    # note on BOTH verdicts below: the indexed-donor entry point and the
    # VAD search are separate additions to the same library, and a
    # reader told about one of them still has to be told about the other.
    from gpuwm.obs.coarse_cost import unavailable_reason

    reason = unavailable_reason()
    search = ("radar coarse VAD search: native"
              if reason is None else
              f"radar coarse VAD search: NumPy ({reason})")
    if not indexed:
        # A staged library can be older than the checkout driving it, and
        # the ABI integer cannot say so: it describes the calls that
        # already existed, and this one is an addition.  Absence is not a
        # fault -- the projected route keeps its NumPy mirror and warns
        # at the first plan -- but it is a whole preparation stage's
        # worth of wall clock, so the estate report says it rather than
        # calling the library simply "verified".
        return Check(
            "cpu preprocess library", "info",
            f"{path} loaded via ctypes, ABI v{abi}, but without "
            "gpuwm_indexed_interp_f32: projected-source horizontal "
            "mapping falls back to the single-core NumPy mirror, which "
            f"dominates nested preparation; {search}",
            "# it has to be rebuilt from this checkout:\n" + remedy,
            action=_build_action(),
            brief="ABI v{0}, projected mapping on the NumPy mirror".format(
                abi),
            group=_GROUP_BRIDGES)
    return Check("cpu preprocess library", "verified",
                 f"{path} loaded via ctypes, ABI v{abi}, "
                 f"indexed-donor horizontal entry present; {search}",
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


def _arbitrary_input_check() -> Check:
    """Can this install ingest a data source nobody blessed in advance?

    The whole reason this check exists: `gpuwm adapt` was a working
    arbitrary-input front door that no user could find and, on a wheel
    install, could not run.  A feature nobody can reach is not shipped, so
    doctor states the surface rather than leaving it to be discovered.
    """

    from gpuwm.adapt import _SUPPORTED_FORMATS

    formats = list(_SUPPORTED_FORMATS)
    # NetCDF authoring is reachable when the RUST decoder is, not when
    # netCDF4 is.  Probing the Python library here was correct while
    # `adapt` read NetCDF through `netCDF4.Dataset`; since the decode
    # moved to `rw_netcdf` the old probe was wrong in both directions --
    # it reported the door shut on an install that can open it, and open
    # on an install whose bridge is unstaged.  `_netcdf_decoder_check`
    # is the blocking verdict on that executable; this check states what
    # the door does with it.
    try:
        from gpuwm import netcdf_bridge
        netcdf_decoder = netcdf_bridge.find_netcdf_bin()
    except (ImportError, FileNotFoundError) as error:
        return Check(
            "arbitrary input", "missing",
            f"NetCDF authoring is unreachable: the Rust NetCDF decoder is "
            f"not usable ({error}); `gpuwm adapt --descriptor ...` cannot "
            "read a NetCDF source without it",
            REINSTALL_HINT, action="pip install -e .",
            brief=_short(f"rw_netcdf unusable: {error}"))
    # Both formats need a decoder now.  Report the split rather than one
    # verdict, because a missing GRIB2 decoder does not stop a NetCDF
    # source from running, and vice versa.
    missing = []
    for name in ("grib2_inventory", "grib2_dump"):
        try:
            if bridges.find_bridge(name) is None:
                missing.append(name)
        except FileNotFoundError:
            missing.append(name)
    if netcdf_decoder is None:
        grib_state = (
            f"GRIB2 authoring is unavailable too ({', '.join(missing)} not "
            "resolvable)." if missing else "GRIB2 authoring is unaffected."
        )
        return Check(
            "arbitrary input", "missing",
            "NetCDF authoring is unreachable: rw_netcdf is not staged, and "
            f"gpuwm keeps no Python NetCDF decoder behind it. {grib_state} "
            "The format contract is docs/intermediate-format.md.",
            netcdf_bridge.netcdf_remedy(),
            action=_build_action(bridges.RUSTWX_CRATE_RELATIVE),
            brief="rw_netcdf not staged")
    detail = (
        "`gpuwm adapt` authors arbitrary sources from your own descriptor, "
        f"no source name required; formats: {', '.join(formats)}. NetCDF is "
        f"decoded by rw_netcdf ({netcdf_decoder}). "
        "The format contract is docs/intermediate-format.md."
    )
    if missing:
        return Check(
            "arbitrary input", "present",
            detail + f" GRIB2 authoring is unavailable: {', '.join(missing)} "
            "not resolvable; NetCDF authoring is unaffected.",
            bridges.bridge_remedy(missing[0]),
            action="gpuwm fetch-bridges",
            brief=f"netcdf ready; grib2 needs {', '.join(missing)}",
            blocking=False)
    return Check(
        "arbitrary input", "verified", detail,
        brief=f"gpuwm adapt authors {', '.join(formats)} from a descriptor")


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
#:
#: DERIVED from the two declarations that between them name every route
#: this build serves, never transcribed.  It read ``SOURCE_DECODERS``
#: alone, which is the three routes with a dedicated Rust decoder, so
#: ``--source`` accepted era5/gfs/hrrr and refused ``hrrr-prs`` -- a
#: route ``gpuwm fetch`` and ``gpuwm prep`` both serve, and the one the
#: persona walk actually used (UX finding N17).  The registry is the
#: other half: a model joins this list by having a row, which is the
#: same act that makes it reachable from every other door.
#:
#: Resolved LAZILY, through the module ``__getattr__`` at the bottom of
#: this file.  ``gpuwm.doctor`` must import on an interpreter that has
#: no scientific stack -- that is what a reader runs when the install is
#: broken, and a pinned test proves it -- and the source registry pulls
#: numpy in behind its coverage windows.  Code inside this module calls
#: :func:`_doctor_sources`; everything outside reads the name.
def _doctor_sources() -> tuple[str, ...]:
    """Every route name ``--source`` accepts, resolved once."""

    from gpuwm import source_adapters

    return tuple(sorted(
        set(bridges.SOURCE_DECODERS)
        | {adapter.source_id
           for adapter in source_adapters.source_adapters()}))

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


def _registry_transport(source: str) -> str:
    """What ``gpuwm fetch --source SOURCE`` would move, in one clause."""

    from gpuwm import fetch_routes

    try:
        route = fetch_routes.route_for(source)
    except ValueError as error:
        return str(error).splitlines()[0]
    hosts = ", ".join(host.name for host in route.hosts)
    return (f"fetch takes whole objects from {hosts} "
            f"({route.label}) through the packaged route table")


def _registry_route_check(source: str, engine: Check) -> Check:
    """One line for a route with no decoder executable of its own.

    A route without a dedicated binary is not a route without an
    answer: the registry row says whether an implementation exists, the
    packaged profile says WHAT the preparation decodes it through, and
    the fetch route table says whether ``gpuwm fetch`` can go and get
    it.  All three are table lookups, so the twenty-eight routes this
    added to the default estate cost no probes -- ``engine`` is the one
    real resolution, made once by the caller and shared across them.

    ONE line rather than the decoder/transport pair the three
    executable-backed routes print, because both halves are one table
    read here and a second line would double the length of the default
    report to repeat a lookup.

    The engine's own verdict stays on the engine's own line.  This one
    DEFERS to it (``info``, never ``missing``) so that one unbuilt
    binary produces one gap with one remedy instead of twenty-eight
    copies of the same gap -- which is the shape that makes a report
    unreadable rather than more complete.
    """

    from gpuwm.source_adapters import get_source_adapter

    adapter = get_source_adapter(source)
    name = f"{source} route"
    transport = _registry_transport(source)
    if not adapter.runnable:
        return Check(
            name, "info",
            "declared in the registry with no strict implementation "
            f"route on this release (status {adapter.status.value}); "
            f"`gpuwm prep --show-source {source}` prints the row and "
            f"what it would need.  {transport}",
            brief="declared, no implementation route", group=_GROUP_ROUTE,
            blocking=False)
    if adapter.packaged_profile is None:
        return Check(
            name, "info",
            f"decoded through a mapping the caller supplies (runner "
            f"{adapter.runner}), so there is no packaged profile for "
            f"this report to resolve.  {transport}",
            brief="caller-supplied mapping", group=_GROUP_ROUTE,
            blocking=False)
    if engine.status == "verified":
        return Check(
            name, "verified",
            f"packaged profile {adapter.packaged_profile}, decoded on "
            f"{engine.brief or 'the mapped engine'}.  {transport}",
            brief=_short(adapter.packaged_profile), group=_GROUP_ROUTE)
    return Check(
        name, "info",
        f"packaged profile {adapter.packaged_profile}, decoded on the "
        "mapped engine -- which this estate has not resolved; see the "
        f"`{engine.name}` line for the verdict and the remedy.  "
        f"{transport}",
        engine.remedy, action=engine.action,
        brief="mapped engine unresolved; see the engine line",
        group=_GROUP_ROUTE, blocking=False)


def _source_route_checks(source: str, engine: Check | None = None
                         ) -> list[Check]:
    """Everything one data route resolves before it reads a byte.

    ``engine`` is the mapped decode engine's own check, resolved once
    per report by :func:`collect_checks` and shared across every
    registry route; omitted, it is resolved here, which is what a
    caller asking about one route wants.
    """

    known = _doctor_sources()
    if source not in known:
        raise ValueError(f"unknown doctor source {source!r}; known: "
                         f"{list(known)}")
    if source not in bridges.SOURCE_DECODERS:
        return [_registry_route_check(
            source,
            engine if engine is not None else _mapped_engine_check())]
    checks = [_decoder_route_check(source)]
    if source == "hrrr":
        checks.append(_hrrr_fetch_path_check())
    if source == "gfs":
        checks.append(_gfs_fetch_path_check())
    if source == "era5":
        # No transport line, and that is the finding rather than an
        # omission: this route has no gpuwm-driven download at all.
        # `gpuwm fetch --source era5` writes a CDS request document for
        # the user to submit; what doctor CAN answer is which decoder
        # the preparation will launch, which is the line above.
        checks.append(Check(
            "era5 route fetch transport", "info",
            "no gpuwm transport: `gpuwm fetch --source era5` writes a "
            "CDS request document and the retrieval happens at the CDS, "
            "so there is no byte transport here to report on",
            brief="CDS request document; no gpuwm transport",
            group=_GROUP_ROUTE))
    return checks


def _default_route_order() -> tuple[str, ...]:
    """Every route, ordered so the terse report can fold the tail.

    Three blocks, and the order is a stated grouping rather than a
    sort of the findings: the routes with a decoder executable of their
    own (which print a decoder line and a transport line each), then
    the registry routes that have an implementation, then the registry
    rows that are declared and not yet runnable.  Adjacent same-verdict
    lines fold, so the twenty-eight routes `--source` gained do not
    turn the default report into a page of alternating one-word
    verdicts; every one of them is still a named line under
    ``--explain`` and in ``--json``, and `--source NAME` prints its own.
    """

    from gpuwm.source_adapters import get_source_adapter

    executable = sorted(bridges.SOURCE_DECODERS)
    rest = [source for source in _doctor_sources()
            if source not in bridges.SOURCE_DECODERS]

    def rank(source: str) -> tuple[int, str]:
        adapter = get_source_adapter(source)
        if not adapter.runnable:
            return (2, source)
        # Runnable through a packaged profile, then runnable only
        # through a mapping the caller brings: three tiers, so each one
        # folds to a line instead of interrupting its neighbour's run.
        return (0 if adapter.packaged_profile is not None else 1, source)

    return tuple(executable) + tuple(sorted(rest, key=rank))


def collect_checks(sources: tuple[str, ...] | None = None) -> list[Check]:
    """The estate, plus every named data route's own resolution.

    ``sources=None`` means every route this build knows, which is what
    a bare ``gpuwm doctor`` runs: the report that said "no gaps" before
    a route died on a path it had never resolved is the reason these
    are in the default estate rather than behind a flag.
    """

    selected = (_default_route_order() if sources is None
                else tuple(sources))
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
    # Between the wheel and the solver, because that is the order the
    # three fail in: a wheel that will not load, then a wheel that
    # loads but cannot compile, then a compile that cannot factor.
    checks.append(_cuda_headers_check())
    checks.append(_da_eigensolver_check())
    # Beside the extras, because they are the same question asked of a
    # different feature: can this install actually run the thing its
    # documentation describes?  This one keeps its own line because
    # rasterio/pyproj stopped being an extra in 2.3.3 -- they are
    # runtime dependencies now, and the extras block below defers to
    # this check rather than probing them a second time.
    checks.append(_geog_stack_check())
    # Every extra this INSTALL declares, in the packaging's own order.
    # Through 2.3.2 the only Python-package line here was a single
    # "render extra" that probed wrf and matplotlib, so `obs`,
    # `dealias` and pyshp were absent from a 40 KB report.
    checks.extend(_extras_checks())
    checks.append(_rust_renderer_check())
    checks.append(_renderer_tree_check())
    checks.append(_fetch_backbone_check())
    checks.append(_nexrad_front_door_check())
    # The three front doors no bundle carries.  Absent from this report
    # through 2.3.3, which is how a box with every radar-adjacent
    # observation route dead could print a fully green estate.
    checks.extend(_obs_front_door_checks())
    checks.append(_netcdf_decoder_check())
    checks.append(_mapped_engine_check())
    checks.append(_ncwrite_check())
    checks.append(_static_builder_check())
    checks.append(_obs_regrid_check())
    checks.append(_region_dealias_check())
    checks.extend(_bridge_checks())
    # ABOUT the report, not about an artifact: does every artifact the
    # bundle carries have a line above?  Twice it did not, and both
    # times doctor called the estate green.
    checks.extend(_bundle_coverage_checks())
    # After the bridges, because it is the question the five `ok bridge`
    # lines above do NOT answer: whether the route that needs two of
    # them can actually reach them.
    checks.append(_mapped_grib2_route_check())
    checks.append(_cpu_library_check())
    checks.append(_thompson_tables_check())
    checks.append(_noah_tables_check())
    checks.append(_arbitrary_input_check())
    checks.extend(_case_data_root_check())
    checks.append(_distribution_manifest_check())
    checks.append(_provenance_check())
    checks.append(_install_identity_check())
    checks.append(_non_git_import_check())
    # Resolved ONCE for every registry route below.  Twenty-eight of
    # the routes decode on the same mapped engine, and asking that
    # binary the same question twenty-eight times would put the whole
    # expansion of `--source` on doctor's wall clock.
    engine = _mapped_engine_check()
    for source in selected:
        checks.extend(_source_route_checks(source, engine))
    return checks


#: Column the first remedy line starts at: ten spaces of gutter plus
#: the ``remedy: `` label itself.  Continuation lines match it.
_REMEDY_LABEL = "          remedy: "

#: Status -> the label both reports print.  One vocabulary, so a reader
#: who runs the terse form and then ``--explain`` is reading the same
#: four words in the same column.
#: ``UNTESTED`` is one character wider than its four siblings and stays
#: that way on purpose: it is the status that must not be skimmed past
#: as though it were ``present``.
_LABELS = {"verified": "ok     ", "present": "present",
           "untested": "UNTESTED", "missing": "MISSING", "info": "info   "}


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
            + _severity_clause(checks)
            + (f"; the other {opt_in} is/are pieces this install has not "
               "staged or installed, each with its own command above"
               if opt_in else "")
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

    :data:`_SCATTERED_FOLD_GROUPS` is the one exception, and it is
    narrow: the extras block declares aliases in packaging order, which
    interleaves them with the real extras, so a walk read three
    consecutive ``info pip extra aliases (3)`` lines as three findings
    (UX finding N24).  A group listed there folds into its FIRST
    appearance wherever its members land, because every line in it says
    the same nothing -- the alias resolves, there is no gap and no
    command.  Every member still survives by name in ``--explain`` and
    ``--json``.
    """

    folded: list[tuple[Check, int]] = []
    scattered: dict[tuple, int] = {}
    for check in checks:
        if check.group in _SCATTERED_FOLD_GROUPS:
            key = (check.group, check.status, check.action)
            index = scattered.get(key)
            if index is not None:
                first, count = folded[index]
                folded[index] = (first, count + 1)
                continue
            scattered[key] = len(folded)
            folded.append((check, 1))
            continue
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
                   f"{1 if blocking else 0})"
                   + _severity_clause(checks) + ".")
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
    """The gaps that justify a nonzero exit.  THE severity rule, once.

    A finding is one of four severities, and exactly two of them fail
    the process:

    ``broken`` (blocks)
        Something is present and wrong: a truncated executable, a table
        that fails its hash, a manifest artifact that fails
        revalidation, an installed package that will not import, a CuPy
        wheel whose cuBLAS will not load on this box.  Nobody chose
        this, and nothing downstream can work around it.

    ``unreachable`` (blocks)
        A command the product's own README or FIRST-LIGHT tells a user
        to run cannot run at all on this box, and no documented
        fallback covers it.  CuPy absent is the type case: ``gpuwm
        run`` is step 5 of First light, and without CuPy it dies in a
        raw ``ModuleNotFoundError`` after fetching gigabytes.  The
        ``[render]`` extra absent is the other: ``gpuwm enprod`` and
        ``gpuwm render --engine matplotlib`` are both README doors and
        both import the ``wrf`` package.

    ``degraded`` (does not block)
        An optional mode, engine or door is unavailable while a
        DOCUMENTED default still works: scipy's ``vad-region``
        dealiasing beside the shipped ``region-global`` engine, the
        rust fetch backbone beside the Python transport, cuSOLVER
        beside the bundled Jacobi kernel.

    ``opt-in`` (does not block)
        An explicit choice the operator has not made: the ~16 GB
        WPS_GEOG download, the ``dev`` extra.

    Two boundary cases are worth stating because they were argued.  A
    door that is genuinely unreachable but lives only in the CLI
    reference or the examples -- ``gpuwm-mapped-inspect``, ``python -m
    tools.obs_fetch_mrms`` -- is reported ``unreachable`` and does NOT
    block: the exit code tracks the paths the product's own front page
    promises, and widening it to every module door would fail every
    install for routes most users never open.  And a ``present`` or
    ``untested`` finding never blocks, because neither is a claim that
    something is wrong.

    What changed on 2026-08-14, and why: CuPy's absence used to carry
    ``blocking=False``.  A bare install therefore printed ``3 gap(s), 0
    of them blocking (exit 0)`` -- a green light over a box that could
    not run a forecast -- and ``gpuwm run-plan --probe`` inherited it
    as ``"ready": true``.  An installer script that trusts an exit code
    shipped that box.  The 1.3.1 -> 1.4.0 ``gpuwm setup`` regression
    this function was originally written for is still handled, by the
    ``opt-in`` severity: an unfetched WPS_GEOG tree still exits 0.
    """

    return [check for check in checks
            if check.status == "missing" and check.blocking]


def severity_census(checks: list[Check]) -> dict[str, int]:
    """How many gaps of each severity, in the report's own vocabulary.

    Printed in the summary because "how many" and "how bad" are
    different questions, and a single number answering both is what let
    ``0 of them blocking`` sit over a broken box.
    """

    census: dict[str, int] = {}
    for check in checks:
        # Every gap carries a severity: Check.__post_init__ translates
        # a legacy `blocking` flag into one, so there is no unclassified
        # bucket to hide a finding in.
        if check.status != "missing" or check.severity is None:
            continue
        census[check.severity] = census.get(check.severity, 0) + 1
    return census


def _severity_clause(checks: list[Check]) -> str:
    """``" -- 1 unreachable, 2 opt-in"``, or ``""`` when there are none."""

    census = severity_census(checks)
    ordered = [severity for severity in
               (SEVERITY_BROKEN, SEVERITY_UNREACHABLE, SEVERITY_DEGRADED,
                SEVERITY_OPT_IN) if census.get(severity)]
    if not ordered:
        return ""
    return " -- " + ", ".join(f"{census[severity]} {severity}"
                              for severity in ordered)


#: Where doctor remembers which version it last reported on, so that
#: the NEXT run after an upgrade can say what moved.  Beside the bridges
#: and the staged tables, in the directory this product already owns;
#: the environment variable exists so a test -- and an operator with a
#: read-only home -- can put it somewhere else.
DOCTOR_STATE_ENV = "GPUWM_DOCTOR_STATE"

#: Schema of that file.  One key today; named so a later key can be
#: added without a reader's file being unreadable.
DOCTOR_STATE_SCHEMA = "gpuwm-doctor-state-v1"


def state_path() -> Path | None:
    """The state file's path, or ``None`` when there is nowhere to put it."""

    override = os.environ.get(DOCTOR_STATE_ENV)
    if override:
        return Path(override)
    try:
        return Path.home() / ".gpuwm" / "doctor-state.json"
    except (RuntimeError, OSError):              # pragma: no cover - no home
        return None


def upgrade_note(installed: str | None, path: Path | None = None
                 ) -> str | None:
    """What changed since the version doctor last reported here.

    Returns the block to print, or ``None`` -- for a first run (nothing
    recorded is not a change), for an unchanged version, for a
    downgrade, and for any release with nothing user-visible to say.
    Either way the current version is recorded, so the note is said
    ONCE: an upgrade announcement that repeats at every doctor run is
    noise the reader learns to skip past, which is how the announcement
    stops working.

    Never raises.  This is a courtesy line on a diagnostic command, and
    a state file that cannot be read or written must not cost a reader
    the report they actually asked for.
    """

    from gpuwm import whats_changed

    path = state_path() if path is None else Path(path)
    if path is None or not installed:
        return None
    previous = None
    try:
        if path.is_file():
            recorded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(recorded, dict):
                previous = recorded.get("version")
    except (OSError, ValueError):
        previous = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"schema": DOCTOR_STATE_SCHEMA,
                        "version": str(installed)}, indent=2) + "\n",
            encoding="utf-8")
    except OSError:                              # pragma: no cover - read-only
        pass
    lines = whats_changed.since(previous, str(installed))
    if not lines:
        return None
    block = [f"gpuwm doctor: this install is {installed}; the last report "
             f"here was from {previous}.  What changed for you:"]
    block += [f"  * {line}" for line in lines]
    block.append("  (said once -- this install is now the recorded one.)")
    return "\n".join(block)


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
    # AFTER the identity line, because the note is about that install,
    # and BEFORE the findings, because "the run folder moved" reframes
    # every path in the report under it.  Only on the human path: the
    # JSON door is a machine surface and gets no prose, and leaving its
    # state alone keeps the note for the reader who will read it.
    note = _upgrade_note_for_this_install()
    if note is not None:
        print(note)
    print(format_report(checks) if explain_enabled(args)
          else format_brief(checks))
    return 1 if blocking_gaps(checks) else 0


def _upgrade_note_for_this_install() -> str | None:
    """:func:`upgrade_note` for whatever gpuwm is running.  Never fails."""

    try:
        from gpuwm.version_cli import install_shape

        return upgrade_note(install_shape().get("version"))
    except Exception:                                   # noqa: BLE001
        return None


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "doctor",
        help="verify the runtime estate for real (a subprocess import of "
             "every package this install's own metadata declares, base "
             "and extras alike; probe executions of every bridge, front "
             "door and render engine; a ctypes load of the CPU library; "
             "table hash/parse validation; sealed-manifest re-hashing; "
             "each data route's decoder and byte transport; WPS_GEOG "
             "index files) and print one line per item with the command "
             "that closes each gap (--explain for the full remedies).  "
             "Exits 1 when a finding is broken or unreachable, 0 when "
             "the only gaps are degraded or opt-in")
    parser.add_argument("--json", action="store_true",
                        help="emit the checks as JSON")
    parser.add_argument(
        "--source", action="append", choices=_doctor_sources(),
        metavar="SOURCE",
        help="report only this data route's own resolution (repeatable) "
             "alongside the shared estate: what its preparation will "
             "decode with, and the byte transport its fetch will use.  "
             "The choices are the source registry -- the same list "
             "`gpuwm fetch` and `gpuwm prep` take.  Omitted, every route "
             "this build knows is reported")
    parser.set_defaults(func=doctor_main)
    return parser


def __getattr__(name: str):
    """``doctor.DOCTOR_SOURCES``, resolved on first read and cached.

    PEP 562, for one name and one reason: the registry that answers it
    imports numpy, and this module has a pinned promise that it imports
    on an interpreter with no scientific stack -- because a reader with
    a broken install runs `gpuwm doctor` FIRST, and a diagnostic that
    cannot import on the estate it diagnoses is worse than no
    diagnostic.  Callers inside this file use ``_doctor_sources()``;
    a module-level ``__getattr__`` does not answer a module's own
    global lookups.
    """

    if name == "DOCTOR_SOURCES":
        value = _doctor_sources()
        globals()["DOCTOR_SOURCES"] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Check", "DOCTOR_SOURCES", "DOCTOR_STATE_ENV",
           "DOCTOR_STATE_SCHEMA",
           "SETUP_ACTIONS", "SEVERITY_BROKEN", "SEVERITY_DEGRADED",
           "SEVERITY_OPT_IN", "SEVERITY_UNREACHABLE", "blocking_gaps",
           "collect_checks", "declared_requirements", "doctor_main",
           "format_brief", "format_report", "geography_gaps",
           "register_cli", "severity_census", "state_path",
           "upgrade_note"]
