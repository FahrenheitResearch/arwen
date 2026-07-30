"""``gpuwm doctor``: verify the runtime estate, print exact remedies.

The pip package deliberately splits its runtime across four estates the
installer cannot see from ``pip install`` alone: the GPU runtime (CuPy),
the render extra (wrf-rust + matplotlib), the compiled Rust bridges
(never shipped in the wheel), and the locally staged data roots
(``WPS_GEOG``/``GPUWM_CASE_DATA_ROOT``).  Doctor checks each one for
real, not by presence: it imports CuPy/wrf/matplotlib in short-lived
subprocesses, probe-executes every bridge executable, loads the CPU
preprocessing library through ctypes and reads its ABI version,
sha256-validates the packaged Thompson tables with the same routine the
model uses at launch, parses the Noah/landuse tables with the model's
own parsers, and requires each WPS_GEOG dataset's ``index`` file.  No
cargo builds, no CUDA context (importing CuPy allocates no device), no
network.  Every gap prints the exact copy-pasteable remedy instead of
letting the user meet a raw traceback three commands later.

Statuses distinguish what was proven: ``verified`` means the deep check
ran and passed; ``present`` is reserved for the few items where nothing
deeper than existence can honestly be checked (and says so); ``missing``
is a gap with a remedy; ``info`` is context.

Exit status: 0 when nothing actionable is missing, 1 otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.util import find_spec
import json
import os
from pathlib import Path
import subprocess
import sys

from gpuwm import bridges
from gpuwm import rustwx

#: The pip extras exactly as the README installs them.
GPU_EXTRA_HINT = "pip install 'gpuwm[gpu]'   (or: pip install cupy-cuda12x)"
RENDER_EXTRA_HINT = ("pip install 'gpuwm[render]'   "
                     "(wrf-rust + matplotlib)")
GEOG_HINT = (
    "download the NCAR WPS geographical data (\"highest resolution "
    "mandatory fields\", ~29 GB unpacked) and place it at "
    "$GPUWM_CASE_DATA_ROOT/WPS_GEOG or pass --geog-root explicitly")
REINSTALL_HINT = ("the installed package is incomplete; reinstall with "
                  "`pip install -e .` from a clone (or a rebuilt wheel)")

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
    """One doctor line: verified/present/MISSING/info plus the remedy."""

    name: str
    status: str  # "verified" | "present" | "missing" | "info"
    detail: str
    remedy: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Python packages: actual imports, in a subprocess
# ---------------------------------------------------------------------------

def _import_probe(module: str) -> tuple[bool, str]:
    """Actually import ``module`` in a subprocess; (ok, evidence).

    ``find_spec`` alone lies green for a package whose install is broken
    (ABI mismatch, missing native dependency, half-removed dist-info).
    The subprocess keeps a failing import from poisoning this process
    and allocates nothing beyond the import itself.
    """

    if find_spec(module) is None:
        return False, "not installed"
    code = (f"import {module} as m; import sys; "
            "sys.stdout.write(str(getattr(m, '__version__', 'imported')))")
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


def _cupy_check() -> Check:
    ok, evidence = _import_probe("cupy")
    if ok:
        return Check("cupy (GPU runtime)", "verified",
                     f"imported in a subprocess (cupy {evidence})")
    if evidence == "not installed":
        return Check(
            "cupy (GPU runtime)", "missing",
            "not installed -- gpuwm check/run and the domain wizard's "
            "sizing estimator need it; fetch/import-namelist/render "
            "do not", GPU_EXTRA_HINT)
    return Check("cupy (GPU runtime)", "missing", evidence, GPU_EXTRA_HINT)


def _render_extra_check() -> Check:
    results = {"wrf-rust": _import_probe("wrf"),
               "matplotlib": _import_probe("matplotlib")}
    broken = {name: evidence for name, (ok, evidence) in results.items()
              if not ok}
    if not broken:
        versions = ", ".join(
            f"{name} {evidence}" for name, (_, evidence) in results.items())
        return Check("render extra (wrf-rust + matplotlib)", "verified",
                     f"imported in subprocesses ({versions})")
    if all(evidence == "not installed" for evidence in broken.values()):
        return Check(
            "render extra (wrf-rust + matplotlib)", "missing",
            f"{' and '.join(sorted(broken))} not installed -- gpuwm "
            "render needs the render extra", RENDER_EXTRA_HINT)
    detail = "; ".join(f"{name}: {evidence}"
                       for name, evidence in sorted(broken.items()))
    return Check("render extra (wrf-rust + matplotlib)", "missing",
                 detail, RENDER_EXTRA_HINT)


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
    """

    try:
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
                f"fix {bridges.BRIDGE_ENV[name]} to name the built "
                "executable, or unset it and "
                + bridges.bridge_remedy(name)))
            continue
        if found is not None:
            ok, evidence = _exec_probe(found)
            if ok:
                checks.append(Check(
                    f"bridge {name}", "verified", f"{found} -- {evidence}"))
            else:
                checks.append(Check(
                    f"bridge {name}", "missing", f"{found} -- {evidence}",
                    f"rebuild it: {bridges.CARGO_BUILD_HINT}   "
                    f"[needed by: {consumer}]"))
        elif crate.is_file():
            checks.append(Check(
                f"bridge {name}", "missing",
                f"not built yet (checkout crate: {crate.parent})",
                f"{bridges.CARGO_BUILD_HINT}   [needed by: {consumer}]"))
        else:
            checks.append(Check(
                f"bridge {name}", "missing",
                "no checkout crate and no prebuilt executable "
                f"(searched {', '.join(str(c) for c in bridges.bridge_candidates(name))})",
                bridges.bridge_remedy(name)
                + f"\n    [needed by: {consumer}]"))
    return checks


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
            f"fix {rustwx.RENDERER_ENV} to name the built executable, "
            "or unset it and " + rustwx.renderer_remedy())
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
            rustwx.CARGO_BUILD_HINT + "   [enables --engine rust and "
            "makes it the default]")
    ok, evidence = rustwx.probe_renderer(found)
    if not ok:
        return Check(
            name, "missing", f"{found} -- {evidence}",
            f"rebuild it: {rustwx.CARGO_BUILD_HINT}")
    basemap = rustwx.basemap_dir()
    if os.environ.get("RUSTWX_BASEMAP_DIR") or os.environ.get(
            "RUSTWX_ASSETS_DIR"):
        basemap_note = "basemaps from the RUSTWX_* environment override"
    elif basemap.is_dir():
        basemap_note = f"basemaps {basemap}"
    else:
        basemap_note = ("NO basemap assets found -- charts render "
                        "without coast/state/county lines; set "
                        "RUSTWX_BASEMAP_DIR to a checkout's "
                        "tools/rustwx/assets/basemap")
    return Check(name, "verified", f"{found} -- {evidence}; {basemap_note}")


def _cpu_library_check() -> Check:
    from gpuwm.ingest.cpu_backend import (
        CPU_BACKEND_ABI, CpuPreprocessBackend)

    remedy = (bridges.CARGO_BUILD_HINT
              + f"  then copy it into {bridges.default_bridge_dir()} or "
              "set GPUWM_CPU_PREPROCESS_BRIDGE")
    try:
        backend = CpuPreprocessBackend()
    except FileNotFoundError as error:
        return Check(
            "cpu preprocess library", "missing",
            "gpuwm_preprocess_cpu shared library not found "
            "(--preprocess-backend cpu needs it; the CUDA backend does "
            f"not): {error}", remedy)
    except (OSError, RuntimeError, AttributeError) as error:
        return Check(
            "cpu preprocess library", "missing",
            f"found but not loadable as ABI v{CPU_BACKEND_ABI}: {error}",
            "rebuild it: " + remedy)
    path, abi = backend.path, backend.abi_version
    backend.close()
    return Check("cpu preprocess library", "verified",
                 f"{path} loaded via ctypes, ABI v{abi}")


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
                f"gpuwm fetch-tables   (one {total_mib:.0f} MiB download, "
                "SHA-256 verified against the packaged pins before "
                "install; --from DIR stages offline)")
        return Check(
            "thompson tables", "missing", str(error),
            REINSTALL_HINT + "; if GPUWM_THOMPSON_TABLE_ROOT is set, "
            "point it at a byte-identical mirror of the packaged tables "
            "or unset it")
    return Check(
        "thompson tables", "verified",
        f"{len(assets)} assets at {root} byte-validated (exact size + "
        f"SHA-256, {sum(asset.bytes for asset in assets):,} B), the same "
        "validation every mp8 run performs at load")


def _noah_tables_check() -> Check:
    try:
        from gpuwm.core.landuse import load_landuse_table
        from gpuwm.core.noah import load_tables

        tables = load_tables()
        landuse = load_landuse_table()
    except Exception as error:  # any parse/read failure is the finding
        return Check("noah tables", "missing",
                     f"packaged tables failed to parse: {error}",
                     REINSTALL_HINT)
    return Check(
        "noah tables", "verified",
        "VEGPARM/SOILPARM/GENPARM parsed by the SOIL_VEG_GEN_PARM "
        f"transcription ({tables.lucats} vegetation / {tables.slcats} "
        f"soil categories); LANDUSE.TBL parsed ({landuse.lucats} "
        "categories)")


# ---------------------------------------------------------------------------
# Data roots
# ---------------------------------------------------------------------------

def _case_data_root_check() -> list[Check]:
    from gpuwm.domain_wizard import GEOG_DATASETS

    raw = os.environ.get("GPUWM_CASE_DATA_ROOT")
    if not raw:
        return [Check(
            "GPUWM_CASE_DATA_ROOT", "info",
            "not set.  Layout when you set it: the root is the directory "
            "that CONTAINS your case bundles and (by default) WPS_GEOG -- "
            "configs reference ${GPUWM_CASE_DATA_ROOT}/<bundle>/... and "
            "geog_root defaults to ${GPUWM_CASE_DATA_ROOT}/WPS_GEOG")]
    root = Path(raw)
    if not root.is_dir():
        return [Check(
            "GPUWM_CASE_DATA_ROOT", "missing",
            f"set to {raw} but that directory does not exist",
            "point it at the directory containing your case bundles "
            "and WPS_GEOG")]
    checks = [Check(
        "GPUWM_CASE_DATA_ROOT", "present",
        f"{root} (directory exists; its datasets are checked "
        "individually below)")]
    geog = root / "WPS_GEOG"
    if not geog.is_dir():
        checks.append(Check(
            "WPS_GEOG", "missing",
            f"{geog} does not exist (the default geog_root)",
            GEOG_HINT))
        return checks
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
            GEOG_HINT))
    else:
        checks.append(Check(
            "WPS_GEOG", "verified",
            f"{geog} carries all {len(GEOG_DATASETS)} required datasets, "
            "each with its WPS index file"))
    return checks


def _distribution_manifest_check() -> Check:
    name = "GPUWM_NATIVE_DISTRIBUTION_MANIFEST"
    raw = os.environ.get(name)
    if not raw:
        return Check(
            name, "info",
            "not set (normal for source clones and pip installs; only "
            "sealed runtime archives set it to bind their decoder "
            "inventory)")
    path = Path(raw)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        payload = None
    if (not isinstance(payload, dict)
            or payload.get("schema") != "gpuwm-native-wrf-runtime-v1"
            or payload.get("status") != "READY"):
        return Check(
            name, "missing",
            f"set to {raw} but not a READY gpuwm-native-wrf-runtime-v1 "
            "document",
            "unset it unless you are running a sealed runtime archive, "
            "whose installer sets it correctly")
    declared = payload.get("payload")
    if not isinstance(declared, dict) or not declared:
        return Check(
            name, "present",
            f"{path}: READY schema, but the manifest declares no "
            "per-artifact hashes, so only its schema and status could be "
            "checked (presence-only)")
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
            "re-extract the sealed runtime archive (its installer wrote "
            "this manifest beside its artifacts), or unset the variable")
    return Check(
        name, "verified",
        f"{path}: READY; all {verified} declared artifacts re-hashed "
        "and match")


def collect_checks() -> list[Check]:
    checks: list[Check] = []
    version = ".".join(str(v) for v in sys.version_info[:3])
    if sys.version_info >= (3, 11):
        checks.append(Check("python", "verified", f"{version} (>= 3.11)"))
    else:
        checks.append(Check(
            "python", "missing", f"{version} is below the 3.11 floor",
            "install Python 3.11 or newer"))
    checks.append(_cupy_check())
    checks.append(_render_extra_check())
    checks.append(_rust_renderer_check())
    checks.extend(_bridge_checks())
    checks.append(_cpu_library_check())
    checks.append(_thompson_tables_check())
    checks.append(_noah_tables_check())
    checks.extend(_case_data_root_check())
    checks.append(_distribution_manifest_check())
    return checks


def format_report(checks: list[Check]) -> str:
    lines = ["gpuwm doctor: runtime estate"]
    labels = {"verified": "ok     ", "present": "present",
              "missing": "MISSING", "info": "info   "}
    for check in checks:
        lines.append(f"  {labels[check.status]} {check.name}: "
                     f"{check.detail}")
        if check.remedy:
            lines.append(f"          remedy: {check.remedy}")
    gaps = sum(1 for check in checks if check.status == "missing")
    presence_only = sum(1 for check in checks if check.status == "present")
    if gaps:
        lines.append(f"gpuwm doctor: {gaps} gap(s); every remedy above "
                     "is copy-pasteable.")
    elif presence_only:
        lines.append(f"gpuwm doctor: no gaps ({presence_only} check(s) "
                     "presence-only as labeled, the rest verified).")
    else:
        lines.append("gpuwm doctor: no gaps; every check verified.")
    return "\n".join(lines)


def doctor_main(args) -> int:
    checks = collect_checks()
    if getattr(args, "json", False):
        print(json.dumps(
            [check.__dict__ for check in checks], indent=2))
    else:
        print(format_report(checks))
    return 1 if any(check.status == "missing" for check in checks) else 0


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "doctor",
        help="verify the runtime estate for real (subprocess imports of "
             "cupy/wrf/matplotlib, bridge probe executions, ctypes load "
             "of the CPU library, table hash/parse validation, WPS_GEOG "
             "index files) and print the exact remedy for each gap")
    parser.add_argument("--json", action="store_true",
                        help="emit the checks as JSON")
    parser.set_defaults(func=doctor_main)
    return parser


__all__ = ["Check", "collect_checks", "doctor_main", "format_report",
           "register_cli"]
