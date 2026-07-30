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
cargo builds, no CUDA context (importing CuPy allocates no device), no
network.  Every gap prints a remedy whose every line is either a
command that runs as printed in this platform's own shell or a ``#``
comment -- never prose fused onto a command -- instead of letting the
user meet a raw traceback three commands later.

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
from gpuwm import rustwx_fetch

# The pip extras exactly as the README installs them.
#
# Each hint is a command first and explanation second, on separate
# lines, because a remedy is read under stress and pasted whole: the
# parenthetical that used to trail the command on the same line --
# `pip install 'gpuwm[gpu]'   (or: pip install cupy-cuda12x)` -- is a
# shell error the moment anyone does the obvious thing with it.
GPU_EXTRA_HINT = ("pip install 'gpuwm[gpu]'\n"
                  "  # or, without the extra: pip install cupy-cuda12x")
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
    results = {"wrf-rust": _import_probe("wrf", "wrf-rust"),
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
                f"# {bridges.BRIDGE_ENV[name]} names a missing "
                "executable: point it at a real build, or unset it "
                "and build one --\n"
                + bridges.bridge_remedy(name)))
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
                    f"bridge {name}", "verified", f"{found} -- {evidence}"))
            else:
                checks.append(Check(
                    f"bridge {name}", "missing", f"{found} -- {evidence}",
                    f"# this one has to be replaced -- needed by: "
                    f"{consumer}\n"
                    + bridges.install_aware_build_hint(
                        bridges.CARGO_BUILD_HINT)))
        elif crate.is_file():
            checks.append(Check(
                f"bridge {name}", "missing",
                f"not built yet (checkout crate: {crate.parent})",
                f"# needed by: {consumer}\n{bridges.CARGO_BUILD_HINT}"))
        else:
            checks.append(Check(
                f"bridge {name}", "missing",
                "no checkout crate and no prebuilt executable "
                f"(searched {', '.join(str(c) for c in bridges.bridge_candidates(name))})",
                bridges.bridge_remedy(name)
                + f"\n  # needed by: {consumer}"))
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
            + rustwx_fetch.fetch_remedy())
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
            "--mode full-file")
    ok, evidence = rustwx_fetch.probe_fetch_bin(found)
    if not ok:
        return Check(
            name, "missing", f"{found} -- {evidence}",
            "# it has to be replaced:\n" + bridges.install_aware_build_hint(
                rustwx_fetch.CARGO_BUILD_HINT, "tools/rustwx"))
    return Check(name, "verified", f"{found} -- {evidence}")


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
            + rustwx.renderer_remedy())
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
            + "\n  # enables --engine rust and makes it the default")
    ok, evidence = rustwx.probe_renderer(found)
    if not ok:
        return Check(
            name, "missing", f"{found} -- {evidence}",
            "# it has to be replaced:\n" + bridges.install_aware_build_hint(
                rustwx.CARGO_BUILD_HINT, "tools/rustwx"))
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
    return Check(name, "verified", f"{found} -- {evidence}; {basemap_note}")


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
            f"not): {error}", remedy)
    except (OSError, RuntimeError, AttributeError) as error:
        return Check(
            "cpu preprocess library", "missing",
            f"found but not loadable as ABI v{CPU_BACKEND_ABI}: {error}",
            "# it has to be replaced:\n" + remedy)
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
                "gpuwm fetch-tables\n"
                f"  # one {total_mib:.0f} MiB download, SHA-256 verified "
                "against the\n"
                "  # packaged pins before install; --from stages it from a\n"
                "  # local directory instead, offline")
        return Check(
            "thompson tables", "missing", str(error),
            REINSTALL_HINT
            + "\n  # if GPUWM_THOMPSON_TABLE_ROOT is set, point it at a"
            "\n  # byte-identical mirror of the packaged tables, or unset it")
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
        return [Check(
            "WPS_GEOG", "missing",
            f"{geog} does not exist (the default geog_root).  Nothing "
            "that builds static fields can run without it",
            GEOG_HINT)]
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
                "defaults to ${GPUWM_CASE_DATA_ROOT}/WPS_GEOG"),
            *_geog_tree_checks(default_geog_root()),
        ]
    root = Path(raw)
    if not root.is_dir():
        return [Check(
            "GPUWM_CASE_DATA_ROOT", "missing",
            f"set to {raw} but that directory does not exist",
            "# point GPUWM_CASE_DATA_ROOT at the directory that CONTAINS\n"
            "  # your case bundles and WPS_GEOG -- there is no command to\n"
            "  # print here, only your path")]
    return [
        Check("GPUWM_CASE_DATA_ROOT", "present",
              f"{root} (directory exists; its datasets are checked "
              "individually below)"),
        *_geog_tree_checks(root / "WPS_GEOG"),
    ]


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
            f"# unset {name} unless you are running a sealed runtime\n"
            "  # archive, whose installer sets it correctly")
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
            "# re-extract the sealed runtime archive (its installer wrote\n"
            f"  # this manifest beside its artifacts), or unset {name}")
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
            "# install Python 3.11 or newer -- which installer is right\n"
            "  # here depends on how this Python was installed"))
    checks.append(_cupy_check())
    checks.append(_render_extra_check())
    checks.append(_rust_renderer_check())
    checks.append(_fetch_backbone_check())
    checks.extend(_bridge_checks())
    checks.append(_cpu_library_check())
    checks.append(_thompson_tables_check())
    checks.append(_noah_tables_check())
    checks.extend(_case_data_root_check())
    checks.append(_distribution_manifest_check())
    return checks


#: Column the first remedy line starts at: ten spaces of gutter plus
#: the ``remedy: `` label itself.  Continuation lines match it.
_REMEDY_LABEL = "          remedy: "


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


def format_report(checks: list[Check]) -> str:
    lines = ["gpuwm doctor: runtime estate"]
    labels = {"verified": "ok     ", "present": "present",
              "missing": "MISSING", "info": "info   "}
    for check in checks:
        lines.append(f"  {labels[check.status]} {check.name}: "
                     f"{check.detail}")
        if check.remedy:
            lines.extend(_remedy_block(check.remedy))
    gaps = sum(1 for check in checks if check.status == "missing")
    presence_only = sum(1 for check in checks if check.status == "present")
    if gaps:
        # Not "every remedy is copy-pasteable": a remedy can be a
        # sequence of steps, and on a pip install the bridge remedy is a
        # clone-and-build rather than a single line.  Claiming one line
        # when six were printed is the kind of small lie that costs the
        # reader their trust in the other five.
        lines.append(
            f"gpuwm doctor: {gaps} gap(s).  Every remedy line above is "
            "either a command to run as printed, in the order printed, "
            "or a '#' comment.")
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
