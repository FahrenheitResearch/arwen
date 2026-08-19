"""Every module that DECODES NetCDF does it in Rust, and none quietly doesn't.

The first pass converted one module and left a dozen. This pins the
result of finishing the sweep, and it pins it by PARSING rather than by
grepping -- a grep counts the comment that explains an absence, which is
how a removal can look like a no-op.

Four lists, and the point is that all four are explicit:

* :data:`_DECODERS` must not import a Python NetCDF library. These read
  meteorological field arrays, so they go through ``rw_netcdf``.
* :data:`_PLUMBING_OR_BLOCKED` may. Each entry carries the reason in the
  table below.
* :data:`_WORKAROUND_ENGINE` may, and has to prove the netCDF4 half is an
  ESCAPE: the default engine is Drew's Rust writer and the environment
  variable that reaches netCDF4 is named in the module.
* :data:`_OPEN_FINDINGS` may, and is the honest one: a data path that has
  NOT been converted, named with the finding against it.  It is EMPTY
  today -- F1 (wrfinput/wrfbdy) and F2 (`gpuwm spectral score`) of the
  2026-08-18 hidden-scope audit were both closed, by separate lanes, and
  each module moved to the list its new state belongs in.  The list stays
  because the next unconverted read needs somewhere to be DECLARED rather
  than excused, and an empty list says "none open" out loud.

A module may not join any of them silently: the test fails if a module
appears in none, so a new ``import netCDF4`` anywhere in the package has
to be argued for here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from gpuwm import netcdf_bridge

_PACKAGE = Path(netcdf_bridge.__file__).resolve().parent

#: Modules that decode meteorological data. None may import netCDF4.
_DECODERS = {
    "gpuwm/mapped_source.py",
    "gpuwm/era5_direct.py",
    "gpuwm/obs/goes_grid.py",
    "gpuwm/ingest/nest_init.py",
    "gpuwm/ingest/water_overlay.py",
    "gpuwm/obs/target_grid.py",
    "gpuwm/verify/field_metrics.py",
    "gpuwm/verify/profiles.py",
    "gpuwm/verify/spectral_io.py",
    "gpuwm/verify/t0_state_digest.py",
}

#: Modules that may still import netCDF4, each with the reason.
#:
#: "plumbing"  -- reads only attributes/dimensions/identity.
#:
#: "writes" is NOT a reason here.  It used to be -- the estate said
#: "writing is not decoding" -- and the law says "NetCDF read/write", both
#: words.  A module that WRITES gridded NetCDF belongs in
#: :data:`_WORKAROUND_ENGINE` below, which demands the Rust writer be the
#: default and the netCDF4 one be a named escape.
#: "char"      -- needs character/string data; netcrust exposes no
#:                character read at all (upstream gap 3).
#: "char+attrs" -- "char", plus the file's global attributes, because the
#:                bridge cannot see an HDF5 attribute that landed in an
#:                object-header continuation block and answers None for one
#:                instead of failing (upstream gap 4, measured 2026-08-18:
#:                seven global attributes with one large among them is
#:                enough to spill).  Metadata only; every VALUE still
#:                decodes on the bridge.
#: "bytes"     -- needs the STORED representation; the bridge promotes
#:                every numeric type to f64, so a byte-identity or FP32
#:                digest cannot round-trip through it.
#: "tables"    -- sweeps a STATIC asset table variable by variable (the
#:                RRTMGP k-distribution and cloud-optics files) for a
#:                finiteness screen.  Not meteorological data, and the
#:                bridge costs one process launch plus one full f64 temp
#:                file PER VARIABLE, on a door a user runs before a run.
_PLUMBING_OR_BLOCKED = {
    "gpuwm/certify/pins.py": "plumbing",
    "gpuwm/downscale.py": "plumbing",
    "gpuwm/ensemble/wrfout_inventory.py": "char",
    # The source-orography FIELD read moved to the bridge (audit HIT-5);
    # what is left on netCDF4 here is the RRTMGP asset-table sweep.
    "gpuwm/ingest/preflight.py": "tables",
    # Writes on the Rust classic writer since the 2026-08-18 audit.  What
    # is left is ONE netCDF4 open for the file's metadata: `radar_id` /
    # `radar_valid_time`, and the global attributes, which the bridge
    # silently drops from a legacy HDF5 product (see "char+attrs").
    "gpuwm/obs/radar_grid.py": "char+attrs",
    "gpuwm/offline_child.py": "plumbing",
    "gpuwm/offline_child_run.py": "plumbing",
    "gpuwm/render.py": "plumbing",
    # One global attribute (SIMULATION_START_DATE) off a wrfout, and None
    # on any failure -- it decodes no field, so it is identity plumbing.
    "gpuwm/run_stamp.py": "plumbing",
    "gpuwm/core/rrtmgp.py": "char",
    "gpuwm/verify/cases/real74_d01.py": "plumbing",
    "gpuwm/verify/cases/real74_d02.py": "bytes",
    "gpuwm/verify/cases/real74_n5b.py": "plumbing",
    "gpuwm/verify/cases/real74_n5s.py": "char",
    "gpuwm/verify/obs/cross_reader.py": "char",
    "gpuwm/wrf_backend_parity.py": "bytes",
}

#: Modules whose netCDF4 import IS an announced workaround engine: the
#: DEFAULT writes through Drew's Rust classic writer, and netCDF4 is
#: reachable only by naming an environment variable.
#:
#: This is the category "plumbing" was hiding.  ``module -> the env var
#: that selects the netCDF4 engine``.
_WORKAROUND_ENGINE = {
    "gpuwm/io/wrfout.py": "GPUWM_WRFOUT_WRITER",
    "gpuwm/obs/grid_product.py": "GPUWM_OBS_GRID_WRITER",
    "gpuwm/wrf_direct.py": "GPUWM_WRFINPUT_WRITER",
}

#: Modules with an OPEN, named finding against them: a real data path that
#: has not been converted yet.  Listed so the module is declared as
#: UNCONVERTED rather than excused, and so this suite fails loudly if one
#: appears that nobody has named.
#: Empty is the truthful state as of 2026-08-18: `gpuwm/wrf_direct.py`
#: (audit F1) now writes every wrfinput and wrfbdy field on the Rust
#: classic writer and sits in :data:`_WORKAROUND_ENGINE`, and
#: `gpuwm/verify/spectral_io.py` (audit F2) reads its planes through
#: :mod:`gpuwm.netcdf_bridge` and sits in :data:`_DECODERS`.
_OPEN_FINDINGS: dict[str, str] = {}

_BANNED = {"netCDF4", "xarray", "h5py", "h5netcdf"}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return found


def _relative(path: Path) -> str:
    return path.relative_to(_PACKAGE.parent).as_posix()


def _python_netcdf_importers() -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for path in sorted(_PACKAGE.rglob("*.py")):
        hits = _imports(path) & _BANNED
        if hits:
            result[_relative(path)] = hits
    return result


def test_no_decoder_module_imports_a_python_netcdf_library():
    """The conversions are real: these modules decode through Rust only."""

    offenders = {
        name: sorted(hits)
        for name, hits in _python_netcdf_importers().items()
        if name in _DECODERS
    }
    assert not offenders, (
        "module(s) that decode meteorological NetCDF still import a Python "
        f"NetCDF library: {offenders}")


def test_every_python_netcdf_importer_is_declared_with_a_reason():
    """A new netCDF4 import has to be argued for, not just added."""

    importers = set(_python_netcdf_importers())
    undeclared = sorted(importers - _DECODERS - set(_PLUMBING_OR_BLOCKED)
                        - set(_WORKAROUND_ENGINE)
                        - set(_OPEN_FINDINGS))
    assert not undeclared, (
        f"{len(undeclared)} module(s) import a Python NetCDF library and are "
        f"in none of the lists; classify each as a decoder (convert it), an "
        f"announced workaround engine, or give it a reason in "
        f"_PLUMBING_OR_BLOCKED: {undeclared}")


def test_the_declared_exemptions_have_not_gone_stale():
    """An exemption must not outlive the import it excuses."""

    importers = set(_python_netcdf_importers())
    stale = sorted((set(_PLUMBING_OR_BLOCKED) | set(_WORKAROUND_ENGINE)
                    | set(_OPEN_FINDINGS)) - importers)
    assert not stale, (
        f"{len(stale)} module(s) are exempted but no longer import a Python "
        f"NetCDF library; drop them from _PLUMBING_OR_BLOCKED: {stale}")


def test_the_sweep_actually_converted_something():
    """Counts, not a boolean: the population really did shrink.

    v2.3.3 had 29 importers. If this number climbs back up, the sweep is
    being undone one module at a time.
    """

    importers = _python_netcdf_importers()
    declared = (len(_PLUMBING_OR_BLOCKED) + len(_WORKAROUND_ENGINE)
                + len(_OPEN_FINDINGS))
    assert len(importers) <= declared, (
        f"{len(importers)} modules import a Python NetCDF library; only "
        f"{declared} are declared exempt")
    assert len(importers) < 29, (
        f"{len(importers)} importers is not fewer than the 29 this started "
        f"from")


@pytest.mark.parametrize("module", sorted(_DECODERS))
def test_each_decoder_reaches_the_bridge(module):
    """Each converted module actually references the Rust front door.

    Absence of ``netCDF4`` is only half the claim -- a module that decodes
    nothing at all would also pass that. This asserts the positive.
    """

    path = _PACKAGE.parent / module
    source = path.read_text(encoding="utf-8", errors="replace")
    assert "netcdf_bridge" in source, (
        f"{module} imports no Python NetCDF library, but does not reach "
        f"gpuwm.netcdf_bridge either")


@pytest.mark.parametrize("module", sorted(_WORKAROUND_ENGINE))
def test_a_workaround_engine_module_proves_the_rust_default(module):
    """Its netCDF4 import must be an ESCAPE, not the shipped path.

    "It writes, so it is plumbing" is the sentence that kept two gridded
    observation writers on the C library through a whole release line.  A
    module in this list has to name the environment variable that selects
    netCDF4, resolve to the Rust engine with nothing set, and actually
    reach the Rust writer seam -- otherwise the exemption is just the old
    sentence with a new label.
    """

    path = _PACKAGE.parent / module
    source = path.read_text(encoding="utf-8", errors="replace")
    env = _WORKAROUND_ENGINE[module]
    assert f'"{env}"' in source, (
        f"{module} is exempted as a workaround engine but never names "
        f"{env}, so nothing tells a reader how the netCDF4 half is reached")
    assert '"rust"' in source, (
        f"{module} does not name a 'rust' engine, so it cannot be defaulting "
        f"to one")
    assert "nc_writer_bridge" in source, (
        f"{module} never reaches gpuwm.io.nc_writer_bridge, so its default "
        f"is not the Rust writer at all")


@pytest.mark.parametrize("module,env", sorted(_WORKAROUND_ENGINE.items()))
def test_a_workaround_engine_resolves_to_rust_with_nothing_set(module, env):
    """The default is proved by RUNNING the resolver, not by reading it.

    The test above reads the source; a module can name "rust", name its
    environment variable and still resolve to netCDF4 with nothing set --
    that is a default-on defect a grep cannot see, and "fixed means
    default" is the ruling it would break.  This one imports the module
    and asks its own resolver, with the escape hatch removed from the
    environment.
    """

    import importlib
    import os

    loaded = importlib.import_module(module[:-3].replace("/", "."))
    resolvers = [value for key, value in vars(loaded).items()
                 if key.startswith("resolve_") and key.endswith("_engine")]
    assert resolvers, (
        f"{module} declares {env} but exposes no resolve_*_engine, so "
        f"nothing here can ask it what a bare default does")
    previous = os.environ.pop(env, None)
    try:
        for resolve in resolvers:
            assert resolve() == "rust", (
                f"{module}: with {env} unset the engine must be rust")
    finally:
        if previous is not None:
            os.environ[env] = previous
