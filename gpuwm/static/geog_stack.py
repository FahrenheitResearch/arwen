"""One source of truth for the high-resolution geography engine.

What turns fetched source tiles into model terrain -- the mosaic, the
void fill, the clip, the CRS construction, the reprojection onto the WPS
spherical grid and the area average -- is the Rust ``static-fields``
library, on the bare default of ``[static.highres] enabled = true``.  It
is a bundled artifact (``gpuwm.bridge_assets.BUNDLED_ARTIFACTS``), staged
into a wheel install exactly as the NetCDF writer is, and ``gpuwm
doctor`` reports it as the ``static builder`` row.

``rasterio`` and ``pyproj`` are the PARITY REFERENCE for that library and
the body of its one explicit workaround, ``GPUWM_STATIC_PYTHON=1``.  They
are not what a default run reads, so they are not runtime dependencies:
they live in the ``[geog]`` extra.  A run that uses them prints a
WORKAROUND line and stamps ``static_compute`` on its receipt.

The history this module exists to prevent has not changed shape, only
subject.  Through 2.3.2 the geography stack was an optional extra nobody
named, and the only check for it sat *after* the tile fetch, deep inside
the mosaic step: following the published terrain doc on a documented
install downloaded 160.7 MiB of Copernicus tiles and then died on an
import, at exit 1, with a traceback and no remedy.  So two things hold
here regardless of which engine is in play:

- the probe is **cheap and callable before any network work**, so the
  front door can refuse in advance of the fetch rather than after it;
- the remedy is **one string, written once**, so the refusal, the deep
  guards and ``gpuwm doctor`` cannot drift into naming different
  commands for the same missing piece.

Nothing here imports rasterio, pyproj or ctypes at module scope: this is
imported by ``gpuwm doctor`` on boxes that may have none of them, and a
doctor that cannot run without the thing it diagnoses is useless.
"""
from __future__ import annotations

from importlib.util import find_spec

#: The libraries the pure-Python high-resolution FALLBACK cannot run
#: without, paired with what each one does there, so a refusal can say
#: why the missing piece is needed instead of only that it is missing.
GEOG_MODULES: tuple[tuple[str, str], ...] = (
    ("rasterio", "reads and mosaics the source DEM tiles"),
    ("pyproj", "projects them onto the model's map projection"),
)

#: Distribution names to install, in pyproject order.  Kept beside the
#: module names because they are not the same string in the general case
#: and a remedy naming an import name is a remedy that does not work.
#:
#: Deliberately UNVERSIONED.  The floors live in pyproject, which is the
#: thing that enforces them; repeating them here would put ``>=`` into a
#: printed remedy, and ``>`` is how this project's remedies mark a
#: placeholder a reader must substitute (``<tile>``, ``<cache>``).  A
#: remedy that looks half-placeholder is a remedy nobody pastes.
GEOG_DISTRIBUTIONS: tuple[str, ...] = ("rasterio", "pyproj")

#: The extra those distributions ship in now that the default engine is
#: the Rust library.  Named in one place so a remedy cannot invent a
#: spelling pyproject does not declare.
GEOG_EXTRA: str = "geog"


def missing_geog_modules() -> tuple[str, ...]:
    """Names of the fallback's geography modules that cannot be imported.

    Uses :func:`importlib.util.find_spec` rather than a real import: this
    runs on the front-door path of every enabled high-resolution build and
    inside ``gpuwm doctor``, and importing rasterio drags GDAL into the
    process for what is only a presence question.

    A module that is installed but broken (a GDAL or PROJ shared library
    that will not load) still passes this probe and fails later at the
    real import.  That is deliberate and documented rather than papered
    over: distinguishing the two costs the full import here, and the
    later failure names the loader error, which is the useful half.
    """
    missing = []
    for name, _role in GEOG_MODULES:
        try:
            found = find_spec(name) is not None
        except (ImportError, ValueError):
            # A broken or partially-removed distribution can leave a
            # finder that raises instead of returning None.  Absent for
            # our purposes: the remedy is the same reinstall either way.
            found = False
        if not found:
            missing.append(name)
    return tuple(missing)


def geog_unavailable_detail(missing: tuple[str, ...] = ()) -> str:
    """The single remedy string for a fallback that cannot run.

    ``missing`` empty means "say it generically" -- the deep guards call
    it that way, because by the time an import actually fails they know
    only that one did.

    This is reached only on the pure-Python path: either
    ``GPUWM_STATIC_PYTHON=1`` was set, or the Rust library would not
    load and the numpy bodies were the last resort.  So the remedy names
    BOTH ways out, default first, because reinstalling the library the
    product actually ships on is the better answer nearly every time.
    """
    roles = dict(GEOG_MODULES)
    if missing:
        named = ", ".join(
            f"{name} ({roles.get(name, 'required')})" for name in missing)
        head = (f"the pure-Python high-resolution fallback needs {named}, "
                "which this environment cannot import")
    else:
        head = ("the pure-Python high-resolution fallback needs rasterio "
                "and pyproj, which this environment cannot import")
    return (
        head
        + ".  The DEFAULT engine is the Rust static-fields library, which "
        "needs neither; you are on the fallback because "
        "GPUWM_STATIC_PYTHON=1 is set or that library could not be "
        "loaded.\n"
        "  remedy: unset GPUWM_STATIC_PYTHON and stage the library:\n"
        "    gpuwm fetch-bridges\n"
        "  #   ... or, to run the fallback anyway, install its libraries:\n"
        f"  #   pip install --upgrade 'gpuwm[{GEOG_EXTRA}]'\n"
        "  #   ... or just those two:\n"
        "  #   pip install --upgrade " + " ".join(GEOG_DISTRIBUTIONS) + "\n"
        "  # `gpuwm doctor` reports the default engine as 'static builder' "
        "and\n"
        "  # this fallback's libraries as 'geography stack'.")


def missing_highres_engine() -> str | None:
    """Why NO high-resolution engine can run here, or ``None``.

    The concrete breakage: a high-resolution build fetches up to 160.7
    MiB of source tiles before it warps anything, so an environment that
    can warp with neither engine must be told before the fetch, not
    after it.  That is the whole reason this is a separate, cheap,
    network-free probe.

    An environment is fine if EITHER engine can run: the Rust library
    (the default) or, when the caller has explicitly selected the
    fallback or the library will not load, rasterio plus pyproj.  It is
    also fine to have only the Rust library, which is the shipped
    configuration.
    """
    from . import rust_bridge

    selected_fallback = rust_bridge.python_fallback_requested()
    library_reason = rust_bridge.unavailable_reason()
    if library_reason is None and not selected_fallback:
        return None
    missing = missing_geog_modules()
    if not missing:
        return None
    if selected_fallback:
        return (
            f"{rust_bridge.STATIC_PYTHON_ENV}=1 selects the pure-Python "
            "high-resolution fallback, and "
            + geog_unavailable_detail(missing))
    return (
        "the Rust static-fields library is not loadable "
        f"({library_reason}), and "
        + geog_unavailable_detail(missing))
