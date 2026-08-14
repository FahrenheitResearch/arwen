"""One source of truth for the high-resolution geography library stack.

``rasterio`` and ``pyproj`` are what turns fetched source tiles into model
terrain: the mosaic, the reprojection onto the WPS spherical grid and the
area average all run through them.  They are runtime dependencies of
``gpuwm`` (pyproject ``dependencies``), not an extra.

Through 2.3.2 they lived in an optional ``[geog]`` extra that ``[all]``
deliberately excluded and no quickstart named.  The check for them sat
*after* the tile fetch, deep inside the mosaic step, and raised a bare
``RuntimeError``.  Following the published terrain doc on a documented
install therefore downloaded 160.7 MiB of Copernicus tiles and then died
on an import, at exit 1, with a traceback and no remedy.

Two things are fixed here and both matter independently:

- the probe is **cheap and callable before any network work**, so the
  front door can refuse in advance of the fetch rather than after it;
- the remedy is **one string, written once**, so the refusal, the deep
  guards and ``gpuwm doctor`` cannot drift into naming different install
  commands for the same missing library.

Nothing in this module imports rasterio or pyproj at module scope: it is
imported by ``gpuwm doctor`` on boxes that may not have them, and a
doctor that cannot run without the thing it diagnoses is useless.
"""
from __future__ import annotations

from importlib.util import find_spec

#: The libraries the high-resolution geography path cannot run without,
#: paired with what each one actually does, so a refusal can say why the
#: missing piece is needed instead of only that it is missing.
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


def missing_geog_modules() -> tuple[str, ...]:
    """Names of the geography modules that cannot be imported.

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
    """The single remedy string, naming the exact command to run.

    ``missing`` empty means "say it generically" -- the deep guards call
    it that way, because by the time an import actually fails they know
    only that one did.
    """
    roles = dict(GEOG_MODULES)
    if missing:
        named = ", ".join(
            f"{name} ({roles.get(name, 'required')})" for name in missing)
        head = (f"high-resolution geography needs {named}, which this "
                "environment cannot import")
    else:
        head = ("high-resolution geography needs rasterio and pyproj, "
                "which this environment cannot import")
    return (
        head
        + ".  Both ship with gpuwm as ordinary runtime dependencies from "
        "2.3.3 on, so an install that is missing them is either older "
        "than that or was built with --no-deps.\n"
        "  remedy: pip install --upgrade gpuwm\n"
        "  #   ... or, to add just these to the environment you have:\n"
        "  #   pip install --upgrade " + " ".join(GEOG_DISTRIBUTIONS) + "\n"
        "  # `gpuwm doctor` reports this check as 'geography stack'.")
