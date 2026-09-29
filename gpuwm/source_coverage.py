"""Where a source's native grid reaches, computed from declared table data.

A REGIONAL source is one whose native grid stops somewhere: ICON-EU ends at
the edge of Europe, RAP at the edge of North America, HRRR and RRFS at the
edge of the CONUS Lambert.  Every front door that plans against a source --
the ``gpuwm domain`` wizard's fitted root, ``gpuwm fetch``'s ``--area``
gate -- has to be able to ask "does this source reach my target", and the
answer must be the SAME answer in both places or the wizard emits a plan the
fetch refuses (the field defect that retired the hand-held CONUS box).

The answer is table data.  A row in :mod:`gpuwm.source_adapters` declares
the source's grid with the numbers a GRIB grid-definition section already
carries; this module turns that declaration into the two things a front door
asks for.  Adding a regional model is therefore a row, not a function -- the
arbitrary acceptance test applied to coverage.

The two questions, and why they are not the same question:

``envelope()``
    The ``(south, west, north, east)`` lat/lon bounding box, for clamping and
    formatting an ``--area`` hint.  For a wide Lambert grid the bounding box
    is LOOSER than the grid (grid 221's top row swings past 85 N), so it is
    the right shape for a box to be clamped into and the wrong shape for a
    refusal -- it would accept ground the grid does not carry.

``outside()`` / ``locate()``
    Exact membership in the source's OWN index space, which is the test the
    preparation stage performs on real bytes.  A refusal is built on this
    one, so the wizard refuses exactly what the preparation would refuse and
    can name the offending point in the source's coordinates.

Two declared families cover every regional product gpuwm ships a profile
for, and neither is a per-model code path:

``RegularLatLonWindow``
    The product is published on a regular lat/lon window, so the window IS
    the grid (DWD's ICON-EU: the four numbers DWD publishes as the grid
    description).

``LambertGridWindow``
    The product is published on a Lambert conformal grid, so membership and
    envelope both come from walking the declared mass grid through the same
    projection transcription WPS uses (:mod:`gpuwm.static.lambert`).  The
    envelope is exact rather than sampled: the projected pole lies outside
    the grid rectangle, latitude is monotone in projected distance from the
    pole point and longitude in angle around it, so every lat/lon extreme
    over the convex rectangle is attained on its boundary.

A source with no window declared is GLOBAL -- it reaches every target, and
the front doors apply no coverage bound at all.

``config_source_coverage_refusal()`` asks the same question of a written
config's root, for plan review and the ``gpuwm prep`` door.  It lives here,
beside the windows it reads, because the RW-WPS preprocessing wheel ships
this module and not the domain wizard.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


#: GRIB2 shape-of-earth code 6, the spherical radius NCEP and DWD stamp on
#: the regional products this module describes.  Declared because a window
#: that does not say which sphere its metres were measured on cannot be
#: reprojected without guessing.
GRIB_SPHERICAL_EARTH_RADIUS_M = 6371229.0


@dataclass(frozen=True)
class RegularLatLonWindow:
    """A regular lat/lon product's published window, in signed degrees.

    ``nx``/``ny`` are the grid's point counts, declared so a refusal can
    name the source index a target maps to -- the coordinates the
    preparation stage refuses in.  They are optional: a window with no
    counts still answers membership, in degrees.
    """

    south: float
    west: float
    north: float
    east: float
    nx: int | None = None
    ny: int | None = None

    def __post_init__(self) -> None:
        if not (self.south < self.north and self.west < self.east):
            raise ValueError(
                "a regular lat/lon coverage window must run south-to-north "
                "and west-to-east in signed degrees; got "
                f"lat {self.south}..{self.north}, lon {self.west}..{self.east}")

    def envelope(self) -> tuple[float, float, float, float]:
        return (float(self.south), float(self.west),
                float(self.north), float(self.east))

    def _shift(self, longitude):
        """LONGITUDE on the branch this window is written on."""

        centre = 0.5 * (float(self.west) + float(self.east))
        longitude = np.asarray(longitude, dtype=np.float64)
        return centre + ((longitude - centre + 180.0) % 360.0) - 180.0

    def outside(self, latitude, longitude) -> np.ndarray:
        latitude = np.asarray(latitude, dtype=np.float64)
        shifted = self._shift(longitude)
        return ((latitude < self.south) | (latitude > self.north)
                | (shifted < self.west) | (shifted > self.east))

    def locate(self, latitude: float, longitude: float) -> str:
        shifted = float(self._shift(longitude))
        if self.nx and self.ny:
            x = ((shifted - self.west)
                 * (self.nx - 1) / (self.east - self.west))
            y = ((float(latitude) - self.south)
                 * (self.ny - 1) / (self.north - self.south))
            return (f"maps to source index x={x:.3f} y={y:.3f}, and the "
                    f"source covers x=0..{self.nx - 1} "
                    f"(lon {self.west:g}..{self.east:g}) "
                    f"y=0..{self.ny - 1} "
                    f"(lat {self.south:g}..{self.north:g})")
        return (f"lies outside the source window lat "
                f"{self.south:g}..{self.north:g}, "
                f"lon {self.west:g}..{self.east:g}")

    def centre(self) -> tuple[float, float]:
        return (0.5 * (float(self.south) + float(self.north)),
                0.5 * (float(self.west) + float(self.east)))

    def describe(self) -> str:
        return (f"regular lat/lon window lat {self.south:g}..{self.north:g}, "
                f"lon {self.west:g}..{self.east:g}")


@dataclass(frozen=True)
class LambertGridWindow:
    """A Lambert conformal product's grid, as its GRIB section declares it.

    ``lat1``/``lon1`` are the FIRST grid point (the south-west mass point
    for the scanning mode every shipped product uses), ``nx``/``ny`` the
    mass-grid counts, ``dx_m`` the grid spacing on the sphere named by
    ``earth_radius_m``.  The spacing is rescaled onto WPS's fixed 6,370 km
    sphere before the projection is walked, which makes the WPS
    transcription geometrically identical to the GRIB shape-of-earth the
    bytes were encoded on -- the same reconciliation
    :data:`gpuwm.ingest.hrrr.HRRR_WPS_EQUIVALENT_DX_M` performs for the
    certified HRRR route.
    """

    nx: int
    ny: int
    lat1: float
    lon1: float
    dx_m: float
    truelat1: float
    truelat2: float
    stand_lon: float
    earth_radius_m: float = GRIB_SPHERICAL_EARTH_RADIUS_M

    def __post_init__(self) -> None:
        if self.nx < 2 or self.ny < 2:
            raise ValueError(
                "a Lambert coverage window needs at least a 2x2 mass grid; "
                f"got {self.nx}x{self.ny}")
        if not self.dx_m > 0.0 or not self.earth_radius_m > 0.0:
            raise ValueError(
                "a Lambert coverage window needs a positive grid spacing "
                "and earth radius; got "
                f"dx {self.dx_m} m on radius {self.earth_radius_m} m")

    def grid(self):
        """The declared grid as a :class:`~gpuwm.static.lambert.LambertGrid`.

        Deferred import: the source registry reads this module, and a base
        install imports the registry long before any projection work.
        """

        from gpuwm.static.lambert import LambertGrid
        from gpuwm.static.projection import EARTH_RADIUS_M

        scaled = float(self.dx_m) * EARTH_RADIUS_M / float(self.earth_radius_m)
        return LambertGrid(
            ref_lat=float(self.lat1), ref_lon=float(self.lon1),
            truelat1=float(self.truelat1), truelat2=float(self.truelat2),
            stand_lon=float(self.stand_lon),
            dx=scaled, dy=scaled,
            e_we=int(self.nx) + 1, e_sn=int(self.ny) + 1,
            known_x=1.0, known_y=1.0)

    def envelope(self) -> tuple[float, float, float, float]:
        grid = self.grid()
        # One-based mass coordinates, as LambertGrid registers them.
        i = np.arange(1, int(self.nx) + 1, dtype=np.float64)
        j = np.arange(1, int(self.ny) + 1, dtype=np.float64)
        ring_x = np.concatenate((i, i, np.full(j.size, 1.0),
                                 np.full(j.size, float(self.nx))))
        ring_y = np.concatenate((np.full(i.size, 1.0),
                                 np.full(i.size, float(self.ny)), j, j))
        latitude, longitude = grid.ij_to_latlon(ring_x, ring_y)
        if not (np.isfinite(latitude).all() and np.isfinite(longitude).all()):
            raise ValueError(
                f"the declared {self.nx}x{self.ny} Lambert coverage window "
                "produced a non-finite envelope")
        return (float(latitude.min()), float(longitude.min()),
                float(latitude.max()), float(longitude.max()))

    def _ij(self, latitude, longitude):
        x, y = self.grid().latlon_to_ij(
            np.asarray(latitude, dtype=np.float64),
            np.asarray(longitude, dtype=np.float64))
        return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)

    def outside(self, latitude, longitude) -> np.ndarray:
        x, y = self._ij(latitude, longitude)
        inside = (np.isfinite(x) & np.isfinite(y)
                  & (x >= 1.0) & (x <= float(self.nx))
                  & (y >= 1.0) & (y <= float(self.ny)))
        return ~inside

    def locate(self, latitude: float, longitude: float) -> str:
        x, y = self._ij(float(latitude), float(longitude))
        # Reported zero-based, the convention the preparation stage's own
        # out-of-grid refusal prints.
        return (f"maps to source index x={float(x) - 1.0:.3f} "
                f"y={float(y) - 1.0:.3f}, and the source covers "
                f"x=0..{self.nx - 1} y=0..{self.ny - 1} "
                f"({self.describe()})")

    def centre(self) -> tuple[float, float]:
        latitude, longitude = self.grid().ij_to_latlon(
            0.5 * (1.0 + float(self.nx)), 0.5 * (1.0 + float(self.ny)))
        return (float(latitude), float(((float(longitude) + 180.0) % 360.0)
                                       - 180.0))

    def describe(self) -> str:
        return (f"{self.nx}x{self.ny} Lambert grid at {self.dx_m:g} m, "
                f"truelat {self.truelat1:g}/{self.truelat2:g}, "
                f"stand_lon {self.stand_lon:g}")


#: Every declared coverage-window family.  A registry row may name one of
#: these or nothing at all; the registry refuses anything else BY TYPE at
#: import, rather than discovering it at plan time.
COVERAGE_WINDOW_TYPES = (RegularLatLonWindow, LambertGridWindow)

CoverageWindow = RegularLatLonWindow | LambertGridWindow


def window_envelope(window: CoverageWindow | None
                    ) -> tuple[float, float, float, float] | None:
    """``(south, west, north, east)`` for WINDOW, or ``None`` when global."""

    if window is None:
        return None
    return window.envelope()


def window_centre(window: CoverageWindow | None
                  ) -> tuple[float, float] | None:
    """``(lat, lon)`` at the middle of WINDOW's grid, or ``None`` if global.

    The one place a refusal can accurately point at: "this source's grid is
    centred here".  Taken from the GRID rather than from the bounding box,
    because a wide Lambert grid's bounding-box centre can lie off the grid
    entirely (AWIPS 221's box spans nearly every meridian).
    """

    if window is None:
        return None
    return window.centre()


def points_outside(window: CoverageWindow | None,
                   latitude, longitude) -> np.ndarray:
    """Boolean mask of the LATITUDE/LONGITUDE points WINDOW does not cover.

    A global window covers everything, which is what the all-false mask
    says -- so a caller never has to branch on regional-versus-global.
    """

    latitude = np.asarray(latitude, dtype=np.float64)
    if window is None:
        return np.zeros(latitude.shape, dtype=bool)
    return np.asarray(window.outside(latitude, longitude), dtype=bool)


def _root_grid(projection: dict, nx: int, ny: int, root_dx_m: float):
    """The WRF root grid a PROJECTION table and an NX x NY mass count describe.

    ``projection`` carries the namelist's six projection entries by name
    (``map_proj``, ``ref_lat``, ``ref_lon``, ``truelat1``, ``truelat2``,
    ``stand_lon``); the grid is centred on ``ref_lat``/``ref_lon`` the way
    WPS places a root.  The wizard's fit and every coverage refusal walk
    this same grid, so they place the same footprint.

    Deferred import, for the reason :meth:`LambertGridWindow.grid` gives.
    """

    from gpuwm.static.projection import projection_class

    cls = projection_class(projection["map_proj"])
    return cls(
        ref_lat=projection["ref_lat"], ref_lon=projection["ref_lon"],
        truelat1=projection["truelat1"], truelat2=projection["truelat2"],
        stand_lon=projection["stand_lon"],
        dx=float(root_dx_m), dy=float(root_dx_m),
        e_we=nx + 1, e_sn=ny + 1)


def _root_coverage_gap(projection: dict, nx: int, ny: int, *, source: str,
                       root_dx_m: float) -> str | None:
    """The first root cell corner SOURCE's declared window does not carry.

    The corners bound every mass point, so a root whose corners the window
    carries is carried whole.  ``None`` when the source is global or its
    window carries every corner.  Otherwise the facts both coverage
    refusals state: the point, where it lands in the source's own index
    space, the window the source covers, how many corners fall outside,
    and where the source's grid is centred.  Shared so the wizard's fit
    and plan review of a written config refuse exactly the same domains
    in the same words.
    """

    from gpuwm.source_adapters import source_coverage_window
    from gpuwm.static.projection import _wrap180

    window = source_coverage_window(source)
    if window is None:
        return None
    latitude, longitude = _root_grid(
        projection, nx, ny, root_dx_m).latlon_c()
    outside = points_outside(window, latitude, longitude)
    if not bool(outside.any()):
        return None
    index = int(np.argmax(outside))
    bad_lat = float(np.asarray(latitude).reshape(-1)[index])
    bad_lon = float(np.asarray(longitude).reshape(-1)[index])
    centre_lat, centre_lon = window_centre(window)
    return (
        f"the {nx}x{ny} root's point at lat/lon ({bad_lat:.4f}, "
        f"{_wrap180(bad_lon):.4f}) {window.locate(bad_lat, bad_lon)}; "
        f"{int(outside.sum())} of the root's {outside.size} cell corners "
        f"are outside it.  {source}'s grid is centred at "
        f"({centre_lat:.2f}, {centre_lon:.2f})")


def config_source_coverage_refusal(experiment, source,
                                   *, source_option: str = "[fetch] source"
                                   ) -> str | None:
    """Why SOURCE cannot force a written config's root, or ``None``.

    Plan review's form of
    :func:`gpuwm.domain_wizard.source_coverage_refusal`, asked of a config
    the wizard did not fit: one written by hand or by a batch driver, or
    one whose source was changed after it was emitted.  The coverage is a
    table fact (the source row's declared window) and the geometry is the
    wizard's own (:func:`_root_grid`, which the wizard imports from here),
    so every door that reviews a config before its download refuses
    exactly what the fitted door refuses.

    It lives here rather than in :mod:`gpuwm.domain_wizard` because
    ``gpuwm prep --experiment-config`` asks it on every regional source,
    and the RW-WPS preprocessing wheel does not ship the wizard: imported
    from there, the question raised ModuleNotFoundError before the config
    was read.

    Breakage it prevents: a domain outside a regional source's grid was
    refused only at the root forcing stage of its preparation, after the
    whole cycle had been downloaded and decoded (three runs of a batch
    sweep, each about 40 s of decode and statics after the download,
    ending in "target points fall outside the source grid").

    ``None`` for a global source (no declared window), a name the source
    registry does not know (the source validators own that sentence), and
    an idealized config (no projection, so no footprint to place).
    ``source_option`` is how the door the reader typed names the source.
    """

    from gpuwm.source_adapters import source_coverage_window

    projection = getattr(experiment, "projection", None)
    if projection is None or not isinstance(source, str) or not source.strip():
        return None
    try:
        window = source_coverage_window(source)
    except ValueError:
        return None
    if window is None:
        return None
    root = experiment.root.run
    gap = _root_coverage_gap(
        {"map_proj": projection.map_proj, "ref_lat": projection.ref_lat,
         "ref_lon": projection.ref_lon, "truelat1": projection.truelat1,
         "truelat2": projection.truelat2,
         "stand_lon": projection.stand_lon},
        int(root.nx), int(root.ny), source=source, root_dx_m=float(root.dx))
    if gap is None:
        return None
    return (
        f"{source_option} {source} does not cover this domain: {gap}.  "
        "Left to the preparation, it is refused only at the root forcing "
        "stage, after the whole cycle is downloaded and decoded.  Next: "
        "move [projection] ref_lat and ref_lon inside that grid, or set "
        f"{source_option} to a source whose coverage includes this domain "
        "(gpuwm sources lists them)")
