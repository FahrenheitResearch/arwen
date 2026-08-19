"""Worldwide map-projection grids with WPS conventions.

This module hosts the shared projected-grid machinery (staggered
coordinate arrays, derived fields, WPS nest registration, namelist and
experiment-config dispatch) plus the Mercator and polar-stereographic
projections.  The Lambert conformal class lives in
:mod:`gpuwm.static.lambert` (unchanged math, now on the shared base).

The projection math is transcribed from WRF v4.6.1
``share/module_llxy.F`` (commit d66e442fccc04111067e29274c9f9eaccc3cef28,
the same code WPS geogrid links): ``set_merc``:1307, ``llij_merc``:1334,
``ijll_merc``:1358, ``set_ps``:696, ``llij_ps``:732, ``ijll_ps``:777.
The derived MAPFAC_* and SINALPHA/COSALPHA formulas are transcribed from
WPS v4.6.0 ``geogrid/src/process_tile_module.F`` ``get_map_factor``
(:1735) and ``get_rotang`` (:1920) -- the code that populates geo_em.
Every branch is gated at binary64 against the committed oracle fixture
``tests/data/llxy_oracle/`` built from the pristine pinned source by
``tools/llxy_wrf461_oracle/build_llxy.py`` (measured max-ULP ceilings
recorded in ``tests/test_projection_oracle.py``).

Wind-rotation convention (the one the stored SINALPHA/COSALPHA fields
follow, per ``get_rotang``): ``alpha = wrap(stand_lon - lon) * cone``
with the Lambert cone from ``lc_cone`` (positive in both hemispheres),
``cone == 1`` for polar stereographic, and identically zero for
Mercator.  There is *no* hemisphere factor in the authority.  The
previous ``hemi *`` factor in the product's Lambert ``rotation_m`` was
an unreachable southern-hemisphere guess (the wizard refused SH); it
multiplied by exactly ``1.0`` on every reachable grid, so dropping it
leaves every northern-hemisphere field byte-identical while making the
southern branch follow geogrid.

Registration, nesting, Coriolis, and constants are shared verbatim with
the Lambert implementation; see :mod:`gpuwm.static.lambert` for the
geo_em-arbitrated conventions.
"""
from __future__ import annotations

from fractions import Fraction
import math
import re
import weakref
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from gpuwm.experiment import ExperimentConfig

#: Earth radius (m) -- module_llxy.F:152 (WPS uses the same value).
EARTH_RADIUS_M = 6370000.0
#: Earth angular velocity (rad/s) -- WPS constants_module OMEGA_E.  This is
#: the value geogrid bakes into geo_em F/E; do not "fix" it to WRF's EOMEG.
OMEGA_E = 7.292e-5

_RAD_PER_DEG = np.pi / 180.0
_DEG_PER_RAD = 180.0 / np.pi

#: WPS map_proj string -> WRF header integer (module_llxy PROJ_* codes).
WRF_MAP_PROJ_CODES = {"lambert": 1, "polar": 2, "mercator": 3}
#: WRF header integer -> WPS map_proj string.
WPS_MAP_PROJ_NAMES = {code: name for name, code
                      in WRF_MAP_PROJ_CODES.items()}
#: wrfout MAP_PROJ_CHAR values, keyed by WPS map_proj string.
MAP_PROJ_CHARS = {"lambert": "Lambert Conformal",
                  "polar": "Polar Stereographic",
                  "mercator": "Mercator"}


def _free_rust_grid_handle(handle: int) -> None:
    """weakref.finalize target: release a crate grid handle, silently
    tolerating a bridge already torn down at interpreter exit."""
    try:
        from . import rust_bridge
        rust_bridge.grid_free(handle)
    except Exception:
        pass


def _wrap180(dlon):
    """Wrap a longitude difference into (-180, 180] (module_llxy cut-zone)."""
    d = np.asarray(dlon, dtype=np.float64)
    d = np.where(d > 180.0, d - 360.0, d)
    d = np.where(d < -180.0, d + 360.0, d)
    return d


class ProjectedGrid:
    """Shared machinery for one WRF domain on a conformal projection.

    Subclasses (:class:`~gpuwm.static.lambert.LambertGrid`,
    :class:`MercatorGrid`, :class:`PolarStereoGrid`) provide
    ``ij_to_latlon``, ``latlon_to_ij``, ``map_factor`` and ``rotation``;
    everything else -- grid registration, staggered coordinate arrays,
    MAPFAC/Coriolis/rotation field assembly, and WPS parent_start/ratio
    nest arithmetic -- is identical across projections and lives here.

    Constructor signature is uniform across projections (parameters a
    projection does not use are validated and stored for receipts but
    do not enter its math; e.g. Mercator ignores ``truelat2`` and
    ``stand_lon`` exactly as WPS does).
    """

    #: WPS map_proj string; set by each subclass.
    map_proj: str = ""

    #: Placement-translation bookkeeping (see :meth:`translated`):
    #: ``None``/zero on ordinary grids; a translated grid carries its
    #: reference grid and the exact integer index offset onto it.
    _translation_reference = None
    _translation_offset = (0, 0)

    def __init__(self, ref_lat: float, ref_lon: float, truelat1: float,
                 truelat2: float, stand_lon: float, dx: float, dy: float,
                 e_we: int, e_sn: int,
                 known_x: float | None = None, known_y: float | None = None,
                 moad_cen_lat: float | None = None,
                 moad_cen_lon: float | None = None):
        if abs(dx - dy) > 1e-9 * abs(dx):
            raise ValueError(
                f"{type(self).__name__} requires dx == dy, "
                f"got {dx} != {dy}")
        self.ref_lat = float(ref_lat)
        self.ref_lon = float(ref_lon)
        self.truelat1 = float(truelat1)
        self.truelat2 = float(truelat2)
        self.stand_lon = float(stand_lon)
        self.dx = float(dx)
        self.dy = float(dy)
        self.e_we = int(e_we)
        self.e_sn = int(e_sn)
        centered_reference = known_x is None and known_y is None
        self.known_x = self.e_we / 2.0 if known_x is None else float(known_x)
        self.known_y = self.e_sn / 2.0 if known_y is None else float(known_y)
        self.moad_cen_lat = (self.ref_lat if moad_cen_lat is None
                             else float(moad_cen_lat))
        self.moad_cen_lon = (self.ref_lon if moad_cen_lon is None
                             else float(moad_cen_lon))
        # map_set (module_llxy.F:533-537): hemisphere from truelat1 for
        # every dx-carrying projection.
        self.hemi = -1.0 if self.truelat1 < 0.0 else 1.0
        self._setup()
        if centered_reference:
            # Avoid a projection round-trip for d01: besides being exact,
            # this preserves the global-attribute bytes of the legacy path.
            self.cen_lat = self.ref_lat
            self.cen_lon = self.ref_lon
        else:
            cen_lat, cen_lon = self.ij_to_latlon(self.e_we / 2.0,
                                                 self.e_sn / 2.0)
            self.cen_lat = float(cen_lat)
            self.cen_lon = float(cen_lon)

    # -- per-projection hooks ------------------------------------------------

    def _setup(self) -> None:
        """set_* transcription: derive projection state from the known
        point (subclass responsibility)."""
        raise NotImplementedError

    def ij_to_latlon(self, x, y):
        raise NotImplementedError

    def latlon_to_ij(self, lat, lon):
        raise NotImplementedError

    def map_factor(self, lat):
        raise NotImplementedError

    def rotation(self, lon):
        """(SINALPHA, COSALPHA) at longitudes LON (get_rotang
        transcription; subclass responsibility)."""
        raise NotImplementedError

    @property
    def wrf_map_proj(self) -> int:
        """WRF header MAP_PROJ integer (1=lambert, 2=polar, 3=mercator)."""
        return WRF_MAP_PROJ_CODES[self.map_proj]

    @property
    def map_proj_char(self) -> str:
        """wrfout MAP_PROJ_CHAR string."""
        return MAP_PROJ_CHARS[self.map_proj]

    # -- the Rust seam (fixed-means-default) --------------------------------
    #
    # The staggered coordinate arrays and derived fields below route to
    # the static-fields cdylib by default; the numpy bodies stay as the
    # parity reference and the explicit fallback (GPUWM_STATIC_PYTHON=1
    # or an unloadable library), reported once per operation through
    # gpuwm.static.rust_bridge.route.  Scalar ij<->latlon transforms
    # stay Python: they are orchestration-scale (namelist corners, nest
    # anchors) and bit-equal to the Rust f64 path by the lane-1 ledger.

    def _rust_spec(self) -> dict:
        """This grid as the crate's GridSpec JSON document (total: every
        default -- known point, MOAD centre -- is already resolved)."""
        return {
            "kind": self.map_proj,
            "ref_lat": self.ref_lat, "ref_lon": self.ref_lon,
            "truelat1": self.truelat1, "truelat2": self.truelat2,
            "stand_lon": self.stand_lon,
            "dx": self.dx, "dy": self.dy,
            "e_we": self.e_we, "e_sn": self.e_sn,
            "known_x": self.known_x, "known_y": self.known_y,
            "moad_cen_lat": self.moad_cen_lat,
            "moad_cen_lon": self.moad_cen_lon,
        }

    def _rust_handle(self, bridge) -> int:
        """The cached crate grid handle for this instance.

        A translated grid crosses as reference handle + integer offset
        (`gpuwm_static_grid_translated`) so the crate delegates through
        the reference's own floats exactly as :meth:`translated`
        documents; every other grid crosses as its resolved spec.  A
        crate refusal here PROPAGATES: with the bridge loaded, a spec
        the crate refuses is a defect, and a silent fallback would hide
        it (fixed-means-default).
        """
        handle = self.__dict__.get("_rust_grid_handle")
        if handle is not None:
            return handle
        base = self._translation_reference
        if base is None:
            handle = bridge.grid_new(self._rust_spec())
        else:
            di, dj = self._translation_offset
            handle = bridge.grid_translated(
                base._rust_handle(bridge), di, dj, self.e_we, self.e_sn)
        self.__dict__["_rust_grid_handle"] = handle
        weakref.finalize(self, _free_rust_grid_handle, handle)
        return handle

    def _stagger_shape(self, stagger: int) -> tuple[int, int]:
        return {
            0: (self.e_sn - 1, self.e_we - 1),  # mass
            1: (self.e_sn - 1, self.e_we),      # U
            2: (self.e_sn, self.e_we - 1),      # V
            3: (self.e_sn, self.e_we),          # corner
        }[stagger]

    def _rust_arrays(self, stagger: int, kinds: tuple[int, ...]):
        """Tuple of derived arrays via the seam, or None on fallback.

        ``stagger`` is a ``rust_bridge.STAGGER_*`` code and each kind a
        ``rust_bridge.ARRAY_*`` code (lat=0, lon=1, mapfac=2, F=3, E=4,
        sinalpha=5, cosalpha=6)."""
        from . import rust_bridge
        bridge = rust_bridge.route("projection arrays")
        if bridge is None:
            return None
        handle = self._rust_handle(bridge)
        ny, nx = self._stagger_shape(stagger)
        return tuple(bridge.grid_array(handle, stagger, kind, ny, nx)
                     for kind in kinds)

    # -- staggered coordinate arrays ----------------------------------------

    def _grid_xy(self, xoff: float, yoff: float, nx: int, ny: int):
        x = np.arange(1, nx + 1, dtype=np.float64) + xoff
        y = np.arange(1, ny + 1, dtype=np.float64) + yoff
        return np.meshgrid(x, y)  # each (ny, nx)

    def latlon_mass(self):
        """(lat, lon) at mass points, shape (e_sn-1, e_we-1)."""
        routed = self._rust_arrays(0, (0, 1))
        if routed is not None:
            return routed
        return self.ij_to_latlon(*self._grid_xy(0.0, 0.0,
                                                self.e_we - 1, self.e_sn - 1))

    def latlon_u(self):
        """(lat, lon) at U points, shape (e_sn-1, e_we)."""
        routed = self._rust_arrays(1, (0, 1))
        if routed is not None:
            return routed
        return self.ij_to_latlon(*self._grid_xy(-0.5, 0.0,
                                                self.e_we, self.e_sn - 1))

    def latlon_v(self):
        """(lat, lon) at V points, shape (e_sn, e_we-1)."""
        routed = self._rust_arrays(2, (0, 1))
        if routed is not None:
            return routed
        return self.ij_to_latlon(*self._grid_xy(0.0, -0.5,
                                                self.e_we - 1, self.e_sn))

    def latlon_c(self):
        """(lat, lon) at corner (C/vorticity) points, shape (e_sn, e_we)."""
        routed = self._rust_arrays(3, (0, 1))
        if routed is not None:
            return routed
        return self.ij_to_latlon(*self._grid_xy(-0.5, -0.5,
                                                self.e_we, self.e_sn))

    # -- derived fields ------------------------------------------------------

    def mapfac_m(self):
        """MAPFAC_M: map factor at mass points (isotropic)."""
        routed = self._rust_arrays(0, (2,))
        if routed is not None:
            return routed[0]
        return self.map_factor(self.latlon_mass()[0])

    def mapfac_u(self):
        """MAPFAC_U: map factor at U points."""
        routed = self._rust_arrays(1, (2,))
        if routed is not None:
            return routed[0]
        return self.map_factor(self.latlon_u()[0])

    def mapfac_v(self):
        """MAPFAC_V: map factor at V points."""
        routed = self._rust_arrays(2, (2,))
        if routed is not None:
            return routed[0]
        return self.map_factor(self.latlon_v()[0])

    def coriolis_m(self):
        """(F, E) = (2*Omega*sin(lat), 2*Omega*cos(lat)) at mass points.

        The sign of F comes from sin(lat): negative in the southern
        hemisphere with no projection-specific handling, exactly as
        geogrid computes geo_em F.
        """
        routed = self._rust_arrays(0, (3, 4))
        if routed is not None:
            return routed
        lat = self.latlon_mass()[0]
        return (2.0 * OMEGA_E * np.sin(lat * _RAD_PER_DEG),
                2.0 * OMEGA_E * np.cos(lat * _RAD_PER_DEG))

    def rotation_m(self):
        """(SINALPHA, COSALPHA) at mass points."""
        routed = self._rust_arrays(0, (5, 6))
        if routed is not None:
            return routed
        return self.rotation(self.latlon_mass()[1])

    def rotation_u(self):
        """(SINALPHA, COSALPHA) at U points."""
        routed = self._rust_arrays(1, (5, 6))
        if routed is not None:
            return routed
        return self.rotation(self.latlon_u()[1])

    def rotation_v(self):
        """(SINALPHA, COSALPHA) at V points."""
        routed = self._rust_arrays(2, (5, 6))
        if routed is not None:
            return routed
        return self.rotation(self.latlon_v()[1])

    # -- nesting -------------------------------------------------------------

    def nest(self, i_parent_start: int, j_parent_start: int,
             parent_grid_ratio: int, e_we: int, e_sn: int, *,
             resolved_dx: float | None = None,
             resolved_dy: float | None = None) -> "ProjectedGrid":
        """Child grid from the WPS parent_start/ratio arithmetic.

        Nest mass coordinate x sits at parent mass coordinate
        ``(i_parent_start - 0.5) + (x - 0.5)/ratio`` (cell edges align).  The
        child is a full grid of the same projection whose known point is its
        own (1, 1) mass point, located through the parent transform.
        ``resolved_dx/dy`` bind callers that own an exact root-ratio chain;
        omitted values preserve the direct one-edge ``parent spacing /
        ratio`` API.
        """
        r = int(parent_grid_ratio)
        if r < 1:
            raise ValueError(f"parent_grid_ratio must be >= 1, got {r}")
        xp = (i_parent_start - 0.5) + 0.5 / r
        yp = (j_parent_start - 0.5) + 0.5 / r
        lat11, lon11 = self.ij_to_latlon(xp, yp)
        child_dx = self.dx / r if resolved_dx is None else float(resolved_dx)
        child_dy = self.dy / r if resolved_dy is None else float(resolved_dy)
        return type(self)(float(lat11), float(lon11), self.truelat1,
                          self.truelat2, self.stand_lon, child_dx,
                          child_dy, e_we, e_sn,
                          known_x=1.0, known_y=1.0,
                          moad_cen_lat=self.moad_cen_lat,
                          moad_cen_lon=self.moad_cen_lon)

    def translated(self, di_cells: int, dj_cells: int, *,
                   e_we: int | None = None,
                   e_sn: int | None = None) -> "ProjectedGrid":
        """The same grid moved by a whole number of its own cells.

        The returned grid's transforms DELEGATE to the reference grid
        with an exact integer index offset (``ij_to_latlon(x, y) ->
        reference.ij_to_latlon(x + di, y + dj)``), so a cell two
        footprints share evaluates through the reference's own floats at
        the reference's own index -- the same computation, hence the
        same latitude/longitude bytes, unconditionally.  That per-cell
        bitwise stability is what lets statics rebuilt for a moved
        footprint equal the outgoing footprint's on shared ground
        (identical source + identical cells = identical bytes), which
        the relocation machinery asserts at every move rather than
        assumes.  Deliberately NOT re-anchored through a parent
        transform, and deliberately not a parameter reconstruction
        either: re-solving the pole from a shifted known point rounds in
        a different binade and can move a source-stencil selection on
        shared ground.  Translating a translated grid composes the
        offsets onto the ORIGINAL reference, so a long move chain never
        accumulates float error.

        ``e_we``/``e_sn`` re-extent the translated grid on the SAME
        lattice (omitted keeps the reference extent).  This is what a
        parent-extent statics corridor is built on: a larger window onto
        the reference grid's own cells, whose per-cell transforms stay
        the reference's exact arithmetic, so a footprint cropped out of
        a corridor build equals a direct footprint build bitwise.
        """
        di = int(di_cells)
        dj = int(dj_cells)
        if (di, dj) != (di_cells, dj_cells):
            raise ValueError(
                f"a placement translation is a whole number of cells, got "
                f"({di_cells!r}, {dj_cells!r})")
        base = self._translation_reference
        if base is None:
            base = self
            di_total, dj_total = di, dj
        else:
            di0, dj0 = self._translation_offset
            di_total, dj_total = di0 + di, dj0 + dj
        e_we = self.e_we if e_we is None else int(e_we)
        e_sn = self.e_sn if e_sn is None else int(e_sn)
        if e_we < 2 or e_sn < 2:
            raise ValueError(
                f"a translated extent needs at least one cell per axis, "
                f"got e_we={e_we}, e_sn={e_sn}")
        cls = type(base)
        translated_cls = _TRANSLATED_GRID_CLASSES.get(cls)
        if translated_cls is None:
            translated_cls = type(
                "Translated" + cls.__name__,
                (_TranslatedGridMixin, cls), {})
            _TRANSLATED_GRID_CLASSES[cls] = translated_cls
        new = translated_cls(
            base.ref_lat, base.ref_lon, base.truelat1, base.truelat2,
            base.stand_lon, base.dx, base.dy, e_we, e_sn,
            known_x=base.known_x - di_total,
            known_y=base.known_y - dj_total,
            moad_cen_lat=base.moad_cen_lat,
            moad_cen_lon=base.moad_cen_lon)
        new._translation_reference = base
        new._translation_offset = (di_total, dj_total)
        # Metadata recomputed through the delegated transform, so the
        # moved footprint's centre is exact reference arithmetic too.
        cen_lat, cen_lon = new.ij_to_latlon(new.e_we / 2.0,
                                            new.e_sn / 2.0)
        new.cen_lat = float(cen_lat)
        new.cen_lon = float(cen_lon)
        return new


#: One dynamically created Translated<X> subclass per projection class,
#: so ``isinstance(translated_grid, LambertGrid)`` (and every other
#: projection dispatch) keeps working.
_TRANSLATED_GRID_CLASSES: dict[type, type] = {}


class _TranslatedGridMixin:
    """Delegates the projection transforms to the reference grid with an
    exact integer index offset.  Until ``_translation_reference`` is set
    (i.e. during ``__init__``'s own centre computation) the ordinary
    class transforms apply; :meth:`ProjectedGrid.translated` re-derives
    the centre once the delegation is live."""

    def ij_to_latlon(self, x, y):
        reference = self._translation_reference
        if reference is None:
            return super().ij_to_latlon(x, y)
        di, dj = self._translation_offset
        return reference.ij_to_latlon(
            np.asarray(x, dtype=np.float64) + float(di),
            np.asarray(y, dtype=np.float64) + float(dj))

    def latlon_to_ij(self, lat, lon):
        reference = self._translation_reference
        if reference is None:
            return super().latlon_to_ij(lat, lon)
        di, dj = self._translation_offset
        x, y = reference.latlon_to_ij(lat, lon)
        return x - float(di), y - float(dj)


class MercatorGrid(ProjectedGrid):
    """One WRF domain on a Mercator projection (WPS ``map_proj='mercator'``).

    Transcription of ``set_merc``/``llij_merc``/``ijll_merc``
    (module_llxy.F:1307/1334/1358).  The projection is anchored at the
    known point's longitude (``lon1``); ``stand_lon`` does not enter the
    Mercator math (exactly as in module_llxy), and the wind rotation is
    identically zero (get_rotang).  ``truelat1`` is the latitude at
    which ``dx`` is true; ``truelat2`` is carried for receipts only.
    """

    map_proj = "mercator"

    def _setup(self) -> None:
        # --- set_merc transcription (module_llxy.F:1307) ---
        clain = np.cos(_RAD_PER_DEG * self.truelat1)
        self.dlon = self.dx / (EARTH_RADIUS_M * clain)
        self.rsw = 0.0
        if self.ref_lat != 0.0:
            self.rsw = float(
                np.log(np.tan(0.5 * ((self.ref_lat + 90.0) * _RAD_PER_DEG)))
                / self.dlon)
        self.dlon = float(self.dlon)

    def ij_to_latlon(self, x, y):
        """Transcription of ijll_merc (module_llxy.F:1358), vectorized."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        lat = (2.0 * np.arctan(np.exp(self.dlon
                                      * (self.rsw + y - self.known_y)))
               * _DEG_PER_RAD - 90.0)
        lon = (x - self.known_x) * self.dlon * _DEG_PER_RAD + self.ref_lon
        lon = np.where(lon > 180.0, lon - 360.0, lon)
        lon = np.where(lon < -180.0, lon + 360.0, lon)
        return lat, lon

    def latlon_to_ij(self, lat, lon):
        """Transcription of llij_merc (module_llxy.F:1334), vectorized."""
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        deltalon = _wrap180(lon - self.ref_lon)
        i = self.known_x + (deltalon / (self.dlon * _DEG_PER_RAD))
        j = (self.known_y
             + (np.log(np.tan(0.5 * ((lat + 90.0) * _RAD_PER_DEG))))
             / self.dlon - self.rsw)
        return i, j

    def map_factor(self, lat):
        """get_map_factor PROJ_MERC branch (process_tile_module.F:1801):
        sin(colat0) / sin(colat); equals 1 at truelat1."""
        lat = np.asarray(lat, dtype=np.float64)
        colat0 = _RAD_PER_DEG * (90.0 - self.truelat1)
        colat = _RAD_PER_DEG * (90.0 - lat)
        return np.sin(colat0) / np.sin(colat)

    def rotation(self, lon):
        """get_rotang PROJ_MERC branch: rotation is identically zero."""
        lon = np.asarray(lon, dtype=np.float64)
        return np.zeros_like(lon), np.ones_like(lon)


class PolarStereoGrid(ProjectedGrid):
    """One WRF domain on a polar stereographic projection
    (WPS ``map_proj='polar'``), either pole.

    Transcription of ``set_ps``/``llij_ps``/``ijll_ps``
    (module_llxy.F:696/732/777).  The pole (``hemi`` from the sign of
    ``truelat1``) is the projection centre; ``truelat1`` is the
    latitude where ``dx`` is true and ``stand_lon`` is the longitude
    parallel to +y.  ``truelat2`` is carried for receipts only.
    """

    map_proj = "polar"

    def _setup(self) -> None:
        # --- set_ps transcription (module_llxy.F:696) ---
        self.rebydx = EARTH_RADIUS_M / self.dx
        reflon = self.stand_lon + 90.0
        scale_top = 1.0 + self.hemi * np.sin(self.truelat1 * _RAD_PER_DEG)
        ala1 = self.ref_lat * _RAD_PER_DEG
        self.rsw = float(self.rebydx * np.cos(ala1) * scale_top
                         / (1.0 + self.hemi * np.sin(ala1)))
        alo1 = (self.ref_lon - reflon) * _RAD_PER_DEG
        self.polei = float(self.known_x - self.rsw * np.cos(alo1))
        self.polej = float(self.known_y
                           - self.hemi * self.rsw * np.sin(alo1))

    def ij_to_latlon(self, x, y):
        """Transcription of ijll_ps (module_llxy.F:777), vectorized."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        reflon = self.stand_lon + 90.0
        scale_top = 1.0 + self.hemi * np.sin(self.truelat1 * _RAD_PER_DEG)
        xx = x - self.polei
        yy = (y - self.polej) * self.hemi
        r2 = xx ** 2 + yy ** 2
        gi2 = (self.rebydx * scale_top) ** 2.0
        with np.errstate(invalid="ignore", divide="ignore"):
            lat = (_DEG_PER_RAD * self.hemi
                   * np.arcsin((gi2 - r2) / (gi2 + r2)))
            arccos = np.arccos(xx / np.sqrt(r2))
        lon = np.where(yy > 0, reflon + _DEG_PER_RAD * arccos,
                       reflon - _DEG_PER_RAD * arccos)
        # pole point (r2 == 0): mirror the Fortran branch explicitly.
        lat = np.where(r2 == 0.0, self.hemi * 90.0, lat)
        lon = np.where(r2 == 0.0, reflon, lon)
        lon = np.where(lon > 180.0, lon - 360.0, lon)
        lon = np.where(lon < -180.0, lon + 360.0, lon)
        return lat, lon

    def latlon_to_ij(self, lat, lon):
        """Transcription of llij_ps (module_llxy.F:732), vectorized.

        Note the source applies no cut-zone wrap to ``lon - reflon``
        (cos/sin absorb any offset); transcribed as-is.
        """
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        reflon = self.stand_lon + 90.0
        scale_top = 1.0 + self.hemi * np.sin(self.truelat1 * _RAD_PER_DEG)
        ala = lat * _RAD_PER_DEG
        rm = (self.rebydx * np.cos(ala) * scale_top
              / (1.0 + self.hemi * np.sin(ala)))
        alo = (lon - reflon) * _RAD_PER_DEG
        i = self.polei + rm * np.cos(alo)
        j = self.polej + self.hemi * rm * np.sin(alo)
        return i, j

    def map_factor(self, lat):
        """get_map_factor PROJ_PS branch (process_tile_module.F:1791):
        (1 + sin|truelat1|) / (1 + sin(sign(truelat1) * lat))."""
        lat = np.asarray(lat, dtype=np.float64)
        return ((1.0 + np.sin(_RAD_PER_DEG * abs(self.truelat1)))
                / (1.0 + np.sin(_RAD_PER_DEG
                                * np.copysign(1.0, self.truelat1) * lat)))

    def rotation(self, lon):
        """get_rotang PROJ_PS branch: alpha = wrap(stand_lon - lon)."""
        lon = np.asarray(lon, dtype=np.float64)
        alpha = _wrap180(self.stand_lon - lon) * _RAD_PER_DEG
        return np.sin(alpha), np.cos(alpha)


# ---------------------------------------------------------------------------
# namelist.wps parsing and dispatch
# ---------------------------------------------------------------------------

def projection_class(map_proj: str) -> type[ProjectedGrid]:
    """The grid class for a WPS map_proj string (lambert/mercator/polar)."""
    from gpuwm.static.lambert import LambertGrid
    classes = {"lambert": LambertGrid, "mercator": MercatorGrid,
               "polar": PolarStereoGrid}
    key = str(map_proj).lower()
    if key not in classes:
        latlon = key in {
            "lat-lon", "latlon", "regular_ll", "rotated-lat-lon",
            "rotated_ll",
        }
        blocker = (
            "; regular/rotated latitude-longitude needs angular dx/dy "
            "rather than metre spacing and WRF's global/pole polar filter; "
            "rotated grids also need pole_lat/pole_lon state and the "
            "map_proj == 6 curvature branch"
            if latlon else "")
        raise NotImplementedError(
            f"map_proj {map_proj!r} not supported (implemented: "
            f"'lambert', 'mercator', 'polar'){blocker}")
    return classes[key]


def _parse_wps_namelist(path) -> dict:
    """Minimal Fortran-namelist reader: ``key = v1, v2, ...`` lines only
    (matches WPS namelist style; no multi-line continuations)."""
    values: dict[str, list] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.split("!", 1)[0].strip()
        if not line or line.startswith("&") or line == "/":
            continue
        if "=" not in line:
            continue
        key, rhs = line.split("=", 1)
        tokens = [t.strip() for t in rhs.strip().rstrip(",").split(",")]
        parsed = []
        for tok in tokens:
            if not tok:
                continue
            if re.fullmatch(r"'[^']*'|\"[^\"]*\"", tok):
                parsed.append(tok[1:-1])
            else:
                try:
                    parsed.append(int(tok))
                except ValueError:
                    try:
                        parsed.append(float(tok))
                    except ValueError:
                        parsed.append(tok)
        values[key.strip().lower()] = parsed
    return values


def _finite_spacing(value, key: str, path) -> float:
    """One namelist grid spacing, refused as a sentence if it is not one."""
    spacing = float(value)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError(
            f"{path}: &geogrid/{key} = {value!r} must be a positive, "
            f"finite grid spacing in metres.")
    return spacing


def grids_from_wps_namelist(path) -> list[ProjectedGrid]:
    """Build the projected grid for every domain of a WPS namelist.

    Reads &share ``max_dom`` and the &geogrid projection + nest layout keys;
    d01 uses ``dx``/``dy``/``ref_lat``/``ref_lon`` directly and each nest is
    placed with :meth:`ProjectedGrid.nest` (parent_start/ratio arithmetic).
    ``map_proj`` selects lambert/mercator/polar; for mercator a missing
    ``truelat2``/``stand_lon`` follows WPS semantics (unused by the math)
    and defaults to ``truelat1``/``ref_lon``.
    """
    v = _parse_wps_namelist(path)
    proj = str(v.get("map_proj", ["lambert"])[0]).lower()
    cls = projection_class(proj)
    max_dom = int(v.get("max_dom", [1])[0])
    e_we, e_sn = v["e_we"], v["e_sn"]
    # &share/max_dom and the &geogrid per-domain arrays are declared
    # independently in a namelist.wps, so nothing but this check stops
    # them disagreeing.  Without it the loop below indexed past the end
    # of whichever array was short and the reader got an IndexError
    # traceback -- and, because gfs_direct.main catches only
    # (ValueError, OSError), it escaped the door that turns refusals
    # into sentences.  Name every short key at once: fixing them one
    # traceback per edit is the slowest possible way to learn the shape.
    # Only the arrays the loop below will actually index.  A
    # single-domain namelist legitimately omits the nest-only keys --
    # `for n in range(1, max_dom)` never runs -- so requiring them at
    # max_dom = 1 would refuse valid files, which is how the first
    # version of this guard broke four contract fixtures.
    _required = [("e_we", e_we), ("e_sn", e_sn)]
    if max_dom > 1:
        _required += [(name, v.get(name, []))
                      for name in ("parent_id", "parent_grid_ratio",
                                   "i_parent_start", "j_parent_start")]
    _short = {name: len(values) for name, values in _required
              if len(values) < max_dom}
    if _short:
        listed = ", ".join(f"{name} has {count}"
                           for name, count in sorted(_short.items()))
        raise ValueError(
            f"{path}: &share/max_dom = {max_dom} declares {max_dom} "
            f"domains, but {listed} -- every &geogrid per-domain array "
            f"must carry one entry per domain.  Either give the missing "
            f"entries, or set max_dom to the number of domains you "
            f"actually described.")
    truelat1 = float(v["truelat1"][0])
    if proj == "lambert":
        truelat2 = float(v["truelat2"][0])
        stand_lon = float(v["stand_lon"][0])
    else:
        truelat2 = float(v.get("truelat2", [truelat1])[0])
        stand_lon = float(v.get("stand_lon", v["ref_lon"])[0])
    # Same guard as the experiment loader's dx: Fraction(inf) raises
    # OverflowError where Fraction(nan) raises ValueError, so without a
    # finiteness test one spelling of "not a number" is a sentence and
    # the other is a traceback.
    root_dx = Fraction(_finite_spacing(v["dx"][0], "dx", path))
    root_dy = Fraction(_finite_spacing(v["dy"][0], "dy", path))
    spacing_by_grid_id = {1: (root_dx, root_dy)}
    grids: list[ProjectedGrid] = [
        cls(ref_lat=float(v["ref_lat"][0]),
            ref_lon=float(v["ref_lon"][0]),
            truelat1=truelat1, truelat2=truelat2, stand_lon=stand_lon,
            dx=float(v["dx"][0]), dy=float(v["dy"][0]),
            e_we=int(e_we[0]), e_sn=int(e_sn[0]))]
    for n in range(1, max_dom):
        parent_id = int(v["parent_id"][n])
        parent = grids[parent_id - 1]
        ratio = int(v["parent_grid_ratio"][n])
        parent_dx, parent_dy = spacing_by_grid_id[parent_id]
        child_dx, child_dy = parent_dx / ratio, parent_dy / ratio
        grids.append(parent.nest(int(v["i_parent_start"][n]),
                                 int(v["j_parent_start"][n]),
                                 ratio, int(e_we[n]), int(e_sn[n]),
                                 resolved_dx=float(child_dx),
                                 resolved_dy=float(child_dy)))
        spacing_by_grid_id[n + 1] = (child_dx, child_dy)
    return grids


def grids_from_projection_config(
        exp: "ExperimentConfig") -> list[ProjectedGrid]:
    """Build one registered projected grid per experiment domain.

    The root consumes :class:`gpuwm.experiment.ProjectionConfig` and its
    resolved per-domain ``RunConfig`` dimensions.  Children are placed via
    :meth:`ProjectedGrid.nest`, using the experiment's parent id/start/ratio
    layout.  Consequently the config and WPS entry points share exactly the
    same projection and parent-child registration math.

    Domains are returned in ``exp.domains`` order (the validated
    parent-before-child order), not indexed by ``grid_id``.
    """
    projection = exp.projection
    if projection is None:
        raise ValueError(
            f"experiment {exp.name!r} has no ProjectionConfig; "
            "a real projected grid requires a [projection] table")
    cls = projection_class(projection.map_proj)

    grids: list[ProjectedGrid] = []
    by_id: dict[int, ProjectedGrid] = {}
    for domain in exp.domains:
        cfg = domain.run
        if domain.parent_id == 0:
            grid = cls(
                ref_lat=projection.ref_lat, ref_lon=projection.ref_lon,
                truelat1=projection.truelat1,
                truelat2=projection.truelat2,
                stand_lon=projection.stand_lon,
                dx=cfg.dx, dy=cfg.dy, e_we=cfg.nx + 1,
                e_sn=cfg.ny + 1)
        else:
            try:
                parent = by_id[domain.parent_id]
            except KeyError as exc:
                raise ValueError(
                    f"domain {domain.grid_id} names parent "
                    f"{domain.parent_id}, which does not precede it") from exc
            grid = parent.nest(
                domain.i_parent_start, domain.j_parent_start,
                domain.parent_grid_ratio, cfg.nx + 1, cfg.ny + 1,
                resolved_dx=cfg.dx, resolved_dy=cfg.dy)

        expected = (cfg.nx, cfg.ny, cfg.dx, cfg.dy)
        actual = (grid.e_we - 1, grid.e_sn - 1, grid.dx, grid.dy)
        if actual != expected:
            raise ValueError(
                f"domain {domain.grid_id} grid layout resolves to {actual}, "
                f"but its RunConfig declares {expected}")
        grids.append(grid)
        by_id[domain.grid_id] = grid
    return grids
