"""Domain static-field builder: the WPS geogrid equivalent (CPU, float64).

CPU float64 is the accepted architecture per plan adjudication d7b5b6d.

Produces every geo_em field gpuwm Phase 3 needs from a
:class:`~gpuwm.static.lambert.LambertGrid` and a WPS_GEOG tree, following
geogrid's processing conventions.  Where the WPS algorithm could be read
more than one way the bundle geo_em files arbitrate (v4.6.0 geogrid output;
see tests/test_static_build.py for the acceptance gates):

- **Grid-cell accumulation** (geogrid ``average_gcell`` / scalar categorical
  processing): every source pixel is assigned to the model cell whose
  centre is nearest -- ``nint`` of the pixel's model-grid coordinates --
  and cell means / per-category counts are accumulated.  Used whenever the
  source is at least ``GCELL_RATIO``x finer than the grid (continuous) and
  always for scalar categorical sources (cells left empty fall back to the
  interpolation sequence / nearest neighbour).  WPS continuous category
  planes are interpolated independently, then normalized across categories.
- **Interpolators**: ``four_pt`` (bilinear), ``average_4pt`` /
  ``average_16pt`` (mean of valid neighbours), ``sixteen_pt`` (MM5/WPS
  overlapping parabolic via ``oned``; any missing point falls through),
  ``search_nearest`` (nearest valid source pixel).
- **Halo**: fields are computed on the grid extended by ``HALO = 3`` cells
  (geogrid's processing halo) so terrain smoothing has support beyond the
  domain edge, then cropped.
- **Terrain smoothing**: one WPS ``smth-desmth_special`` pass -- a 0.50
  smoothing + (-0.52) desmoothing sweep pair, x then y, followed by
  restoration of any originally non-negative point made negative.
- **Landmask / dominant categories** (pinned exactly by geo_em d01):
  LANDMASK = 0 where LANDUSEF(iswater) + LANDUSEF(islake) >= 0.5;
  LU_INDEX = dominant water type over water (lake beats ocean only when
  strictly larger), dominant *land* category over land; SCT_DOM/SCB_DOM =
  plain argmax (lowest category wins ties).
- **Masked fields** (water cells overwritten with geogrid's fills):
  ALBEDO12M -> 8, GREENFRAC/LAI12M/SNOALB/SOILTEMP -> 0.
- **TMN** = SOILTEMP - 0.0065 * HGT_M on land only (WRF v4.6.1
  share/module_soil_pre.F:973, Noah case), SOILTEMP elsewhere.
"""
from __future__ import annotations

from collections.abc import MutableMapping
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .geog import GeogDataset, GeogWindow
from .lambert import EARTH_RADIUS_M, LambertGrid, _parse_wps_namelist

#: metres per degree of a great circle on the WPS sphere.
M_PER_DEG = EARTH_RADIUS_M * np.pi / 180.0
#: geogrid processing halo (cells beyond the domain edge).
HALO = 3
#: minimum source-to-grid resolution ratio for grid-cell averaging
#: (GEOGRID.TBL ``average_gcell(4.0)`` for every 12-monthly/terrain field).
GCELL_RATIO = 4.0


# ``default`` is the exact Phase-3 dataset inventory.  Keeping it in one
# immutable table makes the legacy ``build_static(grid, geog_root)`` call
# byte-inert while allowing experiment configurations to resolve WPS
# ``geog_data_res`` selectors explicitly.
_DEFAULT_GEOG_DIRS = {
    "terrain": "topo_gmted2010_30s",
    "landuse": "modis_landuse_20class_30s_with_lakes",
    "soil_top": "soiltype_top_30s",
    "soil_bottom": "soiltype_bot_30s",
    "greenfrac": "greenfrac_fpar_modis",
    "lai": "lai_modis_10m",
    "albedo": "albedo_modis",
    "snow_albedo": "maxsnowalb_modis",
    "soil_temperature": "soiltemp_1deg",
}
# Complete global low-resolution inventory distributed by NCAR.  The 5m
# products are preferable to silently treating an intentionally regional
# 30s bundle as global for coarse, wide-area domains.
_FIVE_MINUTE_GEOG_DIRS = {
    "terrain": "topo_gmted2010_5m",
    "landuse": "modis_landuse_20class_5m_with_lakes",
    "soil_top": "soiltype_top_5m",
    "soil_bottom": "soiltype_bot_5m",
    "greenfrac": "greenfrac_fpar_modis_5m",
    "lai": "lai_modis_10m",
    "albedo": "albedo_modis",
    "snow_albedo": "maxsnowalb_modis",
    "soil_temperature": "soiltemp_1deg",
}
# Recognized selectors are the names actually represented by this builder's
# declared GEOG inventory.  Directory/resolution-looking aliases such as
# ``30s`` and ``modis_30s`` are not WPS selectors for that inventory and are
# rejected instead of being silently treated as ``default``.
_SUPPORTED_GEOG_TOKENS = frozenset({"5m", "default", "modis_lai"})


@dataclass(frozen=True)
class GeogSelection:
    """Resolved WPS geography datasets for one model domain.

    ``resolution_tokens`` preserves the ordered ``geog_data_res`` value
    declared by the WPS namelist path in :class:`CaseDataConfig`.  Dataset
    paths are always relative directories beneath that config's GEOG root.
    The recognized set is ``default``, ``5m``, and ``modis_lai``.  ``5m``
    selects NCAR's complete global low-resolution mandatory inventory;
    ``modis_lai`` is the available higher-resolution LAI override; and
    ``default`` selects the established Phase-3 inventory.

    :meth:`fallback` is intentionally the old code-constant selection and
    is used whenever a legacy caller supplies only ``geog_root``.
    """

    root: Path
    resolution_tokens: tuple[str, ...]
    terrain: str
    landuse: str
    soil_top: str
    soil_bottom: str
    greenfrac: str
    lai: str
    albedo: str
    snow_albedo: str
    soil_temperature: str

    @classmethod
    def fallback(cls, geog_root) -> "GeogSelection":
        return cls(root=Path(geog_root), resolution_tokens=("default",),
                   **_DEFAULT_GEOG_DIRS)

    @classmethod
    def from_tokens(cls, geog_root, tokens) -> "GeogSelection":
        """Resolve one WPS ``geog_data_res`` token string or sequence."""
        if isinstance(tokens, str):
            values = tokens.split("+")
        else:
            values = [part for value in tokens
                      for part in str(value).split("+")]
        normalized = tuple(value.strip().lower() for value in values
                           if value.strip())
        if not normalized:
            normalized = ("default",)
        unrecognized = tuple(token for token in normalized
                             if token not in _SUPPORTED_GEOG_TOKENS)
        if unrecognized:
            raise ValueError(
                f"geog_data_res contains unrecognized token(s) "
                f"{unrecognized!r}; recognized: "
                f"{sorted(_SUPPORTED_GEOG_TOKENS)}")

        # WPS applies the token list in priority order independently for each
        # field.  Resolve against the inventories this builder understands,
        # retaining the historical default when no earlier token supplies a
        # field.
        token_directories = {
            "default": _DEFAULT_GEOG_DIRS,
            "5m": _FIVE_MINUTE_GEOG_DIRS,
            "modis_lai": {"lai": "lai_modis_30s"},
        }
        directories = {}
        for field, fallback in _DEFAULT_GEOG_DIRS.items():
            directories[field] = next(
                (token_directories[token][field] for token in normalized
                 if field in token_directories[token]),
                fallback,
            )
        return cls(root=Path(geog_root), resolution_tokens=normalized,
                   **directories)

    @classmethod
    def from_case_data(cls, data, domain_id: int = 1) -> "GeogSelection":
        """Resolve a domain selection from a ``CaseDataConfig``.

        ``CaseDataConfig`` owns the GEOG root and the declared WPS namelist
        path.  A namelist without ``geog_data_res`` takes the WPS/default
        fallback, which also keeps minimal test and frozen-profile inputs
        compatible.
        """
        namelist = _parse_wps_namelist(data.wps_namelist)
        values = namelist.get("geog_data_res", ["default"])
        max_dom = int(namelist.get("max_dom", [1])[0])
        index = int(domain_id) - 1
        if index < 0:
            raise ValueError(f"domain_id must be positive, got {domain_id}")
        if index >= max_dom:
            raise ValueError(
                f"domain_id={domain_id} exceeds namelist.wps max_dom={max_dom}")
        # WPS v4.6 gridinfo_module.F initializes every array element to
        # 'default', then Fortran namelist assignment replaces only elements
        # explicitly listed.  A single non-default value therefore affects
        # d01 only; it is not broadcast to child domains.
        value = values[index] if index < len(values) else "default"
        return cls.from_tokens(data.geog_root, str(value))

    def path(self, field: str) -> Path:
        try:
            relative = getattr(self, field)
        except AttributeError as exc:
            raise KeyError(f"unknown GEOG field {field!r}") from exc
        return self.root / relative

    def landuse_global_attrs(self) -> dict[str, object]:
        """WRF global land-use attributes from the selected index file."""
        index = GeogDataset(self.path("landuse")).index
        required = {
            "MMINLU": index.mminlu,
            "ISWATER": index.iswater,
            "ISLAKE": index.islake,
            "ISICE": index.isice,
            "ISURBAN": index.isurban,
        }
        missing = [name for name, value in required.items()
                   if value in (None, "")]
        if missing:
            raise ValueError(
                f"selected land-use dataset {self.path('landuse')} lacks "
                f"required WRF metadata {missing}")
        return required


class _WpsLambert32:
    """Single-precision WPS map transforms used while selecting source cells.

    WPS declares the geogrid projection state and coordinate arrays as
    default ``real``.  The public :class:`LambertGrid` intentionally uses
    float64 setup math, but source points that lie almost exactly on an
    integer row/column can select a different WPS interpolation stencil.
    Recomputing just the geogrid sampling coordinates in float32 reproduces
    that control-flow convention; accumulated/output fields remain float64.
    """

    def __init__(self, grid: LambertGrid):
        f = np.float32
        self.f = f
        self.rad = f(np.pi / 180.0)
        self.deg = f(180.0 / np.pi)
        self.hemi = f(-1.0 if grid.truelat1 < 0.0 else 1.0)
        self.tl1 = f(grid.truelat1)
        self.tl2 = f(grid.truelat2)
        if abs(grid.truelat1 - grid.truelat2) > 0.1:
            num = (np.log10(np.cos(self.tl1 * self.rad))
                   - np.log10(np.cos(self.tl2 * self.rad)))
            den = (np.log10(np.tan((f(45.0) - np.abs(self.tl1) / f(2.0))
                                   * self.rad))
                   - np.log10(np.tan((f(45.0) - np.abs(self.tl2) / f(2.0))
                                     * self.rad)))
            self.cone = f(num / den)
        else:
            self.cone = f(np.sin(np.abs(self.tl1) * self.rad))
        self.rebydx = f(EARTH_RADIUS_M) / f(grid.dx)
        self.stand_lon = f(grid.stand_lon)
        dlon = f(grid.ref_lon) - self.stand_lon
        if dlon > f(180.0):
            dlon -= f(360.0)
        if dlon < f(-180.0):
            dlon += f(360.0)
        ctl1r = np.cos(self.tl1 * self.rad)
        self.rsw = f(
            self.rebydx * ctl1r / self.cone
            * (np.tan((f(90.0) * self.hemi - f(grid.ref_lat))
                      * self.rad / f(2.0))
               / np.tan((f(90.0) * self.hemi - self.tl1)
                        * self.rad / f(2.0))) ** self.cone)
        arg = self.cone * dlon * self.rad
        self.polei = f(self.hemi * f(grid.known_x)
                       - self.hemi * self.rsw * np.sin(arg))
        self.polej = f(self.hemi * f(grid.known_y)
                       + self.rsw * np.cos(arg))

    def ij_to_latlon(self, x, y):
        f = self.f
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        xx = self.hemi * x - self.polei
        yy = self.polej - self.hemi * y
        r2 = xx * xx + yy * yy
        r = np.sqrt(r2) / self.rebydx
        lon = (self.stand_lon
               + self.deg * np.arctan2(self.hemi * xx, yy) / self.cone)
        chi1 = (f(90.0) - self.hemi * self.tl1) * self.rad
        chi2 = (f(90.0) - self.hemi * self.tl2) * self.rad
        if chi1 == chi2:
            chi = f(2.0) * np.arctan(
                (r / np.tan(chi1)) ** (f(1.0) / self.cone)
                * np.tan(chi1 * f(0.5)))
        else:
            chi = f(2.0) * np.arctan(
                (r * self.cone / np.sin(chi1)) ** (f(1.0) / self.cone)
                * np.tan(chi1 * f(0.5)))
        lat = (f(90.0) - chi * self.deg) * self.hemi
        lat = np.where(r2 == f(0.0), f(90.0) * self.hemi, lat)
        # GNU Fortran's scalar evaluation of the WPS expression lands one
        # to two float32 ULPs poleward of NumPy's vector ufunc sequence.
        # Preserve that stencil-selecting result at exact source-grid rows.
        poleward = f(np.inf if self.hemi > 0.0 else -np.inf)
        lat = np.nextafter(np.nextafter(lat, poleward), poleward)
        lon = np.mod(lon + f(360.0), f(360.0))
        lon = np.where(lon > f(180.0), lon - f(360.0), lon)
        return lat.astype(np.float32), lon.astype(np.float32)

    def latlon_to_ij(self, lat, lon):
        f = self.f
        lat = np.asarray(lat, dtype=np.float32)
        lon = np.asarray(lon, dtype=np.float32)
        dlon = lon - self.stand_lon
        dlon = np.where(dlon > f(180.0), dlon - f(360.0), dlon)
        dlon = np.where(dlon < f(-180.0), dlon + f(360.0), dlon)
        ctl1r = np.cos(self.tl1 * self.rad)
        rm = (self.rebydx * ctl1r / self.cone
              * (np.tan((f(90.0) * self.hemi - lat)
                        * self.rad / f(2.0))
                 / np.tan((f(90.0) * self.hemi - self.tl1)
                          * self.rad / f(2.0))) ** self.cone)
        arg = self.cone * dlon * self.rad
        x = self.polei + self.hemi * rm * np.sin(arg)
        y = self.polej - rm * np.cos(arg)
        return (self.hemi * x).astype(np.float32), (self.hemi * y).astype(np.float32)

    def adopt_public_pole(self, grid):
        """Sub-kilometre nests: take the float64 grid's pole solution
        (WPS locates nests from their mass-grid centre)."""
        self.polei = np.float32(grid.polei)
        self.polej = np.float32(grid.polej)
        self.rebydx = np.float32(grid.rebydx)


class _WpsMerc32:
    """Single-precision WPS Mercator transforms (``set_merc``/
    ``ijll_merc``/``llij_merc`` in default REAL) used while selecting
    source cells.

    Unlike :class:`_WpsLambert32`, no GNU-scalar ULP nudges are applied:
    the band reconciliations in :class:`_DomainSampler` were measured
    against GNU-compiled geogrid output for Lambert grids only.
    Mercator sampling keeps the plain float32 transcription, so a source
    pixel sitting exactly on a cell boundary may select the neighbouring
    stencil relative to a GNU geogrid (sub-gate-tolerance differences;
    recorded in the worldwide handoff).
    """

    def __init__(self, grid):
        f = np.float32
        self.f = f
        self.rad = f(np.pi / 180.0)
        self.deg = f(180.0 / np.pi)
        self.lat1 = f(grid.ref_lat)
        self.lon1 = f(grid.ref_lon)
        self.knowni = f(grid.known_x)
        self.knownj = f(grid.known_y)
        clain = np.cos(self.rad * f(grid.truelat1))
        self.dlon = f(f(grid.dx) / (f(EARTH_RADIUS_M) * clain))
        self.rsw = f(0.0)
        if grid.ref_lat != 0.0:
            self.rsw = f(np.log(np.tan(f(0.5) * ((self.lat1 + f(90.0))
                                                 * self.rad))) / self.dlon)

    def ij_to_latlon(self, x, y):
        f = self.f
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        lat = (f(2.0) * np.arctan(np.exp(self.dlon
                                         * (self.rsw + y - self.knownj)))
               * self.deg - f(90.0))
        lon = (x - self.knowni) * self.dlon * self.deg + self.lon1
        lon = np.where(lon > f(180.0), lon - f(360.0), lon)
        lon = np.where(lon < f(-180.0), lon + f(360.0), lon)
        return lat.astype(np.float32), lon.astype(np.float32)

    def latlon_to_ij(self, lat, lon):
        f = self.f
        lat = np.asarray(lat, dtype=np.float32)
        lon = np.asarray(lon, dtype=np.float32)
        dlon = lon - self.lon1
        dlon = np.where(dlon < f(-180.0), dlon + f(360.0), dlon)
        dlon = np.where(dlon > f(180.0), dlon - f(360.0), dlon)
        i = self.knowni + (dlon / (self.dlon * self.deg))
        j = (self.knownj
             + np.log(np.tan(f(0.5) * ((lat + f(90.0)) * self.rad)))
             / self.dlon - self.rsw)
        return i.astype(np.float32), j.astype(np.float32)

    def adopt_public_pole(self, grid):
        """Sub-kilometre nests: take the float64 grid's derived state."""
        self.dlon = np.float32(grid.dlon)
        self.rsw = np.float32(grid.rsw)


class _WpsPs32:
    """Single-precision WPS polar-stereographic transforms (``set_ps``/
    ``ijll_ps``/``llij_ps`` in default REAL) used while selecting source
    cells.  Same no-nudge policy as :class:`_WpsMerc32`.
    """

    def __init__(self, grid):
        f = np.float32
        self.f = f
        self.rad = f(np.pi / 180.0)
        self.deg = f(180.0 / np.pi)
        self.hemi = f(-1.0 if grid.truelat1 < 0.0 else 1.0)
        self.tl1 = f(grid.truelat1)
        self.stand_lon = f(grid.stand_lon)
        self.rebydx = f(EARTH_RADIUS_M) / f(grid.dx)
        reflon = self.stand_lon + f(90.0)
        self.scale_top = f(1.0) + self.hemi * np.sin(self.tl1 * self.rad)
        ala1 = f(grid.ref_lat) * self.rad
        self.rsw = f(self.rebydx * np.cos(ala1) * self.scale_top
                     / (f(1.0) + self.hemi * np.sin(ala1)))
        alo1 = (f(grid.ref_lon) - reflon) * self.rad
        self.polei = f(f(grid.known_x) - self.rsw * np.cos(alo1))
        self.polej = f(f(grid.known_y)
                       - self.hemi * self.rsw * np.sin(alo1))

    def ij_to_latlon(self, x, y):
        f = self.f
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        reflon = self.stand_lon + f(90.0)
        xx = x - self.polei
        yy = (y - self.polej) * self.hemi
        r2 = xx * xx + yy * yy
        gi2 = (self.rebydx * self.scale_top) ** np.float32(2.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            lat = self.deg * self.hemi * np.arcsin((gi2 - r2) / (gi2 + r2))
            arccos = np.arccos(xx / np.sqrt(r2))
        lon = np.where(yy > f(0.0), reflon + self.deg * arccos,
                       reflon - self.deg * arccos)
        lat = np.where(r2 == f(0.0), self.hemi * f(90.0), lat)
        lon = np.where(r2 == f(0.0), reflon, lon)
        lon = np.where(lon > f(180.0), lon - f(360.0), lon)
        lon = np.where(lon < f(-180.0), lon + f(360.0), lon)
        return lat.astype(np.float32), lon.astype(np.float32)

    def latlon_to_ij(self, lat, lon):
        f = self.f
        lat = np.asarray(lat, dtype=np.float32)
        lon = np.asarray(lon, dtype=np.float32)
        reflon = self.stand_lon + f(90.0)
        ala = lat * self.rad
        rm = (self.rebydx * np.cos(ala) * self.scale_top
              / (f(1.0) + self.hemi * np.sin(ala)))
        alo = (lon - reflon) * self.rad
        i = self.polei + rm * np.cos(alo)
        j = self.polej + self.hemi * rm * np.sin(alo)
        return i.astype(np.float32), j.astype(np.float32)

    def adopt_public_pole(self, grid):
        """Sub-kilometre nests: take the float64 grid's pole solution."""
        self.polei = np.float32(grid.polei)
        self.polej = np.float32(grid.polej)
        self.rebydx = np.float32(grid.rebydx)


class _TranslatedWps32:
    """Index-offset delegation for a translated grid's float32 twin.

    A placement-translated grid (``ProjectedGrid.translated``) samples
    sources through its REFERENCE grid's float32 twin at exactly
    integer-shifted indices, so a cell two placements share selects its
    source stencil through the same float32 arithmetic and lands on the
    same source cells -- the sampling half of the statics-on-move
    bitwise-equality claim.  Rebuilding the twin from the shifted known
    point instead would re-round the pole in a different binade and
    could flip knife-edge stencil selections on shared ground.
    """

    def __init__(self, reference_twin, offset):
        self._twin = reference_twin
        self._di, self._dj = (float(offset[0]), float(offset[1]))

    def ij_to_latlon(self, x, y):
        return self._twin.ij_to_latlon(
            np.asarray(x, dtype=np.float32) + np.float32(self._di),
            np.asarray(y, dtype=np.float32) + np.float32(self._dj))

    def latlon_to_ij(self, lat, lon):
        x, y = self._twin.latlon_to_ij(lat, lon)
        return ((x - np.float32(self._di)).astype(np.float32),
                (y - np.float32(self._dj)).astype(np.float32))

    def adopt_public_pole(self, grid):
        reference = getattr(grid, "_translation_reference", None)
        self._twin.adopt_public_pole(
            grid if reference is None else reference)


def _wps32_for(grid):
    """The float32 WPS sampling twin for one projected grid."""
    from gpuwm.static.projection import MercatorGrid, PolarStereoGrid
    reference = getattr(grid, "_translation_reference", None)
    if reference is not None:
        return _TranslatedWps32(_wps32_for(reference),
                                grid._translation_offset)
    if isinstance(grid, LambertGrid):
        return _WpsLambert32(grid)
    if isinstance(grid, MercatorGrid):
        return _WpsMerc32(grid)
    if isinstance(grid, PolarStereoGrid):
        return _WpsPs32(grid)
    raise TypeError(f"no WPS float32 twin for {type(grid).__name__}")


# ---------------------------------------------------------------------------
# Point interpolators.  ``vals``: 2-D float64 window with NaN missing;
# ``xi, yi``: fractional 1-based source coordinates; ``x0, y0``: the window
# origin in the same coordinates.  All return float64 arrays shaped like
# ``xi`` with NaN where the option does not apply (fall through).
# ---------------------------------------------------------------------------

def _corner_indices(vals, xi, yi, x0, y0):
    fx = np.asarray(xi, dtype=np.float64) - x0
    fy = np.asarray(yi, dtype=np.float64) - y0
    ny, nx = vals.shape
    ok = (fx >= 0.0) & (fx <= nx - 1.0) & (fy >= 0.0) & (fy <= ny - 1.0)
    i0 = np.clip(np.floor(fx).astype(np.int64), 0, nx - 2)
    j0 = np.clip(np.floor(fy).astype(np.int64), 0, ny - 2)
    return fx, fy, i0, j0, ok


def four_pt(vals, xi, yi, x0=1, y0=1):
    """Bilinear on the 4 surrounding pixels; NaN if any is missing."""
    fx = np.asarray(xi, dtype=np.float64) - x0
    fy = np.asarray(yi, dtype=np.float64) - y0
    ny, nx = vals.shape
    i0 = np.floor(fx).astype(np.int64)
    i1 = np.ceil(fx).astype(np.int64)
    j0 = np.floor(fy).astype(np.int64)
    j1 = np.ceil(fy).astype(np.int64)
    ok = (i0 >= 0) & (i1 < nx) & (j0 >= 0) & (j1 < ny)
    i0 = np.clip(i0, 0, nx - 1)
    i1 = np.clip(i1, 0, nx - 1)
    j0 = np.clip(j0, 0, ny - 1)
    j1 = np.clip(j1, 0, ny - 1)
    wx, wy = fx - i0, fy - j0
    v00 = vals[j0, i0]
    v01 = vals[j0, i1]
    v10 = vals[j1, i0]
    v11 = vals[j1, i1]
    res = ((1.0 - wy) * ((1.0 - wx) * v00 + wx * v01)
           + wy * ((1.0 - wx) * v10 + wx * v11))
    return np.where(ok, res, np.nan)


def average_4pt(vals, xi, yi, x0=1, y0=1):
    """Mean of the valid pixels among the surrounding 2x2 (>= 1 required)."""
    xx = np.asarray(xi, dtype=np.float64)
    yy = np.asarray(yi, dtype=np.float64)
    ny, nx = vals.shape
    x1, y1 = x0 + nx - 1, y0 + ny - 1
    i0 = np.floor(xx).astype(np.int64)
    i1 = np.ceil(xx).astype(np.int64)
    j0 = np.floor(yy).astype(np.int64)
    j1 = np.ceil(yy).astype(np.int64)

    # WPS permits average_4pt up to half a source cell beyond a loaded tile,
    # clamping the pair of indices to the edge point (interp_module.F).
    lo = (xx > x0 - 0.5) & (i0 < x0)
    hi = (xx < x1 + 0.5) & (i1 > x1)
    i0 = np.where(lo, x0, np.where(hi, x1, i0))
    i1 = np.where(lo, x0, np.where(hi, x1, i1))
    lo = (yy > y0 - 0.5) & (j0 < y0)
    hi = (yy < y1 + 0.5) & (j1 > y1)
    j0 = np.where(lo, y0, np.where(hi, y1, j0))
    j1 = np.where(lo, y0, np.where(hi, y1, j1))
    ok = ((i0 >= x0) & (i1 <= x1) & (j0 >= y0) & (j1 <= y1))
    i0 -= x0
    i1 -= x0
    j0 -= y0
    j1 -= y0
    i0 = np.clip(i0, 0, nx - 1)
    i1 = np.clip(i1, 0, nx - 1)
    j0 = np.clip(j0, 0, ny - 1)
    j1 = np.clip(j1, 0, ny - 1)
    stack = np.stack([vals[j0, i0], vals[j0, i1],
                      vals[j1, i0], vals[j1, i1]])
    cnt = (~np.isnan(stack)).sum(axis=0)
    with np.errstate(invalid="ignore"):
        res = np.nansum(stack, axis=0) / np.maximum(cnt, 1)
    return np.where(ok & (cnt > 0), res, np.nan)


def average_16pt(vals, xi, yi, x0=1, y0=1):
    """WPS ``average_16pt``: mean valid values in the surrounding 4x4."""
    fx = np.asarray(xi, dtype=np.float64) - x0
    fy = np.asarray(yi, dtype=np.float64) - y0
    ny, nx = vals.shape
    i0 = np.floor(fx).astype(np.int64)
    j0 = np.floor(fy).astype(np.int64)
    ok = (i0 >= 1) & (i0 <= nx - 3) & (j0 >= 1) & (j0 <= ny - 3)
    i0c = np.clip(i0, 1, max(nx - 3, 1))
    j0c = np.clip(j0, 1, max(ny - 3, 1))
    stack = np.stack([
        vals[j0c + dj, i0c + di]
        for dj in (-1, 0, 1, 2)
        for di in (-1, 0, 1, 2)
    ])
    cnt = (~np.isnan(stack)).sum(axis=0)
    with np.errstate(invalid="ignore"):
        res = np.nansum(stack, axis=0) / np.maximum(cnt, 1)
    return np.where(ok & (cnt > 0), res, np.nan)


def _oned(x, a, b, c, d):
    """MM5/WPS overlapping parabolic in 1-D (all points valid)."""
    return ((1.0 - x) * (b + x * (0.5 * (c - a) + x * (0.5 * (c + a) - b)))
            + x * (c + (1.0 - x) * (0.5 * (b - d)
                                    + (1.0 - x) * (0.5 * (b + d) - c))))


def sixteen_pt(vals, xi, yi, x0=1, y0=1):
    """Overlapping parabolic on the surrounding 4x4; NaN if any missing."""
    fx = np.asarray(xi, dtype=np.float64) - x0
    fy = np.asarray(yi, dtype=np.float64) - y0
    ny, nx = vals.shape
    i0 = np.floor(fx).astype(np.int64)
    j0 = np.floor(fy).astype(np.int64)
    ok = (i0 >= 1) & (i0 <= nx - 3) & (j0 >= 1) & (j0 <= ny - 3)
    i0c = np.clip(i0, 1, max(nx - 3, 1))
    j0c = np.clip(j0, 1, max(ny - 3, 1))
    x, y = fx - i0c, fy - j0c
    rows = [_oned(x, *(vals[j0c + l - 1, i0c + k - 1] for k in range(4)))
            for l in range(4)]
    res = _oned(y, *rows)
    return np.where(ok, res, np.nan)


def search_nearest(vals, xi, yi, x0=1, y0=1):
    """WPS ``search``: breadth-first extrapolation from ``nint(x),nint(y)``.

    Once the first valid point is popped, WPS compares it with the valid
    points already queued and chooses the Euclidean-closest of that finite
    frontier.  It does not perform an unrestricted Euclidean nearest-neighbor
    search; the distinction matters in sizeable missing-data holes.
    """
    fx = np.atleast_1d(np.asarray(xi, dtype=np.float64) - x0)
    fy = np.atleast_1d(np.asarray(yi, dtype=np.float64) - y0)
    ny, nx = vals.shape
    valid = ~np.isnan(vals)
    out = np.full(fx.shape, np.nan)
    if not valid.any():
        return out
    for n in range(fx.size):
        px, py = fx.flat[n], fy.flat[n]
        ic, jc = int(np.floor(px + 0.5)), int(np.floor(py + 0.5))
        if ic < 0 or ic >= nx or jc < 0 or jc >= ny:
            continue
        q = deque([(ic, jc, 0)])
        seen = {jc * nx + ic}
        found = None
        while q and found is None:
            i, j, depth = q.popleft()
            if valid[j, i]:
                found = (i, j)
            if depth < 1200:  # WPS default maximum search depth
                for ii, jj in ((i - 1, j), (i + 1, j),
                               (i, j - 1), (i, j + 1)):
                    key = jj * nx + ii
                    if (0 <= ii < nx and 0 <= jj < ny and key not in seen):
                        seen.add(key)
                        q.append((ii, jj, depth + 1))
        if found is None:
            continue
        bi, bj = found
        best_d2 = (bi - px) ** 2 + (bj - py) ** 2
        for i, j, _ in q:
            if valid[j, i]:
                d2 = (i - px) ** 2 + (j - py) ** 2
                if d2 < best_d2:
                    bi, bj, best_d2 = i, j, d2
        out.flat[n] = vals[bj, bi]
    return out


_INTERP_OPS = {"four_pt": four_pt, "average_4pt": average_4pt,
               "average_16pt": average_16pt, "sixteen_pt": sixteen_pt,
               "search": search_nearest}


def _interp_seq(vals, xi, yi, x0, y0, seq):
    """Apply interpolation options in order, each filling remaining NaNs."""
    out = np.full(np.shape(xi), np.nan)
    for name in seq:
        todo = np.isnan(out)
        if not todo.any():
            break
        got = _INTERP_OPS[name](vals, np.asarray(xi)[todo],
                                np.asarray(yi)[todo], x0, y0)
        out[todo] = got
    return out


# ---------------------------------------------------------------------------
# Smoothers (geogrid smooth_module conventions: per pass an x sweep then a
# y sweep, one-cell boundaries untouched by the sweep in its direction).
# ---------------------------------------------------------------------------

def _one_pass(a, coef):
    a = np.asarray(a, dtype=np.float64)
    out = a.copy()
    out[:, 1:-1] = (a[:, 1:-1]
                    + coef * (0.5 * (a[:, :-2] + a[:, 2:]) - a[:, 1:-1]))
    mid = out
    out = mid.copy()
    out[1:-1, :] = (mid[1:-1, :]
                    + coef * (0.5 * (mid[:-2, :] + mid[2:, :])
                              - mid[1:-1, :]))
    return out


def one_two_one(a, passes=1):
    """1-2-1 smoother."""
    for _ in range(int(passes)):
        a = _one_pass(a, 0.5)
    return a


def smth_desmth(a, passes=1):
    """Smoother-desmoother: 0.50 pass followed by a -0.52 pass."""
    for _ in range(int(passes)):
        a = _one_pass(a, 0.5)
        a = _one_pass(a, -0.52)
    return a


def smth_desmth_special(a, passes=1):
    """WPS terrain smoother, restoring newly negative source points."""
    original = np.asarray(a, dtype=np.float64)
    out = smth_desmth(original, passes=passes)
    restore = (original >= 0.0) & (out < 0.0)
    out[restore] = original[restore]
    return out


# ---------------------------------------------------------------------------
# Landmask / dominant-category rules (pinned exactly by geo_em d01).
# ---------------------------------------------------------------------------

def landmask_from_landusef(luf, iswater, islake=None):
    """LANDMASK: 0 where the water fraction is >= 0.5, else 1."""
    w = luf[iswater - 1].copy()
    if islake is not None:
        w += luf[islake - 1]
    return np.where(w >= 0.5, 0.0, 1.0)


def lu_index_from_landusef(luf, landmask, iswater, islake=None):
    """Dominant landuse: water cells get the dominant water type (lake only
    when strictly larger than ocean); land cells get the dominant land
    category, water types excluded; ties go to the lowest category."""
    masked = np.array(luf, dtype=np.float64, copy=True)
    masked[iswater - 1] = -1.0
    if islake is not None:
        masked[islake - 1] = -1.0
    dom_land = np.argmax(masked, axis=0) + 1.0
    if islake is not None:
        dom_water = np.where(luf[islake - 1] > luf[iswater - 1],
                             float(islake), float(iswater))
    else:
        dom_water = np.full(luf.shape[1:], float(iswater))
    return np.where(landmask < 0.5, dom_water, dom_land)


def dominant_category(frac):
    """Plain dominant category (argmax + 1; lowest category wins ties)."""
    return np.argmax(frac, axis=0) + 1.0


# ---------------------------------------------------------------------------
# The builder
# ---------------------------------------------------------------------------

class _DomainSampler:
    """Shared machinery: extended grid, source windows, pixel binning."""

    def __init__(self, grid, halo: int = HALO):
        self.grid = grid
        self._wps_grid = _wps32_for(grid)
        is_lambert = isinstance(grid, LambertGrid)
        self.halo = halo
        self.nx = grid.e_we - 1
        self.ny = grid.e_sn - 1
        self.nxe = self.nx + 2 * halo
        self.nye = self.ny + 2 * halo
        xs = np.arange(1 - halo, self.nx + halo + 1, dtype=np.float64)
        ys = np.arange(1 - halo, self.ny + halo + 1, dtype=np.float64)
        X, Y = np.meshgrid(xs, ys)
        if grid.dx < 1000.0:
            # WPS locates nests from their mass-grid centre.  The public
            # float64 projection already carries the resulting pole in child
            # coordinates; using it avoids re-solving the pole from a rounded
            # child-corner latitude/longitude.
            self._wps_grid.adopt_public_pole(grid)
        self.lat_e, lon32 = self._wps_grid.ij_to_latlon(X, Y)
        lat64, lon64 = grid.ij_to_latlon(X, Y)
        self._lat_lower_e = np.nextafter(self.lat_e, np.float32(-np.inf))
        if is_lambert and grid.dx < 1000.0:
            # One of the two scalar-libm ULPs documented in ij_to_latlon is
            # absorbed when geogrid initializes a nest from its centre.
            self.lat_e = self._lat_lower_e
            self._lat_lower_e = np.nextafter(
                self.lat_e, np.float32(-np.inf))
        # GNU optimization only becomes stencil-selecting at the deeply
        # nested sub-kilometre spacing; coarse domains use the public value.
        # The compiler-band reconciliations below were measured against
        # GNU geogrid output for Lambert grids only; other projections
        # sample with the plain transcriptions (bands stay empty).
        self.lon_e = (self._geogrid_longitude(lon32, lon64)
                      if is_lambert and grid.dx < 1000.0 else lon64)
        if is_lambert:
            lon_public = lon64.astype(np.float32)
            lon_ulp = np.abs(np.spacing(lon_public).astype(np.float64))
            lon_frac = np.divide(
                lon64 - lon_public.astype(np.float64), lon_ulp,
                out=np.zeros_like(lon64), where=lon_ulp != 0.0)
            lon_band = lon32.view(np.int32) - lon_public.view(np.int32)
            self._lon_boundary_band = (
                (np.abs(lon_band) == 2)
                & (lon_frac >= -0.15) & (lon_frac <= -0.05))
            lat_public = lat64.astype(np.float32)
            lat_ulp = np.abs(np.spacing(lat_public).astype(np.float64))
            lat_frac = np.divide(
                lat64 - lat_public.astype(np.float64), lat_ulp,
                out=np.zeros_like(lat64), where=lat_ulp != 0.0)
            lat_band = (self._lat_lower_e.view(np.int32)
                        - lat_public.view(np.int32))
            self._lat_integer_band = (
                (lat_band == 4) & (lat_frac >= 0.38) & (lat_frac <= 0.41))
        else:
            self._lon_boundary_band = np.zeros(lon64.shape, dtype=bool)
            self._lat_integer_band = np.zeros(lat64.shape, dtype=bool)
        # cell-corner mesh of the extended grid, for window bounds
        Xc, Yc = np.meshgrid(np.arange(0.5 - halo, self.nx + halo + 1.0),
                             np.arange(0.5 - halo, self.ny + halo + 1.0))
        self.lat_c, _ = self._wps_grid.ij_to_latlon(Xc, Yc)
        _, self.lon_c = grid.ij_to_latlon(Xc, Yc)
        self.crop = (slice(halo, halo + self.ny),
                     slice(halo, halo + self.nx))
        self._cells_cache: dict[tuple, np.ndarray] = {}

    @staticmethod
    def _geogrid_longitude(lon32, lon64):
        """Mirror GNU WPS's optimized single-precision longitude result.

        NumPy's vector ufunc sequence and geogrid's optimized scalar Lambert
        expression can land several float32 ULPs apart.  Bracketing the WPS
        expression with the public float64 transform identifies which of the
        compiler's three evaluation bands applies; moving east by that band
        preserves geogrid's half-cell decisions on deeply nested grids.
        """
        c = np.asarray(lon32, dtype=np.float32)
        d = np.asarray(lon64, dtype=np.float64)
        p = d.astype(np.float32)
        ci = c.view(np.int32)
        pi = p.view(np.int32)
        ulp = np.abs(np.spacing(p).astype(np.float64))
        frac = np.divide(d - p.astype(np.float64), ulp,
                         out=np.zeros_like(d), where=ulp != 0.0)
        band = ci - pi
        steps = np.full(c.shape, 4, dtype=np.int8)
        steps[(band <= -3) | ((band == -2) & (frac < -0.05))] = 0
        steps[(band >= 3) | ((band == 2) & (frac >= -0.05))] = 8
        out = c.copy()
        for n in range(8):
            move = steps > n
            out[move] = np.nextafter(out[move], np.float32(np.inf))
        return out

    def res_ratio(self, ds: GeogDataset) -> float:
        """How many times finer the source is than the model grid."""
        return self.grid.dx / (M_PER_DEG * abs(ds.index.dx))

    def window(self, ds: GeogDataset, margin: int = 3) -> GeogWindow:
        """Read the source window covering the extended grid + margin."""
        x, y = ds.latlon_to_xy(self.lat_c, self.lon_c)
        if ds.wraps_x and float(x.max() - x.min()) > ds.nx_global / 2.0:
            x = np.where(x < ds.nx_global / 2.0, x + ds.nx_global, x)
        x0 = int(np.floor(x.min())) - margin
        x1 = int(np.ceil(x.max())) + margin
        y0 = max(1, int(np.floor(y.min())) - margin)
        y1 = min(ds.ny_global, int(np.ceil(y.max())) + margin)
        return ds.read_window(x0, x1, y0, y1)

    @staticmethod
    def require_source_coverage(
            ds: GeogDataset, win: GeogWindow, *,
            field: str) -> dict[str, object]:
        """Prove every source cell in a mandatory static window is tiled.

        ``GeogDataset.read_window`` intentionally permits absent tiles when
        an index declares a sparse staging tree.  Static fields are dense
        model inputs, however: an absent staged tile must not become a
        physical zero or an empty category vector.  This gate concerns tile
        presence only.  Missing-value sentinels inside present source tiles
        retain their normal WPS interpolation/fallback semantics.
        """

        coverage = win.coverage
        expected_shape = win.raw.shape[1:]
        if coverage is None or coverage.shape != expected_shape:
            raise ValueError(
                f"WPS GEOG coverage metadata for mandatory field {field!r} "
                f"has shape {None if coverage is None else coverage.shape}; "
                f"expected {expected_shape}")
        coverage = np.asarray(coverage, dtype=bool)
        required_cells = int(coverage.size)
        covered_cells = int(np.count_nonzero(coverage))
        missing_cells = required_cells - covered_cells
        extent = ds._extent_mask(win.x0, win.x1, win.y0, win.y1)
        missing_tile_cells = int(np.count_nonzero(~coverage & extent))
        outside_extent_cells = int(np.count_nonzero(~coverage & ~extent))
        missing_tiles = ds.missing_tiles(
            win.x0, win.x1, win.y0, win.y1)
        required_origins = ds.required_tile_origins(
            win.x0, win.x1, win.y0, win.y1)

        if missing_cells:
            first_j, first_i = np.argwhere(~coverage)[0]
            source_x = win.x0 + int(first_i)
            source_y = win.y0 + int(first_j)
            if ds.wraps_x:
                source_x = (source_x - 1) % ds.nx_global + 1
            preview = [list(origin) for origin in missing_tiles[:16]]
            suffix = "..." if len(missing_tiles) > 16 else ""
            raise FileNotFoundError(
                "WPS GEOG mandatory source coverage failed for field "
                f"{field!r} in {ds.path.resolve()}: covered "
                f"{covered_cells}/{required_cells} source cells; first "
                f"uncovered source index=(x={source_x}, y={source_y}); "
                f"missing_tile_cells={missing_tile_cells}; "
                f"outside_extent_cells={outside_extent_cells}; "
                f"missing_tile_origins={preview}{suffix}; source_window="
                f"x={win.x0}..{win.x1},y={win.y0}..{win.y1}; "
                f"declared_sparse={ds.declared_sparse}. A sparse staging "
                "declaration does not permit absent tiles required by a "
                "model domain.")

        required_tiles = []
        for origin in required_origins:
            path = ds.tiles.get(origin)
            if path is None:
                raise AssertionError(
                    f"coverage passed but required GEOG tile {origin} is absent")
            required_tiles.append({
                "origin": list(origin),
                "relative_path": path.name,
                "bytes": path.stat().st_size,
            })
        return {
            "schema": "gpuwm-geog-source-coverage-v1",
            "status": "PASS",
            "field": str(field),
            "dataset": str(ds.path.resolve()),
            "declared_sparse": bool(ds.declared_sparse),
            "source_geometry": {
                "nx_global": int(ds.nx_global),
                "ny_global": int(ds.ny_global),
                "wraps_x": bool(ds.wraps_x),
                "extent_basis": str(ds.extent_basis),
                "tile_inventory_bounds": [
                    int(value) for value in ds.tile_inventory_bounds],
            },
            "source_window": {
                "x_start": int(win.x0), "x_end": int(win.x1),
                "y_start": int(win.y0), "y_end": int(win.y1),
            },
            "required_cells": required_cells,
            "covered_cells": covered_cells,
            "missing_tile_cells": 0,
            "outside_extent_cells": 0,
            "coverage_fraction": 1.0,
            "required_tile_count": len(required_tiles),
            "required_tiles": required_tiles,
        }

    def cell_coords(self, ds: GeogDataset, win: GeogWindow):
        """Extended-grid mass points in window source coordinates."""
        if self.grid.dx >= 1000.0:
            xi, yi = ds.latlon_to_xy(self.lat_e, self.lon_e)
            if ds.wraps_x:
                xi = np.where(xi < win.x0 - 0.5,
                              xi + ds.nx_global, xi)
            return xi, yi

        # WPS regular_ll map state and operands are default REAL.  Preserve
        # its operation ordering: subtract, divide, add, then one wrap check.
        f = np.float32
        idx = ds.index
        dlon = self.lon_e.astype(f) - f(idx.known_lon)
        dlat = self.lat_e.astype(f) - f(idx.known_lat)
        xi = dlon / f(idx.dx)
        yi = dlat / f(idx.dy)
        xi = (xi + f(idx.known_x)).astype(f)
        yi = (yi + f(idx.known_y)).astype(f)
        yi_lower = ((self._lat_lower_e.astype(f) - f(idx.known_lat))
                    / f(idx.dy) + f(idx.known_y)).astype(f)

        # At high zoom, reconcile only the compiler bands that straddle an
        # interpolation-control boundary.  This mirrors the scalar geogrid
        # result without perturbing ordinary bilinear coordinates.
        yint = np.rint(yi).astype(f)
        use_lower = (self._lat_integer_band
                     & (yi > yint) & (yi_lower < yint)
                     & ((yi - yint) < f(0.002)))
        tile_y = f(idx.tile_y)
        ytile = (np.floor((yi - f(0.5)) / tile_y + f(0.5))
                 * tile_y + f(0.5))
        use_lower |= ((yi > ytile) & (yi_lower < ytile)
                      & ((yi - ytile) < f(0.002)))
        yi = np.where(use_lower, yi_lower, yi)

        xint = np.rint(xi).astype(f)
        xdist = xint - xi
        use_east = (self._lon_boundary_band
                    & (xdist > f(0.0)) & (xdist < f(0.0045)))
        xi = np.where(use_east, np.nextafter(xint, f(np.inf)), xi)
        if ds.wraps_x:
            xi = np.where(xi < f(0.5), xi + f(ds.nx_global), xi)
            xi = np.where(xi >= f(ds.nx_global) + f(0.5),
                          xi - f(ds.nx_global), xi)
        if ds.wraps_x:
            xi = np.where(xi < win.x0 - 0.5, xi + ds.nx_global, xi)
        return xi, yi

    def _interp_tile_sequence(self, ds, z, xi, yi, seq):
        """Interpolate points within the one native tile selected by WPS."""
        shape = np.shape(xi)
        coord_dtype = np.float32 if self.grid.dx < 1000.0 else np.float64
        xx = np.asarray(xi, dtype=coord_dtype).ravel()
        yy = np.asarray(yi, dtype=coord_dtype).ravel()
        if ds.wraps_x:
            xx = np.where(xx >= ds.nx_global + 0.5,
                          xx - ds.nx_global, xx)
            xx = np.where(xx < 0.5, xx + ds.nx_global, xx)
        tx, ty = ds.index.tile_x, ds.index.tile_y
        xs = np.floor((xx - np.float32(0.5)) / np.float32(tx)).astype(np.int64)
        ys = np.floor((yy - np.float32(0.5)) / np.float32(ty)).astype(np.int64)
        xs = xs * tx + 1
        ys = ys * ty + 1
        out = np.full(xx.shape, np.nan)
        groups = np.stack((xs, ys), axis=1)
        for origin in np.unique(groups, axis=0):
            pick = (xs == origin[0]) & (ys == origin[1])
            tile = ds.read_tile_window(int(origin[0]), int(origin[1]))
            if tile is None:
                continue
            out[pick] = _interp_seq(tile.values(z), xx[pick], yy[pick],
                                    tile.x0, tile.y0, seq)
        return out.reshape(shape)

    def pixel_cells(self, ds: GeogDataset, win: GeogWindow) -> np.ndarray:
        """Flat extended-cell index for every window pixel (-1 = outside).

        A pixel belongs to the model cell whose centre is nearest
        (Fortran ``nint`` of its model-grid coordinates) -- geogrid's
        accumulation rule.
        """
        idx = ds.index
        key = (idx.dx, idx.dy, idx.known_x, idx.known_y, idx.known_lat,
               idx.known_lon, win.x0, win.y0, win.raw.shape)
        hit = self._cells_cache.get(key)
        if hit is not None:
            return hit
        nyw, nxw = win.raw.shape[1:]
        flat = np.full((nyw, nxw), -1, dtype=np.int64)
        xs_abs = win.x0 + np.arange(nxw, dtype=np.float64)
        chunk = max(1, int(4e6) // max(nxw, 1))
        for j0 in range(0, nyw, chunk):
            j1 = min(j0 + chunk, nyw)
            yy = win.y0 + np.arange(j0, j1, dtype=np.float64)
            lat, lon = ds.xy_to_latlon(xs_abs[None, :], yy[:, None])
            # The accumulation path was already oracle-matched in float64;
            # only target-point stencil selection needs WPS's real precision.
            gx, gy = self.grid.latlon_to_ij(lat, lon)
            # Default-REAL Lambert math can leave an analytically half-cell
            # coordinate a few 1e-5 away from the boundary before NINT.
            hx = np.floor(gx) + 0.5
            hy = np.floor(gy) + 0.5
            gx = np.where(np.abs(gx - hx) < 5e-5, hx, gx)
            gy = np.where(np.abs(gy - hy) < 5e-5, hy, gy)
            ei = np.floor(gx + 0.5).astype(np.int64) + (self.halo - 1)
            ej = np.floor(gy + 0.5).astype(np.int64) + (self.halo - 1)
            ok = ((ei >= 0) & (ei < self.nxe)
                  & (ej >= 0) & (ej < self.nye))
            flat[j0:j1][ok] = (ej * self.nxe + ei)[ok]
        flat = flat.ravel()
        self._cells_cache = {key: flat}          # keep only the last mapping
        return flat

    # -- field builders ------------------------------------------------------

    def accum_mean(self, flat, vals) -> np.ndarray:
        v = vals.ravel()
        ok = (flat >= 0) & ~np.isnan(v)
        n = self.nxe * self.nye
        sums = np.bincount(flat[ok], weights=v[ok], minlength=n)
        cnt = np.bincount(flat[ok], minlength=n)
        out = np.full(n, np.nan)
        got = cnt > 0
        out[got] = sums[got] / cnt[got]
        return out.reshape(self.nye, self.nxe)

    def continuous(self, ds, win, z=0, seq=("four_pt", "average_4pt"),
                   fill=0.0, gcell=True, active=None) -> np.ndarray:
        """One continuous level on the extended grid: grid-cell average when
        the source is fine enough, else/fallback the interp sequence, then
        ``fill`` for anything still missing."""
        vals = win.values(z)
        if gcell and self.res_ratio(ds) >= GCELL_RATIO:
            out = self.accum_mean(self.pixel_cells(ds, win), vals)
        else:
            out = np.full((self.nye, self.nxe), np.nan)
        if active is None:
            active = np.ones(out.shape, dtype=bool)
        else:
            active = np.asarray(active, dtype=bool)
            out[~active] = fill
        todo = np.isnan(out) & active
        if todo.any():
            xi, yi = self.cell_coords(ds, win)
            out[todo] = self._interp_tile_sequence(
                ds, z, xi[todo], yi[todo], seq)
        out[np.isnan(out)] = fill
        return out

    def categorical(self, ds, win, *, fractional_gcell=False) -> np.ndarray:
        """Return category fractions with shape ``(ncat, nye, nxe)``.

        WPS supports two source representations.  A ``categorical`` source
        stores one integer category per pixel; those pixels are accumulated
        into model cells and normalized.  A ``continuous`` source stores one
        fractional plane per category; each plane follows its continuous
        interpolation rule and the interpolated vector is then normalized,
        as geogrid's ``process_tile_module`` does.

        ``fractional_gcell`` enables the leading ``average_gcell(4.0)`` in
        LANDUSEF's interpolation sequence.  Soil category fractions use only
        ``four_pt`` and therefore leave it disabled.
        """
        idx = ds.index
        cmin, cmax = idx.category_min, idx.category_max
        if cmin is None or cmax is None or cmax < cmin:
            raise ValueError(
                "category source requires a valid category_min/category_max")
        ncat = cmax - cmin + 1

        if idx.type == "continuous":
            if idx.nz != ncat:
                raise ValueError(
                    f"continuous category source declares {idx.nz} z planes "
                    f"for {ncat} categories ({cmin}..{cmax})")
            fields = np.stack([
                self.continuous(
                    ds, win, z, seq=("four_pt",), fill=0.0,
                    gcell=fractional_gcell)
                for z in range(ncat)
            ]).astype(np.float32)
            # WPS holds both the planes and their per-cell sum in default
            # REAL.  Cells with no valid category support remain all-zero.
            total = fields.sum(axis=0, dtype=np.float32)
            frac = np.zeros_like(fields)
            np.divide(fields, total[None, :, :], out=frac,
                      where=total[None, :, :] > np.float32(0.0))
            return frac.astype(np.float64)

        if idx.type != "categorical":
            raise ValueError(
                f"unsupported category source type {idx.type!r}")
        if idx.nz != 1:
            raise ValueError(
                f"categorical source must have one z plane, got {idx.nz}")
        flat = self.pixel_cells(ds, win)
        r = win.raw[0].ravel().astype(np.int64)
        ok = (flat >= 0) & (r >= cmin) & (r <= cmax)
        n = self.nxe * self.nye
        counts = np.bincount(flat[ok] * ncat + (r[ok] - cmin),
                             minlength=n * ncat).astype(np.float64)
        counts = counts.reshape(self.nye, self.nxe, ncat)
        tot = counts.sum(axis=2)
        # WPS stores this working field as default REAL; its optimized loop
        # forms one float32 reciprocal per cell and multiplies each category.
        denom = np.maximum(tot, 1.0).astype(np.float32)
        recip = np.float32(1.0) / denom
        frac = (counts.astype(np.float32) * recip[:, :, None]).astype(np.float64)
        empty = tot == 0
        if empty.any():
            xi, yi = self.cell_coords(ds, win)
            ii = (np.floor(xi[empty] + 0.5).astype(np.int64) - win.x0)
            jj = (np.floor(yi[empty] + 0.5).astype(np.int64) - win.y0)
            nyw, nxw = win.raw.shape[1:]
            inside = (ii >= 0) & (ii < nxw) & (jj >= 0) & (jj < nyw)
            cat = np.full(ii.shape, -1, dtype=np.int64)
            cat[inside] = win.raw[0][jj[inside], ii[inside]].astype(np.int64)
            good = (cat >= cmin) & (cat <= cmax)
            ej, ei = np.nonzero(empty)
            frac[ej[good], ei[good], cat[good] - cmin] = 1.0
        return np.moveaxis(frac, 2, 0)


def monthly_interp_to_date(monthly, valid_time):
    """WRF real's ``monthly_interp_to_date`` for 12-monthly climatologies.

    Transcription of dyn_em/module_initialize_real.F:8023-8089 (v4.6.1):
    monthly values are anchored on the 15th of each month at the run
    year's Julian day; the year wrap uses fictitious anchors exactly 31
    days before Jan 15 and after Dec 15 (:8059-8063).  The valid time is
    reduced to ``year*1000 + julian_day`` — WHOLE days, the clock time is
    discarded with get_julgmt's fractional ``gmt`` (:8065-8066) — and the
    two bracketing mid-months (``middle(l) < target <= middle(l+1)``,
    :8067-8068) blend linearly with integer-day weights
    ``(target - middle(l))`` on the later month and
    ``(middle(l+1) - target)`` on the earlier one (:8080-8082).

    ``monthly`` is ``(12, ...)`` with index 0 = January.
    """
    from datetime import datetime as _datetime

    monthly = np.asarray(monthly, dtype=np.float64)
    if monthly.shape[0] != 12:
        raise ValueError("monthly climatology must have a leading 12-axis")
    year = valid_time.year
    middle = np.empty(14, dtype=np.int64)
    for month in range(1, 13):
        middle[month] = (year * 1000
                         + _datetime(year, month, 15).timetuple().tm_yday)
    middle[0] = middle[1] - 31
    middle[13] = middle[12] + 31
    target = year * 1000 + valid_time.timetuple().tm_yday
    for anchor in range(13):
        if middle[anchor] < target <= middle[anchor + 1]:
            if anchor in (0, 12):
                month1, month2 = 12, 1
            else:
                month1, month2 = anchor, anchor + 1
            return ((monthly[month2 - 1] * float(target - middle[anchor])
                     + monthly[month1 - 1] * float(middle[anchor + 1] - target))
                    / float(middle[anchor + 1] - middle[anchor]))
    raise ValueError(f"no mid-month interval brackets {valid_time!r}")


def build_static(
        grid, geog_root, halo: int = HALO, *,
        selection: GeogSelection | None = None,
        source_coverage_report: MutableMapping[str, object] | None = None,
) -> dict:
    """Build all Phase 3 static fields for ``grid`` from a WPS_GEOG tree.

    Returns a dict of float64 arrays keyed by geo_em variable names
    (fraction/monthly fields lead with their category/month axis), plus
    ``TMN`` (elevation-corrected deep soil temperature, real.exe input).

    ``selection`` resolves WPS resolution tokens to dataset directories.
    Omitting it uses :meth:`GeogSelection.fallback`, the exact historical
    code-constant inventory used by frozen verification profiles.
    """
    root = Path(geog_root)
    selection = (GeogSelection.fallback(root) if selection is None
                 else selection)
    if selection.root != root:
        raise ValueError(
            f"GeogSelection root {selection.root} does not match "
            f"geog_root {root}")
    dom = _DomainSampler(grid, halo)
    crop = dom.crop
    out: dict[str, np.ndarray] = {}

    def require_coverage(field: str, ds: GeogDataset,
                         win: GeogWindow) -> None:
        evidence = dom.require_source_coverage(ds, win, field=field)
        if source_coverage_report is not None:
            if field in source_coverage_report:
                raise ValueError(
                    f"duplicate static source-coverage field {field!r}")
            source_coverage_report[field] = evidence

    # --- terrain: average_gcell(4.0)+four_pt+average_4pt, fill 0, one
    #     smoother-desmoother pass (choice arbitrated by geo_em HGT_M) -----
    topo = GeogDataset(selection.path("terrain"))
    win = dom.window(topo)
    require_coverage("terrain", topo, win)
    hgt_e = dom.continuous(topo, win, 0)
    hgt_e = smth_desmth_special(hgt_e, passes=1)
    out["HGT_M"] = hgt_e[crop]
    del win

    # --- landuse -> LANDUSEF / LANDMASK / LU_INDEX ------------------------
    lu_ds = GeogDataset(selection.path("landuse"))
    iswater = lu_ds.index.iswater or 17
    islake = lu_ds.index.islake
    win = dom.window(lu_ds)
    require_coverage("landuse", lu_ds, win)
    luf_e = dom.categorical(lu_ds, win, fractional_gcell=True)
    luf = luf_e[(slice(None),) + crop]
    out["LANDUSEF"] = luf
    landmask_e = landmask_from_landusef(luf_e, iswater, islake)
    out["LANDMASK"] = landmask_e[crop]
    out["LU_INDEX"] = lu_index_from_landusef(luf, out["LANDMASK"],
                                             iswater, islake)
    del win

    # --- soil categories ---------------------------------------------------
    for name, dom_name, field in (
            ("SOILCTOP", "SCT_DOM", "soil_top"),
            ("SOILCBOT", "SCB_DOM", "soil_bottom")):
        ds = GeogDataset(selection.path(field))
        win = dom.window(ds)
        require_coverage(field, ds, win)
        frac = dom.categorical(ds, win)[(slice(None),) + crop]
        out[name] = frac
        out[dom_name] = dominant_category(frac)
        del win

    water = out["LANDMASK"] == 0.0
    water_e = landmask_e == 0.0

    # --- monthly climatologies + masked scalars ---------------------------
    def monthly(field, seq, fill, mask_fill):
        ds = GeogDataset(selection.path(field))
        win = dom.window(ds)
        require_coverage(field, ds, win)
        months = np.stack([dom.continuous(ds, win, z, seq, fill,
                                         active=~water_e)[crop]
                           for z in range(ds.index.nz)])
        months[:, water] = mask_fill
        return months

    seq = ("four_pt", "average_4pt", "average_16pt", "search")
    out["GREENFRAC"] = monthly("greenfrac", seq, 0.0, 0.0)
    # GEOGRID.TBL.ARW's ``default`` dataset is the pre-aggregated 10-minute
    # product.  The 30-second tree remains readable through GeogDataset and
    # is selected only by WPS's explicit ``modis_lai`` resolution token.
    out["LAI12M"] = monthly("lai", seq, 0.0, 0.0)
    out["ALBEDO12M"] = monthly("albedo", seq, 8.0, 8.0)
    out["SNOALB"] = monthly("snow_albedo", seq, 0.0, 0.0)[0]
    out["SOILTEMP"] = monthly(
        "soil_temperature",
        ("sixteen_pt", "four_pt", "average_4pt", "average_16pt", "search"),
        0.0, 0.0)[0]

    # --- deep soil temperature, elevation-corrected (real.exe input;
    #     WRF share/module_soil_pre.F:973, land only) ----------------------
    out["TMN"] = np.where(out["LANDMASK"] > 0.5,
                          out["SOILTEMP"] - 0.0065 * out["HGT_M"],
                          out["SOILTEMP"])
    return out


_CATALOG_STATIC_CACHE: dict[tuple, dict[str, np.ndarray]] = {}


def geog_selection_from_catalog(catalog, domain_id: int) -> GeogSelection:
    """Recover one domain's resolved GEOG selection from preflight inputs."""
    domain_id = int(domain_id)
    if domain_id < 1:
        raise ValueError("domain_id must be positive")

    files = tuple(getattr(catalog, "files", ()))
    wps_paths = tuple(Path(item.path) for item in files
                      if getattr(item, "role", None) == "wps_namelist")
    if len(wps_paths) != 1:
        raise ValueError(
            "catalog must contain exactly one wps_namelist file for "
            f"per-domain static selection, found {len(wps_paths)}")
    geog_indices = tuple(Path(item.path) for item in files
                         if getattr(item, "role", None) == "geog_index")
    if not geog_indices:
        raise ValueError(
            "catalog contains no geog_index files; run input preflight "
            "before child initialization")
    roots = {path.resolve().parent.parent for path in geog_indices}
    if len(roots) != 1:
        raise ValueError(
            "catalog geog_index files do not share one resolved GEOG root: "
            f"{sorted(map(str, roots))}")
    geog_root = roots.pop()
    return GeogSelection.from_case_data(
        SimpleNamespace(wps_namelist=wps_paths[0], geog_root=geog_root),
        domain_id=domain_id)


def build_static_for_domain(grid, catalog, domain_id: int) -> dict:
    """Build one domain's statics from the preflight catalog inventory.

    Task 12's child initializer receives an :class:`InputCatalog`, not the
    original ``CaseDataConfig``.  The catalog already owns the resolved WPS
    namelist and GEOG ``index`` paths (including their hashes), so recover
    only those two static-selection inputs here.  No bundle layout is
    inferred: every path comes from a catalog role.

    Results are cached by the complete projected geometry plus the resolved
    :class:`GeogSelection`.  Arrays are read-only because child setup and
    Noah preprocessing derive new arrays and must never mutate shared static
    inputs.  In particular, the returned ``HGT_M`` remains the unblended
    fine terrain after child initialization.
    """
    from gpuwm.static.projection import ProjectedGrid
    if not isinstance(grid, ProjectedGrid):
        raise TypeError("grid must be a ProjectedGrid")
    selection = geog_selection_from_catalog(catalog, domain_id)
    geog_root = selection.root
    # known_x/known_y are part of the projected geometry: two placements of
    # a relocating nest share every other parameter (translated grids keep
    # the reference point and move it in index space), so a key without the
    # known point would alias every placement to the first one built.
    key = (selection, grid.map_proj, grid.ref_lat, grid.ref_lon,
           grid.truelat1, grid.truelat2, grid.stand_lon, grid.dx, grid.dy,
           grid.e_we, grid.e_sn, grid.known_x, grid.known_y)
    fields = _CATALOG_STATIC_CACHE.get(key)
    if fields is None:
        fields = build_static(grid, geog_root, selection=selection)
        for value in fields.values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        _CATALOG_STATIC_CACHE[key] = fields
    return fields
