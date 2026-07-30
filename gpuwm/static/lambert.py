"""Lambert conformal projection grids with WPS conventions.

Pure NumPy float64, setup-time only (no GPU).  The projection math is
transcribed from WRF v4.6.1 ``share/module_llxy.F`` (``lc_cone``:1138,
``set_lc``:1097, ``ijll_lc``:1174, ``llij_lc``:1250 -- the same code WPS
geogrid links); the derived fields follow the ARW tech note sec 2 / Snyder
with WPS geogrid conventions.  Where a convention could be read two ways the
bundle geo_em files arbitrate (tests/test_lambert.py gates every field on
all four 1974 domains against geogrid v4.6.0 output), and since the
worldwide lane the pinned-source oracle fixture arbitrates both
hemispheres (tests/test_projection_oracle.py, built by
tools/llxy_wrf461_oracle from the pristine pinned source):

- Grid registration: 1-based mass point (i, j) sits at projection
  coordinate (i, j); U at (i - 0.5, j); V at (i, j - 0.5); corner (C) at
  (i - 0.5, j - 0.5).  The reference point (ref_lat, ref_lon) sits at
  (e_we/2, e_sn/2), WPS's default ref_x/ref_y (oracle: XLAT/XLONG to
  ~3e-5 deg, the float32-storage bound).
- Map factor  m(lat) = cos(tl1)/cos(lat)
  * (tan(colat/2) / tan(colat1/2))**cone  -- equals 1 at both true
  latitudes (oracle: MAPFAC_* to ~3e-7 relative).  This is the ARW
  tech-note form referenced to truelat1; geogrid's get_map_factor uses
  the mathematically identical form referenced to truelat2, so the
  binary64 oracle gate for this one quantity is a small relative bound
  rather than a ULP count (recorded in tests/test_projection_oracle.py).
- Coriolis  F = 2*OMEGA_E*sin(lat), E = 2*OMEGA_E*cos(lat)  with WPS's
  ``OMEGA_E = 7.292e-5`` from constants_module (oracle: ~9e-7 relative;
  WRF's ``EOMEG = 7.2921e-5`` does *not* reproduce geogrid's F/E).
- Wind rotation  alpha = cone * (stand_lon - lon):  geo_em
  SINALPHA == sin(alpha), COSALPHA == cos(alpha), transcribed from WPS
  geogrid get_rotang (process_tile_module.F:1920), which applies NO
  hemisphere factor -- the cone from lc_cone is positive in both
  hemispheres.  Divergence record: before the worldwide lane this
  module multiplied alpha by ``hemi``, an unreachable SH guess (the
  wizard refused the southern hemisphere); on every reachable NH grid
  ``hemi`` was exactly 1.0, so all shipped fields are byte-identical.
- Nest registration (namelist parent_start/ratio arithmetic): nest mass
  coordinate x maps to parent mass coordinate
  ``(i_parent_start - 0.5) + (x - 0.5)/ratio`` -- cell edges align, and for
  odd ratios every centre nest point coincides with a parent point (oracle:
  geo_em.d02/d03/d04 to ~5e-5 deg through the d02->d03->d04 chain).

Mercator and polar stereographic live in :mod:`gpuwm.static.projection`
together with the shared grid machinery; this module re-exports the
namelist/config dispatch helpers for compatibility.
"""
from __future__ import annotations

import numpy as np

from gpuwm.static.projection import (  # noqa: F401  (compat re-exports)
    EARTH_RADIUS_M, OMEGA_E, ProjectedGrid, _DEG_PER_RAD, _RAD_PER_DEG,
    _parse_wps_namelist, _wrap180, grids_from_projection_config,
    grids_from_wps_namelist)


def _lc_cone(truelat1: float, truelat2: float) -> float:
    """Cone constant, transcribed from module_llxy.F lc_cone (:1138)."""
    if abs(truelat1 - truelat2) > 0.1:  # secant
        num = (np.log10(np.cos(truelat1 * _RAD_PER_DEG))
               - np.log10(np.cos(truelat2 * _RAD_PER_DEG)))
        den = (np.log10(np.tan((45.0 - abs(truelat1) / 2.0) * _RAD_PER_DEG))
               - np.log10(np.tan((45.0 - abs(truelat2) / 2.0) * _RAD_PER_DEG)))
        return float(num / den)
    return float(np.sin(abs(truelat1) * _RAD_PER_DEG))  # tangent


class LambertGrid(ProjectedGrid):
    """One WRF domain on a Lambert conformal (secant or tangent) projection.

    Parameters mirror the WPS &geogrid namelist: ``(ref_lat, ref_lon)`` is
    pinned at projection coordinate ``(known_x, known_y)`` which defaults to
    the WPS ref_x/ref_y of ``(e_we/2, e_sn/2)`` -- the mass-grid centre.
    ``e_we``/``e_sn`` are the *staggered* dimensions (mass grid is one
    smaller).  ``dy`` must equal ``dx`` (WPS requirement for Lambert).

    All array methods return float64 arrays in ``(south_north, west_east)``
    order, matching geo_em's ``[j, i]`` layout.
    """

    map_proj = "lambert"

    def _setup(self) -> None:
        # --- set_lc transcription (module_llxy.F:1097) ---
        self.cone = _lc_cone(self.truelat1, self.truelat2)
        self.rebydx = EARTH_RADIUS_M / self.dx
        deltalon1 = float(_wrap180(self.ref_lon - self.stand_lon))
        ctl1r = np.cos(self.truelat1 * _RAD_PER_DEG)
        # radius (grid lengths) from the pole to the known point
        self.rsw = float(
            self.rebydx * ctl1r / self.cone
            * (np.tan((90.0 * self.hemi - self.ref_lat) * _RAD_PER_DEG / 2.0)
               / np.tan((90.0 * self.hemi - self.truelat1)
                        * _RAD_PER_DEG / 2.0)) ** self.cone)
        arg = self.cone * (deltalon1 * _RAD_PER_DEG)
        self.polei = float(self.hemi * self.known_x
                           - self.hemi * self.rsw * np.sin(arg))
        self.polej = float(self.hemi * self.known_y
                           + self.rsw * np.cos(arg))

    # -- core transforms ---------------------------------------------------

    def ij_to_latlon(self, x, y):
        """Projection coordinate (x, y) -> (lat, lon) degrees.

        Transcription of ijll_lc (module_llxy.F:1174), vectorized.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        chi1 = (90.0 - self.hemi * self.truelat1) * _RAD_PER_DEG
        chi2 = (90.0 - self.hemi * self.truelat2) * _RAD_PER_DEG
        xx = self.hemi * x - self.polei
        yy = self.polej - self.hemi * y
        r2 = xx * xx + yy * yy
        r = np.sqrt(r2) / self.rebydx
        lon = self.stand_lon + _DEG_PER_RAD * np.arctan2(self.hemi * xx,
                                                         yy) / self.cone
        lon = np.mod(lon + 360.0, 360.0)
        if chi1 == chi2:  # tangent (exact-equality branch, as in Fortran)
            chi = 2.0 * np.arctan((r / np.tan(chi1)) ** (1.0 / self.cone)
                                  * np.tan(chi1 * 0.5))
        else:             # secant
            chi = 2.0 * np.arctan(
                (r * self.cone / np.sin(chi1)) ** (1.0 / self.cone)
                * np.tan(chi1 * 0.5))
        lat = (90.0 - chi * _DEG_PER_RAD) * self.hemi
        # pole point (r2 == 0): the formulas above already yield
        # lat = 90*hemi, lon = stand_lon in IEEE arithmetic, but mirror the
        # Fortran branch explicitly for exactness.
        lat = np.where(r2 == 0.0, 90.0 * self.hemi, lat)
        lon = np.where(r2 == 0.0, np.mod(self.stand_lon + 360.0, 360.0), lon)
        lon = np.where(lon > 180.0, lon - 360.0, lon)
        lon = np.where(lon < -180.0, lon + 360.0, lon)
        return lat, lon

    def latlon_to_ij(self, lat, lon):
        """(lat, lon) degrees -> projection coordinate (x, y).

        Transcription of llij_lc (module_llxy.F:1250), vectorized.
        """
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        deltalon = _wrap180(lon - self.stand_lon)
        ctl1r = np.cos(self.truelat1 * _RAD_PER_DEG)
        rm = (self.rebydx * ctl1r / self.cone
              * (np.tan((90.0 * self.hemi - lat) * _RAD_PER_DEG / 2.0)
                 / np.tan((90.0 * self.hemi - self.truelat1)
                          * _RAD_PER_DEG / 2.0)) ** self.cone)
        arg = self.cone * (deltalon * _RAD_PER_DEG)
        x = self.polei + self.hemi * rm * np.sin(arg)
        y = self.polej - rm * np.cos(arg)
        return self.hemi * x, self.hemi * y

    # -- derived fields ------------------------------------------------------

    def map_factor(self, lat):
        """Map scale factor m(lat); equals 1 at both true latitudes."""
        lat = np.asarray(lat, dtype=np.float64)
        half_colat = (90.0 * self.hemi - lat) * _RAD_PER_DEG / 2.0
        half_colat1 = (90.0 * self.hemi - self.truelat1) * _RAD_PER_DEG / 2.0
        return (np.cos(self.truelat1 * _RAD_PER_DEG)
                / np.cos(lat * _RAD_PER_DEG)
                * (np.tan(half_colat) / np.tan(half_colat1)) ** self.cone)

    def rotation(self, lon):
        """(SINALPHA, COSALPHA) at longitudes LON.

        alpha = cone * (stand_lon - lon): geogrid get_rotang PROJ_LC
        (no hemisphere factor; cone is positive in both hemispheres).
        Operation order preserves the shipped NH bytes: the retired
        ``hemi *`` prefactor was exactly 1.0 on every reachable grid.
        """
        lon = np.asarray(lon, dtype=np.float64)
        alpha = (self.cone * _RAD_PER_DEG
                 * _wrap180(self.stand_lon - lon))
        return np.sin(alpha), np.cos(alpha)
