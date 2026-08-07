"""EXPERIMENTAL (ArWen v1.2): radar observation operators H(x).

Two gridded operators, both pure functions of the model state, both
CuPy-first with a NumPy fallback, neither wired into any default route:

- :func:`simulated_reflectivity` -- 10 cm equivalent reflectivity (dBZ).
- :func:`radial_velocity` -- Doppler radial velocity (m/s) toward or away
  from a radar site, under 4/3-earth beam geometry.

They produce the model-grid half of the ``gpuwm-obs.radar-grid.v1`` obs
schema (gridded ``z_obs``/``vr_obs`` plus masks on the model grid), so an
ensemble filter can difference H(x) against the observation arrays
element-wise with no interpolation of its own.

Reflectivity: NO SECOND Z FORMULA
=================================

ArWen already computes reflectivity on the product side, and this module
does not reimplement it.  ``simulated_reflectivity`` is a thin,
shape-checking adapter over the existing authority:

- device path: ``gpuwm.core.refl.compute_refl_10cm``
  (gpuwm/core/refl.py:448) verbatim for schemes 1/6/8/10 -- the same call
  the microphysics adapters make for wrfout ``REFL_10CM`` -- and
  ``gpuwm.core.nssl2_diagnostics.diagnose_radardd02_if_due`` for scheme 18,
  the same NSSL diagnostic the production coordinator runs;
- host path: the certified float64 column mirrors
  ``gpuwm.verify.npref.np_refl10cm_morrison_column`` (npref.py:9204),
  ``np_refl10cm_wsm6_column`` (npref.py:9351) and
  ``np_refl10cm_kessler_column`` (npref.py:9460), applied column by
  column.  Thompson (8) and NSSL (18) have no host mirror and are CUDA
  only.

That authority is WRF v4.6.1 ``do_radar_ref=1`` / REFL_10CM, dispatched by
``mp_physics`` exactly as ``microphysics.apply`` dispatches, so the
operator inherits the ACTIVE scheme's own PSD assumptions rather than
imposing one scheme's assumptions on another:

- ``mp_physics=10`` -- Morrison ``refl10cm_hm``
  (phys/module_mp_morr_two_moment.F:4502-4675, wrapper floor :913-917).
  Two-moment rain/snow/graupel using the scheme's PROGNOSTIC number
  concentrations; ``morr_rimed_ice`` selects graupel vs hail density.
  Cloud water contributes nothing (the routine reads no cloud moment).
- ``mp_physics=6`` -- WSM6 ``refl10cm_wsm6``
  (phys/mp_wsm6.F90:2275-2444): fixed-intercept exponential PSDs,
  ``hail_opt`` selecting the rimed-ice density/intercept.
- ``mp_physics=8`` -- classic Thompson ``calc_refl10cm``
  (phys/module_mp_thompson.F:5710-6028), which additionally needs the
  scheme's same-call private graupel number moment.
- ``mp_physics=1`` -- Kessler has no reflectivity diagnostic in WRF, so
  the authority's documented fallback is the Smith et al. (1975)
  fixed-intercept (N0r = 8e6 m-4) rain-only Rayleigh form derived at
  gpuwm/core/refl.py:30-43.
- ``mp_physics=18`` -- NSSL two-moment ``radardd02``
  (phys/module_mp_nssl_2mom.F): five ice categories, five predicted
  number moments and the graupel/hail volume moments, S-band, CUDA only.
  It is a genuine scheme port, not an adapter over one of the above, so
  it lives in ``gpuwm.core.nssl2_diagnostics`` and is dispatched to
  verbatim here.  Two departures from the 1/6/8/10 family the caller must
  carry: radardd02 floors at **0 dBZ**, not -35, and it reads five ice
  categories plus their moments rather than the shared rain-first set.

The 1/6/8/10 routes share the -35 dBZ floor, the 1e-9 kg/kg species
activity threshold, and the ``refl10cm_hm`` air-density diagnosis, so
their outputs are interchangeable in one obs array.  NSSL (18) is native
S-band with a 0 dBZ floor and is not floor-interchangeable with them.

Documented simplifications
==========================

1. **Fall speed is not the per-scheme reflectivity-weighted integral.**
   The bound Z formulations compute Ze only; none of them exposes a
   reflectivity-weighted mean terminal velocity.  Rather than invent a
   fourth PSD integration, :func:`reflectivity_fall_speed` uses the
   standard radar-DA relation of Sun and Crook (1997, J. Atmos. Sci. 54,
   1642-1661; 1998, J. Atmos. Sci. 55, 835-852), the same closure
   WRF-DA's ``da_radial_velocity`` and Tong and Xue (2005, Mon. Wea.
   Rev. 133, 1789-1807) use, driven by the SAME dBZ field the bound Z
   operator produced:

       vt = 5.40 * (ps/p)**0.4 * 10**(0.125*(Z - 43.1)/17.5)  [m/s, down]

   ``ps`` is the SURFACE pressure, supplied by the caller, not the
   1000 hPa thermodynamic reference: that is what Sun and Crook wrote
   and what WRFDA's ``da_radial_velocity.inc`` implements, and the
   substitution is a systematic +4.3%/+9.3%/+15.3% error in ``vt`` over
   a 900/800/700 hPa surface.

   That form is Sun and Crook's ``vt = 5.40*a*(rho*qr)**0.125`` with
   ``rho*qr`` eliminated through their ``Z = 43.1 + 17.5*log10(rho*qr)``,
   and their Z leg is exactly the Marshall-Palmer/Smith-1975 PSD the
   Kessler fallback here implements -- so on ``mp_physics=1`` the fall
   speed and the reflectivity share one PSD (checked numerically in
   :func:`reflectivity_fall_speed`).  On the Morrison, WSM6 and Thompson
   paths the richer scheme PSD supplies Z while the fall speed is still
   read off the Marshall-Palmer rain calibration: it consumes that
   scheme's Z, but it is NOT a moment integral over that scheme's PSD,
   and it overstates the fall speed of dry snow aloft.

   Because a power law in Z does not vanish in clear air (it still gives
   ~1.7 m/s at the -35 dBZ floor), the fall speed is gated OFF wherever
   the reflectivity authority itself sees no active hydrometeor: the
   1e-9 kg/kg threshold applied PER SPECIES to qr, qs, qg and qh, active
   where ANY of them clears it, exactly as ``refl10cm_hm`` tests each
   category before adding its Ze.  Gating the SUM instead -- which this
   module did -- activates cells whose every species is individually
   inactive: four at 3e-10 kg/kg contribute no reflectivity at all and
   used to switch on a 5.244 m/s fall speed.  (WRFDA's Sun-Crook gates on
   rain alone above 1e-5 kg/kg; that is the other defensible authority
   and it is NOT this one, because the Z consumed here is the model
   scheme's multi-species Z and not a rain-only one.)  Callers who want a
   different closure pass ``fall_speed=<array>``; callers who want pure
   air motion pass ``fall_speed="none"``.

2. **Straight-ray 4/3-earth geometry, no beam-volume weighting.**  The
   operator evaluates the beam direction at each model mass point and
   projects the point wind onto it.  There is no Gaussian beam-power
   weighting across the sample volume and no along-beam averaging: at
   nest resolutions the model cell is smaller than the radar gate, which
   is a resolution mismatch the obs-error covariance must absorb, not
   something this operator claims to solve.

3. **No attenuation, no beam blockage, no ground clutter, no dual-pol.**

4. **Spherical earth.**  Radius ``1/gpuwm.core.constants.RERADIUS`` =
   6 370 000 m, the same value ``gpuwm.static.projection.EARTH_RADIUS_M``
   uses, so the beam geometry is consistent with the projection the grid
   was built on.

EXPERIMENTAL.  Not covered by the v1 stability promise.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from gpuwm.core import constants as c

EXPERIMENTAL = True

#: Obs schema these operators are the model-side half of.
OBS_SCHEMA = "gpuwm-obs.radar-grid.v1"

#: Spherical earth radius (m).  ``1/RERADIUS`` is the same 6 370 000 m
#: that ``gpuwm.static.projection.EARTH_RADIUS_M`` (module_llxy.F:152)
#: uses, spelled from the core constant so it is derived, never
#: hardcoded twice.
EARTH_RADIUS_M = 1.0 / c.RERADIUS

#: Standard-atmosphere effective-earth radius ratio for radar beam
#: refraction (Doviak and Zrnic, *Doppler Radar and Weather
#: Observations*, 2nd ed., eq. 2.28b).
EFFECTIVE_EARTH_RATIO = 4.0 / 3.0

#: Species activity threshold shared with the reflectivity authority
#: (``refl10cm_hm`` and every gpuwm mirror gate on ``q > 1e-9``).  Applied
#: PER SPECIES, as the authority applies it -- see
#: :func:`precipitating_activity_mask`.
Q_ACTIVE_THRESHOLD = 1.0e-9

#: The dBZ value :func:`simulated_reflectivity` returns in fully clear air,
#: per scheme.  This is the operator's own floor, read off the routes
#: documented at the top of this module and pinned by
#: ``tests/test_da_obsop_thompson_gpu.py`` (mp8 = -35) and
#: ``tests/test_da_obsop_nssl_gpu.py`` (mp18 = 0).
#:
#: It exists as a lookup because clear-air ("zero") assimilation needs the
#: number as *data*: a clear-air observation is differenced against H(x),
#: so the observation must carry the same floor the operator produces or
#: two agreeing clear skies yield a 35 dB innovation.  Typing ``-35.0`` at
#: a call site is the defect this table prevents -- it is right for four
#: schemes and silently catastrophic for the fifth.
CLEAR_AIR_FLOOR_DBZ: dict[int, float] = {
    1: -35.0,
    6: -35.0,
    8: -35.0,
    10: -35.0,
    18: 0.0,
    28: -35.0,
}


def clear_air_floor_dbz(mp_physics: int) -> float:
    """The active scheme's clear-air H(x) value, or a refusal.

    Never falls back to the -35 dBZ majority: an unknown scheme is one
    whose floor nobody has read, and guessing it wrong is invisible in
    every diagnostic except the analysis increment.
    """

    key = int(mp_physics)
    if key not in CLEAR_AIR_FLOOR_DBZ:
        raise ValueError(
            f"no clear-air reflectivity floor is recorded for mp_physics="
            f"{key}; known schemes are "
            f"{sorted(CLEAR_AIR_FLOOR_DBZ)}. The floor is not guessable -- "
            "it is -35 dBZ for the refl10cm family and 0 dBZ for NSSL, and "
            "assimilating clear air against the wrong one manufactures a "
            "35 dB innovation wherever the sky is genuinely clear")
    return CLEAR_AIR_FLOOR_DBZ[key]

#: Sun and Crook fall speed ``vt = 5.40*a*(rho*qr)**0.125`` with
#: ``a = (ps/p)**0.4`` on the SURFACE pressure and ``rho*qr`` in g m-3,
#: and the companion reflectivity relation
#: ``Z[dBZ] = 43.1 + 17.5*log10(rho*qr)`` used to eliminate ``rho*qr``.
#: See :func:`reflectivity_fall_speed` for the elimination and for the
#: check that the Z leg is the same Marshall-Palmer PSD the Kessler
#: fallback in gpuwm/core/refl.py:30-43 implements.
SUN_CROOK_VT_COEFF = 5.40
SUN_CROOK_VT_EXPONENT = 0.125
SUN_CROOK_Z_INTERCEPT_DBZ = 43.1
SUN_CROOK_Z_SLOPE_DB = 17.5
FALL_SPEED_DENSITY_EXPONENT = 0.4

#: Default clamp on the diagnosed fall speed (m/s).  Large enough for
#: hail, small enough that a pathological Z cannot dominate the beam
#: projection.
FALL_SPEED_MAX_MS = 30.0

__all__ = [
    "EXPERIMENTAL", "OBS_SCHEMA", "EARTH_RADIUS_M", "EFFECTIVE_EARTH_RATIO",
    "RadarSite", "GridGeometry", "BeamGeometry",
    "simulated_reflectivity", "radial_velocity",
    "beam_geometry", "mass_point_heights", "mass_point_winds",
    "reflectivity_fall_speed", "precipitating_mixing_ratio",
    "precipitating_activity_mask", "PRECIPITATING_SPECIES",
    "Q_ACTIVE_THRESHOLD",
    "destagger_u", "destagger_v", "destagger_w", "earth_relative_winds",
]


# --------------------------------------------------------------------------
# array-module dispatch: CuPy first, NumPy fallback
# --------------------------------------------------------------------------

def _array_module(value):
    """CuPy for device arrays, NumPy for host arrays.

    Never imports CuPy to answer a question about a NumPy array, so a
    host-only caller (and a CPU-only test module) touches no device.
    """
    if isinstance(value, np.ndarray):
        return np
    cp = sys.modules.get("cupy")
    if cp is None:
        try:
            import cupy as cp  # noqa: PLC0415
        except Exception:
            return np
    try:
        if isinstance(value, cp.ndarray):
            return cp
    except Exception:
        return np
    return np


def _asarray64(xp, value):
    """Host-side float64 view for geometry arithmetic."""
    return np.asarray(_to_host(xp, value), dtype=np.float64)


def _to_host(xp, value):
    if xp is np:
        return value
    return xp.asnumpy(value)


# --------------------------------------------------------------------------
# site / grid geometry
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RadarSite:
    """A radar's position.  Longitude east-positive, altitude m AMSL."""

    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    name: str = ""

    def __post_init__(self) -> None:
        if not -90.0 <= float(self.latitude_deg) <= 90.0:
            raise ValueError(
                f"radar latitude out of range: {self.latitude_deg}")
        if not -360.0 <= float(self.longitude_deg) <= 360.0:
            raise ValueError(
                f"radar longitude out of range: {self.longitude_deg}")
        if not np.isfinite(float(self.altitude_m)):
            raise ValueError("radar altitude must be finite")


@dataclass(frozen=True)
class GridGeometry:
    """Where the model's mass points are, in earth coordinates.

    ``latitude_deg`` / ``longitude_deg`` are ``(ny, nx)``; ``height_m`` is
    ``(nz, ny, nx)`` geometric height AMSL at mass points.

    The model state does not carry its own geolocation -- ``DomainState``
    has no XLAT/XLONG, and every runtime consumer that needs latitude
    (the radiation solvers) takes it as an explicit argument
    (``gpuwm/core/rrtmgp.py:1581`` ``latitude_deg``).  This dataclass
    follows that precedent rather than inventing state fields.
    """

    latitude_deg: object
    longitude_deg: object
    height_m: object

    def __post_init__(self) -> None:
        lat = np.asarray(_to_host(_array_module(self.latitude_deg),
                                  self.latitude_deg))
        lon = np.asarray(_to_host(_array_module(self.longitude_deg),
                                  self.longitude_deg))
        hgt = np.asarray(_to_host(_array_module(self.height_m),
                                  self.height_m))
        if lat.ndim != 2:
            raise ValueError(f"latitude_deg must be (ny, nx), got {lat.shape}")
        if lat.shape != lon.shape:
            raise ValueError(
                f"latitude_deg {lat.shape} and longitude_deg {lon.shape} "
                "must have the same shape")
        if hgt.ndim != 3:
            raise ValueError(f"height_m must be (nz, ny, nx), got {hgt.shape}")
        if hgt.shape[1:] != lat.shape:
            raise ValueError(
                f"height_m {hgt.shape} does not sit on the "
                f"{lat.shape} latitude/longitude grid")
        # Catch a transposed or swapped lat/lon pair here rather than
        # letting it produce a plausible-looking but wrong beam.
        if not np.all(np.isfinite(lat)) or np.any(np.abs(lat) > 90.0):
            raise ValueError(
                "latitude_deg must be finite and within +/-90 degrees "
                f"(got range {float(np.nanmin(lat))} to "
                f"{float(np.nanmax(lat))}); a swapped latitude/longitude "
                "pair is the usual cause")
        if not np.all(np.isfinite(lon)) or np.any(np.abs(lon) > 360.0):
            raise ValueError(
                "longitude_deg must be finite and within +/-360 degrees "
                f"(got range {float(np.nanmin(lon))} to "
                f"{float(np.nanmax(lon))})")
        if not np.all(np.isfinite(hgt)):
            raise ValueError("height_m must be finite everywhere")

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(np.asarray(
            _to_host(_array_module(self.height_m), self.height_m)).shape)

    @classmethod
    def from_state(cls, state, latitude_deg=None, longitude_deg=None):
        """Build from a state plus explicit geolocation.

        Heights come from the state's geopotential; latitude/longitude
        must be supplied, or carried by the state as ``xlat``/``xlong``
        (some ingest paths attach them).  Fails closed otherwise --
        guessing a domain's position would silently produce a plausible,
        wrong beam.
        """
        if latitude_deg is None:
            latitude_deg = getattr(state, "xlat", None)
        if longitude_deg is None:
            longitude_deg = getattr(state, "xlong", None)
        if latitude_deg is None or longitude_deg is None:
            raise ValueError(
                "GridGeometry.from_state needs the domain's mass-point "
                "latitude and longitude: DomainState does not carry XLAT/"
                "XLONG, so pass latitude_deg=/longitude_deg= (for example "
                "from gpuwm.static.projection ProjectedGrid.latlon_mass())")
        return cls(latitude_deg=latitude_deg, longitude_deg=longitude_deg,
                   height_m=mass_point_heights(state))

    @classmethod
    def from_target_grid(cls, grid):
        """Build from the radar lane's :class:`gpuwm.obs.TargetGrid`.

        The observation side and the operator side must agree about where
        a model cell *is*, or an innovation is a difference between two
        different places.  ``TargetGrid`` carries layer-interface heights
        ``z_w`` ``(nz+1, ny, nx)`` and places every gate in the layer whose
        interfaces bracket it, in that gate's own column; the operator
        evaluates H(x) at mass points.  The mass point of that layer is
        the midpoint of the two interfaces that defined it, so this is the
        one reduction that keeps the two stages talking about one cell.

        Passing the same ``TargetGrid`` that wrote a
        ``gpuwm-obs.radar-grid.v1`` file is therefore the whole seam: read
        the file back with ``expected_grid_identity=grid.identity_sha256()``
        and both sides are pinned to the same columns by a digest.
        """

        z_w = np.asarray(grid.z_w, dtype=np.float64)
        return cls(latitude_deg=np.asarray(grid.lat, dtype=np.float64),
                   longitude_deg=np.asarray(grid.lon, dtype=np.float64),
                   height_m=0.5 * (z_w[:-1] + z_w[1:]))

    @classmethod
    def from_wrfout(cls, path, *, frame: int = 0):
        """Build from a run artifact -- the geometry supply for a cycle.

        Goes through :meth:`gpuwm.obs.TargetGrid.from_wrfout`, which
        rebuilds the projection from the file's global attributes and then
        checks it against the file's own XLAT/XLONG, refusing a
        disagreement above 1e-5 deg.  Reading XLAT/XLONG straight out of
        the file would be shorter and would skip that check; the check is
        the reason a nest with an off-centre reference point does not
        quietly shift every observation by a cell.

        Imported lazily so ``gpuwm.da.obsop`` keeps working with no
        netCDF4 present -- H(x) itself needs no file.
        """

        from gpuwm.obs.target_grid import TargetGrid   # noqa: PLC0415

        return cls.from_target_grid(TargetGrid.from_wrfout(path,
                                                           frame=frame))


@dataclass(frozen=True)
class BeamGeometry:
    """Beam direction at every model mass point, in local ENU.

    ``sin_elevation`` / ``cos_elevation`` and ``sin_azimuth`` /
    ``cos_azimuth`` describe the unit vector pointing FROM the radar
    THROUGH the mass point, resolved in that point's own east/north/up
    frame.  Azimuth is measured clockwise from north, so the unit vector
    is ``(cos_el*sin_az, cos_el*cos_az, sin_el)``.

    All arrays are float64 ``(nz, ny, nx)`` except the azimuth pair,
    which is ``(ny, nx)`` -- azimuth does not vary with height along a
    vertical column.  ``slant_range_m`` is the straight-line distance in
    the effective-earth frame.
    """

    sin_elevation: np.ndarray
    cos_elevation: np.ndarray
    sin_azimuth: np.ndarray
    cos_azimuth: np.ndarray
    slant_range_m: np.ndarray
    ground_range_m: np.ndarray
    effective_earth_radius_m: float

    def unit_vector_enu(self):
        """(east, north, up) components, each broadcast to (nz, ny, nx)."""
        east = self.cos_elevation * self.sin_azimuth
        north = self.cos_elevation * self.cos_azimuth
        return east, north, self.sin_elevation


def mass_point_heights(state):
    """Geometric height AMSL ``(nz, ny, nx)`` at mass points.

    ``z = (phb + php)/g`` on full levels, averaged to half levels -- the
    same tree ``gpuwm.core.state._height_half_from_phb`` uses for the
    base state, extended to the perturbed column.
    """
    php = state.php
    xp = _array_module(php)
    phb = state.phb
    if getattr(phb, "ndim", 0) == 1:
        phb = phb[:, None, None]
    z_full = (phb + php) / np.float32(c.G)
    return 0.5 * (z_full[:-1] + z_full[1:])


def beam_geometry(geometry: GridGeometry, site: RadarSite, *,
                  effective_earth_ratio: float = EFFECTIVE_EARTH_RATIO
                  ) -> BeamGeometry:
    """Straight-ray 4/3-earth beam geometry at every mass point.

    In the effective-earth model the ray is straight and the earth radius
    is inflated to ``k*a`` (k = 4/3 for a standard atmosphere), which
    reproduces the ray's true curvature relative to the surface.  The
    physical ground arc ``s`` between radar and target is unchanged, so
    the central angle subtended in the effective frame is ``s/(k*a)``.

    With the earth centre at the origin, the radar at radius
    ``R0 = k*a + alt`` and the target at ``R1 = k*a + z`` separated by
    central angle ``theta``, the chord and the direction cosines at the
    TARGET follow exactly:

        r        = sqrt(R0^2 + R1^2 - 2*R0*R1*cos(theta))
        sin(el)  = (R1 - R0*cos(theta)) / r
        cos(el)  =  R0*sin(theta) / r

    Both are exact (not one derived from the other by a square root), so
    the pair stays a unit vector to round-off even at grazing incidence.

    Azimuth at the target is the forward azimuth of the radar->target
    great circle evaluated at the target, i.e. the reverse of the
    target->radar azimuth.  Directly over the radar ``cos(el)`` is zero
    and azimuth is irrelevant; the code returns the (0, 1) placeholder
    there rather than a NaN, because it is multiplied by zero.

    A target coincident with the radar has no beam direction at all;
    those cells come back NaN.
    """
    if not np.isfinite(effective_earth_ratio) or effective_earth_ratio <= 0.0:
        raise ValueError(
            f"effective_earth_ratio must be positive and finite, "
            f"got {effective_earth_ratio}")

    xp_h = _array_module(geometry.height_m)
    lat = np.deg2rad(_asarray64(_array_module(geometry.latitude_deg),
                                geometry.latitude_deg))
    lon = np.deg2rad(_asarray64(_array_module(geometry.longitude_deg),
                                geometry.longitude_deg))
    height = _asarray64(xp_h, geometry.height_m)

    lat0 = np.deg2rad(np.float64(site.latitude_deg))
    lon0 = np.deg2rad(np.float64(site.longitude_deg))

    # Great-circle central angle on the REAL earth (haversine: accurate
    # for the small separations a radar actually sees).
    dlat = lat - lat0
    dlon = lon - lon0
    hav = (np.sin(0.5 * dlat) ** 2
           + np.cos(lat0) * np.cos(lat) * np.sin(0.5 * dlon) ** 2)
    gamma = 2.0 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))
    ground_range = EARTH_RADIUS_M * gamma

    # Same ground arc, inflated radius -> smaller central angle.
    ae = effective_earth_ratio * EARTH_RADIUS_M
    theta = gamma / effective_earth_ratio
    cos_t = np.cos(theta)[None, :, :]
    sin_t = np.sin(theta)[None, :, :]

    r0 = ae + np.float64(site.altitude_m)
    r1 = ae + height
    slant = np.sqrt(np.maximum(r0 * r0 + r1 * r1 - 2.0 * r0 * r1 * cos_t, 0.0))

    coincident = slant <= 0.0
    safe = np.where(coincident, 1.0, slant)
    sin_el = np.where(coincident, np.nan, (r1 - r0 * cos_t) / safe)
    cos_el = np.where(coincident, np.nan, (r0 * sin_t) / safe)

    # Azimuth at the target of the outgoing beam = reverse of the
    # target->radar great-circle azimuth.
    y_ba = np.sin(lon0 - lon) * np.cos(lat0)
    x_ba = (np.cos(lat) * np.sin(lat0)
            - np.sin(lat) * np.cos(lat0) * np.cos(lon0 - lon))
    norm = np.hypot(y_ba, x_ba)
    overhead = norm <= 0.0
    safe_norm = np.where(overhead, 1.0, norm)
    sin_az = np.where(overhead, 0.0, -y_ba / safe_norm)
    cos_az = np.where(overhead, 1.0, -x_ba / safe_norm)

    return BeamGeometry(
        sin_elevation=sin_el, cos_elevation=cos_el,
        sin_azimuth=sin_az[None, :, :], cos_azimuth=cos_az[None, :, :],
        slant_range_m=slant, ground_range_m=ground_range,
        effective_earth_radius_m=float(ae))


# --------------------------------------------------------------------------
# winds: destaggering and earth-relative rotation
# --------------------------------------------------------------------------

def destagger_u(u):
    """``(nz, ny, nx+1)`` u faces -> ``(nz, ny, nx)`` mass points."""
    if u.ndim != 3:
        raise ValueError(f"u must be 3-D, got {u.shape}")
    return 0.5 * (u[:, :, :-1] + u[:, :, 1:])


def destagger_v(v):
    """``(nz, ny+1, nx)`` v faces -> ``(nz, ny, nx)`` mass points."""
    if v.ndim != 3:
        raise ValueError(f"v must be 3-D, got {v.shape}")
    return 0.5 * (v[:, :-1, :] + v[:, 1:, :])


def destagger_w(w):
    """``(nz+1, ny, nx)`` w faces -> ``(nz, ny, nx)`` mass points."""
    if w.ndim != 3:
        raise ValueError(f"w must be 3-D, got {w.shape}")
    return 0.5 * (w[:-1, :, :] + w[1:, :, :])


def earth_relative_winds(u_mass, v_mass, sina, cosa):
    """Rotate grid-relative mass-point winds to earth-relative.

    ``SINALPHA``/``COSALPHA`` follow the geo_em convention this repo
    transcribes from WPS ``get_rotang`` (gpuwm/static/projection.py:21-30):
    ``alpha = wrap(stand_lon - lon) * cone``.  With true north rotated
    counter-clockwise from grid north by ``beta = -alpha``,

        u_earth = u_grid*COSALPHA - v_grid*SINALPHA
        v_earth = u_grid*SINALPHA + v_grid*COSALPHA

    which is the inverse of the rotation WRF's ``uvmet`` diagnostic
    applies, and the identity when ``sina = 0, cosa = 1`` (the idealized
    default ``DomainState`` installs, gpuwm/core/state.py:528-529).
    """
    if sina is None and cosa is None:
        return u_mass, v_mass
    if sina is None or cosa is None:
        raise ValueError("sina and cosa must be supplied together")
    if getattr(sina, "ndim", 0) == 2:
        sina = sina[None, :, :]
    if getattr(cosa, "ndim", 0) == 2:
        cosa = cosa[None, :, :]
    return (u_mass * cosa - v_mass * sina,
            u_mass * sina + v_mass * cosa)


def mass_point_winds(state, *, rotate_to_earth: bool = True):
    """``(u_east, v_north, w)`` on mass points, ``(nz, ny, nx)`` each.

    Destaggers all three components and, unless ``rotate_to_earth`` is
    False, rotates the horizontal pair out of grid-relative coordinates
    using the state's own ``sina``/``cosa``.
    """
    u_mass = destagger_u(state.u)
    v_mass = destagger_v(state.v)
    w_mass = destagger_w(state.w)
    if rotate_to_earth:
        sina = getattr(state, "sina", None)
        cosa = getattr(state, "cosa", None)
        if sina is not None and cosa is not None:
            u_mass, v_mass = earth_relative_winds(u_mass, v_mass, sina, cosa)
    return u_mass, v_mass, w_mass


# --------------------------------------------------------------------------
# reflectivity: adapter over the product-side authority
# --------------------------------------------------------------------------

def _host_reflectivity(state, cfg, temperature, pressure):
    """Column-by-column float64 mirror path (host states only).

    Reference fallback, not a performance path: it makes one Python call
    per column into the certified ``gpuwm.verify.npref`` mirrors.
    """
    from gpuwm.verify import npref

    nz, ny, nx = np.asarray(state.p).shape
    if temperature is None:
        thb = state.thb
        if getattr(thb, "ndim", 0) == 1:
            thb = thb[:, None, None]
        pressure = state.p
        temperature = ((thb + state.thp)
                       * (np.asarray(pressure) / np.float32(c.P0))
                       ** np.float32(c.RCP))

    t = np.asarray(temperature, dtype=np.float64)
    p = np.asarray(pressure, dtype=np.float64)
    qv = np.asarray(state.qv, dtype=np.float64)

    def field(name):
        value = getattr(state, name, None)
        if value is None:
            raise ValueError(
                f"mp_physics={cfg.mp_physics} reflectivity lacks {name}")
        return np.asarray(value, dtype=np.float64)

    out = np.empty((nz, ny, nx), dtype=np.float32)
    mp = int(cfg.mp_physics)
    if mp == 10:
        qr, nr = field("qr"), field("nr")
        qs, ns = field("qs"), field("ns")
        qg, ng = field("qg"), field("ng")
        rimed = int(getattr(cfg, "morr_rimed_ice", 1))
        for j in range(ny):
            for i in range(nx):
                out[:, j, i] = npref.np_refl10cm_morrison_column(
                    qv[:, j, i], qr[:, j, i], nr[:, j, i],
                    qs[:, j, i], ns[:, j, i], qg[:, j, i], ng[:, j, i],
                    t[:, j, i], p[:, j, i], morr_rimed_ice=rimed)
    elif mp == 6:
        qr, qs, qg = field("qr"), field("qs"), field("qg")
        hail = int(getattr(cfg, "wsm6_hail_opt", 0))
        for j in range(ny):
            for i in range(nx):
                out[:, j, i] = npref.np_refl10cm_wsm6_column(
                    qv[:, j, i], qr[:, j, i], qs[:, j, i], qg[:, j, i],
                    t[:, j, i], p[:, j, i], hail_opt=hail)
    elif mp == 1:
        qr = field("qr")
        for j in range(ny):
            for i in range(nx):
                out[:, j, i] = npref.np_refl10cm_kessler_column(
                    qv[:, j, i], qr[:, j, i], t[:, j, i], p[:, j, i])
    elif mp == 8:
        raise NotImplementedError(
            "the host reflectivity fallback covers mp_physics 1, 6 and 10; "
            "mp_physics=8 (classic Thompson calc_refl10cm) has no float64 "
            "column mirror in gpuwm.verify.npref, so it is only available "
            "on the CUDA path")
    elif mp == 18:
        raise NotImplementedError(
            "the host reflectivity fallback covers mp_physics 1, 6 and 10; "
            "mp_physics=18 (NSSL radardd02) has no float64 column mirror in "
            "gpuwm.verify.npref -- unlike the three mirrored schemes it "
            "reads five ice categories, five number moments and two volume "
            "moments, and porting it is a scheme mirror rather than an "
            "adapter -- so NSSL H(x) is available on the CUDA path only, "
            "through the product's own gpuwm.core.nssl2_diagnostics")
    else:
        raise NotImplementedError(
            f"mp_physics={mp} has no reflectivity formulation here; the "
            "supported set is 1 (Kessler fallback), 6 (WSM6), 8 (classic "
            "Thompson, CUDA only), 10 (Morrison) and 18 (NSSL, CUDA only)")
    return out


def _nssl_reflectivity(state, temperature, pressure):
    """``mp_physics=18`` through the product's own NSSL diagnostic.

    Not a fourth Z formula and not a mapping onto Morrison: the same
    ``gpuwm.core.nssl2_diagnostics.diagnose_radardd02_if_due`` the
    production coordinator calls, reading the state's own five ice
    categories, five number moments and two volume moments.

    Two things differ from the 1/6/8/10 routes and a DA caller has to know
    both.  NSSL's ``radardd02`` floors at **0 dBZ**, not -35, so clear air
    in an NSSL H(x) column reads 0 and an innovation against a -35 dBZ
    observation floor is meaningless -- give reflectivity observations a
    matching floor or mask the clear-air cells.  And the moments here are
    per unit mass of dry air (``concentration_space=False``), which is what
    a ``DomainState`` carries; the coordinator passes ``True`` because it
    hands over the scheme's internal per-volume slab.
    """
    import cupy as cp

    from gpuwm.core.nssl2_diagnostics import (       # noqa: PLC0415
        diagnose_radardd02_if_due,
    )
    from gpuwm.core.state import DTYPE               # noqa: PLC0415

    shape = tuple(state.p.shape)
    # Dry-air density, formed the way the NSSL runtime forms it
    # (nssl2_runtime._prepare_fields: rho = 1/alt).
    rho = state.scratch(shape, "da_nssl_rho")
    rho[...] = DTYPE(1.0) / state.alt
    if temperature is None:
        thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
        t = state.scratch(shape, "da_nssl_t")
        t[...] = (thb + state.thp) * cp.power(
            state.p / DTYPE(c.P0), DTYPE(c.RCP))
    else:
        t = temperature
    refl = state.scratch(shape, "refl_10cm")
    diagnose_radardd02_if_due(state, rho, t, refl, output_due=True,
                              concentration_space=False)
    return refl


def simulated_reflectivity(state, cfg, *, temperature=None, pressure=None,
                           thompson_graupel_number=None):
    """H_Z(x): 10 cm reflectivity ``(nz, ny, nx)`` in dBZ.

    A shape-checking adapter over the product-side authority; see the
    module docstring for which formulation each ``cfg.mp_physics`` binds
    to and what PSD assumptions it embeds.  No Z math lives here.

    ``cfg`` is required, and positional, exactly as it is for
    ``gpuwm.core.refl.compute_refl_10cm(state, cfg, ...)``: the operator
    is only meaningful once the active microphysics scheme is known, and
    ``DomainState`` does not carry its own ``RunConfig``.

    ``temperature``/``pressure`` are the microphysics-time pair WRF hands
    the diagnostic (``t1d``/``p1d``); supply both or neither.  Omitting
    them diagnoses temperature from the current state, which is the
    right choice for a DA operator evaluated between steps.

    ``mp_physics=18`` dispatches to the product's NSSL ``radardd02``
    diagnostic exactly as the production coordinator does (CUDA only, no
    host mirror; see :func:`_nssl_reflectivity` for the two caveats a DA
    caller has to carry -- a 0 dBZ clear-air floor rather than -35, and the
    per-mass moment convention).  It is a supported scheme, not mapped onto
    Morrison or Thompson.

    On a CuPy state this returns the state's persistent ``refl_10cm``
    scratch slot, owned by the state and reused on the next call --
    copy it if you need to keep it.  On a NumPy state it returns a fresh
    float32 array.
    """
    if state.qv is None:
        raise ValueError("simulated reflectivity requires a moist state")
    if (temperature is None) != (pressure is None):
        raise ValueError("temperature and pressure must be supplied together")

    xp = _array_module(state.p)
    if xp is np:
        return _host_reflectivity(state, cfg, temperature, pressure)

    if int(cfg.mp_physics) == 18:
        if pressure is not None:
            raise ValueError(
                "the NSSL radardd02 diagnostic diagnoses its own density "
                "and reads pressure off the state; it does not take the "
                "t1d/p1d microphysics-time pair the 1/6/8/10 routes do")
        return _nssl_reflectivity(state, temperature, pressure)

    from gpuwm.core.refl import compute_refl_10cm

    return compute_refl_10cm(
        state, cfg, temperature=temperature, pressure=pressure,
        thompson_graupel_number=thompson_graupel_number)


# --------------------------------------------------------------------------
# fall speed
# --------------------------------------------------------------------------

#: The precipitating species the fall-speed gate reads, in the order the
#: reflectivity authority reads them.  Cloud water and cloud ice are
#: excluded: they do not sediment at radar fall speeds, and the bound Z
#: formulations read no cloud moment.
PRECIPITATING_SPECIES = ("qr", "qs", "qg", "qh")


def precipitating_mixing_ratio(state):
    """Total precipitating water ``(nz, ny, nx)``: qr + qs + qg (+ qh).

    A DIAGNOSTIC, and no longer the activity gate.  Summing the species
    and thresholding the sum is not what the reflectivity authority does
    and not what this module claimed: see
    :func:`precipitating_activity_mask`.
    """
    total = None
    for name in PRECIPITATING_SPECIES:
        value = getattr(state, name, None)
        if value is None:
            continue
        total = value if total is None else total + value
    if total is None:
        raise ValueError(
            "no precipitating hydrometeor fields (qr/qs/qg/qh) on the state; "
            "pass fall_speed='none' for an air-motion-only radial velocity")
    return total


def precipitating_activity_mask(state):
    """Where ANY precipitating species is active: ``q_species > 1e-9``.

    PER SPECIES, which is what the reflectivity authority this module is
    bound to actually does.  ``refl10cm_hm``
    (module_mp_morr_two_moment.F:4502-4675) and every gpuwm mirror of it
    test each category against ``1e-9 kg/kg`` on its own and contribute
    that category's Ze only where its own test passes; a category below
    the threshold contributes nothing, whatever the other categories are
    carrying.

    Summing ``qr+qs+qg(+qh)`` and thresholding the SUM -- which is what
    this gate did -- is a different rule, and it differs exactly where it
    matters.  Four species at ``3e-10 kg/kg`` each are individually
    inactive by the authority's rule and contribute no reflectivity at
    all, but their ``1.2e-9`` sum cleared the summed gate and switched on
    a 5.244 m/s fall speed in a cell the Z operator calls empty.  A
    Doppler operator that sediments hydrometeors the reflectivity
    operator says are not there is inconsistent with its own H_Z, which is
    the one consistency this module is for.

    The other candidate authority is WRFDA's Sun-Crook implementation,
    which gates on RAIN ONLY above ``1e-5 kg/kg``.  That is a defensible
    rule and it is NOT the one implemented here: this operator's Z comes
    from the model's own multi-species scheme, so gating it on rain alone
    would switch the fall speed off in snow and graupel the same Z
    operator is reporting.  The authority chosen is the reflectivity
    authority, because it is the field being consumed.
    """
    xp = None
    mask = None
    for name in PRECIPITATING_SPECIES:
        value = getattr(state, name, None)
        if value is None:
            continue
        if xp is None:
            xp = _array_module(value)
        active = value > Q_ACTIVE_THRESHOLD
        mask = active if mask is None else (mask | active)
    if mask is None:
        raise ValueError(
            "no precipitating hydrometeor fields (qr/qs/qg/qh) on the state; "
            "pass fall_speed='none' for an air-motion-only radial velocity")
    return mask


def reflectivity_fall_speed(dbz, pressure, active=None, *,
                            surface_pressure=None,
                            max_ms: float = FALL_SPEED_MAX_MS):
    """Sun and Crook reflectivity-driven fall speed (m/s, downward).

    ``surface_pressure`` is the ``(ny, nx)`` pressure at the ground, in Pa,
    and it is REQUIRED.  Sun and Crook's density factor is ``(ps/p)**0.4``
    with ``ps`` the surface pressure, exactly as WRFDA's
    ``da_radial_velocity.inc`` takes it -- a separate ``ps`` argument, then
    ``alpha = (ps/p)**0.4``.  Substituting the thermodynamic reference
    constant ``P0 = 1000 hPa`` is a different closure and a systematic one:
    the ratio of the two is ``(100000/ps)**0.4``, constant through the
    column, so it is +4.3% at a 900 hPa surface, +9.3% at 800 hPa and
    +15.3% at 700 hPa, always making ``vt`` too large and always pushing
    the positive-away radial velocity inbound by ``-dvt*sin(el)``.  There
    is no default because a constant is not a field.

    Sun and Crook state two relations for Marshall-Palmer rain, both with
    ``rho*qr`` in g m-3 and ``a = (ps/p)**0.4``:

        Z[dBZ] = 43.1 + 17.5*log10(rho*qr)
        vt     = 5.40 * a * (rho*qr)**0.125

    Eliminating ``rho*qr`` gives the form used here, which consumes the
    bound Z operator's output directly:

        vt = 5.40 * (ps/p)**0.4 * 10**(0.125*(Z - 43.1)/17.5)

    The Z leg is not an outside assumption: it *is* the Smith et al.
    (1975) exponential-PSD form gpuwm's Kessler fallback implements
    (gpuwm/core/refl.py:30-43).  At ``rho*qr = 1 g m-3`` that PSD gives
    ``lambda = (pi*1000*8e6/1e-3)**0.25 = 2239 m-1`` and
    ``Ze = 720*8e6/lambda**7 * 1e18 = 2.04e4 mm^6 m^-3`` = 43.1 dBZ,
    exactly Sun and Crook's intercept.  So on the ``mp_physics=1`` path
    the fall speed and the reflectivity share one PSD; on the Morrison,
    WSM6 and Thompson paths their richer PSDs supply Z while the fall
    speed is still read off the Marshall-Palmer rain calibration, which
    is the documented approximation (see the module docstring).

    Because the relation is a power law it does not vanish in clear air
    (~1.7 m/s at the -35 dBZ floor), so ``active`` gates it to cells the
    reflectivity authority calls active.  ``active`` is a BOOLEAN mask,
    normally :func:`precipitating_activity_mask` -- ANY species above
    ``1e-9 kg/kg``, tested per species exactly as ``refl10cm_hm`` tests
    them.  A float mixing-ratio array is REFUSED rather than thresholded:
    this argument used to be the summed ``qr+qs+qg(+qh)`` field, gated at
    ``1e-9``, and four species at ``3e-10`` each -- every one of them
    inactive to the Z operator -- summed past it and switched on a
    5.244 m/s fall speed.  Accepting the old argument silently would keep
    that behaviour reachable from every existing caller.

    Non-finite dBZ (Morrison's invalid-number-moment signal) propagates as
    non-finite fall speed rather than being silently zeroed.
    """
    xp = _array_module(dbz)
    if max_ms <= 0.0 or not np.isfinite(max_ms):
        raise ValueError(f"max_ms must be positive and finite, got {max_ms}")
    if surface_pressure is None:
        raise ValueError(
            "reflectivity_fall_speed needs surface_pressure (Pa, (ny, nx)): "
            "Sun and Crook's density factor is (ps/p)**0.4 with ps the "
            "pressure at the ground, not the 1000 hPa thermodynamic "
            "reference.  Substituting the constant is a +4.3% error in vt "
            "over a 900 hPa surface and +15.3% over a 700 hPa one, always "
            "in the same direction, so it is not something to default")

    ps = xp.asarray(surface_pressure)
    if ps.ndim == 2:
        ps = ps[None, :, :]
    elif ps.ndim != 3:
        raise ValueError(
            f"surface_pressure must be (ny, nx) or (nz, ny, nx), got "
            f"shape {tuple(ps.shape)}")
    expected = tuple(np.asarray(_to_host(xp, dbz)).shape)[1:]
    if tuple(np.asarray(_to_host(xp, ps)).shape)[1:] != expected:
        raise ValueError(
            f"surface_pressure {tuple(np.asarray(_to_host(xp, ps)).shape)} "
            f"does not sit on the {expected} horizontal grid the "
            "reflectivity does")
    density_factor = xp.power(ps.astype(dbz.dtype, copy=False) / pressure,
                              FALL_SPEED_DENSITY_EXPONENT)
    exponent = (SUN_CROOK_VT_EXPONENT * (dbz - SUN_CROOK_Z_INTERCEPT_DBZ)
                / SUN_CROOK_Z_SLOPE_DB)
    vt = (SUN_CROOK_VT_COEFF * density_factor
          * xp.power(xp.asarray(10.0, dtype=dbz.dtype), exponent))
    vt = xp.clip(vt, 0.0, max_ms)
    if active is not None:
        mask = xp.asarray(active)
        if mask.dtype != bool:
            raise ValueError(
                "the fall-speed activity gate takes a BOOLEAN mask, got "
                f"dtype {mask.dtype}. Build it with "
                "gpuwm.da.obsop.precipitating_activity_mask(state), which "
                "tests each precipitating species against "
                f"{Q_ACTIVE_THRESHOLD:g} kg/kg the way the reflectivity "
                "authority does. Passing the summed mixing ratio here "
                "gated the SUM, which activates cells whose every species "
                "is inactive")
        vt = xp.where(mask, vt, xp.zeros_like(vt))
    return vt


# --------------------------------------------------------------------------
# radial velocity
# --------------------------------------------------------------------------

def radial_velocity(state, site: RadarSite, geometry: GridGeometry = None, *,
                    cfg=None, reflectivity_dbz=None,
                    fall_speed="reflectivity", surface_pressure=None,
                    effective_earth_ratio: float = EFFECTIVE_EARTH_RATIO,
                    rotate_to_earth: bool = True,
                    fall_speed_max_ms: float = FALL_SPEED_MAX_MS,
                    dtype=None):
    """H_Vr(x): Doppler radial velocity ``(nz, ny, nx)`` in m/s.

    Positive AWAY from the radar (outbound), the standard convention.

        Vr = u_e*cos(el)*sin(az) + v_n*cos(el)*cos(az) + (w - vt)*sin(el)

    with elevation and azimuth taken at the model mass point under 4/3-
    earth beam geometry (:func:`beam_geometry`), winds destaggered to
    mass points and rotated earth-relative, and ``vt`` the downward
    hydrometeor fall speed.

    ``geometry`` may be omitted only if the state carries ``xlat``/
    ``xlong``; otherwise :meth:`GridGeometry.from_state` raises rather
    than guessing where the domain is.

    ``fall_speed`` is one of:

    - ``"reflectivity"`` (default) -- Sun and Crook (1997) from the bound
      Z operator's own dBZ.  Supply ``reflectivity_dbz=`` (cheapest, and
      what an ensemble filter already has in hand) or ``cfg=`` to let
      this call :func:`simulated_reflectivity` itself.  Supplying
      neither raises: silently dropping the fall-speed term would change
      the operator's physics without saying so.  This closure also needs
      ``surface_pressure=`` -- ``(ny, nx)`` in Pa -- for its density
      factor, on the same argument.
    - ``"none"`` -- pure air motion, ``vt = 0``.  A documented
      simplification, not a default.
    - an array ``(nz, ny, nx)`` -- caller's own closure, m/s downward.

    ``dtype`` defaults to the state's wind dtype (float32).  Pass
    ``numpy.float64`` for analytic work where round-off matters.
    """
    if geometry is None:
        geometry = GridGeometry.from_state(state)

    u_e, v_n, w_mass = mass_point_winds(state, rotate_to_earth=rotate_to_earth)
    shape = tuple(np.asarray(_to_host(_array_module(w_mass), w_mass)).shape)
    if geometry.shape != shape:
        raise ValueError(
            f"grid geometry {geometry.shape} does not match the state's "
            f"mass-point shape {shape}")

    out_dtype = np.dtype(dtype) if dtype is not None else np.dtype(np.float32)
    xp = _array_module(w_mass)

    beam = beam_geometry(geometry, site,
                         effective_earth_ratio=effective_earth_ratio)
    east, north, up = beam.unit_vector_enu()
    if xp is not np:
        east = xp.asarray(east, dtype=out_dtype)
        north = xp.asarray(north, dtype=out_dtype)
        up = xp.asarray(up, dtype=out_dtype)
    else:
        east = east.astype(out_dtype, copy=False)
        north = north.astype(out_dtype, copy=False)
        up = up.astype(out_dtype, copy=False)

    vt = _resolve_fall_speed(state, cfg, reflectivity_dbz, fall_speed,
                             shape, xp, out_dtype, fall_speed_max_ms,
                             surface_pressure)

    u_e = u_e.astype(out_dtype, copy=False)
    v_n = v_n.astype(out_dtype, copy=False)
    w_mass = w_mass.astype(out_dtype, copy=False)

    vertical = w_mass if vt is None else (w_mass - vt)
    return (u_e * east + v_n * north + vertical * up).astype(out_dtype,
                                                             copy=False)


def _resolve_fall_speed(state, cfg, reflectivity_dbz, fall_speed, shape, xp,
                        out_dtype, max_ms, surface_pressure=None):
    """Turn the ``fall_speed`` argument into an array or None."""
    if fall_speed == "none":
        return None
    if fall_speed == "reflectivity":
        if reflectivity_dbz is None:
            if cfg is None:
                raise ValueError(
                    "fall_speed='reflectivity' needs the bound Z operator's "
                    "output: pass reflectivity_dbz= (preferred -- an "
                    "ensemble filter already has it) or cfg= so this call "
                    "can evaluate simulated_reflectivity itself, or ask for "
                    "fall_speed='none' to accept the air-motion-only "
                    "simplification explicitly")
            reflectivity_dbz = simulated_reflectivity(state, cfg)
        dbz = reflectivity_dbz.astype(out_dtype, copy=False)
        pressure = state.p.astype(out_dtype, copy=False)
        vt = reflectivity_fall_speed(
            dbz, pressure, precipitating_activity_mask(state),
            surface_pressure=surface_pressure, max_ms=max_ms)
        return vt.astype(out_dtype, copy=False)
    if isinstance(fall_speed, str):
        raise ValueError(
            f"fall_speed must be 'reflectivity', 'none', or an array, "
            f"got {fall_speed!r}")
    vt = fall_speed
    vt_shape = tuple(np.asarray(_to_host(_array_module(vt), vt)).shape)
    if vt_shape != shape:
        raise ValueError(
            f"fall_speed array {vt_shape} must match the mass-point "
            f"shape {shape}")
    return vt.astype(out_dtype, copy=False)
