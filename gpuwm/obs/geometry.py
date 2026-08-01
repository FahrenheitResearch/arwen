"""Radar beam geometry: where a range gate actually is, and which way it looks.

Everything here is the effective-earth ("4/3-earth") model, which replaces
the real atmosphere's refractive gradient with a straight ray over a sphere
of radius ``k * a``.  It is the standard operational approximation and it is
good to a few tens of metres out to 250 km in a standard atmosphere; it is
*not* good in a strong surface duct, which is a documented limitation of
this stage rather than a bug in it.

Three quantities matter downstream and all three come from the same
triangle, so they are derived together rather than approximated separately:

* **height** of the gate above mean sea level — which model level it lands in;
* **surface arc** from the radar to the gate — which column it lands in;
* **local elevation angle at the gate** — the vertical component of the beam
  unit vector, which is what turns a radial velocity into an observation
  operator.  The beam leaves the antenna at ``el`` but arrives over the
  curve tilted *further* up relative to the local horizontal, and using the
  antenna angle in place of the local one is a silent error that grows with
  range.

The sphere radius is :data:`gpuwm.static.projection.EARTH_RADIUS_M`, not
6371 km: the model grid is georeferenced on WRF's 6370 km sphere, so the
propagation model and the map projection must agree or gates land in the
wrong column at long range for no physical reason.
"""

from __future__ import annotations

import numpy as np

from gpuwm.static.projection import EARTH_RADIUS_M

#: Effective-earth radius multiplier for a standard atmosphere.
REFRACTION_FACTOR = 4.0 / 3.0

_DEG = np.pi / 180.0


def effective_earth_radius_m(*, earth_radius_m: float = EARTH_RADIUS_M,
                             refraction_factor: float = REFRACTION_FACTOR
                             ) -> float:
    """``k * a`` — the radius the straight-ray model propagates over."""

    if not np.isfinite(earth_radius_m) or earth_radius_m <= 0.0:
        raise ValueError(f"earth radius must be positive, got {earth_radius_m}")
    if not np.isfinite(refraction_factor) or refraction_factor <= 0.0:
        raise ValueError(
            f"refraction factor must be positive, got {refraction_factor}")
    return float(earth_radius_m) * float(refraction_factor)


def beam_geometry(slant_range_m, elevation_deg, radar_alt_m: float, *,
                  earth_radius_m: float = EARTH_RADIUS_M,
                  refraction_factor: float = REFRACTION_FACTOR):
    """Gate height MSL, surface arc, and local elevation angle.

    Parameters
    ----------
    slant_range_m
        Range along the beam, metres.  Array-like; broadcasts against
        ``elevation_deg``.
    elevation_deg
        Antenna elevation angle, degrees above the local horizontal *at the
        radar*.
    radar_alt_m
        Antenna height above mean sea level, metres.

    Returns
    -------
    (height_msl_m, surface_arc_m, local_elevation_deg)
        All float64 arrays of the broadcast shape.

    Notes
    -----
    With ``R = k*a + radar_alt`` the ray from the antenna to a gate at slant
    range ``r`` subtends a triangle at the effective-earth centre, so

    ``height above antenna = sqrt(r^2 + R^2 + 2 r R sin(el)) - R``

    and the arc along the surface follows from the same triangle.  The local
    elevation comes from the sine rule in that triangle, ``R cos(el) =
    (R + h) cos(el_local)``, which is exact *within* the effective-earth
    model rather than a small-angle expansion of it.
    """

    slant_range_m = np.asarray(slant_range_m, dtype=np.float64)
    elevation_deg = np.asarray(elevation_deg, dtype=np.float64)
    if np.any(slant_range_m < 0.0):
        raise ValueError("slant range must be non-negative")
    effective_radius = effective_earth_radius_m(
        earth_radius_m=earth_radius_m, refraction_factor=refraction_factor)
    radius = effective_radius + float(radar_alt_m)

    sin_el = np.sin(elevation_deg * _DEG)
    cos_el = np.cos(elevation_deg * _DEG)
    slant_from_centre = np.sqrt(
        slant_range_m * slant_range_m + radius * radius
        + 2.0 * slant_range_m * radius * sin_el)
    height_above_antenna = slant_from_centre - radius
    height_msl = height_above_antenna + float(radar_alt_m)

    # Ground distance, not a central angle.  The effective-earth
    # substitution preserves *arc length along the surface*, which is the
    # whole point of it, so the distance is the effective-earth centre
    # angle times the effective radius.  Scaling by the real radius instead
    # would shorten every range by a factor 1/k = 0.75 and put a 100 km
    # gate 25 km closer to the radar than it is.
    centre_angle = np.arcsin(
        np.clip(slant_range_m * cos_el / slant_from_centre, -1.0, 1.0))
    surface_arc = centre_angle * effective_radius

    cos_local = np.clip(radius * cos_el / slant_from_centre, -1.0, 1.0)
    local_elevation_deg = np.arccos(cos_local) / _DEG
    # arccos is non-negative; a downward-pointing cut keeps its sign.
    local_elevation_deg = np.where(elevation_deg < 0.0,
                                   -local_elevation_deg,
                                   local_elevation_deg)
    return height_msl, surface_arc, local_elevation_deg


def great_circle_point(lat_deg: float, lon_deg: float, azimuth_deg,
                       arc_m, *, earth_radius_m: float = EARTH_RADIUS_M):
    """Forward geodesic on the sphere: (lat, lon) of a point at bearing and arc.

    ``azimuth_deg`` is the compass bearing at the origin, degrees clockwise
    from true north — the radar convention.
    """

    azimuth_deg = np.asarray(azimuth_deg, dtype=np.float64)
    arc_m = np.asarray(arc_m, dtype=np.float64)
    delta = arc_m / float(earth_radius_m)
    lat0 = float(lat_deg) * _DEG
    lon0 = float(lon_deg) * _DEG
    az = azimuth_deg * _DEG

    sin_lat = (np.sin(lat0) * np.cos(delta)
               + np.cos(lat0) * np.sin(delta) * np.cos(az))
    sin_lat = np.clip(sin_lat, -1.0, 1.0)
    lat = np.arcsin(sin_lat)
    lon = lon0 + np.arctan2(
        np.sin(az) * np.sin(delta) * np.cos(lat0),
        np.cos(delta) - np.sin(lat0) * sin_lat)
    lon_deg_out = (lon / _DEG + 180.0) % 360.0 - 180.0
    return lat / _DEG, lon_deg_out


def forward_azimuth(origin_lat_deg: float, origin_lon_deg: float,
                    point_lat_deg, point_lon_deg):
    """Bearing of the outbound great circle *at the far point*.

    Over 250 km the great circle turns by up to a couple of degrees relative
    to the launch bearing, and the observation operator wants the direction
    the beam is travelling where the gate is, not where the antenna is.
    """

    lat1 = float(origin_lat_deg) * _DEG
    lon1 = float(origin_lon_deg) * _DEG
    lat2 = np.asarray(point_lat_deg, dtype=np.float64) * _DEG
    lon2 = np.asarray(point_lon_deg, dtype=np.float64) * _DEG
    dlon = lon2 - lon1
    # Bearing back from the gate to the radar, reversed.
    back = np.arctan2(np.sin(-dlon) * np.cos(lat1),
                      np.cos(lat2) * np.sin(lat1)
                      - np.sin(lat2) * np.cos(lat1) * np.cos(dlon))
    return (back / _DEG + 180.0) % 360.0


def beam_unit_vector(azimuth_deg, local_elevation_deg):
    """Beam direction as (east, north, up) components of a unit vector.

    This is the observation operator for radial velocity:
    ``Vr = u*east + v*north + w*up``.  Shipping it per superob cell is not
    an extra — a radial velocity without its look direction is not an
    observation, it is a number.
    """

    azimuth_deg = np.asarray(azimuth_deg, dtype=np.float64)
    local_elevation_deg = np.asarray(local_elevation_deg, dtype=np.float64)
    cos_el = np.cos(local_elevation_deg * _DEG)
    return (np.sin(azimuth_deg * _DEG) * cos_el,
            np.cos(azimuth_deg * _DEG) * cos_el,
            np.sin(local_elevation_deg * _DEG))


def gate_locations(radar_lat_deg: float, radar_lon_deg: float,
                   radar_alt_m: float, azimuth_deg, slant_range_m,
                   elevation_deg, *,
                   earth_radius_m: float = EARTH_RADIUS_M,
                   refraction_factor: float = REFRACTION_FACTOR):
    """Everything the superob needs about a set of gates, in one pass.

    Returns ``(lat, lon, height_msl_m, east, north, up)`` — position and the
    beam unit vector at the gate, all broadcast to the common shape of
    ``azimuth_deg``, ``slant_range_m`` and ``elevation_deg``.
    """

    height_msl, arc, local_el = beam_geometry(
        slant_range_m, elevation_deg, radar_alt_m,
        earth_radius_m=earth_radius_m, refraction_factor=refraction_factor)
    azimuth_deg = np.broadcast_to(
        np.asarray(azimuth_deg, dtype=np.float64), np.shape(arc))
    lat, lon = great_circle_point(radar_lat_deg, radar_lon_deg, azimuth_deg,
                                  arc, earth_radius_m=earth_radius_m)
    gate_azimuth = forward_azimuth(radar_lat_deg, radar_lon_deg, lat, lon)
    # At zero arc the bearing is undefined; the launch azimuth is the limit.
    gate_azimuth = np.where(arc > 0.0, gate_azimuth, azimuth_deg)
    east, north, up = beam_unit_vector(gate_azimuth, local_el)
    return lat, lon, height_msl, east, north, up
