"""Feature targets for a moving child: follow a severe-weather SETUP, not only a storm.

``gpuwm downscale --follow target --follow-target T`` re-places a step-down on a track this module fixes on the
parent's own frames (model state only, never observations):

* ``low``         a surface low: the smoothed sea-level-pressure minimum and its depression centroid;
* ``cape-shear``  the dominant area of mixed-layer CAPE x 0-6 km bulk shear
                  (index = MLCAPE/1000 J kg-1 x S06/20 m s-1);
* ``boundary``    a dryline or front: the dominant 2 m dewpoint / theta-e / temperature gradient segment, oriented
                  like the segment the user drew (``--follow-segment``), then following its own orientation;
* ``object``      a thresholded feature (REFL_10CM column maximum, or UP_HELI_MAX), the dominant object near the
                  prediction;
* ``path``        the user's own timed path (``--follow-path``): the child goes where the user said. Not a tracker.

The same targets and the same equations as gpuwm-hex's moving region (hexcore.storm_tracker), on the WRF frame's own
grid. Each frame's target field is computed in a process pool over the frames (all cores); the fixing walk itself is
sequential and reads only the small 2-D fields.
"""
from __future__ import annotations

import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

TARGETS = ("low", "cape-shear", "boundary", "object", "path")
G = 9.81
RD = 287.04
CP = 1004.6
EARTH_RADIUS_KM = 6371.229


class SetupTargetRefusal(ValueError):
    """A feature target that cannot be fixed, and why."""


@dataclass
class TargetConfig:
    target: str
    stride_km: float = 15.0            # the target field's sampling on the frame (cape-shear, low)
    search_km: float = 300.0
    search_growth_kmh: float = 50.0
    smooth_km: float = 40.0
    threshold: float = 0.25
    min_area_km2: float = 5000.0
    distance_scale_km: float = 250.0
    size_power: float = 0.5
    fix_blend: float = 0.6
    max_speed_kmh: float = 90.0
    boundary_field: str = "dewpoint"
    segment: list | None = None
    object_field: str = "REFL_10CM"
    path: list = field(default_factory=list)
    # cape-shear, carried from gpuwm-hex's moving region (hex/follow-extend 7adab159, ac0aff54, 35e19ea0), where the
    # real run found each (GFS 2026-10-23 18Z f354, 3 km, box L 2026-10-24): one connected area grew to 570,000 km2
    # and its centroid drifted 700 km off the storms (the target is the core: >= core_fraction x the peak, taken
    # within core_reach_km of the prediction); deep convection used up the CAPE it grew from so the index core slid
    # off the storms (column-max reflectivity >= storm_dbz within storm_reach_km of the area pulls the centre); after
    # dark the area's peak was an ocean plume 900 km away and fixes ran 100-150 km per 30 min (one fix moves at most
    # max_step_kmh x the hours since the last fix).
    core_fraction: float = 0.6
    core_reach_km: float | None = 200.0
    storm_field: str = "REFL_10CM"
    storm_dbz: float = 35.0
    storm_weight: float = 1.0
    storm_reach_km: float = 150.0
    max_step_kmh: float | None = 90.0

    @classmethod
    def for_target(cls, target: str, **overrides):
        target = str(target).replace("_", "-")
        if target not in TARGETS:
            raise SetupTargetRefusal(f"--follow-target must be one of {', '.join(TARGETS)}")
        base = {
            "low": dict(stride_km=15.0, search_km=300.0, search_growth_kmh=60.0, smooth_km=75.0, threshold=1.0,
                        min_area_km2=0.0, distance_scale_km=400.0, fix_blend=0.8),
            "cape-shear": dict(),
            "boundary": dict(stride_km=6.0, search_km=200.0, search_growth_kmh=40.0, smooth_km=18.0, threshold=4.0,
                             min_area_km2=2000.0, distance_scale_km=150.0, fix_blend=0.7),
            "object": dict(stride_km=3.0, search_km=100.0, search_growth_kmh=25.0, smooth_km=6.0, threshold=35.0,
                           min_area_km2=200.0, distance_scale_km=60.0, fix_blend=1.0),
            "path": dict(),
        }[target]
        base.update({k: v for k, v in overrides.items() if v is not None})
        return cls(target=target, **base)


# --------------------------------------------------------------------------- thermodynamics (Bolton 1980)

def _saturation_mixing_ratio(t, p):
    es = 611.2 * np.exp(17.67 * (t - 273.15) / np.maximum(t - 29.65, 1.0))
    return 0.622 * es / np.maximum(p - es, 1.0)


def _theta_e(t, p, r, t_lcl):
    rg = r * 1000.0
    return (t * (100000.0 / p) ** (0.2854 * (1.0 - 0.00028 * rg))
            * np.exp((3.376 / t_lcl - 0.00254) * rg * (1.0 + 0.00081 * rg)))


def _lcl_temperature(t, r, p):
    e = np.maximum(r * p / (0.622 + r), 1.0) / 100.0
    return 2840.0 / (3.5 * np.log(t) - np.log(e) - 4.805) + 55.0


def mlcape_and_shear(p, theta, qv, u, v, psfc, mixed_layer_pa=10000.0, shear_depth_m=6000.0):
    """Mixed-layer CAPE (J/kg) and 0-6 km bulk shear (m/s) of columns (n, nlev), level 0 at the bottom."""
    kappa = RD / CP
    qv = np.maximum(qv, 1e-8)
    t = theta * (p / 1.0e5) ** kappa
    tv = t * (1.0 + 0.608 * qv)
    z = np.empty_like(p)
    z[:, 0] = RD * tv[:, 0] / G * np.log(np.maximum(psfc, p[:, 0] + 1.0) / p[:, 0])
    z[:, 1:] = z[:, :1] + np.cumsum(RD * 0.5 * (tv[:, 1:] + tv[:, :-1]) / G * np.log(p[:, :-1] / p[:, 1:]), axis=1)
    ml = p >= (psfc[:, None] - mixed_layer_pa)
    ml[:, 0] = True
    count = ml.sum(axis=1)
    th_ml = (theta * ml).sum(axis=1) / count
    r_ml = ((qv / (1.0 - qv)) * ml).sum(axis=1) / count
    t_start = th_ml * (psfc / 1.0e5) ** kappa
    t_lcl = _lcl_temperature(t_start, r_ml, psfc)
    p_lcl = psfc * (t_lcl / t_start) ** (1.0 / kappa)
    thetae = _theta_e(t_start, psfc, r_ml, t_lcl)
    dry = th_ml[:, None] * (p / 1.0e5) ** kappa
    lo, hi = np.full_like(p, 150.0), np.full_like(p, 350.0)
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        warm = _theta_e(mid, p, _saturation_mixing_ratio(mid, p), mid) > thetae[:, None]
        hi = np.where(warm, mid, hi)
        lo = np.where(warm, lo, mid)
    moist = 0.5 * (lo + hi)
    above = p < p_lcl[:, None]
    tp = np.where(above, moist, dry)
    rp = np.where(above, _saturation_mixing_ratio(moist, p), r_ml[:, None])
    buoyancy = G * (tp * (1.0 + 0.608 * rp / (1.0 + rp)) - tv) / tv
    dz = np.diff(z, axis=1, prepend=0.0)
    cape = np.where((p >= 10000.0) & (z > 0), np.maximum(buoyancy, 0.0) * dz, 0.0).sum(axis=1)
    reach = z >= shear_depth_m
    k1 = np.argmax(reach, axis=1)
    ok = reach.any(axis=1) & (k1 > 0)
    k1 = np.where(ok, k1, 1)
    k0 = k1 - 1
    rows = np.arange(p.shape[0])
    w = (shear_depth_m - z[rows, k0]) / np.maximum(z[rows, k1] - z[rows, k0], 1e-3)
    ut = u[rows, k0] + w * (u[rows, k1] - u[rows, k0])
    vt = v[rows, k0] + w * (v[rows, k1] - v[rows, k0])
    return cape, np.where(ok, np.hypot(ut - u[:, 0], vt - v[:, 0]), np.nan)


def surface_scalar(t2, q2, psfc, name):
    if name == "temperature":
        return t2
    q = np.maximum(q2, 1e-8)
    r = q / (1.0 - q)
    e = np.maximum(r * psfc / (0.622 + r), 1.0)
    if name == "dewpoint":
        ln = np.log(e / 611.2)
        return 273.15 + 243.5 * ln / (17.67 - ln)
    if name == "theta_e":
        return _theta_e(t2, psfc, r, _lcl_temperature(t2, r, psfc))
    raise SetupTargetRefusal(f"unknown boundary field {name!r} (dewpoint, theta_e, temperature)")


# --------------------------------------------------------------------------- one frame -> its target field

def _stride(dx_km, stride_km):
    return max(1, int(round(stride_km / max(dx_km, 1e-3))))


def frame_field(path: str, config: dict) -> dict:
    """The target's 2-D field on one frame, sampled every ``stride`` cells (lat, lon, value, spacing_km)."""
    from gpuwm.io.netcdf_serialization import netCDF4

    target = config["target"]
    with netCDF4.Dataset(path) as d:
        d.set_auto_mask(False)
        dx_km = float(d.getncattr("DX")) / 1000.0
        s = _stride(dx_km, config["stride_km"])
        lat = np.asarray(d["XLAT"][0, ::s, ::s], np.float64)
        lon = np.asarray(d["XLONG"][0, ::s, ::s], np.float64)
        if target == "cape-shear":
            p = np.asarray(d["P"][0, :, ::s, ::s], np.float64) + np.asarray(d["PB"][0, :, ::s, ::s], np.float64)
            theta = np.asarray(d["T"][0, :, ::s, ::s], np.float64) + 300.0
            qv = np.asarray(d["QVAPOR"][0, :, ::s, ::s], np.float64)
            u = np.asarray(d["U"][0], np.float64)
            v = np.asarray(d["V"][0], np.float64)
            u = (0.5 * (u[:, :, :-1] + u[:, :, 1:]))[:, ::s, ::s]
            v = (0.5 * (v[:, :-1, :] + v[:, 1:, :]))[:, ::s, ::s]
            psfc = np.asarray(d["PSFC"][0, ::s, ::s], np.float64)
            nz, ny, nx = p.shape
            col = lambda a: a.reshape(nz, -1).T
            cape, shear = mlcape_and_shear(col(p), col(theta), col(qv), col(u), col(v), psfc.ravel())
            value = ((cape / 1000.0) * (shear / 20.0)).reshape(ny, nx)
            extra = dict(max_mlcape=float(np.nanmax(cape)), max_shear06=float(np.nanmax(shear)))
            storm_field = config.get("storm_field")
            if storm_field and storm_field in d.variables:      # optional: a frame without it fixes on the index
                raw = np.asarray(d[storm_field][0], np.float64)
                extra["storms"] = (raw.max(axis=0) if raw.ndim == 3 else raw)[::s, ::s]
        elif target == "low":
            psfc = np.asarray(d["PSFC"][0, ::s, ::s], np.float64)
            hgt = np.asarray(d["HGT"][0, ::s, ::s], np.float64)
            t2 = np.asarray(d["T2"][0, ::s, ::s], np.float64)
            t_mean = t2 + 0.0065 * hgt / 2.0
            value = psfc * np.exp(G * hgt / (RD * np.maximum(t_mean, 150.0))) / 100.0   # hPa
            extra = {}
        elif target == "boundary":
            t2 = np.asarray(d["T2"][0, ::s, ::s], np.float64)
            q2 = np.asarray(d["Q2"][0, ::s, ::s], np.float64)
            psfc = np.asarray(d["PSFC"][0, ::s, ::s], np.float64)
            value = surface_scalar(t2, q2, psfc, config["boundary_field"])
            extra = {}
        elif target == "object":
            name = config["object_field"]
            if name not in d.variables:
                raise SetupTargetRefusal(f"{Path(path).name} has no {name}; pick a field the parent wrote")
            raw = np.asarray(d[name][0], np.float64)
            value = (raw.max(axis=0) if raw.ndim == 3 else raw)[::s, ::s]
            extra = {}
        else:
            raise SetupTargetRefusal(f"target {target!r} reads no frame field")
    return dict(lat=lat, lon=lon, value=value, spacing_km=dx_km * s, **extra)


# --------------------------------------------------------------------------- fixing on a frame's grid

def _xy_km(lat, lon, lat0, lon0):
    """Azimuthal-equidistant (x east, y north) km about (lat0, lon0)."""
    p0, l0 = math.radians(lat0), math.radians(lon0)
    p, l = np.radians(lat), np.radians(lon)
    cosc = np.clip(math.sin(p0) * np.sin(p) + math.cos(p0) * np.cos(p) * np.cos(l - l0), -1, 1)
    c = np.arccos(cosc)
    k = np.where(c > 1e-12, c / np.where(np.sin(c) == 0, 1, np.sin(c)), 1.0)
    return (EARTH_RADIUS_KM * k * np.cos(p) * np.sin(l - l0),
            EARTH_RADIUS_KM * k * (math.cos(p0) * np.sin(p) - math.sin(p0) * np.cos(p) * np.cos(l - l0)))


def _latlon_from_xy(lat0, lon0, x, y):
    p0, l0 = math.radians(lat0), math.radians(lon0)
    x, y = x / EARTH_RADIUS_KM, y / EARTH_RADIUS_KM
    rho = math.hypot(x, y)
    if rho < 1e-15:
        return lat0, lon0
    c = rho
    lat = math.asin(math.cos(c) * math.sin(p0) + y * math.sin(c) * math.cos(p0) / rho)
    lon = l0 + math.atan2(x * math.sin(c), rho * math.cos(p0) * math.cos(c) - y * math.sin(p0) * math.sin(c))
    return math.degrees(lat), (math.degrees(lon) + 180.0) % 360.0 - 180.0


def _smooth(f, sigma):
    from scipy.ndimage import gaussian_filter
    good = np.isfinite(f)
    if sigma <= 0:
        return f
    num = gaussian_filter(np.where(good, f, 0.0), sigma, mode="nearest")
    den = gaussian_filter(good.astype(float), sigma, mode="nearest")
    return np.where(den > 0.2, num / np.where(den > 0, den, 1), np.nan)


def _axis_deg(x, y, w):
    s = w.sum()
    x0, y0 = (w * x).sum() / s, (w * y).sum() / s
    dx, dy = x - x0, y - y0
    cxx, cyy, cxy = (w * dx * dx).sum() / s, (w * dy * dy).sum() / s, (w * dx * dy).sum() / s
    return (90.0 - math.degrees(0.5 * math.atan2(2 * cxy, cxx - cyy))) % 180.0


def fix_frame(frame: dict, config: TargetConfig, guess, search_km, orientation=None, hours=None) -> dict:
    """One fix on one frame's target field near ``guess``: {ok, lat, lon, reason, ...}.

    ``hours`` since the last good fix bounds a cape-shear fix's step (max_step_kmh); None leaves it unbounded."""
    from scipy.ndimage import label
    x, y = _xy_km(frame["lat"], frame["lon"], guess[0], guess[1])
    spacing = frame["spacing_km"]
    value = _smooth(np.asarray(frame["value"], float), config.smooth_km / spacing)
    near = np.hypot(x, y) <= search_km + 2 * config.smooth_km
    if not near.any():
        return dict(ok=False, reason="the prediction is outside the parent frame")
    if config.target == "low":
        inside = np.where(np.hypot(x, y) <= search_km, value, np.nan)
        if not np.isfinite(inside).any():
            return dict(ok=False, reason="no pressure inside the search radius")
        j = np.nanargmin(inside)
        x0, y0 = float(x.flat[j]), float(y.flat[j])
        ring = (np.hypot(x - x0, y - y0) >= 500.0) & (np.hypot(x - x0, y - y0) <= 900.0) & np.isfinite(value)
        depression = (float(np.nanmean(value[ring])) if ring.any() else float(np.nanmax(value))) - float(inside.flat[j])
        if depression < config.threshold:
            return dict(ok=False, reason=f"pressure minimum only {depression:.1f} hPa below its ring", min_hpa=float(inside.flat[j]))
        w = np.where((np.hypot(x - x0, y - y0) <= 200.0) & np.isfinite(value),
                     np.maximum(float(inside.flat[j]) + depression - value, 0.0), 0.0)
        xc, yc = float((w * x).sum() / w.sum()), float((w * y).sum() / w.sum())
        lat, lon = _latlon_from_xy(guess[0], guess[1], config.fix_blend * xc, config.fix_blend * yc)
        return dict(ok=True, lat=lat, lon=lon, reason="surface low", min_hpa=float(inside.flat[j]),
                    depression_hpa=depression)
    if config.target == "boundary":
        gy, gx = np.gradient(value, spacing / 100.0)
        value = np.hypot(gx, gy)
    mask = near & np.isfinite(value) & (value >= config.threshold)
    labels, n = label(mask)
    best = None
    cell = spacing ** 2
    for k in range(1, n + 1):
        m = labels == k
        area = m.sum() * cell
        if area < config.min_area_km2:
            continue
        w = np.where(m, value - config.threshold + 1e-6, 0.0)
        xc, yc = float((w * x).sum() / w.sum()), float((w * y).sum() / w.sum())
        d = math.hypot(xc, yc)
        if d > search_km:
            continue
        score = float(w.sum() * cell) ** config.size_power * math.exp(-(d / config.distance_scale_km) ** 2)
        axis = None
        if config.target == "boundary":
            axis = _axis_deg(x, y, w)
            if orientation is not None:
                score *= abs(math.cos(math.radians(axis - orientation))) + 1e-3
        if best is None or score > best[0]:
            best = (score, xc, yc, area, float(value[m].max()), axis, m)
    if best is None:
        return dict(ok=False, reason=("nothing above the threshold" if n == 0
                                      else "nothing large enough inside the search radius"))
    _, xc, yc, area, peak, axis, m = best
    extra, reason = {}, f"dominant {config.target} area"
    if config.target == "cape-shear":
        local = m if config.core_reach_km is None else m & (np.hypot(x, y) <= config.core_reach_km)
        if local.any():
            peak = float(value[local].max())
        else:
            local = m
        core_level = max(config.threshold, config.core_fraction * peak)
        core = local & (value >= core_level)
        w = (np.where(core, value - core_level + 1e-6, 0.0) if core.any()
             else np.where(local, value - config.threshold + 1e-6, 0.0))
        storm_km2, storm_share = 0.0, 0.0
        storms = frame.get("storms")
        if storms is not None and config.storm_weight > 0:
            from scipy.ndimage import distance_transform_edt
            hot = np.asarray(storms, float) >= config.storm_dbz
            close = (distance_transform_edt(~m) * spacing <= config.storm_reach_km) & (np.hypot(x, y) <= search_km)
            ws = np.where(close, config.storm_weight * peak * _smooth(hot.astype(float), config.smooth_km / spacing), 0.0)
            ws = np.nan_to_num(ws)
            storm_km2 = float((hot & close).sum() * cell)
            if ws.sum() > 0:
                storm_share = float(ws.sum() / (ws.sum() + w.sum()))
                w = w + ws
        xc, yc = float((w * x).sum() / w.sum()), float((w * y).sum() / w.sum())
        reason = "core of the dominant cape-shear area" + (" with its storms" if storm_share > 0 else "")
        extra = dict(core_km2=float(core.sum() * cell), storm_km2=storm_km2, storm_share=storm_share)
    dx, dy = config.fix_blend * xc, config.fix_blend * yc
    if config.target == "cape-shear" and config.max_step_kmh is not None and hours is not None:
        cap, step = config.max_step_kmh * max(hours, 0.25), math.hypot(dx, dy)
        extra.update(step_km=step, step_capped=bool(step > cap))
        if step > cap:
            dx, dy = dx * cap / step, dy * cap / step
    lat, lon = _latlon_from_xy(guess[0], guess[1], dx, dy)
    raw = _latlon_from_xy(guess[0], guess[1], xc, yc)
    return dict(ok=True, lat=lat, lon=lon, reason=reason, area_km2=area, peak=peak,
                axis_deg=axis, raw_centre=list(raw), **extra)


# --------------------------------------------------------------------------- the walk over the frames

def read_path(path: Path) -> list[tuple[datetime, float, float]]:
    """A user's path: JSON {"rows":[{"time","lat","lon"}...]} or CSV with valid_time/time, lat(_deg), lon(_deg)."""
    path = Path(path)
    rows = []
    if path.suffix.lower() == ".json":
        document = json.loads(path.read_text(encoding="utf-8"))
        for row in document.get("rows", document if isinstance(document, list) else []):
            rows.append((_when(row.get("valid", row.get("time"))), float(row["lat"]), float(row["lon"])))
    else:
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append((_when(row.get("valid_time") or row.get("time")),
                             float(row.get("lat_deg", row.get("lat"))), float(row.get("lon_deg", row.get("lon")))))
    if len(rows) < 2:
        raise SetupTargetRefusal("a path needs at least two timed points")
    return sorted(rows)


def _when(text) -> datetime:
    text = str(text).replace("Z", "").replace("T", "_")
    for fmt in ("%Y-%m-%d_%H:%M:%S", "%Y-%m-%d_%H:%M", "%Y-%m-%d_%H"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise SetupTargetRefusal(f"unreadable path time {text!r}")


def _workers(n):
    return max(1, min(n, os.cpu_count() or 1))


def follow_track(frames, times, config: TargetConfig, start_point) -> tuple[list, list]:
    """Fix the target on every parent frame, in time order. Returns (track rows (time, lat, lon), diagnostics)."""
    if config.target == "path":
        rows = list(config.path)
        return rows, [dict(valid_time=t.strftime("%Y-%m-%dT%H:%M:%SZ"), ok=True, reason="the user's path (not a tracker)")
                      for t, _, _ in rows]
    if start_point is None:
        raise SetupTargetRefusal("--follow target needs --point (or --follow-segment): where the feature is at the start")
    payload = dict(target=config.target, stride_km=config.stride_km, boundary_field=config.boundary_field,
                   object_field=config.object_field, storm_field=config.storm_field)
    with ProcessPoolExecutor(max_workers=_workers(len(frames))) as pool:
        fields = list(pool.map(frame_field, [str(f) for f in frames], [payload] * len(frames)))
    orientation = None
    if config.segment:
        (a_lat, a_lon), (b_lat, b_lon) = config.segment
        bx, by = _xy_km(np.array(b_lat), np.array(b_lon), a_lat, a_lon)
        orientation = math.degrees(math.atan2(float(bx), float(by))) % 180.0
    good, track, diagnostics = [], [], []
    for when, frame in zip(times, fields):
        if good:
            last = good[-1]
            hours = (when - last[0]).total_seconds() / 3600.0
            if len(good) >= 2:
                prev = good[-2]
                span = (last[0] - prev[0]).total_seconds() / 3600.0
                vx, vy = _xy_km(np.array(last[1]), np.array(last[2]), prev[1], prev[2])
                speed = math.hypot(float(vx), float(vy)) / max(span, 1e-6)
                scale = min(1.0, config.max_speed_kmh / speed) if speed > 0 else 1.0
                guess = _latlon_from_xy(last[1], last[2], float(vx) / span * hours * scale, float(vy) / span * hours * scale)
            else:
                guess = (last[1], last[2])
            since = (when - good[-1][0]).total_seconds() / 3600.0
        else:
            guess, since = tuple(start_point), 0.0
        search = config.search_km + config.search_growth_kmh * max(since, 0.0)
        fix = fix_frame(frame, config, guess, search, orientation, hours=since if good else None)
        if fix.get("axis_deg") is not None:
            orientation = fix["axis_deg"]
        row = dict(valid_time=when.strftime("%Y-%m-%dT%H:%M:%SZ"), guess=list(guess), search_km=search, **fix)
        diagnostics.append(row)
        if fix["ok"]:
            good.append((when, fix["lat"], fix["lon"]))
            track.append((when, fix["lat"], fix["lon"]))
        else:
            # a miss holds the prediction, so the child keeps moving the way the feature was going
            track.append((when, float(guess[0]), float(guess[1])))
    return track, diagnostics


def write_track_csv(path: Path, track) -> None:
    """The storm-track CSV shape (position only), so a further step-down inside this child follows the same target."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        handle.write("valid_time,lat_deg,lon_deg\n")
        for when, lat, lon in track:
            handle.write(f"{when:%Y-%m-%d_%H:%M:%S},{lat:.4f},{lon:.4f}\n")
