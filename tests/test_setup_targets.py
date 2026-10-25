"""gpuwm downscale --follow target: a moving child that follows a severe-weather setup.

Synthetic frames carry exactly the fields each target reads; the targets must land where the feature is, frame by
frame, so the child is placed on it.
"""
import json
import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from gpuwm import setup_targets as st


def _frame(path, when, *, dryline_lon=None, low=None, dx_m=6000.0, n=120):
    import netCDF4
    lat0, lon0 = 36.0, -99.0
    dlat = dx_m / 111_000.0
    lat = lat0 + (np.arange(n) - n / 2) * dlat
    lon = lon0 + (np.arange(n) - n / 2) * dlat / math.cos(math.radians(lat0))
    LON, LAT = np.meshgrid(lon, lat)
    t2 = np.full((n, n), 300.0)
    psfc = np.full((n, n), 95000.0)
    if dryline_lon is not None:
        td = 273.15 + 7.5 + 12.5 * np.tanh((LON - dryline_lon) * 90.0 / 20.0)
    else:
        td = np.full((n, n), 285.0)
    es = 611.2 * np.exp(17.67 * (td - 273.15) / (td - 29.65))
    r = 0.622 * es / (psfc - es)
    if low is not None:
        x, y = st._xy_km(LAT, LON, *low)
        psfc = psfc - 1200.0 * np.exp(-(np.hypot(x, y) / 300.0) ** 2)
    with netCDF4.Dataset(path, "w") as d:
        d.DX = dx_m
        d.createDimension("Time", 1)
        d.createDimension("south_north", n)
        d.createDimension("west_east", n)
        for name, value in (("XLAT", LAT), ("XLONG", LON), ("T2", t2), ("Q2", r / (1 + r)), ("PSFC", psfc),
                            ("HGT", np.zeros((n, n)))):
            d.createVariable(name, "f4", ("Time", "south_north", "west_east"))[:] = value[None]
    return path


def test_boundary_target_follows_a_dryline_east(tmp_path):
    start = datetime(2026, 11, 7, 18)
    frames, times = [], []
    for k in range(4):
        when = start + timedelta(hours=k)
        frames.append(_frame(tmp_path / f"wrfout_d01_{k}", when, dryline_lon=-100.0 + 0.25 * k))
        times.append(when)
    config = st.TargetConfig.for_target("boundary", segment=[[34.5, -100.0], [37.5, -100.0]], fix_blend=1.0)
    track, fixes = st.follow_track(frames, times, config, (36.0, -100.0))
    assert all(f["ok"] for f in fixes), fixes
    for k, (when, lat, lon) in enumerate(track):
        assert abs(lon - (-100.0 + 0.25 * k)) < 0.12              # on the line (about 10 km)
    assert all(min(abs(f["axis_deg"]), 180 - abs(f["axis_deg"])) < 15 for f in fixes)   # north-south, as drawn


def test_low_target_finds_the_pressure_minimum(tmp_path):
    when = datetime(2026, 11, 7, 18)
    frame = _frame(tmp_path / "wrfout_d01_0", when, low=(36.8, -98.2), dx_m=15000.0)
    config = st.TargetConfig.for_target("low", fix_blend=1.0)
    track, fixes = st.follow_track([frame], [when], config, (36.0, -99.0))
    assert fixes[0]["ok"], fixes
    x, y = st._xy_km(np.array(track[0][1]), np.array(track[0][2]), 36.8, -98.2)
    assert math.hypot(float(x), float(y)) < 40.0


def test_path_target_is_the_users_path(tmp_path):
    path = tmp_path / "path.json"
    path.write_text(json.dumps({"rows": [{"time": "2026-11-07T18:00:00Z", "lat": 35.0, "lon": -100.0},
                                         {"time": "2026-11-08T00:00:00Z", "lat": 37.0, "lon": -96.0}]}))
    config = st.TargetConfig.for_target("path")
    config.path = st.read_path(path)
    track, fixes = st.follow_track([], [], config, None)
    assert track[0][1:] == (35.0, -100.0) and "not a tracker" in fixes[0]["reason"]


def test_cape_shear_columns_are_plausible():
    nlev = 40
    psfc = np.array([97000.0, 97000.0])
    p = psfc[:, None] * np.linspace(1.0, 0.12, nlev)[None, :]
    z = 7500.0 * np.log(psfc[:, None] / p)
    t = np.maximum(303.0 - 0.0075 * z, 205.0)
    theta = t * (1.0e5 / p) ** (st.RD / st.CP)
    qv = np.stack([0.016 * np.exp(-z[0] / 2500.0), 0.003 * np.exp(-z[1] / 2500.0)])
    u = 5.0 * z / 1000.0
    cape, shear = st.mlcape_and_shear(p, theta, qv, u, np.zeros_like(u), psfc)
    assert 1000.0 < cape[0] < 6000.0 and cape[1] < 0.2 * cape[0]
    assert shear == pytest.approx([30.0, 30.0], abs=4.0)


def test_unknown_target_is_refused():
    with pytest.raises(st.SetupTargetRefusal, match="must be one of"):
        st.TargetConfig.for_target("hail")


def _index_frame(storms_at=None, *, far_peak=None, n=161, spacing=15.0):
    """A cape-shear frame as fix_frame sees it: a broad index area west of the prediction, optional storms."""
    lat0, lon0 = 30.0, -100.0
    axis = (np.arange(n) - n // 2) * spacing
    X, Y = np.meshgrid(axis, axis)
    lat, lon = np.vectorize(lambda x, y: st._latlon_from_xy(lat0, lon0, x, y))(X, Y)
    value = 1.5 * np.exp(-(np.hypot(X + 250.0, Y) / 500.0) ** 2)
    if far_peak is not None:
        value = value + 4.0 * np.exp(-(np.hypot(X - far_peak[0], Y - far_peak[1]) / 120.0) ** 2)
    frame = dict(lat=lat, lon=lon, value=value, spacing_km=spacing)
    if storms_at is not None:
        frame["storms"] = np.where(np.hypot(X - storms_at[0], Y - storms_at[1]) < 100.0, 50.0, 0.0)
    return frame, (lat0, lon0)


def test_cape_shear_keeps_its_storms_and_moves_like_weather():
    """Carried from gpuwm-hex's moving region (GFS 2026-10-23 18Z f354, box L): the index core slid off the storms
    that used up its CAPE, and after dark the fixes ran 100-150 km per 30 min toward a remote ocean plume."""
    config = st.TargetConfig.for_target("cape-shear", fix_blend=1.0)
    quiet, guess = _index_frame()
    stormy, _ = _index_frame(storms_at=(150.0, 0.0))
    a = st.fix_frame(quiet, config, guess, 700.0)
    b = st.fix_frame(stormy, config, guess, 700.0)
    assert a["ok"] and b["ok"] and a["storm_share"] == 0.0 and b["storm_share"] > 0.0
    xa, _ = st._xy_km(np.array(a["lat"]), np.array(a["lon"]), *guess)
    xb, _ = st._xy_km(np.array(b["lat"]), np.array(b["lon"]), *guess)
    assert float(xb) > float(xa) + 100.0                        # pulled east, toward its storms
    far, _ = _index_frame(far_peak=(-600.0, -150.0))
    unbounded = st.TargetConfig.for_target("cape-shear")
    unbounded.core_reach_km = unbounded.max_step_kmh = None        # the old behaviour (for_target drops None overrides)
    free = st.fix_frame(far, unbounded, guess, 700.0, hours=0.5)
    held = st.fix_frame(far, st.TargetConfig.for_target("cape-shear"), guess, 700.0, hours=0.5)
    d = lambda f: math.hypot(*map(float, st._xy_km(np.array(f["lat"]), np.array(f["lon"]), *guess)))
    assert d(free) > 150.0 and d(held) <= 45.0 + 1.0 and held["step_capped"]
