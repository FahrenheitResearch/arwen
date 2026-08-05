"""The inflow-fetch meter, proved against a field whose answer is known.

A meter that reports a spin-up distance has to be shown to read the
distance and not the noise.  These build synthetic parent/child frames in
which the child's turbulence amplitude ramps from zero at one face to a
plateau at a distance chosen by the test, and require the meter to recover
that distance -- and, in the negative control, to report NOT REACHED when
no plateau exists inside the domain.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

nc = pytest.importorskip("netCDF4")

TOOL = Path(__file__).resolve().parents[1] / "tools" / "inflow_fetch_meter.py"

NZ, NY, NX = 8, 192, 192
RATIO = 3
FOOT = NX // RATIO
IPS, JPS = 5, 7
PBLH = 1000.0


def _write(path: Path, nz, ny, nx, w, u, v, pblh, hgt=0.0, top=4000.0):
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("bottom_top", nz)
        ds.createDimension("bottom_top_stag", nz + 1)
        ds.createDimension("south_north", ny)
        ds.createDimension("south_north_stag", ny + 1)
        ds.createDimension("west_east", nx)
        ds.createDimension("west_east_stag", nx + 1)
        var = ds.createVariable(
            "W", "f4", ("Time", "bottom_top_stag", "south_north", "west_east"))
        var[0] = w
        var = ds.createVariable(
            "U", "f4", ("Time", "bottom_top", "south_north", "west_east_stag"))
        var[0] = np.broadcast_to(np.float32(u), (nz, ny, nx + 1))
        var = ds.createVariable(
            "V", "f4", ("Time", "bottom_top", "south_north_stag", "west_east"))
        var[0] = np.broadcast_to(np.float32(v), (nz, ny + 1, nx))
        zstag = np.linspace(0.0, top, nz + 1)
        ph = np.broadcast_to(
            (zstag * 9.81)[:, None, None], (nz + 1, ny, nx)).astype("f4")
        ds.createVariable(
            "PH", "f4", ("Time", "bottom_top_stag", "south_north", "west_east")
        )[0] = ph
        ds.createVariable(
            "PHB", "f4", ("Time", "bottom_top_stag", "south_north", "west_east")
        )[0] = np.zeros_like(ph)
        ds.createVariable(
            "HGT", "f4", ("Time", "south_north", "west_east")
        )[0] = np.full((ny, nx), hgt, dtype="f4")
        ds.createVariable(
            "PBLH", "f4", ("Time", "south_north", "west_east")
        )[0] = np.full((ny, nx), pblh, dtype="f4")


def _child_w(rng, ramp_cells, ny=NY, nx=NX):
    """Turbulence whose amplitude ramps with distance from the WEST face."""
    d = np.arange(nx)[None, :]
    amp = np.clip(d / float(ramp_cells), 0.0, 1.0)
    # a two-cell-scale random field so the child band carries real energy
    field = rng.standard_normal((ny, nx))
    field = field + np.roll(field, 1, axis=1) + np.roll(field, 1, axis=0)
    w = np.zeros((NZ + 1, ny, nx), dtype="f4")
    for k in range(NZ + 1):
        w[k] = (amp * field).astype("f4")
    return w


def _run(tmp_path, ramp_cells, stamp="2000-01-01_00_00_00", seed=0):
    rng = np.random.default_rng(seed)
    out = tmp_path / "wrfout"
    out.mkdir(exist_ok=True)
    _write(out / f"wrfout_d03_{stamp}", NZ, NY, NX,
           _child_w(rng, ramp_cells), 8.0, 0.0, 0.0)
    pw = np.zeros((NZ + 1, FOOT + JPS + 4, FOOT + IPS + 4), dtype="f4")
    pw[:] = 0.1 * rng.standard_normal(pw.shape[1:])[None]
    _write(out / f"wrfout_d02_{stamp}", NZ,
           FOOT + JPS + 4, FOOT + IPS + 4, pw, 8.0, 0.0, PBLH)
    js = tmp_path / "meter.json"
    subprocess.run(
        [sys.executable, str(TOOL), str(out), str(js),
         "--i-parent-start", str(IPS), "--j-parent-start", str(JPS),
         "--parent-grid-ratio", str(RATIO),
         "--child-dx-m", "250", "--parent-dx-m", "750"],
        check=True, capture_output=True)
    import json
    return json.loads(js.read_text())


def test_west_face_is_the_inflow_face_and_d90_recovers_the_ramp(tmp_path):
    ramp = 40                                   # cells -> 10 km
    got = _run(tmp_path, ramp)
    frame = got["frames"][0]
    assert frame["inflow_faces"] == ["west"], frame["inflow_faces"]
    res = frame["faces"]["west"]["d90_var_w"]
    assert res["reached"], res
    # var(w) ~ amp**2, so the 90 %-of-plateau crossing sits at
    # sqrt(0.9) = 0.949 of the ramp: 9.49 km, +- the smoothing width.
    assert 8.0 <= res["d90_km"] <= 11.5, res


def test_a_ramp_longer_than_the_domain_reports_not_reached(tmp_path):
    got = _run(tmp_path, ramp_cells=400, seed=3)
    res = got["frames"][0]["faces"]["west"]["d90_var_w"]
    assert not res["reached"], res
    assert res["lower_bound_km"] > 10.0


def test_no_ramp_at_all_puts_d90_at_the_boundary_zone(tmp_path):
    got = _run(tmp_path, ramp_cells=1, seed=5)
    res = got["frames"][0]["faces"]["west"]["d90_var_w"]
    assert res["reached"], res
    assert res["d90_km"] <= 2.5, res


def test_the_fetch_zone_lowers_a_whole_domain_variance_ratio(tmp_path):
    got = _run(tmp_path, ramp_cells=40, seed=11)
    block = got["frames"][0]["whole_domain_vs_interior"]
    assert 0.0 < block["fetch_zone_area_fraction"] < 0.5
    assert block["child_var_w_interior"] > block["child_var_w_whole_domain"]
    assert block["var_w_ratio_interior"] > block["var_w_ratio_whole_domain"]
