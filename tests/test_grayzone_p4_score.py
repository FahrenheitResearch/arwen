"""Interface-instrument plumbing tests (P4, tools/grayzone_p4_score.py).

The tool is measurement only -- geometry, the transplanted spectral
instrument, and the ladder's partition reduction wired onto wrfout
fields.  These tests pin each piece against an independent computation
so a wiring slip cannot masquerade as a physics finding.  All CPU.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

nc = pytest.importorskip("netCDF4")

REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "grayzone_p4_score", REPO / "tools" / "grayzone_p4_score.py")
score_mod = importlib.util.module_from_spec(_SPEC)
sys.modules["grayzone_p4_score"] = score_mod
_SPEC.loader.exec_module(score_mod)

from gpuwm.verify import gray_zone  # noqa: E402
from gpuwm.verify.cases.cbl_dry import (  # noqa: E402
    MIXED_LAYER_WINDOW, window_mean)
from gpuwm.verify.sase_ref import bulk_richardson_zi  # noqa: E402

NZ, PNY, PNX = 8, 12, 12
RATIO = 3
CNY = CNX = 18                       # child; footprint 6x6 parent cells
IPS = JPS = 4                        # 1-based parent start
Z_FACE = np.linspace(0.0, 1600.0, NZ + 1)


def test_region_masks_geometry():
    far, rim = score_mod.region_masks(
        12, 12, exclude_west=5, exclude_south=3, rim_offset=1,
        rim_width=1)
    # rim = the distance-1 ring: just inside the forced edge cells,
    # which belong to neither band
    assert not rim[0].any() and not rim[:, 0].any()
    assert rim[1, 1:-1].all() and rim[1:-1, 1].all()
    assert rim[-2, 1:-1].all() and rim[1:-1, -2].all()
    assert not rim[2:-2, 2:-2].any()
    # far field: fetch strips and the whole edge band excluded
    assert not far[:, :5].any()          # west strip
    assert not far[:3, :].any()          # south strip
    assert not far[rim].any()            # edge band out on every side
    assert not far[0].any() and not far[:, -1].any()
    assert far[5, 5] and far[3, 5] and not far[3, 4]
    assert far[3:-2, 5:-2].all()


def test_spectral_overlay_is_zero_against_itself_and_pairs_bins():
    rng = np.random.default_rng(7)
    child = rng.standard_normal((24, 24))
    stat, bins = score_mod.spectral_overlay_stat(child, 1.0, child, 1.0,
                                                 converge_above_km=8.0)
    assert stat == 0.0 and bins == 3     # lam = 24, 12, 8 km
    # A field whose energy lives at large scales survives a 3x block
    # average, so the shared bins must be close -- this is the physical
    # situation the convergence screen watches (the child inheriting its
    # large scales through the boundary).  Pure white noise would not
    # satisfy this, and should not.
    yy, xx = np.mgrid[0:24, 0:24] * (2.0 * np.pi / 24.0)
    organised = (np.sin(xx) * np.cos(yy) + 0.5 * np.sin(2.0 * xx)
                 + 0.02 * child)
    parent = organised.reshape(8, 3, 8, 3).mean(axis=(1, 3))
    stat, bins = score_mod.spectral_overlay_stat(
        organised, 1.0, parent, 3.0, converge_above_km=8.0)
    assert bins == 3 and np.isfinite(stat)
    assert stat < 0.1


def test_radial_spectrum_parseval_closure_is_exact():
    rng = np.random.default_rng(11)
    f = rng.standard_normal((30, 30))
    _, power = score_mod._radial_spectrum(f, 0.25)
    f2 = f[:30, :30] - f.mean()
    assert np.isclose(power.sum(), (f2 ** 2).mean(), rtol=1e-12)


def _write_wrfout(path, nz, ny, nx, w, u, v, theta, e_sgs, pblh,
                  tke_name="TKE_TEST"):
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("Time", 1)
        ds.createDimension("bottom_top", nz)
        ds.createDimension("bottom_top_stag", nz + 1)
        ds.createDimension("south_north", ny)
        ds.createDimension("south_north_stag", ny + 1)
        ds.createDimension("west_east", nx)
        ds.createDimension("west_east_stag", nx + 1)

        def var(name, dims, data):
            v_ = ds.createVariable(name, "f8", dims)
            v_[0] = data

        var("W", ("Time", "bottom_top_stag", "south_north", "west_east"),
            w)
        var("U", ("Time", "bottom_top", "south_north", "west_east_stag"),
            u)
        var("V", ("Time", "bottom_top", "south_north_stag", "west_east"),
            v)
        var("T", ("Time", "bottom_top", "south_north", "west_east"),
            theta - 300.0)
        var("PH", ("Time", "bottom_top_stag", "south_north", "west_east"),
            np.zeros((nz + 1, ny, nx)))
        var("PHB", ("Time", "bottom_top_stag", "south_north", "west_east"),
            9.81 * Z_FACE[:, None, None] * np.ones((1, ny, nx)))
        var("HGT", ("Time", "south_north", "west_east"),
            np.zeros((ny, nx)))
        var("PBLH", ("Time", "south_north", "west_east"),
            np.full((ny, nx), pblh))
        if e_sgs is not None:
            var(tke_name,
                ("Time", "bottom_top", "south_north", "west_east"), e_sgs)


def _synthetic_pair(tmp_path, stamp, with_tke):
    rng = np.random.default_rng(20260804)
    zm = 0.5 * (Z_FACE[:-1] + Z_FACE[1:])
    # Parent: convectively shaped theta so bulk-Richardson h is interior.
    theta_prof = np.where(zm < 900.0, 300.0, 300.0 + 0.01 * (zm - 900.0))
    theta = theta_prof[:, None, None] * np.ones((NZ, PNY, PNX))
    u = rng.standard_normal((NZ, PNY, PNX + 1))
    v = rng.standard_normal((NZ, PNY + 1, PNX))
    w = rng.standard_normal((NZ + 1, PNY, PNX))
    e_sgs = np.full((NZ, PNY, PNX), 0.8) if with_tke else None
    _write_wrfout(tmp_path / f"wrfout_d02_{stamp}", NZ, PNY, PNX,
                  w, u, v, theta, e_sgs, pblh=1000.0)
    cw = rng.standard_normal((NZ + 1, CNY, CNX))
    cu = rng.standard_normal((NZ, CNY, CNX + 1))
    cv = rng.standard_normal((NZ, CNY + 1, CNX))
    ctheta = theta_prof[:, None, None] * np.ones((NZ, CNY, CNX))
    _write_wrfout(tmp_path / f"wrfout_d03_{stamp}", NZ, CNY, CNX,
                  cw, cu, cv, ctheta, None, pblh=0.0)
    return w, u, v, theta, e_sgs


def _run_tool(tmp_path, extra=()):
    out = tmp_path / "receipt.json"
    argv = ["--wrfout", str(tmp_path),
            "--i-parent-start", str(IPS), "--j-parent-start", str(JPS),
            "--ratio", str(RATIO),
            "--parent-dx-km", "0.75", "--child-dx-km", "0.25",
            "--exclude-west-cells", "6", "--exclude-south-cells", "3",
            "--rim-offset-cells", "1", "--rim-width-cells", "2",
            "--converge-above-km", "2.0",
            "--out", str(out), *extra]
    assert score_mod.main(argv) == 0
    return json.loads(out.read_text())


def test_partition_block_matches_the_ladder_reduction(tmp_path):
    stamp = "2026-08-01_20_00_00"
    w, u, v, theta, e_sgs = _synthetic_pair(tmp_path, stamp,
                                            with_tke=True)
    receipt = _run_tool(tmp_path,
                        ("--tke-var", "TKE_TEST",
                         "--band-sigma", "0.001929"))
    part = receipt["frames"][stamp]["partition"]

    # Independent reduction, straight from the committed instrument.
    j0, i0 = JPS - 1, IPS - 1
    foot = CNY // RATIO
    sl = (slice(None), slice(j0, j0 + foot), slice(i0, i0 + foot))
    u_m = (0.5 * (u[:, :, :-1] + u[:, :, 1:]))[sl]
    v_m = (0.5 * (v[:, :-1, :] + v[:, 1:, :]))[sl]
    w_m = (0.5 * (w[:-1] + w[1:]))[sl]
    th = theta[sl]
    e = e_sgs[sl]
    zm = 0.5 * (Z_FACE[:-1] + Z_FACE[1:])
    z3 = zm[:, None, None] * np.ones_like(th)
    ref = gray_zone.partition_from_profiles(e, u_m, v_m, w_m)
    h = float(np.mean(bulk_richardson_zi(u_m, v_m, th, z3)))
    frac_ml = window_mean(ref["subgrid_fraction"], zm, h,
                          MIXED_LAYER_WINDOW)
    assert part["h_bulk_richardson_m"] == pytest.approx(h)
    assert part["mixed_layer_subgrid_fraction"] == pytest.approx(frac_ml)
    assert part["x_own"] == pytest.approx(750.0 / h)
    lo, hi = gray_zone.subgrid_tke_envelope(part["x_own"])
    half = 2.0 * 0.001929 / np.sqrt(6.0)
    assert part["band"][0] == pytest.approx(float(lo) - half)
    assert part["band"][1] == pytest.approx(float(hi) + half)
    assert part["in_band"] == (
        part["band"][0] <= part["mixed_layer_subgrid_fraction"]
        <= part["band"][1])


def test_parent_without_tke_records_partition_absent(tmp_path):
    stamp = "2026-08-01_20_00_00"
    _synthetic_pair(tmp_path, stamp, with_tke=False)
    receipt = _run_tool(tmp_path, ("--tke-var", "TKE_TEST"))
    part = receipt["frames"][stamp]["partition"]
    assert part["absent"] is True and "no subgrid TKE" in part["reason"]


def test_frame_selection_and_geometry_echo(tmp_path):
    for stamp in ("2026-08-01_20_00_00", "2026-08-01_21_00_00"):
        _synthetic_pair(tmp_path, stamp, with_tke=False)
    receipt = _run_tool(
        tmp_path, ("--frames", "2026-08-01_21_00_00"))
    assert list(receipt["frames"]) == ["2026-08-01_21_00_00"]
    assert receipt["arguments"]["exclude_west_cells"] == 6
    row = receipt["frames"]["2026-08-01_21_00_00"]
    # the mid-CBL rule: PBLH 1000 m -> nearest level to 500 m
    zm = 0.5 * (Z_FACE[:-1] + Z_FACE[1:])
    assert row["level_index"] == int(np.argmin(np.abs(zm - 500.0)))
    assert row["far_field_w_var_child"] > 0.0
    assert np.isfinite(row["rim_over_far_field"])
