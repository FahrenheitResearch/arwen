"""Canonical synthetic fields exercise every source through one seed path."""
from pathlib import Path
import argparse
import json

import numpy as np
import pytest

from gpuwm import cyclone_seed as seed
from gpuwm import cyclone_sources as cs
from gpuwm import cyclone_setup as tc
from gpuwm.source_adapters import get_source_adapter
from gpuwm.source_coverage import window_centre


def fields_at(point=(20., -65.), *, dateline=False, reverse_levels=False):
    lat0, lon0 = point
    d = np.linspace(-2., 2., 41)
    lon, lat = np.meshgrid(lon0 + d / np.cos(np.deg2rad(lat0)), lat0 + d)
    if dateline:
        lon = (lon + 180.) % 360. - 180.
    x = seed.EARTH_RADIUS_M * np.cos(np.deg2rad(lat0)) * np.deg2rad((lon - lon0 + 180.) % 360. - 180.)
    y = seed.EARTH_RADIUS_M * np.deg2rad(lat - lat0)
    g = np.exp(-(x*x + y*y) / (90000. ** 2))
    sign = -1. if lat0 < 0 else 1.
    u = -sign * 0.0003 * y * g
    v = sign * 0.0003 * x * g
    p = np.array([100000., 85000., 50000., 30000.])
    fields = {"latitude": lat, "longitude": lon,
              "mean_sea_level_pressure": 101000. - 6000. * g,
              "pressure_levels_pa": p,
              "eastward_wind": np.broadcast_to(u, (4, *lat.shape)).copy(),
              "northward_wind": np.broadcast_to(v, (4, *lat.shape)).copy(),
              "air_temperature": np.stack([295 + 2*g, 285 + 3*g, 260 + 6*g, 240 + 6*g])}
    if reverse_levels:
        for key in ("pressure_levels_pa", "eastward_wind", "northward_wind", "air_temperature"):
            fields[key] = fields[key][::-1]
    return fields


@pytest.mark.parametrize("source", cs.source_ids())
def test_each_source_seeds_through_its_inventory(source):
    window = get_source_adapter(source).coverage_window
    point = window_centre(window) if window is not None else (20., -65.)
    result = seed.seed_cyclone(source=source, fields=fields_at(point))
    assert result.point is not None, result.messages
    assert result.point == pytest.approx(point, abs=0.15)
    assert result.method in {"mslp", "low_level_vorticity"}
    assert result.source == source


@pytest.mark.parametrize("point", [(20., -65.), (-20., 65.), (20., 179.9), (-20., -179.9)])
@pytest.mark.parametrize("reverse", [False, True])
def test_vorticity_sign_dateline_and_vertical_order(point, reverse):
    fields = fields_at(point, dateline=True, reverse_levels=reverse)
    fields.pop("mean_sea_level_pressure")
    result = seed.seed_cyclone(source="gfs", fields=fields)
    assert result.method == "low_level_vorticity"
    assert result.point[0] == pytest.approx(point[0], abs=0.15)
    assert abs((result.point[1] - point[1] + 180) % 360 - 180) < 0.15
    assert any("MSLP" in note for note in result.messages)


def test_warm_core_is_fallback_not_a_required_field():
    fields = fields_at()
    for name in ("mean_sea_level_pressure", "eastward_wind", "northward_wind"):
        fields.pop(name)
    result = seed.seed_cyclone(source="aigfs", fields=fields)
    assert result.method == "warm_core"
    assert result.point == pytest.approx((20., -65.))


def test_no_extrapolation_to_missing_pressure_surfaces():
    fields = fields_at()
    fields["pressure_levels_pa"] = np.array([70000., 60000., 50000., 40000.])
    fields.pop("mean_sea_level_pressure")
    result = seed.seed_cyclone(source="gfs", fields=fields, advisory=(20.1, -65.1))
    assert result.method == "advisory"
    assert result.point == (20.1, -65.1)


def test_missing_fields_fall_back_to_advisory_and_explicit_point_wins():
    assert seed.seed_cyclone(source="aifs", advisory=(20., -65.)).method == "advisory"
    result = seed.seed_cyclone(source="aifs", fields=fields_at(), point=(21., -66.))
    assert result.method == "point" and result.point == (21., -66.)
    empty = seed.seed_cyclone(source="aifs")
    assert empty.point is None and "--point" in empty.messages[-1]


def test_all_missing_or_uniform_fields_do_not_choose_first_cell():
    fields = fields_at()
    fields["mean_sea_level_pressure"][:] = 101000.
    for key in ("eastward_wind", "northward_wind", "air_temperature"):
        fields[key][:] = np.nan
    result = seed.seed_cyclone(source="gfs", fields=fields)
    assert result.point is None


def test_advisory_radius_excludes_distant_lower_pressure():
    fields = fields_at()
    fields["mean_sea_level_pressure"][0, 0] = 70000.
    result = seed.seed_cyclone(source="gfs", fields=fields, advisory=(20., -65.), search_radius_km=100.)
    assert result.point == pytest.approx((20., -65.))


def test_log_pressure_interpolation_and_below_ground_columns():
    pressure = np.array([95000., 70000., 50000.])[:, None, None] * np.ones((3, 3, 3))
    field = 2. * np.log(pressure)
    pressure[:, 0, 0] = np.array([70000., 60000., 50000.])
    inventory = seed.SeedInventory(("air_pressure", "air_temperature"))
    plane = seed._pressure_plane({"air_pressure": pressure, "air_temperature": field},
                                 inventory, "air_temperature", 85000., (3, 3))
    assert np.isnan(plane[0, 0])
    assert plane[1, 1] == pytest.approx(2 * np.log(85000.))


def test_spherical_curl_retains_curvature_term():
    lat, lon = np.meshgrid(np.linspace(25., 35., 31), np.linspace(10., 20., 31), indexing="ij")
    result = seed.relative_vorticity(np.full(lat.shape, 10.), np.zeros(lat.shape), lat, lon)
    assert result == pytest.approx(10. * np.tan(np.deg2rad(lat)) / seed.EARTH_RADIUS_M)


@pytest.mark.parametrize("projection", ["lambert", "mercator", "polar"])
def test_projected_grid_uses_earth_coordinates(projection):
    from gpuwm import domain_wizard as dw
    point = (65., 20.) if projection == "polar" else (20., -65.)
    entries = dw._projection_entries(*point, projection)
    grid = dw._root_grid(entries, 41, 41, 12000.)
    lat, lon = grid.latlon_c()
    # Define an analytic vortex in local tangent coordinates, not array axes.
    lat0, lon0 = float(lat[20, 20]), float(lon[20, 20])
    x = seed.EARTH_RADIUS_M * np.cos(np.deg2rad(lat0)) * np.deg2rad((lon-lon0+180)%360-180)
    y = seed.EARTH_RADIUS_M * np.deg2rad(lat-lat0)
    g = np.exp(-(x*x+y*y)/90000.**2)
    fields = {"latitude": lat, "longitude": lon, "pressure_levels_pa": np.array([85000.]),
              "eastward_wind": (-0.0003*y*g)[None, ...],
              "northward_wind": (0.0003*x*g)[None, ...]}
    result = seed.seed_cyclone(source="gfs", fields=fields)
    assert result.method == "low_level_vorticity"
    assert seed._distance_km(*result.point, (lat0, lon0)) < 15.


def test_npz_source_cycle_and_member_are_bound(tmp_path):
    path = tmp_path / "seed.npz"
    np.savez(path, **fields_at(), source="gefs", cycle="2026090900", member="p03")
    assert "latitude" in seed.load_seed_fields(path, source="gefs", cycle="2026090900", member="p03")
    for change in ({"source": "aigefs", "member": "mem003"}, {"cycle": "2026090906"}, {"member": "p04"}):
        kw = {"source": "gefs", "cycle": "2026090900", "member": "p03"} | change
        with pytest.raises(ValueError, match="does not match"):
            seed.load_seed_fields(path, **kw)


def test_cli_field_seed_reaches_configuration(tmp_path, monkeypatch, capsys):
    from gpuwm import domain_wizard as dw
    from types import SimpleNamespace
    path = tmp_path / "seed.npz"
    np.savez(path, **fields_at(), source="aifs", cycle="2026090900", member="")
    parser = argparse.ArgumentParser()
    tc.register_cli(parser.add_subparsers())
    args = parser.parse_args(["cyclone-setup", "--source", "aifs", "--cycle", "2026090900",
                             "--seed-fields", str(path), "--vram-gib", "32"])
    budget = dw.SizingBudget(32., 30*dw.GIB, None, "fixture")
    monkeypatch.setattr(dw, "_domain_target_hardware", lambda args: (budget, None, False))
    monkeypatch.setattr(dw, "_sizing_phases", lambda *a, **kw: SimpleNamespace(
        peak_envelope_bytes=100, binding_phase="forecast"))
    monkeypatch.setattr(dw, "sizing_budget_bytes", lambda *a, **kw: 200)
    assert tc.main(args) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["seed"]["method"] == "low_level_vorticity"
    assert doc["point"] == pytest.approx((20., -65.), abs=0.15)
