"""Conventional observations as table rows, point batches and IAU.

* A point batch is the dense batch it describes: the host filter gives
  the same bytes for both, and the innovation summary the same numbers.
* The pressure-level operator is exact on a column where temperature is
  linear in log pressure, bilinear in the horizontal, and refuses to
  extrapolate; reports landing on one gridpoint become one superob; the
  background check drops an outlier and counts it.
* The surface operator refuses a station whose elevation the grid does
  not resolve.
* IAU's fractions sum to the whole increment over its window, whatever
  the step, and nothing is added outside it.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.da.letkf import (GriddedObs, GridGeometry, LetkfConfig,
                            LetkfDiagnostics, Localization, PointSet,
                            analyze, point_batch)
from gpuwm.da import obs_conventional as oc
from gpuwm.da.radar_assimilation import innovation_summary

HEADER = ("source,station_id,latitude_deg,longitude_deg,elevation_m,level_pa,"
          "valid_time,variable,value,error,measurement,nominal_time,"
          "published_time,received_time,revision")


def _dense_and_points(rng, members=8, shape=(4, 6, 7), n=9):
    size = int(np.prod(shape))
    flat = np.sort(rng.choice(size, n, replace=False))
    values = rng.normal(size=n)
    errors = rng.uniform(0.5, 1.5, size=n)
    sim = rng.normal(size=(members, n))
    points = PointSet(flat_index=flat, values=values, errors=errors,
                      simulated=sim)
    batch = point_batch("p", shape, points, localization=Localization(4000., 1500.))
    dense_sim = np.zeros((members,) + shape)
    dense_sim.reshape(members, -1)[:, flat] = sim
    dense = GriddedObs("p", batch.values, batch.errors, dense_sim, batch.mask,
                       localization=Localization(4000., 1500.))
    return batch, dense


def test_point_batch_is_its_dense_batch_on_the_host_filter():
    rng = np.random.default_rng(3)
    members, shape = 8, (4, 6, 7)
    prior = {"a": rng.normal(size=(members,) + shape),
             "b": rng.normal(size=(members,) + shape) * 2}
    batch, dense = _dense_and_points(rng, members, shape)
    grid = GridGeometry(dx_m=1000., dy_m=1000.,
                        heights_m=np.array([50., 300., 900., 2000.]))
    cfg = LetkfConfig(Localization(3000., 1000.), ("a", "b"), 0.9)
    one = analyze(prior, [batch], grid, cfg, LetkfDiagnostics())
    two = analyze(prior, [dense], grid, cfg, LetkfDiagnostics())
    for name in prior:
        assert np.array_equal(one[name], two[name])
    (a,), (b,) = innovation_summary([batch]), innovation_summary([dense])
    assert a.keys() == b.keys()
    for key in a:
        assert a[key] == (pytest.approx(b[key], rel=1e-12)
                          if isinstance(a[key], float) else b[key])
    assert batch.simulated.shape == dense.simulated.shape


def test_point_batch_refuses_unsorted_or_outside_indices():
    with pytest.raises(ValueError, match="strictly increasing"):
        point_batch("x", (2, 2, 2), PointSet(np.array([3, 1]), np.zeros(2),
                                             np.ones(2), np.zeros((2, 2))))
    with pytest.raises(ValueError, match="strictly increasing"):
        point_batch("x", (2, 2, 2), PointSet(np.array([9]), np.zeros(1),
                                             np.ones(1), np.zeros((2, 1))))


class _Grid:
    """Duck-typed TargetGrid: mass index = (lon, lat) in degrees."""

    def __init__(self, nz=5, ny=6, nx=7, terrain=0.0):
        self.nz, self.ny, self.nx = nz, ny, nx
        self.terrain_m = np.full((ny, nx), terrain)

    def mass_index(self, lat, lon):
        return np.asarray(lon, np.float64), np.asarray(lat, np.float64)


def _states(members=4, nz=5, ny=6, nx=7, seed=0):
    rng = np.random.default_rng(seed)
    p_levels = np.array([95000., 85000., 70000., 50000., 30000.])
    states = []
    for m in range(members):
        p = np.broadcast_to(p_levels[:, None, None], (nz, ny, nx)).copy()
        theta_target = 290.0 + 2.0 * m  # temperature linear in log p below
        # T = a + b ln p  ->  thp = T / (p/P0)^k - thb
        t = (250.0 + m) + 10.0 * np.log(p / 50000.0)
        thb = np.full(nz, 300.0)
        thp = t / (p / 1e5) ** (287.0 / 1004.5) - thb[:, None, None]
        u = np.zeros((nz, ny, nx + 1)) + 5.0 + m
        v = np.zeros((nz, ny + 1, nx)) - 2.0
        qv = np.full((nz, ny, nx), 0.005) + rng.uniform(0, 1e-4, (nz, ny, nx))
        states.append({"p": p, "thp": thp, "qv": qv, "u": u, "v": v})
        del theta_target
    return states, np.full(nz, 300.0)


def _table(tmp_path, rows):
    path = tmp_path / "t.csv"
    path.write_text(HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return oc.read_tables([path])


def _row(source, sid, lat, lon, elev, level, when, var, value, error, meas):
    return (f"{source},{sid},{lat},{lon},{elev},{level},{when},{var},{value},"
            f"{error},{meas},,,,abc")


T0 = datetime(2026, 10, 1, 19, tzinfo=timezone.utc)


def test_pressure_level_operator_superob_and_background_check(tmp_path):
    states, thb = _states()
    grid = _Grid()
    when = "2026-10-01T18:50:00Z"
    rows = _table(tmp_path, [
        # two reports on one gridpoint at 60 kPa: one superob
        _row("madis-acars", "A1", 2.2, 3.1, 4000, 60000, when,
             "temperature_k", 258.0, 1.0, "aircraft_level"),
        _row("madis-acars", "A2", 2.0, 3.0, 4000, 60000, when,
             "temperature_k", 259.0, 1.0, "aircraft_level"),
        # an outlier: background check
        _row("madis-acars", "A3", 4.0, 5.0, 4000, 60000, when,
             "temperature_k", 330.0, 1.0, "aircraft_level"),
        # above the column top: no extrapolation
        _row("madis-acars", "A4", 1.0, 1.0, 12000, 20000, when,
             "temperature_k", 220.0, 1.0, "aircraft_level"),
        # outside the window
        _row("madis-acars", "A5", 1.0, 1.0, 4000, 60000,
             "2026-10-01T18:20:00Z", "temperature_k", 258.0, 1.0,
             "aircraft_level"),
        # wind
        _row("madis-acars", "A6", 1.0, 2.0, 4000, 60000, when,
             "wind_u_m_s", 7.0, 2.0, "aircraft_level"),
    ])
    table = oc.read_type_table()
    batches, prov = oc.conventional_batches(
        rows, table, states=states, thb=thb, grid=grid, analysis_time=T0)
    names = {b.name: b for b in batches}
    t = names["conventional:upper-temperature"]
    assert t.points.flat_index.size == 1
    # H(x) exact: T linear in ln p
    expect = np.array([(250.0 + m) + 10.0 * np.log(60000 / 50000.0)
                       for m in range(4)])
    np.testing.assert_allclose(t.points.simulated[:, 0], expect, atol=1e-9)
    assert t.points.values[0] == pytest.approx(258.5)
    counts = prov["types"]["aircraft-temperature"]
    assert counts["background_check"] == 1
    assert counts["outside_vertical_column"] == 1
    assert prov["totals"]["rows_outside_window"] == 1
    u = names["conventional:upper-wind-u"]
    np.testing.assert_allclose(u.points.simulated[:, 0],
                               [5.0, 6.0, 7.0, 8.0], atol=1e-12)
    # level placement: 60 kPa is nearest 70 kPa in ln p (k = 2)
    k = t.points.flat_index[0] // (grid.ny * grid.nx)
    assert k == 2
    assert t.localization == Localization(200000., 4000.)


def test_surface_operator_and_elevation_gate(tmp_path):
    states, thb = _states()
    grid = _Grid(terrain=100.0)
    when = "2026-10-01T19:10:00Z"
    rows = _table(tmp_path, [
        _row("madis-mesonet", "M1", 2.0, 2.0, 120, "", when,
             "temperature_k", 290.0, 1.5, "screen_temperature_2m"),
        _row("madis-mesonet", "M2", 3.0, 3.0, 900, "", when,
             "temperature_k", 290.0, 1.5, "screen_temperature_2m"),
    ])
    surface = {"t2": np.stack([np.full((6, 7), 289.0 + m) for m in range(4)])}
    batches, prov = oc.conventional_batches(
        rows, oc.read_type_table(), states=states, thb=thb, grid=grid,
        analysis_time=T0, surface=surface)
    (b,) = batches
    np.testing.assert_allclose(b.points.simulated[:, 0], [289, 290, 291, 292])
    assert b.points.flat_index[0] // (grid.ny * grid.nx) == 0
    assert prov["types"]["mesonet-temperature"]["surface_elevation_mismatch"] == 1
    with pytest.raises(oc.ConventionalObsError, match="surface"):
        oc.conventional_batches(rows, oc.read_type_table(), states=states,
                                thb=thb, grid=grid, analysis_time=T0)


def test_type_table_overrides_and_refusals(tmp_path):
    table = oc.read_type_table(overrides={"*": {"horizontal_m": 90000}})
    assert {loc.horizontal_m for loc in table.batch_localizations()} == {90000}
    off = oc.read_type_table(overrides={"upper-temperature": {"use": False}})
    assert all(not t.use for t in off.types if t.batch == "upper-temperature")
    doc = json.loads(oc.DEFAULT_TYPE_TABLE.read_text())
    doc["types"][0]["horizontal_m"] = 1.0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(doc))
    with pytest.raises(oc.ConventionalObsError, match="two localizations"):
        oc.read_type_table(bad)


def test_iau_fractions_sum_to_the_increment(monkeypatch):
    from gpuwm.core import diagnostics
    from gpuwm.da.iau import IncrementalUpdate

    calls = []
    monkeypatch.setattr(diagnostics, "update_diagnostics",
                        lambda state, opt: calls.append(opt))
    state = SimpleNamespace(thp=np.zeros((2, 3, 3), np.float32),
                            qv=np.full((2, 3, 3), 1e-3, np.float32),
                            elapsed_seconds=3600.0)
    inc = {"thp": np.full((2, 3, 3), 1.0), "qv": np.full((2, 3, 3), -2e-3)}
    iau = IncrementalUpdate(inc, start_seconds=3600.0, window_seconds=1800.0)
    run = SimpleNamespace(dt=45.0)

    def step(s, r, **kw):
        return None

    stepper = iau.stepper(step)
    for n in range(60):  # 2700 s, beyond the window
        state.elapsed_seconds = 3600.0 + 45.0 * n
        stepper(state, run)
    assert iau.steps == 40
    np.testing.assert_allclose(state.thp, 1.0, atol=1e-5)
    assert np.all(state.qv >= 0)
    assert iau.clipped["qv"] > 0
    receipt = iau.finish(state)
    assert receipt["applied_fraction"] == pytest.approx(1.0)
    assert receipt["remainder_at_end"] == pytest.approx(0.0, abs=1e-12)
    assert len(calls) == 40
