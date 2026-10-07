"""CPU contracts for retained native-slot observation consumers.

The observation adapters and covariance/reduction implementation are real.
The retained fixtures are distinct arrays with explicit clocks; these tests
do not claim a native forecast producer or weather qualification.
"""
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.da.spread_vts_observers import (
    build_extra_observations, make_cwp_provider, pool_retained_slots)


class Grid:
    def __init__(self, nz=3, ny=3, nx=4):
        self.nz, self.ny, self.nx = nz, ny, nx
        self.terrain_m = np.zeros((ny, nx))

    def identity_sha256(self):
        return "c" * 64

    def mass_index(self, lat, lon):
        return np.asarray(lon), np.asarray(lat)

    def inside(self, i, j):
        return 0 <= i < self.nx and 0 <= j < self.ny


def slots(grid=None, *, members=2, readonly=False):
    grid = Grid() if grid is None else grid
    shape = (grid.nz, grid.ny, grid.nx)
    result = []
    for offset in (-900, 0, 900):
        for member in range(members):
            sample = (offset // 900 + 1) * members + member
            p = np.broadcast_to(np.linspace(95000., 40000., grid.nz)[:, None, None], shape).copy() + 100. * sample
            temp = 275. + sample + np.log(p / 50000.) * 10.
            qv = np.full(shape, 0.003 + 0.0001 * sample)
            state = {"p": p, "qv": qv,
                "alt": temp * 287. * (1. + 461.6 / 287. * qv) / p,
                "thp": temp / (p / 1.e5) ** (287. / 1004.5) - 300.,
                "php": np.full((grid.nz + 1, grid.ny, grid.nx), float(sample)),
                "mup": np.full((grid.ny, grid.nx), 100. * sample),
                "u": np.full((grid.nz, grid.ny, grid.nx + 1), 3. + sample),
                "v": np.full((grid.nz, grid.ny + 1, grid.nx), 1. + 0.5 * sample),
                "w": np.full((grid.nz + 1, grid.ny, grid.nx), 0.2 * sample),
                "qc": np.full(shape, 1.e-4 * (sample + 1)),
                "qi": np.full(shape, 2.e-5 * (sample + 1)),
                "qs": np.full(shape, 3.e-5 * (sample + 1))}
            state = {name: value.astype(np.float32) for name, value in state.items()}
            surface = {"t2": np.full(shape[1:], 290. + sample),
                "q2": np.full(shape[1:], 0.003 + 0.0001 * sample),
                "psfc": np.full(shape[1:], 98000. + 100. * sample),
                "u10": np.full(shape[1:], 3. + sample),
                "v10": np.full(shape[1:], 2. + sample)}
            z = np.full(shape, 20. + sample, np.float32)
            if readonly:
                for value in list(state.values()) + list(surface.values()) + [z]:
                    value.flags.writeable = False
            result.append(SimpleNamespace(member=member, offset_seconds=offset,
                analysis_seconds=3600., valid_ticks=(3600 + offset) * 10,
                tick_den=10, forecast_origin_seconds=0., observation_cutoff_seconds=0.,
                grid_identity_sha256=grid.identity_sha256(), setup_identity_sha256="d" * 64,
                states=state, reflectivity=z, surface=surface))
    return result


def pool(values, grid=None):
    grid = Grid() if grid is None else grid
    return pool_retained_slots(values, members=2, analysis_seconds=3600.,
                               grid_identity_sha256=grid.identity_sha256())


def test_actual_buffer_pools_deterministic_distinct_slots_without_dense_copy():
    values = slots(readonly=True)
    roster = pool(reversed(values))
    assert roster.member_count == 2
    assert roster.receipt["covariance_samples"] == 6
    assert roster.receipt["central_indices"] == [2, 3]
    assert roster.receipt["sample_order"] == "offset-major/member-major"
    assert roster.pooled_fields["p"].arrays[0] is values[0].states["p"]
    np.testing.assert_array_equal(np.asarray(roster.pooled_fields["p"]),
                                  np.stack([slot.states["p"] for slot in values]))
    np.testing.assert_array_equal(roster.surface["t2"][:, 0, 0], np.arange(290., 296.))
    assert [int(roster.reflectivity_provider(i, roster.snapshots[i])[0, 0, 0]) for i in range(6)] == list(range(20, 26))
    assert len({float(state["php"][0, 0, 0]) for state in roster.snapshots.values()}) == 6
    assert "fixed radar-grid" in roster.receipt["observation_geometry"]


def test_mutable_inputs_become_owned_readonly_arrays_once():
    values = slots()
    roster = pool(values)
    original = roster.snapshots[0]["p"].copy()
    values[0].states["p"][:] = 1.
    values[0].surface["t2"][:] = 1.
    values[0].reflectivity[:] = 1.
    np.testing.assert_array_equal(roster.snapshots[0]["p"], original)
    assert roster.surface["t2"][0, 0, 0] == 290.
    assert roster.dbz[0][0, 0, 0] == 20.
    assert not roster.snapshots[0]["p"].flags.writeable
    with pytest.raises(TypeError):
        roster.snapshots[0]["p"] = original
    with pytest.raises(ValueError, match="original captured state"):
        roster.reflectivity_provider(0, dict(roster.snapshots[0]))


@pytest.mark.parametrize("field,value,match", [
    ("valid_ticks", 36001, "native clock"),
    ("tick_den", 0, "native clock"),
    ("analysis_seconds", 3601., "different analysis"),
    ("forecast_origin_seconds", 1., "common forecast origin"),
    ("observation_cutoff_seconds", 1., "common forecast origin"),
    ("grid_identity_sha256", "e" * 64, "grid identity differs"),
    ("setup_identity_sha256", "e" * 64, "same immutable"),
    ("member", True, "exact integer"),
])
def test_invalid_native_identity_refuses_before_analysis(field, value, match):
    values = slots()
    setattr(values[-1], field, value)
    with pytest.raises(ValueError, match=match):
        pool(values)


def test_missing_or_duplicate_member_and_missing_negative_slot_refuse():
    values = slots()
    with pytest.raises(ValueError, match="all three retained offsets"):
        pool(values[1:])
    with pytest.raises(ValueError, match="unique and complete"):
        pool(values[:-1] + [values[0]])


def test_observation_cutoff_cannot_follow_negative_slot():
    values = slots()
    for value in values:
        value.observation_cutoff_seconds = 3000.
    with pytest.raises(ValueError, match="after its valid time"):
        pool(values)


def test_fractional_origin_is_not_silently_truncated():
    values = slots()
    for value in values:
        value.forecast_origin_seconds = 0.5
    with pytest.raises(ValueError, match="whole-second origin"):
        pool(values)


@pytest.mark.parametrize("change,match", [
    ("field", "field rosters"), ("dtype", "geometry or dtype"),
    ("surface", "retained mass plane"), ("reflectivity", "reflectivity geometry"),
])
def test_native_state_and_diagnostic_geometry_are_not_inferred(change, match):
    values = slots()
    if change == "field":
        del values[-1].states["php"]
    elif change == "dtype":
        values[-1].states["p"] = values[-1].states["p"].astype(np.float64)
    elif change == "surface":
        values[-1].surface["t2"] = np.zeros((1, 1))
    else:
        values[-1].reflectivity = np.zeros((1, 1, 1))
    with pytest.raises(ValueError, match=match):
        pool(values)


def test_central_reduction_preserves_full_posterior_mean_with_exact_roster():
    roster = pool(slots())
    prior = np.asarray(roster.pooled_fields["thp"])
    increment = np.arange(6.)[:, None, None, None] * np.ones_like(prior)
    reduced = roster.reduce_to_central({"thp": increment})["thp"]
    original = np.stack([state["thp"] for state in roster.central_states])
    assert reduced.shape == (2,) + prior.shape[1:]
    np.testing.assert_allclose((original + reduced).mean(axis=0),
                              (prior + increment).mean(axis=0), rtol=1.e-12)
    with pytest.raises(ValueError, match="exact 3K"):
        roster.reduce_to_central({"thp": increment[:1]})
    with pytest.raises(ValueError, match="uncaptured fields"):
        roster.reduce_to_central({"unknown": increment})
    increment[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        roster.reduce_to_central({"thp": increment})


def test_real_member_increment_return_contract_reduces_to_k():
    roster = pool(slots())
    shape = roster.snapshots[0]["thp"].shape
    increments = {index: {"thp": np.full(shape, float(index))} for index in range(6)}
    reduced = roster.reduce_member_increments(increments)
    assert list(reduced) == [0, 1]
    expected = roster.reduce_to_central({"thp": np.stack([increments[index]["thp"] for index in range(6)])})
    np.testing.assert_array_equal(np.stack([reduced[index]["thp"] for index in range(2)]), expected["thp"])
    with pytest.raises(ValueError, match="exact ordered 3K"):
        roster.reduce_member_increments({index: increments[index] for index in range(5)})
    increments[5] = {}
    with pytest.raises(ValueError, match="fields must match"):
        roster.reduce_member_increments(increments)


def test_real_wind_projection_uses_every_shifted_wind_state():
    from gpuwm.da.obs_radar import simulated_radial_velocity
    from gpuwm.da.radar_assimilation import member_earth_winds
    roster = pool(slots())
    shape = roster.dbz[0].shape
    rotation = (np.zeros(shape[1:]), np.ones(shape[1:]))
    beam = (np.full(shape, .6), np.full(shape, .8), np.zeros(shape))
    projected = np.stack([simulated_radial_velocity(
        *member_earth_winds(state, rotation, where="retained fixture"), beam)
        for state in roster.snapshots.values()])
    np.testing.assert_allclose(projected[:, 0, 0, 0], 2.6 + np.arange(6.))
    assert len(set(projected[:, 0, 0, 0])) == 6


def test_real_conventional_pressure_and_surface_operators_receive_all_slots():
    from gpuwm.da.obs_conventional import ObsType, Rows, TypeTable
    grid = Grid()
    roster = pool(slots(grid), grid)
    when = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = Rows(source=np.array(["fixture", "fixture"], object),
        station_id=np.array(["column", "surface"], object),
        lat=np.array([1., 1.]), lon=np.array([1., 1.]),
        elevation_m=np.array([0., 0.]), level_pa=np.array([70000., np.nan]),
        valid_seconds=np.array([when.timestamp(), when.timestamp()]),
        variable=np.array(["temperature_k", "temperature_k"], object),
        value=np.array([280., 292.]), error=np.array([2., 2.]),
        measurement=np.array(["column", "surface"], object), receipts=())
    common = dict(source="fixture", variable="temperature_k", error_inflation=1.,
        error_floor=1., gross_sigma=100., horizontal_m=3000., vertical_m=3000.)
    table = TypeTable(types=(
        ObsType(name="column", measurement="column", operator="pressure_level", batch="column", **common),
        ObsType(name="surface", measurement="surface", operator="surface", batch="surface", **common)),
        window_before_seconds=3600., window_after_seconds=0.,
        surface_elevation_max_diff_m=200., path="fixture", sha256="a" * 64)
    batches, receipt = build_extra_observations(roster, target_grid=grid,
        analysis_time=when, conventional_rows=rows, conventional_table=table, thb=None)
    by_name = {batch.name: batch for batch in batches}
    assert receipt["conventional"]["totals"]["rows_read"] == 2
    column = np.asarray(by_name["conventional:column"].points.simulated)[:, 0]
    np.testing.assert_allclose(column, 275. + np.arange(6.) + 10. * np.log(70000. / 50000.), atol=3.e-5)
    np.testing.assert_allclose(by_name["conventional:surface"].points.simulated[:, 0], np.arange(290., 296.))


def test_real_surface_adapter_uses_shifted_native_diagnostics():
    from gpuwm.da.obs_surface import SurfaceObsConfig
    from test_da_obs_surface import RECORD, T12, _grid
    grid = _grid(nz=3)
    roster = pool(slots(grid), grid)
    batches, receipt = build_extra_observations(roster, target_grid=grid,
        analysis_time=T12, surface_source=RECORD,
        surface_config=SurfaceObsConfig(temperature_error_k=2., wind_speed_error_ms=2.))
    assert batches and receipt["surface"]["batches"]
    temperature = next(batch for batch in batches if batch.name.startswith("temperature_2m:"))
    np.testing.assert_array_equal(np.asarray(temperature.simulated)[:, temperature.mask.astype(bool)],
        np.broadcast_to(np.arange(290., 296.)[:, None], (6, int(temperature.mask.sum()))))
    wind = next(batch for batch in batches if batch.name.startswith("wind_speed_10m:"))
    expected = np.hypot(3. + np.arange(6.), 2. + np.arange(6.))
    np.testing.assert_allclose(np.asarray(wind.simulated)[:, wind.mask.astype(bool)],
        np.broadcast_to(expected[:, None], (6, int(wind.mask.sum()))))


def test_real_goes_operator_reads_actual_slot_condensate_and_column_mass():
    from gpuwm.da.obsop_cwp import column_mass_per_area
    roster = pool(slots())
    shape = roster.dbz[0].shape
    setup = {"c1h": np.ones(shape[0]), "c2h": np.zeros(shape[0]),
        "dnw": np.full(shape[0], -1. / shape[0]), "mub2d": np.full(shape[1:], 90000.)}
    provider = make_cwp_provider(roster, SimpleNamespace(mp_physics=28),
        setup_arrays=setup, setup_identity_sha256="d" * 64)
    cwp = np.stack([provider(index, state) for index, state in roster.snapshots.items()])
    for index, state in roster.snapshots.items():
        expected = 1000. * (state["qc"] + state["qi"] + state["qs"]) * column_mass_per_area(
            setup["c1h"], setup["c2h"], setup["dnw"], setup["mub2d"] + state["mup"])
        np.testing.assert_allclose(cwp[index], expected.sum(axis=0), rtol=2.e-7)
    assert len(set(cwp[:, 0, 0])) == 6
    with pytest.raises(ValueError, match="setup differs"):
        make_cwp_provider(roster, SimpleNamespace(mp_physics=28),
            setup_arrays=setup, setup_identity_sha256="e" * 64)


def test_real_native_dewpoint_adapter_uses_all_slot_q2_and_pressure():
    from gpuwm.da.obs_surface import SurfaceObsConfig
    from test_da_obs_surface import RECORD, T12, _grid
    grid = _grid(nz=3)
    values = slots(grid)
    for slot in values[-2:]:
        slot.surface["psfc"] += 1000.
    roster = pool(values, grid)
    batches, receipt = build_extra_observations(roster, target_grid=grid,
        analysis_time=T12, surface_source=RECORD,
        surface_config=SurfaceObsConfig(dewpoint_error_k=2.))
    dewpoint = next(batch for batch in batches if batch.name.startswith("dewpoint_2m:"))
    expected = 0.003 + .0001 * np.arange(6.)
    np.testing.assert_allclose(np.asarray(dewpoint.simulated)[:, dewpoint.mask.astype(bool)],
        np.broadcast_to(expected[:, None], (6, int(dewpoint.mask.sum()))))
    used_pressure = [qc["dewpoint_pressure_proxy_pa"]
        for qc in receipt["surface"]["station_qc"].values()
        if "dewpoint_pressure_proxy_pa" in qc]
    assert used_pressure
    np.testing.assert_allclose(used_pressure, roster.surface["psfc"].mean(axis=0)[0, 0], rtol=1.e-14)
    assert not np.isclose(used_pressure[0], roster.surface["psfc"][2:4].mean(axis=0)[0, 0])


def test_vtsp_does_not_reuse_original_surface_h_as_recentered_h():
    with pytest.raises(ValueError, match="native nonlinear surface rediagnosis"):
        pool_retained_slots(slots(), members=2, analysis_seconds=3600,
                            grid_identity_sha256="c" * 64, mode="vtsp")
