"""Each domain's long step follows its ground and its crest-level wind.

Under a jet at crest height, 3 km ridges of slope 0.4 stopped within two
minutes on four and on six acoustic substeps alike, and 1 km crests of
slope 0.6 stopped on four; a shorter long step or six substeps held them.
These tests pin the measured map and its conservative reading, the wind
read from each door's own inputs, the experiment the rule writes (whole
divisions, so every cadence stays whole) and the doors that apply it;
tests/test_steep_terrain_step.py pins the stability it buys through the
production step().
"""
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from fractions import Fraction
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from gpuwm import terrain_clock as tc
from gpuwm.acoustic_adaptation import AcousticAdaptation, SlopeReading


def _wind(speed, crest=4500.0, label="d01"):
    return tc.CrestWind(label=label, crest_height_m=crest, wind_m_s=speed,
                        when="start", source=label, height_m=crest)


def _wizard_experiment(dims, ratios, root_dx_m, *, ref_lat=-33.0, hours=1,
                       clock="fixed"):
    from gpuwm import domain_wizard as wizard

    text = wizard.render_config(
        name="terrain-clock", start_time=datetime(2025, 7, 1, 0),
        hours=hours, projection={
            "map_proj": "lambert", "ref_lat": ref_lat, "ref_lon": -70.1,
            "truelat1": ref_lat + 10.0, "truelat2": ref_lat - 10.0,
            "stand_lon": -70.1},
        dims=dims, ratios=ratios, fetch_hints={"source": "gfs"},
        case_data=None, root_dx_m=root_dx_m, history_interval_s=300.0,
        clock=clock)
    return wizard.experiment_from_text(text, source="terrain-clock.toml")


def _acoustic(exp, slopes):
    rows = []
    for dc in exp.domains:
        gid = int(dc.grid_id)
        rows.append(AcousticAdaptation(
            grid_id=gid, reading=SlopeReading(f"d{gid:02d}",
                                              float(slopes[gid]),
                                              ("x", 0, 1)),
            epssm=0.5, configured=int(dc.run.time_step_sound),
            time_step_sound=int(dc.run.time_step_sound), four_below=0.7,
            six_below=0.85))
    return tuple(rows)


# ---------------------------------------------------------------------------
# The measured map and its reading.
# ---------------------------------------------------------------------------


def test_the_map_ships_with_the_package_and_covers_the_generated_ladder():
    table = tc.measured_map()
    spacings = {row.dx_m for row in table.rows}
    assert {500.0, 1000.0, 2000.0, 3000.0, 4000.0} <= spacings
    assert {4, 6} == {row.sound_steps for row in table.rows}
    assert table.top == 5.0
    assert max(table.winds) >= 100.0
    assert tc.MAP_PATH.name == "terrain_clock_map.json"


@pytest.mark.parametrize("dx", [500.0, 1000.0, 2000.0, 3000.0, 4000.0])
def test_flat_and_moderate_ground_holds_the_default_step_at_any_wind(dx):
    """The generated step (5 s/km) holds; where the rows were tried at
    longer steps (3 km, A73) the reading is exactly what held there: 11,
    10 and 5 s/km at 20, 60 and 100 m/s, each under a longer step seen to
    stop, so none of them reads as holding every step."""
    measured_3km = {20.0: 11.0, 60.0: 10.0, 100.0: 5.0}
    for wind in (20.0, 60.0, 100.0):
        reading = tc.read_map(dx, 1200.0, 0.05, wind, 4)
        if dx == 3000.0:
            assert reading.per_km == measured_3km[wind]
            assert not reading.held_everything_tried
        else:
            assert reading.per_km == tc.measured_map().top
            assert reading.held_everything_tried
        assert reading.beyond == ()


def test_a_jet_over_a_tall_3km_ridge_needs_a_shorter_step():
    """The open item this rule closes: 4.5 km crests of slope 0.4 under
    60 to 70 m/s stopped on four and six substeps at 15 s."""
    for count in (4, 6):
        reading = tc.read_map(3000.0, 4500.0, 0.35, 70.0, count)
        assert reading.per_km is not None
        assert reading.per_km < 5.0


def test_at_1km_six_substeps_hold_what_four_do_not():
    four = tc.read_map(1000.0, 6456.0, 0.6, 60.0, 4)
    six = tc.read_map(1000.0, 6456.0, 0.6, 60.0, 6)
    assert four.per_km < 5.0
    assert six.per_km == 5.0


def test_the_reading_never_lengthens_with_more_wind_or_slope():
    table = tc.measured_map()
    for dx in (1000.0, 3000.0):
        for count in (4, 6):
            for crest in (1500.0, 3000.0, 4500.0, 6456.0):
                last_slope = None
                for slope in (0.1, 0.2, 0.3, 0.4, 0.5):
                    last_wind = None
                    for wind in table.winds:
                        value = tc.read_map(dx, crest, slope, wind, count,
                                            table).per_km
                        value = -1.0 if value is None else value
                        if last_wind is not None:
                            assert value <= last_wind
                        last_wind = value
                    value = tc.read_map(dx, crest, slope, 60.0, count,
                                        table).per_km
                    value = -1.0 if value is None else value
                    if last_slope is not None:
                        assert value <= last_slope
                    last_slope = value


def test_a_spacing_between_rows_takes_the_less_stable_neighbour():
    table = tc.measured_map()
    between = tc.read_map(2500.0, 4500.0, 0.35, 80.0, 4, table)
    low = tc.read_map(2000.0, 4500.0, 0.35, 80.0, 4, table)
    high = tc.read_map(3000.0, 4500.0, 0.35, 80.0, 4, table)
    assert between.dx_rows == (2000.0, 3000.0)
    if low.per_km is None or high.per_km is None:
        assert between.per_km is None
    else:
        assert between.per_km == min(low.per_km, high.per_km)


def test_readings_past_the_map_say_which_edge():
    table = tc.measured_map()
    assert "wind" in tc.read_map(3000.0, 3000.0, 0.2, 140.0, 4,
                                 table).beyond
    assert "crest" in tc.read_map(3000.0, 9500.0, 0.2, 40.0, 4,
                                  table).beyond
    assert "slope" in tc.read_map(3000.0, 3000.0, 2.0, 40.0, 4,
                                  table).beyond


# ---------------------------------------------------------------------------
# The decision per domain.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Run:
    dx: float = 3000.0
    dy: float = 3000.0
    time_step_sound: int = 4
    use_adaptive_time_step: bool = False
    max_time_step: int = -1
    max_time_step_den: int = 0
    min_time_step: int = -1
    min_time_step_den: int = 0


def test_a_domain_the_map_holds_runs_exactly_as_configured():
    adaptation = tc.derive_clock(1, _Run(), Fraction(15), 0.05,
                                 _wind(90.0, crest=900.0))
    assert not adaptation.adapted
    assert adaptation.status == "AS_CONFIGURED"
    assert adaptation.division == 1 and adaptation.time_step_sound == 4


def test_a_jet_over_steep_3km_ground_divides_the_step():
    adaptation = tc.derive_clock(1, _Run(), Fraction(15), 0.35,
                                 _wind(70.0, crest=4500.0))
    assert adaptation.status == "ADAPTED"
    assert adaptation.division >= 2
    assert float(adaptation.dt) <= adaptation.held_per_km * 3.0
    assert adaptation.dt == Fraction(15, adaptation.division)
    line = adaptation.sentence()
    assert line.startswith("time step: d01's steepest terrain slope is 0.35")
    assert "70 m/s" in line and "instead of 15 s" in line


def test_at_1km_the_cheapest_remedy_is_six_substeps_on_the_same_step():
    adaptation = tc.derive_clock(
        1, _Run(dx=1000.0, dy=1000.0), Fraction(5), 0.6,
        _wind(60.0, crest=6456.0))
    assert adaptation.division == 1
    assert adaptation.time_step_sound == 6
    assert "6 acoustic substeps per step instead of 4" in adaptation.sentence()


def test_a_larger_configured_count_is_never_lowered():
    adaptation = tc.derive_clock(1, _Run(time_step_sound=8), Fraction(15),
                                 0.35, _wind(70.0))
    assert adaptation.time_step_sound == 8


def test_no_wind_reading_leaves_the_domain_alone():
    adaptation = tc.derive_clock(1, _Run(), Fraction(15), 0.9, None)
    assert not adaptation.adapted
    assert adaptation.status == "NO_WIND_READING"


def test_the_adaptive_clock_takes_the_held_step_as_a_ceiling():
    run = _Run(use_adaptive_time_step=True)
    adaptation = tc.derive_clock(1, run, Fraction(12), 0.35,
                                 _wind(70.0, crest=4500.0))
    assert adaptation.ceiling is not None
    assert float(adaptation.ceiling) <= adaptation.held_per_km * 3.0 + 1e-9
    assert (adaptation.ceiling * 100).denominator == 1


# ---------------------------------------------------------------------------
# The experiment the rule writes.
# ---------------------------------------------------------------------------


def test_the_root_step_divides_and_every_cadence_stays_whole():
    from gpuwm.core.clock import build_schedule, resolve_clock

    exp = _wizard_experiment([(60, 60), (60, 60)], (3,), 3000.0)
    assert exp.dt_exact(1) == 15 and exp.dt_exact(2) == 5
    adapted, plan = tc.retime_experiment(exp, {1: 2}, {})
    assert plan == {1: (2, 1), 2: (2, 3)}
    root = adapted.domains[0]
    assert (root.time_step, root.time_step_fract_num,
            root.time_step_fract_den) == (7, 1, 2)
    assert adapted.dt_exact(1) == Fraction(15, 2)
    assert adapted.dt_exact(2) == Fraction(5, 2)
    assert root.run.dt == 7.5 and adapted.domains[1].run.dt == 2.5
    clock = resolve_clock(adapted, lbc_interval_s=3600)
    build_schedule(adapted, clock)


def test_a_parent_cut_deep_enough_leaves_its_nest_on_its_own_step():
    exp = _wizard_experiment([(60, 60), (60, 60)], (3,), 3000.0)
    adapted, plan = tc.retime_experiment(exp, {1: 3}, {})
    assert plan[2] == (1, 1)
    assert adapted.dt_exact(1) == 5 and adapted.dt_exact(2) == 5
    assert adapted.domains[1].run.dt == exp.domains[1].run.dt


def test_a_nest_alone_takes_a_larger_step_ratio_and_its_parent_is_untouched():
    from gpuwm.core.clock import resolve_clock

    exp = _wizard_experiment([(60, 60), (60, 60)], (3,), 3000.0)
    adapted, plan = tc.retime_experiment(exp, {2: 2}, {2: 6})
    assert plan == {1: (1, 1), 2: (2, 6)}
    assert adapted.domains[0] is exp.domains[0]
    assert adapted.domains[1].parent_time_step_ratio == 6
    assert adapted.domains[1].run.time_step_sound == 6
    assert adapted.dt_exact(2) == Fraction(5, 2)
    resolve_clock(adapted, lbc_interval_s=3600)


def test_nothing_to_change_hands_back_the_same_experiment():
    exp = _wizard_experiment([(60, 60)], (), 3000.0)
    same, _ = tc.retime_experiment(exp, {1: 1}, {})
    assert same is exp
    adapted, adaptations = tc.adapt_experiment_clock(
        exp, {1: 0.05}, {1: _wind(80.0, crest=800.0)})
    assert adapted is exp
    assert [a.status for a in adaptations] == ["AS_CONFIGURED"]


def test_one_line_per_changed_domain_and_one_for_a_nest_its_parent_moves():
    exp = _wizard_experiment([(60, 60), (60, 60)], (3,), 3000.0)
    lines = []
    adapted, adaptations = tc.adapt_experiment_clock(
        exp, {1: 0.35, 2: 0.05},
        {1: _wind(70.0), 2: _wind(70.0, crest=800.0, label="d02")},
        announce=lines.append, caution=lines.append)
    assert adapted.dt_exact(1) < 15
    assert [a.grid_id for a in adaptations if a.adapted] == [1, 2]
    assert len(lines) == 4
    assert "d01 keeps the shipped domain-wide terrain clock" in lines[0]
    assert lines[1].startswith("time step: d01's")
    assert "d02 keeps the shipped domain-wide terrain clock" in lines[2]
    assert lines[3].startswith("time step: d02 runs")
    receipt = tc.clock_receipt(adaptations)
    assert receipt["schema"] == tc.TERRAIN_CLOCK_SCHEMA
    assert receipt["domains"][0]["crest_level_wind_m_s"] == 70.0


def test_adaptive_ceiling_is_written_and_min_follows_it_down():
    exp = _wizard_experiment([(60, 60)], (), 3000.0)
    root = exp.domains[0]
    exp = replace(exp, domains=(replace(root, run=replace(
        root.run, use_adaptive_time_step=True)),))
    adapted, adaptations = tc.adapt_experiment_clock(
        exp, {1: 0.35}, {1: _wind(70.0)})
    run = adapted.domains[0].run
    ceiling = adaptations[0].ceiling
    assert ceiling is not None
    assert Fraction(run.max_time_step, run.max_time_step_den or 1) == ceiling
    from gpuwm.config import _adaptive_interval, validate_run_config
    lower = _adaptive_interval(run.min_time_step, run.min_time_step_den,
                               "min_time_step")
    assert lower is None or lower <= ceiling
    # The capped configuration is one the admission battery accepts, and
    # the clock the adaptive run starts from resolves on it.
    validate_run_config(run)
    from gpuwm.core.clock import resolve_clock
    resolve_clock(adapted, lbc_interval_s=3600)


# ---------------------------------------------------------------------------
# The wind read from the inputs.
# ---------------------------------------------------------------------------


def test_the_crest_band_runs_from_the_ground_to_the_first_level_above():
    heights = np.array([[100.0, 4600.0], [2000.0, 4700.0],
                        [4400.0, 5200.0], [4800.0, 6000.0],
                        [7000.0, 8000.0]])
    band = tc.crest_band(heights, 4500.0)
    assert band[:, 0].tolist() == [True, True, True, True, False]
    assert band[:, 1].tolist() == [True, False, False, False, False]


def _column_state(nz=6, ny=12, nx=14, jet=60.0, crest=4500.0):
    """A C-grid state with a jet in one layer, and the heights to find it."""
    rng = np.random.default_rng(3)
    znw = np.linspace(1.0, 0.0, nz + 1)
    c1f = znw.copy()
    c2f = np.zeros(nz + 1)
    c1h = 0.5 * (c1f[1:] + c1f[:-1])
    c2h = np.zeros(nz)
    mub = 80000.0 + 1000.0 * rng.random((ny, nx))
    phb = np.stack([9.81 * 1500.0 * k * np.ones((ny, nx))
                    for k in range(nz + 1)])
    u = np.full((nz, ny, nx + 1), 10.0)
    v = np.full((nz, ny + 1, nx), 5.0)
    return SimpleNamespace(c1h=c1h, c2h=c2h, c1f=c1f, c2f=c2f, mub=mub,
                           phb=phb, u=u, v=v, php=np.zeros((nz + 1, ny, nx)),
                           mup=np.zeros((ny, nx)))


def _coupled(state, msfu, msfv):
    """The coupled boundary fields, as WRF couples them."""
    mu = state.mub + state.mup
    mux = np.concatenate([mu[:, :1], 0.5 * (mu[:, 1:] + mu[:, :-1]),
                          mu[:, -1:]], axis=1)
    muy = np.concatenate([mu[:1], 0.5 * (mu[1:] + mu[:-1]), mu[-1:]], axis=0)
    c1h = state.c1h[:, None, None]
    c2h = state.c2h[:, None, None]
    chf = state.c1f[:, None, None] * mu[None] + state.c2f[:, None, None]
    return {"u": (c1h * mux[None] + c2h) * state.u / msfu[None],
            "v": (c1h * muy[None] + c2h) * state.v / msfv[None],
            "phi": chf * state.php, "mu": state.mup[None]}


def test_boundary_winds_are_read_back_through_their_coupling():
    from gpuwm.ingest.lateral_bc import build_lateral_boundaries

    first = _column_state()
    second = _column_state()
    # A 55 m/s jet on the west edge, at the level whose centre is 2250 m,
    # arrives by the end of the window; a stronger one aloft stays above
    # the crest band.
    second.u[1, :, :3] = 55.0
    second.u[4, :, :] = 90.0
    ny, nx = first.mub.shape
    msfu = np.full((ny, nx + 1), 1.02)
    msfv = np.full((ny + 1, nx), 1.02)
    boundaries = build_lateral_boundaries(
        [_coupled(first, msfu, msfv), _coupled(second, msfu, msfv)],
        [0.0, 10800.0], spec_bdy_width=5)
    geometry = tc.BoundaryGeometry(
        mub=first.mub, phb=first.phb, c1h=first.c1h, c2h=first.c2h,
        c1f=first.c1f, c2f=first.c2f, msfu=msfu, msfv=msfv)
    winds = tc.BoundaryWinds("d01", boundaries, geometry, 10800.0)
    speed, height, when = winds.strongest(3000.0)
    assert when == "boundary at +3 h"
    assert speed == pytest.approx(np.hypot(55.0, 5.0), rel=1e-6)
    assert height == pytest.approx(2250.0)
    # Only half the window: the jet is half arrived.
    half = tc.BoundaryWinds("d01", boundaries, geometry, 5400.0)
    speed, _, when = half.strongest(3000.0)
    assert when == "boundary at +1.5 h"
    assert speed == pytest.approx(np.hypot(32.5, 5.0), rel=1e-6)


@pytest.mark.parametrize("dims, message", [
    ((6, 13, 14), "its u tables belong to a 6 x 12 x 15 field, where the "
                  "base state's 6 levels over 13 x 14 columns make it "
                  "6 x 13 x 15"),
    ((6, 12, 15), "its u tables belong to a 6 x 12 x 15 field, where the "
                  "base state's 6 levels over 12 x 15 columns make it "
                  "6 x 12 x 16"),
    ((4, 12, 14), "its u tables belong to a 6 x 12 x 15 field, where the "
                  "base state's 4 levels over 12 x 14 columns make it "
                  "4 x 12 x 15"),
])
def test_boundaries_from_another_grid_are_refused_by_name(dims, message):
    """Boundary tables built on another grid than the base state they are
    read against stopped a prepared tree's preflight on a bare NumPy
    broadcast error; the reading names the domain and both grids."""
    from gpuwm.ingest.lateral_bc import build_lateral_boundaries

    first = _column_state()
    ny, nx = first.mub.shape
    msfu = np.ones((ny, nx + 1))
    msfv = np.ones((ny + 1, nx))
    boundaries = build_lateral_boundaries(
        [_coupled(first, msfu, msfv), _coupled(first, msfu, msfv)],
        [0.0, 10800.0], spec_bdy_width=5)
    nz, gny, gnx = dims
    other = _column_state(nz=nz, ny=gny, nx=gnx)
    geometry = tc.BoundaryGeometry(
        mub=other.mub, phb=other.phb, c1h=other.c1h, c2h=other.c2h,
        c1f=other.c1f, c2f=other.c2f, msfu=np.ones((gny, gnx + 1)),
        msfv=np.ones((gny + 1, gnx)))
    winds = tc.BoundaryWinds("d01", boundaries, geometry, 10800.0)
    with pytest.raises(ValueError) as refused:
        winds.strongest(3000.0)
    text = str(refused.value)
    assert text.startswith("d01's lateral boundary data is for another "
                           "grid than its base state")
    assert message in text


@pytest.mark.parametrize("name", ["MAPFAC_U", "MAPFAC_V"])
def test_map_factors_from_another_grid_are_refused_by_name(name):
    from gpuwm.ingest.lateral_bc import build_lateral_boundaries

    first = _column_state()
    ny, nx = first.mub.shape
    msfu = np.ones((ny, nx + 1))
    msfv = np.ones((ny + 1, nx))
    boundaries = build_lateral_boundaries(
        [_coupled(first, msfu, msfv), _coupled(first, msfu, msfv)],
        [0.0, 10800.0], spec_bdy_width=5)
    factors = {"MAPFAC_U": msfu, "MAPFAC_V": msfv}
    factors[name] = np.ones((30, 30))
    geometry = tc.BoundaryGeometry(
        mub=first.mub, phb=first.phb, c1h=first.c1h, c2h=first.c2h,
        c1f=first.c1f, c2f=first.c2f, msfu=factors["MAPFAC_U"],
        msfv=factors["MAPFAC_V"])
    winds = tc.BoundaryWinds("d01", boundaries, geometry, 10800.0)
    with pytest.raises(ValueError, match=(
            f"d01's {name} is 30 x 30, where its base state's 12 x 14 "
            "columns make it")):
        winds.strongest(3000.0)


class _Reader:
    """The two methods of PreparedCacheReader the rule reads through."""

    def __init__(self, arrays, lbc=None):
        self.arrays = dict(arrays)
        self.header = {"metadata": {"lbc": lbc}}
        self.path = "cache"

    def read_array(self, key):
        return self.arrays[key]


def _cache_reader(state, *, boundaries=None, msfu=None, msfv=None):
    arrays = {"state/u": state.u, "state/v": state.v,
              "state/php": state.php, "state/mup": state.mup,
              "base/phb": state.phb, "base/mub": state.mub,
              "coord/c1h": state.c1h, "coord/c2h": state.c2h,
              "coord/c1f": state.c1f, "coord/c2f": state.c2f}
    lbc = None
    if boundaries is not None:
        intervals = []
        for index, interval in enumerate(boundaries.intervals):
            intervals.append({"start_seconds": interval.start_seconds,
                              "end_seconds": interval.end_seconds,
                              "fields": sorted(interval.fields)})
            for name, field in interval.fields.items():
                for side in ("west", "east", "south", "north"):
                    part = getattr(field, side)
                    arrays[f"lbc/{index}/{name}/{side}/value"] = part.value
                    arrays[f"lbc/{index}/{name}/{side}/tendency"] = (
                        part.tendency)
        lbc = {"spec_bdy_width": boundaries.spec_bdy_width,
               "spec_zone": boundaries.spec_zone,
               "relax_zone": boundaries.relax_zone, "intervals": intervals}
    return _Reader(arrays, lbc)


def test_start_winds_come_from_the_prepared_cache():
    state = _column_state()
    state.u[2, 4, 6] = 70.0
    state.u[2, 4, 7] = 70.0
    winds = tc.start_winds_from_cache(_cache_reader(state), "d01")
    speed, height, when = winds.strongest(4000.0)
    assert when == "start"
    assert speed == pytest.approx(np.hypot(70.0, 5.0))
    assert height == pytest.approx(3750.0)
    # Above the band: the same jet is not crest level for a 2 km crest.
    speed, _, _ = winds.strongest(2000.0)
    assert speed == pytest.approx(np.hypot(10.0, 5.0))


def _ramp_static(ny, nx, *, slope=0.35, dx=3000.0, crest=4500.0):
    """Ground rising across the grid at one slope to a flat crest."""
    rise = (np.arange(nx) - nx // 3) * slope * dx
    terrain = np.broadcast_to(np.clip(rise, 0.0, crest), (ny, nx)).copy()
    return MappingProxyType({"HGT_M": terrain,
                             "MAPFAC_U": np.ones((ny, nx + 1)),
                             "MAPFAC_V": np.ones((ny + 1, nx))})


def _jet_state(ny, nx, nz=6, jet=70.0):
    state = _column_state(nz=nz, ny=ny, nx=nx)
    state.u[:3] = jet
    return state


# ---------------------------------------------------------------------------
# The doors.
# ---------------------------------------------------------------------------


def _metem_inputs(exp, statics, readers, boundaries=None):
    from gpuwm.metem_forecast import MetemDomainBundle
    from gpuwm.wrfinput_forecast import WrfTreeInputs

    bundles = tuple(
        MetemDomainBundle(grid_id=int(dc.grid_id), cache=None,
                          cache_identity={}, cache_reader=readers[
                              int(dc.grid_id)],
                          static_fields=statics[int(dc.grid_id)],
                          authority_sha256={}, geog_selection=None,
                          fractional_seaice=False, isoilwater=14)
        for dc in exp.domains)
    return WrfTreeInputs(
        prepared_root=None, experiment_config=None, experiment=exp,
        grids=tuple(None for _ in exp.domains), domains=bundles,
        forcing_hours=(0.0, 1.0), boundary_interval_seconds=3600,
        source_identity={}, execution_plan={}, authority_sha256={},
        artifact_paths={}, boundaries=boundaries, source="met_em")


def test_the_prepared_tree_door_divides_a_step_under_a_jet():
    """Every prepared-cache door (prepared tree, met_em) reaches the tree
    runner, which reads the caches' start state and boundary data."""
    from gpuwm.prepared_domain_tree_forecast import _with_terrain_acoustics

    exp = _wizard_experiment([(60, 60)], (), 3000.0)
    static = _ramp_static(60, 60)
    state = _jet_state(60, 60)
    inputs = _metem_inputs(exp, {1: static}, {1: _cache_reader(state)})
    derived = _with_terrain_acoustics(inputs)
    assert derived.experiment.dt_exact(1) < exp.dt_exact(1)
    row = derived.terrain_clock["domains"][0]
    assert row["status"] in {"ADAPTED", "BEYOND_MEASURED"}
    assert row["crest_level_wind_m_s"] == pytest.approx(np.hypot(70.0, 5.0))
    assert _with_terrain_acoustics(derived) is derived
    # The same ground in a light wind: the experiment is handed back as is.
    calm = _metem_inputs(exp, {1: static},
                         {1: _cache_reader(_jet_state(60, 60, jet=10.0))})
    assert _with_terrain_acoustics(calm).experiment is exp


def test_the_wrfinput_door_reads_the_files_own_arrays():
    from gpuwm.prepared_domain_tree_forecast import _with_terrain_acoustics
    from gpuwm.wrfinput_forecast import WrfDomainBundle, WrfTreeInputs

    exp = _wizard_experiment([(60, 60)], (), 3000.0)
    static = _ramp_static(60, 60)
    state = _jet_state(60, 60)
    restored = SimpleNamespace(raw={
        "U": state.u[None], "V": state.v[None], "PH": state.php[None],
        "PHB": state.phb[None], "MUB": state.mub[None],
        "C1H": state.c1h[None], "C2H": state.c2h[None],
        "C1F": state.c1f[None], "C2F": state.c2f[None]})
    inputs = WrfTreeInputs(
        prepared_root=None, experiment_config=None, experiment=exp,
        grids=(None,), domains=(WrfDomainBundle(
            grid_id=1, restored=restored, static_fields=static,
            authority_sha256={}, landuse=None, geog_selection=None),),
        forcing_hours=(0.0, 1.0), boundary_interval_seconds=3600,
        source_identity={}, execution_plan={}, authority_sha256={},
        artifact_paths={}, boundaries=None)
    derived = _with_terrain_acoustics(inputs)
    assert derived.experiment.dt_exact(1) < exp.dt_exact(1)


def test_the_prepared_single_domain_door_applies_the_rule():
    from gpuwm.prepared_single_domain_forecast import _terrain_derivations

    exp = _wizard_experiment([(60, 60)], (), 3000.0)
    static = _ramp_static(60, 60)
    receipt = {}
    adapted = _terrain_derivations(
        exp, static, None, _cache_reader(_jet_state(60, 60)), receipt)
    assert adapted.dt_exact(1) < exp.dt_exact(1)
    assert receipt["terrain_clock"]["domains"][0]["step_division"] >= 2
    assert "acoustic_substeps" in receipt
    calm = {}
    same = _terrain_derivations(
        exp, static, None, _cache_reader(_jet_state(60, 60, jet=10.0)), calm)
    assert same is exp
    assert calm["terrain_clock"]["domains"][0]["status"] == "AS_CONFIGURED"


def _snapshot(valid_time, jet):
    from gpuwm.ingest.grib import Era5Snapshot

    lat = np.arange(-40.0, -26.0, 0.25)
    lon = np.arange(284.0, 296.0, 0.25)
    levels = np.array([850.0, 700.0, 500.0, 400.0, 300.0])
    heights = np.array([1500.0, 3000.0, 5600.0, 7200.0, 9200.0])
    shape = (levels.size, lat.size, lon.size)
    uu = np.full(shape, 10.0)
    uu[2] = jet
    uu[4] = 120.0
    return Era5Snapshot(
        valid_time=valid_time, levels_hpa=levels, latitude=lat,
        longitude=lon, fields={
            "UU": uu, "VV": np.zeros(shape),
            "GHT": np.broadcast_to(heights[:, None, None], shape).copy()})


def test_the_run_route_reads_the_forcing_over_the_window(monkeypatch):
    from pathlib import Path

    from gpuwm import runtime

    exp = _wizard_experiment([(60, 60)], (), 3000.0, hours=3)
    start = exp.start_time
    snapshots = {start: _snapshot(start, 20.0),
                 start + timedelta(hours=3): _snapshot(
                     start + timedelta(hours=3), 72.0),
                 start + timedelta(hours=6): _snapshot(
                     start + timedelta(hours=6), 150.0)}
    monkeypatch.setattr(runtime, "forcing_snapshots",
                        lambda data, catalog=None: snapshots)
    import gpuwm.ingest.preflight as preflight
    monkeypatch.setattr(preflight, "build_input_catalog", lambda data: None)
    from gpuwm.static.projection import grids_from_projection_config
    grids = tuple(grids_from_projection_config(exp))
    terrain = {1: np.full((60, 60), 5000.0)}
    terrain[1][:, :30] = 0.0
    acoustic = _acoustic(exp, {1: 0.35})
    data = SimpleNamespace(forcing=[Path("gfs.grb2")])
    adapted, adaptations = runtime._terrain_clock_for_case(
        exp, data, acoustic, terrain, grids, {})
    assert adaptations[0].crest.wind_m_s == pytest.approx(72.0)
    assert adaptations[0].crest.when == "boundary at +3 h"
    assert adapted.dt_exact(1) < exp.dt_exact(1)


def test_boundary_winds_read_back_from_a_prepared_cache():
    from gpuwm.ingest.lateral_bc import build_lateral_boundaries

    first = _column_state()
    second = _column_state()
    second.u[1, :, :3] = 55.0
    ny, nx = first.mub.shape
    msfu = np.ones((ny, nx + 1))
    msfv = np.ones((ny + 1, nx))
    boundaries = build_lateral_boundaries(
        [_coupled(first, msfu, msfv), _coupled(second, msfu, msfv)],
        [0.0, 10800.0], spec_bdy_width=5)
    reader = _cache_reader(first, boundaries=boundaries)
    read = tc.cache_boundaries(reader)
    assert [i.start_seconds for i in read.intervals] == [0.0]
    assert set(read.intervals[0].fields) == {"u", "v", "mu", "phi"}
    geometry = tc.boundary_geometry_from_cache(
        reader, {"MAPFAC_U": msfu, "MAPFAC_V": msfv})
    speed, _, when = tc.BoundaryWinds("d01", read, geometry,
                                      10800.0).strongest(3000.0)
    assert when == "boundary at +3 h"
    assert speed == pytest.approx(np.hypot(55.0, 5.0), rel=1e-6)


def test_a_cache_without_complete_boundary_tables_gives_no_boundary():
    state = _column_state()
    reader = _cache_reader(state)
    assert tc.cache_boundaries(reader) is None
    reader.header["metadata"]["lbc"] = {
        "spec_bdy_width": 5, "spec_zone": 1, "relax_zone": 4,
        "intervals": [{"start_seconds": 0.0, "end_seconds": 3600.0,
                       "fields": ["mu", "u", "v"]}]}
    assert tc.cache_boundaries(reader) is None


def test_ground_and_wind_past_every_held_step_is_said_even_unchanged():
    """A 1 km domain over an 8.85 km crest under 80 m/s: no measured step
    holds, the most stable pair measured is the configured one, and the
    run says it may still stop."""
    lines = []
    exp = _wizard_experiment([(60, 60)], (), 1000.0)
    root = exp.domains[0]
    exp = replace(exp, domains=(replace(root, run=replace(
        root.run, time_step_sound=6)),))
    adapted, adaptations = tc.adapt_experiment_clock(
        exp, {1: 1.5}, {1: _wind(80.0, crest=8600.0)},
        announce=lines.append, caution=lines.append)
    assert adaptations[0].status == "BEYOND_MEASURED"
    assert len(lines) == 2
    assert "keeps the shipped domain-wide terrain clock" in lines.pop(0)
    assert "holds no step there at any substep count" in lines[0]
    assert "may still stop" in lines[0]


def test_past_the_strongest_held_wind_the_line_names_the_most_stable_pair():
    """A 3 km domain under a 79 m/s jet over an 8 km crest: the map holds
    no step at that wind, so the domain runs the most stable pair it
    measured at a weaker one, and its line never says the map holds a step
    at a wind it holds none at."""
    table = tc.StableStepMap(
        winds=(40.0, 60.0, 80.0), ladder=(5.0, 4.0, 3.5), seconds=1800.0,
        rows=(tc.MapRow(3000.0, 8000.0, 0.7, 4, (5.0, 4.0, None)),
              tc.MapRow(3000.0, 8000.0, 0.7, 6, (5.0, 3.5, None))))
    adaptation = tc.derive_clock(1, _Run(), Fraction(15), 0.63,
                                 _wind(79.0, crest=7964.0), table=table)
    assert adaptation.status == "BEYOND_MEASURED"
    assert adaptation.dt == Fraction(15, 2)
    assert adaptation.time_step_sound == 6
    line = adaptation.beyond_sentence()
    assert "the measured map holds no step at this wind" in line
    assert ("the most stable pair it measured holds steps up to 10.5 s "
            "with 6 substeps at a weaker wind") in line
    assert "d01 runs 7.5 s steps instead of 15 s" in line
    assert line.endswith("and may still stop")


def test_a_boundary_reading_at_the_start_says_so():
    assert tc._boundary_when(0.0) == "boundary at the start"
    assert tc._boundary_when(10800.0) == "boundary at +3 h"


def test_the_run_route_records_a_changed_clock_and_nothing_else(tmp_path):
    import json

    from gpuwm import runtime

    exp = _wizard_experiment([(60, 60)], (), 3000.0)
    _, calm = tc.adapt_experiment_clock(
        exp, {1: 0.05}, {1: _wind(30.0, crest=800.0)})
    assert runtime._write_terrain_clock_receipt(tmp_path, calm) is None
    assert not (tmp_path / runtime.TERRAIN_CLOCK_RECEIPT_NAME).exists()
    _, jet = tc.adapt_experiment_clock(exp, {1: 0.35}, {1: _wind(70.0)})
    path = runtime._write_terrain_clock_receipt(tmp_path, jet)
    receipt = json.loads(path.read_text())
    assert receipt["schema"] == tc.TERRAIN_CLOCK_SCHEMA
    assert receipt["domains"][0]["step_division"] >= 2


def test_a_crest_a_hair_over_a_mapped_one_reads_that_row():
    table = tc.measured_map()
    assert tc.read_map(1000.0, 6456.3, 0.8, 50.0, 6, table).crest_row == 6456.0
    assert tc.read_map(1000.0, 6600.0, 0.8, 50.0, 6, table).crest_row == 8000.0


# ---------------------------------------------------------------------------
# The adaptive clock's substep count.
# ---------------------------------------------------------------------------


def test_on_the_adaptive_clock_the_map_is_read_at_the_count_that_runs():
    """The adaptive clock derives its count from the live step, 4 at every
    short 1 km step, whatever time_step_sound says; a configured six there
    is not what runs, so the map is read at four and the six it needs
    become the floor under the clock's count."""
    run = _Run(dx=1000.0, dy=1000.0, time_step_sound=6,
               use_adaptive_time_step=True, max_time_step=5)
    adaptation = tc.derive_clock(1, run, Fraction(5), 0.6,
                                 _wind(60.0, crest=6456.0))
    assert adaptation.configured_sound == 4
    assert adaptation.time_step_sound == 6 and adaptation.adapted
    assert ("at least 6 acoustic substeps per step instead of 4"
            in adaptation.sentence())
    assert adaptation.receipt()["min_time_step_sound"] == 6
    fixed = tc.derive_clock(1, replace(run, use_adaptive_time_step=False),
                            Fraction(5), 0.6, _wind(60.0, crest=6456.0))
    assert not fixed.adapted
    assert "min_time_step_sound" not in fixed.receipt()


def _adaptive(exp):
    return replace(exp, domains=tuple(
        replace(dc, run=replace(dc.run, use_adaptive_time_step=True))
        for dc in exp.domains))


def test_the_adaptive_count_is_written_as_the_floor_the_clock_keeps():
    from gpuwm.config import validate_run_config
    from gpuwm.core.adaptive_clock import adaptive_sound_steps

    exp = _adaptive(_wizard_experiment([(60, 60)], (), 1000.0))
    lines = []
    adapted, adaptations = tc.adapt_experiment_clock(
        exp, {1: 0.6}, {1: _wind(60.0, crest=6456.0)},
        announce=lines.append, caution=lines.append)
    run = adapted.domains[0].run
    assert adaptations[0].time_step_sound == 6
    assert run.min_time_step_sound == 6 and run.time_step_sound == 6
    assert len(lines) == 2
    assert "keeps the shipped domain-wide terrain clock" in lines[0]
    assert "at least 6 acoustic substeps" in lines[1]
    validate_run_config(run)
    # What the clock runs at the generated 5 s step and at a short one.
    assert adaptive_sound_steps(Fraction(5), run) == 6
    assert adaptive_sound_steps(Fraction(5, 2), run) == 6
    assert adaptive_sound_steps(Fraction(5, 2), exp.domains[0].run) == 4


def test_a_fixed_clock_never_carries_the_floor():
    exp = _wizard_experiment([(60, 60)], (), 1000.0)
    adapted, _ = tc.adapt_experiment_clock(
        exp, {1: 0.6}, {1: _wind(60.0, crest=6456.0)})
    run = adapted.domains[0].run
    assert run.time_step_sound == 6 and run.min_time_step_sound == 0


def test_the_floor_is_adaptive_policy_everywhere_the_clock_is_declared():
    """Only the clock reads it, so it drops out of every fixed-clock
    identity, a resume may change it like min_time_step, and a prepared
    cache is not refused over it; per domain, as the ground is."""
    from gpuwm.core.model import (ADAPTIVE_POLICY_RUN_FIELDS,
                                  ADAPTIVE_TIMESTEP_RUN_FIELDS)
    from gpuwm.experiment import _DOMAIN_RUN_OVERRIDES
    from gpuwm.ingest.prepared_cache import PREPARATION_INERT_RUN_FIELDS

    assert "min_time_step_sound" in ADAPTIVE_TIMESTEP_RUN_FIELDS
    assert "min_time_step_sound" in ADAPTIVE_POLICY_RUN_FIELDS
    assert "run.min_time_step_sound" in PREPARATION_INERT_RUN_FIELDS
    assert "min_time_step_sound" in _DOMAIN_RUN_OVERRIDES


@pytest.mark.parametrize("floor, message", [(-2, "0 or more"), (5, "even")])
def test_a_floor_the_dynamics_cannot_run_is_refused(floor, message):
    from gpuwm.config import validate_run_config

    run = replace(_wizard_experiment([(60, 60)], (), 1000.0).domains[0].run,
                  use_adaptive_time_step=True, min_time_step_sound=floor)
    with pytest.raises(ValueError, match=message):
        validate_run_config(run)


# ---------------------------------------------------------------------------
# The forcing footprint across a longitude seam.
# ---------------------------------------------------------------------------


def _global_snapshot(lon, jet_lon, jet=90.0):
    from gpuwm.ingest.grib import Era5Snapshot

    lat = np.arange(40.0, 50.25, 0.25)
    levels = np.array([850.0, 700.0, 500.0])
    heights = np.array([1500.0, 3000.0, 5600.0])
    shape = (levels.size, lat.size, lon.size)
    uu = np.full(shape, 10.0)
    far = np.abs(((lon - jet_lon) + 180.0) % 360.0 - 180.0) < 1.0
    uu[:, :, far] = jet
    return Era5Snapshot(
        valid_time=datetime(2025, 1, 10, 0), levels_hpa=levels,
        latitude=lat, longitude=lon, fields={
            "UU": uu, "VV": np.zeros(shape),
            "GHT": np.broadcast_to(heights[:, None, None], shape).copy()})


@pytest.mark.parametrize("axis, domain_lon, jet_lon", [
    (np.arange(0.0, 360.0, 0.25), (-1.5, 1.5), 180.0),
    (np.arange(-180.0, 180.0, 0.25), (178.5, -178.5), 0.0),
])
def test_a_domain_across_the_seam_reads_only_its_own_footprint(
        axis, domain_lon, jet_lon):
    """A domain across 0 degrees on a 0 to 360 source, or across the
    dateline on a -180 to 180 one, used to read every column of its
    latitude band and took a jet half a world away."""
    lat, lon = np.meshgrid(np.linspace(44.0, 46.0, 20),
                           np.linspace(domain_lon[0],
                                       domain_lon[0] + 3.0, 20))
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    winds = tc.SnapshotWinds(
        "d01", (_global_snapshot(axis, jet_lon),), lat, lon,
        datetime(2025, 1, 10, 0))
    found = winds.strongest(4000.0)
    assert found is not None and found[0] == pytest.approx(10.0)
    near = tc.SnapshotWinds(
        "d01", (_global_snapshot(axis, domain_lon[1]),), lat, lon,
        datetime(2025, 1, 10, 0))
    assert near.strongest(4000.0)[0] == pytest.approx(90.0)


def test_a_domain_clear_of_the_seam_reads_the_window_it_always_had():
    rng = np.random.default_rng(7)
    for axis in (np.arange(0.0, 360.0, 0.25), np.arange(-180.0, 180.0, 0.5),
                 np.arange(284.0, 296.0, 0.25)):
        for _ in range(200):
            centre = rng.uniform(axis.min() + 3.0, axis.max() - 3.0)
            values = centre + rng.uniform(-2.0, 2.0, 40)
            linear = np.mod(values, 360.0) if axis.max() > 180.0 else values
            assert np.array_equal(tc._longitude_window(axis, values),
                                  tc._axis_window(axis, linear))


@pytest.mark.parametrize("axis, low", [
    (np.arange(0.0, 360.0, 0.25), -1.5),
    (np.arange(-180.0, 180.0, 0.25), 178.5),
])
def test_the_window_across_the_seam_is_split_there(axis, low):
    """The columns read across the seam are the domain's arc on both sides
    of it, one source spacing wider, and nothing between: a window from the
    smallest to the largest longitude read the whole band instead."""
    values = low + np.linspace(0.0, 3.0, 13)
    values = np.where(values > 180.0, values - 360.0, values)
    cols = tc._longitude_window(axis, values)
    east_of_low = np.mod(axis[cols] - (low - 0.25), 360.0)
    assert cols.size == 3.5 / 0.25 + 1
    assert east_of_low.max() == pytest.approx(3.5)
    seam = 0.0 if axis.max() > 180.0 else 180.0
    before = np.mod(axis[cols] - seam, 360.0) > 180.0
    assert before.any() and (~before).any()


# ---------------------------------------------------------------------------
# Rows measured past the ladder's longest step.
# ---------------------------------------------------------------------------


def _tried_table(entry, tried=13.0):
    """A 3 km row held ``entry`` s/km at every wind, with steps up to
    ``tried`` s/km tried there."""
    rows = tuple(tc.MapRow(3000.0, 4500.0, 0.4, count, (entry,) * 3,
                           (tried,) * 3) for count in (4, 6))
    return tc.StableStepMap(winds=(20.0, 60.0, 100.0), ladder=(5.0, 4.0),
                            seconds=1800.0, rows=rows)


def test_an_adaptive_step_past_what_the_ground_held_is_capped_there():
    """The defect: an entry at the ladder's longest step read as "holds
    every step", so a 3 km adaptive run kept a max_time_step of 41 s over
    ground the map measured only to 15 s.  On a row tried to 39 s whose
    entry is 24 s (the next step tried stopped), the step is capped at
    24 s; the fixed 15 s step it holds stays as configured."""
    table = _tried_table(8.0)
    run = _Run(use_adaptive_time_step=True, max_time_step=41)
    adaptation = tc.derive_clock(1, run, Fraction(15), 0.35,
                                 _wind(60.0, crest=4500.0), table=table)
    assert adaptation.reading.top_per_km == 13.0
    assert not adaptation.reading.held_everything_tried
    assert adaptation.ceiling == Fraction(24)
    assert adaptation.division == 1
    assert adaptation.receipt()["max_time_step_s"]["seconds"] == 24.0
    assert adaptation.receipt()["map_rows"]["longest_step_tried_s"] == 39.0
    fixed = tc.derive_clock(1, _Run(), Fraction(15), 0.35,
                            _wind(60.0, crest=4500.0), table=table)
    assert not fixed.adapted


def test_a_row_that_held_every_step_tried_leaves_the_adaptive_clock_alone():
    """Past the longest step measured on ground that held it, the adaptive
    clock's own limits govern, as before the rows were extended."""
    table = _tried_table(13.0)
    run = _Run(use_adaptive_time_step=True, max_time_step=41)
    adaptation = tc.derive_clock(1, run, Fraction(15), 0.35,
                                 _wind(60.0, crest=4500.0), table=table)
    assert adaptation.reading.held_everything_tried
    assert adaptation.ceiling is None and not adaptation.adapted


def test_a_row_without_a_tried_range_reads_the_ladder_top():
    rows = (tc.MapRow(3000.0, 4500.0, 0.4, 4, (5.0, 5.0, 5.0)),
            tc.MapRow(3000.0, 4500.0, 0.4, 6, (5.0, 5.0, 5.0)))
    table = tc.StableStepMap(winds=(20.0, 60.0, 100.0), ladder=(5.0, 4.0),
                             seconds=1800.0, rows=rows)
    reading = tc.read_map(3000.0, 4500.0, 0.35, 60.0, 4, table)
    assert reading.top_per_km == 5.0 and reading.held_everything_tried
    run = _Run(use_adaptive_time_step=True, max_time_step=41)
    adaptation = tc.derive_clock(1, run, Fraction(15), 0.35,
                                 _wind(60.0, crest=4500.0), table=table)
    assert adaptation.ceiling is None


def test_the_committed_probe_builds_the_maps_own_3km_ridges():
    """tools/terrain_clock_probe.py rebuilds the map: its ridge on every
    3 km row it extended reads the row's own grid slope and etac."""
    import json

    from tools.terrain_clock_probe import Ridge, geometry

    document = json.loads(tc.MAP_PATH.read_text(encoding="utf-8"))
    rows = [row for row in document["rows"] if row["dx_m"] == 3000.0
            and row["crest_m"] <= 4500.0 and row["ridge_slope"] <= 0.4
            and row["sound_steps"] == 4]
    assert len(rows) == 21
    for row in rows:
        shape = geometry(Ridge(3000.0, row["crest_m"], row["ridge_slope"]))
        assert shape["slope"] == row["slope"], row
        assert shape["etac"] == row["etac"], row


def test_the_3km_rows_are_measured_from_20_to_40_s():
    """A73: the 3 km rows of crests to 4.5 km and slopes to 0.4 were tried
    from 40 s down to 20 s at 20 to 90 m/s; 238 of their 336 cells held a
    step in that range, and the 100 m/s column keeps the ladder's range."""
    table = tc.measured_map()
    rows = [row for row in table.rows if row.tried is not None]
    assert {(row.dx_m, row.crest_m) for row in rows} == {
        (3000.0, 1500.0), (3000.0, 3000.0), (3000.0, 4500.0)}
    assert len(rows) == 42
    longer = 0
    for row in rows:
        assert row.tried[:8] == pytest.approx((40.0 / 3.0,) * 8)
        assert row.tried[8] == table.top
        for entry, tried in zip(row.stable[:8], row.tried[:8]):
            assert entry is None or entry <= tried + 1e-9
            if entry is not None and entry > table.top:
                assert 20.0 / 3.0 - 1e-9 <= entry
                longer += 1
    assert longer == 238
    assert table.measured_range()[3000.0] == pytest.approx(40.0)
    assert table.measured_range()[1000.0] == 5.0


def test_a_3km_conus_adaptive_run_is_capped_where_20_s_stopped():
    """The 3 km CONUS domain on HRRR's own grid: highest ground 3932 m,
    steepest slope 0.356, with a 12 s time_step.  Under the adaptive
    clock's default 24 s max_time_step, a 20 m/s crest-level wind leaves
    it alone; at 40 m/s the step is capped at 15 s, because 20 s stopped
    there on four substeps (before, the 15 s entry read as "holds every
    step"); at 50 m/s the cap is the 13.5 s the map already held.  At
    30 m/s it was capped at 15 s from those fixed-step stops too; A139 ran
    the rows it reads on the adaptive clock, which held a longest step of
    45 s on every one of them at 20 and 30 m/s, so at 30 m/s it keeps its
    24 s.  At 40 m/s the adaptive clock saw 45 s stop on the gentlest and
    the steepest of them, so the fixed-step cap stands there.
    The 15 s time_step the domain wizard writes picks another pair at
    50 m/s, pinned in the test below."""
    run = _Run(use_adaptive_time_step=True)
    expected = {20.0: None, 30.0: None, 40.0: Fraction(15),
                50.0: Fraction(27, 2)}
    for wind, ceiling in expected.items():
        adaptation = tc.derive_clock(1, run, Fraction(12), 0.356,
                                     _wind(wind, crest=3932.0))
        assert adaptation.ceiling == ceiling, wind
        assert adaptation.time_step_sound == 4
    receipt = adaptation.receipt()
    assert receipt["map_rows"]["longest_step_tried_s"] == pytest.approx(40.0)


def test_a_3km_conus_adaptive_run_on_the_wizard_s_15_s_step():
    """The same CONUS ground under the configuration `gpuwm domain` writes
    at 3 km by default (``--clock auto``: the adaptive clock on a 15 s
    time_step, max_time_step -1), read through the experiment the run
    writes.  At 30 and 40 m/s the cap is 15 s on four substeps, as from
    12 s.  At 50 and 60 m/s four substeps hold 13.5 s, which would halve
    the 15 s step, and six hold 15 s, so the clock takes at least six
    substeps with its step capped at 15 s; before the 3 km rows were
    extended the six substep 15 s entry read as holding every longer
    step, and the clock kept its default 24 s on six.  So the 13.5 s a
    12 s time_step gets at 50 m/s is not this run's cap, before or now.
    From 70 m/s the step is halved to 7.5 s and the cap is 12 s.  At
    30 m/s the cap was 15 s too, until A139's adaptive-clock entries: that
    clock held 45 s on every row read there, so the run keeps -1."""
    exp = _wizard_experiment([(60, 60)], (), 3000.0, clock="auto")
    assert exp.domains[0].run.use_adaptive_time_step
    assert exp.dt_exact(1) == Fraction(15)
    assert exp.domains[0].run.max_time_step == -1
    # wind: (cap, substeps, division)
    expected = {20.0: (None, 4, 1), 30.0: (None, 4, 1),
                40.0: (Fraction(15), 4, 1), 50.0: (Fraction(15), 6, 1),
                60.0: (Fraction(15), 6, 1), 70.0: (Fraction(12), 4, 2)}
    for wind, (ceiling, sound, division) in expected.items():
        adapted, adaptations = tc.adapt_experiment_clock(
            exp, {1: 0.356}, {1: _wind(wind, crest=3932.0)})
        adaptation = adaptations[0]
        assert (adaptation.ceiling, adaptation.time_step_sound,
                adaptation.division) == (ceiling, sound, division), wind
        run = adapted.domains[0].run
        if ceiling is None:
            assert run.max_time_step == -1, wind
        else:
            assert Fraction(run.max_time_step,
                            run.max_time_step_den or 1) == ceiling, wind
        assert run.min_time_step_sound == (6 if sound == 6 else 0), wind
        assert adapted.dt_exact(1) == Fraction(15, division), wind


def test_the_clock_record_states_the_measured_range():
    record = tc.clock_receipt([])
    tried = record["map"]["longest_step_tried_s"]
    assert tried["3000"] == pytest.approx(40.0)
    assert tried["1000"] == 5.0 and tried["12000"] == 60.0


# ---------------------------------------------------------------------------
# A stop is read cell by cell.
# ---------------------------------------------------------------------------


def _cap(adaptation):
    """The adaptive step's cap in seconds, infinite where none is written."""
    return (float("inf") if adaptation.ceiling is None
            else float(adaptation.ceiling))


def test_a_steeper_row_measured_less_far_keeps_the_stop_a_gentler_one_saw():
    """The defect: a reading took its longest step tried as the least over
    every row it read, so a steeper row tried only to 15 s (which held
    it) read as "held everything tried" and hid the gentler row beside it
    that saw 27 s stop, and the steeper ground ran uncapped where the
    gentler ground was capped.  A stop is now read cell by cell: the
    shortest entry of a cell that saw a longer step stop caps the step."""
    rows = tuple(row for count in (4, 6) for row in (
        tc.MapRow(3000.0, 4500.0, 0.3, count, (8.0,) * 3, (13.0,) * 3),
        tc.MapRow(3000.0, 4500.0, 0.4, count, (5.0,) * 3)))
    table = tc.StableStepMap(winds=(20.0, 60.0, 100.0), ladder=(5.0, 4.0),
                             seconds=1800.0, rows=rows)
    run = _Run(use_adaptive_time_step=True, max_time_step=41)
    reading = tc.read_map(3000.0, 4500.0, 0.35, 60.0, 4, table)
    assert reading.per_km == 5.0 and reading.top_per_km == 5.0
    assert reading.stopped_per_km == 8.0
    assert not reading.held_everything_tried
    steep = tc.derive_clock(1, run, Fraction(15), 0.35,
                            _wind(60.0, crest=4500.0), table=table)
    gentle = tc.derive_clock(1, run, Fraction(15), 0.3,
                             _wind(60.0, crest=4500.0), table=table)
    assert gentle.ceiling == Fraction(24) and steep.ceiling == Fraction(24)
    assert steep.receipt()["map_rows"][
        "shortest_entry_under_a_stop_s"] == 24.0
    # A step no cell read saw stop is not divided: 21 s is longer than the
    # steeper row was tried, shorter than any stop seen.
    fixed = tc.derive_clock(1, _Run(), Fraction(21), 0.35,
                            _wind(60.0, crest=4500.0), table=table)
    assert not fixed.adapted


@pytest.mark.parametrize("slope, ceiling, source", [
    (0.40, None, "adaptive held"), (0.42, Fraction(15), "map"),
    (0.46, Fraction(15), "map")])
def test_steep_3km_ground_past_the_extended_rows_is_capped_like_conus(
        slope, ceiling, source):
    """Past the extended rows' slope (0.36), under a 3.9 km crest at
    30 m/s: the tip left 0.40, 0.42 and 0.46 uncapped while 0.30 was
    capped at 20 s and 0.356 at 15 s.  The 15 s the 0.356 row held
    below the 20 s it saw stop now caps them too, on the fixed rows.
    A139b ran the steeper 3 km rows under the 4.5 km crest on the
    adaptive clock: the 0.4105 row held 15 s/km at 30 m/s, so 0.40 reads
    as the CONUS 0.356 does there (A139: no cap under the 24 s default),
    and the 0.4655 row that 0.42 and 0.46 read stopped above 5 s/km, so
    they keep 15 s."""
    run = _Run(use_adaptive_time_step=True)
    adaptation = tc.derive_clock(1, run, Fraction(12), slope,
                                 _wind(30.0, crest=3932.0))
    assert adaptation.limit_per_km == pytest.approx(5.0)
    assert adaptation.ceiling == ceiling
    assert adaptation.cap_source == source
    assert adaptation.time_step_sound == 4


def test_the_100_ms_column_keeps_the_stop_seen_at_90_ms():
    """The 100 m/s column was not extended: under a 1.5 km crest of slope
    0.2 the tip capped 90 m/s at 20 s and left 95 and 100 m/s uncapped.
    The 20 s held under the 22 s seen to stop at 90 m/s caps every
    stronger wind."""
    run = _Run(use_adaptive_time_step=True)
    for wind in (90.0, 95.0, 100.0):
        adaptation = tc.derive_clock(1, run, Fraction(12), 0.2,
                                     _wind(wind, crest=1500.0))
        assert adaptation.ceiling == Fraction(20), wind


def test_a_spacing_between_mapped_ones_keeps_the_3km_rows_stop():
    """A 2.5 km domain on the CONUS ground reads the 2 and 3 km rows.  At
    30 and 40 m/s the tip left it uncapped at the adaptive clock's 20 s,
    though its 3 km rows saw 20 s stop over 15 s held: it is capped at
    what that is per km, 12.5 s.  At 30 m/s A139's adaptive-clock entries
    lift that: the adaptive clock held 15 s/km, 30 and 45 s, on every 2
    and 3 km row read there, 37.5 s here, past the clock's own 20 s, so
    no cap is written; at 40 m/s it saw that stop on some of them."""
    run = _Run(dx=2500.0, dy=2500.0, use_adaptive_time_step=True)
    for wind, ceiling in ((30.0, None), (40.0, Fraction(25, 2))):
        adaptation = tc.derive_clock(1, run, Fraction(10), 0.356,
                                     _wind(wind, crest=3932.0))
        assert adaptation.reading.dx_rows == (2000.0, 3000.0)
        assert adaptation.ceiling == ceiling, wind
    assert adaptation.cap_source == "map"
    lifted = tc.derive_clock(1, run, Fraction(10), 0.356,
                             _wind(30.0, crest=3932.0))
    assert lifted.limit_per_km == pytest.approx(5.0)
    assert lifted.cap_source == "adaptive held"
    assert lifted.cap_per_km == 15.0


_SLOPES = (0.05, 0.1, 0.2, 0.25, 0.3, 0.33, 0.356, 0.37, 0.4, 0.42, 0.46,
           0.5)
_WINDS = (20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 95.0,
          100.0)


def _neighbours(i, j):
    """The gentler slope and the weaker wind next to grid cell (i, j)."""
    found = []
    if i:
        found.append((_SLOPES[i - 1], _WINDS[j]))
    if j:
        found.append((_SLOPES[i], _WINDS[j - 1]))
    return found


@pytest.mark.parametrize("dx", [1000.0, 2500.0, 3000.0])
@pytest.mark.parametrize("crest", [1500.0, 3932.0, 5500.0])
def test_the_cap_never_rises_with_slope_or_wind_at_the_count_it_runs(dx,
                                                                     crest):
    """Adding a row or a wind to a reading can only lower its answer: at
    the substep count a domain runs, a steeper domain's cap is never above
    a gentler one's at the same crest and wind, and a stronger wind's
    never above a weaker one's.  (A domain moved to six substeps runs a
    more stable pair, whose own cap can be longer.)  The tip broke this
    176 times over a sweep like this one at 3 km.  It holds on ground the
    map holds a step on.  Where the steeper of two slopes holds no step at
    its wind, it runs the most stable pair measured with the "may still
    stop" caution, uncapped where no pair was measured, and that can sit
    above a gentler neighbour's cap: over slopes to 1.0 and crests to
    8 km such pairs exist at 1, 1.5 and 2 km, as many as before the 3 km
    rows were extended."""
    table = tc.measured_map()
    for per_km in (4.0, 6.0):
        run = _Run(dx=dx, dy=dx, use_adaptive_time_step=True)
        dt = Fraction(int(round(per_km * dx)), 1000)
        grid = {}
        for slope in _SLOPES:
            for wind in _WINDS:
                adaptation = tc.derive_clock(1, run, dt, slope,
                                             _wind(wind, crest=crest),
                                             table=table)
                grid[slope, wind] = (_cap(adaptation),
                                     adaptation.time_step_sound)
        for i, slope in enumerate(_SLOPES):
            for j, wind in enumerate(_WINDS):
                cap, count = grid[slope, wind]
                for other in _neighbours(i, j):
                    other_cap, other_count = grid[other]
                    if count == other_count:
                        assert cap <= other_cap, (per_km, slope, wind,
                                                  other)
    for count in (4, 6):
        for i, slope in enumerate(_SLOPES):
            for j, wind in enumerate(_WINDS):
                here = tc.read_map(dx, crest, slope, wind, count, table)
                for other in _neighbours(i, j):
                    there = tc.read_map(dx, crest, *other, count, table)
                    if there.stopped_per_km is not None:
                        assert here.stopped_per_km is not None
                        assert here.stopped_per_km <= there.stopped_per_km


def test_a_fixed_3km_step_above_15_s_reads_the_stops_too():
    """A fixed step is read on the same map.  A 3 km step of 15 s or less
    (the domain wizard writes 15 s) reads as before the rows were
    extended; one above 15 s over the CONUS ground now meets the stops
    seen at 20 s.  Fixed 18 s: four substeps held it under 30 m/s before,
    six now (four saw 20 s stop over 15 s held, six held 20 s); from
    40 m/s it is halved to 9 s (six saw 20 s stop over 15 s held too).
    Fixed 20 s is halved to 10 s from 40 m/s."""

    def fixed(seconds, wind):
        return tc.derive_clock(1, _Run(), Fraction(seconds), 0.356,
                               _wind(wind, crest=3932.0))

    for wind in (20.0, 30.0, 40.0):
        assert not fixed(15, wind).adapted, wind
    assert not fixed(18, 20.0).adapted
    at_30 = fixed(18, 30.0)
    assert (at_30.division, at_30.time_step_sound) == (1, 6)
    at_40 = fixed(18, 40.0)
    assert (at_40.division, at_40.dt, at_40.time_step_sound) == (
        2, Fraction(9), 4)
    at_50 = fixed(18, 50.0)
    assert (at_50.division, at_50.time_step_sound) == (2, 4)
    for wind in (40.0, 50.0):
        assert fixed(20, wind).dt == Fraction(10), wind


def test_past_the_strongest_wind_held_the_adaptive_step_is_capped_too():
    """Where no measured step holds at the domain's wind, the domain runs
    the most stable pair measured at a weaker one.  A cell that held
    nothing saw every step stop, so the adaptive step is capped at that
    pair's step as well (under an 8 km crest at 3 km, 60 m/s holds no
    step and 50 m/s held 15 s on six substeps); the tip left it at the
    adaptive clock's 24 s when that pair was the ladder's top."""
    run = _Run(use_adaptive_time_step=True)
    adaptation = tc.derive_clock(1, run, Fraction(12), 0.1,
                                 _wind(60.0, crest=8000.0))
    assert adaptation.status == "BEYOND_MEASURED"
    assert adaptation.time_step_sound == 6
    assert adaptation.ceiling == Fraction(15)


# ---------------------------------------------------------------------------
# A cap only ever shortens the step, and taller ground keeps the stops a
# lower crest saw.
# ---------------------------------------------------------------------------


def _effective_upper(run):
    from gpuwm.core.adaptive_clock import wrf_default_clamps

    upper = tc._adaptive_upper(run)
    return (Fraction(wrf_default_clamps(run.dx, run.dy)[1]) if upper is None
            else upper)


@pytest.mark.parametrize("crest, slope, wind", [
    (1500.0, 0.30, 20.0), (1500.0, 0.30, 30.0), (2000.0, 0.45, 20.0),
    (3932.0, 0.60, 20.0)])
def test_a_cap_never_raises_the_adaptive_clock_s_default_max_step(
        crest, slope, wind):
    """The defect: ground steeper than every row at its crest reads rows
    that held 15 s and were never tried longer beside gentler rows that saw
    27 to 33 s stop, and that stop was written as max_time_step, 27 to 33 s
    on a 3 km run whose max_time_step is -1 (WRF's 24 s).  A cap at or
    above the clock's own longest step is not written: the run keeps -1,
    as before the 3 km rows were extended."""
    exp = _wizard_experiment([(60, 60)], (), 3000.0)
    root = exp.domains[0]
    exp = replace(exp, domains=(replace(root, run=replace(
        root.run, use_adaptive_time_step=True)),))
    assert exp.domains[0].run.max_time_step == -1
    adapted, adaptations = tc.adapt_experiment_clock(
        exp, {1: slope}, {1: _wind(wind, crest=crest)})
    assert adaptations[0].ceiling is None
    assert adaptations[0].status == "AS_CONFIGURED"
    assert adapted.domains[0].run.max_time_step == -1
    # Under an explicit max_time_step longer than the stop, the stop caps.
    run = _Run(use_adaptive_time_step=True, max_time_step=41)
    capped = tc.derive_clock(1, run, Fraction(12), slope,
                             _wind(wind, crest=crest))
    assert capped.ceiling is not None and capped.ceiling < 41


def test_a_six_substep_pick_never_raises_the_default_max_step():
    """The other road to the same defect: a 3 km domain with an 18 s first
    step over a 2 km crest of slope 0.05 at 50 m/s moves to six substeps,
    whose rows held 27 s, and 27 s was written over the default 24 s."""
    run = _Run(use_adaptive_time_step=True)
    adaptation = tc.derive_clock(1, run, Fraction(18), 0.05,
                                 _wind(50.0, crest=2000.0))
    assert adaptation.time_step_sound == 6
    assert adaptation.ceiling is None
    assert adaptation.status == "ADAPTED"


@pytest.mark.parametrize("max_step", [-1, 20, 41])
def test_no_derived_cap_is_ever_at_or_above_the_run_s_longest_step(max_step):
    """Over spacings of 1 to 4 km, crests of 0.5 to 9.5 km, slopes of 0.05
    to 0.9, winds of 20 to 120 m/s and first steps of 4 to 6 s/km, a cap
    written is always below the step the adaptive clock would otherwise
    take.  Over 108108 such cases the tip wrote 2832 above the default."""
    table = tc.measured_map()
    for dx in (1000.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0):
        run = _Run(dx=dx, dy=dx, use_adaptive_time_step=True,
                   max_time_step=(-1 if max_step < 0 else
                                  int(round(max_step * dx / 3000.0))))
        upper = _effective_upper(run)
        for per_km in (4.0, 5.0, 6.0):
            dt = Fraction(int(round(per_km * dx)), 1000)
            for crest in (500.0, 1500.0, 2500.0, 3932.0, 4600.0, 6000.0,
                          8000.0, 9500.0):
                for slope in (0.05, 0.2, 0.3, 0.45, 0.6, 0.9):
                    for wind in (20.0, 30.0, 50.0, 80.0, 120.0):
                        adaptation = tc.derive_clock(
                            1, run, dt, slope, _wind(wind, crest=crest),
                            table=table)
                        if adaptation.ceiling is not None:
                            assert adaptation.ceiling < upper, (
                                dx, per_km, crest, slope, wind)


@pytest.mark.parametrize("slope, wind, crests", [
    (0.356, 30.0, (4600.0, 5000.0, 5500.0, 6000.0, 6456.0)),
    (0.30, 40.0, (3932.0, 4600.0, 5000.0, 5500.0, 6000.0)),
    (0.40, 30.0, (4600.0, 5000.0, 5500.0))])
def test_a_crest_above_the_extended_rows_keeps_the_stop_a_lower_one_saw(
        slope, wind, crests):
    """The 3 km rows of crests above 4.5 km were tried only to 15 s, so at
    the tip a domain whose crest read 5.5 km or more held every step tried
    and ran the default 24 s where the same slope and wind under a 4.4 km
    crest was capped at 15 s: taller ground took the longer step.  A
    longer step seen to stop under a lower crest now counts past the
    longest step the taller crest was tried at.  (Slope 0.356 at 30 m/s
    under the 3.9 and 4.4 km crests is no longer capped: A139 ran their
    4.5 km rows on the adaptive clock, which held 45 s on every one,
    pinned in test_the_adaptive_clock_s_own_entries_lift_a_fixed_step_cap.
    Slope 0.40 at 30 m/s under the 4.4 km crest likewise since A139b ran
    the 4.5 km row of grid slope 0.4105 that way, pinned below.  The
    taller crests were not run that way and keep 15 s.)"""
    run = _Run(use_adaptive_time_step=True)
    for crest in crests:
        adaptation = tc.derive_clock(1, run, Fraction(12), slope,
                                     _wind(wind, crest=crest))
        assert adaptation.ceiling == Fraction(15), crest
        assert adaptation.time_step_sound == 4
    if slope == 0.40:
        lifted = tc.derive_clock(1, run, Fraction(12), slope,
                                 _wind(wind, crest=4400.0))
        assert lifted.ceiling is None
        assert lifted.cap_source == "adaptive held"
    tall = tc.read_map(3000.0, 5500.0, slope, wind, 4)
    assert tall.crest_row == 5500.0 and tall.top_per_km == 5.0
    assert tall.stopped_under_a_lower_crest
    assert tall.stopped_crest_m == 4500.0


def test_a_fixed_step_under_a_crest_above_the_extended_rows_reads_it_too():
    """A fixed 18 s over the CONUS slope at 30 m/s runs on six substeps
    under a 3.9 km crest; under a 4.6 km crest it now does the same
    instead of staying on four."""
    for crest in (3932.0, 4600.0, 5500.0):
        adaptation = tc.derive_clock(1, _Run(), Fraction(18), 0.356,
                                     _wind(30.0, crest=crest))
        assert (adaptation.division, adaptation.time_step_sound) == (1, 6)


def test_a_lower_crest_s_stop_never_undercuts_what_this_crest_was_tried_at():
    """Below the longest step the domain's own crest was tried at, its own
    rows decide: a lower ridge that held 12 s under a longer step that
    stopped gives this crest, which held every step tried to 15 s, a stop
    at 15 s and not under it; a lower ridge tried no further than this
    crest adds nothing."""
    rows = []
    for count in (4, 6):
        rows.append(tc.MapRow(3000.0, 3000.0, 0.3, count, (4.0,) * 3,
                              (13.0,) * 3))
        rows.append(tc.MapRow(3000.0, 5500.0, 0.3, count, (5.0,) * 3))
    table = tc.StableStepMap(winds=(20.0, 60.0, 100.0), ladder=(5.0, 4.0),
                             seconds=1800.0, rows=tuple(rows))
    reading = tc.read_map(3000.0, 5500.0, 0.3, 60.0, 4, table)
    assert reading.per_km == 5.0
    assert reading.stopped_per_km == 5.0
    assert reading.stopped_under_a_lower_crest
    fixed = tc.derive_clock(1, _Run(), Fraction(15), 0.3,
                            _wind(60.0, crest=5500.0), table=table)
    assert not fixed.adapted
    untried = tc.StableStepMap(
        winds=table.winds, ladder=table.ladder, seconds=table.seconds,
        rows=tuple(replace(row, tried=None) for row in rows))
    alone = tc.read_map(3000.0, 5500.0, 0.3, 60.0, 4, untried)
    assert alone.held_everything_tried and alone.stopped_per_km is None


def _before_the_extension(table):
    """The map as it was before the 3 km rows were tried past the ladder:
    every entry at most the ladder's top, no tried range."""
    rows = tuple(replace(row, stable=tuple(
        None if value is None else min(value, table.top)
        for value in row.stable), tried=None) for row in table.rows)
    return tc.StableStepMap(winds=table.winds, ladder=table.ladder,
                            rows=rows, seconds=table.seconds)


_CRESTS = (1000.0, 1500.0, 2000.0, 3000.0, 3932.0, 4400.0, 4600.0,
           5000.0, 5500.0, 6000.0, 6456.0, 7000.0, 8000.0, 8850.0)


@pytest.mark.parametrize("dx", [2500.0, 3000.0, 3500.0])
def test_the_extended_rows_add_no_cap_that_rises_with_crest(dx):
    """The map's own rows are not monotone in crest (a lower ridge of the
    same slope is a narrower one, and its entries were measured as they
    fell), so a taller crest can hold a longer step than a lower one did.
    What the 3 km rows tried past the ladder must not add is a new such
    rise: at every slope and wind, where the cap at a taller crest is
    above the cap at the next lower one at the same substep count and
    division, the same pair rises on the map as it was before the
    extension.  Against cc3cb0ad6 over a wider sweep of spacings,
    crests, slopes, winds and first steps, the tip added 257."""
    table = tc.measured_map()
    before = _before_the_extension(table)
    run = _Run(dx=dx, dy=dx, use_adaptive_time_step=True)
    upper = _effective_upper(run)

    def cap(which, per_km, crest, slope, wind):
        dt = Fraction(int(round(per_km * dx)), 1000)
        adaptation = tc.derive_clock(1, run, dt, slope,
                                     _wind(wind, crest=crest), table=which)
        return (adaptation.time_step_sound, adaptation.division,
                upper if adaptation.ceiling is None else adaptation.ceiling)

    for per_km in (4.0, 6.0):
        for slope in _SLOPES:
            for wind in _WINDS:
                now = [cap(table, per_km, crest, slope, wind)
                       for crest in _CRESTS]
                was = [cap(before, per_km, crest, slope, wind)
                       for crest in _CRESTS]
                for k in range(1, len(_CRESTS)):
                    low, high = now[k - 1], now[k]
                    if low[:2] == high[:2] and high[2] > low[2]:
                        assert was[k][2] > was[k - 1][2], (
                            per_km, slope, wind, _CRESTS[k - 1],
                            _CRESTS[k], low, high)


def test_a_limit_past_what_the_ground_held_is_said_as_a_stop_seen():
    """Where the step a domain may run is longer than its ground held,
    because that ground was never tried longer, the line and the record
    keep the two apart: the ground held 15 s, and a step longer than 33 s
    stopped on the gentler rows read with it (or under a lower crest)."""
    run = _Run(use_adaptive_time_step=True, max_time_step=41)
    beyond = tc.derive_clock(1, run, Fraction(12), 0.30,
                             _wind(20.0, crest=1500.0))
    assert beyond.status == "BEYOND_MEASURED"
    assert beyond.ceiling == Fraction(33)
    line = beyond.beyond_sentence()
    # The step it runs comes first; the shorter range the ground was
    # tried to after it (REVIEW.md gap 3: never "holds 12.5 s" then 15 s).
    assert "the measured map lets d01 run steps up to 33 s there" in line
    assert ("(a step longer than 33 s stopped on the gentler rows or weaker "
            "winds read with it; no cell read saw 33 s or a shorter step "
            "stop, though some of that ground was tried no longer than "
            "15 s)" in line)
    assert "holds steps up to 15 s" not in line
    receipt = beyond.receipt()
    assert receipt["held_s_per_km"] == 5.0
    assert receipt["limit_s_per_km"] == 11.0
    tall = tc.derive_clock(1, run, Fraction(12), 0.1,
                           _wind(20.0, crest=5500.0))
    assert tall.status == "ADAPTED" and tall.ceiling == Fraction(33)
    assert ("a step longer than 33 s stopped under a lower 4500 m crest at "
            "this spacing" in tall.sentence())
    rows = tall.receipt()["map_rows"]
    assert rows["stop_crest_m"] == 4500.0 and rows["stop_under_a_lower_crest"]
    # Where the limit is what the ground held, the line says only that.
    # (At 40 m/s: at 30 the adaptive clock's own entries lift the cap.)
    conus = tc.derive_clock(1, _Run(use_adaptive_time_step=True),
                            Fraction(12), 0.356, _wind(40.0, crest=3932.0))
    assert conus.ceiling == Fraction(15)
    assert "tried no longer than" not in conus.sentence()
    assert "holds steps up to 15 s there with 4 substeps, so" in (
        conus.sentence())


# ---------------------------------------------------------------------------
# The adaptive clock's own entries (A139).
# ---------------------------------------------------------------------------


def _adaptive_rows():
    return [row for row in tc.measured_map().rows if row.adaptive is not None]


def test_the_2_and_3km_rows_carry_the_adaptive_clock_s_own_entries():
    """A139: the 2 and 3 km rows of crests 1.5 to 4.5 km and ridge slopes
    0.1 to 0.4 were run on the production adaptive clock at max_time_step
    15 s/km down to 5 s/km under 20 to 60 m/s, three hours each at both
    CFL target pairs.  At 20 and 30 m/s every one held 15 s/km, the
    longest tried: 30 s at 2 km and 45 s at 3 km.  A139b ran the rows of
    ridge slopes 0.45 and 0.5 under the 4.5 km crest the same way (41 rows
    became 45): every one held 15 s/km at 20 m/s, and all but the 3 km row
    of ridge slope 0.5 (grid 0.4655), which held 5 s/km, at 30 m/s."""
    table = tc.measured_map()
    measured = table.adaptive
    assert measured.targets == ((1.2, 0.84), (1.4, 0.98))
    assert measured.max_step_increase_pct == 5
    assert measured.seconds == 10800.0
    assert measured.ladder[0] == 15.0 and measured.ladder[-1] == 5.0
    rows = _adaptive_rows()
    assert len(rows) == 45
    assert {(row.dx_m, row.crest_m) for row in rows} == {
        (dx, crest) for dx in (2000.0, 3000.0)
        for crest in (1500.0, 3000.0, 4500.0)}
    assert all(row.sound_steps == 4 for row in rows)
    for row in rows:
        expected = ((15.0, 5.0) if (row.dx_m, row.crest_m, row.slope)
                    == (3000.0, 4500.0, 0.4655) else (15.0, 15.0))
        assert row.adaptive[:2] == expected, row
        assert row.adaptive_tried[:2] == (15.0, 15.0), row
        # 70 m/s and up were not run.
        assert set(row.adaptive_tried[5:]) == {None}, row


def test_the_committed_probe_builds_the_rows_the_adaptive_clock_ran_on():
    """Every row carrying adaptive entries is the probe's own ridge: its
    grid slope and etac, row key for row key, at 2 km as at 3 km."""
    import json

    from tools.terrain_clock_probe import Ridge, geometry

    document = json.loads(tc.MAP_PATH.read_text(encoding="utf-8"))
    rows = [row for row in document["rows"] if "adaptive_s_per_km" in row]
    assert len(rows) == 45
    for row in rows:
        shape = geometry(Ridge(row["dx_m"], row["crest_m"],
                               row["ridge_slope"]))
        assert shape["slope"] == row["slope"], row
        assert shape["etac"] == row["etac"], row


def test_the_adaptive_clock_s_own_entries_lift_a_fixed_step_cap():
    """The defect: a 2.25 km domain under a 3.6 km crest of slope 0.30 at
    24 m/s, on the adaptive clock with a 10 s first step and a 30 s
    max_time_step, was capped at 15 s, the per-km reading of a FIXED 22 s
    seen to stop on four substeps on its 3 km rows, though the adaptive
    clock derives eight substeps at such a step and shortens it on CFL.
    Run on the adaptive clock, every 2 and 3 km row it reads held the
    longest step tried there (15 s/km, 33.75 s here), so it runs as
    configured.  The fixed-step reading is unchanged."""
    run = _Run(dx=2250.0, dy=2250.0, use_adaptive_time_step=True,
               max_time_step=30)
    adaptation = tc.derive_clock(1, run, Fraction(10), 0.30,
                                 _wind(24.0, crest=3595.0))
    reading = adaptation.reading
    assert reading.dx_rows == (2000.0, 3000.0)
    assert reading.crest_row == 4500.0
    assert reading.stopped_per_km == pytest.approx(20.0 / 3.0)
    assert reading.adaptive_measured
    assert reading.adaptive_held_everything_tried
    assert reading.adaptive_per_km == 15.0
    assert adaptation.ceiling is None and not adaptation.adapted
    assert adaptation.status == "AS_CONFIGURED"
    assert adaptation.cap_source == "adaptive held"
    rows = adaptation.receipt()["map_rows"]["adaptive_clock"]
    assert rows["every_cell_measured"]
    assert rows["longest_step_held_s"] == pytest.approx(33.75)
    assert rows["shortest_entry_under_a_stop_s"] is None
    # A longer max_time_step is capped at the longest step held there.
    longer = tc.derive_clock(1, replace(run, max_time_step=60),
                             Fraction(10), 0.30, _wind(24.0, crest=3595.0))
    assert longer.ceiling == Fraction(3375, 100)
    assert longer.receipt()["max_time_step_from"] == "adaptive held"
    assert ("and on the adaptive clock every cell read held a longest step "
            "of 33.75 s, the longest tried there, so d01 runs an adaptive "
            "step capped at 33.75 s" in longer.sentence())
    # The 3 km CONUS ground at 30 m/s, under crests read on the 4.5 km rows.
    for crest in (3932.0, 4400.0):
        conus = tc.derive_clock(1, _Run(use_adaptive_time_step=True),
                                Fraction(12), 0.356,
                                _wind(30.0, crest=crest))
        assert conus.ceiling is None and conus.cap_source == "adaptive held"
    # A fixed step reads the fixed-step map, as before.
    fixed = tc.derive_clock(1, replace(run, use_adaptive_time_step=False),
                            Fraction(18), 0.30, _wind(24.0, crest=3595.0))
    assert fixed.cap_source == "map" and fixed.adapted
    assert (fixed.division, fixed.time_step_sound) == (2, 4)


def _faster(run, **settings):
    return SimpleNamespace(**{**vars(run), **settings})


def test_a_clock_faster_than_the_one_measured_keeps_the_fixed_step_cap():
    """The entries held at CFL targets up to 1.4 / 0.98 with 5 percent
    growth.  A clock with a higher target or faster growth was never run
    on that ground, so its entries do not lift its cap."""
    base = _Run(dx=2250.0, dy=2250.0, use_adaptive_time_step=True,
                max_time_step=30)
    for faster in (_faster(base, target_cfl=1.5),
                   _faster(base, max_step_increase_pct=51)):
        adaptation = tc.derive_clock(1, faster, Fraction(10), 0.30,
                                     _wind(24.0, crest=3595.0))
        assert adaptation.ceiling == Fraction(15)
        assert adaptation.cap_source == "map"
    measured = _faster(base, target_cfl=1.4, target_hcfl=0.98,
                       max_step_increase_pct=5)
    assert tc.derive_clock(1, measured, Fraction(10), 0.30,
                           _wind(24.0, crest=3595.0)).ceiling is None


def _steep_2250(**settings):
    """A 2.25 km adaptive domain with a 10 s first step and a 30 s
    max_time_step on the 1.4 / 0.98 targets with 5 percent growth."""
    return _faster(_Run(dx=2250.0, dy=2250.0, use_adaptive_time_step=True,
                        max_time_step=30),
                   **{"target_cfl": 1.4, "target_hcfl": 0.98,
                      "max_step_increase_pct": 5, **settings})


@pytest.mark.parametrize("wind", [20.0, 24.0, 30.0])
def test_a_2250_m_domain_of_slope_037_under_a_3841_m_crest_runs_as_set(wind):
    """A139b, the defect: a 2.25 km adaptive domain of slope 0.37 under a
    3841 m crest (first step 10 s, max_time_step 30 s) was capped at
    20.25 s at 20 m/s and 11.25 s at 24 and 30 m/s, from fixed-step stops,
    because one cell its reading takes, the 3 km row of grid slope 0.4105
    under the 4.5 km crest, had not been run on the adaptive clock.  Run
    so, it held 15 s/km at 20 and 30 m/s, so every cell read held the
    longest step tried there (33.75 s here, above its 30 s) and the domain
    runs as configured.  The 2.25 km ridge under a 3841 m crest at grid
    slopes 0.366 and 0.379 held max_time_step 24 and 30 s for three hours
    at 20 to 40 m/s on both target pairs (24 of 24)."""
    adaptation = tc.derive_clock(1, _steep_2250(), Fraction(10), 0.37,
                                 _wind(wind, crest=3841.0))
    reading = adaptation.reading
    assert reading.dx_rows == (2000.0, 3000.0)
    assert reading.crest_row == 4500.0 and reading.slope_row == 0.4105
    assert reading.beyond == ()
    assert reading.adaptive_measured
    assert reading.adaptive_held_everything_tried
    assert reading.adaptive_per_km == 15.0
    assert adaptation.cap_source == "adaptive held"
    assert adaptation.ceiling is None and not adaptation.adapted
    assert adaptation.status == "AS_CONFIGURED"
    assert (adaptation.division, adaptation.time_step_sound) == (1, 4)
    record = adaptation.receipt()["map_rows"]["adaptive_clock"]
    assert record["longest_step_held_s"] == pytest.approx(33.75)
    # A longer max_time_step is capped at the longest step held there.
    longer = tc.derive_clock(1, _steep_2250(max_time_step=60), Fraction(10),
                             0.37, _wind(wind, crest=3841.0))
    assert longer.ceiling == Fraction(3375, 100)
    assert longer.cap_source == "adaptive held"


def test_at_40_ms_that_domain_keeps_its_cap_and_may_still_stop():
    """At 40 m/s the 3 km row of grid slope 0.4105 held no max_time_step
    on the adaptive clock, down to 5 s/km: the domain keeps the 10.12 s
    cap its fixed rows give it and says it may still stop."""
    adaptation = tc.derive_clock(1, _steep_2250(), Fraction(10), 0.37,
                                 _wind(40.0, crest=3841.0))
    assert adaptation.ceiling == Fraction(1012, 100)
    assert adaptation.cap_source == "map"
    assert adaptation.reading.adaptive_none_held
    assert adaptation.status == "BEYOND_MEASURED"
    line = adaptation.beyond_sentence()
    assert ("on the adaptive clock every longest step tried stopped on some "
            "of the ground read, down to 11.25 s" in line)
    assert line.endswith("and may still stop")


def test_the_steep_rows_under_the_4500_m_crest_carry_what_they_held():
    """A139b's rows, from 20 m/s until a wind held none: the 2 km rows of
    grid slope 0.4416 and 0.4816 and the 3 km rows of 0.4105 and 0.4655."""
    expected = {
        (2000.0, 0.4416): ((15.0,) * 5, (15.0,) * 5),
        (2000.0, 0.4816): ((15.0, 15.0, 15.0, 5.5, 5.0), (15.0,) * 5),
        (3000.0, 0.4105): ((15.0, 15.0, None, None, None),
                           (15.0, 15.0, 15.0, None, None)),
        (3000.0, 0.4655): ((15.0, 5.0, None, None, None),
                           (15.0, 15.0, 15.0, None, None))}
    rows = {(row.dx_m, row.slope): row for row in _adaptive_rows()
            if row.crest_m == 4500.0}
    for key, (entries, tried) in expected.items():
        assert rows[key].adaptive[:5] == entries, key
        assert rows[key].adaptive_tried[:5] == tried, key


@pytest.mark.parametrize("dx", [2000.0, 2250.0, 2500.0, 3000.0])
@pytest.mark.parametrize("slope", [0.37, 0.40, 0.42, 0.45])
def test_steep_ground_under_a_4500_m_crest_reads_only_adaptive_cells(dx,
                                                                     slope):
    """Every cell a 2 to 3 km domain of slope up to 0.45 under a crest
    read on the 4.5 km rows takes at 20 and 30 m/s was run on the
    adaptive clock."""
    for crest in (3100.0, 3841.0, 4500.0):
        for wind in (20.0, 30.0):
            reading = tc.read_map(dx, crest, slope, wind, 4)
            assert reading.crest_row == 4500.0
            assert reading.adaptive_measured, (crest, wind)


def test_ground_steeper_than_every_row_keeps_the_fixed_step_cap():
    """The 3 km rows under the 4.5 km crest carry adaptive entries up to
    grid slope 0.4655, the steepest row there.  A domain steeper than
    that reads every one of them, and all held 15 s/km at 20 m/s, but no
    ridge that steep was run: lifted, a 3 km domain of slope 0.48 with a
    45 s max_time_step ran uncapped and reported as configured.  It keeps
    the fixed-step cap and says it may still stop; at 0.46, within the
    rows, the entries lift it."""
    run = _Run(use_adaptive_time_step=True, max_time_step=45)
    steep = tc.derive_clock(1, run, Fraction(15), 0.48,
                            _wind(20.0, crest=3841.0))
    assert steep.reading.beyond == ("slope",)
    assert steep.reading.adaptive_held_everything_tried
    assert steep.cap_source == "map" and steep.ceiling == Fraction(27)
    assert steep.status == "BEYOND_MEASURED"
    within = tc.derive_clock(1, run, Fraction(15), 0.46,
                             _wind(20.0, crest=3841.0))
    assert within.reading.beyond == ()
    assert within.cap_source == "adaptive held" and within.ceiling is None


def _adaptive_table(entry, tried, *, fixed=(5.0, 5.0, 5.0),
                    fixed_tried=None, steep=True):
    """Two 3 km rows under a 4.5 km crest, slopes 0.2 and 0.4, the first
    run on the adaptive clock with ``entry`` held of ``tried`` s/km at
    every wind, the steeper one too where ``steep``."""
    measured = tc.AdaptiveMeasurement(
        targets=((1.2, 0.84), (1.4, 0.98)), max_step_increase_pct=5,
        ladder=(15.0, 10.0, 5.0), seconds=10800.0)
    rows = []
    for slope in (0.2, 0.4):
        adaptive = ((entry,) * 3, (tried,) * 3)
        if slope == 0.4 and not steep:
            adaptive = (None, None)
        rows.append(tc.MapRow(3000.0, 4500.0, slope, 4, fixed, fixed_tried,
                              *adaptive))
        rows.append(tc.MapRow(3000.0, 4500.0, slope, 6, fixed, fixed_tried))
    return tc.StableStepMap(winds=(20.0, 60.0, 100.0), ladder=(5.0, 4.0),
                            seconds=1800.0, rows=tuple(rows),
                            adaptive=measured)


def test_a_stop_on_the_adaptive_clock_caps_it_whatever_the_fixed_map_says():
    """Where the fixed map held every step tried (no cap), a longest step
    seen to stop on the adaptive clock caps it at the entry held under
    that stop, and the line and the record say where the cap is from."""
    run = _Run(use_adaptive_time_step=True, max_time_step=41)
    table = _adaptive_table(8.0, 15.0)
    adaptation = tc.derive_clock(1, run, Fraction(15), 0.35,
                                 _wind(60.0, crest=4500.0), table=table)
    assert adaptation.reading.held_everything_tried
    assert adaptation.reading.adaptive_stopped_per_km == 8.0
    assert adaptation.ceiling == Fraction(24)
    assert adaptation.cap_source == "adaptive stop"
    assert adaptation.status == "ADAPTED"
    assert ("and on the adaptive clock a longest step longer than 24 s "
            "stopped on the ground read" in adaptation.sentence())
    assert adaptation.receipt()["max_time_step_from"] == "adaptive stop"
    # Below the fixed limit it lowers that too, and a faster clock reads it.
    lower = _adaptive_table(4.0, 15.0, fixed_tried=(13.0, 13.0, 13.0))
    fast = _faster(run, target_cfl=2.0)
    for clock in (run, fast):
        capped = tc.derive_clock(1, clock, Fraction(12), 0.35,
                                 _wind(60.0, crest=4500.0), table=lower)
        assert capped.ceiling == Fraction(12)
        assert capped.cap_source == "adaptive stop"


def test_a_row_not_run_on_the_adaptive_clock_keeps_the_fixed_cap():
    """Every cell read must have been run on the adaptive clock before its
    entries lift a cap: a gentler domain reading only rows run that way is
    lifted, a steeper one reading a row that was not keeps the fixed-step
    cap."""
    run = _Run(use_adaptive_time_step=True, max_time_step=60)
    table = _adaptive_table(15.0, 15.0, fixed_tried=(13.0, 13.0, 13.0),
                            steep=False)
    gentle = tc.derive_clock(1, run, Fraction(12), 0.15,
                             _wind(20.0, crest=4500.0), table=table)
    assert gentle.reading.adaptive_measured
    assert gentle.ceiling == Fraction(45)
    assert gentle.cap_source == "adaptive held"
    steep = tc.derive_clock(1, run, Fraction(12), 0.35,
                            _wind(20.0, crest=4500.0), table=table)
    assert not steep.reading.adaptive_measured
    assert steep.ceiling == Fraction(15) and steep.cap_source == "map"


def test_ground_where_the_adaptive_clock_held_nothing_may_still_stop():
    """A cell where every longest step tried stopped gives no cap (none of
    them held), lifts nothing, and the line says the domain may still
    stop."""
    run = _Run(use_adaptive_time_step=True, max_time_step=41)
    table = _adaptive_table(None, 15.0, fixed_tried=(13.0, 13.0, 13.0))
    adaptation = tc.derive_clock(1, run, Fraction(12), 0.35,
                                 _wind(60.0, crest=4500.0), table=table)
    reading = adaptation.reading
    assert reading.adaptive_none_held and reading.adaptive_per_km is None
    assert reading.adaptive_stopped_per_km is None
    assert adaptation.ceiling == Fraction(15)
    assert adaptation.cap_source == "map"
    assert adaptation.status == "BEYOND_MEASURED"
    line = adaptation.beyond_sentence()
    assert ("and on the adaptive clock every longest step tried stopped on "
            "some of the ground read, down to 15 s" in line)
    assert line.endswith("and may still stop")
    record = adaptation.receipt()["map_rows"]["adaptive_clock"]
    assert record["a_cell_held_none_down_to_s"] == 15.0


def test_an_unchanged_domain_where_the_adaptive_clock_held_none_says_so():
    """A 2 km adaptive domain under a 3 km crest of slope 0.09 at 40 m/s
    with max_time_step -1 (16 s): the fixed rows it reads held every step
    tried there (to 10 s), so its own clock stays as configured, but the
    row it reads was run on the adaptive clock and held no max_time_step
    down to 10 s.  That domain printed no line and reported
    AS_CONFIGURED; it now reports BEYOND_MEASURED, and its line names the
    adaptive stop and says it may still stop."""
    exp = _wizard_experiment([(60, 60)], (), 2000.0)
    root = exp.domains[0]
    exp = replace(exp, domains=(replace(root, run=replace(
        root.run, use_adaptive_time_step=True, max_time_step=-1,
        max_time_step_den=0)),))
    cautions, notes = [], []
    adapted, adaptations = tc.adapt_experiment_clock(
        exp, {1: 0.09}, {1: _wind(40.0, crest=3000.0)},
        announce=notes.append, caution=cautions.append)
    adaptation = adaptations[0]
    assert adapted is exp
    assert not adaptation.adapted and adaptation.ceiling is None
    assert adaptation.adaptive_unheld
    assert adaptation.reading.held_everything_tried
    assert adaptation.status == "BEYOND_MEASURED"
    assert len(notes) == 1 and "domain-wide terrain clock" in notes[0]
    assert len(cautions) == 1
    line = cautions[0]
    assert line == adaptation.beyond_sentence()
    assert line.startswith("time step: d01's steepest terrain slope is 0.09")
    assert ("the measured map holds steps up to 10 s there with 4 substeps"
            in line)
    assert ("and on the adaptive clock every longest step tried stopped on "
            "some of the ground read, down to 10 s" in line)
    assert line.endswith("so d01's adaptive clock keeps its own longest "
                         "step and may still stop")
    row = tc.clock_receipt(adaptations)["domains"][0]
    assert row["status"] == "BEYOND_MEASURED"
    assert "max_time_step_s" not in row
    assert row["map_rows"]["adaptive_clock"][
        "a_cell_held_none_down_to_s"] == 10.0
    # The same ground on a fixed step reads the fixed rows alone: held.
    fixed = replace(root, run=replace(root.run, use_adaptive_time_step=False))
    _, alone = tc.adapt_experiment_clock(
        replace(exp, domains=(fixed,)), {1: 0.09},
        {1: _wind(40.0, crest=3000.0)}, announce=notes.append,
        caution=cautions.append)
    assert alone[0].status == "AS_CONFIGURED" and len(cautions) == 1


_ADAPTIVE_SLOPES =(0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.33, 0.356, 0.37,
                    0.4, 0.42, 0.44, 0.45, 0.48)
_ADAPTIVE_WINDS = (20.0, 24.0, 30.0, 35.0, 40.0, 50.0, 60.0, 70.0)


@pytest.mark.parametrize("dx", [2000.0, 2250.0, 2500.0, 3000.0])
@pytest.mark.parametrize("crest", [1200.0, 1500.0, 2500.0, 3000.0, 3595.0,
                                   3841.0, 4500.0])
def test_the_adaptive_entries_keep_the_sweep_invariants(dx, crest):
    """Over the ground the adaptive clock was run on: at the count a
    domain runs, its cap never rises with slope or wind, never sits at or
    above the clock's own longest step, and never lies past an entry held
    under a longest step seen to stop on the adaptive clock."""
    table = tc.measured_map()
    for max_step in (-1, int(round(15.0 * dx / 1000.0))):
        run = _Run(dx=dx, dy=dx, use_adaptive_time_step=True,
                   max_time_step=max_step)
        upper = _effective_upper(run)
        dt = Fraction(int(round(4.0 * dx)), 1000)
        grid = {}
        for slope in _ADAPTIVE_SLOPES:
            for wind in _ADAPTIVE_WINDS:
                adaptation = tc.derive_clock(1, run, dt, slope,
                                             _wind(wind, crest=crest),
                                             table=table)
                cap = _cap(adaptation)
                if adaptation.ceiling is not None:
                    assert adaptation.ceiling < upper
                if adaptation.adaptive_unheld:
                    # Ground where the adaptive clock held nothing tried is
                    # never reported as held, changed or not.
                    assert adaptation.status == "BEYOND_MEASURED", (
                        max_step, slope, wind)
                stopped = adaptation.reading.adaptive_stopped_per_km
                if (stopped is not None and adaptation.time_step_sound == 4
                        and stopped * dx / 1000.0 < upper):
                    assert cap <= stopped * dx / 1000.0 + 1e-9, (
                        max_step, slope, wind)
                grid[slope, wind] = (cap, adaptation.time_step_sound)
        for i, slope in enumerate(_ADAPTIVE_SLOPES):
            for j, wind in enumerate(_ADAPTIVE_WINDS):
                cap, count = grid[slope, wind]
                others = []
                if i:
                    others.append((_ADAPTIVE_SLOPES[i - 1], wind))
                if j:
                    others.append((slope, _ADAPTIVE_WINDS[j - 1]))
                for other in others:
                    other_cap, other_count = grid[other]
                    if count == other_count:
                        assert cap <= other_cap, (max_step, slope, wind,
                                                  other)


def _map_loader(tmp_path, monkeypatch, edit):
    import json

    document = json.loads(tc.MAP_PATH.read_text(encoding="utf-8"))
    edit(document)
    path = tmp_path / tc.MAP_PATH.name
    path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(tc, "MAP_PATH", path)
    return tc.measured_map.__wrapped__


def _first_adaptive(document):
    return next(row for row in document["rows"]
                if "adaptive_s_per_km" in row)


def _past_tried(document):
    row = _first_adaptive(document)
    row["adaptive_s_per_km"] = [16.0] + row["adaptive_s_per_km"][1:]


def _on_six(document):
    _first_adaptive(document)["sound_steps"] = 6


def _unsaid(document):
    document.pop("adaptive")


def _short(document):
    _first_adaptive(document)["adaptive_top_s_per_km"] = [15.0]


@pytest.mark.parametrize("edit, message", [
    (_past_tried, "past the 15.0 s/km tried there"),
    (_on_six, "adaptive entries on a 6-substep row"),
    (_unsaid, "does not say how they were measured"),
    (_short, "the map declares 9")])
def test_a_map_whose_adaptive_entries_would_misread_is_refused(
        tmp_path, monkeypatch, edit, message):
    """An entry past its tried range would lift a cap over steps never
    run; an entry on a six-substep row would never be read; entries
    without their measurement cannot say which clocks they hold for."""
    load = _map_loader(tmp_path, monkeypatch, edit)
    with pytest.raises(ValueError, match=message):
        load()


def test_the_probe_merges_adaptive_entries_onto_the_four_substep_row():
    """``merge`` writes an ``adaptive-extend`` row's entries on the row's
    four-substep line only, leaves its fixed entries alone, and writes the
    block saying how they were measured."""
    import copy
    import json

    from tools.terrain_clock_probe import adaptive_block, merge

    document = json.loads(tc.MAP_PATH.read_text(encoding="utf-8"))
    for row in document["rows"]:
        row.pop("adaptive_s_per_km", None)
        row.pop("adaptive_top_s_per_km", None)
    document.pop("adaptive")
    before = copy.deepcopy(document)
    measured = {"targets": [[1.2, 0.84], [1.4, 0.98]], "increase_pct": 5,
                "ladder_s_per_km": [15.0, 5.0], "check_seconds": 10800.0,
                "rows": [{"dx_m": 2000.0, "crest_m": 4500.0,
                          "ridge_slope": 0.3, "sound_steps": 4,
                          "adaptive_s_per_km": [15.0] + [None] * 8,
                          "adaptive_top_s_per_km": [15.0] + [None] * 8}]}
    merge(document, measured["rows"], adaptive=adaptive_block(measured))
    changed = [(a, b) for a, b in zip(before["rows"], document["rows"])
               if a != b]
    assert len(changed) == 1
    old, new = changed[0]
    assert new["sound_steps"] == 4 and new["ridge_slope"] == 0.3
    assert new["stable_s_per_km"] == old["stable_s_per_km"]
    assert new["adaptive_s_per_km"][0] == 15.0
    assert document["adaptive"]["max_step_increase_pct"] == 5
    assert document["adaptive"]["seconds"] == 10800.0


# ---------------------------------------------------------------------------
# The pinned clock (RunConfig.terrain_clock = "pinned"): the configured
# step and count run, the measured derivation is advice in the receipt.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PinnedRun(_Run):
    terrain_clock: str = "pinned"


def test_a_pinned_domain_runs_its_configured_clock_under_a_jet():
    """HRRR's own ground: a 3932 m crest of slope 0.356 at 3 km, a fixed
    20 s step on 6 substeps, under a 50 m/s crest-level wind.  Measured,
    the map halves the step; pinned, the configured pair runs and the
    halving is advice."""
    measured = tc.derive_clock(1, _Run(time_step_sound=6), Fraction(20),
                               0.356, _wind(50.0, crest=3932.0))
    assert measured.adapted and measured.division >= 2
    pinned = tc.derive_clock(1, _PinnedRun(time_step_sound=6), Fraction(20),
                             0.356, _wind(50.0, crest=3932.0))
    assert pinned.pinned
    assert not pinned.adapted
    assert pinned.status == "PINNED"
    assert pinned.division == 1 and pinned.dt == Fraction(20)
    assert pinned.time_step_sound == 6
    assert pinned.ceiling is None
    assert pinned.advice == (measured.division, measured.time_step_sound,
                             measured.ceiling)
    assert pinned.advice_differs
    # The reading itself is the measured one: nothing is hidden.
    assert pinned.reading is not None
    assert pinned.held_per_km == measured.held_per_km
    assert pinned.limit_per_km == measured.limit_per_km
    line = pinned.pinned_sentence()
    assert line.startswith("time step: d01's clock is pinned")
    assert "20 s steps on 6 acoustic substeps as configured" in line
    assert "would have run" in line and "not applied" in line


def test_a_pinned_domain_the_map_holds_says_so_without_advice():
    pinned = tc.derive_clock(1, _PinnedRun(), Fraction(15), 0.05,
                             _wind(90.0, crest=900.0))
    assert pinned.pinned and not pinned.advice_differs
    assert pinned.advice == (1, 4, None)
    assert "holds that clock" in pinned.pinned_sentence()


def test_a_pinned_adaptive_domain_takes_no_ceiling():
    run = _PinnedRun(use_adaptive_time_step=True)
    measured = tc.derive_clock(1, replace(run, terrain_clock="measured"),
                               Fraction(12), 0.35, _wind(70.0, crest=4500.0))
    assert measured.ceiling is not None
    pinned = tc.derive_clock(1, run, Fraction(12), 0.35,
                             _wind(70.0, crest=4500.0))
    assert pinned.ceiling is None and pinned.division == 1
    assert pinned.advice[2] == measured.ceiling
    assert "an adaptive step capped at" in pinned.pinned_sentence()


def test_the_pinned_receipt_carries_the_clock_and_the_advice():
    from gpuwm.ingest.boundary_stream import derived_clock

    pinned = tc.derive_clock(1, _PinnedRun(time_step_sound=6), Fraction(20),
                             0.356, _wind(50.0, crest=3932.0))
    measured = tc.derive_clock(1, _Run(time_step_sound=6), Fraction(20),
                               0.356, _wind(50.0, crest=3932.0))
    row = pinned.receipt()
    assert row["status"] == "PINNED"
    assert row["clock"] == "pinned"
    assert row["dt_s"] == tc._rational(Fraction(20))
    assert row["step_division"] == 1
    assert row["time_step_sound"] == 6
    advice = row["advice"]
    assert advice["applied"] is False and advice["differs"] is True
    assert advice["step_division"] == measured.division
    assert advice["time_step_sound"] == measured.time_step_sound
    assert advice["dt_s"] == tc._rational(measured.dt)
    assert advice["max_time_step_s"] is None
    # The map's own rows are still written, so the verdict is readable.
    assert row["map_rows"]["shortest_entry_under_a_stop_s"] is not None
    # What the seal compares is the configured clock, on both sides.
    receipt = tc.clock_receipt((pinned,))
    assert derived_clock(receipt) == {
        1: (tc.json.dumps(tc._rational(Fraction(20)), sort_keys=True),
            1, 6, "null", None)}
    # A measured receipt is byte-for-byte what it was: no new keys.
    assert "clock" not in measured.receipt()
    assert "advice" not in measured.receipt()


def test_a_pinned_experiment_is_handed_back_unchanged_with_a_caution():
    """The experiment door: a pinned root under a jet keeps every field,
    and the run prints the advice as a caution."""
    exp = _wizard_experiment([(60, 60)], (), 3000.0)
    root = exp.domains[0]
    exp = replace(exp, domains=(replace(
        root, run=replace(root.run, terrain_clock="pinned")),))
    said = []
    adapted, adaptations = tc.adapt_experiment_clock(
        exp, {1: 0.35}, {1: _wind(70.0, crest=4500.0)},
        announce=lambda line: said.append(("announce", line)),
        caution=lambda line: said.append(("caution", line)))
    assert adapted is exp
    assert len(adaptations) == 1 and adaptations[0].pinned
    assert [kind for kind, _ in said] == ["caution"]
    assert "clock is pinned" in said[0][1]
    assert adaptations[0].advice[0] >= 2


def test_a_pinned_experiment_the_map_holds_prints_one_announcement():
    exp = _wizard_experiment([(60, 60)], (), 3000.0)
    root = exp.domains[0]
    exp = replace(exp, domains=(replace(
        root, run=replace(root.run, terrain_clock="pinned")),))
    said = []
    adapted, adaptations = tc.adapt_experiment_clock(
        exp, {1: 0.05}, {1: _wind(20.0, crest=900.0)},
        announce=lambda line: said.append(("announce", line)),
        caution=lambda line: said.append(("caution", line)))
    assert adapted is exp
    assert said == [("announce", adaptations[0].pinned_sentence())]


def test_the_terrain_clock_mode_is_validated_by_name():
    from gpuwm.config import TERRAIN_CLOCK_MODES, validate_run_config

    # "local_face" is the default from 2.8.8 (gpuwm/terrain_clock_local.py)
    assert TERRAIN_CLOCK_MODES == ("measured", "pinned", "local_face")
    exp = _wizard_experiment([(60, 60)], (), 3000.0)
    run = exp.domains[0].run
    assert run.terrain_clock == "local_face"
    assert validate_run_config(replace(run, terrain_clock="measured")) \
        .terrain_clock == "measured"
    assert validate_run_config(replace(run, terrain_clock="pinned")) \
        .terrain_clock == "pinned"
    assert validate_run_config(replace(run, terrain_clock="local_face")) \
        .terrain_clock == "local_face"
    with pytest.raises(ValueError, match="terrain_clock='off' is not one of"):
        validate_run_config(replace(run, terrain_clock="off"))


def test_the_terrain_mode_rides_only_in_route_selector_metadata():
    from gpuwm.hrrr_route_inputs import render_namelist_input
    from gpuwm.physics_source_defaults import read_physics_selector_comment
    default = _wizard_experiment([(60, 60)], (), 3000.0)
    exp = replace(default, domains=(replace(
        default.root, run=replace(default.root.run,
                                  terrain_clock="measured")),))
    measured = render_namelist_input(exp)
    # The route reads its pair under the hrrr recipe, which fills
    # "pinned" where the comment is silent, so the measured mode is
    # carried too: an unmarked emission was read back as pinned.
    assert read_physics_selector_comment(measured)["terrain_clock"] == "measured"
    # So is the default (local_face from 2.8.8), for the same reason.
    local = render_namelist_input(default)
    assert read_physics_selector_comment(local)["terrain_clock"] == "local_face"
    assert local.splitlines(keepends=True)[1:] == measured.splitlines(keepends=True)[1:]
    pinned_exp = replace(exp, domains=(replace(
        exp.root, run=replace(exp.root.run, terrain_clock="pinned")),))
    pinned = render_namelist_input(pinned_exp)
    assert read_physics_selector_comment(pinned)["terrain_clock"] == "pinned"
    # The selector is a WRF comment; all existing WRF field bytes stay put.
    assert pinned.splitlines(keepends=True)[1:] == measured.splitlines(keepends=True)[1:]


def test_the_pinned_mode_never_reaches_a_measured_fingerprint():
    """"measured" is dropped from the restart identity and the checkpoint
    echo, so every fingerprint written before the field keeps its value;
    "pinned" binds, and so does "local_face", the default from 2.8.8."""
    from gpuwm.core.model import restart_identity_payload
    from gpuwm.io.restart import configuration_echo

    default = _wizard_experiment([(60, 60)], (), 3000.0)
    local = restart_identity_payload(default)
    assert local["domains"][0]["run"]["terrain_clock"] == "local_face"
    assert configuration_echo(default.root.run)["terrain_clock"] \
        == "local_face"
    exp = replace(default, domains=(replace(
        default.root, run=replace(default.root.run,
                                  terrain_clock="measured")),))
    root = exp.domains[0]
    measured = restart_identity_payload(exp)
    assert "terrain_clock" not in measured["domains"][0]["run"]
    pinned_exp = replace(exp, domains=(replace(
        root, run=replace(root.run, terrain_clock="pinned")),))
    pinned = restart_identity_payload(pinned_exp)
    assert pinned["domains"][0]["run"]["terrain_clock"] == "pinned"
    assert "terrain_clock" not in configuration_echo(root.run)
    assert configuration_echo(
        replace(root.run, terrain_clock="pinned"))["terrain_clock"] == "pinned"


def test_a_pinned_run_reads_a_tree_prepared_under_the_measured_clock():
    """Preparation never reads terrain_clock, so it is skipped outright in
    the prepared-cache comparison: a tree prepared under "measured" (or
    before the field) serves a pinned run at the same dt, in both
    directions.  The step itself still binds."""
    from gpuwm.ingest.prepared_cache import (
        DEFAULT_TOLERANT_IDENTITY_FIELDS, PREPARATION_INERT_RUN_FIELDS,
        compare_prepared_domain_config, effective_prepared_domain_config)

    assert "run.terrain_clock" in PREPARATION_INERT_RUN_FIELDS
    assert "run.terrain_clock" not in DEFAULT_TOLERANT_IDENTITY_FIELDS
    older = {"run": {"nx": 100, "dt": 20.0, "time_step_sound": 6}}
    measured = {"run": {**older["run"], "terrain_clock": "measured"}}
    pinned = {"run": {**older["run"], "terrain_clock": "pinned"}}
    for cached, live in ((older, pinned), (measured, pinned),
                         (pinned, measured)):
        _tolerated, differing = compare_prepared_domain_config(
            effective_prepared_domain_config(cached),
            effective_prepared_domain_config(live))
        assert differing == [], (cached, live, differing)
    moved = {"run": {**pinned["run"], "dt": 10.0}}
    _tolerated, differing = compare_prepared_domain_config(
        effective_prepared_domain_config(measured),
        effective_prepared_domain_config(moved))
    assert differing == ["run.dt"]
