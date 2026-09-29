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


def _wizard_experiment(dims, ratios, root_dx_m, *, ref_lat=-33.0, hours=1):
    from gpuwm import domain_wizard as wizard

    text = wizard.render_config(
        name="terrain-clock", start_time=datetime(2025, 7, 1, 0),
        hours=hours, projection={
            "map_proj": "lambert", "ref_lat": ref_lat, "ref_lon": -70.1,
            "truelat1": ref_lat + 10.0, "truelat2": ref_lat - 10.0,
            "stand_lon": -70.1},
        dims=dims, ratios=ratios, fetch_hints={"source": "gfs"},
        case_data=None, root_dx_m=root_dx_m, history_interval_s=300.0)
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
    for wind in (20.0, 60.0, 100.0):
        reading = tc.read_map(dx, 1200.0, 0.05, wind, 4)
        assert reading.per_km == tc.measured_map().top
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
    assert len(lines) == 2
    assert lines[0].startswith("time step: d01's")
    assert lines[1].startswith("time step: d02 runs")
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
    assert len(lines) == 1
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
    assert len(lines) == 1 and "at least 6 acoustic substeps" in lines[0]
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
