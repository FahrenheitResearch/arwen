"""Actual parent LSM hooks and checkpoint transport; no hydrologic parity claim."""
from datetime import datetime
import sys
from types import ModuleType

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from gpuwm.config import LAND_SURFACE_SOIL_LAYERS, RunConfig
from gpuwm.core.grid import make_base_state, make_vertical_coord
from gpuwm.core.moist import init_moist_balanced
from gpuwm.core.physics import initialize_physics
from gpuwm.core.diagnostics import update_diagnostics
from gpuwm.hydro import ResidentHydroDriver, Stage
from gpuwm.hydro.runtime import settings_scope
from gpuwm.io import restart


class LaunchProbe:
    """GPU instrumentation of hook order, deliberately not a runoff module."""
    restart_identity = {"instrumentation": "nextgen-driver-order/v1"}

    def __init__(self):
        self.arrays = {"order": cp.zeros(1, cp.int32), "word64": cp.full(1, -0.0, cp.float64)}
        self.kernel = cp.RawKernel('extern "C" __global__ void record(int* x,int digit) {x[0]=10*x[0]+digit;}', "record")
        self.kernel.compile()

    def before(self, dt, *, stream=None):
        self.kernel((1,), (1,), (self.arrays["order"], np.int32(1)), stream=stream)

    def after(self, dt, *, stream=None):
        self.kernel((1,), (1,), (self.arrays["order"], np.int32(2)), stream=stream)


def _state(monkeypatch, *, grid_id=1, use_current_settings=False, land_scheme=4,
           factory_override=None):
    cfg = RunConfig(nx=4, ny=2, nz=12, dx=2000., dy=2000., ztop=8000.,
                    dt=10., run_seconds=0., time_step_sound=4, moist=True,
                    mp_physics=0, sf_sfclay_physics=1, sf_surface_physics=land_scheme,
                    bl_pbl_physics=0, bldt=1., grid_id=grid_id,
                    num_soil_layers=LAND_SURFACE_SOIL_LAYERS.validated(land_scheme))
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: 298.+.004*np.asarray(z, np.float64),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, coord, base,
                               lambda z: .01*np.exp(-np.asarray(z, np.float64)/2400.))
    state.u.fill(6.)
    state.v.fill(.5)
    probe = LaunchProbe()
    module = ModuleType("gpuwm.hydro._driver_test")
    def factory(**kwargs):
        return ResidentHydroDriver(device_id=kwargs["device_id"], interval_seconds=60.,
                                   identity={"test": "launch transport only"},
                                   before=[Stage("probe_before", probe, probe.before)],
                                   after=[Stage("probe_after", probe, probe.after)])
    module.prepare = factory if factory_override is None else factory_override
    monkeypatch.setitem(sys.modules, module.__name__, module)
    def initialize(*unused):
        driver = initialize_physics(state, cfg, soil_temperature=285., soil_moisture=.31,
                                     liquid_moisture=.31, swdown=450., glw=310.,
                                     noahmp_start_time=datetime(2026, 7, 1, 18),
                                     noahmp_latitude=40., noahmp_longitude=-100.)
        # Explicit synthetic radiation producer for the real RUC update.
        if "gsw" in driver.fields:
            driver.set_forcing(gsw=350.)
        return driver
    if not use_current_settings:
        with settings_scope({"enabled": True, "driver_factory": module.__name__+":prepare"}):
            physics = initialize()
    else:
        physics = initialize()
    return state, cfg, physics, probe


@pytest.mark.parametrize("land_scheme", [2, 3, 4])
def test_default_cli_attachment_and_explicit_off_on_actual_parent(monkeypatch, tmp_path, land_scheme):
    from types import SimpleNamespace
    from gpuwm.hydro.runtime import current_settings, go_scope
    path = tmp_path / "prepared.toml"
    path.write_text('[hydro]\ndriver_factory="gpuwm.hydro._driver_test:prepare"\n')
    args = SimpleNamespace(config=path, hydro=None)
    @go_scope
    def build(args, grid_id=1):
        return _state(monkeypatch, grid_id=grid_id, use_current_settings=True,
                      land_scheme=land_scheme)
    state, cfg, physics, probe = build(args)
    assert physics.hydro.device_id == state.u.device.id
    state.elapsed_seconds = 0.
    update_diagnostics(state)
    physics.compute(state, cfg)
    cp.cuda.get_current_stream().synchronize()
    assert int(probe.arrays["order"].get()[0]) == 12
    assert physics.hydro.steps == 1
    assert cfg.sf_surface_physics == land_scheme
    assert (physics.noahmp_params is not None) == (land_scheme == 4)
    assert current_settings() is None
    args.hydro = "off"
    _, _, disabled, unused = build(args)
    assert disabled.hydro is None
    assert int(unused.arrays["order"].get()[0]) == 0
    args.hydro = None
    _, _, child, unused = build(args, grid_id=2)
    assert child.hydro is None
    assert current_settings() is None


@pytest.mark.parametrize("land_scheme", [2, 3, 4])
def test_actual_lsm_cadence_and_fixed_parent(monkeypatch, land_scheme):
    state, cfg, physics, probe = _state(monkeypatch, land_scheme=land_scheme)
    assert physics.hydro.device_id == state.u.device.id
    for step in (1, 2, 6):
        state.elapsed_seconds = (step-1)*cfg.dt
        update_diagnostics(state)
        physics.compute(state, cfg)
    cp.cuda.get_current_stream().synchronize()
    # Mandatory first LSM call and the 60-second cadence call; step 2 skips.
    assert int(probe.arrays["order"].get()[0]) == 1212
    assert physics.hydro.steps == 2
    assert physics.hydro.elapsed_seconds == 120.
    _, _, child, child_probe = _state(monkeypatch, grid_id=2, land_scheme=land_scheme)
    assert child.hydro is None
    assert int(child_probe.arrays["order"].get()[0]) == 0


def test_parent_without_lsm_refuses_inactive_hydro_clock():
    from types import SimpleNamespace
    from gpuwm.hydro.runtime import attach_parent
    # Without an LSM dispatch there is no before/after hook to advance hydro.
    with settings_scope({"enabled": True}):
        with pytest.raises(ValueError, match="active parent LSM cadence"):
            attach_parent(SimpleNamespace(), None, SimpleNamespace(sf_surface_physics=0))


@pytest.mark.parametrize("land_scheme", [2, 3])
def test_noahmp_exchange_still_names_missing_parent_fields(monkeypatch, land_scheme):
    from gpuwm.hydro.noahowp.factory import _parent_fields
    _, _, physics, _ = _state(monkeypatch, land_scheme=land_scheme)
    before = {name: value.data.ptr for name, value in physics.fields.items()}
    with pytest.raises(ValueError, match="requires the parent Noah-MP LSM"):
        _parent_fields(physics, "coupled")
    assert before == {name: value.data.ptr for name, value in physics.fields.items()}


@pytest.mark.parametrize("land_scheme", [2, 3, 4])
def test_whole_atmosphere_checkpoint_preserves_hydro_in_place(monkeypatch, tmp_path, land_scheme):
    state, cfg, physics, probe = _state(monkeypatch, land_scheme=land_scheme)
    physics.hydro.before_lsm()
    physics.hydro.advance()
    path = restart.write_restart(tmp_path/"with-hydro.npz", state, cfg)
    fresh, _, fresh_physics, fresh_probe = _state(monkeypatch, land_scheme=land_scheme)
    pointers = {key: value.data.ptr for key, value in fresh_physics.hydro.arrays.items()}
    restart.restore_restart(path, fresh, cfg)
    assert fresh_physics.hydro.steps == 1
    for key, value in physics.hydro.arrays.items():
        target = fresh_physics.hydro.arrays[key]
        assert target.data.ptr == pointers[key]
        assert target.get().tobytes() == value.get().tobytes()
    saved = fresh.u.get().tobytes()
    fresh_physics.hydro = None
    with pytest.raises(restart.RestartMismatchError, match="no hydro driver"):
        restart.restore_restart(path, fresh, cfg)
    assert fresh.u.get().tobytes() == saved


def test_member_context_binds_parent_identity_before_restart(monkeypatch, tmp_path):
    """A member checkpoint must not overwrite another member's atmosphere."""
    from gpuwm.ensemble.runtime_context import (
        MemberOutputCapture, current_capture, member_output_scope,
    )

    def capture(member):
        return member_output_scope(MemberOutputCapture(lambda **unused: None, member))

    with capture(7):
        first, cfg, first_physics, first_probe = _state(monkeypatch)
    assert current_capture() is None
    driver = first_physics.hydro
    driver.before_lsm()
    driver.advance()
    path = restart.write_restart(tmp_path / "member7.npz", first, cfg)
    with capture(22):
        other, _, other_physics, other_probe = _state(monkeypatch)
    other.u.fill(9.)
    before = other.u.get().tobytes()
    before_hydro = {key: value.get().tobytes()
                    for key, value in other_physics.hydro.arrays.items()}
    with pytest.raises(restart.RestartMismatchError, match="NextGen.*differ"):
        restart.restore_restart(path, other, cfg)
    assert other.u.get().tobytes() == before
    assert before_hydro == {key: value.get().tobytes()
                            for key, value in other_physics.hydro.arrays.items()}
    assert driver.member_id == 7
    assert other_physics.hydro.member_id == 22
    assert driver.restart_identity["member_id"] == 7
    assert driver is not other_physics.hydro
    assert {value.data.mem.ptr for value in driver.arrays.values()}.isdisjoint(
        value.data.mem.ptr for value in other_physics.hydro.arrays.values())
    with capture(7):
        fresh, _, fresh_physics, fresh_probe = _state(monkeypatch)
    pointers = {key: value.data.ptr for key, value in fresh_physics.hydro.arrays.items()}
    restart.restore_restart(path, fresh, cfg)
    assert fresh_physics.hydro.member_id == 7
    assert fresh_physics.hydro.steps == 1
    assert fresh_physics.hydro.elapsed_seconds == 60.
    assert pointers == {key: value.data.ptr for key, value in fresh_physics.hydro.arrays.items()}
    for key, value in driver.arrays.items():
        assert value.get().tobytes() == fresh_physics.hydro.arrays[key].get().tobytes()
    print({"scope": "actual parent physics and whole-atmosphere checkpoint; instrumentation only",
           "members": [7, 22], "different_hydro_bytes": 0,
           "restored_hydro_bytes": sum(value.nbytes for value in driver.arrays.values()),
           "wrong_member_refused_before_writes": True,
           "same_member_pointers_preserved": True})


@pytest.mark.parametrize("accumulated", [False, True])
def test_parent_forcing_binds_live_surface_mapping(monkeypatch, tmp_path, accumulated):
    """Real Noah parent plus explicit synthetic canonical fields/CSR, not a basin proof."""
    from types import SimpleNamespace
    from gpuwm.hydro.forcing import FIELDS, prepare_parent

    prepared = []
    fixtures = []

    def factory(**kwargs):
        state = kwargs["state"]
        physics = kwargs["physics"]
        # Canonical synthetic humidity/wind/radiation/rain are declared here.
        # In particular, native Q2 mixing ratio is NOT relabelled Q2D.
        values = (285., .005, 2., 1., 100000., 310., 450., .01)
        fixture = {
            name: cp.full((2, 4), value, cp.float32)
            for name, value in zip(FIELDS, values)}
        fixture["rainc"] = cp.zeros((2, 4), cp.float32)
        fixture["rainnc"] = cp.zeros((2, 4), cp.float32)
        # Use the production surface fields without copying or host conversion.
        paths = {name: "components.fixture." + name for name in FIELDS}
        paths.update(T2D="physics.fields.t2", PSFC="physics.fields.psfc")
        arrays = {"weights_offsets": cp.asarray([0, 2, 4], cp.uint64),
                  "weights_indices": cp.asarray([0, 1, 4, 7], cp.uint32),
                  "weights": cp.asarray([.5, .5, .25, .75], cp.float64)}
        fabric = SimpleNamespace(arrays=arrays, catchment_ids=("cat-a", "cat-b"),
                                 restart_identity={"fixture": "explicit two-row CSR"})
        monkeypatch.setattr("gpuwm.hydrofabric.load_cuda", lambda *a, **k: fabric)
        options = dict(hydrofabric_manifest="synthetic-csr-only", parent_fields=paths,
                       time_origin="2026-07-01T18:00:00Z")
        if accumulated:
            options.update(rain_accumulators=["components.fixture.rainc",
                                              "components.fixture.rainnc"],
                           rain_baseline="at_prepare")
        component = prepare_parent(settings={}, options=options, state=state,
            physics=physics, device_id=kwargs["device_id"], mode="coupled", components={"fixture": fixture})
        assert component.arrays["parent/T2D"].data.ptr == physics.fields["t2"].data.ptr
        assert component.arrays["parent/PSFC"].data.ptr == physics.fields["psfc"].data.ptr
        prepared.append(component)
        fixtures.append(fixture)
        return ResidentHydroDriver(device_id=kwargs["device_id"], interval_seconds=60.,
            identity={"test": "parent forcing binding only"},
            after=[Stage("forcing", component, component.advance)])

    state, cfg, physics, _ = _state(monkeypatch, land_scheme=2, factory_override=factory)
    component = prepared[-1]
    fixture = fixtures[-1]
    if accumulated:
        fixture["rainc"].fill(1.)
        fixture["rainnc"].fill(2.)
    # The real parent field allocation remains live after factory preparation.
    physics.fields["t2"].fill(290.)
    physics.hydro.before_lsm()
    physics.hydro.advance()
    np.testing.assert_array_equal(component.fields["T2D"].get(), [290., 290.])
    np.testing.assert_array_equal(component.fields["RAINRATE"].get(),
        np.full(2, 3./60. if accumulated else .01, np.float32))
    pointers = {key: value.data.ptr for key, value in physics.hydro.arrays.items()}
    checkpoint = restart.write_restart(tmp_path / "live-parent.npz", state, cfg)
    fresh, _, fresh_physics, _ = _state(monkeypatch, land_scheme=2, factory_override=factory)
    fresh_component = prepared[-1]
    fresh_fixture = fixtures[-1]
    fresh_pointers = {key: value.data.ptr for key, value in fresh_physics.hydro.arrays.items()}
    restart.restore_restart(checkpoint, fresh, cfg)
    for key, value in physics.hydro.arrays.items():
        assert value.get().tobytes() == fresh_physics.hydro.arrays[key].get().tobytes()
    # Continue both independently, including the accumulated-rain baseline.
    for parent, driver, fixture in ((state, physics, fixture),
                                    (fresh, fresh_physics, fresh_fixture)):
        driver.fields["t2"].fill(292.)
        if accumulated:
            fixture["rainc"].fill(3.)
            fixture["rainnc"].fill(3.)
        driver.hydro.before_lsm()
        driver.hydro.advance()
    np.testing.assert_array_equal(fresh_component.fields["T2D"].get(), [292., 292.])
    np.testing.assert_array_equal(fresh_component.fields["RAINRATE"].get(),
        np.full(2, 3./60. if accumulated else .01, np.float32))
    for key, value in physics.hydro.arrays.items():
        assert value.get().tobytes() == fresh_physics.hydro.arrays[key].get().tobytes()
    assert pointers == {key: value.data.ptr for key, value in physics.hydro.arrays.items()}
    assert fresh_pointers == {key: value.data.ptr for key, value in fresh_physics.hydro.arrays.items()}
    assert component.arrays["parent/T2D"].data.ptr == physics.fields["t2"].data.ptr
    assert fresh_component.arrays["parent/T2D"].data.ptr == fresh_physics.fields["t2"].data.ptr
    assert physics.hydro.steps == fresh_physics.hydro.steps == 2
    # Also enter the actual Noah dispatch, rather than advancing only by hand.
    for parent, driver in ((state, physics), (fresh, fresh_physics)):
        parent.elapsed_seconds = 0.
        update_diagnostics(parent)
        driver.compute(parent, cfg)
    cp.cuda.get_current_stream().synchronize()
    assert physics.hydro.steps == fresh_physics.hydro.steps == 3
    assert physics.hydro.check_health() == {"forcing/status": 0}
    assert fresh_physics.hydro.check_health() == {"forcing/status": 0}
    print({"scope": "actual Noah parent field aliases; synthetic canonical forcing/CSR",
           "rain": "accumulated" if accumulated else "direct",
           "checkpoint_carriers": len(physics.hydro.arrays),
           "checkpoint_logical_bytes": sum(a.nbytes for a in physics.hydro.arrays.values()),
           "continuation_differing_bytes": 0,
           "forcing_resources": component._kernel.attributes})
    # A forcing fault must reach publication checks, not just produce NaNs.
    if accumulated:
        fresh_fixture["rainc"].fill(0.)
        fresh_fixture["rainnc"].fill(0.)
    else:
        fresh_fixture["RAINRATE"].fill(-1.)
    fresh_physics.hydro.before_lsm()
    fresh_physics.hydro.advance()
    np.testing.assert_array_equal(fresh_component.arrays["status"].get(), [1, 1])
    assert np.isnan(fresh_component.fields["RAINRATE"].get()).all()
    with pytest.raises(RuntimeError, match="forcing/status"):
        fresh_physics.hydro.check_health()
    rejected = tmp_path / "invalid-forcing.npz"
    with pytest.raises(RuntimeError, match="forcing/status"):
        restart.write_restart(rejected, fresh, cfg)
    assert not rejected.exists()


@pytest.mark.parametrize("fault", ["negative_rain", "outside_window"])
def test_series_forcing_fault_reaches_checkpoint_gate(fault):
    """Synthetic prepared series exercises the resident publication contract."""
    from gpuwm.hydro.forcing import PreparedForcingSeries
    frames = cp.ones((1, 8, 2), cp.float32)
    frames[:, 7].fill(.01)
    component = PreparedForcingSeries(frames, interval_seconds=60.,
        time_origin="2026-07-01T18:00:00Z", identity={"test": "synthetic series"})
    driver = ResidentHydroDriver(device_id=frames.device.id, interval_seconds=60.,
        identity={"test": "series publication"},
        after=[Stage("forcing", component, component.advance)])
    driver.before_lsm()
    driver.advance()
    assert driver.check_health() == {"forcing/status": 0}
    if fault == "negative_rain":
        frames[:, 7].fill(-1.)
        component.arrays["clock_seconds"].fill(0.)
    driver.before_lsm()
    driver.advance()
    np.testing.assert_array_equal(component.arrays["status"].get(), [1, 1])
    assert np.isnan(component.fields["RAINRATE"].get()).all()
    with pytest.raises(RuntimeError, match="forcing/status"):
        driver.checkpoint_metadata()
