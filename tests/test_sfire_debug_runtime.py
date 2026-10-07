"""Native debug files and inclusive, uncoupled fire diagnostics."""
from types import SimpleNamespace
import re
import numpy as np
import pytest
from conftest import requires_gpu


def test_positive_debug_interval_uses_first_n_then_every_n(monkeypatch):
    from gpuwm import sfire_debug
    written = []
    monkeypatch.setattr(sfire_debug, "dump_fields", lambda fields, step: written.append(step))
    cfg = SimpleNamespace(fire_print_msg=0, fire_print_file=3)
    fire = SimpleNamespace(grid=SimpleNamespace(step_count=0))
    for step in range(1, 11):
        fire.grid.step_count = step
        sfire_debug.report_step(fire, cfg, fields={})
    assert written == [1, 2, 3, 6, 9]


@pytest.mark.parametrize("count", [1, 2, 5])
def test_diagnostic_inclusive_passes_freeze_inputs_and_stop(count):
    from gpuwm.sfire_debug import run_test_steps, FireDiagnosticComplete
    cfg = SimpleNamespace(fire_test_steps=count)
    state, atmosphere, surface = object(), object(), object()
    calls = []
    fire = SimpleNamespace(grid=SimpleNamespace(time_seconds=10.0), spotting=None)
    fire._report_step = lambda cfg: None
    def once(s, c, a, time, dt, sf, **kwargs):
        calls.append((s, c, a, time, dt, sf, kwargs))
        fire.grid.time_seconds += dt
    fire._advance_once = once
    with pytest.raises(FireDiagnosticComplete) as stop:
        run_test_steps(fire, state, cfg, atmosphere, 10.0, 0.5, surface)
    assert stop.value.passes == count + 1
    assert stop.value.time_seconds == 10.0 + 0.5 * (count + 1)
    assert [row[3] for row in calls] == [10.0 + 0.5*k for k in range(count+1)]
    assert all(row[0] is state and row[2] is atmosphere and row[5] is surface for row in calls)
    assert all(row[-1] == {"feedback": False, "report": False} for row in calls)


def test_zero_test_steps_keeps_one_normal_coupler_call():
    from gpuwm.core.sfire_coupler import FireCoupler
    calls = []
    fire = SimpleNamespace(_advance_once=lambda *args: calls.append(args) or "result")
    cfg = SimpleNamespace(fire_test_steps=0)
    assert FireCoupler.advance(fire, "state", cfg, "atm", 0.0, 1.0, "surface") == "result"
    assert len(calls) == 1


def test_worker_probe_requires_native_debug_symbol(monkeypatch):
    from gpuwm import sfire_capabilities
    from gpuwm.static import sfire
    library = SimpleNamespace(gpuwm_static_sfire_experiment_grids=True,
        gpuwm_static_sfire_load=True,gpuwm_static_sfire_observed_perimeter=True)
    monkeypatch.setattr(sfire,"_library",lambda: library)
    missing = sfire_capabilities.capability_document()
    assert not missing["available"]
    assert "gpuwm_static_sfire_debug_array" in missing["missing"]
    library.gpuwm_static_sfire_debug_array = True
    assert sfire_capabilities.capability_document()["available"]


def _state(cp, **changes):
    from test_restart import _physics_state
    values = dict(nx=24, ny=20, nz=16, dx=90.0, dy=120.0, dt=0.25,
        mp_physics=1, cu_physics=0, ra_physics=0, ifire=2, sr_x=2, sr_y=2,
        fire_fuel_read=0, fire_fuel_cat=1, fire_fmc_read=0, fire_boundary_guard=2,
        fmoist_run=True, fmoist_interp=True, fmoist_freq=1,
        fire_num_ignitions=1, fire_ignition_start_x1=1000.0,
        fire_ignition_start_y1=1000.0, fire_ignition_radius1=100.0,
        fire_ignition_ros1=20.0)
    values.update(changes)
    state, cfg, driver = _physics_state(cp, **values)
    # Seed frozen near-surface inputs from the initialized skin/vapor fields.
    driver.fields["t2"][...] = driver.fields["tsk"]
    driver.fields["q2"][...] = state.qv[0]
    return state, cfg, driver


def _level_files(directory):
    return sorted(path.name for path in directory.iterdir() if re.fullmatch(r"lfn_\d{8}\.txt",path.name))


def _advance(state, cfg):
    from gpuwm.core.physics import _prepare_atmosphere
    driver = state.physics
    atmosphere = _prepare_atmosphere(state)
    surface = dict(driver.fields, psfc=atmosphere["p_interface"][0],
        rainc=driver._zero_accumulator() if driver.rainc is None else driver.rainc,
        rainnc=driver.microphysics.rainnc)
    return driver.fire.advance(state, cfg, atmosphere, driver.fire.grid.time_seconds, cfg.dt, surface)


@requires_gpu
def test_native_debug_files_activate_on_coupled_steps(tmp_path, monkeypatch):
    import cupy as cp
    from gpuwm.core.dycore import step
    from gpuwm.sfire_debug import dump_fields
    monkeypatch.chdir(tmp_path)
    state, cfg, _ = _state(cp, fire_print_file=2)
    for _ in range(4):
        step(state, cfg)
    assert _level_files(tmp_path) == [
        "lfn_00000001.txt", "lfn_00000002.txt", "lfn_00000004.txt"]
    field = state.physics.fire.output_fields()["LFN"]
    checked = dump_fields({"LFN": field}, 4, directory=tmp_path/"verified")
    assert checked[0].read_bytes() == (tmp_path/"lfn_00000004.txt").read_bytes()
    assert checked[0].stat().st_size == (field.size+6)*21


@requires_gpu
def test_moisture_only_early_return_still_writes_native_fields(tmp_path, monkeypatch):
    import cupy as cp
    monkeypatch.chdir(tmp_path)
    state, cfg, _ = _state(cp, fmoist_only=True, fire_print_file=2)
    fire = state.physics.fire
    original = cp.asnumpy(fire.grid.data["lfn"])
    for _ in range(2):
        _advance(state, cfg)
    assert fire.grid.step_count == 2 and fire.grid.time_seconds == 0.5
    assert np.array_equal(cp.asnumpy(fire.grid.data["lfn"]).view(np.uint32), original.view(np.uint32))
    assert (tmp_path/"fmc_gc_00000001.txt").stat().st_size == (cfg.nfmc*cfg.ny*cfg.nx+6)*21
    assert (tmp_path/"lfn_00000002.txt").exists()


@requires_gpu
def test_positive_test_steps_run_native_fire_and_intentionally_stop(tmp_path, monkeypatch):
    import cupy as cp
    from gpuwm.core.physics import _prepare_atmosphere
    from gpuwm.sfire_debug import FireDiagnosticComplete
    monkeypatch.chdir(tmp_path)
    state, cfg, _ = _state(cp, fire_test_steps=2, fire_print_file=2)
    expected, expected_cfg, _ = _state(cp)
    frozen = {name: cp.asnumpy(getattr(state,name)) for name in ("u","v","w","p","thp","mup","php","qv")}
    for _ in range(3):
        atm = _prepare_atmosphere(expected)
        driver = expected.physics
        sf = dict(driver.fields, psfc=atm["p_interface"][0], rainc=driver._zero_accumulator(),
                  rainnc=driver.microphysics.rainnc)
        driver.fire._advance_once(expected, expected_cfg, atm, driver.fire.grid.time_seconds,
                                  cfg.dt, sf, feedback=False, report=False)
    with pytest.raises(FireDiagnosticComplete) as stopped:
        _advance(state, cfg)
    assert stopped.value.passes == 3 and stopped.value.time_seconds == 0.75
    assert state.elapsed_seconds == 0.0
    for name, original in frozen.items():
        assert np.array_equal(cp.asnumpy(getattr(state,name)).view(np.uint32), original.view(np.uint32)), name
    for name, original in expected.physics.fire.arrays().items():
        assert np.array_equal(cp.asnumpy(state.physics.fire.arrays()[name]).view(np.uint8),
                              cp.asnumpy(original).view(np.uint8)), name
    assert _level_files(tmp_path) == ["lfn_00000001.txt","lfn_00000002.txt"]


@requires_gpu
def test_tiled_diagnostic_finishes_complete_domain_and_stops_once(tmp_path, monkeypatch):
    import cupy as cp
    from gpuwm.core import streaming as st
    from tilestream import driver
    from gpuwm.sfire_debug import FireDiagnosticComplete
    monkeypatch.chdir(tmp_path)
    state, cfg, _ = _state(cp, fire_test_steps=2, fire_print_file=2)
    st.prime_lazy_carriers(state,cfg)
    decision = st.StreamingDecision(True,"SFIRE diagnostic domain control",12,10,2,16,
                                   store="host",write_mode="shadow")
    stream = st.attach(state,cfg,decision,tile_state_factory=st.prepared_tile_state_factory(state,cfg),
        geography=driver.geography_store(state,host=True),inventory_fn=st.streamed_store_inventory(),
        observe_stability=False)
    frozen = {name: np.asarray(value).copy() for name,value in stream.store.items() if name.startswith("state/")}
    try:
        with pytest.raises(FireDiagnosticComplete) as stopped:
            stream(state,cfg)
        assert stopped.value.passes == 3 and stopped.value.time_seconds == 0.75
        assert stream.scalars["elapsed_seconds"] == 0.0
        assert stream.scalars["fire_clocks"]["step_count"] == 3
        for name, original in frozen.items():
            assert np.array_equal(np.asarray(stream.store[name]).view(np.uint8),original.view(np.uint8)),name
        assert (tmp_path/"lfn_00000002.txt").stat().st_size == (cfg.nx*cfg.ny*cfg.sr_x*cfg.sr_y+6)*21
        assert (tmp_path/"lfn_1_00000002.txt").stat().st_size == (cfg.nx*cfg.ny*cfg.sr_x*cfg.sr_y+6)*21
    finally:
        stream._run.close()
