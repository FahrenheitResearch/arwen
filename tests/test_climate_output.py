"""The fork's climate windows, native words, and output ownership."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.config import RunConfig
from gpuwm.core.climate_diagnostics import HISTORY, PERSISTENT, update

ROOT = Path(__file__).resolve().parents[1]
VECTORS = ROOT / "tests/data/oracles/climate_output/vectors.npz"


def test_climate_native_route_round_trip(tmp_path):
    from test_route_round_trip_fields import _emit, _replay
    _, paths = _emit(tmp_path, "climate-windows", output_diagnostics=1,
                     diag_int=30, wind_int=10)
    cfg = _replay(paths).root.run
    assert (cfg.output_diagnostics, cfg.diag_int, cfg.wind_int) == (1, 30, 10)


@pytest.mark.parametrize("key,value", [("output_diagnostics", 2), ("output_diagnostics", True),
                                      ("diag_int", 0), ("diag_int", 1.5),
                                      ("wind_int", -1), ("wind_int", True)])
def test_climate_invalid_windows(key, value):
    from gpuwm.config import validate_run_config
    from test_restart import _cfg
    with pytest.raises(ValueError, match=key):
        validate_run_config(_cfg(**{key: value}))


def test_climate_restart_history_memory_and_stream_inventory(monkeypatch, tmp_path):
    from test_restart import _cfg, _shim_state, _fill_setup
    from gpuwm.io import restart
    from gpuwm.io.history_layout import produced_history_shapes, live_state_history_fields
    from gpuwm.core import preflight
    from gpuwm.core.output_diagnostics import scratch_shapes
    from gpuwm.core.streaming import TRACKER_PLANE_CARRIERS
    cfg = _cfg(output_diagnostics=1)
    source = _shim_state(cfg, monkeypatch)
    _fill_setup(source)
    for index, slot in enumerate(PERSISTENT):
        source.existing_scratch(slot)[...] = index + .25
        assert "scratch/" + slot in TRACKER_PLANE_CARRIERS
    for slot, (symbol, _, _) in HISTORY.items():
        assert live_state_history_fields(source)[symbol] is source.existing_scratch(slot)
        assert produced_history_shapes(cfg)[symbol] == (cfg.ny, cfg.nx)
    path = restart.write_restart(tmp_path / "climate.npz", source, cfg)
    target = _shim_state(cfg, monkeypatch)
    _fill_setup(target)
    restart.restore_restart(path, target, cfg)
    for slot in PERSISTENT:
        np.testing.assert_array_equal(target.existing_scratch(slot), source.existing_scratch(slot))
    for slot, shape in scratch_shapes(cfg).items():
        assert preflight.scratch_slot_registry(cfg)[slot] == shape
        assert preflight.scratch_slot_lifetime(slot).kind == "carrying"
        assert restart.classify_scratch_slot(slot) in ("serialize", "rebuild")


@requires_gpu
def test_every_climate_field_against_unmodified_fortran_at_every_call():
    import cupy as cp
    from gpuwm.core.output_diagnostics import initialize
    from gpuwm.io.history_layout import live_state_history_fields
    from gpuwm.core.uh_diag import reset_up_heli_max
    cfg = RunConfig(nx=3, ny=2, nz=3, dx=3000., dy=3000., ztop=10000.,
                    dt=20., run_seconds=3600., output_diagnostics=1, mp_physics=28)
    arrays = {}
    def scratch(shape, slot):
        if slot not in arrays:
            arrays[slot] = cp.zeros(shape, dtype=cp.float32)
        return arrays[slot]
    state = SimpleNamespace(_scratch=arrays, scratch=scratch, existing_scratch=arrays.get,
                            elapsed_seconds=0., physics=SimpleNamespace(fields={}))
    initialize(state, cfg)
    for slot in HISTORY:
        arrays[slot].fill(-123.25)
    with np.load(VECTORS) as vectors:
        assert tuple(vectors["slots"]) == tuple(HISTORY)
        for call, (inputs, clock) in enumerate(zip(vectors["inputs"], vectors["clock"])):
            dt, xtime, seconds, restart, diag, wind = clock
            cfg = replace(cfg, dt=float(dt), diag_int=int(diag), wind_int=int(wind))
            state.elapsed_seconds = float(seconds)
            state.physics.fields = {name: cp.asarray(value) for name, value in
                                    zip(("t2", "q2", "tsk", "swdown", "swddni", "u10", "v10"), inputs[:7])}
            state.physics.cu_raincv = cp.asarray(inputs[7])
            state.physics.microphysics = SimpleNamespace(rainncv=cp.asarray(inputs[8]))
            update(state, cfg, is_restart=bool(restart))
            got = np.stack([cp.asnumpy(arrays[slot]).view(np.uint32) for slot in HISTORY])
            np.testing.assert_array_equal(got, vectors["expected"][call], err_msg=f"call {call}")
            for index, slot in enumerate(("cl_nsteps", "cl_nsteps_wind")):
                np.testing.assert_array_equal(cp.asnumpy(arrays[slot]).view(np.uint32), vectors["counters"][call, index])
            # History may observe in-progress sums; it never finalizes them.
            before = got.copy()
            reset_up_heli_max(state)
            assert all(symbol in live_state_history_fields(state) for symbol, _, _ in HISTORY.values())
            np.testing.assert_array_equal(np.stack([cp.asnumpy(arrays[s]).view(np.uint32) for s in HISTORY]), before)


def test_climate_kernel_is_in_the_recipe_prewarm_plan():
    from gpuwm.experiment import load_experiment
    from gpuwm.core import preflight
    from gpuwm import kernel_prewarm
    exp = load_experiment(ROOT / "gpuwm/data/recipes/hrrr_v4_gsd41.toml")
    assert "climate_diagnostics" in preflight.physics_kernel_modules(exp)
    assert "climate_diagnostics" in kernel_prewarm.static_modules(exp)


@requires_gpu
def test_climate_is_trajectory_inert_and_async_history_is_a_snapshot(tmp_path):
    import cupy as cp
    from gpuwm.core.dycore import step
    from gpuwm.core.physics import initialize_physics
    from gpuwm.core.refl import consume_refl_10cm
    from test_mp_spec_zone_ring import _square_moist_state
    off,cfg0=_square_moist_state(mp=28,open_x=True,open_y=True,dt=2.,hailcast_opt=0)
    on,cfg1=_square_moist_state(mp=28,open_x=True,open_y=True,dt=2.,hailcast_opt=0,
                              output_diagnostics=1)
    initialize_physics(off,cfg0)
    initialize_physics(on,cfg1)
    for _ in range(3):
        step(off,cfg0,refl_10cm_due=True)
        step(on,cfg1,refl_10cm_due=True)
        consume_refl_10cm(off);consume_refl_10cm(on)
    for name in ('u','v','w','thp','php','mup','p','al','alt','qv','qc','qr','qi','qs','qg','qn','qni','h_diabatic'):
        a,b=getattr(off,name,None),getattr(on,name,None)
        if a is not None:
            np.testing.assert_array_equal(cp.asnumpy(a),cp.asnumpy(b),err_msg=name)
    # Exercise the production asynchronous writer and Rust NetCDF path,
    # including every climate plane. Mutating
    # the live accumulators after submit must not change the snapshot.
    from gpuwm.io.wrfout import AsyncDomainWrfoutWriter
    from datetime import datetime
    from netCDF4 import Dataset
    inventory=HISTORY
    expected={symbol:cp.asnumpy(on.existing_scratch(slot))
              for slot,(symbol,_,_) in inventory.items()}
    path=tmp_path/'diagnostic-snapshot.nc'
    writer=AsyncDomainWrfoutWriter(nx=cfg1.nx,ny=cfg1.ny,nz=cfg1.nz,
        dx=cfg1.dx,dy=cfg1.dy,title='output-control-snapshot',global_attrs={})
    try:
        writer.submit(path,datetime(2026,10,21,0,0,6),on)
        for slot in inventory:
            on.existing_scratch(slot)[...] = -999.
    finally:
        writer.close()
    with Dataset(path) as nc:
        for symbol,values in expected.items():
            np.testing.assert_array_equal(nc.variables[symbol][0],values,err_msg=symbol)


@requires_gpu
def test_climate_windows_survive_pinned_host_tile_reuse():
    import cupy as cp
    from tilestream import gather, physics_inventory, spec
    from gpuwm.core.climate_diagnostics import scratch_shapes
    cfg = RunConfig(nx=8, ny=6, nz=3, dx=3000., dy=3000., ztop=10000.,
                    dt=20., run_seconds=240., output_diagnostics=1,
                    diag_int=1, wind_int=2, mp_physics=28)
    names = ("t2", "q2", "tsk", "swdown", "swddni", "u10", "v10", "rainc", "rain")
    def allocate(c):
        return {**{name: cp.zeros((c.ny, c.nx), cp.float32) for name in names},
                **{"scratch/"+slot: cp.zeros(shape, cp.float32)
                   for slot, shape in scratch_shapes(c).items()}}
    def view(arrays, seconds):
        return SimpleNamespace(elapsed_seconds=seconds,
            existing_scratch=lambda slot: arrays.get("scratch/"+slot),
            physics=SimpleNamespace(fields={name: arrays[name] for name in names[:7]},
                cu_raincv=arrays["rainc"], microphysics=SimpleNamespace(rainncv=arrays["rain"])))
    resident = allocate(cfg)
    store = {name: gather.pinned_copy(value) for name, value in resident.items()}
    tile_cfg = replace(cfg, nx=6, ny=5)
    tile = allocate(tile_cfg)
    tiles = spec.plan_tiles(cfg.nx, cfg.ny, 4, 3, halo=1, periodic=True)
    inv = physics_inventory.carrier_inventory
    for call in range(12):
        for index, name in enumerate(names):
            values = (np.arange(48, dtype=np.float32).reshape(6, 8) + index + call % 5) / 16
            resident[name].set(values)
            store[name][...] = values
        seconds = call * cfg.dt
        update(view(resident, seconds), cfg)
        for geometry in tiles:
            gather.gather_tile(store, tile, geometry, inventory_fn=inv, nz=cfg.nz)
            update(view(tile, seconds), tile_cfg)
            gather.scatter_tile(tile, store, geometry, inventory_fn=inv, nz=cfg.nz)
        cp.cuda.get_current_stream().synchronize()
        for slot in PERSISTENT:
            np.testing.assert_array_equal(store["scratch/"+slot].view(np.uint32),
                cp.asnumpy(resident["scratch/"+slot]).view(np.uint32), err_msg=f"{call}: {slot}")
