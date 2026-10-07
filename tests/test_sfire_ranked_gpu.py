"""Actual coupled SFIRE through the public resident rank builder."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu


def _ranked(state, cfg, grid):
    from gpuwm.core import streaming as st
    from gpuwm.core.devices import DeviceOptions
    from tilestream import driver, gather, physics_inventory
    take = st.streamed_store_inventory()
    bundle = SimpleNamespace(store={name: gather.pinned_copy(value)
        for name, value in take(state, None).items()},
        scalars=physics_inventory.carrier_scalars(state),
        geography=driver.geography_store(state, host=True),
        boundaries=None, template=state)
    options = DeviceOptions(count=grid[0]*grid[1], grid=grid, ids=(0,)*(grid[0]*grid[1]))
    return st.ranked_domain_builder(bundle, options=options)(
        None, cfg, st.ranked_decision(cfg, options))


@requires_gpu
@pytest.mark.parametrize("grid,nx,ny", [((1, 2), 96, 72), ((2, 1), 96, 72),
                                       ((1, 2), 97, 73), ((3, 3), 96, 72)])
@pytest.mark.parametrize("spotting,smoke", [(False, False), (True, False), (False, True), (True, True)])
def test_refined_fire_resident_ranks_preserve_full_state_history_and_disk(
        tmp_path, grid, nx, ny, spotting, smoke, record_property):
    import cupy as cp
    from gpuwm.core.dycore import step
    from gpuwm.core.sfire_coupler import FireCoupler
    from gpuwm.core.streaming import streamed_store_inventory
    from gpuwm.io.wrfout import _device_state_frame
    from test_sfire_tilestream_gpu import _state, _assert_words

    def initial():
        if smoke or (nx, ny) != (96, 72):
            from test_restart import _physics_state
            from gpuwm.core.streaming import prime_lazy_carriers
            state, cfg, driver = _physics_state(cp, nx=nx, ny=ny, nz=16,
                dx=90., dy=120., dt=.25, mp_physics=1, cu_physics=0, ra_physics=0,
                ifire=2, sr_x=4, sr_y=3, fire_smoke=smoke, fire_fuel_read=0,
                fire_fuel_cat=1, fire_fmc_read=0, fire_boundary_guard=2,
                fmoist_run=True, fmoist_interp=True, fmoist_freq=1,
                fire_num_ignitions=1, fire_ignition_start_x1=2880.,
                fire_ignition_start_y1=2880., fire_ignition_radius1=160.,
                fire_ignition_ros1=280.)
            prime_lazy_carriers(state, cfg)
        else:
            state, cfg, driver = _state(cp)
        if spotting:
            cfg = replace(cfg, fs_array_maxsize=256, fs_firebrand_gen_lim=7,
                fs_firebrand_gen_dt=2, fs_firebrand_gen_levels=3,
                fs_firebrand_gen_mom3d_dt=1, fs_firebrand_gen_levrand=True,
                trackember=True)
        driver.fire = FireCoupler(state,cfg,driver.fields,{
            "LFN_HIST":cp.full(driver.fire.fine_shape,-17.5,cp.float32),
            "LFN_TIME":cp.asarray([-125.5],cp.float32)})
        return state, cfg

    reference, cfg = initial()
    home, ranked_cfg = initial()
    stream = _ranked(home, ranked_cfg, grid)
    checked_words = checked_arrays = 0
    try:
        if spotting:
            assert len({id(tile.physics.fire.spotting) for tile in stream.tiled_run.tiles}) == 1
        for number in range(8):
            due = (number+1) % 4 == 0
            step(reference, cfg, fire_history_due=due)
            stream(None, ranked_cfg, fire_history_due=due)
            arrays = streamed_store_inventory()(reference, None)
            try:
                _assert_words(cp, arrays, stream.store)
            except AssertionError as exc:
                differences = {name: int(np.count_nonzero(cp.asnumpy(value).view(np.uint32)
                    != np.asarray(stream.store[name]).view(np.uint32)))
                    for name, value in arrays.items()
                    if cp.asnumpy(value).tobytes() != np.asarray(stream.store[name]).tobytes()}
                raise AssertionError(("completed_step", number+1, differences)) from exc
            checked_words += sum(value.nbytes//4 for value in arrays.values())
            checked_arrays += len(arrays)
            native = reference.physics.fire.metadata()
            stored = stream.scalars["fire_header"]["fire"]
            for name in ("time_seconds", "step_count", "moisture_lasttime",
                         "moisture_nexttime", "last_cfl_bound", "last_cfl_exceeded",
                         "last_ignited_counts"):
                assert stored["grid"][name] == native["grid"][name], name
            if spotting:
                assert stored["spotting"] == native["spotting"]
            if number == 3:
                path = stream.write_restart(tmp_path / "ranked-mid.npz", ranked_cfg).path
                stream.tiled_run.close()
                home, ranked_cfg = initial()
                stream = _ranked(home, ranked_cfg, grid)
                stream.restore_restart(path, ranked_cfg)
                native_frame = _device_state_frame(reference)
                restored_frame = stream.history_fields().materialize()
                assert set(native_frame) == set(restored_frame)
                for name, value in native_frame.items():
                    assert cp.asnumpy(value).tobytes() == np.asarray(restored_frame[name]).tobytes(), name
        native = _device_state_frame(reference)
        joined = stream.history_fields().materialize()
        assert set(native) == set(joined)
        for name, value in native.items():
            assert cp.asnumpy(value).tobytes() == np.asarray(joined[name]).tobytes(), name
        record_property("graded_32bit_words", checked_words)
        record_property("graded_arrays", checked_arrays)
        record_property("completed_steps", 8)
        record_property("different_words", 0)
        record_property("rank_grid", str(grid))
        if smoke:
            from gpuwm.core.chem_driver import chem_ledger_receipt
            ledger = chem_ledger_receipt(reference)["rows"]["fire_smoke"]
            assert ledger["emitted_kg"] > 0
            assert abs(ledger["closure_kg"]) < 1e-8
            record_property("emitted_kg", ledger["emitted_kg"])
    finally:
        stream.tiled_run.close()
