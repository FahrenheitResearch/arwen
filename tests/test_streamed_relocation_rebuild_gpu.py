"""Whole-domain relocation operators versus bounded reconstruction phases."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.gpu
def test_reserved_physics_reconstruction_and_post_transplant_match_full(monkeypatch):
    import cupy as cp
    from test_nest_window_gpu import _real_window_case
    from gpuwm.core.nest_relocation import transplant_overlap
    from gpuwm.core.streaming import prime_lazy_carriers, streamed_store_inventory
    from gpuwm.ingest.nest_init import seed_rk_time_t_copies
    from gpuwm.ingest.reconstruction_store import (
        ReconstructionReservation, rederive_reconstructed_store, store_from_reconstruction)
    from gpuwm.runtime import PreparedTreeRelocationChildPreparer
    from tilestream.driver import geography_inventory
    from tilestream.physics_inventory import carrier_scalars, set_carrier_scalars
    from tilestream import test_nest_executor as executor

    initialize, moved, parent = _real_window_case(monkeypatch, size=72, complete_statics=True)
    old_dc = replace(moved, i_parent_start=moved.i_parent_start-1)
    old = initialize(old_dc, parent)
    exp = executor.experiment(parent.cfg.run, old_dc.run, steps=2)
    model = SimpleNamespace(_prepared_by_grid_id={2: SimpleNamespace(static_fields=old.static_fields)})
    preparer = PreparedTreeRelocationChildPreparer(exp=exp, model=model)
    preparer._rebuild_driver(old, old_dc, parent, {})
    prime_lazy_carriers(old.state, old_dc.run)
    old.state.elapsed_seconds = 42.
    old.state.physics.microphysics_updates = 9
    take = streamed_store_inventory()
    source = {key: cp.asnumpy(value) for key, value in take(old.state).items()}
    # Deliberately nonzero held rates and timer memory must survive slicing.
    for key, value in source.items():
        if key.startswith("pbl/") or key in ("scratch/cu_nca", "driver/rthratenlw"):
            value[...] = np.float32(17. if key == "scratch/cu_nca" else 1.e-5)
    scalars = carrier_scalars(old.state)
    stream = SimpleNamespace(store=source, template=old.state, scalars=scalars)
    old.state._streamed_domain = stream
    old_node = SimpleNamespace(cfg=old_dc, state=old.state)
    preparer.capture_outgoing(old_node)
    captured = deepcopy(preparer._captured)
    full = initialize(moved, parent)
    preparer(full, moved, parent)
    assert full.state.physics.microphysics_updates == 9
    assert carrier_scalars(full.state) == scalars
    # The store is authoritative for every outgoing prognostic too.
    old_host = SimpleNamespace(**{key[6:]: value for key, value in source.items() if key.startswith("state/")})
    for key, value in geography_inventory(old.state).items():
        if key.startswith("setup/"):
            setattr(old_host, key[6:], cp.asnumpy(value))
    from gpuwm.ingest.relocation_continuation import stage_relocation_continuation
    staged = stage_relocation_continuation(captured, moved, full.static_fields)
    transplant_overlap(source_state=old_host, target_state=full.state, plan=staged.plan)
    initialize.post_transplant(source_state=old_host, target_state=full.state, plan=staged.plan)
    seed_rk_time_t_copies(full.state)
    full.state.physics.recouple_after_relocation(full.state, moved.run)
    prime_lazy_carriers(full.state, moved.run)
    expected = {key: cp.asnumpy(value) for key, value in take(full.state).items()}

    preparer._captured = captured
    footprint = initialize.prepare_footprint(moved, parent)
    prepare = preparer.prepare_windows(moved, parent, footprint)
    reservation = ReconstructionReservation(192*1024*1024)
    bundle = store_from_reconstruction(
        lambda dc, node, **kw: initialize(dc, node, footprint=footprint, **kw), moved, parent,
        rows_per_slab=12, device_budget_bytes=reservation.budget_bytes,
        host_budget_bytes=512*1024*1024, prepare=prepare, reservation=reservation)
    incoming_host = SimpleNamespace(**{key[6:]: value for key, value in bundle.store.items() if key.startswith("state/")})
    transplant_overlap(source_state=old_host, target_state=incoming_host, plan=staged.plan)
    receipt = rederive_reconstructed_store(bundle, moved.run, reservation=reservation,
                                           tile_nx=22, tile_ny=17)
    assert receipt["forecast_steps"] == 0
    assert bundle.scalars == scalars
    assert set(bundle.store) == set(expected)
    for key in expected:
        np.testing.assert_array_equal(bundle.store[key], expected[key], err_msg=key)


@pytest.mark.gpu
@pytest.mark.parametrize("scheduled", [False, True])
def test_live_bounded_move_keeps_owner_and_continues_exactly(monkeypatch, tmp_path, scheduled):
    import gc
    import cupy as cp
    from test_nest_window_gpu import _real_window_case
    from gpuwm.core.model import execute_experiment
    from gpuwm.core.nest_relocation import relocate_child
    from gpuwm.core.streamed_relocation import StreamedChildReconstruction
    from gpuwm.core.streaming import prime_lazy_carriers, streamed_store_inventory
    from gpuwm.runtime import PreparedTreeRelocationChildPreparer
    from tilestream import test_nest_executor as executor, test_streamed_child
    from tilestream.physics_inventory import carrier_scalars

    monkeypatch.setattr(executor, "PARENT_DT", 6.)
    def run(bounded):
        initialize, moved, parent, model = _real_window_case(
            monkeypatch, size=72, complete_statics=True, return_model=True,
            forecast_steps=3 if scheduled else 2)
        child = model.node(2)
        old = initialize(child.cfg, parent)
        child.state, child.grid = old.state, old.grid
        model._prepared_by_grid_id = {2: SimpleNamespace(static_fields=old.static_fields)}
        exp = executor.experiment(parent.cfg.run, child.cfg.run, steps=2)
        preparer = PreparedTreeRelocationChildPreparer(exp=exp, model=model)
        preparer._rebuild_driver(old, child.cfg, parent, {})
        child.state.physics.microphysics_updates = 9
        prime_lazy_carriers(child.state, child.cfg.run)
        steppers = test_streamed_child.stream_child(model) if bounded else {}
        stream = steppers.get(2)
        before_owner = None if stream is None else stream.tiled_run
        preparer.capture_outgoing(child)
        reconstruction = (None if stream is None else StreamedChildReconstruction(
            stream, initializer=initialize, preparer=preparer,
            device_budget_bytes=256*1024*1024, host_budget_bytes=512*1024*1024,
            rows_per_slab=12))
        receipt = relocate_child(child, i_parent_start=moved.i_parent_start,
            j_parent_start=moved.j_parent_start, initializer=initialize,
            static_provenance=initialize.static_provenance, on_child_built=preparer,
            staging="host", reconstruction=reconstruction)
        preparer.after_move(child)
        if not bounded:
            prime_lazy_carriers(child.state, child.cfg.run)
        assert receipt["parent_bitwise_unchanged"]
        if bounded:
            assert steppers[2] is stream
            assert stream.tiled_run is not before_owner and before_owner.closed
            assert child.state._streamed_domain is stream
            assert stream.template.p.shape[1] == 12
            assert receipt["child_rebuild"]["post_transplant"]["forecast_steps"] == 0
        runner = None
        if scheduled:
            from gpuwm.core.relocation_runner import RelocationRunner
            from gpuwm.experiment import RelocationConfig, ScheduledRelocationMove
            runner = RelocationRunner(
                config=RelocationConfig(enabled=True, grid_id=2,
                    moves=(ScheduledRelocationMove(6., dj_parent_cells=1),
                           ScheduledRelocationMove(12., di_parent_cells=-1))),
                schedule=model.schedule, on_child_built=preparer, initializer=initialize,
                static_provenance=initialize.static_provenance, staging="host")
            if bounded:
                runner.streamed_reconstruction_factory = lambda node, **kw: StreamedChildReconstruction(
                    stream, device_budget_bytes=256*1024*1024,
                    host_budget_bytes=512*1024*1024, rows_per_slab=12, **kw)
        result = execute_experiment(model, steppers=steppers, relocation_runner=runner, validate_state=True,
                                    pool_trim_per_period=False)
        if scheduled:
            assert runner.moves_executed == 2
            if bounded:
                assert stream._reconstruction_reservation is reconstruction.reservation
                assert stream._reconstruction_reservation.pool.total_bytes() == 256*1024*1024
        if not bounded:
            prime_lazy_carriers(child.state, child.cfg.run)
        take = streamed_store_inventory()
        source = take(child.state) if stream is None else stream.store
        arrays = {key: (value.copy() if isinstance(value, np.ndarray) else cp.asnumpy(value))
                  for key, value in source.items()}
        scalars = deepcopy(carrier_scalars(child.state) if stream is None else stream.scalars)
        parent_arrays = {key: cp.asnumpy(value) for key, value in take(parent.state).items()}
        from gpuwm.io import restart, wrfout
        path = tmp_path / ("bounded.npz" if bounded else "resident.npz")
        if stream is None:
            frame = wrfout._device_state_frame(child.state)
            restart.write_restart(path, child.state, child.cfg.run)
        else:
            frame = stream.history_fields()
            stream.write_restart(path, child.cfg.run)
        frame = {key: (value.copy() if isinstance(value, np.ndarray) else cp.asnumpy(value))
                 for key, value in frame.items()}
        header, payload = restart._load_restart(path, with_arrays=True)
        header.pop("created")
        if stream is not None:
            stream.tiled_run.close()
        return arrays, scalars, parent_arrays, int(result.steps), int(child.coupler.force_count), frame, header, payload

    reference = run(False)
    gc.collect()
    actual = run(True)
    assert actual[1] == reference[1]
    assert actual[3:5] == reference[3:5]
    assert actual[6] == reference[6]
    assert actual[4] > 0
    for index in (0, 2, 5, 7):
        assert set(actual[index]) == set(reference[index])
        for key in reference[index]:
            np.testing.assert_array_equal(actual[index][key], reference[index][key], err_msg=key)
