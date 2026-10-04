"""Two complete real-source steps and Rust histories against ordinary runs."""

from dataclasses import fields, is_dataclass, replace
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu
from test_ensemble_batch_physics_init_gpu import card_free_of_earlier_tests, prepared_real  # noqa: F401

pytestmark = [pytest.mark.gpu, requires_gpu]

#: The largest card measured too small for the ten-member arm: 16 GiB class.
_TEN_MEMBER_SHORT_CARD_BYTES = 1 << 34


def _driver_arrays(driver):
    """Walk every stock driver's persistent device array, including aliases."""
    import cupy as cp
    found = {}
    def walk(value, path):
        if isinstance(value, cp.ndarray):
            found[path] = value
        elif isinstance(value, dict):
            for name, child in value.items():
                walk(child, f"{path}/{name}")
        elif is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                walk(getattr(value, field.name), f"{path}/{field.name}")
        elif isinstance(value, (tuple, list)):
            for number, child in enumerate(value):
                walk(child, f"{path}/{number}")
    for name, value in driver.__dict__.items():
        if name not in {"state", "radiation_callable", "cumulus_callable", "noah_params"} and not name.startswith("_ensemble_"):
            walk(value, name)
    return found


def _member_array(array, expected, member, members):
    if array.shape == expected.shape:
        return array
    height = expected.shape[-2]
    wanted = (expected.shape[0], members * height, expected.shape[-1]) if expected.ndim == 3 else (members * height, expected.shape[-1])
    assert array.shape == wanted, (array.shape, expected.shape)
    return array[:, member * height:(member + 1) * height] if expected.ndim == 3 else array[member * height:(member + 1) * height]


def _physics_readout(driver, *, member, members, ny, nx):
    from gpuwm.ensemble.batch_health import _member_driver
    readout=_member_driver(driver,member,members,ny,nx)
    published = driver.output_fields()
    def select(array):
        if members == 1 or array.shape[-2:] == (ny, nx):
            return array
        height = array.shape[-2] // members
        return array[:, member * height:(member + 1) * height] if array.ndim == 3 else array[member * height:(member + 1) * height]
    return SimpleNamespace(surface_enabled=driver.surface_enabled,
                           fields={name: select(array) for name, array in driver.fields.items()},
                           scheme_dispatch=driver.scheme_dispatch,noah_params=driver.noah_params,
                           microphysics=readout.microphysics,
                           output_fields=lambda: {name: select(array) for name, array in published.items()})


def _scalar_physics(state, inputs):
    from gpuwm.core.cam_ozone import cam_ozone_setup
    from gpuwm.core.radiation_composition import make_radiation
    from gpuwm.ingest.wrfinput import initialize_wrfinput_physics
    from gpuwm.runtime import declared_constant_glw
    from gpuwm.case_data import trace_gas_overrides_from_config
    exp = inputs.experiment
    domain, bundle, grid = exp.root, inputs.domains[0], inputs.grids[0]
    cam = cam_ozone_setup(exp=exp, dc=domain, grid=grid)
    trace = trace_gas_overrides_from_config(inputs.experiment_config,
                          expected_sha256=inputs.authority_sha256["experiment_config"])
    radiation = make_radiation(domain.run, exp.start_time,
                  bundle.restored.raw["XLAT"], bundle.restored.raw["XLONG"], p_top=state.p_top,
                  column_chunk=exp.column_chunk, trace_gas_overrides=trace)
    return initialize_wrfinput_physics(state, bundle.restored, domain.run, radiation=radiation,
                  radiation_start_time=exp.start_time, radiation_latitude=bundle.restored.raw["XLAT"],
                  radiation_longitude=bundle.restored.raw["XLONG"], landuse=bundle.landuse,
                  constant_glw_wm2=declared_constant_glw(exp), fractional_seaice=bundle.fractional_seaice,
                  cam_ozone=cam)


def _one_member_batch(inputs, source_state, seed):
    import cupy as cp
    from gpuwm.core.clock import resolve_clock
    from gpuwm.core.device_inventory import state_array_shapes
    from gpuwm.ensemble.batch_state import BatchedDomainState, PreparedHostMember, SHARED_STATE_CANDIDATES
    from gpuwm.ensemble.batch_boundaries import MemberBoundaryTables
    from gpuwm.ensemble.batch_dycore import member_domain_view
    from gpuwm.ensemble.batch_moist_dycore import workspace_specs, required_scratch_slots, prepare_moist_step
    from gpuwm.ensemble.batch_perturbation import initialize_member_winds
    cfg = inputs.experiment.root.run
    clock = resolve_clock(inputs.experiment, lbc_interval_s=float(inputs.boundary_interval_seconds)).domain_clock(1)
    names = state_array_shapes(cfg)
    extras = workspace_specs(cfg)
    arrays = {name: getattr(source_state, name).get() for name in names}
    arrays.update({spec.name: np.zeros(spec.shape, spec.dtype) for spec in extras})
    excluded = {"physics", "lateral_boundaries", "_scratch", "_scratch_arena", "_host_setup_state", "_phb_host"}
    scalars = {name: value for name, value in vars(source_state).items()
               if name not in names and name not in excluded and not name.startswith("_lateral")}
    snapshot = {name: getattr(clock, name) for name in ("ticks", "step_ticks", "tick_den", "run_ticks", "step_count", "dt_fp32", "dtbc_fp32")}
    prepared = PreparedHostMember(cfg, arrays, scalars, snapshot,
            scratch={name: value.get() for name, value in source_state._scratch.items()}, phb_host=source_state._phb_host)
    batch = BatchedDomainState.from_prepared((prepared,), array_module=cp, available_bytes=25 << 30,
            shared_fields=tuple(SHARED_STATE_CANDIDATES & names.keys()), extra_specs=extras,
            scratch_slots=required_scratch_slots(cfg))
    tables = MemberBoundaryTables.from_prepared((inputs.boundaries,), cfg, array_module=cp, available_bytes=25 << 30)
    view = member_domain_view(batch, 0)
    driver = _scalar_physics(view, inputs)
    initialize_member_winds(state=batch, cfg=cfg, member_indices=(0,), seeds=(seed,), phase="after_physics_before_step")
    adapter = SimpleNamespace(driver=driver)
    advance = prepare_moist_step(batch, physics_adapter=adapter, tables=tables, boundary_clock=clock)
    def step():
        if clock.lbc_reset_due():
            clock.mark_force()
        clock.prepare_step()
        batch.elapsed_seconds = float(clock.elapsed_seconds_fp32)
        advance(refl_10cm_due=clock.history_rings_within_step())
        clock.advance()
        batch.elapsed_seconds = clock.elapsed_seconds
    return SimpleNamespace(batch=batch, physics=adapter, clock=clock, step=step)


@pytest.mark.parametrize("members", [1, 4, 10])
def test_real_complete_member_graph_and_rust_history_are_byte_identical(prepared_real, members, tmp_path):
    import cupy as cp
    from gpuwm.core import dycore
    from gpuwm.core.clock import resolve_clock
    from gpuwm.core.device_inventory import state_array_shapes
    from gpuwm.core.state import refresh_model_time
    from gpuwm.ingest.wrfinput import restore_domain_state
    from gpuwm.ingest.lateral_bc import attach_lateral_boundaries, bind_lateral_boundary_clock
    from gpuwm.prepared_domain_tree_forecast import _with_terrain_acoustics
    from gpuwm.ensemble.batch_forecast import PreparedRealEnsemble
    from gpuwm.ensemble.batch_perturbation import initialize_member_winds
    from gpuwm.ensemble.batch_dycore import member_domain_view
    from gpuwm.io import nc_writer_bridge
    from gpuwm.io.wrfout import WrfoutWriter, state_frame
    assert nc_writer_bridge.unavailable_reason() is None
    if members == 10:
        # The ten ordinary references stay resident beside the ten-member
        # batch. Measured on the release card (RTX 5070 Ti, 16,611,278,848
        # bytes by the driver): the references run the device out of memory
        # at 14.3 GB allocated, in a fresh process. A 32 GB card holds all of
        # it. Only a card no larger than the one measured short is excused; a
        # larger one runs the arm and reports its own answer.
        total = int(cp.cuda.runtime.memGetInfo()[1])
        if total <= _TEN_MEMBER_SHORT_CARD_BYTES:
            pytest.skip(f"ten members with ten resident ordinary references need more device memory "
                        f"than this card's {total} bytes; the one- and four-member arms carry the identity here")
    inputs = _with_terrain_acoustics(prepared_real)
    # Every step produces and consumes the same stock microphysics-time
    # reflectivity handoff. A second output must not reuse a pending stash.
    domain=replace(inputs.experiment.root,history_interval_s=inputs.experiment.root.run.dt)
    inputs=replace(inputs,experiment=replace(inputs.experiment,domains=(domain,)))
    cfg, bundle = inputs.experiment.root.run, inputs.domains[0]
    assert cfg.specified and not cfg.use_adaptive_time_step
    assert (cfg.mp_physics, cfg.sf_sfclay_physics, cfg.sf_surface_physics, cfg.bl_pbl_physics) == (8, 1, 2, 1)
    seeds = tuple(2026100200 + member for member in range(members))
    source = restore_domain_state(bundle.restored, cfg)
    runtime = (_one_member_batch(inputs, source, seeds[0]) if members == 1 else
               PreparedRealEnsemble(inputs, members=members, member_seeds=seeds, member_initializer=initialize_member_winds))
    del source
    scalar_states, scalar_drivers, scalar_clocks = [], [], []
    for member, seed in enumerate(seeds):
        state = restore_domain_state(bundle.restored, cfg)
        clock = resolve_clock(inputs.experiment, lbc_interval_s=float(inputs.boundary_interval_seconds)).domain_clock(1)
        attach_lateral_boundaries(state, inputs.boundaries)
        bind_lateral_boundary_clock(state, clock)
        driver = _scalar_physics(state, inputs)
        initialize_member_winds(state=state, cfg=cfg, member_indices=(member,), seeds=(seed,), phase="after_physics_before_step")
        scalar_states.append(state)
        scalar_drivers.append(driver)
        scalar_clocks.append(clock)
    names = state_array_shapes(cfg)
    def check_words(step):
        cp.cuda.get_current_stream().synchronize()
        actual_driver_arrays = _driver_arrays(runtime.physics.driver)
        for member, (state, driver) in enumerate(zip(scalar_states, scalar_drivers)):
            for name in names:
                actual = runtime.batch.member_view(name, member).get()
                expected = getattr(state, name).get()
                assert actual.tobytes() == expected.tobytes(), (step, member, name, int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32))))
            expected_arrays = _driver_arrays(driver)
            assert actual_driver_arrays.keys() == expected_arrays.keys(), (step, member)
            for name, expected in expected_arrays.items():
                actual = _member_array(actual_driver_arrays[name], expected, member, members)
                assert actual.get().tobytes() == expected.get().tobytes(), (step, member, "physics", name)
            assert runtime.physics.driver.call_counts == driver.call_counts
            assert runtime.physics.driver.microphysics_updates == driver.microphysics_updates
    check_words(0)
    for step in range(1, 3):
        runtime.step()
        for state, clock in zip(scalar_states, scalar_clocks):
            if clock.lbc_reset_due():
                clock.mark_force()
            clock.prepare_step()
            refresh_model_time(state, clock, kernel_launch=True)
            dycore.step(state, cfg, acoustic=True, refl_10cm_due=clock.history_rings_within_step())
            clock.advance()
            refresh_model_time(state, clock)
        check_words(step)
        valid = inputs.experiment.start_time + timedelta(seconds=runtime.clock.elapsed_seconds)
        from gpuwm.core.refl import consume_refl_10cm
        batched_refl=(consume_refl_10cm(runtime.physics.driver.state) if members==1 else runtime.consume_output_diagnostics())
        for member, state in enumerate(scalar_states):
            view = member_domain_view(runtime.batch, member)
            view.physics = _physics_readout(runtime.physics.driver, member=member, members=members, ny=cfg.ny, nx=cfg.nx)
            paths = [tmp_path / f"batch-{step}-{member}.nc", tmp_path / f"single-{step}-{member}.nc"]
            member_refl=batched_refl if members==1 else batched_refl[:,member*cfg.ny:(member+1)*cfg.ny]
            scalar_refl=consume_refl_10cm(state)
            for path, domain_state,refl in zip(paths, (view, state),(member_refl,scalar_refl)):
                frame = state_frame(domain_state, include_diagnostic_pressure=True)
                frame['REFL_10CM']=cp.asnumpy(refl)
                with WrfoutWriter(path, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, dx=cfg.dx, dy=cfg.dy,
                                  global_attrs={"START_DATE": inputs.experiment.start_time.strftime("%Y-%m-%d_%H:%M:%S"),
                                                "DT": np.float32(cfg.dt)}, field_schema=frame, engine="rust") as writer:
                    writer.write_frame(valid.strftime("%Y-%m-%d_%H:%M:%S"), frame)
            assert paths[0].read_bytes() == paths[1].read_bytes(), (step, member, "Rust history")
