"""Single-domain adaptive execution keeps initialization and I/O authorities."""
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm import runtime
from gpuwm.core import streaming
from gpuwm.core.clock import resolve_clock
from test_runtime import _fixture_pair


def _experiment(tmp_path):
    exp, data = _fixture_pair(tmp_path)
    run = replace(exp.root.run, dt=3., use_adaptive_time_step=True,
                  starting_time_step=6, min_time_step=1, max_time_step=8,
                  max_step_increase_pct=20, step_to_output_time=True)
    domain = replace(exp.root, run=run, time_step=3, history_interval_s=15.)
    return replace(exp, domains=(domain,), run_seconds=30.,
                   restart_interval_s=15.), data


@pytest.mark.parametrize('stored', [False, True])
def test_existing_initialization_binds_shared_clock_without_reallocation(tmp_path, stored):
    exp, _ = _experiment(tmp_path)
    state = SimpleNamespace(_lateral_boundary_device=SimpleNamespace(
        rolling=False, clock=None))
    bundle = object() if stored else None
    case = runtime.PreparedRealCase(
        cfg=exp.root.run, grid=object(), static_fields={},
        initial_result=SimpleNamespace(state=state, initial_perturbation=None),
        final_analysis=None, initial_snow_water_kgm2=np.zeros((2, 2)),
        forcing_times=(exp.start_time, exp.start_time+timedelta(hours=1)),
        streamed_store=bundle,
        initialization_receipt={'forcing_clock': 'elapsed_seconds'} if stored else None)
    calendar = resolve_clock(exp, lbc_interval_s=3600.)
    model = runtime._model_from_prepared_single(exp, case, calendar, 'a'*64)
    assert model.root.state is state
    assert model.root.grid is case.grid
    assert model.root._started
    assert model.schedule.clock is calendar
    assert model.experiment_fingerprint == 'a'*64
    assert model.memory_ledger is None
    bound = model._prepared_by_grid_id[exp.root.grid_id]
    assert bound.streamed_store is bundle
    if stored:
        assert bound.initialization_receipt['forcing_clock'] == 'DomainClock'
        assert case.initialization_receipt['forcing_clock'] == 'elapsed_seconds'
    else:
        assert state._lateral_boundary_device.clock is model.root.clock


def test_adaptive_halo_refines_actual_fp32_geometry_with_same_cold_budget(tmp_path, monkeypatch):
    exp, _ = _experiment(tmp_path)
    run = replace(exp.root.run, nx=512, ny=512, dx=1000., dy=1000., max_time_step=60)
    maps = [np.array([[.9, 1.234567891]], np.float64),
            np.array([[1.8, 2.234567891]], np.float64)]
    case = SimpleNamespace(cfg=run, grid=SimpleNamespace(
        mapfac_u=lambda: maps[0], mapfac_v=lambda: maps[1]))
    options = streaming.StreamingOptions(mode='on', tile_nx=16, tile_ny=17)
    cold = streaming.decide(run, options)
    actual_decide = streaming.decide
    machine = object()
    def decide(cfg, resolved, *, machine):
        assert machine is cold_machine
        return actual_decide(cfg, resolved)
    cold_machine = machine
    monkeypatch.setattr(streaming, 'decide', decide)
    resolved, decision = runtime._refine_single_streaming_plan(
        case, options, cold, machine)
    assert resolved.acoustic_map_factor == float(np.float32(maps[1].max()))
    assert decision.halo > cold.halo
    assert (decision.tile_nx, decision.tile_ny) == (16, 17)
    assert options.acoustic_map_factor is None
    assert decision.detail['acoustic_envelope']['geometry_status'] == 'resolved'


def test_fixed_step_geometry_keeps_original_decision_without_a_probe(tmp_path):
    exp, _ = _experiment(tmp_path)
    cfg = replace(exp.root.run, use_adaptive_time_step=False)
    case = SimpleNamespace(cfg=cfg)  # deliberately no grid to query
    options = streaming.StreamingOptions(mode='on', tile_nx=8, tile_ny=8)
    decision = streaming.decide(cfg, options)
    actual = runtime._refine_single_streaming_plan(case, options, decision, None)
    assert actual[0] is options
    assert actual[1] is decision


def test_store_restart_setup_retains_actual_bound_clock_before_stepping(monkeypatch):
    from tilestream import restart_stream
    from gpuwm.io.restart import root_external_lbc_clock_identity, ROOT_EXTERNAL_LBC_CLOCK_IDENTITY
    clock = object()
    mirror = SimpleNamespace(rolling=False, clock=clock)
    state = SimpleNamespace(_lateral_boundary_device=mirror)
    stream = object.__new__(streaming.StreamedDomain)
    stream._state = None
    stream._setup = None
    stream._geography = object()
    stream._boundaries = object()
    stream._run = SimpleNamespace(tiles=[state])
    def assemble(geography, template, **kwargs):
        assert kwargs['lateral_boundary_device'] is mirror
        assert kwargs['lateral_boundaries'] is stream._boundaries
        return SimpleNamespace(_lateral_boundary_device=kwargs['lateral_boundary_device'])
    monkeypatch.setattr(restart_stream, 'domain_setup_from_stream', assemble)
    setup = stream.restart_setup()
    assert root_external_lbc_clock_identity(setup, SimpleNamespace(specified=True)) == ROOT_EXTERNAL_LBC_CLOCK_IDENTITY


@pytest.mark.parametrize('fits_resident', [False, True])
def test_live_halo_refinement_cannot_change_the_cold_resident_choice(
        tmp_path, monkeypatch, fits_resident):
    from tilestream.autoplan import Machine

    exp, _ = _experiment(tmp_path)
    cfg = replace(exp.root.run, nx=512, ny=512, nz=49,
                  dx=8000., dy=8000., max_time_step=8)
    options = streaming.StreamingOptions(mode='auto')
    fp = streaming.radiation_footprint(cfg, options)
    resident = fp.resident_bytes(cfg.nx * cfg.ny * cfg.nz)
    machine = Machine(vram_bytes=int(resident * (1.1 if fits_resident else .8)),
                      host_bytes=100 * 1024**3, vram_headroom=0.,
                      pinned_fraction=1.)
    monkeypatch.setattr(Machine, 'detect', lambda **kwargs:
                        pytest.fail('warm device budget was read'))
    before = streaming.decide(cfg, options, machine=machine)
    case = SimpleNamespace(cfg=cfg, grid=SimpleNamespace(
        mapfac_u=lambda: np.array([[1.]], dtype=np.float32),
        mapfac_v=lambda: np.array([[2.]], dtype=np.float32)))
    _, after = runtime._refine_single_streaming_plan(case, options, before, machine)
    assert before.stream is (not fits_resident)
    assert after.stream == before.stream
    assert after.budget_bytes == before.budget_bytes
    assert after.detail['acoustic_envelope']['maximum_map_factor'] == 2.
