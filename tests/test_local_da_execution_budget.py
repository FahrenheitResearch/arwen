"""Runtime scratch scheduling leaves the reviewed experiment unchanged."""
import copy
import json
from contextvars import copy_context

import numpy as np
import pytest

from gpuwm import local_da_runtime as runtime
from gpuwm.da import radar_assimilation as radar
from gpuwm.ensemble.cycle import _method_identity
from test_local_da_runtime import Backend, saved
from test_radar_assimilation import grid, world, _config


GIB = 1 << 30


@pytest.mark.parametrize('device_gib,expected_mib', [(12, 3072), (28, 5120)])
def test_auto_scratch_uses_available_card_and_host_capacity(device_gib, expected_mib):
    result, receipt = runtime.analysis_execution_budget(256, capacity=dict(
        driver_free_bytes=(device_gib-1)*GIB, pool_reusable_bytes=GIB,
        host_available_bytes=40*GIB))
    assert result == expected_mib
    assert receipt['planned_scratch_mib'] == 256
    assert receipt['scientific_settings_changed'] is False


def test_explicit_execution_budget_remains_exact():
    result, receipt = runtime.analysis_execution_budget(256, override_mib=1536.5,
        capacity=dict(driver_free_bytes=28*GIB, pool_reusable_bytes=0,
                      host_available_bytes=40*GIB))
    assert result == 1536.5
    assert receipt['override_environment'] == runtime.SCRATCH_ENV


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf', 'many', ''])
def test_invalid_execution_override_refuses_before_launch_writes(value, tmp_path, monkeypatch):
    path, _ = saved(tmp_path)
    before = {str(p): p.read_bytes() for p in path.parent.rglob('*') if p.is_file()}
    monkeypatch.setenv('GPUWM_LOCAL_DA_SCRATCH_MIB', value)
    with pytest.raises(ValueError, match='finite positive'):
        runtime.launch(path, backend=Backend())
    after = {str(p): p.read_bytes() for p in path.parent.rglob('*') if p.is_file()}
    assert before == after


def test_unknown_resource_readings_retain_recorded_fallback():
    result, receipt = runtime.analysis_execution_budget(384, capacity={})
    assert result == 384
    assert receipt['driver_free_bytes'] is None
    assert 'unavailable' in receipt['basis']


def test_scope_restores_outer_settings_and_does_not_touch_direct_solver_budget():
    from gpuwm.da.letkf import LetkfConfig, Localization
    cfg = LetkfConfig(Localization(3000., 2000.), ('u',), .4, memory_budget_mib=128)
    outer = copy_context()
    assert radar._execution_settings(128, None) == (128, None, None)
    callback = lambda value: None
    with radar.analysis_execution_options(scratch_budget=lambda value: (4096, {'chosen': 4096}), progress=callback):
        assert radar._execution_settings(128, None) == (4096, callback, {'chosen': 4096})
        assert outer.run(radar._execution_settings, 128, None) == (128, None, None)
        assert cfg.memory_budget_mib == 128
        with pytest.raises(RuntimeError):
            with radar.analysis_execution_options(scratch_budget=lambda value: (2048, {})):
                raise RuntimeError('interrupted')
        assert radar._execution_settings(128, None)[0] == 4096
    assert radar._execution_settings(128, None) == (128, None, None)


def test_real_launch_scope_records_budget_progress_and_preserves_saved_review(tmp_path, monkeypatch):
    path, plan = saved(tmp_path)
    original = path.read_bytes()
    original_plan = copy.deepcopy(plan)
    monkeypatch.setenv('GPUWM_LOCAL_DA_SCRATCH_MIB', '4096')
    if hasattr(runtime, 'analysis_execution_budget'):
        resolve = runtime.analysis_execution_budget
        monkeypatch.setattr(runtime, 'analysis_execution_budget',
            lambda planned_mib, override_mib=None: resolve(planned_mib,
                override_mib=override_mib, capacity=dict(driver_free_bytes=28*GIB,
                    pool_reusable_bytes=0, host_available_bytes=40*GIB)))
    class ScheduledBackend(Backend):
        def assimilate(self, cycle_index, member_states):
            # The existing adapter is the scope consumer. The base has no
            # runtime scope, so it retains the former planned scratch.
            setting = getattr(radar, '_execution_settings', lambda value, progress: (value, progress, None))
            budget, progress, receipt = setting(256, None)
            assert budget == 4096
            assert receipt['memory_budget_mib'] == 4096
            progress(dict(schema='gpuwm-da.analysis-progress.v1', phase='solve',
                gridpoints_done=12, gridpoints_total=18, chunks=2, active_points=9,
                elapsed_seconds=.25))
            return super().assimilate(cycle_index, member_states)
    method_before = _method_identity(runtime.PreparedBackend.assimilate)
    backend = ScheduledBackend()
    result = runtime.launch(path, backend=backend)
    assert result['analysis_execution']['memory_budget_mib'] == 4096
    assert result['analysis_progress']['gridpoints_done'] == 12
    assert result['analysis_progress']['chunks'] == 2
    assert json.loads((path.parent/'execution.json').read_text())['analysis_progress'] == result['analysis_progress']
    assert path.read_bytes() == original
    assert plan == original_plan
    assert _method_identity(runtime.PreparedBackend.assimilate) == method_before
    # Completed analysis recovery returns its original decision without
    # invoking the scheduler or recomputing under a new scratch setting.
    monkeypatch.setenv('GPUWM_LOCAL_DA_SCRATCH_MIB', '2048')
    resumed = ScheduledBackend()
    assert runtime.launch(path, backend=resumed)['status'] == 'COMPLETE'
    assert not [c for c in resumed.calls if c[0] in ('analysis', 'member')]


def test_actual_adapter_applies_scoped_budget_and_relays_progress(world, grid, monkeypatch):
    settings = _config(memory_budget_mib=128)
    messages = []
    def solver(prior, batches, geometry, config, diagnostics, **options):
        assert config.memory_budget_mib == 4096
        assert config.localization == settings.localization
        assert config.analysis_fields == settings.analysis_fields
        options['progress']({'phase': 'solve', 'gridpoints_done': 12})
        return {name: np.zeros_like(value) for name, value in prior.items()}
    solver.supports_host_staging = True
    monkeypatch.setattr(radar, 'analyze', solver)
    paths = {member: radar.member_background_checkpoint(state['member_dir'])
             for member, state in world.member_states.items()}
    with radar.analysis_execution_options(scratch_budget=lambda hint: (4096, {'selected': 4096}),
                                          progress=messages.append):
        _, receipt = radar.assimilate_radar_grid(paths, world.obs_path, grid, settings)
    assert settings.memory_budget_mib == 128
    assert receipt['analysis_execution'] == {'selected': 4096}
    assert messages == [{'phase': 'solve', 'gridpoints_done': 12}]
