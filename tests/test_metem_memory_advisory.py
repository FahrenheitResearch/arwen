"""Estimated capacity preserves metgrid requests; actual failures still surface."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from gpuwm import metem_door, metem_forecast
from gpuwm.core import preflight


def _case(tmp_path, monkeypatch, *, device_low, host_low):
    from test_prepared_document_recovery import _source
    text, report, exp, namelist, controls = _source(tmp_path, monkeypatch)
    metadata, paths = {}, {}
    for domain in exp.domains:
        cfg = domain.run
        path = tmp_path / f'met_em.d{domain.grid_id:02d}.nc'
        path.write_bytes(b'preserved input fixture')
        variables = {name: (1, 4, cfg.ny, cfg.nx)
                     for name in ('TT', 'GHT', 'RH')}
        variables.update(UU=(1, 4, cfg.ny, cfg.nx + 1),
                         VV=(1, 4, cfg.ny + 1, cfg.nx),
                         PSFC=(1, cfg.ny, cfg.nx), SOILHGT=(1, cfg.ny, cfg.nx))
        metadata[domain.grid_id] = SimpleNamespace(path=path,
            global_attributes={}, variables=variables, nx=cfg.nx, ny=cfg.ny)
        paths[domain.grid_id] = (path, path)
    run = SimpleNamespace(toml_text=text, experiment=exp, namelist_input=namelist,
                          substitution_report=report, controls=controls,
                          interval_seconds=3600, metadata=metadata, paths=paths)
    phases = SimpleNamespace(peak_envelope_bytes=1024, forecast_envelope_bytes=512,
                             ingest_envelope_bytes=1024,
                             streamed=SimpleNamespace(host_bytes=1024,
                                 host_budget_bytes=0 if host_low else 2048),
                             verdict=lambda free: 'estimated peak exceeds free memory')
    monkeypatch.setattr(preflight, 'device_memory_probe_subprocess',
                        lambda: dict(free_bytes=0 if device_low else 2048))
    monkeypatch.setattr(preflight, 'profile_from_device_probe', lambda probe: None)
    monkeypatch.setattr(preflight, 'estimate_phases', lambda *args, **kw: phases)
    monkeypatch.setattr(metem_forecast, 'resolve_metem_vertical',
                        lambda run, text, **kw: (text, 'explicit', None))
    return run


@pytest.mark.parametrize('device_low,host_low', [(True, False), (False, True), (True, True)])
def test_estimated_capacity_does_not_refuse_or_reduce_request(tmp_path, monkeypatch, capsys,
                                                             device_low, host_low):
    from gpuwm.ingest import preprocess_backend
    run = _case(tmp_path, monkeypatch, device_low=device_low, host_low=host_low)
    before = deepcopy(run)
    receipt = metem_door.metgrid_memory_admission(run, run.experiment)
    assert receipt['policy'] == 'advisory'
    assert receipt['device_estimate_exceeds_available'] is device_low
    assert receipt['host_estimate_exceeds_available'] is host_low
    assert len(receipt['warnings']) == int(device_low) + int(host_low)
    assert receipt['forecast_peak_bytes'] == 512
    assert receipt['preparation_peak_bytes'] == 1024
    assert run == before

    # Continue through the actual preparation entry point to its operation
    # boundary. A real operation's error must propagate, not become a capacity
    # refusal or a successful preparation. No GPU allocation runs in this test.
    failure = MemoryError('native preparation allocation failed')
    reached = []
    def fail_backend(*args, **kwargs):
        reached.append((args, kwargs))
        raise failure
    monkeypatch.setattr(preprocess_backend, 'resolve_preprocess_backend', fail_backend)
    with pytest.raises(MemoryError) as raised:
        metem_forecast.prepare_metem_run(run, tmp_path / 'prepared')
    assert raised.value is failure and len(reached) == 1
    assert run == before
    assert not (tmp_path / 'prepared').exists()
    output = capsys.readouterr().out
    assert 'requested settings are retained' in output
    assert ('VRAM' in output) is device_low
    assert ('host RAM' in output) is host_low


def test_capacity_fit_has_no_advisory(tmp_path, monkeypatch):
    run = _case(tmp_path, monkeypatch, device_low=False, host_low=False)
    receipt = metem_door.metgrid_memory_admission(run, run.experiment)
    assert receipt['policy'] == 'advisory'
    assert receipt['warnings'] == []
    assert not receipt['device_estimate_exceeds_available']
    assert not receipt['host_estimate_exceeds_available']


@pytest.mark.parametrize('invalid', ['shape', 'number_flag'])
def test_actual_data_constraints_still_refuse_before_preparation(tmp_path, monkeypatch, invalid):
    run = _case(tmp_path, monkeypatch, device_low=True, host_low=True)
    if invalid == 'shape':
        run.metadata[1].variables['TT'] = (1, 2, 3)
        message = 'TT needs Time/level/y/x inventory'
    else:
        run.metadata[1].global_attributes['FLAG_QNI'] = 2
        message = 'FLAG_QNI must be 0 or 1'
    with pytest.raises(ValueError, match=message):
        metem_forecast.prepare_metem_run(run, tmp_path / 'prepared')
    assert not (tmp_path / 'prepared').exists()


@pytest.mark.parametrize('requested,expected', [(None, 'cpu'), ('cpu', 'cpu'),
                                               ('cuda', 'cuda'), ('auto', 'auto')])
def test_metem_estimate_and_preparation_use_one_policy(tmp_path, monkeypatch, requested, expected):
    from dataclasses import replace
    from gpuwm.core.streaming import StreamingOptions
    from gpuwm.ingest import preprocess_backend
    run = _case(tmp_path, monkeypatch, device_low=True, host_low=True)
    run.experiment = replace(run.experiment, tiles=StreamingOptions(mode='on', store='host'))
    run.toml_text += '\n[tiles]\nmode = "on"\nstore = "host"\n'
    calls = []
    phases = preflight.estimate_phases()
    monkeypatch.setattr(preflight, 'estimate_phases',
                        lambda *a, **kw: calls.append(kw) or phases)
    failure = MemoryError('actual backend allocation failure')
    def stop_at_backend(selector, **kw):
        assert selector == expected
        raise failure
    monkeypatch.setattr(preprocess_backend, 'resolve_preprocess_backend', stop_at_backend)
    with pytest.raises(MemoryError) as raised:
        metem_forecast.prepare_metem_run(run, tmp_path/'prepared', preprocess_backend=requested)
    assert raised.value is failure
    assert calls[-1]['preprocess_backend'] == expected
    assert not (tmp_path/'prepared').exists()


def test_metem_auto_estimate_uses_the_existing_device_probe(tmp_path, monkeypatch):
    from gpuwm.core import streaming
    run = _case(tmp_path, monkeypatch, device_low=False, host_low=False)
    observed = []
    machine = object()
    def planner(**kw):
        observed.append(kw)
        return machine
    monkeypatch.setattr(streaming, 'planner_machine', planner)
    phases = preflight.estimate_phases()
    estimates = []
    monkeypatch.setattr(preflight, 'estimate_phases',
                        lambda *a, **kw: estimates.append(kw) or phases)
    receipt = metem_door.metgrid_memory_admission(run, run.experiment)
    assert receipt['policy'] == 'advisory'
    assert observed[-1]['vram_bytes'] == 2048
    assert estimates[-1]['machine'] is machine
