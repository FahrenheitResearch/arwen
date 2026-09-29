"""Preparation retries preserve every previously published authority."""
from datetime import timedelta
import json
from types import SimpleNamespace
import tomllib

import numpy as np
import pytest


def _snapshot(root):
    return {p: (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}


def _source(tmp_path, monkeypatch):
    from test_namelist_import import _pair
    from gpuwm.namelist_import import import_namelists, parse_namelist_text
    from gpuwm.experiment import build_experiment
    namelist, wps = _pair(tmp_path)
    text, report = import_namelists(namelist, wps, metgrid_initialization=True)
    exp = build_experiment(tomllib.loads(text), source='preparation fixture')
    return text, report, exp, namelist, parse_namelist_text(namelist.read_text())


@pytest.mark.parametrize('changed', [False, True])
def test_metgrid_preparation_keeps_prior_authorities_on_later_failure(tmp_path, monkeypatch, changed):
    from gpuwm import metem_door, metem_forecast
    from gpuwm.ingest import preprocess_backend
    from gpuwm.static import projection
    text, report, exp, namelist, controls = _source(tmp_path, monkeypatch)
    met = tmp_path/'met_em.d01.2026-05-17_18_00_00.nc'; met.write_bytes(b'input')
    run = SimpleNamespace(toml_text=text, experiment=exp, namelist_input=namelist,
                          paths={1: (met,)}, interval_seconds=3600., controls=controls,
                          coverage_seconds=exp.run_seconds,
                          substitution_report=report)
    monkeypatch.setattr(metem_forecast, 'resolve_metem_vertical', lambda run, text, **kw: (text, 'explicit', None))
    monkeypatch.setattr(metem_door, 'metgrid_memory_admission', lambda *a: {})
    monkeypatch.setattr(preprocess_backend, 'resolve_preprocess_backend',
                        lambda *a, **kw: SimpleNamespace(receipt=lambda: {'backend': 'test'}))
    class StopPreparation(Exception): pass
    monkeypatch.setattr(projection, 'grids_from_projection_config',
                        lambda *a: (_ for _ in ()).throw(StopPreparation()))
    original = tmp_path/'prepared'
    with pytest.raises(StopPreparation):
        metem_forecast.prepare_metem_run(run, original)
    # The stop is after the document phase. These placeholders declare
    # only its completion inventory; no cache reader is exercised here.
    (original/'metgrid-import.json').write_text('{}')
    for domain in exp.domains:
        (original/f'static-d{domain.grid_id:02d}.npz').write_bytes(b'static marker')
        cache = original/f'd{domain.grid_id:02d}'; cache.mkdir()
        (cache/'header.json').write_text('{}')
    before = _snapshot(original)
    duration = exp.run_seconds/2 if changed else None
    with pytest.raises(StopPreparation):
        metem_forecast.prepare_metem_run(run, original, run_seconds=duration)
    assert _snapshot(original) == before
    generations = list(tmp_path.glob('prepared-attempt-*/prepared/experiment.toml'))
    assert bool(generations) is changed
    if changed:
        assert tomllib.loads(generations[0].read_text())['experiment']['run_seconds'] == duration


@pytest.mark.parametrize('changed', [False, True])
def test_wrf_preparation_reuses_documents_without_rewriting_or_changes_generation(tmp_path, monkeypatch, changed):
    from gpuwm import wrfinput_forecast
    from gpuwm.ingest import wrfinput
    from gpuwm.core import landuse
    from gpuwm.static import projection
    text, report, exp, namelist, _ = _source(tmp_path, monkeypatch)
    boundary = tmp_path/'wrfbdy_d01'; boundary.write_bytes(b'boundary input')
    inputs = {}
    for domain in exp.domains:
        path = tmp_path/f'wrfinput_d{domain.grid_id:02d}'; path.write_bytes(b'state input')
        inputs[domain.grid_id] = path
    run = SimpleNamespace(toml_text=text, namelist_input=namelist,
        wrfbdy_path=boundary, wrfinput_paths=inputs, substitution_report=report,
        coverage=SimpleNamespace(coverage_seconds=exp.run_seconds,
            times=(exp.start_time,), end=exp.start_time+timedelta(seconds=exp.run_seconds),
            forcing_interval_seconds=3600.))
    attrs = dict(MMINLU='MODIFIED_IGBP_MODIS_NOAH', NUM_LAND_CAT=21, ISWATER=17,
                 ISLAKE=21, ISICE=15, ISURBAN=13, ISOILWATER=14, CEN_LAT=35., USE_THETA_M=0)
    raw = {name: np.zeros((2,2),np.float32) for name in
           ('LU_INDEX','ISLTYP','LANDMASK','SNOW','XICE','TSLB','SST','MAPFAC_M','MAPFAC_U','MAPFAC_V','F','E','HGT')}
    monkeypatch.setattr(wrfinput, 'read_wrfinput', lambda path, **kw: SimpleNamespace(
        path=path, global_attributes=attrs, raw=raw, surface_input_dispositions={},
        soil_unit_conversions={}))
    monkeypatch.setattr(wrfinput, 'read_wrfbdy', lambda *a, **kw: object())
    monkeypatch.setattr(landuse, 'initialize_landuse', lambda *a, **kw: object())
    monkeypatch.setattr(projection, 'grids_from_projection_config', lambda exp: [object() for _ in exp.domains])
    original = tmp_path/'prepared'
    first = wrfinput_forecast.prepare_wrf_run(run, original)
    before = _snapshot(original)
    result = wrfinput_forecast.prepare_wrf_run(run, original,
        run_seconds=exp.run_seconds/2 if changed else None)
    assert _snapshot(original) == before
    assert (result.prepared_root != first.prepared_root) is changed
    assert result.experiment_config == result.prepared_root/'experiment.toml'
    assert result.artifact_paths['preparation_receipt'] == result.prepared_root/'wrf-import.json'
