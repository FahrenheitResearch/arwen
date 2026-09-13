"""Bind received recovery payloads and preserve the existing healthy identity."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.filesystem_paths import canonical_path, io_path
from gpuwm.wrfinput_forecast import WrfInitialization


def _native_payload(monkeypatch, value, calls=None):
    from gpuwm import netcdf_bridge

    payload = np.full((1, 4, 2, 2), value, dtype='<f8').tobytes()
    monkeypatch.setattr(netcdf_bridge, 'resolve_netcdf_bin', lambda: Path('native-reader'))

    def run(arguments, *, what):
        if calls is not None:
            calls.append(arguments)
        out = Path(arguments[-1])
        out.mkdir()
        (out / 'SMOIS.f64').write_bytes(payload)
        (out / 'metadata.json').write_text(json.dumps({
            'schema': 'gpuwm-wrf-soil-recovery-v1',
            'receipt': {'operation': 'same-authorized-source-recovery'},
            'variables': [{'name': 'SMOIS', 'filename': 'SMOIS.f64',
                           'shape': [1, 4, 2, 2], 'dtype': '<f8', 'units': 'm3 m-3',
                           'dimensions': ['Time', 'soil_layers_stag', 'south_north', 'west_east']}],
        }), encoding='utf-8')

    monkeypatch.setattr(netcdf_bridge, '_run', run)
    return payload


def _domain_identity(receipt):
    initialization = WrfInitialization.__new__(WrfInitialization)
    bundle = SimpleNamespace(authority_sha256={'wrfinput': 'a' * 64},
                             restored=SimpleNamespace(soil_recovery=receipt))
    return initialization.domain_content_sha256(bundle)


def test_changed_native_payload_cannot_keep_the_same_restart_domain_identity(monkeypatch):
    from gpuwm import netcdf_bridge

    results = []
    for value in (.2, .3):
        payload = _native_payload(monkeypatch, value)
        values, receipt = netcdf_bridge.recover_wrf_soil('wrfinput', 'met_em', {})
        np.testing.assert_array_equal(values['SMOIS'], np.full((4, 2, 2), value))
        field = receipt['recovered_fields']['SMOIS']
        assert field == {'sha256': hashlib.sha256(payload).hexdigest(),
                         'shape': [1, 4, 2, 2], 'dtype': '<f8', 'units': 'm3 m-3'}
        results.append(receipt)
    # Original WRF, authority and operation are deliberately identical;
    # only received output payload bytes differ.
    assert results[0]['operation'] == results[1]['operation']
    assert _domain_identity(results[0]) != _domain_identity(results[1])
    assert _domain_identity(results[0]) == _domain_identity(dict(results[0]))


def test_healthy_input_retains_its_existing_domain_identity():
    assert _domain_identity({}) == 'a' * 64


@pytest.mark.skipif(os.name != 'nt', reason='actual Windows extended path spelling')
def test_deep_source_discovery_and_native_bridge_preserve_path_identity(tmp_path, monkeypatch):
    from gpuwm.ingest.wrf_soil_recovery import recover_supplied_soil

    deep = tmp_path
    while len(str(deep)) < 310:
        deep /= 'original-soil-inputs-with-long-path'
    io_path(deep).mkdir(parents=True)
    path = deep / 'wrfinput_d01'
    source = deep / 'met_em.d01.2026-09-12_00_00_00.nc'
    table = deep / 'Vtable'
    io_path(path).write_bytes(b'original WRF input')
    io_path(source).write_bytes(b'original metgrid source')
    io_path(table).write_text(
        '85 | 111 | 1 | | SOILT001 | K | Soil temperature |\n'
        '85 | 111 | 2 | | SOILT002 | K | Soil temperature |\n'
        '86 | 112 | 0 | 1 | SOILM001 | kg m-2 | Soil water |\n'
        '86 | 112 | 1 | 3 | SOILM002 | kg m-2 | Soil water |\n', encoding='utf-8')
    originals = {item.name: io_path(item).read_bytes() for item in (path, source, table)}
    calls = []
    _native_payload(monkeypatch, .25, calls)
    fields = {'SMOIS': np.full((4, 2, 2), 22.), 'LANDMASK': np.ones((2, 2))}
    values, receipt = recover_supplied_soil(path, fields, {'START_DATE': '2026-09-12_00:00:00'})
    assert calls[0][2] == str(io_path(path))
    assert calls[0][3] == str(io_path(source))
    assert receipt['input_files']['met_em']['path'] == str(canonical_path(source))
    assert not receipt['input_files']['met_em']['path'].startswith('\\\\?\\')
    np.testing.assert_array_equal(values['SMOIS'], .25)
    assert {item.name: io_path(item).read_bytes() for item in (path, source, table)} == originals
