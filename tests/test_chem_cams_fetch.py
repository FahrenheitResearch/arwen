from dataclasses import replace
import shlex
import hashlib
import json
import threading

import pytest
from gpuwm import data_store_fetch as ds
from gpuwm import chem_table


@pytest.fixture
def row():
    return chem_table.catalog().sources['cams-global']


@pytest.fixture(autouse=True)
def private_home(tmp_path, monkeypatch):
    monkeypatch.setattr(ds.Path, 'home', classmethod(lambda cls: tmp_path))
    (tmp_path / '.adsapirc').write_text('url: https://example.invalid/api\nkey: test-private-key\n')


def plan(row, **kwargs):
    options = dict(area='30.1,-100.1,31.1,-99.1', start='2025-07-30T13:00Z',
                   end='2025-07-30T17:00Z', variables=['ozone'], now='2026-01-01T00:00Z')
    options.update(kwargs)
    return ds.resolve_acquisition(row, **options)


def test_determinism(row):
    a, b = plan(row), plan(row)
    assert a == b
    assert len(a.requests) == 2
    for r in a.requests:
        assert r.sha256 == hashlib.sha256(ds.canonical(r.request).encode()).hexdigest()
        assert r.request['leadtime_hour'] == ('0', '3', '6')
        assert r.cycle == '2025-07-30T12:00:00+00:00'
        assert r.request['area'] == (32.0, -101.2, 29.2, -98.0)
    assert a.requests[0].request['model_level'] == tuple(str(i) for i in range(1, 138))
    assert a.requests[1].request['variable'] == ('surface_pressure',)


def test_cycle_and_horizon(row):
    p = plan(row, now='2025-07-30T13:00Z')
    assert p.requests[0].cycle == '2025-07-30T00:00:00+00:00'
    assert p.requests[0].request['leadtime_hour'] == ('12', '15', '18')
    with pytest.raises(ds.DataStoreHorizonExceeded, match='lead 123 h exceeds horizon 120 h'):
        plan(row, end='2025-08-04T13:00Z')
    p = plan(row, start='2025-07-30T00:00Z', end='2025-07-30T00:00Z')
    assert p.requests[0].request['leadtime_hour'] == ('0',)


def test_shared_rows_merge(row):
    oxidants = chem_table.catalog().sources['cams-oxidants']
    assert row.acquisition == oxidants.acquisition
    p = plan([row, oxidants], variables={row.name: ['ozone'], oxidants.name: ['hydroxyl_radical']})
    assert len(p.requests) == 2
    assert p.requests[0].request['variable'] == ('hydroxyl_radical', 'ozone', 'specific_humidity')
    assert p.requests[0].variables_by_source[oxidants.name] == ('hydroxyl_radical', 'specific_humidity')
    assert p.requests[0].variables_by_source[row.name] == ('ozone', 'specific_humidity')
    assert p.requests[1].variables_by_source[oxidants.name] == ('surface_pressure',)
    assert p == plan([oxidants, row], variables={row.name: ['ozone'], oxidants.name: ['hydroxyl_radical']})
    with pytest.raises(TypeError):
        row.grid['spacing_deg'] = 1


def test_separate_api_and_incompatible_blocks(row):
    other = replace(row, name='another-source', acquisition=ds.freeze({
        **ds.thaw(row.acquisition), 'api_url': 'https://example.invalid/api'}))
    p = plan([row, other])
    assert len(p.requests) == 4
    other = replace(row, name='another-source', acquisition=ds.freeze({
        **ds.thaw(row.acquisition), 'latency_note': 'different block'}))
    with pytest.raises(ValueError, match='Cannot merge incompatible acquisition rows'):
        plan([row, other])


class Client:
    calls = []
    def __init__(self, **kwargs): pass
    def retrieve(self, dataset, request):
        self.calls.append(request)
        return self
    def download(self, path):
        ds.Path(path).write_bytes(b'opaque fake GRIB bytes')


def test_cache_and_corruption(row, tmp_path):
    Client.calls = []
    p = plan(row)
    root = tmp_path/'cache'
    a = ds.fetch(p, cache_root=root, client_factory=Client, progress=lambda x: None)
    assert a.reused == (False, False)
    assert len(Client.calls) == 2
    b = ds.fetch(p, cache_root=root, client_factory=lambda **kw: pytest.fail('cache submits'), progress=lambda x: None)
    assert b.reused == (True, True)
    a.files[0].write_bytes(b'corruption')
    c = ds.fetch(p, cache_root=root, client_factory=Client, progress=lambda x: None)
    assert c.reused == (False, True)
    assert len(Client.calls) == 3
    receipt = json.loads(a.receipts[0].read_text())
    assert receipt['sources'][0]['file_sha256'] == chem_table.catalog().file_hashes[row.source_file]


def test_concurrency(row, tmp_path):
    barrier = threading.Barrier(2)
    class Concurrent(Client):
        def retrieve(self, dataset, request):
            barrier.wait(timeout=5)
            return self
    ds.fetch(plan(row), cache_root=tmp_path/'cache', client_factory=Concurrent, progress=lambda x: None)


def test_antimeridian_and_full_band(row):
    p = plan(row, area=ds.Area(-2, 170, 2, -170))
    assert p.requests[0].request['area'] == (2.8, 169.2, -2.8, -169.2)
    p = plan(row, area=ds.Area(-90, -180, 90, 180))
    assert p.requests[0].request['area'] == (90, -180, -90, 180)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    def refused(*args, **kwargs):
        pytest.fail('Network is forbidden in data-store tests')
    monkeypatch.setattr(socket.socket, 'connect', refused)
    monkeypatch.setattr(socket, 'create_connection', refused)


def test_eastern_edge_180(row):
    p = plan(row, area=ds.Area(-2, 170, 2, 179.2))
    assert p.requests[0].request['area'] == (2.8, 169.2, -2.8, 180)


def test_cached_files_missing_and_exact_command(row, tmp_path, monkeypatch):
    p = plan(row, now='2025-07-30T13:00Z')
    monkeypatch.setattr(ds.cds_credentials, '_profile', lambda path: pytest.fail('credential read'))
    monkeypatch.setattr(ds, 'load_credentials', lambda: pytest.fail('credential declaration read'))
    root = tmp_path/'empty-cache'
    with pytest.raises(ds.DataStoreFilesMissing) as caught:
        ds.cached_files(p, cache_root=root)
    assert not root.exists()
    text = str(caught.value)
    for r in p.requests:
        assert r.group in text and r.sha256 in text
    command = text.split('Fetch with: ')[1]
    assert command.startswith('python -m gpuwm.data_store_fetch fetch --source cams-global --area ')
    # Reproduce the command's plan without submitting it.
    captured = []
    monkeypatch.setattr(ds, 'fetch', lambda actual, **kwargs: captured.append(actual) or ds.FetchResult((), (), ()))
    assert ds.main(shlex.split(command)[3:]) == 0
    assert captured == [p]


@pytest.mark.parametrize('mapping', [False, True])
def test_user_files_and_cache_only_reader(row, tmp_path, monkeypatch, mapping):
    p = plan(row)
    inputs = [tmp_path/f'input-{i}.grib2' for i in range(len(p.requests))]
    for i, path in enumerate(inputs):
        path.write_bytes(b'user supplied bytes' + bytes([i]))
    monkeypatch.setattr(ds.cds_credentials, '_profile', lambda path: pytest.fail('credential read'))
    monkeypatch.setattr(ds, 'load_credentials', lambda: pytest.fail('credential declaration read'))
    files = {r.sha256: path for r, path in zip(p.requests, inputs)} if mapping else inputs
    root = tmp_path/'cache'
    result = ds.fetch(p, cache_root=root, files=files,
        client_factory=lambda **kwargs: pytest.fail('user files submitted'), progress=lambda text: None)
    assert ds.cached_files(p, cache_root=root) == result.files
    for original, published in zip(inputs, result.files):
        assert original.read_bytes() == published.read_bytes()
    record = json.loads(result.receipts[0].read_text())
    assert record['sources'][0]['file_sha256'] == chem_table.catalog().file_hashes[row.source_file]
    result.files[0].write_bytes(b'corrupt')
    with pytest.raises(ds.DataStoreFilesMissing) as caught:
        ds.cached_files(p, cache_root=root)
    assert p.requests[0].sha256 in str(caught.value)
    assert p.requests[1].sha256 not in str(caught.value)


def test_cli_plan_cache_path(row, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(ds.cds_credentials, '_profile', lambda path: pytest.fail('credential read'))
    root = tmp_path/'custom-cache'
    assert ds.main(['plan', '--source', row.name, '--area', '30,-100,31,-99',
        '--start', '2025-07-30T12:00Z', '--end', '2025-07-30T15:00Z', '--cache-root', str(root)]) == 0
    output = json.loads(capsys.readouterr().out)
    for r in output['requests']:
        assert r['cache_path'] == str(root/(r['sha256']+'.grib2'))
    assert not root.exists()


@pytest.mark.parametrize('field,value', [
    ('request_sha256', 'wrong'), ('request', {}), ('dataset', 'wrong'),
    ('bytes', 0), ('file_sha256', 'wrong'),
])
def test_cache_receipt_integrity(row, tmp_path, monkeypatch, field, value):
    p = plan(row)
    root = tmp_path/'cache'
    result = ds.fetch(p, cache_root=root, client_factory=Client, progress=lambda text: None)
    receipt = json.loads(result.receipts[0].read_text())
    receipt[field] = value
    result.receipts[0].write_text(json.dumps(receipt))
    monkeypatch.setattr(ds.cds_credentials, '_profile', lambda path: pytest.fail('cache reader read credential'))
    with pytest.raises(ds.DataStoreFilesMissing) as caught:
        ds.cached_files(p, cache_root=root)
    assert p.requests[0].sha256 in str(caught.value)
    assert p.requests[1].sha256 not in str(caught.value)


def test_missing_command_negative_latitude(row, tmp_path, monkeypatch):
    p = plan(row, area=ds.Area(-2, 170, 2, 179.2))
    captured = []
    monkeypatch.setattr(ds, 'fetch', lambda actual, **kwargs: captured.append(actual) or ds.FetchResult((), (), ()))
    assert ds.main(shlex.split(p.fetch_command(cache_root=tmp_path/'cache'))[3:]) == 0
    assert captured == [p]


def test_the_fetch_and_initialization_read_one_cache_root(monkeypatch, tmp_path):
    """GPUWM_DATA_STORE_CACHE moved initialization's search and not the
    fetch's publication, so a run with it set refused the files its own
    fetch stage had just written."""
    from gpuwm import chem_source_init, data_store_fetch
    monkeypatch.setenv(data_store_fetch.CACHE_ENV, str(tmp_path / "cache"))
    assert chem_source_init.CACHE_ENV == data_store_fetch.CACHE_ENV
    assert data_store_fetch._cache_root(None) == tmp_path / "cache"
    assert chem_source_init.data_store_cache_root() == tmp_path / "cache"
    monkeypatch.delenv(data_store_fetch.CACHE_ENV)
    assert (data_store_fetch._cache_root(None)
            == chem_source_init.data_store_cache_root())
