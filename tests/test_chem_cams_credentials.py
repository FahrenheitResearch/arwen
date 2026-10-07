import json
import pytest
from gpuwm import data_store_fetch as ds
from gpuwm import chem_table
from gpuwm.source_credentials import credential_present


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(ds.Path, 'home', classmethod(lambda cls: tmp_path))
    return tmp_path


def plan():
    return ds.resolve_acquisition(chem_table.catalog().sources['cams-global'], area='30,-100,31,-99',
        start='2025-07-30T12:00Z', end='2025-07-30T15:00Z', variables=['ozone'], now='2026-01-01')


def test_presence_and_no_plan_read(home, monkeypatch):
    c = ds.load_credentials()['copernicus-ads']
    assert credential_present(c) is False
    (home/'.adsapirc').write_text('not a usable credential')
    monkeypatch.setattr(ds.cds_credentials, '_profile', lambda p: pytest.fail('read at plan time'))
    assert credential_present(c) is True
    assert len(plan().requests) == 2


def test_missing(home):
    with pytest.raises(ds.DataStoreCredentialMissing) as caught:
        ds.fetch(plan(), cache_root=home/'cache')
    assert str(caught.value) == ds.credential_facts(ds.load_credentials()['copernicus-ads'])['absent_message'] + ' ' + ds.load_credentials()['copernicus-ads'].remedy


POLICIES = "user didn't accept all required site policies. Missing policies are: Data protection and privacy statement (rev. 1) - https://example.invalid/privacy, Terms of use of the Copernicus Atmosphere Data Store (rev. 1) - https://example.invalid/terms"


@pytest.mark.parametrize('status,message,kind,fragment', [
    (403, POLICIES, ds.DataStorePolicyNotAccepted, 'log in to https://ads.atmosphere.copernicus.eu with the same account and accept them; nothing else changes'),
    (403, "Missing policies are: Licence to use Copernicus Products - https://example.invalid/licence", ds.DataStoreLicenceNotAccepted, 'open https://ads.atmosphere.copernicus.eu/datasets/cams-global-atmospheric-composition-forecasts'),
    (401, 'authentication failed', ds.DataStoreCredentialRejected, 'is present but refused (HTTP 401)'),
    (500, 'service failure', ValueError, 'service failure'),
])
@pytest.mark.parametrize('stage', ['constructor', 'retrieve', 'download'])
def test_refusals_mask_everywhere(home, status, message, kind, fragment, stage, capsys):
    key = 'very-private-short-secret'
    (home/'.adsapirc').write_text(f'url: https://example.invalid/api\nkey: {key}\n')
    class Error(Exception):
        response = type('Response', (), {'status_code': status})()
    def fail():
        raise Error(f'HTTP {status}: {message} {key}')
    class Client:
        def __init__(self, **kwargs):
            assert kwargs['key'] == key
            for name in ('info_callback', 'error_callback', 'warning_callback', 'debug_callback'):
                kwargs[name](key)
            if stage == 'constructor': fail()
        def retrieve(self, *args):
            if stage == 'retrieve': fail()
            return self
        def download(self, path): fail()
    with pytest.raises(kind) as caught:
        ds.fetch(plan(), cache_root=home/'cache', client_factory=Client)
    assert fragment in str(caught.value)
    if kind is ds.DataStorePolicyNotAccepted:
        assert POLICIES.split('Missing policies are: ')[1] in str(caught.value)
    assert key not in str(caught.value)
    assert key not in capsys.readouterr().out
    assert not list((home/'cache').glob('*.json'))


def test_success_receipt_has_no_key(home):
    key = 'very-private-short-secret'
    (home/'.adsapirc').write_text(f'key: {key}\n')
    class Client:
        def __init__(self, **kw): pass
        def retrieve(self, *args): return self
        def download(self, path): ds.Path(path).write_bytes(b'bytes')
    logs = []
    result = ds.fetch(plan(), cache_root=home/'cache', client_factory=Client, progress=logs.append)
    assert key not in ''.join(p.read_text() for p in result.receipts) + ''.join(logs)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    def refused(*args, **kwargs):
        pytest.fail('Network is forbidden in data-store tests')
    monkeypatch.setattr(socket.socket, 'connect', refused)
    monkeypatch.setattr(socket, 'create_connection', refused)


def test_fetch_loads_public_declarations_once(home, monkeypatch):
    p = plan()
    (home/'.adsapirc').write_text('key: synthetic-key\n')
    reader = ds.load_credentials
    calls = []
    def load():
        calls.append(True)
        return reader()
    monkeypatch.setattr(ds, 'load_credentials', load)
    class Client:
        def __init__(self, **kwargs): pass
        def retrieve(self, *args): return self
        def download(self, path): ds.Path(path).write_bytes(b'bytes')
    ds.fetch(p, cache_root=home/'cache', client_factory=Client, progress=lambda text: None)
    assert len(calls) == 1
