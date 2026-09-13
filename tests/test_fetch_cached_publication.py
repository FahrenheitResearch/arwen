"""Complete bound local inventories survive remote retention expiry."""
import argparse
from datetime import datetime
import json
import os
from urllib.error import HTTPError

import pytest

from gpuwm import fetch, fetch_routes, fetch_endpoints
from test_fetch_routes import _fake_downloader


def _cached(tmp_path, *, source='rap', member=None):
    plan = fetch_routes.resolve_request(source, cycle=datetime(2026, 9, 9, 12), hours=1,
                                        member=member, out=tmp_path)
    fetch_routes.run_plan(plan, out=tmp_path, downloader=_fake_downloader([]),
                          probe=lambda url: False, progress=lambda message: None)
    fetch_routes.write_handoff(plan, tmp_path)
    return plan


def _args(out, *, member=None):
    parser = argparse.ArgumentParser()
    fetch.register_cli(parser.add_subparsers())
    args = ['fetch', '--source', 'rap', '--cycle', '2026-09-09T12', '--hours', '1', '--out', str(out)]
    if member:
        args.extend(['--member', member])
    return parser.parse_args(args)


def test_complete_bound_cache_never_probes_expired_remote_objects(tmp_path, monkeypatch):
    plan = _cached(tmp_path)
    before = {(tmp_path / obj.relpath): (tmp_path / obj.relpath).read_bytes() for obj in plan.objects}
    def forbidden(*args, **kwargs):
        pytest.fail('complete local inventory must be verified without asking the provider')
    monkeypatch.setattr(fetch, 'require_published_cycle', forbidden)
    monkeypatch.setattr(fetch_endpoints, 'object_available', forbidden)
    monkeypatch.setattr(fetch_routes, '_download_object', forbidden)
    assert fetch.fetch_main(_args(tmp_path)) == 0
    assert all(path.read_bytes() == body for path, body in before.items())
    receipt = json.loads((tmp_path / fetch_routes.MANIFEST_NAME).read_text())
    assert all(row['reused'] for row in receipt['files'])


@pytest.mark.parametrize('mutation', ['missing', 'same-size-mtime', 'wrong-cycle', 'wrong-member'])
def test_partial_changed_or_unbound_cache_cannot_become_a_complete_handoff(tmp_path, monkeypatch, mutation):
    plan = _cached(tmp_path)
    first = tmp_path / plan.objects[0].relpath
    if mutation == 'missing':
        first.unlink()
    elif mutation == 'same-size-mtime':
        status = first.stat()
        value = bytearray(first.read_bytes()); value[5] ^= 1
        first.write_bytes(value)
        os.utime(first, ns=(status.st_atime_ns, status.st_mtime_ns))
    else:
        for name in (fetch_routes.MANIFEST_NAME, fetch_routes._RECOVERY_REQUEST_NAME):
            path = tmp_path / name
            document = json.loads(path.read_text())
            document['request']['cycle' if mutation == 'wrong-cycle' else 'member'] = 'different'
            path.write_text(json.dumps(document))
    def missing(url, *args, **kwargs):
        raise HTTPError(url, 404, 'retained object no longer served', {}, None)
    monkeypatch.setattr(fetch_routes, '_download_object', missing)
    monkeypatch.setattr(fetch_endpoints, 'object_available', lambda url: False)
    monkeypatch.setattr(fetch, 'require_published_cycle', lambda *args, **kwargs: None)
    before = (tmp_path / fetch_routes.MANIFEST_NAME).read_bytes()
    with pytest.raises((ValueError, HTTPError)):
        fetch.fetch_main(_args(tmp_path))
    assert (tmp_path / fetch_routes.MANIFEST_NAME).read_bytes() == before
