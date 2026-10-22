"""Fire acquisition must keep source pages and resolve experiment geometry."""
import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from gpuwm import fire_ignition


def test_acquisition_keeps_every_opaque_page_and_binds_uuid(tmp_path, monkeypatch):
    pages = [
        b'{"type":"FeatureCollection","features":[{"id":1}],"exceededTransferLimit":true}',
        b'{"type":"FeatureCollection","features":[{"id":2}],"exceededTransferLimit":false}',
    ]
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return pages[len(requests) - 1]

    def fetch(url, timeout):
        requests.append(url)
        return Response()

    monkeypatch.setattr(fire_ignition.urllib.request, "urlopen", fetch)
    receipt = fire_ignition.acquire("wfigs-perimeters", "00000000-0000-0000-0000-000000000001", tmp_path)
    raw = Path(receipt["path"]).read_bytes()
    assert all(page in raw for page in pages)
    assert receipt["sha256"] == hashlib.sha256(raw).hexdigest()
    queries = [parse_qs(urlsplit(url).query) for url in requests]
    assert [q["resultOffset"] for q in queries] == [["0"], ["1"]]
    assert all(q["where"] == ["poly_IRWINID = '{00000000-0000-0000-0000-000000000001}'"] for q in queries)


def test_acquisition_refuses_server_errors_and_invalid_ids(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        fire_ignition.acquire("wfigs-perimeters", "' OR 1=1", tmp_path)
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return b'{"error":{"code":499,"message":"Token Required"}}'
    monkeypatch.setattr(fire_ignition.urllib.request, "urlopen", lambda *a, **kw: Response())
    with pytest.raises(ValueError, match="Token Required"):
        fire_ignition.acquire("wfigs-incidents", "00000000-0000-0000-0000-000000000001", tmp_path)


def test_request_cannot_override_the_forecast_location(tmp_path, monkeypatch):
    from gpuwm.static import sfire
    monkeypatch.setattr(sfire, "resolve_experiment_request", lambda *a, **kw: dict(
        grid_spec={"ref_lat": 40}, sr_x=10, sr_y=10))
    path = tmp_path / "request.json"
    request = dict(schema="gpuwm-fire-ignition-v1", grid_spec={"ref_lat": 39},
                   output_directory="output", incident={"path": "incident.geojson"})
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError, match="conflicts with experiment geometry"):
        fire_ignition.request_from_file(path, experiment="case.toml")
    del request["grid_spec"]
    path.write_text(json.dumps(request))
    resolved = fire_ignition.request_from_file(path, experiment="case.toml")
    assert resolved["grid_spec"] == {"ref_lat": 40}
    assert resolved["incident"]["path"] == str(tmp_path / "incident.geojson")
    assert resolved["output_directory"] == str(tmp_path / "output")
