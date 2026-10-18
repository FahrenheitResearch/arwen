"""Partial bridge fetch proofs against a local HTTP range server only."""
import hashlib
import http.server
import io
from pathlib import Path
import threading
import zipfile

import pytest

from gpuwm import bridge_assets as assets
from gpuwm.bridge_partial import fetch_partial


@pytest.fixture
def bundle_server(monkeypatch):
    payloads = {"rw_fetch": b"gpuwm-rw-fetch-v1\0" + b"fetch" * 300,
                "rw_wrfbatch": b"renderer" * 300,
                "rw_compare": b"launcher" * 40}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)
        # An omitted incompressible member proves the transfer stays partial.
        import random
        archive.writestr("assets/unused", random.Random(42).randbytes(300000))
    archive_bytes = buffer.getvalue()
    requests = []
    mode = {"ignore": False, "corrupt": False, "size": False}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            range_header = self.headers.get("Range")
            requests.append(range_header)
            if mode["ignore"]:
                self.send_response(200)
                self.end_headers()
                return
            start, end = map(int, range_header.removeprefix("bytes=").split("-"))
            chunk = archive_bytes[start:end + 1]
            if mode["corrupt"] and start == 0:
                chunk = b"X" + chunk[1:]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(archive_bytes) + int(mode['size'])}")
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    bundle = assets.BundlePin("linux-x86_64", "fixture.zip", len(archive_bytes),
                              hashlib.sha256(archive_bytes).hexdigest(), tuple(
        assets.BinaryPin(name, name, len(payload), hashlib.sha256(payload).hexdigest())
        for name, payload in payloads.items()))
    pins = assets.BridgePins("v2.8.9", {bundle.platform: bundle})
    monkeypatch.setattr(assets, "bundle_url", lambda *_: f"http://127.0.0.1:{server.server_port}/fixture.zip")
    # Transport tests use fixture contracts, not fake release binaries.
    monkeypatch.setattr(assets, "verify_contract_marker", lambda *_: None)
    yield pins, bundle, payloads, requests, mode
    server.shutdown()
    server.server_close()
    thread.join()


def test_selected_member_and_idempotence(bundle_server, tmp_path):
    pins, bundle, payloads, requests, _ = bundle_server
    messages = []
    assert fetch_partial(pins, bundle, tmp_path, ["rw_fetch"], progress=messages.append) == [tmp_path / "rw_fetch"]
    assert (tmp_path / "rw_fetch").read_bytes() == payloads["rw_fetch"]
    assert not (tmp_path / "rw_wrfbatch").exists()
    before = len(requests)
    assert fetch_partial(pins, bundle, tmp_path, ["rw_fetch"], progress=messages.append) == []
    assert len(requests) == before
    transferred = int(messages[-1].split("fetched ")[1].split()[0])
    assert transferred < bundle.bytes // 10


def test_launcher_also_fetches_renderer(bundle_server, tmp_path):
    pins, bundle, _, _, _ = bundle_server
    assert {path.name for path in fetch_partial(pins, bundle, tmp_path, ["rw_compare"])} == {"rw_compare", "rw_wrfbatch"}


def test_cli_partial_selection_uses_local_server(bundle_server, tmp_path, monkeypatch):
    import argparse
    pins, _, payloads, requests, _ = bundle_server
    monkeypatch.setattr(assets, "load_pins", lambda: pins)
    monkeypatch.setattr(assets, "host_platform", lambda: "linux-x86_64")
    parser = argparse.ArgumentParser()
    assets.register_cli(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["fetch-bridges", "--partial", "--artifact", "rw_fetch", "--dest", str(tmp_path)])
    assert assets.fetch_bridges_main(args) == 0
    assert (tmp_path / "rw_fetch").read_bytes() == payloads["rw_fetch"]
    assert not (tmp_path / "rw_wrfbatch").exists()
    assert requests and all(request.startswith("bytes=") for request in requests)


@pytest.mark.parametrize("mode", ["ignore", "size", "corrupt"])
def test_bad_response_preserves_existing_binary(bundle_server, tmp_path, mode):
    pins, bundle, _, _, flags = bundle_server
    old = tmp_path / "rw_fetch"
    old.write_bytes(b"previous")
    flags[mode] = True
    with pytest.raises(assets.BridgeAssetError):
        fetch_partial(pins, bundle, tmp_path, ["rw_fetch"])
    assert old.read_bytes() == b"previous"


def test_wrong_pin_and_unknown_selection_refuse(bundle_server, tmp_path):
    from dataclasses import replace
    pins, bundle, _, requests, _ = bundle_server
    with pytest.raises(assets.BridgeAssetError, match="unpinned"):
        fetch_partial(pins, bundle, tmp_path, ["absent"])
    assert not requests
    bad = replace(bundle, binaries=(replace(bundle.binaries[0], sha256="0" * 64),))
    with pytest.raises(assets.BridgeAssetError):
        fetch_partial(pins, bad, tmp_path, ["rw_fetch"])
    assert not (tmp_path / "rw_fetch").exists()


# Named breakage (2.9.1 verify of the on-demand observation doors): with the
# bundle host unreachable or answering 404, the refusal read "partial fetch
# failed: File is not a zip file", because zipfile turns the reader's OSError
# into BadZipFile; an operator then looks for a damaged bundle instead of a
# network or mirror problem.
from urllib.error import URLError as _URLError  # noqa: E402


class _NetPin:
    def __init__(self, artifact, filename):
        self.artifact, self.filename, self.bytes = artifact, filename, 10


class _NetBundle:
    def __init__(self):
        self.binaries = [_NetPin("rw_ndbc", "rw_ndbc")]
        self.bytes = 1000


def test_an_unreachable_bundle_is_named_not_called_a_bad_zip(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "matches_pin", lambda path, pin: False)
    monkeypatch.setattr(assets, "bundle_url", lambda pins, bundle: "https://example.invalid/b.zip")

    def refuse(request):
        raise _URLError("connection refused")

    with pytest.raises(assets.BridgeAssetError) as info:
        fetch_partial(object(), _NetBundle(), Path(tmp_path), ["rw_ndbc"],
                      progress=lambda *_: None, urlopen_fn=refuse)
    text = str(info.value)
    assert "could not be read" in text and "connection refused" in text
    assert "not a zip file" not in text
