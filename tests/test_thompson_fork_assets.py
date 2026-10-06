"""Fork asset acquisition: complete-source pins, atomic writes and real HTTP."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.server
import json
from pathlib import Path
import threading

import pytest

from gpuwm.core.thompson_contract import TableAsset
from gpuwm import table_assets
from gpuwm import thompson_fork_assets as fork


@pytest.fixture
def transfers(tmp_path, monkeypatch):
    # Small transport fixtures exercise acquisition, not scientific coefficients.
    payloads = {f"part-{i}.dat": (f"pinned transfer {i}\n" * 16).encode() for i in range(4)}
    assets = tuple(TableAsset(name, len(data), hashlib.sha256(data).hexdigest())
                   for name, data in payloads.items())
    source = tmp_path / "source"
    source.mkdir()
    for name, data in payloads.items():
        (source / name).write_bytes(data)
    monkeypatch.setattr(fork, "FORK_TABLE_ASSETS", assets)
    monkeypatch.setattr(fork, "_packaged_source", lambda: None)
    monkeypatch.delenv(fork.FORK_TABLE_SOURCE_ROOT_ENV, raising=False)
    monkeypatch.delenv(fork.FORK_TABLE_ASSET_URL_BASE_ENV, raising=False)
    def unavailable(*args):
        raise FileNotFoundError("test prerequisite is unavailable")
    monkeypatch.setattr(fork, "_build_source", unavailable)
    fork._VERIFIED_ROOTS.clear()
    return source, tmp_path / "cache", payloads


def test_complete_local_source_is_pinned_then_reused_without_copy(transfers, monkeypatch):
    source, root, payloads = transfers
    monkeypatch.setenv(fork.FORK_TABLE_SOURCE_ROOT_ENV, str(source))
    assert fork.ensure_thompson_fork_tables(root) == root
    receipt = json.loads((root / "fork-table-acquisition.json").read_text())
    assert receipt["table_set"] == fork.FORK_TABLE_SET_ID
    assert receipt["acquired_from"] == str(source)
    before = {path.name: (path.stat().st_mtime_ns, path.read_bytes()) for path in root.iterdir()}
    monkeypatch.setattr(fork, "fetch_asset_from_dir", lambda *a: pytest.fail("valid cache recopied"))
    assert fork.ensure_thompson_fork_tables(root) == root
    assert before == {path.name: (path.stat().st_mtime_ns, path.read_bytes()) for path in root.iterdir()}
    assert {name: (root / name).read_bytes() for name in payloads} == payloads


@pytest.mark.parametrize("change", ["missing", "size", "hash"])
def test_bad_source_cannot_publish_any_file(transfers, change):
    source, root, payloads = transfers
    path = source / next(iter(payloads))
    if change == "missing":
        path.unlink()
    elif change == "size":
        path.write_bytes(b"short")
    else:
        path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises(table_assets.TableAssetError, match="not the canonical complete set"):
        fork.ensure_thompson_fork_tables(root, source_dir=source)
    assert not list(root.glob("*.dat"))


def test_wrong_cache_is_preserved_and_not_replaced(transfers):
    source, root, payloads = transfers
    fork.ensure_thompson_fork_tables(root, source_dir=source)
    bad = root / next(iter(payloads))
    bad.write_bytes(b"x" * bad.stat().st_size)
    with pytest.raises(table_assets.TableAssetError, match="without overwrite"):
        fork.ensure_thompson_fork_tables(root, source_dir=source)
    assert bad.read_bytes() == b"x" * bad.stat().st_size
    assert not list(root.glob(".*fetch-partial*"))


class _Server:
    def __init__(self, payloads, *, short_first=False):
        self.counts = {}
        self.short_first = short_first
        parent = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                name = self.path.rsplit("/", 1)[-1]
                parent.counts[name] = parent.counts.get(name, 0) + 1
                payload = payloads[name]
                if parent.short_first and parent.counts[name] == 1:
                    payload = payload[:5]
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.url = "http://127.0.0.1:" + str(self.httpd.server_address[1])

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join()


def test_concurrent_cache_misses_fetch_each_verified_file_once(transfers, monkeypatch):
    source, root, payloads = transfers
    server = _Server(payloads)
    monkeypatch.setenv(fork.FORK_TABLE_ASSET_URL_BASE_ENV, server.url)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: fork.ensure_thompson_fork_tables(root), range(2)))
        assert results == [root, root]
        assert server.counts == {name: 1 for name in payloads}
        assert {name: (root / name).read_bytes() for name in payloads} == payloads
        assert not list(root.glob(".*fetch-partial*"))
    finally:
        server.close()


def test_truncated_download_cannot_land_and_fresh_attempt_recovers(transfers, monkeypatch):
    source, root, payloads = transfers
    server = _Server(payloads, short_first=True)
    monkeypatch.setenv(fork.FORK_TABLE_ASSET_URL_BASE_ENV, server.url)
    try:
        with pytest.raises(table_assets.TableAssetError, match="staged 5 bytes"):
            fork.ensure_thompson_fork_tables(root)
        assert not list(root.glob("*.dat"))
        assert not list(root.glob(".*fetch-partial*"))
        server.short_first = False
        assert fork.ensure_thompson_fork_tables(root) == root
        assert {name: (root / name).read_bytes() for name in payloads} == payloads
    finally:
        server.close()


def test_a_named_bad_source_does_not_fall_through_to_a_mirror(transfers, monkeypatch):
    source, root, payloads = transfers
    monkeypatch.setenv(fork.FORK_TABLE_SOURCE_ROOT_ENV, str(source / "wrong"))
    monkeypatch.setenv(fork.FORK_TABLE_ASSET_URL_BASE_ENV, "http://127.0.0.1:1")
    monkeypatch.setattr(fork, "fetch_asset_from_url", lambda *a: pytest.fail("named source ignored"))
    with pytest.raises(table_assets.TableAssetError, match="not the canonical complete set"):
        fork.ensure_thompson_fork_tables(root)


def test_unavailable_canonical_source_names_the_real_offline_command(transfers):
    _source, root, _payloads = transfers
    with pytest.raises(FileNotFoundError, match="--thompson-fork-only --from DIR"):
        fork.ensure_thompson_fork_tables(root)
    assert not list(root.glob("*.dat"))
    assert not list(root.glob(".fork-build-*"))


def test_fork_cli_only_uses_the_same_transaction_and_leaves_classic_alone(transfers, monkeypatch):
    source, root, payloads = transfers
    monkeypatch.setattr(table_assets, "stage_classic_tables", lambda *a: pytest.fail("classic touched"))
    args = argparse.Namespace(thompson_fork=True, thompson_fork_only=True,
                              from_dir=str(source), thompson_fork_root=str(root))
    assert table_assets.fetch_tables_main(args) == 0
    assert {name: (root / name).read_bytes() for name in payloads} == payloads


def test_runtime_root_acquires_the_selected_fork_set(transfers, monkeypatch):
    source, root, payloads = transfers
    monkeypatch.setenv("GPUWM_THOMPSON_FORK_TABLE_ROOT", str(root))
    monkeypatch.setenv(fork.FORK_TABLE_SOURCE_ROOT_ENV, str(source))
    from gpuwm.core.microphysics_aerosol import _wrf39_table_root
    assert _wrf39_table_root() == str(root)
    assert {name: (root / name).read_bytes() for name in payloads} == payloads


def test_fresh_cache_invokes_the_default_source_builder(transfers, monkeypatch):
    source, root, payloads = transfers
    calls = []
    def build(work, log):
        calls.append(work)
        work.mkdir(parents=True)
        (work / "source-build-receipt.json").write_text('{"test": "small transport fixture"}')
        return source
    monkeypatch.setattr(fork, "_build_source", build)
    assert fork.ensure_thompson_fork_tables(root) == root
    assert len(calls) == 1
    assert json.loads((root / "fork-table-acquisition.json").read_text())["acquired_from"] == "pinned-public-source-build"
    assert {name: (root / name).read_bytes() for name in payloads} == payloads


def test_standalone_staging_ships_the_exact_small_generation_harness(tmp_path, monkeypatch):
    import tomllib
    from tools import build_rw_wps_release as release
    from gpuwm.thompson_fork_build import HARNESS_PINS
    # A lane can validate its uncommitted additions before the coordinator
    # commits. This only removes the Git tracking precondition, not the copy
    # or staged-import checks used by the actual standalone builder.
    monkeypatch.setattr(release, "_require_tracked", lambda path: None)
    stage = tmp_path / "standalone"
    report = release._stage_rw_wps_python_project(stage)
    package_data = tomllib.loads((stage / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"]["gpuwm"]
    assert "data/thompson/fork-build/*.F90" in package_data
    for name, digest in HARNESS_PINS.items():
        relative = "gpuwm/data/thompson/fork-build/" + name
        assert relative in report["files"]
        assert hashlib.sha256((stage / relative).read_bytes()).hexdigest() == digest
    assert not any(name.endswith(("freezeH2O.dat", "qr_acr_qg.dat", "qr_acr_qs.dat")) for name in report["files"])
