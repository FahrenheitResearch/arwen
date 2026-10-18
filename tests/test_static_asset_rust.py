"""Exercise the real Rust bridge on a local ranged origin, including pin refusal."""
import http.server
import io
import json
import os
from pathlib import Path
import struct
import tarfile
import threading

import pytest

from gpuwm.geog_assets import _rw_asset, download_archive, extract_datasets, GeogFetchError
from gpuwm.table_assets import fetch_asset_from_url, TableAssetError
from gpuwm.core.thompson_contract import TableAsset

SHA_ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


@pytest.fixture
def origin():
    objects, requests = {}, []
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_HEAD(self):
            payload = objects.get(self.path)
            if payload is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
        def do_GET(self):
            payload = objects.get(self.path)
            if payload is None:
                self.send_error(404)
                return
            span = self.headers.get("Range")
            requests.append((self.path, span))
            if span:
                start, end = span[6:].split("-")
                first = int(start)
                last = min(int(end) if end else len(payload) - 1, len(payload) - 1)
                if first >= len(payload):
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{len(payload)}")
                    self.end_headers()
                    return
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {first}-{last}/{len(payload)}")
                payload = payload[first:last + 1]
            else:
                self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        def log_message(self, *args):
            pass
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", objects, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_ranges_pin_and_atomic_publication(tmp_path, origin, monkeypatch):
    base, objects, requests = origin
    payload = bytes(range(256)) * (140000)
    objects["/static"] = payload
    seed = tmp_path / "pin-source"
    seed.write_bytes(payload)
    pin = _rw_asset("hash", input=seed)["sha256"]
    dest = tmp_path / "source.nc"
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("GPUWM_TIME_LEDGER", str(ledger))
    record = download_archive(base + "/static", dest, expected_bytes=len(payload),
                              expected_sha256=pin,
                              progress=lambda *args: None, label="pinned-source")
    assert dest.read_bytes() == payload
    assert record["bytes"] == len(payload)
    assert sum(bool(span) for _, span in requests) >= 3
    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [(e["stage"], e["phase"]) for e in events] == [
        ("prep.geog.fetch", "begin"), ("prep.geog.fetch", "end")]
    assert {k: v for k, v in events[-1]["attrs"].items() if k != "cpu_s"} == {
        "bytes": len(payload), "datasets": ["pinned-source"]}
    assert not list(tmp_path.glob("*.part-*"))


def test_named_corrupt_mirror_refused_without_final(tmp_path, origin):
    base, objects, _ = origin
    objects["/coeff.dat"] = b"wrong coefficients"
    asset = TableAsset("coeff.dat", len(objects["/coeff.dat"]), "0" * 64)
    with pytest.raises(TableAssetError, match="coeff.dat.*pinned object refused"):
        fetch_asset_from_url(tmp_path, asset, base + "/coeff.dat")
    assert not (tmp_path / asset.filename).exists()
    assert not list(tmp_path.glob(".*fetch-partial*"))


def test_same_size_local_corruption_refused(tmp_path):
    path = tmp_path / "source.nc"
    path.write_bytes(b"abc")
    with pytest.raises(GeogFetchError, match="source.nc.*pin refused"):
        _rw_asset("hash", input=path, bytes=3, sha256="0" * 64)
    assert path.read_bytes() == b"abc"


def test_production_static_fetch_cannot_bypass_pin(tmp_path):
    with pytest.raises(GeogFetchError, match="unpinned geography.*prepared fields"):
        download_archive("https://example.invalid/data", tmp_path / "archive", expected_bytes=3,
                         expected_sha256="0" * 64, strict_size=False,
                         label="selected-version", progress=lambda *args: None)


def test_mirror_with_appended_bytes_refused(tmp_path, origin):
    base, objects, _ = origin
    objects["/appended"] = b"abcBAD"
    with pytest.raises(GeogFetchError, match="pin refused"):
        download_archive(base + "/appended", tmp_path / "static.nc", expected_bytes=3,
                         expected_sha256=SHA_ABC,
                         label="appended-source", progress=lambda *args: None)
    assert not (tmp_path / "static.nc").exists()


def _zstd_stream(payload: bytes, *, chunk: int = 4096) -> bytes:
    """``pzstd``'s layout, written without ``pzstd``: per chunk, a skippable
    frame carrying the next frame's size, then a zstd frame of one raw block
    (RFC 8878 sections 3.1.1 and 3.1.2: single segment, 4-byte content size,
    no checksum). No host zstd compressor exists on every platform this
    runs on, and the codec under test is the product's own, in rw_fetch."""
    out = bytearray()
    for start in range(0, len(payload), chunk):
        piece = payload[start:start + chunk]
        frame = (struct.pack("<IBI", 0xFD2FB528, 0xA0, len(piece))
                 + struct.pack("<I", len(piece) << 3 | 1)[:3] + piece)
        out += struct.pack("<III", 0x184D2A50, 4, len(frame)) + frame
    return bytes(out)


def test_native_zstd_unpack_and_selection(tmp_path):
    from test_fetch_geog import _INDEX_TEXT
    raw = tmp_path / "data.tar"
    with tarfile.open(raw, "w") as archive:
        for name in ("soiltemp_1deg", "other"):
            for filename, payload in (("index", _INDEX_TEXT.encode()), ("00001-00004.00001-00004", b"\x01\x00" * 16)):
                item = tarfile.TarInfo(f"WPS_GEOG/{name}/{filename}")
                item.size = len(payload)
                archive.addfile(item, io.BytesIO(payload))
    compressed = tmp_path / "data.tar.zst"
    compressed.write_bytes(_zstd_stream(raw.read_bytes()))
    root = tmp_path / "geog"
    root.mkdir()
    counts = extract_datasets(compressed, root, ("soiltemp_1deg",), progress=lambda *args: None, label="selected")
    assert counts["soiltemp_1deg"]["files"] == 2
    assert (root / "soiltemp_1deg/index").read_bytes() == _INDEX_TEXT.encode()
    assert not (root / "other").exists()
    audit = tmp_path / "audit"
    _rw_asset("unpack-audit", input=compressed, output=audit, datasets="soiltemp_1deg")
    rows = [json.loads(line) for line in (audit / "unpack-audit.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:
        expected = root / row["path"]
        assert row["bytes"] == expected.stat().st_size
        assert row["sha256"] == _rw_asset("hash", input=expected)["sha256"]
        assert not (audit / row["path"]).exists()


@pytest.mark.parametrize("mode", ["w:bz2", "w:gz"])
def test_native_unpack_decodes_the_shipped_codecs_without_host_programs(tmp_path, monkeypatch, mode):
    # Every pinned geography archive is .tar.bz2 and the mandatory bundle is
    # .tar.gz. rw_fetch spawned the host bzip2 and gzip for them, which a
    # stock Windows PATH does not carry: fetch-geog refused every archive
    # there. With PATH emptied, only an in-process decoder can pass.
    from gpuwm.rustwx_fetch import FETCH_ENV, find_fetch_bin
    from test_fetch_geog import _build_archive, _INDEX_TEXT
    binary = find_fetch_bin()
    if binary is not None:
        monkeypatch.setenv(FETCH_ENV, str(Path(binary).resolve()))
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
    archive = tmp_path / ("data.tar." + mode[2:])
    archive.write_bytes(_build_archive(("soiltemp_1deg", "other"), mode=mode))
    root = tmp_path / "geog"
    root.mkdir()
    counts = extract_datasets(archive, root, ("soiltemp_1deg",), progress=lambda *args: None, label="codec")
    assert counts["soiltemp_1deg"]["files"] == 2
    assert (root / "soiltemp_1deg/index").read_bytes() == _INDEX_TEXT.encode()
    assert not (root / "other").exists()


@pytest.mark.parametrize("kind,name", [(b"0", "../outside"), (b"2", "soiltemp_1deg/link")])
def test_native_unpack_refuses_unsafe_members(tmp_path, kind, name):
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as tar:
        item = tarfile.TarInfo(name)
        item.type = kind
        item.linkname = "../../outside" if kind == b"2" else ""
        tar.addfile(item)
    root = tmp_path / "geog"
    root.mkdir()
    with pytest.raises(GeogFetchError, match="unsafe tar member|non-regular tar member"):
        extract_datasets(archive, root, ("soiltemp_1deg",), progress=lambda *args: None, label="unsafe")
    assert not list(root.iterdir())


def test_geography_scope_uses_selected_reader(tmp_path, monkeypatch):
    # Pure selector test complements the real CLI selection on sealed input configs.
    from gpuwm.geog_assets import datasets_required_by, parse_datasets
    assert "soilgrids" not in parse_datasets("wrf")
    assert "soilgrids" in datasets_required_by("mesh")


def test_table_rows_keep_raw_pins():
    from gpuwm.table_assets import table_source_rows
    from gpuwm.ingest.wif_dataset import WIF_DATASET_ASSET
    from gpuwm.core.thompson_contract import CLASSIC_TABLE_ASSETS, FORK_TABLE_ASSETS
    from gpuwm.core.thompson_aerosol_contract import AEROSOL_TABLE_ASSETS
    for asset in (*CLASSIC_TABLE_ASSETS, *FORK_TABLE_ASSETS, *AEROSOL_TABLE_ASSETS, WIF_DATASET_ASSET):
        rows = table_source_rows(asset)
        assert rows, asset.filename
        if rows:
            assert len(rows) >= 2
            assert all(row["version"] for row in rows)
            assert all(row["bytes"] == asset.bytes and row["sha256"] == asset.sha256 for row in rows)
            # The Hugging Face mirror is declared for every table, last: its
            # objects are not published yet and answer 404 until they are.
            assert "huggingface.co/datasets/deepguess/wps-geog-arwen/" in rows[-1]["url"]
            assert not any("huggingface.co" in row["url"] for row in rows[:-1])


def test_selected_tree_uses_wrf_inventory_without_mesh_soil():
    from gpuwm.geog_assets import geography_for_run, datasets_required_by
    plan = geography_for_run(Path(__file__).parent / "fixtures/static_asset_configs/selected-tree.toml")
    assert tuple(plan["datasets"]) == datasets_required_by("wrf")
    assert plan["static_sources"] == []
    assert plan["archive_bytes"] == 1294439865


def test_selected_source_uses_its_identical_legend_metadata():
    from gpuwm.geog_assets import geography_for_run
    plan = geography_for_run(Path(__file__).parent / "fixtures/static_asset_configs/selected-source.toml")
    assert plan["datasets"] == []
    assert plan["static_sources"] == ["hrrr-conus-v4"]
    assert plan["archive_bytes"] == 0
    assert plan["static_bytes"] == 1112724420


def test_complete_source_legend_matches_original_index_without_tiles(tmp_path):
    from dataclasses import replace
    from gpuwm.static.build import GeogSelection
    from gpuwm.static.external_source import parse_static_source
    from gpuwm.io.wrfout import _DEFAULT_LANDUSE_ATTRS
    selection = replace(GeogSelection.fallback(tmp_path),
                        static_source=parse_static_source({"source": "hrrr-conus-v4"}, source="fixture"))
    assert selection.landuse_global_attrs() == {
        key: value for key, value in _DEFAULT_LANDUSE_ATTRS.items() if key != "ISOILWATER"}
    assert list(tmp_path.iterdir()) == []


def test_configured_geography_gate_keeps_missing_required_index_refusal(tmp_path):
    from gpuwm.go_cli import geography_refusal
    fixtures = Path(__file__).parent / "fixtures/static_asset_configs"
    assert geography_refusal(tmp_path, config=fixtures / "selected-source.toml") is None
    refusal = geography_refusal(tmp_path, config=fixtures / "selected-tree.toml")
    assert "preparation cannot build" in refusal
    # One spelling of the index on every platform, inside the tree named once.
    assert "topo_gmted2010_30s/index" in refusal
    assert f"in {tmp_path}: " in refusal


def test_local_file_url_keeps_pin_and_atomic_contract(tmp_path):
    source = tmp_path / "source with spaces"
    source.write_bytes(b"local static coefficients")
    asset = TableAsset("coeff.dat", source.stat().st_size, _rw_asset("hash", input=source)["sha256"])
    root = tmp_path / "staged"
    root.mkdir()
    final = fetch_asset_from_url(root, asset, source.as_uri())
    assert final.read_bytes() == source.read_bytes()


def test_parallel_geography_publishes_on_lock_owner(tmp_path, origin, monkeypatch):
    from test_fetch_geog import _build_archive, _pin
    from gpuwm import geog_assets
    base, objects, _ = origin
    pins = []
    for name in ("first_ds", "second_ds"):
        payload = _build_archive((name,))
        pin = _pin(name, payload)
        pins.append(pin)
        objects["/" + pin.filename] = payload
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES", tuple(pins))
    monkeypatch.setenv(geog_assets.GEOG_URL_BASE_ENV, base)
    assert geog_assets.fetch_geog(root=tmp_path, datasets=("first_ds", "second_ds"),
                                 source="hf", progress=lambda *args: None) == 2
    manifest = json.loads((tmp_path / geog_assets.GEOG_FETCH_MANIFEST_NAME).read_text())
    assert set(manifest["archives"]) == {pin.filename for pin in pins}
    assert all(row["pinned"] for row in manifest["archives"].values())


def test_sibling_success_keeps_receipt_after_pin_failure(tmp_path, origin, monkeypatch):
    from test_fetch_geog import _build_archive, _pin
    from gpuwm import geog_assets
    from dataclasses import replace
    base, objects, _ = origin
    pins = []
    for name in ("good_ds", "bad_ds"):
        payload = _build_archive((name,))
        pin = _pin(name, payload)
        pins.append(replace(pin, archive_sha256="0" * 64) if name == "bad_ds" else pin)
        objects["/" + pin.filename] = payload
    monkeypatch.setattr(geog_assets, "GEOG_ARCHIVES", tuple(pins))
    monkeypatch.setenv(geog_assets.GEOG_URL_BASE_ENV, base)
    with pytest.raises(GeogFetchError, match="bad_ds.*pinned object refused"):
        geog_assets.fetch_geog(root=tmp_path, datasets=("good_ds", "bad_ds"),
                              source="hf", progress=lambda *args: None)
    manifest = json.loads((tmp_path / geog_assets.GEOG_FETCH_MANIFEST_NAME).read_text())
    assert set(manifest["archives"]) == {pins[0].filename}
    assert (tmp_path / "good_ds/index").is_file()
    assert not (tmp_path / "bad_ds").exists()


def test_ledger_failure_does_not_break_pin_install(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUWM_TIME_LEDGER", str(tmp_path))
    source = tmp_path / "source"
    source.write_bytes(b"abc")
    asset = TableAsset("coeff.dat", 3, SHA_ABC)
    assert fetch_asset_from_url(tmp_path / "root", asset, source.as_uri()).read_bytes() == b"abc"


def test_packaged_source_row_uses_native_pin(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from gpuwm import table_assets
    package = tmp_path / "package data"
    package.mkdir()
    (package / "coeff.dat").write_bytes(b"abc")
    asset = TableAsset("coeff.dat", 3, SHA_ABC)
    table = tmp_path / "sources.toml"
    table.write_text('schema = "gpuwm-table-sources-v1"\n[[source]]\n'
                     'filename = "coeff.dat"\nversion = "pinned-fixture"\nbytes = 3\n'
                     f'sha256 = "{asset.sha256}"\nurl = "package://fixture_data/coeff.dat"\n')
    monkeypatch.setattr(table_assets, "TABLE_SOURCE_PATH", table)
    monkeypatch.setitem(sys.modules, "fixture_data", SimpleNamespace(data_root=lambda: package))
    assert table_assets.fetch_asset_from_sources(tmp_path / "staged", asset).read_bytes() == b"abc"
    (package / "coeff.dat").write_bytes(b"bad")
    with pytest.raises(TableAssetError, match="pinned object refused"):
        table_assets.fetch_asset_from_sources(tmp_path / "other", asset)


def test_corrupt_mirror_of_a_real_shipped_table_refuses_by_its_name(tmp_path, origin):
    from gpuwm.core.thompson_contract import CLASSIC_TABLE_ASSETS
    from gpuwm.physics_compat import packaged_thompson_table_root
    asset = next(row for row in CLASSIC_TABLE_ASSETS if row.filename == "thompson_aux_tables.dat")
    original = Path(packaged_thompson_table_root()) / asset.filename
    if not original.is_file():
        pytest.skip("this installation lacks the pinned data companion fixture")
    _rw_asset("hash", input=original, bytes=asset.bytes, sha256=asset.sha256)
    damaged = bytearray(original.read_bytes())
    damaged[len(damaged) // 2] ^= 1
    base, objects, _ = origin
    objects["/" + asset.filename] = bytes(damaged)
    target = tmp_path / "tables"
    with pytest.raises(TableAssetError, match=asset.filename + ".*pinned object refused"):
        fetch_asset_from_url(target, asset, base + "/" + asset.filename)
    assert not (target / asset.filename).exists()
