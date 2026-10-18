"""The open national radar archives, fetched by table (gpuwm.obs.radar_archives).

Offline: every network is a fake opener serving the bytes a real archive
serves, so each wrapping (one URL per time, a member of a day zip read by
byte range, an S3 listing) is exercised without the network. One live test
(GPUWM_NETWORK_TESTS=1) reads a real BoM volume out of NCI's day zip.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import urllib.error
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gpuwm.obs import radar_archives as ra


class _Response(io.BytesIO):
    def __init__(self, data: bytes, status: int = 200, headers: dict | None = None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}


class _Server:
    """A fake ``urlopen``: URL -> bytes, honouring ``Range``, 404 otherwise."""

    def __init__(self, files: dict[str, bytes]):
        self.files = files
        self.requests: list[tuple[str, str | None]] = []

    def __call__(self, request, timeout=None):
        url = request.full_url
        span = request.get_header("Range")
        self.requests.append((url, span))
        if url not in self.files:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        data = self.files[url]
        if span:
            start, end = (int(x) for x in re.match(r"bytes=(\d+)-(\d+)", span).groups())
            end = min(end, len(data) - 1)
            return _Response(data[start:end + 1], 206,
                             {"Content-Range": f"bytes {start}-{end}/{len(data)}"})
        return _Response(data)


def test_the_shipped_table_is_well_formed():
    table = ra.load_table()
    assert {"jma-nict", "bom-nci", "ideam-s3", "kma-apihub"} <= set(table)
    for archive in table.values():
        entry = archive.entry
        assert entry["container"] in {"file", "zip-member", "s3-listing"}
        templates = entry.get("urls", []) + [entry.get("prefix", "")]
        for template in templates:
            leftover = re.sub(r"\{t:[^}]*\}|\{site\}|\{auth\}", "", template)
            assert "{" not in leftover, template
        assert entry["licence"] and entry["site_ids"]


def test_a_jma_time_fetches_both_tars_of_its_ten_minute_slot(tmp_path):
    base = "https://pawr.nict.go.jp/jmadata/JMA-PolarCoordsRadar/2019/10/12/"
    server = _Server({
        base + "Z__C_RJTD_20191012090000_RDR_JMAGPV_N5_grib2.tar": b"n5",
        base + "Z__C_RJTD_20191012090000_RDR_JMAGPV_N6_grib2.tar": b"n6",
    })
    fetched = ra.fetch("jma-nict", site="AKIT",
                       valid_time=datetime(2019, 10, 12, 9, 3), out_dir=tmp_path,
                       opener=server)
    assert fetched.volume_time == datetime(2019, 10, 12, 9, 0, tzinfo=timezone.utc)
    assert [p.read_bytes() for p in fetched.files] == [b"n5", b"n6"]


def test_an_archive_gap_is_reported_as_one(tmp_path):
    with pytest.raises(ra.RadarArchiveError, match="404.*gaps"):
        ra.fetch("jma-nict", site="AKIT", valid_time=datetime(2024, 8, 29),
                 out_dir=tmp_path, opener=_Server({}))


def test_a_keyed_archive_names_the_variable_and_how_to_get_the_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GPUWM_KMA_AUTHKEY", raising=False)
    with pytest.raises(ra.RadarArchiveError, match="GPUWM_KMA_AUTHKEY.*apihub.kma.go.kr"):
        ra.fetch("kma-apihub", site="KWK", valid_time=datetime(2025, 7, 1),
                 out_dir=tmp_path, opener=_Server({}))


def test_a_key_is_sent_to_its_archive_and_never_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUWM_KMA_AUTHKEY", "SECRET123")
    url = ("https://apihub.kma.go.kr/api/typ01/url/rdr_file_down.php?rdr=UF&stn=KWK"
           "&qcd=0&tm=202507010000&authKey=SECRET123")
    fetched = ra.fetch("kma-apihub", site="KWK", valid_time=datetime(2025, 7, 1),
                       out_dir=tmp_path, opener=_Server({url: b"UF.."}))
    assert fetched.files[0].name == "KMA_KWK_202507010000.uf"
    assert "SECRET123" not in repr(fetched.to_payload())


def test_one_volume_is_read_out_of_a_day_zip_by_range(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for stamp in ("115500", "120000", "120500"):
            archive.writestr(f"2_20261013_{stamp}.pvol.h5", stamp.encode() * 4096)
    data = buffer.getvalue()
    url = "https://thredds.nci.org.au/thredds/fileServer/rq0/2/2026/vol/2_20261013.pvol.zip"
    server = _Server({url: data})
    fetched = ra.fetch("bom-nci", site="2", valid_time=datetime(2026, 10, 13, 12, 2),
                       out_dir=tmp_path, opener=server)
    assert [p.name for p in fetched.files] == ["2_20261013_120000.pvol.h5"]
    assert fetched.files[0].read_bytes() == b"120000" * 4096
    assert all(span is not None for _, span in server.requests), "never the whole zip"


def test_an_s3_listing_takes_every_task_of_the_nearest_time_and_the_gzipped_copy(tmp_path):
    bucket = "https://s3-radaresideam.s3.amazonaws.com/"
    keys = ["l2_data/2025/06/01/Tablazo/TAB250601000003.RAWYSAT",
            "l2_data/2025/06/01/Tablazo/TAB250601000003.RAWYSBR",
            "l2_data/2025/06/01/Tablazo/TAB250601000029.RAWYSAW"]
    listing = "".join(f"<Contents><Key>{k}</Key></Contents>" for k in keys)
    server = _Server({
        bucket + "?list-type=2&prefix=l2_data%2F2025%2F06%2F01%2FTablazo%2F":
            f"<ListBucketResult>{listing}</ListBucketResult>".encode(),
        **{bucket + k: k.encode() for k in keys},
    })
    fetched = ra.fetch("ideam-s3", site="Tablazo",
                       valid_time=datetime(2025, 6, 1, 0, 0, 10), out_dir=tmp_path,
                       opener=server)
    assert sorted(p.name for p in fetched.files) == [
        "TAB250601000003.RAWYSAT", "TAB250601000003.RAWYSBR"]

    bogota = ["l2_data/2025/06/01/Bogota/1399BOG-20250601-000042-PPIVol-7550.nc",
              "l2_data/2025/06/01/Bogota/1399BOG-20250601-000042-PPIVol-7550.nc.gz"]
    listing = "".join(f"<Contents><Key>{k}</Key></Contents>" for k in bogota)
    server = _Server({
        bucket + "?list-type=2&prefix=l2_data%2F2025%2F06%2F01%2FBogota%2F":
            f"<ListBucketResult>{listing}</ListBucketResult>".encode(),
        **{bucket + k: b"x" for k in bogota},
    })
    fetched = ra.fetch("ideam-s3", site="Bogota", valid_time=datetime(2025, 6, 1, 0, 1),
                       out_dir=tmp_path / "b", opener=server)
    assert [p.name for p in fetched.files] == ["1399BOG-20250601-000042-PPIVol-7550.nc.gz"]


@pytest.mark.skipif(os.environ.get("GPUWM_NETWORK_TESTS") != "1",
                    reason="live archive read; set GPUWM_NETWORK_TESTS=1")
def test_a_real_bom_volume_comes_out_of_the_nci_day_zip(tmp_path):
    fetched = ra.fetch("bom-nci", site="2", valid_time=datetime(2026, 10, 13, 12, 0),
                       out_dir=tmp_path)
    data = Path(fetched.files[0]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == (
        "991a0d70a83b0f1a8180396316286b020070e63c8d13fc5c8b5dcdc66d96551c")
    # A ~1 GB day zip; one volume costs a few MB.
    assert fetched.bytes_fetched < 10_000_000
