"""The generic URL-list pool driver: instrument validated both ways.

Five instruments in one night once gave confident wrong answers, so the
measurement driver is held to known answers in both directions before
it measures anything: a good set passes with an honest receipt, a bad
digest refuses by name, and a flaky server is retried exactly once with
the server's own Retry-After honored.  No network anywhere.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

from tools import measure_fetch_concurrency as driver


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _opener_for(payloads: dict[str, bytes]):
    def opener(request, timeout=None):
        url = request.full_url
        name = url.rsplit("/", 1)[-1]
        return _Response(payloads[name])
    return opener


PAYLOADS = {f"object-{i:03d}.grib2.bz2": f"payload {i}".encode() * 100
            for i in range(8)}


def _url_list(tmp_path: Path, *, with_expectations: bool) -> Path:
    lines = []
    for name, payload in PAYLOADS.items():
        url = f"https://example.invalid/cycle/00/{name}"
        if with_expectations:
            digest = hashlib.sha256(payload).hexdigest()
            lines.append(f"{url}\t{digest}\t{len(payload)}")
        else:
            lines.append(url)
    lines.insert(0, "# comment line")
    listing = tmp_path / "urls.txt"
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return listing


def test_a_known_good_set_passes_and_receipts_honestly(tmp_path):
    receipt = driver.run(
        _url_list(tmp_path, with_expectations=True), tmp_path / "out",
        workers=4, label="unit", opener=_opener_for(PAYLOADS))
    assert receipt["schema"] == driver.RECEIPT_SCHEMA
    assert receipt["concurrency"]["files"] == len(PAYLOADS)
    assert receipt["concurrency"]["workers_requested"] == 4
    assert len(receipt["files"]) == len(PAYLOADS)
    for entry in receipt["files"]:
        landed = tmp_path / "out" / entry["name"]
        assert landed.read_bytes() == PAYLOADS[entry["name"]]
        assert entry["sha256"] == hashlib.sha256(
            PAYLOADS[entry["name"]]).hexdigest()
        assert entry["attempts"] == 1


def test_a_wrong_digest_refuses_by_name_and_promotes_nothing(tmp_path):
    listing = _url_list(tmp_path, with_expectations=True)
    tampered = dict(PAYLOADS)
    victim = "object-004.grib2.bz2"
    tampered[victim] = b"not the recorded bytes" * 50
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match=victim):
        driver.run(listing, out, workers=4,
                   opener=_opener_for(tampered))
    assert not (out / victim).exists()
    assert not list(out.glob("*.part"))


def test_a_flaky_server_is_retried_once_honoring_retry_after(tmp_path,
                                                             monkeypatch):
    from urllib.error import HTTPError

    naps: list[float] = []
    calls: dict[str, int] = {}

    def opener(request, timeout=None):
        name = request.full_url.rsplit("/", 1)[-1]
        calls[name] = calls.get(name, 0) + 1
        if name == "object-002.grib2.bz2" and calls[name] == 1:
            raise HTTPError(request.full_url, 503, "busy",
                            {"Retry-After": "9"}, None)
        return _Response(PAYLOADS[name])

    spec = {"url": "https://example.invalid/c/object-002.grib2.bz2",
            "name": "object-002.grib2.bz2", "sha256": None, "bytes": None}
    entry = driver.download_one(spec, tmp_path, opener=opener,
                                sleeper=naps.append)
    assert entry["attempts"] == 2
    assert naps and max(naps) >= 9.0
    assert (tmp_path / spec["name"]).read_bytes() == PAYLOADS[spec["name"]]


def test_the_list_grammar_refuses_junk(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("ftp://example.invalid/x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a fetchable URL"):
        driver.parse_url_list(bad.read_text(encoding="utf-8"))
    dupes = "https://a.invalid/x/name.grib2\nhttps://b.invalid/y/name.grib2"
    with pytest.raises(ValueError, match="duplicate object name"):
        driver.parse_url_list(dupes)
    with pytest.raises(ValueError, match="no objects"):
        driver.parse_url_list("# nothing\n")
    with pytest.raises(ValueError, match="sha256"):
        driver.parse_url_list("https://a.invalid/x.grib2\tnot-a-digest\n")


def test_an_empty_payload_refuses(tmp_path):
    spec = {"url": "https://example.invalid/c/empty.grib2",
            "name": "empty.grib2", "sha256": None, "bytes": None}
    with pytest.raises(RuntimeError, match="empty.grib2"):
        driver.download_one(
            spec, tmp_path,
            opener=lambda request, timeout=None: _Response(b""),
            sleeper=lambda seconds: None)
