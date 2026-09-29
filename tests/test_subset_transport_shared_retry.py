"""The two older subset transports retry on the tree's one shared discipline.

The breakage this guards: ``tools/download_gfs_native_subset._download``
asked a 404 five times, 5 to 20 s apart (50 s spent on an object the
host had said it does not have), and it and
``tools/download_hrrr_native_subset._download_range`` kept their own
wait ladders beside :func:`gpuwm.fetch_endpoints.retry_delay`, so the
same fault waited differently depending on which transport met it.

Only the network and the clock are replaced; the retry loops, the
shared classification and the attempt budget are the production ones.
"""

from __future__ import annotations

from urllib.error import HTTPError, URLError

import pytest

from gpuwm import fetch_endpoints
from tools import download_gfs_native_subset as gfs_transport
from tools import download_hrrr_native_subset as hrrr_transport


SHARED_WAITS = [2.0 ** attempt
                for attempt in range(1, fetch_endpoints.TRANSIENT_ATTEMPTS)]


def _network(transport, monkeypatch, answer):
    """Every request meets ``answer(url)``; returns the list of URLs asked, and the waits."""

    asked: list[str] = []
    naps: list[float] = []

    def urlopen(request, **_kwargs):
        asked.append(request.full_url)
        return answer(request.full_url)

    monkeypatch.setattr(transport, "paced_urlopen", urlopen)
    monkeypatch.setattr(transport.time, "sleep", naps.append)
    return asked, naps


def _missing(url):
    raise HTTPError(url, 404, "Not Found", {}, None)


def _reset(url):
    raise URLError(ConnectionResetError(104, "connection reset by peer"))


def test_gfs_subset_does_not_ask_a_missing_object_again(tmp_path, monkeypatch):
    asked, naps = _network(gfs_transport, monkeypatch, _missing)

    with pytest.raises(HTTPError) as error:
        gfs_transport._download("https://nomads.ncep.noaa.gov/cgi-bin/x",
                                tmp_path / "gfs.t00z.pgrb2.0p25.f003.subset.grib2")

    assert error.value.code == 404
    assert len(asked) == 1
    assert naps == []
    assert not list(tmp_path.iterdir())


def test_gfs_subset_waits_out_a_dropped_connection_on_the_shared_schedule(
        tmp_path, monkeypatch):
    asked, naps = _network(gfs_transport, monkeypatch, _reset)

    with pytest.raises(URLError):
        gfs_transport._download("https://nomads.ncep.noaa.gov/cgi-bin/x",
                                tmp_path / "gfs.t00z.pgrb2.0p25.f003.subset.grib2")

    assert len(asked) == fetch_endpoints.TRANSIENT_ATTEMPTS
    assert naps == SHARED_WAITS


def test_gfs_subset_never_asks_more_than_the_shared_budget(tmp_path, monkeypatch):
    asked, _naps = _network(gfs_transport, monkeypatch, _reset)

    with pytest.raises(URLError):
        gfs_transport._download("https://nomads.ncep.noaa.gov/cgi-bin/x",
                                tmp_path / "subset.grib2", retries=10)

    assert len(asked) == fetch_endpoints.TRANSIENT_ATTEMPTS


def test_gfs_subset_asks_an_incomplete_body_again(tmp_path, monkeypatch):
    class _Truncated:
        def __init__(self):
            self._left = [b"GRIB" + b"\x00" * 2048]

        def read(self, *_args):
            return self._left.pop() if self._left else b""

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    asked, naps = _network(gfs_transport, monkeypatch, lambda url: _Truncated())

    with pytest.raises(RuntimeError, match="not a complete GRIB2 stream"):
        gfs_transport._download("https://nomads.ncep.noaa.gov/cgi-bin/x",
                                tmp_path / "subset.grib2")

    assert len(asked) == fetch_endpoints.TRANSIENT_ATTEMPTS
    assert naps == SHARED_WAITS


def _range(tmp_path):
    return (hrrr_transport.ByteRange(0, 9, "1:0", "1:0"), tmp_path / "range.part")


def test_hrrr_range_does_not_ask_a_missing_object_again(tmp_path, monkeypatch):
    asked, naps = _network(hrrr_transport, monkeypatch, _missing)
    byte_range, path = _range(tmp_path)

    with pytest.raises(HTTPError) as error:
        hrrr_transport._download_range(
            "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/x", byte_range, path, 5)

    assert error.value.code == 404
    assert len(asked) == 1
    assert naps == []


def test_hrrr_range_waits_out_a_dropped_connection_on_the_shared_schedule(
        tmp_path, monkeypatch):
    asked, naps = _network(hrrr_transport, monkeypatch, _reset)
    byte_range, path = _range(tmp_path)

    with pytest.raises(URLError):
        hrrr_transport._download_range(
            "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/x", byte_range, path, 10)

    # --retries 10 is held to the shared budget.
    assert len(asked) == fetch_endpoints.TRANSIENT_ATTEMPTS
    assert naps == SHARED_WAITS


def test_hrrr_range_asks_an_answer_that_did_not_verify_again(tmp_path, monkeypatch):
    class _WholeObject:
        status = 200
        headers: dict = {}

        def read(self, *_args):
            return b""

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    asked, naps = _network(hrrr_transport, monkeypatch, lambda url: _WholeObject())
    byte_range, path = _range(tmp_path)

    with pytest.raises(ValueError, match="range request returned HTTP 200"):
        hrrr_transport._download_range(
            "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/x", byte_range, path, 5)

    assert len(asked) == fetch_endpoints.TRANSIENT_ATTEMPTS
    assert naps == SHARED_WAITS
    assert not path.exists()


def test_the_hrrr_range_tool_refuses_retries_past_the_shared_budget(tmp_path):
    """--retries 10 was accepted and would now be held to five: refused by name instead."""

    with pytest.raises(ValueError, match=f"retries must be 1..{fetch_endpoints.TRANSIENT_ATTEMPTS}"):
        hrrr_transport.main([
            "--cycle", "2026-07-18_00:00:00",
            "--forecast-hours", "0,1",
            "--output-root", str(tmp_path / "download"),
            "--retries", str(fetch_endpoints.TRANSIENT_ATTEMPTS + 1),
        ])
    assert not (tmp_path / "download").exists()
