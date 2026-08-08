"""Analysis-hour-first fetch ordering, pinned as a contract.

Every prepared-route fetch already downloads the analysis hour's files
before any boundary hour's, because each ladder builder returns an
ascending ``range`` and each loop walks it in order.  That was
incidental -- a property of three ``range()`` calls that nothing
asserted -- and it is the property time to first plot depends on: the
analysis frame is what the first plot is OF, and any future work that
starts preparation before the whole window has landed can only do so if
the window lands in this order.

So it is pinned here, at both ends.  The ladder builders must put the
forecast-start lead first (``f000`` by default, ``f{K}`` under
``--forecast-start-hour K``), and each fetch loop must consume the
ladder in the order it was handed -- for the byte-range HRRR transport,
for the NOMADS subset transport, and for the ``rw_fetch`` full-file
backbone.

What this does NOT claim is a speedup.  Reordering buys nothing that is
not already there; see ``docs/run-plan.md`` for why preparation still
waits for the whole window, and what would have to change for it not
to.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from gpuwm import fetch, rustwx_fetch
from tools import download_gfs_native_subset as gfs_transport
from tools import download_hrrr_native_subset as hrrr_transport


def _grib2_stream(messages: int) -> bytes:
    """A stream of minimal, envelope-valid GRIB2 messages."""

    one = (b"GRIB" + b"\x00\x00" + b"\x00" + b"\x02"
           + (20).to_bytes(8, "big") + b"7777")
    return one * messages


def _fake_hrrr_product(request, *, workers, retries, expected_count=-1):
    """The real record cardinalities, so the completeness bar passes."""

    request.destination.write_bytes(_grib2_stream(
        hrrr_transport.SOIL_RECORD_COUNT if request.kind == "soil"
        else hrrr_transport.ATMOSPHERE_RECORD_COUNT))
    request.index_path.write_text("1:0:fixture\n", encoding="ascii")
    return {"kind": request.kind}


# ---------------------------------------------------------------------------
# The ladders themselves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ladder,kwargs,expected", [
    (fetch.gfs_forecast_hours, {"hours": 6, "cadence": 3}, (0, 3, 6)),
    (fetch.gfs_forecast_hours, {"hours": 6, "cadence": 3, "start": 12},
     (12, 15, 18)),
    (fetch.gfs_forecast_hours, {"hours": 3, "cadence": 1}, (0, 1, 2, 3)),
    (fetch.gdas_forecast_hours, {"hours": 6, "cadence": 3}, (0, 3, 6)),
    (fetch.gdas_forecast_hours, {"hours": 0, "cadence": 3}, (0,)),
])
def test_every_ladder_leads_with_the_hour_the_forecast_starts_at(
        ladder, kwargs, expected):
    """The initial condition's own lead is element zero, always."""

    hours = ladder(**kwargs)

    assert hours == expected
    assert hours == tuple(sorted(hours))
    assert hours[0] == kwargs.get("start", 0)


def test_the_hrrr_ladder_leads_with_the_forecast_start_lead_too():
    cycle = datetime(2026, 7, 28, 5)

    assert fetch.hrrr_forecast_hours(2, cycle) == (0, 1, 2)
    assert fetch.hrrr_forecast_hours(2, cycle, start=3) == (3, 4, 5)


# ---------------------------------------------------------------------------
# HRRR: the analysis hour's BOTH files land before any boundary file
# ---------------------------------------------------------------------------


def _hrrr_order(tmp_path, monkeypatch, *, hours):
    seen: list[str] = []

    def product(request, *, workers, retries, expected_count=-1):
        seen.append(request.destination.name)
        return _fake_hrrr_product(request, workers=workers, retries=retries,
                                  expected_count=expected_count)

    monkeypatch.setattr(hrrr_transport, "_download_product", product)
    fetch.fetch_hrrr(
        cycle=datetime(2026, 7, 28, 5), hours=hours, area=None,
        out=tmp_path / "hrrr", progress=lambda line: None)
    return seen


def test_the_hrrr_analysis_hour_lands_whole_before_any_boundary_hour(
        tmp_path, monkeypatch):
    """Both f00 products, then f01's, then f02's.  Never interleaved.

    Preparation reads soil from the analysis hour and atmosphere from
    every hour, so "the analysis has landed" means BOTH f00 files, not
    just the first one downloaded.
    """

    seen = _hrrr_order(tmp_path, monkeypatch, hours=(0, 1, 2))

    assert seen == [
        "hrrr.t05z.wrfnatf00.grib2", "hrrr.t05z.soilf00.grib2",
        "hrrr.t05z.wrfnatf01.grib2", "hrrr.t05z.soilf01.grib2",
        "hrrr.t05z.wrfnatf02.grib2", "hrrr.t05z.soilf02.grib2",
    ]
    # Stated as the property rather than as the literal list, so the
    # contract survives a change to how many products an hour has.
    leads = [int(name[-8:-6]) for name in seen]
    assert leads == sorted(leads)
    assert leads[:2] == [0, 0]


def test_a_lead_started_window_leads_with_that_lead_not_with_f00(
        tmp_path, monkeypatch):
    """``--forecast-start-hour 3`` makes f03 the analysis hour."""

    seen = _hrrr_order(tmp_path, monkeypatch, hours=(3, 4))

    assert seen[:2] == ["hrrr.t05z.wrfnatf03.grib2",
                        "hrrr.t05z.soilf03.grib2"]
    assert not any("f00" in name for name in seen)


def test_the_hrrr_receipt_claims_the_analysis_hour_first(tmp_path,
                                                         monkeypatch):
    """The per-hour checkpoint publishes the prefix, analysis included.

    A consumer waiting for "the analysis is on disk and verified" reads
    ``forecast_hours`` and gets it after the first hour, not after the
    last -- which is the seam any later overlap work would attach to.
    """

    published: list[list[int]] = []
    original = fetch.write_fetch_manifest

    def record(out, payload):
        published.append(list(payload["forecast_hours"]))
        return original(out, payload)

    monkeypatch.setattr(hrrr_transport, "_download_product",
                        _fake_hrrr_product)
    monkeypatch.setattr(fetch, "write_fetch_manifest", record)
    fetch.fetch_hrrr(
        cycle=datetime(2026, 7, 28, 5), hours=(0, 1, 2), area=None,
        out=tmp_path / "hrrr", progress=lambda line: None)

    assert published[0] == [0]
    assert published == [[0], [0, 1], [0, 1, 2], [0, 1, 2]]


# ---------------------------------------------------------------------------
# Order independence: the digest-bound manifest does not move
# ---------------------------------------------------------------------------


def test_the_bound_manifest_is_byte_identical_whatever_order_hours_land(
        tmp_path, monkeypatch):
    """``SHA256SUMS`` is the artifact preparation binds, and it is sorted.

    The one property any reordering work must not break: the digest the
    preparation stage is handed (``--source-manifest-sha256`` over
    ``SHA256SUMS``) is a function of the FILES, not of the sequence they
    arrived in.  Proven by fetching the same window forwards and
    backwards and comparing the bytes -- not by reading the ``sorted``
    call that makes it true.
    """

    monkeypatch.setattr(hrrr_transport, "_download_product",
                        _fake_hrrr_product)
    cycle = datetime(2026, 7, 28, 5)
    forwards = tmp_path / "forwards"
    backwards = tmp_path / "backwards"
    fetch.fetch_hrrr(cycle=cycle, hours=(0, 1, 2), area=None, out=forwards,
                     progress=lambda line: None)
    fetch.fetch_hrrr(cycle=cycle, hours=(2, 1, 0), area=None, out=backwards,
                     progress=lambda line: None)

    assert ((forwards / "SHA256SUMS").read_bytes()
            == (backwards / "SHA256SUMS").read_bytes())

    # And so are the payloads themselves, file for file.
    names = sorted(p.name for p in forwards.glob("*.grib2"))
    assert names == sorted(p.name for p in backwards.glob("*.grib2"))
    for name in names:
        assert ((forwards / name).read_bytes()
                == (backwards / name).read_bytes()), name


def test_the_bound_manifest_claims_only_hours_that_completed(
        tmp_path, monkeypatch):
    """Order independence must not become order blindness.

    ``SHA256SUMS`` is sorted by name, so it cannot say WHICH hours are
    done -- that is ``forecast_hours``' job, and a half-fetched hour
    must appear in neither.
    """

    monkeypatch.setattr(hrrr_transport, "_download_product",
                        _fake_hrrr_product)
    out = tmp_path / "hrrr"
    fetch.fetch_hrrr(cycle=datetime(2026, 7, 28, 5), hours=(0,), area=None,
                     out=out, progress=lambda line: None)

    lines = (out / "SHA256SUMS").read_text().splitlines()
    assert sorted(line.split("  ")[1] for line in lines) == [
        "hrrr.t05z.soilf00.grib2", "hrrr.t05z.wrfnatf00.grib2"]


# ---------------------------------------------------------------------------
# GFS: both transports walk the ladder in order
# ---------------------------------------------------------------------------


def test_the_nomads_subset_transport_walks_the_ladder_in_order(
        tmp_path, monkeypatch):
    requested: list[str] = []

    def download(url, destination, **kwargs):
        requested.append(url)
        destination.write_bytes(
            _grib2_stream(fetch.GFS_SUBSET_RECORD_COUNT))

    monkeypatch.setattr(gfs_transport, "_download", download)
    fetch.fetch_gfs(
        cycle=datetime(2026, 7, 28, 6), hours=(0, 3, 6),
        area=fetch.parse_area("30,-100,40,-90"), out=tmp_path / "gfs",
        progress=lambda line: None)

    leads = [url.split("pgrb2.0p25.f")[1][:3] for url in requested]
    assert leads == ["000", "003", "006"]


def test_the_full_file_backbone_is_driven_one_hour_at_a_time_in_order(
        tmp_path, monkeypatch):
    """``rw_fetch`` is called per file, analysis lead first.

    Full-file mode throughout: the ordering work is caller-side and
    changes nothing about which bytes are asked for.
    """

    messages = 124
    monkeypatch.setattr(fetch, "_gfs_index_record_count",
                        lambda url, **kwargs: messages)
    calls: list[dict] = []

    def run_fetch(binary, **kwargs):
        calls.append(kwargs)
        name = f"gfs.t06z.pgrb2.0p25.f{kwargs['hours'][0]:03d}"
        (kwargs["out"] / name).write_bytes(_grib2_stream(messages))
        return {"files": [{"name": name, "bytes": 1, "wall_seconds": 0.1,
                           "source": "aws", "mode": "full-file",
                           "mode_reason": "requested"}]}

    monkeypatch.setattr(rustwx_fetch, "run_fetch", run_fetch)
    fetch.fetch_gfs_fullfile(
        cycle=datetime(2026, 7, 28, 6), hours=(0, 3, 6), area=None,
        out=tmp_path / "gfs-full", engine="rust",
        engine_bin=tmp_path / "rw_fetch", progress=lambda line: None)

    assert [call["hours"] for call in calls] == [(0,), (3,), (6,)]
    assert {call["mode"] for call in calls} == {"full-file"}
