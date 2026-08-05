"""The radar feed selector: which route served an observation, and its lag.

Pure tests: no network, no GPU, no subprocess execution.  Both front doors
into ``rw_nexrad`` are replaced with stubs that return the records the real
binary prints, so what is under test is the selection, the refusals and the
receipt -- not the bin, which has its own suite in Rust.

The site ids here are four-character placeholders that are not real radars.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gpuwm.obs import radar_source
from gpuwm.obs.nexrad import ARCHIVE_FEED, LIVE_FEED
from gpuwm.obs.radar_source import (SOURCE_SCHEMA, SUPPORTED_SOURCES,
                                    RadarSourceError, acquire_volume,
                                    live_is_plausible)

SITE = "ZZZZ"
NOW = datetime(2026, 8, 5, 7, 50, 23, tzinfo=timezone.utc)
LIVE_VOLUME_TIME = "2026-08-05T07:48:50Z"
ARCHIVE_VOLUME_TIME = "2026-08-05T07:41:52Z"
ARCHIVE_FILENAME = f"{SITE}20260805_074152_V06"
ARCHIVE_KEY = f"2026/08/05/{SITE}/{ARCHIVE_FILENAME}"


def _live_record(out_dir: Path, *, chunks: int = 27, complete: bool = False,
                 observed_at: str | None = "2026-08-05T07:50:23Z",
                 lag_seconds: float | None = 0.0,
                 assembled: bool = True, admissible: bool = True,
                 refusal: str | None = None) -> dict:
    suffix = "" if complete else f"_P{chunks:03d}"
    filename = f"{SITE}20260805_074850_V06{suffix}"
    path = out_dir / filename
    if assembled:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"AR2V0006." + b"\0" * 40)
    volume = {
        "volume_id": 573,
        "volume_time": LIVE_VOLUME_TIME,
        "chunks": chunks,
        "listed_chunks": chunks,
        "complete": complete,
        "partial": not complete,
        "truncation": None,
        "bytes": 4663048,
        "first_chunk_key": f"{SITE}/573/20260805-074850-001-S",
        "newest_chunk_key": f"{SITE}/573/20260805-074850-{chunks:03d}-I",
        "newest_chunk_last_modified": "2026-08-05T07:50:23Z",
        "lag_seconds": lag_seconds,
        "keys": [f"{SITE}/573/20260805-074850-{n:03d}-"
                 f"{'S' if n == 1 else 'I'}" for n in range(1, chunks + 1)],
        "admissible": admissible,
        "refusal": refusal,
    }
    if assembled:
        volume |= {"filename": filename, "path": str(path),
                   "cache_path": str(path), "sha256": "a" * 64,
                   "chunk_cache_hits": 0}
    return {
        "schema": "gpuwm-obs.nexrad-live-fetch.v1",
        "status": "READY" if assembled else "EMPTY",
        "feed": LIVE_FEED,
        "site": SITE,
        "bucket": "unidata-nexrad-level2-chunks",
        "observed_at": observed_at,
        "clock_skew_seconds": 0.733,
        "retained_volume_ids": 616,
        "volume_id_runs": [[1, 573], [957, 999]],
        "probed_volume_ids": [573, 999],
        "requested_volumes": 1,
        "allow_partial": not complete,
        "min_chunks": 2,
        "matched_volumes": 1,
        "admissible_volumes": 1 if assembled else 0,
        "volumes": [volume],
        "total_bytes": 4663048,
    }


def _archive_list(*, observed_at: str | None = "2026-08-05T07:50:23Z",
                  volumes: list[dict] | None = None) -> dict:
    if volumes is None:
        volumes = [{
            "key": ARCHIVE_KEY,
            "filename": ARCHIVE_FILENAME,
            "valid_time": ARCHIVE_VOLUME_TIME,
            "format": "V06",
            "gzipped": False,
            "size_bytes": 16952558,
            "last_modified": "2026-08-05T07:48:25.000Z",
        }]
    return {
        "schema": "gpuwm-obs.nexrad-list.v1",
        "status": "READY" if volumes else "EMPTY",
        "feed": ARCHIVE_FEED,
        "window": {"site": SITE, "start": "", "end": "",
                   "bucket": "unidata-nexrad-level2", "day_prefixes": []},
        "observed_at": observed_at,
        "matched_volumes": len(volumes),
        "volumes": volumes,
        "total_bytes": sum(v["size_bytes"] for v in volumes),
    }


class Feeds:
    """Stubs for the two front doors, recording every call."""

    def __init__(self, out_dir: Path, *, live=None, archive_list=None,
                 live_error: Exception | None = None,
                 archive_error: Exception | None = None,
                 materialise: bool = True):
        self.out_dir = out_dir
        self.live = live
        self.archive_list = archive_list
        self.live_error = live_error
        self.archive_error = archive_error
        self.materialise = materialise
        self.calls: list[tuple[str, dict]] = []

    def run_live_list(self, binary, **kwargs):
        self.calls.append(("live-list", kwargs))
        if self.live_error is not None:
            raise self.live_error
        # `live-list` moves no payload, so it carries no assembled file.
        record = dict(self.live)
        record["schema"] = "gpuwm-obs.nexrad-live-list.v1"
        record["volumes"] = [
            {key: value for key, value in volume.items()
             if key not in ("filename", "path", "cache_path", "sha256",
                            "chunk_cache_hits")}
            for volume in self.live["volumes"]]
        return record

    def run_live_fetch(self, binary, **kwargs):
        self.calls.append(("live-fetch", kwargs))
        if self.live_error is not None:
            raise self.live_error
        return self.live

    def run_list(self, binary, **kwargs):
        self.calls.append(("list", kwargs))
        if self.archive_error is not None:
            raise self.archive_error
        return self.archive_list

    def run_fetch(self, binary, **kwargs):
        self.calls.append(("fetch", kwargs))
        listing = self.archive_list
        served = listing["volumes"][0]
        if self.materialise:
            (self.out_dir / served["filename"]).write_bytes(b"AR2V0006.")
        return {
            "schema": "gpuwm-obs.nexrad-fetch.v1",
            "status": "READY",
            "feed": ARCHIVE_FEED,
            "observed_at": listing["observed_at"],
            "files": [{"key": served["key"], "site": SITE,
                       "path": str(self.out_dir / served["filename"]),
                       "cache_path": "", "valid_time": served["valid_time"],
                       "format": "V06", "bytes": served["size_bytes"],
                       "sha256": "b" * 64, "cache_hit": False}],
        }

    def install(self, monkeypatch):
        monkeypatch.setattr(radar_source, "run_live_list", self.run_live_list)
        monkeypatch.setattr(radar_source, "run_live_fetch",
                            self.run_live_fetch)
        monkeypatch.setattr(radar_source, "run_list", self.run_list)
        monkeypatch.setattr(radar_source, "run_fetch", self.run_fetch)
        return self

    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.calls]


def _feeds(tmp_path, monkeypatch, **kwargs) -> Feeds:
    kwargs.setdefault("live", _live_record(tmp_path))
    kwargs.setdefault("archive_list", _archive_list())
    return Feeds(tmp_path, **kwargs).install(monkeypatch)


# -- the mode is a decision, and an unknown one is a refusal ---------------

def test_the_three_modes_are_the_supported_set_and_auto_is_the_default():
    assert SUPPORTED_SOURCES == ("live", "archive", "auto")
    assert radar_source.DEFAULT_SOURCE == "auto"


def test_an_unknown_source_is_refused_before_any_feed_is_asked(tmp_path,
                                                               monkeypatch):
    feeds = _feeds(tmp_path, monkeypatch)
    with pytest.raises(RadarSourceError, match="unknown radar source"):
        acquire_volume(Path("rw_nexrad"), site=SITE, out_dir=tmp_path,
                       source="latest", now=NOW)
    assert feeds.kinds() == [], "a bad mode must not reach the network"


# -- auto prefers live -----------------------------------------------------

def test_auto_prefers_the_live_feed_and_never_touches_the_archive(tmp_path,
                                                                  monkeypatch):
    feeds = _feeds(tmp_path, monkeypatch)
    selected = acquire_volume(Path("rw_nexrad"), site=SITE, out_dir=tmp_path,
                              source="auto", allow_partial=True, now=NOW)
    assert selected.feed == LIVE_FEED
    assert feeds.kinds() == ["live-list", "live-fetch"]
    assert selected.receipt["attempts"] == [
        {"feed": LIVE_FEED, "outcome": "selected", "reason": None}]


def test_auto_falls_back_to_the_archive_and_records_why(tmp_path, monkeypatch):
    feeds = _feeds(tmp_path, monkeypatch,
                   live_error=RuntimeError("rw_nexrad live-fetch: the site "
                                           "is not publishing chunks"))
    selected = acquire_volume(Path("rw_nexrad"), site=SITE, out_dir=tmp_path,
                              source="auto", now=NOW)
    assert selected.feed == ARCHIVE_FEED
    assert feeds.kinds() == ["live-list", "list", "fetch"]
    attempts = selected.receipt["attempts"]
    assert [a["feed"] for a in attempts] == [LIVE_FEED, ARCHIVE_FEED]
    assert attempts[0]["outcome"] == "failed"
    assert "not publishing chunks" in attempts[0]["reason"]
    assert attempts[1]["outcome"] == "selected"


def test_auto_declines_the_live_feed_for_a_valid_time_it_cannot_cover(
        tmp_path, monkeypatch):
    feeds = _feeds(tmp_path, monkeypatch)
    old = NOW - timedelta(hours=6)
    selected = acquire_volume(Path("rw_nexrad"), site=SITE, out_dir=tmp_path,
                              valid_time=old, source="auto", now=NOW,
                              max_offset_seconds=1e9)
    assert selected.feed == ARCHIVE_FEED
    assert "live-list" not in feeds.kinds(), (
        "a request the live feed provably cannot cover costs no request")
    declined = selected.receipt["attempts"][0]
    assert declined == {"feed": LIVE_FEED, "outcome": "declined",
                        "reason": declined["reason"]}
    assert "ceiling" in declined["reason"]


# -- an explicit mode is honoured exactly ----------------------------------

def test_source_live_refuses_rather_than_quietly_becoming_the_archive(
        tmp_path, monkeypatch):
    feeds = _feeds(tmp_path, monkeypatch,
                   live_error=RuntimeError("no chunks"))
    with pytest.raises(RadarSourceError, match="live feed could not serve"):
        acquire_volume(Path("rw_nexrad"), site=SITE, out_dir=tmp_path,
                       source="live", now=NOW)
    assert feeds.kinds() == ["live-list"], (
        "an explicit live request must not fall back")


def test_source_archive_never_asks_the_live_feed(tmp_path, monkeypatch):
    feeds = _feeds(tmp_path, monkeypatch)
    selected = acquire_volume(Path("rw_nexrad"), site=SITE, out_dir=tmp_path,
                              source="archive", now=NOW)
    assert selected.feed == ARCHIVE_FEED
    assert "live-list" not in feeds.kinds()


# -- partial volumes are marked, never implied -----------------------------

def test_a_partial_live_volume_is_marked_partial_and_carries_its_chunk_count(
        tmp_path, monkeypatch):
    _feeds(tmp_path, monkeypatch, live=_live_record(tmp_path, chunks=27))
    selected = acquire_volume(Path("rw_nexrad"), site=SITE, out_dir=tmp_path,
                              source="live", allow_partial=True, now=NOW)
    assert selected.partial is True
    assert selected.chunks == 27
    assert selected.receipt["complete"] is False
    assert selected.receipt["listed_chunks"] == 27
    assert selected.filename.endswith("_P027"), (
        "a partial must not be named like a finished volume")


def test_a_complete_live_volume_is_not_marked_partial(tmp_path, monkeypatch):
    _feeds(tmp_path, monkeypatch,
           live=_live_record(tmp_path, chunks=106, complete=True))
    selected = acquire_volume(Path("rw_nexrad"), site=SITE, out_dir=tmp_path,
                              source="live", now=NOW)
    assert selected.partial is False
    assert selected.receipt["complete"] is True
    assert not selected.filename.endswith(")")
    assert "_P" not in selected.filename


def test_partial_is_opt_in_and_the_choice_reaches_the_front_door(tmp_path,
                                                                 monkeypatch):
    feeds = _feeds(tmp_path, monkeypatch,
                   live=_live_record(tmp_path, chunks=106, complete=True))
    acquire_volume(Path("rw_nexrad"), site=SITE, out_dir=tmp_path,
                   source="live", now=NOW)
    assert feeds.calls[0][1]["allow_partial"] is False
    feeds.calls.clear()
    acquire_volume(Path("rw_nexrad"), site=SITE, out_dir=tmp_path,
                   source="live", allow_partial=True, min_chunks=7, now=NOW)
    assert feeds.calls[0][1]["allow_partial"] is True
    assert feeds.calls[0][1]["min_chunks"] == 7


# -- provenance: which feed, which objects, what lag -----------------------

def test_the_receipt_names_the_feed_every_object_and_the_measured_lag(
        tmp_path, monkeypatch):
    _feeds(tmp_path, monkeypatch, live=_live_record(tmp_path, chunks=27))
    selected = acquire_volume(Path("rw_nexrad"), site=SITE, out_dir=tmp_path,
                              source="live", allow_partial=True, now=NOW)
    receipt = selected.receipt
    assert receipt["schema"] == SOURCE_SCHEMA
    assert receipt["feed"] == LIVE_FEED
    assert receipt["requested_source"] == "live"
    assert receipt["object_count"] == 27
    assert len(receipt["object_keys"]) == 27
    assert receipt["object_keys"][0].endswith("-001-S")
    assert receipt["lag_seconds"] == 0.0
    assert receipt["lag_measured_against"] == "s3-date-header"
    assert receipt["observed_at"] == "2026-08-05T07:50:23Z"
    assert receipt["clock_skew_seconds"] == 0.733
    assert receipt["volume_sha256"] == "a" * 64


def test_the_archive_lag_is_the_newest_object_against_the_bucket_clock(
        tmp_path, monkeypatch):
    _feeds(tmp_path, monkeypatch)
    selected = acquire_volume(Path("rw_nexrad"), site=SITE, out_dir=tmp_path,
                              source="archive", now=NOW)
    # 07:50:23Z answered, object stamped 07:48:25Z.
    assert selected.lag_seconds == pytest.approx(118.0)
    assert selected.receipt["lag_measured_against"] == "s3-date-header"
    assert selected.receipt["object_keys"] == [ARCHIVE_KEY]
    assert selected.receipt["chunks"] is None, (
        "an archived volume is not made of chunks and must not claim a count")


def test_an_unreadable_bucket_clock_is_stated_as_unmeasured(tmp_path,
                                                            monkeypatch):
    _feeds(tmp_path, monkeypatch, archive_list=_archive_list(observed_at=None))
    selected = acquire_volume(Path("rw_nexrad"), site=SITE, out_dir=tmp_path,
                              source="archive", now=NOW)
    assert selected.lag_seconds is None
    assert selected.receipt["lag_measured_against"] == "unmeasured"


def test_the_offset_from_the_requested_time_is_signed_and_recorded(
        tmp_path, monkeypatch):
    _feeds(tmp_path, monkeypatch)
    target = datetime(2026, 8, 5, 7, 45, 0, tzinfo=timezone.utc)
    selected = acquire_volume(Path("rw_nexrad"), site=SITE, out_dir=tmp_path,
                              valid_time=target, source="archive", now=NOW)
    # the volume starts at 07:41:52Z, 188 s before the analysis time
    assert selected.offset_seconds == pytest.approx(-188.0)
    assert selected.receipt["requested_valid_time"] == "2026-08-05T07:45:00Z"


# -- fail closed -----------------------------------------------------------

def test_a_live_record_with_nothing_assembled_is_a_refusal(tmp_path,
                                                           monkeypatch):
    _feeds(tmp_path, monkeypatch,
           live=_live_record(tmp_path, assembled=False))
    with pytest.raises(RadarSourceError, match="assembled no volume"):
        acquire_volume(Path("rw_nexrad"), site=SITE, out_dir=tmp_path,
                       source="live", now=NOW)


def test_an_empty_archive_window_is_a_refusal_not_an_empty_observation(
        tmp_path, monkeypatch):
    _feeds(tmp_path, monkeypatch, archive_list=_archive_list(volumes=[]))
    with pytest.raises(RadarSourceError, match="holds no volume"):
        acquire_volume(Path("rw_nexrad"), site=SITE, out_dir=tmp_path,
                       source="archive", now=NOW)


def test_a_volume_beyond_the_offset_ceiling_is_refused_by_name(tmp_path,
                                                               monkeypatch):
    _feeds(tmp_path, monkeypatch)
    target = datetime(2026, 8, 5, 7, 0, 0, tzinfo=timezone.utc)
    with pytest.raises(RadarSourceError, match="beyond the 480 s ceiling"):
        acquire_volume(Path("rw_nexrad"), site=SITE, out_dir=tmp_path,
                       valid_time=target, source="archive", now=NOW,
                       max_offset_seconds=480.0)


def test_a_fetch_that_leaves_no_file_is_a_refusal(tmp_path, monkeypatch):
    _feeds(tmp_path, monkeypatch, materialise=False)
    with pytest.raises(RadarSourceError, match="did not materialise"):
        acquire_volume(Path("rw_nexrad"), site=SITE, out_dir=tmp_path,
                       source="archive", now=NOW)


def test_both_feeds_failing_names_both_attempts(tmp_path, monkeypatch):
    _feeds(tmp_path, monkeypatch, live_error=RuntimeError("live is down"),
           archive_error=RuntimeError("archive is down"))
    with pytest.raises(RadarSourceError) as caught:
        acquire_volume(Path("rw_nexrad"), site=SITE, out_dir=tmp_path,
                       source="auto", now=NOW)
    message = str(caught.value)
    assert "live is down" in message and "archive is down" in message


def test_the_metadata_sidecar_is_never_selected_as_a_volume(tmp_path,
                                                            monkeypatch):
    # `..._V06_MDM` sorts after the volume it describes, so a naive
    # max-by-key picks it; it is not a volume and decode would refuse it.
    sidecar = {"key": ARCHIVE_KEY + "_MDM",
               "filename": ARCHIVE_FILENAME + "_MDM",
               "valid_time": ARCHIVE_VOLUME_TIME, "format": "V06",
               "gzipped": False, "size_bytes": 734632,
               "last_modified": "2026-08-05T07:48:26.000Z"}
    listing = _archive_list()
    listing["volumes"].append(sidecar)
    _feeds(tmp_path, monkeypatch, archive_list=listing)
    selected = acquire_volume(Path("rw_nexrad"), site=SITE, out_dir=tmp_path,
                              source="archive", now=NOW)
    assert selected.filename == ARCHIVE_FILENAME


def test_only_the_volume_that_won_the_listing_is_assembled(tmp_path,
                                                           monkeypatch):
    """List first, then fetch the one that won, by id.

    A named analysis time asks the live route about several volumes and
    uses one.  Each is ~110 objects and ~17 MB, so choosing on the
    listing -- which moves no payload -- and naming the winner is the
    difference between one assembly and four.
    """

    feeds = _feeds(tmp_path, monkeypatch)
    acquire_volume(Path("rw_nexrad"), site=SITE, out_dir=tmp_path,
                   valid_time=NOW, source="live", allow_partial=True,
                   max_offset_seconds=600.0, now=NOW)
    assert feeds.kinds() == ["live-list", "live-fetch"]
    assert feeds.calls[0][1]["volumes"] == \
        radar_source.LIVE_VOLUMES_FOR_A_NAMED_TIME
    assert feeds.calls[0][1].get("volume_id") is None
    assert feeds.calls[1][1]["volume_id"] == 573, (
        "the fetch must name the volume the listing chose")


def test_a_listing_that_offers_nothing_refuses_before_any_payload_moves(
        tmp_path, monkeypatch):
    feeds = _feeds(tmp_path, monkeypatch,
                   live=_live_record(tmp_path, admissible=False,
                                     refusal="the scan is still in progress"))
    with pytest.raises(RadarSourceError, match="still in progress"):
        acquire_volume(Path("rw_nexrad"), site=SITE, out_dir=tmp_path,
                       source="live", now=NOW)
    assert feeds.kinds() == ["live-list"]


def test_a_volume_id_reused_between_listing_and_fetch_is_refused(tmp_path,
                                                                 monkeypatch):
    """The id space is a counter that wraps; identity is the start time.

    A fetch that comes back with a different volume start time under the
    id that was chosen is not a fresher version of the same scan, it is a
    different scan.
    """

    feeds = _feeds(tmp_path, monkeypatch)
    moved = _live_record(tmp_path)
    moved["volumes"][0]["volume_time"] = "2026-08-05T07:55:49Z"
    feeds.live_after_list = moved

    def fetch(binary, **kwargs):
        feeds.calls.append(("live-fetch", kwargs))
        return moved

    monkeypatch.setattr(radar_source, "run_live_fetch", fetch)
    with pytest.raises(RadarSourceError, match="reused between the listing"):
        acquire_volume(Path("rw_nexrad"), site=SITE, out_dir=tmp_path,
                       source="live", allow_partial=True, now=NOW)


# -- the auto branch, without a socket -------------------------------------

def test_live_plausibility_is_a_pure_function_of_the_clock():
    assert live_is_plausible(None, NOW) == (True, None)
    recent = NOW - timedelta(minutes=5)
    assert live_is_plausible(recent, NOW)[0] is True
    stale, why = live_is_plausible(NOW - timedelta(hours=6), NOW)
    assert stale is False and "ceiling" in why
    ahead, why_ahead = live_is_plausible(NOW + timedelta(hours=6), NOW)
    assert ahead is False and "future" in why_ahead


# -- the standing rule -----------------------------------------------------

def test_no_real_site_tokens_in_the_feed_selector():
    import re
    root = Path(__file__).resolve().parent.parent
    for relative in ("gpuwm/obs/radar_source.py",
                     "tools/obs_radar_grid_build.py"):
        text = (root / relative).read_text(encoding="utf-8")
        assert not re.search(r'default\s*=\s*"[A-Z][A-Z0-9]{3}"', text), (
            f"{relative} must not default a site id")
        for token in ("KDMX", "KBMX", "KTLX", "KEAX", "KLOT"):
            assert token not in text, f"{relative} names {token}"
