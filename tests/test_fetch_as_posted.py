"""The as-posted fetch against a local replay of a posting cycle (A136 L2).

A local HTTP server serves one cycle's objects of a real table route and
reveals lead ``k`` only once an injected clock passes its reveal time, so
the loop is driven exactly as a live cycle drives it: asked by the
polling rule, fetched the moment one host holds the whole lead, verified,
and published as a per-lead marker in lead order.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import http.server
import json
import threading
from pathlib import Path

import pytest

from gpuwm import fetch
from gpuwm import fetch_as_posted
from gpuwm import fetch_endpoints
from gpuwm import fetch_routes
from gpuwm import source_posting as rows
from gpuwm import source_readiness as readiness
from gpuwm.fetch_endpoints import Endpoint

SOURCE = "gefs"
CYCLE = datetime(2026, 9, 29, 12)


class Clock:
    """Virtual naive-UTC time: every wait the loop takes moves it on."""

    def __init__(self, start: datetime) -> None:
        self._now = start
        self._lock = threading.Lock()

    def now(self) -> datetime:
        with self._lock:
            return self._now

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self._now += timedelta(seconds=max(float(seconds), 0.001))


def payload(name: str) -> bytes:
    body = hashlib.sha256(name.encode()).digest() * 8
    return b"GRIB" + body + b"7777"


class Replay:
    """One cycle's objects on a local host, each revealed at its time."""

    def __init__(self, clock: Clock, plan, reveal) -> None:
        self.clock = clock
        self.objects: dict[str, tuple[bytes, datetime | None]] = {}
        self.unheard_until: dict[str, datetime] = {}
        self.truncate_once: set[str] = set()
        self.gets: list[str] = []
        for obj in plan.objects:
            at = reveal(obj.lead if obj.lead in plan.leads else plan.leads[0])
            self.objects["/" + obj.key] = (payload(obj.key), at)
            idx = [row.idx_sidecar for row in plan.files if row.role == obj.role][0]
            if idx:
                self.objects["/" + obj.key + idx] = (b"1:0:d=x\n", at)
        replay = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):  # quiet
                pass

            def _answer(self, body: bool) -> None:
                path = self.path.split("/", 2)[-1]
                path = "/" + path
                now = replay.clock.now()
                blocked = replay.unheard_until.get(path)
                if blocked is not None and now < blocked:
                    self.send_response(503)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                found = replay.objects.get(path)
                if found is None or found[1] is None or now < found[1]:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                data = found[0]
                if body and path in replay.truncate_once:
                    replay.truncate_once.discard(path)
                    data = data[:-4] + b"XXXX"
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                if body:
                    replay.gets.append(path)
                    self.wfile.write(data)

            def do_HEAD(self):
                self._answer(False)

            def do_GET(self):
                self._answer(True)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}/replay"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def replay(monkeypatch):
    """A factory: ``replay(reveal)`` serves the gefs 12 h window locally."""

    made = []

    def build(reveal, *, hours: int = 12):
        start = rows.expected_at(SOURCE, CYCLE, 0) - timedelta(minutes=10)
        clock = Clock(start)
        plan = fetch_routes.resolve_request(SOURCE, cycle=CYCLE, hours=hours,
                                            cadence=3)
        server = Replay(clock, plan, lambda lead: reveal(lead, clock))
        made.append(server)
        rung = Endpoint(name="nomads", base=server.base, retention_hours=None,
                        why="the local replay of a posting cycle")
        monkeypatch.setattr(fetch_endpoints, "serving_ladder",
                            lambda source_id, **_kw: (rung,))
        monkeypatch.setattr(fetch_endpoints, "SETTLE_BACKOFF_S", ())
        monkeypatch.setattr(fetch_as_posted, "now_utc", clock.now)
        monkeypatch.setattr(fetch_as_posted, "pause", clock.sleep)
        # A transfer that does not verify is asked again by the loop, not
        # after the transport's own real-time backoff.
        monkeypatch.setattr(fetch_routes, "_TRANSFER_ATTEMPTS", 1)
        monkeypatch.setattr(fetch_routes, "_retry_delay",
                            lambda error, attempt: None)
        return server, clock

    yield build
    for server in made:
        server.close()


def on_schedule(lead: int, clock: Clock) -> datetime:
    """Lead ``lead`` posts 20 s after the table says it will."""

    return rows.expected_at(SOURCE, CYCLE, lead) + timedelta(seconds=20)


def fetch_args(out: Path, *extra: str, hours: int = 12) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    fetch.register_cli(parser.add_subparsers())
    return parser.parse_args([
        "fetch", "--source", SOURCE, "--cycle", CYCLE.strftime("%Y-%m-%dT%H"),
        "--hours", str(hours), "--cadence", "3", "--out", str(out),
        "--fetch-workers", "1", *extra])


def markers(out: Path) -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((out / "posting").glob("f[0-9][0-9][0-9].json"))]


def test_markers_land_in_lead_order_as_each_lead_posts(replay, tmp_path):
    server, clock = replay(on_schedule)
    out = tmp_path / "gefs"
    seen_order: list[int] = []
    real_publish = fetch_as_posted.PostingLoop.publish

    def publish(self, lead, objects, **named):
        seen_order.append(lead)
        # A lead's marker is written only once it is on disk and verified,
        # and never before the host held it.
        assert clock.now() >= on_schedule(lead, clock)
        return real_publish(self, lead, objects, **named)

    fetch_as_posted.PostingLoop.publish = publish
    try:
        assert fetch.fetch_main(fetch_args(out)) == 0
    finally:
        fetch_as_posted.PostingLoop.publish = real_publish
    assert seen_order == [0, 3, 6, 9, 12]
    written = markers(out)
    assert [marker["lead"] for marker in written] == [0, 3, 6, 9, 12]
    for marker in written:
        assert marker["schema"] == "gpuwm.posted-lead.v1"
        assert marker["source"] == SOURCE and marker["cycle"] == "2026-09-29T12"
        # The lead's two fetched objects, each with its own lead, and apart
        # from them the pair the route composes from them: an as-posted
        # mapped seal holds the composed primary's manifest row to this
        # marker (A136 L3 ii), and ``objects`` stays what was fetched.
        fetched = marker["objects"]
        composed = marker["composed"]
        assert len(fetched) == 2 and len(composed) == 1
        for item in fetched:
            assert set(item) == {"role", "name", "url", "endpoint", "bytes",
                                 "sha256", "lead"}
            assert item["lead"] == marker["lead"]
        assert set(composed[0]) == {"name", "bytes", "sha256", "lead",
                                    "parts"}
        assert composed[0]["lead"] == marker["lead"]
        assert sorted(composed[0]["parts"]) == sorted(
            item["name"] for item in fetched)
        for item in [*fetched, *composed]:
            data = (out / item["name"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == item["sha256"]
            assert len(data) == item["bytes"]
        assert marker["first_seen_at"] <= marker["fetched_at"]
    schedule = json.loads((out / "posting" / "schedule.json").read_text())
    for key in ("source", "member", "cycle", "as_posted", "shape", "streams",
                "why", "late_after_minutes", "start_needs",
                "expected_ready_at", "expected_final_at", "table_sha256"):
        assert key in schedule, key
    assert schedule["schema"] == "gpuwm.posting-schedule.v1"
    assert [need["role"] for need in schedule["start_needs"]] == [
        "analysis", "first_boundary"]
    for row in schedule["leads"]:
        assert set(("lead", "valid_time", "expected_at", "late_at",
                    "first_seen_at", "fetched_at", "endpoint",
                    "state")) <= set(row)
        assert row["state"] == "ready" and row["endpoint"] == "nomads"
    manifest = json.loads((out / "fetch-manifest.json").read_text())
    assert manifest["complete"] is True
    assert not (out / "posting" / "failed.json").exists()


def test_lead_ready_says_the_fetched_bytes_of_a_route_that_composes(
        replay, tmp_path):
    """DESIGN A136 3.5: ``lead_ready.bytes`` is what was fetched and verified.

    gefs composes each lead's two objects into one file, which is its
    parts' bytes again; with that file among the marker's objects, go's
    chain stream said every composed lead moved twice its bytes.  The
    markers here are the real loop's, relayed by the real chain stream.
    """

    from gpuwm.chain_events import (CHAIN_EVENTS_FILENAME, GoChainEvents,
                                    read_chain_events)

    replay(on_schedule)
    out = tmp_path / "gefs"
    assert fetch.fetch_main(fetch_args(out)) == 0
    fetched = {}
    for marker in markers(out):
        assert marker["composed"], "gefs composes each lead's parts"
        files = [out / item["name"] for item in marker["objects"]]
        fetched[marker["lead"]] = sum(path.stat().st_size for path in files)
        composed = sum(item["bytes"] for item in marker["composed"])
        assert composed == fetched[marker["lead"]]
    run = tmp_path / "run"
    run.mkdir()
    chain = GoChainEvents()
    chain.open(tmp_path / CHAIN_EVENTS_FILENAME,
               plan={"data": out, "run": run, "render": None})
    chain.finish(status="SUCCESS")
    ready = {record["lead"]: record["bytes"]
             for record in read_chain_events(tmp_path / CHAIN_EVENTS_FILENAME)
             if record["event"] == "lead_ready"}
    assert ready == fetched


def test_a_late_lead_stops_with_75_and_keeps_the_prefix(replay, tmp_path):
    def reveal(lead, clock):
        return None if lead == 6 else on_schedule(lead, clock)

    server, clock = replay(reveal)
    out = tmp_path / "gefs"
    assert fetch.fetch_main(fetch_args(out)) == 75
    failed = json.loads((out / "posting" / "failed.json").read_text())
    assert failed["code"] == "source_behind" and failed["lead"] == 6
    assert failed["heard"] is True and failed["last_answer"] == "not_posted"
    # The row's own budget, not a literal: the L1 follow-ups (d25ed8127)
    # raised gefs from 60 to 120 min, its measured late spread plus 30,
    # and a literal 60 here failed once both lanes met on integrate/2.8.
    budget = rows.late_after_minutes(SOURCE)
    assert budget is not None and budget >= 60
    assert failed["late_at"] == rows.instant(
        rows.expected_at(SOURCE, CYCLE, 6) + timedelta(minutes=budget))
    assert failed["late_after_minutes"] == budget
    assert "has not posted by" in failed["message"]
    # Not asked before its due time, and stopped only past its late time.
    assert clock.now() >= readiness.parse_instant(failed["late_at"])
    assert [marker["lead"] for marker in markers(out)] == [0, 3]
    manifest = json.loads((out / "fetch-manifest.json").read_text())
    assert manifest["complete"] is False
    assert sorted({entry["lead"] for entry in manifest["files"]}) == [0, 3]
    schedule = json.loads((out / "posting" / "schedule.json").read_text())
    states = {row["lead"]: row["state"] for row in schedule["leads"]}
    assert states[0] == "ready" and states[3] == "ready" and states[6] == "late"


def test_a_host_not_heard_is_never_counted_absent(replay, tmp_path):
    server, clock = replay(on_schedule)
    key = [path for path in server.objects if path.endswith(".f003")][0]
    # The host answers 503 for f003 for 40 minutes past its posting: the
    # loop keeps asking (not heard is not absent) and fetches it after.
    server.unheard_until[key] = on_schedule(3, clock) + timedelta(minutes=40)
    out = tmp_path / "gefs"
    assert fetch.fetch_main(fetch_args(out)) == 0
    assert [marker["lead"] for marker in markers(out)] == [0, 3, 6, 9, 12]
    seen = {marker["lead"]: marker["first_seen_at"] for marker in markers(out)}
    assert readiness.parse_instant(seen[3]) >= server.unheard_until[key]


def test_a_host_never_heard_is_said_as_not_heard(replay, tmp_path):
    server, clock = replay(on_schedule)
    for path in list(server.objects):
        if path.endswith(".f003"):
            server.unheard_until[path] = datetime(2100, 1, 1)
    out = tmp_path / "gefs"
    assert fetch.fetch_main(fetch_args(out)) == 75
    failed = json.loads((out / "posting" / "failed.json").read_text())
    assert failed["lead"] == 3 and failed["heard"] is False
    assert failed["last_answer"] == "not_heard"
    assert "could not be heard" in failed["message"]
    assert "has not posted" not in failed["message"]


def test_a_verification_failure_is_fetched_again(replay, tmp_path):
    server, clock = replay(on_schedule)
    key = [path for path in server.objects if path.endswith("pgrb2a.0p50.f006")][0]
    server.truncate_once.add(key)
    out = tmp_path / "gefs"
    assert fetch.fetch_main(fetch_args(out)) == 0
    # The truncated GET, then the one asked again on the next round.
    assert server.gets.count(key) == 2
    marker = [m for m in markers(out) if m["lead"] == 6][0]
    item = [i for i in marker["objects"] if i["name"].endswith("pgrb2a.0p50.f006")][0]
    assert item["sha256"] == hashlib.sha256(payload(key[1:])).hexdigest()


def test_side_by_side_backlog_keeps_markers_in_lead_order(replay, tmp_path):
    # Every lead already posted (a late launch): moved side by side under
    # the host caps, the markers still in lead order.
    server, clock = replay(lambda lead, clock: clock.now() - timedelta(hours=1))
    out = tmp_path / "gefs"
    args = fetch_args(out)
    args.fetch_workers = 4
    assert fetch.fetch_main(args) == 0
    assert [marker["lead"] for marker in markers(out)] == [0, 3, 6, 9, 12]


def test_a_late_launch_marks_every_lead_up_when_first_asked(replay, tmp_path):
    """Through the fetch door: a window whose final lead had posted before
    the fetch began moves with no per-lead ask, and every lead says it was
    already up (measured live on gefs 2026-10-01T06, six hours late: each
    lead read 375 to 377 min after its scheduled time, unflagged)."""

    server, clock = replay(lambda lead, clock: clock.now() - timedelta(hours=1))
    out = tmp_path / "gefs"
    assert fetch.fetch_main(fetch_args(out)) == 0
    assert [marker["posted_when_first_asked"] for marker in markers(out)] == [
        True] * 5
    schedule = json.loads((out / "posting" / "schedule.json").read_text())
    assert [row["posted_when_first_asked"] for row in schedule["leads"]] == [
        True] * 5


def comparable(manifest: dict) -> dict:
    """A manifest without its timing and posting fields (DESIGN A136 2.3).

    Where and when each object moved -- its URL and endpoint, seconds,
    whether this run moved or reused it, the pool receipt, the host
    receipt -- and the as-posted ``complete`` flag.
    """

    kept = {key: value for key, value in manifest.items()
            if key not in ("concurrency", "endpoints", "complete")}
    kept["files"] = [{key: value for key, value in entry.items()
                      if key not in ("url", "endpoint", "seconds", "reused")}
                     for entry in manifest["files"]]
    return kept


def test_as_posted_manifest_equals_the_one_shot_manifest(replay, tmp_path):
    server, clock = replay(on_schedule)
    posted = tmp_path / "posted"
    assert fetch.fetch_main(fetch_args(posted)) == 0
    whole = tmp_path / "whole"
    assert fetch.fetch_main(fetch_args(whole, "--whole-cycle")) == 0
    assert not (whole / "posting").exists()
    one = json.loads((posted / "fetch-manifest.json").read_text())
    two = json.loads((whole / "fetch-manifest.json").read_text())
    assert "complete" not in two
    assert comparable(one) == comparable(two)
    assert (posted / "SHA256SUMS").read_bytes() == (whole / "SHA256SUMS").read_bytes()


def test_wait_for_is_as_posted_for_a_table_route(replay, tmp_path):
    server, clock = replay(on_schedule)
    out = tmp_path / "gefs"
    args = fetch_args(out, "--wait-for", "--wait-timeout-minutes", "600")
    assert args.as_posted is True
    assert fetch.fetch_main(args) == 0
    assert [marker["lead"] for marker in markers(out)] == [0, 3, 6, 9, 12]


def test_the_whole_window_cap_stops_with_75(replay, tmp_path):
    def reveal(lead, clock):
        return on_schedule(lead, clock) + timedelta(minutes=30 * (lead > 0))

    server, clock = replay(reveal)
    out = tmp_path / "gefs"
    assert fetch.fetch_main(fetch_args(
        out, "--wait-timeout-minutes", "20")) == 75
    failed = json.loads((out / "posting" / "failed.json").read_text())
    assert failed["budget"] == "wait_timeout_minutes"
    assert "--wait-timeout-minutes" in failed["message"]


@pytest.mark.parametrize("extra, said", [
    (("--whole-cycle", "--late-after-minutes", "30"), "--whole-cycle waits for nothing"),
    (("--late-after-minutes", "0"), "positive and finite"),
    (("--no-probe",), "--no-probe belongs to --readiness"),
])
def test_posting_flags_refuse_what_they_cannot_mean(tmp_path, extra, said):
    with pytest.raises(ValueError, match=said):
        fetch.fetch_main(fetch_args(tmp_path / "x", *extra))


@pytest.mark.parametrize("whole", [True, False])
def test_a_forced_legacy_refetch_leaves_no_marker_of_a_moved_file(
        tmp_path, monkeypatch, whole):
    """A forced refetch sets the old payloads aside; a lead marker left
    beside them would tell a reader (L3's PostedLeads, the tree consumer)
    that a lead is ready whose file is gone, until it is fetched again."""

    cycle = datetime(2026, 9, 29, 12)
    window = readiness.resolve_window("gfs", cycle, 3, cadence=3)
    out = tmp_path / "gfs"
    folder = out / "posting"
    folder.mkdir(parents=True)
    for lead in window.leads:
        (folder / fetch_as_posted.marker_name(lead)).write_text(json.dumps({
            "schema": fetch_as_posted.MARKER_SCHEMA, "source": "gfs",
            "member": None, "cycle": "2026-09-29T12", "lead": lead,
            "objects": [{"name": "old"}]}), encoding="utf-8")
    (out / "old").write_bytes(b"old")
    clock = Clock(rows.expected_at("gfs", cycle, window.leads[-1])
                  + timedelta(minutes=5))
    monkeypatch.setattr(fetch_as_posted, "pause", clock.sleep)
    loop = fetch_as_posted.PostingLoop(
        window, out, probe=lambda url: True, now=clock.now,
        sleep=clock.sleep, progress=lambda *_: None)
    seen = []

    def transfer(hours, force):
        seen.append(sorted(path.name for path in
                           folder.glob("f[0-9][0-9][0-9].json")))
        manifest = out / "fetch-manifest.json"
        manifest.write_text(json.dumps({"files": [
            {"forecast_hour": lead, "name": f"new.f{lead:03d}", "role": "atmos",
             "url": "u", "endpoint": "nomads", "bytes": 3, "sha256": "x"}
            for lead in hours]}), encoding="utf-8")
        return manifest

    fetch._legacy_transfer(loop, transfer, list(window.leads), force=True,
                           whole=whole)
    assert seen[0] == []
    for lead in window.leads:
        marker = json.loads((folder / fetch_as_posted.marker_name(lead)).read_text())
        assert marker["objects"][0]["name"] == f"new.f{lead:03d}"


def test_the_posting_file_names_are_the_ones_the_go_relay_reads():
    """The writer's names live in a module the RW-WPS wheel stages; the
    relay's module is not in that wheel and carries its own copy.  A
    drift between the two would have go's stream watch a folder the fetch
    never writes."""

    from gpuwm import chain_events

    for name in ("POSTING_DIRNAME", "POSTING_SCHEDULE_NAME",
                 "POSTING_FAILED_NAME"):
        assert getattr(rows, name) == getattr(chain_events, name), name


def test_fetch_table_rows_carry_as_posted_and_the_budget():
    assert fetch.FETCH_HINT_ROWS["as_posted"].default is True
    fetch.validate_fetch_hints({"source": "gefs", "as_posted": False},
                               source="t.toml")
    fetch.validate_fetch_hints({"source": "gefs", "late_after_minutes": 90},
                               source="t.toml")
    with pytest.raises(ValueError, match="positive number of minutes"):
        fetch.validate_fetch_hints({"source": "gefs", "late_after_minutes": 0},
                                   source="t.toml")
    with pytest.raises(ValueError, match="waits for nothing"):
        fetch.validate_fetch_hints({"source": "gefs", "as_posted": False,
                                    "late_after_minutes": 30}, source="t.toml")


def _watched_loop(tmp_path, reveal, *, hours: int = 6):
    """A gefs window's loop on the virtual clock, its probe answering by
    ``reveal(lead)`` and counting every ask."""

    import re

    plan = fetch_routes.resolve_request(SOURCE, cycle=CYCLE, hours=hours,
                                        cadence=3)
    window = readiness.Window(source=SOURCE, cycle=CYCLE,
                              leads=tuple(plan.leads), cadence=3, member=None,
                              transport=None, plan=plan, hours=hours)
    clock = Clock(rows.expected_at(SOURCE, CYCLE, 0) + timedelta(seconds=25))
    asked: list[int] = []

    def probe(url):
        lead = int(re.search(r"\.f(\d{3})", url.rsplit("/", 1)[-1]).group(1))
        asked.append(lead)
        at = reveal(lead)
        return at is not None and clock.now() >= at

    loop = fetch_as_posted.PostingLoop(window, tmp_path / SOURCE, probe=probe,
                                       now=clock.now, sleep=clock.sleep,
                                       progress=lambda *_: None)
    return loop, clock, asked


def test_a_lead_is_dated_when_it_posts_not_when_the_transfer_reaches_it(
        tmp_path, monkeypatch):
    """DESIGN A136 2.2 and 3.5: ``first_seen_at`` (and ``lead_posted``) is
    when one host first held the lead.  Measured live 2026-10-01: a source
    of about 125 objects per lead posted faster than the fetch moved them,
    and because a lead was asked only when the transfer reached it, each
    lead was said posted later and later after its scheduled time while
    the host had it on time.  The posting watch asks a due lead every
    ``poll_seconds`` whatever the transfer reaches; its rounds are driven
    by hand here on the virtual clock, and the transfer reaches f006 ten
    minutes after it posted."""

    monkeypatch.setattr(fetch_as_posted.PostingLoop, "watch", lambda self: None)
    loop, clock, asked = _watched_loop(tmp_path, lambda lead: on_schedule(lead, None))
    loop.start()
    loop.wait_start()
    posted = on_schedule(6, None)
    poll = loop.poll
    while clock.now() < posted + timedelta(seconds=3 * poll):
        loop._watch_round()
        clock.sleep(fetch_as_posted.WATCH_WAKE_SECONDS)
    clock.sleep(600)
    before = len(asked)
    assert loop(6)
    assert len(asked) == before, "the gate asked a lead the watch had seen"
    row = [row for row in loop.document["leads"] if row["lead"] == 6][0]
    seen = readiness.parse_instant(row["first_seen_at"])
    assert row["state"] == "posted"
    assert posted - timedelta(seconds=1) <= seen <= posted + timedelta(
        seconds=poll + fetch_as_posted.WATCH_WAKE_SECONDS)


def test_the_posting_watch_keeps_dating_leads_after_one_whose_host_was_not_heard(
        tmp_path, monkeypatch):
    """Measured live on aigfs 2026-10-01T00: its analysis lead was let
    through with its host not heard, the watch stopped at that lead, and
    every later lead was dated by its transfer again.  A lead the transfer
    is trying unheard is left to it; the leads after it are still watched."""

    monkeypatch.setattr(fetch_as_posted.PostingLoop, "watch", lambda self: None)
    loop, clock, asked = _watched_loop(tmp_path, lambda lead: on_schedule(lead, None))
    loop.start()
    loop.wait_start()
    with loop._lock:
        # The analysis lead as the start wait lets it through unheard.
        loop._seen.pop(0, None)
        loop._tried[0] = readiness.NOT_HEARD
    posted = on_schedule(6, None)
    while clock.now() < posted + timedelta(seconds=3 * loop.poll):
        loop._watch_round()
        clock.sleep(fetch_as_posted.WATCH_WAKE_SECONDS)
    row = [row for row in loop.document["leads"] if row["lead"] == 6][0]
    assert row["state"] == "posted"
    seen = readiness.parse_instant(row["first_seen_at"])
    assert posted - timedelta(seconds=1) <= seen <= posted + timedelta(
        seconds=loop.poll + fetch_as_posted.WATCH_WAKE_SECONDS)


def test_a_lead_already_up_at_its_first_ask_says_its_time_is_only_a_bound(
        tmp_path, monkeypatch):
    """DESIGN A136 3.5: ``lead_posted`` is the signal to re-fit a row.
    Measured live on aigfs 2026-10-01T00, fetched three hours after its
    leads posted: every lead was first asked, and dated, three hours late,
    so a re-fit from those times alone would have moved the row by the
    launch, not by the publisher.  A lead a host had already posted at its
    first ask says so (``posted_when_first_asked``), on its schedule row
    and its marker; a lead seen not posted and then posted does not."""

    monkeypatch.setattr(fetch_as_posted.PostingLoop, "watch", lambda self: None)
    # On schedule: the clock starts 5 s after f000 posted, before f003.
    loop, clock, _asked = _watched_loop(
        tmp_path / "live", lambda lead: on_schedule(lead, None))
    loop.start()
    loop.wait_start()
    posted = on_schedule(6, None)
    while clock.now() < posted + timedelta(seconds=3 * loop.poll):
        loop._watch_round()
        clock.sleep(fetch_as_posted.WATCH_WAKE_SECONDS)
    live = {row["lead"]: row for row in loop.document["leads"]}
    assert live[0]["posted_when_first_asked"] is True
    assert live[3]["posted_when_first_asked"] is False
    assert live[6]["posted_when_first_asked"] is False
    marker = json.loads(loop.publish(6, []).read_text(encoding="utf-8"))
    assert marker["posted_when_first_asked"] is False

    # Launched after the whole window posted: every lead was already up.
    late, clock, _asked = _watched_loop(tmp_path / "late", lambda lead: CYCLE)
    late.start()
    late.wait_start()
    while not late._watch_round():
        clock.sleep(fetch_as_posted.WATCH_WAKE_SECONDS)
    assert all(row["posted_when_first_asked"] is True
               for row in late.document["leads"])
    marker = json.loads(late.publish(3, []).read_text(encoding="utf-8"))
    assert marker["posted_when_first_asked"] is True

    # A window already out when the fetch began moves with no per-lead
    # ask (the fetch's whole-window path); its leads were up before any
    # ask too.  A lead let through with its host not heard says nothing.
    whole, clock, asked = _watched_loop(tmp_path / "whole", lambda lead: CYCLE)
    whole.start()
    with whole._lock:
        whole._tried[3] = readiness.NOT_HEARD
    first = json.loads(whole.publish(0, []).read_text(encoding="utf-8"))
    unheard = json.loads(whole.publish(3, []).read_text(encoding="utf-8"))
    assert asked == []
    assert first["posted_when_first_asked"] is True
    assert first["first_seen_at"] == first["fetched_at"]
    assert unheard["posted_when_first_asked"] is None
    rows_by_lead = {row["lead"]: row for row in whole.document["leads"]}
    assert rows_by_lead[0]["posted_when_first_asked"] is True
    assert "posted_when_first_asked" not in rows_by_lead[3]


def test_the_posting_watch_starts_after_the_start_needs_and_stops_on_close(
        tmp_path):
    loop, clock, asked = _watched_loop(
        tmp_path, lambda lead: None if lead == 6 else on_schedule(lead, None))
    loop.start()
    assert loop._watcher is None
    loop.wait_start()
    try:
        assert loop._watcher is not None and loop._watcher.is_alive()
    finally:
        loop.close()
    assert not loop._watcher.is_alive()
    # A watch stopped by close never starts again for that loop.
    loop.watch()
    assert not loop._watcher.is_alive()


def _donor_gated_fetch(tmp_path, monkeypatch, start_after_expected,
                       reveal_after=timedelta(minutes=3)):
    """aigfs 2026-09-29T00 whose own leads are all out; its GDAS donor posts
    ``reveal_after`` its scheduled time (None: never).  Returns (rc or the
    refusal, clock, the transfers asked for, the donor's posting time)."""

    import types

    source, cycle = "aigfs", datetime(2026, 9, 29, 0)
    donor = fetch_routes.route_for(source).donors[0]
    expected = rows.expected_at(donor.source, cycle, 0)
    clock = Clock(expected + start_after_expected)
    reveal = None if reveal_after is None else expected + reveal_after

    def out_now():
        return reveal is not None and clock.now() >= reveal

    def probe(url):
        if f"{donor.source}." in url.rsplit("/", 1)[-1]:
            return out_now()
        return True

    def published(src, cyc, lead, **_kw):
        if src == donor.source and not out_now():
            raise RuntimeError(f"{src.upper()} cycle {cyc:%Y-%m-%dT%H}Z is "
                               "not published through f000 yet")

    calls: list[dict] = []
    monkeypatch.setattr(fetch, "require_published_cycle", published)
    monkeypatch.setattr(fetch, "_fetch_route_donors", lambda plan, args: {})
    monkeypatch.setattr(fetch_routes, "run_plan",
                        lambda plan, **kw: calls.append(kw) or {})
    monkeypatch.setattr(fetch_routes, "write_handoff", lambda *a, **k: None)
    monkeypatch.setattr(fetch_routes, "handoff_lines", lambda *a, **k: [])
    monkeypatch.setattr(fetch_as_posted, "pause", clock.sleep)
    plan = fetch_routes.resolve_request(source, cycle=cycle, hours=6)
    window = readiness.Window(source=source, cycle=cycle, leads=tuple(plan.leads),
                              cadence=None, member=None, transport=None,
                              plan=plan, hours=6)
    out = tmp_path / source
    loop = fetch_as_posted.PostingLoop(window, out, probe=probe, now=clock.now,
                                       sleep=clock.sleep,
                                       progress=lambda *_: None)
    args = types.SimpleNamespace(force_refetch=False, out=out, fetch_workers=1)
    try:
        result = fetch._route_fetch_as_posted(plan, args, loop,
                                              last=max(plan.leads), options={})
    except RuntimeError as error:
        result = error
    return result, clock, calls, reveal


def test_a_donor_gated_window_waits_for_its_donor_instead_of_refusing(
        tmp_path, monkeypatch):
    """DESIGN A136 1b and 3.3: a donor-gated source starts when its donor
    posts.  Measured live 2026-10-01: aigfs 00Z, every AI lead posted, its
    readiness answer 75 (waiting on the GDAS f000 donor, due 06:48Z), and
    the fetch launched at 06:41Z refused with exit 2 instead of waiting, so
    a site launching by the readiness rule skipped the cycle."""

    result, clock, calls, reveal = _donor_gated_fetch(
        tmp_path, monkeypatch, timedelta(minutes=-2))
    assert result == 0, result
    assert clock.now() >= reveal, "the fetch did not wait for its donor"
    assert len(calls) == 1 and calls[0]["lead_gate"] is None


def test_a_donor_past_its_late_time_is_still_refused_in_words(
        tmp_path, monkeypatch):
    """Past its late time a donor that is not out will not appear by
    waiting: the whole-cycle refusal, naming the donor, stands."""

    donor = fetch_routes.route_for("aigfs").donors[0]
    cycle = datetime(2026, 9, 29, 0)
    # The donor need's own budget (its source's row), as the loop reads it.
    past = (rows.late_at(donor.source, cycle, 0)
            - rows.expected_at(donor.source, cycle, 0) + timedelta(minutes=1))
    result, clock, calls, reveal = _donor_gated_fetch(
        tmp_path, monkeypatch, past, reveal_after=None)
    assert isinstance(result, RuntimeError), result
    assert "takes part of its start from the GDAS analysis" in str(result)
    assert not calls


def test_a_start_wait_marks_the_analysis_and_first_boundary_rows_waiting(
        replay, tmp_path):
    """While the loop waits for its start needs, the rows of the needs on
    the window's own source and lead say ``waiting`` (DESIGN A136 3.5), as
    a lead's own wait does; a page drawing schedule.json otherwise showed
    the two leads the run was blocked on as merely scheduled."""

    server, clock = replay(on_schedule)
    out = tmp_path / "gefs"
    seen: dict[int, set] = {0: set(), 3: set()}
    real_write = fetch_as_posted.PostingLoop.write_schedule

    def write_schedule(self):
        for row in self.document["leads"]:
            if row["lead"] in seen:
                seen[row["lead"]].add(row["state"])
        return real_write(self)

    fetch_as_posted.PostingLoop.write_schedule = write_schedule
    try:
        assert fetch.fetch_main(fetch_args(out)) == 0
    finally:
        fetch_as_posted.PostingLoop.write_schedule = real_write
    assert "waiting" in seen[0] and "waiting" in seen[3]


def test_the_hrrr_operational_rung_says_the_as_posted_wait():
    """The rung's reason is printed by every default HRRR fetch that takes
    the operational server, so it names the as-posted wait, not a flag the
    user did not pass."""

    rung = [endpoint for endpoint in fetch_endpoints.ladder("hrrr")
            if endpoint.name == "nomads"][0]
    assert "--wait-for" not in rung.why
    assert "as-posted" in rung.why
