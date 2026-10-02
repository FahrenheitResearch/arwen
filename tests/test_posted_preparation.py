"""A preparation's wait for the leads an as-posted fetch publishes (A136 L3).

:class:`gpuwm.ingest.boundary_stream.PostedLeads` waits for the fetch
loop's per-lead markers (``posting/fNNN.json``), says the wait on the
producer heartbeat as ``waiting_for_source`` with the lead's scheduled and
late times, and ends only on the fetch's own verdict: ``failed.json``
(``source_behind`` is exit 75 naming the lead) or a fetch that ended
without the lead.  Whether a host answered 404, 403 or nothing at all is
the fetch loop's business, so no host answer can fail a preparation.

CPU only; no device, no source data.
"""

from __future__ import annotations

import json
from pathlib import Path
import threading

import pytest

from gpuwm.ingest import boundary_stream
from gpuwm.ingest.boundary_stream import (
    POSTED_LEAD_SCHEMA, BoundaryProducerFailed, BoundaryStreamStopped,
    PostedLeads, SourceBehind, posted_lead_marker_name,
)

SOURCE, CYCLE = "gfs", "2026-09-30T12"


def _marker(lead, *, source=SOURCE, cycle=CYCLE, first_seen="2026-09-30T15:40:31Z"):
    return {
        "schema": POSTED_LEAD_SCHEMA, "source": source, "member": None,
        "cycle": cycle, "lead": lead,
        "valid_time": f"2026-09-30T{12 + lead:02d}:00:00Z",
        "objects": [{"role": "gfs-subset", "name": f"gfs.f{lead:03d}",
                     "url": "u", "endpoint": "nomads", "bytes": 1,
                     "sha256": "0" * 64}],
        "expected_at": "2026-09-30T15:40:00Z",
        "late_at": "2026-09-30T16:40:00Z",
        "first_seen_at": first_seen, "fetched_at": first_seen,
    }


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _schedule(folder: Path, leads=(0, 1, 2, 3)) -> None:
    _write(folder / "schedule.json", {
        "schema": "gpuwm.posting-schedule.v1", "source": SOURCE,
        "cycle": CYCLE, "leads": [
            {"lead": lead, "valid_time": f"2026-09-30T{12 + lead:02d}:00:00Z",
             "expected_at": f"2026-09-30T15:{40 + lead:02d}:00Z",
             "late_at": f"2026-09-30T16:{40 + lead:02d}:00Z",
             "state": "scheduled"} for lead in leads]})


class _Writer:
    """The producer side PostedLeads reports to."""

    def __init__(self, stop_after=None):
        self.calls = []
        self.stop_after = stop_after
        self.checks = 0

    def waiting_for_source(self, waiting_for):
        self.calls.append(("waiting", dict(waiting_for)))

    def source_arrived(self, *, first_seen_at=None):
        self.calls.append(("arrived", first_seen_at))

    def check_stop(self):
        self.checks += 1
        if self.stop_after is not None and self.checks > self.stop_after:
            raise BoundaryStreamStopped("the forecast stopped this preparation")


def _sleeper(actions):
    """A sleep that performs the next scripted action instead of waiting."""

    script = iter(actions)

    def sleep(seconds):
        action = next(script, None)
        if action is not None:
            action()
    return sleep


def test_the_names_are_the_fetch_loops_own():
    from gpuwm import fetch_as_posted, source_posting

    assert boundary_stream.POSTING_DIRNAME == source_posting.POSTING_DIRNAME
    assert boundary_stream.POSTING_SCHEDULE_NAME \
        == source_posting.POSTING_SCHEDULE_NAME
    assert boundary_stream.POSTING_FAILED_NAME \
        == source_posting.POSTING_FAILED_NAME
    assert POSTED_LEAD_SCHEMA == fetch_as_posted.MARKER_SCHEMA
    for lead in (0, 7, 120, 384):
        assert posted_lead_marker_name(lead) == fetch_as_posted.marker_name(lead)


def test_a_posted_lead_returns_at_once_and_says_no_wait(tmp_path):
    folder = tmp_path / "posting"
    _write(folder / "f000.json", _marker(0))
    writer = _Writer()
    leads = PostedLeads(folder, source=SOURCE, cycle=CYCLE, writer=writer,
                        sleep=_sleeper([]))
    assert leads.wait(0)["lead"] == 0
    assert writer.calls == [] and leads.waits == []
    assert leads.posted([0, 1, 2]) == [0]


def test_build_k_waits_for_lead_k_and_names_it(tmp_path, capsys):
    folder = tmp_path / "posting"
    _schedule(folder)
    writer = _Writer()
    arrive = lambda: _write(folder / "f002.json", _marker(2))  # noqa: E731
    leads = PostedLeads(folder, source=SOURCE, cycle=CYCLE, writer=writer,
                        sleep=_sleeper([lambda: None, arrive]))
    marker = leads.wait(2)
    assert marker["lead"] == 2
    (kind, waiting), (after, first_seen) = writer.calls
    assert kind == "waiting" and after == "arrived"
    # The record carries the fetch's word on the lead (its schedule row's
    # state, first_seen_at and, since A136 L10, the host's last answer),
    # so a seam's reason says what the fetch knows: here the row is still
    # "scheduled", the fetch has not asked.
    assert waiting == {
        "source": SOURCE, "cycle": CYCLE, "lead": 2,
        "valid_time": "2026-09-30T14:00:00Z",
        "expected_at": "2026-09-30T15:42:00Z",
        "late_at": "2026-09-30T16:42:00Z", "since_utc": None,
        "state": "scheduled", "first_seen_at": None, "last_answer": None}
    assert first_seen == "2026-09-30T15:40:31Z"
    assert [wait["lead"] for wait in leads.waits] == [2]
    err = capsys.readouterr().err
    assert "waiting for gfs f002 of the 2026-09-30T12 cycle" in err
    assert "scheduled from about 15:42Z" in err


def _row_state(folder: Path, lead: int, state: str, **fields) -> None:
    """The fetch moving one lead's schedule row (as fetch_as_posted._mark)."""

    path = folder / "schedule.json"
    schedule = json.loads(path.read_text(encoding="utf-8"))
    for row in schedule["leads"]:
        if row["lead"] == lead:
            row.update(state=state, **fields)
    _write(path, schedule)


def test_a_lead_the_fetch_saw_posted_is_waited_on_as_downloading(
        tmp_path, capsys):
    """The wait's record follows the fetch's word on the lead, so a seam's
    reason never calls a posted lead, or one not asked for, "not posted"."""

    folder = tmp_path / "posting"
    _schedule(folder)
    _row_state(folder, 2, "waiting")
    writer = _Writer()
    seen = "2026-09-30T15:42:31Z"
    leads = PostedLeads(folder, source=SOURCE, cycle=CYCLE, writer=writer,
                        sleep=_sleeper([
                            lambda: None,
                            lambda: _row_state(folder, 2, "posted",
                                               first_seen_at=seen),
                            lambda: None,
                            lambda: _write(folder / "f002.json",
                                           _marker(2, first_seen=seen))]))
    assert leads.wait(2)["lead"] == 2
    said = [call for call in writer.calls if call[0] == "waiting"]
    assert [(record["state"], record["first_seen_at"])
            for _, record in said] == [("waiting", None), ("posted", seen)]
    assert writer.calls[-1] == ("arrived", seen)
    reasons = [boundary_stream.source_wait_reason(record)
               for _, record in said]
    assert reasons[0] == ("gfs f002 is not posted yet (scheduled from about "
                          "15:42Z; late at 16:42Z)")
    assert reasons[1] == ("gfs f002 has posted (first seen 15:42Z) and is "
                          "still downloading")
    err = capsys.readouterr().err
    assert "gfs f002 of the 2026-09-30T12 cycle, not posted yet" in err
    assert ("gfs f002 of the 2026-09-30T12 cycle posted at 15:42Z, still "
            "downloading") in err


def test_the_seam_reason_says_what_the_fetch_knows_of_the_lead(tmp_path):
    """Read by the forecast from the producer heartbeat (DESIGN A136 3.6:
    what the engine could not see is never the publisher being late)."""

    from datetime import datetime

    root = tmp_path / "prepared"
    stream = boundary_stream.stream_dir(root)
    (stream / boundary_stream.SEGMENTS_DIRNAME).mkdir(parents=True)
    head = {"schema": boundary_stream.HEAD_SCHEMA, "basis": {"cache": {
        "directory": "prepared-cache",
        "lbc": {"schedule": [[0, 3600], [3600, 7200]], "fields": ["u"],
                "spec_bdy_width": 5, "spec_zone": 1, "relax_zone": 4}}},
        "created_utc": "2026-09-30T15:00:00Z", "decision": {}}
    head["head_sha256"] = boundary_stream.head_sha256(head)
    (stream / boundary_stream.HEAD_NAME).write_text(json.dumps(head))
    intervals = boundary_stream.StreamedIntervals(
        root, head=head, start_time=datetime(2026, 9, 30, 12))

    def reason(**known):
        _write(stream / boundary_stream.PRODUCER_NAME, {
            "pid": -1, "host": "elsewhere", "updated_epoch": 0,
            "state": "waiting_for_source", "waiting_for": {
                "source": SOURCE, "cycle": CYCLE, "lead": 2,
                "valid_time": "2026-09-30T14:00:00Z",
                "expected_at": "2026-09-30T15:42:00Z",
                "late_at": "2026-09-30T16:42:00Z",
                "since_utc": "2026-09-30T15:42:00Z", **known}})
        return intervals._wait_report(1, 3.0)["reason"]

    assert reason(state="waiting") == (
        "gfs f002 is not posted yet (scheduled from about 15:42Z; late at "
        "16:42Z)")
    assert reason(state="posted", first_seen_at="2026-09-30T15:42:31Z") == (
        "gfs f002 has posted (first seen 15:42Z) and is still downloading")
    assert reason(state="scheduled") == (
        "gfs f002 is not fetched yet; the fetch has not reached it "
        "(scheduled from about 15:42Z; late at 16:42Z)")
    # The host could not be heard: said so, in either state the row is in.
    for state in ("waiting", "scheduled"):
        assert reason(state=state, last_answer="not_heard") == (
            "gfs f002 is not fetched: its host cannot be heard, so whether "
            "it is posted is not known (scheduled from about 15:42Z; late "
            "at 16:42Z)")
    assert reason(state="waiting", last_answer="not_posted") == (
        "gfs f002 is not posted yet (scheduled from about 15:42Z; late at "
        "16:42Z)")
    # A producer from before the fetch said its word: the old reason.
    assert "gfs f002 is not posted yet" in reason()


def test_a_refreshed_wait_on_the_same_lead_keeps_when_it_began(tmp_path):
    writer = boundary_stream.PreparedTreeWriter.__new__(
        boundary_stream.PreparedTreeWriter)
    writer._state, writer._waiting_for, writer._arrived = "producing", None, None
    writer._heartbeat = None
    record = {"source": SOURCE, "cycle": CYCLE, "lead": 2, "state": "waiting"}
    writer.waiting_for_source(record)
    began = writer._waiting_for["since_utc"]
    assert began
    writer.waiting_for_source({**record, "state": "posted",
                               "first_seen_at": "2026-09-30T15:42:31Z"})
    assert writer._waiting_for["since_utc"] == began
    assert writer._waiting_for["state"] == "posted"
    writer.source_arrived(first_seen_at="2026-09-30T15:42:31Z")
    writer.waiting_for_source({**record, "lead": 3})
    assert writer._waiting_for["lead"] == 3


def test_the_heartbeat_names_the_lead_while_the_producer_waits(tmp_path):
    """Against the real writer: producer.json says waiting_for_source."""

    folder = tmp_path / "posting"
    _schedule(folder)
    staging = tmp_path / ".tmp"
    staging.mkdir()
    writer = boundary_stream.PreparedTreeWriter(
        staging=staging, output_root=tmp_path / "prepared",
        identity={"source": "posted-leads-test"}, chained=True)
    beats = []

    class Beat:
        def beat(self):
            beats.append((writer._state, dict(writer._waiting_for or {})))

    writer._heartbeat = Beat()
    writer.check_stop = lambda: None
    arrive = lambda: _write(folder / "f001.json", _marker(1))  # noqa: E731
    PostedLeads(folder, source=SOURCE, cycle=CYCLE, writer=writer,
                sleep=_sleeper([arrive])).wait(1)
    assert beats[0][0] == "waiting_for_source"
    assert beats[0][1]["lead"] == 1 and beats[0][1]["since_utc"]
    assert beats[-1][0] == "producing"
    assert writer._arrived["lead"] == 1


def test_a_late_lead_ends_the_producer_with_source_behind(tmp_path):
    folder = tmp_path / "posting"
    _schedule(folder)
    late = lambda: _write(folder / "failed.json", {  # noqa: E731
        "code": "source_behind", "source": SOURCE, "member": None,
        "cycle": CYCLE, "lead": 3, "valid_time": "2026-09-30T15:00:00Z",
        "expected_at": "2026-09-30T15:43:00Z",
        "late_at": "2026-09-30T16:43:00Z", "late_after_minutes": 60.0,
        "heard": True, "last_answer": "not_posted",
        "message": "gfs f003 ... has not posted"})
    writer = _Writer()
    leads = PostedLeads(folder, source=SOURCE, cycle=CYCLE, writer=writer,
                        sleep=_sleeper([late]))
    with pytest.raises(SourceBehind) as caught:
        leads.wait(3)
    assert caught.value.exit_code == 75
    assert caught.value.details["lead"] == 3
    assert "gfs f003 of the 2026-09-30T12 cycle has not posted by 16:43Z" \
        in str(caught.value)
    # The wait is closed on the heartbeat either way.
    assert writer.calls[-1][0] == "arrived"


def test_a_host_that_cannot_be_heard_is_said_so_not_as_late(tmp_path):
    folder = tmp_path / "posting"
    _write(folder / "failed.json", {
        "code": "source_behind", "source": SOURCE, "cycle": CYCLE,
        "lead": 3, "late_at": "2026-09-30T16:43:00Z",
        "expected_at": "2026-09-30T15:43:00Z", "late_after_minutes": 60,
        "heard": False, "last_answer": "not_heard"})
    with pytest.raises(SourceBehind, match="could not be heard"):
        PostedLeads(folder, source=SOURCE, cycle=CYCLE,
                    sleep=_sleeper([])).wait(3)


def test_any_other_fetch_failure_fails_the_producer_by_name(tmp_path):
    folder = tmp_path / "posting"
    _write(folder / "failed.json", {"code": "refused",
                                    "message": "the route refused"})
    with pytest.raises(BoundaryProducerFailed, match="before f001 was fetched"
                       ) as caught:
        PostedLeads(folder, source=SOURCE, cycle=CYCLE,
                    sleep=_sleeper([])).wait(1)
    assert not isinstance(caught.value, SourceBehind)


def test_a_fetch_that_ended_without_the_lead_is_not_waited_on(tmp_path):
    folder = tmp_path / "posting"
    _schedule(folder)
    with pytest.raises(BoundaryProducerFailed,
                       match="ended without fetching f002"):
        PostedLeads(folder, source=SOURCE, cycle=CYCLE,
                    fetch_alive=lambda: False, sleep=_sleeper([])).wait(2)


def test_a_marker_that_landed_as_the_fetch_ended_is_taken(tmp_path):
    folder = tmp_path / "posting"
    alive = iter([False])

    def fetch_alive():
        # The fetch publishes its last marker and exits between two looks.
        _write(folder / "f002.json", _marker(2))
        return next(alive)

    assert PostedLeads(folder, source=SOURCE, cycle=CYCLE,
                       fetch_alive=fetch_alive,
                       sleep=_sleeper([])).wait(2)["lead"] == 2


def test_a_marker_of_another_window_is_refused(tmp_path):
    folder = tmp_path / "posting"
    _write(folder / "f001.json", _marker(1, cycle="2026-09-30T06"))
    with pytest.raises(BoundaryProducerFailed, match="2026-09-30T06 cycle"):
        PostedLeads(folder, source=SOURCE, cycle=CYCLE,
                    sleep=_sleeper([])).wait(1)


def test_a_stopped_forecast_stops_the_wait(tmp_path):
    folder = tmp_path / "posting"
    writer = _Writer(stop_after=1)
    with pytest.raises(BoundaryStreamStopped):
        PostedLeads(folder, source=SOURCE, cycle=CYCLE, writer=writer,
                    sleep=_sleeper([lambda: None, lambda: None])).wait(1)
    assert writer.calls[-1][0] == "arrived"


def test_the_wait_runs_beside_a_fetch_on_another_thread(tmp_path):
    """Real time, real threads: the markers land while the wait polls."""

    folder = tmp_path / "posting"
    _schedule(folder)
    leads = PostedLeads(folder, source=SOURCE, cycle=CYCLE,
                        poll_seconds=0.01)

    def fetch():
        for lead in (0, 1, 2, 3):
            _write(folder / f"f{lead:03d}.json", _marker(lead))

    thread = threading.Timer(0.05, fetch)
    thread.start()
    try:
        assert [leads.wait(lead)["lead"] for lead in (0, 1, 2, 3)] \
            == [0, 1, 2, 3]
    finally:
        thread.join()


# ---------------------------------------------------------------------------
# NOMADS answers 403 for a lead whose cycle directory does not exist yet
# ---------------------------------------------------------------------------


def test_nomads_names_403_as_its_missing_path_answer():
    from gpuwm import fetch_endpoints

    assert fetch_endpoints.missing_path_statuses("nomads.ncep.noaa.gov") \
        == frozenset({403})
    assert fetch_endpoints.missing_path_statuses(
        "noaa-gfs-bdp-pds.s3.amazonaws.com") == frozenset()


def test_a_403_is_absent_or_refused_only_where_the_row_says_and_asked(
        monkeypatch):
    from urllib.error import HTTPError

    from gpuwm import fetch_endpoints, nomads_governor

    def network(request, timeout=None, max_wait_s=None, opener=None):
        raise HTTPError(request.full_url, 403, "Forbidden", {}, None)

    monkeypatch.setattr(nomads_governor, "paced_urlopen", network)
    nomads = ("https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/"
              "gfs.20260930/18/atmos/gfs.t18z.pgrb2.0p25.f000")
    s3 = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20260930/18/x"
    answer = fetch_endpoints.object_answer
    assert answer(nomads, timeout=1, missing_path=True) \
        == fetch_endpoints.ABSENT_OR_REFUSED
    # Every reader that did not ask keeps "not heard": a truthy answer
    # would read as present to them.
    assert answer(nomads, timeout=1) is None
    assert answer(s3, timeout=1, missing_path=True) is None
    assert fetch_endpoints.settled_object_answer(
        nomads, backoff_s=(), missing_path=True) \
        == fetch_endpoints.ABSENT_OR_REFUSED


@pytest.mark.parametrize("answers, expected", [
    # NOMADS 403 and AWS 404: not posted anywhere a host could say.
    ([("absent_or_refused", "n", "nomads"), (False, "a", "aws")],
     "not_posted"),
    # NOMADS 403 alone (a pinned host): a blocked client says the same,
    # so it is never reported as the publisher being late.
    ([("absent_or_refused", "n", "nomads")], "not_heard"),
    ([("absent_or_refused", "n", "nomads"), (True, None, "aws")], "posted"),
    ([(None, "n", "nomads"), (False, "a", "aws")], "not_heard"),
    ([(False, "n", "nomads"), (False, "a", "aws")], "not_posted"),
])
def test_the_ladder_reads_a_missing_path_answer_beside_the_others(
        answers, expected):
    from gpuwm import source_readiness

    assert source_readiness._ladder_answer(answers)["answer"] == expected


def test_a_cycle_nomads_has_not_created_is_waited_on_not_failed(
        tmp_path, monkeypatch):
    """The fetch loop and the preparation's wait, on a 403 + 404 ladder.

    The replay's first host answers 403 for every object not posted yet
    (NOMADS before it creates the cycle directory) and the second 404.
    Each lead is asked by HEAD only until it posts (no transfer is tried
    of a lead no host holds), fetched once it does, and the preparation
    waiting on the markers beside the fetch sees every lead in order and
    no failure.
    """

    from test_fetch_as_posted import (
        CYCLE as REPLAY_CYCLE, SOURCE as REPLAY_SOURCE, Clock, Replay,
        fetch_args, on_schedule,
    )
    from http.server import BaseHTTPRequestHandler

    from gpuwm import fetch, fetch_as_posted, fetch_endpoints, fetch_routes
    from gpuwm import source_posting as rows
    from gpuwm.fetch_endpoints import Endpoint
    from datetime import timedelta

    start = rows.expected_at(REPLAY_SOURCE, REPLAY_CYCLE, 0) \
        - timedelta(minutes=10)
    clock = Clock(start)
    plan = fetch_routes.resolve_request(REPLAY_SOURCE, cycle=REPLAY_CYCLE,
                                        hours=12, cadence=3)
    reveal = lambda lead: on_schedule(lead, clock)  # noqa: E731
    forbidding = Replay(clock, plan, reveal)
    absent = Replay(clock, plan, reveal)

    refused_gets: list[tuple[str, int]] = []

    def answering(missing):
        def send_response(self, code, message=None):
            if code == 404:
                code = missing
                if self.command == "GET":
                    refused_gets.append((self.path, code))
            BaseHTTPRequestHandler.send_response(self, code, message)
        return send_response

    forbidding.server.RequestHandlerClass.send_response = answering(403)
    absent.server.RequestHandlerClass.send_response = answering(404)
    try:
        forbidden_host = forbidding.base.split("/")[2]
        rungs = (
            Endpoint(name="nomads", base=forbidding.base,
                     retention_hours=None, why="replay: 403 before posting"),
            Endpoint(name="aws", base=absent.base, retention_hours=None,
                     why="replay: 404 before posting"))
        monkeypatch.setattr(fetch_endpoints, "serving_ladder",
                            lambda source_id, **_kw: rungs)
        monkeypatch.setattr(
            fetch_endpoints, "missing_path_statuses",
            lambda host: frozenset({403}) if host == forbidden_host
            else frozenset())
        monkeypatch.setattr(fetch_endpoints, "SETTLE_BACKOFF_S", ())
        monkeypatch.setattr(fetch_as_posted, "now_utc", clock.now)
        monkeypatch.setattr(fetch_as_posted, "pause", clock.sleep)
        monkeypatch.setattr(fetch_routes, "_TRANSFER_ATTEMPTS", 1)
        monkeypatch.setattr(fetch_routes, "_retry_delay",
                            lambda error, attempt: None)
        out = tmp_path / "gefs"
        result: dict = {}
        worker = threading.Thread(
            target=lambda: result.update(code=fetch.fetch_main(
                fetch_args(out))), daemon=True)
        writer = _Writer()
        leads = PostedLeads(out / "posting", source=REPLAY_SOURCE,
                            cycle=REPLAY_CYCLE.strftime("%Y-%m-%dT%H"),
                            writer=writer, fetch_alive=worker.is_alive,
                            poll_seconds=0.01)
        worker.start()
        got = [leads.wait(lead)["lead"] for lead in (0, 3, 6, 9, 12)]
        worker.join(60)
        assert result.get("code") == 0
        assert got == [0, 3, 6, 9, 12]
        # No transfer was tried of a lead no host held: until it posted
        # it was asked by HEAD only (one round each), and every GET found
        # its object.
        assert refused_gets == []
        assert forbidding.gets or absent.gets
        assert not (out / "posting" / "failed.json").exists()
        schedule = json.loads((out / "posting" / "schedule.json").read_text())
        assert {row["state"] for row in schedule["leads"]} == {"ready"}
    finally:
        forbidding.close()
        absent.close()


# ---------------------------------------------------------------------------
# The head/seal identity split (A136 L3 design ruling, DESIGN 2.4 items 5-6)
# ---------------------------------------------------------------------------

_PLAN_PREFIX = "grib-f"
_ROUTE_TABLE = "7" * 64


def _lead_sha(lead):
    return f"{lead + 1:064x}"


def _manifest(leads=(0, 1, 2), *, names=None, sha=_lead_sha):
    names = names or {lead: f"gfs.f{lead:03d}" for lead in leads}
    return {
        "schema": "gpuwm-gfs-direct-input-manifest-v1",
        "source": {"model": "GFS", "cycle": "2026-09-30T12:00:00Z"},
        "files": {
            "bridge": {"name": "gfs_grib2_bridge", "sha256": "b" * 64},
            "series": {"name": "gfs-series.tsv", "sha256": "c" * 64},
            **{f"{_PLAN_PREFIX}{lead:03d}": {"name": names[lead],
                                              "sha256": sha(lead)}
               for lead in leads},
        },
    }


def _posted_marker(lead):
    marker = _marker(lead)
    marker["objects"][0]["sha256"] = _lead_sha(lead)
    return marker


def _as_posted_tree(tmp_path, *, leads=(0, 1, 2), seal_manifest=None,
                    seal_identity=None, markers=None, seal_route_table=None,
                    segment_markers=None, proof_manifest="sealed"):
    """A chained tree whose head is as posted, sealed as the GFS route seals.

    ``markers`` are the lead markers the seal records, ``segment_markers``
    the ones the segments were built from (the route's own, read as each
    lead is waited for); both default to the leads' posted markers.
    """

    import hashlib

    from test_boundary_stream import _initial, _met, _snapshots, _times
    from gpuwm.ingest.lateral_bc import StateBoundaryFrames

    snapshots = _snapshots(len(leads))
    times = _times(len(leads))
    plan = boundary_stream.input_plan(
        _manifest(leads), lead_role_prefix=_PLAN_PREFIX,
        route_table_sha256=_ROUTE_TABLE, derived_roles=("series",))
    plan_sha256 = boundary_stream.input_plan_sha256(plan)
    placeholder = boundary_stream.as_posted_placeholder(plan_sha256)
    identity = {"source_manifest_sha256": placeholder,
                "source_identity": {"input_manifest_sha256": placeholder,
                                    "adapter": "test"}}
    staging = tmp_path / ".tmp-posted"
    staging.mkdir()
    output = tmp_path / "posted"
    writer = boundary_stream.PreparedTreeWriter(
        staging=staging, output_root=output, identity=identity, chained=True)
    frames = StateBoundaryFrames(spec_bdy_width=5, spec_zone=1, relax_zone=4)
    frames.add_snapshot(snapshots[0], index=0)
    seconds = [(t - times[0]).total_seconds() for t in times]
    writer.write_head(
        initial_result=_initial(), met=_met(), lbc={
            "spec_bdy_width": 5, "spec_zone": 1, "relax_zone": 4,
            "schedule": [[seconds[k], seconds[k + 1]]
                         for k in range(len(times) - 1)],
            "fields": frames.inventory},
        proof_head={"schema": "test"}, forcing=frames,
        as_posted={
            "input_plan": plan,
            "start_markers": {0: _posted_marker(0)},
            "forcing_leads": list(leads),
            "seal_authored_proof_keys": ("input_manifest_sha256",
                                         "source_inputs"),
            "manifest_path": "source-input-manifest.json",
            "lead_role_prefix": _PLAN_PREFIX,
            "derived_roles": ("series",),
            "manifest_bound_identity_keys": ("source_manifest_sha256",
                                             "input_manifest_sha256"),
        })
    writer.bind_posted_leads(
        segment_markers if segment_markers is not None else
        {lead: _posted_marker(lead) for lead in leads})
    for index in range(1, len(leads)):
        frames.add_snapshot(snapshots[index], index=index)
        writer.write_segment(index - 1, frames.interval(index - 1, times))
        frames.release(index - 1)
    manifest = seal_manifest if seal_manifest is not None else _manifest(leads)
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (writer.root / "source-input-manifest.json").write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    writer.write_posted_leads(
        markers if markers is not None else
        {lead: _posted_marker(lead) for lead in leads},
        route_table_sha256=seal_route_table or _ROUTE_TABLE)
    sealed = seal_identity if seal_identity is not None else {
        "source_manifest_sha256": digest,
        "source_identity": {"input_manifest_sha256": digest,
                            "adapter": "test"}}
    receipt = writer.seal_cache(identity=sealed, manifest_sha256=digest)
    named = {"sealed": {"input_manifest_sha256": digest},
             "absent": {}}[proof_manifest]
    writer.publish({"schema": "test", **named,
                    "source_inputs": {"files": manifest["files"]},
                    "prepared_cache": {"content_sha256":
                                       receipt["content_sha256"]},
                    "posting": {"as_posted": True, "waits": [],
                                "leads_late": []},
                    "boundary_stream": writer.boundary_stream_proof()})
    return writer, output, digest


def test_an_as_posted_head_binds_the_plan_and_its_seal_writes_the_manifest(
        tmp_path):
    """The head carries the plan's placeholder where the manifest's digest
    goes and no manifest digest of its own; the seal writes the one-shot
    identity and verify_seal accepts exactly that change."""

    _writer, output, digest = _as_posted_tree(tmp_path)
    head = boundary_stream.read_head(output)
    assert head["basis"]["input_manifest_sha256"] is None
    posted = head["basis"]["as_posted"]
    placeholder = boundary_stream.as_posted_placeholder(
        posted["input_plan_sha256"])
    identity = head["basis"]["cache"]["identity"]
    assert identity["source_manifest_sha256"] == placeholder
    sealed = boundary_stream.verify_seal(output, head=head)
    assert sealed["as_posted"]["input_manifest_sha256"] == digest
    assert sorted(sealed["as_posted"]["identity_changed"]) == [
        "source_identity.input_manifest_sha256", "source_manifest_sha256"]
    header = json.loads(
        (output / "prepared-cache" / "header.json").read_text())
    assert header["identity"]["source_identity"][
        "input_manifest_sha256"] == digest
    # The head binds the start lead's marker and the route table; each
    # segment binds the markers of the two leads it spans (DESIGN 2.4
    # items 4 and 5).
    marker_sha = {lead: boundary_stream.posted_lead_marker_sha256(
        _posted_marker(lead)) for lead in (0, 1, 2)}
    assert posted["start_marker_sha256"] == {"0": marker_sha[0]}
    assert posted["input_plan"]["route_table_sha256"] == _ROUTE_TABLE
    assert posted["forcing_leads"] == [0, 1, 2]
    for k in (0, 1):
        segment = json.loads(boundary_stream.segment_marker_path(
            output, k).read_text())
        assert segment["posted_leads"] == {
            str(k): marker_sha[k], str(k + 1): marker_sha[k + 1]}


def test_the_head_refuses_a_manifest_digest_and_seal_authored_proof_keys(
        tmp_path):
    from test_boundary_stream import _initial, _met

    (tmp_path / ".tmp").mkdir()
    writer = boundary_stream.PreparedTreeWriter(
        staging=tmp_path / ".tmp", output_root=tmp_path / "out",
        identity={}, chained=True)
    posted = {"input_plan": boundary_stream.input_plan(
                  _manifest(), lead_role_prefix=_PLAN_PREFIX,
                  route_table_sha256=_ROUTE_TABLE),
              "start_markers": {}, "forcing_leads": [0, 1, 2],
              "seal_authored_proof_keys": ("source_inputs",),
              "manifest_path": "m.json", "lead_role_prefix": _PLAN_PREFIX}
    with pytest.raises(ValueError, match="never an input manifest digest"):
        writer.write_head(initial_result=_initial(), met=_met(), lbc=None,
                          proof_head={}, input_manifest_sha256="0" * 64,
                          as_posted=posted)
    with pytest.raises(ValueError, match="seal-authored keys"):
        writer.write_head(initial_result=_initial(), met=_met(), lbc=None,
                          proof_head={"source_inputs": {}}, as_posted=posted)


def test_the_seal_refuses_a_manifest_row_that_differs_from_its_lead_marker(
        tmp_path):
    """A lead row whose digest is not the one its posted marker named."""

    manifest = _manifest(sha=lambda lead: "e" * 64 if lead == 2
                         else _lead_sha(lead))
    _writer, output, _digest = _as_posted_tree(tmp_path,
                                               seal_manifest=manifest)
    head = boundary_stream.read_head(output)
    with pytest.raises(boundary_stream.BoundaryStreamError,
                       match="grib-f002 .* posted marker named"):
        boundary_stream.verify_seal(output, head=head)


def test_the_seal_refuses_a_manifest_that_is_not_the_plan(tmp_path):
    """Another object name than the plan's is another input plan."""

    manifest = _manifest(names={0: "gfs.f000", 1: "gfs.f001",
                                2: "other.f002"})
    markers = {lead: _posted_marker(lead) for lead in (0, 1, 2)}
    markers[2]["objects"][0]["name"] = "other.f002"
    _writer, output, _digest = _as_posted_tree(
        tmp_path, seal_manifest=manifest, markers=markers)
    head = boundary_stream.read_head(output)
    with pytest.raises(boundary_stream.BoundaryStreamError,
                       match="not the input plan the head bound"):
        boundary_stream.verify_seal(output, head=head)


def test_the_seal_writes_no_identity_change_but_the_manifest_digest(tmp_path):
    """Any other identity difference is refused before the header is written."""

    with pytest.raises(boundary_stream.BoundaryStreamError,
                       match="only a manifest or composition-receipt digest"):
        _as_posted_tree(tmp_path, seal_identity={
            "source_manifest_sha256": "a" * 64,
            "source_identity": {"input_manifest_sha256": "a" * 64,
                                "adapter": "another"}})


def test_the_identity_rule_names_the_keys_it_allows():
    placeholder = boundary_stream.as_posted_placeholder("p" * 64)
    head = {"bridge_manifest_sha256": placeholder, "static": "s",
            "nested": {"input_manifest_sha256": placeholder}}
    sealed = {"bridge_manifest_sha256": "a" * 64, "static": "s",
              "nested": {"input_manifest_sha256": "a" * 64}}
    assert boundary_stream.check_as_posted_identity(
        head, sealed, plan_sha256="p" * 64, manifest_sha256="a" * 64,
        manifest_bound=("bridge_manifest_sha256", "input_manifest_sha256"),
    ) == ["bridge_manifest_sha256", "nested.input_manifest_sha256"]
    with pytest.raises(boundary_stream.BoundaryStreamError,
                       match="not the sealed input manifest"):
        boundary_stream.check_as_posted_identity(
            head, sealed, plan_sha256="p" * 64, manifest_sha256="b" * 64,
            manifest_bound=("bridge_manifest_sha256",))
    with pytest.raises(boundary_stream.BoundaryStreamError,
                       match="adds or drops"):
        boundary_stream.check_as_posted_identity(
            head, {**sealed, "extra": 1}, plan_sha256="p" * 64,
            manifest_sha256="a" * 64)


def test_the_input_plan_blanks_the_lead_digests_and_the_derived_roles():
    plan = boundary_stream.input_plan(
        _manifest(), lead_role_prefix=_PLAN_PREFIX,
        route_table_sha256=_ROUTE_TABLE, derived_roles=("series",))
    assert plan["route_table_sha256"] == _ROUTE_TABLE
    files = plan["manifest"]["files"]
    assert files["grib-f001"] == {"name": "gfs.f001", "sha256": None}
    assert files["series"]["sha256"] is None
    assert files["bridge"]["sha256"] == "b" * 64
    renamed = _manifest(names={0: "gfs.f000", 1: "x.f001", 2: "gfs.f002"})
    assert boundary_stream.input_plan_sha256(boundary_stream.input_plan(
        renamed, lead_role_prefix=_PLAN_PREFIX,
        route_table_sha256=_ROUTE_TABLE, derived_roles=("series",))) \
        != boundary_stream.input_plan_sha256(plan)
    # Leads fetched under another route table are another plan.
    assert boundary_stream.input_plan_sha256(boundary_stream.input_plan(
        _manifest(), lead_role_prefix=_PLAN_PREFIX,
        route_table_sha256="8" * 64, derived_roles=("series",))) \
        != boundary_stream.input_plan_sha256(plan)
    with pytest.raises(boundary_stream.BoundaryStreamError,
                       match="route table"):
        boundary_stream.input_plan(_manifest(), lead_role_prefix=_PLAN_PREFIX,
                                   route_table_sha256=None)


# ---------------------------------------------------------------------------
# What an as-posted seal may not do (L3 check findings)
# ---------------------------------------------------------------------------


def test_a_placeholder_left_in_the_sealed_identity_is_refused():
    """A sealed identity that still carries the head's placeholder names
    inputs no preparation read, so no one-shot cache could match it."""

    placeholder = boundary_stream.as_posted_placeholder("p" * 64)
    head = {"input_manifest_sha256": placeholder,
            "source_manifest_sha256": placeholder}
    with pytest.raises(boundary_stream.BoundaryStreamError,
                       match="still carries the as-posted placeholder at "
                             "source_manifest_sha256"):
        boundary_stream.check_as_posted_identity(
            head, {**head, "input_manifest_sha256": "a" * 64},
            plan_sha256="p" * 64, manifest_sha256="a" * 64,
            manifest_bound=("source_manifest_sha256",))


def test_only_the_rulings_keys_and_declared_manifest_keys_may_change():
    """The ruling lets the input manifest and composition receipt digests
    change; another key changes only where the route declared it carries
    the manifest's digest, and only to that digest."""

    placeholder = boundary_stream.as_posted_placeholder("p" * 64)
    head = {"bridge_manifest_sha256": placeholder,
            "input_manifest_sha256": placeholder}
    sealed = {"bridge_manifest_sha256": "a" * 64,
              "input_manifest_sha256": "a" * 64}
    assert boundary_stream.AS_POSTED_IDENTITY_KEYS == {
        "input_manifest_sha256", "composition_receipt_sha256"}
    with pytest.raises(boundary_stream.BoundaryStreamError,
                       match="at bridge_manifest_sha256, which only a manifest"):
        boundary_stream.check_as_posted_identity(
            head, sealed, plan_sha256="p" * 64, manifest_sha256="a" * 64)
    # input_manifest_sha256 is the manifest's digest whatever the route says.
    with pytest.raises(boundary_stream.BoundaryStreamError,
                       match="not the sealed input manifest"):
        boundary_stream.check_as_posted_identity(
            {"input_manifest_sha256": placeholder},
            {"input_manifest_sha256": "b" * 64}, plan_sha256="p" * 64,
            manifest_sha256="a" * 64, manifest_bound=())
    # A composition receipt is its own digest.
    assert boundary_stream.check_as_posted_identity(
        {"composition_receipt_sha256": placeholder},
        {"composition_receipt_sha256": "c" * 64}, plan_sha256="p" * 64,
        manifest_sha256="a" * 64) == ["composition_receipt_sha256"]
    # The seal names the manifest it wrote, or nothing is checked against it.
    with pytest.raises(boundary_stream.BoundaryStreamError,
                       match="names the input manifest it wrote"):
        boundary_stream.check_as_posted_identity(
            head, sealed, plan_sha256="p" * 64, manifest_sha256=None)


def test_an_as_posted_proof_must_name_the_sealed_manifest(tmp_path):
    """A proof that omits input_manifest_sha256 is refused, not read as
    naming the manifest the seal wrote."""

    _writer, output, digest = _as_posted_tree(tmp_path,
                                              proof_manifest="absent")
    head = boundary_stream.read_head(output)
    with pytest.raises(boundary_stream.BoundaryStreamError,
                       match=f"names input manifest None, not the one the "
                             f"seal wrote \\({digest}\\)"):
        boundary_stream.verify_seal(output, head=head)


def test_a_segment_built_from_another_marker_is_refused(tmp_path):
    """Interval 1 was built from a lead 2 marker the seal does not record."""

    other = {lead: _posted_marker(lead) for lead in (0, 1, 2)}
    other[2]["first_seen_at"] = "2026-09-30T15:59:59Z"
    _writer, output, _digest = _as_posted_tree(tmp_path,
                                               segment_markers=other)
    head = boundary_stream.read_head(output)
    with pytest.raises(boundary_stream.BoundaryStreamError,
                       match="segment 1 was built from lead markers"):
        boundary_stream.verify_seal(output, head=head)


def test_a_start_marker_that_changed_by_the_seal_is_refused(tmp_path):
    markers = {lead: _posted_marker(lead) for lead in (0, 1, 2)}
    markers[0]["fetched_at"] = "2026-09-30T16:00:00Z"
    _writer, output, _digest = _as_posted_tree(tmp_path, markers=markers)
    head = boundary_stream.read_head(output)
    with pytest.raises(boundary_stream.BoundaryStreamError,
                       match="start lead 0's marker at the seal is not the "
                             "one the head bound"):
        boundary_stream.verify_seal(output, head=head)


def test_leads_fetched_under_another_route_table_are_another_plan(tmp_path):
    _writer, output, _digest = _as_posted_tree(tmp_path,
                                               seal_route_table="8" * 64)
    head = boundary_stream.read_head(output)
    with pytest.raises(boundary_stream.BoundaryStreamError,
                       match="route table differs"):
        boundary_stream.verify_seal(output, head=head)


def test_a_segment_is_not_written_before_its_leads_markers_are_read(tmp_path):
    """The writer refuses a segment whose lead it has no marker for."""

    with pytest.raises(RuntimeError, match="spans lead 2, whose posted "
                                           "marker this preparation has not"):
        _as_posted_tree(tmp_path, segment_markers={
            lead: _posted_marker(lead) for lead in (0, 1)})


# ---------------------------------------------------------------------------
# The GFS route as posted, through prepare_gfs_wrf (stubbed bridge)
# ---------------------------------------------------------------------------

_GFS_CYCLE = "2026-09-05T00"


def _gfs_object(lead):
    return f"gfs.t00z.pgrb2.0p25.f{lead:03d}"


class _ReplayedFetch:
    """An as-posted GFS fetch folder, published lead by lead on demand.

    Each lead is written as the fetch writes it: the object, its series
    row, the fetch manifest's row, then its posting marker last.
    """

    def __init__(self, out: Path, config, leads):
        from test_gfs_initial_perturbation import _fetched_ladder

        self.out = out
        self.posting = out / "posting"
        self.posting.mkdir(parents=True)
        self.leads = tuple(leads)
        self.published = []
        self.ladder = _fetched_ladder(config)
        self.table = "7" * 64
        self.series = out / "gfs-series.tsv"
        self.series.write_text("", encoding="utf-8")
        self.write_schedule()

    def write_schedule(self):
        _write(self.posting / "schedule.json", {
            "schema": "gpuwm.posting-schedule.v1", "source": "gfs",
            "member": None, "cycle": _GFS_CYCLE, "as_posted": True,
            "table_sha256": self.table,
            "leads": [{"lead": lead,
                       "valid_time": f"2026-09-05T{lead:02d}:00:00Z",
                       "expected_at": f"2026-09-05T03:{30 + lead:02d}:00Z",
                       "late_at": f"2026-09-05T05:{30 + lead:02d}:00Z",
                       "state": "scheduled"} for lead in self.leads]})

    def publish(self, lead, *, name=None, sha256=None):
        import hashlib

        from gpuwm import fetch

        name = name or _gfs_object(lead)
        payload = f"GRIB f{lead:03d}".encode()
        (self.out / name).write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        self.published.append((lead, name, digest))
        with self.series.open("a", encoding="utf-8") as series:
            series.write(f"{lead}\t{name}\t{81 if lead == 0 else 96}\n")
        _write(self.out / fetch.FETCH_MANIFEST_NAME, {
            "schema": fetch.FETCH_MANIFEST_SCHEMA, "source": "gfs",
            "cycle": "2026-09-05T00:00:00Z",
            "forecast_hours": [row[0] for row in self.published],
            "files": [{"role": "gfs-subset", "forecast_hour": row[0],
                       "name": row[1], "sha256": row[2]}
                      for row in self.published],
            **self.ladder})
        marker = _marker(lead, cycle=_GFS_CYCLE)
        marker["valid_time"] = f"2026-09-05T{lead:02d}:00:00Z"
        marker["objects"] = [{"role": "gfs-subset", "name": name,
                              "sha256": sha256 or digest,
                              "bytes": len(payload), "endpoint": "s3"}]
        _write(self.posting / posted_lead_marker_name(lead), marker)
        return marker


def _gfs_as_posted_route(tmp_path, monkeypatch, *, actions, domains=1,
                         tree_doubles=None):
    """prepare_gfs_wrf --as-posted over a replayed fetch.

    The real GFS door, input plan, lead waits, batch bookkeeping, seal,
    manifest author and prepared-tree writer run; the bridge, the decode
    and the array numerics are the CPU doubles of
    test_gfs_initial_perturbation, and the cache stream records its head,
    segments and identity.  ``actions`` run one per lead wait, in order.
    ``domains`` above one prepares a tree, whose hierarchy seams
    ``tree_doubles(monkeypatch, exp)`` stands in for; what it returns is
    ``seen["tree"]``.
    """

    from datetime import timedelta
    from types import SimpleNamespace

    import numpy as np

    from gpuwm import gfs_direct
    from gpuwm.experiment import load_experiment
    import gpuwm.ingest.prepared_cache as prepared_cache_module
    from test_gfs_initial_perturbation import _config, _cpu_preparation

    config = _config(tmp_path, domains=domains)
    exp = load_experiment(config)
    root = tmp_path / "fetched"
    replay = _ReplayedFetch(root, config, leads=(0, 1, 2, 3))
    replay.publish(0)
    _cpu_preparation(monkeypatch, exp)
    tree = None if tree_doubles is None else tree_doubles(monkeypatch, exp)
    monkeypatch.setattr(gfs_direct, "_canonical_surface", lambda soil: {})
    monkeypatch.setattr(boundary_stream, "_host_available",
                        lambda: 64 * 1024 ** 3)
    monkeypatch.setenv("GPUWM_CHAINED_PREP", "1")
    seen = {"bridge": [], "loads": [], "waits": [], "head_markers": None,
            "tree": tree}

    def bridge(command, **_kwargs):
        seen["bridge"].append([str(part) for part in command])
        if command[1] == "--merge-batches":
            output = Path(command[2])
        else:
            assert command[-1] == "--lead-batch"
            output = Path(command[3])
        output.mkdir()
        for name in ("gate.tsv", "inventory.tsv", "decoded-sha256.tsv"):
            (output / name).write_text(f"{command[1]} fixture\n")
        return SimpleNamespace(returncode=0, stdout="PASS fixture", stderr="")

    monkeypatch.setattr(gfs_direct.subprocess, "run", bridge)

    def load(decoded, cycle, batch, expected, **_kwargs):
        seen["loads"].append((tuple(hour for hour, _ in batch),
                              dict(expected)))
        return tuple(SimpleNamespace(
            valid_time=cycle + timedelta(hours=hour),
            fields={"SKINTEMP": np.ones((3, 3))}) for hour, _ in batch)

    monkeypatch.setattr(gfs_direct, "_load_bridge_snapshots", load)
    script = iter(actions)

    def sleep(seconds):
        seen["waits"].append(sorted(
            path.name for path in replay.posting.glob("f*.json")))
        action = next(script, None)
        if action is not None:
            action(replay)

    class ScriptedLeads(boundary_stream.PostedLeads):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **{**kwargs, "sleep": sleep})

    monkeypatch.setattr(boundary_stream, "PostedLeads", ScriptedLeads)

    class RecordingCacheStream:
        """The prepared-cache writer, recorded, with a header verify_seal reads."""

        def __init__(self, directory, *, identity, **_kwargs):
            self.directory = Path(directory)
            self.identity = identity
            self.metadata = {}

        def move(self, directory):
            self.directory = Path(directory)

        def write_head(self, **kwargs):
            seen["head_markers"] = sorted(
                path.name for path in replay.posting.glob("f*.json"))
            self.directory.mkdir(parents=True)
            self.metadata = dict(kwargs["metadata"] or {})
            return {"identity": self.identity, "metadata": self.metadata,
                    "arrays": {}, "payload_bytes": 0, "lbc": kwargs["lbc"],
                    "setup_core_fingerprint": "0" * 64}

        def write_segment(self, index, interval):
            return {"index": index,
                    "start_seconds": float(interval.start_seconds),
                    "end_seconds": float(interval.end_seconds),
                    "fields": sorted(interval.fields), "arrays": {},
                    "payload_bytes": 0, "prefix": {}}

        def seal(self, *, identity=None):
            import hashlib

            if identity is not None:
                self.identity = identity
            basis = json.loads(json.dumps({
                "schema": prepared_cache_module.PREPARED_CACHE_SCHEMA,
                "identity": self.identity, "metadata": self.metadata,
                "arrays": {}, "payload_bytes": 0}, default=str))
            content = hashlib.sha256(prepared_cache_module._canonical(
                basis).encode("utf-8")).hexdigest()
            (self.directory / "header.json").write_text(json.dumps(
                {**basis, "content_sha256": content}), encoding="utf-8")
            return {"schema": "gpuwm-prepared-cache-v1", "status": "BUILT",
                    "content_sha256": content, "array_count": 0,
                    "payload_bytes": 0}

    monkeypatch.setattr(
        prepared_cache_module, "PreparedCacheStream", RecordingCacheStream)
    wps = tmp_path / "experiment.namelist.wps"
    bridge_path = root / "gfs_grib2_bridge"
    bridge_path.write_text("bridge", encoding="utf-8")
    bridge_path.chmod(0o700)
    output_root = tmp_path / "prepared"
    manifest = tmp_path / "input-manifest.json"
    proof = gfs_direct.prepare_gfs_wrf(
        series=replay.series, cycle="2026-09-05_00:00:00",
        bridge=bridge_path, wps_namelist=wps, experiment_config=config,
        input_manifest=manifest, input_manifest_sha256=None,
        output_root=output_root, static_input=None, static_receipt=None,
        geog_root=tmp_path / "geog", preprocess_backend="cpu",
        stock_wrf_export=False, as_posted=replay.posting)
    return proof, output_root, manifest, replay, seen


def test_the_gfs_route_decodes_lead_batches_as_they_post_and_seals_one_shot(
        tmp_path, monkeypatch):
    """The head is written from f000 alone; the build of time 1 waits for
    f001 and decodes f001 and f002 together (both posted during the wait);
    f003 is waited for last; the seal merges the three batches' decoder
    receipts, writes the manifest from the fetch, binds every segment to
    its leads' markers and passes verify_seal."""

    import hashlib

    def post_one_and_two(replay):
        replay.publish(1)
        replay.publish(2)

    proof, root, manifest_path, replay, seen = _gfs_as_posted_route(
        tmp_path, monkeypatch,
        actions=[post_one_and_two, lambda replay: replay.publish(3)])

    # The head was published with only f000 posted.
    assert seen["head_markers"] == ["f000.json"]
    # Batch order: the head's lead, the two that posted during the wait
    # for f001, then f003; each batch's loader was handed the digests its
    # markers named.
    digests = {lead: sha for lead, _name, sha in replay.published}
    assert [batch for batch, _ in seen["loads"]] == [(0,), (1, 2), (3,)]
    for batch, expected in seen["loads"]:
        assert expected == {lead: digests[lead] for lead in batch}
    decodes = [call for call in seen["bridge"] if "--lead-batch" in call]
    merges = [call for call in seen["bridge"] if "--merge-batches" in call]
    assert len(decodes) == 3 and len(merges) == 1
    assert len(merges[0]) == 4 + 3  # bridge, flag, output, cycle, 3 batches
    # Two waits: for f001 (f000 alone posted), then for f003.
    assert seen["waits"] == [["f000.json"],
                             ["f000.json", "f001.json", "f002.json"]]
    assert [wait["lead"] for wait in proof["posting"]["waits"]] == [1, 3]

    # The seal wrote the input manifest from the fetch, and the proof and
    # the cache identity name it.
    manifest_bytes = manifest_path.read_bytes()
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    assert (root / "source-input-manifest.json").read_bytes() \
        == manifest_bytes
    manifest = json.loads(manifest_bytes)
    assert {role: spec["sha256"] for role, spec in manifest["files"].items()
            if role.startswith("grib-f")} == {
        f"grib-f{lead:03d}": sha for lead, sha in digests.items()}
    assert proof["input_manifest_sha256"] == digest
    header = json.loads((root / "prepared-cache" / "header.json").read_text())
    assert header["identity"]["source_identity"][
        "input_manifest_sha256"] == digest
    assert header["identity"]["bridge_manifest_sha256"] == digest
    for name in ("decoder-gate.tsv", "decoder-inventory.tsv",
                 "decoder-sha256.tsv"):
        assert (root / name).read_text() == "--merge-batches fixture\n"

    head = boundary_stream.read_head(root)
    posted = head["basis"]["as_posted"]
    assert posted["forcing_leads"] == [0, 1, 2, 3]
    assert list(posted["start_marker_sha256"]) == ["0"]
    assert posted["input_plan"]["route_table_sha256"] == replay.table
    sealed = boundary_stream.verify_seal(root, head=head)
    assert sealed["as_posted"]["input_manifest_sha256"] == digest
    assert sorted(sealed["as_posted"]["identity_changed"]) == [
        "bridge_manifest_sha256", "source_identity.input_manifest_sha256",
        "source_manifest_sha256"]


def test_the_gfs_route_refuses_a_marker_naming_another_object(
        tmp_path, monkeypatch):
    """A lead whose marker names another object than the plan's is
    refused before its decode, naming both."""

    with pytest.raises(ValueError, match="the posted object of f002 is "
                                         "other.f002, not the planned"):
        _gfs_as_posted_route(tmp_path, monkeypatch, actions=[
            lambda replay: (replay.publish(1),
                            replay.publish(2, name="other.f002"))])


def test_the_gfs_seal_refuses_a_marker_whose_digest_is_not_the_object(
        tmp_path, monkeypatch):
    """The fetched f003 is not the object its marker named: the seal's
    manifest row and the marker disagree, and nothing is sealed."""

    def post(replay):
        replay.publish(1)
        replay.publish(2)
        replay.publish(3, sha256="e" * 64)

    with pytest.raises(ValueError, match="for f003, not the object its "
                                         "posted marker named"):
        _gfs_as_posted_route(tmp_path, monkeypatch, actions=[post])
    assert not (tmp_path / "prepared" / "proof.json").exists()


def test_the_gfs_seal_refuses_leads_fetched_under_another_route_table(
        tmp_path, monkeypatch):
    def post(replay):
        replay.table = "8" * 64
        replay.write_schedule()
        for lead in (1, 2, 3):
            replay.publish(lead)

    with pytest.raises(ValueError, match="fetch's route table differs"):
        _gfs_as_posted_route(tmp_path, monkeypatch, actions=[post])


def _as_posted_tree_doubles(monkeypatch, exp):
    """The chained tree's hierarchy seams, each binding recorded."""

    from types import SimpleNamespace

    from gpuwm import gfs_direct
    from test_gfs_chained_tree import _chained_tree_doubles

    seen, tree_head = _chained_tree_doubles(monkeypatch, exp)
    seen["bindings"] = []

    def binding(**kwargs):
        seen["bindings"].append(kwargs)
        return SimpleNamespace(
            identity={"source": "gfs-as-posted-tree-test",
                      "source_manifest_sha256": kwargs[
                          "source_manifest_sha256"]},
            metadata={"source_adapter": "gfs"})

    class StartStates:
        @classmethod
        def release(cls, **kwargs):
            seen["released"].append(kwargs)
            return cls()

        def reread(self, root, **kwargs):
            seen["reread"].append((Path(root), kwargs))
            state = SimpleNamespace(lateral_boundaries="whole series")
            return (SimpleNamespace(state=state, reread=True),
                    SimpleNamespace(fields={}), "whole series",
                    ("children from the head",))

        def require_sealed_is_head(self, receipt, *, root_content_sha256,
                                   as_posted=None):
            seen["sealed_is_head"] = (receipt, root_content_sha256, as_posted)

    monkeypatch.setattr(gfs_direct, "root_domain_artifact_binding", binding)
    monkeypatch.setattr(gfs_direct, "TreeStartStates", StartStates)
    return seen


def test_a_gfs_tree_prepares_as_posted_its_children_binding_the_input_plan(
        tmp_path, monkeypatch):
    """A136 L3 (v): a GFS domain tree prepares as its leads post.

    The breakage this prevents: gfs_direct refused a GFS tree as posted,
    because the tree binds the input manifest's digest into every child,
    so every nested GFS run (a storm-following one included) waited for
    its whole window before its preparation started.  The head is written
    from f000 alone; the children and the root's head identity bind the
    input plan's placeholder where the manifest digest goes; the seal
    writes the manifest and hands its digest to the root's one-shot
    identity, the one-shot tree and the proof, and holds every sealed
    child to its head twin.
    """

    import hashlib

    from gpuwm.ingest.boundary_stream import (
        AS_POSTED_PLACEHOLDER_PREFIX, as_posted_placeholder, read_head)
    from gpuwm import gfs_direct

    def post_the_rest(replay):
        for lead in (1, 2, 3):
            replay.publish(lead)

    proof, root, manifest_path, replay, seen = _gfs_as_posted_route(
        tmp_path, monkeypatch, actions=[post_the_rest], domains=2,
        tree_doubles=_as_posted_tree_doubles)
    tree = seen["tree"]
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    # The head was published with only f000 posted.
    assert seen["head_markers"] == ["f000.json"]
    head = read_head(root)
    posted = head["basis"]["as_posted"]
    placeholder = as_posted_placeholder(posted["input_plan_sha256"])
    assert head["basis"]["input_manifest_sha256"] is None
    assert posted["seal_authored_proof_keys"] == sorted(
        gfs_direct._AS_POSTED_TREE_SEAL_KEYS)
    assert posted["manifest_bound_identity_keys"] == [
        "bridge_manifest_sha256", "input_manifest_sha256",
        "source_manifest_sha256"]
    assert list(posted["start_marker_sha256"]) == ["0"]
    for key in gfs_direct._AS_POSTED_TREE_SEAL_KEYS:
        assert key not in head["basis"]["proof_head"]
    # Children and the root's head identity bind the plan, never a digest.
    ((_written, child), ) = tree["children_written"]
    assert child["bridge_manifest_sha256"] == placeholder
    assert child["source_manifest_sha256"] == placeholder
    assert child["source_identity"]["input_manifest_sha256"] == placeholder
    assert tree["head"]["source_manifest_sha256"] == placeholder
    head_binding, sealed_binding = tree["bindings"]
    assert head_binding["source_manifest_sha256"] == placeholder
    assert head_binding["source_identity"]["input_manifest_sha256"] \
        == placeholder
    # The seal: the root's one-shot identity, the one-shot tree, the proof.
    assert sealed_binding["bridge_manifest_sha256"] == digest
    assert sealed_binding["source_manifest_sha256"] == digest
    assert sealed_binding["source_identity"]["input_manifest_sha256"] \
        == digest
    ((_root, reread),) = tree["reread"]
    assert reread["root_identity"]["source_manifest_sha256"] == digest
    _sealed_head, sealed = tree["seal"]
    assert sealed["bridge_manifest_sha256"] == digest
    assert sealed["source_manifest_sha256"] == digest
    assert sealed["source_identity"]["input_manifest_sha256"] == digest
    assert sealed["input_provenance"]["input_manifest_sha256"] == digest
    receipt, _content, as_posted = tree["sealed_is_head"]
    assert as_posted["manifest_sha256"] == digest
    assert as_posted["head"]["head_sha256"] == head["head_sha256"]
    assert proof["input_manifest_sha256"] == digest
    assert proof["posting"]["as_posted"] is True
    assert [wait["lead"] for wait in proof["posting"]["waits"]] == [1]
    assert AS_POSTED_PLACEHOLDER_PREFIX not in json.dumps(proof)
    assert (root / "source-input-manifest.json").read_bytes() \
        == manifest_path.read_bytes()
    header = json.loads((root / head["basis"]["cache"]["directory"]
                         / "header.json").read_text())
    assert header["identity"]["source_manifest_sha256"] == digest


def _child_tree(tmp_path, *, sealed_arrays=None, sealed_identity=None,
                metadata=None, receipt_digest=None, recorded=None):
    """An as-posted tree's head child and its sealed twin, as the seal leaves them.

    Each receipt records its cache's content digest, as
    ``write_native_domain_artifacts`` writes it; ``recorded`` makes the
    head receipt record another one.
    """

    import hashlib

    plan = "1" * 64
    manifest = "2" * 64
    placeholder = boundary_stream.as_posted_placeholder(plan)

    def header(identity, arrays):
        basis = {"schema": "gpuwm-prepared-cache-v1", "identity": identity,
                 "metadata": metadata or {"user": {"grid": 2}},
                 "arrays": arrays, "payload_bytes": 8}
        return {**basis, "content_sha256": hashlib.sha256(
            boundary_stream._canonical(basis).encode()).hexdigest()}

    head_identity = {"bridge_manifest_sha256": placeholder,
                     "source_manifest_sha256": placeholder,
                     "namelist_sha256": "3" * 64,
                     "source_identity": {"input_manifest_sha256": placeholder,
                                         "grid_id": 2}}
    one_shot = json.loads(json.dumps(head_identity).replace(
        placeholder, manifest))
    arrays = {"state/u": {"sha256": "4" * 64}}
    for folder, identity, table in (
            ("hierarchy-head", head_identity, arrays),
            ("hierarchy-artifacts", sealed_identity or one_shot,
             sealed_arrays or arrays)):
        child = tmp_path / folder / "domains" / "d02"
        (child / "prepared-cache").mkdir(parents=True)
        written = header(identity, table)
        _write(child / "prepared-cache" / "header.json", written)
        _write(child / "receipt.json", {
            "grid_id": 2, "folder": folder, "artifacts": {"prepared_cache": {
                "content_sha256": (recorded if recorded is not None
                                   and folder == "hierarchy-head"
                                   else written["content_sha256"])}}})
        (child / "native-static.npz").write_bytes(b"static")
        (child / "geometry-receipt.json").write_text("{}")
    receipt = (tmp_path / "hierarchy-head" / "domains" / "d02"
               / "receipt.json").read_bytes()
    head = {"basis": {
        "as_posted": {"input_plan_sha256": plan,
                      "manifest_bound_identity_keys": [
                          "bridge_manifest_sha256", "input_manifest_sha256",
                          "source_manifest_sha256"]},
        "tree": {"domains": ["d01", "d02"], "children_receipts": {
            "d02": receipt_digest or hashlib.sha256(receipt).hexdigest()}}}}
    return head, manifest, placeholder, one_shot


def test_each_sealed_child_is_held_to_the_child_its_as_posted_head_prepared(
        tmp_path):
    head, manifest, _placeholder, _one_shot = _child_tree(tmp_path)
    found = boundary_stream.verify_as_posted_tree_children(
        tmp_path, head=head, manifest_sha256=manifest)
    assert sorted(found["d02"]["identity_changed"]) == [
        "bridge_manifest_sha256", "source_identity.input_manifest_sha256",
        "source_manifest_sha256"]
    assert found["d02"]["head_content_sha256"] \
        != found["d02"]["sealed_content_sha256"]


def test_a_sealed_child_whose_arrays_differ_from_its_head_twin_is_refused(
        tmp_path):
    head, manifest, *_ = _child_tree(
        tmp_path, sealed_arrays={"state/u": {"sha256": "5" * 64}})
    with pytest.raises(boundary_stream.BoundaryStreamError,
                       match=r"sealed d02 differs .* \['arrays'\]"):
        boundary_stream.verify_as_posted_tree_children(
            tmp_path, head=head, manifest_sha256=manifest)


def test_a_sealed_child_identity_may_change_only_where_the_manifest_goes(
        tmp_path):
    head, manifest, _placeholder, one_shot = _child_tree(tmp_path)
    other_key = json.loads(json.dumps(one_shot))
    other_key["namelist_sha256"] = "6" * 64
    head, manifest, *_ = _child_tree(tmp_path / "other",
                                     sealed_identity=other_key)
    with pytest.raises(boundary_stream.BoundaryStreamError,
                       match="namelist_sha256"):
        boundary_stream.verify_as_posted_tree_children(
            tmp_path / "other", head=head, manifest_sha256=manifest)
    wrong = json.loads(json.dumps(one_shot).replace(manifest, "7" * 64))
    head, manifest, *_ = _child_tree(tmp_path / "wrong",
                                     sealed_identity=wrong)
    with pytest.raises(boundary_stream.BoundaryStreamError,
                       match="not the sealed input manifest"):
        boundary_stream.verify_as_posted_tree_children(
            tmp_path / "wrong", head=head, manifest_sha256=manifest)


def test_a_sealed_child_that_still_names_the_plan_is_refused(tmp_path):
    placeholder = boundary_stream.as_posted_placeholder("1" * 64)
    head, manifest, *_ = _child_tree(
        tmp_path, metadata={"user": {"provenance": placeholder}})
    with pytest.raises(boundary_stream.BoundaryStreamError,
                       match="still carries the as-posted placeholder at "
                             "metadata.user.provenance"):
        boundary_stream.verify_as_posted_tree_children(
            tmp_path, head=head, manifest_sha256=manifest)


def test_a_head_child_whose_receipt_the_head_does_not_bind_is_refused(
        tmp_path):
    head, manifest, *_ = _child_tree(tmp_path, receipt_digest="8" * 64)
    with pytest.raises(boundary_stream.BoundaryStreamError,
                       match="d02 receipt is not the one the head binds"):
        boundary_stream.verify_as_posted_tree_children(
            tmp_path, head=head, manifest_sha256=manifest)


def test_a_head_child_cache_its_receipt_does_not_record_is_refused(tmp_path):
    """The head digest binds the receipt and the receipt the cache.

    The head content digest this check returns is what a sealed binding's
    restart identity names for the child, so a head header that is not the
    one the bound receipt records (or that fails its own digest) is
    refused rather than handed on.
    """

    head, manifest, *_ = _child_tree(tmp_path, recorded="9" * 64)
    with pytest.raises(boundary_stream.BoundaryStreamError,
                       match="head's d02 prepared cache is not the one its "
                             "receipt records"):
        boundary_stream.verify_as_posted_tree_children(
            tmp_path, head=head, manifest_sha256=manifest)
    head, manifest, *_ = _child_tree(tmp_path / "edited")
    path = (tmp_path / "edited" / "hierarchy-head" / "domains" / "d02"
            / "prepared-cache" / "header.json")
    edited = json.loads(path.read_text())
    edited["metadata"] = {"user": {"grid": 3}}
    _write(path, edited)
    with pytest.raises(boundary_stream.BoundaryStreamError,
                       match="head's d02 prepared cache is not the one its "
                             "receipt records"):
        boundary_stream.verify_as_posted_tree_children(
            tmp_path / "edited", head=head, manifest_sha256=manifest)


def _sealed_as_posted_tree(tmp_path):
    """``_child_tree`` with its head published and the seal's proof."""

    from types import SimpleNamespace

    head, manifest, *_ = _child_tree(tmp_path)
    head = {"schema": boundary_stream.HEAD_SCHEMA, "basis": head["basis"]}
    head["head_sha256"] = boundary_stream.head_sha256(head)
    _write(boundary_stream.stream_dir(tmp_path) / boundary_stream.HEAD_NAME,
           head)
    proof = {"boundary_stream": {"head_sha256": head["head_sha256"]},
             "posting": {"as_posted": True, "waits": [], "leads_late": []},
             "input_manifest_sha256": manifest}

    def digest(folder):
        return json.loads(
            (tmp_path / folder / "domains" / "d02" / "prepared-cache"
             / "header.json").read_text())["content_sha256"]

    def bundles(child_digest):
        return [SimpleNamespace(grid_id=1, parent_id=0, cache_reader=None),
                SimpleNamespace(grid_id=2, parent_id=1,
                                cache_reader=SimpleNamespace(
                                    content_sha256=child_digest))]

    return proof, digest("hierarchy-head"), digest("hierarchy-artifacts"), \
        bundles


def test_a_sealed_as_posted_tree_names_the_children_its_head_prepared(
        tmp_path):
    """The repair: a checkpoint from a GFS tree resumes on its seal.

    An as-posted GFS tree's forecast binds its head, so its checkpoints'
    restart identity names the head's children, which carry the input
    plan where their sealed twins carry the manifest digest.  ``gpuwm go
    --prepared-root P --restart CKPT`` binds the seal; it must name the
    same children, each sealed one held to its head twin, or it refuses
    every checkpoint the run wrote ("written for a different run").
    """

    from gpuwm.prepared_domain_tree_forecast import _as_posted_head_children

    proof, head_digest, sealed_digest, bundles = _sealed_as_posted_tree(
        tmp_path)
    assert head_digest != sealed_digest
    assert dict(_as_posted_head_children(
        tmp_path, proof, bundles(sealed_digest))) == {"d02": head_digest}
    # A tree that was not prepared as posted, or not chained, is untouched.
    for other in ({**proof, "posting": None},
                  {key: value for key, value in proof.items()
                   if key != "boundary_stream"}):
        assert _as_posted_head_children(
            tmp_path, other, bundles(sealed_digest)) is None
    # A loaded child that is not the sealed twin the check read.
    with pytest.raises(ValueError, match="sealed d02 this run restores is "
                                         "not the one held"):
        _as_posted_head_children(tmp_path, proof, bundles(head_digest))
    # A proof naming another head, or a manifest the seal did not write.
    for bad, match in (
            ({**proof, "boundary_stream": {"head_sha256": "a" * 64}},
             "not the pinned head"),
            ({**proof, "input_manifest_sha256": "b" * 64},
             "not the sealed input manifest")):
        with pytest.raises(ValueError, match=match):
            _as_posted_head_children(tmp_path, bad, bundles(sealed_digest))


def test_a_sealed_as_posted_tree_without_its_head_is_refused_by_name(
        tmp_path):
    from gpuwm.prepared_domain_tree_forecast import _as_posted_head_children

    proof, _head_digest, sealed_digest, bundles = _sealed_as_posted_tree(
        tmp_path)
    (boundary_stream.stream_dir(tmp_path) / boundary_stream.HEAD_NAME).unlink()
    with pytest.raises(ValueError, match="cannot be held to the head its "
                                         "proof names"):
        _as_posted_head_children(tmp_path, proof, bundles(sealed_digest))


def test_a_domain_binding_takes_the_plan_placeholder_and_nothing_else():
    from gpuwm import native_domain_artifacts as artifacts

    placeholder = boundary_stream.as_posted_placeholder("a" * 64)
    assert artifacts._manifest_digest(placeholder, "x") == placeholder
    assert artifacts._manifest_digest("B" * 64, "x") == "b" * 64
    for value in ("as-posted:" + "a" * 63, "as-posted:" + "g" * 64,
                  "plan:" + "a" * 64):
        with pytest.raises(ValueError, match="SHA-256"):
            artifacts._manifest_digest(value, "x")
    with pytest.raises(ValueError, match="SHA-256"):
        artifacts._digest(placeholder, "namelist_sha256")


def test_the_tree_runner_takes_its_as_posted_heads_placeholder_and_nothing_else():
    from gpuwm.prepared_domain_tree_forecast import _bound_authority

    plan = "c" * 64
    placeholder = boundary_stream.as_posted_placeholder(plan)
    head = {"basis": {"as_posted": {
        "input_plan_sha256": plan,
        "manifest_bound_identity_keys": [
            "bridge_manifest_sha256", "input_manifest_sha256",
            "source_manifest_sha256"]}}}
    authority = {"bridge_manifest_sha256": placeholder,
                 "source_manifest_sha256": placeholder,
                 "namelist_sha256": "D" * 64}
    assert _bound_authority(authority, head) == {
        **authority, "namelist_sha256": "d" * 64}
    # Another plan's placeholder, a placeholder where no manifest goes, and
    # any placeholder on a head that is not as posted are refused.
    for bad, on in (
            ({**authority, "source_manifest_sha256":
              boundary_stream.as_posted_placeholder("e" * 64)}, head),
            ({**authority, "namelist_sha256": placeholder}, head),
            (authority, {"basis": {}}),
            (authority, None)):
        with pytest.raises(ValueError, match="SHA-256"):
            _bound_authority(bad, on)
