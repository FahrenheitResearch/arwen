"""A forecast run as its source posts: the doors that carry the choice and say the waits.

The breakages each test holds shut:

- A door that passed an as-posted flag or run option the engine does not
  define would have every run it launched refused at exit 2 ("not a run
  option this build understands"), so the choice is read from the
  engine's own parsers and route declarations (``door_takes``), and a
  plan carries ``as_posted = false`` only where its route declares it.
- New forecast drew no schedule and no opt-out, and a run that waited at
  a seam showed only its model hour: the waiting run's status now carries
  what it waits on, the lead and its expected and late times, and where
  the model stands (lead, expected time and model time in the API).
- An MCP caller asking for ``as_posted: true`` on an engine without the
  as-posted fetch would get a run that starts only once the cycle's last
  hour is posted, hours later than asked, with nothing said: it is refused
  with that reason.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest

from gpuwm.gui import posting, runs
from gpuwm.gui.jobs import Refused, Runner
from gpuwm.mcp import doors, engine_tools
from gpuwm.runplan import ROUTES
from test_gui_server import DRAFT, gui, make_run, request  # noqa: F401 - gui is a fixture

WHOLE_CYCLE_ENGINE = {"fetch_as_posted": False, "fetch_whole_cycle": False, "go_whole_cycle": False,
                      "run_plan_readiness": False, "fetch_readiness": False, "run_plan_as_posted": []}
AS_POSTED_ENGINE = {"fetch_as_posted": True, "fetch_whole_cycle": True, "go_whole_cycle": True,
                    "run_plan_readiness": True, "fetch_readiness": True,
                    "run_plan_as_posted": ["experiment", "prepared"]}

READINESS = {
    "schema": "gpuwm.readiness.v1", "checked_at": "2026-09-30T15:42:40Z", "source": "gfs", "member": None,
    "cycle": "2026-09-24T00", "cycle_basis": "named",
    "window": {"start_lead": 0, "hours": 1, "cadence": 1, "final_lead": 1}, "as_posted": True,
    "posting": {"shape": "rolling", "streams": True, "why": "one file per lead", "late_after_minutes": 60,
                "poll_seconds": 30, "rule": {}, "table_sha256": "x"},
    "state": "waiting", "ready": False,
    "start_needs": [{"role": "analysis", "source": "gfs", "lead": 0, "expected_at": "2026-09-24T03:32:00Z",
                     "late_at": "2026-09-24T04:32:00Z", "answer": "posted", "endpoint": "nomads"},
                    {"role": "first_boundary", "source": "gfs", "lead": 1, "expected_at": "2026-09-24T03:33:00Z",
                     "late_at": "2026-09-24T04:33:00Z", "answer": "not_posted", "endpoint": None}],
    "expected_ready_at": "2026-09-24T03:33:00Z", "expected_final_at": "2026-09-24T03:33:00Z",
    "leads": [], "retry_after_seconds": 30, "refusal": None,
}


# ------------------------------------------------------------------ the engine's own report

def test_door_flags_reads_the_engines_own_parsers():
    assert "--wait-for" in doors.door_flags("gpuwm.fetch", "fetch")
    assert "--dry-run" in doors.door_flags("gpuwm.go_cli", "go")
    assert "--resolve" in doors.door_flags("gpuwm.runplan", "run-plan")


def test_a_door_that_gains_the_flag_is_read_as_taking_it(monkeypatch):
    module = types.ModuleType("fake_posting_door")

    def register_cli(subparsers):
        parser = subparsers.add_parser("fetch")
        parser.add_argument("--readiness", action="store_true")

    module.register_cli = register_cli
    monkeypatch.setitem(sys.modules, "fake_posting_door", module)
    assert doors.door_takes("fake_posting_door", "fetch", "--readiness")
    assert not doors.door_takes("fake_posting_door", "fetch", "--whole-cycle")


def test_the_support_report_agrees_with_the_parsers_and_routes():
    support = doors.as_posted_support()
    assert support["run_plan_readiness"] == ("--readiness" in doors.door_flags("gpuwm.runplan", "run-plan"))
    assert support["go_whole_cycle"] == ("--whole-cycle" in doors.door_flags("gpuwm.go_cli", "go"))
    assert support["run_plan_as_posted"] == sorted(
        name for name, route in ROUTES.items() if "as_posted" in route.run_options)


# ------------------------------------------------------------------ New forecast: the plan and the schedule

def test_a_plan_carries_the_opt_out_only_where_its_route_declares_it(monkeypatch):
    monkeypatch.setattr(posting, "support", lambda: AS_POSTED_ENGINE)
    assert posting.plan_options("prepared", True) == {}
    assert posting.plan_options("prepared", False) == {"as_posted": False}
    assert posting.plan_options("config", False) == {}
    monkeypatch.setattr(posting, "support", lambda: WHOLE_CYCLE_ENGINE)
    assert posting.plan_options("prepared", False) == {}


def test_the_opt_out_reaches_the_plan_on_this_engine_exactly_when_run_plan_takes_it(gui):
    """Holds before and after the as-posted fetch lands: read from run-plan's own route declaration."""

    server, _ = gui
    response, body = request(server, "POST", "/api/create/start",
                             body={**DRAFT, "whole_cycle": True, "dry_run": True})
    assert response.status == 200
    declared = "as_posted" in ROUTES[body["plan"]["route"]].run_options
    options = body["plan"].get("run_options") or {}
    assert ("as_posted" in options) == declared
    if declared:
        assert options["as_posted"] is False
    response, body = request(server, "POST", "/api/create/start", body={**DRAFT, "dry_run": True})
    assert "as_posted" not in (body["plan"].get("run_options") or {})


def test_the_opt_out_is_a_true_or_false(gui):
    server, _ = gui
    response, body = request(server, "POST", "/api/create/start",
                             body={**DRAFT, "whole_cycle": "yes", "dry_run": True})
    assert response.status == 400 and "whole_cycle" in body["message"]


def test_an_engine_that_starts_on_the_whole_cycle_is_said_and_asked_nothing(gui, monkeypatch):
    server, runner = gui
    monkeypatch.setattr(posting, "support", lambda: WHOLE_CYCLE_ENGINE)
    asked = len(runner.queries)
    response, body = request(server, "POST", "/api/create/posting", body=DRAFT)
    assert response.status == 200
    assert body["posting"] == {"available": False, "asked_as_posted": True, "as_posted": False,
                               "why": "whole_cycle_engine"}
    assert not any("--readiness" in argv for argv in runner.queries[asked:])


def test_the_schedule_is_the_engines_readiness_answer_for_the_drafts_plan(gui, monkeypatch):
    server, runner = gui
    monkeypatch.setattr(posting, "support", lambda: AS_POSTED_ENGINE)
    asked: list[tuple[list[str], Path | None, tuple[int, ...]]] = []

    def answer(argv, *, cwd=None, codes=(0,), timeout=0):
        # The plan the question is asked of is the one Start would write.
        plan = json.loads((Path(cwd) / "plan.json").read_text(encoding="utf-8"))
        assert plan["config"]["intent"]["source"] == "gfs"
        asked.append((list(argv), cwd, codes))
        return 75, READINESS

    monkeypatch.setattr(runner, "answer", answer, raising=False)
    response, body = request(server, "POST", "/api/create/posting", body=DRAFT)
    assert response.status == 200
    argv, cwd, codes = asked[0]
    assert argv[-3:-1] == ["run-plan", str(Path(cwd) / "plan.json")] and argv[-1] == "--readiness"
    assert codes == (0, 75, 2)
    # The draft's plan folder goes once the question is answered.
    assert not Path(cwd).exists()
    shown = body["posting"]
    assert shown["available"] and shown["as_posted"] and shown["exit_code"] == 75
    assert shown["state"] == "waiting" and shown["ready"] is False
    assert shown["expected_ready_at"] == "2026-09-24T03:33:00Z"
    assert shown["posting"]["late_after_minutes"] == 60 and shown["posting"]["streams"] is True
    assert [need["role"] for need in shown["start_needs"]] == ["analysis", "first_boundary"]
    assert shown["start_needs"][1]["answer"] == "not_posted"


def test_a_schedule_that_cannot_be_read_never_stops_the_page(gui, monkeypatch):
    server, runner = gui
    monkeypatch.setattr(posting, "support", lambda: AS_POSTED_ENGINE)

    def answer(argv, *, cwd=None, codes=(0,), timeout=0):
        raise Refused("the host was not heard")

    monkeypatch.setattr(runner, "answer", answer, raising=False)
    response, body = request(server, "POST", "/api/create/posting", body={**DRAFT, "whole_cycle": True})
    assert response.status == 200
    assert body["posting"]["error"] == "the host was not heard"
    assert body["posting"]["asked_as_posted"] is False


def test_a_foreign_document_is_not_drawn_as_a_schedule():
    said = posting.facts({"schema": "something-else"}, as_posted=True)
    assert said["available"] and "gpuwm.readiness.v1" in said["error"]


def test_the_runner_hands_back_a_document_at_each_of_its_codes(tmp_path):
    script = tmp_path / "answer.py"
    script.write_text("import json, sys\nprint(json.dumps({'schema': 'x', 'code': int(sys.argv[1])}))\n"
                      "sys.exit(int(sys.argv[1]))\n", encoding="utf-8")
    runner = Runner()
    assert runner.answer([sys.executable, str(script), "75"], codes=(0, 75, 2)) == (75, {"schema": "x", "code": 75})
    assert runner.answer([sys.executable, str(script), "0"], codes=(0, 75, 2))[0] == 0
    with pytest.raises(Refused):
        runner.answer([sys.executable, str(script), "3"], codes=(0, 75, 2))


# ------------------------------------------------------------------ a waiting run, as the API exports it

def _heartbeat(run: Path, status: str, wait: dict, elapsed: float = 1800.0) -> None:
    (run / "run-progress.json").write_text(json.dumps({
        "status": status, "model_elapsed_seconds": elapsed, "wait": wait}), encoding="utf-8")


def _append(run: Path, *records: dict) -> None:
    with (run / "events.jsonl").open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")


def test_a_preparation_wait_says_its_interval_and_where_the_model_stands(tmp_path):
    run = make_run(tmp_path, "prep-wait", pid=os.getpid())
    _append(run, {"event": "boundary_wait_started", "interval": 2, "reason": "not prepared yet",
                  "cause": "preparation", "model_elapsed_seconds": 7200.0,
                  "model_valid_time": "2026-09-24T02:00:00Z"})
    _heartbeat(run, "waiting:preparation", {"on": "preparation", "lead": None, "expected_at": None,
                                            "late_at": None, "since_utc": "2026-09-24T03:00:00Z"}, 7200.0)
    info = runs.status(run)
    assert info["phase"] == "waiting:preparation"
    wait = info["wait"]
    assert wait["on"] == "preparation" and wait["interval"] == 2
    assert wait["model_elapsed_seconds"] == 7200.0
    assert wait["model_valid_time"] == "2026-09-24T02:00:00Z"
    assert wait["waited_seconds"] > 0


def test_a_source_wait_says_the_lead_its_times_and_the_model_time(tmp_path):
    run = make_run(tmp_path, "source-wait", pid=os.getpid())
    _append(run, {"event": "posting_schedule", "source": "gefs", "cycle": "2026-09-30T12", "as_posted": True,
                  "shape": "rolling", "streams": True, "late_after_minutes": 60,
                  "expected_ready_at": "2026-09-30T15:43:04Z", "expected_final_at": "2026-09-30T15:58:59Z",
                  "leads": [{"lead": 0}, {"lead": 3}, {"lead": 6}]},
            {"event": "lead_ready", "lead": 0}, {"event": "lead_ready", "lead": 3},
            {"event": "source_wait_started", "phase": "seam", "source": "gefs", "cycle": "2026-09-30T12",
             "lead": 30, "valid_time": "2026-10-01T18:00:00Z", "expected_at": "2026-09-30T15:53:00Z",
             "late_at": "2026-09-30T16:53:00Z", "waited_seconds": 0, "model_elapsed_seconds": 97200.0,
             "model_valid_time": "2026-10-01T15:00:00Z", "interval": 9, "reason": "not posted yet"})
    _heartbeat(run, "waiting:source", {"on": "source", "lead": 30, "expected_at": "2026-09-30T15:53:00Z",
                                       "late_at": "2026-09-30T16:53:00Z", "since_utc": "2026-09-30T15:53:10Z"})
    info = runs.status(run)
    wait = info["wait"]
    assert (wait["on"], wait["lead"], wait["source"]) == ("source", 30, "gefs")
    assert wait["expected_at"] == "2026-09-30T15:53:00Z" and wait["late_at"] == "2026-09-30T16:53:00Z"
    assert wait["model_elapsed_seconds"] == 97200.0 and wait["model_valid_time"] == "2026-10-01T15:00:00Z"
    assert wait["interval"] == 9 and wait["valid_time"] == "2026-10-01T18:00:00Z"
    assert info["posting"]["leads"] == 3 and info["posting"]["leads_ready"] == 2
    assert info["posting"]["expected_final_at"] == "2026-09-30T15:58:59Z"


def test_a_wait_the_events_do_not_describe_takes_the_heartbeats_model_time(tmp_path):
    """A wait event of another lead is not this wait's; the model time comes from the heartbeat and the start."""

    run = make_run(tmp_path, "bare-wait", pid=os.getpid())
    _append(run, {"event": "source_wait_started", "lead": 3, "source": "gefs", "model_elapsed_seconds": 1.0})
    _heartbeat(run, "waiting:source", {"on": "source", "lead": 6, "expected_at": "2026-09-24T03:40:00Z",
                                       "late_at": "2026-09-24T04:40:00Z", "since_utc": "2026-09-24T03:41:00Z"},
               5400.0)
    wait = runs.status(run)["wait"]
    assert wait["lead"] == 6 and wait.get("source") is None
    assert wait["model_elapsed_seconds"] == 5400.0
    # make_run's plan starts at 2026-09-24T00:00:00: the model stands 1 h 30 min in.
    assert wait["model_valid_time"].startswith("2026-09-24") and "01:30" in wait["model_valid_time"]


def test_a_finished_wait_leaves_no_block(tmp_path):
    run = make_run(tmp_path, "waited", pid=os.getpid())
    _append(run, {"event": "boundary_wait_started", "interval": 1, "model_elapsed_seconds": 3600.0},
            {"event": "boundary_wait_finished", "interval": 1, "seconds": 4.0})
    (run / "run-progress.json").write_text(json.dumps({"status": "integrating", "model_elapsed_seconds": 3700.0}),
                                           encoding="utf-8")
    assert runs.status(run)["wait"] is None


# ------------------------------------------------------------------ the MCP run tools

class _Server:
    def __init__(self):
        self.tools = {}

    def tool(self, name):
        def register(fn):
            self.tools[name] = fn
            return fn
        return register


class _Manager:
    def __init__(self):
        self.launched = []

    def launch(self, kind, argv, **_):
        self.launched.append((kind, list(argv)))
        return {"job_id": "job-x", "kind": kind}


@pytest.fixture()
def tools():
    server, manager = _Server(), _Manager()
    engine_tools.register(server, manager)
    return server.tools, manager


def test_mcp_as_posted_on_an_engine_that_runs_as_posted(tools, monkeypatch):
    registered, manager = tools
    monkeypatch.setattr(engine_tools, "as_posted_support", lambda: AS_POSTED_ENGINE)
    reply = registered["arwen_fetch"](source="gfs", out_dir="o", as_posted=True)
    assert reply["as_posted"] is True and "--as-posted" in manager.launched[-1][1]
    reply = registered["arwen_fetch"](source="gfs", out_dir="o", as_posted=False)
    assert reply["as_posted"] is False and "--whole-cycle" in manager.launched[-1][1]
    reply = registered["arwen_fetch"](source="gfs", out_dir="o")
    assert reply["as_posted"] is True
    assert not {"--as-posted", "--whole-cycle"} & set(manager.launched[-1][1])
    reply = registered["arwen_forecast"](config="c.toml", as_posted=False)
    assert reply["as_posted"] is False and "--whole-cycle" in manager.launched[-1][1]
    reply = registered["arwen_forecast"](config="c.toml", as_posted=True)
    assert reply["as_posted"] is True and "--whole-cycle" not in manager.launched[-1][1]


def test_mcp_as_posted_on_an_engine_that_starts_on_the_whole_cycle(tools, monkeypatch):
    registered, manager = tools
    monkeypatch.setattr(engine_tools, "as_posted_support", lambda: WHOLE_CYCLE_ENGINE)
    with pytest.raises(doors.ArwenRefusal, match="last hour is posted"):
        registered["arwen_fetch"](source="gfs", out_dir="o", as_posted=True)
    with pytest.raises(doors.ArwenRefusal, match="last hour is posted"):
        registered["arwen_forecast"](config="c.toml", as_posted=True)
    assert manager.launched == []
    # The opt-out is what this engine does anyway: nothing is passed, and the reply says the whole cycle.
    reply = registered["arwen_forecast"](config="c.toml", as_posted=False)
    assert reply["as_posted"] is False and "--whole-cycle" not in manager.launched[-1][1]


def test_mcp_as_posted_has_no_meaning_for_a_run_that_fetches_nothing(tools):
    registered, manager = tools
    with pytest.raises(doors.ArwenRefusal, match="fetches nothing"):
        registered["arwen_forecast"](config="c.toml", mode="run", as_posted=False)
    assert manager.launched == []
