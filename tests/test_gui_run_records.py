"""A run folder's records are read value by value: one of the wrong kind is left unknown, never a 500.

The defects this file holds shut, each on the page's own server:

- A copied plan holding ``NaN``, ``Infinity`` or ``1e400`` made the run's
  details answer 500 (or reply with JSON no browser reads).
- A ``commands.log`` another program wrote in another encoding made the
  run's details answer 500.
- A heartbeat or event whose figure was not a number, or an elapsed time
  past any date, made the run's status answer 500 or invented progress.
- A forecast start hour that was not a whole number, or was huge, made the
  status answer 500 or showed the cycle as the forecast's start.
- A grid row of the wrong kind dropped every grid of the run.
- A resumed run read Finished while it ran: it kept the end and progress
  of the attempt before it.
- A plan whose values have the wrong kind made the run's article answer 500.
- On Windows ``CON``, ``nul.txt`` and ``run:stream`` were taken as run
  names at lookup; they name a device or a stream there, never a folder.
"""

from __future__ import annotations

import json
import os

import pytest

from gpuwm.gui import runs
from gpuwm.gui.files import PathRefused, folder_part, split_run_id
from test_gui_server import events_file, gui, make_run, request  # noqa: F401 - gui is a fixture


def _plan(folder, intent):
    (folder / runs.PLAN).write_text(json.dumps({"config": {"intent": intent}}), encoding="utf-8")


@pytest.mark.parametrize("config", ["copied configuration", ["incomplete"], 7, {"intent": "text"}])
def test_a_plan_shaped_otherwise_lists_with_a_warning_and_the_others_load(gui, config):  # noqa: F811
    server, _ = gui
    make_run(server.root, "healthy", plan=True)
    broken = make_run(server.root, "copied", plan=True)
    (broken / runs.PLAN).write_text(json.dumps({"config": config}), encoding="utf-8")
    response, body = request(server, "GET", "/api/runs")
    assert response.status == 200
    row = next(row for row in body["runs"] if row["id"] == "copied")
    assert row["status"]["state"] != "unreadable" and row["status"]["run_seconds"] is None
    assert row["status"]["metadata_warnings"]
    assert {row["id"] for row in body["runs"]} == {"healthy", "copied"}
    # The folder is read, never rewritten.
    assert json.loads((broken / runs.PLAN).read_text(encoding="utf-8")) == {"config": config}


@pytest.mark.parametrize("raw", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_a_plan_with_a_number_json_cannot_hold_keeps_the_details_available(gui, raw):  # noqa: F811
    server, _ = gui
    folder = make_run(server.root, "copied", plan=True)
    text = '{"config":{"intent":{"hours":' + raw + ',"source":"gfs"}}}'
    (folder / runs.PLAN).write_text(text, encoding="utf-8")
    response, body = request(server, "GET", "/api/runs/copied")
    assert response.status == 200, body
    assert body["plan"]["config"]["intent"] == {"hours": None, "source": "gfs"}
    assert body["status"]["run_seconds"] is None
    assert (folder / runs.PLAN).read_text(encoding="utf-8") == text


def test_a_commands_log_in_another_encoding_keeps_the_details_available(gui):  # noqa: F811
    server, _ = gui
    folder = make_run(server.root, "copied", plan=True)
    (folder / runs.COMMANDS_LOG).write_bytes(b"gpuwm run-plan old-plan.json\n# copied note \xff\n")
    response, body = request(server, "GET", "/api/runs/copied")
    assert response.status == 200
    assert "gpuwm run-plan old-plan.json" in body["commands_log"]
    assert (folder / runs.COMMANDS_LOG).read_bytes().endswith(b"\xff\n")


@pytest.mark.parametrize("lead", ["unknown", {}, 1e100, 2.5, -3])
def test_a_start_hour_that_is_not_one_leaves_the_start_unknown(gui, lead):  # noqa: F811
    server, _ = gui
    folder = make_run(server.root, "copied", plan=True)
    _plan(folder, {"hours": 2, "cycle": "2026-09-24T00:00:00Z", "forecast_start_hour": lead})
    response, body = request(server, "GET", "/api/runs/copied/status")
    assert response.status == 200
    assert body["start_time"] is None and body["run_seconds"] == 7200
    assert body["metadata_warnings"]


@pytest.mark.parametrize("elapsed", ["unknown", {}, None, -1])
def test_a_heartbeat_figure_that_is_not_a_time_keeps_the_events_progress(gui, elapsed):  # noqa: F811
    server, _ = gui
    folder = make_run(server.root, "copied", plan=True)
    _plan(folder, {"hours": 2, "cycle": "2026-09-24T00:00:00Z"})
    events_file(folder / runs.EVENTS, [{"event": "model_progress", "model_seconds": 1800, "speed_x": 30}])
    (folder / runs.HEARTBEAT).write_text(json.dumps({"model_elapsed_seconds": elapsed}), encoding="utf-8")
    response, body = request(server, "GET", "/api/runs/copied/status")
    assert response.status == 200
    assert body["model_seconds"] == 1800 and body["percent"] == 25
    assert body["storm_time"] == "2026-09-24 00:30 UTC"


@pytest.mark.parametrize("measurement", ["pending", {}, -1, [1]])
def test_event_figures_that_are_not_numbers_stay_unknown(gui, measurement):  # noqa: F811
    server, _ = gui
    folder = make_run(server.root, "copied", plan=True)
    _plan(folder, {"hours": 2})
    events_file(folder / runs.EVENTS, [
        {"event": "resolved_plan", "configuration": {"experiment": {
            "run_seconds": measurement, "start_time": "2026-09-24T00:00:00Z"}}},
        {"event": "model_progress", "model_seconds": measurement, "speed_x": measurement},
    ])
    response, body = request(server, "GET", "/api/runs/copied/status")
    assert response.status == 200
    assert body["model_seconds"] is None and body["percent"] is None and body["speed_x"] is None
    assert body["run_seconds"] == 7200  # the plan's own length still reads
    assert body["metadata_warnings"]


def test_an_elapsed_time_past_any_date_is_left_unknown(gui):  # noqa: F811
    server, _ = gui
    folder = make_run(server.root, "copied", plan=True)
    _plan(folder, {"hours": 2, "cycle": "2026-09-24T00:00:00Z"})
    events_file(folder / runs.EVENTS, [{"event": "model_progress", "model_seconds": 1e100}])
    response, body = request(server, "GET", "/api/runs/copied/status")
    assert response.status == 200
    assert body["storm_time"] == "2026-09-24 00:00 UTC" and body["model_seconds"] is None
    assert body["metadata_warnings"]


@pytest.mark.parametrize("configuration", ["partial", {"experiment": "partial"}, {"experiment": {"domains": 7}}])
def test_a_resolved_plan_of_the_wrong_shape_keeps_the_rest_of_the_run(gui, configuration):  # noqa: F811
    server, _ = gui
    folder = make_run(server.root, "copied", plan=True)
    events_file(folder / runs.EVENTS, [{"event": "resolved_plan", "configuration": configuration},
                                       {"event": "completed"}])
    response, body = request(server, "GET", "/api/runs")
    assert response.status == 200
    assert body["runs"][0]["status"]["state"] == "finished"
    assert body["runs"][0]["status"]["metadata_warnings"]


def test_a_grid_row_of_the_wrong_kind_keeps_the_other_grids(gui):  # noqa: F811
    server, _ = gui
    folder = make_run(server.root, "copied", plan=True)
    events_file(folder / runs.EVENTS, [
        {"event": "resolved_plan", "configuration": {"experiment": {"domains": [
            {"grid_id": "unknown", "run": "incomplete"},
            {"grid_id": 2, "run": {"dx": "nan", "nz": "unknown"}},
            {"grid_id": 3, "run": {"dx": 12000, "nz": 49}},
        ]}}},
        {"event": "model_progress", "domains": [{"domain": "unknown", "step_wall_seconds": 2},
                                                {"domain": 3, "step_wall_seconds": 3}]},
    ])
    response, body = request(server, "GET", "/api/runs")
    assert response.status == 200
    row = body["runs"][0]
    assert row["status"]["grids"]["grids_km"] == [12]
    assert row["card"]["levels"] == [49]


@pytest.mark.parametrize("intent", [{"root_dx_km": "unavailable"}, {"root_dx_km": 1e400},
                                    {"nz": "unavailable"}, {"nz": -4}])
def test_card_facts_of_the_wrong_kind_are_left_out(gui, intent):  # noqa: F811
    server, _ = gui
    folder = make_run(server.root, "copied", plan=True)
    (folder / runs.PLAN).write_text(json.dumps({"config": {"intent": intent}}).replace("Infinity", "1e400"),
                                    encoding="utf-8")
    response, body = request(server, "GET", "/api/runs")
    assert response.status == 200
    assert body["runs"][0]["card"]["dx_km"] == [] and body["runs"][0]["card"]["levels"] == []


def test_a_resumed_run_reads_its_own_attempt_not_the_one_before(gui):  # noqa: F811
    server, _ = gui
    folder = make_run(server.root, "resumed", pid=os.getpid(), plan=True)
    events_file(folder / runs.EVENTS, [
        {"event": "plan_accepted", "run_id": "old-attempt"},
        {"event": "resolved_plan", "configuration": {"experiment": {"run_seconds": 7200}}},
        {"event": "model_progress", "model_seconds": 7200},
        {"event": "completed"},
        {"event": "plan_accepted", "run_id": "new-attempt"},
        {"event": "resolved_plan", "configuration": {"experiment": {"run_seconds": 7200}}},
        {"event": "model_progress", "model_seconds": 1800},
    ])
    response, body = request(server, "GET", "/api/runs/resumed/status")
    assert response.status == 200
    assert body["state"] == "running" and body["end"] is None and body["percent"] == 25


@pytest.mark.parametrize("intent", [{"source": ["gfs"], "hours": "six"}, {"cycle": {"t": 1}, "root_dx_km": "x"},
                                    {"nz": [49], "physics_profile": 7}])
def test_a_runs_article_reads_a_plan_of_the_wrong_kind(gui, intent):  # noqa: F811
    server, _ = gui
    folder = make_run(server.root, "copied", plan=True)
    _plan(folder, intent)
    response, body = request(server, "GET", "/api/wiki/run/copied")
    assert response.status == 200, body


@pytest.mark.parametrize("name", ["CON", "nul.txt", "Com1.log", "run:stream", "run<x", "tab\tname"])
def test_windows_device_and_stream_names_are_never_run_names_there(name):
    assert not folder_part(name, windows=True)
    # Elsewhere they are ordinary folder names a terminal can write, and they open like any other.
    assert folder_part(name, windows=False)


@pytest.mark.parametrize("name", ["../outside", "archive/../outside", "run\\outside", "/absolute", "run\x00name",
                                  ".hidden", "a/b/c"])
def test_lookup_still_refuses_every_way_out_of_the_root(name):
    with pytest.raises(PathRefused):
        split_run_id(name)


@pytest.mark.skipif(os.name == "nt", reason="the name is a stream on Windows, refused above")
def test_a_folder_with_a_colon_lists_and_opens_where_the_file_system_allows_it(gui):  # noqa: F811
    server, _ = gui
    make_run(server.root, "2026-09-24T12:00 run", plan=True)
    response, body = request(server, "GET", "/api/runs")
    assert [row["id"] for row in body["runs"]] == ["2026-09-24T12:00 run"]
    response, _ = request(server, "GET", "/api/runs/" + body["runs"][0]["url"] + "/status")
    assert response.status == 200
