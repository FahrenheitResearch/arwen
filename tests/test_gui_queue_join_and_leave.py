"""A forecast joins the queue, leaves it, or is refused, and the answer says what happened.

The defects this file holds shut:

- Remove answered "Removed from the queue" while a folder it could not
  delete kept its queued marker, so the scheduler's walk of the forecasts
  folder (or a restart of the page server) put the forecast back in line
  and it started.  A marker Remove could not delete was reported as a
  success the same way.
- A queued forecast the scheduler started before the reply was written was
  answered with HTTP 500 ("'t1' is not in list"): the reply read its place
  back from the line after the start had taken it out.
- A disk that filled while a new forecast's files were written, before
  its start or its place in line, answered with the page server's generic
  HTTP 500 and left the folder behind, so the same name was refused as a
  run that already exists once there was room again.  The event page's
  button did the same.  A line that could not be saved after the marker
  was written left the marker for the scheduler's walk to start.
"""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import pytest

from gpuwm.gui import api as api_module
from gpuwm.gui import queue as queue_module
from gpuwm.gui import runs
from gpuwm.gui.server import build_server, serve_in_thread
from test_gui_server import FakeRunner, request
from test_gui_wiki import _event_with, era5_gui  # noqa: F401 - era5_gui is a fixture

DRAFT = {"source": "gfs", "cycle": "2026-09-24T00", "lat": 35.5, "lon": -97.0, "width_km": 300,
         "height_km": 300, "hours": 1, "card": "16gb", "dx_km": 12}
BUSY = {"job_id": "job-busy", "started_utc": "2026-09-27T09:30:00+00:00", "kind": "gui:run-plan", "folder": None}


class CardRunner(FakeRunner):
    """The stand-in engine, with a card the test says is busy or free."""

    def __init__(self) -> None:
        super().__init__()
        self.holder = None

    def card_holder(self):
        return self.holder


@pytest.fixture()
def page(tmp_path, monkeypatch):
    monkeypatch.setattr("gpuwm.source_availability.quick_head", lambda url: True)
    monkeypatch.setattr("gpuwm.fetch._head_ok", lambda url: True)
    runner = CardRunner()
    server = build_server(tmp_path / "runs", port=0, runner=runner, token="t" * 43)
    server.api.queue._cards = lambda: {"devices": [{"name": "Test card", "memory_total_mib": 16 * 1024,
                                                    "memory_used_mib": 512}], "processes": []}
    server.api.queue._disk = lambda: 200.0
    serve_in_thread(server)
    yield server, runner
    server.shutdown()
    server.server_close()


def start(server, name, *, queue=False):
    return request(server, "POST", "/api/create/start", body={**DRAFT, "name": name, "queue": queue, "need_gib": 5.3})


def queue_one(server, name):
    response, body = start(server, name, queue=True)
    assert response.status == 200 and body["queued"], body
    return body


def restarted_order(server):
    again = build_server(server.root, port=0, runner=CardRunner(), token="u" * 43)
    try:
        return again.api.queue.order(scan=True)
    finally:
        again.server_close()


# ---------------------------------------------------------------- Remove


def test_a_forecast_removed_whose_folder_cannot_be_deleted_never_comes_back_or_starts(page, monkeypatch):
    server, runner = page
    runner.holder = BUSY
    queue_one(server, "stuck")
    folder = server.root / "stuck"
    real = queue_module.shutil.rmtree

    def locked(path, ignore_errors=False, **kwargs):
        # A file in the folder another program holds open: nothing deletes, and a strict delete says why.
        if Path(path) == folder:
            if ignore_errors:
                return None
            raise PermissionError(errno.EACCES, "Permission denied", str(folder / runs.PLAN))
        return real(path, ignore_errors=ignore_errors, **kwargs)

    monkeypatch.setattr(queue_module.shutil, "rmtree", locked)
    response, body = request(server, "POST", "/api/queue/stuck/remove")
    assert response.status == 200, body
    # Out of the line, and the folder left behind is said in words, with no machine path.
    assert body["kept"] is True and "could not be deleted" in body["message"] and "Permission denied" in body["message"]
    assert str(server.root) not in body["message"]
    assert folder.is_dir() and not (folder / runs.QUEUED).exists()
    # The scheduler's walk of the folder, a free card and a restarted page server all leave it out of line.
    assert server.api.queue.order(scan=True) == []
    runner.holder = None
    assert server.api.queue.tick() == [] and runner.launched == []
    assert restarted_order(server) == []
    assert runs.status(folder)["state"] != "queued"


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                    reason="needs a folder this user can reach files in by name but cannot list")
def test_a_folder_that_cannot_be_emptied_leaves_the_line_all_the_same(page):
    """The case as it happens: the delete could not list the folder, so it removed nothing, the marker too."""

    server, runner = page
    runner.holder = BUSY
    queue_one(server, "unlisted")
    folder = server.root / "unlisted"
    folder.chmod(0o300)
    try:
        response, body = request(server, "POST", "/api/queue/unlisted/remove")
        assert response.status == 200 and body["kept"] is True, body
        assert not (folder / runs.QUEUED).exists()
        runner.holder = None
        assert server.api.queue.order(scan=True) == [] and server.api.queue.tick() == [] and runner.launched == []
        assert restarted_order(server) == []
    finally:
        folder.chmod(0o700)


def test_a_marker_remove_cannot_delete_is_said_and_the_forecast_keeps_its_place(page, monkeypatch):
    server, runner = page
    runner.holder = BUSY
    queue_one(server, "first")
    queue_one(server, "second")
    marker = server.root / "first" / runs.QUEUED
    real = Path.unlink

    def refused(path, *args, **kwargs):
        if Path(path) == marker:
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refused)
    response, body = request(server, "POST", "/api/queue/first/remove")
    assert response.status == 500, body
    assert not body["ok"] and "still queued" in body["message"] and "Remove" in body["fix"]
    assert server.api.queue.order(scan=True) == ["first", "second"]
    assert marker.is_file() and (server.root / "first" / runs.PLAN).is_file()


def test_remove_deletes_the_folder_and_says_so_when_nothing_holds_it(page):
    server, runner = page
    runner.holder = BUSY
    queue_one(server, "gone")
    response, body = request(server, "POST", "/api/queue/gone/remove")
    assert response.status == 200 and "kept" not in body
    assert body["message"] == "Removed gone from the queue."
    assert not (server.root / "gone").exists() and server.api.queue.order(scan=True) == []


# ---------------------------------------------------------------- the reply to Queue it


def test_a_queued_forecast_the_scheduler_starts_before_the_reply_is_answered_as_queued(page, monkeypatch):
    server, runner = page
    queue = server.api.queue
    real = queue.note_data

    def scheduler_first(run_id):
        # The scheduler thread, woken as the forecast joins the line, starts it before the request goes on.
        assert queue.tick() == [run_id]
        real(run_id)

    monkeypatch.setattr(queue, "note_data", scheduler_first)
    monkeypatch.setattr(server.api, "availability_of", lambda *args, **kwargs: {"starts": "now"})
    response, body = start(server, "t1", queue=True)
    assert response.status == 200, body
    assert body["ok"] and body["queued"] and body["place"] == 1
    assert len(runner.launched) == 1 and (server.root / "t1" / runs.JOB).is_file()
    assert queue.order() == []
    # Its commands.log says it was queued before it says it started.
    log = (server.root / "t1" / runs.COMMANDS_LOG).read_text(encoding="utf-8")
    assert log.index("# queued:") < log.index("# started from the queue")


def test_the_place_given_is_the_one_my_forecasts_shows(page):
    server, runner = page
    runner.holder = BUSY
    assert [queue_one(server, name)["place"] for name in ("a", "b", "c")] == [1, 2, 3]
    _, listing = request(server, "GET", "/api/queue")
    assert [item["run"] for item in listing["items"]] == ["a", "b", "c"]


# ---------------------------------------------------------------- a start that cannot write its files


def _failing(original, folder_name, leaf, code):
    def write(path, document, *args, **kwargs):
        if Path(path).parent.name == folder_name and Path(path).name == leaf:
            raise OSError(code, os.strerror(code), str(path))
        return original(path, document, *args, **kwargs)
    return write


@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.parametrize("leaf", ["region.geojson", runs.PLAN])
def test_a_full_disk_while_the_plan_is_written_starts_nothing_and_leaves_the_name_free(page, monkeypatch, leaf,
                                                                                      queued):
    server, runner = page
    with monkeypatch.context() as patch:
        patch.setattr(api_module, "write_json", _failing(api_module.write_json, "space", leaf, errno.ENOSPC))
        response, body = start(server, "space", queue=queued)
    assert response.status == 507, body
    assert "disk" in body["message"] and "full" in body["message"] and "same name" in body["fix"]
    assert str(server.root) not in body["message"] + body["fix"]
    assert runner.launched == [] and server.api.queue.order(scan=True) == []
    assert not (server.root / "space").exists()
    # Once there is room the same name starts, or queues.
    response, body = start(server, "space", queue=queued)
    assert response.status == 200, body
    assert bool(body.get("queued")) is queued


def test_a_folder_that_cannot_be_written_is_said_in_words_and_released(page, monkeypatch):
    server, runner = page
    with monkeypatch.context() as patch:
        patch.setattr(api_module, "write_json", _failing(api_module.write_json, "locked", runs.PLAN, errno.EACCES))
        response, body = start(server, "locked")
    assert response.status == 500, body
    assert "could not be saved" in body["message"] and "Permission denied" in body["message"]
    assert "Nothing started" in body["message"] and not (server.root / "locked").exists()
    assert runner.launched == []
    response, body = start(server, "locked")
    assert response.status == 200 and len(runner.launched) == 1


def test_a_full_disk_while_the_forecast_joins_the_line_leaves_no_marker_to_start_later(page, monkeypatch):
    server, runner = page
    runner.holder = BUSY
    with monkeypatch.context() as patch:
        patch.setattr(queue_module, "write_json",
                      _failing(queue_module.write_json, "marker", runs.QUEUED, errno.ENOSPC))
        response, body = start(server, "marker", queue=True)
    assert response.status == 507, body
    assert not (server.root / "marker").exists() and server.api.queue.order(scan=True) == []
    response, body = start(server, "marker", queue=True)
    assert response.status == 200 and body["place"] == 1


def test_a_line_that_cannot_be_saved_takes_the_marker_back_before_the_scheduler_can_see_it(page, monkeypatch):
    server, runner = page
    runner.holder = BUSY
    queue = server.api.queue
    real = queue._save
    seen = []

    def full(document):
        if "line" in document["order"]:
            # The scheduler's walk at this moment, under the lock the queue holds, must not find a marker either.
            seen.append((server.root / "line" / runs.QUEUED).is_file())
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), str(queue._file()))
        return real(document)

    monkeypatch.setattr(queue, "_save", full)
    response, body = start(server, "line", queue=True)
    assert response.status == 507, body
    assert seen == [True]
    assert not (server.root / "line").exists()
    monkeypatch.setattr(queue, "_save", real)
    runner.holder = None
    assert queue.order(scan=True) == [] and queue.tick() == [] and runner.launched == []


def test_a_queue_log_line_that_cannot_be_written_queues_nothing(page, monkeypatch):
    server, runner = page
    runner.holder = BUSY

    def no_room(rundir, line, note=""):
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), str(Path(rundir) / runs.COMMANDS_LOG))

    with monkeypatch.context() as patch:
        patch.setattr(api_module, "log_command", no_room)
        response, body = start(server, "logless", queue=True)
    assert response.status == 507, body
    assert not (server.root / "logless").exists() and server.api.queue.order(scan=True) == []


def test_the_event_page_button_releases_its_folder_when_the_disk_fills(era5_gui, monkeypatch):  # noqa: F811
    monkeypatch.setattr(api_module, "disk_free_gib", lambda path: 500.0)
    event = _event_with("era5")
    body = {"event": event["id"], "card_gb": 16}
    _, dry = request(era5_gui, "POST", "/api/wiki/simulate", body={**body, "dry_run": True})
    name = dry["run"]
    with monkeypatch.context() as patch:
        patch.setattr(api_module, "write_json", _failing(api_module.write_json, name, runs.PLAN, errno.ENOSPC))
        response, refused = request(era5_gui, "POST", "/api/wiki/simulate", body=body)
    assert response.status == 507, refused
    assert not (era5_gui.api.root / name).exists()
    # The next press takes the same name, not a "-2" copy beside a folder nobody asked for.
    response, started = request(era5_gui, "POST", "/api/wiki/simulate", body=body)
    assert response.status == 200 and started["run"] == name, started
    assert json.loads((era5_gui.api.root / name / runs.PLAN).read_text(encoding="utf-8"))["name"] == name
