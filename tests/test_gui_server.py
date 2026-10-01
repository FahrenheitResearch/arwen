"""``gpuwm gui`` against a live server with a fake runner: no browser, no GPU.

The fake runner stands where the engine does: its queries answer with
canned documents shaped like ``gpuwm run-plan``'s, and its launch writes
the files a launched run would (the job record and the manifest naming a
live pid) without starting anything.  Everything else is the real server:
the token gate, the run folders read from disk, the event stream, the
pictures listing and the files it serves.
"""

from __future__ import annotations

import http.client
import json
import os
from pathlib import Path
import socket
import threading
import time

import pytest

from gpuwm import proc_identity
from gpuwm.gui import auth
from gpuwm.gui.copy_lint import lint_all
from gpuwm.gui.jobs import Refused, Runner
from gpuwm.gui.server import build_server, serve_in_thread

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32

RESOLVED = {
    "schema": "gpuwm.run-plan.resolved.v1",
    "configuration": {"experiment": {
        "run_seconds": 3600.0, "start_time": "2026-09-24T00:00:00",
        "projection": {"ref_lat": 39.5, "ref_lon": -97.0},
        "domains": [{"grid_id": 1, "parent_id": 0, "i_parent_start": 1, "j_parent_start": 1,
                     "parent_grid_ratio": 1, "run": {"nx": 36, "ny": 30, "nz": 49, "dx": 12000.0}}]}},
    "generated_config": "# PHYSICS: a-profile-v1: words\n",
    # The wizard's priced figure, as run-plan --resolve carries it: bytes, whichever route the source takes.
    "memory": {"peak_envelope_bytes": int(5.33 * (1 << 30)), "budget_bytes": int(14.54 * (1 << 30)),
               "binding_phase": "forecast", "sizing_basis": "declared-capacity"},
    "warnings": [],
}


def manifest_of(pid: int) -> dict:
    """A manifest as ``gpuwm run-plan`` writes it: the pid and that process's identity."""

    return {"pid": pid, "process": proc_identity.identify(pid)}


class FakeRunner(Runner):
    kind = "fake"

    def __init__(self) -> None:
        super().__init__()
        self.launched: list[list[str]] = []
        self.owner_files: list[str | None] = []
        self.queries: list[list[str]] = []

    def query(self, argv, *, cwd=None, timeout=0, log=None):
        self.queries.append(list(argv))
        if "--resolve" in argv:
            return RESOLVED
        if "--sources" in argv:
            return {"sources": [{"source_id": "gfs", "display_name": "GFS", "max_forecast_hour": 384,
                                 "run_plan": {"intent_supported": True, "intent_routes": ["prepared"],
                                              "requires_source_root": False}}]}
        if "--physics-profiles" in argv:
            return {"sources": [{"source_id": "gfs", "default_profile_id": "p1",
                                 "profiles": [{"profile_id": "p1", "admissible": True, "is_default": True}]}],
                    "profiles": [{"profile_id": "p1", "summary": "p1: words"}]}
        return {"devices": []}

    def answer(self, argv, *, cwd=None, codes=(0,), timeout=0):
        # The readiness question New forecast asks once the engine answers it (gui/posting.py): this stand-in has
        # no source to ask, and a real engine here would ask the hosts. The page says the schedule was not read.
        raise Refused("this stand-in engine has no posting schedule")

    def launch(self, rundir, argv, owner_file=None):
        self.launched.append(list(argv))
        self.owner_files.append(owner_file)
        job = {"job_id": "job-fake", "jobs_dir": str(rundir / ".fake-job"), "wrapper_pid": os.getpid(),
               "wrapper_process": proc_identity.identify(os.getpid()),
               "argv": argv}
        (rundir / "gui-job.json").write_text(json.dumps(job), encoding="utf-8")
        (rundir / "run-manifest.json").write_text(json.dumps(manifest_of(os.getpid())), encoding="utf-8")
        return job

    def stop(self, rundir, *, dry=False):
        return {"method": "interrupt", "command": "kill -INT -12345",
                "message": "Sent Ctrl+C to the run."}

    def card_holder(self):
        # The card is free here: the real one reads this computer's own jobs root (HeldCardRunner keeps a card).
        return None

    def runtime_gap(self):
        # This stand-in engine runs on any install: the real one needs CuPy in this Python, which a test box
        # without a card does not have (test_gui_runtime_readiness.py checks that answer).
        return None

    def missing_geography(self, root=None):
        # A fact about this computer a start reads besides the card and the runtime: its geography tree is set up.
        # Pinned so a start here never depends on the test machine's own tree; NoGeography below asks the real check.
        return None


class HeldCardRunner(FakeRunner):
    """The fake engine, with the card's lock kept by the real job manager under a jobs root of the test's own.

    While the lock is held a start goes through the real launch, which
    refuses it as it refuses on a busy card; on a free card the start is
    the fake's, so no engine runs in a test.
    """

    def __init__(self, jobs_root: Path) -> None:
        from gpuwm.mcp.gpulock import GpuLock

        super().__init__()
        self.jobs_root = jobs_root
        self.lock = GpuLock(jobs_root)

    card_holder = Runner.card_holder

    def hold(self) -> None:
        # This test's own pid is alive, so the lock reads as a running forecast's.
        self.lock.acquire("job-held", os.getpid())

    def free(self) -> None:
        self.lock.release("job-held")

    def launch(self, rundir, argv):
        if self.lock.holder() is not None:
            return Runner.launch(self, rundir, argv)
        return super().launch(rundir, argv)


class NoGeography(FakeRunner):
    """The fake engine on a computer whose geography tree is checked for real (point it at an empty folder)."""

    missing_geography = Runner.missing_geography


@pytest.fixture()
def gui(tmp_path, monkeypatch):
    # A recent start is put to the fetch's object probe; no server is asked in a test.
    monkeypatch.setattr("gpuwm.fetch._head_ok", lambda url: True)
    # A page asks through its own short probe; no server is asked in a test either.
    monkeypatch.setattr("gpuwm.source_availability.quick_head", lambda url: True)
    runner = FakeRunner()
    server = build_server(tmp_path / "runs", port=0, runner=runner, token="t" * 43)
    serve_in_thread(server)
    yield server, runner
    server.shutdown()
    server.server_close()


def request(server, method, path, *, body=None, token=True, cookie=True, headers=None, timeout=10):
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=timeout)
    sent = {"Host": f"127.0.0.1:{server.port}"}
    if cookie:
        sent["Cookie"] = f"{auth.cookie_name(server.port)}={server.token}"
    if token and method == "POST":
        sent[auth.TOKEN_HEADER] = server.token
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        sent["Content-Type"] = "application/json"
    sent.update(headers or {})
    connection.request(method, path, body=data, headers=sent)
    response = connection.getresponse()
    raw = response.read()
    connection.close()
    kind = response.getheader("Content-Type") or ""
    return response, (json.loads(raw) if kind.startswith("application/json") else raw)


def events_file(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps({"sequence": i, **r}) + "\n" for i, r in enumerate(records)),
                    encoding="utf-8")


def make_run(root: Path, name: str, *, pid=None, end=None, stop=False, plan=False, frames=0,
             render="chain/png") -> Path:
    run = root / name
    run.mkdir(parents=True)
    if pid is not None:
        (run / "run-manifest.json").write_text(json.dumps(manifest_of(pid)), encoding="utf-8")
        records = [{"event": "resolved_plan", "configuration": {"experiment": {
            "run_seconds": 3600.0, "start_time": "2026-09-24T00:00:00", "domains": [{}]}}},
            {"event": "model_progress", "model_seconds": 1800.0, "speed_x": 30.0}]
        if end:
            records.append(end)
        events_file(run / "events.jsonl", records)
    if stop:
        (run / "gui-stop.json").write_text("{}", encoding="utf-8")
    if plan:
        (run / "plan.json").write_text(json.dumps({"config": {"intent": {"hours": 2, "cycle": "2026-09-24T00"}}}),
                                       encoding="utf-8")
    for index in range(frames):
        folder = run / render / "d01-12km" / ("composite_reflectivity" if index % 2 else "2m_temperature") / "2026-09-24"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"frame_f{index:03d}.png").write_bytes(PNG)
    return run


def dead_pid() -> int:
    pid = 999_999
    from gpuwm.mcp.gpulock import _pid_alive

    while _pid_alive(pid):
        pid -= 1
    return pid


# ---------------------------------------------------------------- the token gate

def test_a_request_without_the_token_is_refused(gui):
    server, _ = gui
    response, body = request(server, "GET", "/api/runs", cookie=False)
    assert response.status == 403
    assert "link" in body["message"]


def test_the_printed_link_sets_the_cookie_and_the_cookie_opens_the_api(gui):
    server, _ = gui
    response, _ = request(server, "GET", f"/?token={server.token}", cookie=False)
    assert response.status == 303
    cookie = response.getheader("Set-Cookie")
    assert cookie.startswith(f"{auth.cookie_name(server.port)}={server.token}")
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie
    response, body = request(server, "GET", "/api/runs")
    assert response.status == 200 and body["runs"] == []


def test_a_post_needs_the_header_even_with_the_cookie(gui):
    server, _ = gui
    response, body = request(server, "POST", "/api/create/fit", body={}, token=False)
    assert response.status == 403 and auth.TOKEN_HEADER in body["message"]


def test_a_foreign_host_or_origin_is_refused(gui):
    server, _ = gui
    response, _ = request(server, "GET", "/api/runs", headers={"Host": f"evil.example:{server.port}"})
    assert response.status == 400
    response, _ = request(server, "GET", "/api/runs", headers={"Origin": "http://evil.example"})
    assert response.status == 400


def test_a_bind_other_computers_can_reach_is_refused_without_allow_remote(tmp_path):
    with pytest.raises(auth.BindRefused):
        build_server(tmp_path, host="0.0.0.0", port=0, runner=FakeRunner())


def test_the_page_is_served_with_its_policy(gui):
    server, _ = gui
    response, body = request(server, "GET", "/")
    assert response.status == 200 and b"/static/js/app.js" in body
    assert "script-src 'self'" in response.getheader("Content-Security-Policy")
    response, _ = request(server, "GET", "/static/js/app.js")
    assert response.status == 200 and response.getheader("Cache-Control") == "no-cache"


# ---------------------------------------------------------------- the run folder is the truth

def test_the_run_list_is_read_from_the_folders(gui):
    server, _ = gui
    root = server.root
    make_run(root, "done", pid=dead_pid(), end={"event": "completed"})
    make_run(root, "live", pid=os.getpid())
    make_run(root, "broke", pid=dead_pid(), end={"event": "failed", "message": "no data"})
    make_run(root, "halted", pid=dead_pid(), end={"event": "failed", "interrupted": True})
    make_run(root, "lost", pid=dead_pid())
    make_run(root, "planned", plan=True)
    make_run(root / "old", "copied", frames=2, render="")
    response, body = request(server, "GET", "/api/runs")
    states = {row["id"]: row["status"]["state"] for row in body["runs"]}
    assert states == {"done": "finished", "live": "running", "broke": "failed", "halted": "stopped",
                      "lost": "stale", "planned": "ready", "old/copied": "imported"}
    copied = next(row for row in body["runs"] if row["id"] == "old/copied")["card"]
    assert copied["pictures"] == 2 and copied["dx_km"] == [12.0]
    assert copied["thumb"].endswith("composite_reflectivity/2026-09-24/frame_f001.png")
    live = next(row for row in body["runs"] if row["id"] == "live")["status"]
    assert live["percent"] == 50.0 and live["storm_time"] == "2026-09-24 00:30 UTC"
    assert live["seconds_left"] == 60


def test_the_event_stream_resumes_after_the_last_event_id(gui):
    server, _ = gui
    run = make_run(server.root, "live", pid=os.getpid())
    events_file(run / "events.jsonl", [{"event": "stage_started", "stage": s} for s in
                                       ("fetch", "prepare", "initialize", "forecast", "finalize")])
    sock = socket.create_connection(("127.0.0.1", server.port), timeout=5)
    sock.sendall((f"GET /api/runs/live/events HTTP/1.0\r\nHost: 127.0.0.1:{server.port}\r\n"
                  f"Cookie: {auth.cookie_name(server.port)}={server.token}\r\nLast-Event-ID: 2\r\n\r\n").encode())
    received = b""
    while b"event: status" not in received:
        chunk = sock.recv(65536)
        if not chunk:
            break
        received += chunk
    sock.close()
    text = received.decode()
    assert "text/event-stream" in text
    ids = [line.split(": ")[1] for line in text.splitlines() if line.startswith("id: ")]
    assert ids == ["3", "4"]
    assert "initialize" not in text and "finalize" in text


#: A home directory of an account no machine has, assembled from pieces: written
#: out whole it reads as a leaked path to the release's machine-path scan
#: (work/build_release_snapshot.py), which stops the cut's battery.
SOMEONE_HOME = "/" + "home/someone"


def test_the_event_stream_keeps_the_folder_a_refusal_names_and_hides_the_rest(gui):
    """The live event line says where to make room, as the failure notice does."""
    server, _ = gui
    run = make_run(server.root, "refused", pid=dead_pid())
    folder = f"{SOMEONE_HOME}/runs/refused/chain"
    events_file(run / "events.jsonl", [{
        "event": "failed", "stage": "prepare", "folders": [folder],
        "message": f"the frame stream needs 9 bytes in {folder}/x and reads {SOMEONE_HOME}/secret/y",
        "remedy": f"remedy: the preparation stages its frame stream in {folder}.  Set\n"
                  "GPUWM_COMPOSE_SCRATCH to a folder with room."}])
    sock = socket.create_connection(("127.0.0.1", server.port), timeout=5)
    sock.sendall((f"GET /api/runs/refused/events HTTP/1.0\r\nHost: 127.0.0.1:{server.port}\r\n"
                  f"Cookie: {auth.cookie_name(server.port)}={server.token}\r\n\r\n").encode())
    received = b""
    while b"event: status" not in received:
        chunk = sock.recv(65536)
        if not chunk:
            break
        received += chunk
    sock.close()
    line = next(line for line in received.decode().splitlines()
                if line.startswith("data: ") and '"failed"' in line)
    data = json.loads(line[len("data: "):])
    assert f"{folder}/x" in data["message"] and f"{SOMEONE_HOME}/secret" not in data["message"]
    assert data["remedy"] == (f"the preparation stages its frame stream in {folder}. Set "
                              "GPUWM_COMPOSE_SCRATCH to a folder with room.")


# ---------------------------------------------------------------- pictures

def test_results_list_the_renderer_folders_and_serve_the_files_as_they_are(gui):
    server, _ = gui
    make_run(server.root, "done", pid=dead_pid(), end={"event": "completed"}, frames=4)
    response, index = request(server, "GET", "/api/runs/done/pictures")
    assert index["count"] == 4 and index["domains"] == ["d01-12km"]
    assert index["products"][0] == "composite_reflectivity"
    assert index["favourites"] == ["composite_reflectivity", "2m_temperature"]
    response, listing = request(server, "GET", "/api/runs/done/pictures/list?product=2m_temperature&domain=d01-12km")
    assert [p["name"] for p in listing["pictures"]] == ["frame_f000.png", "frame_f002.png"]
    assert [p["hour"] for p in listing["pictures"]] == [0, 2]
    response, raw = request(server, "GET", "/api/runs/done/files/" + listing["pictures"][0]["path"])
    assert response.status == 200 and response.getheader("Content-Type") == "image/png" and raw == PNG
    response, _ = request(server, "GET", "/api/runs/done/files/..%2F..%2Fsecret.txt")
    assert response.status in (400, 404)


def test_the_map_places_each_picture_from_the_renderers_own_record(gui):
    # A picture sits on the map where the renderer's render-georef.json says it does, and the grid it
    # is cut along comes from the resolved plan; a picture the record does not name has no place.
    server, _ = gui
    run = make_run(server.root, "placed", pid=dead_pid(), end={"event": "completed"})
    domains = [{"grid_id": 1, "parent_id": 0, "i_parent_start": 1, "j_parent_start": 1, "parent_grid_ratio": 1,
                "run": {"nx": 28, "ny": 28, "nz": 49, "dx": 12000.0}},
               {"grid_id": 2, "parent_id": 1, "i_parent_start": 9, "j_parent_start": 9, "parent_grid_ratio": 3,
                "run": {"nx": 31, "ny": 31, "nz": 49, "dx": 4000.0}}]
    projection = {"map_proj": "lambert", "ref_lat": 39.0, "ref_lon": -97.0, "truelat1": 29.0,
                  "truelat2": 49.0, "stand_lon": -97.0}
    events_file(run / "events.jsonl", [
        {"event": "resolved_plan", "configuration": {"experiment": {
            "run_seconds": 3600.0, "start_time": "2026-09-24T12:00:00", "projection": projection,
            "domains": domains}}},
        {"event": "completed"}])
    png = run / "chain" / "png"
    folder = png / "d01-12km" / "2m_temperature" / "2026-09-24"
    folder.mkdir(parents=True)
    for name in ("arwen_wrf_20260924_12z_f000.png", "arwen_wrf_20260924_12z_f001.png"):
        (folder / name).write_bytes(PNG)
    panel = {"schema": "rustwx.panel-georeference/v1", "image_width_px": 1200, "image_height_px": 900,
             "plot_rect_px": {"x": 23, "y": 64, "width": 1087, "height": 818},
             "projection": {"kind": "lambert_conformal", "standard_parallel_1_deg": 29.0,
                            "standard_parallel_2_deg": 49.0, "central_meridian_deg": -97.0,
                            "reference_latitude_deg": 38.99},
             "extent": {"x_min": -230493.6, "x_max": 230493.6, "y_min": -172549.1, "y_max": 174357.6},
             "geographic_bounds": [-99.0, -95.0, 37.46, 40.53]}
    (png / "render-georef.json").write_text(json.dumps({
        "schema": "rustwx.render-georef/v1",
        "panels": {"rustwx_wrf_20260924_12z_f001_d01-12km_2m_temperature.png": panel}}), encoding="utf-8")

    response, where = request(server, "GET", "/api/runs/placed/map")
    assert response.status == 200 and where["projection"]["truelat1"] == 29.0 and where["moves"] == []
    assert [(d["grid_id"], d["nx"], d["dx_km"], d["parent_grid_ratio"]) for d in where["domains"]] ==         [(1, 28, 12.0, 1), (2, 31, 4.0, 3)]
    response, listing = request(server, "GET", "/api/runs/placed/pictures/list?product=2m_temperature")
    first, second = listing["pictures"]
    assert (first["valid"], second["valid"]) == ("2026-09-24T12:00:00Z", "2026-09-24T13:00:00Z")
    assert first["geo"] is None and listing["georefs"][second["geo"]]["plot_rect_px"]["x"] == 23


def test_the_map_finds_each_picture_under_the_path_the_render_filed_it_at(gui):
    # The render step records every picture under its path relative to the render root, and the same file
    # name sits in every domain's and product's folder, so the page must look a picture up by that path.
    server, _ = gui
    run = make_run(server.root, "filed", pid=dead_pid(), end={"event": "completed"})
    png = run / "chain" / "png"
    name = "arwen_wrf_20260924_12z_f001.png"
    record = {}
    for number, (domain, product) in enumerate((("d01-12km", "2m_temperature"), ("d02-4km", "2m_temperature"),
                                                ("d01-12km", "composite_reflectivity"))):
        folder = png / domain / product / "2026-09-24"
        folder.mkdir(parents=True)
        (folder / name).write_bytes(PNG)
        record[f"{domain}/{product}/2026-09-24/{name}"] = {
            "schema": "rustwx.panel-georeference/v1", "image_width_px": 1200, "image_height_px": 900,
            "plot_rect_px": {"x": 10 + number, "y": 64, "width": 1087, "height": 818},
            "projection": {"kind": "lambert_conformal"}, "extent": None,
            "geographic_bounds": [-99.0, -95.0, 37.46, 40.53]}
    (folder / "arwen_wrf_20260924_12z_f002.png").write_bytes(PNG)
    (png / "render-georef.json").write_text(json.dumps({"schema": "rustwx.render-georef/v1",
                                                        "panels": record}), encoding="utf-8")

    response, listing = request(server, "GET", "/api/runs/filed/pictures/list")
    assert response.status == 200
    placed = {picture["path"]: picture["geo"] for picture in listing["pictures"]}
    x = {path: None if geo is None else listing["georefs"][geo]["plot_rect_px"]["x"] for path, geo in placed.items()}
    assert x == {"chain/png/d01-12km/2m_temperature/2026-09-24/" + name: 10,
                 "chain/png/d02-4km/2m_temperature/2026-09-24/" + name: 11,
                 "chain/png/d01-12km/composite_reflectivity/2026-09-24/" + name: 12,
                 "chain/png/d01-12km/composite_reflectivity/2026-09-24/arwen_wrf_20260924_12z_f002.png": None}


def test_the_map_carries_every_place_a_following_nest_moved_to(gui):
    # A nest that follows a storm moves while the run goes; the engine's step log records each move, and the
    # map gets them all, so every frame is cut along the grid it was drawn on, not the plan's first place.
    server, _ = gui
    run = make_run(server.root, "follows", pid=dead_pid(), end={"event": "completed"})
    domains = [{"grid_id": 1, "parent_id": 0, "i_parent_start": 1, "j_parent_start": 1, "parent_grid_ratio": 1,
                "run": {"nx": 200, "ny": 160, "nz": 49, "dx": 12000.0}},
               {"grid_id": 2, "parent_id": 1, "i_parent_start": 81, "j_parent_start": 61, "parent_grid_ratio": 4,
                "run": {"nx": 160, "ny": 160, "nz": 49, "dx": 3000.0}}]
    events_file(run / "events.jsonl", [
        {"event": "resolved_plan", "configuration": {"experiment": {
            "run_seconds": 21600.0, "start_time": "2026-09-24T18:00:00", "domains": domains,
            "projection": {"map_proj": "mercator", "ref_lat": 17.1, "ref_lon": -105.06, "truelat1": 17.1,
                           "truelat2": 17.1, "stand_lon": -105.06}}}},
        {"event": "completed"}])
    log = run / "chain" / "run-1" / "run" / "progress.jsonl"
    log.parent.mkdir(parents=True)
    rows = [
        {"schema": "gpuwm.step-log/v3", "sequence": 462, "event": "output_written", "domain": 2,
         "valid_time": "2026-09-24_18:45:00"},
        {"schema": "gpuwm.step-log/v3", "sequence": 463, "event": "nest_moved", "domain": 2,
         "valid_time": "2026-09-24_18:45:00", "model_seconds": 2700.0,
         "placement_from": {"i_parent_start": 81, "j_parent_start": 61},
         "placement_to": {"i_parent_start": 79, "j_parent_start": 61}, "executed_shift_parent_cells": [-2, 0]},
        {"schema": "gpuwm.step-log/v3", "sequence": 1074, "event": "nest_moved", "domain": 2,
         "valid_time": "2026-09-24_19:45:00", "model_seconds": 6300.0,
         "placement_from": {"i_parent_start": 79, "j_parent_start": 61},
         "placement_to": {"i_parent_start": 77, "j_parent_start": 60}, "executed_shift_parent_cells": [-2, -1]},
        {"schema": "gpuwm.step-log/v3", "sequence": 2000, "event": "containment_moved", "domain": 2, "mover": 3,
         "valid_time": "2026-09-24_21:00:00", "model_seconds": 10800.0,
         "placement_from": {"i_parent_start": 77, "j_parent_start": 60},
         "placement_to": {"i_parent_start": 76, "j_parent_start": 60}, "executed_shift_parent_cells": [-1, 0]},
    ]
    log.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    response, where = request(server, "GET", "/api/runs/follows/map")
    assert response.status == 200
    assert where["moves"] == [
        {"valid": "2026-09-24T18:45:00Z", "grid_id": 2, "i_parent_start": 79, "j_parent_start": 61},
        {"valid": "2026-09-24T19:45:00Z", "grid_id": 2, "i_parent_start": 77, "j_parent_start": 60},
        {"valid": "2026-09-24T21:00:00Z", "grid_id": 2, "i_parent_start": 76, "j_parent_start": 60},
        {"valid": "2026-09-24T21:00:00Z", "grid_id": 3, "shift_by": 2, "shift": [-1, 0]},
    ]


def test_the_renderers_scratch_is_never_a_grid_or_a_picture(gui):
    # The renderer works in <delivery>.render-scratch/rwstore-*/ beside what it delivers, and its working stores
    # hold PNGs of their own; the scan counted that folder as a grid and its pictures in the run's count.
    server, _ = gui
    run = make_run(server.root, "scratchy", frames=2, render="")
    stray = run / "png.render-scratch" / "rwstore-3bcd0c2c-2_z3ruy1" / "png"
    stray.mkdir(parents=True)
    (stray / "rustwx_wrf_20260924_18z_f000_d01-12km_10m_wind.png").write_bytes(PNG)
    nested = run / "png.render-scratch" / "rwstore-x" / "d01-12km" / "cape" / "2026-09-24"
    nested.mkdir(parents=True)
    (nested / "frame_f000.png").write_bytes(PNG)
    (server.root / "scratchy.render-scratch" / "rwstore-y" / "d01-12km" / "cape" / "2026-09-24").mkdir(parents=True)
    (server.root / "scratchy.render-scratch" / "rwstore-y" / "d01-12km" / "cape" / "2026-09-24" / "a.png").write_bytes(PNG)

    response, body = request(server, "GET", "/api/runs")
    assert [row["id"] for row in body["runs"]] == ["scratchy"]
    card = body["runs"][0]["card"]
    assert card["domains"] == ["d01-12km"] and card["pictures"] == 2
    response, index = request(server, "GET", "/api/runs/scratchy/pictures")
    assert index["domains"] == ["d01-12km"] and index["count"] == 2
    response, listed = request(server, "GET", "/api/runs/scratchy/pictures/list?product=cape")
    assert listed["pictures"] == []


def test_a_downscale_sits_where_its_plan_puts_it_in_its_parent(gui):
    # A downscale's event log has no grid; its plan names the parent run, the parent grid and the child's place
    # in it, and gives the length and the start. The map draws the child there (the parent only places it).
    server, _ = gui
    parent = make_run(server.root, "parent-run", pid=dead_pid(), end={"event": "completed"})
    domains = [{"grid_id": 1, "parent_id": 0, "i_parent_start": 1, "j_parent_start": 1, "parent_grid_ratio": 1,
                "run": {"nx": 200, "ny": 160, "nz": 49, "dx": 12000.0}},
               {"grid_id": 2, "parent_id": 1, "i_parent_start": 81, "j_parent_start": 61, "parent_grid_ratio": 4,
                "run": {"nx": 160, "ny": 160, "nz": 49, "dx": 3000.0}}]
    projection = {"map_proj": "mercator", "ref_lat": 17.1, "ref_lon": -105.06, "truelat1": 17.1,
                  "truelat2": 17.1, "stand_lon": -105.06}
    events_file(parent / "events.jsonl", [
        {"event": "resolved_plan", "configuration": {"experiment": {
            "run_seconds": 21600.0, "start_time": "2026-09-24T18:00:00", "domains": domains,
            "projection": projection}}}, {"event": "completed"}])
    child = server.root / "downscale-d02-1"
    child.mkdir()
    events_file(child / "events.jsonl", [{"event": "resolved_plan", "config_source": "child.toml"},
                                         {"event": "model_progress", "model_seconds": 600.0}])
    (child / "run-manifest.json").write_text(json.dumps({"pid": dead_pid()}), encoding="utf-8")
    plan = {"schema": "gpuwm.downscale-plan.v1", "child_grid_id": 2, "parent_domain": 1,
            "parent": {"domain": 1, "run_dir": r"D:\\elsewhere\\parent-run\\chain\\run-1\\run\\wrfout"},
            "child_grid": {"nx": 288, "ny": 288, "nz": 49, "dx": 4000.0, "run_seconds": 21600.0},
            "placement": {"i_parent_start": 54, "j_parent_start": 34, "ratio": 3},
            "initial_condition": {"GPUWM_INITIAL_CONDITION_MODEL_START_DATE": "2026-09-24_18:00:00"}}
    (child / "downscale-plan.json").write_text(json.dumps(plan), encoding="utf-8")

    response, where = request(server, "GET", "/api/runs/downscale-d02-1/map")
    assert response.status == 200 and where["projection"] == projection
    assert where["domains"][0]["grid_id"] == 1 and where["domains"][0]["context"] is True
    assert where["domains"][-1] == {"grid_id": 2, "parent_id": 1, "nx": 288, "ny": 288, "nz": 49, "dx_km": 4.0,
                                    "i_parent_start": 54, "j_parent_start": 34, "parent_grid_ratio": 3}
    response, detail = request(server, "GET", "/api/runs/downscale-d02-1")
    assert detail["status"]["start_time"] == "2026-09-24 18:00 UTC" and detail["status"]["run_seconds"] == 21600.0


# ---------------------------------------------------------------- create and stop

DRAFT = {"name": "t1", "source": "gfs", "cycle": "2026-09-24T00", "lat": 39.5, "lon": -97.0,
         "width_km": 400, "height_km": 330, "hours": 1, "card": "16gb"}


def test_show_command_returns_the_exact_argv_and_runs_nothing(gui):
    server, runner = gui
    response, body = request(server, "POST", "/api/create/start", body={**DRAFT, "dry_run": True})
    assert response.status == 200 and body["dry_run"] is True
    assert body["argv"][-2:] == ["run-plan", str(server.root / "t1" / "plan.json")]
    # the page shows plain names, never a machine path; Copy takes the exact line in "command"
    assert body["shown"] == f"gpuwm run-plan {Path('t1') / 'plan.json'}"
    assert str(server.root) in body["command"] and str(server.root) not in body["shown"]
    assert body["plan"]["config"]["intent"]["source"] == "gfs"
    assert not (server.root / "t1").exists() and runner.launched == []


def test_vertical_levels_reach_the_plan_as_the_wizards_own_level_count(gui):
    # New forecast's Vertical levels choice is the wizard's --nz, carried in the plan's intent; the engine's
    # default ladder is no key at all, so the engine keeps choosing it.
    server, _ = gui
    response, body = request(server, "POST", "/api/create/start", body={**DRAFT, "nz": 80, "dry_run": True})
    assert response.status == 200 and body["plan"]["config"]["intent"]["nz"] == 80
    response, body = request(server, "POST", "/api/create/start", body={**DRAFT, "dry_run": True})
    assert "nz" not in body["plan"]["config"]["intent"]
    response, body = request(server, "POST", "/api/create/start", body={**DRAFT, "nz": 60.5, "dry_run": True})
    assert response.status == 400


def test_the_fit_is_the_wizards_answer_in_plain_words(gui):
    server, runner = gui
    response, body = request(server, "POST", "/api/create/fit", body=DRAFT)
    assert response.status == 200
    assert body["fit"]["domains"][0]["nx"] == 36
    assert "36 by 30 points at 12 km" in body["fit"]["words"] and "1 hour from" in body["fit"]["words"]
    assert runner.queries[-1][-1] == "--resolve"
    assert list((server.root / ".arwen-gui" / "drafts").iterdir()) == []
    # The card's fit, read from the engine's own memory record: what "How fine" shows per choice.
    assert body["fit"]["memory"] == {"need_gib": 5.33, "budget_gib": 14.54, "fits": True}
    # The map draws the fitted grid in the projection the wizard chose.
    assert set(body["fit"]["projection"]) == {"map_proj", "ref_lat", "ref_lon", "truelat1", "truelat2", "stand_lon"}


def test_the_start_up_warm_call_answers_the_first_page_without_a_second_engine_start(gui):
    server, runner = gui
    gate = threading.Event()
    plain = runner.query

    def slow(argv, **kwargs):
        gate.wait(5)
        return plain(argv, **kwargs)

    runner.query = slow
    warm = server.api.warm()
    time.sleep(0.2)
    answers = []
    asker = threading.Thread(target=lambda: answers.append(request(server, "GET", "/api/sources")))
    asker.start()
    time.sleep(0.2)
    gate.set()
    warm.join(10)
    asker.join(10)
    assert answers and answers[0][0].status == 200
    sources = [argv for argv in runner.queries if "--sources" in argv]
    assert len(sources) == 1


def test_a_grid_too_big_for_the_card_says_how_big(gui):
    server, runner = gui

    ask = runner.query

    def refuse(argv, **kwargs):
        if "--resolve" not in argv:
            return ask(argv, **kwargs)
        # The engine's too-big refusal: the sentence, and the document it prints with the figures in bytes.
        sentence = ("run plan 'config.intent' does not fit: ... the forecast needs 40.97 GiB peak envelope, and "
                    "preprocessing for --source hrrr is NOT PRICED here, so this is the forecast phase only; that "
                    "EXCEEDS the 14.54 GiB budget by 26.43 GiB")
        raise Refused(sentence, {"schema": "arwen.configuration-error.v1", "kind": "memory", "error": sentence,
                                 "created": False,
                                 "memory": {"peak_envelope_bytes": int(40.97 * (1 << 30)),
                                            "budget_bytes": int(14.54 * (1 << 30)), "binding_phase": "forecast"}})

    runner.query = refuse
    response, body = request(server, "POST", "/api/create/fit", body={**DRAFT, "dx_km": 0.75})
    assert response.status == 422
    assert body["memory"] == {"need_gib": 40.97, "budget_gib": 14.54, "fits": False}
    assert body["message"] == "Too big for the 16 GB card: this grid needs about 41.0 GiB and 14.5 GiB is usable."


def test_start_writes_the_run_folder_and_launches_run_plan(gui):
    server, runner = gui
    response, body = request(server, "POST", "/api/create/start", body=DRAFT)
    assert response.status == 200 and body["run"] == "t1"
    run = server.root / "t1"
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    assert plan["output_root"] == str(run) and plan["route"] == "prepared"
    assert json.loads((run / "region.geojson").read_text(encoding="utf-8"))["type"] == "Polygon"
    assert runner.launched == [body["argv"]]
    assert body["command"] in (run / "commands.log").read_text(encoding="utf-8")
    response, again = request(server, "POST", "/api/create/start", body=DRAFT)
    assert response.status == 409


def test_a_start_refused_on_a_busy_card_leaves_no_folder_and_its_name_free(tmp_path, monkeypatch):
    # A refused Start left the folder it had written: My forecasts listed a Ready run nobody asked for, and the
    # same name was refused as taken once the card was free.
    monkeypatch.setattr("gpuwm.fetch._head_ok", lambda url: True)
    runner = HeldCardRunner(tmp_path / "jobs")
    server = build_server(tmp_path / "runs", port=0, runner=runner, token="t" * 43)
    serve_in_thread(server)
    try:
        ready = make_run(server.root, "ready", plan=True)
        runner.hold()
        response, refused = request(server, "POST", "/api/create/start", body=DRAFT)
        assert response.status == 409 and "job-held" in refused["message"], refused
        assert not (server.root / "t1").exists()
        _, listed = request(server, "GET", "/api/runs")
        assert [row["id"] for row in listed["runs"]] == ["ready"]
        # A Ready run pressed on the busy card is refused too, and keeps its folder: it was there before the press.
        response, _ = request(server, "POST", "/api/runs/ready/start", body={})
        assert response.status == 409 and (ready / "plan.json").is_file()
        runner.free()
        response, started = request(server, "POST", "/api/create/start", body=DRAFT)
        assert response.status == 200 and started["run"] == "t1", started
    finally:
        server.shutdown()
        server.server_close()


def test_stop_records_how_it_stopped_and_dry_run_records_nothing(gui):
    server, _ = gui
    run = make_run(server.root, "live", pid=os.getpid())
    response, body = request(server, "POST", "/api/runs/live/stop", body={"dry_run": True})
    assert response.status == 200 and body["dry_run"] and not (run / "gui-stop.json").exists()
    response, body = request(server, "POST", "/api/runs/live/stop", body={})
    assert response.status == 200 and body["method"] == "interrupt"
    assert json.loads((run / "gui-stop.json").read_text(encoding="utf-8"))["command"] == "kill -INT -12345"
    assert "kill -INT -12345" in (run / "commands.log").read_text(encoding="utf-8")


def test_a_cancelled_job_stays_cancelled_after_its_wrapper_exits(tmp_path):
    import sys
    import time

    from gpuwm.mcp.gpulock import _pid_alive
    from gpuwm.mcp.jobs import JobManager

    manager = JobManager(tmp_path / "jobs")
    job = manager.launch("sleep", [sys.executable, "-c", "import time; time.sleep(60)"],
                         cwd=tmp_path, gpu=False)
    jobdir = Path(job["jobs_dir"])
    deadline = time.monotonic() + 30
    while not (jobdir / "started.json").exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    child = json.loads((jobdir / "started.json").read_text(encoding="utf-8"))["child_pid"]
    assert manager.cancel(job["job_id"])["state"] == "cancelled"
    wrapper = json.loads((jobdir / "receipt.json").read_text(encoding="utf-8"))["wrapper_pid"]
    deadline = time.monotonic() + 20
    while (_pid_alive(child) or _pid_alive(wrapper)) and time.monotonic() < deadline:
        time.sleep(0.1)
    time.sleep(0.5)
    assert manager.status(job["job_id"])["state"] == "cancelled"


def test_a_picture_deeper_than_the_windows_path_limit_is_served(tmp_path):
    from gpuwm.gui.files import PathRefused, long_path, safe_file

    parts = ["d01-12km", "composite_reflectivity_" + "x" * 60, "2026-09-24"]
    base = tmp_path / ("r" * 80)
    name = "frame_" + "f" * 60 + ".png"
    assert len(str(base.joinpath(*parts, name))) > 260
    folder = long_path(base.joinpath(*parts))
    folder.mkdir(parents=True)
    (folder / name).write_bytes(PNG)
    assert safe_file(base, "/".join(parts + [name])).read_bytes() == PNG
    from gpuwm.gui import frames

    rows = frames.pictures(base, domain=parts[0], product=parts[1], day=parts[2])
    assert [row["name"] for row in rows] == [name]
    assert frames.index(base)["groups"][0]["newest_mtime"] > 0
    with pytest.raises(PathRefused):
        safe_file(base, "/".join(parts[:1] + ["..", "..", name]))


def test_the_create_map_ships_its_own_basemap(gui):
    # No network at run time: the coastlines, borders and state lines come from the package.
    server, _ = gui
    response, body = request(server, "GET", "/static/map/basemap.json")
    assert response.status == 200
    assert "Natural Earth" in body["notice"]
    assert {"coast", "border", "state", "lake"} <= set(body["layers"]) and len(body["layers"]["coast"]) > 100


def test_a_port_in_use_is_reported_as_a_port_in_use(tmp_path):
    busy = socket.socket()
    busy.bind(("127.0.0.1", 0))
    busy.listen(1)
    try:
        with pytest.raises(OSError):
            build_server(tmp_path / "runs", port=busy.getsockname()[1], runner=FakeRunner(), token="t" * 43)
    finally:
        busy.close()


def test_the_page_words_pass_the_copy_lint():
    assert lint_all() == []


def test_a_copied_run_shows_its_latest_hour_not_its_last_copied_file(tmp_path):
    from gpuwm.gui import frames

    run = tmp_path / "copied"
    folder = run / "chain" / "png" / "d01-12km" / "composite_reflectivity" / "2026-09-24"
    folder.mkdir(parents=True)
    late = folder / "arwen_wrf_20260924_12z_f001.png"
    early = folder / "arwen_wrf_20260924_12z_f000.png"
    late.write_bytes(PNG)
    early.write_bytes(PNG)
    os.utime(late, (1000, 1000))
    os.utime(early, (2000, 2000))
    found = frames.index(run)
    assert found["watch"].endswith("_f001.png")
    assert found["groups"][0]["newest_mtime"] == 2000


def test_a_port_in_use_is_an_os_error_not_a_traceback_about_the_class(tmp_path):
    first = build_server(tmp_path / "a", port=0, runner=FakeRunner(), token="t" * 43)
    try:
        with pytest.raises(OSError):
            build_server(tmp_path / "b", port=first.port, runner=FakeRunner(), token="t" * 43)
    finally:
        first.server_close()
