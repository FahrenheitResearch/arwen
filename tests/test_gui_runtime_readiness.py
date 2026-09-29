"""A computer whose gpuwm cannot run a forecast is told so before Start, not after.

The defect this file holds shut: on an install without the GPU runtime
(CuPy), the page's queue answered ``start_now: true`` for this computer and
Start was on.  The forecast was launched, ``gpuwm run-plan`` refused at its
first step for want of the runtime, and the forecast was listed as failed.
A forecast queued for this computer was started into the same refusal.

The page server refuses to import CuPy itself, so the runtime is looked for
on the import path, never imported; that refusal must not read as a
missing runtime.
"""

from __future__ import annotations

import sys

import pytest

import gpuwm.capabilities  # noqa: F401 - loaded before a test narrows the import path to its own folder
from gpuwm.gui import runs
from gpuwm.gui.api import ApiError
from gpuwm.gui.jobs import Runner
from gpuwm.gui.machines import LOCAL
from gpuwm.gui.server import _NoGpuImports
from test_gui_queue_join_and_leave import DRAFT, page  # noqa: F401 - page is a fixture
from test_gui_server import request

GAP = "gpuwm's GPU runtime (CuPy) is not installed on this computer, so no forecast can run here."


def start(server, name, *, queue=False):
    return request(server, "POST", "/api/create/start", body={**DRAFT, "name": name, "queue": queue, "need_gib": 5.3})


# ---------------------------------------------------------------- the check itself


def test_the_runtime_is_found_on_the_path_even_though_the_page_server_refuses_to_import_it(tmp_path, monkeypatch):
    package = tmp_path / "site" / "cupy"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("raise AssertionError('the page server ran GPU code')\n", encoding="utf-8")
    monkeypatch.setattr(sys, "path", [str(tmp_path / "site")])
    monkeypatch.setattr(sys, "meta_path", [_NoGpuImports(), *sys.meta_path])
    assert Runner().runtime_gap() is None


@pytest.mark.parametrize("layout", ["absent", "bare folder"])
def test_an_install_without_the_runtime_is_named_with_its_remedy(tmp_path, monkeypatch, layout):
    site = tmp_path / "site"
    site.mkdir()
    if layout == "bare folder":
        (site / "cupy").mkdir()     # a folder of that name with no package in it
    monkeypatch.setattr(sys, "path", [str(site)])
    assert Runner().runtime_gap() == GAP


# ---------------------------------------------------------------- what the page is told


def test_new_forecast_offers_no_start_here_without_the_runtime(page):
    server, runner = page
    runner.runtime_gap = lambda: GAP
    response, body = request(server, "GET", "/api/queue?need_gib=5.3")
    assert response.status == 200
    expect = body["expect"]
    assert expect["start_now"] is False and expect["busy"] is False
    assert expect["held"].startswith("Held: " + GAP)
    assert "pip install 'gpuwm[gpu-cu12]' or 'gpuwm[gpu-cu13]'" in expect["held"] and "gpuwm doctor" in expect["held"]
    # Installed, the same card takes Start at once.
    runner.runtime_gap = lambda: None
    _, body = request(server, "GET", "/api/queue?need_gib=5.3")
    assert body["expect"]["start_now"] is True and body["expect"]["held"] is None


def test_start_here_without_the_runtime_launches_nothing_and_leaves_the_name_free(page):
    server, runner = page
    runner.runtime_gap = lambda: GAP
    response, body = start(server, "no-runtime")
    assert response.status == 409, body
    assert body["message"] == f"{GAP} Nothing started."
    assert "gpuwm[gpu-cu12]" in body["fix"] and "Machines page" in body["fix"]
    assert runner.launched == [] and not (server.root / "no-runtime").exists()
    runner.runtime_gap = lambda: None
    response, body = start(server, "no-runtime")
    assert response.status == 200 and len(runner.launched) == 1


def test_a_forecast_queued_here_is_held_until_the_runtime_is_installed_then_starts(page, monkeypatch):
    server, runner = page
    runner.runtime_gap = lambda: GAP
    monkeypatch.setattr(server.api, "availability_of", lambda *args, **kwargs: {"starts": "now"})
    response, body = start(server, "waits", queue=True)
    assert response.status == 200 and body["queued"], body
    queue = server.api.queue
    assert queue.tick() == [] and runner.launched == []
    _, listing = request(server, "GET", "/api/queue")
    item = listing["items"][0]
    assert item["run"] == "waits" and item["held"].startswith("Held: " + GAP)
    assert item["held"].endswith("and it starts by itself.") and item["awaits_data"] is False
    runner.runtime_gap = lambda: None
    assert queue.tick() == ["waits"]
    assert len(runner.launched) == 1 and (server.root / "waits" / runs.JOB).is_file()


def test_every_start_on_this_card_is_refused_before_it_is_launched(page):
    """A downscale's start and a run page's Start go through the same door as New forecast's."""

    server, runner = page
    runner.runtime_gap = lambda: GAP
    launched = []
    with pytest.raises(ApiError) as refused:
        server.api.queue.launch_holding_card("finer", lambda: launched.append(True))
    assert refused.value.status == 409 and refused.value.message.startswith(GAP)
    assert launched == []


def test_another_machine_is_not_held_for_this_computers_runtime(page, monkeypatch):
    server, runner = page
    runner.runtime_gap = lambda: GAP
    queue = server.api.queue
    monkeypatch.setattr(queue, "card", lambda machine: {"machine": machine, "busy": False, "disk_free_gib": 100.0,
                                                          "total_gib": 16.0})
    expect = queue.expect("box", 5.3)
    assert expect["held"] is None and expect["start_now"] is True
    assert queue.expect(LOCAL, 5.3)["start_now"] is False
