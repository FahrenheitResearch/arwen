"""The run folders the engine writes are the run folders the page reads.

The engine accepts any output folder name and writes episodic nests one
segment deeper; the page must list, open and draw every one of them, and
one folder it cannot read must never take the others down with it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import quote

import pytest

from gpuwm.gui import frames, runs
from test_gui_server import PNG, dead_pid, gui, make_run, request  # noqa: F401 - gui is a fixture


def _link(link: Path, target: Path) -> None:
    """A folder link: a symlink where the account may make one, else a junction (Windows)."""

    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        import _winapi

        _winapi.CreateJunction(str(target), str(link))


# ---------------------------------------------------------------- one bad run, not a broken list

def test_a_damaged_plan_and_a_link_outside_the_root_leave_the_list_standing(gui, tmp_path):
    server, _ = gui
    make_run(server.root, "healthy", plan=True)
    damaged = server.root / "damaged"
    damaged.mkdir()
    (damaged / "plan.json").write_text(json.dumps({"config": "invalid"}), encoding="utf-8")
    elsewhere = make_run(tmp_path / "elsewhere", "external", plan=True)
    _link(server.root / "archive" / "linked", elsewhere)

    response, body = request(server, "GET", "/api/runs")
    assert response.status == 200, body
    ids = {row["id"]: row for row in body["runs"]}
    assert ids["healthy"]["status"]["state"] == "ready"
    assert ids["damaged"]["status"]["state"] == "ready"
    assert "archive/linked" not in ids
    response, _ = request(server, "GET", "/api/runs/" + quote("archive/linked", safe=""))
    assert response.status in (400, 404)


def test_a_queued_marker_that_is_not_a_mapping_leaves_the_queue_and_the_list_standing(gui):
    server, _ = gui
    make_run(server.root, "healthy", plan=True)
    odd = make_run(server.root, "odd", plan=True)
    (odd / runs.QUEUED).write_text(json.dumps(["not", "a", "marker"]), encoding="utf-8")
    response, body = request(server, "GET", "/api/queue")
    assert response.status == 200, body
    response, body = request(server, "GET", "/api/runs")
    assert response.status == 200, body
    assert "healthy" in {row["id"] for row in body["runs"]}


@pytest.mark.parametrize("plan", [
    {"config": "invalid"}, {"config": {"intent": "text"}}, ["a", "list"], "text",
    {"config": {"intent": {"hours": "six", "cycle": 12, "forecast_start_hour": "x", "source": ["gfs"]}}},
])
def test_plan_facts_take_only_the_types_they_expect(tmp_path, plan):
    run = tmp_path / "run"
    run.mkdir()
    (run / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    facts = runs._plan_facts(run)
    assert "run_seconds" not in facts and "start_time" not in facts
    assert facts.get("source") is None


def test_a_run_that_cannot_be_read_is_listed_as_unreadable_and_the_rest_still_load(gui, monkeypatch):
    server, _ = gui
    make_run(server.root, "healthy", plan=True)
    make_run(server.root, "broken", plan=True)
    real = runs.status

    def status(rundir):
        if rundir.name == "broken":
            raise RuntimeError(f"cannot read {rundir}")
        return real(rundir)

    monkeypatch.setattr(runs, "status", status)
    response, body = request(server, "GET", "/api/runs")
    assert response.status == 200, body
    rows = {row["id"]: row for row in body["runs"]}
    assert rows["healthy"]["status"]["state"] == "ready"
    bad = rows["broken"]["status"]
    assert bad["state"] == "unreadable"
    # plain words: no machine path and no traceback
    assert bad["message"] and str(server.root) not in bad["message"] and "Traceback" not in bad["message"]
    assert "RuntimeError" not in bad["message"]


def test_a_link_one_level_down_is_not_discovered(tmp_path):
    root = tmp_path / "root"
    make_run(root, "group/inside", plan=True)
    outside = make_run(tmp_path / "far", "away", plan=True)
    _link(root / "group" / "linked", outside)
    found = [p.relative_to(root).as_posix() for p in runs.iter_runs(root)]
    assert found == ["group/inside"]


def test_a_link_inside_the_root_lists_its_run_once_and_it_opens(gui):
    server, _ = gui
    make_run(server.root, "group/inside", plan=True)
    make_run(server.root, "deep/er/still", plan=True)
    _link(server.root / "latest", server.root / "group")
    _link(server.root / "too-deep", server.root / "deep" / "er" / "still")
    response, body = request(server, "GET", "/api/runs")
    assert response.status == 200, body
    assert [row["id"] for row in body["runs"]] == ["group/inside"]
    for run_id in ("group/inside", "latest/inside"):
        response, body = request(server, "GET", "/api/runs/" + quote(run_id, safe="") + "/status")
        assert response.status == 200, (run_id, body)


# ---------------------------------------------------------------- ordinary folder names open

LONG = "a" * 65


@pytest.mark.parametrize("name", ["May 20 Oklahoma", "München-2026", LONG, "grouped/May 3 outbreak"])
def test_a_listed_run_with_an_ordinary_folder_name_opens(gui, name):
    server, _ = gui
    make_run(server.root, name, pid=dead_pid(), end={"event": "completed"}, plan=True, frames=2)
    response, body = request(server, "GET", "/api/runs")
    assert response.status == 200
    row = next(r for r in body["runs"] if r["id"] == name)
    base = "/api/runs/" + row["url"]
    for suffix in ("", "/status", "/pictures", "/files/plan.json"):
        response, body = request(server, "GET", base + suffix)
        assert response.status == 200, (suffix, body)
    response, body = request(server, "GET", "/api/wiki/run/" + row["url"])
    assert response.status == 200, body


@pytest.mark.parametrize("run_id", ["..", ".", "a/..", "../x", ".arwen-gui", "a\\b", "C:", "x\x00y", "a/b/c", "",
                                    ".hidden", " ", "a/ "])
def test_lookup_still_refuses_what_is_not_a_folder_under_the_root(tmp_path, run_id):
    root = tmp_path / "root"
    root.mkdir()
    for name in (".hidden", ".arwen-gui", "a", "a/b/c"):
        (root / name).mkdir(parents=True, exist_ok=True)
        (root / name / "plan.json").write_text("{}", encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        runs.existing_run(root, run_id)


def test_new_names_keep_the_plain_rule():
    from gpuwm.gui.files import PathRefused, require_name

    for bad in ("May 20 Oklahoma", "München-2026", LONG):
        with pytest.raises(PathRefused):
            require_name(bad, "run name")


# ---------------------------------------------------------------- the newest runs are never cut

def test_a_large_collection_lists_its_newest_run_and_says_it_is_cut(gui):
    server, _ = gui
    count = runs.LIST_LIMIT + 1
    for number in range(count):
        run = server.root / f"run-{number:04d}"
        run.mkdir(parents=True)
        plan = run / "plan.json"
        plan.write_text("{}", encoding="utf-8")
        os.utime(plan, (1_700_000_000 + number, 1_700_000_000 + number))
    response, body = request(server, "GET", "/api/runs", timeout=300)
    assert response.status == 200
    ids = [row["id"] for row in body["runs"]]
    assert ids[0] == f"run-{count - 1:04d}"
    assert "run-0000" not in ids and len(ids) == runs.LIST_LIMIT
    assert body["truncated"] is True and body["total"] == count


def test_a_small_collection_is_not_cut(gui):
    server, _ = gui
    make_run(server.root, "one", plan=True)
    response, body = request(server, "GET", "/api/runs")
    assert body["truncated"] is False and body["total"] == 1


# ---------------------------------------------------------------- episodic nests keep their domain

def _episodic(run: Path) -> None:
    png = run / "chain" / "png"
    name = "arwen_wrf_20260924_12z_f001.png"
    record = {}
    for number, domain in enumerate(("d02-3km", "d03-1km")):
        folder = png / domain / "episode-001" / "2m_temperature" / "2026-09-24"
        folder.mkdir(parents=True)
        (folder / name).write_bytes(PNG)
        record[f"{domain}/episode-001/2m_temperature/2026-09-24/{name}"] = {
            "schema": "rustwx.panel-georeference/v1", "image_width_px": 1200, "image_height_px": 900,
            "plot_rect_px": {"x": 10 + number, "y": 64, "width": 1087, "height": 818},
            "projection": {"kind": "lambert_conformal"}, "extent": None,
            "geographic_bounds": [-99.0, -95.0, 37.46, 40.53]}
    (png / "render-georef.json").write_text(json.dumps({"schema": "rustwx.render-georef/v1", "panels": record}),
                                            encoding="utf-8")


def test_episodic_nests_keep_their_domain_their_episode_and_their_place(gui):
    server, _ = gui
    run = make_run(server.root, "episodes", pid=dead_pid(), end={"event": "completed"})
    _episodic(run)

    response, index = request(server, "GET", "/api/runs/episodes/pictures")
    assert response.status == 200
    assert index["domains"] == ["d02-3km", "d03-1km"]
    assert sorted((g["domain"], g["episode"], g["product"], g["day"]) for g in index["groups"]) == [
        ("d02-3km", "episode-001", "2m_temperature", "2026-09-24"),
        ("d03-1km", "episode-001", "2m_temperature", "2026-09-24")]
    assert set(index["by_domain"]) == {"d02-3km", "d03-1km"}

    response, listing = request(server, "GET", "/api/runs/episodes/pictures/list?product=2m_temperature")
    got = {p["domain"]: (p["episode"], None if p["geo"] is None else listing["georefs"][p["geo"]]["plot_rect_px"]["x"])
           for p in listing["pictures"]}
    assert got == {"d02-3km": ("episode-001", 10), "d03-1km": ("episode-001", 11)}
    response, one = request(server, "GET", "/api/runs/episodes/pictures/list?domain=d03-1km&episode=episode-001")
    assert [p["domain"] for p in one["pictures"]] == ["d03-1km"]

    card = frames.card(run)
    assert card["domains"] == ["d02-3km", "d03-1km"] and card["dx_km"] == [3.0, 1.0]


def test_two_lives_of_one_nest_are_two_groups_of_one_domain(tmp_path):
    run = tmp_path / "run"
    for episode in ("episode-001", "episode-002"):
        folder = run / "png" / "d02-3km" / episode / "2m_temperature" / "2026-09-24"
        folder.mkdir(parents=True)
        (folder / "arwen_wrf_20260924_12z_f001.png").write_bytes(PNG)
    found = frames.index(run)
    assert found["domains"] == ["d02-3km"]
    assert sorted(g["episode"] for g in found["groups"]) == ["episode-001", "episode-002"]
    assert found["by_domain"]["d02-3km"]["count"] == 2


def test_a_plain_layout_has_no_episode(tmp_path):
    run = tmp_path / "run"
    folder = run / "png" / "d01-12km" / "2m_temperature" / "2026-09-24"
    folder.mkdir(parents=True)
    (folder / "arwen_wrf_20260924_12z_f001.png").write_bytes(PNG)
    found = frames.index(run)
    assert [(g["domain"], g["episode"]) for g in found["groups"]] == [("d01-12km", "")]
    assert [p["episode"] for p in frames.pictures(run)] == [""]


def test_a_pictures_only_folder_of_episodic_nests_is_found(tmp_path):
    root = tmp_path / "root"
    folder = root / "copied" / "d02-3km" / "episode-002" / "2m_temperature" / "2026-09-24"
    folder.mkdir(parents=True)
    (folder / "arwen_wrf_20260924_12z_f001.png").write_bytes(PNG)
    assert [p.name for p in runs.iter_runs(root)] == ["copied"]
    assert runs.status(root / "copied")["state"] == "imported"
