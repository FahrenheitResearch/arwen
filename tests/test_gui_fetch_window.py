"""New forecast's source check asks about the download the run makes, and one source's answer is only its row.

Picking the offered 3 h length turned the whole source list into a 500.  For AIFS, AIGFS and AIGEFS, whose files
come every 6 hours, the row's check asked the fetch's lead resolver about a 3-hour window, which it refuses, inside
the request, and one row's exception failed every row, GFS included; the fit for the same draft answered 500 too.
No run asks for that window: `gpuwm domain` writes the download rounded up to whole 6-hour steps, hours 0 and 6
(gpuwm.domain_wizard.fetch_window).  The check now asks about that same download.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpuwm.domain_wizard import fetch_window
from gpuwm.source_availability import availability, publication_refusal, verdict

from test_gui_server import FakeRunner, request

SIX_HOURLY = ("aifs", "aigfs", "aigefs")


def _row(source_id: str) -> dict:
    return {"source_id": source_id, "display_name": source_id.upper(), "max_forecast_hour": 360,
            "maturity": {"status": "certified_stock_wrf"}, "coverage": None,
            "run_plan": {"intent_supported": True, "intent_routes": ["prepared"], "requires_source_root": False}}


class FourSources(FakeRunner):
    def query(self, argv, **kwargs):
        if "--sources" in argv:
            self.queries.append(list(argv))
            return {"sources": [_row(source) for source in ("gfs", *SIX_HOURLY)]}
        if "--physics-profiles" in argv:
            self.queries.append(list(argv))
            return {"sources": [{"source_id": source, "default_profile_id": "p1",
                                 "profiles": [{"profile_id": "p1", "admissible": True, "is_default": True}]}
                                for source in ("gfs", *SIX_HOURLY)],
                    "profiles": [{"profile_id": "p1", "summary": "p1: words"}]}
        return super().query(argv, **kwargs)


def recent_start() -> str:
    """A six-hourly start 18 to 24 hours old: inside the window where a start is put to the fetch's object probe."""

    moment = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=18)
    return moment.replace(hour=moment.hour - moment.hour % 6, minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H")


@pytest.fixture()
def gui(tmp_path, monkeypatch):
    from gpuwm.gui.server import build_server, serve_in_thread

    asked: list[str] = []

    def every_object_present(url):
        asked.append(url)
        return True

    # No provider bounds document and no object is fetched in a test: every object answers present.
    monkeypatch.setattr("gpuwm.source_availability.published_stop", lambda window, **_: None)
    monkeypatch.setattr("gpuwm.fetch._head_ok", every_object_present)
    monkeypatch.setattr("gpuwm.source_availability.quick_head", every_object_present, raising=False)
    runner = FourSources()
    server = build_server(tmp_path / "runs", port=0, runner=runner, token="t" * 43)
    serve_in_thread(server)
    yield server, runner, asked
    server.shutdown()
    server.server_close()


def _settled_rows(server, cycle: str, hours: int) -> dict:
    response, body = request(server, "GET", f"/api/sources/availability?time={cycle}&hours={hours}", timeout=30)
    assert response.status == 200, body
    board = getattr(server.api, "board", None)
    if board is not None and body.get("pending"):
        assert board.wait(30)
        response, body = request(server, "GET", f"/api/sources/availability?time={cycle}&hours={hours}", timeout=30)
        assert response.status == 200, body
    return {row["id"]: row for row in body["sources"]}


def test_a_3_hour_forecast_is_checked_as_the_download_the_run_makes(gui):
    server, _, asked = gui
    rows = _settled_rows(server, recent_start(), 3)
    assert all(row["state"] == "yes" and row["starts"] == "now" for row in rows.values()), rows
    # The 6-hourly sources were asked about hour 6, the last file their 3-hour run downloads, never hour 3; GFS,
    # whose download is 3-hourly, about hour 3.
    six_hourly = [url for url in asked if "aifs-single" in url or "/aigfs." in url or "/aigefs." in url]
    assert six_hourly and all(".f006." in url or "-6h-" in url for url in six_hourly), six_hourly
    assert any("gfs.t" in url and url.endswith(".f003") for url in asked), asked


def test_the_fit_for_a_3_hour_forecast_from_6_hourly_files_is_sized_not_refused(gui):
    server, runner, _ = gui
    draft = {"source": "aifs", "cycle": recent_start(), "hours": 3, "lat": 35.5, "lon": -97.0,
             "width_km": 600, "height_km": 600, "card": "16gb"}
    for source in ("aifs", "aigfs", "aigefs", "gfs"):
        response, body = request(server, "POST", "/api/create/fit", body={**draft, "source": source}, timeout=30)
        assert response.status == 200, (source, body)
    assert sum("--resolve" in argv for argv in runner.queries) == 4


@pytest.mark.parametrize("source", ["aifs", "aigfs", "aigefs", "gfs", "era5", "gefs", "icon-eu"])
@pytest.mark.parametrize("hours", [1, 2, 3, 5, 6, 9, 12])
def test_the_download_the_guidance_asks_about_is_the_one_gpuwm_domain_writes(source, hours):
    document = availability(source, hours, now=datetime(2026, 9, 26, 20, 28))
    assert (document["cadence_hours"], document["last_hour"]) == fetch_window(source, hours)
    assert document["hours"] == hours


def test_a_3_hour_start_from_6_hourly_files_is_put_to_the_probe_without_an_error():
    now = datetime(2026, 9, 26, 20, 28)
    asked = []

    def present(url):
        asked.append(url)
        return True

    for source in SIX_HOURLY:
        document = availability(source, 3, now=now)
        assert document["last_hour"] == 6 and document["cycle_hours"] == [0, 6, 12, 18]
        # integrate/2.8 raised "--hours must be an exact multiple of the 6 h cadence" here, inside the page request.
        assert publication_refusal(document, "2026-09-26T06", now=now, probe=present) is None
        assert verdict(document, "2026-09-26T06", now=now, confirm=lambda d, c, now: publication_refusal(
            d, c, now=now, probe=present))["state"] == "yes"
    assert asked and not any("f003" in url or "-3h-" in url for url in asked), asked


def test_one_sources_failed_check_is_its_own_row_and_the_list_still_answers(gui, monkeypatch):
    server, _, _ = gui
    board = server.api.board
    real = board.row

    def broken_for_aifs(source, *args, **kwargs):
        if source == "aifs":
            raise RuntimeError("a check that failed")
        return real(source, *args, **kwargs)

    monkeypatch.setattr(board, "row", broken_for_aifs)
    rows = _settled_rows(server, recent_start(), 6)
    assert rows["aifs"]["state"] == "unknown" and rows["aifs"]["starts"] == "no" and not rows["aifs"]["checking"]
    assert rows["aifs"]["why"] == "This source could not be checked. The full reason is in the page server's log."
    assert rows["gfs"]["state"] == "yes" and rows["aigfs"]["state"] == "yes"


def test_a_fit_whose_source_check_fails_is_refused_in_words_and_never_a_server_error(gui, monkeypatch):
    # The fit reads the same row as the list: the check that raised inside the list's row raised inside the fit too.
    server, runner, _ = gui
    board = server.api.board
    real = board.row

    def broken_for_aifs(source, *args, **kwargs):
        if source == "aifs":
            raise RuntimeError("a check that failed")
        return real(source, *args, **kwargs)

    monkeypatch.setattr(board, "row", broken_for_aifs)
    draft = {"source": "aifs", "cycle": recent_start(), "hours": 3, "lat": 35.5, "lon": -97.0,
             "width_km": 600, "height_km": 600, "card": "16gb"}
    response, body = request(server, "POST", "/api/create/fit", body=draft, timeout=30)
    assert response.status == 422, body
    assert body["message"] == "This source could not be checked. The full reason is in the page server's log."
    assert not any("--resolve" in argv for argv in runner.queries)
    response, body = request(server, "POST", "/api/create/fit", body={**draft, "source": "gfs"}, timeout=30)
    assert response.status == 200, body


@pytest.mark.parametrize("source", sorted(__import__("gpuwm.fetch_routes", fromlist=["route_ids"]).route_ids()))
def test_every_start_hour_the_guidance_offers_is_one_the_fetchs_lead_resolver_takes(source):
    """The calendar and the download read one table: no length and start hour is offered that the fetch refuses."""

    from gpuwm import fetch_routes

    now = datetime(2026, 9, 26, 20, 28)
    route = fetch_routes.table_route(source)
    for hours in (1, 2, 3, 6, 9, 12, 24, 36, 48, 72, 120, 168, 240, 384):
        document = availability(source, hours, now=now)
        assert document["last_hour"] % document["cadence_hours"] == 0
        for hour in document["cycle_hours"]:
            fetch_routes.resolve_leads(route, now.replace(hour=hour), document["last_hour"],
                                       cadence=document["cadence_hours"])


def test_a_source_outside_the_route_table_downloads_on_its_own_spacing():
    # GFS has its own transport; `gpuwm domain` still asks it for 3-hourly files, so a 1-hour run downloads hour 3.
    now = datetime(2026, 9, 26, 20, 28)
    assert (availability("gfs", 3, now=now)["last_hour"], availability("gfs", 1, now=now)["last_hour"]) == (3, 3)
    assert availability("gfs", 3, now=now)["spacing_limit"] is None


def test_a_named_fetch_spacing_is_the_one_judged():
    # ECMWF IFS 00Z and 12Z files are 3-hourly through hour 144 and 6-hourly after (06Z and 18Z stop at 144): a
    # request that names 6-hourly leads takes the whole week from those two, the default 3-hourly request from none.
    now = datetime(2026, 9, 26, 20, 28)
    assert availability("ecmwf-open-data", 168, now=now)["cycle_hours"] == []
    assert availability("ecmwf-open-data", 168, now=now, cadence=6)["cycle_hours"] == [0, 12]


def test_a_run_whose_files_thin_out_before_the_end_says_how_far_they_go():
    now = datetime(2026, 9, 26, 20, 28)
    # ICON-EU's 03Z run is hourly only through hour 30; its 00Z run is hourly through hour 78.
    said = verdict(availability("icon-eu", 48, now=now), "2026-09-26T03", now=now)
    assert said["state"] == "no" and said["reason"] == "Starts at 00, 06, 12, 18 UTC for a forecast this long, not at 03."
    assert verdict(availability("icon-eu", 48, now=now), "2026-09-26T00", now=now)["state"] == "yes"
    # ECMWF IFS files come every 3 hours through hour 144 and every 6 after: the download asks for every 3 hours.
    said = verdict(availability("ecmwf-open-data", 168, now=now), "2026-09-25T00", now=now)
    assert said["state"] == "no"
    assert said["reason"] == ("Its files come every 3 hours only through hour 144, so a 168-hour forecast cannot be "
                              "downloaded.")
    assert said["fix"] == "Pick 144 hours or less, or another source."
