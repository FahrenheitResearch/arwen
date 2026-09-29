"""New forecast starts from any past date: the per-source verdict, the route each source runs on, and the refusal.

The engine side runs for real (the source registry and its archive bounds); the provider's own bounds document is
never fetched here, a fake stands in for it.  The page side is the live server with the fake runner of
test_gui_server, answering with a reanalysis row beside the operational one.
"""

from __future__ import annotations

from datetime import datetime
import json

import pytest

from gpuwm.gui.api import plain_refusal, preferred_order
from gpuwm.gui.server import build_server, serve_in_thread
from gpuwm.source_availability import availability, verdict

from test_gui_server import FakeRunner, request

NOW = datetime(2026, 9, 25, 12)


def answer(source, cycle, hours=6, published=None):
    return verdict(availability(source, hours, now=NOW, published=published), cycle, now=NOW)


def test_a_date_before_the_operational_archive_names_where_it_starts():
    said = answer("gfs", "1999-05-03T18")
    assert said["state"] == "no" and "2021-03-22 12:00 UTC" in said["why"]


def test_the_reanalysis_holds_any_hour_from_1940():
    assert answer("era5", "1999-05-03T17")["state"] == "yes"
    assert answer("era5", "1940-01-01T00")["state"] == "yes"
    assert "1940-01-01" in answer("era5", "1939-12-31T18")["why"]


def test_a_time_still_to_come_and_an_hour_the_model_does_not_run_are_said_plainly():
    assert "future" in answer("gfs", "2026-09-26T00")["why"]
    said = answer("gfs", "2024-06-01T03")
    assert said["state"] == "no" and "00, 06, 12, 18" in said["why"]



def test_a_1999_start_is_no_for_every_downloaded_source_but_the_reanalyses():
    """Every runnable source whose bytes come from a declared transport answers a plain no for 1999, never "may have
    it": a maybe let the page accept a start that could only fail at the download."""

    from gpuwm.source_adapters import _ADAPTERS

    checked = []
    for adapter in _ADAPTERS:
        if not adapter.runnable:
            continue
        document = availability(adapter.source_id, 6, now=NOW)
        if not document["transports"]:
            continue
        said = verdict(document, "1999-05-03T18", now=NOW)
        if adapter.record_kind == "reanalysis":
            assert said["state"] == "yes", adapter.source_id
        else:
            assert said["state"] == "no" and said["why"], (adapter.source_id, said)
            checked.append(adapter.source_id)
    for source in ("gefs", "aigefs", "ecmwf-open-data", "aifs", "icon-eu", "icon-d2", "rap", "rrfs", "gem-gdps"):
        assert source in checked


def test_a_date_no_hour_of_which_the_archive_holds_names_the_archive_not_the_hour():
    for source in ("gfs", "hrrr", "gem-gdps", "rap"):
        said = answer(source, "1974-04-03T13")
        # the archive's own start is named (2021-03-22 12:00 UTC); the hour that was asked for is not
        assert said["state"] == "no" and "13:00" not in said["why"] and "not at 13" not in said["why"], (source, said)
    assert "2021-03-22" in answer("gfs", "1974-04-03T13")["why"]
    assert "last 29 days" in answer("gem-gdps", "1974-04-03T13")["why"]


def test_rap_holds_its_early_morning_runs_before_every_hour_begins():
    """Measured from noaa-rap-pds on 2026-09-25: rap.20210222 holds the 00 to 09 UTC awip32 runs, 2021-02-21 holds
    none, and 10 to 23 UTC begin on 2021-04-26.  A single start of 2021-04-26 refused starts the archive holds."""

    assert answer("rap", "2021-04-01T00")["state"] == "yes"
    assert answer("rap", "2021-02-22T09")["state"] == "yes"
    late = answer("rap", "2021-04-01T18")
    assert late["state"] == "no" and "2021-04-26 00:00 UTC" in late["why"] and "00 to 09" in late["why"]
    assert answer("rap", "2021-04-26T18")["state"] == "yes"
    early = answer("rap", "2021-02-21T00")
    assert early["state"] == "no" and "2021-02-22 00:00 UTC" in early["why"]


def test_a_reason_for_a_list_row_leaves_the_source_name_out():
    said = answer("gfs", "1999-05-03T18")
    assert said["reason"] == "Archive starts 2021-03-22 12:00 UTC."
    assert "GFS" in said["why"]
    assert answer("gem-gdps", "1974-04-03T13")["reason"].startswith("Keeps only its last")


def test_what_a_sources_bytes_are_is_a_registry_column_not_its_horizon():
    from gpuwm.source_adapters import get_source_adapter

    assert get_source_adapter("gdas").record_kind == "analysis"
    assert get_source_adapter("era5").record_kind == "reanalysis"
    assert get_source_adapter("gfs").record_kind == "forecast"

def test_the_providers_own_last_day_bounds_the_newest_start():
    window_end = datetime(2026, 9, 10, 23)
    document = availability("era5", 6, now=NOW, published=lambda window: window_end if window.bounds_url else None)
    assert document["latest_candidate"] == "2026-09-10T12"
    assert verdict(document, "2026-09-15T00", now=NOW)["state"] == "no"
    assert any(row["record_end"] == "2026-09-10T23" for row in document["transports"])


def test_an_intent_for_the_reanalysis_reads_the_keyless_store_unless_it_names_a_provider():
    from gpuwm.runplan import _keyless_era5_default

    assert _keyless_era5_default({"source": "era5"}, "era5")["era5_provider"] == "arco"
    assert _keyless_era5_default({"era5_provider": "cds"}, "era5")["era5_provider"] == "cds"
    assert "era5_provider" not in _keyless_era5_default({}, "gfs")
    members = {"era5_product": "ensemble_members"}
    assert "era5_provider" not in _keyless_era5_default(members, "era5")


def test_sources_are_suggested_from_registry_facts_not_names():
    rows = [{"id": "a", "certified": False, "regional": False, "analysis": False},
            {"id": "b", "certified": True, "regional": True, "analysis": False},
            {"id": "c", "certified": True, "regional": False, "analysis": True},
            {"id": "d", "certified": True, "regional": False, "analysis": False}]
    assert preferred_order(rows) == ["d", "c", "b", "a"]


def test_a_fit_refusal_reads_as_words_without_the_engines_paths():
    text = ("gpuwm run-plan: run plan 'config.intent' does not fit: polygon falls outside HRRR coverage: "
            "window i=2813..3107 (run gpuwm run-plan C:\\runs\\x\\plan.json --resolve --explain)")
    message, fix = plain_refusal(text, {"card": "16gb"}, None, "HRRR")
    assert message == "The box reaches outside the area HRRR covers."
    assert "plan.json" not in message + fix
    message, _ = plain_refusal("", {"card": "8gb"}, {"need_gib": 9.5, "budget_gib": 6.75, "fits": False}, "GFS")
    assert message.startswith("Too big for the 8 GB card")
    assert "about 9.5 GiB and 6.8 GiB is usable" in message


def test_a_data_table_version_mismatch_reads_as_one_line_with_the_fix():
    # gpuwm.data_assets' own refusal, as the engine prints it when gpuwm-data is another release.
    text = ("gpuwm run-plan: ImportError: gpuwm REFUSES to read packaged reference data from a mismatched "
            "companion: gpuwm 2.8.0 requires gpuwm-data 2.8.0, found 2.7.6.  The two are cut from one "
            "commit and pinned `==`, so this install was edited.  The tables are versioned data, not "
            "interchangeable files.  Fix it:\n    pip install gpuwm-data==2.8.0\n")
    message, fix = plain_refusal(text, {"card": "16gb"}, None, "GFS")
    assert message == ("The program is gpuwm 2.8.0 but its data tables are gpuwm-data 2.7.6; "
                       "the two must be the same version (2.8.0).")
    assert fix == "Run pip install gpuwm-data==2.8.0 in a terminal, then check again."
    assert "\n" not in message


def test_a_mismatched_data_package_says_the_line_that_fixes_it():
    # A worktree of one version beside an installed gpuwm-data of another: every fit on the page failed with
    # "the full reason is in the page server's log", and the reason's last line is the whole fix.
    text = ("gpuwm REFUSES to read packaged reference data from a mismatched companion: gpuwm 2.8.0 requires "
            "gpuwm-data 2.8.0, found 2.6.0.  The two are cut from one commit and pinned `==`, so this install was "
            "edited.  Fix it:\n    pip install gpuwm-data==2.8.0")
    message, fix = plain_refusal(text, {"card": "8gb"}, None, "GFS")
    assert "gpuwm-data 2.6.0" in message and "log" not in message
    assert "pip install gpuwm-data==2.8.0" in fix


def test_an_engine_older_than_the_level_choice_says_so():
    text = ("gpuwm run-plan: run plan 'config.intent' does not have the key(s) ['nz']; no key is ignored, because "
            "a dropped key runs a default under the name of your value.")
    message, fix = plain_refusal(text, {"card": "8gb"}, None, "GFS")
    assert "older than the level choice" in message and "Engine default" in fix


class TwoSources(FakeRunner):
    def query(self, argv, **kwargs):
        if "--sources" in argv:
            self.queries.append(list(argv))
            return {"sources": [
                {"source_id": "gfs", "display_name": "GFS", "max_forecast_hour": 384,
                 "maturity": {"status": "certified_stock_wrf"}, "coverage": None,
                 "run_plan": {"intent_supported": True, "intent_routes": ["prepared"], "requires_source_root": False}},
                {"source_id": "era5", "display_name": "ERA5", "max_forecast_hour": 0,
                 "maturity": {"status": "certified_stock_wrf"}, "coverage": None,
                 "run_plan": {"intent_supported": True, "intent_routes": ["experiment"], "requires_source_root": False}}]}
        return super().query(argv, **kwargs)


@pytest.fixture()
def gui(tmp_path, monkeypatch):
    # The provider's bounds document is not fetched in a test.
    monkeypatch.setattr("gpuwm.source_availability.published_stop", lambda window, **_: None)
    # A recent start is put to the fetch's object probe; no server is asked in a test.
    monkeypatch.setattr("gpuwm.fetch._head_ok", lambda url: True)
    # A page asks through its own short probe; no server is asked in a test either.
    monkeypatch.setattr("gpuwm.source_availability.quick_head", lambda url: True)
    runner = TwoSources()
    server = build_server(tmp_path / "runs", port=0, runner=runner, token="t" * 43)
    serve_in_thread(server)
    yield server, runner
    server.shutdown()
    server.server_close()


DRAFT = {"name": "old", "source": "era5", "cycle": "1999-05-03T18", "lat": 35.5, "lon": -97.0,
         "width_km": 600, "height_km": 600, "hours": 6, "card": "8gb"}


def test_the_availability_endpoint_answers_per_source_and_suggests_one_that_has_it(gui):
    server, _ = gui
    response, body = request(server, "GET", "/api/sources/availability?time=1999-05-03T18&hours=6")
    assert response.status == 200
    rows = {row["id"]: row for row in body["sources"]}
    assert rows["era5"]["state"] == "yes" and rows["gfs"]["state"] == "no"
    assert "2021-03-22" in rows["gfs"]["why"]
    # The row shows the name; its reason does not repeat it.
    assert rows["gfs"]["name"] not in rows["gfs"]["why"]
    assert body["best"] == "era5"
    response, _ = request(server, "GET", "/api/sources/availability?time=yesterday")
    assert response.status == 400


def test_a_reanalysis_start_is_written_on_the_route_the_engine_names_for_it(gui):
    server, runner = gui
    response, body = request(server, "POST", "/api/create/start", body=DRAFT)
    assert response.status == 200, body
    plan = json.loads((server.root / "old" / "plan.json").read_text(encoding="utf-8"))
    assert plan["route"] == "experiment"
    assert plan["config"]["intent"]["cycle"] == "1999-05-03T18"


def test_a_start_the_source_does_not_hold_is_refused_before_anything_runs(gui):
    server, runner = gui
    response, body = request(server, "POST", "/api/create/start", body={**DRAFT, "source": "gfs"})
    assert response.status == 422
    assert "2021-03-22" in body["message"]
    assert runner.launched == [] and not (server.root / "old").exists()


def test_an_intent_plan_fetches_what_its_generated_configuration_declares(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from types import SimpleNamespace

    from gpuwm.runplan import _config_for_declared_fetch

    generated = ('[fetch]\nsource = "era5"\ncycle = "1999-05-03T18"\nhours = 6\nout = "data/box"\n'
                 'era5_provider = "arco"\n')
    plan = SimpleNamespace(config_intent={"source": "era5"})
    document = _config_for_declared_fetch(plan, {"generated_config": generated}, tmp_path / "run")
    assert document["fetch"]["out"] == str(tmp_path / "data/box")
    assert document["fetch"]["era5_provider"] == "arco"


def test_an_archived_operational_cycle_is_read_from_the_archive_not_the_rolling_crop_host():
    from gpuwm.fetch import archive_only_cycle

    assert archive_only_cycle("gfs", datetime(2024, 5, 6, 12), now=NOW)
    assert not archive_only_cycle("gfs", datetime(2026, 9, 24, 12), now=NOW)


@pytest.mark.parametrize("intent, asked, drawn", [
    ({"source": "era5"}, None, "all"),
    ({"source": "era5"}, "none", None),
    (None, None, None),
])
def test_an_intent_on_the_config_driven_route_draws_its_pictures_unless_told_not_to(
        monkeypatch, tmp_path, intent, asked, drawn):
    from types import SimpleNamespace

    from gpuwm import runplan, runtime

    finished = []
    monkeypatch.setattr(runtime, "run_experiment", lambda *a, **k: SimpleNamespace(
        wrfout_paths=(), completed_seconds=0.0, nan_free=True))
    monkeypatch.setattr(runplan, "_finish_render", lambda render_plan, observer: finished.append(render_plan))
    observer = SimpleNamespace(arm_first_products=lambda render_plan: None)
    plan = SimpleNamespace(run_dir=tmp_path, config_intent=intent,
                           run_options={} if asked is None else {"render_products": asked})
    runplan._execute_experiment_route(plan, exp=None, data=None, config_path=None, observer=observer)
    assert [row["render_products"] for row in finished] == ([] if drawn is None else [drawn])


# LIVE is four minutes after the 06Z GFS start: the schedule's estimate names that start, but none of its objects is
# on a host yet.
LIVE = datetime(2026, 9, 26, 6, 4)


def test_a_new_draft_opens_at_once_and_then_on_the_newest_start_the_download_accepts(gui, monkeypatch):
    from gpuwm.source_availability import confirmed_latest

    server, _ = gui
    asked = []

    def nothing_from_06z(url):
        asked.append(url)
        return ".t06z." not in url

    monkeypatch.setattr("gpuwm.gui.api._utcnow", lambda: LIVE)
    monkeypatch.setattr("gpuwm.source_availability.quick_head", nothing_from_06z)
    document = availability("gfs", 6, now=LIVE)
    # The schedule's estimate is the start the page used to open on; the download refuses it.
    assert document["latest_candidate"] == "2026-09-26T06"
    expected = confirmed_latest(document, now=LIVE, probe=nothing_from_06z)
    assert expected == "2026-09-26T00"
    asked.clear()

    # The first answer is at once, with the probe running behind it: never the schedule's estimate (the 06Z cycle
    # still being made), but the newest start past GFS's usual publication delay, named as no newest run yet.
    response, body = request(server, "GET", "/api/sources")
    assert response.status == 200, body
    assert body["preferred"][0] == "gfs"
    assert body["default_cycle"] == "2026-09-26T00" and body["default_newest_run"] is None
    assert body["default_checking"] is True
    assert server.api.board.wait(10)
    assert asked, "the newest start was never put to the fetch's probe"

    # Once the probe is in, the draft opens on the start the download accepts, and it is the newest run.
    response, body = request(server, "GET", "/api/sources")
    assert body["default_cycle"] == expected and body["default_newest_run"] == expected
    # The page's own check calls the opening start available, for the same hours, once its row is checked.
    response, check = request(server, "GET", f"/api/sources/availability?time={expected}&hours=6")
    assert server.api.board.wait(10)
    response, check = request(server, "GET", f"/api/sources/availability?time={expected}&hours=6")
    row = {row["id"]: row for row in check["sources"]}["gfs"]
    assert row["state"] == "yes" and row["basis"] == "checked" and not row["checking"]
    assert check["pending"] == 0

    # Reopening the page inside the cache window asks no server again.
    monkeypatch.setattr("gpuwm.source_availability.quick_head", lambda url: pytest.fail(f"reopening probed {url}"))
    response, again = request(server, "GET", "/api/sources")
    assert again["default_cycle"] == expected
    assert server.api.board.wait(5)


def test_the_source_names_the_wiki_reads_ask_no_server(gui, monkeypatch):
    server, _ = gui
    monkeypatch.setattr("gpuwm.gui.api._utcnow", lambda: LIVE)
    monkeypatch.setattr("gpuwm.fetch._head_ok", lambda url: pytest.fail(f"the source names probed {url}"))
    monkeypatch.setattr("gpuwm.source_availability.quick_head",
                        lambda url: pytest.fail(f"the source names probed {url}"))
    assert server.api.offered_sources() == {"gfs": "GFS", "era5": "ERA5"}
