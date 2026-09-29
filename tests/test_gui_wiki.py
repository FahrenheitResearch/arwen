"""The storm wiki: its page store, its citations, and runs linked to the events they cover.

The rule the wiki keeps is that nothing on a page is made up: every fact
names sources the store holds, prose only strings facts together, and a
run is tied to an event by what its own folder says (its box and its
hours), never by a hand-written link.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from gpuwm.gui import wiki
from gpuwm.gui.server import build_server, serve_in_thread

from test_gui_server import FakeRunner, HeldCardRunner, request

SEED = json.loads(wiki.SEED_FILE.read_text(encoding="utf-8"))
RECIPE_DOCS = {p.name: json.loads(p.read_text(encoding="utf-8")) for p in sorted(wiki.SEED_RECIPES.glob("*.json"))}
RECIPES = {r["event"]: r for doc in RECIPE_DOCS.values() for r in doc["recipes"]}


def _cites(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "cite" or key.endswith("_cite"):
                yield from ([item] if isinstance(item, str) else item)
            else:
                yield from _cites(item)
    elif isinstance(value, list):
        for item in value:
            yield from _cites(item)


def test_every_fact_of_the_seed_names_a_source_the_store_holds():
    sources = SEED["sources"]
    for group in ("events", "phenomena", "places"):
        for record in SEED[group]:
            named = list(_cites(record))
            assert named, record["id"]
            missing = [ident for ident in named if ident not in sources]
            assert not missing, (record["id"], missing)
    for event in SEED["events"]:
        for fact in event["facts"]:
            assert fact["cite"], (event["id"], fact["id"])
        assert event["rarity"]["cite"] and event["recipe"]["cite"], event["id"]
    # A computed statistic says how it was computed and from what; a row carries its own fields.
    for source in sources.values():
        if source.get("kind") == "computed":
            assert source["method"] and source["inputs"] and "result" in source, source["id"]
            assert all(i in sources for i in source["inputs"]), source["id"]
        if source.get("kind") == "record":
            assert source["row"] and source.get("dataset") in sources, source["id"]
        if source.get("kind") == "dataset":
            assert source["licence"] and source["url"], source["id"]


def test_seed_prose_only_strings_facts_together():
    for event in SEED["events"]:
        facts = {f["id"] for f in event["facts"]}
        parts = event["summary"]["parts"]
        assert event["summary"]["generated"]
        assert all(("fact" in p) != ("text" in p) for p in parts), event["id"]
        assert {p["fact"] for p in parts if "fact" in p} <= facts, event["id"]
        # The joining words hold no number: every number on a page is a fact with its footnote.
        assert not any(ch.isdigit() for p in parts if "text" in p for ch in p["text"]), event["id"]


def test_the_seed_lands_in_the_runs_root_and_a_later_document_replaces_its_records(tmp_path):
    assert wiki.ensure_seed(tmp_path) == tmp_path / "wiki" / "seed.json"
    first = SEED["events"][0]
    atlas = {"schema": wiki.SCHEMA, "origin": "atlas", "sources": {"atlas-row": {"kind": "record", "title": "t"}},
             "events": [{**first, "title": "Replaced", "seed": False},
                        {"id": "atlas-1", "type": "tornado", "title": "A new card", "facts": [], "places": [],
                         "when": {"start": "2026-01-01T00:00Z"}}]}
    (tmp_path / "wiki" / "atlas").mkdir()
    (tmp_path / "wiki" / "atlas" / "cards.json").write_text(json.dumps(atlas), encoding="utf-8")
    store = wiki.Store(tmp_path).data()
    assert store["events"][first["id"]]["title"] == "Replaced"
    assert store["events"][first["id"]]["origin"] == "atlas"
    assert "atlas-1" in store["events"] and len(store["events"]) == len(SEED["events"]) + 1
    # The seed is the package's file: a changed copy is put back, the atlas's document is left alone.
    (tmp_path / "wiki" / "seed.json").write_text("{}", encoding="utf-8")
    wiki.ensure_seed(tmp_path)
    assert json.loads((tmp_path / "wiki" / "seed.json").read_text(encoding="utf-8"))["schema"] == wiki.SCHEMA
    assert (tmp_path / "wiki" / "atlas" / "cards.json").is_file()


def _run_over(root: Path, name: str, event: dict, *, hours: int) -> Path:
    """A run folder whose box sits on the event's first point, starting at the event's start."""

    lon, lat, _ = wiki.event_points(event)[0]
    run = root / name
    run.mkdir(parents=True)
    ring = [[lon - 2, lat - 2], [lon + 2, lat - 2], [lon + 2, lat + 2], [lon - 2, lat + 2], [lon - 2, lat - 2]]
    (run / "region.geojson").write_text(json.dumps({"type": "Polygon", "coordinates": [ring]}), encoding="utf-8")
    start = event["when"]["start"][:13]
    (run / "plan.json").write_text(json.dumps({"config": {"intent": {"cycle": start, "hours": hours,
                                                                     "source": "gfs"}}}), encoding="utf-8")
    return run


def test_a_run_whose_box_and_hours_cover_an_event_is_listed_on_it_and_it_on_the_run(tmp_path):
    wiki.ensure_seed(tmp_path)
    tornado = next(e for e in SEED["events"] if e["type"] == "tornado")
    _run_over(tmp_path, "covers", tornado, hours=6)
    far = dict(tornado, when={"start": "1900-01-01T00:00Z", "end": "1900-01-01T01:00Z"})
    _run_over(tmp_path, "too-early", far, hours=6)
    pages = wiki.Wiki(tmp_path)
    runs = [row["id"] for row in pages.event_page(tornado["id"])["runs"]]
    assert runs == ["covers"]
    article = pages.run_page("covers", tmp_path / "covers")
    assert [row["id"] for row in article["events"]] == [tornado["id"]]
    # The run's facts cite its own files, which the page opens through the run-file endpoint.
    cited = {c for fact in article["facts"] for c in fact["cite"]}
    assert "run-file:plan.json" in cited and "run-file:region.geojson" in cited
    assert all(article["sources"][c]["kind"] in ("run-file", "run-tree") for c in cited)


def test_a_failed_run_article_says_why_it_stopped_and_what_to_do(tmp_path):
    """The article of a failed run said only "Failed" while its map page gave the reason (GS-09)."""

    wiki.ensure_seed(tmp_path)
    tornado = next(e for e in SEED["events"] if e["type"] == "tornado")
    run = _run_over(tmp_path, "failed", tornado, hours=6)
    (run / "events.jsonl").write_text(json.dumps({
        "event": "failed", "stage": "forecast", "error_class": "RuntimeError", "sequence": 1,
        "message": "per-domain wrfout writer failed: the history file changed after it was written.",
        "remedy": "Keep file sync out of this forecasts folder, then start the forecast again.",
        "interrupted": False, "exit_code": None}) + "\n", encoding="utf-8")
    facts = {fact["label"]: fact for fact in wiki.Wiki(tmp_path).run_page("failed", run)["facts"]}
    assert facts["State"]["text"] == "Failed"
    assert "history file changed" in facts["Why it stopped"]["text"]
    assert facts["What to do"]["text"].endswith("start the forecast again.")
    assert facts["Why it stopped"]["cite"] == ["run-file:events.jsonl"]
    # A run that finished carries neither line.
    done = _run_over(tmp_path, "done", tornado, hours=6)
    (done / "events.jsonl").write_text(json.dumps({"event": "completed", "sequence": 1}) + "\n",
                                       encoding="utf-8")
    labels = {fact["label"] for fact in wiki.Wiki(tmp_path).run_page("done", done)["facts"]}
    assert "Why it stopped" not in labels and "What to do" not in labels


@pytest.fixture()
def gui(tmp_path):
    server = build_server(tmp_path / "runs", port=0, runner=FakeRunner(), token="t" * 43)
    serve_in_thread(server)
    yield server
    server.shutdown()
    server.server_close()


class Era5Runner(FakeRunner):
    """An engine that offers ERA5 on the config-driven route and HRRR on the prepared one, on a 16 GB card."""

    def query(self, argv, *, cwd=None, timeout=0, log=None):
        if "--sources" in argv:
            self.queries.append(list(argv))
            plan = {"intent_supported": True, "requires_source_root": False}
            return {"sources": [
                {"source_id": "era5", "display_name": "ERA5", "max_forecast_hour": 0, "record_kind": "reanalysis",
                 "run_plan": {**plan, "intent_routes": ["experiment"]}},
                {"source_id": "hrrr", "display_name": "HRRR", "max_forecast_hour": 48, "coverage": {"conus": True},
                 "run_plan": {**plan, "intent_routes": ["prepared"]}}]}
        if "--physics-profiles" in argv:
            self.queries.append(list(argv))
            return {"sources": [], "profiles": []}
        if "--probe" in argv:
            self.queries.append(list(argv))
            return {"devices": [{"name": "A 16 GB card", "index": 0, "memory_total_bytes": 16 * 2**30}]}
        return super().query(argv, cwd=cwd, timeout=timeout, log=log)


@pytest.fixture()
def era5_gui(tmp_path):
    server = build_server(tmp_path / "runs", port=0, runner=Era5Runner(), token="t" * 43)
    serve_in_thread(server)
    yield server
    server.shutdown()
    server.server_close()


def test_the_wiki_pages_answer_with_their_sources(gui):
    response, main = request(gui, "GET", "/api/wiki")
    assert response.status == 200 and main["counts"]["events"] == len(SEED["events"])
    assert main["featured"]["event"]["id"] and main["featured"]["sources"]
    event = SEED["events"][0]
    response, page = request(gui, "GET", f"/api/wiki/event/{event['id']}")
    assert response.status == 200 and page["runs"] == []
    assert set(_cites(page["event"])) <= set(page["sources"])
    response, kind = request(gui, "GET", "/api/wiki/kind/tornado")
    assert response.status == 200 and all(row["type"] == "tornado" for row in kind["events"])
    # With best runs in the store, New forecast's recipe is the biggest card's row, nests and all.
    response, recipe = request(gui, "GET", f"/api/wiki/recipe/{event['id']}")
    best = RECIPES[event["id"]]
    biggest = max((row for row in best["cards"] if row["fits"]), key=lambda row: row["card_gb"])
    assert recipe["event"] == event["id"] and recipe["cycle"] == biggest["recipe"]["cycle"]
    assert recipe["card_gb"] == biggest["card_gb"] and recipe.get("chain") == biggest["recipe"].get("chain")
    _, small = request(gui, "GET", f"/api/wiki/recipe/{event['id']}?card=8")
    eight = next(r for r in best["cards"] if r["card_gb"] == 8)
    assert small["card"] == "8gb" and small["width_km"] == eight["recipe"]["width_km"]
    response, _ = request(gui, "GET", "/api/wiki/event/no-such-event")
    assert response.status == 404


def test_search_takes_plain_words_and_filters(gui):
    # A word of a place's name finds the place and every event there.
    place = next(p for p in SEED["places"] if p["kind"] == "country" and " " not in p["title"])
    there = {e["title"] for e in SEED["events"] if place["id"] in e["places"]}
    _, found = request(gui, "GET", f"/api/wiki/search?q={place['title'].lower()}")
    titles = {row["title"] for row in found["results"]}
    assert there and there <= titles and place["title"] in titles
    _, filtered = request(gui, "GET", "/api/wiki/search?type=tornado&sort=rarity")
    rows = filtered["results"]
    assert rows and all(row["type"] == "tornado" for row in rows)
    # Rarity ranks by share of the place's events, not by the raw count.
    assert [row["rarity_pct"] for row in rows] == sorted(row["rarity_pct"] for row in rows)
    state = next(p for p in SEED["places"] if p["kind"] == "state" and p.get("counts"))
    _, page = request(gui, "GET", f"/api/wiki/place/{state['id']}")
    assert page["events"] and page["place"]["counts"]["tornado"]["cite"]


def test_a_recipe_new_forecast_cannot_start_never_asks_for_the_event_cycle(gui):
    # An ERA5 recipe (a 2005 cyclone) must not reach New forecast as a GFS draft for its own 2005 cycle: the GFS
    # archive starts 2021-03-01 and prep fails. Only a recipe from an offered source keeps its cycle.
    old = next(e for e in SEED["events"] if RECIPES[e["id"]]["source"] == "era5")
    _, recipe = request(gui, "GET", f"/api/wiki/recipe/{old['id']}")
    assert recipe["runnable"] is False and recipe["start_cycle"] is None and recipe["start_source"] is None
    _, page = request(gui, "GET", f"/api/wiki/event/{old['id']}")
    assert page["runnable"] is False and not any(row["runnable"] for row in page["best"]["cards"])


def test_an_engine_that_offers_era5_runs_every_old_event_from_its_page(era5_gui):
    old = next(e for e in SEED["events"] if RECIPES[e["id"]]["source"] == "era5")
    _, recipe = request(era5_gui, "GET", f"/api/wiki/recipe/{old['id']}?card=16")
    row = next(r for r in RECIPES[old["id"]]["cards"] if r["card_gb"] == 16)
    assert recipe["runnable"] is True and recipe["start_cycle"] == row["recipe"]["cycle"]
    assert recipe["start_source"] == "era5"
    _, page = request(era5_gui, "GET", f"/api/wiki/event/{old['id']}")
    assert page["runnable"] is True and [r["card_gb"] for r in page["best"]["cards"]] == [8, 12, 16, 24, 32]
    # The page gets what it shows; the engine's own words for a row (intent, args) stay on the server.
    assert all("intent" not in r and "args" not in r for r in page["best"]["cards"])
    assert set(_cites(page["best"])) <= set(page["sources"])
    # The featured event is one whose best run starts from here, and the main page names no folder.
    _, main = request(era5_gui, "GET", "/api/wiki")
    assert main["featured"]["runnable"] is True and "folder" not in main
    _, system = request(era5_gui, "GET", "/api/system")
    assert system["card_gb"] == 16 and system["disk_free_gib"] > 0


def test_search_drops_question_words_and_ranks_by_words_matched(gui):
    place = next(p for p in SEED["places"] if p["kind"] == "country" and " " not in p["title"])
    there = {e["title"] for e in SEED["events"] if place["id"] in e["places"]}
    _, found = request(gui, "GET", f"/api/wiki/search?q=which+storms+hit+{place['title'].lower()}")
    assert found["words"] == [place["title"].lower()] and not found["partial"]
    assert there <= {row["title"] for row in found["results"]}
    # No page holds every word: the pages holding some of them still answer, marked partial.
    _, some = request(gui, "GET", f"/api/wiki/search?q={place['title'].lower()}+zzqx")
    assert some["partial"] and some["results"]


# ---------------------------------------------------------------- the best run for each card size


def test_every_recipe_keeps_the_rules_its_format_names():
    sources = dict(SEED["sources"])
    for doc in RECIPE_DOCS.values():
        sources.update(doc["sources"])
    assert set(RECIPES) == {e["id"] for e in SEED["events"]}
    for ident, recipe in RECIPES.items():
        missing = [c for c in _cites(recipe) if c not in sources]
        assert not missing, (ident, missing)
        rows = sorted(recipe["cards"], key=lambda r: r["card_gb"])
        assert [r["card_gb"] for r in rows] == [8, 12, 16, 24, 32], ident
        finest = [r["finest_km"] for r in rows if r["fits"]]
        # A bigger card never gets a coarser layout.
        assert finest == sorted(finest, reverse=True), ident
        for row in rows:
            if not row["fits"]:
                continue
            assert row["memory"]["need_gib"] <= row["memory"]["budget_gib"], (ident, row["card_gb"])
            # No cumulus scheme on a grid finer than 4 km: the storms there are resolved.
            assert all(d["cu_physics"] == 0 for d in row["domains"] if d["dx_km"] < 4), (ident, row["card_gb"])
            assert row["est_minutes"] and row["est_kind"] in ("measured", "estimated"), (ident, row["card_gb"])
            # The headline is the wait to the last picture: download, preparation, forecast and pictures.
            assert row["est_total_minutes"] >= row["est_minutes"], (ident, row["card_gb"])
            assert "not counting" not in row["est_basis"], (ident, row["card_gb"])
            assert row["disk_gib"] > 0 and row["door"] in ("domain", "cyclone-setup"), (ident, row["card_gb"])
            # A first press must not need more disk than the card's budget: a 1 km run filled a 58 GB disk.
            assert row["disk_gib"] <= row["disk_budget_gib"], (ident, row["card_gb"])
            # buffer_km is measured from the box, so an ERA5 row's 12 km ring is read back from its grids.
            outer = [r["km"] for r in row.get("rings_km") or [] if r["parent_id"] == 1]
            if row["source"] == "era5" and row["door"] == "domain" and outer and row["domains"][0]["dx_km"] >= 12:
                assert outer[0] >= 400, (ident, row["card_gb"], outer)
            # A cyclone's 1 km grid holds the centre 60 km inside its edge for the key hours it promises.
            if recipe["type"] == "tropical-cyclone" and row["finest_km"] == 1.0:
                key = recipe["key_hours"]
                hours = lambda a, b: (datetime.fromisoformat(b[:16]) - datetime.fromisoformat(a[:16])).total_seconds() / 3600
                before, after = row["key_cover_h"]
                assert before >= hours(key["start"], key["key_time"]), (ident, row["card_gb"])
                assert after >= hours(key["key_time"], key["end"]), (ident, row["card_gb"])
    text = json.dumps(RECIPE_DOCS, ensure_ascii=False)
    for bad in ("C:/", "C:\\", "/home/", "/Users/", "\u2014", "\u2013"):
        assert bad not in text, bad


def test_the_seed_brings_its_recipes_and_the_store_reads_them(tmp_path):
    wiki.ensure_seed(tmp_path)
    assert sorted(p.name for p in (tmp_path / "wiki" / "recipes").glob("*.json")) == sorted(RECIPE_DOCS)
    store = wiki.Store(tmp_path).data()
    assert set(store["recipes"]) == set(RECIPES)
    # A later document replaces an event's recipes, as it replaces an event record.
    first = next(iter(RECIPES))
    mine = {"schema": wiki.SCHEMA, "recipes": [{**RECIPES[first], "cards": RECIPES[first]["cards"][:1]}]}
    (tmp_path / "wiki" / "zz-mine.json").write_text(json.dumps(mine), encoding="utf-8")
    assert len(wiki.Store(tmp_path).data()["recipes"][first]["cards"]) == 1


def test_rarity_ranks_by_share_not_by_count():
    common = {"count": 1, "of": 3}
    rare = {"count": 8, "of": 4484}
    assert wiki.rarity_share(rare) < wiki.rarity_share(common)
    assert wiki.rarity_share({"count": 1}) is None and wiki.rarity_share(None) is None


def _event_with(source):
    return next(e for e in SEED["events"] if RECIPES[e["id"]]["source"] == source and e["type"] == "tornado")


def test_the_event_page_button_starts_the_best_run_for_the_card_and_the_run_joins_the_page(era5_gui, monkeypatch):
    from gpuwm.gui import api as api_module

    monkeypatch.setattr(api_module, "disk_free_gib", lambda path: 500.0)
    event = _event_with("era5")
    row = next(r for r in RECIPES[event["id"]]["cards"] if r["card_gb"] == 16)
    response, dry = request(era5_gui, "POST", "/api/wiki/simulate",
                            body={"event": event["id"], "card_gb": 16, "dry_run": True})
    assert response.status == 200 and dry["dry_run"] and dry["plan"]["route"] == "experiment"
    intent = dry["plan"]["config"]["intent"]
    assert intent["chain"] == row["intent"]["chain"] and intent["buffer_km"] == row["intent"]["buffer_km"]
    assert intent["cycle"] == row["intent"]["cycle"] and intent["card"] == "16gb"
    # The default run name reads as the event and its date.
    assert dry["run"].endswith(f"{row['start'][:10]}-16gb") and dry["run"].split("-")[0] in event["title"].lower()
    response, started = request(era5_gui, "POST", "/api/wiki/simulate", body={"event": event["id"], "card_gb": 16})
    assert response.status == 200, started
    rundir = era5_gui.api.root / started["run"]
    plan = json.loads((rundir / "plan.json").read_text(encoding="utf-8"))
    assert plan["config"]["intent"]["chain"] == row["intent"]["chain"]
    assert (rundir / "region.geojson").is_file()
    link = json.loads((rundir / wiki.RUN_LINK).read_text(encoding="utf-8"))
    assert link["event"] == event["id"] and link["card_gb"] == 16
    # A second press gets the next free name.
    _, again = request(era5_gui, "POST", "/api/wiki/simulate", body={"event": event["id"], "card_gb": 16})
    assert again["run"] == f"{started['run']}-2"
    # The run is listed on the event page under a readable title, and its article names the event.
    _, page = request(era5_gui, "GET", f"/api/wiki/event/{event['id']}")
    listed = {r["id"]: r for r in page["runs"]}
    assert started["run"] in listed and listed[started["run"]]["title"].startswith(event["title"])
    _, article = request(era5_gui, "GET", f"/api/wiki/run/{started['run']}")
    assert article["event"]["id"] == event["id"] and article["title"].startswith(event["title"])


def test_a_recipe_that_sets_vertical_levels_runs_and_customises_with_them(era5_gui, monkeypatch):
    # A per-event recipe may set the level count: the event page's button hands it to the engine, and Customise
    # opens New forecast with it chosen.
    from gpuwm.gui import api as api_module

    monkeypatch.setattr(api_module, "disk_free_gib", lambda path: 500.0)
    event = _event_with("era5")
    stored = era5_gui.api.wiki.store.data()["recipes"][event["id"]]
    row = next(r for r in stored["cards"] if r["card_gb"] == 16)
    monkeypatch.setitem(row, "intent", {**row["intent"], "nz": 80})
    monkeypatch.setitem(row, "recipe", {**row["recipe"], "nz": 80})
    response, dry = request(era5_gui, "POST", "/api/wiki/simulate",
                            body={"event": event["id"], "card_gb": 16, "dry_run": True})
    assert response.status == 200 and dry["plan"]["config"]["intent"]["nz"] == 80
    response, recipe = request(era5_gui, "GET", f"/api/wiki/recipe/{event['id']}?card=16")
    assert response.status == 200 and recipe["nz"] == 80


def untouched_customise(recipe: dict, name: str) -> dict:
    """What New forecast sends for a Customise nobody changed (create.js payload()): the recipe as it opened."""

    return {"name": name, "source": recipe["start_source"], "cycle": recipe["start_cycle"],
            "lat": recipe["lat"], "lon": recipe["lon"], "width_km": recipe["width_km"],
            "height_km": recipe["height_km"], "hours": recipe["hours"], "dx_km": recipe["dx_km"],
            "nz": recipe.get("nz"), "card": recipe["card"], "profile": recipe.get("profile"),
            "start_hour": recipe.get("start_hour") or 0, "chain": recipe.get("chain"),
            "buffer_km": recipe.get("buffer_km"), "era5_provider": recipe.get("era5_provider"),
            "event": recipe["event"], "recipe_card_gb": recipe["card_gb"]}


BEST = sorted((ident, row["card_gb"]) for ident, recipe in RECIPES.items()
              for row in recipe["cards"] if row["fits"] and row["door"] == "domain")


@pytest.mark.parametrize(("ident", "card"), BEST)
def test_an_untouched_customise_plans_what_the_events_button_plans(era5_gui, monkeypatch, ident, card):
    # Customise of a 12/3/1 km best run opened with its grids but started without the rest of it: the history
    # intervals that keep the run inside the disk the event page promised, and a cyclone's sea-surface flux, were
    # left out of the plan, so the run New forecast built was not the one the event page's button runs.
    from gpuwm.gui import api as api_module

    monkeypatch.setattr(api_module, "disk_free_gib", lambda path: 500.0)
    monkeypatch.setattr("gpuwm.fetch._head_ok", lambda url: True)
    monkeypatch.setattr("gpuwm.source_availability.quick_head", lambda url: True)
    # Every card size's layout, whatever card this test machine's stand-in engine reports.
    monkeypatch.setattr(era5_gui.api, "_system_or_none", lambda: None)
    response, recipe = request(era5_gui, "GET", f"/api/wiki/recipe/{ident}?card={card}")
    assert response.status == 200 and recipe["runnable"], recipe
    row = next(r for r in RECIPES[ident]["cards"] if r["card_gb"] == card)
    # New forecast is told every grid of the best run, not only its outer one.
    assert [d["dx_km"] for d in recipe["layout"]["domains"]] == [d["dx_km"] for d in row["domains"]]
    response, custom = request(era5_gui, "POST", "/api/create/start",
                               body={**untouched_customise(recipe, "same-run"), "dry_run": True})
    assert response.status == 200, custom
    response, button = request(era5_gui, "POST", "/api/wiki/simulate",
                               body={"event": ident, "card_gb": card, "name": "same-run", "dry_run": True})
    assert response.status == 200, button
    assert custom["plan"] == button["plan"]
    for key, value in row["intent"].items():
        assert custom["plan"]["config"]["intent"][key] == value, key


def test_a_customise_on_other_grids_carries_nothing_of_the_best_run(era5_gui, monkeypatch):
    from gpuwm.gui import api as api_module

    monkeypatch.setattr(api_module, "disk_free_gib", lambda path: 500.0)
    _, recipe = request(era5_gui, "GET", "/api/wiki/recipe/tornado-2011-1105221634-01?card=16")
    body = {**untouched_customise(recipe, "plain-grid"), "dx_km": 3, "chain": None, "buffer_km": None,
            "event": None, "recipe_card_gb": None, "dry_run": True}
    response, plain = request(era5_gui, "POST", "/api/create/start", body=body)
    assert response.status == 200, plain
    intent = plain["plan"]["config"]["intent"]
    assert "history_interval_s" not in intent and "chain" not in intent and intent["root_dx_km"] == 3
    # A layout the store does not hold is refused, rather than started as the event's without the rest of it.
    response, refused = request(era5_gui, "POST", "/api/create/start",
                                body={**untouched_customise(recipe, "no-layout"), "event": "no-such-event",
                                      "dry_run": True})
    assert response.status == 422 and "best run" in refused["message"], refused


def test_the_button_refuses_in_words_what_would_break_the_run(era5_gui, monkeypatch):
    from gpuwm.gui import api as api_module

    event = _event_with("era5")
    # A layout for a bigger card than this one runs out of card memory.
    monkeypatch.setattr(api_module, "disk_free_gib", lambda path: 5000.0)
    response, refused = request(era5_gui, "POST", "/api/wiki/simulate", body={"event": event["id"], "card_gb": 32})
    assert response.status == 422 and "32 GB card" in refused["message"] and "16.0 GiB" in refused["message"]
    # A run that would fill the disk stops partway.
    monkeypatch.setattr(api_module, "disk_free_gib", lambda path: 3.0)
    response, refused = request(era5_gui, "POST", "/api/wiki/simulate", body={"event": event["id"], "card_gb": 8})
    assert response.status == 507 and "3 GiB free" in refused["message"]
    made = [p for p in era5_gui.api.root.iterdir() if p.is_dir() and p.name != "wiki" and not p.name.startswith(".")]
    assert not made


class HeldEra5Runner(HeldCardRunner, Era5Runner):
    """The ERA5 engine above, on a card whose lock the real job manager keeps."""


def test_a_press_on_a_busy_card_is_refused_and_leaves_no_run(tmp_path, monkeypatch):
    # The press was refused with a 409, but the run folder it had written stayed: a Ready run nobody asked for in
    # My forecasts and on the event page, and the next press started a "-2" copy beside it.
    from gpuwm.gui import api as api_module

    monkeypatch.setattr(api_module, "disk_free_gib", lambda path: 500.0)
    runner = HeldEra5Runner(tmp_path / "jobs")
    server = build_server(tmp_path / "runs", port=0, runner=runner, token="t" * 43)
    serve_in_thread(server)
    try:
        event = _event_with("era5")
        press = {"event": event["id"], "card_gb": 16}
        _, dry = request(server, "POST", "/api/wiki/simulate", body={**press, "dry_run": True})
        runner.hold()
        response, refused = request(server, "POST", "/api/wiki/simulate", body=press)
        assert response.status == 409 and "job-held" in refused["message"], refused
        made = [p.name for p in server.api.root.iterdir()
                if p.is_dir() and p.name != "wiki" and not p.name.startswith(".")]
        assert made == []
        _, listed = request(server, "GET", "/api/runs")
        assert listed["runs"] == []
        _, page = request(server, "GET", f"/api/wiki/event/{event['id']}")
        assert page["runs"] == []
        # The card taken between the button's check and its start: the launch refuses, and the folder goes too.
        runner.card_holder = lambda: None
        response, refused = request(server, "POST", "/api/wiki/simulate", body=press)
        assert response.status == 409 and "job-held" in refused["message"], refused
        assert [p.name for p in server.api.root.iterdir()
                if p.is_dir() and p.name != "wiki" and not p.name.startswith(".")] == []
        del runner.card_holder
        # Once the card is free the same press starts the run under the event's own name.
        runner.free()
        response, started = request(server, "POST", "/api/wiki/simulate", body=press)
        assert response.status == 200 and started["run"] == dry["run"], started
    finally:
        server.shutdown()
        server.server_close()


def test_no_refusal_shows_a_machine_path():
    from gpuwm.gui.api import ApiError

    # The home-directory paths are assembled at run time so the release
    # snapshot's machine-path scan does not read this fixture as a real path.
    posix_run = "/home/" + "someone/runs/y"
    windows_run = "C:" + "\\" + "Users" + "\\someone\\runs\\x"
    reply = ApiError(409, f"Busy: {windows_run} is running and {posix_run} too.", "").reply()
    assert "Users" not in reply.body["message"] and "/home/" not in reply.body["message"]


def test_a_failed_run_shows_its_reason_in_words_without_paths(tmp_path):
    from gpuwm.gui import runs as runs_module

    run = tmp_path / "failed-run"
    run.mkdir()
    record = {"event": "failed", "stage": "forecast", "message": "The native Zarr reader is not installed. "
              "# build it once, from " + "/home/" + "someone/src: cd tools/zarr_bridge", "remedy": "see " + "C:/" + "Users/someone/x"}
    (run / "events.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    end = runs_module.status(run).get("end") or {}
    assert end["message"] == "The native Zarr reader is not installed."
    assert "Users" not in (end["remedy"] or "")
